"""
Uncertainty quantification for LyTimeT rollouts: particle-ensemble
propagation + split conformal calibration.

Two distinct ideas, composed:

  1. **Particle rollout** (`particle_rollout`): perturb the initial latent
     state z_0 with small Gaussian noise (representing the encoder's own
     uncertainty about "where exactly is the system right now"), form a
     batch of `num_particles` perturbed initial conditions, and `vmap` the
     *same* trained transition (residual-MLP or Neural ODE -- both share
     the `.rollout(z0, k_steps)` interface) across all of them. This gives
     an empirical ensemble of trajectories whose spread at each horizon
     step is a natural (if approximate) local uncertainty scale. Note this
     specifically captures *sensitivity to initial-condition uncertainty
     propagated through the learned dynamics* -- it does not capture
     model/parameter (epistemic) uncertainty, which would require a deep
     ensemble of independently-trained transitions instead/in addition.

  2. **Split conformal prediction** (`calibrate_conformal`,
     `predict_with_interval`): the raw ensemble std is *not* a calibrated
     error bar -- it's whatever the ensemble happens to produce, with no
     guarantee it matches true error rates. Split conformal fixes this: on
     a held-out calibration set, compute a "nonconformity score" per
     (horizon step, latent dimension) -- here, the ensemble-mean error
     normalized by the ensemble std, a standard *locally-scaled* conformal
     score -- take the finite-sample-corrected empirical quantile of that
     score, and use it to scale the ensemble std at prediction time. The
     result: intervals [mean +/- q_hat * std] with a distribution-free
     marginal coverage guarantee (assuming calibration and test clips are
     exchangeable), even though the underlying ensemble is just a rough,
     uncalibrated Monte Carlo spread.

Calibration (and coverage evaluation) is done *per horizon step k*, i.e.
each step gets its own q_hat -- this gives step-wise marginal coverage,
not a single guarantee over the whole path jointly. That's the standard,
simplest-to-reason-about scope for conformalized multi-step forecasts.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jax_lytimet.model import LyTimeT


# --------------------------------------------------------------------------
# 1. Particle rollout
# --------------------------------------------------------------------------
def particle_rollout(
    model: LyTimeT,
    z0: Float[Array, "Dz"],
    k_steps: int,
    num_particles: int,
    noise_std: float,
    key: PRNGKeyArray,
) -> Float[Array, "P K Dz"]:
    """Perturb z0 into `num_particles` initial conditions and vmap the
    (single, deterministic) trained transition's rollout across all of
    them. Works unmodified for both `dynamics_type`s since both
    `LatentTransition` and `NeuralODETransition` expose
    `.rollout(z0, k_steps) -> (K, Dz)`.
    """
    dz = z0.shape[-1]
    noise = noise_std * jax.random.normal(key, (num_particles, dz))
    particles0 = z0[None, :] + noise  # (P, Dz)
    rollouts = jax.vmap(lambda z: model.transition.rollout(z, k_steps))(particles0)
    return rollouts  # (P, K, Dz)


def ensemble_mean_std(
    particles: Float[Array, "P K Dz"], eps: float = 1e-6
) -> tuple[Float[Array, "K Dz"], Float[Array, "K Dz"]]:
    """Per-(step, dim) ensemble mean and std across the particle axis."""
    mean = jnp.mean(particles, axis=0)
    std = jnp.std(particles, axis=0) + eps
    return mean, std


# --------------------------------------------------------------------------
# 2. Split conformal calibration on top of the ensemble scale
# --------------------------------------------------------------------------
def _nonconformity_scores_for_clip(
    model: LyTimeT,
    clip: Float[Array, "T C H W"],
    k_steps: int,
    num_particles: int,
    noise_std: float,
    key: PRNGKeyArray,
) -> Float[Array, "K Dz"]:
    """For one calibration clip: encode to get the true z_1..z_T, particle-
    roll out from the encoder's own z_0, and return the normalized
    nonconformity score |z_true_k - mean_k| / std_k for each (step, dim)."""
    z_true, _ = model.encode(clip)  # (T, Dz)
    t = clip.shape[0]
    k = min(k_steps, t - 1)
    particles = particle_rollout(model, z_true[0], k, num_particles, noise_std, key)
    mean, std = ensemble_mean_std(particles)  # (k, Dz) each
    target = z_true[1 : 1 + k]  # (k, Dz)
    score = jnp.abs(target - mean) / std
    if k < k_steps:
        pad = jnp.full((k_steps - k, target.shape[-1]), jnp.nan)
        score = jnp.concatenate([score, pad], axis=0)
    return score  # (k_steps, Dz)


def calibrate_conformal(
    model: LyTimeT,
    calib_clips: Float[Array, "N T C H W"],
    k_steps: int,
    num_particles: int,
    noise_std: float,
    alpha: float,
    key: PRNGKeyArray,
) -> Float[Array, "K Dz"]:
    """Split conformal calibration: compute the normalized nonconformity
    score on every calibration clip, then take the finite-sample-corrected
    empirical (1-alpha) quantile *per (horizon step, latent dim)*.

    Returns q_hat of shape (k_steps, Dz); at prediction time the calibrated
    interval at step k, dim d is `mean[k,d] +/- q_hat[k,d] * std[k,d]`.
    """
    n = calib_clips.shape[0]
    keys = jax.random.split(key, n)
    scores = jax.vmap(
        lambda clip, k_: _nonconformity_scores_for_clip(
            model, clip, k_steps, num_particles, noise_std, k_
        )
    )(calib_clips, keys)  # (N, k_steps, Dz)

    # Finite-sample split-conformal correction: use quantile level
    # ceil((n+1)(1-alpha)) / n, clipped to 1.0, with "higher" interpolation.
    level = jnp.minimum(1.0, jnp.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = jnp.quantile(scores, level, axis=0, method="higher")  # (k_steps, Dz)
    return q_hat


def predict_with_interval(
    model: LyTimeT,
    z0: Float[Array, "Dz"],
    k_steps: int,
    num_particles: int,
    noise_std: float,
    q_hat: Float[Array, "K Dz"],
    key: PRNGKeyArray,
):
    """Roll out an ensemble from z0 and return (mean, lower, upper), each of
    shape (k_steps, Dz), using the calibrated per-(step, dim) multiplier
    q_hat from `calibrate_conformal`. These intervals carry the split-
    conformal marginal coverage guarantee (per horizon step) as long as the
    deployment clip is exchangeable with the calibration set.
    """
    particles = particle_rollout(model, z0, k_steps, num_particles, noise_std, key)
    mean, std = ensemble_mean_std(particles)
    radius = q_hat * std
    return mean, mean - radius, mean + radius


# --------------------------------------------------------------------------
# 3. Empirical coverage check (sanity check that the guarantee holds)
# --------------------------------------------------------------------------
def evaluate_coverage(
    model: LyTimeT,
    test_clips: Float[Array, "N T C H W"],
    k_steps: int,
    num_particles: int,
    noise_std: float,
    q_hat: Float[Array, "K Dz"],
    key: PRNGKeyArray,
) -> Float[Array, "K Dz"]:
    """Fraction of held-out test clips whose true z_{t+k} falls inside the
    calibrated interval, per (step, dim). Should be >= 1 - alpha on
    average, up to finite-sample noise, if calibration and test clips are
    exchangeable (i.i.d. from the same distribution)."""
    n = test_clips.shape[0]
    keys = jax.random.split(key, n)

    def per_clip(clip, k_):
        """Per-clip coverage: does the calibrated interval contain the
        true future latent at each horizon step?"""
        z_true, _ = model.encode(clip)
        t = clip.shape[0]
        k = min(k_steps, t - 1)
        mean, lower, upper = predict_with_interval(
            model, z_true[0], k, num_particles, noise_std, q_hat[:k], k_
        )
        target = z_true[1 : 1 + k]
        covered = jnp.logical_and(target >= lower, target <= upper).astype(jnp.float32)
        if k < k_steps:
            pad = jnp.full((k_steps - k, target.shape[-1]), jnp.nan)
            covered = jnp.concatenate([covered, pad], axis=0)
        return covered

    covered_all = jax.vmap(per_clip)(test_clips, keys)  # (N, k_steps, Dz)
    return jnp.nanmean(covered_all, axis=0)  # (k_steps, Dz)
