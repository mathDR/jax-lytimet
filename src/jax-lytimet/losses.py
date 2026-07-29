"""
Loss functions for LyTimeT, matching the paper's equations:

  L_rec   = (1/T) sum_t || x_hat_t - x_t ||_2^2                       (eq. rec)
  L_pred  = (1/K) sum_k || x_hat_{t+k} - x_{t+k} ||_2^2                (eq. pred)
  L_phase1 = L_rec + lambda_pred * L_pred
  V(z_tilde) = || W z_tilde ||_2^2
  L_lyap  = (1/K) sum_k max(0, V(f_theta(z_tilde_k)) - V(z_tilde_k))
  L       = L_phase1 + lambda_lyap * L_lyap
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .model import LyTimeT, LatentTransition


def reconstruction_loss(
    recon: Float[Array, "T C H W"], target: Float[Array, "T C H W"]
) -> Float[Array, ""]:
    """L_rec: mean per-frame squared error, averaged over the T frames."""
    per_frame = jnp.sum((recon - target) ** 2, axis=(-3, -2, -1))  # (T,)
    return jnp.mean(per_frame)


def prediction_loss(
    x_future_hat: Float[Array, "K C H W"], x_future: Float[Array, "K C H W"]
) -> Float[Array, ""]:
    """L_pred: mean per-frame squared error over the K unrolled steps."""
    per_step = jnp.sum((x_future_hat - x_future) ** 2, axis=(-3, -2, -1))  # (K,)
    return jnp.mean(per_step)


def latent_dynamics_matching_loss(
    model: LyTimeT, z_seq: Float[Array, "T Dz"]
) -> Float[Array, ""]:
    """Directly supervise the dynamics module in latent space, using
    *consecutive encoded latents from the same clip* as free teacher-forcing
    targets -- no decoder, no multi-step rollout needed:

        pred_{t+1} = model.transition(z_t)      # one step, either dynamics_type
        loss       = || pred_{t+1} - z_{t+1} ||^2

    Deliberately does NOT use a finite-difference derivative estimate
    (z_{t+1} - z_t) / dt as the target, even for the Neural ODE transition.
    Dividing by dt amplifies any encoder noise as dt shrinks (variance
    scales like 1/dt^2), and for larger dt the difference quotient itself
    is only an O(dt) approximation of the true derivative -- both are
    avoidable. Since `NeuralODETransition.__call__` already integrates the
    vector field over the true elapsed interval (rather than linearizing
    it), comparing its output directly to the encoder's actual z_{t+1}
    supervises exactly the quantity we care about (where the learned flow
    lands after the true elapsed time) with no derivative-estimation error
    at all. This also means the *same* one-line loss works unmodified for
    both the residual-MLP and Neural ODE transitions.

    This decouples dynamics-learning from reconstruction-learning: the
    encoder is still trained via L_rec/L_pred, but the transition module
    gets a direct, low-variance gradient signal about the actual latent
    dynamics the encoder produced, independent of decoder quality.
    """
    z_t = z_seq[:-1]
    z_tp1 = z_seq[1:]
    pred = jax.vmap(model.transition)(z_t)
    return jnp.mean(jnp.sum((pred - z_tp1) ** 2, axis=-1))


def phase1_clip_loss(
    model: LyTimeT,
    clip: Float[Array, "T C H W"],
    k_steps: int,
    lambda_pred: float,
    lambda_dyn: float = 0.0,
):
    """Single-clip Phase-1 loss: reconstruct the whole clip, then roll the
    transition model forward K steps from an early frame and compare against
    the true future frames (which must exist in `clip`, i.e. k_steps <=
    T - start - 1 for whichever start index is used).

    Uses the first frame of the clip as z_t and the remaining frames (up to
    k_steps of them) as the ground-truth future, mirroring the paper's
    "encode a window, unroll K steps, decode, compare" procedure.

    If `lambda_dyn > 0`, also adds `latent_dynamics_matching_loss` -- a
    cheap, direct, latent-space supervision signal for the dynamics module
    (see that function's docstring). This is not part of the paper, but is
    a natural, nearly-free addition since the encoder already produces
    z_1..z_T for the whole clip.
    """
    recon, z = model.reconstruct_clip(clip)
    l_rec = reconstruction_loss(recon, clip)

    t = clip.shape[0]
    k = min(k_steps, t - 1)
    x_future_hat, _ = model.forecast(z[0], k)
    x_future = clip[1 : 1 + k]
    l_pred = prediction_loss(x_future_hat, x_future)

    l_phase1 = l_rec + lambda_pred * l_pred
    aux = {"l_rec": l_rec, "l_pred": l_pred}

    if lambda_dyn > 0.0:
        l_dyn = latent_dynamics_matching_loss(model, z)
        l_phase1 = l_phase1 + lambda_dyn * l_dyn
        aux["l_dyn"] = l_dyn

    aux["l_phase1"] = l_phase1
    return l_phase1, aux


def lyapunov_energy(w: Float[Array, "Dv Dz"], z: Float[Array, "Dz"]) -> Float[Array, ""]:
    """V(z) = ||W z||_2^2"""
    return jnp.sum((w @ z) ** 2)


def lyapunov_loss(
    transition: LatentTransition,
    w: Float[Array, "Dv Ds"],
    z_seq: Float[Array, "K Dz"],
    select_idx: Float[Array, "Ds"] | None = None,
) -> Float[Array, ""]:
    """L_lyap = (1/K) sum_k max(0, V(f_theta(z_k)) - V(z_k))

    `z_seq` is a sequence of *full* latent states z_1 .. z_K along an
    observed or rolled-out trajectory. The transition f_theta always acts
    on the full latent vector (its trained dimensionality); the Lyapunov
    energy V, however, is evaluated only on the selected, interpretable
    dimensions z_tilde (select_idx), consistent with Phase 2 of the paper:
    stability is regularized on the *extracted physical variables*, not on
    nuisance latent dimensions.
    """
    z_next = jax.vmap(transition)(z_seq)  # full-dim transition
    z_tilde_now = z_seq if select_idx is None else z_seq[:, select_idx]
    z_tilde_next = z_next if select_idx is None else z_next[:, select_idx]
    v_now = jax.vmap(lambda z: lyapunov_energy(w, z))(z_tilde_now)
    v_next = jax.vmap(lambda z: lyapunov_energy(w, z))(z_tilde_next)
    violation = jnp.maximum(0.0, v_next - v_now)
    return jnp.mean(violation)


def phase2_clip_loss(
    model: LyTimeT,
    w: Float[Array, "Dv Dz"],
    clip: Float[Array, "T C H W"],
    k_steps: int,
    lambda_pred: float,
    lambda_lyap: float,
    select_idx: Float[Array, "Dv"] | None = None,
    use_continuous_lyapunov: bool = False,
    lambda_dyn: float = 0.0,
):
    """Combined objective L = L_phase1 + lambda_lyap * L_lyap (+ optional
    lambda_dyn * latent_dynamics_matching_loss, see phase1_clip_loss).

    `select_idx` are the indices of the top-ranked (most physically
    meaningful) latent dimensions z_tilde, as chosen by linear-probe
    ranking in Phase 2 (see probe.py). The Lyapunov loss is computed on
    the restriction of the encoded trajectory to those dimensions.

    If `use_continuous_lyapunov` is True, the model's transition must be a
    `NeuralODETransition` (see ode_transition.py); the exact continuous-time
    dV/dt <= 0 penalty is used instead of the paper's discrete
    max(0, V(z_{t+1}) - V(z_t)) check.
    """
    l_phase1, aux = phase1_clip_loss(model, clip, k_steps, lambda_pred, lambda_dyn)

    _, z = model.reconstruct_clip(clip)  # (T, Dz), full latent
    if use_continuous_lyapunov:
        from .ode_transition import continuous_lyapunov_loss

        l_lyap = continuous_lyapunov_loss(model.transition, w, z, select_idx)
    else:
        l_lyap = lyapunov_loss(model.transition, w, z, select_idx)

    total = l_phase1 + lambda_lyap * l_lyap
    aux = {**aux, "l_lyap": l_lyap, "total": total}
    return total, aux
