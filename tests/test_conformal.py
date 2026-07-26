import jax
import jax.numpy as jnp
import pytest

from lytimet.conformal import (
    particle_rollout,
    ensemble_mean_std,
    calibrate_conformal,
    predict_with_interval,
    evaluate_coverage,
)


def test_particle_rollout_shape(tiny_model, rng_key):
    z0 = jax.random.normal(rng_key, (tiny_model.cfg.dz,))
    particles = particle_rollout(tiny_model, z0, k_steps=3, num_particles=8,
                                  noise_std=0.05, key=rng_key)
    assert particles.shape == (8, 3, tiny_model.cfg.dz)
    assert jnp.all(jnp.isfinite(particles))


def test_particle_rollout_zero_noise_collapses_to_single_trajectory(tiny_model, rng_key):
    z0 = jax.random.normal(rng_key, (tiny_model.cfg.dz,))
    particles = particle_rollout(tiny_model, z0, k_steps=3, num_particles=6,
                                  noise_std=0.0, key=rng_key)
    # With zero perturbation, every particle should follow the identical
    # (deterministic) trajectory.
    for i in range(1, 6):
        assert jnp.allclose(particles[0], particles[i], atol=1e-5)


def test_particle_rollout_nonzero_noise_gives_diverse_particles(tiny_model, rng_key):
    z0 = jax.random.normal(rng_key, (tiny_model.cfg.dz,))
    particles = particle_rollout(tiny_model, z0, k_steps=3, num_particles=6,
                                  noise_std=0.5, key=rng_key)
    assert not jnp.allclose(particles[0], particles[1], atol=1e-4)


def test_ensemble_mean_std_shapes_and_std_nonnegative():
    particles = jax.random.normal(jax.random.PRNGKey(0), (10, 4, 3))
    mean, std = ensemble_mean_std(particles)
    assert mean.shape == (4, 3)
    assert std.shape == (4, 3)
    assert jnp.all(std > 0.0)  # eps ensures strictly positive even at zero variance


def test_ensemble_mean_std_matches_manual_computation():
    particles = jax.random.normal(jax.random.PRNGKey(0), (20, 3, 2))
    mean, std = ensemble_mean_std(particles, eps=0.0)
    assert jnp.allclose(mean, jnp.mean(particles, axis=0))
    assert jnp.allclose(std, jnp.std(particles, axis=0))


def test_calibrate_conformal_output_shape_and_nonnegative(tiny_model, pendulum_batch, rng_key):
    clips, _ = pendulum_batch
    k_steps = 3
    q_hat = calibrate_conformal(
        tiny_model, clips, k_steps=k_steps, num_particles=8,
        noise_std=0.05, alpha=0.2, key=rng_key,
    )
    assert q_hat.shape == (k_steps, tiny_model.cfg.dz)
    assert jnp.all(q_hat >= 0.0)
    assert jnp.all(jnp.isfinite(q_hat))


def test_calibrate_conformal_smaller_alpha_gives_larger_or_equal_q_hat(
    tiny_model, pendulum_batch, rng_key
):
    """A stricter coverage target (smaller alpha) should require an equal or
    wider calibrated interval (monotonicity of the empirical quantile)."""
    clips, _ = pendulum_batch
    q_hat_loose = calibrate_conformal(
        tiny_model, clips, k_steps=3, num_particles=8, noise_std=0.05,
        alpha=0.5, key=rng_key,
    )
    q_hat_strict = calibrate_conformal(
        tiny_model, clips, k_steps=3, num_particles=8, noise_std=0.05,
        alpha=0.05, key=rng_key,
    )
    assert jnp.all(q_hat_strict >= q_hat_loose - 1e-6)


def test_predict_with_interval_bounds_are_ordered(tiny_model, rng_key):
    z0 = jax.random.normal(rng_key, (tiny_model.cfg.dz,))
    q_hat = jnp.ones((3, tiny_model.cfg.dz))
    mean, lower, upper = predict_with_interval(
        tiny_model, z0, k_steps=3, num_particles=8, noise_std=0.05,
        q_hat=q_hat, key=rng_key,
    )
    assert jnp.all(lower <= mean + 1e-6)
    assert jnp.all(mean <= upper + 1e-6)


def test_predict_with_interval_zero_q_hat_collapses_to_mean(tiny_model, rng_key):
    z0 = jax.random.normal(rng_key, (tiny_model.cfg.dz,))
    q_hat = jnp.zeros((3, tiny_model.cfg.dz))
    mean, lower, upper = predict_with_interval(
        tiny_model, z0, k_steps=3, num_particles=8, noise_std=0.05,
        q_hat=q_hat, key=rng_key,
    )
    assert jnp.allclose(lower, mean)
    assert jnp.allclose(upper, mean)


@pytest.mark.slow
def test_evaluate_coverage_reasonably_close_to_target(tiny_model, rng_key):
    """Statistical sanity check (not a tight bound, to avoid CI flakiness):
    with a generous alpha and enough clips, empirical coverage should land
    in a broad, plausible band around the target."""
    from lytimet.data import make_pendulum_batch

    k_calib, k_test, k_cal_run, k_test_run = jax.random.split(rng_key, 4)
    calib_clips, _ = make_pendulum_batch(k_calib, 40, tiny_model.cfg.clip_len, tiny_model.cfg.image_size)
    test_clips, _ = make_pendulum_batch(k_test, 60, tiny_model.cfg.clip_len, tiny_model.cfg.image_size)

    alpha = 0.2
    k_steps = 3
    q_hat = calibrate_conformal(
        tiny_model, calib_clips, k_steps=k_steps, num_particles=16,
        noise_std=0.05, alpha=alpha, key=k_cal_run,
    )
    coverage = evaluate_coverage(
        tiny_model, test_clips, k_steps=k_steps, num_particles=16,
        noise_std=0.05, q_hat=q_hat, key=k_test_run,
    )
    mean_coverage = float(jnp.nanmean(coverage))
    # Loose sanity band: should be roughly in the right ballpark, not exactly
    # (1 - alpha) due to finite-sample noise with a small calibration set.
    assert 0.5 <= mean_coverage <= 1.0
