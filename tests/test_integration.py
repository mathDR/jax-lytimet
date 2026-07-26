"""
Integration smoke tests: these don't re-check individual function
correctness (the unit tests do that) -- they exist to catch wiring bugs
that only show up when the full pipeline runs end-to-end (e.g. a shape
mismatch between two modules that each pass their own unit tests in
isolation). All marked `slow` since they run real (tiny) training loops.
"""
import dataclasses

import jax
import pytest

from lytimet.model import LyTimeT, LyTimeTConfig
from lytimet.data import make_pendulum_batch
from lytimet.train import Phase1Config, Phase2Config, train_phase1, train_phase2
from lytimet.conformal import calibrate_conformal, predict_with_interval, evaluate_coverage


pytestmark = pytest.mark.slow


def _tiny_e2e_cfg(**overrides):
    base = dict(
        image_size=16, patch_size=4, in_channels=1, dim=16, depth=1,
        num_heads=2, dz=4, clip_len=6, transition_hidden=16, transition_depth=1,
    )
    base.update(overrides)
    return LyTimeTConfig(**base)


@pytest.mark.parametrize("dynamics_type", ["residual_mlp", "neural_ode"])
def test_full_phase1_phase2_pipeline(dynamics_type):
    cfg = _tiny_e2e_cfg(
        dynamics_type=dynamics_type,
        **({"ode_solver_steps": 2, "ode_dt": 1.0} if dynamics_type == "neural_ode" else {}),
    )
    key = jax.random.PRNGKey(0)
    k_model, k_phase1, k_phase2 = jax.random.split(key, 3)
    model = LyTimeT(cfg, key=k_model)

    def data_fn_clips(k, b):
        clips, _ = make_pendulum_batch(k, b, cfg.clip_len, cfg.image_size)
        return clips

    def data_fn_clips_and_states(k, b):
        return make_pendulum_batch(k, b, cfg.clip_len, cfg.image_size)

    p1_cfg = Phase1Config(lr=3e-3, k_steps=2, num_steps=3, batch_size=2)
    model, hist1 = train_phase1(model, data_fn_clips, p1_cfg, k_phase1, log_every=1000)
    assert len(hist1) == p1_cfg.num_steps

    p2_cfg = Phase2Config(
        lr=1e-3, k_steps=2, lambda_lyap=0.1, num_steps=3, batch_size=2, n_select=2,
        use_continuous_lyapunov=(dynamics_type == "neural_ode"),
    )
    model, w_lyap, select_idx, hist2 = train_phase2(
        model, data_fn_clips_and_states, p2_cfg, k_phase2, log_every=1000
    )
    assert len(hist2) == p2_cfg.num_steps
    assert select_idx.shape == (2,)


def test_full_pipeline_with_conformal_calibration():
    cfg = _tiny_e2e_cfg(dynamics_type="neural_ode", ode_solver_steps=2, ode_dt=1.0)
    key = jax.random.PRNGKey(0)
    k_model, k_train, k_calib, k_test, k_pred = jax.random.split(key, 5)
    model = LyTimeT(cfg, key=k_model)

    def data_fn(k, b):
        clips, _ = make_pendulum_batch(k, b, cfg.clip_len, cfg.image_size)
        return clips

    p1_cfg = Phase1Config(lr=3e-3, k_steps=2, num_steps=5, batch_size=3)
    model, _ = train_phase1(model, data_fn, p1_cfg, k_train, log_every=1000)

    k_steps = 3
    calib_clips = data_fn(k_calib, 10)
    q_hat = calibrate_conformal(
        model, calib_clips, k_steps, num_particles=6, noise_std=0.05,
        alpha=0.2, key=k_calib,
    )
    assert q_hat.shape == (k_steps, cfg.dz)

    test_clips = data_fn(k_test, 10)
    coverage = evaluate_coverage(
        model, test_clips, k_steps, num_particles=6, noise_std=0.05,
        q_hat=q_hat, key=k_test,
    )
    assert coverage.shape == (k_steps, cfg.dz)

    z0, _ = model.encode(test_clips[0])
    mean, lower, upper = predict_with_interval(
        model, z0[0], k_steps, num_particles=6, noise_std=0.05, q_hat=q_hat, key=k_pred,
    )
    assert mean.shape == lower.shape == upper.shape == (k_steps, cfg.dz)
