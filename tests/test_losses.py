import dataclasses

import jax
import jax.numpy as jnp
import pytest

from lytimet.model import LyTimeT
from lytimet.losses import (
    reconstruction_loss,
    prediction_loss,
    phase1_clip_loss,
    lyapunov_energy,
    lyapunov_loss,
    latent_dynamics_matching_loss,
    phase2_clip_loss,
)


def test_reconstruction_loss_zero_for_identical_input():
    x = jax.random.uniform(jax.random.PRNGKey(0), (4, 1, 8, 8))
    assert float(reconstruction_loss(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_positive_for_different_input():
    key = jax.random.PRNGKey(0)
    a = jax.random.uniform(key, (4, 1, 8, 8))
    b = jax.random.uniform(jax.random.PRNGKey(1), (4, 1, 8, 8))
    assert float(reconstruction_loss(a, b)) > 0.0


def test_prediction_loss_zero_for_identical_input():
    x = jax.random.uniform(jax.random.PRNGKey(0), (3, 1, 8, 8))
    assert float(prediction_loss(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_phase1_clip_loss_returns_finite_scalar_and_aux(tiny_model, dummy_clip):
    loss, aux = phase1_clip_loss(tiny_model, dummy_clip, k_steps=2, lambda_pred=1.0)
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    for key in ("l_rec", "l_pred", "l_phase1"):
        assert key in aux
        assert jnp.isfinite(aux[key])


def test_phase1_clip_loss_lambda_dyn_adds_aux_term(tiny_model, dummy_clip):
    loss_no_dyn, aux_no_dyn = phase1_clip_loss(tiny_model, dummy_clip, 2, 1.0, lambda_dyn=0.0)
    loss_dyn, aux_dyn = phase1_clip_loss(tiny_model, dummy_clip, 2, 1.0, lambda_dyn=0.5)
    assert "l_dyn" not in aux_no_dyn
    assert "l_dyn" in aux_dyn
    assert jnp.isfinite(aux_dyn["l_dyn"])


@pytest.mark.parametrize("dynamics_type", ["residual_mlp", "neural_ode"])
def test_latent_dynamics_matching_loss_both_dynamics(tiny_cfg, rng_key, dynamics_type):
    cfg = tiny_cfg if dynamics_type == "residual_mlp" else dataclasses.replace(
        tiny_cfg, dynamics_type="neural_ode", ode_solver_steps=2, ode_dt=1.0
    )
    model = LyTimeT(cfg, key=rng_key)
    z_seq = jax.random.normal(rng_key, (5, cfg.dz))
    loss = latent_dynamics_matching_loss(model, z_seq)
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_lyapunov_energy_is_nonnegative_quadratic_form(rng_key):
    w = jax.random.normal(rng_key, (3, 4))
    z = jax.random.normal(jax.random.PRNGKey(1), (4,))
    energy = lyapunov_energy(w, z)
    assert energy >= 0.0
    assert float(lyapunov_energy(w, jnp.zeros(4))) == pytest.approx(0.0, abs=1e-6)


def test_lyapunov_loss_zero_for_contracting_linear_map():
    """A transition that scales z by 0.5 should have zero Lyapunov
    violation under V(z) = ||z||^2, since V shrinks every step."""

    class Shrink:
        def __call__(self, z):
            return 0.5 * z

    w = jnp.eye(3)
    z_seq = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    loss = lyapunov_loss(Shrink(), w, z_seq, select_idx=None)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_lyapunov_loss_positive_for_expanding_linear_map():
    class Expand:
        def __call__(self, z):
            return 2.0 * z

    w = jnp.eye(3)
    z_seq = jax.random.normal(jax.random.PRNGKey(0), (5, 3)) + 1.0  # avoid all-zero rows
    loss = lyapunov_loss(Expand(), w, z_seq, select_idx=None)
    assert float(loss) > 0.0


def test_lyapunov_loss_select_idx_restricts_dimensions():
    class Identity:
        def __call__(self, z):
            return z

    w = jnp.eye(2)
    z_seq = jax.random.normal(jax.random.PRNGKey(0), (5, 4))
    loss = lyapunov_loss(Identity(), w, z_seq, select_idx=jnp.array([0, 1]))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)  # identity map never violates


def test_phase2_clip_loss_finite_and_has_lyap_term(tiny_model, dummy_clip):
    w = jnp.eye(tiny_model.cfg.dz)
    loss, aux = phase2_clip_loss(
        tiny_model, w, dummy_clip, k_steps=2, lambda_pred=1.0, lambda_lyap=0.1
    )
    assert jnp.isfinite(loss)
    assert "l_lyap" in aux and jnp.isfinite(aux["l_lyap"])
    assert "total" in aux


def test_phase2_clip_loss_continuous_requires_neural_ode(tiny_ode_cfg, rng_key, dummy_clip):
    model = LyTimeT(tiny_ode_cfg, key=rng_key)
    w = jnp.eye(model.cfg.dz)
    loss, aux = phase2_clip_loss(
        model, w, dummy_clip, k_steps=2, lambda_pred=1.0, lambda_lyap=0.1,
        use_continuous_lyapunov=True,
    )
    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux["l_lyap"])
