import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from lytimet.ode_transition import (
    ODEVectorField,
    NeuralODETransition,
    continuous_lyapunov_loss,
)


@pytest.fixture
def small_ode_transition():
    key = jax.random.PRNGKey(0)
    return NeuralODETransition(dz=4, hidden=8, dt=1.0, num_internal_steps=2, key=key)


def test_vector_field_output_shape(small_ode_transition, rng_key):
    z = jax.random.normal(rng_key, (4,))
    dzdt = small_ode_transition.vector_field(z)
    assert dzdt.shape == (4,)
    assert jnp.all(jnp.isfinite(dzdt))


def test_call_integrates_one_step(small_ode_transition, rng_key):
    z0 = jax.random.normal(rng_key, (4,))
    z1 = small_ode_transition(z0)
    assert z1.shape == (4,)
    assert jnp.all(jnp.isfinite(z1))


def test_rollout_shape_matches_k_steps(small_ode_transition, rng_key):
    z0 = jax.random.normal(rng_key, (4,))
    traj = small_ode_transition.rollout(z0, k_steps=5)
    assert traj.shape == (5, 4)
    assert jnp.all(jnp.isfinite(traj))


def test_rollout_consistent_with_repeated_single_steps(small_ode_transition, rng_key):
    """Integrating straight to t=2*dt should (approximately) match applying
    two single-dt steps in sequence, since the underlying vector field and
    solver are the same either way."""
    z0 = jax.random.normal(rng_key, (4,))
    traj = small_ode_transition.rollout(z0, k_steps=2)

    z1 = small_ode_transition(z0)
    z2 = small_ode_transition(z1)

    assert jnp.allclose(traj[0], z1, atol=1e-3)
    assert jnp.allclose(traj[1], z2, atol=1e-3)


def test_gradients_flow_through_ode_solve(small_ode_transition, rng_key):
    z0 = jax.random.normal(rng_key, (4,))

    def loss_fn(model, z0):
        z1 = model(z0)
        return jnp.sum(z1 ** 2)

    grads = eqx.filter_grad(loss_fn)(small_ode_transition, z0)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) > 0
    assert any(jnp.any(leaf != 0) for leaf in leaves)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_continuous_lyapunov_loss_zero_for_contracting_field():
    """A linear vector field dz/dt = -z is globally contracting under
    V(z) = ||z||^2 (dV/dt = -2||z||^2 <= 0 everywhere), so the continuous
    Lyapunov violation should be (near) zero."""

    class LinearContracting(eqx.Module):
        def vector_field(self, z, t=0.0):
            return -z

    w = jnp.eye(3)
    z_seq = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    loss = continuous_lyapunov_loss(LinearContracting(), w, z_seq, select_idx=None)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_continuous_lyapunov_loss_positive_for_expanding_field():
    class LinearExpanding(eqx.Module):
        def vector_field(self, z, t=0.0):
            return z

    w = jnp.eye(3)
    z_seq = jax.random.normal(jax.random.PRNGKey(0), (5, 3)) + 1.0
    loss = continuous_lyapunov_loss(LinearExpanding(), w, z_seq, select_idx=None)
    assert float(loss) > 0.0


def test_continuous_lyapunov_loss_respects_select_idx():
    class LinearContracting(eqx.Module):
        def vector_field(self, z, t=0.0):
            return -z

    w = jnp.eye(2)
    z_seq = jax.random.normal(jax.random.PRNGKey(0), (5, 4))
    loss = continuous_lyapunov_loss(
        LinearContracting(), w, z_seq, select_idx=jnp.array([0, 2])
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
