import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from lytimet.actions import (
    ActionEncoder,
    ConcatActionTransition,
    ConcatActionODETransition,
    ControlAffineActionTransition,
    ControlAffineActionODETransition,
)

DZ = 4
DISCRETE_SIZES = (3, 2)
CONTINUOUS_DIM = 1
EMBED_DIM = 4
HIDDEN = 16


def _all_models(key):
    return {
        "concat_residual": ConcatActionTransition(
            DZ, DISCRETE_SIZES, CONTINUOUS_DIM, EMBED_DIM, HIDDEN, depth=2, key=key
        ),
        "concat_ode": ConcatActionODETransition(
            DZ, DISCRETE_SIZES, CONTINUOUS_DIM, EMBED_DIM, HIDDEN,
            dt=1.0, num_internal_steps=2, key=key,
        ),
        "affine_residual": ControlAffineActionTransition(
            DZ, DISCRETE_SIZES, CONTINUOUS_DIM, EMBED_DIM, HIDDEN, key=key
        ),
        "affine_ode": ControlAffineActionODETransition(
            DZ, DISCRETE_SIZES, CONTINUOUS_DIM, EMBED_DIM, HIDDEN,
            dt=1.0, num_internal_steps=2, key=key,
        ),
    }


@pytest.fixture
def models(rng_key):
    return _all_models(rng_key)


def test_action_encoder_feature_dim():
    key = jax.random.PRNGKey(0)
    enc = ActionEncoder(discrete_sizes=(3, 2), continuous_dim=2, embed_dim=4, key=key)
    assert enc.feature_dim == 2 * 4 + 2
    feat = enc(jnp.array([1, 0]), jnp.array([0.5, -0.5]))
    assert feat.shape == (10,)


def test_action_encoder_discrete_only():
    key = jax.random.PRNGKey(0)
    enc = ActionEncoder(discrete_sizes=(3,), continuous_dim=0, embed_dim=4, key=key)
    feat = enc(jnp.array([2]), jnp.zeros((0,)))
    assert feat.shape == (4,)


def test_action_encoder_continuous_only():
    key = jax.random.PRNGKey(0)
    enc = ActionEncoder(discrete_sizes=(), continuous_dim=3, embed_dim=4, key=key)
    feat = enc(jnp.zeros((0,), dtype=jnp.int32), jnp.array([1.0, 2.0, 3.0]))
    assert feat.shape == (3,)


@pytest.mark.parametrize(
    "name", ["concat_residual", "concat_ode", "affine_residual", "affine_ode"]
)
def test_single_step_shape_and_finite(models, rng_key, name):
    model = models[name]
    z0 = jax.random.normal(rng_key, (DZ,))
    d_a = jnp.array([1, 0])
    c_a = jnp.array([0.5])
    z1 = model(z0, d_a, c_a)
    assert z1.shape == (DZ,)
    assert jnp.all(jnp.isfinite(z1))


@pytest.mark.parametrize(
    "name", ["concat_residual", "concat_ode", "affine_residual", "affine_ode"]
)
def test_rollout_shape_and_finite(models, rng_key, name):
    model = models[name]
    z0 = jax.random.normal(rng_key, (DZ,))
    k_steps = 5
    d_seq = jnp.zeros((k_steps, len(DISCRETE_SIZES)), dtype=jnp.int32)
    c_seq = jax.random.normal(rng_key, (k_steps, CONTINUOUS_DIM))
    traj = model.rollout(z0, d_seq, c_seq, k_steps)
    assert traj.shape == (k_steps, DZ)
    assert jnp.all(jnp.isfinite(traj))


@pytest.mark.parametrize(
    "name", ["concat_residual", "concat_ode", "affine_residual", "affine_ode"]
)
def test_rollout_matches_repeated_single_steps(models, rng_key, name):
    model = models[name]
    z0 = jax.random.normal(rng_key, (DZ,))
    d_seq = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
    c_seq = jnp.array([[0.3], [-0.4]])
    traj = model.rollout(z0, d_seq, c_seq, k_steps=2)

    z1 = model(z0, d_seq[0], c_seq[0])
    z2 = model(z1, d_seq[1], c_seq[1])
    assert jnp.allclose(traj[0], z1, atol=1e-4)
    assert jnp.allclose(traj[1], z2, atol=1e-4)


@pytest.mark.parametrize(
    "name", ["concat_residual", "concat_ode", "affine_residual", "affine_ode"]
)
def test_gradients_flow_and_are_nonzero(models, rng_key, name):
    model = models[name]
    z0 = jax.random.normal(rng_key, (DZ,))
    d_a = jnp.array([1, 0])
    c_a = jnp.array([0.5])

    def loss_fn(m, z0, d_a, c_a):
        z1 = m(z0, d_a, c_a)
        return jnp.sum(z1 ** 2)

    grads = eqx.filter_grad(loss_fn)(model, z0, d_a, c_a)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) > 0
    assert any(jnp.any(leaf != 0) for leaf in leaves)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_control_affine_discrete_only_action_space(rng_key):
    """continuous_dim=0 should still work (control term is just zero)."""
    model = ControlAffineActionTransition(DZ, (3,), 0, EMBED_DIM, HIDDEN, key=rng_key)
    z0 = jax.random.normal(rng_key, (DZ,))
    z1 = model(z0, jnp.array([1]), jnp.zeros((0,)))
    assert z1.shape == (DZ,)
    assert jnp.all(jnp.isfinite(z1))


def test_concat_continuous_only_action_space(rng_key):
    model = ConcatActionTransition(DZ, (), 2, EMBED_DIM, HIDDEN, depth=1, key=rng_key)
    z0 = jax.random.normal(rng_key, (DZ,))
    z1 = model(z0, jnp.zeros((0,), dtype=jnp.int32), jnp.array([0.3, -0.1]))
    assert z1.shape == (DZ,)


def test_control_affine_drift_only_differs_from_full_update_residual(rng_key):
    model = ControlAffineActionTransition(DZ, (3,), 2, EMBED_DIM, HIDDEN, key=rng_key)
    z0 = jax.random.normal(rng_key, (DZ,))
    d_a = jnp.array([2])
    large_action = jnp.array([5.0, -5.0])

    z_drift_only = model.drift_only_step(z0, d_a)
    z_full = model(z0, d_a, large_action)
    assert not jnp.allclose(z_drift_only, z_full, atol=1e-3)


def test_control_affine_drift_only_matches_full_update_at_zero_action(rng_key):
    """At continuous_action=0, the full update should equal the drift-only
    update exactly (the control term vanishes)."""
    model = ControlAffineActionTransition(DZ, (3,), 2, EMBED_DIM, HIDDEN, key=rng_key)
    z0 = jax.random.normal(rng_key, (DZ,))
    d_a = jnp.array([2])
    zero_action = jnp.zeros((2,))

    z_drift_only = model.drift_only_step(z0, d_a)
    z_full = model(z0, d_a, zero_action)
    assert jnp.allclose(z_drift_only, z_full, atol=1e-6)


def test_control_affine_ode_drift_only_differs_from_full_vector_field(rng_key):
    model = ControlAffineActionODETransition(DZ, (3,), 2, EMBED_DIM, HIDDEN, key=rng_key)
    z0 = jax.random.normal(rng_key, (DZ,))
    d_a = jnp.array([2])
    large_action = jnp.array([5.0, -5.0])

    dzdt_drift = model.drift_only_vector_field(z0, d_a)
    dzdt_full = model.vector_field(z0, d_a, large_action)
    assert not jnp.allclose(dzdt_drift, dzdt_full, atol=1e-3)


def test_control_affine_ode_drift_only_matches_full_vector_field_at_zero_action(rng_key):
    model = ControlAffineActionODETransition(DZ, (3,), 2, EMBED_DIM, HIDDEN, key=rng_key)
    z0 = jax.random.normal(rng_key, (DZ,))
    d_a = jnp.array([2])
    zero_action = jnp.zeros((2,))

    dzdt_drift = model.drift_only_vector_field(z0, d_a)
    dzdt_full = model.vector_field(z0, d_a, zero_action)
    assert jnp.allclose(dzdt_drift, dzdt_full, atol=1e-6)


def test_control_affine_linear_in_continuous_action(rng_key):
    """The defining property of control-affine dynamics: the control term
    B(z, discrete) @ a is *exactly* linear in a. Check superposition:
    f(z, d, a1 + a2) - f(z, d, 0) == [f(z, d, a1) - f(z, d, 0)] + [f(z, d, a2) - f(z, d, 0)]."""
    model = ControlAffineActionTransition(DZ, (3,), 2, EMBED_DIM, HIDDEN, key=rng_key)
    z0 = jax.random.normal(rng_key, (DZ,))
    d_a = jnp.array([1])
    a1 = jnp.array([0.3, -0.2])
    a2 = jnp.array([-0.7, 0.4])
    zero_action = jnp.zeros((2,))

    baseline = model(z0, d_a, zero_action)
    delta1 = model(z0, d_a, a1) - baseline
    delta2 = model(z0, d_a, a2) - baseline
    delta_sum = model(z0, d_a, a1 + a2) - baseline

    assert jnp.allclose(delta1 + delta2, delta_sum, atol=1e-4)
