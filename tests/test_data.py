import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lytimet.data import (
    pendulum_dynamics,
    rollout_pendulum,
    render_pendulum_frame,
    sample_pendulum_clip,
    make_pendulum_batch,
)


def test_pendulum_dynamics_fixed_point_at_rest():
    """theta=0, theta_dot=0 is an equilibrium: should stay at (0, 0)."""
    state = jnp.array([0.0, 0.0])
    next_state = pendulum_dynamics(state, dt=0.1)
    assert jnp.allclose(next_state, jnp.zeros(2), atol=1e-6)


def test_pendulum_dynamics_output_shape_and_finite():
    state = jnp.array([0.5, -0.2])
    next_state = pendulum_dynamics(state, dt=0.1)
    assert next_state.shape == (2,)
    assert jnp.all(jnp.isfinite(next_state))


def test_rollout_pendulum_shape_and_starts_at_initial_state():
    state0 = jnp.array([1.0, 0.0])
    states = rollout_pendulum(state0, n_steps=10, dt=0.05)
    assert states.shape == (11, 2)
    assert jnp.allclose(states[0], state0)
    assert jnp.all(jnp.isfinite(states))


def test_rollout_pendulum_energy_roughly_bounded():
    """Semi-implicit Euler is symplectic-ish and should not blow up energy
    over a short, moderate-dt rollout for a moderate initial condition."""
    state0 = jnp.array([1.0, 0.0])
    states = rollout_pendulum(state0, n_steps=50, dt=0.05)
    theta, theta_dot = states[:, 0], states[:, 1]
    energy = 0.5 * theta_dot ** 2 - 9.8 * jnp.cos(theta)
    # energy shouldn't run away to huge values for this short, tame rollout
    assert jnp.all(jnp.abs(energy) < 100.0)


def test_render_pendulum_frame_shape_and_range():
    frame = render_pendulum_frame(jnp.array(0.3), image_size=16)
    assert frame.shape == (1, 16, 16)
    assert jnp.all(frame >= 0.0) and jnp.all(frame <= 1.0 + 1e-6)


def test_render_pendulum_frame_peak_near_expected_bob_position():
    """The brightest pixel should be near where we expect the bob to be
    for theta=0 (straight down from center)."""
    image_size = 32
    frame = render_pendulum_frame(jnp.array(0.0), image_size=image_size, length_px=10.0)
    frame_2d = frame[0]
    peak_idx = jnp.unravel_index(jnp.argmax(frame_2d), frame_2d.shape)
    center = image_size / 2.0
    # theta=0 -> bob_x = center, bob_y = center + length_px
    assert abs(float(peak_idx[1]) - center) < 2.0
    assert abs(float(peak_idx[0]) - (center + 10.0)) < 2.0


def test_sample_pendulum_clip_shapes():
    key = jax.random.PRNGKey(0)
    clip, states = sample_pendulum_clip(key, clip_len=6, image_size=16)
    assert clip.shape == (6, 1, 16, 16)
    assert states.shape == (6, 2)
    assert jnp.all(jnp.isfinite(clip))
    assert jnp.all(jnp.isfinite(states))


def test_actuated_pendulum_dynamics_zero_torque_undamped_matches_passive():
    """With torque=0 and mode=0 (undamped), the actuated dynamics should
    match the passive pendulum_dynamics exactly."""
    from lytimet.data import pendulum_dynamics, actuated_pendulum_dynamics

    state = jnp.array([0.4, -0.2])
    passive_next = pendulum_dynamics(state, dt=0.05)
    actuated_next = actuated_pendulum_dynamics(
        state, discrete_mode=jnp.array(0), continuous_torque=jnp.array(0.0), dt=0.05
    )
    assert jnp.allclose(passive_next, actuated_next, atol=1e-6)


def test_actuated_pendulum_dynamics_damped_mode_reduces_angular_velocity_magnitude():
    from lytimet.data import actuated_pendulum_dynamics

    state = jnp.array([0.0, 2.0])  # swinging through bottom with some speed
    undamped = actuated_pendulum_dynamics(
        state, discrete_mode=jnp.array(0), continuous_torque=jnp.array(0.0), dt=0.05
    )
    damped = actuated_pendulum_dynamics(
        state, discrete_mode=jnp.array(1), continuous_torque=jnp.array(0.0), dt=0.05
    )
    assert abs(float(damped[1])) < abs(float(undamped[1]))


def test_actuated_pendulum_dynamics_linear_in_torque():
    """Control-affine ground truth check: the *change* in next state from
    varying torque alone should be exactly linear in torque."""
    from lytimet.data import actuated_pendulum_dynamics

    state = jnp.array([0.3, 0.1])
    mode = jnp.array(0)
    dt = 0.05
    base = actuated_pendulum_dynamics(state, mode, jnp.array(0.0), dt)
    at_1 = actuated_pendulum_dynamics(state, mode, jnp.array(1.0), dt)
    at_2 = actuated_pendulum_dynamics(state, mode, jnp.array(2.0), dt)
    delta_1 = at_1 - base
    delta_2 = at_2 - base
    assert jnp.allclose(2 * delta_1, delta_2, atol=1e-5)


def test_rollout_actuated_pendulum_shape():
    from lytimet.data import rollout_actuated_pendulum

    state0 = jnp.array([1.0, 0.0])
    modes = jnp.zeros((10,), dtype=jnp.int32)
    torques = jnp.ones((10,)) * 0.1
    states = rollout_actuated_pendulum(state0, modes, torques, dt=0.05)
    assert states.shape == (11, 2)
    assert jnp.allclose(states[0], state0)
    assert jnp.all(jnp.isfinite(states))


def test_sample_actuated_pendulum_episode_shapes():
    from lytimet.data import sample_actuated_pendulum_episode

    key = jax.random.PRNGKey(0)
    states, modes, torques = sample_actuated_pendulum_episode(key, n_steps=15, dt=0.05)
    assert states.shape == (16, 2)
    assert modes.shape == (15,)
    assert torques.shape == (15,)
    assert set(np.unique(np.asarray(modes)).tolist()).issubset({0, 1})
    assert jnp.all(jnp.isfinite(states))


def test_make_actuated_pendulum_batch_shapes_and_diversity():
    from lytimet.data import make_actuated_pendulum_batch

    key = jax.random.PRNGKey(0)
    states, modes, torques = make_actuated_pendulum_batch(key, batch_size=4, n_steps=10, dt=0.05)
    assert states.shape == (4, 11, 2)
    assert modes.shape == (4, 10)
    assert torques.shape == (4, 10)
    assert not jnp.allclose(states[0, 0], states[1, 0])


def test_make_pendulum_batch_shapes_and_batch_diversity():
    key = jax.random.PRNGKey(0)
    clips, states = make_pendulum_batch(key, batch_size=4, clip_len=5, image_size=16)
    assert clips.shape == (4, 5, 1, 16, 16)
    assert states.shape == (4, 5, 2)
    # different batch elements should (almost surely) have different initial
    # conditions -- guards against an accidental key-reuse bug
    assert not jnp.allclose(states[0, 0], states[1, 0])
