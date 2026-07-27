"""
Synthetic dynamical-system video generator, used to exercise the full
LyTimeT pipeline end-to-end (the paper's own datasets/videos aren't public,
so this stands in for one of their "five synthetic benchmarks" — a single
pendulum, theta'' + (g/l) sin(theta) = 0 — with rendered grayscale frames
and known ground-truth state s_t = (theta_t, theta_dot_t) for Phase 2
linear probing/evaluation).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


def pendulum_dynamics(state: Float[Array, "2"], dt: float, g: float = 9.8, length: float = 1.0):
    """Semi-implicit (symplectic) Euler step for theta'' = -(g/l) sin(theta)."""
    theta, theta_dot = state[0], state[1]
    theta_dot_next = theta_dot - dt * (g / length) * jnp.sin(theta)
    theta_next = theta + dt * theta_dot_next
    return jnp.stack([theta_next, theta_dot_next])


def rollout_pendulum(state0: Float[Array, "2"], n_steps: int, dt: float = 0.1):
    def step(s, _):
        s_next = pendulum_dynamics(s, dt)
        return s_next, s_next

    _, states = jax.lax.scan(step, state0, xs=None, length=n_steps)
    return jnp.concatenate([state0[None, :], states], axis=0)  # (n_steps+1, 2)


def render_pendulum_frame(
    theta: Float[Array, ""], image_size: int, length_px: float = 12.0, sigma: float = 1.6
) -> Float[Array, "1 H W"]:
    """Render a single pendulum bob as a soft Gaussian blob on a blank
    background, as a stand-in for a real rendered video frame."""
    center = image_size / 2.0
    bob_x = center + length_px * jnp.sin(theta)
    bob_y = center + length_px * jnp.cos(theta)

    ys, xs = jnp.meshgrid(
        jnp.arange(image_size), jnp.arange(image_size), indexing="ij"
    )
    dist2 = (xs - bob_x) ** 2 + (ys - bob_y) ** 2
    frame = jnp.exp(-dist2 / (2 * sigma ** 2))
    return frame[None, :, :]  # (1, H, W)


def sample_pendulum_clip(
    key: PRNGKeyArray,
    clip_len: int,
    image_size: int,
    dt: float = 0.1,
    theta0_range: float = 2.0,
):
    """Sample a random initial condition and return (clip, states):
    clip: (T, 1, H, W) rendered frames
    states: (T, 2) ground-truth (theta, theta_dot)
    """
    k_theta, k_omega = jax.random.split(key)
    theta0 = jax.random.uniform(k_theta, (), minval=-theta0_range, maxval=theta0_range)
    omega0 = jax.random.uniform(k_omega, (), minval=-1.0, maxval=1.0)
    state0 = jnp.stack([theta0, omega0])

    states = rollout_pendulum(state0, clip_len - 1, dt=dt)  # (T, 2)
    frames = jax.vmap(lambda s: render_pendulum_frame(s[0], image_size))(states)
    return frames, states


def make_pendulum_batch(
    key: PRNGKeyArray, batch_size: int, clip_len: int, image_size: int, dt: float = 0.1
):
    keys = jax.random.split(key, batch_size)
    clips, states = jax.vmap(
        lambda k: sample_pendulum_clip(k, clip_len, image_size, dt)
    )(keys)
    return clips, states  # (B,T,1,H,W), (B,T,2)


# --------------------------------------------------------------------------
# Actuated pendulum: adds a continuous torque input and a discrete damping-
# mode switch, for exercising/comparing action-conditioned dynamics models
# (see actions.py). Genuinely control-affine in the continuous torque:
#   theta'' = -(g/l) sin(theta) - damping(mode) * theta_dot + torque / (m l^2)
# -- additive and *linear* in torque, which is exactly the structural
# assumption `ControlAffineAction*Transition` encodes.
# --------------------------------------------------------------------------
_DAMPING_BY_MODE = jnp.array([0.0, 0.8])  # mode 0: undamped, mode 1: damped


def actuated_pendulum_dynamics(
    state: Float[Array, "2"],
    discrete_mode: Int[Array, ""],
    continuous_torque: Float[Array, ""],
    dt: float,
    g: float = 9.8,
    length: float = 1.0,
    mass: float = 1.0,
):
    """Semi-implicit Euler step for the torque-driven, mode-damped pendulum."""
    theta, theta_dot = state[0], state[1]
    damping = _DAMPING_BY_MODE[discrete_mode]
    torque_term = continuous_torque / (mass * length ** 2)
    theta_dot_next = theta_dot + dt * (
        -(g / length) * jnp.sin(theta) - damping * theta_dot + torque_term
    )
    theta_next = theta + dt * theta_dot_next
    return jnp.stack([theta_next, theta_dot_next])


def rollout_actuated_pendulum(
    state0: Float[Array, "2"],
    discrete_modes: Int[Array, "n_steps"],
    continuous_torques: Float[Array, "n_steps"],
    dt: float = 0.1,
):
    """discrete_modes, continuous_torques: per-step actions, length n_steps.
    Returns states of shape (n_steps+1, 2), states[0] = state0."""

    def step(s, a):
        mode, torque = a
        s_next = actuated_pendulum_dynamics(s, mode, torque, dt)
        return s_next, s_next

    _, states = jax.lax.scan(step, state0, xs=(discrete_modes, continuous_torques))
    return jnp.concatenate([state0[None, :], states], axis=0)


def sample_actuated_pendulum_episode(
    key: PRNGKeyArray,
    n_steps: int,
    dt: float = 0.1,
    theta0_range: float = 2.0,
    torque_scale: float = 1.5,
):
    """Sample a random initial condition and a random action sequence
    (piecewise-constant discrete mode per episode, smoothly-varying
    continuous torque), and roll out the true actuated dynamics.

    Returns (states, discrete_modes, continuous_torques):
      states:            (n_steps+1, 2)
      discrete_modes:     (n_steps,) int32 in {0, 1}
      continuous_torques: (n_steps,) float32, roughly in [-torque_scale, torque_scale]
    """
    k_theta, k_omega, k_mode, k_torque = jax.random.split(key, 4)
    theta0 = jax.random.uniform(k_theta, (), minval=-theta0_range, maxval=theta0_range)
    omega0 = jax.random.uniform(k_omega, (), minval=-1.0, maxval=1.0)
    state0 = jnp.stack([theta0, omega0])

    discrete_modes = jax.random.bernoulli(k_mode, p=0.5, shape=(n_steps,)).astype(jnp.int32)

    # Smoothly-varying torque: low-frequency sinusoid with random phase/
    # amplitude, rather than i.i.d. noise, so it's a plausible control signal.
    k_amp, k_phase, k_freq = jax.random.split(k_torque, 3)
    amp = jax.random.uniform(k_amp, (), minval=0.2, maxval=torque_scale)
    phase = jax.random.uniform(k_phase, (), minval=0.0, maxval=2 * jnp.pi)
    freq = jax.random.uniform(k_freq, (), minval=0.5, maxval=2.0)
    t = jnp.arange(n_steps) * dt
    continuous_torques = amp * jnp.sin(2 * jnp.pi * freq * t + phase)

    states = rollout_actuated_pendulum(state0, discrete_modes, continuous_torques, dt=dt)
    return states, discrete_modes, continuous_torques


def make_actuated_pendulum_batch(
    key: PRNGKeyArray, batch_size: int, n_steps: int, dt: float = 0.1, torque_scale: float = 1.5
):
    keys = jax.random.split(key, batch_size)
    states, discrete_modes, continuous_torques = jax.vmap(
        lambda k: sample_actuated_pendulum_episode(k, n_steps, dt, torque_scale=torque_scale)
    )(keys)
    return states, discrete_modes, continuous_torques
    # states: (B, n_steps+1, 2), discrete_modes: (B, n_steps), continuous_torques: (B, n_steps)
