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


def pendulum_dynamics(
    state: Float[Array, "2"], dt: float, g: float = 9.8, length: float = 1.0
) -> Float[Array, "2"]:
    """Semi-implicit (symplectic) Euler step for theta'' = -(g/l) sin(theta).

    Args:
        state: Current `(theta, theta_dot)`, shape `(2,)`.
        dt: Integration time step.
        g: Gravitational acceleration.
        length: Pendulum rod length.

    Returns:
        Next state `(theta, theta_dot)`, shape `(2,)`.
    """
    theta, theta_dot = state[0], state[1]
    theta_dot_next = theta_dot - dt * (g / length) * jnp.sin(theta)
    theta_next = theta + dt * theta_dot_next
    return jnp.stack([theta_next, theta_dot_next])


def rollout_pendulum(
    state0: Float[Array, "2"], n_steps: int, dt: float = 0.1
) -> Float[Array, "n_steps_plus_1 2"]:
    """Roll out the passive pendulum dynamics for `n_steps` steps.

    Args:
        state0: Initial `(theta, theta_dot)`, shape `(2,)`.
        n_steps: Number of steps to simulate forward.
        dt: Integration time step.

    Returns:
        States `state0, state_1, ..., state_{n_steps}`, shape
        `(n_steps + 1, 2)`.
    """

    def step(s, _):
        """One scan step: apply the passive pendulum dynamics once."""
        s_next = pendulum_dynamics(s, dt)
        return s_next, s_next

    _, states = jax.lax.scan(step, state0, xs=None, length=n_steps)
    return jnp.concatenate([state0[None, :], states], axis=0)  # (n_steps+1, 2)


def render_pendulum_frame(
    theta: Float[Array, ""], image_size: int, length_px: float = 12.0, sigma: float = 1.6
) -> Float[Array, "1 H W"]:
    """Render a single pendulum bob as a soft Gaussian blob on a blank
    background, as a stand-in for a real rendered video frame.

    Args:
        theta: Pendulum angle (scalar), measured from straight down.
        image_size: Height/width of the square output frame.
        length_px: Pendulum rod length in pixels (bob distance from center).
        sigma: Standard deviation (in pixels) of the rendered Gaussian blob.

    Returns:
        Rendered grayscale frame, shape `(1, image_size, image_size)`,
        values in `[0, 1]`.
    """
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
) -> tuple[Float[Array, "T 1 H W"], Float[Array, "T 2"]]:
    """Sample a random initial condition, roll out the passive pendulum,
    and render each state as a frame.

    Args:
        key: PRNG key used to sample the initial `(theta, theta_dot)`.
        clip_len: Number of frames `T` to generate.
        image_size: Height/width of each square rendered frame.
        dt: Integration time step between consecutive frames.
        theta0_range: Initial `theta` is sampled uniformly from
            `[-theta0_range, theta0_range]`.

    Returns:
        A tuple `(clip, states)`:
          - `clip`: rendered frames, shape `(T, 1, H, W)`.
          - `states`: ground-truth `(theta, theta_dot)` per frame, shape
            `(T, 2)`.
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
) -> tuple[Float[Array, "B T 1 H W"], Float[Array, "B T 2"]]:
    """Sample a batch of independent pendulum clips.

    Args:
        key: PRNG key, split internally across the batch.
        batch_size: Number of independent clips to sample, `B`.
        clip_len: Number of frames `T` per clip.
        image_size: Height/width of each square rendered frame.
        dt: Integration time step between consecutive frames.

    Returns:
        A tuple `(clips, states)`:
          - `clips`: rendered frames, shape `(B, T, 1, H, W)`.
          - `states`: ground-truth `(theta, theta_dot)` per frame, shape
            `(B, T, 2)`.
    """
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
) -> Float[Array, "2"]:
    """Semi-implicit Euler step for the torque-driven, mode-damped pendulum:
    `theta'' = -(g/l) sin(theta) - damping(mode) * theta_dot + torque / (m l^2)`.

    Args:
        state: Current `(theta, theta_dot)`, shape `(2,)`.
        discrete_mode: Damping-mode index (scalar int), `0` (undamped) or
            `1` (damped); indexes `_DAMPING_BY_MODE`.
        continuous_torque: Applied torque (scalar).
        dt: Integration time step.
        g: Gravitational acceleration.
        length: Pendulum rod length.
        mass: Pendulum bob mass.

    Returns:
        Next state `(theta, theta_dot)`, shape `(2,)`.
    """
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
) -> Float[Array, "n_steps_plus_1 2"]:
    """Roll out the actuated pendulum dynamics under a given action sequence.

    Args:
        state0: Initial `(theta, theta_dot)`, shape `(2,)`.
        discrete_modes: Per-step damping-mode indices, shape `(n_steps,)`.
        continuous_torques: Per-step applied torques, shape `(n_steps,)`.
        dt: Integration time step.

    Returns:
        States `state0, state_1, ..., state_{n_steps}`, shape
        `(n_steps + 1, 2)`.
    """

    def step(s, a):
        """One scan step: apply the actuated dynamics with that step's
        (mode, torque) action."""
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
) -> tuple[Float[Array, "n_steps_plus_1 2"], Int[Array, "n_steps"], Float[Array, "n_steps"]]:
    """Sample a random initial condition and a random action sequence
    (piecewise-constant discrete mode per episode, smoothly-varying
    continuous torque), and roll out the true actuated dynamics.

    Args:
        key: PRNG key used to sample the initial state, the discrete mode,
            and the continuous torque waveform.
        n_steps: Number of action steps (episode has `n_steps + 1` states).
        dt: Integration time step between consecutive states.
        theta0_range: Initial `theta` is sampled uniformly from
            `[-theta0_range, theta0_range]`.
        torque_scale: Upper bound on the sampled torque waveform's
            amplitude.

    Returns:
        A tuple `(states, discrete_modes, continuous_torques)`:
          - `states`: shape `(n_steps + 1, 2)`.
          - `discrete_modes`: `int32` in `{0, 1}`, shape `(n_steps,)`,
            constant for the whole episode.
          - `continuous_torques`: `float32`, shape `(n_steps,)`, roughly in
            `[-torque_scale, torque_scale]` (a smooth, random-phase/
            frequency/amplitude sinusoid rather than i.i.d. noise, so it's
            a plausible control signal).
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
) -> tuple[Float[Array, "B n_steps_plus_1 2"], Int[Array, "B n_steps"], Float[Array, "B n_steps"]]:
    """Sample a batch of independent actuated-pendulum episodes.

    Args:
        key: PRNG key, split internally across the batch.
        batch_size: Number of independent episodes to sample, `B`.
        n_steps: Number of action steps per episode.
        dt: Integration time step between consecutive states.
        torque_scale: Upper bound on each episode's sampled torque
            amplitude.

    Returns:
        A tuple `(states, discrete_modes, continuous_torques)`:
          - `states`: shape `(B, n_steps + 1, 2)`.
          - `discrete_modes`: shape `(B, n_steps)`.
          - `continuous_torques`: shape `(B, n_steps)`.
    """
    keys = jax.random.split(key, batch_size)
    states, discrete_modes, continuous_torques = jax.vmap(
        lambda k: sample_actuated_pendulum_episode(k, n_steps, dt, torque_scale=torque_scale)
    )(keys)
    return states, discrete_modes, continuous_torques
