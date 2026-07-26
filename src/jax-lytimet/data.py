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
from jaxtyping import Array, Float, PRNGKeyArray


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
