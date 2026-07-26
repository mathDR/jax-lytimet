"""
Shared fixtures for the LyTimeT test suite.

Everything here is deliberately tiny (small dim/depth/dz, short clips) so
the *default* test run (excluding tests marked `slow`) completes in well
under a minute on a CPU-only CI runner. `slow`-marked tests do a few dozen
real training steps to check loss goes down / coverage is sane; run them
with `pytest -m slow` or as part of a separate, longer CI job.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from lytimet.model import LyTimeT, LyTimeTConfig
from lytimet.data import make_pendulum_batch


@pytest.fixture(scope="session")
def rng_key():
    return jax.random.PRNGKey(0)


def _tiny_cfg(**overrides):
    base = dict(
        image_size=16,
        patch_size=4,
        in_channels=1,
        dim=16,
        depth=1,
        num_heads=2,
        dz=4,
        clip_len=5,
        transition_hidden=16,
        transition_depth=1,
        dynamics_type="residual_mlp",
    )
    base.update(overrides)
    return LyTimeTConfig(**base)


@pytest.fixture
def tiny_cfg():
    return _tiny_cfg()


@pytest.fixture
def tiny_ode_cfg():
    return _tiny_cfg(dynamics_type="neural_ode", ode_solver_steps=2, ode_dt=1.0)


@pytest.fixture(params=["residual_mlp", "neural_ode"])
def tiny_cfg_both_dynamics(request):
    if request.param == "neural_ode":
        return _tiny_cfg(dynamics_type="neural_ode", ode_solver_steps=2, ode_dt=1.0)
    return _tiny_cfg(dynamics_type="residual_mlp")


@pytest.fixture
def tiny_model(tiny_cfg, rng_key):
    return LyTimeT(tiny_cfg, key=rng_key)


@pytest.fixture
def tiny_model_any_dynamics(tiny_cfg_both_dynamics, rng_key):
    return LyTimeT(tiny_cfg_both_dynamics, key=rng_key)


@pytest.fixture
def dummy_clip(tiny_cfg, rng_key):
    return jax.random.uniform(
        rng_key, (tiny_cfg.clip_len, tiny_cfg.in_channels, tiny_cfg.image_size, tiny_cfg.image_size)
    )


@pytest.fixture
def pendulum_batch(tiny_cfg, rng_key):
    clips, states = make_pendulum_batch(rng_key, 4, tiny_cfg.clip_len, tiny_cfg.image_size)
    return clips, states
