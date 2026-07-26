import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import pytest

from lytimet.model import LyTimeT
from lytimet.data import make_pendulum_batch
from lytimet.train import (
    Phase1Config,
    Phase2Config,
    train_phase1,
    train_phase2,
    phase1_train_step,
    phase2_train_step,
    probe_and_select,
)


def _data_fn_clips(cfg):
    def fn(key, batch_size):
        clips, _ = make_pendulum_batch(key, batch_size, cfg.clip_len, cfg.image_size)
        return clips

    return fn


def _data_fn_clips_and_states(cfg):
    def fn(key, batch_size):
        return make_pendulum_batch(key, batch_size, cfg.clip_len, cfg.image_size)

    return fn


def test_phase1_train_step_runs_and_updates_params(tiny_model, rng_key):
    cfg = tiny_model.cfg
    data_fn = _data_fn_clips(cfg)
    clips = data_fn(rng_key, 3)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(tiny_model, eqx.is_array))

    new_model, new_opt_state, loss, aux = phase1_train_step(
        tiny_model, opt_state, clips, optimizer, k_steps=2, lambda_pred=1.0
    )
    assert jnp.isfinite(loss)

    # parameters should actually have moved
    old_leaves = jax.tree_util.tree_leaves(eqx.filter(tiny_model, eqx.is_array))
    new_leaves = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
    changed = any(
        not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves)
    )
    assert changed


@pytest.mark.slow
def test_train_phase1_loss_decreases(tiny_model, rng_key):
    cfg = tiny_model.cfg
    data_fn = _data_fn_clips(cfg)
    p1_cfg = Phase1Config(lr=3e-3, k_steps=2, num_steps=15, batch_size=4)
    model, history = train_phase1(tiny_model, data_fn, p1_cfg, rng_key, log_every=1000)

    assert len(history) == p1_cfg.num_steps
    assert all(jnp.isfinite(jnp.asarray(h)) for h in history)
    # loose monotonicity check: average of the second half should be lower
    # than the average of the first half
    first_half = sum(history[: len(history) // 2])
    second_half = sum(history[len(history) // 2 :])
    assert second_half < first_half


def test_probe_and_select_returns_valid_indices(tiny_model, pendulum_batch):
    clips, states = pendulum_batch
    select_idx, w_probe, err = probe_and_select(tiny_model, clips, states, n_select=2)
    assert select_idx.shape == (2,)
    assert jnp.all(select_idx >= 0) and jnp.all(select_idx < tiny_model.cfg.dz)
    assert jnp.isfinite(err)


@pytest.mark.slow
def test_train_phase2_runs_end_to_end(tiny_model, rng_key):
    cfg = tiny_model.cfg
    data_fn = _data_fn_clips_and_states(cfg)
    p2_cfg = Phase2Config(
        lr=1e-3, k_steps=2, lambda_lyap=0.1, num_steps=5, batch_size=3, n_select=2
    )
    model, w_lyap, select_idx, history = train_phase2(
        tiny_model, data_fn, p2_cfg, rng_key, log_every=1000
    )
    assert len(history) == p2_cfg.num_steps
    assert all(jnp.isfinite(jnp.asarray(h)) for h in history)
    assert select_idx.shape == (2,)
    assert w_lyap.shape == (2, 2)


@pytest.mark.slow
def test_train_phase2_with_neural_ode_and_continuous_lyapunov(tiny_ode_cfg, rng_key):
    model = LyTimeT(tiny_ode_cfg, key=rng_key)
    data_fn = _data_fn_clips_and_states(tiny_ode_cfg)
    p2_cfg = Phase2Config(
        lr=1e-3, k_steps=2, lambda_lyap=0.1, num_steps=5, batch_size=3,
        n_select=2, use_continuous_lyapunov=True,
    )
    model, w_lyap, select_idx, history = train_phase2(
        model, data_fn, p2_cfg, rng_key, log_every=1000
    )
    assert all(jnp.isfinite(jnp.asarray(h)) for h in history)
