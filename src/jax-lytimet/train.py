"""
Training loops for LyTimeT's two phases, using optax.

Phase 1: train encoder/decoder/transition end-to-end with
         L_phase1 = L_rec + lambda_pred * L_pred (+ data augmentation,
         omitted here for brevity — see `augment.py`-style hook below).

Phase 2: (a) linear-probe the frozen (or lightly fine-tuned) latents to
         rank and select the interpretable dimensions z_tilde, then
         (b) fine-tune the transition model f_theta and the Lyapunov
         matrix W with L = L_phase1 + lambda_lyap * L_lyap.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from .model import LyTimeT, LyTimeTConfig
from .losses import phase1_clip_loss, phase2_clip_loss
from .probe import rank_and_select_dimensions, fit_linear_probe, amse


@dataclasses.dataclass
class Phase1Config:
    lr: float = 3e-4
    k_steps: int = 4
    lambda_pred: float = 1.0
    num_steps: int = 200
    batch_size: int = 8
    grad_clip: float = 1.0


@dataclasses.dataclass
class Phase2Config:
    lr: float = 1e-4
    k_steps: int = 4
    lambda_pred: float = 1.0
    lambda_lyap: float = 0.1
    num_steps: int = 100
    batch_size: int = 8
    n_select: int = 4
    grad_clip: float = 1.0


def _batched_phase1_loss(model, clips, k_steps, lambda_pred):
    def single(clip):
        loss, aux = phase1_clip_loss(model, clip, k_steps, lambda_pred)
        return loss, aux

    losses, auxs = jax.vmap(single)(clips)
    aux_mean = jax.tree_util.tree_map(jnp.mean, auxs)
    return jnp.mean(losses), aux_mean


@eqx.filter_jit
def phase1_train_step(model, opt_state, clips, optimizer, k_steps, lambda_pred):
    loss_fn = lambda m: _batched_phase1_loss(m, clips, k_steps, lambda_pred)
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


def train_phase1(
    model: LyTimeT,
    data_fn: Callable[[PRNGKeyArray, int], Float[Array, "B T C H W"]],
    cfg: Phase1Config,
    key: PRNGKeyArray,
    log_every: int = 20,
):
    """`data_fn(key, batch_size) -> clips` should return a fresh batch of
    training clips of shape (B, T, C, H, W) each step (e.g. wrapping
    `data.make_pendulum_batch`, discarding the ground-truth states, and
    applying nuisance augmentation — background swap / occlusion / jitter —
    to encourage distraction-robust representations, per Sec. 2.1)."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(cfg.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history = []
    for step in range(cfg.num_steps):
        key, k_data = jax.random.split(key)
        clips = data_fn(k_data, cfg.batch_size)
        model, opt_state, loss, aux = phase1_train_step(
            model, opt_state, clips, optimizer, cfg.k_steps, cfg.lambda_pred
        )
        history.append(float(loss))
        if step % log_every == 0 or step == cfg.num_steps - 1:
            print(
                f"[phase1] step {step:4d}  loss={float(loss):.5f}  "
                f"rec={float(aux['l_rec']):.5f}  pred={float(aux['l_pred']):.5f}"
            )
    return model, history


def probe_and_select(
    model: LyTimeT,
    clips: Float[Array, "B T C H W"],
    states: Float[Array, "B T Ds"],
    n_select: int,
):
    """Phase 2, Step 1: encode a batch of clips, flatten across batch/time,
    fit a linear probe against ground-truth states, rank latent dimensions
    by best-aligned R^2, and select the top `n_select` as z_tilde."""

    def encode_only(clip):
        z, _ = model.encode(clip)
        return z

    z_all = jax.vmap(encode_only)(clips)  # (B, T, Dz)
    b, t, dz = z_all.shape
    ds = states.shape[-1]
    z_flat = z_all.reshape(b * t, dz)
    s_flat = states.reshape(b * t, ds)

    select_idx, scores = rank_and_select_dimensions(z_flat, s_flat, n_select)
    w_probe = fit_linear_probe(z_flat[:, select_idx], s_flat)
    err = amse(z_flat[:, select_idx], s_flat, w_probe)
    print(f"[phase2] selected dims={list(map(int, select_idx))}  "
          f"scores={[round(float(x), 3) for x in scores]}  AMSE={float(err):.5f}")
    return select_idx, w_probe, float(err)


@eqx.filter_jit
def phase2_train_step(
    model, w_lyap, opt_state, clips, optimizer, k_steps, lambda_pred, lambda_lyap, select_idx
):
    def loss_fn(m, w):
        def single(clip):
            return phase2_clip_loss(
                m, w, clip, k_steps, lambda_pred, lambda_lyap, select_idx
            )

        losses, auxs = jax.vmap(single)(clips)
        aux_mean = jax.tree_util.tree_map(jnp.mean, auxs)
        return jnp.mean(losses), aux_mean

    params = (model, w_lyap)
    (loss, aux), grads = eqx.filter_value_and_grad(
        lambda p: loss_fn(p[0], p[1]), has_aux=True
    )(params)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(params, eqx.is_array)
    )
    params = eqx.apply_updates(params, updates)
    model, w_lyap = params
    return model, w_lyap, opt_state, loss, aux


def train_phase2(
    model: LyTimeT,
    data_fn: Callable[[PRNGKeyArray, int], tuple],
    cfg: Phase2Config,
    key: PRNGKeyArray,
    log_every: int = 20,
):
    """`data_fn(key, batch_size) -> (clips, states)` returns clips together
    with ground-truth state variables used for probing/ranking."""
    key, k_probe = jax.random.split(key)
    probe_clips, probe_states = data_fn(k_probe, cfg.batch_size * 2)
    select_idx, w_probe, probe_amse = probe_and_select(
        model, probe_clips, probe_states, cfg.n_select
    )

    dz = model.cfg.dz
    dv = cfg.n_select  # Lyapunov map W: R^{n_select} -> R^{n_select}
    w_lyap = jnp.eye(cfg.n_select) + 0.01 * jax.random.normal(
        jax.random.PRNGKey(42), (dv, cfg.n_select)
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(cfg.lr),
    )
    params = (model, w_lyap)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    history = []
    for step in range(cfg.num_steps):
        key, k_data = jax.random.split(key)
        clips, _states = data_fn(k_data, cfg.batch_size)
        model, w_lyap, opt_state, loss, aux = phase2_train_step(
            model, w_lyap, opt_state, clips, optimizer,
            cfg.k_steps, cfg.lambda_pred, cfg.lambda_lyap, select_idx,
        )
        history.append(float(loss))
        if step % log_every == 0 or step == cfg.num_steps - 1:
            print(
                f"[phase2] step {step:4d}  loss={float(loss):.5f}  "
                f"rec={float(aux['l_rec']):.5f}  pred={float(aux['l_pred']):.5f}  "
                f"lyap={float(aux['l_lyap']):.5f}"
            )
    return model, w_lyap, select_idx, history
