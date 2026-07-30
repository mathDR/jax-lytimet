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
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jax_lytimet.model import LyTimeT, LyTimeTConfig
from jax_lytimet.losses import phase1_clip_loss, phase2_clip_loss
from jax_lytimet.probe import rank_and_select_dimensions, fit_linear_probe, amse


@dataclasses.dataclass
class Phase1Config:
    """Hyperparameters for `train_phase1`.

    Attributes:
        lr: Adam learning rate.
        k_steps: Number of unrolled forecast steps `K` used in `L_pred`.
        lambda_pred: Weight on the K-step prediction loss `L_pred`.
        lambda_dyn: Weight on the optional direct latent-space dynamics-
            matching loss (see `losses.latent_dynamics_matching_loss`); `0.0`
            disables it.
        num_steps: Total number of optimizer steps.
        batch_size: Number of clips per training step.
        grad_clip: Global-norm gradient clipping threshold.
    """

    lr: float = 3e-4
    k_steps: int = 4
    lambda_pred: float = 1.0
    lambda_dyn: float = 0.0
    num_steps: int = 200
    batch_size: int = 8
    grad_clip: float = 1.0


@dataclasses.dataclass
class Phase2Config:
    """Hyperparameters for `train_phase2`.

    Attributes:
        lr: Adam learning rate.
        k_steps: Number of unrolled forecast steps `K` used in `L_pred`.
        lambda_pred: Weight on the K-step prediction loss `L_pred`.
        lambda_lyap: Weight on the Lyapunov stability loss `L_lyap`.
        lambda_dyn: Weight on the optional direct latent-space dynamics-
            matching loss; `0.0` disables it.
        num_steps: Total number of optimizer steps.
        batch_size: Number of clips per training step.
        n_select: Number of latent dimensions to select as the
            interpretable subset `z_tilde` during probing.
        grad_clip: Global-norm gradient clipping threshold.
        use_continuous_lyapunov: If `True`, use the continuous-time
            Lyapunov loss (requires `dynamics_type="neural_ode"`); if
            `False`, use the paper's discrete Lyapunov loss.
    """

    lr: float = 1e-4
    k_steps: int = 4
    lambda_pred: float = 1.0
    lambda_lyap: float = 0.1
    lambda_dyn: float = 0.0
    num_steps: int = 100
    batch_size: int = 8
    n_select: int = 4
    grad_clip: float = 1.0
    use_continuous_lyapunov: bool = False


def _batched_phase1_loss(
    model: LyTimeT,
    clips: Float[Array, "B T C H W"],
    k_steps: int,
    lambda_pred: float,
    lambda_dyn: float,
) -> tuple[Float[Array, ""], dict[str, Any]]:
    """Batched Phase-1 loss: `vmap`s `phase1_clip_loss` over a batch of
    clips and averages both the scalar loss and its auxiliary metrics.

    Args:
        model: The `LyTimeT` model.
        clips: Batch of clips, shape `(B, T, C, H, W)`.
        k_steps: Number of unrolled forecast steps.
        lambda_pred: Weight on the prediction loss.
        lambda_dyn: Weight on the latent dynamics-matching loss.

    Returns:
        A tuple `(loss, aux)`: the batch-mean scalar loss, and a dict of
        batch-mean auxiliary loss terms (see `phase1_clip_loss`).
    """

    def single(clip: Float[Array, "T C H W"]) -> tuple[Float[Array, ""], dict[str, Any]]:
        """Per-clip Phase-1 loss (see `losses.phase1_clip_loss`)."""
        loss, aux = phase1_clip_loss(model, clip, k_steps, lambda_pred, lambda_dyn)
        return loss, aux

    losses, auxs = jax.vmap(single)(clips)
    aux_mean = jax.tree_util.tree_map(jnp.mean, auxs)
    return jnp.mean(losses), aux_mean


@eqx.filter_jit
def phase1_train_step(
    model: LyTimeT,
    opt_state: optax.OptState,
    clips: Float[Array, "B T C H W"],
    optimizer: optax.GradientTransformation,
    k_steps: int,
    lambda_pred: float,
    lambda_dyn: float = 0.0,
) -> tuple[LyTimeT, optax.OptState, Float[Array, ""], dict[str, Any]]:
    """One Phase-1 optimizer step: compute the batched loss, its gradient
    w.r.t. `model`, and apply one optax update.

    Args:
        model: Current `LyTimeT` model.
        opt_state: Current optax optimizer state.
        clips: Batch of training clips, shape `(B, T, C, H, W)`.
        optimizer: The optax `GradientTransformation` in use.
        k_steps: Number of unrolled forecast steps.
        lambda_pred: Weight on the prediction loss.
        lambda_dyn: Weight on the latent dynamics-matching loss.

    Returns:
        A tuple `(model, opt_state, loss, aux)` with the updated model and
        optimizer state, and this step's scalar loss and auxiliary metrics.
    """
    loss_fn = lambda m: _batched_phase1_loss(m, clips, k_steps, lambda_pred, lambda_dyn)
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
) -> tuple[LyTimeT, list[float]]:
    """Run the Phase-1 training loop (representation learning + forecasting).

    `data_fn(key, batch_size) -> clips` should return a fresh batch of
    training clips of shape (B, T, C, H, W) each step (e.g. wrapping
    `data.make_pendulum_batch`, discarding the ground-truth states, and
    applying nuisance augmentation — background swap / occlusion / jitter —
    to encourage distraction-robust representations, per Sec. 2.1).

    Args:
        model: Initial `LyTimeT` model.
        data_fn: Callable returning a fresh batch of clips each step.
        cfg: Phase-1 hyperparameters.
        key: PRNG key, split internally for each step's data sample.
        log_every: Print progress every this many steps (and on the last
            step).

    Returns:
        A tuple `(model, history)`: the trained model, and the list of
        per-step scalar losses (length `cfg.num_steps`).
    """
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
            model, opt_state, clips, optimizer, cfg.k_steps, cfg.lambda_pred, cfg.lambda_dyn
        )
        history.append(float(loss))
        if step % log_every == 0 or step == cfg.num_steps - 1:
            dyn_str = f"  dyn={float(aux['l_dyn']):.5f}" if "l_dyn" in aux else ""
            print(
                f"[phase1] step {step:4d}  loss={float(loss):.5f}  "
                f"rec={float(aux['l_rec']):.5f}  pred={float(aux['l_pred']):.5f}{dyn_str}"
            )
    return model, history


def probe_and_select(
    model: LyTimeT,
    clips: Float[Array, "B T C H W"],
    states: Float[Array, "B T Ds"],
    n_select: int,
) -> tuple[Int[Array, "n_select"], Float[Array, "n_select Ds"], float]:
    """Phase 2, Step 1: encode a batch of clips, flatten across batch/time,
    fit a linear probe against ground-truth states, rank latent dimensions
    by best-aligned R^2, and select the top `n_select` as z_tilde.

    Args:
        model: Trained `LyTimeT` model (typically after Phase 1).
        clips: Batch of clips, shape `(B, T, C, H, W)`.
        states: Ground-truth state variables per frame, shape `(B, T, Ds)`.
        n_select: Number of latent dimensions to select.

    Returns:
        A tuple `(select_idx, w_probe, err)`:
          - `select_idx`: indices of the selected latent dimensions, shape
            `(n_select,)`.
          - `w_probe`: fitted linear-probe weights on the selected
            dimensions, shape `(n_select, Ds)`.
          - `err`: the probe's AMSE on the selected dimensions (Python
            float).
    """

    def encode_only(clip: Float[Array, "T C H W"]) -> Float[Array, "T Dz"]:
        """Encode one clip, discarding the patch tokens (unused for probing)."""
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
    model: LyTimeT,
    w_lyap: Float[Array, "Dv Ds"],
    opt_state: optax.OptState,
    clips: Float[Array, "B T C H W"],
    optimizer: optax.GradientTransformation,
    k_steps: int,
    lambda_pred: float,
    lambda_lyap: float,
    select_idx: Int[Array, "Ds"],
    use_continuous_lyapunov: bool = False,
    lambda_dyn: float = 0.0,
) -> tuple[LyTimeT, Float[Array, "Dv Ds"], optax.OptState, Float[Array, ""], dict[str, Any]]:
    """One Phase-2 optimizer step: jointly update `model` and the Lyapunov
    matrix `w_lyap` on the combined Phase-1 + Lyapunov (+ optional dynamics-
    matching) loss.

    Args:
        model: Current `LyTimeT` model.
        w_lyap: Current Lyapunov energy matrix, shape `(Dv, Ds)`.
        opt_state: Current optax optimizer state (for the joint
            `(model, w_lyap)` parameter tree).
        clips: Batch of training clips, shape `(B, T, C, H, W)`.
        optimizer: The optax `GradientTransformation` in use.
        k_steps: Number of unrolled forecast steps.
        lambda_pred: Weight on the prediction loss.
        lambda_lyap: Weight on the Lyapunov loss.
        select_idx: Indices of the selected interpretable latent
            dimensions, shape `(Ds,)`.
        use_continuous_lyapunov: Whether to use the continuous-time
            Lyapunov loss (requires a Neural ODE transition).
        lambda_dyn: Weight on the latent dynamics-matching loss.

    Returns:
        A tuple `(model, w_lyap, opt_state, loss, aux)` with the updated
        model, Lyapunov matrix, and optimizer state, and this step's scalar
        loss and auxiliary metrics.
    """
    def loss_fn(
        m: LyTimeT, w: Float[Array, "Dv Ds"]
    ) -> tuple[Float[Array, ""], dict[str, Any]]:
        """Batched Phase-2 loss for a given `(model, w_lyap)` pair."""

        def single(clip: Float[Array, "T C H W"]) -> tuple[Float[Array, ""], dict[str, Any]]:
            """Per-clip Phase-2 loss (see `losses.phase2_clip_loss`)."""
            return phase2_clip_loss(
                m, w, clip, k_steps, lambda_pred, lambda_lyap, select_idx,
                use_continuous_lyapunov, lambda_dyn,
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
    data_fn: Callable[[PRNGKeyArray, int], tuple[Float[Array, "B T C H W"], Float[Array, "B T Ds"]]],
    cfg: Phase2Config,
    key: PRNGKeyArray,
    log_every: int = 20,
) -> tuple[LyTimeT, Float[Array, "n_select n_select"], Int[Array, "n_select"], list[float]]:
    """Run the Phase-2 training loop: probe + select `z_tilde`, then jointly
    fine-tune the model and Lyapunov matrix.

    `data_fn(key, batch_size) -> (clips, states)` returns clips together
    with ground-truth state variables used for probing/ranking.

    Args:
        model: `LyTimeT` model, typically already trained via `train_phase1`.
        data_fn: Callable returning a fresh batch of `(clips, states)` each
            step (and for the initial probing batch).
        cfg: Phase-2 hyperparameters.
        key: PRNG key, split internally for the probing batch and each
            training step's data sample.
        log_every: Print progress every this many steps (and on the last
            step).

    Returns:
        A tuple `(model, w_lyap, select_idx, history)`:
          - `model`: the fine-tuned model.
          - `w_lyap`: the fine-tuned Lyapunov energy matrix, shape
            `(n_select, n_select)`.
          - `select_idx`: indices of the selected interpretable latent
            dimensions, shape `(n_select,)`.
          - `history`: list of per-step scalar losses (length
            `cfg.num_steps`).
    """
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
            cfg.use_continuous_lyapunov, cfg.lambda_dyn,
        )
        history.append(float(loss))
        if step % log_every == 0 or step == cfg.num_steps - 1:
            dyn_str = f"  dyn={float(aux['l_dyn']):.5f}" if "l_dyn" in aux else ""
            print(
                f"[phase2] step {step:4d}  loss={float(loss):.5f}  "
                f"rec={float(aux['l_rec']):.5f}  pred={float(aux['l_pred']):.5f}  "
                f"lyap={float(aux['l_lyap']):.5f}{dyn_str}"
            )
    return model, w_lyap, select_idx, history
