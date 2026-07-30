"""
Plotting utilities for LyTimeT.

Visualizes what actually matters for judging "did the model learn good
rollouts": (1) predicted vs. ground-truth frames over an unrolled horizon,
(2) how per-step rollout error grows with horizon (and whether Phase 2's
Lyapunov regularization tames that growth), (3) how well the selected
z_tilde dimensions track the true state variables, and (4) the Phase 1 /
Phase 2 training loss curves.

All functions take/return a matplotlib Figure and optionally save it, so
they compose with `plt.show()`, artifact saving, or further tweaking.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; irrelevant if a display is present

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .model import LyTimeT

sns.set_theme(style="whitegrid", context="talk", palette="deep")


# --------------------------------------------------------------------------
# 1. Training curves
# --------------------------------------------------------------------------
def plot_training_curves(
    hist1: list[float],
    hist2: Optional[list[float]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Line plot of Phase 1 (and optionally Phase 2) loss vs. training step.

    Args:
        hist1: Per-step Phase-1 losses (e.g. from `train.train_phase1`).
        hist2: Optional per-step Phase-2 losses, plotted continuing from
            where `hist1` ends, with a vertical marker at the transition.
        save_path: If given, save the figure as a PNG to this path.

    Returns:
        The matplotlib `Figure`.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps1 = np.arange(len(hist1))
    sns.lineplot(x=steps1, y=hist1, ax=ax, label="Phase 1 (rec + pred)", linewidth=2)

    if hist2 is not None:
        steps2 = np.arange(len(hist2)) + len(hist1)
        sns.lineplot(
            x=steps2, y=hist2, ax=ax,
            label="Phase 2 (rec + pred + lyap)", linewidth=2,
        )
        ax.axvline(len(hist1), color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(len(hist1), max(hist1) * 0.95, "  Phase 2 starts",
                color="gray", va="top", fontsize=10)

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("LyTimeT training loss")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 2. Rollout frame grid: ground truth vs. predicted, over a horizon
# --------------------------------------------------------------------------
def plot_rollout_frames(
    model: LyTimeT,
    clip: Float[Array, "T C H W"],
    k_steps: int,
    save_path: Optional[str] = None,
    max_cols: int = 8,
) -> Figure:
    """Encode the first frame of `clip`, roll the transition model forward
    `k_steps`, decode each predicted latent, and show ground-truth vs.
    predicted frames side by side (one column per rollout step).

    Args:
        model: Trained `LyTimeT` model.
        clip: A single clip of shape `(T, C, H, W)`.
        k_steps: Number of rollout steps to display.
        save_path: If given, save the figure as a PNG to this path.
        max_cols: Maximum number of columns (rollout steps) to display.

    Returns:
        The matplotlib `Figure`.
    """
    z, _ = model.encode(clip)
    x_future_hat, _ = model.forecast(z[0], k_steps)
    x_future_true = clip[1 : 1 + k_steps]

    k = x_future_hat.shape[0]
    n_cols = min(k, max_cols)
    fig, axes = plt.subplots(2, n_cols, figsize=(1.6 * n_cols, 3.4))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_cols):
        gt = np.asarray(x_future_true[i, 0])
        pred = np.asarray(x_future_hat[i, 0])
        axes[0, i].imshow(gt, cmap="magma", vmin=0, vmax=1)
        axes[0, i].set_title(f"t+{i+1}", fontsize=11)
        axes[0, i].axis("off")
        axes[1, i].imshow(pred, cmap="magma", vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Ground truth", fontsize=11)
    axes[1, 0].set_ylabel("Predicted", fontsize=11)
    for row, label in zip(axes[:, 0], ["Ground truth", "Predicted"]):
        row.axis("on")
        row.set_xticks([])
        row.set_yticks([])
        row.set_ylabel(label, fontsize=12)
        for spine in row.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Rollout: {k_steps}-step unrolled forecast", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 3. Rollout error vs. horizon (shows whether error accumulates/contracts)
# --------------------------------------------------------------------------
def plot_rollout_error_vs_horizon(
    models: dict[str, LyTimeT],
    clips: Float[Array, "B T C H W"],
    max_k: int,
    save_path: Optional[str] = None,
) -> Figure:
    """For each named model in `models` (e.g. {"before Lyapunov": m1,
    "after Lyapunov": m2}), roll out `max_k` steps on a batch of clips and
    plot mean per-step pixel MSE vs. horizon with a bootstrap CI band —
    this is the plot that shows whether error accumulation (roll-out
    instability, the paper's central concern) is being controlled.

    Args:
        models: Mapping from a display label to a trained `LyTimeT` model.
        clips: Batch of clips, shape `(B, T, C, H, W)`, shared across all
            models being compared.
        max_k: Maximum rollout horizon to evaluate.
        save_path: If given, save the figure as a PNG to this path.

    Returns:
        The matplotlib `Figure`.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for label, model in models.items():
        def per_clip_errors(clip):
            """Per-step pixel MSE for one clip, rolled out from its first
            encoded frame."""
            z, _ = model.encode(clip)
            k = min(max_k, clip.shape[0] - 1)
            x_hat, _ = model.forecast(z[0], k)
            x_true = clip[1 : 1 + k]
            return jnp.mean((x_hat - x_true) ** 2, axis=(1, 2, 3))  # (k,)

        errors = jax.vmap(per_clip_errors)(clips)  # (B, k)
        errors = np.asarray(errors)
        b, k = errors.shape
        horizon = np.tile(np.arange(1, k + 1), b)
        err_flat = errors.reshape(-1)
        sns.lineplot(x=horizon, y=err_flat, ax=ax, label=label,
                     marker="o", errorbar=("ci", 95))

    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("Per-frame MSE")
    ax.set_title("Roll-out error accumulation")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 4. Latent-vs-ground-truth trajectory alignment (interpretability check)
# --------------------------------------------------------------------------
def plot_latent_vs_truth(
    z_tilde: np.ndarray,
    states: np.ndarray,
    dt: float = 0.1,
    save_path: Optional[str] = None,
) -> Figure:
    """For each extracted z_tilde dimension, overlay its (z-scored) trace
    against each (z-scored) ground-truth state variable it best aligns
    with, over time -- a visual check of what Table 1's MI/AMSE numbers
    are quantifying.

    Args:
        z_tilde: Selected latent dimensions over time, shape `(T, n_select)`.
        states: Ground-truth state variables over time, shape `(T, Ds)`.
        dt: Time step between consecutive rows, used for the x-axis.
        save_path: If given, save the figure as a PNG to this path.

    Returns:
        The matplotlib `Figure`.
    """
    z_tilde = np.asarray(z_tilde)
    states = np.asarray(states)
    t = np.arange(z_tilde.shape[0]) * dt

    def zscore(x):
        """Per-column z-score normalization: `(x - mean) / (std + eps)`."""
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    z_n = zscore(z_tilde)
    s_n = zscore(states)

    n_dims = z_tilde.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(8, 2.6 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    state_names = [f"s_{j}" for j in range(states.shape[1])]
    for i in range(n_dims):
        # best-aligned ground truth dim for this latent dim, by |correlation|
        corrs = [abs(np.corrcoef(z_n[:, i], s_n[:, j])[0, 1]) for j in range(s_n.shape[1])]
        best_j = int(np.argmax(corrs))
        axes[i].plot(t, z_n[:, i], label=f"z̃_{i} (learned)", linewidth=2)
        axes[i].plot(
            t, s_n[:, best_j], label=f"{state_names[best_j]} (ground truth)",
            linestyle="--", linewidth=2,
        )
        axes[i].set_ylabel("z-scored value")
        axes[i].set_title(f"z̃ dim {i}  ↔  {state_names[best_j]}  "
                           f"(|corr|={corrs[best_j]:.2f})", fontsize=12)
        axes[i].legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Extracted variables vs. ground-truth state", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 5. Conformal prediction bands + coverage diagnostic
# --------------------------------------------------------------------------
def plot_conformal_band(
    z_true: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dims: Optional[list[int]] = None,
    dt: float = 0.1,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot the true latent trajectory against the ensemble mean and its
    calibrated conformal interval, for one or more latent dimensions.

    `z_true`, `mean`, `lower`, `upper` are all (K, Dz) (z_true should be the
    K ground-truth future states, e.g. z_seq[1:1+K]); `dims` selects which
    latent dimensions to plot (defaults to the first 3).

    Args:
        z_true: Ground-truth future latent states, shape `(K, Dz)`.
        mean: Particle-ensemble mean prediction, shape `(K, Dz)`.
        lower: Calibrated lower interval bound, shape `(K, Dz)`.
        upper: Calibrated upper interval bound, shape `(K, Dz)`.
        dims: Which latent dimensions to plot; defaults to the first 3.
        dt: Time step between consecutive rows, used for the x-axis.
        save_path: If given, save the figure as a PNG to this path.

    Returns:
        The matplotlib `Figure`.
    """
    dz = mean.shape[-1]
    dims = dims if dims is not None else list(range(min(3, dz)))
    k = mean.shape[0]
    t = np.arange(1, k + 1) * dt

    fig, axes = plt.subplots(len(dims), 1, figsize=(7.5, 2.6 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        ax.fill_between(t, lower[:, d], upper[:, d], alpha=0.25,
                         label="Conformal interval", color=sns.color_palette()[0])
        ax.plot(t, mean[:, d], label="Ensemble mean", linewidth=2,
                color=sns.color_palette()[0])
        ax.plot(t, z_true[:, d], label="True", linestyle="--", linewidth=2,
                color=sns.color_palette()[1])
        ax.set_ylabel(f"z_{d}")
        ax.legend(fontsize=9, loc="upper left")

    axes[-1].set_xlabel("Time")
    fig.suptitle("Calibrated rollout intervals (particle ensemble + conformal)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_coverage_diagnostic(
    coverage: np.ndarray,
    target_coverage: float,
    save_path: Optional[str] = None,
) -> Figure:
    """Bar plot of empirical per-step coverage (averaged over latent dims)
    against the nominal target (1 - alpha) -- a sanity check that the
    conformal calibration is actually achieving its guarantee.

    Args:
        coverage: Empirical per-(step, dim) coverage, shape `(K, Dz)` (or
            `(K,)` if already dimension-averaged).
        target_coverage: The nominal target coverage `1 - alpha`.
        save_path: If given, save the figure as a PNG to this path.

    Returns:
        The matplotlib `Figure`.
    """
    coverage = np.asarray(coverage)
    per_step = coverage.mean(axis=-1) if coverage.ndim > 1 else coverage
    steps = np.arange(1, len(per_step) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(x=steps, y=per_step, ax=ax, color=sns.color_palette()[0])
    ax.axhline(target_coverage, color="crimson", linestyle="--", linewidth=2,
               label=f"Target coverage ({target_coverage:.0%})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal coverage diagnostic")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 6. Convenience: run everything and save a folder of figures
# --------------------------------------------------------------------------
def make_all_plots(
    model_before_phase2: LyTimeT,
    model_after_phase2: LyTimeT,
    hist1: list[float],
    hist2: list[float],
    eval_clips: Float[Array, "B T C H W"],
    eval_states: Float[Array, "B T Ds"],
    select_idx: Int[Array, "n_select"],
    k_steps: int,
    out_dir: str = "plots",
) -> dict[str, Figure]:
    """Generate the full set of diagnostic plots and save them as PNGs in
    `out_dir`.

    Args:
        model_before_phase2: Model checkpoint after Phase 1 only.
        model_after_phase2: Model checkpoint after Phase 2 (Lyapunov
            fine-tuning).
        hist1: Per-step Phase-1 losses.
        hist2: Per-step Phase-2 losses.
        eval_clips: Held-out clips for evaluation, shape `(B, T, C, H, W)`.
        eval_states: Ground-truth states for the same clips, shape
            `(B, T, Ds)`.
        select_idx: Indices of the selected interpretable latent
            dimensions, shape `(n_select,)`.
        k_steps: Rollout horizon used for the rollout-related plots.
        out_dir: Directory to save the PNGs into (created if missing).

    Returns:
        Dict mapping plot name (`"loss_curves"`, `"rollout_frames"`,
        `"rollout_error_vs_horizon"`, `"latent_vs_truth"`) to its
        matplotlib `Figure`.
    """
    os.makedirs(out_dir, exist_ok=True)
    figs = {}

    figs["loss_curves"] = plot_training_curves(
        hist1, hist2, save_path=os.path.join(out_dir, "loss_curves.png")
    )

    figs["rollout_frames"] = plot_rollout_frames(
        model_after_phase2, eval_clips[0], k_steps,
        save_path=os.path.join(out_dir, "rollout_frames.png"),
    )

    figs["rollout_error_vs_horizon"] = plot_rollout_error_vs_horizon(
        {"Phase 1 only": model_before_phase2, "Phase 1 + Lyapunov (Phase 2)": model_after_phase2},
        eval_clips,
        max_k=k_steps,
        save_path=os.path.join(out_dir, "rollout_error_vs_horizon.png"),
    )

    def encode_only(clip):
        """Encode one clip, discarding the patch tokens."""
        z, _ = model_after_phase2.encode(clip)
        return z

    z_all = jax.vmap(encode_only)(eval_clips)  # (B, T, Dz)
    z_tilde = np.asarray(z_all[0])[:, np.asarray(select_idx)]
    figs["latent_vs_truth"] = plot_latent_vs_truth(
        z_tilde, np.asarray(eval_states[0]),
        save_path=os.path.join(out_dir, "latent_vs_truth.png"),
    )

    return figs
