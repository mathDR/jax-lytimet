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

import matplotlib

matplotlib.use("Agg")  # headless-safe; irrelevant if a display is present

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import jax
import jax.numpy as jnp

sns.set_theme(style="whitegrid", context="talk", palette="deep")


# --------------------------------------------------------------------------
# 1. Training curves
# --------------------------------------------------------------------------
def plot_training_curves(hist1, hist2=None, save_path: str | None = None):
    """Line plot of Phase 1 (and optionally Phase 2) loss vs. training step."""
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
    model,
    clip: jnp.ndarray,
    k_steps: int,
    save_path: str | None = None,
    max_cols: int = 8,
):
    """Encode the first frame of `clip`, roll the transition model forward
    `k_steps`, decode each predicted latent, and show ground-truth vs.
    predicted frames side by side (one column per rollout step)."""
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
    models: dict,
    clips: jnp.ndarray,
    max_k: int,
    save_path: str | None = None,
):
    """For each named model in `models` (e.g. {"before Lyapunov": m1,
    "after Lyapunov": m2}), roll out `max_k` steps on a batch of clips and
    plot mean per-step pixel MSE vs. horizon with a bootstrap CI band —
    this is the plot that shows whether error accumulation (roll-out
    instability, the paper's central concern) is being controlled."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for label, model in models.items():
        def per_clip_errors(clip):
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
    save_path: str | None = None,
):
    """For each extracted z_tilde dimension, overlay its (z-scored) trace
    against each (z-scored) ground-truth state variable it best aligns
    with, over time -- a visual check of what Table 1's MI/AMSE numbers
    are quantifying."""
    z_tilde = np.asarray(z_tilde)
    states = np.asarray(states)
    t = np.arange(z_tilde.shape[0]) * dt

    def zscore(x):
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
# 5. Convenience: run everything and save a folder of figures
# --------------------------------------------------------------------------
def make_all_plots(
    model_before_phase2,
    model_after_phase2,
    hist1,
    hist2,
    eval_clips: jnp.ndarray,
    eval_states: jnp.ndarray,
    select_idx,
    k_steps: int,
    out_dir: str = "plots",
):
    """Generate the full set of diagnostic plots and save them as PNGs in
    `out_dir`. Returns the dict of {name: matplotlib Figure}."""
    import os

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
        z, _ = model_after_phase2.encode(clip)
        return z

    z_all = jax.vmap(encode_only)(eval_clips)  # (B, T, Dz)
    z_tilde = np.asarray(z_all[0])[:, np.asarray(select_idx)]
    figs["latent_vs_truth"] = plot_latent_vs_truth(
        z_tilde, np.asarray(eval_states[0]),
        save_path=os.path.join(out_dir, "latent_vs_truth.png"),
    )

    return figs
