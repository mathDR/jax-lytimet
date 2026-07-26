import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lytimet.plot import (
    plot_training_curves,
    plot_rollout_frames,
    plot_rollout_error_vs_horizon,
    plot_latent_vs_truth,
    plot_conformal_band,
    plot_coverage_diagnostic,
    make_all_plots,
)


def _assert_valid_png(path):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_plot_training_curves_phase1_only(tmp_path):
    hist1 = [10.0, 8.0, 6.0, 4.0]
    out = str(tmp_path / "loss.png")
    fig = plot_training_curves(hist1, None, save_path=out)
    assert isinstance(fig, plt.Figure)
    _assert_valid_png(out)
    plt.close(fig)


def test_plot_training_curves_both_phases(tmp_path):
    hist1 = [10.0, 8.0, 6.0]
    hist2 = [5.0, 4.5, 4.0]
    out = str(tmp_path / "loss.png")
    fig = plot_training_curves(hist1, hist2, save_path=out)
    _assert_valid_png(out)
    plt.close(fig)


def test_plot_rollout_frames(tiny_model, dummy_clip, tmp_path):
    out = str(tmp_path / "rollout.png")
    fig = plot_rollout_frames(tiny_model, dummy_clip, k_steps=2, save_path=out)
    assert isinstance(fig, plt.Figure)
    _assert_valid_png(out)
    plt.close(fig)


def test_plot_rollout_error_vs_horizon(tiny_model, pendulum_batch, tmp_path):
    clips, _ = pendulum_batch
    out = str(tmp_path / "error.png")
    fig = plot_rollout_error_vs_horizon(
        {"model": tiny_model}, clips, max_k=2, save_path=out
    )
    _assert_valid_png(out)
    plt.close(fig)


def test_plot_latent_vs_truth(tmp_path):
    key = jax.random.PRNGKey(0)
    z_tilde = np.asarray(jax.random.normal(key, (10, 2)))
    states = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (10, 2)))
    out = str(tmp_path / "latent.png")
    fig = plot_latent_vs_truth(z_tilde, states, save_path=out)
    _assert_valid_png(out)
    plt.close(fig)


def test_plot_conformal_band(tmp_path):
    k, dz = 4, 3
    z_true = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (k, dz)))
    mean = z_true + 0.1
    lower = mean - 0.5
    upper = mean + 0.5
    out = str(tmp_path / "conformal.png")
    fig = plot_conformal_band(z_true, mean, lower, upper, save_path=out)
    _assert_valid_png(out)
    plt.close(fig)


def test_plot_coverage_diagnostic(tmp_path):
    coverage = np.array([[0.9, 0.92], [0.88, 0.91], [0.85, 0.89]])
    out = str(tmp_path / "coverage.png")
    fig = plot_coverage_diagnostic(coverage, target_coverage=0.9, save_path=out)
    _assert_valid_png(out)
    plt.close(fig)


@pytest.mark.slow
def test_make_all_plots_end_to_end(tiny_model, pendulum_batch, tmp_path):
    clips, states = pendulum_batch
    hist1 = [10.0, 8.0, 6.0]
    hist2 = [5.0, 4.5, 4.0]
    out_dir = str(tmp_path / "plots")
    figs = make_all_plots(
        model_before_phase2=tiny_model,
        model_after_phase2=tiny_model,
        hist1=hist1,
        hist2=hist2,
        eval_clips=clips,
        eval_states=states,
        select_idx=jnp.array([0, 1]),
        k_steps=2,
        out_dir=out_dir,
    )
    assert set(figs.keys()) == {
        "loss_curves", "rollout_frames", "rollout_error_vs_horizon", "latent_vs_truth",
    }
    for name in figs:
        _assert_valid_png(os.path.join(out_dir, f"{name}.png"))
        plt.close(figs[name])
