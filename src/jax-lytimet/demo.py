"""
End-to-end smoke test / demo of the LyTimeT implementation on a synthetic
single-pendulum video dataset (one of the paper's five synthetic
benchmarks). Run with:

    python -m lytimet.demo

This is intentionally small (few steps, small model) so it runs quickly on
CPU as a correctness check; scale up dim/depth/num_steps for real use.
"""
import jax

from lytimet.model import LyTimeT, LyTimeTConfig
from lytimet.data import make_pendulum_batch
from lytimet.train import Phase1Config, Phase2Config, train_phase1, train_phase2
from lytimet.plot import make_all_plots


def main():
    cfg = LyTimeTConfig(
        image_size=32,
        patch_size=4,
        in_channels=1,
        dim=64,
        depth=2,
        num_heads=4,
        dz=8,
        clip_len=8,
        transition_hidden=64,
        transition_depth=2,
    )
    key = jax.random.PRNGKey(0)
    k_model, k_phase1, k_phase2, k_eval = jax.random.split(key, 4)
    model = LyTimeT(cfg, key=k_model)

    def data_fn_clips(k, batch_size):
        clips, _states = make_pendulum_batch(k, batch_size, cfg.clip_len, cfg.image_size)
        return clips

    def data_fn_clips_and_states(k, batch_size):
        return make_pendulum_batch(k, batch_size, cfg.clip_len, cfg.image_size)

    print("=== Phase 1: representation + forecasting ===")
    p1_cfg = Phase1Config(lr=3e-4, k_steps=3, lambda_pred=1.0, num_steps=30, batch_size=4)
    model, hist1 = train_phase1(model, data_fn_clips, p1_cfg, k_phase1, log_every=5)
    model_after_phase1 = model  # eqx models are immutable pytrees, so this
                                 # reference is unaffected by Phase 2 training

    print("\n=== Phase 2: probing, ranking, Lyapunov fine-tuning ===")
    p2_cfg = Phase2Config(
        lr=1e-4, k_steps=3, lambda_pred=1.0, lambda_lyap=0.1,
        num_steps=20, batch_size=4, n_select=2,
    )
    model, w_lyap, select_idx, hist2 = train_phase2(
        model, data_fn_clips_and_states, p2_cfg, k_phase2, log_every=5
    )

    print("\nDone.")
    print("Selected interpretable latent dims (z_tilde indices):", select_idx)

    print("\n=== Plotting rollout learning diagnostics ===")
    eval_clips, eval_states = data_fn_clips_and_states(k_eval, 8)
    make_all_plots(
        model_before_phase2=model_after_phase1,
        model_after_phase2=model,
        hist1=hist1,
        hist2=hist2,
        eval_clips=eval_clips,
        eval_states=eval_states,
        select_idx=select_idx,
        k_steps=p2_cfg.k_steps,
        out_dir="plots",
    )
    print("Saved plots to ./plots/ "
          "(loss_curves.png, rollout_frames.png, "
          "rollout_error_vs_horizon.png, latent_vs_truth.png)")


if __name__ == "__main__":
    main()
