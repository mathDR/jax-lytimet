# LyTimeT (JAX / Equinox / Optax)

Implementation of **"LyTimeT: Towards Robust and Interpretable State-Variable
Discovery"** (Yu, Su, Liu, Goldfeder, Shao, Lipson — Columbia/NUS, ICASSP
2026, [arXiv:2510.19716](https://arxiv.org/abs/2510.19716)).

The paper is a two-phase framework for extracting interpretable,
physically-meaningful state variables from video of a dynamical system:

- **Phase 1** — a TimeSformer-style spatio-temporal autoencoder learns a
  distraction-robust latent state `z_t` and a latent transition model
  `f_theta` for multi-step video forecasting.
- **Phase 2** — the learned latent space is probed with linear regression
  against ground-truth variables, the most physically meaningful dimensions
  are selected (`z_tilde`), and the transition dynamics are refined with a
  Lyapunov-based stability regularizer that penalizes non-contractive
  roll-outs.

## Files

| File | Contents |
|---|---|
| `model.py` | `PatchEmbed`, `MultiHeadSelfAttention`, `TimeSformerBlock` (factorized temporal-then-spatial attention), `Encoder`, `Decoder` (deconv + skip connections), `LatentTransition` (residual MLP, LayerNorm+GELU), and the combined `LyTimeT` module. Includes `LyTimeTConfig.lite()` for the paper's **LyTimeT-Lite** variant (fewer heads, smaller hidden dim). |
| `losses.py` | `L_rec`, `L_pred` (K-step unrolled prediction), `L_phase1 = L_rec + λ_pred·L_pred`, the Lyapunov energy `V(z̃) = ‖Wz̃‖²`, `L_lyap`, and the combined `L = L_phase1 + λ_lyap·L_lyap`. |
| `probe.py` | Closed-form linear probing (`fit_linear_probe`), `AMSE` (paper eq. 2-3), R²-based dimension ranking (`rank_and_select_dimensions`, a practical proxy for the paper's mutual-information ranking) used to select `z_tilde`, and a disentanglement-consistency check across nuisance conditions. |
| `data.py` | A synthetic single-pendulum video generator (`θ'' + (g/l) sin θ = 0`) with rendered grayscale frames and known ground truth `(θ, θ̇)`, standing in for one of the paper's five synthetic benchmarks so the pipeline is runnable end-to-end without proprietary data. |
| `train.py` | Optax training loops: `train_phase1` (representation + forecasting) and `train_phase2` (probe → select `z_tilde` → fine-tune `f_theta` and `W` with the Lyapunov loss). |
| `plot.py` | Matplotlib + Seaborn plotting utilities: training-loss curves, ground-truth-vs-predicted rollout frame grids, rollout-error-vs-horizon curves (the key "did stability regularization work" plot, comparing Phase 1 alone vs. Phase 1+Lyapunov), and extracted-variable-vs-ground-truth trajectory overlays. `make_all_plots(...)` runs all of them and saves PNGs to a folder. |
| `demo.py` | Small runnable end-to-end example (`python -m lytimet.demo`) — trains both phases and then calls `make_all_plots` to save diagnostics to `./plots/`. |

## Quick start

```bash
pip install jax jaxlib equinox optax einops jaxtyping
python -m lytimet.demo
```

```python
import jax
from lytimet import LyTimeT, LyTimeTConfig
from lytimet.data import make_pendulum_batch
from lytimet.train import Phase1Config, Phase2Config, train_phase1, train_phase2

cfg = LyTimeTConfig(image_size=64, patch_size=8, dim=192, depth=6,
                     num_heads=6, dz=32, clip_len=16)
# For the compute-efficient variant from the paper:
# cfg = cfg.lite()

key = jax.random.PRNGKey(0)
model = LyTimeT(cfg, key=key)

def data_fn(k, b): 
    clips, _ = make_pendulum_batch(k, b, cfg.clip_len, cfg.image_size)
    return clips

model, _ = train_phase1(model, data_fn, Phase1Config(num_steps=2000), key)

def data_fn2(k, b):
    return make_pendulum_batch(k, b, cfg.clip_len, cfg.image_size)

model, w_lyap, select_idx, _ = train_phase2(model, data_fn2, Phase2Config(n_select=4), key)
```

## Notes / design choices

- **Batching**: modules operate on a single clip `(T, C, H, W)`; `jax.vmap`
  handles the batch dimension at the training-loop level (idiomatic
  Equinox style), and `eqx.filter_jit` / `eqx.filter_value_and_grad` handle
  the static config fields correctly.
- **Attention**: implemented manually (rather than via a library layer) so
  the *factorized* temporal-then-spatial structure central to TimeSformer
  is explicit and easy to modify (e.g. to add axial masking or sparsified
  patches for `LyTimeT-Lite`).
- **Decoder skip connections**: the decoder optionally takes the encoder's
  early per-frame patch tokens (pre-attention) as a skip input, matching
  Fig. 1's description; when forecasting *future*, un-encoded frames these
  aren't available, so forecasting decodes from the latent alone.
- **Lyapunov loss**: `f_theta` always acts on the full `d_z`-dimensional
  latent (it's trained once in Phase 1), but the energy `V` — and hence the
  stability penalty — is evaluated only on the selected interpretable
  dimensions `z̃`, per the paper's Phase 2 description of regularizing the
  *extracted* variables.
- **Data augmentation** for distraction-robustness (background swap,
  texture perturbation, occlusion masks, brightness jitter — Sec. 2.1) is
  not implemented since it's dataset-specific; hook it into your own
  `data_fn` passed to `train_phase1`.
- **Metrics**: `probe.py` gives you `AMSE` directly; intrinsic-dimension
  (2-NN estimator, paper eq. 4) and full mutual-information (Gaussian KDE)
  aren't included but are straightforward to add on top of the extracted
  `z_tilde` trajectories if you want the paper's exact Table 1 metrics.

This is a from-scratch, paper-faithful reimplementation (there is no
official public code release for LyTimeT at the time of writing) — some
low-level details (exact decoder upsampling schedule, exact augmentation
recipe, exact ranking criterion for dimension selection) are not fully
specified in the paper and were filled in with reasonable, clearly-marked
choices.
