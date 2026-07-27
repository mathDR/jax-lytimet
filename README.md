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
| `model.py` | `PatchEmbed`, `MultiHeadSelfAttention`, `TimeSformerBlock` (factorized temporal-then-spatial attention), `Encoder`, `Decoder` (deconv + skip connections), `LatentTransition` (the paper's default: residual MLP, LayerNorm+GELU), and the combined `LyTimeT` module. Includes `LyTimeTConfig.lite()` for the paper's **LyTimeT-Lite** variant (fewer heads, smaller hidden dim), and `LyTimeTConfig(dynamics_type=...)` to select the transition model (see `ode_transition.py`). |
| `ode_transition.py` | **Optional, non-paper** alternative dynamics model: a Neural ODE (`dz/dt = g_theta(z,t)`, integrated with `diffrax`) as a drop-in replacement for the residual-MLP transition, plus a continuous-time Lyapunov loss `dV/dt <= 0` (see "Residual MLP vs. Neural ODE" below). |
| `losses.py` | `L_rec`, `L_pred` (K-step unrolled prediction), `L_phase1 = L_rec + λ_pred·L_pred`, the Lyapunov energy `V(z̃) = ‖Wz̃‖²`, `L_lyap`, the combined `L = L_phase1 + λ_lyap·L_lyap`, and an optional `latent_dynamics_matching_loss` (weight `λ_dyn`, see "Direct latent-space dynamics supervision" below). |
| `probe.py` | Closed-form linear probing (`fit_linear_probe`), `AMSE` (paper eq. 2-3), R²-based dimension ranking (`rank_and_select_dimensions`, a practical proxy for the paper's mutual-information ranking) used to select `z_tilde`, and a disentanglement-consistency check across nuisance conditions. |
| `data.py` | A synthetic single-pendulum video generator (`θ'' + (g/l) sin θ = 0`) with rendered grayscale frames and known ground truth `(θ, θ̇)`, standing in for one of the paper's five synthetic benchmarks so the pipeline is runnable end-to-end without proprietary data. Also includes an **actuated** pendulum variant (`make_actuated_pendulum_batch`) with a continuous torque input and a discrete damping-mode switch, for exercising/comparing action-conditioned dynamics (see `actions.py`). |
| `actions.py` | **Optional, non-paper** action-conditioned latent dynamics, supporting a mix of discrete and continuous actions, in two architectures x two time conventions: `ConcatActionTransition`/`ConcatActionODETransition` (concatenation baseline) and `ControlAffineActionTransition`/`ControlAffineActionODETransition` (structured control-affine: `drift(z) + B(z) @ continuous_action`). See "Action-conditioned dynamics" below. |
| `train.py` | Optax training loops: `train_phase1` (representation + forecasting) and `train_phase2` (probe → select `z_tilde` → fine-tune `f_theta` and `W` with the Lyapunov loss). |
| `plot.py` | Matplotlib + Seaborn plotting utilities: training-loss curves, ground-truth-vs-predicted rollout frame grids, rollout-error-vs-horizon curves (the key "did stability regularization work" plot, comparing Phase 1 alone vs. Phase 1+Lyapunov), and extracted-variable-vs-ground-truth trajectory overlays. `make_all_plots(...)` runs all of them and saves PNGs to a folder. |
| `demo.py` | Small runnable end-to-end example (`python -m lytimet.demo`) — trains both phases and then calls `make_all_plots` to save diagnostics to `./plots/`. |
| `demo_compare_dynamics.py` | Trains the paper-default residual-MLP model and the alternative Neural ODE model side by side and plots rollout error vs. horizon for both (`python -m lytimet.demo_compare_dynamics`). |
| `conformal.py` | Particle-ensemble rollouts (`particle_rollout`, `vmap`ping the trained transition across perturbed initial conditions) + split conformal calibration (`calibrate_conformal`, `predict_with_interval`, `evaluate_coverage`) for distribution-free, per-timestep calibrated error bars on the latent rollout. See "Uncertainty quantification" below. |
| `demo_conformal.py` | Trains a Neural ODE model, calibrates conformal intervals on held-out clips, checks empirical coverage on a fresh test set, and plots the calibrated bands (`python -m lytimet.demo_conformal`). |
| `demo_compare_actions.py` | Trains the concatenation baseline vs. control-affine model directly on the actuated pendulum's true 2D state, then evaluates both in-distribution and on out-of-distribution (larger) torque magnitudes (`python -m lytimet.demo_compare_actions`). |

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

## Action-conditioned dynamics: concatenation vs. control-affine

`actions.py` extends the latent transition to take actions -- a mix of
discrete (e.g. mode switches) and continuous (e.g. torque/force) -- via two
architectures, each provided in both discrete-time and Neural ODE form so
all four can be trained and compared directly:

- **Concatenation** (`ConcatActionTransition`, `ConcatActionODETransition`):
  discrete actions are embedded, concatenated with the continuous action
  and the latent state, and fed through an otherwise-ordinary MLP. General-
  purpose, makes no assumption about *how* actions affect dynamics.
- **Control-affine** (`ControlAffineActionTransition`,
  `ControlAffineActionODETransition`): structured for physical control,
  which is very often affine in the continuous input (e.g. a torque-driven
  pendulum: `theta'' = -(g/l) sin(theta) + tau/(mL^2)` -- additive and
  *linear* in `tau`). Splits the update into `drift(z, discrete_embed) +
  B(z, discrete_embed) @ continuous_action`, where `B` is a learned,
  state-and-mode-dependent input-gain matrix. More sample-efficient and
  should extrapolate better to action magnitudes outside the training
  range *if* the system really is control-affine. Discrete actions are
  *not* forced into this affine structure (that would be a strange
  assumption, e.g. "gear 2" isn't "2x gear 1") -- they condition `drift`
  and `B` through an embedding, same as in the concatenation baseline;
  only the continuous channel gets the linear structure.

Both share a common interface:

```python
z_next = model(z, discrete_actions, continuous_action)   # discrete_actions: int[n_discrete], continuous_action: float[d_c]
traj = model.rollout(z0, discrete_actions_seq, continuous_actions_seq, k_steps)
```

using fixed-size (possibly zero-length) arrays for the unused action type
rather than `None`, to stay `vmap`/`scan`-friendly.

**Lyapunov-loss caveat**: once actions exist, the stability penalty should
apply to the **drift term only** (`model.drift_only_step` /
`model.drift_only_vector_field` on the control-affine variants) -- a
controller is *supposed* to be able to add energy (e.g. swing-up), and
penalizing that unconditionally would fight against correct control
effort.

`demo_compare_actions.py` trains both architectures directly on the
actuated pendulum's true state and compares in-distribution vs.
out-of-distribution (3x larger torque) one-step prediction error. In our
short sandbox run the gap was modest (concat: 1.04x error growth
OOD/ID vs. control-affine: 1.07x) -- for this particular 2D, fairly smooth
system a well-trained MLP already extrapolates reasonably over a 3x
range, so the benefit of the structural prior wasn't dramatic here. The
comparison is more likely to matter with less training data, a larger
extrapolation gap, or a higher-dimensional/less smooth action-response
surface -- worth re-running with your actual system and action ranges
rather than trusting this toy result.

## Uncertainty quantification: particle ensembles + conformal prediction

`conformal.py` gives you calibrated, per-timestep error bars on a latent
rollout, in two composed steps:

1. **Particle rollout** (`particle_rollout`): perturb the encoder's
   initial latent `z_0` with small Gaussian noise, forming `num_particles`
   initial conditions, and `vmap` the *same trained* transition's
   `.rollout()` across all of them (works for either `dynamics_type`, since
   both share the same interface). The resulting ensemble's mean/std at
   each horizon step is a rough, uncalibrated local uncertainty scale.
   **Caveat**: this captures sensitivity to *initial-condition* noise
   propagated through the dynamics -- it does *not* capture model/parameter
   (epistemic) uncertainty, which would need a deep ensemble of
   independently-trained transitions instead of/in addition to this.

2. **Split conformal calibration** (`calibrate_conformal`,
   `predict_with_interval`): the raw ensemble spread has no coverage
   guarantee on its own. On a held-out calibration set, we compute a
   *normalized* nonconformity score per (horizon step, latent dimension) --
   `|z_true - ensemble_mean| / ensemble_std` -- and take its finite-sample-
   corrected empirical `(1-alpha)` quantile, `q_hat`. At deployment,
   `[mean - q_hat*std, mean + q_hat*std]` is a distribution-free interval
   with marginal coverage `>= 1 - alpha` **per horizon step** (not a joint
   guarantee over the whole path), as long as calibration and deployment
   clips are exchangeable. `evaluate_coverage` checks this empirically on a
   fresh test set. In our sandbox run (target 90%), empirical coverage came
   out at 92-94% across a 5-step horizon -- correctly conservative, as
   expected from the finite-sample correction.

```python
from lytimet.conformal import calibrate_conformal, predict_with_interval, evaluate_coverage

q_hat = calibrate_conformal(model, calib_clips, k_steps=5, num_particles=30,
                             noise_std=0.05, alpha=0.1, key=key)
mean, lower, upper = predict_with_interval(model, z0, k_steps=5, num_particles=30,
                                           noise_std=0.05, q_hat=q_hat, key=key)
coverage = evaluate_coverage(model, test_clips, k_steps=5, num_particles=30,
                              noise_std=0.05, q_hat=q_hat, key=key)
```

See `demo_conformal.py` for the full pipeline (train -> calibrate -> check
coverage -> plot).

## Direct latent-space dynamics supervision (`lambda_dyn`)

The rollout loss `L_pred` trains the transition module only indirectly
(unroll → decode → compare pixels), which is expensive and gives a noisy,
high-variance gradient to the dynamics module specifically. Since the
encoder already produces `z_1..z_T` for every clip, consecutive pairs
`(z_t, z_{t+1})` are free teacher-forcing targets. `Phase1Config(lambda_dyn=...)`
(and the corresponding `Phase2Config` field) turns on

```
pred_{t+1} = model.transition(z_t)
loss       = mean || pred_{t+1} - z_{t+1} ||^2
```

as an added, cheap, direct loss (no decoder, no multi-step rollout). Note
this deliberately compares *states*, not a finite-difference derivative
estimate `(z_{t+1}-z_t)/dt` — differencing amplifies encoder noise as `dt`
shrinks and is only first-order accurate as `dt` grows, whereas
`NeuralODETransition.__call__` already integrates the vector field over the
true elapsed interval, so comparing its output directly to `z_{t+1}` avoids
derivative-estimation error entirely and uses the exact same one-liner for
both `dynamics_type`s.

This is a genuinely useful *auxiliary* signal, not a replacement for
`L_pred`: one-step-accurate dynamics do not imply good multi-step rollouts
(compounding/exposure-bias error, especially for the chaotic double-pendulum
benchmark, where nearby trajectories diverge exponentially). Keep
`lambda_dyn` small relative to `lambda_pred`.

## Residual MLP vs. Neural ODE dynamics

The paper's `f_theta` is a **discrete** residual MLP (LayerNorm + GELU) —
that's `LyTimeTConfig.dynamics_type = "residual_mlp"` (the default) and
matches the paper exactly. A **Neural ODE** transition
(`dynamics_type = "neural_ode"`, implemented in `ode_transition.py` via
`diffrax`) is provided as an optional, non-paper alternative:

- It's a better physical prior for the benchmarks in the paper (pendulum,
  double/elastic pendulum, reaction-diffusion), which are themselves
  continuous-time ODEs/PDEs.
- It pairs with a *continuous* Lyapunov check, `dV/dt <= 0`
  (`continuous_lyapunov_loss`), which is exact and step-size-independent,
  vs. the paper's discrete `max(0, V(z_{t+1}) - V(z_t))`.
- It costs more compute per rollout step (multiple solver stages vs. one
  MLP call), which is why it's opt-in rather than the default — the paper
  explicitly avoided combining ODE-style stability constraints with a
  heavy attention encoder for efficiency reasons (see Sec. 1).

Both transition models share the same interface (`__call__(z) -> z_next`,
`.rollout(z0, k_steps) -> (K, Dz)`), so switching is a one-line config
change:

```python
cfg = LyTimeTConfig(..., dynamics_type="neural_ode", ode_solver_steps=4, ode_dt=1.0)
```

and `Phase2Config(use_continuous_lyapunov=True)` switches the stability
loss to the continuous form (only valid with `dynamics_type="neural_ode"`).
See `demo_compare_dynamics.py` for a full side-by-side training + rollout-
error comparison.

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
