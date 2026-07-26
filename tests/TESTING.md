# Testing

This repo's test suite lives in `tests/` and covers every module in
`lytimet/`: `model.py`, `losses.py`, `data.py`, `probe.py`,
`ode_transition.py`, `conformal.py`, `train.py`, and `plot.py`, plus
end-to-end integration smoke tests in `test_integration.py`.

## Install

```bash
pip install -e ".[test]"
```

## Run the fast suite (recommended for every commit / PR)

```bash
pytest -q -m "not slow"
```

Everything in this suite uses tiny model configs (e.g. `dim=16`, `dz=4`,
1-2 attention layers, 5-6 frame clips) so it runs in ~1-2 minutes on a
CPU-only runner. It checks shapes, numerical edge cases (e.g. Lyapunov
loss is exactly zero for a provably contracting map, non-zero for an
expanding one), gradient flow through the Neural ODE solver, and that a
single optimizer step actually changes parameters.

## Run the slow / integration suite

```bash
pytest -q -m slow
# or, to run everything:
pytest -q
```

`slow`-marked tests run real (tiny) training loops for a handful of steps
-- e.g. checking that Phase 1 loss actually decreases over 15 steps, that
the full Phase 1 -> Phase 2 pipeline runs for both `dynamics_type`s, and
that conformal calibration + coverage evaluation work end-to-end. These
take a few minutes total and are run nightly / on-demand in CI (see
`.github/workflows/ci.yml`) rather than on every push, to keep the fast
path fast.

## Coverage

```bash
pytest -q --cov=lytimet --cov-report=term-missing
```

Demo scripts (`lytimet/demo*.py`) are excluded from the coverage target
(see `[tool.coverage.run] omit` in `pyproject.toml`) since they're runnable
examples, not library code exercised by unit tests. Core library coverage
is currently ~99% with the full (fast + slow) suite.

## CI

`.github/workflows/ci.yml` runs the fast suite on every push/PR across
Python 3.11 and 3.12, and the full suite (including slow tests) nightly,
on manual dispatch, or on a PR labeled `run-slow-tests`.
