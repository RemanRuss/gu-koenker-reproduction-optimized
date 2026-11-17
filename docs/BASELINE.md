# Baseline performance: Unit 2 Gu–Koenker simulation

## Environment and reproducibility

The baseline is **the original Unit 2 implementation** of `fit_kw` copied without algorithm changes into `src/model_baseline.py`. The study has two scenarios (`two_point` and `uniform`), `n=400`, `GRID_POINTS=121`, `MAX_ITER=350`, and `REPS=40` per scenario (80 fits in the full original study). Use `make profile`, `make complexity`, and `make stability-check` to reproduce the measurements. The profiling and benchmark figures below were measured on the local Windows Python 3.13 environment; timings are machine-dependent. The full-study baseline timing was also measured on the local Windows machine.

## Whole original-study runtime

The original implementation can now be timed end to end with:

```powershell
python -m src.baseline_timing
```

This script uses the preserved `src.model_baseline` routines and the same 2 scenarios × 40 replications, random-number child streams, grid, and iteration limit. It measures data generation, EM fitting, posterior predictions, metrics, and CSV writes, but not subsequent analysis/figure creation. Output is saved separately under `results/baseline_raw/`; the elapsed seconds are recorded in `results/performance/baseline_full_runtime.json` so the optimized `results/raw/` stays intact. **Local full-run timing:** 40.974 seconds for 2 scenarios x 40 replicates (n=400, 121 grid points, cap 350), measured on the local machine using `python -m src.baseline_timing`, including generation, EM fitting, posterior means, metrics and CSV writing (excluding analysis/figures). Saved in `results/performance/baseline_full_runtime.json`.

## Runtime profiler

One representative `n=400`, `two_point` EM fit from the Windows profiler (`results/performance/profile_baseline.txt`):

- The profiler recorded **0.281 seconds** in the original `fit_kw` (instrumented time; differs from median wall-clock timing).
- `scipy.special.logsumexp`: **350 calls**, **0.202 seconds cumulative**, the main identifiable bottleneck.
- `fit_kw` performs a new `n × m` array of log-responsibilities, a row-wise `logsumexp` reduction and exponentiation on every iteration.

The original EM fitter runs at most 350 iterations per fit, with memory for the `n × m` kernel and intermediates. Time complexity is **O(T × n × m)**, with `T` the number of iterations and `m=121`; space is **O(n × m)**. As both `T` and `m` are fixed in this experiment, runtime should be near-linear in `n`, up to fixed overhead and memory-system effects.

## Empirical complexity

Measured median runtimes for 350-iteration EM fits on matched generated data are in `results/performance/complexity.csv`. For n = 100, 200, 400, 800, the most recent local log-log slopes were **1.2035** (two_point baseline) and **1.3588** (uniform baseline), computed from `results/performance/complexity_slopes.csv`. The corresponding optimized slopes were **0.9779** and **0.9708**. The plot is saved as `complexity.png`. These finite-range slopes are not asymptotic proofs.

## Numerical warnings and convergence

`make stability-check` executed 18 baseline EM fits (2 scenarios × 3 sample sizes × 3 seeds): **0 warnings**, **0 exceptions**, **18/18 finite valid weight vectors**. **0/18** reached the stopping tolerance before 350 iterations, so **18/18 reached the iteration cap**. This is a convergence limitation: absence of warnings does *not* establish full optimization convergence or NPMLE accuracy. The EM algorithm can converge slowly near the boundary of the weight simplex. The fixed 350-iteration cap is retained for baseline comparability.

## Interpretation

The most actionable bottleneck is the repeated log-space normalization in the EM E-step, not data generation. This motivated a row-scaled, cached likelihood kernel and matrix-vector EM update.
