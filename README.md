# Gu & Koenker (2017) — Unit 3 High-Performance Simulation Study

This repository extends the Unit 2 partial reproduction of Gu and Koenker's Gaussian location-mixture experiments (Section 3.4.1) with profiling, numerical checks, EM and posterior-mean optimization, and optional multiprocessing. This is a *partial methodological reproduction*, not a reproduction of the entire paper or its income-panel-data analysis.

## Design

Two simulated latent-effect priors, a two-point distribution and a uniform distribution, are observed through Gaussian noise. Default settings: n=400; h=0.5; 40 independent replicates per scenario; 121 grid points; at most 350 EM iterations. Compare naive, empirical Bayes and oracle Bayes estimators.

## Setup and final verification (Windows PowerShell)

From the extracted project directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_FINAL.ps1
```

The script installs requirements, runs tests and performance measurements, runs the original Unit 2 baseline (80 replicates), updates documentation and verifies deliverables. It also reruns the optimized analysis and figures. **Allow the baseline timing to finish.** It is substantially slower than one optimized EM fit.

Alternative manual commands: `python -m pip install -r requirements.txt`, `make test`, `make profile`, `make complexity`, `make benchmark`, `make stability-check`, `make parallel`, `make all`, `python -m src.baseline_timing`, `python -m src.finalize_submission`.

## Files

- `src/model_baseline.py`: preserved Unit 2 implementation; `src/model.py`: optimized estimator.
- `src/simulation.py`, `src/analyze.py`, `src/figures.py`: main experiment, summaries, figures.
- `src/performance.py`: profiling, complexity, benchmark, stability checks.
- `src/baseline_timing.py`: original 80-fit study time, isolated output under `results/baseline_raw/`.
- `src/finalize_submission.py`: inserts end-to-end runtime and checks that results are present.
- `docs/BASELINE.md` and `docs/OPTIMIZATION.md`: required Unit 3 reports.
- `results/performance/`: empirical results, profiler traces and visualizations; `results/figures/`: reproduction figures.
- `ADEMP.md`, `ANALYSIS.md`: methodological background from Unit 2.

## Statistical caveat

No warnings and valid weight vectors do not imply EM convergence. The checked configurations reached the 350-iteration cap; refer to the reports for interpretation. Speedup values in `benchmark.csv` are **EM-only**, not full-study speedups.

Original paper: Gu and Koenker (2017), *Unobserved Heterogeneity in Income Dynamics: An Empirical Bayes Perspective*, Journal of Business & Economic Statistics, 35(1), 1–16, https://doi.org/10.1080/07350015.2015.1052457.
