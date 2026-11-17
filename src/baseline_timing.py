"""Time the preserved Unit 2 numerical pipeline without changing optimized outputs.

Run from the project root: python -m src.baseline_timing
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N, H, SEED, REPS, SCENARIOS, GRID_POINTS, MAX_ITER
from . import model_baseline as baseline


def run(reps=REPS, seed=SEED, n=N):
    if reps < 1 or n < 1:
        raise ValueError("reps and n must be positive")
    started = time.perf_counter()
    seedseq = np.random.SeedSequence(seed)
    streams = seedseq.spawn(len(SCENARIOS) * reps)
    results, diagnostics = [], []
    dest = Path("results/baseline_raw")
    dest.mkdir(parents=True, exist_ok=True)
    iterations = []

    for i, scenario in enumerate(SCENARIOS):
        for r in range(reps):
            rng = np.random.default_rng(streams[i * reps + r])
            y, alpha = baseline.generate(n=n, h=H, scenario=scenario, rng=rng)
            grid = np.linspace(min(-2.0, float(y.min()) - 0.2),
                               max(2.0, float(y.max()) + 0.2), GRID_POINTS)
            grid, weights, iters = baseline.fit_kw(y, grid, max_iter=MAX_ITER, tol=1e-7)
            iterations.append(iters)
            eb = baseline.posterior_mean(y, grid, weights)
            truth = baseline.oracle(y, scenario, H)
            for method, pred in (("naive", y), ("empirical_bayes", eb),
                                 ("oracle_bayes", truth)):
                results.append({"scenario": scenario, "replicate": r, "n": n,
                                "method": method,
                                "mse": float(np.mean((pred - alpha)**2)),
                                "bias": float(np.mean(pred - alpha)),
                                "mae": float(np.mean(np.abs(pred - alpha)))})
            if r == 0:
                diagnostics.append(pd.DataFrame({"scenario": scenario, "y": y,
                                                 "alpha": alpha, "eb": eb,
                                                 "oracle": truth}))
                pd.DataFrame({"grid": grid, "weight": weights,
                              "scenario": scenario}).to_csv(
                    dest / f"weights_{scenario}.csv", index=False)

    pd.DataFrame(results).to_csv(dest / "simulation.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(dest / "diagnostic.csv", index=False)
    elapsed = time.perf_counter() - started
    record = {"baseline_full_runtime_seconds": round(elapsed, 6),
              "reps_per_scenario": reps, "n": n, "total_replicates": len(SCENARIOS)*reps,
              "grid_points": GRID_POINTS, "max_iter": MAX_ITER,
              "seed": seed, "iterations_min": min(iterations),
              "iterations_max": max(iterations),
              "note": "Generation, fitting, posterior predictions, metrics and CSV export; excludes analysis and figures."}
    perf = Path("results/performance")
    perf.mkdir(parents=True, exist_ok=True)
    (perf / "baseline_full_runtime.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print("Baseline output saved to results/baseline_raw; optimized results/raw untouched.")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n", type=int, default=N)
    args = parser.parse_args()
    run(reps=args.reps, seed=args.seed, n=args.n)
