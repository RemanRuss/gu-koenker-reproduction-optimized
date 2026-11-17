# Unit 3 optimization: Gaussian location mixture simulation

## Scope

We preserve the Unit 2 estimand, prior scenarios, random seed and child `SeedSequence` streams, sample sizes, finite grid, iteration cap, convergence tolerance and three original estimators. Original `fit_kw` and `posterior_mean` are retained in `src/model_baseline.py` for comparisons. The optimized default implementation is in `src/model.py`.

## Optimization 1 — algorithmic caching and matrix-vector EM

**Problem:** Local Windows profiling of the original `fit_kw` identified 350 calls to `scipy.special.logsumexp`, cumulatively consuming 0.202 seconds out of 0.281 profiled seconds for one n=400 fit; the optimized `fit_kw` required 0.012 seconds in the same profiler run. Each iteration materializes an `n × m` matrix of log-joint component densities and new responsibility arrays.

**Original (abbreviated):**

```python
lw = log_kernel + np.log(np.maximum(weights, 1e-300))[None, :]
den = logsumexp(lw, axis=1)
weights = np.exp(lw - den[:, None]).mean(axis=0)
```

**Optimized:**

```python
z = -0.5 * (y[:, None] - grid[None, :])**2
rowmax = z.max(axis=1)
kernel = np.exp(z - rowmax[:, None])  # computed once
# Each iteration:
den = kernel @ weights
weights *= (kernel.T @ (1.0 / den)) / y.size
weights /= weights.sum()
```

The E-step uses `w_j^{new} = w_j * (1/n) sum_i K_ij / sum_k(K_ik w_k)`. Row scaling cancels in posterior responsibilities and keeps `den` positive for extreme observations. The cache avoids repeated creation of expensive `logsumexp` normalization intermediates. The per-iteration operation count stays **O(nm)**; the optimization reduces constant factors, allocation pressure and special-function overhead, not the big-O order. Cached matrix space is **O(nm)**.

**Trade-offs:** More algebraic code and an additional renormalization step; floating-point associativity causes negligible rounding differences. EM convergence remains slow and 350 iterations can still be insufficient. The algorithm does not certify that the optimum has been reached.

## Optimization 2 — array-programming posterior mean

**Problem:** The original posterior mean creates a full log-posterior/responsibility array, normalizes each row through `logsumexp` and multiplies by grid values.

**Original:**

```python
logp = norm.logpdf(y[..., None] - grid) + np.log(np.maximum(weights, 1e-300))
post = np.exp(logp - logsumexp(logp, axis=-1, keepdims=True))
return np.sum(post * grid, axis=-1)
```

**Optimized:**

```python
z = -0.5 * (y[..., None] - grid)**2
k = np.exp(z - np.max(z, axis=-1, keepdims=True))
return (k @ (weights * grid)) / (k @ weights)
```

The optimized routine uses broadcasting and BLAS matrix-vector products; a safe log-space fallback handles numerically degenerate weights. The oracle estimator remains as in Unit 2, including its 1,201-point approximation for the uniform prior. This preserves the study's numerical target rather than silently replacing it by an analytically different oracle.

**Trade-offs:** The dot-product expression is less immediately recognizable than normalized posterior probabilities; the fallback adds code. Floating-point summation order can differ.

## Optimization 3 — optional independent-replicate parallelization

`src/simulation.py` adds `--workers K` and `make parallel` uses 2 workers. Each task receives the exact same `SeedSequence.spawn(...)` child stream assigned by its scenario and replicate index regardless of worker count. The parent writes output files in deterministic task order. This is process-level parallelism and is **optional** to achieve the assignment's two-category requirement above.

**Trade-offs:** Process startup, inter-process transfer and memory duplication can dominate short runs; actual speedup depends on processor count and BLAS threading. For apples-to-apples runs consider using `OPENBLAS_NUM_THREADS=1` and compare `--workers 1` with `--workers 2` on the full 40-replicate study. No claim of parallel speedup is made without such a timed comparison.

## Measured performance

The matched-input EM-only benchmark run on the local Windows machine (`make benchmark`, `results/performance/benchmark.csv`) measured the following median wall times in seconds. Timings depend on hardware and reflect **EM fitting only**, not full end-to-end simulation, writing files or plotting.

| Scenario | n | Baseline (s) | Optimized (s) | Speedup |
|---|---:|---:|---:|---:|
| two_point | 100 | 0.079184 | 0.005344 | 14.82× |
| uniform | 100 | 0.073408 | 0.005231 | 14.03× |
| two_point | 400 | 0.228834 | 0.007958 | 28.76× |
| uniform | 400 | 0.236446 | 0.021477 | 11.01× |
| two_point | 800 | 0.956050 | 0.025335 | 37.74× |
| uniform | 800 | 0.902794 | 0.025186 | 35.84× |

Profiling evidence: `results/performance/profile_baseline.txt`, `profile_optimized.txt`, and corresponding `.prof` files. Figures: `benchmark.png` and `complexity.png`.

**Complexity:** The most recent local fitted log-log slopes (n=100–800) are **two_point** baseline **1.2035**, optimized **0.9779**; **uniform** baseline **1.3588**, optimized **0.9708** (from `results/performance/complexity_slopes.csv`). Both implementations retain theoretical O(Tnm) time complexity; finite-range fitted slopes depend on overhead, cache behavior and fixed iterations and do not establish a lower asymptotic complexity class.

## End-to-end baseline timing

The complete preserved baseline simulation took **40.974 seconds** for 80 replicates (2 scenarios × 40, n=400) on the local Windows machine. This includes data generation, fitting, posterior predictions, metric calculations and CSV export, but excludes downstream analysis and figures. The measured duration is recorded in `results/performance/baseline_full_runtime.json` and baseline outputs in `results/baseline_raw/`. An optimized full-study runtime was not measured, so the EM-only speedups above are not claimed as end-to-end speedups. To reproduce the baseline measurement, run `python -m src.baseline_timing`, followed by `python -m src.finalize_submission`.

## Numerical stability and convergence checks

`make stability-check` examined 36 total fits (18 baseline and 18 optimized). No warnings or exceptions occurred; every fit produced finite nonnegative weights summing to one. **Every fit reached the 350-iteration cap** without meeting the specified stopping tolerance, for both versions. In other words, the optimization did not solve the pre-existing EM convergence limitation.

A separate extreme-observation test (`y = [-1000,-40,0,40,1000]`) verifies finite optimized mixture weights and posterior means. The row-scaling prevents total underflow in the density representation, although very extreme inputs are not representative of the simulation's statistical model.

## Regression tests

Run `make test` or `python -m pytest tests/test_regression.py -q`. The additional 10 tests validate fitted weights and EB predictions against the preserved baseline for both scenarios at 3 sample sizes, posterior and oracle predictions, output metrics, extreme values and invalid inputs. The tests passed locally (**10 passed**). Numerical results match within modest tolerance instead of demanding bitwise equality because multiplication/reduction order is different.

## Lessons learned

1. Replacing repeated log-space special-function reductions with a cached, row-scaled kernel gave the largest measured improvement.
2. Profiling revealed a special-function and allocation bottleneck, not a fundamentally different complexity class.
3. Parallelism is likely most useful for the full 80-fit study; for tiny workloads, the overhead can cancel its benefit.
4. Faster computation does not imply better statistical convergence: the fixed EM iteration cap remains a substantive limitation.
