# Critical Analysis — Partial Reproduction of Gu and Koenker (2017)

## Findings

The two simulated scenarios use 400 subjects with normal observation noise and latent means drawn from either a two-point or a uniform prior. The study extends the illustrative plots with 40 Monte Carlo replications for each condition.

| Scenario | Naive MSE | Empirical Bayes MSE | Oracle Bayes MSE |
| --- | ---: | ---: | ---: |
| Two-point | 0.9953 | 0.3353 | 0.3207 |
| Uniform | 0.9978 | 0.1678 | 0.1574 |

Results are from the 40-replicate simulated study (`results/raw/summary.csv`); they are not numerical results quoted from the Gu–Koenker paper. EB reduces mean MSE relative to naive estimation by about 66% in the two-point scenario and 83% in the uniform scenario, with the oracle doing slightly better.

## Limitations and interpretation

All three estimators are evaluated on identical generated observations, which makes comparisons fair. However, the DGP agrees with the Gaussian-mixture assumptions; misspecification, heavy tails, dependence and other challenges are not studied. Oracle Bayes is an informative ideal but is not feasible without knowing the true prior. The finite-grid estimator uses EM rather than the interior-point solver used by the paper. The cap of 350 EM iterations was reached in tested scenarios, so fitted mixing weights cannot be assumed to be fully converged or an exact reproduction of the paper's optimization output. The qualitative shrinkage benefit is more reliable than the exact recovered mixing masses.

The Unit 3 work accelerates the finite-grid computation while retaining the same statistical experiment. See `docs/BASELINE.md` and `docs/OPTIMIZATION.md` for measured performance, correctness checks and convergence limitations.

## Reference

Gu, J., & Koenker, R. (2017). Unobserved heterogeneity in income dynamics: An empirical Bayes perspective. *Journal of Business & Economic Statistics*, 35(1), 1–16. https://doi.org/10.1080/07350015.2015.1052457
