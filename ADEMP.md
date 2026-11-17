# ADEMP — Gu & Koenker (2017), Section 3.4.1

**Aims.** Partially reproduce the two illustrative Gaussian location-mixture scenarios in the open working-paper version, and evaluate whether nonparametric empirical Bayes (EB) shrinkage improves estimation of individual latent means relative to naive observations and approaches the oracle Bayes rule.

**Data generating processes.** Independently for individuals: Y_i = alpha_i + U_i, with U_i ~ N(0,1). We study (1) a two-point prior with P(alpha_i=-h)=2/3 and P(alpha_i=2h)=1/3; (2) alpha_i ~ Uniform[-h,2h]. Set h=0.5, n=400, seed=20261021, and 40 independent simulated datasets per prior. Replicated trials are an extension beyond the paper's illustrative single runs.

**Estimands.** Individual latent means alpha_i, conditional posterior mean E(alpha_i|Y_i), mixing distribution F, and its induced observation density g(y).

**Methods.** Naive (Y_i), empirical Bayes (finite-grid Kiefer–Wolfowitz mixture likelihood estimated by EM, then posterior mean), and oracle Bayes (posterior mean using the known prior). The implementation uses 121 equally spaced grid points and stops EM after at most 350 iterations rather than the paper's interior-point optimization.

**Performance.** Compute per-run MSE, MAE, bias, and across-run mean MSE, standard deviation and Monte Carlo standard error (sample standard deviation of replicate-level MSE divided by sqrt(40)). Compare mixture/density/estimation-rule figures.

| Scenario | n | Prior | Noise | Replicates | Methods |
| --- | ---: | --- | --- | ---: | --- |
| Two-point | 400 | 2/3 at -0.5, 1/3 at 1.0 | Normal(0,1) | 40 | Naive / EB / Oracle |
| Uniform | 400 | Uniform(-0.5, 1.0) | Normal(0,1) | 40 | Naive / EB / Oracle |

**Scope.** Partial methodological reproduction of Section 3.4.1 rather than the full article, the income-panel-data application, or exact published numerical figures. Random seeds, EM optimization precision and Monte Carlo summaries differ from the publication.
