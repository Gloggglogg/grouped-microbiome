"""Run every public method using entirely random, synthetic observations."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from grouped_microbiome import (
    adjust_pvalues, alpha_diversity, fit_logistic_association, kernel_test,
)


def main() -> None:
    rng = np.random.default_rng(173)
    groups = np.repeat(np.arange(100), 3)
    n = len(groups)
    covariates = rng.normal(size=(n, 2))
    group_effect = rng.normal(scale=0.35, size=100)[groups]
    log_rates = rng.normal(size=(n, 16))
    counts = rng.poisson(np.exp(log_rates))
    counts[:, 0] += 1
    y = rng.binomial(1, expit(-0.3 + 0.4 * covariates[:, 0] + group_effect))

    diversity = alpha_diversity(counts)
    # Each diversity measure is fitted separately, with the same adjustment.
    results = {}
    for name, values in diversity.items():
        # Caller-selected scaling gives an odds ratio per sample standard deviation.
        values = (values - values.mean()) / values.std(ddof=1)
        results[name] = fit_logistic_association(
            values, y, covariates=covariates, groups=groups)
    adjusted = adjust_pvalues([result.p_values[1] for result in results.values()])

    # The illustrative log-ratio transformation adds 0.5 to every count and
    # subtracts each sample's mean log count. Its dot product defines the kernel.
    log_counts = np.log(counts + 0.5)
    factors = log_counts - log_counts.mean(axis=1, keepdims=True)
    community = kernel_test(factors, y, groups=groups, covariates=covariates,
                            draws=1999, seed=174)
    print("Synthetic demonstration")
    for (name, result), p_adjusted in zip(results.items(), adjusted):
        print(f"{name}: odds ratio={result.odds_ratios[1]:.3f}; adjusted p={p_adjusted:.3f}")
    print(f"Kernel test: Monte Carlo p={community.p_value:.3f}")
    print(f"Monte Carlo resolution={community.reference['p_value_resolution']:.4f}")


if __name__ == "__main__":
    main()
