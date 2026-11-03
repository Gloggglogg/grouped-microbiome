"""Formula checks and independent references for the generic array interfaces."""
from __future__ import annotations

import unittest

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from statsmodels.stats.multitest import multipletests

from grouped_microbiome import (
    NumericalFailure, SeparationError, adjust_pvalues, alpha_diversity,
    fit_logistic_association, fit_logistic_null, gaussian_multiplier_pvalue,
    group_kernel_score, kernel_test,
)
from grouped_microbiome._kernel import group_codes


class DiversityTests(unittest.TestCase):
    def test_known_profiles_and_scale_invariance(self):
        counts = np.array([[4., 0., 0.], [1., 1., 1.], [1., 3., 0.]])
        actual = alpha_diversity(counts)
        np.testing.assert_array_equal(actual["richness"], [1, 3, 2])
        expected_shannon = [0., np.log(3), -.25 * np.log(.25) - .75 * np.log(.75)]
        np.testing.assert_allclose(actual["shannon"], expected_shannon)
        np.testing.assert_allclose(actual["inverse_simpson"], [1., 3., 1.6])
        for name, values in alpha_diversity(counts * np.array([[3.], [7.], [11.]])).items():
            np.testing.assert_allclose(values, actual[name])

    def test_invalid_profiles(self):
        for counts in ([[0., 0.]], [[-1., 2.]], [[np.nan, 2.]],
                       [[np.inf, 2.]], [], [1., 2.]):
            with self.subTest(counts=counts), self.assertRaises(NumericalFailure):
                alpha_diversity(counts)


class MultiplicityTests(unittest.TestCase):
    def test_known_values_and_reference_in_input_order(self):
        p = np.array([.04, .001, .2, .02])
        np.testing.assert_allclose(adjust_pvalues(p, "bh"), [.04 * 4 / 3, .004, .2, .04])
        np.testing.assert_allclose(adjust_pvalues(p, "holm"), [.08, .004, .2, .06])
        rng = np.random.default_rng(72)
        for p in (rng.uniform(size=97), np.array([0., 0., 1., .5, .5])):
            for method, reference in (("bh", "fdr_bh"), ("holm", "holm")):
                np.testing.assert_allclose(adjust_pvalues(p, method), multipletests(p, method=reference)[1])
        self.assertEqual(adjust_pvalues([]).shape, (0,))

    def test_invalid_values_and_method(self):
        for p in ([np.nan], [-.1], [1.1], [[.1, .2]]):
            with self.assertRaises(ValueError):
                adjust_pvalues(p)
        with self.assertRaises(ValueError):
            adjust_pvalues([.1], "unknown")


class AssociationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(817)
        self.groups = np.repeat(np.arange(70), 3)
        self.x = rng.normal(size=(len(self.groups), 2))
        self.covariates = rng.normal(size=(len(self.groups), 1))
        latent = rng.normal(scale=.3, size=70)[self.groups]
        self.y = rng.binomial(1, expit(-.2 + .4 * self.x[:, 0] + .2 * self.covariates[:, 0] + latent))
        self.design = np.column_stack([np.ones(len(self.y)), self.x, self.covariates])

    def test_cluster_covariance_matches_independent_sandwich_and_direct_fit(self):
        result = fit_logistic_association(self.x, self.y, covariates=self.covariates, groups=self.groups)
        fitted = sm.GLM(self.y, self.design, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": self.groups})
        np.testing.assert_allclose(result.coefficients, fitted.params, atol=1e-12)
        np.testing.assert_allclose(result.confidence_intervals, fitted.conf_int(), atol=1e-12)
        np.testing.assert_allclose(result.p_values, fitted.pvalues, atol=1e-12)
        mu = expit(self.design @ result.coefficients)
        bread = np.linalg.inv(self.design.T @ ((mu * (1 - mu))[:, None] * self.design))
        scores = np.zeros((70, self.design.shape[1]))
        np.add.at(scores, self.groups, (self.y - mu)[:, None] * self.design)
        n, p = self.design.shape
        reference = bread @ (scores.T @ scores) @ bread * 70 / 69 * (n - 1) / (n - p)
        np.testing.assert_allclose(result.covariance, reference, atol=1e-10, rtol=1e-7)
        np.testing.assert_allclose(result.odds_ratios, np.exp(result.coefficients))
        self.assertEqual(result.terms, ("intercept", "x0", "x1", "covariate0"))
        self.assertEqual(result.diagnostics["groups"], 70)

    def test_model_based_covariance_and_supplied_units(self):
        original = fit_logistic_association(self.x, self.y, covariates=self.covariates)
        scaled = fit_logistic_association(self.x * 5, self.y, covariates=self.covariates)
        fitted = sm.GLM(self.y, self.design, family=sm.families.Binomial()).fit()
        np.testing.assert_allclose(original.covariance, fitted.cov_params(), atol=1e-12)
        np.testing.assert_allclose(scaled.coefficients[1:3] * 5, original.coefficients[1:3], atol=1e-10)
        np.testing.assert_allclose(scaled.p_values, original.p_values, atol=1e-10)
        self.assertEqual(original.diagnostics["covariance"], "model_based")

    def test_invalid_inputs_and_separation(self):
        for x, y, cov, groups in (
            (self.x, self.y, self.x[:, 0], self.groups),
            (self.x, self.y, None, np.zeros(len(self.y))),
            (self.x[:-1], self.y, None, self.groups),
            (self.x, self.y + .1, None, self.groups),
            (self.x, np.ones(len(self.y)), None, self.groups),
        ):
            with self.assertRaises(NumericalFailure):
                fit_logistic_association(x, y, covariates=cov, groups=groups)
        with self.assertRaises(SeparationError):
            fit_logistic_association(self.y, self.y, groups=self.groups)

    def test_row_order_and_group_relabel_invariance(self):
        original = fit_logistic_association(self.x, self.y, groups=self.groups)
        order = np.random.default_rng(73).permutation(len(self.y))
        labels = np.array([f"g{g}" for g in self.groups])
        reordered = fit_logistic_association(self.x[order], self.y[order], groups=labels[order])
        np.testing.assert_allclose(reordered.coefficients, original.coefficients, atol=1e-12)
        np.testing.assert_allclose(reordered.covariance, original.covariance, atol=1e-12)

    def test_odds_ratio_underflow_stops_with_actionable_error(self):
        x = np.repeat([0., .0005], 100)
        y = np.r_[np.ones(40), np.zeros(60), np.ones(10), np.zeros(90)]
        with self.assertRaisesRegex(NumericalFailure, "floating-point resolution"):
            fit_logistic_association(x, y)


class KernelInterfaceTests(unittest.TestCase):
    def test_wrapper_equals_explicit_design_and_reference(self):
        rng = np.random.default_rng(92)
        cov = rng.normal(size=(120, 2))
        y = rng.binomial(1, expit(cov[:, 0] * .2))
        groups = np.repeat(np.arange(40), 3)
        factors = rng.normal(size=(120, 6))
        result = kernel_test(factors, y, groups=groups, covariates=cov, draws=503, seed=91)
        fit = fit_logistic_null(np.column_stack([np.ones(120), cov]), y)
        score = group_kernel_score(fit, groups, features=factors)
        reference = gaussian_multiplier_pvalue(score.statistic, score.eigenvalues, draws=503, seed=91)
        np.testing.assert_allclose(result.null_fit.means, fit.means)
        self.assertEqual(result.score.statistic, score.statistic)
        self.assertEqual(result.reference, reference)
        self.assertEqual(result.p_value, reference["p_value"])

    def test_mixed_scalar_group_labels_remain_distinct(self):
        codes, sizes = group_codes([1, "1", 1, "1"], 4)
        np.testing.assert_array_equal(codes, [0, 1, 0, 1])
        np.testing.assert_array_equal(sizes, [2, 2])

    def test_reference_counts_must_be_integer_and_nonboolean(self):
        for draws in (True, 3.5, 0):
            with self.assertRaises(ValueError):
                gaussian_multiplier_pvalue(1., np.array([1.]), draws=draws, seed=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
