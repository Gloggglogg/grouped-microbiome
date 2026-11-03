"""Independent numerical identities and adversarial checks for the kernel score."""
from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

from grouped_microbiome._kernel import (NumericalFailure, SeparationError, group_codes,
    group_kernel_score, fit_logistic_null, gaussian_multiplier_pvalue,
    gower_from_profiles, project_features, project_kernel,
    project_observed_residual)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(281)
        self.n = 96
        self.x = np.column_stack([np.ones(self.n), rng.normal(size=(self.n, 2))])
        self.y = rng.binomial(1, expit(-1 + self.x[:, 1] * 0.4)).astype(float)
        self.groups = np.r_[np.arange(16), np.repeat(np.arange(16, 56), 2)]
        self.phi = rng.normal(size=(self.n, 9))
        self.phi -= self.phi.mean(axis=1, keepdims=True)
        self.fit = fit_logistic_null(self.x, self.y)

    def explicit_projection_and_membership(self):
        w = np.diag(self.fit.weights)
        h = self.x.T @ w @ self.x
        p = np.eye(self.n) - self.x @ np.linalg.solve(h, self.x.T @ w)
        codes, sizes = group_codes(self.groups, self.n)
        a = np.zeros((len(sizes), self.n))
        a[codes, np.arange(self.n)] = self.fit.residuals
        return p, a

    def test_firth_intercept_closed_form(self):
        n = 41
        for events in (1, 3, 18, 40):
            y = (np.arange(n) < events).astype(float)
            fitted = fit_logistic_null(np.ones((n, 1)), y)
            np.testing.assert_allclose(fitted.means, (events + 0.5) / (n + 1), atol=1e-13)

    def test_firth_matches_independent_penalized_likelihood_optimizer(self):
        x, y = self.x, self.y
        def objective(beta):
            eta = x @ beta
            mu = expit(eta)
            h = x.T @ ((mu * (1 - mu))[:, None] * x)
            sign, ld = np.linalg.slogdet(h)
            if sign <= 0:
                return np.inf
            return float(np.sum(np.logaddexp(0., eta) - y * eta) - ld / 2)
        reference = minimize(objective, np.zeros(x.shape[1]), method="BFGS",
                             options={"gtol": 1e-7, "maxiter": 1000})
        self.assertLess(np.linalg.norm(reference.jac), 1e-5)
        np.testing.assert_allclose(self.fit.coefficients, reference.x, atol=2e-6, rtol=2e-6)
        self.assertLessEqual(objective(self.fit.coefficients), reference.fun + 1e-9)

    def test_separated_category_firth_is_finite_ordinary_is_diagnosed(self):
        x = np.column_stack([np.ones(70), np.arange(70) < 15,
                             np.sin(np.arange(70))]).astype(float)
        y = ((np.arange(70) % 5 == 0) & (np.arange(70) >= 15)).astype(float)
        fitted = fit_logistic_null(x, y)
        self.assertTrue(np.isfinite(fitted.coefficients).all())
        self.assertTrue((fitted.means > 0).all() and (fitted.means < 1).all())
        self.assertTrue(fitted.diagnostics["converged"])
        with self.assertRaises(SeparationError):
            fit_logistic_null(x, y, method="ordinary")

    def test_ordinary_fit_score_equations_when_mle_is_finite(self):
        fitted = fit_logistic_null(self.x, self.y, method="ordinary")
        self.assertLess(float(np.max(abs(self.x.T @ fitted.residuals))), 1e-8)
        np.testing.assert_allclose(project_observed_residual(fitted), fitted.residuals,
                                   atol=1e-10, rtol=1e-9)

    def test_projection_orientation_and_raw_firth_score(self):
        p, _ = self.explicit_projection_and_membership()
        np.testing.assert_allclose(project_features(self.fit, self.phi), p @ self.phi, atol=1e-12)
        np.testing.assert_allclose(project_observed_residual(self.fit), p.T @ self.fit.residuals, atol=1e-12)
        np.testing.assert_allclose(self.x.T @ project_observed_residual(self.fit), 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(self.x.T @ self.fit.residuals), .01)
        self.assertGreater(np.linalg.norm(p - p.T), .01)

    def test_exact_factor_and_kernel_agree_with_explicit_group_matrix(self):
        p, a = self.explicit_projection_and_membership()
        k = self.phi @ self.phi.T
        m = a @ p @ k @ p.T @ a.T
        expected_q = float(self.fit.residuals @ p @ k @ p.T @ self.fit.residuals)
        factored = group_kernel_score(self.fit, self.groups, features=self.phi)
        direct = group_kernel_score(self.fit, self.groups, kernel=k)
        np.testing.assert_allclose(factored.group_scores @ factored.group_scores.T, m, atol=2e-12)
        np.testing.assert_allclose(direct.group_quadratic, m, atol=2e-12)
        self.assertAlmostEqual(factored.statistic, expected_q, places=10)
        self.assertAlmostEqual(direct.statistic, expected_q, places=10)
        self.assertAlmostEqual(factored.diagnostics["mixture_mean"], float(np.trace(m)), places=9)
        self.assertAlmostEqual(direct.diagnostics["mixture_variance"], float(2 * np.trace(m @ m)), places=8)
        np.testing.assert_allclose(project_kernel(self.fit, k), p @ k @ p.T, atol=3e-12)

    def test_same_gaussian_group_weights_equal_rotated_eigenvalue_formula(self):
        score = group_kernel_score(self.fit, self.groups, kernel=self.phi @ self.phi.T)
        m = score.group_quadratic
        eigenvalues, vectors = np.linalg.eigh(m)
        rng = np.random.default_rng(715)
        xi = rng.normal(size=(200, len(m)))
        explicit = np.einsum("bi,ij,bj->b", xi, m, xi)
        spectral = ((xi @ vectors) ** 2) @ eigenvalues
        np.testing.assert_allclose(explicit, spectral, atol=3e-11, rtol=3e-12)

    def test_row_order_group_relabel_and_member_switch_invariance(self):
        original = group_kernel_score(self.fit, self.groups, features=self.phi)
        rng = np.random.default_rng(512)
        for order in (rng.permutation(self.n), np.r_[np.arange(16),
                      np.arange(16, self.n).reshape(-1, 2)[:, ::-1].ravel()]):
            fitted = fit_logistic_null(self.x[order], self.y[order])
            relabeled = np.array([f"group-{int(f)}" for f in self.groups[order]])
            score = group_kernel_score(fitted, relabeled, features=self.phi[order])
            self.assertAlmostEqual(score.statistic, original.statistic, places=8)
            np.testing.assert_allclose(score.eigenvalues[:8], original.eigenvalues[:8], rtol=1e-10, atol=1e-10)

    def test_nuisance_span_and_kernel_covariate_component_invariance(self):
        transformation = np.array([[1., 3., -.5], [0., 2., 1.], [0., .2, 4.]])
        fitted = fit_logistic_null(self.x @ transformation, self.y)
        np.testing.assert_allclose(fitted.means, self.fit.means, atol=1e-10)
        score = group_kernel_score(self.fit, self.groups, features=self.phi)
        altered = group_kernel_score(fitted, self.groups,
            features=self.phi + self.x @ np.arange(27).reshape(3, 9))
        self.assertAlmostEqual(altered.statistic, score.statistic, places=8)
        np.testing.assert_allclose(altered.eigenvalues[:8], score.eigenvalues[:8], atol=1e-9, rtol=1e-9)

    def test_exact_distance_kernels_are_preserved_and_positive(self):
        rng = np.random.default_rng(816)
        counts = rng.poisson(1, size=(self.n, 12))
        counts[:, 0] += 1
        for metric in ("jaccard", "braycurtis"):
            profiles = counts > 0 if metric == "jaccard" else counts / counts.sum(axis=1)[:, None]
            k = gower_from_profiles(profiles, metric)
            before = k.copy()
            score = group_kernel_score(self.fit, self.groups, kernel=k)
            np.testing.assert_array_equal(k, before)
            self.assertTrue((score.eigenvalues > 0).all())
            self.assertGreaterEqual(score.statistic, 0)

    def test_single_eigenvalue_reference_matches_known_chi_square_tail(self):
        observed = float(chi2.isf(.05, 1) * 2)
        result = gaussian_multiplier_pvalue(observed, np.array([2.]), draws=100000, seed=916)
        self.assertLess(abs(result["p_value"] - .05), .003)
        self.assertEqual(result["finite"], 100000)
        self.assertEqual(result["failed"], 0)
        self.assertFalse(result["finite_sample_exact_test"])

    def test_reference_reproducibility_and_batch_partition(self):
        eigen = np.array([2., .4, .01])
        a = gaussian_multiplier_pvalue(3., eigen, draws=1007, seed=29, batch_size=1007)
        b = gaussian_multiplier_pvalue(3., eigen, draws=1007, seed=29, batch_size=17)
        self.assertEqual(a, b)
        self.assertEqual(a["p_value"], (a["exceedances"] + 1) / 1008)

    def test_constant_collinear_missing_and_invalid_inputs_stop(self):
        for x, y in ((np.c_[self.x, self.x[:, 1]], self.y),
                     (self.x, np.ones(self.n)), (self.x, self.y + .2),
                     (self.x[:, 1:], self.y)):
            with self.assertRaises(NumericalFailure):
                fit_logistic_null(x, y)
        for features in (np.ones((self.n, 2)), np.zeros((self.n, 2)),
                         np.full((self.n, 2), np.nan)):
            with self.assertRaises(NumericalFailure):
                group_kernel_score(self.fit, self.groups, features=features)
        for k in (np.ones((self.n, self.n)), np.zeros((self.n, self.n)),
                  np.diag(np.r_[-1., np.ones(self.n - 1)])):
            with self.assertRaises(NumericalFailure):
                group_kernel_score(self.fit, self.groups, kernel=k)
        for groups in (np.zeros(self.n), np.full(self.n, np.nan), self.groups[:-1]):
            with self.assertRaises(NumericalFailure):
                group_kernel_score(self.fit, groups, features=self.phi)

    def test_asymmetric_kernel_and_unconverged_fit_stop(self):
        k = self.phi @ self.phi.T
        k[0, 1] += 1.
        with self.assertRaises(NumericalFailure):
            group_kernel_score(self.fit, self.groups, kernel=k)
        with self.assertRaises(NumericalFailure):
            fit_logistic_null(self.x, self.y, max_iterations=1)

    def test_invalid_reference_and_profiles_stop(self):
        for eig in (np.array([]), np.array([-1.]), np.array([np.nan])):
            with self.assertRaises(NumericalFailure):
                gaussian_multiplier_pvalue(1., eig, draws=10, seed=1)
        with self.assertRaises(NumericalFailure):
            gower_from_profiles(np.zeros((8, 4)), "jaccard")
        with self.assertRaises(NumericalFailure):
            gower_from_profiles(np.ones((8, 4)), "braycurtis")


if __name__ == "__main__":
    unittest.main(verbosity=2)
