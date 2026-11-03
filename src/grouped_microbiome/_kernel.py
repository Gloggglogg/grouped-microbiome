"""Logistic kernel scores with independent Gaussian multipliers per group.

For W=diag(mu*(1-mu)), H=X'WX and P=I-X H^-1 X'W, the statistic is
Q=(P'r)' K (P'r). With A[g,i]=r[i] for observations in group g, the
multiplier matrix is M=A P K P' A'. Its eigenvalues define the conditional
Gaussian reference. Hypothesis-test calibration is asymptotic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import linprog
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit, logit
from scipy.stats import binomtest


EPS = np.finfo(np.float64).eps
DEFAULT_SCORE_TOLERANCE = 1e-10
DEFAULT_MAX_ITERATIONS = 200
PSD_TOLERANCE_FACTOR = 64.0


class NumericalFailure(ValueError):
    """The requested calculation cannot support inference without assessment."""


class SeparationError(NumericalFailure):
    """An ordinary logistic maximum likelihood estimate is not finite."""


@dataclass
class NullFit:
    means: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray
    basis: np.ndarray
    coefficients: np.ndarray
    linear_predictor: np.ndarray
    information: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class KernelScore:
    statistic: float
    eigenvalues: np.ndarray
    diagnostics: dict[str, Any]
    group_scores: np.ndarray | None = None
    group_quadratic: np.ndarray | None = None


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise NumericalFailure(f"{name} must be a nonempty finite matrix")
    return values


def _information(basis: np.ndarray, weights: np.ndarray):
    information = basis.T @ (weights[:, None] * basis)
    try:
        factor = cho_factor(information, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise NumericalFailure("The weighted adjustment information is not positive definite") from exc
    condition = float(np.linalg.cond(information))
    if not np.isfinite(condition) or condition > 1e12:
        raise NumericalFailure("The weighted adjustment information has lost numerical precision")
    return information, factor, condition


def _fit_state(basis: np.ndarray, outcome: np.ndarray, gamma: np.ndarray,
               firth: bool):
    eta = basis @ gamma
    means = expit(eta)
    complement = expit(-eta)
    weights = means * complement
    if not np.isfinite(eta).all() or (weights <= 0).any():
        raise NumericalFailure("Logistic fitted probabilities exceed floating-point resolution")
    residuals = np.where(outcome == 1, complement, -means)
    information, factor, condition = _information(basis, weights)
    likelihood = float(np.sum(outcome * eta - np.logaddexp(0.0, eta)))
    hat = weights * np.einsum("ij,ij->i", basis,
                            cho_solve(factor, basis.T, check_finite=False).T)
    if (hat < -1e-10).any() or (hat > 1 + 1e-10).any():
        raise NumericalFailure("Adjustment leverages fall outside their numerical bounds")
    score_residuals = residuals + hat * (0.5 - means) if firth else residuals
    score = basis.T @ score_residuals
    logdet = float(2 * np.log(np.diag(factor[0])).sum())
    objective = likelihood + (0.5 * logdet if firth else 0.0)
    return objective, score, means, residuals, weights, information, factor, condition, hat, eta


def _ordinary_separation_check(basis: np.ndarray, outcome: np.ndarray) -> None:
    # A nonzero separating direction has all signed fitted values nonnegative
    # and at least one positive. Normalizing their sum to one fixes its scale.
    signed = (2 * outcome - 1)[:, None] * basis
    certificate = linprog(np.zeros(basis.shape[1]), A_ub=-signed,
        b_ub=np.zeros(len(outcome)), A_eq=signed.sum(axis=0)[None, :],
        b_eq=np.ones(1), bounds=[(None, None)] * basis.shape[1], method="highs")
    if certificate.success:
        raise SeparationError("Ordinary logistic fitting has complete or quasi-complete separation")
    if certificate.status != 2:
        raise NumericalFailure("The ordinary logistic separation assessment did not finish")


def fit_logistic_null(design: np.ndarray, outcome: np.ndarray, *,
                      method: str = "firth", score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
                      max_iterations: int = DEFAULT_MAX_ITERATIONS) -> NullFit:
    """Fit a fixed-design logistic mean; default uses the Jeffreys penalty.

    Singular value decomposition coordinates preserve the supplied adjustment
    span and stabilize scaling.
    Rank deficiency is an error. No observation or adjustment column is removed.
    Ordinary maximum likelihood is a numerical comparator, never an automatic
    alternative selected from the observed result or separation status.
    """
    design = _matrix(design, "Adjustment design")
    outcome = np.asarray(outcome, dtype=np.float64)
    n, p = design.shape
    if outcome.shape != (n,) or not np.isfinite(outcome).all() or not np.isin(outcome, [0, 1]).all():
        raise NumericalFailure("The outcome must be a finite binary vector matching the design")
    if len(np.unique(outcome)) != 2 or n <= p:
        raise NumericalFailure("Both outcome groups and more observations than adjustment columns are required")
    if method not in {"firth", "ordinary"}:
        raise ValueError("The mean method must be firth or ordinary")
    if (not np.isfinite(score_tolerance) or score_tolerance <= 0
            or isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("Convergence settings must be positive")
    basis, singular, vt = np.linalg.svd(design, full_matrices=False)
    tolerance = EPS * max(design.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    if rank != p:
        raise NumericalFailure("Adjustment columns contain a linear redundancy at SVD precision")
    ones = np.ones(n)
    if np.linalg.norm(ones - basis @ (basis.T @ ones)) > 64 * EPS * n:
        raise NumericalFailure("The adjustment design must include an intercept in its column span")
    firth = method == "firth"
    if not firth:
        _ordinary_separation_check(basis, outcome)
    initial_mean = (outcome.sum() + 0.5) / (n + 1) if firth else outcome.mean()
    gamma = (basis.T @ ones) * logit(initial_mean)
    backtracks = 0
    accepted_step = 0.0
    for iteration in range(max_iterations + 1):
        state = _fit_state(basis, outcome, gamma, firth)
        objective, score, means, residuals, weights, information, factor, condition, hat, eta = state
        score_max = float(np.max(abs(score)))
        if score_max <= score_tolerance:
            break
        if iteration == max_iterations:
            raise NumericalFailure("Logistic adjusted score did not reach the fixed convergence tolerance")
        step = cho_solve(factor, score, check_finite=False)
        direction = float(score @ step)
        if not np.isfinite(direction) or direction <= 0:
            raise NumericalFailure("Logistic scoring did not give an improving direction")
        accepted_step = 1.0
        for _ in range(50):
            trial = gamma + accepted_step * step
            try:
                candidate = _fit_state(basis, outcome, trial, firth)
            except NumericalFailure:
                candidate = None
            # Roundoff allowance is used only at objective resolution. The
            # adjusted score remains the mandatory convergence condition.
            allowance = 32 * EPS * max(1., abs(objective))
            if candidate is not None and candidate[0] >= objective + 1e-4 * accepted_step * direction - allowance:
                gamma = trial
                break
            accepted_step *= 0.5
            backtracks += 1
        else:
            raise NumericalFailure("Logistic line search could not increase the fixed objective")
    coefficients = vt.T @ (gamma / singular)
    raw_score = basis.T @ residuals
    diagnostics = {
        "method": "firth_jeffreys" if firth else "ordinary_maximum_likelihood",
        "n": n, "cases": int(outcome.sum()), "controls": int(n - outcome.sum()),
        "design_columns": p, "design_rank": rank,
        "design_svd_tolerance": float(tolerance),
        "design_condition_number": float(singular[0] / singular[-1]),
        "weighted_information_condition_number": condition,
        "iterations": iteration, "backtracks": backtracks,
        "last_step_fraction": accepted_step, "converged": True,
        "score_tolerance": score_tolerance, "max_iterations": max_iterations,
        "adjusted_score_max_abs_basis": score_max,
        "raw_score_max_abs_basis": float(np.max(abs(raw_score))),
        "mean_min": float(means.min()), "mean_max": float(means.max()),
        "leverage_min": float(hat.min()), "leverage_max": float(hat.max()),
        "leverage_sum": float(hat.sum()), "objective_in_svd_coordinates": objective,
        "ordinary_fit_is_comparator_only": not firth,
    }
    return NullFit(means, residuals, weights, basis, coefficients, eta, information, diagnostics)


def project_features(fit: NullFit, features: np.ndarray) -> np.ndarray:
    """Compute P Phi with the weighted nuisance projection, in the given row order."""
    features = _matrix(features, "Kernel factors")
    if len(features) != len(fit.means):
        raise NumericalFailure("Kernel factors do not match the null-fit rows")
    adjustment = np.linalg.solve(fit.information,
        fit.basis.T @ (fit.weights[:, None] * features))
    return features - fit.basis @ adjustment


def project_observed_residual(fit: NullFit) -> np.ndarray:
    """Compute P' r; Firth's raw residual score need not vanish."""
    adjustment = np.linalg.solve(fit.information, fit.basis.T @ fit.residuals)
    return fit.residuals - fit.weights * (fit.basis @ adjustment)


def _symmetric(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = _matrix(matrix, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise NumericalFailure(f"{name} must be square")
    scale = max(float(np.max(abs(matrix))), np.finfo(float).tiny)
    tolerance = PSD_TOLERANCE_FACTOR * EPS * len(matrix) * scale
    if float(np.max(abs(matrix - matrix.T))) > tolerance:
        raise NumericalFailure(f"{name} is not symmetric to numerical precision")
    return (matrix + matrix.T) * 0.5


def _spectrum(matrix: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = _symmetric(matrix, name)
    eigenvalues = eigh(matrix, eigvals_only=True, check_finite=False, driver="evr")
    scale = max(float(np.max(abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = PSD_TOLERANCE_FACTOR * EPS * len(matrix) * scale
    if eigenvalues[0] < -tolerance:
        raise NumericalFailure(f"{name} has a negative eigenvalue beyond roundoff tolerance")
    negative = eigenvalues < 0
    positive = eigenvalues[eigenvalues > 0]
    if not len(positive):
        raise NumericalFailure(f"{name} contains no positive variation")
    return positive[::-1].copy(), {
        "dimension": len(matrix), "positive_eigenvalues": len(positive),
        "smallest_eigenvalue": float(eigenvalues[0]),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "negative_roundoff_count": int(negative.sum()),
        "negative_roundoff_absolute_sum": float(abs(eigenvalues[negative]).sum()),
        "negative_eigenvalue_tolerance": float(tolerance),
        "positive_eigenvalues_discarded": 0,
    }


def project_kernel(fit: NullFit, kernel: np.ndarray) -> np.ndarray:
    """Compute P K P' by exact low-rank adjustment, without truncating K."""
    kernel = _symmetric(kernel, "Kernel")
    if len(kernel) != len(fit.means):
        raise NumericalFailure("Kernel does not match the null-fit rows")
    t = np.linalg.solve(fit.information, fit.basis.T * fit.weights[None, :])
    tk = t @ kernel
    cross = fit.basis @ tk
    projected = kernel - cross - cross.T + fit.basis @ (tk @ t.T) @ fit.basis.T
    return (projected + projected.T) * 0.5


def group_codes(group_ids: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(group_ids, dtype=object)
    if values.shape != (n,):
        raise NumericalFailure("Group labels must match the null-fit rows")
    # Canonical grouping is used only for numerical bookkeeping. Labels carry
    # no structural assumptions beyond membership. Members share one multiplier.
    groups: dict[object, int] = {}
    codes = np.empty(n, dtype=np.int64)
    for i, raw in enumerate(values):
        value = raw.item() if isinstance(raw, np.generic) else raw
        if value is None or (isinstance(value, (float, complex)) and not np.isfinite(value)):
            raise NumericalFailure("Group labels must be observed")
        try:
            if value not in groups:
                groups[value] = len(groups)
            codes[i] = groups[value]
        except TypeError as exc:
            raise NumericalFailure("Group labels must be scalar and hashable") from exc
    if len(groups) < 2:
        raise NumericalFailure("At least two independent groups are required")
    return codes, np.bincount(codes)


def group_kernel_score(fit: NullFit, group_ids: np.ndarray, *,
                        features: np.ndarray | None = None,
                        kernel: np.ndarray | None = None) -> KernelScore:
    """Construct Q and the empirical group-covariance Gaussian reference.

    Supply either exact factors Phi, where K=Phi Phi', or the full kernel K.
    All positive eigenvalues are retained. Tiny negative roundoff values are
    recorded and clipped; larger negative eigenvalues stop the calculation.
    """
    if (features is None) == (kernel is None):
        raise ValueError("Supply exactly one of exact kernel factors or a full kernel")
    n = len(fit.means)
    codes, sizes = group_codes(group_ids, n)
    f = len(sizes)
    efficient_residual = project_observed_residual(fit)
    input_diagnostics: dict[str, Any] = {}
    group_scores = group_quadratic = None
    if features is not None:
        features = _matrix(features, "Kernel factors")
        projected = project_features(fit, features)
        if np.linalg.norm(projected) <= 64 * EPS * n * max(np.linalg.norm(features), np.finfo(float).tiny):
            raise NumericalFailure("Kernel factors have no variation beyond the adjustment span")
        group_scores = np.zeros((f, features.shape[1]))
        np.add.at(group_scores, codes, fit.residuals[:, None] * projected)
        covariance = group_scores.T @ group_scores
        eigenvalues, eigen_diagnostics = _spectrum(covariance, "Group score covariance")
        score = efficient_residual @ features
        observed = float(score @ score)
        raw_score = fit.residuals @ features
        raw_q = float(raw_score @ raw_score)
        aggregate_q = float(np.square(group_scores.sum(axis=0)).sum())
        form = "exact_kernel_factor"
    else:
        kernel = _symmetric(kernel, "Kernel")
        if kernel.shape != (n, n):
            raise NumericalFailure("Kernel does not match the null-fit rows")
        _, input_diagnostics = _spectrum(kernel, "Input kernel")
        projected = project_kernel(fit, kernel)
        if np.linalg.norm(projected) <= 64 * EPS * n * max(np.linalg.norm(kernel), np.finfo(float).tiny):
            raise NumericalFailure("Kernel has no variation beyond the adjustment span")
        row_sums = np.zeros((f, n))
        np.add.at(row_sums, codes, fit.residuals[:, None] * projected)
        group_quadratic = np.zeros((f, f))
        np.add.at(group_quadratic.T, codes, (row_sums * fit.residuals[None, :]).T)
        group_quadratic = (group_quadratic + group_quadratic.T) * 0.5
        eigenvalues, eigen_diagnostics = _spectrum(group_quadratic, "Group multiplier matrix")
        observed = float(efficient_residual @ kernel @ efficient_residual)
        raw_q = float(fit.residuals @ kernel @ fit.residuals)
        aggregate_q = float(group_quadratic.sum())
        form = "full_exact_kernel"
    if not np.isfinite(observed) or observed < 0:
        raise NumericalFailure("The observed efficient kernel score is invalid")
    if not np.isclose(observed, aggregate_q, rtol=1e-8, atol=1e-10 * max(1., observed)):
        raise NumericalFailure("Observed efficient score and summed group scores disagree")
    diagnostics = {
        "statistic": "Q=(P'r)' K (P'r)", "kernel_form": form,
        "n": n, "groups": f,
        "group_size_counts": {str(int(k)): int(np.sum(sizes == k)) for k in np.unique(sizes)},
        "group_exchange_performed": False,
        "raw_residual_quadratic": raw_q,
        "observed_group_sum_difference": observed - aggregate_q,
        "efficient_residual_nuisance_score_max": float(np.max(abs(fit.basis.T @ efficient_residual))),
        "mixture_mean": float(eigenvalues.sum()),
        "mixture_variance": float(2 * (eigenvalues ** 2).sum()),
        "eigenvalue_diagnostics": eigen_diagnostics,
        "input_kernel_eigenvalue_diagnostics": input_diagnostics,
        "null_fit": fit.diagnostics.copy(),
        "reference_scope": "Conditional Gaussian multiplier law using empirical group covariance; asymptotic hypothesis-test approximation",
    }
    return KernelScore(observed, eigenvalues, diagnostics, group_scores, group_quadratic)


def gower_from_profiles(profiles: np.ndarray, metric: str) -> np.ndarray:
    """Exact Gower kernel of square-root Jaccard or relative-abundance Bray-Curtis."""
    profiles = _matrix(profiles, "Microbial profiles")
    if metric not in {"jaccard", "braycurtis"}:
        raise ValueError("Only jaccard and braycurtis are supported")
    if metric == "jaccard":
        if not np.isin(profiles, [0, 1]).all():
            raise NumericalFailure("Jaccard profiles must encode presence or absence")
        profiles = profiles.astype(bool)
    elif (profiles < 0).any() or not np.allclose(profiles.sum(axis=1), 1, rtol=1e-10, atol=1e-12):
        raise NumericalFailure("Bray-Curtis profiles must be nonnegative relative abundances summing to one")
    if (profiles.sum(axis=1) <= 0).any():
        raise NumericalFailure("Microbial profiles must not contain empty rows")
    distances = np.sqrt(squareform(pdist(profiles, metric=metric)))
    squared = distances ** 2
    return -0.5 * (squared - squared.mean(axis=1, keepdims=True)
                   - squared.mean(axis=0, keepdims=True) + squared.mean())


def gaussian_multiplier_pvalue(statistic: float, eigenvalues: np.ndarray, *,
                                draws: int, seed: int,
                                batch_size: int = 4096) -> dict[str, Any]:
    """Sample the exact conditional Gaussian quadratic-form law by eigenvalues.

    The add-one Monte Carlo tail estimate and its resolution are recorded. This
    is a multiplier reference calculation, not a permutation or exact-size test.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or not len(eigenvalues) or not np.isfinite(eigenvalues).all() or (eigenvalues <= 0).any():
        raise NumericalFailure("The reference requires positive finite eigenvalues")
    if not np.isfinite(statistic) or statistic < 0:
        raise NumericalFailure("The observed kernel score must be finite and nonnegative")
    if (isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws < 1
            or isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("Draw and batch counts must be positive integers")
    rng = np.random.default_rng(seed)
    exceed = finite = 0
    for start in range(0, draws, batch_size):
        reference = rng.chisquare(1., size=(min(batch_size, draws - start), len(eigenvalues))) @ eigenvalues
        if not np.isfinite(reference).all():
            raise NumericalFailure("A Gaussian multiplier reference statistic is not finite")
        finite += len(reference)
        exceed += int(np.sum(reference >= statistic))
    p_value = (exceed + 1) / (draws + 1)
    ci = binomtest(exceed, draws).proportion_ci(confidence_level=0.95, method="exact")
    return {"p_value": p_value, "requested": draws, "finite": finite,
        "failed": draws - finite, "exceedances": exceed, "seed": int(seed),
        "p_value_resolution": 1 / (draws + 1),
        "monte_carlo_tail_probability_95_ci": [float(ci.low), float(ci.high)],
        "reference": "Weighted independent chi-square(1) draws, equal to the conditional Gaussian group-multiplier law",
        "finite_sample_exact_test": False}
