"""Ordinary logistic associations and a grouped kernel test."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from ._kernel import (
    KernelScore, NullFit, NumericalFailure, SeparationError,
    _ordinary_separation_check, fit_logistic_null, gaussian_multiplier_pvalue,
    group_codes, group_kernel_score,
)
from ._validation import adjustment_design, binary_outcome, check_design, columns


@dataclass
class LogisticAssociation:
    """All model terms in intercept, X-column, covariate-column order."""

    terms: tuple[str, ...]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals: np.ndarray
    p_values: np.ndarray
    odds_ratios: np.ndarray
    odds_ratio_confidence_intervals: np.ndarray
    covariance: np.ndarray
    diagnostics: dict[str, Any]


def fit_logistic_association(X, y, *, covariates=None, groups=None,
                             confidence_level: float = 0.95) -> LogisticAssociation:
    """Fit ordinary logistic regression with optional group-robust covariance.

    X contains one or more jointly fitted predictors, one sample per row.
    A one-dimensional X denotes one predictor. Covariates are numeric columns;
    callers encode categories and choose scaling. An intercept is added. Rows
    are retained in supplied order and every input must be observed and finite.
    With groups, use the cluster sandwich covariance, its small-sample correction,
    and normal Wald intervals. Without groups, use model-based covariance.
    """
    y = binary_outcome(y)
    X = columns(X, "Predictors", len(y))
    adjustment = adjustment_design(len(y), covariates)
    design = np.column_stack([adjustment[:, :1], X, adjustment[:, 1:]])
    basis = check_design(design)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("The confidence level must be between zero and one")
    _ordinary_separation_check(basis, y)
    n_groups = None
    kwargs: dict[str, Any] = {"use_t": False}
    if groups is not None:
        codes, sizes = group_codes(groups, len(y))
        n_groups = len(sizes)
        kwargs.update(cov_type="cluster", cov_kwds={"groups": codes, "use_correction": True})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            fitted = sm.GLM(y, design, family=sm.families.Binomial()).fit(**kwargs)
    except PerfectSeparationWarning as exc:
        raise SeparationError("Ordinary logistic fitting detected separation") from exc
    if not fitted.converged:
        raise NumericalFailure("Ordinary logistic fitting did not converge")
    coefficients = np.asarray(fitted.params)
    covariance = np.asarray(fitted.cov_params())
    intervals = np.asarray(fitted.conf_int(alpha=1 - confidence_level))
    standard_errors = np.asarray(fitted.bse)
    p_values = np.asarray(fitted.pvalues)
    if not all(np.isfinite(v).all() for v in (coefficients, covariance, intervals, standard_errors, p_values)):
        raise NumericalFailure("The fitted model has nonfinite coefficients or uncertainty estimates")
    if (standard_errors <= 0).any():
        raise NumericalFailure("The fitted covariance gives a nonpositive standard error")
    with np.errstate(over="ignore"):
        odds_ratios = np.exp(coefficients)
        odds_intervals = np.exp(intervals)
    if (not np.isfinite(odds_ratios).all() or not np.isfinite(odds_intervals).all()
            or (odds_ratios <= 0).any() or (odds_intervals <= 0).any()):
        raise NumericalFailure("Odds ratios or their interval bounds exceed floating-point resolution")
    terms = ("intercept", *(f"x{i}" for i in range(X.shape[1])),
             *(f"covariate{i}" for i in range(adjustment.shape[1] - 1)))
    return LogisticAssociation(terms, coefficients, standard_errors, intervals,
        p_values, odds_ratios, odds_intervals, covariance, {
            "n": len(y), "groups": n_groups, "converged": True,
            "covariance": "cluster" if groups is not None else "model_based",
            "small_sample_correction": groups is not None,
            "wald_reference": "normal", "confidence_level": confidence_level,
            "predictor_scaling": "supplied_units", "intercept_added": True,
        })


@dataclass
class KernelTestResult:
    """Observed score, fitted null mean, and Monte Carlo reference diagnostics."""

    score: KernelScore
    null_fit: NullFit
    reference: dict[str, Any]

    @property
    def p_value(self) -> float:
        return self.reference["p_value"]


def kernel_test(X, y, *, groups, covariates=None, draws: int = 9999,
                seed: int = 0) -> KernelTestResult:
    """Test supplied exact kernel factors X with a Firth logistic null mean.

    X is a finite sample-by-factor matrix defining K=X X'. Covariates are
    numeric adjustment columns; an intercept is added. Group members share
    one Gaussian multiplier. The conditional reference is sampled using all
    positive eigenvalues. Hypothesis-test validity is asymptotic and depends
    on the design and adequate independent group information.
    """
    y = binary_outcome(y)
    X = columns(X, "Kernel factors", len(y))
    design = adjustment_design(len(y), covariates)
    fitted = fit_logistic_null(design, y)
    score = group_kernel_score(fitted, groups, features=X)
    reference = gaussian_multiplier_pvalue(score.statistic, score.eigenvalues,
                                            draws=draws, seed=seed)
    return KernelTestResult(score, fitted, reference)
