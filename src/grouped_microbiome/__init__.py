"""Diversity and association methods for arrays of grouped observations."""
from ._kernel import (
    KernelScore, NullFit, NumericalFailure, SeparationError,
    fit_logistic_null, gaussian_multiplier_pvalue, gower_from_profiles,
    group_kernel_score,
)
from .association import (
    KernelTestResult, LogisticAssociation, fit_logistic_association, kernel_test,
)
from .diversity import alpha_diversity
from .multiplicity import adjust_pvalues

__version__ = "0.1.0"
__all__ = [
    "KernelScore", "KernelTestResult", "LogisticAssociation", "NullFit",
    "NumericalFailure", "SeparationError", "adjust_pvalues", "alpha_diversity",
    "fit_logistic_association", "fit_logistic_null", "gaussian_multiplier_pvalue",
    "gower_from_profiles", "group_kernel_score", "kernel_test",
]
