"""Input checks shared by the array interfaces."""
from __future__ import annotations

import numpy as np

from ._kernel import NumericalFailure, _matrix


def columns(values, name: str, n: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    values = _matrix(values, name)
    if n is not None and len(values) != n:
        raise NumericalFailure(f"{name} must have the same row count as the outcome")
    return values


def binary_outcome(y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise NumericalFailure("The outcome must be a finite binary vector")
    if len(np.unique(y)) != 2:
        raise NumericalFailure("The outcome must include both zero and one")
    return y


def adjustment_design(n: int, covariates=None) -> np.ndarray:
    design = np.ones((n, 1))
    if covariates is not None:
        design = np.column_stack([design, columns(covariates, "Covariates", n)])
    return design


def check_design(design: np.ndarray) -> np.ndarray:
    if len(design) <= design.shape[1]:
        raise NumericalFailure("More observations than model columns are required")
    basis, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
    if np.sum(singular > tolerance) != design.shape[1]:
        raise NumericalFailure("The model columns contain a linear redundancy")
    return basis
