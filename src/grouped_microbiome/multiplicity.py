"""Multiple-testing adjustments for a caller-defined set of tests."""
from __future__ import annotations

import numpy as np


def adjust_pvalues(pvalues, method: str = "bh") -> np.ndarray:
    """Return adjusted p-values in input order using 'bh' or 'holm'.

    The supplied one-dimensional vector defines the entire correction family.
    Benjamini-Hochberg ('bh') targets the false discovery rate under independence
    or suitable positive dependence. Holm controls the family-wise error rate.
    Values must be finite and between zero and one. An empty vector is accepted.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("P-values must be a finite vector with values between zero and one")
    if method not in {"bh", "holm"}:
        raise ValueError("The adjustment method must be 'bh' or 'holm'")
    m = len(pvalues)
    if m == 0:
        return pvalues.copy()
    order = np.argsort(pvalues, kind="stable")
    ordered = pvalues[order]
    if method == "bh":
        adjusted = np.minimum.accumulate((ordered * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        adjusted = np.maximum.accumulate(ordered * np.arange(m, 0, -1))
    output = np.empty(m)
    output[order] = np.minimum(adjusted, 1.0)
    return output
