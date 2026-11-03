"""Within-sample diversity from a nonnegative count matrix."""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy

from ._kernel import NumericalFailure, _matrix


def alpha_diversity(counts) -> dict[str, np.ndarray]:
    """Return observed richness, Shannon entropy, and inverse Simpson diversity.

    Rows are samples and columns are taxa. Nonnegative counts or weights are
    accepted. Each row must have a positive finite total. Shannon entropy uses
    natural logarithms. Counts are normalized within each sample for Shannon
    and inverse Simpson calculations; richness counts strictly positive entries.
    """
    counts = _matrix(counts, "Counts")
    if (counts < 0).any():
        raise NumericalFailure("Counts must be nonnegative")
    totals = counts.sum(axis=1)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise NumericalFailure("Every sample must have a positive finite count total")
    proportions = counts / totals[:, None]
    return {
        "richness": np.count_nonzero(counts, axis=1),
        "shannon": -np.sum(xlogy(proportions, proportions), axis=1),
        "inverse_simpson": 1.0 / np.sum(proportions ** 2, axis=1),
    }
