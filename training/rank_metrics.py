"""Tie-aware descriptive rank correlations; no model or dataset access."""
import numpy as np


def average_ranks(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional vector")
    order = np.argsort(x, kind="stable")
    ranked = np.empty(len(x), dtype=float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(x[order])) + 1, len(x)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        ranked[order[lo:hi]] = (lo + hi - 1) / 2 + 1
    return ranked


def spearman(x, y):
    """Pearson correlation of average tied ranks; undefined constants fail."""
    rx, ry = average_ranks(x), average_ranks(y)
    if len(rx) != len(ry) or len(rx) < 2:
        raise ValueError("Expected equally sized vectors with at least two values")
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        raise ValueError("Rank correlation is undefined for a constant vector")
    return float(np.corrcoef(rx, ry)[0, 1])
