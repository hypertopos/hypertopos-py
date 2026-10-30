# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Drift tests of new rows against a frozen reference, from counts that merge by addition.

The executors count; the driver sums the counts and tests them. The p-value test
is a Kolmogorov-Smirnov test of the rows' conformal p-values against uniform,
read from a histogram: the supremum over the bin edges plus one bin width, never
below the exact statistic. The per-dimension test compares the rows' values with
the reference's CDF sketch at its grid values, tie-correct: at a value carried
by several grid points the new rows' share below it is compared with the first
level there and their share at or below it with the last.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import betabinom, kstwo

P_BINS = 4096
P_EDGES = np.arange(P_BINS + 1) / P_BINS


def p_histogram(p: np.ndarray) -> np.ndarray:
    """Counts of ``p`` in the ``P_BINS`` bins ``[k/B, (k+1)/B)``, the last closed at 1."""
    bins = np.clip(np.searchsorted(P_EDGES, p, side="right") - 1, 0, P_BINS - 1)
    return np.bincount(bins, minlength=P_BINS).astype(np.int64)


def uniform_ks_binned(hist: np.ndarray) -> float:
    """KS statistic of the histogrammed p-values against uniform, an upper bound."""
    cdf = np.r_[0, np.cumsum(hist)] / max(int(hist.sum()), 1)
    return float(min(1.0, np.max(np.abs(cdf - P_EDGES)) + 1.0 / P_BINS))


def grid_levels(grid: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique grid values of one dimension with the first and last level stored at each."""
    values, first = np.unique(grid, return_index=True)
    last = np.r_[first[1:], len(grid)] - 1
    return values, levels[first], levels[last]


def grid_counts(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Counts of ``x`` below, at and between the unique grid ``values``, interleaved.

    Index ``2j`` counts the open interval just below ``values[j]`` (index ``2m``
    the one above the last value) and ``2j + 1`` the rows equal to ``values[j]``.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    j = np.searchsorted(values, x, side="left")
    at = j < len(values)
    at[at] = values[j[at]] == x[at]
    return np.bincount(2 * j + at, minlength=2 * len(values) + 1).astype(np.int64)


def grid_ks(counts: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Tie-correct KS statistic of the counted rows against the sketch levels."""
    cdf = np.cumsum(counts) / max(int(counts.sum()), 1)
    below, at_or_below = cdf[0:-1:2], cdf[1::2]
    return float(max(np.max(np.abs(below - low)), np.max(np.abs(at_or_below - high))))


def effective_n(n_fitted: int, n_new: int) -> int:
    """Two-sample effective size: the reference is itself a sample of ``n_fitted`` rows."""
    return max(1, int(n_fitted * n_new / (n_fitted + n_new)))


def ks_p(stat: float, n_eff: int) -> float:
    return float(kstwo.sf(stat, n_eff))


def flag_count_p(flagged: int, rows: int, n_fitted: int, percentile: float) -> float | None:
    """One-sided tail of the flagged count, beta-binomial under one shared calibration.

    None when no new row can be flagged at ``percentile``.
    """
    # A new row is flagged when it ranks among the top ``a`` of the n + 1.
    a = int(np.floor((1.0 - percentile / 100.0) * (n_fitted + 1)))
    if a < 1:
        return None
    return float(betabinom.sf(flagged - 1, rows, a, n_fitted + 1 - a))
