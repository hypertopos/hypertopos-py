# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Multiple-testing correction primitives for the navigator layer.

Pure NumPy. No state. No I/O. Closed-form per-batch operations on numeric arrays.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "benjamini_hochberg",
    "empirical_p_values_from_rank",
    "q_values_from_p_values",
]


def empirical_p_values_from_rank(
    delta_rank_pct: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert delta rank percentile (0-100 or 0-1) into empirical-null p-values.

    delta_rank_pct[i] is the entity's position in the population sorted by
    delta_norm ascending --- 100 (or 1.0) is the most extreme entity.
    p-value = 1 - rank_pct/100 (if input > 1, assume 0-100 scale).

    Returns: 1-D array of p-values in (0, 1].
    """
    p = np.asarray(delta_rank_pct, dtype=np.float64)
    # Auto-detect scale: if max > 1, assume 0-100 range
    if p.size > 0 and np.max(p) > 1.0:
        p = p / 100.0
    p = 1.0 - p
    # clip into (eps, 1) — use a fixed floor (not 1/N) because
    # delta_rank_pct is computed against the FULL population, so
    # the input array may be a small top-N subset with very small p-values.
    return np.clip(p, 1e-10, 1.0)


def q_values_from_p_values(
    p_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute BH q-value for each input p-value.

    q-value of entity i = minimum FDR alpha at which i would be rejected.
    Implementation: right-to-left running minimum of (p_(j) * m / j).

    Returns: 1-D array of q-values in [0, 1] aligned with input order.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    n = p_values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    raw_q_sorted = sorted_p * n / ranks
    # right-to-left cumulative min keeps q monotonic
    q_sorted = np.minimum.accumulate(raw_q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_values = np.empty(n, dtype=np.float64)
    q_values[order] = q_sorted
    return q_values


def benjamini_hochberg(
    p_values: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Apply BH procedure at level alpha.

    Returns:
        rejected: 1-D bool array, True for entities passing BH cutoff
        q_values: 1-D float array of per-entity q-values

    Guarantees: E[FDR | rejected] <= alpha (under independence or PRDS)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    p_values = np.asarray(p_values, dtype=np.float64)
    n = p_values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float64)
    q_values = q_values_from_p_values(p_values)
    rejected = q_values <= alpha
    return rejected, q_values
