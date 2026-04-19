# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Multiple-testing correction primitives for the navigator layer.

Pure NumPy + scipy.special. No state. No I/O. Closed-form per-batch operations.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaincc

__all__ = [
    "benjamini_hochberg",
    "empirical_p_values_from_rank",
    "parametric_p_values_chi2",
    "q_values_from_p_values",
    "storey_pi0",
    "storey_q_values",
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


def parametric_p_values_chi2(
    delta_norms: NDArray[np.float64],
    df: int,
) -> NDArray[np.float64]:
    """Upper-tail p-values under the χ²(df) null for ||delta||².

    If delta_i ~ N(0, 1) iid across df dimensions (the population-relative
    construction), then ||delta||² ~ χ²(df) and the upper-tail survival
    function gives per-entity p-values that concentrate near 0 for genuine
    alternatives — unlike the rank-based p-values, which are uniform by
    construction and carry no null/alternative signal for the Storey estimator.

    Returns: p-values in (0, 1], clipped to [1e-10, 1].
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    chi2 = np.asarray(delta_norms, dtype=np.float64) ** 2
    # gammaincc(df/2, x/2) == 1 - F_χ²(df, x) (upper-tail survival function)
    p = gammaincc(df / 2.0, chi2 / 2.0)
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


def storey_pi0(
    p_values: NDArray[np.float64],
    lam: float = 0.5,
) -> float:
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    p = np.asarray(p_values, dtype=np.float64)
    n = p.shape[0]
    if n == 0:
        return 1.0
    tail_count = float(np.sum(p > lam))
    pi0 = tail_count / ((1.0 - lam) * n)
    # Floor at 1/n — prevents collapse to 0 when tail count is 0 on small n,
    # preserves q-value ranking. Matches the R qvalue package behaviour.
    return float(min(max(pi0, 1.0 / n), 1.0))


def storey_q_values(
    p_values: NDArray[np.float64],
    lam: float = 0.5,
) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=np.float64)
    n = p.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    q_bh = q_values_from_p_values(p)
    pi0 = storey_pi0(p, lam=lam)
    return np.clip(pi0 * q_bh, 0.0, 1.0)


def benjamini_hochberg(
    p_values: NDArray[np.float64],
    alpha: float,
    method: str = "bh",
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Apply BH-family FDR procedure at level alpha.

    method: "bh" (default) — Benjamini-Hochberg, assumes π₀ = 1.
            "storey" — BH scaled by Storey π̂₀, recovers power when π₀ < 1.

    Guarantees: E[FDR | rejected] <= alpha
        - under independence or PRDS for method="bh"
        - under independence for method="storey" (Storey 2002)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method not in ("bh", "storey"):
        raise ValueError(f"method must be 'bh' or 'storey', got {method!r}")
    p_values = np.asarray(p_values, dtype=np.float64)
    n = p_values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float64)
    q_values = q_values_from_p_values(p_values) if method == "bh" else storey_q_values(p_values)
    rejected = q_values <= alpha
    return rejected, q_values
