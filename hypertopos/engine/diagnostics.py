# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Build-time statistical diagnostics over population-level arrays.

Pure NumPy + scipy.stats primitives — no I/O, no storage scan, no
navigator dependency. Functions in this module operate on the
``(delta_norms, group_ids)`` pair already materialised by the
calibration pass and return JSON-serialisable dicts the builder can
persist on the pattern node.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import levene

MIN_GROUP_SIZE = 30


def levene_test_per_group(
    values: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, Any]:
    """Brown-Forsythe (median-centred Levene) test for equal variance.

    Tests H0: ``Var(values | group_id = g_1) = ... = Var(values | group_id = g_k)``
    across the groups present in ``group_ids``. Uses
    ``scipy.stats.levene(..., center='median')`` — the Brown-Forsythe
    variant, more robust for skewed distributions than mean-centred
    Levene. Groups with fewer than ``MIN_GROUP_SIZE`` entities are
    dropped silently and counted in ``skipped_groups_low_n`` — Levene's
    test has low power on small samples and produces unreliable W
    statistics.

    Args:
        values: 1-D float array of per-entity values (typically
            ``delta_norms`` against global mu/sigma).
        group_ids: 1-D array of group labels, same length as ``values``.
            Castable to string via ``np.asarray(..., dtype=str)``.

    Returns:
        Dict with:

        - ``W_statistic`` (float | None): Brown-Forsythe W. None if
          fewer than two qualifying groups survive the low-N filter.
        - ``p_value`` (float | None): p-value under F(k-1, N-k). None
          under the same condition as ``W_statistic``.
        - ``k_groups`` (int): number of qualifying groups (≥ MIN_GROUP_SIZE).
        - ``per_group_variance`` (dict[str, float]): variance of
          ``values`` within each qualifying group, keyed by group id.
        - ``per_group_n`` (dict[str, int]): entity count per qualifying
          group.
        - ``skipped_groups_low_n`` (int): groups dropped due to N <
          MIN_GROUP_SIZE.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    group_arr = np.asarray(group_ids, dtype=str)
    if values_arr.shape != group_arr.shape:
        raise ValueError(
            f"values and group_ids must have the same shape; "
            f"got {values_arr.shape} and {group_arr.shape}"
        )

    unique_groups, counts = np.unique(group_arr, return_counts=True)
    qualifying_mask = counts >= MIN_GROUP_SIZE
    qualifying_groups = unique_groups[qualifying_mask]
    skipped = int((~qualifying_mask).sum())

    per_group_variance: dict[str, float] = {}
    per_group_n: dict[str, int] = {}
    samples: list[np.ndarray] = []
    for gid in qualifying_groups:
        mask = group_arr == gid
        sample = values_arr[mask]
        per_group_variance[str(gid)] = float(np.var(sample, ddof=1))
        per_group_n[str(gid)] = int(sample.size)
        samples.append(sample)

    if len(samples) < 2:
        return {
            "W_statistic": None,
            "p_value": None,
            "k_groups": len(samples),
            "per_group_variance": per_group_variance,
            "per_group_n": per_group_n,
            "skipped_groups_low_n": skipped,
        }

    w_stat, p_value = levene(*samples, center="median")
    # Levene returns NaN when every qualifying group has zero residual
    # variance (degenerate). Surface as None so the persisted JSON is
    # well-formed (NaN literals are not valid JSON) and downstream
    # consumers can treat "computed-but-degenerate" the same way as
    # "fewer than two groups".
    w_out = float(w_stat) if np.isfinite(w_stat) else None
    p_out = float(p_value) if np.isfinite(p_value) else None
    return {
        "W_statistic": w_out,
        "p_value": p_out,
        "k_groups": len(samples),
        "per_group_variance": per_group_variance,
        "per_group_n": per_group_n,
        "skipped_groups_low_n": skipped,
    }
