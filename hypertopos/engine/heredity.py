# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Geometric heredity — expected-position and novelty scoring.

Pure functions that compare an entity's actual delta vector against
the (weighted) mean of its graph neighbors' deltas. The L2 distance
between actual and expected is the *novelty score*: high values mean
the entity's geometric position cannot be explained by its
neighborhood, flagging cold-start entities, data-quality issues,
or genuinely novel behavior.
"""

from __future__ import annotations

import numpy as np


def compute_expected_delta(
    neighbor_deltas: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute expected delta as (weighted) mean of neighbor deltas.

    Parameters
    ----------
    neighbor_deltas:
        Shape ``(N, D)`` float32 matrix of neighbor delta vectors.
    weights:
        Optional ``(N,)`` float32 weight per neighbor. Normalised
        internally so they sum to 1.

    Returns
    -------
    np.ndarray
        Shape ``(D,)`` float32 expected delta. Returns zeros when
        *neighbor_deltas* is empty.
    """
    if neighbor_deltas.shape[0] == 0:
        d = neighbor_deltas.shape[1] if neighbor_deltas.ndim == 2 else 0
        return np.zeros(d, dtype=np.float32)

    if weights is not None:
        w = weights.astype(np.float32)
        w_sum = w.sum()
        if w_sum == 0.0:
            return np.zeros(neighbor_deltas.shape[1], dtype=np.float32)
        return np.asarray(
            np.average(neighbor_deltas, axis=0, weights=w),
            dtype=np.float32,
        )

    return np.mean(neighbor_deltas, axis=0).astype(np.float32)


def compute_novelty_score(
    actual_delta: np.ndarray,
    expected_delta: np.ndarray,
) -> float:
    """L2 distance between actual and expected delta.

    Parameters
    ----------
    actual_delta:
        Shape ``(D,)`` entity's current delta vector.
    expected_delta:
        Shape ``(D,)`` neighbor-derived expected delta.

    Returns
    -------
    float
        Euclidean distance (>= 0).
    """
    return float(np.linalg.norm(actual_delta - expected_delta))


def compute_novelty_decomposition(
    actual_delta: np.ndarray,
    expected_delta: np.ndarray,
    dimension_names: list[str],
) -> list[dict]:
    """Per-dimension decomposition sorted by |deviation| descending.

    Parameters
    ----------
    actual_delta:
        Shape ``(D,)`` entity's current delta vector.
    expected_delta:
        Shape ``(D,)`` neighbor-derived expected delta.
    dimension_names:
        Human-readable names for each dimension.

    Returns
    -------
    list[dict]
        Each entry: ``{dimension, expected, actual, deviation}``.
    """
    deviations = actual_delta - expected_delta
    result = []
    for i, name in enumerate(dimension_names):
        result.append({
            "dimension": name,
            "expected": round(float(expected_delta[i]), 6),
            "actual": round(float(actual_delta[i]), 6),
            "deviation": round(float(deviations[i]), 6),
        })
    result.sort(key=lambda r: abs(r["deviation"]), reverse=True)
    return result
