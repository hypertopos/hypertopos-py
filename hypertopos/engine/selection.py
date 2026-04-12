# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Submodular subset-selection primitives for the navigator layer.

Pure NumPy. No state. No I/O. Lazy-greedy facility location with the
Nemhauser-Wolsey-Fisher (1-1/e) optimality guarantee.
"""
from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_similarity_matrix",
    "lazy_greedy_facility_location",
]


def compute_similarity_matrix(
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute pairwise cosine similarity over rows of X.

    Returns: (N, N) similarity matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = X / norms
    return Xn @ Xn.T


def lazy_greedy_facility_location(
    X: NDArray[np.float64],
    K: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run lazy-greedy submodular maximisation of facility location over rows of X.

    Uses cosine similarity (hardcoded -- delta vectors are already z-scored,
    so cosine measures directional diversity which is the right objective
    for "find diverse anomaly typologies").

    Returns:
        selected_idx:       1-D int array of K row indices, in selection order
        representativeness: 1-D int array, len K, where representativeness[k]
                            is the count of population rows for which selected_idx[k]
                            is the closest selected facility

    Guarantees:
        f(selected) >= (1 - 1/e) * f(optimal)   for monotone submodular f
        sum(representativeness) == N
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if K <= 0 or N == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    K = min(K, N)
    sim = compute_similarity_matrix(X)

    # current best similarity from any selected facility (0 = no facility)
    max_sim = np.zeros(N, dtype=np.float64)
    selected: list[int] = []

    # heap entries: (negative_marginal, idx, stamp_at_compute)
    iter_count = 0
    # initial upper bound: marginal gain of adding element i to empty set
    # = sum_j max(0, sim[i,j]) - 0 = sum_j max(0, sim[i,j])
    initial_marginals = np.maximum(sim, 0.0).sum(axis=1)
    heap: list[tuple[float, int, int]] = [
        (-float(initial_marginals[i]), int(i), 0) for i in range(N)
    ]
    heapq.heapify(heap)
    in_selected = np.zeros(N, dtype=bool)

    while len(selected) < K and heap:
        neg_marginal, idx, stamp = heapq.heappop(heap)
        if in_selected[idx]:
            continue
        if stamp < iter_count:
            # stale: recompute
            new_max = np.maximum(max_sim, sim[idx])
            current_marginal = float(new_max.sum() - max_sim.sum())
            heapq.heappush(heap, (-current_marginal, idx, iter_count))
            continue
        # accept
        selected.append(idx)
        in_selected[idx] = True
        max_sim = np.maximum(max_sim, sim[idx])
        iter_count += 1

    selected_idx = np.array(selected, dtype=np.int64)
    if len(selected_idx) == 0:
        return selected_idx, np.empty(0, dtype=np.int64)
    # representativeness: assign each row to its closest selected facility
    sim_to_selected = sim[selected_idx]  # (K, N)
    assignment = np.argmax(sim_to_selected, axis=0)  # (N,)
    representativeness = np.bincount(assignment, minlength=len(selected_idx))
    return selected_idx, representativeness.astype(np.int64)
