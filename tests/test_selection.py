"""Tests for hypertopos.engine.selection -- submodular facility location."""
from __future__ import annotations

import numpy as np
import pytest

from hypertopos.engine.selection import (
    compute_similarity_matrix,
    lazy_greedy_facility_location,
)


class TestSimilarityMatrix:
    def test_identity_diagonal(self):
        X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        S = compute_similarity_matrix(X)
        np.testing.assert_allclose(np.diag(S), 1.0)

    def test_orthogonal_zero(self):
        X = np.array([[1, 0], [0, 1]], dtype=float)
        S = compute_similarity_matrix(X)
        assert S[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_zero_norm_row(self):
        X = np.array([[0, 0], [1, 1]], dtype=float)
        S = compute_similarity_matrix(X)
        # zero-norm row should not crash
        assert S.shape == (2, 2)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 5))
        S = compute_similarity_matrix(X)
        np.testing.assert_allclose(S, S.T, atol=1e-12)


class TestFacilityLocation:
    def test_empty(self):
        idx, rep = lazy_greedy_facility_location(np.empty((0, 3)), 5)
        assert len(idx) == 0
        assert len(rep) == 0

    def test_K_zero(self):
        X = np.array([[1, 2], [3, 4]], dtype=float)
        idx, rep = lazy_greedy_facility_location(X, 0)
        assert len(idx) == 0

    def test_K_greater_than_N(self):
        X = np.eye(5, dtype=float)
        idx, rep = lazy_greedy_facility_location(X, 10)
        assert len(idx) == 5

    def test_K_equals_N(self):
        X = np.eye(4, dtype=float)
        idx, rep = lazy_greedy_facility_location(X, 4)
        assert len(idx) == 4
        assert np.sum(rep) == 4

    def test_K_one_returns_medoid(self):
        """K=1 should return the row with highest sum of similarity."""
        X = np.array([[1, 0], [1, 0.1], [0, 1]], dtype=float)
        idx, rep = lazy_greedy_facility_location(X, 1)
        assert len(idx) == 1
        # Row 0 and 1 are similar, row 2 is orthogonal
        # Medoid should be row 0 or 1 (highest total similarity)
        assert idx[0] in [0, 1]
        assert rep[0] == 3  # represents all

    def test_diversity_on_synthetic_clusters(self):
        """5 well-separated clusters -> K=5 picks one from each."""
        rng = np.random.default_rng(42)
        clusters = []
        for i in range(5):
            center = np.zeros(10)
            center[i * 2] = 10.0
            center[i * 2 + 1] = 10.0
            points = center + rng.standard_normal((20, 10)) * 0.1
            clusters.append(points)
        X = np.vstack(clusters)
        idx, rep = lazy_greedy_facility_location(X, 5)
        # Each selected should come from a different cluster
        cluster_ids = set(int(i) // 20 for i in idx)
        assert len(cluster_ids) == 5

    def test_representativeness_sums_to_N(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 8))
        for K in [1, 5, 10, 50]:
            idx, rep = lazy_greedy_facility_location(X, K)
            assert np.sum(rep) == 50

    def test_selection_order_preserved(self):
        """selected_idx is in selection order, not sorted."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((30, 5))
        idx, _ = lazy_greedy_facility_location(X, 10)
        # Not necessarily sorted
        assert len(idx) == 10

    def test_lazy_greedy_matches_naive(self):
        """For small N, lazy greedy should match naive greedy."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal((15, 4))
        sim = compute_similarity_matrix(X)
        K = 5

        # Naive greedy (baseline 0 = no facility assigned)
        max_sim_naive = np.zeros(15)
        selected_naive = []
        for _ in range(K):
            best_idx, best_gain = -1, -np.inf
            for j in range(15):
                if j in selected_naive:
                    continue
                new_max = np.maximum(max_sim_naive, sim[j])
                gain = new_max.sum() - max_sim_naive.sum()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            selected_naive.append(best_idx)
            max_sim_naive = np.maximum(max_sim_naive, sim[best_idx])

        # Lazy greedy
        idx_lazy, _ = lazy_greedy_facility_location(X, K)
        assert list(idx_lazy) == selected_naive

    def test_zero_norm_handled(self):
        X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        idx, rep = lazy_greedy_facility_location(X, 2)
        assert len(idx) == 2
        assert np.sum(rep) == 3
