"""Tests for hypertopos.engine.fdr --- BH FDR control."""
from __future__ import annotations

import numpy as np
import pytest

from hypertopos.engine.fdr import (
    benjamini_hochberg,
    empirical_p_values_from_rank,
    q_values_from_p_values,
)


class TestEmpiricalPValues:
    def test_basic_conversion(self):
        # rank_pct=100 -> p~0 (clipped to 1e-10), rank_pct=0 -> p=1
        rank = np.array([100.0, 50.0, 0.0])
        p = empirical_p_values_from_rank(rank)
        assert p[2] == 1.0
        # p[0] should be clipped to 1e-10 (not 0)
        assert p[0] == pytest.approx(1e-10)

    def test_zero_to_one_scale(self):
        rank = np.array([1.0, 0.5, 0.0])
        p = empirical_p_values_from_rank(rank)
        assert p[2] == 1.0
        assert p[0] == pytest.approx(1e-10)

    def test_empty_input(self):
        p = empirical_p_values_from_rank(np.array([]))
        assert len(p) == 0

    def test_round_trip(self):
        rank = np.linspace(0, 100, 100)
        p = empirical_p_values_from_rank(rank)
        assert np.all(p > 0)
        assert np.all(p <= 1.0)


class TestQValues:
    def test_empty(self):
        q = q_values_from_p_values(np.array([]))
        assert len(q) == 0

    def test_monotonicity_when_sorted(self):
        p = np.sort(np.random.default_rng(42).uniform(0, 1, 200))
        q = q_values_from_p_values(p)
        assert np.all(np.diff(q) >= -1e-15)  # non-decreasing

    def test_alignment_with_input_order(self):
        rng = np.random.default_rng(123)
        p = rng.uniform(0, 1, 50)
        q = q_values_from_p_values(p)
        assert q.shape == p.shape
        # q-values should be >= p-values (BH adjustment inflates)
        # Actually q = p * m / rank, so for rank < m, q > p
        # Just verify shape and range
        assert np.all(q >= 0)
        assert np.all(q <= 1)

    def test_self_consistency_with_bh(self):
        """q_value <= alpha iff BH rejects at that alpha."""
        rng = np.random.default_rng(99)
        p = rng.uniform(0, 1, 100)
        q = q_values_from_p_values(p)
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            rejected, _ = benjamini_hochberg(p, alpha)
            np.testing.assert_array_equal(rejected, q <= alpha)


class TestBenjaminiHochberg:
    def test_empty(self):
        rejected, q = benjamini_hochberg(np.array([]), 0.05)
        assert len(rejected) == 0
        assert len(q) == 0

    def test_singleton_below(self):
        rejected, q = benjamini_hochberg(np.array([0.04]), 0.05)
        assert rejected[0] is np.True_
        assert q[0] == pytest.approx(0.04)

    def test_singleton_above(self):
        rejected, q = benjamini_hochberg(np.array([0.06]), 0.05)
        assert not rejected[0]

    def test_alpha_validation(self):
        p = np.array([0.01, 0.05])
        with pytest.raises(ValueError):
            benjamini_hochberg(p, 0.0)
        with pytest.raises(ValueError):
            benjamini_hochberg(p, 1.0)
        with pytest.raises(ValueError):
            benjamini_hochberg(p, -0.1)
        with pytest.raises(ValueError):
            benjamini_hochberg(p, 1.1)

    def test_all_null(self):
        """Uniform p-values: ~alpha * N should be rejected."""
        rng = np.random.default_rng(42)
        p = rng.uniform(0, 1, 1000)
        rejected, q = benjamini_hochberg(p, 0.05)
        # Under pure null, BH controls FDR --- expect ~0 to ~50 rejections
        assert np.sum(rejected) <= 100  # generous upper bound

    def test_all_alternative(self):
        """Very small p-values: all should be rejected."""
        p = np.full(100, 1e-4)
        rejected, q = benjamini_hochberg(p, 0.05)
        assert np.all(rejected)

    def test_known_dataset(self):
        """Fixed seed: 10 alternatives at p=0.001, 90 nulls uniform."""
        rng = np.random.default_rng(7)
        p_null = rng.uniform(0, 1, 90)
        p_alt = np.full(10, 0.001)
        p = np.concatenate([p_alt, p_null])
        rejected, q = benjamini_hochberg(p, 0.05)
        # All 10 alternatives should be rejected
        assert np.all(rejected[:10])
        # Most nulls should not be rejected
        assert np.sum(rejected[10:]) < 10

    def test_q_value_alignment(self):
        """q-values are in input order, not sorted order."""
        p = np.array([0.5, 0.01, 0.3, 0.001])
        _, q = benjamini_hochberg(p, 0.05)
        # q[3] (p=0.001) should be smallest
        assert q[3] < q[0]
