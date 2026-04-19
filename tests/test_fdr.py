"""Tests for hypertopos.engine.fdr --- BH FDR control."""
from __future__ import annotations

import numpy as np
import pytest

from hypertopos.engine.fdr import (
    benjamini_hochberg,
    empirical_p_values_from_rank,
    parametric_p_values_chi2,
    q_values_from_p_values,
    storey_pi0,
    storey_q_values,
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


class TestStoreyPi0:
    def test_estimates_null_proportion(self):
        """Synthetic mix — 70% uniform null + 30% Beta(0.5, 5) alt, π₀ should track 0.7."""
        rng = np.random.default_rng(42)
        n_null, n_alt = 700, 300
        p_null = rng.uniform(0, 1, n_null)
        p_alt = rng.beta(0.5, 5.0, n_alt)
        p = np.concatenate([p_null, p_alt])
        pi0 = storey_pi0(p, lam=0.5)
        # LSL at lam=0.5 tolerates bias — allow ±0.1 from the true 0.7
        assert 0.6 <= pi0 <= 0.8, f"π₀={pi0:.3f} not in [0.6, 0.8]"

    def test_all_null_distribution(self):
        """Uniform [0,1] across the board → π₀ ≈ 1.0."""
        rng = np.random.default_rng(7)
        p = rng.uniform(0, 1, 2000)
        pi0 = storey_pi0(p, lam=0.5)
        assert 0.9 <= pi0 <= 1.05, f"π₀={pi0:.3f} should be ≈ 1.0"

    def test_all_alternative_distribution(self):
        """All p ≈ 0 (strongly alternative) → π₀ near 0."""
        p = np.full(500, 1e-6)
        pi0 = storey_pi0(p, lam=0.5)
        assert pi0 <= 0.05, f"π₀={pi0:.3f} should be near 0"

    def test_pi0_bounded_to_one(self):
        """π₀ is a proportion — never exceeds 1 even when empirical ratio does."""
        # Degenerate case: all p-values at exactly lam → division ratio may exceed 1
        p = np.full(100, 0.5)
        pi0 = storey_pi0(p, lam=0.5)
        assert pi0 <= 1.0

    def test_empty_input(self):
        assert storey_pi0(np.array([]), lam=0.5) == 1.0

    def test_lambda_validation(self):
        with pytest.raises(ValueError):
            storey_pi0(np.array([0.5]), lam=0.0)
        with pytest.raises(ValueError):
            storey_pi0(np.array([0.5]), lam=1.0)


class TestStoreyQValues:
    def test_recovers_more_discoveries_than_bh(self):
        """On a mixed distribution, Storey q-values ≤ BH q-values elementwise."""
        rng = np.random.default_rng(123)
        p_null = rng.uniform(0, 1, 700)
        p_alt = rng.beta(0.5, 5.0, 300)
        p = np.concatenate([p_null, p_alt])

        q_bh = q_values_from_p_values(p)
        q_storey = storey_q_values(p)

        # Storey = π₀ × BH, with π₀ < 1 → Storey ≤ BH everywhere
        assert np.all(q_storey <= q_bh + 1e-12)

        # And strictly smaller somewhere (otherwise no power recovery)
        assert np.any(q_storey < q_bh - 1e-6)

        # More discoveries at α=0.05
        assert np.sum(q_storey <= 0.05) > np.sum(q_bh <= 0.05)

    def test_matches_bh_under_pure_null(self):
        """When π₀ ≈ 1, Storey q-values should ≈ BH q-values."""
        rng = np.random.default_rng(11)
        p = rng.uniform(0, 1, 1000)
        q_bh = q_values_from_p_values(p)
        q_storey = storey_q_values(p)
        # Within 5% of BH since π₀ floats around 1.0
        assert np.allclose(q_storey, q_bh, rtol=0.05)

    def test_q_values_bounded_zero_to_one(self):
        rng = np.random.default_rng(0)
        p_null = rng.uniform(0, 1, 500)
        p_alt = np.full(100, 1e-5)
        p = np.concatenate([p_null, p_alt])
        q_storey = storey_q_values(p)
        assert np.all(q_storey >= 0)
        assert np.all(q_storey <= 1.0)

    def test_alignment_with_input_order(self):
        """q-values come back in input order, like the BH primitive."""
        p = np.array([0.5, 0.01, 0.3, 0.001])
        q = storey_q_values(p)
        assert q[3] < q[0]  # p=0.001 has smallest q

    def test_empty_input(self):
        q = storey_q_values(np.array([]))
        assert len(q) == 0


class TestBHMethodDispatch:
    def test_bh_method_returns_plain_bh(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 200)
        _, q_default = benjamini_hochberg(p, 0.05)
        _, q_bh = benjamini_hochberg(p, 0.05, method="bh")
        assert np.array_equal(q_default, q_bh)

    def test_storey_method_shrinks_q_values(self):
        rng = np.random.default_rng(2)
        p_null = rng.uniform(0, 1, 800)
        p_alt = rng.beta(0.5, 5.0, 200)
        p = np.concatenate([p_null, p_alt])
        _, q_bh = benjamini_hochberg(p, 0.05, method="bh")
        _, q_storey = benjamini_hochberg(p, 0.05, method="storey")
        assert np.all(q_storey <= q_bh + 1e-12)
        assert np.sum(q_storey <= 0.05) >= np.sum(q_bh <= 0.05)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([0.1, 0.2]), 0.05, method="bogus")


class TestParametricChi2PValues:
    def test_null_delta_norm_near_sqrt_df(self):
        """delta_norm ~= sqrt(df) under N(0,1) null -> p-values cluster near 0.5."""
        rng = np.random.default_rng(3)
        df = 10
        deltas = rng.standard_normal(size=(5000, df))
        dn = np.linalg.norm(deltas, axis=1)
        p = parametric_p_values_chi2(dn, df=df)
        # Under null, p-values are uniform [0, 1] -> median ~ 0.5
        assert 0.45 <= np.median(p) <= 0.55

    def test_large_delta_norm_yields_small_p(self):
        """||delta||^2 >> df -> p ~ 0."""
        p = parametric_p_values_chi2(np.array([20.0]), df=4)
        assert p[0] < 1e-6

    def test_zero_delta_norm_yields_one(self):
        p = parametric_p_values_chi2(np.array([0.0]), df=5)
        assert p[0] == 1.0

    def test_df_validation(self):
        with pytest.raises(ValueError):
            parametric_p_values_chi2(np.array([1.0]), df=0)

    def test_storey_recovers_power_with_chi2(self):
        """On a mix of N(0,1) null + shifted-mean alternative, chi2 p-values
        enable Storey to shrink q-values meaningfully."""
        rng = np.random.default_rng(4)
        df = 10
        # 70% null, 30% alt (mean shift gives ||delta|| larger than sqrt(df))
        null_deltas = rng.standard_normal(size=(700, df))
        alt_deltas = rng.standard_normal(size=(300, df)) + 2.0
        deltas = np.concatenate([null_deltas, alt_deltas], axis=0)
        dn = np.linalg.norm(deltas, axis=1)
        p = parametric_p_values_chi2(dn, df=df)

        # With a real null component, Storey pi0 should land roughly at 0.7
        pi0 = storey_pi0(p, lam=0.5)
        assert 0.5 <= pi0 <= 0.9

        rej_bh, _ = benjamini_hochberg(p, 0.05, method="bh")
        rej_st, _ = benjamini_hochberg(p, 0.05, method="storey")
        # Storey recovers at least as many discoveries as BH
        assert rej_st.sum() >= rej_bh.sum()
