# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for cross-pattern composition via Fisher's method and harmonic-mean p."""

import math

import numpy as np
import pytest
from hypertopos.engine.composition import (
    co_dispersion,
    fisher_combine_pvalues,
    harmonic_mean_p,
    hmp_threshold_at_alpha,
)


class TestFisherCombine:
    def test_all_significant(self):
        p_values = [0.01, 0.02, 0.03]
        result = fisher_combine_pvalues(p_values)
        assert result["combined_p"] < 0.001
        assert result["chi2"] > 20
        assert result["df"] == 6

    def test_all_insignificant(self):
        p_values = [0.8, 0.7, 0.9]
        result = fisher_combine_pvalues(p_values)
        assert result["combined_p"] > 0.5

    def test_mixed_signals(self):
        p_values = [0.01, 0.5, 0.9]
        result = fisher_combine_pvalues(p_values)
        assert 0.01 < result["combined_p"] < 0.5

    def test_single_pattern(self):
        result = fisher_combine_pvalues([0.05])
        assert abs(result["combined_p"] - 0.05) < 0.02

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fisher_combine_pvalues([])

    def test_zero_pvalue_clamped(self):
        result = fisher_combine_pvalues([0.0, 0.5])
        assert result["combined_p"] < 0.01


class TestCoDispersion:
    def test_perfectly_correlated(self):
        norms_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norms_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = co_dispersion(norms_a, norms_b)
        assert result["spearman_rho"] > 0.9
        assert result["insufficient_data"] is False

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        norms_a = rng.random(100)
        norms_b = rng.random(100)
        result = co_dispersion(norms_a, norms_b)
        assert abs(result["spearman_rho"]) < 0.3

    def test_too_few_entities(self):
        result = co_dispersion(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert result["insufficient_data"] is True


class TestHarmonicMeanP:
    def test_uniform_weights_basic(self):
        # All p_i = 0.05; HMP = 0.05
        p = {"a": 0.05, "b": 0.05, "c": 0.05}
        assert harmonic_mean_p(p) == pytest.approx(0.05)

    def test_explicit_weights(self):
        # HMP = (sum w_i) / sum(w_i / p_i)
        p = {"a": 0.01, "b": 0.5}
        w = {"a": 0.7, "b": 0.3}
        expected = (0.7 + 0.3) / (0.7 / 0.01 + 0.3 / 0.5)
        assert harmonic_mean_p(p, weights=w) == pytest.approx(expected)

    def test_default_weights_uniform(self):
        p = {"a": 0.1, "b": 0.2, "c": 0.4}
        n = 3
        expected = 1.0 / sum((1.0 / n) / pi for pi in p.values())
        assert harmonic_mean_p(p) == pytest.approx(expected)

    def test_zero_p_clamped_to_floor(self):
        # p=0 should not produce inf/nan; sanitized to 1e-300
        p = {"a": 0.0, "b": 0.5}
        result = harmonic_mean_p(p)
        assert math.isfinite(result)
        assert 0.0 < result <= 1.0

    def test_inf_or_nan_p_treated_as_one(self):
        p = {"a": float("inf"), "b": float("nan"), "c": 0.5}
        result = harmonic_mean_p(p)
        assert math.isfinite(result)
        # inf/nan map to 1.0 → contribute 1/1=1, c contributes 1/0.5=2
        # HMP = 3 / (1 + 1 + 2) = 0.75
        assert result == pytest.approx(0.75)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            harmonic_mean_p({})

    def test_zero_weight_sum_returns_one(self):
        p = {"a": 0.01, "b": 0.5}
        w = {"a": 0.0, "b": 0.0}
        assert harmonic_mean_p(p, weights=w) == pytest.approx(1.0)

    def test_hmp_bounded_by_min_and_arithmetic_mean(self):
        # HMP is always in [min(p_i), arithmetic_mean(p_i)] (uniform weights).
        p = {"a": 0.05, "b": 0.04, "c": 0.06}
        result = harmonic_mean_p(p)
        ari_mean = sum(p.values()) / len(p)
        assert min(p.values()) <= result <= ari_mean

    def test_hmp_dominated_by_smallest_p(self):
        # Single very small p drags HMP toward min, never above arithmetic mean.
        p = {"a": 1e-6, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5}
        result = harmonic_mean_p(p)
        ari_mean = sum(p.values()) / len(p)
        assert result < ari_mean
        # And result must remain > min (HMP >= min for non-negative weights).
        assert result >= min(p.values()) - 1e-12

    def test_hmp_at_l5_independent_uniform(self):
        # When all p_i are uniform on (0,1) and independent, HMP is calibrated
        # by hmp_threshold_at_alpha — verify produces a valid number on a sample.
        rng = np.random.default_rng(123)
        n_trials = 2000
        L = 5
        hmps = []
        for _ in range(n_trials):
            ps = rng.uniform(1e-6, 1.0, size=L)
            hmps.append(harmonic_mean_p({f"d{i}": float(ps[i]) for i in range(L)}))
        # Sanity: median HMP roughly in (0, 1)
        med = float(np.median(hmps))
        assert 0.0 < med < 1.0

    def test_no_nan_or_inf(self):
        p = {"a": 0.0, "b": float("inf"), "c": 1.0, "d": 0.5}
        result = harmonic_mean_p(p)
        assert math.isfinite(result)
        assert 0.0 < result <= 1.0


class TestHmpThresholdAtAlpha:
    def test_returns_value_in_unit_interval(self):
        t = hmp_threshold_at_alpha(L=5, alpha=0.05, n_simulation_draws=10_000)
        assert 0.0 < t < 1.0

    def test_smaller_alpha_yields_smaller_threshold(self):
        t01 = hmp_threshold_at_alpha(L=5, alpha=0.01, n_simulation_draws=20_000)
        t05 = hmp_threshold_at_alpha(L=5, alpha=0.05, n_simulation_draws=20_000)
        t10 = hmp_threshold_at_alpha(L=5, alpha=0.10, n_simulation_draws=20_000)
        assert t01 < t05 < t10

    def test_deterministic_with_same_args(self):
        t1 = hmp_threshold_at_alpha(L=4, alpha=0.05, n_simulation_draws=10_000)
        t2 = hmp_threshold_at_alpha(L=4, alpha=0.05, n_simulation_draws=10_000)
        assert t1 == t2

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            hmp_threshold_at_alpha(L=0, alpha=0.05)
        with pytest.raises(ValueError):
            hmp_threshold_at_alpha(L=5, alpha=0.0)
        with pytest.raises(ValueError):
            hmp_threshold_at_alpha(L=5, alpha=1.5)
