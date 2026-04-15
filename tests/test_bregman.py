# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import math

import numpy as np
import pytest

from hypertopos.builder._bregman import (
    BREGMAN_KINDS,
    bregman_divergence,
    bregman_divergence_batch,
    bregman_norms,
    per_dim_theta,
)


class TestConstants:
    def test_kinds_tuple(self):
        assert BREGMAN_KINDS == ("gaussian", "poisson", "bernoulli")


class TestBregmanDivergenceSingle:
    """Unit tests for bregman_divergence — single entity, per-dim output."""

    def test_gaussian_hand_computed(self):
        # (3.0 - 1.0)^2 / (2 * 0.5^2) = 4 / 0.5 = 8.0
        x = np.array([3.0], dtype=np.float64)
        mu = np.array([1.0], dtype=np.float64)
        sigma = np.array([0.5], dtype=np.float64)
        kinds = ["gaussian"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(8.0, rel=1e-6)

    def test_poisson_hand_computed(self):
        # 5 * log(5/2) - (5 - 2) = 5 * log(2.5) - 3
        x = np.array([5.0], dtype=np.float64)
        mu = np.array([2.0], dtype=np.float64)
        sigma = np.array([1.0], dtype=np.float64)  # ignored for poisson
        kinds = ["poisson"]
        expected = 5.0 * math.log(5.0 / 2.0) - (5.0 - 2.0)
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_bernoulli_hand_computed(self):
        # 0.8*log(0.8/0.2) + 0.2*log(0.2/0.8)
        x = np.array([0.8], dtype=np.float64)
        mu = np.array([0.2], dtype=np.float64)
        sigma = np.array([1.0], dtype=np.float64)  # ignored for bernoulli
        kinds = ["bernoulli"]
        expected = 0.8 * math.log(0.8 / 0.2) + 0.2 * math.log(0.2 / 0.8)
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_poisson_x_zero_returns_mu(self):
        # When x=0: result = mu
        x = np.array([0.0], dtype=np.float64)
        mu = np.array([2.0], dtype=np.float64)
        sigma = np.array([1.0], dtype=np.float64)
        kinds = ["poisson"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(2.0, rel=1e-6)

    def test_bernoulli_at_x_zero_finite(self):
        # x=0 must be clamped and produce a finite result
        x = np.array([0.0], dtype=np.float64)
        mu = np.array([0.5], dtype=np.float64)
        sigma = np.array([1.0], dtype=np.float64)
        kinds = ["bernoulli"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_bernoulli_at_x_one_finite(self):
        # x=1 must be clamped and produce a finite result
        x = np.array([1.0], dtype=np.float64)
        mu = np.array([0.5], dtype=np.float64)
        sigma = np.array([1.0], dtype=np.float64)
        kinds = ["bernoulli"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_mixed_kinds_multi_dim(self):
        # 3-dim entity with one of each kind
        x = np.array([3.0, 5.0, 0.8], dtype=np.float64)
        mu = np.array([1.0, 2.0, 0.2], dtype=np.float64)
        sigma = np.array([0.5, 1.0, 1.0], dtype=np.float64)
        kinds = ["gaussian", "poisson", "bernoulli"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

        expected_gaussian = (3.0 - 1.0) ** 2 / (2 * 0.5 ** 2)
        expected_poisson = 5.0 * math.log(5.0 / 2.0) - (5.0 - 2.0)
        expected_bernoulli = 0.8 * math.log(0.8 / 0.2) + 0.2 * math.log(0.2 / 0.8)
        assert result[0] == pytest.approx(expected_gaussian, rel=1e-5)
        assert result[1] == pytest.approx(expected_poisson, rel=1e-5)
        assert result[2] == pytest.approx(expected_bernoulli, rel=1e-5)

    def test_gaussian_at_mean_is_zero(self):
        x = np.array([1.0], dtype=np.float64)
        mu = np.array([1.0], dtype=np.float64)
        sigma = np.array([0.5], dtype=np.float64)
        kinds = ["gaussian"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_returns_float64_array(self):
        x = np.array([1.0], dtype=np.float32)
        mu = np.array([1.0], dtype=np.float32)
        sigma = np.array([1.0], dtype=np.float32)
        kinds = ["gaussian"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert result.dtype == np.float64


class TestBregmanDivergenceBatch:
    """Unit tests for bregman_divergence_batch — (N, D) matrix output."""

    def test_shape(self):
        N, D = 50, 3
        X = np.random.default_rng(0).random((N, D))
        mu = np.array([0.5, 2.0, 0.3], dtype=np.float64)
        sigma = np.array([0.1, 1.0, 0.1], dtype=np.float64)
        kinds = ["gaussian", "poisson", "bernoulli"]
        result = bregman_divergence_batch(X, mu, sigma, kinds)
        assert result.shape == (N, D)

    def test_all_finite_non_negative(self):
        rng = np.random.default_rng(42)
        N, D = 100, 3
        X = rng.uniform(0.01, 0.99, size=(N, D))
        mu = np.array([0.5, 0.3, 0.5], dtype=np.float64)
        sigma = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        kinds = ["bernoulli", "bernoulli", "bernoulli"]
        result = bregman_divergence_batch(X, mu, sigma, kinds)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

    def test_consistent_with_single(self):
        # Row i of batch must match single-entity call for row i
        rng = np.random.default_rng(7)
        N, D = 10, 3
        X = rng.uniform(0.5, 5.0, size=(N, D))
        mu = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.array([0.5, 1.0, 2.0], dtype=np.float64)
        kinds = ["gaussian", "gaussian", "gaussian"]
        batch_result = bregman_divergence_batch(X, mu, sigma, kinds)
        for i in range(N):
            single = bregman_divergence(X[i], mu, sigma, kinds)
            np.testing.assert_allclose(batch_result[i], single, rtol=1e-10)

    def test_gaussian_formula(self):
        # All gaussian — verify formula exactly
        X = np.array([[3.0, 0.0], [1.0, 2.0]], dtype=np.float64)
        mu = np.array([1.0, 1.0], dtype=np.float64)
        sigma = np.array([0.5, 0.5], dtype=np.float64)
        kinds = ["gaussian", "gaussian"]
        result = bregman_divergence_batch(X, mu, sigma, kinds)
        expected = (X - mu) ** 2 / (2 * sigma ** 2)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_poisson_zeros_row(self):
        # Row with x=0 everywhere — each dim should equal mu[d]
        X = np.zeros((1, 3), dtype=np.float64)
        mu = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sigma = np.ones(3, dtype=np.float64)
        kinds = ["poisson", "poisson", "poisson"]
        result = bregman_divergence_batch(X, mu, sigma, kinds)
        np.testing.assert_allclose(result[0], mu, rtol=1e-10)

    def test_returns_float64(self):
        X = np.ones((5, 2), dtype=np.float32)
        mu = np.ones(2, dtype=np.float32)
        sigma = np.ones(2, dtype=np.float32)
        kinds = ["gaussian", "gaussian"]
        result = bregman_divergence_batch(X, mu, sigma, kinds)
        assert result.dtype == np.float64


class TestBregmanNorms:
    """Unit tests for bregman_norms — (N,) total per-entity divergence."""

    def test_unweighted_equals_row_sum(self):
        rng = np.random.default_rng(3)
        N, D = 20, 4
        X = rng.uniform(0.1, 5.0, size=(N, D))
        mu = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        sigma = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        kinds = ["gaussian"] * D
        per_dim = bregman_divergence_batch(X, mu, sigma, kinds)
        norms = bregman_norms(X, mu, sigma, kinds)
        assert norms.shape == (N,)
        np.testing.assert_allclose(norms, per_dim.sum(axis=1), rtol=1e-10)

    def test_weighted_equals_weighted_row_sum(self):
        rng = np.random.default_rng(5)
        N, D = 15, 3
        X = rng.uniform(0.1, 3.0, size=(N, D))
        mu = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        sigma = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        kinds = ["gaussian", "gaussian", "gaussian"]
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        per_dim = bregman_divergence_batch(X, mu, sigma, kinds)
        norms = bregman_norms(X, mu, sigma, kinds, weights=weights)
        assert norms.shape == (N,)
        np.testing.assert_allclose(norms, (per_dim * weights).sum(axis=1), rtol=1e-10)

    def test_shape_only(self):
        X = np.ones((7, 2), dtype=np.float64) * 0.5
        mu = np.array([0.5, 0.5])
        sigma = np.array([0.1, 0.1])
        kinds = ["gaussian", "gaussian"]
        result = bregman_norms(X, mu, sigma, kinds)
        assert result.shape == (7,)

    def test_at_mean_is_zero(self):
        # X == mu for all rows — all divergences should be 0
        X = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
        mu = np.array([1.0, 2.0], dtype=np.float64)
        sigma = np.array([0.5, 1.0], dtype=np.float64)
        kinds = ["gaussian", "gaussian"]
        result = bregman_norms(X, mu, sigma, kinds)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)


class TestPerDimTheta:
    """Unit tests for per_dim_theta — (D,) anomaly thresholds."""

    def test_shape(self):
        rng = np.random.default_rng(9)
        N, D = 200, 4
        X = rng.uniform(0.1, 5.0, size=(N, D))
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 0.1
        kinds = ["gaussian"] * D
        theta = per_dim_theta(X, mu, sigma, kinds)
        assert theta.shape == (D,)

    def test_all_positive(self):
        rng = np.random.default_rng(11)
        N, D = 300, 3
        X = rng.uniform(0.5, 10.0, size=(N, D))
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 0.1
        kinds = ["poisson", "gaussian", "bernoulli"]
        # clamp X to valid ranges
        X[:, 2] = np.clip(X[:, 2] / 10.0, 0.01, 0.99)
        mu[2] = np.clip(mu[2] / 10.0, 0.01, 0.99)
        theta = per_dim_theta(X, mu, sigma, kinds)
        assert np.all(theta > 0.0)

    def test_uniform_population_returns_zero_thresholds(self):
        # All rows == mu → Bregman divergences are 0 → per-dim theta is 0.
        # Chernoff floor is applied to total theta only, not per-dim.
        D = 3
        X = np.ones((100, D), dtype=np.float64) * 2.0
        mu = np.ones(D, dtype=np.float64) * 2.0
        sigma = np.ones(D, dtype=np.float64) * 0.5
        kinds = ["gaussian"] * D
        theta = per_dim_theta(X, mu, sigma, kinds)
        assert np.allclose(theta, 0.0, atol=1e-10)

    def test_mixed_kinds_finite(self):
        rng = np.random.default_rng(13)
        N, D = 100, 3
        X_gauss = rng.normal(2.0, 0.5, size=(N, 1))
        X_pois = rng.poisson(3, size=(N, 1)).astype(np.float64)
        X_bern = rng.uniform(0.01, 0.99, size=(N, 1))
        X = np.hstack([X_gauss, X_pois, X_bern])
        mu = np.array([2.0, 3.0, 0.5])
        sigma = np.array([0.5, 1.0, 0.1])
        kinds = ["gaussian", "poisson", "bernoulli"]
        theta = per_dim_theta(X, mu, sigma, kinds)
        assert np.all(np.isfinite(theta))
        assert np.all(theta > 0.0)

    def test_no_weights_parameter(self):
        """per_dim_theta does not accept weights — it is always unweighted."""
        rng = np.random.default_rng(17)
        N, D = 200, 2
        X = rng.uniform(0.5, 3.0, size=(N, D))
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 0.1
        kinds = ["gaussian", "gaussian"]
        theta = per_dim_theta(X, mu, sigma, kinds)
        assert np.all(np.isfinite(theta))
        assert np.all(theta > 0.0)

    def test_default_percentile_95(self):
        # At least 5% of the population should exceed the threshold
        rng = np.random.default_rng(19)
        N, D = 500, 2
        X = rng.uniform(0.1, 5.0, size=(N, D))
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 0.1
        kinds = ["gaussian", "gaussian"]
        per_dim = bregman_divergence_batch(X, mu, sigma, kinds)
        theta = per_dim_theta(X, mu, sigma, kinds, anomaly_percentile=95.0)
        # At most ~10% should exceed (loose bound — Chernoff can raise the bar)
        for d in range(D):
            exceed_rate = float((per_dim[:, d] > theta[d]).mean())
            assert exceed_rate <= 0.10, f"dim {d}: exceed_rate={exceed_rate:.3f}"


class TestEdgeCases:
    """Tests for edge cases in Bregman divergence functions."""

    def test_sigma_zero_gaussian_clamped(self):
        """sigma=0 for gaussian kind must produce finite result (not inf)."""
        x = np.array([3.0], dtype=np.float64)
        mu = np.array([1.0], dtype=np.float64)
        sigma = np.array([0.0], dtype=np.float64)
        kinds = ["gaussian"]
        result = bregman_divergence(x, mu, sigma, kinds)
        assert np.isfinite(result[0])
        assert result[0] > 0.0

    def test_sigma_zero_gaussian_batch_clamped(self):
        """sigma=0 in batch for gaussian kind must produce finite result."""
        X = np.array([[3.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        mu = np.array([1.0, 1.0], dtype=np.float64)
        sigma = np.array([0.0, 0.0], dtype=np.float64)
        kinds = ["gaussian", "gaussian"]
        result = bregman_divergence_batch(X, mu, sigma, kinds)
        assert np.all(np.isfinite(result))

    def test_single_entity_bregman_divergence_batch(self):
        """N=1 input to bregman_divergence_batch should work."""
        X = np.array([[3.0, 5.0]], dtype=np.float64)
        mu = np.array([1.0, 2.0], dtype=np.float64)
        sigma = np.array([0.5, 1.0], dtype=np.float64)
        kinds = ["gaussian", "poisson"]
        result = bregman_divergence_batch(X, mu, sigma, kinds)
        assert result.shape == (1, 2)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
