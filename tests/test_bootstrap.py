# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import numpy as np
import pytest

from hypertopos.builder._bootstrap import compute_bootstrap_confidence


def _make_gaussian_population(N: int = 1000, D: int = 3, seed: int = 0) -> np.ndarray:
    """Return (N, D) float32 sample from N(0, 1)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D)).astype(np.float32)


class TestBootstrapConfidence:
    def test_obvious_anomaly_high_confidence(self):
        """Extreme outlier (x=10,10,10 in N(0,1) population) should have confidence >= 0.9."""
        base = _make_gaussian_population(N=1000, D=3, seed=1)
        # Insert an extreme outlier at index 0
        outlier = np.full((1, 3), 10.0, dtype=np.float32)
        X = np.vstack([outlier, base])  # (1001, 3)
        kinds = ["gaussian", "gaussian", "gaussian"]
        conf = compute_bootstrap_confidence(X, kinds, anomaly_percentile=95.0, B=200, seed=42)
        assert conf is not None
        assert conf[0] >= 0.9, f"Expected outlier confidence >= 0.9, got {conf[0]}"

    def test_normal_entity_low_confidence(self):
        """Entity at mean (0,0,0) should have confidence < 0.15."""
        base = _make_gaussian_population(N=1000, D=3, seed=2)
        # Insert a perfectly average entity at index 0
        mean_entity = np.zeros((1, 3), dtype=np.float32)
        X = np.vstack([mean_entity, base])  # (1001, 3)
        kinds = ["gaussian", "gaussian", "gaussian"]
        conf = compute_bootstrap_confidence(X, kinds, anomaly_percentile=95.0, B=200, seed=42)
        assert conf is not None
        assert conf[0] < 0.15, f"Expected mean entity confidence < 0.15, got {conf[0]}"

    def test_borderline_entity_intermediate(self):
        """Entity near the total anomaly threshold should have 0.05 < conf < 0.95.

        The algorithm flags an entity when its total Bregman norm (sum over
        all dimensions) exceeds theta_total = sum of per-dimension thresholds.
        An entity placed exactly at the threshold boundary produces intermediate
        confidence because bootstrap variation shifts the threshold both above
        and below the entity across iterations.
        """
        rng = np.random.default_rng(3)
        base = rng.standard_normal((1000, 3)).astype(np.float32)
        # 2.0 sits near the theta_total boundary for this population
        # (verified: conf ≈ 0.565 at B=200, seed=42)
        borderline = np.full((1, 3), 2.0, dtype=np.float32)
        X = np.vstack([borderline, base])
        kinds = ["gaussian", "gaussian", "gaussian"]
        conf = compute_bootstrap_confidence(X, kinds, anomaly_percentile=95.0, B=200, seed=42)
        assert conf is not None
        assert 0.05 < conf[0] < 0.95, f"Expected borderline in (0.05, 0.95), got {conf[0]}"

    def test_B_zero_returns_none(self):
        """B=0 should return None."""
        X = _make_gaussian_population(N=100, D=3, seed=4)
        kinds = ["gaussian", "gaussian", "gaussian"]
        result = compute_bootstrap_confidence(X, kinds, B=0)
        assert result is None

    def test_stratified_with_groups(self):
        """group_ids should stratify resampling; outlier in group A still detected."""
        rng = np.random.default_rng(5)
        base = rng.standard_normal((500, 3)).astype(np.float32)
        outlier = np.full((1, 3), 10.0, dtype=np.float32)
        X = np.vstack([outlier, base])  # (501, 3)
        kinds = ["gaussian", "gaussian", "gaussian"]
        # Assign outlier (idx 0) to group 0, rest split between groups 0 and 1
        group_ids = np.array([0] + [i % 2 for i in range(500)], dtype=np.int32)
        conf = compute_bootstrap_confidence(
            X, kinds, anomaly_percentile=95.0, B=200, seed=42, group_ids=group_ids
        )
        assert conf is not None
        assert conf[0] >= 0.85, f"Expected outlier confidence >= 0.85 with stratified sampling, got {conf[0]}"

    def test_mixed_kinds(self):
        """Works with heterogeneous kinds (gaussian + poisson + bernoulli)."""
        rng = np.random.default_rng(6)
        N = 500
        # gaussian: N(2, 1), poisson-like: non-negative counts, bernoulli: [0,1]
        gauss_col = rng.normal(2.0, 1.0, N).astype(np.float32)
        pois_col = rng.poisson(3, N).astype(np.float32)
        bern_col = rng.beta(2, 5, N).astype(np.float32)
        X = np.stack([gauss_col, pois_col, bern_col], axis=1)
        kinds = ["gaussian", "poisson", "bernoulli"]
        conf = compute_bootstrap_confidence(X, kinds, anomaly_percentile=95.0, B=200, seed=42)
        assert conf is not None
        assert conf.shape == (N,)
        assert conf.dtype == np.float32
        assert np.all(conf >= 0.0) and np.all(conf <= 1.0)

    def test_output_shape_and_range(self):
        """Output is (N,) float32, all values in [0, 1]."""
        X = _make_gaussian_population(N=200, D=4, seed=7)
        kinds = ["gaussian"] * 4
        conf = compute_bootstrap_confidence(X, kinds, anomaly_percentile=95.0, B=200, seed=42)
        assert conf is not None
        assert conf.shape == (200,)
        assert conf.dtype == np.float32
        assert float(conf.min()) >= 0.0
        assert float(conf.max()) <= 1.0

    def test_deterministic_with_seed(self):
        """Same seed produces same result."""
        X = _make_gaussian_population(N=300, D=3, seed=8)
        kinds = ["gaussian", "gaussian", "gaussian"]
        conf1 = compute_bootstrap_confidence(X, kinds, B=200, seed=42)
        conf2 = compute_bootstrap_confidence(X, kinds, B=200, seed=42)
        assert conf1 is not None and conf2 is not None
        np.testing.assert_array_equal(conf1, conf2)

    def test_all_constant_dims_handled(self):
        """All-zero/constant input should return None (all bootstrap iters degenerate).

        When every dimension is constant, per_dim_theta returns all-zero
        thresholds and theta_total_b == 0, so every bootstrap iteration is
        skipped as degenerate.  The function should return None rather than
        confidence=1.0 for everyone.
        """
        N, D = 100, 3
        X = np.ones((N, D), dtype=np.float32) * 5.0
        kinds = ["gaussian", "gaussian", "gaussian"]
        result = compute_bootstrap_confidence(X, kinds, anomaly_percentile=95.0, B=200, seed=42)
        assert result is None
