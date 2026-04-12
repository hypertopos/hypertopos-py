# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for Welford rolling z-score."""
from __future__ import annotations

import numpy as np
import pytest

from hypertopos.builder.builder import GDSBuilder


def _naive_rolling_z(shape_tensor, n_anchor, n_buckets):
    """Reference implementation — O(n²), expanding window."""
    max_z = np.zeros(n_anchor, dtype=np.float32)
    for t in range(2, n_buckets):
        current = shape_tensor[:, t, :]
        if current.sum() == 0:
            continue
        window = shape_tensor[:, :t, :]
        mu_w = window.mean(axis=1)
        std_w = np.maximum(window.std(axis=1), 0.01)
        z = np.abs((current - mu_w) / std_w)
        z_max = z.max(axis=1)
        max_z = np.maximum(max_z, z_max)
    return max_z


class TestWelfordRollingZ:
    def test_equivalence_with_naive(self):
        rng = np.random.default_rng(42)
        n_anchor, n_buckets, D = 100, 20, 8
        tensor = rng.uniform(0, 1, (n_anchor, n_buckets, D)).astype(np.float32)
        tensor[50:, :5, :] = 0.0

        result = GDSBuilder._compute_max_rolling_z(tensor, n_anchor, n_buckets)
        expected = _naive_rolling_z(tensor, n_anchor, n_buckets)

        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_empty_buckets_skipped(self):
        tensor = np.zeros((5, 10, 4), dtype=np.float32)
        tensor[:, 0, :] = 1.0
        result = GDSBuilder._compute_max_rolling_z(tensor, 5, 10)
        assert np.all(result == 0.0)

    def test_all_zeros(self):
        tensor = np.zeros((10, 5, 3), dtype=np.float32)
        result = GDSBuilder._compute_max_rolling_z(tensor, 10, 5)
        assert np.all(result == 0.0)

    def test_single_bucket(self):
        tensor = np.ones((5, 1, 3), dtype=np.float32)
        result = GDSBuilder._compute_max_rolling_z(tensor, 5, 1)
        assert np.all(result == 0.0)

    def test_two_buckets(self):
        tensor = np.ones((5, 2, 3), dtype=np.float32)
        result = GDSBuilder._compute_max_rolling_z(tensor, 5, 2)
        assert np.all(result == 0.0)  # need >= 3 observations

    def test_large_input(self):
        n_anchor, n_buckets, D = 1000, 200, 16
        tensor = np.random.default_rng(7).uniform(
            0, 1, (n_anchor, n_buckets, D),
        ).astype(np.float32)
        result = GDSBuilder._compute_max_rolling_z(tensor, n_anchor, n_buckets)
        assert result.shape == (n_anchor,)
        assert np.all(np.isfinite(result))
        assert result.max() > 0

    def test_sparse_activity(self):
        rng = np.random.default_rng(99)
        n_anchor, n_buckets, D = 50, 30, 6
        tensor = np.zeros((n_anchor, n_buckets, D), dtype=np.float32)
        for i in range(n_anchor):
            active_buckets = rng.choice(n_buckets, size=rng.integers(3, 10), replace=False)
            tensor[i, active_buckets, :] = rng.uniform(0, 1, (len(active_buckets), D))

        result = GDSBuilder._compute_max_rolling_z(tensor, n_anchor, n_buckets)
        expected = _naive_rolling_z(tensor, n_anchor, n_buckets)
        np.testing.assert_allclose(result, expected, atol=1e-3)
