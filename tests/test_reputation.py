# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for geometric reputation (Beta distribution anomaly history)."""

import numpy as np
from hypertopos.engine.geometry import GDSEngine


class TestReputation:
    def test_all_normal(self):
        delta_norms = np.array([0.5, 0.6, 0.4, 0.7, 0.3], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["alpha"] == 0
        assert result["beta"] == 5
        assert result["reputation"] < 0.15

    def test_all_anomalous(self):
        delta_norms = np.array([4.0, 5.0, 3.5, 4.5, 3.1], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["alpha"] == 5
        assert result["beta"] == 0
        assert result["reputation"] > 0.8

    def test_half_and_half(self):
        delta_norms = np.array([4.0, 0.5, 4.0, 0.5, 4.0, 0.5], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["alpha"] == 3
        assert result["beta"] == 3
        assert 0.35 < result["reputation"] < 0.65

    def test_single_observation_shrinkage(self):
        delta_norms = np.array([4.0], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["alpha"] == 1
        assert result["beta"] == 0
        assert 0.6 < result["reputation"] < 0.7

    def test_empty_history(self):
        delta_norms = np.array([], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["reputation"] == 0.5
        assert result["alpha"] == 0
        assert result["beta"] == 0

    def test_first_time_offender(self):
        norms = [0.5] * 30 + [4.0]
        delta_norms = np.array(norms, dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["alpha"] == 1
        assert result["beta"] == 30
        assert result["reputation"] < 0.1

    def test_tenure_at_end(self):
        norms = [0.5, 0.5, 4.0, 4.0, 4.0]
        delta_norms = np.array(norms, dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["anomaly_tenure"] == 3

    def test_tenure_mid_series(self):
        """Max consecutive run in the middle, not at the end."""
        norms = [0.5, 4.0, 4.0, 4.0, 4.0, 0.5, 0.5, 0.5, 0.5]
        delta_norms = np.array(norms, dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.compute_reputation(delta_norms, theta_norm)
        assert result["anomaly_tenure"] == 4
        assert result["alpha"] == 4
        assert result["beta"] == 5
