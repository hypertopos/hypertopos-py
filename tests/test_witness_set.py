# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for witness_set and anti_witness."""

import numpy as np
from hypertopos.engine.geometry import GDSEngine


class TestWitnessSet:
    def test_single_dominant_dimension(self):
        delta = np.array([4.0, 0.1, 0.1, 0.1], dtype=np.float32)
        theta_norm = 3.0
        dim_labels = ["customers", "products", "stores", "regions"]
        result = GDSEngine.witness_set(delta, theta_norm, dim_labels)
        assert result["witness_size"] == 1
        assert result["witness_dims"][0]["label"] == "customers"

    def test_two_dimensions_needed(self):
        delta = np.array([2.5, 2.5, 0.1, 0.1], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.witness_set(delta, theta_norm, ["a", "b", "c", "d"])
        assert result["witness_size"] == 2

    def test_all_dimensions_needed(self):
        delta = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        theta_norm = 1.9
        result = GDSEngine.witness_set(delta, theta_norm, ["a", "b", "c", "d"])
        assert result["witness_size"] == 4

    def test_non_anomaly_returns_empty(self):
        delta = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.witness_set(delta, theta_norm, ["a", "b", "c"])
        assert result["witness_size"] == 0
        assert result["witness_dims"] == []

    def test_anti_witness_repair_set(self):
        delta = np.array([3.0, 2.0, 0.5, 0.1], dtype=np.float32)
        theta_norm = 2.5
        result = GDSEngine.anti_witness(delta, theta_norm, ["a", "b", "c", "d"])
        assert result["repair_size"] == 1
        assert result["repair_dims"][0]["label"] == "a"

    def test_anti_witness_two_dims_needed(self):
        delta = np.array([2.0, 2.0, 2.0, 0.1], dtype=np.float32)
        theta_norm = 2.5
        result = GDSEngine.anti_witness(delta, theta_norm, ["a", "b", "c", "d"])
        assert result["repair_size"] == 2

    def test_anti_witness_non_anomaly(self):
        delta = np.array([0.5, 0.5], dtype=np.float32)
        theta_norm = 3.0
        result = GDSEngine.anti_witness(delta, theta_norm, ["a", "b"])
        assert result["repair_size"] == 0
