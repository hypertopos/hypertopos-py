# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for per-detector p-value calibration adapters."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.engine.p_value_calibration import (
    detector_p_value_anomaly,
    detector_p_value_density_gap,
    detector_p_value_neighbor_contamination,
    detector_p_value_segment_shift,
    detector_p_value_trajectory_continuous,
)


class TestAnomalyAdapter:
    def test_uses_stored_conformal_p(self):
        tbl = pa.table(
            {
                "primary_key": ["A", "B", "C"],
                "conformal_p": [0.9, 0.5, 0.01],
                "anomaly_confidence": [0.1, 0.1, 0.1],  # stability, never read as a p-value
            }
        )
        result = detector_p_value_anomaly(tbl, ["A", "B", "C"])
        assert result == pytest.approx({"A": 0.9, "B": 0.5, "C": 0.01})

    def test_null_or_missing_conformal_p_carries_no_evidence(self):
        tbl = pa.table(
            {
                "primary_key": ["A", "B"],
                "conformal_p": pa.array([None, 0.02], type=pa.float32()),
            }
        )
        assert detector_p_value_anomaly(tbl, ["A", "B"]) == pytest.approx({"A": 1.0, "B": 0.02})
        no_column = pa.table({"primary_key": ["A"], "delta_norm": [9.0]})
        assert detector_p_value_anomaly(no_column, ["A"]) == {"A": 1.0}

    def test_empty_input(self):
        tbl = pa.table(
            {
                "primary_key": pa.array([], type=pa.string()),
                "conformal_p": pa.array([], type=pa.float64()),
            }
        )
        assert detector_p_value_anomaly(tbl, []) == {}

    def test_filters_to_primary_keys(self):
        tbl = pa.table({"primary_key": ["A", "B", "C"], "conformal_p": [0.5, 0.5, 0.5]})
        assert set(detector_p_value_anomaly(tbl, ["A", "C"])) == {"A", "C"}

    def test_no_nan_or_inf(self):
        tbl = pa.table({"primary_key": ["A", "B"], "conformal_p": [0.0, float("nan")]})
        for p in detector_p_value_anomaly(tbl, ["A", "B"]).values():
            assert np.isfinite(p)
            assert 0.0 < p <= 1.0


class TestNeighborContaminationAdapter:
    def test_hypergeometric_more_anomalies_than_expected(self):
        # population 1000 of which 100 anomalies; entity has 10 neighbors and 8 are anomalous
        # baseline rate 10%, observed 80% — strong signal => low p
        observations = {"E1": (10, 8)}
        result = detector_p_value_neighbor_contamination(
            observations,
            total_population=1000,
            total_anomalies=100,
            k=10,
        )
        assert "E1" in result
        assert 0.0 < result["E1"] <= 1.0
        assert result["E1"] < 0.01

    def test_hypergeometric_at_baseline_returns_high_p(self):
        # baseline rate 10%, observed 1/10 = 10% — no signal
        observations = {"E1": (10, 1)}
        result = detector_p_value_neighbor_contamination(
            observations,
            total_population=1000,
            total_anomalies=100,
            k=10,
        )
        assert result["E1"] > 0.3

    def test_uniform_under_null(self):
        # When observed equals expected, p should be near 0.5
        observations = {f"E{i}": (10, 1) for i in range(20)}
        result = detector_p_value_neighbor_contamination(
            observations,
            total_population=1000,
            total_anomalies=100,
            k=10,
        )
        assert all(0.0 < p <= 1.0 for p in result.values())
        # All identical observations → all identical p
        ps = list(result.values())
        assert max(ps) - min(ps) < 1e-9

    def test_zero_observed_returns_p_one(self):
        observations = {"E1": (10, 0)}
        result = detector_p_value_neighbor_contamination(
            observations,
            total_population=1000,
            total_anomalies=100,
            k=10,
        )
        # P(X >= 0) = 1
        assert result["E1"] == pytest.approx(1.0)

    def test_empty_input(self):
        result = detector_p_value_neighbor_contamination(
            {},
            total_population=1000,
            total_anomalies=100,
            k=10,
        )
        assert result == {}

    def test_no_nan_or_inf(self):
        observations = {f"E{i}": (5, i % 6) for i in range(6)}
        result = detector_p_value_neighbor_contamination(
            observations,
            total_population=500,
            total_anomalies=50,
            k=5,
        )
        for p in result.values():
            assert np.isfinite(p)
            assert 0.0 < p <= 1.0


class TestSegmentShiftAdapter:
    def test_concentrated_segment_anomalies_low_p(self):
        # Segment X has 90/100 anomalies; segment Y has 10/100 — strong concentration
        observations = {
            "S_X": {
                "in_segment_anomalous": 90,
                "in_segment_total": 100,
                "out_segment_anomalous": 10,
                "out_segment_total": 100,
            },
        }
        result = detector_p_value_segment_shift(observations)
        assert "S_X" in result
        assert 0.0 < result["S_X"] <= 1.0
        assert result["S_X"] < 0.001

    def test_uniform_segment_high_p(self):
        # Same anomaly rate in both segments
        observations = {
            "S_X": {
                "in_segment_anomalous": 10,
                "in_segment_total": 100,
                "out_segment_anomalous": 10,
                "out_segment_total": 100,
            },
        }
        result = detector_p_value_segment_shift(observations)
        assert result["S_X"] > 0.5

    def test_depleted_segment_is_not_evidence_of_anomaly(self):
        # Segment has a far LOWER anomaly rate than the rest: its members are
        # less likely to be anomalous, so the detector must not give them a tiny p.
        observations = {
            "S_low": {
                "in_segment_anomalous": 5,
                "in_segment_total": 10_000,
                "out_segment_anomalous": 5_000,
                "out_segment_total": 100_000,
            },
            "S_high": {
                "in_segment_anomalous": 500,
                "in_segment_total": 1_000,
                "out_segment_anomalous": 5_000,
                "out_segment_total": 100_000,
            },
        }
        result = detector_p_value_segment_shift(observations)
        assert result["S_low"] > 0.5
        assert result["S_high"] < 1e-10

    def test_empty_input(self):
        assert detector_p_value_segment_shift({}) == {}

    def test_zero_marginal_returns_p_one(self):
        # No anomalies at all → p = 1
        observations = {
            "S_X": {
                "in_segment_anomalous": 0,
                "in_segment_total": 50,
                "out_segment_anomalous": 0,
                "out_segment_total": 50,
            },
        }
        result = detector_p_value_segment_shift(observations)
        assert result["S_X"] == pytest.approx(1.0)

    def test_no_nan_or_inf(self):
        observations = {
            f"S{i}": {
                "in_segment_anomalous": i,
                "in_segment_total": 50,
                "out_segment_anomalous": 1,
                "out_segment_total": 50,
            }
            for i in range(0, 10)
        }
        result = detector_p_value_segment_shift(observations)
        for p in result.values():
            assert np.isfinite(p)
            assert 0.0 < p <= 1.0


class TestDensityGapAdapter:
    def test_inverts_q_to_p_via_bh_formula(self):
        # BH: q = p * m / rank; given (q, rank, m) → recover p
        # Three entities ranked ascending by p. Synthetic: p = [0.001, 0.05, 0.5], m=3
        # q = [min(0.003, 0.075, 1.5)=0.003, min(0.075, 1.5)=0.075, 1.0_clipped]
        results = [
            {"primary_key": "A", "q_value": 0.003, "rank": 1, "m": 3},
            {"primary_key": "B", "q_value": 0.075, "rank": 2, "m": 3},
            {"primary_key": "C", "q_value": 1.0, "rank": 3, "m": 3},
        ]
        out = detector_p_value_density_gap(results)
        # A: p = 0.003 * 1 / 3 = 0.001
        assert out["A"] == pytest.approx(0.001, abs=1e-6)
        # B: p = 0.075 * 2 / 3 = 0.05
        assert out["B"] == pytest.approx(0.05, abs=1e-6)
        # C: p = min(1.0 * 3 / 3, 1) = 1
        assert out["C"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_input(self):
        assert detector_p_value_density_gap([]) == {}

    def test_uses_p_value_directly_when_present(self):
        # If raw 'p_value' provided, use it directly (no inversion)
        results = [
            {"primary_key": "A", "p_value": 0.01},
            {"primary_key": "B", "p_value": 0.5},
        ]
        out = detector_p_value_density_gap(results)
        assert out["A"] == pytest.approx(0.01)
        assert out["B"] == pytest.approx(0.5)

    def test_no_nan_or_inf(self):
        results = [{"primary_key": f"E{i}", "p_value": 1.0 / (i + 1)} for i in range(20)]
        out = detector_p_value_density_gap(results)
        for p in out.values():
            assert np.isfinite(p)
            assert 0.0 < p <= 1.0


class TestTrajectoryContinuousAdapter:
    def test_ecdf_mapping(self):
        # 5 entities with known DTW distances; expected p = 1 - ecdf(score)
        scores = {"A": 0.0, "B": 0.5, "C": 1.0, "D": 1.5, "E": 5.0}
        result = detector_p_value_trajectory_continuous(scores)
        assert set(result.keys()) == set(scores.keys())
        # Highest score → smallest p; rank-based
        assert result["E"] < result["A"]
        # p = (n - rank + 1) / (n + 1): extremes are 1/(n+1) and n/(n+1)
        assert result["E"] == pytest.approx(1 / 6)
        assert result["A"] == pytest.approx(5 / 6)
        for p in result.values():
            assert 0.0 < p <= 1.0

    def test_uniform_under_null(self):
        # All identical scores → all identical p
        scores = {f"E{i}": 1.0 for i in range(10)}
        result = detector_p_value_trajectory_continuous(scores)
        ps = list(result.values())
        assert max(ps) - min(ps) < 1e-9

    def test_empty_input(self):
        assert detector_p_value_trajectory_continuous({}) == {}

    def test_synthetic_alternative_ranking(self):
        rng = np.random.default_rng(42)
        # 50 background entities + 5 outliers
        scores = {f"E{i}": float(rng.uniform(0, 1)) for i in range(50)}
        scores.update({f"O{i}": float(rng.uniform(5, 10)) for i in range(5)})
        result = detector_p_value_trajectory_continuous(scores)
        # Outliers should all have lower p than median background
        bg_median = float(np.median([result[f"E{i}"] for i in range(50)]))
        for i in range(5):
            assert result[f"O{i}"] < bg_median

    def test_no_nan_or_inf(self):
        scores = {f"E{i}": float(i) for i in range(10)}
        result = detector_p_value_trajectory_continuous(scores)
        for p in result.values():
            assert np.isfinite(p)
            assert 0.0 < p <= 1.0
