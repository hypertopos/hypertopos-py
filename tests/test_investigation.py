# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for investigative explanation engine."""

import numpy as np
import pytest

from hypertopos.engine.investigation import build_explanation


class TestBuildExplanation:
    def test_basic_explanation_structure(self):
        delta = np.array([3.0, 0.5, 0.1], dtype=np.float32)
        dim_labels = ["customers", "products", "stores"]
        theta_norm = 2.5
        result = build_explanation(
            delta=delta,
            dim_labels=dim_labels,
            theta_norm=theta_norm,
            delta_norm=float(np.linalg.norm(delta)),
            conformal_p=0.02,
        )
        assert "witness" in result
        assert "top_dimensions" in result
        assert "severity" in result
        assert result["severity"] in ("low", "medium", "high", "extreme")

    def test_severity_high(self):
        dim_labels = ["a", "b"]
        theta = 2.0
        # ratio = 3.0/2.0 = 1.5 → high (>= 1.5)
        r1 = build_explanation(np.array([3.0, 0.0]), dim_labels, theta, 3.0, 0.1)
        assert r1["severity"] == "high"

    def test_severity_extreme(self):
        dim_labels = ["a", "b"]
        theta = 2.0
        # ratio = 6.0/2.0 = 3.0 → extreme (>= 2.5)
        r2 = build_explanation(np.array([6.0, 0.0]), dim_labels, theta, 6.0, 0.001)
        assert r2["severity"] == "extreme"

    def test_non_anomaly_returns_normal(self):
        result = build_explanation(np.array([0.5, 0.5]), ["a", "b"], 3.0, 0.71, 0.8)
        assert result["severity"] == "normal"

    def test_includes_conformal_p(self):
        result = build_explanation(np.array([4.0, 0.1]), ["a", "b"], 2.0, 4.0, 0.01)
        assert result["conformal_p"] == 0.01

    def test_includes_reputation_when_provided(self):
        rep = {"alpha": 5, "beta": 10, "reputation": 0.4, "anomaly_tenure": 2}
        result = build_explanation(
            np.array([4.0, 0.1]),
            ["a", "b"],
            2.0,
            4.0,
            0.01,
            reputation=rep,
        )
        assert result["reputation"] == rep

    def test_top_dimensions_share_of_delta_norm_sq(self):
        """``pct_of_total`` is the dim's share of ``delta_norm²`` — the
        metric behind ``is_anomaly`` — whatever the dims' distribution
        family. A rare poisson count with a tiny sigma carries a huge
        z-score and must rank first even though its family-level
        divergence is small."""
        delta = np.array([299.6, -197.1], dtype=np.float64)
        result = build_explanation(
            delta=delta,
            dim_labels=["rare_count", "pair_count"],
            theta_norm=9.94,
            delta_norm=float(np.linalg.norm(delta)),
            dimension_kinds=["poisson", "poisson"],
        )
        top = result["top_dimensions"]
        total = float((delta**2).sum())
        assert [d["dim"] for d in top] == ["rare_count", "pair_count"]
        assert top[0] == {
            "dim": "rare_count",
            "kind": "poisson",
            "delta": 299.6,
            "pct_of_total": round(299.6**2 / total, 4),
        }
        assert top[0]["pct_of_total"] == pytest.approx(0.698, abs=1e-3)
        assert top[1]["delta"] == -197.1
        assert sum(d["pct_of_total"] for d in top) == pytest.approx(1.0, abs=1e-3)

    def test_top_dimensions_without_kinds_has_same_shape(self):
        """No kind tags: same entry shape minus ``kind``; zero-share dims
        are dropped."""
        result = build_explanation(np.array([4.0, 3.0, 0.0]), ["a", "b", "c"], 2.0, 5.0, 0.01)
        assert result["top_dimensions"] == [
            {"dim": "a", "delta": 4.0, "pct_of_total": 0.64},
            {"dim": "b", "delta": 3.0, "pct_of_total": 0.36},
        ]

    def test_includes_temporal_slices(self):
        result = build_explanation(
            np.array([4.0, 0.1]),
            ["a", "b"],
            2.0,
            4.0,
            0.01,
            temporal_slices=15,
        )
        assert result["temporal_slices"] == 15
