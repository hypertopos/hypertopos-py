# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for engine.geometry.compute_reliability_flags."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from hypertopos.engine.geometry import compute_reliability_flags
from hypertopos.model.sphere import Pattern, RelationDef

_DT = datetime(2024, 1, 1, tzinfo=UTC)


def _pattern(
    n_relations: int,
    *,
    dim_labels_relations: list[str] | None = None,
    dimension_kinds: list[str] | None = None,
    with_calibration: bool = False,
) -> Pattern:
    labels = dim_labels_relations or [f"rel_{i}" for i in range(n_relations)]
    relations = [RelationDef(line_id=lab, direction="in", required=True) for lab in labels]
    return Pattern(
        pattern_id="p_x",
        entity_type="x",
        pattern_type="anchor",
        relations=relations,
        mu=np.zeros(n_relations, dtype=np.float32),
        sigma_diag=np.ones(n_relations, dtype=np.float32),
        theta=np.ones(n_relations, dtype=np.float32) * 3.0,
        population_size=100,
        computed_at=_DT,
        version=1,
        status="production",
        dimension_kinds=dimension_kinds if with_calibration else None,
    )


class TestSingleDimDriven:
    def test_high_concentration_flags_single_dim(self):
        flags = compute_reliability_flags(
            [10.0, 1.0, 1.0, 1.0],
            pattern=_pattern(4),
            dominant_dim_threshold=0.7,
        )
        assert flags["single_dim_driven"] is True
        assert flags["dominant_dim"] == "rel_0"
        assert flags["dominant_dim_share"] > 0.9
        assert "single_dim_driven" in flags["flags"]

    def test_balanced_contribution_does_not_flag(self):
        flags = compute_reliability_flags(
            [3.0, 3.0, 3.0, 3.0],
            pattern=_pattern(4),
            dominant_dim_threshold=0.7,
        )
        assert flags["single_dim_driven"] is False
        assert flags["dominant_dim_share"] == pytest.approx(0.25, abs=1e-6)
        assert flags["dominant_dim"] == "rel_0"
        assert "single_dim_driven" not in flags["flags"]

    def test_zero_delta_returns_no_dominant(self):
        flags = compute_reliability_flags([0.0, 0.0, 0.0], pattern=_pattern(3))
        assert flags["single_dim_driven"] is False
        assert flags["dominant_dim"] is None
        assert flags["dominant_dim_share"] == 0.0
        assert flags["flags"] == []

    def test_dominant_dim_uses_display_name_when_set(self):
        pat = Pattern(
            pattern_id="p_x",
            entity_type="x",
            pattern_type="anchor",
            relations=[
                RelationDef(
                    line_id="line_alpha",
                    direction="in",
                    required=True,
                    display_name="Alpha (display)",
                ),
                RelationDef(line_id="line_beta", direction="in", required=True),
                RelationDef(line_id="line_gamma", direction="in", required=True),
            ],
            mu=np.zeros(3, dtype=np.float32),
            sigma_diag=np.ones(3, dtype=np.float32),
            theta=np.array([3.0, 3.0, 3.0], dtype=np.float32),
            population_size=10,
            computed_at=_DT,
            version=1,
            status="production",
        )
        flags = compute_reliability_flags([10.0, 0.5, 0.5], pattern=pat)
        assert flags["dominant_dim"] == "Alpha (display)"


class TestLowConfidenceBucket:
    def test_below_threshold_flags(self):
        flags = compute_reliability_flags(
            [1.0, 1.0, 1.0],
            pattern=_pattern(3),
            anomaly_confidence=0.3,
            confidence_threshold=0.5,
        )
        assert flags["low_confidence_bucket"] is True
        assert flags["confidence"] == pytest.approx(0.3, abs=1e-6)
        assert "low_confidence_bucket" in flags["flags"]

    def test_at_or_above_threshold_does_not_flag(self):
        flags = compute_reliability_flags(
            [1.0, 1.0, 1.0],
            pattern=_pattern(3),
            anomaly_confidence=0.5,
            confidence_threshold=0.5,
        )
        assert flags["low_confidence_bucket"] is False

    def test_missing_confidence_does_not_flag(self):
        flags = compute_reliability_flags(
            [1.0, 1.0, 1.0],
            pattern=_pattern(3),
            anomaly_confidence=None,
        )
        assert flags["low_confidence_bucket"] is False
        assert flags["confidence"] is None

    def test_nan_confidence_sanitised_to_none(self):
        flags = compute_reliability_flags(
            [1.0, 1.0, 1.0],
            pattern=_pattern(3),
            anomaly_confidence=float("nan"),
        )
        assert flags["confidence"] is None
        assert flags["low_confidence_bucket"] is False

    def test_inf_confidence_sanitised_to_none(self):
        flags = compute_reliability_flags(
            [1.0, 1.0, 1.0],
            pattern=_pattern(3),
            anomaly_confidence=float("inf"),
        )
        assert flags["confidence"] is None
        assert flags["low_confidence_bucket"] is False


class TestDominantDimFollowsDetectionMetric:
    def test_dominant_dim_matches_squared_delta_regardless_of_kinds(self):
        """``dominant_dim_share`` is the dim's share of ``delta_norm²`` —
        the metric behind ``is_anomaly`` — even on mixed-family patterns.

        - dim 0: Bernoulli, mu=0.001 (rare-positive prior), delta=10.
        - dim 1: Gaussian, mu=0, sigma=1, delta=2.

        delta² ranks dim 0 (100 vs 4, share 100/104). A family-level
        divergence would rank the Gaussian dim first because the
        Bernoulli term saturates; the kind tags must not change the
        attribution.
        """
        pat = Pattern(
            pattern_id="p_x",
            entity_type="x",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="bern_rare", direction="in", required=True),
                RelationDef(line_id="gauss_std", direction="in", required=True),
            ],
            mu=np.array([0.001, 0.0], dtype=np.float32),
            sigma_diag=np.array(
                [float(np.sqrt(0.001 * 0.999)), 1.0],
                dtype=np.float32,
            ),
            theta=np.array([3.0, 3.0], dtype=np.float32),
            population_size=100,
            computed_at=_DT,
            version=1,
            status="production",
            dimension_kinds=["bernoulli", "gaussian"],
        )
        flags = compute_reliability_flags([10.0, 2.0], pattern=pat)
        assert flags["dominant_dim"] == "bern_rare"
        assert flags["dominant_dim_share"] == pytest.approx(100.0 / 104.0, abs=1e-4)
        assert flags["single_dim_driven"] is True


class TestCrossSurfaceAgreement:
    """compute_reliability_flags must agree with build_explanation on the
    same polygon. Both surfaces are documented to share the per-dim
    contribution semantic — verify with a head-to-head comparison.
    """

    def test_dominant_dim_matches_explain_anomaly_top_dim_gaussian_only(self):
        from hypertopos.engine.investigation import build_explanation

        delta = np.array([3.0, 7.0, 1.0], dtype=np.float64)
        pat = _pattern(
            3,
            dim_labels_relations=["alpha", "beta", "gamma"],
            dimension_kinds=["gaussian", "gaussian", "gaussian"],
            with_calibration=True,
        )
        rf = compute_reliability_flags(
            delta,
            pattern=pat,
            anomaly_confidence=0.9,
        )
        exp = build_explanation(
            delta=delta,
            dim_labels=pat.dim_labels,
            theta_norm=2.0,
            delta_norm=float(np.linalg.norm(delta)),
            dimension_kinds=pat.dimension_kinds,
        )
        # explain_anomaly's top_dimensions[0]["dim"] is the highest-
        # contribution dim; reliability_flags.dominant_dim must match.
        assert exp["top_dimensions"], "build_explanation returned no top_dimensions"
        assert rf["dominant_dim"] == exp["top_dimensions"][0]["dim"]

    def test_dominant_dim_matches_explain_anomaly_top_dim_mixed_kinds(self):
        """Same comparison on a mixed-family (bernoulli + gaussian) pattern."""
        from hypertopos.engine.investigation import build_explanation

        pat = Pattern(
            pattern_id="p_x",
            entity_type="x",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="bern_rare", direction="in", required=True),
                RelationDef(line_id="gauss_std", direction="in", required=True),
            ],
            mu=np.array([0.001, 0.0], dtype=np.float32),
            sigma_diag=np.array(
                [float(np.sqrt(0.001 * 0.999)), 1.0],
                dtype=np.float32,
            ),
            theta=np.array([3.0, 3.0], dtype=np.float32),
            population_size=100,
            computed_at=_DT,
            version=1,
            status="production",
            dimension_kinds=["bernoulli", "gaussian"],
        )
        delta = np.array([10.0, 2.0], dtype=np.float64)
        rf = compute_reliability_flags(delta, pattern=pat)
        exp = build_explanation(
            delta=delta,
            dim_labels=pat.dim_labels,
            theta_norm=2.0,
            delta_norm=float(np.linalg.norm(delta)),
            dimension_kinds=pat.dimension_kinds,
        )
        assert exp["top_dimensions"], "build_explanation returned no top_dimensions"
        assert rf["dominant_dim"] == exp["top_dimensions"][0]["dim"]
        assert rf["dominant_dim_share"] == pytest.approx(
            exp["top_dimensions"][0]["pct_of_total"], abs=1e-4
        )


class TestCombinedFlags:
    def test_both_flags_fire_independently(self):
        flags = compute_reliability_flags(
            [10.0, 0.5, 0.5],
            pattern=_pattern(3),
            anomaly_confidence=0.2,
        )
        assert flags["single_dim_driven"] is True
        assert flags["low_confidence_bucket"] is True
        assert set(flags["flags"]) == {"single_dim_driven", "low_confidence_bucket"}

    def test_no_flags_clean_polygon(self):
        flags = compute_reliability_flags(
            [3.0, 3.0, 3.0],
            pattern=_pattern(3),
            anomaly_confidence=0.9,
        )
        assert flags["flags"] == []
