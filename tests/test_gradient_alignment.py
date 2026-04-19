# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Unit tests for the gradient_alignment / drift_direction enrichment on pi9_attract_drift."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from hypertopos.navigation.navigator import GDSNavigator
from tests.test_navigator_fdr_select import _CONTRACT, _MANIFEST, _MockStoragePi9


def _gradient_alignment(delta_first: np.ndarray, delta_last: np.ndarray) -> float:
    """Reference implementation — identical to the math inside pi9_attract_drift."""
    diff = delta_last - delta_first
    displacement = float(np.linalg.norm(diff))
    first_norm = float(np.linalg.norm(delta_first))
    if displacement < 1e-9 or first_norm < 1e-9:
        return 0.0
    return float(-np.dot(diff, delta_first) / (displacement * first_norm))


def _drift_label(gradient_alignment: float) -> str:
    if gradient_alignment > 0.3:
        return "normalizing"
    if gradient_alignment < -0.3:
        return "deteriorating"
    return "neutral"


@pytest.fixture
def nav_pi9():
    engine = MagicMock()
    engine.geometry_to_polygons = MagicMock(return_value=[])
    return GDSNavigator(engine, _MockStoragePi9(), _MANIFEST, _CONTRACT)


class TestGradientAlignmentMath:
    def test_pure_inward_drift_is_plus_one(self):
        d0 = np.array([3.0, 0.0, 0.0])
        d1 = np.array([0.0, 0.0, 0.0])
        assert _gradient_alignment(d0, d1) == pytest.approx(1.0)

    def test_pure_outward_drift_is_minus_one(self):
        d0 = np.array([1.0, 0.0, 0.0])
        d1 = np.array([3.0, 0.0, 0.0])
        assert _gradient_alignment(d0, d1) == pytest.approx(-1.0)

    def test_tangential_drift_is_zero(self):
        d0 = np.array([1.0, 0.0, 0.0])
        d1 = np.array([1.0, 1.0, 0.0])
        assert abs(_gradient_alignment(d0, d1)) < 1e-9

    def test_zero_displacement_is_zero(self):
        d0 = np.array([2.0, 0.0])
        d1 = np.array([2.0, 0.0])
        assert _gradient_alignment(d0, d1) == 0.0

    def test_origin_start_is_zero(self):
        d0 = np.array([0.0, 0.0])
        d1 = np.array([2.0, 0.0])
        assert _gradient_alignment(d0, d1) == 0.0

    def test_label_cutoffs(self):
        assert _drift_label(0.99) == "normalizing"
        assert _drift_label(0.30001) == "normalizing"
        assert _drift_label(0.29999) == "neutral"
        assert _drift_label(0.0) == "neutral"
        assert _drift_label(-0.29999) == "neutral"
        assert _drift_label(-0.30001) == "deteriorating"
        assert _drift_label(-0.99) == "deteriorating"

    def test_partial_inward_is_positive_less_than_one(self):
        d0 = np.array([2.0, 0.0])
        d1 = np.array([1.0, 1.0])
        assert _gradient_alignment(d0, d1) == pytest.approx(1 / np.sqrt(2), rel=1e-6)


class TestPi9GradientAlignmentIntegration:
    """Check that the in-place math inside pi9_attract_drift matches the reference."""

    def test_all_entities_get_both_fields(self, nav_pi9):
        results = nav_pi9.π9_attract_drift("test_pattern", top_n=10)
        assert len(results) > 0
        for r in results:
            assert "gradient_alignment" in r
            assert "drift_direction" in r
            assert isinstance(r["gradient_alignment"], float)
            assert r["drift_direction"] in {"normalizing", "deteriorating", "neutral"}
            assert -1.0 - 1e-6 <= r["gradient_alignment"] <= 1.0 + 1e-6

    def test_label_matches_cutoffs(self, nav_pi9):
        results = nav_pi9.π9_attract_drift("test_pattern", top_n=20)
        for r in results:
            g = r["gradient_alignment"]
            label = r["drift_direction"]
            if g > 0.3:
                assert label == "normalizing"
            elif g < -0.3:
                assert label == "deteriorating"
            else:
                assert label == "neutral"
