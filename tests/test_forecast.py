# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from hypertopos.engine.forecast import (
    AnomalyForecast,
    ForecastProvider,
    ForecastResult,
    PopulationForecast,
    SegmentCrossing,
    check_stale_forecast,
    extrapolate_trajectory,
    forecast_anomaly_status,
    forecast_segment_crossing,
    reliability_label,
)
from hypertopos.model.sphere import CuttingPlane


# ---------------------------------------------------------------------------
# extrapolate_trajectory
# ---------------------------------------------------------------------------
class TestExtrapolateTrajectory:
    def test_linear_increasing(self) -> None:
        """Linear increasing deltas => prediction should extrapolate linearly."""
        deltas = [
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            np.array([3.0, 6.0]),
            np.array([4.0, 8.0]),
            np.array([5.0, 10.0]),
        ]
        result = extrapolate_trajectory(deltas, horizon=1)
        assert isinstance(result, ForecastResult)
        assert result.horizon == 1
        # Should predict approximately [6.0, 12.0]
        np.testing.assert_allclose(result.predicted_delta, [6.0, 12.0], atol=0.5)
        assert result.predicted_delta_norm == pytest.approx(
            np.linalg.norm(result.predicted_delta), rel=1e-5
        )
        assert result.reliability in ("high", "medium", "low")
        assert 0.0 <= result.r_squared <= 1.0

    def test_constant_deltas(self) -> None:
        """Constant deltas => prediction should be the same constant."""
        deltas = [np.array([3.0, 5.0])] * 10
        result = extrapolate_trajectory(deltas, horizon=1)
        np.testing.assert_allclose(result.predicted_delta, [3.0, 5.0], atol=0.1)

    def test_single_slice(self) -> None:
        """Single slice => should still return a result, low reliability."""
        deltas = [np.array([2.0, 4.0])]
        result = extrapolate_trajectory(deltas, horizon=1)
        assert isinstance(result, ForecastResult)
        assert result.reliability == "low"
        # With a single point, prediction is the value itself
        np.testing.assert_allclose(result.predicted_delta, [2.0, 4.0], atol=0.1)

    def test_two_slices(self) -> None:
        """Two slices => basic linear extrapolation should work."""
        deltas = [np.array([1.0, 0.0]), np.array([3.0, 2.0])]
        result = extrapolate_trajectory(deltas, horizon=1)
        assert isinstance(result, ForecastResult)
        # Linear: next should be approximately [5.0, 4.0]
        np.testing.assert_allclose(result.predicted_delta, [5.0, 4.0], atol=0.5)

    def test_horizon_greater_than_one(self) -> None:
        """Horizon > 1 should extrapolate further."""
        deltas = [
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
            np.array([5.0]),
        ]
        result = extrapolate_trajectory(deltas, horizon=3)
        assert result.horizon == 3
        # Should predict approximately 8.0
        assert result.predicted_delta[0] == pytest.approx(8.0, abs=0.5)

    def test_r_squared_high_for_perfect_linear(self) -> None:
        """Perfect linear data should give r_squared close to 1."""
        deltas = [np.array([float(i)]) for i in range(20)]
        result = extrapolate_trajectory(deltas, horizon=1)
        assert result.r_squared > 0.9
        assert result.reliability == "high"


# ---------------------------------------------------------------------------
# forecast_anomaly_status
# ---------------------------------------------------------------------------
class TestForecastAnomalyStatus:
    def test_trending_toward_anomaly(self) -> None:
        """Entity with increasing delta_norm, will cross theta_norm."""
        deltas = [np.array([float(i)]) for i in range(1, 11)]
        theta_norm = 12.0
        result = forecast_anomaly_status(deltas, theta_norm, horizon=3)
        assert isinstance(result, AnomalyForecast)
        assert result.horizon == 3
        assert result.forecast_is_anomaly is True
        assert result.current_is_anomaly is False  # current delta_norm = 10

    def test_trending_away_from_anomaly(self) -> None:
        """Entity with decreasing delta_norm, moving away from threshold."""
        deltas = [np.array([float(10 - i)]) for i in range(10)]
        theta_norm = 5.0
        result = forecast_anomaly_status(deltas, theta_norm, horizon=1)
        assert isinstance(result, AnomalyForecast)
        # Current is delta_norm = 1.0, which is below threshold
        assert result.current_is_anomaly is False
        assert result.forecast_is_anomaly is False

    def test_currently_anomalous(self) -> None:
        """Entity already above threshold."""
        deltas = [np.array([20.0])] * 5
        theta_norm = 5.0
        result = forecast_anomaly_status(deltas, theta_norm, horizon=1)
        assert result.current_is_anomaly is True
        assert result.forecast_is_anomaly is True

    def test_current_is_anomaly_uses_base_not_last_slice(self) -> None:
        """current_is_anomaly must use base_delta_norm, not last temporal slice."""
        theta_norm = 3.0
        # last slice was anomalous (norm=3.5 > theta), but current base is recovered (norm=1.0)
        anomalous_delta = np.array([3.5, 0.0, 0.0])
        recovered_delta = np.array([1.0, 0.0, 0.0])
        deltas = [recovered_delta, anomalous_delta]  # last slice is anomalous
        base_delta_norm = float(np.linalg.norm(recovered_delta))  # 1.0

        result = forecast_anomaly_status(
            deltas, theta_norm, horizon=1, current_delta_norm=base_delta_norm
        )
        assert result.current_is_anomaly is False, (
            "current_is_anomaly must use base_delta_norm (1.0 < theta=3.0), not last slice (3.5)"
        )


# ---------------------------------------------------------------------------
# forecast_segment_crossing
# ---------------------------------------------------------------------------
class TestForecastSegmentCrossing:
    def test_crossing_boundary(self) -> None:
        """Entity moving from one side of cutting plane to the other."""
        # Cutting plane: x > 0 boundary at x=0
        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.0)
        # Delta moving from negative to positive side
        deltas = [np.array([-3.0, 0.0]), np.array([-2.0, 0.0]), np.array([-1.0, 0.0])]
        result = forecast_segment_crossing(deltas, {"seg_a": cp}, horizon=2)
        assert len(result) == 1
        crossing = result[0]
        assert isinstance(crossing, SegmentCrossing)
        assert crossing.cutting_plane_id == "seg_a"
        assert crossing.current_signed_dist < 0
        assert crossing.predicted_signed_dist > 0 or crossing.crosses_boundary is True

    def test_staying_on_same_side(self) -> None:
        """Entity moving parallel to boundary, staying on same side."""
        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.0)
        # Moving along y axis, staying positive x
        deltas = [np.array([5.0, 0.0]), np.array([5.0, 1.0]), np.array([5.0, 2.0])]
        result = forecast_segment_crossing(deltas, {"seg_a": cp}, horizon=1)
        assert len(result) == 1
        crossing = result[0]
        assert crossing.crosses_boundary is False

    def test_time_to_boundary_converging(self) -> None:
        """Entity converging toward boundary, time_to_boundary should be estimated."""
        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.0)
        deltas = [
            np.array([-5.0, 0.0]),
            np.array([-4.0, 0.0]),
            np.array([-3.0, 0.0]),
            np.array([-2.0, 0.0]),
            np.array([-1.0, 0.0]),
        ]
        result = forecast_segment_crossing(deltas, {"seg_a": cp}, horizon=1)
        crossing = result[0]
        assert crossing.time_to_boundary is not None
        assert crossing.time_to_boundary > 0

    def test_time_to_boundary_diverging(self) -> None:
        """Entity moving away from boundary, time_to_boundary should be None."""
        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.0)
        # Moving further into negative side
        deltas = [np.array([-1.0, 0.0]), np.array([-2.0, 0.0]), np.array([-3.0, 0.0])]
        result = forecast_segment_crossing(deltas, {"seg_a": cp}, horizon=1)
        crossing = result[0]
        assert crossing.time_to_boundary is None

    def test_multiple_cutting_planes(self) -> None:
        """Multiple cutting planes produce one SegmentCrossing each."""
        cp1 = CuttingPlane(normal=[1.0, 0.0], bias=0.0)
        cp2 = CuttingPlane(normal=[0.0, 1.0], bias=0.0)
        deltas = [np.array([1.0, 1.0])] * 5
        result = forecast_segment_crossing(deltas, {"seg_a": cp1, "seg_b": cp2}, horizon=1)
        assert len(result) == 2
        ids = {r.cutting_plane_id for r in result}
        assert ids == {"seg_a", "seg_b"}


# ---------------------------------------------------------------------------
# reliability_label
# ---------------------------------------------------------------------------
class TestReliabilityLabel:
    def test_high(self) -> None:
        assert reliability_label(10, 0.7) == "high"
        assert reliability_label(15, 0.9) == "high"

    def test_medium(self) -> None:
        assert reliability_label(5, 0.3) == "medium"
        assert reliability_label(3, 0.4) == "medium"
        assert reliability_label(8, 0.5) == "medium"

    def test_low(self) -> None:
        assert reliability_label(3, 0.2) == "low"
        assert reliability_label(1, 0.0) == "low"
        assert reliability_label(4, 0.39) == "low"


# ---------------------------------------------------------------------------
# ForecastProvider protocol
# ---------------------------------------------------------------------------
class TestForecastProvider:
    def test_protocol_isinstance_check(self) -> None:
        """A class implementing both methods satisfies ForecastProvider."""

        class _MyProvider:
            def forecast_entity(
                self,
                key: str,
                pattern_id: str,
                slices: list[np.ndarray],
                horizon: int,
            ) -> ForecastResult | None:
                return None

            def forecast_population(
                self,
                pattern_id: str,
                stats: list[float],
                horizon: int,
            ) -> PopulationForecast | None:
                return None

        assert isinstance(_MyProvider(), ForecastProvider)

    def test_non_conforming_object_fails(self) -> None:
        """An object without the required methods is not a ForecastProvider."""
        assert not isinstance(object(), ForecastProvider)


# ---------------------------------------------------------------------------
# HyperSession.forecast_provider
# ---------------------------------------------------------------------------
class TestHyperSessionForecastProvider:
    @staticmethod
    def _make_session():
        """Build a minimal HyperSession using mocks (no real sphere)."""
        from hypertopos.sphere import HyperSession

        return HyperSession(
            manifest=MagicMock(),
            contract=MagicMock(),
            engine=MagicMock(),
            reader=MagicMock(),
            writer=MagicMock(),
        )

    def test_default_is_none(self) -> None:
        session = self._make_session()
        assert session.forecast_provider is None

    def test_set_and_get_provider(self) -> None:
        session = self._make_session()
        provider = MagicMock(spec=ForecastProvider)
        session.set_forecast_provider(provider)
        assert session.forecast_provider is provider

    def test_reset_to_none(self) -> None:
        session = self._make_session()
        session.set_forecast_provider(MagicMock(spec=ForecastProvider))
        session.set_forecast_provider(None)
        assert session.forecast_provider is None


# ---------------------------------------------------------------------------
# check_stale_forecast
# ---------------------------------------------------------------------------
class TestCheckStaleForecast:
    def test_check_stale_forecast_recent(self) -> None:
        """Timestamp 10 days ago — no modification."""
        from datetime import UTC, timedelta
        from datetime import datetime as dt

        ts = dt.now(UTC) - timedelta(days=10)
        forecast = {"reliability": "high", "value": 42}
        result = check_stale_forecast(ts, forecast)
        assert result["reliability"] == "high"
        assert "stale_warning" not in result

    def test_check_stale_forecast_old(self) -> None:
        """Timestamp 200 days ago — reliability becomes 'low', stale_warning added."""
        from datetime import UTC, timedelta
        from datetime import datetime as dt

        ts = dt.now(UTC) - timedelta(days=200)
        forecast = {"reliability": "high", "value": 42}
        result = check_stale_forecast(ts, forecast)
        assert result["reliability"] == "low"
        assert "stale_warning" in result
        assert "200 days ago" in result["stale_warning"]
