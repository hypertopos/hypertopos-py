# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Trajectory extrapolation and forecast primitives for GDS solids.

All functions are pure (no I/O, no storage access). They operate on numpy
arrays extracted from SolidSlice sequences and produce lightweight dataclass
results suitable for agent consumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from hypertopos.model.sphere import CuttingPlane

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    horizon: int
    predicted_delta: np.ndarray       # d_{n+h}
    predicted_delta_norm: float       # ||d_{n+h}||
    reliability: str                  # "high" / "medium" / "low"
    r_squared: float                  # goodness of fit


@dataclass
class AnomalyForecast:
    horizon: int
    predicted_delta_norm: float
    forecast_is_anomaly: bool         # theta_norm > 0 and predicted_delta_norm >= theta_norm
    current_is_anomaly: bool          # theta_norm > 0 and current_delta_norm >= theta_norm
    reliability: str


@dataclass
class SegmentCrossing:
    cutting_plane_id: str
    current_signed_dist: float
    predicted_signed_dist: float
    crosses_boundary: bool            # sign change
    time_to_boundary: int | None      # versions until crossing (None if not converging)
    reliability: str


@dataclass
class PopulationForecast:
    metric: str                       # "anomaly_rate", "mean_delta_norm", "entity_count"
    current_value: float
    forecast_value: float
    horizon: int
    direction: str                    # "rising", "falling", "stable"
    reliability: str


# ---------------------------------------------------------------------------
# Reliability helper
# ---------------------------------------------------------------------------

def reliability_label(n_slices: int, r_squared: float) -> str:
    """Classify forecast reliability based on data quantity and fit quality.

    - slices >= 10 AND r_squared >= 0.7 -> "high"
    - slices >= 5 OR  r_squared >= 0.4 -> "medium"
    - else -> "low"
    """
    if n_slices >= 10 and r_squared >= 0.7:
        return "high"
    if n_slices >= 5 or r_squared >= 0.4:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Core extrapolation
# ---------------------------------------------------------------------------

_DEFAULT_ALPHA = 0.3


def _weighted_linear_regression(
    t: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """Fit weighted linear regression y = a*t + b.

    Returns (slope, intercept, r_squared).
    """
    w = weights / weights.sum()
    t_mean = np.dot(w, t)
    y_mean = np.dot(w, y)
    t_centered = t - t_mean
    y_centered = y - y_mean
    ss_tt = np.dot(w, t_centered ** 2)
    ss_ty = np.dot(w, t_centered * y_centered)

    if ss_tt < 1e-15:
        # All time points identical or single point — no slope
        return 0.0, float(y_mean), 0.0

    slope = float(ss_ty / ss_tt)
    intercept = float(y_mean - slope * t_mean)

    # Weighted R²
    y_pred = slope * t + intercept
    ss_res = np.dot(w, (y - y_pred) ** 2)
    ss_tot = np.dot(w, y_centered ** 2)
    r_squared = (
        1.0 if ss_tot < 1e-15 else max(0.0, 1.0 - ss_res / ss_tot)
    )  # constant data, perfect fit

    return slope, intercept, r_squared


def extrapolate_trajectory(
    deltas: list[np.ndarray],
    horizon: int = 1,
    alpha: float = _DEFAULT_ALPHA,
) -> ForecastResult:
    """Extrapolate entity trajectory from a sequence of delta vectors.

    Parameters
    ----------
    deltas:
        Sequence of delta vectors d_1, ..., d_n (from SolidSlice.delta_snapshot).
    horizon:
        Number of versions ahead to predict.
    alpha:
        Exponential decay rate for weighting (higher = more weight on recent).

    Returns
    -------
    ForecastResult with predicted delta vector at t = n + horizon.
    Cost: O(S * d) where S = len(deltas), d = delta dimensionality.
    """
    n = len(deltas)
    d = len(deltas[0])

    t = np.arange(1, n + 1, dtype=np.float64)
    weights = np.exp(-alpha * (n - t))  # w_i = exp(-alpha * (n - i))

    # Stack deltas into (n, d) matrix
    delta_matrix = np.stack([dd.astype(np.float64) for dd in deltas])

    predicted = np.zeros(d, dtype=np.float64)
    r_squared_sum = 0.0

    for dim in range(d):
        y = delta_matrix[:, dim]
        slope, intercept, r_sq = _weighted_linear_regression(t, y, weights)
        predicted[dim] = slope * (n + horizon) + intercept
        r_squared_sum += r_sq

    r_squared_avg = r_squared_sum / d if d > 0 else 0.0
    predicted_f32 = predicted.astype(np.float32)

    return ForecastResult(
        horizon=horizon,
        predicted_delta=predicted_f32,
        predicted_delta_norm=float(np.linalg.norm(predicted_f32)),
        reliability=reliability_label(n, r_squared_avg),
        r_squared=r_squared_avg,
    )


# ---------------------------------------------------------------------------
# Anomaly forecast
# ---------------------------------------------------------------------------

def forecast_anomaly_status(
    deltas: list[np.ndarray],
    theta_norm: float,
    horizon: int = 1,
    current_delta_norm: float | None = None,
) -> AnomalyForecast:
    """Predict whether an entity will be anomalous at t + horizon.

    Parameters
    ----------
    deltas:
        Sequence of delta vectors from SolidSlice history.
    theta_norm:
        Anomaly threshold (||theta||) from the pattern.
    horizon:
        Versions ahead to predict.
    current_delta_norm:
        The base polygon's current delta norm. When provided, used for
        ``current_is_anomaly`` instead of the last slice norm. Pass
        ``solid.base_polygon.delta_norm`` to avoid stale-slice false positives.
        Defaults to ``norm(deltas[-1])`` when not provided.
    """
    result = extrapolate_trajectory(deltas, horizon=horizon)
    if current_delta_norm is None:
        current_delta_norm = float(np.linalg.norm(deltas[-1]))
    return AnomalyForecast(
        horizon=horizon,
        predicted_delta_norm=result.predicted_delta_norm,
        forecast_is_anomaly=theta_norm > 0.0 and result.predicted_delta_norm >= theta_norm,
        current_is_anomaly=theta_norm > 0.0 and current_delta_norm >= theta_norm,
        reliability=result.reliability,
    )


# ---------------------------------------------------------------------------
# Segment crossing forecast
# ---------------------------------------------------------------------------

def forecast_segment_crossing(
    deltas: list[np.ndarray],
    cutting_planes: dict[str, CuttingPlane],
    horizon: int = 1,
) -> list[SegmentCrossing]:
    """Predict whether an entity's trajectory will cross segment boundaries.

    For each cutting plane, projects the trajectory onto the plane's normal
    vector and performs linear extrapolation of the signed distance.

    Parameters
    ----------
    deltas:
        Sequence of delta vectors from SolidSlice history.
    cutting_planes:
        Mapping of cutting_plane_id -> CuttingPlane.
    horizon:
        Versions ahead to predict.
    """
    n = len(deltas)
    t = np.arange(1, n + 1, dtype=np.float64)
    weights = np.exp(-_DEFAULT_ALPHA * (n - t))

    results: list[SegmentCrossing] = []

    for cp_id, cp in cutting_planes.items():
        # Compute signed distances for each delta
        signed_dists = np.array(
            [cp.signed_distance(d) for d in deltas], dtype=np.float64
        )

        current_sd = float(signed_dists[-1])

        # Weighted linear regression on signed distances
        slope, intercept, r_sq = _weighted_linear_regression(t, signed_dists, weights)
        predicted_sd = slope * (n + horizon) + intercept

        # Sign change?
        crosses = (current_sd * predicted_sd) < 0

        # Time to boundary: find t* where slope * t* + intercept = 0
        time_to_boundary: int | None = None
        if abs(slope) > 1e-12:
            t_cross = -intercept / slope
            if t_cross > n:
                # Boundary is in the future — converging
                time_to_boundary = max(1, int(np.ceil(t_cross - n)))
            # If t_cross <= n, the crossing is in the past or already at boundary

        results.append(SegmentCrossing(
            cutting_plane_id=cp_id,
            current_signed_dist=current_sd,
            predicted_signed_dist=float(predicted_sd),
            crosses_boundary=crosses,
            time_to_boundary=time_to_boundary,
            reliability=reliability_label(n, r_sq),
        ))

    return results


# ---------------------------------------------------------------------------
# ForecastProvider protocol — pluggable forecast backend
# ---------------------------------------------------------------------------

@runtime_checkable
class ForecastProvider(Protocol):
    """Protocol for external forecast providers.

    Either the built-in geometric extrapolation (default) or a registered
    external provider is used — no fallback chain.
    """

    def forecast_entity(
        self,
        key: str,
        pattern_id: str,
        slices: list[np.ndarray],
        horizon: int,
    ) -> ForecastResult | None: ...

    def forecast_population(
        self,
        pattern_id: str,
        stats: list[float],
        horizon: int,
    ) -> PopulationForecast | None: ...


# ---------------------------------------------------------------------------
# Stale forecast check
# ---------------------------------------------------------------------------

_STALE_THRESHOLD_DAYS = 180


def check_stale_forecast(
    last_timestamp: datetime,
    forecast: dict,
) -> dict:
    """Override reliability to 'low' and add stale_warning if temporal data is old."""
    from datetime import UTC

    age_days = (datetime.now(UTC) - last_timestamp).days
    if age_days > _STALE_THRESHOLD_DAYS:
        forecast["reliability"] = "low"
        forecast["stale_warning"] = (
            f"Forecast extrapolated from slices ending {last_timestamp.date().isoformat()} "
            f"({age_days} days ago). Drift trajectory may no longer reflect current behavior."
        )
    return forecast
