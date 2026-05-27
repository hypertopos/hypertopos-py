# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for cluster C3 detector closures.

C3a — boundary-aware sampling on find_anomalies.
C3b — DTW trajectory classifier (engine.topology.classify_trajectory).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
from hypertopos.engine.topology import classify_trajectory
from hypertopos.navigation.navigator import GDSNavigator

# --------------------------------------------------------------------------
# C3a — boundary-aware sampling
# --------------------------------------------------------------------------


def _engineered_light_table(
    *,
    n_boundary: int,
    n_anomalous: int,
    n_normal: int,
    theta_norm: float,
    seed: int = 0,
) -> pa.Table:
    """Build a synthetic `light` Arrow table with engineered delta_norm bands.

    Three strata:
      - boundary: delta_norm uniformly in [0.85 * theta, 1.15 * theta]
      - anomalous: delta_norm uniformly in [1.5 * theta, 3.0 * theta]
      - normal: delta_norm uniformly in [0.1 * theta, 0.5 * theta]

    Other columns are placeholders sized to match the LIGHT_COLUMNS contract
    so `_stratified_sample_light.take` works without further plumbing.
    """
    rng = np.random.default_rng(seed)
    norms_boundary = rng.uniform(0.85 * theta_norm, 1.15 * theta_norm, n_boundary)
    norms_anom = rng.uniform(1.5 * theta_norm, 3.0 * theta_norm, n_anomalous)
    norms_normal = rng.uniform(0.1 * theta_norm, 0.5 * theta_norm, n_normal)
    all_norms = np.concatenate([norms_boundary, norms_anom, norms_normal])
    n_total = all_norms.shape[0]
    pks = [f"E{i:04d}" for i in range(n_total)]
    # Build a (n_total, 2) delta vector with norm matching all_norms (not
    # critical for the sampling helper — it only reads delta_norm).
    delta = np.zeros((n_total, 2), dtype=np.float32)
    delta[:, 0] = all_norms.astype(np.float32)
    return pa.table({
        "primary_key": pa.array(pks, type=pa.string()),
        "delta": pa.array(delta.tolist(), type=pa.list_(pa.float32())),
        "delta_norm": pa.array(all_norms.astype(np.float32)),
        "is_anomaly": pa.array(all_norms >= theta_norm),
        "delta_rank_pct": pa.array(np.linspace(0.0, 1.0, n_total, dtype=np.float32)),
    })


def _count_boundary(sample: pa.Table, theta_norm: float) -> int:
    norms = sample["delta_norm"].to_numpy(zero_copy_only=False)
    lo = 0.8 * theta_norm
    hi = 1.2 * theta_norm
    return int(np.sum((norms >= lo) & (norms <= hi)))


def test_stratified_sample_boundary_aware_hits_budget():
    theta = 1.0
    light = _engineered_light_table(
        n_boundary=300, n_anomalous=300, n_normal=400, theta_norm=theta,
    )
    out = GDSNavigator._stratified_sample_light(
        light, sample_size=100, boundary_aware=True, theta_norm=theta,
    )
    assert out.num_rows == 100
    boundary_count = _count_boundary(out, theta)
    # Spec: ~50 boundary entries with budget split 50/50 and 300 boundary
    # available (>= half budget). Should be exactly 50.
    assert boundary_count == 50, f"expected 50, got {boundary_count}"


def test_stratified_sample_uniform_proportional_boundary_ratio():
    theta = 1.0
    light = _engineered_light_table(
        n_boundary=300, n_anomalous=300, n_normal=400, theta_norm=theta,
    )
    out = GDSNavigator._stratified_sample_light(
        light, sample_size=100, boundary_aware=False, theta_norm=theta,
    )
    assert out.num_rows == 100
    boundary_count = _count_boundary(out, theta)
    # Uniform random over 1000 entities, 300 boundary → expect ~30.
    # Tolerate 20 ≤ x ≤ 40 (deterministic seed 0 should give ~30).
    assert 20 <= boundary_count <= 40, (
        f"expected ~30 (uniform proportional), got {boundary_count}"
    )


def test_stratified_sample_spills_when_boundary_short():
    """When boundary stratum has fewer entries than half-budget, the leftover
    must spill to the other stratum to keep total budget == sample_size."""
    theta = 1.0
    light = _engineered_light_table(
        n_boundary=10, n_anomalous=300, n_normal=400, theta_norm=theta,
    )
    out = GDSNavigator._stratified_sample_light(
        light, sample_size=100, boundary_aware=True, theta_norm=theta,
    )
    # All 10 boundary entries should be picked, plus 90 from the rest.
    assert out.num_rows == 100
    boundary_count = _count_boundary(out, theta)
    assert boundary_count == 10


def test_stratified_sample_noop_when_under_budget():
    theta = 1.0
    light = _engineered_light_table(
        n_boundary=5, n_anomalous=5, n_normal=5, theta_norm=theta,
    )
    out = GDSNavigator._stratified_sample_light(
        light, sample_size=100, boundary_aware=True, theta_norm=theta,
    )
    # 15 rows total, budget 100 — return as-is.
    assert out.num_rows == 15


# --------------------------------------------------------------------------
# C3b — DTW trajectory classifier
# --------------------------------------------------------------------------


def _make_solid_table(trajectories: dict[str, list[list[float]]]) -> pa.Table:
    pks: list[str] = []
    snaps: list[list[float]] = []
    for pk, traj in trajectories.items():
        for snap in traj:
            pks.append(pk)
            snaps.append(list(snap))
    return pa.table({
        "primary_key": pa.array(pks, type=pa.string()),
        "delta_snapshot": pa.array(snaps, type=pa.list_(pa.float64())),
    })


def test_classify_trajectory_three_categories_identified():
    """3-solid synthetic fixture: outlier-shaped, lagging-slope, leading-slope.

    Need a population large enough so the median-trajectory reference is
    informative; we pad with 'typical' baseline trajectories.
    """
    T = 8
    # Baseline (typical): slope ≈ 1.0 per step, around the median.
    baseline = [[float(t)] for t in range(T)]
    typical_population = {f"T{i}": baseline for i in range(20)}

    # Lagging: slope ≈ 0.2 per step (well below median slope 1.0).
    lagging = [[0.2 * t] for t in range(T)]
    # Leading: slope ≈ 1.8 per step (well above median slope 1.0).
    leading = [[1.8 * t] for t in range(T)]
    # Outlier: large constant offset → DTW dominates over slope; slope
    # also matches baseline so it doesn't accidentally land in lagging/leading.
    outlier = [[100.0 + float(t)] for t in range(T)]

    trajectories = {
        **typical_population,
        "LAG": lagging,
        "LEAD": leading,
        "OUT": outlier,
    }
    tbl = _make_solid_table(trajectories)
    results = classify_trajectory(tbl, sample_size=10_000)
    by_pk = {r["primary_key"]: r for r in results}

    assert by_pk["OUT"]["category"] == "outlier", (
        f"OUT category: {by_pk['OUT']}"
    )
    assert by_pk["LAG"]["category"] == "lagging", (
        f"LAG category: {by_pk['LAG']}"
    )
    assert by_pk["LEAD"]["category"] == "leading", (
        f"LEAD category: {by_pk['LEAD']}"
    )

    # Evidence signs match category direction.
    assert by_pk["LAG"]["category_evidence"] < 0
    assert by_pk["LEAD"]["category_evidence"] > 0
    assert by_pk["OUT"]["category_evidence"] > 0

    # No NaN / inf in output.
    for r in results:
        assert np.isfinite(r["dtw_distance"])
        assert np.isfinite(r["category_evidence"])


def test_classify_trajectory_empty_table_returns_empty():
    tbl = _make_solid_table({})
    assert classify_trajectory(tbl) == []


def test_classify_trajectory_missing_columns_raises():
    import pytest

    bad_tbl = pa.table({"primary_key": pa.array(["A"])})
    with pytest.raises(ValueError, match="delta_snapshot"):
        classify_trajectory(bad_tbl)


def test_classify_trajectory_returns_one_entry_per_entity():
    """Every entity in the input table gets exactly one classification row."""
    T = 6
    trajectories = {
        f"E{i}": [[float(j) + 0.1 * i] for j in range(T)] for i in range(10)
    }
    tbl = _make_solid_table(trajectories)
    results = classify_trajectory(tbl)
    assert len(results) == 10
    assert {r["primary_key"] for r in results} == set(trajectories.keys())
