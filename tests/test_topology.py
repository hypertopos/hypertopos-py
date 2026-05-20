# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for engine/topology.trajectory_continuous_score."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from hypertopos.engine.topology import trajectory_continuous_score


def _make_solid_table(trajectories: dict[str, list[list[float]]]) -> pa.Table:
    """Build a synthetic solid-table with cols (primary_key, t, delta_snapshot).

    delta_snapshot is a list<float> per row.
    """
    pks: list[str] = []
    ts: list[int] = []
    snapshots: list[list[float]] = []
    for pk, traj in trajectories.items():
        for t, snap in enumerate(traj):
            pks.append(pk)
            ts.append(t)
            snapshots.append(list(snap))
    return pa.table(
        {
            "primary_key": pks,
            "t": ts,
            "delta_snapshot": snapshots,
        }
    )


class TestTrajectoryContinuousScore:
    def test_outlier_ranks_highest(self):
        # 4 entities follow the same baseline trajectory; 1 outlier is far away.
        baseline = [[0.0, 0.0], [0.1, 0.1], [0.0, 0.0], [0.1, 0.0]]
        outlier = [[5.0, 5.0], [5.1, 4.9], [5.2, 5.1], [5.0, 5.0]]
        trajectories = {
            "A": baseline,
            "B": baseline,
            "C": baseline,
            "D": baseline,
            "OUT": outlier,
        }
        tbl = _make_solid_table(trajectories)
        scores = trajectory_continuous_score(tbl)
        assert set(scores.keys()) == {"A", "B", "C", "D", "OUT"}
        # Outlier has strictly higher DTW distance to median trajectory.
        for pk in ("A", "B", "C", "D"):
            assert scores["OUT"] > scores[pk]
        # All scores >= 0
        for s in scores.values():
            assert s >= 0.0
            assert np.isfinite(s)

    def test_empty_solid_table(self):
        tbl = _make_solid_table({})
        scores = trajectory_continuous_score(tbl)
        assert scores == {}

    def test_single_entity_returns_zero(self):
        # Median trajectory equals its own — distance is zero.
        tbl = _make_solid_table({"A": [[1.0, 1.0], [2.0, 2.0]]})
        scores = trajectory_continuous_score(tbl)
        assert "A" in scores
        assert scores["A"] == 0.0

    def test_identical_trajectories_zero_score(self):
        baseline = [[0.0], [1.0], [2.0]]
        tbl = _make_solid_table({f"E{i}": baseline for i in range(5)})
        scores = trajectory_continuous_score(tbl)
        for s in scores.values():
            assert s == 0.0

    def test_sample_size_caps_population(self):
        rng = np.random.default_rng(0)
        traj_len = 4
        trajectories = {
            f"E{i}": rng.normal(0.0, 1.0, (traj_len, 2)).tolist()
            for i in range(20)
        }
        tbl = _make_solid_table(trajectories)
        scores = trajectory_continuous_score(tbl, sample_size=5)
        # All entities receive a score (sample is for median estimation only)
        assert len(scores) == 20

    def test_no_nan_or_inf(self):
        rng = np.random.default_rng(7)
        trajectories = {
            f"E{i}": rng.normal(0.0, 1.0, (3, 2)).tolist() for i in range(8)
        }
        tbl = _make_solid_table(trajectories)
        scores = trajectory_continuous_score(tbl)
        for s in scores.values():
            assert np.isfinite(s)
            assert s >= 0.0
