"""TDD tests for M3.2 find_topological_trajectory_anomalies.

Per-entity solid-PH on (T x D) trajectory matrices. Tested geometry:
- circular trajectory: T points completing a closed loop → high H_1 → high score
- monotonic trajectory: T points drifting linearly → no H_1 → low score
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest


def _circular_trajectory(T: int, radius: float = 1.0, dim: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2 * np.pi, T, endpoint=True)
    traj = np.zeros((T, dim))
    traj[:, 0] = radius * np.cos(theta)
    traj[:, 1] = radius * np.sin(theta)
    traj += rng.normal(0.0, 0.005, (T, dim))
    return traj


def _monotonic_trajectory(T: int, dim: int = 5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, T)
    traj = np.zeros((T, dim))
    traj[:, 0] = base
    traj += rng.normal(0.0, 0.005, (T, dim))
    return traj


def _build_solid_table(trajectories: dict[str, np.ndarray]) -> pa.Table:
    pks: list[str] = []
    snaps: list[list[float]] = []
    for pk, traj in trajectories.items():
        for t in range(traj.shape[0]):
            pks.append(pk)
            snaps.append(traj[t].tolist())
    return pa.table({
        "primary_key": pa.array(pks, type=pa.string()),
        "delta_snapshot": pa.array(snaps, type=pa.list_(pa.float64())),
    })


def test_circular_trajectory_dominates_top_n():
    from hypertopos.engine.topology import find_topological_trajectory_anomalies

    trajectories: dict[str, np.ndarray] = {}
    for i in range(30):
        trajectories[f"C{i}"] = _circular_trajectory(T=25, seed=i)
    for i in range(70):
        trajectories[f"M{i}"] = _monotonic_trajectory(T=25, seed=100 + i)

    solid_table = _build_solid_table(trajectories)
    result = find_topological_trajectory_anomalies(
        solid_table, top_n=30, min_timesteps=8, pca_dim=5, sample_size=100,
    )

    top_pks = {r["primary_key"] for r in result}
    circular_pks = {f"C{i}" for i in range(30)}
    overlap = len(top_pks & circular_pks)
    assert overlap >= 25, (
        f"only {overlap}/30 top entities are circular trajectories; "
        f"top: {sorted(top_pks)[:10]}"
    )


def test_entity_below_min_timesteps_excluded():
    from hypertopos.engine.topology import find_topological_trajectory_anomalies

    trajectories: dict[str, np.ndarray] = {
        "TINY": _circular_trajectory(T=4, seed=0),
        "GOOD1": _circular_trajectory(T=20, seed=1),
        "GOOD2": _monotonic_trajectory(T=20, seed=2),
    }
    solid_table = _build_solid_table(trajectories)

    result = find_topological_trajectory_anomalies(
        solid_table, top_n=5, min_timesteps=8,
    )
    top_pks = {r["primary_key"] for r in result}
    assert "TINY" not in top_pks
    assert len(result) <= 2


def test_return_shape_and_fields():
    from hypertopos.engine.topology import find_topological_trajectory_anomalies

    trajectories = {
        f"E{i}": _circular_trajectory(T=20, seed=i) for i in range(10)
    }
    solid_table = _build_solid_table(trajectories)

    result = find_topological_trajectory_anomalies(
        solid_table, top_n=5, min_timesteps=8, pca_dim=5,
    )
    assert isinstance(result, list)
    assert len(result) == 5
    required = {
        "primary_key", "trajectory_topo_score", "n_timesteps",
        "h1_total_persistence", "dominant_feature_birth",
        "dominant_feature_death", "computed_at",
    }
    for row in result:
        assert required.issubset(row.keys()), f"missing: {required - row.keys()}"
        assert isinstance(row["primary_key"], str)
        assert np.isfinite(row["trajectory_topo_score"])
        assert row["trajectory_topo_score"] >= 0.0
        assert row["n_timesteps"] >= 8


def test_empty_table_returns_empty():
    from hypertopos.engine.topology import find_topological_trajectory_anomalies

    empty = pa.table({
        "primary_key": pa.array([], type=pa.string()),
        "delta_snapshot": pa.array([], type=pa.list_(pa.float64())),
    })
    assert find_topological_trajectory_anomalies(empty, top_n=10) == []
