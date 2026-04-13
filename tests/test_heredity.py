# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for geometric heredity — expected-position and novelty scoring."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import lance
import numpy as np
import pyarrow as pa
import pytest

from hypertopos.engine.heredity import (
    compute_expected_delta,
    compute_novelty_decomposition,
    compute_novelty_score,
)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestComputeExpectedDelta:
    def test_expected_delta_from_neighbors(self) -> None:
        neighbor_deltas = np.array(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32,
        )
        expected = compute_expected_delta(neighbor_deltas)
        np.testing.assert_allclose(expected, [2.0, 3.0, 4.0])

    def test_expected_delta_weighted(self) -> None:
        neighbor_deltas = np.array(
            [[1.0, 0.0], [3.0, 0.0]], dtype=np.float32,
        )
        weights = np.array([3.0, 1.0], dtype=np.float32)
        expected = compute_expected_delta(neighbor_deltas, weights=weights)
        np.testing.assert_allclose(expected, [1.5, 0.0])

    def test_expected_delta_no_neighbors_returns_zeros(self) -> None:
        neighbor_deltas = np.empty((0, 3), dtype=np.float32)
        expected = compute_expected_delta(neighbor_deltas)
        np.testing.assert_allclose(expected, [0.0, 0.0, 0.0])

    def test_expected_delta_single_neighbor(self) -> None:
        neighbor_deltas = np.array([[5.0, 6.0]], dtype=np.float32)
        expected = compute_expected_delta(neighbor_deltas)
        np.testing.assert_allclose(expected, [5.0, 6.0])

    def test_expected_delta_zero_weights_returns_zeros(self) -> None:
        neighbor_deltas = np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32,
        )
        weights = np.array([0.0, 0.0], dtype=np.float32)
        expected = compute_expected_delta(neighbor_deltas, weights=weights)
        np.testing.assert_allclose(expected, [0.0, 0.0])


class TestComputeNoveltyScore:
    def test_novelty_score(self) -> None:
        actual = np.array([4.0, 3.0], dtype=np.float32)
        expected = np.array([1.0, 3.0], dtype=np.float32)
        score = compute_novelty_score(actual, expected)
        np.testing.assert_allclose(score, 3.0)

    def test_novelty_score_identical(self) -> None:
        delta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        score = compute_novelty_score(delta, delta)
        assert score == pytest.approx(0.0)

    def test_novelty_score_multidimensional(self) -> None:
        actual = np.array([3.0, 4.0], dtype=np.float32)
        expected = np.array([0.0, 0.0], dtype=np.float32)
        score = compute_novelty_score(actual, expected)
        np.testing.assert_allclose(score, 5.0)


class TestComputeNoveltyDecomposition:
    def test_decomposition_sorted_by_deviation(self) -> None:
        actual = np.array([5.0, 1.0, 10.0], dtype=np.float32)
        expected = np.array([2.0, 1.0, 3.0], dtype=np.float32)
        names = ["dim_a", "dim_b", "dim_c"]
        result = compute_novelty_decomposition(actual, expected, names)
        assert len(result) == 3
        # dim_c has deviation 7.0, dim_a has 3.0, dim_b has 0.0
        assert result[0]["dimension"] == "dim_c"
        assert result[0]["deviation"] == pytest.approx(7.0)
        assert result[1]["dimension"] == "dim_a"
        assert result[1]["deviation"] == pytest.approx(3.0)
        assert result[2]["dimension"] == "dim_b"
        assert result[2]["deviation"] == pytest.approx(0.0)

    def test_decomposition_has_all_fields(self) -> None:
        actual = np.array([2.0], dtype=np.float32)
        expected = np.array([1.0], dtype=np.float32)
        result = compute_novelty_decomposition(actual, expected, ["x"])
        assert len(result) == 1
        entry = result[0]
        assert set(entry.keys()) == {"dimension", "expected", "actual", "deviation"}
        assert entry["actual"] == pytest.approx(2.0)
        assert entry["expected"] == pytest.approx(1.0)
        assert entry["deviation"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Navigator integration test
# ---------------------------------------------------------------------------


def _build_edge_sphere(tmp_path: Path) -> Path:
    """Create a minimal sphere with edge table for integration testing."""
    sphere_dir = tmp_path / "test_sphere"

    # sphere.json
    sphere = {
        "sphere_id": "heredity_test",
        "name": "Heredity Test Sphere",
        "lines": {
            "accounts": {
                "line_id": "accounts",
                "entity_type": "account",
                "line_role": "anchor",
                "pattern_id": "account_pattern",
                "partitioning": {"mode": "static", "columns": []},
                "versions": [1],
            },
            "transactions": {
                "line_id": "transactions",
                "entity_type": "transaction",
                "line_role": "event",
                "pattern_id": "tx_pattern",
                "partitioning": {"mode": "static", "columns": []},
                "versions": [1],
            },
        },
        "patterns": {
            "account_pattern": {
                "pattern_id": "account_pattern",
                "entity_type": "account",
                "entity_line": "accounts",
                "pattern_type": "anchor",
                "version": 1,
                "status": "production",
                "relations": [],
                "mu": [0.5, 0.5],
                "sigma_diag": [0.2, 0.2],
                "theta": [5.0, 5.0],
                "population_size": 5,
                "computed_at": "2024-01-01T00:00:00+00:00",
            },
            "tx_pattern": {
                "pattern_id": "tx_pattern",
                "entity_type": "transaction",
                "pattern_type": "event",
                "version": 1,
                "status": "production",
                "relations": [
                    {"line_id": "accounts", "direction": "out", "required": True},
                ],
                "mu": [0.5],
                "sigma_diag": [0.2],
                "theta": [5.0],
                "population_size": 10,
                "computed_at": "2024-01-01T00:00:00+00:00",
            },
        },
        "aliases": {},
        "storage": {
            "geometry": {"format": "lance"},
            "points": {"format": "lance"},
        },
    }
    meta_dir = sphere_dir / "_gds_meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "sphere.json").write_text(json.dumps(sphere, indent=2))

    # Geometry for account_pattern — 5 accounts with 2D deltas
    # A is in the centre, B/C/D are neighbors of A, E is far away
    edge_struct_type = pa.struct([
        pa.field("line_id", pa.string()),
        pa.field("point_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("direction", pa.string()),
    ])
    deltas = [
        [1.0, 1.0],   # A — neighbors B,C,D avg=[2,2] → score=sqrt(2)≈1.414
        [2.0, 2.0],   # B — neighbor of A
        [2.0, 2.0],   # C — neighbor of A
        [2.0, 2.0],   # D — neighbor of A
        [10.0, 10.0],  # E — neighbor of A, very different
    ]
    delta_norms = [float(np.linalg.norm(d)) for d in deltas]
    sorted_norms = np.sort(delta_norms)
    ranks = np.searchsorted(sorted_norms, delta_norms, side="right")
    rank_pcts = [float(r / len(delta_norms) * 100) for r in ranks]
    theta_norm = float(np.linalg.norm([5.0, 5.0]))
    is_anomaly = [n > theta_norm for n in delta_norms]

    from datetime import UTC, datetime

    ts = datetime(2024, 1, 1, tzinfo=UTC)
    empty_edges = [[] for _ in range(5)]

    geo_table = pa.table({
        "primary_key": ["A", "B", "C", "D", "E"],
        "scale": [1, 1, 1, 1, 1],
        "delta": [d for d in deltas],
        "delta_norm": pa.array(delta_norms, type=pa.float32()),
        "delta_rank_pct": pa.array(rank_pcts, type=pa.float32()),
        "is_anomaly": is_anomaly,
        "edges": pa.array(empty_edges, type=pa.list_(edge_struct_type)),
        "last_refresh_at": [ts] * 5,
        "updated_at": [ts] * 5,
    })
    # Per-dimension scalar columns
    delta_col = geo_table["delta"]
    list_size = 2
    fixed_type = pa.list_(pa.float32(), list_size)
    fixed_delta = delta_col.cast(fixed_type)
    geo_table = geo_table.set_column(
        geo_table.schema.get_field_index("delta"), "delta", fixed_delta,
    )
    flat = fixed_delta.combine_chunks().values.to_numpy(zero_copy_only=False)
    matrix = flat.reshape(-1, list_size)
    for dim_idx in range(list_size):
        geo_table = geo_table.append_column(
            f"delta_dim_{dim_idx}",
            pa.array(matrix[:, dim_idx], type=pa.float32()),
        )
    geo_path = sphere_dir / "geometry" / "account_pattern" / "v=1" / "data.lance"
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(geo_table, str(geo_path), mode="overwrite")

    # Geometry stats
    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(base_path=str(sphere_dir))
    writer.write_geometry_stats(
        "account_pattern",
        version=1,
        delta_norms=np.array(delta_norms, dtype=np.float64),
        theta_norm=theta_norm,
    )

    # Edge table for tx_pattern: A↔B, A↔C, A↔D, A↔E, B↔C
    edge_table = pa.table({
        "from_key": ["A", "A", "A", "A", "B"],
        "to_key": ["B", "C", "D", "E", "C"],
        "event_key": ["tx1", "tx2", "tx3", "tx4", "tx5"],
        "timestamp": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64()),
        "amount": pa.array([100.0, 200.0, 300.0, 400.0, 50.0], type=pa.float64()),
    })
    edge_path = sphere_dir / "edges" / "tx_pattern" / "data.lance"
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(edge_table, str(edge_path), mode="overwrite")

    # Points for accounts (minimal)
    pts_table = pa.table({
        "primary_key": ["A", "B", "C", "D", "E"],
        "version": [1, 1, 1, 1, 1],
        "status": ["active"] * 5,
        "created_at": [ts] * 5,
        "changed_at": [ts] * 5,
    })
    pts_path = sphere_dir / "points" / "accounts" / "v=1" / "data.lance"
    pts_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(pts_table, str(pts_path), mode="overwrite")

    return sphere_dir


@pytest.fixture(scope="module")
def heredity_session(tmp_path_factory):
    """Session with a sphere that has edge tables for heredity testing."""
    tmp = tmp_path_factory.mktemp("heredity")
    sphere_dir = _build_edge_sphere(tmp)

    from hypertopos.sphere import HyperSphere

    hs = HyperSphere.open(str(sphere_dir))
    session = hs.session("test-heredity")
    yield session


class TestFindNovelEntities:
    def test_returns_ranked_list(self, heredity_session) -> None:
        nav = heredity_session.navigator()
        result = nav.find_novel_entities("tx_pattern", top_n=5)
        assert len(result) <= 5
        assert all("novelty_score" in r for r in result)
        assert all("primary_key" in r for r in result)
        assert all("n_neighbors" in r for r in result)
        scores = [r["novelty_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self, heredity_session) -> None:
        nav = heredity_session.navigator()
        result = nav.find_novel_entities("tx_pattern", top_n=2)
        assert len(result) <= 2

    def test_raises_without_edge_table(self, heredity_session) -> None:
        nav = heredity_session.navigator()
        from hypertopos.navigation.navigator import GDSNavigationError

        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.find_novel_entities("account_pattern")

    def test_novelty_scores_are_positive(self, heredity_session) -> None:
        nav = heredity_session.navigator()
        result = nav.find_novel_entities("tx_pattern", top_n=10)
        for r in result:
            assert r["novelty_score"] >= 0.0
            assert r["n_neighbors"] > 0
