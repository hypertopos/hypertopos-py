"""Per-dim correctness tests for engine/edge_features.py."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from hypertopos.engine import edge_features as ef


def _edges(rows: list[tuple[str, str, str, float, float]]) -> pa.Table:
    return pa.table({
        "from_key":  [r[0] for r in rows],
        "to_key":    [r[1] for r in rows],
        "event_key": [r[2] for r in rows],
        "timestamp": [r[3] for r in rows],
        "amount":    [r[4] for r in rows],
    })


def test_edge_dim_kinds_table():
    assert ef.EDGE_DIM_KINDS["pair_edge_count"]              == "poisson"
    assert ef.EDGE_DIM_KINDS["position_in_chain"]            == "poisson"
    assert ef.EDGE_DIM_KINDS["time_since_pair_last_edge"]    == "gaussian"
    assert ef.EDGE_DIM_KINDS["pair_amount_zscore"]           == "gaussian"
    assert ef.EDGE_DIM_KINDS["find_motif_structuring"]       == "bernoulli"


# ─── pair_edge_count ───────────────────────────────────────────────────

def test_pair_edge_count_basic():
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("A", "B", "ek2", 1.0, 100.0),
        ("A", "B", "ek3", 2.0, 100.0),
        ("X", "Y", "ek4", 0.0, 100.0),
    ])
    arr = ef.compute_pair_edge_count(edges)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([3, 3, 3, 1], dtype=np.float32),
    )


def test_pair_edge_count_directional():
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("B", "A", "ek2", 1.0, 100.0),
    ])
    arr = ef.compute_pair_edge_count(edges)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([1, 1], dtype=np.float32),
    )


# ─── time_since_pair_last_edge ─────────────────────────────────────────

def test_time_since_pair_last_edge_first_edge_dormant():
    edges = _edges([
        ("A", "B", "ek1", 100.0, 50.0),
        ("A", "B", "ek2", 200.0, 50.0),
        ("A", "B", "ek3", 250.0, 50.0),
    ])
    arr = ef.compute_time_since_pair_last_edge(
        edges, burst_seconds=60.0, dormant_seconds=999.0,
    )
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([999.0, 100.0, 50.0], dtype=np.float32),
    )


def test_time_since_pair_last_edge_per_pair():
    edges = _edges([
        ("A", "B", "ek1", 0.0,    50.0),
        ("X", "Y", "ek2", 100.0,  50.0),
        ("A", "B", "ek3", 200.0,  50.0),
        ("X", "Y", "ek4", 250.0,  50.0),
    ])
    arr = ef.compute_time_since_pair_last_edge(
        edges, burst_seconds=60.0, dormant_seconds=999.0,
    )
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([999.0, 999.0, 200.0, 150.0], dtype=np.float32),
    )


# ─── pair_amount_zscore ────────────────────────────────────────────────

def test_pair_amount_zscore_low_var_pair():
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("A", "B", "ek2", 1.0, 102.0),
        ("A", "B", "ek3", 2.0,  98.0),
    ])
    arr = ef.compute_pair_amount_zscore(edges, cv_threshold=0.05, min_count=3)
    vals = arr.to_numpy()
    assert abs(vals[0]) < 0.1
    assert vals[1] > 0.5
    assert vals[2] < -0.5


def test_pair_amount_zscore_high_var_returns_zero():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  100.0),
        ("A", "B", "ek2", 1.0, 1000.0),
        ("A", "B", "ek3", 2.0,    1.0),
    ])
    arr = ef.compute_pair_amount_zscore(edges, cv_threshold=0.05, min_count=3)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.zeros(3, dtype=np.float32),
    )


def test_pair_amount_zscore_below_min_count():
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("A", "B", "ek2", 1.0, 102.0),
    ])
    arr = ef.compute_pair_amount_zscore(edges, cv_threshold=0.05, min_count=3)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.zeros(2, dtype=np.float32),
    )


# ─── position_in_chain ─────────────────────────────────────────────────

def test_position_in_chain_simple_chain():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  100.0),
        ("B", "C", "ek2", 10.0, 100.0),
        ("C", "D", "ek3", 20.0, 100.0),
    ])
    arr = ef.compute_position_in_chain(edges, min_position=3)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([0.0, 0.0, 3.0], dtype=np.float32),
    )


def test_position_in_chain_threshold_5():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  100.0),
        ("B", "C", "ek2", 10.0, 100.0),
        ("C", "D", "ek3", 20.0, 100.0),
        ("D", "E", "ek4", 30.0, 100.0),
        ("E", "F", "ek5", 40.0, 100.0),
    ])
    arr = ef.compute_position_in_chain(edges, min_position=5)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([0.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float32),
    )


def test_position_in_chain_independent_chains():
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("X", "Y", "ek2", 0.0, 100.0),
    ])
    arr = ef.compute_position_in_chain(edges, min_position=3)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([0.0, 0.0], dtype=np.float32),
    )


# ─── find_motif_structuring ────────────────────────────────────────────

def test_find_motif_structuring_flag_match():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  20000.0),
        ("B", "C", "ek2", 10.0, 5000.0),
        ("C", "D", "ek3", 20.0, 5000.0),
    ])
    arr = ef.compute_find_motif_structuring(
        edges, time_window_hours=1.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_find_motif_structuring_no_match():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  5000.0),
        ("B", "C", "ek2", 10.0, 1000.0),
        ("C", "D", "ek3", 20.0, 1000.0),
    ])
    arr = ef.compute_find_motif_structuring(
        edges, time_window_hours=1.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    np.testing.assert_array_equal(
        arr.to_numpy(), np.zeros(3, dtype=np.float32),
    )


# ─── orchestrator ──────────────────────────────────────────────────────

def test_compute_all_edge_dims_full_config():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  20000.0),
        ("B", "C", "ek2", 10.0, 5000.0),
        ("C", "D", "ek3", 20.0, 5000.0),
    ])
    config = {
        "pair_edge_count":           {},
        "position_in_chain":         {"min_position": 3},
        "time_since_pair_last_edge": {"burst_seconds": 60.0,
                                       "dormant_seconds": 999.0},
        "pair_amount_zscore":        {"cv_threshold": 0.05, "min_count": 3},
        "find_motif_structuring":    {"time_window_hours": 1.0,
                                       "amt1_min": 10000.0, "amt2_max": 10000.0},
        "edge_curvature_frc":        {},
    }
    out = ef.compute_all_edge_dims(edges, config)
    assert out.num_rows == 3
    assert set(out.column_names) == {
        "event_key", "pair_edge_count", "position_in_chain",
        "time_since_pair_last_edge", "pair_amount_zscore",
        "find_motif_structuring", "edge_curvature_frc",
    }
    assert sorted(out["event_key"].to_pylist()) == ["ek1", "ek2", "ek3"]


def test_compute_all_edge_dims_subset_config():
    edges = _edges([("A", "B", "ek1", 0.0, 100.0)])
    out = ef.compute_all_edge_dims(edges, config={"pair_edge_count": {}})
    assert set(out.column_names) == {"event_key", "pair_edge_count"}


def test_compute_all_edge_dims_unknown_dim_raises():
    edges = _edges([("A", "B", "ek1", 0.0, 100.0)])
    import pytest
    with pytest.raises(ValueError, match="unknown edge dimension"):
        ef.compute_all_edge_dims(edges, config={"nonsense": {}})


# ─── edge_curvature_frc ────────────────────────────────────────────────

def test_edge_dim_kinds_includes_frc():
    """FRC is real-valued (can be negative on sparse local neighbourhoods)
    → gaussian for Bregman."""
    assert ef.EDGE_DIM_KINDS["edge_curvature_frc"] == "gaussian"


def test_frc_triangle_each_edge_curvature_one():
    """Triangle A-B-C-A: every edge sits on 1 triangle, every endpoint
    degree 2 → F = 4 − 2 − 2 + 1 = 1."""
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("B", "C", "ek2", 1.0, 100.0),
        ("C", "A", "ek3", 2.0, 100.0),
    ])
    arr = ef.compute_edge_curvature_frc(edges)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_frc_path_no_triangles():
    """Path A-B, B-C: no triangles. A has degree 1, B has degree 2, C has degree 1.
    F(A,B) = 4 − 1 − 2 + 0 = 1; F(B,C) = 4 − 2 − 1 + 0 = 1."""
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("B", "C", "ek2", 1.0, 100.0),
    ])
    arr = ef.compute_edge_curvature_frc(edges)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([1.0, 1.0], dtype=np.float32),
    )


def test_frc_star_negative_curvature():
    """Star: H-L1, H-L2, H-L3, H-L4. Hub degree 4, each leaf degree 1, no triangles.
    F = 4 − 4 − 1 + 0 = −1 per edge."""
    edges = _edges([
        ("H", "L1", "ek1", 0.0, 100.0),
        ("H", "L2", "ek2", 1.0, 100.0),
        ("H", "L3", "ek3", 2.0, 100.0),
        ("H", "L4", "ek4", 3.0, 100.0),
    ])
    arr = ef.compute_edge_curvature_frc(edges)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
    )


def test_frc_k4_zero_curvature():
    """K4 (4 nodes, 6 edges, fully connected): every edge sits on 2 triangles,
    every degree 3 → F = 4 − 3 − 3 + 2 = 0."""
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("A", "C", "ek2", 1.0, 100.0),
        ("A", "D", "ek3", 2.0, 100.0),
        ("B", "C", "ek4", 3.0, 100.0),
        ("B", "D", "ek5", 4.0, 100.0),
        ("C", "D", "ek6", 5.0, 100.0),
    ])
    arr = ef.compute_edge_curvature_frc(edges)
    np.testing.assert_array_equal(
        arr.to_numpy(), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_frc_direction_canonicalised_undirected():
    """`(A, B)` and `(B, A)` events between the same accounts share the same
    FRC value because the underlying graph is undirected."""
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("B", "A", "ek2", 1.0, 100.0),  # reversed direction, same pair
        ("B", "C", "ek3", 2.0, 100.0),
        ("C", "A", "ek4", 3.0, 100.0),
    ])
    arr = ef.compute_edge_curvature_frc(edges).to_numpy()
    # Underlying undirected graph is the triangle A-B-C → every edge FRC = 1.
    # The (A,B) and (B,A) events both belong to the same undirected edge.
    np.testing.assert_array_equal(
        arr, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_frc_multigraph_collapse_no_count_inflation():
    """Duplicate `(A, B)` events must NOT inflate the triangle count.
    Underlying simple graph is the triangle A-B-C → FRC = 1 for every event."""
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("A", "B", "ek2", 1.0, 100.0),  # duplicate
        ("A", "B", "ek3", 2.0, 100.0),  # duplicate
        ("B", "C", "ek4", 3.0, 100.0),
        ("C", "A", "ek5", 4.0, 100.0),
    ])
    arr = ef.compute_edge_curvature_frc(edges).to_numpy()
    np.testing.assert_array_equal(
        arr, np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_frc_single_mega_hub_pathological():
    """Single-hub topology: 1 hub + 200 leaves, no triangles. Exercises the
    pathological case where A @ A could blow up without the .multiply(A) mask
    — the hub has degree 200 so its row in A @ A would have 200 non-zero
    entries each pointing back to itself; the mask restricts the result to
    actual edges. FRC = 4 − 200 − 1 + 0 = −197 per edge."""
    edges = _edges([
        ("H", f"L{i}", f"ek{i}", float(i), 100.0) for i in range(200)
    ])
    arr = ef.compute_edge_curvature_frc(edges).to_numpy()
    np.testing.assert_array_equal(
        arr, np.full(200, -197.0, dtype=np.float32),
    )


def test_frc_empty_table_returns_empty():
    edges = _edges([])
    arr = ef.compute_edge_curvature_frc(edges)
    assert arr.type == pa.float32()
    assert len(arr) == 0


def test_compute_all_edge_dims_dispatches_frc():
    edges = _edges([
        ("A", "B", "ek1", 0.0, 100.0),
        ("B", "C", "ek2", 1.0, 100.0),
        ("C", "A", "ek3", 2.0, 100.0),
    ])
    result = ef.compute_all_edge_dims(edges, config={"edge_curvature_frc": {}})
    assert "edge_curvature_frc" in result.column_names
    np.testing.assert_array_equal(
        result["edge_curvature_frc"].to_numpy(),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
