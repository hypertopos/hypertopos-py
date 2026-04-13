# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for compute_graph_features_temporal — single-pass graph features."""
import numpy as np
import pyarrow as pa

from hypertopos.builder.derived import (
    compute_graph_features,
    compute_graph_features_temporal,
)


def _make_event_table(n_events: int, n_anchors: int, n_buckets: int):
    rng = np.random.default_rng(42)
    from_keys = rng.choice([f"A{i}" for i in range(n_anchors)], size=n_events)
    to_keys = rng.choice([f"A{i}" for i in range(n_anchors)], size=n_events)
    buckets = rng.integers(0, n_buckets, size=n_events)
    table = pa.table({
        "from_key": pa.array(from_keys),
        "to_key": pa.array(to_keys),
    })
    return table, buckets


def test_temporal_graph_features_match_per_window():
    n_events, n_anchors, n_buckets = 500, 10, 5
    table, buckets = _make_event_table(n_events, n_anchors, n_buckets)
    anchor_keys = pa.array([f"A{i}" for i in range(n_anchors)])
    features = ["in_degree", "out_degree"]

    expected = np.zeros((n_anchors, n_buckets, len(features)), dtype=np.float32)
    for b in range(n_buckets):
        mask = buckets == b
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        filtered = table.take(pa.array(indices, type=pa.int64()))
        results = compute_graph_features(
            filtered, anchor_keys, "from_key", "to_key", features,
        )
        for f_idx, feat in enumerate(features):
            if feat in results:
                values, _ = results[feat]
                expected[:, b, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, n_buckets,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_graph_features_with_nulls():
    """Null from/to values should be filtered the same way."""
    table = pa.table({
        "from_key": pa.array(["A0", None, "A1", "A0", None]),
        "to_key": pa.array(["A1", "A0", None, "A1", None]),
    })
    anchor_keys = pa.array(["A0", "A1"])
    buckets = np.array([0, 0, 0, 1, 1])
    features = ["in_degree", "out_degree"]

    # Per-window reference
    expected = np.zeros((2, 2, 2), dtype=np.float32)
    for b in range(2):
        mask = buckets == b
        indices = np.where(mask)[0]
        filtered = table.take(pa.array(indices, type=pa.int64()))
        results = compute_graph_features(
            filtered, anchor_keys, "from_key", "to_key", features,
        )
        for f_idx, feat in enumerate(features):
            if feat in results:
                values, _ = results[feat]
                expected[:, b, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, 2,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_graph_features_empty_bucket():
    """Buckets with no events should remain zero."""
    table = pa.table({
        "from_key": pa.array(["A0", "A1"]),
        "to_key": pa.array(["A1", "A0"]),
    })
    anchor_keys = pa.array(["A0", "A1"])
    # All events in bucket 0, bucket 1 is empty
    buckets = np.array([0, 0])
    features = ["in_degree", "out_degree"]

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, 2,
    )
    # Bucket 1 should be all zeros
    np.testing.assert_array_equal(actual[:, 1, :], 0.0)
    # Bucket 0 should have non-zero values
    assert actual[:, 0, :].sum() > 0


def test_temporal_graph_features_reciprocity():
    """Reciprocity via single-pass matches per-window."""
    n_events, n_anchors, n_buckets = 200, 5, 3
    table, buckets = _make_event_table(n_events, n_anchors, n_buckets)
    anchor_keys = pa.array([f"A{i}" for i in range(n_anchors)])
    features = ["reciprocity"]

    expected = np.zeros((n_anchors, n_buckets, 1), dtype=np.float32)
    for b in range(n_buckets):
        mask = buckets == b
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        filtered = table.take(pa.array(indices, type=pa.int64()))
        results = compute_graph_features(
            filtered, anchor_keys, "from_key", "to_key", features,
        )
        for f_idx, feat in enumerate(features):
            if feat in results:
                values, _ = results[feat]
                expected[:, b, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, n_buckets,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_graph_features_counterpart_overlap():
    """counterpart_overlap via single-pass matches per-window."""
    n_events, n_anchors, n_buckets = 300, 5, 3
    table, buckets = _make_event_table(n_events, n_anchors, n_buckets)
    anchor_keys = pa.array([f"A{i}" for i in range(n_anchors)])
    features = ["counterpart_overlap"]

    expected = np.zeros((n_anchors, n_buckets, 1), dtype=np.float32)
    for b in range(n_buckets):
        mask = buckets == b
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        filtered = table.take(pa.array(indices, type=pa.int64()))
        results = compute_graph_features(
            filtered, anchor_keys, "from_key", "to_key", features,
        )
        for f_idx, feat in enumerate(features):
            if feat in results:
                values, _ = results[feat]
                expected[:, b, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, n_buckets,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_graph_features_all_four():
    """All 4 graph features via single-pass match per-window."""
    n_events, n_anchors, n_buckets = 500, 8, 4
    table, buckets = _make_event_table(n_events, n_anchors, n_buckets)
    anchor_keys = pa.array([f"A{i}" for i in range(n_anchors)])
    features = ["in_degree", "out_degree", "reciprocity", "counterpart_overlap"]

    expected = np.zeros((n_anchors, n_buckets, 4), dtype=np.float32)
    for b in range(n_buckets):
        mask = buckets == b
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        filtered = table.take(pa.array(indices, type=pa.int64()))
        results = compute_graph_features(
            filtered, anchor_keys, "from_key", "to_key", features,
        )
        for f_idx, feat in enumerate(features):
            if feat in results:
                values, _ = results[feat]
                expected[:, b, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, n_buckets,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_graph_features_non_anchor_entities():
    """Edges to/from non-anchor entities must be counted correctly.

    Non-anchor counterparties contribute to degree/overlap counts
    but never appear as result rows.
    """
    # Anchors: A0, A1. Non-anchor: X0, X1 (appear in edges only)
    table = pa.table({
        "from_key": pa.array(["A0", "A0", "X0", "A1", "X1"]),
        "to_key":   pa.array(["X0", "A1", "A0", "A0", "A1"]),
    })
    anchor_keys = pa.array(["A0", "A1"])
    buckets = np.array([0, 0, 0, 0, 0])
    features = ["reciprocity", "counterpart_overlap"]

    expected = np.zeros((2, 1, 2), dtype=np.float32)
    filtered = table  # single bucket
    results = compute_graph_features(
        filtered, anchor_keys, "from_key", "to_key", features,
    )
    for f_idx, feat in enumerate(features):
        if feat in results:
            values, _ = results[feat]
            expected[:, 0, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, 1,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_graph_features_single_entity_self_loop():
    """Self-loops (A0→A0) should be handled without errors."""
    table = pa.table({
        "from_key": pa.array(["A0", "A0", "A0"]),
        "to_key":   pa.array(["A0", "A1", "A0"]),
    })
    anchor_keys = pa.array(["A0", "A1"])
    buckets = np.array([0, 0, 1])
    features = ["reciprocity", "counterpart_overlap"]

    expected = np.zeros((2, 2, 2), dtype=np.float32)
    for b in range(2):
        mask = buckets == b
        indices = np.where(mask)[0]
        filtered = table.take(pa.array(indices, type=pa.int64()))
        results = compute_graph_features(
            filtered, anchor_keys, "from_key", "to_key", features,
        )
        for f_idx, feat in enumerate(features):
            if feat in results:
                values, _ = results[feat]
                expected[:, b, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, 2,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_graph_features_many_buckets():
    """Stress test with many buckets to verify boundary slicing."""
    n_events, n_anchors, n_buckets = 2000, 10, 50
    table, buckets = _make_event_table(n_events, n_anchors, n_buckets)
    anchor_keys = pa.array([f"A{i}" for i in range(n_anchors)])
    features = ["in_degree", "out_degree", "reciprocity", "counterpart_overlap"]

    expected = np.zeros((n_anchors, n_buckets, 4), dtype=np.float32)
    for b in range(n_buckets):
        mask = buckets == b
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        filtered = table.take(pa.array(indices, type=pa.int64()))
        results = compute_graph_features(
            filtered, anchor_keys, "from_key", "to_key", features,
        )
        for f_idx, feat in enumerate(features):
            if feat in results:
                values, _ = results[feat]
                expected[:, b, f_idx] = values

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, n_buckets,
    )
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_temporal_overlap_no_bidirectional_edges():
    """When all edges are unidirectional, overlap should be 0."""
    # A0→A1, A1→A2, A2→A0 — no bidirectional pairs
    table = pa.table({
        "from_key": pa.array(["A0", "A1", "A2"]),
        "to_key":   pa.array(["A1", "A2", "A0"]),
    })
    anchor_keys = pa.array(["A0", "A1", "A2"])
    buckets = np.array([0, 0, 0])
    features = ["counterpart_overlap"]

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, 1,
    )
    np.testing.assert_array_equal(actual[:, 0, 0], 0.0)


def test_temporal_overlap_full_bidirectional():
    """When all edges are bidirectional, overlap should be 1.0."""
    # A0↔A1 — full bidirectionality
    table = pa.table({
        "from_key": pa.array(["A0", "A1"]),
        "to_key":   pa.array(["A1", "A0"]),
    })
    anchor_keys = pa.array(["A0", "A1"])
    buckets = np.array([0, 0])
    features = ["counterpart_overlap"]

    actual = compute_graph_features_temporal(
        table, anchor_keys, "from_key", "to_key", features, buckets, 1,
    )
    # A0: out={A1}, in={A1}, bidir={A1} → 1/(1+1-1)=1.0
    # A1: out={A0}, in={A0}, bidir={A0} → 1/(1+1-1)=1.0
    np.testing.assert_array_almost_equal(actual[:, 0, 0], [1.0, 1.0])
