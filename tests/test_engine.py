# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from datetime import UTC, datetime, timezone

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.engine.geometry import GDSEngine
from hypertopos.model.manifest import Manifest
from hypertopos.model.objects import Edge, Polygon, Solid
from hypertopos.model.sphere import (
    Alias,
    AliasFilter,
    CuttingPlane,
    DerivedPattern,
    Pattern,
    RelationDef,
)
from hypertopos.storage.cache import GDSCache
from hypertopos.storage.reader import GDSReader


def _make_pattern() -> Pattern:
    return Pattern(
        pattern_id="customer_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.array([0.7, 0.4], dtype=np.float32),
        sigma_diag=np.array([0.2, 0.3], dtype=np.float32),
        theta=np.array([2.5, 1.0], dtype=np.float32),  # z-scored: [0.5/0.2, 0.3/0.3]
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )


def _make_polygon(alive_lines: list[str]) -> Polygon:
    edges = [
        Edge(line_id=line_id, point_key=f"P-{i}", status="alive", direction="out")
        for i, line_id in enumerate(alive_lines)
    ]
    return Polygon(
        primary_key="CUST-001",
        pattern_id="customer_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(2),
        delta_norm=0.0,
        is_anomaly=False,
        edges=edges,
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_compute_delta_all_alive():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    polygon = _make_polygon(["products", "stores"])
    delta = engine.compute_delta(polygon, pattern)
    # shape_vector = [1.0, 1.0], mu = [0.7, 0.4], sigma = [0.2, 0.3]
    # delta_z = (shape - mu) / sigma = [0.3/0.2, 0.6/0.3] = [1.5, 2.0]
    expected = np.array([(1.0 - 0.7) / 0.2, (1.0 - 0.4) / 0.3], dtype=np.float32)
    np.testing.assert_allclose(delta, expected, rtol=1e-5)


def test_compute_delta_partial():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    polygon = _make_polygon(["products"])  # stores missing
    delta = engine.compute_delta(polygon, pattern)
    # shape = [1.0, 0.0], mu = [0.7, 0.4], sigma = [0.2, 0.3]
    expected = np.array([(1.0 - 0.7) / 0.2, (0.0 - 0.4) / 0.3], dtype=np.float32)
    np.testing.assert_allclose(delta, expected, rtol=1e-5)


def test_compute_delta_anomaly():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    polygon = _make_polygon([])  # no alive edges -> far from mean
    delta = engine.compute_delta(polygon, pattern)
    norm = float(np.linalg.norm(delta))
    theta_norm = float(np.linalg.norm(pattern.theta))
    assert norm > theta_norm  # this polygon is anomalous


def _make_manifest() -> Manifest:
    return Manifest(
        manifest_id="m-001",
        agent_id="agent-001",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "products": 1},
        pattern_versions={"customer_pattern": 1},
    )


def test_build_polygon(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = _make_manifest()
    polygon = engine.build_polygon("CUST-001", "customer_pattern", manifest)
    assert polygon.primary_key == "CUST-001"
    assert len(polygon.edges) > 0
    assert polygon.delta is not None
    assert polygon.delta.shape == (2,)


def test_build_polygon_uses_cache(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = _make_manifest()
    p1 = engine.build_polygon("CUST-001", "customer_pattern", manifest)
    p2 = engine.build_polygon("CUST-001", "customer_pattern", manifest)
    assert p1 is p2


def test_build_solid(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = _make_manifest()
    solid = engine.build_solid("CUST-001", "customer_pattern", manifest, filters={"year": ["2024"]})
    assert isinstance(solid, Solid)
    assert solid.primary_key == "CUST-001"
    assert len(solid.slices) > 0


def test_build_solid_recomputes_delta():
    """build_solid recomputes delta_snapshot from shape_snapshot + pattern mu/sigma."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Pattern, RelationDef, Sphere

    mu = np.array([0.5, 0.4], dtype=np.float32)
    sigma_diag = np.array([0.2, 0.3], dtype=np.float32)
    shape = np.array([0.7, 0.1], dtype=np.float32)
    now = datetime(2024, 6, 1, tzinfo=timezone.utc)  # noqa: UP017

    temporal_table = pa.table(
        {
            "slice_index": pa.array([0], type=pa.int64()),
            "timestamp": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "deformation_type": pa.array(["structural"]),
            "shape_snapshot": pa.array([shape.tolist()]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "changed_property": pa.array([None], type=pa.null()),
            "changed_line_id": pa.array([None], type=pa.null()),
        }
    )

    edge_type = pa.list_(
        pa.struct(
            [
                pa.field("line_id", pa.string()),
                pa.field("point_key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("direction", pa.string()),
            ]
        )
    )
    geometry_table = pa.table(
        {
            "primary_key": pa.array(["CUST-X"]),
            "pattern_id": pa.array(["customer_pattern"]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "pattern_type": pa.array(["anchor"]),
            "scale": pa.array([1], type=pa.int64()),
            "delta": pa.array([[0.5, 0.5]]),
            "delta_norm": pa.array([0.7], type=pa.float64()),
            "is_anomaly": pa.array([False]),
            "edges": pa.array([[]], type=edge_type),
            "version": pa.array([1], type=pa.int64()),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        }
    )

    pattern = Pattern(
        pattern_id="customer_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=mu,
        sigma_diag=sigma_diag,
        theta=np.array([3.0, 3.0], dtype=np.float32),
        population_size=50,
        computed_at=now,
        version=1,
        status="production",
    )
    sphere = Sphere(  # noqa: E501
        sphere_id="test",
        name="test",
        base_path=".",
        patterns={"customer_pattern": pattern},
    )

    mock_storage = MagicMock()
    mock_storage.read_temporal.return_value = temporal_table
    mock_storage.read_geometry.return_value = geometry_table
    mock_storage.read_sphere.return_value = sphere

    engine = GDSEngine(storage=mock_storage, cache=GDSCache())
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="test-agent",
        snapshot_time=now,
        status="active",
        line_versions={},
        pattern_versions={"customer_pattern": 1},
    )

    solid = engine.build_solid("CUST-X", "customer_pattern", manifest)

    assert len(solid.slices) == 1
    sigma_eff = np.maximum(sigma_diag, 1e-2)
    expected_delta = (shape - mu) / sigma_eff
    np.testing.assert_allclose(solid.slices[0].delta_snapshot, expected_delta, rtol=1e-5)


def test_build_solid_continuous_mode_base_polygon_uses_stored_delta():
    """build_solid on continuous-mode pattern must use stored delta, not recompute from alive_count.

    In continuous mode (edge_max set), _polygon_to_shape_vector yields alive_count=1 per relation
    for every entity (single edge with point_key=""). This would make all base_polygons identical.
    The fix: skip compute_delta for continuous-mode; use stored geometry delta as ground truth.
    """
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Pattern, RelationDef, Sphere

    now = datetime(2024, 6, 1, tzinfo=timezone.utc)  # noqa: UP017
    mu = np.array([0.5, 0.4], dtype=np.float32)
    sigma_diag = np.array([0.2, 0.3], dtype=np.float32)
    edge_max = np.array([5.0, 3.0], dtype=np.float32)
    stored_delta = np.array([2.5, 1.8], dtype=np.float32)

    edge_type = pa.list_(
        pa.struct(
            [
                pa.field("line_id", pa.string()),
                pa.field("point_key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("direction", pa.string()),
            ]
        )
    )
    geometry_table = pa.table(
        {
            "primary_key": pa.array(["CUST-X"]),
            "pattern_id": pa.array(["cust_pattern"]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "pattern_type": pa.array(["anchor"]),
            "scale": pa.array([1], type=pa.int64()),
            "delta": pa.array([stored_delta.tolist()]),
            "delta_norm": pa.array([float(np.linalg.norm(stored_delta))], type=pa.float64()),
            "is_anomaly": pa.array([True]),
            "edges": pa.array(
                [
                    [
                        {
                            "line_id": "products",
                            "point_key": "",
                            "status": "alive",
                            "direction": "out",
                        },
                        {
                            "line_id": "stores",
                            "point_key": "",
                            "status": "alive",
                            "direction": "in",
                        },
                    ]
                ],
                type=edge_type,
            ),
            "version": pa.array([1], type=pa.int64()),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        }
    )
    temporal_table = pa.table(
        {
            "slice_index": pa.array([], type=pa.int64()),
            "timestamp": pa.array([], type=pa.timestamp("us", tz="UTC")),
            "deformation_type": pa.array([], type=pa.string()),
            "shape_snapshot": pa.array([], type=pa.list_(pa.float32())),
            "pattern_ver": pa.array([], type=pa.int64()),
            "changed_property": pa.array([], type=pa.null()),
            "changed_line_id": pa.array([], type=pa.null()),
        }
    )

    pattern = Pattern(
        pattern_id="cust_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=mu,
        sigma_diag=sigma_diag,
        theta=np.array([3.0, 3.0], dtype=np.float32),
        population_size=50,
        computed_at=now,
        version=1,
        status="production",
        edge_max=edge_max,
    )
    sphere = Sphere(
        sphere_id="test",
        name="test",
        base_path=".",
        patterns={"cust_pattern": pattern},
    )
    mock_storage = MagicMock()
    mock_storage.read_geometry.return_value = geometry_table
    mock_storage.read_temporal.return_value = temporal_table
    mock_storage.read_sphere.return_value = sphere

    engine = GDSEngine(storage=mock_storage, cache=GDSCache())
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="test-agent",
        snapshot_time=now,
        status="active",
        line_versions={},
        pattern_versions={"cust_pattern": 1},
    )

    solid = engine.build_solid("CUST-X", "cust_pattern", manifest)

    # Stored delta must be preserved — recomputation would yield ~[-1.5, -0.22] for alive_count=1
    np.testing.assert_allclose(solid.base_polygon.delta, stored_delta, rtol=1e-5)
    assert solid.base_polygon.is_anomaly is True  # stored value preserved


def test_build_solid_sorts_by_timestamp_not_slice_index():
    """build_solid must return slices in chronological order even when slice_index
    is assigned out of timestamp order (e.g. re-ingestion, late corrections)."""
    from unittest.mock import MagicMock

    import pyarrow as pa
    from hypertopos.model.sphere import Pattern, RelationDef, Sphere

    # Arrange: two slices where slice_index=0 has a LATER timestamp than slice_index=1
    ts_early = datetime(2022, 7, 1, tzinfo=UTC)
    ts_late = datetime(2023, 6, 1, tzinfo=UTC)
    now = datetime(2024, 1, 1, tzinfo=UTC)

    temporal_table = pa.table(
        {
            "slice_index": pa.array([0, 1], type=pa.int64()),
            "timestamp": pa.array([ts_late, ts_early], type=pa.timestamp("us", tz="UTC")),
            "deformation_type": pa.array(["internal", "internal"]),
            "shape_snapshot": pa.array([[0.1, 0.2], [0.3, 0.4]]),
            "pattern_ver": pa.array([1, 1], type=pa.int64()),
            "changed_property": pa.array([None, None], type=pa.null()),
            "changed_line_id": pa.array([None, None], type=pa.null()),
        }
    )

    geometry_table = pa.table(
        {
            "primary_key": pa.array(["CUST-X"]),
            "pattern_id": pa.array(["customer_pattern"]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "pattern_type": pa.array(["anchor"]),
            "scale": pa.array([1], type=pa.int64()),
            "delta": pa.array([[0.5, 0.5]]),
            "delta_norm": pa.array([0.7], type=pa.float64()),
            "is_anomaly": pa.array([False]),
            "edges": pa.array(
                [[]],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
            "version": pa.array([1], type=pa.int64()),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        }
    )

    pattern = Pattern(
        pattern_id="customer_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="in", required=False),
            RelationDef(line_id="products", direction="out", required=False),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.ones(2, dtype=np.float32) * 2.0,
        population_size=100,
        computed_at=now,
        version=1,
        status="production",
    )
    sphere = Sphere(
        sphere_id="test", name="test", base_path=".", patterns={"customer_pattern": pattern}
    )

    mock_storage = MagicMock()
    mock_storage.read_temporal.return_value = temporal_table
    mock_storage.read_geometry.return_value = geometry_table
    mock_storage.read_sphere.return_value = sphere

    engine = GDSEngine(storage=mock_storage, cache=GDSCache())
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="test-agent",
        snapshot_time=now,
        status="active",
        line_versions={},
        pattern_versions={"customer_pattern": 1},
    )

    # Act
    solid = engine.build_solid("CUST-X", "customer_pattern", manifest)

    # Assert: slices must be in ascending timestamp order
    timestamps = [s.timestamp for s in solid.slices]
    assert timestamps == sorted(timestamps), f"Expected chronological order, got: {timestamps}"
    assert solid.slices[0].timestamp == ts_early
    assert solid.slices[1].timestamp == ts_late


def test_build_solid_filters_slices_by_timestamp():
    """build_solid with timestamp parameter returns only slices at or before timestamp."""
    from unittest.mock import MagicMock

    import pyarrow as pa
    from hypertopos.model.sphere import Pattern, RelationDef, Sphere

    ts_early = datetime(2022, 1, 1, tzinfo=UTC)
    ts_mid = datetime(2023, 1, 1, tzinfo=UTC)
    ts_late = datetime(2024, 1, 1, tzinfo=UTC)
    now = datetime(2025, 1, 1, tzinfo=UTC)

    temporal_table = pa.table(
        {
            "slice_index": pa.array([0, 1, 2], type=pa.int64()),
            "timestamp": pa.array([ts_early, ts_mid, ts_late], type=pa.timestamp("us", tz="UTC")),
            "deformation_type": pa.array(["internal", "internal", "internal"]),
            "shape_snapshot": pa.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "pattern_ver": pa.array([1, 1, 1], type=pa.int64()),
            "changed_property": pa.array([None, None, None], type=pa.null()),
            "changed_line_id": pa.array([None, None, None], type=pa.null()),
        }
    )

    geometry_table = pa.table(
        {
            "primary_key": pa.array(["CUST-X"]),
            "pattern_id": pa.array(["customer_pattern"]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "pattern_type": pa.array(["anchor"]),
            "scale": pa.array([1], type=pa.int64()),
            "delta": pa.array([[0.5, 0.5]]),
            "delta_norm": pa.array([0.7], type=pa.float64()),
            "is_anomaly": pa.array([False]),
            "edges": pa.array(
                [[]],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
            "version": pa.array([1], type=pa.int64()),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        }
    )

    pattern = Pattern(
        pattern_id="customer_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="in", required=False),
            RelationDef(line_id="products", direction="out", required=False),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.ones(2, dtype=np.float32) * 2.0,
        population_size=100,
        computed_at=now,
        version=1,
        status="production",
    )
    sphere = Sphere(
        sphere_id="test", name="test", base_path=".", patterns={"customer_pattern": pattern}
    )

    mock_storage = MagicMock()
    mock_storage.read_temporal.return_value = temporal_table
    mock_storage.read_geometry.return_value = geometry_table
    mock_storage.read_sphere.return_value = sphere

    engine = GDSEngine(storage=mock_storage, cache=GDSCache())
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="test-agent",
        snapshot_time=now,
        status="active",
        line_versions={},
        pattern_versions={"customer_pattern": 1},
    )

    # no timestamp → full history
    solid = engine.build_solid("CUST-X", "customer_pattern", manifest)
    assert len(solid.slices) == 3

    # timestamp before first slice → empty slices
    before_all = datetime(2021, 6, 1, tzinfo=UTC)
    solid_empty = engine.build_solid("CUST-X", "customer_pattern", manifest, timestamp=before_all)
    assert len(solid_empty.slices) == 0

    # timestamp mid-history (= ts_mid) → two slices (ts_early and ts_mid included)
    solid_mid = engine.build_solid("CUST-X", "customer_pattern", manifest, timestamp=ts_mid)
    assert len(solid_mid.slices) == 2
    assert all(s.timestamp <= ts_mid for s in solid_mid.slices)


def test_compute_temporal_distance(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = _make_manifest()
    solid_a = engine.build_solid(
        "CUST-001", "customer_pattern", manifest, filters={"year": ["2024"]}
    )
    solid_b = engine.build_solid(
        "CUST-001", "customer_pattern", manifest, filters={"year": ["2024"]}
    )
    dist = engine.compute_distance_temporal(solid_a, solid_b)
    assert dist == 0.0


# --- _find_common_polygons tests ---


def _make_polygon_with_edges(edges: list[Edge]) -> Polygon:
    """Helper to build a Polygon with an explicit edge list."""
    return Polygon(
        primary_key="TEST-001",
        pattern_id="test_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(2),
        delta_norm=0.0,
        is_anomaly=False,
        edges=edges,
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_find_common_polygons_overlap():
    """Two polygons sharing some alive edges return the common (line_id, point_key) pairs."""
    engine = GDSEngine(storage=None, cache=None)

    poly_a = _make_polygon_with_edges(
        [
            Edge(line_id="products", point_key="P-1", status="alive", direction="out"),
            Edge(line_id="stores", point_key="P-2", status="alive", direction="in"),
            Edge(line_id="regions", point_key="P-3", status="alive", direction="out"),
        ]
    )
    poly_b = _make_polygon_with_edges(
        [
            Edge(line_id="stores", point_key="P-2", status="alive", direction="in"),
            Edge(line_id="regions", point_key="P-3", status="alive", direction="out"),
            Edge(line_id="orders", point_key="P-4", status="alive", direction="out"),
        ]
    )

    result = engine._find_common_polygons(poly_a, poly_b)

    assert result == {("stores", "P-2"), ("regions", "P-3")}


def test_find_common_polygons_no_overlap():
    """Two polygons with no shared alive edges return an empty set."""
    engine = GDSEngine(storage=None, cache=None)

    poly_a = _make_polygon_with_edges(
        [
            Edge(line_id="products", point_key="P-1", status="alive", direction="out"),
        ]
    )
    poly_b = _make_polygon_with_edges(
        [
            Edge(line_id="orders", point_key="P-9", status="alive", direction="out"),
        ]
    )

    result = engine._find_common_polygons(poly_a, poly_b)

    assert result == set()


def test_find_common_polygons_dead_edges_excluded():
    """Dead edges are never included even if both polygons share the same (line_id, point_key)."""
    engine = GDSEngine(storage=None, cache=None)

    poly_a = _make_polygon_with_edges(
        [
            Edge(line_id="products", point_key="P-1", status="alive", direction="out"),
            Edge(line_id="stores", point_key="P-2", status="dead", direction="in"),
        ]
    )
    poly_b = _make_polygon_with_edges(
        [
            Edge(line_id="products", point_key="P-1", status="alive", direction="out"),
            Edge(line_id="stores", point_key="P-2", status="alive", direction="in"),
        ]
    )

    result = engine._find_common_polygons(poly_a, poly_b)

    # ("stores", "P-2") must NOT appear because it is dead in poly_a
    assert result == {("products", "P-1")}


# --- Continuous shape vector (edge_max) ---


def _make_pattern_with_edge_max() -> Pattern:
    return Pattern(
        pattern_id="cp",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.array([0.5, 0.5], dtype=np.float32),
        sigma_diag=np.array([0.2, 0.3], dtype=np.float32),
        theta=np.array([2.5, 1.0], dtype=np.float32),  # z-scored: [0.5/0.2, 0.3/0.3]
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=np.array([5, 3], dtype=np.float32),
    )


def test_shape_vector_continuous_multiple_edges():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern_with_edge_max()
    edges = [
        Edge(line_id="products", point_key=f"P-{i}", status="alive", direction="out")
        for i in range(3)
    ] + [
        Edge(line_id="stores", point_key="S-1", status="alive", direction="in"),
    ]
    polygon = _make_polygon_with_edges(edges)
    vec = engine._polygon_to_shape_vector(polygon, pattern)
    np.testing.assert_allclose(vec, [3.0 / 5.0, 1.0 / 3.0], rtol=1e-5)


def test_shape_vector_continuous_no_edges():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern_with_edge_max()
    polygon = _make_polygon_with_edges([])
    vec = engine._polygon_to_shape_vector(polygon, pattern)
    np.testing.assert_allclose(vec, [0.0, 0.0])


def test_shape_vector_binary_when_no_edge_max():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()  # no edge_max
    edges = [
        Edge(line_id="products", point_key=f"P-{i}", status="alive", direction="out")
        for i in range(3)
    ]
    polygon = _make_polygon_with_edges(edges)
    vec = engine._polygon_to_shape_vector(polygon, pattern)
    np.testing.assert_allclose(vec, [1.0, 0.0])


# --- Anomaly classification ---


def test_classify_anomalies_two_clusters():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    ts = datetime(2024, 1, 1, tzinfo=UTC)

    def _anomaly(bk: str, delta: list[float]) -> Polygon:
        return Polygon(
            primary_key=bk,
            pattern_id="cp",
            pattern_ver=1,
            pattern_type="anchor",
            scale=1,
            delta=np.array(delta, dtype=np.float32),
            delta_norm=float(np.linalg.norm(delta)),
            is_anomaly=True,
            edges=[],
            last_refresh_at=ts,
            updated_at=ts,
        )

    polygons = [
        _anomaly("C-1", [-0.7, 0.2]),
        _anomaly("C-2", [-0.7, 0.2]),
        _anomaly("C-3", [-0.7, -0.8]),
    ]
    clusters = engine.classify_anomalies(polygons, pattern)
    assert len(clusters) == 2
    assert clusters[0]["count"] == 2
    assert clusters[1]["count"] == 1


def test_classify_anomalies_skips_non_anomalies():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    normal = Polygon(
        primary_key="C-N",
        pattern_id="cp",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.array([0.1, 0.1], dtype=np.float32),
        delta_norm=0.14,
        is_anomaly=False,
        edges=[],
        last_refresh_at=ts,
        updated_at=ts,
    )
    assert engine.classify_anomalies([normal], pattern) == []


def test_classify_anomalies_label_all_negative():
    """All-negative delta → 'missing: ...' label ordered by |delta|."""
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()  # relations: products, stores
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    p = Polygon(
        primary_key="C-1",
        pattern_id="cp",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.array([-2.0, -0.8], dtype=np.float32),
        delta_norm=2.15,
        is_anomaly=True,
        edges=[],
        last_refresh_at=ts,
        updated_at=ts,
    )
    clusters = engine.classify_anomalies([p], pattern)
    assert clusters[0]["label"] == "missing: products, stores"


def test_classify_anomalies_label_positive_top_driver():
    """Positive top driver → 'elevated: ...' appears first."""
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()  # relations: products, stores
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    # products=+5.0 (top driver, positive), stores=-0.8 (secondary, negative)
    p = Polygon(
        primary_key="C-1",
        pattern_id="cp",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.array([5.0, -0.8], dtype=np.float32),
        delta_norm=5.06,
        is_anomaly=True,
        edges=[],
        last_refresh_at=ts,
        updated_at=ts,
    )
    clusters = engine.classify_anomalies([p], pattern)
    label = clusters[0]["label"]
    assert label.startswith("elevated: products"), f"got: {label!r}"
    assert "missing: stores" in label


def test_classify_anomalies_label_display_name():
    """display_name is preferred over line_id in labels."""
    from hypertopos.model.sphere import RelationDef

    engine = GDSEngine(storage=None, cache=None)
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    pattern = Pattern(
        pattern_id="cp",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(
                line_id="products", direction="out", required=True, display_name="Product links"
            ),
        ],
        mu=np.zeros(1),
        sigma_diag=np.ones(1),
        theta=np.ones(1),
        population_size=10,
        computed_at=ts,
        version=1,
        status="production",
    )
    p = Polygon(
        primary_key="C-1",
        pattern_id="cp",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.array([-3.0], dtype=np.float32),
        delta_norm=3.0,
        is_anomaly=True,
        edges=[],
        last_refresh_at=ts,
        updated_at=ts,
    )
    clusters = engine.classify_anomalies([p], pattern)
    assert "Product links" in clusters[0]["label"]


# --- find_nearest ---


def test_find_nearest_returns_top_n():
    """find_nearest returns top_n closest entities by Euclidean delta distance."""
    import pyarrow as pa

    # 5 entities in 2D delta space
    geo_table = pa.table(
        {
            "primary_key": ["A", "B", "C", "D", "E"],
            "pattern_id": ["p"] * 5,
            "pattern_ver": [1] * 5,
            "pattern_type": ["anchor"] * 5,
            "scale": [1] * 5,
            "delta": [[0.0, 0.0], [0.1, 0.0], [0.5, 0.5], [1.0, 1.0], [0.05, 0.05]],
            "delta_norm": [0.0, 0.1, 0.707, 1.414, 0.0707],
            "is_anomaly": [False] * 5,
            "edges": pa.array(
                [[] for _ in range(5)],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
            "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 5,
            "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 5,
        }
    )

    class _StubReader:
        def read_geometry(self, pattern_id, version, **kw):
            return geo_table

    engine = GDSEngine(storage=_StubReader(), cache=GDSCache())
    ref_delta = np.array([0.0, 0.0], dtype=np.float32)

    results = engine.find_nearest(ref_delta, "p", version=1, top_n=3)

    assert len(results) == 3
    keys = [bk for bk, _ in results]
    # Closest to [0,0]: A(0.0), E(0.07), B(0.1)
    assert keys == ["A", "E", "B"]
    # Distances are sorted ascending
    assert results[0][1] < results[1][1] < results[2][1]


def test_find_nearest_exclude_keys():
    """find_nearest excludes specified business keys from results."""
    import pyarrow as pa

    geo_table = pa.table(
        {
            "primary_key": ["A", "B", "C"],
            "pattern_id": ["p"] * 3,
            "pattern_ver": [1] * 3,
            "pattern_type": ["anchor"] * 3,
            "scale": [1] * 3,
            "delta": [[0.0, 0.0], [0.1, 0.0], [0.5, 0.5]],
            "delta_norm": [0.0, 0.1, 0.707],
            "is_anomaly": [False] * 3,
            "edges": pa.array(
                [[] for _ in range(3)],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
            "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
            "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
        }
    )

    class _StubReader:
        def read_geometry(self, pattern_id, version, **kw):
            return geo_table

    engine = GDSEngine(storage=_StubReader(), cache=GDSCache())
    ref_delta = np.array([0.0, 0.0], dtype=np.float32)

    results = engine.find_nearest(ref_delta, "p", version=1, top_n=2, exclude_keys={"A"})

    keys = [bk for bk, _ in results]
    assert "A" not in keys
    assert keys == ["B", "C"]


def test_find_nearest_top_n_exceeds_population():
    """find_nearest returns all entities when top_n > population size."""
    import pyarrow as pa

    geo_table = pa.table(
        {
            "primary_key": ["A", "B"],
            "pattern_id": ["p"] * 2,
            "pattern_ver": [1] * 2,
            "pattern_type": ["anchor"] * 2,
            "scale": [1] * 2,
            "delta": [[0.0, 0.0], [1.0, 1.0]],
            "delta_norm": [0.0, 1.414],
            "is_anomaly": [False] * 2,
            "edges": pa.array(
                [[] for _ in range(2)],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
            "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
        }
    )

    class _StubReader:
        def read_geometry(self, pattern_id, version, **kw):
            return geo_table

    engine = GDSEngine(storage=_StubReader(), cache=GDSCache())
    ref_delta = np.array([0.0, 0.0], dtype=np.float32)

    results = engine.find_nearest(ref_delta, "p", version=1, top_n=100)
    assert len(results) == 2


# --- contrast_populations ---


def test_contrast_populations_basic():
    """Dimensions with larger mean differences get higher effect sizes and appear first."""
    engine = GDSEngine(storage=None, cache=None)
    rng = np.random.default_rng(42)
    # Group A: high dim0, low dim1; Group B: low dim0, high dim1
    delta_a = rng.normal(loc=[1.0, 0.0], scale=0.1, size=(20, 2)).astype(np.float32)
    delta_b = rng.normal(loc=[0.0, 1.0], scale=0.1, size=(20, 2)).astype(np.float32)
    delta_matrix = np.vstack([delta_a, delta_b])
    mask_a = np.array([True] * 20 + [False] * 20)
    mask_b = np.array([False] * 20 + [True] * 20)

    result = engine.contrast_populations(delta_matrix, mask_a, mask_b)

    assert len(result) == 2
    assert abs(result[0]["effect_size"]) >= abs(result[1]["effect_size"])
    assert abs(result[0]["effect_size"]) > 5.0
    assert abs(result[1]["effect_size"]) > 5.0
    for item in result:
        assert {"dim_index", "dim_label", "mean_a", "mean_b", "diff", "effect_size"} <= item.keys()
    assert result[0]["dim_label"].startswith("dim_")


def test_contrast_populations_custom_labels():
    """dim_labels parameter overrides default dim_N labels."""
    engine = GDSEngine(storage=None, cache=None)
    delta_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mask_a = np.array([True, False])
    mask_b = np.array([False, True])

    result = engine.contrast_populations(
        delta_matrix, mask_a, mask_b, dim_labels=["products", "stores"]
    )

    labels = {item["dim_index"]: item["dim_label"] for item in result}
    assert labels[0] == "products"
    assert labels[1] == "stores"


def test_contrast_populations_sigma_zero():
    """When within-group variance is zero, effect_size equals the raw diff."""
    engine = GDSEngine(storage=None, cache=None)
    delta_matrix = np.array(
        [
            [2.0, 5.0],
            [2.0, 5.0],
            [0.0, 3.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    mask_a = np.array([True, True, False, False])
    mask_b = np.array([False, False, True, True])

    result = engine.contrast_populations(delta_matrix, mask_a, mask_b)

    by_dim = {item["dim_index"]: item for item in result}
    assert pytest.approx(by_dim[0]["diff"], abs=1e-5) == 2.0
    assert pytest.approx(by_dim[0]["effect_size"], abs=1e-5) == 2.0
    assert pytest.approx(by_dim[1]["diff"], abs=1e-5) == 2.0


def test_contrast_populations_empty_group_raises():
    """Empty group_b raises ValueError."""
    engine = GDSEngine(storage=None, cache=None)
    delta_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_a = np.array([True, False])
    mask_b = np.array([False, False])

    with pytest.raises(ValueError, match="group_b"):
        engine.contrast_populations(delta_matrix, mask_a, mask_b)


# --- centroid_map ---


def test_centroid_map_basic():
    """Global + per-group centroids, inter-centroid distances, structural outlier."""
    engine = GDSEngine(storage=None, cache=None)
    # 3 groups: A (2 entities), B (2 entities), C (1 entity far out)
    delta_matrix = np.array(
        [
            [1.0, 0.0],  # group A
            [1.0, 0.2],  # group A
            [0.0, 1.0],  # group B
            [0.0, 1.2],  # group B
            [3.0, 3.0],  # group C (outlier)
        ],
        dtype=np.float32,
    )
    labels = ["A", "A", "B", "B", "C"]

    result = engine.compute_centroid_map(delta_matrix, labels)

    # Structure
    assert "global_centroid" in result
    assert "group_centroids" in result
    assert "inter_centroid_distances" in result
    assert "structural_outlier" in result

    # Global centroid = mean of all rows
    gc = result["global_centroid"]
    expected_global = np.mean(delta_matrix, axis=0)
    np.testing.assert_allclose(gc["vector"], expected_global, atol=1e-5)
    assert gc["count"] == 5

    # Group centroids
    groups = {g["key"]: g for g in result["group_centroids"]}
    assert len(groups) == 3
    np.testing.assert_allclose(groups["A"]["vector"], [1.0, 0.1], atol=1e-5)
    np.testing.assert_allclose(groups["B"]["vector"], [0.0, 1.1], atol=1e-5)
    np.testing.assert_allclose(groups["C"]["vector"], [3.0, 3.0], atol=1e-5)
    assert groups["A"]["count"] == 2
    assert groups["B"]["count"] == 2
    assert groups["C"]["count"] == 1

    # distance_to_global is L2 distance between group centroid and global centroid
    for g in result["group_centroids"]:
        expected_dist = float(np.linalg.norm(np.array(g["vector"]) - expected_global))
        assert abs(g["distance_to_global"] - expected_dist) < 1e-4

    # Structural outlier = group with max distance_to_global
    assert result["structural_outlier"]["key"] == "C"

    # Inter-centroid distances: 3 pairs (A-B, A-C, B-C)
    assert len(result["inter_centroid_distances"]) == 3
    pairs = {tuple(sorted(d["pair"])): d["distance"] for d in result["inter_centroid_distances"]}
    assert ("A", "B") in pairs
    assert ("A", "C") in pairs
    assert ("B", "C") in pairs


def test_centroid_map_dim_labels():
    """dim_labels parameter adds dimension names to output."""
    engine = GDSEngine(storage=None, cache=None)
    delta_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = ["X", "Y"]

    result = engine.compute_centroid_map(delta_matrix, labels, dim_labels=["products", "stores"])
    assert result["dimensions"] == ["products", "stores"]


def test_centroid_map_single_group():
    """With one group, inter-centroid distances is empty."""
    engine = GDSEngine(storage=None, cache=None)
    delta_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = ["A", "A"]

    result = engine.compute_centroid_map(delta_matrix, labels)

    assert len(result["group_centroids"]) == 1
    assert result["inter_centroid_distances"] == []
    assert result["structural_outlier"]["key"] == "A"


def test_centroid_map_radius_and_spread():
    """radius = mean ||delta|| within group, spread = std ||delta||."""
    engine = GDSEngine(storage=None, cache=None)
    delta_matrix = np.array(
        [
            [3.0, 4.0],  # norm = 5.0
            [0.0, 0.0],  # norm = 0.0
        ],
        dtype=np.float32,
    )
    labels = ["A", "A"]

    result = engine.compute_centroid_map(delta_matrix, labels)

    group = result["group_centroids"][0]
    assert abs(group["radius"] - 2.5) < 1e-5  # mean(5.0, 0.0)
    assert abs(group["spread"] - 2.5) < 1e-5  # std(5.0, 0.0) ddof=0


def test_centroid_map_empty_raises():
    """Empty delta matrix raises ValueError."""
    engine = GDSEngine(storage=None, cache=None)
    delta_matrix = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="empty"):
        engine.compute_centroid_map(delta_matrix, [])


def test_build_polygon_includes_delta_rank_pct(sphere_path):
    """build_polygon reads delta_rank_pct from geometry table."""
    reader = GDSReader(base_path=sphere_path)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = _make_manifest()
    polygon = engine.build_polygon("CUST-001", "customer_pattern", manifest)
    assert polygon.delta_rank_pct is not None
    assert 0.0 <= polygon.delta_rank_pct <= 100.0


# --- find_nearest Lance ANN delegation ---


def test_find_nearest_delegates_to_lance_ann(tmp_path):
    """GDSEngine.find_nearest() uses Lance ANN when available."""
    from unittest.mock import patch

    reader = GDSReader(base_path=str(tmp_path))
    engine = GDSEngine(storage=reader, cache=GDSCache())
    query = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ann_result = [("K-001", 0.1), ("K-002", 0.3)]

    with patch.object(reader, "find_nearest_lance", return_value=ann_result) as mock_ann:
        result = engine.find_nearest(query, "pat", 1, top_n=2)

    mock_ann.assert_called_once_with("pat", 1, query, 2, None, None)
    assert result == ann_result


# --- count_inside_alias ---


def _make_alias_with_cp(normal: list[float], bias: float) -> Alias:
    cp = CuttingPlane(normal=normal, bias=bias)
    return Alias(
        alias_id="test_alias",
        base_pattern_id="customer_pattern",
        filter=AliasFilter(include_relations=[], cutting_plane=cp),
        derived_pattern=DerivedPattern(
            mu=np.zeros(2, dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.zeros(2, dtype=np.float32),
            population_size=3,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        version=1,
        status="production",
    )


def _make_geo_table(deltas: list[list[float]]) -> pa.Table:
    """Minimal geometry Arrow table with delta column as list<float32>."""
    return pa.table(
        {
            "primary_key": [f"K-{i}" for i in range(len(deltas))],
            "delta": pa.array(deltas, type=pa.list_(pa.float32())),
        }
    )


def test_count_inside_alias_all_inside():
    engine = GDSEngine(storage=None, cache=None)
    alias = _make_alias_with_cp(normal=[1.0, 0.0], bias=0.0)
    geo = _make_geo_table([[1.0, 0.5], [2.0, 0.3], [0.5, 1.0]])
    assert engine.count_inside_alias(alias, geo) == 3


def test_count_inside_alias_partial():
    engine = GDSEngine(storage=None, cache=None)
    alias = _make_alias_with_cp(normal=[1.0, 0.0], bias=0.0)
    geo = _make_geo_table([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]])
    assert engine.count_inside_alias(alias, geo) == 2


def test_count_inside_alias_empty_geo():
    engine = GDSEngine(storage=None, cache=None)
    alias = _make_alias_with_cp(normal=[1.0, 0.0], bias=0.0)
    geo = _make_geo_table([])
    assert engine.count_inside_alias(alias, geo) == 0


def test_filter_geometry_by_delta_dim_gt():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()  # relations: [products (idx 0), stores (idx 1)]
    deltas = [[2.0, 0.5], [-1.0, 1.5], [0.5, -2.0]]
    geo = pa.table(
        {
            "primary_key": ["A", "B", "C"],
            "delta": pa.array(deltas, type=pa.list_(pa.float32())),
        }
    )
    # Keep rows where "products" (index 0) > 0.0 → A (2.0) and C (0.5), not B (-1.0)
    result = engine.filter_geometry_by_delta_dim(geo, pattern, {"products": {"gt": 0.0}})
    assert result.num_rows == 2
    assert set(result["primary_key"].to_pylist()) == {"A", "C"}


def test_filter_geometry_by_delta_dim_display_name():
    engine = GDSEngine(storage=None, cache=None)
    from datetime import datetime

    pattern_with_dn = Pattern(
        pattern_id="p1",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(
                line_id="products",
                direction="out",
                required=True,
                display_name="Purchased Products",
            ),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.zeros(2, dtype=np.float32),
        population_size=10,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    deltas = [[1.0, 0.5], [-0.5, 1.5]]
    geo = pa.table(
        {
            "primary_key": ["A", "B"],
            "delta": pa.array(deltas, type=pa.list_(pa.float32())),
        }
    )
    result = engine.filter_geometry_by_delta_dim(
        geo, pattern_with_dn, {"Purchased Products": {"gt": 0.0}}
    )
    assert result.num_rows == 1
    assert result["primary_key"][0].as_py() == "A"


def test_filter_geometry_by_delta_dim_empty_returns_empty():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    geo = pa.table(
        {
            "primary_key": pa.array([], type=pa.string()),
            "delta": pa.array([], type=pa.list_(pa.float32())),
        }
    )
    result = engine.filter_geometry_by_delta_dim(geo, pattern, {"products": {"gt": 0.0}})
    assert result.num_rows == 0


def test_filter_geometry_by_delta_dim_unknown_dim_raises():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    geo = pa.table(
        {
            "primary_key": ["A"],
            "delta": pa.array([[1.0, 0.5]], type=pa.list_(pa.float32())),
        }
    )
    with pytest.raises(ValueError, match="Dimension 'nonexistent'"):
        engine.filter_geometry_by_delta_dim(geo, pattern, {"nonexistent": {"gt": 0.0}})


def test_filter_geometry_by_delta_dim_multi_dim_and():
    """Multiple dimension specs combine with AND semantics."""
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()  # relations: [products (idx 0), stores (idx 1)]
    deltas = [[2.0, -1.0], [2.0, 1.0], [-1.0, -1.0]]
    geo = pa.table(
        {
            "primary_key": ["A", "B", "C"],
            "delta": pa.array(deltas, type=pa.list_(pa.float32())),
        }
    )
    # products > 0 AND stores < 0 → only A (2.0, -1.0)
    result = engine.filter_geometry_by_delta_dim(
        geo, pattern, {"products": {"gt": 0.0}, "stores": {"lt": 0.0}}
    )
    assert result.num_rows == 1
    assert result["primary_key"][0].as_py() == "A"


def test_filter_geometry_by_delta_dim_unknown_op_raises():
    engine = GDSEngine(storage=None, cache=None)
    pattern = _make_pattern()
    geo = pa.table(
        {
            "primary_key": ["A"],
            "delta": pa.array([[1.0, 0.5]], type=pa.list_(pa.float32())),
        }
    )
    with pytest.raises(ValueError, match="Unknown comparison op 'neq'"):
        engine.filter_geometry_by_delta_dim(geo, pattern, {"products": {"neq": 0.0}})


# --- filter_geometry_inside_alias ---


def test_filter_geometry_inside_alias_keeps_inside():
    """Only rows with positive signed distance are kept."""
    engine = GDSEngine(storage=None, cache=None)
    # normal=[1.0, 0.0], bias=0.0  →  signed_dist = delta[0]
    alias = _make_alias_with_cp(normal=[1.0, 0.0], bias=0.0)
    # delta[0]=1.0 → inside; delta[0]=-1.0 → outside; delta[0]=0.5 → inside
    geo = _make_geo_table([[1.0, 0.5], [-1.0, 0.3], [0.5, 0.0]])
    result = engine.filter_geometry_inside_alias(geo, alias)
    assert result.num_rows == 2
    assert set(result["primary_key"].to_pylist()) == {"K-0", "K-2"}


def test_filter_geometry_inside_alias_all_outside():
    """All rows outside → 0 rows returned."""
    engine = GDSEngine(storage=None, cache=None)
    alias = _make_alias_with_cp(normal=[1.0, 0.0], bias=0.0)
    geo = _make_geo_table([[-1.0, 0.5], [-2.0, 0.3]])
    result = engine.filter_geometry_inside_alias(geo, alias)
    assert result.num_rows == 0


def test_filter_geometry_inside_alias_empty_geo():
    """Empty geometry table is returned unchanged without error."""
    engine = GDSEngine(storage=None, cache=None)
    alias = _make_alias_with_cp(normal=[1.0, 0.0], bias=0.0)
    geo = _make_geo_table([])
    result = engine.filter_geometry_inside_alias(geo, alias)
    assert result.num_rows == 0
    assert result is geo


def test_filter_geometry_inside_alias_no_cutting_plane_returns_unchanged():
    """Alias without cutting_plane → geo returned unchanged."""
    from unittest.mock import MagicMock

    engine = GDSEngine(storage=None, cache=None)
    alias = MagicMock()
    alias.filter.cutting_plane = None
    geo = _make_geo_table([[1.0, 0.5], [0.5, 0.3]])
    result = engine.filter_geometry_inside_alias(geo, alias)
    assert result is geo


class TestKmeans:
    """Unit tests for GDSEngine._kmeans — no storage needed."""

    def test_kmeans_partitions_all_entities(self):
        from hypertopos.engine.geometry import GDSEngine

        rng = np.random.default_rng(0)
        group_a = rng.normal([0.0, 0.0], 0.05, size=(20, 2)).astype(np.float32)
        group_b = rng.normal([5.0, 5.0], 0.05, size=(20, 2)).astype(np.float32)
        matrix = np.vstack([group_a, group_b])

        labels, centroids = GDSEngine._kmeans(matrix, n_clusters=2, seed=0)

        assert labels.shape == (40,)
        assert centroids.shape == (2, 2)
        assert set(labels.tolist()) == {0, 1}

    def test_kmeans_discovers_separated_clusters(self):
        from hypertopos.engine.geometry import GDSEngine

        rng = np.random.default_rng(1)
        group_a = rng.normal([0.0, 0.0], 0.01, size=(10, 2)).astype(np.float32)
        group_b = rng.normal([10.0, 10.0], 0.01, size=(10, 2)).astype(np.float32)
        matrix = np.vstack([group_a, group_b])

        labels, centroids = GDSEngine._kmeans(matrix, n_clusters=2, seed=0)

        assert len(set(labels[:10].tolist())) == 1
        assert len(set(labels[10:].tolist())) == 1
        assert labels[0] != labels[10]

    def test_kmeans_n_clusters_capped_at_n(self):
        from hypertopos.engine.geometry import GDSEngine

        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        labels, centroids = GDSEngine._kmeans(matrix, n_clusters=5, seed=0)
        assert centroids.shape[0] <= 2

    def test_kmeans_single_cluster_centroid_is_mean(self):
        from hypertopos.engine.geometry import GDSEngine

        matrix = np.array([[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]], dtype=np.float32)
        labels, centroids = GDSEngine._kmeans(matrix, n_clusters=1, seed=0)
        assert labels.tolist() == [0, 0, 0]
        np.testing.assert_allclose(centroids[0], matrix.mean(axis=0), atol=1e-6)


class TestFindClusters:
    def _make_engine(self):
        from unittest.mock import MagicMock

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache

        return GDSEngine(storage=MagicMock(), cache=GDSCache())

    def test_find_clusters_returns_correct_structure(self):
        engine = self._make_engine()
        rng = np.random.default_rng(0)
        matrix = np.vstack(
            [
                rng.normal([0.0, 0.0], 0.05, (10, 2)),
                rng.normal([5.0, 5.0], 0.05, (10, 2)),
            ]
        ).astype(np.float32)
        keys = [f"K-{i:03d}" for i in range(20)]
        is_anomaly_flags = [False] * 10 + [True] * 10
        delta_norms = [float(np.linalg.norm(row)) for row in matrix]

        clusters = engine.find_clusters(
            delta_matrix=matrix,
            keys=keys,
            is_anomaly_flags=is_anomaly_flags,
            delta_norms=delta_norms,
            n_clusters=2,
            dim_names=["dim_a", "dim_b"],
        )

        assert len(clusters) == 2
        required = {
            "cluster_id",
            "size",
            "anomaly_rate",
            "centroid_delta",
            "delta_norm_mean",
            "delta_norm_std",
            "representative_key",
            "dim_profile",
            "member_keys",
        }
        for c in clusters:
            assert required == set(c.keys())
            assert 0.0 <= c["anomaly_rate"] <= 1.0
            assert len(c["dim_profile"]) == 2
            assert c["dim_profile"][0]["dimension"] == "dim_a"

    def test_find_clusters_sizes_sum_to_n(self):
        engine = self._make_engine()
        rng = np.random.default_rng(2)
        matrix = rng.random((15, 3)).astype(np.float32)
        keys = [f"K-{i}" for i in range(15)]

        clusters = engine.find_clusters(
            delta_matrix=matrix,
            keys=keys,
            is_anomaly_flags=[False] * 15,
            delta_norms=[0.5] * 15,
            n_clusters=3,
            dim_names=["a", "b", "c"],
        )

        assert sum(c["size"] for c in clusters) == 15

    def test_find_clusters_sorted_by_size_descending(self):
        engine = self._make_engine()
        rng = np.random.default_rng(3)
        matrix = np.vstack(
            [
                rng.normal([0.0, 0.0], 0.1, (8, 2)),
                rng.normal([10.0, 10.0], 0.1, (2, 2)),
            ]
        ).astype(np.float32)
        keys = [f"K-{i}" for i in range(10)]

        clusters = engine.find_clusters(
            delta_matrix=matrix,
            keys=keys,
            is_anomaly_flags=[False] * 10,
            delta_norms=[0.5] * 10,
            n_clusters=2,
            dim_names=["x", "y"],
        )

        sizes = [c["size"] for c in clusters]
        assert sizes == sorted(sizes, reverse=True)
        assert clusters[0]["size"] == 8

    def test_find_clusters_empty_returns_empty(self):
        engine = self._make_engine()
        result = engine.find_clusters(
            delta_matrix=np.empty((0, 3), dtype=np.float32),
            keys=[],
            is_anomaly_flags=[],
            delta_norms=[],
            n_clusters=3,
            dim_names=["a", "b", "c"],
        )
        assert result == []


# ---------------------------------------------------------------------------
# Property fill vector and compute_delta with prop_fill
# ---------------------------------------------------------------------------


def test_prop_fill_vector_present():
    pts = pa.table({"primary_key": ["C-1"], "name": ["Acme"], "region": ["EMEA"]})
    engine = GDSEngine(storage=None, cache=None)
    v = engine._prop_fill_vector(pts, ["name", "region"])
    np.testing.assert_array_equal(v, [1.0, 1.0])


def test_prop_fill_vector_missing():
    pts = pa.table({"primary_key": ["C-1"], "name": [None], "region": ["EMEA"]})
    engine = GDSEngine(storage=None, cache=None)
    v = engine._prop_fill_vector(pts, ["name", "region"])
    np.testing.assert_array_equal(v, [0.0, 1.0])


def test_prop_fill_vector_empty_table():
    pts = pa.table(
        {
            "primary_key": pa.array([], type=pa.string()),
            "name": pa.array([], type=pa.string()),
        }
    )
    engine = GDSEngine(storage=None, cache=None)
    v = engine._prop_fill_vector(pts, ["name", "region"])
    np.testing.assert_array_equal(v, [0.0, 0.0])


def test_compute_delta_with_prop_fill():
    """Property fill dims are appended to the shape vector before normalization."""
    from hypertopos.model.sphere import RelationDef

    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[RelationDef(line_id="orders", direction="out", required=True)],
        mu=np.array([0.5, 0.9, 0.3], dtype=np.float32),
        sigma_diag=np.array([0.3, 0.1, 0.4], dtype=np.float32),
        theta=np.array([1.5, 2.0, 2.0], dtype=np.float32),
        population_size=10,
        computed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),  # noqa: UP017
        version=1,
        status="production",
        prop_columns=["name", "region"],
    )
    polygon = Polygon(
        primary_key="C-1",
        pattern_id="p",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(3, dtype=np.float32),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[Edge(line_id="orders", point_key="O-1", status="alive", direction="out")],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=timezone.utc),  # noqa: UP017
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),  # noqa: UP017
    )
    engine = GDSEngine(storage=None, cache=None)
    prop_fill = np.array([1.0, 0.0], dtype=np.float32)
    delta = engine.compute_delta(polygon, pattern, prop_fill=prop_fill)
    # edge dim: shape=1.0 → delta[0] = (1.0 - 0.5) / 0.3 ≈ 1.667
    # name dim: fill=1.0 → delta[1] = (1.0 - 0.9) / 0.1 = 1.0
    # region dim: fill=0.0 → delta[2] = (0.0 - 0.3) / 0.4 = -0.75
    assert len(delta) == 3
    assert abs(delta[0] - 1.667) < 0.01
    assert abs(delta[1] - 1.0) < 0.01
    assert abs(delta[2] - (-0.75)) < 0.01


def test_compute_centroid_map_member_samples_are_group_specific():
    """member_samples in each group must contain only keys from that group."""
    engine = GDSEngine(storage=None, cache=None)

    # Group A: 4 entities with high delta[0]
    group_a_keys = ["A-1", "A-2", "A-3", "A-4"]
    group_a_deltas = np.array(
        [
            [2.0, 0.0],
            [2.1, 0.1],
            [1.9, -0.1],
            [2.2, 0.2],
        ],
        dtype=np.float32,
    )

    # Group B: 4 entities with high delta[1]
    group_b_keys = ["B-1", "B-2", "B-3", "B-4"]
    group_b_deltas = np.array(
        [
            [0.0, 3.0],
            [0.1, 3.1],
            [-0.1, 2.9],
            [0.2, 3.2],
        ],
        dtype=np.float32,
    )

    delta_matrix = np.vstack([group_a_deltas, group_b_deltas])
    entity_keys = group_a_keys + group_b_keys
    group_labels = ["A"] * 4 + ["B"] * 4

    result = engine.compute_centroid_map(delta_matrix, group_labels, entity_keys=entity_keys)

    groups = {g["key"]: g for g in result["group_centroids"]}

    # Each group must have member_samples
    assert "member_samples" in groups["A"], "Group A missing member_samples"
    assert "member_samples" in groups["B"], "Group B missing member_samples"

    # Samples must be non-empty
    assert len(groups["A"]["member_samples"]) > 0
    assert len(groups["B"]["member_samples"]) > 0

    # Critical: samples must only contain keys from their own group
    for key in groups["A"]["member_samples"]:
        assert key in group_a_keys, f"Group A sample {key!r} is not an A-key"
    for key in groups["B"]["member_samples"]:
        assert key in group_b_keys, f"Group B sample {key!r} is not a B-key"

    # The two groups must NOT share the same member_samples
    assert set(groups["A"]["member_samples"]) != set(groups["B"]["member_samples"])


# --- Degenerate pattern: theta_norm=0 must not produce is_anomaly=True ---


def test_build_polygon_zero_theta_not_anomaly():
    """When theta=zeros (degenerate pattern), build_polygon must return is_anomaly=False.

    A pattern where all entities have identical zero-delta geometry gets theta=zero.
    In that state, 0.0 >= 0.0 is True, which incorrectly flags every entity anomalous.
    The fix: is_anomaly is False whenever theta_norm == 0.0.
    """
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Pattern, RelationDef, Sphere

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: UP017

    # Pattern with theta = zero vector → theta_norm = 0.0
    pattern = Pattern(
        pattern_id="zero_theta_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=False),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.zeros(1, dtype=np.float32),  # degenerate: theta_norm = 0.0
        population_size=10,
        computed_at=now,
        version=1,
        status="production",
    )
    sphere = Sphere(
        sphere_id="test",
        name="test",
        base_path=".",
        patterns={"zero_theta_pattern": pattern},
    )

    geometry_table = pa.table(
        {
            "primary_key": pa.array(["CUST-1"]),
            "pattern_id": pa.array(["zero_theta_pattern"]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "pattern_type": pa.array(["anchor"]),
            "scale": pa.array([1], type=pa.int64()),
            "delta": pa.array([[0.0]]),  # delta_norm = 0.0
            "delta_norm": pa.array([0.0], type=pa.float64()),
            "is_anomaly": pa.array([False]),
            "edges": pa.array(
                [[]],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
            "version": pa.array([1], type=pa.int64()),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        }
    )

    mock_storage = MagicMock()
    mock_storage.read_geometry.return_value = geometry_table
    mock_storage.read_sphere.return_value = sphere

    engine = GDSEngine(storage=mock_storage, cache=GDSCache())
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="test-agent",
        snapshot_time=now,
        status="active",
        line_versions={},
        pattern_versions={"zero_theta_pattern": 1},
    )

    polygon = engine.build_polygon("CUST-1", "zero_theta_pattern", manifest)

    # theta_norm = 0.0 → degenerate pattern → is_anomaly must be False
    assert not polygon.is_anomaly, (
        f"Expected is_anomaly=False for degenerate pattern (theta_norm=0), "
        f"got is_anomaly={polygon.is_anomaly}"
    )


def test_build_polygon_sets_is_jumpable_based_on_point_key():
    """Edges with non-empty point_key must have is_jumpable=True.
    Edges with empty point_key (continuous mode) must have is_jumpable=False."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Pattern, RelationDef, Sphere

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: UP017

    pattern = Pattern(
        pattern_id="test_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=False),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.zeros(2, dtype=np.float32),
        population_size=10,
        computed_at=now,
        version=1,
        status="production",
    )
    sphere = Sphere(
        sphere_id="test",
        name="test",
        base_path=".",
        patterns={"test_pattern": pattern},
    )

    geometry_table = pa.table(
        {
            "primary_key": pa.array(["CUST-1"]),
            "pattern_id": pa.array(["test_pattern"]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "pattern_type": pa.array(["anchor"]),
            "scale": pa.array([1], type=pa.int64()),
            "delta": pa.array([[0.0, 0.0]]),
            "delta_norm": pa.array([0.0], type=pa.float64()),
            "is_anomaly": pa.array([False]),
            "edges": pa.array(
                [
                    [
                        {
                            "line_id": "products",
                            "point_key": "MERCH-001",
                            "status": "alive",
                            "direction": "out",
                        },  # noqa: E501
                        {
                            "line_id": "stores",
                            "point_key": "",
                            "status": "alive",
                            "direction": "in",
                        },  # noqa: E501
                    ]
                ],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
            "version": pa.array([1], type=pa.int64()),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        }
    )

    mock_storage = MagicMock()
    mock_storage.read_geometry.return_value = geometry_table
    mock_storage.read_sphere.return_value = sphere

    engine = GDSEngine(storage=mock_storage, cache=GDSCache())
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="test-agent",
        snapshot_time=now,
        status="active",
        line_versions={},
        pattern_versions={"test_pattern": 1},
    )

    polygon = engine.build_polygon("CUST-1", "test_pattern", manifest)

    assert len(polygon.edges) == 2
    binary_edge = next(e for e in polygon.edges if e.point_key == "MERCH-001")
    continuous_edge = next(e for e in polygon.edges if e.point_key == "")
    assert binary_edge.is_jumpable is True, (
        f"Edge with non-empty point_key must be jumpable, got is_jumpable={binary_edge.is_jumpable}"
    )
    assert continuous_edge.is_jumpable is False, (
        f"Edge with empty point_key (continuous mode) must not be jumpable, "
        f"got is_jumpable={continuous_edge.is_jumpable}"
    )


class TestFindOptimalK:
    def _make_engine(self):
        from unittest.mock import MagicMock

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache

        return GDSEngine(storage=MagicMock(), cache=GDSCache())

    def test_two_blobs_finds_k2(self):
        """Two well-separated blobs should yield optimal k=2."""
        rng = np.random.default_rng(42)
        blob_a = rng.normal(loc=0, scale=0.5, size=(100, 4)).astype(np.float32)
        blob_b = rng.normal(loc=5, scale=0.5, size=(100, 4)).astype(np.float32)
        matrix = np.vstack([blob_a, blob_b])
        engine = self._make_engine()
        result = engine.find_optimal_k(matrix, k_max=10)
        assert result["best_k"] == 2
        assert "silhouette_per_k" in result
        assert result["best_silhouette"] > 0.5

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((50, 3)).astype(np.float32)
        engine = self._make_engine()
        result = engine.find_optimal_k(matrix, k_max=5)
        assert {"best_k", "silhouette_per_k", "best_silhouette", "silhouette_margin"} <= set(
            result.keys()
        )

    def test_k_max_capped_by_sqrt_n(self):
        """k_max is capped at min(k_max, N-1, sqrt(N), 15)."""
        rng = np.random.default_rng(42)
        # N=36 → sqrt(N)=6, so k_max should be capped at 6
        matrix = rng.standard_normal((36, 3)).astype(np.float32)
        engine = self._make_engine()
        result = engine.find_optimal_k(matrix, k_max=100)
        assert max(result["silhouette_per_k"].keys()) <= 6

    def test_tiny_population_returns_k1(self):
        """N=3 → sqrt(N)≈1.7 → k_max<2 → returns best_k=1."""
        matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        engine = self._make_engine()
        result = engine.find_optimal_k(matrix, k_max=100)
        assert result["best_k"] == 1
        assert result["silhouette_per_k"] == {}


# --- geometry_to_polygons ---


def test_geometry_to_polygons_edges_struct():
    """Build Polygon objects from Arrow table with edges struct encoding."""
    from hypertopos.storage._schemas import EDGE_STRUCT_TYPE

    now = datetime(2024, 1, 1, tzinfo=UTC)
    edges_a = [
        {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"},
        {"line_id": "stores", "point_key": "S-1", "status": "alive", "direction": "in"},
    ]
    edges_b = [
        {"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "in"},
    ]
    geo = pa.table(
        {
            "primary_key": ["A", "B"],
            "scale": [1, 1],
            "delta": pa.array([[1.0, 0.0], [0.0, 2.0]], type=pa.list_(pa.float32())),
            "delta_norm": [1.0, 2.0],
            "is_anomaly": [False, True],
            "edges": pa.array([edges_a, edges_b], type=pa.list_(EDGE_STRUCT_TYPE)),
            "last_refresh_at": [now, now],
            "updated_at": [now, now],
        }
    )
    engine = GDSEngine(storage=None, cache=None)
    polygons = engine.geometry_to_polygons(
        geo,
        pattern_id="p",
        pattern_type="anchor",
        pattern_ver=1,
    )

    assert len(polygons) == 2
    # Sorted by delta_norm descending -> B first
    assert polygons[0].primary_key == "B"
    assert polygons[0].delta_norm == 2.0
    assert polygons[0].is_anomaly is True
    assert polygons[1].primary_key == "A"
    assert polygons[1].delta_norm == 1.0

    # Check edges decoded from struct column
    assert len(polygons[1].edges) == 2
    assert polygons[1].edges[0].line_id == "products"
    assert polygons[1].edges[0].point_key == "P-1"
    assert polygons[1].edges[0].is_jumpable is True
    assert len(polygons[0].edges) == 1


def test_geometry_to_polygons_top_n():
    """Truncation via top_n returns only the top entries."""
    from hypertopos.storage._schemas import EDGE_STRUCT_TYPE

    now = datetime(2024, 1, 1, tzinfo=UTC)
    geo = pa.table(
        {
            "primary_key": ["A", "B", "C"],
            "scale": [1] * 3,
            "delta": pa.array([[1.0, 0.0], [0.0, 3.0], [2.0, 0.0]], type=pa.list_(pa.float32())),
            "delta_norm": [1.0, 3.0, 2.0],
            "is_anomaly": [False, True, False],
            "edges": pa.array([[], [], []], type=pa.list_(EDGE_STRUCT_TYPE)),
            "last_refresh_at": [now] * 3,
            "updated_at": [now] * 3,
        }
    )
    engine = GDSEngine(storage=None, cache=None)
    polygons = engine.geometry_to_polygons(
        geo,
        top_n=2,
        pattern_id="p",
        pattern_type="anchor",
        pattern_ver=1,
    )

    assert len(polygons) == 2
    assert polygons[0].primary_key == "B"  # delta_norm=3.0
    assert polygons[1].primary_key == "C"  # delta_norm=2.0


def test_geometry_to_polygons_norm_lookup():
    """norm_lookup overrides delta_norm from the table."""
    from hypertopos.storage._schemas import EDGE_STRUCT_TYPE

    now = datetime(2024, 1, 1, tzinfo=UTC)
    geo = pa.table(
        {
            "primary_key": ["A"],
            "scale": [1],
            "delta": pa.array([[3.0, 4.0]], type=pa.list_(pa.float32())),
            "delta_norm": [0.0],  # will be overridden
            "is_anomaly": [False],
            "edges": pa.array([[]], type=pa.list_(EDGE_STRUCT_TYPE)),
            "last_refresh_at": [now],
            "updated_at": [now],
        }
    )
    engine = GDSEngine(storage=None, cache=None)
    polygons = engine.geometry_to_polygons(
        geo,
        norm_lookup={"A": 99.0},
        pattern_id="p",
        pattern_type="anchor",
        pattern_ver=1,
    )
    assert polygons[0].delta_norm == 99.0


# --- anomaly_dimensions ---


def test_anomaly_dimensions_basic():
    """delta=[3.0, 4.0, 0.0]: b ~64%, a ~36%, c filtered (<5%)."""
    result = GDSEngine.anomaly_dimensions(
        [3.0, 4.0, 0.0],
        ["a", "b", "c"],
        top_n=3,
    )
    assert len(result) == 2  # c should be filtered out
    assert result[0]["label"] == "b"
    assert result[0]["contribution_pct"] == pytest.approx(64.0, abs=1.0)
    assert result[1]["label"] == "a"
    assert result[1]["contribution_pct"] == pytest.approx(36.0, abs=1.0)


def test_anomaly_dimensions_empty():
    """Near-zero delta returns empty list."""
    result = GDSEngine.anomaly_dimensions(
        [0.0, 0.0, 0.0],
        ["a", "b", "c"],
    )
    assert result == []


class TestFindClustersAutoK:
    def _make_engine(self):
        from unittest.mock import MagicMock

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache

        return GDSEngine(storage=MagicMock(), cache=GDSCache())

    def test_auto_k_uses_optimal(self):
        """n_clusters=0 should auto-detect and use optimal k."""
        rng = np.random.default_rng(42)
        blob_a = rng.normal(loc=0, scale=0.3, size=(80, 3)).astype(np.float32)
        blob_b = rng.normal(loc=5, scale=0.3, size=(80, 3)).astype(np.float32)
        matrix = np.vstack([blob_a, blob_b])
        keys = [f"e{i}" for i in range(160)]
        is_anom = [False] * 160
        norms = [float(np.linalg.norm(row)) for row in matrix]
        dim_names = ["d0", "d1", "d2"]
        engine = self._make_engine()
        clusters = engine.find_clusters(
            matrix, keys, is_anom, norms, n_clusters=0, dim_names=dim_names
        )
        # Auto-k should produce ~2 clusters for well-separated blobs
        assert 1 <= len(clusters) <= 4

    def test_auto_k_produces_clusters(self):
        """Auto-k should produce valid clusters for random data."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((50, 3)).astype(np.float32)
        keys = [f"e{i}" for i in range(50)]
        is_anom = [False] * 50
        norms = [float(np.linalg.norm(row)) for row in matrix]
        engine = self._make_engine()
        clusters = engine.find_clusters(
            matrix, keys, is_anom, norms, n_clusters=0, dim_names=["a", "b", "c"]
        )
        assert len(clusters) > 0
