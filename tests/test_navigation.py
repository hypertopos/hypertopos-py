# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.engine.subprocess_agg import _find_anomalies_table
from hypertopos.model.manifest import Contract, Manifest
from hypertopos.model.objects import Edge, Point, Polygon, Solid, SolidSlice
from hypertopos.navigation.navigator import (
    GDSCorruptedFileError,
    GDSEntityNotFoundError,
    GDSError,
    GDSMissingFileError,
    GDSNavigationError,
    GDSNavigator,
    GDSNoAliveEdgeError,
    GDSPositionError,
    GDSStorageError,
    GDSVersionError,
)

# Edge struct type used in geometry tables
_EDGE_STRUCT = pa.struct(
    [
        pa.field("line_id", pa.string()),
        pa.field("point_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("direction", pa.string()),
    ]
)


def _make_polygon_with_edge(target_key: str, target_line: str) -> Polygon:
    return Polygon(
        primary_key="SALE-001",
        pattern_id="sale_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(2),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[
            Edge(line_id="customers", point_key="CUST-001", status="alive", direction="in"),
            Edge(line_id=target_line, point_key=target_key, status="alive", direction="in"),
        ],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


class _MockEngine:
    def build_polygon(self, bk, pid, manifest):
        return _make_polygon_with_edge("PROD-001", "products")

    def build_solid(self, bk, pid, manifest, filters=None, timestamp=None):
        from hypertopos.model.objects import Solid

        return Solid(
            primary_key=bk,
            pattern_id=pid,
            base_polygon=_make_polygon_with_edge("PROD-001", "products"),
            slices=[],
        )


class _MockStorage:
    _POINTS: dict[str, list[str]] = {
        "customers": ["CUST-001", "CUST-002"],
        "products": ["PROD-001", "PROD-002"],
    }

    def read_points(self, line_id, version, filters=None):
        import pyarrow as pa

        keys = self._POINTS.get(line_id, ["UNKNOWN-001"])
        n = len(keys)
        return pa.table(
            {
                "primary_key": keys,
                "version": [1] * n,
                "status": ["active"] * n,
                "created_at": [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                "changed_at": [datetime(2024, 1, 1, tzinfo=UTC)] * n,
            }
        )

    def read_geometry(self, *a, **kw):
        import pyarrow as pa

        return pa.table({})

    def read_geometry_stats(self, *a, **kw):
        return None  # No cache — triggers full scan fallback in navigator

    def read_sphere(self):
        from hypertopos.model.sphere import Sphere

        return Sphere("s", "s", ".")

    def count_geometry_rows(self, *a, **kw):
        return 0  # Small mock dataset — never triggers subprocess fast path

    def resolve_primary_keys_by_edge(self, *a, **kw):
        return None  # No secondary index — triggers fallback path in navigator


class _MockStorageStaleNorm(_MockStorage):
    """Mock storage with deliberately stale (inconsistent) delta_norm values.

    E-001: stored delta_norm=3.5 but actual np.linalg.norm([1.0, 1.0])=√2≈1.414 → EXCLUDED
    E-002: stored delta_norm=4.0 and actual np.linalg.norm([3.0, 3.0])=3√2≈4.243 → INCLUDED
    Pattern theta=[0.0, 3.0] → theta_norm=3.0; threshold (radius=1)=3.0.
    """

    def read_geometry(
        self,
        pattern_id,
        version,
        primary_key=None,
        filters=None,
        point_keys=None,
        columns=None,
        filter=None,
    ):
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        table = pa.table(
            {
                "primary_key": ["E-001", "E-002"],
                "scale": [1, 1],
                "delta": [[1.0, 1.0], [3.0, 3.0]],
                "delta_norm": [3.5, 4.0],
                "is_anomaly": [True, True],
                "delta_rank_pct": pa.array([50.0, 90.0], type=pa.float64()),
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [dt, dt],
                "updated_at": [dt, dt],
            }
        )
        if filter is not None:
            import re as _re

            import pyarrow.compute as _pc

            if "primary_key IN" in str(filter):
                keys = _re.findall(r"'([^']*)'", str(filter))
                table = table.filter(
                    _pc.is_in(table["primary_key"], pa.array(keys)),
                )
            else:
                # Simulate Lance "delta_norm >= X" pushdown
                threshold = float(filter.split(">=")[1].strip())
                table = table.filter(
                    _pc.greater_equal(table["delta_norm"], threshold),
                )
        if point_keys is not None:
            import pyarrow.compute as _pc

            mask = _pc.is_in(table["primary_key"], pa.array(point_keys))
            table = table.filter(mask)
        if columns is not None:
            available = set(table.schema.names)
            columns = [c for c in columns if c in available]
            if columns:
                table = table.select(columns)
        return table

    def read_sphere(self):
        from hypertopos.model.sphere import Pattern, RelationDef, Sphere

        pat = Pattern(
            pattern_id="test_pattern",
            entity_type="test",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="line_a", direction="in", required=True),
                RelationDef(line_id="line_b", direction="in", required=True),
            ],
            mu=np.zeros(2, dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.array([0.0, 3.0], dtype=np.float32),
            population_size=10,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = Sphere("s", "s", ".")
        sphere.patterns["test_pattern"] = pat
        return sphere


class _MockStorageWithDelta(_MockStorage):
    """Mock storage that returns geometry for find_similar_entities tests."""

    def read_geometry(
        self,
        pattern_id,
        version,
        primary_key=None,  # noqa: E501
        filters=None,
        point_keys=None,
        columns=None,
        filter=None,
    ):
        import pyarrow as pa

        if primary_key is None:
            return pa.table({})
        all_cols = {
            "delta": [[0.1, 0.2]],
            "delta_norm": [0.224],
            "is_anomaly": [False],
            "delta_rank_pct": [25.0],
        }
        wanted = set(columns) if columns else {"delta"}
        return pa.table({k: v for k, v in all_cols.items() if k in wanted})


class _MockStorageWithGeometry(_MockStorage):
    """Mock storage that returns geometry rows for event_polygons_for tests."""

    def read_sphere(self):
        from hypertopos.model.sphere import Pattern, RelationDef, Sphere

        pat = Pattern(
            pattern_id="sale_pattern",
            entity_type="sales",
            pattern_type="event",
            relations=[
                RelationDef(line_id="customers", direction="in", required=True),
                RelationDef(line_id="products", direction="in", required=True),
            ],
            mu=np.zeros(2, dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.array([3.0, 3.0], dtype=np.float32),
            population_size=3,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        return Sphere("s", "s", ".", patterns={"sale_pattern": pat})

    def read_geometry(
        self,
        pattern_id,
        version,
        primary_key=None,
        filters=None,
        point_keys=None,
        columns=None,
        filter=None,
    ):
        import pyarrow as pa

        if pattern_id != "sale_pattern":
            return pa.table({})
        # Two sale polygons: SALE-001 references CUST-001, SALE-002 references CUST-999
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        edges_s1 = [
            {"line_id": "customers", "point_key": "CUST-001", "status": "alive", "direction": "in"},
            {"line_id": "products", "point_key": "PROD-001", "status": "alive", "direction": "in"},
        ]
        edges_s2 = [
            {"line_id": "customers", "point_key": "CUST-999", "status": "alive", "direction": "in"},
            {"line_id": "products", "point_key": "PROD-002", "status": "alive", "direction": "in"},
        ]
        edges_s3 = [
            {"line_id": "customers", "point_key": "CUST-001", "status": "alive", "direction": "in"},
            {"line_id": "products", "point_key": "PROD-003", "status": "alive", "direction": "in"},
        ]
        table = pa.table(
            {
                "primary_key": ["SALE-001", "SALE-002", "SALE-003"],
                "scale": [1, 1, 1],
                "delta": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                "delta_norm": [0.22, 0.5, 0.78],
                "is_anomaly": [False, True, True],
                "edges": pa.array([edges_s1, edges_s2, edges_s3], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [dt, dt, dt],
                "updated_at": [dt, dt, dt],
            }
        )
        # Simulate point_keys filtering (like GDSReader._filter_by_point_keys)
        if point_keys is not None:
            key_set = set(point_keys)
            keep = []
            for i in range(table.num_rows):
                edges = table["edges"][i].as_py() or []
                if any(e["point_key"] in key_set for e in edges):
                    keep.append(i)
            if not keep:
                return pa.table(
                    {
                        col: pa.array([], type=table.schema.field(col).type)
                        for col in table.schema.names
                    }
                )
            table = table.take(keep)
        return table


def _make_manifest() -> Manifest:
    return Manifest(
        manifest_id="m-001",
        agent_id="agent-001",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "products": 1},
        pattern_versions={"customer_pattern": 1, "cust_pattern": 1},
    )


def test_goto_sets_position():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    assert nav.position is not None
    assert nav.position.primary_key == "CUST-001"


def test_pi2_jump_polygon():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    polygon = _make_polygon_with_edge("PROD-001", "products")
    nav.π2_jump_polygon(polygon, target_line_id="products")
    assert nav.position.primary_key == "PROD-001"
    assert nav.position.line_id == "products"


def _make_polygon_with_multi_edge(target_line: str) -> Polygon:
    return Polygon(
        primary_key="SALE-001",
        pattern_id="sale_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(2),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[
            Edge(line_id=target_line, point_key="PROD-001", status="alive", direction="in"),
            Edge(line_id=target_line, point_key="PROD-002", status="alive", direction="in"),
            Edge(line_id=target_line, point_key="PROD-003", status="dead", direction="in"),
        ],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_jump_polygon_multi_edge_default():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    polygon = _make_polygon_with_multi_edge("products")
    nav.π2_jump_polygon(polygon, target_line_id="products")
    assert nav.position.primary_key == "PROD-001"


def test_jump_polygon_edge_index_1():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    polygon = _make_polygon_with_multi_edge("products")
    nav.π2_jump_polygon(polygon, target_line_id="products", edge_index=1)
    assert nav.position.primary_key == "PROD-002"


def test_jump_polygon_edge_index_out_of_range():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    polygon = _make_polygon_with_multi_edge("products")
    with pytest.raises(GDSNavigationError):
        nav.π2_jump_polygon(polygon, target_line_id="products", edge_index=2)


def test_jump_polygon_negative_edge_index_raises():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    polygon = _make_polygon_with_multi_edge("products")
    with pytest.raises(GDSNavigationError):
        nav.π2_jump_polygon(polygon, target_line_id="products", edge_index=-1)


def test_jump_polygon_single_edge_index_0():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    polygon = _make_polygon_with_edge("PROD-001", "products")
    nav.π2_jump_polygon(polygon, target_line_id="products", edge_index=0)
    assert nav.position.primary_key == "PROD-001"


def test_pi2_raises_if_no_alive_edge():
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    dead_polygon = Polygon(
        primary_key="SALE-001",
        pattern_id="sale_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(2),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[
            Edge(line_id="products", point_key="PROD-001", status="dead", direction="in"),
        ],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(GDSNavigationError):
        nav.π2_jump_polygon(dead_polygon, target_line_id="products")


def test_jump_polygon_raises_on_continuous_edge():
    """π2_jump_polygon must raise ValueError when edge has empty point_key (continuous mode)."""
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )
    nav.goto("CUST-001", "customers")
    continuous_polygon = Polygon(
        primary_key="CUST-001",
        pattern_id="anchor_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(2),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[
            Edge(line_id="customers", point_key="CUST-001", status="alive", direction="self"),
            Edge(line_id="company_codes", point_key="", status="alive", direction="in"),
        ],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="continuous mode"):
        nav.π2_jump_polygon(continuous_polygon, target_line_id="company_codes")


# ---------- π3_dive_solid tests ----------


class _SlicedEngine:
    """Engine that returns a Solid with three slices for timestamp-filtering tests."""

    _ts_early = datetime(2022, 1, 1, tzinfo=UTC)
    _ts_mid = datetime(2023, 1, 1, tzinfo=UTC)
    _ts_late = datetime(2024, 1, 1, tzinfo=UTC)

    def build_polygon(self, bk, pid, manifest):
        return _make_polygon_with_edge("PROD-001", "products")

    def build_solid(self, bk, pid, manifest, filters=None, timestamp=None):
        base = _make_polygon_with_edge("PROD-001", "products")
        all_slices = [
            SolidSlice(
                slice_index=i,
                timestamp=ts,
                deformation_type="internal",
                delta_snapshot=np.zeros(2, dtype=np.float32),
                delta_norm_snapshot=0.0,
                pattern_ver=1,
                changed_property=None,
                changed_line_id=None,
                added_edge=None,
            )
            for i, ts in enumerate([self._ts_early, self._ts_mid, self._ts_late])
        ]
        if timestamp is not None:
            all_slices = [s for s in all_slices if s.timestamp <= timestamp]
        return Solid(primary_key=bk, pattern_id=pid, base_polygon=base, slices=all_slices)


def _make_pi3_nav() -> GDSNavigator:
    return GDSNavigator(
        engine=_SlicedEngine(),
        storage=_MockStorage(),
        manifest=_make_manifest(),
        contract=Contract("m-001", []),
    )


def test_pi3_no_timestamp_returns_full_history():
    nav = _make_pi3_nav()
    nav.π3_dive_solid("CUST-001", "customer_pattern")
    assert isinstance(nav.position, Solid)
    assert len(nav.position.slices) == 3


def test_pi3_timestamp_before_first_slice_returns_empty():
    nav = _make_pi3_nav()
    before_all = datetime(2021, 6, 1, tzinfo=UTC)
    nav.π3_dive_solid("CUST-001", "customer_pattern", timestamp=before_all)
    assert isinstance(nav.position, Solid)
    assert len(nav.position.slices) == 0


def test_pi3_timestamp_mid_history_returns_truncated():
    nav = _make_pi3_nav()
    ts_mid = datetime(2023, 1, 1, tzinfo=UTC)
    nav.π3_dive_solid("CUST-001", "customer_pattern", timestamp=ts_mid)
    assert isinstance(nav.position, Solid)
    assert len(nav.position.slices) == 2
    assert all(s.timestamp <= ts_mid for s in nav.position.slices)


# ---------- GDSError hierarchy tests ----------


class TestErrorHierarchy:
    """Verify that the error class hierarchy is wired correctly."""

    def test_gds_error_is_base_exception(self):
        assert issubclass(GDSError, Exception)

    def test_navigation_error_is_subclass_of_gds_error(self):
        assert issubclass(GDSNavigationError, GDSError)

    def test_no_alive_edge_error_is_subclass_of_navigation_error(self):
        assert issubclass(GDSNoAliveEdgeError, GDSNavigationError)

    def test_position_error_is_subclass_of_navigation_error(self):
        assert issubclass(GDSPositionError, GDSNavigationError)

    def test_entity_not_found_error_is_subclass_of_navigation_error(self):
        assert issubclass(GDSEntityNotFoundError, GDSNavigationError)

    def test_storage_error_is_subclass_of_gds_error(self):
        assert issubclass(GDSStorageError, GDSError)

    def test_missing_file_error_is_subclass_of_storage_error(self):
        assert issubclass(GDSMissingFileError, GDSStorageError)

    def test_corrupted_file_error_is_subclass_of_storage_error(self):
        assert issubclass(GDSCorruptedFileError, GDSStorageError)

    def test_version_error_is_subclass_of_gds_error(self):
        assert issubclass(GDSVersionError, GDSError)

    def test_all_nav_errors_catchable_as_gds_error(self):
        """Every navigation-specific error must be catchable via `except GDSError`."""
        for cls in (GDSNoAliveEdgeError, GDSPositionError, GDSEntityNotFoundError):
            with pytest.raises(GDSError):
                raise cls("test")


class TestSpecificErrorsRaised:
    """Verify that navigator methods raise the *specific* error subclass."""

    def _make_nav(self):
        return GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorage(),
            manifest=_make_manifest(),
            contract=Contract("m-001", []),
        )

    def test_goto_raises_entity_not_found(self):
        nav = self._make_nav()
        with pytest.raises(GDSEntityNotFoundError, match="NONEXISTENT"):
            nav.goto("NONEXISTENT", "customers")

    def test_pi1_raises_entity_not_found_when_key_not_in_line(self):
        nav = self._make_nav()
        nav.goto("CUST-001", "customers")
        # CUST-001 is not in 'products' line
        with pytest.raises(GDSEntityNotFoundError):
            nav.π1_walk_line("products")

    def test_pi1_raises_position_error_when_not_a_point(self):
        nav = self._make_nav()
        # position is None initially — not a Point
        with pytest.raises(GDSPositionError):
            nav.π1_walk_line("customers")

    def test_pi2_raises_no_alive_edge_error(self):
        nav = self._make_nav()
        nav.goto("CUST-001", "customers")
        dead_polygon = Polygon(
            primary_key="SALE-001",
            pattern_id="sale_pattern",
            pattern_ver=1,
            pattern_type="anchor",
            scale=1,
            delta=np.zeros(2),
            delta_norm=0.0,
            is_anomaly=False,
            edges=[
                Edge(line_id="products", point_key="PROD-001", status="dead", direction="in"),
            ],
            last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
            updated_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(GDSNoAliveEdgeError, match="No alive edge"):
            nav.π2_jump_polygon(dead_polygon, target_line_id="products")

    def test_current_polygon_raises_position_error(self):
        nav = self._make_nav()
        # position is None — not a Point
        with pytest.raises(GDSPositionError, match="requires position to be a Point"):
            nav.current_polygon("customer_pattern")

    def test_current_solid_raises_position_error(self):
        nav = self._make_nav()
        with pytest.raises(GDSPositionError, match="requires position to be a Point"):
            nav.current_solid("customer_pattern")

    def test_pi1_boundary_raises_navigation_error(self):
        """Walking past the end of a line still raises GDSNavigationError."""
        nav = self._make_nav()
        nav.goto("CUST-002", "customers")
        with pytest.raises(GDSNavigationError):
            nav.π1_walk_line("customers", direction="+")


# ---------- event_polygons_for tests ----------


class TestEventPolygonsFor:
    """Tests for GDSNavigator.event_polygons_for helper method."""

    def _make_nav(self, storage=None):
        return GDSNavigator(
            engine=_MockEngine(),
            storage=storage or _MockStorageWithGeometry(),
            manifest=Manifest(
                manifest_id="m-001",
                agent_id="agent-001",
                snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
                status="active",
                line_versions={"customers": 1, "products": 1},
                pattern_versions={"sale_pattern": 1, "customer_pattern": 1},
            ),
            contract=Contract("m-001", ["sale_pattern"]),
        )

    def test_returns_polygons_referencing_entity(self):
        """event_polygons_for returns polygons whose edges contain the entity_key."""
        nav = self._make_nav()
        result = nav.event_polygons_for("CUST-001", "sale_pattern")
        assert len(result) == 2
        keys = {p.primary_key for p in result}
        assert keys == {"SALE-001", "SALE-003"}
        assert all(p.pattern_id == "sale_pattern" for p in result)
        edge_keys = [e.point_key for r in result for e in r.edges]
        assert "CUST-001" in edge_keys

    def test_returns_empty_list_when_no_match(self):
        """event_polygons_for returns [] when no polygons reference the entity."""
        nav = self._make_nav()
        result = nav.event_polygons_for("NONEXISTENT", "sale_pattern")
        assert result == []

    def test_raises_for_pattern_not_in_manifest(self):
        """event_polygons_for raises GDSNavigationError when pattern absent from manifest."""
        nav = self._make_nav()
        with pytest.raises(GDSNavigationError, match="No geometry version"):
            nav.event_polygons_for("CUST-001", "unknown_pattern")

    def test_polygon_fields_populated_correctly(self):
        """Verify all polygon fields are populated from the geometry row."""
        nav = self._make_nav()
        result = nav.event_polygons_for("CUST-001", "sale_pattern")
        bk_to_poly = {p.primary_key: p for p in result}
        poly = bk_to_poly["SALE-001"]
        assert poly.pattern_ver == 1
        assert poly.pattern_type == "event"

    def test_geometry_filters_anomaly_true(self):
        """geometry_filters={"is_anomaly": True} returns only anomalous polygons."""
        nav = self._make_nav()
        result = nav.event_polygons_for(
            "CUST-001", "sale_pattern", geometry_filters={"is_anomaly": True}
        )
        assert len(result) == 1
        assert result[0].primary_key == "SALE-003"
        assert result[0].is_anomaly is True

    def test_geometry_filters_anomaly_false(self):
        """geometry_filters={"is_anomaly": False} returns only non-anomalous polygons."""
        nav = self._make_nav()
        result = nav.event_polygons_for(
            "CUST-001", "sale_pattern", geometry_filters={"is_anomaly": False}
        )
        assert len(result) == 1
        assert result[0].primary_key == "SALE-001"
        assert result[0].is_anomaly is False

    def test_geometry_filters_unknown_key_raises(self):
        """geometry_filters with unknown key raises ValueError."""
        nav = self._make_nav()
        with pytest.raises(ValueError, match="Unknown geometry_filters"):
            nav.event_polygons_for("CUST-001", "sale_pattern", geometry_filters={"unknown_col": 1})

    def test_limit_offset_pagination_at_arrow_level(self):
        """limit/offset slices Arrow table before Polygon construction."""
        nav = self._make_nav()
        # CUST-001 matches SALE-001 and SALE-003 (2 rows)
        result_all = nav.event_polygons_for("CUST-001", "sale_pattern")
        assert len(result_all) == 2

        result_limited = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            limit=1,
            offset=0,
        )
        assert len(result_limited) == 1
        assert nav._last_total_post_geometry_filter == 2

        result_offset = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            limit=1,
            offset=1,
        )
        assert len(result_offset) == 1
        assert nav._last_total_post_geometry_filter == 2

        # Different polygon returned at different offset
        assert result_limited[0].primary_key != result_offset[0].primary_key

    def test_offset_beyond_total_returns_empty(self):
        """offset beyond total rows returns empty list."""
        nav = self._make_nav()
        result = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            limit=10,
            offset=100,
        )
        assert result == []
        assert nav._last_total_post_geometry_filter == 2

    def test_edge_filters_narrow_by_line_and_key(self):
        """filters=[{line, key}] returns only polygons with that edge."""
        # Mock returns SALE-001 (CUST-001 + PROD-001), SALE-003 (CUST-001 + PROD-003)
        # for entity_key=CUST-001. Filter for PROD-001 should return only SALE-001.
        nav = self._make_nav()
        result = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            filters=[{"line": "products", "key": "PROD-001"}],
        )
        assert len(result) == 1
        assert result[0].primary_key == "SALE-001"

    def test_edge_filters_multi_filter_all_must_match(self):
        """All filters must match (AND semantics)."""
        nav = self._make_nav()
        result = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            filters=[
                {"line": "customers", "key": "CUST-001"},
                {"line": "products", "key": "PROD-999"},  # no polygon has this
            ],
        )
        assert result == []

    def test_edge_filters_empty_list_returns_all(self):
        """Empty filters list is a no-op — returns all entity polygons."""
        nav = self._make_nav()
        result = nav.event_polygons_for("CUST-001", "sale_pattern", filters=[])
        assert len(result) == 2  # SALE-001 and SALE-003


class TestEventPolygonsForEntityKeysIndex:
    """Entity keys LABEL_LIST index path in event_polygons_for."""

    def _make_storage_with_index(self, entity_pks, filter_pks):
        """Mock storage that simulates entity_keys index availability."""

        class _StorageWithIndex(_MockStorageWithGeometry):
            def resolve_primary_keys_by_edge(self_, pattern_id, version, line_id, point_key):
                if line_id is None:
                    # entity_key lookup
                    return entity_pks
                # filter lookup
                return filter_pks

            def read_geometry(
                self_,
                pattern_id,
                version,
                primary_key=None,
                filters=None,
                point_keys=None,
                columns=None,
                filter=None,
            ):
                if filter is not None and "primary_key IN" in filter:
                    # Secondary index path: filter by PK set
                    # Parse keys from "primary_key IN ('K1', 'K2')" string
                    import re

                    found = re.findall(r"'([^']+)'", filter)
                    pk_set = set(found)
                    table = super().read_geometry(
                        pattern_id,
                        version,
                        primary_key=None,
                        filters=None,
                        point_keys=None,
                        columns=None,
                        filter=None,
                    )
                    import pyarrow.compute as _pc

                    mask = _pc.is_in(table["primary_key"], pa.array(list(pk_set)))
                    return table.filter(mask)
                if filter is not None and "__no_match__" in filter:
                    table = super().read_geometry(pattern_id, version)
                    return table.slice(0, 0)
                return super().read_geometry(
                    pattern_id,
                    version,
                    primary_key=primary_key,
                    filters=filters,
                    point_keys=point_keys,
                    columns=columns,
                    filter=filter,
                )

        return _StorageWithIndex()

    def _make_nav(self, storage):
        return GDSNavigator(
            engine=_MockEngine(),
            storage=storage,
            manifest=Manifest(
                manifest_id="m-001",
                agent_id="agent-001",
                snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
                status="active",
                line_versions={"customers": 1, "products": 1},
                pattern_versions={"sale_pattern": 1, "customer_pattern": 1},
            ),
            contract=Contract("m-001", ["sale_pattern"]),
        )

    def test_entity_keys_index_path_returns_correct_results(self):
        """When entity_keys index returns PKs, only those polygons are returned."""
        # CUST-001 appears in SALE-001 and SALE-003, PROD-001 only in SALE-001
        storage = self._make_storage_with_index(
            entity_pks=["SALE-001", "SALE-003"],
            filter_pks=["SALE-001"],
        )
        nav = self._make_nav(storage)
        result = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            filters=[{"line": "products", "key": "PROD-001"}],
        )
        assert len(result) == 1
        assert result[0].primary_key == "SALE-001"

    def test_entity_keys_index_intersection_empty_returns_no_rows(self):
        """When intersection of index sets is empty, no rows are returned."""
        # entity_pks and filter_pks have no overlap
        storage = self._make_storage_with_index(
            entity_pks=["SALE-001"],
            filter_pks=["SALE-002"],  # SALE-002 doesn't match entity_key
        )
        nav = self._make_nav(storage)
        result = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            filters=[{"line": "products", "key": "PROD-999"}],
        )
        assert result == []

    def test_fallback_when_index_returns_none(self):
        """When resolve_primary_keys_by_edge returns None, falls back to Python filter."""
        # Use default mock which returns None -> fallback path
        nav_default = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageWithGeometry(),
            manifest=Manifest(
                manifest_id="m-001",
                agent_id="agent-001",
                snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
                status="active",
                line_versions={"customers": 1, "products": 1},
                pattern_versions={"sale_pattern": 1, "customer_pattern": 1},
            ),
            contract=Contract("m-001", ["sale_pattern"]),
        )
        result = nav_default.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            filters=[{"line": "products", "key": "PROD-001"}],
        )
        assert len(result) == 1
        assert result[0].primary_key == "SALE-001"


class TestEventPolygonsForDeltaDim:
    """delta_dim geometry filter in event_polygons_for."""

    def _make_storage_with_sphere(self):
        """Mock storage returning geometry with delta values + a sphere with 2 relations."""
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Pattern, Sphere

        # Relations: index 0 = customers, index 1 = products
        rel_customers = MagicMock()
        rel_customers.line_id = "customers"
        rel_customers.display_name = None
        rel_products = MagicMock()
        rel_products.line_id = "products"
        rel_products.display_name = None

        pattern = MagicMock(spec=Pattern)
        pattern.relations = [rel_customers, rel_products]
        pattern.pattern_type = "event"

        def _dim_index(name):
            if name == "customers":
                return 0
            if name == "products":
                return 1
            raise ValueError(
                f"Dimension '{name}' not in pattern relations "
                f"of 'sale_pattern'. Available: ['customers', 'products']"
            )

        pattern.dim_index = _dim_index

        sphere = MagicMock(spec=Sphere)
        sphere.patterns = {"sale_pattern": pattern}

        # Geometry:
        # SALE-001 → CUST-001, delta=[0.1, 0.2]
        # SALE-003 → CUST-001, delta=[0.5, 0.6]
        geo = pa.table(
            {
                "primary_key": ["SALE-001", "SALE-003"],
                "scale": [1, 1],
                "delta": [[0.1, 0.2], [0.5, 0.6]],
                "delta_norm": [0.22, 0.78],
                "delta_rank_pct": [20.0, 80.0],
                "is_anomaly": [False, True],
                "delta_dim_0": pa.array([0.1, 0.5], type=pa.float32()),
                "delta_dim_1": pa.array([0.2, 0.6], type=pa.float32()),
                "edges": pa.array(
                    [
                        [
                            {
                                "line_id": "customers",
                                "point_key": "CUST-001",
                                "status": "alive",
                                "direction": "in",
                            }
                        ],
                        [
                            {
                                "line_id": "customers",
                                "point_key": "CUST-001",
                                "status": "alive",
                                "direction": "in",
                            }
                        ],
                    ],
                    type=pa.list_(_EDGE_STRUCT),
                ),
                "last_refresh_at": [None, None],
                "updated_at": [None, None],
            }
        )

        storage = MagicMock()
        storage.read_geometry.return_value = geo
        storage.read_sphere.return_value = sphere
        return storage

    def _make_nav(self, storage=None):
        return GDSNavigator(
            engine=_MockEngine(),
            storage=storage or self._make_storage_with_sphere(),
            manifest=Manifest(
                manifest_id="m-001",
                agent_id="agent-001",
                snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
                status="active",
                line_versions={"customers": 1, "products": 1},
                pattern_versions={"sale_pattern": 1},
            ),
            contract=Contract("m-001", ["sale_pattern"]),
        )

    def test_delta_dim_gt_filters_correctly(self):
        """delta_dim customers>0.3 keeps only SALE-003 (delta[0]=0.5)."""
        nav = self._make_nav()
        result = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            geometry_filters={"delta_dim": {"customers": {"gt": 0.3}}},
        )
        assert len(result) == 1
        assert result[0].primary_key == "SALE-003"

    def test_delta_dim_lt_filters_correctly(self):
        """delta_dim customers<0.3 keeps only SALE-001 (delta[0]=0.1)."""
        nav = self._make_nav()
        result = nav.event_polygons_for(
            "CUST-001",
            "sale_pattern",
            geometry_filters={"delta_dim": {"customers": {"lt": 0.3}}},
        )
        assert len(result) == 1
        assert result[0].primary_key == "SALE-001"

    def test_delta_dim_unknown_dimension_raises(self):
        """Unknown dimension name raises ValueError."""
        nav = self._make_nav()
        with pytest.raises(ValueError, match="Dimension 'nonexistent' not in pattern"):
            nav.event_polygons_for(
                "CUST-001",
                "sale_pattern",
                geometry_filters={"delta_dim": {"nonexistent": {"gt": 0.0}}},
            )


# ---------- π5 radius fix + anomaly_summary ----------


class TestPi5RadiusFix:
    def test_radius_none_returns_all(self, sphere_path):
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )
        results, _, _, _ = nav.π5_attract_anomaly("customer_pattern", radius=None, top_n=100)
        assert len(results) == 1  # only the anomaly entity
        assert results[0].is_anomaly is True

    def test_radius_zero_disables_threshold(self, sphere_path):
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )
        # radius=0 → threshold=0*theta_norm=0 → all entities pass delta_norm > 0
        results, _, _, _ = nav.π5_attract_anomaly("customer_pattern", radius=0.0, top_n=100)
        assert len(results) == 2  # both CUST-001 (anomaly) and CUST-002 (normal) returned

    def test_π5_returns_only_anomaly_rows(self, sphere_path):
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )
        # theta for fixture customer_pattern: [1.5, 1.5] → theta_norm ≈ 2.12
        # π5 must only return rows where delta_norm > theta_norm
        results, _, _, _ = nav.π5_attract_anomaly("customer_pattern", radius=None, top_n=100)
        assert all(p.is_anomaly for p in results)

    def test_radius_filters_upper_bound(self):
        """π5 must exclude entities whose recomputed delta_norm is below threshold
        even if the stored delta_norm passed the Arrow pre-filter."""
        nav = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageStaleNorm(),
            manifest=Manifest(
                manifest_id="m-1",
                agent_id="a-1",
                snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
                status="active",
                line_versions={},
                pattern_versions={"test_pattern": 1},
            ),
            contract=Contract("m-1", ["test_pattern"]),
        )
        results, _, _, _ = nav.π5_attract_anomaly("test_pattern", radius=1.0, top_n=10)
        # E-001 stored 3.5 but actual norm([1.0,1.0])=√2≈1.414 < 3.0 → excluded
        # E-002 stored 4.0 and actual norm([3.0,3.0])=3√2≈4.243 > 3.0 → included
        assert len(results) == 1
        assert results[0].primary_key == "E-002"
        assert abs(results[0].delta_norm - float(np.linalg.norm([3.0, 3.0]))) < 1e-4


class TestAnomalySummary:
    def test_anomaly_summary(self, sphere_path):
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )
        summary = nav.anomaly_summary("customer_pattern")
        assert summary["total_entities"] == 2
        assert "anomaly_rate" in summary
        assert "clusters" in summary
        assert "delta_norm_percentiles" in summary
        assert "delta_norm_histogram" not in summary
        assert "delta_norm_distribution" in summary
        dist = summary["delta_norm_distribution"]
        expected_labels = {"0–0.25θ", "0.25θ–0.5θ", "0.5θ–0.75θ", "0.75θ–θ", "θ–1.5θ", "1.5θ+"}
        assert set(dist.keys()) == expected_labels
        assert sum(dist.values()) == summary["total_entities"]

    def test_anomaly_summary_returns_percentiles_not_histogram(self, sphere_path):
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )
        summary = nav.anomaly_summary("customer_pattern")
        assert "delta_norm_percentiles" in summary
        assert "delta_norm_histogram" not in summary
        assert "delta_norm_distribution" in summary
        p = summary["delta_norm_percentiles"]
        assert set(p.keys()) == {"p50", "p75", "p90", "p95", "p99", "max"}
        assert p["p50"] <= p["p75"] <= p["p90"] <= p["p95"] <= p["p99"] <= p["max"]
        assert all(v >= 0.0 for v in p.values())

    def _make_many_cluster_nav(self, n_unique=40):
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.model.sphere import Pattern, RelationDef
        from hypertopos.storage.cache import GDSCache

        pattern = Pattern(
            pattern_id="p1",
            entity_type="items",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="cats", direction="out", required=False),
                RelationDef(line_id="shops", direction="out", required=False),
            ],
            mu=np.zeros(2, dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.array([1.0, 1.0], dtype=np.float32),
            population_size=n_unique,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        deltas = [[float(i) * 0.5, 2.0] for i in range(n_unique)]
        norms = [float(np.linalg.norm(d)) for d in deltas]
        geo = pa.table(
            {
                "primary_key": [f"I-{i:03d}" for i in range(n_unique)],
                "scale": [1] * n_unique,
                "delta": deltas,
                "delta_norm": norms,
                "is_anomaly": [True] * n_unique,
                "delta_rank_pct": pa.array([90.0] * n_unique, type=pa.float64()),
                "edges": pa.array([[]] * n_unique, type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [dt] * n_unique,
                "updated_at": [dt] * n_unique,
            }
        )
        storage = _MockStorageWithPatterns({"p1": pattern}, geometry_table=geo)
        return GDSNavigator(
            engine=GDSEngine(storage=storage, cache=GDSCache()),
            storage=storage,
            manifest=Manifest(
                manifest_id="m",
                agent_id="a",
                snapshot_time=dt,
                status="active",
                line_versions={"items": 1},
                pattern_versions={"p1": 1},
            ),
            contract=Contract("m", ["p1"]),
        )

    def test_anomaly_summary_cluster_cap_default(self):
        """Default max_clusters=20 caps output with metadata."""
        nav = self._make_many_cluster_nav(40)
        summary = nav.anomaly_summary("p1")
        assert summary["total_clusters"] == 40
        assert summary["clusters_truncated"] is True
        assert len(summary["clusters"]) == 20

    def test_anomaly_summary_cluster_cap_custom(self):
        """Agent can override max_clusters."""
        nav = self._make_many_cluster_nav(40)
        summary = nav.anomaly_summary("p1", max_clusters=5)
        assert summary["total_clusters"] == 40
        assert summary["clusters_truncated"] is True
        assert len(summary["clusters"]) == 5

    def test_anomaly_summary_cluster_cap_unlimited(self):
        """max_clusters=0 returns all clusters."""
        nav = self._make_many_cluster_nav(40)
        summary = nav.anomaly_summary("p1", max_clusters=0)
        assert summary["total_clusters"] == 40
        assert summary["clusters_truncated"] is False
        assert len(summary["clusters"]) == 40


# ---------- find_similar_entities ----------


def test_find_similar_entities():
    """Navigator.find_similar_entities delegates to engine.find_nearest."""

    ref_poly = _make_polygon_with_edge("PROD-001", "products")
    ref_poly.delta = np.array([0.1, 0.2], dtype=np.float32)
    ref_poly.delta_norm = float(np.linalg.norm(ref_poly.delta))

    class _SimEngine:
        def build_polygon(self, bk, pid, manifest):
            return ref_poly

        def build_solid(self, *a, **kw):
            pass

        def find_nearest(
            self, ref_delta, pattern_id, version, top_n=5, exclude_keys=None, filter_expr=None
        ):
            return [("CUST-002", 0.15), ("CUST-003", 0.42)]

    manifest = Manifest(
        manifest_id="test",
        agent_id="a",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "products": 1},
        pattern_versions={"customer_pattern": 1},
    )
    contract = Contract("test", ["customer_pattern"])
    nav = GDSNavigator(
        engine=_SimEngine(),
        storage=_MockStorageWithDelta(),
        manifest=manifest,
        contract=contract,
    )

    results = nav.find_similar_entities("CUST-001", "customer_pattern", top_n=2)

    assert len(results) == 2
    assert results[0][0] == "CUST-002"
    assert results[0][1] == pytest.approx(0.15)
    assert results[1][0] == "CUST-003"


def test_find_similar_entities_event_pattern():
    """find_similar_entities works for event patterns."""

    ref_poly = _make_polygon_with_edge("PROD-001", "products")
    ref_poly.pattern_id = "gl_pattern"
    ref_poly.pattern_type = "event"
    ref_poly.delta = np.array([0.5, -0.3], dtype=np.float32)
    ref_poly.delta_norm = float(np.linalg.norm(ref_poly.delta))

    class _EventEngine:
        def build_polygon(self, bk, pid, manifest):
            return ref_poly

        def build_solid(self, *a, **kw):
            pass

        def find_nearest(
            self, ref_delta, pattern_id, version, top_n=5, exclude_keys=None, filter_expr=None
        ):
            # Confirm event pattern_id is passed through unchanged
            assert pattern_id == "gl_pattern"
            return [("GL-002", 0.05), ("GL-003", 0.20)]

    manifest = Manifest(
        manifest_id="test",
        agent_id="a",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"gl_entries": 1},
        pattern_versions={"gl_pattern": 1},
    )
    contract = Contract("test", ["gl_pattern"])
    nav = GDSNavigator(
        engine=_EventEngine(),
        storage=_MockStorageWithDelta(),
        manifest=manifest,
        contract=contract,
    )

    results = nav.find_similar_entities("GL-001", "gl_pattern", top_n=2)

    assert len(results) == 2
    assert results[0][0] == "GL-002"
    assert results[0][1] == pytest.approx(0.05)


def test_find_similar_entities_passes_filter_expr():
    """find_similar_entities forwards filter_expr to engine.find_nearest."""

    ref_poly = _make_polygon_with_edge("PROD-001", "products")
    ref_poly.delta = np.array([0.1, 0.2], dtype=np.float32)
    ref_poly.delta_norm = float(np.linalg.norm(ref_poly.delta))

    captured: dict = {}

    class _FilterEngine:
        def build_polygon(self, bk, pid, manifest):
            return ref_poly

        def build_solid(self, *a, **kw):
            pass

        def find_nearest(
            self, ref_delta, pattern_id, version, top_n=5, exclude_keys=None, filter_expr=None
        ):
            captured["filter_expr"] = filter_expr
            return [("CUST-002", 0.10)]

    manifest = Manifest(
        manifest_id="test",
        agent_id="a",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "products": 1},
        pattern_versions={"customer_pattern": 1},
    )
    contract = Contract("test", ["customer_pattern"])
    nav = GDSNavigator(
        engine=_FilterEngine(),
        storage=_MockStorageWithDelta(),
        manifest=manifest,
        contract=contract,
    )

    results = nav.find_similar_entities(
        "CUST-001", "customer_pattern", top_n=1, filter_expr="is_anomaly = true"
    )

    assert captured["filter_expr"] == "is_anomaly = true"
    assert len(results) == 1
    assert results[0][0] == "CUST-002"


def test_get_entity_geometry_meta_returns_stored_values():
    """get_entity_geometry_meta reads delta_norm, is_anomaly, delta_rank_pct from storage."""
    from unittest.mock import MagicMock

    manifest = Manifest(
        manifest_id="test",
        agent_id="a",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1},
        pattern_versions={"customer_pattern": 1},
    )
    contract = Contract("test", ["customer_pattern"])
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=_MockStorageWithDelta(),
        manifest=manifest,
        contract=contract,
    )

    meta = nav.get_entity_geometry_meta("CUST-001", "customer_pattern")

    assert meta["delta_norm"] == pytest.approx(0.224)
    assert meta["is_anomaly"] is False
    assert meta["delta_rank_pct"] == pytest.approx(25.0)


def test_get_entity_geometry_meta_raises_key_error_when_not_found():
    """get_entity_geometry_meta raises KeyError for unknown entity."""
    from unittest.mock import MagicMock

    class _EmptyStorage(_MockStorage):
        def read_geometry(self, *a, **kw):
            return pa.table(
                {
                    "delta_norm": pa.array([], type=pa.float32()),
                    "is_anomaly": pa.array([], type=pa.bool_()),
                    "delta_rank_pct": pa.array([], type=pa.float32()),
                }
            )

    manifest = Manifest(
        manifest_id="test",
        agent_id="a",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1},
        pattern_versions={"customer_pattern": 1},
    )
    contract = Contract("test", ["customer_pattern"])
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=_EmptyStorage(),
        manifest=manifest,
        contract=contract,
    )

    with pytest.raises(KeyError, match="UNKNOWN"):
        nav.get_entity_geometry_meta("UNKNOWN", "customer_pattern")


# --- π6 attract_boundary ---


class TestPi6AttractBoundary:
    """Tests for π6_attract_boundary navigation primitive."""

    def _make_nav_with_geometry(self):
        """Build a navigator with mocked storage containing 2 geometry rows."""
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import (
            Alias,
            AliasFilter,
            CuttingPlane,
            DerivedPattern,
        )

        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.5)
        alias = Alias(
            alias_id="seg",
            base_pattern_id="p",
            filter=AliasFilter(include_relations=[], cutting_plane=cp),
            derived_pattern=DerivedPattern(
                mu=np.array([0.5, 0.5], dtype=np.float32),
                sigma_diag=np.array([0.1, 0.1], dtype=np.float32),
                theta=np.array([0.3, 0.3], dtype=np.float32),
                population_size=10,
                computed_at=datetime.now(UTC),
            ),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.aliases = {"seg": alias}
        sphere.patterns = {"p": MagicMock(version=1)}

        # Two geometry rows: IN-001 inside (delta[0]=0.8>=0.5), OUT-001 outside (0.2<0.5)
        now_str = "2025-01-01T00:00:00"
        table = pa.table(
            {
                "primary_key": ["IN-001", "OUT-001"],
                "scale": [1, 1],
                "delta": [[0.8, 0.0], [0.2, 0.0]],
                "delta_norm": [0.8, 0.2],
                "delta_rank_pct": [80.0, 20.0],
                "is_anomaly": [False, False],
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [now_str, now_str],
                "updated_at": [now_str, now_str],
            }
        )

        storage = MagicMock()
        storage.read_sphere.return_value = sphere

        def _read_geo(*args, point_keys=None, columns=None, filter=None, **kw):
            import re as _re

            import pyarrow.compute as pc

            t = table
            if point_keys is not None:
                mask = pc.is_in(t["primary_key"], pa.array(point_keys))
                t = t.filter(mask)
            if filter is not None and "primary_key IN" in str(filter):
                keys = _re.findall(r"'([^']*)'", str(filter))
                mask = pc.is_in(t["primary_key"], pa.array(keys))
                t = t.filter(mask)
            if columns is not None:
                available = set(t.schema.names)
                columns = [c for c in columns if c in available]
                if columns:
                    t = t.select(columns)
            return t

        storage.read_geometry.side_effect = _read_geo

        manifest = MagicMock()
        manifest.pattern_version.return_value = 1

        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )
        return nav

    def test_direction_in(self):
        nav = self._make_nav_with_geometry()
        results = nav.π6_attract_boundary("seg", "p", direction="in", top_n=10)
        assert len(results) == 1
        assert results[0][0].primary_key == "IN-001"
        assert results[0][1] > 0

    def test_direction_out(self):
        nav = self._make_nav_with_geometry()
        results = nav.π6_attract_boundary("seg", "p", direction="out", top_n=10)
        assert len(results) == 1
        assert results[0][0].primary_key == "OUT-001"
        assert results[0][1] < 0

    def test_direction_both_sorted_by_proximity(self):
        nav = self._make_nav_with_geometry()
        results = nav.π6_attract_boundary("seg", "p", direction="both", top_n=10)
        assert len(results) == 2
        assert abs(results[0][1]) <= abs(results[1][1])

    def test_signed_distances_not_flat(self):
        """Returned entities must have varying signed_distance values."""
        nav = self._make_nav_with_geometry()
        results = nav.π6_attract_boundary("seg", "p", direction="both", top_n=10)
        distances = [sd for _, sd in results]
        # With varied delta vectors (0.8 vs 0.2), distances must not all be 0.0
        assert not all(d == 0.0 for d in distances), (
            f"All signed_distances are 0.0 — score_lookup key mismatch. "
            f"Got: {distances}"
        )

    def test_missing_alias_raises(self):
        from unittest.mock import MagicMock

        sphere = MagicMock()
        sphere.aliases = {}
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=MagicMock(),
            contract=MagicMock(),
        )
        with pytest.raises(GDSNavigationError, match="not found"):
            nav.π6_attract_boundary("nonexistent", "p")

    def test_no_cutting_plane_raises(self):
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import (
            Alias,
            AliasFilter,
            DerivedPattern,
        )

        alias = Alias(
            alias_id="no_cp",
            base_pattern_id="p",
            filter=AliasFilter(include_relations=[]),
            derived_pattern=DerivedPattern(
                mu=np.array([0.5], dtype=np.float32),
                sigma_diag=np.array([0.1], dtype=np.float32),
                theta=np.array([0.3], dtype=np.float32),
                population_size=10,
                computed_at=datetime.now(UTC),
            ),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.aliases = {"no_cp": alias}
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=MagicMock(),
            contract=MagicMock(),
        )
        with pytest.raises(GDSNavigationError, match="cutting_plane"):
            nav.π6_attract_boundary("no_cp", "p")

    def test_signed_distances_batch(self):
        """signed_distances_batch(matrix) must equal [signed_distance(row) for row in matrix]."""
        from hypertopos.model.sphere import CuttingPlane

        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.5)
        deltas = np.array([[0.8, 0.0], [0.2, 0.0], [0.5, 0.0]], dtype=np.float32)
        batch = cp.signed_distances_batch(deltas)
        expected = np.array([cp.signed_distance(d) for d in deltas], dtype=np.float32)
        np.testing.assert_allclose(batch, expected, rtol=1e-5)

    def test_vectorized_returns_same_results_as_scalar(self):
        """Vectorized path must return identical results to the old scalar loop on 100 rows."""
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Alias, AliasFilter, CuttingPlane, DerivedPattern

        rng = np.random.default_rng(42)
        n = 100
        cp = CuttingPlane(normal=[1.0, 0.5], bias=0.3)
        alias = Alias(
            alias_id="seg100",
            base_pattern_id="p",
            filter=AliasFilter(include_relations=[], cutting_plane=cp),
            derived_pattern=DerivedPattern(
                mu=np.zeros(2, dtype=np.float32),
                sigma_diag=np.ones(2, dtype=np.float32),
                theta=np.ones(2, dtype=np.float32),
                population_size=n,
                computed_at=datetime.now(UTC),
            ),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.aliases = {"seg100": alias}
        sphere.patterns = {"p": MagicMock(version=1)}

        now_str = "2025-01-01T00:00:00"
        deltas = rng.standard_normal((n, 2)).astype(np.float32).tolist()
        table = pa.table(
            {
                "primary_key": [f"K-{i:03d}" for i in range(n)],
                "scale": [1] * n,
                "delta": deltas,
                "delta_norm": [0.5] * n,
                "delta_rank_pct": [50.0] * n,
                "is_anomaly": [False] * n,
                "edges": pa.array([[] for _ in range(n)], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [now_str] * n,
                "updated_at": [now_str] * n,
            }
        )

        storage = MagicMock()
        storage.read_sphere.return_value = sphere

        def _read_geo(*args, point_keys=None, columns=None, filter=None, **kw):
            import re as _re

            import pyarrow.compute as pc

            t = table
            if point_keys is not None:
                mask = pc.is_in(t["primary_key"], pa.array(point_keys))
                t = t.filter(mask)
            if filter is not None and "primary_key IN" in str(filter):
                keys = _re.findall(r"'([^']*)'", str(filter))
                mask = pc.is_in(t["primary_key"], pa.array(keys))
                t = t.filter(mask)
            if columns is not None:
                available = set(t.schema.names)
                columns = [c for c in columns if c in available]
                if columns:
                    t = t.select(columns)
            return t

        storage.read_geometry.side_effect = _read_geo
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=MagicMock(), storage=storage, manifest=manifest, contract=MagicMock()
        )

        results = nav.π6_attract_boundary("seg100", "p", direction="both", top_n=5)
        assert len(results) == 5
        # Results must be sorted by |signed_dist| ascending
        abs_dists = [abs(r[1]) for r in results]
        assert abs_dists == sorted(abs_dists)
        # Top result is closest to boundary
        all_scores = [cp.signed_distance(np.array(d, dtype=np.float32)) for d in deltas]
        expected_top_key = min(range(n), key=lambda i: abs(all_scores[i]))
        assert results[0][0].primary_key == f"K-{expected_top_key:03d}"

    def test_signed_distances_batch_zero_norm_raises(self):
        from hypertopos.model.sphere import CuttingPlane

        cp = CuttingPlane(normal=[0.0, 0.0], bias=0.0)
        deltas = np.array([[1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="zero norm"):
            cp.signed_distances_batch(deltas)

    def test_empty_geometry_returns_empty(self):
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Alias, AliasFilter, CuttingPlane, DerivedPattern

        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.5)
        alias = Alias(
            alias_id="seg",
            base_pattern_id="p",
            filter=AliasFilter(include_relations=[], cutting_plane=cp),
            derived_pattern=DerivedPattern(
                mu=np.zeros(2, dtype=np.float32),
                sigma_diag=np.ones(2, dtype=np.float32),
                theta=np.ones(2, dtype=np.float32),
                population_size=0,
                computed_at=datetime.now(UTC),
            ),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.aliases = {"seg": alias}
        sphere.patterns = {"p": MagicMock(version=1)}
        empty_table = pa.table(
            {
                "primary_key": pa.array([], type=pa.string()),
                "scale": pa.array([], type=pa.int64()),
                "delta": pa.array([], type=pa.list_(pa.float32())),
                "delta_norm": pa.array([], type=pa.float64()),
                "is_anomaly": pa.array([], type=pa.bool_()),
                "edges": pa.array([], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": pa.array([], type=pa.string()),
                "updated_at": pa.array([], type=pa.string()),
            }
        )
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        storage.read_geometry.return_value = empty_table
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=MagicMock(), storage=storage, manifest=manifest, contract=MagicMock()
        )
        results = nav.π6_attract_boundary("seg", "p", direction="both", top_n=10)
        assert results == []

    def test_all_filtered_returns_empty(self):
        """When all entities are inside but direction='out', result must be []."""
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Alias, AliasFilter, CuttingPlane, DerivedPattern

        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.5)
        alias = Alias(
            alias_id="inside_only",
            base_pattern_id="p",
            filter=AliasFilter(include_relations=[], cutting_plane=cp),
            derived_pattern=DerivedPattern(
                mu=np.zeros(2, dtype=np.float32),
                sigma_diag=np.ones(2, dtype=np.float32),
                theta=np.ones(2, dtype=np.float32),
                population_size=2,
                computed_at=datetime.now(UTC),
            ),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.aliases = {"inside_only": alias}
        sphere.patterns = {"p": MagicMock(version=1)}
        now_str = "2025-01-01T00:00:00"
        # All entities have delta[0]=0.8 >= 0.5 → all inside
        table = pa.table(
            {
                "primary_key": ["A-001", "A-002"],
                "scale": [1, 1],
                "delta": [[0.8, 0.0], [0.9, 0.0]],
                "delta_norm": [0.8, 0.9],
                "is_anomaly": [False, False],
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [now_str, now_str],
                "updated_at": [now_str, now_str],
            }
        )
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        storage.read_geometry.return_value = table
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav2 = GDSNavigator(
            engine=MagicMock(), storage=storage, manifest=manifest, contract=MagicMock()
        )
        # direction="out" but all are inside → empty
        results = nav2.π6_attract_boundary("inside_only", "p", direction="out", top_n=10)
        assert results == []

    def test_top_n_larger_than_available(self):
        """When top_n > available filtered rows, return only what's available."""
        nav = self._make_nav_with_geometry()
        # table has 2 rows total; direction="in" gives 1 row; top_n=100 should return 1 row
        results = nav.π6_attract_boundary("seg", "p", direction="in", top_n=100)
        assert len(results) == 1

    def test_signed_distances_not_flat(self):
        """Returned entities must have varying signed_distance values (str conversion)."""
        nav = self._make_nav_with_geometry()
        results = nav.π6_attract_boundary(
            alias_id="seg",
            pattern_id="p",
            direction="both",
            top_n=10,
        )
        distances = [sd for _, sd in results]
        # With varied delta vectors, distances must not all be 0.0
        assert not all(d == 0.0 for d in distances), (
            f"All signed_distances are 0.0 — score_lookup key mismatch. "
            f"Got: {distances}"
        )


# --- contrast_populations ---


def _make_contrast_nav(anomaly_flags: list[bool], deltas: list[list[float]]):
    """Build a navigator with mocked geometry and a 2-relation pattern."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef

    now_str = "2025-01-01T00:00:00"
    n = len(deltas)
    table = pa.table(
        {
            "primary_key": [f"E-{i:03d}" for i in range(n)],
            "scale": [1] * n,
            "delta": deltas,
            "delta_norm": [float(np.linalg.norm(d)) for d in deltas],
            "is_anomaly": anomaly_flags,
            "edges": pa.array([[] for _ in range(n)], type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [now_str] * n,
            "updated_at": [now_str] * n,
        }
    )
    pattern = Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=[
            RelationDef(
                line_id="products", direction="out", required=True, display_name="Products"
            ),
            RelationDef(line_id="stores", direction="in", required=False, display_name="Stores"),
        ],
        mu=np.array([0.5, 0.5], dtype=np.float32),
        sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
        theta=np.array([0.3, 0.3], dtype=np.float32),
        population_size=len(deltas),
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=None, cache=None)
    return GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())


class TestContrastPopulations:
    def test_anomaly_mode(self):
        """anomaly mode splits anomalous vs normal entities."""
        # 4 normal (low dim0), 4 anomalous (high dim0)
        deltas = [[0.1, 0.5]] * 4 + [[0.9, 0.5]] * 4
        flags = [False] * 4 + [True] * 4
        nav = _make_contrast_nav(flags, deltas)

        result = nav.contrast_populations("p", {"anomaly": True}, {"anomaly": False})

        assert len(result) == 2
        # dim0 (products) is most different; group_a=anomalous has higher mean
        by_dim = {r["dim_index"]: r for r in result}
        assert abs(by_dim[0]["diff"]) > abs(by_dim[1]["diff"])
        assert by_dim[0]["dim_label"] == "Products"
        assert by_dim[1]["dim_label"] == "Stores"

    def test_keys_mode(self):
        """keys mode selects entities by explicit business key list."""
        deltas = [[0.1, 0.5], [0.9, 0.5], [0.1, 0.5], [0.9, 0.5]]
        nav = _make_contrast_nav([False] * 4, deltas)

        # group_a = E-000, E-002 (low); group_b = E-001, E-003 (high)
        result = nav.contrast_populations(
            "p",
            {"keys": ["E-000", "E-002"]},
            {"keys": ["E-001", "E-003"]},
        )

        by_dim = {r["dim_index"]: r for r in result}
        # dim0 diff ≈ -0.8 (group_a low, group_b high)
        assert pytest.approx(by_dim[0]["diff"], abs=0.01) == -0.8

    def test_complement_mode(self):
        """group_b=None uses complement of group_a."""
        deltas = [[0.1, 0.5]] * 3 + [[0.9, 0.5]] * 3
        flags = [True] * 3 + [False] * 3
        nav = _make_contrast_nav(flags, deltas)

        result_explicit = nav.contrast_populations("p", {"anomaly": True}, {"anomaly": False})
        result_complement = nav.contrast_populations("p", {"anomaly": True})  # group_b=None

        # Results should be identical
        assert len(result_explicit) == len(result_complement)
        for a, b in zip(result_explicit, result_complement, strict=False):
            assert pytest.approx(a["effect_size"], abs=1e-5) == b["effect_size"]

    def test_alias_mode(self):
        """alias mode uses cutting plane to determine segment membership."""
        from hypertopos.model.sphere import Alias, AliasFilter, CuttingPlane, DerivedPattern

        deltas = [[0.8, 0.5]] * 3 + [[0.2, 0.5]] * 3
        nav = _make_contrast_nav([False] * 6, deltas)

        # cutting plane: delta[0] >= 0.5 → "in"
        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.5)
        alias = Alias(
            alias_id="seg",
            base_pattern_id="p",
            filter=AliasFilter(include_relations=[], cutting_plane=cp),
            derived_pattern=DerivedPattern(
                mu=np.array([0.5, 0.5], dtype=np.float32),
                sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
                theta=np.array([0.3, 0.3], dtype=np.float32),
                population_size=6,
                computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            version=1,
            status="production",
        )
        nav._storage.read_sphere.return_value.aliases = {"seg": alias}

        result = nav.contrast_populations(
            "p", {"alias": "seg", "side": "in"}, {"alias": "seg", "side": "out"}
        )

        by_dim = {r["dim_index"]: r for r in result}
        # in-segment has mean dim0 ≈ 0.8, out-segment ≈ 0.2 → diff ≈ 0.6
        assert pytest.approx(by_dim[0]["diff"], abs=0.05) == 0.6

    def test_invalid_spec_raises(self):
        """Unknown group spec key raises GDSNavigationError."""
        nav = _make_contrast_nav([False] * 2, [[0.1, 0.5], [0.9, 0.5]])

        with pytest.raises(GDSNavigationError, match="Unknown group spec"):
            nav.contrast_populations("p", {"unknown_key": True})

    def test_alias_wrong_base_pattern_raises_clear_error(self):
        """Alias based on a different pattern raises ValueError before numpy broadcast."""
        from hypertopos.model.sphere import Alias, AliasFilter, CuttingPlane, DerivedPattern

        # Pattern "p" has delta_dim=2 (products, stores).
        # Alias "wrong_seg" claims base_pattern_id="other_pattern" (delta_dim=3).
        # Calling contrast_populations with pattern_id="p" and this alias must raise
        # ValueError with a clear message — not a raw numpy shape error.
        deltas = [[0.8, 0.5]] * 3 + [[0.2, 0.5]] * 3
        nav = _make_contrast_nav([False] * 6, deltas)

        # Cutting plane has 3 weights — matches "other_pattern" (dim=3), not "p" (dim=2).
        cp = CuttingPlane(normal=[1.0, 0.0, 0.0], bias=0.5)
        wrong_alias = Alias(
            alias_id="wrong_seg",
            base_pattern_id="other_pattern",
            filter=AliasFilter(include_relations=[], cutting_plane=cp),
            derived_pattern=DerivedPattern(
                mu=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                sigma_diag=np.array([0.2, 0.2, 0.2], dtype=np.float32),
                theta=np.array([0.3, 0.3, 0.3], dtype=np.float32),
                population_size=6,
                computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            version=1,
            status="production",
        )
        nav._storage.read_sphere.return_value.aliases = {"wrong_seg": wrong_alias}

        with pytest.raises(ValueError, match="base_pattern_id"):
            nav.contrast_populations("p", {"alias": "wrong_seg", "side": "in"})

    def test_prop_columns_are_labeled_in_dim_labels(self):
        """contrast_populations labels prop_column dims by column name, not 'dim_N'."""
        from hypertopos.model.sphere import RelationDef

        # Pattern with 2 relations + 2 prop_columns → delta_dim=4
        deltas = [
            [0.5, 0.3, 1.0, 0.0],  # group A: has tax_id, missing credit_limit
            [0.5, 0.3, 0.0, 1.0],  # group B: missing tax_id, has credit_limit
        ]
        anomaly_flags = [False, False]
        nav = _make_contrast_nav(anomaly_flags, deltas)

        # Patch the pattern to have 2 relations + 2 prop_columns
        pattern = nav._storage.read_sphere.return_value.patterns["p"]
        pattern.relations = [
            RelationDef(
                line_id="products", direction="out", required=True, display_name="Products"
            ),  # noqa: E501
            RelationDef(line_id="stores", direction="in", required=False, display_name="Stores"),
        ]
        pattern.prop_columns = ["tax_id", "credit_limit"]
        pattern.mu = np.array([0.5, 0.5, 0.8, 0.9], dtype=np.float32)
        pattern.sigma_diag = np.array([0.2, 0.2, 0.3, 0.3], dtype=np.float32)
        pattern.theta = np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32)

        result = nav.contrast_populations("p", {"keys": ["E-000"]}, {"keys": ["E-001"]})

        labels = {r["dim_index"]: r["dim_label"] for r in result}
        # Prop dims must use column names, NOT "dim_2" / "dim_3"
        assert labels[2] == "tax_id", f"Expected 'tax_id', got '{labels[2]}'"  # noqa: E501
        assert labels[3] == "credit_limit", f"Expected 'credit_limit', got '{labels[3]}'"  # noqa: E501
        # Relation dims use display_name
        assert labels[0] == "Products"
        assert labels[1] == "Stores"

    def test_edge_mode(self):
        """edge mode filters by point_key in edges without extra I/O."""
        from unittest.mock import MagicMock

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.model.sphere import Pattern, RelationDef

        now_str = "2025-01-01T00:00:00"
        # E-000, E-001 linked to CC-01 (low dim0)
        # E-002, E-003 linked to CC-07 (high dim0)
        edges_pl = [
            {"line_id": "company_codes", "point_key": "CC-01", "status": "alive", "direction": "in"}
        ]
        edges_it = [
            {"line_id": "company_codes", "point_key": "CC-07", "status": "alive", "direction": "in"}
        ]
        table = pa.table(
            {
                "primary_key": ["E-000", "E-001", "E-002", "E-003"],
                "scale": [1] * 4,
                "delta": [[0.1, 0.5], [0.1, 0.5], [0.9, 0.5], [0.9, 0.5]],
                "delta_norm": [
                    float(np.linalg.norm(d)) for d in [[0.1, 0.5]] * 2 + [[0.9, 0.5]] * 2
                ],
                "is_anomaly": [False] * 4,
                "edges": pa.array(
                    [edges_pl, edges_pl, edges_it, edges_it], type=pa.list_(_EDGE_STRUCT)
                ),
                "last_refresh_at": [now_str] * 4,
                "updated_at": [now_str] * 4,
            }
        )
        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="anchor",
            relations=[
                RelationDef(
                    line_id="company_codes",
                    direction="in",
                    required=True,
                    display_name="Company code",
                ),
                RelationDef(
                    line_id="stores", direction="in", required=False, display_name="Stores"
                ),
            ],
            mu=np.array([0.5, 0.5], dtype=np.float32),
            sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
            theta=np.array([0.3, 0.3], dtype=np.float32),
            population_size=4,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        storage = MagicMock()
        storage.read_geometry.return_value = table
        storage.read_sphere.return_value = sphere
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        engine = GDSEngine(storage=None, cache=None)
        nav = GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())

        result = nav.contrast_populations(
            "p",
            {"edge": {"line_id": "company_codes", "key": "CC-01"}},
            {"edge": {"line_id": "company_codes", "key": "CC-07"}},
        )

        by_dim = {r["dim_index"]: r for r in result}
        # group_a (PL) mean dim0=0.1, group_b (IT) mean dim0=0.9 → diff ≈ -0.8
        assert pytest.approx(by_dim[0]["diff"], abs=0.01) == -0.8

    def test_edge_spec_continuous_mode_raises(self):
        """contrast_populations with edge spec on continuous-mode pattern raises GDSNavigationError.

        Continuous-mode patterns store empty string point_keys; the edge
        spec cannot uniquely identify entities → must raise GDSNavigationError.
        """
        nav = _make_continuous_mode_nav(n=6)
        with pytest.raises(GDSNavigationError, match="continuous mode"):
            nav.contrast_populations(
                "p",
                {"edge": {"line_id": "products", "key": "PROD-001"}},
            )


# --- π7 attract_hub ---


class TestPi7AttractHub:
    def _make_hub_nav(self, *, edge_max=None):
        """Build navigator with mock geometry for hub tests."""
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Pattern, RelationDef

        now_str = "2025-01-01T00:00:00"
        # 4 entities with varying connectivity:
        # K-001: 3 edges to "customers", 2 to "products" = 5 total
        # K-002: 1 edge to "customers", 1 to "products" = 2 total
        # K-003: 2 edges to "customers", 3 to "products" = 5 total
        # K-004: 0 edges = 0 total
        edges_k001 = [
            {"line_id": "customers", "point_key": "C-1", "status": "alive", "direction": "out"},
            {"line_id": "customers", "point_key": "C-2", "status": "alive", "direction": "out"},
            {"line_id": "customers", "point_key": "C-3", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "out"},
        ]
        edges_k002 = [
            {"line_id": "customers", "point_key": "C-1", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "out"},
        ]
        edges_k003 = [
            {"line_id": "customers", "point_key": "C-1", "status": "alive", "direction": "out"},
            {"line_id": "customers", "point_key": "C-4", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-3", "status": "alive", "direction": "out"},
        ]
        edges_k004 = []

        # For continuous path: shape = delta_z * sigma + mu
        # mu = [0.5, 0.5], sigma = [0.2, 0.2], edge_max = [5, 5]
        # K-001 has shape [3/5, 2/5] = [0.6, 0.4] → delta_z = [0.5, -0.5]
        # K-002 has shape [1/5, 1/5] = [0.2, 0.2] → delta_z = [-1.5, -1.5]
        # K-003 has shape [2/5, 3/5] = [0.4, 0.6] → delta_z = [-0.5, 0.5]
        # K-004 has shape [0, 0] = [0.0, 0.0] → delta_z = [-2.5, -2.5]
        mu = np.array([0.5, 0.5], dtype=np.float32)
        table = pa.table(
            {
                "primary_key": ["K-001", "K-002", "K-003", "K-004"],
                "scale": ["full", "full", "full", "full"],
                "delta": [[0.5, -0.5], [-1.5, -1.5], [-0.5, 0.5], [-2.5, -2.5]],
                "delta_norm": [0.707, 2.121, 0.707, 3.536],
                "is_anomaly": [False, False, False, True],
                "edges": pa.array(
                    [edges_k001, edges_k002, edges_k003, edges_k004], type=pa.list_(_EDGE_STRUCT)
                ),
                "last_refresh_at": [now_str] * 4,
                "updated_at": [now_str] * 4,
            }
        )

        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="customers", direction="out", required=True),
                RelationDef(line_id="products", direction="out", required=False),
            ],
            mu=mu,
            sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
            theta=np.array([1.5, 1.5], dtype=np.float32),  # z-scored: [0.3/0.2, 0.3/0.2]
            population_size=4,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
            edge_max=np.array(edge_max, dtype=np.float32) if edge_max else None,
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        storage = MagicMock()
        storage.read_geometry.return_value = table
        storage.read_sphere.return_value = sphere
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )
        return nav

    def test_attract_hub_continuous_top_n(self):
        nav = self._make_hub_nav(edge_max=[5, 5])
        results = nav.π7_attract_hub("p", top_n=3)
        # K-001: 3+2=5, K-003: 2+3=5, K-002: 1+1=2, K-004: 0
        assert len(results) == 3
        keys = [r[0] for r in results]
        assert "K-001" in keys[:2]  # tied at 5
        assert "K-003" in keys[:2]  # tied at 5
        assert keys[2] == "K-002"  # 2
        # alive_edge_count (int) and hub_score (float)
        assert results[0][1] == 5
        assert results[0][2] == pytest.approx(5.0, abs=0.5)

    def test_attract_hub_binary_fallback(self):
        nav = self._make_hub_nav(edge_max=None)
        results = nav.π7_attract_hub("p", top_n=4)
        # Binary: counts raw alive edges from JSON
        # K-001: 5, K-003: 5, K-002: 2, K-004: 0
        assert len(results) == 4
        assert results[0][1] == 5  # alive_edge_count
        assert results[-1][1] == 0

    def test_attract_hub_line_id_filter(self):
        nav = self._make_hub_nav(edge_max=[5, 5])
        results = nav.π7_attract_hub("p", top_n=4, line_id_filter="products")
        # products only: K-003=3, K-001=2, K-002=1, K-004=0
        assert results[0][0] == "K-003"
        assert results[0][1] == 3
        assert results[1][0] == "K-001"
        assert results[1][1] == 2

    def test_attract_hub_empty_geometry(self):
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Pattern

        table = pa.table(
            {
                "primary_key": pa.array([], type=pa.string()),
                "scale": pa.array([], type=pa.string()),
                "delta": pa.array([], type=pa.list_(pa.float32())),
                "delta_norm": pa.array([], type=pa.float64()),
                "is_anomaly": pa.array([], type=pa.bool_()),
                "edges": pa.array([], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": pa.array([], type=pa.string()),
                "updated_at": pa.array([], type=pa.string()),
            }
        )
        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="anchor",
            relations=[],
            mu=np.array([], dtype=np.float32),
            sigma_diag=np.array([], dtype=np.float32),
            theta=np.array([], dtype=np.float32),
            population_size=0,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        storage = MagicMock()
        storage.read_geometry.return_value = table
        storage.read_sphere.return_value = sphere
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )
        assert nav.π7_attract_hub("p") == []

    def test_attract_hub_continuous_scores_equal_raw_counts(self):
        """Verify numpy path produces same counts as manual JSON edge counting."""
        nav = self._make_hub_nav(edge_max=[5, 5])
        continuous_results = nav.π7_attract_hub("p", top_n=4)

        nav_bin = self._make_hub_nav(edge_max=None)
        binary_results = nav_bin.π7_attract_hub("p", top_n=4)

        # Same entities and same alive_edge_count per entity (order may differ for ties)
        continuous_by_key = {r[0]: r[1] for r in continuous_results}
        binary_by_key = {r[0]: r[1] for r in binary_results}
        assert set(continuous_by_key.keys()) == set(binary_by_key.keys())
        for key in continuous_by_key:
            assert continuous_by_key[key] == binary_by_key[key], (
                f"Count mismatch for {key}: {continuous_by_key[key]} != {binary_by_key[key]}"
            )


class TestHubScoreStats:
    """Tests for hub_score_stats population statistics."""

    def _make_hub_nav(self, *, edge_max=None):
        """Reuse the same fixture as TestPi7AttractHub."""
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Pattern, RelationDef

        now_str = "2025-01-01T00:00:00"
        edges_k001 = [
            {"line_id": "customers", "point_key": "C-1", "status": "alive", "direction": "out"},
            {"line_id": "customers", "point_key": "C-2", "status": "alive", "direction": "out"},
            {"line_id": "customers", "point_key": "C-3", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "out"},
        ]
        edges_k002 = [
            {"line_id": "customers", "point_key": "C-1", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "out"},
        ]
        edges_k003 = [
            {"line_id": "customers", "point_key": "C-1", "status": "alive", "direction": "out"},
            {"line_id": "customers", "point_key": "C-4", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "out"},
            {"line_id": "products", "point_key": "P-3", "status": "alive", "direction": "out"},
        ]
        edges_k004 = []

        mu = np.array([0.5, 0.5], dtype=np.float32)
        table = pa.table(
            {
                "primary_key": ["K-001", "K-002", "K-003", "K-004"],
                "scale": ["full", "full", "full", "full"],
                "delta": [[0.5, -0.5], [-1.5, -1.5], [-0.5, 0.5], [-2.5, -2.5]],
                "delta_norm": [0.707, 2.121, 0.707, 3.536],
                "is_anomaly": [False, False, False, True],
                "edges": pa.array(
                    [edges_k001, edges_k002, edges_k003, edges_k004], type=pa.list_(_EDGE_STRUCT)
                ),
                "last_refresh_at": [now_str] * 4,
                "updated_at": [now_str] * 4,
            }
        )
        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="customers", direction="out", required=True),
                RelationDef(line_id="products", direction="out", required=False),
            ],
            mu=mu,
            sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
            theta=np.array([1.5, 1.5], dtype=np.float32),  # z-scored
            population_size=4,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
            edge_max=np.array(edge_max, dtype=np.float32) if edge_max else None,
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        storage = MagicMock()
        storage.read_geometry.return_value = table
        storage.read_sphere.return_value = sphere
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )
        return nav

    def test_stats_continuous_correct_percentiles(self):
        # Continuous path: mu=[0.5,0.5], edge_max=[5,5]
        # scores: K-001=5, K-002=2, K-003=5, K-004=0
        nav = self._make_hub_nav(edge_max=[5, 5])
        stats = nav.hub_score_stats("p")
        assert stats["total_entities"] == 4
        assert stats["max"] == pytest.approx(5.0, abs=0.1)
        assert stats["mean"] == pytest.approx(3.0, abs=0.1)
        assert stats["p50"] == pytest.approx(3.5, abs=0.5)

    def test_stats_binary_mode(self):
        # Binary path: edges K-001=5, K-002=2, K-003=5, K-004=0
        nav = self._make_hub_nav(edge_max=None)
        stats = nav.hub_score_stats("p")
        assert stats["total_entities"] == 4
        assert stats["max"] == pytest.approx(5.0, abs=0.1)

    def test_stats_empty_returns_zeros(self):
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Pattern, RelationDef

        mu = np.array([0.5, 0.5], dtype=np.float32)
        table = pa.table(
            {
                "primary_key": pa.array([], type=pa.string()),
                "scale": pa.array([], type=pa.string()),
                "delta": pa.array([], type=pa.list_(pa.float32())),
                "delta_norm": pa.array([], type=pa.float32()),
                "is_anomaly": pa.array([], type=pa.bool_()),
                "edges": pa.array([], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": pa.array([], type=pa.string()),
                "updated_at": pa.array([], type=pa.string()),
            }
        )
        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="anchor",
            relations=[RelationDef(line_id="customers", direction="out", required=True)],
            mu=mu,
            sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
            theta=np.array([0.3, 0.3], dtype=np.float32),
            population_size=0,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
            edge_max=np.array([5, 5], dtype=np.float32),
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        storage = MagicMock()
        storage.read_geometry.return_value = table
        storage.read_sphere.return_value = sphere
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )
        stats = nav.hub_score_stats("p")
        assert stats["total_entities"] == 0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0

    def test_stats_with_line_id_filter_continuous(self):
        # Filtered by "customers" (dim_idx=0, edge_max[0]=5)
        # scores: K-001=3.0, K-002=1.0, K-003=2.0, K-004=0.0
        # mean=1.5, max=3.0, total_entities=4
        nav = self._make_hub_nav(edge_max=[5, 5])
        stats = nav.hub_score_stats("p", line_id_filter="customers")
        assert stats["total_entities"] == 4
        assert stats["max"] == pytest.approx(3.0, abs=0.1)
        assert stats["mean"] == pytest.approx(1.5, abs=0.1)
        # Must differ from global stats (max=5.0)
        global_stats = nav.hub_score_stats("p")
        assert stats["max"] < global_stats["max"]

    def test_stats_with_line_id_filter_binary(self):
        # Binary mode, filtered by "customers"
        # K-001=3, K-002=1, K-003=2, K-004=0 → max=3, mean=1.5
        nav = self._make_hub_nav(edge_max=None)
        stats = nav.hub_score_stats("p", line_id_filter="customers")
        assert stats["total_entities"] == 4
        assert stats["max"] == pytest.approx(3.0, abs=0.1)
        assert stats["mean"] == pytest.approx(1.5, abs=0.1)


class TestHubScoreHistory:
    """Tests for hub_score_history temporal hub evolution."""

    def _make_nav(self, *, edge_max, slices=None):
        """Build navigator with mocked engine returning a Solid."""
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Pattern, RelationDef

        mu = np.array([0.5, 0.5], dtype=np.float32)
        em = np.array(edge_max, dtype=np.float32) if edge_max else None

        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="customers", direction="out", required=True),
                RelationDef(line_id="products", direction="out", required=False),
            ],
            mu=mu,
            sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
            theta=np.array([0.3, 0.3], dtype=np.float32),
            population_size=4,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
            edge_max=em,
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        import pyarrow as pa

        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        storage.read_geometry.return_value = pa.table(
            {
                "delta": [[0.1, -0.1]],
                "delta_norm": pa.array([0.141], type=pa.float64()),
            }
        )
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1

        base_polygon = Polygon(
            primary_key="K-001",
            pattern_id="p",
            pattern_ver=1,
            pattern_type="anchor",
            scale=4,
            delta=np.array([0.1, -0.1], dtype=np.float32),
            delta_norm=0.141,
            is_anomaly=False,
            edges=[],
            last_refresh_at=datetime(2025, 6, 1, tzinfo=UTC),
            updated_at=datetime(2025, 6, 1, tzinfo=UTC),
        )

        solid = Solid(
            primary_key="K-001",
            pattern_id="p",
            base_polygon=base_polygon,
            slices=slices or [],
        )
        engine = MagicMock()
        engine.build_solid.return_value = solid

        nav = GDSNavigator(
            engine=engine,
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )
        return nav

    def _make_slice(self, idx, delta, ts):
        return SolidSlice(
            slice_index=idx,
            timestamp=ts,
            deformation_type="edge",
            delta_snapshot=np.array(delta, dtype=np.float32),
            delta_norm_snapshot=float(np.linalg.norm(delta)),
            pattern_ver=1,
            changed_property=None,
            changed_line_id="products",
            added_edge=None,
        )

    def test_binary_returns_empty(self):
        nav = self._make_nav(edge_max=None)
        result = nav.hub_score_history("K-001", "p")
        assert result == []

    def test_no_slices_returns_only_current(self):
        nav = self._make_nav(edge_max=[5, 5], slices=[])
        result = nav.hub_score_history("K-001", "p")
        assert len(result) == 1
        assert result[0]["deformation_type"] == "current"

    def test_slices_plus_current(self):
        sl1 = self._make_slice(0, [0.0, -0.2], datetime(2025, 1, 1, tzinfo=UTC))
        sl2 = self._make_slice(1, [0.05, -0.15], datetime(2025, 3, 1, tzinfo=UTC))
        nav = self._make_nav(edge_max=[5, 5], slices=[sl1, sl2])
        result = nav.hub_score_history("K-001", "p")
        assert len(result) == 3
        assert result[-1]["deformation_type"] == "current"

    def test_score_increases_with_growing_delta(self):
        # delta grows → hub_score increases
        # mu=[0.5,0.5], edge_max=[5,5]
        # sl1: delta=[-0.4,-0.4] → shape=[0.1,0.1] → score=1.0
        # sl2: delta=[-0.1,-0.1] → shape=[0.4,0.4] → score=4.0
        sl1 = self._make_slice(0, [-0.4, -0.4], datetime(2025, 1, 1, tzinfo=UTC))
        sl2 = self._make_slice(1, [-0.1, -0.1], datetime(2025, 3, 1, tzinfo=UTC))
        nav = self._make_nav(edge_max=[5, 5], slices=[sl1, sl2])
        result = nav.hub_score_history("K-001", "p")
        assert result[0]["hub_score"] < result[1]["hub_score"]

    def test_required_fields_in_each_entry(self):
        sl = self._make_slice(0, [0.0, 0.0], datetime(2025, 1, 1, tzinfo=UTC))
        nav = self._make_nav(edge_max=[5, 5], slices=[sl])
        result = nav.hub_score_history("K-001", "p")
        required = {
            "timestamp",
            "hub_score",
            "alive_edges_est",
            "deformation_type",
            "changed_line_id",
            "delta_norm",
        }
        for entry in result:
            assert required.issubset(entry.keys())


# --- centroid_map ---


def _make_centroid_nav(
    deltas: list[list[float]],
    edges_per_row: list[list[dict]],
    relations: list | None = None,
):
    """Build a navigator with geometry rows that have edges for group-by extraction."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef

    now_str = "2025-01-01T00:00:00"
    n = len(deltas)
    table = pa.table(
        {
            "primary_key": [f"E-{i:03d}" for i in range(n)],
            "scale": [1] * n,
            "delta": deltas,
            "delta_norm": [float(np.linalg.norm(d)) for d in deltas],
            "is_anomaly": [False] * n,
            "edges": pa.array(edges_per_row, type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [now_str] * n,
            "updated_at": [now_str] * n,
        }
    )
    if relations is None:
        relations = [
            RelationDef(
                line_id="products", direction="out", required=True, display_name="Products"
            ),
            RelationDef(line_id="stores", direction="in", required=False, display_name="Stores"),
        ]
    pattern = Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=relations,
        mu=np.array([0.5] * len(relations), dtype=np.float32),
        sigma_diag=np.array([0.2] * len(relations), dtype=np.float32),
        theta=np.array([0.3] * len(relations), dtype=np.float32),
        population_size=n,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    manifest.line_version.return_value = 1
    engine = GDSEngine(storage=None, cache=None)
    return GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())


class TestCentroidMap:
    def test_basic_group_by_line(self):
        """Groups entities by edge key for group_by_line."""
        deltas = [[1.0, 0.0], [1.0, 0.2], [0.0, 1.0], [0.0, 1.2]]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],
        ]
        nav = _make_centroid_nav(deltas, edges)

        result = nav.centroid_map("p", group_by_line="company_codes")

        groups = {g["key"]: g for g in result["group_centroids"]}
        assert len(groups) == 2
        assert groups["CC-01"]["count"] == 2
        assert groups["CC-02"]["count"] == 2
        np.testing.assert_allclose(groups["CC-01"]["vector"], [1.0, 0.1], atol=1e-5)
        np.testing.assert_allclose(groups["CC-02"]["vector"], [0.0, 1.1], atol=1e-5)
        assert result["dimensions"] == ["Products", "Stores"]

    def test_entities_without_edge_excluded(self):
        """Entities with no edge to group_by_line are excluded."""
        deltas = [[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [],  # no edge to company_codes
        ]
        nav = _make_centroid_nav(deltas, edges)

        result = nav.centroid_map("p", group_by_line="company_codes")

        # Only 2 entities included, outlier [5,5] excluded
        assert result["global_centroid"]["count"] == 2
        assert len(result["group_centroids"]) == 1

    def test_group_by_property(self):
        """group_by_property maps edge keys to property values."""
        deltas = [[1.0, 0.0], [1.0, 0.2], [0.0, 1.0], [0.0, 1.2]]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-03",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-04",
                    "status": "alive",
                    "direction": "in",
                }
            ],
        ]
        nav = _make_centroid_nav(deltas, edges)

        nav._storage.read_points.return_value = pa.table(
            {
                "primary_key": ["CC-01", "CC-02", "CC-03", "CC-04"],
                "version": [1, 1, 1, 1],
                "status": ["active"] * 4,
                "country": ["DE", "DE", "PL", "PL"],
                "created_at": ["2024-01-01T00:00:00"] * 4,
                "changed_at": ["2024-01-01T00:00:00"] * 4,
            }
        )

        result = nav.centroid_map(
            "p", group_by_line="company_codes", group_by_property="company_codes:country"
        )

        groups = {g["key"]: g for g in result["group_centroids"]}
        assert set(groups.keys()) == {"DE", "PL"}
        assert groups["DE"]["count"] == 2  # CC-01 + CC-02
        assert groups["PL"]["count"] == 2  # CC-03 + CC-04

    def test_empty_geometry_returns_empty(self):
        """Empty geometry table returns empty result."""
        from unittest.mock import MagicMock

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.model.sphere import Pattern, RelationDef

        storage = MagicMock()
        storage.read_geometry.return_value = pa.table(
            {
                "primary_key": pa.array([], type=pa.utf8()),
                "delta": pa.array([], type=pa.list_(pa.float32())),
                "edges": pa.array([], type=pa.list_(_EDGE_STRUCT)),
            }
        )
        sphere = MagicMock()
        sphere.patterns = {
            "p": Pattern(
                pattern_id="p",
                entity_type="e",
                pattern_type="anchor",
                relations=[RelationDef(line_id="x", direction="out", required=True)],
                mu=np.array([0.5], dtype=np.float32),
                sigma_diag=np.array([0.2], dtype=np.float32),
                theta=np.array([0.3], dtype=np.float32),
                population_size=0,
                computed_at=datetime(2024, 1, 1, tzinfo=UTC),
                version=1,
                status="production",
            )
        }
        storage.read_sphere.return_value = sphere
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=GDSEngine(storage=None, cache=None),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )

        result = nav.centroid_map("p", group_by_line="company_codes")

        assert result == {}

    def test_invalid_group_by_property_format_raises(self):
        """group_by_property without 'line_id:property' format raises GDSNavigationError."""
        deltas = [[1.0, 0.0], [0.0, 1.0]]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],
        ]
        nav = _make_centroid_nav(deltas, edges)

        with pytest.raises(GDSNavigationError, match="group_by_property"):
            nav.centroid_map("p", group_by_line="company_codes", group_by_property="country")

    def test_group_by_property_none_values_excluded(self):
        """Entities whose group_by_property value is None are excluded from centroid."""
        deltas = [[1.0, 0.0], [1.0, 0.2], [0.0, 1.0]]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-03",
                    "status": "alive",
                    "direction": "in",
                }
            ],
        ]
        nav = _make_centroid_nav(deltas, edges)

        # CC-03 has country=None → excluded
        nav._storage.read_points.return_value = pa.table(
            {
                "primary_key": ["CC-01", "CC-02", "CC-03"],
                "version": [1, 1, 1],
                "status": ["active"] * 3,
                "country": ["DE", "DE", None],
                "created_at": ["2024-01-01T00:00:00"] * 3,
                "changed_at": ["2024-01-01T00:00:00"] * 3,
            }
        )

        result = nav.centroid_map(
            "p", group_by_line="company_codes", group_by_property="company_codes:country"
        )

        # Only 2 entities included (CC-01 + CC-02 → DE), CC-03 excluded
        assert result["global_centroid"]["count"] == 2
        groups = {g["key"]: g for g in result["group_centroids"]}
        assert set(groups.keys()) == {"DE"}

    def test_group_by_line_raises_on_continuous_edge(self):
        """centroid_map group_by_line must raise ValueError for continuous-mode patterns."""
        # Continuous-mode: all point_keys for the target line are empty strings
        deltas = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        edges = [
            [{"line_id": "company_codes", "point_key": "", "status": "alive", "direction": "in"}],
            [{"line_id": "company_codes", "point_key": "", "status": "alive", "direction": "in"}],
            [{"line_id": "company_codes", "point_key": "", "status": "alive", "direction": "in"}],
        ]
        nav = _make_centroid_nav(deltas, edges)
        with pytest.raises(ValueError, match="continuous mode"):
            nav.centroid_map("p", group_by_line="company_codes")

    def test_centroid_map_continuous_mode_early_exit(self):
        """centroid_map must fail fast for continuous-mode patterns without reading geometry."""
        from unittest.mock import MagicMock

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="event",
            relations=[RelationDef(line_id="company_codes", direction="in", required=True)],
            mu=np.array([0.5], dtype=np.float32),
            sigma_diag=np.array([0.2], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            edge_max=np.array([1000.0], dtype=np.float32),  # continuous mode
            population_size=100,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=GDSEngine(storage=None, cache=None),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )

        with pytest.raises(ValueError, match="continuous mode"):
            nav.centroid_map("p", group_by_line="company_codes")

        storage.read_geometry.assert_not_called()

    def test_prop_columns_appear_in_dimensions_field(self):
        """centroid_map dimensions field includes prop_column names for all delta dims."""
        from hypertopos.model.sphere import RelationDef

        # 4 entities, 2 groups by company_codes, delta has 3 elements:
        # [0]=products (relation), [1]=stores (relation), [2]=tax_id (prop_column)
        deltas = [
            [0.5, 0.3, 1.0],  # E-000 → CC-01
            [0.5, 0.3, 0.0],  # E-001 → CC-01
            [0.1, 0.8, 1.0],  # E-002 → CC-02
            [0.1, 0.8, 0.0],  # E-003 → CC-02
        ]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "out",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "out",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "out",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "out",
                }
            ],  # noqa: E501
        ]
        relations = [
            RelationDef(
                line_id="products", direction="out", required=True, display_name="Products"
            ),
            RelationDef(line_id="stores", direction="in", required=False, display_name="Stores"),
        ]
        nav = _make_centroid_nav(deltas, edges, relations=relations)

        # Patch pattern to add prop_columns
        pattern = nav._storage.read_sphere.return_value.patterns["p"]
        pattern.prop_columns = ["tax_id"]

        result = nav.centroid_map("p", group_by_line="company_codes")

        dims = result["dimensions"]
        assert len(dims) == 3, f"Expected 3 dimensions, got {len(dims)}"
        assert dims[0] == "Products"
        assert dims[1] == "Stores"
        assert dims[2] == "tax_id", f"Expected 'tax_id' at index 2, got '{dims[2]}'"

    def test_explicit_sample_size(self):
        """centroid_map samples when sample_size is passed explicitly."""
        deltas = [
            [1.0, 0.0],
            [1.0, 0.2],
            [0.0, 1.0],
            [0.0, 1.2],
            [0.5, 0.5],
            [0.5, 0.7],
            [0.3, 0.3],
            [0.3, 0.1],
        ]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
        ]
        nav = _make_centroid_nav(deltas, edges)

        result = nav.centroid_map("p", group_by_line="company_codes", sample_size=4)

        assert result["sampled"] is True
        assert result["sample_size"] == 4
        assert result["total_entities"] == 8
        assert "global_centroid" in result
        assert len(result["group_centroids"]) > 0

    def test_no_sample_without_param(self):
        """centroid_map does NOT sample when sample_size is not passed."""
        deltas = [[1.0, 0.0], [0.0, 1.0]]
        edges = [
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-01",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
            [
                {
                    "line_id": "company_codes",
                    "point_key": "CC-02",
                    "status": "alive",
                    "direction": "in",
                }
            ],  # noqa: E501
        ]
        nav = _make_centroid_nav(deltas, edges)

        result = nav.centroid_map("p", group_by_line="company_codes")

        assert "sampled" not in result
        assert "sample_size" not in result


# ---------- π9 attract_drift ----------


class _MockSphereWithPatterns:
    """Sphere stub with configurable patterns dict."""

    def __init__(self, patterns):
        self.patterns = patterns


def _make_temporal_table(
    rows: list[dict],
    mu: np.ndarray | None = None,
    sigma_diag: np.ndarray | None = None,
) -> pa.Table:
    """Build a temporal slice table for π9 tests.

    rows: each dict has keys pk, idx, ts, delta (z-scored delta values).
    mu / sigma_diag: if provided, shape = delta * sigma + mu is stored so
    that the navigator can round-trip back to the original delta via
    (shape - mu) / sigma. When absent, delta values are stored directly
    as shape (only valid when the test pattern has mu=0, sigma=1).
    """
    if not rows:
        return pa.table(
            {
                "primary_key": pa.array([], type=pa.string()),
                "slice_index": pa.array([], type=pa.int32()),
                "timestamp": pa.array([], type=pa.timestamp("us", tz="UTC")),
                "deformation_type": pa.array([], type=pa.string()),
                "shape_snapshot": pa.array([], type=pa.list_(pa.float32())),
                "pattern_ver": pa.array([], type=pa.int32()),
                "changed_property": pa.array([], type=pa.string()),
                "changed_line_id": pa.array([], type=pa.string()),
            }
        )

    def _to_shape(delta: list[float]) -> list[float]:
        d = np.array(delta, dtype=np.float32)
        if mu is not None and sigma_diag is not None:
            s = np.maximum(sigma_diag, 1e-2)
            return (d * s + mu).tolist()
        return d.tolist()

    return pa.table(
        {
            "primary_key": [r["pk"] for r in rows],
            "slice_index": pa.array([r["idx"] for r in rows], type=pa.int32()),
            "timestamp": pa.array([r["ts"] for r in rows], type=pa.timestamp("us", tz="UTC")),
            "deformation_type": ["edge"] * len(rows),
            "shape_snapshot": [_to_shape(r["delta"]) for r in rows],
            "pattern_ver": pa.array([1] * len(rows), type=pa.int32()),
            "changed_property": [None] * len(rows),
            "changed_line_id": [None] * len(rows),
        }
    )


class _MockStorageWithPatterns(_MockStorage):
    def __init__(self, patterns, geometry_table=None, temporal_table=None):
        self._patterns = patterns
        self._geometry_table = geometry_table
        self._temporal_table = temporal_table

    def read_sphere(self):
        return _MockSphereWithPatterns(self._patterns)

    def read_geometry(self, pattern_id, version, **kw):
        if self._geometry_table is not None:
            return self._geometry_table
        return pa.table({})

    def read_temporal_centroids(self, pattern_id):
        return None

    def read_temporal_batch(self, pattern_id, filters=None):
        if self._temporal_table is not None:
            return self._temporal_table
        return _make_temporal_table([])

    def read_temporal_batched(
        self,
        pattern_id,
        batch_size=65_536,
        timestamp_from=None,
        timestamp_to=None,
        keys=None,
    ):
        table = self.read_temporal_batch(pattern_id)
        if table.num_rows > 0:
            if keys is not None:
                key_set = set(keys)
                pk_list = table["primary_key"].to_pylist()
                keep = [i for i, pk in enumerate(pk_list) if pk in key_set]
                table = table.take(keep) if keep else table.slice(0, 0)
            if table.num_rows > 0:
                yield table.to_batches()[0]

    def _apply_temporal_filters(self, table, filters):
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader.__new__(GDSReader)
        return reader._apply_temporal_filters(table, filters)


def _make_continuous_centroid_nav(
    deltas: list[list[float]],
    keys: list[str],
    prop_map: dict[str, str],
    prop_name: str = "category",
):
    """Build a navigator simulating a continuous-mode pattern (edge_max set).

    In continuous mode, geometry edges store numeric weights, not entity keys.
    The point_key field is empty string for all edges.
    """
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef

    now_str = "2025-01-01T00:00:00"
    n = len(deltas)
    dim = len(deltas[0])
    # Continuous-mode self-edges: point_key = "" for all
    edges_per_row = [
        [{"line_id": "entities", "point_key": "", "status": "alive", "direction": "self"}]
        for _ in range(n)
    ]
    table = pa.table(
        {
            "primary_key": keys,
            "scale": [1] * n,
            "delta": deltas,
            "delta_norm": [float(np.linalg.norm(d)) for d in deltas],
            "is_anomaly": [False] * n,
            "edges": pa.array(edges_per_row, type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [now_str] * n,
            "updated_at": [now_str] * n,
        }
    )
    relations = [
        RelationDef(line_id=f"dim_{i}", direction="out", required=True, display_name=f"Dim{i}")
        for i in range(dim)
    ]
    pattern = Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=relations,
        mu=np.array([0.5] * dim, dtype=np.float32),
        sigma_diag=np.array([0.2] * dim, dtype=np.float32),
        theta=np.array([0.3] * dim, dtype=np.float32),
        population_size=n,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=[1] * dim,  # continuous mode
    )
    # entity line: line_id="entities", pattern_id="p"
    entity_line = MagicMock()
    entity_line.pattern_id = "p"
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    sphere.lines = {"entities": entity_line}
    sphere.entity_line.return_value = "entities"
    # prop_table returned by read_points("entities", 1)
    all_keys = list(prop_map.keys())
    all_vals = [prop_map[k] for k in all_keys]
    prop_table = pa.table({"primary_key": all_keys, prop_name: all_vals})
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    storage.read_points.return_value = prop_table
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    manifest.line_version.return_value = 1
    engine = GDSEngine(storage=None, cache=None)
    return GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())


class TestCentroidMapContinuousMode:
    """centroid_map self-group + group_by_property on continuous-mode patterns."""

    def test_self_group_with_property_returns_groups(self):
        """Continuous-mode pattern: group_by_line=entity_line + group_by_property must succeed."""
        keys = ["E-000", "E-001", "E-002", "E-003", "E-004"]
        deltas = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]]
        prop_map = {"E-000": "A", "E-001": "A", "E-002": "B", "E-003": "B", "E-004": "A"}
        nav = _make_continuous_centroid_nav(deltas, keys, prop_map)

        result = nav.centroid_map(
            "p",
            group_by_line="entities",
            group_by_property="entities:category",
        )

        groups = {g["key"]: g for g in result["group_centroids"]}
        assert len(groups) == 2, f"Expected 2 groups (A, B), got: {list(groups)}"
        assert groups["A"]["count"] == 3
        assert groups["B"]["count"] == 2
        # Centroid of A ≈ mean([1,0], [0.9,0.1], [0.5,0.5]) = [0.8, 0.2]
        np.testing.assert_allclose(groups["A"]["vector"], [0.8, 0.2], atol=1e-5)

    def test_self_group_without_property_raises(self):
        """Continuous-mode pattern: group_by_line=entity_line, no group_by_property → raises."""
        keys = ["E-000", "E-001"]
        deltas = [[1.0, 0.0], [0.0, 1.0]]
        prop_map = {"E-000": "A", "E-001": "B"}
        nav = _make_continuous_centroid_nav(deltas, keys, prop_map)

        with pytest.raises(ValueError, match="continuous"):
            nav.centroid_map("p", group_by_line="entities")

    def test_discrete_mode_self_group_still_raises(self):
        """Discrete-mode pattern: self-referential guard still fires regardless of property."""
        deltas = [[1.0, 0.0], [0.0, 1.0]]
        edges = [
            [{"line_id": "entities", "point_key": "E-000", "status": "alive", "direction": "in"}],
            [{"line_id": "entities", "point_key": "E-001", "status": "alive", "direction": "in"}],
        ]
        nav = _make_centroid_nav(deltas, edges)
        # Override entity line to match group_by_line
        from unittest.mock import MagicMock

        entity_line = MagicMock()
        entity_line.pattern_id = "p"
        nav._storage.read_sphere.return_value.lines = {"entities": entity_line}
        nav._storage.read_sphere.return_value.entity_line.return_value = "entities"

        with pytest.raises(ValueError, match="own line"):
            nav.centroid_map("p", group_by_line="entities")


class TestPi9AttractDrift:
    def test_fact_pattern_raises_value_error(self):
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="sale_pattern",
            entity_type="sale",
            pattern_type="event",
            relations=[RelationDef("customers", "in", True)],
            mu=np.array([0.8], dtype=np.float32),
            sigma_diag=np.array([0.1], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=100,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        storage = _MockStorageWithPatterns({"sale_pattern": pattern})
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        with pytest.raises(ValueError, match="event"):
            nav.π9_attract_drift("sale_pattern")

    def test_basic_drift_ranking(self):
        """Two entities with different drift — higher displacement ranked first."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
                RelationDef("profit_centers", "in", False),
            ],
            mu=np.array([0.8, 0.3], dtype=np.float32),
            sigma_diag=np.array([0.1, 0.1], dtype=np.float32),
            theta=np.array([0.3, 0.3], dtype=np.float32),
            population_size=2,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        geometry = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002"],
                "scale": [1, 1],
                "delta": [[0.1, 0.2], [0.05, 0.05]],
                "delta_norm": [0.22, 0.07],
                "is_anomaly": [False, False],
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)
        t3 = datetime(2025, 1, 1, tzinfo=UTC)

        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0, 0.0]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.3, 0.1]},
                {"pk": "CUST-001", "idx": 2, "ts": t3, "delta": [0.6, 0.2]},
                {"pk": "CUST-002", "idx": 0, "ts": ts, "delta": [0.0, 0.0]},
                {"pk": "CUST-002", "idx": 1, "ts": t3, "delta": [0.05, 0.05]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern", top_n=10)

        assert len(results) == 2
        assert results[0]["primary_key"] == "CUST-001"
        assert results[1]["primary_key"] == "CUST-002"
        assert abs(results[0]["displacement"] - 0.6325) < 0.01
        assert results[0]["path_length"] > results[0]["displacement"] - 0.01
        assert 0.0 <= results[0]["ratio"] <= 1.0
        assert "dimension_diffs" in results[0]
        diffs = results[0]["dimension_diffs"]
        assert "company_codes" in diffs
        assert "profit_centers" in diffs
        assert abs(diffs["company_codes"] - 0.6) < 0.01
        assert abs(diffs["profit_centers"] - 0.2) < 0.01

    def test_includes_displacement_current(self):
        """displacement_current uses stored geometry delta, not slices[-1]."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
                RelationDef("profit_centers", "in", False),
            ],
            mu=np.array([0.8, 0.3], dtype=np.float32),
            sigma_diag=np.array([0.1, 0.1], dtype=np.float32),
            theta=np.array([0.3, 0.3], dtype=np.float32),
            population_size=1,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        # geometry delta = [0.5, 0.3] — current state, different from last slice [0.6, 0.2]
        geometry = pa.table(
            {
                "primary_key": ["CUST-001"],
                "scale": [1],
                "delta": [[0.5, 0.3]],
                "delta_norm": [0.583],
                "is_anomaly": [False],
                "edges": pa.array([[]], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)],
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)],
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)

        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0, 0.0]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.6, 0.2]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern", top_n=10)

        assert len(results) == 1
        r = results[0]

        # displacement (trajectory): ||slices[-1] - slices[0]|| = ||[0.6,0.2]|| ≈ 0.6325
        assert abs(r["displacement"] - 0.6325) < 0.001

        # displacement_current: ||geometry_delta - slices[0]|| = ||[0.5,0.3]|| ≈ 0.5831
        assert "displacement_current" in r
        assert abs(r["displacement_current"] - 0.5831) < 0.001

        # dimension_diffs_current: [0.5-0.0, 0.3-0.0] = [0.5, 0.3]
        assert "dimension_diffs_current" in r
        assert abs(r["dimension_diffs_current"]["company_codes"] - 0.5) < 0.001
        assert abs(r["dimension_diffs_current"]["profit_centers"] - 0.3) < 0.001

    def test_skips_entities_with_fewer_than_2_slices(self):
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[RelationDef("company_codes", "in", True)],
            mu=np.array([0.8], dtype=np.float32),
            sigma_diag=np.array([0.1], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=2,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        geometry = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002"],
                "scale": [1, 1],
                "delta": [[0.1], [0.05]],
                "delta_norm": [0.1, 0.05],
                "is_anomaly": [False, False],
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        # CUST-001 has 1 slice, CUST-002 has 0 slices — both skipped
        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.1]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern")

        assert results == []

    def test_sample_size_limits_processed_entities(self):
        """With sample_size=1, only 1 of 3 entities is processed."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[RelationDef("company_codes", "in", True)],
            mu=np.array([0.8], dtype=np.float32),
            sigma_diag=np.array([0.1], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=3,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        geometry = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002", "CUST-003"],
                "scale": [1] * 3,
                "delta": [[0.1]] * 3,
                "delta_norm": [0.1] * 3,
                "is_anomaly": [False] * 3,
                "edges": pa.array([[], [], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 1, 1, tzinfo=UTC)
        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.1]},
                {"pk": "CUST-002", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-002", "idx": 1, "ts": t2, "delta": [0.1]},
                {"pk": "CUST-003", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-003", "idx": 1, "ts": t2, "delta": [0.1]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern", sample_size=1)

        assert len(results) == 1

    def test_sample_size_passes_keys_to_temporal_read(self):
        """When sample_size is given, read_temporal_batched receives keys≤sample_size,
        not None — ensuring the Lance scanner is scoped to the sampled subset."""
        from unittest.mock import patch

        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[RelationDef("company_codes", "in", True)],
            mu=np.array([0.8], dtype=np.float32),
            sigma_diag=np.array([0.1], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=3,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        geometry = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002", "CUST-003"],
                "scale": [1] * 3,
                "delta": [[0.1]] * 3,
                "delta_norm": [0.1] * 3,
                "is_anomaly": [False] * 3,
                "edges": pa.array([[], [], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 1, 1, tzinfo=UTC)
        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.1]},
                {"pk": "CUST-002", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-002", "idx": 1, "ts": t2, "delta": [0.1]},
                {"pk": "CUST-003", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-003", "idx": 1, "ts": t2, "delta": [0.1]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))

        captured_keys: list = []
        original_batched = storage.read_temporal_batched

        def capturing_batched(
            pattern_id, batch_size=65_536, timestamp_from=None, timestamp_to=None, keys=None
        ):
            captured_keys.extend(keys if keys is not None else [])
            yield from original_batched(
                pattern_id,
                batch_size=batch_size,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                keys=keys,
            )

        with patch.object(storage, "read_temporal_batched", side_effect=capturing_batched):
            nav.π9_attract_drift("cust_pattern", sample_size=2)

        # keys must be passed (not None) and bounded by sample_size
        assert len(captured_keys) <= 2, (
            f"Expected ≤2 keys forwarded to temporal reader, got {len(captured_keys)}"
        )
        assert len(captured_keys) > 0, "keys must not be None when sample_size is set"

    def test_no_sample_size_passes_no_keys_to_temporal_read(self):
        """When sample_size is None, read_temporal_batched receives keys=None
        (full population scan — same behavior as before the optimization)."""
        from unittest.mock import patch

        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[RelationDef("company_codes", "in", True)],
            mu=np.array([0.8], dtype=np.float32),
            sigma_diag=np.array([0.1], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=2,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        geometry = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002"],
                "scale": [1] * 2,
                "delta": [[0.1]] * 2,
                "delta_norm": [0.1] * 2,
                "is_anomaly": [False] * 2,
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 1, 1, tzinfo=UTC)
        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.1]},
                {"pk": "CUST-002", "idx": 0, "ts": ts, "delta": [0.0]},
                {"pk": "CUST-002", "idx": 1, "ts": t2, "delta": [0.1]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))

        captured: dict = {"keys_arg": "SENTINEL"}

        def capturing_batched(
            pattern_id, batch_size=65_536, timestamp_from=None, timestamp_to=None, keys=None
        ):
            captured["keys_arg"] = keys
            yield from storage.read_temporal_batch(pattern_id).to_batches()

        with patch.object(storage, "read_temporal_batched", side_effect=capturing_batched):
            nav.π9_attract_drift("cust_pattern")  # no sample_size

        assert captured["keys_arg"] is None, (
            f"Expected keys=None for full population scan, got {captured['keys_arg']!r}"
        )

    def test_find_drifting_entities_tac_score(self):
        """TAC score: stable history → ≈1.0, volatile → 0.0, n==2 → None."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[RelationDef("companies", "in", True)],
            mu=np.array([0.0], dtype=np.float32),
            sigma_diag=np.array([1.0], dtype=np.float32),
            theta=np.array([0.5], dtype=np.float32),
            population_size=3,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        geometry = pa.table(
            {
                "primary_key": ["STABLE-001", "VOLAT-001", "TWO-001"],
                "scale": [1, 1, 1],
                "delta": [[1.0], [0.1], [1.0]],
                "delta_norm": [1.0, 0.1, 1.0],
                "is_anomaly": [True, True, False],
                "edges": pa.array([[], [], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)
        t3 = datetime(2025, 1, 1, tzinfo=UTC)

        # STABLE-001: 3 slices, delta_norms ≈ [1.0, 0.95, 1.05] → low CV → TAC ≈ 1
        # VOLAT-001:  3 slices, delta_norms = [0.1, 5.0, 0.1]   → high CV → TAC = 0
        # TWO-001:    2 slices only → tac = None
        temporal = _make_temporal_table(
            [
                {"pk": "STABLE-001", "idx": 0, "ts": ts, "delta": [1.0]},
                {"pk": "STABLE-001", "idx": 1, "ts": t2, "delta": [0.95]},
                {"pk": "STABLE-001", "idx": 2, "ts": t3, "delta": [1.05]},
                {"pk": "VOLAT-001", "idx": 0, "ts": ts, "delta": [0.1]},
                {"pk": "VOLAT-001", "idx": 1, "ts": t2, "delta": [5.0]},
                {"pk": "VOLAT-001", "idx": 2, "ts": t3, "delta": [0.1]},
                {"pk": "TWO-001", "idx": 0, "ts": ts, "delta": [1.0]},
                {"pk": "TWO-001", "idx": 1, "ts": t2, "delta": [2.0]},
            ]
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern", top_n=10)

        by_key = {r["primary_key"]: r for r in results}
        assert "STABLE-001" in by_key
        assert "VOLAT-001" in by_key
        assert "TWO-001" in by_key

        assert by_key["STABLE-001"]["tac"] is not None
        assert by_key["STABLE-001"]["tac"] >= 0.9  # stable → near 1.0

        assert by_key["VOLAT-001"]["tac"] == 0.0  # volatile → clamped to 0

        assert by_key["TWO-001"]["tac"] is None  # n < 3 → None

    def test_dimension_diffs_excludes_prop_columns(self):
        """dimension_diffs_current must only use relation line_ids, not prop columns."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
                RelationDef("profit_centers", "in", False),
            ],
            prop_columns=["fashion_news_frequency"],
            mu=np.array([0.8, 0.3, 50.0], dtype=np.float32),
            sigma_diag=np.array([0.1, 0.1, 1.0], dtype=np.float32),
            theta=np.array([0.3, 0.3, 100.0], dtype=np.float32),
            population_size=1,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        geometry = pa.table(
            {
                "primary_key": ["CUST-001"],
                "scale": [1],
                "delta": [[0.5, 0.3, 99.0]],
                "delta_norm": [0.583],
                "is_anomaly": [False],
                "edges": pa.array([[]], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)],
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)],
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)

        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0, 0.0, 0.0]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.6, 0.2, 100.0]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern", top_n=10)

        assert len(results) == 1
        r = results[0]

        # dimension_diffs_current must only contain structural relation dims
        assert set(r["dimension_diffs_current"].keys()) == {"company_codes", "profit_centers"}
        assert "fashion_news_frequency" not in r["dimension_diffs_current"]

        # dimension_diffs must only contain structural relation dims
        assert set(r["dimension_diffs"].keys()) == {"company_codes", "profit_centers"}
        assert "fashion_news_frequency" not in r["dimension_diffs"]

    def test_prop_column_changes_field_present(self):
        """prop_column_changes must appear in results with boolean values per prop_column."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[RelationDef("company_codes", "in", True)],
            prop_columns=["fashion_news_frequency"],
            mu=np.array([0.8, 50.0], dtype=np.float32),
            sigma_diag=np.array([0.1, 1.0], dtype=np.float32),
            theta=np.array([0.3, 100.0], dtype=np.float32),
            population_size=2,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        # CUST-001: prop changed (diff_current > 50 in standardised space)
        # CUST-002: prop unchanged (diff_current ≈ 0)
        geometry = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002"],
                "scale": [1, 1],
                "delta": [[0.5, 99.0], [0.5, 0.0]],
                "delta_norm": [0.583, 0.583],
                "is_anomaly": [False, False],
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)

        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0, 0.0]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.6, 0.0]},
                {"pk": "CUST-002", "idx": 0, "ts": ts, "delta": [0.0, 0.0]},
                {"pk": "CUST-002", "idx": 1, "ts": t2, "delta": [0.6, 0.0]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern", top_n=10)

        assert len(results) == 2
        by_key = {r["primary_key"]: r for r in results}

        # Both results must have prop_column_changes
        assert "prop_column_changes" in by_key["CUST-001"]
        assert "prop_column_changes" in by_key["CUST-002"]

        # CUST-001: geometry delta prop dim = 99.0, delta_first prop = 0.0
        # diff_current prop dim = 99.0 - 0.0 = 99.0 > 0.5 → True
        assert by_key["CUST-001"]["prop_column_changes"]["fashion_news_frequency"] is True

        # CUST-002: geometry delta prop dim = 0.0, delta_first prop = 0.0
        # diff_current prop dim = 0.0 - 0.0 = 0.0 ≤ 0.5 → False
        assert by_key["CUST-002"]["prop_column_changes"]["fashion_news_frequency"] is False

        # Values must be booleans
        assert isinstance(by_key["CUST-001"]["prop_column_changes"]["fashion_news_frequency"], bool)

    def test_prop_column_changes_moderate_delta_detected(self):
        """With SIGMA_EPS_PROP=0.2, prop_column diff ~ 4 must be detected as change (> 0.5)."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[RelationDef("company_codes", "in", True)],
            prop_columns=["fashion_news_frequency"],
            mu=np.array([0.8, 0.95], dtype=np.float32),
            sigma_diag=np.array([0.1, 0.2], dtype=np.float32),
            theta=np.array([0.3, 0.5], dtype=np.float32),
            population_size=2,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        # CUST-001: prop dim delta = -4.75 (entity without prop: (0-0.95)/0.2 = -4.75)
        # CUST-002: prop dim delta = 0.25 (entity with prop: (1-0.95)/0.2 = 0.25)
        geometry = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002"],
                "scale": [1, 1],
                "delta": [[0.5, -4.75], [0.5, 0.25]],
                "delta_norm": [4.78, 0.56],
                "is_anomaly": [False, False],
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, tzinfo=UTC)

        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts, "delta": [0.0, 0.25]},
                {"pk": "CUST-001", "idx": 1, "ts": t2, "delta": [0.6, 0.25]},
                {"pk": "CUST-002", "idx": 0, "ts": ts, "delta": [0.0, 0.25]},
                {"pk": "CUST-002", "idx": 1, "ts": t2, "delta": [0.6, 0.25]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        results = nav.π9_attract_drift("cust_pattern", top_n=10)

        by_key = {r["primary_key"]: r for r in results}

        # CUST-001: diff_current prop = -4.75 - 0.25 = -5.0, abs = 5.0 > 0.5 → True
        assert by_key["CUST-001"]["prop_column_changes"]["fashion_news_frequency"] is True
        # CUST-002: diff_current prop = 0.25 - 0.25 = 0.0, abs = 0.0 ≤ 0.5 → False
        assert by_key["CUST-002"]["prop_column_changes"]["fashion_news_frequency"] is False

    def test_drift_rank_by_dimension(self):
        """rank_by_dimension changes which entity is ranked first."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
                RelationDef("profit_centers", "in", False),
            ],
            mu=np.array([0.8, 0.3], dtype=np.float32),
            sigma_diag=np.array([0.1, 0.1], dtype=np.float32),
            theta=np.array([0.3, 0.3], dtype=np.float32),
            population_size=2,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        geometry = pa.table(
            {
                "primary_key": ["CUST-A", "CUST-B"],
                "scale": [1, 1],
                "delta": [[0.9, 0.1], [0.1, 0.8]],
                "delta_norm": [0.91, 0.81],
                "is_anomaly": [False, False],
                "edges": pa.array([[], []], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                "updated_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            }
        )

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 1, 1, tzinfo=UTC)

        temporal = _make_temporal_table(
            [
                {"pk": "CUST-A", "idx": 0, "ts": ts, "delta": [0.0, 0.0]},
                {"pk": "CUST-A", "idx": 1, "ts": t2, "delta": [0.9, 0.1]},
                {"pk": "CUST-B", "idx": 0, "ts": ts, "delta": [0.0, 0.0]},
                {"pk": "CUST-B", "idx": 1, "ts": t2, "delta": [0.1, 0.8]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, geometry, temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))

        # Default ranking: CUST-A first (higher total displacement)
        results_default = nav.π9_attract_drift("cust_pattern", top_n=10)
        assert results_default[0]["primary_key"] == "CUST-A"

        # Rank by profit_centers: CUST-B first (0.8 > 0.1)
        results_pc = nav.π9_attract_drift(
            "cust_pattern", top_n=10, rank_by_dimension="profit_centers"
        )
        assert results_pc[0]["primary_key"] == "CUST-B"

        # Rank by company_codes: CUST-A first (0.9 > 0.1)
        results_cc = nav.π9_attract_drift(
            "cust_pattern", top_n=10, rank_by_dimension="company_codes"
        )
        assert results_cc[0]["primary_key"] == "CUST-A"

    def test_drift_rank_by_dimension_invalid(self):
        """rank_by_dimension with nonexistent dimension raises GDSNavigationError."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
            ],
            mu=np.array([0.8], dtype=np.float32),
            sigma_diag=np.array([0.1], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=1,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern})
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))

        with pytest.raises(GDSNavigationError, match="not found in structural dimensions"):
            nav.π9_attract_drift("cust_pattern", rank_by_dimension="nonexistent")


class TestPi11AttractPopulationCompare:
    def test_compare_time_windows_excludes_prop_columns(self):
        """top_changed_dimensions must only contain structural relation dims, not prop_columns."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
                RelationDef("profit_centers", "in", False),
            ],
            prop_columns=["fashion_news_frequency"],
            mu=np.array([0.8, 0.3, 50.0], dtype=np.float32),
            sigma_diag=np.array([0.1, 0.1, 1.0], dtype=np.float32),
            theta=np.array([0.3, 0.3, 100.0], dtype=np.float32),
            population_size=4,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

        ts_a1 = datetime(2024, 1, 15, tzinfo=UTC)
        ts_a2 = datetime(2024, 2, 15, tzinfo=UTC)
        ts_b1 = datetime(2024, 7, 15, tzinfo=UTC)
        ts_b2 = datetime(2024, 8, 15, tzinfo=UTC)

        # Two entities in window A, two in window B.
        # prop_column has high change (±100) to confirm it's excluded.
        temporal = _make_temporal_table(
            [
                {"pk": "CUST-001", "idx": 0, "ts": ts_a1, "delta": [0.2, 0.1, 0.0]},
                {"pk": "CUST-002", "idx": 0, "ts": ts_a2, "delta": [0.3, 0.2, 0.0]},
                {"pk": "CUST-003", "idx": 0, "ts": ts_b1, "delta": [0.8, 0.9, 100.0]},
                {"pk": "CUST-004", "idx": 0, "ts": ts_b2, "delta": [0.9, 0.8, 100.0]},
            ],
            mu=pattern.mu,
            sigma_diag=pattern.sigma_diag,
        )

        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, temporal_table=temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))

        result = nav.π11_attract_population_compare(
            pattern_id="cust_pattern",
            window_a_from="2024-01-01",
            window_a_to="2024-06-01",
            window_b_from="2024-07-01",
            window_b_to="2025-01-01",
        )

        assert result["centroid_shift"] is not None
        top_dims = result["top_changed_dimensions"]
        assert len(top_dims) > 0

        dim_names_in_result = {d["dimension"] for d in top_dims}
        assert "fashion_news_frequency" not in dim_names_in_result
        assert "company_codes" in dim_names_in_result or "profit_centers" in dim_names_in_result

        # centroid_shift must be computed over structural dims only (2 dims), not 3
        # If prop_col were included it would dominate; structural shift ≈ norm([0.65, 0.7]) ≈ 0.96
        # With prop_col it would be >> 99. So shift < 10 confirms structural-only.
        assert result["centroid_shift"] < 10.0, (
            f"centroid_shift={result['centroid_shift']} suggests prop_cols were included"
        )


# --- π12_attract_regime_change empty result + prop_col masking ---


class TestPi12AttractRegimeChange:
    def _make_anchor_pattern(self, prop_columns=None):
        from hypertopos.model.sphere import Pattern, RelationDef

        return Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
                RelationDef("profit_centers", "in", False),
            ],
            mu=np.array([0.5, 0.5], dtype=np.float32),
            sigma_diag=np.array([0.1, 0.1], dtype=np.float32),
            theta=np.array([0.3, 0.3], dtype=np.float32),
            population_size=10,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
            prop_columns=list(prop_columns) if prop_columns else [],
        )

    def test_empty_result_returns_warning(self):
        """When all bucket shifts are below threshold, return structured warning."""
        pattern = self._make_anchor_pattern()

        # Uniform data: all deltas are identical — no centroid shift between buckets
        datetime(2024, 1, 1, tzinfo=UTC)
        rows = [
            {
                "pk": f"C-{i:03d}",
                "idx": 0,
                "ts": datetime(2024, i % 12 + 1, 1, tzinfo=UTC),
                "delta": [0.5, 0.5],
            }
            for i in range(12)
        ]
        temporal = _make_temporal_table(rows, mu=pattern.mu, sigma_diag=pattern.sigma_diag)
        storage = _MockStorageWithPatterns({"cust_pattern": pattern}, temporal_table=temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))

        result = nav.π12_attract_regime_change("cust_pattern")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "warning" in result[0]
        assert "no_regime_changes_detected" in result[0]["warning"]
        assert "compare_time_windows" in result[0]["warning"]

    def test_event_pattern_raises_value_error(self):
        """pi12 must reject event patterns with ValueError."""
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="sale_pattern",
            entity_type="sale",
            pattern_type="event",
            relations=[RelationDef("customers", "in", True)],
            mu=np.array([0.8], dtype=np.float32),
            sigma_diag=np.array([0.1], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=100,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        storage = _MockStorageWithPatterns({"sale_pattern": pattern})
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))
        with pytest.raises(ValueError, match="event"):
            nav.π12_attract_regime_change("sale_pattern")

    def test_top_changed_dimensions_excludes_prop_columns(self):
        """top_changed_dimensions must contain only relation dimensions, not prop_cols."""
        self._make_anchor_pattern(prop_columns=["fashion_news_frequency"])

        # 3 dims total: company_codes, profit_centers, fashion_news_frequency
        # All shapes have 3 values; prop_col (idx 2) has extreme values
        mu = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        sigma = np.array([0.1, 0.1, 0.1], dtype=np.float32)

        # Override pattern mu/sigma to match 3-dim data
        from hypertopos.model.sphere import Pattern, RelationDef

        pattern3 = Pattern(
            pattern_id="cust_pattern",
            entity_type="customer",
            pattern_type="anchor",
            relations=[
                RelationDef("company_codes", "in", True),
                RelationDef("profit_centers", "in", False),
            ],
            mu=mu,
            sigma_diag=sigma,
            theta=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            population_size=10,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
            prop_columns=["fashion_news_frequency"],
        )

        # Two groups of timestamps with a large centroid shift in dim 0 and 1,
        # and a huge value in dim 2 (prop_col) to verify it's masked out.
        rows = [
            {
                "pk": "C-001",
                "idx": 0,
                "ts": datetime(2024, 1, 1, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-002",
                "idx": 0,
                "ts": datetime(2024, 1, 2, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-003",
                "idx": 0,
                "ts": datetime(2024, 1, 3, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-004",
                "idx": 0,
                "ts": datetime(2024, 1, 4, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-005",
                "idx": 0,
                "ts": datetime(2024, 1, 5, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-006",
                "idx": 0,
                "ts": datetime(2024, 6, 1, tzinfo=UTC),
                "delta": [5.0, 5.0, 0.0],
            },
            {
                "pk": "C-007",
                "idx": 0,
                "ts": datetime(2024, 6, 2, tzinfo=UTC),
                "delta": [5.0, 5.0, 0.0],
            },
            {
                "pk": "C-008",
                "idx": 0,
                "ts": datetime(2024, 6, 3, tzinfo=UTC),
                "delta": [5.0, 5.0, 0.0],
            },
            {
                "pk": "C-009",
                "idx": 0,
                "ts": datetime(2024, 6, 4, tzinfo=UTC),
                "delta": [5.0, 5.0, 0.0],
            },
            {
                "pk": "C-010",
                "idx": 0,
                "ts": datetime(2024, 6, 5, tzinfo=UTC),
                "delta": [5.0, 5.0, 0.0],
            },
            {
                "pk": "C-011",
                "idx": 0,
                "ts": datetime(2024, 11, 1, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-012",
                "idx": 0,
                "ts": datetime(2024, 11, 2, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-013",
                "idx": 0,
                "ts": datetime(2024, 11, 3, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-014",
                "idx": 0,
                "ts": datetime(2024, 11, 4, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
            {
                "pk": "C-015",
                "idx": 0,
                "ts": datetime(2024, 11, 5, tzinfo=UTC),
                "delta": [0.0, 0.0, 999.0],
            },
        ]
        temporal = _make_temporal_table(rows, mu=mu, sigma_diag=sigma)
        storage = _MockStorageWithPatterns({"cust_pattern": pattern3}, temporal_table=temporal)
        nav = GDSNavigator(_MockEngine(), storage, _make_manifest(), Contract("m-001", []))

        result = nav.π12_attract_regime_change("cust_pattern")

        # If there are regime changes, verify prop_col is absent from top_changed_dimensions
        # If uniform (warning), verify warning structure
        if len(result) == 1 and "warning" in result[0]:
            assert (
                "no_regime_changes_detected" in result[0]["warning"]
                or "temporal" in result[0]["warning"]
            )
        else:
            for change in result:
                dim_names_in_result = {d["dimension"] for d in change["top_changed_dimensions"]}
                assert "fashion_news_frequency" not in dim_names_in_result, (
                    "prop_col appeared in top_changed_dimensions: "
                    f"{change['top_changed_dimensions']}"
                )


# --- _resolve_version raises on missing pattern ---


class TestResolveVersionRaisesOnMissingPattern:
    """_resolve_version must raise GDSNavigationError, never silently fall back to v1."""

    def _make_nav_missing_pattern(self):
        from unittest.mock import MagicMock

        from hypertopos.model.sphere import Pattern, RelationDef

        pattern = Pattern(
            pattern_id="p",
            entity_type="e",
            pattern_type="anchor",
            relations=[RelationDef(line_id="customers", direction="in", required=True)],
            mu=np.array([0.5], dtype=np.float32),
            sigma_diag=np.array([0.2], dtype=np.float32),
            theta=np.array([0.3], dtype=np.float32),
            population_size=2,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = MagicMock()
        sphere.patterns = {"p": pattern}
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        # pattern_version returns None → simulates pattern absent from manifest
        manifest = MagicMock()
        manifest.pattern_version.return_value = None
        return GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )

    def test_pi5_attract_anomaly_raises(self):
        nav = self._make_nav_missing_pattern()
        with pytest.raises(GDSNavigationError, match="No geometry version"):
            nav.π5_attract_anomaly("p")

    def test_anomaly_summary_raises(self):
        nav = self._make_nav_missing_pattern()
        with pytest.raises(GDSNavigationError, match="No geometry version"):
            nav.anomaly_summary("p")


# ---------- π4_emerge tests ----------


class TestPi4Emerge:
    def _make_nav(self) -> GDSNavigator:
        return GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorage(),
            manifest=_make_manifest(),
            contract=Contract("m-001", []),
        )

    def test_pi4_emerge_after_solid(self):
        nav = self._make_nav()
        nav.π3_dive_solid("CUST-001", "customer_pattern")
        assert isinstance(nav.position, Solid)
        nav.π4_emerge()
        assert isinstance(nav.position, Point)
        assert nav.position.line_id == "emerged"
        assert nav.position.primary_key == "CUST-001"

    def test_pi4_noop_on_point(self):
        nav = self._make_nav()
        nav.goto("CUST-001", "customers")
        assert isinstance(nav.position, Point)
        nav.π4_emerge()
        assert isinstance(nav.position, Point)
        assert nav.position.primary_key == "CUST-001"
        assert nav.position.line_id == "customers"

    def test_pi4_raises_on_none_position(self):
        nav = self._make_nav()
        assert nav.position is None
        with pytest.raises(GDSPositionError, match="π4 requires active position"):
            nav.π4_emerge()

    def test_pi4_emerge_after_polygon_dead_code_branch(self):
        # dead-code branch — no public primitive sets Polygon as navigator position
        nav = self._make_nav()
        nav._position = _make_polygon_with_edge("CUST-001", "customers")
        nav.π4_emerge()
        assert nav.position.line_id == "emerged"


# ---------- relation-line error enrichment ----------


class _MockEngineKeyError:
    """Engine stub that always raises KeyError on build_polygon/build_solid."""

    def build_polygon(self, bk, pid, manifest):
        raise KeyError(f"No geometry for {bk} in {pid} v1")

    def build_solid(self, bk, pid, manifest, filters=None, timestamp=None):
        raise KeyError(f"No geometry for {bk} in {pid} v1")


class _MockStorageWithRelations(_MockStorage):
    """Storage that knows customer_pattern has company_codes as a relation line."""

    def read_sphere(self):
        from hypertopos.model.sphere import Pattern, RelationDef, Sphere

        pat = Pattern(
            pattern_id="customer_pattern",
            entity_type="customers",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="company_codes", direction="in", required=True),
                RelationDef(line_id="products", direction="out", required=False),
            ],
            mu=np.zeros(2, dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.array([1.0, 1.0], dtype=np.float32),
            population_size=10,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = Sphere("s", "s", ".")
        sphere.patterns["customer_pattern"] = pat
        return sphere


def _make_navigator_at(line_id: str, bk: str, engine, storage):
    """Helper: create a navigator pre-positioned at the given entity."""
    nav = GDSNavigator(
        engine=engine,
        storage=storage,
        manifest=Manifest(
            manifest_id="m-test",
            agent_id="a-test",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "company_codes": 1},
            pattern_versions={"customer_pattern": 1},
        ),
        contract=Contract("m-test", ["customer_pattern"]),
    )
    nav._position = Point(
        primary_key=bk,
        line_id=line_id,
        version=1,
        status="active",
        properties={},
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        changed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return nav


def test_current_polygon_relation_line_error_is_enriched():
    """current_polygon raises GDSNavigationError with relation-line hint."""
    nav = _make_navigator_at(
        "company_codes", "CC-04", _MockEngineKeyError(), _MockStorageWithRelations()
    )
    with pytest.raises(GDSNavigationError) as exc_info:
        nav.current_polygon("customer_pattern")
    msg = str(exc_info.value)
    assert "relation line" in msg
    assert "company_codes" in msg
    assert "get_event_polygons" in msg


def test_current_polygon_non_relation_line_raises_navigation_error():
    """current_polygon on a non-relation-line entity still raises GDSNavigationError."""
    nav = _make_navigator_at(
        "unknown_line", "X-001", _MockEngineKeyError(), _MockStorageWithRelations()
    )
    with pytest.raises(GDSNavigationError):
        nav.current_polygon("customer_pattern")


def test_current_solid_relation_line_error_is_enriched():
    """current_solid raises GDSNavigationError with relation-line hint."""
    nav = _make_navigator_at(
        "company_codes", "CC-04", _MockEngineKeyError(), _MockStorageWithRelations()
    )
    with pytest.raises(GDSNavigationError) as exc_info:
        nav.current_solid("customer_pattern")
    msg = str(exc_info.value)
    assert "relation line" in msg
    assert "company_codes" in msg


def test_current_solid_non_relation_line_raises_navigation_error():
    """current_solid on a non-relation-line entity still raises GDSNavigationError."""
    nav = _make_navigator_at(
        "unknown_line", "X-001", _MockEngineKeyError(), _MockStorageWithRelations()
    )
    with pytest.raises(GDSNavigationError):
        nav.current_solid("customer_pattern")


def test_current_polygon_uses_stored_delta():
    """current_polygon patches delta/delta_norm/is_anomaly/delta_rank_pct from geometry dataset."""
    from unittest.mock import MagicMock

    # build_polygon returns wrong (recomputed) values
    recomputed_delta = np.array([0.01, 0.02], dtype=np.float32)
    recomputed_norm = float(np.linalg.norm(recomputed_delta))
    built_polygon = Polygon(
        primary_key="E-001",
        pattern_id="customer_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=10,
        delta=recomputed_delta,
        delta_norm=recomputed_norm,
        is_anomaly=False,
        delta_rank_pct=None,
        edges=[],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    engine = MagicMock()
    engine.build_polygon.return_value = built_polygon

    # Geometry dataset stores the authoritative (different) values
    stored_delta = [1.5857, -0.3210]
    stored_norm = float(np.linalg.norm(stored_delta))
    geo_table = pa.table(
        {
            "delta": pa.array([stored_delta], type=pa.list_(pa.float32())),
            "delta_norm": pa.array([stored_norm], type=pa.float64()),
            "is_anomaly": pa.array([True], type=pa.bool_()),
            "delta_rank_pct": pa.array([92.5], type=pa.float64()),
        }
    )

    storage = MagicMock()
    storage.read_geometry.return_value = geo_table

    manifest = MagicMock()
    manifest.pattern_version.return_value = 1

    nav = GDSNavigator(
        engine=engine,
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )
    nav._position = Point(
        primary_key="E-001",
        line_id="customers",
        version=1,
        status="active",
        properties={},
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        changed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    polygon = nav.current_polygon("customer_pattern")

    # Stored values must override recomputed ones
    assert polygon.delta == pytest.approx(stored_delta, abs=1e-4)
    assert polygon.delta_norm == pytest.approx(stored_norm, abs=1e-4)
    assert polygon.is_anomaly is True
    assert polygon.delta_rank_pct == pytest.approx(92.5, abs=1e-4)

    # read_geometry called with primary_key filter and correct columns
    storage.read_geometry.assert_called_once_with(
        "customer_pattern",
        1,
        primary_key="E-001",
        columns=["delta", "delta_norm", "is_anomaly", "delta_rank_pct"],
    )


def test_current_polygon_no_stored_row_keeps_built_values():
    """When geometry returns no rows (entity missing), built polygon is returned unchanged."""
    from unittest.mock import MagicMock

    recomputed_delta = np.array([0.05, 0.10], dtype=np.float32)
    built_polygon = Polygon(
        primary_key="E-NEW",
        pattern_id="customer_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=3,
        delta=recomputed_delta,
        delta_norm=float(np.linalg.norm(recomputed_delta)),
        is_anomaly=False,
        delta_rank_pct=None,
        edges=[],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    engine = MagicMock()
    engine.build_polygon.return_value = built_polygon

    empty_geo = pa.table(
        {
            "delta": pa.array([], type=pa.list_(pa.float32())),
            "delta_norm": pa.array([], type=pa.float64()),
            "is_anomaly": pa.array([], type=pa.bool_()),
            "delta_rank_pct": pa.array([], type=pa.float64()),
        }
    )

    storage = MagicMock()
    storage.read_geometry.return_value = empty_geo

    manifest = MagicMock()
    manifest.pattern_version.return_value = 1

    nav = GDSNavigator(
        engine=engine,
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )
    nav._position = Point(
        primary_key="E-NEW",
        line_id="customers",
        version=1,
        status="active",
        properties={},
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        changed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    polygon = nav.current_polygon("customer_pattern")

    # No stored row → built values unchanged
    assert polygon.delta == pytest.approx(recomputed_delta.tolist(), abs=1e-4)
    assert polygon.is_anomaly is False
    assert polygon.delta_rank_pct is None


def _make_π8_nav(n_entities: int = 12, n_relations: int = 2) -> GDSNavigator:
    from unittest.mock import MagicMock

    import pyarrow as pa
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef
    from hypertopos.storage.cache import GDSCache

    rng = np.random.default_rng(99)
    deltas = (
        np.vstack(
            [
                rng.normal([0.0, 0.0], 0.05, (n_entities // 2, n_relations)),
                rng.normal([1.0, 1.0], 0.05, (n_entities - n_entities // 2, n_relations)),
            ]
        )
        .astype(np.float32)
        .tolist()
    )

    table = pa.table(
        {
            "primary_key": [f"E-{i:03d}" for i in range(n_entities)],
            "scale": [1] * n_entities,
            "delta": deltas,
            "delta_norm": [float(np.linalg.norm(d)) for d in deltas],
            "is_anomaly": [False] * (n_entities // 2) + [True] * (n_entities - n_entities // 2),
            "delta_rank_pct": pa.array([50.0] * n_entities, type=pa.float64()),
            "edges": pa.array([[] for _ in range(n_entities)], type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": ["2024-01-01T00:00:00"] * n_entities,
            "updated_at": ["2024-01-01T00:00:00"] * n_entities,
        }
    )

    rels = [
        RelationDef(line_id=f"line_{i}", direction="out", required=True) for i in range(n_relations)
    ]  # noqa: E501
    pattern = Pattern(
        pattern_id="p",
        entity_type="entity",
        pattern_type="anchor",
        relations=rels,
        mu=np.zeros(n_relations, dtype=np.float32),
        sigma_diag=np.ones(n_relations, dtype=np.float32),  # noqa: E501
        theta=np.ones(n_relations, dtype=np.float32),
        population_size=n_entities,
        computed_at=None,
        version=1,
        status="production",
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()

    def _read_geometry_side_effect(*args, sample_size=None, **kwargs):
        if sample_size is not None and sample_size < table.num_rows:
            return table.slice(0, sample_size)
        return table

    storage.read_geometry.side_effect = _read_geometry_side_effect
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=storage, cache=GDSCache())
    return GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())


class TestPi8AttractCluster:
    def test_returns_n_clusters(self):
        nav = _make_π8_nav()
        clusters = nav.π8_attract_cluster("p", n_clusters=2)
        assert len(clusters) == 2

    def test_cluster_has_required_keys(self):
        nav = _make_π8_nav()
        clusters = nav.π8_attract_cluster("p", n_clusters=2)
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

    def test_member_keys_trimmed_to_top_n(self):
        nav = _make_π8_nav(n_entities=12)
        clusters = nav.π8_attract_cluster("p", n_clusters=2, top_n=3)
        for c in clusters:
            assert len(c["member_keys"]) <= 3

    def test_sample_size_reduces_coverage(self):
        nav = _make_π8_nav(n_entities=20)
        clusters = nav.π8_attract_cluster("p", n_clusters=2, top_n=100, sample_size=6)
        assert sum(c["size"] for c in clusters) == 6

    def test_empty_geometry_returns_empty(self):
        from unittest.mock import MagicMock

        import pyarrow as pa
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.navigation.navigator import GDSNavigator
        from hypertopos.storage.cache import GDSCache

        empty_table = pa.table(
            {
                "primary_key": pa.array([], type=pa.string()),
                "scale": pa.array([], type=pa.int64()),
                "delta": pa.array([], type=pa.list_(pa.float32())),
                "delta_norm": pa.array([], type=pa.float64()),
                "is_anomaly": pa.array([], type=pa.bool_()),
                "delta_rank_pct": pa.array([], type=pa.float64()),
                "edges": pa.array([], type=pa.list_(_EDGE_STRUCT)),
                "last_refresh_at": pa.array([], type=pa.string()),
                "updated_at": pa.array([], type=pa.string()),
            }
        )
        storage = MagicMock()
        storage.read_geometry.return_value = empty_table
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        engine = GDSEngine(storage=storage, cache=GDSCache())
        nav = GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())
        assert nav.π8_attract_cluster("p", n_clusters=3) == []


# --- prop-fill anomaly label ---


def _make_prop_pattern():
    """Pattern with K=1 relation + M=1 prop_column, delta_dim=2."""
    from hypertopos.model.sphere import Pattern, RelationDef

    return Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[RelationDef(line_id="orders", direction="out", required=True)],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.array([3.0, 3.0], dtype=np.float32),
        population_size=10,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        prop_columns=["country"],
    )


def test_classify_anomalies_prop_fill_label():
    """Anomaly caused only by missing prop fill must not be labelled 'excess edges'."""
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.objects import Polygon

    pattern = _make_prop_pattern()
    engine = GDSEngine(storage=None, cache=None)

    # edge dim normal (0.0), prop fill dim very negative (-5.0) → prop-fill anomaly
    poly = Polygon(
        primary_key="CUST-001",
        pattern_id="p",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.array([0.0, -5.0], dtype=np.float32),
        delta_norm=5.0,
        is_anomaly=True,
        edges=[],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    clusters = engine.classify_anomalies([poly], pattern)

    assert len(clusters) == 1
    label = clusters[0]["label"]
    assert "prop:country" in label
    assert label != "excess edges"


def test_anomaly_summary_prop_fill_label():
    """navigator.anomaly_summary cluster label must include prop:country for prop-fill anomaly."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.storage.cache import GDSCache

    pattern = _make_prop_pattern()
    # theta_norm = sqrt(9+9) ≈ 4.24; anomalous entity: delta=[0.0, -5.0], norm=5.0
    table = pa.table(
        {
            "primary_key": ["CUST-001", "CUST-002"],
            "delta": [[0.0, -5.0], [0.5, 0.5]],
            "delta_norm": [5.0, float(np.linalg.norm([0.5, 0.5]))],
            "is_anomaly": [True, False],
        }
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry_stats.return_value = None  # cache miss → full scan path
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=None, cache=GDSCache())
    nav = GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())

    summary = nav.anomaly_summary("p")

    assert summary["total_anomalies"] == 1
    assert len(summary["clusters"]) == 1
    label = summary["clusters"][0]["label"]
    assert "prop:country" in label
    assert label != "excess edges"


def test_anomaly_summary_cache_miss_boundary_included():
    """In cache-miss path, entity with delta_norm == theta_norm must be counted (>= not >)."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.storage.cache import GDSCache

    pattern = _make_prop_pattern()
    # theta_norm = sqrt(9+9) ≈ 4.2426; set boundary entity exactly at theta_norm
    theta_norm = float(np.linalg.norm(pattern.theta))
    table = pa.table(
        {
            "primary_key": ["below", "exact", "above"],
            "delta": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # unused in count path
            "delta_norm": [theta_norm * 0.5, theta_norm, theta_norm * 1.5],
            "is_anomaly": [False, True, True],
        }
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry_stats.return_value = None  # force cache-miss path
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=None, cache=GDSCache())
    nav = GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())

    summary = nav.anomaly_summary("p")

    assert summary["total_anomalies"] == 2, (
        f"Entity at delta_norm==theta_norm must be counted; got {summary['total_anomalies']}"
    )


def test_anomaly_summary_theta_zero_returns_zero_anomalies():
    """anomaly_summary cache-miss path must return 0 anomalies when theta_norm=0."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef
    from hypertopos.storage.cache import GDSCache

    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[RelationDef(line_id="orders", direction="out", required=True)],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.zeros(2, dtype=np.float32),  # theta_norm == 0
        population_size=5,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        prop_columns=["country"],
    )
    table = pa.table(
        {
            "primary_key": ["e1", "e2", "e3", "e4", "e5"],
            "delta": [[0.0, 0.0]] * 5,
            "delta_norm": [0.0, 0.0, 0.0, 0.0, 0.0],
            "is_anomaly": [False, False, False, False, False],
        }
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry_stats.return_value = None  # force cache-miss path
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=None, cache=GDSCache())
    nav = GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())

    summary = nav.anomaly_summary("p")

    assert summary["total_anomalies"] == 0, (
        f"theta_norm=0 must produce 0 anomalies; got {summary['total_anomalies']}"
    )
    assert summary["anomaly_rate"] == 0.0
    assert summary["clusters"] == []


# --- dim_names includes prop_columns in find_clusters output ---


def test_find_clusters_dim_profile_includes_prop_columns():
    """dim_profile in find_clusters must list prop_column names, not only relation names."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine

    pattern = _make_prop_pattern()
    now_str = "2025-01-01T00:00:00"
    # 4 entities, delta vectors of length K+M=2
    deltas = [[0.1, 0.9], [0.2, 0.8], [0.8, 0.1], [0.9, 0.2]]
    n = len(deltas)
    table = pa.table(
        {
            "primary_key": [f"C-{i}" for i in range(n)],
            "scale": [1] * n,
            "delta": deltas,
            "delta_norm": [float(np.linalg.norm(d)) for d in deltas],
            "is_anomaly": [False] * n,
            "edges": pa.array([[] for _ in range(n)], type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [now_str] * n,
            "updated_at": [now_str] * n,
        }
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=None, cache=None)
    nav = GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())

    clusters = nav.π8_attract_cluster("p", n_clusters=2, seed=0)

    assert len(clusters) > 0
    dim_names_in_profile = [e["dimension"] for e in clusters[0]["dim_profile"]]
    assert "orders" in dim_names_in_profile
    assert "country" in dim_names_in_profile
    assert len(dim_names_in_profile) == 2


# ---------- compare_entities_intraclass — direct key lookup, ANN-independent ----------


def test_compare_entities_intraclass_ignores_ann_index():
    """compare_entities_intraclass must read stored delta, not recompute from edges.

    The test verifies that even if the ANN index returned stale (zero) vectors,
    compare_entities_intraclass returns the correct non-zero distance by reading
    delta directly from the geometry table via primary_key BTREE lookup.
    """
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Contract, Manifest

    # Two customers with clearly distinct stored delta vectors
    delta_a = np.array([1.0, 0.0], dtype=np.float32)
    delta_b = np.array([0.0, 1.0], dtype=np.float32)
    expected_dist = float(np.linalg.norm(delta_a - delta_b))  # sqrt(2) ≈ 1.4142

    def _make_geo_row(pk: str, delta: np.ndarray) -> pa.Table:
        return pa.table(
            {
                "primary_key": pa.array([pk], type=pa.string()),
                "delta": pa.array([delta.tolist()], type=pa.list_(pa.float32())),
                "delta_norm": pa.array([float(np.linalg.norm(delta))], type=pa.float32()),
                "delta_rank_pct": pa.array([50.0], type=pa.float32()),
                "is_anomaly": pa.array([False], type=pa.bool_()),
            }
        )

    # Simulate ANN-stale storage: read_geometry returns correct per-key rows
    # (direct BTREE lookup), but the ANN index would have returned all-zero vectors.
    call_log: list[str] = []

    def _read_geometry(pid, ver, primary_key=None, columns=None, **kw):
        call_log.append(primary_key or "scan")
        if primary_key == "CUST-A":
            return _make_geo_row("CUST-A", delta_a)
        if primary_key == "CUST-B":
            return _make_geo_row("CUST-B", delta_b)
        return pa.table({})

    storage = MagicMock()
    storage.read_geometry.side_effect = _read_geometry

    manifest = Manifest(
        manifest_id="test",
        agent_id="a",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={},
        pattern_versions={"customer_pattern": 1},
    )
    contract = Contract("test", ["customer_pattern"])
    engine = GDSEngine(storage=None, cache=MagicMock())
    nav = GDSNavigator(
        engine=engine,
        storage=storage,
        manifest=manifest,
        contract=contract,
    )

    result = nav.compare_entities_intraclass("CUST-A", "CUST-B", "customer_pattern")

    # Verify correct distance from stored deltas (not recomputed from edges)
    assert result["distance"] == pytest.approx(expected_dist, abs=1e-4)
    assert result["delta_norm_a"] == pytest.approx(float(np.linalg.norm(delta_a)), abs=1e-4)
    assert result["delta_norm_b"] == pytest.approx(float(np.linalg.norm(delta_b)), abs=1e-4)
    assert result["is_anomaly_a"] is False
    assert result["is_anomaly_b"] is False

    # Both lookups must have used primary_key (BTREE path), not ANN
    assert "CUST-A" in call_log
    assert "CUST-B" in call_log


def test_compare_entities_intraclass_raises_for_missing_entity():
    """compare_entities_intraclass raises KeyError if an entity is not found."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Contract, Manifest

    storage = MagicMock()
    storage.read_geometry.return_value = pa.table({})  # empty = not found

    manifest = Manifest(
        manifest_id="test",
        agent_id="a",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={},
        pattern_versions={"customer_pattern": 1},
    )
    contract = Contract("test", ["customer_pattern"])
    engine = GDSEngine(storage=None, cache=MagicMock())
    nav = GDSNavigator(
        engine=engine,
        storage=storage,
        manifest=manifest,
        contract=contract,
    )

    with pytest.raises(KeyError, match="CUST-MISSING"):
        nav.compare_entities_intraclass("CUST-MISSING", "CUST-B", "customer_pattern")


# ---------------------------------------------------------------------------
# TestTemporalQualitySummary
# ---------------------------------------------------------------------------


class TestTemporalQualitySummary:
    """Tests for GDSNavigator.temporal_quality_summary()."""

    @staticmethod
    def _make_pattern(pattern_type="anchor", mu=None, sigma=None, theta=None):
        from hypertopos.model.sphere import Pattern, RelationDef

        return Pattern(
            pattern_id="test_pattern",
            entity_type="test",
            pattern_type=pattern_type,
            relations=[
                RelationDef("line_a", "in", True),
                RelationDef("line_b", "in", True),
            ],
            mu=mu if mu is not None else np.array([0.0, 0.0], dtype=np.float32),
            sigma_diag=sigma if sigma is not None else np.array([1.0, 1.0], dtype=np.float32),
            theta=theta if theta is not None else np.array([1.0, 1.0], dtype=np.float32),
            population_size=100,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )

    @staticmethod
    def _make_nav(pattern, temporal_table=None):
        storage = _MockStorageWithPatterns(
            {"test_pattern": pattern},
            temporal_table=temporal_table,
        )
        return GDSNavigator(
            _MockEngine(),
            storage,
            _make_manifest(),
            Contract("m-001", []),
        )

    def test_persistent_anomalies(self):
        """Entities with stable anomaly status → persistence_rate > 0.7."""
        pattern = self._make_pattern(
            mu=np.array([0.0, 0.0], dtype=np.float32),
            sigma=np.array([1.0, 1.0], dtype=np.float32),
            theta=np.array([1.0, 1.0], dtype=np.float32),  # theta_norm ≈ 1.414
        )
        # Two entities, each with 4 slices that are always anomalous
        # (delta_norm >> theta_norm).
        rows = []
        for pk in ["E-001", "E-002"]:
            for i in range(4):
                rows.append(
                    {
                        "pk": pk,
                        "idx": i,
                        "ts": datetime(2024, 1, 1 + i, tzinfo=UTC),
                        "delta": [5.0, 5.0],  # norm ≈ 7.07 >> theta_norm ≈ 1.414
                    }
                )
        temporal = _make_temporal_table(rows)
        nav = self._make_nav(pattern, temporal)

        result = nav.temporal_quality_summary("test_pattern")

        assert result is not None
        assert result["persistence_rate"] > 0.7
        assert result["signal_quality"] == "persistent"
        assert result["n_entities_sampled"] == 2
        assert result["n_anomaly_transitions"] > 0

    def test_volatile_anomalies(self):
        """Entities where anomaly status flips every slice → persistence_rate < 0.3."""
        pattern = self._make_pattern(
            mu=np.array([0.0, 0.0], dtype=np.float32),
            sigma=np.array([1.0, 1.0], dtype=np.float32),
            theta=np.array([2.0, 2.0], dtype=np.float32),  # theta_norm ≈ 2.828
        )
        # Two entities, alternating anomalous / normal slices
        rows = []
        for pk in ["E-001", "E-002"]:
            for i in range(6):
                delta = (
                    [5.0, 5.0] if i % 2 == 0 else [0.1, 0.1]
                )  # anomalous vs normal
                rows.append(
                    {
                        "pk": pk,
                        "idx": i,
                        "ts": datetime(2024, 1, 1 + i, tzinfo=UTC),
                        "delta": delta,
                    }
                )
        temporal = _make_temporal_table(rows)
        nav = self._make_nav(pattern, temporal)

        result = nav.temporal_quality_summary("test_pattern")

        assert result is not None
        assert result["persistence_rate"] < 0.3
        assert result["signal_quality"] == "volatile"
        assert result["n_entities_sampled"] == 2

    def test_event_pattern_returns_none(self):
        """Event patterns should return None."""
        pattern = self._make_pattern(pattern_type="event")
        nav = self._make_nav(pattern)

        result = nav.temporal_quality_summary("test_pattern")

        assert result is None

    def test_no_temporal_data_returns_none(self):
        """Empty temporal table → None."""
        pattern = self._make_pattern()
        temporal = _make_temporal_table([])
        nav = self._make_nav(pattern, temporal)

        result = nav.temporal_quality_summary("test_pattern")

        assert result is None

    def test_returns_expected_keys(self):
        """Result dict contains all expected keys."""
        pattern = self._make_pattern(
            mu=np.array([0.0, 0.0], dtype=np.float32),
            sigma=np.array([1.0, 1.0], dtype=np.float32),
            theta=np.array([1.0, 1.0], dtype=np.float32),
        )
        rows = []
        for pk in ["E-001"]:
            for i in range(3):
                rows.append(
                    {
                        "pk": pk,
                        "idx": i,
                        "ts": datetime(2024, 1, 1 + i, tzinfo=UTC),
                        "delta": [3.0, 3.0],
                    }
                )
        temporal = _make_temporal_table(rows)
        nav = self._make_nav(pattern, temporal)

        result = nav.temporal_quality_summary("test_pattern")

        assert result is not None
        expected_keys = {
            "persistence_rate",
            "transition_rate",
            "signal_quality",
            "n_entities_sampled",
            "n_anomaly_transitions",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# is_jumpable=False for continuous-mode edges in all polygon paths
# ---------------------------------------------------------------------------


def _make_continuous_mode_nav(n: int = 3):
    """Build a navigator where the pattern uses continuous mode (edge_max set).

    Edges have empty point_key ('') as is the case in continuous mode.
    """
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef

    now_str = "2025-01-01T00:00:00"
    theta = np.array([0.3, 0.3], dtype=np.float32)
    theta_norm = float(np.linalg.norm(theta))

    # Continuous-mode edges: point_key is always empty string
    cont_edges = [
        {"line_id": "products", "point_key": "", "status": "alive", "direction": "out"},
        {"line_id": "stores", "point_key": "", "status": "alive", "direction": "in"},
    ]
    table = pa.table(
        {
            "primary_key": [f"E-{i:03d}" for i in range(n)],
            "scale": [1] * n,
            "delta": [[0.5, 0.4]] * n,
            "delta_norm": [float(np.linalg.norm([0.5, 0.4]))] * n,
            "is_anomaly": [True] * n,
            "delta_rank_pct": [95.0] * n,
            "edges": pa.array([cont_edges] * n, type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [now_str] * n,
            "updated_at": [now_str] * n,
        }
    )
    pattern = Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.array([0.5, 0.5], dtype=np.float32),
        sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
        theta=theta,
        population_size=n,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=np.array([5.0, 3.0], dtype=np.float32),  # continuous mode
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=None, cache=None)
    nav = GDSNavigator(engine=engine, storage=storage, manifest=manifest, contract=MagicMock())
    nav._last_theta_norm = theta_norm
    return nav


class TestContinuousModeIsJumpable:
    def test_pi5_attract_anomaly_continuous_mode_not_jumpable(self):
        """π5_attract_anomaly: continuous-mode edges must have is_jumpable=False."""
        nav = _make_continuous_mode_nav(n=3)
        nav._storage.read_sphere.return_value.patterns["p"].theta = np.array(
            [0.1, 0.1], dtype=np.float32
        )
        polygons, _, _, _ = nav.π5_attract_anomaly("p", top_n=3)
        assert len(polygons) > 0
        for polygon in polygons:
            for edge in polygon.edges:
                assert not edge.is_jumpable, (
                    f"Edge to '{edge.line_id}' with point_key='{edge.point_key}' "
                    f"must have is_jumpable=False in continuous mode"
                )

    def test_event_polygons_for_continuous_mode_not_jumpable(self):
        """event_polygons_for: continuous-mode edges must have is_jumpable=False."""
        nav = _make_continuous_mode_nav(n=3)
        polygons = nav.event_polygons_for("", "p")
        assert len(polygons) > 0
        for polygon in polygons:
            for edge in polygon.edges:
                assert not edge.is_jumpable, (
                    f"Edge to '{edge.line_id}' with point_key='{edge.point_key}' "
                    f"must have is_jumpable=False in continuous mode"
                )


class TestFindAnomaliesTable:
    def test_total_found_and_page0(self):
        """total_found = rows above threshold; offset=0 returns top rows."""
        table = pa.table(
            {
                "primary_key": ["A", "B", "C", "D"],
                "delta_norm": [5.0, 4.0, 3.0, 1.0],
            }
        )
        result = _find_anomalies_table(table, threshold=2.5, top_n=2, offset=0)
        assert result["total_found"] == 3  # A, B, C pass threshold
        assert result["keys"] == ["A", "B"]
        assert result["delta_norms"] == [5.0, 4.0]

    def test_offset_skips_rows(self):
        """offset=1 skips rank-1, returns ranks 2 and 3."""
        table = pa.table(
            {
                "primary_key": ["A", "B", "C", "D"],
                "delta_norm": [5.0, 4.0, 3.0, 1.0],
            }
        )
        result = _find_anomalies_table(table, threshold=2.5, top_n=2, offset=1)
        assert result["total_found"] == 3
        assert result["keys"] == ["B", "C"]
        assert result["delta_norms"] == [4.0, 3.0]

    def test_offset_beyond_total_returns_empty(self):
        """offset >= total_found returns empty keys with correct total_found."""
        table = pa.table(
            {
                "primary_key": ["A", "B"],
                "delta_norm": [5.0, 4.0],
            }
        )
        result = _find_anomalies_table(table, threshold=2.5, top_n=2, offset=5)
        assert result["total_found"] == 2
        assert result["keys"] == []
        assert result["delta_norms"] == []


class TestPi5Offset:
    def test_pi5_returns_tuple(self, sphere_path):
        """π5 returns (polygons, total_found) tuple."""
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )
        result = nav.π5_attract_anomaly("customer_pattern", radius=None, top_n=10)
        assert isinstance(result, tuple) and len(result) == 4
        polygons, total_found, emerging, meta = result
        assert isinstance(polygons, list)
        assert isinstance(total_found, int)
        assert total_found >= len(polygons)

    def test_pi5_offset_skips_top_results(self, sphere_path):
        """offset=1 skips the single anomaly, returns empty list but total_found=1."""
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )
        # fixture has 1 anomaly; offset=1 skips it → empty page, total_found=1
        polygons, total_found, _, _ = nav.π5_attract_anomaly(
            "customer_pattern", radius=None, top_n=10, offset=1
        )
        assert polygons == []
        assert total_found == 1


# ---------- dead_dim_indices ----------


class _MockStorageDeadDim(_MockStorage):
    """Mock storage returning geometry with known dead dimensions."""

    def __init__(self, deltas: list[list[float]]):
        self._deltas = deltas

    def read_geometry(self, pattern_id, version, **kw):
        return pa.table({"delta": self._deltas})

    def read_sphere(self):
        from hypertopos.model.sphere import Pattern, RelationDef

        pat = Pattern(
            pattern_id="p1",
            entity_type="test",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="a", direction="in", required=True),
                RelationDef(line_id="b", direction="in", required=True),
                RelationDef(line_id="c", direction="in", required=True),
            ],
            mu=np.zeros(3, dtype=np.float32),
            sigma_diag=np.ones(3, dtype=np.float32),
            theta=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            population_size=10,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = _MockSphereWithPatterns({"p1": pat})
        return sphere


def _make_dead_dim_nav(deltas):
    """Helper to build a navigator with a DeadDim mock storage."""
    storage = _MockStorageDeadDim(deltas)
    return GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=Manifest(
            manifest_id="m",
            agent_id="a",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"test": 1},
            pattern_versions={"p1": 1},
        ),
        contract=Contract("m", ["p1"]),
    )


def test_dead_dim_indices_basic():
    """Dimensions with zero variance are flagged as dead."""
    # dim0 varies, dim1 is constant (0.0), dim2 is constant (1.0)
    deltas = [[float(i), 0.0, 1.0] for i in range(20)]
    nav = _make_dead_dim_nav(deltas)
    dead = nav.dead_dim_indices("p1")
    # dim1 and dim2 have zero variance
    assert 1 in dead
    assert 2 in dead
    # dim0 has variance > 0.01
    assert 0 not in dead


def test_dead_dim_indices_cached():
    """Second call returns cached result without re-reading geometry."""
    deltas = [[1.0, 0.0, 0.0] for _ in range(10)]
    nav = _make_dead_dim_nav(deltas)

    result1 = nav.dead_dim_indices("p1")
    # Sabotage the storage to verify caching
    nav._storage._deltas = [[99.0, 99.0, 99.0] for _ in range(10)]
    result2 = nav.dead_dim_indices("p1")
    assert result1 == result2  # same object from cache


# ---------- anomaly_summary top_driving_dimensions ----------


def test_anomaly_summary_top_driving_dimensions():
    """anomaly_summary includes top_driving_dimensions from cluster deltas."""
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef
    from hypertopos.storage.cache import GDSCache

    n = 10
    pattern = Pattern(
        pattern_id="p1",
        entity_type="items",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="dim_a", direction="in", required=True),
            RelationDef(line_id="dim_b", direction="in", required=True),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.array([1.0, 1.0], dtype=np.float32),
        population_size=n,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    dt = datetime(2024, 1, 1, tzinfo=UTC)
    # All anomalies have delta [3.0, 1.0] — dim_a should dominate
    deltas = [[3.0, 1.0]] * n
    norms = [float(np.linalg.norm(d)) for d in deltas]
    geo = pa.table(
        {
            "primary_key": [f"I-{i:03d}" for i in range(n)],
            "scale": [1] * n,
            "delta": deltas,
            "delta_norm": norms,
            "is_anomaly": [True] * n,
            "delta_rank_pct": pa.array([90.0] * n, type=pa.float64()),
            "edges": pa.array([[]] * n, type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [dt] * n,
            "updated_at": [dt] * n,
        }
    )
    storage = _MockStorageWithPatterns({"p1": pattern}, geometry_table=geo)
    nav = GDSNavigator(
        engine=GDSEngine(storage=storage, cache=GDSCache()),
        storage=storage,
        manifest=Manifest(
            manifest_id="m",
            agent_id="a",
            snapshot_time=dt,
            status="active",
            line_versions={"items": 1},
            pattern_versions={"p1": 1},
        ),
        contract=Contract("m", ["p1"]),
    )
    summary = nav.anomaly_summary("p1")
    assert "top_driving_dimensions" in summary
    top = summary["top_driving_dimensions"]
    assert len(top) >= 1
    # dim_a (index 0, delta=3.0) contributes 90% vs dim_b (delta=1.0) at 10%
    assert top[0]["dim"] == 0
    assert top[0]["label"] == "dim_a"
    assert top[0]["mean_contribution_pct"] == 90.0


def test_anomaly_summary_count_matches_find_anomalies():
    """anomaly_summary.total_anomalies must be consistent with π5 total_found."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.sphere import Pattern, RelationDef
    from hypertopos.storage.cache import GDSCache

    n = 20
    pattern = Pattern(
        pattern_id="p",
        entity_type="items",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="a", direction="in", required=True),
            RelationDef(line_id="b", direction="in", required=True),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.array([1.5, 1.5], dtype=np.float32),
        population_size=n,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    theta_norm = float(np.linalg.norm(pattern.theta))
    rng = np.random.default_rng(42)
    deltas = rng.normal(0, 1.5, size=(n, 2)).astype(np.float32)
    norms = np.sqrt(np.einsum("ij,ij->i", deltas, deltas)).astype(np.float32)
    is_anomaly = norms >= theta_norm
    dt = datetime(2024, 1, 1, tzinfo=UTC)

    geo = pa.table(
        {
            "primary_key": [f"I-{i:03d}" for i in range(n)],
            "scale": [1] * n,
            "delta": deltas.tolist(),
            "delta_norm": norms.tolist(),
            "is_anomaly": is_anomaly.tolist(),
            "delta_rank_pct": pa.array([50.0] * n, type=pa.float64()),
            "edges": pa.array([[]] * n, type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [dt] * n,
            "updated_at": [dt] * n,
        }
    )

    # Build a stats cache that intentionally diverges (simulates stale cache)
    stale_cache = {
        "total_entities": n,
        "total_anomalies": 0,  # wrong on purpose
        "theta_norm": theta_norm,
        "percentiles": {
            "p50": 1.0, "p75": 1.5, "p90": 2.0,
            "p95": 2.5, "p99": 3.0, "max": 4.0,
        },
    }
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    storage.read_geometry_stats.return_value = stale_cache
    storage.read_geometry.return_value = geo

    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    engine = GDSEngine(storage=None, cache=GDSCache())
    nav = GDSNavigator(
        engine=engine, storage=storage, manifest=manifest, contract=MagicMock(),
    )

    summary = nav.anomaly_summary("p")
    expected = int((norms >= theta_norm).sum())
    assert summary["total_anomalies"] == expected
    assert summary["total_anomalies"] != stale_cache["total_anomalies"]

    _, total_found, _, _ = nav.π5_attract_anomaly("p", radius=None, top_n=n)
    assert abs(summary["total_anomalies"] - total_found) <= 2


# ---------- sphere_overview inactive_ratio ----------


class _MockStorageOverview(_MockStorage):
    """Mock storage returning delta_norm for sphere_overview inactive_ratio test."""

    def __init__(self, norms: list[float], pattern_type: str = "anchor"):
        self._norms = norms
        self._pattern_type = pattern_type

    def read_geometry(self, pattern_id, version, **kw):
        columns = kw.get("columns")
        if columns and "delta_norm" in columns:
            return pa.table({"delta_norm": self._norms})
        return pa.table({})

    def read_geometry_stats(self, *a, **kw):
        stats = {"total_entities": len(self._norms), "total_anomalies": 0}
        # Pre-compute inactive_ratio to match writer.write_geometry_stats
        n = len(self._norms)
        if n > 0:
            arr = np.array(self._norms, dtype=np.float32)
            counts, bin_edges = np.histogram(arr, bins=50)
            mode_bin = int(np.argmax(counts))
            mode_count = int(counts[mode_bin])
            mode_center = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2)
            median_val = float(np.median(arr))
            if mode_count / n > 0.25 and mode_center < median_val:
                threshold = float(bin_edges[mode_bin + 1])
                inactive_count = int(np.sum(arr <= threshold))
                stats["inactive_ratio"] = round(inactive_count / n, 4)
                stats["inactive_count"] = inactive_count
        return stats

    def count_geometry_rows(self, pattern_id=None, version=None, filter=None):
        if filter and "delta_norm >=" in filter:
            threshold = float(filter.split(">=")[1].strip())
            return int(sum(1 for n in self._norms if n >= threshold))
        return len(self._norms)

    def read_sphere(self):
        from hypertopos.model.sphere import Pattern, RelationDef

        pat = Pattern(
            pattern_id="p1",
            entity_type="test",
            pattern_type=self._pattern_type,
            relations=[
                RelationDef(line_id="a", direction="in", required=True),
            ],
            mu=np.zeros(1, dtype=np.float32),
            sigma_diag=np.ones(1, dtype=np.float32),
            theta=np.array([2.0], dtype=np.float32),
            population_size=len(self._norms),
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        sphere = _MockSphereWithPatterns({"p1": pat})
        return sphere


def test_sphere_overview_inactive_ratio():
    """sphere_overview computes inactive_ratio for bimodal delta_norm distribution."""
    # Bimodal distribution: 30 entities near 0.1, 70 entities spread over 2.0-8.0
    # Mode bin near 0.1 should contain >25% and be well below the median (~3.0)
    # → triggers inactive_ratio
    rng = np.random.default_rng(42)
    norms = [0.1] * 30 + [float(v) for v in rng.uniform(2.0, 8.0, size=70)]
    storage = _MockStorageOverview(norms, pattern_type="anchor")
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=Manifest(
            manifest_id="m",
            agent_id="a",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"test": 1},
            pattern_versions={"p1": 1},
        ),
        contract=Contract("m", ["p1"]),
    )
    result = nav.sphere_overview("p1")
    assert len(result) == 1
    entry = result[0]
    assert "inactive_ratio" in entry
    # 30 out of 100 are in the low-norm mode bin
    assert entry["inactive_ratio"] >= 0.2


def test_sphere_overview_anomaly_rate_source():
    """sphere_overview includes anomaly_rate_source field."""
    norms = [float(i) * 0.1 for i in range(100)]
    storage = _MockStorageOverview(norms, pattern_type="anchor")
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=Manifest(
            manifest_id="m",
            agent_id="a",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"test": 1},
            pattern_versions={"p1": 1},
        ),
        contract=Contract("m", ["p1"]),
    )
    result = nav.sphere_overview("p1")
    assert len(result) == 1
    assert "anomaly_rate_source" in result[0]
    assert result[0]["anomaly_rate_source"] == "delta_norm_scan"


def test_check_alerts_high_anomaly_rate(fixture_sphere_path):
    """check_alerts fires high_anomaly_rate when >20% anomalies."""
    import json
    import uuid

    sphere_json_path = fixture_sphere_path / "_gds_meta" / "sphere.json"
    sphere_data = json.loads(sphere_json_path.read_text())

    pid = next(
        (k for k, v in sphere_data["patterns"].items() if v.get("pattern_type") == "anchor"),
        None,
    )
    if pid is None:
        pytest.skip("No anchor pattern in fixture")

    # Write fake geometry_stats with >20% anomaly rate
    stats_dir = fixture_sphere_path / "_gds_meta" / "geometry_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / f"{pid}_v1.json"
    stats_file.write_text(
        json.dumps(
            {
                "pattern_id": pid,
                "version": 1,
                "theta_norm": 1.0,
                "total_entities": 10000,
                "total_anomalies": 5000,
                "percentiles": {
                    "p50": 0.5,
                    "p75": 1.0,
                    "p90": 1.5,
                    "p95": 2.0,
                    "p99": 3.0,
                    "max": 5.0,
                },
            }
        )
    )

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Contract, Manifest
    from hypertopos.storage.cache import GDSCache
    from hypertopos.storage.reader import GDSReader

    reader = GDSReader(str(fixture_sphere_path))
    sphere = reader.read_sphere()
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id=str(uuid.uuid4()),
        agent_id="test",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={
            lid: line.current_version() for lid, line in sphere.lines.items()
        },
        pattern_versions={p: pat.version for p, pat in sphere.patterns.items()},
    )
    nav = GDSNavigator(
        engine=engine,
        storage=reader,
        manifest=manifest,
        contract=Contract(manifest.manifest_id, list(sphere.patterns.keys())),
    )
    result = nav.check_alerts(pid)
    alerts = result["alerts"]
    high_rate = [a for a in alerts if a.get("check_type") == "theta_miscalibration"]
    assert len(high_rate) >= 1
    findings = high_rate[0].get("details", {}).get("findings", [])
    has_high = any(f.get("issue_type") == "high_anomaly_rate" for f in findings)
    assert has_high, f"Expected high_anomaly_rate finding, got: {findings}"


# ---------- π9 drift slice_window_days ----------


class _MockStorageDrift(_MockStorage):
    """Mock storage for π9 drift tests with shape_snapshot temporal data."""

    def __init__(self, pattern, temporal_rows):
        self._pattern = pattern
        self._temporal_rows = temporal_rows

    def read_geometry(self, pattern_id, version, **kw):
        # Return geometry with primary_key and delta
        pks = list({r["pk"] for r in self._temporal_rows})
        d = len(self._pattern.mu)
        return pa.table(
            {
                "primary_key": pks,
                "delta": [[0.0] * d for _ in pks],
            }
        )

    def read_sphere(self):
        sphere = _MockSphereWithPatterns({self._pattern.pattern_id: self._pattern})
        return sphere

    def read_temporal_batched(
        self,
        pattern_id,
        batch_size=65_536,
        timestamp_from=None,
        timestamp_to=None,
        keys=None,
    ):
        import pyarrow as _pa

        rows = self._temporal_rows
        if keys is not None:
            key_set = set(keys)
            rows = [r for r in rows if r["pk"] in key_set]
        if not rows:
            return
        table = _pa.table(
            {
                "primary_key": [r["pk"] for r in rows],
                "timestamp": [r["ts"] for r in rows],
                "shape_snapshot": [r["shape"] for r in rows],
                "slice_index": list(range(len(rows))),
                "deformation_type": ["internal"] * len(rows),
            }
        )
        yield table.to_batches()[0]

    def _apply_temporal_filters(self, table, filters):
        return table


def test_drift_slice_window_days():
    """π9 drift results include slice_window_days computed from timestamps."""
    from hypertopos.model.sphere import Pattern, RelationDef

    d = 2
    pattern = Pattern(
        pattern_id="p1",
        entity_type="test",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="a", direction="in", required=True),
            RelationDef(line_id="b", direction="in", required=True),
        ],
        mu=np.zeros(d, dtype=np.float32),
        sigma_diag=np.ones(d, dtype=np.float32),
        theta=np.array([2.0, 2.0], dtype=np.float32),
        population_size=5,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    # Entity with 3 slices spanning 90 days
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 2, 1, tzinfo=UTC)
    ts3 = datetime(2024, 4, 1, tzinfo=UTC)
    temporal_rows = [
        {"pk": "E-001", "ts": ts1, "shape": [1.0, 0.0]},
        {"pk": "E-001", "ts": ts2, "shape": [2.0, 0.0]},
        {"pk": "E-001", "ts": ts3, "shape": [5.0, 0.0]},
    ]
    storage = _MockStorageDrift(pattern, temporal_rows)
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=Manifest(
            manifest_id="m",
            agent_id="a",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"test": 1},
            pattern_versions={"p1": 1},
        ),
        contract=Contract("m", ["p1"]),
    )
    results = nav.π9_attract_drift("p1", top_n=10)
    assert len(results) == 1
    entry = results[0]
    assert "slice_window_days" in entry
    # Jan 1 to Apr 1 = 91 days
    assert entry["slice_window_days"] == 91


# ---------------------------------------------------------------------------
# B6: π7_attract_hub_and_stats — hub_score_pct and max_hub_score
# ---------------------------------------------------------------------------


def test_hub_and_stats_max_hub_score():
    """π7_attract_hub_and_stats returns max_hub_score in stats and hub_score_pct per entity."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Pattern, RelationDef

    mu = np.array([0.5, 0.5], dtype=np.float32)
    table = pa.table(
        {
            "primary_key": ["K-001", "K-002"],
            "delta": [[0.5, -0.5], [-1.5, -1.5]],
        }
    )
    edge_max = np.array([5, 5], dtype=np.float32)
    pattern = Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="customers", direction="out", required=True),
            RelationDef(line_id="products", direction="out", required=False),
        ],
        mu=mu,
        sigma_diag=np.array([0.2, 0.2], dtype=np.float32),
        theta=np.array([1.5, 1.5], dtype=np.float32),
        population_size=2,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=edge_max,
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )

    results, stats = nav.π7_attract_hub_and_stats("p", top_n=2)

    # max_hub_score = sum(edge_max) = 10
    assert stats["max_hub_score"] == pytest.approx(10.0, abs=0.01)
    # Each result tuple has 4 elements: (key, count, score, pct)
    for _key, _count, score, pct in results:
        assert isinstance(pct, float)
        assert pct == pytest.approx(score / 10.0 * 100, abs=0.5)


def test_hub_and_stats_binary_mode_pct_is_none():
    """In binary mode (no edge_max), hub_score_pct is None."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Pattern, RelationDef

    now_str = "2025-01-01T00:00:00"
    edges = [{"line_id": "c", "point_key": "C-1", "status": "alive", "direction": "out"}]
    table = pa.table(
        {
            "primary_key": ["K-001"],
            "scale": ["full"],
            "delta": [[0.5, -0.5]],
            "delta_norm": [0.707],
            "is_anomaly": [False],
            "edges": pa.array([edges], type=pa.list_(_EDGE_STRUCT)),
            "last_refresh_at": [now_str],
            "updated_at": [now_str],
        }
    )
    pattern = Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="c", direction="out", required=True),
        ],
        mu=np.array([0.5], dtype=np.float32),
        sigma_diag=np.array([0.2], dtype=np.float32),
        theta=np.array([1.5], dtype=np.float32),
        population_size=1,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=None,
    )
    sphere = MagicMock()
    sphere.patterns = {"p": pattern}
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )

    results, stats = nav.π7_attract_hub_and_stats("p", top_n=1)

    assert stats["max_hub_score"] is None
    # Binary mode: pct is None
    assert results[0][3] is None


# ---------------------------------------------------------------------------
# B8: suggest_grouping_properties
# ---------------------------------------------------------------------------


def test_suggest_grouping_properties():
    """suggest_grouping_properties returns only string columns, skipping system fields."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Line, PartitionConfig, Sphere

    points_table = pa.table(
        {
            "primary_key": ["E-001"],
            "version": [1],
            "status": ["active"],
            "created_at": [datetime(2024, 1, 1, tzinfo=UTC)],
            "changed_at": [datetime(2024, 1, 1, tzinfo=UTC)],
            "region": ["EMEA"],
            "country": ["PL"],
            "score": [42.0],
        }
    )
    sphere = Sphere("s", "s", ".")
    sphere.lines["customers"] = Line(
        line_id="customers",
        entity_type="customer",
        line_role="anchor",
        versions=[1],
        pattern_id="cust_pat",
        partitioning=PartitionConfig(mode="static", columns=[]),
    )
    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    storage.read_points.return_value = points_table
    manifest = MagicMock()
    manifest.line_version.return_value = 1
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )

    result = nav.suggest_grouping_properties("cust_pat")

    assert "region" in result
    assert "country" in result
    # Float and system columns excluded
    assert "score" not in result
    assert "primary_key" not in result
    assert "version" not in result


# ---------------------------------------------------------------------------
# B13: temporal_hint
# ---------------------------------------------------------------------------


def test_temporal_hint():
    """temporal_hint returns num_slices and last_timestamp from temporal table."""
    from unittest.mock import MagicMock

    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 6, 15, tzinfo=UTC)
    temporal_table = pa.table(
        {
            "primary_key": ["E-001", "E-001"],
            "timestamp": [ts1, ts2],
            "slice_index": [0, 1],
        }
    )
    storage = MagicMock()
    storage.read_temporal.return_value = temporal_table
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )

    result = nav.temporal_hint("E-001", "p1")

    assert result is not None
    assert result["num_slices"] == 2
    assert "2024-06-15" in result["last_timestamp"]


def test_temporal_hint_empty_returns_none():
    """temporal_hint returns None when no temporal data exists."""
    from unittest.mock import MagicMock

    storage = MagicMock()
    storage.read_temporal.return_value = pa.table({})
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=MagicMock(),
        contract=MagicMock(),
    )
    assert nav.temporal_hint("E-001", "p1") is None


def test_solid_reputation_propagates_unexpected_runtime_error():
    """solid_reputation should not swallow unexpected runtime failures."""
    from unittest.mock import MagicMock

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.storage.cache import GDSCache

    pattern = MagicMock()
    pattern.pattern_type = "anchor"
    pattern.theta = np.array([1.0], dtype=np.float32)
    sphere = MagicMock()
    sphere.patterns = {"p1": pattern}
    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    storage.read_temporal.side_effect = RuntimeError("boom")
    nav = GDSNavigator(
        engine=GDSEngine(storage=storage, cache=GDSCache()),
        storage=storage,
        manifest=MagicMock(),
        contract=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        nav.solid_reputation("E-001", "p1")


# ---------------------------------------------------------------------------
# B14: search_entities
# ---------------------------------------------------------------------------


def test_search_entities_basic():
    """search_entities filters by property value and returns correct structure."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Line, PartitionConfig, Sphere

    points_table = pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003"],
            "version": [1, 1, 1],
            "status": ["active", "active", "active"],
            "created_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
            "changed_at": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
            "region": ["EMEA", "APAC", "EMEA"],
        }
    )
    sphere = Sphere("s", "s", ".")
    sphere.lines["customers"] = Line(
        line_id="customers",
        entity_type="customer",
        line_role="anchor",
        versions=[1],
        pattern_id="cust_pat",
        partitioning=PartitionConfig(mode="static", columns=[]),
    )
    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    storage.read_points.return_value = points_table
    manifest = MagicMock()
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )

    result = nav.search_entities("customers", "region", "EMEA")

    assert result["total"] == 2
    assert result["returned"] == 2
    assert len(result["entities"]) == 2
    pks = {e["primary_key"] for e in result["entities"]}
    assert pks == {"E-001", "E-003"}


def test_search_entities_missing_line_raises():
    """search_entities raises GDSNavigationError for unknown line."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Sphere

    sphere = Sphere("s", "s", ".")
    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )

    with pytest.raises(GDSNavigationError, match="not found"):
        nav.search_entities("nonexistent", "region", "EMEA")


def test_search_entities_missing_property_raises():
    """search_entities raises GDSNavigationError for unknown property."""
    from unittest.mock import MagicMock

    from hypertopos.model.sphere import Line, PartitionConfig, Sphere

    points_table = pa.table(
        {
            "primary_key": ["E-001"],
            "version": [1],
            "status": ["active"],
            "created_at": [datetime(2024, 1, 1, tzinfo=UTC)],
            "changed_at": [datetime(2024, 1, 1, tzinfo=UTC)],
        }
    )
    sphere = Sphere("s", "s", ".")
    sphere.lines["customers"] = Line(
        line_id="customers",
        entity_type="customer",
        line_role="anchor",
        versions=[1],
        pattern_id="cust_pat",
        partitioning=PartitionConfig(mode="static", columns=[]),
    )
    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    storage.read_points.return_value = points_table
    manifest = MagicMock()
    nav = GDSNavigator(
        engine=MagicMock(),
        storage=storage,
        manifest=manifest,
        contract=MagicMock(),
    )

    with pytest.raises(GDSNavigationError, match="not found"):
        nav.search_entities("customers", "nonexistent_prop", "val")


# ---------------------------------------------------------------------------
# B16: compare_entities_intraclass — interpretation field
# ---------------------------------------------------------------------------


def test_compare_entities_interpretation():
    """compare_entities_intraclass returns interpretation string for known distances."""
    from unittest.mock import MagicMock

    def _make_result(dist: float) -> dict:
        """Build a compare_entities_intraclass result with a given distance."""
        delta_a = np.array([dist / np.sqrt(2), dist / np.sqrt(2)], dtype=np.float32)
        delta_b = np.zeros(2, dtype=np.float32)

        def _read_geometry(pid, ver, primary_key=None, columns=None, **kw):
            if primary_key == "A":
                return pa.table(
                    {
                        "primary_key": ["A"],
                        "delta": [delta_a.tolist()],
                        "delta_norm": [float(np.linalg.norm(delta_a))],
                        "delta_rank_pct": [50.0],
                        "is_anomaly": [False],
                    }
                )
            return pa.table(
                {
                    "primary_key": ["B"],
                    "delta": [delta_b.tolist()],
                    "delta_norm": [0.0],
                    "delta_rank_pct": [10.0],
                    "is_anomaly": [False],
                }
            )

        storage = MagicMock()
        storage.read_geometry.side_effect = _read_geometry
        manifest = MagicMock()
        manifest.pattern_version.return_value = 1
        nav = GDSNavigator(
            engine=MagicMock(),
            storage=storage,
            manifest=manifest,
            contract=MagicMock(),
        )
        return nav.compare_entities_intraclass("A", "B", "p")

    # identical shapes (distance == 0)
    r0 = _make_result(0.0)
    assert r0["interpretation"] == "identical shapes"

    # very similar (distance < 0.1)
    r1 = _make_result(0.05)
    assert r1["interpretation"] == "very similar"

    # similar (0.1 <= distance < 0.5)
    r2 = _make_result(0.3)
    assert r2["interpretation"] == "similar"

    # moderately different (0.5 <= distance < 1.0)
    r3 = _make_result(0.7)
    assert r3["interpretation"] == "moderately different"

    # very different (distance >= 1.0)
    r4 = _make_result(2.0)
    assert r4["interpretation"] == "very different"


# ---------------------------------------------------------------------------
# find_neighborhood — BFS through jumpable polygon edges
# ---------------------------------------------------------------------------


class _NeighborhoodMockStorage(_MockStorage):
    """Mock storage returning edges and anomaly info per entity for BFS tests."""

    def __init__(
        self,
        graph: dict[str, list[dict]],
        anomaly_map: dict[str, bool] | None = None,
        rank_map: dict[str, float] | None = None,
    ):
        super().__init__()
        self._graph = graph  # key -> list of edge dicts
        self._anomaly_map = anomaly_map or {}
        self._rank_map = rank_map or {}

    def read_sphere(self):
        from hypertopos.model.sphere import Pattern, RelationDef, Sphere

        pat = Pattern(
            pattern_id="test_pattern",
            entity_type="test",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="line1", direction="in", required=True),
            ],
            mu=np.zeros(1, dtype=np.float32),
            sigma_diag=np.ones(1, dtype=np.float32),
            theta=np.array([3.0], dtype=np.float32),
            population_size=10,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
            version=1,
            status="production",
        )
        return Sphere("s", "s", ".", patterns={"test_pattern": pat})

    def read_geometry(
        self,
        pattern_id,
        version,
        primary_key=None,
        filters=None,
        point_keys=None,
        columns=None,
        filter=None,
    ):
        if primary_key is None or primary_key not in self._graph:
            return pa.table({"primary_key": pa.array([], type=pa.string())})
        edges_py = self._graph[primary_key]
        edge_arr = pa.array([edges_py], type=pa.list_(_EDGE_STRUCT))
        is_anom = self._anomaly_map.get(primary_key, False)
        rank = self._rank_map.get(primary_key)
        return pa.table(
            {
                "primary_key": [primary_key],
                "edges": edge_arr,
                "is_anomaly": [is_anom],
                "delta_rank_pct": pa.array(
                    [rank],
                    type=pa.float64(),
                ),
            }
        )


class _NeighborhoodMockManifest:
    def pattern_version(self, pid):
        return 1

    def line_version(self, lid):
        return 1


def _make_edge(line_id: str, point_key: str, status: str = "alive", direction: str = "in") -> dict:
    return {
        "line_id": line_id,
        "point_key": point_key,
        "status": status,
        "direction": direction,
    }


def test_find_neighborhood_happy_path():
    """BFS finds entities at hop 1 and hop 2."""
    graph = {
        "A": [_make_edge("line1", "B"), _make_edge("line1", "C")],
        "B": [_make_edge("line1", "A"), _make_edge("line1", "D")],
        "C": [_make_edge("line1", "A")],
        "D": [_make_edge("line1", "B")],
    }
    storage = _NeighborhoodMockStorage(
        graph,
        anomaly_map={"B": True, "D": True},
        rank_map={"B": 97.2, "C": 23.1, "D": 88.5},
    )
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=_NeighborhoodMockManifest(),
        contract=Contract("m-001", []),
    )
    result = nav.find_neighborhood("A", "test_pattern", max_hops=2)

    assert result["center"] == "A"
    assert result["pattern_id"] == "test_pattern"
    assert result["max_hops"] == 2
    keys = {e["key"] for e in result["entities"]}
    assert "B" in keys
    assert "C" in keys
    assert "D" in keys
    assert "A" not in keys  # center excluded

    # Check hop distances
    by_key = {e["key"]: e for e in result["entities"]}
    assert by_key["B"]["hop"] == 1
    assert by_key["C"]["hop"] == 1
    assert by_key["D"]["hop"] == 2

    # Anomaly info
    assert by_key["B"]["is_anomaly"] is True
    assert by_key["B"]["delta_rank_pct"] == 97.2
    assert by_key["C"]["is_anomaly"] is False

    # Summary
    s = result["summary"]
    assert s["total"] == 3
    assert s["anomalous"] == 2  # B and D
    assert s["max_hop_reached"] == 2
    assert s["capped"] is False


def test_find_neighborhood_empty_geometry():
    """Entity not found in geometry returns empty result."""
    storage = _NeighborhoodMockStorage(graph={})
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=_NeighborhoodMockManifest(),
        contract=Contract("m-001", []),
    )
    result = nav.find_neighborhood("MISSING", "test_pattern")

    assert result["center"] == "MISSING"
    assert result["entities"] == []
    assert result["summary"]["total"] == 0
    assert result["summary"]["anomalous"] == 0
    assert result["summary"]["max_hop_reached"] == 0
    assert result["summary"]["capped"] is False


def test_find_neighborhood_cap_reached():
    """max_entities cap triggers capped=True."""
    # A connects to B, C, D — but cap is 2
    graph = {
        "A": [
            _make_edge("line1", "B"),
            _make_edge("line1", "C"),
            _make_edge("line1", "D"),
        ],
        "B": [],
        "C": [],
        "D": [],
    }
    storage = _NeighborhoodMockStorage(graph)
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=_NeighborhoodMockManifest(),
        contract=Contract("m-001", []),
    )
    result = nav.find_neighborhood("A", "test_pattern", max_entities=2)

    assert result["summary"]["capped"] is True
    assert result["summary"]["total"] == 2


def test_find_neighborhood_no_jumpable_edges():
    """Edges with empty point_key are not jumpable — nothing traversed."""
    graph = {
        "A": [
            _make_edge("line1", ""),  # continuous mode — not jumpable
            _make_edge("line1", ""),
        ],
    }
    storage = _NeighborhoodMockStorage(graph)
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=_NeighborhoodMockManifest(),
        contract=Contract("m-001", []),
    )
    result = nav.find_neighborhood("A", "test_pattern")

    assert result["entities"] == []
    assert result["summary"]["total"] == 0
    assert result["summary"]["capped"] is False


# ---------------------------------------------------------------------------
# find_chains_for_entity tests
# ---------------------------------------------------------------------------


class _ChainMockStorage(_MockStorage):
    """Mock storage for find_chains_for_entity tests."""

    def __init__(
        self,
        chain_keys_map: dict[str, str],
        geometry_rows: list[dict],
        *,
        has_chain_keys: bool = True,
    ):
        super().__init__()
        self._chain_keys_map = chain_keys_map  # pk -> chain_keys csv
        self._geometry_rows = geometry_rows
        self._has_chain_keys = has_chain_keys

    def read_points(self, line_id, version, filters=None, primary_key=None):
        pks = list(self._chain_keys_map.keys())
        n = len(pks)
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        cols: dict = {
            "primary_key": pks,
            "version": [1] * n,
            "status": ["active"] * n,
            "created_at": [dt] * n,
            "changed_at": [dt] * n,
        }
        if self._has_chain_keys:
            cols["chain_keys"] = [self._chain_keys_map[pk] for pk in pks]
        return pa.table(cols)

    def read_geometry(
        self,
        pattern_id,
        version,
        primary_key=None,
        filters=None,
        point_keys=None,
        columns=None,
        filter=None,
        sample_size=None,
    ):
        if not self._geometry_rows:
            return pa.table({"primary_key": pa.array([], type=pa.string())})
        full = pa.table(
            {
                "primary_key": [r["primary_key"] for r in self._geometry_rows],
                "is_anomaly": [r["is_anomaly"] for r in self._geometry_rows],
                "delta_norm": [r["delta_norm"] for r in self._geometry_rows],
                "delta_rank_pct": pa.array(
                    [r["delta_rank_pct"] for r in self._geometry_rows],
                    type=pa.float64(),
                ),
            }
        )
        if columns is not None:
            full = full.select(columns)
        return full

    def read_sphere(self):
        from hypertopos.model.sphere import (
            Line,
            PartitionConfig,
            Pattern,
            RelationDef,
            Sphere,
        )

        return Sphere(
            sphere_id="s",
            name="test",
            base_path=".",
            lines={
                "chain_entities": Line(
                    line_id="chain_entities",
                    entity_type="account",
                    line_role="anchor",
                    pattern_id="chain_pattern",
                    partitioning=PartitionConfig(mode="static", columns=[]),
                    versions=[1],
                ),
            },
            patterns={
                "chain_pattern": Pattern(
                    pattern_id="chain_pattern",
                    entity_type="account",
                    pattern_type="anchor",
                    relations=[
                        RelationDef(
                            line_id="chain_entities",
                            direction="self",
                            required=True,
                        ),
                    ],
                    mu=np.zeros(2),
                    sigma_diag=np.ones(2),
                    theta=np.array([3.0, 3.0]),
                    population_size=100,
                    computed_at=datetime(2024, 1, 1, tzinfo=UTC),
                    version=1,
                    status="production",
                ),
            },
        )


class _ChainMockManifest:
    def pattern_version(self, pid):
        return 1

    def line_version(self, lid):
        return 1


def test_find_chains_for_entity_happy_path():
    """Entity in 2 chains, one anomalous — correct counts + sorting."""
    # In chain patterns, primary_key = chain ID, chain_keys = entity keys
    chain_keys_map = {
        "CH-001": "ACCT-001,ACCT-002",
        "CH-002": "ACCT-001,ACCT-003",
        "CH-003": "ACCT-004",
    }
    geometry_rows = [
        {
            "primary_key": "CH-001",
            "is_anomaly": True,
            "delta_norm": 12.3,
            "delta_rank_pct": 97.2,
        },
        {
            "primary_key": "CH-002",
            "is_anomaly": False,
            "delta_norm": 2.1,
            "delta_rank_pct": 34.5,
        },
        {
            "primary_key": "CH-003",
            "is_anomaly": True,
            "delta_norm": 8.0,
            "delta_rank_pct": 85.0,
        },
    ]
    storage = _ChainMockStorage(chain_keys_map, geometry_rows)
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=_ChainMockManifest(),
        contract=Contract("m-001", []),
    )

    result = nav.find_chains_for_entity("ACCT-001", "chain_pattern")

    assert result["primary_key"] == "ACCT-001"
    assert result["pattern_id"] == "chain_pattern"
    assert len(result["chains"]) == 2
    # Sorted by delta_norm desc: CH-001 (12.3) before CH-002 (2.1)
    assert result["chains"][0]["chain_id"] == "CH-001"
    assert result["chains"][0]["is_anomaly"] is True
    assert result["chains"][0]["delta_norm"] == 12.3
    assert result["chains"][0]["delta_rank_pct"] == 97.2
    assert result["chains"][1]["chain_id"] == "CH-002"
    assert result["chains"][1]["is_anomaly"] is False
    assert result["chains"][1]["delta_norm"] == 2.1
    assert result["chains"][1]["delta_rank_pct"] == 34.5
    # CH-003 not included (ACCT-001 not in CH-003)
    assert result["summary"]["total"] == 2
    assert result["summary"]["anomalous"] == 1


def test_find_chains_for_entity_not_in_any_chain():
    """Entity not in any chain returns empty chains list."""
    # chain_keys contain entity keys; ACCT-999 is not in any chain
    chain_keys_map = {
        "CH-001": "ACCT-001",
    }
    geometry_rows = [
        {
            "primary_key": "CH-001",
            "is_anomaly": False,
            "delta_norm": 1.0,
            "delta_rank_pct": 10.0,
        },
    ]
    storage = _ChainMockStorage(chain_keys_map, geometry_rows)
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=_ChainMockManifest(),
        contract=Contract("m-001", []),
    )

    result = nav.find_chains_for_entity("ACCT-999", "chain_pattern")

    assert result["primary_key"] == "ACCT-999"
    assert result["chains"] == []
    assert result["summary"]["total"] == 0
    assert result["summary"]["anomalous"] == 0


def test_find_chains_for_entity_no_chain_keys_column():
    """Pattern without chain_keys column raises GDSNavigationError."""
    storage = _ChainMockStorage(
        chain_keys_map={"CH-001": ""},
        geometry_rows=[],
        has_chain_keys=False,
    )
    nav = GDSNavigator(
        engine=_MockEngine(),
        storage=storage,
        manifest=_ChainMockManifest(),
        contract=Contract("m-001", []),
    )

    with pytest.raises(GDSNavigationError, match="no chain_keys column"):
        nav.find_chains_for_entity("ACCT-001", "chain_pattern")


class TestLineProfile:
    """Tests for GDSNavigator.line_profile — direct points-table profiling."""

    @staticmethod
    def _make_nav(sphere_path):
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache
        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-1",
            agent_id="a-1",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        return GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-1", ["customer_pattern"]),
        )

    # -- Path A: string/bool → value counts --

    def test_string_value_counts(self, sphere_path):
        nav = self._make_nav(sphere_path)
        r = nav.line_profile("customers", "region")
        assert r["type"] == "categorical"
        assert r["total"] == 3
        assert r["distinct"] == 2
        assert r["null_count"] == 0
        values = {v["value"]: v["count"] for v in r["top_values"]}
        assert values["EMEA"] == 2
        assert values["APAC"] == 1

    def test_string_sorted_desc(self, sphere_path):
        nav = self._make_nav(sphere_path)
        r = nav.line_profile("customers", "region")
        counts = [v["count"] for v in r["top_values"]]
        assert counts == sorted(counts, reverse=True)

    def test_string_limit(self, sphere_path):
        nav = self._make_nav(sphere_path)
        r = nav.line_profile("customers", "region", limit=1)
        assert len(r["top_values"]) == 1
        assert r["distinct"] == 2  # full count preserved

    # -- Path B: numeric → stats --

    def test_numeric_stats(self, sphere_path):
        nav = self._make_nav(sphere_path)
        r = nav.line_profile("customers", "balance")
        assert r["type"] == "numeric"
        assert r["total"] == 3
        assert r["min"] == pytest.approx(-500.0)
        assert r["max"] == pytest.approx(9999.5)
        assert r["mean"] == pytest.approx(3216.75, rel=1e-2)
        assert "std" in r
        assert "median" in r
        assert "p25" in r
        assert "p75" in r
        assert r["p25"] <= r["median"] <= r["p75"]
        assert "top_values" not in r

    # -- Path B+group: numeric grouped by string --

    def test_numeric_grouped(self, sphere_path):
        nav = self._make_nav(sphere_path)
        r = nav.line_profile("customers", "balance", group_by="region")
        assert r["type"] == "numeric_grouped"
        assert r["group_by"] == "region"
        groups = {g["group"]: g for g in r["groups"]}
        assert groups["EMEA"]["count"] == 2
        assert groups["EMEA"]["mean"] == pytest.approx(5075.125, rel=1e-2)
        assert "p25" in groups["EMEA"]
        assert "p75" in groups["EMEA"]
        assert groups["APAC"]["count"] == 1
        assert groups["APAC"]["mean"] == pytest.approx(-500.0)

    def test_group_by_on_string_property_raises(self, sphere_path):
        nav = self._make_nav(sphere_path)
        with pytest.raises(GDSNavigationError, match="numeric"):
            nav.line_profile("customers", "region", group_by="status")

    def test_group_by_numeric_column_raises(self, sphere_path):
        nav = self._make_nav(sphere_path)
        with pytest.raises(GDSNavigationError, match="categorical"):
            nav.line_profile("customers", "balance", group_by="balance")

    # -- Path temporal: date/timestamp → min/max --

    def test_timestamp_range(self, sphere_path):
        nav = self._make_nav(sphere_path)
        r = nav.line_profile("customers", "created_at")
        assert r["type"] == "temporal"
        assert r["total"] == 3
        assert "min" in r
        assert "max" in r
        assert "mean" not in r

    # -- Errors --

    def test_unknown_property_raises(self, sphere_path):
        nav = self._make_nav(sphere_path)
        with pytest.raises(GDSNavigationError, match="not found"):
            nav.line_profile("customers", "nonexistent")

    def test_unknown_line_raises(self, sphere_path):
        nav = self._make_nav(sphere_path)
        with pytest.raises(GDSNavigationError, match="not found"):
            nav.line_profile("nonexistent", "region")

    def test_group_by_unknown_raises(self, sphere_path):
        nav = self._make_nav(sphere_path)
        with pytest.raises(GDSNavigationError, match="not found"):
            nav.line_profile("customers", "balance", group_by="nope")


# ---------------------------------------------------------------------------
# Sibling-line pattern discovery
# ---------------------------------------------------------------------------


class _MockStorageSibling(_MockStorage):
    """Mock storage with two anchor lines sharing the same source_id."""

    def read_sphere(self):
        from hypertopos.model.sphere import Line, PartitionConfig, Pattern, RelationDef, Sphere

        dt = datetime(2024, 1, 1, tzinfo=UTC)
        part = PartitionConfig(mode="static", columns=[])

        pat_a = Pattern(
            pattern_id="pat_a",
            entity_type="accounts",
            pattern_type="anchor",
            relations=[RelationDef(line_id="txns", direction="in", required=True)],
            mu=np.zeros(1, dtype=np.float32),
            sigma_diag=np.ones(1, dtype=np.float32),
            theta=np.array([3.0], dtype=np.float32),
            population_size=10,
            computed_at=dt,
            version=1,
            status="production",
            entity_line_id="line_a",
        )
        pat_b = Pattern(
            pattern_id="pat_b",
            entity_type="accounts",
            pattern_type="anchor",
            relations=[RelationDef(line_id="txns", direction="in", required=True)],
            mu=np.zeros(1, dtype=np.float32),
            sigma_diag=np.ones(1, dtype=np.float32),
            theta=np.array([3.0], dtype=np.float32),
            population_size=10,
            computed_at=dt,
            version=1,
            status="production",
            entity_line_id="line_b",
        )

        line_a = Line(
            line_id="line_a",
            entity_type="accounts",
            line_role="anchor",
            pattern_id="pat_a",
            partitioning=part,
            versions=[1],
            source_id="accounts",
        )
        line_b = Line(
            line_id="line_b",
            entity_type="accounts",
            line_role="anchor",
            pattern_id="pat_b",
            partitioning=part,
            versions=[1],
            source_id="accounts",
        )
        line_txns = Line(
            line_id="txns",
            entity_type="transactions",
            line_role="event",
            pattern_id="pat_a",
            partitioning=part,
            versions=[1],
        )

        return Sphere(
            "s", "s", ".",
            lines={"line_a": line_a, "line_b": line_b, "txns": line_txns},
            patterns={"pat_a": pat_a, "pat_b": pat_b},
        )

    def read_points(self, line_id, version, filters=None, primary_key=None):
        keys = ["ACC-001", "ACC-002"]
        if primary_key is not None:
            keys = [k for k in keys if k == primary_key]
        n = len(keys)
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        return pa.table({
            "primary_key": keys,
            "version": [1] * n,
            "status": ["active"] * n,
            "created_at": [dt] * n,
            "changed_at": [dt] * n,
        })

    def read_geometry(
        self, pattern_id, version, primary_key=None, filters=None,
        point_keys=None, columns=None, filter=None, sample_size=None,
    ):
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        all_data = {
            "primary_key": ["ACC-001", "ACC-002"],
            "delta_norm": [4.5, 1.0],
            "is_anomaly": [True, False],
            "delta_rank_pct": pa.array([95.0, 30.0], type=pa.float64()),
            "conformal_p": pa.array([0.02, 0.85], type=pa.float64()),
        }
        if primary_key is not None:
            idx = [i for i, k in enumerate(all_data["primary_key"]) if k == primary_key]
            if not idx:
                cols_out = columns or list(all_data.keys())
                return pa.table({c: pa.array([], type=pa.float64() if c != "primary_key" else pa.string())
                                 for c in cols_out if c in all_data})
            i = idx[0]
            all_data = {k: [v[i]] if isinstance(v, list) else v.take([i])
                        for k, v in all_data.items()}
        tbl = pa.table(all_data)
        if columns is not None:
            available = set(tbl.schema.names)
            columns = [c for c in columns if c in available]
            if columns:
                tbl = tbl.select(columns)
        return tbl


def _make_sibling_manifest() -> Manifest:
    return Manifest(
        manifest_id="m-sib",
        agent_id="agent-001",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"line_a": 1, "line_b": 1, "txns": 1},
        pattern_versions={"pat_a": 1, "pat_b": 1},
    )


class TestSiblingPatternDiscovery:
    """Tests for _discover_pattern_map with sibling lines."""

    def test_discover_pattern_map_classifies_sibling(self):
        """Pattern on a sibling line (same source_id) is classified as 'sibling'."""
        nav = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageSibling(),
            manifest=_make_sibling_manifest(),
            contract=Contract("m-sib", []),
        )
        pmap = nav._discover_pattern_map("line_a")
        assert pmap["pat_a"] == "direct"
        assert pmap["pat_b"] == "sibling"

    def test_discover_pattern_map_symmetric(self):
        """Sibling classification works from either direction."""
        nav = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageSibling(),
            manifest=_make_sibling_manifest(),
            contract=Contract("m-sib", []),
        )
        pmap_b = nav._discover_pattern_map("line_b")
        assert pmap_b["pat_b"] == "direct"
        assert pmap_b["pat_a"] == "sibling"

    def test_cross_pattern_profile_includes_sibling(self):
        """cross_pattern_profile returns signals from both direct and sibling patterns."""
        nav = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageSibling(),
            manifest=_make_sibling_manifest(),
            contract=Contract("m-sib", []),
        )
        profile = nav.cross_pattern_profile("ACC-001", line_id="line_a")
        assert "pat_a" in profile["signals"]
        assert "pat_b" in profile["signals"]
        assert profile["signals"]["pat_a"]["key_type"] == "direct"
        assert profile["signals"]["pat_b"]["key_type"] == "sibling"
        assert profile["total_patterns"] == 2

    def test_composite_risk_combines_sibling_pvalues(self):
        """composite_risk combines conformal_p from both direct and sibling patterns."""
        nav = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageSibling(),
            manifest=_make_sibling_manifest(),
            contract=Contract("m-sib", []),
        )
        result = nav.composite_risk("ACC-001", line_id="line_a")
        assert result["n_patterns"] == 2
        assert result["combined_p"] is not None
        assert "pat_a" in result["per_pattern"]
        assert "pat_b" in result["per_pattern"]

    def test_composite_risk_batch(self):
        """Batch returns per-key results with summary counts."""
        nav = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageSibling(),
            manifest=_make_sibling_manifest(),
            contract=Contract("m-sib", []),
        )
        keys = ["ACC-001", "ACC-002"]
        result = nav.composite_risk_batch(keys, line_id="line_a")
        assert result["total_requested"] == 2
        assert result["total_checked"] == 2
        assert "caught_p010" in result
        assert "caught_p005" in result
        assert len(result["results"]) == 2
        # Results sorted by combined_p ascending
        ps = [r["combined_p"] for r in result["results"] if r["combined_p"] is not None]
        assert ps == sorted(ps)

    def test_composite_risk_batch_missing_key(self):
        """Batch handles non-existent keys gracefully."""
        nav = GDSNavigator(
            engine=_MockEngine(),
            storage=_MockStorageSibling(),
            manifest=_make_sibling_manifest(),
            contract=Contract("m-sib", []),
        )
        result = nav.composite_risk_batch(["NONEXISTENT"], line_id="line_a")
        assert result["total_checked"] == 1
        assert result["results"][0].get("error") == "not_found"


# --- aggregate_anomalies ---


def test_aggregate_anomalies_basic(tmp_path):
    """aggregate_anomalies groups anomalous entities by a property column."""
    from hypertopos.builder.builder import GDSBuilder, RelationSpec
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.storage.cache import GDSCache
    from hypertopos.storage.reader import GDSReader

    b = GDSBuilder("s", str(tmp_path / "gds"))
    # 20 customers: 15 have both FKs filled, 5 missing seg_id → anomalous (missing edge)
    data = [
        {"cid": f"C-{i}", "seg_id": "SEG-A", "cat_id": "CAT-X", "region": "EMEA"}
        for i in range(1, 16)
    ] + [
        {"cid": f"C-{i}", "seg_id": None, "cat_id": "CAT-X", "region": "APAC"}
        for i in range(16, 21)
    ]
    b.add_line("customers", data, key_col="cid", source_id="test")
    b.add_line("segments", [{"sid": "SEG-A"}, {"sid": "SEG-B"}], key_col="sid", source_id="test")
    b.add_line("categories", [{"cid": "CAT-X"}, {"cid": "CAT-Y"}], key_col="cid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("segments", fk_col="seg_id", direction="in", required=False),
            RelationSpec("categories", fk_col="cat_id", direction="in", required=True),
        ],
        anomaly_percentile=60.0,
    )
    out = b.build()

    reader = GDSReader(base_path=out)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id="m-1",
        agent_id="a-1",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "segments": 1, "categories": 1},
        pattern_versions={"cp": 1},
    )
    nav = GDSNavigator(
        engine=engine, storage=reader, manifest=manifest,
        contract=Contract("m-1", ["cp"]),
    )
    result = nav.aggregate_anomalies("cp", group_by="region")
    assert result["pattern_id"] == "cp"
    assert result["group_by"] == "region"
    assert result["total_anomalies"] > 0
    assert len(result["groups"]) > 0
    for g in result["groups"]:
        assert "group_key" in g
        assert "anomaly_count" in g
        assert "mean_delta_norm" in g
    # groups sorted by anomaly_count desc
    counts = [g["anomaly_count"] for g in result["groups"]]
    assert counts == sorted(counts, reverse=True)


def test_aggregate_anomalies_property_filters(tmp_path):
    """property_filters narrows the anomalous set before grouping."""
    from hypertopos.builder.builder import GDSBuilder, RelationSpec
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.storage.cache import GDSCache
    from hypertopos.storage.reader import GDSReader

    b = GDSBuilder("s", str(tmp_path / "gds"))
    data = [
        {"cid": f"C-{i}", "seg_id": "SEG-A", "cat_id": "CAT-X", "region": "EMEA"}
        for i in range(1, 16)
    ] + [
        {"cid": f"C-{i}", "seg_id": None, "cat_id": "CAT-X", "region": "APAC"}
        for i in range(16, 21)
    ]
    b.add_line("customers", data, key_col="cid", source_id="test")
    b.add_line("segments", [{"sid": "SEG-A"}, {"sid": "SEG-B"}], key_col="sid", source_id="test")
    b.add_line("categories", [{"cid": "CAT-X"}, {"cid": "CAT-Y"}], key_col="cid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("segments", fk_col="seg_id", direction="in", required=False),
            RelationSpec("categories", fk_col="cat_id", direction="in", required=True),
        ],
        anomaly_percentile=60.0,
    )
    out = b.build()

    reader = GDSReader(base_path=out)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id="m-1",
        agent_id="a-1",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "segments": 1, "categories": 1},
        pattern_versions={"cp": 1},
    )
    nav = GDSNavigator(
        engine=engine, storage=reader, manifest=manifest,
        contract=Contract("m-1", ["cp"]),
    )
    # Without filter
    result_all = nav.aggregate_anomalies("cp", group_by="region")
    total_all = result_all["total_anomalies"]

    # With filter — only APAC
    result_filtered = nav.aggregate_anomalies(
        "cp", group_by="region",
        property_filters={"region": "APAC"},
    )
    assert result_filtered["total_anomalies"] <= total_all
    if result_filtered["groups"]:
        for g in result_filtered["groups"]:
            assert g["group_key"] == "APAC"


def test_aggregate_anomalies_invalid_column(tmp_path):
    """aggregate_anomalies raises on nonexistent group_by column."""
    from hypertopos.builder.builder import GDSBuilder, RelationSpec
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.storage.cache import GDSCache
    from hypertopos.storage.reader import GDSReader

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [{"cid": "C-1", "seg_id": "SEG-A"}],
        key_col="cid", source_id="test",
    )
    b.add_line("segments", [{"sid": "SEG-A"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cp", pattern_type="anchor", entity_line="customers",
        relations=[RelationSpec("segments", fk_col="seg_id", direction="in", required=True)],
    )
    out = b.build()

    reader = GDSReader(base_path=out)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id="m-1", agent_id="a-1",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "segments": 1, "categories": 1},
        pattern_versions={"cp": 1},
    )
    nav = GDSNavigator(
        engine=engine, storage=reader, manifest=manifest,
        contract=Contract("m-1", ["cp"]),
    )
    with pytest.raises(ValueError, match="not found"):
        nav.aggregate_anomalies("cp", group_by="nonexistent_column")


def test_aggregate_anomalies_ungrouped_field(tmp_path):
    """aggregate_anomalies exposes ungrouped_anomalies count."""
    from hypertopos.builder.builder import GDSBuilder, RelationSpec
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.storage.cache import GDSCache
    from hypertopos.storage.reader import GDSReader

    b = GDSBuilder("s", str(tmp_path / "gds"))
    data = [
        {"cid": f"C-{i}", "seg_id": "SEG-A", "cat_id": "CAT-X", "region": "EMEA"}
        for i in range(1, 16)
    ] + [
        {"cid": f"C-{i}", "seg_id": None, "cat_id": "CAT-X", "region": "APAC"}
        for i in range(16, 21)
    ]
    b.add_line("customers", data, key_col="cid", source_id="test")
    b.add_line("segments", [{"sid": "SEG-A"}, {"sid": "SEG-B"}], key_col="sid", source_id="test")
    b.add_line("categories", [{"cid": "CAT-X"}, {"cid": "CAT-Y"}], key_col="cid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("segments", fk_col="seg_id", direction="in", required=False),
            RelationSpec("categories", fk_col="cat_id", direction="in", required=True),
        ],
        anomaly_percentile=60.0,
    )
    out = b.build()

    reader = GDSReader(base_path=out)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id="m-1",
        agent_id="a-1",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "segments": 1, "categories": 1},
        pattern_versions={"cp": 1},
    )
    nav = GDSNavigator(
        engine=engine, storage=reader, manifest=manifest,
        contract=Contract("m-1", ["cp"]),
    )
    result = nav.aggregate_anomalies("cp", group_by="region")

    # ungrouped_anomalies field must always be present
    assert "ungrouped_anomalies" in result
    assert isinstance(result["ungrouped_anomalies"], int)
    assert result["ungrouped_anomalies"] >= 0

    # invariant: total_anomalies == sum(group counts) + ungrouped_anomalies
    grouped_sum = sum(g["anomaly_count"] for g in result["groups"])
    assert result["total_anomalies"] == grouped_sum + result["ungrouped_anomalies"]
