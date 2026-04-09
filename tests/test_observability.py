# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Observability methods for GDSNavigator: sphere_overview, pi11, detect_data_quality,
pi12, line_geometry_stats.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.model.manifest import Manifest
from hypertopos.navigation.navigator import GDSNavigator

# ---------------------------------------------------------------------------
# Helpers — pattern / sphere factories
# ---------------------------------------------------------------------------


def _make_relation(line_id: str, required: bool = True):
    from hypertopos.model.sphere import RelationDef

    return RelationDef(
        line_id=line_id,
        direction="in",
        required=required,
        display_name=line_id,
    )


def _make_pattern(pid: str, ptype: str = "anchor"):
    from hypertopos.model.sphere import Pattern, RelationDef

    rel = RelationDef(
        line_id="products",
        direction="in",
        required=True,
        display_name="products",
    )
    return Pattern(
        pattern_id=pid,
        entity_type="customers",
        pattern_type=ptype,
        relations=[rel],
        mu=np.array([0.5], dtype=np.float32),
        sigma_diag=np.array([0.2], dtype=np.float32),
        theta=np.array([1.5], dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=None,
        prop_columns=[],
        excluded_properties=[],
    )


def _make_sphere(*pattern_ids) -> MagicMock:
    sphere = MagicMock()
    sphere.patterns = {pid: _make_pattern(pid) for pid in pattern_ids}
    sphere.lines = {}
    return sphere


def _make_manifest(pattern_ids: list[str]) -> Manifest:
    return Manifest(
        manifest_id=str(uuid.uuid4()),
        agent_id="test",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1},
        pattern_versions=dict.fromkeys(pattern_ids, 1),
    )


def _make_navigator(sphere=None, geo_stats=None, geo_rows=0, geo_anomalies=0):
    """Return a GDSNavigator with minimal mocked storage."""
    storage = MagicMock()
    engine = MagicMock()
    cache = MagicMock()

    if sphere is None:
        sphere = _make_sphere("customer_pattern")

    storage.read_sphere.return_value = sphere
    storage.read_geometry_stats.return_value = geo_stats
    storage.read_temporal_centroids.return_value = None
    storage.count_geometry_rows.side_effect = lambda pid, ver, filter=None: (
        geo_anomalies if filter is not None else geo_rows
    )

    manifest = _make_manifest(list(sphere.patterns.keys()))

    nav = GDSNavigator.__new__(GDSNavigator)
    nav._storage = storage
    nav._engine = engine
    nav._cache = cache
    nav._manifest = manifest
    return nav


# ---------------------------------------------------------------------------
# Temporal table helpers
# ---------------------------------------------------------------------------


def _make_temporal_table(entries: list[tuple[str, str, list[float]]]) -> pa.Table:
    """entries: [(primary_key, timestamp_iso, shape_snapshot), ...]"""
    pks = [e[0] for e in entries]
    ts_list = [datetime.fromisoformat(e[1]).replace(tzinfo=UTC) for e in entries]
    shapes = [e[2] for e in entries]
    return pa.table(
        {
            "primary_key": pa.array(pks, type=pa.string()),
            "timestamp": pa.array(ts_list, type=pa.timestamp("us", tz="UTC")),
            "shape_snapshot": pa.array(shapes, type=pa.list_(pa.float32())),
            "deformation_type": pa.array(["structural"] * len(entries), type=pa.string()),
            "slice_index": pa.array(list(range(len(entries))), type=pa.int32()),
        }
    )


def _make_navigator_with_temporal(temporal_entries):
    """Navigator with temporal data — read_temporal_batched accepts kwargs."""
    nav = _make_navigator(geo_rows=100)
    tbl = _make_temporal_table(temporal_entries)
    batches = [tbl.to_batches()[0]]

    def _batched_iter(pid, batch_size=65_536, timestamp_from=None, timestamp_to=None):
        return iter(batches)

    nav._storage.read_temporal_batched.side_effect = _batched_iter
    return nav


# ---------------------------------------------------------------------------
# Geometry table helper
# ---------------------------------------------------------------------------

_EDGE_STRUCT = pa.struct(
    [
        pa.field("line_id", pa.string()),
        pa.field("point_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("direction", pa.string()),
    ]
)


def _make_geo_table(
    edges_per_entity: list[list[dict]],
    delta_norms: list[float],
    is_anomaly: list[bool],
) -> pa.Table:
    n = len(edges_per_entity)
    edges_arr = pa.array(
        [[pa.scalar(e, type=_EDGE_STRUCT) for e in row] for row in edges_per_entity],
        type=pa.list_(_EDGE_STRUCT),
    )
    return pa.table(
        {
            "primary_key": pa.array([f"E-{i}" for i in range(n)]),
            "delta_norm": pa.array(delta_norms, type=pa.float32()),
            "is_anomaly": pa.array(is_anomaly, type=pa.bool_()),
            "edges": edges_arr,
            "delta": pa.array([[0.0] for _ in range(n)], type=pa.list_(pa.float32())),
        }
    )


def _make_navigator_for_dqi(
    geo_rows, geo_anomalies, degenerate_count, edges_per_entity, near_ceiling_count=0
):
    """Navigator mocked for detect_data_quality_issues with streaming API.

    read_geometry_batched is called twice:
      1st call — edges coverage check (columns=["edges"])
      2nd call — delta_norm mismatch check (columns=["primary_key","delta","delta_norm"])
    The 2nd call returns a consistent table (delta_norm == ||delta||) so existing tests
    do not produce unexpected delta_norm_mismatch findings.
    """
    nav = _make_navigator(geo_rows=geo_rows, geo_anomalies=geo_anomalies)
    # theta_norm for the mock pattern is 1.5 (theta=[1.5], 1-dim)
    _theta_norm = 1.5

    def _count_rows(pid, ver, filter=None):
        if filter and "delta_norm <" in filter:
            return degenerate_count
        if filter and "delta_norm >=" in filter:
            # Parse the threshold to distinguish anomaly vs near-ceiling
            try:
                threshold = float(filter.split(">=")[1].strip())
            except (IndexError, ValueError):
                return near_ceiling_count
            # Anomaly count: threshold == theta_norm (1.5)
            if abs(threshold - _theta_norm) < 0.01:
                return geo_anomalies
            # Near-ceiling: threshold < theta_norm (e.g. 0.75 * theta)
            return near_ceiling_count
        if filter and "is_anomaly" in filter:
            return geo_anomalies
        return geo_rows

    nav._storage.count_geometry_rows.side_effect = _count_rows
    n = len(edges_per_entity)
    # Build edges struct column for coverage scan
    _edge_struct_type = pa.struct(
        [
            pa.field("line_id", pa.string()),
            pa.field("point_key", pa.string()),
            pa.field("status", pa.string()),
            pa.field("direction", pa.string()),
        ]
    )
    edges_table = pa.table(
        {
            "edges": pa.array(edges_per_entity, type=pa.list_(_edge_struct_type)),
        }
    )
    # Consistent geometry table: delta=[1.0], delta_norm=1.0 — no mismatch
    consistent_table = pa.table(
        {
            "primary_key": pa.array([f"E-{i}" for i in range(n)]),
            "delta": pa.array([[1.0]] * n, type=pa.list_(pa.float32())),
            "delta_norm": pa.array([1.0] * n, type=pa.float32()),
        }
    )
    call_count = [0]

    def _batched(pid, ver, columns=None, filter_expr=None, batch_size=65_536):
        call_count[0] += 1
        if call_count[0] == 1:
            return iter(edges_table.to_batches())
        return iter(consistent_table.to_batches())

    nav._storage.read_geometry_batched.side_effect = _batched
    return nav


# ===========================================================================
# Task 1: sphere_overview
# ===========================================================================


class TestSphereOverview:
    def test_returns_list_for_all_patterns(self):
        sphere = _make_sphere("customer_pattern", "gl_entry_pattern")
        sphere.patterns["gl_entry_pattern"] = _make_pattern("gl_entry_pattern", "event")
        nav = _make_navigator(sphere=sphere, geo_rows=100, geo_anomalies=5)
        result = nav.sphere_overview()
        assert isinstance(result, list)
        assert len(result) == 2
        ids = {r["pattern_id"] for r in result}
        assert ids == {"customer_pattern", "gl_entry_pattern"}

    def test_returns_single_pattern_when_id_given(self):
        sphere = _make_sphere("customer_pattern", "gl_entry_pattern")
        sphere.patterns["gl_entry_pattern"] = _make_pattern("gl_entry_pattern", "event")
        nav = _make_navigator(sphere=sphere, geo_rows=100, geo_anomalies=5)
        result = nav.sphere_overview("customer_pattern")
        assert len(result) == 1
        assert result[0]["pattern_id"] == "customer_pattern"

    def test_anomaly_rate_from_stats_cache(self):
        stats = {
            "total_entities": 1000,
            "total_anomalies": 50,
            "percentiles": {
                "p50": 1.0,
                "p75": 1.5,
                "p90": 2.0,
                "p95": 2.5,
                "p99": 3.0,
                "max": 5.0,
            },
        }
        # sphere_overview counts anomalies via delta_norm >= theta_norm
        # (not stats cache), so geo_anomalies must match the expected count.
        nav = _make_navigator(geo_stats=stats, geo_anomalies=50)
        result = nav.sphere_overview()
        assert result[0]["anomaly_rate"] == pytest.approx(0.05)
        assert result[0]["total_entities"] == 1000

    def test_anomaly_rate_fallback_when_no_stats(self):
        nav = _make_navigator(geo_stats=None, geo_rows=200, geo_anomalies=20)
        result = nav.sphere_overview()
        assert result[0]["anomaly_rate"] == pytest.approx(0.1)


class TestSphereOverviewCalibrationHealth:
    def _nav_with_rate(self, total: int, anomalies: int):
        """Navigator with geo_stats returning the given anomaly rate."""
        stats = {
            "total_entities": total,
            "total_anomalies": anomalies,
            "percentiles": {
                "p50": 1.0,
                "p75": 1.5,
                "p90": 2.0,
                "p95": 2.5,
                "p99": 3.0,
                "max": 5.0,
            },
        }
        # sphere_overview counts via delta_norm filter, not stats cache
        return _make_navigator(geo_stats=stats, geo_anomalies=anomalies)

    def test_good_when_rate_in_normal_range(self):
        # 5% → good
        nav = self._nav_with_rate(total=1000, anomalies=50)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "good"

    def test_suspect_when_rate_above_20_pct(self):
        # 25% → suspect
        nav = self._nav_with_rate(total=1000, anomalies=250)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "suspect"

    def test_poor_when_rate_above_30_pct(self):
        # 40% → poor
        nav = self._nav_with_rate(total=1000, anomalies=400)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "poor"

    def test_suspect_when_rate_below_1_pct(self):
        # 0.5% → suspect
        nav = self._nav_with_rate(total=2000, anomalies=10)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "suspect"

    def test_poor_when_rate_zero(self):
        # 0% on non-empty population → poor
        nav = self._nav_with_rate(total=1000, anomalies=0)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "poor"

    def test_good_at_boundary_1_pct(self):
        # exactly 1% → good (boundary inclusive)
        nav = self._nav_with_rate(total=1000, anomalies=10)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "good"

    def test_good_at_boundary_20_pct(self):
        # exactly 20% → good (boundary inclusive)
        nav = self._nav_with_rate(total=1000, anomalies=200)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "good"

    def test_good_when_zero_entities(self):
        # empty pattern (total=0, anomaly_rate=0.0) → good
        nav = self._nav_with_rate(total=0, anomalies=0)
        result = nav.sphere_overview()
        assert result[0]["calibration_health"] == "good"


# ===========================================================================
# Task 2: π11_attract_population_compare
# ===========================================================================


class TestCompareTimeWindows:
    ENTRIES = [
        ("CUST-001", "2024-01-15", [0.5]),
        ("CUST-002", "2024-01-16", [0.6]),
        ("CUST-003", "2024-06-15", [1.5]),
        ("CUST-004", "2024-06-16", [1.6]),
    ]

    def test_same_window_returns_near_zero_shift(self):
        nav = _make_navigator_with_temporal(self.ENTRIES)
        result = nav.π11_attract_population_compare(
            "customer_pattern",
            "2024-01-01",
            "2024-12-31",
            "2024-01-01",
            "2024-12-31",
        )
        assert result["centroid_shift"] == pytest.approx(0.0, abs=1e-5)

    def test_different_windows_return_positive_shift(self):
        nav = _make_navigator_with_temporal(self.ENTRIES)
        call_count = [0]

        def _batched_iter(pid, batch_size=65_536, timestamp_from=None, timestamp_to=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter([_make_temporal_table(self.ENTRIES[:2]).to_batches()[0]])
            return iter([_make_temporal_table(self.ENTRIES[2:]).to_batches()[0]])

        nav._storage.read_temporal_batched.side_effect = _batched_iter
        result = nav.π11_attract_population_compare(
            "customer_pattern",
            "2024-01-01",
            "2024-03-31",
            "2024-06-01",
            "2024-09-30",
        )
        assert result["centroid_shift"] > 0.5
        assert "top_changed_dimensions" in result
        assert "interpretation" in result

    def test_empty_window_returns_warning(self):
        nav = _make_navigator_with_temporal(self.ENTRIES)
        nav._storage.read_temporal_batched.side_effect = lambda *a, **kw: iter([])
        result = nav.π11_attract_population_compare(
            "customer_pattern",
            "2099-01-01",
            "2099-12-31",
            "2099-01-01",
            "2099-12-31",
        )
        assert result.get("centroid_shift") is None
        assert "warning" in result


# ===========================================================================
# Task 3: detect_data_quality_issues
# ===========================================================================


class TestDetectDataQualityIssues:
    def test_clean_data_returns_no_findings(self):
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 10
        nav = _make_navigator_for_dqi(
            geo_rows=10, geo_anomalies=0, degenerate_count=0, edges_per_entity=edges
        )
        result = nav.detect_data_quality_issues("customer_pattern")
        assert result == []

    def test_required_line_low_coverage_is_high_severity(self):
        edges = (
            [[{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]]
            * 2  # noqa: E501
            + [[]] * 8
        )
        nav = _make_navigator_for_dqi(
            geo_rows=10, geo_anomalies=0, degenerate_count=0, edges_per_entity=edges
        )
        result = nav.detect_data_quality_issues("customer_pattern")
        coverage_findings = [f for f in result if f["issue_type"] == "low_coverage"]
        assert any(
            f["severity"] == "HIGH" and f["line_id"] == "products" for f in coverage_findings
        )

    def test_high_anomaly_rate_is_flagged(self):
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 10
        nav = _make_navigator_for_dqi(
            geo_rows=10, geo_anomalies=4, degenerate_count=0, edges_per_entity=edges
        )
        result = nav.detect_data_quality_issues("customer_pattern")
        assert any(f["issue_type"] == "high_anomaly_rate" for f in result)

    def test_zero_anomaly_rate_large_population_is_flagged(self):
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 10
        nav = _make_navigator_for_dqi(
            geo_rows=2000, geo_anomalies=0, degenerate_count=0, edges_per_entity=edges
        )
        result = nav.detect_data_quality_issues("customer_pattern")
        assert any(f["issue_type"] == "zero_anomaly_rate" for f in result)
        finding = next(f for f in result if f["issue_type"] == "zero_anomaly_rate")
        assert finding["severity"] == "MEDIUM"
        desc = finding["description"].lower()
        assert "theta" in desc or "recalibrat" in desc

    def test_zero_anomaly_rate_small_population_not_flagged(self):
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 10
        nav = _make_navigator_for_dqi(
            geo_rows=500, geo_anomalies=0, degenerate_count=0, edges_per_entity=edges
        )
        result = nav.detect_data_quality_issues("customer_pattern")
        assert not any(f["issue_type"] == "zero_anomaly_rate" for f in result)

    def test_theta_ceiling_distribution_is_flagged(self):
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 10
        nav = _make_navigator_for_dqi(
            geo_rows=1000,
            geo_anomalies=0,
            degenerate_count=0,
            edges_per_entity=edges,
            near_ceiling_count=700,  # 70% >= 0.75θ
        )
        result = nav.detect_data_quality_issues("customer_pattern")
        assert any(f["issue_type"] == "theta_ceiling" for f in result)
        finding = next(f for f in result if f["issue_type"] == "theta_ceiling")
        assert finding["severity"] == "MEDIUM"
        assert finding["pct"] == pytest.approx(0.7)

    def test_theta_ceiling_below_threshold_not_flagged(self):
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 10
        nav = _make_navigator_for_dqi(
            geo_rows=1000,
            geo_anomalies=0,
            degenerate_count=0,
            edges_per_entity=edges,
            near_ceiling_count=400,  # 40% < 50% threshold
        )
        result = nav.detect_data_quality_issues("customer_pattern")
        assert not any(f["issue_type"] == "theta_ceiling" for f in result)

    def test_detects_delta_norm_mismatch(self):
        """detect_data_quality_issues must flag delta_norm != ||delta|| mismatch."""
        edges_per_entity = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 2
        nav = _make_navigator_for_dqi(
            geo_rows=2, geo_anomalies=0, degenerate_count=0, edges_per_entity=edges_per_entity
        )
        # Table for call 1 (edges coverage check)
        _est = pa.struct(
            [
                pa.field("line_id", pa.string()),
                pa.field("point_key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("direction", pa.string()),
            ]
        )
        edges_cov_table = pa.table(
            {
                "edges": pa.array(edges_per_entity, type=pa.list_(_est)),
            }
        )
        # Table for call 2 (delta_norm mismatch check):
        # stored delta_norm=9.9 does NOT match ||delta||=1.0
        stale_table = pa.table(
            {
                "primary_key": pa.array(["E-0", "E-1"]),
                "delta_norm": pa.array([9.9, 9.9], type=pa.float32()),
                "delta": pa.array([[1.0], [1.0]], type=pa.list_(pa.float32())),
            }
        )
        call_count = [0]

        def _batched(pid, ver, columns=None, filter_expr=None, batch_size=65_536):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(edges_cov_table.to_batches())
            return iter(stale_table.to_batches())

        nav._storage.read_geometry_batched.side_effect = _batched

        result = nav.detect_data_quality_issues("customer_pattern")

        mismatch = [f for f in result if f["issue_type"] == "delta_norm_mismatch"]
        assert len(mismatch) == 1
        assert mismatch[0]["severity"] == "HIGH"
        assert mismatch[0]["stored_delta_norm"] == pytest.approx(9.9, abs=0.01)
        assert mismatch[0]["actual_delta_norm"] == pytest.approx(1.0, abs=0.01)
        assert "does not match" in mismatch[0]["message"]
        assert "recompute_delta_rank_pct" in mismatch[0]["message"]

    def test_no_delta_norm_mismatch_when_consistent(self):
        """detect_data_quality_issues must NOT flag delta_norm when it matches ||delta||."""
        edges_per_entity = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]
        ] * 2
        nav = _make_navigator_for_dqi(
            geo_rows=2, geo_anomalies=0, degenerate_count=0, edges_per_entity=edges_per_entity
        )
        _est = pa.struct(
            [
                pa.field("line_id", pa.string()),
                pa.field("point_key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("direction", pa.string()),
            ]
        )
        edges_cov_table = pa.table(
            {
                "edges": pa.array(edges_per_entity, type=pa.list_(_est)),
            }
        )
        # delta=[1.0], delta_norm=1.0 — consistent (no mismatch)
        consistent_table = pa.table(
            {
                "primary_key": pa.array(["E-0", "E-1"]),
                "delta_norm": pa.array([1.0, 1.0], type=pa.float32()),
                "delta": pa.array([[1.0], [1.0]], type=pa.list_(pa.float32())),
            }
        )
        call_count = [0]

        def _batched(pid, ver, columns=None, filter_expr=None, batch_size=65_536):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(edges_cov_table.to_batches())
            return iter(consistent_table.to_batches())

        nav._storage.read_geometry_batched.side_effect = _batched

        result = nav.detect_data_quality_issues("customer_pattern")

        assert not any(f["issue_type"] == "delta_norm_mismatch" for f in result)


# ===========================================================================
# Task 4: π12_attract_regime_change
# ===========================================================================


class TestFindRegimeChanges:
    def _make_nav_with_temporal(self, entries):
        nav = _make_navigator()
        nav._storage.read_temporal_centroids = lambda *a, **kw: None
        tbl = _make_temporal_table(entries)

        def _batched_iter(pid, batch_size=65_536, timestamp_from=None, timestamp_to=None):
            return iter([tbl.to_batches()[0]])

        nav._storage.read_temporal_batched.side_effect = _batched_iter
        return nav

    def test_no_history_returns_empty(self):
        nav = _make_navigator()
        nav._storage.read_temporal_centroids = lambda *a, **kw: None
        nav._storage.read_temporal_batched.side_effect = lambda *a, **kw: iter([])
        result = nav.π12_attract_regime_change("customer_pattern")
        assert len(result) == 1
        assert result[0].get("warning") is not None

    def test_event_pattern_raises_value_error(self):
        sphere = _make_sphere("gl_entry_pattern")
        sphere.patterns["gl_entry_pattern"] = _make_pattern("gl_entry_pattern", "event")
        nav = _make_navigator(sphere=sphere)
        nav._storage.read_temporal_centroids = lambda *a, **kw: None
        nav._storage.read_temporal_batched.side_effect = lambda *a, **kw: iter([])
        with pytest.raises(ValueError, match="anchor"):
            nav.π12_attract_regime_change("gl_entry_pattern")

    def test_detects_large_shift_as_regime_change(self):
        entries = [("C-1", f"2024-0{i + 1}-15", [0.1]) for i in range(8)] + [
            ("C-2", "2024-11-15", [2.0]),
            ("C-2", "2024-12-15", [2.0]),
        ]
        nav = self._make_nav_with_temporal(entries)
        result = nav.π12_attract_regime_change("customer_pattern", n_regimes=3)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["magnitude"] > 0
        assert "timestamp" in result[0]
        assert "top_changed_dimensions" in result[0]

    def test_time_based_bucketing_detects_concentrated_changepoint(self):
        """Time-based bucketing must detect a changepoint concentrated in a short window.

        100 baseline entries spread over 2 years (2022–2023) + 4 entries in Jan 2024
        with a large delta shift.  Count-based bucketing dilutes the 4-entry spike into
        a 26-entry bucket → signal lost.  Time-based bucketing isolates the spike →
        changepoint detected.
        """
        # 100 baseline entries spread across 2 years (every ~7 days)
        from datetime import timedelta

        base_start = datetime(2022, 1, 1, tzinfo=UTC)
        baseline = [
            (f"C-{i}", (base_start + timedelta(days=i * 7)).strftime("%Y-%m-%d"), [0.1])
            for i in range(100)
        ]
        # 4 entries concentrated in Jan 2024 — sharp regime change
        spike = [
            ("S-1", "2024-01-05", [8.0]),
            ("S-2", "2024-01-10", [8.0]),
            ("S-3", "2024-01-15", [8.0]),
            ("S-4", "2024-01-20", [8.0]),
        ]
        entries = baseline + spike
        nav = self._make_nav_with_temporal(entries)
        result = nav.π12_attract_regime_change("customer_pattern", n_regimes=3)
        assert len(result) >= 1, (
            "Time-based bucketing must detect the concentrated changepoint in Jan 2024; "
            "count-based bucketing dilutes it and returns 0 changepoints"
        )


# ===========================================================================
# Task 5: line_geometry_stats
# ===========================================================================


def _make_navigator_for_lgs(edges_per_entity, delta_norms=None):
    """Navigator mocked for line_geometry_stats with streaming API."""
    nav = _make_navigator(geo_rows=len(edges_per_entity))
    n = len(edges_per_entity)
    _est = pa.struct(
        [
            pa.field("line_id", pa.string()),
            pa.field("point_key", pa.string()),
            pa.field("status", pa.string()),
            pa.field("direction", pa.string()),
        ]
    )
    edges_arr = pa.array(edges_per_entity, type=pa.list_(_est))
    if delta_norms is None:
        delta_norms = [0.0] * n
    delta_arr = pa.array([[d] for d in delta_norms], type=pa.list_(pa.float32()))
    edges_table = pa.table({"edges": edges_arr})
    delta_table = pa.table({"delta": delta_arr})

    def _batched(pid, ver, columns=None, filter_expr=None, batch_size=65_536):
        if columns and "delta" in columns and "edges" not in columns:
            return iter(delta_table.to_batches())
        return iter(edges_table.to_batches())

    nav._storage.read_geometry_batched.side_effect = _batched
    nav._storage.count_geometry_rows.side_effect = lambda pid, ver, filter=None: 0 if filter else n
    return nav


class TestLineGeometryStats:
    def test_returns_correct_coverage(self):
        edges = (
            [[{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}]]
            * 7  # noqa: E501
            + [[]] * 3
        )
        nav = _make_navigator_for_lgs(edges)
        result = nav.line_geometry_stats("products", "customer_pattern")
        assert result["coverage_pct"] == pytest.approx(0.7)
        assert result["total_entities"] == 10
        assert result["required"] is True

    def test_edge_distribution_correct(self):
        edges = [
            [
                {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"},
                {"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "in"},
            ],
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}],
            [],
        ]
        nav = _make_navigator_for_lgs(edges)
        result = nav.line_geometry_stats("products", "customer_pattern")
        assert result["edge_distribution"]["2"] == 1
        assert result["edge_distribution"]["1"] == 1
        assert result["edge_distribution"]["0"] == 1

    def test_unknown_line_raises(self):
        nav = _make_navigator()
        with pytest.raises(ValueError, match="not a relation"):
            nav.line_geometry_stats("nonexistent_line", "customer_pattern")

    def test_mean_delta_contribution(self):
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}],
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}],
        ]
        nav = _make_navigator_for_lgs(edges, delta_norms=[0.4, 0.8])
        result = nav.line_geometry_stats("products", "customer_pattern")
        assert result["mean_delta_contribution"] == pytest.approx(0.6, abs=1e-5)

    def test_mean_delta_contribution_mixed_signs(self):
        # Entities have z-scored deltas of opposite sign (+1.0 and -1.0).
        # Signed mean would cancel to 0.0; abs mean must be 1.0.
        edges = [
            [{"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"}],
            [{"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "in"}],
        ]
        nav = _make_navigator_for_lgs(edges, delta_norms=[1.0, -1.0])
        result = nav.line_geometry_stats("products", "customer_pattern")
        assert result["mean_delta_contribution"] > 0.0, (
            "mean_delta_contribution must use abs z-score to avoid sign cancellation"
        )

    def test_three_plus_edges_bucket(self):
        edges = [
            [
                {"line_id": "products", "point_key": "P-1", "status": "alive", "direction": "in"},
                {"line_id": "products", "point_key": "P-2", "status": "alive", "direction": "in"},
                {"line_id": "products", "point_key": "P-3", "status": "alive", "direction": "in"},
            ],
        ]
        nav = _make_navigator_for_lgs(edges)
        result = nav.line_geometry_stats("products", "customer_pattern")
        assert result["edge_distribution"]["3+"] == 1

    def test_empty_geometry_returns_zeros(self):
        nav = _make_navigator_for_lgs([])
        result = nav.line_geometry_stats("products", "customer_pattern")
        assert result["total_entities"] == 0
        assert result["coverage_pct"] == 0.0
        assert result["mean_delta_contribution"] == 0.0


# ===========================================================================
# Task 6: check_alerts — implicit geometric health checks
# ===========================================================================


def _stats(total_entities=1000, total_anomalies=50, theta_norm=1.5, version=2):
    """Minimal geometry_stats dict."""
    return {
        "pattern_id": "customer_pattern",
        "version": version,
        "theta_norm": theta_norm,
        "total_entities": total_entities,
        "total_anomalies": total_anomalies,
        "percentiles": {
            "p50": 1.0,
            "p75": 1.5,
            "p90": 2.0,
            "p95": 2.5,
            "p99": 3.0,
            "max": 5.0,
        },
    }


def _make_alert_navigator(
    sphere=None,
    current_stats=None,
    prev_stats=None,
    pattern_version=2,
):
    """Navigator pre-wired for check_alerts tests.

    read_geometry_stats returns current_stats for the pattern_version
    and prev_stats for pattern_version - 1.
    """
    if sphere is None:
        sphere = _make_sphere("customer_pattern")
    if current_stats is None:
        current_stats = _stats()

    nav = _make_navigator(sphere=sphere, geo_stats=current_stats)

    # Override manifest to set the desired version
    manifest = _make_manifest(list(sphere.patterns.keys()))
    manifest = Manifest(
        manifest_id=manifest.manifest_id,
        agent_id=manifest.agent_id,
        snapshot_time=manifest.snapshot_time,
        status=manifest.status,
        line_versions=manifest.line_versions,
        pattern_versions=dict.fromkeys(sphere.patterns, pattern_version),
    )
    nav._manifest = manifest

    # Wire read_geometry_stats to dispatch on version
    def _read_stats(pid, ver):
        if ver == pattern_version:
            return current_stats
        if ver == pattern_version - 1:
            return prev_stats
        return None

    nav._storage.read_geometry_stats.side_effect = _read_stats

    # Default: data quality returns nothing, pi11/pi12 are safe
    nav.detect_data_quality_issues = MagicMock(return_value=[])
    nav.π11_attract_population_compare = MagicMock(return_value={"centroid_shift": 0.0})
    nav.π12_attract_regime_change = MagicMock(return_value=[])

    # Default: no calibration tracker — avoids MagicMock format issues
    nav._storage.read_calibration_tracker.return_value = None

    return nav


class TestCheckAlerts:
    # ── 1. anomaly_rate_spike detected ─────────────────────────────────
    def test_anomaly_rate_spike_detected(self):
        prev = _stats(total_entities=1000, total_anomalies=100)  # 10%
        cur = _stats(total_entities=1000, total_anomalies=200)  # 20%
        nav = _make_alert_navigator(current_stats=cur, prev_stats=prev)
        result = nav.check_alerts("customer_pattern")
        spike = [a for a in result["alerts"] if a["check_type"] == "anomaly_rate_spike"]
        assert len(spike) == 1
        assert spike[0]["severity"] == "HIGH"
        assert spike[0]["details"]["diff_pp"] == pytest.approx(0.1, abs=1e-3)

    # ── 2. no alerts when stable ───────────────────────────────────────
    def test_no_alerts_when_stable(self):
        prev = _stats(total_entities=1000, total_anomalies=100)  # 10%
        cur = _stats(total_entities=1000, total_anomalies=110)  # 11% (+1pp)
        nav = _make_alert_navigator(current_stats=cur, prev_stats=prev)
        result = nav.check_alerts("customer_pattern")
        spike = [a for a in result["alerts"] if a["check_type"] == "anomaly_rate_spike"]
        assert spike == []

    # ── 3. no previous version — skip comparison, no crash ─────────────
    def test_no_previous_version_no_crash(self):
        cur = _stats(version=1)
        nav = _make_alert_navigator(
            current_stats=cur,
            prev_stats=None,
            pattern_version=1,
        )
        result = nav.check_alerts("customer_pattern")
        # no anomaly_rate_spike or population_size_shock
        comparison = [
            a
            for a in result["alerts"]
            if a["check_type"] in ("anomaly_rate_spike", "population_size_shock")
        ]
        assert comparison == []
        assert result["patterns_checked"] == 1

    # ── 4. all patterns checked when pattern_id=None ───────────────────
    def test_all_patterns_checked(self):
        sphere = _make_sphere("p1", "p2", "p3")
        cur = _stats()
        nav = _make_alert_navigator(sphere=sphere, current_stats=cur)
        result = nav.check_alerts()  # pattern_id=None
        assert result["patterns_checked"] == 3

    # ── 5. population_size_shock — growth ──────────────────────────────
    def test_population_size_shock_growth(self):
        prev = _stats(total_entities=1000)
        cur = _stats(total_entities=1200)  # +20%
        nav = _make_alert_navigator(current_stats=cur, prev_stats=prev)
        result = nav.check_alerts("customer_pattern")
        shock = [a for a in result["alerts"] if a["check_type"] == "population_size_shock"]
        assert len(shock) == 1
        assert shock[0]["severity"] == "HIGH"
        assert shock[0]["details"]["change_pct"] == pytest.approx(0.2, abs=1e-3)

    # ── 6. population_size_shock — shrinkage ───────────────────────────
    def test_population_size_shock_shrinkage(self):
        prev = _stats(total_entities=1000)
        cur = _stats(total_entities=800)  # -20%
        nav = _make_alert_navigator(current_stats=cur, prev_stats=prev)
        result = nav.check_alerts("customer_pattern")
        shock = [a for a in result["alerts"] if a["check_type"] == "population_size_shock"]
        assert len(shock) == 1
        assert "shrank" in shock[0]["message"]

    # ── 7. data_quality_high — HIGH anomaly rate from stats ────────────
    def test_data_quality_high(self):
        # >50% anomaly rate → HIGH; 30–50% → MEDIUM (mirrors detect_data_quality_issues)
        cur_high = _stats(total_entities=1000, total_anomalies=600)  # 60% > 50%
        nav_high = _make_alert_navigator(current_stats=cur_high)
        result_high = nav_high.check_alerts("customer_pattern")
        dq_high = [
            a for a in result_high["alerts"] if a["check_type"] == "data_quality_high_anomaly_rate"
        ]
        assert len(dq_high) == 1
        assert dq_high[0]["severity"] == "HIGH"

        cur_med = _stats(total_entities=1000, total_anomalies=400)  # 40% → MEDIUM
        nav_med = _make_alert_navigator(current_stats=cur_med)
        result_med = nav_med.check_alerts("customer_pattern")
        dq_med = [
            a for a in result_med["alerts"] if a["check_type"] == "data_quality_high_anomaly_rate"
        ]
        assert len(dq_med) == 1
        assert dq_med[0]["severity"] == "MEDIUM"

    # ── 8. theta_miscalibration — zero anomalies on large population ───
    def test_theta_miscalibration(self):
        # zero anomalies on large population triggers theta_miscalibration
        cur = _stats(total_entities=2000, total_anomalies=0)
        nav = _make_alert_navigator(current_stats=cur)
        result = nav.check_alerts("customer_pattern")
        theta = [a for a in result["alerts"] if a["check_type"] == "theta_miscalibration"]
        assert len(theta) == 1
        assert theta[0]["severity"] == "MEDIUM"
        assert len(theta[0]["details"]["findings"]) >= 1

    # ── 10. regime_changepoint — changepoints detected ─────────────────
    def test_regime_changepoint(self):
        cur = _stats(version=5)
        nav = _make_alert_navigator(current_stats=cur, pattern_version=5)
        nav.π12_attract_regime_change = MagicMock(
            return_value=[
                {
                    "timestamp": "2024-06-15",
                    "magnitude": 1.2,
                    "top_changed_dimensions": [],
                    "description": "shift",
                },
            ]
        )
        result = nav.check_alerts("customer_pattern")
        regime = [a for a in result["alerts"] if a["check_type"] == "regime_changepoint"]
        assert len(regime) == 1
        assert regime[0]["severity"] == "HIGH"

    # ── 11. no regime_changepoint when pi12 empty ──────────────────────
    def test_no_regime_changepoint_when_empty(self):
        cur = _stats(version=5)
        nav = _make_alert_navigator(current_stats=cur, pattern_version=5)
        nav.π12_attract_regime_change = MagicMock(return_value=[])
        result = nav.check_alerts("customer_pattern")
        regime = [a for a in result["alerts"] if a["check_type"] == "regime_changepoint"]
        assert regime == []

    # ── 12. sorting — HIGH before MEDIUM ───────────────────────────────
    def test_sorting_high_before_medium(self):
        # HIGH: high anomaly rate (>30%) + MEDIUM: zero_anomaly_rate from different
        # pattern — use two patterns to get both severities at once.
        # Simpler: use high anomaly rate (HIGH) and theta=0 (MEDIUM) on same pattern.
        cur = _stats(total_entities=1000, total_anomalies=350, theta_norm=0.0)
        nav = _make_alert_navigator(current_stats=cur)
        result = nav.check_alerts("customer_pattern")
        severities = [a["severity"] for a in result["alerts"]]
        high_indices = [i for i, s in enumerate(severities) if s == "HIGH"]
        medium_indices = [i for i, s in enumerate(severities) if s == "MEDIUM"]
        if high_indices and medium_indices:
            assert max(high_indices) < min(medium_indices)

    # ── 13. _check_data_quality uses current_stats — high anomaly rate ──
    def test_data_quality_high_anomaly_rate_from_stats(self):
        """MEDIUM alert when 30–50% anomaly rate, computed from current_stats only."""
        cur = _stats(total_entities=1000, total_anomalies=350)  # 35% → MEDIUM
        nav = _make_alert_navigator(current_stats=cur)
        # detect_data_quality_issues must NOT be called (would require a full scan)
        nav.detect_data_quality_issues = MagicMock(
            side_effect=AssertionError(
                "_check_data_quality must not call detect_data_quality_issues"
            )
        )
        result = nav.check_alerts("customer_pattern")
        dq = [a for a in result["alerts"] if a["check_type"] == "data_quality_high_anomaly_rate"]
        assert len(dq) == 1
        assert dq[0]["severity"] == "MEDIUM"
        assert "35" in dq[0]["message"] or "anomaly" in dq[0]["message"].lower()

    # ── 14. _check_data_quality — zero anomaly rate on large population ─
    def test_data_quality_zero_anomaly_rate_from_stats(self):
        """MEDIUM theta_miscalibration when zero anomalies on large population."""
        cur = _stats(total_entities=2000, total_anomalies=0)
        nav = _make_alert_navigator(current_stats=cur)
        nav.detect_data_quality_issues = MagicMock(
            side_effect=AssertionError(
                "_check_data_quality must not call detect_data_quality_issues"
            )
        )
        result = nav.check_alerts("customer_pattern")
        theta = [a for a in result["alerts"] if a["check_type"] == "theta_miscalibration"]
        assert len(theta) == 1
        assert theta[0]["severity"] == "MEDIUM"

    # ── 15. _check_data_quality — theta=0 miscalibration ───────────────
    def test_data_quality_theta_zero_from_stats(self):
        """MEDIUM theta_miscalibration when theta_norm=0 on large population."""
        cur = _stats(total_entities=1500, total_anomalies=5, theta_norm=0.0)
        nav = _make_alert_navigator(current_stats=cur)
        nav.detect_data_quality_issues = MagicMock(
            side_effect=AssertionError(
                "_check_data_quality must not call detect_data_quality_issues"
            )
        )
        result = nav.check_alerts("customer_pattern")
        theta = [a for a in result["alerts"] if a["check_type"] == "theta_miscalibration"]
        assert len(theta) == 1
        assert theta[0]["severity"] == "MEDIUM"

    # ── 16. _check_regime_changepoint bounded to last 90 days ───────────
    def test_regime_changepoint_bounded_to_90_days(self):
        """π12_attract_regime_change must be called with timestamp_from as ISO str ~90 days ago."""
        from datetime import datetime, timedelta

        cur = _stats(version=5)
        nav = _make_alert_navigator(current_stats=cur, pattern_version=5)

        captured_kwargs: dict = {}

        def _capture_pi12(pid, **kwargs):
            captured_kwargs.update(kwargs)
            return []

        nav.π12_attract_regime_change = _capture_pi12
        nav.check_alerts("customer_pattern")

        assert "timestamp_from" in captured_kwargs, (
            "π12_attract_regime_change must be called with timestamp_from= keyword"
        )
        cutoff = captured_kwargs["timestamp_from"]
        # timestamp_from must be an ISO string (pi12 signature is str | None)
        assert isinstance(cutoff, str), (
            f"timestamp_from must be a str, got {type(cutoff)!r}: {cutoff!r}"
        )
        now = datetime.now(UTC)
        expected_cutoff = now - timedelta(days=90)
        parsed_cutoff = datetime.fromisoformat(cutoff)
        # Allow ±5 seconds tolerance
        assert abs((parsed_cutoff - expected_cutoff).total_seconds()) < 5, (
            f"timestamp_from={cutoff!r} is not ~90 days ago (expected ~{expected_cutoff!r})"
        )


# ===========================================================================
# Task 7: sphere_overview — geometry_mode detection
# ===========================================================================

_GEO_STATS_DEFAULT = {
    "total_entities": 1000,
    "total_anomalies": 50,
    "percentiles": {
        "p50": 1.0,
        "p75": 1.5,
        "p90": 2.0,
        "p95": 2.5,
        "p99": 3.0,
        "max": 5.0,
    },
}


def _make_delta_table(matrix: np.ndarray) -> pa.Table:
    """Create a PyArrow table with a FixedSizeList delta column from an ndarray."""
    flat = matrix.flatten().tolist()
    d = matrix.shape[1]
    return pa.table(
        {
            "delta": pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=pa.float32()),
                d,
            ),
        }
    )


class TestSphereOverviewGeometryMode:
    def test_binary_when_two_unique_values_per_dim(self):
        """All dims have exactly 2 unique values → geometry_mode == 'binary'."""
        rng = np.random.default_rng(42)
        # 4 dims, each only taking values 0.0 or 1.0 (≤3 unique)
        raw = rng.integers(0, 2, size=(50, 4)).astype(np.float32)
        tbl = _make_delta_table(raw)
        nav = _make_navigator(geo_stats=_GEO_STATS_DEFAULT)
        nav._storage.read_geometry.return_value = tbl
        result = nav.sphere_overview()
        assert result[0]["geometry_mode"] == "binary"

    def test_continuous_when_many_unique_values(self):
        """All dims have many unique random values → geometry_mode == 'continuous'."""
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((100, 4)).astype(np.float32)
        tbl = _make_delta_table(raw)
        nav = _make_navigator(geo_stats=_GEO_STATS_DEFAULT)
        nav._storage.read_geometry.return_value = tbl
        result = nav.sphere_overview()
        assert result[0]["geometry_mode"] == "continuous"

    def test_mixed_when_some_dims_binary(self):
        """2 binary dims + 2 continuous dims → geometry_mode == 'mixed'."""
        rng = np.random.default_rng(7)
        binary_part = rng.integers(0, 2, size=(60, 2)).astype(np.float32)
        continuous_part = rng.standard_normal((60, 2)).astype(np.float32)
        raw = np.concatenate([binary_part, continuous_part], axis=1)
        tbl = _make_delta_table(raw)
        nav = _make_navigator(geo_stats=_GEO_STATS_DEFAULT)
        nav._storage.read_geometry.return_value = tbl
        result = nav.sphere_overview()
        assert result[0]["geometry_mode"] == "mixed"

    def test_continuous_when_read_geometry_fails(self):
        """If read_geometry raises a recoverable error → default to 'continuous'."""
        nav = _make_navigator(geo_stats=_GEO_STATS_DEFAULT)
        nav._storage.read_geometry.side_effect = OSError("storage error")
        result = nav.sphere_overview()
        assert result[0]["geometry_mode"] == "continuous"

    def test_continuous_when_zero_entities(self):
        """If total entities == 0 → skip sampling, default to 'continuous'."""
        nav = _make_navigator(geo_stats=None, geo_rows=0, geo_anomalies=0)
        result = nav.sphere_overview()
        assert result[0]["geometry_mode"] == "continuous"


# ---------------------------------------------------------------------------
# _compute_event_rate_divergence
# ---------------------------------------------------------------------------


def _make_event_divergence_sphere(anchor_theta: float = 3.77):
    """Sphere with one anchor pattern and one event pattern linked by 'customers' line."""
    from hypertopos.model.sphere import Pattern, RelationDef

    anchor_pat = Pattern(
        pattern_id="anchor_pattern",
        entity_type="customers",
        pattern_type="anchor",
        entity_line_id="customers",
        relations=[],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.array([anchor_theta], dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    event_pat = Pattern(
        pattern_id="event_pattern",
        entity_type="events",
        pattern_type="event",
        entity_line_id="events",
        relations=[RelationDef(line_id="customers", direction="in", required=True)],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.array([2.0], dtype=np.float32),
        population_size=1000,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    sphere = MagicMock()
    sphere.patterns = {
        "anchor_pattern": anchor_pat,
        "event_pattern": event_pat,
    }
    sphere.entity_line.side_effect = lambda pid: (
        "customers" if pid == "anchor_pattern" else None
    )
    return sphere


def _make_divergence_nav(sphere=None):
    nav = _make_navigator(sphere=sphere or _make_event_divergence_sphere())
    nav._manifest = _make_manifest(["anchor_pattern", "event_pattern"])
    return nav


def _make_event_geo(entity_keys_per_row: list[list[str]], is_anomaly_per_row: list[bool]) -> pa.Table:
    """Build mock event geometry table with entity_keys + is_anomaly columns."""
    return pa.table({
        "entity_keys": pa.array(entity_keys_per_row, type=pa.list_(pa.string())),
        "is_anomaly": pa.array(is_anomaly_per_row, type=pa.bool_()),
    })


def _make_anchor_geo(keys: list[str], delta_norms: list[float], is_anomalies: list[bool]) -> pa.Table:
    """Build mock anchor geometry table."""
    return pa.table({
        "primary_key": pa.array(keys, type=pa.string()),
        "delta_norm": pa.array(delta_norms, type=pa.float32()),
        "is_anomaly": pa.array(is_anomalies, type=pa.bool_()),
    })


class TestComputeEventRateDivergence:
    """Tests for GDSNavigator._compute_event_rate_divergence().

    New implementation reads event geometry once (entity_keys + is_anomaly)
    and computes per-anchor rates from a single consistent sample.
    """

    def _setup_geo(self, nav, event_rows, anchor_keys, anchor_norms, anchor_anomalies):
        """Configure read_geometry mock: event geo first call, anchor geo second call."""
        event_table = _make_event_geo(*zip(*event_rows)) if event_rows else pa.table({
            "entity_keys": pa.array([], type=pa.list_(pa.string())),
            "is_anomaly": pa.array([], type=pa.bool_()),
        })
        anchor_table = _make_anchor_geo(anchor_keys, anchor_norms, anchor_anomalies)

        # First call = event geometry (sample_size present), second = anchor geometry (filter)
        nav._storage.read_geometry.side_effect = [event_table, anchor_table]

    def test_detects_high_event_anomaly_rate(self):
        """Alert when entity has >15% event anomaly rate AND delta_norm < theta."""
        nav = _make_divergence_nav()
        # CUST-001: 3/10 anomalous = 30% rate
        # CUST-002: 1/10 anomalous = 10% rate (below threshold)
        event_rows = [
            (["CUST-001"], True), (["CUST-001"], True), (["CUST-001"], True),
            (["CUST-001"], False), (["CUST-001"], False), (["CUST-001"], False),
            (["CUST-001"], False), (["CUST-001"], False), (["CUST-001"], False),
            (["CUST-001"], False),
            (["CUST-002"], True),
            (["CUST-002"], False), (["CUST-002"], False), (["CUST-002"], False),
            (["CUST-002"], False), (["CUST-002"], False), (["CUST-002"], False),
            (["CUST-002"], False), (["CUST-002"], False), (["CUST-002"], False),
        ]
        self._setup_geo(nav, event_rows, ["CUST-001"], [1.5], [False])

        alerts = nav._compute_event_rate_divergence()

        assert len(alerts) == 1
        a = alerts[0]
        assert a["entity_key"] == "CUST-001"
        assert a["pattern_id"] == "anchor_pattern"
        assert a["event_pattern_id"] == "event_pattern"
        assert a["event_anomaly_rate"] == pytest.approx(0.3)
        assert a["delta_norm"] == pytest.approx(1.5, abs=0.01)
        assert a["theta_norm"] == pytest.approx(3.77, abs=0.01)
        assert "investigate temporal" in a["alert"]

    def test_skips_entities_already_static_anomaly(self):
        """No alert when entity is already flagged as static anomaly (is_anomaly=True)."""
        nav = _make_divergence_nav()
        event_rows = [(["CUST-001"], True)] * 3 + [(["CUST-001"], False)] * 7
        self._setup_geo(nav, event_rows, ["CUST-001"], [5.0], [True])  # is_anomaly=True

        assert nav._compute_event_rate_divergence() == []

    def test_skips_entities_below_rate_threshold(self):
        """No alert when event anomaly rate is ≤15%."""
        nav = _make_divergence_nav()
        # 1/10 = 10% — below 15% threshold
        event_rows = [(["CUST-001"], True)] + [(["CUST-001"], False)] * 9
        self._setup_geo(nav, event_rows, ["CUST-001"], [1.0], [False])

        assert nav._compute_event_rate_divergence() == []

    def test_no_alerts_when_event_geo_empty(self):
        """No alerts when event geometry is empty."""
        nav = _make_divergence_nav()
        nav._storage.read_geometry.return_value = pa.table({
            "entity_keys": pa.array([], type=pa.list_(pa.string())),
            "is_anomaly": pa.array([], type=pa.bool_()),
        })

        assert nav._compute_event_rate_divergence() == []

    def test_sorted_descending_by_rate(self):
        """Alerts sorted by event_anomaly_rate descending."""
        nav = _make_divergence_nav()
        # CUST-A: 2/10 = 20%, CUST-B: 5/10 = 50%
        event_rows = (
            [(["CUST-A"], True)] * 2 + [(["CUST-A"], False)] * 8 +
            [(["CUST-B"], True)] * 5 + [(["CUST-B"], False)] * 5
        )
        self._setup_geo(nav, event_rows, ["CUST-A", "CUST-B"], [1.0, 1.0], [False, False])

        alerts = nav._compute_event_rate_divergence()
        assert len(alerts) == 2
        assert alerts[0]["entity_key"] == "CUST-B"  # 50%
        assert alerts[1]["entity_key"] == "CUST-A"  # 20%

    def test_geo_read_exception_is_swallowed(self):
        """If event geometry read raises FileNotFoundError, the pair is skipped."""
        nav = _make_divergence_nav()
        nav._storage.read_geometry.side_effect = FileNotFoundError("missing geometry")

        assert nav._compute_event_rate_divergence() == []

    def test_truncates_to_max_20_alerts(self):
        """More than 20 qualifying entities are truncated to top 20 by rate."""
        nav = _make_divergence_nav()
        # 25 entities, each with 50% event anomaly rate, all non-anomalous statically
        event_rows = []
        for i in range(25):
            key = f"CUST-{i:03d}"
            event_rows += [([key], True)] * 5 + [([key], False)] * 5
        anchor_keys = [f"CUST-{i:03d}" for i in range(25)]
        anchor_norms = [1.0] * 25
        anchor_anomalies = [False] * 25
        self._setup_geo(nav, event_rows, anchor_keys, anchor_norms, anchor_anomalies)

        alerts = nav._compute_event_rate_divergence()
        assert len(alerts) == 20
