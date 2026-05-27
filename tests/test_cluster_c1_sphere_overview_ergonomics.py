# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Cluster C1: per-entity anomaly_rate in aggregate +
cross_pattern_discrepancy in sphere_overview.

C1a (aggregate): when ``metric='count'`` and the result rows are not composite
(no ``group_by_property`` / ``group_by_line_2`` / ``pivot_event_field`` /
``distinct``), each row carries ``anomaly_rate: float | null`` =
``n_anomalous_events / n_total_events`` over the group's events.

C1b (sphere_overview): ``GDSNavigator._compute_cross_pattern_discrepancy``
returns pairwise Jaccard overlap of anomalous primary_keys across patterns
sharing an ``entity_line``. ``None`` on spheres with fewer than two
cover-overlapping patterns.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
from hypertopos.engine.aggregation import aggregate
from hypertopos.model.manifest import Manifest
from hypertopos.model.sphere import (
    Line,
    PartitionConfig,
    Pattern,
    RelationDef,
    Sphere,
)
from hypertopos.navigation.navigator import GDSNavigator

_DT = datetime(2024, 1, 1, tzinfo=UTC)
_PARTITION = PartitionConfig(mode="static", columns=[])

_EDGE_STRUCT_TYPE = pa.struct(
    [
        pa.field("line_id", pa.string()),
        pa.field("point_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("direction", pa.string()),
    ]
)


def _build_geo_with_anomaly(
    pks: list[str],
    customer_keys: list[str],
    is_anomaly: list[bool],
) -> pa.Table:
    """Build geometry table with edges + is_anomaly column."""
    n = len(pks)
    edges_rows = [
        [
            {
                "line_id": "customers",
                "point_key": customer_keys[i],
                "status": "alive",
                "direction": "in",
            },
        ]
        for i in range(n)
    ]
    return pa.table(
        {
            "primary_key": pks,
            "edges": pa.array(edges_rows, type=pa.list_(_EDGE_STRUCT_TYPE)),
            "is_anomaly": pa.array(is_anomaly, type=pa.bool_()),
        }
    )


def _make_event_pattern() -> Pattern:
    return Pattern(
        pattern_id="tx_pattern",
        entity_type="transaction",
        pattern_type="event",
        relations=[
            RelationDef(line_id="customers", direction="in", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=100,
        computed_at=_DT,
        version=1,
        status="production",
    )


def _make_event_sphere() -> Sphere:
    pat = _make_event_pattern()
    lines = {
        "transactions": Line(
            line_id="transactions",
            entity_type="transaction",
            line_role="event",
            pattern_id="tx_pattern",
            partitioning=_PARTITION,
            versions=[1],
        ),
        "customers": Line(
            line_id="customers",
            entity_type="customer",
            line_role="anchor",
            pattern_id="tx_pattern",
            partitioning=_PARTITION,
            versions=[1],
        ),
    }
    return Sphere(
        sphere_id="test",
        name="test",
        base_path=".",
        lines=lines,
        patterns={pat.pattern_id: pat},
    )


_MANIFEST_EVENT = Manifest(
    manifest_id="m",
    agent_id="a",
    snapshot_time=_DT,
    status="active",
    line_versions={"transactions": 1, "customers": 1},
    pattern_versions={"tx_pattern": 1},
)


class _MockReader:
    def __init__(self, geometry: pa.Table, points: dict[str, pa.Table] | None = None):
        self._geometry = geometry
        self._points = points or {}

    def read_geometry(
        self,
        pattern_id,
        version,
        *,
        point_keys=None,
        columns=None,
        filter=None,
        sample_size=None,
    ):
        geo = self._geometry
        if point_keys is not None:
            import pyarrow.compute as pc

            geo = geo.filter(
                pc.is_in(
                    geo["primary_key"],
                    value_set=pa.array(point_keys, type=pa.string()),
                ),
            )
        if columns is not None:
            available = [c for c in columns if c in geo.schema.names]
            geo = geo.select(available)
        return geo

    def count_geometry_rows(self, pattern_id, filter=None):
        raise RuntimeError("stub — use vectorized path")

    def read_points(self, line_id, version, **kwargs):
        tbl = self._points.get(line_id, pa.table({"primary_key": []}))
        columns = kwargs.get("columns")
        if columns is not None:
            available = [c for c in columns if c in tbl.schema.names]
            tbl = tbl.select(available)
        return tbl

    def read_points_schema(self, line_id, version):
        return self._points.get(line_id, pa.table({"primary_key": []})).schema

    def read_points_batch(self, line_id, version, primary_keys):
        tbl = self.read_points(line_id, version)
        if not primary_keys:
            return tbl.slice(0, 0)
        import pyarrow.compute as pc

        return tbl.filter(
            pc.is_in(
                tbl["primary_key"],
                value_set=pa.array(primary_keys, type=pa.string()),
            ),
        )


class _MockEngine:
    pass


# ============================================================================
# C1a — per-entity anomaly_rate in aggregate
# ============================================================================


class TestC1aAnomalyRatePerEntity:
    """aggregate(metric='count') without anomaly filter must emit per-row
    anomaly_rate = n_anom / n_total."""

    def test_anomaly_rate_per_entity_matches_closed_form(self):
        # CUST-A: 4 events, 1 anomalous → 0.25
        # CUST-B: 2 events, 2 anomalous → 1.0
        # CUST-C: 3 events, 0 anomalous → 0.0
        pks = [f"TX-{i:04d}" for i in range(9)]
        cust = [
            "CUST-A", "CUST-A", "CUST-A", "CUST-A",
            "CUST-B", "CUST-B",
            "CUST-C", "CUST-C", "CUST-C",
        ]
        anom = [
            True, False, False, False,
            True, True,
            False, False, False,
        ]
        geo = _build_geo_with_anomaly(pks, cust, anom)
        reader = _MockReader(
            geo,
            points={
                "customers": pa.table(
                    {"primary_key": ["CUST-A", "CUST-B", "CUST-C"]},
                ),
            },
        )
        sphere = _make_event_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST_EVENT,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
        )
        rows = {r["key"]: r for r in result["results"]}
        assert rows["CUST-A"]["anomaly_rate"] == round(1 / 4, 4)
        assert rows["CUST-B"]["anomaly_rate"] == round(2 / 2, 4)
        assert rows["CUST-C"]["anomaly_rate"] == 0.0

    def test_anomaly_rate_null_when_no_is_anomaly_column(self):
        """When the geometry schema lacks is_anomaly (pre-existing fixture
        contract), the field is omitted — no zero-div crash, no spurious
        anomaly_rate. Backwards-compatible with mocks that predate is_anomaly.
        """
        # Build geo WITHOUT is_anomaly column.
        pks = [f"TX-{i:04d}" for i in range(3)]
        cust = ["CUST-A", "CUST-A", "CUST-B"]
        edges_rows = [
            [
                {
                    "line_id": "customers",
                    "point_key": cust[i],
                    "status": "alive",
                    "direction": "in",
                },
            ]
            for i in range(3)
        ]
        geo = pa.table(
            {
                "primary_key": pks,
                "edges": pa.array(edges_rows, type=pa.list_(_EDGE_STRUCT_TYPE)),
            }
        )
        reader = _MockReader(
            geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
            },
        )
        sphere = _make_event_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST_EVENT,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
        )
        for row in result["results"]:
            assert "anomaly_rate" not in row

    def test_anomaly_rate_omitted_for_sum_metric(self):
        """Non-count metric → anomaly_rate is not added (rate semantics undefined)."""
        pks = [f"TX-{i:04d}" for i in range(3)]
        cust = ["CUST-A", "CUST-A", "CUST-B"]
        anom = [True, False, False]
        geo = _build_geo_with_anomaly(pks, cust, anom)
        tx_table = pa.table(
            {"primary_key": pks, "amount": [10.0, 20.0, 30.0]},
        )
        reader = _MockReader(
            geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_event_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST_EVENT,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="sum:amount",
        )
        for row in result["results"]:
            assert "anomaly_rate" not in row


# ============================================================================
# C1b — cross_pattern_discrepancy
# ============================================================================


class TestC1bCrossPatternDiscrepancy:
    """GDSNavigator._compute_cross_pattern_discrepancy emits pairwise Jaccard
    overlap of anomalous primary_keys across patterns sharing entity_line."""

    @staticmethod
    def _build_nav(
        anomalous_per_pattern: dict[str, set[str]],
        universe_per_pattern: dict[str, set[str]] | None = None,
        entity_line_per_pattern: dict[str, str | None] | None = None,
    ) -> GDSNavigator:
        """Construct a GDSNavigator with mocked storage + sphere just enough
        for _compute_cross_pattern_discrepancy to run.

        anomalous_per_pattern: {pat_id: {anomalous primary_keys}}
        universe_per_pattern: {pat_id: {all primary_keys}} (defaults to
            anomalous set + a few extra non-anomalous keys per pattern)
        entity_line_per_pattern: {pat_id: entity_line_id or None}
        """
        if universe_per_pattern is None:
            universe_per_pattern = {
                pid: anom | {f"{pid}_extra_{i}" for i in range(2)}
                for pid, anom in anomalous_per_pattern.items()
            }
        if entity_line_per_pattern is None:
            entity_line_per_pattern = dict.fromkeys(anomalous_per_pattern, "customers")

        sphere_mock = MagicMock()
        sphere_mock.patterns = {pid: MagicMock() for pid in anomalous_per_pattern}

        def _entity_line(pid: str) -> str | None:
            return entity_line_per_pattern.get(pid)

        sphere_mock.entity_line = _entity_line

        storage_mock = MagicMock()
        storage_mock.read_sphere.return_value = sphere_mock

        def _read_geometry(pat_id, version, *, columns=None, **kwargs):
            anom_set = anomalous_per_pattern[pat_id]
            universe = universe_per_pattern[pat_id]
            all_keys = list(universe)
            anom_flags = [k in anom_set for k in all_keys]
            cols = columns or ["primary_key", "is_anomaly"]
            data: dict = {}
            if "primary_key" in cols:
                data["primary_key"] = pa.array(all_keys, type=pa.string())
            if "is_anomaly" in cols:
                data["is_anomaly"] = pa.array(anom_flags, type=pa.bool_())
            return pa.table(data)

        storage_mock.read_geometry = _read_geometry

        manifest_mock = MagicMock()
        manifest_mock.pattern_version.return_value = 1

        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage_mock
        nav._manifest = manifest_mock
        nav._engine = MagicMock()
        nav._resolve_version = lambda pid: 1  # type: ignore[assignment]
        return nav

    def test_two_patterns_known_overlap(self):
        """Engineered anomaly sets — all 5 fields match closed form."""
        # pattern_a anomalies: {K1, K2, K3}
        # pattern_b anomalies: {K2, K3, K4}
        # both = {K2, K3} (size 2)
        # only_a = {K1} (size 1)
        # only_b = {K4} (size 1)
        # universe = {K1, K2, K3, K4, K5} (size 5, K5 in neither)
        # jaccard = 2 / 4 = 0.5
        nav = self._build_nav(
            anomalous_per_pattern={
                "pattern_a": {"K1", "K2", "K3"},
                "pattern_b": {"K2", "K3", "K4"},
            },
            universe_per_pattern={
                "pattern_a": {"K1", "K2", "K3", "K5"},
                "pattern_b": {"K2", "K3", "K4", "K5"},
            },
        )
        result = nav._compute_cross_pattern_discrepancy()
        assert result is not None
        assert len(result["pairs"]) == 1
        pair = result["pairs"][0]
        assert pair["pattern_a"] == "pattern_a"
        assert pair["pattern_b"] == "pattern_b"
        assert pair["shared_line"] == "customers"
        assert pair["n_anomalous_only_in_a"] == 1
        assert pair["n_anomalous_only_in_b"] == 1
        assert pair["n_anomalous_in_both"] == 2
        assert pair["n_anomalous_in_neither"] == 1
        assert pair["jaccard_anomaly_overlap"] == round(2 / 4, 4)

    def test_single_pattern_returns_none(self):
        """Sphere with only one pattern → no pairs → None."""
        nav = self._build_nav(
            anomalous_per_pattern={"pattern_only": {"K1", "K2"}},
        )
        assert nav._compute_cross_pattern_discrepancy() is None

    def test_no_shared_line_returns_none(self):
        """Two patterns but different entity_lines → no pairs → None."""
        nav = self._build_nav(
            anomalous_per_pattern={
                "pat_x": {"K1", "K2"},
                "pat_y": {"K3", "K4"},
            },
            entity_line_per_pattern={
                "pat_x": "line_x",
                "pat_y": "line_y",
            },
        )
        assert nav._compute_cross_pattern_discrepancy() is None

    def test_empty_anomaly_sets_yields_null_jaccard(self):
        """Both patterns have zero anomalies → jaccard is null (0/0), counts 0."""
        nav = self._build_nav(
            anomalous_per_pattern={
                "pat_a": set(),
                "pat_b": set(),
            },
            universe_per_pattern={
                "pat_a": {"K1", "K2"},
                "pat_b": {"K2", "K3"},
            },
        )
        result = nav._compute_cross_pattern_discrepancy()
        assert result is not None
        pair = result["pairs"][0]
        assert pair["n_anomalous_only_in_a"] == 0
        assert pair["n_anomalous_only_in_b"] == 0
        assert pair["n_anomalous_in_both"] == 0
        # universe = {K1, K2, K3}, neither = 3
        assert pair["n_anomalous_in_neither"] == 3
        assert pair["jaccard_anomaly_overlap"] is None

    def test_three_patterns_emit_three_pairs(self):
        """N=3 patterns sharing a line → C(3,2)=3 pairs."""
        nav = self._build_nav(
            anomalous_per_pattern={
                "pat_a": {"K1"},
                "pat_b": {"K1"},
                "pat_c": {"K2"},
            },
        )
        result = nav._compute_cross_pattern_discrepancy()
        assert result is not None
        assert len(result["pairs"]) == 3
        pair_keys = {(p["pattern_a"], p["pattern_b"]) for p in result["pairs"]}
        assert pair_keys == {
            ("pat_a", "pat_b"),
            ("pat_a", "pat_c"),
            ("pat_b", "pat_c"),
        }
