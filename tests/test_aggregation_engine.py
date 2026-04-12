# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for the core aggregation engine (packages/hypertopos-py/hypertopos/engine/aggregation.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.engine.aggregation import aggregate
from hypertopos.model.manifest import Manifest
from hypertopos.model.sphere import (
    Line,
    PartitionConfig,
    Pattern,
    RelationDef,
    Sphere,
)

# ---------------------------------------------------------------------------
# Shared helpers and mocks
# ---------------------------------------------------------------------------

_DT = datetime(2024, 1, 1, tzinfo=UTC)
_PARTITION = PartitionConfig(mode="static", columns=[])


def _make_pattern(
    pattern_id: str = "tx_pattern",
    entity_type: str = "transaction",
    relations: list[RelationDef] | None = None,
) -> Pattern:
    rels = relations or [
        RelationDef(
            line_id="customers",
            direction="in",
            required=True,
        ),
        RelationDef(
            line_id="products",
            direction="in",
            required=True,
        ),
    ]
    return Pattern(
        pattern_id=pattern_id,
        entity_type=entity_type,
        pattern_type="event",
        relations=rels,
        mu=np.zeros(len(rels), dtype=np.float32),
        sigma_diag=np.ones(len(rels), dtype=np.float32),
        theta=np.ones(len(rels), dtype=np.float32),
        population_size=100,
        computed_at=_DT,
        version=1,
        status="production",
    )


def _make_sphere(
    pattern: Pattern | None = None,
    extra_lines: dict[str, Line] | None = None,
) -> Sphere:
    pat = pattern or _make_pattern()
    lines: dict[str, Line] = {
        "transactions": Line(
            line_id="transactions",
            entity_type=pat.entity_type,
            line_role="event",
            pattern_id=pat.pattern_id,
            partitioning=_PARTITION,
            versions=[1],
        ),
        "customers": Line(
            line_id="customers",
            entity_type="customer",
            line_role="anchor",
            pattern_id=pat.pattern_id,
            partitioning=_PARTITION,
            versions=[1],
        ),
        "products": Line(
            line_id="products",
            entity_type="product",
            line_role="anchor",
            pattern_id="product_pattern",
            partitioning=_PARTITION,
            versions=[1],
        ),
    }
    if extra_lines:
        lines.update(extra_lines)
    return Sphere(
        sphere_id="test",
        name="test",
        base_path=".",
        lines=lines,
        patterns={pat.pattern_id: pat},
    )


_EDGE_STRUCT_TYPE = pa.struct(
    [
        pa.field("line_id", pa.string()),
        pa.field("point_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("direction", pa.string()),
    ]
)


def _make_geometry_table(
    n: int = 10,
    customer_keys: list[str] | None = None,
    product_keys: list[str] | None = None,
) -> pa.Table:
    """Build a minimal geometry table with edges struct column."""
    cust = customer_keys or [f"CUST-{i % 3:03d}" for i in range(n)]
    prod = product_keys or [f"PROD-{i % 2:03d}" for i in range(n)]
    edges_rows = [
        [
            {"line_id": "customers", "point_key": cust[i], "status": "alive", "direction": "in"},
            {"line_id": "products", "point_key": prod[i], "status": "alive", "direction": "in"},
        ]
        for i in range(n)
    ]
    return pa.table(
        {
            "primary_key": [f"TX-{i:04d}" for i in range(n)],
            "edges": pa.array(edges_rows, type=pa.list_(_EDGE_STRUCT_TYPE)),
        }
    )


def _make_entity_keys_geometry(
    n: int = 10,
    customer_keys: list[str] | None = None,
    product_keys: list[str] | None = None,
) -> pa.Table:
    """Build a geometry table with entity_keys only (no edges) — like event patterns."""
    cust = customer_keys or [f"CUST-{i % 3:03d}" for i in range(n)]
    prod = product_keys or [f"PROD-{i % 2:03d}" for i in range(n)]
    entity_keys_rows = [[cust[i], prod[i]] for i in range(n)]
    return pa.table(
        {
            "primary_key": [f"TX-{i:04d}" for i in range(n)],
            "entity_keys": pa.array(entity_keys_rows, type=pa.list_(pa.string())),
        }
    )


class _MockReader:
    """Minimal reader mock returning pre-configured tables."""

    def __init__(
        self,
        geometry: pa.Table | None = None,
        points: dict[str, pa.Table] | None = None,
    ):
        self._geometry = geometry or _make_geometry_table()
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

    def count_geometry_rows(self, pattern_id, version, filter=None):
        geo = self._geometry
        if filter and "entity_keys" in geo.schema.names:
            import re

            m = re.search(r"array_contains\(entity_keys,\s*'(.+?)'\)", filter)
            if m:
                target = m.group(1).replace("''", "'")
                ek = geo["entity_keys"]
                count = 0
                for i in range(len(geo)):
                    row_keys = ek[i].as_py()
                    if target in row_keys:
                        count += 1
                return count
        # No entity_keys column — fall through to vectorized path
        # (reversed scan catches this and uses read_geometry instead).
        raise RuntimeError("stub — use vectorized path")

    def read_points(self, line_id, version, **kwargs):
        tbl = self._points.get(line_id, pa.table({"primary_key": []}))
        columns = kwargs.get("columns")
        if columns is not None:
            available = [c for c in columns if c in tbl.schema.names]
            tbl = tbl.select(available)
        return tbl

    def read_points_schema(self, line_id, version):
        tbl = self._points.get(line_id, pa.table({"primary_key": []}))
        return tbl.schema

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


_MANIFEST = Manifest(
    manifest_id="test",
    agent_id="test-agent",
    snapshot_time=_DT,
    status="active",
    line_versions={"transactions": 1, "customers": 1, "products": 1},
    pattern_versions={"tx_pattern": 1},
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAggregateCountBasic:
    """Basic count aggregation via the vectorized fast path (C block)."""

    def test_count_groups(self):
        geo = _make_geometry_table(
            n=6,
            customer_keys=[
                "CUST-A",
                "CUST-A",
                "CUST-A",
                "CUST-B",
                "CUST-B",
                "CUST-C",
            ],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-A", "CUST-B", "CUST-C"],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
        )
        assert result["total_groups"] == 3
        assert result["sampled"] is False
        # Default sort=desc: CUST-A (3), CUST-B (2), CUST-C (1)
        assert result["results"][0]["key"] == "CUST-A"
        assert result["results"][0]["value"] == 3.0
        assert result["results"][0]["count"] == 3
        assert result["results"][1]["key"] == "CUST-B"
        assert result["results"][1]["value"] == 2.0
        assert result["results"][2]["key"] == "CUST-C"
        assert result["results"][2]["value"] == 1.0

    def test_count_sort_asc(self):
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B"],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-A", "CUST-B"],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            sort="asc",
        )
        assert result["results"][0]["key"] == "CUST-B"
        assert result["results"][1]["key"] == "CUST-A"

    def test_count_limit_offset(self):
        geo = _make_geometry_table(
            n=6,
            customer_keys=[
                "CUST-A",
                "CUST-A",
                "CUST-A",
                "CUST-B",
                "CUST-B",
                "CUST-C",
            ],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-A", "CUST-B", "CUST-C"],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            limit=1,
            offset=1,
        )
        assert len(result["results"]) == 1
        assert result["results"][0]["key"] == "CUST-B"


class TestAggregateMetricSum:
    """Sum metric aggregation via the in-process Arrow E block."""

    def test_sum_basic(self):
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        amounts = [100.0, 200.0, 50.0, 150.0]
        tx_table = pa.table(
            {
                "primary_key": pks,
                "amount": amounts[: len(pks)],
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-A", "CUST-B"],
                    }
                ),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="sum:amount",
        )
        assert result["total_groups"] == 2
        # CUST-A: 100+200=300, CUST-B: 50+150=200
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 300.0
        assert vals["CUST-B"] == 200.0

    def test_avg_basic(self):
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        amounts = [100.0, 200.0, 50.0, 150.0]
        tx_table = pa.table(
            {
                "primary_key": pks,
                "amount": amounts[: len(pks)],
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-A", "CUST-B"],
                    }
                ),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="avg:amount",
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 150.0
        assert vals["CUST-B"] == 100.0


class TestEventPatternEntityKeys:
    """Vectorized aggregation on event patterns using entity_keys (no edges column)."""

    def test_count_via_entity_keys(self):
        geo = _make_entity_keys_geometry(
            n=6,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B", "CUST-B", "CUST-C"],
        )
        reader = _MockReader(
            geometry=geo,
            points={"customers": pa.table({"primary_key": ["CUST-A", "CUST-B", "CUST-C"]})},
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
        )
        assert result["total_groups"] == 3
        assert result["results"][0]["key"] == "CUST-A"
        assert result["results"][0]["value"] == 3.0
        assert result["results"][1]["key"] == "CUST-B"
        assert result["results"][1]["value"] == 2.0

    def test_metric_sum_via_entity_keys(self):
        geo = _make_entity_keys_geometry(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": pa.table({
                    "primary_key": ["TX-0000", "TX-0001", "TX-0002", "TX-0003"],
                    "amount": [100.0, 200.0, 50.0, 75.0],
                }),
            },
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="sum:amount",
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 300.0
        assert vals["CUST-B"] == 125.0

    def test_sample_via_entity_keys(self):
        geo = _make_entity_keys_geometry(n=100)
        reader = _MockReader(
            geometry=geo,
            points={"customers": pa.table({"primary_key": [f"CUST-{i % 3:03d}" for i in range(3)]})},
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            sample_size=10,
            seed=42,
        )
        assert result["sampled"] is True
        assert result["sample_size"] == 10

    def test_dead_edges_excluded_in_entity_keys(self):
        """Empty string in entity_keys = dead edge, should be excluded."""
        entity_keys_rows = [["CUST-A", "PROD-1"], ["", "PROD-2"], ["CUST-A", ""]]
        geo = pa.table({
            "primary_key": ["TX-0", "TX-1", "TX-2"],
            "entity_keys": pa.array(entity_keys_rows, type=pa.list_(pa.string())),
        })
        reader = _MockReader(
            geometry=geo,
            points={"customers": pa.table({"primary_key": ["CUST-A"]})},
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
        )
        assert result["total_groups"] == 1
        assert result["results"][0]["key"] == "CUST-A"
        assert result["results"][0]["value"] == 2.0


class TestAggregateSampling:
    """Sampling reduces the population."""

    def test_sample_size(self):
        geo = _make_geometry_table(n=100)
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": [f"CUST-{i % 3:03d}" for i in range(3)],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            sample_size=10,
            seed=42,
        )
        assert result["sampled"] is True
        assert result["sample_size"] == 10
        assert result["total_eligible"] == 100

    def test_sample_pct(self):
        geo = _make_geometry_table(n=100)
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": [f"CUST-{i % 3:03d}" for i in range(3)],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            sample_pct=0.05,
            seed=42,
        )
        assert result["sampled"] is True
        assert result["sample_size"] == 5


class TestAggregateHaving:
    """Post-aggregation HAVING filter."""

    def test_having_gt(self):
        geo = _make_geometry_table(
            n=6,
            customer_keys=[
                "CUST-A",
                "CUST-A",
                "CUST-A",
                "CUST-B",
                "CUST-B",
                "CUST-C",
            ],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-A", "CUST-B", "CUST-C"],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            having={"gt": 1},
        )
        # Only CUST-A (3) and CUST-B (2) pass gt:1
        assert result["total_groups"] == 3  # pre-having count
        assert result["having_matched"] == 2
        keys = {r["key"] for r in result["results"]}
        assert keys == {"CUST-A", "CUST-B"}

    def test_having_lte(self):
        geo = _make_geometry_table(
            n=6,
            customer_keys=[
                "CUST-A",
                "CUST-A",
                "CUST-A",
                "CUST-B",
                "CUST-B",
                "CUST-C",
            ],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-A", "CUST-B", "CUST-C"],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            having={"lte": 2},
        )
        assert result["having_matched"] == 2
        keys = {r["key"] for r in result["results"]}
        assert keys == {"CUST-B", "CUST-C"}

    def test_total_groups_is_pre_having_count(self):
        """total_groups must reflect ALL groups, not just those passing having."""
        geo = _make_geometry_table(
            n=6,
            customer_keys=[
                "CUST-A", "CUST-A", "CUST-A",
                "CUST-B", "CUST-B",
                "CUST-C",
            ],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B", "CUST-C"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader, _MockEngine(), sphere, _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            having={"gt": 1},
        )
        # 3 groups total, only 2 pass gt:1
        assert result["total_groups"] == 3, (
            f"total_groups should be pre-having count (3), got {result['total_groups']}"
        )
        assert result["having_matched"] == 2
        assert "total_groups_before_having" not in result

    def test_total_groups_without_having_unchanged(self):
        """Without having, total_groups still reflects all groups."""
        geo = _make_geometry_table(
            n=6,
            customer_keys=[
                "CUST-A", "CUST-A", "CUST-A",
                "CUST-B", "CUST-B",
                "CUST-C",
            ],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B", "CUST-C"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader, _MockEngine(), sphere, _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
        )
        assert result["total_groups"] == 3
        assert "having_matched" not in result


class TestAggregateValidation:
    """Input validation raises RuntimeError."""

    def test_unknown_pattern(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="not found"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="no_such_pattern",
                group_by_line="customers",
            )

    def test_invalid_group_by_line(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="not a relation"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="no_such_line",
            )

    def test_distinct_requires_group_by_property(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="distinct.*group_by_property"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                distinct=True,
            )

    def test_sample_size_and_pct_exclusive(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                sample_size=10,
                sample_pct=0.1,
            )

    def test_metric_column_not_found_on_event_line(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(ValueError, match="not found on event entity line"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                metric="sum:amount",
            )

    def test_unknown_metric(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="Unknown metric"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                metric="bogus",
            )

    def test_filters_must_be_list(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="filters must be a list"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                filters={"line": "customers", "key": "CUST-A"},
            )


class TestAggregateReturnType:
    """Core aggregate returns dict (not JSON string)."""

    def test_returns_dict(self):
        geo = _make_geometry_table(n=3)
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table(
                    {
                        "primary_key": ["CUST-000", "CUST-001", "CUST-002"],
                    }
                ),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
        )
        assert isinstance(result, dict)
        assert "results" in result
        assert "total_groups" in result


class TestAggregateMultiLevel:
    """Multi-level GROUP BY (group_by_line_2)."""

    def test_multi_level_count(self):
        """Count by (customers x products) produces composite keys."""
        geo = _make_geometry_table(
            n=6,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B", "CUST-B", "CUST-C"],
            product_keys=["PROD-X", "PROD-X", "PROD-Y", "PROD-X", "PROD-Y", "PROD-X"],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B", "CUST-C"]}),
                "products": pa.table({"primary_key": ["PROD-X", "PROD-Y"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            group_by_line_2="products",
        )
        rows = result["results"]
        # CUST-A x PROD-X = 2, CUST-A x PROD-Y = 1, CUST-B x PROD-X = 1,
        # CUST-B x PROD-Y = 1, CUST-C x PROD-X = 1
        lookup = {(r["key"], r["key_2"]): r["value"] for r in rows}
        assert lookup[("CUST-A", "PROD-X")] == 2
        assert lookup[("CUST-A", "PROD-Y")] == 1
        assert lookup[("CUST-B", "PROD-X")] == 1
        assert result["group_by_line_2"] == "products"

    def test_multi_level_sum(self):
        """Sum metric by two lines."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
            product_keys=["PROD-X", "PROD-Y", "PROD-X", "PROD-X"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "amount": [10.0, 20.0, 30.0, 40.0][: len(pks)],
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "products": pa.table({"primary_key": ["PROD-X", "PROD-Y"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            group_by_line_2="products",
            metric="sum:amount",
        )
        lookup = {(r["key"], r["key_2"]): r["value"] for r in result["results"]}
        assert lookup[("CUST-A", "PROD-X")] == 10.0
        assert lookup[("CUST-A", "PROD-Y")] == 20.0
        assert lookup[("CUST-B", "PROD-X")] == 70.0  # 30 + 40

    def test_multi_level_rejects_with_group_by_property(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(  # noqa: E501
            RuntimeError,
            match="group_by_line_2 cannot be combined with group_by_property",
        ):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                group_by_line_2="products",
                group_by_property="customers:name",
            )

    def test_multi_level_rejects_with_pivot(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(
            RuntimeError,
            match="group_by_line_2 cannot be combined with pivot_event_field",
        ):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                group_by_line_2="products",
                pivot_event_field="category",
            )

    def test_multi_level_rejects_same_line(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="must differ"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                group_by_line_2="customers",
            )

    def test_multi_level_rejects_invalid_line(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="not a relation"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                group_by_line_2="nonexistent",
            )

    def test_multi_level_sorting_and_limit(self):
        """Results sorted desc by value, limited."""
        geo = _make_geometry_table(
            n=6,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B", "CUST-B", "CUST-C"],
            product_keys=["PROD-X", "PROD-X", "PROD-X", "PROD-X", "PROD-Y", "PROD-X"],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B", "CUST-C"]}),
                "products": pa.table({"primary_key": ["PROD-X", "PROD-Y"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            group_by_line_2="products",
            limit=2,
        )
        rows = result["results"]
        assert len(rows) == 2
        # Top by count: CUST-A x PROD-X = 3 (highest)
        assert rows[0]["key"] == "CUST-A"
        assert rows[0]["key_2"] == "PROD-X"
        assert rows[0]["value"] == 3


class TestAggregateDeprecatedParams:
    """Passing removed parameters raises ValueError with migration guidance."""

    def test_context_line_raises_deprecation(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(ValueError, match="context_line removed"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                metric="sum:amount",
                context_line="transactions",
            )

    def test_context_filters_raises_deprecation(self):
        reader = _MockReader()
        sphere = _make_sphere()
        with pytest.raises(ValueError, match="context_filters.*renamed.*event_filters"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                context_filters={"col": "val"},
            )


class TestAggregateColumnProjection:
    """Aggregate uses schema-only validation and column projection for metrics."""

    def test_validation_uses_schema_not_full_read(self):
        """agg_col validation must call read_points_schema, not read_points."""
        geo = _make_geometry_table(n=2, customer_keys=["CUST-A", "CUST-A"])
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "amount": [10.0, 20.0],
            }
        )

        class _SpyReader(_MockReader):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.schema_calls: list[tuple] = []
                self.points_calls: list[tuple] = []

            def read_points_schema(self, line_id, version):
                self.schema_calls.append((line_id, version))
                return super().read_points_schema(line_id, version)

            def read_points(self, line_id, version, **kwargs):
                self.points_calls.append((line_id, version, kwargs))
                return super().read_points(line_id, version, **kwargs)

        reader = _SpyReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="sum:amount",
        )
        # Validation must use schema, not full read
        assert any(lid == "transactions" for lid, _ in reader.schema_calls), (
            "expected read_points_schema for event line validation"
        )

    def test_metric_read_uses_column_projection(self):
        """In-process Arrow path must request only primary_key + agg_col."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "amount": [100.0, 200.0, 50.0, 150.0],
                "quantity": [1, 2, 3, 4],
                "extra_col": ["a", "b", "c", "d"],
            }
        )

        class _SpyReader(_MockReader):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.points_calls: list[tuple] = []

            def read_points(self, line_id, version, **kwargs):
                self.points_calls.append((line_id, version, kwargs))
                return super().read_points(line_id, version, **kwargs)

            def read_points_schema(self, line_id, version):
                return super().read_points_schema(line_id, version)

        reader = _SpyReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="sum:amount",
        )
        # Result must still be correct
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 300.0
        assert vals["CUST-B"] == 200.0
        # The event line read must use column projection
        event_reads = [(lid, kw) for lid, _, kw in reader.points_calls if lid == "transactions"]
        assert len(event_reads) >= 1
        for _lid, kw in event_reads:
            if kw.get("columns"):
                assert set(kw["columns"]) == {"primary_key", "amount"}


class TestAggregateEventFiltersNull:
    """event_filters with null/not_null support."""

    def test_null_filter_keeps_null_rows(self):
        """event_filters={"col": None} keeps only rows where col IS NULL."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "bank": pa.array(["AB", None, "CD", None], type=pa.string()),
                "amount": [100.0, 200.0, 50.0, 150.0],
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            event_filters={"bank": None},
        )
        # Only 2 rows have bank=null (TX-0001 for CUST-A, TX-0003 for CUST-B)
        assert result["total_groups"] == 2
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 1
        assert vals["CUST-B"] == 1

    def test_not_null_filter_keeps_non_null_rows(self):
        """event_filters={"col": {"not_null": True}} keeps only non-null rows."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "bank": pa.array(["AB", None, "CD", None], type=pa.string()),
                "amount": [100.0, 200.0, 50.0, 150.0],
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            event_filters={"bank": {"not_null": True}},
        )
        # Only 2 rows have bank != null (TX-0000 for CUST-A, TX-0002 for CUST-B)
        assert result["total_groups"] == 2
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 1
        assert vals["CUST-B"] == 1

    def test_null_filter_with_metric(self):
        """Null filter combined with sum metric."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "bank": pa.array(["AB", None, "CD", None], type=pa.string()),
                "amount": [100.0, 200.0, 50.0, 150.0],
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="sum:amount",
            event_filters={"bank": None},
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        # CUST-A: only null-bank row has amount=200
        assert vals["CUST-A"] == 200.0
        # CUST-B: only null-bank row has amount=150
        assert vals["CUST-B"] == 150.0

    def test_null_filter_on_numeric_column(self):
        """Null filter works on float64 columns too."""
        geo = _make_geometry_table(
            n=3,
            customer_keys=["CUST-A", "CUST-A", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "amount": pa.array([100.0, None, 50.0], type=pa.float64()),
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            event_filters={"amount": None},
        )
        assert result["total_groups"] == 1
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 1

    def test_not_null_mixed_with_other_ops_raises(self):
        """not_null combined with comparison ops raises error."""
        geo = _make_geometry_table(n=2, customer_keys=["CUST-A", "CUST-A"])
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table(
            {
                "primary_key": pks,
                "amount": [100.0, 200.0],
            }
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere()
        with pytest.raises(RuntimeError, match="not_null cannot be combined"):
            aggregate(
                reader,
                _MockEngine(),
                sphere,
                _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                metric="count",
                event_filters={"amount": {"not_null": True, "gt": 5}},
            )


class TestAggregateTimeWindow:
    """time_from/time_to inject event_filters using pattern.timestamp_col."""

    def _make_pattern_with_ts(self) -> Pattern:
        return Pattern(
            pattern_id="tx_pattern",
            entity_type="transaction",
            pattern_type="event",
            relations=[
                RelationDef(line_id="customers", direction="in", required=True),
                RelationDef(line_id="products", direction="in", required=True),
            ],
            mu=np.zeros(2, dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.ones(2, dtype=np.float32),
            population_size=100,
            computed_at=_DT,
            version=1,
            status="production",
            timestamp_col="orderdate",
        )

    def test_time_from_filters_events(self):
        """time_from injects gte event_filter on timestamp_col."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "orderdate": ["1994-06-01", "1995-03-15", "1994-11-20", "1995-07-10"],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere(pattern=self._make_pattern_with_ts())
        result = aggregate(
            reader, _MockEngine(), sphere, _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            time_from="1995-01-01",
        )
        # Only 2 rows >= 1995-01-01
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 1
        assert vals["CUST-B"] == 1

    def test_time_from_to_both(self):
        """time_from + time_to creates half-open interval [from, to)."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "orderdate": ["1994-06-01", "1995-03-15", "1995-07-10", "1996-01-05"],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere(pattern=self._make_pattern_with_ts())
        result = aggregate(
            reader, _MockEngine(), sphere, _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            time_from="1995-01-01",
            time_to="1996-01-01",
        )
        # Only 2 rows in [1995, 1996): 1995-03-15 and 1995-07-10
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 1
        assert vals["CUST-B"] == 1

    def test_raises_without_timestamp_col(self):
        """time_from on a pattern without timestamp_col raises ValueError."""
        geo = _make_geometry_table(n=2, customer_keys=["CUST-A", "CUST-A"])
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A"]}),
                "transactions": pa.table({"primary_key": geo["primary_key"].to_pylist()}),
            },
        )
        sphere = _make_sphere()  # default pattern has no timestamp_col
        with pytest.raises(ValueError, match="timestamp_col"):
            aggregate(
                reader, _MockEngine(), sphere, _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                metric="count",
                time_from="1995-01-01",
            )

    def test_raises_when_conflict_with_event_filters(self):
        """time_from + event_filters on same column raises ValueError."""
        geo = _make_geometry_table(n=2, customer_keys=["CUST-A", "CUST-A"])
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A"]}),
                "transactions": pa.table({"primary_key": geo["primary_key"].to_pylist()}),
            },
        )
        sphere = _make_sphere(pattern=self._make_pattern_with_ts())
        with pytest.raises(ValueError, match="Cannot use time_from"):
            aggregate(
                reader, _MockEngine(), sphere, _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                metric="count",
                time_from="1995-01-01",
                event_filters={"orderdate": {"gte": "1994-01-01"}},
            )

    def test_merges_with_other_event_filters(self):
        """time_from merges with event_filters on different columns."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "orderdate": ["1995-03-15", "1995-07-10", "1994-06-01", "1995-08-20"],
            "amount": [100.0, 200.0, 50.0, 300.0],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        sphere = _make_sphere(pattern=self._make_pattern_with_ts())
        result = aggregate(
            reader, _MockEngine(), sphere, _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            time_from="1995-01-01",
            event_filters={"amount": {"gt": 150}},
        )
        # time >= 1995 AND amount > 150: only 1995-07-10 (200) and 1995-08-20 (300)
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == 1
        assert vals["CUST-B"] == 1

    def test_rejects_invalid_date_format(self):
        """time_from with non-ISO date raises ValueError with clear message."""
        geo = _make_geometry_table(n=2, customer_keys=["CUST-A", "CUST-A"])
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A"]}),
                "transactions": pa.table({"primary_key": geo["primary_key"].to_pylist()}),
            },
        )
        sphere = _make_sphere(pattern=self._make_pattern_with_ts())
        with pytest.raises(ValueError, match="not a valid ISO-8601"):
            aggregate(
                reader, _MockEngine(), sphere, _MANIFEST,
                event_pattern_id="tx_pattern",
                group_by_line="customers",
                metric="count",
                time_from="not-a-date",
            )


class TestAnomalyRate:
    """Anomaly-rate enrichment when geometry_filters includes is_anomaly."""

    @staticmethod
    def _geo_with_anomaly(
        n: int,
        customer_keys: list[str],
        is_anomaly: list[bool],
    ) -> pa.Table:
        """Build geometry table with edges + is_anomaly column."""
        geo = _make_geometry_table(n=n, customer_keys=customer_keys)
        return geo.append_column("is_anomaly", pa.array(is_anomaly, type=pa.bool_()))

    def test_anomaly_rate_in_output(self):
        """aggregate with geometry_filters={'is_anomaly': True} returns anomaly_rate per row."""
        # 6 events: CUST-A has 3 total (2 anomalous), CUST-B has 2 total (1 anomalous),
        # CUST-C has 1 total (0 anomalous)
        geo = self._geo_with_anomaly(
            n=6,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B", "CUST-B", "CUST-C"],
            is_anomaly=[True, True, False, True, False, False],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B", "CUST-C"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            geometry_filters={"is_anomaly": True},
        )
        # Only anomalous rows: CUST-A=2, CUST-B=1; CUST-C has 0 anomalies
        assert result["total_groups"] == 2
        rows = {r["key"]: r for r in result["results"]}
        assert "CUST-A" in rows
        assert "CUST-B" in rows
        # Each row has total_events and anomaly_rate
        assert rows["CUST-A"]["total_events"] == 3
        assert rows["CUST-A"]["anomaly_rate"] == round(2 / 3, 4)
        assert rows["CUST-B"]["total_events"] == 2
        assert rows["CUST-B"]["anomaly_rate"] == round(1 / 2, 4)

    def test_no_anomaly_rate_without_geometry_filters(self):
        """Normal aggregate (no geometry_filters) does NOT have anomaly_rate."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
        )
        for row in result["results"]:
            assert "anomaly_rate" not in row
            assert "total_events" not in row

    def test_anomaly_rate_computation(self):
        """Verify rate = count / total_events with known data."""
        # 10 events: all for CUST-A, 3 anomalous
        geo = self._geo_with_anomaly(
            n=10,
            customer_keys=["CUST-A"] * 10,
            is_anomaly=[True, True, True, False, False, False, False, False, False, False],
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            geometry_filters={"is_anomaly": True},
        )
        assert len(result["results"]) == 1
        row = result["results"][0]
        assert row["key"] == "CUST-A"
        assert row["count"] == 3
        assert row["total_events"] == 10
        assert row["anomaly_rate"] == round(3 / 10, 4)

    def test_anomaly_rate_never_exceeds_1(self):
        """Regression: anomaly_rate must never exceed 1.0."""
        # 20 events for CUST-A: 5 anomalous, 15 normal
        geo = self._geo_with_anomaly(
            n=20,
            customer_keys=["CUST-A"] * 20,
            is_anomaly=[True] * 5 + [False] * 15,
        )
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A"]}),
            },
        )
        sphere = _make_sphere()
        result = aggregate(
            reader,
            _MockEngine(),
            sphere,
            _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="count",
            geometry_filters={"is_anomaly": True},
        )
        assert len(result["results"]) == 1
        row = result["results"][0]
        assert row["count"] == 5
        assert row["total_events"] == 20
        assert row["anomaly_rate"] == round(5 / 20, 4)
        assert row["anomaly_rate"] <= 1.0


class TestAggregateMedianBlockEPct:
    """Block E-pct: vectorized median/pct<N> via PyArrow+numpy path."""

    def test_median_basic_via_edges(self):
        """median:amount on anchor pattern (edges column) — sorted correctly."""
        # CUST-A: 100, 200, 300 → median 200.0
        # CUST-B: 50, 150 → median 100.0
        geo = _make_geometry_table(
            n=5,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "amount": [100.0, 200.0, 300.0, 50.0, 150.0],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="median:amount",
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == pytest.approx(200.0)
        assert vals["CUST-B"] == pytest.approx(100.0)
        assert result["sampled"] is False

    def test_pct75_basic_via_edges(self):
        """pct75:amount on anchor pattern — 75th percentile per group."""
        # CUST-A: 100, 200, 300 → pct75 = 275.0
        # CUST-B: 50, 150 → pct75 = 125.0
        geo = _make_geometry_table(
            n=5,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "amount": [100.0, 200.0, 300.0, 50.0, 150.0],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="pct75:amount",
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        # pct75 >= median for each group
        assert vals["CUST-A"] >= 200.0
        assert vals["CUST-B"] >= 100.0

    def test_median_via_entity_keys(self):
        """median:amount on event pattern (entity_keys column, no edges) works correctly."""
        # CUST-A: 10, 20, 30 → median 20.0
        # CUST-B: 5, 25 → median 15.0
        geo = _make_entity_keys_geometry(
            n=5,
            customer_keys=["CUST-A", "CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "amount": [10.0, 20.0, 30.0, 5.0, 25.0],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="median:amount",
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == pytest.approx(20.0)
        assert vals["CUST-B"] == pytest.approx(15.0)
        assert result["sampled"] is False

    def test_pct0_equals_min(self):
        """pct0 should return the minimum value for each group."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "amount": [100.0, 200.0, 50.0, 150.0],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="pct0:amount",
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == pytest.approx(100.0)
        assert vals["CUST-B"] == pytest.approx(50.0)

    def test_pct100_equals_max(self):
        """pct100 should return the maximum value for each group."""
        geo = _make_geometry_table(
            n=4,
            customer_keys=["CUST-A", "CUST-A", "CUST-B", "CUST-B"],
        )
        pks = geo["primary_key"].to_pylist()
        tx_table = pa.table({
            "primary_key": pks,
            "amount": [100.0, 200.0, 50.0, 150.0],
        })
        reader = _MockReader(
            geometry=geo,
            points={
                "customers": pa.table({"primary_key": ["CUST-A", "CUST-B"]}),
                "transactions": tx_table,
            },
        )
        result = aggregate(
            reader, _MockEngine(), _make_sphere(), _MANIFEST,
            event_pattern_id="tx_pattern",
            group_by_line="customers",
            metric="pct100:amount",
        )
        vals = {r["key"]: r["value"] for r in result["results"]}
        assert vals["CUST-A"] == pytest.approx(200.0)
        assert vals["CUST-B"] == pytest.approx(150.0)
