# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for subprocess_agg command dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from hypertopos.engine.subprocess_agg import (
    _COMMANDS,
    _TABLE_CACHE_MAX,
    _cmd_find_anomalies,
    _cmd_metric_aggregate,
    _cmd_pivot_aggregate,
    _cmd_property_aggregate,
    _dispatch,
    _ds_cache,
    _filtered_metric_aggregate_tables,
    _find_anomalies_table,
    _get_dataset,
    _get_table,
    _pivot_aggregate_tables,
    _property_aggregate_tables,
    _table_cache,
)

_EDGE_STRUCT = pa.struct(
    [
        pa.field("line_id", pa.string()),
        pa.field("point_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("direction", pa.string()),
    ]
)


def _edges_from_flat(line_ids_rows, point_keys_rows):
    """Build an edges struct array from flat line_ids/point_keys lists.

    All edges are marked alive/in by default.
    """
    rows = []
    for lids, pks in zip(line_ids_rows, point_keys_rows, strict=False):
        rows.append(
            [
                {"line_id": lid, "point_key": pk, "status": "alive", "direction": "in"}
                for lid, pk in zip(lids, pks, strict=False)
            ]
        )
    return pa.array(rows, type=pa.list_(_EDGE_STRUCT))


class TestDispatch:
    """Test command dispatch logic."""

    def test_unknown_command_returns_error(self):
        result = _dispatch({"command": "nonexistent"})
        assert "error" in result
        assert "unknown command" in result["error"]

    def test_missing_command_defaults_to_metric_aggregate(self):
        """Missing command field should default to metric_aggregate."""
        mock_fn = MagicMock(return_value={"metrics": {}, "counts": {}})
        orig = _COMMANDS["metric_aggregate"]
        _COMMANDS["metric_aggregate"] = mock_fn
        try:
            result = _dispatch({"geo_lance_path": "/fake"})
            mock_fn.assert_called_once_with(geo_lance_path="/fake")
            assert result == {"metrics": {}, "counts": {}}
        finally:
            _COMMANDS["metric_aggregate"] = orig

    def test_explicit_metric_aggregate_command(self):
        """Explicit command=metric_aggregate dispatches correctly."""
        mock_fn = MagicMock(return_value={"metrics": {}, "counts": {}})
        orig = _COMMANDS["metric_aggregate"]
        _COMMANDS["metric_aggregate"] = mock_fn
        try:
            result = _dispatch(
                {
                    "command": "metric_aggregate",
                    "geo_lance_path": "/fake",
                }
            )
            mock_fn.assert_called_once_with(geo_lance_path="/fake")
            assert result == {"metrics": {}, "counts": {}}
        finally:
            _COMMANDS["metric_aggregate"] = orig

    def test_anomaly_summary_stub_returns_not_implemented(self):
        result = _dispatch({"command": "anomaly_summary"})
        assert "error" in result
        assert "not implemented" in result["error"]

    def test_command_field_is_consumed(self):
        """The command field should not be passed to the handler."""
        mock_fn = MagicMock(return_value={"ok": True})
        orig = _COMMANDS["metric_aggregate"]
        _COMMANDS["metric_aggregate"] = mock_fn
        try:
            _dispatch({"command": "metric_aggregate", "x": 1})
            # 'command' must not appear in kwargs
            mock_fn.assert_called_once_with(x=1)
        finally:
            _COMMANDS["metric_aggregate"] = orig


class TestCmdMetricAggregate:
    """Verify _cmd_metric_aggregate is the renamed run_aggregate."""

    def test_function_exists_and_is_callable(self):
        assert callable(_cmd_metric_aggregate)

    def test_registered_in_commands_dict(self):
        assert _COMMANDS["metric_aggregate"] is _cmd_metric_aggregate


def _make_pivot_tables():
    """Build minimal geo + event Arrow tables for pivot tests."""
    # 4 polygons, each with edges to "customers" line
    lids = [["customers"], ["customers"], ["customers"], ["customers"]]
    pks = [["CUST-A"], ["CUST-A"], ["CUST-B"], ["CUST-B"]]
    geo = pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003", "E-004"],
            "edges": _edges_from_flat(lids, pks),
        }
    )
    # Event points with fiscal_year pivot field
    event = pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003", "E-004"],
            "fiscal_year": [2024, 2024, 2024, 2025],
        }
    )
    return geo, event


class TestPivotAggregateTables:
    """Test _pivot_aggregate_tables with in-memory Arrow tables."""

    def test_count_mode(self):
        geo, event = _make_pivot_tables()
        results = _pivot_aggregate_tables(
            geo,
            event,
            "customers",
            "fiscal_year",
        )
        # Convert to dict for easy assertion
        by_key = {(r["group_key"], r["pivot_val"]): r for r in results}
        assert by_key[("CUST-A", "2024")]["count"] == 2
        assert by_key[("CUST-B", "2024")]["count"] == 1
        assert by_key[("CUST-B", "2025")]["count"] == 1
        # No metric field in count mode
        assert "metric" not in by_key[("CUST-A", "2024")]

    def test_metric_mode_sum(self):
        geo, event = _make_pivot_tables()
        ctx = pa.table(
            {
                "primary_key": ["E-001", "E-002", "E-003", "E-004"],
                "amount": [100.0, 200.0, 50.0, 75.0],
            }
        )
        results = _pivot_aggregate_tables(
            geo,
            event,
            "customers",
            "fiscal_year",
            ctx_table=ctx,
            metric_col="amount",
            agg_fn="sum",
        )
        by_key = {(r["group_key"], r["pivot_val"]): r for r in results}
        assert by_key[("CUST-A", "2024")]["count"] == 2
        assert by_key[("CUST-A", "2024")]["metric"] == pytest.approx(300.0)
        assert by_key[("CUST-B", "2024")]["metric"] == pytest.approx(50.0)
        assert by_key[("CUST-B", "2025")]["metric"] == pytest.approx(75.0)

    def test_no_matching_edges_returns_empty(self):
        geo, event = _make_pivot_tables()
        results = _pivot_aggregate_tables(
            geo,
            event,
            "nonexistent_line",
            "fiscal_year",
        )
        assert results == []


class TestCmdPivotAggregate:
    """Verify _cmd_pivot_aggregate is registered and callable."""

    def test_function_exists_and_is_callable(self):
        assert callable(_cmd_pivot_aggregate)

    def test_registered_in_commands_dict(self):
        assert _COMMANDS["pivot_aggregate"] is _cmd_pivot_aggregate


class TestCmdPivotAggregateDataFusion:
    """Verify _cmd_pivot_aggregate uses DataFusion fast path for count-only mode."""

    def test_datafusion_path_used_for_count_only(self, monkeypatch):
        """When metric_col is None and DataFusion is available, uses DataFusion."""
        from unittest.mock import MagicMock, patch

        import pyarrow as pa

        geo = pa.table(
            {
                "primary_key": ["E-001", "E-002", "E-003"],
                "edges": _edges_from_flat(
                    [["customers"], ["customers"], ["customers"]],
                    [["C1"], ["C1"], ["C2"]],
                ),
            }
        )
        event = pa.table(
            {
                "primary_key": ["E-001", "E-002", "E-003"],
                "season": ["summer", "winter", "summer"],
            }
        )

        expected_results = [
            {"group_key": "C1", "pivot_val": "summer", "count": 1},
            {"group_key": "C1", "pivot_val": "winter", "count": 1},
            {"group_key": "C2", "pivot_val": "summer", "count": 1},
        ]

        mock_df_pivot = MagicMock(return_value=expected_results)

        with patch("hypertopos.engine.subprocess_agg._get_table") as mock_get_table:
            mock_get_table.side_effect = [geo, event]
            with (
                patch(
                    "hypertopos.engine.datafusion_agg.is_available",
                    return_value=True,
                ),
                patch(
                    "hypertopos.engine.datafusion_agg.aggregate_pivot_count",
                    mock_df_pivot,
                ),
            ):
                result = _cmd_pivot_aggregate(
                    geo_lance_path="/fake/geo.lance",
                    event_lance_path="/fake/event.lance",
                    group_by_line="customers",
                    pivot_field="season",
                    geo_columns=["primary_key", "edges"],
                )

        assert result == {"results": expected_results}

    def test_falls_back_to_arrow_when_datafusion_unavailable(self, monkeypatch):
        """When DataFusion is not available, falls back to _pivot_aggregate_tables."""
        import hypertopos.engine.datafusion_agg as df_mod

        monkeypatch.setattr(df_mod, "_datafusion", None)

        geo, event = _make_pivot_tables()

        with patch("hypertopos.engine.subprocess_agg._get_table") as mock_get_table:
            mock_get_table.side_effect = [geo, event]
            result = _cmd_pivot_aggregate(
                geo_lance_path="/fake/geo.lance",
                event_lance_path="/fake/event.lance",
                group_by_line="customers",
                pivot_field="fiscal_year",
                geo_columns=["primary_key", "edges"],
            )

        assert "results" in result
        by_key = {(r["group_key"], r["pivot_val"]): r["count"] for r in result["results"]}
        assert by_key[("CUST-A", "2024")] == 2
        assert by_key[("CUST-B", "2024")] == 1
        assert by_key[("CUST-B", "2025")] == 1

    def test_metric_mode_skips_datafusion(self):
        """When metric_col is set, DataFusion fast path is skipped."""
        geo, event = _make_pivot_tables()
        ctx = pa.table(
            {
                "primary_key": ["E-001", "E-002", "E-003", "E-004"],
                "amount": [100.0, 200.0, 50.0, 75.0],
            }
        )

        mock_df_pivot = MagicMock()

        with patch("hypertopos.engine.subprocess_agg._get_table") as mock_get_table:
            mock_get_table.side_effect = [geo, event, ctx]
            with patch(
                "hypertopos.engine.datafusion_agg.aggregate_pivot_count",
                mock_df_pivot,
            ):
                result = _cmd_pivot_aggregate(
                    geo_lance_path="/fake/geo.lance",
                    event_lance_path="/fake/event.lance",
                    group_by_line="customers",
                    pivot_field="fiscal_year",
                    geo_columns=["primary_key", "edges"],
                    ctx_lance_path="/fake/ctx.lance",
                    metric_col="amount",
                    agg_fn="sum",
                )

        # DataFusion pivot count should NOT have been called (metric_col is set)
        mock_df_pivot.assert_not_called()
        assert "results" in result


def _make_property_tables_same_line():
    """Build geo + prop tables where prop_line_id == group_by_line."""
    # 4 polygons, each with edges to "customers" line
    geo = pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003", "E-004"],
            "edges": _edges_from_flat(
                [["customers"], ["customers"], ["customers"], ["customers"]],
                [["CUST-A"], ["CUST-A"], ["CUST-B"], ["CUST-B"]],
            ),
        }
    )
    # Property table: customers with region property
    prop = pa.table(
        {
            "primary_key": ["CUST-A", "CUST-B"],
            "region": ["EMEA", "APAC"],
        }
    )
    return geo, prop


def _make_property_tables_different_line():
    """Build geo + prop tables where prop_line_id != group_by_line."""
    # 4 polygons with edges to both "customers" and "company_codes"
    geo = pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003", "E-004"],
            "edges": _edges_from_flat(
                [["customers", "company_codes"]] * 4,
                [
                    ["CUST-A", "CC-01"],
                    ["CUST-A", "CC-01"],
                    ["CUST-B", "CC-02"],
                    ["CUST-B", "CC-02"],
                ],
            ),
        }
    )
    # Property table: company_codes with country property
    prop = pa.table(
        {
            "primary_key": ["CC-01", "CC-02"],
            "country": ["DE", "US"],
        }
    )
    return geo, prop


class TestPropertyAggregateTables:
    """Test _property_aggregate_tables with in-memory Arrow tables."""

    def test_count_same_line(self):
        geo, prop = _make_property_tables_same_line()
        result = _property_aggregate_tables(
            geo,
            prop,
            "customers",
            "customers",
            "region",
        )
        assert "results" in result
        by_key = {(r["group_key"], r["prop_val"]): r for r in result["results"]}
        assert by_key[("CUST-A", "EMEA")]["count"] == 2
        assert by_key[("CUST-B", "APAC")]["count"] == 2

    def test_distinct_same_line(self):
        geo, prop = _make_property_tables_same_line()
        result = _property_aggregate_tables(
            geo,
            prop,
            "customers",
            "customers",
            "region",
            distinct=True,
        )
        assert "distinct_results" in result
        assert result["distinct_results"]["EMEA"] == 1  # only CUST-A
        assert result["distinct_results"]["APAC"] == 1  # only CUST-B

    def test_count_different_line(self):
        geo, prop = _make_property_tables_different_line()
        result = _property_aggregate_tables(
            geo,
            prop,
            "customers",
            "company_codes",
            "country",
        )
        assert "results" in result
        by_key = {(r["group_key"], r["prop_val"]): r for r in result["results"]}
        assert by_key[("CUST-A", "DE")]["count"] == 2
        assert by_key[("CUST-B", "US")]["count"] == 2

    def test_distinct_different_line(self):
        geo, prop = _make_property_tables_different_line()
        result = _property_aggregate_tables(
            geo,
            prop,
            "customers",
            "company_codes",
            "country",
            distinct=True,
        )
        assert "distinct_results" in result
        assert result["distinct_results"]["DE"] == 1
        assert result["distinct_results"]["US"] == 1

    def test_metric_sum_same_line(self):
        geo, prop = _make_property_tables_same_line()
        ctx = pa.table(
            {
                "primary_key": ["E-001", "E-002", "E-003", "E-004"],
                "amount": [100.0, 200.0, 50.0, 75.0],
            }
        )
        result = _property_aggregate_tables(
            geo,
            prop,
            "customers",
            "customers",
            "region",
            ctx_table=ctx,
            metric_col="amount",
            agg_fn="sum",
        )
        assert "results" in result
        by_key = {(r["group_key"], r["prop_val"]): r for r in result["results"]}
        assert by_key[("CUST-A", "EMEA")]["metric"] == pytest.approx(300.0)
        assert by_key[("CUST-B", "APAC")]["metric"] == pytest.approx(125.0)
        assert by_key[("CUST-A", "EMEA")]["count"] == 2

    def test_no_matching_edges_returns_empty(self):
        geo, prop = _make_property_tables_same_line()
        result = _property_aggregate_tables(
            geo,
            prop,
            "nonexistent_line",
            "customers",
            "region",
        )
        assert result["results"] == []


class TestCmdPropertyAggregate:
    """Verify _cmd_property_aggregate is registered and callable."""

    def test_function_exists_and_is_callable(self):
        assert callable(_cmd_property_aggregate)

    def test_registered_in_commands_dict(self):
        assert _COMMANDS["property_aggregate"] is _cmd_property_aggregate


def _make_filtered_tables():
    """Build geo + ctx tables for filtered metric aggregate tests.

    4 polygons with edges to customers and company_codes.
    Filters will select subsets based on edge membership.
    """
    geo = pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003", "E-004"],
            "edges": _edges_from_flat(
                [["customers", "company_codes"]] * 4,
                [
                    ["CUST-A", "CC-01"],
                    ["CUST-A", "CC-01"],
                    ["CUST-B", "CC-02"],
                    ["CUST-B", "CC-02"],
                ],
            ),
        }
    )
    ctx = pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003", "E-004"],
            "amount_local": [100.0, 200.0, 50.0, 75.0],
        }
    )
    return geo, ctx


class TestFilteredMetricAggregateTables:
    """Test _filtered_metric_aggregate_tables with in-memory Arrow tables."""

    def test_sum_with_passing_filter(self):
        geo, ctx = _make_filtered_tables()
        # Filter: only CC-01 company code → E-001, E-002 pass
        result = _filtered_metric_aggregate_tables(
            geo,
            ctx,
            "customers",
            "amount_local",
            "sum",
            resolved_filters=[["company_codes", ["CC-01"]]],
            event_line_id="gl_entries",
        )
        assert result["metrics"]["CUST-A"] == pytest.approx(300.0)
        assert "CUST-B" not in result["metrics"]
        assert result["counts"]["CUST-A"] == 2

    def test_sum_with_filter_excluding_some(self):
        geo, ctx = _make_filtered_tables()
        # Filter: only CC-02 → E-003, E-004 pass
        result = _filtered_metric_aggregate_tables(
            geo,
            ctx,
            "customers",
            "amount_local",
            "sum",
            resolved_filters=[["company_codes", ["CC-02"]]],
            event_line_id="gl_entries",
        )
        assert "CUST-A" not in result["metrics"]
        assert result["metrics"]["CUST-B"] == pytest.approx(125.0)
        assert result["counts"]["CUST-B"] == 2

    def test_avg_metric(self):
        geo, ctx = _make_filtered_tables()
        # All pass (both CC-01 and CC-02 allowed)
        result = _filtered_metric_aggregate_tables(
            geo,
            ctx,
            "customers",
            "amount_local",
            "avg",
            resolved_filters=[["company_codes", ["CC-01", "CC-02"]]],
            event_line_id="gl_entries",
        )
        # CUST-A: avg(100, 200) = 150
        assert result["metrics"]["CUST-A"] == pytest.approx(150.0)
        # CUST-B: avg(50, 75) = 62.5
        assert result["metrics"]["CUST-B"] == pytest.approx(62.5)

    def test_empty_result_when_no_polygons_pass(self):
        geo, ctx = _make_filtered_tables()
        # Filter: nonexistent company code → nothing passes
        result = _filtered_metric_aggregate_tables(
            geo,
            ctx,
            "customers",
            "amount_local",
            "sum",
            resolved_filters=[["company_codes", ["CC-99"]]],
            event_line_id="gl_entries",
        )
        assert result == {"metrics": {}, "counts": {}}

    def test_event_line_filter(self):
        geo, ctx = _make_filtered_tables()
        # Filter on event_line_id itself → only E-001 allowed
        result = _filtered_metric_aggregate_tables(
            geo,
            ctx,
            "customers",
            "amount_local",
            "sum",
            resolved_filters=[["gl_entries", ["E-001"]]],
            event_line_id="gl_entries",
        )
        assert result["metrics"]["CUST-A"] == pytest.approx(100.0)
        assert result["counts"]["CUST-A"] == 1
        assert "CUST-B" not in result["metrics"]

    def test_multiple_filters_and_logic(self):
        geo, ctx = _make_filtered_tables()
        # Both filters must pass (AND): CC-01 AND CUST-A
        result = _filtered_metric_aggregate_tables(
            geo,
            ctx,
            "customers",
            "amount_local",
            "sum",
            resolved_filters=[
                ["company_codes", ["CC-01"]],
                ["customers", ["CUST-A"]],
            ],
            event_line_id="gl_entries",
        )
        assert result["metrics"]["CUST-A"] == pytest.approx(300.0)
        assert "CUST-B" not in result["metrics"]

    def test_with_alive_mask(self):
        """Polygons with dead edges should not match filters on dead lines."""
        geo = pa.table(
            {
                "primary_key": ["E-001", "E-002"],
                "edges": _edges_from_flat(
                    [["customers", "company_codes"], ["customers", "company_codes"]],
                    [["CUST-A", "CC-01"], ["CUST-A", "CC-01"]],
                ),
            }
        )
        # Manually mark CC-01 edge as dead for E-002
        edges_py = geo["edges"].to_pylist()
        edges_py[1][1]["status"] = "dead"
        geo = pa.table(
            {
                "primary_key": ["E-001", "E-002"],
                "edges": pa.array(edges_py, type=pa.list_(_EDGE_STRUCT)),
            }
        )
        ctx = pa.table(
            {
                "primary_key": ["E-001", "E-002"],
                "amount_local": [100.0, 200.0],
            }
        )
        result = _filtered_metric_aggregate_tables(
            geo,
            ctx,
            "customers",
            "amount_local",
            "sum",
            resolved_filters=[["company_codes", ["CC-01"]]],
            event_line_id="gl_entries",
        )
        # Only E-001 passes (E-002's CC-01 edge is dead)
        assert result["metrics"]["CUST-A"] == pytest.approx(100.0)
        assert result["counts"]["CUST-A"] == 1


class TestDatasetCache:
    """Test _get_dataset caching."""

    def setup_method(self):
        _ds_cache.clear()

    def teardown_method(self):
        _ds_cache.clear()

    @patch("hypertopos.engine.subprocess_agg.lance")
    def test_caches_dataset(self, mock_lance):
        sentinel = object()
        mock_lance.dataset.return_value = sentinel

        ds1 = _get_dataset("/fake/path")
        ds2 = _get_dataset("/fake/path")

        assert ds1 is sentinel
        assert ds2 is sentinel
        mock_lance.dataset.assert_called_once_with("/fake/path")

    @patch("hypertopos.engine.subprocess_agg.lance")
    def test_different_paths_cached_separately(self, mock_lance):
        mock_lance.dataset.side_effect = lambda p: f"ds-{p}"

        ds_a = _get_dataset("/a")
        ds_b = _get_dataset("/b")

        assert ds_a == "ds-/a"
        assert ds_b == "ds-/b"
        assert mock_lance.dataset.call_count == 2


class TestTableCache:
    """Test _get_table LRU caching."""

    def setup_method(self):
        _ds_cache.clear()
        _table_cache.clear()

    def teardown_method(self):
        _ds_cache.clear()
        _table_cache.clear()

    @patch("hypertopos.engine.subprocess_agg.lance")
    def test_caches_table(self, mock_lance):
        table = pa.table({"x": [1, 2, 3]})
        mock_ds = MagicMock()
        mock_ds.to_table.return_value = table
        mock_lance.dataset.return_value = mock_ds

        t1 = _get_table("/fake", ["x"])
        t2 = _get_table("/fake", ["x"])

        assert t1 is table
        assert t2 is table
        mock_ds.to_table.assert_called_once()

    @patch("hypertopos.engine.subprocess_agg.lance")
    def test_cache_key_includes_columns(self, mock_lance):
        table_a = pa.table({"x": [1]})
        table_b = pa.table({"x": [1], "y": [2]})
        mock_ds = MagicMock()
        mock_ds.to_table.side_effect = [table_a, table_b]
        mock_lance.dataset.return_value = mock_ds

        t1 = _get_table("/fake", ["x"])
        t2 = _get_table("/fake", ["x", "y"])

        assert t1 is table_a
        assert t2 is table_b
        assert mock_ds.to_table.call_count == 2

    @patch("hypertopos.engine.subprocess_agg.lance")
    def test_column_order_irrelevant(self, mock_lance):
        """Columns ['a','b'] and ['b','a'] should hit the same cache entry."""
        table = pa.table({"a": [1], "b": [2]})
        mock_ds = MagicMock()
        mock_ds.to_table.return_value = table
        mock_lance.dataset.return_value = mock_ds

        t1 = _get_table("/fake", ["b", "a"])
        t2 = _get_table("/fake", ["a", "b"])

        assert t1 is t2
        mock_ds.to_table.assert_called_once()

    @patch("hypertopos.engine.subprocess_agg.lance")
    def test_lru_eviction(self, mock_lance):
        """Exceeding _TABLE_CACHE_MAX evicts the oldest entry."""
        n = _TABLE_CACHE_MAX + 1
        tables = [pa.table({"i": [i]}) for i in range(n)]
        mock_ds = MagicMock()
        mock_ds.to_table.side_effect = tables
        mock_lance.dataset.return_value = mock_ds

        for i in range(_TABLE_CACHE_MAX):
            _get_table(f"/{i}")
        assert len(_table_cache) == _TABLE_CACHE_MAX

        _get_table(f"/{_TABLE_CACHE_MAX}")
        assert len(_table_cache) == _TABLE_CACHE_MAX
        # First entry should have been evicted
        assert ("/0", None) not in _table_cache
        assert (f"/{_TABLE_CACHE_MAX}", None) in _table_cache

    @patch("hypertopos.engine.subprocess_agg.lance")
    def test_lru_access_refreshes(self, mock_lance):
        """Accessing an entry moves it to end, preventing eviction."""
        n = _TABLE_CACHE_MAX + 1
        tables = [pa.table({"i": [i]}) for i in range(n)]
        mock_ds = MagicMock()
        mock_ds.to_table.side_effect = tables
        mock_lance.dataset.return_value = mock_ds

        for i in range(_TABLE_CACHE_MAX):
            _get_table(f"/{i}")
        # Access /0 again to refresh it
        _get_table("/0")
        # Now insert one more — /1 should be evicted (oldest unreferenced)
        _get_table(f"/{_TABLE_CACHE_MAX}")

        assert ("/0", None) in _table_cache
        assert ("/1", None) not in _table_cache
        assert (f"/{_TABLE_CACHE_MAX}", None) in _table_cache


def _make_anomaly_table():
    """Build a geometry table with delta_norm values for anomaly tests."""
    return pa.table(
        {
            "primary_key": ["E-001", "E-002", "E-003", "E-004", "E-005"],
            "delta_norm": [0.5, 2.5, 1.8, 3.0, 0.1],
        }
    )


class TestFindAnomaliesTable:
    """Test _find_anomalies_table with in-memory Arrow tables."""

    def test_basic_filtering(self):
        table = _make_anomaly_table()
        result = _find_anomalies_table(table, threshold=1.0, top_n=10)
        assert result["keys"] == ["E-004", "E-002", "E-003"]
        assert result["delta_norms"] == pytest.approx([3.0, 2.5, 1.8])

    def test_top_n_limiting(self):
        table = _make_anomaly_table()
        result = _find_anomalies_table(table, threshold=1.0, top_n=2)
        assert len(result["keys"]) == 2
        assert result["keys"] == ["E-004", "E-002"]

    def test_empty_when_nothing_exceeds_threshold(self):
        table = _make_anomaly_table()
        result = _find_anomalies_table(table, threshold=10.0, top_n=10)
        assert result["keys"] == []
        assert result["delta_norms"] == []

    def test_sorted_descending(self):
        table = _make_anomaly_table()
        result = _find_anomalies_table(table, threshold=0.0, top_n=5)
        norms = result["delta_norms"]
        assert norms == sorted(norms, reverse=True)

    def test_boundary_exact_threshold_included(self):
        """Entity with delta_norm == threshold must be returned (>= not >)."""
        theta = 2.0
        table = pa.table(
            {
                "primary_key": ["below", "exact", "above"],
                "delta_norm": [1.5, theta, 2.5],
            }
        )
        result = _find_anomalies_table(table, threshold=theta, top_n=10)
        # "below" must not appear
        assert "below" not in result["keys"]
        # "exact" at the boundary must be included
        assert "exact" in result["keys"]
        # "above" must be included
        assert "above" in result["keys"]
        # Results are sorted descending
        assert result["delta_norms"] == pytest.approx([2.5, 2.0])


class TestCmdFindAnomalies:
    """Verify _cmd_find_anomalies is registered and callable."""

    def test_function_exists_and_is_callable(self):
        assert callable(_cmd_find_anomalies)

    def test_registered_in_commands_dict(self):
        assert _COMMANDS["find_anomalies"] is _cmd_find_anomalies


class TestCountAggregateTables:
    """Test _cmd_count_aggregate with in-memory Arrow tables."""

    def test_basic_count(self):
        geo = pa.table(
            {
                "primary_key": ["E-1", "E-2", "E-3", "E-4"],
                "edges": _edges_from_flat(
                    [["customers"], ["customers"], ["customers"], ["customers"]],
                    [["CUST-A"], ["CUST-A"], ["CUST-B"], ["CUST-B"]],
                ),
            }
        )
        from hypertopos.engine.subprocess_agg import _count_aggregate_tables

        result = _count_aggregate_tables(geo, "customers")
        assert result == {"CUST-A": 2, "CUST-B": 2}

    def test_dead_edges_excluded(self):
        edges_rows = [
            [
                {"line_id": "customers", "point_key": "CUST-A", "status": "alive", "direction": "in"},
            ],
            [
                {"line_id": "customers", "point_key": "CUST-A", "status": "dead", "direction": "in"},
            ],
        ]
        geo = pa.table(
            {
                "primary_key": ["E-1", "E-2"],
                "edges": pa.array(edges_rows, type=pa.list_(_EDGE_STRUCT)),
            }
        )
        from hypertopos.engine.subprocess_agg import _count_aggregate_tables

        result = _count_aggregate_tables(geo, "customers")
        assert result == {"CUST-A": 1}

    def test_no_matching_line(self):
        geo = pa.table(
            {
                "primary_key": ["E-1"],
                "edges": _edges_from_flat([["customers"]], [["CUST-A"]]),
            }
        )
        from hypertopos.engine.subprocess_agg import _count_aggregate_tables

        result = _count_aggregate_tables(geo, "nonexistent")
        assert result == {}

    def test_registered_in_commands_dict(self):
        assert "count_aggregate" in _COMMANDS
        assert callable(_COMMANDS["count_aggregate"])


class TestCmdMultiLevelMetricAggregate:
    """Test _cmd_metric_aggregate with group_by_line_2."""

    def test_multi_level_returns_composite_keys(self):
        """Two-line grouping produces \\x00-separated composite keys."""
        geo = pa.table(
            {
                "primary_key": ["E-1", "E-2", "E-3"],
                "edges": _edges_from_flat(
                    [
                        ["customers", "products"],
                        ["customers", "products"],
                        ["customers", "products"],
                    ],
                    [["CUST-A", "PROD-X"], ["CUST-A", "PROD-Y"], ["CUST-B", "PROD-X"]],
                ),
            }
        )
        ctx = pa.table(
            {
                "primary_key": ["E-1", "E-2", "E-3"],
                "amount": [10.0, 20.0, 30.0],
            }
        )
        with patch("hypertopos.engine.subprocess_agg._get_table") as mock_get:
            mock_get.side_effect = lambda path, cols=None: geo if "geometry" in path else ctx
            result = _cmd_metric_aggregate(
                geo_lance_path="/fake/geometry",
                ctx_lance_path="/fake/ctx",
                group_by_line="customers",
                metric_col="amount",
                agg_fn="sum",
                geo_columns=["primary_key", "edges"],
                group_by_line_2="products",
            )
        assert "CUST-A\x00PROD-X" in result["metrics"]
        assert "CUST-A\x00PROD-Y" in result["metrics"]
        assert "CUST-B\x00PROD-X" in result["metrics"]
        assert result["metrics"]["CUST-A\x00PROD-X"] == 10.0
        assert result["metrics"]["CUST-A\x00PROD-Y"] == 20.0
        assert result["metrics"]["CUST-B\x00PROD-X"] == 30.0

    def test_without_gbl2_unchanged(self):
        """Single-GBL path still works."""
        geo = pa.table(
            {
                "primary_key": ["E-1", "E-2"],
                "edges": _edges_from_flat(
                    [["customers"], ["customers"]],
                    [["CUST-A"], ["CUST-A"]],
                ),
            }
        )
        ctx = pa.table(
            {
                "primary_key": ["E-1", "E-2"],
                "amount": [5.0, 15.0],
            }
        )
        with patch("hypertopos.engine.subprocess_agg._get_table") as mock_get:
            mock_get.side_effect = lambda path, cols=None: geo if "geometry" in path else ctx
            result = _cmd_metric_aggregate(
                geo_lance_path="/fake/geometry",
                ctx_lance_path="/fake/ctx",
                group_by_line="customers",
                metric_col="amount",
                agg_fn="sum",
                geo_columns=["primary_key", "edges"],
            )
        assert "CUST-A" in result["metrics"]
        assert result["metrics"]["CUST-A"] == 20.0
