# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for DataFusion-backed aggregate engine."""

from __future__ import annotations

import pyarrow as pa
import pytest
from hypertopos.engine.datafusion_agg import (
    DATAFUSION_THRESHOLD,
    aggregate_count,
    aggregate_filtered_gbp_count,
    aggregate_filtered_metric_with_count,
    aggregate_filtered_pivot_metric_with_count,
    aggregate_metric,
    aggregate_metric_with_count,
    aggregate_pivot_count,
    is_available,
)

from tests.conftest import make_edges_column


@pytest.fixture
def geo_table():
    return pa.table(
        {
            "primary_key": ["TX-001", "TX-002", "TX-003", "TX-004", "TX-005"],
            "edges": make_edges_column(
                edge_line_ids=[
                    ["customers", "categories"],
                    ["customers", "categories"],
                    ["customers"],
                    ["customers", "categories"],
                    ["customers"],
                ],
                edge_point_keys=[
                    ["CUST-A", "CAT-1"],
                    ["CUST-B", "CAT-2"],
                    ["CUST-A"],
                    ["CUST-A", "CAT-1"],
                    ["CUST-B"],
                ],
            ),
        }
    )


@pytest.fixture
def points_table():
    return pa.table(
        {
            "primary_key": ["TX-001", "TX-002", "TX-003", "TX-004", "TX-005"],
            "price": [0.10, 0.50, 0.05, 0.30, 0.20],
        }
    )


class TestIsAvailable:
    def test_is_available_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)
        # datafusion is installed in this env, so should be True
        assert result is True

    def test_is_available_false_when_no_datafusion(self, monkeypatch):
        import hypertopos.engine.datafusion_agg as mod

        monkeypatch.setattr(mod, "_datafusion", None)
        assert is_available() is False


class TestAggregateCount:
    def test_aggregate_count(self, geo_table):
        result = aggregate_count(geo_table, "customers")
        assert result == {"CUST-A": 3, "CUST-B": 2}

    def test_aggregate_count_excludes_dead_edges(self):
        geo = pa.table(
            {
                "primary_key": ["TX-001", "TX-002", "TX-003", "TX-004", "TX-005"],
                "edges": make_edges_column(
                    edge_line_ids=[
                        ["customers", "categories"],
                        ["customers", "categories"],
                        ["customers"],
                        ["customers", "categories"],
                        ["customers"],
                    ],
                    edge_point_keys=[
                        ["CUST-A", "CAT-1"],
                        ["CUST-B", "CAT-2"],
                        ["CUST-A"],
                        ["CUST-A", "CAT-1"],
                        ["CUST-B"],
                    ],
                    edge_alive_mask=[
                        [True, True],
                        [True, True],
                        [False],  # TX-003 customer edge is dead
                        [True, True],
                        [True],
                    ],
                ),
            }
        )
        result = aggregate_count(geo, "customers")
        # CUST-A: TX-001 + TX-004 = 2 (TX-003 dead)
        # CUST-B: TX-002 + TX-005 = 2
        assert result == {"CUST-A": 2, "CUST-B": 2}

    def test_aggregate_count_returns_empty_when_line_not_present(self, geo_table):
        result = aggregate_count(geo_table, "merchants")
        assert result == {}


class TestAggregateMetric:
    def test_aggregate_max(self, geo_table, points_table):
        result = aggregate_metric(geo_table, "customers", points_table, "price", "max")
        # CUST-A: max(0.10, 0.05, 0.30) = 0.30
        # CUST-B: max(0.50, 0.20) = 0.50
        assert result["CUST-A"] == pytest.approx(0.30)
        assert result["CUST-B"] == pytest.approx(0.50)

    def test_aggregate_sum(self, geo_table, points_table):
        result = aggregate_metric(geo_table, "customers", points_table, "price", "sum")
        # CUST-A: 0.10 + 0.05 + 0.30 = 0.45
        # CUST-B: 0.50 + 0.20 = 0.70
        assert result["CUST-A"] == pytest.approx(0.45)
        assert result["CUST-B"] == pytest.approx(0.70)

    def test_aggregate_min(self, geo_table, points_table):
        result = aggregate_metric(geo_table, "customers", points_table, "price", "min")
        # CUST-A: min(0.10, 0.05, 0.30) = 0.05
        # CUST-B: min(0.50, 0.20) = 0.20
        assert result["CUST-A"] == pytest.approx(0.05)
        assert result["CUST-B"] == pytest.approx(0.20)

    def test_aggregate_avg(self, geo_table, points_table):
        result = aggregate_metric(geo_table, "customers", points_table, "price", "avg")
        # CUST-A: avg(0.10, 0.05, 0.30) = 0.15
        # CUST-B: avg(0.50, 0.20) = 0.35
        assert result["CUST-A"] == pytest.approx(0.15)
        assert result["CUST-B"] == pytest.approx(0.35)

    def test_aggregate_metric_invalid_fn(self, geo_table, points_table):
        with pytest.raises(ValueError, match="Unsupported"):
            aggregate_metric(geo_table, "customers", points_table, "price", "median")

    def test_aggregate_metric_returns_empty_when_line_not_present(self, geo_table, points_table):
        result = aggregate_metric(geo_table, "merchants", points_table, "price", "sum")
        assert result == {}


class TestAggregateMetricWithCount:
    def test_combined_max(self, geo_table, points_table):
        metrics, counts = aggregate_metric_with_count(
            geo_table, "customers", points_table, "price", "max"
        )
        assert metrics["CUST-A"] == pytest.approx(0.30)
        assert metrics["CUST-B"] == pytest.approx(0.50)
        assert counts == {"CUST-A": 3, "CUST-B": 2}

    def test_combined_sum(self, geo_table, points_table):
        metrics, counts = aggregate_metric_with_count(
            geo_table, "customers", points_table, "price", "sum"
        )
        assert metrics["CUST-A"] == pytest.approx(0.45)
        assert metrics["CUST-B"] == pytest.approx(0.70)
        assert counts == {"CUST-A": 3, "CUST-B": 2}

    def test_combined_returns_empty_when_line_not_present(self, geo_table, points_table):
        metrics, counts = aggregate_metric_with_count(
            geo_table, "merchants", points_table, "price", "sum"
        )
        assert metrics == {}
        assert counts == {}

    def test_combined_invalid_fn(self, geo_table, points_table):
        with pytest.raises(ValueError, match="Unsupported"):
            aggregate_metric_with_count(geo_table, "customers", points_table, "price", "median")


class TestAggregateFilteredMetricWithCount:
    def test_single_filter_sum(self, geo_table, points_table):
        resolved_filters = [("categories", {"CAT-1"})]
        metrics, counts = aggregate_filtered_metric_with_count(
            geo_table, "customers", points_table, "price", "sum", resolved_filters
        )
        # TX-001 (CUST-A, CAT-1, 0.10) pass  TX-004 (CUST-A, CAT-1, 0.30) pass
        # TX-002 CAT-2 fail  TX-003 no cat fail  TX-005 no cat fail
        assert metrics == pytest.approx({"CUST-A": 0.40})
        assert counts == {"CUST-A": 2}

    def test_multi_key_filter(self, geo_table, points_table):
        resolved_filters = [("categories", {"CAT-1", "CAT-2"})]
        metrics, counts = aggregate_filtered_metric_with_count(
            geo_table, "customers", points_table, "price", "sum", resolved_filters
        )
        # TX-001 (CUST-A, CAT-1, 0.10) pass  TX-002 (CUST-B, CAT-2, 0.50) pass
        # TX-004 (CUST-A, CAT-1, 0.30) pass
        assert metrics == pytest.approx({"CUST-A": 0.40, "CUST-B": 0.50})
        assert counts == {"CUST-A": 2, "CUST-B": 1}

    def test_no_filters_equals_unfiltered(self, geo_table, points_table):
        metrics, counts = aggregate_filtered_metric_with_count(
            geo_table, "customers", points_table, "price", "sum", []
        )
        assert metrics == pytest.approx({"CUST-A": 0.45, "CUST-B": 0.70})

    def test_no_match_returns_empty(self, geo_table, points_table):
        resolved_filters = [("categories", {"CAT-NONEXISTENT"})]
        metrics, counts = aggregate_filtered_metric_with_count(
            geo_table, "customers", points_table, "price", "sum", resolved_filters
        )
        assert metrics == {}
        assert counts == {}

    def test_and_semantics_two_filters(self, geo_table, points_table):
        # TX-001: CUST-A, CAT-1 — both filters match pass
        # TX-002: CUST-B, CAT-2 — CAT-2 not in CAT-1 filter fail
        resolved_filters = [("categories", {"CAT-1"}), ("customers", {"CUST-A"})]
        metrics, counts = aggregate_filtered_metric_with_count(
            geo_table, "customers", points_table, "price", "sum", resolved_filters
        )
        # Only CUST-A with CAT-1: TX-001 (0.10) + TX-004 (0.30) = 0.40
        assert metrics == pytest.approx({"CUST-A": 0.40})
        assert counts == {"CUST-A": 2}

    def test_invalid_agg_fn(self, geo_table, points_table):
        with pytest.raises(ValueError, match="Unsupported"):
            aggregate_filtered_metric_with_count(
                geo_table, "customers", points_table, "price", "median", []
            )


@pytest.fixture
def points_with_pivot():
    return pa.table(
        {
            "primary_key": ["TX-001", "TX-002", "TX-003", "TX-004", "TX-005"],
            "price": [0.10, 0.50, 0.05, 0.30, 0.20],
            "t_dat": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-01"],
        }
    )


class TestAggregateFilteredPivotMetricWithCount:
    def test_pivot_with_filter(self, geo_table, points_with_pivot):
        resolved_filters = [("categories", {"CAT-1"})]
        results = aggregate_filtered_pivot_metric_with_count(
            geo_table, "customers", points_with_pivot, "price", "sum", "t_dat", resolved_filters
        )
        # TX-001: CUST-A, CAT-1, 2023-01-01, 0.10 pass
        # TX-004: CUST-A, CAT-1, 2023-01-02, 0.30 pass
        by_key = {(r["group_key"], r["pivot_val"]): r for r in results}
        assert len(results) == 2
        assert by_key[("CUST-A", "2023-01-01")]["metric"] == pytest.approx(0.10)
        assert by_key[("CUST-A", "2023-01-02")]["metric"] == pytest.approx(0.30)

    def test_no_filters_all_transactions(self, geo_table, points_with_pivot):
        results = aggregate_filtered_pivot_metric_with_count(
            geo_table, "customers", points_with_pivot, "price", "sum", "t_dat", []
        )
        # TX-001: CUST-A, 2023-01-01, 0.10  TX-003: CUST-A, 2023-01-02, 0.05
        # TX-004: CUST-A, 2023-01-02, 0.30  TX-002: CUST-B, 2023-01-01, 0.50
        # TX-005: CUST-B, 2023-01-01, 0.20
        by_key = {(r["group_key"], r["pivot_val"]): r for r in results}
        assert by_key[("CUST-A", "2023-01-01")]["metric"] == pytest.approx(0.10)
        assert by_key[("CUST-A", "2023-01-02")]["metric"] == pytest.approx(0.35)
        assert by_key[("CUST-B", "2023-01-01")]["metric"] == pytest.approx(0.70)

    def test_no_match_returns_empty(self, geo_table, points_with_pivot):
        resolved_filters = [("categories", {"CAT-NONEXISTENT"})]
        results = aggregate_filtered_pivot_metric_with_count(
            geo_table, "customers", points_with_pivot, "price", "sum", "t_dat", resolved_filters
        )
        assert results == []

    def test_invalid_agg_fn(self, geo_table, points_with_pivot):
        with pytest.raises(ValueError, match="Unsupported"):
            aggregate_filtered_pivot_metric_with_count(
                geo_table, "customers", points_with_pivot, "price", "median", "t_dat", []
            )


class TestAggregatePivotCount:
    """Tests for COUNT(*) GROUP BY group_key x pivot_field (no edge filters)."""

    def test_basic_grouped_counts(self):
        """aggregate_pivot_count returns correct grouped counts."""
        if not is_available():
            pytest.skip("datafusion not installed")

        geo_table = pa.table(
            {
                "primary_key": ["A", "B", "C"],
                "edges": make_edges_column(
                    edge_line_ids=[["customers"], ["customers"], ["customers"]],
                    edge_point_keys=[["C1"], ["C1"], ["C2"]],
                ),
            }
        )
        event_table = pa.table(
            {
                "primary_key": ["A", "B", "C"],
                "season": ["summer", "winter", "summer"],
            }
        )

        results = aggregate_pivot_count(geo_table, "customers", event_table, "season")

        result_dict = {(r["group_key"], r["pivot_val"]): r["count"] for r in results}
        assert result_dict[("C1", "summer")] == 1
        assert result_dict[("C1", "winter")] == 1
        assert result_dict[("C2", "summer")] == 1

    def test_returns_empty_when_line_not_present(self):
        """Returns empty list when group_by_line has no edges."""
        if not is_available():
            pytest.skip("datafusion not installed")

        geo_table = pa.table(
            {
                "primary_key": ["A", "B"],
                "edges": make_edges_column(
                    edge_line_ids=[["customers"], ["customers"]],
                    edge_point_keys=[["C1"], ["C2"]],
                ),
            }
        )
        event_table = pa.table(
            {
                "primary_key": ["A", "B"],
                "season": ["summer", "winter"],
            }
        )

        results = aggregate_pivot_count(geo_table, "nonexistent_line", event_table, "season")
        assert results == []

    def test_excludes_dead_edges(self):
        """Dead edges (status=dead) are excluded from counts."""
        if not is_available():
            pytest.skip("datafusion not installed")

        geo_table = pa.table(
            {
                "primary_key": ["A", "B", "C"],
                "edges": make_edges_column(
                    edge_line_ids=[["customers"], ["customers"], ["customers"]],
                    edge_point_keys=[["C1"], ["C1"], ["C2"]],
                    edge_alive_mask=[[True], [False], [True]],
                ),
            }
        )
        event_table = pa.table(
            {
                "primary_key": ["A", "B", "C"],
                "season": ["summer", "winter", "summer"],
            }
        )

        results = aggregate_pivot_count(geo_table, "customers", event_table, "season")
        result_dict = {(r["group_key"], r["pivot_val"]): r["count"] for r in results}
        # B is dead -> only A and C remain
        assert result_dict.get(("C1", "summer")) == 1
        assert ("C1", "winter") not in result_dict
        assert result_dict.get(("C2", "summer")) == 1

    def test_raises_when_datafusion_not_installed(self, monkeypatch):
        """RuntimeError when datafusion package is not installed."""
        import hypertopos.engine.datafusion_agg as mod

        monkeypatch.setattr(mod, "_datafusion", None)
        geo_table = pa.table(
            {
                "primary_key": ["A"],
                "edges": make_edges_column(
                    edge_line_ids=[["customers"]],
                    edge_point_keys=[["C1"]],
                ),
            }
        )
        event_table = pa.table(
            {
                "primary_key": ["A"],
                "season": ["summer"],
            }
        )
        with pytest.raises(RuntimeError, match="datafusion is not installed"):
            aggregate_pivot_count(geo_table, "customers", event_table, "season")

    def test_null_pivot_val_excluded(self):
        """NULL pivot_field values are excluded -- matches Arrow fallback behaviour."""
        if not is_available():
            pytest.skip("datafusion not installed")

        geo_table = pa.table(
            {
                "primary_key": ["A", "B", "C"],
                "edges": make_edges_column(
                    edge_line_ids=[["customers"], ["customers"], ["customers"]],
                    edge_point_keys=[["C1"], ["C1"], ["C2"]],
                ),
            }
        )
        event_table = pa.table(
            {
                "primary_key": ["A", "B", "C"],
                "season": pa.array(["summer", None, "summer"], type=pa.string()),
            }
        )

        results = aggregate_pivot_count(geo_table, "customers", event_table, "season")
        result_dict = {(r["group_key"], r["pivot_val"]): r["count"] for r in results}

        assert ("C1", "summer") in result_dict
        assert all(r["pivot_val"] is not None for r in results)
        assert ("C1", None) not in result_dict

    def test_matches_arrow_fallback_with_nulls(self):
        """DataFusion and Arrow fallback return identical counts when nulls present."""
        if not is_available():
            pytest.skip("datafusion not installed")

        from hypertopos.engine.subprocess_agg import _pivot_aggregate_tables

        geo_table = pa.table(
            {
                "primary_key": ["A", "B", "C", "D"],
                "edges": make_edges_column(
                    edge_line_ids=[["customers"], ["customers"], ["customers"], ["customers"]],
                    edge_point_keys=[["C1"], ["C1"], ["C2"], ["C2"]],
                ),
            }
        )
        event_table = pa.table(
            {
                "primary_key": ["A", "B", "C", "D"],
                "season": pa.array(["summer", None, "summer", "winter"], type=pa.string()),
            }
        )

        df_results = aggregate_pivot_count(geo_table, "customers", event_table, "season")
        arrow_results = _pivot_aggregate_tables(geo_table, event_table, "customers", "season")

        df_dict = {(r["group_key"], r["pivot_val"]): r["count"] for r in df_results}
        arrow_dict = {(r["group_key"], r["pivot_val"]): r["count"] for r in arrow_results}
        assert df_dict == arrow_dict


class TestAggregateFilteredGbpCount:
    """Tests for COUNT GROUP BY group_key x prop_val with edge filters."""

    @pytest.fixture
    def geo_gbp(self):
        """4 polygons: merchants x customers, filtered by fx_rates."""
        return pa.table(
            {
                "primary_key": ["TX-001", "TX-002", "TX-003", "TX-004"],
                "edges": make_edges_column(
                    edge_line_ids=[
                        ["merchants", "customers", "fx_rates"],
                        ["merchants", "customers", "fx_rates"],
                        ["merchants", "customers", "fx_rates"],
                        ["merchants", "customers", "fx_rates"],
                    ],
                    edge_point_keys=[
                        ["M-A", "CUST-1", "EUR"],
                        ["M-A", "CUST-2", "USD"],
                        ["M-B", "CUST-1", "EUR"],
                        ["M-B", "CUST-3", "EUR"],
                    ],
                ),
            }
        )

    @pytest.fixture
    def prop_tbl(self):
        """customers with loyalty_tier property."""
        return pa.table(
            {
                "primary_key": ["CUST-1", "CUST-2", "CUST-3"],
                "loyalty_tier": ["gold", "silver", "gold"],
            }
        )

    def test_basic_count(self, geo_gbp, prop_tbl):
        """COUNT GROUP BY merchants x loyalty_tier filtered to EUR."""
        pytest.importorskip("datafusion")
        resolved_filters = [("fx_rates", {"EUR"})]
        result = aggregate_filtered_gbp_count(
            geo_gbp,
            group_by_line="merchants",
            prop_table=prop_tbl,
            prop_name="loyalty_tier",
            prop_line_id="customers",
            resolved_filters=resolved_filters,
        )
        # TX-001: M-A, EUR, CUST-1 -> gold  pass
        # TX-002: M-A, USD -- filtered out   fail
        # TX-003: M-B, EUR, CUST-1 -> gold  pass
        # TX-004: M-B, EUR, CUST-3 -> gold  pass
        assert result == {("M-A", "gold"): 1, ("M-B", "gold"): 2}

    def test_multi_key_filter(self, geo_gbp, prop_tbl):
        """Filter with multiple keys in key_set -- OR semantics within one filter."""
        pytest.importorskip("datafusion")
        resolved_filters = [("fx_rates", {"EUR", "USD"})]
        result = aggregate_filtered_gbp_count(
            geo_gbp,
            group_by_line="merchants",
            prop_table=prop_tbl,
            prop_name="loyalty_tier",
            prop_line_id="customers",
            resolved_filters=resolved_filters,
        )
        # All 4 transactions pass (EUR or USD)
        # TX-001: M-A, gold  TX-002: M-A, silver  TX-003: M-B, gold  TX-004: M-B, gold
        assert result == {
            ("M-A", "gold"): 1,
            ("M-A", "silver"): 1,
            ("M-B", "gold"): 2,
        }

    def test_empty_filters_raises(self, geo_gbp, prop_tbl):
        """Empty resolved_filters must raise ValueError."""
        pytest.importorskip("datafusion")
        with pytest.raises(ValueError, match="resolved_filters must not be empty"):
            aggregate_filtered_gbp_count(
                geo_gbp,
                group_by_line="merchants",
                prop_table=prop_tbl,
                prop_name="loyalty_tier",
                prop_line_id="customers",
                resolved_filters=[],
            )

    def test_no_match_returns_empty(self, geo_gbp, prop_tbl):
        """Filter that matches no polygon returns empty dict."""
        pytest.importorskip("datafusion")
        resolved_filters = [("fx_rates", {"GBP"})]
        result = aggregate_filtered_gbp_count(
            geo_gbp,
            group_by_line="merchants",
            prop_table=prop_tbl,
            prop_name="loyalty_tier",
            prop_line_id="customers",
            resolved_filters=resolved_filters,
        )
        assert result == {}

    def test_and_semantics_two_filters(self, geo_gbp, prop_tbl):
        """Two filters: AND semantics -- polygon must satisfy both."""
        pytest.importorskip("datafusion")
        resolved_filters = [
            ("fx_rates", {"EUR"}),
            ("customers", {"CUST-1"}),
        ]
        result = aggregate_filtered_gbp_count(
            geo_gbp,
            group_by_line="merchants",
            prop_table=prop_tbl,
            prop_name="loyalty_tier",
            prop_line_id="customers",
            resolved_filters=resolved_filters,
        )
        # TX-001: EUR pass, CUST-1 pass -> M-A, gold
        # TX-003: EUR pass, CUST-1 pass -> M-B, gold
        assert result == {("M-A", "gold"): 1, ("M-B", "gold"): 1}

    def test_no_alive_mask(self, prop_tbl):
        """Works when all edges are alive (no dead edges)."""
        pytest.importorskip("datafusion")
        geo = pa.table(
            {
                "primary_key": ["TX-001", "TX-002"],
                "edges": make_edges_column(
                    edge_line_ids=[
                        ["merchants", "customers", "fx_rates"],
                        ["merchants", "customers", "fx_rates"],
                    ],
                    edge_point_keys=[
                        ["M-A", "CUST-1", "EUR"],
                        ["M-A", "CUST-2", "USD"],
                    ],
                ),
            }
        )
        resolved_filters = [("fx_rates", {"EUR"})]
        result = aggregate_filtered_gbp_count(
            geo,
            group_by_line="merchants",
            prop_table=prop_tbl,
            prop_name="loyalty_tier",
            prop_line_id="customers",
            resolved_filters=resolved_filters,
        )
        assert result == {("M-A", "gold"): 1}

    def test_prop_line_same_as_group_by_line(self, geo_gbp, prop_tbl):
        """prop_line_id == group_by_line: join on same edge -- counts customers by own tier."""
        pytest.importorskip("datafusion")
        # Group by customers, cross-tab by customers:loyalty_tier (same line as group_by_line).
        # prop_tbl contains CUST-1=gold, CUST-2=silver, CUST-3=gold.
        resolved_filters = [("fx_rates", {"EUR"})]
        result = aggregate_filtered_gbp_count(
            geo_gbp,
            group_by_line="customers",
            prop_table=prop_tbl,
            prop_name="loyalty_tier",
            prop_line_id="customers",
            resolved_filters=resolved_filters,
        )
        # TX-001: CUST-1, EUR -> gold  TX-003: CUST-1, EUR -> gold  TX-004: CUST-3, EUR -> gold
        assert result == {("CUST-1", "gold"): 2, ("CUST-3", "gold"): 1}

    def test_datafusion_not_installed_raises(self, geo_gbp, prop_tbl, monkeypatch):
        """RuntimeError when datafusion package is not installed."""
        import hypertopos.engine.datafusion_agg as mod

        monkeypatch.setattr(mod, "_datafusion", None)
        with pytest.raises(RuntimeError, match="datafusion is not installed"):
            aggregate_filtered_gbp_count(
                geo_gbp,
                group_by_line="merchants",
                prop_table=prop_tbl,
                prop_name="loyalty_tier",
                prop_line_id="customers",
                resolved_filters=[("fx_rates", {"EUR"})],
            )


class TestThreshold:
    def test_threshold_value(self):
        assert DATAFUSION_THRESHOLD == 500_000


class TestIsPercentile:
    """Tests for _is_percentile and _percentile_fraction helpers."""

    def test_is_percentile_median(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("median") is True

    def test_is_percentile_pct50(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("pct50") is True

    def test_is_percentile_pct0(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("pct0") is True

    def test_is_percentile_pct100(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("pct100") is True

    def test_is_percentile_pct75(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("pct75") is True

    def test_is_percentile_sum_is_false(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("sum") is False

    def test_is_percentile_avg_is_false(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("avg") is False

    def test_is_percentile_pct_out_of_range_is_false(self):
        from hypertopos.engine.datafusion_agg import _is_percentile
        assert _is_percentile("pct101") is False

    def test_percentile_fraction_median(self):
        from hypertopos.engine.datafusion_agg import _percentile_fraction
        assert _percentile_fraction("median") == pytest.approx(0.5)

    def test_percentile_fraction_pct75(self):
        from hypertopos.engine.datafusion_agg import _percentile_fraction
        assert _percentile_fraction("pct75") == pytest.approx(0.75)

    def test_percentile_fraction_pct0(self):
        from hypertopos.engine.datafusion_agg import _percentile_fraction
        assert _percentile_fraction("pct0") == pytest.approx(0.0)

    def test_percentile_fraction_pct100(self):
        from hypertopos.engine.datafusion_agg import _percentile_fraction
        assert _percentile_fraction("pct100") == pytest.approx(1.0)


class TestAggregatePercentile:
    """Tests for aggregate_percentile (DataFusion APPROX_PERCENTILE_CONT)."""

    group_by_line = "customers"
    metric_col = "price"

    @pytest.fixture
    def geo_table(self):
        # 5 transactions: CUST-A has 3 (prices 0.10, 0.30, 0.50), CUST-B has 2 (0.20, 0.40)
        return pa.table(
            {
                "primary_key": ["TX-001", "TX-002", "TX-003", "TX-004", "TX-005"],
                "edges": make_edges_column(
                    edge_line_ids=[
                        ["customers"],
                        ["customers"],
                        ["customers"],
                        ["customers"],
                        ["customers"],
                    ],
                    edge_point_keys=[
                        ["CUST-A"],
                        ["CUST-A"],
                        ["CUST-A"],
                        ["CUST-B"],
                        ["CUST-B"],
                    ],
                ),
            }
        )

    @pytest.fixture
    def points_table(self):
        return pa.table(
            {
                "primary_key": ["TX-001", "TX-002", "TX-003", "TX-004", "TX-005"],
                "price": [0.10, 0.30, 0.50, 0.20, 0.40],
            }
        )

    def test_aggregate_percentile_median(self, geo_table, points_table):
        """DataFusion: APPROX_PERCENTILE_CONT for median returns float per group."""
        from hypertopos.engine.datafusion_agg import aggregate_percentile, is_available
        if not is_available():
            pytest.skip("datafusion not installed")
        vals, cnts = aggregate_percentile(
            geo_table, self.group_by_line, points_table,
            self.metric_col, 0.5,
        )
        assert len(vals) > 0
        for k, v in vals.items():
            assert isinstance(v, float)
            assert cnts[k] > 0
        # CUST-A has 3 values (0.10, 0.30, 0.50), median should be close to 0.30
        assert vals["CUST-A"] == pytest.approx(0.30, abs=0.05)
        assert cnts["CUST-A"] == 3
        assert cnts["CUST-B"] == 2

    def test_aggregate_percentile_p75(self, geo_table, points_table):
        """DataFusion: pct75 >= median for each group (with approximate tolerance)."""
        from hypertopos.engine.datafusion_agg import aggregate_percentile, is_available
        if not is_available():
            pytest.skip("datafusion not installed")
        medians, _ = aggregate_percentile(
            geo_table, self.group_by_line, points_table,
            self.metric_col, 0.5,
        )
        p75s, _ = aggregate_percentile(
            geo_table, self.group_by_line, points_table,
            self.metric_col, 0.75,
        )
        for k in medians:
            if k in p75s:
                assert p75s[k] >= medians[k] - 0.01  # approximate tolerance

    def test_aggregate_percentile_no_datafusion_raises(self, geo_table, points_table, monkeypatch):
        """RuntimeError when datafusion is not installed."""
        import hypertopos.engine.datafusion_agg as mod
        monkeypatch.setattr(mod, "_datafusion", None)
        from hypertopos.engine.datafusion_agg import aggregate_percentile
        with pytest.raises(RuntimeError, match="datafusion is not installed"):
            aggregate_percentile(
                geo_table, self.group_by_line, points_table, self.metric_col, 0.5,
            )

    def test_aggregate_percentile_empty_result(self, geo_table, points_table):
        """Returns empty dicts when group_by_line has no matches."""
        from hypertopos.engine.datafusion_agg import aggregate_percentile, is_available
        if not is_available():
            pytest.skip("datafusion not installed")
        vals, cnts = aggregate_percentile(
            geo_table, "merchants", points_table, self.metric_col, 0.5,
        )
        assert vals == {}
        assert cnts == {}
