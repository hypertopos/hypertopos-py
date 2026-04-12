# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Direct unit tests for the lance_sql_agg helpers.

These tests bypass the full aggregate engine wrapper and exercise each
helper against a tmp_path Lance dataset built by hand. The integration
side of things is covered by test_aggregation_engine.py and the MCP
tools tests; this file is the contract layer for the helpers themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from hypertopos.engine.lance_sql_agg import (
    _escape_sql_string,
    _validate_sql_identifier,
    aggregate_count,
    aggregate_filtered_metric,
    aggregate_metric,
    aggregate_percentile,
    aggregate_pivot,
    aggregate_property,
    find_anomalies,
)


@dataclass
class _Rel:
    line_id: str


@dataclass
class _Pat:
    relations: list


def _write_geo(path: Path, table: pa.Table) -> str:
    p = str(path / "geo.lance")
    lance.write_dataset(table, p, data_storage_version="2.2")
    return p


def _write_ctx(path: Path, table: pa.Table) -> str:
    p = str(path / "ctx.lance")
    lance.write_dataset(table, p, data_storage_version="2.2")
    return p


def _accounts_pattern() -> _Pat:
    return _Pat(relations=[
        _Rel("accounts"),  # entity_keys[1] = sender
        _Rel("accounts"),  # entity_keys[2] = receiver
        _Rel("currencies"),
    ])


def _basic_geo_table() -> pa.Table:
    return pa.table({
        "primary_key": ["t1", "t2", "t3", "t4"],
        "entity_keys": pa.array(
            [
                ["A", "B", "USD"],
                ["B", "C", "EUR"],
                ["A", "C", "USD"],
                ["A", "A", "USD"],
            ],
            type=pa.list_(pa.string()),
        ),
    })


def _basic_ctx_table() -> pa.Table:
    return pa.table({
        "primary_key": ["t1", "t2", "t3", "t4"],
        "amount": [10.0, 20.0, 30.0, 40.0],
    })


# ── _escape_sql_string ────────────────────────────────────────────────


class TestLinePositions:
    def test_empty_pattern_returns_empty(self):
        pat = _Pat(relations=[])
        from hypertopos.engine.lance_sql_agg import _line_positions
        assert _line_positions(pat, "anything") == []

    def test_unicode_key_escaping(self):
        assert _escape_sql_string("Müller") == "Müller"
        assert _escape_sql_string("日本語") == "日本語"


class TestEscapeSqlString:
    def test_doubles_single_quote(self):
        assert _escape_sql_string("O'Brien") == "O''Brien"

    def test_passes_alnum_and_punctuation(self):
        assert _escape_sql_string("ACCT-001.AB:42") == "ACCT-001.AB:42"

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="backslash"):
            _escape_sql_string("a\\b")

    def test_rejects_nul(self):
        with pytest.raises(ValueError, match="control character"):
            _escape_sql_string("a\x00b")

    def test_rejects_other_control_chars(self):
        for ch in ("\x01", "\n", "\t", "\x7f"):
            with pytest.raises(ValueError, match="control character"):
                _escape_sql_string(f"a{ch}b")

    def test_rejects_non_str(self):
        with pytest.raises(TypeError):
            _escape_sql_string(42)  # type: ignore[arg-type]


# ── _validate_sql_identifier ──────────────────────────────────────────


class TestValidateSqlIdentifier:
    def test_passes_simple_alpha(self):
        assert _validate_sql_identifier("amount") == "amount"

    def test_passes_underscore_start(self):
        assert _validate_sql_identifier("_internal") == "_internal"

    def test_passes_alnum(self):
        assert _validate_sql_identifier("col_42_x") == "col_42_x"

    def test_rejects_digit_start(self):
        with pytest.raises(ValueError, match="not a safe SQL identifier"):
            _validate_sql_identifier("42col")

    def test_rejects_dash(self):
        with pytest.raises(ValueError, match="not a safe SQL identifier"):
            _validate_sql_identifier("col-name")

    def test_rejects_space(self):
        with pytest.raises(ValueError, match="not a safe SQL identifier"):
            _validate_sql_identifier("col name")

    def test_rejects_injection_attempt(self):
        with pytest.raises(ValueError, match="not a safe SQL identifier"):
            _validate_sql_identifier("amount; DROP TABLE x; --")

    def test_rejects_quote(self):
        with pytest.raises(ValueError, match="not a safe SQL identifier"):
            _validate_sql_identifier("a'b")

    def test_rejects_non_str(self):
        with pytest.raises(TypeError):
            _validate_sql_identifier(42)  # type: ignore[arg-type]


# ── aggregate_count ────────────────────────────────────────────────────


class TestAggregateCount:
    def test_basic_positional_union(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        result = aggregate_count(geo_path, _accounts_pattern(), "accounts")
        # A: t1[1], t3[1], t4[1], t4[2] = 4
        # B: t1[2], t2[1] = 2
        # C: t2[2], t3[2] = 2
        assert result == {"A": 4, "B": 2, "C": 2}

    def test_no_matching_relation_returns_empty(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        result = aggregate_count(geo_path, _accounts_pattern(), "merchants")
        assert result == {}

    def test_currencies_path_one_position(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        result = aggregate_count(geo_path, _accounts_pattern(), "currencies")
        assert result == {"USD": 3, "EUR": 1}


# ── aggregate_metric ───────────────────────────────────────────────────


class TestAggregateMetric:
    def test_sum_join_with_ctx(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        metrics, counts = aggregate_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "sum",
        )
        # A: 10 + 30 + 40 + 40 = 120 (t4 contributes twice for A,A)
        # B: 10 + 20 = 30
        # C: 20 + 30 = 50
        assert metrics == {"A": 120.0, "B": 30.0, "C": 50.0}
        assert counts == {"A": 4, "B": 2, "C": 2}

    def test_avg(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        metrics, _ = aggregate_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "avg",
        )
        assert metrics["A"] == pytest.approx(120.0 / 4)
        assert metrics["B"] == pytest.approx(15.0)

    def test_min_max(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        m_min, _ = aggregate_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "min",
        )
        m_max, _ = aggregate_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "max",
        )
        assert m_min["A"] == 10.0
        assert m_max["A"] == 40.0

    def test_invalid_agg_fn_raises(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        with pytest.raises(ValueError, match="agg_fn must be one of"):
            aggregate_metric(
                geo_path, ctx_path, _accounts_pattern(),
                "accounts", "amount", "stddev",
            )

    def test_no_matching_relation(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        metrics, counts = aggregate_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "merchants", "amount", "sum",
        )
        assert metrics == {} and counts == {}


# ── aggregate_filtered_metric ──────────────────────────────────────────


class TestAggregateFilteredMetric:
    def test_filter_to_one_account(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        # Only count edges of polygons whose accounts include A
        metrics, counts = aggregate_filtered_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "sum",
            resolved_filters=[("accounts", {"A"})],
            event_line_id="transactions",
        )
        # Polygons containing A: t1, t3, t4 — t4 has A,A
        # A: 10 + 30 + 40 + 40 = 120; B: 10 (from t1 only); C: 30 (from t3)
        assert metrics["A"] == 120.0
        assert metrics["B"] == 10.0
        assert metrics["C"] == 30.0

    def test_event_line_filter(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        # Filter on the event line itself — polygons whose primary_key in {t1, t2}
        metrics, counts = aggregate_filtered_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "sum",
            resolved_filters=[("transactions", {"t1", "t2"})],
            event_line_id="transactions",
        )
        # A: 10 (t1); B: 10 + 20 = 30; C: 20
        assert metrics["A"] == 10.0
        assert metrics["B"] == 30.0
        assert metrics["C"] == 20.0

    def test_multiple_filters_and(self, tmp_path):
        """Multiple resolved filters are AND'd — both must match."""
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        # Filter accounts={A} AND currencies={USD} → polygons t1, t3, t4
        metrics, counts = aggregate_filtered_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "sum",
            resolved_filters=[("accounts", {"A"}), ("currencies", {"USD"})],
            event_line_id="transactions",
        )
        # t2 excluded (EUR). A: 10+30+40+40=120; B: 10 (t1); C: 30 (t3)
        assert metrics["A"] == 120.0
        assert "B" in metrics
        assert "C" in metrics

    def test_empty_filter_set_returns_empty(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        metrics, counts = aggregate_filtered_metric(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", "sum",
            resolved_filters=[("accounts", set())],
            event_line_id="transactions",
        )
        assert metrics == {} and counts == {}


# ── aggregate_pivot ────────────────────────────────────────────────────


class TestAggregatePivot:
    def test_count_only_pivot(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, pa.table({
            "primary_key": ["t1", "t2", "t3", "t4"],
            "channel": ["online", "atm", "online", "atm"],
        }))
        results = aggregate_pivot(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "channel",
        )
        cells = {(r["group_key"], r["pivot_val"]): r["count"] for r in results}
        # A in [t1, t3, t4×2]: online=2 (t1, t3), atm=2 (t4, t4)
        assert cells[("A", "online")] == 2
        assert cells[("A", "atm")] == 2

    def test_metric_pivot(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        event_path = _write_ctx(tmp_path, pa.table({
            "primary_key": ["t1", "t2", "t3", "t4"],
            "channel": ["online", "atm", "online", "atm"],
        }))
        amt_path = str(tmp_path / "amt.lance")
        lance.write_dataset(_basic_ctx_table(), amt_path, data_storage_version="2.2")
        results = aggregate_pivot(
            geo_path, event_path, _accounts_pattern(),
            "accounts", "channel",
            ctx_lance_path=amt_path,
            metric_col="amount", agg_fn="sum",
        )
        cells = {(r["group_key"], r["pivot_val"]): r["metric"] for r in results}
        assert cells[("A", "online")] == pytest.approx(40.0)  # t1 + t3
        assert cells[("A", "atm")] == pytest.approx(80.0)  # t4 + t4


# ── aggregate_property ─────────────────────────────────────────────────


class TestAggregateProperty:
    def test_distinct_count_per_prop_val(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        prop_path = _write_ctx(tmp_path, pa.table({
            "primary_key": ["A", "B", "C"],
            "tier": ["gold", "silver", "gold"],
        }))
        result = aggregate_property(
            geo_path, prop_path, _accounts_pattern(),
            "accounts", "accounts", "tier",
            distinct=True,
        )
        # gold = {A, C} = 2 distinct, silver = {B} = 1
        assert result["distinct_results"] == {"gold": 2, "silver": 1}

    def test_unsafe_prop_name_identifier_raises(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        prop_path = _write_ctx(tmp_path, pa.table({
            "primary_key": ["A", "B", "C"],
            "tier": ["gold", "silver", "gold"],
        }))
        with pytest.raises(ValueError, match="not a safe SQL identifier"):
            aggregate_property(
                geo_path, prop_path, _accounts_pattern(),
                "accounts", "accounts", "tier; DROP TABLE x; --",
            )

    def test_metric_sum_by_prop(self, tmp_path):
        """Property aggregation with metric sum."""
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        prop_path = _write_ctx(tmp_path, pa.table({
            "primary_key": ["A", "B", "C"],
            "tier": ["gold", "silver", "gold"],
        }))
        amt_path = str(tmp_path / "amt.lance")
        lance.write_dataset(_basic_ctx_table(), amt_path, data_storage_version="2.2")
        result = aggregate_property(
            geo_path, prop_path, _accounts_pattern(),
            "accounts", "accounts", "tier",
            ctx_lance_path=amt_path,
            metric_col="amount", agg_fn="sum",
        )
        assert "results" in result
        cells = {(r["group_key"], r["prop_val"]): r["metric"] for r in result["results"]}
        assert ("A", "gold") in cells

    def test_cross_line_property(self, tmp_path):
        """Property from a different line than group_by_line."""
        geo = pa.table({
            "primary_key": ["t1", "t2", "t3"],
            "entity_keys": pa.array(
                [["A", "USD"], ["B", "EUR"], ["A", "USD"]],
                type=pa.list_(pa.string()),
            ),
        })
        geo_path = _write_geo(tmp_path, geo)
        prop_path = _write_ctx(tmp_path, pa.table({
            "primary_key": ["USD", "EUR"],
            "region": ["americas", "europe"],
        }))
        pat = _Pat(relations=[_Rel("accounts"), _Rel("currencies")])
        result = aggregate_property(
            geo_path, prop_path, pat,
            "accounts", "currencies", "region",
        )
        cells = {(r["group_key"], r["prop_val"]): r["count"] for r in result["results"]}
        assert ("A", "americas") in cells
        assert ("B", "europe") in cells

    def test_count_by_prop(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        prop_path = _write_ctx(tmp_path, pa.table({
            "primary_key": ["A", "B", "C"],
            "tier": ["gold", "silver", "gold"],
        }))
        result = aggregate_property(
            geo_path, prop_path, _accounts_pattern(),
            "accounts", "accounts", "tier",
        )
        cells = {(r["group_key"], r["prop_val"]): r["count"] for r in result["results"]}
        assert cells[("A", "gold")] == 4  # A appears 4 times across positions
        assert cells[("B", "silver")] == 2
        assert cells[("C", "gold")] == 2


# ── aggregate_percentile ───────────────────────────────────────────────


class TestAggregatePercentile:
    def test_median(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        metrics, counts = aggregate_percentile(
            geo_path, ctx_path, _accounts_pattern(),
            "accounts", "amount", percentile=0.5,
        )
        # A's amounts: [10, 30, 40, 40] — median = 35.0 (approx)
        assert metrics["A"] == pytest.approx(35.0, abs=10.0)
        assert counts["A"] == 4

    def test_invalid_percentile_raises(self, tmp_path):
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        with pytest.raises(ValueError, match="percentile must be in"):
            aggregate_percentile(
                geo_path, ctx_path, _accounts_pattern(),
                "accounts", "amount", percentile=1.5,
            )

    def test_unsafe_metric_col_identifier_raises(self, tmp_path):
        """Integration: helper validates metric_col before any SQL is built."""
        geo_path = _write_geo(tmp_path, _basic_geo_table())
        ctx_path = _write_ctx(tmp_path, _basic_ctx_table())
        with pytest.raises(ValueError, match="not a safe SQL identifier"):
            aggregate_percentile(
                geo_path, ctx_path, _accounts_pattern(),
                "accounts", "amount; DROP TABLE x; --", percentile=0.5,
            )

    def test_invalid_threshold(self, tmp_path):
        from hypertopos.engine.lance_sql_agg import find_anomalies as _fa
        p = _write_geo(tmp_path, _basic_geo_table())
        with pytest.raises(ValueError, match="finite"):
            _fa(p, threshold=float("nan"), top_n=5)

    def test_negative_top_n(self, tmp_path):
        from hypertopos.engine.lance_sql_agg import find_anomalies as _fa
        p = _write_geo(tmp_path, _basic_geo_table())
        with pytest.raises(ValueError, match="top_n"):
            _fa(p, threshold=0, top_n=-1)

    def test_negative_offset(self, tmp_path):
        from hypertopos.engine.lance_sql_agg import find_anomalies as _fa
        p = _write_geo(tmp_path, _basic_geo_table())
        with pytest.raises(ValueError, match="offset"):
            _fa(p, threshold=0, top_n=5, offset=-1)


# ── find_anomalies ─────────────────────────────────────────────────────


class TestFindAnomalies:
    def test_top_n_above_threshold(self, tmp_path):
        p = str(tmp_path / "g.lance")
        lance.write_dataset(
            pa.table({
                "primary_key": ["A", "B", "C", "D"],
                "delta_norm": [5.0, 4.0, 3.0, 1.0],
            }),
            p,
            data_storage_version="2.2",
        )
        result = find_anomalies(p, threshold=2.5, top_n=2, offset=0)
        assert result["total_found"] == 3
        assert result["keys"] == ["A", "B"]
        assert result["delta_norms"] == [5.0, 4.0]

    def test_dedupes_duplicate_primary_keys(self, tmp_path):
        p = str(tmp_path / "g.lance")
        lance.write_dataset(
            pa.table({
                "primary_key": ["A", "A", "B"],
                "delta_norm": [3.0, 5.0, 4.0],
            }),
            p,
            data_storage_version="2.2",
        )
        result = find_anomalies(p, threshold=2.5, top_n=2, offset=0)
        # A keeps the higher norm (5.0)
        assert result["keys"] == ["A", "B"]
        assert result["delta_norms"] == [5.0, 4.0]

    def test_offset_skips(self, tmp_path):
        p = str(tmp_path / "g.lance")
        lance.write_dataset(
            pa.table({
                "primary_key": ["A", "B", "C"],
                "delta_norm": [5.0, 4.0, 3.0],
            }),
            p,
            data_storage_version="2.2",
        )
        result = find_anomalies(p, threshold=2.5, top_n=2, offset=1)
        assert result["keys"] == ["B", "C"]

    def test_offset_beyond_total(self, tmp_path):
        p = str(tmp_path / "g.lance")
        lance.write_dataset(
            pa.table({
                "primary_key": ["A", "B"],
                "delta_norm": [5.0, 4.0],
            }),
            p,
            data_storage_version="2.2",
        )
        result = find_anomalies(p, threshold=2.5, top_n=2, offset=5)
        assert result["total_found"] == 2
        assert result["keys"] == []
