# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""DataFusion-backed aggregate engine for large event patterns.

Replaces pure-Python group-by with SQL-based aggregation via Apache DataFusion.
Achieves ~30x speedup on 5M+ event patterns by pushing group-by and joins
into DataFusion's vectorized execution engine.

All functions expect geometry tables with an ``edges`` struct column
(``list<struct<line_id, point_key, status, direction>>``).  Flat
``edge_line_ids`` / ``edge_point_keys`` columns are materialized internally
from alive edges before SQL execution.

Usage:
    if is_available() and geo_table.num_rows >= DATAFUSION_THRESHOLD:
        result = aggregate_count(geo_table, "customers")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

# Threshold: use DataFusion when geometry table exceeds this row count.
DATAFUSION_THRESHOLD = 500_000

# Lazy import — datafusion is an optional dependency.
try:
    import datafusion as _datafusion
except ImportError:
    _datafusion = None  # type: ignore[assignment]

_VALID_AGG_FNS = frozenset({"sum", "max", "min", "avg"})


def _is_percentile(fn: str) -> bool:
    """Check if fn is 'median' or 'pct<N>' (0-100)."""
    if fn == "median":
        return True
    if fn.startswith("pct") and fn[3:].isdigit():
        n = int(fn[3:])
        return 0 <= n <= 100
    return False


def _percentile_fraction(fn: str) -> float:
    """Convert 'median' → 0.5, 'pct75' → 0.75."""
    if fn == "median":
        return 0.5
    return int(fn[3:]) / 100.0


def is_available() -> bool:
    """Return True if datafusion is importable."""
    return _datafusion is not None


def _materialize_flat_edges(geo_table: pa.Table) -> pa.Table:
    """Derive edge_line_ids/edge_point_keys from ``edges`` struct for SQL.

    Always materializes alive-only flat arrays from the ``edges`` struct
    column.  The resulting table has ``edge_line_ids`` and ``edge_point_keys``
    list<string> columns suitable for ``list_position()`` SQL lookups.

    No fallback for legacy flat columns -- callers must provide ``edges``.
    """
    from hypertopos.utils.arrow import flatten_edges_for_sql
    return flatten_edges_for_sql(geo_table)


def _group_key_expr(group_by_line: str) -> str:
    """Build the SQL expression that extracts the group key from flat-edge arrays."""
    return (
        f"edge_point_keys[CAST(list_position(edge_line_ids, '{group_by_line}') "
        f"AS BIGINT)]"
    )


def _build_where_clause(group_by_line: str) -> str:
    """Build WHERE clause filtering NULLs."""
    gk = _group_key_expr(group_by_line)
    return f"{gk} IS NOT NULL"


def aggregate_count(
    geo_table: pa.Table,
    group_by_line: str,
) -> dict[str, int]:
    """COUNT(*) GROUP BY group_by_line using DataFusion.

    Parameters
    ----------
    geo_table:
        Geometry table with ``edges`` struct column.
    group_by_line:
        Line ID to group by (e.g. "customers").

    Returns
    -------
    dict mapping group key to count. Empty dict if line not present.
    """
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])

    gk = _group_key_expr(group_by_line)
    where = _build_where_clause(group_by_line)

    sql = (
        f"SELECT {gk} AS group_key, COUNT(*) AS value "
        f"FROM geo "
        f"WHERE {where} "
        f"GROUP BY group_key"
    )

    result = ctx.sql(sql).to_arrow_table()
    return _table_to_dict(result, int)


def aggregate_metric(
    geo_table: pa.Table,
    group_by_line: str,
    points_table: pa.Table,
    metric_col: str,
    agg_fn: str,
) -> dict[str, float]:
    """Aggregate a metric column with JOIN using DataFusion.

    Parameters
    ----------
    geo_table:
        Geometry table with ``edges`` struct column.
    group_by_line:
        Line ID to group by.
    points_table:
        Points table with primary_key and the metric column.
    metric_col:
        Column name in points_table to aggregate.
    agg_fn:
        One of "sum", "max", "min", "avg".

    Returns
    -------
    dict mapping group key to aggregated metric value. Empty if line not present.

    Raises
    ------
    ValueError
        If agg_fn is not one of the supported functions.
    """
    if agg_fn not in _VALID_AGG_FNS:
        raise ValueError(
            f"Unsupported agg_fn '{agg_fn}'. Must be one of: {sorted(_VALID_AGG_FNS)}"
        )
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])
    ctx.register_record_batches("points", [points_table.to_batches()])

    gk = _group_key_expr(group_by_line)
    where = _build_where_clause(group_by_line)

    sql = (
        f"SELECT {gk} AS group_key, "
        f"{agg_fn.upper()}(p.{metric_col}) AS value "
        f"FROM geo g "
        f"JOIN points p ON g.primary_key = p.primary_key "
        f"WHERE {where} "
        f"GROUP BY group_key"
    )

    result = ctx.sql(sql).to_arrow_table()
    return _table_to_dict(result, float)


def aggregate_metric_with_count(
    geo_table: pa.Table,
    group_by_line: str,
    points_table: pa.Table,
    metric_col: str,
    agg_fn: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Combined metric + count in one SQL query (single DataFusion session).

    Returns (metric_dict, count_dict) — avoids double registration and
    double scan of the geometry table.
    """
    if agg_fn not in _VALID_AGG_FNS:
        raise ValueError(
            f"Unsupported agg_fn '{agg_fn}'. Must be one of: {sorted(_VALID_AGG_FNS)}"
        )
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])
    ctx.register_record_batches("points", [points_table.to_batches()])

    gk = _group_key_expr(group_by_line)
    where = _build_where_clause(group_by_line)

    sql = (
        f"SELECT {gk} AS group_key, "
        f"{agg_fn.upper()}(p.{metric_col}) AS agg_value, "
        f"COUNT(*) AS cnt "
        f"FROM geo g "
        f"JOIN points p ON g.primary_key = p.primary_key "
        f"WHERE {where} "
        f"GROUP BY group_key"
    )

    result = ctx.sql(sql).to_arrow_table()
    if result.num_rows == 0:
        return {}, {}
    keys = result.column("group_key").to_pylist()
    agg_vals = result.column("agg_value").to_pylist()
    cnts = result.column("cnt").to_pylist()
    return (
        {k: float(v) for k, v in zip(keys, agg_vals, strict=False)},
        {k: int(c) for k, c in zip(keys, cnts, strict=False)},
    )


def aggregate_filtered_metric_with_count(
    geo_table: pa.Table,
    group_by_line: str,
    points_table: pa.Table,
    metric_col: str,
    agg_fn: str,
    resolved_filters: list[tuple[str, set[str]]],
) -> tuple[dict[str, float], dict[str, int]]:
    """Aggregate metric + count with AND edge filters using DataFusion SQL.

    resolved_filters: list of (line_id, key_set) — AND semantics.
    Each filter requires the polygon's alive edge to line_id to have a key in key_set.

    Returns (metric_dict, count_dict). Empty dicts when no rows pass filters.
    """
    if agg_fn not in _VALID_AGG_FNS:
        raise ValueError(
            f"Unsupported agg_fn '{agg_fn}'. Must be one of: {sorted(_VALID_AGG_FNS)}"
        )
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])
    ctx.register_record_batches("points", [points_table.to_batches()])

    gk = _group_key_expr(group_by_line)
    where_parts = [f"{gk} IS NOT NULL"]

    for line_id, key_set in resolved_filters:
        pos_expr = f"CAST(list_position(edge_line_ids, '{line_id}') AS BIGINT)"
        key_expr = f"edge_point_keys[{pos_expr}]"
        if len(key_set) == 1:
            where_parts.append(f"{key_expr} = '{next(iter(key_set))}'")
        else:
            keys_sql = ", ".join(f"'{k}'" for k in sorted(key_set))
            where_parts.append(f"{key_expr} IN ({keys_sql})")

    where = " AND ".join(where_parts)
    sql = (
        f"SELECT {gk} AS group_key, "
        f"{agg_fn.upper()}(p.{metric_col}) AS agg_value, "
        f"COUNT(*) AS cnt "
        f"FROM geo g "
        f"JOIN points p ON g.primary_key = p.primary_key "
        f"WHERE {where} "
        f"GROUP BY group_key"
    )

    result = ctx.sql(sql).to_arrow_table()
    if result.num_rows == 0:
        return {}, {}
    keys = result.column("group_key").to_pylist()
    agg_vals = result.column("agg_value").to_pylist()
    cnts = result.column("cnt").to_pylist()
    return (
        {k: float(v) for k, v in zip(keys, agg_vals, strict=False)},
        {k: int(c) for k, c in zip(keys, cnts, strict=False)},
    )


def aggregate_filtered_pivot_metric_with_count(
    geo_table: pa.Table,
    group_by_line: str,
    ctx_table: pa.Table,
    metric_col: str,
    agg_fn: str,
    pivot_field: str,
    resolved_filters: list[tuple[str, set[str]]],
) -> list[dict]:
    """Aggregate metric + pivot with AND edge filters using DataFusion SQL.

    Groups by (group_by_line key, pivot_field value) and computes metric.
    resolved_filters: list of (line_id, key_set) — AND semantics.

    Returns list of {group_key, pivot_val, metric, count} dicts.
    Empty list when no rows pass filters.
    """
    if agg_fn not in _VALID_AGG_FNS:
        raise ValueError(
            f"Unsupported agg_fn '{agg_fn}'. Must be one of: {sorted(_VALID_AGG_FNS)}"
        )
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])
    ctx.register_record_batches("points", [ctx_table.to_batches()])

    gk = _group_key_expr(group_by_line)
    where_parts = [f"{gk} IS NOT NULL"]

    for line_id, key_set in resolved_filters:
        pos_expr = f"CAST(list_position(edge_line_ids, '{line_id}') AS BIGINT)"
        key_expr = f"edge_point_keys[{pos_expr}]"
        if len(key_set) == 1:
            where_parts.append(f"{key_expr} = '{next(iter(key_set))}'")
        else:
            keys_sql = ", ".join(f"'{k}'" for k in sorted(key_set))
            where_parts.append(f"{key_expr} IN ({keys_sql})")

    where = " AND ".join(where_parts)
    sql = (
        f"SELECT {gk} AS group_key, "
        f"CAST(p.{pivot_field} AS VARCHAR) AS pivot_val, "
        f"{agg_fn.upper()}(p.{metric_col}) AS agg_value, "
        f"COUNT(*) AS cnt "
        f"FROM geo g "
        f"JOIN points p ON g.primary_key = p.primary_key "
        f"WHERE {where} "
        f"GROUP BY group_key, pivot_val"
    )

    result = ctx.sql(sql).to_arrow_table()
    if result.num_rows == 0:
        return []
    return [
        {
            "group_key": gk_val,
            "pivot_val": pv,
            "metric": float(m),
            "count": int(c),
        }
        for gk_val, pv, m, c in zip(
            result.column("group_key").to_pylist(),
            result.column("pivot_val").to_pylist(),
            result.column("agg_value").to_pylist(),
            result.column("cnt").to_pylist(), strict=False,
        )
    ]


def aggregate_pivot_count(
    geo_table: pa.Table,
    group_by_line: str,
    event_table: pa.Table,
    pivot_field: str,
) -> list[dict]:
    """COUNT(*) GROUP BY group_by_line x pivot_field using DataFusion.

    No edge filters. Returns list of {group_key, pivot_val, count} dicts.

    Parameters
    ----------
    geo_table:
        Geometry table with ``edges`` struct column.
    group_by_line:
        Line ID to group by (e.g. "customers").
    event_table:
        Event points table — must contain primary_key and pivot_field columns.
        JOIN is performed on geo.primary_key = event_table.primary_key.
    pivot_field:
        Column name in event_table to use as the pivot dimension.
    """
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])
    ctx.register_record_batches("points", [event_table.to_batches()])

    gk = _group_key_expr(group_by_line)
    where_parts = [f"{gk} IS NOT NULL", f"p.\"{pivot_field}\" IS NOT NULL"]
    where = " AND ".join(where_parts)

    sql = (
        f"SELECT {gk} AS group_key, "
        f"CAST(p.\"{pivot_field}\" AS VARCHAR) AS pivot_val, "
        f"COUNT(*) AS cnt "
        f"FROM geo g "
        f"JOIN points p ON g.primary_key = p.primary_key "
        f"WHERE {where} "
        f"GROUP BY group_key, pivot_val"
    )

    result = ctx.sql(sql).to_arrow_table()
    if result.num_rows == 0:
        return []
    return [
        {"group_key": gk_val, "pivot_val": pv, "count": int(c)}
        for gk_val, pv, c in zip(
            result.column("group_key").to_pylist(),
            result.column("pivot_val").to_pylist(),
            result.column("cnt").to_pylist(), strict=False,
        )
    ]


def aggregate_filtered_gbp_count(
    geo_table: pa.Table,
    group_by_line: str,
    prop_table: pa.Table,
    prop_name: str,
    prop_line_id: str,
    resolved_filters: list[tuple[str, set[str]]],
) -> dict[tuple[str, str | None], int]:
    """COUNT(*) GROUP BY group_by_line x prop_name with AND edge filters.

    Performs a cross-tab count: for each (group_key, prop_val) pair, returns
    the number of polygons where the group_by_line edge matches group_key and
    the prop_line_id edge joins to a point with the given prop_val.

    Parameters
    ----------
    geo_table:
        Geometry table with ``edges`` struct column.
    group_by_line:
        Line ID to group by (e.g. "fx_rates").
    prop_table:
        Points table with columns [primary_key, prop_name]. Provided by
        caller — not read inside this function.
    prop_name:
        Column name in prop_table to use as the second dimension (e.g.
        "loyalty_tier").
    prop_line_id:
        Line ID from which prop_table originates (e.g. "customers"). Used
        to locate the join key in the materialized edge_point_keys.
    resolved_filters:
        Non-empty list of (line_id, key_set) — AND semantics. Each filter
        requires the polygon's edge to line_id to have a key in key_set.

    Returns
    -------
    dict mapping (group_key, prop_val) to count. Empty dict if no rows pass
    filters.

    Raises
    ------
    ValueError
        If resolved_filters is empty (use aggregate_count instead).
    RuntimeError
        If datafusion is not installed.
    """
    if not resolved_filters:
        raise ValueError(
            "resolved_filters must not be empty — use aggregate_count instead"
        )
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])
    ctx.register_record_batches("prop", [prop_table.to_batches()])

    gk = _group_key_expr(group_by_line)

    # Join key: edge pointing to prop_line_id
    prop_pos_expr = (
        f"CAST(list_position(edge_line_ids, '{prop_line_id}') AS BIGINT)"
    )
    prop_key_expr = f"edge_point_keys[{prop_pos_expr}]"

    where_parts = [f"{gk} IS NOT NULL"]
    # Ensure join key is not NULL
    where_parts.append(f"{prop_key_expr} IS NOT NULL")

    for line_id, key_set in resolved_filters:
        pos_expr = f"CAST(list_position(edge_line_ids, '{line_id}') AS BIGINT)"
        key_expr = f"edge_point_keys[{pos_expr}]"
        if len(key_set) == 1:
            where_parts.append(f"{key_expr} = '{next(iter(key_set))}'")
        else:
            keys_sql = ", ".join(f"'{k}'" for k in sorted(key_set))
            where_parts.append(f"{key_expr} IN ({keys_sql})")

    where = " AND ".join(where_parts)
    sql = (
        f"SELECT {gk} AS group_key, "
        f"p.{prop_name} AS prop_val, "
        f"COUNT(*) AS cnt "
        f"FROM geo g "
        f"JOIN prop p ON {prop_key_expr} = p.primary_key "
        f"WHERE {where} "
        f"GROUP BY group_key, prop_val"
    )

    result = ctx.sql(sql).to_arrow_table()
    if result.num_rows == 0:
        return {}
    return {
        (gk_val, pv): int(c)
        for gk_val, pv, c in zip(
            result.column("group_key").to_pylist(),
            result.column("prop_val").to_pylist(),
            result.column("cnt").to_pylist(), strict=False,
        )
    }


def aggregate_percentile(
    geo_table: pa.Table,
    group_by_line: str,
    points_table: pa.Table,
    metric_col: str,
    pct_fraction: float,
) -> tuple[dict[str, float], dict[str, int]]:
    """Approximate percentile via DataFusion APPROX_PERCENTILE_CONT.

    Requires geo_table to have 'edges' struct column (anchor patterns only).
    Returns (percentile_dict, count_dict).

    Parameters
    ----------
    geo_table:
        Geometry table with ``edges`` struct column.
    group_by_line:
        Line ID to group by (e.g. "customers").
    points_table:
        Points table with primary_key and the metric column.
    metric_col:
        Column name in points_table to aggregate.
    pct_fraction:
        Percentile as a fraction in [0.0, 1.0] (e.g. 0.5 for median, 0.75 for pct75).

    Returns
    -------
    tuple of (percentile_dict, count_dict) mapping group key to value/count.
    Empty dicts when no rows match.

    Raises
    ------
    RuntimeError
        If datafusion is not installed.
    """
    if _datafusion is None:
        raise RuntimeError("datafusion is not installed")

    geo_table = _materialize_flat_edges(geo_table)
    ctx = _datafusion.SessionContext()
    ctx.register_record_batches("geo", [geo_table.to_batches()])
    ctx.register_record_batches("points", [points_table.to_batches()])

    gk = _group_key_expr(group_by_line)
    where = _build_where_clause(group_by_line)

    sql = (
        f"SELECT {gk} AS group_key, "
        f"APPROX_PERCENTILE_CONT(p.{metric_col}, {pct_fraction}) AS value, "
        f"COUNT(*) AS cnt "
        f"FROM geo g "
        f"JOIN points p ON g.primary_key = p.primary_key "
        f"WHERE {where} "
        f"GROUP BY group_key"
    )

    result = ctx.sql(sql).to_arrow_table()
    if result.num_rows == 0:
        return {}, {}
    keys = result["group_key"].to_pylist()
    vals = result["value"].to_pylist()
    cnts = result["cnt"].to_pylist()
    return (
        {k: float(v) for k, v in zip(keys, vals, strict=False)},
        {k: int(c) for k, c in zip(keys, cnts, strict=False)},
    )


def _table_to_dict(table: pa.Table, value_type: type) -> dict:
    """Convert a 2-column Arrow table (group_key, value) to a dict."""
    if table.num_rows == 0:
        return {}
    keys = table.column("group_key").to_pylist()
    values = table.column("value").to_pylist()
    return {k: value_type(v) for k, v in zip(keys, values, strict=False)}
