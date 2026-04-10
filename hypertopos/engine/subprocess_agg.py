# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Persistent subprocess worker for large event pattern aggregation.

Runs as a long-lived subprocess to avoid MCP process overhead (25x slower
in-process) and Python startup cost (~5s per cold start on Windows).

Protocol (line-delimited JSON over stdin/stdout):
  Parent sends: JSON line with aggregate args + \\n
  Worker responds: JSON line with results + \\n
  Worker stays alive until stdin is closed (EOF).

First call includes Python startup (~5s). Subsequent calls ~2-3s.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# ---------------------------------------------------------------------------
# Lance dataset & table caching
# ---------------------------------------------------------------------------

_ds_cache: dict[str, lance.LanceDataset] = {}


def _get_dataset(path: str) -> lance.LanceDataset:
    """Return a cached Lance dataset handle (avoids ~0.3s per re-open)."""
    if path not in _ds_cache:
        _ds_cache[path] = lance.dataset(path)
    return _ds_cache[path]


_table_cache: OrderedDict[tuple, pa.Table] = OrderedDict()
_TABLE_CACHE_MAX = 4


def _get_table(
    path: str,
    columns: list[str] | None = None,
    lance_version: int | None = None,
) -> pa.Table:
    """Return a cached Arrow table, LRU-evicting beyond _TABLE_CACHE_MAX entries.

    When lance_version is provided and differs from the dataset's latest version,
    the pinned version is opened directly (MVCC isolation) and its result is not
    cached (version-specific reads are ephemeral).
    """
    if lance_version is not None:
        ds = _get_dataset(path)
        if lance_version != ds.latest_version:
            versioned_ds = lance.dataset(path, version=lance_version)
            return versioned_ds.to_table(columns=columns) if columns else versioned_ds.to_table()

    key = (path, tuple(sorted(columns)) if columns else None)
    if key in _table_cache:
        _table_cache.move_to_end(key)
        return _table_cache[key]
    ds = _get_dataset(path)
    table = ds.to_table(columns=columns) if columns else ds.to_table()
    _table_cache[key] = table
    while len(_table_cache) > _TABLE_CACHE_MAX:
        _table_cache.popitem(last=False)
    return table


def _flatten_edges(geo: pa.Table) -> tuple:
    """Flatten edges struct column into flat arrays.

    Returns (flat_lids, flat_pkeys, offsets, alive_np).
    """
    edges_col = geo["edges"]
    flat = pc.list_flatten(edges_col)
    flat_lids = pc.struct_field(flat, "line_id")
    flat_pkeys = pc.struct_field(flat, "point_key")
    flat_status = pc.struct_field(flat, "status")
    offsets = edges_col.combine_chunks().offsets.to_numpy()
    alive_np = pc.equal(flat_status, "alive").to_numpy(zero_copy_only=False)
    flat_lids_np = flat_lids.to_numpy(zero_copy_only=False)
    return flat_lids_np, flat_pkeys, offsets, alive_np


def _count_aggregate_tables(
    geo: pa.Table,
    group_by_line: str,
) -> dict[str, int]:
    """Vectorized count of edges to group_by_line. Returns {key: count}."""
    flat_lids_np, flat_pkeys, offsets, alive_np = _flatten_edges(geo)
    mask = (flat_lids_np == group_by_line) & alive_np
    matched_keys = pc.take(flat_pkeys, np.where(mask)[0])
    vc = pc.value_counts(matched_keys)
    return {v["values"].as_py(): v["counts"].as_py() for v in vc}


def _cmd_count_aggregate(
    geo_lance_path: str,
    group_by_line: str,
    geo_columns: list[str],
) -> dict:
    """COUNT(*) GROUP BY group_by_line from Lance geometry."""
    geo = _get_table(geo_lance_path, geo_columns)
    counts = _count_aggregate_tables(geo, group_by_line)
    return {"counts": counts}


def _cmd_metric_aggregate(
    geo_lance_path: str,
    ctx_lance_path: str,
    group_by_line: str,
    metric_col: str,
    agg_fn: str,
    geo_columns: list[str],
    group_by_line_2: str | None = None,
) -> dict:
    """Run full Arrow aggregation pipeline and return results dict."""
    geo = _get_table(geo_lance_path, geo_columns)
    ctx = _get_table(ctx_lance_path)

    # Arrow-native join: geo.primary_key → ctx.metric_col
    idx = pc.index_in(geo["primary_key"], ctx["primary_key"])
    polygon_amounts = pc.cast(pc.take(ctx[metric_col], idx), pa.float64())

    # Flatten edges struct → find group_by_line edges
    flat_lids_np, flat_pkeys, offsets, alive_np = _flatten_edges(geo)

    mask = (flat_lids_np == group_by_line) & alive_np

    flat_idx = np.where(mask)[0]
    row_idx = np.searchsorted(offsets[1:], flat_idx, side="right")

    gbl_keys = pc.take(flat_pkeys, flat_idx)
    edge_amounts = pc.take(polygon_amounts, row_idx)

    # Multi-level GROUP BY: join two anchor lines
    if group_by_line_2 is not None:
        mask2 = (flat_lids_np == group_by_line_2) & alive_np
        flat_idx2 = np.where(mask2)[0]
        row_idx2 = np.searchsorted(offsets[1:], flat_idx2, side="right")
        gbl2_keys = pc.take(flat_pkeys, flat_idx2)

        # Join the two GBL tables on row_idx
        t1 = pa.table({
            "row_idx": pa.array(row_idx, type=pa.int64()),
            "key_1": gbl_keys,
            "amount": edge_amounts,
        })
        t2 = pa.table({
            "row_idx": pa.array(row_idx2, type=pa.int64()),
            "key_2": gbl2_keys,
        })
        joined = t1.join(t2, keys="row_idx", join_type="inner")

        agg_op_map = {
            "sum": "sum", "avg": "mean",
            "min": "min", "max": "max",
        }
        agg_op = agg_op_map[agg_fn]

        valid = pc.is_valid(joined["amount"])
        agg_tbl = pa.table({
            "key_1": joined["key_1"].filter(valid),
            "key_2": joined["key_2"].filter(valid),
            "amount": joined["amount"].filter(valid),
        })
        result = agg_tbl.group_by(["key_1", "key_2"]).aggregate([
            ("amount", agg_op),
            ("amount", "count"),
        ])

        keys = [
            f"{r['key_1']}\x00{r['key_2']}" for r in result.to_pylist()
        ]
        vals = result[f"amount_{agg_op}"].to_pylist()
        cnts = result["amount_count"].to_pylist()
        return {
            "metrics": dict(zip(keys, vals, strict=False)),
            "counts": dict(zip(keys, cnts, strict=False)),
        }

    # Group-by aggregate (single line)
    valid = pc.is_valid(edge_amounts)
    agg_op_map = {
        "sum": "sum", "avg": "mean",
        "min": "min", "max": "max",
    }
    agg_op = agg_op_map[agg_fn]

    agg_tbl = pa.table({
        "group_key": gbl_keys.filter(valid),
        "amount": edge_amounts.filter(valid),
    })
    result = agg_tbl.group_by("group_key").aggregate([
        ("amount", agg_op),
        ("amount", "count"),
    ])

    keys = result["group_key"].to_pylist()
    vals = result[f"amount_{agg_op}"].to_pylist()
    cnts = result["amount_count"].to_pylist()

    return {
        "metrics": dict(zip(keys, vals, strict=False)),
        "counts": dict(zip(keys, cnts, strict=False)),
    }


def _pivot_aggregate_tables(
    geo_table: pa.Table,
    event_table: pa.Table,
    group_by_line: str,
    pivot_field: str,
    ctx_table: pa.Table | None = None,
    metric_col: str | None = None,
    agg_fn: str | None = None,
) -> list[dict]:
    """Core pivot aggregation on Arrow tables.

    Groups edges by (group_key, pivot_val) and computes count (always)
    plus an optional metric (sum/avg/min/max) when ctx_table is provided.

    Returns list of dicts: [{group_key, pivot_val, count, metric?}, ...].
    """
    # Build pivot_lookup: primary_key → str(pivot_val)
    pivot_lookup_arr = pa.array(
        [str(v) if v is not None else None
         for v in event_table[pivot_field].to_pylist()]
    )
    pivot_keys = event_table["primary_key"]

    # Flatten edges struct → find group_by_line edges
    flat_lids_np, flat_pkeys, offsets, alive_np = _flatten_edges(geo_table)

    mask = (flat_lids_np == group_by_line) & alive_np

    flat_idx = np.where(mask)[0]
    row_idx = np.searchsorted(offsets[1:], flat_idx, side="right")

    gbl_keys = pc.take(flat_pkeys, flat_idx)

    # Per-polygon pivot values via lookup
    geo_pk_idx = pc.index_in(geo_table["primary_key"], pivot_keys)
    polygon_pivot_vals = pc.take(pivot_lookup_arr, geo_pk_idx)

    # Fancy index: pivot val for each edge
    edge_pivot_vals = pc.take(polygon_pivot_vals, row_idx)

    # Filter valid (non-null group key and pivot val)
    valid = pc.and_(pc.is_valid(gbl_keys), pc.is_valid(edge_pivot_vals))

    if agg_fn and ctx_table is not None and metric_col:
        # Metric mode: count + aggregate
        ctx_idx = pc.index_in(geo_table["primary_key"], ctx_table["primary_key"])
        polygon_amounts = pc.cast(pc.take(ctx_table[metric_col], ctx_idx), pa.float64())
        edge_amounts = pc.take(polygon_amounts, row_idx)

        all_valid = pc.and_(valid, pc.is_valid(edge_amounts))
        agg_op_map = {"sum": "sum", "avg": "mean", "min": "min", "max": "max"}
        agg_op = agg_op_map[agg_fn]

        agg_tbl = pa.table({
            "group_key": gbl_keys.filter(all_valid),
            "pivot_val": edge_pivot_vals.filter(all_valid),
            "amount": edge_amounts.filter(all_valid),
        })
        result = agg_tbl.group_by(["group_key", "pivot_val"]).aggregate([
            ("amount", agg_op),
            ("amount", "count"),
        ])
        return [
            {
                "group_key": row["group_key"],
                "pivot_val": row["pivot_val"],
                "count": row["amount_count"],
                "metric": row[f"amount_{agg_op}"],
            }
            for row in result.to_pylist()
        ]
    else:
        # Count-only mode
        agg_tbl = pa.table({
            "group_key": gbl_keys.filter(valid),
            "pivot_val": edge_pivot_vals.filter(valid),
        })
        result = agg_tbl.group_by(["group_key", "pivot_val"]).aggregate(
            [("group_key", "count")]
        )
        return [
            {
                "group_key": row["group_key"],
                "pivot_val": row["pivot_val"],
                "count": row["group_key_count"],
            }
            for row in result.to_pylist()
        ]


def _cmd_pivot_aggregate(
    geo_lance_path: str,
    event_lance_path: str,
    group_by_line: str,
    pivot_field: str,
    geo_columns: list[str],
    ctx_lance_path: str | None = None,
    metric_col: str | None = None,
    agg_fn: str | None = None,
) -> dict:
    """Run pivot aggregation reading from Lance datasets."""
    geo_table = _get_table(geo_lance_path, geo_columns)
    event_table = _get_table(event_lance_path, ["primary_key", pivot_field])
    ctx_table = None
    if ctx_lance_path and metric_col and agg_fn:
        ctx_table = _get_table(ctx_lance_path)

    # DataFusion fast path: count-only pivot (no metric aggregation)
    if metric_col is None:
        try:
            from hypertopos.engine.datafusion_agg import (
                aggregate_pivot_count,
                is_available,
            )
            if is_available():
                results = aggregate_pivot_count(
                    geo_table, group_by_line, event_table, pivot_field,
                )
                return {"results": results}
        except Exception:
            pass

    results = _pivot_aggregate_tables(
        geo_table, event_table, group_by_line, pivot_field,
        ctx_table=ctx_table, metric_col=metric_col, agg_fn=agg_fn,
    )
    return {"results": results}


def _property_aggregate_tables(
    geo_table: pa.Table,
    prop_table: pa.Table,
    group_by_line: str,
    prop_line_id: str,
    prop_name: str,
    distinct: bool = False,
    ctx_table: pa.Table | None = None,
    metric_col: str | None = None,
    agg_fn: str | None = None,
) -> dict:
    """Core property aggregation on Arrow tables.

    Groups edges by (group_key, prop_val) and computes count or metric.
    When distinct=True, returns count of distinct group_keys per prop_val.

    Returns dict with either "results" or "distinct_results" key.
    """
    # Build prop_lookup: primary_key → prop_value
    prop_lookup_arr = pa.array(
        [str(v) if v is not None else None
         for v in prop_table[prop_name].to_pylist()]
    )
    prop_keys = prop_table["primary_key"]

    # Flatten edges struct → find group_by_line edges
    flat_lids_np, flat_pkeys, offsets, alive_np = _flatten_edges(geo_table)

    gbl_mask = (flat_lids_np == group_by_line) & alive_np

    gbl_flat_idx = np.where(gbl_mask)[0]
    gbl_row_idx = np.searchsorted(offsets[1:], gbl_flat_idx, side="right")
    gbl_keys = pc.take(flat_pkeys, gbl_flat_idx)

    # Resolve prop_entity_keys
    if prop_line_id == group_by_line:
        prop_entity_keys = gbl_keys
    else:
        # Different line → extract prop_line edges and join via row_idx
        prop_mask = (flat_lids_np == prop_line_id) & alive_np
        prop_flat_idx = np.where(prop_mask)[0]
        prop_row_idx = np.searchsorted(offsets[1:], prop_flat_idx, side="right")
        prop_edge_keys = pc.take(flat_pkeys, prop_flat_idx)

        gbl_join = pa.table({
            "row_idx": pa.array(gbl_row_idx, type=pa.int64()),
            "group_key": gbl_keys,
        })
        prop_join = pa.table({
            "row_idx": pa.array(prop_row_idx, type=pa.int64()),
            "prop_entity": prop_edge_keys,
        })
        joined = gbl_join.join(prop_join, keys="row_idx", join_type="inner")
        gbl_keys = joined["group_key"]
        prop_entity_keys = joined["prop_entity"]
        gbl_row_idx = joined["row_idx"].to_numpy()

    # Map prop_entity_key → prop_val via lookup
    prop_idx = pc.index_in(prop_entity_keys, prop_keys)
    prop_vals = pc.take(prop_lookup_arr, prop_idx)

    valid = pc.and_(pc.is_valid(gbl_keys), pc.is_valid(prop_vals))

    if distinct:
        # Count distinct group_keys per prop_val
        distinct_tbl = pa.table({
            "group_key": gbl_keys.filter(valid),
            "prop_val": prop_vals.filter(valid),
        })
        result = distinct_tbl.group_by("prop_val").aggregate(
            [("group_key", "count_distinct")]
        )
        return {
            "distinct_results": {
                row["prop_val"]: row["group_key_count_distinct"]
                for row in result.to_pylist()
            }
        }

    if agg_fn and ctx_table is not None and metric_col:
        # Metric mode: count + aggregate by (group_key, prop_val)
        ctx_idx = pc.index_in(
            geo_table["primary_key"], ctx_table["primary_key"]
        )
        polygon_amounts = pc.cast(
            pc.take(ctx_table[metric_col], ctx_idx), pa.float64()
        )
        edge_amounts = pc.take(polygon_amounts, gbl_row_idx)
        all_valid = pc.and_(valid, pc.is_valid(edge_amounts))

        agg_op_map = {"sum": "sum", "avg": "mean", "min": "min", "max": "max"}
        agg_op = agg_op_map[agg_fn]

        agg_tbl = pa.table({
            "group_key": gbl_keys.filter(all_valid),
            "prop_val": prop_vals.filter(all_valid),
            "amount": edge_amounts.filter(all_valid),
        })
        result = agg_tbl.group_by(["group_key", "prop_val"]).aggregate([
            ("amount", agg_op),
            ("amount", "count"),
        ])
        return {
            "results": [
                {
                    "group_key": row["group_key"],
                    "prop_val": row["prop_val"],
                    "count": row["amount_count"],
                    "metric": row[f"amount_{agg_op}"],
                }
                for row in result.to_pylist()
            ]
        }

    # Count-only mode: group by (group_key, prop_val) → count
    agg_tbl = pa.table({
        "group_key": gbl_keys.filter(valid),
        "prop_val": prop_vals.filter(valid),
    })
    result = agg_tbl.group_by(["group_key", "prop_val"]).aggregate(
        [("group_key", "count")]
    )
    return {
        "results": [
            {
                "group_key": row["group_key"],
                "prop_val": row["prop_val"],
                "count": row["group_key_count"],
            }
            for row in result.to_pylist()
        ]
    }


def _cmd_property_aggregate(
    geo_lance_path: str,
    prop_lance_path: str,
    group_by_line: str,
    prop_line_id: str,
    prop_name: str,
    geo_columns: list[str],
    distinct: bool = False,
    ctx_lance_path: str | None = None,
    metric_col: str | None = None,
    agg_fn: str | None = None,
) -> dict:
    """Run property aggregation reading from Lance datasets."""
    geo_table = _get_table(geo_lance_path, geo_columns)
    prop_table = _get_table(prop_lance_path, ["primary_key", prop_name])
    ctx_table = None
    if ctx_lance_path and metric_col and agg_fn:
        ctx_table = _get_table(ctx_lance_path)

    return _property_aggregate_tables(
        geo_table, prop_table, group_by_line, prop_line_id, prop_name,
        distinct=distinct, ctx_table=ctx_table,
        metric_col=metric_col, agg_fn=agg_fn,
    )


def _filtered_metric_aggregate_tables(
    geo_table: pa.Table,
    ctx_table: pa.Table,
    group_by_line: str,
    metric_col: str,
    agg_fn: str,
    resolved_filters: list[list],
    event_line_id: str,
) -> dict:
    """Core filtered metric aggregation on Arrow tables.

    Applies edge filters per polygon before aggregating metric values.
    resolved_filters is a list of [line_id, [allowed_keys]] pairs.

    Returns {"metrics": {key: val}, "counts": {key: cnt}}.
    """
    # Convert resolved_filters lists to sets for O(1) lookup
    filter_pairs: list[tuple[str, set[str]]] = [
        (f[0], set(f[1])) for f in resolved_filters
    ]

    # Flatten edges struct
    flat_lids_np, flat_pkeys, offsets, alive_np = _flatten_edges(geo_table)
    flat_pkeys_list = flat_pkeys.to_pylist()

    bk_list = geo_table["primary_key"].to_pylist()
    n_rows = len(bk_list)

    # Build per-polygon edge maps and filter
    pass_indices = []
    group_keys_out: list[str] = []

    for i in range(n_rows):
        start = offsets[i]
        end = offsets[i + 1]
        # Build edge_map for this polygon
        edge_map: dict[str, str] = {}
        for j in range(start, end):
            if alive_np is not None and not alive_np[j]:
                continue
            lid = flat_lids_np[j]
            edge_map[lid] = flat_pkeys_list[j]
        # Add event_line_id → polygon's own key
        edge_map[event_line_id] = bk_list[i]

        # Check all filters
        if not all(edge_map.get(fl) in ks for fl, ks in filter_pairs):
            continue

        # Extract group_by_line key
        gk = edge_map.get(group_by_line)
        if gk is None:
            continue

        pass_indices.append(i)
        group_keys_out.append(gk)

    if not pass_indices:
        return {"metrics": {}, "counts": {}}

    # Lookup metric values for passing polygons
    idx_arr = pa.array(pass_indices, type=pa.int64())
    passing_pkeys = pc.take(geo_table["primary_key"], idx_arr)
    ctx_idx = pc.index_in(passing_pkeys, ctx_table["primary_key"])
    amounts = pc.cast(pc.take(ctx_table[metric_col], ctx_idx), pa.float64())

    # Group-by aggregate
    valid = pc.is_valid(amounts)
    agg_op_map = {"sum": "sum", "avg": "mean", "min": "min", "max": "max"}
    agg_op = agg_op_map[agg_fn]

    agg_tbl = pa.table({
        "group_key": pa.array(group_keys_out).filter(valid),
        "amount": amounts.filter(valid),
    })
    result = agg_tbl.group_by("group_key").aggregate([
        ("amount", agg_op),
        ("amount", "count"),
    ])

    keys = result["group_key"].to_pylist()
    vals = result[f"amount_{agg_op}"].to_pylist()
    cnts = result["amount_count"].to_pylist()

    return {
        "metrics": dict(zip(keys, vals, strict=False)),
        "counts": dict(zip(keys, cnts, strict=False)),
    }


def _cmd_filtered_metric_aggregate(
    geo_lance_path: str,
    ctx_lance_path: str,
    group_by_line: str,
    metric_col: str,
    agg_fn: str,
    geo_columns: list[str],
    resolved_filters: list[list],
    event_line_id: str,
) -> dict:
    """Run filtered metric aggregation reading from Lance datasets."""
    geo_table = _get_table(geo_lance_path, geo_columns)
    ctx_table = _get_table(ctx_lance_path)

    try:
        from hypertopos.engine.datafusion_agg import (
            aggregate_filtered_metric_with_count,
            is_available,
        )
        if is_available():
            filter_pairs = [(f[0], set(f[1])) for f in resolved_filters]
            metrics, counts = aggregate_filtered_metric_with_count(
                geo_table, group_by_line, ctx_table, metric_col, agg_fn, filter_pairs,
            )
            return {"metrics": metrics, "counts": counts}
    except Exception:
        pass

    return _filtered_metric_aggregate_tables(
        geo_table, ctx_table, group_by_line, metric_col, agg_fn,
        resolved_filters=resolved_filters, event_line_id=event_line_id,
    )


def _find_anomalies_table(
    geo_table: pa.Table,
    threshold: float,
    top_n: int,
    offset: int = 0,
) -> dict:
    """Core find_anomalies on an Arrow table.

    Filters rows where delta_norm >= threshold, sorts descending,
    returns top_n keys and their delta_norms starting at offset.

    Always returns total_found (count of rows meeting threshold) so
    callers can implement pagination without a separate count query.
    """
    delta_norms = geo_table["delta_norm"]
    mask = pc.greater_equal(delta_norms, threshold)
    filtered = geo_table.filter(mask)

    total_found = filtered.num_rows
    if total_found == 0 or offset >= total_found:
        return {"keys": [], "delta_norms": [], "total_found": total_found}

    # Sort descending by delta_norm, slice [offset : offset + top_n]
    indices = pc.sort_indices(filtered, sort_keys=[("delta_norm", "descending")])
    page_indices = indices[offset : offset + top_n]

    keys = pc.take(filtered["primary_key"], page_indices).to_pylist()
    norms = pc.take(filtered["delta_norm"], page_indices).to_pylist()

    # Deduplicate on primary_key, keeping highest delta_norm
    seen: dict[str, float] = {}
    for k, n in zip(keys, norms, strict=False):
        if k not in seen or n > seen[k]:
            seen[k] = n
    keys = list(seen.keys())
    norms = list(seen.values())

    return {"keys": keys, "delta_norms": norms, "total_found": total_found}


def _cmd_find_anomalies(
    geo_lance_path: str,
    threshold: float,
    top_n: int,
    lance_version: int | None = None,
    offset: int = 0,
) -> dict:
    """Find top-N anomalous polygons by delta_norm from Lance geometry."""
    geo_table = _get_table(
        geo_lance_path, ["primary_key", "delta_norm"], lance_version=lance_version
    )
    return _find_anomalies_table(geo_table, threshold, top_n, offset)


def _cmd_anomaly_summary(**_kwargs: object) -> dict:
    """Deferred — anomaly_summary has a well-optimized cache path in the navigator.

    Subprocess acceleration is not needed; the navigator reads geometry_stats
    (a few KB) and only falls back to a full scan on cache miss.
    """
    return {"error": "not implemented: anomaly_summary (deferred — cache path sufficient)"}


_COMMANDS: dict[str, object] = {
    "count_aggregate": _cmd_count_aggregate,
    "metric_aggregate": _cmd_metric_aggregate,
    "pivot_aggregate": _cmd_pivot_aggregate,
    "property_aggregate": _cmd_property_aggregate,
    "filtered_metric_aggregate": _cmd_filtered_metric_aggregate,
    "find_anomalies": _cmd_find_anomalies,
    "anomaly_summary": _cmd_anomaly_summary,
}


def _dispatch(args: dict) -> dict:
    """Dispatch a command dict to the appropriate handler."""
    command = args.pop("command", "metric_aggregate")
    handler = _COMMANDS.get(command)
    if handler is None:
        return {"error": f"unknown command: {command}"}
    return handler(**args)


def _worker_loop() -> None:
    """Read JSON commands from stdin, process, write JSON results to stdout."""
    # Signal readiness after imports are done
    sys.stdout.write('{"ready":true}\n')
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            args = json.loads(line)
            result = _dispatch(args)
            sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
        except Exception as exc:
            sys.stdout.write(
                json.dumps({"error": str(exc)}, separators=(",", ":"))
                + "\n"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_loop()
    else:
        # One-shot mode (backward compat)
        raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
        args = json.loads(raw)
        result = _dispatch(args)
        print(json.dumps(result, separators=(",", ":")))
