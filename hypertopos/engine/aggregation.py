# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Core aggregation engine — count, sum, avg, min, max with filtering.

The fast path for queries above ``_SUBPROCESS_THRESHOLD`` rows pushes
GROUP BY computation into the Lance scanner via :mod:`hypertopos.engine.lance_sql_agg`.
Smaller queries and queries that combine the new path with filters
``lance_sql_agg`` does not yet handle (``filter_by_keys`` /
``event_filters`` / ``entity_filters``) drop into the in-process pyarrow
handlers below — those are sibling code paths, not legacy fallbacks.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Manifest
    from hypertopos.model.sphere import Sphere
    from hypertopos.storage.reader import GDSReader

# Lance SQL fast path threshold — geometry tables above this row count
# push GROUP BY into the Lance scanner via lance_sql_agg.
_SUBPROCESS_THRESHOLD = 500_000

# Reversed-scan threshold — reversed scan (per-group Lance count) is faster
# than vectorized when group cardinality is small. Above this limit the
# vectorized path (read all geometry once + flatten/value_counts) wins.
# Benchmarked: reversed scan ~2ms/group, vectorized ~700ms flat cost on 1M rows.
# Breakeven at ~350 groups; use 500 as conservative threshold.
_REVERSED_SCAN_MAX_GROUPS = 500


def _apply_comparison_filter(table: Any, column: str, spec: dict) -> Any:
    """Apply comparison filter: {"gt": 5}, {"lt": 10}, {"gte": 0, "lte": 100}."""
    import pyarrow as pa
    import pyarrow.compute as pc
    _OPS = {"gt": pc.greater, "gte": pc.greater_equal,
            "lt": pc.less, "lte": pc.less_equal, "eq": pc.equal}
    mask = None
    col_type = table.schema.field(column).type
    for op_name, value in spec.items():
        if op_name not in _OPS:
            raise RuntimeError(f"Unknown comparison op '{op_name}'. Supported: {list(_OPS)}")
        if isinstance(value, str) and pa.types.is_date(col_type):
            from datetime import date
            scalar = pa.scalar(date.fromisoformat(value), type=col_type)
        else:
            scalar = value
        pred = _OPS[op_name](table[column], scalar)
        mask = pred if mask is None else pc.and_(mask, pred)
    return table.filter(mask)


def _apply_event_filters(table: Any, event_filters: dict) -> Any:
    """Apply column predicates to an event-line table.

    Format: {"col": value} for equality, {"col": {"gt": v, "lt": v}} for comparison,
    {"col": null} for IS NULL, {"col": {"not_null": true}} for IS NOT NULL.
    Multiple columns = AND. Multiple ops per column = AND.
    """
    import pyarrow.compute as pc
    for col, spec in event_filters.items():
        if col not in table.schema.names:
            raise RuntimeError(
                f"Context column '{col}' not found. "
                f"Available: {[n for n in table.schema.names if n != 'primary_key']}"
            )
        if spec is None:
            table = table.filter(pc.is_null(table[col]))
        elif isinstance(spec, dict):
            if spec.get("not_null") in (True, "true", "True"):
                if len(spec) > 1:
                    raise RuntimeError(
                        f"not_null cannot be combined with other operators "
                        f"for column '{col}'. Use separate filter calls."
                    )
                table = table.filter(pc.is_valid(table[col]))
            else:
                table = _apply_comparison_filter(table, col, spec)
        else:
            table = _apply_comparison_filter(table, col, {"eq": spec})
    return table


def _build_property_filter_mask(
    table: Any, property_filters: dict, context: str = "entity line",
) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc
    _OPS = {"=": pc.equal, "<": pc.less, "<=": pc.less_equal,
            ">": pc.greater, ">=": pc.greater_equal, "!=": pc.not_equal}
    masks = []
    for prop, spec in property_filters.items():
        if prop not in table.schema.names:
            raise RuntimeError(
                f"Property '{prop}' not found in {context}. "
                f"Available: {list(table.schema.names)}"
            )
        col = table[prop]
        op, val = ("=", spec) if not isinstance(spec, dict) else (spec["op"], spec["value"])
        if op not in _OPS:
            raise RuntimeError(f"Unsupported op '{op}'. Use one of: {list(_OPS)}")
        col_type = table.schema.field(prop).type
        if isinstance(val, str) and pa.types.is_date(col_type):
            from datetime import date
            scalar = pa.scalar(date.fromisoformat(val), type=col_type)
        else:
            scalar = pa.scalar(val, type=col_type)
        masks.append(_OPS[op](col, scalar))
    result = masks[0]
    for m in masks[1:]:
        result = pc.and_(result, m)
    return result


def _resolve_filter_keys(reader: Any, s: Any, line_id: str, key: str) -> set[str]:
    """Resolve filter_key to primary_key set.

    'col:val' resolves by column, plain key returns as-is.
    """
    if ":" not in key:
        return {key}
    import pyarrow.compute as pc
    col, target = key.split(":", 1)
    line = s.lines.get(line_id)
    if line is None:
        return set()
    table = reader.read_points(line_id, line.versions[-1])
    if col not in table.schema.names:
        return {key}  # not a column name — treat as plain primary_key
    import pyarrow as pa
    col_type = table.schema.field(col).type
    scalar = pa.array([target]).cast(col_type)[0]
    return set(table.filter(pc.equal(table[col], scalar))["primary_key"].to_pylist())


def _make_edge_map_fn(geo: Any, relations: list | None = None):
    """Return a closure that builds {line_id: point_key} for polygon i.

    Uses edges struct column when present, otherwise falls back to
    entity_keys + relations.
    """
    if "edges" in geo.schema.names:
        _edges = geo["edges"].to_pylist()
        def _from_edges(i: int) -> dict:
            return {
                e["line_id"]: e["point_key"]
                for e in (_edges[i] or [])
                if e["status"] == "alive"
            }
        return _from_edges
    elif "entity_keys" in geo.schema.names and relations:
        _ek = geo["entity_keys"].to_pylist()
        def _from_entity_keys(i: int) -> dict:
            keys = _ek[i] or []
            return {
                rel.line_id: keys[j]
                for j, rel in enumerate(relations)
                if j < len(keys) and keys[j]
            }
        return _from_entity_keys
    else:
        return lambda i: {}


def _edge_arrays(geo: Any, relations: list | None = None) -> tuple:
    """Extract flat edge arrays from geometry table.

    Supports both edges struct column (anchor) and entity_keys fallback (event).
    Pass relations for event patterns without edges column.
    Returns (row_idx, flat_line_ids, flat_pt_keys, alive_mask).
    """
    import pyarrow.compute as pc

    if "edges" in geo.schema.names:
        flat = pc.list_flatten(geo["edges"])
        row_idx = pc.list_parent_indices(geo["edges"])
        flat_line_ids = pc.struct_field(flat, "line_id")
        flat_pt_keys = pc.struct_field(flat, "point_key")
        alive_mask = pc.equal(pc.struct_field(flat, "status"), "alive")
        return row_idx, flat_line_ids, flat_pt_keys, alive_mask

    # Event geometry: vectorized construction from entity_keys + relations.
    # entity_keys is fixed-width list<string> with D = len(relations) entries
    # per row, positionally mapped: entity_keys[row][j] = relations[j].line_id.
    # Empty string = dead edge.
    if relations is None:
        raise RuntimeError(
            "Geometry table has no 'edges' column and no relations "
            "provided for reconstruction. Rebuild the sphere."
        )
    import pyarrow as pa

    ek = geo["entity_keys"].combine_chunks()
    n = len(ek)
    D = len(relations)
    flat_pkeys = pc.list_flatten(ek)
    lid_pattern = [r.line_id for r in relations]
    flat_line_ids = pa.array(lid_pattern * n, type=pa.string())
    alive_mask = pc.not_equal(flat_pkeys, "")
    row_idx = pa.array(np.repeat(np.arange(n, dtype=np.int64), D), type=pa.int64())
    return row_idx, flat_line_ids, flat_pkeys, alive_mask


def _gbl_edge_arrays(
    group_by_line: str,
    row_idx: Any,
    flat_line_ids: Any,
    flat_pt_keys: Any,
    alive_mask: Any,
) -> tuple:
    """Filter flat edge arrays to (gbl_row_idx, gbl_keys) for the given line.

    Dead edges are excluded via alive_mask.
    Returns (gbl_row_idx, gbl_keys): row indices and point keys for edges on group_by_line.
    """
    import pyarrow.compute as pc
    mask = pc.equal(flat_line_ids, group_by_line)
    if alive_mask is not None:
        mask = pc.and_(mask, alive_mask)
    return row_idx.filter(mask), flat_pt_keys.filter(mask)


def _vectorized_count_with_warning(
    geo: Any,
    group_by_line: str,
    geometry_filters: dict | None,
    relations: list | None = None,
) -> tuple[dict, str | None]:
    """Vectorized count of edges to group_by_line.

    Returns (computed, warning):
    - computed: {group_key: count}
    - warning: human-readable string if geometry_filters are present and some
               geometry-filtered polygons had no edge to group_by_line, else None.

    Uses edges struct column or entity_keys + relations.
    """
    import pyarrow.compute as pc

    row_idx, flat_line_ids, flat_pt_keys, alive_mask = _edge_arrays(geo, relations=relations)
    combined_mask = pc.equal(flat_line_ids, group_by_line)
    if alive_mask is not None:
        combined_mask = pc.and_(combined_mask, alive_mask)
    matched_keys = flat_pt_keys.filter(combined_mask)

    vc = pc.value_counts(matched_keys)
    computed = {v["values"].as_py(): v["counts"].as_py() for v in vc}

    warning: str | None = None
    if geometry_filters:
        matched_row_idx = row_idx.filter(combined_mask)
        unique_covered = pc.count_distinct(matched_row_idx).as_py()
        dropped = geo.num_rows - unique_covered
        if dropped > 0:
            warning = (
                f"{dropped} of {geo.num_rows} geometry-filtered polygons had no edge to "
                f"'{group_by_line}' and were excluded from aggregation. "
                f"Shown counts reflect only the {unique_covered} polygons "
                f"with a '{group_by_line}' edge. "
                f"Use anomaly_summary for population-wide totals."
            )

    return computed, warning


def _vectorized_sample_count(
    geo: Any,
    group_by_line: str,
    sample_size: int | None = None,
    sample_pct: float | None = None,
    seed: int | None = None,
    geometry_filters: dict | None = None,
    relations: list | None = None,
) -> tuple[dict, str | None]:
    """
    Fast vectorized count with sampling. Avoids O(n) Python loop by:
    1. Sampling row indices with numpy (index array only, no string conversion)
    2. Arrow take() to slice geometry to sampled rows
    3. Vectorized count on the small slice via _vectorized_count_with_warning

    Returns (computed, warning) — same contract as _vectorized_count_with_warning.
    Requires geo to have edges struct or entity_keys column.
    """
    import numpy as np
    import pyarrow as pa

    total = geo.num_rows
    if sample_size is not None:
        n = min(sample_size, total)
    elif sample_pct is not None:
        n = min(max(1, int(total * sample_pct)), total)
    else:
        raise ValueError("Either sample_size or sample_pct must be set.")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(total, size=n, replace=False))
    geo_sampled = geo.take(pa.array(chosen, type=pa.int64()))

    return _vectorized_count_with_warning(geo_sampled, group_by_line, geometry_filters, relations=relations)


def aggregate(
    reader: GDSReader,
    engine: GDSEngine,
    sphere: Sphere,
    manifest: Manifest,
    *,
    event_pattern_id: str,
    group_by_line: str,
    group_by_line_2: str | None = None,
    metric: str = "count",
    # context_line removed — metrics read from event entity line
    filters: list[dict] | None = None,
    group_by_property: str | None = None,
    distinct: bool = False,
    collapse_by_property: bool = False,
    geometry_filters: dict | None = None,
    property_filters: dict | None = None,
    entity_filters: dict | None = None,
    event_filters: dict | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    filter_by_keys: list[str] | None = None,
    missing_edge_to: str | None = None,
    having: dict | None = None,
    pivot_event_field: str | None = None,
    sample_size: int | None = None,
    sample_pct: float | None = None,
    seed: int | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "desc",
    **kwargs: Any,
) -> dict:
    """Aggregate event polygons — count, count_distinct, sum, avg, min, max, median, pct<N>.

    Returns a dict with aggregation results. MCP layer handles enrichment
    (entity names, include_properties, pivot_labels) and serialization.
    """
    if "context_line" in kwargs:
        raise ValueError("context_line removed — put metric columns on event entity line")
    if "context_filters" in kwargs:
        raise ValueError("context_filters has been renamed to event_filters")
    if filters is not None and not isinstance(filters, list):
        raise RuntimeError(
            f"filters must be a list of {{'line': str, 'key': str}} dicts, "
            f"got {type(filters).__name__}. "
            "Example: filters=[{'line': 'company_codes', 'key': 'CC-PL'}]"
        )
    if distinct and not group_by_property:
        raise RuntimeError("distinct=True requires group_by_property.")
    if collapse_by_property and not group_by_property:
        raise RuntimeError("collapse_by_property=True requires group_by_property.")
    if collapse_by_property and distinct:
        raise RuntimeError(
            "collapse_by_property=True and distinct=True cannot be combined. "
            "Use collapse_by_property for metric aggregation per tier; "
            "use distinct=True only for counting unique entities per tier."
        )
    if having is not None and pivot_event_field:
        raise RuntimeError("having filter is not supported with pivot_event_field.")
    if pivot_event_field and group_by_property:
        raise RuntimeError("pivot_event_field and group_by_property cannot be combined.")

    # Time window → event_filters injection
    if time_from is not None or time_to is not None:
        for val, label in ((time_from, "time_from"), (time_to, "time_to")):
            if val is not None:
                try:
                    datetime.fromisoformat(val)
                except ValueError:
                    raise ValueError(
                        f"{label}='{val}' is not a valid ISO-8601 date. "
                        f"Expected format: YYYY-MM-DD (e.g. '1995-01-01')."
                    )
        event_pat = sphere.patterns.get(event_pattern_id)
        ts_col = event_pat.timestamp_col if event_pat else None
        if not ts_col:
            raise ValueError(
                f"time_from/time_to requires pattern '{event_pattern_id}' to have "
                f"timestamp_col in sphere.json (set via temporal config in sphere.yaml). "
                f"Use event_filters directly instead."
            )
        time_pred: dict = {}
        if time_from is not None:
            time_pred["gte"] = time_from
        if time_to is not None:
            time_pred["lt"] = time_to
        if event_filters is None:
            event_filters = {}
        if ts_col in event_filters:
            raise ValueError(
                f"Cannot use time_from/time_to and event_filters['{ts_col}'] simultaneously."
            )
        event_filters = {**event_filters, ts_col: time_pred}
    if sample_size is not None and sample_pct is not None:
        raise RuntimeError("sample_size and sample_pct are mutually exclusive.")
    if sample_size is not None and sample_size <= 0:
        raise RuntimeError("sample_size must be a positive integer.")
    if sample_pct is not None and not (0.0 < sample_pct <= 1.0):
        raise RuntimeError("sample_pct must be in range (0.0, 1.0].")
    s = sphere

    if missing_edge_to is not None:
        if missing_edge_to not in s.lines:
            raise RuntimeError(
                f"Unknown line '{missing_edge_to}' in missing_edge_to. "
                f"Available lines: {sorted(s.lines.keys())}"
            )
        _met_pat = s.patterns[event_pattern_id]
        _met_in_pattern = any(r.line_id == missing_edge_to for r in _met_pat.relations)
        if not _met_in_pattern:
            raise RuntimeError(
                f"Pattern '{event_pattern_id}' has no relation to line '{missing_edge_to}'. "
                f"Available relation lines: {[r.line_id for r in _met_pat.relations]}"
            )

    pattern = s.patterns.get(event_pattern_id)
    if pattern is None:
        raise RuntimeError(f"Pattern '{event_pattern_id}' not found.")

    # Validate group_by_line is a declared relation of the pattern
    _gbl_valid = any(r.line_id == group_by_line for r in pattern.relations)
    if not _gbl_valid:
        _valid_lines = sorted(r.line_id for r in pattern.relations)
        raise RuntimeError(
            f"group_by_line='{group_by_line}' is not a relation in "
            f"'{event_pattern_id}'. "
            f"Valid relation lines: {_valid_lines}."
        )

    # Validate group_by_line_2
    if group_by_line_2 is not None:
        _gbl2_valid = any(r.line_id == group_by_line_2 for r in pattern.relations)
        if not _gbl2_valid:
            _valid_lines = sorted(r.line_id for r in pattern.relations)
            raise RuntimeError(
                f"group_by_line_2='{group_by_line_2}' is not a relation in "
                f"'{event_pattern_id}'. Valid relation lines: {_valid_lines}."
            )
        if group_by_line_2 == group_by_line:
            raise RuntimeError(
                "group_by_line_2 must differ from group_by_line."
            )
        if group_by_property:
            raise RuntimeError(
                "group_by_line_2 cannot be combined with group_by_property."
            )
        if pivot_event_field:
            raise RuntimeError(
                "group_by_line_2 cannot be combined with pivot_event_field."
            )

    # --- A: Resolve filters BEFORE read_geometry for point_keys pushdown ---
    resolved_filters: list[tuple[str, set[str]]] = []
    for f in (filters or []):
        keys = _resolve_filter_keys(reader, s, f["line"], f["key"])
        resolved_filters.append((f["line"], keys))

    # Collect entity keys (OR superset) for pushdown; AND applied in-loop
    pushdown_keys: list[str] | None = None
    if resolved_filters:
        all_keys: set[str] = set()
        for _, keys in resolved_filters:
            all_keys.update(keys)
        pushdown_keys = list(all_keys)

    # --- D: Reversed scan count (Lance count_rows pushdown) ---
    # Skips read_geometry entirely — counts via Lance index on entity_keys.
    # geometry_filters always fall through to the vectorized read_geometry path (BUG-6).
    _SQL_OPS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "="}
    use_sampling = sample_size is not None or sample_pct is not None
    use_reversed_scan = (
        metric == "count"
        and ":" not in metric
        and not pivot_event_field
        and not group_by_property
        and not group_by_line_2
        and not use_sampling
        and not distinct
        and geometry_filters is None
        and filter_by_keys is None
        and event_filters is None
        and missing_edge_to is None
        and having is None
        and entity_filters is None
    )
    if use_reversed_scan:
        gbl = s.lines.get(group_by_line)
        if gbl is not None:
            # Resolve allowed_group_keys from property_filters (same logic as below)
            _rs_allowed: set[str] | None = None
            if property_filters:
                gbl_table = reader.read_points(group_by_line, gbl.versions[-1])
                _ctx = f"group_by_line '{group_by_line}'"
                _mask = _build_property_filter_mask(gbl_table, property_filters, context=_ctx)
                _rs_allowed = set(gbl_table.filter(_mask)["primary_key"].to_pylist())

            group_points = reader.read_points(group_by_line, gbl.versions[-1])
            all_group_keys = group_points["primary_key"].to_pylist()
            if _rs_allowed is not None:
                all_group_keys = [k for k in all_group_keys if k in _rs_allowed]

            def _build_lance_filter(group_key: str) -> str:
                parts = [f"array_contains(entity_keys, '{group_key}')"]
                for _, key_set in resolved_filters:
                    if len(key_set) == 1:
                        parts.append(f"array_contains(entity_keys, '{next(iter(key_set))}')")
                    else:
                        keys_str = ", ".join(f"'{k}'" for k in key_set)
                        parts.append(f"array_has_any(entity_keys, [{keys_str}])")
                if geometry_filters:
                    if "is_anomaly" in geometry_filters:
                        val = "true" if bool(geometry_filters["is_anomaly"]) else "false"
                        parts.append(f"is_anomaly = {val}")
                    if "delta_rank_pct" in geometry_filters:
                        spec = geometry_filters["delta_rank_pct"]
                        if isinstance(spec, dict):
                            for op_name, value in spec.items():
                                if op_name in _SQL_OPS:
                                    parts.append(
                        f"delta_rank_pct {_SQL_OPS[op_name]} {float(value)}"
                    )
                        else:
                            parts.append(f"delta_rank_pct = {float(spec)}")
                    if "delta_dim" in geometry_filters:
                        for dim_name, predicates in geometry_filters["delta_dim"].items():
                            idx = pattern.dim_index(dim_name)
                            for op_name, value in predicates.items():
                                if op_name in _SQL_OPS:
                                    parts.append(
                                        f"delta_dim_{idx} {_SQL_OPS[op_name]} {float(value)}"
                                    )
                return " AND ".join(parts)

            try:
                if len(all_group_keys) > _REVERSED_SCAN_MAX_GROUPS:
                    raise RuntimeError("high-cardinality group_by — use vectorized path")
                computed: dict = {}
                group_values: dict = {}
                for gk in all_group_keys:
                    lance_filter = _build_lance_filter(gk)
                    cnt = reader.count_geometry_rows(
                        event_pattern_id, pattern.version, filter=lance_filter,
                    )
                    if cnt > 0:
                        computed[gk] = cnt
                        group_values[gk] = [1] * cnt

                ascending = sort == "asc"

                top = sorted(
                    computed.items(), key=lambda x: x[1], reverse=not ascending,
                )[offset:offset + limit]
                results = [
                    {"key": key, "value": round(val, 2), "count": len(group_values[key])}
                    for key, val in top
                ]
                result = {
                    "event_pattern_id": event_pattern_id,
                    "group_by_line": group_by_line,
                    "group_by_property": group_by_property,
                    "metric": metric,
                    "distinct": distinct,
                    "filters": filters or None,
                    "geometry_filters": geometry_filters,
                    "total_groups": len(computed),
                    "offset": offset,
                    "results": results,
                    "sampled": False,
                }
                # DATA-1: warn when geometry_filters silently drops polygons without GBL edge.
                # Count total polygons matching geometry_filter alone (no group_by_line constraint).
                if geometry_filters:
                    _d_gf_parts: list[str] = []
                    if "is_anomaly" in geometry_filters:
                        _d_val = "true" if bool(geometry_filters["is_anomaly"]) else "false"
                        _d_gf_parts.append(f"is_anomaly = {_d_val}")
                    if "delta_rank_pct" in geometry_filters:
                        _d_spec = geometry_filters["delta_rank_pct"]
                        if isinstance(_d_spec, dict):
                            for _d_op, _d_v in _d_spec.items():
                                if _d_op in _SQL_OPS:
                                    _d_gf_parts.append(
                                        f"delta_rank_pct {_SQL_OPS[_d_op]} {float(_d_v)}"
                                    )
                        else:
                            _d_gf_parts.append(f"delta_rank_pct = {float(_d_spec)}")
                    if "delta_dim" in geometry_filters:
                        for _d_dim, _d_preds in geometry_filters["delta_dim"].items():
                            _d_idx = pattern.dim_index(_d_dim)
                            for _d_op, _d_v in _d_preds.items():
                                if _d_op in _SQL_OPS:
                                    _d_gf_parts.append(
                                        f"delta_dim_{_d_idx} {_SQL_OPS[_d_op]} {float(_d_v)}"
                                    )
                    if resolved_filters:
                        for _, _d_key_set in resolved_filters:
                            if len(_d_key_set) == 1:
                                _d_gf_parts.append(
                                    f"array_contains(entity_keys, '{next(iter(_d_key_set))}')"
                                )
                            else:
                                _d_keys_str = ", ".join(f"'{k}'" for k in _d_key_set)
                                _d_gf_parts.append(f"array_has_any(entity_keys, [{_d_keys_str}])")
                    _d_geo_filter = " AND ".join(_d_gf_parts) if _d_gf_parts else None
                    _d_geo_total = reader.count_geometry_rows(
                        event_pattern_id, pattern.version, filter=_d_geo_filter,
                    )
                    _d_total = sum(computed.values())
                    _d_dropped = _d_geo_total - _d_total
                    if _d_dropped > 0:
                        result["warning"] = (
                            f"{_d_dropped} of {_d_geo_total} geometry-filtered polygons "
                            f"had no edge to "
                            f"'{group_by_line}' and were excluded from aggregation. "
                            f"Shown counts reflect only the {_d_total} polygons "
                            f"with a '{group_by_line}' edge. "
                            f"Use anomaly_summary for population-wide totals."
                        )
                return result
            except Exception:
                pass  # Fall through to read_geometry path

    # --- B: Column projection — minimal columns for read_geometry ---
    # Request edges + entity_keys; reader drops missing columns from schema.
    # Event geometry has entity_keys only (no edges). Anchor has both.
    columns: list[str] = ["primary_key", "edges", "entity_keys"]
    if geometry_filters:
        if "is_anomaly" in geometry_filters:
            columns.append("is_anomaly")
        if "delta_rank_pct" in geometry_filters:
            columns.append("delta_rank_pct")
        if "alias_inside" in geometry_filters:
            columns.append("delta")

    # --- B2: Lance scalar-index prefilter for is_anomaly / delta_rank_pct / delta_dim ---
    # Pushes predicates into Lance before materialising rows — uses scalar indices
    # built by GDSWriter. alias_inside requires full-vector ops and is handled
    # in Python after the read (see geometry pre-filter below).
    # Only effective when point_keys is None (full-scan path); ignored otherwise.
    lance_prefilter: str | None = None
    if geometry_filters and pushdown_keys is None:
        _lance_parts: list[str] = []
        if "is_anomaly" in geometry_filters:
            val = "true" if bool(geometry_filters["is_anomaly"]) else "false"
            _lance_parts.append(f"is_anomaly = {val}")
        if "delta_rank_pct" in geometry_filters:
            spec = geometry_filters["delta_rank_pct"]
            if isinstance(spec, dict):
                for op_name, value in spec.items():
                    if op_name in _SQL_OPS:
                        _lance_parts.append(f"delta_rank_pct {_SQL_OPS[op_name]} {float(value)}")
            else:
                _lance_parts.append(f"delta_rank_pct = {float(spec)}")
        if "delta_dim" in geometry_filters:
            for dim_name, predicates in geometry_filters["delta_dim"].items():
                idx = pattern.dim_index(dim_name)
                for op_name, value in predicates.items():
                    if op_name in _SQL_OPS:
                        _lance_parts.append(f"delta_dim_{idx} {_SQL_OPS[op_name]} {float(value)}")
        if _lance_parts:
            lance_prefilter = " AND ".join(_lance_parts)

    geo = reader.read_geometry(
        event_pattern_id, pattern.version,
        point_keys=pushdown_keys, columns=columns,
        filter=lance_prefilter,
    )

    # Safety check: ensure edges or entity_keys column is present.
    if "edges" not in geo.schema.names and "entity_keys" not in geo.schema.names:
        fallback_cols: list[str] = ["primary_key", "edges", "entity_keys"]
        if geometry_filters:
            if "is_anomaly" in geometry_filters:
                fallback_cols.append("is_anomaly")
            if "delta_rank_pct" in geometry_filters:
                fallback_cols.append("delta_rank_pct")
            if "alias_inside" in geometry_filters:
                fallback_cols.append("delta")
        geo = reader.read_geometry(
            event_pattern_id, pattern.version,
            point_keys=pushdown_keys,
            columns=fallback_cols,
            filter=lance_prefilter,
        )

    # --- Anomaly-rate enrichment flag ---
    _anomaly_rate_requested = bool(geometry_filters and "is_anomaly" in geometry_filters)

    # --- Geometry pre-filter ---
    if geometry_filters:
        import pyarrow.compute as pc
        supported = {"is_anomaly", "delta_rank_pct", "delta_dim", "alias_inside"}
        unknown = set(geometry_filters) - supported
        if unknown:
            raise RuntimeError(f"Unknown geometry_filters keys: {unknown}")
        if "is_anomaly" in geometry_filters:
            geo = geo.filter(
                pc.equal(geo["is_anomaly"], bool(geometry_filters["is_anomaly"]))
            )
        if "delta_rank_pct" in geometry_filters:
            spec = geometry_filters["delta_rank_pct"]
            if isinstance(spec, dict):
                geo = _apply_comparison_filter(geo, "delta_rank_pct", spec)
            else:
                geo = geo.filter(pc.equal(geo["delta_rank_pct"], float(spec)))
        # delta_dim: Lance pushdown applied above; Arrow post-filter ensures
        # correctness when reader doesn't execute Lance predicates (e.g. mocked storage).
        if "delta_dim" in geometry_filters:
            _pc_ops = {
                "gt": pc.greater, "gte": pc.greater_equal,
                "lt": pc.less, "lte": pc.less_equal, "eq": pc.equal,
            }
            for dim_name, predicates in geometry_filters["delta_dim"].items():
                idx = pattern.dim_index(dim_name)
                col_name = f"delta_dim_{idx}"
                if col_name in geo.schema.names:
                    col = geo[col_name]
                    for op_name, threshold in predicates.items():
                        if op_name not in _pc_ops:
                            raise ValueError(
                                f"Unknown comparison op '{op_name}'. "
                                f"Supported: {list(_pc_ops)}"
                            )
                        geo = geo.filter(_pc_ops[op_name](col, float(threshold)))
                        col = geo[col_name]
        if "alias_inside" in geometry_filters:
            alias_id = str(geometry_filters["alias_inside"])
            alias = s.aliases.get(alias_id)
            if alias is None:
                return {
                    "error": f"Alias '{alias_id}' not found. Available: {list(s.aliases.keys())}"
                }
            if alias.filter.cutting_plane is None:
                return {
                    "error": (
                        f"Alias '{alias_id}' has no cutting_plane. "
                        "alias_inside requires an alias with a geometric boundary (cutting_plane)."
                    )
                }
            geo = engine.filter_geometry_inside_alias(geo, alias)

    # --- filter_by_keys ---
    _fbk_pre_count: int = 0
    if filter_by_keys is not None:
        import pyarrow as pa
        import pyarrow.compute as pc
        _fbk_pre_count = geo.num_rows
        _fbk_set = pa.array(filter_by_keys, type=pa.string())
        geo = geo.filter(pc.is_in(geo["primary_key"], value_set=_fbk_set))

    # --- entity_filters ---
    _ef_matched_count: int = 0
    _ef_pre_count: int = 0
    if entity_filters is not None:
        import pyarrow as pa  # noqa: F811
        import pyarrow.compute as pc  # noqa: F811

        _ef_pre_count = geo.num_rows
        pat = s.patterns[event_pattern_id]
        if pat.pattern_type == "event":
            _ef_line_id = s.event_line(event_pattern_id)
        else:
            _ef_line_id = s.entity_line(event_pattern_id)
        if _ef_line_id is None:
            raise RuntimeError(
                f"entity_filters: cannot resolve entity line for pattern '{event_pattern_id}' "
                f"(entity_type='{pat.entity_type}', pattern_type='{pat.pattern_type}'). "
                f"Available lines: {list(s.lines.keys())}."
            )
        _ef_line = s.lines.get(_ef_line_id)
        if _ef_line is None:
            raise RuntimeError(f"entity_filters: entity line '{_ef_line_id}' not found in sphere.")
        _ef_table = reader.read_points(_ef_line_id, _ef_line.versions[-1])
        _ef_mask = _build_property_filter_mask(
            _ef_table, entity_filters, context=f"entity line '{_ef_line_id}'"
        )
        _ef_keys = _ef_table.filter(_ef_mask)["primary_key"].cast(pa.string())
        _ef_matched_count = len(_ef_keys)
        geo = geo.filter(pc.is_in(geo["primary_key"], value_set=_ef_keys))

    # --- Pre-sample before absent-edge filter to bound scan cost ---
    _met_pre_total = geo.num_rows
    _met_pre_sampled = False
    if missing_edge_to is not None and use_sampling and geo.num_rows > 0:
        import pyarrow as pa  # noqa: F811
        _met_n = (
            sample_size if sample_size is not None
            else max(1, int(geo.num_rows * (sample_pct or 1.0)))
        )
        _met_n = min(_met_n, geo.num_rows)
        if _met_n < geo.num_rows:
            _met_rng = np.random.default_rng(seed)
            _met_idx = np.sort(_met_rng.choice(geo.num_rows, size=_met_n, replace=False))
            geo = geo.take(pa.array(_met_idx, type=pa.int64()))
            _met_pre_sampled = True

    # --- Absent-edge filter ---
    if missing_edge_to is not None and geo.num_rows > 0:
        import pyarrow as pa  # noqa: F811
        import pyarrow.compute as pc  # noqa: F811
        _met_pat = sphere.patterns.get(event_pattern_id)
        _met_rels = _met_pat.relations if _met_pat else None
        _met_row_idx, _met_fl, _met_fp, _met_am = _edge_arrays(
            geo, relations=_met_rels,
        )
        _has_edge_ri, _ = _gbl_edge_arrays(
            missing_edge_to, _met_row_idx, _met_fl, _met_fp, _met_am
        )
        _all_idx = pa.array(range(geo.num_rows), type=pa.int64())
        _absent_mask = pc.invert(pc.is_in(_all_idx, value_set=_has_edge_ri))
        geo = geo.filter(_absent_mask)

    # --- Resolve event entity line (source for metrics + event_filters) ---
    event_line_id = s.event_line(event_pattern_id)
    if event_line_id is None:
        # Fallback: use pattern entity_type as line_id (legacy spheres)
        event_line_id = pattern.entity_type
    event_line = s.lines[event_line_id]
    event_version = event_line.versions[-1]

    # --- event_filters ---
    _ctx_table_filtered: Any = None
    if event_filters:
        _ef_cols: list[str] = ["primary_key"] + [
            c for c in event_filters if c != "primary_key"
        ]
        if ":" in metric and not metric.startswith("count_distinct"):
            _maybe_col = metric.split(":", 1)[1]
            _ef_schema = reader.read_points_schema(event_line_id, event_version)
            if _maybe_col in _ef_schema.names and _maybe_col not in _ef_cols:
                _ef_cols.append(_maybe_col)
        _cf_table = reader.read_points(
            event_line_id, event_version, columns=_ef_cols,
        )
        _ctx_table_filtered = _apply_event_filters(_cf_table, event_filters)
        import pyarrow as pa
        import pyarrow.compute as pc
        _cf_pk_set = pa.array(
            _ctx_table_filtered["primary_key"].to_pylist(), type=pa.string()
        )
        geo = geo.filter(pc.is_in(geo["primary_key"], value_set=_cf_pk_set))

    # Build allowed_group_keys for property_filters.
    allowed_group_keys: set[str] | None = None
    group_props_lookup: dict[str, dict] = {}
    if property_filters:
        gbl_line = s.lines.get(group_by_line)
        if gbl_line is None:
            raise RuntimeError(f"group_by_line '{group_by_line}' not found.")
        gbl_table = reader.read_points(group_by_line, gbl_line.versions[-1])
        _ctx = f"group_by_line '{group_by_line}'"
        mask = _build_property_filter_mask(gbl_table, property_filters, context=_ctx)
        filtered_bks = gbl_table.filter(mask)["primary_key"].to_pylist()
        allowed_group_keys = set(filtered_bks)

    # Build property lookup for cross-tab grouping
    prop_line_id: str | None = None
    prop_name: str | None = None
    prop_lookup: dict = {}
    if group_by_property:
        prop_line_id, prop_name = group_by_property.split(":", 1)
        prop_line = s.lines.get(prop_line_id)
        if prop_line is None:
            raise RuntimeError(f"Property line '{prop_line_id}' not found.")
        prop_table = reader.read_points(prop_line_id, prop_line.versions[-1])
        if prop_name not in prop_table.schema.names:
            raise RuntimeError(
                f"Property '{prop_name}' not found in line '{prop_line_id}'."
            )
        prop_lookup = dict(zip(
            prop_table["primary_key"].to_pylist(),
            prop_table[prop_name].to_pylist(), strict=False,
        ))

    # --- Internal timer for pivot slow-operation warning ---
    _t0_pivot = time.perf_counter()

    # Parse metric — defer context_map building (DataFusion path skips it).
    agg_func: str | None = None
    agg_col: str | None = None
    cd_target_line: str | None = None
    context_map: dict = {}
    _context_map_built = False
    _pct_n: int | None = None
    if metric == "count":
        pass
    elif ":" in metric:
        agg_op, agg_col = metric.split(":", 1)
        if agg_op == "count_distinct":
            cd_target_line = agg_col
            if cd_target_line not in s.lines:
                raise RuntimeError(
                    f"Unknown target line '{cd_target_line}' in count_distinct metric. "
                    f"Available lines: {sorted(s.lines.keys())}"
                )
            if cd_target_line == group_by_line:
                raise RuntimeError(
                    f"count_distinct target '{cd_target_line}' is the same as group_by_line — "
                    "counting distinct X per X is not meaningful."
                )
            pat = s.patterns[event_pattern_id]
            target_in_pattern = any(r.line_id == cd_target_line for r in pat.relations)
            if not target_in_pattern:
                raise RuntimeError(
                    f"Pattern '{event_pattern_id}' has no relation to line '{cd_target_line}'. "
                    f"Available relation lines: {[r.line_id for r in pat.relations]}"
                )
            agg_func = "count_distinct"
        elif agg_op == "median":
            agg_func = "median"
            _pct_n = 50
        elif agg_op.startswith("pct"):
            _pct_str = agg_op[3:]
            if not _pct_str.isdigit() or not (0 <= int(_pct_str) <= 100):
                raise RuntimeError(
                    f"Invalid percentile '{agg_op}': N must be an integer 0\u2013100 "
                    f"(e.g. pct90:price_eur, pct50:price_eur)."
                )
            _pct_n = int(_pct_str)
            agg_func = agg_op  # e.g. "pct90"
        else:
            _VALID_SCALAR_OPS = {"sum", "avg", "min", "max"}
            if agg_op not in _VALID_SCALAR_OPS:
                raise RuntimeError(
                    f"Unknown metric '{agg_op}'. Use: count, sum:<field>, avg:<field>, "
                    f"min:<field>, max:<field>, median:<field>, pct<N>:<field>, "
                    f"count_distinct:<line>."
                )
            agg_func = agg_op
        # Validate agg_col exists on event entity line (schema-only, no data read)
        if agg_func not in ("count_distinct", None):
            _ev_schema = reader.read_points_schema(event_line_id, event_version)
            if agg_col not in _ev_schema.names:
                raise ValueError(
                    f"Column '{agg_col}' not found on event entity line "
                    f"'{event_line_id}'. Metric columns must be on the "
                    f"event entity line."
                )
        # context_map is built lazily — only when the Arrow fallback path needs it.
        # DataFusion path does its own JOIN and never touches context_map.
    else:
        raise RuntimeError(
            f"Unknown metric '{metric}'. Use 'count', 'count_distinct:<line>', "
            "'sum:field', 'avg:field', 'min:field', 'max:field', "
            "'median:field', or 'pct<N>:field'."
        )

    if agg_func == "count_distinct":
        if group_by_property:
            raise RuntimeError("count_distinct metric cannot be combined with group_by_property.")
        if distinct:
            raise RuntimeError("count_distinct metric cannot be combined with distinct=True.")
        if collapse_by_property:
            raise RuntimeError(
                "count_distinct metric cannot be combined with collapse_by_property."
            )
        if pivot_event_field:
            raise RuntimeError(
                "count_distinct metric cannot be combined with pivot_event_field."
            )
        if event_filters:
            raise RuntimeError(
                "count_distinct metric does not use event_filters."
            )

    if _pct_n is not None and pivot_event_field:
        raise RuntimeError(
            "median/pct<N> metrics cannot be combined with pivot_event_field."
        )

    pivot_lookup: dict = {}

    # --- C: Vectorized count fast path ---
    has_edges = "edges" in geo.schema.names
    has_entity_keys = "entity_keys" in geo.schema.names and bool(pattern.relations)
    has_edge_data = has_edges or has_entity_keys

    use_vectorized = (
        metric == "count"
        and not agg_func
        and not pivot_event_field
        and not group_by_property
        and not group_by_line_2
        and not use_sampling
        and not distinct
        and allowed_group_keys is None
        and has_edge_data
    )

    # E block: vectorized sum/avg/min/max
    # Metrics now always come from event entity line — primary_key matches polygon key.
    _use_vectorized_agg = (
        agg_func in ("sum", "avg", "min", "max")
        and _pct_n is None
        and not pivot_event_field
        and not group_by_property
        and not group_by_line_2
        and not use_sampling
        and not distinct
        and allowed_group_keys is None
        and not resolved_filters
        and has_edge_data
    )

    # E-pct block: vectorized median/pct<N> — PyArrow+numpy for both event and anchor patterns,
    # DataFusion APPROX_PERCENTILE_CONT for anchor patterns (edges) with large row counts.
    _use_vectorized_pct = (
        _pct_n is not None
        and not pivot_event_field
        and not group_by_property
        and not group_by_line_2
        and not use_sampling
        and not distinct
        and allowed_group_keys is None
        and not resolved_filters
        and has_edge_data
    )

    # E2 block: filtered metric aggregate via Lance SQL
    _use_filtered_lance_sql = (
        agg_func in ("sum", "avg", "min", "max")
        and _pct_n is None
        and resolved_filters
        and not pivot_event_field
        and not group_by_property
        and not group_by_line_2
        and not use_sampling
        and not distinct
        and allowed_group_keys is None
        and has_edge_data
        and geo.num_rows > _SUBPROCESS_THRESHOLD
        and filter_by_keys is None
        and event_filters is None
        and entity_filters is None
    )

    # F block: vectorized pivot_event_field count (no edge filters)
    _use_vectorized_pivot = (
        metric == "count"
        and not agg_func
        and bool(pivot_event_field)
        and not group_by_property
        and not group_by_line_2
        and not use_sampling
        and not distinct
        and allowed_group_keys is None
        and not resolved_filters
        and has_edge_data
        and filter_by_keys is None
        and event_filters is None
        and entity_filters is None
    )

    # G block: vectorized group_by_property count (no edge filters)
    _use_vectorized_gbp = (
        metric == "count"
        and not agg_func
        and bool(group_by_property)
        and not pivot_event_field
        and not group_by_line_2
        and not use_sampling
        and not distinct
        and allowed_group_keys is None
        and not resolved_filters
        and has_edge_data
    )

    # G-distinct block: vectorized distinct count by property value.
    # Same conditions as G but distinct=True. Falls back to Python loop
    # when resolved_filters, sampling, or allowed_group_keys are set.
    _use_vectorized_gbp_distinct = (
        metric == "count"
        and not agg_func
        and bool(group_by_property)
        and not pivot_event_field
        and not group_by_line_2
        and not use_sampling
        and distinct
        and allowed_group_keys is None
        and not resolved_filters
        and has_edge_data
    )

    # CD block: vectorized count_distinct path
    _use_vectorized_cd = (
        agg_func == "count_distinct"
        and has_edge_data
        and not group_by_property
        and not group_by_line_2
        and allowed_group_keys is None
    )

    # ML block: multi-level GROUP BY (two anchor lines)
    _use_multi_level = group_by_line_2 is not None and has_edge_data

    if use_vectorized:
        import pyarrow as pa
        import pyarrow.compute as pc

        # Lance SQL count GROUP BY: pushed directly into Lance scanner via
        # LanceDataset.sql() with positional UNION over the entity_keys slots
        # that point at group_by_line. No in-memory geo load, no subprocess
        # fork, no external datafusion package.
        _lance_sql_count_done = False
        if (
            geo.num_rows > _SUBPROCESS_THRESHOLD
            and not resolved_filters
            and filter_by_keys is None
            and event_filters is None
            and entity_filters is None
            and geometry_filters is None
        ):
            from hypertopos.engine.lance_sql_agg import (
                aggregate_count as _lance_sql_count,
            )
            _base = reader._base.resolve()
            _geo_lance = str(
                _base / "geometry" / event_pattern_id
                / f"v={pattern.version}" / "data.lance"
            )
            computed = _lance_sql_count(_geo_lance, pattern, group_by_line)
            group_values: dict = {k: [1] * v for k, v in computed.items()}
            actually_sampled = False
            _lance_sql_count_done = True

        if not _lance_sql_count_done:
            geo_vc = geo
            if resolved_filters:
                _edge_map_fn = _make_edge_map_fn(geo, pattern.relations)
                bk_list = geo["primary_key"].to_pylist()
                pass_mask = []
                for i in range(len(bk_list)):
                    edge_map = _edge_map_fn(i)
                    edge_map[event_line_id] = bk_list[i]
                    pass_mask.append(
                        all(edge_map.get(ln) in ks for ln, ks in resolved_filters)
                    )
                geo_vc = geo.filter(pa.array(pass_mask, type=pa.bool_()))

            computed, _gbl_edge_warning = _vectorized_count_with_warning(
                geo_vc, group_by_line, geometry_filters, relations=pattern.relations
            )
            group_values: dict = {k: [1] * v for k, v in computed.items()}
            actually_sampled = False

    elif _use_vectorized_agg:
        # --- E: sum/avg/min/max — Lance SQL for large tables, in-process pyarrow handler for small / filter-bearing queries ---
        _lance_sql_metric_done = False
        if (
            geo.num_rows > _SUBPROCESS_THRESHOLD
            and filter_by_keys is None
            and event_filters is None
            and entity_filters is None
            and not group_by_line_2
        ):
            from hypertopos.engine.lance_sql_agg import (
                aggregate_metric as _lance_sql_metric,
            )
            _base = reader._base.resolve()
            _geo_lance = str(
                _base / "geometry" / event_pattern_id
                / f"v={pattern.version}" / "data.lance"
            )
            _ctx_lance = str(
                _base / "points" / event_line_id
                / f"v={event_version}" / "data.lance"
            )
            computed_metric, computed_counts = _lance_sql_metric(
                _geo_lance, _ctx_lance, pattern, group_by_line,
                agg_col, agg_func,
            )
            computed: dict = computed_metric
            group_values: dict = {
                k: range(v) for k, v in computed_counts.items()
            }
            actually_sampled = False
            _lance_sql_metric_done = True

        if not _lance_sql_metric_done:
            import pyarrow as pa
            import pyarrow.compute as pc

            if _ctx_table_filtered is not None:
                ctx_table = _ctx_table_filtered
            else:
                ctx_table = reader.read_points(
                    event_line_id, event_version,
                    columns=["primary_key", agg_col],
                )

            row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v = _edge_arrays(geo, relations=pattern.relations)
            gbl_row_idx, gbl_keys = _gbl_edge_arrays(
                group_by_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
            )

            idx = pc.index_in(geo["primary_key"], ctx_table["primary_key"])
            polygon_amounts = pc.cast(pc.take(ctx_table[agg_col], idx), pa.float64())
            edge_amounts = pc.take(polygon_amounts, gbl_row_idx)

            valid = pc.is_valid(edge_amounts)
            _agg_op_map = {"sum": "sum", "avg": "mean", "min": "min", "max": "max"}
            agg_op = _agg_op_map[agg_func]
            agg_tbl = pa.table({
                "group_key": gbl_keys.filter(valid),
                "amount": edge_amounts.filter(valid),
            })
            result_e = agg_tbl.group_by("group_key").aggregate([
                ("amount", agg_op),
                ("amount", "count"),
            ])
            keys_e = result_e["group_key"].to_pylist()
            vals_e = result_e[f"amount_{agg_op}"].to_pylist()
            cnts_e = result_e["amount_count"].to_pylist()
            computed: dict = dict(zip(keys_e, vals_e, strict=False))
            group_values: dict = {k: range(c) for k, c in zip(keys_e, cnts_e, strict=False)}
            actually_sampled = False

    elif _use_vectorized_pct:
        # --- E-pct: median/pct<N> via Lance SQL approx_percentile_cont
        # for large tables, in-process pyarrow+numpy (exact) fallback below ---
        import pyarrow as pa
        import pyarrow.compute as pc

        if _ctx_table_filtered is not None:
            ctx_table = _ctx_table_filtered
        else:
            ctx_table = reader.read_points(
                event_line_id, event_version,
                columns=["primary_key", agg_col],
            )

        _pct_done = False

        if geo.num_rows >= 500_000:
            from hypertopos.engine.lance_sql_agg import (
                aggregate_percentile as _lance_sql_pct,
            )
            _pct_frac = (
                0.5 if agg_func == "median"
                else int(agg_func[3:]) / 100.0
            )
            _base = reader._base.resolve()
            _geo_lance = str(
                _base / "geometry" / event_pattern_id
                / f"v={pattern.version}" / "data.lance"
            )
            _ctx_lance = str(
                _base / "points" / event_line_id
                / f"v={event_version}" / "data.lance"
            )
            computed, _cnt_dict = _lance_sql_pct(
                _geo_lance, _ctx_lance, pattern, group_by_line,
                agg_col, _pct_frac,
            )
            group_values = {k: range(v) for k, v in _cnt_dict.items()}
            actually_sampled = False
            _pct_done = True

        # Exact pyarrow+numpy path for small tables and as the fallback
        if not _pct_done:
            row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v = _edge_arrays(
                geo, relations=pattern.relations,
            )
            gbl_row_idx, gbl_keys = _gbl_edge_arrays(
                group_by_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v,
            )

            idx = pc.index_in(geo["primary_key"], ctx_table["primary_key"])
            polygon_amounts = pc.cast(pc.take(ctx_table[agg_col], idx), pa.float64())
            edge_amounts = pc.take(polygon_amounts, gbl_row_idx)
            valid = pc.is_valid(edge_amounts)

            agg_keys = gbl_keys.filter(valid).to_pylist()
            agg_vals = edge_amounts.filter(valid).to_pylist()
            _pct_groups: dict[str, list[float]] = defaultdict(list)
            for k, v in zip(agg_keys, agg_vals, strict=False):
                _pct_groups[k].append(v)

            computed = {}
            group_values = {}
            for k, group_list in _pct_groups.items():
                computed[k] = float(np.percentile(group_list, _pct_n))
                group_values[k] = range(len(group_list))
            actually_sampled = False

    elif _use_filtered_lance_sql:
        # --- E2: filtered metric aggregate via Lance SQL ---
        from hypertopos.engine.lance_sql_agg import (
            aggregate_filtered_metric as _lance_sql_filtered_metric,
        )
        _base = reader._base.resolve()
        _geo_lance = str(
            _base / "geometry" / event_pattern_id
            / f"v={pattern.version}" / "data.lance"
        )
        _ctx_lance = str(
            _base / "points" / event_line_id
            / f"v={event_version}" / "data.lance"
        )
        computed_metric, computed_counts = _lance_sql_filtered_metric(
            _geo_lance, _ctx_lance, pattern, group_by_line,
            agg_col, agg_func, resolved_filters, event_line_id,
        )
        computed: dict = computed_metric
        group_values: dict = {
            k: range(v) for k, v in computed_counts.items()
        }
        actually_sampled = False

    elif _use_vectorized_pivot:
        # --- F: Vectorized pivot_event_field count via Lance SQL ---
        _lance_sql_pivot_done = False
        if (
            geo.num_rows > _SUBPROCESS_THRESHOLD
            and filter_by_keys is None
            and event_filters is None
            and entity_filters is None
        ):
            from hypertopos.engine.lance_sql_agg import (
                aggregate_pivot as _lance_sql_pivot,
            )
            _base = reader._base.resolve()
            _geo_lance = str(
                _base / "geometry" / event_pattern_id
                / f"v={pattern.version}" / "data.lance"
            )
            _event_lance = str(
                _base / "points" / event_line_id
                / f"v={event_line.versions[-1]}" / "data.lance"
            )
            _ctx_lance = None
            if agg_func:
                _ctx_lance = str(
                    _base / "points" / event_line_id
                    / f"v={event_version}" / "data.lance"
                )
            results = _lance_sql_pivot(
                _geo_lance, _event_lance, pattern, group_by_line,
                pivot_event_field,
                ctx_lance_path=_ctx_lance,
                metric_col=agg_col if agg_func else None,
                agg_fn=agg_func,
            )
            computed = {}
            group_values = {}
            for row in results:
                key = (row["group_key"], row["pivot_val"])
                cnt = row["count"]
                computed[key] = row.get("metric", cnt)
                group_values[key] = range(cnt)
            actually_sampled = False
            _lance_sql_pivot_done = True

        if not _lance_sql_pivot_done:
            # Lazy build: only reached when the lance_sql_pivot path is gated off
            # (small geometry or filter-bearing query).
            if event_line is None:
                raise RuntimeError("Cannot use pivot_event_field: event line not found in sphere.")
            event_points = reader.read_points(event_line_id, event_line.versions[-1])
            if pivot_event_field not in event_points.schema.names:
                raise RuntimeError(
                    f"Field '{pivot_event_field}' not found in event line '{event_line_id}'."
                )
            pivot_lookup = {
                bk: str(pv)
                for bk, pv in zip(
                    event_points["primary_key"].to_pylist(),
                    event_points[pivot_event_field].to_pylist(), strict=False,
                )
            }
            import pyarrow as pa
            import pyarrow.compute as pc

            row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v = _edge_arrays(geo, relations=pattern.relations)
            gbl_row_idx, gbl_keys = _gbl_edge_arrays(
                group_by_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
            )

            # Per-polygon pivot values from pivot_lookup
            polygon_pivot_vals = pa.array(
                [pivot_lookup.get(k) for k in geo["primary_key"].to_pylist()]
            )

            # Fancy index: pivot val for each gbl edge
            edge_pivot_vals = pc.take(polygon_pivot_vals, gbl_row_idx)

            # Filter valid (non-null group key and pivot val) -> group by both
            valid = pc.and_(pc.is_valid(gbl_keys), pc.is_valid(edge_pivot_vals))
            pivot_agg_tbl = pa.table({
                "group_key": gbl_keys.filter(valid),
                "pivot_val": edge_pivot_vals.filter(valid),
            })
            result_f = pivot_agg_tbl.group_by(["group_key", "pivot_val"]).aggregate(
                [("group_key", "count")]
            )
            computed: dict = {
                (row["group_key"], row["pivot_val"]): row["group_key_count"]
                for row in result_f.to_pylist()
            }
            group_values: dict = {k: range(v) for k, v in computed.items()}
            actually_sampled = False

    elif (
        _use_vectorized_gbp or _use_vectorized_gbp_distinct
        or (
            metric == "count"
            and not agg_func
            and bool(group_by_property)
            and bool(resolved_filters)
            and not use_sampling
            and not distinct
            and has_edges
        )
    ):
        # --- G/G-distinct: Vectorized group_by_property count ---
        # Lance SQL fast path for large tables, in-process pyarrow handler
        # for small tables and queries with filter_by_keys / event_filters.
        _sub_done = False

        if (
            geo.num_rows > _SUBPROCESS_THRESHOLD
            and filter_by_keys is None
            and event_filters is None
            and entity_filters is None
            and not collapse_by_property
        ):
            from hypertopos.engine.lance_sql_agg import (
                aggregate_property as _lance_sql_property,
            )
            _base = reader._base.resolve()
            _geo_lance = str(
                _base / "geometry" / event_pattern_id
                / f"v={pattern.version}" / "data.lance"
            )
            _prop_lance = str(
                _base / "points" / prop_line_id
                / f"v={s.lines[prop_line_id].versions[-1]}" / "data.lance"
            )
            _ctx_lance = None
            if agg_func:
                _ctx_lance = str(
                    _base / "points" / event_line_id
                    / f"v={event_version}" / "data.lance"
                )
            sub_result = _lance_sql_property(
                _geo_lance, _prop_lance, pattern, group_by_line,
                prop_line_id, prop_name,
                distinct=distinct,
                ctx_lance_path=_ctx_lance,
                metric_col=agg_col if agg_func else None,
                agg_fn=agg_func,
            )
            if "distinct_results" in sub_result:
                computed = sub_result["distinct_results"]
                group_values = {}
            else:
                computed: dict = {}
                group_values: dict = {}
                for row in sub_result["results"]:
                    key = (row["group_key"], row["prop_val"])
                    cnt = row["count"]
                    computed[key] = row.get("metric", cnt)
                    group_values[key] = range(cnt)
            actually_sampled = False
            _sub_done = True

        if not _sub_done and _use_vectorized_gbp:
            # --- G in-process: group_by_property count ---
            import pyarrow as pa
            import pyarrow.compute as pc

            row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v = _edge_arrays(geo, relations=pattern.relations)
            gbl_row_idx, gbl_keys = _gbl_edge_arrays(
                group_by_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
            )

            if prop_line_id == group_by_line:
                prop_entity_keys_g = gbl_keys
            else:
                prop_row_idx, prop_entity_keys_raw = _gbl_edge_arrays(
                    prop_line_id, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
                )
                gbl_join = pa.table({"row_idx": gbl_row_idx, "group_key": gbl_keys})
                prop_join = pa.table({"row_idx": prop_row_idx, "prop_entity": prop_entity_keys_raw})
                joined_g = gbl_join.join(prop_join, keys="row_idx", join_type="inner")
                gbl_keys = joined_g["group_key"]
                prop_entity_keys_g = joined_g["prop_entity"]

            prop_vals_g = pa.array(
                [prop_lookup.get(k) for k in prop_entity_keys_g.to_pylist()]
            )
            valid_g = pc.and_(pc.is_valid(gbl_keys), pc.is_valid(prop_vals_g))
            gbp_agg_tbl = pa.table({
                "group_key": gbl_keys.filter(valid_g),
                "prop_val": prop_vals_g.filter(valid_g),
            })
            if collapse_by_property:
                result_g = gbp_agg_tbl.group_by(["prop_val"]).aggregate(
                    [("prop_val", "count")]
                )
                computed: dict = {
                    row["prop_val"]: row["prop_val_count"]
                    for row in result_g.to_pylist()
                }
            else:
                result_g = gbp_agg_tbl.group_by(["group_key", "prop_val"]).aggregate(
                    [("group_key", "count")]
                )
                computed: dict = {
                    (row["group_key"], row["prop_val"]): row["group_key_count"]
                    for row in result_g.to_pylist()
                }
            group_values: dict = {k: range(v) for k, v in computed.items()}
            actually_sampled = False

        elif not _sub_done and _use_vectorized_gbp_distinct:
            # --- G-distinct in-process: distinct count by property value ---
            import pyarrow as pa
            import pyarrow.compute as pc

            row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v = _edge_arrays(geo, relations=pattern.relations)
            gbl_row_idx, gbl_keys = _gbl_edge_arrays(
                group_by_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
            )

            if prop_line_id == group_by_line:
                prop_entity_keys_gd = gbl_keys
            else:
                prop_row_idx, prop_entity_keys_raw = _gbl_edge_arrays(
                    prop_line_id, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
                )
                gbl_join = pa.table({"row_idx": gbl_row_idx, "group_key": gbl_keys})
                prop_join = pa.table({"row_idx": prop_row_idx, "prop_entity": prop_entity_keys_raw})
                joined_gd = gbl_join.join(prop_join, keys="row_idx", join_type="inner")
                gbl_keys = joined_gd["group_key"]
                prop_entity_keys_gd = joined_gd["prop_entity"]

            prop_vals_gd = pa.array(
                [prop_lookup.get(k) for k in prop_entity_keys_gd.to_pylist()]
            )
            valid_gd = pc.and_(pc.is_valid(gbl_keys), pc.is_valid(prop_vals_gd))
            distinct_tbl = pa.table({
                "group_key": gbl_keys.filter(valid_gd),
                "prop_val": prop_vals_gd.filter(valid_gd),
            })
            result_gd = distinct_tbl.group_by("prop_val").aggregate(
                [("group_key", "count_distinct")]
            )
            computed = {
                row["prop_val"]: row["group_key_count_distinct"]
                for row in result_gd.to_pylist()
            }
            group_values = {}
            actually_sampled = False

    # --- H: Vectorized sample count fast path ---
    _use_vectorized_sample = (
        metric == "count"
        and not agg_func
        and use_sampling
        and not resolved_filters
        and not group_by_property
        and not pivot_event_field
        and not distinct
        and allowed_group_keys is None
        and has_edge_data
    )

    # --- CD block: count_distinct vectorized path ---
    if _use_vectorized_cd:
        import pyarrow as pa  # noqa: F811
        import pyarrow.compute as pc  # noqa: F811

        row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v = _edge_arrays(geo, relations=pattern.relations)

        # Group-by axis
        gbl_row_idx, gbl_keys = _gbl_edge_arrays(
            group_by_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
        )
        # Target axis
        tgt_row_idx, tgt_keys = _gbl_edge_arrays(
            cd_target_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
        )

        # Join on row_idx: pair (group_key, target_key) for each polygon
        tgt_by_row: dict[int, list[str]] = {}
        for ri, tk in zip(tgt_row_idx.to_pylist(), tgt_keys.to_pylist(), strict=False):
            tgt_by_row.setdefault(ri, []).append(tk)

        pairs_group: list[str] = []
        pairs_target: list[str] = []
        for ri, gk in zip(gbl_row_idx.to_pylist(), gbl_keys.to_pylist(), strict=False):
            for tk in tgt_by_row.get(ri, []):
                pairs_group.append(gk)
                pairs_target.append(tk)

        if pairs_group:
            cd_table = pa.table({
                "group_key": pa.array(pairs_group, type=pa.string()),
                "tgt_key": pa.array(pairs_target, type=pa.string()),
            })
            cd_agg = cd_table.group_by("group_key").aggregate(
                [("tgt_key", "count_distinct")]
            )
            computed = {
                row["group_key"]: row["tgt_key_count_distinct"]
                for row in cd_agg.to_pylist()
            }
            group_values = {k: range(v) for k, v in computed.items()}
        else:
            computed = {}
            group_values = {}
        actually_sampled = False

    # --- ML: Multi-level GROUP BY (two anchor lines) ---
    _ml_done = False
    if _use_multi_level and not (
        use_vectorized or _use_vectorized_agg or _use_vectorized_pivot
        or _use_vectorized_gbp or _use_vectorized_gbp_distinct
        or _use_vectorized_cd
    ):
        # Multi-level GROUP BY runs in-process pyarrow only — there is no
        # Lance SQL fast path for two-level grouping in 0.3.0.
        import pyarrow as pa
        import pyarrow.compute as pc

        row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v = _edge_arrays(geo, relations=pattern.relations)
        gbl_row_idx, gbl_keys = _gbl_edge_arrays(
            group_by_line, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
        )
        gbl2_row_idx, gbl2_keys = _gbl_edge_arrays(
            group_by_line_2, row_idx, flat_line_ids_v, flat_pt_keys_v, alive_mask_v
        )

        # Join on row_idx to pair the two group keys per polygon
        gbl_join = pa.table({"row_idx": gbl_row_idx, "key_1": gbl_keys})
        gbl2_join = pa.table({"row_idx": gbl2_row_idx, "key_2": gbl2_keys})
        joined = gbl_join.join(gbl2_join, keys="row_idx", join_type="inner")

        if agg_func and agg_func in ("sum", "avg", "min", "max"):
            # Need metric values from event entity line
            if _ctx_table_filtered is not None:
                ctx_table = _ctx_table_filtered
            else:
                ctx_table = reader.read_points(
                    event_line_id, event_version,
                    columns=["primary_key", agg_col],
                )

            idx = pc.index_in(geo["primary_key"], ctx_table["primary_key"])
            polygon_amounts = pc.cast(
                pc.take(ctx_table[agg_col], idx), pa.float64()
            )
            edge_amounts = pc.take(polygon_amounts, joined["row_idx"])

            valid = pc.is_valid(edge_amounts)
            _agg_op_map = {
                "sum": "sum", "avg": "mean", "min": "min", "max": "max",
            }
            agg_op = _agg_op_map[agg_func]
            agg_tbl = pa.table({
                "key_1": joined["key_1"].filter(valid),
                "key_2": joined["key_2"].filter(valid),
                "amount": edge_amounts.filter(valid),
            })
            result_ml = agg_tbl.group_by(["key_1", "key_2"]).aggregate([
                ("amount", agg_op),
                ("amount", "count"),
            ])
            computed = {
                (row["key_1"], row["key_2"]): row[f"amount_{agg_op}"]
                for row in result_ml.to_pylist()
            }
            group_values = {
                (row["key_1"], row["key_2"]): range(row["amount_count"])
                for row in result_ml.to_pylist()
            }
        else:
            # Count metric (no context needed)
            count_tbl = pa.table({
                "key_1": joined["key_1"],
                "key_2": joined["key_2"],
            })
            result_ml = count_tbl.group_by(["key_1", "key_2"]).aggregate([
                ("key_1", "count"),
            ])
            computed = {
                (row["key_1"], row["key_2"]): row["key_1_count"]
                for row in result_ml.to_pylist()
            }
            group_values = {k: [1] * v for k, v in computed.items()}

        actually_sampled = False
        _ml_done = True

    # Guard: skip sample/fallback if a vectorized path already computed results.
    # The lance_sql_agg helpers in the elif branches above always set computed
    # when their gating predicate is True (no try/except, no internal failure
    # mode), so each branch's gating boolean is sufficient — no need to thread
    # an internal "_sub_done" through.
    _agg_done = (
        use_vectorized or _use_vectorized_agg or _use_vectorized_pct
        or _use_vectorized_pivot
        or _use_vectorized_gbp or _use_vectorized_gbp_distinct
        or _use_filtered_lance_sql
        or _use_vectorized_cd
        or _ml_done
    )

    if _agg_done:
        pass
    elif _use_vectorized_sample:
        computed, _gbl_edge_warning = _vectorized_sample_count(
            geo,
            group_by_line=group_by_line,
            sample_size=sample_size,
            sample_pct=sample_pct,
            seed=seed,
            geometry_filters=geometry_filters,
            relations=pattern.relations,
        )
        total_eligible = geo.num_rows
        _n_sampled = (
            sample_size if sample_size is not None
            else max(1, int(total_eligible * (sample_pct or 1.0)))
        )
        n = min(_n_sampled, total_eligible)
        group_values = {k: [1] * v for k, v in computed.items()}
        actually_sampled = n < total_eligible

    else:
        # --- Python loop fallback (complex cases) ---
        # Lazy context_map build — deferred from metric parsing.
        if agg_func and agg_func != "count_distinct" and not _context_map_built:
            if _ctx_table_filtered is not None:
                ctx_table = _ctx_table_filtered
            else:
                ctx_table = reader.read_points(
                    event_line_id, event_version,
                    columns=["primary_key", agg_col],
                )
            ctx_keys = ctx_table["primary_key"].to_pylist()
            ctx_vals = ctx_table[agg_col].to_pylist()
            context_map = dict(zip(ctx_keys, ctx_vals, strict=False))
            _context_map_built = True
        # For simple sample+no-filter case, pre-slice geo to avoid O(n) .to_pylist().
        # This avoids the 37s edges.to_pylist() bottleneck on 1M-row legacy geometry tables.
        _loop_pre_total = geo.num_rows
        _loop_pre_sampled = False
        if (
            use_sampling
            and not resolved_filters
            and not pivot_event_field
            and not distinct
        ):
            _n_ps = (
                sample_size if sample_size is not None
                else max(1, int(_loop_pre_total * (sample_pct or 1.0)))
            )
            _n_ps = min(_n_ps, _loop_pre_total)
            _rng_ps = np.random.default_rng(seed)
            _idx_ps = np.sort(_rng_ps.choice(_loop_pre_total, size=_n_ps, replace=False))
            import pyarrow as _pa_pre
            geo = geo.take(_pa_pre.array(_idx_ps, type=_pa_pre.int64()))
            _loop_pre_sampled = _n_ps < _loop_pre_total

        bk_col = geo["primary_key"].to_pylist()
        _make_edge_map = _make_edge_map_fn(geo, pattern.relations)
        group_values = defaultdict(list)
        distinct_sets: dict = defaultdict(set)
        cd_sets: dict = defaultdict(set)
        _cd_tgt_by_row: dict[int, list[str]] = defaultdict(list)
        if agg_func == "count_distinct":
            if has_edges:
                _cd_row, _cd_fl, _cd_fp, _cd_am = _edge_arrays(geo, relations=pattern.relations)
                _cd_tgt_row, _cd_tgt_keys = _gbl_edge_arrays(
                    cd_target_line, _cd_row, _cd_fl, _cd_fp, _cd_am
                )
                for ri, tk in zip(_cd_tgt_row.to_pylist(), _cd_tgt_keys.to_pylist(), strict=False):
                    _cd_tgt_by_row[ri].append(tk)
            elif "edges" in geo.schema.names:
                edges_col = geo["edges"]
                for i in range(geo.num_rows):
                    for e in edges_col[i].as_py():
                        if (e["line_id"] == cd_target_line
                                and e.get("status", "alive") == "alive"):
                            _cd_tgt_by_row[i].append(e["point_key"])
            elif "entity_keys" in geo.schema.names and pattern.relations:
                _cd_ek = geo["entity_keys"].to_pylist()
                for i, keys in enumerate(_cd_ek):
                    for j, rel in enumerate(pattern.relations):
                        if rel.line_id == cd_target_line and j < len(keys or []) and keys[j]:
                            _cd_tgt_by_row[i].append(keys[j])

        # Legacy edges-struct datafusion fast path for filtered + pivot + metric
        # was retired in 0.3.0 — the in-process pyarrow path below handles
        # both schemas.

        if pivot_event_field and not pivot_lookup and bk_col:
            if event_line is None:
                raise RuntimeError(
                    "Cannot use pivot_event_field: event line not found in sphere."
                )
            _ep = reader.read_points(event_line_id, event_line.versions[-1])
            if pivot_event_field not in _ep.schema.names:
                raise RuntimeError(
                    f"Field '{pivot_event_field}' not found in event line '{event_line_id}'."
                )
            pivot_lookup = {
                bk: str(pv)
                for bk, pv in zip(
                    _ep["primary_key"].to_pylist(),
                    _ep[pivot_event_field].to_pylist(), strict=False,
                )
            }

        def _passes_filters(edge_map: dict) -> bool:
            return all(edge_map.get(line) in keys for line, keys in resolved_filters)

        if use_sampling and resolved_filters:
            eligible_indices = []
            for i in range(len(bk_col)):
                edge_map = _make_edge_map(i)
                edge_map[event_line_id] = bk_col[i]
                if _passes_filters(edge_map):
                    eligible_indices.append(i)
        else:
            eligible_indices = list(range(len(bk_col)))

        total_eligible = len(eligible_indices)

        # When pre-sampling already reduced geo, don't double-sample.
        if _met_pre_sampled and not _loop_pre_sampled:
            n = total_eligible
            actually_sampled = True
            sampled_indices = set(eligible_indices)
            total_eligible = _met_pre_total
        elif _loop_pre_sampled:
            n = total_eligible
            actually_sampled = True
            sampled_indices: set[int] = set(eligible_indices)
            total_eligible = _loop_pre_total
        elif sample_size is not None:
            n = min(sample_size, total_eligible)
            actually_sampled = n < total_eligible
            if actually_sampled:
                rng = np.random.default_rng(seed)
                chosen = rng.choice(eligible_indices, size=n, replace=False)
                sampled_indices = set(chosen.tolist())
            else:
                sampled_indices = set(eligible_indices)
        elif sample_pct is not None:
            n = min(int(total_eligible * sample_pct), total_eligible)
            actually_sampled = n < total_eligible
            if actually_sampled:
                rng = np.random.default_rng(seed)
                chosen = rng.choice(eligible_indices, size=n, replace=False)
                sampled_indices = set(chosen.tolist())
            else:
                sampled_indices = set(eligible_indices)
        else:
            n = total_eligible
            actually_sampled = False
            sampled_indices = set(eligible_indices)

        for i in (sampled_indices if use_sampling else range(len(bk_col))):
            polygon_bk = bk_col[i]
            edge_map = _make_edge_map(i)
            edge_map[event_line_id] = polygon_bk

            if not use_sampling and resolved_filters and not _passes_filters(edge_map):
                continue

            group_key = edge_map.get(group_by_line)
            if group_key is None:
                continue

            if allowed_group_keys is not None and group_key not in allowed_group_keys:
                continue

            if pivot_event_field:
                pv = pivot_lookup.get(polygon_bk)
                if pv is None:
                    continue
                effective_key = (group_key, pv)
            elif prop_line_id:
                prop_entity = edge_map.get(prop_line_id)
                prop_val = prop_lookup.get(prop_entity)
                if prop_val is None:
                    continue
                effective_key = prop_val if (distinct or collapse_by_property) else (group_key, prop_val)  # noqa: E501
            elif group_by_line_2:
                key_2 = edge_map.get(group_by_line_2)
                if key_2 is None:
                    continue
                effective_key = (group_key, key_2)
            else:
                effective_key = group_key

            if distinct and metric == "count":
                distinct_sets[effective_key].add(group_key)
            elif agg_func == "count_distinct":
                for tk in _cd_tgt_by_row.get(i, []):
                    cd_sets[effective_key].add(tk)
            elif agg_func:
                val = context_map.get(polygon_bk)
                if val is not None:
                    group_values[effective_key].append(float(val))
            else:
                group_values[effective_key].append(1)

        # Compute metric value per group
        computed = {}
        if distinct and metric == "count":
            for key, entity_set in distinct_sets.items():
                computed[key] = len(entity_set)
        elif agg_func == "count_distinct":
            for key, tgt_set in cd_sets.items():
                computed[key] = len(tgt_set)
                group_values[key] = range(len(tgt_set))
        else:
            for key, vals in group_values.items():
                if agg_func == "sum":
                    computed[key] = sum(vals)
                elif agg_func == "avg":
                    computed[key] = sum(vals) / len(vals)
                elif agg_func == "min":
                    computed[key] = min(vals)
                elif agg_func == "max":
                    computed[key] = max(vals)
                elif agg_func == "median":
                    computed[key] = float(np.median(vals))
                elif agg_func is not None and agg_func.startswith("pct"):
                    computed[key] = float(np.percentile(vals, _pct_n))
                else:
                    computed[key] = len(vals)

    # --- Post-aggregation safety: enforce allowed_group_keys on ALL paths ---
    # Fast paths (E, E2, F, G, CD, etc.) check allowed_group_keys in their entry
    # conditions, but this safety net ensures correctness even if a fast path is
    # reached through unexpected conditions (e.g., future refactors).
    if allowed_group_keys is not None:
        def _key_matches(k: Any) -> bool:
            if isinstance(k, tuple):
                return k[0] in allowed_group_keys
            return k in allowed_group_keys
        computed = {k: v for k, v in computed.items() if _key_matches(k)}
        group_values = {k: v for k, v in group_values.items() if _key_matches(k)}

    # --- Post-aggregation HAVING filter ---
    _total_groups_pre_having = len(computed)
    _having_matched: int | None = None
    if having is not None:
        _h_gt = having.get("gt", float("-inf"))
        _h_gte = having.get("gte", float("-inf"))
        _h_lt = having.get("lt", float("inf"))
        _h_lte = having.get("lte", float("inf"))
        computed = {
            k: v for k, v in computed.items()
            if v > _h_gt and v >= _h_gte and v < _h_lt and v <= _h_lte
        }
        group_values = {k: v for k, v in group_values.items() if k in computed}
        _having_matched = len(computed)

    ascending = sort == "asc"

    # Pre-sort to extract top-N keys.
    _sort_key = lambda x: x[1]  # noqa: E731
    _rev = not ascending
    if pivot_event_field:
        _piv_grouped: dict = defaultdict(dict)
        for (gk, pv), val in computed.items():
            _piv_grouped[gk][pv] = val
        _piv_totals = {gk: sum(pvs.values()) for gk, pvs in _piv_grouped.items()}
        _piv_sorted: list = sorted(
            _piv_totals.items(), key=_sort_key, reverse=_rev,
        )[offset:offset + limit]
        _pre_top: list | None = None
    elif distinct:
        _pre_top = None  # computed fresh in result block
    elif collapse_by_property:
        _pre_top = None         # sorted fresh in result block
    elif group_by_property:
        _pre_top = sorted(computed.items(), key=_sort_key, reverse=_rev)[offset:offset + limit]
    else:
        _pre_top = sorted(computed.items(), key=_sort_key, reverse=_rev)[offset:offset + limit]

    if pivot_event_field:
        # Reuse pre-sorted structures computed above (no second sort needed).
        grouped = _piv_grouped
        sorted_groups = _piv_sorted
        all_pivot_vals = sorted({pv for pvs in grouped.values() for pv in pvs})
        results = []
        for gk, _ in sorted_groups:
            row: dict = {"key": gk}
            for pv in all_pivot_vals:
                val = grouped[gk].get(pv)
                row[pv] = round(val, 2) if val is not None else None
            results.append(row)
        result = {
            "event_pattern_id": event_pattern_id,
            "group_by_line": group_by_line,
            "pivot_event_field": pivot_event_field,
            "metric": metric,
            "filters": filters or None,
            "geometry_filters": geometry_filters,
            "total_groups": _total_groups_pre_having,
            "offset": offset,
            "results": results,
        }
    else:
        # distinct has no pre_top (property values, not entity keys) — sort fresh.
        # All other cases reuse the pre-sorted slice from the enrichment step above.
        top = (
            sorted(computed.items(), key=_sort_key, reverse=_rev)[offset:offset + limit]
            if (distinct or collapse_by_property)
            else _pre_top
        )
        if distinct:
            results = [
                {
                    prop_name: pv,
                    "value": round(val, 2),
                    "count": int(val) if metric == "count" else len(group_values.get(pv, [])),
                }
                for pv, val in top
            ]
        elif collapse_by_property:
            results = [
                {
                    prop_name: pv,
                    "value": round(val, 2),
                    "count": (
                        len(group_values.get(pv, []))
                        if isinstance(group_values.get(pv), (list, range))
                        else int(val)
                    ),
                }
                for pv, val in top
            ]
        elif group_by_property:
            results = [
                {
                    "key": gk, prop_name: pv,
                    "value": round(val, 2),
                    "count": len(group_values[(gk, pv)]),
                }
                for (gk, pv), val in top
            ]
        elif group_by_line_2:
            results = [
                {
                    "key": k1, "key_2": k2,
                    "value": round(val, 2),
                    "count": len(group_values[(k1, k2)]),
                }
                for (k1, k2), val in top
            ]
        else:
            results = [
                {"key": key, "value": round(val, 2), "count": len(group_values[key])}
                for key, val in top
            ]
        result = {
            "event_pattern_id": event_pattern_id,
            "group_by_line": group_by_line,
            "group_by_property": group_by_property,
            "metric": metric,
            "distinct": distinct,
            "collapse_by_property": collapse_by_property,
            "filters": filters or None,
            "geometry_filters": geometry_filters,
            "total_groups": _total_groups_pre_having,
            "offset": offset,
            "results": results,
        }
    if group_by_line_2:
        result["group_by_line_2"] = group_by_line_2
    # When group_by_property is active (not distinct, not collapse_by_property), total_groups
    # counts (entity, property_value) pairs. Add total_entities so agents are not misled.
    # collapse_by_property: keys are plain strings; k[0] would index the string, not entity key.
    if group_by_property and not distinct and not collapse_by_property:
        distinct_entity_keys = {k[0] for k in computed}
        result["total_entities"] = len(distinct_entity_keys)

    # DATA-1: warn when geometry_filters silently drops polygons without GBL edge
    _gbl_edge_warning_final: str | None = locals().get("_gbl_edge_warning")
    if _gbl_edge_warning_final:
        result["warning"] = _gbl_edge_warning_final
    elif result.get("total_groups") == 0 and geometry_filters:
        if "delta_rank_pct" in geometry_filters:
            result["warning"] = (
                "delta_rank_pct filter matched 0 polygons. "
                "In binary-mode or discretized patterns, percentile values cluster "
                "at a few levels — many entities share identical delta vectors. "
                "Use anomaly_summary to inspect the actual percentile distribution. "
                "Suggested alternatives: geometry_filters={'is_anomaly': True} "
                "to scope to anomalous entities, or remove the geometry_filter "
                "and use having={'gt': <threshold>} to filter by metric value instead."
            )
        elif "is_anomaly" in geometry_filters:
            result["warning"] = (
                f"0 matching polygons have edges to the group_by_line "
                f"'{group_by_line}'. Anomalous polygons often lack edges to "
                f"optional relation lines — use a required relation as group_by_line."
            )
        else:
            result["warning"] = (
                f"0 matching polygons (after geometry_filters) have edges to "
                f"the group_by_line '{group_by_line}'."
            )

    if filter_by_keys is not None:
        result["filter_by_keys"] = filter_by_keys
        if result.get("total_groups") == 0 and _fbk_pre_count > 0 and "warning" not in result:
            result["warning"] = (
                "filter_by_keys matched 0 event polygons. "
                f"Ensure the keys belong to '{event_pattern_id}' event polygons "
                "(e.g. TX-* or SALE-* keys). Passing anchor entity keys "
                "(e.g. CUST-*, PROD-*) silently returns empty results."
            )
    if entity_filters is not None:
        result["entity_filters"] = entity_filters
        result["entity_filtered_count"] = _ef_matched_count
        if result.get("total_groups") == 0 and _ef_pre_count > 0 and "warning" not in result:
            result["warning"] = (
                f"entity_filters matched {_ef_matched_count} entities but 0 polygons contributed. "
                f"Check that property values exist in entity line — use get_line_schema('{_ef_line_id}') "  # noqa: E501
                "to verify available columns and exact values."
            )
    if event_filters:
        result["event_filters"] = event_filters
    if missing_edge_to is not None:
        result["missing_edge_to"] = missing_edge_to
    if having is not None:
        result["having"] = having
        result["having_matched"] = _having_matched

    if actually_sampled:
        result["sampled"] = True
        result["sample_size"] = n
        result["total_eligible"] = total_eligible
    else:
        result["sampled"] = False

    # --- Anomaly-rate enrichment ---
    # When geometry_filters includes is_anomaly, do a full (unsampled) second
    # read to get per-group totals using the same edge-counting logic as the
    # primary aggregate.
    if _anomaly_rate_requested and result.get("results"):
        _ar_cols = ["entity_keys"] if "entity_keys" in geo.schema.names else ["edges"]
        _ar_geo = reader.read_geometry(
            event_pattern_id, pattern.version,
            columns=_ar_cols,
        )
        _ar_totals, _ = _vectorized_count_with_warning(
            _ar_geo, group_by_line, geometry_filters=None, relations=pattern.relations,
        )
        for row in result.get("results", []):
            key = row.get("key")
            if key is not None and key in _ar_totals:
                total = _ar_totals[key]
                row["total_events"] = total
                row["anomaly_rate"] = round(row["count"] / total, 4) if total > 0 else 0.0

    return result
