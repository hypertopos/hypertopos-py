# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Lance-native SQL aggregate engine.

Pushes GROUP BY aggregations directly into the Lance scanner via
``LanceDataset.sql(...)``. Operates on the modern ``entity_keys: list<string>``
event-pattern geometry schema, where ``entity_keys[i]`` is the entity key for
the ``i``-th declared relation. The SQL builder pulls the positional slots
that correspond to ``group_by_line`` and unions them, matching the historical
edge-flatten semantics.

The geometry table is never loaded into Python memory — Lance streams the
required columns from disk through its built-in DataFusion executor and
returns a small post-aggregate result.
"""

from __future__ import annotations

import re

import lance as _lance
import pyarrow as pa
import pyarrow.compute as pc

_AGG_OP_MAP = {
    "sum": "sum",
    "avg": "mean",
    "min": "min",
    "max": "max",
}


# A column or line identifier that is safe to inline into a Lance SQL query
# without quoting. We require it to match this pattern to defend against
# identifier injection through user-controlled metric / pivot / prop column
# names. Lance's SqlQueryBuilder does not expose parameter binding for either
# values or identifiers, so the only line of defense is up-front validation.
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _escape_sql_string(value: str) -> str:
    """Escape a single-quoted SQL literal for inlining into a Lance SQL query.

    Lance's ``LanceDataset.sql(...)`` does not expose parameter binding, so
    user-controlled key values must be inlined into the SQL string. This
    helper:

    - rejects values containing NUL or other ASCII control characters
      (``\\x00``-``\\x1f``, ``\\x7f``) — these have no legitimate place in
      an entity primary key and are common SQL injection / parser-confusion
      vectors;
    - rejects values containing backslashes — DataFusion treats them as
      escape characters in some string contexts and the threat model is
      "no backslash in primary keys" which holds for every entity
      identifier we have ever seen in a real sphere;
    - doubles single quotes (the standard SQL literal escape).

    Raising up front means a malformed key crashes the query loudly instead
    of silently producing wrong results or executing injected SQL.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"_escape_sql_string: expected str, got {type(value).__name__}"
        )
    if "\\" in value:
        raise ValueError(
            f"_escape_sql_string: backslash not allowed in key: {value!r}"
        )
    for ch in value:
        cp = ord(ch)
        if cp < 0x20 or cp == 0x7f:
            raise ValueError(
                f"_escape_sql_string: control character U+{cp:04X} not allowed "
                f"in key: {value!r}"
            )
    return value.replace("'", "''")


def _validate_sql_identifier(value: str) -> str:
    """Return *value* iff it is a safe SQL identifier, otherwise raise.

    Used for column names (``metric_col``, ``pivot_field``, ``prop_name``)
    that are inlined directly into Lance SQL strings as identifiers (no
    surrounding quotes). The conservative pattern matches Arrow / Lance
    column-name conventions: ASCII letter or underscore start, then letters,
    digits, underscores. Anything outside that pattern raises so that an
    injection attempt — or an actual unusual column name — fails the query
    loudly instead of being passed through to the SQL parser.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"_validate_sql_identifier: expected str, got {type(value).__name__}"
        )
    if not _SQL_IDENT_RE.match(value):
        raise ValueError(
            f"_validate_sql_identifier: {value!r} is not a safe SQL identifier "
            f"(must match {_SQL_IDENT_RE.pattern})"
        )
    return value


def _line_positions(pattern: object, line_id: str) -> list[int]:
    """Return 1-based positions of *line_id* in the pattern's relations.

    Lance SQL list indexing is 1-based.
    """
    positions: list[int] = []
    for i, rel in enumerate(pattern.relations):  # type: ignore[attr-defined]
        if rel.line_id == line_id:
            positions.append(i + 1)
    return positions


def aggregate_count(
    geo_lance_path: str,
    pattern: object,
    group_by_line: str,
) -> dict[str, int]:
    """COUNT(*) GROUP BY group_by_line via Lance SQL on entity_keys.

    For every relation slot in the pattern that points at ``group_by_line``,
    the query reads ``entity_keys[slot]`` and unions the per-slot rows. The
    final GROUP BY counts how many times each entity key shows up across all
    slots — matches the legacy edge-flatten semantics one-to-one.

    Parameters
    ----------
    geo_lance_path
        Filesystem path to the event pattern's geometry Lance dataset
        (``geometry/{pattern_id}/v={version}/data.lance``).
    pattern
        The Pattern model object — read for its ``relations`` list to resolve
        which positional slots in ``entity_keys`` belong to ``group_by_line``.
    group_by_line
        Line ID to group by (e.g. ``"accounts"``).

    Returns
    -------
    dict
        ``{group_key: count}``. Empty dict when ``group_by_line`` is not a
        declared relation of the pattern, or when the dataset is empty.
    """
    positions = _line_positions(pattern, group_by_line)
    if not positions:
        return {}

    ds = _lance.dataset(geo_lance_path)
    union_parts = " UNION ALL ".join(
        f"SELECT entity_keys[{p}] AS k FROM dataset" for p in positions
    )
    sql = (
        f"SELECT k AS group_key, COUNT(*) AS value "
        f"FROM ({union_parts}) "
        f"WHERE k IS NOT NULL "
        f"GROUP BY k"
    )
    return _batches_to_int_dict(ds.sql(sql).build().to_batch_records())


def _batches_to_int_dict(batches: list) -> dict[str, int]:
    """Collapse a list of (group_key, value) record batches into a dict."""
    result: dict[str, int] = {}
    for batch in batches:
        gk_col = batch["group_key"]
        val_col = batch["value"]
        for i in range(batch.num_rows):
            gk = gk_col[i].as_py()
            if gk is None:
                continue
            result[gk] = int(val_col[i].as_py())
    return result


def _read_geo_edge_pairs(
    geo_lance_path: str,
    pattern: object,
    group_by_line: str,
) -> pa.Table:
    """Stream (event_polygon_pk, group_key) pairs from the event geometry.

    Emits one row per (polygon, position) where the position belongs to a
    relation pointing at *group_by_line* — exactly mirrors the historical
    edge-flatten semantics for the entity_keys schema. Returns an Arrow table
    with columns ``primary_key`` (event polygon key) and ``k`` (entity key).
    Empty table when ``group_by_line`` has no matching slot in the pattern.
    """
    positions = _line_positions(pattern, group_by_line)
    if not positions:
        return pa.table({
            "primary_key": pa.array([], type=pa.string()),
            "k": pa.array([], type=pa.string()),
        })
    ds = _lance.dataset(geo_lance_path)
    union_parts = " UNION ALL ".join(
        f"SELECT primary_key, entity_keys[{p}] AS k FROM dataset"
        for p in positions
    )
    batches = ds.sql(union_parts).build().to_batch_records()
    if not batches:
        return pa.table({
            "primary_key": pa.array([], type=pa.string()),
            "k": pa.array([], type=pa.string()),
        })
    return pa.Table.from_batches(batches)


def _read_metric_pairs(
    ctx_lance_path: str,
    metric_col: str,
) -> pa.Table:
    """Stream ``(primary_key, <metric_col>)`` pairs from the event line points
    Lance dataset. Cast to float64 so downstream aggregates are numerically
    stable across int / float source columns.

    Validates ``metric_col`` is a safe SQL identifier — pyarrow's scanner
    raises KeyError on unknown columns, but the validation moves the error
    upstream and prevents injection through any future code path that
    inlines ``metric_col`` into a SQL string.
    """
    metric_col = _validate_sql_identifier(metric_col)
    ds = _lance.dataset(ctx_lance_path)
    batches = ds.scanner(columns=["primary_key", metric_col]).to_table()
    if batches.num_rows == 0:
        return pa.table({
            "primary_key": pa.array([], type=pa.string()),
            metric_col: pa.array([], type=pa.float64()),
        })
    return pa.table({
        "primary_key": batches["primary_key"],
        metric_col: pc.cast(batches[metric_col], pa.float64()),
    })


def _build_filter_predicates(
    pattern: object,
    resolved_filters: list[tuple[str, set[str]]],
    event_line_id: str,
) -> list[str]:
    """Translate (line_id, key_set) filter pairs into SQL predicates against
    the entity_keys positions of the matching relations.

    A filter on the event line itself constrains ``primary_key``. Filters on
    relation lines constrain at least one of the positional ``entity_keys``
    slots whose relation points at that line — encoded as an OR over the
    matching positions, so polygons with the matching key in any of the
    line's slots pass.

    Returns a list of SQL fragments meant to be AND-joined into a WHERE clause.
    """
    predicates: list[str] = []
    for line, key_set in resolved_filters:
        if not key_set:
            return ["1=0"]
        keys_in = ", ".join(
            f"'{_escape_sql_string(k)}'" for k in sorted(key_set)
        )
        if line == event_line_id:
            predicates.append(f"primary_key IN ({keys_in})")
            continue
        positions = _line_positions(pattern, line)
        if not positions:
            return ["1=0"]
        per_pos = " OR ".join(
            f"entity_keys[{p}] IN ({keys_in})" for p in positions
        )
        predicates.append(f"({per_pos})")
    return predicates


def _read_filtered_geo_edge_pairs(
    geo_lance_path: str,
    pattern: object,
    group_by_line: str,
    resolved_filters: list[tuple[str, set[str]]],
    event_line_id: str,
) -> pa.Table:
    """Same as :func:`_read_geo_edge_pairs` but pre-filters polygons by
    ``resolved_filters`` directly inside the Lance SQL ``WHERE`` clause.
    """
    positions = _line_positions(pattern, group_by_line)
    if not positions:
        return pa.table({
            "primary_key": pa.array([], type=pa.string()),
            "k": pa.array([], type=pa.string()),
        })
    predicates = _build_filter_predicates(
        pattern, resolved_filters, event_line_id,
    )
    where = " AND ".join(predicates) if predicates else None
    where_clause = f" WHERE {where}" if where else ""
    union_parts = " UNION ALL ".join(
        f"SELECT primary_key, entity_keys[{p}] AS k FROM dataset{where_clause}"
        for p in positions
    )
    ds = _lance.dataset(geo_lance_path)
    batches = ds.sql(union_parts).build().to_batch_records()
    if not batches:
        return pa.table({
            "primary_key": pa.array([], type=pa.string()),
            "k": pa.array([], type=pa.string()),
        })
    return pa.Table.from_batches(batches)


def find_anomalies(
    geo_lance_path: str,
    threshold: float,
    top_n: int,
    offset: int = 0,
    lance_version: int | None = None,
) -> dict:
    """Top-N polygons by ``delta_norm`` above ``threshold`` via Lance SQL.

    Returns ``{"keys": list[str], "delta_norms": list[float], "total_found": int}``,
    deduplicated on ``primary_key`` (keeping the highest ``delta_norm`` per key).
    ``total_found`` is the pre-dedup row count from the Lance scanner — the
    actual number of unique keys may be lower when duplicate primary_keys exist
    in the geometry table. Callers paging with ``offset`` should treat
    ``total_found`` as an upper bound, not an exact page count.
    Mirrors the historical subprocess find_anomalies output.
    """
    import math
    if not math.isfinite(threshold):
        raise ValueError(f"find_anomalies: threshold must be finite, got {threshold}")
    if top_n < 1:
        raise ValueError(f"find_anomalies: top_n must be >= 1, got {top_n}")
    if offset < 0:
        raise ValueError(f"find_anomalies: offset must be >= 0, got {offset}")

    if lance_version is not None:
        ds = _lance.dataset(geo_lance_path, version=lance_version)
    else:
        ds = _lance.dataset(geo_lance_path)

    total_found = ds.count_rows(filter=f"delta_norm >= {threshold}")
    if total_found == 0 or offset >= total_found:
        return {"keys": [], "delta_norms": [], "total_found": total_found}

    sql = (
        f"SELECT primary_key, delta_norm "
        f"FROM dataset "
        f"WHERE delta_norm >= {threshold} "
        f"ORDER BY delta_norm DESC "
        f"LIMIT {offset + top_n}"
    )
    batches = ds.sql(sql).build().to_batch_records()
    seen: dict[str, float] = {}
    for batch in batches:
        pk_col = batch["primary_key"]
        dn_col = batch["delta_norm"]
        for i in range(batch.num_rows):
            pk = pk_col[i].as_py()
            dn = float(dn_col[i].as_py())
            if pk not in seen or dn > seen[pk]:
                seen[pk] = dn

    items = list(seen.items())
    items.sort(key=lambda kv: -kv[1])
    paged = items[offset : offset + top_n]
    return {
        "keys": [k for k, _ in paged],
        "delta_norms": [v for _, v in paged],
        "total_found": total_found,
    }


def aggregate_percentile(
    geo_lance_path: str,
    ctx_lance_path: str,
    pattern: object,
    group_by_line: str,
    metric_col: str,
    percentile: float,
) -> tuple[dict[str, float], dict[str, int]]:
    """Per-group percentile of ``metric_col`` GROUP BY group_by_line.

    Streams ``(polygon_pk, group_key)`` pairs from the geometry, joins
    against ``(polygon_pk, metric_col)`` from the event line points, then
    runs DataFusion's ``approx_percentile_cont`` over the joined column for
    each group key.

    *Approximate* (T-Digest) — same semantics the legacy datafusion_agg
    path provided. For exact percentiles you need the in-process numpy
    fallback that aggregation.py keeps for the small-table case.
    """
    if not (0.0 <= percentile <= 1.0):
        raise ValueError(
            f"aggregate_percentile: percentile must be in [0, 1], got {percentile}"
        )
    metric_col = _validate_sql_identifier(metric_col)

    geo_pairs = _read_geo_edge_pairs(geo_lance_path, pattern, group_by_line)
    if geo_pairs.num_rows == 0:
        return {}, {}

    metric_pairs = _read_metric_pairs(ctx_lance_path, metric_col)
    if metric_pairs.num_rows == 0:
        return {}, {}

    joined = geo_pairs.join(
        metric_pairs, keys="primary_key", join_type="inner",
    )
    valid = pc.and_(pc.is_valid(joined[metric_col]), pc.is_valid(joined["k"]))
    filtered = joined.filter(valid)
    if filtered.num_rows == 0:
        return {}, {}

    # Hand the joined edge-level table to a tmp Lance dataset so we can run
    # the DataFusion approx_percentile_cont aggregate over it. The join
    # already happened in pyarrow, so writing the small joined table is
    # cheap and the percentile computation streams from disk.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = f"{td}/joined.lance"
        from hypertopos.storage.writer import _LANCE_WRITE_DEFAULTS
        _lance.write_dataset(
            filtered.select(["k", metric_col]).combine_chunks(),
            tmp_path,
            **_LANCE_WRITE_DEFAULTS,
        )
        tmp_ds = _lance.dataset(tmp_path)
        sql = (
            f"SELECT k AS group_key, "
            f"approx_percentile_cont({metric_col}, {percentile}) AS metric, "
            f"COUNT(*) AS cnt "
            f"FROM dataset GROUP BY k"
        )
        batches = tmp_ds.sql(sql).build().to_batch_records()

    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in batches:
        gk_col = batch["group_key"]
        m_col = batch["metric"]
        c_col = batch["cnt"]
        for i in range(batch.num_rows):
            gk = gk_col[i].as_py()
            if gk is None:
                continue
            metrics[gk] = float(m_col[i].as_py())
            counts[gk] = int(c_col[i].as_py())
    return metrics, counts


def aggregate_property(
    geo_lance_path: str,
    prop_lance_path: str,
    pattern: object,
    group_by_line: str,
    prop_line_id: str,
    prop_name: str,
    distinct: bool = False,
    ctx_lance_path: str | None = None,
    metric_col: str | None = None,
    agg_fn: str | None = None,
) -> dict:
    """Group-by-property aggregation: GROUP BY (group_by_line, prop_value).

    The property value is read from a separate line's points dataset by
    looking up each polygon's relation to ``prop_line_id`` and joining the
    resulting key against ``prop_lance_path[primary_key, prop_name]``. When
    ``distinct=True``, returns the count of distinct group_keys per prop_val.

    When ``ctx_lance_path`` + ``metric_col`` + ``agg_fn`` are supplied, also
    aggregates a metric column per (group_key, prop_val) cell.

    Returns a dict shaped like the legacy ``_property_aggregate_tables`` output:
    either ``{"results": [...]}`` or ``{"distinct_results": {...}}``.
    """
    prop_name = _validate_sql_identifier(prop_name)

    geo_pairs_g = _read_geo_edge_pairs(geo_lance_path, pattern, group_by_line)
    if geo_pairs_g.num_rows == 0:
        return {"distinct_results": {}} if distinct else {"results": []}

    if prop_line_id == group_by_line:
        # Single-line case: prop_entity is the same as group_key
        joined_pe = geo_pairs_g.append_column(
            "prop_entity", geo_pairs_g["k"],
        )
    else:
        geo_pairs_p = _read_geo_edge_pairs(geo_lance_path, pattern, prop_line_id)
        if geo_pairs_p.num_rows == 0:
            return {"distinct_results": {}} if distinct else {"results": []}
        # Join group-line and prop-line edges of the same polygon
        geo_pairs_p = geo_pairs_p.rename_columns(["primary_key", "prop_entity"])
        joined_pe = geo_pairs_g.join(
            geo_pairs_p, keys="primary_key", join_type="inner",
        )

    prop_ds = _lance.dataset(prop_lance_path)
    prop_table = prop_ds.scanner(
        columns=["primary_key", prop_name],
    ).to_table()
    if prop_table.num_rows == 0:
        return {"distinct_results": {}} if distinct else {"results": []}
    prop_lookup = pa.table({
        "prop_entity": prop_table["primary_key"],
        "prop_val": pc.cast(prop_table[prop_name], pa.string()),
    })

    enriched = joined_pe.join(
        prop_lookup, keys="prop_entity", join_type="inner",
    )
    valid = pc.and_(pc.is_valid(enriched["k"]), pc.is_valid(enriched["prop_val"]))
    f = enriched.filter(valid)
    if f.num_rows == 0:
        return {"distinct_results": {}} if distinct else {"results": []}

    if distinct:
        result = f.group_by("prop_val").aggregate([("k", "count_distinct")])
        return {
            "distinct_results": {
                row["prop_val"]: row["k_count_distinct"]
                for row in result.to_pylist()
            },
        }

    if agg_fn and ctx_lance_path and metric_col:
        if agg_fn not in _AGG_OP_MAP:
            raise ValueError(
                f"aggregate_property: agg_fn must be one of {sorted(_AGG_OP_MAP)}, "
                f"got {agg_fn!r}"
            )
        metric_pairs = _read_metric_pairs(ctx_lance_path, metric_col)
        f = f.join(metric_pairs, keys="primary_key", join_type="inner")
        f = f.filter(pc.is_valid(f[metric_col]))
        if f.num_rows == 0:
            return {"results": []}
        arrow_op = _AGG_OP_MAP[agg_fn]
        result = f.group_by(["k", "prop_val"]).aggregate(
            [(metric_col, arrow_op), (metric_col, "count")],
        )
        metric_field = f"{metric_col}_{arrow_op}"
        count_field = f"{metric_col}_count"
        return {
            "results": [
                {
                    "group_key": r["k"],
                    "prop_val": r["prop_val"],
                    "count": int(r[count_field]),
                    "metric": (
                        float(r[metric_field])
                        if r[metric_field] is not None else None
                    ),
                }
                for r in result.to_pylist()
            ],
        }

    result = f.group_by(["k", "prop_val"]).aggregate([("k", "count")])
    return {
        "results": [
            {
                "group_key": r["k"],
                "prop_val": r["prop_val"],
                "count": int(r["k_count"]),
            }
            for r in result.to_pylist()
        ],
    }


def aggregate_pivot(
    geo_lance_path: str,
    event_lance_path: str,
    pattern: object,
    group_by_line: str,
    pivot_field: str,
    ctx_lance_path: str | None = None,
    metric_col: str | None = None,
    agg_fn: str | None = None,
) -> list[dict]:
    """Pivot aggregation: GROUP BY (group_key, pivot_val) over the
    event geometry.

    The pivot_field is read from the event line points dataset and joined onto
    each polygon by ``primary_key``. When ``ctx_lance_path`` + ``metric_col`` +
    ``agg_fn`` are supplied the result also carries the requested metric
    aggregate per (group_key, pivot_val) cell; otherwise it returns count only.

    Returns a list of dicts shaped like the legacy
    ``_pivot_aggregate_tables`` output:
    ``[{group_key, pivot_val, count[, metric]}, ...]``.
    """
    pivot_field = _validate_sql_identifier(pivot_field)

    geo_pairs = _read_geo_edge_pairs(geo_lance_path, pattern, group_by_line)
    if geo_pairs.num_rows == 0:
        return []

    event_ds = _lance.dataset(event_lance_path)
    event_table = event_ds.scanner(
        columns=["primary_key", pivot_field],
    ).to_table()
    if event_table.num_rows == 0:
        return []
    event_pivot = pa.table({
        "primary_key": event_table["primary_key"],
        "pivot_val": pc.cast(event_table[pivot_field], pa.string()),
    })

    joined = geo_pairs.join(
        event_pivot, keys="primary_key", join_type="inner",
    )
    valid = pc.and_(pc.is_valid(joined["k"]), pc.is_valid(joined["pivot_val"]))

    if agg_fn and ctx_lance_path and metric_col:
        if agg_fn not in _AGG_OP_MAP:
            raise ValueError(
                f"aggregate_pivot: agg_fn must be one of {sorted(_AGG_OP_MAP)}, "
                f"got {agg_fn!r}"
            )
        metric_pairs = _read_metric_pairs(ctx_lance_path, metric_col)
        joined = joined.join(
            metric_pairs, keys="primary_key", join_type="inner",
        )
        valid = pc.and_(valid, pc.is_valid(joined[metric_col]))
        f = joined.filter(valid)
        if f.num_rows == 0:
            return []
        arrow_op = _AGG_OP_MAP[agg_fn]
        result = f.group_by(["k", "pivot_val"]).aggregate(
            [(metric_col, arrow_op), (metric_col, "count")],
        )
        metric_field = f"{metric_col}_{arrow_op}"
        count_field = f"{metric_col}_count"
        rows: list[dict] = []
        for r in result.to_pylist():
            rows.append({
                "group_key": r["k"],
                "pivot_val": r["pivot_val"],
                "count": int(r[count_field]),
                "metric": float(r[metric_field]) if r[metric_field] is not None else None,
            })
        return rows

    f = joined.filter(valid)
    if f.num_rows == 0:
        return []
    result = f.group_by(["k", "pivot_val"]).aggregate([("k", "count")])
    rows = []
    for r in result.to_pylist():
        rows.append({
            "group_key": r["k"],
            "pivot_val": r["pivot_val"],
            "count": int(r["k_count"]),
        })
    return rows


def aggregate_filtered_metric(
    geo_lance_path: str,
    ctx_lance_path: str,
    pattern: object,
    group_by_line: str,
    metric_col: str,
    agg_fn: str,
    resolved_filters: list[tuple[str, set[str]]],
    event_line_id: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Same shape as :func:`aggregate_metric` but pre-filters polygons by
    a list of ``(line_id, allowed_keys)`` constraints. Each filter is
    translated into a SQL predicate over the matching ``entity_keys``
    positions and AND-ed inside the Lance scan, so the network of "only
    polygons whose accounts ∈ {…} AND currencies ∈ {…}" filtering happens
    on disk before the join with the metric column ever loads anything
    into Python memory.
    """
    if agg_fn not in _AGG_OP_MAP:
        raise ValueError(
            f"aggregate_filtered_metric: agg_fn must be one of "
            f"{sorted(_AGG_OP_MAP)}, got {agg_fn!r}"
        )

    geo_pairs = _read_filtered_geo_edge_pairs(
        geo_lance_path, pattern, group_by_line, resolved_filters, event_line_id,
    )
    if geo_pairs.num_rows == 0:
        return {}, {}

    metric_pairs = _read_metric_pairs(ctx_lance_path, metric_col)
    if metric_pairs.num_rows == 0:
        return {}, {}

    joined = geo_pairs.join(
        metric_pairs, keys="primary_key", join_type="inner",
    )
    valid = pc.and_(pc.is_valid(joined[metric_col]), pc.is_valid(joined["k"]))
    filtered = joined.filter(valid)
    if filtered.num_rows == 0:
        return {}, {}

    arrow_op = _AGG_OP_MAP[agg_fn]
    result = filtered.group_by("k").aggregate(
        [(metric_col, arrow_op), (metric_col, "count")],
    )
    metric_field = f"{metric_col}_{arrow_op}"
    count_field = f"{metric_col}_count"
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    k_col = result["k"]
    m_col = result[metric_field]
    c_col = result[count_field]
    for i in range(result.num_rows):
        k = k_col[i].as_py()
        if k is None:
            continue
        metrics[k] = float(m_col[i].as_py())
        counts[k] = int(c_col[i].as_py())
    return metrics, counts


def aggregate_metric(
    geo_lance_path: str,
    ctx_lance_path: str,
    pattern: object,
    group_by_line: str,
    metric_col: str,
    agg_fn: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """sum / avg / min / max GROUP BY group_by_line via Lance SQL + pyarrow.

    Two-step pipeline:

    1. Lance SQL on the event geometry returns ``(primary_key, k)`` pairs —
       one per (polygon, position) where the position points at
       ``group_by_line``. Streams from disk through the Lance scanner; the
       full geometry table is never materialized in Python memory.
    2. Lance scanner on the event line points returns ``(primary_key,
       metric_col)`` pairs.
    3. PyArrow hash-joins the two on ``primary_key``, group-bys ``k``, and
       runs the requested aggregate. Both inputs are projection-pruned and
       small relative to the full geometry / event tables, so the join +
       aggregate fit comfortably in memory even on million-row patterns.

    Parameters
    ----------
    geo_lance_path
        Filesystem path to ``geometry/{event_pattern}/v={version}/data.lance``.
    ctx_lance_path
        Filesystem path to ``points/{event_line}/v={version}/data.lance``.
    pattern
        Pattern model object — read for ``relations`` to resolve which
        ``entity_keys`` positions belong to ``group_by_line``.
    group_by_line, metric_col
        The aggregate input.
    agg_fn
        One of ``"sum"``, ``"avg"``, ``"min"``, ``"max"``.

    Returns
    -------
    metrics, counts
        ``(metrics, counts)`` where ``metrics[k]`` is the aggregate value for
        group key ``k`` and ``counts[k]`` is how many edges contributed.
    """
    if agg_fn not in _AGG_OP_MAP:
        raise ValueError(
            f"aggregate_metric: agg_fn must be one of {sorted(_AGG_OP_MAP)}, "
            f"got {agg_fn!r}"
        )

    geo_pairs = _read_geo_edge_pairs(geo_lance_path, pattern, group_by_line)
    if geo_pairs.num_rows == 0:
        return {}, {}

    metric_pairs = _read_metric_pairs(ctx_lance_path, metric_col)
    if metric_pairs.num_rows == 0:
        return {}, {}

    joined = geo_pairs.join(
        metric_pairs, keys="primary_key", join_type="inner",
    )
    valid = pc.and_(pc.is_valid(joined[metric_col]), pc.is_valid(joined["k"]))
    filtered = joined.filter(valid)
    if filtered.num_rows == 0:
        return {}, {}

    arrow_op = _AGG_OP_MAP[agg_fn]
    result = filtered.group_by("k").aggregate(
        [(metric_col, arrow_op), (metric_col, "count")],
    )
    metric_field = f"{metric_col}_{arrow_op}"
    count_field = f"{metric_col}_count"
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    k_col = result["k"]
    m_col = result[metric_field]
    c_col = result[count_field]
    for i in range(result.num_rows):
        k = k_col[i].as_py()
        if k is None:
            continue
        metrics[k] = float(m_col[i].as_py())
        counts[k] = int(c_col[i].as_py())
    return metrics, counts
