# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Derived dimension computation for GDSBuilder.

Aggregates event-level data into per-anchor-entity features
that become continuous dimensions in the shape vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hypertopos.engine.chains import parse_timestamps_to_epoch


@dataclass
class DerivedDimSpec:
    """Specification for a derived dimension."""

    anchor_line: str
    event_line: str
    anchor_fk: str | list[str]
    metric: Literal["count", "count_distinct", "sum", "max", "std", "mean",
                    "iet_mean", "iet_std", "iet_min"]
    metric_col: str | None
    dimension_name: str
    edge_max: int | str  # int = fixed, "auto" = p99
    percentile: float  # for auto edge_max
    time_col: str | None = None
    time_window: str | None = None
    window_aggregation: str = "max"


@dataclass
class PrecomputedDimSpec:
    """Specification for a pre-computed dimension (column already on entity table)."""

    anchor_line: str
    dimension_name: str  # column name on entity table
    edge_max: int | str = "auto"  # int = fixed, "auto" = p{percentile}
    percentile: float = 99.0
    display_name: str | None = None


@dataclass
class CompositeLineSpec:
    """Specification for a composite (multi-key) line."""

    line_id: str
    event_line: str
    key_cols: list[str]
    separator: str


@dataclass
class GraphFeaturesSpec:
    """Specification for auto-computed graph features."""

    anchor_line: str
    event_line: str
    from_col: str
    to_col: str
    features: list[str]


def _parse_time_window(window: str) -> float:
    """Parse time window string to seconds. Supports 'd', 'h', 'm'."""
    unit = window[-1].lower()
    val = float(window[:-1])
    if unit == "d":
        return val * 86400
    elif unit == "h":
        return val * 3600
    elif unit == "m":
        return val * 60
    raise ValueError(f"Unknown time window unit: {window}")


def _apply_temporal_windowing(
    event_table: pa.Table,
    anchor_keys: pa.Array,
    anchor_fk: str,
    metric: str,
    metric_col: str | None,
    time_col: str,
    time_window: str,
    window_aggregation: str,
    edge_max: int | str,
    percentile: float,
) -> tuple[np.ndarray, int]:
    """Compute metric per anchor per time window, then aggregate across windows.

    Uses PyArrow groupby with time bucketing instead of pure-Python loops
    to avoid materialising millions of Python objects.
    """
    window_secs = _parse_time_window(time_window)
    n_anchor = len(anchor_keys)
    anchor_list = anchor_keys.to_pylist()
    key_to_idx = {k: i for i, k in enumerate(anchor_list)}

    # --- 1. Convert timestamps to epoch seconds (PyArrow-native) ----------
    ts_col = event_table[time_col]

    if pa.types.is_timestamp(ts_col.type):
        unit = ts_col.type.unit
        divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
        divisor = divisors.get(unit, 1e6)
        epoch_arr = pc.cast(ts_col, pa.int64())
        epoch_arr = pc.divide(pc.cast(epoch_arr, pa.float64()), divisor)
    elif pa.types.is_floating(ts_col.type) or pa.types.is_integer(ts_col.type):
        epoch_arr = pc.cast(ts_col, pa.float64())
    else:
        # String / other — fall back to parse_timestamps_to_epoch
        epoch_arr = pa.chunked_array(
            [pa.array(
                parse_timestamps_to_epoch(ts_col.to_pylist()),
                type=pa.float64(),
            )]
        )

    # --- 2. Compute time buckets: floor((ts - min_ts) / window_secs) ------
    min_ts = pc.min(epoch_arr).as_py()
    if min_ts is None:
        return np.zeros(n_anchor, dtype=np.float64), 1

    diff = pc.subtract(epoch_arr, min_ts)
    bucket_arr = pc.cast(
        pc.floor(pc.divide(pc.cast(diff, pa.float64()), window_secs)),
        pa.int64(),
    )

    # --- 3. Build work table with bucket column ---------------------------
    work_table = event_table.append_column("_bucket", bucket_arr)

    # --- 4. Group by (anchor_fk, _bucket) and compute per-bucket metric ---
    _agg_map = {
        "count": ("primary_key", "count"),
        "count_distinct": (metric_col, "count_distinct"),
        "sum": (metric_col, "sum"),
        "max": (metric_col, "max"),
        "mean": (metric_col, "mean"),
        "std": (metric_col, "stddev"),
    }
    if metric not in _agg_map:
        raise ValueError(f"Unknown metric: {metric}")

    agg_col, agg_func = _agg_map[metric]
    grouped = work_table.group_by([anchor_fk, "_bucket"]).aggregate(
        [(agg_col, agg_func)]
    )
    metric_result_col = f"{agg_col}_{agg_func}"

    # --- 5. Aggregate across buckets per anchor (window_aggregation) -------
    if window_aggregation == "last":
        # "last" = value from the highest bucket per anchor — no native
        # PyArrow "last" aggregation, so fall back to a lightweight loop
        # on the *already-reduced* grouped table (not the raw events).
        gk = grouped[anchor_fk].to_pylist()
        gb = grouped["_bucket"].to_pylist()
        gv = grouped[metric_result_col].to_pylist()

        last_per_anchor: dict[str, float] = {}
        last_bucket: dict[str, int] = {}
        for k, b, v in zip(gk, gb, gv, strict=False):
            if k is not None and (k not in last_bucket or b > last_bucket[k]):
                last_bucket[k] = b
                last_per_anchor[k] = float(v) if v is not None else 0.0

        values = np.zeros(n_anchor, dtype=np.float64)
        for k, v in last_per_anchor.items():
            idx = key_to_idx.get(k)
            if idx is not None:
                values[idx] = v
        em = _resolve_edge_max(values, edge_max, percentile)
        return values, em

    _win_agg_map = {"max": "max", "mean": "mean"}
    if window_aggregation not in _win_agg_map:
        raise ValueError(f"Unknown window_aggregation: {window_aggregation}")

    win_func = _win_agg_map[window_aggregation]
    final = grouped.group_by(anchor_fk).aggregate(
        [(metric_result_col, win_func)]
    )
    final_col = f"{metric_result_col}_{win_func}"

    # --- 6. Map results back to anchor array ------------------------------
    values = np.zeros(n_anchor, dtype=np.float64)
    result_keys = final[anchor_fk].to_pylist()
    result_vals = final[final_col].to_pylist()
    for k, v in zip(result_keys, result_vals, strict=False):
        idx = key_to_idx.get(k)
        if idx is not None:
            values[idx] = float(v) if v is not None else 0.0

    em = _resolve_edge_max(values, edge_max, percentile)
    return values, em


def compute_derived_dimension(
    event_table: pa.Table,
    anchor_keys: pa.Array,
    anchor_fk: str | list[str],
    metric: str,
    metric_col: str | None,
    edge_max: int | str,
    percentile: float,
    time_col: str | None = None,
    time_window: str | None = None,
    window_aggregation: str = "max",
    separator: str = "→",
) -> tuple[np.ndarray, int]:
    """Compute aggregated values per anchor entity from event data.

    Returns (values_array aligned to anchor_keys, computed_edge_max).
    """
    # Temporal windowing path
    if time_col and time_window and isinstance(anchor_fk, list):
        raise ValueError(
            "temporal windowing (time_col/time_window) is not supported "
            "for composite (list) anchor_fk"
        )
    if time_col and time_window and isinstance(anchor_fk, str):
        return _apply_temporal_windowing(
            event_table, anchor_keys, anchor_fk, metric, metric_col,
            time_col, time_window, window_aggregation,
            edge_max, percentile,
        )

    if isinstance(anchor_fk, list):
        return _compute_composite_derived(
            event_table, anchor_keys, anchor_fk, metric, metric_col,
            edge_max, percentile, separator=separator,
        )


    fk_col = event_table[anchor_fk]
    n_anchor = len(anchor_keys)

    # Build key→index mapping for anchor
    anchor_list = anchor_keys.to_pylist()
    key_to_idx: dict[str, int] = {k: i for i, k in enumerate(anchor_list)}

    values = np.zeros(n_anchor, dtype=np.float64)

    # Fast path: PyArrow group_by for count (most common case)
    if metric == "count":
        try:
            grouped = event_table.group_by(anchor_fk).aggregate(
                [("primary_key", "count")]
            )
            gk = grouped[anchor_fk].to_pylist()
            gc = grouped["primary_key_count"].to_pylist()
            for k, c in zip(gk, gc, strict=False):
                idx = key_to_idx.get(k)
                if idx is not None:
                    values[idx] = c
        except Exception:
            _aggregate_count(fk_col, key_to_idx, values)
    elif metric == "count_distinct":
        try:
            grouped = event_table.group_by(anchor_fk).aggregate(
                [(metric_col, "count_distinct")]
            )
            gk = grouped[anchor_fk].to_pylist()
            gc = grouped[f"{metric_col}_count_distinct"].to_pylist()
            for k, c in zip(gk, gc, strict=False):
                idx = key_to_idx.get(k)
                if idx is not None:
                    values[idx] = c
        except Exception:
            _aggregate_count_distinct(fk_col, event_table[metric_col], key_to_idx, values)
    elif metric in ("sum", "max", "mean", "std"):
        pa_agg_map = {"sum": "sum", "max": "max", "mean": "mean", "std": "stddev"}
        pa_func = pa_agg_map[metric]
        try:
            grouped = event_table.group_by(anchor_fk).aggregate(
                [(metric_col, pa_func)]
            )
            gk = grouped[anchor_fk].to_pylist()
            gc = grouped[f"{metric_col}_{pa_func}"].to_pylist()
            for k, c in zip(gk, gc, strict=False):
                idx = key_to_idx.get(k)
                if idx is not None:
                    values[idx] = float(c) if c is not None else 0.0
        except Exception:
            _aggregate_numeric(fk_col, event_table[metric_col], key_to_idx, values, metric)
    elif metric.startswith("iet_"):
        if time_col is None:
            raise ValueError(f"IET metric '{metric}' requires time_col")
        agg = metric.split("_", 1)[1]  # "mean", "std", or "min"
        values = _aggregate_iet(
            event_table, anchor_keys, anchor_fk, time_col, agg, key_to_idx,
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    em = _resolve_edge_max(values, edge_max, percentile)
    return values, em


def _ts_column_to_epoch_array(col: pa.ChunkedArray) -> pa.Array:
    """Convert an Arrow column to float64 epoch seconds.

    Handles Arrow timestamp types (via cast to int64 microseconds → divide),
    integer/float types (passthrough or unit conversion), and string types
    (fromisoformat fallback via parse_timestamps_to_epoch).
    """
    flat = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    t = flat.type

    if pa.types.is_timestamp(t):
        # Cast to int64 (unit preserved), then convert to seconds
        as_int = pc.cast(flat, pa.int64())
        unit = t.unit  # "s", "ms", "us", "ns"
        divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
        divisor = divisors.get(unit, 1e6)
        return pc.cast(pc.divide(pc.cast(as_int, pa.float64()), divisor), pa.float64())

    if pa.types.is_integer(t):
        return pc.cast(flat, pa.float64())

    if pa.types.is_floating(t):
        return pc.cast(flat, pa.float64())

    # String or other — parse via Python helper (single pylist call, unavoidable)
    epoch_list = parse_timestamps_to_epoch(flat.to_pylist())
    return pa.array(epoch_list, type=pa.float64())


def _aggregate_iet(
    event_table: pa.Table,
    anchor_keys: pa.Array,
    anchor_fk: str,
    time_col: str,
    agg: str,
    key_to_idx: dict[str, int],
) -> np.ndarray:
    """Compute inter-event-time aggregates per anchor entity (Arrow-native).

    Converts timestamps to epoch floats via Arrow compute, sorts by
    (fk, epoch), computes consecutive diffs with NumPy, masks cross-entity
    boundaries, then aggregates per entity. No per-row Python loop.
    Returns values in seconds (float64).
    """
    n_anchor = len(anchor_keys)
    values = np.zeros(n_anchor, dtype=np.float64)

    # --- 1. Build a minimal table with fk and epoch_ts, drop nulls ---
    fk_col = event_table[anchor_fk]
    ts_col = event_table[time_col]

    epoch_col = _ts_column_to_epoch_array(ts_col)

    # Combine fk to a flat array if chunked
    fk_flat = fk_col.combine_chunks() if isinstance(fk_col, pa.ChunkedArray) else fk_col

    # Drop rows where fk or epoch is null/zero-epoch (null timestamps become 0.0)
    fk_valid = pc.is_valid(fk_flat)
    ts_valid = pc.is_valid(epoch_col)
    mask = pc.and_(fk_valid, ts_valid)

    mini = pa.table(
        {"fk": pc.filter(fk_flat, mask), "epoch": pc.filter(epoch_col, mask)}
    )

    if len(mini) == 0:
        return values

    # --- 2. Sort by (fk, epoch) using Arrow sort ---
    sort_indices = pc.sort_indices(
        mini, sort_keys=[("fk", "ascending"), ("epoch", "ascending")]
    )
    sorted_mini = mini.take(sort_indices)

    fk_np = sorted_mini["fk"].to_pylist()  # string list — used for entity key lookup per group
    epoch_col_sorted = sorted_mini["epoch"]
    if isinstance(epoch_col_sorted, pa.ChunkedArray):
        epoch_col_sorted = epoch_col_sorted.combine_chunks()
    epoch_np = epoch_col_sorted.to_numpy().astype(np.float64)

    n = len(epoch_np)
    if n < 2:
        return values

    # --- 3. Compute consecutive diffs; find group boundaries ---
    diffs = np.empty(n, dtype=np.float64)
    diffs[0] = 0.0
    diffs[1:] = epoch_np[1:] - epoch_np[:-1]

    # Detect entity-change boundaries via Arrow comparison on the fk column —
    # compare fk[:-1] vs fk[1:] without materializing Python objects per row.
    fk_arr = sorted_mini["fk"]
    if isinstance(fk_arr, pa.ChunkedArray):
        fk_arr = fk_arr.combine_chunks()
    is_new_entity = pc.not_equal(fk_arr[1:], fk_arr[:-1])  # BooleanArray length n-1
    # boundary[i] True means row i starts a new entity group
    boundary_np = np.empty(n, dtype=bool)
    boundary_np[0] = True   # first row always starts a group
    boundary_np[1:] = is_new_entity.to_numpy(zero_copy_only=False)

    # Group start indices: where boundary_np is True
    group_starts = np.flatnonzero(boundary_np)
    # Group end indices (exclusive): next start or n
    group_ends = np.empty(len(group_starts), dtype=np.intp)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = n

    # --- 4. Aggregate per entity ---
    # Each entity spans rows [start, end); valid diffs are diffs[start+1 : end]
    for g_start, g_end in zip(group_starts, group_ends, strict=False):
        entity = fk_np[g_start]
        idx = key_to_idx.get(entity)
        if idx is None:
            continue
        entity_diffs = diffs[g_start + 1 : g_end]  # within-entity consecutive diffs
        if len(entity_diffs) == 0:
            continue  # only 1 event — IET stays 0.0
        if agg == "mean":
            values[idx] = float(entity_diffs.mean())
        elif agg == "std":
            values[idx] = float(entity_diffs.std())
        elif agg == "min":
            values[idx] = float(entity_diffs.min())

    return values


def _aggregate_count(
    fk_col: pa.ChunkedArray, key_to_idx: dict[str, int], out: np.ndarray,
) -> None:
    """Count events per anchor key."""
    counts: dict[str, int] = {}
    for chunk in fk_col.chunks:
        for val in chunk.to_pylist():
            if val is not None:
                counts[val] = counts.get(val, 0) + 1
    for k, c in counts.items():
        idx = key_to_idx.get(k)
        if idx is not None:
            out[idx] = c


def _aggregate_count_distinct(
    fk_col: pa.ChunkedArray,
    metric_col: pa.ChunkedArray,
    key_to_idx: dict[str, int],
    out: np.ndarray,
) -> None:
    """Count distinct metric_col values per anchor key."""
    sets: dict[str, set] = {}
    fk_chunks = fk_col.chunks
    mc_chunks = metric_col.chunks

    # Flatten both columns in sync
    fk_flat = []
    mc_flat = []
    for chunk in fk_chunks:
        fk_flat.extend(chunk.to_pylist())
    for chunk in mc_chunks:
        mc_flat.extend(chunk.to_pylist())

    for fk_val, mc_val in zip(fk_flat, mc_flat, strict=False):
        if fk_val is not None and mc_val is not None:
            if fk_val not in sets:
                sets[fk_val] = set()
            sets[fk_val].add(mc_val)

    for k, s in sets.items():
        idx = key_to_idx.get(k)
        if idx is not None:
            out[idx] = len(s)


def _aggregate_numeric(
    fk_col: pa.ChunkedArray,
    metric_col: pa.ChunkedArray,
    key_to_idx: dict[str, int],
    out: np.ndarray,
    metric: str,
) -> None:
    """Aggregate numeric metric_col per anchor key (sum/max/mean/std)."""
    from collections import defaultdict

    groups: dict[str, list[float]] = defaultdict(list)

    fk_flat = fk_col.to_pylist()
    mc_flat = metric_col.to_pylist()

    for fk_val, mc_val in zip(fk_flat, mc_flat, strict=False):
        if fk_val is not None and mc_val is not None:
            groups[fk_val].append(float(mc_val))

    for k, vals in groups.items():
        idx = key_to_idx.get(k)
        if idx is None:
            continue
        arr = np.array(vals)
        if metric == "sum":
            out[idx] = arr.sum()
        elif metric == "max":
            out[idx] = arr.max()
        elif metric == "mean":
            out[idx] = arr.mean()
        elif metric == "std":
            out[idx] = arr.std() if len(arr) > 1 else 0.0


def _compute_composite_derived(
    event_table: pa.Table,
    anchor_keys: pa.Array,
    anchor_fk: list[str],
    metric: str,
    metric_col: str | None,
    edge_max: int | str,
    percentile: float,
    separator: str = "→",
) -> tuple[np.ndarray, int]:
    """Derived dimension on composite (multi-column) anchor keys."""
    sep = separator
    n_anchor = len(anchor_keys)
    anchor_list = anchor_keys.to_pylist()
    key_to_idx = {k: i for i, k in enumerate(anchor_list)}

    # Build composite FK from event table (Arrow-native string concat)
    str_cols = [pc.cast(event_table[col], pa.string()) for col in anchor_fk]
    composite_fk_arr = pc.binary_join_element_wise(*str_cols, sep)

    values = np.zeros(n_anchor, dtype=np.float64)

    if metric == "count":
        _aggregate_count(composite_fk_arr, key_to_idx, values)
    elif metric == "count_distinct":
        mc = event_table[metric_col]
        _aggregate_count_distinct(composite_fk_arr, mc, key_to_idx, values)
    elif metric in ("sum", "max", "mean", "std"):
        mc = event_table[metric_col]
        _aggregate_numeric(composite_fk_arr, mc, key_to_idx, values, metric)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    em = _resolve_edge_max(values, edge_max, percentile)
    return values, em


_BATCH_METRICS = frozenset({"count", "count_distinct", "sum", "max", "mean", "std"})

# Arrow aggregate function names for each metric
_PA_AGG_MAP = {
    "sum": "sum",
    "max": "max",
    "mean": "mean",
    "std": "stddev",
}


def _is_batchable(spec: DerivedDimSpec) -> bool:
    """Return True if spec can be handled via a batched multi-aggregate group_by.

    Specs that require special handling (IET, temporal windowing)
    must stay on the single-dim path. Composite FK specs are now batchable.
    """
    if spec.metric not in _BATCH_METRICS:
        return False
    return not (spec.time_col and spec.time_window)


def compute_derived_batch(
    event_table: pa.Table,
    anchor_keys: pa.Array,
    anchor_fk: str | list[str],
    specs: list[DerivedDimSpec],
    separator: str = "→",
) -> dict[str, tuple[np.ndarray, int]]:
    """Compute multiple derived dimensions in ONE group_by call.

    All *specs* must share the same ``(event_line, anchor_fk)`` and must be
    batchable (simple metrics, no IET/temporal).  Supports both single-column
    and composite (list) anchor_fk.

    Returns ``{dimension_name: (values_array, edge_max)}``.
    """
    n_anchor = len(anchor_keys)
    anchor_list = anchor_keys.to_pylist()
    key_to_idx: dict[str, int] = {k: i for i, k in enumerate(anchor_list)}

    # For composite FK, build a synthetic string column and group by that
    if isinstance(anchor_fk, list):
        str_cols = [pc.cast(event_table[col], pa.string()) for col in anchor_fk]
        composite_col = pc.binary_join_element_wise(*str_cols, separator)
        group_col = "__composite_fk__"
        event_table = event_table.append_column(group_col, composite_col)
    else:
        group_col = anchor_fk

    # Build aggregate expressions, deduplicating identical (col, func) pairs.
    agg_exprs: list[tuple[str, str]] = []
    seen_exprs: set[tuple[str, str]] = set()

    spec_to_result_col: dict[str, str] = {}

    for spec in specs:
        if spec.metric == "count":
            agg_col = group_col
            agg_func = "count"
        elif spec.metric == "count_distinct":
            agg_col = spec.metric_col  # type: ignore[assignment]
            agg_func = "count_distinct"
        else:
            agg_col = spec.metric_col  # type: ignore[assignment]
            agg_func = _PA_AGG_MAP[spec.metric]

        result_col_name = f"{agg_col}_{agg_func}"
        spec_to_result_col[spec.dimension_name] = result_col_name

        expr_key = (agg_col, agg_func)
        if expr_key not in seen_exprs:
            seen_exprs.add(expr_key)
            agg_exprs.append(expr_key)

    # Single group_by with all aggregate expressions
    grouped = event_table.group_by(group_col).aggregate(agg_exprs)

    # Materialise the FK column once — shared across all specs
    grouped_keys = grouped[group_col].to_pylist()

    # Build result for each spec
    results: dict[str, tuple[np.ndarray, int]] = {}
    # Cache already-materialised result columns to avoid repeated to_pylist()
    _col_cache: dict[str, list] = {}

    for spec in specs:
        result_col_name = spec_to_result_col[spec.dimension_name]

        if result_col_name not in _col_cache:
            _col_cache[result_col_name] = grouped[result_col_name].to_pylist()
        grouped_vals = _col_cache[result_col_name]

        values = np.zeros(n_anchor, dtype=np.float64)
        for k, v in zip(grouped_keys, grouped_vals, strict=False):
            idx = key_to_idx.get(k)
            if idx is not None:
                values[idx] = float(v) if v is not None else 0.0

        em = _resolve_edge_max(values, spec.edge_max, spec.percentile)
        results[spec.dimension_name] = (values, em)

    return results


def _scatter_grouped(
    grouped: pa.Table,
    key_col: str,
    val_col: str,
    key_to_idx: dict[str, int],
    n: int,
) -> np.ndarray:
    """Map grouped Arrow result to anchor-aligned numpy array."""
    vals = np.zeros(n, dtype=np.float64)
    gk = grouped[key_col].to_pylist()
    gv = grouped[val_col].to_pylist()
    for k, v in zip(gk, gv, strict=False):
        idx = key_to_idx.get(k)
        if idx is not None:
            vals[idx] = float(v) if v is not None else 0.0
    return vals


def _resolve_edge_max(values: np.ndarray, edge_max: int | str, percentile: float) -> int:
    """Compute edge_max from data or return fixed value."""
    if isinstance(edge_max, int):
        return edge_max
    # "auto" — use percentile
    nonzero = values[values > 0]
    if len(nonzero) == 0:
        return 1
    return max(1, int(np.percentile(nonzero, percentile)))


def build_composite_table(
    event_table: pa.Table,
    key_cols: list[str],
    separator: str = "→",
) -> pa.Table:
    """Extract unique composite keys from event data and build anchor table.

    Arrow-native implementation: casts key columns to string, joins with
    separator via pc.binary_join_element_wise, deduplicates via pc.unique.
    Rows where any key column is null are dropped before joining.
    Component columns are preserved alongside primary_key.
    Zero Python materialization for the dedup path.
    """
    # Cast each key column to string
    str_cols = [pc.cast(event_table[col], pa.string()) for col in key_cols]

    # Drop rows where any key column is null
    valid_mask = pc.is_valid(str_cols[0])
    for col_arr in str_cols[1:]:
        valid_mask = pc.and_(valid_mask, pc.is_valid(col_arr))
    str_cols = [pc.filter(col_arr, valid_mask) for col_arr in str_cols]

    # Build composite key: separator is the LAST positional argument
    composite = pc.binary_join_element_wise(*str_cols, separator)

    # Deduplicate via sort on composite key, keeping first occurrence per key.
    # Build a work table with composite + component columns, sort by composite,
    # then keep rows where composite differs from the previous row.
    work = pa.table({"primary_key": composite, **dict(zip(key_cols, str_cols, strict=False))})
    sort_indices = pc.sort_indices(work, sort_keys=[("primary_key", "ascending")])
    sorted_work = work.take(sort_indices)

    pk_sorted = sorted_work["primary_key"]
    if isinstance(pk_sorted, pa.ChunkedArray):
        pk_sorted = pk_sorted.combine_chunks()

    # Keep rows where key differs from previous (first row always kept)
    if len(pk_sorted) == 0:
        return sorted_work

    is_first = pc.not_equal(pk_sorted[1:], pk_sorted[:-1])
    keep = pa.concat_arrays([pa.array([True]), is_first])
    return sorted_work.filter(keep)


def compute_graph_features(
    event_table: pa.Table,
    anchor_keys: pa.Array,
    from_col: str,
    to_col: str,
    features: list[str],
) -> dict[str, tuple[np.ndarray, int]]:
    """Compute graph features for anchor entities (Arrow-native).

    Uses Arrow group_by for degree metrics and Arrow joins for
    reciprocity/overlap. No Python loops over event rows.

    Returns {feature_name: (values_array, edge_max)}.
    """
    anchor_list = anchor_keys.to_pylist()
    key_to_idx = {k: i for i, k in enumerate(anchor_list)}
    n = len(anchor_list)

    need_out = "out_degree" in features
    need_in = "in_degree" in features
    need_recip = "reciprocity" in features
    need_overlap = "counterpart_overlap" in features

    # Drop rows where from or to is null — one Arrow filter
    f_col = event_table[from_col]
    t_col = event_table[to_col]
    both_valid = pc.and_(pc.is_valid(f_col), pc.is_valid(t_col))
    edges = pa.table({"_f": pc.filter(f_col, both_valid),
                       "_t": pc.filter(t_col, both_valid)})

    results: dict[str, tuple[np.ndarray, int]] = {}

    # --- out_degree: count_distinct(to) GROUP BY from ---
    out_grouped = None
    if need_out or need_overlap:
        out_grouped = edges.group_by("_f").aggregate([("_t", "count_distinct")])
        if need_out:
            vals = _scatter_grouped(out_grouped, "_f", "_t_count_distinct",
                                    key_to_idx, n)
            results["out_degree"] = (vals, _resolve_edge_max(vals, "auto", 99.0))

    # --- in_degree: count_distinct(from) GROUP BY to ---
    in_grouped = None
    if need_in or need_overlap:
        in_grouped = edges.group_by("_t").aggregate([("_f", "count_distinct")])
        if need_in:
            vals = _scatter_grouped(in_grouped, "_t", "_f_count_distinct",
                                    key_to_idx, n)
            results["in_degree"] = (vals, _resolve_edge_max(vals, "auto", 99.0))

    # --- reciprocity: 1.0 if entity is both sender AND receiver ---
    if need_recip:
        senders = pc.unique(edges["_f"])
        receivers = pc.unique(edges["_t"])
        # Entities in both sets
        is_recip = pc.is_in(senders, value_set=receivers)
        recip_keys = pc.filter(senders, is_recip).to_pylist()
        vals = np.zeros(n, dtype=np.float64)
        for k in recip_keys:
            idx = key_to_idx.get(k)
            if idx is not None:
                vals[idx] = 1.0
        results["reciprocity"] = (vals, 1)

    # --- counterpart_overlap: Jaccard(out_targets, in_sources) per entity ---
    if need_overlap:
        # Unique directed edges
        unique_edges = pa.table({
            "_f": edges["_f"], "_t": edges["_t"],
        }).group_by(["_f", "_t"]).aggregate([("_f", "count")]).select(["_f", "_t"])

        # out_edges: (entity, counterparty) where entity sends to counterparty
        # in_edges:  (entity, counterparty) where entity receives from counterparty
        # → swap columns on in_edges to get same schema
        in_edges = pa.table({"_f": unique_edges["_t"], "_t": unique_edges["_f"]})

        # Bidirectional: inner join → pairs that exist in BOTH directions
        bidir = unique_edges.join(in_edges, keys=["_f", "_t"], join_type="inner")
        bidir_grouped = bidir.group_by("_f").aggregate([("_t", "count_distinct")])
        bidir_vals = _scatter_grouped(bidir_grouped, "_f", "_t_count_distinct",
                                      key_to_idx, n)

        # out_degree and in_degree (reuse if already computed)
        out_vals = (_scatter_grouped(out_grouped, "_f", "_t_count_distinct",
                                     key_to_idx, n)
                    if out_grouped is not None
                    else np.zeros(n, dtype=np.float64))
        in_vals = (_scatter_grouped(in_grouped, "_t", "_f_count_distinct",
                                    key_to_idx, n)
                   if in_grouped is not None
                   else np.zeros(n, dtype=np.float64))

        # Jaccard = |intersection| / |union| = bidir / (out + in - bidir)
        union_size = out_vals + in_vals - bidir_vals
        vals = np.where(union_size > 0, bidir_vals / union_size, 0.0)
        results["counterpart_overlap"] = (vals, 1)

    return results
