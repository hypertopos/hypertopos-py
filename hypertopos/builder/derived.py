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

    # Build result for each spec — vectorized scatter via Arrow pc.index_in
    from hypertopos.builder._scatter import vectorized_scatter_1d

    results: dict[str, tuple[np.ndarray, int]] = {}
    anchor_keys_arr = pa.array(anchor_list)
    grouped_fk_pa = grouped[group_col]

    for spec in specs:
        result_col_name = spec_to_result_col[spec.dimension_name]

        values = np.zeros(n_anchor, dtype=np.float64)
        vectorized_scatter_1d(
            values,
            anchor_keys_arr=anchor_keys_arr,
            grouped_fk_col=grouped_fk_pa,
            grouped_values_col=grouped[result_col_name],
        )

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
    from hypertopos.builder._scatter import vectorized_scatter_1d

    vals = np.zeros(n, dtype=np.float64)
    # Build ordered anchor keys array matching key_to_idx
    anchor_keys_ordered = [""] * n
    for k, i in key_to_idx.items():
        anchor_keys_ordered[i] = k
    anchor_keys_arr = pa.array(anchor_keys_ordered)
    vectorized_scatter_1d(
        vals,
        anchor_keys_arr=anchor_keys_arr,
        grouped_fk_col=grouped[key_col],
        grouped_values_col=grouped[val_col],
    )
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
        safe_union = np.where(union_size > 0, union_size, 1.0)
        vals = np.where(union_size > 0, bidir_vals / safe_union, 0.0)
        results["counterpart_overlap"] = (vals, 1)

    # --- Graph algorithm features (computed on AdjacencyIndex) ---
    _ALGO_FEATURES = {
        "pagerank", "connected_component", "clustering_coefficient",
        "community", "betweenness",
    }
    needed_algos = _ALGO_FEATURES & set(features)
    if needed_algos:
        from hypertopos.engine.graph_algorithms import compute_all_from_lists

        f_list = edges["_f"].to_pylist()
        t_list = edges["_t"].to_pylist()
        algo_results = compute_all_from_lists(f_list, t_list, needed_algos)

        _EDGE_MAX_OVERRIDES = {
            "clustering_coefficient": 1,
            "betweenness": 1,
        }
        for feat_name, scores in algo_results.items():
            vals = np.zeros(n, dtype=np.float64)
            for k, v in scores.items():
                idx = key_to_idx.get(k)
                if idx is not None:
                    vals[idx] = float(v)
            if feat_name in ("connected_component", "community"):
                n_unique = len(set(scores.values())) if scores else 1
                edge_max = max(n_unique - 1, 1)
            elif feat_name in _EDGE_MAX_OVERRIDES:
                edge_max = _EDGE_MAX_OVERRIDES[feat_name]
            else:
                edge_max = _resolve_edge_max(vals, "auto", 99.0)
            results[feat_name] = (vals, edge_max)

    return results


def compute_graph_features_temporal(
    event_table: pa.Table,
    anchor_keys: pa.Array,
    from_col: str,
    to_col: str,
    features: list[str],
    bucket_assignments: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """Compute graph features across all temporal buckets in a single pass.

    For ``in_degree`` and ``out_degree``, uses Arrow group_by with a
    bucket column so all buckets are computed in one aggregation.
    For ``reciprocity``, ``counterpart_overlap``, and unknown features,
    falls back to per-window :func:`compute_graph_features` calls.

    Returns an (n_anchor, n_buckets, n_features) float32 array of raw
    counts (not normalised — caller handles edge_max scaling).
    """
    anchor_list = anchor_keys.to_pylist()
    key_to_idx = {k: i for i, k in enumerate(anchor_list)}
    n = len(anchor_list)
    n_feats = len(features)

    result = np.zeros((n, n_buckets, n_feats), dtype=np.float32)

    # Classify features
    FAST_FEATURES = {"in_degree", "out_degree"}
    NUMPY_FEATURES = {"reciprocity", "counterpart_overlap"}
    # Graph algo features are static-only — skip in temporal (per-window
    # PageRank/betweenness is meaningless and extremely expensive)
    STATIC_ONLY_FEATURES = {
        "pagerank", "connected_component", "clustering_coefficient",
        "community", "betweenness",
    }
    fast_indices = [i for i, f in enumerate(features) if f in FAST_FEATURES]
    need_recip = "reciprocity" in features
    need_overlap = "counterpart_overlap" in features
    need_numpy = need_recip or need_overlap
    unknown_indices = [
        i for i, f in enumerate(features)
        if f not in FAST_FEATURES and f not in NUMPY_FEATURES
        and f not in STATIC_ONLY_FEATURES
    ]

    # --- Shared: null-filter and integer-encode once ---
    f_arr: pa.Array | None = None
    t_arr: pa.Array | None = None
    valid_mask: np.ndarray | None = None
    anchor_keys_arr: pa.Array | None = None

    if fast_indices or need_numpy:
        f_col = event_table[from_col]
        t_col = event_table[to_col]
        both_valid = pc.and_(pc.is_valid(f_col), pc.is_valid(t_col))
        valid_mask = both_valid.to_numpy(zero_copy_only=False)

        f_arr = pc.filter(f_col, both_valid)
        t_arr = pc.filter(t_col, both_valid)

        # Build ordered anchor keys array for vectorized scatter
        anchor_keys_ordered = [""] * n
        for k, i in key_to_idx.items():
            anchor_keys_ordered[i] = k
        anchor_keys_arr = pa.array(anchor_keys_ordered)

    # --- Degree features: single-pass Arrow group_by ---
    if fast_indices:
        b_arr = pa.array(bucket_assignments[valid_mask], type=pa.int32())
        edges_tbl = pa.table({"_f": f_arr, "_t": t_arr, "_b": b_arr})

        if "out_degree" in features:
            out_grouped = edges_tbl.group_by(["_f", "_b"]).aggregate(
                [("_t", "count_distinct")],
            )
            _scatter_temporal(
                result, features.index("out_degree"), out_grouped,
                "_f", "_b", "_t_count_distinct",
                anchor_keys_arr, n,
            )

        if "in_degree" in features:
            in_grouped = edges_tbl.group_by(["_t", "_b"]).aggregate(
                [("_f", "count_distinct")],
            )
            _scatter_temporal(
                result, features.index("in_degree"), in_grouped,
                "_t", "_b", "_f_count_distinct",
                anchor_keys_arr, n,
            )

    # --- NumPy fast path for reciprocity / counterpart_overlap ---
    if need_numpy:
        b_np = bucket_assignments[valid_mask]

        # Anchor-space integer indices via pc.index_in (null → NaN → -1)
        f_anchor_raw = pc.index_in(f_arr, anchor_keys_arr)
        t_anchor_raw = pc.index_in(t_arr, anchor_keys_arr)
        f_anchor_idx = pc.fill_null(f_anchor_raw, -1).to_numpy(
            zero_copy_only=False,
        ).astype(np.intp)
        t_anchor_idx = pc.fill_null(t_anchor_raw, -1).to_numpy(
            zero_copy_only=False,
        ).astype(np.intp)
        f_anchor = f_anchor_idx
        t_anchor = t_anchor_idx

        # Pre-sort by bucket for O(1) slicing
        sort_order = np.argsort(b_np)
        sorted_b = b_np[sort_order]
        sorted_f_anchor = f_anchor[sort_order]
        sorted_t_anchor = t_anchor[sort_order]
        boundaries = np.searchsorted(sorted_b, np.arange(n_buckets + 1))

        # Global integer encoding for overlap (all entities, not just anchors)
        if need_overlap:
            # f_arr/t_arr may be ChunkedArray from pc.filter
            f_flat = f_arr.combine_chunks() if hasattr(f_arr, "combine_chunks") else f_arr
            t_flat = t_arr.combine_chunks() if hasattr(t_arr, "combine_chunks") else t_arr
            all_uniq = pc.unique(pa.concat_arrays([f_flat, t_flat]))
            n_global = len(all_uniq)
            # Map from/to → global int via pc.index_in
            f_global = pc.index_in(f_arr, all_uniq).to_numpy(
                zero_copy_only=False,
            ).astype(np.int64)
            t_global = pc.index_in(t_arr, all_uniq).to_numpy(
                zero_copy_only=False,
            ).astype(np.int64)
            sorted_f_global = f_global[sort_order]
            sorted_t_global = t_global[sort_order]
            # Anchor mapping: global_idx → anchor_idx (-1 if not anchor)
            global_to_anchor = np.full(n_global, -1, dtype=np.intp)
            anchor_global = pc.fill_null(
                pc.index_in(anchor_keys_arr, all_uniq), -1,
            ).to_numpy(zero_copy_only=False).astype(np.intp)
            for ai in range(n):
                gi = anchor_global[ai]
                if gi >= 0:
                    global_to_anchor[gi] = ai

        recip_feat_idx = features.index("reciprocity") if need_recip else -1
        overlap_feat_idx = features.index("counterpart_overlap") if need_overlap else -1

        for bucket_idx in range(n_buckets):
            b_start = boundaries[bucket_idx]
            b_end = boundaries[bucket_idx + 1]
            if b_start == b_end:
                continue

            if need_recip:
                bf = sorted_f_anchor[b_start:b_end]
                bt = sorted_t_anchor[b_start:b_end]
                senders = np.unique(bf[bf >= 0])
                receivers = np.unique(bt[bt >= 0])
                reciprocal = np.intersect1d(senders, receivers)
                if len(reciprocal) > 0:
                    result[reciprocal, bucket_idx, recip_feat_idx] = 1.0

            if need_overlap:
                bfg = sorted_f_global[b_start:b_end]
                btg = sorted_t_global[b_start:b_end]
                # Packed int64 edge encoding for fast set ops
                fwd_packed = np.unique(bfg * n_global + btg)
                rev_packed = np.unique(btg * n_global + bfg)
                bidir_packed = np.intersect1d(fwd_packed, rev_packed)

                # Per-anchor counts via np.bincount
                # out_degree: unique targets per anchor sender
                fwd_senders = fwd_packed // n_global
                fwd_s_anchor = global_to_anchor[fwd_senders]
                valid_fwd = fwd_s_anchor >= 0
                out_count = np.bincount(
                    fwd_s_anchor[valid_fwd], minlength=n,
                ).astype(np.float32)

                # in_degree: unique sources per anchor receiver
                fwd_targets = fwd_packed % n_global
                fwd_t_anchor = global_to_anchor[fwd_targets]
                valid_in = fwd_t_anchor >= 0
                in_count = np.bincount(
                    fwd_t_anchor[valid_in], minlength=n,
                ).astype(np.float32)

                # bidir_count: per anchor sender in bidirectional edges
                if len(bidir_packed) > 0:
                    bidir_senders = bidir_packed // n_global
                    bidir_s_anchor = global_to_anchor[bidir_senders]
                    valid_bidir = bidir_s_anchor >= 0
                    bidir_count = np.bincount(
                        bidir_s_anchor[valid_bidir], minlength=n,
                    ).astype(np.float32)
                else:
                    bidir_count = np.zeros(n, dtype=np.float32)

                union = out_count + in_count - bidir_count
                safe_union = np.where(union > 0, union, 1.0)
                jaccard = np.where(union > 0, bidir_count / safe_union, 0.0)
                result[:, bucket_idx, overlap_feat_idx] = jaccard

    # --- Fallback: per-window for truly unknown features ---
    if unknown_indices:
        unknown_features = [features[i] for i in unknown_indices]

        sort_order_fb = np.argsort(bucket_assignments)
        sorted_buckets_fb = bucket_assignments[sort_order_fb]
        boundaries_fb = np.searchsorted(
            sorted_buckets_fb, np.arange(n_buckets + 1),
        )

        for bucket_idx in range(n_buckets):
            b_start = boundaries_fb[bucket_idx]
            b_end = boundaries_fb[bucket_idx + 1]
            if b_start == b_end:
                continue

            indices = sort_order_fb[b_start:b_end]
            filtered = event_table.take(pa.array(indices, type=pa.int64()))
            feature_results = compute_graph_features(
                filtered, anchor_keys, from_col, to_col, unknown_features,
            )
            for fi, feat in zip(
                unknown_indices, unknown_features, strict=False,
            ):
                if feat in feature_results:
                    values, _ = feature_results[feat]
                    result[:, bucket_idx, fi] = values

    return result


def _scatter_temporal(
    tensor: np.ndarray,
    feat_idx: int,
    grouped: pa.Table,
    key_col: str,
    bucket_col: str,
    val_col: str,
    anchor_keys_arr: pa.Array,
    n: int,
) -> None:
    """Scatter grouped (key, bucket, value) into the 3D tensor."""
    if len(grouped) == 0:
        return

    entity_idx_arr = pc.index_in(grouped[key_col], anchor_keys_arr)
    valid = pc.and_(
        pc.is_valid(entity_idx_arr),
        pc.is_valid(grouped[val_col]),
    )

    entity_indices = (
        entity_idx_arr.filter(valid)
        .to_numpy(zero_copy_only=False)
        .astype(np.intp)
    )
    bucket_indices = (
        grouped[bucket_col]
        .filter(valid)
        .to_numpy(zero_copy_only=False)
        .astype(np.intp)
    )
    values = (
        grouped[val_col]
        .filter(valid)
        .to_numpy(zero_copy_only=False)
        .astype(np.float32)
    )

    tensor[entity_indices, bucket_indices, feat_idx] = values


