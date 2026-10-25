# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Edge-derived dimension catalog for event patterns.

Five build-time dimensions. Each takes the event pattern's edge_table
(PyArrow) and returns one float32 array of length ``edges.num_rows``.
The orchestrator :func:`compute_all_edge_dims` runs all dims listed in
a config dict and returns a single Arrow table keyed by ``event_key``
for sidecar persistence.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

EDGE_DIM_KINDS: dict[str, str] = {
    "pair_edge_count": "poisson",
    "position_in_chain": "poisson",
    "time_since_pair_last_edge": "gaussian",
    "pair_amount_zscore": "gaussian",
    "find_motif_structuring": "bernoulli",
    "edge_curvature_frc": "gaussian",
}


def _pair_codes(edges: pa.Table) -> np.ndarray:
    """One int64 id per ``(from_key, to_key)`` pair, equal ids for equal pairs."""
    from_codes = pc.dictionary_encode(edges["from_key"].combine_chunks())
    to_codes = pc.dictionary_encode(edges["to_key"].combine_chunks())
    n_to = max(len(to_codes.dictionary), 1)
    from_ids = np.asarray(from_codes.indices.to_numpy(), dtype=np.int64)
    to_ids = np.asarray(to_codes.indices.to_numpy(), dtype=np.int64)
    return from_ids * n_to + to_ids


def compute_pair_edge_count(edges: pa.Table) -> pa.Array:
    """Per-edge count of how many times the (from, to) pair appears."""
    if edges.num_rows == 0:
        return pa.array([], type=pa.float32())
    _, inverse, counts = np.unique(_pair_codes(edges), return_inverse=True, return_counts=True)
    return pa.array(counts[inverse].astype(np.float32), type=pa.float32())


def compute_position_in_chain(
    edges: pa.Table,
    *,
    min_position: int,
) -> pa.Array:
    if edges.num_rows == 0:
        return pa.array([], type=pa.float32())

    tss = np.asarray(edges["timestamp"].to_pylist(), dtype=np.float64)
    order = np.argsort(tss, kind="stable")
    fks = edges["from_key"].to_pylist()
    tks = edges["to_key"].to_pylist()

    chain_at_node: dict[str, int] = {}
    pos = np.zeros(edges.num_rows, dtype=np.int64)
    for idx in order:
        fk = fks[idx]
        tk = tks[idx]
        prior = chain_at_node.get(fk, 0)
        my_pos = prior + 1
        pos[idx] = my_pos
        if my_pos > chain_at_node.get(tk, 0):
            chain_at_node[tk] = my_pos

    out = np.where(
        pos >= min_position,
        pos.astype(np.float32),
        np.float32(0.0),
    )
    return pa.array(out, type=pa.float32())


def compute_time_since_pair_last_edge(
    edges: pa.Table,
    *,
    burst_seconds: float,  # noqa: ARG001 — kept for YAML parity
    dormant_seconds: float,
) -> pa.Array:
    """Per-edge: seconds since the previous edge in the same `(from, to)`
    pair; first edge in a pair gets `dormant_seconds`.

    Vectorised: sort by `(from, to, timestamp)`, take row-to-row diff,
    mask diffs across pair boundaries with `dormant_seconds`, scatter
    back to original edge order. Replaces a Python per-pair groupby +
    sort + scan that scaled O(n_edges) on plain Python objects.
    """
    n = edges.num_rows
    if n == 0:
        return pa.array([], type=pa.float32())
    # Order rows by (pair, timestamp); stable sorts keep ties in row order.
    ts_all = edges["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    pair = _pair_codes(edges)
    sort_idx = np.argsort(ts_all, kind="stable")
    sort_idx = sort_idx[np.argsort(pair[sort_idx], kind="stable")]
    pairs_sorted = pair[sort_idx]
    ts = ts_all[sort_idx]
    # `same_pair_as_prev[i]` = True iff row i belongs to the same (from,to)
    # group as row i-1 (so we should diff with row i-1's timestamp).
    same_prev = np.zeros(n, dtype=bool)
    if n > 1:
        same_prev[1:] = pairs_sorted[1:] == pairs_sorted[:-1]
    diffs = np.empty(n, dtype=np.float32)
    diffs[0] = np.float32(dormant_seconds)
    if n > 1:
        delta = (ts[1:] - ts[:-1]).astype(np.float32)
        diffs[1:] = np.where(
            same_prev[1:],
            delta,
            np.float32(dormant_seconds),
        )
    out = np.empty(n, dtype=np.float32)
    out[sort_idx] = diffs
    return pa.array(out, type=pa.float32())


def compute_pair_amount_zscore(
    edges: pa.Table,
    *,
    cv_threshold: float,
    min_count: int,
) -> pa.Array:
    """Per-edge signed z-score of amount within `(from, to)` pairs whose
    `CV(amount) < cv_threshold` and `count >= min_count`. HIGH-variance and
    too-small pairs emit 0.0.

    Vectorised: sort by `(from, to)`, derive contiguous group ids via
    cumsum on pair boundaries, compute per-group mean/std via numpy
    `bincount`, broadcast back to per-row z-score via group-id indexing,
    scatter to original order. Replaces a Python defaultdict + per-group
    `np.array(...)` + scan that scaled O(n_edges).
    """
    n = edges.num_rows
    if n == 0:
        return pa.array([], type=pa.float32())
    pair = _pair_codes(edges)
    sort_idx = np.argsort(pair, kind="stable")
    pairs_sorted = pair[sort_idx]
    amts = (
        pc.fill_null(edges["amount"], 0.0)
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.float64)[sort_idx]
    )

    # Assign each row a contiguous group id per (from, to) pair.
    same_prev = np.zeros(n, dtype=bool)
    if n > 1:
        same_prev[1:] = pairs_sorted[1:] == pairs_sorted[:-1]
    # `new_group[i] = True` exactly at first row of each group.
    new_group = ~same_prev
    group_id = (np.cumsum(new_group) - 1).astype(np.int64)
    n_groups = int(group_id[-1]) + 1

    # A missing amount is stored as 0.0; only positive amounts enter the pair
    # statistics, and missing rows get z = 0.0 below.
    present = amts > 0
    weights = present.astype(np.float64)

    # Per-group running statistics via bincount.
    counts = np.bincount(group_id, weights=weights, minlength=n_groups)
    sums = np.bincount(group_id, weights=amts * weights, minlength=n_groups)
    sq_sums = np.bincount(
        group_id,
        weights=(amts * amts * weights),
        minlength=n_groups,
    )
    safe_counts = np.where(counts > 0, counts, 1.0)
    means = sums / safe_counts
    variances = np.maximum(sq_sums / safe_counts - means * means, 0.0)
    stds = np.sqrt(variances)

    # Filter mask: count >= min_count AND mean != 0 AND std != 0
    # AND cv = std/|mean| < cv_threshold.
    abs_means = np.abs(means)
    safe_abs_means = np.where(abs_means > 0, abs_means, 1.0)
    cvs = stds / safe_abs_means
    valid_group = (counts >= min_count) & (means != 0.0) & (stds != 0.0) & (cvs < cv_threshold)

    # Per-row z-score for valid groups, 0.0 otherwise.
    valid_per_row = valid_group[group_id] & present
    safe_stds_per_row = np.where(stds[group_id] != 0, stds[group_id], 1.0)
    z = np.where(
        valid_per_row,
        (amts - means[group_id]) / safe_stds_per_row,
        0.0,
    ).astype(np.float32)

    out = np.empty(n, dtype=np.float32)
    out[sort_idx] = z
    return pa.array(out, type=pa.float32())


def compute_find_motif_structuring(
    edges: pa.Table,
    *,
    time_window_hours: float,
    amt1_min: float,
    amt2_max: float,
) -> pa.Array:
    from hypertopos.engine.structuring import enumerate_structuring_event_keys

    motif_keys = enumerate_structuring_event_keys(
        edges,
        time_window_sec=time_window_hours * 3600.0,
        amt1_min=amt1_min,
        amt2_max=amt2_max,
    )
    event_keys = edges["event_key"]
    flagged = pc.is_in(event_keys, value_set=pa.array(sorted(motif_keys), type=event_keys.type))
    return pa.array(flagged.to_numpy(zero_copy_only=False).astype(np.float32), type=pa.float32())


AGGREGATE_NAMES: tuple[str, ...] = (
    "mean",
    "max",
    "std",
    "p95",
    "count_above_threshold",
)

# Map our aggregate name -> (pyarrow group_by aggregate spec, options, output column suffix).
# pyarrow names hash-aggregate output columns as "<col>_<pyarrow-agg-name>";
# we rename to canonical "<col>_<our-name>" in _fill_aggregate_columns.
# When the pyarrow agg name is None, the aggregate is computed manually.
_AGG_TO_PYARROW: dict[str, tuple[str | None, Any | None, str]] = {
    "mean": ("mean", None, "mean"),
    "max": ("max", None, "max"),
    "std": ("stddev", pc.VarianceOptions(ddof=0), "stddev"),
    # p95 was previously delegated to pyarrow's `tdigest` group_by aggregate
    # (the only quantile aggregation in the hash-aggregate suite). Empirical
    # benchmark on 10M-row × 515k-group input showed tdigest = ~88% of the
    # whole `_group_by` cost (801s vs 1-5s for mean/max/stddev). Replaced by
    # a vectorized exact-quantile path post-group_by — see
    # `_fill_aggregate_columns`. The (None, None, …) marker keeps p95 out of
    # the pyarrow aggregate spec list.
    "p95": (None, None, "p95"),
    # count_above_threshold is computed manually post-group_by from the
    # underlying values + a per-dim threshold (passed via the `thresholds`
    # kwarg, default = population p95 from the sidecar).
    "count_above_threshold": (None, None, "count_above_threshold"),
}


def _build_pyarrow_aggs(
    aggregates_per_dim: dict[str, tuple[str, ...]],
) -> list[tuple[str, str] | tuple[str, str, Any]]:
    """Build the pyarrow group_by aggregation spec list for the per-dim
    aggregate selection. Skips manual aggregates (count_above_threshold)."""
    aggs: list[tuple[str, str] | tuple[str, str, Any]] = []
    for d, agg_list in aggregates_per_dim.items():
        for agg_name in agg_list:
            pyarrow_agg, options, _suffix = _AGG_TO_PYARROW[agg_name]
            if pyarrow_agg is None:
                continue  # manual aggregate, computed post-group_by
            if options is None:
                aggs.append((d, pyarrow_agg))
            else:
                aggs.append((d, pyarrow_agg, options))
    return aggs


def _resolve_count_above_thresholds(
    sidecar: pa.Table,
    dims: list[str],
    overrides: dict[str, float] | None,
) -> dict[str, float]:
    """Per-dim threshold for count_above_threshold. Default = population p95
    of the source dim from the sidecar; user can override via `thresholds`.
    """
    out: dict[str, float] = {}
    for d in dims:
        if overrides is not None and d in overrides:
            out[d] = float(overrides[d])
            continue
        col = sidecar[d]
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        # Quantile on full sidecar — exact, no group_by required.
        if len(col) == 0:
            out[d] = 0.0
            continue
        # pc.quantile returns an Array of length len(q); we passed scalar 0.95.
        raw_arr = pc.quantile(col, q=0.95)
        raw = raw_arr.as_py() if hasattr(raw_arr, "as_py") else raw_arr.to_pylist()
        q = raw[0] if isinstance(raw, list) else raw
        if q is None:
            out[d] = 0.0
            continue
        qf = float(q)
        out[d] = qf if np.isfinite(qf) else 0.0
    return out


def _anchor_positions(keys: pa.Array | pa.ChunkedArray, anchors: pa.Array) -> np.ndarray:
    """Row of each key in ``anchors`` (-1 when the key is not an anchor)."""
    found = pc.index_in(keys, value_set=anchors)
    return np.asarray(pc.fill_null(found, -1).to_numpy(zero_copy_only=False), dtype=np.int64)


def _fill_aggregate_columns(
    *,
    out_cols: dict[str, pa.Array],
    grouped: pa.Table,
    anchor_keys: list[str],
    aggregates_per_dim: dict[str, tuple[str, ...]],
    n_anchors: int,
    base: pa.Table | None = None,
    thresholds: dict[str, float] | None = None,
) -> None:
    """Read each per-aggregate column from the grouped table and write it
    into out_cols under canonical naming `<dim>_<our_agg_name>`.

    Every key is mapped once to its anchor row with ``pc.index_in``; all
    later work is on integer rows (no string sorting). Anchors without a
    value get 0, NaN results become 0.

    1. **pyarrow group_by aggregates** (mean / max / std) — scattered from the
       post-group_by table.
    2. **p95** — exact quantile: rows ordered by (anchor, value) with two
       stable sorts, then the ``int((size - 1) * 0.95)``-th value of each
       anchor's run.
    3. **count_above_threshold** — ``bincount`` of the anchor rows whose
       value exceeds the dim's threshold.
    """
    anchors = pa.array(anchor_keys, type=pa.string())

    # 1. pyarrow group_by aggregates (mean / max / std).
    grouped_rows = _anchor_positions(grouped["primary_key"], anchors)
    hit = grouped_rows >= 0
    for d, agg_list in aggregates_per_dim.items():
        for agg_name in agg_list:
            pyarrow_agg, _options, suffix = _AGG_TO_PYARROW[agg_name]
            if pyarrow_agg is None:
                continue  # manual: handled in sections (2) and (3) below
            col_arr = grouped[f"{d}_{suffix}"]
            if isinstance(col_arr, pa.ChunkedArray):
                col_arr = col_arr.combine_chunks()
            col_np = col_arr.to_numpy(zero_copy_only=False)
            buf = np.zeros(n_anchors, dtype=np.float32)
            vals = col_np[hit]
            buf[grouped_rows[hit]] = np.where(np.isnan(vals), 0.0, vals).astype(np.float32)
            out_cols[f"{d}_{agg_name}"] = pa.array(buf)

    if base is None:
        return
    need_p95 = any("p95" in agg_list for agg_list in aggregates_per_dim.values())
    need_count = any(
        "count_above_threshold" in agg_list for agg_list in aggregates_per_dim.values()
    )
    if not (need_p95 or need_count):
        return
    base_rows = _anchor_positions(base["primary_key"], anchors)
    on_anchor = base_rows >= 0
    rows = base_rows[on_anchor]

    # 2. Manual p95 — exact quantile per anchor.
    if need_p95:
        for d, agg_list in aggregates_per_dim.items():
            if "p95" not in agg_list:
                continue
            vals_arr = base[d]
            if isinstance(vals_arr, pa.ChunkedArray):
                vals_arr = vals_arr.combine_chunks()
            vals = vals_arr.to_numpy(zero_copy_only=False)[on_anchor]
            buf = np.zeros(n_anchors, dtype=np.float32)
            if len(rows):
                order = np.argsort(vals, kind="stable")
                order = order[np.argsort(rows[order], kind="stable")]
                sorted_rows, sorted_vals = rows[order], vals[order]
                starts = np.flatnonzero(np.r_[True, sorted_rows[1:] != sorted_rows[:-1]])
                sizes = np.diff(np.r_[starts, len(sorted_rows)])
                p95 = sorted_vals[starts + ((sizes - 1) * 0.95).astype(np.int64)]
                buf[sorted_rows[starts]] = np.where(np.isnan(p95), 0.0, p95).astype(np.float32)
            out_cols[f"{d}_p95"] = pa.array(buf)

    # 3. count_above_threshold — rows above the dim's threshold per anchor.
    if need_count:
        thresholds = thresholds or {}
        for d, agg_list in aggregates_per_dim.items():
            if "count_above_threshold" not in agg_list:
                continue
            thr = float(thresholds.get(d, 0.0))
            # NaN and null values are not above any threshold.
            above = pc.fill_null(pc.greater(base[d], pa.scalar(thr, type=pa.float32())), False)
            mask = np.asarray(above.to_numpy(zero_copy_only=False), dtype=bool)[on_anchor]
            counts = np.bincount(rows[mask], minlength=n_anchors).astype(np.float32)
            out_cols[f"{d}_count_above_threshold"] = pa.array(counts)


def aggregate_kind(source_kind: str, agg: str) -> str:
    if agg == "mean":
        return "gaussian"
    if agg == "max":
        return "bernoulli" if source_kind == "bernoulli" else "gaussian"
    if agg == "std":
        return "gaussian"
    if agg == "p95":
        return "bernoulli" if source_kind == "bernoulli" else "gaussian"
    if agg == "count_above_threshold":
        return "poisson"  # count of edges crossing the threshold
    raise ValueError(f"unknown aggregate: {agg!r}")


def aggregate_edge_dims_for_anchor(
    *,
    anchor_keys: list[str],
    edges: pa.Table,
    sidecar: pa.Table,
    dims: list[str],
    anchor_kind: str,
    pair_separator: str = "→",
    chain_events: list[str] | None = None,
    key_cols: list[str] | None = None,
    event_table: pa.Table | None = None,
    thresholds: dict[str, float] | None = None,
    aggregates_per_dim: dict[str, tuple[str, ...]] | None = None,
) -> pa.Table:
    if anchor_kind not in ("single", "pair", "chain"):
        raise ValueError(
            f"anchor_kind must be one of 'single' / 'pair' / 'chain'; got {anchor_kind!r}",
        )
    if anchor_kind == "chain":
        if chain_events is None:
            raise ValueError(
                "chain regime requires chain_events="
                "list[str] of comma-joined event_keys per anchor",
            )
        if len(chain_events) != len(anchor_keys):
            raise ValueError(
                f"len(chain_events)={len(chain_events)} must match "
                f"len(anchor_keys)={len(anchor_keys)}",
            )
    for d in dims:
        if d not in EDGE_DIM_KINDS:
            raise ValueError(
                f"unknown edge dimension: {d!r}; valid: {sorted(EDGE_DIM_KINDS)}",
            )
        if d not in sidecar.column_names:
            raise ValueError(
                f"edge dimension {d!r} not present in sidecar; "
                f"available: {sorted(set(sidecar.column_names) - {'event_key'})}",
            )
    # Default for direct API callers / unit tests that don't pass an
    # explicit selector: every dim gets all five canonical aggregates,
    # matching the pre-selector behavior. Callers from the builder always
    # pass a fully populated mapping resolved at YAML parse time.
    if aggregates_per_dim is None:
        aggregates_per_dim = {d: tuple(AGGREGATE_NAMES) for d in dims}
    else:
        if set(aggregates_per_dim) != set(dims):
            raise ValueError(
                f"aggregates_per_dim keys {sorted(aggregates_per_dim)!r} "
                f"must match dims {sorted(dims)!r}",
            )
        for d, agg_list in aggregates_per_dim.items():
            if not agg_list:
                raise ValueError(
                    f"aggregates_per_dim[{d!r}] must be a non-empty tuple of aggregate names",
                )
            for agg in agg_list:
                if agg not in AGGREGATE_NAMES:
                    raise ValueError(
                        f"unknown aggregate {agg!r} for dim {d!r}; valid: {list(AGGREGATE_NAMES)}",
                    )

    n_anchors = len(anchor_keys)
    out_cols: dict[str, pa.Array] = {
        "primary_key": pa.array(anchor_keys, type=pa.string()),
    }

    if (anchor_kind != "chain" and edges.num_rows == 0) or sidecar.num_rows == 0:
        for d, agg_list in aggregates_per_dim.items():
            for agg in agg_list:
                out_cols[f"{d}_{agg}"] = pa.array(
                    np.zeros(n_anchors, dtype=np.float32),
                )
        return pa.table(out_cols)

    # Resolve count_above_threshold thresholds from population p95 of sidecar
    # (overridable via `thresholds` kwarg). Computed once on full sidecar
    # before per-anchor group_by, so threshold is a population property,
    # not a per-anchor property.
    resolved_thresholds = _resolve_count_above_thresholds(
        sidecar,
        dims,
        thresholds,
    )

    if anchor_kind == "chain":
        chain_id_flat: list[str] = []
        event_key_flat: list[str] = []
        # chain_events is validated non-None for the chain regime above.
        chain_events_list = cast("list[str]", chain_events)
        for cid, evs_str in zip(anchor_keys, chain_events_list, strict=True):
            if not evs_str:
                continue
            evs = evs_str.split(",")
            chain_id_flat.extend([cid] * len(evs))
            event_key_flat.extend(evs)

        if not chain_id_flat:
            for d, agg_list in aggregates_per_dim.items():
                for agg in agg_list:
                    out_cols[f"{d}_{agg}"] = pa.array(
                        np.zeros(n_anchors, dtype=np.float32),
                    )
            return pa.table(out_cols)

        exploded = pa.table(
            {
                "primary_key": pa.array(chain_id_flat, type=pa.string()),
                "event_key": pa.array(event_key_flat, type=pa.string()),
            }
        )
        joined = exploded.join(sidecar, keys="event_key")
        aggs = _build_pyarrow_aggs(aggregates_per_dim)
        grouped = joined.group_by("primary_key").aggregate(aggs)
        _fill_aggregate_columns(
            out_cols=out_cols,
            grouped=grouped,
            anchor_keys=anchor_keys,
            aggregates_per_dim=aggregates_per_dim,
            n_anchors=n_anchors,
            base=joined,
            thresholds=resolved_thresholds,
        )
        return pa.table(out_cols)

    joined = edges.join(sidecar, keys="event_key")

    if anchor_kind == "single":
        from_pk_arr = joined["from_key"].combine_chunks()
        to_pk_arr = joined["to_key"].combine_chunks()
        anchor_pk_arr = pa.concat_arrays([from_pk_arr, to_pk_arr])
        dim_arrays: dict[str, pa.Array] = {}
        for d in dims:
            v = joined[d].combine_chunks()
            dim_arrays[d] = pa.concat_arrays([v, v])
    else:
        # pair / k>2 composite regime — construct PK from positional key_cols.
        # k=2 (backward compat): use edge endpoints (from_key, to_key) hard-coded.
        # k>2: first two key_cols still map positionally to edge endpoints
        # (the edge_table extract renames event_table.{from_col,to_col} to
        # from_key / to_key); remaining key_cols are looked up in `joined`
        # first (covers callers passing pre-merged tables and unit tests),
        # then in event_table if provided (production builder path).
        joined_for_pk = joined
        if not key_cols or len(key_cols) == 2:
            # k=2 backward compat: ignore actual key_cols names, hard-code
            # edge endpoints. Matches the original pair convention.
            cols = ["from_key", "to_key"]
        else:
            property_cols = list(key_cols[2:])
            missing_in_joined = [c for c in property_cols if c not in joined.column_names]
            if missing_in_joined:
                if event_table is None:
                    raise ValueError(
                        f"k>2 composite anchor property key_cols[2:]="
                        f"{property_cols!r} not present in joined edges+sidecar "
                        f"and event_table=None — pass event_table so the "
                        f"missing columns {missing_in_joined!r} can be joined",
                    )
                missing_in_event = [
                    c for c in missing_in_joined if c not in event_table.column_names
                ]
                if missing_in_event:
                    raise ValueError(
                        f"k>2 composite anchor property key_cols not present "
                        f"in event_table either: missing={missing_in_event!r}; "
                        f"event_table available="
                        f"{sorted(event_table.column_names)!r}",
                    )
                event_key_col = (
                    "event_key"
                    if "event_key" in event_table.column_names
                    else "primary_key"  # event line uses primary_key
                )
                prop_subset = event_table.select(
                    [event_key_col, *missing_in_joined],
                )
                if event_key_col != "event_key":
                    prop_subset = prop_subset.rename_columns(
                        ["event_key", *missing_in_joined],
                    )
                joined_for_pk = joined.join(prop_subset, keys="event_key")
            cols = ["from_key", "to_key", *property_cols]
        # Cast each key column to string and join positionally with pair_separator.
        # Mirrors build_composite_table convention so anchor PKs match what
        # was registered at composite_line resolution time.
        str_cols = [pc.cast(joined_for_pk[c].combine_chunks(), pa.string()) for c in cols]
        anchor_pk_arr = pc.binary_join_element_wise(*str_cols, pair_separator)
        dim_arrays = {d: joined_for_pk[d].combine_chunks() for d in dims}

    base = pa.table({"primary_key": anchor_pk_arr, **dim_arrays})
    aggs = _build_pyarrow_aggs(aggregates_per_dim)
    grouped = base.group_by("primary_key").aggregate(aggs)
    _fill_aggregate_columns(
        out_cols=out_cols,
        grouped=grouped,
        anchor_keys=anchor_keys,
        aggregates_per_dim=aggregates_per_dim,
        n_anchors=n_anchors,
        base=base,
        thresholds=resolved_thresholds,
    )
    return pa.table(out_cols)


def compute_edge_curvature_frc(edges: pa.Table) -> pa.Array:
    """Combinatorial Forman-Ricci curvature per edge.

    For an undirected graph G = (V, E), the combinatorial Forman-Ricci
    curvature of edge ``(u, v)`` is::

        F(u, v) = 4 − deg(u) − deg(v) + #triangles_containing(u, v)

    Algebraically equivalent to ``2 + #triangles − (deg(u) − 1) − (deg(v) − 1)``.

    Triangle counts come from ``(A @ A).multiply(A)`` — the elementwise
    product of the adjacency-squared matrix with the adjacency matrix
    itself, restricted to entries where an edge exists. This avoids
    materialising the full ``V × V`` density of ``A @ A`` on graphs with
    hub structure.

    Multigraph collapse: the adjacency matrix carries presence (1), not
    multiplicity. Combinatorial FRC is defined on simple graphs; the same
    ``(u, v)`` pair appearing N times in the event table still represents
    a single undirected edge for curvature purposes — ``(kA) @ (kA) = k² · (A @ A)``
    would otherwise inflate triangle counts quadratically.

    Direction canonicalisation: AML transactions are directed
    (sender → receiver) but FRC is undirected. Edges ``(u, v)`` and
    ``(v, u)`` between the same accounts share the same FRC value because
    they share the same triangle structure in the undirected sense — the
    pair key is canonicalised via ``sorted([from_key, to_key])``.

    Per-event output: all events sharing the same canonical ``(min, max)``
    pair receive the same FRC value, broadcast back to the input event
    order via the packed-key + ``searchsorted`` pattern (mirror of
    ``compute_pair_edge_count``).
    """
    if edges.num_rows == 0:
        return pa.array([], type=pa.float32())

    from scipy import sparse

    # ── 1. Per-event canonical undirected pair key ──────────────────────
    fks = np.asarray(edges["from_key"].to_pylist(), dtype=object)
    tks = np.asarray(edges["to_key"].to_pylist(), dtype=object)
    lo = np.where(fks <= tks, fks, tks)
    hi = np.where(fks <= tks, tks, fks)
    sep = "\x00"
    e_key = np.array([f"{a}{sep}{b}" for a, b in zip(lo, hi, strict=False)], dtype=object)

    # ── 2. Unique undirected pairs ──────────────────────────────────────
    unique_keys, e_to_pair = np.unique(e_key, return_inverse=True)
    n_pairs = len(unique_keys)

    pair_lo = np.empty(n_pairs, dtype=object)
    pair_hi = np.empty(n_pairs, dtype=object)
    for p_idx in range(n_pairs):
        a, b = unique_keys[p_idx].split(sep)
        pair_lo[p_idx] = a
        pair_hi[p_idx] = b

    # ── 3. Node index over union of endpoints ───────────────────────────
    # np.unique returns sorted array; pure-numpy searchsorted avoids a
    # Python dict (~50 MB overhead on 515 k-node spheres) and matches the
    # searchsorted discipline used for the per-event broadcast below.
    all_nodes = np.unique(np.concatenate([pair_lo, pair_hi]))
    V = len(all_nodes)
    u_idx = np.searchsorted(all_nodes, pair_lo)
    v_idx = np.searchsorted(all_nodes, pair_hi)

    # ── 4. Symmetric sparse adjacency (presence, not multiplicity) ──────
    # Multigraph collapse: data is np.ones(...) — same pair appearing many
    # times in events still contributes a single 1 to A (see docstring).
    # Self-transfers (u == v) are not graph edges: they would enter A twice on
    # the diagonal and inflate deg(u). They are left out and get FRC 0.0.
    is_self = u_idx == v_idx
    nu, nv = u_idx[~is_self], v_idx[~is_self]
    rows = np.concatenate([nu, nv])
    cols = np.concatenate([nv, nu])
    data = np.ones(rows.size, dtype=np.float64)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()

    # ── 5. Triangle counts: (A @ A).multiply(A), restricted to edges ───
    A2_restricted = (A @ A).multiply(A)
    # Pull per-pair triangle counts in one batched sparse fancy-index.
    tri_counts = np.asarray(A2_restricted[u_idx, v_idx]).flatten()

    # ── 6. Degrees ──────────────────────────────────────────────────────
    deg = np.asarray(A.sum(axis=1)).flatten()

    # ── 7. Per-pair FRC: 4 − deg(u) − deg(v) + #triangles ───────────────
    pair_frc = (4.0 - deg[u_idx] - deg[v_idx] + tri_counts).astype(np.float32)
    pair_frc[is_self] = 0.0

    # ── 8. Broadcast per-pair → per-event order ─────────────────────────
    out = pair_frc[e_to_pair]
    return pa.array(out, type=pa.float32())


def compute_all_edge_dims(
    edges: pa.Table,
    config: dict[str, dict[str, Any]],
) -> pa.Table:
    """Run each declared dim, return Arrow table keyed by event_key."""
    columns: dict[str, pa.Array] = {"event_key": edges["event_key"]}
    for dim_name, params in config.items():
        if dim_name == "pair_edge_count":
            columns[dim_name] = compute_pair_edge_count(edges)
        elif dim_name == "position_in_chain":
            columns[dim_name] = compute_position_in_chain(
                edges,
                min_position=int(params["min_position"]),
            )
        elif dim_name == "time_since_pair_last_edge":
            columns[dim_name] = compute_time_since_pair_last_edge(
                edges,
                burst_seconds=float(params["burst_seconds"]),
                dormant_seconds=float(params["dormant_seconds"]),
            )
        elif dim_name == "pair_amount_zscore":
            columns[dim_name] = compute_pair_amount_zscore(
                edges,
                cv_threshold=float(params["cv_threshold"]),
                min_count=int(params["min_count"]),
            )
        elif dim_name == "find_motif_structuring":
            columns[dim_name] = compute_find_motif_structuring(
                edges,
                time_window_hours=float(params["time_window_hours"]),
                amt1_min=float(params["amt1_min"]),
                amt2_max=float(params["amt2_max"]),
            )
        elif dim_name == "edge_curvature_frc":
            columns[dim_name] = compute_edge_curvature_frc(edges)
        else:
            raise ValueError(
                f"unknown edge dimension: {dim_name!r}; valid: {sorted(EDGE_DIM_KINDS)}",
            )
    return pa.table(columns)
