"""Edge-derived dimension catalog for event patterns.

Five build-time dimensions. Each takes the event pattern's edge_table
(PyArrow) and returns one float32 array of length ``edges.num_rows``.
The orchestrator :func:`compute_all_edge_dims` runs all dims listed in
a config dict and returns a single Arrow table keyed by ``event_key``
for sidecar persistence.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pyarrow as pa

EDGE_DIM_KINDS: dict[str, str] = {
    "pair_edge_count":           "poisson",
    "position_in_chain":         "poisson",
    "time_since_pair_last_edge": "gaussian",
    "pair_amount_zscore":        "gaussian",
    "find_motif_structuring":    "bernoulli",
}


def compute_pair_edge_count(edges: pa.Table) -> pa.Array:
    pair_counts = (
        edges.group_by(["from_key", "to_key"])
             .aggregate([("event_key", "count")])
    )
    fks = pair_counts["from_key"].to_pylist()
    tks = pair_counts["to_key"].to_pylist()
    cts = pair_counts["event_key_count"].to_pylist()
    lookup: dict[tuple[str, str], int] = {
        (fk, tk): ct for fk, tk, ct in zip(fks, tks, cts, strict=False)
    }
    edge_fks = edges["from_key"].to_pylist()
    edge_tks = edges["to_key"].to_pylist()
    counts = np.array(
        [lookup[(fk, tk)] for fk, tk in zip(edge_fks, edge_tks, strict=False)],
        dtype=np.float32,
    )
    return pa.array(counts, type=pa.float32())


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
        pos >= min_position, pos.astype(np.float32), np.float32(0.0),
    )
    return pa.array(out, type=pa.float32())


def compute_time_since_pair_last_edge(
    edges: pa.Table,
    *,
    burst_seconds: float,  # noqa: ARG001 — kept for YAML parity
    dormant_seconds: float,
) -> pa.Array:
    fks = edges["from_key"].to_pylist()
    tks = edges["to_key"].to_pylist()
    tss = edges["timestamp"].to_pylist()
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for i, (fk, tk, ts) in enumerate(zip(fks, tks, tss, strict=False)):
        groups[(fk, tk)].append((i, ts))
    out = np.empty(edges.num_rows, dtype=np.float32)
    for items in groups.values():
        items.sort(key=lambda x: x[1])
        prev_ts: float | None = None
        for (idx, ts) in items:
            out[idx] = (
                np.float32(dormant_seconds) if prev_ts is None
                else np.float32(ts - prev_ts)
            )
            prev_ts = ts
    return pa.array(out, type=pa.float32())


def compute_pair_amount_zscore(
    edges: pa.Table,
    *,
    cv_threshold: float,
    min_count: int,
) -> pa.Array:
    fks = edges["from_key"].to_pylist()
    tks = edges["to_key"].to_pylist()
    amts = edges["amount"].to_pylist()
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for i, (fk, tk, amt) in enumerate(zip(fks, tks, amts, strict=False)):
        groups[(fk, tk)].append((i, float(amt) if amt is not None else 0.0))
    out = np.zeros(edges.num_rows, dtype=np.float32)
    for items in groups.values():
        if len(items) < min_count:
            continue
        amounts = np.array([a for (_, a) in items], dtype=np.float64)
        mean = amounts.mean()
        std = amounts.std(ddof=0)
        if mean == 0.0 or std == 0.0:
            continue
        cv = std / abs(mean)
        if cv >= cv_threshold:
            continue
        for (idx, a) in items:
            out[idx] = np.float32((a - mean) / std)
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
    eks = edges["event_key"].to_pylist()
    out = np.array(
        [1.0 if ek in motif_keys else 0.0 for ek in eks],
        dtype=np.float32,
    )
    return pa.array(out, type=pa.float32())


AGGREGATE_NAMES: tuple[str, ...] = ("mean", "max")


def aggregate_kind(source_kind: str, agg: str) -> str:
    if agg == "mean":
        return "gaussian"
    if agg == "max":
        return "bernoulli" if source_kind == "bernoulli" else "gaussian"
    raise ValueError(f"unknown aggregate: {agg!r}")


def aggregate_edge_dims_for_anchor(
    *,
    anchor_keys: list[str],
    edges: pa.Table,
    sidecar: pa.Table,
    dims: list[str],
    anchor_kind: str,
    pair_separator: str = "→",
) -> pa.Table:
    if anchor_kind == "chain":
        raise NotImplementedError(
            "anchor_kind='chain' aggregation ships in 0.6.2 — chain "
            "membership lookup is not yet exposed at the polygon-build site",
        )
    if anchor_kind not in ("single", "pair"):
        raise ValueError(
            f"anchor_kind must be one of 'single' / 'pair' / 'chain'; "
            f"got {anchor_kind!r}",
        )
    for d in dims:
        if d not in EDGE_DIM_KINDS:
            raise ValueError(
                f"unknown edge dimension: {d!r}; "
                f"valid: {sorted(EDGE_DIM_KINDS)}",
            )
        if d not in sidecar.column_names:
            raise ValueError(
                f"edge dimension {d!r} not present in sidecar; "
                f"available: {sorted(set(sidecar.column_names) - {'event_key'})}",
            )

    n_anchors = len(anchor_keys)
    out_cols: dict[str, pa.Array] = {
        "primary_key": pa.array(anchor_keys, type=pa.string()),
    }

    if edges.num_rows == 0 or sidecar.num_rows == 0:
        for d in dims:
            for agg in AGGREGATE_NAMES:
                out_cols[f"{d}_{agg}"] = pa.array(
                    np.zeros(n_anchors, dtype=np.float32),
                )
        return pa.table(out_cols)

    joined = edges.join(sidecar, keys="event_key")

    if anchor_kind == "single":
        from_pk_arr = joined["from_key"].combine_chunks()
        to_pk_arr   = joined["to_key"].combine_chunks()
        anchor_pk_arr = pa.concat_arrays([from_pk_arr, to_pk_arr])
        dim_arrays: dict[str, pa.Array] = {}
        for d in dims:
            v = joined[d].combine_chunks()
            dim_arrays[d] = pa.concat_arrays([v, v])
    else:
        from_pks = joined["from_key"].combine_chunks().to_pylist()
        to_pks   = joined["to_key"].combine_chunks().to_pylist()
        anchor_pk_arr = pa.array(
            [
                f"{f}{pair_separator}{t}"
                for f, t in zip(from_pks, to_pks, strict=False)
            ],
            type=pa.string(),
        )
        dim_arrays = {
            d: joined[d].combine_chunks() for d in dims
        }

    base = pa.table({"primary_key": anchor_pk_arr, **dim_arrays})
    aggs: list[tuple[str, str]] = []
    for d in dims:
        aggs.append((d, "mean"))
        aggs.append((d, "max"))
    grouped = base.group_by("primary_key").aggregate(aggs)
    pk_to_idx = {
        pk: i for i, pk in enumerate(grouped["primary_key"].to_pylist())
    }

    for d in dims:
        for agg in AGGREGATE_NAMES:
            col_name = f"{d}_{agg}"
            grouped_col = f"{d}_{agg}"
            buf = np.zeros(n_anchors, dtype=np.float32)
            for i, pk in enumerate(anchor_keys):
                gi = pk_to_idx.get(pk)
                if gi is not None:
                    val = grouped[grouped_col][gi].as_py()
                    if val is not None and not (val != val):
                        buf[i] = float(val)
            out_cols[col_name] = pa.array(buf)
    return pa.table(out_cols)


def compute_all_edge_dims(
    edges: pa.Table, config: dict[str, dict[str, Any]],
) -> pa.Table:
    """Run each declared dim, return Arrow table keyed by event_key."""
    columns: dict[str, pa.Array] = {"event_key": edges["event_key"]}
    for dim_name, params in config.items():
        if dim_name == "pair_edge_count":
            columns[dim_name] = compute_pair_edge_count(edges)
        elif dim_name == "position_in_chain":
            columns[dim_name] = compute_position_in_chain(
                edges, min_position=int(params["min_position"]),
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
        else:
            raise ValueError(
                f"unknown edge dimension: {dim_name!r}; "
                f"valid: {sorted(EDGE_DIM_KINDS)}",
            )
    return pa.table(columns)
