# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
"""Temporal bucket materialiser for fdr_temporal_hierarchy.

Computes a per-anchor-entity coarse time-bucket label from the entity's
incident event timestamps. Used by the builder when a Pattern declares
`fdr_temporal_hierarchy:` with a slice_dimension that does not yet exist
as a column on the geometry table.
"""
from __future__ import annotations

import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["materialise_temporal_bucket", "parse_bucket_duration"]

_DURATION_RE = re.compile(r"^(\d+)([dh])$")
_UNIT_SECONDS = {"d": 86400, "h": 3600}


def parse_bucket_duration(duration: str) -> int:
    """Parse '90d' / '24h' -> total seconds.

    Raises ValueError for unknown units or bad format.
    """
    m = _DURATION_RE.match(duration)
    if m is None:
        raise ValueError(
            f"bucket duration must match '<int><d|h>', got {duration!r}",
        )
    n, unit = int(m.group(1)), m.group(2)
    return n * _UNIT_SECONDS[unit]


def materialise_temporal_bucket(
    *,
    event_table: pa.Table,
    anchor_keys: list[str],
    anchor_key_col_options: tuple[str, ...],
    timestamp_col: str,
    bucket: str = "90d",
) -> pa.Table:
    """Compute per-anchor temporal_bucket label from event timestamps.

    For each anchor entity, gather every event where ANY column in
    ``anchor_key_col_options`` matches the entity's primary key, take the
    median timestamp ('centroid'), and bucket it to ``bucket``-aligned
    string label of the form ``"b<bucket_index>"``.

    Args:
        event_table: Arrow table with timestamp_col + anchor_key_col_options.
        anchor_keys: anchor primary keys to materialise buckets for.
        anchor_key_col_options: column names that may reference an anchor
            (e.g. ('from_account', 'to_account')).
        timestamp_col: column carrying the event timestamp
            (timestamp[us, tz=UTC] expected).
        bucket: bucket duration string (default '90d').

    Returns:
        Arrow table with primary_key (string) and temporal_bucket (string,
        null when no events reference the anchor).
    """
    bucket_seconds = parse_bucket_duration(bucket)

    # Normalise timestamp column to PyArrow timestamp dtype. Some sources store
    # timestamps as strings (e.g. AML transactions use '%Y/%m/%d %H:%M'); we try
    # a few common formats before giving up.
    ts_col = event_table[timestamp_col]
    if pa.types.is_string(ts_col.type) or pa.types.is_large_string(ts_col.type):
        last_err: Exception | None = None
        for fmt in (
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                ts_col = pc.strptime(ts_col, format=fmt, unit="us")
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise ValueError(
                f"could not parse string timestamp column "
                f"{timestamp_col!r}; tried %Y/%m/%d %H:%M, %Y/%m/%d %H:%M:%S, "
                f"%Y-%m-%d %H:%M:%S, %Y-%m-%dT%H:%M:%S — last error: {last_err}",
            )

    # Convert timestamps to unix seconds (int64). PyArrow's timestamp cast to
    # int64 returns microseconds (or whatever unit was set); divide by 1e6 for
    # seconds.
    ts_int = pc.cast(
        pc.divide(
            pc.cast(ts_col, pa.int64()),
            pa.scalar(1_000_000, type=pa.int64()),
        ),
        pa.int64(),
    )

    # Union over anchor_key_col_options -> long-format (anchor_key, ts)
    pieces: list[pa.Table] = []
    for col in anchor_key_col_options:
        if col not in event_table.column_names:
            continue
        pieces.append(pa.table({"anchor_key": event_table[col], "ts": ts_int}))
    if not pieces:
        return pa.table({
            "primary_key": anchor_keys,
            "temporal_bucket": pa.array(
                [None] * len(anchor_keys), type=pa.string()),
        })
    long = pa.concat_tables(pieces)

    # Per-anchor median timestamp — PyArrow groupby doesn't expose median;
    # collect into per-anchor lists and use numpy.
    grouped = long.group_by(["anchor_key"]).aggregate([("ts", "list")])
    anchor_keys_seen = grouped["anchor_key"].to_pylist()
    ts_lists = grouped["ts_list"].to_pylist()

    centroid_by_anchor: dict[str, int] = {}
    for k, lst in zip(anchor_keys_seen, ts_lists, strict=True):
        if not lst:
            continue
        centroid_by_anchor[k] = int(np.median(np.asarray(lst, dtype=np.int64)))

    out_keys: list[str] = []
    out_buckets: list[str | None] = []
    for k in anchor_keys:
        c = centroid_by_anchor.get(k)
        if c is None:
            out_keys.append(k)
            out_buckets.append(None)
            continue
        bucket_idx = c // bucket_seconds
        out_keys.append(k)
        out_buckets.append(f"b{bucket_idx}")
    return pa.table({
        "primary_key": out_keys,
        "temporal_bucket": pa.array(out_buckets, type=pa.string()),
    })
