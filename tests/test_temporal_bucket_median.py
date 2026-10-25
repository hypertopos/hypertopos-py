# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""materialise_temporal_bucket: per-anchor median timestamp -> bucket label."""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa

from hypertopos.builder.temporal_bucket import _sorted_by_group, materialise_temporal_bucket


def _reference(from_keys, to_keys, ts, anchor_keys, bucket_seconds):
    by_key: dict[str, list[int]] = {}
    for f, t, s in zip(from_keys, to_keys, ts, strict=True):
        for k in (f, t):
            if k is not None:
                by_key.setdefault(k, []).append(s)
    out = []
    for k in anchor_keys:
        lst = by_key.get(k)
        out.append(None if not lst else f"b{int(np.median(np.asarray(lst))) // bucket_seconds}")
    return out


def test_buckets_match_the_per_anchor_median() -> None:
    rng = random.Random(3)
    n, keys = 4000, [f"A{i}" for i in range(500)]
    from_keys = [rng.choice(keys) for _ in range(n)]
    to_keys = [rng.choice(keys) if rng.random() > 0.1 else None for _ in range(n)]
    # Wide range so medians of even-sized groups land on half-seconds and bucket edges.
    ts = [rng.randrange(-5 * 86400, 400 * 86400) for _ in range(n)]
    anchor_keys = keys + ["MISSING-1", "MISSING-2"]
    table = pa.table(
        {
            "from_account": from_keys,
            "to_account": to_keys,
            "ts": pa.array([s * 1_000_000 for s in ts], type=pa.timestamp("us", tz="UTC")),
        }
    )
    out = materialise_temporal_bucket(
        event_table=table,
        anchor_keys=anchor_keys,
        anchor_key_col_options=("from_account", "to_account"),
        timestamp_col="ts",
        bucket="7d",
    )
    assert out["primary_key"].to_pylist() == anchor_keys
    expected = _reference(from_keys, to_keys, ts, anchor_keys, 7 * 86400)
    assert out["temporal_bucket"].to_pylist() == expected
    assert expected[-2:] == [None, None]


def test_group_sort_handles_spans_too_wide_for_one_key() -> None:
    codes = np.array([1, 0, 1, 0, 2], dtype=np.int64)
    ts = np.array([2**62, -(2**62), 5, 7, 0], dtype=np.int64)
    assert _sorted_by_group(codes, ts, 3).tolist() == [-(2**62), 7, 5, 2**62, 0]
    narrow = np.array([30, 10, 20, 40, 0], dtype=np.int64)
    assert _sorted_by_group(codes, narrow, 3).tolist() == [10, 40, 20, 30, 0]
