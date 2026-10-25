# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Timestamp handling in temporal bucketing and windowed derived dimensions."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa

from hypertopos.builder.builder import GDSBuilder, _bucket_timestamps_us
from hypertopos.builder.derived import _apply_temporal_windowing


def _events() -> pa.Table:
    return pa.table(
        {
            "primary_key": ["e1", "e2", "e3", "e4"],
            "account": ["a", "a", "b", "a"],
            "ts": ["2022/09/01 00:08", None, "2022/09/02 10:00", "2022/09/03 00:00"],
        }
    )


def test_event_buckets_drop_rows_without_timestamp() -> None:
    table, buckets, n_buckets, min_ts, _ = GDSBuilder._parse_event_buckets(_events(), "ts", "1d")
    assert table["primary_key"].to_pylist() == ["e1", "e3", "e4"]
    assert buckets.tolist() == [0, 1, 1]
    assert n_buckets == 2
    assert min_ts == 1661990880.0


def test_windowing_last_ignores_rows_without_timestamp() -> None:
    values, _ = _apply_temporal_windowing(
        _events(),
        pa.array(["a", "b"]),
        "account",
        "count",
        None,
        "ts",
        "1d",
        "last",
        "auto",
        99.0,
    )
    assert values.tolist() == [1.0, 1.0]


def test_bucket_timestamps_match_datetime_rounding() -> None:
    min_ts, win = 1661990880.123456, 1.0 / 3.0
    buckets = np.arange(300)
    as_datetimes = [datetime.fromtimestamp(min_ts + int(b) * win, tz=UTC) for b in buckets]
    expected = pa.array(as_datetimes, pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_pylist()
    assert _bucket_timestamps_us(min_ts, win, buckets).tolist() == expected
