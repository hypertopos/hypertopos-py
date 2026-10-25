# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Temporal timestamp_from / timestamp_to honour a UTC offset in the bound."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa

from hypertopos.storage.reader import GDSReader


def _table() -> pa.Table:
    hours = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(6)]
    return pa.table(
        {"primary_key": ["A"] * 6, "timestamp": pa.array(hours, type=pa.timestamp("us", tz="UTC"))}
    )


def _hours(table: pa.Table) -> list[int]:
    return [t.hour for t in table["timestamp"].to_pylist()]


def test_offset_bounds_are_converted_not_relabelled(tmp_path) -> None:
    reader = GDSReader(str(tmp_path))
    # 05:00+02:00 is 03:00 UTC; 02:00 is excluded, 03:00 included.
    out = reader._apply_temporal_filters(_table(), {"timestamp_from": "2024-01-01T05:00:00+02:00"})
    assert _hours(out) == [3, 4, 5]
    out = reader._apply_temporal_filters(_table(), {"timestamp_to": "2024-01-01T05:00:00+02:00"})
    assert _hours(out) == [0, 1, 2]


def test_naive_bounds_stay_utc(tmp_path) -> None:
    reader = GDSReader(str(tmp_path))
    out = reader._apply_temporal_filters(
        _table(), {"timestamp_from": "2024-01-01T02:00:00", "timestamp_to": "2024-01-01T04:00:00"}
    )
    assert _hours(out) == [2, 3]
