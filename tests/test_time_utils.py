# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Bulk timestamp → epoch-seconds conversion."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from hypertopos.utils.time import to_epoch_seconds

MIDNIGHT = 1661990400.0  # 2022-09-01T00:00:00Z


def _epochs(values: list[object]) -> list[float | None]:
    return to_epoch_seconds(pa.array(values)).to_pylist()


def test_aml_slash_minutes_format() -> None:
    assert _epochs(["2022/09/01 00:08"]) == [MIDNIGHT + 480]


def test_iso_z_and_offset() -> None:
    assert _epochs(["2022-09-01T00:08:00Z", "2022-09-01T02:08:00+02:00"]) == [
        MIDNIGHT + 480,
        MIDNIGHT + 480,
    ]


def test_date_only_and_dash_seconds() -> None:
    assert _epochs(["2022-09-01", "2022-09-01 00:08:30"]) == [MIDNIGHT, MIDNIGHT + 510]


def test_null_stays_null() -> None:
    assert _epochs(["2022/09/01 00:08", None]) == [MIDNIGHT + 480, None]


def test_mixed_formats_parse_per_row() -> None:
    assert _epochs(["2022/09/01 00:08", "2022-09-01T00:08:00Z"]) == [
        MIDNIGHT + 480,
        MIDNIGHT + 480,
    ]


def test_naive_is_utc() -> None:
    # Expectation is an explicit UTC epoch, independent of the machine timezone.
    assert _epochs(["2022-09-01 00:00:00"]) == [MIDNIGHT]


def test_timestamp_type_and_numeric_passthrough() -> None:
    aware = pa.array([datetime(2022, 9, 1, tzinfo=UTC)])
    assert to_epoch_seconds(aware).to_pylist() == [MIDNIGHT]
    assert to_epoch_seconds(pa.array([1661990880])).to_pylist() == [MIDNIGHT + 480]


def test_chunked_input() -> None:
    chunked = pa.chunked_array([["2022/09/01 00:08"], ["2022-09-01"]])
    assert to_epoch_seconds(chunked).to_pylist() == [MIDNIGHT + 480, MIDNIGHT]


def test_unparseable_raises_with_example() -> None:
    with pytest.raises(ValueError, match="not a date"):
        to_epoch_seconds(pa.array(["2022-09-01", "not a date"]))


def test_date_columns() -> None:
    day = datetime(2022, 9, 1).date()
    assert to_epoch_seconds(pa.array([day], pa.date32())).to_pylist() == [MIDNIGHT]
    assert to_epoch_seconds(pa.array([day], pa.date64())).to_pylist() == [MIDNIGHT]


def test_dictionary_encoded_strings() -> None:
    values = pa.array(["2022/09/01 00:08", "2022/09/01 00:08"]).dictionary_encode()
    assert to_epoch_seconds(values).to_pylist() == [MIDNIGHT + 480] * 2


def test_blank_strings_are_null() -> None:
    assert _epochs(["2022/09/01 00:08", "", "  "]) == [MIDNIGHT + 480, None, None]


@pytest.mark.parametrize("value", ["2022-02-30", "2023-02-29", "2022/04/31 10:00"])
def test_nonexistent_day_is_rejected_not_rolled_over(value: str) -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        to_epoch_seconds(pa.array([value]))
    assert to_epoch_seconds(pa.array([value]), errors="null").to_pylist() == [None]


def test_surrounding_whitespace_is_ignored() -> None:
    assert _epochs([" 2022-09-01", "2022/09/01 00:08\t"]) == [MIDNIGHT, MIDNIGHT + 480]


def test_sub_microsecond_fractions_truncate_to_microseconds() -> None:
    assert _epochs(["2022-09-01T00:00:00.123456789", "2022-09-01T00:00:00.123456789Z"]) == [
        MIDNIGHT + 0.123456,
        MIDNIGHT + 0.123456,
    ]


def test_errors_null_keeps_parseable_values() -> None:
    out = to_epoch_seconds(pa.array(["2022-09-01", "not a date"]), errors="null")
    assert out.to_pylist() == [MIDNIGHT, None]


def test_single_digit_month_and_day_are_vectorised() -> None:
    # Relative to a zero-padded column of the same size: a wall-clock limit
    # fails under parallel test load, the ratio does not.
    clean = pa.array(["2022-09-01 00:08"] * 200_000)
    started = time.perf_counter()
    to_epoch_seconds(clean)
    baseline = time.perf_counter() - started
    values = pa.array(["2022-9-1 00:08"] * 200_000)
    started = time.perf_counter()
    out = to_epoch_seconds(values)
    assert time.perf_counter() - started < 20 * max(baseline, 0.05)
    assert out[0].as_py() == MIDNIGHT + 480


def test_one_malformed_value_stays_fast() -> None:
    # Relative to a clean column of the same size: wall-clock limits fail under
    # parallel test load, the ratio does not. A per-row fallback would be ~100x.
    clean = pa.array(["2022-09-01 00:08"] * 200_001)
    started = time.perf_counter()
    to_epoch_seconds(clean, errors="null")
    baseline = time.perf_counter() - started
    values = pa.array(["2022-09-01T00:08:00"] * 200_000 + ["not a date"])
    started = time.perf_counter()
    out = to_epoch_seconds(values, errors="null")
    with pytest.raises(ValueError, match="not a date"):
        to_epoch_seconds(values)
    assert time.perf_counter() - started < 20 * max(baseline, 0.05)
    assert out[0].as_py() == MIDNIGHT + 480
    assert out[-1].as_py() is None


def test_large_column_is_vectorised() -> None:
    values = pa.array(["2022/09/01 00:08"] * 200_000)
    started = time.perf_counter()
    out = to_epoch_seconds(values)
    assert time.perf_counter() - started < 5.0
    assert out[0].as_py() == MIDNIGHT + 480
