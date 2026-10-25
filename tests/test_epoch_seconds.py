# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""String timestamps in the edge table convert to UTC epoch seconds without a timezone pass."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from hypertopos.builder.builder import GDSBuilder


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2022/09/01 00:20", 1661991600.0),
        ("2022-09-01 00:20:30", 1661991630.0),
        ("2022-09-01", 1661990400.0),
    ],
)
def test_string_timestamps_convert_to_utc_epoch_seconds(value: str, expected: float) -> None:
    out = GDSBuilder._to_epoch_seconds(pa.array([value, value]))
    assert out.to_pylist() == [expected, expected]


def test_string_conversion_skips_the_timezone_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(pc, "assume_timezone", lambda *a, **k: calls.append(1))
    GDSBuilder._to_epoch_seconds(pa.array(["2022/09/01 00:20"]))
    assert calls == [], "assume_timezone(UTC) leaves epoch values unchanged and is slow"
