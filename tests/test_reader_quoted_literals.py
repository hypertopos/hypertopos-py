# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Keys and filter values with apostrophes reach Lance filters as literals."""

from __future__ import annotations

import pyarrow as pa

from hypertopos.builder.builder import GDSBuilder
from hypertopos.storage.reader import GDSReader


def _sphere(tmp_path):
    builder = GDSBuilder("quotes", str(tmp_path / "sphere"))
    builder.add_line(
        "people",
        pa.table({"primary_key": ["O'Brien", "Smith"], "name": ["D'Arcy", "Jane"]}),
        key_col="primary_key",
        source_id="t",
        role="anchor",
    )
    builder.add_pattern("p", "anchor", "people", relations=[], tracked_properties=["name"])
    return builder.build()


def test_point_key_with_apostrophe_is_found(tmp_path) -> None:
    reader = GDSReader(_sphere(tmp_path))
    rows = reader.read_points("people", 1, primary_key="O'Brien")
    assert rows["primary_key"].to_pylist() == ["O'Brien"]


def test_string_filter_value_with_apostrophe_matches(tmp_path) -> None:
    reader = GDSReader(_sphere(tmp_path))
    rows = reader.read_points("people", 1, filters={"name": "D'Arcy"})
    assert rows["primary_key"].to_pylist() == ["O'Brien"]
