# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""goto reads only the requested point, not the whole line."""

from __future__ import annotations

import pyarrow as pa
import pytest

from hypertopos.builder.builder import GDSBuilder
from hypertopos.navigation.navigator import GDSEntityNotFoundError
from hypertopos.sphere import HyperSphere


@pytest.fixture
def navigator(tmp_path):
    builder = GDSBuilder("goto", str(tmp_path / "sphere"))
    builder.add_line(
        "people",
        pa.table(
            {"primary_key": [f"P{i}" for i in range(50)], "name": [f"n{i}" for i in range(50)]}
        ),
        key_col="primary_key",
        source_id="t",
        role="anchor",
    )
    builder.add_pattern("p", "anchor", "people", relations=[], tracked_properties=["name"])
    sphere = HyperSphere.open(builder.build())
    return sphere.session("goto-test").navigator()


def test_goto_reads_by_key(navigator, monkeypatch) -> None:
    calls = []
    storage = navigator._storage
    real = storage.read_points

    def spy(line_id, version, *args, **kwargs):
        calls.append(kwargs.get("primary_key"))
        return real(line_id, version, *args, **kwargs)

    monkeypatch.setattr(storage, "read_points", spy)
    navigator.goto("P7", "people")
    assert navigator.position.primary_key == "P7"
    assert calls == ["P7"], "goto must ask storage for the one key, not the whole line"


def test_goto_unknown_key_raises(navigator) -> None:
    with pytest.raises(GDSEntityNotFoundError):
        navigator.goto("NOPE", "people")
