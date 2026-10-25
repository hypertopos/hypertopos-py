# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Every Lance read in GDSReader goes through one resolver that honours the session pin.

A session pins geometry, temporal and edge versions when it opens; reads that
opened the dataset by path saw rows written after that (nearest-neighbour
search, entity-key lookups, temporal history, the edge-count fallback, the
geometry schema).
"""

from __future__ import annotations

import ast
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from hypertopos.storage.reader import GDSReader

PACKAGE = Path(__file__).resolve().parents[1] / "hypertopos"
PID = "p"


def _geometry(keys: list[str], vec: float) -> pa.Table:
    n = len(keys)
    return pa.table(
        {
            "primary_key": pa.array(keys),
            "delta": pa.FixedSizeListArray.from_arrays(
                pa.array(np.full(n * 2, vec, dtype=np.float32)), 2
            ),
            "entity_keys": pa.array([["A1", "B1"]] * n, type=pa.list_(pa.string())),
        }
    )


def _pinned_reader(tmp_path: Path, sub: str, first: pa.Table, second: pa.Table) -> GDSReader:
    """A reader pinned to v1 of base/<sub>/p/data.lance, with v2 appended after the pin."""
    path = tmp_path / sub / PID / "data.lance"
    path.parent.mkdir(parents=True)
    lance.write_dataset(first, str(path))
    key = PID if sub == "geometry" else f"{sub}:{PID}"
    reader = GDSReader(str(tmp_path))
    reader._pinned_lance_versions = {key: lance.dataset(str(path)).version}
    lance.write_dataset(second, str(path), mode="append")
    return reader


def test_find_nearest_honours_pin(tmp_path):
    reader = _pinned_reader(
        tmp_path, "geometry", _geometry(["K1", "K2"], 1.0), _geometry(["NEW"], 1.0)
    )
    hits = reader.find_nearest_lance(PID, 1, np.array([1.0, 1.0], dtype=np.float32), k=10)
    assert {k for k, _ in hits} == {"K1", "K2"}


def test_resolve_primary_keys_by_edge_honours_pin(tmp_path):
    reader = _pinned_reader(
        tmp_path, "geometry", _geometry(["K1", "K2"], 1.0), _geometry(["NEW"], 2.0)
    )
    assert sorted(reader.resolve_primary_keys_by_edge(PID, 1, None, "A1")) == ["K1", "K2"]


def test_geometry_column_names_honours_pin(tmp_path):
    path = tmp_path / "geometry" / PID / "data.lance"
    path.parent.mkdir(parents=True)
    lance.write_dataset(_geometry(["K1"], 1.0), str(path))
    reader = GDSReader(str(tmp_path))
    reader._pinned_lance_versions = {PID: lance.dataset(str(path)).version}
    lance.dataset(str(path)).add_columns({"extra": "1"})
    assert "extra" not in reader.geometry_column_names(PID, 1)


def test_read_temporal_honours_pin(tmp_path):
    slice_ = pa.table({"primary_key": ["A"], "slice_index": [0]})
    later = pa.table({"primary_key": ["A"], "slice_index": [1]})
    reader = _pinned_reader(tmp_path, "temporal", slice_, later)
    table = reader._read_lance_temporal(
        tmp_path / "temporal" / PID / "data.lance", "A", None, key=f"temporal:{PID}"
    )
    assert table["slice_index"].to_pylist() == [0]


def test_edge_table_stats_fallback_honours_pin(tmp_path, monkeypatch):
    edges = pa.table({"from_key": ["a"], "to_key": ["b"], "timestamp": [1.0], "amount": [1.0]})
    reader = _pinned_reader(tmp_path, "edges", edges, edges)
    monkeypatch.setattr(reader, "edge_stats_cached", lambda pid: None)
    assert reader.edge_table_stats(PID)["row_count"] == 1


def test_every_reader_open_goes_through_open():
    """Ratchet: open_dataset( in reader.py only inside _open and the version probes."""
    tree = ast.parse((PACKAGE / "storage" / "reader.py").read_text(encoding="utf-8"))
    allowed = {"_open", "latest_lance_versions", "_branch_versions"}
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name in allowed:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open_dataset":
                offenders.append(f"{fn.name}:{node.lineno}")
    assert not offenders, f"open_dataset outside GDSReader._open: {offenders}"


@pytest.mark.parametrize("explicit", [True, False])
def test_open_prefers_an_explicit_version(tmp_path, explicit):
    path = tmp_path / "geometry" / PID / "data.lance"
    path.parent.mkdir(parents=True)
    lance.write_dataset(_geometry(["K1"], 1.0), str(path))
    v1 = lance.dataset(str(path)).version
    lance.write_dataset(_geometry(["K2"], 1.0), str(path), mode="append")
    reader = GDSReader(str(tmp_path))
    ds = reader._open(None, path, version=v1 if explicit else None)
    assert ds.count_rows() == (1 if explicit else 2)
