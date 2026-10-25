# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The lazy primary_key index build on points datasets."""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa

from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.reader import GDSReader


def _points(tmp_path: Path, with_index: bool) -> Path:
    path = tmp_path / "s" / "points" / "acc" / "v=1" / "data.lance"
    lance.write_dataset(pa.table({"primary_key": ["a", "b", "c"], "x": [1, 2, 3]}), str(path))
    if with_index:
        lance.dataset(str(path)).create_scalar_index("primary_key", "BTREE")
    return path


def test_existing_btree_is_not_rebuilt(tmp_path: Path) -> None:
    path = _points(tmp_path, with_index=True)
    before = lance.dataset(str(path)).latest_version
    for _ in range(3):
        GDSReader(str(tmp_path / "s"))._ensure_points_btree(lance.dataset(str(path)), "acc", 1)
    assert lance.dataset(str(path)).latest_version == before


def test_missing_btree_is_built_once_on_main(tmp_path: Path) -> None:
    path = _points(tmp_path, with_index=False)
    GDSReader(str(tmp_path / "s"))._ensure_points_btree(lance.dataset(str(path)), "acc", 1)
    built = lance.dataset(str(path)).latest_version
    GDSReader(str(tmp_path / "s"))._ensure_points_btree(lance.dataset(str(path)), "acc", 1)
    assert lance.dataset(str(path)).latest_version == built
    names = {i.name for i in lance.dataset(str(path)).describe_indices()}
    assert any("primary_key" in n for n in names)


def test_branch_reader_never_writes_main_points(tmp_path: Path, monkeypatch) -> None:
    path = _points(tmp_path, with_index=False)
    before = lance.dataset(str(path)).latest_version
    reader = GDSReader(str(tmp_path / "s"))
    reader.branch = "exp"  # a branch reader; the build must not commit to main
    monkeypatch.setattr(lb, "overlay_dir", lambda base, name: Path(base))
    reader._ensure_points_btree(lance.dataset(str(path)), "acc", 1)
    assert lance.dataset(str(path)).latest_version == before
