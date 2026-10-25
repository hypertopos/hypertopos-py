# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Read paths open Lance datasets through one shared ``lance.Session``.

Without a shared session every ``lance.dataset()`` call starts with cold
index and metadata caches, so a filtered count on an indexed column reloads
the index pages on each call.
"""

from __future__ import annotations

import re
from pathlib import Path

import lance
import pyarrow as pa

from hypertopos.storage._lance_session import open_dataset

PACKAGE = Path(__file__).resolve().parents[1] / "hypertopos"
READ_PATH_MODULES = (
    "storage/reader.py",
    "storage/topology_cache.py",
    "engine/lance_sql_agg.py",
    "navigation/navigator.py",
    "sphere.py",
)
RAW_OPEN = re.compile(r"\b_?lance(?:_local)?\.dataset\(")


def _write(path: Path, table: pa.Table, mode: str = "create") -> None:
    lance.write_dataset(table, str(path), mode=mode)


def test_opens_share_one_session(tmp_path: Path) -> None:
    _write(tmp_path / "a.lance", pa.table({"k": [1, 2]}))
    _write(tmp_path / "b.lance", pa.table({"k": [3]}))
    a = open_dataset(tmp_path / "a.lance")
    b = open_dataset(str(tmp_path / "b.lance"))
    assert a.session().is_same_as(b.session())
    assert not lance.dataset(str(tmp_path / "a.lance")).session().is_same_as(a.session())


def test_shared_session_sees_later_writes(tmp_path: Path) -> None:
    path = tmp_path / "t.lance"
    _write(path, pa.table({"k": [1, 2, 3], "f": [True, False, True]}))
    open_dataset(path).create_scalar_index("f", "BITMAP")
    assert open_dataset(path).count_rows(filter="f = true") == 2

    _write(path, pa.table({"k": [4], "f": [True]}), mode="append")
    after_append = open_dataset(path)
    assert after_append.count_rows() == 4
    assert after_append.count_rows(filter="f = true") == 3

    _write(path, pa.table({"k": [9], "f": [False]}), mode="overwrite")
    assert open_dataset(path).count_rows() == 1
    assert open_dataset(path, version=after_append.version).count_rows() == 4


def test_read_paths_do_not_open_datasets_directly() -> None:
    offenders = {
        rel: n
        for rel in READ_PATH_MODULES
        if (n := len(RAW_OPEN.findall((PACKAGE / rel).read_text(encoding="utf-8"))))
    }
    assert not offenders, (
        f"raw lance.dataset() on a read path (file: count): {offenders} — "
        "use hypertopos.storage._lance_session.open_dataset"
    )
