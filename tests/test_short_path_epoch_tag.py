# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The first calibration epoch is tagged even when the sphere path is a Windows 8.3 short path.

Lance percent-encodes the ``~`` of a short path segment (``KAROL~1.KED``) when
it creates the ``_refs/tags`` directory, so the tag write fails. Paths under the
default Windows temp directory are commonly in short form.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from hypertopos.builder._writer import _tag_initial_epoch

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="8.3 short paths are Windows-only")


def _short_path(path: Path) -> Path:
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    return Path(buf.value) if n else path


def test_initial_epoch_tag_on_short_path(tmp_path: Path) -> None:
    long_dir = tmp_path / "a_long_directory_name"
    long_dir.mkdir()
    short_dir = _short_path(long_dir)
    if "~" not in str(short_dir):
        pytest.skip("8.3 short names are disabled on this volume")

    lance_path = short_dir / "data.lance"
    lance.write_dataset(pa.table({"k": [1, 2, 3]}), str(lance_path))

    _tag_initial_epoch(lance_path)

    assert "epoch_1" in lance.dataset(str(long_dir / "data.lance")).tags.list()
