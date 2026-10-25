# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Metadata JSON files are replaced atomically, never written in place.

Sessions read geometry stats, edge stats and the calibration tracker while a
writer may be replacing them; an in-place write can be read half-written.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hypertopos.engine.calibration import CalibrationTracker
from hypertopos.storage import writer as writer_module
from hypertopos.storage.writer import GDSWriter


def test_stats_files_are_replaced_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[Path] = []
    real_replace = os.replace

    def recording_replace(src: str | Path, dst: str | Path) -> None:
        replaced.append(Path(dst))
        real_replace(src, dst)

    monkeypatch.setattr(writer_module.os, "replace", recording_replace)
    writer = GDSWriter(str(tmp_path))
    meta = tmp_path / "_gds_meta"

    writer.write_geometry_stats("p", 1, np.array([1.0, 2.0, 3.0]), theta_norm=2.5)
    writer.write_edge_stats("p", {"n_edges": 3})
    tracker = CalibrationTracker.from_stats(
        np.zeros(2, np.float32), np.ones(2, np.float32), np.ones(2, np.float32), n=10
    )
    writer.write_calibration_tracker("p", tracker)

    assert replaced == [
        meta / "geometry_stats" / "p_v1.json",
        meta / "edge_stats" / "p.json",
        meta / "calibration" / "p.json",
    ]
    assert not list(meta.rglob("*.tmp"))
