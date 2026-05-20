# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for storage._tag_timestamps forward-compat layer."""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import lance as _lance
import pyarrow as pa
import pytest
from hypertopos.storage._tag_timestamps import (
    cleanup_calibration_epochs,
    tag_timestamp,
)


def _make_dataset(tmp_path: Path) -> _lance.LanceDataset:
    """Build a small Lance dataset with 3 commits — one per epoch."""
    table = pa.table({"k": pa.array([1, 2, 3], type=pa.int32())})
    path = str(tmp_path / "test.lance")
    ds = _lance.write_dataset(table, path, mode="create")
    # Two more commits to give us multiple versions
    _lance.write_dataset(table, path, mode="append")
    _lance.write_dataset(table, path, mode="append")
    return _lance.dataset(path)


class TestTagTimestamp:
    def test_known_tag_returns_version_timestamp(self, tmp_path):
        ds = _make_dataset(tmp_path)
        ds.tags.create("epoch_a", 1)
        ts = tag_timestamp(ds, "epoch_a")
        # Should match version 1's commit timestamp
        v1_ts = next(
            v["timestamp"] for v in ds.versions() if v["version"] == 1
        )
        assert ts == v1_ts

    def test_unknown_tag_raises_keyerror(self, tmp_path):
        ds = _make_dataset(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            tag_timestamp(ds, "no_such_tag")

    def test_multiple_tags_resolve_independently(self, tmp_path):
        ds = _make_dataset(tmp_path)
        ds.tags.create("epoch_a", 1)
        ds.tags.create("epoch_b", 2)
        ds.tags.create("epoch_c", 3)
        ts_a = tag_timestamp(ds, "epoch_a")
        ts_b = tag_timestamp(ds, "epoch_b")
        ts_c = tag_timestamp(ds, "epoch_c")
        # Order preserved — version 1 ≤ version 2 ≤ version 3
        assert ts_a <= ts_b <= ts_c


class TestCleanupCalibrationEpochs:
    def test_drops_only_old_tags(self, tmp_path):
        ds = _make_dataset(tmp_path)
        ds.tags.create("epoch_old", 1)
        ds.tags.create("epoch_new", 3)
        # Cutoff between v2 and v3 timestamps
        v2_ts = next(
            v["timestamp"] for v in ds.versions() if v["version"] == 2
        )
        v3_ts = next(
            v["timestamp"] for v in ds.versions() if v["version"] == 3
        )
        cutoff = v2_ts + (v3_ts - v2_ts) / 2
        dropped = cleanup_calibration_epochs(ds, older_than=cutoff)
        assert dropped == 1
        remaining = ds.tags.list()
        assert "epoch_new" in remaining
        assert "epoch_old" not in remaining

    def test_keep_all_when_cutoff_in_past(self, tmp_path):
        ds = _make_dataset(tmp_path)
        ds.tags.create("epoch_recent", 1)
        v1_ts = next(
            v["timestamp"] for v in ds.versions() if v["version"] == 1
        )
        cutoff = v1_ts - timedelta(days=365)
        dropped = cleanup_calibration_epochs(ds, older_than=cutoff)
        assert dropped == 0
        assert "epoch_recent" in ds.tags.list()
