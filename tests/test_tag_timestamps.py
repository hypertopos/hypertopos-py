# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for storage._tag_timestamps forward-compat layer.

On Lance 7.0 ``tag_timestamp`` resolves the tag's native ``created_at``
(tz-aware UTC) carried on ``tags.list()``; on older Lance, or for a tag whose
listing entry lacks ``created_at``, it falls back to the version-commit
timestamp. These tests assert the public contract — ordering, unknown-tag
``KeyError``, tz-robust cleanup — without pinning to the representation of
either path, so they hold under both.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lance as _lance
import pyarrow as pa
import pytest
from hypertopos.storage._tag_timestamps import (
    _as_comparable,
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


def _make_dataset_with_epoch_tags(tmp_path: Path) -> _lance.LanceDataset:
    """Build a 3-version dataset, tagging each version immediately after its
    commit so tag-creation time tracks version order (mirrors how the builder
    tags a calibration epoch right after committing the calibration version)."""
    table = pa.table({"k": pa.array([1, 2, 3], type=pa.int32())})
    path = str(tmp_path / "test.lance")
    ds = _lance.write_dataset(table, path, mode="create")
    ds.tags.create("epoch_a", 1)
    time.sleep(0.01)
    _lance.write_dataset(table, path, mode="append")
    ds = _lance.dataset(path)
    ds.tags.create("epoch_b", 2)
    time.sleep(0.01)
    _lance.write_dataset(table, path, mode="append")
    ds = _lance.dataset(path)
    ds.tags.create("epoch_c", 3)
    return ds


class TestTagTimestamp:
    def test_known_tag_returns_a_datetime(self, tmp_path):
        ds = _make_dataset(tmp_path)
        ds.tags.create("epoch_a", 1)
        ts = tag_timestamp(ds, "epoch_a")
        assert isinstance(ts, datetime)
        # The tag was created moments ago — its timestamp is recent, never in
        # the future. Normalize tz before comparing against an aware "now".
        now = datetime.now(UTC)
        assert _as_comparable(ts, now) <= _as_comparable(now, ts)

    def test_unknown_tag_raises_keyerror(self, tmp_path):
        ds = _make_dataset(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            tag_timestamp(ds, "no_such_tag")

    def test_multiple_tags_resolve_in_creation_order(self, tmp_path):
        ds = _make_dataset_with_epoch_tags(tmp_path)
        ts_a = tag_timestamp(ds, "epoch_a")
        ts_b = tag_timestamp(ds, "epoch_b")
        ts_c = tag_timestamp(ds, "epoch_c")
        # Tags created in order a → b → c with a sleep between each.
        assert ts_a < ts_b < ts_c


class TestCleanupCalibrationEpochs:
    def test_drops_only_old_tags(self, tmp_path):
        ds = _make_dataset_with_epoch_tags(tmp_path)
        # Cutoff strictly between epoch_a's and epoch_b's creation time. Derive
        # it from the tags themselves so the test is path-agnostic (native
        # created_at or version-commit fallback both work).
        ts_a = tag_timestamp(ds, "epoch_a")
        ts_b = tag_timestamp(ds, "epoch_b")
        cutoff = ts_a + (ts_b - ts_a) / 2
        dropped = cleanup_calibration_epochs(ds, older_than=cutoff)
        assert dropped == 1
        remaining = ds.tags.list()
        assert "epoch_b" in remaining
        assert "epoch_c" in remaining
        assert "epoch_a" not in remaining

    def test_tz_naive_cutoff_does_not_raise(self, tmp_path):
        """A tz-naive cutoff must compare cleanly against the native tz-aware
        ``created_at`` — the normalization guards against a TypeError."""
        ds = _make_dataset_with_epoch_tags(tmp_path)
        naive_cutoff = datetime.now().replace(tzinfo=None) + timedelta(days=1)
        # Every tag is older than "tomorrow" → all dropped, no TypeError.
        dropped = cleanup_calibration_epochs(ds, older_than=naive_cutoff)
        assert dropped == 3
        assert ds.tags.list() == {}

    def test_keep_all_when_cutoff_in_past(self, tmp_path):
        ds = _make_dataset(tmp_path)
        ds.tags.create("epoch_recent", 1)
        ts = tag_timestamp(ds, "epoch_recent")
        cutoff = ts - timedelta(days=365)
        dropped = cleanup_calibration_epochs(ds, older_than=cutoff)
        assert dropped == 0
        assert "epoch_recent" in ds.tags.list()
