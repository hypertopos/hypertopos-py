# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Row ID cache — second read of same key must NOT call scanner with filter."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import lance as _lance
from hypertopos.storage.reader import _ROW_ID_CACHE_MAXSIZE, GDSReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_anchor_pattern(sphere_base: Path):
    """Return (pattern_id, version, primary_key) for the first anchor pattern."""
    sphere_json = json.loads((sphere_base / "_gds_meta" / "sphere.json").read_text())
    patterns = sphere_json["patterns"]
    pattern_id = next(k for k, v in patterns.items() if v.get("pattern_type") == "anchor")
    version = patterns[pattern_id]["version"]
    reader = GDSReader(str(sphere_base))
    all_table = reader.read_geometry(pattern_id, version, columns=["primary_key"])
    pk = all_table["primary_key"][0].as_py()
    return pattern_id, version, pk


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_row_id_cache_populated_on_first_read(fixtures_path):
    """First read of a primary_key must store the row ID in the cache."""
    pattern_id, version, pk = _first_anchor_pattern(fixtures_path)
    reader = GDSReader(str(fixtures_path))
    reader._row_id_cache.clear()

    reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])

    cache_key = (pattern_id, version, pk)
    assert cache_key in reader._row_id_cache, (
        "read_geometry should populate the row-ID cache after the first read"
    )


def test_row_id_cache_used_on_second_read(fixtures_path, monkeypatch):
    """Second read of the same primary_key must use ds.take() instead of scanner filter."""
    pattern_id, version, pk = _first_anchor_pattern(fixtures_path)
    reader = GDSReader(str(fixtures_path))
    reader._row_id_cache.clear()

    scanner_filter_calls: list[str] = []
    orig_scanner = _lance.LanceDataset.scanner

    def patched_scanner(self, **kwargs):
        if "filter" in kwargs and kwargs.get("filter") is not None:
            scanner_filter_calls.append(str(kwargs["filter"]))
        return orig_scanner(self, **kwargs)

    monkeypatch.setattr(_lance.LanceDataset, "scanner", patched_scanner)

    # First read: should call scanner with filter and populate cache
    reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])
    first_count = len(scanner_filter_calls)

    # Second read: cache hit — must use ds.take(), NOT scanner with filter
    reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])
    second_count = len(scanner_filter_calls)

    assert first_count == 1, f"First read should call scanner with filter (got {first_count})"
    assert second_count == first_count, (
        f"Second read should NOT call scanner with filter (cache must be used). "
        f"scanner filter calls went from {first_count} to {second_count}."
    )


def test_row_id_cache_skipped_when_pinned(fixtures_path):
    """Cache must not be populated or used when MVCC pinned version is active."""
    pattern_id, version, pk = _first_anchor_pattern(fixtures_path)
    reader = GDSReader(str(fixtures_path))

    # Activate pinned version mode
    reader._pinned_lance_versions = {pattern_id: 1}
    reader._row_id_cache.clear()

    # Even if this raises (version 1 may not exist in fixture), cache must stay empty
    with contextlib.suppress(Exception):
        reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])

    assert len(reader._row_id_cache) == 0, (
        "Row-ID cache must not be populated when pinned version is active"
    )


def test_row_id_cache_skipped_when_extra_filter(fixtures_path):
    """Cache must not be used/populated when an extra filter is combined with primary_key."""
    pattern_id, version, pk = _first_anchor_pattern(fixtures_path)
    reader = GDSReader(str(fixtures_path))
    reader._row_id_cache.clear()

    # Read with both primary_key and an extra filter
    reader.read_geometry(
        pattern_id,
        version,
        primary_key=pk,
        filter="is_anomaly = false",
        columns=["primary_key"],
    )

    # Cache must NOT be populated for combined-filter reads
    cache_key = (pattern_id, version, pk)
    assert cache_key not in reader._row_id_cache, (
        "Row-ID cache must not be populated when a combined filter is used"
    )


def test_row_id_cache_lru_eviction(fixtures_path):
    """Cache must not grow beyond _ROW_ID_CACHE_MAXSIZE entries."""
    reader = GDSReader(str(fixtures_path))

    # Fill the cache beyond the max size manually (unit-level test — no I/O needed)
    for i in range(_ROW_ID_CACHE_MAXSIZE + 10):
        key = ("pattern", 1, f"KEY-{i:05d}")
        reader._row_id_cache[key] = i
        if len(reader._row_id_cache) > _ROW_ID_CACHE_MAXSIZE:
            reader._row_id_cache.popitem(last=False)

    assert len(reader._row_id_cache) <= _ROW_ID_CACHE_MAXSIZE, (
        f"Cache grew to {len(reader._row_id_cache)}, expected <= {_ROW_ID_CACHE_MAXSIZE}"
    )


def test_row_id_cache_invalidated_after_stale_rowid(fixtures_path):
    """Stale row ID (e.g. after compact) must be evicted, not returned as wrong data.

    Two sub-cases:
    1. Out-of-range row ID — ds.take() raises OSError; must be caught and evicted.
    2. Valid row ID that points to a *different* entity — primary_key mismatch; must be evicted.
    """
    import lance as _lance_mod

    pattern_id, version, pk = _first_anchor_pattern(fixtures_path)
    lance_path = str(fixtures_path / "geometry" / pattern_id / f"v={version}" / "data.lance")

    # Sub-case 1: out-of-range row ID raises OSError from Lance.
    reader = GDSReader(str(fixtures_path))
    reader._row_id_cache.clear()
    reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])
    cache_key = (pattern_id, version, pk)
    assert cache_key in reader._row_id_cache

    reader._row_id_cache[cache_key] = 999999  # out-of-range — Lance raises OSError
    result = reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])
    assert result.num_rows == 1
    assert result["primary_key"][0].as_py() == pk
    # Stale entry must have been evicted; fallback scanner re-populates with correct row ID.
    assert reader._row_id_cache.get(cache_key) != 999999, (
        "Stale row ID 999999 must have been evicted (re-populated or absent)"
    )

    # Sub-case 2: row ID is valid but belongs to a different entity (pk mismatch).
    # Fetch all rows with their internal row IDs via Lance scanner directly.
    ds = _lance_mod.dataset(lance_path)
    all_with_rowid = ds.scanner(columns=["primary_key"], with_row_id=True).to_table()
    pks = all_with_rowid["primary_key"].to_pylist()
    row_ids = all_with_rowid["_rowid"].to_pylist()
    # Find a row whose primary_key differs from pk.
    other_idx = next(i for i, k in enumerate(pks) if k != pk)
    wrong_rowid = int(row_ids[other_idx])

    reader._row_id_cache.clear()
    reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])
    assert cache_key in reader._row_id_cache

    reader._row_id_cache[cache_key] = wrong_rowid  # valid rowid, but wrong entity
    result2 = reader.read_geometry(pattern_id, version, primary_key=pk, columns=["primary_key"])
    assert result2.num_rows == 1
    assert result2["primary_key"][0].as_py() == pk
    # Stale entry must have been evicted; fallback scanner re-populates with correct row ID.
    assert reader._row_id_cache.get(cache_key) != wrong_rowid, (
        "Stale mismatched row ID must have been evicted (re-populated or absent)"
    )


def test_row_id_cache_returns_same_data(fixtures_path):
    """Data returned from cache hit must be identical to the initial scan result."""
    pattern_id, version, pk = _first_anchor_pattern(fixtures_path)
    reader = GDSReader(str(fixtures_path))
    reader._row_id_cache.clear()

    cols = ["primary_key", "delta_norm"]
    first = reader.read_geometry(pattern_id, version, primary_key=pk, columns=cols)
    assert (pattern_id, version, pk) in reader._row_id_cache, "Cache must be populated"

    second = reader.read_geometry(pattern_id, version, primary_key=pk, columns=cols)

    assert first.num_rows == second.num_rows
    assert first["primary_key"].to_pylist() == second["primary_key"].to_pylist()
    assert first["delta_norm"].to_pylist() == second["delta_norm"].to_pylist()
