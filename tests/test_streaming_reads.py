# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Streaming reads — read_geometry_batched and read_temporal_batched."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC
from pathlib import Path

import pyarrow as pa
from hypertopos.storage.reader import BATCH_SCAN_THRESHOLD, GDSReader  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_anchor_pattern(sphere_base: Path):
    """Return (pattern_id, version) for the first anchor pattern."""
    sphere_json = json.loads((sphere_base / "_gds_meta" / "sphere.json").read_text())
    patterns = sphere_json["patterns"]
    pattern_id = next(k for k, v in patterns.items() if v.get("pattern_type") == "anchor")
    version = patterns[pattern_id]["version"]
    return pattern_id, version


def _first_pattern_with_temporal(sphere_base: Path) -> str | None:
    """Return pattern_id that has a temporal Lance dataset, or None."""
    sphere_json = json.loads((sphere_base / "_gds_meta" / "sphere.json").read_text())
    patterns = sphere_json["patterns"]
    for k in patterns:
        if (sphere_base / "temporal" / k / "data.lance").exists():
            return k
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_batch_scan_threshold_exported():
    assert BATCH_SCAN_THRESHOLD == 10_000_000


def test_read_geometry_batched_yields_batches(fixtures_path):
    reader = GDSReader(str(fixtures_path))
    pattern_id, version = _first_anchor_pattern(fixtures_path)

    batches = list(reader.read_geometry_batched(pattern_id, version, columns=["primary_key"]))
    assert len(batches) >= 1
    for batch in batches:
        assert isinstance(batch, pa.RecordBatch)
        assert "primary_key" in batch.schema.names


def test_read_geometry_batched_total_rows(fixtures_path):
    reader = GDSReader(str(fixtures_path))
    pattern_id, version = _first_anchor_pattern(fixtures_path)

    batches = reader.read_geometry_batched(pattern_id, version, columns=["primary_key"])
    total_batched = sum(len(b) for b in batches)
    total_eager = reader.count_geometry_rows(pattern_id, version)
    assert total_batched == total_eager


def test_read_temporal_batched_yields_batches(fixtures_path):
    reader = GDSReader(str(fixtures_path))
    pattern_id = _first_pattern_with_temporal(fixtures_path)
    if pattern_id is None:
        return  # No temporal data in fixture — skip gracefully
    batches = list(reader.read_temporal_batched(pattern_id))
    for batch in batches:
        assert isinstance(batch, pa.RecordBatch)


def test_read_temporal_batched_empty_for_missing(fixtures_path):
    reader = GDSReader(str(fixtures_path))
    batches = list(reader.read_temporal_batched("nonexistent_pattern_xyz"))
    assert batches == []


def test_drift_uses_batched_temporal(fixtures_path, monkeypatch):
    """find_drifting_entities must call read_temporal_batched (not old eager path)."""
    batched_called = []
    orig = GDSReader.read_temporal_batched

    def patched(self, *a, **kw):
        batched_called.append(True)
        return orig(self, *a, **kw)

    monkeypatch.setattr(GDSReader, "read_temporal_batched", patched)

    sphere_json = json.loads((fixtures_path / "_gds_meta" / "sphere.json").read_text())
    patterns = sphere_json["patterns"]
    # Find an anchor pattern that has temporal data
    pattern_id = None
    for k, v in patterns.items():
        if (
            v.get("pattern_type") == "anchor"
            and (fixtures_path / "temporal" / k / "data.lance").exists()
        ):
            pattern_id = k
            break

    if pattern_id is None:
        # No anchor pattern with temporal data — still verify the method is wired
        # by picking any anchor pattern (will return [] but must call batched)
        pattern_id = next(k for k, v in patterns.items() if v.get("pattern_type") == "anchor")

    import uuid
    from datetime import datetime

    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Contract, Manifest
    from hypertopos.navigation.navigator import GDSNavigator
    from hypertopos.storage.cache import GDSCache

    reader = GDSReader(str(fixtures_path))
    sphere_data = reader.read_sphere()
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id=str(uuid.uuid4()),
        agent_id="test",
        snapshot_time=datetime.now(tz=UTC),
        status="active",
        line_versions={lid: line.current_version() for lid, line in sphere_data.lines.items()},
        pattern_versions={pid: p.version for pid, p in sphere_data.patterns.items()},
    )
    contract = Contract(
        manifest_id=manifest.manifest_id,
        pattern_ids=list(sphere_data.patterns.keys()),
    )
    nav = GDSNavigator(engine=engine, storage=reader, manifest=manifest, contract=contract)

    with suppress(Exception):
        nav.π9_attract_drift(pattern_id, top_n=3)

    assert batched_called, "find_drifting_entities must use read_temporal_batched"
