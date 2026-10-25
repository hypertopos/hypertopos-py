# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""A session reads one consistent snapshot of sphere.json with its pinned Lance versions.

Writers replace sphere.json atomically; a session sees its own writes and can be
refreshed to the latest state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.sphere import HyperSphere
from hypertopos.storage.exceptions import GDSStorageError
from hypertopos.storage.reader import GDSReader
from tests.conftest import clone_sphere

PID = "order_pattern"


def _build_minimal_sphere(out: Path) -> str:
    """One anchor line, one event line and one event pattern (as in test_mvcc_session)."""
    b = GDSBuilder("pin_test", str(out))
    b.add_line(
        "customers",
        [
            {"cust_id": "C-1", "name": "Alpha"},
            {"cust_id": "C-2", "name": "Beta"},
            {"cust_id": "C-3", "name": "Gamma"},
        ],
        key_col="cust_id",
        source_id="test",
    )
    b.add_line(
        "orders",
        [
            {"order_id": "O-1", "cust_id": "C-1"},
            {"order_id": "O-2", "cust_id": "C-2"},
            {"order_id": "O-3", "cust_id": "C-3"},
            {"order_id": "O-4", "cust_id": "C-1"},
        ],
        key_col="order_id",
        source_id="test",
        role="event",
    )
    b.add_pattern(
        PID,
        pattern_type="event",
        entity_line="orders",
        relations=[RelationSpec("customers", fk_col="cust_id", direction="in", required=True)],
    )
    return b.build()


@pytest.fixture(scope="module")
def built_sphere(tmp_path_factory) -> Path:
    return Path(_build_minimal_sphere(tmp_path_factory.mktemp("pin") / "gds"))


def test_sphere_json_write_is_atomic(tmp_path, monkeypatch):
    from hypertopos.storage import writer as writer_mod

    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text('{"old": true}')

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(writer_mod.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        writer_mod.write_sphere_json(meta, {"new": True})
    assert (meta / "sphere.json").read_text() == '{"old": true}'


def test_sphere_json_write_matches_previous_format(tmp_path):
    from hypertopos.storage.writer import write_sphere_json

    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    data = {"format_version": "3.1", "patterns": {"p": {"mu": [1.0]}}}
    write_sphere_json(meta, data)
    assert (meta / "sphere.json").read_text() == json.dumps(data, indent=2)
    assert not (meta / "sphere.json.tmp").exists()


# ---------------------------------------------------------------------------
# Task 2: the session snapshot
# ---------------------------------------------------------------------------


def test_session_does_not_see_external_sphere_json_change(built_sphere, tmp_path):
    base = clone_sphere(built_sphere, tmp_path / "s")
    session = HyperSphere.open(str(base)).session("pin")
    reader = session.navigator()._storage
    before = reader.read_sphere().patterns[PID].mu.copy()
    path = base / "_gds_meta" / "sphere.json"
    data = json.loads(path.read_text())
    data["patterns"][PID]["mu"][0] += 1.0
    path.write_text(json.dumps(data))
    np.testing.assert_array_equal(reader.read_sphere().patterns[PID].mu, before)


def test_read_sphere_does_not_touch_disk_after_open(built_sphere, monkeypatch):
    session = HyperSphere.open(str(built_sphere)).session("io")
    reader = session.navigator()._storage

    def no_disk(*args, **kwargs):
        raise AssertionError("sphere.json read from disk after the session opened")

    monkeypatch.setattr(Path, "read_text", no_disk)
    for _ in range(10):
        reader.read_sphere()
    reader._load_sphere_json()


def test_reader_without_snapshot_reads_live(built_sphere, tmp_path):
    base = clone_sphere(built_sphere, tmp_path / "s")
    reader = GDSReader(str(base))
    path = base / "_gds_meta" / "sphere.json"
    data = json.loads(path.read_text())
    data["patterns"][PID]["population_size"] = 12345
    path.write_text(json.dumps(data))
    assert reader.read_sphere().patterns[PID].population_size == 12345


def test_snapshot_retries_once_on_a_concurrent_commit(built_sphere, monkeypatch):
    reader = GDSReader(str(built_sphere))
    real = reader.latest_lance_versions
    calls: list[int] = []

    def flaky(ids):
        calls.append(1)
        out = real(ids)
        if len(calls) == 2:  # the second read of attempt 1 sees a newer commit
            out = {k: v + 1 for k, v in out.items()}
        return out

    monkeypatch.setattr(reader, "latest_lance_versions", flaky)
    reader.take_snapshot()
    assert len(calls) == 4
    assert reader._pinned_lance_versions == real([PID])


def test_snapshot_gives_up_after_three_attempts(built_sphere, monkeypatch):
    reader = GDSReader(str(built_sphere))
    counter = iter(range(1000))
    monkeypatch.setattr(reader, "latest_lance_versions", lambda ids: {PID: next(counter)})
    with pytest.raises(GDSStorageError, match="being written"):
        reader.take_snapshot()


# ---------------------------------------------------------------------------
# Task 3: read-your-own-writes, refresh, staleness
# ---------------------------------------------------------------------------

ANCHOR = "cust_pattern"


def _anchor(tmp_path: Path) -> str:
    from tests.test_incremental import _build_anchor_sphere

    return str(_build_anchor_sphere(tmp_path / "anchor", 300))


def _meta(base: str) -> Path:
    return Path(base) / "_gds_meta"


def test_session_sees_own_recalibrate(tmp_path):
    from hypertopos.storage.writer import write_sphere_json

    base = _anchor(tmp_path)
    session = HyperSphere.open(base).session("own")
    nav = session.navigator()
    data = json.loads((_meta(base) / "sphere.json").read_text())
    data["patterns"][ANCHOR]["population_size"] = 1
    write_sphere_json(_meta(base), data)
    session.refresh()
    assert nav._storage.read_sphere().patterns[ANCHOR].population_size == 1

    result = session.recalibrate(ANCHOR)

    pattern = nav._storage.read_sphere().patterns[ANCHOR]
    assert pattern.population_size == result["records_recalibrated"]
    assert float(np.linalg.norm(pattern.theta)) == pytest.approx(result["new_theta_norm"], abs=1e-4)


def test_other_process_ingest_is_invisible_until_refresh(tmp_path):
    import pyarrow as pa

    base = _anchor(tmp_path)
    session = HyperSphere.open(base).session("isolated")
    reader = session.navigator()._storage
    pop0 = reader.read_sphere().patterns[ANCHOR].population_size
    rows0 = reader.count_geometry_rows(ANCHOR)
    extra = pa.table(
        {
            "primary_key": pa.array([f"C{300 + i}" for i in range(5)], type=pa.string()),
            "event_count": pa.array([10.0] * 5, type=pa.float64()),
        }
    )
    GDSBuilder("other", base).incremental_update(
        ANCHOR, changed_entities=extra, recalibrate="never"
    )

    assert reader.read_sphere().patterns[ANCHOR].population_size == pop0
    assert reader.count_geometry_rows(ANCHOR) == rows0
    status = session.is_stale()
    assert status.stale and status.metadata_changed

    session.refresh()
    assert reader.read_sphere().patterns[ANCHOR].population_size == pop0 + 5
    assert reader.count_geometry_rows(ANCHOR) == rows0 + 5
    assert session.is_stale().stale is False


def test_navigator_created_before_refresh_sees_new_snapshot(tmp_path):
    import pyarrow as pa

    base = _anchor(tmp_path)
    session = HyperSphere.open(base).session("nav")
    nav = session.navigator()
    extra = pa.table(
        {
            "primary_key": pa.array(["C900"], type=pa.string()),
            "event_count": pa.array([10.0], type=pa.float64()),
        }
    )
    GDSBuilder("other", base).incremental_update(
        ANCHOR, changed_entities=extra, recalibrate="never"
    )
    session.refresh()
    keys = nav._storage.read_geometry(ANCHOR, 1, columns=["primary_key"])["primary_key"]
    assert "C900" in set(keys.to_pylist())


def test_recalibrate_preserves_concurrent_sphere_json_change(tmp_path):
    from hypertopos.storage.writer import write_sphere_json

    base = _anchor(tmp_path)
    session = HyperSphere.open(base).session("rmw")
    data = json.loads((_meta(base) / "sphere.json").read_text())
    data["description"] = "written by another process after the session opened"
    write_sphere_json(_meta(base), data)

    session.recalibrate(ANCHOR)

    after = json.loads((_meta(base) / "sphere.json").read_text())
    assert after["description"] == "written by another process after the session opened"


def test_is_stale_is_false_right_after_open(tmp_path):
    base = _anchor(tmp_path)
    session = HyperSphere.open(base).session("fresh")
    status = session.is_stale()
    assert status.stale is False
    assert status.metadata_changed is False
    assert status.pinned_versions == status.latest_versions


# ---------------------------------------------------------------------------
# Final-review fixes
# ---------------------------------------------------------------------------


def _ingest(base: str, keys: list[str], count: float) -> None:
    import pyarrow as pa

    extra = pa.table(
        {
            "primary_key": pa.array(keys, type=pa.string()),
            "event_count": pa.array([count] * len(keys), type=pa.float64()),
        }
    )
    GDSBuilder("other", base).incremental_update(
        ANCHOR, changed_entities=extra, recalibrate="never"
    )


def test_navigator_and_engine_caches_follow_own_recalibrate(tmp_path):
    base = _anchor(tmp_path)
    _ingest(base, [f"X{i}" for i in range(60)], 5000.0)  # outliers shift the fit
    session = HyperSphere.open(base).session("caches")
    nav = session.navigator()
    theta_before = nav._get_cached_theta(ANCHOR).copy()
    nav._get_anomaly_map(ANCHOR, 1)
    session._engine.build_polygon("C0", ANCHOR, session._manifest)

    session.recalibrate(ANCHOR)

    fresh = HyperSphere.open(base).session("fresh")
    fresh_nav = fresh.navigator()
    np.testing.assert_allclose(nav._get_cached_theta(ANCHOR), fresh_nav._get_cached_theta(ANCHOR))
    assert not np.allclose(nav._get_cached_theta(ANCHOR), theta_before)
    assert nav._get_anomaly_map(ANCHOR, 1) == fresh_nav._get_anomaly_map(ANCHOR, 1)
    own = session._engine.build_polygon("C0", ANCHOR, session._manifest)
    other = fresh._engine.build_polygon("C0", ANCHOR, fresh._manifest)
    assert own.delta_norm == pytest.approx(other.delta_norm)


def test_anomaly_summary_stays_on_the_snapshot(tmp_path):
    base = _anchor(tmp_path)
    session = HyperSphere.open(base).session("summary")
    nav = session.navigator()
    before = nav.anomaly_summary(ANCHOR)

    _ingest(base, [f"Y{i}" for i in range(60)], 10.0)

    after = nav.anomaly_summary(ANCHOR)
    assert after["total_entities"] == before["total_entities"]
    assert after["anomaly_rate"] == before["anomaly_rate"]
    session.refresh()
    assert nav.anomaly_summary(ANCHOR)["total_entities"] == before["total_entities"] + 60


def test_recalibrate_refuses_when_geometry_changed_after_open(tmp_path):
    base = _anchor(tmp_path)
    session = HyperSphere.open(base).session("guard")
    _ingest(base, [f"Z{i}" for i in range(5)], 10.0)
    with pytest.raises(GDSStorageError, match="refresh"):
        session.recalibrate(ANCHOR)
    assert GDSReader(base).count_geometry_rows(ANCHOR) == 305
    session.refresh()
    assert session.recalibrate(ANCHOR)["records_recalibrated"] == 305


def test_sphere_json_write_retries_while_another_handle_has_it_open(tmp_path, monkeypatch):
    from hypertopos.storage import writer as writer_mod

    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text("{}")
    real_replace = writer_mod.os.replace
    failures = iter([PermissionError(5, "Access is denied")] * 2)

    def busy_then_free(src, dst):
        err = next(failures, None)
        if err is not None:
            raise err
        real_replace(src, dst)

    monkeypatch.setattr(writer_mod.os, "replace", busy_then_free)
    writer_mod.write_sphere_json(meta, {"ok": True})
    assert json.loads((meta / "sphere.json").read_text()) == {"ok": True}
    assert list(meta.glob("*.tmp")) == []


def test_sphere_json_write_gives_up_and_cleans_up(tmp_path, monkeypatch):
    from hypertopos.storage import writer as writer_mod

    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text("{}")

    def always_busy(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(writer_mod.os, "replace", always_busy)
    monkeypatch.setattr(writer_mod, "_REPLACE_RETRY_SECONDS", 0.05)
    with pytest.raises(PermissionError):
        writer_mod.write_sphere_json(meta, {"ok": True})
    assert (meta / "sphere.json").read_text() == "{}"
    assert list(meta.glob("*.tmp")) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file-sharing semantics")
def test_sphere_json_write_succeeds_after_a_reader_closes_its_handle(tmp_path):
    import threading
    import time

    from hypertopos.storage.writer import write_sphere_json

    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text("{}")
    handle = open(meta / "sphere.json")  # noqa: SIM115
    threading.Thread(target=lambda: (time.sleep(0.2), handle.close())).start()
    write_sphere_json(meta, {"ok": True})
    assert json.loads((meta / "sphere.json").read_text()) == {"ok": True}
