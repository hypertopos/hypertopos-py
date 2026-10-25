# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Session pins cover the derived datasets; branch metadata writes are race-safe."""

from __future__ import annotations

import threading
from contextlib import suppress
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.exceptions import GDSError
from tests.conftest import FIXTURES_PATH, clone_sphere

PID = "customer_pattern"
SIDECARS = [
    f"forecast/population/{PID}/data.lance",
    f"_gds_meta/edge_features/{PID}/data.lance",
    f"_gds_meta/trajectory/{PID}.lance",
    f"_gds_meta/temporal_centroids/{PID}.lance",
    f"_gds_meta/contagion_stats/{PID}.lance",
]


@pytest.fixture
def sphere_dir(tmp_path: Path) -> Path:
    return clone_sphere(FIXTURES_PATH, tmp_path / "s")


def _rows(n: int) -> pa.Table:
    return pa.table({"primary_key": [f"k{i}" for i in range(n)]})


@pytest.mark.parametrize("rel", SIDECARS)
def test_main_session_reads_derived_datasets_at_its_snapshot(sphere_dir: Path, rel: str) -> None:
    path = sphere_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(_rows(2), str(path), mode="overwrite")
    session = HyperSphere.open(str(sphere_dir)).session("a")
    assert not session.is_stale().stale

    lance.write_dataset(_rows(5), str(path), mode="overwrite")

    assert session._reader.dataset(path).count_rows() == 2
    status = session.is_stale()
    assert status.stale
    key = lb._key_for(rel)
    assert status.pinned_versions[key] != status.latest_versions[key]
    session.refresh()
    assert session._reader.dataset(path).count_rows() == 5
    assert not session.is_stale().stale


def test_concurrent_recipe_appends_keep_every_recipe(sphere_dir: Path, monkeypatch) -> None:
    lb.create_branch(sphere_dir, "exp")
    real_read = lb.read_branch_json
    barrier = threading.Barrier(2, timeout=1.0)

    def racing_read(base, name):
        info = real_read(base, name)
        with suppress(threading.BrokenBarrierError):
            barrier.wait()
        return info

    monkeypatch.setattr(lb, "read_branch_json", racing_read)
    errors: list[BaseException] = []

    def append(i: int) -> None:
        try:
            lb.append_recipe(sphere_dir, "exp", {"op": "recalibrate", "n": i})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    monkeypatch.undo()

    assert not errors
    recipes = lb.read_branch_json(sphere_dir, "exp")["recipes"]
    assert sorted(r["seq"] for r in recipes) == [1, 2]
    assert sorted(r["n"] for r in recipes) == [0, 1]


def test_recipe_append_during_a_promotion_is_refused_not_lost(
    sphere_dir: Path, monkeypatch
) -> None:
    lb.create_branch(sphere_dir, "exp")
    monkeypatch.setattr(lb, "_RECIPE_LOCK_WAIT_SECONDS", 0.2)
    seen: list[str] = []

    def concurrent_append(base, name):
        with pytest.raises(GDSError, match="recipe log") as exc:
            lb.append_recipe(sphere_dir, "exp", {"op": "recalibrate"})
        seen.append(str(exc.value))
        raise GDSError("stop the promotion here")

    monkeypatch.setattr(lb, "check_promotable", concurrent_append)
    with pytest.raises(GDSError, match="stop the promotion"):
        HyperSphere.open(str(sphere_dir)).promote_branch("exp")

    assert seen
    assert lb.append_recipe(sphere_dir, "exp", {"op": "recalibrate"})["seq"] == 1


def test_session_reader_starts_with_an_empty_adjacency_cache(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    hs._reader._adjacency_cache[PID] = object()
    assert hs.session("a")._reader._adjacency_cache == {}


def test_concurrent_create_branch_raises_gds_error(sphere_dir: Path, monkeypatch) -> None:
    real_inventory = lb.dataset_inventory
    overlay = lb.overlay_dir(sphere_dir.resolve(), "exp")

    def racing_inventory(base):
        overlay.mkdir(parents=True)
        (overlay / "marker").write_text("winner", encoding="utf-8")
        return real_inventory(base)

    monkeypatch.setattr(lb, "dataset_inventory", racing_inventory)
    with pytest.raises(GDSError, match="already exists"):
        lb.create_branch(sphere_dir, "exp")
    assert (overlay / "marker").read_text(encoding="utf-8") == "winner"
