# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Sessions on a sphere branch read the branch's pinned state."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pytest

from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.exceptions import GDSError
from tests.conftest import FIXTURES_PATH, clone_sphere

PID = "customer_pattern"


@pytest.fixture
def sphere_dir(tmp_path: Path) -> Path:
    base = clone_sphere(FIXTURES_PATH, tmp_path / "s")
    lb.create_branch(base, "exp")
    return base


def _geo(base: Path) -> Path:
    return base / "geometry" / PID / "data.lance"


def test_branch_session_pins_eager_as_tuples_and_lazy_as_parent_ints(sphere_dir: Path) -> None:
    info = lb.read_branch_json(sphere_dir, "exp")
    pins = (
        HyperSphere.open(str(sphere_dir)).session("t", branch="exp")._reader._pinned_lance_versions
    )
    assert pins[PID] == ("exp", info["parent"]["lance_versions"][PID])
    assert pins[f"temporal:{PID}"] == info["parent"]["lance_versions"][f"temporal:{PID}"]
    assert pins["points:customers:1"] == info["parent"]["lance_versions"]["points:customers:1"]


def test_main_session_pins_stay_int(sphere_dir: Path) -> None:
    pins = HyperSphere.open(str(sphere_dir)).session("t")._reader._pinned_lance_versions
    assert pins and all(type(v) is int for v in pins.values())


def test_branch_session_reads_overlay_metadata(sphere_dir: Path) -> None:
    overlay_json = lb.overlay_dir(sphere_dir, "exp") / "_gds_meta" / "sphere.json"
    raw = json.loads(overlay_json.read_text(encoding="utf-8"))
    raw["name"] = "branch-copy"
    overlay_json.write_text(json.dumps(raw), encoding="utf-8")
    hs = HyperSphere.open(str(sphere_dir))
    assert hs.session("t", branch="exp").sphere().name == "branch-copy"
    assert hs.session("t").sphere().name != "branch-copy"


def test_main_geometry_write_is_invisible_to_branch(sphere_dir: Path) -> None:
    n = lance.dataset(str(_geo(sphere_dir))).count_rows()
    head = lance.dataset(str(_geo(sphere_dir))).to_table().slice(0, 1)
    lance.write_dataset(head, str(_geo(sphere_dir)), mode="overwrite")
    hs = HyperSphere.open(str(sphere_dir))
    assert hs.session("t", branch="exp")._reader.count_geometry_rows(PID) == n
    assert hs.session("t")._reader.count_geometry_rows(PID) == 1


def test_branch_write_is_invisible_to_main(sphere_dir: Path) -> None:
    n = lance.dataset(str(_geo(sphere_dir))).count_rows()
    branch_ds = lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None))
    lance.write_dataset(branch_ds.to_table().slice(0, 1), branch_ds, mode="overwrite")
    hs = HyperSphere.open(str(sphere_dir))
    assert hs.session("t", branch="exp")._reader.count_geometry_rows(PID) == 1
    assert hs.session("t")._reader.count_geometry_rows(PID) == n


def test_open_with_branch_sets_session_default(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir), branch="exp")
    assert hs.session("t")._reader.branch == "exp"
    assert hs.session("t", branch=None)._reader.branch == "exp"


def test_open_half_deleted_branch_raises_naming_dataset(sphere_dir: Path) -> None:
    lance.dataset(str(_geo(sphere_dir))).branches.delete("exp")
    with pytest.raises(GDSError, match=PID):
        HyperSphere.open(str(sphere_dir), branch="exp")


def test_unknown_branch_raises(sphere_dir: Path) -> None:
    with pytest.raises(GDSError, match="no branch"):
        HyperSphere.open(str(sphere_dir)).session("t", branch="nope")


def test_branch_meta_root_is_the_overlay(sphere_dir: Path) -> None:
    reader = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")._reader
    assert reader.meta_root == lb.overlay_dir(sphere_dir.resolve(), "exp")
    main_reader = HyperSphere.open(str(sphere_dir)).session("t")._reader
    assert main_reader.meta_root == main_reader._base


def test_branch_session_is_stale_after_branch_write(sphere_dir: Path) -> None:
    session = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")
    assert session.is_stale().stale is False
    branch_ds = lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None))
    lance.write_dataset(branch_ds.to_table(), branch_ds, mode="overwrite")
    assert session.is_stale().stale is True


def test_branch_navigator_reads_branch_geometry(sphere_dir: Path) -> None:
    branch_ds = lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None))
    lance.write_dataset(branch_ds.to_table().slice(0, 1), branch_ds, mode="overwrite")
    nav = HyperSphere.open(str(sphere_dir)).session("t", branch="exp").navigator()
    version = nav._resolve_version(PID)
    table = nav._storage.read_geometry(PID, version, columns=["primary_key"])
    assert table.num_rows == 1


def test_branch_topology_cache_lives_in_the_branch_directory(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hypertopos.storage import topology_cache

    roots: list[Path] = []

    class _Stop(Exception):
        pass

    def spy(root: Path, *_a: object, **_k: object) -> Path:
        roots.append(Path(root))
        raise _Stop

    monkeypatch.setattr(topology_cache, "cache_path", spy)
    nav = HyperSphere.open(str(sphere_dir)).session("t", branch="exp").navigator()
    with pytest.raises(_Stop):
        nav.find_topological_anomalies(PID)
    assert roots == [lb.overlay_dir(sphere_dir.resolve(), "exp")]


def test_branch_reads_its_own_conformance_manifest(sphere_dir: Path) -> None:
    from hypertopos.engine.conformance import read_violations

    rel = Path("_gds_meta") / "conformance" / "violations" / PID / "manifest.json"
    for root, value in ((sphere_dir, "main"), (lb.overlay_dir(sphere_dir, "exp"), "branch")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps({"rule_set_hash": value}), encoding="utf-8")
    reader = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")._reader
    _, manifest = read_violations(reader, reader._base, PID, 1)
    assert manifest == {"rule_set_hash": "branch"}


def test_main_recalibrate_is_invisible_to_branch(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    branch = hs.session("b", branch="exp")
    before_sphere = branch._reader.sphere_json_text()
    before_rows = branch._reader.read_geometry(PID, 1, columns=["primary_key", "delta_norm"])
    before_tracker = branch._reader.meta_file_text(f"calibration/{PID}.json")
    hs.session("m").recalibrate(PID)
    fresh = hs.session("b2", branch="exp")._reader
    assert fresh.sphere_json_text() == before_sphere
    assert fresh.read_geometry(PID, 1, columns=["primary_key", "delta_norm"]).equals(before_rows)
    assert fresh.meta_file_text(f"calibration/{PID}.json") == before_tracker


def test_delete_refused_while_a_live_session_holds_the_branch(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    session = hs.session("t", branch="exp")

    with pytest.raises(GDSError, match="open session"):
        hs.delete_branch("exp")
    assert "exp" in lb.branch_names(sphere_dir)

    session.close()
    hs.delete_branch("exp")
    assert lb.branch_names(sphere_dir) == []


def test_main_sessions_do_not_block_branch_delete(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    hs.session("t")

    hs.delete_branch("exp")
    assert lb.branch_names(sphere_dir) == []
