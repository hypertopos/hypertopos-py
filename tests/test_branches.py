# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Sphere branches: create, list and delete branch storage."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lance
import pytest

from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.exceptions import GDSError
from tests.conftest import FIXTURES_PATH, clone_sphere


@pytest.fixture
def sphere_dir(tmp_path: Path) -> Path:
    return clone_sphere(FIXTURES_PATH, tmp_path / "s")


def _data_bytes(ds_dir: Path) -> int:
    return sum(p.stat().st_size for p in (ds_dir / "data").rglob("*") if p.is_file())


def _meta_files(base: Path) -> dict[str, bytes]:
    meta = base / "_gds_meta"
    out = {}
    for p in meta.rglob("*"):
        rel = p.relative_to(meta).as_posix()
        if p.is_file() and ".lance/" not in rel and not rel.startswith("branches/"):
            out[rel] = p.read_bytes()
    return out


def test_dataset_inventory_keys_every_lance_dataset(sphere_dir: Path) -> None:
    inv = lb.dataset_inventory(sphere_dir)
    assert set(inv) == {
        "customer_pattern",
        "temporal:customer_pattern",
        "points:customers:1",
        "traj:customer_pattern",
    }
    assert inv["customer_pattern"] == sphere_dir / "geometry" / "customer_pattern" / "data.lance"


def test_create_branch_is_metadata_only(sphere_dir: Path) -> None:
    geo = sphere_dir / "geometry" / "customer_pattern" / "data.lance"
    before_bytes = _data_bytes(geo)
    meta_before = _meta_files(sphere_dir)

    info = lb.create_branch(sphere_dir, "exp", created_by="t", purpose="p")

    assert _data_bytes(geo) == before_bytes
    assert (geo / "tree" / "exp").is_dir()
    for key, path in lb.dataset_inventory(sphere_dir).items():
        if key != "customer_pattern":
            assert not (path / "tree").exists(), key
    assert info["datasets"] == {
        "customer_pattern": "lance-branch",
        "temporal:customer_pattern": "parent-pin",
        "points:customers:1": "parent-pin",
        "traj:customer_pattern": "parent-pin",
    }
    assert set(info["parent"]["lance_versions"]) == set(info["datasets"])
    overlay = sphere_dir / "_gds_meta" / "branches" / "exp"
    for rel, data in meta_before.items():
        assert (overlay / "_gds_meta" / rel).read_bytes() == data, rel
    on_disk = json.loads((overlay / "branch.json").read_text(encoding="utf-8"))
    assert on_disk == info
    assert on_disk["recipes"] == [] and on_disk["replayable"] is True


def test_create_branch_records_parent_fingerprint(sphere_dir: Path) -> None:
    info = lb.create_branch(sphere_dir, "exp")
    fp = info["parent"]["meta_fingerprint"]
    assert set(fp) == {"customer_pattern"}
    assert fp["customer_pattern"].startswith("sha256:")


def test_create_branch_cost_bounded(sphere_dir: Path) -> None:
    t = time.perf_counter()
    lb.create_branch(sphere_dir, "exp")
    assert time.perf_counter() - t < 2.0
    trees = [p for p in sphere_dir.rglob("tree") if p.is_dir()]
    assert len(trees) == 1


def test_duplicate_branch_rejected(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    with pytest.raises(GDSError, match="already exists"):
        lb.create_branch(sphere_dir, "exp")


@pytest.mark.parametrize("name", ["a/b", "..", "", "x y", "a\\b", "-x", "main"])
def test_bad_branch_name_rejected(sphere_dir: Path, name: str) -> None:
    with pytest.raises(GDSError, match="branch name"):
        lb.create_branch(sphere_dir, name)


def test_list_branches_reads_overlay_only(sphere_dir: Path) -> None:
    assert lb.list_branches(sphere_dir) == []
    lb.create_branch(sphere_dir, "b2", purpose="second")
    lb.create_branch(sphere_dir, "b1", created_by="me")
    infos = lb.list_branches(sphere_dir)
    assert [i.name for i in infos] == ["b1", "b2"]
    assert infos[0].created_by == "me" and infos[1].purpose == "second"
    assert infos[0].n_recipes == 0 and infos[0].replayable is True
    assert infos[0].stale is False


def test_list_branches_marks_stale_after_main_geometry_write(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    geo = sphere_dir / "geometry" / "customer_pattern" / "data.lance"
    ds = lance.dataset(str(geo))
    lance.write_dataset(ds.to_table(), str(geo), mode="overwrite")
    assert lb.list_branches(sphere_dir)[0].stale is True


def test_delete_branch_removes_tree_and_overlay(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    lb.delete_branch(sphere_dir, "exp")
    geo = sphere_dir / "geometry" / "customer_pattern" / "data.lance"
    assert "exp" not in lance.dataset(str(geo)).branches.list()
    assert not (geo / "tree" / "exp").exists()
    assert not (sphere_dir / "_gds_meta" / "branches" / "exp").exists()
    assert lb.list_branches(sphere_dir) == []


def test_delete_unknown_branch_raises(sphere_dir: Path) -> None:
    with pytest.raises(GDSError, match="no branch"):
        lb.delete_branch(sphere_dir, "nope")


def test_half_deleted_branch_is_reported_and_cleaned(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    geo = sphere_dir / "geometry" / "customer_pattern" / "data.lance"
    lance.dataset(str(geo)).branches.delete("exp")
    with pytest.raises(GDSError, match="customer_pattern"):
        lb.verify_branch(sphere_dir, "exp")
    lb.delete_branch(sphere_dir, "exp")
    assert not (sphere_dir / "_gds_meta" / "branches" / "exp").exists()


def test_verify_branch_passes_on_fresh_branch(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    assert lb.verify_branch(sphere_dir, "exp")["name"] == "exp"


def test_hypersphere_branch_verbs(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    hs.create_branch("exp", purpose="p")
    assert [b.name for b in hs.branches()] == ["exp"]
    hs.delete_branch("exp")
    assert hs.branches() == []


def test_build_refused_while_branch_exists(sphere_dir: Path) -> None:
    from hypertopos.builder.builder import GDSBuilder

    lb.create_branch(sphere_dir, "exp")
    builder = GDSBuilder("sales", str(sphere_dir))
    with pytest.raises(GDSError, match="exp"):
        builder.build()


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short paths are Windows-only")
def test_short_path_spelling_is_resolved(sphere_dir: Path) -> None:
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(sphere_dir), buf, 1024)
    short = buf.value if n else str(sphere_dir)
    lb.create_branch(Path(short), "exp")
    geo = sphere_dir / "geometry" / "customer_pattern" / "data.lance"
    assert "exp" in lance.dataset(str(geo)).branches.list()


def _add_centroids(base: Path) -> Path:
    import pyarrow as pa

    path = base / "_gds_meta" / "temporal_centroids" / "customer_pattern.lance"
    table = pa.table(
        {
            "window_start": ["2024-01-01", "2024-02-01"],
            "window_end": ["2024-02-01", "2024-03-01"],
            "centroid": [[0.0, 1.0], [1.0, 0.0]],
            "entity_count": [3, 4],
            "anomaly_rate": [0.1, 0.2],
        }
    )
    lance.write_dataset(table, str(path))
    return path


def test_centroids_are_in_the_eager_set(sphere_dir: Path) -> None:
    path = _add_centroids(sphere_dir)
    info = lb.create_branch(sphere_dir, "exp")
    assert info["datasets"]["centroids:customer_pattern"] == "lance-branch"
    assert "exp" in lance.dataset(str(path)).branches.list()


def test_main_recalibrate_keeps_branch_centroids(sphere_dir: Path) -> None:
    path = _add_centroids(sphere_dir)
    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t")
    session.recalibrate("customer_pattern")
    lb.verify_branch(sphere_dir, "exp")
    assert lance.dataset(str(path)).checkout_version(("exp", None)).count_rows() == 2
    assert lance.dataset(str(path)).count_rows() == 0
    assert session._reader.read_temporal_centroids("customer_pattern") is None


def test_cli_build_force_refused_while_branch_exists(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from hypertopos.cli import build as cli_build

    lb.create_branch(sphere_dir, "exp")
    monkeypatch.setattr(cli_build, "parse_config", lambda _p: SimpleNamespace(sphere_id="s"))
    with pytest.raises(SystemExit):
        cli_build.run_build(str(sphere_dir / "sphere.yaml"), str(sphere_dir), True, False)
    assert "exp" in capsys.readouterr().err
    assert (sphere_dir / "_gds_meta" / "sphere.json").exists()


def test_interrupted_create_is_listed_and_deletable(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    (sphere_dir / "_gds_meta" / "branches" / "exp" / "branch.json").unlink()
    [info] = lb.list_branches(sphere_dir)
    assert info.name == "exp" and info.complete is False
    lb.delete_branch(sphere_dir, "exp")
    geo = sphere_dir / "geometry" / "customer_pattern" / "data.lance"
    assert "exp" not in lance.dataset(str(geo)).branches.list()
    assert lb.branch_names(sphere_dir) == []


def test_failed_create_rolls_back(sphere_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(lb, "write_branch_json", boom)
    with pytest.raises(OSError, match="disk full"):
        lb.create_branch(sphere_dir, "exp")
    geo = sphere_dir / "geometry" / "customer_pattern" / "data.lance"
    assert "exp" not in lance.dataset(str(geo)).branches.list()
    assert lb.branch_names(sphere_dir) == []
