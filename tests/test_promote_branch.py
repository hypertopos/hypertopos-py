# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Promoting a sphere branch to main by replaying its recipes."""

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
    return clone_sphere(FIXTURES_PATH, tmp_path / "s")


def _geo(base: Path) -> Path:
    return base / "geometry" / PID / "data.lance"


def _keys(base: Path) -> list[str]:
    table = lance.dataset(str(_geo(base))).to_table(columns=["primary_key"])
    return table["primary_key"].to_pylist()


def _norms(ds: lance.LanceDataset) -> dict[str, float]:
    t = ds.to_table(columns=["primary_key", "delta_norm"])
    return dict(zip(t["primary_key"].to_pylist(), t["delta_norm"].to_pylist(), strict=True))


def _branch_with_recipe(base: Path, exclude: bool = True) -> HyperSphere:
    hs = HyperSphere.open(str(base))
    hs.session("t").recalibrate(
        PID, branch="exp", exclude_keys=[_keys(base)[0]] if exclude else None
    )
    return hs


def _meta(base: Path) -> dict[str, bytes]:
    meta = base / "_gds_meta"
    return {
        p.relative_to(meta).as_posix(): p.read_bytes()
        for p in meta.rglob("*")
        if p.is_file()
        and ".lance/" not in p.relative_to(meta).as_posix()
        and not p.relative_to(meta).as_posix().startswith("branches/")
    }


def test_promote_makes_main_equal_to_the_branch(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    overlay = lb.overlay_dir(sphere_dir, "exp") / "_gds_meta" / "sphere.json"
    branch_mu = json.loads(overlay.read_text("utf-8"))["patterns"][PID]["mu"]
    branch_norms = _norms(lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None)))
    report = hs.promote_branch("exp")
    assert report.name == "exp" and report.recipes_replayed == 1
    main = json.loads((sphere_dir / "_gds_meta" / "sphere.json").read_text("utf-8"))
    assert main["patterns"][PID]["mu"] == branch_mu
    assert _norms(lance.dataset(str(_geo(sphere_dir)))) == branch_norms
    assert lb.branch_names(sphere_dir) == []
    assert report.versions_after[PID] > report.versions_before[PID]
    assert hs.session("x").sphere().patterns[PID].mu.tolist() == pytest.approx(branch_mu)


def test_promote_refuses_when_main_moved(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    hs.session("m").recalibrate(PID)
    with pytest.raises(GDSError, match="main has moved"):
        hs.promote_branch("exp")
    assert lb.branch_names(sphere_dir) == ["exp"]


def test_promote_refuses_a_branch_without_recipes(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    hs.create_branch("exp")
    with pytest.raises(GDSError, match="no recipes"):
        hs.promote_branch("exp")


def test_promote_refuses_a_foreign_commit_on_the_branch(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    branch_ds = lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None))
    lance.write_dataset(branch_ds.to_table(), branch_ds, mode="overwrite")
    with pytest.raises(GDSError, match="not recorded"):
        hs.promote_branch("exp")


def test_diverged_replay_rolls_main_back(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    info = lb.read_branch_json(sphere_dir, "exp")
    info["recipes"][0]["result"]["mu_sha256"] = "sha256:" + "0" * 64
    lb.write_branch_json(sphere_dir, "exp", info)
    meta_before = _meta(sphere_dir)
    norms_before = _norms(lance.dataset(str(_geo(sphere_dir))))
    with pytest.raises(GDSError, match="diverged"):
        hs.promote_branch("exp")
    assert _meta(sphere_dir) == meta_before
    assert _norms(lance.dataset(str(_geo(sphere_dir)))) == norms_before
    assert lb.branch_names(sphere_dir) == ["exp"]


def test_diverged_replay_restores_caches_the_recipes_do_not_record(sphere_dir: Path) -> None:
    import pyarrow as pa

    centroids = sphere_dir / "_gds_meta" / "temporal_centroids" / f"{PID}.lance"
    rows = pa.table(
        {
            "window_start": ["2024-01-01"],
            "window_end": ["2024-02-01"],
            "centroid": [[0.0, 1.0]],
            "entity_count": [3],
            "anomaly_rate": [0.1],
        }
    )
    lance.write_dataset(rows, str(centroids))
    hs = _branch_with_recipe(sphere_dir)
    info = lb.read_branch_json(sphere_dir, "exp")
    info["recipes"][0]["result"]["mu_sha256"] = "sha256:" + "0" * 64
    lb.write_branch_json(sphere_dir, "exp", info)

    with pytest.raises(GDSError, match="diverged"):
        hs.promote_branch("exp")

    assert lance.dataset(str(centroids)).count_rows() == 1


def test_promote_refuses_while_another_promotion_holds_the_lock(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    (lb.branches_dir(sphere_dir) / ".promote.lock").write_text("other", encoding="utf-8")
    with pytest.raises(GDSError, match="promotion"):
        hs.promote_branch("exp")


def test_recalibrate_reports_the_mu_fingerprint(sphere_dir: Path) -> None:
    result = HyperSphere.open(str(sphere_dir)).session("t").recalibrate(PID)
    assert result["mu_sha256"].startswith("sha256:")


def test_promote_ignores_index_only_commits_on_main(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    points = lance.dataset(str(sphere_dir / "points" / "customers" / "v=1" / "data.lance"))
    column = next(f.name for f in points.schema if f.name != "primary_key")
    points.create_scalar_index(column, "BTREE", replace=True)
    report = hs.promote_branch("exp")
    assert report.recipes_replayed == 1


def test_promote_refuses_a_data_change_on_a_pinned_dataset(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    path = sphere_dir / "points" / "customers" / "v=1" / "data.lance"
    points = lance.dataset(str(path))
    lance.write_dataset(points.to_table().slice(0, 1), str(path), mode="append")
    with pytest.raises(GDSError, match="main has moved"):
        hs.promote_branch("exp")


def test_promote_from_a_sphere_opened_on_another_branch_writes_main(sphere_dir: Path) -> None:
    _branch_with_recipe(sphere_dir)
    HyperSphere.open(str(sphere_dir)).create_branch("other")
    other_geo_before = (
        lance.dataset(str(_geo(sphere_dir))).checkout_version(("other", None)).version
    )
    overlay = lb.overlay_dir(sphere_dir, "exp") / "_gds_meta" / "sphere.json"
    branch_mu = json.loads(overlay.read_text("utf-8"))["patterns"][PID]["mu"]
    on_other = HyperSphere.open(str(sphere_dir), branch="other")
    on_other.promote_branch("exp")
    main = json.loads((sphere_dir / "_gds_meta" / "sphere.json").read_text("utf-8"))
    assert main["patterns"][PID]["mu"] == branch_mu
    other = lb.read_branch_json(sphere_dir, "other")
    assert other["recipes"] == []
    assert (
        lance.dataset(str(_geo(sphere_dir))).checkout_version(("other", None)).version
        == other_geo_before
    )
    assert lb.branch_names(sphere_dir) == ["other"]


def test_promote_checks_the_replayed_theta(sphere_dir: Path) -> None:
    hs = _branch_with_recipe(sphere_dir)
    info = lb.read_branch_json(sphere_dir, "exp")
    info["recipes"][0]["result"]["theta_norm"] += 1.0
    lb.write_branch_json(sphere_dir, "exp", info)
    with pytest.raises(GDSError, match="diverged"):
        hs.promote_branch("exp")
