# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Incremental ingest on a sphere branch, its recipes, and their promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from hypertopos.builder.builder import GDSBuilder
from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from tests.conftest import clone_sphere

PID = "cust_pattern"


@pytest.fixture(scope="module")
def base_sphere(tmp_path_factory: pytest.TempPathFactory) -> str:
    out = str(tmp_path_factory.mktemp("branch_ingest") / "gds")
    rng = np.random.default_rng(7)
    n = 100
    custs = pa.table({"primary_key": [f"C{i}" for i in range(n)]})
    events = pa.table(
        {
            "primary_key": [f"E{i}" for i in range(n * 5)],
            "cust_fk": [f"C{rng.integers(0, n)}" for _ in range(n * 5)],
        }
    )
    builder = GDSBuilder("branch_ingest", out)
    builder.add_line("customers", custs, key_col="primary_key", source_id="t", role="anchor")
    builder.add_line("events", events, key_col="primary_key", source_id="t", role="event")
    builder.add_derived_dimension("customers", "events", "cust_fk", "count", None, "event_count")
    builder.add_pattern("cust_pattern", "anchor", "customers", relations=[])
    builder.build()
    return out


@pytest.fixture
def sphere_dir(base_sphere: str, tmp_path: Path) -> Path:
    return clone_sphere(base_sphere, tmp_path / "s")


def _new_rows(n: int = 10, value: float = 60.0) -> pa.Table:
    return pa.table(
        {
            "primary_key": [f"N{i}" for i in range(n)],
            "event_count": pa.array([value] * n, type=pa.float64()),
        }
    )


def _geo(base: Path) -> Path:
    return base / "geometry" / PID / "data.lance"


def _main_state(base: Path) -> dict[str, object]:
    state: dict[str, object] = {
        k: lance.dataset(str(p)).latest_version for k, p in lb.dataset_inventory(base).items()
    }
    meta = base / "_gds_meta"
    for p in meta.rglob("*"):
        rel = p.relative_to(meta).as_posix()
        if p.is_file() and ".lance/" not in rel and not rel.startswith("branches/"):
            state[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return state


def _norms(ds: lance.LanceDataset) -> dict[str, float]:
    t = ds.to_table(columns=["primary_key", "delta_norm"])
    return dict(zip(t["primary_key"].to_pylist(), t["delta_norm"].to_pylist(), strict=True))


def test_branch_ingest_leaves_every_main_dataset_untouched(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "b")
    before = _main_state(sphere_dir)
    n_before = lance.dataset(str(_geo(sphere_dir))).count_rows()
    builder = GDSBuilder("branch_ingest", str(sphere_dir))
    result = builder.incremental_update(
        PID, changed_entities=_new_rows(), recalibrate="never", reindex=True, branch="b"
    )
    builder.finalize_incremental(PID, branch="b")
    assert result.added == 10
    assert _main_state(sphere_dir) == before
    branch = lance.dataset(str(_geo(sphere_dir))).checkout_version(("b", None))
    assert branch.count_rows() == n_before + 10
    # Ranks were recomputed over the whole branch population: the appended
    # rows (shapes clip at the maximum) share the top rank with the old maximum.
    ranks = branch.to_table(columns=["primary_key", "score_rank_pct"])
    pct = ranks["score_rank_pct"].to_numpy()
    top_keys = {
        k for k, r in zip(ranks["primary_key"].to_pylist(), pct, strict=True) if r == pct.max()
    }
    assert {f"N{i}" for i in range(10)} <= top_keys
    overlay = json.loads(
        (lb.overlay_dir(sphere_dir, "b") / "_gds_meta" / "sphere.json").read_text("utf-8")
    )
    assert overlay["patterns"][PID]["population_size"] == n_before + 10


def test_branch_ingest_records_recipes(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "b")
    builder = GDSBuilder("branch_ingest", str(sphere_dir))
    builder.incremental_update(
        PID, changed_entities=_new_rows(), deleted_keys=["C0"], recalibrate="never", branch="b"
    )
    builder.finalize_incremental(PID, branch="b")
    ingest, finalize = lb.read_branch_json(sphere_dir, "b")["recipes"]
    assert ingest["op"] == "incremental_update" and finalize["op"] == "finalize_incremental"
    points = lb.overlay_dir(sphere_dir, "b") / ingest["points_file"]
    assert points.is_file()
    assert ingest["points_sha256"] == "sha256:" + hashlib.sha256(points.read_bytes()).hexdigest()
    assert ingest["deleted_keys"] == ["C0"]
    assert ingest["result"]["added"] == 10 and ingest["result"]["deleted"] == 1
    head = lance.dataset(str(_geo(sphere_dir))).checkout_version(("b", None)).version
    assert finalize["lance_version_after"][PID] == head


def test_auto_recalibration_on_a_branch_is_its_own_recipe(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "b")
    builder = GDSBuilder("branch_ingest", str(sphere_dir))
    result = builder.incremental_update(
        PID, changed_entities=_new_rows(), recalibrate="force", branch="b"
    )
    assert result.recalibrated
    ops = [r["op"] for r in lb.read_branch_json(sphere_dir, "b")["recipes"]]
    assert ops == ["incremental_update", "recalibrate"]
    main = json.loads((sphere_dir / "_gds_meta" / "sphere.json").read_text("utf-8"))
    overlay = json.loads(
        (lb.overlay_dir(sphere_dir, "b") / "_gds_meta" / "sphere.json").read_text("utf-8")
    )
    assert overlay["patterns"][PID]["mu"] != main["patterns"][PID]["mu"]


def test_promote_replays_ingest_and_recalibration(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "b")
    builder = GDSBuilder("branch_ingest", str(sphere_dir))
    builder.incremental_update(
        PID, changed_entities=_new_rows(), deleted_keys=["C1"], recalibrate="force", branch="b"
    )
    builder.finalize_incremental(PID, branch="b")
    branch_norms = _norms(lance.dataset(str(_geo(sphere_dir))).checkout_version(("b", None)))
    report = HyperSphere.open(str(sphere_dir)).promote_branch("b")
    assert report.recipes_replayed == 3
    assert _norms(lance.dataset(str(_geo(sphere_dir)))) == pytest.approx(branch_norms)
    assert lb.branch_names(sphere_dir) == []


def test_failed_promote_after_an_ingest_replay_restores_all_of_main(sphere_dir: Path) -> None:
    from hypertopos.storage.exceptions import GDSError

    lb.create_branch(sphere_dir, "b")
    builder = GDSBuilder("branch_ingest", str(sphere_dir))
    builder.incremental_update(PID, changed_entities=_new_rows(), recalibrate="force", branch="b")
    info = lb.read_branch_json(sphere_dir, "b")
    info["recipes"][-1]["result"]["mu_sha256"] = "sha256:" + "0" * 64
    lb.write_branch_json(sphere_dir, "b", info)
    before = _main_state(sphere_dir)

    with pytest.raises(GDSError, match="diverged"):
        HyperSphere.open(str(sphere_dir)).promote_branch("b")

    # Rollback restores content with a new Lance commit, so compare rows, not
    # version numbers, for datasets; metadata files must be byte-identical.
    after = _main_state(sphere_dir)
    inventory = lb.dataset_inventory(sphere_dir)
    for key, path in inventory.items():
        assert lb.same_rows(path, before[key], after[key]), key
    files = {k: v for k, v in before.items() if k not in inventory}
    assert {k: v for k, v in after.items() if k not in inventory} == files


def test_excluding_extreme_entities_flips_anomalies(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    table = lance.dataset(str(_geo(sphere_dir))).to_table(columns=["primary_key", "delta_norm"])
    order = np.argsort(-table["delta_norm"].to_numpy())
    extreme = [table["primary_key"][int(i)].as_py() for i in order[:10]]
    hs.session("t").recalibrate(PID, branch="x", exclude_keys=extreme)
    report = hs.session("u").navigator().compare_branches(PID, "x")
    assert report.flipped_in + report.flipped_out >= 1


def test_cli_ingest_into_a_branch(
    sphere_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from hypertopos.cli.sphere_ops import run_sphere_ingest

    lb.create_branch(sphere_dir, "b")
    points = tmp_path / "rows.arrow"
    with pa.OSFile(str(points), "wb") as sink, pa.ipc.new_file(sink, _new_rows().schema) as w:
        w.write_table(_new_rows())
    before = _main_state(sphere_dir)
    run_sphere_ingest(str(sphere_dir), str(points), PID, "never", False, True, True, "b")
    payload = json.loads(capsys.readouterr().out)
    assert payload["branch"] == "b" and payload["added"] == 10
    assert payload["geometry_version_after"] > payload["geometry_version_before"]
    assert _main_state(sphere_dir) == before
    ops = [r["op"] for r in lb.read_branch_json(sphere_dir, "b")["recipes"]]
    assert ops == ["incremental_update", "finalize_incremental"]


def test_flip_report_carries_thresholds_and_causes(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    table = lance.dataset(str(_geo(sphere_dir))).to_table(columns=["primary_key", "delta_norm"])
    order = np.argsort(-table["delta_norm"].to_numpy())
    hs.session("t").recalibrate(
        PID, branch="x", exclude_keys=[table["primary_key"][int(i)].as_py() for i in order[:10]]
    )
    report = hs.session("u").navigator().compare_branches(PID, "x")
    # A calibrated pattern's threshold is its lowest flagged anomaly_score.
    main = lance.dataset(str(_geo(sphere_dir))).to_table(
        columns=["anomaly_score"], filter="is_anomaly = true"
    )
    assert report.theta_from == pytest.approx(min(main["anomaly_score"].to_pylist()))
    assert report.theta_to != report.theta_from
    flipped = report.flipped_in_top + report.flipped_out_top
    assert flipped
    assert {row["cause"] for row in flipped} <= {"threshold", "score", "both"}
