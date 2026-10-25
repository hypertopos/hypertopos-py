# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Recalibrating on a sphere branch, excluding keys, recording the recipe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import lance
import pyarrow as pa
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


def _centroids(base: Path) -> Path:
    path = base / "_gds_meta" / "temporal_centroids" / f"{PID}.lance"
    table = pa.table(
        {
            "window_start": ["2024-01-01"],
            "window_end": ["2024-02-01"],
            "centroid": [[0.0, 1.0]],
            "entity_count": [3],
            "anomaly_rate": [0.1],
        }
    )
    lance.write_dataset(table, str(path))
    return path


def _main_state(base: Path) -> dict[str, object]:
    state: dict[str, object] = {}
    for key, path in lb.dataset_inventory(base).items():
        state[key] = lance.dataset(str(path)).latest_version
    meta = base / "_gds_meta"
    for p in meta.rglob("*"):
        rel = p.relative_to(meta).as_posix()
        if p.is_file() and ".lance/" not in rel and not rel.startswith("branches/"):
            state[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return state


def _keys(base: Path) -> list[str]:
    return (
        lance.dataset(str(_geo(base))).to_table(columns=["primary_key"])["primary_key"].to_pylist()
    )


def test_branch_recalibrate_leaves_every_main_dataset_untouched(sphere_dir: Path) -> None:
    _centroids(sphere_dir)
    before = _main_state(sphere_dir)
    geo_before = lance.dataset(str(_geo(sphere_dir))).latest_version
    session = HyperSphere.open(str(sphere_dir)).session("t")
    result = session.recalibrate(PID, branch="exp", exclude_keys=[_keys(sphere_dir)[0]])
    assert result["branch"] == "exp"
    assert _main_state(sphere_dir) == before
    branch_geo = lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None))
    assert branch_geo.version > geo_before
    overlay = json.loads(
        (lb.overlay_dir(sphere_dir, "exp") / "_gds_meta" / "sphere.json").read_text("utf-8")
    )
    main = json.loads((sphere_dir / "_gds_meta" / "sphere.json").read_text("utf-8"))
    assert overlay["patterns"][PID]["mu"] != main["patterns"][PID]["mu"]
    assert overlay["patterns"][PID]["calibration_epoch"] == (
        main["patterns"][PID].get("calibration_epoch", 1) + 1
    )


def test_branch_recalibrate_empties_branch_centroids_only(sphere_dir: Path) -> None:
    path = _centroids(sphere_dir)
    lb.create_branch(sphere_dir, "exp")
    HyperSphere.open(str(sphere_dir)).session("t", branch="exp").recalibrate(PID)
    assert lance.dataset(str(path)).count_rows() == 1
    assert lance.dataset(str(path)).checkout_version(("exp", None)).count_rows() == 0


def test_branch_recalibrate_records_recipe(sphere_dir: Path) -> None:
    excluded = _keys(sphere_dir)[0]
    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")
    result = session.recalibrate(PID, exclude_keys=[excluded])
    info = lb.read_branch_json(sphere_dir, "exp")
    [recipe] = info["recipes"]
    assert recipe["seq"] == 1 and recipe["op"] == "recalibrate"
    assert recipe["pattern_id"] == PID and recipe["exclude_keys"] == [excluded]
    assert recipe["soft_threshold"] is None and recipe["hard_threshold"] is None
    assert recipe["result"]["n_excluded"] == 1
    assert recipe["result"]["population_size"] == result["records_recalibrated"]
    assert recipe["result"]["mu_sha256"].startswith("sha256:")
    head = lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None)).version
    assert recipe["lance_version_after"][PID] == head
    assert info["replayable"] is True


def test_branch_session_reads_its_own_recalibration(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")
    result = session.recalibrate(PID, exclude_keys=[_keys(sphere_dir)[0]])
    pattern = session.sphere().patterns[PID]
    assert round(pattern.theta_norm, 4) == result["new_theta_norm"]
    assert session.is_stale().stale is False


def test_exclude_keys_changes_the_fit_on_main(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    mu_before = hs.session("a").sphere().patterns[PID].mu
    result = hs.session("b").recalibrate(PID, exclude_keys=[_keys(sphere_dir)[0]])
    assert result["n_excluded"] == 1
    assert list(hs.session("c").sphere().patterns[PID].mu) != list(mu_before)


def test_unknown_exclude_key_raises(sphere_dir: Path) -> None:
    session = HyperSphere.open(str(sphere_dir)).session("t")
    with pytest.raises(ValueError, match="no-such-key"):
        session.recalibrate(PID, exclude_keys=["no-such-key"])


def test_branch_argument_must_match_branch_session(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")
    with pytest.raises(ValueError, match="other"):
        session.recalibrate(PID, branch="other")


def test_recalibrate_on_existing_branch_from_main_appends_recipe(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t")
    session.recalibrate(PID, branch="exp")
    session.recalibrate(PID, branch="exp")
    assert [r["seq"] for r in lb.read_branch_json(sphere_dir, "exp")["recipes"]] == [1, 2]


def test_main_recalibrate_records_no_recipe_and_no_branch(sphere_dir: Path) -> None:
    HyperSphere.open(str(sphere_dir)).session("t").recalibrate(PID)
    assert lb.branch_names(sphere_dir) == []


def test_branch_recalibrate_refuses_when_branch_moved(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")
    HyperSphere.open(str(sphere_dir)).session("u", branch="exp").recalibrate(PID)
    with pytest.raises(GDSError, match="changed"):
        session.recalibrate(PID)


def test_failed_branch_recalibrate_from_main_removes_the_branch_it_created(
    sphere_dir: Path,
) -> None:
    session = HyperSphere.open(str(sphere_dir)).session("t")
    with pytest.raises(ValueError, match="no-such-key"):
        session.recalibrate(PID, branch="exp", exclude_keys=["no-such-key"])
    with pytest.raises(KeyError):
        session.recalibrate("nope_pattern", branch="exp2")
    assert lb.branch_names(sphere_dir) == []


def test_failed_branch_recalibrate_keeps_a_branch_that_existed(sphere_dir: Path) -> None:
    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t")
    with pytest.raises(ValueError):
        session.recalibrate(PID, branch="exp", exclude_keys=["no-such-key"])
    assert lb.branch_names(sphere_dir) == ["exp"]


def test_exclude_keys_must_not_be_a_string(sphere_dir: Path) -> None:
    session = HyperSphere.open(str(sphere_dir)).session("t")
    with pytest.raises(TypeError, match="exclude_keys"):
        session.recalibrate(PID, exclude_keys=_keys(sphere_dir)[0])


def test_excluding_every_entity_raises(sphere_dir: Path) -> None:
    session = HyperSphere.open(str(sphere_dir)).session("t")
    with pytest.raises(ValueError, match="no entity left"):
        session.recalibrate(PID, exclude_keys=_keys(sphere_dir))


def test_recipe_is_recorded_before_the_session_refreshes(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hypertopos.sphere import HyperSession
    from hypertopos.storage.exceptions import GDSStorageError

    lb.create_branch(sphere_dir, "exp")
    session = HyperSphere.open(str(sphere_dir)).session("t", branch="exp")

    def busy(self: HyperSession) -> None:
        raise GDSStorageError("sphere is being written")

    monkeypatch.setattr(HyperSession, "refresh", busy)
    with pytest.raises(GDSStorageError):
        session.recalibrate(PID)
    [recipe] = lb.read_branch_json(sphere_dir, "exp")["recipes"]
    head = lance.dataset(str(_geo(sphere_dir))).checkout_version(("exp", None)).version
    assert recipe["lance_version_after"][PID] == head


@pytest.mark.skipif(os.name != "nt", reason="MAX_PATH applies to Windows only")
def test_branch_past_windows_max_path_is_refused_with_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main paths fit in 260 characters, the branch's data files would not."""
    monkeypatch.setattr(lb, "_long_paths_enabled", lambda: False)
    name = "what-if-without-known-fraud-01"
    # geometry/<pid>/data.lance + tree/<name>/data/<file>.lance is about base + 135.
    root = tmp_path.resolve()
    pad = 135 - len(str(root)) - len("/s") - 1
    if pad < 1:
        pytest.skip("temporary directory is already too long")
    base = clone_sphere(FIXTURES_PATH, root / ("p" * pad) / "s")
    longest_main = max(
        len(os.path.join(r, n)) for r, dirs, files in os.walk(base) for n in dirs + files
    )
    assert longest_main < 260
    before = _main_state(base)
    session = HyperSphere.open(str(base)).session("t")
    with pytest.raises(GDSError, match=r"260") as excinfo:
        session.recalibrate(PID, branch=name, exclude_keys=[_keys(base)[0]])
    message = str(excinfo.value)
    assert "shorter" in message and str(_geo(base)) in message
    assert lb.branch_names(base) == []
    assert _main_state(base) == before
