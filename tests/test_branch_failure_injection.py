# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Failures injected in the middle of branch writes: main and the branch stay consistent."""

from __future__ import annotations

from pathlib import Path

import lance
import pytest

from hypertopos.builder.builder import GDSBuilder
from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage import calibration_history, writer
from hypertopos.storage.exceptions import GDSError
from tests.conftest import clone_sphere
from tests.test_branch_ingest import (  # noqa: F401  (base_sphere is a fixture)
    PID,
    _geo,
    _main_state,
    _new_rows,
    _norms,
    base_sphere,
)


@pytest.fixture
def sphere_dir(base_sphere: str, tmp_path: Path) -> Path:  # noqa: F811
    return clone_sphere(base_sphere, tmp_path / "s")


class Injected(Exception):
    pass


def _fail(*_a: object, **_k: object) -> None:
    raise Injected("injected")


def _assert_main_is(base: Path, before: dict[str, object]) -> None:
    after = _main_state(base)
    inventory = lb.dataset_inventory(base)
    for key, path in inventory.items():
        assert lb.same_rows(path, before[key], after[key]), key
    files = {k: v for k, v in before.items() if k not in inventory}
    assert {k: v for k, v in after.items() if k not in inventory} == files


def _branch_with_ingest_and_recalibration(base: Path) -> dict[str, float]:
    lb.create_branch(base, "b")
    builder = GDSBuilder("branch_ingest", str(base))
    builder.incremental_update(
        PID, changed_entities=_new_rows(), deleted_keys=["C1"], recalibrate="force", branch="b"
    )
    return _norms(lance.dataset(str(_geo(base))).checkout_version(("b", None)))


@pytest.mark.parametrize("error", [Injected, KeyboardInterrupt], ids=["error", "interrupt"])
def test_promote_interrupted_after_the_ingest_replay_wrote_geometry(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    branch_norms = _branch_with_ingest_and_recalibration(sphere_dir)
    before = _main_state(sphere_dir)

    def _raise(*_a: object, **_k: object) -> None:
        raise error()

    # The replayed ingest appends its rows, then writes sphere.json.
    monkeypatch.setattr(writer, "write_sphere_json", _raise)
    with pytest.raises(error):
        HyperSphere.open(str(sphere_dir)).promote_branch("b")
    monkeypatch.undo()

    _assert_main_is(sphere_dir, before)
    assert lb.branch_names(sphere_dir) == ["b"]
    HyperSphere.open(str(sphere_dir)).promote_branch("b")
    assert _norms(lance.dataset(str(_geo(sphere_dir)))) == pytest.approx(branch_norms)
    assert lb.branch_names(sphere_dir) == []


def test_promote_interrupted_inside_the_replayed_recalibration(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    branch_norms = _branch_with_ingest_and_recalibration(sphere_dir)
    before = _main_state(sphere_dir)

    # The replayed recalibration has rewritten the geometry columns when it
    # records the calibration epoch.
    monkeypatch.setattr(calibration_history, "write_calibration_history_epoch", _fail)
    with pytest.raises(Injected):
        HyperSphere.open(str(sphere_dir)).promote_branch("b")
    monkeypatch.undo()

    _assert_main_is(sphere_dir, before)
    assert lb.branch_names(sphere_dir) == ["b"]
    HyperSphere.open(str(sphere_dir)).promote_branch("b")
    assert _norms(lance.dataset(str(_geo(sphere_dir)))) == pytest.approx(branch_norms)


def test_promote_interrupted_while_deleting_the_branch(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    branch_norms = _branch_with_ingest_and_recalibration(sphere_dir)
    real_delete = lb.delete_branch

    def _delete_lance_branches_then_fail(base: Path | str, name: str) -> None:
        inventory = lb.dataset_inventory(base)
        for key, mechanism in lb.read_branch_json(base, name)["datasets"].items():
            if mechanism == lb.LANCE_BRANCH and key in inventory:
                lance.dataset(str(inventory[key])).branches.delete(name)
        raise Injected("injected")

    monkeypatch.setattr(lb, "delete_branch", _delete_lance_branches_then_fail)
    with pytest.raises(Injected):
        HyperSphere.open(str(sphere_dir)).promote_branch("b")
    monkeypatch.undo()

    # Main carries the branch; the half-deleted branch is finished by delete.
    assert _norms(lance.dataset(str(_geo(sphere_dir)))) == pytest.approx(branch_norms)
    real_delete(sphere_dir, "b")
    assert lb.branch_names(sphere_dir) == []


def test_create_branch_interrupted_leaves_no_branch(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lb, "write_branch_json", _fail)
    with pytest.raises(Injected):
        lb.create_branch(sphere_dir, "b")
    monkeypatch.undo()

    assert lb.branch_names(sphere_dir) == []
    for path in lb.dataset_inventory(sphere_dir).values():
        assert "b" not in lance.dataset(str(path)).branches.list()
    lb.create_branch(sphere_dir, "b")
    assert lb.branch_names(sphere_dir) == ["b"]


def test_branch_recalibration_interrupted_removes_the_branch_it_created(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _main_state(sphere_dir)
    monkeypatch.setattr(calibration_history, "write_calibration_history_epoch", _fail)
    with pytest.raises(Injected):
        HyperSphere.open(str(sphere_dir)).session("t").recalibrate(PID, branch="b")
    monkeypatch.undo()

    _assert_main_is(sphere_dir, before)
    assert lb.branch_names(sphere_dir) == []
    for path in lb.dataset_inventory(sphere_dir).values():
        assert "b" not in lance.dataset(str(path)).branches.list()


def test_branch_ingest_interrupted_after_its_geometry_write(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lb.create_branch(sphere_dir, "b")
    before = _main_state(sphere_dir)
    monkeypatch.setattr(writer, "write_sphere_json", _fail)
    with pytest.raises(Injected):
        GDSBuilder("branch_ingest", str(sphere_dir)).incremental_update(
            PID, changed_entities=_new_rows(), recalibrate="never", branch="b"
        )
    monkeypatch.undo()

    # Main is untouched; the branch holds a commit no recipe records, so it
    # cannot be promoted by mistake.
    _assert_main_is(sphere_dir, before)
    with pytest.raises(GDSError, match="not recorded|no recipes"):
        HyperSphere.open(str(sphere_dir)).promote_branch("b")
    _assert_main_is(sphere_dir, before)
