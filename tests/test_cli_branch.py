# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""`hypertopos sphere branch` verb (in-process)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypertopos.cli.sphere_ops import run_sphere_branch
from tests.conftest import FIXTURES_PATH, clone_sphere


@pytest.fixture
def sphere_dir(tmp_path: Path) -> str:
    return str(clone_sphere(FIXTURES_PATH, tmp_path / "s"))


def _run(capsys: pytest.CaptureFixture[str], *args: object) -> dict:
    run_sphere_branch(*args)  # type: ignore[arg-type]
    return json.loads(capsys.readouterr().out)


def test_branch_verbs_round_trip(sphere_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
    created = _run(capsys, sphere_dir, "create", "exp", "try it", True)
    assert created["created"]["name"] == "exp" and created["created"]["purpose"] == "try it"
    listed = _run(capsys, sphere_dir, "list", None, "", True)
    assert [b["name"] for b in listed["branches"]] == ["exp"]
    assert _run(capsys, sphere_dir, "delete", "exp", "", True) == {"deleted": "exp"}
    assert _run(capsys, sphere_dir, "list", None, "", True) == {"branches": []}


def test_branch_verb_needs_a_name(sphere_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run_sphere_branch(sphere_dir, "create", None, "", False)
    assert exc.value.code == 1
    assert "needs a branch name" in capsys.readouterr().err


def test_promote_without_recipes_exits_one(
    sphere_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run_sphere_branch(sphere_dir, "create", "exp", "", True)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        run_sphere_branch(sphere_dir, "promote", "exp", "", False)
    assert exc.value.code == 1
    assert "no recipes" in capsys.readouterr().err
