# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
import shutil
import subprocess
import sys
from pathlib import Path

import lance
import pytest

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "gds" / "sales_sphere"


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures() -> None:
    """Generate test fixtures once per session if they don't exist."""
    sphere_json = FIXTURES_PATH / "sphere.json"
    if sphere_json.exists():
        return
    fixture_script = Path(__file__).parent / "fixtures" / "generate_fixtures.py"
    subprocess.run(
        [sys.executable, str(fixture_script)],
        check=True,
    )


@pytest.fixture
def fixtures_path() -> Path:
    return FIXTURES_PATH


@pytest.fixture
def fixture_sphere_path() -> Path:
    return FIXTURES_PATH


@pytest.fixture
def sphere_path(fixtures_path) -> str:
    return str(fixtures_path)


# ── Sphere cloning for tests that mutate the copy ──


def clone_sphere(src: Path | str, dst: Path | str) -> Path:
    """Copy a sphere directory tree from *src* to *dst*, fast.

    For every Lance dataset directory found inside the tree (any directory
    that contains a ``_versions/`` child) ``shallow_clone`` writes only
    metadata and references — no data file rewrite. Non-Lance bits
    (sphere.json, calibration JSONs, traj indexes, FTS indexes living
    next to data.lance, etc.) fall through to ``shutil.copy2``.

    The win is most visible on Windows, where deep ``copytree`` over a sphere
    that contains many small Lance fragment files is dominated by per-file
    NTFS overhead.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists():
        raise FileExistsError(f"clone_sphere target already exists: {dst_path}")
    dst_path.mkdir(parents=True)

    for entry in src_path.iterdir():
        _clone_entry(entry, dst_path / entry.name)
    return dst_path


def _is_lance_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "_versions").exists()


def _clone_entry(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    if _is_lance_dataset_dir(src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        ds = lance.dataset(str(src))
        ds.shallow_clone(str(dst), reference=ds.version)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        _clone_entry(child, dst / child.name)


