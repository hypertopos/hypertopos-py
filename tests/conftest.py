# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import lance
import pytest

from hypertopos.engine.geometry import GDSEngine
from hypertopos.navigation.navigator import GDSNavigator

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "gds" / "sales_sphere"
SYNTHETIC_CHAIN_FIXTURES_PATH = (
    Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"
)


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures() -> None:
    """Generate test fixtures once per session if they don't exist."""
    sphere_json = FIXTURES_PATH / "sphere.json"
    if not sphere_json.exists():
        fixture_script = Path(__file__).parent / "fixtures" / "generate_fixtures.py"
        subprocess.run(
            [sys.executable, str(fixture_script)],
            check=True,
        )
    synthetic_sphere_json = SYNTHETIC_CHAIN_FIXTURES_PATH / "_gds_meta" / "sphere.json"
    if not synthetic_sphere_json.exists():
        synthetic_script = (
            Path(__file__).parent / "fixtures" / "synthetic_chain_sphere.py"
        )
        subprocess.run(
            [sys.executable, str(synthetic_script)],
            check=True,
        )


@pytest.fixture(autouse=True)
def _restore_navigator_engine_class_attrs() -> Iterator[None]:
    """Snapshot+restore GDSNavigator / GDSEngine class attributes around each test.

    Defends against the MagicMock-leak class: when a test replaces a class-level
    method via raw ``setattr(GDSNavigator, "...", MagicMock())`` (i.e. NOT via
    ``monkeypatch.setattr``) and then short-circuits via ``pytest.raises``, the
    MagicMock leaks into the next test that touches the same method. The teardown
    half of this fixture restores every non-dunder attribute whose identity
    changed during the test, regardless of how the test exited. ``monkeypatch``
    callers are unaffected — their own teardown runs first and restores the
    original, so the identity check here is a no-op for them.
    """
    snapshots = {
        cls: {
            name: cls.__dict__[name]
            for name in list(cls.__dict__)
            if not (name.startswith("__") and name.endswith("__"))
        }
        for cls in (GDSNavigator, GDSEngine)
    }
    try:
        yield
    finally:
        for cls, snap in snapshots.items():
            for name, original in snap.items():
                if cls.__dict__.get(name) is not original:
                    setattr(cls, name, original)


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
