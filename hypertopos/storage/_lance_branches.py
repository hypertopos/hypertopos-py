# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Sphere branch storage: create, list, verify and delete branches.

A branch ``b`` of a sphere is:

* a Lance branch named ``b`` on every dataset a branch write can rewrite
  (geometry and temporal centroids, the *eager set*), created from the
  dataset's latest version, so Lance retains the parent version by ref;
* for every other Lance dataset, the main version at creation recorded in
  ``branch.json`` (a *parent pin*); a full build, the only writer of those
  datasets, is refused while any branch exists;
* an overlay directory ``_gds_meta/branches/<b>/`` with a copy of every
  metadata file of ``_gds_meta`` and ``branch.json`` (provenance and the
  recipe log).

Every Lance ref operation runs on the resolved path: pylance percent-encodes
``~`` of a Windows 8.3 short path when it creates a ref directory and fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lance

from hypertopos.storage.exceptions import GDSError

BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RESERVED_NAMES = frozenset({"main"})
# A session pin: a main version, or (branch, version) on a branch.
Pin = int | tuple[str, int]
LANCE_BRANCH = "lance-branch"
PARENT_PIN = "parent-pin"
_REF_METADATA = {"hypertopos": "sphere-branch"}


@dataclass(frozen=True)
class BranchInfo:
    """One sphere branch as recorded in its ``branch.json``."""

    name: str
    created_at: str
    created_by: str
    purpose: str
    replayable: bool
    n_recipes: int
    parent_versions: dict[str, int]
    stale: bool
    # False when the branch directory has no branch.json (creation was
    # interrupted); delete_branch removes such a branch.
    complete: bool = True


def branches_dir(base: Path | str) -> Path:
    return Path(base) / "_gds_meta" / "branches"


def overlay_dir(base: Path | str, name: str) -> Path:
    """Branch directory: ``branch.json`` plus the branch's ``_gds_meta`` copy.

    It is laid out like a sphere root, so a helper that takes a sphere root
    and appends ``_gds_meta/...`` resolves the branch's copy when handed it.
    """
    return branches_dir(base) / name


def branch_names(base: Path | str) -> list[str]:
    """Names of the branches recorded under ``_gds_meta/branches/``."""
    root = branches_dir(base)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _is_eager(key: str) -> bool:
    return ":" not in key or key.startswith("centroids:")


def _key_for(rel: str) -> str | None:
    """Session pin key of a Lance dataset at ``rel`` (posix, sphere-relative)."""
    parts = rel.split("/")
    name = parts[-1]
    match parts:
        case ["geometry", pid, "data.lance"]:
            return pid
        case ["temporal", "_agents", *_]:
            return None
        case ["temporal", pid, "data.lance"]:
            return f"temporal:{pid}"
        case ["edges", pid, "data.lance"]:
            return f"edges:{pid}"
        case ["points", line, version, "data.lance"] if version.startswith("v="):
            return f"points:{line}:{version[2:]}"
        case ["forecast", "population", pid, "data.lance"]:
            return f"forecast:{pid}"
        case ["_gds_meta", "branches", *_] | ["_gds_meta", "topology_cache", *_]:
            return None
        case ["_gds_meta", "trajectory", _]:
            return f"traj:{name.removesuffix('.lance')}"
        case ["_gds_meta", "temporal_centroids", _]:
            return f"centroids:{name.removesuffix('.lance')}"
        case ["_gds_meta", "contagion_stats", _]:
            return f"contagion:{name.removesuffix('.lance')}"
        case ["_gds_meta", "edge_features", pid, "data.lance"]:
            return f"edge_features:{pid}"
        case ["_gds_meta", "conformance", "violations", pid, version] if version.startswith("v="):
            return f"conformance:{pid}:{version[2:].removesuffix('.lance')}"
    return f"lance:{rel}"


def _lance_dirs(base: Path) -> Iterator[Path]:
    for root, dirs, _files in os.walk(base):
        if "_versions" in dirs:
            dirs.clear()
            yield Path(root)


def dataset_inventory(base: Path | str) -> dict[str, Path]:
    """Every Lance dataset of the sphere a branch covers, by session pin key.

    Agent scratch, the per-branch overlays and the topology cache (recomputed
    on demand) are not part of a branch.
    """
    base = Path(base)
    inventory: dict[str, Path] = {}
    for path in _lance_dirs(base):
        key = _key_for(path.relative_to(base).as_posix())
        if key is not None:
            inventory[key] = path
    return inventory


def _fingerprints(sphere_json: dict[str, Any]) -> dict[str, str]:
    return {
        pid: "sha256:"
        + hashlib.sha256(json.dumps(node, sort_keys=True).encode("utf-8")).hexdigest()
        for pid, node in sphere_json.get("patterns", {}).items()
    }


def _validate_name(name: str) -> None:
    if name in _RESERVED_NAMES or not BRANCH_NAME_RE.match(name):
        raise GDSError(
            f"invalid branch name {name!r}: use letters, digits, '_', '.', '-' "
            "(starting with a letter or digit, at most 64 characters)"
        )


def write_branch_json(base: Path | str, name: str, data: dict[str, Any]) -> None:
    """Atomically replace ``_gds_meta/branches/<name>/branch.json``."""
    target = overlay_dir(base, name) / "branch.json"
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".branch.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_branch_json(base: Path | str, name: str) -> dict[str, Any]:
    path = overlay_dir(base, name) / "branch.json"
    if not path.is_file():
        raise GDSError(f"no branch {name!r} in sphere {base}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _copy_metadata(base: Path, overlay: Path) -> None:
    meta = base / "_gds_meta"
    for root, dirs, files in os.walk(meta):
        rel_root = Path(root).relative_to(meta)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d not in ("branches", "topology_cache")]
        dirs[:] = [d for d in dirs if not d.endswith(".lance")]
        for f in files:
            dst = overlay / rel_root / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(root) / f, dst)


_MAX_PATH = 260
# Longest relative path Lance writes under ``<dataset>/tree/<branch>``:
# geometry carries scalar index pages (``_indices/<uuid>/bitmap_page_lookup.lance``);
# a branch only empties centroids, so they get transaction files
# (``_transactions/<version>-<uuid>.txn``, two-digit version assumed).
_GEOMETRY_FILE_SUFFIX = 71
_CENTROIDS_FILE_SUFFIX = 58


def _long_paths_enabled() -> bool:
    """False only on Windows without ``LongPathsEnabled`` (the 260-character limit)."""
    if sys.platform != "win32":
        return True
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


def _check_branch_path_length(inventory: dict[str, Path], name: str) -> None:
    """Refuse a branch whose Lance files would pass the Windows path limit."""
    if _long_paths_enabled():
        return
    for key, path in sorted(inventory.items()):
        if not _is_eager(key):
            continue
        suffix = _GEOMETRY_FILE_SUFFIX if ":" not in key else _CENTROIDS_FILE_SUFFIX
        length = len(str(path)) + len(f"\\tree\\{name}") + suffix
        if length >= _MAX_PATH:
            raise GDSError(
                f"branch {name!r} cannot be created: its files under {path} would need "
                f"paths of about {length} characters, past the Windows limit of "
                f"{_MAX_PATH}. Move the sphere to a shorter directory, use a shorter "
                "branch name, or enable Windows long paths (LongPathsEnabled)."
            )


def create_branch(
    base: Path | str, name: str, *, created_by: str = "", purpose: str = ""
) -> dict[str, Any]:
    """Create branch ``name`` from the current main state; return its branch.json.

    Writes ~1 KB per eager-set dataset and copies no data file.
    """
    _validate_name(name)
    base = Path(base).resolve()
    overlay = overlay_dir(base, name)
    if overlay.exists():
        raise GDSError(f"branch {name!r} already exists in sphere {base}")
    sphere_json = json.loads((base / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"))
    inventory = dataset_inventory(base)
    _check_branch_path_length(inventory, name)

    try:
        overlay.mkdir(parents=True)
    except FileExistsError:
        raise GDSError(f"branch {name!r} already exists in sphere {base}") from None
    created: list[Path] = []
    try:
        _copy_metadata(base, overlay / "_gds_meta")
        versions: dict[str, int] = {}
        mechanisms: dict[str, str] = {}
        for key, path in sorted(inventory.items()):
            ds = lance.dataset(str(path))
            versions[key] = ds.latest_version
            if _is_eager(key):
                ds.create_branch(name, reference=ds.latest_version)
                created.append(path)
                ds.branches.replace_metadata(
                    name, {**_REF_METADATA, "created_at": datetime.now(UTC).isoformat()}
                )
                mechanisms[key] = LANCE_BRANCH
            else:
                mechanisms[key] = PARENT_PIN
        info = {
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "created_by": created_by,
            "purpose": purpose,
            "parent": {
                "lance_versions": versions,
                "meta_fingerprint": _fingerprints(sphere_json),
            },
            "datasets": mechanisms,
            "replayable": True,
            "recipes": [],
        }
        write_branch_json(base, name, info)
    except BaseException:
        for path in created:
            with suppress(Exception):
                lance.dataset(str(path)).branches.delete(name)
        shutil.rmtree(overlay, ignore_errors=True)
        raise
    return info


def verify_branch(base: Path | str, name: str) -> dict[str, Any]:
    """Return branch.json after checking every Lance branch it records exists."""
    base = Path(base).resolve()
    info = read_branch_json(base, name)
    inventory = dataset_inventory(base)
    for key, mechanism in sorted(info["datasets"].items()):
        if mechanism != LANCE_BRANCH:
            continue
        path = inventory.get(key)
        if path is None or name not in lance.dataset(str(path)).branches.list():
            raise GDSError(
                f"branch {name!r} is incomplete: dataset {key!r} has no Lance branch "
                f"{name!r}; delete the branch with delete_branch({name!r})"
            )
    return info


def delete_branch(base: Path | str, name: str) -> None:
    """Delete the Lance branches and the overlay of ``name``.

    Finishes a half-deleted branch: missing Lance branches are skipped. A
    branch whose creation was interrupted before ``branch.json`` was written
    is removed too, by deleting Lance branch ``name`` wherever it exists.
    """
    base = Path(base).resolve()
    overlay = overlay_dir(base, name)
    inventory = dataset_inventory(base)
    if overlay.is_dir() and not (overlay / "branch.json").is_file():
        keys = [k for k in inventory if _is_eager(k)]
    else:
        info = read_branch_json(base, name)
        keys = [k for k, m in info["datasets"].items() if m == LANCE_BRANCH]
    for key in sorted(keys):
        path = inventory.get(key)
        if path is None:
            continue
        ds = lance.dataset(str(path))
        if name in ds.branches.list():
            ds.branches.delete(name)
    shutil.rmtree(overlay)


def list_branches(base: Path | str) -> list[BranchInfo]:
    """Branches of the sphere, sorted by name.

    ``stale`` is True when main's geometry moved past the version the branch
    was created from (one manifest read per geometry dataset).
    """
    base = Path(base)
    out = []
    for name in branch_names(base):
        try:
            info = read_branch_json(base, name)
        except GDSError:
            out.append(BranchInfo(name, "", "", "", False, 0, {}, False, complete=False))
            continue
        parent = info["parent"]["lance_versions"]
        stale = False
        for key, version in parent.items():
            path = base / "geometry" / key / "data.lance"
            if ":" not in key and path.is_dir():
                stale = stale or lance.dataset(str(path)).latest_version != version
        out.append(
            BranchInfo(
                name=name,
                created_at=info["created_at"],
                created_by=info["created_by"],
                purpose=info["purpose"],
                replayable=info["replayable"],
                n_recipes=len(info["recipes"]),
                parent_versions=dict(parent),
                stale=stale,
            )
        )
    return out


_RECIPE_LOCK_WAIT_SECONDS = 30.0


@contextmanager
def recipe_lock(base: Path | str, name: str) -> Iterator[None]:
    """Hold ``branches/<name>/.recipes.lock``, waiting while another writer holds it.

    Taken by every recipe append and, after ``promote_lock``, for a whole
    promotion. A lock older than an hour is treated as left by a crashed writer.
    """
    path = overlay_dir(base, name) / ".recipes.lock"
    deadline = time.monotonic() + _RECIPE_LOCK_WAIT_SECONDS
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileNotFoundError:
            raise GDSError(f"no branch {name!r} in sphere {base}") from None
        except FileExistsError:
            with suppress(FileNotFoundError):
                if time.time() - path.stat().st_mtime > _LOCK_STALE_SECONDS:
                    path.unlink()
                    continue
            if time.monotonic() > deadline:
                raise GDSError(
                    f"another writer holds the recipe log of branch {name!r} (lock {path}); "
                    "retry when it ends"
                ) from None
            time.sleep(0.01)
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def append_recipe(base: Path | str, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Append ``entry`` to the branch's recipe log; return it with its ``seq``.

    Concurrent appends to one branch are serialised, so none is lost.
    """
    with recipe_lock(base, name):
        info = read_branch_json(base, name)
        recorded = {"seq": len(info["recipes"]) + 1, **entry}
        info["recipes"].append(recorded)
        write_branch_json(base, name, info)
    return recorded


@dataclass(frozen=True)
class PromoteReport:
    """Outcome of promoting a branch to main by replaying its recipes."""

    name: str
    recipes_replayed: int
    versions_before: dict[str, int]
    versions_after: dict[str, int]


_LOCK_STALE_SECONDS = 3600.0


@contextmanager
def promote_lock(base: Path | str) -> Iterator[None]:
    """Hold ``_gds_meta/branches/.promote.lock``; refuse while another promotion runs.

    A lock older than an hour is treated as left by a crashed promotion.
    """
    path = branches_dir(base) / ".promote.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(FileNotFoundError):
        if time.time() - path.stat().st_mtime > _LOCK_STALE_SECONDS:
            path.unlink()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise GDSError(
            f"another promotion is running on {base} (lock {path}); retry when it ends"
        ) from None
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def main_versions(base: Path | str, keys: Iterable[str]) -> dict[str, int]:
    """Main latest version of each dataset key present in the sphere."""
    inventory = dataset_inventory(base)
    return {
        key: lance.dataset(str(inventory[key])).latest_version for key in keys if key in inventory
    }


def _row_content(ds: lance.LanceDataset) -> list[tuple[Any, ...]]:
    """What a version's rows are made of: each fragment's data and deletion files."""
    content = []
    for fragment in ds.get_fragments():
        deletion = fragment.deletion_file()
        content.append(
            (
                fragment.fragment_id,
                tuple(sorted(f.path for f in fragment.data_files())),
                deletion if isinstance(deletion, str) or deletion is None else str(deletion),
            )
        )
    return sorted(content)


def same_rows(path: Path | str, version_a: int, version_b: int) -> bool:
    """True when two versions of a dataset hold the same rows.

    Index builds and other metadata-only commits keep the data and deletion
    files, so they compare equal.
    """
    ds = lance.dataset(str(path))
    return _row_content(ds.checkout_version(version_a)) == _row_content(
        ds.checkout_version(version_b)
    )


def check_promotable(base: Path | str, name: str) -> dict[str, Any]:
    """Return branch.json once every precondition of a promotion holds.

    The branch is replayable and has recipes; main has not moved since the
    branch was created (Lance versions and pattern nodes); no commit landed
    on the branch that a recipe did not record.
    """
    base = Path(base).resolve()
    info = verify_branch(base, name)
    if not info["recipes"]:
        raise GDSError(f"branch {name!r} has no recipes to promote")
    if not info["replayable"]:
        raise GDSError(f"branch {name!r} is not replayable")
    parent = info["parent"]["lance_versions"]
    inventory = dataset_inventory(base)
    for key, now in main_versions(base, parent).items():
        # A commit that adds an index but keeps the rows (main readers build
        # scalar indexes lazily) does not move main.
        if now != parent[key] and not same_rows(inventory[key], parent[key], now):
            raise GDSError(
                f"main has moved since branch {name!r} was created: {key!r} was at "
                f"v{parent[key]}, main is at v{now}; delete the branch and recreate it"
            )
    sphere_json = json.loads((base / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"))
    current = _fingerprints(sphere_json)
    for pid in {r["pattern_id"] for r in info["recipes"]}:
        if current.get(pid) != info["parent"]["meta_fingerprint"].get(pid):
            raise GDSError(
                f"main has moved since branch {name!r} was created: pattern {pid!r} "
                "calibration changed; delete the branch and recreate it"
            )
    recorded: dict[str, int] = {}
    for recipe in info["recipes"]:
        recorded.update(recipe.get("lance_version_after", {}))
    for key, version in recorded.items():
        head = lance.dataset(str(inventory[key])).checkout_version((name, None)).version
        if head != version:
            raise GDSError(
                f"branch {name!r} has a commit on {key!r} (v{head}) not recorded by "
                f"its recipes (last recorded v{version}); it cannot be replayed"
            )
    return info


def _meta_paths_for(base: Path, pattern_ids: Iterable[str]) -> list[Path]:
    meta = base / "_gds_meta"
    paths = [meta / "sphere.json"]
    for pid in pattern_ids:
        paths += sorted((meta / "geometry_stats").glob(f"{pid}_v*.json"))
        paths += [meta / "calibration" / f"{pid}.json", meta / "calibration_history" / pid]
    return paths


def backup_main_metadata(base: Path | str, name: str, pattern_ids: Iterable[str]) -> Path:
    """Copy main metadata a replay may change to ``branches/<name>/.promote-backup``."""
    base = Path(base).resolve()
    pids = list(pattern_ids)
    backup = overlay_dir(base, name) / ".promote-backup"
    shutil.rmtree(backup, ignore_errors=True)
    meta = base / "_gds_meta"
    present: list[str] = []
    for path in _meta_paths_for(base, pids):
        if not path.exists():
            continue
        rel = path.relative_to(meta)
        present.append(rel.as_posix())
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dst)
        else:
            shutil.copy2(path, dst)
    (backup / "manifest.json").write_text(
        json.dumps({"pattern_ids": pids, "present": present}), encoding="utf-8"
    )
    return backup


def restore_main_metadata(base: Path | str, name: str) -> None:
    """Put main metadata back as backed up; remove files the replay created."""
    base = Path(base).resolve()
    backup = overlay_dir(base, name) / ".promote-backup"
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    meta = base / "_gds_meta"
    for path in _meta_paths_for(base, manifest["pattern_ids"]):
        rel = path.relative_to(meta).as_posix()
        if rel in manifest["present"]:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    for rel in manifest["present"]:
        src, dst = backup / rel, meta / rel
        if src.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            os.replace(shutil.copy2(src, dst.with_name(dst.name + ".restore")), dst)


def restore_main_versions(base: Path | str, versions: dict[str, int]) -> None:
    """Commit, on main, the content of each dataset at the given version."""
    inventory = dataset_inventory(base)
    for key, version in versions.items():
        ds = lance.dataset(str(inventory[key]))
        if ds.latest_version != version:
            ds.checkout_version(version).restore()  # type: ignore[no-untyped-call]


def open_ref(path: Path | str, branch: str | None) -> lance.LanceDataset:
    """A Lance dataset at the head of ``branch`` (main when ``branch`` is None)."""
    if branch is None:
        return lance.dataset(str(path))
    return lance.dataset(str(Path(path).resolve())).checkout_version((branch, None))
