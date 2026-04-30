# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Multi-epoch calibration retention — exception types, schema-hash helpers,
JSON serialization for CalibrationFit, and history write/GC helpers.

On-disk layout: `_gds_meta/calibration_history/{pattern_id}/v={N}.json`
"""
from __future__ import annotations

import hashlib
import json as _json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hypertopos.navigation.navigator import GDSError


class CalibrationNotFoundError(GDSError):
    """Requested calibration epoch is not on disk — wrong N, GC'd, or schema bump wiped history."""


def compute_pattern_schema_hash(payload: dict[str, Any]) -> str:
    """Deterministic sha256 hex digest over schema-relevant pattern fields.

    The payload MUST contain (and only contain) the following keys, with the
    semantics defined in the M1 design §6.1:
      - relations: list of {"line_id": str, "event_columns": list[str]}
      - event_dimensions: list[str]
      - prop_columns: list[str]
      - dimension_kinds: list[str]

    `sort_keys=True` makes the digest insensitive to dict-key ordering inside
    a single relation entry, but list ORDER (relations, dimension_kinds) is
    significant — order corresponds to dimension index in the shape vector.
    """
    encoded = _json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_schema_hash_from_pattern_node(pattern_node: dict[str, Any]) -> str:
    """2.3-fallback: reconstruct schema_hash from a sphere.json patterns.{pid} node.

    Best-effort — used only when reading a 2.3 sphere that has no explicit
    `schema_hash` field. Self-description label, NOT a correctness check;
    the post-rebuild builder always writes a fresh hash before any
    reset-decision is made.
    """
    relations_raw = pattern_node.get("relations") or []
    relations = []
    for rel in relations_raw:
        relations.append(
            {
                "line_id": rel.get("line_id") or rel.get("line"),
                "event_columns": list(rel.get("event_columns") or []),
            }
        )
    payload = {
        "relations": relations,
        "event_dimensions": list(pattern_node.get("event_dimensions") or []),
        "prop_columns": list(pattern_node.get("prop_columns") or []),
        "dimension_kinds": list(pattern_node.get("dimension_kinds") or []),
    }
    return compute_pattern_schema_hash(payload)


# ---------------------------------------------------------------------------
# JSON serialization helpers for CalibrationFit
# ---------------------------------------------------------------------------


def _opt_list(arr: np.ndarray | None) -> list | None:
    return None if arr is None else arr.astype(np.float32).tolist()


def _opt_array(value: list | None) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=np.float32)


def _format_dt(value: datetime) -> str:
    return value.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def serialize_fit(fit: Any) -> dict[str, Any]:
    """Convert a CalibrationFit into a JSON-serializable dict."""
    return {
        "pattern_id": fit.pattern_id,
        "calibration_epoch": fit.calibration_epoch,
        "schema_version": fit.schema_version,
        "schema_hash": fit.schema_hash,
        "mu": fit.mu.astype(np.float32).tolist(),
        "sigma_diag": fit.sigma_diag.astype(np.float32).tolist(),
        "theta": fit.theta.astype(np.float32).tolist(),
        "population_size": fit.population_size,
        "dimension_weights": _opt_list(fit.dimension_weights),
        "dimension_kinds": fit.dimension_kinds,
        "dim_percentiles": fit.dim_percentiles,
        "group_stats": fit.group_stats,
        "gmm_components": fit.gmm_components,
        "edge_max": _opt_list(fit.edge_max),
        "computed_at": _format_dt(fit.computed_at),
        "last_calibrated_at": _format_dt(fit.last_calibrated_at),
    }


def deserialize_fit(blob: dict[str, Any]) -> Any:
    """Reconstruct a CalibrationFit from a JSON-loaded dict."""
    from hypertopos.model.sphere import CalibrationFit

    return CalibrationFit(
        pattern_id=blob["pattern_id"],
        calibration_epoch=blob["calibration_epoch"],
        schema_version=blob["schema_version"],
        schema_hash=blob["schema_hash"],
        mu=np.asarray(blob["mu"], dtype=np.float32),
        sigma_diag=np.asarray(blob["sigma_diag"], dtype=np.float32),
        theta=np.asarray(blob["theta"], dtype=np.float32),
        population_size=blob["population_size"],
        dimension_weights=_opt_array(blob.get("dimension_weights")),
        dimension_kinds=blob.get("dimension_kinds"),
        dim_percentiles=blob.get("dim_percentiles"),
        group_stats=blob.get("group_stats"),
        gmm_components=blob.get("gmm_components"),
        edge_max=_opt_array(blob.get("edge_max")),
        computed_at=_parse_dt(blob["computed_at"]),
        last_calibrated_at=_parse_dt(blob["last_calibrated_at"]),
    )


# ---------------------------------------------------------------------------
# History write / GC helpers
# ---------------------------------------------------------------------------

_VERSION_FILENAME_RE = re.compile(r"^v=(\d+)\.json$")


def history_dir(base: Path, pattern_id: str) -> Path:
    """Return the path to `_gds_meta/calibration_history/{pattern_id}/`."""
    return Path(base) / "_gds_meta" / "calibration_history" / pattern_id


def list_calibration_versions(base: Path, pattern_id: str) -> list[int]:
    """Return calibration epochs present on disk for pattern_id, ascending."""
    pdir = history_dir(base, pattern_id)
    if not pdir.exists():
        return []
    versions: list[int] = []
    for entry in pdir.iterdir():
        if not entry.is_file():
            continue
        m = _VERSION_FILENAME_RE.match(entry.name)
        if m:
            versions.append(int(m.group(1)))
    versions.sort()
    return versions


def write_calibration_history_epoch(base: Path, fit: Any, last_k: int) -> Path:
    """Write the fit to `v={epoch}.json` and trim oldest if count > last_k.

    Returns the path written. Caller is responsible for ensuring `fit.calibration_epoch`
    is the correct N (reset -> 1, increment -> previous + 1).

    GC: after the write, files older than the most-recent `last_k` are deleted.
    `last_k < 1` is rejected here as a defensive check (callers should validate
    sphere.json policy at sphere-load time, but this guard catches programmer error).
    """
    if last_k < 1:
        raise ValueError(f"last_k must be >= 1, got {last_k}")

    pdir = history_dir(base, fit.pattern_id)
    pdir.mkdir(parents=True, exist_ok=True)
    out = pdir / f"v={fit.calibration_epoch}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(serialize_fit(fit), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)

    versions = list_calibration_versions(base, fit.pattern_id)
    if len(versions) > last_k:
        to_delete = versions[: len(versions) - last_k]
        for n in to_delete:
            (pdir / f"v={n}.json").unlink()

    return out


def reset_calibration_history(base: Path, pattern_id: str) -> None:
    """Wipe `_gds_meta/calibration_history/{pid}/` entirely (used on schema change).

    Removes the entire pattern history directory including any sidecar files
    or future subdirectories.
    """
    import shutil
    pdir = history_dir(base, pattern_id)
    if pdir.exists():
        shutil.rmtree(pdir)
