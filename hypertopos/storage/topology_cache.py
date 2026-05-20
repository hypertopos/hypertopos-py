# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Sidecar Lance cache for topology anomaly scores.

Cache lives at ``<sphere_root>/_gds_meta/topology_cache/<pattern_id>/v=<N>.lance``
where ``N`` is the pattern version at scoring time. Different versions live
side by side; on a pattern re-calibration the new version writes a fresh
``v=N+1.lance`` and the stale file is collected by the regular sphere GC pass.

Two schemas live here: ``ANOMALIES_SCHEMA`` for ``find_topological_anomalies``
output and ``TRAJECTORY_SCHEMA`` for ``find_topological_trajectory_anomalies``.
"""
from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa

ANOMALIES_SCHEMA = pa.schema([
    ("primary_key", pa.string()),
    ("topo_score", pa.float64()),
    ("h1_max_persistence", pa.float64()),
    ("h0_mean_death", pa.float64()),
    ("n_h1_features", pa.int32()),
    ("computed_at", pa.timestamp("us", tz="UTC")),
])

TRAJECTORY_SCHEMA = pa.schema([
    ("primary_key", pa.string()),
    ("trajectory_topo_score", pa.float64()),
    ("n_timesteps", pa.int32()),
    ("h1_total_persistence", pa.float64()),
    ("dominant_feature_birth", pa.float64()),
    ("dominant_feature_death", pa.float64()),
    ("computed_at", pa.timestamp("us", tz="UTC")),
])


def cache_path(sphere_root: Path, kind: str, pattern_id: str, version: int) -> Path:
    """Resolve the sidecar Lance path for one (kind, pattern_id, version) tuple.

    ``kind`` is ``"anomalies"`` or ``"trajectory"`` and selects the subfolder.
    """
    return sphere_root / "_gds_meta" / "topology_cache" / kind / pattern_id / f"v={version}.lance"


def read_cache(path: Path) -> pa.Table | None:
    """Return the cached table, or ``None`` when the file is absent / unreadable."""
    if not path.exists():
        return None
    try:
        return lance.dataset(str(path)).to_table()
    except (OSError, ValueError, RuntimeError):
        return None


def write_cache(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    """Overwrite the cache at ``path`` with ``rows`` (schema-validated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(rows, schema=schema)
    lance.write_dataset(tbl, str(path), mode="overwrite")
