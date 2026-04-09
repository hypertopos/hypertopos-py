# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``hypertopos info`` — print sphere summary from a built sphere directory."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def run_info(sphere_path: str) -> None:
    """Print a human-readable summary of a built sphere."""
    p = Path(sphere_path)
    meta_path = p / "_gds_meta" / "sphere.json"
    if not meta_path.exists():
        print(
            f"error: not a sphere directory (no _gds_meta/sphere.json): {p}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(meta_path, encoding="utf-8") as f:
        sphere = json.load(f)

    sphere_id = sphere.get("sphere_id", "?")
    print(f"Sphere: {sphere_id}")
    print(f"Path:   {p.resolve()}")
    print()

    # Lines
    lines = sphere.get("lines", {})
    print(f"Lines ({len(lines)}):")
    for lid, linfo in lines.items():
        role = linfo.get("role", "?")
        rows = linfo.get("total_rows", "?")
        cols = len(linfo.get("columns", []))
        print(f"  {lid:30s}  role={role:8s}  rows={rows}  cols={cols}")
    print()

    # Patterns
    patterns = sphere.get("patterns", {})
    print(f"Patterns ({len(patterns)}):")
    for pid, pinfo in patterns.items():
        ptype = pinfo.get("type", "?")
        entity = pinfo.get("entity_line", "?")
        n_dims = len(pinfo.get("relations", []))
        event_dims = pinfo.get("event_dimensions", [])
        if event_dims:
            n_dims += len(event_dims)
        print(f"  {pid:30s}  type={ptype:8s}  entity={entity}  dims={n_dims}")
    print()

    # Disk size
    total_bytes = sum(
        f.stat().st_size for f in p.rglob("*") if f.is_file()
    )
    if total_bytes < 1024 * 1024:
        size_str = f"{total_bytes / 1024:.1f} KB"
    else:
        size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
    print(f"Total disk size: {size_str}")
