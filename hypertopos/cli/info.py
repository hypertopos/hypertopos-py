# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``hypertopos info`` — print sphere summary from a built sphere directory."""

from __future__ import annotations

import sys
from pathlib import Path

from hypertopos.storage.exceptions import GDSError
from hypertopos.storage.reader import GDSReader


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

    reader = GDSReader(str(p))
    try:
        sphere = reader.read_sphere()
    except GDSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Sphere: {sphere.sphere_id}")
    print(f"Path:   {p.resolve()}")
    print()

    # Lines
    print(f"Lines ({len(sphere.lines)}):")
    for lid, line in sphere.lines.items():
        rows = reader.count_points_rows(lid, line.current_version())
        cols = len(line.columns or [])
        print(f"  {lid:30s}  role={line.line_role:8s}  rows={rows}  cols={cols}")
    print()

    # Patterns
    print(f"Patterns ({len(sphere.patterns)}):")
    for pid, pattern in sphere.patterns.items():
        entity = sphere.entity_line(pid) or "-"
        n_dims = len(pattern.dim_labels)
        print(f"  {pid:30s}  type={pattern.pattern_type:8s}  entity={entity}  dims={n_dims}")
    print()

    # Disk size
    total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if total_bytes < 1024 * 1024:
        size_str = f"{total_bytes / 1024:.1f} KB"
    else:
        size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
    print(f"Total disk size: {size_str}")
