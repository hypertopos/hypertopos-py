# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
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


# ── Shared test helpers for building geometry tables with edges struct ──

EDGE_STRUCT_TYPE = pa.struct(
    [
        pa.field("line_id", pa.string()),
        pa.field("point_key", pa.string()),
        pa.field("status", pa.string()),
        pa.field("direction", pa.string()),
    ]
)


def make_edges_column(
    edge_line_ids: list[list[str]],
    edge_point_keys: list[list[str]],
    edge_alive_mask: list[list[bool]] | None = None,
) -> pa.Array:
    """Build an ``edges`` struct column from flat per-row line_ids/point_keys.

    When *edge_alive_mask* is provided, edges where the mask is False get
    status="dead"; otherwise all edges are "alive".

    This is a test-only helper that bridges the old flat-column test format
    to the canonical ``edges`` struct schema.
    """
    rows: list[list[dict]] = []
    for i in range(len(edge_line_ids)):
        lids = edge_line_ids[i]
        pks = edge_point_keys[i]
        alive = edge_alive_mask[i] if edge_alive_mask else [True] * len(lids)
        row = [
            {
                "line_id": lid,
                "point_key": pk,
                "status": "alive" if a else "dead",
                "direction": "in",
            }
            for lid, pk, a in zip(lids, pks, alive, strict=False)
        ]
        rows.append(row)
    return pa.array(rows, type=pa.list_(EDGE_STRUCT_TYPE))
