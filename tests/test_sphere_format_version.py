# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Sphere format_version gate — major-only comparator (3.x accepted)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.navigation.navigator import GDSVersionError
from hypertopos.sphere import HyperSphere


def _build_minimal_sphere(tmp_path: Path) -> str:
    b = GDSBuilder("fmt_version_test", str(tmp_path / "gds_fmt"))
    b.add_line(
        "customers",
        [{"cust_id": "C-1"}, {"cust_id": "C-2"}],
        key_col="cust_id",
        source_id="test",
    )
    b.add_line(
        "orders",
        [
            {"order_id": "O-1", "cust_id": "C-1"},
            {"order_id": "O-2", "cust_id": "C-2"},
        ],
        key_col="order_id",
        source_id="test",
        role="event",
    )
    b.add_pattern(
        "order_pattern",
        pattern_type="event",
        entity_line="orders",
        relations=[
            RelationSpec("customers", fk_col="cust_id", direction="in", required=True),
        ],
    )
    return b.build()


def test_fresh_build_writes_format_3_0(tmp_path: Path) -> None:
    out = _build_minimal_sphere(tmp_path)
    meta = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
    assert meta["format_version"] == "3.0"


def test_open_rejects_format_2_4(tmp_path: Path) -> None:
    out = _build_minimal_sphere(tmp_path)
    sphere_path = Path(out) / "_gds_meta" / "sphere.json"
    meta = json.loads(sphere_path.read_text())
    meta["format_version"] = "2.4"
    sphere_path.write_text(json.dumps(meta, indent=2))

    with pytest.raises(GDSVersionError) as exc:
        HyperSphere.open(out)
    msg = str(exc.value)
    assert "major 3" in msg
    assert "rebuild" in msg.lower()


def test_open_rejects_format_2_3(tmp_path: Path) -> None:
    out = _build_minimal_sphere(tmp_path)
    sphere_path = Path(out) / "_gds_meta" / "sphere.json"
    meta = json.loads(sphere_path.read_text())
    meta["format_version"] = "2.3"
    sphere_path.write_text(json.dumps(meta, indent=2))

    with pytest.raises(GDSVersionError):
        HyperSphere.open(out)


def test_open_accepts_format_3_1(tmp_path: Path) -> None:
    """A 3.0 reader (this code) must accept a sphere stamped 3.1 —
    minor bump is backward-compatible by design."""
    out = _build_minimal_sphere(tmp_path)
    sphere_path = Path(out) / "_gds_meta" / "sphere.json"
    meta = json.loads(sphere_path.read_text())
    meta["format_version"] = "3.1"
    sphere_path.write_text(json.dumps(meta, indent=2))

    # Should not raise
    HyperSphere.open(out)


def test_open_rejects_malformed_format_version(tmp_path: Path) -> None:
    out = _build_minimal_sphere(tmp_path)
    sphere_path = Path(out) / "_gds_meta" / "sphere.json"
    meta = json.loads(sphere_path.read_text())
    meta["format_version"] = "not-a-version"
    sphere_path.write_text(json.dumps(meta, indent=2))

    with pytest.raises(GDSVersionError) as exc:
        HyperSphere.open(out)
    assert "malformed" in str(exc.value).lower()
