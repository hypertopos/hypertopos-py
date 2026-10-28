# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Sphere format 4.0: version gate, ``use_mahalanobis`` removed, ``score_axes`` key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypertopos.builder.builder import GDSBuilder
from hypertopos.cli.schema import parse_config
from hypertopos.storage.exceptions import GDSVersionError
from hypertopos.storage.reader import GDSReader


def _yaml(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "sphere.yaml"
    path.write_text(
        'version: "0.1.0"\n'
        "sphere_id: format_test\n"
        "sources:\n"
        "  src1:\n"
        "    path: data.csv\n"
        "lines:\n"
        "  customers:\n"
        "    source: src1\n"
        "    key: primary_key\n"
        "    role: anchor\n"
        "patterns:\n"
        "  customer_pattern:\n"
        "    type: anchor\n"
        "    entity_line: customers\n"
        "    relations: []\n"
        f"{extra}",
        encoding="utf-8",
    )
    return path


def test_format_3_sphere_requires_rebuild(tmp_path: Path) -> None:
    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text(
        json.dumps({"sphere_id": "old", "format_version": "3.1", "lines": {}, "patterns": {}})
    )
    with pytest.raises(GDSVersionError, match="Rebuild the sphere"):
        GDSReader(str(tmp_path)).read_sphere()


def test_use_mahalanobis_key_rejected(tmp_path: Path) -> None:
    cfg = _yaml(tmp_path, "    use_mahalanobis: true\n")
    with pytest.raises(ValueError, match=r"use_mahalanobis.*part of the default score"):
        parse_config(cfg)


@pytest.mark.parametrize(
    ("axes", "message"),
    [("[knn]", "must contain 'mahal'"), ("[mahal, foo]", "foo")],
)
def test_score_axes_validation(tmp_path: Path, axes: str, message: str) -> None:
    cfg = _yaml(tmp_path, f"    score_axes: {axes}\n")
    with pytest.raises(ValueError, match=message):
        parse_config(cfg)


def test_score_axes_accepted_and_defaulted(tmp_path: Path) -> None:
    cfg = parse_config(_yaml(tmp_path, "    score_axes: [mahal, current, knn]\n"))
    assert cfg.patterns["customer_pattern"].score_axes == ["mahal", "current", "knn"]
    default = parse_config(_yaml(tmp_path, ""))
    assert default.patterns["customer_pattern"].score_axes == ["mahal", "current"]


def test_add_pattern_rejects_score_axes_without_mahal(tmp_path: Path) -> None:
    builder = GDSBuilder("s", str(tmp_path / "out"))
    with pytest.raises(ValueError, match="must contain 'mahal'"):
        builder.add_pattern("p", "anchor", "customers", relations=[], score_axes=["current"])
