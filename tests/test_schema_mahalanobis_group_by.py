# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``use_mahalanobis`` and ``group_by_property`` cannot be combined in sphere.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypertopos.cli.schema import parse_config


def _yaml(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "sphere.yaml"
    path.write_text(
        'version: "0.1.0"\n'
        "sphere_id: conflict_test\n"
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


def test_use_mahalanobis_with_group_by_rejected(tmp_path: Path) -> None:
    cfg = _yaml(tmp_path, "    use_mahalanobis: true\n    group_by_property: segment\n")
    with pytest.raises(
        ValueError, match="use_mahalanobis cannot be combined with group_by_property"
    ):
        parse_config(cfg)


@pytest.mark.parametrize(
    "extra",
    ["    use_mahalanobis: true\n", "    group_by_property: segment\n"],
)
def test_either_option_alone_is_accepted(tmp_path: Path, extra: str) -> None:
    parse_config(_yaml(tmp_path, extra))
