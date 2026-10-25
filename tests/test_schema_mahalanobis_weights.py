# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``use_mahalanobis`` requires strictly positive explicit ``dimension_weights``."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypertopos.builder.builder import GDSBuilder
from hypertopos.cli.schema import parse_config


def _yaml(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "sphere.yaml"
    path.write_text(
        'version: "0.1.0"\n'
        "sphere_id: weights_test\n"
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


@pytest.mark.parametrize("bad", ["[1.0, 0.0]", "[1.0, -2.0]"])
def test_yaml_mahalanobis_with_non_positive_weight_rejected(tmp_path: Path, bad: str) -> None:
    cfg = _yaml(tmp_path, f"    use_mahalanobis: true\n    dimension_weights: {bad}\n")
    with pytest.raises(ValueError, match="customer_pattern.*dimension_weights"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "extra",
    [
        "    use_mahalanobis: true\n    dimension_weights: [1.0, 2.0]\n",
        "    use_mahalanobis: true\n    dimension_weights: kurtosis\n",
        "    dimension_weights: [1.0, 0.0]\n",
    ],
)
def test_yaml_accepted_combinations(tmp_path: Path, extra: str) -> None:
    parse_config(_yaml(tmp_path, extra))


def test_add_pattern_mahalanobis_with_zero_weight_rejected(tmp_path: Path) -> None:
    builder = GDSBuilder("weights_test", str(tmp_path / "out"))
    with pytest.raises(ValueError, match="p1.*dimension_weights"):
        builder.add_pattern(
            "p1",
            pattern_type="anchor",
            entity_line="items",
            relations=[],
            use_mahalanobis=True,
            dimension_weights=[1.0, 0.0],
        )


def test_add_pattern_mahalanobis_with_positive_weights_accepted(tmp_path: Path) -> None:
    builder = GDSBuilder("weights_test", str(tmp_path / "out"))
    builder.add_pattern(
        "p1",
        pattern_type="anchor",
        entity_line="items",
        relations=[],
        use_mahalanobis=True,
        dimension_weights=[1.0, 2.0],
    )
