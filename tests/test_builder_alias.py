# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import pyarrow as pa
import pytest


def _make_builder(tmp_path):
    from hypertopos.builder.builder import GDSBuilder

    builder = GDSBuilder("test_sphere", output_path=tmp_path / "gds")
    table = pa.table(
        {
            "pk": pa.array([f"E-{i}" for i in range(100)], type=pa.string()),
            "region": pa.array(["A"] * 50 + ["B"] * 50, type=pa.string()),
        }
    )
    regions = pa.table(
        {
            "primary_key": pa.array(["A", "B"], type=pa.string()),
        }
    )
    builder.add_line("entities", table, key_col="pk", source_id="test", role="anchor")
    builder.add_line("regions", regions, key_col="primary_key", source_id="test", role="anchor")
    from hypertopos.builder.builder import RelationSpec

    builder.add_pattern(
        "test_pattern",
        pattern_type="anchor",
        entity_line="entities",
        relations=[RelationSpec(line_id="regions", fk_col="region", direction="out")],
    )
    return builder


def test_add_alias_stores_config(tmp_path):
    builder = _make_builder(tmp_path)
    builder.add_alias(
        "high_region",
        base_pattern_id="test_pattern",
        cutting_plane_normal=[1.0],
        cutting_plane_bias=0.5,
    )
    assert "high_region" in builder._aliases


def test_add_alias_bad_pattern(tmp_path):
    builder = _make_builder(tmp_path)
    with pytest.raises(ValueError, match="nonexistent"):
        builder.add_alias(
            "bad",
            base_pattern_id="nonexistent",
            cutting_plane_normal=[1.0],
            cutting_plane_bias=0.0,
        )


def test_build_with_alias_writes_sphere_json(tmp_path):
    import json

    builder = _make_builder(tmp_path)
    builder.add_alias(
        "high_region",
        base_pattern_id="test_pattern",
        cutting_plane_normal=[1.0],
        cutting_plane_bias=0.0,
    )
    builder.build()
    sphere_json = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
    assert "high_region" in sphere_json["aliases"]
    alias = sphere_json["aliases"]["high_region"]
    assert alias["base_pattern_id"] == "test_pattern"
    assert alias["filter"]["cutting_plane"]["normal"] == [1.0]
    assert alias["filter"]["cutting_plane"]["bias"] == 0.0
    dp = alias["derived_pattern"]
    assert "mu" in dp
    assert "sigma_diag" in dp
    assert "theta" in dp
    assert dp["population_size"] > 0
    assert dp["population_size"] <= 100


def test_build_alias_dimension_sugar(tmp_path):
    """dimension=0, threshold sugar -> auto-generated unit normal on dim 0."""
    import json

    builder = _make_builder(tmp_path)
    # threshold must be <= 0 for this test data (all deltas ~ 0 due to
    # uniform binary FK values -> zero variance after z-scoring)
    builder.add_alias(
        "seg",
        base_pattern_id="test_pattern",
        cutting_plane_dimension=0,
        cutting_plane_threshold=-0.5,
    )
    builder.build()
    sphere_json = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
    cp = sphere_json["aliases"]["seg"]["filter"]["cutting_plane"]
    assert cp["normal"] == [1.0]  # unit vector on dim 0
    assert cp["bias"] == -0.5
