# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sphere.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_parse_alias_basic(tmp_path):
    from hypertopos.cli.schema import parse_config

    yaml_text = """\
    version: "0.1.0"
    sphere_id: test_sphere
    sources:
      src:
        path: data.csv
    lines:
      entities:
        source: src
        key: pk
        role: anchor
    patterns:
      my_pattern:
        type: anchor
        entity_line: entities
    aliases:
      high_value:
        base_pattern: my_pattern
        cutting_plane:
          dimension: 0
          threshold: 2.0
    """
    cfg = parse_config(_write_yaml(tmp_path, yaml_text))
    assert "high_value" in cfg.aliases
    alias = cfg.aliases["high_value"]
    assert alias.base_pattern == "my_pattern"
    assert alias.cutting_plane_dimension == 0
    assert alias.cutting_plane_threshold == 2.0


def test_parse_alias_explicit_normal(tmp_path):
    from hypertopos.cli.schema import parse_config

    yaml_text = """\
    version: "0.1.0"
    sphere_id: test_sphere
    sources:
      src:
        path: data.csv
    lines:
      entities:
        source: src
        key: pk
        role: anchor
    patterns:
      my_pattern:
        type: anchor
        entity_line: entities
    aliases:
      custom_segment:
        base_pattern: my_pattern
        cutting_plane:
          normal: [0.5, -0.3, 0.8]
          bias: 1.5
    """
    cfg = parse_config(_write_yaml(tmp_path, yaml_text))
    alias = cfg.aliases["custom_segment"]
    assert alias.cutting_plane_normal == [0.5, -0.3, 0.8]
    assert alias.cutting_plane_bias == 1.5


def test_parse_alias_missing_base_pattern(tmp_path):
    from hypertopos.cli.schema import parse_config

    yaml_text = """\
    version: "0.1.0"
    sphere_id: test_sphere
    sources:
      src:
        path: data.csv
    lines:
      entities:
        source: src
        key: pk
        role: anchor
    patterns:
      my_pattern:
        type: anchor
        entity_line: entities
    aliases:
      bad_alias:
        base_pattern: nonexistent
        cutting_plane:
          dimension: 0
          threshold: 2.0
    """
    with pytest.raises(ValueError, match="nonexistent"):
        parse_config(_write_yaml(tmp_path, yaml_text))


def test_parse_no_aliases(tmp_path):
    from hypertopos.cli.schema import parse_config

    yaml_text = """\
    version: "0.1.0"
    sphere_id: test_sphere
    sources:
      src:
        path: data.csv
    lines:
      entities:
        source: src
        key: pk
        role: anchor
    patterns:
      my_pattern:
        type: anchor
        entity_line: entities
    """
    cfg = parse_config(_write_yaml(tmp_path, yaml_text))
    assert cfg.aliases == {}


def test_build_yaml_with_aliases(tmp_path):
    """End-to-end: YAML with aliases -> build -> sphere.json has aliases."""
    import json

    # Create minimal CSV
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "entities.csv").write_text(
        "pk,region\n" + "\n".join(f"E-{i},{'A' if i < 50 else 'B'}" for i in range(100))
    )
    (data_dir / "regions.csv").write_text("primary_key\nA\nB\n")

    yaml_text = f"""\
    version: "0.1.0"
    sphere_id: test_alias
    sources:
      entities_src:
        path: {data_dir / "entities.csv"}
      regions_src:
        path: {data_dir / "regions.csv"}
    lines:
      entities:
        source: entities_src
        key: pk
        role: anchor
      regions:
        source: regions_src
        key: primary_key
        role: anchor
    patterns:
      entity_pattern:
        type: anchor
        entity_line: entities
        relations:
          - line: regions
            direction: out
            key_on_entity: region
    aliases:
      region_a:
        base_pattern: entity_pattern
        cutting_plane:
          dimension: 0
          threshold: -0.5
        description: "Entities with high region-A signal"
    """
    yaml_path = tmp_path / "sphere.yaml"
    yaml_path.write_text(textwrap.dedent(yaml_text), encoding="utf-8")

    from hypertopos.cli.build import build_from_config
    from hypertopos.cli.schema import parse_config

    cfg = parse_config(yaml_path)
    out = tmp_path / "gds_out"
    build_from_config(cfg, output_path=out, source_dir=tmp_path)

    sphere_json = json.loads((out / "_gds_meta" / "sphere.json").read_text())
    assert "region_a" in sphere_json["aliases"]
    assert sphere_json["aliases"]["region_a"]["derived_pattern"]["population_size"] > 0


def test_relation_edge_max_parsed(tmp_path):
    """edge_max in sphere.yaml relation must reach RelationConfig."""
    from hypertopos.cli.schema import parse_config

    yaml_text = """\
    version: "0.1.0"
    sphere_id: test_edge_max
    sources:
      src:
        path: data.csv
    lines:
      events:
        source: src
        key: pk
        role: event
      accounts:
        source: src
        key: pk
        role: anchor
    patterns:
      p:
        type: anchor
        entity_line: accounts
        relations:
          - line: events
            direction: in
            edge_max: 5
    """
    cfg = parse_config(_write_yaml(tmp_path, yaml_text))
    rel = cfg.patterns["p"].relations[0]
    assert rel.edge_max == 5


def test_relation_edge_max_default_none(tmp_path):
    """Relations without edge_max default to None."""
    from hypertopos.cli.schema import parse_config

    yaml_text = """\
    version: "0.1.0"
    sphere_id: test_no_edge_max
    sources:
      src:
        path: data.csv
    lines:
      events:
        source: src
        key: pk
        role: event
      accounts:
        source: src
        key: pk
        role: anchor
    patterns:
      p:
        type: anchor
        entity_line: accounts
        relations:
          - line: events
            direction: in
    """
    cfg = parse_config(_write_yaml(tmp_path, yaml_text))
    rel = cfg.patterns["p"].relations[0]
    assert rel.edge_max is None
