# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
"""Tests for Pattern.fdr_hierarchy + fdr_temporal_hierarchy fields."""
from __future__ import annotations

import yaml
from hypertopos.model.sphere import Pattern
from hypertopos.storage.reader import GDSReader


def _pattern_from_yaml_block(yaml_text: str) -> Pattern:
    """Parse one pattern entry through the canonical sphere.json loader.

    The test YAML uses the sphere.json shape (`pattern_id`, `entity_type`,
    `pattern_type`, `relations`) which routes through `GDSReader._parse_pattern`
    at runtime. The helper merges minimal defaults for the required statistical
    fields so tests can focus on the new FDR hierarchy parsing.
    """
    raw = yaml.safe_load(yaml_text)
    raw.setdefault("mu", [0.0])
    raw.setdefault("sigma_diag", [1.0])
    raw.setdefault("theta", [3.0])
    raw.setdefault("population_size", 1)
    raw.setdefault("computed_at", "2024-01-01T00:00:00")
    raw.setdefault("version", 1)
    raw.setdefault("status", "production")
    # Bypass GDSReader.__init__ — we only need _parse_pattern.
    reader = GDSReader.__new__(GDSReader)
    return reader._parse_pattern(raw)


class TestFDRHierarchyParsing:
    def test_pattern_without_fdr_blocks_keeps_defaults(self):
        yaml_text = """
        pattern_id: p_x
        entity_type: x
        pattern_type: anchor
        relations: []
        """
        p = _pattern_from_yaml_block(yaml_text)
        assert p.fdr_hierarchy == []
        assert p.fdr_temporal_hierarchy == []

    def test_pattern_with_spatial_hierarchy(self):
        yaml_text = """
        pattern_id: p_x
        entity_type: x
        pattern_type: anchor
        relations: []
        fdr_hierarchy:
          - level: bank
            from_dimension: bank_id
          - level: community
            from_dimension: community_id
        """
        p = _pattern_from_yaml_block(yaml_text)
        assert len(p.fdr_hierarchy) == 2
        assert p.fdr_hierarchy[0].level == "bank"
        assert p.fdr_hierarchy[0].from_dimension == "bank_id"
        assert p.fdr_hierarchy[1].level == "community"
        assert p.fdr_hierarchy[1].from_dimension == "community_id"

    def test_pattern_with_temporal_hierarchy(self):
        yaml_text = """
        pattern_id: p_x
        entity_type: x
        pattern_type: anchor
        relations: []
        fdr_temporal_hierarchy:
          - level: quarter
            slice_dimension: temporal_bucket
            bucket: 90d
        """
        p = _pattern_from_yaml_block(yaml_text)
        assert len(p.fdr_temporal_hierarchy) == 1
        assert p.fdr_temporal_hierarchy[0].level == "quarter"
        assert p.fdr_temporal_hierarchy[0].slice_dimension == "temporal_bucket"
        assert p.fdr_temporal_hierarchy[0].bucket == "90d"

    def test_pattern_with_temporal_hierarchy_default_bucket(self):
        yaml_text = """
        pattern_id: p_x
        entity_type: x
        pattern_type: anchor
        relations: []
        fdr_temporal_hierarchy:
          - level: quarter
            slice_dimension: temporal_bucket
        """
        p = _pattern_from_yaml_block(yaml_text)
        assert p.fdr_temporal_hierarchy[0].bucket == "90d"  # default
