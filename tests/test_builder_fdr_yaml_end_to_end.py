# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for end-to-end YAML → builder wiring of fdr_hierarchy and
fdr_temporal_hierarchy: schema pass-through, CLI build conversion, and
the auto-discover event-pattern helper used by the builder."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest
from hypertopos.builder.builder import (
    EdgeTableConfig,
    GDSBuilder,
    RelationSpec,
    _auto_discover_event_pattern_for_anchor,
)
from hypertopos.cli.schema import PatternConfig, parse_config
from hypertopos.model.sphere import FDRHierarchyLevel, FDRTemporalLevel
from hypertopos.storage.reader import GDSReader


class TestYamlSchemaRoundTrip:
    """Minimal sphere.yaml with fdr_hierarchy + fdr_temporal_hierarchy
    parses into PatternConfig without TypeError and carries the raw lists."""

    def test_pattern_config_accepts_new_fields(self):
        # Direct PatternConfig construction with new kwargs — no TypeError.
        cfg = PatternConfig(
            type="anchor",
            entity_line="accounts",
            fdr_hierarchy=[{"level": "bank", "from_dimension": "bank_id"}],
            fdr_temporal_hierarchy=[
                {"level": "quarter", "slice_dimension": "temporal_bucket"},
            ],
        )
        assert cfg.fdr_hierarchy == [
            {"level": "bank", "from_dimension": "bank_id"},
        ]
        assert cfg.fdr_temporal_hierarchy == [
            {"level": "quarter", "slice_dimension": "temporal_bucket"},
        ]

    def test_yaml_parse_extracts_hierarchies(self, tmp_path: Path):
        # parse_config reads sphere.yaml and surfaces the raw lists.
        yaml_text = """
version: "0.1.0"
sphere_id: test_sphere
sources:
  src1:
    path: data.csv
lines:
  accounts:
    source: src1
    key: primary_key
    role: anchor
  transactions:
    source: src1
    key: primary_key
    role: event
patterns:
  account_pattern:
    type: anchor
    entity_line: accounts
    relations: []
    fdr_hierarchy:
      - level: bank
        from_dimension: bank_id
    fdr_temporal_hierarchy:
      - level: quarter
        slice_dimension: temporal_bucket
        bucket: 90d
"""
        path = tmp_path / "sphere.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = parse_config(path)
        pat = cfg.patterns["account_pattern"]
        assert pat.fdr_hierarchy == [
            {"level": "bank", "from_dimension": "bank_id"},
        ]
        assert pat.fdr_temporal_hierarchy == [
            {
                "level": "quarter",
                "slice_dimension": "temporal_bucket",
                "bucket": "90d",
            },
        ]

    def test_yaml_omitting_fdr_blocks_keeps_defaults(self, tmp_path: Path):
        yaml_text = """
version: "0.1.0"
sphere_id: test_sphere
sources:
  src1:
    path: data.csv
lines:
  accounts:
    source: src1
    key: primary_key
    role: anchor
patterns:
  account_pattern:
    type: anchor
    entity_line: accounts
    relations: []
"""
        path = tmp_path / "sphere.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = parse_config(path)
        pat = cfg.patterns["account_pattern"]
        assert pat.fdr_hierarchy is None
        assert pat.fdr_temporal_hierarchy is None


def _empty_table() -> pa.Table:
    return pa.table({"primary_key": pa.array([], type=pa.string())})


def _make_builder_with_patterns(tmp_path: Path) -> GDSBuilder:
    """Build a minimal builder with anchor + event patterns registered.

    Lines need only be present in self._lines so that add_pattern() doesn't
    reject them; their content does not matter for auto-discover.
    """
    builder = GDSBuilder(sphere_id="t", output_path=str(tmp_path))
    builder.add_line(
        "accounts", _empty_table(), key_col="primary_key", source_id="src1",
        role="anchor", entity_type="account",
    )
    builder.add_line(
        "transactions", _empty_table(), key_col="primary_key", source_id="src1",
        role="event", entity_type="tx",
    )
    return builder


class TestAutoDiscoverEventPattern:
    """_auto_discover_event_pattern_for_anchor heuristic: an event pattern
    'references' the anchor's line iff one of its relations.line_id matches
    the anchor's entity_line.
    """

    def test_happy_path_single_candidate(self, tmp_path: Path):
        b = _make_builder_with_patterns(tmp_path)
        b.add_pattern(
            "tx_pattern",
            pattern_type="event",
            entity_line="transactions",
            relations=[
                RelationSpec(
                    line_id="accounts", fk_col="from_account",
                    direction="in", required=True,
                ),
                RelationSpec(
                    line_id="accounts", fk_col="to_account",
                    direction="in", required=False,
                ),
            ],
            edge_table=EdgeTableConfig(
                from_col="from_account",
                to_col="to_account",
                timestamp_col="timestamp",
            ),
        )
        b.add_pattern(
            "account_pattern",
            pattern_type="anchor",
            entity_line="accounts",
            relations=[],
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter", slice_dimension="temporal_bucket",
                ),
            ],
        )

        anchor = b._patterns["account_pattern"]
        result = _auto_discover_event_pattern_for_anchor(anchor, b._patterns)
        assert result.pattern_id == "tx_pattern"

    def test_zero_candidates_raises(self, tmp_path: Path):
        b = _make_builder_with_patterns(tmp_path)
        b.add_pattern(
            "account_pattern",
            pattern_type="anchor",
            entity_line="accounts",
            relations=[],
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter", slice_dimension="temporal_bucket",
                ),
            ],
        )
        anchor = b._patterns["account_pattern"]
        with pytest.raises(ValueError, match="account_pattern"):
            _auto_discover_event_pattern_for_anchor(anchor, b._patterns)

    def test_multiple_candidates_raises(self, tmp_path: Path):
        b = _make_builder_with_patterns(tmp_path)
        b.add_line(
            "transactions_2", _empty_table(),
            key_col="primary_key", source_id="src1",
            role="event", entity_type="tx",
        )
        b.add_pattern(
            "tx_pattern_a",
            pattern_type="event",
            entity_line="transactions",
            relations=[
                RelationSpec(
                    line_id="accounts", fk_col="from_account",
                    direction="in", required=True,
                ),
            ],
            edge_table=EdgeTableConfig(
                from_col="from_account", to_col="to_account",
                timestamp_col="timestamp",
            ),
        )
        b.add_pattern(
            "tx_pattern_b",
            pattern_type="event",
            entity_line="transactions_2",
            relations=[
                RelationSpec(
                    line_id="accounts", fk_col="from_account",
                    direction="in", required=True,
                ),
            ],
            edge_table=EdgeTableConfig(
                from_col="from_account", to_col="to_account",
                timestamp_col="timestamp",
            ),
        )
        b.add_pattern(
            "account_pattern",
            pattern_type="anchor",
            entity_line="accounts",
            relations=[],
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter", slice_dimension="temporal_bucket",
                ),
            ],
        )
        anchor = b._patterns["account_pattern"]
        with pytest.raises(
            ValueError, match="Multiple event patterns|ambiguous",
        ):
            _auto_discover_event_pattern_for_anchor(anchor, b._patterns)

    def test_candidate_without_edge_table_raises(self, tmp_path: Path):
        # Event pattern matches by relations but has no edge_table — can't
        # provide from_col/to_col/timestamp_col. Distinct error from
        # zero-candidates.
        b = _make_builder_with_patterns(tmp_path)
        b.add_pattern(
            "tx_pattern",
            pattern_type="event",
            entity_line="transactions",
            relations=[
                RelationSpec(
                    line_id="accounts", fk_col="from_account",
                    direction="in", required=True,
                ),
            ],
            # No edge_table.
        )
        b.add_pattern(
            "account_pattern",
            pattern_type="anchor",
            entity_line="accounts",
            relations=[],
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter", slice_dimension="temporal_bucket",
                ),
            ],
        )
        anchor = b._patterns["account_pattern"]
        with pytest.raises(ValueError, match="edge_table|timestamp"):
            _auto_discover_event_pattern_for_anchor(anchor, b._patterns)

    def test_candidate_without_timestamp_col_raises(self, tmp_path: Path):
        b = _make_builder_with_patterns(tmp_path)
        b.add_pattern(
            "tx_pattern",
            pattern_type="event",
            entity_line="transactions",
            relations=[
                RelationSpec(
                    line_id="accounts", fk_col="from_account",
                    direction="in", required=True,
                ),
            ],
            edge_table=EdgeTableConfig(
                from_col="from_account", to_col="to_account",
                timestamp_col=None,
            ),
        )
        b.add_pattern(
            "account_pattern",
            pattern_type="anchor",
            entity_line="accounts",
            relations=[],
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter", slice_dimension="temporal_bucket",
                ),
            ],
        )
        anchor = b._patterns["account_pattern"]
        with pytest.raises(ValueError, match="timestamp"):
            _auto_discover_event_pattern_for_anchor(anchor, b._patterns)


class TestCliBuildPassThrough:
    """cli/build.py::_add_pattern converts raw YAML lists into model dataclasses
    and passes them to builder.add_pattern."""

    def test_add_pattern_converts_raw_lists_to_model_dataclasses(self, tmp_path: Path):
        # PatternConfig carries raw dicts (the YAML payload); _add_pattern
        # must convert them to FDRHierarchyLevel / FDRTemporalLevel before
        # invoking builder.add_pattern, so that the registered _PatternReg
        # holds well-typed instances that the geometry build can read.
        from hypertopos.cli.build import _add_pattern
        from hypertopos.cli.schema import SphereConfig

        b = _make_builder_with_patterns(tmp_path)
        pat_cfg = PatternConfig(
            type="anchor",
            entity_line="accounts",
            relations=[],
            fdr_hierarchy=[
                {"level": "bank", "from_dimension": "bank_id"},
            ],
            fdr_temporal_hierarchy=[
                {
                    "level": "quarter",
                    "slice_dimension": "temporal_bucket",
                    "bucket": "90d",
                },
            ],
        )
        sphere_cfg = SphereConfig(sphere_id="t")
        _add_pattern(b, "account_pattern", pat_cfg, sphere_cfg)

        reg = b._patterns["account_pattern"]
        assert reg.fdr_hierarchy == [
            FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
        ]
        assert reg.fdr_temporal_hierarchy == [
            FDRTemporalLevel(
                level="quarter",
                slice_dimension="temporal_bucket",
                bucket="90d",
            ),
        ]


class TestSphereJsonPersistence:
    """Regression test for the M1.1 persistence bug: builder kept
    Pattern.fdr_hierarchy / fdr_temporal_hierarchy in memory (so build-time
    column injection + validation worked), but the sphere.json writer
    dropped both fields, so on reload reader.py saw empty lists and
    find_anomalies with fdr_resolution rejected the call.
    """

    def _build_minimal_anchor_sphere(self, tmp_path: Path) -> Path:
        """Build a minimal anchor-pattern sphere with fdr_hierarchy declared,
        return the sphere directory path. Uses 5 customers, no event pattern
        (fdr_temporal_hierarchy is not exercised — it requires an edge_table
        with timestamp_col which would add unrelated setup).
        """
        sphere_path = tmp_path / "gds_test"
        b = GDSBuilder("test", str(sphere_path))
        b.add_line(
            "customers",
            [
                {"cust_id": "C-1", "bank_id": "bank_a"},
                {"cust_id": "C-2", "bank_id": "bank_a"},
                {"cust_id": "C-3", "bank_id": "bank_b"},
                {"cust_id": "C-4", "bank_id": "bank_b"},
                {"cust_id": "C-5", "bank_id": "bank_c"},
            ],
            key_col="cust_id",
            source_id="test",
            role="anchor",
        )
        b.add_pattern(
            "customer_pattern",
            pattern_type="anchor",
            entity_line="customers",
            relations=[
                RelationSpec("customers", fk_col=None, direction="self"),
            ],
            anomaly_percentile=60.0,
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
            ],
        )
        b.build()
        return sphere_path

    def test_sphere_json_contains_fdr_hierarchy(self, tmp_path: Path):
        sphere_path = self._build_minimal_anchor_sphere(tmp_path)
        sphere_json = json.loads(
            (sphere_path / "_gds_meta" / "sphere.json").read_text(),
        )
        pat = sphere_json["patterns"]["customer_pattern"]
        # Direct check of the on-disk JSON — catches a serializer drop
        # even if the reader silently defaults to [].
        assert "fdr_hierarchy" in pat
        assert pat["fdr_hierarchy"] == [
            {"level": "bank", "from_dimension": "bank_id"},
        ]

    def test_reader_round_trip_preserves_fdr_hierarchy(self, tmp_path: Path):
        sphere_path = self._build_minimal_anchor_sphere(tmp_path)
        # Load through the real reader, not the raw JSON
        reader = GDSReader(str(sphere_path))
        sphere = reader.read_sphere()
        pat = sphere.patterns["customer_pattern"]
        assert pat.fdr_hierarchy == [
            FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
        ]

    def test_no_fdr_hierarchy_omits_key_from_json(self, tmp_path: Path):
        """A pattern that does NOT declare fdr_hierarchy must not emit the
        key (avoid persisting empty lists that could confuse downstream
        diffing / tooling)."""
        sphere_path = tmp_path / "gds_test_nofdr"
        b = GDSBuilder("test", str(sphere_path))
        b.add_line(
            "customers",
            [{"cust_id": f"C-{i}"} for i in range(1, 6)],
            key_col="cust_id",
            source_id="test",
            role="anchor",
        )
        b.add_pattern(
            "customer_pattern",
            pattern_type="anchor",
            entity_line="customers",
            relations=[
                RelationSpec("customers", fk_col=None, direction="self"),
            ],
            anomaly_percentile=60.0,
        )
        b.build()
        sphere_json = json.loads(
            (sphere_path / "_gds_meta" / "sphere.json").read_text(),
        )
        pat = sphere_json["patterns"]["customer_pattern"]
        assert "fdr_hierarchy" not in pat
        assert "fdr_temporal_hierarchy" not in pat
