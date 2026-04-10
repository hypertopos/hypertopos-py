# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for builder edge table emission."""
from __future__ import annotations

import json

import pyarrow as pa
import pytest

from hypertopos.builder.builder import (
    EdgeTableConfig,
    GDSBuilder,
    RelationSpec,
)
from hypertopos.storage._schemas import EDGE_TABLE_SCHEMA
from hypertopos.storage.reader import GDSReader


def _build_tx_sphere(tmp_path, *, edge_table_cfg=None):
    """Build a minimal sphere with accounts + transactions (from/to structure)."""
    builder = GDSBuilder("test_sphere", str(tmp_path / "gds"))

    accounts = pa.table({
        "primary_key": [f"A{i:03d}" for i in range(20)],
        "name": [f"Account {i}" for i in range(20)],
    })
    builder.add_line("accounts", accounts, key_col="primary_key",
                     source_id="accounts", role="anchor")

    txns = pa.table({
        "primary_key": [f"TX{i:05d}" for i in range(200)],
        "sender_id": [f"A{i % 20:03d}" for i in range(200)],
        "receiver_id": [f"A{(i + 3) % 20:03d}" for i in range(200)],
        "amount": [float(i * 10) for i in range(200)],
        "tx_date": pa.array(
            [1_700_000_000.0 + i * 3600.0 for i in range(200)],
            type=pa.float64(),
        ),
    })
    builder.add_line("transactions", txns, key_col="primary_key",
                     source_id="transactions", role="event")

    builder.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="transactions",
        relations=[
            RelationSpec(line_id="accounts", fk_col="sender_id", direction="out"),
            RelationSpec(line_id="accounts", fk_col="receiver_id", direction="in"),
        ],
        edge_table=edge_table_cfg,
    )

    path = builder.build()
    return path


class TestEdgeEmission:
    def test_auto_detect_from_relations(self, tmp_path):
        """Two FK relations to same anchor → auto-detect from/to → emit edge table."""
        path = _build_tx_sphere(tmp_path)
        reader = GDSReader(path)
        assert reader.has_edge_table("tx_pattern")

    def test_edge_table_schema(self, tmp_path):
        path = _build_tx_sphere(tmp_path)
        reader = GDSReader(path)
        edges = reader.read_edges("tx_pattern")
        assert edges.num_rows == 200
        for field in EDGE_TABLE_SCHEMA:
            assert field.name in edges.schema.names

    def test_edge_table_from_to_values(self, tmp_path):
        path = _build_tx_sphere(tmp_path)
        reader = GDSReader(path)
        edges = reader.read_edges("tx_pattern")
        from_keys = set(edges["from_key"].to_pylist())
        to_keys = set(edges["to_key"].to_pylist())
        # All from/to should be account keys
        assert all(k.startswith("A") for k in from_keys)
        assert all(k.startswith("A") for k in to_keys)

    def test_edge_table_row_count_matches_events(self, tmp_path):
        path = _build_tx_sphere(tmp_path)
        reader = GDSReader(path)
        edges = reader.read_edges("tx_pattern")
        assert edges.num_rows == 200  # 1:1 with event line

    def test_explicit_edge_table_config(self, tmp_path):
        cfg = EdgeTableConfig(
            from_col="sender_id",
            to_col="receiver_id",
            amount_col="amount",
        )
        path = _build_tx_sphere(tmp_path, edge_table_cfg=cfg)
        reader = GDSReader(path)
        edges = reader.read_edges("tx_pattern")
        assert edges.num_rows == 200
        # Amount should be populated
        amounts = edges["amount"].to_pylist()
        assert any(a > 0 for a in amounts)

    def test_btree_indexed_lookup(self, tmp_path):
        path = _build_tx_sphere(tmp_path)
        reader = GDSReader(path)
        edges = reader.read_edges("tx_pattern", from_keys=["A000"])
        assert edges.num_rows > 0
        assert all(k == "A000" for k in edges["from_key"].to_pylist())


class TestSphereJsonMetadata:
    def test_has_edge_table_in_sphere_json(self, tmp_path):
        path = _build_tx_sphere(tmp_path)
        sphere_json = json.loads(
            (tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text()
        )
        pat = sphere_json["patterns"]["tx_pattern"]
        assert pat.get("has_edge_table") is True
        assert "edge_table" in pat
        assert pat["edge_table"]["from_col"] == "sender_id"
        assert pat["edge_table"]["to_col"] == "receiver_id"


class TestAnchorPatternNoEdges:
    def test_anchor_pattern_skips_edge_table(self, tmp_path):
        """Anchor patterns without from/to structure don't emit edge table."""
        builder = GDSBuilder("test_sphere", str(tmp_path / "gds"))
        customers = pa.table({
            "primary_key": [f"C{i:03d}" for i in range(50)],
            "region": ["east", "west"] * 25,
        })
        builder.add_line("customers", customers, key_col="primary_key",
                         source_id="customers", role="anchor")
        builder.add_pattern(
            "cust_pattern",
            pattern_type="anchor",
            entity_line="customers",
            relations=[
                RelationSpec(line_id="customers", fk_col=None, direction="self"),
            ],
        )
        path = builder.build()
        reader = GDSReader(path)
        assert reader.has_edge_table("cust_pattern") is False
