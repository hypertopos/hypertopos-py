# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for edge table storage layer (Lance-based)."""
from __future__ import annotations

import pyarrow as pa
import pytest

from hypertopos.storage._schemas import EDGE_TABLE_SCHEMA
from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter


def _make_edges(n: int = 100) -> pa.Table:
    """Build a synthetic edge table with n rows."""
    return pa.table(
        {
            "from_key": [f"A{i % 10:03d}" for i in range(n)],
            "to_key": [f"B{(i + 1) % 10:03d}" for i in range(n)],
            "event_key": [f"E{i:05d}" for i in range(n)],
            "timestamp": [1_000_000.0 + i * 3600.0 for i in range(n)],
            "amount": [float(i * 10) for i in range(n)],
        },
        schema=EDGE_TABLE_SCHEMA,
    )


class TestWriteReadRoundtrip:
    def test_roundtrip(self, tmp_path):
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        edges = _make_edges(50)

        writer.write_edges("tx_pattern", edges)
        result = reader.read_edges("tx_pattern")

        assert result.num_rows == 50
        assert set(result.schema.names) == set(EDGE_TABLE_SCHEMA.names)

    def test_schema_matches(self, tmp_path):
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        writer.write_edges("tx_pattern", _make_edges(10))

        result = reader.read_edges("tx_pattern")
        for field in EDGE_TABLE_SCHEMA:
            assert field.name in result.schema.names

    def test_empty_table(self, tmp_path):
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in EDGE_TABLE_SCHEMA},
        )
        writer.write_edges("tx_pattern", empty)

        result = reader.read_edges("tx_pattern")
        assert result.num_rows == 0


class TestReadEdgesFilters:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.writer = GDSWriter(str(tmp_path))
        self.reader = GDSReader(str(tmp_path))
        self.writer.write_edges("tx_pattern", _make_edges(100))

    def test_filter_from_keys(self):
        result = self.reader.read_edges("tx_pattern", from_keys=["A000"])
        assert result.num_rows > 0
        assert all(k == "A000" for k in result["from_key"].to_pylist())

    def test_filter_to_keys(self):
        result = self.reader.read_edges("tx_pattern", to_keys=["B001"])
        assert result.num_rows > 0
        assert all(k == "B001" for k in result["to_key"].to_pylist())

    def test_filter_timestamp_range(self):
        ts_from = 1_000_000.0 + 10 * 3600.0
        ts_to = 1_000_000.0 + 20 * 3600.0
        result = self.reader.read_edges(
            "tx_pattern", timestamp_from=ts_from, timestamp_to=ts_to,
        )
        timestamps = result["timestamp"].to_pylist()
        assert all(ts_from <= t <= ts_to for t in timestamps)

    def test_combined_filters(self):
        result = self.reader.read_edges(
            "tx_pattern",
            from_keys=["A000"],
            timestamp_from=1_000_000.0,
            timestamp_to=1_000_000.0 + 50 * 3600.0,
        )
        assert result.num_rows > 0
        assert all(k == "A000" for k in result["from_key"].to_pylist())

    def test_column_projection(self):
        result = self.reader.read_edges(
            "tx_pattern", columns=["from_key", "to_key"],
        )
        assert set(result.schema.names) == {"from_key", "to_key"}
        assert result.num_rows == 100


class TestNonexistent:
    def test_nonexistent_pattern_returns_empty(self, tmp_path):
        reader = GDSReader(str(tmp_path))
        result = reader.read_edges("no_such_pattern")
        assert result.num_rows == 0
        assert set(result.schema.names) == set(EDGE_TABLE_SCHEMA.names)


class TestHasEdgeTable:
    def test_exists(self, tmp_path):
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        writer.write_edges("tx_pattern", _make_edges(10))
        assert reader.has_edge_table("tx_pattern") is True

    def test_not_exists(self, tmp_path):
        reader = GDSReader(str(tmp_path))
        assert reader.has_edge_table("no_pattern") is False


class TestEdgeTableStats:
    def test_stats(self, tmp_path):
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        writer.write_edges("tx_pattern", _make_edges(100))

        stats = reader.edge_table_stats("tx_pattern")
        assert stats is not None
        assert stats["row_count"] == 100
        assert stats["unique_from"] == 10  # A000-A009
        assert stats["unique_to"] == 10    # B000-B009
        assert stats["avg_out_degree"] == 10.0

    def test_stats_nonexistent(self, tmp_path):
        reader = GDSReader(str(tmp_path))
        assert reader.edge_table_stats("no_pattern") is None


class TestAppendEdges:
    def test_append_and_read(self, tmp_path):
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))

        writer.write_edges("tx_pattern", _make_edges(50))
        assert reader.read_edges("tx_pattern").num_rows == 50

        extra = pa.table(
            {
                "from_key": ["X001", "X002"],
                "to_key": ["Y001", "Y002"],
                "event_key": ["EX01", "EX02"],
                "timestamp": [2_000_000.0, 2_000_001.0],
                "amount": [999.0, 888.0],
            },
            schema=EDGE_TABLE_SCHEMA,
        )
        writer.append_edges("tx_pattern", extra)
        writer.create_edge_indexes("tx_pattern")

        result = reader.read_edges("tx_pattern")
        assert result.num_rows == 52

    def test_append_indexed_lookup(self, tmp_path):
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))

        writer.write_edges("tx_pattern", _make_edges(50))
        extra = pa.table(
            {
                "from_key": ["XNEW"],
                "to_key": ["YNEW"],
                "event_key": ["EXNEW"],
                "timestamp": [9_999_999.0],
                "amount": [42.0],
            },
            schema=EDGE_TABLE_SCHEMA,
        )
        writer.append_edges("tx_pattern", extra)
        writer.create_edge_indexes("tx_pattern")

        result = reader.read_edges("tx_pattern", from_keys=["XNEW"])
        assert result.num_rows == 1
        assert result["to_key"][0].as_py() == "YNEW"
