# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
"""Tests for the temporal_bucket materialiser."""
from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pytest
from hypertopos.builder.temporal_bucket import (
    materialise_temporal_bucket,
    parse_bucket_duration,
)


class TestParseBucketDuration:
    def test_days(self):
        assert parse_bucket_duration("90d") == 90 * 86400

    def test_hours(self):
        assert parse_bucket_duration("24h") == 24 * 3600

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="bucket"):
            parse_bucket_duration("90x")

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="bucket"):
            parse_bucket_duration("not-a-duration")


class TestMaterialiseTemporalBucket:
    def test_single_entity_one_transaction(self):
        ts1 = datetime(2024, 1, 15, tzinfo=UTC)
        events = pa.table({
            "primary_key": ["tx1"],
            "from_account": ["A1"],
            "to_account": ["A2"],
            "timestamp": [ts1],
        })
        result = materialise_temporal_bucket(
            event_table=events,
            anchor_keys=["A1", "A2"],
            anchor_key_col_options=("from_account", "to_account"),
            timestamp_col="timestamp",
            bucket="90d",
        )
        assert set(result.column_names) == {"primary_key", "temporal_bucket"}
        ddict = dict(zip(result["primary_key"].to_pylist(),
                          result["temporal_bucket"].to_pylist(), strict=True))
        assert ddict["A1"] == ddict["A2"]
        assert isinstance(ddict["A1"], str)

    def test_entity_with_multiple_tx_uses_centroid(self):
        events = pa.table({
            "primary_key": ["tx1", "tx2", "tx3"],
            "from_account": ["A1", "A1", "A1"],
            "to_account": ["A2", "A2", "A2"],
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
                datetime(2024, 3, 1, tzinfo=UTC),
            ],
        })
        result = materialise_temporal_bucket(
            event_table=events,
            anchor_keys=["A1"],
            anchor_key_col_options=("from_account",),
            timestamp_col="timestamp",
            bucket="90d",
        )
        ddict = dict(zip(result["primary_key"].to_pylist(),
                          result["temporal_bucket"].to_pylist(), strict=True))
        assert "A1" in ddict
        assert ddict["A1"] is not None

    def test_entity_no_transactions_returns_null(self):
        events = pa.table({
            "primary_key": pa.array([], type=pa.string()),
            "from_account": pa.array([], type=pa.string()),
            "to_account": pa.array([], type=pa.string()),
            "timestamp": pa.array([], type=pa.timestamp("us", tz="UTC")),
        })
        result = materialise_temporal_bucket(
            event_table=events,
            anchor_keys=["A1"],
            anchor_key_col_options=("from_account",),
            timestamp_col="timestamp",
            bucket="90d",
        )
        ddict = dict(zip(result["primary_key"].to_pylist(),
                          result["temporal_bucket"].to_pylist(), strict=True))
        assert ddict.get("A1") is None
