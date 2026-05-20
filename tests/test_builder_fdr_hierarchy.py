# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
"""Integration tests for builder's fdr_hierarchy column validation +
temporal_bucket auto-materialisation hook."""
from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pytest
from hypertopos.model.sphere import (
    FDRHierarchyLevel,
    FDRTemporalLevel,
    Pattern,
)


def _make_pattern(
    *,
    fdr_hierarchy: list[FDRHierarchyLevel] | None = None,
    fdr_temporal_hierarchy: list[FDRTemporalLevel] | None = None,
) -> Pattern:
    return Pattern(
        pattern_id="p_x",
        entity_type="x",
        pattern_type="anchor",
        relations=[],
        mu=None,
        sigma_diag=None,
        theta=None,
        population_size=0,
        computed_at=None,
        version=1,
        status="production",
        fdr_hierarchy=fdr_hierarchy or [],
        fdr_temporal_hierarchy=fdr_temporal_hierarchy or [],
    )


class TestFDRHierarchyColumnValidation:
    """fdr_hierarchy.from_dimension MUST exist as a column on the anchor
    geometry before write — builder errors at build time if missing."""

    def test_missing_from_dimension_raises(self):
        from hypertopos.builder.builder import _validate_fdr_hierarchy_columns

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id_typo"),
            ],
        )
        existing_columns = ["primary_key", "bank_id"]
        with pytest.raises(ValueError, match="bank_id_typo"):
            _validate_fdr_hierarchy_columns(pat, existing_columns)

    def test_correct_from_dimension_passes(self):
        from hypertopos.builder.builder import _validate_fdr_hierarchy_columns

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
            ],
        )
        existing_columns = ["primary_key", "bank_id"]
        _validate_fdr_hierarchy_columns(pat, existing_columns)

    def test_empty_fdr_hierarchy_is_noop(self):
        from hypertopos.builder.builder import _validate_fdr_hierarchy_columns

        pat = _make_pattern()
        _validate_fdr_hierarchy_columns(pat, ["primary_key"])


class TestTemporalBucketMaterialisationHook:
    """When fdr_temporal_hierarchy.slice_dimension is not yet a column on the
    geometry, the builder auto-materialises it from event timestamps."""

    def test_missing_slice_dim_auto_materialised(self):
        from hypertopos.builder.builder import _maybe_materialise_temporal_buckets

        pat = _make_pattern(
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter",
                    slice_dimension="temporal_bucket",
                    bucket="90d",
                ),
            ],
        )
        geometry = pa.table({"primary_key": ["A1", "A2"]})
        events = pa.table({
            "primary_key": ["tx1", "tx2"],
            "from_account": ["A1", "A2"],
            "to_account": ["A2", "A1"],
            "timestamp": [
                datetime(2024, 1, 15, tzinfo=UTC),
                datetime(2024, 1, 16, tzinfo=UTC),
            ],
        })
        out = _maybe_materialise_temporal_buckets(
            pat,
            geometry_table=geometry,
            event_table=events,
            anchor_key_col_options=("from_account", "to_account"),
            timestamp_col="timestamp",
        )
        assert "temporal_bucket" in out.column_names
        bucket_vals = out["temporal_bucket"].to_pylist()
        assert all(v is not None for v in bucket_vals)

    def test_slice_dim_already_present_skipped(self):
        from hypertopos.builder.builder import _maybe_materialise_temporal_buckets

        pat = _make_pattern(
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter",
                    slice_dimension="temporal_bucket",
                    bucket="90d",
                ),
            ],
        )
        geometry = pa.table({
            "primary_key": ["A1", "A2"],
            "temporal_bucket": ["existing_value_1", "existing_value_2"],
        })
        events = pa.table({
            "primary_key": pa.array([], type=pa.string()),
            "from_account": pa.array([], type=pa.string()),
            "to_account": pa.array([], type=pa.string()),
            "timestamp": pa.array([], type=pa.timestamp("us", tz="UTC")),
        })
        out = _maybe_materialise_temporal_buckets(
            pat,
            geometry_table=geometry,
            event_table=events,
            anchor_key_col_options=("from_account", "to_account"),
            timestamp_col="timestamp",
        )
        assert out["temporal_bucket"].to_pylist() == [
            "existing_value_1", "existing_value_2",
        ]

    def test_no_fdr_temporal_hierarchy_is_noop(self):
        from hypertopos.builder.builder import _maybe_materialise_temporal_buckets

        pat = _make_pattern()
        geometry = pa.table({"primary_key": ["A1"]})
        out = _maybe_materialise_temporal_buckets(
            pat,
            geometry_table=geometry,
            event_table=None,
            anchor_key_col_options=(),
            timestamp_col="timestamp",
        )
        assert out.column_names == ["primary_key"]

    def test_missing_event_table_with_declared_level_raises(self):
        from hypertopos.builder.builder import _maybe_materialise_temporal_buckets

        pat = _make_pattern(
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter",
                    slice_dimension="temporal_bucket",
                    bucket="90d",
                ),
            ],
        )
        geometry = pa.table({"primary_key": ["A1"]})
        with pytest.raises(ValueError, match="event_table"):
            _maybe_materialise_temporal_buckets(
                pat,
                geometry_table=geometry,
                event_table=None,
                anchor_key_col_options=("from_account", "to_account"),
                timestamp_col="timestamp",
            )

    def test_custom_slice_dim_name_renamed(self):
        from hypertopos.builder.builder import _maybe_materialise_temporal_buckets

        pat = _make_pattern(
            fdr_temporal_hierarchy=[
                FDRTemporalLevel(
                    level="quarter",
                    slice_dimension="my_quarter",
                    bucket="90d",
                ),
            ],
        )
        geometry = pa.table({"primary_key": ["A1"]})
        events = pa.table({
            "primary_key": ["tx1"],
            "from_account": ["A1"],
            "to_account": ["A1"],
            "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
        })
        out = _maybe_materialise_temporal_buckets(
            pat,
            geometry_table=geometry,
            event_table=events,
            anchor_key_col_options=("from_account", "to_account"),
            timestamp_col="timestamp",
        )
        assert "my_quarter" in out.column_names
        assert "temporal_bucket" not in out.column_names
