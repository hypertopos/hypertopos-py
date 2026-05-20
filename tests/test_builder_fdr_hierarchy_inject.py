# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
"""Unit tests for builder's fdr_hierarchy.from_dimension column injection +
the inject-before-validate ordering at the three _build_and_write call sites.
"""
from __future__ import annotations

from pathlib import Path

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
    entity_line: str = "accounts",
) -> Pattern:
    pat = Pattern(
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
    pat.entity_line_id = entity_line
    return pat


class TestFDRHierarchyColumnInjection:
    """`_inject_fdr_hierarchy_columns` carries fdr_hierarchy.from_dimension
    columns from the anchor line onto the geometry table via primary_key join,
    skipping columns already present and erroring when the anchor lacks the
    column entirely."""

    def test_inject_missing_column_from_anchor(self):
        from hypertopos.builder.builder import _inject_fdr_hierarchy_columns

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
            ],
        )
        geometry = pa.table({"primary_key": ["A1", "A2", "A3"]})
        anchor = pa.table({
            "primary_key": ["A1", "A2", "A3", "A4"],
            "bank_id": ["B01", "B02", "B01", "B99"],
            "other_col": ["x", "y", "z", "w"],
        })
        out = _inject_fdr_hierarchy_columns(
            pat, geometry_table=geometry, anchor_table=anchor,
        )
        assert "bank_id" in out.column_names
        # Order-preserving join keyed on primary_key
        rows = dict(zip(
            out["primary_key"].to_pylist(),
            out["bank_id"].to_pylist(),
            strict=True,
        ))
        assert rows == {"A1": "B01", "A2": "B02", "A3": "B01"}
        # Only requested column was injected, not "other_col"
        assert "other_col" not in out.column_names

    def test_inject_skipped_when_already_present(self):
        from hypertopos.builder.builder import _inject_fdr_hierarchy_columns

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
            ],
        )
        geometry = pa.table({
            "primary_key": ["A1", "A2"],
            "bank_id": ["preset_1", "preset_2"],
        })
        anchor = pa.table({
            "primary_key": ["A1", "A2"],
            "bank_id": ["DIFFERENT_1", "DIFFERENT_2"],
        })
        out = _inject_fdr_hierarchy_columns(
            pat, geometry_table=geometry, anchor_table=anchor,
        )
        # Helper is a no-op when column already on geometry — preset values win
        assert out["bank_id"].to_pylist() == ["preset_1", "preset_2"]

    def test_inject_raises_when_anchor_lacks_column(self):
        from hypertopos.builder.builder import _inject_fdr_hierarchy_columns

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
            ],
            entity_line="accounts",
        )
        geometry = pa.table({"primary_key": ["A1"]})
        anchor = pa.table({
            "primary_key": ["A1"],
            "country_code": ["PL"],
        })
        with pytest.raises(ValueError) as exc_info:
            _inject_fdr_hierarchy_columns(
                pat, geometry_table=geometry, anchor_table=anchor,
            )
        msg = str(exc_info.value)
        assert "bank_id" in msg
        assert "accounts" in msg

    def test_inject_noop_when_no_fdr_hierarchy(self):
        from hypertopos.builder.builder import _inject_fdr_hierarchy_columns

        pat = _make_pattern()
        geometry = pa.table({"primary_key": ["A1"]})
        anchor = pa.table({"primary_key": ["A1"], "bank_id": ["B01"]})
        out = _inject_fdr_hierarchy_columns(
            pat, geometry_table=geometry, anchor_table=anchor,
        )
        assert out.column_names == ["primary_key"]

    def test_inject_multiple_levels(self):
        from hypertopos.builder.builder import _inject_fdr_hierarchy_columns

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
                FDRHierarchyLevel(
                    level="community", from_dimension="community_id",
                ),
            ],
        )
        geometry = pa.table({"primary_key": ["A1", "A2"]})
        anchor = pa.table({
            "primary_key": ["A1", "A2"],
            "bank_id": ["B01", "B02"],
            "community_id": ["C7", "C8"],
        })
        out = _inject_fdr_hierarchy_columns(
            pat, geometry_table=geometry, anchor_table=anchor,
        )
        assert "bank_id" in out.column_names
        assert "community_id" in out.column_names


class TestInjectBeforeValidateOrder:
    """The three geometry-write paths in `_build_and_write` must execute the
    sequence `inject_hierarchy → inject_temporal → validate`. If validate runs
    first, fdr_hierarchy.from_dimension columns sourced from the anchor line
    will not yet be on the geometry table and validation will erroneously
    raise.
    """

    @staticmethod
    def _call_site_line_numbers(builder_src: str) -> list[tuple[int, int, int]]:
        """Locate the three call sites by their textual markers and return
        (inject_hierarchy_line, inject_temporal_line, validate_line) per site.

        A call site is `self._inject_fdr_hierarchy_carriers(...)` (the method
        wrapper used at the three _build_and_write paths) followed within ~10
        lines by `self._inject_fdr_temporal_buckets(...)` and then
        `_validate_fdr_hierarchy_columns(...)`.
        """
        lines = builder_src.splitlines()
        sites: list[tuple[int, int, int]] = []
        i = 0
        while i < len(lines):
            if "self._inject_fdr_hierarchy_carriers" in lines[i]:
                hier_line = i + 1
                temp_line = None
                val_line = None
                for j in range(i + 1, min(i + 15, len(lines))):
                    if (
                        temp_line is None
                        and "self._inject_fdr_temporal_buckets" in lines[j]
                    ):
                        temp_line = j + 1
                    elif (
                        val_line is None
                        and "_validate_fdr_hierarchy_columns(" in lines[j]
                    ):
                        val_line = j + 1
                        break
                if temp_line is not None and val_line is not None:
                    sites.append((hier_line, temp_line, val_line))
                i = (val_line or i) + 1
            else:
                i += 1
        return sites

    def test_three_call_sites_inject_hierarchy_before_temporal_before_validate(
        self,
    ):
        builder_path = (
            Path(__file__).parent.parent
            / "hypertopos" / "builder" / "builder.py"
        )
        src = builder_path.read_text(encoding="utf-8")
        sites = self._call_site_line_numbers(src)
        assert len(sites) == 3, (
            f"Expected three (hierarchy, temporal, validate) call sites in "
            f"_build_and_write paths, found {len(sites)}: {sites}"
        )
        for hier_line, temp_line, val_line in sites:
            assert hier_line < temp_line < val_line, (
                f"Call-site ordering broken at site "
                f"(hier={hier_line}, temp={temp_line}, val={val_line}): "
                f"required inject_hierarchy → inject_temporal → validate"
            )


class TestInjectThenValidateEndToEnd:
    """When inject runs before validate, a pattern with fdr_hierarchy that
    only lives on the anchor line passes the validation gate.
    """

    def test_inject_then_validate_passes_for_anchor_sourced_column(self):
        from hypertopos.builder.builder import (
            _inject_fdr_hierarchy_columns,
            _validate_fdr_hierarchy_columns,
        )

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
            ],
        )
        geometry = pa.table({
            "primary_key": ["A1", "A2"],
            "delta_norm": [0.1, 0.2],
        })
        anchor = pa.table({
            "primary_key": ["A1", "A2"],
            "bank_id": ["B01", "B02"],
        })
        # 1. Inject hierarchy from anchor
        out = _inject_fdr_hierarchy_columns(
            pat, geometry_table=geometry, anchor_table=anchor,
        )
        # 2. Validate against the post-inject column set
        _validate_fdr_hierarchy_columns(pat, list(out.column_names))
        # Both should pass without raising
        assert "bank_id" in out.column_names

    def test_validate_before_inject_would_have_raised(self):
        """Sanity: confirms the bug exists if order is reversed."""
        from hypertopos.builder.builder import (
            _validate_fdr_hierarchy_columns,
        )

        pat = _make_pattern(
            fdr_hierarchy=[
                FDRHierarchyLevel(level="bank", from_dimension="bank_id"),
            ],
        )
        # Pre-inject column set — bank_id NOT present
        geometry_cols = ["primary_key", "delta_norm"]
        with pytest.raises(ValueError, match="bank_id"):
            _validate_fdr_hierarchy_columns(pat, geometry_cols)
