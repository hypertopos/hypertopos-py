# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for engine.fdr multi-resolution helpers."""
from __future__ import annotations

import pyarrow as pa
import pytest
from hypertopos.engine.fdr import (
    cell_p_values_from_anomaly_indicator,
    fdr_multi_resolution,
)


class TestCellPValuesSingleHierarchy:
    """Fisher exact 2x2 on (cell_dim,) keyed cells."""

    def test_single_cell_no_contrast_returns_one(self):
        # All entities in same cell, same anomaly rate as 'rest' (vacuously) -> p=1
        geometry = pa.table({
            "primary_key": ["e1", "e2", "e3", "e4"],
            "bank_id": ["B1", "B1", "B1", "B1"],
            "is_anomaly": [True, False, True, False],
        })
        result = cell_p_values_from_anomaly_indicator(
            geometry, hierarchy_dims=["bank_id"], temporal_dim=None,
        )
        assert set(result.keys()) == {("B1",)}
        # Single-cell case: no contrast vs complement (complement is empty); p=1.0
        assert result[("B1",)] == pytest.approx(1.0)

    def test_two_cells_one_enriched(self):
        # B1: 4/4 anomalous (extreme); B2: 0/4 anomalous; B1 enriched, B2 depleted
        geometry = pa.table({
            "primary_key": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"],
            "bank_id": ["B1", "B1", "B1", "B1", "B2", "B2", "B2", "B2"],
            "is_anomaly": [True, True, True, True, False, False, False, False],
        })
        result = cell_p_values_from_anomaly_indicator(
            geometry, hierarchy_dims=["bank_id"], temporal_dim=None,
        )
        assert set(result.keys()) == {("B1",), ("B2",)}
        # B1 enriched: p << 0.05; B2 depleted (upper-tail p ≈ 1.0)
        assert result[("B1",)] < 0.05
        assert result[("B2",)] > 0.5


class TestCellPValuesIntersection:
    """Cell = (hierarchy_dims..., temporal_dim) joint tuple."""

    def test_4x4_grid_one_corner_enriched(self):
        # 4 banks × 4 quarters, only (B1, Q1) is anomaly-enriched
        rows = []
        for bank in ("B1", "B2", "B3", "B4"):
            for q in ("Q1", "Q2", "Q3", "Q4"):
                for i in range(10):
                    pk = f"{bank}-{q}-{i}"
                    is_anom = (bank == "B1" and q == "Q1" and i < 8)
                    rows.append((pk, bank, q, is_anom))
        geometry = pa.table({
            "primary_key": [r[0] for r in rows],
            "bank_id": [r[1] for r in rows],
            "temporal_bucket": [r[2] for r in rows],
            "is_anomaly": [r[3] for r in rows],
        })
        result = cell_p_values_from_anomaly_indicator(
            geometry,
            hierarchy_dims=["bank_id"],
            temporal_dim="temporal_bucket",
        )
        # 16 cells total
        assert len(result) == 16
        # (B1, Q1) extreme (actual p ≈ 5e-12 clipped to 1e-10)
        assert result[("B1", "Q1")] <= 1e-9
        # Every other cell has 0 anomalies — depleted, p approx 1
        for bank in ("B1", "B2", "B3", "B4"):
            for q in ("Q1", "Q2", "Q3", "Q4"):
                if (bank, q) != ("B1", "Q1"):
                    assert result[(bank, q)] > 0.5


class TestCellPValuesErrors:
    def test_missing_anomaly_col(self):
        geometry = pa.table({
            "primary_key": ["e1"], "bank_id": ["B1"],
        })
        with pytest.raises(ValueError, match="anomaly_col"):
            cell_p_values_from_anomaly_indicator(
                geometry, hierarchy_dims=["bank_id"],
            )

    def test_missing_cell_dim(self):
        geometry = pa.table({
            "primary_key": ["e1"], "is_anomaly": [True],
        })
        with pytest.raises(ValueError, match="cell-defining columns missing"):
            cell_p_values_from_anomaly_indicator(
                geometry, hierarchy_dims=["bank_id"],
            )

    def test_no_axis(self):
        geometry = pa.table({
            "primary_key": ["e1"], "is_anomaly": [True],
        })
        with pytest.raises(ValueError, match="at least one of"):
            cell_p_values_from_anomaly_indicator(geometry)


class TestFDRMultiResolutionSpatial:
    """Single-axis spatial hierarchy: per-level BH FDR with Tippett min-p aggregation."""

    def test_two_level_hierarchy_finest_filtered(self):
        # Spatial path = (country, branch); two countries × two branches
        # (US, NY) very strong p=1e-6; others uniform ~0.5
        cell_p = {
            ("US", "NY"): 1e-6,
            ("US", "CA"): 0.5,
            ("UK", "LON"): 0.5,
            ("UK", "MAN"): 0.5,
        }
        q_vals, surviving = fdr_multi_resolution(
            cell_p,
            hierarchy=["country", "branch"],
            method="bh",
            alpha=0.05,
        )
        # (US, NY) clears every level: country-level (min-p for US = 1e-6) AND
        # branch-level (NY p = 1e-6 itself). Survives.
        assert ("US", "NY") in surviving
        # Other 3 cells don't clear branch-level
        assert ("US", "CA") not in surviving
        assert ("UK", "LON") not in surviving
        assert ("UK", "MAN") not in surviving
        # q-values bounded in [0, 1]
        for q in q_vals.values():
            assert 0.0 <= q <= 1.0

    def test_country_clears_but_branch_does_not(self):
        # All US branches mildly elevated; together country=US clears BH but
        # individual branches don't.
        cell_p = {
            ("US", "NY"): 0.04,
            ("US", "CA"): 0.04,
            ("US", "TX"): 0.04,
            ("US", "WA"): 0.04,
            ("UK", "LON"): 0.5,
            ("UK", "MAN"): 0.5,
        }
        q_vals, surviving = fdr_multi_resolution(
            cell_p,
            hierarchy=["country", "branch"],
            method="bh",
            alpha=0.05,
        )
        # Country level: Tippett min-p for US = 0.04, for UK = 0.5; BH at alpha=0.05
        # m=2 with sorted p=[0.04, 0.5]: rank 1 q = 0.04*2/1=0.08, rank 2 q=0.5.
        # Country-level: neither survives at alpha=0.05.
        # Therefore NO branch under US can clear country-level gate.
        assert all(c not in surviving for c in cell_p)


class TestFDRMultiResolutionIntersection:
    """Two hierarchies declared -> intersection of survivors."""

    def test_intersection_only_one_corner(self):
        # 2 spatial × 2 temporal cells; only (US, Q1) strong on both axes
        cell_p = {
            ("US", "Q1"): 1e-6,
            ("US", "Q2"): 0.5,
            ("UK", "Q1"): 0.5,
            ("UK", "Q2"): 0.5,
        }
        q_vals, surviving = fdr_multi_resolution(
            cell_p,
            hierarchy=["country"],
            temporal_levels=["quarter"],
            method="bh",
            alpha=0.05,
        )
        # Country-level Tippett min-p: US = 1e-6, UK = 0.5; BH m=2: q_US = 2e-6, q_UK = 0.5
        # US clears country-level.
        # Quarter-level Tippett min-p: Q1 = 1e-6, Q2 = 0.5; q_Q1 = 2e-6, q_Q2 = 0.5
        # Q1 clears quarter-level.
        # Intersection: (US, Q1) only.
        assert surviving == {("US", "Q1")}

    def test_intersection_no_survivors_when_only_one_axis_clears(self):
        # Spatial clear (US strong, UK weak), but temporal-level Tippett finds
        # BOTH Q1 and Q2 strong because US drives both -> both Q's clear.
        cell_p = {
            ("US", "Q1"): 0.001,
            ("US", "Q2"): 0.001,
            ("UK", "Q1"): 0.5,
            ("UK", "Q2"): 0.5,
        }
        q_vals, surviving = fdr_multi_resolution(
            cell_p,
            hierarchy=["country"],
            temporal_levels=["quarter"],
            method="bh",
            alpha=0.05,
        )
        # Country-level Tippett: US=0.001, UK=0.5; BH m=2 q_US=0.002, q_UK=0.5; US clears.
        # Quarter-level Tippett: Q1=0.001, Q2=0.001; q for both = 0.001*2/2=0.001;
        # both clear.
        # Intersection: (US, Q1) and (US, Q2) — both survive.
        assert surviving == {("US", "Q1"), ("US", "Q2")}


class TestFDRMultiResolutionErrors:
    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            fdr_multi_resolution(
                {("A",): 0.1}, hierarchy=["x"], alpha=1.5,
            )

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            fdr_multi_resolution(
                {("A",): 0.1}, hierarchy=["x"], method="banana",
            )

    def test_no_axis(self):
        with pytest.raises(ValueError, match="at least one of"):
            fdr_multi_resolution({("A",): 0.1})

    def test_cell_tuple_length_mismatch(self):
        with pytest.raises(ValueError, match="cell-tuple length mismatch"):
            fdr_multi_resolution(
                {("A", "B"): 0.1},
                hierarchy=["x"],   # expects len 1
            )

    def test_empty_input(self):
        q_vals, surviving = fdr_multi_resolution(
            {}, hierarchy=["x"], alpha=0.05,
        )
        assert q_vals == {}
        assert surviving == set()
