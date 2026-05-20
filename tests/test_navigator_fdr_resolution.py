# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for π5_attract_anomaly fdr_resolution + fdr_temporal_resolution wiring."""
from __future__ import annotations

import re
import warnings
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from hypertopos.model.manifest import Contract, Manifest
from hypertopos.model.sphere import (
    FDRHierarchyLevel,
    FDRTemporalLevel,
    Pattern,
    RelationDef,
    Sphere,
)
from hypertopos.navigation.navigator import GDSNavigator

_DT = datetime(2024, 1, 1, tzinfo=UTC)


def _make_pattern_with_hierarchy(spatial: bool, temporal: bool) -> Pattern:
    return Pattern(
        pattern_id="p_x",
        entity_type="x",
        pattern_type="anchor",
        relations=[RelationDef(line_id="line_a", direction="in", required=True)],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.array([0.0, 3.0], dtype=np.float32),
        population_size=20,
        computed_at=_DT,
        version=1,
        status="production",
        fdr_hierarchy=(
            [FDRHierarchyLevel(level="bank", from_dimension="bank_id")] if spatial else []
        ),
        fdr_temporal_hierarchy=(
            [FDRTemporalLevel(level="quarter", slice_dimension="temporal_bucket")]
            if temporal else []
        ),
    )


def _make_geometry_table(banks=("B1", "B2"), n_per_bank=8, temporal=False):
    """B1 entities are all anomalous; B2 entities are all normal."""
    keys: list[str] = []
    deltas: list[list[float]] = []
    norms: list[float] = []
    is_anomaly: list[bool] = []
    bank_ids: list[str] = []
    quarters: list[str] = []
    for bank in banks:
        # B1 entities: above threshold (anomalous, large delta)
        # B2 entities: below threshold (normal, small delta)
        for i in range(n_per_bank):
            keys.append(f"{bank}-{i}")
            if bank == "B1":
                d = np.array(
                    [3.0 + float(i) / n_per_bank, 3.0 + float(i) / n_per_bank],
                    dtype=np.float32,
                )
            else:
                # below theta_norm ~3.0 -> not anomalous
                d = np.array([0.5, 0.5], dtype=np.float32)
            deltas.append(d.tolist())
            norms.append(float(np.linalg.norm(d)))
            is_anomaly.append(bank == "B1")
            bank_ids.append(bank)
            quarters.append("Q1" if bank == "B1" else "Q2")
    # delta_rank_pct: 100 = most-extreme. B1 entities sit at the top end of
    # the population (≥ 95) so they sail through entity-axis FDR; B2 entities
    # land near 0 (irrelevant — they're below threshold and never reach FDR).
    n_total = len(keys)
    rank_pcts: list[float] = []
    for bank in banks:
        for i in range(n_per_bank):
            if bank == "B1":
                # Spread B1 across (95, 99.999) for distinct p-values
                rank_pcts.append(95.0 + 4.999 * (i / max(n_per_bank - 1, 1)))
            else:
                rank_pcts.append(float(i) / n_total * 5.0)
    cols: dict = {
        "primary_key": keys,
        "scale": [1] * len(keys),
        "delta": deltas,
        "delta_norm": norms,
        "is_anomaly": is_anomaly,
        "delta_rank_pct": pa.array(rank_pcts, type=pa.float64()),
        "last_refresh_at": [_DT] * len(keys),
        "updated_at": [_DT] * len(keys),
        "bank_id": bank_ids,
    }
    if temporal:
        cols["temporal_bucket"] = quarters
    return pa.table(cols)


class _MockStorage:
    """Mock storage exposing one pattern with hierarchy + geometry that
    includes bank_id (and optionally temporal_bucket)."""

    def __init__(self, pattern: Pattern, geometry: pa.Table):
        self._pattern = pattern
        self._geometry = geometry

    def read_sphere(self):
        sphere = Sphere("s", "s", ".")
        sphere.patterns[self._pattern.pattern_id] = self._pattern
        return sphere

    def count_geometry_rows(self, *a, **kw):
        return 0  # force in-process path

    def read_geometry(self, pattern_id, version, primary_key=None,
                      filters=None, point_keys=None, columns=None, filter=None):
        table = self._geometry
        if filter is not None:
            f = str(filter)
            if "primary_key IN" in f:
                keys = re.findall(r"'([^']*)'", f)
                table = table.filter(pc.is_in(table["primary_key"], pa.array(keys)))
            elif "delta_norm >=" in f:
                threshold = float(f.split(">=")[1].strip())
                table = table.filter(pc.greater_equal(table["delta_norm"], threshold))
        if columns is not None:
            keep = [c for c in columns if c in table.schema.names]
            table = table.select(keep)
        return table

    def read_geometry_stats(self, *a, **kw):
        return None


def _nav_with(pattern: Pattern, geometry: pa.Table) -> GDSNavigator:
    storage = _MockStorage(pattern, geometry)
    manifest = Manifest(
        manifest_id="m1", agent_id="t", snapshot_time=_DT,
        status="active", line_versions={"line_a": 1},
        pattern_versions={"p_x": 1},
    )
    contract = Contract(manifest_id="m1", pattern_ids=["p_x"])
    engine = MagicMock()
    engine.geometry_to_polygons = MagicMock(return_value=[])
    return GDSNavigator(engine, storage, manifest, contract)


class TestReliabilityFlagsAttachedByNavigator:
    """Reliability flags must appear on the polygons returned by
    π5_attract_anomaly. Verifies the navigator-level wiring of
    `_attach_reliability_flags` — engine-level unit tests in
    test_reliability_flags.py cover the math, this one covers the wire.
    """

    def test_pi5_attract_anomaly_attaches_reliability_flags(self):
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        polys, _t, _e, _m = nav.π5_attract_anomaly(
            "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
        )
        assert len(polys) > 0
        for poly in polys:
            assert hasattr(poly, "reliability_flags")
            flags = poly.reliability_flags
            assert isinstance(flags, dict)
            assert set(flags.keys()) == {
                "single_dim_driven",
                "dominant_dim",
                "dominant_dim_share",
                "low_confidence_bucket",
                "confidence",
                "flags",
            }
            assert isinstance(flags["single_dim_driven"], bool)
            assert isinstance(flags["low_confidence_bucket"], bool)
            assert isinstance(flags["flags"], list)


class TestFDRResolutionSpatial:
    def test_spatial_gate_keeps_only_anomalous_bank(self):
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        polys, total_found, _emerging, _meta = nav.π5_attract_anomaly(
            "p_x", top_n=30, fdr_alpha=0.05, fdr_resolution="bank",
            p_value_method="chi2", fdr_method="storey",
        )
        # Every survivor must come from anomaly-enriched bank B1
        assert len(polys) > 0
        for poly in polys:
            assert poly.primary_key.startswith("B1-")
        # B1 entities are 8/8 -> should not all be filtered out
        assert any(p.primary_key.startswith("B1-") for p in polys)

    def test_unknown_resolution_raises(self):
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        with pytest.raises(ValueError, match="fdr_resolution"):
            nav.π5_attract_anomaly(
                "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="not_a_level",
            )

    def test_resolution_without_alpha_raises(self):
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        with pytest.raises(ValueError, match="fdr_alpha"):
            nav.π5_attract_anomaly(
                "p_x", top_n=10, fdr_resolution="bank",
            )


class TestFDRResolutionIntersection:
    def test_both_axes_intersect(self):
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=True)
        nav = _nav_with(pat, _make_geometry_table(temporal=True))
        polys, _total, _emerging, _meta = nav.π5_attract_anomaly(
            "p_x", top_n=50,
            fdr_alpha=0.05,
            fdr_resolution="bank",
            fdr_temporal_resolution="quarter",
            p_value_method="chi2", fdr_method="storey",
        )
        # B1 is in Q1 + all anomalous -> intersection should keep B1-* only
        assert len(polys) > 0
        for poly in polys:
            assert poly.primary_key.startswith("B1-")

    def test_unknown_temporal_resolution_raises(self):
        pat = _make_pattern_with_hierarchy(spatial=False, temporal=True)
        nav = _nav_with(pat, _make_geometry_table(temporal=True))
        with pytest.raises(ValueError, match="fdr_temporal_resolution"):
            nav.π5_attract_anomaly(
                "p_x", top_n=10, fdr_alpha=0.05,
                fdr_temporal_resolution="not_a_level",
            )

    def test_survivors_carry_annotations(self):
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=True)
        nav = _nav_with(pat, _make_geometry_table(temporal=True))
        polys, *_ = nav.π5_attract_anomaly(
            "p_x", top_n=30, fdr_alpha=0.05,
            fdr_resolution="bank", fdr_temporal_resolution="quarter",
            p_value_method="chi2", fdr_method="storey",
        )
        assert polys
        p = polys[0]
        assert hasattr(p, "cell_q_spatial") and p.cell_q_spatial is not None
        assert hasattr(p, "cell_q_temporal") and p.cell_q_temporal is not None
        assert p.cell_path == (("bank", "B1"), ("quarter", "Q1"))

    def test_survivors_are_subset_of_pre_gate_polygons(self):
        """Gating must filter, never re-fetch — survivors are a subset."""
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        # First call: no gate -> baseline
        baseline, _t, _e, _m = nav.π5_attract_anomaly("p_x", top_n=30)
        baseline_keys = {p.primary_key for p in baseline}
        # Second call: with gate (auto-upgrades rank+bh -> chi2+storey
        # silently when fdr_resolution is set on entity axis)
        gated, _t, _e, _m = nav.π5_attract_anomaly(
            "p_x", top_n=30, fdr_alpha=0.05, fdr_resolution="bank",
        )
        gated_keys = {p.primary_key for p in gated}
        assert gated_keys.issubset(baseline_keys)


class TestFDRResolutionDefaultsAutoUpgrade:
    """When fdr_resolution / fdr_temporal_resolution is set on the
    entity axis, p_value_method='rank' + fdr_method='bh' (the documented
    defaults) silently upgrade to chi2 + storey because the rank+bh
    combo is degenerate (uniform p-values, BH rejects nothing). Explicit
    non-default values are not overridden. per_dim / both axes don't
    use entity-axis FDR, so no upgrade happens there.
    """

    def test_rank_bh_with_spatial_resolution_keeps_entities(self):
        """Pre-upgrade rank+bh combo returned zero meaningful survivors on
        AML scale because BH rejected nothing. After auto-upgrade to
        chi2+storey under the hood, the call returns non-empty top-K."""
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        polys, _t, _e, _m = nav.π5_attract_anomaly(
            "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
        )
        assert len(polys) > 0
        # And every survivor must come from the anomaly-enriched bank
        for poly in polys:
            assert poly.primary_key.startswith("B1-")

    def test_no_warning_on_any_combo(self):
        """No combo should emit a warning — auto-upgrade replaces warning."""
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        # rank+bh (would have warned before)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            nav.π5_attract_anomaly(
                "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
            )
        # explicit chi2+storey
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            nav.π5_attract_anomaly(
                "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
                p_value_method="chi2", fdr_method="storey",
            )
        # no resolution at all
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            nav.π5_attract_anomaly("p_x", top_n=10)
        # per_dim axis (entity-axis FDR not on path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            nav.π5_attract_anomaly(
                "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
                fdr_axis="per_dim",
            )

    def test_explicit_chi2_storey_matches_auto_upgrade(self):
        """Explicit chi2+storey produces the same output as the auto-upgrade
        path — sanity that the upgrade picks the documented combo, not some
        third hidden code branch."""
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        polys_explicit, *_ = nav.π5_attract_anomaly(
            "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
            p_value_method="chi2", fdr_method="storey",
        )
        polys_auto, *_ = nav.π5_attract_anomaly(
            "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
        )
        assert {p.primary_key for p in polys_explicit} == {
            p.primary_key for p in polys_auto
        }

    def test_explicit_rank_is_not_overridden(self):
        """A caller who explicitly passes p_value_method='rank' wants the
        rank-uniform p-values (e.g. for pre-upgrade reproduction, migration
        validation, benchmarking the degenerate path on purpose). The
        sentinel-None pattern must keep that escape hatch — explicit 'rank'
        bypasses the auto-upgrade and produces different (zero-survivor on
        this fixture) output than the upgraded path."""
        pat = _make_pattern_with_hierarchy(spatial=True, temporal=False)
        nav = _nav_with(pat, _make_geometry_table())
        polys_rank, *_ = nav.π5_attract_anomaly(
            "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
            p_value_method="rank", fdr_method="bh",
        )
        polys_auto, *_ = nav.π5_attract_anomaly(
            "p_x", top_n=10, fdr_alpha=0.05, fdr_resolution="bank",
        )
        # Auto-upgrade returns survivors with real chi2 ranking inside the
        # surviving cells; explicit rank+bh hands the entity-level FDR a
        # uniform p-value distribution where BH at alpha=0.05 cannot
        # reject — so the survivor sets must differ.
        assert {p.primary_key for p in polys_rank} != {
            p.primary_key for p in polys_auto
        }
