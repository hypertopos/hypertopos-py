"""Unit tests for the M4 find_calibration_influencers / find_group_influence
primitives.

Covers: DimensionContribution + InfluenceEntry + InfluenceReport +
GroupInfluenceReport dataclasses, math helpers, classification, cascading
flips, group leave-set-out, orchestrator validation gates, π5 additive fields.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Layer 1a — Dataclass construction + frozen
# ---------------------------------------------------------------------------

def test_dimension_contribution_construction_and_frozen():
    from hypertopos.model.sphere import DimensionContribution

    d = DimensionContribution(
        dim_index=2,
        dim_kind="gaussian",
        dim_label="amount",
        mu_shift=0.05,
        sigma_shift=0.12,
        contribution=0.13,
    )
    assert d.dim_index == 2
    assert d.dim_label == "amount"
    assert d.contribution == pytest.approx(0.13)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.mu_shift = 99.0  # type: ignore[misc]


def test_influence_entry_construction_and_frozen():
    from hypertopos.model.sphere import DimensionContribution, InfluenceEntry

    dd = DimensionContribution(
        dim_index=0,
        dim_kind="gaussian",
        dim_label="d0",
        mu_shift=0.0,
        sigma_shift=0.0,
        contribution=0.0,
    )
    e = InfluenceEntry(
        entity_key="E1",
        mu_impact=0.3,
        sigma_impact=0.5,
        total_impact=0.583,
        delta_norm=1.2,
        classification="hidden",
        top_dim_contributions=[dd],
        cascading_flip_count=None,
    )
    assert e.entity_key == "E1"
    assert e.classification == "hidden"
    assert e.cascading_flip_count is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.total_impact = 99.0  # type: ignore[misc]


def test_influence_report_construction_and_frozen():
    from hypertopos.model.sphere import (
        InfluenceEntry,
        InfluenceReport,
    )

    e = InfluenceEntry(
        entity_key="E1",
        mu_impact=0.3,
        sigma_impact=0.5,
        total_impact=0.583,
        delta_norm=1.2,
        classification="hidden",
        top_dim_contributions=[],
        cascading_flip_count=None,
    )
    report = InfluenceReport(
        pattern_id="p",
        pattern_version=1,
        population_size=100,
        high_threshold_pct=90.0,
        total_impact_threshold=0.4,
        theta_norm=2.5,
        classify_filter="hidden",
        cell_counts={"hidden": 5, "distorter": 1, "standard_anomaly": 4, "normal": 90},
        entries=[e],
    )
    assert report.population_size == 100
    assert report.cell_counts["hidden"] == 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.population_size = 200  # type: ignore[misc]


def test_group_influence_report_construction_and_frozen():
    from hypertopos.model.sphere import (
        GroupInfluenceReport,
    )

    g = GroupInfluenceReport(
        pattern_id="p",
        pattern_version=1,
        group_index=0,
        member_count=2,
        members=["E1", "E2"],
        mu_impact_set=0.4,
        sigma_impact_set=0.2,
        total_impact_set=0.447,
        sum_individual_impacts=0.3,
        reinforcing_factor=1.49,
        top_dim_contributions=[],
    )
    assert g.member_count == 2
    assert g.reinforcing_factor == pytest.approx(1.49)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.member_count = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Layer 1b — Math helper: _compute_leave_one_out_impact
# ---------------------------------------------------------------------------

def test_leave_one_out_two_entities_closed_form():
    from hypertopos.engine.geometry import _compute_leave_one_out_impact

    shapes = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    mu_imp, sigma_imp, total_imp, contrib = _compute_leave_one_out_impact(
        shapes, mu_full, sigma_full,
    )
    assert mu_imp.shape == (2,)
    assert total_imp.shape == (2,)
    assert contrib.shape == (2, 2)
    assert total_imp[0] == pytest.approx(total_imp[1], rel=1e-6)


def test_leave_one_out_pure_mu_shift():
    from hypertopos.engine.geometry import _compute_leave_one_out_impact

    shapes = np.zeros((10, 2), dtype=np.float64)
    shapes[9] = [10.0, 0.0]
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    mu_imp, sigma_imp, total_imp, _ = _compute_leave_one_out_impact(
        shapes, mu_full, sigma_full,
    )
    assert np.argmax(total_imp) == 9


def test_leave_one_out_pure_sigma_shift_HIDDEN_INFLUENCER_GUARD():
    """PATENT-DEFENSIVE GUARD: this test fails if math regresses to plan-as-
    written A (μ-only first-order approximation).

    Construction: 9 entities clustered at [0, 0], 1 entity at [0, 5]. Cluster
    has σ ≈ 0 on dim 0, σ inflated on dim 1 by the single outlier on dim 1.
    The outlier sits AT μ on dim 0 (δ_norm contribution from dim 0 = 0),
    AT 5 on dim 1 (δ_norm contribution from dim 1 normalised by inflated σ).

    Expected: outlier's sigma_impact >> mu_impact (sigma_shift comes from
    removing the only entity inflating dim-1 variance)."""
    from hypertopos.engine.geometry import _compute_leave_one_out_impact

    shapes = np.zeros((10, 2), dtype=np.float64)
    shapes[9] = [0.0, 5.0]
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    mu_imp, sigma_imp, total_imp, _ = _compute_leave_one_out_impact(
        shapes, mu_full, sigma_full,
    )
    assert sigma_imp[9] > mu_imp[9], (
        "Hidden-influencer guard failed: σ-shift component of outlier did "
        "NOT dominate μ-shift. Math regressed to plan-as-written μ-only "
        "approximation; hidden influencer cell is now empty by construction."
    )


def test_leave_one_out_zero_drift_yields_zero_impact():
    from hypertopos.engine.geometry import _compute_leave_one_out_impact

    shapes = np.full((5, 3), 7.0, dtype=np.float64)
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    mu_imp, sigma_imp, total_imp, _ = _compute_leave_one_out_impact(
        shapes, mu_full, sigma_full,
    )
    np.testing.assert_allclose(mu_imp, 0.0, atol=1e-9)
    np.testing.assert_allclose(sigma_imp, 0.0, atol=1e-9)
    np.testing.assert_allclose(total_imp, 0.0, atol=1e-9)


def test_leave_one_out_sigma_safe_on_degenerate_dim():
    from hypertopos.engine.geometry import _compute_leave_one_out_impact

    shapes = np.zeros((5, 2), dtype=np.float64)
    shapes[:, 1] = [1.0, 2.0, 3.0, 4.0, 5.0]
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    mu_imp, sigma_imp, total_imp, _ = _compute_leave_one_out_impact(
        shapes, mu_full, sigma_full,
    )
    assert np.all(np.isfinite(mu_imp))
    assert np.all(np.isfinite(sigma_imp))
    assert np.all(np.isfinite(total_imp))


def test_leave_one_out_per_dim_contributions_ranked():
    from hypertopos.engine.geometry import _compute_leave_one_out_impact

    shapes = np.zeros((10, 3), dtype=np.float64)
    shapes[0] = [0.0, 8.0, 0.0]
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    _, _, _, contrib = _compute_leave_one_out_impact(shapes, mu_full, sigma_full)
    abs_contrib_e0 = np.abs(contrib[0])
    assert np.argmax(abs_contrib_e0) == 1


# ---------------------------------------------------------------------------
# Layer 1c — Classification: _classify_influence
# ---------------------------------------------------------------------------

def test_classify_hidden_cell():
    from hypertopos.engine.geometry import _classify_influence

    total_impact = np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    delta_norm = np.array([0.1, 0.1, 0.1, 9.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    theta_norm = 5.0
    classes = _classify_influence(
        total_impact, delta_norm, theta_norm, high_threshold_pct=90.0,
    )
    assert classes[0] == "hidden"
    assert classes[3] == "distorter"


def test_classify_distorter_cell():
    from hypertopos.engine.geometry import _classify_influence

    total_impact = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    delta_norm = np.array([9.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    theta_norm = 5.0
    classes = _classify_influence(
        total_impact, delta_norm, theta_norm, high_threshold_pct=90.0,
    )
    assert classes[0] == "distorter"


def test_classify_standard_anomaly_cell():
    """Entity 0 has lowest total_impact (below 90th percentile) but high
    delta_norm — must classify as standard_anomaly."""
    from hypertopos.engine.geometry import _classify_influence

    # Distribution where percentile_90 cleanly separates entity 0 (low) from rest.
    total_impact = np.array(
        [0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )
    delta_norm = np.array([9.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    theta_norm = 5.0
    classes = _classify_influence(
        total_impact, delta_norm, theta_norm, high_threshold_pct=90.0,
    )
    # 90th percentile of the array = 0.5; entity 0 at 0.05 is below cutoff.
    assert classes[0] == "standard_anomaly"


def test_classify_normal_cell():
    """Entities 0..8 below 90th percentile + low anomaly → normal cell."""
    from hypertopos.engine.geometry import _classify_influence

    # 9 entities at low impact, 1 at high impact. 90th percentile picks the
    # high one only; bottom 9 are below cutoff. All have low anomaly.
    total_impact = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 5.0])
    delta_norm = np.full(10, 0.1)
    theta_norm = 5.0
    classes = _classify_influence(
        total_impact, delta_norm, theta_norm, high_threshold_pct=90.0,
    )
    # Bottom 9 → normal; entity 9 → hidden (high impact + low anomaly).
    assert all(c == "normal" for c in classes[:9])
    assert classes[9] == "hidden"


def test_classify_threshold_pct_param_sensitivity():
    from hypertopos.engine.geometry import _classify_influence

    rng = np.random.default_rng(42)
    total_impact = rng.uniform(0.0, 1.0, size=100)
    delta_norm = np.full(100, 0.1)
    theta_norm = 5.0
    classes_50 = _classify_influence(
        total_impact, delta_norm, theta_norm, high_threshold_pct=50.0,
    )
    classes_99 = _classify_influence(
        total_impact, delta_norm, theta_norm, high_threshold_pct=99.0,
    )
    n_hidden_50 = sum(c == "hidden" for c in classes_50)
    n_hidden_99 = sum(c == "hidden" for c in classes_99)
    assert n_hidden_50 > n_hidden_99
    assert n_hidden_50 >= 45
    assert n_hidden_99 <= 5


# ---------------------------------------------------------------------------
# Layer 1d — Cascading flips: _count_cascading_flips
# ---------------------------------------------------------------------------

def test_cascading_no_flips_when_population_stable():
    from hypertopos.engine.geometry import _count_cascading_flips

    rng = np.random.default_rng(0)
    shapes = rng.normal(0.0, 1.0, size=(50, 3)).astype(np.float64)
    sum_s = shapes.sum(axis=0)
    sum_s_sq = (shapes ** 2).sum(axis=0)
    mu_full = sum_s / 50
    sigma_full = np.sqrt(sum_s_sq / 50 - mu_full ** 2)
    sigma_full_safe = np.maximum(sigma_full, 1e-12)
    deltas_full = (shapes - mu_full) / sigma_full_safe
    delta_norm_full = np.linalg.norm(deltas_full, axis=1)
    theta_norm = 3.0
    is_anom_full = delta_norm_full >= theta_norm
    e_idx = int(np.argmin(delta_norm_full))
    flips = _count_cascading_flips(
        shape_E=shapes[e_idx],
        sum_s=sum_s,
        sum_s_sq=sum_s_sq,
        shapes=shapes,
        is_anomaly_full=is_anom_full,
        e_idx=e_idx,
        theta_norm=theta_norm,
    )
    assert flips <= 2


def test_cascading_only_with_verbose_true():
    """Helper returns non-negative int. Orchestrator-level verbose gating
    tested separately below."""
    from hypertopos.engine.geometry import _count_cascading_flips

    rng = np.random.default_rng(1)
    shapes = rng.normal(0.0, 1.0, size=(30, 2)).astype(np.float64)
    sum_s = shapes.sum(axis=0)
    sum_s_sq = (shapes ** 2).sum(axis=0)
    mu_full = sum_s / 30
    sigma_full = np.sqrt(sum_s_sq / 30 - mu_full ** 2)
    sigma_full_safe = np.maximum(sigma_full, 1e-12)
    deltas_full = (shapes - mu_full) / sigma_full_safe
    delta_norm_full = np.linalg.norm(deltas_full, axis=1)
    is_anom_full = delta_norm_full >= 2.0
    flips = _count_cascading_flips(
        shape_E=shapes[0],
        sum_s=sum_s,
        sum_s_sq=sum_s_sq,
        shapes=shapes,
        is_anomaly_full=is_anom_full,
        e_idx=0,
        theta_norm=2.0,
    )
    assert isinstance(flips, int)
    assert flips >= 0


def test_cascading_flip_count_differs_for_engineered_distinct_outliers():
    """Discriminator test for advisor concern: 3 engineered outliers with
    VERY DIFFERENT shapes must produce DIFFERENT cascading_flip_count.

    Berka live smoke showed `cascading_flip_count=131` identical across 3
    distinct hidden influencers (delta_norm 5.460/5.425/5.399). The
    self-rationalisation was 'similar shapes → similar leave-one-out → same
    flip count'. This test puts the helper under engineered-distinct shapes
    where flip-count uniformity would prove a real bug (e.g. wrong
    is_anomaly_full baseline, mutated state, etc.).

    Expected: flip counts differ for at least one pair of the 3 distinct
    outliers. If they're all identical here, _count_cascading_flips has a
    real bug — investigate before shipping."""
    from hypertopos.engine.geometry import _count_cascading_flips

    rng = np.random.default_rng(11)
    shapes = rng.normal(0.0, 0.3, size=(50, 3)).astype(np.float64)
    # Three engineered outliers with shapes of WILDLY DIFFERENT magnitudes
    # along orthogonal axes. Orthogonal alone is not enough — σ-shrink on
    # different dims at similar magnitudes produces similar flip counts
    # (legit math). Asymmetric magnitudes create asymmetric σ-shrinks →
    # asymmetric flip counts. If counts are STILL identical, bug confirmed.
    shapes[0] = [50.0, 0.0, 0.0]   # huge on dim 0 → big σ-shrink on dim 0
    shapes[1] = [0.0, 5.0, 0.0]    # medium on dim 1 → medium σ-shrink on dim 1
    shapes[2] = [0.0, 0.0, 1.0]    # small on dim 2 → small σ-shrink on dim 2

    sum_s = shapes.sum(axis=0)
    sum_s_sq = (shapes ** 2).sum(axis=0)
    mu_full = sum_s / 50
    sigma_full = np.sqrt(np.maximum(sum_s_sq / 50 - mu_full ** 2, 0.0))
    sigma_full_safe = np.maximum(sigma_full, 1e-12)
    deltas_full = (shapes - mu_full) / sigma_full_safe
    delta_norm_full = np.linalg.norm(deltas_full, axis=1)
    # θ chosen so ~30% of population (excluding outliers) is anomalous —
    # leaves room for σ-perturbation to flip neighbours either direction.
    theta_norm = float(np.percentile(delta_norm_full[3:], 70))
    is_anom_full = delta_norm_full >= theta_norm

    flips = []
    for i in range(3):
        f = _count_cascading_flips(
            shape_E=shapes[i],
            sum_s=sum_s,
            sum_s_sq=sum_s_sq,
            shapes=shapes,
            is_anomaly_full=is_anom_full,
            e_idx=i,
            theta_norm=theta_norm,
        )
        flips.append(f)

    assert len(set(flips)) > 1, (
        f"Engineered outliers with orthogonal shapes "
        f"[10,0,0]/[0,10,0]/[0,0,10] produced identical cascading_flip_count="
        f"{flips}. Likely bug in _count_cascading_flips — possibly using "
        f"wrong is_anomaly_full baseline or mutating shared state."
    )


def test_cascading_flips_neighbours_near_threshold_POSITIVE_CASE():
    """Positive case: removing a high-σ outlier shifts variance
    enough to push near-threshold neighbours over θ. Without this test,
    cascading-flips only has 'returns 0' coverage."""
    from hypertopos.engine.geometry import _count_cascading_flips

    # Construct: 1 outlier inflates σ on dim 0; remaining 19 entities
    # spread evenly so several sit within (0.7θ, θ) — within striking
    # distance of θ once σ shrinks.
    rng = np.random.default_rng(7)
    shapes = rng.normal(0.0, 0.3, size=(20, 2)).astype(np.float64)
    shapes[0] = [10.0, 0.0]  # σ-inflating outlier
    sum_s = shapes.sum(axis=0)
    sum_s_sq = (shapes ** 2).sum(axis=0)
    mu_full = sum_s / 20
    sigma_full = np.sqrt(np.maximum(sum_s_sq / 20 - mu_full ** 2, 0.0))
    sigma_full_safe = np.maximum(sigma_full, 1e-12)
    deltas_full = (shapes - mu_full) / sigma_full_safe
    delta_norm_full = np.linalg.norm(deltas_full, axis=1)
    # Pick θ so that ~30% of the population is anomalous before E0 removal —
    # leaves room for flips after σ-shrink to push more entities over.
    theta_norm = float(np.percentile(delta_norm_full[1:], 70))
    is_anom_full = delta_norm_full >= theta_norm
    flips = _count_cascading_flips(
        shape_E=shapes[0],
        sum_s=sum_s,
        sum_s_sq=sum_s_sq,
        shapes=shapes,
        is_anomaly_full=is_anom_full,
        e_idx=0,
        theta_norm=theta_norm,
    )
    assert flips >= 1, (
        "Removing a high-σ outlier should flip ≥1 near-threshold neighbour. "
        "If flips == 0, the is_anomaly comparison mask is broken."
    )


# ---------------------------------------------------------------------------
# Layer 1e — Group leave-set-out: _compute_leave_set_out_impact
# ---------------------------------------------------------------------------

def test_group_reinforcing_pure_duplicates():
    from hypertopos.engine.geometry import (
        _compute_leave_one_out_impact,
        _compute_leave_set_out_impact,
    )

    shapes = np.zeros((20, 2), dtype=np.float64)
    shapes[0] = [5.0, 0.0]
    shapes[1] = [5.0, 0.0]
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    mu_imp, sigma_imp, total_imp, _ = _compute_leave_one_out_impact(
        shapes, mu_full, sigma_full,
    )
    sum_individual = total_imp[0] + total_imp[1]

    set_imp = _compute_leave_set_out_impact(
        shapes=shapes,
        members_idx=np.array([0, 1]),
        mu_full=mu_full,
        sigma_full=sigma_full,
    )
    _, _, total_imp_set, _ = set_imp
    reinforcing = total_imp_set / sum_individual if sum_individual > 0 else 0.0
    assert reinforcing > 1.0


def test_group_canceling_opposite_shapes():
    """Two entities mirrored across μ in a population where σ is dominated by
    OTHER entities (not the pair) — μ-shift cancels, σ-shift is small relative
    to individual impacts → reinforcing_factor < 1.0.

    Construction: 100 entities with normal(0, 1.5) noise + one pair at
    [+5, 0] / [-5, 0]. The pair contributes ~22% of total variance on dim 0,
    so removing them shrinks σ but not catastrophically. Individually each
    pulls μ in opposite directions (cancels at set level)."""
    from hypertopos.engine.geometry import (
        _compute_leave_one_out_impact,
        _compute_leave_set_out_impact,
    )

    rng = np.random.default_rng(42)
    shapes = rng.normal(0.0, 1.5, size=(100, 2)).astype(np.float64)
    shapes[0] = [5.0, 0.0]
    shapes[1] = [-5.0, 0.0]
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    mu_imp, sigma_imp, total_imp, _ = _compute_leave_one_out_impact(
        shapes, mu_full, sigma_full,
    )
    sum_individual = total_imp[0] + total_imp[1]

    set_imp = _compute_leave_set_out_impact(
        shapes=shapes,
        members_idx=np.array([0, 1]),
        mu_full=mu_full,
        sigma_full=sigma_full,
    )
    _, _, total_imp_set, _ = set_imp
    reinforcing = total_imp_set / sum_individual if sum_individual > 0 else 0.0
    assert reinforcing < 1.0


def test_group_per_dim_contributions_present():
    from hypertopos.engine.geometry import _compute_leave_set_out_impact

    shapes = np.zeros((20, 3), dtype=np.float64)
    shapes[0] = [0.0, 10.0, 0.0]
    shapes[1] = [0.0, 10.0, 0.0]
    mu_full = shapes.mean(axis=0)
    sigma_full = shapes.std(axis=0, ddof=0)
    _, _, _, contrib = _compute_leave_set_out_impact(
        shapes=shapes,
        members_idx=np.array([0, 1]),
        mu_full=mu_full,
        sigma_full=sigma_full,
    )
    assert contrib.shape == (3,)
    assert np.argmax(np.abs(contrib)) == 1


# ---------------------------------------------------------------------------
# Layer 1f — Orchestrator validation gates + happy paths
# ---------------------------------------------------------------------------

def _make_pattern_mock(
    pattern_type="anchor",
    n_dims=2,
    population_size=10,
    dim_labels=None,
):
    from unittest.mock import MagicMock
    pattern = MagicMock()
    pattern.pattern_type = pattern_type
    pattern.dim_labels = dim_labels if dim_labels is not None else [f"d{i}" for i in range(n_dims)]
    pattern.dimension_kinds = ["gaussian"] * n_dims
    pattern.mu = np.zeros(n_dims, dtype=np.float32)
    pattern.sigma_diag = np.ones(n_dims, dtype=np.float32)
    pattern.theta = np.full(n_dims, 2.0, dtype=np.float32)
    pattern.theta_norm = float(np.linalg.norm(pattern.theta))
    pattern.population_size = population_size
    pattern.relations = []
    pattern.prop_columns = []
    return pattern


def _make_navigator_with_geometry(shapes, pattern_id="p", pattern_type="anchor"):
    """Construct a GDSNavigator whose storage returns a fixed geometry table
    derived from shapes; pattern is configured to round-trip those shapes via
    delta = (shape - mu) / sigma."""
    from unittest.mock import MagicMock

    import pyarrow as pa
    from hypertopos.navigation.navigator import GDSNavigator

    N, D = shapes.shape
    pattern = _make_pattern_mock(pattern_type=pattern_type, n_dims=D, population_size=N)
    pattern.mu = np.zeros(D, dtype=np.float32)
    pattern.sigma_diag = np.ones(D, dtype=np.float32)
    deltas = (shapes - pattern.mu) / np.maximum(pattern.sigma_diag, 1e-2)
    delta_norms = np.linalg.norm(deltas, axis=1)
    pks = [f"E{i}" for i in range(N)]

    geo_table = pa.table({
        "primary_key": pks,
        "delta": [deltas[i].astype(np.float32).tolist() for i in range(N)],
        "delta_norm": pa.array(delta_norms.astype(np.float32)),
    })

    sphere = MagicMock()
    sphere.patterns = {pattern_id: pattern}

    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    storage.read_geometry.return_value = geo_table

    manifest = MagicMock()
    manifest.agent_id = "test"
    manifest.pattern_versions = {pattern_id: 1}

    contract = MagicMock()
    engine = MagicMock()
    nav = GDSNavigator(
        engine=engine, storage=storage, manifest=manifest, contract=contract,
    )
    return nav


def test_find_calibration_influencers_raises_on_event_pattern():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(10, 2)),
        pattern_type="event",
    )
    with pytest.raises(ValueError, match="pattern_type 'event'"):
        nav.find_calibration_influencers(pattern_id="p")


def test_find_calibration_influencers_raises_on_n_too_small():
    nav = _make_navigator_with_geometry(
        np.array([[0.0, 0.0]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="leave-one-out requires N >= 2"):
        nav.find_calibration_influencers(pattern_id="p")


def test_find_calibration_influencers_raises_on_invalid_threshold_pct():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(10, 2)),
    )
    with pytest.raises(ValueError, match="high_threshold_pct must be in"):
        nav.find_calibration_influencers(pattern_id="p", high_threshold_pct=0.0)
    with pytest.raises(ValueError, match="high_threshold_pct must be in"):
        nav.find_calibration_influencers(pattern_id="p", high_threshold_pct=100.0)


def test_find_calibration_influencers_raises_on_invalid_classify():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(10, 2)),
    )
    with pytest.raises(ValueError, match="classify must be one of"):
        nav.find_calibration_influencers(pattern_id="p", classify="weird")


def test_find_calibration_influencers_raises_on_invalid_top_n():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(10, 2)),
    )
    with pytest.raises(ValueError, match="top_n must be in"):
        nav.find_calibration_influencers(pattern_id="p", top_n=0)
    with pytest.raises(ValueError, match="top_n must be in"):
        nav.find_calibration_influencers(pattern_id="p", top_n=51)


def test_find_calibration_influencers_returns_report_with_cell_counts():
    """Smoke: classify='all', top_n=3 returns InfluenceReport with cell_counts
    summing to N and entries length ≤ top_n."""
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    report = nav.find_calibration_influencers(
        pattern_id="p", classify="all", top_n=3,
    )
    assert report.population_size == 20
    assert sum(report.cell_counts.values()) == 20
    assert set(report.cell_counts.keys()) == {
        "hidden", "distorter", "standard_anomaly", "normal",
    }
    assert len(report.entries) <= 3


def test_find_calibration_influencers_classify_filter_returns_only_filtered():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(1).normal(0, 1, size=(50, 3)),
    )
    report = nav.find_calibration_influencers(
        pattern_id="p", classify="hidden", top_n=10,
    )
    for e in report.entries:
        assert e.classification == "hidden"


def test_find_calibration_influencers_verbose_attaches_cascading():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(2).normal(0, 1, size=(30, 3)),
    )
    report_no = nav.find_calibration_influencers(
        pattern_id="p", classify="all", top_n=3, verbose=False,
    )
    report_yes = nav.find_calibration_influencers(
        pattern_id="p", classify="all", top_n=3, verbose=True,
    )
    for e in report_no.entries:
        assert e.cascading_flip_count is None
    for e in report_yes.entries:
        assert e.cascading_flip_count is not None
        assert isinstance(e.cascading_flip_count, int)


# ---------------------------------------------------------------------------
# Layer 1g — find_group_influence orchestrator
# ---------------------------------------------------------------------------

def test_find_group_influence_returns_per_group_reports():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    reports = nav.find_group_influence(
        pattern_id="p",
        groups=[["E0", "E1"], ["E2", "E3", "E4"]],
    )
    assert len(reports) == 2
    assert reports[0].member_count == 2
    assert reports[1].member_count == 3
    assert reports[0].members == ["E0", "E1"]
    assert reports[1].members == ["E2", "E3", "E4"]


def test_find_group_influence_raises_on_empty_groups():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    with pytest.raises(ValueError, match="groups list must be non-empty"):
        nav.find_group_influence(pattern_id="p", groups=[])


def test_find_group_influence_raises_on_single_member_group():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    with pytest.raises(ValueError, match="single-entity groups"):
        nav.find_group_influence(pattern_id="p", groups=[["E0"]])


def test_find_group_influence_raises_on_missing_entity():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    with pytest.raises(ValueError, match="not found in pattern"):
        nav.find_group_influence(pattern_id="p", groups=[["E0", "GHOST"]])


def test_find_group_influence_raises_on_duplicate_in_group():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    with pytest.raises(ValueError, match="duplicate entity"):
        nav.find_group_influence(pattern_id="p", groups=[["E0", "E0"]])


def test_find_group_influence_raises_on_event_pattern():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
        pattern_type="event",
    )
    with pytest.raises(ValueError, match="pattern_type 'event'"):
        nav.find_group_influence(pattern_id="p", groups=[["E0", "E1"]])


# ---------------------------------------------------------------------------
# Layer 1h — _attach_influence_fields_to_anomaly_entries helper
# (additive surface used by MCP find_anomalies; π5 navigator returns Polygon
# objects — additive happens at MCP→dict conversion layer)
# ---------------------------------------------------------------------------

def test_attach_influence_fields_returns_total_impact_and_classification():
    """Helper takes list[dict] of anomaly entries + pattern_id, returns
    enriched list[dict] with total_impact + classification per entry."""
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    fake_entries = [{"primary_key": "E0"}, {"primary_key": "E5"}, {"primary_key": "E19"}]
    enriched = nav._attach_influence_fields_to_anomaly_entries(fake_entries, "p")
    for entry in enriched:
        assert "total_impact" in entry
        assert "classification" in entry
        assert isinstance(entry["total_impact"], float)
        assert entry["classification"] in {
            "hidden", "distorter", "standard_anomaly", "normal",
        }


def test_attach_influence_fields_resolves_to_none_on_event_pattern():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
        pattern_type="event",
    )
    fake_entries = [{"primary_key": "E0"}]
    enriched = nav._attach_influence_fields_to_anomaly_entries(fake_entries, "p")
    assert enriched[0]["total_impact"] is None
    assert enriched[0]["classification"] is None


def test_attach_influence_fields_resolves_to_none_on_unknown_entity():
    nav = _make_navigator_with_geometry(
        np.random.default_rng(0).normal(0, 1, size=(20, 3)),
    )
    fake_entries = [{"primary_key": "GHOST"}]
    enriched = nav._attach_influence_fields_to_anomaly_entries(fake_entries, "p")
    assert enriched[0]["total_impact"] is None
    assert enriched[0]["classification"] is None
