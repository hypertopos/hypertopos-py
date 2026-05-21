"""Tests for `_compute_dim_quality_warnings` — surfaces dead-dim
(sigma_diag near zero) and sparse-dim (median zero with rare nonzero)
build-time issues from the cached pattern state. Direct static-helper
tests; sphere_overview integration is exercised by the existing nav
test suite plus a fixture-level sanity check."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from hypertopos.model.sphere import Pattern, RelationDef
from hypertopos.navigation.navigator import GDSNavigator


def _make_pattern(
    *,
    sigma_diag: list[float],
    dim_percentiles: dict | None = None,
    relations: list[RelationDef] | None = None,
    heteroscedasticity_diagnostic: dict | None = None,
    dimension_kinds: list[str] | None = None,
    dim_normality_pvalues: dict[str, float] | None = None,
) -> Pattern:
    rels = relations or [
        RelationDef(line_id=f"line_{i}", direction="in", required=True)
        for i in range(len(sigma_diag))
    ]
    return Pattern(
        pattern_id="p_test",
        entity_type="test",
        pattern_type="anchor",
        relations=rels,
        mu=np.zeros(len(sigma_diag), dtype=np.float32),
        sigma_diag=np.asarray(sigma_diag, dtype=np.float32),
        theta=np.ones(len(sigma_diag), dtype=np.float32),
        population_size=1000,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        dim_percentiles=dim_percentiles,
        heteroscedasticity_diagnostic=heteroscedasticity_diagnostic,
        dimension_kinds=dimension_kinds,
        dim_normality_pvalues=dim_normality_pvalues,
    )


def test_no_warnings_on_healthy_pattern():
    """Pattern with non-zero sigma everywhere and no sparse dims → no
    dead_dim or sparse_dim warnings (other auditor types are out of
    scope for this test)."""
    pat = _make_pattern(
        sigma_diag=[0.5, 1.0, 0.8],
        dim_percentiles={
            "line_0": {"p25": 1.0, "p50": 5.0, "p75": 10.0, "p99": 50.0, "max": 100.0},
            "line_1": {"p25": 2.0, "p50": 8.0, "p75": 15.0, "p99": 40.0, "max": 80.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    scoped = [w for w in warnings if w["type"] in {"dead_dim", "sparse_dim"}]
    assert scoped == []


def test_dead_dim_flagged_when_sigma_zero():
    """sigma_diag < 1e-10 → dead_dim warning with the dim label."""
    pat = _make_pattern(
        sigma_diag=[0.5, 0.0, 1.0],
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    dead = [w for w in warnings if w["type"] == "dead_dim"]
    assert len(dead) == 1
    assert dead[0]["dim_label"] == "line_1"
    assert "sigma_diag" in dead[0]["reason"]
    assert "z-score" in dead[0]["advice"]


def test_dead_dim_flagged_for_subnormal_sigma():
    """Sigma below 1e-10 (not strict zero) also triggers dead-dim."""
    pat = _make_pattern(sigma_diag=[1.0, 1e-15, 1.0])
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    dead = [w for w in warnings if w["type"] == "dead_dim"]
    assert len(dead) == 1
    assert dead[0]["dim_label"] == "line_1"


def test_dead_dim_NOT_flagged_at_threshold_boundary():
    """Sigma at 1e-10 (the threshold) is NOT flagged — strict <
    comparison."""
    pat = _make_pattern(sigma_diag=[1.0, 1e-10, 1.0])
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "dead_dim" for w in warnings)


def test_sparse_dim_flagged_when_median_zero_p99_positive():
    """p50 == 0 AND p99 > 0 → sparse_dim warning."""
    pat = _make_pattern(
        sigma_diag=[1.0, 1.0],
        dim_percentiles={
            "active_dim": {"p25": 0.5, "p50": 1.0, "p75": 2.0, "p99": 5.0, "max": 10.0},
            "sparse_dim": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 5.0, "max": 100.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    sparse = [w for w in warnings if w["type"] == "sparse_dim"]
    assert len(sparse) == 1
    assert sparse[0]["dim_label"] == "sparse_dim"
    assert "median = 0" in sparse[0]["reason"]
    assert "Bregman" in sparse[0]["advice"]


def test_sparse_dim_not_flagged_when_p99_zero():
    """p50 == 0 AND p99 == 0 — fully zero, NOT flagged as sparse
    (would be dead-via-percentiles, and there's nothing to surface
    against). Keep the false-positive count low."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "all_zero": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 0.0, "max": 0.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "sparse_dim" for w in warnings)


def test_sparse_dim_fraction_zero_buckets():
    """Fraction-zero estimate from percentile pattern."""
    # ≥75% zeros: p25/p50/p75 all zero, p99 positive
    pat_75 = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "d75": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 10.0, "max": 100.0},
        },
    )
    w75 = GDSNavigator._compute_dim_quality_warnings(pat_75)
    assert ">=0.75" in w75[0]["reason"]

    # 50-75% zeros: p25/p50 zero, p75 positive
    pat_50 = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "d50": {"p25": 0.0, "p50": 0.0, "p75": 1.0, "p99": 10.0, "max": 100.0},
        },
    )
    w50 = GDSNavigator._compute_dim_quality_warnings(pat_50)
    assert "0.50-0.75" in w50[0]["reason"]

    # 25-50% zeros: p25 zero, p50 positive
    # (sparse signature requires p50 == 0 in the current rule, so this
    # case actually does NOT fire a sparse_dim warning — it's borderline
    # and the rule prefers no false positive.)
    pat_25 = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "d25": {"p25": 0.0, "p50": 0.5, "p75": 1.0, "p99": 10.0, "max": 100.0},
        },
    )
    w25 = GDSNavigator._compute_dim_quality_warnings(pat_25)
    assert all(w["type"] != "sparse_dim" for w in w25)


def test_dead_and_sparse_can_coexist():
    """A pattern with both classes flagged returns both warnings.
    Filters out other auditor types that may coincidentally fire on the
    fixture (e.g. dominant_dim_mass when one surviving-sigma dim drives
    the p99-tail mass)."""
    pat = _make_pattern(
        sigma_diag=[1.0, 0.0, 1.0],   # dim 1 is dead
        dim_percentiles={
            "line_0": {"p25": 1.0, "p50": 5.0, "p75": 10.0, "p99": 50.0, "max": 100.0},
            "line_2": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 5.0, "max": 100.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    scoped = {
        w["type"] for w in warnings if w["type"] in {"dead_dim", "sparse_dim"}
    }
    assert scoped == {"dead_dim", "sparse_dim"}


def test_warning_carries_advice_field():
    """Every warning must carry a dim_label, reason, and advice — the
    advice is the actionable surface for investigators."""
    pat = _make_pattern(
        sigma_diag=[0.0],
        dim_percentiles={
            "d": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 5.0, "max": 100.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    for w in warnings:
        assert "type" in w
        assert "dim_label" in w
        assert "reason" in w
        assert "advice" in w
        assert w["type"] in ("dead_dim", "sparse_dim")
        assert isinstance(w["advice"], str) and len(w["advice"]) > 20


def test_pattern_with_no_dim_percentiles_only_dead_warnings():
    """Spheres built before dim_percentiles cache populated → only
    dead-dim path fires (sigma_diag is always there)."""
    pat = _make_pattern(
        sigma_diag=[1.0, 0.0, 1.0],
        dim_percentiles=None,
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    types = {w["type"] for w in warnings}
    assert types == {"dead_dim"}


def test_pattern_with_no_sigma_returns_only_sparse_warnings():
    """Defensive — if sigma_diag is somehow None, dead-dim path is
    skipped silently (sparse-dim path still works)."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "d": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 5.0, "max": 100.0},
        },
    )
    pat.sigma_diag = None  # type: ignore[assignment]
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    types = {w["type"] for w in warnings}
    assert types == {"sparse_dim"}


def test_heteroscedasticity_flagged_when_p_below_threshold():
    """Persisted Levene p < 0.01 → heteroscedasticity warning carrying
    the grouping variable name in dim_label, the W statistic + p-value
    in the reason text, and the p-value/threshold in evidence."""
    pat = _make_pattern(
        sigma_diag=[1.0, 1.0],
        heteroscedasticity_diagnostic={
            "country": {
                "W_statistic": 25.4,
                "p_value": 1e-5,
                "k_groups": 3,
                "per_group_variance": {"DE": 1.0, "FR": 0.95, "US": 12.3},
                "per_group_n": {"DE": 500, "FR": 480, "US": 450},
                "skipped_groups_low_n": 0,
            },
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    het = [w for w in warnings if w["type"] == "heteroscedasticity"]
    assert len(het) == 1
    w = het[0]
    assert w["dim_label"] == "country"
    assert "Levene W=25.40" in w["reason"]
    assert "group_by=country" in w["reason"]
    assert "per-group θ" in w["advice"] or "per-group" in w["advice"]
    assert w["evidence_value"] == pytest.approx(1e-5)
    assert w["threshold"] == 0.01


def test_heteroscedasticity_not_flagged_above_threshold():
    """Persisted Levene p >= 0.01 → no heteroscedasticity warning; the
    global θ assumption is statistically tenable for this grouping."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        heteroscedasticity_diagnostic={
            "country": {
                "W_statistic": 0.42,
                "p_value": 0.65,
                "k_groups": 3,
                "per_group_variance": {"DE": 1.0, "FR": 1.05, "US": 0.98},
                "per_group_n": {"DE": 500, "FR": 480, "US": 450},
                "skipped_groups_low_n": 0,
            },
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "heteroscedasticity" for w in warnings)


def test_heteroscedasticity_skipped_when_p_value_none():
    """Fewer than two qualifying groups → p_value is None and no
    heteroscedasticity warning fires (the diagnostic itself surfaces
    `skipped_groups_low_n` for the agent to inspect via the persisted
    entry)."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        heteroscedasticity_diagnostic={
            "country": {
                "W_statistic": None,
                "p_value": None,
                "k_groups": 1,
                "per_group_variance": {"DE": 1.0},
                "per_group_n": {"DE": 500},
                "skipped_groups_low_n": 2,
            },
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "heteroscedasticity" for w in warnings)


def test_heteroscedasticity_skipped_when_p_value_nan():
    """Every qualifying group had zero residual variance — Levene
    returns NaN, the diagnostic persists it as None (via the engine
    primitive's finite-check). Defensive: a NaN that slips through (an
    older sphere built before the finite-check landed, say) must NOT
    fire a 'Levene W=nan p=nan' warning."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        heteroscedasticity_diagnostic={
            "country": {
                "W_statistic": float("nan"),
                "p_value": float("nan"),
                "k_groups": 3,
                "per_group_variance": {"DE": 0.0, "FR": 0.0, "US": 0.0},
                "per_group_n": {"DE": 500, "FR": 480, "US": 450},
                "skipped_groups_low_n": 0,
            },
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "heteroscedasticity" for w in warnings)
# ── non_normal_dim warning ────────────────────────────────────────────


def test_non_normal_dim_fires_when_gaussian_dim_fails_normality():
    """Gaussian dim with p < 0.01 → non_normal_dim warning surfaces."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "line_0": {"p25": 1.0, "p50": 5.0, "p75": 10.0, "p99": 50.0, "max": 100.0},
        },
        dimension_kinds=["gaussian"],
        dim_normality_pvalues={"line_0": 1e-8},
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    non_normal = [w for w in warnings if w["type"] == "non_normal_dim"]
    assert len(non_normal) == 1
    assert non_normal[0]["dim_label"] == "line_0"
    assert "heavy-tailed" in non_normal[0]["reason"]
    assert "log1p" in non_normal[0]["advice"]
    assert non_normal[0]["evidence_value"] == pytest.approx(1e-8, rel=1e-3)
    assert non_normal[0]["threshold"] == 0.01


def test_non_normal_dim_silent_when_p_value_above_threshold():
    """Gaussian dim with p >= 0.01 → no non_normal_dim warning."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "line_0": {"p25": 1.0, "p50": 5.0, "p75": 10.0, "p99": 50.0, "max": 100.0},
        },
        dimension_kinds=["gaussian"],
        dim_normality_pvalues={"line_0": 0.42},
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "non_normal_dim" for w in warnings)


def test_non_normal_dim_skipped_for_bernoulli_kind():
    """Bernoulli dim with persisted low p-value is silently ignored —
    the normality test is meaningless on binary kinds."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "line_0": {"p25": 0.0, "p50": 1.0, "p75": 1.0, "p99": 1.0, "max": 1.0},
        },
        dimension_kinds=["bernoulli"],
        dim_normality_pvalues={"line_0": 1e-30},
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "non_normal_dim" for w in warnings)


def test_non_normal_dim_skipped_for_poisson_kind():
    """Poisson (discrete count) dim is silently ignored — normality
    test does not apply to count distributions."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "line_0": {"p25": 1.0, "p50": 2.0, "p75": 4.0, "p99": 10.0, "max": 20.0},
        },
        dimension_kinds=["poisson"],
        dim_normality_pvalues={"line_0": 1e-30},
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "non_normal_dim" for w in warnings)


def test_non_normal_dim_suppressed_when_negative_space_fires():
    """A gaussian dim that ALSO has p50=0 will fire negative_space; the
    non_normal_dim warning is suppressed because the negative_space
    remediation (re-declare kind) supersedes it. Avoids double-flagging
    the same dim."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "line_0": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 5.0, "max": 100.0},
        },
        dimension_kinds=["gaussian"],
        dim_normality_pvalues={"line_0": 1e-30},
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    types = [w["type"] for w in warnings if w["dim_label"] == "line_0"]
    assert "negative_space" in types
    assert "non_normal_dim" not in types


def test_non_normal_dim_returns_empty_when_pvalues_absent():
    """Legacy pattern with no dim_normality_pvalues field → no
    non_normal_dim warning emitted, sphere is silent on the test."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "line_0": {"p25": 1.0, "p50": 5.0, "p75": 10.0, "p99": 50.0, "max": 100.0},
        },
        dimension_kinds=["gaussian"],
        dim_normality_pvalues=None,
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "non_normal_dim" for w in warnings)


def test_non_normal_dim_ignores_nan_p_value():
    """NaN p-values (insufficient data path) → silent, not crash."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dim_percentiles={
            "line_0": {"p25": 1.0, "p50": 5.0, "p75": 10.0, "p99": 50.0, "max": 100.0},
        },
        dimension_kinds=["gaussian"],
        dim_normality_pvalues={"line_0": float("nan")},
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "non_normal_dim" for w in warnings)
