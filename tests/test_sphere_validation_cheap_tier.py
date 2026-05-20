"""Tests for the cheap-tier pattern-level sphere-validation auditors:
``dominant_dim_mass`` and ``negative_space``. Both ride along on
``_compute_dim_quality_warnings`` so the existing dead_dim / sparse_dim
warnings continue to fire unchanged when the new auditors apply."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from hypertopos.model.sphere import Pattern, RelationDef
from hypertopos.navigation.navigator import GDSNavigator


def _make_pattern(
    *,
    dim_labels: list[str],
    mu: list[float] | None = None,
    sigma_diag: list[float] | None = None,
    dim_percentiles: dict | None = None,
    dimension_kinds: list[str] | None = None,
) -> Pattern:
    """Build a minimal Pattern whose ``dim_labels`` property matches the
    supplied label list. Each label becomes a relation's ``line_id`` so
    the property returns it verbatim."""
    relations = [
        RelationDef(line_id=label, direction="in", required=True)
        for label in dim_labels
    ]
    n = len(dim_labels)
    mu_arr = (
        np.asarray(mu, dtype=np.float32) if mu is not None
        else np.zeros(n, dtype=np.float32)
    )
    sigma_arr = (
        np.asarray(sigma_diag, dtype=np.float32) if sigma_diag is not None
        else np.ones(n, dtype=np.float32)
    )
    return Pattern(
        pattern_id="p_test",
        entity_type="test",
        pattern_type="anchor",
        relations=relations,
        mu=mu_arr,
        sigma_diag=sigma_arr,
        theta=np.ones(n, dtype=np.float32),
        population_size=1000,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        dim_percentiles=dim_percentiles,
        dimension_kinds=dimension_kinds,
    )


# ---------------------------------------------------------------------------
# dominant_dim_mass
# ---------------------------------------------------------------------------


def test_dominant_dim_mass_fires_when_one_dim_dominates_p99_tail():
    """One dim with z_p99=100 versus two dims with z_p99=1.5 → the
    dominant dim's mass share is overwhelmingly above the 0.7
    threshold."""
    pat = _make_pattern(
        dim_labels=["a", "b", "c"],
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0],
        dim_percentiles={
            "a": {"p25": 1.0, "p50": 5.0, "p75": 20.0, "p99": 100.0, "max": 200.0},
            "b": {"p25": 0.2, "p50": 0.5, "p75": 1.0, "p99": 1.5, "max": 3.0},
            "c": {"p25": 0.2, "p50": 0.5, "p75": 1.0, "p99": 1.5, "max": 3.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    dom = [w for w in warnings if w["type"] == "dominant_dim_mass"]
    assert len(dom) == 1
    assert dom[0]["dim_label"] == "a"
    assert dom[0]["evidence_value"] > 0.95
    assert dom[0]["threshold"] == 0.7
    assert "single-dim-driven" in dom[0]["advice"]


def test_dominant_dim_mass_no_fire_when_uniform_p99():
    """Three dims with identical p99/mu/sigma → equal mass shares
    (~0.33) → no fire."""
    pat = _make_pattern(
        dim_labels=["a", "b", "c"],
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0],
        dim_percentiles={
            "a": {"p25": 0.5, "p50": 1.0, "p75": 1.5, "p99": 2.0, "max": 4.0},
            "b": {"p25": 0.5, "p50": 1.0, "p75": 1.5, "p99": 2.0, "max": 4.0},
            "c": {"p25": 0.5, "p50": 1.0, "p75": 1.5, "p99": 2.0, "max": 4.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "dominant_dim_mass" for w in warnings)


def test_dominant_dim_mass_skips_aggregated_edge_dims_without_percentiles():
    """When ``dim_percentiles`` lacks coverage for a dim (e.g.
    aggregated edge dim) it is excluded from the tail-mass loop rather
    than crashing with an AttributeError on ``None.get``."""
    pat = _make_pattern(
        dim_labels=["a", "b", "aggr_c"],
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0],
        dim_percentiles={
            "a": {"p25": 0.5, "p50": 1.0, "p75": 1.5, "p99": 2.0, "max": 4.0},
            "b": {"p25": 0.5, "p50": 1.0, "p75": 1.5, "p99": 2.0, "max": 4.0},
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    # No fire (uniform across the two surviving dims) and no exception.
    assert all(w["type"] != "dominant_dim_mass" for w in warnings)


def test_dominant_dim_mass_no_fire_when_sigma_diag_missing():
    """``sigma_diag is None`` → auditor returns None silently, leaving
    the rest of the warning list intact."""
    pat = _make_pattern(
        dim_labels=["a", "b"],
        mu=[0.0, 0.0],
        sigma_diag=[1.0, 1.0],
        dim_percentiles={
            "a": {"p25": 1.0, "p50": 5.0, "p75": 20.0, "p99": 100.0, "max": 200.0},
            "b": {"p25": 0.2, "p50": 0.5, "p75": 1.0, "p99": 1.5, "max": 3.0},
        },
    )
    pat.sigma_diag = None  # type: ignore[assignment]
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "dominant_dim_mass" for w in warnings)


# ---------------------------------------------------------------------------
# negative_space
# ---------------------------------------------------------------------------


def test_negative_space_fires_on_gaussian_kind_with_zero_p50():
    """gaussian-declared dim with p50==0 and p99==0 → fire with the
    ``all_zero`` fraction estimate."""
    pat = _make_pattern(
        dim_labels=["x"],
        sigma_diag=[1.0],
        dim_percentiles={
            "x": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 0.0, "max": 0.0},
        },
        dimension_kinds=["gaussian"],
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    ns = [w for w in warnings if w["type"] == "negative_space"]
    assert len(ns) == 1
    assert ns[0]["dim_label"] == "x"
    # Categorical fraction-zero estimate lands in the reason text so the
    # warning carries the same {type, dim_label, reason, advice} shape as
    # the existing dead_dim / sparse_dim warnings — no evidence_value /
    # threshold keys when the evidence is non-numeric.
    assert "all_zero" in ns[0]["reason"]
    assert "evidence_value" not in ns[0]
    assert "threshold" not in ns[0]


def test_negative_space_no_fire_on_bernoulli_kind_with_zero_p50():
    """Same percentile signature but ``kind='bernoulli'`` → expected
    sparsity, no fire."""
    pat = _make_pattern(
        dim_labels=["x"],
        sigma_diag=[1.0],
        dim_percentiles={
            "x": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 0.0, "max": 0.0},
        },
        dimension_kinds=["bernoulli"],
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "negative_space" for w in warnings)


def test_negative_space_no_fire_on_gaussian_with_nonzero_p50():
    """gaussian-declared dim with positive p50 → the empirical
    distribution is not zero-centered, no fire."""
    pat = _make_pattern(
        dim_labels=["x"],
        sigma_diag=[1.0],
        dim_percentiles={
            "x": {"p25": 0.1, "p50": 0.5, "p75": 1.0, "p99": 5.0, "max": 10.0},
        },
        dimension_kinds=["gaussian"],
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "negative_space" for w in warnings)


# ---------------------------------------------------------------------------
# regression — existing classes still fire next to the new ones
# ---------------------------------------------------------------------------


def test_existing_dead_dim_and_sparse_dim_still_fire_unchanged():
    """A pattern that triggers dead_dim + sparse_dim + at least one of
    the new auditors must report all four warning types in a single
    call — regression guard against shadowing the existing rules."""
    pat = _make_pattern(
        dim_labels=["dead", "sparse_or_dom", "uniform"],
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 0.0, 1.0],  # dim 1 is dead
        dim_percentiles={
            # sparse + drives the p99-tail mass on the surviving-sigma
            # dims (dead dim is excluded from the dominant-mass denom).
            "sparse_or_dom": {
                "p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 100.0, "max": 200.0,
            },
            "uniform": {
                "p25": 0.5, "p50": 1.0, "p75": 1.5, "p99": 2.0, "max": 4.0,
            },
        },
        dimension_kinds=["gaussian", "gaussian", "gaussian"],
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    types = {w["type"] for w in warnings}
    assert "dead_dim" in types
    assert "sparse_dim" in types
    # sparse_or_dom is both sparse (p50=0, p99=100) and the dominant
    # tail-mass contributor — z_p99=100 vs z_p99=2 on the uniform dim,
    # share ~ 99.96 % which crosses the 0.7 threshold. It is also a
    # gaussian-kind zero-median dim so negative_space fires too. All
    # three new-or-old surfaces must be present.
    assert "dominant_dim_mass" in types
    assert "negative_space" in types
