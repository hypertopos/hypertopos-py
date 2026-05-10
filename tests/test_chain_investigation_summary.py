"""Tests for `chain_investigation_summary` — pre-investigation triage
primitive. Aggregates one find_chains_with_coherent_anomaly sweep + a
chain-pattern geometry scan into a population-level diagnostic.

Synthetic fixture has 12 chains (CH-001..CH-012) over 8 accounts; CH-001
A1->A2->A3->A4 is the headline coherent cascade on dim_0, CH-003 / CH-010
are clean. See test_chain_coherent_synthetic.py for fixture topology.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypertopos import HyperSphere

FIXTURES = Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"


@pytest.fixture(scope="module")
def synthetic_nav():
    hs = HyperSphere.open(str(FIXTURES))
    return hs.session("chain-summary-test").navigator()


def test_returns_expected_top_level_keys(synthetic_nav):
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
    )
    expected = {
        "chain_pattern_id", "anchor_pattern_id",
        "n_chains_total", "n_chains_with_coherent_anomaly_run",
        "coherent_run_rate",
        "n_chains_with_shape_anomaly", "shape_anomaly_rate",
        "cross_pattern_overlap",
        "top_dims_in_coherent_runs",
        "run_length_distribution",
        "recommended_min_hops",
        "elapsed_ms",
    }
    assert expected.issubset(out.keys())


def test_pattern_ids_round_trip(synthetic_nav):
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert out["chain_pattern_id"] == "chain_pattern"
    assert out["anchor_pattern_id"] == "account_pattern"


def test_n_chains_total_matches_underlying_sweep(synthetic_nav):
    """`n_chains_total` agrees with the diagnostics block from
    `find_chains_with_coherent_anomaly` on the same inputs."""
    sweep = synthetic_nav.find_chains_with_coherent_anomaly(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2, max_results=10000,
    )
    summary = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2,
    )
    assert summary["n_chains_total"] == sweep["diagnostics"]["n_chains_total"]


def test_coherent_run_rate_is_fraction(synthetic_nav):
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2,
    )
    rate = out["coherent_run_rate"]
    assert 0.0 <= rate <= 1.0
    if out["n_chains_total"] > 0:
        expected = round(
            out["n_chains_with_coherent_anomaly_run"] / out["n_chains_total"],
            6,
        )
        assert rate == expected


def test_top_dims_sorted_descending_by_count(synthetic_nav):
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2,
    )
    counts = [t["count"] for t in out["top_dims_in_coherent_runs"]]
    assert counts == sorted(counts, reverse=True)


def test_top_dims_capped_at_ten(synthetic_nav):
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2,
    )
    assert len(out["top_dims_in_coherent_runs"]) <= 10


def test_run_length_distribution_shape(synthetic_nav):
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2,
    )
    rd = out["run_length_distribution"]
    assert {"min", "p50", "p90", "max", "mean"}.issubset(rd.keys())
    if out["n_chains_with_coherent_anomaly_run"] > 0:
        assert rd["min"] >= 2  # min_hops=2 floor
        assert rd["max"] >= rd["p90"] >= rd["p50"] >= rd["min"]


def test_cross_pattern_overlap_partition_consistent(synthetic_nav):
    """n_both + n_coherent_only == total coherent runs;
    n_both + n_shape_only == total shape anomalies."""
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2,
    )
    overlap = out["cross_pattern_overlap"]
    assert overlap["n_both"] + overlap["n_coherent_only"] == \
        out["n_chains_with_coherent_anomaly_run"]
    assert overlap["n_both"] + overlap["n_shape_only"] == \
        out["n_chains_with_shape_anomaly"]
    assert 0.0 <= overlap["jaccard"] <= 1.0


def test_recommended_min_hops_floor_at_input(synthetic_nav):
    """recommended_min_hops never drops below the input min_hops floor."""
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=3,
    )
    assert out["recommended_min_hops"] >= 3


def test_min_hops_below_two_raises(synthetic_nav):
    with pytest.raises(ValueError, match="min_hops must be >= 2"):
        synthetic_nav.chain_investigation_summary(
            "chain_pattern", anchor_pattern_id="account_pattern",
            min_hops=1,
        )


def test_max_runs_negative_raises(synthetic_nav):
    with pytest.raises(ValueError, match="max_runs must be >= 0"):
        synthetic_nav.chain_investigation_summary(
            "chain_pattern", anchor_pattern_id="account_pattern",
            max_runs=-1,
        )


def test_unknown_chain_pattern_raises(synthetic_nav):
    from hypertopos.navigation.navigator import GDSNavigationError

    with pytest.raises(GDSNavigationError, match="pattern not found"):
        synthetic_nav.chain_investigation_summary(
            "nonexistent_chain_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_unknown_anchor_pattern_raises(synthetic_nav):
    from hypertopos.navigation.navigator import GDSNavigationError

    with pytest.raises(GDSNavigationError, match="anchor pattern not found"):
        synthetic_nav.chain_investigation_summary(
            "chain_pattern",
            anchor_pattern_id="nonexistent_anchor_pattern",
        )


def test_elapsed_ms_is_positive(synthetic_nav):
    out = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert out["elapsed_ms"] > 0


def test_aggregates_unbiased_under_truncation(synthetic_nav):
    """Regression: aggregates must reflect the FULL coherent population,
    not the top-K slice of the underlying find_chains_with_coherent_anomaly
    sweep. With max_runs=1 the sweep returns at most 1 chain in chains[],
    but n_chains_with_coherent_anomaly_run / coherent_run_rate /
    top_dims_in_coherent_runs / run_length_distribution / cross_pattern_overlap
    must equal what an unbounded call sees."""
    full = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2, max_runs=10000,
    )
    truncated = synthetic_nav.chain_investigation_summary(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2, max_runs=1,
    )
    assert truncated["n_chains_with_coherent_anomaly_run"] == \
        full["n_chains_with_coherent_anomaly_run"]
    assert truncated["coherent_run_rate"] == full["coherent_run_rate"]
    assert truncated["top_dims_in_coherent_runs"] == \
        full["top_dims_in_coherent_runs"]
    assert truncated["run_length_distribution"] == \
        full["run_length_distribution"]
    assert truncated["cross_pattern_overlap"] == \
        full["cross_pattern_overlap"]
    assert truncated["recommended_min_hops"] == full["recommended_min_hops"]
