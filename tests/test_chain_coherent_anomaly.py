"""Navigator-level integration for find_chains_with_coherent_anomaly.

Uses the bundled AML HI-small sphere (chain anchor tx_chains_pattern over
account_pattern entities) since it's the only sphere with both a chain
anchor and an entity anchor with calibrated is_anomaly + delta vectors.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypertopos import HyperSphere

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AML_PATH = (
    PROJECT_ROOT / "benchmark" / "ibm-aml" / "hi_small_sphere"
    / "gds_aml_hi_small"
)


pytestmark = pytest.mark.skipif(
    not (AML_PATH / "_gds_meta" / "sphere.json").exists(),
    reason="AML HI-small sphere not built",
)


@pytest.fixture(scope="module")
def aml_nav():
    hs = HyperSphere.open(AML_PATH)
    return hs.session("chain-coherent-test").navigator()


def test_returns_dict_with_chains_block(aml_nav):
    out = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=3,
        max_results=10,
    )
    assert isinstance(out, dict)
    assert out["pattern_id"] == "tx_chains_pattern"
    assert out["anchor_pattern_id"] == "account_pattern"
    assert "chains" in out
    assert "n_results" in out
    assert "diagnostics" in out
    assert out["diagnostics"]["elapsed_ms"] > 0
    assert out["diagnostics"]["n_chains_total"] > 0


def test_chains_have_required_fields(aml_nav):
    out = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=2,
        max_results=5,
    )
    for c in out["chains"]:
        assert "chain_id" in c
        assert "run_start_idx" in c
        assert "run_length" in c
        assert "top_dim" in c
        assert "run_keys" in c
        assert "max_delta_norm" in c
        assert c["run_length"] >= 2
        assert len(c["run_keys"]) == c["run_length"]


def test_chains_sorted_by_run_length_desc(aml_nav):
    out = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=2,
        max_results=20,
    )
    if len(out["chains"]) >= 2:
        for a, b in zip(out["chains"], out["chains"][1:], strict=False):
            assert a["run_length"] >= b["run_length"]


def test_min_hops_below_2_raises(aml_nav):
    with pytest.raises(ValueError, match="min_hops"):
        aml_nav.find_chains_with_coherent_anomaly(
            "tx_chains_pattern",
            anchor_pattern_id="account_pattern",
            min_hops=1,
        )


def test_unknown_pattern_raises(aml_nav):
    with pytest.raises(Exception, match="pattern not found"):
        aml_nav.find_chains_with_coherent_anomaly(
            "nonexistent",
            anchor_pattern_id="account_pattern",
        )


def test_unknown_anchor_raises(aml_nav):
    with pytest.raises(Exception, match="anchor pattern not found"):
        aml_nav.find_chains_with_coherent_anomaly(
            "tx_chains_pattern",
            anchor_pattern_id="nonexistent",
        )


def test_event_pattern_id_raises(aml_nav):
    with pytest.raises(Exception, match="anchor"):
        aml_nav.find_chains_with_coherent_anomaly(
            "tx_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_max_results_zero_returns_empty_chains(aml_nav):
    out = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=2,
        max_results=0,
    )
    assert out["chains"] == []
    assert out["n_results"] == 0


def test_high_min_hops_returns_few_or_no_results(aml_nav):
    out = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=15,
        max_results=10,
    )
    for c in out["chains"]:
        assert c["run_length"] >= 15
