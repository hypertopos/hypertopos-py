"""Chain-coherent investigative loop tests against the synthetic mini
fixture (no AML HI-small dependency).

Mirrors the smoke set covered by `test_chain_coherent_anomaly.py` (which
runs against AML HI-small when present) plus per-primitive correctness
tests for `find_chains_with_coherent_anomaly`, `anomaly_propagation_in_
chain`, `classify_chain_typology`, `extend_chain`, and the deep-dive
accessor `find_chains_for_entity` — all on a hand-crafted 8-account /
12-chain sphere where every cascade and dedup path is enumerable.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypertopos import HyperSphere

FIXTURES = Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"


@pytest.fixture(scope="module")
def synthetic_nav():
    """Open the synthetic chain sphere; generation handled by the
    session-scoped autouse fixture in conftest.py.
    """
    hs = HyperSphere.open(str(FIXTURES))
    return hs.session("synthetic-chain-test").navigator()


# Note on `top_dim` labels: `Pattern.dim_labels` is a *computed* property
# derived from the pattern's relations + event_dimensions + prop_columns +
# edge_dim_aggregations. A fully-built sphere like AML HI-small has those
# all populated, so navigator returns rich labels like
# "find_motif_structuring_max" for top_dim. The synthetic sphere has empty
# relations/event_dims by design (chain primitives don't need them), so
# dim_labels resolves to [] and the navigator correctly falls back to raw
# `dim_0`/`dim_1` indices. This is the documented behaviour, not a bug.


def test_find_chains_with_coherent_anomaly_returns_full_cascade(synthetic_nav):
    """CH-001 (A1->A2->A3->A4) must surface as a 4-hop coherent run."""
    out = synthetic_nav.find_chains_with_coherent_anomaly(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=3, max_results=10,
    )
    chain_ids = [c["chain_id"] for c in out["chains"]]
    assert "CH-001" in chain_ids
    cascade = next(c for c in out["chains"] if c["chain_id"] == "CH-001")
    assert cascade["run_length"] == 4
    assert cascade["run_keys"] == ["A1", "A2", "A3", "A4"]
    assert cascade["top_dim"] == "dim_0"


def test_find_chains_with_coherent_anomaly_skips_clean_chains(synthetic_nav):
    """CH-003 (B1->B2->B3) and CH-010 (B3->B4) must not appear."""
    out = synthetic_nav.find_chains_with_coherent_anomaly(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=2, max_results=20,
    )
    chain_ids = {c["chain_id"] for c in out["chains"]}
    assert "CH-003" not in chain_ids
    assert "CH-010" not in chain_ids


def test_find_chains_with_coherent_anomaly_min_hops_filters(synthetic_nav):
    """min_hops=4 returns only the n>=4 cascades."""
    out = synthetic_nav.find_chains_with_coherent_anomaly(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=4, max_results=10,
    )
    for c in out["chains"]:
        assert c["run_length"] >= 4


def test_find_chains_with_coherent_anomaly_diagnostics(synthetic_nav):
    out = synthetic_nav.find_chains_with_coherent_anomaly(
        "chain_pattern", anchor_pattern_id="account_pattern",
        min_hops=3, max_results=10,
    )
    diag = out["diagnostics"]
    assert diag["n_chains_total"] == 12
    assert diag["n_anomaly_entities"] == 4  # A1..A4
    assert diag["elapsed_ms"] >= 0


def test_anomaly_propagation_in_chain_traces_cascade(synthetic_nav):
    """Per-hop trace on CH-001 — every hop is anomalous on dim_0."""
    out = synthetic_nav.anomaly_propagation_in_chain(
        chain_id="CH-001", pattern_id="chain_pattern",
        anchor_pattern_id="account_pattern",
    )
    assert out["chain_id"] == "CH-001"
    hops = out["hops"]
    assert len(hops) == 4
    assert all(h["is_anomaly"] for h in hops)
    summary = out["summary"]
    assert summary["n_hops"] == 4
    assert summary["n_anomalous"] == 4
    assert summary["dominant_top_dim"] == "dim_0"


def test_anomaly_propagation_in_chain_partial_cascade(synthetic_nav):
    """CH-002 (A1->A2->B1) — first 2 hops anomalous, 3rd clean."""
    out = synthetic_nav.anomaly_propagation_in_chain(
        chain_id="CH-002", pattern_id="chain_pattern",
        anchor_pattern_id="account_pattern",
    )
    hops = out["hops"]
    assert len(hops) == 3
    assert [h["is_anomaly"] for h in hops] == [True, True, False]


def test_classify_chain_typology_full_cascade(synthetic_nav):
    """CH-001 — full cascade, leading position, no boundaries."""
    out = synthetic_nav.classify_chain_typology(
        chain_id="CH-001", pattern_id="chain_pattern",
        anchor_pattern_id="account_pattern",
    )
    typ = out["typology"]
    assert typ["run_length"] == 4
    assert typ["run_start_idx"] == 0
    assert typ["dominant_top_dim"] == "dim_0"


def test_classify_chain_typology_extension_signal(synthetic_nav):
    """CH-009 (A1->A2->B1->A3->A4) — has a forward-extension boundary
    after A4 (none, since it's last) but the typology call itself
    must succeed and surface a sane label.
    """
    out = synthetic_nav.classify_chain_typology(
        chain_id="CH-009", pattern_id="chain_pattern",
        anchor_pattern_id="account_pattern",
    )
    typ = out["typology"]
    assert typ["run_length"] >= 2
    assert typ["dominant_top_dim"] in {"dim_0", None}


def test_extend_chain_forward_returns_candidates(synthetic_nav):
    """CH-001 ends at A4. Forward extension finds A4's successors in
    OTHER chains:
    - CH-001: A4 last, no successor
    - CH-006: A4 at index 1, successor = A1
    - CH-009: A4 last, no successor
    - CH-011: A4 at index 3, successor = B4
    Expected exact set: {A1, B4}.
    """
    out = synthetic_nav.extend_chain(
        chain_id="CH-001", pattern_id="chain_pattern",
        anchor_pattern_id="account_pattern",
        direction="forward", max_results=10,
    )
    assert out["chain_id"] == "CH-001"
    assert out["direction"] == "forward"
    candidate_keys = {c["entity_key"] for c in out["candidates"]}
    assert candidate_keys == {"A1", "B4"}


def test_find_chains_for_entity_dedups_chain_pks(synthetic_nav):
    """A1 appears in 9 chains (CH-001/002/004/005/006/008/009/011/012);
    self-loop CH-008 lists A1 twice in chain_keys but the API contract
    surfaces it exactly once.
    """
    out = synthetic_nav.find_chains_for_entity(
        "A1", "chain_pattern", top_n=20,
    )
    chain_ids = [c["chain_id"] for c in out["chains"]]
    assert chain_ids.count("CH-008") == 1
    assert out["summary"]["total"] == 9


def test_find_chains_for_entity_clean_account(synthetic_nav):
    """B3 appears in CH-003 (B1->B2->B3) and CH-010 (B3->B4)."""
    out = synthetic_nav.find_chains_for_entity(
        "B3", "chain_pattern", top_n=20,
    )
    chain_ids = {c["chain_id"] for c in out["chains"]}
    assert chain_ids == {"CH-003", "CH-010"}
    assert out["summary"]["total"] == 2
    assert out["summary"]["anomalous"] == 0


def test_find_chains_for_entity_unknown_returns_empty(synthetic_nav):
    out = synthetic_nav.find_chains_for_entity(
        "NONEXISTENT", "chain_pattern", top_n=10,
    )
    assert out["chains"] == []
    assert out["summary"]["total"] == 0
