"""Tests for `investigate_chain` — one-shot R9 orchestrator that wraps
trace + typology + shape-anomaly + forward/backward extension into a
single SAR-ready report.

Synthetic fixture: CH-001 (A1->A2->A3->A4) is the headline coherent
cascade on dim_0 — every R9 surface should fire on it. CH-003
(B1->B2->B3) is clean — every surface should be muted.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypertopos import HyperSphere

FIXTURES = Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"


@pytest.fixture(scope="module")
def synthetic_nav():
    hs = HyperSphere.open(str(FIXTURES))
    return hs.session("investigate-chain-test").navigator()


def test_returns_expected_top_level_keys(synthetic_nav):
    out = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    expected = {
        "chain_id", "pattern_id", "anchor_pattern_id",
        "trace", "typology", "shape_anomaly",
        "extension_forward", "extension_backward",
        "summary", "elapsed_ms",
    }
    assert expected.issubset(out.keys())


def test_each_step_wraps_in_ok_or_error(synthetic_nav):
    """Per-step blocks always carry an `ok` boolean — `data` when ok, `error`
    when not."""
    out = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    for key in (
        "trace", "typology", "shape_anomaly",
        "extension_forward", "extension_backward",
    ):
        block = out[key]
        assert "ok" in block, f"{key} missing 'ok' field"
        if block["ok"]:
            assert "data" in block, f"{key} ok but no 'data'"
        else:
            assert "error" in block, f"{key} not ok but no 'error'"


def test_strong_cascade_yields_high_strength(synthetic_nav):
    """CH-001 is the headline 4-hop coherent cascade — strength should be
    'strong' or 'moderate' depending on extension coverage."""
    out = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    summary = out["summary"]
    assert summary["investigation_strength"] in ("strong", "moderate")
    assert summary["score"] >= 2
    assert summary["recommended_action"] in (
        "escalate to SAR", "continue investigation",
    )
    assert "Coherent anomaly run of length" in summary["rationale"]


def test_summary_keys_present(synthetic_nav):
    out = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    summary = out["summary"]
    assert {"investigation_strength", "recommended_action",
            "score", "rationale"}.issubset(summary.keys())
    assert summary["investigation_strength"] in ("strong", "moderate", "weak")
    assert summary["recommended_action"] in (
        "escalate to SAR", "continue investigation",
        "false-positive candidate",
    )
    assert isinstance(summary["score"], int)
    assert 0 <= summary["score"] <= 5


def test_clean_chain_yields_weak_strength(synthetic_nav):
    """CH-003 has no anomalous run — strength must be 'weak'
    (chain-shape anomaly is no longer in the score)."""
    out = synthetic_nav.investigate_chain(
        "CH-003", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    summary = out["summary"]
    assert summary["investigation_strength"] == "weak"
    assert summary["score"] <= 1
    assert summary["recommended_action"] == "false-positive candidate"


def test_unknown_chain_id_keeps_partial_steps(synthetic_nav):
    """Unknown chain_id: per-step blocks fail individually but the call
    still returns a structured report."""
    out = synthetic_nav.investigate_chain(
        "CHAIN-DOES-NOT-EXIST", "chain_pattern",
        anchor_pattern_id="account_pattern",
    )
    # All the per-step blocks should report ok=False (no chain → nothing
    # to trace / classify / extend / look up).
    assert out["chain_id"] == "CHAIN-DOES-NOT-EXIST"
    failure_blocks = [
        b for b in (
            out["trace"], out["typology"], out["shape_anomaly"],
            out["extension_forward"], out["extension_backward"],
        )
        if not b.get("ok")
    ]
    assert len(failure_blocks) >= 1
    # Whole call must not abort — summary is still returned.
    assert out["summary"]["investigation_strength"] == "weak"


def test_unknown_chain_pattern_raises(synthetic_nav):
    from hypertopos.navigation.navigator import GDSNavigationError

    with pytest.raises(GDSNavigationError, match="pattern not found"):
        synthetic_nav.investigate_chain(
            "CH-001", "nonexistent_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_unknown_anchor_pattern_raises(synthetic_nav):
    from hypertopos.navigation.navigator import GDSNavigationError

    with pytest.raises(GDSNavigationError, match="anchor pattern not found"):
        synthetic_nav.investigate_chain(
            "CH-001", "chain_pattern",
            anchor_pattern_id="nonexistent_anchor",
        )


def test_extension_max_results_propagates(synthetic_nav):
    """extension_max_results caps the candidate list when extension data
    is present."""
    out = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
        extension_max_results=1,
    )
    for key in ("extension_forward", "extension_backward"):
        block = out[key]
        if block.get("ok"):
            assert len(block["data"]["candidates"]) <= 1


def test_elapsed_ms_is_positive(synthetic_nav):
    out = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert out["elapsed_ms"] > 0


def test_score_signal_parity_with_per_step_blocks(synthetic_nav):
    """The summary's score must equal the sum of the four 0/1
    chain-composition signals derived from the per-step blocks.
    chain_shape_anomaly is intentionally NOT scored — see docstring."""
    out = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )

    expected = 0
    if out["trace"].get("ok"):
        if int(out["trace"]["data"]["summary"].get(
            "max_run_length_same_top_dim", 0,
        )) >= 3:
            expected += 1
    if out["typology"].get("ok"):
        if out["typology"]["data"]["typology"].get(
            "position_in_chain", "no-run",
        ) != "no-run":
            expected += 1
    if out["extension_forward"].get("ok"):
        if int(out["extension_forward"]["data"]["summary"].get(
            "n_anomalous_candidates", 0,
        )) >= 1:
            expected += 1
    if out["extension_backward"].get("ok"):
        if int(out["extension_backward"]["data"]["summary"].get(
            "n_anomalous_candidates", 0,
        )) >= 1:
            expected += 1

    assert out["summary"]["score"] == expected
    assert 0 <= out["summary"]["score"] <= 4


def test_shape_anomaly_does_not_increment_score(synthetic_nav):
    """Locks the design intent: even if shape_anomaly.is_anomaly fires,
    it does NOT contribute to the score. R9 sweet spot stays at strong
    (composition-anomalous, shape-normal) without needing shape
    agreement."""
    # Build mock blocks that fire ALL four composition signals + the
    # shape signal. Score must be 4 (the four composition signals),
    # NOT 5.
    from hypertopos.navigation.navigator import GDSNavigator
    summary = GDSNavigator._derive_investigation_summary(
        trace_block={"ok": True, "data": {"summary": {
            "max_run_length_same_top_dim": 4,
            "dominant_top_dim": "x",
        }}},
        typology_block={"ok": True, "data": {"typology": {
            "position_in_chain": "leading", "shape": "monotone-rising",
        }}},
        shape_anomaly={"ok": True, "data": {
            "is_anomaly": True, "delta_rank_pct": 99.5,
        }},
        extension_forward={"ok": True, "data": {"summary": {
            "n_anomalous_candidates": 2,
        }}},
        extension_backward={"ok": True, "data": {"summary": {
            "n_anomalous_candidates": 1,
        }}},
    )
    assert summary["score"] == 4
    assert summary["investigation_strength"] == "strong"
    # Shape evidence must STILL surface in the rationale.
    assert "Chain-shape anomaly also flags" in summary["rationale"]


def test_r9_sweet_spot_reaches_strong_without_shape(synthetic_nav):
    """The textbook R9 hit: composition-anomalous + shape-normal +
    one-sided extension. With four composition signals where 3 fire,
    score=3 → strong → escalate to SAR. Locks against regression to
    the prior 5-signal scoring that capped this case at moderate."""
    from hypertopos.navigation.navigator import GDSNavigator
    summary = GDSNavigator._derive_investigation_summary(
        trace_block={"ok": True, "data": {"summary": {
            "max_run_length_same_top_dim": 4,
            "dominant_top_dim": "find_motif_structuring_max",
        }}},
        typology_block={"ok": True, "data": {"typology": {
            "position_in_chain": "leading", "shape": "peak-in-middle",
        }}},
        shape_anomaly={"ok": True, "data": {
            "is_anomaly": False,  # shape-normal — the R9 sweet spot
            "delta_rank_pct": 92.0,
        }},
        extension_forward={"ok": True, "data": {"summary": {
            "n_anomalous_candidates": 1,
        }}},
        extension_backward={"ok": True, "data": {"summary": {
            "n_anomalous_candidates": 0,
        }}},  # one-sided extension — boundary chain
    )
    assert summary["score"] == 3
    assert summary["investigation_strength"] == "strong"
    assert summary["recommended_action"] == "escalate to SAR"
