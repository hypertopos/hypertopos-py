"""Tests for `generate_sar_rationale` — template-based SAR narrative
composition over the structured `investigate_chain` evidence dict.

Synthetic fixture: CH-001 (A1->A2->A3->A4) is the headline coherent
cascade — narrative should fire all 5 paragraphs (typology, trace,
extension, shape, summary). CH-003 is clean — narrative should be
shorter and recommend false-positive.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypertopos import HyperSphere
from hypertopos.navigation.navigator import GDSNavigator

FIXTURES = Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"


@pytest.fixture(scope="module")
def synthetic_nav():
    hs = HyperSphere.open(str(FIXTURES))
    return hs.session("sar-rationale-test").navigator()


def test_returns_expected_top_level_keys(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    expected = {
        "chain_id", "pattern_id", "anchor_pattern_id",
        "sar_narrative", "evidence_anchors",
        "regulatory_template_hint", "confidence", "elapsed_ms",
    }
    assert expected.issubset(out.keys())


def test_pattern_ids_round_trip(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert out["chain_id"] == "CH-001"
    assert out["pattern_id"] == "chain_pattern"
    assert out["anchor_pattern_id"] == "account_pattern"


def test_narrative_is_paragraph_separated(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    paragraphs = out["sar_narrative"].split("\n\n")
    # CH-001 should fire at least 4 paragraphs (typology + trace +
    # shape + summary). Boundary extension is conditional on candidates.
    assert len(paragraphs) >= 4
    # Every paragraph must be non-empty.
    assert all(p.strip() for p in paragraphs)


def test_narrative_mentions_chain_id(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert "CH-001" in out["sar_narrative"]


def test_evidence_anchors_carry_per_step_pointers(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    anchors = out["evidence_anchors"]
    assert {"typology_axes", "per_hop_trace", "boundary_extensions",
            "chain_shape_anomaly", "summary"}.issubset(anchors.keys())
    # At minimum, summary should always be populated (it's a derived block).
    assert anchors["summary"] is not None
    assert "investigation_strength" in anchors["summary"]


def test_default_regulatory_template_hint(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert out["regulatory_template_hint"] == "FinCEN SAR"


def test_custom_regulatory_template_round_trips(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
        regulatory_template="EU AMLR Annex II",
    )
    assert out["regulatory_template_hint"] == "EU AMLR Annex II"


def test_evidence_passthrough_avoids_re_running_loop(synthetic_nav):
    """Pre-supplied evidence dict is used verbatim (no R9 re-run);
    the resulting narrative is identical to the auto-run path."""
    inv = synthetic_nav.investigate_chain(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    out_auto = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    out_supplied = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
        evidence=inv,
    )
    # Narrative content is deterministic given the evidence.
    assert out_auto["sar_narrative"] == out_supplied["sar_narrative"]
    assert out_auto["evidence_anchors"] == out_supplied["evidence_anchors"]
    assert out_auto["confidence"] == out_supplied["confidence"]


def test_clean_chain_yields_low_confidence(synthetic_nav):
    """CH-003 has no anomalous run — confidence must be 'low'."""
    out = synthetic_nav.generate_sar_rationale(
        "CH-003", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert out["confidence"] == "low"
    # Recommended action must propagate into the narrative.
    assert "false-positive candidate" in out["sar_narrative"]


def test_narrative_uses_honest_language(synthetic_nav):
    """Evidence-language discipline: narrative must NOT use the word
    'confirms' (suggests certainty) — uses 'indicates' / 'shows' /
    'corroborating' instead."""
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    narrative_lower = out["sar_narrative"].lower()
    assert "confirms" not in narrative_lower
    assert "is confirmed" not in narrative_lower


def test_unknown_chain_id_returns_low_confidence(synthetic_nav):
    """Unknown chain_id: confidence must drop to low and chain_id
    surfaces in the narrative for traceability."""
    out = synthetic_nav.generate_sar_rationale(
        "CHAIN-DOES-NOT-EXIST", "chain_pattern",
        anchor_pattern_id="account_pattern",
    )
    assert out["confidence"] == "low"
    assert "CHAIN-DOES-NOT-EXIST" in out["sar_narrative"]


def test_untriaged_guard_when_all_r9_surfaces_fail():
    """When ALL 5 R9 surfaces return ok=False (chain truly missing
    from storage, sphere broken, etc.), the narrative MUST surface
    the chain as 'untriaged' — NOT 'false-positive candidate'. The
    latter would read as 'we evaluated and found it clear', which in
    a SAR context is the worst silent-error class (investigator
    pastes 'cleared' on a chain that was never actually checked).

    Direct composer test — bypasses storage so the all-failed case
    can be exercised regardless of fixture behaviour for a missing
    key on the underlying Lance read."""
    all_failed_evidence = {
        "trace": {"ok": False, "error": "GDSNavigationError: ..."},
        "typology": {"ok": False, "error": "GDSNavigationError: ..."},
        "shape_anomaly": {"ok": False, "error": "GDSNavigationError: ..."},
        "extension_forward": {"ok": False, "error": "GDSNavigationError: ..."},
        "extension_backward": {"ok": False, "error": "GDSNavigationError: ..."},
        "summary": {
            "investigation_strength": "weak",
            "recommended_action": "false-positive candidate",
            "score": 0,
            "rationale": "No coherent investigative signal across the R9 surfaces.",
        },
    }
    narrative, anchors = GDSNavigator._compose_sar_narrative(
        "CHAIN-XYZ", "chain_pattern", "account_pattern", all_failed_evidence,
    )
    # Untriaged guard must fire.
    assert "false-positive candidate" not in narrative
    assert "untriaged" in narrative
    assert "could not complete" in narrative
    assert "not as cleared" in narrative
    assert anchors["summary"] is None
    # Per-step anchors must also all be None.
    for k in ("typology_axes", "per_hop_trace", "chain_shape_anomaly"):
        assert anchors[k] is None, f"{k} should be None, got {anchors[k]!r}"
    assert anchors["boundary_extensions"]["forward"] is None
    assert anchors["boundary_extensions"]["backward"] is None


def test_unknown_pattern_raises(synthetic_nav):
    from hypertopos.navigation.navigator import GDSNavigationError

    with pytest.raises(GDSNavigationError, match="pattern not found"):
        synthetic_nav.generate_sar_rationale(
            "CH-001", "nonexistent_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_unknown_anchor_pattern_raises(synthetic_nav):
    from hypertopos.navigation.navigator import GDSNavigationError

    with pytest.raises(GDSNavigationError, match="anchor pattern not found"):
        synthetic_nav.generate_sar_rationale(
            "CH-001", "chain_pattern",
            anchor_pattern_id="nonexistent_anchor",
        )


def test_elapsed_ms_is_positive(synthetic_nav):
    out = synthetic_nav.generate_sar_rationale(
        "CH-001", "chain_pattern", anchor_pattern_id="account_pattern",
    )
    assert out["elapsed_ms"] > 0


def test_confidence_derivation_full_ladder():
    """Direct test of the confidence-derivation helper. Locks the
    full ladder including the strong+4 fix that prevents the
    contradiction 'confidence=low, recommended_action=escalate to SAR'."""
    full_strong = {
        "trace": {"ok": True, "data": {}},
        "typology": {"ok": True, "data": {}},
        "shape_anomaly": {"ok": True, "data": {}},
        "extension_forward": {"ok": True, "data": {}},
        "extension_backward": {"ok": True, "data": {}},
        "summary": {"investigation_strength": "strong"},
    }
    # strong + 5 → high
    assert GDSNavigator._derive_sar_confidence(full_strong) == "high"

    # strong + 4 → moderate (the cliff fix — was low, now moderate to
    # avoid contradiction with recommended_action=escalate)
    one_failed = dict(full_strong)
    one_failed["extension_backward"] = {"ok": False, "error": "x"}
    assert GDSNavigator._derive_sar_confidence(one_failed) == "moderate"

    # strong + 3 → low (truly degraded evidence)
    two_failed = dict(one_failed)
    two_failed["extension_forward"] = {"ok": False, "error": "x"}
    assert GDSNavigator._derive_sar_confidence(two_failed) == "low"

    # moderate + 5 → moderate
    moderate_full = dict(full_strong)
    moderate_full["summary"] = {"investigation_strength": "moderate"}
    assert GDSNavigator._derive_sar_confidence(moderate_full) == "moderate"

    # moderate + 4 → moderate
    moderate_4 = dict(moderate_full)
    moderate_4["extension_backward"] = {"ok": False, "error": "x"}
    assert GDSNavigator._derive_sar_confidence(moderate_4) == "moderate"

    # moderate + 3 → low
    moderate_3 = dict(moderate_4)
    moderate_3["extension_forward"] = {"ok": False, "error": "x"}
    assert GDSNavigator._derive_sar_confidence(moderate_3) == "low"

    # weak → always low regardless of completeness
    weak = dict(full_strong)
    weak["summary"] = {"investigation_strength": "weak"}
    assert GDSNavigator._derive_sar_confidence(weak) == "low"


def test_no_contradiction_strong_plus_one_failed_surface():
    """Regression: strong strength + 4 ok surfaces must NOT yield
    confidence=low while recommended_action=escalate to SAR. That
    contradiction in the same response would mislead investigators
    triaging the SAR draft."""
    evidence = {
        "trace": {"ok": True, "data": {"summary": {
            "max_run_length_same_top_dim": 4,
            "dominant_top_dim": "x",
        }}},
        "typology": {"ok": True, "data": {"typology": {
            "position_in_chain": "leading",
            "shape": "monotone-rising",
        }}},
        "shape_anomaly": {"ok": True, "data": {
            "is_anomaly": False, "delta_rank_pct": 50.0,
        }},
        "extension_forward": {"ok": True, "data": {"summary": {
            "n_anomalous_candidates": 1, "n_candidates": 3,
        }, "boundary_key": "X", "boundary_position": "run-end"}},
        "extension_backward": {"ok": False, "error": "no run boundary"},
        "summary": {
            "investigation_strength": "strong",
            "recommended_action": "escalate to SAR",
            "score": 3,
            "rationale": "x",
        },
    }
    assert GDSNavigator._derive_sar_confidence(evidence) == "moderate"
    # No "low + escalate" contradiction.
