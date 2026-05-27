"""Tests for GDSNavigator.chain_signed_confidence_rollup.

Pure-composition primitive: resolves a chain's unique member keys via
``chain_keys`` on the chain anchor's points table, reads per-member
polygon geometry on the anchor pattern, attaches reliability flags +
signed-confidence triad via the existing static helpers
(``_attach_reliability_flags`` / ``_attach_signed_confidence_fields``),
then aggregates the per-member values into the four chain-level
reliability fields plus a verdict.

These tests exercise the aggregation arithmetic + verdict thresholds
against engineered polygons — geometry reads and signed-confidence
attachment are monkeypatched so the verdict logic is tested
independently of the storage / engine layers.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypertopos import HyperSphere
from hypertopos.model.objects import Polygon

FIXTURES = Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"


@pytest.fixture(scope="module")
def synthetic_nav():
    hs = HyperSphere.open(str(FIXTURES))
    return hs.session("signed-confidence-rollup-test").navigator()


# ---------------------------------------------------------------------------
# Gate tests — pattern / chain resolution (reach the early-return branches)
# ---------------------------------------------------------------------------


def test_unknown_chain_pattern_raises(synthetic_nav):
    with pytest.raises(ValueError, match="chain_pattern"):
        synthetic_nav.chain_signed_confidence_rollup(
            "CH-001",
            chain_pattern="NONEXISTENT_PATTERN",
            anchor_pattern="account_pattern",
        )


def test_unknown_anchor_pattern_raises(synthetic_nav):
    with pytest.raises(ValueError, match="anchor_pattern"):
        synthetic_nav.chain_signed_confidence_rollup(
            "CH-001",
            chain_pattern="chain_pattern",
            anchor_pattern="NONEXISTENT_PATTERN",
        )


def test_unknown_chain_id_raises(synthetic_nav):
    with pytest.raises(ValueError, match="UNKNOWN-CHAIN"):
        synthetic_nav.chain_signed_confidence_rollup(
            "UNKNOWN-CHAIN",
            chain_pattern="chain_pattern",
            anchor_pattern="account_pattern",
        )


# ---------------------------------------------------------------------------
# Verdict threshold tests — monkeypatched polygons
# ---------------------------------------------------------------------------


def _make_polygon(
    pk: str,
    score: float,
    penalty: float,
    single_dim_driven: bool,
) -> Polygon:
    """Synthesize a Polygon with the rollup-relevant attributes set."""
    poly = Polygon(
        primary_key=pk,
        pattern_id="account_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1.0,
        delta=np.array([0.0], dtype=np.float32),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[],
        last_refresh_at=0,
        updated_at=0,
    )
    poly.signed_confidence_score = score  # type: ignore[attr-defined]
    poly.reliability_penalty = penalty  # type: ignore[attr-defined]
    poly.reliability_flags = {  # type: ignore[attr-defined]
        "single_dim_driven": single_dim_driven,
        "low_confidence_bucket": False,
    }
    return poly


def _patch_label_aware_pipeline(
    monkeypatch,
    nav,
    *,
    polygons: list[Polygon],
):
    """Force the rollup to operate on engineered polygons.

    Bypasses storage I/O + engine polygon construction + the static
    attachers — the polygons already have ``signed_confidence_score``
    / ``reliability_penalty`` / ``reliability_flags`` set, so the
    rollup's aggregation arithmetic and verdict thresholds are
    isolated from the calibration computation.

    Also monkeypatches the anchor pattern's ``label_aware_calibration``
    to a truthy stub so the label-aware-unavailable gate is bypassed.
    """
    # Patch label-aware calibration on the anchor pattern in the
    # navigator's currently-cached sphere read.
    real_read_sphere = nav._storage.read_sphere

    def fake_read_sphere():
        sphere = real_read_sphere()
        # Inject a non-None calibration so the rollup proceeds past
        # the label-aware-unavailable gate. The polygons we pass in
        # already carry their signed_confidence_score, so the actual
        # contents of label_aware_calibration are never read after
        # the gate.
        sphere.patterns["account_pattern"].label_aware_calibration = {
            "fake_label": MagicMock(direction=1.0),
        }
        return sphere
    monkeypatch.setattr(nav._storage, "read_sphere", fake_read_sphere)

    # Stub read_geometry — payload contents are ignored downstream
    # (we replace geometry_to_polygons immediately after).
    monkeypatch.setattr(
        nav._storage, "read_geometry",
        lambda *a, **k: MagicMock(),
    )
    monkeypatch.setattr(
        nav._engine, "geometry_to_polygons",
        lambda *a, **k: list(polygons),
    )
    # No-op the attachers — polygons already carry the fields they
    # would set.
    monkeypatch.setattr(
        nav, "_attach_reliability_flags",
        lambda polys, *, pattern: None,
    )
    monkeypatch.setattr(
        nav, "_attach_signed_confidence_fields",
        lambda polys, *, pattern: None,
    )


def test_low_verdict_when_half_or_more_members_are_low_confidence(
    synthetic_nav, monkeypatch,
):
    """4 members, 3 with reliability_penalty >= 0.5 → verdict 'low'."""
    polygons = [
        _make_polygon("A1", score=0.2, penalty=0.5, single_dim_driven=True),
        _make_polygon("A2", score=0.1, penalty=0.5, single_dim_driven=True),
        _make_polygon("A3", score=0.05, penalty=1.0, single_dim_driven=False),
        _make_polygon("A4", score=1.5, penalty=0.0, single_dim_driven=False),
    ]
    _patch_label_aware_pipeline(monkeypatch, synthetic_nav, polygons=polygons)

    out = synthetic_nav.chain_signed_confidence_rollup(
        "CH-001",
        chain_pattern="chain_pattern",
        anchor_pattern="account_pattern",
    )
    # Mean = (0.2 + 0.1 + 0.05 + 1.5) / 4 = 0.4625
    assert out["chain_mean_signed_confidence"] == pytest.approx(0.4625)
    # 3 of 4 members have penalty >= 0.5
    assert out["chain_n_low_confidence_members"] == 3
    # 2 of 4 are single_dim_driven
    assert out["chain_n_single_dim_driven_members"] == 2
    # 3 >= 0.5 * 4 → "low"
    assert out["chain_confidence_verdict"] == "low"
    assert out["n_members_resolved"] == 4


def test_medium_verdict_when_mean_below_one_but_not_low(
    synthetic_nav, monkeypatch,
):
    """4 members, only 1 low-confidence, mean < 1.0 → 'medium'."""
    polygons = [
        _make_polygon("A1", score=0.6, penalty=0.5, single_dim_driven=False),
        _make_polygon("A2", score=0.7, penalty=0.0, single_dim_driven=False),
        _make_polygon("A3", score=0.8, penalty=0.0, single_dim_driven=False),
        _make_polygon("A4", score=0.9, penalty=0.0, single_dim_driven=False),
    ]
    _patch_label_aware_pipeline(monkeypatch, synthetic_nav, polygons=polygons)

    out = synthetic_nav.chain_signed_confidence_rollup(
        "CH-001",
        chain_pattern="chain_pattern",
        anchor_pattern="account_pattern",
    )
    # Mean = (0.6 + 0.7 + 0.8 + 0.9) / 4 = 0.75 → < 1.0
    assert out["chain_mean_signed_confidence"] == pytest.approx(0.75)
    # Only 1 low-confidence member (1 < 0.5 * 4 = 2)
    assert out["chain_n_low_confidence_members"] == 1
    # 0 single-dim-driven
    assert out["chain_n_single_dim_driven_members"] == 0
    # Mean < 1.0 and not low → "medium"
    assert out["chain_confidence_verdict"] == "medium"


def test_high_verdict_when_mean_at_or_above_one_and_clean(
    synthetic_nav, monkeypatch,
):
    """4 clean members, mean >= 1.0 → 'high'."""
    polygons = [
        _make_polygon("A1", score=1.5, penalty=0.0, single_dim_driven=False),
        _make_polygon("A2", score=1.8, penalty=0.0, single_dim_driven=False),
        _make_polygon("A3", score=2.0, penalty=0.0, single_dim_driven=False),
        _make_polygon("A4", score=2.2, penalty=0.0, single_dim_driven=False),
    ]
    _patch_label_aware_pipeline(monkeypatch, synthetic_nav, polygons=polygons)

    out = synthetic_nav.chain_signed_confidence_rollup(
        "CH-001",
        chain_pattern="chain_pattern",
        anchor_pattern="account_pattern",
    )
    # Mean = (1.5 + 1.8 + 2.0 + 2.2) / 4 = 1.875
    assert out["chain_mean_signed_confidence"] == pytest.approx(1.875)
    assert out["chain_n_low_confidence_members"] == 0
    assert out["chain_n_single_dim_driven_members"] == 0
    # No low-confidence and mean >= 1.0 → "high"
    assert out["chain_confidence_verdict"] == "high"


def test_label_aware_unavailable_yields_null_fields(synthetic_nav):
    """No monkeypatch — the synthetic sphere's account_pattern has no
    label_aware_calibration. Verdict must be 'label-aware-unavailable'
    with all four numeric fields = None.
    """
    out = synthetic_nav.chain_signed_confidence_rollup(
        "CH-001",
        chain_pattern="chain_pattern",
        anchor_pattern="account_pattern",
    )
    assert out["chain_mean_signed_confidence"] is None
    assert out["chain_n_low_confidence_members"] is None
    assert out["chain_n_single_dim_driven_members"] is None
    assert out["chain_confidence_verdict"] == "label-aware-unavailable"
    # n_members is still surfaced — the gate is on the verdict only.
    assert out["n_members"] >= 1
    assert out["n_members_resolved"] == 0


def test_low_verdict_takes_precedence_over_high_mean(
    synthetic_nav, monkeypatch,
):
    """Verdict precedence test — even if mean is high, when >= half of
    members are low-confidence, verdict must collapse to 'low'.
    """
    polygons = [
        _make_polygon("A1", score=5.0, penalty=0.5, single_dim_driven=True),
        _make_polygon("A2", score=5.0, penalty=0.5, single_dim_driven=True),
        _make_polygon("A3", score=5.0, penalty=0.0, single_dim_driven=False),
        _make_polygon("A4", score=5.0, penalty=0.0, single_dim_driven=False),
    ]
    _patch_label_aware_pipeline(monkeypatch, synthetic_nav, polygons=polygons)

    out = synthetic_nav.chain_signed_confidence_rollup(
        "CH-001",
        chain_pattern="chain_pattern",
        anchor_pattern="account_pattern",
    )
    assert out["chain_mean_signed_confidence"] == pytest.approx(5.0)
    # 2 of 4 → 2 >= 0.5 * 4 = 2.0 → "low" takes precedence
    assert out["chain_n_low_confidence_members"] == 2
    assert out["chain_confidence_verdict"] == "low"


def test_anti_aligned_chain_mean_lands_in_medium(synthetic_nav, monkeypatch):
    """Anti-aligned chain (mean < 0) — by literal spec lands in 'medium'.

    Documents the design choice: 'medium' is the literal mean-below-1.0
    bucket; demoting anti-aligned chains further would require a
    different threshold ladder, which the current spec does not have.
    """
    polygons = [
        _make_polygon("A1", score=-0.5, penalty=0.0, single_dim_driven=False),
        _make_polygon("A2", score=-0.3, penalty=0.0, single_dim_driven=False),
        _make_polygon("A3", score=-0.2, penalty=0.0, single_dim_driven=False),
        _make_polygon("A4", score=-0.4, penalty=0.0, single_dim_driven=False),
    ]
    _patch_label_aware_pipeline(monkeypatch, synthetic_nav, polygons=polygons)

    out = synthetic_nav.chain_signed_confidence_rollup(
        "CH-001",
        chain_pattern="chain_pattern",
        anchor_pattern="account_pattern",
    )
    assert out["chain_mean_signed_confidence"] < 0.0
    assert out["chain_n_low_confidence_members"] == 0
    # No low-confidence + mean < 1.0 → "medium" by literal threshold
    assert out["chain_confidence_verdict"] == "medium"
