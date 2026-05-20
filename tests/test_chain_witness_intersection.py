"""Tests for GDSNavigator.chain_witness_intersection.

Pure-composition primitive that intersects per-member top witness
dimensions across all members of a chain. Members sharing >=min_jaccard
witness sets imply coordinated anomaly mechanism. The synthetic chain
sphere exposes real chain_keys resolution / pattern-type / unknown-id
gates without rebuilding polygons; explain_anomaly is monkeypatched
throughout (mirrors `test_investigate_entity.py`) because the synthetic
fixture's hand-injected geometry skips the polygon-build path that
`explain_anomaly` needs.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypertopos import HyperSphere

FIXTURES = Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"


@pytest.fixture(scope="module")
def synthetic_nav():
    hs = HyperSphere.open(str(FIXTURES))
    return hs.session("witness-intersection-test").navigator()


# ---------------------------------------------------------------------------
# Gate tests — chain_keys resolution / pattern type / unknown id
# (no explain_anomaly call reached, so no monkeypatch needed)
# ---------------------------------------------------------------------------


def test_unknown_chain_pattern_raises(synthetic_nav):
    with pytest.raises(ValueError, match="chain_pattern"):
        synthetic_nav.chain_witness_intersection(
            "CH-001",
            chain_pattern="NONEXISTENT_PATTERN",
            member_pattern="account_pattern",
        )


def test_unknown_member_pattern_raises(synthetic_nav):
    with pytest.raises(ValueError, match="member_pattern"):
        synthetic_nav.chain_witness_intersection(
            "CH-001",
            chain_pattern="chain_pattern",
            member_pattern="NONEXISTENT_PATTERN",
        )


def test_missing_chain_keys_column_raises(synthetic_nav):
    """account_pattern is an anchor pattern but its line (accounts) lacks
    chain_keys. The call must reject before reaching explain_anomaly.
    """
    with pytest.raises(ValueError, match="chain_keys"):
        synthetic_nav.chain_witness_intersection(
            "A1",
            chain_pattern="account_pattern",
            member_pattern="account_pattern",
        )


def test_unknown_chain_id_raises(synthetic_nav):
    with pytest.raises(ValueError, match="UNKNOWN-CHAIN"):
        synthetic_nav.chain_witness_intersection(
            "UNKNOWN-CHAIN",
            chain_pattern="chain_pattern",
            member_pattern="account_pattern",
        )


# ---------------------------------------------------------------------------
# Composition tests — explain_anomaly monkeypatched
# ---------------------------------------------------------------------------


def _make_fake_explain(witness_map: dict[str, list[str] | None]):
    """Build a fake explain_anomaly that returns engineered top_dimensions
    or raises ValueError for None entries.
    """
    def fake(primary_key: str, pattern_id: str) -> dict:
        result = witness_map.get(primary_key, "missing")
        if result == "missing":
            raise ValueError(f"unknown entity {primary_key!r}")
        if result is None:
            raise ValueError(f"forced skip for {primary_key!r}")
        return {
            "severity": "high",
            "delta_norm": 99.0,
            "theta_norm": 5.0,
            "top_dimensions": [
                {"dim": i, "label": lbl, "delta": 1.0, "contribution_pct": 50.0}
                for i, lbl in enumerate(result)
            ],
        }
    return fake


def test_coordinated_chain_returns_intersection(synthetic_nav, monkeypatch):
    """CH-001 = [A1, A2, A3, A4] — engineered so all four share top dim
    `risk_score`. Intersection must surface risk_score and
    mean_pairwise_witness_jaccard >= min_jaccard => coordinated True.
    """
    fake = _make_fake_explain({
        "A1": ["risk_score", "diversity", "regularity"],
        "A2": ["risk_score", "diversity", "regularity"],
        "A3": ["risk_score", "diversity", "regularity"],
        "A4": ["risk_score", "diversity", "regularity"],
    })
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    out = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        min_jaccard=0.5,
        top_k_witness=5,
    )
    assert out["chain_id"] == "CH-001"
    assert out["chain_pattern"] == "chain_pattern"
    assert out["member_pattern"] == "account_pattern"
    assert out["n_members"] == 4
    assert out["n_members_explained"] == 4
    assert out["n_members_skipped"] == 0
    assert out["intersected_witness_dims"] == ["diversity", "regularity", "risk_score"]
    assert out["union_witness_dims"] == ["diversity", "regularity", "risk_score"]
    assert out["mean_pairwise_witness_jaccard"] == 1.0
    assert out["coordinated"] is True
    assert isinstance(out["interpretation"], str)
    assert "coordinated" in out["interpretation"].lower()
    assert len(out["per_member_top_dims"]) == 4
    member_keys = [m["primary_key"] for m in out["per_member_top_dims"]]
    assert member_keys == sorted(member_keys)


def test_self_loop_chain_dedupes_members_and_raises(synthetic_nav, monkeypatch):
    """CH-008 = [A1, A1] — after dedupe only 1 unique member, so
    n_members_explained < 2 and the function must raise. With
    explain_anomaly monkeypatched to succeed, the raise comes from the
    dedupe gate (not from explain skipping).
    """
    fake = _make_fake_explain({"A1": ["risk_score", "diversity"]})
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    with pytest.raises(ValueError, match="CH-008"):
        synthetic_nav.chain_witness_intersection(
            "CH-008",
            chain_pattern="chain_pattern",
            member_pattern="account_pattern",
        )


def test_deterministic_output_ordering(synthetic_nav, monkeypatch):
    """Same input twice — identical output (alphabetical sort verified)."""
    fake = _make_fake_explain({
        "A1": ["zeta", "alpha", "mu"],
        "A2": ["zeta", "alpha", "nu"],
        "A3": ["zeta", "alpha", "xi"],
        "A4": ["zeta", "alpha", "omicron"],
    })
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    out1 = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
    )
    out2 = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
    )
    assert out1["intersected_witness_dims"] == out2["intersected_witness_dims"]
    assert out1["union_witness_dims"] == out2["union_witness_dims"]
    assert out1["intersected_witness_dims"] == sorted(
        out1["intersected_witness_dims"],
    )
    assert out1["union_witness_dims"] == sorted(out1["union_witness_dims"])
    assert out1["per_member_top_dims"] == out2["per_member_top_dims"]
    # Verify alphabetical sort is non-trivial here (zeta would come last)
    assert out1["union_witness_dims"][0] == "alpha"


def test_disjoint_witness_sets_returns_empty_intersection(
    synthetic_nav, monkeypatch,
):
    """4 members with totally disjoint witness sets — intersection empty,
    pairwise jaccard 0, coordinated False.
    """
    fake = _make_fake_explain({
        "A1": ["a", "b"],
        "A2": ["c", "d"],
        "A3": ["e", "f"],
        "A4": ["g", "h"],
    })
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    out = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        min_jaccard=0.5,
    )
    assert out["n_members_explained"] == 4
    assert out["intersected_witness_dims"] == []
    assert sorted(out["union_witness_dims"]) == ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert out["mean_pairwise_witness_jaccard"] == 0.0
    assert out["coordinated"] is False
    assert isinstance(out["interpretation"], str)


def test_partial_overlap_yields_expected_jaccard(synthetic_nav, monkeypatch):
    """4 members each with top-5: every pair shares dims 'x','y' out of 5 unique
    each → pairwise jaccard = 2/8 = 0.25. Intersection = {x,y}.
    """
    fake = _make_fake_explain({
        "A1": ["x", "y", "a", "b", "c"],
        "A2": ["x", "y", "d", "e", "f"],
        "A3": ["x", "y", "g", "h", "i"],
        "A4": ["x", "y", "j", "k", "l"],
    })
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    out = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        min_jaccard=0.5,
    )
    assert out["intersected_witness_dims"] == ["x", "y"]
    assert abs(out["mean_pairwise_witness_jaccard"] - 0.25) < 1e-6
    assert out["coordinated"] is False


def test_skipped_member_handling(synthetic_nav, monkeypatch):
    """When one member's explain_anomaly raises, n_members_skipped reflects it
    and jaccard is computed over the remaining members.
    """
    fake = _make_fake_explain({
        "A1": ["alpha", "beta"],
        "A2": ["alpha", "beta"],
        "A3": None,  # forced raise
        "A4": ["alpha", "beta"],
    })
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    out = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
    )
    assert out["n_members"] == 4
    assert out["n_members_explained"] == 3
    assert out["n_members_skipped"] == 1
    member_keys = sorted(m["primary_key"] for m in out["per_member_top_dims"])
    assert member_keys == ["A1", "A2", "A4"]
    assert out["intersected_witness_dims"] == ["alpha", "beta"]
    assert out["mean_pairwise_witness_jaccard"] == 1.0
    assert out["coordinated"] is True
    # Interpretation must report the explained count over total unique
    # members (not over itself) so skipped members are visible.
    interpretation = out["interpretation"]
    assert "coordinated" in interpretation.lower()
    assert "3 of 4 members" in interpretation


def test_empty_top_dims_produces_zero_jaccard(synthetic_nav, monkeypatch):
    """When every member's explain_anomaly returns empty top_dimensions
    (e.g. non-anomalous members surfaced via `severity: normal`),
    the function must not crash and pairwise jaccard must be 0.0
    (never NaN — sanitisation path).
    """
    def fake_empty(primary_key: str, pattern_id: str) -> dict:
        return {
            "severity": "normal",
            "delta_norm": 1.0,
            "theta_norm": 5.0,
            # no top_dimensions key — handled by .get with default
        }
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake_empty)
    out = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
    )
    assert out["n_members_explained"] == 4
    assert out["intersected_witness_dims"] == []
    assert out["union_witness_dims"] == []
    # Critical: NaN never surfaces — either 0.0 or None, never NaN/Infinity
    j = out["mean_pairwise_witness_jaccard"]
    assert j is None or j == 0.0
    assert out["coordinated"] is False


def test_top_k_witness_truncates(synthetic_nav, monkeypatch):
    """top_k_witness=2 must truncate each member's witness list, leaving
    only the first 2 dims for intersection.
    """
    fake = _make_fake_explain({
        "A1": ["x", "y", "rare1"],
        "A2": ["x", "y", "rare2"],
        "A3": ["x", "y", "rare3"],
        "A4": ["x", "y", "rare4"],
    })
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    out = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        top_k_witness=2,
    )
    assert out["intersected_witness_dims"] == ["x", "y"]
    # rare* dims are truncated and never appear
    for dim in ("rare1", "rare2", "rare3", "rare4"):
        assert dim not in out["union_witness_dims"]


def test_min_jaccard_threshold_controls_coordinated_flag(
    synthetic_nav, monkeypatch,
):
    """Two members sharing exactly half their witness set (jaccard ~ 0.33) —
    flips coordinated True/False depending on min_jaccard.
    """
    fake = _make_fake_explain({
        "A1": ["x", "y"],
        "A2": ["x", "z"],
        "A3": ["x", "w"],
        "A4": ["x", "v"],
    })
    monkeypatch.setattr(synthetic_nav, "explain_anomaly", fake)
    out_strict = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        min_jaccard=0.5,
    )
    assert out_strict["coordinated"] is False
    out_loose = synthetic_nav.chain_witness_intersection(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        min_jaccard=0.3,
    )
    assert out_loose["coordinated"] is True
    # Same jaccard regardless of threshold
    assert (
        out_strict["mean_pairwise_witness_jaccard"]
        == out_loose["mean_pairwise_witness_jaccard"]
    )
