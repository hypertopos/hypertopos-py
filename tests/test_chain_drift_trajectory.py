"""Tests for GDSNavigator.chain_drift_trajectory.

Pure-composition primitive that slices each chain member's temporal
history (`engine.build_solid`) into n_windows windows, computes per-window
mean delta_norm, fits a least-squares slope, and labels each member's
regime as normalizing / deteriorating / neutral. Chain-level regime is the
consensus over members; chain_drift_score is the |slope| weighted by
each member's final-window delta_norm.

The synthetic chain sphere supplies real chain_keys / pattern-type /
unknown-id gates. `engine.build_solid` is monkeypatched throughout so
tests inject deterministic SolidSlice trajectories without exercising
the temporal Lance reader.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from hypertopos import HyperSphere
from hypertopos.model.objects import Polygon, Solid, SolidSlice

FIXTURES = Path(__file__).parent / "fixtures" / "gds" / "synthetic_chain_sphere"


@pytest.fixture(scope="module")
def synthetic_nav():
    hs = HyperSphere.open(str(FIXTURES))
    return hs.session("chain-drift-trajectory-test").navigator()


def _make_slice(idx: int, delta_norm: float, *, dim: int = 3) -> SolidSlice:
    ts = datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=idx)
    return SolidSlice(
        slice_index=idx,
        timestamp=ts,
        deformation_type="internal",
        delta_snapshot=np.zeros(dim, dtype=np.float32),
        delta_norm_snapshot=delta_norm,
        pattern_ver=1,
        changed_property=None,
        changed_line_id=None,
        added_edge=None,
    )


def _make_solid(primary_key: str, delta_norms: list[float]) -> Solid:
    slices = [_make_slice(i, dn) for i, dn in enumerate(delta_norms)]
    ts = datetime(2026, 5, 1, tzinfo=UTC)
    base = Polygon(
        primary_key=primary_key,
        pattern_id="account_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(3, dtype=np.float32),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[],
        last_refresh_at=ts,
        updated_at=ts,
    )
    return Solid(
        primary_key=primary_key,
        pattern_id="account_pattern",
        base_polygon=base,
        slices=slices,
    )


def _make_build_solid(trajectories: dict[str, list[float] | str]):
    """Build a fake engine.build_solid.

    Value semantics in `trajectories`:
      - list[float] of delta_norms -> Solid with those slices in order
      - "missing" -> raises ValueError (member key not in geometry)
    """
    def fake(primary_key, pattern_id, manifest, filters=None, timestamp=None):
        traj = trajectories.get(primary_key)
        if traj is None or traj == "missing":
            raise ValueError(f"unknown entity {primary_key!r}")
        return _make_solid(primary_key, traj)
    return fake


# ---------------------------------------------------------------------------
# Gate tests — chain_keys / pattern type / unknown id / n_windows
# (no build_solid call reached, so no monkeypatch needed)
# ---------------------------------------------------------------------------


def test_unknown_chain_pattern_raises(synthetic_nav):
    with pytest.raises(ValueError, match="chain_pattern"):
        synthetic_nav.chain_drift_trajectory(
            "CH-001",
            chain_pattern="NONEXISTENT",
            member_pattern="account_pattern",
        )


def test_unknown_member_pattern_raises(synthetic_nav):
    with pytest.raises(ValueError, match="member_pattern"):
        synthetic_nav.chain_drift_trajectory(
            "CH-001",
            chain_pattern="chain_pattern",
            member_pattern="NONEXISTENT",
        )


def test_wrong_pattern_type_raises(synthetic_nav):
    """account_pattern is anchor — but chain_pattern role expects chain_keys.
    Passing an anchor without chain_keys must surface the chain_keys gate.
    The synthetic fixture's account_pattern is anchor type AND its line has
    no chain_keys column, so this exercises the chain_keys-missing branch.
    """
    with pytest.raises(ValueError, match="chain_keys"):
        synthetic_nav.chain_drift_trajectory(
            "A1",
            chain_pattern="account_pattern",
            member_pattern="account_pattern",
        )


def test_unknown_chain_id_raises(synthetic_nav):
    with pytest.raises(ValueError, match="UNKNOWN"):
        synthetic_nav.chain_drift_trajectory(
            "UNKNOWN-CHAIN",
            chain_pattern="chain_pattern",
            member_pattern="account_pattern",
        )


def test_n_windows_below_two_raises(synthetic_nav):
    with pytest.raises(ValueError, match="n_windows"):
        synthetic_nav.chain_drift_trajectory(
            "CH-001",
            chain_pattern="chain_pattern",
            member_pattern="account_pattern",
            n_windows=1,
        )


# ---------------------------------------------------------------------------
# Composition tests — engine.build_solid monkeypatched
# ---------------------------------------------------------------------------


def test_deteriorating_member(synthetic_nav, monkeypatch):
    """CH-005 = [A1, A2] — both members linearly increasing delta_norm with
    8 slices each. n_windows=4 → 2 slices per window, mean strictly
    increasing → slope > 0, regime = 'deteriorating'.
    """
    # 8 slices linearly rising 0.0..7.0 — windows mean 0.5, 2.5, 4.5, 6.5
    rising = [float(i) for i in range(8)]
    fake = _make_build_solid({"A1": rising, "A2": rising})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out["n_members"] == 2
    assert out["n_members_with_history"] == 2
    assert out["n_members_skipped"] == 0
    assert out["n_members_short_history"] == 0
    assert len(out["per_position_trajectory"]) == 2
    for entry in out["per_position_trajectory"]:
        assert entry["delta_norms_over_time"] == [0.5, 2.5, 4.5, 6.5]
        assert entry["slope"] > 0
        assert entry["regime"] == "deteriorating"
    assert out["chain_level_regime"] == "deteriorating"
    assert out["chain_drift_score"] is not None
    assert out["chain_drift_score"] > 0


def test_normalizing_member(synthetic_nav, monkeypatch):
    """8 slices linearly falling — slope < 0, regime = 'normalizing'."""
    falling = [float(7 - i) for i in range(8)]
    fake = _make_build_solid({"A1": falling, "A2": falling})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    for entry in out["per_position_trajectory"]:
        assert entry["slope"] < 0
        assert entry["regime"] == "normalizing"
    assert out["chain_level_regime"] == "normalizing"


def test_flat_member(synthetic_nav, monkeypatch):
    """All slices have identical delta_norm — slope ≈ 0, regime = 'neutral'."""
    flat = [3.0] * 8
    fake = _make_build_solid({"A1": flat, "A2": flat})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    for entry in out["per_position_trajectory"]:
        assert abs(entry["slope"]) < 1e-9
        assert entry["regime"] == "neutral"
    assert out["chain_level_regime"] == "neutral"


def test_mixed_chain_returns_mixed(synthetic_nav, monkeypatch):
    """CH-006 = [A3, A4, A1] — one deteriorating, one normalizing, one neutral.
    Chain-level regime must be 'mixed'.
    """
    rising = [float(i) for i in range(8)]
    falling = [float(7 - i) for i in range(8)]
    flat = [3.0] * 8
    fake = _make_build_solid({"A3": rising, "A4": falling, "A1": flat})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-006",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out["n_members"] == 3
    assert out["n_members_with_history"] == 3
    regimes = {e["regime"] for e in out["per_position_trajectory"]}
    assert regimes == {"deteriorating", "normalizing", "neutral"}
    assert out["chain_level_regime"] == "mixed"


def test_all_deteriorating_chain(synthetic_nav, monkeypatch):
    """All 3 members deteriorating → chain_level_regime = 'deteriorating'."""
    rising = [float(i) for i in range(8)]
    fake = _make_build_solid({"A3": rising, "A4": rising, "A1": rising})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-006",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
    )
    assert out["chain_level_regime"] == "deteriorating"


def test_short_history_is_soft_skip(synthetic_nav, monkeypatch):
    """CH-009 = [A1, A2, B1, A3, A4] — 5 members. A1 and A3 have 8 slices.
    A2, B1, A4 have only 2 slices (< n_windows=4). They are soft-skipped
    into n_members_short_history; A1 and A3 remain in per_position_trajectory
    with their original chain positions (0 and 3, gaps preserved).
    """
    rising = [float(i) for i in range(8)]
    short = [1.0, 2.0]
    fake = _make_build_solid({
        "A1": rising,
        "A2": short,
        "B1": short,
        "A3": rising,
        "A4": short,
    })
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-009",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out["n_members"] == 5
    assert out["n_members_with_history"] == 2
    assert out["n_members_short_history"] == 3
    assert out["n_members_skipped"] == 0
    # Position semantics: original deduped chain index preserved, with gaps.
    positions = [e["position"] for e in out["per_position_trajectory"]]
    assert positions == [0, 3]
    # Members at gap positions are absent
    keys = [e["member_key"] for e in out["per_position_trajectory"]]
    assert keys == ["A1", "A3"]


def test_skipped_member_is_distinct_from_short_history(
    synthetic_nav, monkeypatch,
):
    """Member whose build_solid RAISES is counted as n_members_skipped, not
    n_members_short_history. Position gap still preserved.
    """
    rising = [float(i) for i in range(8)]
    fake = _make_build_solid({
        "A1": rising,
        "A2": "missing",   # build_solid raises
        "A3": rising,
        "A4": rising,
    })
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out["n_members"] == 4
    assert out["n_members_with_history"] == 3
    assert out["n_members_skipped"] == 1
    assert out["n_members_short_history"] == 0
    positions = [e["position"] for e in out["per_position_trajectory"]]
    assert positions == [0, 2, 3]


def test_all_members_short_raises(synthetic_nav, monkeypatch):
    """Every member has insufficient history → n_members_with_history < 1
    triggers ValueError.
    """
    short = [1.0, 2.0]
    fake = _make_build_solid({
        "A1": short, "A2": short, "A3": short, "A4": short,
    })
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    with pytest.raises(ValueError, match="n_members_with_history"):
        synthetic_nav.chain_drift_trajectory(
            "CH-001",
            chain_pattern="chain_pattern",
            member_pattern="account_pattern",
            n_windows=4,
        )


def test_determinism(synthetic_nav, monkeypatch):
    """Same input twice — identical output ordering."""
    rising = [float(i) for i in range(8)]
    fake = _make_build_solid({
        "A1": rising, "A2": rising, "A3": rising, "A4": rising,
    })
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out1 = synthetic_nav.chain_drift_trajectory(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    out2 = synthetic_nav.chain_drift_trajectory(
        "CH-001",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out1 == out2
    positions = [e["position"] for e in out1["per_position_trajectory"]]
    assert positions == sorted(positions)


def test_nan_slice_sanitised_to_none(synthetic_nav, monkeypatch):
    """A member with a NaN delta_norm_snapshot produces a NaN slope —
    the output must surface None (strict JSON contract) for both the
    NaN window mean and the slope.
    """
    rising_with_nan = [0.0, 1.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0]
    rising = [float(i) for i in range(8)]
    fake = _make_build_solid({
        "A1": rising_with_nan, "A2": rising,
    })
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    a1 = next(e for e in out["per_position_trajectory"] if e["member_key"] == "A1")
    # The window containing the NaN slice has a NaN mean → sanitised to None
    assert None in a1["delta_norms_over_time"]
    # Slope on a series containing None/NaN must be None (sanitised)
    assert a1["slope"] is None
    # Regime on a None slope defaults to 'neutral' (no signal)
    assert a1["regime"] == "neutral"
    # chain_drift_score must remain finite (defence in depth)
    assert out["chain_drift_score"] is None or np.isfinite(out["chain_drift_score"])


def test_stride_sample_correctness(synthetic_nav, monkeypatch):
    """Member with 10 slices, n_windows=4: stride = 2 (10//4), windows are
    slices [0,1], [2,3], [4,5], [6,7], tail slices [8,9] dropped.
    Verify exact per-window means.
    """
    delta_norms = [10.0, 20.0,  30.0, 40.0,  50.0, 60.0,  70.0, 80.0,
                   999.0, 999.0]  # tail dropped
    fake = _make_build_solid({"A1": delta_norms, "A2": delta_norms})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    expected = [15.0, 35.0, 55.0, 75.0]
    for entry in out["per_position_trajectory"]:
        assert entry["delta_norms_over_time"] == expected


def test_revisits_dedupe_members(synthetic_nav, monkeypatch):
    """CH-008 = [A1, A1] — after dedupe, 1 unique member. With only 1 member
    and rising history, n_members_with_history=1; chain-level regime
    classification still requires ≥1 explained member.
    """
    rising = [float(i) for i in range(8)]
    fake = _make_build_solid({"A1": rising})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-008",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out["n_members"] == 1
    assert out["n_members_with_history"] == 1
    assert len(out["per_position_trajectory"]) == 1
    assert out["per_position_trajectory"][0]["member_key"] == "A1"
    assert out["chain_level_regime"] == "deteriorating"


def test_zero_theta_norm_labels_all_flat(synthetic_nav, monkeypatch):
    """When member_pattern.theta_norm == 0 the regime cutoff degenerates;
    fall back to labelling every member 'neutral' (no scale, no signal).
    """
    rising = [float(i) for i in range(8)]
    fake = _make_build_solid({"A1": rising, "A2": rising})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    # read_sphere returns a freshly deserialised Sphere each call, so we
    # wrap it to zero out the account_pattern theta in-flight.
    real_read = synthetic_nav._storage.read_sphere

    def patched_read():
        sphere = real_read()
        sphere.patterns["account_pattern"].theta = np.zeros(3, dtype=np.float32)
        return sphere

    monkeypatch.setattr(
        synthetic_nav._storage, "read_sphere", patched_read,
    )
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    for entry in out["per_position_trajectory"]:
        assert entry["regime"] == "neutral"
    assert out["chain_level_regime"] == "neutral"


def test_chain_drift_score_weighted_by_last_window(
    synthetic_nav, monkeypatch,
):
    """chain_drift_score = sum(|slope_i| * last_window_i) / sum(last_window_i).
    Engineer two members: one with large final delta_norm + steep slope,
    one with small final delta_norm + steep slope. The weighted score
    must lie closer to the steep-large-weight slope than the unweighted
    mean of slopes.
    """
    # Member 1: rises 0..7 (final window mean 6.5, big weight)
    big = [float(i) for i in range(8)]
    # Member 2: rises 0..0.7 (final window mean 0.65, small weight)
    small = [i * 0.1 for i in range(8)]
    fake = _make_build_solid({"A1": big, "A2": small})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    slopes = [e["slope"] for e in out["per_position_trajectory"]]
    last_windows = [
        e["delta_norms_over_time"][-1]
        for e in out["per_position_trajectory"]
    ]
    weighted = (
        sum(abs(s) * w for s, w in zip(slopes, last_windows, strict=True))
        / sum(last_windows)
    )
    assert abs(out["chain_drift_score"] - weighted) < 1e-6


def test_direct_map_when_slices_equal_n_windows(synthetic_nav, monkeypatch):
    """When ``len(slices) == n_windows`` the stride is 1, so each window
    contains exactly one slice and ``delta_norms_over_time`` must equal the
    input slice deltas verbatim (within float tolerance).
    """
    deltas = [1.5, 2.5, 4.0, 7.0]  # n_windows == 4, stride == 1
    fake = _make_build_solid({"A1": deltas, "A2": deltas})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out["n_members_with_history"] == 2
    for entry in out["per_position_trajectory"]:
        per_window = entry["delta_norms_over_time"]
        assert len(per_window) == 4
        for got, expected in zip(per_window, deltas, strict=True):
            assert abs(got - expected) < 1e-6


def test_non_anchor_chain_pattern_raises(synthetic_nav, monkeypatch):
    """When ``chain_pattern`` resolves to a non-anchor pattern (e.g. event)
    the anchor-only guard at the top of the primitive must reject before
    reaching the chain_keys / chain_id branches.
    """
    real_read = synthetic_nav._storage.read_sphere

    def patched_read():
        sphere = real_read()
        sphere.patterns["chain_pattern"].pattern_type = "event"
        return sphere

    monkeypatch.setattr(
        synthetic_nav._storage, "read_sphere", patched_read,
    )
    with pytest.raises(ValueError, match="anchor pattern"):
        synthetic_nav.chain_drift_trajectory(
            "CH-001",
            chain_pattern="chain_pattern",
            member_pattern="account_pattern",
        )


def test_uniform_fallback_when_all_last_window_zero(
    synthetic_nav, monkeypatch,
):
    """When every member's final-window delta_norm is 0.0, the weighted-mean
    denominator collapses to zero and the score falls back to an unweighted
    mean of |slope|. With trajectory [6, 4, 2, 0] and stride==1 each window
    is a single value verbatim; least-squares slope is -2.0 → |slope| = 2.0.
    Two members with the same trajectory → unweighted fallback yields 2.0.
    """
    trajectory = [6.0, 4.0, 2.0, 0.0]  # last value 0 → weight clipped to 0
    fake = _make_build_solid({"A1": trajectory, "A2": trajectory})
    monkeypatch.setattr(synthetic_nav._engine, "build_solid", fake)
    out = synthetic_nav.chain_drift_trajectory(
        "CH-005",
        chain_pattern="chain_pattern",
        member_pattern="account_pattern",
        n_windows=4,
    )
    assert out["n_members_with_history"] == 2
    for entry in out["per_position_trajectory"]:
        # Both members share the same trajectory; slope is the polyfit of
        # x=[0..3], y=[6,4,2,0] which is exactly -2.0.
        assert abs(entry["slope"] - (-2.0)) < 1e-6
        assert entry["delta_norms_over_time"][-1] == 0.0
    # Weighted-mean denominator is 0 → fallback to unweighted mean of
    # |slope| = mean([2.0, 2.0]) = 2.0.
    assert out["chain_drift_score"] is not None
    assert abs(out["chain_drift_score"] - 2.0) < 1e-6
