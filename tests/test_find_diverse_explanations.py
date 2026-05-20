# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for ``GDSNavigator.find_diverse_explanations``.

Greedy max-K strictly disjoint cover over per-dim Bregman
contributions. Each hypothesis is a disjoint set of dim labels whose
joint contribution clears ``min_contribution_pct``; once a dim is
claimed by a hypothesis it is removed from the pool, so the Jaccard
between any two hypotheses is structurally 0 and the post-hoc
``diversity_score`` (mean pairwise ``1 - jaccard``) is always 1.0
when at least two hypotheses are returned. Tests use ``MagicMock``-
backed storage so the navigator sees engineered deltas without
opening a real sphere.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.model.sphere import Pattern, RelationDef
from hypertopos.navigation.navigator import GDSNavigationError, GDSNavigator


# ── Synthetic pattern helpers ──────────────────────────────────────────────


def _make_relations(line_ids: list[str]) -> list[RelationDef]:
    return [
        RelationDef(line_id=lid, direction="out", required=False)
        for lid in line_ids
    ]


def _make_pattern(
    *,
    line_ids: list[str],
    mu: list[float],
    sigma_diag: list[float],
    theta: list[float],
    dimension_kinds: list[str] | None = None,
) -> Pattern:
    """Build a synthetic Pattern. ``dimension_kinds=None`` skips the
    Bregman branch in ``_per_dim_anomaly_contributions`` and falls
    back to ``delta**2`` — convenient for tests that want to fully
    control the contribution vector."""
    return Pattern(
        pattern_id="test_pattern",
        entity_type="account",
        pattern_type="anchor",
        relations=_make_relations(line_ids),
        mu=np.asarray(mu, dtype=np.float64),
        sigma_diag=np.asarray(sigma_diag, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        population_size=1000,
        computed_at=datetime(2026, 5, 17, tzinfo=UTC),
        version=1,
        status="production",
        dimension_kinds=dimension_kinds,
    )


def _make_nav(pattern: Pattern, *, delta: list[float]) -> GDSNavigator:
    """Build a GDSNavigator with a mocked storage layer that returns a
    single geometry row matching ``delta``. ``delta_norm`` is computed
    inline so the navigator sees a self-consistent row."""
    delta_np = np.asarray(delta, dtype=np.float64)
    delta_norm = float(np.linalg.norm(delta_np))

    nav = GDSNavigator.__new__(GDSNavigator)
    nav._storage = MagicMock()
    nav._storage.read_sphere.return_value = MagicMock(
        patterns={"test_pattern": pattern},
    )
    nav._storage.read_geometry.return_value = pa.table({
        "primary_key": ["E1"],
        "delta": pa.array([delta_np.tolist()], type=pa.list_(pa.float64())),
        "delta_norm": [delta_norm],
    })
    nav._resolve_version = MagicMock(return_value=1)
    return nav


# ── Tests ──────────────────────────────────────────────────────────────────


def test_single_dim_anomaly_returns_one_hypothesis_and_degrades():
    """delta=[5,0,0,0] with sigma=1 — contributions=[25,0,0,0]. The
    single dim alone covers 100% of the mass, so one hypothesis is
    emitted and the cover loop breaks (no further mass left to
    clear the 10% floor)."""
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D"],
        mu=[0.0, 0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=[5.0, 0.0, 0.0, 0.0])

    result = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=3,
        min_contribution_pct=0.10,
    )

    assert result["n_hypotheses_returned"] == 1
    assert result["degraded_reason"] == "insufficient_diverse_mass"
    assert result["hypotheses"][0]["dim_labels"] == ["A"]
    # Single hypothesis ⇒ no pair to compare ⇒ diversity_score is None.
    assert result["diversity_score"] is None


def test_two_singleton_disjoint_hypotheses():
    """Two top dims each individually clear the floor; K=2 must
    return them as two singletons, each in its own hypothesis.

    Note on semantic: disjoint+greedy partitions top-mass dims into
    K singletons when each individually meets the floor — this is
    the correct semantic for K disjoint hypotheses, NOT "one per
    cluster" which would require a different algorithm (k-means on
    the contribution vector).

    Fixture: contributions [10, 10, 5, 5, 5, 5] (total 40). Floor
    0.10 ⇒ a 4-unit hypothesis (10% of 40) clears it. The two top
    dims (0 and 1) are each 25% on their own, so each is emitted
    as a singleton.
    """
    # delta = sqrt(contribution) under sigma=1, dimension_kinds=None
    # (falls back to delta**2).
    delta = [
        float(np.sqrt(10.0)),
        float(np.sqrt(10.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
    ]
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D", "E", "F"],
        mu=[0.0] * 6,
        sigma_diag=[1.0] * 6,
        theta=[3.0] * 6,
    )
    nav = _make_nav(pattern, delta=delta)

    result = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=2,
        min_contribution_pct=0.10,
    )

    assert result["n_hypotheses_returned"] == 2
    assert result["degraded_reason"] is None

    h1_dims = set(result["hypotheses"][0]["dim_labels"])
    h2_dims = set(result["hypotheses"][1]["dim_labels"])
    # Two singletons, each a top-mass dim, disjoint.
    assert h1_dims == {"A"}
    assert h2_dims == {"B"}
    # Both clear the floor: 10/40 = 25% ≥ 10%.
    assert result["hypotheses"][0]["joint_contribution_pct"] == 25.0
    assert result["hypotheses"][1]["joint_contribution_pct"] == 25.0
    # Strict disjoint ⇒ Jaccard 0 on every pair ⇒ diversity 1.0.
    assert result["diversity_score"] == 1.0


def test_k3_degrades_when_remaining_mass_below_floor():
    """K=3 over a fixture where only two dims individually clear the
    floor and the third hypothesis cannot be filled.

    Fixture: contributions [20, 20, 5] (total 45). Floor 0.30 ⇒ a
    hypothesis needs ≥ 13.5 mass. dim0 = 20/45 ≈ 44.4% ✓,
    dim1 = 20/45 ≈ 44.4% ✓ — both emit as singletons. H3 starts
    with only dim2 left (5/45 ≈ 11.1% < 30%), pool empties before
    the floor is cleared, degrade.
    """
    delta = [
        float(np.sqrt(20.0)),
        float(np.sqrt(20.0)),
        float(np.sqrt(5.0)),
    ]
    pattern = _make_pattern(
        line_ids=["A", "B", "C"],
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=delta)

    result = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=3,
        min_contribution_pct=0.30,
    )

    assert result["n_hypotheses_returned"] == 2
    assert result["degraded_reason"] == "insufficient_diverse_mass"
    # Strict disjoint ⇒ diversity score 1.0 between the two singletons.
    assert result["diversity_score"] == 1.0


def test_diverse_explanation_degrades_to_top2_when_floor_unmet():
    """Single-dim-dominates fixture: contributions = [96, 2.5, 0.5,
    0.5, 0.5] (total 100). At ``min_contribution_pct=0.10`` only dim
    0 individually clears the floor (96 % ≥ 10 %). The remaining
    pool has 4 % of total mass — no second hypothesis can reach
    10 %. Pre-fix the response was a single hypothesis with
    ``degraded_reason='insufficient_diverse_mass'`` and the agent
    never saw the runner-up dim. Post-fix a second hypothesis is
    emitted with the next-rank dim alone, flagged as degraded,
    carrying its actual joint share (below floor), and
    ``degraded_reason`` becomes ``'diversity_unavailable_top1_only'``
    so the agent can tell why the floor was missed.
    """
    delta = [
        float(np.sqrt(96.0)),
        float(np.sqrt(2.5)),
        float(np.sqrt(0.5)),
        float(np.sqrt(0.5)),
        float(np.sqrt(0.5)),
    ]
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D", "E"],
        mu=[0.0] * 5,
        sigma_diag=[1.0] * 5,
        theta=[3.0] * 5,
    )
    nav = _make_nav(pattern, delta=delta)

    result = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=3,
        min_contribution_pct=0.10,
    )

    assert result["n_hypotheses_returned"] == 2
    assert result["degraded_reason"] == "diversity_unavailable_top1_only"

    # Primary hypothesis: dim A alone (96 %).
    primary = result["hypotheses"][0]
    assert primary["dim_labels"] == ["A"]
    assert primary["joint_contribution_pct"] == 96.0
    assert "is_degraded" not in primary

    # Secondary hypothesis: dim B alone (2.5 %), flagged as degraded.
    secondary = result["hypotheses"][1]
    assert secondary["dim_labels"] == ["B"]
    assert secondary["joint_contribution_pct"] == 2.5
    assert secondary["is_degraded"] is True
    # Narrative must name the dim and explain it's below the floor.
    assert "B" in secondary["narrative"]
    assert "below" in secondary["narrative"]
    # Two emitted hypotheses ⇒ diversity_score is the pair distance.
    assert result["diversity_score"] == 1.0


def test_n_hypotheses_zero_raises():
    pattern = _make_pattern(
        line_ids=["A"], mu=[0.0], sigma_diag=[1.0], theta=[3.0],
    )
    nav = _make_nav(pattern, delta=[1.0])

    with pytest.raises(GDSNavigationError, match="must be >= 1"):
        nav.find_diverse_explanations(
            "E1", pattern_id="test_pattern", n_hypotheses=0,
        )


def test_n_hypotheses_exceeds_dim_count_caps_silently():
    """4-dim pattern, K=100 — cap to D=4 and surface as
    ``capped_to_dim_count``. Cap-degradation takes priority over
    alg-degradation by design (cap is the more upstream problem to
    flag)."""
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D"],
        mu=[0.0] * 4,
        sigma_diag=[1.0] * 4,
        theta=[3.0] * 4,
    )
    # All dims equally anomalous so the cover can actually fill 4
    # hypotheses at min_contribution_pct=0.10 (each dim is 25% of
    # the mass).
    nav = _make_nav(pattern, delta=[2.0, 2.0, 2.0, 2.0])

    result = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=100,
        min_contribution_pct=0.10,
    )

    assert result["n_hypotheses_requested"] == 100
    # Cap-degradation set even though all 4 hypotheses successfully
    # emitted — the user still asked for more than D and that fact
    # is worth surfacing.
    assert result["degraded_reason"] == "capped_to_dim_count"
    assert result["n_hypotheses_returned"] == 4


def test_unknown_pattern_id_raises():
    pattern = _make_pattern(
        line_ids=["A"], mu=[0.0], sigma_diag=[1.0], theta=[3.0],
    )
    nav = _make_nav(pattern, delta=[1.0])

    with pytest.raises(GDSNavigationError, match="pattern not found"):
        nav.find_diverse_explanations(
            "E1", pattern_id="does_not_exist", n_hypotheses=2,
        )


def test_unknown_primary_key_raises():
    pattern = _make_pattern(
        line_ids=["A"], mu=[0.0], sigma_diag=[1.0], theta=[3.0],
    )
    nav = _make_nav(pattern, delta=[1.0])
    # Override read_geometry to return an empty table.
    nav._storage.read_geometry.return_value = pa.table({
        "primary_key": pa.array([], type=pa.string()),
        "delta": pa.array([], type=pa.list_(pa.float64())),
        "delta_norm": pa.array([], type=pa.float64()),
    })

    with pytest.raises(GDSNavigationError, match="not found in"):
        nav.find_diverse_explanations(
            "missing_entity", pattern_id="test_pattern", n_hypotheses=2,
        )


def test_determinism():
    """Two consecutive calls with the same input must produce
    byte-identical hypotheses (modulo rounding)."""
    delta = [
        float(np.sqrt(10.0)),
        float(np.sqrt(10.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
    ]
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D", "E", "F"],
        mu=[0.0] * 6,
        sigma_diag=[1.0] * 6,
        theta=[3.0] * 6,
    )
    nav = _make_nav(pattern, delta=delta)

    result_a = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=2,
        min_contribution_pct=0.10,
    )
    result_b = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=2,
        min_contribution_pct=0.10,
    )

    assert result_a["hypotheses"] == result_b["hypotheses"]
    assert result_a["diversity_score"] == result_b["diversity_score"]
    assert result_a["degraded_reason"] == result_b["degraded_reason"]


def test_validate_true_calls_simulate_dimension_change_per_hypothesis():
    """With ``validate=True`` each hypothesis triggers one
    ``simulate_dimension_change`` call; the mock returns a successful
    neutralisation so each hypothesis carries
    ``neutralizes_anomaly=True``."""
    delta = [
        float(np.sqrt(10.0)),
        float(np.sqrt(10.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
        float(np.sqrt(5.0)),
    ]
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D", "E", "F"],
        mu=[0.0] * 6,
        sigma_diag=[1.0] * 6,
        theta=[3.0] * 6,
    )
    nav = _make_nav(pattern, delta=delta)
    nav.simulate_dimension_change = MagicMock(return_value={
        "delta_norm_after": 0.5,
        "is_anomaly_after": False,
    })

    result = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=2,
        min_contribution_pct=0.10,
        validate=True,
    )

    assert nav.simulate_dimension_change.call_count == 2
    for h in result["hypotheses"]:
        assert h["validation"]["neutralizes_anomaly"] is True
        assert h["validation"]["delta_norm_after_override"] == 0.5


def test_diversity_score_none_when_single_hypothesis():
    """Single-hypothesis case has no pair to compare; ``diversity_score``
    must be ``None`` rather than a numeric default. 0.0 would falsely
    imply "perfectly identical" and 1.0 would falsely imply
    "maximally diverse" — both are lies about a degenerate input."""
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D"],
        mu=[0.0, 0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=[5.0, 0.0, 0.0, 0.0])

    result = nav.find_diverse_explanations(
        "E1",
        pattern_id="test_pattern",
        n_hypotheses=3,
        min_contribution_pct=0.10,
    )

    assert result["n_hypotheses_returned"] == 1
    assert result["diversity_score"] is None
