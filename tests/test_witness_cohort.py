# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for witness cohort discovery."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hypertopos.engine.geometry import GDSEngine
from hypertopos.navigation.navigator import (
    CohortMember,
    WitnessCohortResult,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class TestCohortMemberDataclass:
    def test_member_instantiation(self):
        member = CohortMember(
            primary_key="ACC-001",
            score=0.84,
            delta_similarity=0.92,
            witness_overlap=0.67,
            trajectory_alignment=0.71,
            is_anomaly=True,
            delta_rank_pct=97.2,
            explanation="Similar profile, shared witness {a, b}, converging.",
            component_scores={
                "delta": 0.368, "witness": 0.201,
                "trajectory": 0.142, "anomaly": 0.10,
            },
        )
        assert member.primary_key == "ACC-001"
        assert member.score == pytest.approx(0.84)
        assert member.is_anomaly
        assert member.trajectory_alignment == pytest.approx(0.71)

    def test_member_is_frozen(self):
        member = CohortMember(
            primary_key="X", score=0.5, delta_similarity=0.5,
            witness_overlap=0.5, trajectory_alignment=None,
            is_anomaly=False, delta_rank_pct=50.0,
            explanation="", component_scores={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            member.primary_key = "Y"  # type: ignore[misc]

    def test_member_serializable(self):
        member = CohortMember(
            primary_key="X", score=0.5, delta_similarity=0.5,
            witness_overlap=0.5, trajectory_alignment=None,
            is_anomaly=False, delta_rank_pct=50.0,
            explanation="", component_scores={"delta": 0.5},
        )
        d = dataclasses.asdict(member)
        assert d["primary_key"] == "X"
        assert d["component_scores"] == {"delta": 0.5}

    def test_witness_cohort_result_instantiation(self):
        result = WitnessCohortResult(
            primary_key="X",
            pattern_id="account_pattern",
            edge_pattern_id="tx_pattern",
            members=[],
            excluded_existing_edges=12,
            excluded_low_score=3,
            candidate_pool_size=100,
            weights_used={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
            summary={"max_score": 0.0, "anomaly_count": 0},
        )
        assert result.primary_key == "X"
        assert result.candidate_pool_size == 100
        assert result.members == []


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


class TestWitnessJaccard:
    """GDSEngine.witness_jaccard — Jaccard index on witness dim sets."""

    def test_identical_sets(self):
        assert GDSEngine.witness_jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self):
        assert GDSEngine.witness_jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # |A ∩ B| / |A ∪ B| = 1 / 3
        assert GDSEngine.witness_jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self):
        # Convention: empty ∩ empty / empty ∪ empty → 0 (no signal)
        assert GDSEngine.witness_jaccard(set(), set()) == 0.0

    def test_one_empty(self):
        assert GDSEngine.witness_jaccard({"a"}, set()) == 0.0
        assert GDSEngine.witness_jaccard(set(), {"a"}) == 0.0

    def test_half_overlap(self):
        # 2 / 4
        assert GDSEngine.witness_jaccard({"a", "b", "c"}, {"a", "b", "d", "e"}) == pytest.approx(0.4)


class TestTrajectoryCosine:
    """GDSEngine.trajectory_cosine — cosine sim remapped to [0, 1]."""

    def test_identical_trajectories(self):
        traj = np.array([1.0, 2.0, 3.0])
        assert GDSEngine.trajectory_cosine(traj, traj) == pytest.approx(1.0)

    def test_opposite_trajectories(self):
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        # cos = -1 → remapped → 0
        assert GDSEngine.trajectory_cosine(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_trajectories(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        # cos = 0 → remapped → 0.5
        assert GDSEngine.trajectory_cosine(a, b) == pytest.approx(0.5)

    def test_zero_norm_trajectory(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        # Convention: zero-norm → no signal → 0.5 (neutral)
        assert GDSEngine.trajectory_cosine(a, b) == pytest.approx(0.5)


class TestCompositeLinkScore:
    """GDSEngine.composite_link_score — blends 4 components with weight handling."""

    def test_max_signal_all_components(self):
        score, components = GDSEngine.composite_link_score(
            delta_similarity=1.0,
            witness_overlap=1.0,
            trajectory_alignment=1.0,
            anomaly_bonus=1.0,
            weights={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
        )
        assert score == pytest.approx(1.0)
        assert components["delta"] == pytest.approx(0.4)
        assert components["witness"] == pytest.approx(0.3)
        assert components["trajectory"] == pytest.approx(0.2)
        assert components["anomaly"] == pytest.approx(0.1)

    def test_zero_signal(self):
        score, _ = GDSEngine.composite_link_score(
            delta_similarity=0.0,
            witness_overlap=0.0,
            trajectory_alignment=0.0,
            anomaly_bonus=0.0,
            weights={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
        )
        assert score == pytest.approx(0.0)

    def test_trajectory_none_renormalizes_weights(self):
        score, components = GDSEngine.composite_link_score(
            delta_similarity=1.0,
            witness_overlap=1.0,
            trajectory_alignment=None,
            anomaly_bonus=1.0,
            weights={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
        )
        # trajectory weight (0.2) redistributed proportionally:
        # delta: 0.4 / 0.8 = 0.50
        # witness: 0.3 / 0.8 = 0.375
        # anomaly: 0.1 / 0.8 = 0.125
        assert score == pytest.approx(1.0)
        assert "trajectory" not in components
        assert components["delta"] == pytest.approx(0.50)
        assert components["witness"] == pytest.approx(0.375)
        assert components["anomaly"] == pytest.approx(0.125)

    def test_partial_signal(self):
        score, _ = GDSEngine.composite_link_score(
            delta_similarity=0.8,
            witness_overlap=0.5,
            trajectory_alignment=0.6,
            anomaly_bonus=0.0,
            weights={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
        )
        # 0.4*0.8 + 0.3*0.5 + 0.2*0.6 + 0.1*0 = 0.32 + 0.15 + 0.12 + 0 = 0.59
        assert score == pytest.approx(0.59)

    def test_score_bounded_in_unit_interval(self):
        # Random values, weights sum to 1 — score must stay in [0, 1]
        rng = np.random.default_rng(42)
        for _ in range(50):
            d, w, t, a = rng.uniform(0, 1, 4)
            score, _ = GDSEngine.composite_link_score(
                delta_similarity=float(d),
                witness_overlap=float(w),
                trajectory_alignment=float(t),
                anomaly_bonus=float(a),
                weights={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
            )
            assert 0.0 <= score <= 1.0

    def test_monotonic_in_each_component(self):
        # Increasing one component while keeping others fixed must not decrease score
        baseline, _ = GDSEngine.composite_link_score(
            delta_similarity=0.5, witness_overlap=0.5,
            trajectory_alignment=0.5, anomaly_bonus=0.0,
            weights={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
        )
        higher_delta, _ = GDSEngine.composite_link_score(
            delta_similarity=0.9, witness_overlap=0.5,
            trajectory_alignment=0.5, anomaly_bonus=0.0,
            weights={"delta": 0.4, "witness": 0.3, "trajectory": 0.2, "anomaly": 0.1},
        )
        assert higher_delta > baseline
