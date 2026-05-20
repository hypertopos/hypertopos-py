# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""M1.2 — Per-edge counterfactual.

Tests confirm the algorithm with HAND-VERIFIED math on a synthetic 3-dim
polygon. No tolerance shortcuts — exact comparison (within float32 rounding)
against closed-form expected values.

Synthetic pattern setup (used across most tests):

    Pattern has 3 relations dims (line "A", line "B", line "C"), all gaussian.
        mu          = [2.0, 1.5, 0.5]
        sigma_diag  = [1.0, 0.5, 0.5]

    Entity shape (raw aggregates): [4.0, 3.0, 1.0]
        delta = ([4-2]/1, [3-1.5]/0.5, [1-0.5]/0.5) = [2.0, 3.0, 1.0]
        delta_norm = sqrt(4 + 9 + 1) = sqrt(14) ≈ 3.7416574

    Entity edges in line "A" (out direction, contributes to relation A):
        E1 (partner X), E2 (partner Y) — both out-direction in line A
    Entity edges in line "B" (in direction, contributes to relation B):
        E3 (partner Z) — in-direction in line B
    Entity edge in line "B" (out direction, does NOT contribute to relation B):
        E4 (partner W) — out-direction in line B (line B configured for "in" only)

    Remove E1:
        new_shape = [3.0, 3.0, 1.0]
        new_delta = [1.0, 3.0, 1.0]
        new_delta_norm = sqrt(1 + 9 + 1) = sqrt(11) ≈ 3.3166248
        drop_pct = (3.7416574 - 3.3166248) / 3.7416574 ≈ 11.3582%

    Remove E3:
        new_shape = [4.0, 2.0, 1.0]
        new_delta = [2.0, 1.0, 1.0]
        new_delta_norm = sqrt(4 + 1 + 1) = sqrt(6) ≈ 2.4494897
        drop_pct = (3.7416574 - 2.4494897) / 3.7416574 ≈ 34.5347%
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ── Shared synthetic fixture ────────────────────────────────────────────────

_MU = np.array([2.0, 1.5, 0.5], dtype=np.float32)
_SIGMA = np.array([1.0, 0.5, 0.5], dtype=np.float32)
_SHAPE = np.array([4.0, 3.0, 1.0], dtype=np.float32)
_DELTA_NORM_BEFORE = float(math.sqrt(14.0))  # ≈ 3.7416574


def _edges(specs: list[tuple[str, str, str, str]]) -> list[dict]:
    """Build edge records: (edge_id, partner_key, direction, line_id)."""
    return [
        {
            "edge_id": eid,
            "partner_key": partner,
            "direction": direction,
            "line_id": line,
        }
        for eid, partner, direction, line in specs
    ]


def _relations(specs: list[tuple[str, str]]) -> list[dict]:
    """Build relation defs: (line_id, direction)."""
    return [
        {"line_id": line, "direction": direction}
        for line, direction in specs
    ]


# ── Tests ──────────────────────────────────────────────────────────────────


def test_relations_counterfactual_hand_computed():
    """Exact math match: remove E1, expect drop_pct ≈ 11.358%."""
    from hypertopos.engine.counterfactual import simulate_edge_removal_naive

    relations = _relations([("A", "out"), ("B", "in"), ("C", "out")])
    edges = _edges([
        ("E1", "X", "out", "A"),
        ("E2", "Y", "out", "A"),
        ("E3", "Z", "in", "B"),
        ("E4", "W", "out", "B"),  # mismatched direction → no contribution
    ])

    result = simulate_edge_removal_naive(
        shape=_SHAPE.copy(),
        mu=_MU,
        sigma_diag=_SIGMA,
        delta_norm=_DELTA_NORM_BEFORE,
        edges=edges,
        relations=relations,
        candidate_edge_ids=["E1"],
        top_n=10,
    )

    assert len(result) == 1
    row = result[0]
    assert row["edge_id"] == "E1"
    expected_drop_pct = (math.sqrt(14.0) - math.sqrt(11.0)) / math.sqrt(14.0) * 100
    assert row["drop_pct"] == pytest.approx(expected_drop_pct, abs=0.01)
    assert row["delta_norm_before"] == pytest.approx(_DELTA_NORM_BEFORE, abs=1e-5)
    assert row["delta_norm_after"] == pytest.approx(math.sqrt(11.0), abs=1e-5)
    assert row["dominant_dim_idx"] == 0  # relation A is the one that changed


def test_e3_removes_relation_b_in_direction():
    """E3 is in-direction on line B, relation B is configured 'in' — must
    contribute. Removing E3 changes dim 1 (relation B)."""
    from hypertopos.engine.counterfactual import simulate_edge_removal_naive

    relations = _relations([("A", "out"), ("B", "in"), ("C", "out")])
    edges = _edges([("E3", "Z", "in", "B")])

    result = simulate_edge_removal_naive(
        shape=_SHAPE.copy(),
        mu=_MU,
        sigma_diag=_SIGMA,
        delta_norm=_DELTA_NORM_BEFORE,
        edges=edges,
        relations=relations,
        candidate_edge_ids=None,
        top_n=5,
    )

    assert len(result) == 1
    row = result[0]
    expected_drop_pct = (math.sqrt(14.0) - math.sqrt(6.0)) / math.sqrt(14.0) * 100
    assert row["drop_pct"] == pytest.approx(expected_drop_pct, abs=0.01)
    assert row["dominant_dim_idx"] == 1  # relation B is the one that changed


def test_direction_mismatch_zero_contribution():
    """E4 is out-direction on line B, but relation B is configured 'in' —
    must NOT contribute. The edge is in `edges` but produces drop_pct ≈ 0."""
    from hypertopos.engine.counterfactual import simulate_edge_removal_naive

    relations = _relations([("A", "out"), ("B", "in"), ("C", "out")])
    edges = _edges([("E4", "W", "out", "B")])  # direction mismatch

    result = simulate_edge_removal_naive(
        shape=_SHAPE.copy(),
        mu=_MU,
        sigma_diag=_SIGMA,
        delta_norm=_DELTA_NORM_BEFORE,
        edges=edges,
        relations=relations,
        candidate_edge_ids=None,
        top_n=5,
    )

    assert len(result) == 1
    assert result[0]["drop_pct"] == pytest.approx(0.0, abs=1e-5)
    assert result[0]["delta_norm_after"] == pytest.approx(_DELTA_NORM_BEFORE, abs=1e-5)


def test_edge_ids_filter():
    """`candidate_edge_ids=['E1', 'E3']` returns exactly those two (in
    drop_pct order), not E2 or E4."""
    from hypertopos.engine.counterfactual import simulate_edge_removal_naive

    relations = _relations([("A", "out"), ("B", "in"), ("C", "out")])
    edges = _edges([
        ("E1", "X", "out", "A"),
        ("E2", "Y", "out", "A"),
        ("E3", "Z", "in", "B"),
        ("E4", "W", "out", "B"),
    ])

    result = simulate_edge_removal_naive(
        shape=_SHAPE.copy(),
        mu=_MU,
        sigma_diag=_SIGMA,
        delta_norm=_DELTA_NORM_BEFORE,
        edges=edges,
        relations=relations,
        candidate_edge_ids=["E1", "E3"],
        top_n=10,
    )

    returned_ids = {r["edge_id"] for r in result}
    assert returned_ids == {"E1", "E3"}
    # E3 has bigger drop_pct than E1, so E3 should be first
    assert result[0]["edge_id"] == "E3"
    assert result[1]["edge_id"] == "E1"


def test_top_n_truncates_after_ranking():
    """top_n=1 returns only the highest-drop_pct edge (E3 from synthetic set)."""
    from hypertopos.engine.counterfactual import simulate_edge_removal_naive

    relations = _relations([("A", "out"), ("B", "in"), ("C", "out")])
    edges = _edges([
        ("E1", "X", "out", "A"),
        ("E2", "Y", "out", "A"),
        ("E3", "Z", "in", "B"),
    ])

    result = simulate_edge_removal_naive(
        shape=_SHAPE.copy(),
        mu=_MU,
        sigma_diag=_SIGMA,
        delta_norm=_DELTA_NORM_BEFORE,
        edges=edges,
        relations=relations,
        candidate_edge_ids=None,
        top_n=1,
    )

    assert len(result) == 1
    assert result[0]["edge_id"] == "E3"  # biggest drop


def test_no_simulatable_dims_returns_empty():
    """Entity has edges but pattern has NO relations matching their line_id —
    no dim can be simulated. Return empty list."""
    from hypertopos.engine.counterfactual import simulate_edge_removal_naive

    relations = _relations([("X", "out"), ("Y", "in")])  # entirely different lines
    edges = _edges([("E1", "P", "out", "A"), ("E2", "Q", "in", "B")])

    result = simulate_edge_removal_naive(
        shape=np.array([1.0, 1.0], dtype=np.float32),
        mu=np.array([0.0, 0.0], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0], dtype=np.float32),
        delta_norm=math.sqrt(2.0),
        edges=edges,
        relations=relations,
        candidate_edge_ids=None,
        top_n=5,
    )

    assert result == []


def test_v1_mean_aggregation_hand_computed():
    """v1 path with one edge_dim_aggregation `(pair_edge_count, mean)`.
    Entity has 3 events with pair_edge_count values [2.0, 4.0, 6.0].
    mean = 4.0. Remove event with value 6.0 → new_mean = (2.0+4.0)/2 = 3.0.

    Pattern dim 0 carries the mean value.
    Pattern: mu=[3.0], sigma=[1.0], shape=[4.0]
        delta = [(4-3)/1] = [1.0], delta_norm = 1.0
    Remove event3 (value=6.0):
        new_shape = [3.0] (the new mean)
        new_delta = [(3-3)/1] = [0.0]
        new_delta_norm = 0.0
        drop_pct = 100%
    """
    from hypertopos.engine.counterfactual import (
        simulate_edge_removal_with_aggregations,
    )
    edges = [
        {"edge_id": "ev1", "partner_key": "X", "direction": "out", "line_id": "L"},
        {"edge_id": "ev2", "partner_key": "Y", "direction": "out", "line_id": "L"},
        {"edge_id": "ev3", "partner_key": "Z", "direction": "out", "line_id": "L"},
    ]

    result = simulate_edge_removal_with_aggregations(
        shape=np.array([4.0], dtype=np.float32),
        mu=np.array([3.0], dtype=np.float32),
        sigma_diag=np.array([1.0], dtype=np.float32),
        delta_norm=1.0,
        edges=edges,
        relations=[],
        edge_agg_dim_offset=0,
        edge_agg_specs=[("pair_edge_count", "mean")],
        event_source_values={
            "ev1": {"pair_edge_count": 2.0},
            "ev2": {"pair_edge_count": 4.0},
            "ev3": {"pair_edge_count": 6.0},
        },
        candidate_edge_ids=["ev3"],
        top_n=5,
    )
    assert len(result) == 1
    row = result[0]
    assert row["edge_id"] == "ev3"
    assert row["delta_norm_after"] == pytest.approx(0.0, abs=1e-5)
    assert row["drop_pct"] == pytest.approx(100.0, abs=0.01)


def test_v1_max_aggregation_rescan():
    """Max aggregation: removing the current max edge should drop to second-max.
    Values [10, 20, 30, 40] → max=40. Remove the 40 edge → new_max=30."""
    from hypertopos.engine.counterfactual import (
        simulate_edge_removal_with_aggregations,
    )
    edges = [
        {"edge_id": "e1", "partner_key": "A", "direction": "out", "line_id": "L"},
        {"edge_id": "e2", "partner_key": "B", "direction": "out", "line_id": "L"},
        {"edge_id": "e3", "partner_key": "C", "direction": "out", "line_id": "L"},
        {"edge_id": "e4", "partner_key": "D", "direction": "out", "line_id": "L"},
    ]
    # shape = [40.0], mu = [20.0], sigma = [10.0] → delta = [2.0], delta_norm = 2.0
    # Remove e4 (value 40): new_max = 30, new_delta = (30-20)/10 = 1.0,
    # new_delta_norm = 1.0, drop_pct = 50%
    result = simulate_edge_removal_with_aggregations(
        shape=np.array([40.0], dtype=np.float32),
        mu=np.array([20.0], dtype=np.float32),
        sigma_diag=np.array([10.0], dtype=np.float32),
        delta_norm=2.0,
        edges=edges,
        relations=[],
        edge_agg_dim_offset=0,
        edge_agg_specs=[("amount", "max")],
        event_source_values={
            "e1": {"amount": 10.0},
            "e2": {"amount": 20.0},
            "e3": {"amount": 30.0},
            "e4": {"amount": 40.0},
        },
        candidate_edge_ids=["e4"],
        top_n=5,
    )
    assert result[0]["delta_norm_after"] == pytest.approx(1.0, abs=1e-5)
    assert result[0]["drop_pct"] == pytest.approx(50.0, abs=0.01)


def test_v1_std_aggregation_rescan():
    """Std aggregation (ddof=0): values [1,2,3,4,5], std = sqrt(2.0) ≈ 1.414.
    Remove value 5 → values [1,2,3,4], std = sqrt(1.25) ≈ 1.118.
    Verify std rescan via _aggregate matches numpy.std(ddof=0)."""
    from hypertopos.engine.counterfactual import (
        simulate_edge_removal_with_aggregations,
    )
    edges = [
        {"edge_id": f"e{i}", "partner_key": f"P{i}", "direction": "out", "line_id": "L"}
        for i in range(1, 6)
    ]
    pop_std = float(np.std(np.array([1, 2, 3, 4, 5]), ddof=0))  # sqrt(2)
    new_std = float(np.std(np.array([1, 2, 3, 4]), ddof=0))  # sqrt(1.25)
    # mu=0, sigma=1 → shape = std value directly → delta = std
    result = simulate_edge_removal_with_aggregations(
        shape=np.array([pop_std], dtype=np.float32),
        mu=np.array([0.0], dtype=np.float32),
        sigma_diag=np.array([1.0], dtype=np.float32),
        delta_norm=pop_std,
        edges=edges,
        relations=[],
        edge_agg_dim_offset=0,
        edge_agg_specs=[("amount", "std")],
        event_source_values={f"e{i}": {"amount": float(i)} for i in range(1, 6)},
        candidate_edge_ids=["e5"],
        top_n=5,
    )
    assert result[0]["delta_norm_after"] == pytest.approx(new_std, abs=1e-4)


def test_v1_count_above_threshold_held_constant():
    """count_above_threshold not in _SUPPORTED_AGGS → dim reported in
    dimensions_skipped, shape held constant for that agg dim."""
    from hypertopos.engine.counterfactual import (
        simulate_edge_removal_with_aggregations,
    )
    edges = [
        {"edge_id": "e1", "partner_key": "A", "direction": "out", "line_id": "L"},
    ]
    result = simulate_edge_removal_with_aggregations(
        shape=np.array([2.0, 3.0], dtype=np.float32),
        mu=np.array([0.0, 0.0], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0], dtype=np.float32),
        delta_norm=math.sqrt(13.0),
        edges=edges,
        relations=[],
        edge_agg_dim_offset=0,
        edge_agg_specs=[
            ("pair_count", "mean"),
            ("pair_count", "count_above_threshold"),
        ],
        event_source_values={"e1": {"pair_count": 5.0}},
        candidate_edge_ids=None,
        top_n=5,
    )
    assert 1 in result[0]["dimensions_skipped"]  # count_above_threshold dim
    assert 0 not in result[0]["dimensions_skipped"]  # mean is supported


def test_v1_combined_relations_and_aggregations():
    """Full path: 2 relations + 1 edge_dim_agg dim, hand-verified.

    Pattern:
        dim 0: relation A, direction out (count-based)
        dim 1: relation B, direction out (count-based)
        dim 2: edge_dim_agg (pair_count, mean)

    mu = [1.0, 0.0, 2.0], sigma = [1.0, 1.0, 1.0]
    shape = [3.0, 1.0, 5.0] (entity has 3 A-out edges, 1 B-out edge, mean pair_count=5)
    delta = [2.0, 1.0, 3.0], delta_norm = sqrt(4+1+9) = sqrt(14) ≈ 3.7417

    Remove edge E1 (out-A, pair_count=8):
        - relation A contribution: -1 → new_shape[0] = 2.0
        - edge_dim_agg: mean of remaining 2 events. Total events with same edge: 3 events?
        - Actually for cleanness: entity has 3 events e1,e2,e3 with pair_count [8, 4, 3]
        - Current mean = (8+4+3)/3 = 5.0 ✓
        - Remove e1 → new_mean = (4+3)/2 = 3.5
    new_shape = [2.0, 1.0, 3.5]
    new_delta = [(2-1)/1, (1-0)/1, (3.5-2)/1] = [1.0, 1.0, 1.5]
    new_delta_norm = sqrt(1+1+2.25) = sqrt(4.25) ≈ 2.0616
    drop_pct = (3.7417 - 2.0616)/3.7417 ≈ 44.9%
    """
    from hypertopos.engine.counterfactual import (
        simulate_edge_removal_with_aggregations,
    )
    relations = [
        {"line_id": "A", "direction": "out"},
        {"line_id": "B", "direction": "out"},
    ]
    edges = [
        {"edge_id": "e1", "partner_key": "X", "direction": "out", "line_id": "A"},
        {"edge_id": "e2", "partner_key": "Y", "direction": "out", "line_id": "A"},
        {"edge_id": "e3", "partner_key": "Z", "direction": "out", "line_id": "A"},
    ]
    result = simulate_edge_removal_with_aggregations(
        shape=np.array([3.0, 1.0, 5.0], dtype=np.float32),
        mu=np.array([1.0, 0.0, 2.0], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        delta_norm=math.sqrt(14.0),
        edges=edges,
        relations=relations,
        edge_agg_dim_offset=2,
        edge_agg_specs=[("pair_count", "mean")],
        event_source_values={
            "e1": {"pair_count": 8.0},
            "e2": {"pair_count": 4.0},
            "e3": {"pair_count": 3.0},
        },
        candidate_edge_ids=["e1"],
        top_n=5,
    )
    assert len(result) == 1
    expected_new_delta_norm = math.sqrt(1.0 + 1.0 + 2.25)
    expected_drop_pct = (
        (math.sqrt(14.0) - expected_new_delta_norm) / math.sqrt(14.0) * 100
    )
    assert result[0]["delta_norm_after"] == pytest.approx(expected_new_delta_norm, abs=1e-4)
    assert result[0]["drop_pct"] == pytest.approx(expected_drop_pct, abs=0.01)


def test_zero_sigma_dim_skipped():
    """A dim with sigma_diag ≈ 0 is undefined under z-score normalisation;
    should be reported in `dimensions_skipped` and held constant in shape."""
    from hypertopos.engine.counterfactual import simulate_edge_removal_naive

    relations = _relations([("A", "out"), ("B", "in")])
    edges = _edges([("E1", "X", "out", "A")])
    sigma_with_zero = np.array([1.0, 0.0], dtype=np.float32)  # dim 1 is dead

    result = simulate_edge_removal_naive(
        shape=np.array([3.0, 1.0], dtype=np.float32),
        mu=np.array([1.0, 0.0], dtype=np.float32),
        sigma_diag=sigma_with_zero,
        delta_norm=math.sqrt(4.0),
        edges=edges,
        relations=relations,
        candidate_edge_ids=None,
        top_n=5,
    )

    assert len(result) == 1
    row = result[0]
    assert 1 in row["dimensions_skipped"]
    assert 0 in row["dimensions_simulated"]


# ── Phase 1.C: count_above_threshold via population threshold ───────────────


def test_v1_count_above_threshold_with_threshold():
    """When threshold supplied, count_above_threshold IS simulated.

    Setup: 4 edges, source dim "pair_count" values = [15.0, 20.0, 5.0, 12.0].
    Threshold = 10.0. Baseline count_above = 3 (values 15, 20, 12 > 10).
    Shape[0] = baseline count = 3.0; mu=0, sigma=1 → delta_norm = 3.0.

    Remove e1 (val 15): [20, 5, 12] → count = 2 → new_shape[0] = 2.0,
        new_delta_norm = 2.0, drop_pct = 33.33%.
    Remove e3 (val 5):  [15, 20, 12] → count = 3 → unchanged, drop_pct = 0.
    """
    from hypertopos.engine.counterfactual import (
        simulate_edge_removal_with_aggregations,
    )
    edges = [
        {"edge_id": "e1", "partner_key": "A", "direction": "out", "line_id": "L"},
        {"edge_id": "e2", "partner_key": "B", "direction": "out", "line_id": "L"},
        {"edge_id": "e3", "partner_key": "C", "direction": "out", "line_id": "L"},
        {"edge_id": "e4", "partner_key": "D", "direction": "out", "line_id": "L"},
    ]
    result = simulate_edge_removal_with_aggregations(
        shape=np.array([3.0], dtype=np.float32),
        mu=np.array([0.0], dtype=np.float32),
        sigma_diag=np.array([1.0], dtype=np.float32),
        delta_norm=3.0,
        edges=edges,
        relations=[],
        edge_agg_dim_offset=0,
        edge_agg_specs=[("pair_count", "count_above_threshold")],
        event_source_values={
            "e1": {"pair_count": 15.0},
            "e2": {"pair_count": 20.0},
            "e3": {"pair_count": 5.0},
            "e4": {"pair_count": 12.0},
        },
        candidate_edge_ids=None,
        top_n=5,
        thresholds={"pair_count": 10.0},
    )
    # All 4 edges scored, none skipped (dim 0 simulated)
    assert len(result) == 4
    for r in result:
        assert r["dimensions_skipped"] == []
        assert r["dimensions_simulated"] == [0]
    by_edge = {r["edge_id"]: r for r in result}
    # e1 (val=15, above threshold): removing it drops count 3→2 → drop_pct = 33.33%
    assert by_edge["e1"]["delta_norm_after"] == pytest.approx(2.0, abs=1e-4)
    assert by_edge["e1"]["drop_pct"] == pytest.approx(33.333333, abs=0.01)
    # e3 (val=5, below threshold): removing it doesn't change count → drop_pct = 0
    assert by_edge["e3"]["delta_norm_after"] == pytest.approx(3.0, abs=1e-4)
    assert by_edge["e3"]["drop_pct"] == pytest.approx(0.0, abs=0.01)


def test_v1_threshold_missing_keeps_count_above_in_skipped():
    """No threshold for a source_dim → count_above_threshold classified as
    unsupported (back-compat with thresholds=None call site).

    Edge has no other simulatable dims AND no relation match → early-return
    contract emits empty list (same as v0 no-simulatable-surface case).
    Verifies the engine doesn't crash and doesn't fabricate results when
    the only declared agg can't be computed.
    """
    from hypertopos.engine.counterfactual import (
        simulate_edge_removal_with_aggregations,
    )
    edges = [
        {"edge_id": "e1", "partner_key": "A", "direction": "out", "line_id": "L"},
    ]
    # No relations match AND only agg is unsupported → empty result.
    result = simulate_edge_removal_with_aggregations(
        shape=np.array([2.0], dtype=np.float32),
        mu=np.array([0.0], dtype=np.float32),
        sigma_diag=np.array([1.0], dtype=np.float32),
        delta_norm=2.0,
        edges=edges,
        relations=[],
        edge_agg_dim_offset=0,
        edge_agg_specs=[("pair_count", "count_above_threshold")],
        event_source_values={"e1": {"pair_count": 50.0}},
        candidate_edge_ids=None,
        top_n=5,
        thresholds=None,
    )
    assert result == []
    # When a supported agg coexists with an unsupported count_above_threshold,
    # the unsupported dim must surface in dimensions_skipped, the supported
    # one must NOT — keeps the back-compat semantics visible to the caller.
    result2 = simulate_edge_removal_with_aggregations(
        shape=np.array([2.0, 5.0], dtype=np.float32),
        mu=np.array([0.0, 0.0], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0], dtype=np.float32),
        delta_norm=math.sqrt(4 + 25),
        edges=edges,
        relations=[],
        edge_agg_dim_offset=0,
        edge_agg_specs=[
            ("pair_count", "mean"),                    # supported
            ("pair_count", "count_above_threshold"),   # unsupported (no thr)
        ],
        event_source_values={"e1": {"pair_count": 50.0}},
        candidate_edge_ids=None,
        top_n=5,
        thresholds=None,
    )
    assert len(result2) == 1
    assert 0 not in result2[0]["dimensions_skipped"]  # mean is simulated
    assert 1 in result2[0]["dimensions_skipped"]      # count_above skipped


# ── Phase 1.A: per-counterparty rollup ──────────────────────────────────────


def test_counterparty_rollup_aggregates_by_partner():
    """Group per-edge results by partner_key — sum, max, n_edges per partner.

    Three partners, six edges:
        partner X: e1 (drop=5.0), e2 (drop=3.0), e3 (drop=-2.0)
            → sum=6.0, sum_abs=10.0, max_abs=5.0, n=3
        partner Y: e4 (drop=10.0), e5 (drop=4.0)
            → sum=14.0, sum_abs=14.0, max_abs=10.0, n=2
        partner Z: e6 (drop=1.0)
            → sum=1.0, sum_abs=1.0, max_abs=1.0, n=1

    Sorted by sum_abs descending: Y (14), X (10), Z (1).
    """
    from hypertopos.engine.counterfactual import (
        aggregate_edge_removals_by_counterparty,
    )
    per_edge = [
        {"edge_id": "e1", "edge_partner_key": "X", "drop_pct": 5.0,
         "dominant_dim_label": "rel_A"},
        {"edge_id": "e2", "edge_partner_key": "X", "drop_pct": 3.0,
         "dominant_dim_label": "rel_A"},
        {"edge_id": "e3", "edge_partner_key": "X", "drop_pct": -2.0,
         "dominant_dim_label": "rel_B"},
        {"edge_id": "e4", "edge_partner_key": "Y", "drop_pct": 10.0,
         "dominant_dim_label": "rel_A"},
        {"edge_id": "e5", "edge_partner_key": "Y", "drop_pct": 4.0,
         "dominant_dim_label": "rel_C"},
        {"edge_id": "e6", "edge_partner_key": "Z", "drop_pct": 1.0,
         "dominant_dim_label": "rel_A"},
    ]
    result = aggregate_edge_removals_by_counterparty(per_edge, top_n=10)
    assert len(result) == 3
    by_partner = {r["partner_key"]: r for r in result}

    assert by_partner["Y"]["n_edges"] == 2
    assert by_partner["Y"]["sum_drop_pct"] == pytest.approx(14.0)
    assert by_partner["Y"]["sum_abs_drop_pct"] == pytest.approx(14.0)
    assert by_partner["Y"]["max_abs_drop_pct"] == pytest.approx(10.0)
    assert set(by_partner["Y"]["edge_ids"]) == {"e4", "e5"}

    assert by_partner["X"]["n_edges"] == 3
    assert by_partner["X"]["sum_drop_pct"] == pytest.approx(6.0)
    assert by_partner["X"]["sum_abs_drop_pct"] == pytest.approx(10.0)
    assert by_partner["X"]["max_abs_drop_pct"] == pytest.approx(5.0)

    assert by_partner["Z"]["n_edges"] == 1

    # Sorted by sum_abs_drop_pct desc → Y first
    assert result[0]["partner_key"] == "Y"
    assert result[1]["partner_key"] == "X"
    assert result[2]["partner_key"] == "Z"


def test_counterparty_rollup_top_n_truncates():
    """top_n caps returned partners; ranking by sum_abs_drop_pct."""
    from hypertopos.engine.counterfactual import (
        aggregate_edge_removals_by_counterparty,
    )
    per_edge = [
        {"edge_id": f"e{i}", "edge_partner_key": f"P{i}",
         "drop_pct": float(10 - i), "dominant_dim_label": "rel_A"}
        for i in range(5)
    ]
    result = aggregate_edge_removals_by_counterparty(per_edge, top_n=2)
    assert len(result) == 2
    assert result[0]["partner_key"] == "P0"  # drop=10.0
    assert result[1]["partner_key"] == "P1"  # drop=9.0


def test_counterparty_rollup_dominant_dim_is_per_partner_max():
    """dominant_dim_label of a partner = label of THAT partner's worst edge."""
    from hypertopos.engine.counterfactual import (
        aggregate_edge_removals_by_counterparty,
    )
    per_edge = [
        {"edge_id": "e1", "edge_partner_key": "X", "drop_pct": 3.0,
         "dominant_dim_label": "rel_A"},
        {"edge_id": "e2", "edge_partner_key": "X", "drop_pct": -8.0,
         "dominant_dim_label": "rel_B"},  # max-abs for X
    ]
    result = aggregate_edge_removals_by_counterparty(per_edge, top_n=5)
    assert len(result) == 1
    assert result[0]["dominant_dim_label"] == "rel_B"


def test_counterparty_rollup_handles_missing_partner():
    """edge_partner_key is None for self-edges → grouped under '__unknown__'."""
    from hypertopos.engine.counterfactual import (
        aggregate_edge_removals_by_counterparty,
    )
    per_edge = [
        {"edge_id": "e1", "edge_partner_key": None, "drop_pct": 2.0,
         "dominant_dim_label": "rel_A"},
        {"edge_id": "e2", "edge_partner_key": "X", "drop_pct": 1.0,
         "dominant_dim_label": "rel_A"},
    ]
    result = aggregate_edge_removals_by_counterparty(per_edge, top_n=5)
    assert len(result) == 2
    partners = {r["partner_key"] for r in result}
    assert partners == {"__unknown__", "X"}


def test_counterparty_rollup_dedup_same_edge_both_directions():
    """Same edge_id appearing under multiple direction entries (multi-graph)
    must count ONCE per partner — not double."""
    from hypertopos.engine.counterfactual import (
        aggregate_edge_removals_by_counterparty,
    )
    per_edge = [
        {"edge_id": "TX-001", "edge_partner_key": "X", "drop_pct": 5.0,
         "dominant_dim_label": "rel_A"},
        {"edge_id": "TX-001", "edge_partner_key": "X", "drop_pct": 5.0,
         "dominant_dim_label": "rel_A"},  # duplicate
    ]
    result = aggregate_edge_removals_by_counterparty(per_edge, top_n=5)
    assert len(result) == 1
    assert result[0]["n_edges"] == 1
    assert result[0]["sum_drop_pct"] == pytest.approx(5.0)


# ── Phase 2.D: per-edge source-value ECDF significance ──────────────────────


def test_ecdf_pvalue_upper_tail_basic():
    """Upper-tail ECDF p-value: P(V >= v) on a small population.

    Population sorted = [1, 2, 3, 4, 5] (N=5). For edge value v:
        v = 5  → 1 value ≥ 5 in pop → p = 1/5 = 0.20
        v = 3  → 3 values ≥ 3 in pop → p = 3/5 = 0.60
        v = 100 → 0 values ≥ 100 → p = 1/(N+1) (Phipson-Smyth floor) = 1/6 ≈ 0.167
            (NB: floor prevents p=0; documented in docstring)
        v = 0  → 5 values ≥ 0 → p = 5/5 = 1.0
    """
    from hypertopos.engine.counterfactual import (
        ecdf_pvalue_upper_tail,
    )
    pop_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ecdf_pvalue_upper_tail(5.0, pop_sorted) == pytest.approx(0.20)
    assert ecdf_pvalue_upper_tail(3.0, pop_sorted) == pytest.approx(0.60)
    assert ecdf_pvalue_upper_tail(0.0, pop_sorted) == pytest.approx(1.0)
    # Phipson-Smyth-style floor: never zero — extreme value above max returns
    # 1/(N+1) instead of 0 so HMP downstream doesn't crash on log(0).
    assert ecdf_pvalue_upper_tail(100.0, pop_sorted) == pytest.approx(1.0 / 6.0)


def test_ecdf_pvalue_handles_ties_correctly():
    """v exactly at population boundary value: rank counts equal-value entries.

    Population [1, 2, 3, 3, 3, 5] (N=6). v=3 → 4 values ≥ 3 → p = 4/6 ≈ 0.667.
    """
    from hypertopos.engine.counterfactual import ecdf_pvalue_upper_tail
    pop_sorted = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 5.0])
    assert ecdf_pvalue_upper_tail(3.0, pop_sorted) == pytest.approx(4.0 / 6.0)


def test_compute_per_edge_pvalues_picks_min_across_dims():
    """Per-edge p-value across multiple source dims uses MIN (most extreme).

    Two source dims:
        dim_X: pop = [0..99]  (N=100)
        dim_Y: pop = [0..9]   (N=10)

    Edge e1: dim_X = 95 (top 5/100 = 0.05), dim_Y = 5 (5/10+1 = ... actually
        5 ≤ values: {5,6,7,8,9} = 5 → p = 5/10 = 0.50)
        → min_pvalue = 0.05 (dim_X)

    Edge e2: dim_X = 50 (top 50/100 = 0.50), dim_Y = 9 (top 1/10 = 0.10)
        → min_pvalue = 0.10 (dim_Y)

    e1 ranks MORE extreme (lower min_pvalue).
    """
    from hypertopos.engine.counterfactual import (
        compute_per_edge_source_value_pvalues,
    )
    pop_ecdfs = {
        "dim_X": np.arange(100, dtype=np.float64),
        "dim_Y": np.arange(10, dtype=np.float64),
    }
    edges = [
        {"edge_id": "e1", "partner_key": "A", "direction": "out", "line_id": "L"},
        {"edge_id": "e2", "partner_key": "B", "direction": "out", "line_id": "L"},
    ]
    event_source_values = {
        "e1": {"dim_X": 95.0, "dim_Y": 5.0},
        "e2": {"dim_X": 50.0, "dim_Y": 9.0},
    }
    result = compute_per_edge_source_value_pvalues(
        edges=edges,
        event_source_values=event_source_values,
        population_ecdfs=pop_ecdfs,
        source_dims=["dim_X", "dim_Y"],
    )
    assert set(result.keys()) == {"e1", "e2"}
    # e1: dim_X has 5 values ≥ 95 → 5/100 = 0.05
    assert result["e1"]["dim_X"] == pytest.approx(0.05)
    # e1: dim_Y has 5 values ≥ 5 → 5/10 = 0.50
    assert result["e1"]["dim_Y"] == pytest.approx(0.50)
    assert result["e1"]["min_pvalue"] == pytest.approx(0.05)
    assert result["e1"]["dominant_significance_dim"] == "dim_X"

    # e2: dim_X has 50 values ≥ 50 → 50/100 = 0.50
    assert result["e2"]["dim_X"] == pytest.approx(0.50)
    # e2: dim_Y has 1 value ≥ 9 → 1/10 = 0.10
    assert result["e2"]["dim_Y"] == pytest.approx(0.10)
    assert result["e2"]["min_pvalue"] == pytest.approx(0.10)
    assert result["e2"]["dominant_significance_dim"] == "dim_Y"


def test_compute_per_edge_pvalues_missing_source_value_handled():
    """Edge with no entry in event_source_values defaults to value=0.0
    (matches engine's existing handling). p-value computed against that."""
    from hypertopos.engine.counterfactual import (
        compute_per_edge_source_value_pvalues,
    )
    pop_ecdfs = {"dim_X": np.arange(100, dtype=np.float64)}
    edges = [
        {"edge_id": "e_missing", "partner_key": "A",
         "direction": "out", "line_id": "L"},
    ]
    result = compute_per_edge_source_value_pvalues(
        edges=edges,
        event_source_values={},  # e_missing has no entry
        population_ecdfs=pop_ecdfs,
        source_dims=["dim_X"],
    )
    # value defaults to 0.0; all 100 pop values ≥ 0 → p = 100/100 = 1.0
    assert result["e_missing"]["dim_X"] == pytest.approx(1.0)


def test_simulate_joint_edge_removal_two_edges_hand_computed():
    """Joint removal of TWO edges from the relations-only pattern.

    Same fixture as the v0 single-edge tests (3 relations, A out / B in /
    no C). Entity has edges E1, E2 (out-A), E3 (in-B), E4 (out-B).

    Remove {E1, E2} jointly:
        new_shape = [shape[0] - 2, shape[1], shape[2]] = [2.0, 3.0, 1.0]
        new_delta = [(2-2)/1, (3-1.5)/0.5, (1-0.5)/0.5] = [0.0, 3.0, 1.0]
        new_delta_norm = sqrt(0 + 9 + 1) = sqrt(10) ≈ 3.1623
        drop_pct = (3.7417 - 3.1623) / 3.7417 ≈ 15.49%

    Remove {E1, E3} jointly:
        new_shape = [3.0, 2.0, 1.0]
        new_delta = [1.0, 1.0, 1.0], new_delta_norm = sqrt(3) ≈ 1.7321
        drop_pct ≈ 53.71%
    """
    from hypertopos.engine.counterfactual import simulate_joint_edge_removal
    relations = [
        {"line_id": "A", "direction": "out"},
        {"line_id": "B", "direction": "in"},
        {"line_id": "C", "direction": "out"},
    ]
    edges = [
        {"edge_id": "E1", "partner_key": "X", "direction": "out", "line_id": "A"},
        {"edge_id": "E2", "partner_key": "Y", "direction": "out", "line_id": "A"},
        {"edge_id": "E3", "partner_key": "Z", "direction": "in",  "line_id": "B"},
        {"edge_id": "E4", "partner_key": "W", "direction": "out", "line_id": "B"},
    ]
    shape = np.array([4.0, 3.0, 1.0], dtype=np.float32)
    mu = np.array([2.0, 1.5, 0.5], dtype=np.float32)
    sigma_diag = np.array([1.0, 0.5, 0.5], dtype=np.float32)
    delta_norm = math.sqrt(14.0)

    # {E1, E2}
    res_12 = simulate_joint_edge_removal(
        shape=shape, mu=mu, sigma_diag=sigma_diag, delta_norm=delta_norm,
        edges_to_remove=[edges[0], edges[1]],
        all_edges=edges,
        relations=relations,
        edge_agg_dim_offset=3,
        edge_agg_specs=[],
        event_source_values={},
        thresholds=None,
    )
    expected_norm = math.sqrt(0 + 9 + 1)
    expected_drop = (delta_norm - expected_norm) / delta_norm * 100
    assert res_12["delta_norm_after"] == pytest.approx(expected_norm, abs=1e-4)
    assert res_12["joint_drop_pct"] == pytest.approx(expected_drop, abs=0.01)
    assert set(res_12["removed_edge_ids"]) == {"E1", "E2"}

    # {E1, E3}
    res_13 = simulate_joint_edge_removal(
        shape=shape, mu=mu, sigma_diag=sigma_diag, delta_norm=delta_norm,
        edges_to_remove=[edges[0], edges[2]],
        all_edges=edges,
        relations=relations,
        edge_agg_dim_offset=3,
        edge_agg_specs=[],
        event_source_values={},
        thresholds=None,
    )
    expected_norm_13 = math.sqrt(3)
    expected_drop_13 = (delta_norm - expected_norm_13) / delta_norm * 100
    assert res_13["joint_drop_pct"] == pytest.approx(expected_drop_13, abs=0.01)


def test_simulate_joint_edge_removal_with_aggregation_rescan():
    """Joint removal under edge_dim_aggregations rescans the agg over
    REMAINING events (not just one removed).

    1 relation (line L, out) + 1 edge_dim_agg (pair_count, mean).
    mu = [0, 5], sigma = [1, 1]. Entity has 3 edges E1/E2/E3 (out, line L).
    Their pair_count values: [10, 6, 2]. Mean = 6.
    shape = [3, 6], delta = [3, 1], delta_norm = sqrt(10) ≈ 3.1623.

    Remove {E1, E3}:
        Relations: 3 - 2 = 1 → new_shape[0] = 1
        Mean of remaining [6] = 6 → new_shape[1] = 6
        new_shape = [1, 6], new_delta = [1, 1], new_norm = sqrt(2) ≈ 1.4142
        drop_pct = (3.1623 - 1.4142) / 3.1623 ≈ 55.28%
    """
    from hypertopos.engine.counterfactual import simulate_joint_edge_removal
    relations = [{"line_id": "L", "direction": "out"}]
    edges = [
        {"edge_id": "E1", "partner_key": "X", "direction": "out", "line_id": "L"},
        {"edge_id": "E2", "partner_key": "Y", "direction": "out", "line_id": "L"},
        {"edge_id": "E3", "partner_key": "Z", "direction": "out", "line_id": "L"},
    ]
    result = simulate_joint_edge_removal(
        shape=np.array([3.0, 6.0], dtype=np.float32),
        mu=np.array([0.0, 5.0], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0], dtype=np.float32),
        delta_norm=math.sqrt(10.0),
        edges_to_remove=[edges[0], edges[2]],
        all_edges=edges,
        relations=relations,
        edge_agg_dim_offset=1,
        edge_agg_specs=[("pair_count", "mean")],
        event_source_values={
            "E1": {"pair_count": 10.0},
            "E2": {"pair_count": 6.0},
            "E3": {"pair_count": 2.0},
        },
        thresholds=None,
    )
    expected_norm = math.sqrt(2.0)
    expected_drop = (math.sqrt(10.0) - expected_norm) / math.sqrt(10.0) * 100
    assert result["delta_norm_after"] == pytest.approx(expected_norm, abs=1e-3)
    assert result["joint_drop_pct"] == pytest.approx(expected_drop, abs=0.1)


def test_select_minimal_joint_removal_greedy_terminates_at_target():
    """Greedy selection: add edges one at a time until joint drop ≥ target.

    3 relations dims, all "out" direction with mu=0 sigma=1.
    Entity has 6 out edges in 3 different lines (2 each).
    shape = [4, 4, 4], delta = shape, delta_norm = sqrt(48) ≈ 6.928.

    Removing 1 edge from any line: shape[i] -=1 → norm changes by some amount.
    Target_drop_pct = 20.0%, k_max = 6.

    Greedy picks the most-effective edge each step. Expected: picks one
    from each line family before doubling up. Achieves target around k=2-3.
    Verify: termination + selected sequence is valid + achieved >= target.
    """
    from hypertopos.engine.counterfactual import select_minimal_joint_removal
    relations = [
        {"line_id": "A", "direction": "out"},
        {"line_id": "B", "direction": "out"},
        {"line_id": "C", "direction": "out"},
    ]
    edges = []
    for line in ("A", "B", "C"):
        for i in range(2):
            edges.append({
                "edge_id": f"{line}{i}", "partner_key": f"P_{line}{i}",
                "direction": "out", "line_id": line,
            })
    result = select_minimal_joint_removal(
        shape=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        mu=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        delta_norm=math.sqrt(48.0),
        candidate_edges=edges,
        relations=relations,
        edge_agg_dim_offset=3,
        edge_agg_specs=[],
        event_source_values={},
        target_drop_pct=20.0,
        k_max=6,
        thresholds=None,
    )
    assert result["target_reached"] is True
    assert result["achieved_abs_drop_pct"] >= 20.0
    assert 1 <= len(result["selected_edge_ids"]) <= 6
    # Selected sequence is monotone in |joint_drop_pct|: each step adds
    # absolute magnitude (direction-agnostic selection).
    seq_abs_drops = [s["abs_joint_drop_pct"] for s in result["selection_sequence"]]
    assert all(
        seq_abs_drops[i] <= seq_abs_drops[i + 1]
        for i in range(len(seq_abs_drops) - 1)
    )
    # All selected edges are distinct.
    assert len(set(result["selected_edge_ids"])) == len(result["selected_edge_ids"])


def test_select_minimal_joint_removal_target_unreachable_caps_at_k_max():
    """When target_drop_pct cannot be reached within k_max, return what
    we have with target_reached=False."""
    from hypertopos.engine.counterfactual import select_minimal_joint_removal
    # Single edge that can contribute at most ~12% — target 50% unreachable.
    relations = [{"line_id": "A", "direction": "out"}]
    edges = [
        {"edge_id": "E1", "partner_key": "X", "direction": "out", "line_id": "A"},
    ]
    result = select_minimal_joint_removal(
        shape=np.array([4.0], dtype=np.float32),
        mu=np.array([2.0], dtype=np.float32),
        sigma_diag=np.array([1.0], dtype=np.float32),
        delta_norm=2.0,
        candidate_edges=edges,
        relations=relations,
        edge_agg_dim_offset=1,
        edge_agg_specs=[],
        event_source_values={},
        target_drop_pct=99.0,
        k_max=5,
        thresholds=None,
    )
    assert result["target_reached"] is False
    # k_max=5 but only 1 candidate edge available — neither cap fires:
    # selection exits via plateau (no improving candidate left after the
    # single edge is taken). target_reached AND k_max_reached both False
    # means "candidates exhausted before either cap fired".
    assert result["k_max_reached"] is False
    assert len(result["selected_edge_ids"]) == 1


def test_compute_per_edge_pvalues_breaks_uniform_drop_pct_ties():
    """Discrimination check: when raw drop_pct is uniform across edges
    (e.g. p95 robust-tail entity), per-edge p-values STILL discriminate
    because source values differ.

    Pop = [0..99]. Three edges with same uniform drop_pct=-0.3 but different
    pair_count values: e_high=95, e_mid=50, e_low=5.
    Expected min_pvalue ordering: e_high (0.05) < e_mid (0.50) < e_low (0.95).
    This is THE failure mode Phase 2 exists to fix.
    """
    from hypertopos.engine.counterfactual import (
        compute_per_edge_source_value_pvalues,
    )
    pop_ecdfs = {"pair_count": np.arange(100, dtype=np.float64)}
    edges = [
        {"edge_id": f"e_{label}", "partner_key": label,
         "direction": "out", "line_id": "L"}
        for label in ("high", "mid", "low")
    ]
    event_source_values = {
        "e_high": {"pair_count": 95.0},
        "e_mid":  {"pair_count": 50.0},
        "e_low":  {"pair_count":  5.0},
    }
    result = compute_per_edge_source_value_pvalues(
        edges=edges,
        event_source_values=event_source_values,
        population_ecdfs=pop_ecdfs,
        source_dims=["pair_count"],
    )
    assert result["e_high"]["min_pvalue"] < result["e_mid"]["min_pvalue"]
    assert result["e_mid"]["min_pvalue"] < result["e_low"]["min_pvalue"]
    assert result["e_high"]["min_pvalue"] == pytest.approx(0.05)
    assert result["e_low"]["min_pvalue"] == pytest.approx(0.95)
