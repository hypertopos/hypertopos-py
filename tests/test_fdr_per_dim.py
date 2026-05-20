# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Per-dimension FDR — hand-verified math.

Per-entity-per-dim p-value under univariate chi²(1) survival on |delta_i|.
BH/Storey applied PER DIM (each dim gets its own correction; no global pool).

Replaces the entity-level single-FDR with per-dim independence: a dim
driving many anomalies cannot inflate the global FDR for other dims.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypertopos.engine.fdr import (
    fdr_per_dimension,
    per_dim_p_values_chi2_univariate,
)


def test_per_dim_pvalues_chi2_under_null():
    """Under N(0,1) null, |delta| ~ Folded-Normal. Chi²(1) survival on
    delta² gives the two-sided p-value = 2*(1 - Φ(|delta|)).

    For delta = 0 → p = 1.0 exactly.
    For delta = 1.96 → p ≈ 0.05 (the two-sided critical value).
    For delta = 2.58 → p ≈ 0.01.
    """
    deltas = np.array([
        [0.0,  1.96, 2.58],
        [1.96, 2.58, 0.0],
    ])
    p = per_dim_p_values_chi2_univariate(deltas)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(p[0, 1], 0.05, atol=1e-2)
    np.testing.assert_allclose(p[0, 2], 0.01, atol=1e-2)
    np.testing.assert_allclose(p[1, 0], 0.05, atol=1e-2)
    # Same delta values should produce same p-values regardless of position
    np.testing.assert_allclose(p[0, 1], p[1, 0], atol=1e-10)


def test_per_dim_pvalues_clip_floor():
    """Extreme delta should not produce p == 0 (would crash log-based combiners
    downstream). Clip floor 1e-10."""
    deltas = np.array([[100.0]])
    p = per_dim_p_values_chi2_univariate(deltas)
    assert p[0, 0] >= 1e-10
    assert p[0, 0] < 1e-6  # very small but not zero


def test_fdr_per_dim_each_dim_corrected_independently():
    """Hand-built: 3 dims × 5 entities. Dim 0 has perfect signal (1 strong
    positive, 4 nulls). Dim 1 all nulls. Dim 2 has 3 mid signals.

    Under BH per-dim with alpha=0.10:
        Dim 0 (m=5, p=[0.01, 0.5, 0.5, 0.5, 0.5]):
            BH-sorted: [0.01, 0.5, 0.5, 0.5, 0.5]
            q-values: 0.01*5/1=0.05, 0.5*5/2=1.25→clipped to 1.0, ...
            Rejected at α=0.10: just entity 0.
        Dim 1 all 0.5: no rejection.
        Dim 2 (m=5, p=[0.04, 0.03, 0.02, 0.5, 0.5]):
            BH-sorted ascending: 0.02, 0.03, 0.04, 0.5, 0.5
            q-values: 0.02*5/1=0.10, 0.03*5/2=0.075, 0.04*5/3=0.067,
                       0.5*5/4=0.625, 0.5*5/5=0.5
            After right-to-left min: 0.067, 0.067, 0.067, 0.5, 0.5
            Rejected at α=0.10: entities 0, 1, 2 (the three small p-values).
    """
    p_matrix = np.array([
        # entity 0 has dim 0 strong AND dim 2 mid
        [0.01, 0.5, 0.04],
        # entity 1 has dim 2 mid
        [0.5,  0.5, 0.03],
        # entity 2 has dim 2 mid
        [0.5,  0.5, 0.02],
        [0.5,  0.5, 0.5],
        [0.5,  0.5, 0.5],
    ])
    rejected, q_values = fdr_per_dimension(p_matrix, alpha=0.10, method="bh")
    # Shape preserved
    assert rejected.shape == (5, 3)
    assert q_values.shape == (5, 3)
    # Dim 0: only entity 0 rejected
    assert rejected[0, 0]
    assert not rejected[1, 0]
    # Dim 1: nothing rejected
    assert not rejected[:, 1].any()
    # Dim 2: entities 0, 1, 2 rejected
    assert rejected[0, 2] and rejected[1, 2] and rejected[2, 2]
    assert not rejected[3, 2] and not rejected[4, 2]


def test_fdr_per_dim_does_not_pool_across_dims():
    """Per-dim FDR must not pool. If dim 0 has 1 anomaly out of N and dim 1
    has N/2 anomalies out of N, dim 0's threshold should NOT be inflated by
    dim 1's many discoveries.

    Setup:
        Dim 0: [0.001, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            → BH q[0] = 0.001*10/1 = 0.01 → rejected at α=0.05
        Dim 1: [0.001, 0.002, 0.003, 0.004, 0.005, 0.5, 0.5, 0.5, 0.5, 0.5]
            → all 5 small p's get rejected at α=0.05

    Global pool (20 tests): same entity 0 dim 0 would get q = 0.001*20/1 = 0.02
        — still rejected at α=0.05, but per-dim gives 0.01 (lower q, more power).
    """
    n = 10
    p_matrix = np.empty((n, 2))
    p_matrix[:, 0] = [0.001] + [0.5] * (n - 1)
    p_matrix[:, 1] = [0.001, 0.002, 0.003, 0.004, 0.005] + [0.5] * (n - 5)
    rejected, q = fdr_per_dimension(p_matrix, alpha=0.05, method="bh")
    # Dim 0 entity 0 should have q ≈ 0.01 (per-dim n=10), not 0.02 (pooled n=20)
    assert q[0, 0] == pytest.approx(0.01, abs=1e-6)
    assert rejected[0, 0]
    # Dim 1 should have 5 rejections
    assert rejected[:5, 1].all()
    assert not rejected[5:, 1].any()


def test_fdr_per_dim_storey_method():
    """Storey method scales BH q-values by pi_0_hat per dim.

    With many large p-values (>0.5), pi_0 < 1 and Storey shrinks q-values
    relative to BH — recovers power.
    """
    rng = np.random.RandomState(0)
    # Build a single dim with 90% true nulls (uniform p) + 10% small p
    p_dim = np.concatenate([
        rng.uniform(0, 1, 90),    # nulls
        rng.uniform(0, 0.01, 10),  # signals
    ])
    p_matrix = p_dim.reshape(-1, 1)
    _, q_bh = fdr_per_dimension(p_matrix, alpha=0.05, method="bh")
    _, q_storey = fdr_per_dimension(p_matrix, alpha=0.05, method="storey")
    # Storey q-values <= BH q-values (more power)
    assert (q_storey <= q_bh).all()


def test_fdr_per_dim_empty_input():
    """Empty p-value matrix returns empty result without crash."""
    p_matrix = np.empty((0, 3))
    rejected, q = fdr_per_dimension(p_matrix, alpha=0.05, method="bh")
    assert rejected.shape == (0, 3)
    assert q.shape == (0, 3)


def test_fdr_per_dim_validates_inputs():
    """Bad alpha, bad method raise ValueError."""
    p = np.array([[0.05]])
    with pytest.raises(ValueError, match="alpha"):
        fdr_per_dimension(p, alpha=0.0, method="bh")
    with pytest.raises(ValueError, match="alpha"):
        fdr_per_dimension(p, alpha=1.5, method="bh")
    with pytest.raises(ValueError, match="method"):
        fdr_per_dimension(p, alpha=0.05, method="unknown")


def test_per_dim_pvalues_pipeline_with_fdr_per_dim():
    """End-to-end: deltas → chi² p-values → per-dim BH FDR.

    Construct 5 entities × 2 dims where:
        Entity 0 has |delta_0| = 3.0 (very anomalous on dim 0)
        Entity 1 has |delta_1| = 3.0 (anomalous on dim 1)
        Entities 2-4 have |delta_*| = 0.1 (nulls)

    Chi²(1) p-value at |delta|=3.0 is ≈ 0.0027.
    BH at α=0.10 per dim: q-value for the single small p-value =
        0.0027 * 5/1 = 0.0135 < 0.10 → rejected.

    Expected: entity 0 rejected on dim 0, entity 1 rejected on dim 1.
    """
    deltas = np.array([
        [3.0, 0.1],
        [0.1, 3.0],
        [0.1, 0.1],
        [0.1, 0.1],
        [0.1, 0.1],
    ])
    p = per_dim_p_values_chi2_univariate(deltas)
    rejected, q = fdr_per_dimension(p, alpha=0.10, method="bh")
    assert rejected[0, 0]
    assert rejected[1, 1]
    assert not rejected[0, 1]
    assert not rejected[1, 0]
    assert not rejected[2:].any()


# -----------------------------------------------------------------------------
# Navigator-side polygon mutation contract
# -----------------------------------------------------------------------------


def _make_polygon(primary_key: str, delta: list[float]):
    from datetime import datetime

    from hypertopos.model.objects import Polygon

    arr = np.array(delta, dtype=np.float32)
    return Polygon(
        primary_key=primary_key,
        pattern_id="p_test",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=arr,
        delta_norm=float(np.linalg.norm(arr)),
        is_anomaly=True,
        edges=[],
        last_refresh_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        delta_rank_pct=99.0,
    )


def test_apply_fdr_attaches_per_dim_attrs_with_engineered_distinct_dims():
    """π5 invokes _apply_fdr_select_polygons; per_dim mode must attach
    q_values_per_dim, min_q_per_dim, dominant_q_dim_idx to each polygon
    AND those attributes must DISCRIMINATE per entity when inputs are
    engineered to have different dominant dimensions.

    Guards against uniform_smoke_needs_discriminator failure mode: if the
    mutation logic accidentally writes the same per-dim vector to every
    polygon, top-K would look correct but all dominant_q_dim_idx values
    would collapse to one index.

    Engineering:
        entity_0: delta = [3.0, 0.1, 0.1]  → dominant dim 0
        entity_1: delta = [0.1, 3.0, 0.1]  → dominant dim 1
        entity_2: delta = [0.1, 0.1, 3.0]  → dominant dim 2
    """
    from hypertopos.navigation.navigator import GDSNavigator

    polys = [
        _make_polygon("e0", [3.0, 0.1, 0.1]),
        _make_polygon("e1", [0.1, 3.0, 0.1]),
        _make_polygon("e2", [0.1, 0.1, 3.0]),
    ]
    out = GDSNavigator._apply_fdr_select_polygons(
        polys,
        fdr_alpha=0.10,
        select="top_norm",
        top_n=3,
        fdr_method="bh",
        p_value_method="chi2",
        pattern_df=3,
        fdr_axis="per_dim",
    )
    assert len(out) == 3, "all three engineered anomalies must survive"
    for poly in out:
        assert hasattr(poly, "q_values_per_dim")
        assert hasattr(poly, "min_q_per_dim")
        assert hasattr(poly, "dominant_q_dim_idx")
        assert len(poly.q_values_per_dim) == 3
        assert 0.0 <= poly.min_q_per_dim <= 1.0
    by_key = {p.primary_key: p for p in out}
    # Discriminator: dominant dim differs per engineered entity
    assert by_key["e0"].dominant_q_dim_idx == 0
    assert by_key["e1"].dominant_q_dim_idx == 1
    assert by_key["e2"].dominant_q_dim_idx == 2
    # Discriminator: q-value vectors are NOT identical across entities
    q_vectors = [tuple(p.q_values_per_dim) for p in out]
    assert len(set(q_vectors)) == 3, (
        f"per-dim q-values collapsed across entities: {q_vectors}"
    )


def test_apply_fdr_per_dim_min_q_supports_rank_by_min_q_sort():
    """The rank_by='min_q_per_dim' sort step in π5_attract_anomaly is a
    plain ascending sort by min_q_per_dim. This test verifies that the
    sort produces a different order than delta_norm desc when the
    engineered inputs make those two orderings disagree.

    Engineering:
        strong_norm_weak_signal: delta = [2.0, 2.0, 2.0]
            → ||delta|| = sqrt(12) ≈ 3.46 (high) but every dim has identical
              mid-strength p (chi²(1) at |delta|=2 → p ≈ 0.045),
              so min_q_per_dim is mid.
        weak_norm_strong_signal: delta = [3.0, 0.0, 0.0]
            → ||delta|| = 3.0 (lower) but dim 0 has tiny chi²(1) p-value
              (chi²(1) at |delta|=3 → p ≈ 0.0027), so min_q_per_dim is tiny.

    Expected: delta_norm desc ranks "strong_norm_weak_signal" first.
              min_q_per_dim asc ranks "weak_norm_strong_signal" first.
    """
    from hypertopos.navigation.navigator import GDSNavigator

    polys = [
        _make_polygon("strong_norm_weak_signal", [2.0, 2.0, 2.0]),
        _make_polygon("weak_norm_strong_signal", [3.0, 0.0, 0.0]),
    ]
    out = GDSNavigator._apply_fdr_select_polygons(
        list(polys),
        fdr_alpha=0.20,
        select="top_norm",
        top_n=2,
        fdr_method="bh",
        p_value_method="chi2",
        pattern_df=3,
        fdr_axis="per_dim",
    )
    # Default ranking after FDR is by delta_norm desc (caller sorts; here we
    # just verify the asymmetry between the two ranking keys).
    by_key = {p.primary_key: p for p in out}
    assert by_key["strong_norm_weak_signal"].delta_norm > by_key[
        "weak_norm_strong_signal"
    ].delta_norm
    assert by_key["weak_norm_strong_signal"].min_q_per_dim < by_key[
        "strong_norm_weak_signal"
    ].min_q_per_dim
    # Simulate the rank_by='min_q_per_dim' sort step
    sorted_by_min_q = sorted(out, key=lambda p: p.min_q_per_dim)
    assert sorted_by_min_q[0].primary_key == "weak_norm_strong_signal", (
        "rank_by='min_q_per_dim' must surface the entity with the strongest "
        "single-dim signal, not the entity with the largest joint norm"
    )
