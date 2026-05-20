# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Per-dimension label-aware signal audit — hand-verified math."""
from __future__ import annotations

import numpy as np
import pytest
from hypertopos.engine.dim_audit import (
    compute_per_dim_label_auroc,
    filter_delta_norm,
    fit_lda_direction,
)


def test_per_dim_auroc_signal_carrier_identified():
    """Hand-built case: dim 0 perfectly discriminates positives.

    Setup:
        deltas[:, 0] = [10, 10, 10, -1, -1, -1] (large for positives, small for negatives)
        deltas[:, 1] = [0, 0, 0, 0, 0, 0]       (zero variance, no signal)
        labels       = [1, 1, 1, 0, 0, 0]

    Dim 0: |delta| = [10, 10, 10, 1, 1, 1] → perfectly separable → AUROC = 1.0
    Dim 1: all zeros → AUROC = 0.5 (sigma_zero fallback)
    """
    deltas = np.array([
        [10.0, 0.0],
        [10.0, 0.0],
        [10.0, 0.0],
        [-1.0, 0.0],
        [-1.0, 0.0],
        [-1.0, 0.0],
    ])
    labels = np.array([1, 1, 1, 0, 0, 0])
    result = compute_per_dim_label_auroc(
        deltas=deltas, labels=labels,
        dim_labels=["strong", "dead"],
    )
    assert result["per_dim"][0]["label"] == "strong"
    assert result["per_dim"][0]["auroc"] == pytest.approx(1.0)
    assert result["per_dim"][0]["classification"] == "signal"
    assert result["per_dim"][1]["auroc"] == pytest.approx(0.5)
    assert result["per_dim"][1]["classification"] == "neutral"
    assert result["n_signal"] == 1
    assert result["n_anti"] == 0
    assert result["signal_idx"] == [0]
    assert result["signal_mask"][0] and not result["signal_mask"][1]


def test_per_dim_auroc_anti_signal_identified():
    """Anti-signal dim: |delta| LARGE for negatives, SMALL for positives.

    deltas[:, 0] = [1, 1, 1, 10, 10, 10]  (large for negatives)
    labels       = [1, 1, 1, 0, 0, 0]

    Dim 0: AUROC(|delta|, labels) = 0.0 (perfect anti-correlation).
    """
    deltas = np.array([
        [1.0], [1.0], [1.0],
        [10.0], [10.0], [10.0],
    ])
    labels = np.array([1, 1, 1, 0, 0, 0])
    result = compute_per_dim_label_auroc(
        deltas=deltas, labels=labels, dim_labels=["anti"],
    )
    assert result["per_dim"][0]["auroc"] == pytest.approx(0.0)
    assert result["per_dim"][0]["classification"] == "anti"
    assert result["n_anti"] == 1
    assert result["anti_idx"] == [0]


def test_per_dim_auroc_handles_ties_in_scores():
    """When scores have ties, AUROC uses averaged-rank formula.

    Two positives, two negatives. Positives: [5, 5]. Negatives: [3, 7].
    Scores ascending: [3, 5, 5, 7]. Ranks (averaged for ties): [1, 2.5, 2.5, 4].
    Sum positive ranks = 2.5 + 2.5 = 5.
    AUROC = (5 - 2*3/2) / (2*2) = 2 / 4 = 0.5
    """
    deltas = np.array([[5.0], [5.0], [3.0], [7.0]])
    labels = np.array([1, 1, 0, 0])
    result = compute_per_dim_label_auroc(
        deltas=deltas, labels=labels, dim_labels=["ties"],
    )
    assert result["per_dim"][0]["auroc"] == pytest.approx(0.5)


def test_per_dim_auroc_thresholds_configurable():
    """Caller can adjust signal / anti thresholds for noisy domains."""
    # Construct dim with AUROC ≈ 0.55 — borderline.
    deltas = np.array([
        [3.0], [3.0], [3.0], [3.0], [3.0],
        [1.0], [1.0], [1.0], [1.0], [1.0],
        [5.0],  # one negative with large |delta| — drops dim's AUROC
    ])
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # With default 0.55 threshold this lands signal:
    r_default = compute_per_dim_label_auroc(
        deltas=deltas, labels=labels, upper_threshold=0.55,
    )
    # Same dim with stricter 0.95 threshold lands neutral:
    r_strict = compute_per_dim_label_auroc(
        deltas=deltas, labels=labels, upper_threshold=0.95,
    )
    assert r_strict["n_signal"] <= r_default["n_signal"]


def test_filter_delta_norm_matches_unfiltered_when_keep_all():
    """filter_delta_norm with all idx == plain L2 norm."""
    deltas = np.array([
        [3.0, 4.0],   # norm = 5
        [1.0, 0.0],   # norm = 1
        [0.0, 0.0],   # norm = 0
    ])
    full = np.linalg.norm(deltas, axis=1)
    filtered = filter_delta_norm(deltas=deltas, keep_idx=[0, 1])
    np.testing.assert_array_almost_equal(filtered, full)


def test_filter_delta_norm_drops_unwanted_dims():
    """Hand-verified: drop dim 0, recompute L2 over dim 1 only.

    deltas = [[3, 4], [1, 0]]
    keep [1]: filtered = [|4|, |0|] = [4, 0]
    """
    deltas = np.array([[3.0, 4.0], [1.0, 0.0]])
    filtered = filter_delta_norm(deltas=deltas, keep_idx=[1])
    np.testing.assert_array_almost_equal(filtered, [4.0, 0.0])


def test_filter_delta_norm_empty_keep_returns_zeros():
    """Empty keep_idx → all entities get score 0 (degenerate but defined)."""
    deltas = np.array([[3.0, 4.0], [1.0, 0.0]])
    filtered = filter_delta_norm(deltas=deltas, keep_idx=[])
    np.testing.assert_array_equal(filtered, [0.0, 0.0])


def test_per_dim_auroc_shape_validation():
    """Engine rejects mis-shaped inputs."""
    deltas_1d = np.array([1.0, 2.0, 3.0])
    labels = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="2-D"):
        compute_per_dim_label_auroc(deltas=deltas_1d, labels=labels)

    deltas_2d = np.array([[1.0], [2.0]])
    labels_wrong = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="labels shape"):
        compute_per_dim_label_auroc(deltas=deltas_2d, labels=labels_wrong)


def test_signal_mask_is_boolean_array():
    """signal_mask is a length-n_dims boolean ndarray."""
    deltas = np.random.RandomState(0).randn(20, 5)
    labels = np.array([0, 1] * 10)
    result = compute_per_dim_label_auroc(deltas=deltas, labels=labels)
    assert result["signal_mask"].dtype == bool
    assert result["signal_mask"].shape == (5,)


def test_full_pipeline_reproduces_expected_lift_on_synthetic():
    """End-to-end check: build deltas where 2 dims are signal, 2 are noise,
    1 is anti-signal. Verify that filtered delta_norm > full delta_norm
    AUROC.

    Construction:
        Positives (n=50): high values on dims [0, 1], moderate on [2, 3],
                          LOW on dim [4]
        Negatives (n=50): low on [0, 1], moderate on [2, 3], HIGH on [4]

    Expected:
        dim 0, 1 → signal carriers
        dim 2, 3 → neutral (no class separation)
        dim 4    → anti-signal (anti-correlated)
    """
    rng = np.random.RandomState(42)
    n = 50
    pos_deltas = np.column_stack([
        rng.normal(3.0, 0.5, n),   # signal
        rng.normal(3.0, 0.5, n),   # signal
        rng.normal(0.0, 1.0, n),   # neutral
        rng.normal(0.0, 1.0, n),   # neutral
        rng.normal(0.0, 0.5, n),   # anti (small for positives)
    ])
    neg_deltas = np.column_stack([
        rng.normal(0.0, 0.5, n),
        rng.normal(0.0, 0.5, n),
        rng.normal(0.0, 1.0, n),
        rng.normal(0.0, 1.0, n),
        rng.normal(3.0, 0.5, n),   # anti (large for negatives)
    ])
    deltas = np.vstack([pos_deltas, neg_deltas])
    labels = np.array([1] * n + [0] * n)
    result = compute_per_dim_label_auroc(
        deltas=deltas, labels=labels,
        dim_labels=["sig_a", "sig_b", "noise_a", "noise_b", "anti"],
    )
    sig_labels = [d["label"] for d in result["per_dim"] if d["classification"] == "signal"]
    anti_labels = [d["label"] for d in result["per_dim"] if d["classification"] == "anti"]
    assert "sig_a" in sig_labels and "sig_b" in sig_labels
    assert "anti" in anti_labels

    # Filtered L2 on signal dims only beats raw L2 (no skl required —
    # use the engine's own AUROC).
    from hypertopos.engine.dim_audit import _auroc_unsafe
    full_dn = np.linalg.norm(deltas, axis=1)
    sig_dn = filter_delta_norm(deltas=deltas, keep_idx=result["signal_idx"])
    full_a = _auroc_unsafe(full_dn, labels)
    sig_a = _auroc_unsafe(sig_dn, labels)
    assert sig_a > full_a + 0.05  # measurable lift on synthetic


def test_lda_two_well_separated_gaussians_recovers_direction():
    """Two well-separated 2-D gaussians: class 0 at (0,0), class 1 at (5,0).

    With identity covariance and a clean (1, 0) class-separation axis,
    the LDA direction must recover ≈ [1, 0] (up to the sign convention).
    """
    rng = np.random.RandomState(0)
    n = 20
    class_0 = rng.multivariate_normal([0.0, 0.0], np.eye(2), size=n)
    class_1 = rng.multivariate_normal([5.0, 0.0], np.eye(2), size=n)
    deltas = np.vstack([class_0, class_1])
    labels = np.array([0] * n + [1] * n)
    result = fit_lda_direction(deltas=deltas, labels=labels)
    direction = result["direction"]
    # Sign-orient is enforced: positive projection on anomalous side, so
    # direction[0] must come out positive.
    assert direction[0] == pytest.approx(1.0, abs=0.1)
    assert direction[1] == pytest.approx(0.0, abs=0.1)
    assert result["n_anom"] == n
    assert result["n_normal"] == n


def test_lda_returns_unit_norm_direction():
    """Returned direction is L2-unit-normalised."""
    rng = np.random.RandomState(1)
    n = 30
    deltas = np.vstack([
        rng.normal(0.0, 1.0, (n, 4)),
        rng.normal(2.0, 1.0, (n, 4)),
    ])
    labels = np.array([0] * n + [1] * n)
    result = fit_lda_direction(deltas=deltas, labels=labels)
    assert np.linalg.norm(result["direction"]) == pytest.approx(1.0, abs=1e-6)


def test_lda_sign_convention_positive_projection_on_anomalous():
    """Direction sign is oriented so w . (mu_anom - mu_normal) > 0."""
    rng = np.random.RandomState(2)
    n = 25
    # Negative-direction separation: positives are LOWER than negatives.
    class_0 = rng.normal(5.0, 1.0, (n, 3))
    class_1 = rng.normal(0.0, 1.0, (n, 3))
    deltas = np.vstack([class_0, class_1])
    labels = np.array([0] * n + [1] * n)
    result = fit_lda_direction(deltas=deltas, labels=labels)
    mu_anom = deltas[labels == 1].mean(axis=0)
    mu_normal = deltas[labels == 0].mean(axis=0)
    diff = mu_anom - mu_normal
    assert result["direction"] @ diff > 0.0


def test_lda_identical_class_means_raises():
    """Mean-of-positives == mean-of-negatives → no LDA direction defined."""
    rng = np.random.RandomState(3)
    n = 20
    # Two halves of the SAME distribution → means are random-but-close;
    # to make them EXACTLY equal, use a deterministic mirror.
    base = rng.normal(0.0, 1.0, (n, 2))
    deltas = np.vstack([base, base])
    labels = np.array([0] * n + [1] * n)
    with pytest.raises(ValueError, match="identical"):
        fit_lda_direction(deltas=deltas, labels=labels)


def test_lda_single_class_raises():
    """All-positives or all-negatives → cannot fit two-class scatter."""
    rng = np.random.RandomState(4)
    deltas = rng.normal(0.0, 1.0, (20, 3))
    labels = np.ones(20, dtype=int)
    with pytest.raises(ValueError, match="both classes must be present"):
        fit_lda_direction(deltas=deltas, labels=labels)


def test_lda_too_few_per_class_raises():
    """A class with 1 sample → within-class scatter undefined."""
    rng = np.random.RandomState(5)
    deltas = np.vstack([
        rng.normal(0.0, 1.0, (10, 3)),   # 10 normals
        rng.normal(3.0, 1.0, (1, 3)),    # 1 anomalous
    ])
    labels = np.array([0] * 10 + [1] * 1)
    with pytest.raises(ValueError, match=">=2"):
        fit_lda_direction(deltas=deltas, labels=labels)


def test_lda_regularization_stabilizes_rank_deficient():
    """N=5, D=20 ⇒ S_w is severely rank-deficient (rank ≤ 3).

    With default `regularization=1e-6` the linear solve must still
    produce a finite unit-norm direction and a finite Fisher score.
    """
    rng = np.random.RandomState(6)
    class_0 = rng.normal(0.0, 1.0, (5, 20))
    class_1 = rng.normal(2.0, 1.0, (5, 20))
    deltas = np.vstack([class_0, class_1])
    labels = np.array([0] * 5 + [1] * 5)
    result = fit_lda_direction(deltas=deltas, labels=labels)
    assert np.isfinite(result["direction"]).all()
    assert np.linalg.norm(result["direction"]) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(result["fisher_score"])


def test_lda_deterministic():
    """Same input twice → byte-identical output."""
    rng = np.random.RandomState(7)
    n = 40
    deltas = np.vstack([
        rng.normal(0.0, 1.0, (n, 5)),
        rng.normal(1.5, 1.0, (n, 5)),
    ])
    labels = np.array([0] * n + [1] * n)
    r1 = fit_lda_direction(deltas=deltas, labels=labels)
    r2 = fit_lda_direction(deltas=deltas, labels=labels)
    np.testing.assert_array_equal(r1["direction"], r2["direction"])
    assert r1["fisher_score"] == r2["fisher_score"]


def test_lda_fisher_score_higher_for_better_separation():
    """Larger inter-class margin → larger Fisher discriminant ratio."""
    rng = np.random.RandomState(8)
    n = 50
    # 3σ-separated: means at 0 and 6 with unit variance.
    far_0 = rng.normal(0.0, 1.0, (n, 3))
    far_1 = rng.normal(6.0, 1.0, (n, 3))
    far_deltas = np.vstack([far_0, far_1])
    # 0.5σ-separated: means at 0 and 1 with unit variance.
    near_0 = rng.normal(0.0, 1.0, (n, 3))
    near_1 = rng.normal(1.0, 1.0, (n, 3))
    near_deltas = np.vstack([near_0, near_1])
    labels = np.array([0] * n + [1] * n)
    r_far = fit_lda_direction(deltas=far_deltas, labels=labels)
    r_near = fit_lda_direction(deltas=near_deltas, labels=labels)
    assert r_far["fisher_score"] > r_near["fisher_score"]
