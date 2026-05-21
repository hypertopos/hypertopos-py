# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Label-aware per-dim calibration — hand-verified math + acceptance test."""
from __future__ import annotations

import numpy as np
import pytest
from hypertopos.engine.calibration_label_aware import (
    CalibrationResult,
    DimCalibration,
    calibrate_label_aware,
)


def _synthetic_two_class(
    *,
    sep_mean: float = 3.0,
    n_per_class: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2-D 2-class case: dim 0 separates, dim 1 is noise."""
    rng = np.random.RandomState(seed)
    pos = np.column_stack([
        rng.normal(sep_mean, 1.0, n_per_class),  # separating dim
        rng.normal(0.0, 1.0, n_per_class),       # noise dim
    ])
    neg = np.column_stack([
        rng.normal(0.0, 1.0, n_per_class),
        rng.normal(0.0, 1.0, n_per_class),
    ])
    deltas = np.vstack([pos, neg]).astype(np.float32)
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    return deltas, labels


# ── Acceptance test (plan §M1.1) ──────────────────────────────────────


def test_synthetic_two_class_separating_dim_aligned_noise_dim_suppressed():
    """Plan acceptance test.

    Synthetic 2-class data with one separating dim and one noise dim:
    - The Fisher LDA direction must align with the class-mean diff on
      the separating dim (large positive ``direction``).
    - The noise dim's direction component must satisfy
      ``|direction| < 0.05`` per the plan threshold.
    """
    # Large n + strong separation drives finite-sample drift on the
    # noise dim well under the plan's 0.05 acceptance threshold. With
    # n=2000 per class, the noise dim's class-mean difference is
    # O(1/sqrt(n)) ≈ 0.022, and the unit-normalised LDA direction
    # component on the noise dim is well below 0.05.
    deltas, labels = _synthetic_two_class(
        sep_mean=5.0, n_per_class=2000, seed=42,
    )
    result = calibrate_label_aware(
        deltas=deltas, labels=labels,
        dim_labels=["separating", "noise"],
    )

    # The signed direction on the separating dim must be aligned with
    # the class-mean diff (positive after sign-orientation).
    sep_cal = result.per_dim["separating"]
    assert sep_cal.mu_pos > sep_cal.mu_neg
    assert sep_cal.direction > 0.9, (
        f"separating dim should carry near-all of the unit LDA axis, "
        f"got direction={sep_cal.direction:.4f}"
    )

    # Plan-mandated threshold: noise dim's direction component <0.05.
    noise_cal = result.per_dim["noise"]
    assert abs(noise_cal.direction) < 0.05, (
        f"noise dim direction must be < 0.05, got "
        f"{noise_cal.direction:.4f}"
    )

    # The full unit-norm vector is consistent with both per-dim entries.
    assert np.linalg.norm(result.signed_direction_vector) == pytest.approx(
        1.0, abs=1e-6,
    )
    assert result.signed_direction_vector[0] == pytest.approx(
        sep_cal.direction, abs=1e-6,
    )
    assert result.signed_direction_vector[1] == pytest.approx(
        noise_cal.direction, abs=1e-6,
    )


# ── Per-dim moments correctness ───────────────────────────────────────


def test_per_dim_moments_match_numpy_groupby_means_and_stds():
    """Per-dim ``mu_pos`` / ``sigma_pos`` / ``mu_neg`` / ``sigma_neg``
    must equal numpy-computed means/stds on the labelled subsets.
    """
    deltas, labels = _synthetic_two_class(seed=7)
    result = calibrate_label_aware(deltas=deltas, labels=labels)

    pos = deltas[labels == 1]
    neg = deltas[labels == 0]
    expected_mu_pos = pos.mean(axis=0)
    expected_sigma_pos = pos.std(axis=0, ddof=0)
    expected_mu_neg = neg.mean(axis=0)
    expected_sigma_neg = neg.std(axis=0, ddof=0)

    for i, name in enumerate(["dim_0", "dim_1"]):
        cal = result.per_dim[name]
        assert cal.mu_pos == pytest.approx(float(expected_mu_pos[i]), abs=1e-5)
        assert cal.sigma_pos == pytest.approx(
            float(expected_sigma_pos[i]), abs=1e-5,
        )
        assert cal.mu_neg == pytest.approx(float(expected_mu_neg[i]), abs=1e-5)
        assert cal.sigma_neg == pytest.approx(
            float(expected_sigma_neg[i]), abs=1e-5,
        )


def test_class_counts_match_labels():
    """``n_pos`` / ``n_neg`` echo the label vector counts."""
    deltas, labels = _synthetic_two_class(n_per_class=30, seed=11)
    result = calibrate_label_aware(deltas=deltas, labels=labels)
    assert result.n_pos == 30
    assert result.n_neg == 30


def test_dim_labels_default_when_none_supplied():
    """``dim_labels=None`` defaults to ``dim_0``, ``dim_1`` …"""
    deltas, labels = _synthetic_two_class(seed=2)
    result = calibrate_label_aware(deltas=deltas, labels=labels)
    assert list(result.per_dim.keys()) == ["dim_0", "dim_1"]


def test_dim_labels_custom_preserved():
    """Custom ``dim_labels`` keys are preserved in iteration order."""
    deltas, labels = _synthetic_two_class(seed=3)
    result = calibrate_label_aware(
        deltas=deltas, labels=labels,
        dim_labels=["my_sep", "my_noise"],
    )
    assert list(result.per_dim.keys()) == ["my_sep", "my_noise"]


def test_direction_vector_is_unit_norm_and_sign_oriented():
    """LDA direction is unit-norm and sign-oriented (pos . diff > 0)."""
    deltas, labels = _synthetic_two_class(sep_mean=4.0, seed=8)
    result = calibrate_label_aware(deltas=deltas, labels=labels)
    w = result.signed_direction_vector
    assert np.linalg.norm(w) == pytest.approx(1.0, abs=1e-6)
    mu_pos = deltas[labels == 1].mean(axis=0)
    mu_neg = deltas[labels == 0].mean(axis=0)
    assert w @ (mu_pos - mu_neg) > 0.0


def test_fisher_score_is_finite_and_positive_for_separated_classes():
    """Two well-separated classes → positive finite Fisher score."""
    deltas, labels = _synthetic_two_class(sep_mean=4.0, seed=9)
    result = calibrate_label_aware(deltas=deltas, labels=labels)
    assert np.isfinite(result.fisher_score)
    assert result.fisher_score > 0.0


def test_result_dataclass_shape():
    """Returned object is ``CalibrationResult`` carrying ``DimCalibration``."""
    deltas, labels = _synthetic_two_class(seed=4)
    result = calibrate_label_aware(deltas=deltas, labels=labels)
    assert isinstance(result, CalibrationResult)
    for cal in result.per_dim.values():
        assert isinstance(cal, DimCalibration)
        assert isinstance(cal.mu_pos, float)
        assert isinstance(cal.sigma_pos, float)
        assert isinstance(cal.mu_neg, float)
        assert isinstance(cal.sigma_neg, float)
        assert isinstance(cal.direction, float)


# ── Input validation ──────────────────────────────────────────────────


def test_rejects_1d_deltas():
    """1-D deltas → ValueError (mirrors dim_audit contract)."""
    deltas = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([1, 0, 1, 0])
    with pytest.raises(ValueError, match="2-D"):
        calibrate_label_aware(deltas=deltas, labels=labels)


def test_rejects_shape_mismatched_labels():
    """labels.shape != (n,) → ValueError."""
    deltas = np.zeros((4, 2))
    labels = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="labels shape"):
        calibrate_label_aware(deltas=deltas, labels=labels)


def test_rejects_dim_labels_length_mismatch():
    """len(dim_labels) != n_dims → ValueError."""
    deltas, labels = _synthetic_two_class(seed=5)
    with pytest.raises(ValueError, match="dim_labels length"):
        calibrate_label_aware(
            deltas=deltas, labels=labels,
            dim_labels=["only_one"],  # n_dims is 2
        )


def test_single_class_propagates_lda_error():
    """``fit_lda_direction`` errors are re-raised verbatim — caller's policy."""
    deltas = np.random.RandomState(0).randn(20, 3)
    labels = np.ones(20, dtype=int)
    with pytest.raises(ValueError, match="both classes must be present"):
        calibrate_label_aware(deltas=deltas, labels=labels)


def test_deterministic():
    """Identical inputs produce byte-identical outputs."""
    deltas, labels = _synthetic_two_class(seed=12)
    r1 = calibrate_label_aware(deltas=deltas, labels=labels)
    r2 = calibrate_label_aware(deltas=deltas, labels=labels)
    np.testing.assert_array_equal(
        r1.signed_direction_vector, r2.signed_direction_vector,
    )
    assert r1.fisher_score == r2.fisher_score
    for name in r1.per_dim:
        assert r1.per_dim[name] == r2.per_dim[name]
