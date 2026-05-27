# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Unit tests for engine.topology.local_trajectory_shape.

Engineered-input tests prove the classifier discriminates between the four
categories (arch / V / linear / flat) and degenerate inputs (<3 samples,
all-zero).
"""
from __future__ import annotations

from hypertopos.engine.topology import local_trajectory_shape


def test_arch_shape():
    """Up-then-down series with interior maximum → arch."""
    assert local_trajectory_shape([1.0, 2.0, 3.0, 2.0, 1.0]) == "arch"


def test_arch_asymmetric():
    """Asymmetric up-then-down still classified as arch (max strictly interior)."""
    assert local_trajectory_shape([0.5, 4.0, 2.0]) == "arch"


def test_v_shape():
    """Down-then-up series with interior minimum → V."""
    assert local_trajectory_shape([3.0, 2.0, 1.0, 2.0, 3.0]) == "V"


def test_v_asymmetric():
    """Asymmetric down-then-up still classified as V (min strictly interior)."""
    assert local_trajectory_shape([4.0, 0.5, 2.0]) == "V"


def test_linear_monotone_increasing():
    """Strictly increasing series → linear."""
    assert local_trajectory_shape([1.0, 2.0, 3.0, 4.0]) == "linear"


def test_linear_monotone_decreasing():
    """Strictly decreasing series → linear."""
    assert local_trajectory_shape([4.0, 3.0, 2.0, 1.0]) == "linear"


def test_flat_low_variance():
    """Range under 10% of mean → flat."""
    # mean=1.0, range=0.02 → 2% of mean, well below the 10% threshold.
    assert local_trajectory_shape([1.0, 1.01, 0.99]) == "flat"


def test_flat_exactly_at_threshold():
    """Range exactly 10% of mean is NOT flat (strict inequality)."""
    # mean=1.0, range=0.1 → 10% of mean, classifier requires < 10%.
    result = local_trajectory_shape([0.95, 1.0, 1.05])
    assert result != "flat"


def test_too_short_two_samples():
    """<3 samples returns None — temporal shape needs at least 3 points."""
    assert local_trajectory_shape([1.0, 2.0]) is None


def test_too_short_one_sample():
    """Single sample returns None."""
    assert local_trajectory_shape([1.0]) is None


def test_too_short_empty():
    """Empty input returns None."""
    assert local_trajectory_shape([]) is None


def test_all_zero_series():
    """All-zero series — pathological but stable: classifier returns 'linear'.

    `mean=0` defeats the flat guard (`mean > 0` required); all diffs are 0
    so neither pos nor neg dominates; max_idx == min_idx == 0 (both at the
    first index by Python's argmax/argmin tie-break), so the arch/V interior
    checks fail and the helper falls through to 'linear'. Locked here so a
    future "fix" doesn't silently flip this category for zero-delta solids.
    """
    assert local_trajectory_shape([0.0, 0.0, 0.0]) == "linear"


def test_constant_nonzero_series():
    """Constant non-zero series → flat (range = 0, mean > 0)."""
    assert local_trajectory_shape([2.0, 2.0, 2.0]) == "flat"
