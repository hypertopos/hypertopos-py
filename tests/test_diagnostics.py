"""Tests for `engine.diagnostics.levene_test_per_group` — Brown-Forsythe
(median-centred Levene) homoscedasticity primitive. Synthetic
equal-variance, 10x-variance, and low-N-skip fixtures exercise the
three branches the builder calibration pass relies on.
"""
from __future__ import annotations

import numpy as np
import pytest

from hypertopos.engine.diagnostics import (
    MIN_GROUP_SIZE,
    levene_test_per_group,
)


def test_equal_variance_no_rejection():
    """Three groups, identical N(0, 1) — W small, p > 0.5 (well above
    the 0.01 warning threshold)."""
    rng = np.random.default_rng(0)
    n = 200
    values = np.concatenate([
        rng.normal(0.0, 1.0, n),
        rng.normal(0.0, 1.0, n),
        rng.normal(0.0, 1.0, n),
    ])
    group_ids = np.concatenate([
        np.full(n, "A"),
        np.full(n, "B"),
        np.full(n, "C"),
    ])
    result = levene_test_per_group(values, group_ids)
    assert result["k_groups"] == 3
    assert result["skipped_groups_low_n"] == 0
    assert result["W_statistic"] is not None
    assert result["p_value"] > 0.5


def test_one_group_with_10x_variance_strong_rejection():
    """Three groups, one with 10x the variance of the others — W large,
    p < 0.001 (orders of magnitude below the warning threshold)."""
    rng = np.random.default_rng(1)
    n = 200
    values = np.concatenate([
        rng.normal(0.0, 1.0, n),
        rng.normal(0.0, 1.0, n),
        rng.normal(0.0, 10.0, n),
    ])
    group_ids = np.concatenate([
        np.full(n, "low_1"),
        np.full(n, "low_2"),
        np.full(n, "high_var"),
    ])
    result = levene_test_per_group(values, group_ids)
    assert result["k_groups"] == 3
    assert result["W_statistic"] > 10.0
    assert result["p_value"] < 1e-3
    # High-variance group has roughly 100x the variance of low groups
    var_high = result["per_group_variance"]["high_var"]
    var_low = result["per_group_variance"]["low_1"]
    assert var_high / var_low > 50.0


def test_low_n_groups_silently_skipped():
    """Groups with N < MIN_GROUP_SIZE are dropped from the test and
    counted in skipped_groups_low_n — no error raised."""
    rng = np.random.default_rng(2)
    big_n = 100
    small_n = MIN_GROUP_SIZE - 1  # below the threshold
    values = np.concatenate([
        rng.normal(0.0, 1.0, big_n),
        rng.normal(0.0, 1.0, big_n),
        rng.normal(0.0, 1.0, small_n),
        rng.normal(0.0, 1.0, small_n),
    ])
    group_ids = np.concatenate([
        np.full(big_n, "big_A"),
        np.full(big_n, "big_B"),
        np.full(small_n, "small_X"),
        np.full(small_n, "small_Y"),
    ])
    result = levene_test_per_group(values, group_ids)
    assert result["k_groups"] == 2
    assert result["skipped_groups_low_n"] == 2
    assert set(result["per_group_n"].keys()) == {"big_A", "big_B"}
    assert result["W_statistic"] is not None


def test_fewer_than_two_qualifying_groups_returns_none():
    """Only one group survives the low-N filter — Levene's test cannot
    be computed; W and p are None but no error raised."""
    rng = np.random.default_rng(3)
    values = np.concatenate([
        rng.normal(0.0, 1.0, 100),
        rng.normal(0.0, 1.0, 5),  # below MIN_GROUP_SIZE
    ])
    group_ids = np.concatenate([
        np.full(100, "big"),
        np.full(5, "tiny"),
    ])
    result = levene_test_per_group(values, group_ids)
    assert result["k_groups"] == 1
    assert result["W_statistic"] is None
    assert result["p_value"] is None
    assert result["skipped_groups_low_n"] == 1


def test_shape_mismatch_raises():
    """Shape mismatch between values and group_ids surfaces an explicit
    ValueError rather than silently truncating."""
    with pytest.raises(ValueError, match="same shape"):
        levene_test_per_group(
            np.zeros(10, dtype=np.float64),
            np.zeros(5, dtype=str),
        )
