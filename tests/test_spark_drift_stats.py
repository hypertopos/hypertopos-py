# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Drift statistics from mergeable counts agree with the exact per-row forms."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyspark")

from scipy.stats import betabinom, kstest, kstwo  # noqa: E402

from hypertopos.engine import robust  # noqa: E402
from hypertopos.spark import _drift_stats as ds  # noqa: E402

D = 4
ZERO_SHARE = {2: 0.90, 3: 0.99}  # tied dimensions: most of their mass sits at one value


def _draw(rng: np.random.Generator, n: int, shift: float = 0.0) -> np.ndarray:
    x = np.exp(rng.standard_normal((n, D)) + shift)
    for d, share in ZERO_SHARE.items():
        x[rng.random(n) < share, d] = 0.0
    return x


@pytest.fixture(scope="module")
def batch() -> tuple[robust.Reference, np.ndarray]:
    rng = np.random.default_rng(11)
    ref, _ = robust.fit_reference(_draw(rng, 3_000), tuple(f"d{i}" for i in range(D)))
    return ref, _draw(rng, 1_500)


def _exact_grid_stat(grid: np.ndarray, levels: np.ndarray, x: np.ndarray) -> float:
    # The per-row tie-correct grid form, as in the experiment that validated it.
    col = np.sort(x)
    values, first = np.unique(grid, return_index=True)
    last = np.r_[first[1:], len(grid)] - 1
    below = np.searchsorted(col, values, side="left") / len(col)
    at_or_below = np.searchsorted(col, values, side="right") / len(col)
    return float(
        max(np.max(np.abs(below - levels[first])), np.max(np.abs(at_or_below - levels[last])))
    )


@pytest.mark.parametrize("shift", [0.0, 0.5])
def test_grid_form_from_counts_equals_the_exact_form(batch, shift):
    ref, _ = batch
    x = _draw(np.random.default_rng(12), 1_500, shift)
    for d in range(D):
        values, low, high = ds.grid_levels(ref.cdf_grid[d], ref.cdf_levels)
        counts = ds.grid_counts(values, x[:, d])
        assert counts.sum() == len(x)
        expected = _exact_grid_stat(ref.cdf_grid[d], ref.cdf_levels, x[:, d])
        assert ds.grid_ks(counts, low, high) == pytest.approx(expected, abs=1e-12)


def test_grid_counts_merge_across_partitions(batch):
    ref, x = batch
    values, _, _ = ds.grid_levels(ref.cdf_grid[2], ref.cdf_levels)
    parts = ds.grid_counts(values, x[:700, 2]) + ds.grid_counts(values, x[700:, 2])
    assert np.array_equal(parts, ds.grid_counts(values, x[:, 2]))


def _binned(p: np.ndarray) -> float:
    return ds.uniform_ks_binned(ds.p_histogram(p[: len(p) // 2]) + ds.p_histogram(p[len(p) // 2 :]))


def test_binned_p_statistic_brackets_the_exact_one(batch):
    ref, x = batch
    for p in (robust.score(ref, x).conformal_p, robust.score(ref, x * 3).conformal_p):
        exact = float(kstest(p, "uniform").statistic)
        assert exact <= _binned(p) <= exact + 1.0 / ds.P_BINS + 1e-12


def test_binned_p_statistic_is_conservative_on_bin_edges():
    # Conformal p-values sit on a lattice; with n + 1 dividing the bin count every
    # value lands on a bin edge, where rounding could undercut the exact statistic.
    rng = np.random.default_rng(3)
    for skew in (1.0, 0.7):
        p = np.ceil(512 * rng.random(2_000) ** skew) / 512
        exact = float(kstest(p, "uniform").statistic)
        assert exact <= _binned(p) <= exact + 1.0 / ds.P_BINS + 1e-12
    assert ds.p_histogram(np.array([1.0, 0.0]))[[0, -1]].tolist() == [1, 1]


def test_effective_size_and_ks_p():
    assert ds.effective_n(500, 2_000) == 400
    assert ds.effective_n(1, 1) == 1
    assert ds.ks_p(0.05, 400) == pytest.approx(float(kstwo.sf(0.05, 400)))


def test_flag_count_p_is_the_beta_binomial_tail():
    # 3,000 fitted rows at the 99th percentile: a new row is flagged among the top 30.
    assert ds.flag_count_p(40, 2_000, 3_000, 99.0) == pytest.approx(
        float(betabinom.sf(39, 2_000, 30, 3_001 - 30))
    )
    assert ds.flag_count_p(5, 2_000, 50, 99.0) is None  # no row can be flagged at all
