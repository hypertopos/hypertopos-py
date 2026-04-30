"""Unit tests for engine.density_gaps building blocks."""
from __future__ import annotations

import numpy as np
from scipy import stats

from hypertopos.engine.density_gaps import (
    ECDFEntry,
    compute_density_gaps_for_pair,
    is_usable_for_gap,
    select_pairs_by_corr,
)


def test_ecdf_transform_produces_uniform_marginals():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    e = ECDFEntry.from_values(x)
    u = e.transform(x)
    ks = stats.kstest(u, "uniform")
    assert ks.pvalue > 0.05


def test_ecdf_inverse_roundtrip_within_quantile_resolution():
    x = np.linspace(0, 1, 1000)
    e = ECDFEntry.from_values(x)
    u = e.transform(x)
    x_back = e.inverse(u)
    assert np.allclose(x_back, x, atol=1e-2)


def test_select_pairs_by_corr_respects_window():
    corr = np.array([
        [1.0, 0.05, 0.5, 0.9],
        [0.05, 1.0, 0.3, 0.8],
        [0.5, 0.3, 1.0, 0.4],
        [0.9, 0.8, 0.4, 1.0],
    ])
    pairs = select_pairs_by_corr(corr, r_min=0.1, r_max=0.7, top_k=10)
    pair_set = {(i, j) for (i, j, _r) in pairs}
    assert (0, 2) in pair_set
    assert (0, 3) not in pair_set
    assert (0, 1) not in pair_set


def test_density_gaps_flags_planted_hole():
    rng = np.random.default_rng(0)
    n = 5000
    u_i = rng.uniform(size=n)
    u_j = rng.uniform(size=n)
    mask = ~((u_i > 0.3) & (u_i < 0.6) & (u_j > 0.3) & (u_j < 0.6))
    u_i, u_j = u_i[mask], u_j[mask]
    cells = compute_density_gaps_for_pair(
        u_i, u_j, n=len(u_i), bins=10, alpha=0.05,
    )
    flagged = [c for c in cells if c["is_gap"]]
    assert len(flagged) >= 4
    in_hole = [
        c for c in flagged
        if 0.3 <= c["u_range_i"][0] < 0.6 and 0.3 <= c["u_range_j"][0] < 0.6
    ]
    assert len(in_hole) >= 4


def test_is_usable_for_gap_rejects_degenerate():
    assert not is_usable_for_gap(np.zeros(100))[0]
    assert not is_usable_for_gap(np.array([0, 1] * 50))[0]
    assert not is_usable_for_gap(np.arange(20))[0]
    assert is_usable_for_gap(np.linspace(0, 100, 200))[0]


def test_is_usable_for_gap_too_sparse_returns_reason():
    ok, reason = is_usable_for_gap(np.arange(20))
    assert not ok
    assert reason == "too_sparse"


def test_density_gaps_independence_null_yields_few_rejections():
    rng = np.random.default_rng(7)
    n = 10000
    u_i = rng.uniform(size=n)
    u_j = rng.uniform(size=n)
    cells = compute_density_gaps_for_pair(
        u_i, u_j, n=n, bins=10, alpha=0.05,
    )
    flagged = [c for c in cells if c["is_gap"]]
    assert len(flagged) <= 5


def test_select_pairs_by_corr_top_k_truncation():
    corr = np.array([
        [1.0, 0.2, 0.3, 0.4],
        [0.2, 1.0, 0.5, 0.6],
        [0.3, 0.5, 1.0, 0.55],
        [0.4, 0.6, 0.55, 1.0],
    ])
    pairs = select_pairs_by_corr(corr, r_min=0.1, r_max=0.7, top_k=2)
    assert len(pairs) == 2
    assert pairs[0][2] >= pairs[1][2]


def test_density_gaps_returns_q_values_in_unit_interval():
    rng = np.random.default_rng(1)
    u_i = rng.uniform(size=2000)
    u_j = rng.uniform(size=2000)
    cells = compute_density_gaps_for_pair(
        u_i, u_j, n=2000, bins=10, alpha=0.05,
    )
    for c in cells:
        assert 0.0 <= c["q_value"] <= 1.0
        assert 0.0 <= c["p_value"] <= 1.0
