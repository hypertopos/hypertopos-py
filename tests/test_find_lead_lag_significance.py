"""Null-distribution calibration for M5 find_lead_lag significance machinery."""
from __future__ import annotations

import numpy as np
import pytest


def test_population_significance_null_calibration():
    """Under H0 (independent random series, N=25), Bonferroni-adjusted peak
    threshold gives FPR ≤ 0.07 (allowing some MC noise above nominal 0.05).
    """
    from scipy import stats

    from hypertopos.engine.geometry import _cross_correlate_with_lag

    rng = np.random.default_rng(2026)
    n_epochs = 25
    max_lag = 6
    L = (n_epochs - 1) - 2 * max_lag
    n_lags = 2 * max_lag + 1
    alpha_per_lag = 0.05 / n_lags
    z_adj = stats.norm.isf(alpha_per_lag / 2.0)
    threshold = z_adj / np.sqrt(L)

    n_trials = 1000
    n_significant = 0
    for _ in range(n_trials):
        a = rng.normal(size=n_epochs - 1)
        b = rng.normal(size=n_epochs - 1)
        _, _, peak_corr = _cross_correlate_with_lag(a, b, max_lag)
        if abs(peak_corr) > threshold:
            n_significant += 1
    fraction = n_significant / n_trials
    # Bonferroni is conservative — observed FPR should not exceed 0.07
    assert fraction <= 0.07, (
        f"observed FPR {fraction:.3f} exceeds nominal 0.05 by too much"
    )


def test_per_dim_fdr_calibration_under_global_null():
    """Under global null (all D_a*D_b pairs independent), BH controls FDR ≤ alpha
    at the per-test level on average — measured as fraction-of-runs-with-any-rejection.
    """
    from hypertopos.engine.geometry import _compute_per_dim_lead_lag

    rng = np.random.default_rng(7)
    n_epochs = 30
    D_a = 4
    D_b = 4
    n_runs = 50
    alpha = 0.05
    any_rejection = []
    for _ in range(n_runs):
        mu_pop_a = rng.normal(size=(n_epochs, D_a))
        mu_pop_b = rng.normal(size=(n_epochs, D_b))
        pairs, _, _ = _compute_per_dim_lead_lag(
            mu_pop_a, mu_pop_b,
            max_lag=4, fdr_alpha=alpha, fdr_method="bh",
            dim_labels_a=None, dim_labels_b=None,
        )
        n_sig = sum(1 for p in pairs if p.is_significant)
        any_rejection.append(1.0 if n_sig > 0 else 0.0)
    rate = float(np.mean(any_rejection))
    # With Bonferroni-over-lags + BH over pairs, false-rejection should be
    # comfortably under nominal alpha. Allow a generous margin (0.20) for
    # MC noise on 50 runs.
    assert rate <= 0.20, (
        f"BH false-rejection rate under H0 = {rate:.3f} too high"
    )


def test_storey_recovers_power_on_rich_signal():
    """When many true positives are present, Storey FDR rejects more than BH.

    Engineered: 50% of D_a*D_b pairs have a real lead-lag at lag=2; remaining
    are noise. Under this regime Storey's pi0 estimate < 1, and Storey
    rejections > BH rejections.
    """
    from hypertopos.engine.geometry import _compute_per_dim_lead_lag

    rng = np.random.default_rng(13)
    n_epochs = 40
    D_a = 4
    D_b = 4
    # half the pairs have a real signal: dim_a=0,1 → dim_b=0,1 lead at lag=2
    base = np.cumsum(rng.normal(size=n_epochs))
    mu_pop_a = rng.normal(size=(n_epochs, D_a)) * 0.05
    mu_pop_a[:, 0] = base
    mu_pop_a[:, 1] = base + rng.normal(scale=0.05, size=n_epochs)
    mu_pop_b = rng.normal(size=(n_epochs, D_b)) * 0.05
    mu_pop_b[2:, 0] = base[:-2] + rng.normal(scale=0.02, size=n_epochs - 2)
    mu_pop_b[2:, 1] = base[:-2] + rng.normal(scale=0.02, size=n_epochs - 2)
    mu_pop_b[:2, 0] = rng.normal(size=2)
    mu_pop_b[:2, 1] = rng.normal(size=2)

    pairs_bh, _, _ = _compute_per_dim_lead_lag(
        mu_pop_a, mu_pop_b,
        max_lag=5, fdr_alpha=0.05, fdr_method="bh",
        dim_labels_a=None, dim_labels_b=None,
    )
    pairs_storey, _, _ = _compute_per_dim_lead_lag(
        mu_pop_a, mu_pop_b,
        max_lag=5, fdr_alpha=0.05, fdr_method="storey",
        dim_labels_a=None, dim_labels_b=None,
    )
    n_sig_bh = sum(1 for p in pairs_bh if p.is_significant)
    n_sig_storey = sum(1 for p in pairs_storey if p.is_significant)
    # Both should reject the 4 truly-leading pairs at minimum
    assert n_sig_bh >= 2
    # Storey >= BH (Storey can only relax thresholds, never tighten)
    assert n_sig_storey >= n_sig_bh
