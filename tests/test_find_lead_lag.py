"""Engine + navigator tests for find_lead_lag (cross-pattern lead-lag)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest


# ── engine: _compute_centroid_drift_series ────────────────────────────────────


def test_centroid_drift_series_simple():
    """Centroid drift = ||mean_E(delta_t+1) - mean_E(delta_t)||."""
    from hypertopos.engine.geometry import _compute_centroid_drift_series

    shapes = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 2.0], [1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    mu = np.zeros(2, dtype=np.float32)
    sigma = np.ones(2, dtype=np.float32)
    centroid_series, volatility_series, mu_pop = _compute_centroid_drift_series(
        shapes, mu, sigma,
    )
    assert centroid_series.shape == (2,)
    assert centroid_series[0] == pytest.approx(1.0, abs=1e-3)
    assert centroid_series[1] == pytest.approx(2.0, abs=1e-3)
    assert volatility_series[0] == pytest.approx(1.0, abs=1e-3)
    assert volatility_series[1] == pytest.approx(2.0, abs=1e-3)
    assert mu_pop.shape == (3, 2)


def test_centroid_drift_uses_sigma_floor():
    """Sigma floor 1e-2 applied — does not divide by zero on σ=0 dim."""
    from hypertopos.engine.geometry import _compute_centroid_drift_series

    shapes = np.zeros((3, 5, 2), dtype=np.float32)
    shapes[1, :, 0] = 1.0
    mu = np.zeros(2, dtype=np.float32)
    sigma = np.array([0.0, 1.0], dtype=np.float32)  # σ=0 → flooring kicks in
    centroid_series, volatility_series, mu_pop = _compute_centroid_drift_series(
        shapes, mu, sigma,
    )
    assert np.isfinite(centroid_series).all()
    assert np.isfinite(volatility_series).all()


# ── engine: _cross_correlate_with_lag ─────────────────────────────────────────


def test_cross_correlate_known_lag():
    """A leads B by 2 epochs at corr ≈ 1.0."""
    from hypertopos.engine.geometry import _cross_correlate_with_lag

    n = 30
    rng = np.random.default_rng(42)
    a = rng.normal(size=n)
    # B follows A by 2: b[t] = a[t-2] + noise. Then a[t] correlates with b[t+2] → positive lag 2.
    b = np.concatenate([rng.normal(size=2), a[:-2]]) + rng.normal(scale=0.05, size=n)
    max_lag = 5
    corr_by_lag, peak_lag, peak_corr = _cross_correlate_with_lag(a, b, max_lag)
    assert corr_by_lag.shape == (2 * max_lag + 1,)
    assert peak_lag == 2
    assert peak_corr > 0.95


def test_cross_correlate_no_signal_returns_low_corr():
    """Independent random series — peak |corr| should be modest at N=30."""
    from hypertopos.engine.geometry import _cross_correlate_with_lag

    rng = np.random.default_rng(0)
    n = 30
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    _, _, peak_corr = _cross_correlate_with_lag(a, b, max_lag=5)
    assert abs(peak_corr) < 0.7


def test_cross_correlate_negative_lag():
    """B leads A by 3 → peak lag = -3."""
    from hypertopos.engine.geometry import _cross_correlate_with_lag

    n = 40
    rng = np.random.default_rng(11)
    base = rng.normal(size=n)
    # B leads A by 3: a[t] = b[t-3]. So a[t] correlates with b[t-3] → lag = -3.
    a = np.concatenate([rng.normal(size=3), base[:-3]]) + rng.normal(scale=0.05, size=n)
    b = base.copy()
    _, peak_lag, peak_corr = _cross_correlate_with_lag(a, b, max_lag=6)
    assert peak_lag == -3
    assert peak_corr > 0.9


def test_cross_correlate_zero_variance_safe():
    """Constant series → returns 0 corr, no division-by-zero."""
    from hypertopos.engine.geometry import _cross_correlate_with_lag

    n = 20
    a = np.zeros(n)
    b = np.arange(n, dtype=np.float64)
    corr_by_lag, _, peak_corr = _cross_correlate_with_lag(a, b, max_lag=4)
    assert np.all(corr_by_lag == 0.0)
    assert peak_corr == 0.0


# ── engine: _compute_per_dim_lead_lag ─────────────────────────────────────────


def test_per_dim_matrix_engineered_leading_pair():
    """One (dim_a=0, dim_b=0) pair leads at lag=2; rest is noise."""
    from hypertopos.engine.geometry import _compute_per_dim_lead_lag

    rng = np.random.default_rng(7)
    n_epochs = 40
    D_a = 4
    D_b = 4
    base = np.cumsum(rng.normal(size=n_epochs))
    mu_pop_a = rng.normal(size=(n_epochs, D_a)) * 0.05
    mu_pop_a[:, 0] = base
    mu_pop_b = rng.normal(size=(n_epochs, D_b)) * 0.05
    mu_pop_b[2:, 0] = base[:-2]
    mu_pop_b[:2, 0] = rng.normal(size=2)
    pairs, _, _ = _compute_per_dim_lead_lag(
        mu_pop_a, mu_pop_b,
        max_lag=5,
        fdr_alpha=0.05,
        fdr_method="bh",
        dim_labels_a=[f"a{i}" for i in range(D_a)],
        dim_labels_b=[f"b{j}" for j in range(D_b)],
    )
    assert len(pairs) == 16
    pair_00 = next(p for p in pairs if p.dim_index_a == 0 and p.dim_index_b == 0)
    assert pair_00.lag == 2
    assert pair_00.correlation > 0.7
    assert pair_00.is_significant is True
    n_sig = sum(1 for p in pairs if p.is_significant)
    # Some false positives possible at α=0.05 over 16 tests; tolerate up to 3
    assert n_sig <= 3


def test_per_dim_matrix_dim_labels_passthrough():
    """dim_label_a/b populated when dim_labels lists are passed in."""
    from hypertopos.engine.geometry import _compute_per_dim_lead_lag

    rng = np.random.default_rng(123)
    n_epochs = 30
    D_a = 2
    D_b = 3
    mu_pop_a = rng.normal(size=(n_epochs, D_a))
    mu_pop_b = rng.normal(size=(n_epochs, D_b))
    pairs, _, _ = _compute_per_dim_lead_lag(
        mu_pop_a, mu_pop_b,
        max_lag=4, fdr_alpha=0.05, fdr_method="bh",
        dim_labels_a=["alpha", "beta"],
        dim_labels_b=["one", "two", "three"],
    )
    assert len(pairs) == 6
    labels = {(p.dim_label_a, p.dim_label_b) for p in pairs}
    assert ("alpha", "one") in labels
    assert ("beta", "three") in labels


def test_per_dim_matrix_dim_labels_none_safe():
    """dim_labels_a/b = None → dim_label fields are None (no crash)."""
    from hypertopos.engine.geometry import _compute_per_dim_lead_lag

    rng = np.random.default_rng(1)
    n_epochs = 30
    mu_pop_a = rng.normal(size=(n_epochs, 2))
    mu_pop_b = rng.normal(size=(n_epochs, 2))
    pairs, _, _ = _compute_per_dim_lead_lag(
        mu_pop_a, mu_pop_b,
        max_lag=3, fdr_alpha=0.05, fdr_method="bh",
        dim_labels_a=None, dim_labels_b=None,
    )
    assert all(p.dim_label_a is None and p.dim_label_b is None for p in pairs)


# ── engine: _compute_lead_lag_report orchestrator ─────────────────────────────


def test_compute_lead_lag_report_synthetic_a_leads_b():
    """End-to-end: synthetic shapes where A leads B by 2 → headline lag=2."""
    from hypertopos.engine.geometry import _compute_lead_lag_report

    rng = np.random.default_rng(11)
    n_epochs = 30
    n_entities = 50
    D = 3
    base_traj_a = np.cumsum(rng.normal(size=n_epochs))
    base_traj_b = np.concatenate([rng.normal(size=2), base_traj_a[:-2]])
    shapes_a = np.zeros((n_epochs, n_entities, D), dtype=np.float32)
    shapes_b = np.zeros((n_epochs, n_entities, D), dtype=np.float32)
    for e in range(n_entities):
        shapes_a[:, e, 0] = base_traj_a + rng.normal(scale=0.05, size=n_epochs)
        shapes_b[:, e, 0] = base_traj_b + rng.normal(scale=0.05, size=n_epochs)
    shapes_a[:, :, 1:] = rng.normal(scale=0.1, size=(n_epochs, n_entities, D - 1)).astype(np.float32)
    shapes_b[:, :, 1:] = rng.normal(scale=0.1, size=(n_epochs, n_entities, D - 1)).astype(np.float32)
    mu = np.zeros(D, dtype=np.float32)
    sigma = np.ones(D, dtype=np.float32)
    epoch0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [epoch0 + timedelta(days=2 * i) for i in range(n_epochs)]
    report = _compute_lead_lag_report(
        pattern_a="P_A",
        pattern_b="P_B",
        entity_key=None,
        shapes_a=shapes_a,
        shapes_b=shapes_b,
        mu_a=mu, sigma_a=sigma,
        mu_b=mu, sigma_b=sigma,
        dim_labels_a=["a0", "a1", "a2"],
        dim_labels_b=["b0", "b1", "b2"],
        timestamps=timestamps,
        n_dropped_a=0, n_dropped_b=0,
        cohort_size=n_entities, cohort_dropped=0,
        schema_hash_a="aaa", schema_hash_b="bbb",
        max_lag=5,
        fdr_alpha=0.05,
        fdr_method="bh",
        verbose=False,
    )
    assert report.lag == 2
    assert report.correlation > 0.7
    assert report.is_significant is True
    assert report.n_epochs_used == n_epochs
    assert report.cohort_size == n_entities
    assert report.reliability == "high"
    pair_00 = next(p for p in report.top_dim_pairs if p.dim_index_a == 0 and p.dim_index_b == 0)
    assert pair_00.lag == 2
    assert report.per_dim_pairs is None
    assert report.coverage_warning is False
    assert isinstance(report.timestamp_from, datetime)


def test_compute_lead_lag_report_verbose_includes_full_matrix():
    from hypertopos.engine.geometry import _compute_lead_lag_report

    rng = np.random.default_rng(1)
    n_epochs = 16
    n_entities = 40
    D = 2
    shapes_a = rng.normal(size=(n_epochs, n_entities, D)).astype(np.float32)
    shapes_b = rng.normal(size=(n_epochs, n_entities, D)).astype(np.float32)
    mu = np.zeros(D, dtype=np.float32)
    sigma = np.ones(D, dtype=np.float32)
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        for i in range(n_epochs)
    ]
    report = _compute_lead_lag_report(
        pattern_a="P_A", pattern_b="P_B", entity_key=None,
        shapes_a=shapes_a, shapes_b=shapes_b,
        mu_a=mu, sigma_a=sigma, mu_b=mu, sigma_b=sigma,
        dim_labels_a=None, dim_labels_b=None,
        timestamps=timestamps,
        n_dropped_a=0, n_dropped_b=0,
        cohort_size=n_entities, cohort_dropped=0,
        schema_hash_a="x", schema_hash_b="x",
        max_lag=3, fdr_alpha=0.05, fdr_method="bh",
        verbose=True,
    )
    assert report.per_dim_pairs is not None
    assert len(report.per_dim_pairs) == report.n_dim_pairs
    assert report.reliability == "medium"


def test_reliability_label_thresholds():
    """N - 1 >= 24 → high; >= 12 → medium; else low."""
    from hypertopos.engine.geometry import _reliability_label_for_lead_lag

    assert _reliability_label_for_lead_lag(25) == "high"
    assert _reliability_label_for_lead_lag(13) == "medium"
    assert _reliability_label_for_lead_lag(12) == "low"
    assert _reliability_label_for_lead_lag(8) == "low"


# ── discriminator (advisor-style) ─────────────────────────────────────────────


def test_compute_lead_lag_report_degenerate_signal_flag():
    """When centroid drift series is essentially constant, degenerate_signal=True
    and agreement='divergent' regardless of correlation values.
    """
    from hypertopos.engine.geometry import _compute_lead_lag_report

    n_epochs = 16
    n_entities = 30
    D = 2
    # Constant shapes per (entity, epoch): centroid never moves
    shapes_a = np.ones((n_epochs, n_entities, D), dtype=np.float32)
    shapes_b = np.ones((n_epochs, n_entities, D), dtype=np.float32) * 2.0
    mu = np.zeros(D, dtype=np.float32)
    sigma = np.ones(D, dtype=np.float32)
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        for i in range(n_epochs)
    ]
    report = _compute_lead_lag_report(
        pattern_a="P_A", pattern_b="P_B", entity_key=None,
        shapes_a=shapes_a, shapes_b=shapes_b,
        mu_a=mu, sigma_a=sigma, mu_b=mu, sigma_b=sigma,
        dim_labels_a=None, dim_labels_b=None,
        timestamps=timestamps,
        n_dropped_a=0, n_dropped_b=0,
        cohort_size=n_entities, cohort_dropped=0,
        schema_hash_a="x", schema_hash_b="x",
        max_lag=3, fdr_alpha=0.05, fdr_method="bh",
        verbose=False,
    )
    assert report.degenerate_signal is True
    assert report.agreement == "divergent"
    assert report.is_significant is False


def test_compute_lead_lag_report_coverage_warning_small_cohort():
    """coverage_warning=True when cohort_size < 30."""
    from hypertopos.engine.geometry import _compute_lead_lag_report

    rng = np.random.default_rng(3)
    n_epochs = 16
    n_entities = 20  # below 30 threshold
    D = 2
    shapes_a = rng.normal(size=(n_epochs, n_entities, D)).astype(np.float32)
    shapes_b = rng.normal(size=(n_epochs, n_entities, D)).astype(np.float32)
    mu = np.zeros(D, dtype=np.float32)
    sigma = np.ones(D, dtype=np.float32)
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        for i in range(n_epochs)
    ]
    report = _compute_lead_lag_report(
        pattern_a="P_A", pattern_b="P_B", entity_key=None,
        shapes_a=shapes_a, shapes_b=shapes_b,
        mu_a=mu, sigma_a=sigma, mu_b=mu, sigma_b=sigma,
        dim_labels_a=None, dim_labels_b=None,
        timestamps=timestamps,
        n_dropped_a=0, n_dropped_b=0,
        cohort_size=n_entities, cohort_dropped=0,
        schema_hash_a="x", schema_hash_b="x",
        max_lag=3, fdr_alpha=0.05, fdr_method="bh",
        verbose=False,
    )
    assert report.coverage_warning is True


def test_compute_lead_lag_report_entity_key_mode():
    """entity_key drill-down: cohort_size=1, entity_key surfaced in report."""
    from hypertopos.engine.geometry import _compute_lead_lag_report

    rng = np.random.default_rng(17)
    n_epochs = 30
    D = 2
    # Single-entity tensor (n_epochs, 1, D)
    base = np.cumsum(rng.normal(size=n_epochs)).astype(np.float32)
    shapes_a = np.zeros((n_epochs, 1, D), dtype=np.float32)
    shapes_b = np.zeros((n_epochs, 1, D), dtype=np.float32)
    shapes_a[:, 0, 0] = base
    shapes_b[2:, 0, 0] = base[:-2]
    shapes_b[:2, 0, 0] = rng.normal(size=2).astype(np.float32)
    mu = np.zeros(D, dtype=np.float32)
    sigma = np.ones(D, dtype=np.float32)
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        for i in range(n_epochs)
    ]
    report = _compute_lead_lag_report(
        pattern_a="P_A", pattern_b="P_B",
        entity_key="entity_42",
        shapes_a=shapes_a, shapes_b=shapes_b,
        mu_a=mu, sigma_a=sigma, mu_b=mu, sigma_b=sigma,
        dim_labels_a=None, dim_labels_b=None,
        timestamps=timestamps,
        n_dropped_a=0, n_dropped_b=0,
        cohort_size=1, cohort_dropped=None,
        schema_hash_a="x", schema_hash_b="x",
        max_lag=5, fdr_alpha=0.05, fdr_method="bh",
        verbose=False,
    )
    assert report.entity_key == "entity_42"
    assert report.cohort_size == 1
    assert report.cohort_dropped is None
    # Should still detect lag=2 in the engineered single-entity series
    assert report.lag == 2
    assert report.correlation > 0.7
    # cohort_size=1 < 30 → coverage warning
    assert report.coverage_warning is True


def test_compute_lead_lag_report_schema_hash_mismatch_passes():
    """schema_hash differs across patterns is NOT a raise — cross-pattern is the point.

    Hashes are reported for traceability, not as a gate.
    """
    from hypertopos.engine.geometry import _compute_lead_lag_report

    rng = np.random.default_rng(2)
    n_epochs = 16
    n_entities = 40
    D = 3
    shapes_a = rng.normal(size=(n_epochs, n_entities, D)).astype(np.float32)
    shapes_b = rng.normal(size=(n_epochs, n_entities, D)).astype(np.float32)
    mu = np.zeros(D, dtype=np.float32)
    sigma = np.ones(D, dtype=np.float32)
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        for i in range(n_epochs)
    ]
    report = _compute_lead_lag_report(
        pattern_a="P_A", pattern_b="P_B", entity_key=None,
        shapes_a=shapes_a, shapes_b=shapes_b,
        mu_a=mu, sigma_a=sigma, mu_b=mu, sigma_b=sigma,
        dim_labels_a=None, dim_labels_b=None,
        timestamps=timestamps,
        n_dropped_a=0, n_dropped_b=0,
        cohort_size=n_entities, cohort_dropped=0,
        schema_hash_a="hash_aaa_1234567890ab",
        schema_hash_b="hash_bbb_0987654321cd",
        max_lag=3, fdr_alpha=0.05, fdr_method="bh",
        verbose=False,
    )
    assert report.schema_hash_a == "hash_aaa_1234567890ab"
    assert report.schema_hash_b == "hash_bbb_0987654321cd"


def test_per_dim_discriminator_distinct_lags():
    """Engineered: pair (0,0) leads at lag=2, pair (0,1) at lag=0. Distinct lags
    must surface — guards against lag-collapse regressions.
    """
    from hypertopos.engine.geometry import _compute_per_dim_lead_lag

    rng = np.random.default_rng(99)
    n_epochs = 40
    D_a = 3
    D_b = 3
    base_a0 = np.cumsum(rng.normal(size=n_epochs))
    base_b1_concurrent = base_a0 + rng.normal(scale=0.05, size=n_epochs)
    base_b0_lag2 = np.concatenate([rng.normal(size=2), base_a0[:-2]])
    mu_pop_a = rng.normal(size=(n_epochs, D_a)) * 0.05
    mu_pop_a[:, 0] = base_a0
    mu_pop_b = rng.normal(size=(n_epochs, D_b)) * 0.05
    mu_pop_b[:, 0] = base_b0_lag2
    mu_pop_b[:, 1] = base_b1_concurrent
    pairs, _, _ = _compute_per_dim_lead_lag(
        mu_pop_a, mu_pop_b,
        max_lag=5, fdr_alpha=0.05, fdr_method="bh",
        dim_labels_a=None, dim_labels_b=None,
    )
    pair_00 = next(p for p in pairs if p.dim_index_a == 0 and p.dim_index_b == 0)
    pair_01 = next(p for p in pairs if p.dim_index_a == 0 and p.dim_index_b == 1)
    assert pair_00.lag == 2
    assert pair_01.lag == 0
    assert pair_00.is_significant
    assert pair_01.is_significant
    assert pair_00.lag != pair_01.lag
