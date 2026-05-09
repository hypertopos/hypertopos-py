"""Tests for `compute_theta_sensitivity` + `compute_theta_sensitivity_from_sorted`.

These functions characterise how stable the anomaly threshold (`theta`) is
to perturbations of the `anomaly_percentile` parameter. The cheap path is
wired into `_BuildState._compute_population_stats` (glued onto the existing
`sorted_norms` for `delta_rank_pcts`) and populates the `theta_sensitivity`
field on `CalibrationFit`.

Tests cover:
- Output shape + key naming
- Smooth-distribution behaviour (Gaussian / uniform): adjacent-percentile
  ratios should be smooth (no cliff)
- Heavy-tail behaviour: cliffs in the upper tail are expected
- Edge cases: small n, constant array, empty
- Determinism: same seed produces identical bootstrap output
- Cheap path equivalence: from_sorted ≡ compute(n_bootstraps=0)
"""
from __future__ import annotations

import numpy as np
import pytest
from hypertopos.builder._theta_sensitivity import (
    CLIFF_RATIO_THRESHOLD,
    STABLE_BAND_RATIO_THRESHOLD,
    compute_theta_sensitivity,
    compute_theta_sensitivity_from_sorted,
    derive_stable_band_and_cliffs,
)

# ---------------------------------------------------------------------------
# Output shape + naming
# ---------------------------------------------------------------------------


def test_output_keys_match_default_percentiles():
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=10_000)
    out = compute_theta_sensitivity(delta_norms, n_bootstraps=20)
    assert set(out.keys()) == {f"p{p}" for p in range(90, 100)}


def test_output_keys_match_custom_percentiles():
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=10_000)
    out = compute_theta_sensitivity(
        delta_norms, percentiles=(85, 90, 95, 99), n_bootstraps=20,
    )
    assert set(out.keys()) == {"p85", "p90", "p95", "p99"}


def test_output_value_fields():
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=10_000)
    out = compute_theta_sensitivity(delta_norms, n_bootstraps=20)
    expected_fields = {
        "theta_mean", "theta_std",
        "anomaly_count_mean", "anomaly_count_std",
        "anomaly_rate",
    }
    for p_key, stats in out.items():
        assert set(stats.keys()) == expected_fields, f"missing fields at {p_key}"


# ---------------------------------------------------------------------------
# Smooth-distribution behaviour
# ---------------------------------------------------------------------------


def test_gaussian_anomaly_count_monotone_decreasing():
    """Higher percentile → smaller anomaly count (no exceptions)."""
    rng = np.random.default_rng(42)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=20_000)
    out = compute_theta_sensitivity(delta_norms, n_bootstraps=50)
    counts = [out[f"p{p}"]["anomaly_count_mean"] for p in range(90, 100)]
    for i in range(len(counts) - 1):
        assert counts[i] > counts[i + 1], (
            f"anomaly_count not strictly decreasing at p{90 + i} → p{91 + i}: "
            f"{counts[i]} → {counts[i + 1]}"
        )


def test_gaussian_theta_converges_to_analytical_percentile():
    """Bootstrap mean of theta should converge to the analytical
    percentile of N(0, 1). 95th = 1.645, 99th = 2.326. This tests
    bootstrap correctness, not just the percentile semantics
    (which would be a tautology since np.percentile gives the result
    by construction).
    """
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=50_000)
    out = compute_theta_sensitivity(delta_norms, n_bootstraps=50)
    # Analytical percentiles of N(0, 1)
    assert abs(out["p95"]["theta_mean"] - 1.645) < 0.05, (
        f"p95 theta_mean should converge to 1.645; got {out['p95']['theta_mean']}"
    )
    assert abs(out["p99"]["theta_mean"] - 2.326) < 0.10, (
        f"p99 theta_mean should converge to 2.326; got {out['p99']['theta_mean']}"
    )


# ---------------------------------------------------------------------------
# Heavy-tail behaviour
# ---------------------------------------------------------------------------


def test_heavy_tail_shows_cliff_at_upper_percentiles():
    """Heavy-tail distribution: ratio at the upper-percentile boundary
    is at least as large as ratio at the low-percentile boundary —
    directional test rather than a hand-tuned magnitude. Smooth
    Gaussian gives ratio ≈ 1.11 at both ends; heavy-tail magnifies the
    upper ratio relative to its smooth-distribution counterpart.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(loc=0.0, scale=1.0, size=10_000)
    # Add heavy upper tail: 200 extreme values to make the cliff
    # robust against RNG noise within the test
    base = np.concatenate([base, rng.exponential(scale=10.0, size=200) + 5.0])
    out = compute_theta_sensitivity(base, n_bootstraps=50)
    ratio_low = (
        out["p90"]["anomaly_count_mean"]
        / max(out["p91"]["anomaly_count_mean"], 1e-9)
    )
    ratio_high = (
        out["p98"]["anomaly_count_mean"]
        / max(out["p99"]["anomaly_count_mean"], 1e-9)
    )
    assert ratio_high >= ratio_low, (
        f"heavy-tail should produce ratio_high ≥ ratio_low; got "
        f"ratio_low={ratio_low:.3f}, ratio_high={ratio_high:.3f}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_constant_array_zero_std():
    """Constant delta_norm: theta_mean = constant, theta_std = 0,
    bootstrap doesn't change anything."""
    delta_norms = np.full(1000, 5.0)
    out = compute_theta_sensitivity(delta_norms, n_bootstraps=10)
    for p in range(90, 100):
        stats = out[f"p{p}"]
        assert stats["theta_mean"] == pytest.approx(5.0)
        assert stats["theta_std"] == pytest.approx(0.0, abs=1e-9)


def test_small_n_widens_bootstrap_ci():
    """Smaller n → wider bootstrap CI on theta. Verifies the bootstrap
    actually responds to sample size — at n=200 theta_std should be
    measurably larger than at n=20_000 on the same Gaussian DGP.
    """
    rng = np.random.default_rng(0)
    small_norms = rng.normal(loc=0.0, scale=1.0, size=200)
    large_norms = rng.normal(loc=0.0, scale=1.0, size=20_000)
    out_small = compute_theta_sensitivity(small_norms, n_bootstraps=50, seed=1)
    out_large = compute_theta_sensitivity(large_norms, n_bootstraps=50, seed=1)
    # Bootstrap CI at p95 should widen by at least 5× when n drops 100×.
    assert out_small["p95"]["theta_std"] > out_large["p95"]["theta_std"] * 5.0, (
        f"expected small-n theta_std > large-n × 5; got "
        f"small={out_small['p95']['theta_std']:.4f}, "
        f"large={out_large['p95']['theta_std']:.4f}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_output():
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=5_000)
    out_a = compute_theta_sensitivity(delta_norms, n_bootstraps=20, seed=42)
    out_b = compute_theta_sensitivity(delta_norms, n_bootstraps=20, seed=42)
    for p in range(90, 100):
        for field in (
            "theta_mean", "theta_std",
            "anomaly_count_mean", "anomaly_count_std",
        ):
            assert out_a[f"p{p}"][field] == out_b[f"p{p}"][field], (
                f"determinism broken at p{p}.{field}"
            )


def test_different_seeds_produce_different_output():
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=5_000)
    out_a = compute_theta_sensitivity(delta_norms, n_bootstraps=20, seed=1)
    out_b = compute_theta_sensitivity(delta_norms, n_bootstraps=20, seed=2)
    # At least one stat field should differ across the percentile band.
    differs = False
    for p in range(90, 100):
        if out_a[f"p{p}"]["theta_mean"] != out_b[f"p{p}"]["theta_mean"]:
            differs = True
            break
    assert differs, "different seeds produced identical output (broken RNG)"


# ---------------------------------------------------------------------------
# Cheap path (n_bootstraps=0, default) — wired into the build path
# ---------------------------------------------------------------------------


def test_default_uses_cheap_path_zero_std():
    """Default `n_bootstraps=0` returns zero std fields (no bootstrap)."""
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=10_000)
    out = compute_theta_sensitivity(delta_norms)
    for p in range(90, 100):
        assert out[f"p{p}"]["theta_std"] == 0.0
        assert out[f"p{p}"]["anomaly_count_std"] == 0.0


def test_cheap_path_theta_matches_numpy_percentile():
    """Cheap-path `theta_mean` reproduces `np.percentile` exactly."""
    rng = np.random.default_rng(42)
    delta_norms = rng.normal(loc=5.0, scale=2.0, size=20_000)
    out = compute_theta_sensitivity(delta_norms)
    for p in (90, 91, 95, 99):
        np_pct = float(np.percentile(delta_norms, p))
        assert out[f"p{p}"]["theta_mean"] == pytest.approx(np_pct, rel=1e-6)


def test_cheap_path_anomaly_count_matches_explicit_sum():
    """Cheap-path `anomaly_count_mean` equals `sum(delta_norms >= theta)`."""
    rng = np.random.default_rng(7)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=5_000)
    out = compute_theta_sensitivity(delta_norms)
    for p in range(90, 100):
        theta = out[f"p{p}"]["theta_mean"]
        explicit = int(np.sum(delta_norms >= theta))
        assert int(out[f"p{p}"]["anomaly_count_mean"]) == explicit


def test_from_sorted_matches_compute_with_zero_bootstraps():
    """`compute_theta_sensitivity_from_sorted(np.sort(x))` equivalent to
    `compute_theta_sensitivity(x, n_bootstraps=0)` — proves the cheap
    path delegates correctly."""
    rng = np.random.default_rng(13)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=10_000)
    sorted_norms = np.sort(delta_norms)
    direct = compute_theta_sensitivity_from_sorted(sorted_norms)
    delegated = compute_theta_sensitivity(delta_norms, n_bootstraps=0)
    for p in range(90, 100):
        for field in (
            "theta_mean", "theta_std",
            "anomaly_count_mean", "anomaly_count_std",
            "anomaly_rate",
        ):
            assert direct[f"p{p}"][field] == delegated[f"p{p}"][field], (
                f"divergence at p{p}.{field}"
            )


def test_from_sorted_empty_array_returns_zeros():
    """Defensive: empty input yields all-zero stats for every percentile."""
    out = compute_theta_sensitivity_from_sorted(np.array([], dtype=np.float64))
    for p in range(90, 100):
        for field in (
            "theta_mean", "theta_std",
            "anomaly_count_mean", "anomaly_count_std",
            "anomaly_rate",
        ):
            assert out[f"p{p}"][field] == 0.0


def test_cheap_path_anomaly_rate_complement():
    """For a smooth distribution, anomaly_rate at percentile p should
    be close to `(100 - p) / 100` — verifies the rate field reflects
    the percentile semantics on the cheap path (bootstrap noise excluded
    here because the cheap path is deterministic)."""
    rng = np.random.default_rng(0)
    delta_norms = rng.normal(loc=0.0, scale=1.0, size=20_000)
    out = compute_theta_sensitivity(delta_norms)
    for p in (90, 95, 99):
        expected = (100 - p) / 100.0
        assert abs(out[f"p{p}"]["anomaly_rate"] - expected) < 0.005, (
            f"p{p} anomaly_rate {out[f'p{p}']['anomaly_rate']} far from "
            f"expected {expected}"
        )


# ---------------------------------------------------------------------------
# derive_stable_band_and_cliffs
# ---------------------------------------------------------------------------


def _build_ts_from_thetas(thetas: list[float]) -> dict[str, dict[str, float]]:
    """Helper: build a `theta_sensitivity` dict from a theta_mean sequence
    so we can drive the derivation with crafted theta ratios. Counts
    are filled with mechanical percentile values and are NOT used by
    the derivation (theta-based)."""
    return {
        f"p{90 + i}": {
            "theta_mean": float(thetas[i]),
            "theta_std": 0.0,
            "anomaly_count_mean": float(1000 - 100 * i),
            "anomaly_count_std": 0.0,
            "anomaly_rate": float(1000 - 100 * i) / 1000.0,
        }
        for i in range(len(thetas))
    }


def test_derive_empty_input_returns_zero_band():
    out = derive_stable_band_and_cliffs({})
    assert out["stable_band"] == {"from": None, "to": None, "length": 0}
    assert out["cliffs"] == []
    assert out["n_cliffs"] == 0
    assert out["stable_band_length"] == 0


def test_derive_smooth_distribution_full_band_no_cliffs():
    """Light-tail (e.g. Gaussian) theta progression — small geometric
    increase per percentile. All adjacent ratios < 1.30 → stable band
    covers all 10 percentiles, no cliffs."""
    # Geometric increase ratio 1.05 per step (~5 % theta jump per percentile)
    thetas = [1.0 * (1.05 ** i) for i in range(10)]
    out = derive_stable_band_and_cliffs(_build_ts_from_thetas(thetas))
    assert out["stable_band"] == {"from": "p90", "to": "p99", "length": 10}
    assert out["cliffs"] == []
    assert out["n_cliffs"] == 0


def test_derive_heavy_tail_pattern_band_seven_two_cliffs():
    """Heavy-tail distribution with smooth p90-p96 progression and
    sharp upper-tail jumps at p97-p98 + p98-p99. Reproduce a
    canonical heavy-tail signature."""
    # Hand-crafted theta ratios:
    # p90..p96: 1.05 per step (smooth)
    # p96→p97: 1.10 (still smooth)
    # p97→p98: 1.50 EXACTLY (cliff)
    # p98→p99: 2.00 (cliff)
    thetas = [1.0]
    for _ in range(6):  # p90..p96, 6 transitions × 1.05
        thetas.append(thetas[-1] * 1.05)
    thetas.append(thetas[-1] * 1.10)  # p96→p97 smooth
    thetas.append(thetas[-1] * 1.50)  # p97→p98 cliff
    thetas.append(thetas[-1] * 2.00)  # p98→p99 cliff
    out = derive_stable_band_and_cliffs(_build_ts_from_thetas(thetas))
    # Smooth run is i=0..6 (7 smooth transitions including p96→p97 at 1.10),
    # so stable band covers p90..p97 (8 percentiles)
    assert out["stable_band"]["from"] == "p90"
    assert out["stable_band"]["to"] == "p97"
    assert out["stable_band"]["length"] == 8
    assert out["n_cliffs"] == 2
    assert out["cliffs"][0]["from"] == "p97"
    assert out["cliffs"][0]["to"] == "p98"
    assert out["cliffs"][0]["ratio"] == pytest.approx(1.50, rel=1e-3)
    assert out["cliffs"][1]["from"] == "p98"
    assert out["cliffs"][1]["to"] == "p99"
    assert out["cliffs"][1]["ratio"] == pytest.approx(2.00, rel=1e-3)


def test_derive_all_cliffs_returns_zero_band():
    """Every adjacent theta ratio >= 1.50 → no smooth transition → stable
    band length 0 (rather than the degenerate 'one percentile alone')."""
    thetas = [1.0 * (2.0 ** i) for i in range(10)]  # ratio 2.0 everywhere
    out = derive_stable_band_and_cliffs(_build_ts_from_thetas(thetas))
    assert out["stable_band_length"] == 0
    assert out["stable_band"]["from"] is None
    assert out["stable_band"]["to"] is None
    assert out["n_cliffs"] == 9


def test_derive_cliff_in_middle_picks_longest_run():
    """When two stable runs are separated by a cliff in the middle, the
    derivation picks the longer one. Tests run-tracking correctness AND
    the exact `>=` boundary semantics: ratio == 1.50 IS a cliff (not
    a smooth pair), so a code regression that flipped to `>` would
    leak into a longer false 'stable' run.
    """
    # Hand-crafted theta ratios:
    # p90→p93: 3 smooth transitions @ 1.10 (run = p90..p93, 4 percentiles)
    # p93→p94: ratio EXACTLY 1.5000 (cliff terminates first run)
    # p94→p99: 5 smooth transitions @ 1.10 (run = p94..p99, 6 percentiles)
    thetas = [1.0, 1.10, 1.21, 1.331, 1.9965, 2.196, 2.416, 2.658, 2.924, 3.216]
    out = derive_stable_band_and_cliffs(_build_ts_from_thetas(thetas))
    # Exact band — the post-cliff run is longer (6 vs 4 percentiles)
    assert out["stable_band"]["from"] == "p94"
    assert out["stable_band"]["to"] == "p99"
    assert out["stable_band"]["length"] == 6
    # Exactly one cliff at the boundary, ratio exactly 1.5000
    assert out["n_cliffs"] == 1
    assert out["cliffs"][0]["from"] == "p93"
    assert out["cliffs"][0]["to"] == "p94"
    assert out["cliffs"][0]["ratio"] == pytest.approx(1.5000, rel=1e-3)


def test_derive_zero_theta_division_safe():
    """Adjacent-pair where the lower-percentile theta is 0 → ratio inf,
    treated as cliff, no division crash. Defensive — populated theta
    is positive in practice but the helper must not crash on zero."""
    thetas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    out = derive_stable_band_and_cliffs(_build_ts_from_thetas(thetas))
    # The first transition has lo_theta=0 → inf ratio, treated as cliff
    assert any(c["ratio"] == float("inf") for c in out["cliffs"])


def test_derive_count_ratios_alone_do_not_create_cliffs():
    """REGRESSION GUARD: a perfectly smooth theta progression must
    produce zero cliffs even though the underlying anomaly_count
    ratios mechanically reach 1.5 at p97→p98 and 2.0 at p98→p99.
    This is the bug the redesign fixed — the prior count-based
    derivation flagged every distribution with the same false
    cliffs at p97-p98 + p98-p99."""
    # Theta increases by exactly 5 % per percentile end-to-end
    thetas = [1.0 * (1.05 ** i) for i in range(10)]
    ts = _build_ts_from_thetas(thetas)
    # Verify the underlying count ratios DO cross thresholds (would
    # trigger cliffs under the old design)
    p97_count = ts["p97"]["anomaly_count_mean"]
    p98_count = ts["p98"]["anomaly_count_mean"]
    p99_count = ts["p99"]["anomaly_count_mean"]
    assert p97_count / p98_count == pytest.approx(1.5, rel=1e-9)
    assert p98_count / p99_count == pytest.approx(2.0, rel=1e-9)
    # But the new theta-based derivation correctly returns no cliffs
    out = derive_stable_band_and_cliffs(ts)
    assert out["n_cliffs"] == 0
    assert out["stable_band_length"] == 10


def test_derive_thresholds_are_module_constants():
    """The 1.30 / 1.50 boundaries are exposed as module constants for
    cross-package consistency (so MCP tool docstring + skill copy
    stay in sync with implementation)."""
    assert STABLE_BAND_RATIO_THRESHOLD == 1.30
    assert CLIFF_RATIO_THRESHOLD == 1.50
