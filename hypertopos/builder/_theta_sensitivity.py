"""Theta sensitivity surface — calibration-quality diagnostic.

Per-percentile sweep over a `delta_norm` distribution. Used at build
time by `_BuildState._compute_population_stats` (which already sorts
`delta_norms` for `delta_rank_pcts` + conformal p-values, so the
diagnostic glues onto that existing sort at zero new I/O cost) to
populate the `theta_sensitivity` field on `CalibrationFit` written via
`write_calibration_history_epoch`. Surfaces to investigators via the
`theta_sensitivity(pattern_id)` MCP tool how stable the anomaly
threshold (`theta`) is to perturbations of the `anomaly_percentile`
parameter:

- Stable band: contiguous range of percentiles where adjacent-pair
  `theta_mean` ratios stay below 1.30 — within this band the
  threshold scales smoothly with percentile choice
- Cliff zone: adjacent-pair `theta_mean` ratios >= 1.50 —
  recalibration across this boundary jumps the threshold by 50 % or
  more, signalling a heavy-tail region of the distribution

Theta ratios (not anomaly_count ratios) carry the distribution-specific
signal. Count ratios at percentile boundaries are mechanically
determined by `(100-i) / (100-(i+1))` and identical across all
distributions, so they are NOT used in the cliff/band derivation.

Pure numerics: no Lance, no pyarrow, no builder state. Two surfaces:

- `compute_theta_sensitivity_from_sorted` — O(P) cheap path; default
  for build-time wiring. Reuses an existing `sorted_norms` and never
  re-sorts. No bootstrap, no CI.
- `compute_theta_sensitivity` — opt-in path with `n_bootstraps > 0` for
  on-demand CI estimation (e.g. from the MCP tool). Default
  `n_bootstraps=0` delegates to the cheap path.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PERCENTILES: tuple[int, ...] = (90, 91, 92, 93, 94, 95, 96, 97, 98, 99)
DEFAULT_N_BOOTSTRAPS = 0
DEFAULT_SEED = 0

STABLE_BAND_RATIO_THRESHOLD = 1.30
CLIFF_RATIO_THRESHOLD = 1.50


def derive_stable_band_and_cliffs(
    theta_sensitivity: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Derive `stable_band` + `cliffs` from a `theta_sensitivity` dict.

    Pure derivation — same input always gives the same output, no
    population data needed. Used by the `theta_sensitivity(pattern_id)`
    MCP tool to surface agent-actionable structure on top of the
    populated calibration field.

    The percentile keys are sorted ascending by their integer suffix
    (so `p90 .. p99` regardless of insertion order). The structure
    surfaces **threshold sensitivity** — how much the anomaly threshold
    `theta_mean` jumps when the `anomaly_percentile` parameter is moved
    by one step. Adjacent-pair ratios on `theta_mean` carry the
    distribution-specific signal:

    - `stable_band`: longest contiguous run of percentile pairs
      whose adjacent-pair theta ratio is `< STABLE_BAND_RATIO_THRESHOLD`
      (1.30) — within this band, the threshold scales smoothly and
      recalibration moves shift `theta` by less than 30 %.
    - `cliffs`: every adjacent-pair whose theta ratio is
      `>= CLIFF_RATIO_THRESHOLD` (1.50) — a recalibration move
      across this boundary jumps `theta` by 50 % or more,
      typically because the underlying `delta_norm` distribution
      has a heavy tail in that region.

    Note: ratios are computed on `theta_mean`, NOT on
    `anomaly_count_mean`. Count ratios are mechanically determined by
    percentile arithmetic (`(100-i)/(100-(i+1))`) and carry no
    distribution-specific information; theta ratios reflect the
    actual distributional shape of `delta_norm` and differ across
    patterns.

    Args:
        theta_sensitivity: dict keyed by `"p<percentile>"` (matching
            `compute_theta_sensitivity` output schema).

    Returns:
        Dict with keys:
        - `stable_band`: dict of `{"from": "p<low>", "to": "p<high>",
          "length": int}` (length is the number of percentiles in the
          band, inclusive). When the input dict is empty or contains
          no smooth transitions, `length=0` and `from`/`to` are both
          `None`.
        - `cliffs`: list of `{"from": "p<low>", "to": "p<high>",
          "ratio": float}`, ordered by the percentile boundary;
          `ratio` is the theta_mean ratio.
        - `n_cliffs`: count of cliffs.
        - `stable_band_length`: convenience copy of
          `stable_band["length"]`.
    """
    keys = sorted(theta_sensitivity.keys(), key=lambda k: int(k[1:]))
    if not keys:
        return {
            "stable_band": {"from": None, "to": None, "length": 0},
            "cliffs": [],
            "n_cliffs": 0,
            "stable_band_length": 0,
        }

    cliffs: list[dict[str, object]] = []
    longest_run_lo_idx = -1
    longest_run_hi_idx = -1
    current_run_lo_idx = 0
    for i in range(len(keys) - 1):
        lo_theta = theta_sensitivity[keys[i]]["theta_mean"]
        hi_theta = theta_sensitivity[keys[i + 1]]["theta_mean"]
        ratio = hi_theta / lo_theta if lo_theta > 0.0 else float("inf")
        if ratio >= CLIFF_RATIO_THRESHOLD:
            cliffs.append(
                {
                    "from": keys[i],
                    "to": keys[i + 1],
                    "ratio": ratio,
                },
            )
        if ratio < STABLE_BAND_RATIO_THRESHOLD:
            run_length = (i + 1) - current_run_lo_idx
            best_length = (
                -1
                if longest_run_lo_idx < 0
                else longest_run_hi_idx - longest_run_lo_idx
            )
            if run_length > best_length:
                longest_run_lo_idx = current_run_lo_idx
                longest_run_hi_idx = i + 1
        else:
            current_run_lo_idx = i + 1

    if longest_run_lo_idx < 0:
        return {
            "stable_band": {"from": None, "to": None, "length": 0},
            "cliffs": cliffs,
            "n_cliffs": len(cliffs),
            "stable_band_length": 0,
        }

    band_length = longest_run_hi_idx - longest_run_lo_idx + 1
    return {
        "stable_band": {
            "from": keys[longest_run_lo_idx],
            "to": keys[longest_run_hi_idx],
            "length": band_length,
        },
        "cliffs": cliffs,
        "n_cliffs": len(cliffs),
        "stable_band_length": band_length,
    }


def _zero_stats(percentiles: tuple[int, ...]) -> dict[str, dict[str, float]]:
    return {
        f"p{p}": {
            "theta_mean": 0.0,
            "theta_std": 0.0,
            "anomaly_count_mean": 0.0,
            "anomaly_count_std": 0.0,
            "anomaly_rate": 0.0,
        }
        for p in percentiles
    }


def compute_theta_sensitivity_from_sorted(
    sorted_norms: np.ndarray,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
) -> dict[str, dict[str, float]]:
    """Cheap-path percentile sweep on a pre-sorted delta_norm array.

    O(P) per pattern (P = len(percentiles)). No new sort, no bootstrap.
    Glues onto the builder's existing `sorted_norms` (computed for
    `delta_rank_pcts`), so wiring this in adds zero I/O cost to the
    build path.

    Args:
        sorted_norms: 1-D array of delta_norms sorted ascending.
        percentiles: integer percentile points to sweep (default p90..p99).

    Returns:
        Dict keyed by `"p<percentile>"` with per-percentile stats:
        - `theta_mean`: percentile value (linear interp, matches
          `np.percentile` default)
        - `theta_std`: 0.0 (no bootstrap on the cheap path)
        - `anomaly_count_mean`: `sum(delta_norms >= theta)`
        - `anomaly_count_std`: 0.0
        - `anomaly_rate`: `anomaly_count_mean / n`
    """
    n = len(sorted_norms)
    if n == 0:
        return _zero_stats(percentiles)
    qs = np.asarray(percentiles, dtype=np.float64) / 100.0
    idx = qs * (n - 1)
    lo = np.floor(idx).astype(np.int64)
    hi = np.ceil(idx).astype(np.int64)
    frac = idx - lo
    thetas = sorted_norms[lo] * (1.0 - frac) + sorted_norms[hi] * frac
    count_idx = np.searchsorted(sorted_norms, thetas, side="left")
    counts = (n - count_idx).astype(np.int64)
    result: dict[str, dict[str, float]] = {}
    for p, theta, count in zip(percentiles, thetas, counts, strict=True):
        result[f"p{p}"] = {
            "theta_mean": float(theta),
            "theta_std": 0.0,
            "anomaly_count_mean": float(count),
            "anomaly_count_std": 0.0,
            "anomaly_rate": float(count) / n,
        }
    return result


def compute_theta_sensitivity(
    delta_norms: np.ndarray,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float]]:
    """Percentile sweep over a delta_norm distribution.

    With `n_bootstraps == 0` (default) returns the cheap single-pass
    percentile + searchsorted output, equivalent to
    `compute_theta_sensitivity_from_sorted(np.sort(delta_norms))`.

    With `n_bootstraps > 0`, runs a resampling loop and reports
    bootstrap mean and ddof=1 std for `theta` and `anomaly_count` at
    each percentile. The bootstrap CI is rarely the actionable signal
    on production-scale populations (sub-2 % of the mean on AML
    HI-small) — keep it for on-demand CI estimation, not the build
    path.

    Args:
        delta_norms: 1-D array of population delta_norms.
        percentiles: integer percentile points to sweep (default p90..p99).
        n_bootstraps: bootstrap budget per percentile. 0 (default) → no
            bootstrap, cheap path. > 0 → bootstrap path.
        seed: RNG seed for determinism (bootstrap path only).

    Returns:
        Dict keyed by `"p<percentile>"` with per-percentile stats:
        - `theta_mean` / `theta_std`: cheap-path → percentile value /
          0.0; bootstrap-path → mean / ddof=1 std of theta across
          resamples
        - `anomaly_count_mean` / `anomaly_count_std`: same convention
        - `anomaly_rate`: anomaly_count_mean / len(delta_norms)
    """
    n = len(delta_norms)
    if n == 0:
        return _zero_stats(percentiles)

    if n_bootstraps == 0:
        sorted_norms = np.sort(delta_norms)
        return compute_theta_sensitivity_from_sorted(sorted_norms, percentiles)

    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    for p in percentiles:
        thetas = np.empty(n_bootstraps, dtype=np.float64)
        counts = np.empty(n_bootstraps, dtype=np.int64)
        for b in range(n_bootstraps):
            sample_idx = rng.integers(0, n, size=n)
            boot_norms = delta_norms[sample_idx]
            theta_b = float(np.percentile(boot_norms, p))
            thetas[b] = theta_b
            counts[b] = int(np.sum(delta_norms >= theta_b))
        result[f"p{p}"] = {
            "theta_mean": float(thetas.mean()),
            "theta_std": float(thetas.std(ddof=1) if n_bootstraps > 1 else 0.0),
            "anomaly_count_mean": float(counts.mean()),
            "anomaly_count_std": float(
                counts.std(ddof=1) if n_bootstraps > 1 else 0.0,
            ),
            "anomaly_rate": float(counts.mean() / n),
        }
    return result
