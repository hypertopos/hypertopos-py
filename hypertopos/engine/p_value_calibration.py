# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Per-detector p-value calibration adapters for multi-detector consensus.

Each detector emits a heterogeneous score (delta_norm, neighbor anomaly count,
segment shift, density gap q-value, trajectory DTW). To combine them via the
harmonic-mean p-value (`engine/composition.harmonic_mean_p`), each score is
calibrated to a p-value in (0, 1] under that detector's null hypothesis.

Adapter contract:
    Input shape: detector-specific (Arrow table / dict / list).
    Output: dict[primary_key] -> p in (0, 1], deterministic, no NaN/inf.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pyarrow as pa
from scipy.special import gammaincc
from scipy.stats import fisher_exact, hypergeom

# Lower clamp shared with engine/composition.py to keep HMP denominators finite.
_P_FLOOR = 1e-300
# Upper-bound p clamp avoids returning exactly 0 (which would dominate HMP).
_P_CEIL = 1.0


def _clip_p(value: float) -> float:
    """Clamp a single float into (_P_FLOOR, 1.0], replacing NaN/inf with 1.0."""
    if not np.isfinite(value):
        return _P_CEIL
    if value <= 0.0:
        return _P_FLOOR
    if value > _P_CEIL:
        return _P_CEIL
    return float(value)


def detector_p_value_delta_norm(
    geo_table: pa.Table,
    primary_keys: list[str] | tuple[str, ...],
    *,
    use_anomaly_confidence: bool = True,
    df: int | None = None,
) -> dict[str, float]:
    """Calibrate delta_norm-based anomaly score to per-entity p-values.

    Primary path: ``p = 1 - anomaly_confidence`` (sphere format >= 2.4 stores
    `anomaly_confidence` in geometry; null per-row entries fall back to chi2).
    Fallback path: ``p = 1 - F_chi2(delta_norm**2; df)``, where df defaults to
    the dimensionality of the polygon delta vector.

    Args:
        geo_table: Arrow table containing at minimum `primary_key` and
            `delta_norm` columns; optionally `anomaly_confidence`.
        primary_keys: Subset to evaluate; ordering preserved in the output dict.
        use_anomaly_confidence: When False, force the chi2 fallback.
        df: Degrees of freedom for the chi2 fallback. Required when the
            anomaly_confidence column is absent OR a row's value is null.

    Returns:
        ``{primary_key: p}`` for entities present in ``geo_table``.
    """
    if not primary_keys:
        return {}
    if geo_table.num_rows == 0:
        return {}

    pk_array = geo_table["primary_key"].to_pylist()
    pk_to_idx: dict[str, int] = {}
    for i, pk in enumerate(pk_array):
        # First occurrence wins — guards against duplicate rows in the input table.
        pk_to_idx.setdefault(pk, i)

    delta_norm = geo_table["delta_norm"]
    has_conf = (
        use_anomaly_confidence and "anomaly_confidence" in geo_table.schema.names
    )
    conf_col = geo_table["anomaly_confidence"] if has_conf else None

    out: dict[str, float] = {}
    for pk in primary_keys:
        idx = pk_to_idx.get(pk)
        if idx is None:
            continue
        conf_val: float | None = None
        if has_conf:
            scalar = conf_col[idx]
            if scalar.is_valid:
                conf_val = float(scalar.as_py())
        if conf_val is not None:
            out[pk] = _clip_p(1.0 - conf_val)
            continue
        # Fallback: chi2 survival on delta_norm**2
        d_scalar = delta_norm[idx]
        if not d_scalar.is_valid:
            out[pk] = _P_CEIL
            continue
        dn = float(d_scalar.as_py())
        if df is None or df <= 0:
            # Without df, no defensible parametric mapping — return a uniform p.
            out[pk] = _P_CEIL
            continue
        p = float(gammaincc(df / 2.0, (dn * dn) / 2.0))
        out[pk] = _clip_p(p)
    return out


def detector_p_value_neighbor_contamination(
    observations: Mapping[str, tuple[int, int]],
    *,
    total_population: int,
    total_anomalies: int,
    k: int,
) -> dict[str, float]:
    """Hypergeometric upper-tail p-value for "neighbor anomaly contamination".

    For each entity, ``observations[entity] = (k_observed, x_observed)`` where
    ``k_observed`` is the number of inspected neighbors and ``x_observed`` is
    how many of them are anomalous. Under H0 (uniform draw from a population
    of size ``total_population`` containing ``total_anomalies`` anomalies),
    ``X ~ Hypergeom(M=total_population, n=total_anomalies, N=k_observed)``.

    Returns ``P(X >= x_observed)``.
    """
    if not observations:
        return {}
    M = max(int(total_population), 1)
    n_anom = max(min(int(total_anomalies), M), 0)
    out: dict[str, float] = {}
    for entity, (k_obs, x_obs) in observations.items():
        k_eff = max(int(k_obs), 0)
        x_eff = max(int(x_obs), 0)
        if k_eff == 0 or n_anom == 0 or x_eff == 0:
            out[entity] = _P_CEIL
            continue
        # hypergeom.sf(x-1, M, n, N) = P(X >= x)
        p = float(hypergeom.sf(x_eff - 1, M, n_anom, k_eff))
        out[entity] = _clip_p(p)
    # Fall through for entities with k_obs=0 etc handled in loop.
    # Ensure we did not silently drop a key via continue -> already handled.
    if k is not None:
        # k is part of the public signature for self-documentation;
        # actual per-row k_obs comes from observations.
        _ = k
    return out


def detector_p_value_segment_shift(
    observations: Mapping[str, Mapping[str, int]],
) -> dict[str, float]:
    """Fisher's exact 2x2 p-value for "anomaly rate differs by segment".

    For each segment id S, ``observations[S]`` is a mapping with keys:
        - in_segment_anomalous, in_segment_total
        - out_segment_anomalous, out_segment_total

    Returns the two-sided Fisher exact p-value comparing in-segment anomaly
    rate against out-segment anomaly rate.
    """
    if not observations:
        return {}
    out: dict[str, float] = {}
    for segment, cells in observations.items():
        in_a = max(int(cells.get("in_segment_anomalous", 0)), 0)
        in_t = max(int(cells.get("in_segment_total", 0)), 0)
        out_a = max(int(cells.get("out_segment_anomalous", 0)), 0)
        out_t = max(int(cells.get("out_segment_total", 0)), 0)
        in_n = max(in_t - in_a, 0)
        out_n = max(out_t - out_a, 0)
        if (in_a + out_a) == 0 or (in_t == 0 and out_t == 0):
            # No anomalies at all — no signal.
            out[segment] = _P_CEIL
            continue
        # 2x2: rows = segment / not segment, cols = anomaly / not
        table = [[in_a, in_n], [out_a, out_n]]
        try:
            _odds, p_value = fisher_exact(table, alternative="two-sided")
        except (ValueError, ZeroDivisionError):
            p_value = _P_CEIL
        out[segment] = _clip_p(float(p_value))
    return out


def detector_p_value_density_gap(
    density_gap_results: list[Mapping[str, float | str]],
) -> dict[str, float]:
    """Recover raw p-values from density-gap detector output.

    `find_density_gaps` may return either raw `p_value` (preferred) or
    BH-corrected `q_value` together with `rank` and `m`. When `p_value`
    is present we use it directly; otherwise we invert
    ``q = p * m / rank`` to recover ``p = q * rank / m``.
    """
    if not density_gap_results:
        return {}
    out: dict[str, float] = {}
    for entry in density_gap_results:
        pk = entry.get("primary_key")
        if pk is None:
            continue
        if "p_value" in entry and entry["p_value"] is not None:
            p = float(entry["p_value"])
            out[str(pk)] = _clip_p(p)
            continue
        q = entry.get("q_value")
        rank = entry.get("rank")
        m = entry.get("m")
        if q is None or rank is None or m is None:
            out[str(pk)] = _P_CEIL
            continue
        try:
            p = float(q) * float(rank) / max(float(m), 1.0)
        except (TypeError, ValueError, ZeroDivisionError):
            p = _P_CEIL
        out[str(pk)] = _clip_p(p)
    return out


def detector_p_value_trajectory_continuous(
    scores: Mapping[str, float],
) -> dict[str, float]:
    """ECDF-based p-value for a continuous trajectory anomaly score.

    Higher score = more anomalous. We map score to ``p = 1 - F_n(score) + 1/(n+1)``
    so the largest score maps to ``1/(n+1)`` (smallest p, never exactly 0)
    and the smallest score maps to 1.0.
    """
    if not scores:
        return {}
    items = list(scores.items())
    keys = [k for k, _ in items]
    values = np.asarray([float(v) for _, v in items], dtype=np.float64)
    # Replace any non-finite scores with the median so they neither dominate
    # nor vanish when ranking.
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        if finite_mask.any():
            fill = float(np.median(values[finite_mask]))
        else:
            fill = 0.0
        values = np.where(finite_mask, values, fill)
    n = values.shape[0]
    # rankdata-equivalent (average ranks for ties)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    rank_so_far = 0.0
    while i < n:
        j = i
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        # Tied group [i, j); average rank is (i+1 + j)/2
        avg_rank = (i + 1 + j) / 2.0
        for kk in range(i, j):
            ranks[order[kk]] = avg_rank
        i = j
        rank_so_far = avg_rank
    _ = rank_so_far
    # p = (n - rank + 1) / (n + 1) — survival of the empirical CDF.
    ps = (n - ranks + 1.0) / (n + 1.0)
    return {keys[i]: _clip_p(float(ps[i])) for i in range(n)}
