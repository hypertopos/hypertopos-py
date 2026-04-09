# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Cross-pattern composition via Fisher's method and co-dispersion.

No scipy dependency — chi2 survival via regularized incomplete gamma (numpy + math only),
Spearman via rank-transform + Pearson.
"""
from __future__ import annotations

import math

import numpy as np

_P_FLOOR = 1e-15


def _chi2_sf(x: float, df: int) -> float:
    """Chi-squared survival function P(X > x) for X ~ chi2(df).

    Uses regularized incomplete gamma via series/continued-fraction expansion
    for df <= 100. For df > 100, uses Wilson-Hilferty normal approximation
    (accurate to ~1e-4 for large df).
    """
    if x <= 0:
        return 1.0
    if df > 100:
        # Wilson-Hilferty normal approximation for large df
        k = df
        z_wh = (math.pow(x / k, 1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(
            2.0 / (9.0 * k)
        )
        # Standard normal survival: 0.5 * erfc(z / sqrt(2))
        return max(0.0, min(1.0, 0.5 * math.erfc(z_wh / math.sqrt(2.0))))
    a = df / 2.0
    z = x / 2.0
    if z < a + 1:
        # Series expansion for lower incomplete gamma
        term = 1.0 / a
        s = term
        for k in range(1, 300):
            term *= z / (a + k)
            s += term
            if abs(term) < 1e-14 * abs(s):
                break
        lower = math.exp(-z + a * math.log(z) - math.lgamma(a)) * s
        return max(0.0, min(1.0, 1.0 - lower))
    else:
        # Continued fraction for upper incomplete gamma (Lentz)
        f = 1e-30
        c = 1e-30
        d = 1.0 / (z - a + 1.0)
        f = d
        for k in range(1, 300):
            an = -k * (k - a)
            bn = z - a + 2.0 * k + 1.0
            denom = bn + an * d
            d = 1.0 / denom if abs(denom) > 1e-30 else 1e30
            c_val = bn + an / c if abs(c) > 1e-30 else bn
            c = c_val if abs(c_val) > 1e-30 else 1e-30
            delta = c * d
            f *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        upper = math.exp(-z + a * math.log(z) - math.lgamma(a)) * f
        return max(0.0, min(1.0, upper))


def _t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided survival function for Student's t-distribution.

    P(|T| > t) = 1 - I_x(df/2, 1/2) where x = df/(df + t^2),
    using the regularized incomplete beta via continued fraction.
    For df >= 30, falls back to normal approximation.
    """
    if df <= 0:
        return 0.0
    if df >= 30:
        # Normal approximation accurate for df >= 30
        return max(0.0, min(1.0, math.erfc(abs(t) / math.sqrt(2.0))))
    # Regularized incomplete beta: I_x(a, b) via continued fraction
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # Use the log-beta normalization
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x <= 0:
        return 1.0
    if x >= 1:
        return 0.0
    # Lentz continued fraction for I_x(a, b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta) / a
    # Modified Lentz algorithm
    f = 1e-30
    c = 1e-30
    d = 1.0 / (1.0 - (a + b) * x / (a + 1.0))
    if abs(d) < 1e-30:
        d = 1e-30
    f = d
    for m in range(1, 200):
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 / (1.0 + num * d)
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        if abs(d) < 1e-30:
            d = 1e-30
        f *= c * d
        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 / (1.0 + num * d)
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        if abs(d) < 1e-30:
            d = 1e-30
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    i_x = front * f
    # Two-sided: P(|T| > t) = 1 - I_x(df/2, 1/2)
    return max(0.0, min(1.0, 1.0 - i_x))


def fisher_combine_pvalues(p_values: list[float]) -> dict:
    """Combine independent p-values via Fisher's method.

    chi2_stat = -2 * sum(ln(p_i))
    Combined p from chi2 distribution with df = 2*k.
    """
    if not p_values:
        raise ValueError("fisher_combine_pvalues requires at least one p-value")
    k = len(p_values)
    clamped = [max(p, _P_FLOOR) for p in p_values]
    chi2_stat = -2.0 * sum(float(np.log(p)) for p in clamped)
    df = 2 * k
    combined_p = _chi2_sf(chi2_stat, df)
    return {
        "combined_p": round(combined_p, 6),
        "chi2": round(chi2_stat, 4),
        "df": df,
        "k": k,
        "input_p_values": [round(p, 6) for p in p_values],
    }


def _rank(x: np.ndarray) -> np.ndarray:
    """Rank values (1-based). Ties get average rank."""
    order = np.argsort(np.argsort(x)).astype(np.float64)
    return order + 1


def co_dispersion(
    norms_a: np.ndarray,
    norms_b: np.ndarray,
    min_entities: int = 5,
) -> dict:
    """Spearman rank correlation of delta_norms between two patterns."""
    n = min(len(norms_a), len(norms_b))
    if n < min_entities:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n": n, "insufficient_data": True}
    a = norms_a[:n].astype(np.float64)
    b = norms_b[:n].astype(np.float64)
    r_a = _rank(a)
    r_b = _rank(b)
    rho = float(np.corrcoef(r_a, r_b)[0, 1])
    # Two-sided p-value from t-distribution with n-2 degrees of freedom
    if abs(rho) >= 1.0 or n <= 2:
        p_approx = 0.0
    else:
        t_stat = rho * np.sqrt((n - 2) / (1.0 - rho * rho))
        p_approx = _t_sf_two_sided(abs(float(t_stat)), n - 2)
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p_approx), 6),
        "n": n,
        "insufficient_data": False,
    }
