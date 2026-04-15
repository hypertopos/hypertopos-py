# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Bregman divergence module — distribution-aware per-dimension anomaly scoring.

Each dimension is assigned a *kind* that determines the Bregman generating
function used to measure how far an observation deviates from its population
centre.  The three supported kinds cover the main distribution families found
in business data:

    gaussian  — squared z-score; appropriate for real-valued, approximately
                normal dimensions (amounts, ratios without hard boundaries).
    poisson   — KL divergence for non-negative count data (transactions,
                events, calls).
    bernoulli — KL divergence for binary or proportion data in (0, 1).

All functions operate on float64 arrays and return float64.  The module has
no dependencies beyond NumPy and is intended to be a pure-computation leaf
that higher-level builder stages call.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

BREGMAN_KINDS: tuple[str, ...] = ("gaussian", "poisson", "bernoulli")

_EPS: float = 1e-7  # clamping epsilon for bernoulli boundaries


# ---------------------------------------------------------------------------
# Single-entity divergence
# ---------------------------------------------------------------------------

def bregman_divergence(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    kinds: list[str],
) -> np.ndarray:
    """Per-dimension Bregman divergence for a single entity.

    Args:
        x: (D,) observed feature vector.
        mu: (D,) population mean vector.
        sigma: (D,) population std-dev vector (used only for gaussian kind).
        kinds: list of D strings, each one of BREGMAN_KINDS.

    Returns:
        (D,) float64 — non-negative per-dimension divergence values.

    Formulas:
        gaussian  : (x - mu)^2 / (2 * sigma^2)
        poisson   : x * log(x / mu) - (x - mu)  [when x=0: result = mu]
        bernoulli : x * log(x / mu) + (1-x) * log((1-x) / (1-mu))
                    [x and mu clamped to [eps, 1-eps]]
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    D = len(kinds)
    result = np.empty(D, dtype=np.float64)

    for d, kind in enumerate(kinds):
        result[d] = _divergence_scalar(float(x[d]), float(mu[d]), float(sigma[d]), kind)

    return result


def _divergence_scalar(x: float, mu: float, sigma: float, kind: str) -> float:
    """Compute Bregman divergence for a single scalar observation."""
    if kind == "gaussian":
        diff = x - mu
        s = max(sigma, _EPS)
        return (diff * diff) / (2.0 * s * s)

    if kind == "poisson":
        mu = max(mu, _EPS)  # clamp to avoid log(0)
        if x <= 0.0:
            return mu
        return x * math.log(x / mu) - (x - mu)

    if kind == "bernoulli":
        x_c = max(_EPS, min(1.0 - _EPS, x))
        mu_c = max(_EPS, min(1.0 - _EPS, mu))
        return x_c * math.log(x_c / mu_c) + (1.0 - x_c) * math.log(
            (1.0 - x_c) / (1.0 - mu_c)
        )

    raise ValueError(f"Unknown Bregman kind: {kind!r}. Expected one of {BREGMAN_KINDS}.")


# ---------------------------------------------------------------------------
# Batch divergence (vectorized, no Python loops over entities)
# ---------------------------------------------------------------------------

def bregman_divergence_batch(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    kinds: list[str],
) -> np.ndarray:
    """Vectorized per-dimension Bregman divergence for a population matrix.

    Args:
        X: (N, D) observation matrix.
        mu: (D,) population mean vector.
        sigma: (D,) population std-dev vector (used only for gaussian kind).
        kinds: list of D strings, each one of BREGMAN_KINDS.

    Returns:
        (N, D) float64 — non-negative per-dimension divergence values.

    Implementation note: all arithmetic is fully vectorized over the N axis.
    No Python loops over entities.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    N, D = X.shape
    result = np.empty((N, D), dtype=np.float64)

    # Process dimensions grouped by kind to keep the hot path branchless.
    # Grouping by kind allows fully vectorized column operations.
    gaussian_cols = [d for d, k in enumerate(kinds) if k == "gaussian"]
    poisson_cols = [d for d, k in enumerate(kinds) if k == "poisson"]
    bernoulli_cols = [d for d, k in enumerate(kinds) if k == "bernoulli"]
    unknown = [k for k in kinds if k not in BREGMAN_KINDS]
    if unknown:
        raise ValueError(
            f"Unknown Bregman kinds: {unknown!r}. Expected subsets of {BREGMAN_KINDS}."
        )

    # --- gaussian -----------------------------------------------------------
    if gaussian_cols:
        gc = np.array(gaussian_cols)
        diff = X[:, gc] - mu[gc]          # (N, |gc|)
        s = np.maximum(sigma[gc], _EPS)   # guard against sigma=0
        s2 = s ** 2                        # (|gc|,)
        result[:, gc] = (diff ** 2) / (2.0 * s2)

    # --- poisson ------------------------------------------------------------
    if poisson_cols:
        pc = np.array(poisson_cols)
        xp = X[:, pc]                      # (N, |pc|)
        mup = np.maximum(mu[pc], _EPS)     # (|pc|,) clamp to avoid log(0)
        # x=0 edge-case: result = mu
        zero_mask = xp <= 0.0
        # Safe log: replace 0 with 1 temporarily (result overwritten below)
        xp_safe = np.where(zero_mask, 1.0, xp)
        d_pois = xp * np.log(xp_safe / mup) - (xp - mup)
        d_pois = np.where(zero_mask, mup, d_pois)
        result[:, pc] = d_pois

    # --- bernoulli ----------------------------------------------------------
    if bernoulli_cols:
        bc = np.array(bernoulli_cols)
        xb = np.clip(X[:, bc], _EPS, 1.0 - _EPS)
        mub = np.clip(mu[bc], _EPS, 1.0 - _EPS)
        result[:, bc] = (
            xb * np.log(xb / mub)
            + (1.0 - xb) * np.log((1.0 - xb) / (1.0 - mub))
        )

    return result


# ---------------------------------------------------------------------------
# Bregman norms — total divergence per entity
# ---------------------------------------------------------------------------

def bregman_norms(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    kinds: list[str],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Total Bregman divergence per entity (sum of per-dimension values).

    Args:
        X: (N, D) observation matrix.
        mu: (D,) population mean vector.
        sigma: (D,) population std-dev vector.
        kinds: list of D kind strings.
        weights: optional (D,) weight vector applied before summing.

    Returns:
        (N,) float64 — total divergence per entity.
    """
    per_dim = bregman_divergence_batch(X, mu, sigma, kinds)  # (N, D)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        per_dim = per_dim * weights  # broadcast over N
    return per_dim.sum(axis=1)


# ---------------------------------------------------------------------------
# Kind auto-detection
# ---------------------------------------------------------------------------

def detect_kind_for_column(values: np.ndarray) -> str:
    """Auto-detect the Bregman kind for a column of observed values.

    Rules (applied in order):
        1. If all finite values are in {0.0, 1.0}     → "bernoulli"
        2. If all finite values are non-negative AND
           all are integers (x == floor(x))            → "poisson"
        3. Otherwise                                   → "gaussian"
        4. If no finite values exist                   → "gaussian"

    Args:
        values: 1-D array of observed values (may contain NaN/Inf).

    Returns:
        One of "bernoulli", "poisson", "gaussian".
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "gaussian"

    # Rule 1 — bernoulli: all values strictly in {0, 1}
    if np.all((finite == 0.0) | (finite == 1.0)):
        return "bernoulli"

    # Rule 2 — poisson: all non-negative integers
    if np.all(finite >= 0.0) and np.all(finite == np.floor(finite)):
        return "poisson"

    return "gaussian"


def detect_kinds_for_pattern(
    relation_edge_maxes: list,
    derived_metrics: list[str],
    precomputed_edge_maxes: list,
    prop_count: int,
    dim_block_count: int,
    event_column_values: list[np.ndarray] | None = None,
    event_kind_overrides: list[str | None] | None = None,
    derived_kind_overrides: list[str | None] | None = None,
    precomputed_kind_overrides: list[str | None] | None = None,
) -> list[str]:
    """Build ordered list of dimension kinds matching the delta vector layout.

    Dimension order:
        1. Relations       — one per entry in relation_edge_maxes
        2. Event dims      — one per entry in event_column_values (if provided)
        3. Derived dims    — one per entry in derived_metrics
        4. Precomputed dims— one per entry in precomputed_edge_maxes
        5. Prop fill       — prop_count entries, always "bernoulli"
        6. Dim blocks      — dim_block_count entries, always "gaussian"

    Args:
        relation_edge_maxes: list of edge_max values per relation; None means
            binary FK (→ "bernoulli"), not None means count/max (→ "poisson").
        derived_metrics: list of metric strings like "count", "sum:amount",
            "count:window=1d:agg=max".  Base metric is the part before the
            first ":".  "count" and "count_distinct" → "poisson", else "gaussian".
        precomputed_edge_maxes: list of edge_max integers for precomputed dims;
            edge_max == 1 → "bernoulli", else "gaussian".
        prop_count: number of property-fill dimensions (always "bernoulli").
        dim_block_count: number of dim-block dimensions (always "gaussian").
        event_column_values: optional list of 1-D arrays, one per event dim.
            If None, event dims are not added.
        event_kind_overrides: optional list of kind strings (or None) for each
            event dim.  None at position i → auto-detect via
            detect_kind_for_column.
        derived_kind_overrides: optional list of kind strings (or None) for
            each derived dim.  None at position i → auto-detect.
        precomputed_kind_overrides: optional list of kind strings (or None) for
            each precomputed dim.  None at position i → auto-detect.

    Returns:
        Ordered list of kind strings matching the delta vector layout.
    """
    kinds: list[str] = []

    # 1. Relations
    for edge_max in relation_edge_maxes:
        kinds.append("bernoulli" if edge_max is None else "poisson")

    # 2. Event dims
    if event_column_values is not None:
        for i, col_values in enumerate(event_column_values):
            if event_kind_overrides is not None and i < len(event_kind_overrides):
                override = event_kind_overrides[i]
            else:
                override = None
            if override is not None:
                kinds.append(override)
            else:
                kinds.append(detect_kind_for_column(col_values))

    # 3. Derived dims
    for i, metric in enumerate(derived_metrics):
        if derived_kind_overrides is not None and i < len(derived_kind_overrides):
            override = derived_kind_overrides[i]
        else:
            override = None
        if override is not None:
            kinds.append(override)
        else:
            base_metric = metric.split(":")[0]
            if base_metric in ("count", "count_distinct"):
                kinds.append("poisson")
            else:
                kinds.append("gaussian")

    # 4. Precomputed dims
    for i, edge_max in enumerate(precomputed_edge_maxes):
        if precomputed_kind_overrides is not None and i < len(precomputed_kind_overrides):
            override = precomputed_kind_overrides[i]
        else:
            override = None
        if override is not None:
            kinds.append(override)
        else:
            kinds.append("bernoulli" if edge_max is not None and int(edge_max) == 1 else "gaussian")

    # 5. Prop fill — always bernoulli
    kinds.extend(["bernoulli"] * prop_count)

    # 6. Dim blocks — always gaussian
    kinds.extend(["gaussian"] * dim_block_count)

    return kinds


def format_kinds_summary(kinds: list[str]) -> str:
    """Format a kinds list as a human-readable summary.

    Example:
        ["bernoulli", "bernoulli", "poisson", "gaussian", "gaussian"]
        → "bernoulli x2, poisson x1, gaussian x2"

    Order in the output matches the first occurrence of each kind in the list.

    Args:
        kinds: list of kind strings.

    Returns:
        Compact summary string, e.g. "bernoulli x4, poisson x2, gaussian x8".
        Returns "" for an empty list.
    """
    if not kinds:
        return ""

    # Preserve first-seen order
    seen: dict[str, int] = {}
    for k in kinds:
        seen[k] = seen.get(k, 0) + 1

    parts = [f"{kind} x{count}" for kind, count in seen.items()]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Per-dimension anomaly threshold
# ---------------------------------------------------------------------------

def per_dim_theta(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    kinds: list[str],
    anomaly_percentile: float = 95.0,
) -> np.ndarray:
    """Per-dimension anomaly threshold via empirical percentile.

    For each dimension d, the threshold is the empirical percentile of the
    per-dimension Bregman divergence values.

    Args:
        X: (N, D) observation matrix.
        mu: (D,) population mean vector.
        sigma: (D,) population std-dev vector.
        kinds: list of D kind strings.
        anomaly_percentile: empirical percentile to use (default 95.0).

    Returns:
        (D,) float64 — per-dimension anomaly thresholds.
    """
    per_dim = bregman_divergence_batch(X, mu, sigma, kinds)  # (N, D)
    D = per_dim.shape[1]

    empirical = np.percentile(per_dim, anomaly_percentile, axis=0)  # (D,)

    # Chernoff floor applied to TOTAL theta only (not per-dim).
    # Per-dim thresholds use empirical percentiles — the Chernoff bound
    # -log(alpha/D) is a concentration inequality on the total divergence,
    # not meaningful as a per-dimension floor.
    return empirical.astype(np.float64)
