# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SIGMA_EPS: float = 1e-2  # minimum sigma — matches engine/geometry.py SIGMA_EPSILON
SIGMA_EPS_PROP: float = 0.2  # minimum sigma for binary prop_columns — caps deltas at ±5


def compute_stats(
    shape_vectors: np.ndarray,
    anomaly_percentile: float = 95.0,
    use_mahalanobis: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray | None]:
    """
    Auto-compute pattern statistics from a population of shape vectors.

    Returns:
        mu: (D,) float32 — mean shape vector
        sigma: (D,) float32 — per-dim std dev, clamped to >= SIGMA_EPS
        theta: (D,) float32 — z-scored threshold vector; norm(theta) == anomaly boundary
        deltas: (N, D) float32 — z-scored delta vectors
        delta_norms: (N,) float32 — L2 norms of delta vectors
        cholesky_inv: (D, D) float32 | None — inverse Cholesky factor (Mahalanobis only)

    When use_mahalanobis=True, deltas are computed via full covariance matrix
    (Cholesky decomposition) instead of per-dimension sigma. This produces an
    ellipsoidal anomaly boundary that accounts for inter-dimension correlations.
    """
    if len(shape_vectors) == 0:
        raise ValueError("shape_vectors must be non-empty")

    D = shape_vectors.shape[1]
    mu = shape_vectors.mean(axis=0).astype(np.float32)
    sigma_raw = shape_vectors.std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma_raw, SIGMA_EPS)

    cholesky_inv: np.ndarray | None = None

    if use_mahalanobis and D > 1:
        # Full covariance with Tikhonov regularization
        cov = np.cov(shape_vectors.T).astype(np.float64)
        cov += (SIGMA_EPS ** 2) * np.eye(D)
        try:
            L = np.linalg.cholesky(cov)
            centered = (shape_vectors - mu).astype(np.float64)
            deltas = np.linalg.solve(L, centered.T).T.astype(np.float32)
            L_inv = np.linalg.inv(L).astype(np.float32)
            cholesky_inv = L_inv
        except np.linalg.LinAlgError:
            logger.warning(
                "Cholesky decomposition failed — falling back to diagonal sigma"
            )
            deltas = ((shape_vectors - mu) / sigma).astype(np.float32)
    else:
        deltas = ((shape_vectors - mu) / sigma).astype(np.float32)

    delta_norms = np.sqrt(np.einsum('ij,ij->i', deltas, deltas)).astype(np.float32)

    # Target scalar radius: p-th percentile of delta_norms
    theta_scalar = float(np.percentile(delta_norms, anomaly_percentile))

    # Store as uniform vector: norm(theta) == theta_scalar
    component = theta_scalar / np.sqrt(D) if D > 0 else 0.0
    theta = np.full(D, component, dtype=np.float32)

    # Anomaly rate guard
    expected_rate = 1.0 - anomaly_percentile / 100.0
    actual_rate = float((delta_norms >= theta_scalar).mean()) if theta_scalar > 0 else 0.0
    if actual_rate > expected_rate * 2.5:
        logger.warning(
            "Anomaly rate %.1f%% exceeds %.1f\u00d7 expected %.1f%% \u2014 "
            "discrete delta_norm distribution (unique values: %d). "
            "Consider increasing population diversity or adjusting anomaly_percentile.",
            actual_rate * 100, actual_rate / expected_rate, expected_rate * 100,
            len(np.unique(delta_norms)),
        )

    return mu, sigma, theta, deltas, delta_norms, cholesky_inv


def compute_conformal_p(
    delta_norms: np.ndarray,
    sorted_norms: np.ndarray | None = None,
) -> np.ndarray:
    """Compute conformal p-values from delta norms.

    p = fraction of population with delta_norm >= this entity's delta_norm.
    p=0.01 means only 1% of the population has a higher delta_norm.

    Args:
        delta_norms: Per-entity delta norms.
        sorted_norms: Pre-sorted delta_norms. If provided, skips the internal
            sort — allows callers that already hold a sorted copy to avoid
            duplicate work.
    """
    n = len(delta_norms)
    if n == 0:
        return np.array([], dtype=np.float32)
    if sorted_norms is None:
        sorted_norms = np.sort(delta_norms)
    # Number of entities with norm >= this entity's norm
    right_ranks = n - np.searchsorted(sorted_norms, delta_norms, side="left")
    return (right_ranks / n).astype(np.float32)


def compute_stats_grouped(
    shape_vectors: np.ndarray,
    group_ids: np.ndarray,
    anomaly_percentile: float = 95.0,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    np.ndarray,
    np.ndarray,
]:
    """Compute per-group population statistics.

    Returns:
        group_results: {group_name: (mu, sigma, theta, population_size)}
        deltas: (N, D) float32 — z-scored against per-group mu/sigma
        delta_norms: (N,) float32 — L2 norms of per-group deltas
    """
    n, D = shape_vectors.shape
    deltas = np.zeros_like(shape_vectors, dtype=np.float32)
    delta_norms = np.zeros(n, dtype=np.float32)
    group_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

    unique_groups = np.unique(group_ids)
    for gid in unique_groups:
        gid_str = str(gid)
        mask = group_ids == gid
        group_shape = shape_vectors[mask]

        if len(group_shape) < 2:
            # Too few entities for meaningful stats — use global-like single point
            mu_g = group_shape.mean(axis=0).astype(np.float32)
            sigma_g = np.full(D, SIGMA_EPS, dtype=np.float32)
            theta_g = np.zeros(D, dtype=np.float32)
            deltas[mask] = ((group_shape - mu_g) / sigma_g).astype(np.float32)
            _masked = deltas[mask]
            delta_norms[mask] = np.sqrt(
                np.einsum('ij,ij->i', _masked, _masked),
            ).astype(np.float32)
            group_results[gid_str] = (mu_g, sigma_g, theta_g, int(mask.sum()))
            continue

        mu_g, sigma_g, theta_g, group_deltas, group_norms, _ = compute_stats(
            group_shape, anomaly_percentile,
        )
        deltas[mask] = group_deltas
        delta_norms[mask] = group_norms
        group_results[gid_str] = (mu_g, sigma_g, theta_g, int(mask.sum()))

    return group_results, deltas, delta_norms


def compute_dimension_weights(
    shape_vectors: np.ndarray,
    method: str = "kurtosis",
) -> np.ndarray:
    """Compute per-dimension importance weights from shape vector distribution.

    Methods:
        kurtosis: w[d] = max(1.0, excess_kurtosis(d) / 3.0)
            Dims with heavy tails (outlier-generating) get amplified.
            Normal distribution has kurtosis=3 (excess=0) → weight=1.0
            Heavy-tailed dim with kurtosis=12 → weight=4.0
        uniform: w[d] = 1.0 (no weighting, current default behavior)

    Returns:
        weights: (D,) float32 — per-dimension weights, all >= 1.0
    """
    D = shape_vectors.shape[1]

    if method == "uniform" or D == 0:
        return np.ones(D, dtype=np.float32)

    if method == "kurtosis":
        raw_sigma = shape_vectors.std(axis=0)
        low_var = raw_sigma < SIGMA_EPS
        sigma = np.maximum(raw_sigma, SIGMA_EPS)
        z = (shape_vectors - shape_vectors.mean(axis=0)) / sigma
        z[:, low_var] = 0.0  # prevent inf from near-zero sigma
        kurt = (z ** 4).mean(axis=0) - 3.0
        weights = np.maximum(1.0, (kurt + 3.0) / 3.0).astype(np.float32)
        # Near-constant dims have no informative signal — reset to 1.0
        weights[low_var] = 1.0
        return weights

    raise ValueError(f"Unknown weight method: {method!r}")


def compute_per_dim_anomaly_count(
    deltas: np.ndarray,
    percentile: float = 99.0,
) -> np.ndarray:
    """Count dimensions where |delta[d]| exceeds the p-th percentile of |deltas[:, d]|.

    Returns (N,) int32 — number of anomalous dimensions per entity.
    Entity with count >= 3 has multi-dimensional anomaly signal even if global
    delta_norm is below theta.
    """
    abs_deltas = np.abs(deltas)
    thresholds = np.percentile(abs_deltas, percentile, axis=0)  # (D,)
    exceeds = (abs_deltas > thresholds).astype(np.int32)
    return exceeds.sum(axis=1).astype(np.int32)


def fit_kmeans_components(
    shape_vectors: np.ndarray,
    n_components: int = 5,
    anomaly_percentile: float = 95.0,
    max_iter: int = 30,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, int]], np.ndarray]:
    """Fit k-means++ components and compute per-component anomaly thresholds.

    Returns:
        components: [(mu, sigma, theta, pop_size), ...] per cluster
        assignments: (N,) int — cluster index per entity
    """
    n, D = shape_vectors.shape
    rng = np.random.default_rng(seed)

    # k-means++ initialization
    centers = np.zeros((n_components, D), dtype=np.float32)
    centers[0] = shape_vectors[rng.integers(n)]
    for k in range(1, n_components):
        dists = np.min([
            np.sum((shape_vectors - centers[j]) ** 2, axis=1)
            for j in range(k)
        ], axis=0)
        probs = dists / dists.sum()
        centers[k] = shape_vectors[rng.choice(n, p=probs)]

    # k-means iterations
    assignments = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # Assign to nearest center
        dists = np.stack([
            np.sum((shape_vectors - centers[k]) ** 2, axis=1)
            for k in range(n_components)
        ], axis=1)
        new_assignments = dists.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        # Update centers
        for k in range(n_components):
            mask = assignments == k
            if mask.sum() > 0:
                centers[k] = shape_vectors[mask].mean(axis=0)

    # Compute per-cluster stats
    components = []
    for k in range(n_components):
        mask = assignments == k
        cluster_shapes = shape_vectors[mask]
        pop = int(mask.sum())

        if pop < 2:
            mu_k = centers[k].astype(np.float32)
            sigma_k = np.full(D, SIGMA_EPS, dtype=np.float32)
            theta_k = np.zeros(D, dtype=np.float32)
            components.append((mu_k, sigma_k, theta_k, pop))
            continue

        mu_k, sigma_k, theta_k, _, _, _ = compute_stats(
            cluster_shapes, anomaly_percentile,
        )
        components.append((mu_k, sigma_k, theta_k, pop))

    return components, assignments


def welford_batch_update(
    running_mean: np.ndarray,
    running_m2: np.ndarray,
    n_total: int,
    batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Chan's parallel merge formula for batch Welford updates.

    Incrementally update running mean and M2 (sum of squared deviations)
    with a new batch of observations. Works in float64 for numerical stability.

    Args:
        running_mean: (D,) float64 — current running mean.
        running_m2: (D,) float64 — current sum of squared deviations.
        n_total: int — number of observations seen so far.
        batch: (m, D) float32/64 — new batch of observations.

    Returns:
        (updated_mean, updated_m2, new_n_total)
    """
    m = len(batch)
    if m == 0:
        return running_mean, running_m2, n_total
    batch_mean = batch.mean(axis=0).astype(np.float64)
    batch_m2 = batch.var(axis=0, ddof=0).astype(np.float64) * m
    new_n = n_total + m
    delta = batch_mean - running_mean
    running_mean = running_mean + delta * m / new_n
    running_m2 = running_m2 + batch_m2 + delta ** 2 * n_total * m / new_n
    return running_mean, running_m2, new_n


def reservoir_update(
    reservoir: np.ndarray,
    reservoir_count: int,
    batch: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Reservoir sampling update (Algorithm R).

    Maintains a fixed-size reservoir of samples from a stream.
    reservoir is pre-allocated (K, D) and updated in-place.

    Args:
        reservoir: (K, D) float32 — pre-allocated reservoir buffer.
        reservoir_count: int — total observations seen so far (before this batch).
        batch: (m, D) float32 — new batch to sample from.
        rng: numpy random Generator for reproducibility.

    Returns:
        new_count: int — updated total observations count.
    """
    K = reservoir.shape[0]
    m = len(batch)
    if m == 0:
        return reservoir_count

    for i in range(m):
        idx = reservoir_count + i
        if idx < K:
            reservoir[idx] = batch[i]
        else:
            j = rng.integers(0, idx + 1)
            if j < K:
                reservoir[j] = batch[i]

    return reservoir_count + m
