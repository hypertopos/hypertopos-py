# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Bootstrap anomaly confidence module.

Estimates the stability of anomaly detection by running B bootstrap
iterations with stratified resampling.  For each bootstrap iteration a fresh
population centre (mu_b, sigma_b) and per-dimension anomaly threshold
(theta_b) are computed from the resampled sample, and every entity is scored
under those bootstrap statistics.  The final confidence value for entity i is
the fraction of bootstrap iterations in which entity i was flagged as anomalous.

A confidence value close to 1.0 means the entity is robustly anomalous across
nearly all plausible realisations of the population statistics.  A value close
to 0.0 means it is rarely anomalous — i.e. its apparent deviation is fragile.
"""
from __future__ import annotations

import numpy as np

from hypertopos.builder._bregman import bregman_norms, per_dim_theta

# Floor on per-dimension sigma to avoid division by zero in bootstrap samples
SIGMA_EPS: float = 1e-2


def compute_bootstrap_confidence(
    shape_vectors: np.ndarray,
    kinds: list[str],
    anomaly_percentile: float = 95.0,
    B: int = 1000,
    weights: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray | None:
    """Estimate anomaly confidence via bootstrap resampling.

    For each of B bootstrap iterations a stratified (or uniform) resample of
    the population is drawn, new population statistics are fitted, and every
    entity is evaluated under those statistics.  The returned confidence for
    each entity is the fraction of iterations in which it exceeded the
    bootstrap anomaly threshold.

    Both per-dimension thresholds and entity norms use the same ``weights``
    (typically kurtosis weights from the build) to ensure unit consistency.
    Weights amplify high-kurtosis dimensions in both the threshold computation
    and the entity scoring, preserving the same ranking as weighted delta_norm.

    Args:
        shape_vectors: (N, D) float array of entity feature vectors.  Values
            are interpreted according to ``kinds``.
        kinds: list of D strings specifying the Bregman divergence kind for
            each dimension.  Each element must be one of ``"gaussian"``,
            ``"poisson"``, or ``"bernoulli"``.
        anomaly_percentile: empirical percentile used to derive the per-dimension
            anomaly threshold inside each bootstrap iteration (default 95.0).
        B: number of bootstrap iterations.  Pass ``B=0`` (or any non-positive
            value) to skip bootstrap and return ``None``.
        group_ids: optional (N,) integer array for stratified resampling.
            When provided, each bootstrap resample draws a replacement sample
            of the same size within every group independently, then concatenates
            them.  When ``None``, uniform sampling over all N entities is used.
        seed: integer seed for the NumPy random generator (default 42).

    Returns:
        ``None`` if ``B <= 0``, otherwise a ``(N,)`` float32 array with values
        in ``[0.0, 1.0]`` representing the bootstrap anomaly confidence for
        each entity.
    """
    if B <= 0:
        return None

    X = np.asarray(shape_vectors, dtype=np.float64)
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    counts = np.zeros(N, dtype=np.int32)
    valid_iters = 0

    # Pre-compute group structure for stratified resampling
    if group_ids is not None:
        group_ids_arr = np.asarray(group_ids)
        unique_groups = np.unique(group_ids_arr)
        group_indices: dict[int, np.ndarray] = {
            int(g): np.where(group_ids_arr == g)[0] for g in unique_groups
        }
    else:
        group_indices = {}
        unique_groups = np.array([], dtype=np.int64)

    for _ in range(B):
        # --- stratified or uniform resample ---
        if group_ids is not None:
            parts = [
                rng.choice(group_indices[int(g)], size=len(group_indices[int(g)]), replace=True)
                for g in unique_groups
            ]
            sample_idx = np.concatenate(parts)
        else:
            sample_idx = rng.choice(N, size=N, replace=True)

        sample = X[sample_idx]  # (N, D)

        # Fit bootstrap statistics (keep float64 throughout the loop)
        mu_b = sample.mean(axis=0)
        sigma_b = np.maximum(sample.std(axis=0), SIGMA_EPS)

        # Per-dimension threshold from bootstrap sample
        theta_b = per_dim_theta(
            sample,
            mu_b,
            sigma_b,
            kinds,
            anomaly_percentile,
        )
        # Apply weights to per-dim thresholds (same as bregman_norms weighting)
        if weights is not None:
            theta_b = theta_b * weights
        theta_total_b = float(theta_b.sum())

        if theta_total_b <= 0.0:
            continue  # skip degenerate iteration (all-constant dimensions)

        valid_iters += 1

        # Score ALL entities under bootstrap calibration (weighted consistently)
        norms_b = bregman_norms(
            X,
            mu_b,
            sigma_b,
            kinds,
            weights=weights,
        )
        counts += (norms_b >= theta_total_b).astype(np.int32)

    if valid_iters == 0:
        return None

    return (counts / valid_iters).astype(np.float32)
