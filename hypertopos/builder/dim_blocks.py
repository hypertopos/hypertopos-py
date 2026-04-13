# Copyright (C) 2026 Karol Kedzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Normalization functions for generalized dimension blocks (g/t/s).

Three block types extend the shape vector beyond the structural (d)
and fill-indicator (m) blocks:

- **g** (geographic): lat/lon or other continuous spatial coordinates
- **t** (metric): arbitrary continuous numeric properties (balance, income, ...)
- **s** (semantic): high-dimensional embeddings reduced via PCA

All normalization uses empirical mu/sigma z-scoring.  For semantic blocks
PCA is applied first (pure numpy, no sklearn).
"""
from __future__ import annotations

import numpy as np

# Minimum sigma floor — prevents division by zero for constant columns.
_SIGMA_FLOOR = 1.0


def normalize_metric_block(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize continuous metric columns.

    Args:
        values: (N, D) float array of raw metric values.

    Returns:
        (normalized, mu, sigma) where:
        - normalized: (N, D) z-scored values
        - mu: (D,) column means
        - sigma: (D,) column standard deviations (floored to _SIGMA_FLOOR)
    """
    values = np.asarray(values, dtype=np.float32)
    mu = values.mean(axis=0)
    sigma = values.std(axis=0)
    sigma = np.maximum(sigma, _SIGMA_FLOOR)
    normalized = ((values - mu) / sigma).astype(np.float32)
    return normalized, mu, sigma


def normalize_geo_block(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize geographic coordinates (lat/lon).

    Delegates to :func:`normalize_metric_block` — lat/lon are continuous
    values that benefit from the same mu/sigma treatment.

    Args:
        values: (N, D) float array of geographic coordinates.

    Returns:
        (normalized, mu, sigma) — same as normalize_metric_block.
    """
    return normalize_metric_block(values)


def normalize_semantic_block(
    values: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA-reduce and z-score normalize semantic embeddings.

    Uses truncated SVD (no sklearn dependency).  n_components is clamped
    to ``min(n_components, D_raw)`` so requests exceeding the input
    dimensionality degrade gracefully.

    Args:
        values: (N, D_raw) float array of embedding vectors.
        n_components: Target dimensionality after PCA.

    Returns:
        (normalized, mu, sigma, pca_components) where:
        - normalized: (N, n_components) z-scored PCA-reduced values
        - mu: (n_components,) post-PCA column means
        - sigma: (n_components,) post-PCA column stds (floored)
        - pca_components: (n_components, D_raw) PCA projection matrix
    """
    values = np.asarray(values, dtype=np.float32)
    _n, d_raw = values.shape

    # Clamp n_components to available dimensions
    n_components = min(n_components, d_raw)

    # Center the data
    center_mu = values.mean(axis=0)
    centered = values - center_mu

    # Truncated SVD — only need top-k right singular vectors
    # Full SVD is fine for moderate D_raw; for very large D_raw the caller
    # should pre-truncate columns.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = vt[:n_components]  # (n_components, D_raw)

    # Project
    projected = centered @ pca_components.T  # (N, n_components)

    # Z-score the projected values
    mu = projected.mean(axis=0)
    sigma = projected.std(axis=0)
    sigma = np.maximum(sigma, _SIGMA_FLOOR)
    normalized = ((projected - mu) / sigma).astype(np.float32)

    return normalized, mu, sigma, pca_components
