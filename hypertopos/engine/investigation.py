# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Investigative explanation engine for anomalous entities."""
from __future__ import annotations

import numpy as np

from hypertopos.engine.geometry import GDSEngine


def build_explanation(
    delta: np.ndarray | list[float],
    dim_labels: list[str],
    theta_norm: float,
    delta_norm: float,
    conformal_p: float | None = None,
    temporal_slices: int | None = None,
    reputation: dict | None = None,
    dimension_kinds: list[str] | None = None,
    sigma: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    dimension_weights: np.ndarray | None = None,
) -> dict:
    """Build structured anomaly explanation combining all available signals.

    When *dimension_kinds*, *sigma*, and *mu* are provided, the
    ``top_dimensions`` section is replaced by per-dimension Bregman
    contributions, which more accurately attribute anomaly mass across
    mixed-family dimensions (Bernoulli, Poisson, Gaussian).
    """
    delta = np.asarray(delta, dtype=np.float64)

    if delta_norm <= theta_norm:
        return {
            "severity": "normal",
            "delta_norm": round(delta_norm, 4),
            "theta_norm": round(theta_norm, 4),
        }

    ratio = delta_norm / theta_norm if theta_norm > 0 else 0.0
    if ratio >= 2.5:
        severity = "extreme"
    elif ratio >= 1.5:
        severity = "high"
    elif ratio >= 1.1:
        severity = "medium"
    else:
        severity = "low"

    witness = GDSEngine.witness_set(delta, theta_norm, dim_labels)
    anti_w = GDSEngine.anti_witness(delta, theta_norm, dim_labels)

    if dimension_kinds is not None and sigma is not None and mu is not None:
        if len(dimension_kinds) != len(delta):
            # Dimension count mismatch — fall through to legacy abs_delta path
            dimension_kinds = None

    if dimension_kinds is not None and sigma is not None and mu is not None:
        from hypertopos.builder._bregman import bregman_divergence
        d_arr = np.array(delta)
        if dimension_weights is not None:
            w = np.maximum(np.array(dimension_weights), 1e-9)
            d_arr = d_arr / w
        shape = d_arr * np.array(sigma) + np.array(mu)
        contribs = bregman_divergence(shape, np.array(mu), np.array(sigma), dimension_kinds)
        total = float(contribs.sum())
        top_dims = sorted(
            [
                {
                    "dim": dim_labels[i],
                    "kind": dimension_kinds[i],
                    "bregman": round(float(contribs[i]), 4),
                    "pct_of_total": round(float(contribs[i]) / total, 4) if total > 0 else 0.0,
                }
                for i in range(min(len(dim_labels), len(contribs)))
            ],
            key=lambda x: x["bregman"],
            reverse=True,
        )
    else:
        top_dims = GDSEngine.anomaly_dimensions(delta, dim_labels, top_n=5)

    result: dict = {
        "severity": severity,
        "delta_norm": round(delta_norm, 4),
        "theta_norm": round(theta_norm, 4),
        "ratio": round(ratio, 2),
        "witness": witness,
        "repair": anti_w,
        "top_dimensions": top_dims,
    }
    if conformal_p is not None:
        result["conformal_p"] = round(conformal_p, 6)
    if temporal_slices is not None:
        result["temporal_slices"] = temporal_slices
    if reputation is not None:
        result["reputation"] = reputation
    return result
