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
) -> dict:
    """Build structured anomaly explanation combining all available signals."""
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
    top_dims = GDSEngine.anomaly_dimensions(delta, dim_labels, top_n=5)
    anti_w = GDSEngine.anti_witness(delta, theta_norm, dim_labels)

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
