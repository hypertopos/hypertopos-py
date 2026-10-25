# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Investigative explanation engine for anomalous entities."""

from __future__ import annotations

from typing import Any

import numpy as np

from hypertopos.engine.geometry import GDSEngine, _per_dim_anomaly_contributions


def build_explanation(
    delta: np.ndarray | list[float],
    dim_labels: list[str],
    theta_norm: float,
    delta_norm: float,
    conformal_p: float | None = None,
    temporal_slices: int | None = None,
    reputation: dict[str, Any] | None = None,
    dimension_kinds: list[str] | None = None,
) -> dict[str, Any]:
    """Build structured anomaly explanation combining all available signals.

    ``top_dimensions`` ranks every dim by its share of ``delta_norm**2``
    (``delta[i]**2 / sum(delta**2)``) — the metric behind ``is_anomaly`` —
    so ``top_dimensions[0]`` is the dim whose return to its population mu
    drops ``delta_norm`` most. Each entry is ``{dim, kind, delta,
    pct_of_total}``: ``dim`` is the label, ``kind`` the dim's distribution
    family tag (present when *dimension_kinds* is given), ``delta`` the
    signed stored delta. Dims with no share are left out.
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

    kinds = (
        dimension_kinds
        if dimension_kinds is not None and len(dimension_kinds) == len(delta)
        else None
    )
    contribs = _per_dim_anomaly_contributions(delta)
    total = float(contribs.sum())
    n = min(len(dim_labels), len(contribs))
    # Stable sort: ties keep dim order, so top_dimensions[0] matches the
    # np.argmax pick of reliability_flags.dominant_dim.
    order = sorted(range(n), key=lambda j: -contribs[j]) if total > 0 else []
    top_dims: list[dict[str, Any]] = []
    for i in order:
        pct = round(float(contribs[i]) / total, 4)
        if pct <= 0.0:
            continue
        entry: dict[str, Any] = {"dim": dim_labels[i]}
        if kinds is not None:
            entry["kind"] = kinds[i]
        entry["delta"] = round(float(delta[i]), 4)
        entry["pct_of_total"] = pct
        top_dims.append(entry)

    result: dict[str, Any] = {
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
