# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Label-aware per-dim calibration on top of standard mu/sigma stats.

Given a pattern's stacked delta vectors and a binary class label per
entity, compute four per-dim moments (``mu_pos``, ``sigma_pos``,
``mu_neg``, ``sigma_neg``) and the Fisher LDA direction across all dims
(``signed_direction_vector``). Per-dim ``direction`` exposes the
component of the global LDA axis along that dim — large absolute value
means the dim contributes to label separation, small means it does not.

Engine-pure: no I/O, no sphere format coupling, no builder dependency.
Builder consumes the result via an opt-in flag; serialization into
``pattern.json`` is the responsibility of the calling layer.

Consistent with ``engine.dim_audit.fit_lda_direction``: the LDA
direction is sign-oriented so projecting onto it lands the
positive-labelled class on the positive side
(``w . (mu_pos - mu_neg) > 0``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hypertopos.engine.dim_audit import fit_lda_direction

__all__ = [
    "CalibrationResult",
    "DimCalibration",
    "calibrate_label_aware",
]


@dataclass
class DimCalibration:
    """Per-dim label-aware moments + Fisher direction component."""

    mu_pos: float
    sigma_pos: float
    mu_neg: float
    sigma_neg: float
    direction: float


@dataclass
class CalibrationResult:
    """Output of ``calibrate_label_aware`` over one pattern.

    Attributes:
        per_dim: ``{dim_label: DimCalibration}`` mapping. Iteration order
            mirrors ``dim_labels`` so callers that need a positional view
            can zip back.
        signed_direction_vector: ``(n_dims,)`` unit-norm Fisher LDA
            direction across all dims. Sign-oriented so projection onto
            it lands the positive class on the positive side.
        fisher_score: Fisher discriminant ratio reported by the LDA fit
            — a higher number means more class separation along the
            direction.
        n_pos: number of positive-labelled samples used.
        n_neg: number of negative-labelled samples used.
    """

    per_dim: dict[str, DimCalibration] = field(default_factory=dict)
    signed_direction_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
    )
    fisher_score: float = 0.0
    n_pos: int = 0
    n_neg: int = 0


def calibrate_label_aware(
    *,
    deltas: np.ndarray,
    labels: np.ndarray,
    dim_labels: list[str] | None = None,
    regularization: float = 1e-6,
) -> CalibrationResult:
    """Per-dim label-aware calibration over stacked delta vectors.

    For each dim ``i`` compute the four moments on the labelled subset:
        - ``mu_pos = mean(deltas[labels==1, i])``
        - ``sigma_pos = std(deltas[labels==1, i], ddof=0)``
        - ``mu_neg = mean(deltas[labels==0, i])``
        - ``sigma_neg = std(deltas[labels==0, i], ddof=0)``

    The Fisher LDA direction is fitted once over the full delta matrix
    via ``engine.dim_audit.fit_lda_direction``; per-dim ``direction``
    holds the ``i``-th component of that unit-norm vector. Large
    ``|direction|`` means the dim carries the separating signal; values
    near zero mean the dim is irrelevant once the global axis is fixed.

    Args:
        deltas: ``(n_entities, n_dims)`` delta vectors. Rows are
            entities, columns are pattern dims. Must be 2-D and finite.
        labels: ``(n_entities,)`` binary labels in ``{0, 1}``. Positive
            label ``1`` is treated as the "anomalous" / "of interest"
            class.
        dim_labels: optional per-dim names of length ``n_dims``. When
            ``None``, dims are keyed as ``"dim_0"``, ``"dim_1"``, ….
        regularization: ridge added to within-class scatter before the
            LDA linear solve. Default ``1e-6`` matches the engine-wide
            epsilon style and ``fit_lda_direction``.

    Returns:
        ``CalibrationResult`` with the per-dim moments + Fisher
        direction. Caller decides whether to persist the result into
        sphere format and how to handle missing-class / degenerate
        inputs — this function re-raises every ``fit_lda_direction``
        error verbatim.

    Raises:
        ValueError: propagated from input validation or from
            ``fit_lda_direction`` (missing class, identical class means,
            fewer than 2 samples per class, non-finite values, etc.).
    """
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be 2-D, got shape {deltas.shape}")
    if labels.shape != (deltas.shape[0],):
        raise ValueError(
            f"labels shape mismatch: {labels.shape} vs "
            f"({deltas.shape[0]},)",
        )
    n_dims = deltas.shape[1]
    if dim_labels is None:
        dim_labels = [f"dim_{i}" for i in range(n_dims)]
    if len(dim_labels) != n_dims:
        raise ValueError(
            f"dim_labels length {len(dim_labels)} != n_dims {n_dims}",
        )

    lda = fit_lda_direction(
        deltas=deltas, labels=labels, regularization=regularization,
    )
    direction: np.ndarray = lda["direction"]

    mask_pos = labels == 1
    mask_neg = labels == 0
    pos = deltas[mask_pos]
    neg = deltas[mask_neg]

    mu_pos = pos.mean(axis=0)
    sigma_pos = pos.std(axis=0, ddof=0)
    mu_neg = neg.mean(axis=0)
    sigma_neg = neg.std(axis=0, ddof=0)

    per_dim: dict[str, DimCalibration] = {}
    for i, name in enumerate(dim_labels):
        per_dim[name] = DimCalibration(
            mu_pos=float(mu_pos[i]),
            sigma_pos=float(sigma_pos[i]),
            mu_neg=float(mu_neg[i]),
            sigma_neg=float(sigma_neg[i]),
            direction=float(direction[i]),
        )

    return CalibrationResult(
        per_dim=per_dim,
        signed_direction_vector=direction.astype(np.float32),
        fisher_score=float(lda["fisher_score"]),
        n_pos=int(lda["n_anom"]),
        n_neg=int(lda["n_normal"]),
    )
