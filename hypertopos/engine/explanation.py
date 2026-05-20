# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Diverse-cover explanation engine.

Pure numpy math for greedy max-K disjoint cover over per-dim
anomaly contributions. Used by ``GDSNavigator.find_diverse_explanations``
to surface K strictly disjoint hypothesis subsets, each one a small set
of dimensions that jointly explain at least ``min_contribution_pct``
of the entity's anomaly mass.

Pure function — no I/O, no storage access, no ``self``. Operates over
a contribution vector that the caller has already routed through
``_per_dim_anomaly_contributions`` (Bregman when calibration trio is
present, ``delta**2`` otherwise).

Design note (no diversity_alpha):
    Earlier revisions exposed a ``diversity_alpha`` knob that weighted
    a Jaccard penalty into the greedy marginal-gain rule. Under the
    strict-disjoint constraint (each dim appears in at most one
    hypothesis), the Jaccard between any candidate-extended hypothesis
    and any prior hypothesis is structurally 0 — the candidate dim
    cannot be in the prior set (we removed it from the remaining pool
    when we emitted the prior). The penalty term therefore vanishes
    on every comparison and ``diversity_alpha`` is a math no-op.
    Removed entirely so the API does not advertise a knob that does
    nothing.
"""
from __future__ import annotations

import numpy as np


def _jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard similarity over int sets.

    Returns 0.0 when both sets are empty (no overlap is the safest
    diversity prior for the greedy initialisation step); otherwise
    ``|a ∩ b| / |a ∪ b|``. Kept as a module-level helper because the
    navigator uses it for the post-hoc ``diversity_score`` (mean
    pairwise ``1 - jaccard`` over the emitted dim sets).
    """
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def submodular_diverse_cover(
    contributions: np.ndarray,
    dim_labels: list[str],  # noqa: ARG001
    *,
    n_hypotheses: int,
    min_contribution_pct: float = 0.10,
) -> tuple[list[set[int]], str | None]:
    """Greedy max-K strictly disjoint subsets of dim indices.

    For each hypothesis ``h`` in ``1..n_hypotheses``:

    - Start with an empty ``hypothesis_dims`` set.
    - Greedy inner loop: pick the dim ``d`` from the remaining pool
      (dims not yet claimed by any prior hypothesis) with the highest
      ``contributions[d]``. Add it to ``hypothesis_dims`` and drop it
      from the pool.
    - Stop adding dims once the joint share of ``hypothesis_dims``
      clears ``min_contribution_pct``.
    - If the pool empties before the joint share clears the floor, the
      partial hypothesis is dropped and the outer loop breaks — any
      subsequent hypothesis would have strictly less mass to draw on,
      so it cannot clear the floor either.

    Each dim appears in at most one hypothesis (strict disjoint sets,
    per the design intent).

    Args:
        contributions: per-dim Bregman mass, shape ``(D,)``.
        dim_labels: unused at this layer (kept for symmetry with the
            navigator-facing return contract); ``len == D`` is the
            caller's responsibility.
        n_hypotheses: requested number of hypotheses ``K``.
        min_contribution_pct: joint-share floor a hypothesis must clear
            before it is emitted, expressed as a fraction in
            ``[0, 1]``.

    Returns:
        Pair ``(dim_sets, degraded_reason)`` where ``dim_sets`` is the
        list of emitted hypotheses (each a ``set[int]`` of dim
        indices) and ``degraded_reason`` is:

        - ``None`` — all ``n_hypotheses`` returned.
        - ``"insufficient_diverse_mass"`` — fewer than ``n_hypotheses``
          returned because the remaining mass could not clear
          ``min_contribution_pct``.
        - ``"capped_to_dim_count"`` — ``n_hypotheses`` exceeded the
          dim count and was silently capped to ``D`` before the
          greedy loop ran.
    """
    contributions = np.asarray(contributions, dtype=np.float64)
    d_count = contributions.shape[0]

    degraded_reason: str | None = None
    requested = n_hypotheses
    if n_hypotheses > d_count:
        n_hypotheses = d_count
        degraded_reason = "capped_to_dim_count"

    total_mass = float(contributions.sum())
    if total_mass <= 0.0:
        # No mass anywhere — nothing to cover. Degrade with the same
        # reason the partial-mass branch uses; "capped_to_dim_count" is
        # preserved if it was already set (caller asked for more
        # hypotheses than dims).
        if degraded_reason is None:
            degraded_reason = "insufficient_diverse_mass"
        return [], degraded_reason

    remaining_dims: set[int] = set(range(d_count))
    emitted: list[set[int]] = []

    for _ in range(n_hypotheses):
        if not remaining_dims:
            break

        hypothesis_dims: set[int] = set()
        joint_mass = 0.0

        while remaining_dims and joint_mass / total_mass < min_contribution_pct:
            best_dim = max(remaining_dims, key=lambda d: contributions[d])
            hypothesis_dims.add(best_dim)
            remaining_dims.discard(best_dim)
            joint_mass = float(contributions[sorted(hypothesis_dims)].sum())

        joint_share = joint_mass / total_mass if total_mass > 0.0 else 0.0
        if joint_share + 1e-12 < min_contribution_pct:
            # Cannot clear the floor — drop the partial hypothesis and
            # stop. Any further hypothesis would have strictly less
            # mass to draw on, so it cannot clear the floor either.
            if requested > len(emitted):
                if degraded_reason != "capped_to_dim_count":
                    degraded_reason = "insufficient_diverse_mass"
            break

        emitted.append(hypothesis_dims)

    if len(emitted) < requested and degraded_reason is None:
        degraded_reason = "insufficient_diverse_mass"

    return emitted, degraded_reason
