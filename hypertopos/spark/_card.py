# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``calibration_card``: what a calibration can and cannot see."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from pyspark.sql import DataFrame

from hypertopos.engine.robust import POOLED_GROUP, min_reference_size
from hypertopos.spark._explain import Sphere, sphere_of

TIED_SHARE = 0.95


def calibration_card(like: DataFrame | Sphere) -> dict[str, Any]:
    """The limits of a calibration's evidence, read on the driver from the saved calibration.

    Args:
        like: A ``find_anomalies`` result, or its ``Sphere``.

    Returns:
        ``entities`` (rows the calibration was fitted on), ``percentile``,
        ``alpha`` (``1 - percentile / 100``, the p-value at or below which an
        entity is flagged), ``flagged`` (fitted entities flagged), ``score_axes``,
        ``window_days``; ``p_floor`` (``2 / (n + 1)``, the smallest p-value a
        fitted entity can get: it counts itself in the reference) and ``can_flag``
        (whether that reaches ``alpha``);
        ``tied_dimensions``: ``{dim: share}`` for dimensions where one value
        holds at least 95 % of the entities, so a single other value is an
        extreme deviation by itself. With ``group_by`` also ``groups``:
        ``{group: {entities, scored_by, reference_size, p_floor, can_flag}}``,
        where ``scored_by`` is ``group`` (its own reference, with
        ``score_per_group``) or ``global``, and ``pooled_groups``: the groups
        calibrated together as one pool because they were too small.
    """
    sphere = like if isinstance(like, Sphere) else sphere_of(like)
    calibration = sphere.calibration()
    entities = sphere.entities()
    ref = calibration.reference
    alpha = 1.0 - calibration.percentile / 100.0
    scores = calibration.score(entities, members=True)

    can_flag_from = min_reference_size(calibration.percentile)

    def floor(n: int) -> float:
        return 2.0 / (n + 1)

    # Ties are read on the measured values: the small-count shrinkage spreads a
    # column of zeros into as many values as there are event counts.
    tied: dict[str, float] = {}
    for dim in calibration.dims:
        values = np.nan_to_num(entities[dim].to_numpy(zero_copy_only=False).astype(np.float64))
        _, counts = np.unique(values, return_counts=True)
        share = float(counts.max() / len(values)) if len(values) else 0.0
        if share >= TIED_SHARE:
            tied[dim] = share

    card: dict[str, Any] = {
        "entities": entities.num_rows,
        "percentile": calibration.percentile,
        "alpha": alpha,
        "flagged": int(np.count_nonzero(scores.is_anomaly)),
        "score_axes": list(ref.axes),
        "window_days": calibration.window_days,
        "p_floor": floor(len(ref.cal_sorted)),
        "can_flag": len(ref.cal_sorted) >= can_flag_from,
        "tied_dimensions": tied,
    }
    if calibration.group_by is not None and calibration.groups is not None:
        members = Counter(str(v) for v in entities[calibration.group_by].to_pylist())
        groups: dict[str, Any] = {}
        for name in sorted(members):
            key = POOLED_GROUP if name in calibration.pooled else name
            group = calibration.groups.get(key)
            own = group is not None and group.reference is not None and calibration.score_per_group
            size = len(group.reference.cal_sorted) if own else len(ref.cal_sorted)  # type: ignore[union-attr]
            groups[name] = {
                "entities": members[name],
                "scored_by": "group" if own else "global",
                "reference_size": size,
                "p_floor": floor(size),
                "can_flag": size >= can_flag_from,
            }
        card["groups"] = groups
        card["pooled_groups"] = sorted(calibration.pooled)
    return card
