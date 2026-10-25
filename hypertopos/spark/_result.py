# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Driver pass: score the entity table against its calibration, one row per entity."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from hypertopos.scoring import FrozenCalibration
from hypertopos.spark._build import KEY

TOP_REASONS = 3


def result_table(entities: pa.Table, calibration: FrozenCalibration) -> pa.Table:
    """Per-entity result with flat reason columns (``reason_<i>_dim/_delta/_pct``).

    Reason shares are each dimension's part of ``delta_norm**2`` in percent, so
    all shares of an entity sum to 100; a dimension with no share is not a reason.
    """
    scores = calibration.score(entities)
    delta = scores.delta.astype(np.float64)
    squared = delta**2
    total = squared.sum(axis=1, keepdims=True)
    shares = np.divide(squared * 100.0, total, out=np.zeros_like(squared), where=total > 0)
    k = min(TOP_REASONS, len(calibration.dims))
    top = np.argsort(-shares, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(delta))
    names = np.asarray(calibration.dims, dtype=object)

    columns: dict[str, pa.Array] = {
        KEY: entities[KEY].combine_chunks(),
        "delta_norm": pa.array(scores.delta_norm.astype(np.float64)),
        "is_anomaly": pa.array(scores.is_anomaly),
        "p_value": pa.array(scores.conformal_p.astype(np.float64)),
        "rank_pct": pa.array(scores.rank_pct.astype(np.float64)),
    }
    for i in range(k):
        columns[f"reason_{i + 1}_dim"] = pa.array(names[top[:, i]].tolist(), type=pa.string())
        columns[f"reason_{i + 1}_delta"] = pa.array(delta[rows, top[:, i]])
        columns[f"reason_{i + 1}_pct"] = pa.array(shares[rows, top[:, i]])
    if calibration.group_by is not None and scores.calibration_scope is not None:
        columns[calibration.group_by] = entities[calibration.group_by].combine_chunks()
        columns["calibration_scope"] = pa.array(scores.calibration_scope.tolist(), pa.string())
    return pa.table(columns)
