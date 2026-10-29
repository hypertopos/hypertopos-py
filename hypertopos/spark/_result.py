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


def result_table(
    entities: pa.Table, calibration: FrozenCalibration, *, members: bool = False
) -> pa.Table:
    """Per-entity result with flat reason columns (``reason_<i>_dim/_delta/_pct``).

    Reason shares are each dimension's Shapley share of the calibrated score's
    Mahalanobis evidence in percent (the split ``explain_anomaly`` reports on a
    sphere), so all shares of an entity sum to 100; a dimension with no share is
    not a reason. ``_delta`` is the dimension's deviation in standard deviations.
    ``members=True``: ``entities`` is the table the calibration was fitted on.
    """
    scores = calibration.score(entities, members=members)
    delta = scores.delta.astype(np.float64)
    shares = 100.0 * calibration.score_shares(entities, scores.score_delta)
    k = min(TOP_REASONS, len(calibration.dims))
    top = np.argsort(-shares, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(delta))
    names = np.asarray(calibration.dims, dtype=object)

    columns: dict[str, pa.Array] = {
        KEY: entities[KEY].combine_chunks(),
        "anomaly_score": pa.array(scores.anomaly_score.astype(np.float64)),
        "is_anomaly": pa.array(scores.is_anomaly),
        "p_value": pa.array(scores.conformal_p.astype(np.float64)),
        "score_rank_pct": pa.array(scores.score_rank_pct.astype(np.float64)),
        "delta_norm": pa.array(scores.delta_norm.astype(np.float64)),
    }
    for i in range(k):
        columns[f"reason_{i + 1}_dim"] = pa.array(names[top[:, i]].tolist(), type=pa.string())
        columns[f"reason_{i + 1}_delta"] = pa.array(delta[rows, top[:, i]])
        columns[f"reason_{i + 1}_pct"] = pa.array(shares[rows, top[:, i]])
    if calibration.group_by is not None and scores.calibration_scope is not None:
        columns[calibration.group_by] = entities[calibration.group_by].combine_chunks()
        columns["calibration_scope"] = pa.array(scores.calibration_scope.tolist(), pa.string())
    return pa.table(columns)
