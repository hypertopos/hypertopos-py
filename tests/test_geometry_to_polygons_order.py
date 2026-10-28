# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``geometry_to_polygons`` keeps a caller's row order and carries the score layer."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa

from hypertopos.engine.geometry import GDSEngine


def _geo() -> pa.Table:
    now = datetime.now(UTC)
    # Rows ordered by anomaly_score; delta_norm orders them differently.
    return pa.table(
        {
            "primary_key": ["a", "b", "c"],
            "scale": [1, 1, 1],
            "delta": [[0.1, 0.0], [3.0, 0.0], [1.0, 0.0]],
            "delta_norm": [0.1, 3.0, 1.0],
            "is_anomaly": [True, False, False],
            "last_refresh_at": [now] * 3,
            "updated_at": [now] * 3,
            "anomaly_score": [0.99, 0.6, 0.4],
            "conformal_p": [0.01, 0.4, 0.6],
            "score_rank_pct": [99.0, 60.0, 40.0],
            "score_delta": [[2.5, -1.0], [0.3, 0.1], None],
        }
    )


def test_preserve_order_keeps_rows_and_truncates_in_order() -> None:
    polys = GDSEngine(None, None).geometry_to_polygons(
        _geo(), top_n=2, preserve_order=True, pattern_id="p", pattern_type="anchor"
    )
    assert [p.primary_key for p in polys] == ["a", "b"]


def test_default_order_is_still_by_delta_norm() -> None:
    polys = GDSEngine(None, None).geometry_to_polygons(_geo(), pattern_id="p")
    assert [p.primary_key for p in polys] == ["b", "c", "a"]


def test_score_layer_fields_are_populated() -> None:
    polys = GDSEngine(None, None).geometry_to_polygons(_geo(), preserve_order=True, pattern_id="p")
    assert len(polys) == 3
    first, _, last = polys
    assert first.anomaly_score == 0.99 and first.conformal_p == 0.01
    assert first.score_rank_pct == 99.0
    np.testing.assert_allclose(first.score_delta, [2.5, -1.0])
    assert last.score_delta is None
