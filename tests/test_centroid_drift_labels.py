# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""centroid_map drift direction is labelled by dimension, not positional.

Temporal slices carry only the leading structural dims, so the drift vector can
be shorter than the pattern's dim_labels; a bare list cannot be aligned.
"""

from __future__ import annotations

import pytest

from hypertopos.sphere import HyperSphere
from tests.conftest import FIXTURES_PATH


def test_centroid_drift_direction_is_a_dimension_map() -> None:
    session = HyperSphere.open(str(FIXTURES_PATH)).session("drift-labels")
    pattern = session.sphere().patterns["customer_pattern"]
    result = session.navigator().centroid_map("customer_pattern", "products")
    drifts = [g["centroid_drift"] for g in result["group_centroids"] if "centroid_drift" in g]
    assert drifts, "fixture must give at least one group a drift"
    for drift in drifts:
        direction = drift["direction"]
        assert isinstance(direction, dict)
        labels = list(direction)
        assert labels == list(pattern.dim_labels)[: len(labels)]
        assert drift["magnitude"] == pytest.approx(
            sum(v * v for v in direction.values()) ** 0.5, abs=1e-5
        )
