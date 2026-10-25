# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""scatter_columns: per-event values into entity rows, values kept as they are."""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa

from hypertopos.builder._scatter import scatter_columns, vectorized_scatter, vectorized_scatter_1d


def test_scatter_columns_matches_the_row_by_row_assignment() -> None:
    rng = random.Random(5)
    entity_keys = [f"E{i}" for i in range(300)]
    rng.shuffle(entity_keys)
    event_keys = [f"E{rng.randrange(400)}" for _ in range(250)]  # some keys are not entities
    columns = {
        "a": [rng.random() for _ in event_keys],
        "b": [None if rng.random() < 0.1 else float(rng.randrange(9)) for _ in event_keys],
    }
    features = pa.table({"event_key": event_keys, **columns})

    expected = np.zeros((len(entity_keys), 2), dtype=np.float32)
    pk_to_idx = {pk: i for i, pk in enumerate(entity_keys)}
    for col_idx, name in enumerate(("a", "b")):
        vals = features[name].to_numpy()
        for row_idx, ek in enumerate(event_keys):
            ent_idx = pk_to_idx.get(ek)
            if ent_idx is not None:
                expected[ent_idx, col_idx] = vals[row_idx]

    out = np.zeros((len(entity_keys), 2), dtype=np.float32)
    scatter_columns(
        out, pa.array(entity_keys), features["event_key"], [features["a"], features["b"]]
    )
    np.testing.assert_array_equal(out, expected)
    assert np.isnan(out).any(), "null values must land as NaN, like the row-by-row loop"


def test_scatters_accept_precomputed_anchor_positions() -> None:
    import pyarrow.compute as pc

    rng = random.Random(9)
    anchors = pa.array([f"K{i}" for i in range(50)])
    keys = pa.array([f"K{rng.randrange(60)}" for _ in range(400)])
    buckets = pa.array([rng.randrange(4) for _ in range(400)])
    values = pa.array([None if rng.random() < 0.1 else rng.random() * 10 for _ in range(400)])
    positions = pc.index_in(keys, anchors)

    a = np.zeros(50)
    b = np.zeros(50)
    vectorized_scatter_1d(a, anchors, keys, values)
    vectorized_scatter_1d(b, anchors, keys, values, entity_idx=positions)
    np.testing.assert_array_equal(a, b)

    ta = np.zeros((50, 4, 2), dtype=np.float32)
    tb = np.zeros((50, 4, 2), dtype=np.float32)
    vectorized_scatter(ta, 1, 5.0, anchors, keys, buckets, values)
    vectorized_scatter(tb, 1, 5.0, anchors, keys, buckets, values, entity_idx=positions)
    np.testing.assert_array_equal(ta, tb)
