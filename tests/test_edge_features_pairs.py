# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Pair-grouped edge dimensions against per-pair Python references."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.engine.edge_features import (
    compute_pair_amount_zscore,
    compute_pair_edge_count,
    compute_time_since_pair_last_edge,
)


@pytest.fixture(scope="module")
def edges() -> pa.Table:
    rng = random.Random(21)
    keys = [f"A{i}" for i in range(40)] + ["ż", "Z"]
    n = 3000
    fk = [rng.choice(keys) for _ in range(n)]
    tk = [rng.choice(keys) for _ in range(n)]
    ts = [float(rng.randrange(500)) for _ in range(n)]  # many equal timestamps
    base = {(f, t): rng.choice((100.0, 5000.0)) for f in keys for t in keys}
    amt = [base[(f, t)] + rng.choice((0.0, 1.0, 2.0, 900.0)) for f, t in zip(fk, tk, strict=True)]
    return pa.table(
        {
            "from_key": fk,
            "to_key": tk,
            "event_key": [f"E{i}" for i in range(n)],
            "timestamp": ts,
            "amount": amt,
        }
    )


def _pairs(edges: pa.Table):
    return list(zip(edges["from_key"].to_pylist(), edges["to_key"].to_pylist(), strict=True))


def test_pair_edge_count_counts_each_pair(edges: pa.Table) -> None:
    pairs = _pairs(edges)
    counts = Counter(pairs)
    expected = [float(counts[p]) for p in pairs]
    assert compute_pair_edge_count(edges).to_pylist() == expected


def test_time_since_pair_last_edge_uses_the_previous_edge_of_the_pair(edges: pa.Table) -> None:
    pairs, ts = _pairs(edges), edges["timestamp"].to_pylist()
    by_pair: dict[tuple, list[int]] = defaultdict(list)
    for i in sorted(range(len(pairs)), key=lambda i: (ts[i], i)):
        by_pair[pairs[i]].append(i)
    expected = [0.0] * len(pairs)
    for rows in by_pair.values():
        expected[rows[0]] = 7.0
        for prev, cur in zip(rows, rows[1:], strict=False):
            expected[cur] = ts[cur] - ts[prev]
    out = compute_time_since_pair_last_edge(edges, burst_seconds=60, dormant_seconds=7.0)
    assert out.to_pylist() == expected


def test_pair_amount_zscore_matches_per_pair_statistics(edges: pa.Table) -> None:
    pairs, amt = _pairs(edges), edges["amount"].to_pylist()
    by_pair: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by_pair[p].append(i)
    expected = np.zeros(len(pairs))
    for rows in by_pair.values():
        vals = np.asarray([amt[i] for i in rows])
        mean, std = vals.mean(), vals.std()
        if len(rows) >= 3 and mean != 0 and std != 0 and std / abs(mean) < 0.05:
            expected[rows] = (vals - mean) / std
    out = np.asarray(compute_pair_amount_zscore(edges, cv_threshold=0.05, min_count=3))
    assert (expected != 0).sum() > 20, "fixture must produce scored pairs"
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-4, atol=1e-5)


def test_pair_amount_zscore_ignores_missing_amounts() -> None:
    # The builder stores a missing amount as 0.0; it must not drag the pair's
    # mean / std nor get an "anomalous" z-score itself.
    amounts = [100.0, 101.0, 99.0, 100.0, 0.0, None]
    edges = pa.table(
        {
            "from_key": ["A"] * 6,
            "to_key": ["B"] * 6,
            "event_key": [f"E{i}" for i in range(6)],
            "timestamp": [float(i) for i in range(6)],
            "amount": pa.array(amounts, pa.float64()),
        }
    )
    # cv_threshold=1.0 keeps the pair even when the zero inflates its std.
    z = np.asarray(compute_pair_amount_zscore(edges, cv_threshold=1.0, min_count=3))
    assert z[4] == 0.0 and z[5] == 0.0
    assert np.all(np.abs(z[:4]) < 1.5)
