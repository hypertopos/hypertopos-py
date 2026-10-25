# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Anchor edge-dim aggregates are pinned for single, pair and chain anchors."""

from __future__ import annotations

import hashlib
import random

import pyarrow as pa
import pytest

from hypertopos.engine.edge_features import aggregate_edge_dims_for_anchor


def _inputs(seed: int = 8):
    rng = random.Random(seed)
    accounts = [f"A{i:03d}" for i in range(120)]
    n = 3000
    fk = [rng.choice(accounts) for _ in range(n)]
    tk = [rng.choice(accounts) for _ in range(n)]
    eks = [f"E{i}" for i in range(n)]
    edges = pa.table(
        {
            "from_key": fk,
            "to_key": tk,
            "event_key": eks,
            "timestamp": [float(rng.randrange(10_000)) for _ in range(n)],
            "amount": [float(rng.randrange(1, 9_000)) for _ in range(n)],
        }
    )
    sidecar = pa.table(
        {
            "event_key": eks,
            "pair_edge_count": pa.array(
                [float(rng.randrange(1, 9)) for _ in range(n)], pa.float32()
            ),
            "find_motif_structuring": pa.array(
                [rng.choice((0.0, 0.0, 1.0)) for _ in range(n)], pa.float32()
            ),
        }
    )
    anchors = accounts + ["NO-EDGES"]
    pairs = sorted({f"{f}→{t}" for f, t in zip(fk, tk, strict=True)})[:200] + ["X→Y"]
    chains = [f"C{i}" for i in range(80)]
    chain_events = [",".join(rng.sample(eks, rng.randrange(0, 6))) for _ in chains]
    return edges, sidecar, anchors, pairs, chains, chain_events


def _fingerprint(table: pa.Table) -> str:
    rows = [table[c].to_pylist() for c in table.column_names]
    return hashlib.sha256(repr((table.column_names, rows)).encode()).hexdigest()


_DIMS = ["pair_edge_count", "find_motif_structuring"]
# Recorded from the implementation that mapped anchors with string searchsorted.
_EXPECTED = {
    "single": "c7a1277713b87743c1a2b0578934ed5efe89dd4a9ddb6d6fc2423e4b9dff71ed",
    "pair": "de5632f4525e9218a4e92eca9e19aba9d4a587fb1f5e0f527de81a3d87e3e741",
    "chain": "25c36feaf59602c6b2defbb527af5731e6862d495bab0918ca50c9bb41d5d107",
}


def _run(kind: str) -> pa.Table:
    edges, sidecar, anchors, pairs, chains, chain_events = _inputs()
    if kind == "single":
        return aggregate_edge_dims_for_anchor(
            anchor_keys=anchors, edges=edges, sidecar=sidecar, dims=_DIMS, anchor_kind="single"
        )
    if kind == "pair":
        return aggregate_edge_dims_for_anchor(
            anchor_keys=pairs, edges=edges, sidecar=sidecar, dims=_DIMS, anchor_kind="pair"
        )
    return aggregate_edge_dims_for_anchor(
        anchor_keys=chains,
        edges=edges,
        sidecar=sidecar,
        dims=_DIMS,
        anchor_kind="chain",
        chain_events=chain_events,
    )


@pytest.mark.parametrize("kind", sorted(_EXPECTED))
def test_anchor_aggregates_are_unchanged(kind: str) -> None:
    assert _fingerprint(_run(kind)) == _EXPECTED[kind]
