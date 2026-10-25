# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""_edge_arrays: undirected endpoint arrays for the build-time graph features."""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa

from hypertopos.engine.adjacency import AdjacencyIndex
from hypertopos.engine.graph import _edge_arrays, edge_arrays_from_columns


def _reference(adj: AdjacencyIndex):
    src_list, dst_list = [], []
    for src, dst, _ts, _amt, _ek in adj.all_edges():
        if src != dst:
            src_list.append(src)
            dst_list.append(dst)
    nodes = sorted(set(src_list) | set(dst_list) | adj.all_nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    return nodes, [idx[s] for s in src_list], [idx[d] for d in dst_list]


def _adjacency(n_edges: int, n_nodes: int, seed: int) -> AdjacencyIndex:
    rng = random.Random(seed)
    keys = [f"K{i:03d}" for i in range(n_nodes)] + ["ż-unicode", "Z"]
    fk = [rng.choice(keys) for _ in range(n_edges)]
    tk = [rng.choice(keys) if rng.random() > 0.05 else f for f in fk]  # some self-loops
    ts = [float(rng.randrange(10_000)) for _ in range(n_edges)]
    return AdjacencyIndex.from_edge_lists(
        fk, tk, ts, [1.0] * n_edges, [f"E{i}" for i in range(n_edges)]
    )


def test_edge_arrays_match_the_edge_by_edge_reference() -> None:
    adj = _adjacency(3000, 200, seed=11)
    nodes, src_idx, dst_idx = _edge_arrays(adj)
    ref_nodes, ref_src, ref_dst = _reference(adj)
    assert list(nodes) == ref_nodes
    assert src_idx.dtype == np.int64 and dst_idx.dtype == np.int64
    assert src_idx.tolist() == ref_src
    assert dst_idx.tolist() == ref_dst


def test_only_self_loops_give_nodes_and_no_edges() -> None:
    adj = AdjacencyIndex.from_edge_lists(
        ["B", "A"], ["B", "A"], [1.0, 2.0], [1.0, 1.0], ["e1", "e2"]
    )
    nodes, src_idx, dst_idx = _edge_arrays(adj)
    assert list(nodes) == ["A", "B"]
    assert len(src_idx) == 0 and len(dst_idx) == 0


def test_edge_arrays_from_columns_match_the_adjacency_edges_in_input_order() -> None:
    rng = random.Random(13)
    keys = [f"K{i:03d}" for i in range(150)] + ["ż-unicode", "Z"]
    fk = [rng.choice(keys) for _ in range(2500)]
    tk = [rng.choice(keys) if rng.random() > 0.05 else f for f in fk]
    n = len(fk)
    adj = AdjacencyIndex.from_edge_lists(fk, tk, [0.0] * n, [1.0] * n, [f"E{i}" for i in range(n)])

    nodes, src, dst = edge_arrays_from_columns(pa.array(fk), pa.array(tk))
    ref_nodes, ref_src, ref_dst = _edge_arrays(adj)
    assert list(nodes) == list(ref_nodes)
    assert sorted(zip(src.tolist(), dst.tolist(), strict=True)) == sorted(
        zip(ref_src.tolist(), ref_dst.tolist(), strict=True)
    )
    idx = {k: i for i, k in enumerate(nodes)}
    pairs = [(idx[f], idx[t]) for f, t in zip(fk, tk, strict=True) if f != t]
    assert list(zip(src.tolist(), dst.tolist(), strict=True)) == pairs
