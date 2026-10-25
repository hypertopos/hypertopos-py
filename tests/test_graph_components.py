# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""compute_connected_components: ids by component size, ties by first node."""

from __future__ import annotations

import random
from collections import Counter

from hypertopos.engine.adjacency import AdjacencyIndex
from hypertopos.engine.graph import compute_connected_components


def _reference(adj: AdjacencyIndex) -> dict[str, int]:
    nodes = sorted(adj.all_nodes())
    neighbours: dict[str, set[str]] = {n: set() for n in nodes}
    for src, dst, *_ in adj.all_edges():
        if src != dst:
            neighbours[src].add(dst)
            neighbours[dst].add(src)
    label: dict[str, int] = {}
    for start in nodes:
        if start in label:
            continue
        label[start] = len(set(label.values()))
        stack = [start]
        while stack:
            for nb in neighbours[stack.pop()]:
                if nb not in label:
                    label[nb] = label[start]
                    stack.append(nb)
    counts = Counter(label[n] for n in nodes)
    rank = {cid: r for r, (cid, _) in enumerate(counts.most_common())}
    return {n: rank[label[n]] for n in nodes}


def test_components_match_a_bfs_reference() -> None:
    rng = random.Random(9)
    keys = [f"N{i:03d}" for i in range(400)]
    fk = [rng.choice(keys) for _ in range(330)]
    tk = [rng.choice(keys) if rng.random() > 0.1 else f for f in fk]
    adj = AdjacencyIndex.from_edge_lists(
        fk, tk, [0.0] * len(fk), [1.0] * len(fk), [f"E{i}" for i in range(len(fk))]
    )
    assert compute_connected_components(adj) == _reference(adj)


def test_equal_sized_components_rank_by_first_node() -> None:
    adj = AdjacencyIndex.from_edge_lists(
        ["C", "A", "E"], ["D", "B", "E"], [0.0] * 3, [1.0] * 3, ["e1", "e2", "e3"]
    )
    assert compute_connected_components(adj) == {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2}
