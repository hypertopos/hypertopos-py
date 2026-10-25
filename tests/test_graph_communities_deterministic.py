# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Community detection gives the same partition on every run."""

from __future__ import annotations

import random

import pytest

igraph = pytest.importorskip("igraph")

from hypertopos.engine.adjacency import AdjacencyIndex  # noqa: E402
from hypertopos.engine.graph import compute_louvain_community  # noqa: E402
from hypertopos.engine.graph_algorithms import compute_all_from_lists  # noqa: E402


def _edges(seed: int = 4) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    nodes = [f"N{i}" for i in range(3000)]
    fk = [rng.choice(nodes) for _ in range(6000)]
    tk = [rng.choice(nodes) for _ in range(6000)]
    return fk, tk


def test_label_propagation_communities_repeat() -> None:
    fk, tk = _edges()
    runs = [compute_all_from_lists(fk, tk, {"community"})["community"] for _ in range(4)]
    assert all(r == runs[0] for r in runs[1:])


def test_louvain_communities_repeat() -> None:
    fk, tk = _edges()
    n = len(fk)
    adj = AdjacencyIndex.from_edge_lists(fk, tk, [0.0] * n, [0.0] * n, [f"e{i}" for i in range(n)])
    runs = [compute_louvain_community(adj) for _ in range(4)]
    assert all(r == runs[0] for r in runs[1:])


def test_global_random_state_is_left_alone() -> None:
    fk, tk = _edges()
    random.seed(123)
    expected = random.random()
    random.seed(123)
    compute_all_from_lists(fk, tk, {"community"})
    assert random.random() == expected
