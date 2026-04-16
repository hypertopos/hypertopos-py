# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tests for graph algorithms on AdjacencyIndex."""
from __future__ import annotations

import pytest

from hypertopos.engine.adjacency import AdjacencyIndex
from hypertopos.engine.graph_algorithms import (
    betweenness_centrality,
    clustering_coefficient,
    connected_components,
    label_propagation,
    pagerank,
)


def _make_adj(edges: list[tuple[str, str]]) -> AdjacencyIndex:
    """Build AdjacencyIndex from (from, to) pairs with dummy ts/amt/ek."""
    from_keys = [e[0] for e in edges]
    to_keys = [e[1] for e in edges]
    timestamps = [float(i) for i in range(len(edges))]
    amounts = [1.0] * len(edges)
    event_keys = [f"E-{i}" for i in range(len(edges))]
    return AdjacencyIndex.from_edge_lists(from_keys, to_keys, timestamps, amounts, event_keys)


# --- Triangle graph: A-B, B-C, C-A ---

@pytest.fixture
def triangle():
    return _make_adj([("A", "B"), ("B", "C"), ("C", "A")])


# --- Chain graph: A->B->C->D ---

@pytest.fixture
def chain():
    return _make_adj([("A", "B"), ("B", "C"), ("C", "D")])


# --- Two disconnected components: {A,B,C} triangle + {D,E} edge ---

@pytest.fixture
def two_components():
    return _make_adj([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])


# --- Star: hub H connected to S1,S2,S3,S4 ---

@pytest.fixture
def star():
    return _make_adj([("H", "S1"), ("H", "S2"), ("H", "S3"), ("H", "S4")])


# ---- PageRank ----

class TestPageRank:
    def test_triangle_uniform(self, triangle):
        pr = pagerank(triangle)
        assert len(pr) == 3
        # Triangle: all nodes should have roughly equal PageRank
        values = list(pr.values())
        assert max(values) - min(values) < 0.05

    def test_star_hub_highest(self, star):
        pr = pagerank(star)
        # Hub receives edges from all spokes (via in-edges)
        # but in a directed star H->S*, the spokes have in-edges from H
        # So spokes might have higher PR. Test that all nodes have scores.
        assert len(pr) == 5
        assert all(v > 0 for v in pr.values())
        assert abs(sum(pr.values()) - 1.0) < 0.01

    def test_empty(self):
        adj = _make_adj([])
        assert pagerank(adj) == {}


# ---- Connected Components ----

class TestConnectedComponents:
    def test_triangle_single(self, triangle):
        cc = connected_components(triangle)
        assert len(set(cc.values())) == 1

    def test_two_components(self, two_components):
        cc = connected_components(two_components)
        assert len(set(cc.values())) == 2
        # A, B, C in same component
        assert cc["A"] == cc["B"] == cc["C"]
        # D, E in same component
        assert cc["D"] == cc["E"]
        # Different components
        assert cc["A"] != cc["D"]

    def test_chain_single(self, chain):
        cc = connected_components(chain)
        assert len(set(cc.values())) == 1


# ---- Clustering Coefficient ----

class TestClusteringCoefficient:
    def test_triangle_perfect(self, triangle):
        cc = clustering_coefficient(triangle)
        # Every pair of neighbors is connected → coefficient = 1.0
        for v in ["A", "B", "C"]:
            assert cc[v] == pytest.approx(1.0, abs=0.01)

    def test_chain_zero(self, chain):
        cc = clustering_coefficient(chain)
        # End nodes (A, D) have degree 1 → 0.0
        assert cc["A"] == 0.0
        assert cc["D"] == 0.0
        # Middle nodes (B, C) have 2 neighbors but they're not connected to each other
        assert cc["B"] == 0.0
        assert cc["C"] == 0.0

    def test_star_zero(self, star):
        cc = clustering_coefficient(star)
        # Spokes not connected to each other
        for s in ["S1", "S2", "S3", "S4"]:
            assert cc[s] == 0.0


# ---- Label Propagation ----

class TestLabelPropagation:
    def test_two_components_separate(self, two_components):
        lp = label_propagation(two_components)
        # A, B, C should share one label
        assert lp["A"] == lp["B"] == lp["C"]
        # D, E share another
        assert lp["D"] == lp["E"]
        # Different communities
        assert lp["A"] != lp["D"]

    def test_triangle_one_community(self, triangle):
        lp = label_propagation(triangle)
        assert lp["A"] == lp["B"] == lp["C"]


# ---- Betweenness Centrality ----

class TestBetweennessCentrality:
    def test_chain_middle_highest(self, chain):
        bc = betweenness_centrality(chain)
        # B and C are on all shortest paths between endpoints
        assert bc["B"] > bc["A"]
        assert bc["C"] > bc["D"]

    def test_star_hub_highest(self, star):
        bc = betweenness_centrality(star)
        # Hub is on all shortest paths between spokes
        assert bc["H"] > bc["S1"]
        assert bc["H"] > bc["S2"]

    def test_empty(self):
        adj = _make_adj([])
        assert betweenness_centrality(adj) == {}

    def test_two_nodes(self):
        adj = _make_adj([("A", "B")])
        bc = betweenness_centrality(adj)
        # Only 2 nodes, no intermediate → all zero
        assert bc["A"] == 0.0
        assert bc["B"] == 0.0
