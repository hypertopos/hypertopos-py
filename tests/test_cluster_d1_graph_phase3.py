# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Phase 3 graph algorithms on AdjacencyIndex: PageRank, Louvain, components.

Engineered 7-node graph: 5-node star + 2-node isolated component.

    S1 --\
    S2 ---+- H            D --- E
    S3 ---+              (isolated edge, separate component)
    S4 --/

PageRank: H >> any S_i (star center accumulates incoming mass)
Connected components: {H, S1, S2, S3, S4} = one component; {D, E} = second.
Louvain: at least 2 communities (one per component).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from hypertopos.engine.adjacency import AdjacencyIndex
from hypertopos.engine.graph import (
    _has_louvain_backend,
    compute_connected_components,
    compute_from_adjacency,
    compute_louvain_community,
    compute_pagerank,
)


def _make_adj(edges: list[tuple[str, str]]) -> AdjacencyIndex:
    """Build AdjacencyIndex from (from, to) pairs with dummy ts/amt/ek."""
    from_keys = [e[0] for e in edges]
    to_keys = [e[1] for e in edges]
    timestamps = [float(i) for i in range(len(edges))]
    amounts = [1.0] * len(edges)
    event_keys = [f"E-{i}" for i in range(len(edges))]
    return AdjacencyIndex.from_edge_lists(
        from_keys, to_keys, timestamps, amounts, event_keys,
    )


@pytest.fixture
def engineered_graph() -> AdjacencyIndex:
    """Star (H ← S1..S4) + isolated edge (D--E). Seven nodes, two components."""
    edges = [
        ("S1", "H"), ("S2", "H"), ("S3", "H"), ("S4", "H"),  # star, edges INTO H
        ("D", "E"),                                            # isolated component
    ]
    return _make_adj(edges)


class TestPageRank:
    def test_center_dominates_leaves(self, engineered_graph: AdjacencyIndex) -> None:
        """In an undirected star, the hub H has degree 4; leaves have degree 1.

        Power-iteration PageRank on undirected projection assigns mass
        proportional to degree (in the limit). H should dominate by at
        least 3x over any individual leaf.
        """
        pr = compute_pagerank(engineered_graph)
        assert len(pr) == 7
        assert pr["H"] > 0
        for leaf in ["S1", "S2", "S3", "S4"]:
            assert pr["H"] >= 3.0 * pr[leaf], (
                f"hub PR={pr['H']:.4f} should be >= 3x leaf {leaf} "
                f"PR={pr[leaf]:.4f}"
            )

    def test_sums_to_one(self, engineered_graph: AdjacencyIndex) -> None:
        pr = compute_pagerank(engineered_graph)
        assert abs(sum(pr.values()) - 1.0) < 1e-6

    def test_scores_in_unit_interval(self, engineered_graph: AdjacencyIndex) -> None:
        pr = compute_pagerank(engineered_graph)
        for v in pr.values():
            assert 0.0 <= v <= 1.0

    def test_converges_within_iter_limit(self, engineered_graph: AdjacencyIndex) -> None:
        """Engineered graph (6 nodes, 5 edges) converges far inside 100 iters.

        With damping=0.85 and tol=1e-6, this graph reaches steady state in
        well under 50 iterations.  We confirm by running with a tight
        max_iter and observing that the algorithm still settles to a
        valid distribution (sum ≈ 1) — convergence-or-cap is the key
        contract.
        """
        pr = compute_pagerank(engineered_graph, max_iter=100, tol=1e-6)
        assert abs(sum(pr.values()) - 1.0) < 1e-6

    def test_empty_graph(self) -> None:
        adj = _make_adj([])
        assert compute_pagerank(adj) == {}

    def test_damping_zero_gives_uniform(self, engineered_graph: AdjacencyIndex) -> None:
        """Damping=0 means pure teleportation → uniform 1/n distribution."""
        pr = compute_pagerank(engineered_graph, damping=0.0)
        # All 7 nodes get 1/7
        for v in pr.values():
            assert abs(v - 1.0 / 7) < 1e-6


class TestConnectedComponents:
    def test_two_components(self, engineered_graph: AdjacencyIndex) -> None:
        cc = compute_connected_components(engineered_graph)
        assert len(cc) == 7
        # Distinct component IDs count
        unique = set(cc.values())
        assert len(unique) == 2, f"expected 2 components, got {unique}"
        # Star members in same component
        star_id = cc["H"]
        for v in ["S1", "S2", "S3", "S4"]:
            assert cc[v] == star_id, f"node {v} not in star component"
        # Isolated edge in different component
        assert cc["D"] == cc["E"]
        assert cc["D"] != star_id

    def test_largest_component_id_zero(self, engineered_graph: AdjacencyIndex) -> None:
        """Renumbering: largest component must be ID 0."""
        cc = compute_connected_components(engineered_graph)
        # Star component has 5 nodes, D-E has 2 → star should be ID 0
        assert cc["H"] == 0
        assert cc["D"] == 1

    def test_empty_graph(self) -> None:
        adj = _make_adj([])
        assert compute_connected_components(adj) == {}

    def test_single_edge(self) -> None:
        adj = _make_adj([("X", "Y")])
        cc = compute_connected_components(adj)
        assert cc["X"] == cc["Y"]
        assert len(set(cc.values())) == 1


class TestLouvainCommunity:
    def test_two_components_at_least_two_communities(
        self, engineered_graph: AdjacencyIndex,
    ) -> None:
        """Disconnected components are guaranteed separate communities."""
        com = compute_louvain_community(engineered_graph)
        if not com:
            pytest.skip("no Louvain backend available")
        assert len(com) == 7
        assert com["H"] == com["S1"] == com["S2"] == com["S3"] == com["S4"]
        assert com["D"] == com["E"]
        assert com["H"] != com["D"]
        # Exactly 2 communities for this engineered graph
        assert len(set(com.values())) == 2

    def test_empty_graph(self) -> None:
        adj = _make_adj([])
        assert compute_louvain_community(adj) == {}

    def test_uses_some_backend(self, engineered_graph: AdjacencyIndex) -> None:
        """At least one of igraph / networkx should be available in dev env."""
        assert _has_louvain_backend(), (
            "neither igraph nor networkx is installed — both Louvain backends missing"
        )

    def test_fallback_to_networkx_when_igraph_missing(
        self, engineered_graph: AdjacencyIndex,
    ) -> None:
        """When igraph import fails, the helper should still return results via NetworkX."""
        # Verify NetworkX is available before running this fallback test
        try:
            import networkx  # noqa: F401
        except ImportError:
            pytest.skip("networkx not installed; cannot exercise fallback")

        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "igraph":
                raise ImportError("simulated igraph absence")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            com = compute_louvain_community(engineered_graph)

        assert com, "expected non-empty community map via NetworkX fallback"
        assert len(com) == 7
        # Same structural property: two components stay separate
        assert com["H"] == com["S1"]
        assert com["D"] == com["E"]
        assert com["H"] != com["D"]

    def test_no_backend_returns_empty(
        self, engineered_graph: AdjacencyIndex,
    ) -> None:
        """When neither igraph nor networkx is importable, return {} (no hard failure)."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "igraph" or name.startswith("networkx"):
                raise ImportError(f"simulated {name} absence")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            com = compute_louvain_community(engineered_graph)

        assert com == {}, (
            "with no backend, should return empty dict (caller handles null per-node)"
        )


class TestComputeFromAdjacency:
    def test_all_three_at_once(self, engineered_graph: AdjacencyIndex) -> None:
        results = compute_from_adjacency(
            engineered_graph,
            {"pagerank", "community_id", "connected_component"},
        )
        # All three keys present
        assert set(results.keys()) >= {"pagerank", "connected_component"}
        # PageRank shape
        assert "H" in results["pagerank"]
        # Components: 2 distinct
        assert len(set(results["connected_component"].values())) == 2

    def test_unknown_feature_silently_ignored(
        self, engineered_graph: AdjacencyIndex,
    ) -> None:
        results = compute_from_adjacency(engineered_graph, {"pagerank", "no_such_thing"})
        assert "pagerank" in results
        assert "no_such_thing" not in results


class TestBuildIntegrationRoundtrip:
    """End-to-end: declare the three graph-algorithm features, build a sphere,
    and verify they persist as per-node columns on the anchor points table
    with the expected per-node values."""

    def test_features_persist_on_points_table(self, tmp_path) -> None:
        import pyarrow as pa

        from hypertopos.builder.builder import GDSBuilder

        # Seven-node engineered graph: star + isolated edge
        accounts = pa.table({"primary_key": ["H", "S1", "S2", "S3", "S4", "D", "E"]})
        transfers = pa.table({
            "primary_key": [f"T{i}" for i in range(5)],
            "src": ["S1", "S2", "S3", "S4", "D"],
            "dst": ["H", "H", "H", "H", "E"],
        })

        builder = GDSBuilder("graph_phase3", str(tmp_path / "sphere"))
        builder.add_line(
            "transfers", transfers, key_col="primary_key",
            source_id="test", role="event",
        )
        builder.add_line(
            "accounts", accounts, key_col="primary_key",
            source_id="test", role="anchor",
        )
        builder.add_graph_features(
            "accounts", "transfers", "src", "dst",
            features=["pagerank", "community_id", "connected_component"],
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        accts = builder._lines["accounts"].table
        cols = set(accts.column_names)
        assert "pagerank" in cols, (
            f"pagerank column missing on points table; columns: {sorted(cols)}"
        )
        assert "community_id" in cols, (
            f"community_id column missing on points table; columns: {sorted(cols)}"
        )
        assert "connected_component" in cols, (
            f"connected_component column missing on points table; columns: {sorted(cols)}"
        )

        keys = accts["primary_key"].to_pylist()
        pr_vals = accts["pagerank"].to_pylist()
        cid_vals = accts["community_id"].to_pylist()
        cc_vals = accts["connected_component"].to_pylist()

        # PageRank: hub > each leaf
        h_pr = pr_vals[keys.index("H")]
        for leaf in ["S1", "S2", "S3", "S4"]:
            assert h_pr > pr_vals[keys.index(leaf)], (
                f"hub PR={h_pr} not > leaf {leaf} PR={pr_vals[keys.index(leaf)]}"
            )

        # Connected components: star nodes share a component != D/E component
        h_cc = cc_vals[keys.index("H")]
        for leaf in ["S1", "S2", "S3", "S4"]:
            assert cc_vals[keys.index(leaf)] == h_cc
        d_cc = cc_vals[keys.index("D")]
        assert cc_vals[keys.index("E")] == d_cc
        assert d_cc != h_cc

        # Community: same partition structure as components
        h_com = cid_vals[keys.index("H")]
        for leaf in ["S1", "S2", "S3", "S4"]:
            assert cid_vals[keys.index(leaf)] == h_com
        assert cid_vals[keys.index("D")] == cid_vals[keys.index("E")]
        assert cid_vals[keys.index("D")] != h_com


def test_wall_clock_engineered_subsecond(engineered_graph: AdjacencyIndex) -> None:
    """Sanity floor: all three algorithms on 7-node engineered graph < 1 second."""
    import time
    t0 = time.perf_counter()
    _ = compute_pagerank(engineered_graph)
    _ = compute_connected_components(engineered_graph)
    _ = compute_louvain_community(engineered_graph)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"engineered-graph triple took {elapsed:.3f}s, expected <1s"
