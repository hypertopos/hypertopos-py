# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Graph algorithms via igraph C backend.

All heavy lifting in C. Betweenness uses edge-sampled subgraph
to keep build time constant regardless of graph size.
Requires: pip install hypertopos[graph]
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from hypertopos.engine.adjacency import AdjacencyIndex


def _import_igraph() -> Any:
    try:
        import igraph
    except ImportError:
        raise ImportError(
            "igraph is required for graph algorithm dimensions. "
            "Install with: pip install hypertopos[graph]"
        ) from None
    return igraph


@contextlib.contextmanager
def seeded_igraph_rng(seed: int = 42) -> Iterator[None]:
    """Run igraph's randomized algorithms on a private, seeded generator.

    igraph draws from the global ``random`` module by default, so community
    detection changed between runs and moved the caller's random state.
    """
    ig = _import_igraph()
    ig.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        ig.set_random_number_generator(random)


def compute_all_from_lists(
    from_keys: list[str],
    to_keys: list[str],
    features: set[str],
    betweenness_sample_edges: int = 50_000,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Compute all graph algorithms from edge lists via igraph C backend.

    Full graph for cheap algorithms (pagerank, clustering, components, community).
    Edge-sampled subgraph for betweenness (O(V*E) exact on small graph ≈ O(1) on full).
    """
    import pyarrow as pa

    from hypertopos.engine.graph import edge_arrays_from_columns

    nodes, src, dst = edge_arrays_from_columns(
        pa.array(from_keys, type=pa.string()), pa.array(to_keys, type=pa.string())
    )
    return compute_all_from_edge_arrays(
        nodes, src, dst, features, betweenness_sample_edges=betweenness_sample_edges, seed=seed
    )


def compute_all_from_edge_arrays(
    nodes: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    features: set[str],
    betweenness_sample_edges: int = 50_000,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """:func:`compute_all_from_lists` on sorted ``nodes`` and self-loop-free int edges."""
    ig = _import_igraph()
    G = ig.Graph(n=len(nodes), edges=np.column_stack([src_idx, dst_idx]), directed=False)
    G.vs["name"] = nodes.tolist()
    G.simplify()  # remove multi-edges
    names = G.vs["name"]
    n = G.vcount()

    if n == 0:
        return {f: {} for f in features}

    results: dict[str, dict[str, float | int]] = {}

    if "pagerank" in features:
        scores = G.pagerank(directed=False)
        results["pagerank"] = {names[i]: scores[i] for i in range(n)}

    if "connected_component" in features:
        membership = G.connected_components(mode="weak").membership
        # Renumber: largest component = 0
        from collections import Counter

        counts = Counter(membership)
        rank = {cid: rank for rank, (cid, _) in enumerate(counts.most_common())}
        results["connected_component"] = {names[i]: rank[membership[i]] for i in range(n)}

    if "clustering_coefficient" in features:
        cc = G.transitivity_local_undirected(mode="zero")
        results["clustering_coefficient"] = {names[i]: cc[i] for i in range(n)}

    if "community" in features:
        with seeded_igraph_rng(seed):
            membership = G.community_label_propagation().membership
        from collections import Counter

        counts = Counter(membership)
        rank = {cid: r for r, (cid, _) in enumerate(counts.most_common())}
        results["community"] = {names[i]: rank[membership[i]] for i in range(n)}

    if "betweenness" in features:
        n_edges = G.ecount()
        if n_edges <= betweenness_sample_edges:
            # Small graph: exact betweenness
            bc = G.betweenness(directed=False)
        else:
            # Large graph: sample edges, compute on subgraph
            rng = np.random.default_rng(seed)
            edge_idx = rng.choice(n_edges, betweenness_sample_edges, replace=False)
            sampled_edges = [G.es[int(i)].tuple for i in edge_idx]
            ig = _import_igraph()
            G_small = ig.Graph(n=n, edges=sampled_edges, directed=False)
            G_small.simplify()
            bc = G_small.betweenness(directed=False)
        # Normalize: undirected betweenness counts each unordered pair once,
        # so the maximum is (n-1)(n-2)/2 and the result lies in [0, 1].
        norm = (n - 1) * (n - 2) / 2 if n > 2 else 1
        results["betweenness"] = {names[i]: bc[i] / norm for i in range(n)}

    return results


def compute_all(
    adj: AdjacencyIndex,
    features: set[str],
    **kwargs: Any,
) -> dict[str, dict[str, float | int]]:
    """Compute from AdjacencyIndex (convenience wrapper for tests)."""
    from_keys = []
    to_keys = []
    for src, tgt, _ts, _amt, _ek in adj.all_edges():
        from_keys.append(src)
        to_keys.append(tgt)
    return compute_all_from_lists(from_keys, to_keys, features, **kwargs)


# --- Convenience wrappers for tests ---


def pagerank(adj: AdjacencyIndex, **kw: Any) -> dict[str, float]:
    return compute_all(adj, {"pagerank"}, **kw).get("pagerank", {})


def connected_components(adj: AdjacencyIndex) -> dict[str, int]:
    # Component ids are int ranks (see compute_all_from_lists).
    return cast(
        "dict[str, int]",
        compute_all(adj, {"connected_component"}).get("connected_component", {}),
    )


def clustering_coefficient(adj: AdjacencyIndex) -> dict[str, float]:
    return compute_all(adj, {"clustering_coefficient"}).get("clustering_coefficient", {})


def label_propagation(adj: AdjacencyIndex, **kw: Any) -> dict[str, int]:
    # Community ids are int ranks (see compute_all_from_lists).
    return cast("dict[str, int]", compute_all(adj, {"community"}, **kw).get("community", {}))


def betweenness_centrality(adj: AdjacencyIndex, **kw: Any) -> dict[str, float]:
    return compute_all(adj, {"betweenness"}, **kw).get("betweenness", {})
