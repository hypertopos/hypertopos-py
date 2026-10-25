# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Pure-numpy / library-agnostic graph algorithms on AdjacencyIndex.

Three primitives suitable for build-time per-node feature columns:

* ``compute_pagerank``        — power-iteration PageRank (damping 0.85,
  max 100 iters, L2 tolerance 1e-6). Pure-numpy, no external dependency.
* ``compute_louvain_community`` — Louvain modularity community detection.
  Uses ``igraph`` if installed; falls back to ``networkx.community.louvain_communities``;
  returns ``{}`` if neither is available (caller must handle missing values).
* ``compute_connected_components`` — undirected weakly-connected components
  via Union-Find. Pure-numpy, no external dependency.

All three operate on an :class:`AdjacencyIndex` and return ``dict[str, T]``
mapping node key to per-node value, where ``T`` is ``float`` for PageRank
and ``int`` for community / component IDs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

    from hypertopos.engine.adjacency import AdjacencyIndex


def _edge_arrays(adj: AdjacencyIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract undirected edge endpoint arrays + node-name table from adj.

    Returns ``(nodes, src_idx, dst_idx)`` where ``nodes`` is the sorted
    array of unique node keys, and ``src_idx``/``dst_idx`` are int64
    arrays of edge endpoint positions into ``nodes``. Self-loops are
    dropped (they don't affect PageRank stationary distribution or
    component / community structure on simple graphs).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    # Every edge endpoint is an out- or in-key of the index, so the node table
    # is the index's own node set.
    nodes = np.asarray(sorted(adj.all_nodes()), dtype=object)
    src, dst = adj.all_edge_endpoints()
    keep = pc.not_equal(src, dst)
    node_table = pa.array(nodes.tolist(), type=pa.string())

    def _positions(keys: pa.Array) -> np.ndarray:
        found = pc.index_in(pc.filter(keys, keep), value_set=node_table)
        return np.asarray(found.to_numpy(zero_copy_only=False), dtype=np.int64)

    return nodes, _positions(src), _positions(dst)


def edge_arrays_from_columns(
    from_keys: pa.Array | pa.ChunkedArray,
    to_keys: pa.Array | pa.ChunkedArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`_edge_arrays` straight from two non-null key columns.

    Returns ``(nodes, src_idx, dst_idx)``: the sorted node keys (every key on
    either side) and the self-loop-free edges in input order, so the result
    is the same on every run.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    def _flat(keys: pa.Array | pa.ChunkedArray) -> pa.Array:
        arr = keys.combine_chunks() if isinstance(keys, pa.ChunkedArray) else keys
        return arr.cast(pa.string())

    from_arr, to_arr = _flat(from_keys), _flat(to_keys)
    m = len(from_arr)
    encoded = pc.dictionary_encode(pa.concat_arrays([from_arr, to_arr]))
    ids = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
    src_raw, dst_raw = ids[:m], ids[m:]
    keep = src_raw != dst_raw
    src_raw, dst_raw = src_raw[keep], dst_raw[keep]
    sorted_ids = np.asarray(pc.sort_indices(encoded.dictionary).to_numpy(), dtype=np.int64)
    rank = np.empty(len(sorted_ids), dtype=np.int64)
    rank[sorted_ids] = np.arange(len(sorted_ids), dtype=np.int64)
    nodes = np.asarray(encoded.dictionary.take(sorted_ids).to_pylist(), dtype=object)
    return nodes, rank[src_raw], rank[dst_raw]


def compute_pagerank(
    adj: AdjacencyIndex,
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Damped PageRank on the undirected projection of ``adj`` (see :func:`_pagerank`)."""
    return _pagerank(*_edge_arrays(adj), damping=damping, max_iter=max_iter, tol=tol)


def _pagerank(
    nodes: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Damped PageRank on the undirected projection of ``adj``.

    Power-iteration implementation in pure numpy. Self-loops are dropped;
    multi-edges between the same pair contribute multiplicatively to that
    pair's transition mass (matches the standard "edge presence" semantic
    for weighted PR on multigraphs).

    Args:
        nodes, src_idx, dst_idx: sorted node keys and self-loop-free edges.
        damping: Teleportation parameter ``alpha`` (default 0.85).
        max_iter: Maximum power-iteration steps (default 100).
        tol: L2-norm convergence threshold on per-iteration delta
            (default 1e-6).

    Returns:
        Mapping from node key to PageRank score in ``[0, 1]``, normalized
        to sum to ``1`` across all nodes. Empty dict for an empty graph.
    """
    n = len(nodes)
    if n == 0:
        return {}

    # Out-degree in the undirected projection: count incident edges per node
    out_deg = np.zeros(n, dtype=np.float64)
    np.add.at(out_deg, src_idx, 1.0)
    np.add.at(out_deg, dst_idx, 1.0)

    # Dangling mass (nodes with no incident edge) is redistributed uniformly
    dangling = out_deg == 0
    # Avoid division by zero; we treat dangling specially in the loop
    inv_out = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, out_deg))

    pr = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        # Contribution from each edge endpoint (undirected: src->dst AND dst->src)
        contrib_src = pr * inv_out
        new_pr = np.full(n, teleport, dtype=np.float64)
        # Each undirected edge contributes mass in both directions
        np.add.at(new_pr, dst_idx, damping * contrib_src[src_idx])
        np.add.at(new_pr, src_idx, damping * contrib_src[dst_idx])
        # Dangling node mass redistributed uniformly
        dangling_mass = damping * pr[dangling].sum() / n
        new_pr += dangling_mass

        delta = float(np.linalg.norm(new_pr - pr))
        pr = new_pr
        if delta < tol:
            break

    # Final renormalize to sum=1 (drift may be at numeric precision level)
    s = pr.sum()
    if s > 0:
        pr = pr / s

    return {nodes[i]: float(pr[i]) for i in range(n)}


def compute_louvain_community(adj: AdjacencyIndex) -> dict[str, int]:
    """Louvain communities of ``adj`` (see :func:`_louvain_community`)."""
    return _louvain_community(*_edge_arrays(adj))


def _louvain_community(
    nodes: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray
) -> dict[str, int]:
    """Louvain modularity-optimizing community detection.

    Tries ``igraph.Graph.community_multilevel`` first (fast C
    implementation, the canonical Louvain). If ``igraph`` is missing,
    falls back to ``networkx.community.louvain_communities``. If neither
    library is installed, returns an empty dict — callers should treat
    a missing node as ``None`` (no community assignment) and surface
    nulls at the column boundary.

    Community IDs are renumbered so the largest community is ``0``,
    second-largest ``1``, etc. — deterministic across runs given the
    same input graph.
    """
    n = len(nodes)
    if n == 0:
        return {}

    edges = list(zip(src_idx.tolist(), dst_idx.tolist(), strict=True))

    membership: list[int] | None = None

    # Try igraph (Louvain = community_multilevel)
    try:
        import igraph as ig

        from hypertopos.engine.graph_algorithms import seeded_igraph_rng

        G = ig.Graph(n=n, edges=edges, directed=False)
        G.simplify()
        with seeded_igraph_rng():
            membership = G.community_multilevel().membership
    except ImportError:
        pass

    # Fall back to NetworkX
    if membership is None:
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities

            G_nx = nx.Graph()
            G_nx.add_nodes_from(range(n))
            G_nx.add_edges_from(edges)
            communities = louvain_communities(G_nx, seed=42)
            membership = [0] * n
            for cid, members in enumerate(communities):
                for v in members:
                    membership[v] = cid
        except ImportError:
            return {}

    # Renumber: largest community = 0
    from collections import Counter

    counts = Counter(membership)
    rank = {cid: r for r, (cid, _) in enumerate(counts.most_common())}
    return {nodes[i]: int(rank[membership[i]]) for i in range(n)}


def compute_connected_components(adj: AdjacencyIndex) -> dict[str, int]:
    """Connected components of ``adj`` (see :func:`_connected_components`)."""
    return _connected_components(*_edge_arrays(adj))


def _connected_components(
    nodes: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray
) -> dict[str, int]:
    """Undirected weakly-connected components (``scipy.sparse.csgraph``).

    Self-loops are dropped (they don't affect component structure).

    Component IDs are renumbered so the largest component is ``0``,
    second-largest ``1``, etc.; equal sizes rank by their first node in
    sorted node order — deterministic across runs.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(nodes)
    if n == 0:
        return {}

    graph = coo_matrix((np.ones(len(src_idx), dtype=np.int8), (src_idx, dst_idx)), shape=(n, n))
    _, roots = connected_components(graph, directed=False)

    # Renumber: largest component = 0
    from collections import Counter

    counts = Counter(roots.tolist())
    rank_map = {cid: r for r, (cid, _) in enumerate(counts.most_common())}
    return {nodes[i]: int(rank_map[int(roots[i])]) for i in range(n)}


def _has_louvain_backend() -> bool:
    """Return True iff at least one Louvain backend (igraph / networkx) is importable."""
    try:
        import igraph  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from networkx.algorithms.community import louvain_communities  # noqa: F401

        return True
    except ImportError:
        return False


def compute_from_adjacency(
    adj: AdjacencyIndex,
    features: set[str],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Dispatch helper — route requested feature tokens to the helpers above.

    Recognised tokens: ``pagerank``, ``community_id``, ``connected_component``.
    Unknown tokens are silently ignored (caller's responsibility to validate
    the feature set). The edges are extracted once for all requested features.
    """
    return compute_from_edge_arrays(*_edge_arrays(adj), features, **kwargs)


def compute_from_edge_arrays(
    nodes: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    features: set[str],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """:func:`compute_from_adjacency` on prepared ``(nodes, src_idx, dst_idx)`` arrays."""
    results: dict[str, dict[str, Any]] = {}
    if "pagerank" in features:
        results["pagerank"] = _pagerank(
            nodes,
            src_idx,
            dst_idx,
            **{k: v for k, v in kwargs.items() if k in ("damping", "max_iter", "tol")},
        )
    if "community_id" in features:
        results["community_id"] = _louvain_community(nodes, src_idx, dst_idx)
    if "connected_component" in features:
        results["connected_component"] = _connected_components(nodes, src_idx, dst_idx)
    return results
