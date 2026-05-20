# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Hub-safety cap regression tests.

Covers the per-entity caps added to ``simulate_edge_removal`` and
``select_minimal_joint_edge_removal`` that bound per-call latency on hub
entities. Without these caps a single account with hundreds of thousands of
edges pushes per-call wall clock past several minutes; the cap clips the
candidate list to a bounded prefix BEFORE the Lance sidecar IN-clause and
the engine call so the work is constant-bounded regardless of degree.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa

from hypertopos.model.sphere import Pattern, RelationDef, Sphere
from hypertopos.navigation.navigator import GDSNavigator


def _make_plain_anchor(n_relations: int = 2) -> Pattern:
    """Minimal anchor pattern — no aggregation dim tail, no edge features."""
    relations = [
        RelationDef(line_id=f"L{i}", direction="out", required=False)
        for i in range(n_relations)
    ]
    return Pattern(
        pattern_id="P",
        entity_type="ents",
        pattern_type="anchor",
        relations=relations,
        mu=np.zeros(n_relations, dtype=np.float32),
        sigma_diag=np.ones(n_relations, dtype=np.float32),
        theta=np.ones(n_relations, dtype=np.float32),
        edge_max=np.ones(n_relations, dtype=np.float32),
        population_size=10,
        computed_at=datetime.now(UTC),
        version=1,
        status="production",
    )


def _make_navigator(pattern: Pattern) -> GDSNavigator:
    sphere = MagicMock(spec=Sphere)
    sphere.patterns = {pattern.pattern_id: pattern}
    sphere.lines = {}
    sphere.aliases = {}
    sphere.entity_line = MagicMock(return_value=None)

    storage = MagicMock()
    storage.read_sphere = MagicMock(return_value=sphere)
    storage.read_geometry_stats = MagicMock(return_value=None)

    nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())
    nav._resolve_version = MagicMock(return_value=1)  # type: ignore[method-assign]
    return nav


def test_select_minimal_joint_truncates_candidates_to_max_candidates():
    """When the entity's adjacency exceeds ``max_candidates``, the result
    must carry ``candidates_truncated=True``, ``n_candidates_seen`` equal
    to the full adjacency count, and ``n_candidates_used`` equal to the
    cap. Engine receives only the truncated prefix.
    """
    pat = _make_plain_anchor(n_relations=2)
    nav = _make_navigator(pat)
    delta_dim = len(pat.mu)

    # Geometry table — one row for the seed entity with a small finite delta
    delta_vec = np.array([1.0, 2.0], dtype=np.float32)
    geo_table = pa.table({
        "primary_key": ["SEED"],
        "delta": pa.array(
            [delta_vec.tolist()], type=pa.list_(pa.float32()),
        ),
        "delta_norm": pa.array([float(np.linalg.norm(delta_vec))], type=pa.float32()),
    })
    nav._storage.read_geometry = MagicMock(return_value=geo_table)

    # Adjacency for the line — 1500 outbound edges from the seed
    n_total_edges = 1500
    adj = MagicMock()
    adj.neighbors_out = MagicMock(return_value=[
        (f"P{i}", "out", 0, f"EID-{i}") for i in range(n_total_edges)
    ])
    adj.neighbors_in = MagicMock(return_value=[])
    nav._storage.get_adjacency = MagicMock(return_value=adj)

    # Bypass the inner simulate_edge_removal warm-cache call — the joint
    # counterfactual only needs it to populate the threshold cache, which
    # is empty for an aggregation-less pattern.
    nav.simulate_edge_removal = MagicMock(return_value=[])  # type: ignore[method-assign]

    # Stub the engine call to record what it received and return a
    # deterministic minimal joint result.
    from hypertopos.engine import counterfactual as cf_module
    captured: dict[str, object] = {}
    original_engine = cf_module.select_minimal_joint_removal

    def _spy(*args, **kwargs):
        captured["candidate_edges"] = kwargs.get("candidate_edges", args[4] if len(args) > 4 else None)
        return {
            "selected_edge_ids": [],
            "selected_partner_keys": [],
            "achieved_drop_pct": 0.0,
            "achieved_abs_drop_pct": 0.0,
            "selection_sequence": [],
            "target_reached": False,
            "k_max_reached": False,
        }

    cf_module.select_minimal_joint_removal = _spy
    try:
        result = nav.select_minimal_joint_edge_removal(
            "SEED", pattern_id="P", line_id="L0",
            target_drop_pct=50.0, k_max=3, max_candidates=300,
        )
    finally:
        cf_module.select_minimal_joint_removal = original_engine

    assert result["candidates_truncated"] is True
    assert result["n_candidates_seen"] == n_total_edges
    assert result["n_candidates_used"] == 300
    received = captured["candidate_edges"]
    assert received is not None
    assert len(received) == 300, (
        f"engine must receive the truncated prefix (300), got {len(received)}"
    )


def test_select_minimal_joint_no_truncation_when_below_cap():
    """When adjacency fits under the cap, no truncation flag is set and
    the engine receives the full candidate list."""
    pat = _make_plain_anchor(n_relations=2)
    nav = _make_navigator(pat)

    delta_vec = np.array([1.0, 2.0], dtype=np.float32)
    geo_table = pa.table({
        "primary_key": ["SEED"],
        "delta": pa.array(
            [delta_vec.tolist()], type=pa.list_(pa.float32()),
        ),
        "delta_norm": pa.array([float(np.linalg.norm(delta_vec))], type=pa.float32()),
    })
    nav._storage.read_geometry = MagicMock(return_value=geo_table)

    n_total_edges = 20
    adj = MagicMock()
    adj.neighbors_out = MagicMock(return_value=[
        (f"P{i}", "out", 0, f"EID-{i}") for i in range(n_total_edges)
    ])
    adj.neighbors_in = MagicMock(return_value=[])
    nav._storage.get_adjacency = MagicMock(return_value=adj)

    nav.simulate_edge_removal = MagicMock(return_value=[])  # type: ignore[method-assign]

    from hypertopos.engine import counterfactual as cf_module
    captured: dict[str, object] = {}
    original = cf_module.select_minimal_joint_removal

    def _spy(*args, **kwargs):
        captured["candidate_edges"] = kwargs.get("candidate_edges", args[4] if len(args) > 4 else None)
        return {
            "selected_edge_ids": [],
            "selected_partner_keys": [],
            "achieved_drop_pct": 0.0,
            "achieved_abs_drop_pct": 0.0,
            "selection_sequence": [],
            "target_reached": False,
            "k_max_reached": False,
        }

    cf_module.select_minimal_joint_removal = _spy
    try:
        result = nav.select_minimal_joint_edge_removal(
            "SEED", pattern_id="P", line_id="L0",
            target_drop_pct=50.0, k_max=3, max_candidates=500,
        )
    finally:
        cf_module.select_minimal_joint_removal = original

    assert result["candidates_truncated"] is False
    assert result["n_candidates_seen"] == n_total_edges
    assert result["n_candidates_used"] == n_total_edges
    received = captured["candidate_edges"]
    assert received is not None
    assert len(received) == n_total_edges


def test_simulate_edge_removal_max_edges_loaded_truncates_to_prefix():
    """``simulate_edge_removal`` must clip adjacency to ``max_edges_loaded``
    BEFORE the engine evaluation. Engineered with 3000 fallback-path edges
    against a cap of 250 — engine should see exactly 250 candidates."""
    pat = _make_plain_anchor(n_relations=2)
    nav = _make_navigator(pat)

    delta_vec = np.array([1.0, 2.0], dtype=np.float32)
    geo_table = pa.table({
        "primary_key": ["SEED"],
        "delta": pa.array(
            [delta_vec.tolist()], type=pa.list_(pa.float32()),
        ),
        "delta_norm": pa.array([float(np.linalg.norm(delta_vec))], type=pa.float32()),
    })
    nav._storage.read_geometry = MagicMock(return_value=geo_table)

    n_total_edges = 3000
    adj = MagicMock()
    adj.neighbors_out = MagicMock(return_value=[
        (f"P{i}", "out", 0, f"EID-{i}") for i in range(n_total_edges)
    ])
    adj.neighbors_in = MagicMock(return_value=[])
    nav._storage.get_adjacency = MagicMock(return_value=adj)

    # Spy on the engine evaluator — record the edges it received.
    from hypertopos.engine import counterfactual as cf_module
    captured: dict[str, object] = {}
    original = cf_module.simulate_edge_removal_with_aggregations

    def _spy(*args, **kwargs):
        captured["edges"] = kwargs.get("edges", args[4] if len(args) > 4 else None)
        return []

    cf_module.simulate_edge_removal_with_aggregations = _spy
    try:
        nav.simulate_edge_removal(
            "SEED", pattern_id="P", line_id="L0",
            top_n=5, max_edges_loaded=250,
        )
    finally:
        cf_module.simulate_edge_removal_with_aggregations = original

    received = captured["edges"]
    assert received is not None
    assert len(received) == 250, (
        f"engine must see the truncated prefix (250), got {len(received)}"
    )
