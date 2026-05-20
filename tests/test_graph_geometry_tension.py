# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for find_graph_geometry_tension — kNN×adjacency 2×2 cross-tab.

Hidden cluster = behavioural k-NN entities WITHOUT a graph edge to anchor.
Suspicious link = entities with edges to anchor that are NOT in the behavioural k-NN.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from hypertopos.navigation.navigator import SimilarityResult


def _make_similarity_result(neighbors: list[tuple[str, float]]) -> SimilarityResult:
    """SimilarityResult is a list[tuple[primary_key, distance]] subclass."""
    return SimilarityResult(neighbors)


class _FakeAdj:
    def __init__(self, out_edges: dict[str, list[str]], in_edges: dict[str, list[str]]):
        self._out = out_edges
        self._in = in_edges

    def neighbors_out(self, key, ts_from=None, ts_to=None):
        return [(t, 0.0, 0.0, "") for t in self._out.get(key, [])]

    def neighbors_in(self, key, ts_from=None, ts_to=None):
        return [(t, 0.0, 0.0, "") for t in self._in.get(key, [])]


@pytest.fixture
def patched_nav():
    """Build a GDSNavigator with patched find_similar_entities + adjacency."""
    from hypertopos.navigation.navigator import GDSNavigator
    nav = GDSNavigator.__new__(GDSNavigator)
    return nav


def test_hidden_cluster_and_suspicious_link_split(patched_nav):
    similarity = _make_similarity_result([
        ("BEHAV_AND_EDGE", 0.1),
        ("BEHAV_NO_EDGE_1", 0.2),
        ("BEHAV_NO_EDGE_2", 0.3),
    ])
    adj = _FakeAdj(
        out_edges={"ANCHOR": ["BEHAV_AND_EDGE", "EDGE_NOT_BEHAV"]},
        in_edges={"ANCHOR": []},
    )
    with patch.object(patched_nav, "find_similar_entities", return_value=similarity), \
         patch.object(patched_nav, "_storage", create=True) as mock_storage:
        mock_storage.get_adjacency.return_value = adj
        result = patched_nav.find_graph_geometry_tension(
            "ANCHOR",
            pattern_id="account_pattern",
            line_id="accounts",
            k_geometric=3,
        )

    assert result["primary_key"] == "ANCHOR"
    hidden_keys = {r["neighbor_key"] for r in result["hidden_cluster"]}
    suspicious_keys = {r["neighbor_key"] for r in result["suspicious_links"]}

    assert hidden_keys == {"BEHAV_NO_EDGE_1", "BEHAV_NO_EDGE_2"}
    assert suspicious_keys == {"EDGE_NOT_BEHAV"}
    assert all(r["edge_present"] is False for r in result["hidden_cluster"])
    assert all(r["edge_present"] is True for r in result["suspicious_links"])

    expected_tension = (2 + 1) / 3
    assert result["tension_score"] == pytest.approx(expected_tension, rel=1e-6)


def test_all_behav_neighbors_have_edges_zero_hidden(patched_nav):
    similarity = _make_similarity_result([
        ("E1", 0.1), ("E2", 0.2), ("E3", 0.3),
    ])
    adj = _FakeAdj(
        out_edges={"A": ["E1", "E2", "E3"]},
        in_edges={"A": []},
    )
    with patch.object(patched_nav, "find_similar_entities", return_value=similarity), \
         patch.object(patched_nav, "_storage", create=True) as mock_storage:
        mock_storage.get_adjacency.return_value = adj
        result = patched_nav.find_graph_geometry_tension(
            "A", pattern_id="p", line_id="l", k_geometric=3,
        )
    assert result["hidden_cluster"] == []
    assert result["suspicious_links"] == []
    assert result["tension_score"] == 0.0


def test_in_edges_also_count_as_adjacency(patched_nav):
    """Incoming edges contribute to adjacency cohort, not just outgoing."""
    similarity = _make_similarity_result([("X", 0.1)])
    adj = _FakeAdj(
        out_edges={"A": []},
        in_edges={"A": ["X"]},  # X is connected via incoming edge
    )
    with patch.object(patched_nav, "find_similar_entities", return_value=similarity), \
         patch.object(patched_nav, "_storage", create=True) as mock_storage:
        mock_storage.get_adjacency.return_value = adj
        result = patched_nav.find_graph_geometry_tension(
            "A", pattern_id="p", line_id="l", k_geometric=1,
        )
    assert result["hidden_cluster"] == []  # X has edge (incoming)
    assert result["suspicious_links"] == []  # X is in behav, not extra


def test_geometric_distance_preserved_in_output(patched_nav):
    similarity = _make_similarity_result([("E_HIDDEN", 0.42)])
    adj = _FakeAdj(out_edges={"A": ["E_EDGE"]}, in_edges={"A": []})
    with patch.object(patched_nav, "find_similar_entities", return_value=similarity), \
         patch.object(patched_nav, "_storage", create=True) as mock_storage:
        mock_storage.get_adjacency.return_value = adj
        result = patched_nav.find_graph_geometry_tension(
            "A", pattern_id="p", line_id="l", k_geometric=1,
        )
    hidden = result["hidden_cluster"][0]
    assert hidden["neighbor_key"] == "E_HIDDEN"
    assert hidden["geometric_distance"] == pytest.approx(0.42)


def test_top_n_caps_apply_but_tension_uses_full_counts(patched_nav):
    """top_n caps the RETURNED list but tension_score reflects pre-cap totals
    (otherwise we'd underestimate the real signal — see PR #474 loopka fix)."""
    similarity = _make_similarity_result([
        (f"BEHAV_NO_EDGE_{i}", float(i) * 0.1) for i in range(10)
    ])
    adj = _FakeAdj(out_edges={"A": [f"EDGE_{i}" for i in range(10)]}, in_edges={"A": []})
    with patch.object(patched_nav, "find_similar_entities", return_value=similarity), \
         patch.object(patched_nav, "_storage", create=True) as mock_storage:
        mock_storage.get_adjacency.return_value = adj
        result = patched_nav.find_graph_geometry_tension(
            "A", pattern_id="p", line_id="l", k_geometric=10,
            top_n_hidden=3, top_n_suspicious=2,
        )
    assert len(result["hidden_cluster"]) == 3
    assert len(result["suspicious_links"]) == 2
    # 10 hidden + 10 suspicious / 10 k_geometric = 2.0 (NOT (3+2)/10 = 0.5)
    assert result["tension_score"] == pytest.approx(2.0)


def test_anchor_excluded_from_both_cells(patched_nav):
    """If find_similar_entities returns the anchor itself, it must not appear
    in hidden_cluster (self is not a 'hidden cohort member') nor count toward
    suspicious_links (self-loops on graph adjacency are not 'suspicious')."""
    similarity = _make_similarity_result([("A", 0.0), ("OTHER", 0.5)])
    adj = _FakeAdj(out_edges={"A": ["A", "DISTANT"]}, in_edges={"A": []})
    with patch.object(patched_nav, "find_similar_entities", return_value=similarity), \
         patch.object(patched_nav, "_storage", create=True) as mock_storage:
        mock_storage.get_adjacency.return_value = adj
        result = patched_nav.find_graph_geometry_tension(
            "A", pattern_id="p", line_id="l", k_geometric=2,
        )
    hidden_keys = {r["neighbor_key"] for r in result["hidden_cluster"]}
    suspicious_keys = {r["neighbor_key"] for r in result["suspicious_links"]}
    assert "A" not in hidden_keys
    assert "A" not in suspicious_keys
    assert hidden_keys == {"OTHER"}
    assert suspicious_keys == {"DISTANT"}
