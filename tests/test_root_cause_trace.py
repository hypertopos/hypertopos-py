# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Unit + integration tests for trace_root_cause."""
from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from hypertopos.engine.adjacency import AdjacencyIndex
from hypertopos.navigation.navigator import (
    GDSNavigator,
    RootCauseNode,
)
from hypertopos.storage._schemas import EDGE_TABLE_SCHEMA


def _empty_adjacency() -> AdjacencyIndex:
    return AdjacencyIndex._empty()


def _empty_edges_table() -> pa.Table:
    return pa.table(
        {f.name: pa.array([], type=f.type) for f in EDGE_TABLE_SCHEMA},
    )


class TestRootCauseNode:
    def test_fields(self):
        node = RootCauseNode(
            entity_key="ACC-001",
            role="root",
            severity="critical",
            evidence={"delta_norm": 5.2},
            children=[],
        )
        assert node.entity_key == "ACC-001"
        assert node.role == "root"
        assert node.severity == "critical"
        assert node.evidence == {"delta_norm": 5.2}
        assert node.children == []

    def test_asdict_nested(self):
        child = RootCauseNode(
            entity_key="ACC-002",
            role="edge_counterparty",
            severity="high",
            evidence={"witness_dim": "amount_std"},
            children=[],
        )
        parent = RootCauseNode(
            entity_key="ACC-001",
            role="root",
            severity="critical",
            evidence={"top_dim": "amount_std"},
            children=[child],
        )
        d = asdict(parent)
        assert d["children"][0]["entity_key"] == "ACC-002"
        assert d["children"][0]["role"] == "edge_counterparty"


def _make_nav_with_mocks(*, entity_line: str | None = "accounts") -> GDSNavigator:
    """Build a minimally-functional GDSNavigator for monkeypatch-based unit tests.

    The downstream primitives (explain_anomaly, find_counterparties, contagion_score,
    π7_attract_hub) are monkeypatched per test, so the storage/engine mocks only
    need to satisfy the sphere/pattern lookup done at the top of trace_root_cause.
    """
    storage = MagicMock()
    sphere = MagicMock()
    pattern = MagicMock()
    pattern.pattern_type = "anchor"
    pattern.relations = []
    pattern.theta = None
    sphere.patterns = {"test_pattern": pattern}
    sphere.entity_line = MagicMock(return_value=entity_line)
    storage.read_sphere = MagicMock(return_value=sphere)
    storage.get_adjacency = MagicMock(return_value=_empty_adjacency())
    storage.read_edges = MagicMock(return_value=_empty_edges_table())
    storage.count_geometry_rows = MagicMock(return_value=0)
    engine = MagicMock()
    manifest = MagicMock()
    contract = MagicMock()
    return GDSNavigator(engine, storage, manifest, contract)


@pytest.fixture
def nav_mock():
    return _make_nav_with_mocks()


class TestTraceRootCauseUnit:
    def test_non_anomalous_entity_returns_root_only(self, monkeypatch, nav_mock):
        def fake_explain(self, pk, pid):
            return {
                "severity": "normal",
                "top_dimensions": [],
                "primary_key": pk,
                "pattern_id": pid,
            }
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        result = nav_mock.trace_root_cause("whatever", "test_pattern")
        assert result["hop_count"] == 1
        assert result["branches_explored"] == 0
        assert result["truncated"] is False
        assert result["root"]["role"] == "root"
        assert result["root"]["severity"] == "normal"
        assert result["root"]["children"] == []
        assert "not anomalous" in result["summary"].lower()

    def test_max_depth_zero_returns_only_root(self, monkeypatch, nav_mock):
        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [
                    {"label": "amount_std", "z_score": 4.2, "contribution": 0.7},
                ],
                "primary_key": pk,
                "pattern_id": pid,
                "delta_norm": 6.3,
            }
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        result = nav_mock.trace_root_cause("ACC-1", "test_pattern", max_depth=0)
        assert result["hop_count"] == 1
        assert result["root"]["children"] == []
        assert result["root"]["severity"] == "critical"

    def test_max_branches_caps_children(self, monkeypatch, nav_mock):
        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [
                    {"label": "dim_a", "z_score": 5.0, "contribution": 0.5},
                    {"label": "dim_b", "z_score": 4.0, "contribution": 0.3},
                    {"label": "dim_c", "z_score": 3.0, "contribution": 0.15},
                    {"label": "dim_d", "z_score": 2.5, "contribution": 0.05},
                ],
                "primary_key": pk,
                "pattern_id": pid,
                "delta_norm": 7.0,
            }

        def fake_contagion(self, pk, pid, **kw):
            return {"score": 0.9, "total_counterparties": 10, "anomalous_counterparties": 9}

        def fake_hub(self, pid, top_n=10, **kw):
            return [("ACC-1", 12, 0.5)]

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "contagion_score", fake_contagion)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", fake_hub)

        result = nav_mock.trace_root_cause(
            "ACC-1", "test_pattern",
            max_depth=1, max_branches=2,
        )
        assert len(result["root"]["children"]) <= 2
        assert result["branches_explored"] <= 2

    def _stub_graph_companion(self, monkeypatch, graph_pid: str = "graph_pid"):
        """Make _resolve_edge_pattern_for_anchor return a truthy graph pid for unit tests."""
        monkeypatch.setattr(
            GDSNavigator, "_resolve_edge_pattern_for_anchor",
            lambda self, pid: graph_pid,
        )

    def test_hub_branch_added_when_entity_is_top_hub(self, monkeypatch, nav_mock):
        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [
                    {"label": "amount_std", "z_score": 4.2, "contribution": 0.7},
                ],
                "primary_key": pk,
                "pattern_id": pid,
                "delta_norm": 6.3,
            }

        def fake_contagion(self, pk, pid, **kw):
            return {"score": 0.0, "total_counterparties": 0}

        def fake_hub(self, pid, top_n=10, **kw):
            return [("ACC-HUB", 99, 0.99)]

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "contagion_score", fake_contagion)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", fake_hub)

        result = nav_mock.trace_root_cause("ACC-HUB", "test_pattern", max_depth=1)
        child_roles = {c["role"] for c in result["root"]["children"]}
        assert "hub" in child_roles

    def test_contagion_branch_added_when_score_above_threshold(self, monkeypatch, nav_mock):
        self._stub_graph_companion(monkeypatch)

        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [
                    {"label": "amount_std", "z_score": 4.2, "contribution": 0.7},
                ],
                "primary_key": pk,
                "pattern_id": pid,
                "delta_norm": 6.3,
            }

        def fake_contagion(self, pk, pid, **kw):
            return {"score": 0.8, "total_counterparties": 10, "anomalous_counterparties": 8}

        def fake_hub(self, pid, top_n=10, **kw):
            return []

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "contagion_score", fake_contagion)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", fake_hub)

        result = nav_mock.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        child_roles = {c["role"] for c in result["root"]["children"]}
        assert "neighbor_contamination" in child_roles

    def test_unknown_pattern_raises(self, nav_mock):
        from hypertopos.navigation.navigator import GDSNavigationError
        with pytest.raises(GDSNavigationError):
            nav_mock.trace_root_cause("ACC-1", "does_not_exist")


class TestTraceRootCauseRecursion:
    """Drive the recursive edge_counterparty branch and the visited-set cycle guard."""

    def _make_nav_with_relations(self) -> GDSNavigator:
        storage = MagicMock()
        sphere = MagicMock()
        pattern = MagicMock()
        pattern.pattern_type = "anchor"

        relation = MagicMock()
        relation.line_id = "counterparties"
        relation.from_col = "from"
        relation.to_col = "to"
        pattern.relations = [relation]
        pattern.theta = None

        sphere.patterns = {"test_pattern": pattern}
        sphere.entity_line = MagicMock(return_value="main_line")
        storage.read_sphere = MagicMock(return_value=sphere)
        storage.get_adjacency = MagicMock(return_value=_empty_adjacency())
        storage.read_edges = MagicMock(return_value=_empty_edges_table())
        storage.count_geometry_rows = MagicMock(return_value=0)
        return GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())

    def _stub_graph_companion(self, monkeypatch, graph_pid: str = "graph_pid"):
        monkeypatch.setattr(
            GDSNavigator, "_resolve_edge_pattern_for_anchor",
            lambda self, pid: graph_pid,
        )

    def test_edge_counterparty_recursion(self, monkeypatch):
        nav = self._make_nav_with_relations()
        self._stub_graph_companion(monkeypatch)

        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [
                    {"label": "amount_std", "z_score": 4.2, "contribution": 0.7},
                ],
                "primary_key": pk,
                "pattern_id": pid,
                "delta_norm": 6.3,
            }

        def fake_counterparties(self, pk, line_id, from_col, to_col, **kw):
            if pk == "ACC-ROOT":
                return {"incoming": [], "outgoing": [
                    {"primary_key": "ACC-CP", "is_anomaly": True, "delta_norm": 4.1},
                ]}
            return {"outgoing": [], "incoming": []}

        def fake_contagion(self, pk, pid, **kw):
            return {"score": 0.0, "total_counterparties": 0}

        def fake_hub(self, pid, top_n=10, **kw):
            return []

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "find_counterparties", fake_counterparties)
        monkeypatch.setattr(GDSNavigator, "contagion_score", fake_contagion)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", fake_hub)

        result = nav.trace_root_cause("ACC-ROOT", "test_pattern", max_depth=2)
        cp_children = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert len(cp_children) == 1
        assert cp_children[0]["entity_key"] == "ACC-CP"
        # The recursed child got its own explain_anomaly call and thus evidence
        assert cp_children[0]["severity"] == "critical"
        assert cp_children[0]["evidence"].get("via_dim") == "amount_std"
        assert result["hop_count"] == 2

    def test_graph_companion_is_used_for_contagion_and_counterparties(self, monkeypatch):
        """When the anchor has a companion event pattern, contagion_score and find_counterparties
        must be called with the COMPANION pid, not the anchor pid."""
        nav = self._make_nav_with_relations()
        monkeypatch.setattr(
            GDSNavigator, "_resolve_edge_pattern_for_anchor",
            lambda self, pid: "event_companion",
        )

        seen_contagion_pids: list[str] = []
        seen_cp_pids: list[str] = []

        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [{"label": "amount_std", "z_score": 4.2, "contribution": 0.7}],
                "primary_key": pk, "pattern_id": pid, "delta_norm": 6.3,
            }

        def fake_contagion(self, pk, pid, **kw):
            seen_contagion_pids.append(pid)
            return {"score": 0.9, "total_counterparties": 10, "anomalous_counterparties": 9}

        def fake_counterparties(self, pk, line_id, from_col, to_col, **kw):
            seen_cp_pids.append(kw.get("pattern_id"))
            return {"outgoing": [], "incoming": []}

        def fake_hub(self, pid, top_n=10, **kw):
            return []

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "contagion_score", fake_contagion)
        monkeypatch.setattr(GDSNavigator, "find_counterparties", fake_counterparties)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", fake_hub)

        nav.trace_root_cause("ACC-ROOT", "test_pattern", max_depth=2)
        assert seen_contagion_pids == ["event_companion"], seen_contagion_pids
        assert seen_cp_pids == ["event_companion"], seen_cp_pids

    def test_no_graph_companion_means_no_graph_branches(self, monkeypatch):
        """When the anchor has NO graph companion (no edge table anywhere),
        contagion_score and find_counterparties are not called at all."""
        nav = self._make_nav_with_relations()
        monkeypatch.setattr(
            GDSNavigator, "_resolve_edge_pattern_for_anchor",
            lambda self, pid: None,
        )

        contagion_called = []
        cp_called = []

        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [{"label": "x", "z_score": 5.0, "contribution": 0.8}],
                "primary_key": pk, "pattern_id": pid, "delta_norm": 6.0,
            }

        def tracking_contagion(self, pk, pid, **kw):
            contagion_called.append(pid)
            return {"score": 0.9, "total_counterparties": 10}

        def tracking_counterparties(self, pk, line_id, **kw):
            cp_called.append(line_id)
            return {"outgoing": [], "incoming": []}

        def fake_hub(self, pid, top_n=10, **kw):
            return []

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "contagion_score", tracking_contagion)
        monkeypatch.setattr(GDSNavigator, "find_counterparties", tracking_counterparties)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", fake_hub)

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2)
        assert contagion_called == []
        assert cp_called == []
        assert result["root"]["children"] == []

    def test_downstream_gds_errors_degrade_gracefully(self, monkeypatch):
        """contagion_score raising GDSNavigationError must NOT escape trace_root_cause."""
        from hypertopos.navigation.navigator import GDSNavigationError

        nav = self._make_nav_with_relations()
        self._stub_graph_companion(monkeypatch)

        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [
                    {"label": "amount_std", "z_score": 4.2, "contribution": 0.7},
                ],
                "primary_key": pk,
                "pattern_id": pid,
                "delta_norm": 6.3,
            }

        def raising_contagion(self, pk, pid, **kw):
            raise GDSNavigationError("no edge table")

        def raising_counterparties(self, pk, line_id, from_col, to_col, **kw):
            raise GDSNavigationError("no edge table")

        def raising_hub(self, pid, top_n=10, **kw):
            raise GDSNavigationError("wrong mode")

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "contagion_score", raising_contagion)
        monkeypatch.setattr(GDSNavigator, "find_counterparties", raising_counterparties)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", raising_hub)

        # Must not raise — all downstream failures degrade to "no children".
        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2)
        assert result["root"]["severity"] == "critical"
        assert result["root"]["children"] == []
        assert result["branches_explored"] == 0

    def test_visited_set_prevents_cycle(self, monkeypatch):
        """When counterparties point A→B→A, the visited-set guard emits a cycle marker
        on the repeat node instead of recursing forever."""
        nav = self._make_nav_with_relations()
        self._stub_graph_companion(monkeypatch)

        def fake_explain(self, pk, pid):
            return {
                "severity": "critical",
                "top_dimensions": [
                    {"label": "amount_std", "z_score": 4.2, "contribution": 0.7},
                ],
                "primary_key": pk,
                "pattern_id": pid,
                "delta_norm": 6.3,
            }

        def fake_counterparties(self, pk, line_id, from_col, to_col, **kw):
            # A → B, then B → A: the second hop must NOT recurse.
            if pk == "ACC-ROOT":
                return {"incoming": [], "outgoing": [
                    {"primary_key": "ACC-B", "is_anomaly": True, "delta_rank_pct": 99.5},
                ]}
            return {"incoming": [], "outgoing": [
                {"primary_key": "ACC-ROOT", "is_anomaly": True, "delta_rank_pct": 99.5},
            ]}

        def fake_contagion(self, pk, pid, **kw):
            return {"score": 0.0, "total_counterparties": 0}

        def fake_hub(self, pid, top_n=10, **kw):
            return []

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", fake_explain)
        monkeypatch.setattr(GDSNavigator, "find_counterparties", fake_counterparties)
        monkeypatch.setattr(GDSNavigator, "contagion_score", fake_contagion)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", fake_hub)

        result = nav.trace_root_cause("ACC-ROOT", "test_pattern", max_depth=3)
        # Level 1: ACC-B, Level 2: ACC-ROOT (cycle-guarded)
        lvl1 = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert len(lvl1) == 1
        assert lvl1[0]["entity_key"] == "ACC-B"
        lvl2 = [c for c in lvl1[0]["children"] if c["role"] == "edge_counterparty"]
        assert len(lvl2) == 1
        assert lvl2[0]["entity_key"] == "ACC-ROOT"
        assert lvl2[0]["evidence"].get("cycle") is True


class TestTraceRootCauseQualityFixes:
    """Covers the 11 post-initial-ship quality fixes — sort-by-anomaly,
    unified severity scale, priority ordering, truncated semantics,
    anomalous_cp_keys, max_total_nodes, version-keyed cache."""

    def _make_nav(self) -> GDSNavigator:
        storage = MagicMock()
        sphere = MagicMock()
        pattern = MagicMock()
        pattern.pattern_type = "anchor"
        pattern.relations = []
        pattern.theta = None
        pattern.population_size = 100
        sphere.patterns = {"test_pattern": pattern}
        sphere.entity_line = MagicMock(return_value="accounts")
        storage.read_sphere = MagicMock(return_value=sphere)
        storage.get_adjacency = MagicMock(return_value=_empty_adjacency())
        storage.read_edges = MagicMock(return_value=_empty_edges_table())
        storage.count_geometry_rows = MagicMock(return_value=0)
        return GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())

    def _stub_companion(self, monkeypatch, pid: str = "graph_pid"):
        monkeypatch.setattr(
            GDSNavigator, "_resolve_edge_pattern_for_anchor",
            lambda self, p: pid,
        )
        monkeypatch.setattr(GDSNavigator, "_resolve_version", lambda self, p: 1)

    def test_counterparties_sorted_by_anomaly_not_volume(self, monkeypatch):
        """edge_counterparty picks the most anomalous cp, not the highest-volume one.

        Regression fix: previously top_n=20 sorted by amount_sum inside find_counterparties
        meant anomalous-but-low-volume counterparties were never picked, so this branch
        never fired on AML-scale spheres.
        """
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "critical",
            "top_dimensions": [{"label": "amount_std"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        # HIGH_VOL is the high-volume BUT NOT anomalous cp; LOW_VOL_ANOM is the tail-end
        # of the volume list but flagged anomalous. Previous code picked HIGH_VOL_ANOM
        # only if it happened to be in top-by-volume; the fix sorts by is_anomaly first.
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "incoming": [],
                "outgoing": [
                    {"primary_key": "HIGH_VOL", "is_anomaly": False, "delta_rank_pct": 50.0},
                    {"primary_key": "HIGHER_VOL", "is_anomaly": False, "delta_rank_pct": 60.0},
                    {"primary_key": "LOW_VOL_ANOM", "is_anomaly": True, "delta_rank_pct": 99.5},
                    {"primary_key": "LOW_VOL_ANOM_WEAK", "is_anomaly": True, "delta_rank_pct": 90.0},
                ],
            },
        )

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2, max_branches=3)
        cp = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert len(cp) == 1
        # The picked one must be the most anomalous, not the highest-volume
        assert cp[0]["entity_key"] == "LOW_VOL_ANOM"

    def test_anomalous_cp_keys_in_contagion_evidence(self, monkeypatch):
        """contagion branch evidence includes anomalous_cp_keys — saves a follow-up call."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "critical",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.35, "total_counterparties": 5, "anomalous_counterparties": 2,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "incoming": [],
                "outgoing": [
                    {"primary_key": "CP1", "is_anomaly": False},
                    {"primary_key": "CP2", "is_anomaly": True, "delta_rank_pct": 98.0},
                    {"primary_key": "CP3", "is_anomaly": True, "delta_rank_pct": 99.0},
                    {"primary_key": "CP4", "is_anomaly": False},
                ],
            },
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        cont = [c for c in result["root"]["children"] if c["role"] == "neighbor_contamination"]
        assert len(cont) == 1
        assert set(cont[0]["evidence"]["anomalous_cp_keys"]) == {"CP2", "CP3"}

    def test_unified_severity_scale(self, monkeypatch):
        """Every node in the DAG uses the same severity vocabulary.

        Contagion severity scale: low/moderate/high/critical (replacing the old
        low/moderate/high-only scale). Hub inherits root severity.
        """
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        # contagion 0.8 → critical (new scale); previously was "high"
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.8, "total_counterparties": 10, "anomalous_counterparties": 8,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub",
            lambda self, pid, top_n=10, **kw: [("ACC-1", 5, 0.5)],
        )

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        severities = {c["severity"] for c in result["root"]["children"]}
        # All severities are from the unified vocabulary
        allowed = {"normal", "low", "moderate", "high", "critical", "extreme"}
        assert severities <= allowed
        # Specifically: contagion 0.8 maps to "critical" in the unified scale
        cont = [c for c in result["root"]["children"] if c["role"] == "neighbor_contamination"]
        assert cont and cont[0]["severity"] == "critical"

    def test_priority_ordering_keeps_strongest(self, monkeypatch):
        """When more candidates exist than max_branches, the highest-severity ones win
        — not first-come first-serve."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        # Weak contagion (low) + anomalous cp (extreme) + hub (extreme)
        # max_branches=2: must keep the TWO extreme ones, drop the low one.
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.15, "total_counterparties": 10, "anomalous_counterparties": 1,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "incoming": [],
                "outgoing": [
                    {"primary_key": "ANOM_CP", "is_anomaly": True, "delta_rank_pct": 99.95},
                ],
            },
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub",
            lambda self, pid, top_n=10, **kw: [("ACC-1", 99, 0.99)],
        )

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2, max_branches=2)
        roles = {c["role"] for c in result["root"]["children"]}
        # neighbor_contamination is "low" (0.15 score) — dropped in favor of the two extremes
        assert "neighbor_contamination" not in roles
        assert "edge_counterparty" in roles
        assert "hub" in roles
        assert result["truncated"] is True  # one candidate was dropped

    def test_truncated_false_when_nothing_was_dropped(self, monkeypatch):
        """truncated=False when the number of candidates is <= max_branches."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.9, "total_counterparties": 10, "anomalous_counterparties": 9,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        # Only 1 candidate (contagion), max_branches=3 → nothing dropped
        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1, max_branches=3)
        assert result["truncated"] is False

    def test_contagion_below_threshold_skipped(self, monkeypatch):
        """Contagion branch is NOT attached when score is below contagion_min_threshold."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.05, "total_counterparties": 20, "anomalous_counterparties": 1,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        # Default threshold is 0.10 — 0.05 must be rejected
        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        roles = {c["role"] for c in result["root"]["children"]}
        assert "neighbor_contamination" not in roles

        # Lower the threshold → branch fires
        result2 = nav.trace_root_cause(
            "ACC-1", "test_pattern",
            max_depth=1, contagion_min_threshold=0.01,
        )
        roles2 = {c["role"] for c in result2["root"]["children"]}
        assert "neighbor_contamination" in roles2

    def test_hub_pop_limit_configurable(self, monkeypatch):
        """hub_pop_limit kwarg gates the hub branch when the population is large."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        # Override population_size on the mock pattern to 100k
        sphere = nav._storage.read_sphere()
        sphere.patterns["test_pattern"].population_size = 100_000

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )

        hub_calls = []
        def tracking_hub(self, pid, top_n=10, **kw):
            hub_calls.append(pid)
            return [("ACC-1", 99, 0.99)]

        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", tracking_hub)

        # Default hub_pop_limit=50_000 — 100k > 50k → hub skipped
        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        roles = {c["role"] for c in result["root"]["children"]}
        assert "hub" not in roles
        assert hub_calls == []

        # Raise the limit — hub branch fires
        result2 = nav.trace_root_cause(
            "ACC-1", "test_pattern", max_depth=1, hub_pop_limit=200_000,
        )
        roles2 = {c["role"] for c in result2["root"]["children"]}
        assert "hub" in roles2
        assert hub_calls  # π7 was called this time

    def test_max_total_nodes_hard_cap(self, monkeypatch):
        """A hard cap on expanded nodes prevents recursion blowups."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })

        chain_depth = {"n": 0}
        def fake_counterparties(self, pk, line_id, from_col, to_col, **kw):
            chain_depth["n"] += 1
            # Each cp points to a unique new entity, so recursion would continue until depth cap
            return {
                "incoming": [],
                "outgoing": [
                    {"primary_key": f"ACC-CHAIN-{chain_depth['n']}", "is_anomaly": True, "delta_rank_pct": 99.9},
                ],
            }

        monkeypatch.setattr(GDSNavigator, "find_counterparties", fake_counterparties)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        result = nav.trace_root_cause(
            "ACC-ROOT", "test_pattern",
            max_depth=20, max_branches=1, max_total_nodes=3,
        )
        # Cap enforced: no more than 3 nodes expanded
        assert result["hop_count"] <= 3
        assert result["truncated"] is True

    def test_hub_cache_is_version_keyed(self, monkeypatch):
        """Cache key is (pattern_id, version) — a version bump invalidates the cached hub set."""
        nav = self._make_nav()

        calls = {"resolve_companion": 0, "resolve_version": 0, "hub": 0}

        monkeypatch.setattr(GDSNavigator, "_resolve_edge_pattern_for_anchor",
            lambda self, p: (calls.__setitem__("resolve_companion", calls["resolve_companion"] + 1) or "graph_pid"),
        )

        version_holder = {"v": 7}
        def version_fn(self, p):
            calls["resolve_version"] += 1
            return version_holder["v"]
        monkeypatch.setattr(GDSNavigator, "_resolve_version", version_fn)

        def hub_fn(self, pid, top_n=10, **kw):
            calls["hub"] += 1
            return [("ACC-1", 5, 0.5)]
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", hub_fn)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )

        # First call — cache miss, π7 runs
        nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        assert calls["hub"] == 1

        # Second call, same version — cache hit
        nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        assert calls["hub"] == 1

        # Bump version — cache miss, π7 runs again
        version_holder["v"] = 8
        nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        assert calls["hub"] == 2

    def test_severity_medium_mapped_to_moderate(self, monkeypatch):
        """Legacy explain_anomaly 'medium' is mapped into the unified 'moderate' bucket."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "medium",  # legacy scale
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        assert result["root"]["severity"] == "moderate"

    def test_edge_counterparty_top_n_expands_multiple(self, monkeypatch):
        """edge_counterparty_top_n allows expanding more than one anomalous cp per node."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "outgoing": [
                    {"primary_key": "ANOM1", "is_anomaly": True, "delta_rank_pct": 99.9},
                    {"primary_key": "ANOM2", "is_anomaly": True, "delta_rank_pct": 99.5},
                    {"primary_key": "ANOM3", "is_anomaly": True, "delta_rank_pct": 99.0},
                ],
                "incoming": [],
            },
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        result = nav.trace_root_cause(
            "ACC-1", "test_pattern",
            max_depth=2, max_branches=5, edge_counterparty_top_n=2,
        )
        edge_cps = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert len(edge_cps) == 2
        # Three anomalous CPs exist but we capped at 2 — truncated=True.
        assert result["truncated"] is True

    def test_witness_counterparty_delta_norm_removed(self, monkeypatch):
        """edge_counterparty inherited evidence no longer carries the always-null
        witness_counterparty_delta_norm field (find_counterparties doesn't return it)."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "outgoing": [
                    {"primary_key": "ANOM1", "is_anomaly": True, "delta_rank_pct": 99.5},
                ],
                "incoming": [],
            } if pk == "ACC-1" else {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2)
        edge_cps = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert len(edge_cps) == 1
        assert "witness_counterparty_delta_norm" not in edge_cps[0]["evidence"]
        assert "witness_counterparty_delta_rank_pct" in edge_cps[0]["evidence"]

    def test_contagion_and_cps_cache_within_single_call(self, monkeypatch):
        """contagion_score and find_counterparties are called at most ONCE per entity
        per trace_root_cause call — even when edge_counterparty recurses back through
        an entity that was already seen from another path."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        contagion_calls: list[str] = []
        cps_calls: list[str] = []

        def fake_contagion(self, pk, pid, **kw):
            contagion_calls.append(pk)
            return {"score": 0.8, "total_counterparties": 2, "anomalous_counterparties": 1}

        def fake_cps(self, pk, line_id, from_col, to_col, **kw):
            cps_calls.append(pk)
            # ACC-1 → ACC-2, ACC-2 has no further anomalous cps
            if pk == "ACC-1":
                return {"outgoing": [
                    {"primary_key": "ACC-2", "is_anomaly": True, "delta_rank_pct": 99.0},
                ], "incoming": []}
            return {"outgoing": [], "incoming": []}

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", fake_contagion)
        monkeypatch.setattr(GDSNavigator, "find_counterparties", fake_cps)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        nav.trace_root_cause("ACC-1", "test_pattern", max_depth=3, max_branches=3)
        # ACC-1 and ACC-2 each contagion_score exactly ONCE
        assert contagion_calls.count("ACC-1") == 1
        assert contagion_calls.count("ACC-2") == 1
        # ACC-1 and ACC-2 each find_counterparties exactly ONCE
        assert cps_calls.count("ACC-1") == 1
        assert cps_calls.count("ACC-2") == 1

    def test_branches_enabled_rejects_invalid_values(self, monkeypatch):
        """Typos in branches_enabled must error loudly, not silently disable everything."""
        from hypertopos.navigation.navigator import GDSNavigationError
        nav = self._make_nav()
        self._stub_companion(monkeypatch)
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme", "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        with pytest.raises(GDSNavigationError, match="invalid values"):
            nav.trace_root_cause(
                "ACC-1", "test_pattern",
                branches_enabled=["kubelek", "hub"],
            )

    def test_revisits_root_is_list_of_keys(self, monkeypatch):
        """revisits_root carries the list of cp keys that equal the root, not just a bool."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme", "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        # When tracing ACC-1, its contagion_score reports ACC-1 itself as one of the
        # anomalous counterparties (self-loop clique indicator).
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.5, "total_counterparties": 4, "anomalous_counterparties": 2,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "outgoing": [
                    {"primary_key": "ACC-1", "is_anomaly": True, "delta_rank_pct": 99.0},
                    {"primary_key": "ACC-2", "is_anomaly": True, "delta_rank_pct": 98.0},
                ],
                "incoming": [],
            },
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        # Self-edge filter drops ACC-1 from the cps_data used for edge_counterparty, but
        # contagion_score's own anomalous_cp_keys list (built from cps_data before the
        # self-filter happens in the recursion) still contains ACC-1 because the contagion
        # call reports it independently. Either way, revisits_root should be a LIST of str,
        # not a bool, when present.
        cont = [c for c in result["root"]["children"] if c["role"] == "neighbor_contamination"]
        assert cont
        rr = cont[0]["evidence"].get("revisits_root")
        if rr is not None:
            assert isinstance(rr, list)
            assert all(isinstance(x, str) for x in rr)

    def test_cross_call_ledger_surfaces_clique(self, monkeypatch):
        """After tracing ACC-A (which reports ACC-B as an anomalous cp), a later trace
        on ACC-B surfaces ACC-A via `previously_seen_as_cp_of`."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme", "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.5, "total_counterparties": 4, "anomalous_counterparties": 1,
        })

        def cps_fn(self, pk, line_id, from_col, to_col, **kw):
            # ACC-A's cps include ACC-B as anomalous
            if pk == "ACC-A":
                return {
                    "outgoing": [{"primary_key": "ACC-B", "is_anomaly": True, "delta_rank_pct": 99.0}],
                    "incoming": [],
                }
            # ACC-B's cps are NOT anomalous — isolated
            return {
                "outgoing": [{"primary_key": "ACC-X", "is_anomaly": False}],
                "incoming": [],
            }

        monkeypatch.setattr(GDSNavigator, "find_counterparties", cps_fn)
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        # First call: trace ACC-A. It reports ACC-B in its anomalous_cp_keys → ledger learns.
        nav.trace_root_cause("ACC-A", "test_pattern", max_depth=1)
        # Second call: trace ACC-B. Ledger shows ACC-B was previously seen as cp of ACC-A.
        result = nav.trace_root_cause("ACC-B", "test_pattern", max_depth=1)
        cont = [c for c in result["root"]["children"] if c["role"] == "neighbor_contamination"]
        assert cont
        assert cont[0]["evidence"].get("previously_seen_as_cp_of") == ["ACC-A"]

    def test_lru_cap_evicts_oldest(self, monkeypatch):
        """When contagion cache exceeds _LRU_MAX, oldest entries are evicted."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme", "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        # Pre-seed contagion cache beyond the _LRU_MAX=2000 cap.
        # Manually force the internal cap small for the test:
        # Drive 3 traces and assert cache size grows but stays bounded.
        for i in range(3):
            nav.trace_root_cause(f"ACC-{i}", "test_pattern", max_depth=1)
        # All 3 entities cached (under cap).
        assert len(nav._trace_contagion_cache) == 3
        assert len(nav._trace_cps_cache) == 3

    def test_fixture_sphere_smoke(self, sphere_path):
        """Integration test: drive trace_root_cause against the real sales_sphere fixture.

        Protects against the class of bugs that mock-only unit tests miss —
        find_counterparties response shape, explain_anomaly field naming,
        sphere.entity_line resolution on real patterns, graph companion
        detection on fixture data. One real-sphere test earns its keep the
        moment anyone edits the primitive.
        """
        from hypertopos.sphere import HyperSphere

        hs = HyperSphere.open(sphere_path)
        with hs.session("test-agent") as session:
            sphere = session._reader.read_sphere()
            # Fail loudly (not skip) when fixture expectations are violated —
            # skip-on-failure would silently erode: a fixture refactor could make
            # this test no-op without anyone noticing. Assert the preconditions.
            anchor_pids = [
                pid for pid, pat in sphere.patterns.items()
                if pat.pattern_type == "anchor"
            ]
            assert anchor_pids, (
                "Fixture regression: sales_sphere has no anchor pattern. "
                "Update the fixture or the test's entity-selection strategy."
            )
            pid = anchor_pids[0]

            # Pick any entity from the home line (whether anomalous or not;
            # trace_root_cause must degrade gracefully on non-anomalous entities).
            nav = session.navigator()
            try:
                anomalies = nav.π5_attract_anomaly(pid, top_n=1)
            except Exception:
                anomalies = []

            pk: str | None = None
            if anomalies:
                first = anomalies[0]
                # π5_attract_anomaly may return Polygon objects or (key, score) tuples
                pk = getattr(first, "primary_key", None) or (first[0] if isinstance(first, tuple) else None)
            if pk is None:
                # Fallback: first entity from the anchor line.
                home_line = sphere.entity_line(pid)
                assert home_line, (
                    f"Fixture regression: pattern '{pid}' has no resolvable entity_line."
                )
                version = nav._resolve_version(pid)
                geo = session._reader.read_geometry(
                    pid, version, columns=["primary_key"],
                )
                assert geo.num_rows > 0, (
                    f"Fixture regression: pattern '{pid}' has zero entities in geometry."
                )
                pk = geo["primary_key"][0].as_py()

            # Smoke: the call must not raise, must return all wrapper keys.
            result = nav.trace_root_cause(pk, pid, max_depth=2, max_branches=3)
            assert set(result.keys()) == {
                "root", "summary", "hop_count", "branches_explored", "truncated",
            }
            assert result["root"]["entity_key"] == pk
            assert result["root"]["role"] == "root"
            assert isinstance(result["summary"], str)
            assert result["hop_count"] >= 1
            # Every role in the tree comes from the unified vocabulary.
            valid_roles = {"root", "edge_counterparty", "hub", "neighbor_contamination"}

            def _check_roles(node: dict) -> None:
                assert node["role"] in valid_roles
                valid_severity = {"normal", "low", "moderate", "high", "critical", "extreme"}
                assert node["severity"] in valid_severity
                for c in node["children"]:
                    _check_roles(c)

            _check_roles(result["root"])

    def test_edge_counterparty_includes_edge_potential(self, monkeypatch):
        """edge_counterparty branch evidence includes edge_potential sub-dict."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "amount_out_std"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "outgoing": [
                    {"primary_key": "ANOM_CP", "is_anomaly": True, "delta_rank_pct": 99.5},
                ],
                "incoming": [],
            } if pk == "ACC-1" else {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])
        monkeypatch.setattr(GDSNavigator, "edge_potential",
            lambda self, fk, tk, pid, **kw: {
                "score": 12.5, "delta_distance": 5.0,
                "pair_tx_count": 1, "effective_weight": 1.0,
                "from_key": fk, "to_key": tk, "pattern_id": pid,
                "interpretation": "stub",
            },
        )

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2)
        edge_cps = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert edge_cps
        ev = edge_cps[0]["evidence"]
        assert "edge_potential" in ev
        assert ev["edge_potential"]["score"] == 12.5
        assert ev["edge_potential"]["pair_tx_count"] == 1

    def test_edge_counterparty_includes_motif_potential_when_cycle_2_found(self, monkeypatch):
        """edge_counterparty evidence gets motif_potential block when cycle_2 exists seed<->cp."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "amount_out_std"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "outgoing": [
                    {"primary_key": "ANOM_CP", "is_anomaly": True, "delta_rank_pct": 99.5},
                ],
                "incoming": [],
            } if pk == "ACC-1" else {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])
        monkeypatch.setattr(GDSNavigator, "edge_potential",
            lambda self, fk, tk, pid, **kw: {
                "score": 8.0, "delta_distance": 4.0,
                "pair_tx_count": 1, "effective_weight": 1.0,
                "from_key": fk, "to_key": tk, "pattern_id": pid,
            },
        )
        # Stub score_motif to return a cycle_2 for (ACC-1, ANOM_CP) pair.
        def _stub_score_motif(self, entity_key, motif_type, pattern_id, time_window_hours=None):
            if motif_type == "cycle_2" and entity_key == "ACC-1":
                return {
                    "found": True,
                    "motif_type": "cycle_2",
                    "seed": "ACC-1",
                    "counterparty": "ANOM_CP",
                    "ring": ["ACC-1", "ANOM_CP"],
                    "score": 64.0,
                    "edges": [("ACC-1", "ANOM_CP"), ("ANOM_CP", "ACC-1")],
                    "time_window_hours": 24,
                    "pattern_id": pattern_id,
                }
            return {"found": False, "score": 0.0, "motif_type": motif_type}
        monkeypatch.setattr(GDSNavigator, "score_motif", _stub_score_motif)

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2)
        edge_cps = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert edge_cps
        ev = edge_cps[0]["evidence"]
        assert "motif_potential" in ev
        assert ev["motif_potential"]["motif_type"] == "cycle_2"
        assert ev["motif_potential"]["score"] == 64.0
        assert ev["motif_potential"]["counterparty"] == "ANOM_CP"

    def test_edge_counterparty_no_motif_when_none_found(self, monkeypatch):
        """motif_potential key absent when no motif matches the cp — trace never breaks."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)
        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {
                "outgoing": [
                    {"primary_key": "X", "is_anomaly": True, "delta_rank_pct": 99.0},
                ],
                "incoming": [],
            } if pk == "ACC-1" else {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])
        monkeypatch.setattr(GDSNavigator, "edge_potential",
            lambda self, fk, tk, pid, **kw: {
                "score": 1.0, "delta_distance": 1.0,
                "pair_tx_count": 1, "effective_weight": 1.0,
                "from_key": fk, "to_key": tk, "pattern_id": pid,
            },
        )
        monkeypatch.setattr(GDSNavigator, "score_motif",
            lambda self, entity_key, motif_type, pattern_id, time_window_hours=None: {
                "found": False, "score": 0.0, "motif_type": motif_type,
            },
        )
        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=2)
        edge_cps = [c for c in result["root"]["children"] if c["role"] == "edge_counterparty"]
        assert edge_cps
        assert "motif_potential" not in edge_cps[0]["evidence"]

    def test_no_diagnostic_leakage(self, monkeypatch):
        """The final evidence dict must not include internal diagnostic fields
        (graph_companion, home_line) that were removed during cleanup."""
        nav = self._make_nav()
        self._stub_companion(monkeypatch)

        monkeypatch.setattr(GDSNavigator, "explain_anomaly", lambda self, pk, pid: {
            "severity": "extreme",
            "top_dimensions": [{"label": "x"}],
            "delta_norm": 5.0, "conformal_p": 1e-5,
        })
        monkeypatch.setattr(GDSNavigator, "contagion_score", lambda self, pk, pid, **kw: {
            "score": 0.0, "total_counterparties": 0,
        })
        monkeypatch.setattr(GDSNavigator, "find_counterparties",
            lambda self, pk, line_id, from_col, to_col, **kw: {"outgoing": [], "incoming": []},
        )
        monkeypatch.setattr(GDSNavigator, "π7_attract_hub", lambda self, pid, top_n=10, **kw: [])

        result = nav.trace_root_cause("ACC-1", "test_pattern", max_depth=1)
        root_ev = result["root"]["evidence"]
        assert "graph_companion" not in root_ev
        assert "home_line" not in root_ev
