# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for investigate_entity orchestrator — chains entity-side primitives
into one aggregated report. Mirror of investigate_chain (0.6.7)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest


@pytest.fixture
def nav():
    from hypertopos.navigation.navigator import GDSNavigator
    n = GDSNavigator.__new__(GDSNavigator)
    n._storage = MagicMock()
    n._storage.read_geometry.return_value = pa.table({
        "primary_key": ["A"],
        "delta_norm": [1.5],
        "is_anomaly": [True],
        "delta_rank_pct": [97.2],
    })
    n._storage.read_sphere.return_value = MagicMock(
        patterns={"account_pattern": MagicMock(), "tx_pattern": MagicMock()},
    )
    return n


def test_orchestrator_returns_all_blocks_and_steps_status(nav):
    with patch.object(nav, "explain_anomaly", return_value={"top_witness_dims": ["d1"]}), \
         patch.object(nav, "find_witness_cohort", return_value={"members": []}), \
         patch.object(nav, "find_chains_for_entity", return_value=[]), \
         patch.object(nav, "trace_root_cause", return_value={"steps": []}), \
         patch.object(nav, "find_graph_geometry_tension", return_value={
             "primary_key": "A", "hidden_cluster": [], "suspicious_links": [], "tension_score": 0.0,
         }), \
         patch.object(nav, "_resolve_version", return_value=1):
        result = nav.investigate_entity(
            "A", pattern_id="account_pattern", line_id="tx_pattern",
        )

    assert result["primary_key"] == "A"
    assert "polygon" in result
    assert "explain_anomaly" in result
    assert "witness_cohort" in result
    assert "chains" in result
    assert "root_cause" in result
    assert "graph_geometry_tension" in result
    assert "steps_status" in result
    for step in ("polygon", "explain_anomaly", "witness_cohort", "chains",
                 "root_cause", "graph_geometry_tension"):
        assert result["steps_status"][step]["ok"], f"step {step} should be ok"
    assert "elapsed_ms" in result


def test_partial_failure_does_not_abort_whole_call(nav):
    with patch.object(nav, "explain_anomaly", side_effect=KeyError("missing dim")), \
         patch.object(nav, "find_witness_cohort", return_value={"members": []}), \
         patch.object(nav, "find_chains_for_entity", return_value=[]), \
         patch.object(nav, "trace_root_cause", return_value={"steps": []}), \
         patch.object(nav, "find_graph_geometry_tension", return_value={
             "primary_key": "A", "hidden_cluster": [], "suspicious_links": [], "tension_score": 0.0,
         }), \
         patch.object(nav, "_resolve_version", return_value=1):
        result = nav.investigate_entity(
            "A", pattern_id="p", line_id="l",
        )

    assert result["steps_status"]["polygon"]["ok"] is True
    assert result["steps_status"]["explain_anomaly"]["ok"] is False
    assert "KeyError" in result["steps_status"]["explain_anomaly"]["error"]
    assert result["steps_status"]["witness_cohort"]["ok"] is True


def test_include_flags_control_step_execution(nav):
    with patch.object(nav, "explain_anomaly", return_value={"top_witness_dims": []}), \
         patch.object(nav, "find_witness_cohort", return_value={"members": []}) as mock_witness, \
         patch.object(nav, "find_chains_for_entity", return_value=[]) as mock_chains, \
         patch.object(nav, "trace_root_cause", return_value={"steps": []}), \
         patch.object(nav, "find_graph_geometry_tension", return_value={"tension_score": 0.0}), \
         patch.object(nav, "_resolve_version", return_value=1):
        result = nav.investigate_entity(
            "A", pattern_id="p", line_id="l",
            include_witness_cohort=False,
            include_chains=False,
        )

    mock_witness.assert_not_called()
    mock_chains.assert_not_called()
    assert "witness_cohort" not in result["steps_status"]
    assert "chains" not in result["steps_status"]
    assert result["steps_status"]["polygon"]["ok"] is True


def test_dataclass_return_unwrapped_to_dict(nav):
    """Primitives that return dataclasses (e.g. find_witness_cohort returns
    WitnessCohortResult) must be asdict'd at the orchestrator level so the
    block is a real dict on the wire, not a repr string."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class FakeWitness:
        primary_key: str
        members: list

    with patch.object(nav, "explain_anomaly", return_value={"top_witness_dims": []}), \
         patch.object(nav, "find_witness_cohort",
                      return_value=FakeWitness(primary_key="A", members=[])), \
         patch.object(nav, "find_chains_for_entity", return_value=[]), \
         patch.object(nav, "trace_root_cause", return_value={"steps": []}), \
         patch.object(nav, "find_graph_geometry_tension", return_value={"tension_score": 0.0}), \
         patch.object(nav, "_resolve_version", return_value=1):
        result = nav.investigate_entity(
            "A", pattern_id="p", line_id="l",
        )

    # NOT a string repr — dict-shaped
    assert isinstance(result["witness_cohort"], dict)
    assert result["witness_cohort"]["primary_key"] == "A"
    assert result["witness_cohort"]["members"] == []


def test_per_edge_counterfactual_off_by_default(nav):
    """M1.2 (simulate_edge_removal) not shipped — default include_per_edge_counterfactual=False
    so the orchestrator never tries to call it."""
    with patch.object(nav, "explain_anomaly", return_value={"top_witness_dims": []}), \
         patch.object(nav, "find_witness_cohort", return_value={"members": []}), \
         patch.object(nav, "find_chains_for_entity", return_value=[]), \
         patch.object(nav, "trace_root_cause", return_value={"steps": []}), \
         patch.object(nav, "find_graph_geometry_tension", return_value={"tension_score": 0.0}), \
         patch.object(nav, "_resolve_version", return_value=1):
        result = nav.investigate_entity(
            "A", pattern_id="p", line_id="l",
        )

    assert "per_edge_counterfactual" not in result["steps_status"]


def test_chains_block_passes_top_n_not_python_slice(nav):
    """find_chains_for_entity returns a DICT (with `chains` field), not a list —
    so the orchestrator must pass top_n as a kwarg, NOT apply `[:top_n_chains]`
    slicing (which produces `KeyError: slice(None, 3, None)` on a dict)."""
    captured: dict[str, object] = {}

    def fake_find_chains(pk, chain_pid, top_n=20):
        captured["top_n"] = top_n
        return {"primary_key": pk, "pattern_id": chain_pid, "chains": [], "summary": {}}

    with patch.object(nav, "explain_anomaly", return_value={"top_witness_dims": []}), \
         patch.object(nav, "find_witness_cohort", return_value={"members": []}), \
         patch.object(nav, "find_chains_for_entity", side_effect=fake_find_chains), \
         patch.object(nav, "trace_root_cause", return_value={"steps": []}), \
         patch.object(nav, "find_graph_geometry_tension", return_value={"tension_score": 0.0}), \
         patch.object(nav, "_resolve_version", return_value=1):
        result = nav.investigate_entity(
            "A", pattern_id="p", line_id="l",
            chain_pattern_id="tx_chains_pattern",
            top_n_chains=3,
        )

    assert captured["top_n"] == 3
    assert result["steps_status"]["chains"]["ok"] is True
    # Plan spec line 198-208: chains is a FLAT list at the orchestrator level,
    # not the dict that find_chains_for_entity returns natively. Orchestrator
    # unwraps the inner "chains" key to remove double-nesting for consumers.
    assert isinstance(result["chains"], list)


def test_per_edge_counterfactual_opt_in_calls_simulate_edge_removal(nav):
    """M1.2 v0 shipped — opt-in now calls the real simulate_edge_removal.
    Verifies the wire-up: orchestrator delegates to the navigator method
    with the correct args, and the result lands in the per_edge_counterfactual
    block. Earlier test (pre-M1.2) expected `ok: False` with `not yet
    available` — replaced as M1.2 v0 has landed."""
    expected_rows = [
        {"edge_id": "E1", "drop_pct": 12.5, "delta_norm_after": 3.1},
        {"edge_id": "E2", "drop_pct": 8.0, "delta_norm_after": 3.4},
    ]
    with patch.object(nav, "explain_anomaly", return_value={"top_witness_dims": []}), \
         patch.object(nav, "find_witness_cohort", return_value={"members": []}), \
         patch.object(nav, "find_chains_for_entity", return_value=[]), \
         patch.object(nav, "trace_root_cause", return_value={"steps": []}), \
         patch.object(nav, "find_graph_geometry_tension", return_value={"tension_score": 0.0}), \
         patch.object(nav, "simulate_edge_removal", return_value=expected_rows) as mock_sim, \
         patch.object(nav, "_resolve_version", return_value=1):
        result = nav.investigate_entity(
            "A", pattern_id="p", line_id="l",
            include_per_edge_counterfactual=True,
            top_n_edges=3,
        )

    pec = result["steps_status"]["per_edge_counterfactual"]
    assert pec["ok"] is True
    assert pec["error"] is None
    assert result["per_edge_counterfactual"] == expected_rows
    mock_sim.assert_called_once_with(
        "A", pattern_id="p", line_id="l", top_n=3,
    )
