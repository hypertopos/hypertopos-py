# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Edge-removal counterfactuals compare before/after in the pattern's metric."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.engine.counterfactual import (
    simulate_edge_removal_with_aggregations,
    simulate_joint_edge_removal,
)
from hypertopos.model.sphere import Pattern, RelationDef, Sphere
from hypertopos.navigation.navigator import GDSNavigator

MU = np.array([3.0, 5.0])
SIGMA = np.array([1.0, 2.0])
SHAPE = np.array([6.0, 9.0])
WEIGHTS = np.array([2.0, 2.0])
L_INV = np.array([[1.0, 0.0], [0.5, 0.5]])
RELATIONS = [{"line_id": "L0", "direction": "out"}, {"line_id": "L1", "direction": "out"}]
EDGES = [
    {"edge_id": "E-real", "partner_key": "P1", "direction": "out", "line_id": "L0"},
    {"edge_id": "E-noop", "partner_key": "P2", "direction": "out", "line_id": "LX"},
]

CASES = {
    "weights": (None, WEIGHTS),
}


def _delta(shape: np.ndarray, chol: np.ndarray | None, weights: np.ndarray | None) -> np.ndarray:
    d = chol @ (shape - MU) if chol is not None else (shape - MU) / SIGMA
    return d * weights if weights is not None else d


def _expected_drop(chol: np.ndarray | None, weights: np.ndarray | None) -> float:
    before = np.linalg.norm(_delta(SHAPE, chol, weights))
    after = np.linalg.norm(_delta(SHAPE - [1.0, 0.0], chol, weights))
    return float((before - after) / before * 100.0)


@pytest.mark.parametrize("case", list(CASES))
def test_per_edge_drop_in_pattern_metric(case: str) -> None:
    chol, weights = CASES[case]
    rows = simulate_edge_removal_with_aggregations(
        shape=SHAPE,
        mu=MU,
        sigma_diag=SIGMA,
        delta_norm=float(np.linalg.norm(_delta(SHAPE, chol, weights))),
        edges=EDGES,
        relations=RELATIONS,
        edge_agg_dim_offset=2,
        edge_agg_specs=[],
        event_source_values={},
        candidate_edge_ids=["E-real", "E-noop"],
        top_n=10,
        dimension_weights=weights,
    )
    by_id = {r["edge_id"]: r for r in rows}
    assert by_id["E-noop"]["drop_pct"] == pytest.approx(0.0, abs=1e-9)
    assert by_id["E-real"]["drop_pct"] == pytest.approx(_expected_drop(chol, weights), abs=1e-6)


@pytest.mark.parametrize("case", list(CASES))
def test_joint_drop_in_pattern_metric(case: str) -> None:
    chol, weights = CASES[case]
    out = simulate_joint_edge_removal(
        shape=SHAPE,
        mu=MU,
        sigma_diag=SIGMA,
        delta_norm=float(np.linalg.norm(_delta(SHAPE, chol, weights))),
        edges_to_remove=EDGES,
        all_edges=EDGES,
        relations=RELATIONS,
        edge_agg_dim_offset=2,
        edge_agg_specs=[],
        event_source_values={},
        dimension_weights=weights,
    )
    assert out["joint_drop_pct"] == pytest.approx(_expected_drop(chol, weights), abs=1e-6)
    noop = simulate_joint_edge_removal(
        shape=SHAPE,
        mu=MU,
        sigma_diag=SIGMA,
        delta_norm=float(np.linalg.norm(_delta(SHAPE, chol, weights))),
        edges_to_remove=[EDGES[1]],
        all_edges=EDGES,
        relations=RELATIONS,
        edge_agg_dim_offset=2,
        edge_agg_specs=[],
        event_source_values={},
        dimension_weights=weights,
    )
    assert noop["joint_drop_pct"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("case", list(CASES))
def test_navigator_simulate_edge_removal_in_pattern_metric(case: str) -> None:
    chol, weights = CASES[case]
    pattern = Pattern(
        pattern_id="P",
        entity_type="ents",
        pattern_type="anchor",
        relations=[RelationDef("L0", "out", False), RelationDef("L1", "out", False)],
        mu=MU.astype(np.float32),
        sigma_diag=SIGMA.astype(np.float32),
        theta=np.ones(2, dtype=np.float32),
        edge_max=np.ones(2, dtype=np.float32),
        population_size=10,
        computed_at=datetime.now(UTC),
        version=1,
        status="production",
        dimension_weights=weights,
    )
    sphere = MagicMock(spec=Sphere)
    sphere.patterns = {"P": pattern}
    sphere.lines = {}
    sphere.aliases = {}
    sphere.entity_line = MagicMock(return_value=None)
    storage = MagicMock()
    storage.read_sphere = MagicMock(return_value=sphere)
    storage.read_geometry_stats = MagicMock(return_value=None)
    delta = _delta(SHAPE, chol, weights).astype(np.float32)
    storage.read_geometry = MagicMock(
        return_value=pa.table(
            {
                "primary_key": ["SEED"],
                "delta": pa.array([delta.tolist()], type=pa.list_(pa.float32())),
                "delta_norm": pa.array([float(np.linalg.norm(delta))], type=pa.float32()),
            }
        )
    )
    adj = MagicMock()
    adj.neighbors_out = MagicMock(return_value=[("P1", "out", 0, "E-real")])
    adj.neighbors_in = MagicMock(return_value=[])
    storage.get_adjacency = MagicMock(return_value=adj)
    nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())
    nav._resolve_version = MagicMock(return_value=1)  # type: ignore[method-assign]

    envelope = nav.simulate_edge_removal("SEED", pattern_id="P", line_id="L0", top_n=5)
    (row,) = envelope["edges"]
    assert row["drop_pct"] == pytest.approx(_expected_drop(chol, weights), abs=1e-3)
