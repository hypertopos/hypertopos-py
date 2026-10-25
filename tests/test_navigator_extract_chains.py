# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""GDSNavigator.extract_chains — the single implementation behind the MCP tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.navigation.navigator import GDSNavigationError, GDSNavigator

DELTAS = {
    "A": [0.0, 0.0],
    "B": [3.0, 4.0],
    "C": [3.0, 0.0],
    "D": [0.0, 0.0],
}


def _nav() -> GDSNavigator:
    points = pa.table(
        {
            "primary_key": ["T1", "T2", "T3"],
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "D"],
            "ts": ["2024-01-01T00:00:00", "2024-01-01T01:00:00", "2024-01-01T02:00:00"],
            "amount": [100.0, 90.0, 80.0],
        }
    )
    sphere = MagicMock()
    sphere.patterns = {
        "tx": MagicMock(entity_type="transactions"),
        "acct": MagicMock(entity_type="accounts"),
    }
    sphere.lines = {}

    def read_geometry(pattern_id, version, point_keys=None, columns=None):
        keys = [k for k in DELTAS if point_keys is None or k in point_keys]
        return pa.table({"primary_key": keys, "delta": [DELTAS[k] for k in keys]})

    storage = MagicMock()
    storage.read_sphere.return_value = sphere
    storage.read_points.return_value = points
    storage.read_geometry.side_effect = read_geometry
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    return GDSNavigator(
        engine=MagicMock(), storage=storage, manifest=manifest, contract=MagicMock()
    )


def _call(nav: GDSNavigator, **kwargs):
    return nav.extract_chains(
        "tx", "src", "dst", time_col="ts", amount_col="amount", min_hops=2, **kwargs
    )


def test_anchor_pattern_adds_edge_potentials_per_hop() -> None:
    nav = _nav()
    plain = _call(nav)
    with_anchor = _call(nav, anchor_pattern_id="acct")

    longest = with_anchor["chains"][0]
    assert longest["keys"] == ["A", "B", "C", "D"]
    np.testing.assert_allclose(longest["edge_potentials"], [5.0, 4.0, 3.0])
    assert [c["keys"] for c in with_anchor["chains"]] == [c["keys"] for c in plain["chains"]]
    assert all(p is None for c in plain["chains"] for p in c["edge_potentials"])
    assert "interpretation" not in with_anchor


def test_unknown_anchor_pattern_raises() -> None:
    with pytest.raises(GDSNavigationError, match="anchor_pattern_id='nope' not found"):
        _call(_nav(), anchor_pattern_id="nope")


def test_reads_points_of_the_pattern_entity_type() -> None:
    nav = _nav()
    _call(nav)
    assert nav._storage.read_points.call_args.args[0] == "transactions"
