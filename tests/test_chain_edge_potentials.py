# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Per-hop edge_potentials field on Chain.to_dict()."""
from __future__ import annotations

import json
import math

import numpy as np

from hypertopos.engine.chains import Chain


def _make_chain(keys: list[str]) -> Chain:
    """Construct a Chain with fixed per-hop metadata. Only `keys` matters
    for edge_potentials; the other fields are filled with neutral values."""
    hop_count = max(0, len(keys) - 1)
    return Chain(
        chain_id="CHAIN-000001",
        keys=list(keys),
        event_keys=[f"EV-{i:03d}" for i in range(hop_count)],
        hop_count=hop_count,
        is_cyclic=False,
        time_span_hours=0.0,
        categories=[""] * hop_count,
        amounts=[0.0] * hop_count,
        amount_decay=0.0,
    )


def test_edge_potentials_engineered_distances() -> None:
    """4-entity chain with engineered delta vectors → known pair distances."""
    # delta_A − delta_B = (2, 0, 0) → ||·|| = 2.0
    # delta_B − delta_C = (-3, 4, 0) → ||·|| = 5.0
    # delta_C − delta_D = (0, 0, -1) → ||·|| = 1.0
    delta_by_key = {
        "A": np.array([2.0, 0.0, 0.0], dtype=np.float32),
        "B": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "C": np.array([3.0, -4.0, 0.0], dtype=np.float32),
        "D": np.array([3.0, -4.0, 1.0], dtype=np.float32),
    }
    chain = _make_chain(["A", "B", "C", "D"])
    d = chain.to_dict(delta_by_key=delta_by_key)

    assert "edge_potentials" in d
    ep = d["edge_potentials"]
    assert len(ep) == 3
    assert ep[0] == approx_close(2.0)
    assert ep[1] == approx_close(5.0)
    assert ep[2] == approx_close(1.0)


def test_edge_potentials_single_entity_chain_empty_list() -> None:
    chain = _make_chain(["A"])
    d = chain.to_dict(delta_by_key={"A": np.zeros(3, dtype=np.float32)})
    assert d["edge_potentials"] == []


def test_edge_potentials_missing_polygon_emits_none() -> None:
    """When one endpoint at a hop lacks a polygon in delta_by_key,
    that hop's potential is None; surrounding hops compute normally."""
    delta_by_key = {
        "A": np.array([0.0, 0.0], dtype=np.float32),
        # "B" deliberately absent
        "C": np.array([1.0, 0.0], dtype=np.float32),
        "D": np.array([1.0, 1.0], dtype=np.float32),
    }
    chain = _make_chain(["A", "B", "C", "D"])
    d = chain.to_dict(delta_by_key=delta_by_key)

    ep = d["edge_potentials"]
    assert ep[0] is None  # A→B (B missing)
    assert ep[1] is None  # B→C (B missing)
    assert ep[2] == approx_close(1.0)  # C→D


def test_edge_potentials_nan_delta_emits_none() -> None:
    """NaN in either endpoint's delta vector → None for that hop;
    JSON-serialised output must not carry NaN literals."""
    delta_by_key = {
        "A": np.array([0.0, 0.0], dtype=np.float32),
        "B": np.array([float("nan"), 0.0], dtype=np.float32),
        "C": np.array([1.0, 0.0], dtype=np.float32),
    }
    chain = _make_chain(["A", "B", "C"])
    d = chain.to_dict(delta_by_key=delta_by_key)

    ep = d["edge_potentials"]
    assert ep[0] is None  # NaN in B
    assert ep[1] is None  # NaN in B
    # Strict JSON round-trip must succeed
    json.dumps(d, allow_nan=False)


def test_edge_potentials_no_delta_by_key_all_none() -> None:
    """When delta_by_key is not supplied, the field is still emitted —
    all hops report None."""
    chain = _make_chain(["A", "B", "C", "D"])
    d = chain.to_dict()
    assert d["edge_potentials"] == [None, None, None]


def test_edge_potentials_dim_mismatch_emits_none() -> None:
    """Delta vectors of different lengths cannot be subtracted → None."""
    delta_by_key = {
        "A": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "B": np.array([0.0, 0.0], dtype=np.float32),  # different dim
    }
    chain = _make_chain(["A", "B"])
    d = chain.to_dict(delta_by_key=delta_by_key)
    assert d["edge_potentials"] == [None]


def test_edge_potentials_zero_dim_delta_emits_none() -> None:
    """Empty delta vectors (zero-dim pattern) → None per hop."""
    delta_by_key = {
        "A": np.array([], dtype=np.float32),
        "B": np.array([], dtype=np.float32),
    }
    chain = _make_chain(["A", "B"])
    d = chain.to_dict(delta_by_key=delta_by_key)
    assert d["edge_potentials"] == [None]


def test_edge_potentials_inf_delta_emits_none() -> None:
    """Inf in delta vector → None (strict-JSON sanitisation)."""
    delta_by_key = {
        "A": np.array([0.0, 0.0], dtype=np.float32),
        "B": np.array([float("inf"), 0.0], dtype=np.float32),
    }
    chain = _make_chain(["A", "B"])
    d = chain.to_dict(delta_by_key=delta_by_key)
    assert d["edge_potentials"] == [None]
    json.dumps(d, allow_nan=False)


# Lightweight tolerance helper — avoids the pytest.approx import overhead and
# keeps the test file free of additional fixtures.
class _Approx:
    def __init__(self, value: float, abs_tol: float = 1e-6) -> None:
        self.value = float(value)
        self.abs_tol = abs_tol

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        return math.isclose(float(other), self.value, abs_tol=self.abs_tol)

    def __repr__(self) -> str:
        return f"~{self.value}"


def approx_close(value: float, abs_tol: float = 1e-6) -> _Approx:
    return _Approx(value, abs_tol)
