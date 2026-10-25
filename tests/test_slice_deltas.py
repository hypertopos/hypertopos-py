# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""One temporal slice-delta computation shared by every temporal consumer."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from hypertopos.engine.geometry import slice_deltas
from hypertopos.model.sphere import Pattern, RelationDef


def _pattern(
    *,
    cholesky_inv: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> Pattern:
    return Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=[RelationDef("a", "in", True), RelationDef("b", "in", True)],
        mu=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        sigma_diag=np.array([0.5, 0.001, 2.0], dtype=np.float32),
        theta=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        population_size=10,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        dimension_weights=weights,
        cholesky_inv=cholesky_inv,
    )


MU = np.array([1.0, 2.0, 3.0])
SIGMA = np.array([0.5, 0.001, 2.0])
SHAPES = np.array([[2.0, 2.5, 5.0], [0.0, 2.0, 3.0]])
L_INV = np.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [-1.0, 0.25, 0.5]])


def test_matches_build_solid_diag() -> None:
    sigma = np.maximum(np.array([0.5, 0.001, 2.0]), 1e-2)
    expected = (SHAPES - [1.0, 2.0, 3.0]) / sigma
    np.testing.assert_allclose(slice_deltas(SHAPES, _pattern()), expected, rtol=1e-6)


def test_matches_build_solid_mahalanobis() -> None:
    expected = (SHAPES - [1.0, 2.0, 3.0]) @ L_INV.T
    got = slice_deltas(SHAPES, _pattern(cholesky_inv=L_INV))
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_applies_dimension_weights() -> None:
    weights = np.array([2.0, 0.0, 0.5])
    expected = (SHAPES - [1.0, 2.0, 3.0]) @ L_INV.T * weights
    got = slice_deltas(SHAPES, _pattern(cholesky_inv=L_INV, weights=weights))
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_chunked_trajectory_uses_pattern_metric() -> None:
    from hypertopos.builder.builder import _build_traj_chunk

    tensor = np.stack([SHAPES, SHAPES[::-1]])[None, :, 0, :].astype(np.float32)  # (1, 2, 3)
    weights = np.array([2.0, 1.0, 0.5])
    tables: list = []
    ts = [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)]
    _build_traj_chunk(tables, tensor, ["k"], ts, MU, SIGMA, L_INV, weights)
    deltas = (tensor[0] - MU) @ L_INV.T * weights
    expected = np.concatenate([deltas.mean(axis=0), deltas.std(axis=0)])
    got = np.array(tables[0]["trajectory_vector"][0].as_py())
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_float32_input_stays_float32() -> None:
    # Build-time tensors are float32; promoting them to float64 multiplies peak memory.
    tensor = np.ones((4, 3, 3), dtype=np.float32)
    out = slice_deltas(tensor, _pattern(cholesky_inv=L_INV, weights=np.ones(3)))
    assert out.dtype == np.float32
    assert out.shape == (4, 3, 3)


def test_narrow_snapshot_clamps() -> None:
    narrow = SHAPES[:, :2]
    weights = np.array([2.0, 3.0, 4.0])
    expected = (narrow - [1.0, 2.0]) @ L_INV[:2, :2].T * weights[:2]
    got = slice_deltas(narrow, _pattern(cholesky_inv=L_INV, weights=weights))
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
