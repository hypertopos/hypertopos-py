# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Shape reconstruction from stored deltas inverts the stored metric."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa

from hypertopos.engine.geometry import shapes_from_deltas, slice_deltas
from hypertopos.model.sphere import Pattern, RelationDef
from hypertopos.navigation.navigator import GDSNavigator

MU = np.array([0.5, 0.4, 0.3])
SIGMA = np.array([0.2, 0.001, 0.25])
# Non-diagonal lower-triangular whitening: a diagonal one equal to 1/sigma
# would make the naive ``delta * sigma + mu`` inversion look correct.
L_INV = np.array([[4.0, 0.0, 0.0], [-1.5, 3.0, 0.0], [1.0, 0.75, 2.5]])
SHAPES = np.array([[0.6, 0.4, 0.2], [0.2, 0.2, 0.9], [0.4, 0.6, 0.0], [0.0, 0.0, 0.5]])


def _pattern(
    *,
    cholesky_inv: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    edge_max: np.ndarray | None = None,
) -> Pattern:
    return Pattern(
        pattern_id="p",
        entity_type="e",
        pattern_type="anchor",
        relations=[
            RelationDef("customers", "out", True),
            RelationDef("products", "out", False),
            RelationDef("stores", "out", False),
        ],
        mu=MU.astype(np.float32),
        sigma_diag=SIGMA.astype(np.float32),
        theta=np.ones(3, dtype=np.float32),
        population_size=4,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=edge_max,
        dimension_weights=weights,
        cholesky_inv=cholesky_inv,
    )


def test_round_trip_diag() -> None:
    p = _pattern()
    np.testing.assert_allclose(shapes_from_deltas(slice_deltas(SHAPES, p), p), SHAPES, atol=1e-6)


def test_round_trip_mahalanobis() -> None:
    p = _pattern(cholesky_inv=L_INV)
    np.testing.assert_allclose(shapes_from_deltas(slice_deltas(SHAPES, p), p), SHAPES, atol=1e-6)


def test_round_trip_mahalanobis_weighted() -> None:
    p = _pattern(cholesky_inv=L_INV, weights=np.array([2.0, 0.5, 1.5]))
    np.testing.assert_allclose(shapes_from_deltas(slice_deltas(SHAPES, p), p), SHAPES, atol=1e-6)


def test_round_trip_diag_weighted() -> None:
    p = _pattern(weights=np.array([2.0, 0.5, 1.5]))
    np.testing.assert_allclose(shapes_from_deltas(slice_deltas(SHAPES, p), p), SHAPES, atol=1e-6)


def test_round_trip_narrow_width() -> None:
    p = _pattern(cholesky_inv=L_INV, weights=np.array([2.0, 0.5, 1.5]))
    narrow = SHAPES[:, :2]
    got = shapes_from_deltas(slice_deltas(narrow, p), p)
    assert got.shape == (4, 2)
    np.testing.assert_allclose(got, narrow, atol=1e-6)


def test_one_dimensional_input_stays_one_dimensional() -> None:
    p = _pattern(cholesky_inv=L_INV, weights=np.array([2.0, 0.5, 1.5]))
    delta = slice_deltas(SHAPES[1], p)[0]
    got = shapes_from_deltas(delta, p)
    assert got.shape == (3,)
    np.testing.assert_allclose(got, SHAPES[1], atol=1e-6)


def test_zero_weight_coordinate_falls_back_to_mu() -> None:
    p = _pattern(weights=np.array([2.0, 0.0, 1.5]))
    got = shapes_from_deltas(slice_deltas(SHAPES, p), p)
    np.testing.assert_allclose(got[:, 1], MU[1], atol=1e-6)
    np.testing.assert_allclose(got[:, [0, 2]], SHAPES[:, [0, 2]], atol=1e-6)


def test_float32_input_stays_float32() -> None:
    p = _pattern(cholesky_inv=L_INV, weights=np.ones(3))
    out = shapes_from_deltas(np.ones((4, 3), dtype=np.float32), p)
    assert out.dtype == np.float32


def test_hub_scores_on_mahalanobis_pattern_use_true_shapes() -> None:
    edge_max = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    p = _pattern(cholesky_inv=L_INV, weights=np.array([2.0, 0.5, 1.5]), edge_max=edge_max)
    deltas = slice_deltas(SHAPES, p).astype(np.float32)
    keys = ["K-001", "K-002", "K-003", "K-004"]
    table = pa.table(
        {
            "primary_key": keys,
            "delta": pa.array(deltas.tolist(), type=pa.list_(pa.float32())),
        }
    )
    sphere = MagicMock()
    sphere.patterns = {"p": p}
    storage = MagicMock()
    storage.read_geometry.return_value = table
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1
    nav = GDSNavigator(engine=MagicMock(), storage=storage, manifest=manifest, contract=MagicMock())

    expected = dict(zip(keys, (np.clip(SHAPES, 0.0, 1.0) * edge_max).sum(axis=1), strict=True))
    results = nav.π7_attract_hub("p", top_n=4)
    got = {k: score for k, _, score in results}
    for k in keys:
        np.testing.assert_allclose(got[k], expected[k], atol=1e-4)

    stats = nav.π7_attract_hub_and_stats("p")[1]
    np.testing.assert_allclose(stats["max"], max(expected.values()), atol=1e-3)


def test_attribution_on_mahalanobis_pattern_is_share_of_stored_delta_sq() -> None:
    """Under Cholesky + weights, attribution is the share of the stored
    (whitened) delta's squared norm — the same approximation ``witness``
    and ``repair`` make — whatever the dims' kind tags."""
    from hypertopos.engine.geometry import compute_reliability_flags
    from hypertopos.engine.investigation import build_explanation

    kinds = ["gaussian", "poisson", "gaussian"]
    p = _pattern(cholesky_inv=L_INV, weights=np.array([2.0, 0.5, 1.5]))
    p.dimension_kinds = kinds
    delta = slice_deltas(SHAPES[1], p)[0].astype(np.float64)
    sq = delta**2
    shares = sq / sq.sum()

    flags = compute_reliability_flags(delta, pattern=p)
    np.testing.assert_allclose(flags["dominant_dim_share"], shares.max(), atol=1e-6)
    assert flags["dominant_dim"] == p.dim_labels[int(np.argmax(sq))]

    expl = build_explanation(
        delta,
        p.dim_labels,
        theta_norm=0.1,
        delta_norm=float(np.linalg.norm(delta)),
        dimension_kinds=kinds,
    )
    got = {d["dim"]: d["pct_of_total"] for d in expl["top_dimensions"]}
    expected = dict(zip(p.dim_labels, shares, strict=True))
    for label, value in got.items():
        np.testing.assert_allclose(value, expected[label], atol=1e-4)
