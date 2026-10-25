# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Every stored delta is computed in the metric saved for its pattern.

Invariant: ``stored delta == calibrated_deltas(shape, pattern as saved in
sphere.json)`` after a full build, an incremental update and a recalibration.
Prop-column-only patterns are used so the true shape (column validity, 0/1)
is known independently of any stored delta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.builder.builder import GDSBuilder
from hypertopos.engine.geometry import slice_deltas
from hypertopos.model.sphere import Pattern
from hypertopos.storage.reader import GDSReader

PID = "item_pattern"
PROPS = ["p_a", "p_b", "p_c"]


def _prop_table(keys: list[str], seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    n = len(keys)
    base = rng.random(n)
    # Correlated validity across columns so the covariance is not diagonal.
    valid = {
        "p_a": base < 0.6,
        "p_b": (base < 0.5) | (rng.random(n) < 0.15),
        "p_c": (base > 0.3) & (rng.random(n) < 0.8),
    }
    cols: dict[str, pa.Array] = {"id": pa.array(keys, type=pa.string())}
    for name in PROPS:
        cols[name] = pa.array(
            [1.0 if v else None for v in valid[name]],
            type=pa.float64(),
        )
    return pa.table(cols)


def _build(
    tmp_path: Path,
    *,
    use_mahalanobis: bool = False,
    weights: list[float] | None = None,
    anomaly_percentile: float = 95.0,
) -> tuple[str, pa.Table]:
    table = _prop_table([f"e_{i}" for i in range(80)], seed=3)
    builder = GDSBuilder(sphere_id="metric_test", output_path=str(tmp_path / "metric_test"))
    builder.add_line("items", table, key_col="id", source_id="test")
    builder.add_pattern(
        PID,
        pattern_type="anchor",
        entity_line="items",
        relations=[],
        tracked_properties=PROPS,
        use_mahalanobis=use_mahalanobis,
        dimension_weights=weights,
        bootstrap_iterations=0,
        anomaly_percentile=anomaly_percentile,
    )
    return builder.build(), table


def _pattern(sphere_path: str) -> Pattern:
    return GDSReader(sphere_path).read_sphere().patterns[PID]


def _shapes(table: pa.Table, pattern: Pattern) -> dict[str, np.ndarray]:
    assert len(pattern.relations) == 0
    cols = [
        np.array(table[c].is_valid().to_pylist(), dtype=np.float64) for c in pattern.prop_columns
    ]
    mat = np.stack(cols, axis=1)
    return {k: mat[i] for i, k in enumerate(table["id"].to_pylist())}


def _stored(sphere_path: str, pattern: Pattern) -> dict[str, tuple[np.ndarray, float, bool]]:
    geo = GDSReader(sphere_path).read_geometry(PID, pattern.version)
    out = {}
    for pk, delta, norm, anom in zip(
        geo["primary_key"].to_pylist(),
        geo["delta"].to_pylist(),
        geo["delta_norm"].to_pylist(),
        geo["is_anomaly"].to_pylist(),
        strict=True,
    ):
        out[pk] = (np.asarray(delta, dtype=np.float64), float(norm), bool(anom))
    return out


def _assert_consistent(sphere_path: str, shapes: dict[str, np.ndarray]) -> None:
    pattern = _pattern(sphere_path)
    stored = _stored(sphere_path, pattern)
    keys = sorted(shapes)
    expected = slice_deltas(np.stack([shapes[k] for k in keys]), pattern)
    got = np.stack([stored[k][0] for k in keys])
    np.testing.assert_allclose(got, expected, atol=1e-4)
    norms = np.array([stored[k][1] for k in keys])
    np.testing.assert_allclose(norms, np.linalg.norm(got, axis=1), atol=1e-4)
    anom = np.array([stored[k][2] for k in keys])
    np.testing.assert_array_equal(anom, norms >= pattern.theta_norm)


def test_build_prop_columns_with_mahalanobis_stores_saved_metric(tmp_path: Path) -> None:
    sphere_path, table = _build(tmp_path, use_mahalanobis=True)
    pattern = _pattern(sphere_path)
    assert pattern.cholesky_inv is not None
    _assert_consistent(sphere_path, _shapes(table, pattern))


@pytest.mark.parametrize(
    ("use_mahalanobis", "weights"),
    [(True, None), (False, [2.0, 0.5, 1.5]), (True, [2.0, 0.5, 1.5])],
)
def test_incremental_update_stores_saved_metric(
    tmp_path: Path, use_mahalanobis: bool, weights: list[float] | None
) -> None:
    sphere_path, table = _build(tmp_path, use_mahalanobis=use_mahalanobis, weights=weights)
    new = _prop_table([f"n_{i}" for i in range(12)], seed=11).rename_columns(
        ["primary_key", *PROPS]
    )
    GDSBuilder("metric_test", sphere_path).incremental_update(
        PID, changed_entities=new, recalibrate="never"
    )
    pattern = _pattern(sphere_path)
    shapes = _shapes(new.rename_columns(["id", *PROPS]), pattern)
    shapes.update(_shapes(table, pattern))
    _assert_consistent(sphere_path, shapes)


@pytest.mark.parametrize(
    ("use_mahalanobis", "weights"),
    [(True, None), (False, [2.0, 0.5, 1.5]), (True, [2.0, 0.5, 1.5])],
)
def test_recalibrate_stores_saved_metric(
    tmp_path: Path, use_mahalanobis: bool, weights: list[float] | None
) -> None:
    from hypertopos.sphere import HyperSphere

    sphere_path, table = _build(tmp_path, use_mahalanobis=use_mahalanobis, weights=weights)
    before = _pattern(sphere_path)
    shapes = _shapes(table, before)
    HyperSphere.open(sphere_path).session("test").recalibrate(PID)
    after = _pattern(sphere_path)
    assert (after.cholesky_inv is not None) == use_mahalanobis
    if weights is not None:
        assert after.dimension_weights is not None
        np.testing.assert_allclose(after.dimension_weights, weights)
    _assert_consistent(sphere_path, shapes)


def test_recalibrate_uses_pattern_anomaly_percentile(tmp_path: Path) -> None:
    from hypertopos.sphere import HyperSphere

    sphere_path, _ = _build(tmp_path, anomaly_percentile=70.0)
    HyperSphere.open(sphere_path).session("test").recalibrate(PID)
    pattern = _pattern(sphere_path)
    norms = np.array([v[1] for v in _stored(sphere_path, pattern).values()])
    assert pattern.theta_norm == pytest.approx(float(np.percentile(norms, 70.0)), rel=1e-4)


def test_incremental_bregman_matches_build_on_weighted_pattern(tmp_path: Path) -> None:
    sphere_path, table = _build(tmp_path, weights=[2.0, 0.5, 1.5])
    geo = GDSReader(sphere_path).read_geometry(PID, _pattern(sphere_path).version)
    built = dict(
        zip(geo["primary_key"].to_pylist(), geo["bregman_divergence"].to_pylist(), strict=True)
    )
    keys = [f"e_{i}" for i in range(10)]
    again = table.filter(pa.compute.is_in(table["id"], pa.array(keys))).rename_columns(
        ["primary_key", *PROPS]
    )
    GDSBuilder("metric_test", sphere_path).incremental_update(
        PID, changed_entities=again, recalibrate="never"
    )
    geo = GDSReader(sphere_path).read_geometry(PID, _pattern(sphere_path).version)
    after = dict(
        zip(geo["primary_key"].to_pylist(), geo["bregman_divergence"].to_pylist(), strict=True)
    )
    assert any(built[k] > 0 for k in keys)
    for k in keys:
        assert after[k] == pytest.approx(built[k], rel=1e-4, abs=1e-5)
