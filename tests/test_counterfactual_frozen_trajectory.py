# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Counterfactual frozen-population trajectory tests.

Synthetic 3-slice solid:
    - Slice 0: shape = [1.0, 1.0]
    - Slice 1: shape = [1.0, 1.0]   (entity standing still)
    - Slice 2: shape = [1.0, 1.0]   (entity standing still)

Pattern mu drifts (the calibrated mu sits FAR from slice-0's shape) so the
default ``delta_norm_snapshot`` trajectory looks anomalous, while the
frozen-trajectory ``delta_norm_frozen_pop`` collapses to zero for every slice
— the canonical "is it real movement or population drift?" signal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
from hypertopos.engine.counterfactual import (
    recompute_delta_norm_against_frozen,
)
from hypertopos.engine.geometry import GDSEngine
from hypertopos.model.manifest import Manifest
from hypertopos.model.sphere import Pattern, RelationDef, Sphere
from hypertopos.storage.cache import GDSCache

UTC = timezone.utc  # noqa: UP017


def _make_pattern(mu: np.ndarray, sigma_diag: np.ndarray) -> Pattern:
    now = datetime(2024, 6, 1, tzinfo=UTC)
    return Pattern(
        pattern_id="customer_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="products", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=mu,
        sigma_diag=sigma_diag,
        theta=np.array([3.0, 3.0], dtype=np.float32),
        population_size=50,
        computed_at=now,
        version=1,
        status="production",
    )


def _make_engine(
    pattern: Pattern, shape_per_slice: list[np.ndarray]
) -> tuple[GDSEngine, Manifest]:
    now = datetime(2024, 6, 1, tzinfo=UTC)
    timestamps = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        datetime(2024, 3, 1, tzinfo=UTC),
    ]
    temporal_table = pa.table(
        {
            "slice_index": pa.array(
                list(range(len(shape_per_slice))), type=pa.int64(),
            ),
            "timestamp": pa.array(
                timestamps[: len(shape_per_slice)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "deformation_type": pa.array(["structural"] * len(shape_per_slice)),
            "shape_snapshot": pa.array([s.tolist() for s in shape_per_slice]),
            "pattern_ver": pa.array([1] * len(shape_per_slice), type=pa.int64()),
            "changed_property": pa.array([None] * len(shape_per_slice), type=pa.null()),
            "changed_line_id": pa.array([None] * len(shape_per_slice), type=pa.null()),
        }
    )
    edge_type = pa.list_(
        pa.struct(
            [
                pa.field("line_id", pa.string()),
                pa.field("point_key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("direction", pa.string()),
            ]
        )
    )
    geometry_table = pa.table(
        {
            "primary_key": pa.array(["CUST-X"]),
            "pattern_id": pa.array(["customer_pattern"]),
            "pattern_ver": pa.array([1], type=pa.int64()),
            "pattern_type": pa.array(["anchor"]),
            "scale": pa.array([1], type=pa.int64()),
            "delta": pa.array([[0.5, 0.5]]),
            "delta_norm": pa.array([0.7], type=pa.float64()),
            "is_anomaly": pa.array([False]),
            "edges": pa.array([[]], type=edge_type),
            "version": pa.array([1], type=pa.int64()),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        }
    )

    sphere = Sphere(
        sphere_id="test",
        name="test",
        base_path=".",
        patterns={"customer_pattern": pattern},
    )

    mock_storage = MagicMock()
    mock_storage.read_temporal.return_value = temporal_table
    mock_storage.read_geometry.return_value = geometry_table
    mock_storage.read_sphere.return_value = sphere

    engine = GDSEngine(storage=mock_storage, cache=GDSCache())
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="test-agent",
        snapshot_time=now,
        status="active",
        line_versions={},
        pattern_versions={"customer_pattern": 1},
    )
    return engine, manifest


def test_default_path_leaves_delta_norm_frozen_pop_none():
    """``counterfactual_frozen_population=False`` (default) keeps slice
    ``delta_norm_frozen_pop`` at ``None`` — backward-compatible default."""
    mu = np.array([0.0, 0.0], dtype=np.float32)
    sigma = np.array([1.0, 1.0], dtype=np.float32)
    shapes = [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([2.0, 2.0], dtype=np.float32),
        np.array([3.0, 3.0], dtype=np.float32),
    ]
    pattern = _make_pattern(mu, sigma)
    engine, manifest = _make_engine(pattern, shapes)

    solid = engine.build_solid("CUST-X", "customer_pattern", manifest)
    assert len(solid.slices) == 3
    for slc in solid.slices:
        assert slc.delta_norm_frozen_pop is None


def test_frozen_path_zero_for_stationary_entity_in_drifting_population():
    """Entity at constant shape across slices yields
    ``delta_norm_frozen_pop = 0`` for every slice while
    ``delta_norm_snapshot`` is non-zero (current pattern's mu is far from
    the entity's stationary state)."""
    # Pattern mu = [0.0, 0.0] sits FAR from the entity's stationary shape.
    mu = np.array([0.0, 0.0], dtype=np.float32)
    sigma = np.array([1.0, 1.0], dtype=np.float32)
    stationary_shape = np.array([5.0, 5.0], dtype=np.float32)
    shapes = [stationary_shape.copy() for _ in range(3)]
    pattern = _make_pattern(mu, sigma)
    engine, manifest = _make_engine(pattern, shapes)

    solid = engine.build_solid(
        "CUST-X",
        "customer_pattern",
        manifest,
        counterfactual_frozen_population=True,
    )
    assert len(solid.slices) == 3

    # Default delta_norm_snapshot reflects current-pop normalisation —
    # non-zero because the entity sits far from current mu.
    for slc in solid.slices:
        assert slc.delta_norm_snapshot > 0.0

    # Frozen trajectory: stationary entity, slice-0 IS the reference, every
    # slice's frozen delta_norm collapses to zero.
    for slc in solid.slices:
        assert slc.delta_norm_frozen_pop is not None
        assert slc.delta_norm_frozen_pop == 0.0


def test_frozen_path_diverges_from_current_pop_when_entity_moves():
    """When the entity moves between slices the frozen-trajectory measures
    the per-slice distance from the FIRST slice's raw shape — diverges from
    the default trajectory because the reference is different."""
    mu = np.array([0.5, 0.5], dtype=np.float32)
    sigma = np.array([1.0, 1.0], dtype=np.float32)
    shapes = [
        np.array([1.0, 1.0], dtype=np.float32),  # slice 0 — frozen reference
        np.array([1.0, 2.0], dtype=np.float32),  # slice 1
        np.array([2.0, 3.0], dtype=np.float32),  # slice 2
    ]
    pattern = _make_pattern(mu, sigma)
    engine, manifest = _make_engine(pattern, shapes)

    solid = engine.build_solid(
        "CUST-X",
        "customer_pattern",
        manifest,
        counterfactual_frozen_population=True,
    )
    assert len(solid.slices) == 3

    # Slice 0 against itself → zero.
    assert solid.slices[0].delta_norm_frozen_pop == 0.0
    # Slice 1 raw delta from slice-0 reference: [0.0, 1.0] / [1, 1] → norm 1.0
    assert solid.slices[1].delta_norm_frozen_pop is not None
    np.testing.assert_allclose(solid.slices[1].delta_norm_frozen_pop, 1.0, rtol=1e-5)
    # Slice 2 raw delta from slice-0 reference: [1.0, 2.0] / [1, 1] → norm sqrt(5)
    np.testing.assert_allclose(
        solid.slices[2].delta_norm_frozen_pop, float(np.sqrt(5.0)), rtol=1e-5,
    )

    # The default ``delta_norm_snapshot`` trajectory uses the current pattern
    # mu — different from the frozen trajectory by construction.
    snap_vs_frozen_diverges = any(
        abs(slc.delta_norm_snapshot - (slc.delta_norm_frozen_pop or 0.0)) > 1e-6
        for slc in solid.slices
    )
    assert snap_vs_frozen_diverges


def test_recompute_delta_norm_against_frozen_helper():
    """Pure-math helper computes L2 norm of ``(shape - mu_frozen) / sigma``."""
    shape = np.array([2.0, 4.0], dtype=np.float32)
    mu_frozen = np.array([1.0, 1.0], dtype=np.float32)
    sigma = np.array([1.0, 3.0], dtype=np.float32)
    # delta = [(2-1)/1, (4-1)/3] = [1, 1] → norm sqrt(2)
    result = recompute_delta_norm_against_frozen(
        shape=shape, mu_frozen=mu_frozen, sigma=sigma,
    )
    np.testing.assert_allclose(result, float(np.sqrt(2.0)), rtol=1e-5)


def test_recompute_delta_norm_handles_sigma_dead_dims():
    """Sigma-dead dims (sigma < 1e-10) contribute zero rather than blowing up."""
    shape = np.array([2.0, 4.0], dtype=np.float32)
    mu_frozen = np.array([1.0, 1.0], dtype=np.float32)
    sigma = np.array([1.0, 0.0], dtype=np.float32)  # second dim is dead
    # Only first dim contributes: (2-1)/1 = 1 → norm 1.0
    result = recompute_delta_norm_against_frozen(
        shape=shape, mu_frozen=mu_frozen, sigma=sigma,
    )
    np.testing.assert_allclose(result, 1.0, rtol=1e-5)


def test_serializer_round_trips_delta_norm_frozen_pop():
    """MCP serializer emits ``delta_norm_frozen_pop`` when populated, omits
    when None (default path)."""
    from hypertopos_mcp.serializers import _serialize_slice

    mu = np.array([0.0, 0.0], dtype=np.float32)
    sigma = np.array([1.0, 1.0], dtype=np.float32)
    shapes = [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.0, 3.0], dtype=np.float32),
    ]
    pattern = _make_pattern(mu, sigma)
    engine, manifest = _make_engine(pattern, shapes)

    solid_default = engine.build_solid("CUST-X", "customer_pattern", manifest)
    serialised_default = _serialize_slice(solid_default.slices[0])
    assert "delta_norm_frozen_pop" not in serialised_default

    solid_frozen = engine.build_solid(
        "CUST-X",
        "customer_pattern",
        manifest,
        counterfactual_frozen_population=True,
    )
    serialised_frozen_first = _serialize_slice(solid_frozen.slices[0])
    serialised_frozen_second = _serialize_slice(solid_frozen.slices[1])
    assert serialised_frozen_first.get("delta_norm_frozen_pop") == 0.0
    # Slice 1 raw delta from slice-0 raw shape [1,1]: [0, 2] / [1,1] → norm 2.0
    np.testing.assert_allclose(
        serialised_frozen_second["delta_norm_frozen_pop"], 2.0, rtol=1e-5,
    )
