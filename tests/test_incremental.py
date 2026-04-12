# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for incremental geometry updates."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.builder.builder import (
    GDSBuilder,
    IncrementalUpdateResult,
    compute_entity_geometry,
)


@pytest.fixture(scope="module")
def base_sphere(tmp_path_factory):
    """Build a small anchor-only sphere for incremental update tests."""
    tmp = tmp_path_factory.mktemp("incremental")
    out = str(tmp / "gds_inc")

    rng = np.random.default_rng(42)
    n = 100
    custs = pa.table(
        {
            "primary_key": pa.array([f"C{i}" for i in range(n)], type=pa.string()),
            "region": pa.array(
                [rng.choice(["A", "B"]) for _ in range(n)],
                type=pa.string(),
            ),
        }
    )
    events = pa.table(
        {
            "primary_key": pa.array(
                [f"E{i}" for i in range(n * 5)],
                type=pa.string(),
            ),
            "cust_fk": pa.array(
                [f"C{rng.integers(0, n)}" for _ in range(n * 5)],
                type=pa.string(),
            ),
            "amount": pa.array(
                rng.uniform(10, 100, size=n * 5).tolist(),
                type=pa.float64(),
            ),
        }
    )

    builder = GDSBuilder("test_inc", out)
    builder.add_line("customers", custs, key_col="primary_key", source_id="test", role="anchor")
    builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")
    builder.add_derived_dimension(
        "customers",
        "events",
        "cust_fk",
        "count",
        None,
        "event_count",
    )
    builder.add_pattern(
        "cust_pattern",
        "anchor",
        "customers",
        relations=[],
    )
    builder.build()
    return out


def _copy_sphere(base_sphere, tmp_path_factory, name):
    """Clone sphere to a fresh dir so tests don't interfere."""
    from tests.conftest import clone_sphere

    tmp = tmp_path_factory.mktemp(name)
    dest = tmp / "gds_inc"
    clone_sphere(base_sphere, dest)
    return str(dest)


@pytest.fixture
def sphere_for_add(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_add")


@pytest.fixture
def sphere_for_delete(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_delete")


@pytest.fixture
def sphere_for_modify(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_modify")


@pytest.fixture
def sphere_for_drift(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_drift")


def _read_sphere_json(sphere_path):
    return json.loads((Path(sphere_path) / "_gds_meta" / "sphere.json").read_text())


def _get_pattern_meta(sphere_path, pattern_id="cust_pattern"):
    sj = _read_sphere_json(sphere_path)
    return sj["patterns"][pattern_id]


def test_incremental_add_new_entities(sphere_for_add):
    """Add 10 new customers, verify geometry grows."""
    sphere_path = sphere_for_add
    old_pop = _get_pattern_meta(sphere_path)["population_size"]

    # Build entity table with event_count column (matches derived dim)
    new_custs = pa.table(
        {
            "primary_key": pa.array(
                [f"C{100 + i}" for i in range(10)],
                type=pa.string(),
            ),
            "event_count": pa.array([5.0] * 10, type=pa.float64()),
        }
    )

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update("cust_pattern", changed_entities=new_custs)

    assert isinstance(result, IncrementalUpdateResult)
    assert result.added == 10
    assert result.modified == 0
    assert result.deleted == 0
    assert result.population_size == old_pop + 10

    # Verify sphere.json was updated
    new_pop = _get_pattern_meta(sphere_path)["population_size"]
    assert new_pop == old_pop + 10


def test_incremental_delete_entities(sphere_for_delete):
    """Delete 5 entities, verify population shrinks."""
    sphere_path = sphere_for_delete
    old_pop = _get_pattern_meta(sphere_path)["population_size"]

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update(
        "cust_pattern",
        deleted_keys=["C0", "C1", "C2", "C3", "C4"],
    )

    assert result.deleted == 5
    assert result.added == 0
    assert result.modified == 0
    assert result.population_size == old_pop - 5

    # Verify sphere.json was updated
    new_pop = _get_pattern_meta(sphere_path)["population_size"]
    assert new_pop == old_pop - 5


def test_incremental_modify_entities(sphere_for_modify):
    """Modify existing entities, verify geometry updated."""
    sphere_path = sphere_for_modify
    old_pop = _get_pattern_meta(sphere_path)["population_size"]

    modified = pa.table(
        {
            "primary_key": pa.array(["C10", "C11"], type=pa.string()),
            "event_count": pa.array([999.0, 999.0], type=pa.float64()),
        }
    )

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update("cust_pattern", changed_entities=modified)

    assert result.modified == 2
    assert result.added == 0
    # Population size unchanged for modifications
    assert result.population_size == old_pop


def test_incremental_returns_drift_pct(sphere_for_drift):
    """Verify drift_pct is reported."""
    sphere_path = sphere_for_drift

    new_custs = pa.table(
        {
            "primary_key": pa.array(["C200"], type=pa.string()),
            "event_count": pa.array([5.0], type=pa.float64()),
        }
    )

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update("cust_pattern", changed_entities=new_custs)

    assert isinstance(result.drift_pct, float)
    assert result.drift_pct >= 0.0
    assert isinstance(result.theta_norm, float)
    assert result.theta_norm > 0.0


# ── Unit tests for helpers ──


def test_compute_entity_geometry_basic():
    """Test compute_entity_geometry with simple relation metadata."""
    entity_table = pa.table(
        {
            "primary_key": pa.array(["A", "B", "C"], type=pa.string()),
            "fk_line1": pa.array(["X", "", "Y"], type=pa.string()),
        }
    )
    mu = np.array([0.5], dtype=np.float32)
    sigma = np.array([0.3], dtype=np.float32)
    relations = [{"line_id": "line1", "direction": "in", "fk_col": "fk_line1"}]

    deltas, norms, shapes = compute_entity_geometry(
        entity_table,
        mu,
        sigma,
        relations,
    )

    assert deltas.shape == (3, 1)
    assert norms.shape == (3,)
    assert shapes.shape == (3, 1)
    # A has FK "X" → shape 1.0, B has "" → shape 0.0, C has "Y" → shape 1.0
    assert shapes[0, 0] == 1.0
    assert shapes[1, 0] == 0.0
    assert shapes[2, 0] == 1.0
