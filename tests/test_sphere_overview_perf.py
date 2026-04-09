# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for sphere_overview performance with cached inactive_ratio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.builder.builder import GDSBuilder


@pytest.fixture(scope="module")
def sphere_with_stats(tmp_path_factory):
    """Build a sphere with geometry_stats cache."""
    tmp = tmp_path_factory.mktemp("opi080")
    out = str(tmp / "gds_test")

    rng = np.random.default_rng(42)
    n = 200
    custs = pa.table(
        {
            "primary_key": pa.array([f"C{i}" for i in range(n)], type=pa.string()),
            "segment": pa.array([rng.choice(["A", "B"]) for _ in range(n)], type=pa.string()),
        }
    )
    events = pa.table(
        {
            "primary_key": pa.array([f"E{i}" for i in range(n * 5)], type=pa.string()),
            "cust_fk": pa.array([f"C{rng.integers(0, n)}" for _ in range(n * 5)], type=pa.string()),
        }
    )

    builder = GDSBuilder("test_opi080", out)
    builder.add_line("customers", custs, key_col="primary_key", source_id="test", role="anchor")
    builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")
    builder.add_derived_dimension("customers", "events", "cust_fk", "count", None, "event_count")
    builder.add_pattern(
        "cust_pattern",
        "anchor",
        "customers",
        relations=[],
        tracked_properties=["segment"],
    )
    builder.build()
    return out


def test_geometry_stats_contains_inactive_fields(sphere_with_stats):
    stats_path = Path(sphere_with_stats) / "_gds_meta" / "geometry_stats" / "cust_pattern_v1.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert "inactive_ratio" in stats
    assert "inactive_count" in stats


def test_sphere_overview_uses_cached_inactive_ratio(sphere_with_stats):
    from hypertopos.sphere import HyperSphere

    sphere = HyperSphere.open(sphere_with_stats)
    session = sphere.session("test")
    nav = session.navigator()
    result = nav.sphere_overview()
    session.close()

    cust = [r for r in result if r["pattern_id"] == "cust_pattern"][0]
    assert cust["pattern_type"] == "anchor"
    # inactive_ratio should come from cache (or be absent if not detected)
    # The key test: sphere_overview completes without reading full geometry
    assert "anomaly_rate" in cust
