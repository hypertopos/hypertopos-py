# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for GDSNavigator.vector_index_health and the stale-index alert.

Metadata-only: builds small synthetic anchor spheres (no copytree, no heavy
scan) that are large enough (n >= 256) to carry a real IVF_FLAT index on the
geometry delta column, then reads Lance index metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import lance
import pyarrow as pa
from hypertopos.engine.geometry import GDSEngine
from hypertopos.model.manifest import Contract, Manifest
from hypertopos.navigation.navigator import GDSNavigator
from hypertopos.storage.cache import GDSCache
from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter
from tests.test_incremental import _build_anchor_sphere


def _build_indexed_sphere(out_root, n=300):
    """Build a synthetic anchor sphere and force the IVF index on geometry
    delta (the builder skips it for tiny synthetic spheres). Returns the path
    with a real IVF_FLAT index covering all *n* rows."""
    sphere = _build_anchor_sphere(out_root, n)
    GDSWriter(sphere).build_index_if_needed("cust_pattern", version=1)
    return sphere


def _navigator(sphere_path: str) -> GDSNavigator:
    reader = GDSReader(base_path=sphere_path)
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id="m-test",
        agent_id="a-test",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1, "events": 1},
        pattern_versions={"cust_pattern": 1},
    )
    return GDSNavigator(
        engine=engine,
        storage=reader,
        manifest=manifest,
        contract=Contract("m-test", ["cust_pattern"]),
    )


def test_vector_index_health_fully_indexed(tmp_path):
    """A freshly built sphere with an IVF index covers all rows: not stale."""
    sphere = _build_indexed_sphere(tmp_path / "gds_full", 300)
    nav = _navigator(sphere)

    health = nav.vector_index_health("cust_pattern")

    assert health["pattern_id"] == "cust_pattern"
    assert health["index_present"] is True
    assert health["index_type"] == "IVF_FLAT"
    assert health["total_rows"] == 300
    assert health["num_indexed_rows"] == 300
    assert health["num_unindexed_rows"] == 0
    assert health["indexed_fraction"] == 1.0
    assert health["is_stale"] is False
    # num_partitions is sourced from index_statistics (not describe_indices).
    assert isinstance(health["num_partitions"], int)
    assert health["num_partitions"] > 0
    assert "covers all rows" in health["recommendation"]


def test_vector_index_health_stale_when_rows_appended_without_reindex(tmp_path):
    """Engineered staleness: append ~13% new rows to the geometry dataset
    WITHOUT reindexing. The IVF index still covers only the original 300, so
    the unindexed fraction (40/340 ≈ 11.8%) exceeds the 0.1 threshold."""
    sphere = _build_indexed_sphere(tmp_path / "gds_stale", 300)
    geo_path = f"{sphere}/geometry/cust_pattern/data.lance"
    ds = lance.dataset(geo_path)

    # Clone 40 existing rows under fresh primary keys; schema is preserved so
    # no IVF reindex is triggered by the raw append.
    head = ds.to_table().slice(0, 40)
    new_keys = pa.array([f"NEW{i}" for i in range(40)], type=pa.string())
    head = head.set_column(
        head.schema.get_field_index("primary_key"), "primary_key", new_keys,
    )
    ds.insert(head)

    nav = _navigator(sphere)
    health = nav.vector_index_health("cust_pattern")

    assert health["total_rows"] == 340
    assert health["num_indexed_rows"] == 300
    assert health["num_unindexed_rows"] == 40
    assert health["is_stale"] is True
    assert health["indexed_fraction"] < 1.0
    assert "outside the IVF index" in health["recommendation"]


def test_vector_index_health_missing_pattern(tmp_path):
    """Absent geometry dataset → index_present False, zero counts, no crash."""
    sphere = _build_anchor_sphere(tmp_path / "gds_missing", 300)
    nav = _navigator(sphere)

    health = nav.vector_index_health("nonexistent_pattern")

    assert health["index_present"] is False
    assert health["total_rows"] == 0
    assert health["is_stale"] is False
    assert "nothing to index" in health["recommendation"]


def test_vector_index_health_no_index_present_is_not_stale(tmp_path):
    """A populated geometry dataset that carries NO IVF index is NOT stale.

    Regression: previously ``index_present=False`` drove ``unindexed_fraction``
    to 1.0 and flagged ``is_stale=True`` — but with no index, ANN tools fall
    back to a full flat scan that sees every row, so nothing is missed and the
    index cannot be stale. ``_build_anchor_sphere`` leaves the geometry dataset
    un-indexed (the builder skips IVF on tiny synthetic spheres), so the
    geometry exists with 300 rows yet ``index_present`` is False."""
    sphere = _build_anchor_sphere(tmp_path / "gds_no_index", 300)
    nav = _navigator(sphere)

    health = nav.vector_index_health("cust_pattern")

    assert health["index_present"] is False
    assert health["total_rows"] == 300
    # Every row is "unindexed" by the fraction accounting, but is_stale must
    # remain False because a full flat scan covers the whole population.
    assert health["num_unindexed_rows"] == 300
    assert health["is_stale"] is False
    assert "no IVF index" in health["recommendation"]


def test_vector_index_health_line_id_ignored(tmp_path):
    """line_id is informational — it is echoed back but does not change the
    pattern-scoped index that is inspected."""
    sphere = _build_anchor_sphere(tmp_path / "gds_line", 300)
    nav = _navigator(sphere)

    with_line = nav.vector_index_health("cust_pattern", line_id="customers")
    without_line = nav.vector_index_health("cust_pattern")

    assert with_line["line_id"] == "customers"
    assert with_line["num_indexed_rows"] == without_line["num_indexed_rows"]
    assert with_line["total_rows"] == without_line["total_rows"]


def test_check_alerts_emits_stale_index_alert_end_to_end(tmp_path):
    """Integration: a sphere with rows appended outside the IVF index makes
    check_alerts() surface a MEDIUM stale_vector_index alert."""
    sphere = _build_indexed_sphere(tmp_path / "gds_alert", 300)
    geo_path = f"{sphere}/geometry/cust_pattern/data.lance"
    ds = lance.dataset(geo_path)
    head = ds.to_table().slice(0, 40)
    new_keys = pa.array([f"NEW{i}" for i in range(40)], type=pa.string())
    head = head.set_column(
        head.schema.get_field_index("primary_key"), "primary_key", new_keys,
    )
    ds.insert(head)

    nav = _navigator(sphere)
    result = nav.check_alerts("cust_pattern")
    stale = [
        a for a in result["alerts"] if a["check_type"] == "stale_vector_index"
    ]
    assert len(stale) == 1
    assert stale[0]["severity"] == "MEDIUM"
    assert stale[0]["details"]["num_unindexed_rows"] == 40


class TestStaleVectorIndexAlert:
    """Unit-test the alert helper in isolation (MagicMock navigator, no sphere)."""

    def _nav(self, health: dict):
        nav = MagicMock(spec=GDSNavigator)
        nav.vector_index_health.return_value = health
        nav._check_stale_vector_index = (
            GDSNavigator._check_stale_vector_index.__get__(nav)
        )
        return nav

    def test_stale_index_emits_medium_alert(self):
        nav = self._nav({
            "index_present": True,
            "is_stale": True,
            "num_unindexed_rows": 40,
            "total_rows": 340,
            "indexed_fraction": 300 / 340,
            "stale_threshold": 0.1,
        })
        alerts = nav._check_stale_vector_index("cust_pattern")
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "MEDIUM"
        assert alerts[0]["check_type"] == "stale_vector_index"
        assert alerts[0]["pattern_id"] == "cust_pattern"
        assert "outside the IVF index" in alerts[0]["message"]
        assert alerts[0]["details"]["num_unindexed_rows"] == 40

    def test_fresh_index_emits_no_alert(self):
        nav = self._nav({
            "index_present": True,
            "is_stale": False,
            "num_unindexed_rows": 0,
            "total_rows": 300,
            "indexed_fraction": 1.0,
            "stale_threshold": 0.1,
        })
        assert nav._check_stale_vector_index("cust_pattern") == []

    def test_no_index_emits_no_alert(self):
        nav = self._nav({
            "index_present": False,
            "is_stale": False,
            "num_unindexed_rows": 0,
            "total_rows": 100,
            "indexed_fraction": 0.0,
            "stale_threshold": 0.1,
        })
        assert nav._check_stale_vector_index("cust_pattern") == []
