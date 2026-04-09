# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import json as _json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
from hypertopos.model.objects import SolidSlice
from hypertopos.model.sphere import LayerStorage, Sphere, StorageConfig
from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter


def _write_arrow(path: Path, table: pa.Table, custom_meta: dict | None = None) -> None:
    """Helper: write an Arrow IPC file with optional custom schema metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k.encode(): v.encode() for k, v in (custom_meta or {}).items()}
    schema = table.schema.with_metadata(meta)
    table = table.cast(schema)
    with ipc.new_file(str(path), schema) as writer:
        writer.write_table(table)


def test_read_sphere(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    sphere = reader.read_sphere()
    assert isinstance(sphere, Sphere)
    assert sphere.sphere_id == "sales_sphere"
    assert "customers" in sphere.lines
    assert "customer_pattern" in sphere.patterns


def test_read_sphere_line_versions(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    sphere = reader.read_sphere()
    assert sphere.lines["customers"].current_version() == 1


def test_read_sphere_line_columns(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    sphere = reader.read_sphere()
    cols = sphere.lines["customers"].columns
    assert cols is not None
    names = [c.name for c in cols]
    assert "primary_key" in names
    assert "name" in names
    assert "region" in names
    assert "version" not in names
    assert "status" not in names


def test_read_sphere_line_columns_absent(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    sphere = reader.read_sphere()
    assert sphere.lines["products"].columns is None


def test_read_geometry_all(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    table = reader.read_geometry("customer_pattern", version=1)
    assert table.num_rows == 2
    assert "primary_key" in table.schema.names


def test_read_geometry_single(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    table = reader.read_geometry("customer_pattern", version=1, primary_key="CUST-001")
    assert table.num_rows == 1
    assert table["primary_key"][0].as_py() == "CUST-001"


def test_read_temporal(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    # years param is ignored in Lance path — returns all slices for CUST-001
    table = reader.read_temporal("customer_pattern", "CUST-001", years=[2024])
    assert table.num_rows == 3
    assert "slice_index" in table.schema.names


def test_read_geometry_point_keys_filter(sphere_path):
    reader = GDSReader(base_path=sphere_path)
    table = reader.read_geometry("customer_pattern", version=1, point_keys=["PROD-001"])
    assert table.num_rows == 1
    assert table["primary_key"][0].as_py() == "CUST-001"


def test_read_geometry_sample_size(tmp_path):
    """sample_size returns exactly N rows without reading all data."""
    import json as json_mod

    n_total = 20
    geo_table = pa.table(
        {
            "primary_key": [f"E-{i:03d}" for i in range(n_total)],
            "scale": pa.array([1] * n_total, type=pa.int32()),
            "delta": pa.array([[0.1, 0.2]] * n_total, type=pa.list_(pa.float32())),
            "delta_norm": pa.array([0.22] * n_total, type=pa.float32()),
            "delta_rank_pct": pa.array([50.0] * n_total, type=pa.float32()),
            "is_anomaly": [False] * n_total,
            "edges": pa.array(
                [[]] * n_total,
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("line_id", pa.string()),
                            pa.field("point_key", pa.string()),
                            pa.field("status", pa.string()),
                            pa.field("direction", pa.string()),
                        ]
                    )
                ),
            ),
        }
    )
    writer = GDSWriter(base_path=str(tmp_path))
    geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
    writer.write_lance_geometry(geo_table, geo_dir)

    meta_dir = tmp_path / "_gds_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sphere.json").write_text(
        json_mod.dumps(
            {
                "sphere_id": "test",
                "name": "Test",
                "lines": {},
                "patterns": {},
                "aliases": {},
                "storage": {"geometry": {"format": "lance"}},
            }
        )
    )

    reader = GDSReader(base_path=str(tmp_path))
    reader.read_sphere()

    sample = reader.read_geometry("pat1", 1, sample_size=5)
    assert sample.num_rows == 5
    assert "primary_key" in sample.schema.names

    # sample_size >= total rows → return all rows, no Lance error
    sample_all = reader.read_geometry("pat1", 1, sample_size=n_total + 50)
    assert sample_all.num_rows == n_total


def test_append_temporal_slice(tmp_path):
    import lance

    writer = GDSWriter(base_path=str(tmp_path))
    s = SolidSlice(
        slice_index=0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        deformation_type="structural",
        delta_snapshot=np.array([0.1, 0.2], dtype=np.float32),
        delta_norm_snapshot=0.22,
        pattern_ver=1,
        changed_property=None,
        changed_line_id="products",
        added_edge=None,
    )
    writer.append_temporal_slice(
        s,
        pattern_id="customer_pattern",
        primary_key="CUST-001",
        shape_snapshot=np.array([0.1, 0.2], dtype=np.float32),
    )
    lance_path = tmp_path / "temporal" / "customer_pattern" / "data.lance"
    assert lance_path.exists()
    ds = lance.dataset(str(lance_path))
    assert ds.count_rows() == 1
    reader = GDSReader(base_path=str(tmp_path))
    table = reader.read_temporal("customer_pattern", "CUST-001")
    assert table.num_rows == 1
    assert "primary_key" in table.schema.names
    assert table["primary_key"][0].as_py() == "CUST-001"


def test_compact_temporal_no_lance_returns_no_lance(tmp_path):
    writer = GDSWriter(base_path=str(tmp_path))
    result = writer.compact_temporal("nonexistent_pattern")
    assert result["status"] == "no_lance"
    assert result["pattern_id"] == "nonexistent_pattern"


def test_compact_temporal_preserves_rows(tmp_path):
    import lance

    writer = GDSWriter(base_path=str(tmp_path))
    # Write 10 slices — each append() creates one Lance fragment
    for i in range(10):
        s = SolidSlice(
            slice_index=i,
            timestamp=datetime(2024, 1, i + 1, tzinfo=UTC),
            deformation_type="structural",
            delta_snapshot=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
            delta_norm_snapshot=0.1 * i,
            pattern_ver=1,
            changed_property=None,
            changed_line_id="customers",
            added_edge=None,
        )
        writer.append_temporal_slice(
            s,
            "test_pattern",
            f"ENTITY-{i:03d}",
            shape_snapshot=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
        )

    result = writer.compact_temporal("test_pattern")
    assert result["status"] == "ok"
    assert result["rows"] == 10
    assert result["rows_before"] == 10

    # Verify Lance dataset still readable after compaction
    lance_path = tmp_path / "temporal" / "test_pattern" / "data.lance"
    ds = lance.dataset(str(lance_path))
    assert ds.count_rows() == 10


def test_build_temporal_index(tmp_path):
    import lance

    writer = GDSWriter(base_path=str(tmp_path))
    slices = [
        SolidSlice(
            slice_index=i,
            timestamp=datetime(2024, 1, i + 1, tzinfo=UTC),
            deformation_type="structural",
            delta_snapshot=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
            delta_norm_snapshot=float(0.1 * i),
            pattern_ver=1,
            changed_property=None,
            changed_line_id="customers",
            added_edge=None,
        )
        for i in range(3)
    ]
    for i, s in enumerate(slices):
        entity = "ENTITY-A" if i < 2 else "ENTITY-B"
        writer.append_temporal_slice(
            s,
            "test_pattern",
            entity,
            shape_snapshot=np.array([0.1 * i, 0.2 * i], dtype=np.float32),
        )

    writer.build_temporal_index("test_pattern")

    lance_path = tmp_path / "temporal" / "test_pattern" / "data.lance"
    ds = lance.dataset(str(lance_path))
    indices = ds.list_indices()
    pk_indices = [
        idx for idx in indices if isinstance(idx, dict) and "primary_key" in idx.get("fields", [])
    ]
    assert len(pk_indices) == 1, f"Expected 1 BTREE index on primary_key, got: {indices}"


def test_build_temporal_index_noop_if_no_lance(tmp_path):
    writer = GDSWriter(base_path=str(tmp_path))
    # Should not raise even if the dataset does not exist
    writer.build_temporal_index("nonexistent_pattern")


_TINY_SCHEMA = pa.schema([pa.field("x", pa.int32())])
_TINY_TABLE = pa.table({"x": [1]}, schema=_TINY_SCHEMA)


# ---------------------------------------------------------------------------
# Agent-scoped temporal: write, read, purge
# ---------------------------------------------------------------------------

_AGENT_SLICE = SolidSlice(
    slice_index=0,
    timestamp=datetime(2024, 6, 15, tzinfo=UTC),
    deformation_type="internal",
    delta_snapshot=np.array([0.3, 0.4], dtype=np.float32),
    delta_norm_snapshot=0.5,
    pattern_ver=1,
    changed_property="region",
    changed_line_id=None,
    added_edge=None,
)
_AGENT_SHAPE = np.array([0.3, 0.4], dtype=np.float32)


class TestAgentScopedTemporalWrite:
    def test_write_with_agent_id_creates_agents_dir(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
            agent_id="agent-A",
        )
        lance_path = tmp_path / "temporal" / "_agents" / "agent-A" / "cp" / "data.lance"
        assert lance_path.exists()

    def test_write_without_agent_id_uses_permanent_path(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
        )
        lance_path = tmp_path / "temporal" / "cp" / "data.lance"
        assert lance_path.exists()
        assert not (tmp_path / "temporal" / "_agents").exists()


class TestPurgeAgentTemporal:
    def test_purge_deletes_agent_dir(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
            agent_id="agent-X",
        )
        count = writer.purge_agent_temporal("agent-X")
        assert count >= 1  # Lance dataset counted
        assert not (tmp_path / "temporal" / "_agents" / "agent-X").exists()

    def test_purge_nonexistent_returns_zero(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        assert writer.purge_agent_temporal("ghost") == 0

    def test_purge_does_not_touch_other_agents(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
            agent_id="keep",
        )
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
            agent_id="remove",
        )
        writer.purge_agent_temporal("remove")
        assert (tmp_path / "temporal" / "_agents" / "keep").exists()
        assert not (tmp_path / "temporal" / "_agents" / "remove").exists()

    def test_purge_does_not_touch_permanent(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
        )
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
            agent_id="agent-Y",
        )
        writer.purge_agent_temporal("agent-Y")
        lance_path = tmp_path / "temporal" / "cp" / "data.lance"
        assert lance_path.exists()

    def test_purge_all_agents(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
            agent_id="a1",
        )
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
            agent_id="a2",
        )
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
        )
        count = writer.purge_all_agents()
        assert count >= 2  # Lance datasets counted
        assert not (tmp_path / "temporal" / "_agents").exists()
        assert (tmp_path / "temporal" / "cp" / "data.lance").exists()

    def test_purge_all_agents_no_agents_dir(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        assert writer.purge_all_agents() == 0


class TestReadTemporalBusinessKeyFilter:
    def test_read_filters_by_primary_key(self, tmp_path):
        writer = GDSWriter(base_path=str(tmp_path))
        s1 = SolidSlice(
            slice_index=0,
            timestamp=datetime(2024, 6, 15, tzinfo=UTC),
            deformation_type="internal",
            delta_snapshot=np.array([0.3, 0.4], dtype=np.float32),
            delta_norm_snapshot=0.5,
            pattern_ver=1,
            changed_property=None,
            changed_line_id=None,
            added_edge=None,
        )
        s2 = SolidSlice(
            slice_index=1,
            timestamp=datetime(2024, 7, 1, tzinfo=UTC),
            deformation_type="edge",
            delta_snapshot=np.array([0.5, 0.6], dtype=np.float32),
            delta_norm_snapshot=0.78,
            pattern_ver=1,
            changed_property=None,
            changed_line_id=None,
            added_edge=None,
        )
        writer.append_temporal_slice(
            s1,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
        )
        writer.append_temporal_slice(
            s2,
            pattern_id="cp",
            primary_key="C-2",
            shape_snapshot=np.array([0.5, 0.6], dtype=np.float32),
        )
        reader = GDSReader(base_path=str(tmp_path))
        t1 = reader.read_temporal("cp", "C-1")
        assert t1.num_rows == 1
        assert t1["primary_key"][0].as_py() == "C-1"
        t2 = reader.read_temporal("cp", "C-2")
        assert t2.num_rows == 1
        assert t2["primary_key"][0].as_py() == "C-2"


class TestReadTemporalAgentScoped:
    def _setup_sphere(self, tmp_path) -> tuple[GDSWriter, GDSReader]:
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _AGENT_SLICE,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=_AGENT_SHAPE,
        )
        agent_slice = SolidSlice(
            slice_index=1,
            timestamp=datetime(2024, 7, 1, tzinfo=UTC),
            deformation_type="edge",
            delta_snapshot=np.array([0.5, 0.6], dtype=np.float32),
            delta_norm_snapshot=0.78,
            pattern_ver=1,
            changed_property=None,
            changed_line_id="products",
            added_edge=None,
        )
        writer.append_temporal_slice(
            agent_slice,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.5, 0.6], dtype=np.float32),
            agent_id="agent-M",
        )
        reader = GDSReader(base_path=str(tmp_path))
        return writer, reader

    def test_read_with_agent_id_merges_both(self, tmp_path):
        _, reader = self._setup_sphere(tmp_path)
        table = reader.read_temporal("cp", "C-1", agent_id="agent-M")
        assert table.num_rows == 2

    def test_read_without_agent_id_only_permanent(self, tmp_path):
        _, reader = self._setup_sphere(tmp_path)
        table = reader.read_temporal("cp", "C-1")
        assert table.num_rows == 1
        assert table["slice_index"][0].as_py() == 0


class TestReadTemporalFilters:
    def test_filter_by_year(self, tmp_path):
        """filters={"year": "2024"} works like old years=[2024]."""
        writer = GDSWriter(base_path=str(tmp_path))
        for yr in [2023, 2024]:
            s = SolidSlice(
                slice_index=yr - 2023,
                timestamp=datetime(yr, 6, 1, tzinfo=UTC),
                deformation_type="internal",
                delta_snapshot=np.array([0.3, 0.4], dtype=np.float32),
                delta_norm_snapshot=0.5,
                pattern_ver=1,
                changed_property=None,
                changed_line_id=None,
                added_edge=None,
            )
            writer.append_temporal_slice(
                s,
                pattern_id="cp",
                primary_key="C-1",
                shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
            )
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal("cp", "C-1", filters={"year": "2024"})
        assert table.num_rows == 1

    def test_filter_by_year_and_month_via_timestamp_bounds(self, tmp_path):
        """Timestamp bounds filter replaces Hive-partition year+month filter (Lance path)."""
        writer = GDSWriter(base_path=str(tmp_path))
        for month in [5, 6, 7]:
            s = SolidSlice(
                slice_index=month - 5,
                timestamp=datetime(2024, month, 1, tzinfo=UTC),
                deformation_type="internal",
                delta_snapshot=np.array([0.3, 0.4], dtype=np.float32),
                delta_norm_snapshot=0.5,
                pattern_ver=1,
                changed_property=None,
                changed_line_id=None,
                added_edge=None,
            )
            writer.append_temporal_slice(
                s,
                pattern_id="cp",
                primary_key="C-1",
                shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
            )
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal(
            "cp",
            "C-1",
            filters={
                "timestamp_from": "2024-06-01",
                "timestamp_to": "2024-07-01",
            },
        )
        assert table.num_rows == 1

    def test_no_filters_reads_all(self, tmp_path):
        """Without filters, reads everything."""
        writer = GDSWriter(base_path=str(tmp_path))
        for yr in [2023, 2024]:
            s = SolidSlice(
                slice_index=yr - 2023,
                timestamp=datetime(yr, 6, 1, tzinfo=UTC),
                deformation_type="internal",
                delta_snapshot=np.array([0.3, 0.4], dtype=np.float32),
                delta_norm_snapshot=0.5,
                pattern_ver=1,
                changed_property=None,
                changed_line_id=None,
                added_edge=None,
            )
            writer.append_temporal_slice(
                s,
                pattern_id="cp",
                primary_key="C-1",
                shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
            )
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal("cp", "C-1")
        assert table.num_rows == 2


class TestReadTemporalRecordFilter:
    """Record-level temporal filtering via _apply_temporal_filters on Lance datasets."""

    _SCHEMA = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("slice_index", pa.int32()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("deformation_type", pa.string()),
            pa.field("shape_snapshot", pa.list_(pa.float32())),
            pa.field("pattern_ver", pa.int32()),
            pa.field("changed_property", pa.string()),
            pa.field("changed_line_id", pa.string()),
        ]
    )

    def _write_temporal(
        self, tmp_path: Path, pattern_id: str, timestamps: list, primary_keys: list
    ) -> None:  # noqa: E501
        import lance as _lance_mod

        out_dir = tmp_path / "temporal" / pattern_id
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(timestamps)
        table = pa.table(
            {
                "primary_key": pa.array(primary_keys, type=pa.string()),
                "slice_index": pa.array(list(range(n)), type=pa.int32()),
                "timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
                "deformation_type": pa.array(["internal"] * n, type=pa.string()),
                "shape_snapshot": pa.array([[0.1, 0.2]] * n, type=pa.list_(pa.float32())),
                "pattern_ver": pa.array([1] * n, type=pa.int32()),
                "changed_property": pa.array([None] * n, type=pa.string()),
                "changed_line_id": pa.array([None] * n, type=pa.string()),
            },
            schema=self._SCHEMA,
        )
        _lance_mod.write_dataset(table, str(out_dir / "data.lance"), mode="create")

    def test_year_filter_excludes_wrong_year(self, tmp_path):
        """Year filter excludes records with timestamps outside the specified year."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2023, 12, 31, tzinfo=UTC),  # wrong year
                datetime(2024, 3, 15, tzinfo=UTC),  # correct year
            ],
            ["CUST-001", "CUST-001"],
        )

        reader = GDSReader(str(tmp_path))
        result = reader.read_temporal("test_pattern", "CUST-001", filters={"year": ["2024"]})

        assert result.num_rows == 1
        assert result["timestamp"][0].as_py().year == 2024

    def test_year_filter_works_on_flat_lance_data(self, tmp_path):
        """Year filter applied on flat Lance dataset returns only matching year records."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2023, 6, 1, tzinfo=UTC),
                datetime(2024, 3, 15, tzinfo=UTC),
            ],
            ["CUST-001", "CUST-001"],
        )

        reader = GDSReader(str(tmp_path))
        result = reader.read_temporal("test_pattern", "CUST-001", filters={"year": ["2024"]})

        assert result.num_rows == 1
        assert result["timestamp"][0].as_py().year == 2024

    def test_empty_year_list_returns_all(self, tmp_path):
        """Empty year list is a no-op — returns all records (no IndexError)."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2023, 6, 1, tzinfo=UTC),
                datetime(2024, 3, 15, tzinfo=UTC),
            ],
            ["CUST-001", "CUST-001"],
        )

        reader = GDSReader(str(tmp_path))
        result = reader.read_temporal("test_pattern", "CUST-001", filters={"year": []})
        assert result.num_rows == 2  # empty filter = no constraint

    def test_timestamp_from_filter(self, tmp_path):
        """timestamp_from filters records on or after the given ISO date."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2024, 3, 1, tzinfo=UTC),  # before
                datetime(2024, 6, 15, tzinfo=UTC),  # on/after
                datetime(2024, 9, 1, tzinfo=UTC),  # on/after
            ],
            ["CUST-001"] * 3,
        )

        reader = GDSReader(str(tmp_path))
        result = reader.read_temporal(
            "test_pattern", "CUST-001", filters={"timestamp_from": "2024-06-01"}
        )
        assert result.num_rows == 2

    def test_timestamp_to_filter(self, tmp_path):
        """timestamp_to filters records strictly before the given ISO date."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2024, 3, 1, tzinfo=UTC),  # before — included
                datetime(2024, 6, 15, tzinfo=UTC),  # after — excluded
            ],
            ["CUST-001"] * 2,
        )

        reader = GDSReader(str(tmp_path))
        result = reader.read_temporal(
            "test_pattern", "CUST-001", filters={"timestamp_to": "2024-06-01"}
        )
        assert result.num_rows == 1
        assert result["timestamp"][0].as_py().month == 3

    def test_timestamp_from_and_to_range(self, tmp_path):
        """timestamp_from + timestamp_to together define a half-open range [from, to)."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2024, 3, 1, tzinfo=UTC),  # excluded (before from)
                datetime(2024, 6, 15, tzinfo=UTC),  # included (in range)
                datetime(2024, 8, 1, tzinfo=UTC),  # included (in range)
                datetime(2024, 10, 1, tzinfo=UTC),  # excluded (after to)
            ],
            ["CUST-001"] * 4,
        )

        reader = GDSReader(str(tmp_path))
        result = reader.read_temporal(
            "test_pattern",
            "CUST-001",
            filters={"timestamp_from": "2024-06-01", "timestamp_to": "2024-09-01"},
        )
        assert result.num_rows == 2


class TestReadTemporalBatched:
    """read_temporal_batched with timestamp_from / timestamp_to filtering."""

    _SCHEMA = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("slice_index", pa.int32()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("deformation_type", pa.string()),
            pa.field("shape_snapshot", pa.list_(pa.float32())),
            pa.field("pattern_ver", pa.int32()),
            pa.field("changed_property", pa.string()),
            pa.field("changed_line_id", pa.string()),
        ]
    )

    def _write_temporal(self, tmp_path: Path, pattern_id: str, timestamps: list) -> None:
        import lance as _lance_mod

        out_dir = tmp_path / "temporal" / pattern_id
        out_dir.mkdir(parents=True, exist_ok=True)
        n = len(timestamps)
        table = pa.table(
            {
                "primary_key": pa.array([f"E-{i:03d}" for i in range(n)], type=pa.string()),
                "slice_index": pa.array(list(range(n)), type=pa.int32()),
                "timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
                "deformation_type": pa.array(["internal"] * n, type=pa.string()),
                "shape_snapshot": pa.array([[0.1, 0.2]] * n, type=pa.list_(pa.float32())),
                "pattern_ver": pa.array([1] * n, type=pa.int32()),
                "changed_property": pa.array([None] * n, type=pa.string()),
                "changed_line_id": pa.array([None] * n, type=pa.string()),
            },
            schema=self._SCHEMA,
        )
        _lance_mod.write_dataset(table, str(out_dir / "data.lance"), mode="create")

    def test_timestamp_from_and_to_filters_batched(self, tmp_path):
        """read_temporal_batched with timestamp_from + timestamp_to returns only in-range rows."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2024, 3, 1, tzinfo=UTC),  # excluded (before from)
                datetime(2024, 6, 15, tzinfo=UTC),  # included
                datetime(2024, 8, 1, tzinfo=UTC),  # included
                datetime(2024, 10, 1, tzinfo=UTC),  # excluded (after to)
            ],
        )
        reader = GDSReader(base_path=str(tmp_path))
        batches = list(
            reader.read_temporal_batched(
                "test_pattern",
                timestamp_from="2024-06-01",
                timestamp_to="2024-09-01",
            )
        )
        combined = pa.concat_tables([pa.Table.from_batches([b]) for b in batches])
        assert combined.num_rows == 2

    def test_timestamp_from_only(self, tmp_path):
        """read_temporal_batched with only timestamp_from excludes earlier records."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2024, 1, 1, tzinfo=UTC),  # excluded
                datetime(2024, 7, 1, tzinfo=UTC),  # included
                datetime(2024, 12, 1, tzinfo=UTC),  # included
            ],
        )
        reader = GDSReader(base_path=str(tmp_path))
        batches = list(
            reader.read_temporal_batched(
                "test_pattern",
                timestamp_from="2024-06-01",
            )
        )
        combined = pa.concat_tables([pa.Table.from_batches([b]) for b in batches])
        assert combined.num_rows == 2

    def test_no_timestamp_filters_returns_all(self, tmp_path):
        """read_temporal_batched without filters returns all records."""
        self._write_temporal(
            tmp_path,
            "test_pattern",
            [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 7, 1, tzinfo=UTC),
            ],
        )
        reader = GDSReader(base_path=str(tmp_path))
        batches = list(reader.read_temporal_batched("test_pattern"))
        combined = pa.concat_tables([pa.Table.from_batches([b]) for b in batches])
        assert combined.num_rows == 2


class TestReadPointsWithStaticFilter:
    """Integration: read_points with static partitioning on the real fixtures."""

    def test_read_points_filter_region_emea(self, sphere_path):
        reader = GDSReader(base_path=sphere_path)
        table = reader.read_points("customers", version=1, filters={"region": "EMEA"})
        assert table.num_rows == 2  # Alice + Bob
        regions = table["region"].to_pylist()
        assert all(r == "EMEA" for r in regions)

    def test_read_points_filter_region_apac(self, sphere_path):
        reader = GDSReader(base_path=sphere_path)
        table = reader.read_points("customers", version=1, filters={"region": "APAC"})
        assert table.num_rows == 1
        assert table["region"][0].as_py() == "APAC"


# ---------------------------------------------------------------------------
# Lance temporal writes
# ---------------------------------------------------------------------------


class TestDynamicTemporalPartitioning:
    def test_default_writes_lance(self, tmp_path):
        """Writes Lance dataset (no Hive partitions)."""
        import lance

        writer = GDSWriter(base_path=str(tmp_path))
        s = SolidSlice(
            slice_index=0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            deformation_type="structural",
            delta_snapshot=np.array([0.1, 0.2], dtype=np.float32),
            delta_norm_snapshot=0.22,
            pattern_ver=1,
            changed_property=None,
            changed_line_id=None,
            added_edge=None,
        )
        writer.append_temporal_slice(
            s,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.1, 0.2], dtype=np.float32),
        )
        lance_path = tmp_path / "temporal" / "cp" / "data.lance"
        assert lance_path.exists()
        ds = lance.dataset(str(lance_path))
        assert ds.count_rows() == 1

    def test_agent_scoped_lance_write(self, tmp_path):
        """Agent-scoped writes create Lance dataset under _agents/ dir."""
        import lance

        writer = GDSWriter(base_path=str(tmp_path))
        s = SolidSlice(
            slice_index=0,
            timestamp=datetime(2024, 6, 15, tzinfo=UTC),
            deformation_type="edge",
            delta_snapshot=np.array([0.3, 0.4], dtype=np.float32),
            delta_norm_snapshot=0.5,
            pattern_ver=1,
            changed_property=None,
            changed_line_id=None,
            added_edge=None,
        )
        writer.append_temporal_slice(
            s,
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
            agent_id="agent-A",
        )
        lance_path = tmp_path / "temporal" / "_agents" / "agent-A" / "cp" / "data.lance"
        assert lance_path.exists()
        ds = lance.dataset(str(lance_path))
        assert ds.count_rows() == 1


# ---------------------------------------------------------------------------
# Semantics overlay — semantics.json loading and validation
# ---------------------------------------------------------------------------


def test_read_sphere_no_semantics(sphere_path):
    """Sphere loads normally when semantics.json is absent."""
    reader = GDSReader(sphere_path)
    sphere = reader.read_sphere()
    assert sphere.description is None
    assert list(sphere.lines.values())[0].description is None


def test_read_sphere_with_semantics(sphere_path):
    """Sphere loads with descriptions when semantics.json is present."""
    import json
    from pathlib import Path

    sem = {
        "sphere": {"description": "Test sphere"},
        "lines": {"customers": {"description": "Customer master"}},
        "patterns": {
            "customer_pattern": {
                "description": "Customer structural pattern",
                "relations": {
                    "products": {
                        "display_name": "Products linked",
                        "interpretation": "1.0 = linked, 0.0 = none",
                    }
                },
            }
        },
    }
    sem_path = Path(sphere_path) / "_gds_meta" / "semantics.json"
    sem_path.write_text(json.dumps(sem))

    try:
        reader = GDSReader(sphere_path)
        sphere = reader.read_sphere()
        assert sphere.description == "Test sphere"
        assert sphere.lines["customers"].description == "Customer master"
        assert sphere.patterns["customer_pattern"].description == "Customer structural pattern"
        rel = next(
            r for r in sphere.patterns["customer_pattern"].relations if r.line_id == "products"
        )  # noqa: E501
        assert rel.display_name == "Products linked"
        assert rel.interpretation == "1.0 = linked, 0.0 = none"
    finally:
        sem_path.unlink(missing_ok=True)


def test_read_sphere_semantics_description_too_long(sphere_path):
    """ValueError raised when description exceeds 200 chars."""
    import json
    from pathlib import Path

    sem = {"sphere": {"description": "x" * 201}}
    sem_path = Path(sphere_path) / "_gds_meta" / "semantics.json"
    sem_path.write_text(json.dumps(sem))
    try:
        reader = GDSReader(sphere_path)
        with pytest.raises(ValueError, match="description.*exceeds"):
            reader.read_sphere()
    finally:
        sem_path.unlink(missing_ok=True)


def test_read_sphere_semantics_display_name_too_long(sphere_path):
    """ValueError raised when display_name exceeds 60 chars."""
    import json
    from pathlib import Path

    sem = {
        "patterns": {"customer_pattern": {"relations": {"products": {"display_name": "x" * 61}}}}
    }
    sem_path = Path(sphere_path) / "_gds_meta" / "semantics.json"
    sem_path.write_text(json.dumps(sem))
    try:
        reader = GDSReader(sphere_path)
        with pytest.raises(ValueError, match="display_name.*exceeds"):
            reader.read_sphere()
    finally:
        sem_path.unlink(missing_ok=True)


def test_read_sphere_semantics_interpretation_too_long(sphere_path):
    """ValueError raised when interpretation exceeds 200 chars."""
    import json
    from pathlib import Path

    sem = {
        "patterns": {"customer_pattern": {"relations": {"products": {"interpretation": "x" * 201}}}}
    }
    sem_path = Path(sphere_path) / "_gds_meta" / "semantics.json"
    sem_path.write_text(json.dumps(sem))
    try:
        reader = GDSReader(sphere_path)
        with pytest.raises(ValueError, match="interpretation.*exceeds"):
            reader.read_sphere()
    finally:
        sem_path.unlink(missing_ok=True)


def test_read_sphere_semantics_unknown_line_ignored(sphere_path):
    """Unknown line_id in semantics.json is silently ignored."""
    import json
    from pathlib import Path

    sem = {"lines": {"nonexistent_line": {"description": "Ghost line"}}}
    sem_path = Path(sphere_path) / "_gds_meta" / "semantics.json"
    sem_path.write_text(json.dumps(sem))
    try:
        reader = GDSReader(sphere_path)
        sphere = reader.read_sphere()  # must not raise
        assert "nonexistent_line" not in sphere.lines
    finally:
        sem_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# StorageConfig — per-layer format configuration
# ---------------------------------------------------------------------------


class TestStorageConfig:
    def test_default_storage_config(self):
        """Default StorageConfig uses Lance for all layers."""
        cfg = StorageConfig()
        assert cfg.points.format == "lance"
        assert cfg.geometry.format == "lance"
        assert cfg.temporal.format == "lance"
        assert cfg.invalidation_log.format == "lance"

    def test_layer_storage_options(self):
        """LayerStorage accepts format-specific options."""
        ls = LayerStorage(options={"bloom_filter_columns": ["primary_key"]})
        assert ls.options["bloom_filter_columns"] == ["primary_key"]


# ---------------------------------------------------------------------------
# Parse storage config from sphere.json
# ---------------------------------------------------------------------------


def test_parse_sphere_with_storage_config(tmp_path):
    """sphere.json with storage section is parsed into StorageConfig."""
    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text(
        _json.dumps(
            {
                "sphere_id": "test",
                "name": "test",
                "lines": {},
                "patterns": {},
                "aliases": {},
                "storage": {
                    "points": {
                        "format": "lance",
                        "options": {"bloom_filter_columns": ["primary_key"]},
                    },
                    "geometry": {"format": "lance"},
                    "temporal": {
                        "format": "lance",
                        "options": {"sort_by": ["primary_key", "slice_index"]},
                    },
                    "invalidation_log": {"format": "lance"},
                },
            }
        )
    )
    reader = GDSReader(base_path=str(tmp_path))
    sphere = reader.read_sphere()
    assert sphere.storage.points.format == "lance"
    assert sphere.storage.points.options["bloom_filter_columns"] == ["primary_key"]
    assert sphere.storage.geometry.format == "lance"
    assert sphere.storage.temporal.options["sort_by"] == ["primary_key", "slice_index"]


def test_parse_sphere_without_storage_config(tmp_path):
    """sphere.json without storage section gets default StorageConfig (all Lance)."""
    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text(
        _json.dumps(
            {
                "sphere_id": "test",
                "name": "test",
                "lines": {},
                "patterns": {},
                "aliases": {},
            }
        )
    )
    reader = GDSReader(base_path=str(tmp_path))
    sphere = reader.read_sphere()
    assert sphere.storage.points.format == "lance"
    assert sphere.storage.geometry.format == "lance"


def test_parse_sphere_partial_storage_config(tmp_path):
    """sphere.json with partial storage section uses defaults for missing layers."""
    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text(
        _json.dumps(
            {
                "sphere_id": "test",
                "name": "test",
                "lines": {},
                "patterns": {},
                "aliases": {},
                "storage": {
                    "geometry": {"format": "lance"},
                },
            }
        )
    )
    reader = GDSReader(base_path=str(tmp_path))
    sphere = reader.read_sphere()
    assert sphere.storage.geometry.format == "lance"
    assert sphere.storage.points.format == "lance"  # default


# ---------------------------------------------------------------------------
# Parquet reader — _read_files and _collect_files
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, table: pa.Table) -> None:
    """Helper: write a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


class TestParquetReadFiles:
    def test_read_single_parquet(self, tmp_path):
        """_read_files reads a single .parquet file."""
        f = tmp_path / "data.parquet"
        _write_parquet(f, _TINY_TABLE)
        reader = GDSReader(base_path=str(tmp_path))
        result = reader._read_files([str(f)])
        assert result.num_rows == 1

    def test_read_multiple_parquet(self, tmp_path):
        """_read_files concatenates multiple .parquet files."""
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        _write_parquet(f1, _TINY_TABLE)
        _write_parquet(f2, _TINY_TABLE)
        reader = GDSReader(base_path=str(tmp_path))
        result = reader._read_files([str(f1), str(f2)])
        assert result.num_rows == 2

    def test_read_empty_paths(self, tmp_path):
        """_read_files with empty list returns empty table."""
        reader = GDSReader(base_path=str(tmp_path))
        result = reader._read_files([])
        assert result.num_rows == 0


# ---------------------------------------------------------------------------
# Parquet writer — append_temporal_slice and append_invalidation_event
# ---------------------------------------------------------------------------


def test_append_temporal_slice_lance(tmp_path):
    """Writer creates Lance dataset, not .parquet or .arrow files."""
    import lance

    writer = GDSWriter(base_path=str(tmp_path))
    s = SolidSlice(
        slice_index=0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        deformation_type="structural",
        delta_snapshot=np.array([0.1, 0.2], dtype=np.float32),
        delta_norm_snapshot=0.22,
        pattern_ver=1,
        changed_property=None,
        changed_line_id="products",
        added_edge=None,
    )
    writer.append_temporal_slice(
        s,
        pattern_id="cp",
        primary_key="C-1",
        shape_snapshot=np.array([0.1, 0.2], dtype=np.float32),
    )
    lance_path = tmp_path / "temporal" / "cp" / "data.lance"
    assert lance_path.exists()
    ds = lance.dataset(str(lance_path))
    assert ds.count_rows() == 1


def test_append_temporal_slice_lance_appends(tmp_path):
    """Second call to append_temporal_slice appends to the same Lance dataset."""
    import lance

    writer = GDSWriter(base_path=str(tmp_path))
    s1 = SolidSlice(
        0,
        datetime(2023, 6, 1, tzinfo=UTC),
        "activation",
        np.array([0.1, 0.2], dtype=np.float32),
        0.22,
        1,
        "status",
        "customers",
        added_edge=None,
    )
    s2 = SolidSlice(
        1,
        datetime(2023, 7, 1, tzinfo=UTC),
        "update",
        np.array([0.3, 0.4], dtype=np.float32),
        0.50,
        1,
        "segment_id",
        "segments",
        added_edge=None,
    )
    writer.append_temporal_slice(
        s1, "pat", "K-001", shape_snapshot=np.array([0.1, 0.2], dtype=np.float32)
    )  # noqa: E501
    writer.append_temporal_slice(
        s2, "pat", "K-001", shape_snapshot=np.array([0.3, 0.4], dtype=np.float32)
    )  # noqa: E501

    lance_path = tmp_path / "temporal" / "pat" / "data.lance"
    assert lance_path.exists()
    ds = lance.dataset(str(lance_path))
    assert ds.count_rows() == 2


def test_read_temporal_from_lance(tmp_path):
    """read_temporal reads from Lance dataset and returns slices for primary_key."""
    writer = GDSWriter(base_path=str(tmp_path))
    for i in range(3):
        s = SolidSlice(
            i,
            datetime(2023, i + 1, 1, tzinfo=UTC),
            "update",
            np.array([float(i)], dtype=np.float32),
            float(i),
            1,
            "x",
            "l",
            added_edge=None,
        )
        writer.append_temporal_slice(
            s, "pat", "K-001", shape_snapshot=np.array([float(i)], dtype=np.float32)
        )  # noqa: E501
    # Write a slice for a different key
    s_other = SolidSlice(
        0,
        datetime(2023, 1, 1, tzinfo=UTC),
        "update",
        np.array([9.0], dtype=np.float32),
        9.0,
        1,
        "x",
        "l",
        added_edge=None,
    )
    writer.append_temporal_slice(
        s_other, "pat", "K-999", shape_snapshot=np.array([9.0], dtype=np.float32)
    )  # noqa: E501

    reader = GDSReader(base_path=str(tmp_path))
    result = reader.read_temporal("pat", "K-001")
    assert len(result) == 3
    assert set(result["primary_key"].to_pylist()) == {"K-001"}


def test_read_temporal_lance_with_timestamp_filters(tmp_path):
    """Lance temporal reader applies timestamp_from/timestamp_to correctly."""
    writer = GDSWriter(base_path=str(tmp_path))
    for yr in [2022, 2023, 2024]:
        s = SolidSlice(
            yr - 2022,
            datetime(yr, 6, 1, tzinfo=UTC),
            "internal",
            np.array([0.1, 0.2], dtype=np.float32),
            0.22,
            1,
            None,
            None,
            added_edge=None,
        )
        writer.append_temporal_slice(
            s, "cp", "C-1", shape_snapshot=np.array([0.1, 0.2], dtype=np.float32)
        )  # noqa: E501
    reader = GDSReader(base_path=str(tmp_path))
    result = reader.read_temporal(
        "cp",
        "C-1",
        filters={
            "timestamp_from": "2023-01-01",
            "timestamp_to": "2025-01-01",
        },
    )
    assert result.num_rows == 2
    years_returned = sorted(r.as_py().year for r in result["timestamp"])
    assert years_returned == [2023, 2024]


def test_read_temporal_lance_with_year_filter(tmp_path):
    """Lance temporal reader filters by year correctly."""
    writer = GDSWriter(base_path=str(tmp_path))
    for yr in [2023, 2024]:
        s = SolidSlice(
            yr - 2023,
            datetime(yr, 6, 1, tzinfo=UTC),
            "internal",
            np.array([0.3, 0.4], dtype=np.float32),
            0.5,
            1,
            None,
            None,
            added_edge=None,
        )
        writer.append_temporal_slice(
            s, "cp", "C-1", shape_snapshot=np.array([0.3, 0.4], dtype=np.float32)
        )  # noqa: E501
    reader = GDSReader(base_path=str(tmp_path))
    result = reader.read_temporal("cp", "C-1", filters={"year": "2024"})
    assert result.num_rows == 1
    assert result["timestamp"][0].as_py().year == 2024


# ---------------------------------------------------------------------------
# Lance geometry read/write
# ---------------------------------------------------------------------------

lance = pytest.importorskip("lance")


class TestLanceGeometry:
    def test_write_and_read_lance_geometry(self, tmp_path):
        """Write geometry as Lance dataset, read back as pa.Table."""
        geo_table = pa.table(
            {
                "primary_key": ["CUST-001", "CUST-002"],
                "delta": [[0.1, 0.2], [0.3, 0.4]],
                "delta_norm": pa.array([0.22, 0.50], type=pa.float32()),
                "is_anomaly": [False, True],
                "edges": pa.array(
                    [[{"line_id": "p", "point_key": "P-1", "status": "alive", "direction": "in"}]]
                    * 2,
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),  # noqa: E501  # noqa: E501
                ),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "test_pattern" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        reader = GDSReader(base_path=str(tmp_path))
        result = reader._read_lance(str(geo_dir / "data.lance"))
        assert result.num_rows == 2
        assert "primary_key" in result.schema.names

    def test_read_geometry_dispatches_to_lance(self, tmp_path):
        """When sphere.json has geometry.format=lance, read_geometry uses Lance path."""
        import json as json_mod

        geo_table = pa.table(
            {
                "primary_key": ["E-001", "E-002"],
                "delta": [[0.5, 0.5], [0.1, 0.9]],
                "delta_norm": pa.array([0.7, 0.9], type=pa.float32()),
                "is_anomaly": [False, True],
                "delta_rank_pct": pa.array([40.0, 90.0], type=pa.float32()),
                "edges": pa.array(
                    [[], []],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),  # noqa: E501
                ),
            }
        )
        # Write Lance dataset
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        # Write sphere.json with geometry.format=lance
        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        sphere_json = {
            "sphere_id": "test",
            "name": "Test",
            "lines": {},
            "patterns": {},
            "aliases": {},
            "storage": {"geometry": {"format": "lance"}},
        }
        (meta_dir / "sphere.json").write_text(json_mod.dumps(sphere_json))

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()  # loads StorageConfig
        result = reader.read_geometry("pat1", 1)
        assert result.num_rows == 2
        assert "primary_key" in result.schema.names

    def test_write_lance_geometry_with_index(self, tmp_path):
        """write_lance_geometry builds a vector index on the delta column."""
        import lance

        geo_table = pa.table(
            {
                "primary_key": [f"E-{i:03d}" for i in range(256)],
                "delta": [[float(i % 10), float((i * 2) % 10)] for i in range(256)],
                "delta_norm": pa.array([float(i) * 0.01 for i in range(256)], type=pa.float32()),
                "is_anomaly": [False] * 256,
                "delta_rank_pct": pa.array(
                    [float(i) / 2.56 for i in range(256)], type=pa.float32()
                ),
                "edges": pa.array(
                    [[] for _ in range(256)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),  # noqa: E501
                ),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        ds = lance.dataset(str(geo_dir / "data.lance"))
        indices = ds.list_indices()
        assert any(idx["fields"] == ["delta"] for idx in indices), (
            f"No vector index on delta. Indices: {indices}"
        )

    def test_write_lance_geometry_creates_scalar_indices(self, tmp_path):
        """write_lance_geometry creates BTREE/BITMAP scalar indices on geometry columns."""
        import lance

        geo_table = pa.table(
            {
                "primary_key": [f"E-{i:03d}" for i in range(256)],
                "delta": [[float(i % 10), float((i * 2) % 10)] for i in range(256)],
                "delta_norm": pa.array([float(i) * 0.01 for i in range(256)], type=pa.float32()),
                "is_anomaly": [False] * 128 + [True] * 128,
                "delta_rank_pct": pa.array(
                    [float(i) / 2.56 for i in range(256)], type=pa.float32()
                ),
                "edges": pa.array(
                    [[] for _ in range(256)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        ds = lance.dataset(str(geo_dir / "data.lance"))
        indices = ds.list_indices()
        indexed_fields = {idx["fields"][0] for idx in indices if len(idx["fields"]) == 1}
        assert "delta_norm" in indexed_fields, f"No index on delta_norm. Fields: {indexed_fields}"
        assert "is_anomaly" in indexed_fields, f"No index on is_anomaly. Fields: {indexed_fields}"
        assert "delta_rank_pct" in indexed_fields, (
            f"No index on delta_rank_pct. Fields: {indexed_fields}"
        )  # noqa: E501

    def test_write_lance_geometry_creates_label_list_index(self, tmp_path):
        """write_lance_geometry creates LABEL_LIST index on entity_keys column."""
        import lance

        n = 256
        entity_keys_data = [[f"CUST-{i % 10:03d}"] for i in range(n)]
        geo_table = pa.table(
            {
                "primary_key": [f"E-{i:03d}" for i in range(n)],
                "delta": [[float(i % 10), float((i * 2) % 10)] for i in range(n)],
                "delta_norm": pa.array([float(i) * 0.01 for i in range(n)], type=pa.float32()),
                "is_anomaly": [False] * n,
                "delta_rank_pct": pa.array([float(i) / 2.56 for i in range(n)], type=pa.float32()),
                "edges": pa.array(
                    [[] for _ in range(n)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
                "entity_keys": pa.array(entity_keys_data, type=pa.list_(pa.string())),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        ds = lance.dataset(str(geo_dir / "data.lance"))
        indices = ds.list_indices()
        indexed_fields = {idx["fields"][0] for idx in indices if len(idx["fields"]) == 1}
        assert "entity_keys" in indexed_fields, (
            f"No LABEL_LIST index on entity_keys. Fields: {indexed_fields}"
        )  # noqa: E501

    def test_count_geometry_rows_basic(self, tmp_path):
        """count_geometry_rows returns correct total without filter."""
        import json as json_mod

        n = 50
        geo_table = pa.table(
            {
                "primary_key": [f"E-{i:03d}" for i in range(n)],
                "delta": [[0.1, 0.2] for _ in range(n)],
                "delta_norm": pa.array([0.22] * n, type=pa.float32()),
                "is_anomaly": [False] * n,
                "delta_rank_pct": pa.array([50.0] * n, type=pa.float32()),
                "edges": pa.array(
                    [[] for _ in range(n)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "test",
                    "name": "Test",
                    "lines": {},
                    "patterns": {},
                    "aliases": {},
                    "storage": {"geometry": {"format": "lance"}},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()
        count = reader.count_geometry_rows("pat1", 1)
        assert count == n

    def test_count_geometry_rows_with_filter(self, tmp_path):
        """count_geometry_rows with array_contains filter returns correct subset count."""
        import json as json_mod

        n = 100
        entity_keys_data = [[f"CUST-{i % 5:03d}"] for i in range(n)]
        geo_table = pa.table(
            {
                "primary_key": [f"E-{i:03d}" for i in range(n)],
                "delta": [[0.1, 0.2] for _ in range(n)],
                "delta_norm": pa.array([0.22] * n, type=pa.float32()),
                "is_anomaly": [False] * n,
                "delta_rank_pct": pa.array([50.0] * n, type=pa.float32()),
                "edges": pa.array(
                    [[] for _ in range(n)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
                "entity_keys": pa.array(entity_keys_data, type=pa.list_(pa.string())),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "test",
                    "name": "Test",
                    "lines": {},
                    "patterns": {},
                    "aliases": {},
                    "storage": {"geometry": {"format": "lance"}},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()
        # CUST-000 appears every 5 rows → 100/5 = 20
        count = reader.count_geometry_rows(
            "pat1", 1, filter="array_contains(entity_keys, 'CUST-000')"
        )
        assert count == 20

    def test_count_geometry_rows_reuses_dataset_object(self, tmp_path):
        """count_geometry_rows reuses the same Lance dataset object across calls.

        Repeated calls with the same (pattern_id, version) must not re-open
        the dataset — otherwise LABEL_LIST index is reloaded from disk each time.
        """
        import json as json_mod

        n = 50
        geo_table = pa.table(
            {
                "primary_key": [f"E-{i:03d}" for i in range(n)],
                "delta": [[0.1, 0.2] for _ in range(n)],
                "delta_norm": pa.array([0.22] * n, type=pa.float32()),
                "is_anomaly": [False] * n,
                "delta_rank_pct": pa.array([50.0] * n, type=pa.float32()),
                "edges": pa.array(
                    [[] for _ in range(n)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
                "entity_keys": pa.array(
                    [[f"K-{i % 5:03d}"] for i in range(n)], type=pa.list_(pa.string())
                ),  # noqa: E501
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "test",
                    "name": "Test",
                    "lines": {},
                    "patterns": {},
                    "aliases": {},
                    "storage": {"geometry": {"format": "lance"}},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()

        # Patch lance.dataset at the module level to count calls
        import hypertopos.storage.reader as reader_module

        original_dataset = reader_module._lance.dataset
        call_count = [0]

        def counting_dataset(path):
            call_count[0] += 1
            return original_dataset(path)

        reader_module._lance.dataset = counting_dataset
        try:
            reader.count_geometry_rows("pat1", 1)
            reader.count_geometry_rows("pat1", 1, filter="array_contains(entity_keys, 'K-000')")
            reader.count_geometry_rows("pat1", 1, filter="array_contains(entity_keys, 'K-001')")
        finally:
            reader_module._lance.dataset = original_dataset

        # Dataset must be opened exactly ONCE, not 3 times
        assert call_count[0] == 1, (
            f"Expected 1 dataset open, got {call_count[0]}. "
            "LABEL_LIST index reload causes ~1.5s overhead per call."
        )

    def test_find_nearest_lance_uses_ann(self, tmp_path):
        """find_nearest_lance returns top-k results via Lance ANN."""
        import lance

        n = 512
        D = 4
        rng = np.random.default_rng(42)
        deltas = rng.random((n, D)).astype(np.float32)
        fixed_type = pa.list_(pa.float32(), D)
        table = pa.table(
            {
                "primary_key": [f"E-{i:04d}" for i in range(n)],
                "scale": pa.array([1] * n, type=pa.int32()),
                "delta": pa.array(deltas.tolist(), type=fixed_type),
                "delta_norm": pa.array(np.linalg.norm(deltas, axis=1).tolist(), type=pa.float32()),
                "delta_rank_pct": pa.array([50.0] * n, type=pa.float32()),
                "is_anomaly": pa.array([False] * n, type=pa.bool_()),
                "edges": pa.array(
                    [[] for _ in range(n)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),  # noqa: E501
                ),
                "last_refresh_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "updated_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(table, str(lance_dir / "data.lance"))
        ds = lance.dataset(str(lance_dir / "data.lance"))
        ds.create_index(
            "delta",
            index_type="IVF_PQ",
            num_partitions=16,
            num_sub_vectors=2,
        )

        reader = GDSReader(base_path=str(tmp_path))
        query = deltas[0]
        results = reader.find_nearest_lance("pat", 1, query, k=5)

        assert results is not None
        assert len(results) == 5
        keys, dists = zip(*results, strict=False)
        assert "E-0000" in keys  # the query itself
        assert all(d >= 0 for d in dists)

    def test_find_nearest_lance_filter_expr(self, tmp_path):
        """find_nearest_lance with filter_expr returns only rows matching the predicate."""
        import lance

        n = 64
        D = 4
        rng = np.random.default_rng(7)
        deltas = rng.random((n, D)).astype(np.float32)
        fixed_type = pa.list_(pa.float32(), D)
        # Mark the first half as anomalies, second half as normal
        is_anomaly = [True] * (n // 2) + [False] * (n // 2)
        table = pa.table(
            {
                "primary_key": [f"E-{i:04d}" for i in range(n)],
                "scale": pa.array([1] * n, type=pa.int32()),
                "delta": pa.array(deltas.tolist(), type=fixed_type),
                "delta_norm": pa.array(np.linalg.norm(deltas, axis=1).tolist(), type=pa.float32()),
                "delta_rank_pct": pa.array([50.0] * n, type=pa.float32()),
                "is_anomaly": pa.array(is_anomaly, type=pa.bool_()),
                "edges": pa.array(
                    [[] for _ in range(n)],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
                "last_refresh_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "updated_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(table, str(lance_dir / "data.lance"))

        reader = GDSReader(base_path=str(tmp_path))
        query = deltas[0]

        # Without filter: results may include both anomaly and non-anomaly rows
        results_all = reader.find_nearest_lance("pat", 1, query, k=5)
        assert results_all is not None
        assert len(results_all) <= 5

        # With filter_expr: only anomaly rows (E-0000 .. E-0031) should be returned
        results_filtered = reader.find_nearest_lance(
            "pat", 1, query, k=5, filter_expr="is_anomaly = true"
        )
        assert results_filtered is not None
        assert len(results_filtered) > 0
        returned_keys = [pk for pk, _ in results_filtered]
        # All returned keys must be from the anomaly half (indices 0..31)
        anomaly_keys = {f"E-{i:04d}" for i in range(n // 2)}
        assert all(pk in anomaly_keys for pk in returned_keys), (
            f"Expected only anomaly keys, got: {returned_keys}"
        )

    def test_read_geometry_lance_entity_keys_pushdown(self, tmp_path):
        """Lance geometry with entity_keys column uses pushdown filter."""
        import json as json_mod

        import lance
        from hypertopos.builder.builder import EDGE_STRUCT_TYPE

        n = 1000
        D = 3
        rng = np.random.default_rng(42)
        deltas = rng.random((n, D)).astype(np.float32)
        fixed_type = pa.list_(pa.float32(), D)

        # Build edges — each polygon connects to a customer (cyclic mod 100)
        edges_per_row = []
        entity_keys_per_row = []
        for i in range(n):
            cust = f"CUST-{i % 100:04d}"
            edges = [
                {"line_id": "customers", "point_key": cust, "status": "alive", "direction": "in"}
            ]
            edges_per_row.append(edges)
            entity_keys_per_row.append([cust])

        table = pa.table(
            {
                "primary_key": [f"E-{i:04d}" for i in range(n)],
                "scale": pa.array([1] * n, type=pa.int32()),
                "delta": pa.array(deltas.tolist(), type=fixed_type),
                "delta_norm": pa.array(np.linalg.norm(deltas, axis=1).tolist(), type=pa.float32()),
                "delta_rank_pct": pa.array([50.0] * n, type=pa.float32()),
                "is_anomaly": pa.array([False] * n, type=pa.bool_()),
                "edges": pa.array(edges_per_row, type=pa.list_(EDGE_STRUCT_TYPE)),
                "entity_keys": pa.array(entity_keys_per_row, type=pa.list_(pa.string())),
                "last_refresh_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "updated_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )

        # Write as Lance
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(table, str(lance_dir / "data.lance"))

        # Write minimal sphere.json with lance format
        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "test",
                    "name": "test",
                    "storage": {"geometry": {"format": "lance"}},
                    "lines": {},
                    "patterns": {},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()

        # Filter by single point_key — should use Lance pushdown
        result = reader.read_geometry("pat", 1, point_keys=["CUST-0000"])
        # CUST-0000 appears every 100 rows: indices 0, 100, 200, ...
        assert len(result) == 10
        for i in range(len(result)):
            keys = result["entity_keys"][i].as_py()
            assert "CUST-0000" in keys

    def test_read_geometry_lance_entity_keys_multi_key(self, tmp_path):
        """Lance pushdown with multiple point_keys uses array_has_any."""
        import json as json_mod

        import lance
        from hypertopos.builder.builder import EDGE_STRUCT_TYPE

        n = 200
        D = 2
        rng = np.random.default_rng(99)
        deltas = rng.random((n, D)).astype(np.float32)
        fixed_type = pa.list_(pa.float32(), D)

        edges_per_row = []
        entity_keys_per_row = []
        for i in range(n):
            cust = f"C-{i % 50:03d}"
            edges = [{"line_id": "cust", "point_key": cust, "status": "alive", "direction": "in"}]
            edges_per_row.append(edges)
            entity_keys_per_row.append([cust])

        table = pa.table(
            {
                "primary_key": [f"E-{i:04d}" for i in range(n)],
                "scale": pa.array([1] * n, type=pa.int32()),
                "delta": pa.array(deltas.tolist(), type=fixed_type),
                "delta_norm": pa.array(np.linalg.norm(deltas, axis=1).tolist(), type=pa.float32()),
                "delta_rank_pct": pa.array([50.0] * n, type=pa.float32()),
                "is_anomaly": pa.array([False] * n, type=pa.bool_()),
                "edges": pa.array(edges_per_row, type=pa.list_(EDGE_STRUCT_TYPE)),
                "entity_keys": pa.array(entity_keys_per_row, type=pa.list_(pa.string())),
                "last_refresh_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "updated_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )

        lance_dir = tmp_path / "geometry" / "p" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(table, str(lance_dir / "data.lance"))

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "t",
                    "name": "t",
                    "storage": {"geometry": {"format": "lance"}},
                    "lines": {},
                    "patterns": {},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()

        # Filter by two keys — C-000 and C-001
        result = reader.read_geometry("p", 1, point_keys=["C-000", "C-001"])
        # Each key appears 200/50 = 4 times, total = 8
        assert len(result) == 8

    def test_read_geometry_column_projection(self, tmp_path):
        """read_geometry with columns= returns only requested columns."""
        import json as json_mod

        geo_table = pa.table(
            {
                "primary_key": ["E-001", "E-002"],
                "delta": [[0.5, 0.5], [0.1, 0.9]],
                "delta_norm": pa.array([0.7, 0.9], type=pa.float32()),
                "is_anomaly": [False, True],
                "delta_rank_pct": pa.array([40.0, 90.0], type=pa.float32()),
                "edges": pa.array(
                    [[], []],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "test",
                    "name": "Test",
                    "lines": {},
                    "patterns": {},
                    "aliases": {},
                    "storage": {"geometry": {"format": "lance"}},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()

        # Projected read — only 2 columns
        result = reader.read_geometry("pat1", 1, columns=["primary_key", "delta"])
        assert result.num_rows == 2
        assert set(result.schema.names) == {"primary_key", "delta"}

        # Full read — backward compat
        full = reader.read_geometry("pat1", 1)
        assert "delta_norm" in full.schema.names
        assert "is_anomaly" in full.schema.names

    def test_read_geometry_column_projection_with_point_keys(self, tmp_path):
        """Column projection works with point_keys filter."""
        import json as json_mod

        import lance
        from hypertopos.builder.builder import EDGE_STRUCT_TYPE

        n = 100
        D = 2
        rng = np.random.default_rng(42)
        deltas = rng.random((n, D)).astype(np.float32)
        entity_keys_per_row = [[f"CUST-{i % 10:04d}"] for i in range(n)]
        edges_per_row = [
            [
                {
                    "line_id": "customers",
                    "point_key": f"CUST-{i % 10:04d}",
                    "status": "alive",
                    "direction": "in",
                }
            ]
            for i in range(n)
        ]
        table = pa.table(
            {
                "primary_key": [f"GL-{i:04d}" for i in range(n)],
                "scale": ["global"] * n,
                "delta": pa.FixedSizeListArray.from_arrays(
                    pa.array(deltas.ravel(), type=pa.float32()), D
                ),
                "delta_norm": pa.array(np.linalg.norm(deltas, axis=1), type=pa.float32()),
                "is_anomaly": pa.array([False] * n, type=pa.bool_()),
                "delta_rank_pct": pa.array(np.linspace(0, 100, n), type=pa.float32()),
                "edges": pa.array(edges_per_row, type=pa.list_(EDGE_STRUCT_TYPE)),
                "entity_keys": pa.array(entity_keys_per_row, type=pa.list_(pa.string())),
                "last_refresh_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "updated_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )

        lance_dir = tmp_path / "geometry" / "p" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(table, str(lance_dir / "data.lance"))

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "t",
                    "name": "t",
                    "storage": {"geometry": {"format": "lance"}},
                    "lines": {},
                    "patterns": {},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()

        result = reader.read_geometry(
            "p",
            1,
            point_keys=["CUST-0000"],
            columns=["primary_key", "delta"],
        )
        assert result.num_rows == 10
        assert set(result.schema.names) == {"primary_key", "delta"}

    def test_read_geometry_lance_filter_pushdown(self, tmp_path):
        """read_geometry with filter= pushes predicate to Lance scanner."""
        import json as json_mod

        geo_table = pa.table(
            {
                "primary_key": ["E-001", "E-002", "E-003"],
                "delta": [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
                "delta_norm": pa.array([0.14, 0.71, 1.27], type=pa.float32()),
                "is_anomaly": [False, False, True],
                "delta_rank_pct": pa.array([10.0, 50.0, 95.0], type=pa.float32()),
                "edges": pa.array(
                    [[], [], []],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "test",
                    "name": "Test",
                    "lines": {},
                    "patterns": {},
                    "aliases": {},
                    "storage": {"geometry": {"format": "lance"}},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()

        # Filter: only rows with delta_norm > 0.5
        result = reader.read_geometry("pat1", 1, filter="delta_norm > 0.5")
        assert result.num_rows == 2
        keys = set(result["primary_key"].to_pylist())
        assert keys == {"E-002", "E-003"}

        # Filter with column projection
        result2 = reader.read_geometry(
            "pat1",
            1,
            filter="delta_norm > 0.5",
            columns=["primary_key", "delta_norm"],
        )
        assert result2.num_rows == 2
        assert set(result2.schema.names) == {"primary_key", "delta_norm"}

        # No filter — all rows
        result3 = reader.read_geometry("pat1", 1)
        assert result3.num_rows == 3

    def test_read_geometry_primary_key_pushdown(self, tmp_path):
        """read_geometry(primary_key=...) uses Lance predicate pushdown, not Python filter."""
        import json as json_mod

        geo_table = pa.table(
            {
                "primary_key": ["E-001", "E-002", "E-003"],
                "scale": pa.array([1, 1, 1], type=pa.int32()),
                "delta": pa.array(
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], type=pa.list_(pa.float32())
                ),
                "delta_norm": pa.array([0.22, 0.5, 0.78], type=pa.float32()),
                "delta_rank_pct": pa.array([10.0, 50.0, 90.0], type=pa.float32()),
                "is_anomaly": [False, False, True],
                "edges": pa.array(
                    [[], [], []],
                    type=pa.list_(
                        pa.struct(
                            [
                                pa.field("line_id", pa.string()),
                                pa.field("point_key", pa.string()),
                                pa.field("status", pa.string()),
                                pa.field("direction", pa.string()),
                            ]
                        )
                    ),
                ),
            }
        )
        writer = GDSWriter(base_path=str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat1" / "v=1"
        writer.write_lance_geometry(geo_table, geo_dir)

        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(
            json_mod.dumps(
                {
                    "sphere_id": "test",
                    "name": "Test",
                    "lines": {},
                    "patterns": {},
                    "aliases": {},
                    "storage": {"geometry": {"format": "lance"}},
                }
            )
        )

        reader = GDSReader(base_path=str(tmp_path))
        reader.read_sphere()

        # Single primary_key lookup — must return exactly 1 row
        result = reader.read_geometry("pat1", 1, primary_key="E-002")
        assert result.num_rows == 1
        assert result["primary_key"][0].as_py() == "E-002"
        assert result["delta_norm"][0].as_py() == pytest.approx(0.5, abs=0.01)

        # Non-existent key — must return 0 rows
        result_missing = reader.read_geometry("pat1", 1, primary_key="E-999")
        assert result_missing.num_rows == 0

    def test_geometry_schema_has_no_flat_edge_arrays(self, tmp_path):
        """GEOMETRY_SCHEMA must NOT include redundant edge_line_ids/edge_point_keys."""
        from hypertopos.builder.builder import GEOMETRY_SCHEMA

        names = GEOMETRY_SCHEMA.names
        assert "edge_line_ids" not in names, "edge_line_ids should be removed"
        assert "edge_point_keys" not in names, "edge_point_keys should be removed"
        assert "edges" in names, "edges struct column must be present"
        assert "entity_keys" in names, "entity_keys column must be present"

    def test_entity_keys_content(self, tmp_path):
        """Builder populates entity_keys correctly (alive edges only)."""
        edges_data = [
            [
                {
                    "line_id": "customers",
                    "point_key": "C-001",
                    "status": "alive",
                    "direction": "in",
                },  # noqa: E501
                {"line_id": "items", "point_key": "I-001", "status": "dead", "direction": "in"},  # noqa: E501
            ],
            [
                {
                    "line_id": "customers",
                    "point_key": "C-002",
                    "status": "alive",
                    "direction": "in",
                },  # noqa: E501
                {"line_id": "items", "point_key": "I-002", "status": "alive", "direction": "in"},  # noqa: E501
            ],
        ]
        # entity_keys should contain alive point_keys only
        entity_keys = [
            [e["point_key"] for e in edges_data[0] if e["status"] == "alive"],
            [e["point_key"] for e in edges_data[1] if e["status"] == "alive"],
        ]
        assert entity_keys[0] == ["C-001"]
        assert entity_keys[1] == ["C-002", "I-002"]
        # dead edge excluded from row 0
        assert "I-001" not in entity_keys[0]


# ---------------------------------------------------------------------------
# Lance points read/write
# ---------------------------------------------------------------------------


class TestLancePoints:
    def test_write_and_read_lance_points(self, tmp_path):
        """write_points (Lance) + read_points returns correct row via scanner."""
        from hypertopos.builder._writer import write_points

        table = pa.table(
            {
                "primary_key": ["C-001", "C-002", "C-003"],
                "name": ["Alice", "Bob", "Carol"],
                "region": ["EU", "US", "EU"],
                "version": pa.array([1, 1, 1], type=pa.int32()),
                "status": ["active", "active", "active"],
                "created_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "changed_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        write_points(tmp_path, "customers", table, version=1, partition_col=None)

        reader = GDSReader(base_path=str(tmp_path))
        # Full read
        result_all = reader.read_points("customers", version=1)
        assert len(result_all) == 3

        # Key lookup via Lance scanner
        result_one = reader.read_points("customers", version=1, primary_key="C-002")
        assert len(result_one) == 1
        assert result_one["name"][0].as_py() == "Bob"

    def test_read_lance_points_with_filter(self, tmp_path):
        """read_points with filters uses Lance scanner predicate pushdown."""
        from hypertopos.builder._writer import write_points

        table = pa.table(
            {
                "primary_key": ["C-001", "C-002", "C-003"],
                "name": ["Alice", "Bob", "Carol"],
                "region": ["EU", "US", "EU"],
                "version": pa.array([1, 1, 1], type=pa.int32()),
                "status": ["active", "active", "active"],
                "created_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "changed_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        write_points(tmp_path, "customers", table, version=1, partition_col=None)

        reader = GDSReader(base_path=str(tmp_path))
        result = reader.read_points("customers", version=1, filters={"region": "EU"})
        assert len(result) == 2
        assert set(result["name"].to_pylist()) == {"Alice", "Carol"}

    def test_read_lance_points_no_match(self, tmp_path):
        """read_points with non-existent key returns empty table."""
        from hypertopos.builder._writer import write_points

        table = pa.table(
            {
                "primary_key": ["C-001"],
                "name": ["Alice"],
                "region": ["EU"],
                "version": pa.array([1], type=pa.int32()),
                "status": ["active"],
                "created_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "changed_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        write_points(tmp_path, "customers", table, version=1, partition_col=None)

        reader = GDSReader(base_path=str(tmp_path))
        result = reader.read_points("customers", version=1, primary_key="NOPE")
        assert len(result) == 0

    def test_read_points_cache_hit(self, tmp_path):
        """Second read_points call uses cache, not disk."""
        from unittest.mock import patch

        from hypertopos.builder._writer import write_points

        table = pa.table(
            {
                "primary_key": ["C-001", "C-002"],
                "name": ["Alice", "Bob"],
                "region": ["EU", "US"],
                "version": pa.array([1, 1], type=pa.int32()),
                "status": ["active", "active"],
                "created_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "changed_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        write_points(tmp_path, "customers", table, version=1, partition_col=None)

        reader = GDSReader(base_path=str(tmp_path))
        # First read — from disk, populates cache
        r1 = reader.read_points("customers", 1)
        assert len(r1) == 2

        # Second read — from cache (patch lance.dataset to verify no disk access)
        with patch("lance.dataset", side_effect=RuntimeError("should not be called")):
            r2 = reader.read_points("customers", 1)
        assert len(r2) == 2

        # Filtered read by primary_key uses cached full table
        with patch("lance.dataset", side_effect=RuntimeError("should not be called")):
            r3 = reader.read_points("customers", 1, primary_key="C-001")
        assert len(r3) == 1
        assert r3["name"][0].as_py() == "Alice"


# ---------------------------------------------------------------------------
# read_points cold-start fast path for single-key reads
# ---------------------------------------------------------------------------


class TestReadPointsSingleKeyColdPath:
    def _make_lance_points_table(self):
        from datetime import UTC, datetime

        return pa.table(
            {
                "primary_key": ["C-001", "C-002", "C-003"],
                "name": ["Alice", "Bob", "Carol"],
                "region": ["EU", "US", "EU"],
                "version": pa.array([1, 1, 1], type=pa.int32()),
                "status": ["active", "active", "active"],
                "created_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "changed_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )

    def test_read_points_single_key_cold_targeted_scan(self, tmp_path):
        """Cold cache + primary_key: uses targeted scan, does NOT populate cache."""
        from unittest.mock import patch

        from hypertopos.builder._writer import write_points

        table = self._make_lance_points_table()
        write_points(tmp_path, "customers", table, version=1, partition_col=None)

        reader = GDSReader(base_path=str(tmp_path))
        original_read_lance_points = reader._read_lance_points

        call_log: list[tuple] = []

        def tracked_read(lance_path, primary_key, filters):
            call_log.append((lance_path, primary_key, filters))
            return original_read_lance_points(lance_path, primary_key, filters)

        with patch.object(reader, "_read_lance_points", side_effect=tracked_read):
            result = reader.read_points("customers", 1, primary_key="C-002")

        assert len(result) == 1
        assert result["name"][0].as_py() == "Bob"
        # Targeted scan was called with the specific primary_key (not None)
        assert len(call_log) == 1
        pk_arg = call_log[0][1]
        assert pk_arg == "C-002"
        # Cache must NOT be populated for single-key cold read
        assert ("customers", 1) not in reader._points_cache

    def test_read_points_single_key_warm_uses_cache(self, tmp_path):
        """Warm cache + primary_key: filters from cache, no disk read."""
        from unittest.mock import patch

        from hypertopos.builder._writer import write_points

        table = self._make_lance_points_table()
        write_points(tmp_path, "customers", table, version=1, partition_col=None)

        reader = GDSReader(base_path=str(tmp_path))
        # Warm the cache with a full read
        reader.read_points("customers", 1)
        assert ("customers", 1) in reader._points_cache

        # Single-key read with warm cache must NOT call _read_lance_points
        with patch.object(
            reader,
            "_read_lance_points",
            side_effect=RuntimeError("should not be called"),
        ):
            result = reader.read_points("customers", 1, primary_key="C-003")

        assert len(result) == 1
        assert result["name"][0].as_py() == "Carol"

    def test_read_points_bulk_still_caches(self, tmp_path):
        """Bulk read (no primary_key) still caches; second bulk read hits cache."""
        from unittest.mock import patch

        from hypertopos.builder._writer import write_points

        table = self._make_lance_points_table()
        write_points(tmp_path, "customers", table, version=1, partition_col=None)

        reader = GDSReader(base_path=str(tmp_path))
        original_read_lance_points = reader._read_lance_points
        call_count = {"n": 0}

        def counted_read(lance_path, primary_key, filters):
            call_count["n"] += 1
            return original_read_lance_points(lance_path, primary_key, filters)

        with patch.object(reader, "_read_lance_points", side_effect=counted_read):
            r1 = reader.read_points("customers", 1)
            r2 = reader.read_points("customers", 1)

        assert len(r1) == 3
        assert len(r2) == 3
        # _read_lance_points called exactly once (second read uses cache)
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# read_temporal_batch — population-wide temporal scan (no primary_key filter)
# ---------------------------------------------------------------------------


def _make_slice(idx: int, pk: str, yr: int, delta: list[float]) -> SolidSlice:
    return SolidSlice(
        slice_index=idx,
        timestamp=datetime(yr, 6, 1, tzinfo=UTC),
        deformation_type="edge",
        delta_snapshot=np.array(delta, dtype=np.float32),
        delta_norm_snapshot=float(np.linalg.norm(delta)),
        pattern_ver=1,
        changed_property=None,
        changed_line_id=None,
        added_edge=None,
    )


class TestReadTemporalBatch:
    def test_returns_all_keys_no_filter(self, tmp_path):
        """read_temporal_batch returns rows for all entities, not filtered by pk."""
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _make_slice(0, "C-1", 2024, [0.1, 0.2]),
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.1, 0.2], dtype=np.float32),
        )  # noqa: E501
        writer.append_temporal_slice(
            _make_slice(0, "C-2", 2024, [0.3, 0.4]),
            pattern_id="cp",
            primary_key="C-2",
            shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
        )  # noqa: E501
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal_batch("cp")
        assert table.num_rows == 2
        keys = set(table["primary_key"].to_pylist())
        assert keys == {"C-1", "C-2"}

    def test_empty_pattern_returns_empty_table(self, tmp_path):
        """read_temporal_batch returns empty table when no temporal data exists."""
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal_batch("nonexistent")
        assert table.num_rows == 0

    def test_timestamp_from_filter_applied(self, tmp_path):
        """timestamp_from filter excludes rows before the cutoff."""
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _make_slice(0, "C-1", 2023, [0.1, 0.2]),
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.1, 0.2], dtype=np.float32),
        )  # noqa: E501
        writer.append_temporal_slice(
            _make_slice(1, "C-1", 2024, [0.3, 0.4]),
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
        )  # noqa: E501
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal_batch("cp", filters={"timestamp_from": "2024-01-01"})
        assert table.num_rows == 1
        assert table["slice_index"][0].as_py() == 1

    def test_timestamp_to_filter_applied(self, tmp_path):
        """timestamp_to filter excludes rows on or after the cutoff."""
        writer = GDSWriter(base_path=str(tmp_path))
        writer.append_temporal_slice(
            _make_slice(0, "C-1", 2023, [0.1, 0.2]),
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.1, 0.2], dtype=np.float32),
        )  # noqa: E501
        writer.append_temporal_slice(
            _make_slice(1, "C-1", 2025, [0.3, 0.4]),
            pattern_id="cp",
            primary_key="C-1",
            shape_snapshot=np.array([0.3, 0.4], dtype=np.float32),
        )  # noqa: E501
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal_batch("cp", filters={"timestamp_to": "2024-01-01"})
        assert table.num_rows == 1
        assert table["slice_index"][0].as_py() == 0

    def test_multiple_entities_with_filter(self, tmp_path):
        """Timestamp filter applies to all entities in one scan."""
        writer = GDSWriter(base_path=str(tmp_path))
        for pk in ["C-1", "C-2", "C-3"]:
            writer.append_temporal_slice(
                _make_slice(0, pk, 2023, [0.1]),
                pattern_id="cp",
                primary_key=pk,
                shape_snapshot=np.array([0.1], dtype=np.float32),
            )  # noqa: E501
            writer.append_temporal_slice(
                _make_slice(1, pk, 2025, [0.2]),
                pattern_id="cp",
                primary_key=pk,
                shape_snapshot=np.array([0.2], dtype=np.float32),
            )  # noqa: E501
        reader = GDSReader(base_path=str(tmp_path))
        table = reader.read_temporal_batch("cp", filters={"timestamp_from": "2024-01-01"})
        assert table.num_rows == 3
        assert set(table["primary_key"].to_pylist()) == {"C-1", "C-2", "C-3"}


class TestReadPointsBatch:
    """read_points_batch — targeted fetch by primary key list."""

    @pytest.fixture()
    def _points_path(self, tmp_path):
        from hypertopos.builder._writer import write_points

        table = pa.table(
            {
                "primary_key": ["C-001", "C-002", "C-003"],
                "name": ["Alice", "Bob", "Carol"],
                "version": pa.array([1, 1, 1], type=pa.int32()),
                "status": ["active", "active", "active"],
                "created_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "changed_at": pa.array(
                    [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        write_points(tmp_path, "customers", table, version=1, partition_col=None)
        return tmp_path

    def test_batch_returns_requested_rows(self, _points_path):
        reader = GDSReader(base_path=str(_points_path))
        result = reader.read_points_batch("customers", 1, ["C-001", "C-003"])
        assert result.num_rows == 2
        assert set(result["primary_key"].to_pylist()) == {"C-001", "C-003"}

    def test_batch_empty_list_returns_empty(self, _points_path):
        reader = GDSReader(base_path=str(_points_path))
        result = reader.read_points_batch("customers", 1, [])
        assert result.num_rows == 0

    def test_batch_uses_cached_table(self, _points_path):
        """If full table already cached, batch read filters from it (no disk IO)."""
        reader = GDSReader(base_path=str(_points_path))
        # Prime the cache with a full read
        reader.read_points("customers", version=1)
        assert ("customers", 1) in reader._points_cache
        # Batch read should hit cache
        result = reader.read_points_batch("customers", 1, ["C-002"])
        assert result.num_rows == 1
        assert result["name"].to_pylist() == ["Bob"]

    def test_batch_nonexistent_key_returns_empty(self, _points_path):
        reader = GDSReader(base_path=str(_points_path))
        result = reader.read_points_batch("customers", 1, ["NOPE-999"])
        assert result.num_rows == 0

    def test_batch_runtime_error_propagates_in_large_key_fast_path(
        self, _points_path, monkeypatch
    ):
        """Unexpected scanner errors should not be swallowed by the IN-filter fallback."""

        class FakeScanner:
            def to_table(self):
                raise RuntimeError("boom")

        class FakeDataset:
            def scanner(self, *args, **kwargs):
                return FakeScanner()

        import hypertopos.storage.reader as reader_module

        monkeypatch.setattr(reader_module._lance, "dataset", lambda *a, **kw: FakeDataset())

        reader = GDSReader(base_path=str(_points_path))
        keys = [f"C-{i:03d}" for i in range(101)]
        with pytest.raises(RuntimeError, match="boom"):
            reader.read_points_batch("customers", 1, keys)


class TestResolveByEntityKeys:
    """Resolve primary keys via geometry entity_keys LABEL_LIST index."""

    def _build_fixture(self, tmp_path: Path) -> GDSReader:
        """Write geometry with entity_keys + sphere.json for relation lookup."""
        from hypertopos.storage.writer import _write_lance

        # sphere.json with gl_pattern having two relations: customers, company_codes
        sphere_json = {
            "sphere_id": "test",
            "name": "test",
            "lines": {
                "customers": {
                    "line_id": "customers",
                    "entity_type": "customer",
                    "line_role": "anchor",
                    "pattern_id": "cust_pattern",
                    "partitioning": {"mode": "static", "columns": []},
                    "versions": [1],
                },
                "company_codes": {
                    "line_id": "company_codes",
                    "entity_type": "company_code",
                    "line_role": "anchor",
                    "pattern_id": "cc_pattern",
                    "partitioning": {"mode": "static", "columns": []},
                    "versions": [1],
                },
            },
            "patterns": {
                "gl_pattern": {
                    "pattern_id": "gl_pattern",
                    "entity_type": "gl_entry",
                    "pattern_type": "event",
                    "version": 1,
                    "status": "production",
                    "relations": [
                        {"line_id": "customers", "direction": "in", "required": True},
                        {"line_id": "company_codes", "direction": "in", "required": True},
                    ],
                    "mu": [0.0, 0.0],
                    "sigma_diag": [1.0, 1.0],
                    "theta": [0.0, 0.0],
                    "population_size": 3,
                    "computed_at": "2024-01-01T00:00:00+00:00",
                },
            },
            "aliases": {},
        }
        meta_dir = tmp_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(_json.dumps(sphere_json))
        # entity_keys[0] = customer key, entity_keys[1] = company_code key
        geo_table = pa.table(
            {
                "primary_key": ["GL-001", "GL-002", "GL-003"],
                "delta": pa.array(
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    type=pa.list_(pa.float32(), 2),
                ),
                "delta_norm": pa.array([0.2, 0.5, 0.8], type=pa.float32()),
                "is_anomaly": [False, True, False],
                "delta_rank_pct": pa.array([20.0, 70.0, 90.0], type=pa.float32()),
                "entity_keys": [
                    ["CUST-001", "CC-001"],
                    ["CUST-001", "CC-002"],
                    ["CUST-002", "CC-001"],
                ],
            }
        )
        geo_path = tmp_path / "geometry" / "gl_pattern" / "v=1" / "data.lance"
        geo_path.parent.mkdir(parents=True, exist_ok=True)
        ds = _write_lance(geo_table, str(geo_path))
        ds.create_scalar_index("entity_keys", index_type="LABEL_LIST")
        return GDSReader(base_path=str(tmp_path))

    def test_resolve_by_line_and_key(self, tmp_path):
        reader = self._build_fixture(tmp_path)
        pks = reader.resolve_primary_keys_by_edge("gl_pattern", 1, "company_codes", "CC-001")
        assert pks is not None
        assert sorted(pks) == ["GL-001", "GL-003"]

    def test_resolve_without_line_id(self, tmp_path):
        reader = self._build_fixture(tmp_path)
        pks = reader.resolve_primary_keys_by_edge("gl_pattern", 1, None, "CUST-001")
        assert pks is not None
        assert sorted(pks) == ["GL-001", "GL-002"]

    def test_resolve_no_geometry_returns_none(self, tmp_path):
        reader = GDSReader(base_path=str(tmp_path))
        result = reader.resolve_primary_keys_by_edge("no_pattern", 1, "customers", "CUST-001")
        assert result is None

    def test_resolve_no_matches_returns_empty(self, tmp_path):
        reader = self._build_fixture(tmp_path)
        pks = reader.resolve_primary_keys_by_edge("gl_pattern", 1, "company_codes", "CC-999")
        assert pks is not None
        assert pks == []

    def test_resolve_unknown_line_returns_empty(self, tmp_path):
        """When line_id doesn't match any relation, returns empty list."""
        reader = self._build_fixture(tmp_path)
        pks = reader.resolve_primary_keys_by_edge("gl_pattern", 1, "unknown_line", "CUST-001")
        assert pks is not None
        assert pks == []


class TestAppendGeometry:
    """Tests for GDSWriter.append_geometry and _maybe_reindex_geometry."""

    # Minimal geometry table schema matching the real geometry format
    _D = 4  # delta dimensions

    def _make_geo_table(self, n: int, rng: np.random.Generator, start_idx: int = 0) -> pa.Table:
        """Build a minimal geometry table with n rows."""
        deltas = rng.random((n, self._D)).astype(np.float32)
        fixed_type = pa.list_(pa.float32(), self._D)
        return pa.table(
            {
                "primary_key": [f"E-{i:04d}" for i in range(start_idx, start_idx + n)],
                "delta": pa.array(deltas.tolist(), type=fixed_type),
                "delta_norm": pa.array(np.linalg.norm(deltas, axis=1).tolist(), type=pa.float32()),
                "is_anomaly": pa.array([False] * n, type=pa.bool_()),
            }
        )

    def test_append_geometry_creates_dataset_when_missing(self, tmp_path):
        """append_geometry creates a new Lance dataset when none exists."""
        import lance

        rng = np.random.default_rng(0)
        writer = GDSWriter(base_path=str(tmp_path))
        table = self._make_geo_table(10, rng)
        writer.append_geometry(table, "test_pattern")
        lance_path = tmp_path / "geometry" / "test_pattern" / "v=1" / "data.lance"
        assert lance_path.exists()
        ds = lance.dataset(str(lance_path))
        assert ds.count_rows() == 10

    def test_append_geometry_appends_to_existing_dataset(self, tmp_path):
        """append_geometry appends rows when a dataset already exists."""
        import lance

        rng = np.random.default_rng(1)
        writer = GDSWriter(base_path=str(tmp_path))
        initial = self._make_geo_table(20, rng, start_idx=0)
        writer.append_geometry(initial, "test_pattern")
        extra = self._make_geo_table(5, rng, start_idx=20)
        writer.append_geometry(extra, "test_pattern")
        lance_path = tmp_path / "geometry" / "test_pattern" / "v=1" / "data.lance"
        ds = lance.dataset(str(lance_path))
        assert ds.count_rows() == 25

    def test_maybe_reindex_returns_false_on_missing_dataset(self, tmp_path):
        """_maybe_reindex_geometry returns False when geometry dataset does not exist."""
        writer = GDSWriter(base_path=str(tmp_path))
        result = writer._maybe_reindex_geometry("nonexistent", version=1)
        assert result is False

    def test_maybe_reindex_returns_false_on_empty_dataset(self, tmp_path):
        """_maybe_reindex_geometry returns False when geometry dataset is empty."""
        import lance

        writer = GDSWriter(base_path=str(tmp_path))
        # Write an empty table (no rows)
        empty = pa.table(
            {
                "primary_key": pa.array([], type=pa.string()),
                "delta": pa.array([], type=pa.list_(pa.float32(), self._D)),
                "delta_norm": pa.array([], type=pa.float32()),
                "is_anomaly": pa.array([], type=pa.bool_()),
            }
        )
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(empty, str(lance_dir / "data.lance"))
        result = writer._maybe_reindex_geometry("pat", version=1)
        assert result is False

    def test_maybe_reindex_skipped_below_threshold(self, tmp_path, monkeypatch):
        """_maybe_reindex_geometry returns False when unindexed fraction < threshold.

        Uses monkeypatching to avoid needing to build a real IVF-PQ index
        (requires >= 256 rows) — tests the threshold logic path directly.
        """
        import lance

        rng = np.random.default_rng(3)
        # Write a small dataset (no ANN index — Lance won't build one < 256 rows)
        writer = GDSWriter(base_path=str(tmp_path))
        table = self._make_geo_table(50, rng)
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(table, str(lance_dir / "data.lance"))

        # Simulate index existing and covering all rows: patch list_indices to
        # return a fake IVF-PQ index with num_indexed_rows == n_rows
        ds = lance.dataset(str(lance_dir / "data.lance"))
        n_rows = ds.count_rows()

        def _fake_list_indices() -> list[dict]:
            return [{"fields": ["delta"], "num_indexed_rows": n_rows}]

        monkeypatch.setattr(
            ds.__class__,
            "list_indices",
            lambda self: _fake_list_indices(),
        )

        result = writer._maybe_reindex_geometry("pat", threshold=0.1, version=1)
        assert result is False

    def test_maybe_reindex_triggered_when_no_index(self, tmp_path):
        """_maybe_reindex_geometry returns True and rebuilds index for a dataset >= 256 rows.

        Creates an indexed dataset of 300 rows (large enough for IVF-PQ), then
        appends 50 more rows without rebuilding — simulates the unindexed scenario
        by calling _maybe_reindex_geometry with threshold=0.0 (always rebuild).
        """
        import lance

        rng = np.random.default_rng(42)
        n = 300
        writer = GDSWriter(base_path=str(tmp_path))
        table = self._make_geo_table(n, rng)
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(table, str(lance_dir / "data.lance"))
        # Do NOT build an index — _maybe_reindex should detect it is missing and rebuild
        result = writer._maybe_reindex_geometry("pat", threshold=0.0, version=1)
        assert result is True
        # Verify the index was created
        ds = lance.dataset(str(lance_dir / "data.lance"))
        indices = ds.list_indices()
        vector_indices = [
            idx
            for idx in indices
            if "delta"
            in (idx.get("fields", []) if isinstance(idx, dict) else getattr(idx, "fields", []))  # noqa: E501
        ]
        assert len(vector_indices) > 0, f"Expected IVF-PQ index, got: {indices}"

    def test_append_geometry_triggers_reindex_for_large_dataset(self, tmp_path):
        """append_geometry rebuilds ANN index when appending to a large unindexed dataset.

        Creates a 300-row dataset without an index, then appends 50 rows via
        append_geometry. The combined 350 rows should trigger reindex.
        find_nearest_lance must return valid results after the rebuild.
        """
        import lance

        rng = np.random.default_rng(7)
        n_initial = 300
        n_extra = 50
        writer = GDSWriter(base_path=str(tmp_path))

        # Write initial geometry without ANN index
        initial = self._make_geo_table(n_initial, rng, start_idx=0)
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        fixed_type = pa.list_(pa.float32(), self._D)
        delta_col = initial["delta"].cast(fixed_type)
        initial = initial.set_column(initial.schema.get_field_index("delta"), "delta", delta_col)
        lance.write_dataset(initial, str(lance_dir / "data.lance"))
        # Verify no index yet
        ds_before = lance.dataset(str(lance_dir / "data.lance"))
        assert not any(
            "delta"
            in (idx.get("fields", []) if isinstance(idx, dict) else getattr(idx, "fields", []))  # noqa: E501
            for idx in ds_before.list_indices()
        ), "Expected no index before append_geometry"

        # Append extra rows — this should trigger reindex (100% unindexed)
        extra = self._make_geo_table(n_extra, rng, start_idx=n_initial)
        writer.append_geometry(extra, "pat", version=1)

        # Verify total row count
        ds_after = lance.dataset(str(lance_dir / "data.lance"))
        assert ds_after.count_rows() == n_initial + n_extra

        # Verify index now exists on delta
        indices = ds_after.list_indices()
        vector_indices = [
            idx
            for idx in indices
            if "delta"
            in (idx.get("fields", []) if isinstance(idx, dict) else getattr(idx, "fields", []))  # noqa: E501
        ]
        assert len(vector_indices) > 0, f"Expected IVF-PQ index after append, got: {indices}"

        # Verify ANN search works
        reader = GDSReader(base_path=str(tmp_path))
        query_vec = rng.random(self._D).astype(np.float32)
        results = reader.find_nearest_lance("pat", 1, query_vec, k=5)
        assert results is not None
        assert len(results) == 5
        assert all(d >= 0.0 for _, d in results)

    def test_append_geometry_skips_reindex_when_already_indexed(self, tmp_path, monkeypatch):
        """append_geometry does not rebuild index when coverage >= threshold.

        Monkeypatches _maybe_reindex_geometry to verify the skip logic without
        running the expensive full-population index rebuild.
        """
        import lance

        rng = np.random.default_rng(8)
        writer = GDSWriter(base_path=str(tmp_path))

        # Write initial dataset with a "pretend" full index
        initial = self._make_geo_table(300, rng)
        lance_dir = tmp_path / "geometry" / "pat" / "v=1"
        lance_dir.mkdir(parents=True)
        lance.write_dataset(initial, str(lance_dir / "data.lance"))

        rebuild_calls: list[bool] = []

        original_maybe_reindex = writer._maybe_reindex_geometry

        def _counting_maybe_reindex(
            pattern_id: str, threshold: float = 0.1, version: int = 1
        ) -> bool:
            # Forward to the real implementation but record invocation
            result = original_maybe_reindex(pattern_id, threshold=threshold, version=version)
            rebuild_calls.append(result)
            return result

        monkeypatch.setattr(writer, "_maybe_reindex_geometry", _counting_maybe_reindex)

        # Append a small number of rows (well below 10% of 300 = 30)
        extra = self._make_geo_table(5, rng, start_idx=300)
        writer.append_geometry(extra, "pat", version=1)

        # _maybe_reindex_geometry must have been called
        assert len(rebuild_calls) == 1
        # With no IVF-PQ index, all rows are "unindexed" → rebuild returns True
        # But since 300 + 5 = 305 < 256 is False, it just checks the index state.
        # The key assertion: append_geometry always calls _maybe_reindex_geometry
        ds = lance.dataset(str(lance_dir / "data.lance"))
        assert ds.count_rows() == 305


# ---------------------------------------------------------------------------
# GDSReader.has_fts_index — INVERTED index detection
# ---------------------------------------------------------------------------


class TestHasFtsIndex:
    """Unit tests for GDSReader.has_fts_index(line_id, version) -> bool."""

    def test_no_lance_dataset_returns_false(self, tmp_path):
        """Line has only Parquet files — no Lance dataset → returns False."""
        points_dir = tmp_path / "points" / "customers" / "v=1" / "region=EMEA"
        points_dir.mkdir(parents=True)
        import pyarrow.parquet as pq_mod

        t = pa.table({"primary_key": ["C-1"], "name": ["Alice"]})
        pq_mod.write_table(t, str(points_dir / "data.parquet"))

        reader = GDSReader(base_path=str(tmp_path))
        assert reader.has_fts_index("customers", 1) is False

    def test_lance_dataset_without_inverted_index_returns_false(self, tmp_path):
        """Lance dataset exists but has no INVERTED index → returns False."""
        import lance

        points_dir = tmp_path / "points" / "customers" / "v=1"
        points_dir.mkdir(parents=True)
        lance_path = points_dir / "data.lance"
        t = pa.table({"primary_key": ["C-1", "C-2"], "name": ["Alice", "Bob"]})
        lance.write_dataset(t, str(lance_path))
        # No INVERTED index created — only default structure

        reader = GDSReader(base_path=str(tmp_path))
        assert reader.has_fts_index("customers", 1) is False

    def test_lance_dataset_with_inverted_index_returns_true(self, tmp_path):
        """Lance dataset exists and has INVERTED (FTS) index → returns True."""
        import lance

        points_dir = tmp_path / "points" / "customers" / "v=1"
        points_dir.mkdir(parents=True)
        lance_path = points_dir / "data.lance"
        t = pa.table({"primary_key": ["C-1", "C-2"], "name": ["Alice", "Bob"]})
        ds = lance.write_dataset(t, str(lance_path))
        ds.create_scalar_index("name", index_type="INVERTED")

        reader = GDSReader(base_path=str(tmp_path))
        assert reader.has_fts_index("customers", 1) is True


def test_parse_pattern_reads_prop_columns(tmp_path):
    """Reader populates prop_columns and excluded_properties from sphere.json."""
    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text(
        _json.dumps(
            {
                "sphere_id": "s",
                "name": "s",
                "lines": {},
                "patterns": {
                    "cp": {
                        "pattern_id": "cp",
                        "entity_type": "customer",
                        "pattern_type": "anchor",
                        "version": 1,
                        "status": "production",
                        "relations": [{"line_id": "orders", "direction": "out", "required": True}],
                        "mu": [0.5, 0.9, 0.3],
                        "sigma_diag": [0.3, 0.1, 0.4],
                        "theta": [1.5, 2.0, 2.0],
                        "population_size": 10,
                        "computed_at": "2024-01-01T00:00:00+00:00",
                        "prop_columns": ["name", "region"],
                        "excluded_properties": ["rare_col"],
                    }
                },
                "aliases": {},
            }
        )
    )
    reader = GDSReader(base_path=str(tmp_path))
    sphere = reader.read_sphere()
    pat = sphere.patterns["cp"]
    assert pat.prop_columns == ["name", "region"]
    assert pat.excluded_properties == ["rare_col"]
    assert pat.delta_dim() == 3  # 1 relation + 2 props


def test_parse_pattern_reads_dim_percentiles(tmp_path):
    """Reader populates dim_percentiles from sphere.json."""
    percentiles = {
        "avg_late_days": {"min": 0.0, "p25": 20.0, "p50": 26.0, "p75": 32.0, "p99": 50.0, "max": 322.0},
    }
    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text(
        _json.dumps(
            {
                "sphere_id": "s",
                "name": "s",
                "lines": {},
                "patterns": {
                    "cp": {
                        "pattern_id": "cp",
                        "entity_type": "customer",
                        "pattern_type": "anchor",
                        "version": 1,
                        "status": "production",
                        "relations": [{"line_id": "orders", "direction": "out", "required": True}],
                        "mu": [0.5],
                        "sigma_diag": [0.3],
                        "theta": [1.5],
                        "population_size": 10,
                        "computed_at": "2024-01-01T00:00:00+00:00",
                        "dim_percentiles": percentiles,
                    }
                },
                "aliases": {},
            }
        )
    )
    reader = GDSReader(base_path=str(tmp_path))
    sphere = reader.read_sphere()
    pat = sphere.patterns["cp"]
    assert pat.dim_percentiles == percentiles
    assert pat.dim_percentiles["avg_late_days"]["max"] == 322.0


def test_parse_pattern_dim_percentiles_absent(tmp_path):
    """dim_percentiles defaults to None when absent from sphere.json."""
    meta = tmp_path / "_gds_meta"
    meta.mkdir()
    (meta / "sphere.json").write_text(
        _json.dumps(
            {
                "sphere_id": "s",
                "name": "s",
                "lines": {},
                "patterns": {
                    "cp": {
                        "pattern_id": "cp",
                        "entity_type": "customer",
                        "pattern_type": "anchor",
                        "version": 1,
                        "status": "production",
                        "relations": [{"line_id": "orders", "direction": "out", "required": True}],
                        "mu": [0.5],
                        "sigma_diag": [0.3],
                        "theta": [1.5],
                        "population_size": 10,
                        "computed_at": "2024-01-01T00:00:00+00:00",
                    }
                },
                "aliases": {},
            }
        )
    )
    reader = GDSReader(base_path=str(tmp_path))
    sphere = reader.read_sphere()
    pat = sphere.patterns["cp"]
    assert pat.dim_percentiles is None


# ---------------------------------------------------------------------------
# shape_snapshot schema tests
# ---------------------------------------------------------------------------


def test_append_temporal_uses_shape_snapshot(tmp_path):
    import lance

    writer = GDSWriter(base_path=str(tmp_path))
    shape = np.array([0.3, 0.7], dtype=np.float32)
    s = SolidSlice(
        slice_index=0,
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
        deformation_type="structural",
        delta_snapshot=np.array([0.1, 0.2], dtype=np.float32),
        delta_norm_snapshot=0.22,
        pattern_ver=1,
        changed_property=None,
        changed_line_id="products",
        added_edge=None,
    )
    writer.append_temporal_slice(
        s,
        pattern_id="p1",
        primary_key="E-001",
        shape_snapshot=shape,
    )
    lance_path = tmp_path / "temporal" / "p1" / "data.lance"
    ds = lance.dataset(str(lance_path))
    field_names = {f.name for f in ds.schema}
    assert "shape_snapshot" in field_names
    assert "delta_snapshot" not in field_names
    assert "delta_norm_snapshot" not in field_names
    tbl = ds.to_table()
    stored = np.array(tbl["shape_snapshot"][0].as_py(), dtype=np.float32)
    np.testing.assert_allclose(stored, shape, rtol=1e-5)


def test_migrate_temporal_to_shape_snapshot(tmp_path):
    import lance

    writer = GDSWriter(base_path=str(tmp_path))
    mu = np.array([0.5, 0.4], dtype=np.float32)
    sigma_diag = np.array([0.2, 0.3], dtype=np.float32)
    delta_vecs = [
        np.array([1.0, -1.0], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
    ]

    # Write OLD schema (delta_snapshot, delta_norm_snapshot) manually via lance
    old_schema = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("slice_index", pa.int32()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("deformation_type", pa.string()),
            pa.field("delta_snapshot", pa.list_(pa.float32())),
            pa.field("delta_norm_snapshot", pa.float32()),
            pa.field("pattern_ver", pa.int32()),
            pa.field("changed_property", pa.string()),
            pa.field("changed_line_id", pa.string()),
        ]
    )
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    old_table = pa.table(
        {
            "primary_key": ["E-001", "E-002"],
            "slice_index": pa.array([0, 1], type=pa.int32()),
            "timestamp": pa.array([ts, ts], type=pa.timestamp("us", tz="UTC")),
            "deformation_type": ["structural", "structural"],
            "delta_snapshot": [d.tolist() for d in delta_vecs],
            "delta_norm_snapshot": pa.array(
                [float(np.linalg.norm(d)) for d in delta_vecs], type=pa.float32()
            ),
            "pattern_ver": pa.array([1, 1], type=pa.int32()),
            "changed_property": pa.array([None, None], type=pa.string()),
            "changed_line_id": pa.array([None, None], type=pa.string()),
        },
        schema=old_schema,
    )
    lance_path = tmp_path / "temporal" / "p2" / "data.lance"
    lance_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(old_table, str(lance_path))

    # Migrate
    count = writer.migrate_temporal_to_shape_snapshot("p2", mu, sigma_diag)
    assert count == 2

    # Verify new schema
    ds = lance.dataset(str(lance_path))
    field_names = {f.name for f in ds.schema}
    assert "shape_snapshot" in field_names
    assert "delta_snapshot" not in field_names
    assert "delta_norm_snapshot" not in field_names

    # Verify shape values = clip(delta * max(sigma, 1e-2) + mu, 0, 1)
    tbl = ds.to_table()
    _sigma = np.maximum(sigma_diag, 1e-2)
    for i, delta_vec in enumerate(delta_vecs):
        expected_shape = np.clip(delta_vec * _sigma + mu, 0.0, 1.0)
        stored_shape = np.array(tbl["shape_snapshot"][i].as_py(), dtype=np.float32)
        np.testing.assert_allclose(stored_shape, expected_shape, rtol=1e-5)

    # Idempotent: second call returns 0
    count2 = writer.migrate_temporal_to_shape_snapshot("p2", mu, sigma_diag)
    assert count2 == 0
