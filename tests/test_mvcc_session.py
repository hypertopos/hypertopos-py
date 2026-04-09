# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""MVCC session isolation — session must not see writes after open."""

from __future__ import annotations

from pathlib import Path

import lance as _lance
import pyarrow as pa
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.sphere import HyperSphere

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_minimal_sphere(tmp_path: Path) -> str:
    """Build a minimal sphere with one anchor pattern and return its path."""
    b = GDSBuilder("mvcc_test", str(tmp_path / "gds_mvcc"))
    b.add_line(
        "customers",
        [
            {"cust_id": "C-1", "name": "Alpha"},
            {"cust_id": "C-2", "name": "Beta"},
            {"cust_id": "C-3", "name": "Gamma"},
        ],
        key_col="cust_id",
        source_id="test",
    )
    b.add_line(
        "orders",
        [
            {"order_id": "O-1", "cust_id": "C-1"},
            {"order_id": "O-2", "cust_id": "C-2"},
            {"order_id": "O-3", "cust_id": "C-3"},
            {"order_id": "O-4", "cust_id": "C-1"},
        ],
        key_col="order_id",
        source_id="test",
        role="event",
    )
    b.add_pattern(
        "order_pattern",
        pattern_type="event",
        entity_line="orders",
        relations=[
            RelationSpec("customers", fk_col="cust_id", direction="in", required=True),
        ],
    )
    return b.build()


def _append_geometry_row(geo_lance_path: str, new_pk: str) -> None:
    """Append a single new row to an existing geometry Lance dataset.

    Reads the schema from the existing dataset and appends a minimal zero row
    so the row count changes.  This simulates a concurrent write arriving after
    session open.
    """
    ds = _lance.dataset(geo_lance_path)
    schema = ds.schema

    _EDGE_STRUCT = pa.struct(
        [
            pa.field("line_id", pa.string()),
            pa.field("point_key", pa.string()),
            pa.field("status", pa.string()),
            pa.field("direction", pa.string()),
        ]
    )

    row_data: dict[str, pa.Array] = {}
    for field in schema:
        fname = field.name
        ftype = field.type
        if fname == "primary_key":
            row_data[fname] = pa.array([new_pk], type=pa.string())
        elif pa.types.is_boolean(ftype):
            row_data[fname] = pa.array([False], type=ftype)
        elif pa.types.is_floating(ftype):
            row_data[fname] = pa.array([0.0], type=ftype)
        elif pa.types.is_integer(ftype):
            row_data[fname] = pa.array([0], type=ftype)
        elif pa.types.is_string(ftype) or pa.types.is_large_string(ftype):
            row_data[fname] = pa.array([""], type=ftype)
        elif pa.types.is_list(ftype):
            if ftype == pa.list_(_EDGE_STRUCT):
                row_data[fname] = pa.array([[]], type=ftype)
            elif pa.types.is_floating(ftype.value_type):
                row_data[fname] = pa.array([[0.0]], type=ftype)
            elif pa.types.is_integer(ftype.value_type):
                row_data[fname] = pa.array([[0]], type=ftype)
            elif pa.types.is_string(ftype.value_type) or pa.types.is_large_string(ftype.value_type):
                row_data[fname] = pa.array([[]], type=ftype)
            else:
                row_data[fname] = pa.array([[]], type=ftype)
        elif pa.types.is_timestamp(ftype):
            import datetime

            row_data[fname] = pa.array(
                [datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)],
                type=ftype,
            )
        elif pa.types.is_fixed_size_list(ftype):
            n = ftype.list_size
            row_data[fname] = pa.array([[0.0] * n], type=ftype)
        else:
            row_data[fname] = pa.array([None], type=ftype)

    new_table = pa.table(row_data, schema=schema)
    _lance.write_dataset(new_table, geo_lance_path, mode="append")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMvccSessionIsolation:
    def test_session_does_not_see_write_after_open(self, tmp_path: Path) -> None:
        """A session opened BEFORE a write must NOT see the new row."""
        out = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)

        pattern_id = next(
            k for k, p in sphere._sphere.patterns.items() if p.pattern_type == "event"
        )
        version = sphere._sphere.patterns[pattern_id].version

        # Open session first — this pins the current Lance version.
        session = sphere.session("agent-mvcc-before")
        nav = session.navigator()

        # Read the row count before write.
        before = nav._storage.read_geometry(pattern_id, version, columns=["primary_key"])
        before_keys = set(before["primary_key"].to_pylist())

        # Append a new row to the geometry dataset AFTER session was opened.
        geo_path = str(Path(out) / "geometry" / pattern_id / f"v={version}" / "data.lance")
        _append_geometry_row(geo_path, "TEST-MVCC-NEW-999")

        # The session must NOT see the newly written row.
        after = nav._storage.read_geometry(pattern_id, version, columns=["primary_key"])
        after_keys = set(after["primary_key"].to_pylist())

        assert "TEST-MVCC-NEW-999" not in after_keys, (
            "Session must not see a row written after session open (MVCC isolation broken)"
        )
        assert after_keys == before_keys

        session.close()

    def test_new_session_sees_write_after_it(self, tmp_path: Path) -> None:
        """A session opened AFTER a write must see the new row."""
        out = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)

        pattern_id = next(
            k for k, p in sphere._sphere.patterns.items() if p.pattern_type == "event"
        )
        version = sphere._sphere.patterns[pattern_id].version

        # Append a row BEFORE opening the new session.
        geo_path = str(Path(out) / "geometry" / pattern_id / f"v={version}" / "data.lance")
        _append_geometry_row(geo_path, "TEST-MVCC-AFTER-999")

        # Open a fresh session after the write — it should see the new row.
        session2 = sphere.session("agent-mvcc-after")
        nav2 = session2.navigator()

        keys_after = set(
            nav2._storage.read_geometry(pattern_id, version, columns=["primary_key"])[
                "primary_key"
            ].to_pylist()
        )

        assert "TEST-MVCC-AFTER-999" in keys_after, (
            "A session opened after a write must see the new row"
        )

        session2.close()

    def test_two_concurrent_sessions_are_isolated(self, tmp_path: Path) -> None:
        """Two sessions opened at different times must see different snapshots."""
        out = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)

        pattern_id = next(
            k for k, p in sphere._sphere.patterns.items() if p.pattern_type == "event"
        )
        version = sphere._sphere.patterns[pattern_id].version
        geo_path = str(Path(out) / "geometry" / pattern_id / f"v={version}" / "data.lance")

        # Session A opened before the write.
        session_a = sphere.session("agent-concurrent-a")
        nav_a = session_a.navigator()

        # Write a new row.
        _append_geometry_row(geo_path, "TEST-CONCURRENT-ROW")

        # Session B opened after the write.
        session_b = sphere.session("agent-concurrent-b")
        nav_b = session_b.navigator()

        keys_a = set(
            nav_a._storage.read_geometry(pattern_id, version, columns=["primary_key"])[
                "primary_key"
            ].to_pylist()
        )
        keys_b = set(
            nav_b._storage.read_geometry(pattern_id, version, columns=["primary_key"])[
                "primary_key"
            ].to_pylist()
        )

        assert "TEST-CONCURRENT-ROW" not in keys_a, "Session A must not see the new row"
        assert "TEST-CONCURRENT-ROW" in keys_b, "Session B must see the new row"
        # Session A has fewer rows than session B
        assert len(keys_a) < len(keys_b)

        session_a.close()
        session_b.close()

    def test_session_reader_is_independent_per_session(self, tmp_path: Path) -> None:
        """Each session must hold its own GDSReader instance."""
        out = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)

        session1 = sphere.session("agent-reader-check-1")
        session2 = sphere.session("agent-reader-check-2")

        assert session1._reader is not session2._reader, (
            "Each session must have its own GDSReader for MVCC isolation"
        )
        assert session1._reader is not sphere._reader, (
            "Session reader must not be the shared HyperSphere reader"
        )

        session1.close()
        session2.close()

    def test_pinned_versions_populated_at_session_open(self, tmp_path: Path) -> None:
        """Session reader must have non-empty _pinned_lance_versions after open."""
        out = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)

        session = sphere.session("agent-pin-check")
        pinned = session._reader._pinned_lance_versions

        assert len(pinned) > 0, "Session reader must have at least one pinned Lance version"
        # All values must be non-negative integers (Lance version numbers).
        for key, ver in pinned.items():
            assert isinstance(ver, int) and ver >= 0, (
                f"Pinned version for '{key}' must be a non-negative int, got {ver!r}"
            )

        session.close()
