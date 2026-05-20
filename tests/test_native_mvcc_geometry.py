# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Native Lance MVCC geometry layout — sphere format 3.0.

Geometry persists as a single ``geometry/<pid>/data.lance`` dataset; each
calibration epoch is a Lance internal version pinned by an ``epoch_<N>`` tag.
"""
from __future__ import annotations

from pathlib import Path

import lance as _lance
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.sphere import HyperSphere


def _build_minimal_sphere(tmp_path: Path) -> tuple[str, str]:
    """Build a small sphere with one event pattern; return (sphere_root, pid)."""
    b = GDSBuilder("native_mvcc_test", str(tmp_path / "gds_mvcc"))
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
            RelationSpec(
                "customers", fk_col="cust_id", direction="in", required=True,
            ),
        ],
    )
    out = b.build()
    return out, "order_pattern"


class TestNativeMvccGeometryLayout:
    def test_flat_geometry_lance_path_exists(self, tmp_path: Path) -> None:
        out, pid = _build_minimal_sphere(tmp_path)
        flat = Path(out) / "geometry" / pid / "data.lance"
        assert flat.exists(), "geometry/<pid>/data.lance must exist on a 3.0 sphere"

    def test_no_versioned_geometry_dir(self, tmp_path: Path) -> None:
        out, pid = _build_minimal_sphere(tmp_path)
        versioned = Path(out) / "geometry" / pid / "v=1"
        assert not versioned.exists(), (
            "geometry/<pid>/v=1 must not exist on a 3.0 sphere"
        )

    def test_epoch_1_tag_present_after_build(self, tmp_path: Path) -> None:
        out, pid = _build_minimal_sphere(tmp_path)
        ds = _lance.dataset(str(Path(out) / "geometry" / pid / "data.lance"))
        tags = ds.tags.list()
        assert "epoch_1" in tags, f"expected epoch_1 tag, got {list(tags)}"

    def test_reader_returns_rows_for_epoch_1(self, tmp_path: Path) -> None:
        out, pid = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)
        version = sphere._sphere.patterns[pid].version
        table = sphere._reader.read_geometry(pid, version, columns=["primary_key"])
        assert table.num_rows > 0


class TestNativeMvccRecalibrateIsolation:
    def test_recalibrate_overwrites_in_place_no_versioned_dir(
        self, tmp_path: Path,
    ) -> None:
        out, pid = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)
        session = sphere.session("agent-recalib")
        session.recalibrate(pid)
        flat = Path(out) / "geometry" / pid / "data.lance"
        assert flat.exists()
        versioned = Path(out) / "geometry" / pid / "v=2"
        assert not versioned.exists(), (
            "recalibrate must not create v=2/ — native MVCC keeps a flat dataset"
        )
        session.close()

    def test_pre_recalibrate_session_isolated_after_recalibrate(
        self, tmp_path: Path,
    ) -> None:
        """Session opened before recalibrate sees the pre-recalibrate snapshot."""
        out, pid = _build_minimal_sphere(tmp_path)
        sphere = HyperSphere.open(out)
        version = sphere._sphere.patterns[pid].version

        pre = sphere.session("agent-pre")
        nav_pre = pre.navigator()
        pre_norms = nav_pre._storage.read_geometry(
            pid, version, columns=["primary_key", "delta_norm"],
        )["delta_norm"].to_pylist()

        recalib_session = sphere.session("agent-recalib")
        recalib_session.recalibrate(pid)
        recalib_session.close()

        same_norms_after = nav_pre._storage.read_geometry(
            pid, version, columns=["primary_key", "delta_norm"],
        )["delta_norm"].to_pylist()

        assert same_norms_after == pre_norms, (
            "Pre-recalibrate session must observe unchanged geometry"
        )

        post = sphere.session("agent-post")
        nav_post = post.navigator()
        post_norms = nav_post._storage.read_geometry(
            pid, version, columns=["primary_key", "delta_norm"],
        )["delta_norm"].to_pylist()
        # New session sees latest version (may differ from pre-recalibrate norms
        # only when the recalibration actually changes them; the test only
        # verifies that the new session reads the latest snapshot without error).
        assert len(post_norms) == len(pre_norms)

        pre.close()
        post.close()
