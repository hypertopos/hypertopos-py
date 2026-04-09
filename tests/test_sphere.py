# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from hypertopos.model.objects import SolidSlice
from hypertopos.sphere import HyperSphere
from hypertopos.storage.writer import GDSWriter


def test_hypersphere_open(sphere_path):
    hs = HyperSphere.open(sphere_path)
    assert hs is not None


def test_hypersession_context_manager(sphere_path):
    hs = HyperSphere.open(sphere_path)
    with hs.session("agent-001") as session:
        nav = session.navigator()
        assert nav is not None


def test_session_close_purge_temporal(sphere_path):
    hs = HyperSphere.open(sphere_path)
    agent_id = "test-purge-agent"
    writer = GDSWriter(base_path=sphere_path)
    s = SolidSlice(
        slice_index=99,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        deformation_type="internal",
        delta_snapshot=np.array([0.1], dtype=np.float32),
        delta_norm_snapshot=0.1,
        pattern_ver=1,
        changed_property=None,
        changed_line_id=None,
        added_edge=None,
    )
    writer.append_temporal_slice(
        s,
        "customer_pattern",
        "CUST-001",
        shape_snapshot=np.array([0.5], dtype=np.float32),
        agent_id=agent_id,
    )
    agent_dir = Path(sphere_path) / "temporal" / "_agents" / agent_id
    assert agent_dir.exists()

    session = hs.session(agent_id)
    session.close(purge_temporal=True)
    assert not agent_dir.exists()


def test_session_close_no_purge_by_default(sphere_path):
    hs = HyperSphere.open(sphere_path)
    agent_id = "test-no-purge"
    writer = GDSWriter(base_path=sphere_path)
    s = SolidSlice(
        slice_index=99,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        deformation_type="internal",
        delta_snapshot=np.array([0.1], dtype=np.float32),
        delta_norm_snapshot=0.1,
        pattern_ver=1,
        changed_property=None,
        changed_line_id=None,
        added_edge=None,
    )
    writer.append_temporal_slice(
        s,
        "customer_pattern",
        "CUST-001",
        shape_snapshot=np.array([0.5], dtype=np.float32),
        agent_id=agent_id,
    )
    agent_dir = Path(sphere_path) / "temporal" / "_agents" / agent_id

    session = hs.session(agent_id)
    session.close()
    assert agent_dir.exists()

    # cleanup
    import shutil

    shutil.rmtree(agent_dir)


def test_parse_alias_with_cutting_plane(tmp_path):
    """Reader parses cutting_plane from sphere.json into AliasFilter."""
    import json

    from hypertopos.storage.reader import GDSReader

    sphere_json = {
        "sphere_id": "test",
        "name": "Test",
        "lines": {},
        "patterns": {
            "p1": {
                "pattern_id": "p1",
                "entity_type": "e",
                "pattern_type": "anchor",
                "version": 1,
                "status": "production",
                "relations": [
                    {"line_id": "a", "direction": "in", "required": True},
                    {"line_id": "b", "direction": "in", "required": True},
                ],
                "mu": [0.5, 0.5],
                "sigma_diag": [0.1, 0.1],
                "theta": [0.5, 0.5],
                "population_size": 100,
                "computed_at": "2025-01-01T00:00:00+00:00",
            }
        },
        "aliases": {
            "seg1": {
                "alias_id": "seg1",
                "base_pattern_id": "p1",
                "version": 1,
                "status": "production",
                "filter": {
                    "include_relations": ["a"],
                    "cutting_plane": {"normal": [1.0, 1.0], "bias": 1.5},
                },
                "derived_pattern": {
                    "mu": [0.8, 0.8],
                    "sigma_diag": [0.1, 0.1],
                    "theta": [0.3, 0.3],
                    "population_size": 50,
                    "computed_at": "2025-01-01T00:00:00+00:00",
                },
            }
        },
    }
    meta_dir = tmp_path / "_gds_meta"
    meta_dir.mkdir()
    (meta_dir / "sphere.json").write_text(json.dumps(sphere_json))
    reader = GDSReader(str(tmp_path))
    sphere = reader.read_sphere()

    alias = sphere.aliases["seg1"]
    assert alias.filter.cutting_plane is not None
    assert alias.filter.cutting_plane.bias == 1.5
    assert alias.filter.cutting_plane.normal == [1.0, 1.0]


def test_parse_alias_cutting_plane_validates_dim(tmp_path):
    """Reader raises ValueError when cutting_plane.normal length != pattern.delta_dim."""
    import json

    import pytest
    from hypertopos.storage.reader import GDSReader

    sphere_json = {
        "sphere_id": "test",
        "name": "Test",
        "lines": {},
        "patterns": {
            "p1": {
                "pattern_id": "p1",
                "entity_type": "e",
                "pattern_type": "anchor",
                "version": 1,
                "status": "production",
                "relations": [
                    {"line_id": "a", "direction": "in", "required": True},
                ],
                "mu": [0.5],
                "sigma_diag": [0.1],
                "theta": [0.5],
                "population_size": 100,
                "computed_at": "2025-01-01T00:00:00+00:00",
            }
        },
        "aliases": {
            "seg1": {
                "alias_id": "seg1",
                "base_pattern_id": "p1",
                "version": 1,
                "status": "production",
                "filter": {
                    "include_relations": [],
                    "cutting_plane": {"normal": [1.0, 1.0], "bias": 0.5},
                },
                "derived_pattern": {
                    "mu": [0.5],
                    "sigma_diag": [0.1],
                    "theta": [0.3],
                    "population_size": 50,
                    "computed_at": "2025-01-01T00:00:00+00:00",
                },
            }
        },
    }
    meta_dir = tmp_path / "_gds_meta"
    meta_dir.mkdir()
    (meta_dir / "sphere.json").write_text(json.dumps(sphere_json))
    reader = GDSReader(str(tmp_path))
    with pytest.raises(ValueError, match="cutting_plane"):
        reader.read_sphere()
