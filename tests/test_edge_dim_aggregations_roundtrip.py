from __future__ import annotations

import json
from pathlib import Path

from hypertopos.storage.reader import GDSReader


def test_pattern_edge_dim_aggregations_roundtrips_via_sphere_json(tmp_path: Path):
    raw = {
        "sphere_id": "s",
        "format_version": "3.0",
        "name": "s",
        "lines": {},
        "patterns": {
            "p": {
                "pattern_id": "p",
                "entity_type": "x",
                "pattern_type": "anchor",
                "relations": [],
                "mu": [],
                "sigma_diag": [],
                "theta": [],
                "population_size": 0,
                "computed_at": "2026-04-30T00:00:00",
                "version": 1,
                "status": "production",
                "edge_dim_aggregations": {
                    "from": "tx_pattern",
                    "dims": ["pair_edge_count", "find_motif_structuring"],
                },
            },
        },
        "aliases": {},
    }
    (tmp_path / "_gds_meta").mkdir()
    (tmp_path / "_gds_meta" / "sphere.json").write_text(json.dumps(raw))

    sphere = GDSReader(str(tmp_path)).read_sphere()
    pat = sphere.patterns["p"]
    assert pat.edge_dim_aggregations is not None
    assert pat.edge_dim_aggregations.from_event_pattern == "tx_pattern"
    assert pat.edge_dim_aggregations.dims == (
        "pair_edge_count",
        "find_motif_structuring",
    )


def test_pattern_without_edge_dim_aggregations_returns_none(tmp_path: Path):
    raw = {
        "sphere_id": "s",
        "format_version": "3.0",
        "name": "s",
        "lines": {},
        "patterns": {
            "p": {
                "pattern_id": "p",
                "entity_type": "x",
                "pattern_type": "anchor",
                "relations": [],
                "mu": [],
                "sigma_diag": [],
                "theta": [],
                "population_size": 0,
                "computed_at": "2026-04-30T00:00:00",
                "version": 1,
                "status": "production",
            },
        },
        "aliases": {},
    }
    (tmp_path / "_gds_meta").mkdir()
    (tmp_path / "_gds_meta" / "sphere.json").write_text(json.dumps(raw))

    sphere = GDSReader(str(tmp_path)).read_sphere()
    assert sphere.patterns["p"].edge_dim_aggregations is None


def test_pattern_edge_dim_aggregations_dims_default_to_all(tmp_path: Path):
    raw = {
        "sphere_id": "s",
        "format_version": "3.0",
        "name": "s",
        "lines": {},
        "patterns": {
            "p": {
                "pattern_id": "p",
                "entity_type": "x",
                "pattern_type": "anchor",
                "relations": [],
                "mu": [],
                "sigma_diag": [],
                "theta": [],
                "population_size": 0,
                "computed_at": "2026-04-30T00:00:00",
                "version": 1,
                "status": "production",
                "edge_dim_aggregations": {"from": "tx_pattern"},
            },
        },
        "aliases": {},
    }
    (tmp_path / "_gds_meta").mkdir()
    (tmp_path / "_gds_meta" / "sphere.json").write_text(json.dumps(raw))

    sphere = GDSReader(str(tmp_path)).read_sphere()
    pat = sphere.patterns["p"]
    assert pat.edge_dim_aggregations is not None
    assert pat.edge_dim_aggregations.dims is None
