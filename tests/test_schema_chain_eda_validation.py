"""Cross-block parse-time validation for chain_lines.edge_dim_aggregations."""
from pathlib import Path

import pytest
import yaml

from hypertopos.cli.schema import parse_config


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _base_yaml() -> dict:
    return {
        "version": "0.1.0",
        "sphere_id": "test",
        "format_version": "3.0",
        "sources": {"events": {"path": "events.csv"}},
        "lines": {
            "events": {"source": "events", "key": "primary_key", "role": "event"},
        },
        "patterns": {
            "tx_pattern": {
                "type": "event",
                "entity_line": "events",
                "relations": [],
                "edge_dimensions": [
                    {"find_motif_structuring": {
                        "amt1_min": 10000.0, "amt2_max": 5000.0,
                        "time_window_hours": 24.0,
                    }},
                ],
            },
        },
        "chain_lines": {
            "tx_chains": {
                "event_line": "events",
                "from_col": "from_acct",
                "to_col": "to_acct",
                "features": ["hop_count"],
            },
        },
    }


def test_chain_eda_event_line_match_ok(tmp_path):
    """Matching event_line passes parse-time validation."""
    spec = _base_yaml()
    spec["chain_lines"]["tx_chains"]["edge_dim_aggregations"] = {
        "from": "tx_pattern",
        "dims": ["find_motif_structuring"],
    }
    yaml_path = _write_yaml(tmp_path / "sphere.yaml", spec)
    cfg = parse_config(yaml_path)
    assert cfg.chain_lines["tx_chains"].edge_dim_aggregations == {
        "from": "tx_pattern",
        "dims": ["find_motif_structuring"],
    }


def test_chain_eda_event_line_mismatch_raises(tmp_path):
    """Mismatched event_line raises ValueError at parse time."""
    spec = _base_yaml()
    spec["lines"]["other_events"] = {
        "source": "events", "key": "primary_key", "role": "event",
    }
    spec["patterns"]["other_pattern"] = {
        "type": "event",
        "entity_line": "other_events",
        "relations": [],
        "edge_dimensions": [
            {"find_motif_structuring": {
                "amt1_min": 10000.0, "amt2_max": 5000.0,
                "time_window_hours": 24.0,
            }},
        ],
    }
    spec["chain_lines"]["tx_chains"]["edge_dim_aggregations"] = {
        "from": "other_pattern",
        "dims": ["find_motif_structuring"],
    }
    yaml_path = _write_yaml(tmp_path / "sphere.yaml", spec)
    with pytest.raises(ValueError, match="event_keys cannot match in the join"):
        parse_config(yaml_path)


def test_chain_eda_unknown_from_raises(tmp_path):
    """Unknown event pattern in from: raises."""
    spec = _base_yaml()
    spec["chain_lines"]["tx_chains"]["edge_dim_aggregations"] = {
        "from": "ghost_pattern",
        "dims": ["find_motif_structuring"],
    }
    yaml_path = _write_yaml(tmp_path / "sphere.yaml", spec)
    with pytest.raises(ValueError, match="must reference a registered event pattern"):
        parse_config(yaml_path)


def test_chain_eda_anchor_pattern_in_from_raises(tmp_path):
    """from: must reference an event pattern, not anchor."""
    spec = _base_yaml()
    spec["lines"]["accounts"] = {
        "source": "events", "key": "primary_key", "role": "anchor",
    }
    spec["patterns"]["acct_pattern"] = {
        "type": "anchor",
        "entity_line": "accounts",
        "relations": [],
    }
    spec["chain_lines"]["tx_chains"]["edge_dim_aggregations"] = {
        "from": "acct_pattern",
        "dims": ["find_motif_structuring"],
    }
    yaml_path = _write_yaml(tmp_path / "sphere.yaml", spec)
    with pytest.raises(ValueError, match="must be an event pattern"):
        parse_config(yaml_path)


def test_chain_eda_missing_from_raises(tmp_path):
    """Missing 'from' field raises."""
    spec = _base_yaml()
    spec["chain_lines"]["tx_chains"]["edge_dim_aggregations"] = {
        "dims": ["find_motif_structuring"],
    }
    yaml_path = _write_yaml(tmp_path / "sphere.yaml", spec)
    with pytest.raises(ValueError, match="must specify 'from"):
        parse_config(yaml_path)
