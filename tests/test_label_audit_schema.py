# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""YAML ``label_audit:`` block — schema parsing + round-trip through build."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.cli.schema import LabelAuditConfig, parse_config
from hypertopos.sphere import HyperSphere

# ── helpers ──────────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, body: str) -> Path:
    """Write a minimal sphere.yaml — sources empty so parse_config returns
    a SphereConfig without touching the filesystem-source loader."""
    yaml_path = tmp_path / "sphere.yaml"
    yaml_path.write_text(body, encoding="utf-8")
    return yaml_path


def _yaml_with_block(audit_block_yaml: str | None) -> str:
    """Render a minimal sphere.yaml with optional label_audit block."""
    audit_part = f"\nlabel_audit:\n{audit_block_yaml}" if audit_block_yaml else ""
    return (
        'version: "0.1.0"\n'
        "sphere_id: label_audit_test\n"
        "sources:\n"
        "  src1:\n"
        "    path: data.csv\n"
        "lines:\n"
        "  customers:\n"
        "    source: src1\n"
        "    key: primary_key\n"
        "    role: anchor\n"
        "  orders:\n"
        "    source: src1\n"
        "    key: primary_key\n"
        "    role: event\n"
        "patterns:\n"
        "  account_pattern:\n"
        "    type: event\n"
        "    entity_line: orders\n"
        "    relations:\n"
        "      - line: customers\n"
        "        direction: in\n"
        "        required: true\n"
        f"{audit_part}"
    )


def _build_sphere_with_block(
    tmp_path: Path, audit_block: object | None,
) -> str:
    """Build a tiny sphere optionally carrying a label_audit block.

    The block is set directly on the builder (mimicking what cli.build
    does after parsing the YAML), so this fixture doesn't need to drive
    the full YAML → source-loader path.
    """
    b = GDSBuilder("audit_round_trip", str(tmp_path / "gds_audit"))
    b.add_line(
        "customers",
        [{"cust_id": "C-1"}, {"cust_id": "C-2"}],
        key_col="cust_id",
        source_id="test",
    )
    b.add_line(
        "orders",
        [
            {"order_id": "O-1", "cust_id": "C-1"},
            {"order_id": "O-2", "cust_id": "C-2"},
        ],
        key_col="order_id",
        source_id="test",
        role="event",
    )
    b.add_pattern(
        "account_pattern",
        pattern_type="event",
        entity_line="orders",
        relations=[
            RelationSpec(
                "customers", fk_col="cust_id", direction="in", required=True,
            ),
        ],
    )
    if audit_block is not None:
        b._label_audit_block = audit_block
    return b.build()


# ── schema parsing ───────────────────────────────────────────────────


def test_parse_accepts_well_formed_block(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml_with_block(
            "  label_column: laundering\n"
            "  label_positive_value: true\n"
            "  patterns:\n"
            "    - account_pattern\n"
        ),
    )
    cfg = parse_config(yaml_path)
    assert isinstance(cfg.label_audit, LabelAuditConfig)
    assert cfg.label_audit.label_column == "laundering"
    assert cfg.label_audit.label_positive_value is True
    assert cfg.label_audit.patterns == ["account_pattern"]


def test_parse_omitted_block_yields_none(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, _yaml_with_block(None))
    cfg = parse_config(yaml_path)
    assert cfg.label_audit is None


def test_parse_rejects_missing_label_column(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml_with_block(
            "  label_positive_value: true\n"
            "  patterns:\n"
            "    - account_pattern\n"
        ),
    )
    with pytest.raises(ValueError, match="label_column"):
        parse_config(yaml_path)


def test_parse_rejects_missing_label_positive_value(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml_with_block(
            "  label_column: laundering\n"
            "  patterns:\n"
            "    - account_pattern\n"
        ),
    )
    with pytest.raises(ValueError, match="label_positive_value"):
        parse_config(yaml_path)


def test_parse_rejects_missing_patterns(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml_with_block(
            "  label_column: laundering\n"
            "  label_positive_value: true\n"
        ),
    )
    with pytest.raises(ValueError, match="patterns"):
        parse_config(yaml_path)


def test_parse_rejects_empty_patterns(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml_with_block(
            "  label_column: laundering\n"
            "  label_positive_value: true\n"
            "  patterns: []\n"
        ),
    )
    with pytest.raises(ValueError, match="patterns"):
        parse_config(yaml_path)


def test_parse_rejects_unknown_pattern(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml_with_block(
            "  label_column: laundering\n"
            "  label_positive_value: true\n"
            "  patterns:\n"
            "    - no_such_pattern\n"
        ),
    )
    with pytest.raises(ValueError, match="unknown pattern"):
        parse_config(yaml_path)


def test_parse_rejects_non_mapping(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        _yaml_with_block("  - this is a list not a mapping\n"),
    )
    with pytest.raises(ValueError, match="mapping"):
        parse_config(yaml_path)


# ── round trip: builder → sphere.json → reader → sphere_info ─────────


def test_round_trip_block_present(tmp_path: Path) -> None:
    """A LabelAuditConfig set on the builder lands in sphere.json with
    format_version bumped to 3.1; reader reconstructs Sphere.label_audit
    and sphere_info reports label_aware_available=True."""
    block = LabelAuditConfig(
        label_column="laundering",
        label_positive_value=True,
        patterns=["account_pattern"],
    )
    out = _build_sphere_with_block(tmp_path, block)

    meta = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
    assert meta["format_version"] == "3.1"
    assert meta["label_audit"]["label_column"] == "laundering"
    assert meta["label_audit"]["label_positive_value"] is True
    assert meta["label_audit"]["patterns"] == ["account_pattern"]

    sphere = HyperSphere.open(out)
    assert sphere._sphere.label_audit is not None
    assert sphere._sphere.label_audit["label_column"] == "laundering"
    assert sphere._sphere.label_audit["patterns"] == ["account_pattern"]


def test_round_trip_block_absent(tmp_path: Path) -> None:
    """Without a block, the sphere stays at format 3.0 and Sphere.label_audit is None."""
    out = _build_sphere_with_block(tmp_path, None)

    meta = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
    assert meta["format_version"] == "3.0"
    assert "label_audit" not in meta

    sphere = HyperSphere.open(out)
    assert sphere._sphere.label_audit is None
