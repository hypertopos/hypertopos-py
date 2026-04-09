# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
import json
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypertopos.builder.mapping import (
    LineMapping,
    MappingSpec,
    PatternMapping,
    RelationMapping,
    _load_source,
    build_from_mapping,
    load_mapping,
)


def test_relation_mapping_defaults():
    r = RelationMapping(line_id="customers", fk_col="customer_id")
    assert r.direction == "in"
    assert r.required is True
    assert r.display_name is None


def test_line_mapping_defaults():
    lm = LineMapping(source="customers.csv", key_col="customer_id")
    assert lm.role == "anchor"
    assert lm.partition_col is None
    assert lm.entity_type is None


def test_pattern_mapping_defaults():
    p = PatternMapping(pattern_type="event", entity_line="orders", relations=[])
    assert p.anomaly_percentile == 95.0
    assert p.tracked_properties == []


def test_mapping_spec_construction():
    spec = MappingSpec(
        sphere_id="test",
        output_path="./gds_test",
        lines={"customers": LineMapping("customers.csv", "customer_id")},
        patterns={},
    )
    assert spec.sphere_id == "test"
    assert "customers" in spec.lines


# ---------------------------------------------------------------------------
# Stage B: load_mapping() YAML parser
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "mapping.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_load_mapping_minimal(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: my_sphere
        output_path: ./gds_out

        lines:
          customers:
            source: customers.csv
            key_col: customer_id

        patterns:
          cust_pattern:
            type: anchor
            entity_line: customers
            relations:
              - line_id: customers
                direction: self
    """,
    )
    spec = load_mapping(yaml_file)
    assert spec.sphere_id == "my_sphere"
    assert "customers" in spec.lines
    assert spec.lines["customers"].key_col == "customer_id"
    assert "cust_pattern" in spec.patterns
    assert spec.patterns["cust_pattern"].relations[0].direction == "self"


def test_load_mapping_missing_sphere_id(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        output_path: ./gds_out
        lines: {}
        patterns: {}
    """,
    )
    with pytest.raises(ValueError, match="sphere_id"):
        load_mapping(yaml_file)


def test_load_mapping_empty_file(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    with pytest.raises(ValueError, match="empty file"):
        load_mapping(yaml_file)


def test_load_mapping_missing_key_col(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: s
        output_path: ./gds
        lines:
          customers:
            source: c.csv
        patterns: {}
    """,
    )
    with pytest.raises(ValueError, match="key_col"):
        load_mapping(yaml_file)


def test_load_mapping_unknown_pattern_entity_line(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: s
        output_path: ./gds
        lines:
          customers:
            source: c.csv
            key_col: id
        patterns:
          p:
            type: event
            entity_line: nonexistent_line
            relations: []
    """,
    )
    with pytest.raises(ValueError, match="nonexistent_line"):
        load_mapping(yaml_file)


def test_load_mapping_relation_defaults(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: s
        output_path: ./gds
        lines:
          a:
            source: a.csv
            key_col: id
          b:
            source: b.csv
            key_col: id
        patterns:
          p:
            type: event
            entity_line: a
            relations:
              - line_id: b
                fk_col: b_id
    """,
    )
    spec = load_mapping(yaml_file)
    rel = spec.patterns["p"].relations[0]
    assert rel.direction == "in"
    assert rel.required is True
    assert rel.display_name is None


def test_load_mapping_full_relation(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: s
        output_path: ./gds
        lines:
          orders:
            source: orders.csv
            key_col: order_id
            role: event
          customers:
            source: customers.csv
            key_col: customer_id
        patterns:
          order_pattern:
            type: event
            entity_line: orders
            anomaly_percentile: 90.0
            tracked_properties:
              - amount
            relations:
              - line_id: customers
                fk_col: customer_id
                direction: in
                required: false
                display_name: Customer
    """,
    )
    spec = load_mapping(yaml_file)
    pat = spec.patterns["order_pattern"]
    assert pat.anomaly_percentile == 90.0
    assert pat.tracked_properties == ["amount"]
    rel = pat.relations[0]
    assert rel.required is False
    assert rel.display_name == "Customer"
    assert spec.lines["orders"].role == "event"


def test_load_mapping_invalid_role(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: s
        output_path: ./gds
        lines:
          items:
            source: items.csv
            key_col: id
            role: invalid_role
        patterns: {}
    """,
    )
    with pytest.raises(ValueError, match="invalid role"):
        load_mapping(yaml_file)


def test_load_mapping_invalid_pattern_type(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: s
        output_path: ./gds
        lines:
          items:
            source: items.csv
            key_col: id
        patterns:
          p:
            type: invalid_type
            entity_line: items
            relations: []
    """,
    )
    with pytest.raises(ValueError, match="invalid type"):
        load_mapping(yaml_file)


def test_load_mapping_invalid_direction(tmp_path):
    yaml_file = _write_yaml(
        tmp_path,
        """\
        sphere_id: s
        output_path: ./gds
        lines:
          a:
            source: a.csv
            key_col: id
          b:
            source: b.csv
            key_col: id
        patterns:
          p:
            type: event
            entity_line: a
            relations:
              - line_id: b
                fk_col: b_id
                direction: sideways
    """,
    )
    with pytest.raises(ValueError, match="invalid direction"):
        load_mapping(yaml_file)


# ---------------------------------------------------------------------------
# Stage C: _load_source() data loader
# ---------------------------------------------------------------------------


def test_load_source_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,name\nC-1,Alpha\nC-2,Beta\n")
    table = _load_source(str(csv_file), base_dir=tmp_path)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert "id" in table.schema.names


def test_load_source_parquet(tmp_path):
    tbl = pa.table({"id": ["C-1", "C-2"], "val": [1, 2]})
    pq_file = tmp_path / "data.parquet"
    pq.write_table(tbl, str(pq_file))
    result = _load_source(str(pq_file), base_dir=tmp_path)
    assert result.num_rows == 2


def test_load_source_resolves_relative_path(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x\n1\n2\n")
    result = _load_source("data.csv", base_dir=tmp_path)
    assert result.num_rows == 2


def test_load_source_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported source format"):
        _load_source("data.xls", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Stage D: build_from_mapping()
# ---------------------------------------------------------------------------


def test_build_from_mapping_e2e(tmp_path):
    """End-to-end: write YAML + CSV, build sphere, verify sphere.json exists."""
    (tmp_path / "customers.csv").write_text(
        "customer_id,name\nC-1,Alpha\nC-2,Beta\nC-3,Gamma\nC-4,Delta\nC-5,Eps\n"
    )
    (tmp_path / "orders.csv").write_text(
        "order_id,customer_id\nO-1,C-1\nO-2,C-1\nO-3,C-2\nO-4,C-3\nO-5,\nO-6,\n"
    )

    gds_path = str(tmp_path / "gds_test")
    yaml_content = textwrap.dedent(f"""\
        sphere_id: test_sphere
        output_path: {gds_path}

        lines:
          customers:
            source: customers.csv
            key_col: customer_id
            role: anchor

          orders:
            source: orders.csv
            key_col: order_id
            role: event

        patterns:
          order_pattern:
            type: event
            entity_line: orders
            relations:
              - line_id: customers
                fk_col: customer_id
                direction: in
                required: false
            anomaly_percentile: 60.0
    """)
    yaml_file = tmp_path / "mapping.yaml"
    yaml_file.write_text(yaml_content)

    spec = load_mapping(yaml_file)
    out = build_from_mapping(spec, base_dir=tmp_path)

    sphere_json = Path(out) / "_gds_meta" / "sphere.json"
    assert sphere_json.exists()
    data = json.loads(sphere_json.read_text())
    assert data["sphere_id"] == "test_sphere"
    assert "customers" in data["lines"]
    assert "orders" in data["lines"]
    assert "order_pattern" in data["patterns"]


def test_build_from_mapping_output_path_override(tmp_path):
    """build_from_mapping respects explicit output_path override."""
    (tmp_path / "items.csv").write_text("id,name\nI-1,Widget\nI-2,Gadget\n")
    spec = MappingSpec(
        sphere_id="override_test",
        output_path=str(tmp_path / "default_output"),
        lines={"items": LineMapping(source="items.csv", key_col="id")},
        patterns={},
    )
    out = build_from_mapping(spec, base_dir=tmp_path, output_path=str(tmp_path / "custom_out"))
    assert Path(out).name == "custom_out"
    assert (Path(out) / "_gds_meta" / "sphere.json").exists()
