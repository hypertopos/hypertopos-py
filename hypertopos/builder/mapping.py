# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import yaml

from hypertopos.builder.builder import GDSBuilder, RelationSpec

_VALID_PATTERN_TYPES: frozenset[str] = frozenset({"anchor", "event"})
_VALID_ROLES: frozenset[str] = frozenset({"anchor", "context", "event"})
_VALID_DIRECTIONS: frozenset[str] = frozenset({"in", "out", "self"})


@dataclass
class RelationMapping:
    line_id: str
    fk_col: str | None = None
    direction: str = "in"
    required: bool = True
    display_name: str | None = None
    edge_max: int | None = None


@dataclass
class LineMapping:
    source: str
    key_col: str
    role: str = "anchor"
    partition_col: str | None = None
    entity_type: str | None = None
    fts_columns: list[str] | str | None = None


@dataclass
class PatternMapping:
    pattern_type: str
    entity_line: str
    relations: list[RelationMapping] = field(default_factory=list)
    anomaly_percentile: float = 95.0
    tracked_properties: list[str] = field(default_factory=list)


@dataclass
class MappingSpec:
    sphere_id: str
    output_path: str
    lines: dict[str, LineMapping]
    patterns: dict[str, PatternMapping]


def load_mapping(path: str | Path) -> MappingSpec:
    """Load and validate a gds_mapping.yaml file. Raises ValueError on schema errors."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        got = type(raw).__name__ if raw is not None else "empty file"
        raise ValueError(f"mapping.yaml must be a YAML mapping, got {got}")

    sphere_id = raw.get("sphere_id")
    if not sphere_id:
        raise ValueError("mapping.yaml must specify 'sphere_id'")

    output_path = raw.get("output_path")
    if not output_path:
        raise ValueError("mapping.yaml must specify 'output_path'")

    lines = _parse_lines(raw.get("lines") or {})
    patterns = _parse_patterns(raw.get("patterns") or {}, lines)

    return MappingSpec(
        sphere_id=str(sphere_id),
        output_path=str(output_path),
        lines=lines,
        patterns=patterns,
    )


def _parse_lines(raw_lines: dict[str, Any]) -> dict[str, LineMapping]:
    result: dict[str, LineMapping] = {}
    for line_id, spec in raw_lines.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Line '{line_id}' spec must be a mapping")
        if "key_col" not in spec:
            raise ValueError(f"Line '{line_id}' must specify 'key_col'")
        if "source" not in spec:
            raise ValueError(f"Line '{line_id}' must specify 'source'")
        role = str(spec.get("role", "anchor"))
        if role not in _VALID_ROLES:
            raise ValueError(
                f"Line '{line_id}' has invalid role '{role}'. Valid values: {sorted(_VALID_ROLES)}"
            )
        result[line_id] = LineMapping(
            source=str(spec["source"]),
            key_col=str(spec["key_col"]),
            role=role,
            partition_col=spec.get("partition_col"),
            entity_type=spec.get("entity_type"),
            fts_columns=spec.get("fts_columns"),
        )
    return result


def _parse_patterns(
    raw_patterns: dict[str, Any],
    lines: dict[str, LineMapping],
) -> dict[str, PatternMapping]:
    result: dict[str, PatternMapping] = {}
    for pattern_id, spec in raw_patterns.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Pattern '{pattern_id}' spec must be a mapping")
        entity_line = spec.get("entity_line")
        if not entity_line:
            raise ValueError(f"Pattern '{pattern_id}' must specify 'entity_line'")
        if entity_line not in lines:
            raise ValueError(
                f"Pattern '{pattern_id}' references unknown entity_line '{entity_line}'. "
                f"Available lines: {list(lines)}"
            )
        relations = [
            _parse_relation(pattern_id, r)
            for r in (spec.get("relations") or [])
        ]
        pattern_type = str(spec.get("type", "event"))
        if pattern_type not in _VALID_PATTERN_TYPES:
            raise ValueError(
                f"Pattern '{pattern_id}' has invalid type '{pattern_type}'. "
                f"Valid values: {sorted(_VALID_PATTERN_TYPES)}"
            )
        result[pattern_id] = PatternMapping(
            pattern_type=pattern_type,
            entity_line=str(entity_line),
            relations=relations,
            anomaly_percentile=float(spec.get("anomaly_percentile", 95.0)),
            tracked_properties=list(spec.get("tracked_properties") or []),
        )
    return result


def _parse_relation(pattern_id: str, spec: Any) -> RelationMapping:
    if not isinstance(spec, dict):
        raise ValueError(f"Pattern '{pattern_id}' relation must be a mapping")
    if "line_id" not in spec:
        raise ValueError(f"Pattern '{pattern_id}' relation must specify 'line_id'")
    direction = str(spec.get("direction", "in"))
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Pattern '{pattern_id}' relation to '{spec['line_id']}' has invalid direction "
            f"'{direction}'. Valid values: {sorted(_VALID_DIRECTIONS)}"
        )
    raw_em = spec.get("edge_max")
    return RelationMapping(
        line_id=str(spec["line_id"]),
        fk_col=spec.get("fk_col"),
        direction=direction,
        required=bool(spec.get("required", True)),
        display_name=spec.get("display_name"),
        edge_max=int(raw_em) if raw_em is not None else None,
    )


def _load_source(source: str, base_dir: Path) -> pa.Table:
    """Load a data source file and return a PyArrow Table.

    Relative paths are resolved against base_dir (the YAML file's directory).
    Supported: .csv, .csv.gz, .parquet, .pq
    """
    p = Path(source)
    if not p.is_absolute():
        p = base_dir / p

    suffix = "".join(p.suffixes).lower()
    if suffix in (".csv", ".csv.gz"):
        return pa_csv.read_csv(str(p))
    if suffix in (".parquet", ".pq"):
        return pq.ParquetFile(str(p)).read()
    raise ValueError(
        f"Unsupported source format '{suffix}' for file '{source}'. "
        "Supported: .csv, .csv.gz, .parquet, .pq"
    )


def build_from_mapping(
    spec: MappingSpec,
    base_dir: str | Path | None = None,
    output_path: str | None = None,
) -> str:
    """Build a GDS sphere from a MappingSpec. Returns the output path string.

    Args:
        spec: Parsed mapping specification.
        base_dir: Directory to resolve relative source paths against.
                  Defaults to current working directory.
        output_path: Override the output_path from spec if provided.
    """
    resolved_output = output_path or spec.output_path
    resolved_base = Path(base_dir) if base_dir else Path.cwd()

    builder = GDSBuilder(spec.sphere_id, resolved_output)

    for line_id, line_spec in spec.lines.items():
        table = _load_source(line_spec.source, base_dir=resolved_base)
        builder.add_line(
            line_id,
            table,
            key_col=line_spec.key_col,
            source_id=line_id,
            role=line_spec.role,
            partition_col=line_spec.partition_col,
            entity_type=line_spec.entity_type,
            fts_columns=line_spec.fts_columns,
        )

    for pattern_id, pattern_spec in spec.patterns.items():
        relations = [
            RelationSpec(
                line_id=r.line_id,
                fk_col=r.fk_col,
                direction=r.direction,
                required=r.required,
                display_name=r.display_name,
                edge_max=r.edge_max,
            )
            for r in pattern_spec.relations
        ]
        builder.add_pattern(
            pattern_id,
            pattern_type=pattern_spec.pattern_type,
            entity_line=pattern_spec.entity_line,
            relations=relations,
            anomaly_percentile=pattern_spec.anomaly_percentile,
            tracked_properties=pattern_spec.tracked_properties,
        )

    return builder.build()
