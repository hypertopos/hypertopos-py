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

from hypertopos._file_formats import normalized_suffix
from hypertopos.builder.builder import GDSBuilder, RelationSpec
from hypertopos.engine.edge_features import EDGE_DIM_KINDS

_VALID_PATTERN_TYPES: frozenset[str] = frozenset({"anchor", "event"})
_VALID_ROLES: frozenset[str] = frozenset({"anchor", "context", "event"})
_VALID_DIRECTIONS: frozenset[str] = frozenset({"in", "out", "self"})


_EDGE_DIM_DEFAULTS: dict[str, dict[str, Any]] = {
    "pair_edge_count":           {},
    "position_in_chain":         {"min_position": 5},
    "time_since_pair_last_edge": {"burst_seconds": 60.0,
                                  "dormant_seconds": "auto"},
    "pair_amount_zscore":        {"cv_threshold": 0.05, "min_count": 3},
    "find_motif_structuring":    {"time_window_hours": 1.0,
                                  "amt1_min": 10000.0, "amt2_max": 10000.0},
}


@dataclass(frozen=True)
class EdgeDimensionsConfig:
    """Parsed edge_dimensions: block from a pattern YAML stanza."""
    dims: dict[str, dict[str, Any]] = field(default_factory=dict)


def _validate_dim_params(name: str, params: dict[str, Any]) -> None:
    if name == "position_in_chain":
        mp = int(params.get("min_position", 5))
        if mp < 3:
            raise ValueError(
                f"min_position must be >= 3 (pos2+ flags too much "
                f"of the population to be selective); got {mp}",
            )
    elif name == "pair_amount_zscore":
        cv = float(params.get("cv_threshold", 0.05))
        if not (0.0 < cv <= 1.0):
            raise ValueError(f"cv_threshold must be in (0, 1]; got {cv}")
        mc = int(params.get("min_count", 3))
        if mc < 2:
            raise ValueError(f"min_count must be >= 2; got {mc}")
    elif name == "find_motif_structuring":
        if float(params.get("amt1_min", 0)) <= 0:
            raise ValueError(
                "amt1_min must be positive for find_motif_structuring",
            )
        if float(params.get("amt2_max", 0)) <= 0:
            raise ValueError(
                "amt2_max must be positive for find_motif_structuring",
            )
        if float(params.get("time_window_hours", 0)) <= 0:
            raise ValueError(
                "time_window_hours must be positive for find_motif_structuring",
            )
    elif name == "time_since_pair_last_edge":
        bs = float(params.get("burst_seconds", 0))
        if bs < 0:
            raise ValueError(f"burst_seconds must be >= 0; got {bs}")


def parse_edge_dimensions(
    raw: list[Any], *, pattern_type: str,
) -> EdgeDimensionsConfig:
    """Parse the YAML edge_dimensions: list and validate.

    Raises ValueError on any rule violation. Returns frozen dataclass.
    """
    if pattern_type != "event":
        raise ValueError(
            f"edge_dimensions are only supported on event patterns; "
            f"got pattern_type={pattern_type!r}",
        )
    if not isinstance(raw, list):
        raise ValueError(
            f"edge_dimensions must be a list of dim entries; "
            f"got {type(raw).__name__}",
        )

    out: dict[str, dict[str, Any]] = {}
    for entry in raw:
        if isinstance(entry, str):
            name, override = entry, {}
        elif isinstance(entry, dict) and len(entry) == 1:
            name, override = next(iter(entry.items()))
            override = override or {}
            if not isinstance(override, dict):
                raise ValueError(
                    f"edge dimension {name!r} parameters must be a dict; "
                    f"got {type(override).__name__}",
                )
        else:
            raise ValueError(
                f"edge_dimensions entries must be a string or single-key dict; "
                f"got {entry!r}",
            )

        if name not in EDGE_DIM_KINDS:
            raise ValueError(
                f"unknown edge dimension: {name!r}; "
                f"valid: {sorted(EDGE_DIM_KINDS)}",
            )
        if name in out:
            raise ValueError(f"edge dimension {name!r} declared twice")

        merged = {**_EDGE_DIM_DEFAULTS[name], **override}
        _validate_dim_params(name, merged)
        out[name] = merged

    return EdgeDimensionsConfig(dims=out)


@dataclass(frozen=True)
class EdgeDimAggregationsConfig:
    from_event_pattern: str
    dims: tuple[str, ...]
    aggregates_per_dim: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Direct constructor callers (tests, downstream pipelines) frequently
        # pass `dims=` only and omit `aggregates_per_dim`. Materialise the
        # all-five canonical default so the engine-level dispatch always
        # receives a fully populated per-dim mapping — same back-compat
        # contract the YAML list form provides.
        if not self.aggregates_per_dim:
            from hypertopos.engine.edge_features import AGGREGATE_NAMES
            object.__setattr__(
                self, "aggregates_per_dim",
                {d: tuple(AGGREGATE_NAMES) for d in self.dims},
            )


def parse_edge_dim_aggregations(
    raw: Any, *, pattern_type: str,
) -> EdgeDimAggregationsConfig:
    from hypertopos.engine.edge_features import AGGREGATE_NAMES

    if pattern_type != "anchor":
        raise ValueError(
            f"edge_dim_aggregations are only supported on anchor patterns; "
            f"got pattern_type={pattern_type!r}",
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"edge_dim_aggregations must be a YAML mapping; "
            f"got {type(raw).__name__}",
        )
    src = raw.get("from")
    if not src or not isinstance(src, str):
        raise ValueError(
            "edge_dim_aggregations must specify 'from: <event_pattern_id>'",
        )
    raw_dims = raw.get("dims")
    if raw_dims is None:
        # The previous "dims = None means aggregate every applicable
        # source edge dim" shorthand was a latent build-vs-runtime
        # inconsistency: the builder resolved `dims=None` to the source
        # event pattern's `avail` list and appended N columns to the
        # polygon delta on disk, but the runtime Pattern model retained
        # `dims=None`, so `Pattern.dim_labels` and `Pattern.delta_dim()`
        # had no record of the appended columns. Forcing explicit
        # declaration keeps the build-time and runtime-time views aligned.
        raise ValueError(
            "edge_dim_aggregations.dims must be a non-empty list or "
            "mapping of source edge dim names",
        )
    aggregates_per_dim: dict[str, tuple[str, ...]]
    if isinstance(raw_dims, list):
        # Form A — list sugar: every dim emits all five canonical aggregates.
        if not raw_dims:
            raise ValueError(
                "edge_dim_aggregations.dims must be a non-empty list",
            )
        for d in raw_dims:
            if not isinstance(d, str):
                raise ValueError(
                    f"edge_dim_aggregations.dims entries must be strings; "
                    f"got {type(d).__name__}",
                )
            if d not in EDGE_DIM_KINDS:
                raise ValueError(
                    f"unknown edge dimension: {d!r}; "
                    f"valid: {sorted(EDGE_DIM_KINDS)}",
                )
        dims = tuple(raw_dims)
        aggregates_per_dim = {d: tuple(AGGREGATE_NAMES) for d in dims}
    elif isinstance(raw_dims, dict):
        # Form B — mapping: each source dim declares its own subset.
        if not raw_dims:
            raise ValueError(
                "edge_dim_aggregations.dims mapping must be non-empty",
            )
        aggregates_per_dim = {}
        for d, agg_list in raw_dims.items():
            if not isinstance(d, str):
                raise ValueError(
                    f"edge_dim_aggregations.dims keys must be strings; "
                    f"got {type(d).__name__}",
                )
            if d not in EDGE_DIM_KINDS:
                raise ValueError(
                    f"unknown edge dimension: {d!r}; "
                    f"valid: {sorted(EDGE_DIM_KINDS)}",
                )
            if not isinstance(agg_list, list):
                raise ValueError(
                    f"edge_dim_aggregations.dims[{d!r}] must be a list of "
                    f"aggregate names; got {type(agg_list).__name__}",
                )
            if not agg_list:
                raise ValueError(
                    f"edge_dim_aggregations.dims[{d!r}] must be a non-empty "
                    f"list of aggregate names",
                )
            for agg in agg_list:
                if agg not in AGGREGATE_NAMES:
                    raise ValueError(
                        f"unknown aggregate {agg!r} for dim {d!r}; "
                        f"valid: {list(AGGREGATE_NAMES)}",
                    )
            # Canonical-order materialisation: filter AGGREGATE_NAMES by user
            # selection so polygon-dim layout is insensitive to YAML order.
            user_set = set(agg_list)
            aggregates_per_dim[d] = tuple(
                a for a in AGGREGATE_NAMES if a in user_set
            )
        dims = tuple(raw_dims.keys())
    else:
        raise ValueError(
            f"edge_dim_aggregations.dims must be a list or mapping; "
            f"got {type(raw_dims).__name__}",
        )
    return EdgeDimAggregationsConfig(
        from_event_pattern=src,
        dims=dims,
        aggregates_per_dim=aggregates_per_dim,
    )


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
    edge_dimensions: "EdgeDimensionsConfig | None" = None
    edge_dim_aggregations: "EdgeDimAggregationsConfig | None" = None


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
        edge_dims_raw = spec.get("edge_dimensions")
        edge_dimensions = (
            parse_edge_dimensions(edge_dims_raw, pattern_type=pattern_type)
            if edge_dims_raw is not None
            else None
        )
        eda_raw = spec.get("edge_dim_aggregations")
        edge_dim_aggregations = (
            parse_edge_dim_aggregations(eda_raw, pattern_type=pattern_type)
            if eda_raw is not None
            else None
        )
        result[pattern_id] = PatternMapping(
            pattern_type=pattern_type,
            entity_line=str(entity_line),
            relations=relations,
            anomaly_percentile=float(spec.get("anomaly_percentile", 95.0)),
            tracked_properties=list(spec.get("tracked_properties") or []),
            edge_dimensions=edge_dimensions,
            edge_dim_aggregations=edge_dim_aggregations,
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

    suffix = normalized_suffix(p)
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
            edge_dimensions=pattern_spec.edge_dimensions,
        )

    return builder.build()
