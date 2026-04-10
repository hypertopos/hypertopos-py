# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""YAML config schema for ``hypertopos build``.

Parses a ``sphere.yaml`` file into typed dataclasses that the build bridge
can translate into GDSBuilder calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── valid enum values ────────────────────────────────────────────────

_VALID_ROLES: frozenset[str] = frozenset({"anchor", "event", "context"})
_VALID_PATTERN_TYPES: frozenset[str] = frozenset({"anchor", "event"})
_VALID_DIRECTIONS: frozenset[str] = frozenset({"in", "out", "self"})
_VALID_JOIN_TYPES: frozenset[str] = frozenset({"left", "inner"})
_VALID_CAST_TYPES: frozenset[str] = frozenset({
    "string", "int32", "int64", "float32", "float64", "bool",
})
_VALID_METRICS: frozenset[str] = frozenset({
    "count", "count_distinct", "sum", "max", "std", "mean", "avg",
    "iet_mean", "iet_std", "iet_min",
})

# ── dataclasses ──────────────────────────────────────────────────────


@dataclass
class JoinSpec:
    file: str
    on: str
    type: str = "left"
    columns: list[str] | None = None


@dataclass
class TransformSpec:
    """Per-column transform (cast type and/or fill_null)."""
    type: str | None = None
    fill_null: Any = None


@dataclass
class SourceConfig:
    path: str | None = None
    format: str | None = None
    delimiter: str | None = None
    encoding: str | None = None
    join: list[JoinSpec] = field(default_factory=list)
    transform: dict[str, TransformSpec] = field(default_factory=dict)
    script: str | None = None


@dataclass
class LineConfig:
    source: str
    key: str
    role: str = "anchor"
    columns: dict[str, str] | None = None  # rename map: yaml_name -> source_name
    fts: bool | list[str] | None = None
    partition_col: str | None = None
    description: str | None = None


@dataclass
class RelationConfig:
    line: str
    direction: str = "in"
    key_on_entity: str | None = None
    required: bool = True
    display_name: str | None = None
    edge_max: int | None = None


@dataclass
class EventDimConfig:
    column: str
    display_name: str | None = None


@dataclass
class FeatureSpec:
    """Parsed feature like ``tx_count: count`` or ``burst_daily: count:window=1d:agg=max``."""
    dimension_name: str
    metric: str
    metric_col: str | None = None
    time_col: str | None = None
    time_window: str | None = None
    window_aggregation: str = "max"


@dataclass
class DerivedDimGroup:
    from_pattern: str | None = None
    anchor_fk: str | None = None
    features: list[FeatureSpec] = field(default_factory=list)


@dataclass
class PatternConfig:
    type: str
    entity_line: str
    relations: list[RelationConfig] | str | None = None  # list, "auto", or None
    event_dimensions: list[EventDimConfig] | None = None
    derived_dimensions: list[DerivedDimGroup] | None = None
    precomputed_dimensions: list[PrecomputedDimConfig] | None = None
    graph_features: GraphFeaturesConfig | None = None
    edge_table: EdgeTableYamlConfig | None = None
    anomaly_percentile: float = 95.0
    dimension_weights: str | list[float] | None = None
    gmm_n_components: int | None = None
    group_by_property: str | None = None
    tracked_properties: list[str] | None = None
    use_mahalanobis: bool = False
    description: str | None = None


@dataclass
class PrecomputedDimConfig:
    column: str
    edge_max: int | str = "auto"
    percentile: float = 99.0
    display_name: str | None = None


@dataclass
class EdgeTableYamlConfig:
    """Optional explicit edge table config in YAML."""

    from_col: str
    to_col: str
    timestamp_col: str | None = None
    amount_col: str | None = None


@dataclass
class GraphFeaturesConfig:
    event_line: str
    from_col: str
    to_col: str
    features: list[str] | None = None


@dataclass
class AliasConfig:
    """Alias spec parsed from YAML — either dimension+threshold or normal+bias."""
    base_pattern: str
    cutting_plane_dimension: int | str | None = None
    cutting_plane_threshold: float | None = None
    cutting_plane_normal: list[float] | None = None
    cutting_plane_bias: float | None = None
    description: str | None = None


@dataclass
class ChainLineConfig:
    event_line: str
    from_col: str
    to_col: str
    features: list[str] | None = None
    seed_percentile_fan_out: float = 95.0
    seed_percentile_cross_bank: float = 90.0
    seed_multi_currency: int = 2
    seed_pass_through: bool = True
    time_window_hours: int = 168
    max_hops: int = 15
    min_hops: int = 2
    max_chains: int = 300000
    bidirectional: bool = True
    anomaly_percentile: float = 95.0
    description: str | None = None


@dataclass
class CompositeLineConfig:
    event_line: str
    key_cols: list[str]
    separator: str = "|"
    derived_dimensions: list[DerivedDimGroup] | None = None
    anomaly_percentile: float = 95.0
    dimension_weights: str | list[float] | None = None
    description: str | None = None


@dataclass
class TemporalConfig:
    pattern: str
    event_line: str
    timestamp_col: str
    window: str


@dataclass
class SphereConfig:
    sphere_id: str
    version: str = "0.1.0"
    name: str | None = None
    description: str | None = None
    sources: dict[str, SourceConfig] = field(default_factory=dict)
    lines: dict[str, LineConfig] = field(default_factory=dict)
    patterns: dict[str, PatternConfig] = field(default_factory=dict)
    aliases: dict[str, AliasConfig] = field(default_factory=dict)
    composite_lines: dict[str, CompositeLineConfig] = field(default_factory=dict)
    chain_lines: dict[str, ChainLineConfig] = field(default_factory=dict)
    temporal: list[TemporalConfig] = field(default_factory=list)


# ── parser ───────────────────────────────────────────────────────────


def parse_config(yaml_path: str | Path) -> SphereConfig:
    """Load and validate a ``sphere.yaml``. Returns a typed ``SphereConfig``."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        got = type(raw).__name__ if raw is not None else "empty file"
        raise ValueError(f"sphere.yaml must be a YAML mapping, got {got}")

    # Version check
    yaml_version = str(raw.get("version", "0.1.0"))
    if not yaml_version.startswith("0.1"):
        raise ValueError(
            f"Unsupported sphere.yaml version '{yaml_version}'. "
            f"This builder supports version 0.1.x."
        )

    sphere_id = raw.get("sphere_id")
    if not sphere_id:
        raise ValueError("sphere.yaml must specify 'sphere_id'")

    sources = _parse_sources(raw.get("sources") or {})
    lines = _parse_lines(raw.get("lines") or {}, sources)
    composite_lines = _parse_composite_lines(raw.get("composite_lines") or {}, lines)
    chain_lines = _parse_chain_lines(raw.get("chain_lines") or {}, lines)
    all_extra_ids = set(composite_lines.keys()) | set(chain_lines.keys())
    patterns = _parse_patterns(raw.get("patterns") or {}, lines, all_extra_ids)
    # Auto-generated pattern IDs for composite/chain lines are valid
    # temporal targets even though they aren't in the YAML patterns block
    all_patterns = dict(patterns)
    for extra_id in all_extra_ids:
        auto_pid = f"{extra_id}_pattern"
        if auto_pid not in all_patterns:
            all_patterns[auto_pid] = PatternConfig(
                type="anchor", entity_line=extra_id,
            )
    temporal = _parse_temporal(raw.get("temporal") or [], all_patterns)
    aliases = _parse_aliases(raw.get("aliases") or {}, all_patterns)

    return SphereConfig(
        sphere_id=str(sphere_id),
        version=yaml_version,
        name=raw.get("name"),
        description=raw.get("description"),
        sources=sources,
        lines=lines,
        patterns=patterns,
        aliases=aliases,
        composite_lines=composite_lines,
        chain_lines=chain_lines,
        temporal=temporal,
    )


# ── sources ──────────────────────────────────────────────────────────


def _parse_sources(raw: dict[str, Any]) -> dict[str, SourceConfig]:
    result: dict[str, SourceConfig] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Source '{name}' must be a mapping")
        result[name] = _parse_one_source(name, spec)
    return result


def _parse_one_source(name: str, spec: dict[str, Any]) -> SourceConfig:
    path = spec.get("path")
    script = spec.get("script")

    if not path and not script:
        raise ValueError(
            f"Source '{name}' must specify either 'path' or 'script'"
        )
    if path and script:
        raise ValueError(
            f"Source '{name}' cannot specify both 'path' and 'script'"
        )

    joins: list[JoinSpec] = []
    for j in spec.get("join") or []:
        if not isinstance(j, dict):
            raise ValueError(f"Source '{name}' join entry must be a mapping")
        if "file" not in j:
            raise ValueError(f"Source '{name}' join entry must specify 'file'")
        # YAML 1.1 parses bare `on:` as boolean True — handle both
        on_val = j.get("on") or j.get(True)
        if not on_val:
            raise ValueError(f"Source '{name}' join entry must specify 'on'")
        jtype = str(j.get("type", "left"))
        if jtype not in _VALID_JOIN_TYPES:
            raise ValueError(
                f"Source '{name}' join type '{jtype}' invalid. "
                f"Valid: {sorted(_VALID_JOIN_TYPES)}"
            )
        joins.append(JoinSpec(
            file=str(j["file"]),
            on=str(on_val),
            type=jtype,
            columns=j.get("columns"),
        ))

    transforms: dict[str, TransformSpec] = {}
    for col_name, tspec in (spec.get("transform") or {}).items():
        if not isinstance(tspec, dict):
            raise ValueError(
                f"Source '{name}' transform for column '{col_name}' "
                "must be a mapping"
            )
        cast_type = tspec.get("type")
        if cast_type and str(cast_type) not in _VALID_CAST_TYPES:
            raise ValueError(
                f"Source '{name}' transform type '{cast_type}' for column "
                f"'{col_name}' invalid. Valid: {sorted(_VALID_CAST_TYPES)}"
            )
        transforms[col_name] = TransformSpec(
            type=str(cast_type) if cast_type else None,
            fill_null=tspec.get("fill_null"),
        )

    return SourceConfig(
        path=str(path) if path else None,
        format=spec.get("format"),
        delimiter=spec.get("delimiter"),
        encoding=spec.get("encoding"),
        join=joins,
        transform=transforms,
        script=str(script) if script else None,
    )


# ── lines ────────────────────────────────────────────────────────────


def _parse_lines(
    raw: dict[str, Any],
    sources: dict[str, SourceConfig],
) -> dict[str, LineConfig]:
    result: dict[str, LineConfig] = {}
    for line_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Line '{line_id}' must be a mapping")
        source_name = spec.get("source")
        if not source_name:
            raise ValueError(f"Line '{line_id}' must specify 'source'")
        if str(source_name) not in sources:
            raise ValueError(
                f"Line '{line_id}' references unknown source '{source_name}'. "
                f"Available sources: {list(sources)}"
            )
        key = spec.get("key")
        if not key:
            raise ValueError(f"Line '{line_id}' must specify 'key'")
        role = str(spec.get("role", "anchor"))
        if role not in _VALID_ROLES:
            raise ValueError(
                f"Line '{line_id}' has invalid role '{role}'. "
                f"Valid: {sorted(_VALID_ROLES)}"
            )

        # fts: true | false | [col1, col2]
        fts_raw = spec.get("fts")
        fts: bool | list[str] | None = None
        if isinstance(fts_raw, bool):
            fts = fts_raw
        elif isinstance(fts_raw, list):
            fts = [str(c) for c in fts_raw]
        elif fts_raw is not None:
            fts = bool(fts_raw)

        result[line_id] = LineConfig(
            source=str(source_name),
            key=str(key),
            role=role,
            columns=spec.get("columns"),
            fts=fts,
            partition_col=spec.get("partition_col"),
            description=spec.get("description"),
        )
    return result


# ── patterns ─────────────────────────────────────────────────────────


def _parse_patterns(
    raw: dict[str, Any],
    lines: dict[str, LineConfig],
    composite_line_ids: set[str] | None = None,
) -> dict[str, PatternConfig]:
    result: dict[str, PatternConfig] = {}
    _cl_ids = composite_line_ids or set()
    for pid, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Pattern '{pid}' must be a mapping")
        result[pid] = _parse_one_pattern(pid, spec, lines, _cl_ids)
    return result


def _parse_one_pattern(
    pid: str,
    spec: dict[str, Any],
    lines: dict[str, LineConfig],
    _composite_line_ids: set[str] | None = None,
) -> PatternConfig:
    entity_line = spec.get("entity_line")
    if not entity_line:
        raise ValueError(f"Pattern '{pid}' must specify 'entity_line'")
    if str(entity_line) not in lines and str(entity_line) not in _composite_line_ids:
        raise ValueError(
            f"Pattern '{pid}' references unknown entity_line '{entity_line}'. "
            f"Available lines: {list(lines) + list(_composite_line_ids)}"
        )

    ptype = str(spec.get("type", "event"))
    if ptype not in _VALID_PATTERN_TYPES:
        raise ValueError(
            f"Pattern '{pid}' has invalid type '{ptype}'. "
            f"Valid: {sorted(_VALID_PATTERN_TYPES)}"
        )

    # relations: list of dicts | "auto" | None
    raw_rels = spec.get("relations")
    relations: list[RelationConfig] | str | None = None
    if raw_rels == "auto":
        relations = "auto"
    elif isinstance(raw_rels, list):
        all_line_ids = set(lines) | (_composite_line_ids or set())
        relations = [_parse_relation(pid, r, all_line_ids) for r in raw_rels]
    elif raw_rels is not None:
        raise ValueError(
            f"Pattern '{pid}' relations must be a list or 'auto', "
            f"got {type(raw_rels).__name__}"
        )

    # event_dimensions
    event_dims: list[EventDimConfig] | None = None
    raw_ed = spec.get("event_dimensions")
    if raw_ed:
        event_dims = []
        for ed in raw_ed:
            if not isinstance(ed, dict) or "column" not in ed:
                raise ValueError(
                    f"Pattern '{pid}' event_dimensions entries must have 'column'"
                )
            event_dims.append(EventDimConfig(
                column=str(ed["column"]),
                display_name=ed.get("display_name"),
            ))

    # derived_dimensions
    derived_dims: list[DerivedDimGroup] | None = None
    raw_dd = spec.get("derived_dimensions")
    if raw_dd:
        derived_dims = [_parse_derived_group(pid, g) for g in raw_dd]

    # precomputed_dimensions
    precomp_dims: list[PrecomputedDimConfig] | None = None
    raw_pc = spec.get("precomputed_dimensions")
    if raw_pc:
        precomp_dims = []
        for pc in raw_pc:
            if not isinstance(pc, dict) or "column" not in pc:
                raise ValueError(
                    f"Pattern '{pid}' precomputed_dimensions entries must have 'column'"
                )
            precomp_dims.append(PrecomputedDimConfig(
                column=str(pc["column"]),
                edge_max=pc.get("edge_max", "auto"),
                percentile=float(pc.get("percentile", 99.0)),
                display_name=pc.get("display_name"),
            ))

    # graph_features
    graph_feat: GraphFeaturesConfig | None = None
    raw_gf = spec.get("graph_features")
    if raw_gf:
        if not isinstance(raw_gf, dict):
            raise ValueError(f"Pattern '{pid}' graph_features must be a mapping")
        gf_event_line = str(raw_gf.get("event_line", ""))
        if gf_event_line and gf_event_line not in lines:
            raise ValueError(
                f"Pattern '{pid}' graph_features.event_line '{gf_event_line}' "
                f"not in lines. Available: {list(lines)}"
            )
        graph_feat = GraphFeaturesConfig(
            event_line=gf_event_line,
            from_col=str(raw_gf.get("from_col", "")),
            to_col=str(raw_gf.get("to_col", "")),
            features=raw_gf.get("features"),
        )

    # edge_table (optional explicit config)
    edge_tbl_cfg: EdgeTableYamlConfig | None = None
    raw_et = spec.get("edge_table")
    if raw_et and isinstance(raw_et, dict):
        edge_tbl_cfg = EdgeTableYamlConfig(
            from_col=str(raw_et.get("from_col", "")),
            to_col=str(raw_et.get("to_col", "")),
            timestamp_col=raw_et.get("timestamp_col"),
            amount_col=raw_et.get("amount_col"),
        )

    # dimension_weights: "kurtosis" | "auto" | [0.5, 0.3, ...]
    dw_raw = spec.get("dimension_weights")
    dim_weights: str | list[float] | None = None
    if isinstance(dw_raw, str):
        dim_weights = dw_raw
    elif isinstance(dw_raw, list):
        dim_weights = [float(w) for w in dw_raw]

    return PatternConfig(
        type=ptype,
        entity_line=str(entity_line),
        relations=relations,
        event_dimensions=event_dims,
        derived_dimensions=derived_dims,
        precomputed_dimensions=precomp_dims,
        graph_features=graph_feat,
        edge_table=edge_tbl_cfg,
        anomaly_percentile=float(spec.get("anomaly_percentile", 95.0)),
        dimension_weights=dim_weights,
        gmm_n_components=spec.get("gmm_n_components"),
        group_by_property=spec.get("group_by_property"),
        tracked_properties=spec.get("tracked_properties"),
        use_mahalanobis=bool(spec.get("use_mahalanobis", False)),
        description=spec.get("description"),
    )


def _parse_relation(
    pid: str,
    raw: Any,
    known_lines: set[str] | None = None,
) -> RelationConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Pattern '{pid}' relation must be a mapping")
    if "line" not in raw:
        raise ValueError(f"Pattern '{pid}' relation must specify 'line'")
    line = str(raw["line"])
    if known_lines is not None and line not in known_lines:
        raise ValueError(
            f"Pattern '{pid}' relation references unknown line '{line}'. "
            f"Available: {sorted(known_lines)}"
        )
    direction = str(raw.get("direction", "in"))
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Pattern '{pid}' relation direction '{direction}' invalid. "
            f"Valid: {sorted(_VALID_DIRECTIONS)}"
        )
    raw_em = raw.get("edge_max")
    edge_max = int(raw_em) if raw_em is not None else None
    return RelationConfig(
        line=line,
        direction=direction,
        key_on_entity=raw.get("key_on_entity"),
        required=bool(raw.get("required", True)),
        display_name=raw.get("display_name"),
        edge_max=edge_max,
    )


def _parse_derived_group(pid: str, raw: Any) -> DerivedDimGroup:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern '{pid}' derived_dimensions entry must be a mapping"
        )
    from_pattern = raw.get("from_pattern")
    raw_features = raw.get("features")
    if not raw_features:
        raise ValueError(
            f"Pattern '{pid}' derived_dimensions entry must have 'features'"
        )
    features = [_parse_feature_spec(pid, f) for f in raw_features]
    anchor_fk = raw.get("anchor_fk")
    return DerivedDimGroup(
        from_pattern=from_pattern,
        anchor_fk=str(anchor_fk) if anchor_fk else None,
        features=features,
    )


def _parse_feature_spec(pid: str, raw: Any) -> FeatureSpec:
    """Parse one feature entry like ``{tx_count: count}`` or
    ``{burst_daily: "count:window=1d:agg=max"}``."""
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ValueError(
            f"Pattern '{pid}' feature must be a single-key mapping "
            f"like {{dim_name: spec}}, got {raw}"
        )
    dim_name, spec_str = next(iter(raw.items()))
    dim_name = str(dim_name)
    spec_str = str(spec_str)

    # Split spec: "count" | "sum:amount" | "count:window=1d:agg=max"
    parts = spec_str.split(":")
    metric = parts[0].strip()

    # Map user-friendly aliases
    if metric == "avg":
        metric = "mean"

    if metric not in _VALID_METRICS and metric != "mean":
        raise ValueError(
            f"Pattern '{pid}' feature '{dim_name}' has unknown metric "
            f"'{metric}'. Valid: {sorted(_VALID_METRICS)}"
        )

    metric_col: str | None = None
    time_window: str | None = None
    window_agg: str = "max"

    # Parse remaining parts
    idx = 1
    if idx < len(parts):
        # Second part: either a column name or a key=value pair
        second = parts[idx].strip()
        if "=" not in second:
            # It's a metric_col
            metric_col = second
            idx += 1

    # Parse key=value pairs from remaining parts
    while idx < len(parts):
        kv = parts[idx].strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "window":
                time_window = v
            elif k == "agg":
                window_agg = v
            else:
                raise ValueError(
                    f"Pattern '{pid}' feature '{dim_name}' unknown option "
                    f"'{k}'. Valid options: window, agg"
                )
        else:
            raise ValueError(
                f"Pattern '{pid}' feature '{dim_name}' unexpected token "
                f"'{kv}' in spec '{spec_str}'"
            )
        idx += 1

    return FeatureSpec(
        dimension_name=dim_name,
        metric=metric,
        metric_col=metric_col,
        time_window=time_window,
        window_aggregation=window_agg,
    )


# ── composite lines ─────────────────────────────────────────────────


def _parse_composite_lines(
    raw: dict[str, Any],
    lines: dict[str, LineConfig],
) -> dict[str, CompositeLineConfig]:
    result: dict[str, CompositeLineConfig] = {}
    for cl_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Composite line '{cl_id}' must be a mapping")
        event_line = spec.get("event_line")
        if not event_line:
            raise ValueError(
                f"Composite line '{cl_id}' must specify 'event_line'"
            )
        if str(event_line) not in lines:
            raise ValueError(
                f"Composite line '{cl_id}' references unknown event_line "
                f"'{event_line}'. Available lines: {list(lines)}"
            )
        key_cols = spec.get("key_cols")
        if not key_cols or not isinstance(key_cols, list):
            raise ValueError(
                f"Composite line '{cl_id}' must specify 'key_cols' as a list"
            )

        derived_dims: list[DerivedDimGroup] | None = None
        raw_dd = spec.get("derived_dimensions")
        if raw_dd:
            derived_dims = [
                _parse_derived_group(cl_id, g) for g in raw_dd
            ]

        dw_raw = spec.get("dimension_weights")
        dim_weights: str | list[float] | None = None
        if isinstance(dw_raw, str):
            dim_weights = dw_raw
        elif isinstance(dw_raw, list):
            dim_weights = [float(w) for w in dw_raw]

        result[cl_id] = CompositeLineConfig(
            event_line=str(event_line),
            key_cols=[str(c) for c in key_cols],
            separator=str(spec.get("separator", "|")),
            derived_dimensions=derived_dims,
            anomaly_percentile=float(spec.get("anomaly_percentile", 95.0)),
            dimension_weights=dim_weights,
            description=spec.get("description"),
        )
    return result


# ── temporal ─────────────────────────────────────────────────────────


def _parse_temporal(
    raw: list[Any],
    patterns: dict[str, PatternConfig],
) -> list[TemporalConfig]:
    result: list[TemporalConfig] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"temporal[{i}] must be a mapping")
        pattern = entry.get("pattern")
        if not pattern:
            raise ValueError(f"temporal[{i}] must specify 'pattern'")
        if str(pattern) not in patterns:
            raise ValueError(
                f"temporal[{i}] references unknown pattern '{pattern}'. "
                f"Available: {list(patterns)}"
            )
        event_line = entry.get("event_line")
        if not event_line:
            raise ValueError(f"temporal[{i}] must specify 'event_line'")
        ts_col = entry.get("timestamp_col")
        if not ts_col:
            raise ValueError(f"temporal[{i}] must specify 'timestamp_col'")
        window = entry.get("window")
        if not window:
            raise ValueError(f"temporal[{i}] must specify 'window'")

        result.append(TemporalConfig(
            pattern=str(pattern),
            event_line=str(event_line),
            timestamp_col=str(ts_col),
            window=str(window),
        ))
    return result


# ── aliases ──────────────────────────────────────────────────────────


def _parse_aliases(
    raw: dict[str, Any],
    patterns: dict[str, PatternConfig],
) -> dict[str, AliasConfig]:
    result: dict[str, AliasConfig] = {}
    for alias_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Alias '{alias_id}' must be a mapping")
        base_pattern = spec.get("base_pattern")
        if not base_pattern:
            raise ValueError(f"Alias '{alias_id}' must specify 'base_pattern'")
        if str(base_pattern) not in patterns:
            raise ValueError(
                f"Alias '{alias_id}' references unknown base_pattern "
                f"'{base_pattern}'. Available: {list(patterns)}"
            )
        cp = spec.get("cutting_plane")
        if not cp or not isinstance(cp, dict):
            raise ValueError(
                f"Alias '{alias_id}' must specify 'cutting_plane' as a mapping"
            )
        # Two modes: dimension+threshold (sugar) or normal+bias (explicit)
        normal = cp.get("normal")
        bias = cp.get("bias")
        dimension = cp.get("dimension")
        threshold = cp.get("threshold")

        acfg = AliasConfig(
            base_pattern=str(base_pattern),
            cutting_plane_normal=[float(x) for x in normal] if normal else None,
            cutting_plane_bias=float(bias) if bias is not None else None,
            cutting_plane_dimension=dimension,
            cutting_plane_threshold=float(threshold) if threshold is not None else None,
            description=spec.get("description"),
        )
        result[alias_id] = acfg
    return result


# ── chain lines ─────────────────────────────────────────────────────


def _parse_chain_lines(
    raw: dict[str, Any],
    lines: dict[str, LineConfig] | None = None,
) -> dict[str, ChainLineConfig]:
    result: dict[str, ChainLineConfig] = {}
    for cl_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"chain_lines.{cl_id} must be a mapping")
        event_line = spec.get("event_line")
        if not event_line:
            raise ValueError(f"chain_lines.{cl_id} must specify 'event_line'")
        if lines is not None and str(event_line) not in lines:
            raise ValueError(
                f"chain_lines.{cl_id} references unknown event_line "
                f"'{event_line}'. Available: {list(lines)}"
            )
        from_col = spec.get("from_col")
        to_col = spec.get("to_col")
        if not from_col or not to_col:
            raise ValueError(
                f"chain_lines.{cl_id} must specify 'from_col' and 'to_col'"
            )
        result[cl_id] = ChainLineConfig(
            event_line=str(event_line),
            from_col=str(from_col),
            to_col=str(to_col),
            features=spec.get("features"),
            seed_percentile_fan_out=float(spec.get("seed_percentile_fan_out", 95.0)),
            seed_percentile_cross_bank=float(spec.get("seed_percentile_cross_bank", 90.0)),
            seed_multi_currency=int(spec.get("seed_multi_currency", 2)),
            seed_pass_through=bool(spec.get("seed_pass_through", True)),
            time_window_hours=int(spec.get("time_window_hours", 168)),
            max_hops=int(spec.get("max_hops", 15)),
            min_hops=int(spec.get("min_hops", 2)),
            max_chains=int(spec.get("max_chains", 300000)),
            bidirectional=bool(spec.get("bidirectional", True)),
            anomaly_percentile=float(spec.get("anomaly_percentile", 95.0)),
            description=spec.get("description"),
        )
    return result
