# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hypertopos.storage._schemas import (  # noqa: F401,E501
    EDGE_STRUCT_TYPE,
    GEOMETRY_EVENT_SCHEMA,
    GEOMETRY_SCHEMA,
)
from hypertopos.storage.calibration_history import compute_pattern_schema_hash

logger = logging.getLogger(__name__)

MIN_FILL_RATE: float = 0.05  # props with lower fill rate excluded from delta
MAX_FILL_RATE: float = 0.999  # props with higher fill rate excluded (zero-variance)
GEOMETRY_CHUNK_SIZE: int = 500_000  # entities above this threshold trigger chunked writes
_BOOTSTRAP_MAX_N: int = 50_000  # bootstrap skipped for populations above this size

_INTERNAL_COLUMNS = {"version", "status", "created_at", "changed_at"}


def _compute_schema_hash_for_pattern_def(
    pattern_def,
    *,
    prop_columns: list[str] | None = None,
    dimension_kinds: list[str] | None = None,
) -> str:
    """Compute schema_hash from a builder Pattern definition (duck-typed).

    The payload composition matches `_compute_schema_hash_from_pattern_node`
    so a fresh build produces a hash byte-identical to the one a reader
    reconstructs from the resulting sphere.json node.

    `prop_columns` / `dimension_kinds` overrides exist because builder's
    `_PatternReg` does not carry them — they are produced as part of the
    geometry refit. Callers in the build pipeline must pass the freshly
    computed values so the hash matches what gets written into sphere.json.
    """
    relations = []
    for rel in getattr(pattern_def, "relations", None) or []:
        relations.append(
            {
                "line_id": getattr(rel, "line_id", None) or getattr(rel, "line", None),
                "event_columns": list(getattr(rel, "event_columns", None) or []),
            }
        )
    # event_dimensions on a builder Pattern can be either list[str] (test
    # doubles) or list[EventDimSpec] (real builds). Reduce to column names so
    # the resulting payload matches the list-of-strings shape stored in
    # sphere.json that _compute_schema_hash_from_pattern_node sees on read.
    event_dims_raw = getattr(pattern_def, "event_dimensions", None) or []
    event_dimensions = [
        getattr(ed, "column", ed) for ed in event_dims_raw
    ]
    # prop_columns / dimension_kinds: prefer caller-supplied (post-refit)
    # values when given; otherwise fall back to whatever the pattern_def
    # carries (test doubles set these directly). Falling back to an empty
    # list keeps the 2.3 reconstructor reachable from a builder Pattern that
    # has not been through a refit yet.
    if prop_columns is None:
        prop_columns_raw = getattr(pattern_def, "prop_columns", None) or []
        prop_columns_final = list(prop_columns_raw)
    else:
        prop_columns_final = list(prop_columns)
    if dimension_kinds is None:
        dimension_kinds_raw = (
            getattr(pattern_def, "dimension_kinds", None) or []
        )
        dimension_kinds_final = list(dimension_kinds_raw)
    else:
        dimension_kinds_final = list(dimension_kinds)
    payload = {
        "relations": relations,
        "event_dimensions": event_dimensions,
        "prop_columns": prop_columns_final,
        "dimension_kinds": dimension_kinds_final,
    }
    return compute_pattern_schema_hash(payload)


def _is_textual_or_binary_col(field: pa.Field) -> bool:
    arrow_type = field.type
    if pa.types.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    return (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
        or pa.types.is_fixed_size_binary(arrow_type)
    )


def _arrow_type_to_str(arrow_type: pa.DataType) -> str:
    _mapping = {
        pa.string(): "string", pa.large_string(): "string",
        pa.int32(): "int32", pa.int64(): "int64",
        pa.float32(): "float32", pa.float64(): "float64",
        pa.bool_(): "bool", pa.date32(): "date32",
    }
    if arrow_type in _mapping:
        return _mapping[arrow_type]
    if pa.types.is_timestamp(arrow_type):
        return "timestamp"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        inner = _arrow_type_to_str(arrow_type.value_type)
        return f"list<{inner}>"
    return str(arrow_type)


def _validate_fdr_hierarchy_columns(
    pattern: Any,
    geometry_columns: list[str],
) -> None:
    """Verify fdr_hierarchy.from_dimension columns are present in geometry.

    Run AFTER `_inject_fdr_hierarchy_columns` and
    `_maybe_materialise_temporal_buckets` so the post-injection column set is
    what the validator sees. fdr_temporal_hierarchy.slice_dimension columns
    are NOT validated here — the builder materialises missing slice_dimension
    columns at build time via `_maybe_materialise_temporal_buckets`.
    """
    for level in getattr(pattern, "fdr_hierarchy", []) or []:
        if level.from_dimension not in geometry_columns:
            raise ValueError(
                f"Pattern {pattern.pattern_id!r}: fdr_hierarchy level "
                f"{level.level!r} references from_dimension "
                f"{level.from_dimension!r} which is not present in geometry "
                f"columns. Add this column to the anchor line schema or "
                f"correct the level name.",
            )


def _inject_fdr_hierarchy_columns(
    pattern: Any,
    *,
    geometry_table: pa.Table,
    anchor_table: pa.Table,
) -> pa.Table:
    """For each fdr_hierarchy.from_dimension column that is not yet on the
    geometry table, project it from the anchor line table via primary_key
    join and append it as a flat column.

    No-op when fdr_hierarchy is empty or every from_dimension is already
    present on the geometry table. Raises ValueError when a from_dimension
    is missing from BOTH geometry AND the anchor line — the anchor line is
    the only valid source for these carrier columns.
    """
    levels = getattr(pattern, "fdr_hierarchy", []) or []
    if not levels:
        return geometry_table
    existing = set(geometry_table.column_names)
    anchor_cols = set(anchor_table.column_names)
    entity_line = getattr(pattern, "entity_line_id", None) or "<unknown>"
    out = geometry_table
    for level in levels:
        col = level.from_dimension
        if col in existing:
            continue
        if col not in anchor_cols:
            raise ValueError(
                f"Pattern {pattern.pattern_id!r}: fdr_hierarchy level "
                f"{level.level!r} references from_dimension {col!r} which is "
                f"present on neither the geometry nor the anchor line "
                f"{entity_line!r}. Add {col!r} as a column on the anchor line "
                f"source or correct the from_dimension name.",
            )
        # PyArrow's Table.join doesn't support list/fixed_size_list non-key
        # fields (geometry's `delta` column is a fixed_size_list<float>), so
        # use a hash-based take instead: index by primary_key, then take
        # ordered by geometry's primary_key sequence.
        anchor_pk = anchor_table["primary_key"].to_pylist()
        anchor_vals = anchor_table[col].to_pylist()
        index = {pk: i for i, pk in enumerate(anchor_pk)}
        geom_pk = out["primary_key"].to_pylist()
        aligned = [
            anchor_vals[index[pk]] if pk in index else None for pk in geom_pk
        ]
        out = out.append_column(
            col,
            pa.array(aligned, type=anchor_table.schema.field(col).type),
        )
        existing.add(col)
    return out


def _auto_discover_event_pattern_for_anchor(
    anchor_pattern: Any,
    all_patterns: dict[str, Any],
) -> Any:
    """Find the unique event pattern whose ``relations`` reference the anchor's
    entity_line. Used to source the event_table + edge_table columns for
    materialising ``fdr_temporal_hierarchy.slice_dimension`` from event
    timestamps when the anchor pattern declares the hierarchy.

    The heuristic: an event pattern P references an anchor A iff at least one
    of P.relations has ``line_id == A.entity_line``. This is direction-agnostic
    at the pattern level — a single event pattern with two relations both
    pointing at the anchor's line is one candidate, not two. The auto-discover
    helper additionally requires the candidate to declare an ``edge_table``
    with a non-null ``timestamp_col``, since both are required by
    ``materialise_temporal_bucket``.

    Raises ValueError when zero, multiple, or no-edge-table candidates exist.
    """
    anchor_line = anchor_pattern.entity_line
    candidates: list[Any] = []
    for pat in all_patterns.values():
        if pat.pattern_type != "event":
            continue
        if pat.pattern_id == anchor_pattern.pattern_id:
            continue
        for rel in pat.relations:
            if rel.line_id == anchor_line:
                candidates.append(pat)
                break

    if not candidates:
        raise ValueError(
            f"Anchor pattern {anchor_pattern.pattern_id!r} declares "
            f"fdr_temporal_hierarchy but no event pattern references its "
            f"entity_line {anchor_line!r}; declare an event pattern with a "
            f"relation pointing at line {anchor_line!r}.",
        )

    if len(candidates) > 1:
        names = sorted(p.pattern_id for p in candidates)
        raise ValueError(
            f"Multiple event patterns {names} reference anchor "
            f"{anchor_pattern.pattern_id!r}'s entity_line {anchor_line!r}; "
            f"ambiguous — auto-discover for fdr_temporal_hierarchy supports "
            f"exactly one event pattern per anchor line. Disambiguate by "
            f"removing fdr_temporal_hierarchy from anchors that share "
            f"the line.",
        )

    event_pat = candidates[0]
    if event_pat.edge_table is None:
        raise ValueError(
            f"Anchor pattern {anchor_pattern.pattern_id!r} declares "
            f"fdr_temporal_hierarchy and the matching event pattern "
            f"{event_pat.pattern_id!r} has no edge_table; declare an "
            f"edge_table with timestamp_col on {event_pat.pattern_id!r} "
            f"so the bucket materialiser can read event timestamps.",
        )
    if event_pat.edge_table.timestamp_col is None:
        raise ValueError(
            f"Anchor pattern {anchor_pattern.pattern_id!r} declares "
            f"fdr_temporal_hierarchy and the matching event pattern "
            f"{event_pat.pattern_id!r} has an edge_table without "
            f"timestamp_col; declare a timestamp_col so the bucket "
            f"materialiser can derive per-anchor centroid timestamps.",
        )
    return event_pat


def _maybe_materialise_temporal_buckets(
    pattern: Any,
    *,
    geometry_table: pa.Table,
    event_table: pa.Table | None,
    anchor_key_col_options: tuple[str, ...],
    timestamp_col: str,
) -> pa.Table:
    """For each fdr_temporal_hierarchy level whose slice_dimension is not yet
    on geometry_table, materialise it from event timestamps via
    builder.temporal_bucket.materialise_temporal_bucket.

    No-op when no fdr_temporal_hierarchy declared OR all slice_dimensions
    already present. Returns the (possibly-extended) geometry table.
    """
    from hypertopos.builder.temporal_bucket import materialise_temporal_bucket

    levels = getattr(pattern, "fdr_temporal_hierarchy", []) or []
    if not levels:
        return geometry_table
    existing = set(geometry_table.column_names)
    out = geometry_table
    for level in levels:
        if level.slice_dimension in existing:
            continue
        if event_table is None:
            raise ValueError(
                f"Pattern {pattern.pattern_id!r}: fdr_temporal_hierarchy level "
                f"{level.level!r} requires materialising "
                f"{level.slice_dimension!r} from event timestamps, but no "
                f"event_table is in scope (anchor patterns without an event "
                f"line cannot materialise temporal buckets).",
            )
        bucket_table = materialise_temporal_bucket(
            event_table=event_table,
            anchor_keys=out["primary_key"].to_pylist(),
            anchor_key_col_options=anchor_key_col_options,
            timestamp_col=timestamp_col,
            bucket=level.bucket,
        )
        if level.slice_dimension != "temporal_bucket":
            bucket_table = bucket_table.rename_columns(
                ["primary_key", level.slice_dimension],
            )
        # Same list-typed-field limitation as in _inject_fdr_hierarchy_columns:
        # pa.Table.join rejects fixed_size_list non-key fields (geometry's
        # `delta`). Hash-based take instead.
        col_name = level.slice_dimension
        bucket_pk = bucket_table["primary_key"].to_pylist()
        bucket_vals = bucket_table[col_name].to_pylist()
        index = {pk: i for i, pk in enumerate(bucket_pk)}
        geom_pk = out["primary_key"].to_pylist()
        aligned = [
            bucket_vals[index[pk]] if pk in index else None for pk in geom_pk
        ]
        out = out.append_column(
            col_name,
            pa.array(aligned, type=bucket_table.schema.field(col_name).type),
        )
        existing.add(col_name)
    return out


def compute_entity_geometry(
    entity_table: pa.Table,
    mu: np.ndarray,
    sigma: np.ndarray,
    relations_meta: list[dict],
    event_dimensions_meta: list[dict] | None = None,
    dimension_weights: np.ndarray | None = None,
    prop_columns: list[str] | None = None,
    edge_dim_agg_labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute deltas, norms, and shape vectors for entities using existing pattern stats.

    Builds shape vectors from entity_table columns based on relations, event
    dimensions, edge_dim_aggregation, and prop_columns metadata (from
    sphere.json), then z-scores against the provided mu/sigma to produce delta
    vectors.

    Block layout matches the full-build concatenation order: relations →
    event_dimensions → edge_dim_aggregations → prop_columns. The aggregated
    edge-dim values are read directly from ``entity_table`` columns named by
    ``edge_dim_agg_labels`` (the caller-supplied precomputed ``{dim}_{agg}``
    columns) and z-scored against mu/sigma like any other precomputed feature.

    Returns:
        (deltas, delta_norms, shape_vectors) all as float32 arrays.
    """
    n = len(entity_table)
    D_rel = len(relations_meta)
    D_event = len(event_dimensions_meta) if event_dimensions_meta else 0
    D_agg = len(edge_dim_agg_labels) if edge_dim_agg_labels else 0
    D_prop = len(prop_columns) if prop_columns else 0

    shape_vectors = np.zeros(
        (n, D_rel + D_event + D_agg + D_prop), dtype=np.float32,
    )

    for j, rel in enumerate(relations_meta):
        direction = rel.get("direction", "in")
        edge_max = rel.get("edge_max")

        if direction == "self":
            shape_vectors[:, j] = 1.0
        elif edge_max is not None:
            # Continuous count dimension — find FK column
            fk_col_name = rel.get("fk_col")
            if fk_col_name and fk_col_name in entity_table.schema.names:
                col = entity_table[fk_col_name]
                count_arr = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)
                shape_vectors[:, j] = np.clip(count_arr, 0, edge_max) / edge_max
            # If fk_col not found, shape stays 0.0 (dead edge)
        else:
            # Binary FK presence
            fk_col_name = rel.get("fk_col")
            if fk_col_name and fk_col_name in entity_table.schema.names:
                col_arrow = entity_table[fk_col_name]
                valid_mask = pc.fill_null(
                    pc.and_(
                        pc.is_valid(col_arrow),
                        pc.not_equal(col_arrow, ""),
                    ),
                    False,
                )
                shape_vectors[:, j] = valid_mask.to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)

    # Event dimensions
    if event_dimensions_meta:
        for k, edim in enumerate(event_dimensions_meta):
            col_name = edim["column"]
            em = edim["edge_max"]
            if col_name in entity_table.schema.names:
                col = entity_table[col_name]
                raw_arr = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)
                if isinstance(em, (int, float)) and em > 0:
                    shape_vectors[:, D_rel + k] = np.clip(raw_arr / em, 0.0, 3.0)

    # Edge-dim aggregation block — raw precomputed values read by label.
    # The full build appends these AFTER event_dims and BEFORE prop fill
    # (builder._compute_population_stats concatenation order). Values are
    # stored unscaled; mu/sigma carry the scale, so the global z-score below
    # normalizes them like any other precomputed feature.
    if edge_dim_agg_labels:
        D_base = D_rel + D_event
        for k, label in enumerate(edge_dim_agg_labels):
            if label in entity_table.schema.names:
                col = entity_table[label]
                shape_vectors[:, D_base + k] = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)
            # Missing column → stays 0.0 (no contribution, consistent with
            # absent FK relations above).

    # Prop columns — binary fill (0/1 based on is_valid)
    if prop_columns:
        D_base = D_rel + D_event + D_agg
        for k, prop in enumerate(prop_columns):
            if prop in entity_table.schema.names:
                col = entity_table[prop]
                fill_vec = pc.is_valid(col).to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)
                shape_vectors[:, D_base + k] = fill_vec

    # Z-score against existing mu/sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    d = shape_vectors.shape[1]
    deltas = (
        (shape_vectors - mu[:d]) / sigma_safe[:d]
    ).astype(np.float32)

    # Apply dimension weights if present
    if dimension_weights is not None:
        deltas = (deltas * dimension_weights[: deltas.shape[1]]).astype(np.float32)

    delta_norms = np.sqrt(
        np.einsum("ij,ij->i", deltas, deltas),
    ).astype(np.float32)

    return deltas, delta_norms, shape_vectors


def _edge_dim_aggregation_labels(eda_meta: dict | None) -> list[str]:
    """Reconstruct ``{source_dim}_{aggregate}`` labels in build order from the
    serialized ``edge_dim_aggregations`` sphere.json node.

    Mirrors ``Pattern._edge_dim_aggregation_names``: one label per aggregate
    per source dim, in ``dims`` insertion order. Returns ``[]`` when the
    pattern declares no aggregations.
    """
    if not eda_meta:
        return []
    dims = eda_meta.get("dims") or []
    per_dim = eda_meta.get("aggregates_per_dim")
    if per_dim is None:
        from hypertopos.engine.edge_features import AGGREGATE_NAMES
        per_dim = {d: list(AGGREGATE_NAMES) for d in dims}
    labels: list[str] = []
    for d in dims:
        for agg_name in per_dim.get(d, ()):
            labels.append(f"{d}_{agg_name}")
    return labels


def _classify_changed_keys(
    lance_path: str,
    primary_keys: list[str],
) -> tuple[list[str], list[str]]:
    """Classify keys as new (not in geometry) or modified (already exists).

    Returns (new_keys, modified_keys).
    """
    import lance

    ds = lance.dataset(lance_path)
    escaped = [k.replace("'", "''") for k in primary_keys]
    in_clause = ", ".join(f"'{k}'" for k in escaped)
    existing_table = ds.to_table(
        columns=["primary_key"],
        filter=f"primary_key IN ({in_clause})",
    )
    existing_set = set(existing_table["primary_key"].to_pylist())
    new_keys = [k for k in primary_keys if k not in existing_set]
    modified_keys = [k for k in primary_keys if k in existing_set]
    return new_keys, modified_keys


@dataclass
class RelationSpec:
    """Defines one dimension of a pattern: which line and how to find the FK."""

    line_id: str
    fk_col: str | None  # column name in entity data; None for direction="self"
    direction: Literal["in", "out", "self"] = "in"
    required: bool = True
    display_name: str | None = None
    edge_max: int | None = None  # None = binary; int = continuous count cap


@dataclass
class EventDimSpec:
    """Continuous dimension for event patterns: reads a value column from entity data."""
    column: str                      # column name in entity table
    edge_max: float | str = "auto"   # float = fixed, "auto" = p99
    display_name: str | None = None  # label in dim_labels (defaults to column)
    percentile: float = 99.0         # percentile for auto edge_max


@dataclass
class _LineReg:
    line_id: str
    table: pa.Table  # normalized Arrow table with mandatory columns
    role: str  # "anchor" or "event"
    partition_col: str | None
    entity_type: str
    source_id: str
    fts_columns: list[str] | str | None = None
    description: str | None = None


@dataclass
class EdgeTableConfig:
    """Config for edge table emission during build."""

    from_col: str
    to_col: str
    timestamp_col: str | None = None
    amount_col: str | None = None


@dataclass
class _PatternReg:
    pattern_id: str
    pattern_type: Literal["anchor", "event"]
    entity_line: str  # line_id of primary entity
    relations: list[RelationSpec]
    anomaly_percentile: float  # default 95.0
    tracked_properties: list[str] = field(default_factory=list)
    group_by_property: str | None = None
    dimension_weights: list[float] | str | None = None  # None/"uniform"/list/"auto"
    gmm_n_components: int | None = None  # None = disabled, int = fit GMM with k components
    use_mahalanobis: bool = False
    event_dimensions: list[EventDimSpec] = field(default_factory=list)
    description: str | None = None
    edge_table: EdgeTableConfig | None = None  # None = auto-detect or skip
    bootstrap_iterations: int = 200
    # Generalized dimension blocks (g/t/s)
    geo_properties: list[str] | None = None
    metric_properties: list[str] | None = None
    semantic_dim: dict | None = None  # {"columns": [...], "n_components": int}
    # Per-edge derived dim catalog (event patterns only).
    # ``EdgeDimensionsConfig.dims`` keyed by dim_name → params dict.
    edge_dimensions: Any = None  # EdgeDimensionsConfig | None
    edge_dim_aggregations: Any = None  # EdgeDimAggregationsConfig | None
    # Multi-resolution FDR hierarchies (lists of FDRHierarchyLevel /
    # FDRTemporalLevel from hypertopos.model.sphere). Empty defaults mean
    # no behaviour change for patterns that do not opt in.
    fdr_hierarchy: list = field(default_factory=list)
    fdr_temporal_hierarchy: list = field(default_factory=list)
    # Declarative conformance rules (M1.7) — list of ConformanceRule from
    # hypertopos.model.sphere. Empty default is the cost-neutral fast path;
    # when populated, the builder evaluates them after points materialization
    # and writes a sidecar Lance dataset of violations.
    conformance_rules: list = field(default_factory=list)


@dataclass
class _AliasReg:
    alias_id: str
    base_pattern_id: str
    cutting_plane_normal: list[float] | None = None
    cutting_plane_bias: float | None = None
    cutting_plane_dimension: int | str | None = None
    cutting_plane_threshold: float | None = None
    description: str | None = None


@dataclass
class PopulationStats:
    """All population-level statistics returned by _compute_population_stats."""

    mu: np.ndarray
    sigma: np.ndarray
    theta: np.ndarray
    deltas: np.ndarray
    delta_norms: np.ndarray
    delta_rank_pcts: np.ndarray
    conformal_p: np.ndarray
    fk_arrays: list  # list[pa.ChunkedArray | np.ndarray | None]
    prop_columns: list[str]
    excluded_properties: list[str]
    group_stats_dict: (
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] | None
    )
    is_anomaly_arr: np.ndarray | None
    dim_weights: np.ndarray | None
    gmm_components: list | None
    cholesky_inv: np.ndarray | None
    n_anom_dims: np.ndarray
    bregman_norms: np.ndarray | None = None
    dimension_kinds: list[str] | None = None
    theta_per_dim: np.ndarray | None = None
    dim_block_names: list[str] = field(default_factory=list)
    dim_block_stats: dict[str, Any] | None = None
    theta_sensitivity: dict[str, dict[str, float]] | None = None
    # Per-entity values for aggregated edge dims, (N, n_agg_cols) float32.
    # None when the pattern declares no edge_dim_aggregations.
    # Parallel label list — canonical "{dim}_{agg}" names in column order,
    # matching ``Pattern._edge_dim_aggregation_names`` at read time.
    edge_dim_agg_matrix: np.ndarray | None = None
    edge_dim_agg_labels: list[str] = field(default_factory=list)
    # Brown-Forsythe (median-centred Levene) diagnostic of delta_norm
    # variance equality across the levels of `group_by_property`. None
    # when the pattern has no group_by_property or when fewer than two
    # groups survive the low-N filter inside the diagnostic. Keyed by
    # the grouping column name to leave room for future multi-group
    # variants without a schema break.
    heteroscedasticity_diagnostic: dict[str, dict[str, Any]] | None = None
    # Edge-derived per-event dim names emitted on event patterns when
    # ``edge_dimensions:`` is declared. Order matches the shape-vector
    # concatenation between event_dim_matrix and prop_fill_matrix; empty
    # for patterns without edge_dimensions.
    edge_dim_names: list[str] = field(default_factory=list)


@dataclass
class PatternBuildResult:
    """Fields passed from geometry build to sphere.json generation."""

    mu: np.ndarray
    sigma: np.ndarray
    theta: np.ndarray
    population_size: int
    prop_columns: list[str]
    excluded_properties: list[str]
    group_stats: (
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] | None
    )
    dimension_weights: np.ndarray | None
    gmm_components: list | None
    cholesky_inv: np.ndarray | None
    dim_percentiles: dict[str, dict[str, float]] | None = None
    dimension_kinds: list[str] | None = None
    dim_block_names: list[str] = field(default_factory=list)
    dim_block_stats: dict[str, Any] | None = None
    edge_dim_thresholds: dict[str, float] | None = None
    theta_sensitivity: dict[str, dict[str, float]] | None = None
    heteroscedasticity_diagnostic: dict[str, dict[str, Any]] | None = None
    dim_normality_pvalues: dict[str, float] | None = None
    # Edge-derived per-event dim names — populated on event patterns with
    # ``edge_dimensions:`` declared, empty otherwise.
    edge_dim_names: list[str] = field(default_factory=list)
    # Label-aware per-dim calibration (Fisher LDA per-dim moments + global
    # direction). Populated by ``_run_label_aware_calibration`` when the
    # pattern is listed under ``label_audit.patterns`` in sphere.yaml AND
    # ``_label_aware_calibration`` is set. ``None`` otherwise. Persisted
    # into sphere.json as ``label_aware_calibration`` and hydrated back
    # into ``Pattern.label_aware_calibration`` by the storage reader.
    label_aware_calibration: dict[str, Any] | None = None
    # Co-populated with ``label_aware_calibration`` — positive / negative
    # labelled sample counts and the percentiles + intrinsic / extrinsic
    # displacement means computed from the signed delta projection. ``None``
    # on patterns without label-aware calibration.
    label_aware_n_pos: int | None = None
    label_aware_n_neg: int | None = None
    signed_percentiles: dict[str, float] | None = None
    intrinsic_displacement_mean: float | None = None
    extrinsic_displacement_mean: float | None = None


@dataclass
class IncrementalUpdateResult:
    """Result of an incremental geometry update."""

    pattern_id: str
    added: int
    modified: int
    deleted: int
    drift_pct: float
    recalibrated: bool
    theta_norm: float
    population_size: int


# ── Temporal helpers (extracted from build_temporal closure) ──────────


def _temporal_tensor_to_lance(
    shape_tensor: np.ndarray,
    keys_list: list[str],
    n_ent: int,
    non_empty: np.ndarray,
    lance_p: Any,
    min_ts: float,
    win_secs: float,
    D: int,
) -> None:
    from hypertopos.storage.writer import _TEMPORAL_SCHEMA, _write_lance

    n_w = len(non_empty)
    if n_w == 0:
        return
    total = n_ent * n_w
    flat = shape_tensor[:, non_empty, :].reshape(total, D)
    pk_idx = np.repeat(np.arange(n_ent), n_w)
    bkt_idx = np.tile(non_empty, n_ent)
    bkt_ts = [
        datetime.fromtimestamp(min_ts + int(b) * win_secs, tz=UTC)
        for b in non_empty
    ]
    pk_c = pa.array([keys_list[i] for i in pk_idx], type=pa.string())
    ts_c = pa.array(
        [bkt_ts[i % n_w] for i in range(total)],
        type=pa.timestamp("us", tz="UTC"),
    )
    sc = (
        pa.FixedSizeListArray.from_arrays(
            pa.array(flat.ravel(), type=pa.float32()), list_size=D,
        ).cast(pa.list_(pa.float32()))
        if D > 0
        else pa.array([[] for _ in range(total)], type=pa.list_(pa.float32()))
    )
    tbl = pa.table({
        "primary_key": pk_c,
        "slice_index": pa.array(bkt_idx.astype(np.int32), type=pa.int32()),
        "timestamp": ts_c,
        "deformation_type": pa.array(["window_snapshot"] * total),
        "shape_snapshot": sc,
        "pattern_ver": pa.array(np.full(total, 1, dtype=np.int32)),
        "changed_property": pa.nulls(total, type=pa.string()),
        "changed_line_id": pa.nulls(total, type=pa.string()),
    }, schema=_TEMPORAL_SCHEMA)
    mode = "append" if lance_p.exists() else "create"
    _write_lance(tbl, str(lance_p), mode=mode)


def _build_traj_chunk(
    traj_tables: list,
    tensor: np.ndarray,
    pks: list[str],
    bkt_ts: list,
    mu_a: np.ndarray | None,
    sig_a: np.ndarray | None,
) -> None:
    nt, nb, dt = tensor.shape
    if nt == 0 or nb < 2:
        return
    d2 = dt * 2
    _s = np.maximum(sig_a, 1e-2) if sig_a is not None else None
    d = (tensor - mu_a) / _s if mu_a is not None and _s is not None else tensor
    tv = np.concatenate([d.mean(axis=1), d.std(axis=1)], axis=1).astype(np.float32)
    disp = np.linalg.norm(d[:, -1, :] - d[:, 0, :], axis=1).astype(np.float32)
    ts = pa.schema([
        pa.field("primary_key", pa.string()),
        pa.field("trajectory_vector", pa.list_(pa.float32(), d2)),
        pa.field("displacement", pa.float32()),
        pa.field("num_slices", pa.int32()),
        pa.field("first_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("last_timestamp", pa.timestamp("us", tz="UTC")),
    ])
    tvc = pa.FixedSizeListArray.from_arrays(
        pa.array(tv.ravel(), type=pa.float32()), list_size=d2,
    ).cast(pa.list_(pa.float32(), d2))
    traj_tables.append(pa.table({
        "primary_key": pa.array(pks, type=pa.string()),
        "trajectory_vector": tvc,
        "displacement": pa.array(disp, type=pa.float32()),
        "num_slices": pa.array(np.full(nt, nb, dtype=np.int32)),
        "first_timestamp": pa.array([bkt_ts[0]] * nt, type=pa.timestamp("us", tz="UTC")),
        "last_timestamp": pa.array([bkt_ts[-1]] * nt, type=pa.timestamp("us", tz="UTC")),
    }, schema=ts))


class GDSBuilder:
    """Build a navigable GDS sphere from DataFrames / Arrow Tables."""

    def __init__(
        self,
        sphere_id: str,
        output_path: str,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.sphere_id = sphere_id
        self.output_path = Path(output_path)
        self._name = name
        self._description = description
        self._lines: dict[str, _LineReg] = {}
        self._patterns: dict[str, _PatternReg] = {}
        self._derived_dims: list = []  # DerivedDimSpec list
        self._composite_lines: list = []  # CompositeLineSpec list
        self._graph_features: list = []  # GraphFeaturesSpec list
        self._chain_dims: list = []  # (line_id, feature_name, edge_max) tuples
        self._chain_lines: set[str] = set()  # line_ids registered via add_chain_line
        self._precomputed_dims: list = []  # PrecomputedDimSpec list
        self._aliases: dict[str, _AliasReg] = {}
        self._no_edges: bool = False  # set by CLI --no-edges
        # Opt-in label-aware calibration plumbing. The CLI sets
        # `_label_aware_calibration = True`; the YAML loader populates
        # `_label_audit_block` with the pattern selection and label
        # column resolution. Until both are set, the build path
        # short-circuits — no behavior change for unlabeled spheres.
        self._label_aware_calibration: bool = False
        self._label_audit_block: object | None = None
        # In-build registry of per-pattern label-aware Fisher LDA direction
        # vectors. Populated by the label-aware calibration hook (driven by
        # ``_label_aware_calibration`` + ``_label_audit_block``) before the
        # geometry pass; the geometry pass projects each polygon's delta
        # vector onto the registered direction and writes the scalar to
        # the ``delta_norm_signed`` Lance column. Patterns without an entry
        # get all-null ``delta_norm_signed`` — column is nullable.
        self._label_aware_directions: dict[str, np.ndarray] = {}
        # Per-pattern calibration epoch state populated during build by
        # _write_calibration_epoch_for_pattern. Each thread writes a distinct
        # pattern_id key, so plain assignment is safe under ThreadPoolExecutor.
        self._calibration_state: dict[str, dict] = {}
        # Per-pattern edge_dim_aggregations thresholds populated during the
        # population-stats dispatch when the anchor pattern declares
        # `edge_dim_aggregations:`. Keyed by anchor pattern_id, value is the
        # per-source-dim threshold (population p95 by default; user override
        # path planned). Persisted into the calibration epoch JSON so
        # `compare_calibrations` can surface threshold drift across epochs.
        self._edge_dim_thresholds: dict[str, dict[str, float]] = {}

    def _resolve_and_persist_edge_dim_thresholds(
        self,
        pattern_id: str,
        sidecar: pa.Table,
        dims: list[str],
        user_overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        from hypertopos.engine.edge_features import _resolve_count_above_thresholds

        resolved = _resolve_count_above_thresholds(sidecar, dims, user_overrides)
        self._edge_dim_thresholds[pattern_id] = resolved
        return resolved

    def _dim_labels_for_pattern(
        self,
        pat: _PatternReg,
        ps: PopulationStats,
    ) -> list[str]:
        """Compose the dim_label list for a pattern in storage-layout order.

        Mirrors ``Pattern.dim_labels``: relations → event_dimensions →
        edge_dim_names → prop_columns → edge_dim_agg_labels. Built from
        ``_PatternReg`` + ``PopulationStats`` because at calibration time
        the runtime ``Pattern`` dataclass has not been constructed yet.
        """
        labels: list[str] = []
        for r in pat.relations:
            labels.append(r.display_name if r.display_name else r.line_id)
        for ed in pat.event_dimensions:
            labels.append(ed.display_name or ed.column)
        labels.extend(ps.edge_dim_names)
        labels.extend(ps.prop_columns)
        labels.extend(ps.edge_dim_agg_labels)
        return labels

    def _run_label_aware_calibration(
        self,
        pat: _PatternReg,
        ps: PopulationStats,
    ) -> dict[str, Any] | None:
        """Fit label-aware per-dim calibration when the pattern opts in.

        Returns a bundle ``{"per_dim": {dim_label: DimCalibration},
        "n_pos": int, "n_neg": int, "direction": np.ndarray}`` and
        registers the global Fisher LDA direction in
        ``self._label_aware_directions[pat.pattern_id]`` so the downstream
        geometry pass can populate ``delta_norm_signed``. Returns ``None``
        when the pattern is not opted in, when prerequisites are missing,
        or when the LDA fit raises (degenerate inputs are logged and
        treated as "no calibration available" — never abort the build).

        Hook contract: callers MUST invoke this AFTER
        ``_compute_population_stats`` returns (``ps.deltas`` is the full
        delta matrix) AND BEFORE ``_build_geometry_slice`` runs (the
        slice reads ``_label_aware_directions``).
        """
        if not self._label_aware_calibration:
            return None
        block = self._label_audit_block
        if block is None:
            return None
        if pat.pattern_id not in set(block.patterns):
            return None
        # Streaming path skips this hook entirely (the delta matrix is
        # not materialised). The plan accepts this as a known limitation;
        # warn so the user can re-run with a non-streaming pattern shape.
        deltas = ps.deltas
        if deltas is None or deltas.ndim != 2 or deltas.shape[1] == 0:
            logger.warning(
                "label-aware calibration skipped for pattern %r — "
                "no delta matrix available (streaming or zero-dim path)",
                pat.pattern_id,
            )
            return None

        entity_table = self._lines[pat.entity_line].table
        if block.label_column not in entity_table.schema.names:
            logger.warning(
                "label-aware calibration skipped for pattern %r — "
                "label_column %r not present on entity line %r",
                pat.pattern_id, block.label_column, pat.entity_line,
            )
            return None
        label_col = entity_table[block.label_column]
        # Binarise: positive value matches → 1, everything else → 0.
        # PyArrow's ``equal`` handles mixed-type comparison safely (the
        # YAML loader keeps ``label_positive_value`` as the raw scalar).
        # ``fill_null(False)`` turns null entries into non-matches so the
        # bool mask round-trips cleanly through ``to_numpy``.
        pos_value = block.label_positive_value
        try:
            mask = pc.fill_null(
                pc.equal(label_col, pa.scalar(pos_value)),
                False,
            ).to_numpy(zero_copy_only=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "label-aware calibration skipped for pattern %r — "
                "could not compare label_column %r against positive "
                "value %r: %s",
                pat.pattern_id, block.label_column, pos_value, exc,
            )
            return None
        labels = np.where(mask, 1, 0).astype(np.int32)
        if labels.shape[0] != deltas.shape[0]:
            logger.warning(
                "label-aware calibration skipped for pattern %r — "
                "label vector length %d != delta row count %d",
                pat.pattern_id, labels.shape[0], deltas.shape[0],
            )
            return None
        if labels.sum() == 0 or labels.sum() == labels.shape[0]:
            logger.warning(
                "label-aware calibration skipped for pattern %r — "
                "only one class present in label_column %r",
                pat.pattern_id, block.label_column,
            )
            return None

        dim_labels = self._dim_labels_for_pattern(pat, ps)
        if len(dim_labels) != deltas.shape[1]:
            logger.warning(
                "label-aware calibration skipped for pattern %r — "
                "dim_label count %d != delta column count %d",
                pat.pattern_id, len(dim_labels), deltas.shape[1],
            )
            return None

        from hypertopos.engine.calibration_label_aware import (
            calibrate_label_aware,
        )

        try:
            result = calibrate_label_aware(
                deltas=deltas.astype(np.float32, copy=False),
                labels=labels,
                dim_labels=dim_labels,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "label-aware calibration failed for pattern %r — %s",
                pat.pattern_id, exc,
            )
            return None

        direction_vec = np.asarray(
            result.signed_direction_vector, dtype=np.float32,
        )
        self._label_aware_directions[pat.pattern_id] = direction_vec
        return {
            "per_dim": dict(result.per_dim),
            "n_pos": int(result.n_pos),
            "n_neg": int(result.n_neg),
            "direction": direction_vec,
        }

    def add_line(
        self,
        line_id: str,
        data: pa.Table | list[dict[str, Any]],
        key_col: str,
        source_id: str,
        role: str = "anchor",
        partition_col: str | None = None,
        entity_type: str | None = None,
        fts_columns: list[str] | str | None = None,
        description: str | None = None,
    ) -> GDSBuilder:
        # 1. Normalize input to pa.Table
        if isinstance(data, list):
            if not data:
                table = pa.table({})
            else:
                all_keys = dict.fromkeys(k for r in data for k in r)
                table = pa.table({k: [r.get(k) for r in data] for k in all_keys})
        else:
            table = data

        # 2. Rename key column → primary_key
        if key_col != "primary_key" and key_col in table.schema.names:
            table = table.rename_columns(
                ["primary_key" if name == key_col else name for name in table.schema.names]
            )

        # 3. Add mandatory columns with defaults if missing
        n = len(table)
        now = datetime.now(UTC)
        ts_type = pa.timestamp("us", tz="UTC")
        if n > 0:
            now_arr = pa.array([now], type=ts_type).take(
                pa.array(np.zeros(n, dtype=np.int32))
            )
        else:
            now_arr = pa.array([], type=ts_type)
        if "version" not in table.schema.names:
            table = table.append_column("version", pa.array([1] * n, type=pa.int32()))
        if "status" not in table.schema.names:
            table = table.append_column("status", pa.array(["active"] * n, type=pa.string()))
        if "created_at" not in table.schema.names:
            table = table.append_column("created_at", now_arr)
        if "changed_at" not in table.schema.names:
            table = table.append_column("changed_at", now_arr)

        self._lines[line_id] = _LineReg(
            line_id=line_id,
            table=table,
            role=role,
            partition_col=partition_col,
            entity_type=entity_type or line_id,
            source_id=source_id,
            fts_columns=fts_columns,
            description=description,
        )
        return self

    def add_pattern(
        self,
        pattern_id: str,
        pattern_type: Literal["anchor", "event"],
        entity_line: str,
        relations: list[RelationSpec],
        anomaly_percentile: float = 95.0,
        tracked_properties: list[str] | None = None,
        group_by_property: str | None = None,
        dimension_weights: list[float] | str | None = None,
        gmm_n_components: int | None = None,
        use_mahalanobis: bool = False,
        description: str | None = None,
        edge_table: EdgeTableConfig | None = None,
        geo_properties: list[str] | None = None,
        metric_properties: list[str] | None = None,
        semantic_dim: dict | None = None,
        bootstrap_iterations: int = 200,
        edge_dimensions: Any = None,
        edge_dim_aggregations: Any = None,
        fdr_hierarchy: list | None = None,
        fdr_temporal_hierarchy: list | None = None,
        conformance_rules: list | None = None,
    ) -> GDSBuilder:
        _VALID_DW = ("auto", "kurtosis", "uniform")
        if isinstance(dimension_weights, str) and dimension_weights not in _VALID_DW:
            raise ValueError(
                f"Pattern '{pattern_id}': dimension_weights='{dimension_weights}' "
                f"is not valid. Expected one of {_VALID_DW}, an explicit list, or None."
            )
        self._patterns[pattern_id] = _PatternReg(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            entity_line=entity_line,
            relations=relations,
            anomaly_percentile=anomaly_percentile,
            tracked_properties=tracked_properties or [],
            group_by_property=group_by_property,
            dimension_weights=dimension_weights,
            gmm_n_components=gmm_n_components,
            use_mahalanobis=use_mahalanobis,
            description=description,
            edge_table=edge_table,
            bootstrap_iterations=bootstrap_iterations,
            geo_properties=geo_properties,
            metric_properties=metric_properties,
            semantic_dim=semantic_dim,
            edge_dimensions=edge_dimensions,
            edge_dim_aggregations=edge_dim_aggregations,
            fdr_hierarchy=fdr_hierarchy or [],
            fdr_temporal_hierarchy=fdr_temporal_hierarchy or [],
            conformance_rules=conformance_rules or [],
        )
        return self

    def add_event_dimension(
        self,
        pattern_id: str,
        column: str,
        edge_max: float | str = "auto",
        display_name: str | None = None,
        percentile: float = 99.0,
    ) -> GDSBuilder:
        """Add a continuous dimension to an event pattern.

        Reads numeric values from the entity table column and normalizes
        by edge_max. Use for per-event anomaly detection on monetary
        amounts, quantities, etc.
        """
        if pattern_id not in self._patterns:
            raise ValueError(
                f"Pattern '{pattern_id}' not registered — call add_pattern() first"
            )
        pat = self._patterns[pattern_id]
        if pat.pattern_type != "event":
            raise ValueError(
                f"add_event_dimension only applies to event patterns, "
                f"'{pattern_id}' is '{pat.pattern_type}'"
            )
        pat.event_dimensions.append(EventDimSpec(
            column=column,
            edge_max=edge_max,
            display_name=display_name,
            percentile=percentile,
        ))
        return self

    def add_derived_dimension(
        self,
        anchor_line: str,
        event_line: str,
        anchor_fk: str | list[str],
        metric: str,
        metric_col: str | None,
        dimension_name: str,
        edge_max: int | str = "auto",
        percentile: float = 99.0,
        time_col: str | None = None,
        time_window: str | None = None,
        window_aggregation: str = "max",
    ) -> GDSBuilder:
        """Add a continuous dimension derived from event data aggregation.

        Computes per-anchor-entity aggregates from event data and creates
        a continuous dimension with edge_max normalization.

        Args:
            anchor_line: Target anchor line to add the dimension to.
            event_line: Source event line with raw data.
            anchor_fk: FK column(s) in event line pointing to anchor.
                str for single-key, list[str] for composite key anchors.
            metric: "count" | "count_distinct" | "sum" | "max" | "std" | "mean"
            metric_col: Column to aggregate (None for "count").
            dimension_name: Name in delta-space.
            edge_max: int = fixed, "auto" = p{percentile} of distribution.
            percentile: Percentile for auto edge_max (default 99.0).
            time_col: Timestamp column for temporal windowing (None = lifetime).
            time_window: Window size, e.g. "7d", "24h", "30d" (None = lifetime).
            window_aggregation: How to pick across windows: "max" | "mean" | "last".
        """
        if metric not in ("count",) and not metric.startswith("iet_") and metric_col is None:
            raise ValueError(
                f"metric_col is required for metric='{metric}' "
                f"(only 'count' and 'iet_*' allow metric_col=None)"
            )
        if isinstance(edge_max, str) and edge_max != "auto":
            raise ValueError(
                f"edge_max must be int or 'auto', got '{edge_max}'"
            )

        from hypertopos.builder.derived import DerivedDimSpec

        self._derived_dims.append(DerivedDimSpec(
            anchor_line=anchor_line,
            event_line=event_line,
            anchor_fk=anchor_fk,
            metric=metric,
            metric_col=metric_col,
            dimension_name=dimension_name,
            edge_max=edge_max,
            percentile=percentile,
            time_col=time_col,
            time_window=time_window,
            window_aggregation=window_aggregation,
        ))
        return self

    def add_composite_line(
        self,
        line_id: str,
        event_line: str,
        key_cols: list[str],
        separator: str = "→",
    ) -> GDSBuilder:
        """Create an anchor line with composite keys from event data.

        Extracts unique (key_cols[0], key_cols[1], ...) tuples from the event
        line and registers them as an anchor line with composite primary_key.

        Args:
            line_id: Name for the new composite line.
            event_line: Source event line.
            key_cols: Columns whose unique combinations form entity keys.
            separator: Separator for composite key (default "→").
        """
        from hypertopos.builder.derived import CompositeLineSpec

        self._composite_lines.append(CompositeLineSpec(
            line_id=line_id,
            event_line=event_line,
            key_cols=key_cols,
            separator=separator,
        ))
        return self

    def add_precomputed_dimension(
        self,
        anchor_line: str,
        dimension_name: str,
        edge_max: int | str = "auto",
        percentile: float = 99.0,
        display_name: str | None = None,
    ) -> GDSBuilder:
        """Add a dimension from a column already present on the anchor entity table.

        Use when the caller pre-computes aggregates in their own pipeline
        (SQL, Polars, pandas) and passes them as columns on the entity table.
        Eliminates the groupby that add_derived_dimension performs.

        The column `dimension_name` must already exist on the anchor table
        (added via add_line). Builder computes edge_max and creates
        the RelationSpec — no groupby needed.

        Args:
            anchor_line: Anchor line that has the column.
            dimension_name: Column name on the entity table.
            edge_max: int = fixed cap, "auto" = p{percentile} of column values.
            percentile: Percentile for auto edge_max (default 99.0).
            display_name: Label in dim_labels (defaults to dimension_name).
        """
        from hypertopos.builder.derived import PrecomputedDimSpec

        self._precomputed_dims.append(PrecomputedDimSpec(
            anchor_line=anchor_line,
            dimension_name=dimension_name,
            edge_max=edge_max,
            percentile=percentile,
            display_name=display_name,
        ))
        return self

    def add_graph_features(
        self,
        anchor_line: str,
        event_line: str,
        from_col: str,
        to_col: str,
        features: list[str] | None = None,
    ) -> GDSBuilder:
        """Auto-compute graph structural features from event data.

        Supported features: "in_degree", "out_degree", "reciprocity", "counterpart_overlap".
        Each feature becomes a continuous dimension on the anchor pattern.

        Args:
            anchor_line: Target anchor line.
            event_line: Source event line with from/to columns.
            from_col: Column name for source entity FK.
            to_col: Column name for destination entity FK.
            features: List of features to compute (default: all four).
        """
        from hypertopos.builder.derived import GraphFeaturesSpec

        self._graph_features.append(GraphFeaturesSpec(
            anchor_line=anchor_line,
            event_line=event_line,
            from_col=from_col,
            to_col=to_col,
            features=features or ["in_degree", "out_degree", "reciprocity", "counterpart_overlap"],
        ))
        return self

    def add_chain_line(
        self,
        line_id: str,
        chains: list[dict],
        features: list[str] | None = None,
    ) -> GDSBuilder:
        """Create an anchor line from extracted chains.

        Converts chain dicts (output of engine.chains.extract_chains) into
        an anchor line with chain features as columns. Each feature becomes
        a continuous dimension via auto-created RelationSpec.

        Args:
            line_id: Name for the chain line.
            chains: List of chain dicts (from Chain.to_dict()).
            features: Which chain features become dimensions.
                Default: hop_count, is_cyclic, n_distinct_categories,
                amount_decay, cross_bank_count, amount_monotone_decreasing.
                The latter two are AML-oriented (jurisdictional layering
                + structuring pattern) and default to 0 / False on
                event lines without bank data — additive, no breakage.
        """
        if features is None:
            features = [
                "hop_count", "is_cyclic", "n_distinct_categories",
                "amount_decay", "cross_bank_count",
                "amount_monotone_decreasing",
            ]

        if not chains:
            # Empty chains → empty line
            cols: dict[str, list] = {"primary_key": []}
            for f in features:
                cols[f] = []
            self.add_line(line_id, pa.table(cols), key_col="primary_key", source_id=line_id, role="anchor")
            self._chain_lines.add(line_id)
            return self

        # Validate chain dict structure
        required_keys = {"chain_id"}
        for f in features:
            required_keys.add(f)
        first = chains[0]
        missing = required_keys - set(first.keys())
        if missing:
            raise ValueError(
                f"add_chain_line: chain dicts missing keys: {missing}. "
                f"Required: {required_keys}. Got: {set(first.keys())}"
            )

        # Build table columns AND collect arrays for edge_max in one pass
        cols: dict[str, list] = {"primary_key": [c["chain_id"] for c in chains]}
        feature_arrays: dict[str, np.ndarray] = {}

        for f in features:
            vals = []
            for c in chains:
                if f == "is_cyclic":
                    vals.append(1.0 if c.get(f, False) else 0.0)
                else:
                    vals.append(float(c.get(f, 0.0)))
            cols[f] = vals
            feature_arrays[f] = np.array(vals)

        # Store chain keys as property for navigation
        if "keys" in chains[0]:
            cols["chain_keys"] = [",".join(c["keys"]) for c in chains]
        if "event_keys" in chains[0]:
            cols["chain_events"] = [",".join(c["event_keys"]) for c in chains]

        table = pa.table(cols)
        self.add_line(line_id, table, key_col="primary_key", source_id=line_id, role="anchor")
        self._chain_lines.add(line_id)

        # Auto-create derived dims using pre-computed arrays (no second loop over chains)
        for f in features:
            vals = feature_arrays[f]
            nonzero = vals[vals > 0]
            em = max(1, int(np.percentile(nonzero, 99))) if len(nonzero) > 0 else 1

            dim_line_id = f"_d_chain_{f}"
            if dim_line_id not in self._lines:
                self.add_line(dim_line_id, pa.table({"primary_key": ["_dummy"]}),
                              key_col="primary_key", source_id=dim_line_id, role="anchor")

            # Store in _chain_dims for pattern resolution
            self._chain_dims.append((line_id, f, em))

        return self

    def add_alias(
        self,
        alias_id: str,
        base_pattern_id: str,
        *,
        cutting_plane_normal: list[float] | None = None,
        cutting_plane_bias: float | None = None,
        cutting_plane_dimension: int | str | None = None,
        cutting_plane_threshold: float | None = None,
        description: str | None = None,
    ) -> GDSBuilder:
        """Register an alias with a cutting plane for sub-population stats.

        Two specification modes:
        - Explicit: cutting_plane_normal + cutting_plane_bias
        - Sugar: cutting_plane_dimension + cutting_plane_threshold
        """
        if base_pattern_id not in self._patterns:
            raise ValueError(
                f"Alias '{alias_id}': base_pattern_id '{base_pattern_id}' "
                f"not registered. Available: {list(self._patterns)}"
            )
        has_normal = cutting_plane_normal is not None
        has_dim = cutting_plane_dimension is not None
        if not has_normal and not has_dim:
            raise ValueError(
                f"Alias '{alias_id}': must specify either "
                "cutting_plane_normal+bias or cutting_plane_dimension+threshold"
            )
        self._aliases[alias_id] = _AliasReg(
            alias_id=alias_id,
            base_pattern_id=base_pattern_id,
            cutting_plane_normal=cutting_plane_normal,
            cutting_plane_bias=(
                cutting_plane_bias if cutting_plane_bias is not None else 0.0
            ),
            cutting_plane_dimension=cutting_plane_dimension,
            cutting_plane_threshold=(
                cutting_plane_threshold
                if cutting_plane_threshold is not None else 0.0
            ),
            description=description,
        )
        return self

    def _compute_population_stats(
        self, pat: _PatternReg,
    ) -> PopulationStats:
        """Compute shape vectors and population statistics for a pattern."""
        from hypertopos.builder._stats import compute_conformal_p, compute_stats

        entity_line = self._lines[pat.entity_line]
        entity_table = entity_line.table
        n = len(entity_table)
        D = len(pat.relations)

        # 1. Build shape_vectors (N, D)
        shape_vectors = np.zeros((n, D), dtype=np.float32)
        fk_arrays: list[pa.ChunkedArray | np.ndarray | None] = []

        for j, rel in enumerate(pat.relations):
            if rel.direction == "self":
                shape_vectors[:, j] = 1.0
                fk_arrays.append(None)
            elif rel.edge_max is not None:
                col = entity_table[rel.fk_col]
                count_arr = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False
                ).astype(np.float32)
                shape_vectors[:, j] = np.clip(count_arr, 0, rel.edge_max) / rel.edge_max
                fk_arrays.append(count_arr)
            else:
                col_arrow = entity_table[rel.fk_col]
                fk_arrays.append(col_arrow)
                valid_mask = pc.fill_null(
                    pc.and_(
                        pc.is_valid(col_arrow),
                        pc.not_equal(col_arrow, ""),
                    ),
                    False,
                )
                shape_vectors[:, j] = valid_mask.to_numpy(
                    zero_copy_only=False
                ).astype(np.float32)

        # 1a. Build edge-derived dimension values (event patterns only).
        # Computed at edge-table emission time, baked into shape vector AND
        # written to sidecar at _gds_meta/edge_features/{pid}/data.lance.
        edge_dim_matrix = np.empty((n, 0), dtype=np.float32)
        edge_dim_names: list[str] = []
        if (
            pat.edge_dimensions is not None
            and pat.pattern_type == "event"
            and getattr(pat.edge_dimensions, "dims", None)
        ):
            from hypertopos.engine.edge_features import (
                EDGE_DIM_KINDS,
                compute_all_edge_dims,
            )

            edge_cfg = pat.edge_table
            if edge_cfg is None:
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} declares edge_dimensions "
                    f"but no edge_table — edge_dimensions require an "
                    f"edge_table block on the same pattern.",
                )
            edges_tbl = self._extract_edge_table(pat, edge_cfg)

            # Resolve dormant_seconds: "auto" → sphere temporal span.
            dims_cfg = dict(pat.edge_dimensions.dims)
            tsl = dims_cfg.get("time_since_pair_last_edge")
            if (
                tsl is not None
                and isinstance(tsl, dict)
                and tsl.get("dormant_seconds") == "auto"
            ):
                if edges_tbl.num_rows > 0:
                    ts = edges_tbl["timestamp"].to_numpy()
                    span = float(ts.max() - ts.min())
                    span = max(span, 1.0)
                else:
                    span = 1.0
                dims_cfg["time_since_pair_last_edge"] = {
                    **tsl, "dormant_seconds": span,
                }

            features = compute_all_edge_dims(edges_tbl, dims_cfg)
            edge_dim_names = [
                c for c in features.column_names if c != "event_key"
            ]
            if edge_dim_names and edges_tbl.num_rows > 0:
                # Map per-event values to per-entity rows. For event patterns
                # entity.primary_key == edge.event_key 1:1.
                pk_to_idx = {
                    pk: i
                    for i, pk in enumerate(
                        entity_table["primary_key"].to_pylist(),
                    )
                }
                edge_dim_matrix = np.zeros(
                    (n, len(edge_dim_names)), dtype=np.float32,
                )
                event_keys = features["event_key"].to_pylist()
                for col_idx, name in enumerate(edge_dim_names):
                    vals = features[name].to_numpy()
                    for row_idx, ek in enumerate(event_keys):
                        ent_idx = pk_to_idx.get(ek)
                        if ent_idx is not None:
                            edge_dim_matrix[ent_idx, col_idx] = vals[row_idx]
            elif edge_dim_names:
                edge_dim_matrix = np.zeros(
                    (n, len(edge_dim_names)), dtype=np.float32,
                )

            # Persist sidecar Lance — forward-compat for the planned
            # HopPredicate.edge_dim_predicates query API.
            if edge_dim_names:
                try:
                    import lance

                    sidecar_dir = (
                        self.output_path
                        / "_gds_meta" / "edge_features" / pat.pattern_id
                    )
                    sidecar_dir.mkdir(parents=True, exist_ok=True)
                    lance.write_dataset(
                        features,
                        str(sidecar_dir / "data.lance"),
                        mode="overwrite",
                    )
                except Exception as exc:
                    logger.warning(
                        "edge_features sidecar write failed for %s: %s",
                        pat.pattern_id, exc,
                    )

        # Stash kinds for downstream concatenation step (3.2).
        if edge_dim_names:
            from hypertopos.engine.edge_features import EDGE_DIM_KINDS
            edge_dim_kinds = [EDGE_DIM_KINDS[name] for name in edge_dim_names]
        else:
            edge_dim_kinds = []

        # 1a-bis. Anchor-pattern aggregation of edge-derived dims (S1 ext, 0.6.1).
        edge_dim_agg_matrix = np.empty((n, 0), dtype=np.float32)
        edge_dim_agg_kinds: list[str] = []
        edge_dim_agg_labels: list[str] = []
        if (
            pat.edge_dim_aggregations is not None
            and pat.pattern_type == "anchor"
        ):
            from hypertopos.engine.edge_features import (
                AGGREGATE_NAMES,
                EDGE_DIM_KINDS,
                aggregate_edge_dims_for_anchor,
                aggregate_kind,
            )

            cfg = pat.edge_dim_aggregations
            src_pat = self._patterns.get(cfg.from_event_pattern)
            if src_pat is None or src_pat.pattern_type != "event":
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} edge_dim_aggregations.from "
                    f"={cfg.from_event_pattern!r} must reference an event "
                    f"pattern in this build",
                )
            if src_pat.edge_table is None:
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} edge_dim_aggregations.from "
                    f"={cfg.from_event_pattern!r} has no edge_table — "
                    f"declare edge_table on that event pattern",
                )
            sidecar_path = (
                self.output_path / "_gds_meta" / "edge_features"
                / cfg.from_event_pattern / "data.lance"
            )
            if not sidecar_path.exists():
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} edge_dim_aggregations expects "
                    f"sidecar at {sidecar_path}; declare edge_dimensions: on "
                    f"{cfg.from_event_pattern!r} and order it BEFORE this "
                    f"anchor pattern in YAML so it is built first",
                )
            import lance
            sidecar_tbl = lance.dataset(str(sidecar_path)).to_table()
            avail = [c for c in sidecar_tbl.column_names if c != "event_key"]
            agg_dims = list(cfg.dims) if cfg.dims is not None else avail
            src_edges = self._extract_edge_table(src_pat, src_pat.edge_table)

            src_lines = {r.line_id for r in src_pat.relations}
            composite_match = next(
                (cs for cs in self._composite_lines
                 if cs.line_id == pat.entity_line),
                None,
            )
            chain_events_col: list[str] | None = None
            composite_key_cols: list[str] | None = None
            if pat.entity_line in src_lines:
                anchor_kind = "single"
                pair_separator = "→"
            elif composite_match is not None:
                anchor_kind = "pair"
                pair_separator = composite_match.separator
                composite_key_cols = list(composite_match.key_cols)
                # Convention: first two key_cols positionally map to the
                # source event_pattern's edge_table endpoints (renamed in
                # _extract_edge_table to from_key / to_key). Reject k>=3
                # composite anchors that violate this — silent mismatch
                # would produce all-zero aggregates because the anchor PK
                # built from event_table.{key_cols[0],key_cols[1],...} would
                # not match the engine's PK constructed from
                # edges.{from_key, to_key, key_cols[2:]}.
                if len(composite_key_cols) > 2 and src_pat.edge_table is not None:
                    expected_from = src_pat.edge_table.from_col
                    expected_to = src_pat.edge_table.to_col
                    if (
                        composite_key_cols[0] != expected_from
                        or composite_key_cols[1] != expected_to
                    ):
                        raise ValueError(
                            f"Pattern {pat.pattern_id!r}: composite_line "
                            f"{composite_match.line_id!r} declares "
                            f"key_cols={composite_key_cols!r} but "
                            f"edge_dim_aggregations on a k>=3 composite "
                            f"anchor requires key_cols[0:2] to positionally "
                            f"match the source event pattern "
                            f"{cfg.from_event_pattern!r} edge_table endpoints "
                            f"(from_col={expected_from!r}, "
                            f"to_col={expected_to!r}). Property columns "
                            f"key_cols[2:] can be any event_table column.",
                        )
            elif pat.entity_line in self._chain_lines:
                # Chain regime — entity_line registered as chain via chain_lines: block.
                # event_line consistency was validated at parse time (cli/schema.py).
                # Zero-chain extraction is caught earlier at _validate() with a
                # chain-specific message; by the time we reach the dispatch the
                # entity_table is guaranteed to have rows AND a chain_events column
                # populated by `add_chain_line`.
                anchor_kind = "chain"
                pair_separator = "→"  # unused in chain regime
                chain_events_col = entity_table["chain_events"].to_pylist()
            else:
                raise NotImplementedError(
                    f"Pattern {pat.pattern_id!r}: edge_dim_aggregations "
                    f"could not resolve anchor regime — entity_line "
                    f"{pat.entity_line!r} is neither a relation of the "
                    f"source event pattern {cfg.from_event_pattern!r} "
                    f"(single-key regime), nor a registered composite_line "
                    f"(pair / k>2 regime), nor a registered chain_line. "
                    f"Supported regimes: single, pair, chain.",
                )

            primary_keys = entity_table["primary_key"].to_pylist()
            edge_dim_thresholds_resolved = (
                self._resolve_and_persist_edge_dim_thresholds(
                    pat.pattern_id, sidecar_tbl, agg_dims,
                )
            )
            aggregates_per_dim = cfg.aggregates_per_dim
            extra = aggregate_edge_dims_for_anchor(
                anchor_keys=primary_keys,
                edges=src_edges,
                sidecar=sidecar_tbl,
                dims=agg_dims,
                anchor_kind=anchor_kind,
                pair_separator=pair_separator,
                chain_events=chain_events_col,
                key_cols=composite_key_cols,
                event_table=self._lines[src_pat.entity_line].table,
                thresholds=edge_dim_thresholds_resolved,
                aggregates_per_dim=aggregates_per_dim,
            )
            n_cols = sum(len(aggregates_per_dim[d]) for d in agg_dims)
            edge_dim_agg_matrix = np.zeros((n, n_cols), dtype=np.float32)
            col_idx = 0
            for d in agg_dims:
                src_kind = EDGE_DIM_KINDS[d]
                for agg in aggregates_per_dim[d]:
                    edge_dim_agg_matrix[:, col_idx] = (
                        extra[f"{d}_{agg}"].to_numpy()
                    )
                    edge_dim_agg_kinds.append(aggregate_kind(src_kind, agg))
                    edge_dim_agg_labels.append(f"{d}_{agg}")
                    col_idx += 1

        # 1b. Build event dimension values
        event_dim_matrix = np.empty((n, 0), dtype=np.float32)
        if pat.event_dimensions:
            event_dim_matrix = np.zeros(
                (n, len(pat.event_dimensions)), dtype=np.float32
            )
            for k, edim in enumerate(pat.event_dimensions):
                col = entity_table[edim.column]
                raw_arr = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False
                ).astype(np.float32)
                em = edim.edge_max
                if em is None or em == "auto":
                    positive = raw_arr[raw_arr > 0]
                    computed = (
                        float(np.percentile(positive, edim.percentile))
                        if len(positive) > 0 else 1.0
                    )
                    em = max(computed, 1e-9)
                    edim.edge_max = em  # store computed value for sphere.json
                event_dim_matrix[:, k] = np.clip(
                    raw_arr / em, 0.0, 3.0
                )

        # 2. Property fill calibration (skip for event patterns)
        prop_columns: list[str] = []
        excluded_properties: list[str] = []
        prop_fill_matrix = np.empty((n, 0), dtype=np.float32)

        tracked = pat.tracked_properties if pat.pattern_type == "anchor" else []
        if tracked:
            schema_names = set(entity_table.schema.names)
            candidate_fill = np.zeros((n, len(tracked)), dtype=np.float32)
            fill_rates: list[float] = []
            for j, prop in enumerate(tracked):
                if prop not in schema_names:
                    fill_rates.append(0.0)
                    continue
                col = entity_table[prop]
                fill_vec = pc.is_valid(col).to_numpy(
                    zero_copy_only=False
                ).astype(np.float32)
                candidate_fill[:, j] = fill_vec
                fill_rates.append(float(fill_vec.mean()))

            for j, prop in enumerate(tracked):
                if fill_rates[j] < MIN_FILL_RATE:
                    excluded_properties.append(prop)
                    logger.info(
                        "Excluding '%s': fill_rate=%.3f < MIN_FILL_RATE",
                        prop, fill_rates[j],
                    )
                elif fill_rates[j] >= MAX_FILL_RATE and not _is_textual_or_binary_col(
                    entity_table.schema.field(prop)
                ):
                    excluded_properties.append(prop)
                    logger.info(
                        "Excluding '%s': fill_rate=%.3f >= MAX_FILL_RATE (zero-variance)",
                        prop, fill_rates[j],
                    )
                else:
                    prop_columns.append(prop)

            if prop_columns:
                included_indices = [
                    j for j, prop in enumerate(tracked) if prop in prop_columns
                ]
                prop_fill_matrix = candidate_fill[:, included_indices]

        # 2c. Build generalized dimension blocks (g/t/s)
        from hypertopos.builder.dim_blocks import (
            normalize_geo_block,
            normalize_metric_block,
            normalize_semantic_block,
        )

        dim_block_matrices: list[np.ndarray] = []
        dim_block_names: list[str] = []
        dim_block_stats: dict[str, Any] = {}  # stored in sphere.json
        schema_names = set(entity_table.schema.names)

        # Geographic block (g)
        if pat.geo_properties:
            geo_cols = []
            for col_name in pat.geo_properties:
                if col_name not in schema_names:
                    raise ValueError(
                        f"geo_properties column '{col_name}' not found on "
                        f"entity table '{pat.entity_line}'. "
                        f"Available: {sorted(schema_names)}"
                    )
                col = entity_table[col_name]
                geo_cols.append(
                    pc.fill_null(col, 0).to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float32)
                )
            geo_raw = np.column_stack(geo_cols)
            geo_norm, geo_mu, geo_sigma = normalize_geo_block(geo_raw)
            dim_block_matrices.append(geo_norm)
            dim_block_names.extend(f"g:{c}" for c in pat.geo_properties)
            dim_block_stats["geo"] = {
                "columns": list(pat.geo_properties),
                "mu": geo_mu.tolist(),
                "sigma": geo_sigma.tolist(),
            }

        # Metric block (t)
        if pat.metric_properties:
            metric_cols = []
            for col_name in pat.metric_properties:
                if col_name not in schema_names:
                    raise ValueError(
                        f"metric_properties column '{col_name}' not found on "
                        f"entity table '{pat.entity_line}'. "
                        f"Available: {sorted(schema_names)}"
                    )
                col = entity_table[col_name]
                metric_cols.append(
                    pc.fill_null(col, 0).to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float32)
                )
            metric_raw = np.column_stack(metric_cols)
            metric_norm, metric_mu, metric_sigma = normalize_metric_block(
                metric_raw,
            )
            dim_block_matrices.append(metric_norm)
            dim_block_names.extend(f"t:{c}" for c in pat.metric_properties)
            dim_block_stats["metric"] = {
                "columns": list(pat.metric_properties),
                "mu": metric_mu.tolist(),
                "sigma": metric_sigma.tolist(),
            }

        # Semantic block (s)
        if pat.semantic_dim:
            sem_cols_list = pat.semantic_dim["columns"]
            sem_n_comp = pat.semantic_dim["n_components"]
            sem_raw_cols = []
            for col_name in sem_cols_list:
                if col_name not in schema_names:
                    raise ValueError(
                        f"semantic_dim column '{col_name}' not found on "
                        f"entity table '{pat.entity_line}'. "
                        f"Available: {sorted(schema_names)}"
                    )
                col = entity_table[col_name]
                sem_raw_cols.append(
                    pc.fill_null(col, 0).to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float32)
                )
            sem_raw = np.column_stack(sem_raw_cols)
            sem_norm, sem_mu, sem_sigma, sem_pca = normalize_semantic_block(
                sem_raw, n_components=sem_n_comp,
            )
            dim_block_matrices.append(sem_norm)
            actual_n_comp = sem_norm.shape[1]
            dim_block_names.extend(f"s:pc{k}" for k in range(actual_n_comp))
            dim_block_stats["semantic"] = {
                "columns": list(sem_cols_list),
                "n_components": actual_n_comp,
                "mu": sem_mu.tolist(),
                "sigma": sem_sigma.tolist(),
                "pca_components": sem_pca.tolist(),
            }

        # 3. Concatenate edge + event dims + prop fill + dim blocks into full shape matrix
        parts = [shape_vectors]
        if event_dim_matrix.shape[1] > 0:
            parts.append(event_dim_matrix)
        if edge_dim_matrix.shape[1] > 0:
            parts.append(edge_dim_matrix)
        if edge_dim_agg_matrix.shape[1] > 0:
            parts.append(edge_dim_agg_matrix)
        if prop_fill_matrix.shape[1] > 0:
            parts.append(prop_fill_matrix)
        for blk in dim_block_matrices:
            parts.append(blk)
        full_shape_vectors = (
            np.concatenate(parts, axis=1) if len(parts) > 1
            else shape_vectors
        )

        # 3.2 Auto-detect Bregman dimension kinds
        from hypertopos.builder._bregman import (
            detect_kinds_for_pattern,
            format_kinds_summary,
        )

        # Gather dimension kinds — one per dimension in delta vector order:
        # [relations (incl. _d_* derived/precomputed)] → [event_dims] → [prop_fill] → [dim_blocks]
        from hypertopos.builder._bregman import (
            detect_kind_for_column,
        )

        # Build lookup: dimension_name → metric for derived dims
        _derived_metric_map: dict[str, str] = {}
        for dspec in self._derived_dims:
            _derived_metric_map[dspec.dimension_name] = dspec.metric

        # Build lookup: dimension_name → edge_max for precomputed dims
        _precomputed_map: dict[str, int | float | str | None] = {}
        for pspec in self._precomputed_dims:
            _precomputed_map[pspec.dimension_name] = pspec.edge_max

        _POISSON_METRICS = {"count", "count_distinct"}

        dimension_kinds: list[str] = []
        for rel in pat.relations:
            dim_name = rel.fk_col
            if dim_name in _derived_metric_map:
                # Derived dim — classify by metric
                metric = _derived_metric_map[dim_name]
                base_metric = metric.split(":")[0]
                dimension_kinds.append("poisson" if base_metric in _POISSON_METRICS else "gaussian")
            elif dim_name in _precomputed_map:
                # Precomputed dim — bernoulli if edge_max=1 (ratio), else gaussian
                em = _precomputed_map[dim_name]
                dimension_kinds.append(
                    "bernoulli" if isinstance(em, (int, float)) and int(em) == 1 else "gaussian"
                )
            else:
                # Regular FK or graph feature relation.
                # edge_max=1 means a 0-1 ratio (bernoulli), edge_max>1 or
                # auto-resolved means count-like (poisson), None means binary.
                em = rel.edge_max
                if isinstance(em, (int, float)) and int(em) == 1:
                    dimension_kinds.append("bernoulli")
                elif em is not None and em != "auto":
                    dimension_kinds.append("poisson")
                else:
                    dimension_kinds.append("bernoulli")

        # Event dimensions
        for edim in pat.event_dimensions:
            kind_override = getattr(edim, "kind", None)
            if kind_override:
                dimension_kinds.append(kind_override)
            else:
                col = entity_table[edim.column]
                vals = pc.fill_null(col, 0).to_numpy(zero_copy_only=False).astype(np.float64)
                dimension_kinds.append(detect_kind_for_column(vals))

        # Edge-derived dimensions
        dimension_kinds.extend(edge_dim_kinds)

        # Edge-derived aggregations on anchor patterns (S1 ext)
        dimension_kinds.extend(edge_dim_agg_kinds)

        # Prop fill (binary 0/1)
        dimension_kinds.extend(["bernoulli"] * len(prop_columns))

        # Dim blocks (geo/metric/semantic — all gaussian)
        _dim_block_count = sum(blk.shape[1] for blk in dim_block_matrices)
        dimension_kinds.extend(["gaussian"] * _dim_block_count)

        # Validate kind count matches shape dimension count
        D_full = full_shape_vectors.shape[1]
        if len(dimension_kinds) != D_full:
            logger.warning(
                "Bregman kind count (%d) != shape dim count (%d) for %s — "
                "falling back to uniform gaussian kinds",
                len(dimension_kinds), D_full, pat.pattern_id,
            )
            dimension_kinds = ["gaussian"] * D_full

        logger.info(
            "Bregman kinds for %s: %s (D=%d)",
            pat.pattern_id, format_kinds_summary(dimension_kinds), D_full,
        )

        # 3.5 Resolve dimension weights
        from hypertopos.builder._stats import compute_dimension_weights

        dim_weights: np.ndarray | None = None
        if pat.dimension_weights in ("auto", "kurtosis"):
            dim_weights = compute_dimension_weights(full_shape_vectors, method="kurtosis")
            logger.info(
                "Auto-computed dimension weights for %s: %s",
                pat.pattern_id, dim_weights.tolist(),
            )
        elif isinstance(pat.dimension_weights, list):
            dim_weights = np.array(pat.dimension_weights, dtype=np.float32)
            if len(dim_weights) != full_shape_vectors.shape[1]:
                raise ValueError(
                    f"dimension_weights length ({len(dim_weights)}) != "
                    f"shape dimensions ({full_shape_vectors.shape[1]})"
                )

        # 4. Compute stats + deltas
        group_stats_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] | None = None
        mah_cov_inv: np.ndarray | None = None
        heteroscedasticity_diagnostic: dict[str, dict[str, Any]] | None = None

        if pat.group_by_property:
            if pat.use_mahalanobis:
                logger.warning(
                    "use_mahalanobis is ignored when group_by_property is set for %s",
                    pat.pattern_id,
                )
            from hypertopos.builder._stats import compute_stats_grouped

            # Read group column from entity table
            schema_names = set(entity_table.schema.names)
            if pat.group_by_property not in schema_names:
                raise ValueError(
                    f"group_by_property '{pat.group_by_property}' "
                    f"not found in entity table columns: {sorted(schema_names)}"
                )
            group_col = entity_table[pat.group_by_property]
            group_ids = group_col.to_numpy(zero_copy_only=False).astype(str)

            # Global stats (for backward compat + fallback)
            mu, sigma, theta, global_deltas, global_norms, _ = compute_stats(
                full_shape_vectors, pat.anomaly_percentile,
            )

            # Per-group stats — prop_dim_start applies SIGMA_EPS_PROP inside
            # the grouped loop, eliminating a separate re-run pass
            n_rel = len(pat.relations)
            n_event = len(pat.event_dimensions) if pat.event_dimensions else 0
            _prop_start = (n_rel + n_event) if prop_columns else None
            group_stats_dict, deltas, delta_norms = compute_stats_grouped(
                full_shape_vectors, group_ids, pat.anomaly_percentile,
                prop_dim_start=_prop_start,
            )

            # Pre-compute boolean masks once — reused in all three group loops
            unique_groups = list(group_stats_dict.keys())
            group_mask_map: dict[str, np.ndarray] = {
                gid: group_ids == gid for gid in unique_groups
            }

            # Prop_columns sigma override for global stats
            if prop_columns:
                from hypertopos.builder._stats import SIGMA_EPS_PROP

                prop_start = n_rel + n_event
                sigma[prop_start:prop_start + len(prop_columns)] = np.maximum(
                    sigma[prop_start:prop_start + len(prop_columns)], SIGMA_EPS_PROP
                )
                global_deltas = ((full_shape_vectors - mu) / sigma).astype(np.float32)
                global_norms = np.sqrt(
                    np.einsum('ij,ij->i', global_deltas, global_deltas),
                ).astype(np.float32)
                theta_scalar = float(np.percentile(global_norms, pat.anomaly_percentile))
                D_full = full_shape_vectors.shape[1]
                component = theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
                theta = np.full(D_full, component, dtype=np.float32)

            # Brown-Forsythe homoscedasticity diagnostic on the global-
            # normalised delta_norms across `group_by_property` levels.
            # The diagnostic answers "is global θ statistically valid
            # for this pattern given the grouping it carries?", so the
            # array fed must be the GLOBAL-mu/sigma norms, not the
            # per-group-normalised ones (which would silently test
            # residual heteroscedasticity after calibration — a
            # different question).
            from hypertopos.engine.diagnostics import levene_test_per_group

            heteroscedasticity_diagnostic = {
                pat.group_by_property: levene_test_per_group(
                    global_norms, group_ids,
                ),
            }
        else:
            mu, sigma, theta, deltas, delta_norms, mah_cov_inv = compute_stats(
                full_shape_vectors, pat.anomaly_percentile,
                use_mahalanobis=bool(pat.use_mahalanobis),
            )

            # Prop_columns are binary (0/1) — use higher sigma floor
            n_rel = len(pat.relations)
            n_event = len(pat.event_dimensions) if pat.event_dimensions else 0
            if prop_columns:
                from hypertopos.builder._stats import SIGMA_EPS_PROP

                prop_start = n_rel + n_event
                sigma[prop_start:prop_start + len(prop_columns)] = np.maximum(
                    sigma[prop_start:prop_start + len(prop_columns)], SIGMA_EPS_PROP
                )
                deltas = ((full_shape_vectors - mu) / sigma).astype(np.float32)
                delta_norms = np.sqrt(np.einsum('ij,ij->i', deltas, deltas)).astype(np.float32)
                theta_scalar = float(
                    np.percentile(delta_norms, pat.anomaly_percentile)
                )
                D_full = full_shape_vectors.shape[1]
                component = theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
                theta = np.full(D_full, component, dtype=np.float32)

        # 4.5 Apply dimension weights to deltas (after z-scoring, before norms)
        if dim_weights is not None:
            deltas = (deltas * dim_weights).astype(np.float32)
            delta_norms = np.sqrt(np.einsum('ij,ij->i', deltas, deltas)).astype(np.float32)
            # Recompute theta in weighted space
            theta_scalar = float(np.percentile(delta_norms, pat.anomaly_percentile))
            D_full = full_shape_vectors.shape[1]
            component = theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
            theta = np.full(D_full, component, dtype=np.float32)

            # Recompute per-group theta in weighted space
            if group_stats_dict:
                for gid, (mu_g, sigma_g, _, pop_g) in group_stats_dict.items():
                    mask = group_mask_map[gid]
                    g_norms = delta_norms[mask]
                    g_theta_scalar = (
                        float(np.percentile(g_norms, pat.anomaly_percentile))
                        if len(g_norms) > 1 else 0.0
                    )
                    g_comp = g_theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
                    g_theta = np.full(D_full, g_comp, dtype=np.float32)
                    group_stats_dict[gid] = (mu_g, sigma_g, g_theta, pop_g)

        # 4.55 Compute Bregman norms (reuse dimension_kinds from step 3.2)
        bregman_norms_arr: np.ndarray | None = None
        theta_per_dim_arr: np.ndarray | None = None

        if dimension_kinds is not None:
            from hypertopos.builder._bregman import (
                bregman_norms as _bregman_norms_fn,
                per_dim_theta as _per_dim_theta_fn,
            )

            # Global Bregman — always scored against population mu/sigma.
            # Per-group calibration is already captured in is_anomaly (L2).
            # Bregman as supplementary metric uses global for consistent ranking.
            bregman_norms_arr = _bregman_norms_fn(
                full_shape_vectors, mu, sigma, dimension_kinds,
                weights=dim_weights,
            ).astype(np.float32)
            theta_per_dim_arr = _per_dim_theta_fn(
                full_shape_vectors, mu, sigma, dimension_kinds,
                anomaly_percentile=pat.anomaly_percentile,
            )

        # 4.6 GMM per-cluster theta
        gmm_components_result: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] | None = None
        if pat.gmm_n_components:
            from hypertopos.builder._stats import fit_kmeans_components

            k = pat.gmm_n_components
            gmm_components_result, gmm_assignments = fit_kmeans_components(
                full_shape_vectors, n_components=k,
                anomaly_percentile=pat.anomaly_percentile,
            )

            # Re-compute deltas per cluster
            n_rel_gmm = len(pat.relations)
            n_event_gmm = len(pat.event_dimensions) if pat.event_dimensions else 0
            prop_start_gmm = n_rel_gmm + n_event_gmm
            for c_idx, (mu_c, sigma_c, theta_c, pop_c) in enumerate(gmm_components_result):
                mask = gmm_assignments == c_idx
                if mask.sum() == 0:
                    continue
                # Apply SIGMA_EPS_PROP floor to prop dims in cluster sigma
                if prop_columns:
                    sigma_c = sigma_c.copy()
                    sigma_c[prop_start_gmm:prop_start_gmm + len(prop_columns)] = np.maximum(
                        sigma_c[prop_start_gmm:prop_start_gmm + len(prop_columns)], SIGMA_EPS_PROP
                    )
                    gmm_components_result[c_idx] = (mu_c, sigma_c, theta_c, pop_c)
                c_shapes = full_shape_vectors[mask]
                c_deltas = ((c_shapes - mu_c) / sigma_c).astype(np.float32)
                if dim_weights is not None:
                    c_deltas = (c_deltas * dim_weights).astype(np.float32)
                c_norms = np.sqrt(np.einsum('ij,ij->i', c_deltas, c_deltas)).astype(np.float32)
                deltas[mask] = c_deltas
                delta_norms[mask] = c_norms
                # Recompute theta in (possibly weighted) space
                if pop_c > 1:
                    c_theta_scalar = float(np.percentile(c_norms, pat.anomaly_percentile))
                    D_full = full_shape_vectors.shape[1]
                    c_comp = c_theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
                    c_theta = np.full(D_full, c_comp, dtype=np.float32)
                    gmm_components_result[c_idx] = (mu_c, sigma_c, c_theta, pop_c)

            # Recompute global theta from GMM-adjusted norms
            theta_scalar = float(np.percentile(delta_norms, pat.anomaly_percentile))
            D_full = full_shape_vectors.shape[1]
            component_val = theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
            theta = np.full(D_full, component_val, dtype=np.float32)

            logger.info(
                "GMM k=%d for %s — cluster sizes: %s",
                k, pat.pattern_id,
                [c[3] for c in gmm_components_result],
            )

            # Recompute Bregman norms per cluster (each entity scored against
            # its cluster's mu_c/sigma_c, not the global mu/sigma)
            if bregman_norms_arr is not None and dimension_kinds is not None:
                from hypertopos.builder._bregman import (
                    bregman_norms as _bregman_norms_fn,
                )
                bregman_norms_arr = np.zeros(n, dtype=np.float32)
                for c_idx, (c_mu, c_sigma, _c_theta, _c_pop) in enumerate(gmm_components_result):
                    mask = gmm_assignments == c_idx
                    if mask.any():
                        bregman_norms_arr[mask] = _bregman_norms_fn(
                            full_shape_vectors[mask], c_mu, c_sigma,
                            dimension_kinds, weights=dim_weights,
                        ).astype(np.float32)

        # 5. Compute delta_rank_pct (population-level percentile rank)
        sorted_norms = np.sort(delta_norms)
        ranks = np.searchsorted(sorted_norms, delta_norms, side="left")
        delta_rank_pcts = (ranks / n * 100).astype(np.float32)

        # 6. Compute conformal p-values (reuse sorted_norms from step 5)
        conformal_p = compute_conformal_p(delta_norms, sorted_norms=sorted_norms)

        # 6b. Theta sensitivity surface — calibration-quality diagnostic.
        # Glues onto the same sorted_norms (no new sort, O(P) per pattern).
        from hypertopos.builder._theta_sensitivity import (
            compute_theta_sensitivity_from_sorted,
        )
        theta_sensitivity = compute_theta_sensitivity_from_sorted(sorted_norms)

        # 7. Compute is_anomaly (per-cluster if GMM, per-group if grouped, else global)
        #    When Bregman norms are available, use sum(theta_per_dim) as the
        #    anomaly threshold instead of the L2 theta_norm of the uniform theta.
        is_anomaly_arr: np.ndarray | None = None
        if gmm_components_result is not None:
            is_anomaly_arr = np.zeros(n, dtype=bool)
            for c_idx, (_c_mu, _c_sigma, c_theta, _c_pop) in enumerate(gmm_components_result):
                c_theta_norm = float(np.linalg.norm(c_theta))
                mask = gmm_assignments == c_idx
                is_anomaly_arr[mask] = (c_theta_norm > 0.0) & (delta_norms[mask] >= c_theta_norm)
        elif group_stats_dict:
            is_anomaly_arr = np.zeros(n, dtype=bool)
            for gid, (_g_mu, _g_sigma, g_theta, _g_pop) in group_stats_dict.items():
                g_theta_norm = float(np.linalg.norm(g_theta))
                mask = group_mask_map[gid]
                is_anomaly_arr[mask] = (g_theta_norm > 0.0) & (delta_norms[mask] >= g_theta_norm)

        # 7b. Bregman-based anomaly flag — computed and stored on
        #     PopulationStats for downstream use (geometry columns, navigator),
        #     but does NOT override the delta_norms-based is_anomaly yet.
        #     Full switchover deferred until threshold calibration is validated.

        # Mahalanobis cholesky_inv (only set in non-grouped path)
        cov_inv = mah_cov_inv if not pat.group_by_property else None

        # 8. Per-dimension anomaly count
        from hypertopos.builder._stats import compute_per_dim_anomaly_count
        n_anom_dims = compute_per_dim_anomaly_count(deltas, percentile=99.0)

        return PopulationStats(
            mu=mu, sigma=sigma, theta=theta,
            deltas=deltas, delta_norms=delta_norms,
            delta_rank_pcts=delta_rank_pcts,
            conformal_p=conformal_p, fk_arrays=fk_arrays,
            prop_columns=prop_columns,
            excluded_properties=excluded_properties,
            group_stats_dict=group_stats_dict,
            is_anomaly_arr=is_anomaly_arr,
            dim_weights=dim_weights,
            gmm_components=gmm_components_result,
            cholesky_inv=cov_inv,
            n_anom_dims=n_anom_dims,
            bregman_norms=bregman_norms_arr,
            dimension_kinds=dimension_kinds,
            theta_per_dim=theta_per_dim_arr,
            dim_block_names=dim_block_names,
            dim_block_stats=dim_block_stats if dim_block_stats else None,
            theta_sensitivity=theta_sensitivity,
            edge_dim_agg_matrix=(
                edge_dim_agg_matrix
                if edge_dim_agg_matrix.shape[1] > 0 else None
            ),
            edge_dim_agg_labels=edge_dim_agg_labels,
            heteroscedasticity_diagnostic=heteroscedasticity_diagnostic,
            edge_dim_names=edge_dim_names,
        )

    def _build_geometry_slice(
        self,
        pat: _PatternReg,
        start: int,
        end: int,
        deltas: np.ndarray,
        delta_norms: np.ndarray,
        delta_rank_pcts: np.ndarray,
        theta_norm: float,
        fk_arrays: list[pa.ChunkedArray | np.ndarray | None],
        conformal_p: np.ndarray | None = None,
        is_anomaly_precomputed: np.ndarray | None = None,
        n_anom_dims: np.ndarray | None = None,
        bregman_norms_arr: np.ndarray | None = None,
        anomaly_confidence_arr: np.ndarray | None = None,
        entity_table_override: pa.Table | None = None,
    ) -> pa.Table:
        """Build a geometry Arrow table for entities [start:end).

        Uses pre-computed deltas/norms/ranks from full population stats.
        """
        entity_table = (
            entity_table_override
            if entity_table_override is not None
            else self._lines[pat.entity_line].table
        )
        chunk_table = entity_table.slice(start, end - start)
        cn = end - start
        D = len(pat.relations)
        now = datetime.now(UTC)

        chunk_deltas = deltas[start:end]
        chunk_norms = delta_norms[start:end]
        chunk_ranks = delta_rank_pcts[start:end]
        if is_anomaly_precomputed is not None:
            is_anomaly_arr = is_anomaly_precomputed[start:end]
        else:
            is_anomaly_arr = (theta_norm > 0.0) & (chunk_norms >= theta_norm)

        # Build edges for this chunk
        pk_str = chunk_table["primary_key"].cast(pa.string()).combine_chunks()
        _zeros_idx = pa.array(np.zeros(cn, dtype=np.int32))

        def _const_str(val: str) -> pa.Array:
            return pa.array([val], type=pa.string()).take(_zeros_idx)

        rel_line_ids: list[pa.Array] = []
        rel_point_keys: list[pa.Array] = []
        rel_statuses: list[pa.Array] = []
        rel_directions: list[pa.Array] = []
        alive_masks: list[np.ndarray] = []

        for j, rel in enumerate(pat.relations):
            if rel.direction == "self":
                rel_line_ids.append(_const_str(rel.line_id))
                rel_point_keys.append(pk_str)
                rel_statuses.append(_const_str("alive"))
                rel_directions.append(_const_str("self"))
                alive_masks.append(np.ones(cn, dtype=bool))
            elif rel.edge_max is not None:
                count_arr = fk_arrays[j][start:end]
                alive_np = count_arr > 0
                alive_mask_pa = pa.array(alive_np)
                status_arr = pc.if_else(alive_mask_pa, "alive", "dead")
                rel_line_ids.append(_const_str(rel.line_id))
                rel_point_keys.append(_const_str(""))
                rel_statuses.append(status_arr)
                rel_directions.append(_const_str(rel.direction))
                alive_masks.append(alive_np)
            else:
                fk_col = fk_arrays[j].slice(start, cn)
                alive_mask = pc.fill_null(
                    pc.and_(
                        pc.is_valid(fk_col), pc.not_equal(fk_col, "")
                    ),
                    False,
                )
                alive_np = alive_mask.to_numpy(zero_copy_only=False)
                status_arr = pc.if_else(alive_mask, "alive", "dead")
                point_key_arr = fk_col.combine_chunks().cast(pa.string())

                rel_line_ids.append(_const_str(rel.line_id))
                rel_point_keys.append(point_key_arr)
                rel_statuses.append(status_arr.combine_chunks())
                rel_directions.append(_const_str(rel.direction))
                alive_masks.append(alive_np)

        if D == 0:
            empty_offsets = pa.array(
                np.zeros(cn + 1, dtype=np.int32), type=pa.int32()
            )
            empty_structs = pa.StructArray.from_arrays(
                [
                    pa.array([], type=pa.string()),
                    pa.array([], type=pa.string()),
                    pa.array([], type=pa.string()),
                    pa.array([], type=pa.string()),
                ],
                fields=[
                    pa.field("line_id", pa.string()),
                    pa.field("point_key", pa.string()),
                    pa.field("status", pa.string()),
                    pa.field("direction", pa.string()),
                ],
            )
            edges_col = pa.ListArray.from_arrays(empty_offsets, empty_structs)
            entity_keys_col = pa.array(
                [[] for _ in range(cn)], type=pa.list_(pa.string())
            )
        else:
            interleave_idx = pa.array(
                np.arange(cn * D, dtype=np.int32).reshape(D, cn).T.ravel()
            )

            def _interleave_arrays(arrays: list[pa.Array]) -> pa.Array:
                return pa.concat_arrays(arrays).take(interleave_idx)

            flat_line_ids = _interleave_arrays(rel_line_ids)
            flat_point_keys = _interleave_arrays(rel_point_keys)
            flat_statuses = _interleave_arrays(rel_statuses)
            flat_directions = _interleave_arrays(rel_directions)

            flat_structs = pa.StructArray.from_arrays(
                [flat_line_ids, flat_point_keys,
                 flat_statuses, flat_directions],
                fields=[
                    pa.field("line_id", pa.string()),
                    pa.field("point_key", pa.string()),
                    pa.field("status", pa.string()),
                    pa.field("direction", pa.string()),
                ],
            )
            offsets = pa.array(
                np.arange(0, cn * D + 1, D, dtype=np.int32),
                type=pa.int32(),
            )
            edges_col = pa.ListArray.from_arrays(offsets, flat_structs)

            # Entity keys: positional list — entity_keys[j] corresponds to
            # relations[j]. Dead edge = empty string "", alive = point_key.
            # This enables edge reconstruction from entity_keys + relations.
            positional_offsets = pa.array(
                np.arange(0, cn * D + 1, D, dtype=np.int32),
                type=pa.int32(),
            )
            # Replace nulls with "" so dead edges are represented as empty strings
            flat_point_keys_clean = pc.fill_null(flat_point_keys, "")
            entity_keys_col = pa.ListArray.from_arrays(
                positional_offsets,
                flat_point_keys_clean,
            )

        ts_type = pa.timestamp("us", tz="UTC")
        now_arr = pa.array([now], type=ts_type).take(_zeros_idx)

        d = chunk_deltas.shape[1]
        if d > 0:
            delta_col = pa.FixedSizeListArray.from_arrays(
                pa.array(chunk_deltas.ravel(), type=pa.float32()),
                list_size=d,
            )
            delta_col = delta_col.cast(pa.list_(pa.float32()))
        else:
            delta_col = pa.array(
                [[] for _ in range(cn)], type=pa.list_(pa.float32())
            )

        _zeros_i32 = pa.array(np.zeros(cn, dtype=np.int32))

        def _const_i32(val: int) -> pa.Array:
            return pa.array([val], type=pa.int32()).take(_zeros_i32)

        chunk_conformal = (
            pa.array(conformal_p[start:end], type=pa.float32())
            if conformal_p is not None
            else pa.array(np.full(cn, 0.05, dtype=np.float32), type=pa.float32())
        )

        # Label-aware signed delta projection. The registered direction
        # vector is unit-norm with the same dim count as ``chunk_deltas``;
        # ``chunk_deltas @ direction`` yields one signed scalar per
        # polygon — positive = pushed toward the positive-labelled
        # centroid, negative = toward negative. Patterns without a
        # registered direction get a full-null column.
        direction_vec = self._label_aware_directions.get(pat.pattern_id)
        if direction_vec is not None and chunk_deltas.shape[1] == direction_vec.shape[0]:
            signed_norms = chunk_deltas.astype(np.float32) @ direction_vec.astype(np.float32)
            chunk_signed_arr = pa.array(signed_norms, type=pa.float32())
        else:
            chunk_signed_arr = pa.array([None] * cn, type=pa.float32())

        common_cols = {
            "primary_key":     pk_str,
            "scale":           _const_i32(1),
            "delta":           delta_col,
            "delta_norm":      pa.array(chunk_norms, type=pa.float32()),
            "delta_rank_pct":  pa.array(chunk_ranks, type=pa.float32()),
            "is_anomaly":      pa.array(is_anomaly_arr, type=pa.bool_()),
            "conformal_p":     chunk_conformal,
            "bregman_divergence": (
                pa.array(bregman_norms_arr[start:end], type=pa.float32())
                if bregman_norms_arr is not None
                else pa.array(np.zeros(cn, dtype=np.float32), type=pa.float32())
            ),
            "anomaly_confidence": (
                pa.array(anomaly_confidence_arr[start:end], type=pa.float32())
                if anomaly_confidence_arr is not None
                else pa.array([None] * cn, type=pa.float32())
            ),
            "n_anomalous_dims": (
                pa.array(n_anom_dims[start:end], type=pa.int32())
                if n_anom_dims is not None
                else pa.array(np.zeros(cn, dtype=np.int32), type=pa.int32())
            ),
            "delta_norm_signed": chunk_signed_arr,
        }

        if pat.pattern_type == "event":
            # Event patterns: skip edges — reconstruct from entity_keys + relations at read time
            return pa.table({
                **common_cols,
                "entity_keys":     entity_keys_col,
                "last_refresh_at": now_arr,
                "updated_at":      now_arr,
            }, schema=GEOMETRY_EVENT_SCHEMA)
        else:
            # Anchor patterns: keep edges (small population, needed for display)
            return pa.table({
                **common_cols,
                "edges":           edges_col,
                "entity_keys":     entity_keys_col,
                "last_refresh_at": now_arr,
                "updated_at":      now_arr,
            }, schema=GEOMETRY_SCHEMA)

    def _build_geometry_table(
        self, pat: _PatternReg,
    ) -> tuple[pa.Table, PopulationStats, dict[str, Any] | None]:
        """Build geometry Arrow table for a pattern (single-pass, in-memory).

        Returns:
            (geometry_table, population_stats, label_aware_calibration).
            ``label_aware_calibration`` is the ``{dim_label: DimCalibration}``
            mapping when the pattern opted into label-aware calibration
            and the LDA fit succeeded, ``None`` otherwise. Populating it
            here so the result reaches sphere.json via PatternBuildResult.
        """
        n = len(self._lines[pat.entity_line].table)
        ps = self._compute_population_stats(pat)
        # Run label-aware calibration BEFORE _build_geometry_slice so the
        # Fisher LDA direction is registered in _label_aware_directions
        # and the slice can project deltas to delta_norm_signed.
        lac = self._run_label_aware_calibration(pat, ps)

        # Bootstrap confidence — requires full shape vectors in memory.
        # Skip when: use_mahalanobis, population > _BOOTSTRAP_MAX_N
        # (conformal_p sufficient), or bootstrap_iterations=0.
        confidence_arr: np.ndarray | None = None
        n = len(ps.deltas)
        if (
            pat.bootstrap_iterations > 0
            and ps.dimension_kinds is not None
            and not pat.use_mahalanobis
            and not pat.group_by_property
            and n <= _BOOTSTRAP_MAX_N
        ):
            raw_deltas = ps.deltas
            if ps.dim_weights is not None:
                raw_deltas = (raw_deltas / ps.dim_weights).astype(np.float32)
            shape_vectors = (raw_deltas * ps.sigma + ps.mu).astype(np.float32)

            from hypertopos.builder._bootstrap import (
                compute_bootstrap_confidence,
            )

            confidence_arr = compute_bootstrap_confidence(
                shape_vectors=shape_vectors,
                kinds=ps.dimension_kinds,
                anomaly_percentile=pat.anomaly_percentile,
                B=pat.bootstrap_iterations,
                weights=ps.dim_weights,
                seed=42,
            )
        elif n > _BOOTSTRAP_MAX_N and pat.bootstrap_iterations > 0:
            logger.info(
                "Bootstrap skipped for '%s' (n=%d > %d) — use conformal_p",
                pat.pattern_id, n, _BOOTSTRAP_MAX_N,
            )

        theta_norm = float(np.linalg.norm(ps.theta))
        table = self._build_geometry_slice(
            pat, 0, n, ps.deltas, ps.delta_norms,
            ps.delta_rank_pcts, theta_norm, ps.fk_arrays,
            ps.conformal_p, ps.is_anomaly_arr, ps.n_anom_dims,
            bregman_norms_arr=ps.bregman_norms,
            anomaly_confidence_arr=confidence_arr,
        )
        return table, ps, lac

    # ── Edge table helpers ─────────────────────────────────────

    def _resolve_edge_table_config(
        self, pat: _PatternReg,
    ) -> EdgeTableConfig | None:
        """Determine edge table config for a pattern.

        Priority:
        1. Check _no_edges flag (CLI --no-edges)
        2. Explicit pat.edge_table
        3. Auto-detect from graph_features (same event_line)
        4. Infer from relations (2 FKs to same anchor line)
        Returns None if pattern doesn't have from/to structure.
        """
        if self._no_edges:
            return None
        if pat.edge_table is not None:
            return pat.edge_table

        # Auto-detect from graph_features
        for gf in self._graph_features:
            if gf.event_line == pat.entity_line:
                ts_col, amt_col = self._infer_edge_temporal_amount(
                    pat.entity_line,
                )
                return EdgeTableConfig(
                    from_col=gf.from_col,
                    to_col=gf.to_col,
                    timestamp_col=ts_col,
                    amount_col=amt_col,
                )

        # Infer from relations: 2+ FK relations to the same anchor line
        fk_rels = [
            r for r in pat.relations
            if r.fk_col and r.direction != "self"
        ]
        by_line: dict[str, list[RelationSpec]] = {}
        for r in fk_rels:
            by_line.setdefault(r.line_id, []).append(r)
        for line_id, rels in by_line.items():
            if len(rels) >= 2:
                ts_col, amt_col = self._infer_edge_temporal_amount(
                    pat.entity_line,
                )
                return EdgeTableConfig(
                    from_col=rels[0].fk_col,
                    to_col=rels[1].fk_col,
                    timestamp_col=ts_col,
                    amount_col=amt_col,
                )

        return None

    def _infer_edge_temporal_amount(
        self, entity_line_id: str,
    ) -> tuple[str | None, str | None]:
        """Heuristic: pick a timestamp + amount column from event line schema.

        Used when edge_table config is auto-detected (no explicit YAML).
        Returns (timestamp_col, amount_col), either may be None.
        """
        line_reg = self._lines.get(entity_line_id)
        if line_reg is None:
            return None, None
        schema_names = set(line_reg.table.schema.names)

        ts_candidates = (
            "timestamp", "ts", "event_time", "tx_date", "date",
        )
        amt_candidates = (
            "amount_received", "amount", "amount_paid", "value",
            "total", "amt", "fare_amount", "total_amount",
        )

        ts_col = next(
            (c for c in ts_candidates if c in schema_names), None,
        )
        # Type-based fallback: first non-metadata column with timestamp type
        _META_COLS = {"created_at", "changed_at", "version"}
        if ts_col is None:
            import pyarrow as pa
            for field in line_reg.table.schema:
                if pa.types.is_timestamp(field.type) and field.name not in _META_COLS:
                    ts_col = field.name
                    break
        amt_col = next(
            (c for c in amt_candidates if c in schema_names), None,
        )
        return ts_col, amt_col

    def _extract_edge_table(
        self,
        pat: _PatternReg,
        cfg: EdgeTableConfig,
    ) -> pa.Table:
        """Build edge Arrow table from the event line's source data."""
        import pyarrow.compute as pc

        from hypertopos.storage._schemas import EDGE_TABLE_SCHEMA

        event_table = self._lines[pat.entity_line].table
        schema_names = set(event_table.schema.names)

        if cfg.from_col not in schema_names or cfg.to_col not in schema_names:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in EDGE_TABLE_SCHEMA},
            )

        from_arr = event_table[cfg.from_col]
        to_arr = event_table[cfg.to_col]
        event_key_arr = event_table["primary_key"]

        # Timestamp — resolve from config, then name heuristic, then type fallback
        ts_arr: pa.Array
        ts_col = cfg.timestamp_col
        if ts_col and ts_col in schema_names:
            ts_arr = self._to_epoch_seconds(event_table[ts_col])
        else:
            for name in ("timestamp", "ts", "date", "created_at", "tx_date"):
                if name in schema_names:
                    ts_arr = self._to_epoch_seconds(event_table[name])
                    break
            else:
                # Type-based fallback: first non-metadata column with timestamp type
                _META = {"created_at", "changed_at", "version"}
                ts_resolved = False
                for field in event_table.schema:
                    if pa.types.is_timestamp(field.type) and field.name not in _META:
                        ts_arr = self._to_epoch_seconds(event_table[field.name])
                        ts_resolved = True
                        break
                if not ts_resolved:
                    ts_arr = pa.array(
                        [0.0] * len(event_table), type=pa.float64(),
                    )

        # Amount — resolve from config, then name heuristic
        amt_arr: pa.Array
        amt_col = cfg.amount_col
        if amt_col and amt_col in schema_names:
            amt_arr = pc.cast(
                pc.fill_null(event_table[amt_col], 0.0), pa.float64(),
            )
        else:
            for name in (
                "amount", "value", "total", "amt",
                "fare_amount", "total_amount", "amount_received", "amount_paid",
            ):
                if name in schema_names:
                    amt_arr = pc.cast(
                        pc.fill_null(event_table[name], 0.0), pa.float64(),
                    )
                    break
            else:
                amt_arr = pa.array(
                    [0.0] * len(event_table), type=pa.float64(),
                )

        # Filter out rows with null from/to keys
        valid = pc.and_(pc.is_valid(from_arr), pc.is_valid(to_arr))

        return pa.table(
            {
                "from_key": pc.filter(pc.cast(from_arr, pa.string()), valid),
                "to_key": pc.filter(pc.cast(to_arr, pa.string()), valid),
                "event_key": pc.filter(pc.cast(event_key_arr, pa.string()), valid),
                "timestamp": pc.filter(ts_arr, valid),
                "amount": pc.filter(amt_arr, valid),
            },
            schema=EDGE_TABLE_SCHEMA,
        )

    def _inject_fdr_hierarchy_carriers(
        self,
        pat: _PatternReg,
        geometry_table: pa.Table,
    ) -> pa.Table:
        """For an anchor pattern with ``fdr_hierarchy``, copy each
        ``from_dimension`` that lives only on the anchor line onto
        ``geometry_table`` so the validation gate and downstream
        find_anomalies aggregation can see it. No-op for event patterns and
        for patterns without ``fdr_hierarchy``.

        Called by all three geometry-write paths BEFORE
        ``_inject_fdr_temporal_buckets`` and BEFORE
        ``_validate_fdr_hierarchy_columns``.
        """
        if not pat.fdr_hierarchy:
            return geometry_table
        if pat.pattern_type != "anchor":
            return geometry_table
        line_reg = self._lines.get(pat.entity_line)
        if line_reg is None:
            raise ValueError(
                f"Pattern {pat.pattern_id!r}: fdr_hierarchy declared on anchor "
                f"with entity_line {pat.entity_line!r} which is not registered "
                f"on this builder.",
            )
        return _inject_fdr_hierarchy_columns(
            pat,
            geometry_table=geometry_table,
            anchor_table=line_reg.table,
        )

    def _inject_fdr_temporal_buckets(
        self,
        pat: _PatternReg,
        geometry_table: pa.Table,
    ) -> pa.Table:
        """For an anchor pattern that declares ``fdr_temporal_hierarchy``,
        materialise missing slice_dimension columns onto ``geometry_table``
        via the event pattern auto-discovered from this builder's pattern
        registry. No-op for patterns without ``fdr_temporal_hierarchy``.

        Called by all three geometry-write paths so the materialised buckets
        land before write_geometry / write_chunk_fn.
        """
        if not pat.fdr_temporal_hierarchy:
            return geometry_table
        if pat.pattern_type != "anchor":
            return geometry_table
        event_pat = _auto_discover_event_pattern_for_anchor(
            pat, self._patterns,
        )
        event_reg = self._lines.get(event_pat.entity_line)
        if event_reg is None:
            raise ValueError(
                f"Pattern {pat.pattern_id!r}: auto-discovered event pattern "
                f"{event_pat.pattern_id!r} references entity_line "
                f"{event_pat.entity_line!r} which is not registered on this "
                f"builder.",
            )
        return _maybe_materialise_temporal_buckets(
            pat,
            geometry_table=geometry_table,
            event_table=event_reg.table,
            anchor_key_col_options=(
                event_pat.edge_table.from_col,
                event_pat.edge_table.to_col,
            ),
            timestamp_col=event_pat.edge_table.timestamp_col,
        )

    @staticmethod
    def _to_epoch_seconds(col: pa.Array) -> pa.Array:
        """Convert Arrow column to float64 epoch seconds."""
        import pyarrow.compute as pc

        if pa.types.is_timestamp(col.type):
            divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
            d = divisors.get(col.type.unit, 1e6)
            epoch = pc.cast(col, pa.int64())
            return pc.divide(pc.cast(epoch, pa.float64()), d)
        if pa.types.is_floating(col.type):
            return pc.cast(col, pa.float64())
        if pa.types.is_integer(col.type):
            return pc.cast(col, pa.float64())
        # String: try common formats on sample, then parse full column
        _FORMATS = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
        ]
        sample = col.slice(0, 1)
        for fmt in _FORMATS:
            try:
                pc.strptime(sample, fmt, "us")
                # Format matched on sample — parse full column
                parsed = pc.strptime(col, fmt, "us")
                try:
                    parsed = pc.assume_timezone(parsed, timezone="UTC")
                except Exception:
                    pass  # tz database missing (Windows) — treat as UTC
                return pc.divide(pc.cast(pc.cast(parsed, pa.int64()), pa.float64()), 1e6)
            except Exception:
                continue
        # Fallback: zeros
        return pa.array([0.0] * len(col), type=pa.float64())

    def _build_aliases(
        self,
        pattern_stats: dict[str, PatternBuildResult],
    ) -> dict[str, Any]:
        """Compute sub-population stats for each registered alias.

        Reads geometry from Lance, filters by cutting plane, and computes
        mu/sigma/theta for the sub-population.
        """
        from hypertopos.model.sphere import CuttingPlane

        result: dict[str, Any] = {}
        now_str = datetime.now(UTC).isoformat()
        _delta_cache: dict[str, np.ndarray] = {}

        for alias_id, areg in self._aliases.items():
            pat = self._patterns[areg.base_pattern_id]
            pbr = pattern_stats[areg.base_pattern_id]
            D = len(pat.relations) + len(pat.event_dimensions) + len(pbr.prop_columns)

            # Resolve cutting plane
            if areg.cutting_plane_normal is not None:
                normal = list(areg.cutting_plane_normal)
                bias = areg.cutting_plane_bias
            else:
                dim_idx = areg.cutting_plane_dimension
                if isinstance(dim_idx, str):
                    names: list[str] = [r.line_id for r in pat.relations]
                    names += [e.column for e in pat.event_dimensions]
                    names += pbr.prop_columns
                    if dim_idx not in names:
                        raise ValueError(
                            f"Alias '{alias_id}': dimension '{dim_idx}' "
                            f"not found. Available: {names}"
                        )
                    dim_idx = names.index(dim_idx)
                normal = [0.0] * D
                normal[dim_idx] = 1.0
                bias = areg.cutting_plane_threshold

            cp = CuttingPlane(normal=normal, bias=bias)

            # Read geometry deltas from Lance (cached per base pattern)
            import lance

            if areg.base_pattern_id not in _delta_cache:
                geo_path = (
                    self.output_path / "geometry" / areg.base_pattern_id
                    / "data.lance"
                )
                ds = lance.dataset(str(geo_path))
                delta_col = ds.to_table(columns=["delta"])["delta"]
                if hasattr(delta_col, "combine_chunks"):
                    delta_col = delta_col.combine_chunks()
                list_size = (
                    delta_col.type.list_size
                    if hasattr(delta_col.type, "list_size")
                    else D
                )
                _delta_cache[areg.base_pattern_id] = delta_col.values.to_numpy(
                    zero_copy_only=False,
                ).reshape(-1, list_size).astype(np.float32)
            deltas = _delta_cache[areg.base_pattern_id]

            # Filter by cutting plane: signed_distance >= 0 means inside
            inside_mask = cp.signed_distances_batch(deltas) >= 0
            sub_deltas = deltas[inside_mask]
            sub_n = int(inside_mask.sum())

            if sub_n == 0:
                raise ValueError(
                    f"Alias '{alias_id}': cutting plane selects 0 entities"
                )

            # Compute sub-population statistics
            sub_mu = sub_deltas.mean(axis=0)
            sub_sigma = sub_deltas.std(axis=0)
            sub_sigma = np.where(sub_sigma < 1e-9, 1e-9, sub_sigma)

            sub_norms = np.linalg.norm(sub_deltas - sub_mu, axis=1)
            pct = min(pat.anomaly_percentile, 99.0)
            sub_theta_norm = float(np.percentile(sub_norms, pct))
            sub_theta = np.full(len(normal), sub_theta_norm, dtype=np.float32)

            alias_dict: dict[str, Any] = {
                "alias_id": alias_id,
                "base_pattern_id": areg.base_pattern_id,
                "filter": {
                    "include_relations": [r.line_id for r in pat.relations],
                    "cutting_plane": {
                        "normal": [float(x) for x in normal],
                        "bias": float(bias),
                    },
                },
                "derived_pattern": {
                    "mu": sub_mu.tolist(),
                    "sigma_diag": sub_sigma.tolist(),
                    "theta": sub_theta.tolist(),
                    "population_size": sub_n,
                    "computed_at": now_str,
                },
                "version": 1,
                "status": "production",
            }
            if areg.description:
                alias_dict["description"] = areg.description

            result[alias_id] = alias_dict

        return result

    @staticmethod
    def _unpack_lac_bundle(
        lac_bundle: dict[str, Any] | None,
        deltas: np.ndarray | None,
    ) -> dict[str, Any]:
        """Translate the ``_run_label_aware_calibration`` bundle into
        PBR kwargs.

        Returns a dict with keys ``label_aware_calibration``,
        ``label_aware_n_pos``, ``label_aware_n_neg``,
        ``signed_percentiles``, ``intrinsic_displacement_mean``,
        ``extrinsic_displacement_mean`` — every value ``None`` when the
        bundle is missing or the delta matrix is unavailable, ready to
        splat into PatternBuildResult.
        """
        if lac_bundle is None or deltas is None or deltas.ndim != 2:
            return {
                "label_aware_calibration": None,
                "label_aware_n_pos": None,
                "label_aware_n_neg": None,
                "signed_percentiles": None,
                "intrinsic_displacement_mean": None,
                "extrinsic_displacement_mean": None,
            }
        direction = lac_bundle.get("direction")
        if direction is None or direction.shape[0] != deltas.shape[1]:
            return {
                "label_aware_calibration": lac_bundle.get("per_dim"),
                "label_aware_n_pos": lac_bundle.get("n_pos"),
                "label_aware_n_neg": lac_bundle.get("n_neg"),
                "signed_percentiles": None,
                "intrinsic_displacement_mean": None,
                "extrinsic_displacement_mean": None,
            }
        signed_p, intr_mean, extr_mean = (
            GDSBuilder._compute_signed_artefacts(deltas, direction)
        )
        return {
            "label_aware_calibration": lac_bundle.get("per_dim"),
            "label_aware_n_pos": lac_bundle.get("n_pos"),
            "label_aware_n_neg": lac_bundle.get("n_neg"),
            "signed_percentiles": signed_p or None,
            "intrinsic_displacement_mean": intr_mean,
            "extrinsic_displacement_mean": extr_mean,
        }

    @staticmethod
    def _compute_signed_artefacts(
        deltas: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[dict[str, float], float, float]:
        """Derive signed-projection artefacts from deltas + Fisher direction.

        Computes:

        - ``signed_percentiles``: dict with keys ``p1``, ``p5``, ``p50``,
          ``p95``, ``p99`` of the signed projection
          ``deltas @ direction_unit``. The LDA fit returns a unit-norm
          direction so today's caller passes the same vector the
          geometry pass uses to populate ``delta_norm_signed``; we still
          normalise defensively so any future non-unit caller gets the
          contract everyone else assumes.
        - ``intrinsic_displacement_mean``: mean of
          ``|delta . direction_unit|`` across rows. Magnitude of motion
          along the label axis.
        - ``extrinsic_displacement_mean``: mean of
          ``sqrt(||delta||^2 - intrinsic^2)`` across rows. Magnitude of
          motion orthogonal to the label axis.

        Returns empty percentiles dict + (0.0, 0.0) when the delta matrix
        is empty or the direction is the zero vector.
        """
        if deltas.ndim != 2 or deltas.shape[0] == 0:
            return {}, 0.0, 0.0
        d = deltas.astype(np.float64, copy=False)
        v = direction.astype(np.float64, copy=False)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 0.0:
            return {}, 0.0, 0.0
        v = v / v_norm
        projected = d @ v
        percs = np.percentile(projected, [1, 5, 50, 95, 99])
        signed_percentiles = {
            "p1": round(float(percs[0]), 4),
            "p5": round(float(percs[1]), 4),
            "p50": round(float(percs[2]), 4),
            "p95": round(float(percs[3]), 4),
            "p99": round(float(percs[4]), 4),
        }
        intrinsic = np.abs(projected)
        row_norms_sq = np.einsum("ij,ij->i", d, d)
        residual_sq = np.maximum(row_norms_sq - intrinsic * intrinsic, 0.0)
        extrinsic = np.sqrt(residual_sq)
        return (
            signed_percentiles,
            float(intrinsic.mean()),
            float(extrinsic.mean()),
        )

    def _compute_dim_percentiles(
        self,
        entity_line_id: str,
        edge_dim_agg_matrix: np.ndarray | None = None,
        edge_dim_agg_labels: list[str] | None = None,
    ) -> dict[str, dict[str, float]] | None:
        """Compute per-dim percentile cache.

        Walks ``entity_table`` float columns for event_dims / prop_cols
        (legacy path) and, when ``edge_dim_agg_matrix`` is supplied,
        appends one entry per aggregated edge dim keyed by its canonical
        ``{source_dim}_{aggregate}`` label. Labels and matrix columns
        must be 1:1 in order — the caller hands us the parallel list it
        built when populating the matrix. Same six percentile keys
        (``min / p25 / p50 / p75 / p99 / max``) and same numpy.percentile
        call, so downstream consumers (find_anomalies percentile-based
        scoring, sphere_overview profiling_alerts, audit_pattern_dims)
        see a uniform schema across all dim families.
        """
        entity_table = self._lines[entity_line_id].table
        percentiles: dict[str, dict[str, float]] = {}
        for col_field in entity_table.schema:
            if col_field.name == "primary_key":
                continue
            if col_field.type not in (pa.float32(), pa.float64()):
                continue
            arr = entity_table[col_field.name].to_numpy(zero_copy_only=False)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                continue
            percentiles[col_field.name] = {
                "min": round(float(np.min(valid)), 4),
                "p25": round(float(np.percentile(valid, 25)), 4),
                "p50": round(float(np.percentile(valid, 50)), 4),
                "p75": round(float(np.percentile(valid, 75)), 4),
                "p99": round(float(np.percentile(valid, 99)), 4),
                "max": round(float(np.max(valid)), 4),
            }
        # Aggregated edge dims live in the in-memory matrix, not in the
        # entity table. Compute percentiles per column with the same six
        # keys + the same rounding so the cache schema is identical to
        # the event_dim / prop_col entries above. Skip the block silently
        # when no aggregations are declared (matrix is None or empty).
        if (
            edge_dim_agg_matrix is not None
            and edge_dim_agg_labels is not None
            and edge_dim_agg_matrix.shape[1] > 0
        ):
            if len(edge_dim_agg_labels) != edge_dim_agg_matrix.shape[1]:
                raise ValueError(
                    f"edge_dim_agg_labels length ({len(edge_dim_agg_labels)}) "
                    f"does not match matrix column count "
                    f"({edge_dim_agg_matrix.shape[1]}) — label list must be "
                    f"1:1 with matrix columns",
                )
            for col_idx, label in enumerate(edge_dim_agg_labels):
                col_arr = edge_dim_agg_matrix[:, col_idx]
                valid = col_arr[~np.isnan(col_arr)]
                if len(valid) == 0:
                    continue
                percentiles[label] = {
                    "min": round(float(np.min(valid)), 4),
                    "p25": round(float(np.percentile(valid, 25)), 4),
                    "p50": round(float(np.percentile(valid, 50)), 4),
                    "p75": round(float(np.percentile(valid, 75)), 4),
                    "p99": round(float(np.percentile(valid, 99)), 4),
                    "max": round(float(np.max(valid)), 4),
                }
        return percentiles if percentiles else None

    def _compute_dim_normality_pvalues(
        self,
        entity_line_id: str,
        edge_dim_agg_matrix: np.ndarray | None = None,
        edge_dim_agg_labels: list[str] | None = None,
    ) -> dict[str, float] | None:
        """Compute per-dim normality test p-values.

        Mirrors ``_compute_dim_percentiles`` — walks the same float
        columns on the entity table plus aggregated edge-dim matrix
        columns, keyed by the same raw column / label name so the
        navigator's ``_build_raw_dim_name_to_index`` mapping can look
        up the entry at warning time. Stores only the p-value (caller
        compares against alpha); the test family + statistic are
        diagnostic detail the warning path does not need to surface.

        The test family (Shapiro-Wilk vs Kolmogorov-Smirnov) is picked
        per-dim by sample size inside ``normality_test_per_dim`` — no
        per-dim plumbing required here. Dims with fewer than three
        finite values, or zero variance, return ``nan`` from the
        primitive; we skip those keys so the persisted dict only
        carries actionable p-values.

        Kind filtering (gaussian-only) is deferred to the navigator
        warning emitter — keeping the persisted blob test-everything
        means a future re-declaration of a column's kind does not
        require a rebuild to surface the appropriate warning.
        """
        from hypertopos.engine.dim_audit import normality_test_per_dim

        entity_table = self._lines[entity_line_id].table
        pvalues: dict[str, float] = {}
        for col_field in entity_table.schema:
            if col_field.name == "primary_key":
                continue
            if col_field.type not in (pa.float32(), pa.float64()):
                continue
            arr = entity_table[col_field.name].to_numpy(zero_copy_only=False)
            result = normality_test_per_dim(np.asarray(arr, dtype=np.float64))
            p = result["p_value"]
            if np.isfinite(p):
                pvalues[col_field.name] = float(p)
        if (
            edge_dim_agg_matrix is not None
            and edge_dim_agg_labels is not None
            and edge_dim_agg_matrix.shape[1] > 0
        ):
            if len(edge_dim_agg_labels) != edge_dim_agg_matrix.shape[1]:
                raise ValueError(
                    f"edge_dim_agg_labels length ({len(edge_dim_agg_labels)}) "
                    f"does not match matrix column count "
                    f"({edge_dim_agg_matrix.shape[1]}) — label list must be "
                    f"1:1 with matrix columns",
                )
            for col_idx, label in enumerate(edge_dim_agg_labels):
                col_arr = edge_dim_agg_matrix[:, col_idx]
                result = normality_test_per_dim(
                    np.asarray(col_arr, dtype=np.float64),
                )
                p = result["p_value"]
                if np.isfinite(p):
                    pvalues[label] = float(p)
        return pvalues if pvalues else None

    def _build_sphere_json(
        self,
        pattern_stats: dict[str, PatternBuildResult],
    ) -> dict[str, Any]:
        """Build the sphere.json dict."""
        now_str = datetime.now(UTC).isoformat()

        # Determine which lines are entity lines
        entity_lines: dict[str, str] = {}  # line_id -> pattern_id
        for pat_id, pat in self._patterns.items():
            entity_lines[pat.entity_line] = pat_id

        lines_dict = {}
        for line_id, line_reg in self._lines.items():
            line_dict: dict[str, Any] = {
                "line_id": line_id,
                "entity_type": line_reg.entity_type,
                "line_role": line_reg.role,
                "pattern_id": entity_lines.get(line_id, ""),
                "partitioning": {
                    "mode": "static",
                    "columns": (
                        [line_reg.partition_col]
                        if line_reg.partition_col else []
                    ),
                },
                "versions": [1],
            }
            if line_reg.description:
                line_dict["description"] = line_reg.description
            lines_dict[line_id] = line_dict
            columns = []
            for f in line_reg.table.schema:
                if f.name not in _INTERNAL_COLUMNS:
                    columns.append({
                        "name": f.name,
                        "type": _arrow_type_to_str(f.type),
                    })
            lines_dict[line_id]["columns"] = columns
            # Store resolved fts_columns: None → auto-resolved value
            fts = line_reg.fts_columns
            if fts is None:
                fts = "all" if line_reg.role != "event" else []
            lines_dict[line_id]["fts_columns"] = fts
            line_dict["source_id"] = line_reg.source_id

        patterns_dict = {}
        for pat_id, pat in self._patterns.items():
            pbr = pattern_stats[pat_id]
            pat_dict: dict[str, Any] = {
                "pattern_id": pat_id,
                "entity_type": (
                    self._lines[pat.entity_line].entity_type
                ),
                "pattern_type": pat.pattern_type,
                "entity_line": pat.entity_line,
                "version": 1,
                "status": "production",
                "relations": [
                    {
                        "line_id": rel.line_id,
                        "direction": rel.direction,
                        "required": rel.required,
                        **(
                            {"edge_max": rel.edge_max}
                            if rel.edge_max is not None else {}
                        ),
                        **(
                            {"display_name": rel.display_name}
                            if rel.display_name else {}
                        ),
                    }
                    for rel in pat.relations
                ],
                "event_dimensions": [
                    {
                        "column": edim.column,
                        "edge_max": edim.edge_max,
                        **({"display_name": edim.display_name} if edim.display_name else {}),
                    }
                    for edim in pat.event_dimensions
                ] if pat.event_dimensions else [],
                "mu": pbr.mu.tolist(),
                "sigma_diag": pbr.sigma.tolist(),
                "theta": pbr.theta.tolist(),
                "edge_max": (
                    [
                        1 if r.direction == "self"
                        else r.edge_max if r.edge_max is not None
                        else 1
                        for r in pat.relations
                    ]
                    + [edim.edge_max for edim in pat.event_dimensions]
                    + [0] * len(pbr.prop_columns)
                ) if (
                    any(r.edge_max is not None for r in pat.relations)
                    or pat.event_dimensions
                ) else None,
                "population_size": pbr.population_size,
                "computed_at": now_str,
                "last_calibrated_at": now_str,
                "prop_columns": pbr.prop_columns,
                "excluded_properties": pbr.excluded_properties,
                **(
                    {"edge_dim_names": list(pbr.edge_dim_names)}
                    if pbr.edge_dim_names else {}
                ),
            }
            if pbr.dim_block_names:
                pat_dict["dim_block_names"] = pbr.dim_block_names
            if pbr.dim_block_stats:
                pat_dict["dim_block_stats"] = pbr.dim_block_stats
            if pbr.group_stats:
                pat_dict["group_by_property"] = (
                    pat.group_by_property
                )
                pat_dict["group_stats"] = {
                    gid: {
                        "mu": g_mu.tolist(),
                        "sigma_diag": g_sigma.tolist(),
                        "theta": g_theta.tolist(),
                        "population_size": g_pop,
                    }
                    for gid, (g_mu, g_sigma, g_theta, g_pop)
                    in pbr.group_stats.items()
                }
            if pbr.dimension_weights is not None:
                pat_dict["dimension_weights"] = (
                    pbr.dimension_weights.tolist()
                )
            if pbr.gmm_components is not None:
                pat_dict["gmm_components"] = [
                    {
                        "mu": c_mu.tolist(),
                        "sigma_diag": c_sig.tolist(),
                        "theta": c_th.tolist(),
                        "population_size": c_pop,
                    }
                    for c_mu, c_sig, c_th, c_pop
                    in pbr.gmm_components
                ]
            if pbr.cholesky_inv is not None:
                pat_dict["cholesky_inv"] = (
                    pbr.cholesky_inv.tolist()
                )
            if pbr.dimension_kinds is not None:
                pat_dict["dimension_kinds"] = pbr.dimension_kinds
            if pbr.dim_percentiles:
                pat_dict["dim_percentiles"] = pbr.dim_percentiles
            if pbr.heteroscedasticity_diagnostic:
                pat_dict["heteroscedasticity_diagnostic"] = (
                    pbr.heteroscedasticity_diagnostic
                )
            if pbr.dim_normality_pvalues:
                pat_dict["dim_normality_pvalues"] = pbr.dim_normality_pvalues
            if pbr.label_aware_calibration:
                # Flatten DimCalibration dataclasses to JSON-safe dicts.
                # Reader hydrates each entry back to a SimpleNamespace
                # (engine.DimCalibration shape: mu_pos/sigma_pos/mu_neg/
                # sigma_neg/direction) so audit_pattern_dims' attribute
                # access (`dim_cal.mu_pos`) keeps working on round-trip.
                pat_dict["label_aware_calibration"] = {
                    label: {
                        "mu_pos": float(dc.mu_pos),
                        "sigma_pos": float(dc.sigma_pos),
                        "mu_neg": float(dc.mu_neg),
                        "sigma_neg": float(dc.sigma_neg),
                        "direction": float(dc.direction),
                    }
                    for label, dc in pbr.label_aware_calibration.items()
                }
                if pbr.label_aware_n_pos is not None:
                    pat_dict["label_aware_n_pos"] = int(pbr.label_aware_n_pos)
                if pbr.label_aware_n_neg is not None:
                    pat_dict["label_aware_n_neg"] = int(pbr.label_aware_n_neg)
                if pbr.signed_percentiles:
                    pat_dict["signed_percentiles"] = pbr.signed_percentiles
                if pbr.intrinsic_displacement_mean is not None:
                    pat_dict["intrinsic_displacement_mean"] = float(
                        pbr.intrinsic_displacement_mean,
                    )
                if pbr.extrinsic_displacement_mean is not None:
                    pat_dict["extrinsic_displacement_mean"] = float(
                        pbr.extrinsic_displacement_mean,
                    )
            if pat.description:
                pat_dict["description"] = pat.description
            if pat.fdr_hierarchy:
                pat_dict["fdr_hierarchy"] = [
                    {"level": lvl.level, "from_dimension": lvl.from_dimension}
                    for lvl in pat.fdr_hierarchy
                ]
            if pat.fdr_temporal_hierarchy:
                pat_dict["fdr_temporal_hierarchy"] = [
                    {
                        "level": lvl.level,
                        "slice_dimension": lvl.slice_dimension,
                        "bucket": lvl.bucket,
                    }
                    for lvl in pat.fdr_temporal_hierarchy
                ]
            if pat.conformance_rules:
                from hypertopos.model.sphere import _rule_to_dict
                pat_dict["conformance_rules"] = [
                    _rule_to_dict(r) for r in pat.conformance_rules
                ]
            # Edge table metadata
            edge_cfg = self._resolve_edge_table_config(pat)
            if edge_cfg is not None:
                edge_path = self.output_path / "edges" / pat_id / "data.lance"
                if edge_path.exists():
                    pat_dict["has_edge_table"] = True
                    edge_meta: dict[str, str] = {
                        "from_col": edge_cfg.from_col,
                        "to_col": edge_cfg.to_col,
                    }
                    if edge_cfg.timestamp_col:
                        edge_meta["timestamp_col"] = edge_cfg.timestamp_col
                    if edge_cfg.amount_col:
                        edge_meta["amount_col"] = edge_cfg.amount_col
                    pat_dict["edge_table"] = edge_meta
            # Edge-dim aggregations metadata (S1 ext)
            if pat.edge_dim_aggregations is not None:
                eda = pat.edge_dim_aggregations
                pat_dict["edge_dim_aggregations"] = {
                    "from": eda.from_event_pattern,
                    "dims": (
                        list(eda.dims) if eda.dims is not None else None
                    ),
                    "aggregates_per_dim": {
                        d: list(aggs)
                        for d, aggs in eda.aggregates_per_dim.items()
                    },
                }
            # Calibration epoch metadata (populated by _build_and_write)
            cal_state = self._calibration_state.get(pat_id, {})
            pat_dict["calibration_epoch"] = cal_state.get("calibration_epoch", 1)
            pat_dict["schema_hash"] = cal_state.get("schema_hash")
            patterns_dict[pat_id] = pat_dict

        # Sphere format minor-bump: 3.1 when a label_audit block is
        # registered, else 3.0. Readers compare on major only, so 3.0
        # readers transparently load 3.1 spheres.
        fmt_version = "3.1" if self._label_audit_block is not None else "3.0"
        sphere_dict: dict[str, Any] = {
            "sphere_id": self.sphere_id,
            "format_version": fmt_version,
            "calibration_history_policy": {"last_k": 5},
            "name": self._name or self.sphere_id,
            "lines": lines_dict,
            "patterns": patterns_dict,
            "aliases": self._build_aliases(pattern_stats),
            "storage": {
                "geometry": {"format": "lance"},
                "points": {"format": "lance"},
            },
        }
        if self._description:
            sphere_dict["description"] = self._description
        if self._label_audit_block is not None:
            la = self._label_audit_block
            sphere_dict["label_audit"] = {
                "label_column": la.label_column,
                "label_positive_value": la.label_positive_value,
                "patterns": list(la.patterns),
            }
        return sphere_dict

    def _resolve_derived(self) -> None:
        """Resolve composite lines, derived dimensions, and graph features.

        Must be called before _validate() — modifies self._lines and
        auto-creates RelationSpecs on patterns.
        """
        from collections import defaultdict

        from hypertopos.builder.derived import (
            _is_batchable,
            build_composite_table,
            compute_derived_batch,
            compute_derived_dimension,
            compute_graph_features,
        )

        # 1. Resolve composite lines — create anchor lines from event data
        for spec in self._composite_lines:
            if spec.event_line not in self._lines:
                raise ValueError(
                    f"Composite line '{spec.line_id}': event_line "
                    f"'{spec.event_line}' not registered"
                )
            event_table = self._lines[spec.event_line].table
            composite_table = build_composite_table(
                event_table, spec.key_cols, spec.separator,
            )
            self.add_line(
                spec.line_id, composite_table,
                key_col="primary_key", source_id=spec.line_id, role="anchor",
            )

        # 2. Resolve derived dimensions — aggregate and add columns + RelationSpecs
        #    Validate all specs upfront
        for spec in self._derived_dims:
            if spec.anchor_line not in self._lines:
                raise ValueError(
                    f"Derived dim '{spec.dimension_name}': anchor_line "
                    f"'{spec.anchor_line}' not registered"
                )
            if spec.event_line not in self._lines:
                raise ValueError(
                    f"Derived dim '{spec.dimension_name}': event_line "
                    f"'{spec.event_line}' not registered"
                )

        # Group batchable specs by (anchor_line, event_line, anchor_fk)
        # so one group_by call handles multiple metrics on the same FK.
        batch_groups: dict[
            tuple[str, str, str], list
        ] = defaultdict(list)
        single_specs: list = []

        for spec in self._derived_dims:
            if _is_batchable(spec):
                fk = spec.anchor_fk
                fk_key = tuple(fk) if isinstance(fk, list) else fk
                group_key = (spec.anchor_line, spec.event_line, fk_key)
                batch_groups[group_key].append(spec)
            else:
                single_specs.append(spec)

        # 2a. Batch path — one group_by per (anchor_line, event_line, anchor_fk)
        for (anchor_line, event_line, _fk_key), specs in batch_groups.items():
            anchor_reg = self._lines[anchor_line]
            event_table = self._lines[event_line].table
            anchor_keys = anchor_reg.table["primary_key"]

            # Recover original anchor_fk (list for composite, str for single)
            actual_fk = specs[0].anchor_fk
            separator = "→"
            if isinstance(actual_fk, list):
                for cs in self._composite_lines:
                    if cs.line_id == anchor_line:
                        separator = cs.separator
                        break

            batch_results = compute_derived_batch(
                event_table, anchor_keys, actual_fk, specs,
                separator=separator,
            )

            # Append all columns from this batch at once
            new_columns: list[tuple[str, pa.Array]] = []
            for spec in specs:
                dim_name = spec.dimension_name
                values, em = batch_results[dim_name]
                new_columns.append((
                    dim_name,
                    pa.array(values, type=pa.float64()),
                ))
                # Auto-create dummy dim line + RelationSpec
                dim_line_id = f"_d_{dim_name}"
                if dim_line_id not in self._lines:
                    self.add_line(
                        dim_line_id,
                        pa.table({"primary_key": ["_dummy"]}),
                        key_col="primary_key", source_id=dim_line_id, role="anchor",
                    )
                for pat in self._patterns.values():
                    if pat.entity_line == anchor_line:
                        pat.relations.append(RelationSpec(
                            line_id=dim_line_id,
                            fk_col=dim_name,
                            direction="out",
                            required=False,
                            display_name=dim_name,
                            edge_max=em,
                        ))

            for col_name, col_arr in new_columns:
                anchor_reg.table = anchor_reg.table.append_column(
                    col_name, col_arr,
                )

        # 2b. Single path — specs that need special handling
        for spec in single_specs:
            anchor_reg = self._lines[spec.anchor_line]
            event_table = self._lines[spec.event_line].table
            anchor_keys = anchor_reg.table["primary_key"]

            # For composite anchors, find matching CompositeLineSpec separator
            separator = "→"
            if isinstance(spec.anchor_fk, list):
                for cs in self._composite_lines:
                    if cs.line_id == spec.anchor_line:
                        separator = cs.separator
                        break

            values, em = compute_derived_dimension(
                event_table, anchor_keys, spec.anchor_fk,
                spec.metric, spec.metric_col,
                spec.edge_max, spec.percentile,
                time_col=spec.time_col,
                time_window=spec.time_window,
                window_aggregation=spec.window_aggregation,
                separator=separator,
            )

            # Add column to anchor table
            anchor_reg.table = anchor_reg.table.append_column(
                spec.dimension_name,
                pa.array(values, type=pa.float64()),
            )

            # Auto-create dummy dim line + RelationSpec on matching patterns
            dim_line_id = f"_d_{spec.dimension_name}"
            if dim_line_id not in self._lines:
                self.add_line(
                    dim_line_id,
                    pa.table({"primary_key": ["_dummy"]}),
                    key_col="primary_key", source_id=dim_line_id, role="anchor",
                )
            for pat in self._patterns.values():
                if pat.entity_line == spec.anchor_line:
                    pat.relations.append(RelationSpec(
                        line_id=dim_line_id,
                        fk_col=spec.dimension_name,
                        direction="out",
                        required=False,
                        display_name=spec.dimension_name,
                        edge_max=em,
                    ))

        # 2b. Resolve precomputed dimensions — column already on entity table
        for spec in self._precomputed_dims:
            if spec.anchor_line not in self._lines:
                raise ValueError(
                    f"Precomputed dim '{spec.dimension_name}': anchor_line "
                    f"'{spec.anchor_line}' not registered"
                )
            anchor_reg = self._lines[spec.anchor_line]
            if spec.dimension_name not in anchor_reg.table.schema.names:
                raise ValueError(
                    f"Precomputed dim '{spec.dimension_name}': column not found "
                    f"on '{spec.anchor_line}' entity table. "
                    f"Available: {anchor_reg.table.schema.names}"
                )

            # Compute edge_max from existing column values
            col = anchor_reg.table[spec.dimension_name]
            vals = pc.fill_null(col, 0).to_numpy(
                zero_copy_only=False,
            ).astype(np.float32)
            if spec.edge_max == "auto":
                nonzero = vals[vals > 0]
                em = (
                    max(1, int(np.percentile(nonzero, spec.percentile)))
                    if len(nonzero) > 0 else 1
                )
            else:
                em = int(spec.edge_max)

            # Create dummy dim line + RelationSpec (same pattern as derived dims)
            dim_line_id = f"_d_{spec.dimension_name}"
            if dim_line_id not in self._lines:
                self.add_line(
                    dim_line_id,
                    pa.table({"primary_key": ["_dummy"]}),
                    key_col="primary_key", source_id=dim_line_id, role="anchor",
                )
            for pat in self._patterns.values():
                if pat.entity_line == spec.anchor_line:
                    pat.relations.append(RelationSpec(
                        line_id=dim_line_id,
                        fk_col=spec.dimension_name,
                        direction="out",
                        required=False,
                        display_name=spec.display_name or spec.dimension_name,
                        edge_max=em,
                    ))

        # 3. Resolve graph features — compute and add as derived dims
        for spec in self._graph_features:
            if spec.anchor_line not in self._lines:
                raise ValueError(
                    f"Graph features: anchor_line '{spec.anchor_line}' not registered"
                )
            if spec.event_line not in self._lines:
                raise ValueError(
                    f"Graph features: event_line '{spec.event_line}' not registered"
                )
            anchor_reg = self._lines[spec.anchor_line]
            event_table = self._lines[spec.event_line].table
            anchor_keys = anchor_reg.table["primary_key"]

            feature_results = compute_graph_features(
                event_table, anchor_keys,
                spec.from_col, spec.to_col, spec.features,
            )

            for feat_name, (values, em) in feature_results.items():
                anchor_reg.table = anchor_reg.table.append_column(
                    feat_name,
                    pa.array(values, type=pa.float64()),
                )
                dim_line_id = f"_d_{feat_name}"
                if dim_line_id not in self._lines:
                    self.add_line(
                        dim_line_id,
                        pa.table({"primary_key": ["_dummy"]}),
                        key_col="primary_key", source_id=dim_line_id, role="anchor",
                    )
                for pat in self._patterns.values():
                    if pat.entity_line == spec.anchor_line:
                        pat.relations.append(RelationSpec(
                            line_id=dim_line_id,
                            fk_col=feat_name,
                            direction="out",
                            required=False,
                            display_name=feat_name,
                            edge_max=em,
                        ))

    def _resolve_chain_dims(self) -> None:
        """Resolve chain dimensions — auto-create RelationSpecs on chain patterns."""
        for line_id, feat_name, em in self._chain_dims:
            dim_line_id = f"_d_chain_{feat_name}"
            for pat in self._patterns.values():
                if pat.entity_line == line_id:
                    pat.relations.append(RelationSpec(
                        line_id=dim_line_id,
                        fk_col=feat_name,
                        direction="out",
                        required=False,
                        display_name=feat_name,
                        edge_max=em,
                    ))

    def _evaluate_conformance_rules(self) -> None:
        """Build conformance sidecars for every pattern that declares rules.

        Cost-neutral fast path: short-circuits when no pattern in the
        builder declares any rules. Called once at the start of
        ``build()`` after points are materialized and before geometry.

        Sidecars land at version=1 alongside the geometry write — every
        pattern in the builder writes version=1 in the same build call.
        """
        # Cost-neutral guard — single linear scan of _patterns.
        if not any(p.conformance_rules for p in self._patterns.values()):
            return

        from hypertopos.builder.conformance import build_conformance_for_pattern
        from hypertopos.model.sphere import Pattern as _Pattern

        for pat_id, pat in self._patterns.items():
            if not pat.conformance_rules:
                continue
            line_reg = self._lines.get(pat.entity_line)
            if line_reg is None:
                raise ValueError(
                    f"conformance evaluation: pattern '{pat_id}' references "
                    f"unknown entity_line '{pat.entity_line}'",
                )
            points_table = line_reg.table
            # Shallow ``Pattern`` shim — the conformance evaluator reads
            # ``pattern_id`` + ``conformance_rules`` only. The full Pattern
            # dataclass is not materialised until after geometry build.
            shim = _Pattern.__new__(_Pattern)
            object.__setattr__(shim, "pattern_id", pat_id)
            object.__setattr__(shim, "conformance_rules", pat.conformance_rules)
            build_conformance_for_pattern(
                base_path=self.output_path,
                pattern=shim,
                points_table=points_table,
                version=1,
            )

    def build(
        self,
        *,
        temporal_configs: list[dict[str, str]] | None = None,
    ) -> str:
        """Validate, compute stats, write all files. Returns output_path as string.

        Args:
            temporal_configs: When provided, runs per-pattern pipeline
                (geometry → temporal) in parallel instead of geometry-only.
                Each dict: {"time_col", "time_window"} and optionally
                {"event_line", "anchor_pattern"}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._resolve_derived()
        self._resolve_chain_dims()
        self._validate()

        pattern_stats: dict[str, PatternBuildResult] = {}

        # 1. Write all points files (skip internal dummy lines)
        from hypertopos.builder._writer import (
            finalize_geometry_chunks,
            write_geometry,
            write_geometry_chunk,
            write_points,
        )
        from hypertopos.storage.writer import GDSWriter

        def _write_line_points(line_id: str, line_reg: _LineReg) -> None:
            if line_id.startswith("_d_"):
                return
            fts = line_reg.fts_columns
            if fts is None:
                fts = [] if line_reg.role == "event" else "all"
            write_points(
                self.output_path,
                line_id,
                line_reg.table,
                version=1,
                partition_col=line_reg.partition_col,
                fts_columns=fts,
            )

        real_lines = [
            (lid, lr) for lid, lr in self._lines.items()
            if not lid.startswith("_d_")
        ]
        if len(real_lines) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(real_lines))) as pool:
                futures = [
                    pool.submit(_write_line_points, lid, lr)
                    for lid, lr in real_lines
                ]
                for fut in futures:
                    fut.result()
        else:
            for lid, lr in real_lines:
                _write_line_points(lid, lr)

        # 1.5 Conformance rules — evaluate after points are materialized
        # and BEFORE geometry build. Cost-neutral fast path when no
        # pattern declares any rules: short-circuit on the first check.
        self._evaluate_conformance_rules()

        # 2. Build geometry (and optionally temporal) per pattern
        _stats_writer = GDSWriter(str(self.output_path))

        def _build_and_write(
            pat_id: str, pat: _PatternReg,
        ) -> tuple[str, PatternBuildResult]:
            n = len(self._lines[pat.entity_line].table)

            if n > GEOMETRY_CHUNK_SIZE:
                if (pat.group_by_property
                        or pat.gmm_n_components
                        or pat.use_mahalanobis):
                    pat_id_out, pbr = self._build_and_write_chunked(
                        pat_id, pat, _stats_writer,
                        write_geometry_chunk, finalize_geometry_chunks,
                    )
                else:
                    pat_id_out, pbr = self._build_and_write_streaming(
                        pat_id, pat, _stats_writer,
                        write_geometry_chunk, finalize_geometry_chunks,
                    )
                self._write_calibration_epoch_for_pattern(pat, pbr)
                return pat_id_out, pbr

            geom_table, ps, lac = self._build_geometry_table(pat)
            geom_table = self._inject_fdr_hierarchy_carriers(pat, geom_table)
            geom_table = self._inject_fdr_temporal_buckets(pat, geom_table)
            _validate_fdr_hierarchy_columns(
                pat, list(geom_table.column_names),
            )
            write_geometry(
                self.output_path, pat_id, geom_table, version=1,
            )

            delta_norms = geom_table["delta_norm"].to_numpy(
                zero_copy_only=False,
            )
            is_anomaly_arr = geom_table["is_anomaly"].to_numpy(
                zero_copy_only=False,
            )
            theta_norm = float(np.linalg.norm(ps.theta))
            _stats_writer.write_geometry_stats(
                pat_id, version=1,
                delta_norms=delta_norms, theta_norm=theta_norm,
                is_anomaly_arr=is_anomaly_arr,
            )

            edge_cfg = self._resolve_edge_table_config(pat)
            if edge_cfg is not None:
                edge_table = self._extract_edge_table(pat, edge_cfg)
                if edge_table.num_rows > 0:
                    _stats_writer.write_edges(pat_id, edge_table)
                    import pyarrow.compute as _pc
                    _stats_writer.write_edge_stats(pat_id, {
                        "row_count": edge_table.num_rows,
                        "unique_from": _pc.count_distinct(edge_table["from_key"]).as_py(),
                        "unique_to": _pc.count_distinct(edge_table["to_key"]).as_py(),
                        "timestamp_min": float(_pc.min(edge_table["timestamp"]).as_py()),
                        "timestamp_max": float(_pc.max(edge_table["timestamp"]).as_py()),
                        "amount_min": float(_pc.min(edge_table["amount"]).as_py()),
                        "amount_max": float(_pc.max(edge_table["amount"]).as_py()),
                    })

            pbr = PatternBuildResult(
                mu=ps.mu, sigma=ps.sigma, theta=ps.theta,
                population_size=n,
                prop_columns=ps.prop_columns,
                excluded_properties=ps.excluded_properties,
                group_stats=ps.group_stats_dict,
                dimension_weights=ps.dim_weights,
                gmm_components=ps.gmm_components,
                cholesky_inv=ps.cholesky_inv,
                dim_percentiles=self._compute_dim_percentiles(
                    pat.entity_line,
                    edge_dim_agg_matrix=ps.edge_dim_agg_matrix,
                    edge_dim_agg_labels=ps.edge_dim_agg_labels,
                ),
                dimension_kinds=ps.dimension_kinds,
                dim_block_names=ps.dim_block_names,
                dim_block_stats=ps.dim_block_stats,
                theta_sensitivity=ps.theta_sensitivity,
                heteroscedasticity_diagnostic=ps.heteroscedasticity_diagnostic,
                dim_normality_pvalues=self._compute_dim_normality_pvalues(
                    pat.entity_line,
                    edge_dim_agg_matrix=ps.edge_dim_agg_matrix,
                    edge_dim_agg_labels=ps.edge_dim_agg_labels,
                ),
                edge_dim_names=ps.edge_dim_names,
                **self._unpack_lac_bundle(lac, ps.deltas),
            )
            self._write_calibration_epoch_for_pattern(pat, pbr)
            return pat_id, pbr

        # ── Pipeline mode: geometry → temporal per-pattern ──
        if temporal_configs:
            temporal_ctx = self._prepare_temporal_context(temporal_configs)
            temporal_writer = GDSWriter(str(self.output_path))

            def _pipeline(pid: str, p: _PatternReg) -> tuple[str, PatternBuildResult]:
                pat_id, stats = _build_and_write(pid, p)
                ctx = temporal_ctx.get(pat_id)
                if ctx is not None and p.pattern_type == "anchor":
                    pat_meta = self._build_pat_meta_for_temporal(p, stats)
                    self._build_temporal_one(
                        pat_id, p, pat_meta, ctx["event_table"],
                        ctx["bucket_np"], ctx["n_buckets"],
                        ctx["min_ts"], ctx["window_secs"],
                        temporal_writer,
                    )
                return pat_id, stats

            event_items_p = [
                (pid, p) for pid, p in self._patterns.items()
                if p.pattern_type == "event"
            ]
            anchor_items_p = [
                (pid, p) for pid, p in self._patterns.items()
                if p.pattern_type == "anchor"
            ]

            def _run_phase_pipeline(
                items: list[tuple[str, _PatternReg]],
            ) -> None:
                if not items:
                    return
                if len(items) > 1:
                    with ThreadPoolExecutor(
                        max_workers=min(4, len(items)),
                    ) as pool:
                        futs = {
                            pool.submit(_pipeline, pid, p): pid
                            for pid, p in items
                        }
                        for fut in as_completed(futs):
                            pat_id, stats = fut.result()
                            pattern_stats[pat_id] = stats
                else:
                    for pat_id, pat in items:
                        _, stats = _pipeline(pat_id, pat)
                        pattern_stats[pat_id] = stats

            _run_phase_pipeline(event_items_p)
            _run_phase_pipeline(anchor_items_p)
        else:
            # ── Original mode: geometry only ──
            # Anchor patterns with edge_dim_aggregations consume the
            # event pattern's edge_features sidecar — built in two phases
            # so the dependency holds without a topo sort.
            event_items = [
                (pid, p) for pid, p in self._patterns.items()
                if p.pattern_type == "event"
            ]
            anchor_items = [
                (pid, p) for pid, p in self._patterns.items()
                if p.pattern_type == "anchor"
            ]

            def _run_phase(items: list[tuple[str, _PatternReg]]) -> None:
                if not items:
                    return
                if len(items) > 1:
                    with ThreadPoolExecutor(
                        max_workers=min(4, len(items)),
                    ) as pool:
                        futures = [
                            pool.submit(_build_and_write, pid, p)
                            for pid, p in items
                        ]
                        for fut in futures:
                            pat_id, stats = fut.result()
                            pattern_stats[pat_id] = stats
                else:
                    for pat_id, pat in items:
                        _, stats = _build_and_write(pat_id, pat)
                        pattern_stats[pat_id] = stats

            _run_phase(event_items)
            _run_phase(anchor_items)

        # 2.5 Precompute per-entity contagion stats
        self._build_contagion_stats(_stats_writer)

        # 3. Write sphere.json
        sphere_data = self._build_sphere_json(pattern_stats)
        meta_dir = self.output_path / "_gds_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "sphere.json").write_text(json.dumps(sphere_data, indent=2))

        # 4. Initialize calibration tracker for each pattern
        from hypertopos.engine.calibration import CalibrationTracker

        for pid, pbr in pattern_stats.items():
            tracker = CalibrationTracker.from_stats(
                pbr.mu, pbr.sigma, pbr.theta, n=pbr.population_size,
            )
            _stats_writer.write_calibration_tracker(pid, tracker)

        return str(self.output_path)

    def _write_calibration_epoch_for_pattern(
        self,
        pat: _PatternReg,
        pbr: PatternBuildResult,
    ) -> None:
        """Write a fresh calibration_history epoch for one pattern.

        Decides reset-vs-increment by comparing the new schema_hash against
        the prior sphere.json's pattern.schema_hash, writes v={N}.json under
        _gds_meta/calibration_history/{pattern_id}/, trims to last_k, and
        stashes (calibration_epoch, schema_hash) on self._calibration_state
        for the sphere.json composer to pick up.

        Safe to call from inside ThreadPoolExecutor: each pattern_id has a
        distinct on-disk dir and a distinct dict key on _calibration_state.
        The prior sphere.json is only read here (not written) — the new
        sphere.json is written sequentially after the pool joins.
        """
        from datetime import datetime, timezone

        from hypertopos.model.sphere import CalibrationFit
        from hypertopos.storage.calibration_history import (
            history_dir,
            reset_calibration_history,
            write_calibration_history_epoch,
        )

        # Use the post-refit prop_columns + dimension_kinds so the hash
        # matches what _build_sphere_json writes into the pattern node.
        new_schema_hash = _compute_schema_hash_for_pattern_def(
            pat,
            prop_columns=pbr.prop_columns,
            dimension_kinds=pbr.dimension_kinds,
        )

        prev_sphere_path = self.output_path / "_gds_meta" / "sphere.json"
        prev_meta: dict = {}
        prev_schema_hash: str | None = None
        prev_calibration_epoch: int = 0
        if prev_sphere_path.exists():
            prev_meta = json.loads(
                prev_sphere_path.read_text(encoding="utf-8"),
            )
            prev_pattern_node = prev_meta.get("patterns", {}).get(
                pat.pattern_id,
            )
            if prev_pattern_node is not None:
                prev_schema_hash = prev_pattern_node.get("schema_hash")
                prev_calibration_epoch = prev_pattern_node.get(
                    "calibration_epoch", 0,
                ) or 0

        last_k = (
            prev_meta.get("calibration_history_policy") or {}
        ).get("last_k", 5)
        if last_k < 1:
            raise ValueError(
                f"calibration_history_policy.last_k must be >= 1, got {last_k} "
                f"in {prev_sphere_path}"
            )

        history_present = history_dir(
            self.output_path, pat.pattern_id,
        ).exists()

        reset = (
            prev_schema_hash is None
            or prev_schema_hash != new_schema_hash
            or not history_present
        )
        if reset:
            reset_calibration_history(self.output_path, pat.pattern_id)
            new_epoch = 1
        else:
            new_epoch = prev_calibration_epoch + 1

        # Convert group_stats / gmm_components from internal numpy tuples
        # into the JSON-friendly dicts that sphere.json itself stores. The
        # CalibrationFit serializer round-trips these via json.dumps, so
        # numpy arrays must be tolist()'d here.
        if pbr.group_stats:
            group_stats_json: dict | None = {
                gid: {
                    "mu": g_mu.tolist(),
                    "sigma_diag": g_sigma.tolist(),
                    "theta": g_theta.tolist(),
                    "population_size": int(g_pop),
                }
                for gid, (g_mu, g_sigma, g_theta, g_pop)
                in pbr.group_stats.items()
            }
        else:
            group_stats_json = None

        if pbr.gmm_components is not None:
            gmm_components_json: list | None = [
                {
                    "mu": c_mu.tolist(),
                    "sigma_diag": c_sig.tolist(),
                    "theta": c_th.tolist(),
                    "population_size": int(c_pop),
                }
                for c_mu, c_sig, c_th, c_pop in pbr.gmm_components
            ]
        else:
            gmm_components_json = None

        # Reconstruct edge_max vector matching sphere.json composition: one
        # entry per relation, then per event_dim, then zero per prop_column.
        # Mirrors the shape used at sphere.json build time.
        if (
            any(r.edge_max is not None for r in pat.relations)
            or pat.event_dimensions
        ):
            edge_max_list = (
                [
                    1 if r.direction == "self"
                    else r.edge_max if r.edge_max is not None
                    else 1
                    for r in pat.relations
                ]
                + [edim.edge_max for edim in pat.event_dimensions]
                + [0] * len(pbr.prop_columns)
            )
            edge_max_arr: np.ndarray | None = np.asarray(
                edge_max_list, dtype=np.float32,
            )
        else:
            edge_max_arr = None

        now = datetime.now(timezone.utc)
        fit = CalibrationFit(
            pattern_id=pat.pattern_id,
            calibration_epoch=new_epoch,
            schema_version=1,
            schema_hash=new_schema_hash,
            mu=np.asarray(pbr.mu, dtype=np.float32),
            sigma_diag=np.asarray(pbr.sigma, dtype=np.float32),
            theta=np.asarray(pbr.theta, dtype=np.float32),
            population_size=int(pbr.population_size),
            dimension_weights=(
                np.asarray(pbr.dimension_weights, dtype=np.float32)
                if pbr.dimension_weights is not None else None
            ),
            dimension_kinds=(
                list(pbr.dimension_kinds)
                if pbr.dimension_kinds is not None else None
            ),
            dim_percentiles=pbr.dim_percentiles,
            group_stats=group_stats_json,
            gmm_components=gmm_components_json,
            edge_max=edge_max_arr,
            computed_at=now,
            last_calibrated_at=now,
            edge_dim_thresholds=self._edge_dim_thresholds.get(pat.pattern_id),
            theta_sensitivity=pbr.theta_sensitivity,
            dim_normality_pvalues=pbr.dim_normality_pvalues,
        )
        write_calibration_history_epoch(
            self.output_path, fit, last_k=last_k,
        )

        self._calibration_state[pat.pattern_id] = {
            "calibration_epoch": new_epoch,
            "schema_hash": new_schema_hash,
        }

    def _prepare_temporal_context(
        self,
        temporal_configs: list[dict[str, str]],
    ) -> dict[str, dict[str, Any]]:
        """Pre-compute shared temporal state for pipeline mode.

        Returns mapping: pattern_id → {event_table, bucket_np, n_buckets,
        min_ts, window_secs}.
        """
        result: dict[str, dict[str, Any]] = {}

        for tc in temporal_configs:
            time_col = tc["time_col"]
            time_window = tc["time_window"]
            event_line_id = tc.get("event_line")
            anchor_pattern = tc.get("anchor_pattern")

            if event_line_id is None:
                event_lines = [
                    lid for lid, lr in self._lines.items()
                    if lr.role == "event"
                ]
                if len(event_lines) == 0:
                    raise ValueError("No event lines registered")
                if len(event_lines) > 1:
                    raise ValueError(
                        f"Multiple event lines found: {event_lines}. "
                        "Specify event_line explicitly."
                    )
                event_line_id = event_lines[0]

            event_table = self._lines[event_line_id].table
            bucket_np, n_buckets, min_ts, window_secs = (
                self._parse_event_buckets(event_table, time_col, time_window)
            )
            if n_buckets == 0:
                continue

            ctx = {
                "event_table": event_table,
                "bucket_np": bucket_np,
                "n_buckets": n_buckets,
                "min_ts": min_ts,
                "window_secs": window_secs,
            }

            # Map to target patterns
            if anchor_pattern is not None:
                result[anchor_pattern] = ctx
            else:
                for pid, pat in self._patterns.items():
                    if pat.pattern_type == "anchor":
                        result[pid] = ctx

        return result

    def _resolve_anchor_pattern_for_edge_table(
        self, pat_id: str, pat: _PatternReg,
    ) -> str | None:
        """Find which anchor pattern's geometry holds the is_anomaly bitmap
        for entities appearing in *pat*'s edge table.

        For non-event patterns the pattern is its own anchor (its edges connect
        entities of its own line). For event patterns we walk the relations and
        return the anchor that owns the entity line of the first matching
        relation. Mirrors PassiveScanner._resolve_anchor_for_event so the
        precomputed table covers exactly the keys the scanner asks about at
        runtime.
        """
        if pat.pattern_type != "event":
            return pat_id
        for other_id, other in self._patterns.items():
            if other.pattern_type != "anchor" or other_id == pat_id:
                continue
            for rel in pat.relations:
                if rel.line_id == other.entity_line:
                    return other_id
        return None

    @staticmethod
    def _compute_contagion_arrow(
        edges_table: pa.Table,
        geom_table: pa.Table,
    ) -> pa.Table:
        """Arrow-native contagion stats — no Python loops over rows."""
        import math

        _empty = pa.table({
            "primary_key": pa.array([], type=pa.string()),
            "neighbor_count": pa.array([], type=pa.int32()),
            "anomalous_neighbor_count": pa.array([], type=pa.int32()),
            "contagion_ratio": pa.array([], type=pa.float32()),
        })

        if edges_table.num_rows == 0:
            return _empty

        # Filter self-loops
        not_self = pc.not_equal(
            edges_table["from_key"], edges_table["to_key"],
        )
        edges_table = edges_table.filter(not_self)
        if edges_table.num_rows == 0:
            return _empty

        # Bidirectional edges
        fwd = edges_table.select(["from_key", "to_key"])
        rev = edges_table.select(["to_key", "from_key"]).rename_columns(
            ["from_key", "to_key"],
        )
        all_edges = pa.concat_tables([fwd, rev])

        # Neighbor counts (distinct)
        neighbor_counts = all_edges.group_by("from_key").aggregate(
            [("to_key", "count_distinct")],
        )

        # Anomalous neighbors
        anom_mask = geom_table["is_anomaly"]
        anomalous_keys = pc.filter(geom_table["primary_key"], anom_mask)

        anom_edges = all_edges.filter(
            pc.is_in(all_edges["to_key"], value_set=anomalous_keys),
        )
        if anom_edges.num_rows > 0:
            anom_counts = anom_edges.group_by("from_key").aggregate(
                [("to_key", "count_distinct")],
            )
        else:
            anom_counts = pa.table({
                "from_key": pa.array([], type=pa.string()),
                "to_key_count_distinct": pa.array([], type=pa.int64()),
            })

        # Join
        joined = neighbor_counts.join(
            anom_counts, keys="from_key", join_type="left outer",
            right_suffix="_anom",
        )

        pk = joined["from_key"]
        nc = joined["to_key_count_distinct"]
        ac_col_name = None
        if "to_key_count_distinct_anom" in joined.schema.names:
            ac_col_name = "to_key_count_distinct_anom"
        else:
            anom_cols = [n for n in joined.schema.names if n.endswith("_anom")]
            if anom_cols:
                ac_col_name = anom_cols[0]

        if ac_col_name is not None:
            ac_raw = joined.column(ac_col_name)
            ac = pc.if_else(
                pc.is_valid(ac_raw),
                pc.cast(ac_raw, pa.int64()),
                pa.scalar(0, type=pa.int64()),
            )
        else:
            ac = pa.array(
                np.zeros(joined.num_rows, dtype=np.int64), type=pa.int64(),
            )

        nc_np = nc.to_numpy(zero_copy_only=False).astype(np.float32)
        ac_np = ac.to_numpy(zero_copy_only=False).astype(np.float32)
        ratio = np.where(nc_np > 0, ac_np / nc_np, float("nan"))

        return pa.table({
            "primary_key": pk,
            "neighbor_count": pa.array(
                nc.to_numpy(zero_copy_only=False), type=pa.int32(),
            ),
            "anomalous_neighbor_count": pa.array(
                ac.to_numpy(zero_copy_only=False), type=pa.int32(),
            ),
            "contagion_ratio": pa.array(ratio, type=pa.float32()),
        })

    def _build_contagion_stats(self, stats_writer: Any) -> None:
        """Precompute per-entity contagion statistics for every pattern that
        has an edge table on disk. Arrow-native — no Python loops.
        """
        import lance as _lance

        for pat_id, pat in self._patterns.items():
            edge_path = self.output_path / "edges" / pat_id / "data.lance"
            if not edge_path.exists():
                continue
            anchor_pat_id = self._resolve_anchor_pattern_for_edge_table(pat_id, pat)
            if anchor_pat_id is None:
                continue
            edges_ds = _lance.dataset(str(edge_path))
            if edges_ds.count_rows() == 0:
                continue
            edges_table = edges_ds.scanner(
                columns=["from_key", "to_key"],
            ).to_table()

            geom_path = (
                self.output_path / "geometry" / anchor_pat_id / "data.lance"
            )
            if not geom_path.exists():
                continue
            geom_table = _lance.dataset(str(geom_path)).scanner(
                columns=["primary_key", "is_anomaly"],
            ).to_table()

            contagion_table = self._compute_contagion_arrow(edges_table, geom_table)
            stats_writer.write_contagion_stats(pat_id, contagion_table)

    def incremental_update(
        self,
        pattern_id: str,
        changed_entities: pa.Table | None = None,
        deleted_keys: list[str] | None = None,
        recalibrate: str = "auto",
        reindex: bool = False,
        recompute_ranks: bool = True,
    ) -> IncrementalUpdateResult:
        """Update geometry incrementally for changed/deleted entities.

        Reads existing sphere.json for mu/sigma/theta and normalizes changed
        entities against existing population statistics. Uses CalibrationTracker
        for drift detection.

        Args:
            pattern_id: Which pattern to update.
            changed_entities: Arrow table with new or modified entities.
                Must have primary_key column matching the pattern's entity schema.
            deleted_keys: Primary keys to remove from geometry.
            recalibrate: "auto" (recalibrate if drift exceeds soft threshold),
                "force" (always recalibrate), or "never".
            reindex: When True, force a rebuild of the ANN (IVF_FLAT) vector
                index so the appended rows are immediately visible to
                ANN-dependent navigation (e.g. π10_attract_trajectory).
                When False (default) the index is rebuilt only once the
                unindexed fraction crosses the standard 10% threshold — except
                that a ``recompute_ranks=True`` call drops the index as a side
                effect (its merge_insert invalidates it), so the default path
                then rebuilds on every call. In the ``recompute_ranks=False``
                deferred path the index is preserved and the 10% threshold
                genuinely gates rebuilds; call ``finalize_incremental`` to force
                a final rebuild at session end.
            recompute_ranks: When True (default), recompute the global
                delta_rank_pct percentile for the whole population on every
                call, so each call is standalone-correct. When False, defer the
                O(N) recompute — existing rows keep their prior (now stale)
                delta_rank_pct until ``finalize_incremental`` is called, while
                new rows carry a batch-local rank. Use False for batched
                ingestion of many small appends, then call
                ``finalize_incremental`` once at the end of the session.
        """
        import lance as _lance

        from hypertopos.builder._writer import _prepare_geometry_for_lance
        from hypertopos.storage.reader import GDSReader
        from hypertopos.storage.writer import GDSWriter, _write_lance

        # 1. Read sphere.json for pattern metadata
        sphere_path = self.output_path / "_gds_meta" / "sphere.json"
        sphere_data = json.loads(sphere_path.read_text())
        pat_meta = sphere_data["patterns"][pattern_id]

        mu = np.array(pat_meta["mu"], dtype=np.float32)
        sigma = np.array(pat_meta["sigma_diag"], dtype=np.float32)
        theta = np.array(pat_meta["theta"], dtype=np.float32)
        theta_norm = float(np.linalg.norm(theta))
        version = pat_meta.get("version", 1)
        relations_meta = pat_meta.get("relations", [])
        event_dims_meta = pat_meta.get("event_dimensions") or None
        dim_weights_raw = pat_meta.get("dimension_weights")
        dim_weights = (
            np.array(dim_weights_raw, dtype=np.float32)
            if dim_weights_raw else None
        )

        # 2. Read calibration tracker
        reader = GDSReader(str(self.output_path))
        tracker = reader.read_calibration_tracker(pattern_id)

        # 3. Resolve geometry Lance path (native MVCC — flat directory).
        lance_path = str(
            self.output_path / "geometry" / pattern_id / "data.lance"
        )

        # 4. Classify changed entity keys
        added_count = 0
        modified_count = 0
        deleted_count = 0
        keys_to_delete: list[str] = list(deleted_keys) if deleted_keys else []

        if changed_entities is not None and len(changed_entities) > 0:
            changed_pks = changed_entities["primary_key"].to_pylist()
            new_keys, mod_keys = _classify_changed_keys(lance_path, changed_pks)
            added_count = len(new_keys)
            modified_count = len(mod_keys)
            # Modified keys need old rows deleted before re-insert
            keys_to_delete.extend(mod_keys)

        # 5. Delete old rows (deleted_keys + modified_keys)
        if keys_to_delete:
            ds = _lance.dataset(lance_path)
            escaped = [k.replace("'", "''") for k in keys_to_delete]
            in_clause = ", ".join(f"'{k}'" for k in escaped)
            ds.delete(f"primary_key IN ({in_clause})")

        deleted_count = len(deleted_keys) if deleted_keys else 0

        # 6. Compute geometry for changed entities
        if changed_entities is not None and len(changed_entities) > 0:
            # Enrich relations_meta with fk_col info from sphere.json
            # Relations in sphere.json use line_id; we need to find the fk_col
            # from the entity table columns. For continuous dims (edge_max set),
            # the fk_col is the dimension_name (derived dim column).
            # For binary, it's the line_id column.
            enriched_relations = []
            for rel in relations_meta:
                enriched = dict(rel)
                line_id = rel["line_id"]
                # Heuristic: check if line_id column exists in entity table
                if line_id in changed_entities.schema.names:
                    enriched["fk_col"] = line_id
                else:
                    # Check for columns that might be FK columns
                    # For derived dims (_d_ prefix), fk_col = display_name or
                    # dimension name (column in entity table)
                    for col_name in changed_entities.schema.names:
                        if col_name == "primary_key":
                            continue
                        # Match by display_name from relation if available
                        if rel.get("display_name") == col_name:
                            enriched["fk_col"] = col_name
                            break
                if "fk_col" not in enriched:
                    enriched["fk_col"] = None
                enriched_relations.append(enriched)

            prop_cols = pat_meta.get("prop_columns", [])
            edge_dim_agg_labels = _edge_dim_aggregation_labels(
                pat_meta.get("edge_dim_aggregations"),
            )
            deltas, delta_norms, shape_vectors = compute_entity_geometry(
                changed_entities, mu, sigma,
                enriched_relations, event_dims_meta, dim_weights,
                prop_columns=prop_cols,
                edge_dim_agg_labels=edge_dim_agg_labels,
            )

            # Hard-guard against a silent width mismatch. compute_entity_geometry
            # supports the relations + event_dimensions + edge_dim_aggregations +
            # prop_columns blocks; a pattern that also carries generalized
            # dimension blocks (geo / metric / semantic) would produce a
            # narrower delta than mu and otherwise fail with a cryptic Lance
            # append error. Surface the unsupported case truthfully instead.
            if deltas.shape[1] != len(mu):
                raise ValueError(
                    f"incremental_update cannot reconstruct full geometry for "
                    f"pattern {pattern_id!r}: computed {deltas.shape[1]}-wide "
                    f"deltas but the pattern mu is {len(mu)}-wide. Incremental "
                    f"ingest supports patterns whose dimensions are relations, "
                    f"event_dimensions, edge_dim_aggregations, and "
                    f"prop_columns; generalized dimension blocks "
                    f"(geo / metric / semantic) are not yet supported and "
                    f"require a full rebuild.",
                )

            # 7. Build geometry rows
            n_new = len(changed_entities)
            is_anomaly = (
                (theta_norm > 0.0) & (delta_norms >= theta_norm)
            )

            # Conformal p-values (simplified — rank within batch)
            sorted_norms = np.sort(delta_norms)
            ranks = np.searchsorted(sorted_norms, delta_norms, side="left")
            conformal_p = ((ranks + 1) / (n_new + 1)).astype(np.float32)
            delta_rank_pcts = (ranks / max(n_new, 1) * 100).astype(np.float32)

            # Per-dimension anomaly count
            n_anom_dims = np.sum(np.abs(deltas) > 2.576, axis=1).astype(np.int32)

            # Build entity_keys (positional list per relation)
            pk_arr = changed_entities["primary_key"].cast(pa.string()).combine_chunks()

            entity_key_lists: list[list[str]] = []
            for i in range(n_new):
                ek = []
                for rel in enriched_relations:
                    fk_col = rel.get("fk_col")
                    if fk_col and fk_col in changed_entities.schema.names:
                        val = changed_entities[fk_col][i].as_py()
                        ek.append(str(val) if val is not None else "")
                    else:
                        ek.append("")
                entity_key_lists.append(ek)

            now = datetime.now(UTC)
            ts_type = pa.timestamp("us", tz="UTC")

            d = deltas.shape[1]
            if d > 0:
                delta_col = pa.FixedSizeListArray.from_arrays(
                    pa.array(deltas.ravel(), type=pa.float32()),
                    list_size=d,
                )
                delta_col = delta_col.cast(pa.list_(pa.float32()))
            else:
                delta_col = pa.array(
                    [[] for _ in range(n_new)], type=pa.list_(pa.float32()),
                )

            geom_cols = {
                "primary_key": pk_arr,
                "scale": pa.array([1] * n_new, type=pa.int32()),
                "delta": delta_col,
                "delta_norm": pa.array(delta_norms, type=pa.float32()),
                "delta_rank_pct": pa.array(delta_rank_pcts, type=pa.float32()),
                "is_anomaly": pa.array(is_anomaly, type=pa.bool_()),
                "conformal_p": pa.array(conformal_p, type=pa.float32()),
                "bregman_divergence": pa.array(
                    self._compute_incremental_bregman(
                        deltas, mu, sigma, pat_meta, n_new,
                    ),
                    type=pa.float32(),
                ),
                "anomaly_confidence": pa.array([None] * n_new, type=pa.float32()),
                "n_anomalous_dims": pa.array(n_anom_dims, type=pa.int32()),
                # ``update_sphere`` does not yet rehydrate the
                # label-aware direction from sphere.json — incremental
                # rows on a label-aware pattern receive null values
                # until pattern.json carries the direction vector.
                "delta_norm_signed": pa.array([None] * n_new, type=pa.float32()),
                "entity_keys": pa.array(entity_key_lists, type=pa.list_(pa.string())),
                "last_refresh_at": pa.array([now] * n_new, type=ts_type),
                "updated_at": pa.array([now] * n_new, type=ts_type),
            }

            pat_type = pat_meta.get("pattern_type", "anchor")
            if pat_type == "event":
                geom_table = pa.table(geom_cols, schema=GEOMETRY_EVENT_SCHEMA)
            else:
                # Anchor patterns need edges column — build simplified edges
                edge_rows: list[list[dict]] = []
                for i in range(n_new):
                    row_edges = []
                    for j_rel, rel in enumerate(enriched_relations):
                        fk_col = rel.get("fk_col")
                        direction = rel.get("direction", "in")
                        ek_row = entity_key_lists[i]
                        point_key = (
                            ek_row[j_rel] if j_rel < len(ek_row) else ""
                        )
                        status = "alive" if point_key else "dead"
                        row_edges.append({
                            "line_id": rel["line_id"],
                            "point_key": point_key,
                            "status": status,
                            "direction": direction,
                        })
                    edge_rows.append(row_edges)
                edges_col = pa.array(edge_rows, type=pa.list_(EDGE_STRUCT_TYPE))
                geom_cols["edges"] = edges_col
                geom_table = pa.table(geom_cols, schema=GEOMETRY_SCHEMA)

            # 8. Prepare and append geometry
            prepared, _list_size = _prepare_geometry_for_lance(geom_table)
            mode = "append" if Path(lance_path).exists() else "create"
            _write_lance(prepared, lance_path, mode=mode)

            # 9. Update CalibrationTracker
            if tracker is not None:
                tracker.update(shape_vectors)
                tracker.update_norms(delta_norms)

        writer = GDSWriter(str(self.output_path))

        # 10. Recompute delta_rank_pct (O(N) global population percentile).
        # delta_rank_pct is a whole-population rank, so adding rows shifts the
        # percentile of every existing row — there is no correct sub-O(N) fast
        # path. recompute_ranks=False defers it (existing rows keep stale ranks
        # until finalize_incremental); the new rows still carry their
        # batch-local rank from step 6. The default recomputes every call so
        # each call is standalone-correct for all delta_rank_pct consumers.
        if recompute_ranks:
            writer.recompute_delta_rank_pct(pattern_id)

        # 10b. Maybe rebuild the ANN index so appended rows are visible to
        # ANN-dependent navigation. Runs AFTER recompute_delta_rank_pct: the
        # rank recompute's merge_insert rewrites matched rows into new
        # fragments, dropping them out of any pre-existing index — reindexing
        # first would leave the index covering zero current rows. reindex=True
        # forces a rebuild; otherwise the 10% unindexed-fraction threshold
        # gates it.
        writer._maybe_reindex_geometry(
            pattern_id, threshold=0.0 if reindex else 0.1, version=version,
        )

        # 11. Compute new population size
        ds = _lance.dataset(lance_path)
        new_pop = ds.count_rows()

        # 12. Auto-recalibrate check
        recalibrated = False
        current_drift = tracker.drift_pct if tracker else 0.0
        if recalibrate == "force" or (
            recalibrate == "auto" and tracker is not None and tracker.is_stale
        ):
            recalibrated = True
            # Full recalibration not implemented inline — mark as stale
            # The caller should trigger a full rebuild for proper recalibration

        # 13. Update sphere.json
        pat_meta["population_size"] = new_pop
        pat_meta["computed_at"] = datetime.now(UTC).isoformat()
        sphere_data["patterns"][pattern_id] = pat_meta
        sphere_path.write_text(json.dumps(sphere_data, indent=2))

        # 14. Rewrite geometry_stats
        all_norms_tbl = ds.to_table(columns=["delta_norm"])
        all_norms = all_norms_tbl["delta_norm"].to_numpy(zero_copy_only=False)
        writer.write_geometry_stats(
            pattern_id, version=version,
            delta_norms=all_norms, theta_norm=theta_norm,
        )

        # 15. Persist updated calibration tracker
        if tracker is not None:
            writer.write_calibration_tracker(pattern_id, tracker)

        return IncrementalUpdateResult(
            pattern_id=pattern_id,
            added=added_count,
            modified=modified_count,
            deleted=deleted_count,
            drift_pct=current_drift,
            recalibrated=recalibrated,
            theta_norm=theta_norm,
            population_size=new_pop,
        )

    def finalize_incremental(self, pattern_id: str) -> None:
        """Finalize a batched incremental-ingest session for one pattern.

        Recomputes the global ``delta_rank_pct`` percentile once across the
        whole population (making every row standalone-correct again after a run
        of ``incremental_update(recompute_ranks=False)`` calls) and rebuilds the
        ANN (IVF_FLAT) vector index so all appended rows are indexed. Call once
        at the end of an ingestion session instead of paying the O(N) rank
        recompute on every individual append.

        Idempotent and safe to call after ``recompute_ranks=True`` updates too.
        """
        from hypertopos.storage.writer import GDSWriter

        writer = GDSWriter(str(self.output_path))
        writer.recompute_delta_rank_pct(pattern_id)
        # Reindex AFTER the rank recompute — the recompute's merge_insert
        # rewrites matched rows into new fragments, dropping them out of any
        # pre-existing index.
        writer._maybe_reindex_geometry(pattern_id, threshold=0.0, version=1)

    @staticmethod
    def _compute_incremental_bregman(
        deltas: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        pat_meta: dict,
        n: int,
    ) -> np.ndarray:
        """Compute Bregman norms for incrementally-added entities.

        Reconstructs shape vectors from deltas and the stored mu/sigma,
        then scores against the pattern's dimension_kinds. Returns zeros
        when dimension_kinds is unavailable (pre-0.4 spheres).
        """
        dim_kinds = pat_meta.get("dimension_kinds")
        if dim_kinds is not None:
            from hypertopos.builder._bregman import bregman_norms as _bregman_norms_fn

            shapes = (deltas * sigma + mu).astype(np.float32)
            return _bregman_norms_fn(shapes, mu, sigma, dim_kinds).astype(np.float32)
        return np.zeros(n, dtype=np.float32)

    def _build_and_write_chunked(
        self,
        pat_id: str,
        pat: _PatternReg,
        stats_writer: Any,
        write_chunk_fn: Any,
        finalize_fn: Any,
    ) -> tuple[str, PatternBuildResult]:
        """Chunked geometry build: compute stats once, write in chunks.

        Bounds peak RAM to O(chunk_size) instead of O(N) for the geometry
        table, while mu/sigma/theta are computed from the full population.
        """
        n = len(self._lines[pat.entity_line].table)

        # 1. Compute population stats
        ps = self._compute_population_stats(pat)
        # Label-aware calibration — same hook as the in-memory path. Must
        # run BEFORE the per-chunk _build_geometry_slice loop so the
        # registered direction is available when each chunk projects its
        # deltas to delta_norm_signed.
        lac = self._run_label_aware_calibration(pat, ps)

        theta_norm = float(np.linalg.norm(ps.theta))

        # 1b. Bootstrap confidence (chunked path keeps full deltas in memory)
        # Skip when: use_mahalanobis, population > _BOOTSTRAP_MAX_N,
        # group_by_property, or bootstrap_iterations=0
        confidence_arr: np.ndarray | None = None
        if (
            pat.bootstrap_iterations > 0
            and ps.dimension_kinds is not None
            and not pat.use_mahalanobis
            and not pat.group_by_property
            and n <= _BOOTSTRAP_MAX_N
        ):
            raw_deltas = ps.deltas
            if ps.dim_weights is not None:
                raw_deltas = (raw_deltas / ps.dim_weights).astype(np.float32)
            shape_vectors = (raw_deltas * ps.sigma + ps.mu).astype(np.float32)

            from hypertopos.builder._bootstrap import (
                compute_bootstrap_confidence,
            )

            confidence_arr = compute_bootstrap_confidence(
                shape_vectors=shape_vectors,
                kinds=ps.dimension_kinds,
                anomaly_percentile=pat.anomaly_percentile,
                B=pat.bootstrap_iterations,
                weights=ps.dim_weights,
                seed=42,
            )

        # 2. Write geometry in chunks
        for start in range(0, n, GEOMETRY_CHUNK_SIZE):
            end = min(start + GEOMETRY_CHUNK_SIZE, n)
            chunk_table = self._build_geometry_slice(
                pat, start, end,
                ps.deltas, ps.delta_norms, ps.delta_rank_pcts,
                theta_norm, ps.fk_arrays, ps.conformal_p,
                ps.is_anomaly_arr, ps.n_anom_dims,
                bregman_norms_arr=ps.bregman_norms,
                anomaly_confidence_arr=confidence_arr,
            )
            chunk_table = self._inject_fdr_hierarchy_carriers(pat, chunk_table)
            chunk_table = self._inject_fdr_temporal_buckets(pat, chunk_table)
            _validate_fdr_hierarchy_columns(
                pat, list(chunk_table.column_names),
            )
            write_chunk_fn(
                self.output_path, pat_id, chunk_table, version=1,
            )

        # 3. Finalize: compact fragments, build indices
        finalize_fn(self.output_path, pat_id, version=1)

        # 4. Persist geometry stats cache
        stats_writer.write_geometry_stats(
            pat_id, version=1,
            delta_norms=ps.delta_norms, theta_norm=theta_norm,
            is_anomaly_arr=ps.is_anomaly_arr,
        )

        # 5. Emit edge table (chunked path)
        edge_cfg = self._resolve_edge_table_config(pat)
        if edge_cfg is not None:
            edge_table = self._extract_edge_table(pat, edge_cfg)
            if edge_table.num_rows > 0:
                stats_writer.write_edges(pat_id, edge_table)
                # Cache edge stats
                import pyarrow.compute as _pc
                stats_writer.write_edge_stats(pat_id, {
                    "row_count": edge_table.num_rows,
                    "unique_from": _pc.count_distinct(edge_table["from_key"]).as_py(),
                    "unique_to": _pc.count_distinct(edge_table["to_key"]).as_py(),
                    "timestamp_min": float(_pc.min(edge_table["timestamp"]).as_py()),
                    "timestamp_max": float(_pc.max(edge_table["timestamp"]).as_py()),
                    "amount_min": float(_pc.min(edge_table["amount"]).as_py()),
                    "amount_max": float(_pc.max(edge_table["amount"]).as_py()),
                })

        return pat_id, PatternBuildResult(
            mu=ps.mu, sigma=ps.sigma, theta=ps.theta,
            population_size=n,
            prop_columns=ps.prop_columns,
            excluded_properties=ps.excluded_properties,
            group_stats=ps.group_stats_dict,
            dimension_weights=ps.dim_weights,
            gmm_components=ps.gmm_components,
            cholesky_inv=ps.cholesky_inv,
            dim_percentiles=self._compute_dim_percentiles(
                pat.entity_line,
                edge_dim_agg_matrix=ps.edge_dim_agg_matrix,
                edge_dim_agg_labels=ps.edge_dim_agg_labels,
            ),
            dimension_kinds=ps.dimension_kinds,
            dim_block_names=ps.dim_block_names,
            dim_block_stats=ps.dim_block_stats,
            theta_sensitivity=ps.theta_sensitivity,
            heteroscedasticity_diagnostic=ps.heteroscedasticity_diagnostic,
            dim_normality_pvalues=self._compute_dim_normality_pvalues(
                pat.entity_line,
                edge_dim_agg_matrix=ps.edge_dim_agg_matrix,
                edge_dim_agg_labels=ps.edge_dim_agg_labels,
            ),
            edge_dim_names=ps.edge_dim_names,
            **self._unpack_lac_bundle(lac, ps.deltas),
        )

    def _build_shape_chunk(
        self,
        pat: _PatternReg,
        chunk_table: pa.Table,
    ) -> tuple[np.ndarray, list]:
        """Build shape vectors and FK arrays for a chunk of entity table.

        Returns:
            shapes: (cn, D_rel + D_event) float32 — shape vectors for this chunk.
            fk_slices: list of (pa.ChunkedArray | np.ndarray | None) per relation,
                each sized to the chunk.
        """
        cn = len(chunk_table)
        D_rel = len(pat.relations)
        D_event = len(pat.event_dimensions)
        shapes = np.zeros((cn, D_rel + D_event), dtype=np.float32)
        fk_slices: list[pa.ChunkedArray | np.ndarray | None] = []

        for j, rel in enumerate(pat.relations):
            if rel.direction == "self":
                shapes[:, j] = 1.0
                fk_slices.append(None)
            elif rel.edge_max is not None:
                col = chunk_table[rel.fk_col]
                count_arr = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)
                shapes[:, j] = np.clip(count_arr, 0, rel.edge_max) / rel.edge_max
                fk_slices.append(count_arr)
            else:
                col_arrow = chunk_table[rel.fk_col]
                fk_slices.append(col_arrow)
                valid_mask = pc.fill_null(
                    pc.and_(
                        pc.is_valid(col_arrow),
                        pc.not_equal(col_arrow, ""),
                    ),
                    False,
                )
                shapes[:, j] = valid_mask.to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)

        # Event dimensions — continuous values from entity columns
        for k, edim in enumerate(pat.event_dimensions):
            col = chunk_table[edim.column]
            raw_arr = pc.fill_null(col, 0).to_numpy(
                zero_copy_only=False,
            ).astype(np.float32)
            em = edim.edge_max
            if isinstance(em, (int, float)) and em > 0:
                shapes[:, D_rel + k] = np.clip(raw_arr / em, 0.0, 3.0)
            else:
                raise ValueError(
                    f"EventDimSpec '{edim.column}': edge_max must be a "
                    f"positive number at build time, got {em!r}. "
                    f"Use edge_max='auto' or call add_event_dimension()."
                )

        return shapes, fk_slices

    def _build_prop_fill_chunk(
        self,
        pat: _PatternReg,
        chunk_table: pa.Table,
        prop_columns: list[str],
    ) -> np.ndarray:
        """Build property fill matrix for a chunk. Returns (cn, n_props) float32."""
        cn = len(chunk_table)
        if not prop_columns:
            return np.empty((cn, 0), dtype=np.float32)

        schema_names = set(chunk_table.schema.names)
        fill_matrix = np.zeros((cn, len(prop_columns)), dtype=np.float32)
        for j, prop in enumerate(prop_columns):
            if prop not in schema_names:
                continue
            col = chunk_table[prop]
            fill_vec = pc.is_valid(col).to_numpy(
                zero_copy_only=False,
            ).astype(np.float32)
            fill_matrix[:, j] = fill_vec
        return fill_matrix

    def _build_and_write_streaming(
        self,
        pat_id: str,
        pat: _PatternReg,
        stats_writer: Any,
        write_chunk_fn: Any,
        finalize_fn: Any,
    ) -> tuple[str, PatternBuildResult]:
        """Three-pass streaming geometry build with O(chunk) peak RAM.

        Only used for the simple case: no group_by_property, no GMM,
        no Mahalanobis. Complex modes fall back to _build_and_write_chunked.

        Pass 1: Welford streaming for mu/sigma, fill rates for props,
                raw moments for dim_weights if auto.
        Pass 2: Compute final norms into O(N) float32 array + reservoir
                sampling for per-dim anomaly thresholds.
        Pass 3: Write geometry chunks using final population stats.
        """
        from hypertopos.builder._stats import (
            SIGMA_EPS,
            SIGMA_EPS_PROP,
            compute_conformal_p,
            reservoir_update,
            welford_batch_update,
        )

        # Streaming path never materialises the full delta matrix, so
        # label-aware calibration (which needs every row to fit Fisher
        # LDA) silently degrades to "no calibration". Warn loudly when
        # the pattern was selected for label-aware calibration so the
        # user can fall back to a non-streaming build shape.
        if (
            self._label_aware_calibration
            and self._label_audit_block is not None
            and pat.pattern_id in set(self._label_audit_block.patterns)
        ):
            logger.warning(
                "label-aware calibration skipped for pattern %r — "
                "streaming build path does not materialise the full "
                "delta matrix; rebuild with a non-streaming pattern "
                "shape (avoid group_by_property / gmm / mahalanobis "
                "OR reduce population below GEOMETRY_CHUNK_SIZE) to "
                "produce delta_norm_signed and label_aware_calibration",
                pat.pattern_id,
            )

        entity_line = self._lines[pat.entity_line]
        entity_table = entity_line.table
        n = len(entity_table)
        D_rel = len(pat.relations)
        D_event = len(pat.event_dimensions)
        chunk_size = GEOMETRY_CHUNK_SIZE

        # ── Edge-derived dims — precompute matrix + sidecar once ──
        # Streaming path doesn't go through _compute_population_stats, so we
        # bake the edge_dim values here and slice them into each chunk's
        # shape vector below.
        edge_dim_matrix = np.empty((n, 0), dtype=np.float32)
        edge_dim_names: list[str] = []
        edge_dim_kinds: list[str] = []
        if (
            pat.edge_dimensions is not None
            and pat.pattern_type == "event"
            and getattr(pat.edge_dimensions, "dims", None)
        ):
            from hypertopos.engine.edge_features import (
                EDGE_DIM_KINDS,
                compute_all_edge_dims,
            )

            edge_cfg = pat.edge_table
            if edge_cfg is None:
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} declares edge_dimensions "
                    f"but no edge_table — edge_dimensions require an "
                    f"edge_table block on the same pattern.",
                )
            edges_tbl = self._extract_edge_table(pat, edge_cfg)

            dims_cfg = dict(pat.edge_dimensions.dims)
            tsl = dims_cfg.get("time_since_pair_last_edge")
            if (
                tsl is not None
                and isinstance(tsl, dict)
                and tsl.get("dormant_seconds") == "auto"
            ):
                if edges_tbl.num_rows > 0:
                    ts = edges_tbl["timestamp"].to_numpy()
                    span = float(ts.max() - ts.min())
                    span = max(span, 1.0)
                else:
                    span = 1.0
                dims_cfg["time_since_pair_last_edge"] = {
                    **tsl, "dormant_seconds": span,
                }

            features = compute_all_edge_dims(edges_tbl, dims_cfg)
            edge_dim_names = [
                c for c in features.column_names if c != "event_key"
            ]
            edge_dim_kinds = [EDGE_DIM_KINDS[name] for name in edge_dim_names]
            if edge_dim_names and edges_tbl.num_rows > 0:
                pk_to_idx = {
                    pk: i
                    for i, pk in enumerate(
                        entity_table["primary_key"].to_pylist(),
                    )
                }
                edge_dim_matrix = np.zeros(
                    (n, len(edge_dim_names)), dtype=np.float32,
                )
                event_keys = features["event_key"].to_pylist()
                for col_idx, name in enumerate(edge_dim_names):
                    vals = features[name].to_numpy()
                    for row_idx, ek in enumerate(event_keys):
                        ent_idx = pk_to_idx.get(ek)
                        if ent_idx is not None:
                            edge_dim_matrix[ent_idx, col_idx] = vals[row_idx]
            elif edge_dim_names:
                edge_dim_matrix = np.zeros(
                    (n, len(edge_dim_names)), dtype=np.float32,
                )

            if edge_dim_names:
                try:
                    import lance
                    sidecar_dir = (
                        self.output_path
                        / "_gds_meta" / "edge_features" / pat.pattern_id
                    )
                    sidecar_dir.mkdir(parents=True, exist_ok=True)
                    lance.write_dataset(
                        features,
                        str(sidecar_dir / "data.lance"),
                        mode="overwrite",
                    )
                except Exception as exc:
                    logger.warning(
                        "edge_features sidecar write failed for %s: %s",
                        pat.pattern_id, exc,
                    )

        D_edge = len(edge_dim_names)

        # ── Anchor-pattern edge-dim aggregation (S1 ext, 0.6.1) ──
        edge_dim_agg_matrix = np.empty((n, 0), dtype=np.float32)
        edge_dim_agg_kinds: list[str] = []
        edge_dim_agg_labels: list[str] = []
        if (
            pat.edge_dim_aggregations is not None
            and pat.pattern_type == "anchor"
        ):
            from hypertopos.engine.edge_features import (
                AGGREGATE_NAMES,
                EDGE_DIM_KINDS,
                aggregate_edge_dims_for_anchor,
                aggregate_kind,
            )

            cfg = pat.edge_dim_aggregations
            src_pat = self._patterns.get(cfg.from_event_pattern)
            if src_pat is None or src_pat.pattern_type != "event":
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} edge_dim_aggregations.from "
                    f"={cfg.from_event_pattern!r} must reference an event "
                    f"pattern in this build",
                )
            if src_pat.edge_table is None:
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} edge_dim_aggregations.from "
                    f"={cfg.from_event_pattern!r} has no edge_table",
                )
            sidecar_path = (
                self.output_path / "_gds_meta" / "edge_features"
                / cfg.from_event_pattern / "data.lance"
            )
            if not sidecar_path.exists():
                raise ValueError(
                    f"Pattern {pat.pattern_id!r} edge_dim_aggregations expects "
                    f"sidecar at {sidecar_path}; declare edge_dimensions: on "
                    f"{cfg.from_event_pattern!r} and order it BEFORE this "
                    f"anchor pattern in YAML so it is built first",
                )
            import lance
            sidecar_tbl = lance.dataset(str(sidecar_path)).to_table()
            avail = [c for c in sidecar_tbl.column_names if c != "event_key"]
            agg_dims = list(cfg.dims) if cfg.dims is not None else avail
            src_edges = self._extract_edge_table(src_pat, src_pat.edge_table)

            src_lines = {r.line_id for r in src_pat.relations}
            composite_match = next(
                (cs for cs in self._composite_lines
                 if cs.line_id == pat.entity_line),
                None,
            )
            chain_events_col: list[str] | None = None
            composite_key_cols: list[str] | None = None
            if pat.entity_line in src_lines:
                anchor_kind = "single"
                pair_separator = "→"
            elif composite_match is not None:
                anchor_kind = "pair"
                pair_separator = composite_match.separator
                composite_key_cols = list(composite_match.key_cols)
                # Convention: first two key_cols positionally map to the
                # source event_pattern's edge_table endpoints (renamed in
                # _extract_edge_table to from_key / to_key). Reject k>=3
                # composite anchors that violate this — silent mismatch
                # would produce all-zero aggregates because the anchor PK
                # built from event_table.{key_cols[0],key_cols[1],...} would
                # not match the engine's PK constructed from
                # edges.{from_key, to_key, key_cols[2:]}.
                if len(composite_key_cols) > 2 and src_pat.edge_table is not None:
                    expected_from = src_pat.edge_table.from_col
                    expected_to = src_pat.edge_table.to_col
                    if (
                        composite_key_cols[0] != expected_from
                        or composite_key_cols[1] != expected_to
                    ):
                        raise ValueError(
                            f"Pattern {pat.pattern_id!r}: composite_line "
                            f"{composite_match.line_id!r} declares "
                            f"key_cols={composite_key_cols!r} but "
                            f"edge_dim_aggregations on a k>=3 composite "
                            f"anchor requires key_cols[0:2] to positionally "
                            f"match the source event pattern "
                            f"{cfg.from_event_pattern!r} edge_table endpoints "
                            f"(from_col={expected_from!r}, "
                            f"to_col={expected_to!r}). Property columns "
                            f"key_cols[2:] can be any event_table column.",
                        )
            elif pat.entity_line in self._chain_lines:
                # Chain regime — entity_line registered as chain via chain_lines: block.
                # event_line consistency was validated at parse time (cli/schema.py).
                # Zero-chain extraction is caught earlier at _validate() with a
                # chain-specific message; by the time we reach the dispatch the
                # entity_table is guaranteed to have rows AND a chain_events column
                # populated by `add_chain_line`.
                anchor_kind = "chain"
                pair_separator = "→"  # unused in chain regime
                chain_events_col = entity_table["chain_events"].to_pylist()
            else:
                raise NotImplementedError(
                    f"Pattern {pat.pattern_id!r}: edge_dim_aggregations "
                    f"could not resolve anchor regime — entity_line "
                    f"{pat.entity_line!r} is neither a relation of the "
                    f"source event pattern {cfg.from_event_pattern!r} "
                    f"(single-key regime), nor a registered composite_line "
                    f"(pair / k>2 regime), nor a registered chain_line. "
                    f"Supported regimes: single, pair, chain.",
                )

            primary_keys = entity_table["primary_key"].to_pylist()
            edge_dim_thresholds_resolved = (
                self._resolve_and_persist_edge_dim_thresholds(
                    pat.pattern_id, sidecar_tbl, agg_dims,
                )
            )
            aggregates_per_dim = cfg.aggregates_per_dim
            extra = aggregate_edge_dims_for_anchor(
                anchor_keys=primary_keys,
                edges=src_edges,
                sidecar=sidecar_tbl,
                dims=agg_dims,
                anchor_kind=anchor_kind,
                pair_separator=pair_separator,
                chain_events=chain_events_col,
                key_cols=composite_key_cols,
                event_table=self._lines[src_pat.entity_line].table,
                thresholds=edge_dim_thresholds_resolved,
                aggregates_per_dim=aggregates_per_dim,
            )
            n_cols = sum(len(aggregates_per_dim[d]) for d in agg_dims)
            edge_dim_agg_matrix = np.zeros((n, n_cols), dtype=np.float32)
            col_idx = 0
            for d in agg_dims:
                src_kind = EDGE_DIM_KINDS[d]
                for agg in aggregates_per_dim[d]:
                    edge_dim_agg_matrix[:, col_idx] = (
                        extra[f"{d}_{agg}"].to_numpy()
                    )
                    edge_dim_agg_kinds.append(aggregate_kind(src_kind, agg))
                    edge_dim_agg_labels.append(f"{d}_{agg}")
                    col_idx += 1

        D_edge_agg = edge_dim_agg_matrix.shape[1]

        # Pre-compute event dimension edge_max (needs full column scan)
        for edim in pat.event_dimensions:
            if edim.edge_max is None or edim.edge_max == "auto":
                col = entity_table[edim.column]
                raw_arr = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False,
                ).astype(np.float32)
                positive = raw_arr[raw_arr > 0]
                computed = (
                    float(np.percentile(positive, edim.percentile))
                    if len(positive) > 0 else 1.0
                )
                edim.edge_max = max(computed, 1e-9)

        # ── Pass 1: Welford streaming for mu/sigma + prop fill rates ──

        # Determine tracked properties and their fill rates
        tracked = pat.tracked_properties if pat.pattern_type == "anchor" else []
        schema_names = set(entity_table.schema.names)

        # Accumulate fill rates for tracked properties across chunks
        prop_fill_sums = np.zeros(len(tracked), dtype=np.float64) if tracked else None

        # Welford accumulators — we don't know D_full yet (depends on prop_columns)
        # So first pass accumulates edge dims + event dims + ALL tracked prop candidates,
        # then we trim after determining which props pass MIN_FILL_RATE.
        D_max = D_rel + D_event + D_edge + D_edge_agg + len(tracked)
        w_mean = np.zeros(D_max, dtype=np.float64)
        w_m2 = np.zeros(D_max, dtype=np.float64)
        w_n = 0

        # For dim_weights="auto": accumulate raw 4th moments
        need_kurtosis = pat.dimension_weights in ("auto", "kurtosis")
        kurt_sum_z4 = np.zeros(D_max, dtype=np.float64) if need_kurtosis else None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_table = entity_table.slice(start, end - start)
            shapes, _ = self._build_shape_chunk(pat, chunk_table)

            # Append edge_dim slice
            if D_edge > 0:
                shapes = np.concatenate(
                    [shapes, edge_dim_matrix[start:end]], axis=1,
                )

            # Append edge_dim aggregations slice (S1 ext)
            if D_edge_agg > 0:
                shapes = np.concatenate(
                    [shapes, edge_dim_agg_matrix[start:end]], axis=1,
                )

            # Prop fill for this chunk (all tracked, not yet filtered)
            if tracked:
                cn = len(chunk_table)
                prop_fill = np.zeros((cn, len(tracked)), dtype=np.float32)
                for j, prop in enumerate(tracked):
                    if prop not in schema_names:
                        continue
                    col = chunk_table[prop]
                    fill_vec = pc.is_valid(col).to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float32)
                    prop_fill[:, j] = fill_vec
                    prop_fill_sums[j] += fill_vec.sum()

                full_chunk = np.concatenate([shapes, prop_fill], axis=1)
            else:
                full_chunk = shapes

            w_mean, w_m2, w_n = welford_batch_update(w_mean, w_m2, w_n, full_chunk)

        # Finalize mu/sigma for edge dims
        mu_full = w_mean.astype(np.float32)
        sigma_full = np.sqrt(w_m2 / max(w_n, 1)).astype(np.float32)
        sigma_full = np.maximum(sigma_full, SIGMA_EPS)

        # Determine prop_columns from fill rates
        prop_columns: list[str] = []
        excluded_properties: list[str] = []
        prop_indices: list[int] = []  # indices into tracked list

        if tracked:
            fill_rates = prop_fill_sums / n
            for j, prop in enumerate(tracked):
                if fill_rates[j] < MIN_FILL_RATE:
                    excluded_properties.append(prop)
                    logger.info(
                        "Excluding '%s': fill_rate=%.3f < MIN_FILL_RATE",
                        prop, fill_rates[j],
                    )
                elif fill_rates[j] >= MAX_FILL_RATE and not _is_textual_or_binary_col(
                    chunk_table.schema.field(prop)
                ):
                    excluded_properties.append(prop)
                    logger.info(
                        "Excluding '%s': fill_rate=%.3f >= MAX_FILL_RATE (zero-variance)",
                        prop, fill_rates[j],
                    )
                else:
                    prop_columns.append(prop)
                    prop_indices.append(j)

        # Trim mu/sigma to included prop dims only
        # Dims order: [relations] + [event dims] + [edge dims] + [included props]
        D_base = D_rel + D_event + D_edge
        if prop_columns:
            keep_dims = list(range(D_base)) + [D_base + j for j in prop_indices]
            mu = mu_full[keep_dims]
            sigma = sigma_full[keep_dims]
        else:
            keep_dims = list(range(D_base))
            mu = mu_full[:D_base]
            sigma = sigma_full[:D_base]

        D_full = len(mu)

        # Apply SIGMA_EPS_PROP for binary prop columns.
        # In the streaming path D_base = n_rel + n_event + n_dim_block, and
        # sigma[D_base:] covers only prop_columns (no additional dim blocks
        # are appended after props in this path).
        if prop_columns:
            sigma[D_base:] = np.maximum(sigma[D_base:], SIGMA_EPS_PROP)

        # Compute dimension weights if auto
        dim_weights: np.ndarray | None = None
        if need_kurtosis and D_full > 0:
            # Second mini-pass for kurtosis on final dims (reuse Welford sigma)
            # Kurtosis = E[(x-mu)^4] / sigma^4 - 3
            # We need to accumulate z^4 in a streaming pass.
            # Since we need the final mu/sigma, we do a second scan.
            kurt_sum_z4 = np.zeros(D_full, dtype=np.float64)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_table = entity_table.slice(start, end - start)
                shapes, _ = self._build_shape_chunk(pat, chunk_table)
                if D_edge > 0:
                    shapes = np.concatenate(
                        [shapes, edge_dim_matrix[start:end]], axis=1,
                    )
                if D_edge_agg > 0:
                    shapes = np.concatenate(
                        [shapes, edge_dim_agg_matrix[start:end]], axis=1,
                    )
                if prop_columns:
                    prop_fill = self._build_prop_fill_chunk(
                        pat, chunk_table, prop_columns,
                    )
                    full_chunk = np.concatenate([shapes, prop_fill], axis=1)
                else:
                    full_chunk = shapes
                z = ((full_chunk - mu) / sigma).astype(np.float64)
                kurt_sum_z4 += (z ** 4).sum(axis=0)

            kurt_mean_z4 = kurt_sum_z4 / n
            # excess_kurtosis = mean(z^4) - 3; weight = max(1.0, (kurt+3)/3)
            dim_weights = np.maximum(
                1.0, kurt_mean_z4 / 3.0,
            ).astype(np.float32)
            logger.info(
                "Auto-computed dimension weights for %s: %s",
                pat.pattern_id, dim_weights.tolist(),
            )
        elif isinstance(pat.dimension_weights, list):
            dim_weights = np.array(pat.dimension_weights, dtype=np.float32)
            if len(dim_weights) != D_full:
                raise ValueError(
                    f"dimension_weights length ({len(dim_weights)}) != "
                    f"shape dimensions ({D_full})"
                )

        # ── Detect dimension kinds (needed for Bregman norms in Pass 3) ──
        from hypertopos.builder._bregman import (
            bregman_norms as _bregman_norms_fn,
            detect_kind_for_column as _detect_kind_for_column,
            format_kinds_summary,
        )

        # Build lookup maps for derived/precomputed dims (same as in-memory path)
        _derived_metric_map_s: dict[str, str] = {}
        for dspec in self._derived_dims:
            _derived_metric_map_s[dspec.dimension_name] = dspec.metric

        _precomputed_map_s: dict[str, int | float | str | None] = {}
        for pspec in self._precomputed_dims:
            _precomputed_map_s[pspec.dimension_name] = pspec.edge_max

        _POISSON_METRICS_S = {"count", "count_distinct"}

        dimension_kinds: list[str] = []
        for rel in pat.relations:
            dim_name = rel.fk_col
            if dim_name in _derived_metric_map_s:
                metric = _derived_metric_map_s[dim_name]
                base_metric = metric.split(":")[0]
                dimension_kinds.append(
                    "poisson" if base_metric in _POISSON_METRICS_S else "gaussian",
                )
            elif dim_name in _precomputed_map_s:
                em = _precomputed_map_s[dim_name]
                dimension_kinds.append(
                    "bernoulli" if isinstance(em, (int, float)) and int(em) == 1 else "gaussian"
                )
            else:
                # Regular FK or graph feature relation.
                # edge_max=1 → 0-1 ratio (bernoulli), edge_max>1 → count (poisson),
                # None → binary FK (bernoulli).
                em = rel.edge_max
                if isinstance(em, (int, float)) and int(em) == 1:
                    dimension_kinds.append("bernoulli")
                elif em is not None and em != "auto":
                    dimension_kinds.append("poisson")
                else:
                    dimension_kinds.append("bernoulli")

        # Event dimensions
        for edim in pat.event_dimensions:
            kind_override = getattr(edim, "kind", None)
            if kind_override:
                dimension_kinds.append(kind_override)
            else:
                col = entity_table[edim.column]
                vals = pc.fill_null(col, 0).to_numpy(
                    zero_copy_only=False,
                ).astype(np.float64)
                dimension_kinds.append(_detect_kind_for_column(vals))

        # Edge-derived dimensions
        dimension_kinds.extend(edge_dim_kinds)

        # Edge-derived aggregations on anchor patterns (S1 ext)
        dimension_kinds.extend(edge_dim_agg_kinds)

        # Prop fill (binary 0/1)
        dimension_kinds.extend(["bernoulli"] * len(prop_columns))

        if len(dimension_kinds) != D_full:
            logger.warning(
                "Bregman kind count (%d) != shape dim count (%d) for %s — "
                "falling back to uniform gaussian kinds",
                len(dimension_kinds), D_full, pat.pattern_id,
            )
            dimension_kinds = ["gaussian"] * D_full

        logger.info(
            "Bregman kinds for %s (streaming): %s (D=%d)",
            pat.pattern_id, format_kinds_summary(dimension_kinds), D_full,
        )

        # ── Pass 2: Compute all norms (O(N) float32) + reservoir ──

        RESERVOIR_K = 10_000
        all_norms = np.zeros(n, dtype=np.float32)
        reservoir = np.zeros(
            (min(RESERVOIR_K, n), D_full), dtype=np.float32,
        )
        reservoir_count = 0
        rng = np.random.default_rng(42)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_table = entity_table.slice(start, end - start)
            shapes, _ = self._build_shape_chunk(pat, chunk_table)
            if D_edge > 0:
                shapes = np.concatenate(
                    [shapes, edge_dim_matrix[start:end]], axis=1,
                )
            if D_edge_agg > 0:
                shapes = np.concatenate(
                    [shapes, edge_dim_agg_matrix[start:end]], axis=1,
                )
            if prop_columns:
                prop_fill = self._build_prop_fill_chunk(
                    pat, chunk_table, prop_columns,
                )
                full_chunk = np.concatenate([shapes, prop_fill], axis=1)
            else:
                full_chunk = shapes

            chunk_deltas = ((full_chunk - mu) / sigma).astype(np.float32)
            if dim_weights is not None:
                chunk_deltas = (chunk_deltas * dim_weights).astype(np.float32)
            chunk_norms = np.sqrt(
                np.einsum('ij,ij->i', chunk_deltas, chunk_deltas),
            ).astype(np.float32)
            all_norms[start:end] = chunk_norms

            # Reservoir sampling of absolute deltas for per-dim thresholds
            abs_deltas = np.abs(chunk_deltas)
            reservoir_count = reservoir_update(
                reservoir, reservoir_count, abs_deltas, rng,
            )

        # Compute theta from percentile of all norms
        theta_scalar = float(np.percentile(all_norms, pat.anomaly_percentile))
        component_val = theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
        theta = np.full(D_full, component_val, dtype=np.float32)
        theta_norm = float(np.linalg.norm(theta))

        # Delta rank pcts from sorted norms
        sorted_norms = np.sort(all_norms)
        ranks = np.searchsorted(sorted_norms, all_norms, side="left")
        delta_rank_pcts = (ranks / n * 100).astype(np.float32)

        # Conformal p-values
        conformal_p = compute_conformal_p(all_norms)

        # Per-dim anomaly thresholds from reservoir
        actual_reservoir = reservoir[:min(reservoir_count, RESERVOIR_K)]
        if len(actual_reservoir) > 0:
            per_dim_thresholds = np.percentile(
                actual_reservoir, 99.0, axis=0,
            ).astype(np.float32)
        else:
            per_dim_thresholds = np.zeros(D_full, dtype=np.float32)

        # ── Pass 3: Write geometry chunks ──
        entity_table_full = entity_line.table
        all_is_anomaly = np.empty(n, dtype=bool)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_entity_table = entity_table_full.slice(
                start, end - start,
            )
            cn = end - start
            shapes, fk_slices = self._build_shape_chunk(
                pat, chunk_entity_table,
            )
            if D_edge > 0:
                shapes = np.concatenate(
                    [shapes, edge_dim_matrix[start:end]], axis=1,
                )
            if D_edge_agg > 0:
                shapes = np.concatenate(
                    [shapes, edge_dim_agg_matrix[start:end]], axis=1,
                )
            if prop_columns:
                prop_fill = self._build_prop_fill_chunk(
                    pat, chunk_entity_table, prop_columns,
                )
                full_chunk = np.concatenate(
                    [shapes, prop_fill], axis=1,
                )
            else:
                full_chunk = shapes

            chunk_deltas = (
                (full_chunk - mu) / sigma
            ).astype(np.float32)
            if dim_weights is not None:
                chunk_deltas = (
                    chunk_deltas * dim_weights
                ).astype(np.float32)
            chunk_norms = all_norms[start:end]
            chunk_ranks = delta_rank_pcts[start:end]
            chunk_conformal = conformal_p[start:end]

            is_anomaly_arr = (
                (theta_norm > 0.0) & (chunk_norms >= theta_norm)
            )
            all_is_anomaly[start:end] = is_anomaly_arr

            abs_chunk_deltas = np.abs(chunk_deltas)
            exceeds = (
                abs_chunk_deltas > per_dim_thresholds
            ).astype(np.int32)
            chunk_n_anom_dims = exceeds.sum(axis=1).astype(np.int32)

            chunk_bregman: np.ndarray | None = None
            if dimension_kinds is not None:
                chunk_bregman = _bregman_norms_fn(
                    full_chunk, mu, sigma, dimension_kinds,
                    weights=dim_weights,
                ).astype(np.float32)

            geom_table = self._build_geometry_slice(
                pat, start=0, end=cn,
                deltas=chunk_deltas,
                delta_norms=chunk_norms,
                delta_rank_pcts=chunk_ranks,
                theta_norm=theta_norm,
                fk_arrays=fk_slices,
                conformal_p=chunk_conformal,
                is_anomaly_precomputed=is_anomaly_arr,
                n_anom_dims=chunk_n_anom_dims,
                bregman_norms_arr=chunk_bregman,
                anomaly_confidence_arr=None,
                entity_table_override=chunk_entity_table,
            )
            geom_table = self._inject_fdr_hierarchy_carriers(pat, geom_table)
            geom_table = self._inject_fdr_temporal_buckets(pat, geom_table)
            _validate_fdr_hierarchy_columns(
                pat, list(geom_table.column_names),
            )

            write_chunk_fn(
                self.output_path, pat_id, geom_table, version=1,
            )

        # Finalize: compact fragments, build indices
        finalize_fn(self.output_path, pat_id, version=1)

        # Persist geometry stats cache — use in-memory is_anomaly from Pass 3
        stats_writer.write_geometry_stats(
            pat_id, version=1,
            delta_norms=all_norms, theta_norm=theta_norm,
            is_anomaly_arr=all_is_anomaly,
        )

        # Emit edge table (streaming path)
        edge_cfg = self._resolve_edge_table_config(pat)
        if edge_cfg is not None:
            edge_table = self._extract_edge_table(pat, edge_cfg)
            if edge_table.num_rows > 0:
                stats_writer.write_edges(pat_id, edge_table)
                # Cache edge stats
                import pyarrow.compute as _pc
                stats_writer.write_edge_stats(pat_id, {
                    "row_count": edge_table.num_rows,
                    "unique_from": _pc.count_distinct(edge_table["from_key"]).as_py(),
                    "unique_to": _pc.count_distinct(edge_table["to_key"]).as_py(),
                    "timestamp_min": float(_pc.min(edge_table["timestamp"]).as_py()),
                    "timestamp_max": float(_pc.max(edge_table["timestamp"]).as_py()),
                    "amount_min": float(_pc.min(edge_table["amount"]).as_py()),
                    "amount_max": float(_pc.max(edge_table["amount"]).as_py()),
                })

        from hypertopos.builder._theta_sensitivity import (
            compute_theta_sensitivity_from_sorted,
        )
        return pat_id, PatternBuildResult(
            mu=mu, sigma=sigma, theta=theta,
            population_size=n,
            prop_columns=prop_columns,
            excluded_properties=excluded_properties,
            group_stats=None,
            dimension_weights=dim_weights,
            gmm_components=None,
            cholesky_inv=None,
            dim_percentiles=self._compute_dim_percentiles(
                pat.entity_line,
                edge_dim_agg_matrix=(
                    edge_dim_agg_matrix
                    if edge_dim_agg_matrix.shape[1] > 0 else None
                ),
                edge_dim_agg_labels=(
                    edge_dim_agg_labels if edge_dim_agg_labels else None
                ),
            ),
            dimension_kinds=dimension_kinds,
            theta_sensitivity=compute_theta_sensitivity_from_sorted(sorted_norms),
            dim_normality_pvalues=self._compute_dim_normality_pvalues(
                pat.entity_line,
                edge_dim_agg_matrix=(
                    edge_dim_agg_matrix
                    if edge_dim_agg_matrix.shape[1] > 0 else None
                ),
                edge_dim_agg_labels=(
                    edge_dim_agg_labels if edge_dim_agg_labels else None
                ),
            ),
            edge_dim_names=edge_dim_names,
        )

    @staticmethod
    def _parse_event_buckets(
        event_table: pa.Table,
        time_col: str,
        time_window: str,
    ) -> tuple[np.ndarray, int, float, float]:
        """Parse timestamps into bucket indices. Returns (bucket_np, n_buckets, min_ts, window_secs)."""
        from hypertopos.builder.derived import _parse_time_window

        ts_col = event_table[time_col]
        if pa.types.is_timestamp(ts_col.type):
            unit = ts_col.type.unit
            divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
            divisor = divisors.get(unit, 1e6)
            epoch_arr = pc.divide(
                pc.cast(pc.cast(ts_col, pa.int64()), pa.float64()),
                divisor,
            )
        elif pa.types.is_floating(ts_col.type) or pa.types.is_integer(ts_col.type):
            epoch_arr = pc.cast(ts_col, pa.float64())
        else:
            from hypertopos.engine.chains import parse_timestamps_to_epoch
            epoch_arr = pa.chunked_array([pa.array(
                parse_timestamps_to_epoch(ts_col.to_pylist()),
                type=pa.float64(),
            )])

        window_secs = _parse_time_window(time_window)
        min_ts = pc.min(epoch_arr).as_py()
        if min_ts is None:
            return np.array([], dtype=np.int64), 0, 0.0, window_secs
        diff = pc.subtract(epoch_arr, min_ts)
        bucket_arr = pc.cast(
            pc.floor(pc.divide(pc.cast(diff, pa.float64()), window_secs)),
            pa.int64(),
        )
        bucket_np = bucket_arr.to_numpy(zero_copy_only=False).astype(np.int64)
        n_buckets = int(bucket_np.max()) + 1
        return bucket_np, n_buckets, min_ts, window_secs

    @staticmethod
    def _build_pat_meta_for_temporal(
        pat: _PatternReg,
        pbr: PatternBuildResult,
    ) -> dict[str, Any]:
        """Build the pattern metadata dict needed by _build_temporal_one.

        Produces the same keys that build_temporal() reads from sphere.json:
        relations, prop_columns, mu, sigma_diag, theta.
        """
        return {
            "relations": [
                {
                    "line_id": rel.line_id,
                    "direction": rel.direction,
                    "required": rel.required,
                    **({"edge_max": rel.edge_max} if rel.edge_max is not None else {}),
                    **({"display_name": rel.display_name} if rel.display_name else {}),
                }
                for rel in pat.relations
            ],
            "prop_columns": pbr.prop_columns,
            "mu": pbr.mu.tolist(),
            "sigma_diag": pbr.sigma.tolist(),
            "theta": pbr.theta.tolist(),
        }

    def _build_temporal_one(
        self,
        pat_id: str,
        pat: _PatternReg,
        pat_meta: dict[str, Any],
        event_table: pa.Table,
        bucket_np: np.ndarray,
        n_buckets: int,
        min_ts: float,
        window_secs: float,
        writer: Any,
    ) -> tuple[str, int]:
        """Build temporal data for a single pattern.

        Extracted from build_temporal closure for reuse in pipeline mode.
        """
        from contextlib import suppress

        from hypertopos.storage.writer import (
            _IVF_MIN_ROWS_PER_PARTITION,
            _ivf_index_worthwhile,
            _ivf_num_partitions,
            _write_lance,
        )

        relations_meta = pat_meta.get("relations", [])
        prop_columns = pat_meta.get("prop_columns", [])

        anchor_line_reg = self._lines[pat.entity_line]
        anchor_table_full = anchor_line_reg.table
        anchor_keys = anchor_table_full["primary_key"]
        anchor_keys_list = anchor_keys.to_pylist()
        n_anchor = len(anchor_keys_list)

        lance_dir = self.output_path / "temporal" / pat_id
        lance_dir.mkdir(parents=True, exist_ok=True)
        lance_path = lance_dir / "data.lance"

        D = len(relations_meta) + len(prop_columns)
        non_empty_buckets = np.unique(bucket_np)
        slices_written = len(non_empty_buckets)

        mem_budget = self._detect_available_memory()
        chunk_size = self._compute_chunk_size(
            n_anchor, n_buckets, D, int(mem_budget * 0.5),
        )

        all_bkt_ts = [
            datetime.fromtimestamp(min_ts + b * window_secs, tz=UTC)
            for b in range(n_buckets)
        ]
        _mu_arr = (
            np.array(pat_meta["mu"], dtype=np.float32)
            if pat_meta.get("mu") else None
        )
        _sig_arr = (
            np.array(pat_meta["sigma_diag"], dtype=np.float32)
            if pat_meta.get("sigma_diag") else None
        )

        if chunk_size >= n_anchor:
            # ── Single-pass (fits in RAM) ──
            shape_tensor = self._precompute_shape_tensor(
                pat_id, event_table, bucket_np, n_buckets,
                anchor_keys, anchor_keys_list, n_anchor,
                relations_meta, prop_columns,
            )
            _temporal_tensor_to_lance(
                shape_tensor, anchor_keys_list, n_anchor,
                non_empty_buckets, lance_path, min_ts, window_secs, D,
            )

            if slices_written > 0:
                mu = np.array(pat_meta["mu"], dtype=np.float32)
                sigma = np.array(pat_meta["sigma_diag"], dtype=np.float32)
                sigma_safe = np.where(sigma < 1e-6, 1.0, sigma)
                # Temporal shape_tensor covers only the time-varying dims
                # (relations + event/prop). When the pattern has time-invariant
                # build-time aggregates (edge_dim_aggregations, S1 ext) those
                # tail dims are NOT in the tensor — slice mu/sigma to match.
                D_shape = shape_tensor.shape[2]
                mu = mu[:D_shape]
                sigma_safe = sigma_safe[:D_shape]
                theta_norm = float(np.linalg.norm(
                    np.array(pat_meta["theta"], dtype=np.float32)[:D_shape],
                ))
                centroids: list[dict] = []
                for b in range(n_buckets):
                    ws = shape_tensor[:, b, :]
                    am = np.any(ws != 0, axis=1)
                    na = int(am.sum())
                    if na == 0:
                        continue
                    ad = (ws[am] - mu) / sigma_safe
                    c = ad.mean(axis=0).tolist()
                    ar = 0.0
                    if theta_norm > 0:
                        nr = np.sqrt(np.einsum('ij,ij->i', ad, ad))
                        ar = float((nr > theta_norm).sum() / len(nr))
                    centroids.append({
                        "window_start": datetime.fromtimestamp(
                            min_ts + b * window_secs, tz=UTC),
                        "window_end": datetime.fromtimestamp(
                            min_ts + (b + 1) * window_secs, tz=UTC),
                        "centroid": c, "entity_count": na,
                        "anomaly_rate": ar,
                    })
                if centroids:
                    writer.write_temporal_centroids(pat_id, centroids)

            if slices_written >= 3:
                mz = self._compute_max_rolling_z(
                    shape_tensor, n_anchor, n_buckets,
                )
                self._write_max_rolling_z(pat_id, anchor_keys_list, mz)

            if slices_written >= 2:
                am = np.any(
                    shape_tensor.reshape(n_anchor, -1) != 0, axis=1,
                )
                if am.any():
                    writer.write_trajectory_from_tensor(
                        pat_id, shape_tensor[am],
                        [anchor_keys_list[i] for i in np.flatnonzero(am)],
                        bucket_timestamps=all_bkt_ts,
                        mu=_mu_arr, sigma_diag=_sig_arr,
                    )

        else:
            # ── Chunked path (adaptive memory) ──
            logger.info(
                "Temporal %s: chunking %d entities into chunks of %d",
                pat_id, n_anchor, chunk_size,
            )
            mu = np.array(pat_meta["mu"], dtype=np.float32)
            sigma = np.array(pat_meta["sigma_diag"], dtype=np.float32)
            sigma_safe = np.where(sigma < 1e-6, 1.0, sigma)
            theta_norm = float(np.linalg.norm(
                np.array(pat_meta["theta"], dtype=np.float32),
            ))

            c_n_active = np.zeros(n_buckets, dtype=np.int64)
            c_sum_delta = np.zeros((n_buckets, D), dtype=np.float64)
            c_n_anom = np.zeros(n_buckets, dtype=np.int64)
            all_max_z = np.zeros(n_anchor, dtype=np.float32)
            traj_tables: list[pa.Table] = []

            _pre_grouped = self._precompute_derived_grouped(
                pat_id, event_table, bucket_np, relations_meta,
            )
            _pre_graph = self._precompute_graph_features(
                pat_id, event_table, anchor_keys, bucket_np,
                n_buckets, relations_meta,
            )

            for cs in range(0, n_anchor, chunk_size):
                ce = min(cs + chunk_size, n_anchor)
                nc = ce - cs
                ckl = anchor_keys_list[cs:ce]
                cka = pa.array(ckl, type=pa.string())

                chunk_table = anchor_table_full.slice(cs, nc)

                _chunk_graph: dict[int, np.ndarray] = {}
                for sk, full_t in _pre_graph.items():
                    _chunk_graph[sk] = full_t[cs:ce]

                ct = self._precompute_shape_tensor(
                    pat_id, event_table, bucket_np, n_buckets,
                    cka, ckl, nc,
                    relations_meta, prop_columns,
                    pre_grouped=_pre_grouped,
                    pre_graph=_chunk_graph,
                    anchor_table_override=chunk_table,
                )

                _temporal_tensor_to_lance(
                    ct, ckl, nc, non_empty_buckets,
                    lance_path, min_ts, window_secs, D,
                )

                for b in range(n_buckets):
                    sh = ct[:, b, :]
                    act = np.any(sh != 0, axis=1)
                    if not act.any():
                        continue
                    ad = ((sh[act] - mu) / sigma_safe).astype(
                        np.float64,
                    )
                    nr = np.sqrt(np.einsum('ij,ij->i', ad, ad))
                    c_n_active[b] += int(act.sum())
                    c_sum_delta[b] += ad.sum(axis=0)
                    if theta_norm > 0:
                        c_n_anom[b] += int(
                            (nr > theta_norm).sum(),
                        )

                all_max_z[cs:ce] = self._compute_max_rolling_z(
                    ct, nc, n_buckets,
                )

                if slices_written >= 2:
                    am = np.any(
                        ct.reshape(nc, -1) != 0, axis=1,
                    )
                    if am.any():
                        _build_traj_chunk(
                            traj_tables, ct[am],
                            [ckl[i] for i in np.flatnonzero(am)],
                            all_bkt_ts, _mu_arr, _sig_arr,
                        )

                del ct

            if slices_written > 0:
                centroids = []
                for b in range(n_buckets):
                    if c_n_active[b] == 0:
                        continue
                    cv = (c_sum_delta[b] / c_n_active[b]).tolist()
                    ar = float(c_n_anom[b] / c_n_active[b])
                    centroids.append({
                        "window_start": datetime.fromtimestamp(
                            min_ts + b * window_secs, tz=UTC),
                        "window_end": datetime.fromtimestamp(
                            min_ts + (b + 1) * window_secs, tz=UTC),
                        "centroid": cv,
                        "entity_count": int(c_n_active[b]),
                        "anomaly_rate": ar,
                    })
                if centroids:
                    writer.write_temporal_centroids(pat_id, centroids)

            if slices_written >= 3:
                self._write_max_rolling_z(
                    pat_id, anchor_keys_list, all_max_z,
                )

            if traj_tables:
                ct_traj = pa.concat_tables(traj_tables)
                tp = (
                    self.output_path / "_gds_meta" / "trajectory"
                    / f"{pat_id}.lance"
                )
                tp.parent.mkdir(parents=True, exist_ok=True)
                _write_lance(ct_traj, str(tp), mode="overwrite")
                nt = ct_traj.num_rows
                # Below num_partitions * 256 rows the IVF KMeans training is
                # starved and the index is degenerate; skip it so the build
                # stays short — trajectory search falls back to a correct
                # brute-force scan (zero recall loss).
                if _ivf_index_worthwhile(nt):
                    import lance as _ltr

                    tds = _ltr.dataset(str(tp))
                    np_ = _ivf_num_partitions(nt)
                    with suppress(Exception):
                        tds.create_index(
                            "trajectory_vector",
                            index_type="IVF_FLAT",
                            num_partitions=np_,
                        )
                elif nt > 0:
                    logger.info(
                        "trajectory %s: %d rows below ANN training "
                        "threshold (%d) — using full-scan fallback",
                        pat_id, nt,
                        _ivf_num_partitions(nt) * _IVF_MIN_ROWS_PER_PARTITION,
                    )

        if slices_written > 0:
            writer.compact_temporal(pat_id)
            writer.build_temporal_index(pat_id)

        return pat_id, slices_written

    def build_temporal(
        self,
        time_col: str,
        time_window: str,
        event_line: str | None = None,
        anchor_pattern: str | None = None,
    ) -> dict[str, int]:
        """Generate temporal snapshots from time-windowed event data.

        Must be called AFTER build() on the same builder instance.
        Writes raw [0..1] shape vectors per time bucket into temporal
        Lance datasets so that dive_solid / drift / regime_change
        primitives can operate on temporal data.

        Performance: pre-computes a (n_anchor, n_buckets, D) shape tensor
        using one groupby(anchor_fk, bucket) per derived dim instead of
        one groupby per dim per window. Graph features are batched per
        window. Arrow columns use broadcast + FixedSizeListArray.

        Args:
            time_col: Column in event line containing timestamps.
            time_window: Window size, e.g. "1d", "7d", "30d".
            event_line: Source event line id. Auto-detected if only one.
            anchor_pattern: Process only this pattern (None = all eligible).

        Returns:
            {pattern_id: n_slices_written}
        """
        # 1. Precondition: build() must have been called
        sphere_json_path = self.output_path / "_gds_meta" / "sphere.json"
        if not sphere_json_path.exists():
            raise ValueError("build() must be called before build_temporal()")

        # 2. Read sphere.json for pattern metadata
        sphere_data = json.loads(sphere_json_path.read_text())

        # 3. Identify event line
        if event_line is None:
            event_lines = [
                lid for lid, lr in self._lines.items() if lr.role == "event"
            ]
            if len(event_lines) == 0:
                raise ValueError("No event lines registered")
            if len(event_lines) > 1:
                raise ValueError(
                    f"Multiple event lines found: {event_lines}. "
                    "Specify event_line explicitly."
                )
            event_line = event_lines[0]

        if event_line not in self._lines:
            raise ValueError(f"Event line '{event_line}' not registered")
        event_table = self._lines[event_line].table

        # 4-5. Parse timestamps and compute bucket IDs
        bucket_np, n_buckets, min_ts, window_secs = (
            self._parse_event_buckets(event_table, time_col, time_window)
        )
        if n_buckets == 0:
            return {}

        # 6. Determine which anchor patterns to process
        patterns_to_process: dict[str, _PatternReg] = {}
        if anchor_pattern is not None:
            if anchor_pattern not in self._patterns:
                raise ValueError(f"Pattern '{anchor_pattern}' not registered")
            patterns_to_process[anchor_pattern] = self._patterns[anchor_pattern]
        else:
            for pid, pat in self._patterns.items():
                if pat.pattern_type == "anchor":
                    patterns_to_process[pid] = pat

        for pid, pat in patterns_to_process.items():
            if pat.pattern_type != "anchor":
                raise ValueError(
                    f"build_temporal only supports anchor patterns, "
                    f"got '{pat.pattern_type}' for '{pid}'"
                )

        from concurrent.futures import ThreadPoolExecutor

        from hypertopos.storage.writer import GDSWriter

        writer = GDSWriter(str(self.output_path))
        result: dict[str, int] = {}

        # 7. Process each anchor pattern — parallel when >1
        def _run_one(pid: str, p: _PatternReg) -> tuple[str, int]:
            pat_meta = sphere_data["patterns"].get(pid)
            if pat_meta is None:
                return pid, 0
            return self._build_temporal_one(
                pid, p, pat_meta, event_table,
                bucket_np, n_buckets, min_ts, window_secs, writer,
            )

        if len(patterns_to_process) > 1:
            with ThreadPoolExecutor(max_workers=len(patterns_to_process)) as pool:
                futures = [
                    pool.submit(_run_one, pid, p)
                    for pid, p in patterns_to_process.items()
                ]
                for fut in futures:
                    pid, n_slices = fut.result()
                    result[pid] = n_slices
        else:
            for pat_id, pat in patterns_to_process.items():
                pid, n_slices = _run_one(pat_id, pat)
                result[pid] = n_slices

        return result

    @staticmethod
    def _compute_chunk_size(
        n_entities: int,
        n_buckets: int,
        n_dims: int,
        memory_budget_bytes: int,
    ) -> int:
        bytes_per_entity = n_buckets * n_dims * 4  # float32
        overhead_per_entity = n_dims * 8 * 3 + 64  # Welford + indices
        total_per_entity = bytes_per_entity + overhead_per_entity
        chunk_size = memory_budget_bytes // max(total_per_entity, 1)
        floor = min(1000, n_entities)
        return max(floor, min(chunk_size, n_entities))

    @staticmethod
    def _plan_execution(
        patterns_info: dict[str, tuple[int, int, int]],
        available_ram: int,
    ) -> tuple[int, dict[str, int]]:
        budget = int(available_ram * 0.5)
        pattern_sizes = {
            pid: n * b * d * 4
            for pid, (n, b, d) in patterns_info.items()
        }
        total_full = sum(pattern_sizes.values())

        if total_full <= budget:
            return len(patterns_info), {
                pid: info[0] for pid, info in patterns_info.items()
            }

        largest = max(pattern_sizes.values()) if pattern_sizes else 1
        n_workers = max(1, min(budget // max(largest, 1), len(patterns_info)))
        per_worker_budget = budget // max(n_workers, 1)
        chunk_sizes = {}
        for pid, (n, b, d) in patterns_info.items():
            bpe = b * d * 4
            chunk_sizes[pid] = max(1000, min(per_worker_budget // max(bpe, 1), n))

        return n_workers, chunk_sizes

    @staticmethod
    def _detect_available_memory() -> int:
        try:
            import ctypes
            if hasattr(ctypes, "windll"):
                class _MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = _MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return int(stat.ullAvailPhys)
        except Exception:
            pass
        try:
            import os
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size
        except Exception:
            pass
        return 4 * 1024**3  # 4 GB fallback

    @staticmethod
    def _compute_max_rolling_z(
        shape_tensor: np.ndarray,
        n_anchor: int,
        n_buckets: int,
    ) -> np.ndarray:
        """Compute max rolling z-score across temporal windows.

        Uses Welford online algorithm: O(n_buckets × n_anchor × D) time,
        O(n_anchor × D) constant memory. Semantics match the naive
        expanding-window implementation: zeros for inactive buckets are
        included in running stats (detects re-activation), z is computed
        against PRIOR history before updating, population std (ddof=0)
        with floor at 0.01.
        """
        D = shape_tensor.shape[2]
        count = 0
        mean = np.zeros((n_anchor, D), dtype=np.float64)
        M2 = np.zeros((n_anchor, D), dtype=np.float64)
        max_z = np.zeros(n_anchor, dtype=np.float32)

        for t in range(n_buckets):
            current = shape_tensor[:, t, :].astype(np.float64)
            if current.sum() == 0:
                continue

            # Z against PRIOR history (before updating with current)
            if count >= 2:
                std = np.maximum(np.sqrt(np.maximum(M2 / count, 0.0)), 0.01)
                z = np.abs((current - mean) / std)
                z_max_t = z.max(axis=1).astype(np.float32)
                max_z = np.maximum(max_z, z_max_t)

            # Update running stats WITH current
            count += 1
            delta = current - mean
            mean += delta / count
            delta2 = current - mean
            M2 += delta * delta2

        return max_z

    def _write_max_rolling_z(
        self,
        pat_id: str,
        anchor_keys: list[str],
        max_rolling_z: np.ndarray,
    ) -> None:
        """Write max_rolling_z column to existing geometry Lance dataset."""
        import lance as _lance

        geo_path = (
            self.output_path / "geometry" / pat_id / "data.lance"
        )
        if not geo_path.exists():
            return

        ds = _lance.dataset(str(geo_path))
        geo_table = ds.to_table(columns=["primary_key"])
        geo_keys = geo_table["primary_key"].to_pylist()

        # Align rolling_z to geometry order
        key_to_z: dict[str, float] = dict(zip(anchor_keys, max_rolling_z.tolist(), strict=False))
        aligned_z = pa.array(
            [key_to_z.get(k, 0.0) for k in geo_keys], type=pa.float32(),
        )
        ds.merge(
            pa.table({"primary_key": geo_table["primary_key"], "max_rolling_z": aligned_z}),
            "primary_key",
        )

    def _precompute_derived_grouped(
        self,
        pat_id: str,
        event_table: pa.Table,
        bucket_np: np.ndarray,
        relations_meta: list[dict[str, Any]],
    ) -> dict[str, pa.Table]:
        """Pre-compute derived-dim groupby tables once for chunked reuse.

        Returns {fk_key: grouped_arrow_table} for each FK batch.
        """
        pat = self._patterns[pat_id]
        derived_dim_names: dict[str, Any] = {}
        for spec in self._derived_dims:
            if spec.anchor_line == pat.entity_line:
                derived_dim_names[spec.dimension_name] = spec

        from collections import defaultdict as _ddict
        fk_batches: dict[str, list] = _ddict(list)
        for _j, rel_meta in enumerate(relations_meta):
            direction = rel_meta.get("direction", "in")
            line_id = rel_meta.get("line_id", "")
            if direction == "self":
                continue
            matching_rel = None
            for rel in pat.relations:
                if rel.line_id == line_id and rel.direction == direction:
                    matching_rel = rel
                    break
            if matching_rel is None or matching_rel.fk_col is None:
                continue
            fk_col_name = matching_rel.fk_col
            if fk_col_name not in derived_dim_names:
                continue
            spec = derived_dim_names[fk_col_name]
            if spec.metric.startswith("iet_"):
                continue
            anchor_fk = spec.anchor_fk
            fk_key = "|".join(anchor_fk) if isinstance(anchor_fk, list) else anchor_fk
            if fk_key not in fk_batches:
                fk_batches[fk_key] = spec

        if not fk_batches:
            return {}

        _agg_map = {
            "count": lambda _mc: ("primary_key", "count"),
            "count_distinct": lambda mc: (mc, "count_distinct"),
            "sum": lambda mc: (mc, "sum"),
            "max": lambda mc: (mc, "max"),
            "mean": lambda mc: (mc, "mean"),
            "std": lambda mc: (mc, "stddev"),
        }

        bucket_pa = pa.array(bucket_np, type=pa.int64())
        work_table = event_table.append_column("_bucket", bucket_pa)

        result: dict[str, pa.Table] = {}
        # Collect all specs per fk_key, then compute
        fk_to_specs: dict[str, list] = _ddict(list)
        for _j, rel_meta in enumerate(relations_meta):
            direction = rel_meta.get("direction", "in")
            line_id = rel_meta.get("line_id", "")
            if direction == "self":
                continue
            matching_rel = None
            for rel in pat.relations:
                if rel.line_id == line_id and rel.direction == direction:
                    matching_rel = rel
                    break
            if matching_rel is None or matching_rel.fk_col is None:
                continue
            fk_col_name = matching_rel.fk_col
            if fk_col_name not in derived_dim_names:
                continue
            spec = derived_dim_names[fk_col_name]
            if spec.metric.startswith("iet_"):
                continue
            anchor_fk = spec.anchor_fk
            fk_key = "|".join(anchor_fk) if isinstance(anchor_fk, list) else anchor_fk
            fk_to_specs[fk_key].append(spec)

        for fk_key, specs in fk_to_specs.items():
            sample_spec = specs[0]
            anchor_fk = sample_spec.anchor_fk

            if isinstance(anchor_fk, list):
                separator = "→"
                for cs in self._composite_lines:
                    if cs.line_id == sample_spec.anchor_line:
                        separator = cs.separator
                        break
                str_cols = [
                    pc.cast(work_table[col], pa.string())
                    for col in anchor_fk
                ]
                composite_fk = pc.binary_join_element_wise(
                    *str_cols, separator,
                )
                gb_table = work_table.append_column(
                    "_composite_fk", composite_fk,
                )
                fk_group_col = "_composite_fk"
            else:
                gb_table = work_table
                fk_group_col = anchor_fk

            # Build agg exprs from all specs sharing this FK
            agg_exprs: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for spec in specs:
                agg_col, agg_func = _agg_map[spec.metric](spec.metric_col)
                key = (agg_col, agg_func)
                if key not in seen:
                    seen.add(key)
                    agg_exprs.append(key)

            result[fk_key] = gb_table.group_by(
                [fk_group_col, "_bucket"],
            ).aggregate(agg_exprs)

        return result

    def _precompute_graph_features(
        self,
        pat_id: str,
        event_table: pa.Table,
        anchor_keys: pa.Array,
        bucket_np: np.ndarray,
        n_buckets: int,
        relations_meta: list[dict[str, Any]],
    ) -> dict[int, np.ndarray]:
        """Pre-compute graph features once for chunked reuse.

        Returns {spec_key: (n_anchor, n_buckets, n_feats)} tensor per spec.
        Keys use id(spec) matching _precompute_shape_tensor's lookup.
        """
        from hypertopos.builder.derived import compute_graph_features_temporal

        pat = self._patterns[pat_id]
        graph_feature_names: dict[str, Any] = {}
        for spec in self._graph_features:
            if spec.anchor_line == pat.entity_line:
                for feat in spec.features:
                    graph_feature_names[feat] = spec

        # Classify graph dims (mirroring _precompute_shape_tensor logic)
        graph_dims: list[tuple[int, dict, Any, Any]] = []
        for j, rel_meta in enumerate(relations_meta):
            direction = rel_meta.get("direction", "in")
            line_id = rel_meta.get("line_id", "")
            if direction == "self":
                continue
            matching_rel = None
            for rel in pat.relations:
                if rel.line_id == line_id and rel.direction == direction:
                    matching_rel = rel
                    break
            if matching_rel is None or matching_rel.fk_col is None:
                continue
            if matching_rel.fk_col in graph_feature_names:
                graph_dims.append(
                    (j, rel_meta, matching_rel,
                     graph_feature_names[matching_rel.fk_col]),
                )

        if not graph_dims:
            return {}

        spec_to_dims: dict[int, list[tuple[int, dict, Any]]] = {}
        spec_map: dict[int, Any] = {}
        for j, rel_meta, rel, gf_spec in graph_dims:
            spec_key = id(gf_spec)
            if spec_key not in spec_to_dims:
                spec_to_dims[spec_key] = []
                spec_map[spec_key] = gf_spec
            spec_to_dims[spec_key].append((j, rel_meta, rel))

        result: dict[int, np.ndarray] = {}
        for spec_key, dims_list in spec_to_dims.items():
            gf_spec = spec_map[spec_key]
            all_features = [rel.fk_col for _, _, rel in dims_list]
            result[spec_key] = compute_graph_features_temporal(
                event_table, anchor_keys,
                gf_spec.from_col, gf_spec.to_col,
                all_features, bucket_np, n_buckets,
            )

        return result

    def _precompute_shape_tensor(
        self,
        pat_id: str,
        event_table: pa.Table,
        bucket_np: np.ndarray,
        n_buckets: int,
        anchor_keys: pa.Array,
        anchor_keys_list: list[str],
        n_anchor: int,
        relations_meta: list[dict[str, Any]],
        prop_columns: list[str],
        *,
        pre_grouped: dict[str, pa.Table] | None = None,
        pre_graph: dict[int, np.ndarray] | None = None,
        anchor_table_override: pa.Table | None = None,
    ) -> np.ndarray:
        """Pre-compute shape tensor (n_anchor, n_buckets, D) in single-pass.

        For derived dims: one groupby(anchor_fk, bucket) per dim fills
        all windows at once. For graph features: one batched call per
        window. Static dims and props are filled once and broadcast.

        When called from the chunked path, *pre_grouped* and *pre_graph*
        provide pre-computed aggregates so the expensive event-table scans
        happen only once (outside the chunk loop).

        Returns (n_anchor, n_buckets, D) float32 array.
        """
        pat = self._patterns[pat_id]
        n_rel = len(relations_meta)
        n_prop = len(prop_columns)
        D = n_rel + n_prop

        shape_tensor = np.zeros((n_anchor, n_buckets, D), dtype=np.float32)
        _anchor_tbl = (
            anchor_table_override
            if anchor_table_override is not None
            else self._lines[pat.entity_line].table
        )

        # Build lookup maps: dim_index, derived specs, graph feature specs
        key_to_idx = {k: i for i, k in enumerate(anchor_keys_list)}

        # Classify each relation dimension
        derived_dim_names: dict[str, Any] = {}
        for spec in self._derived_dims:
            if spec.anchor_line == pat.entity_line:
                derived_dim_names[spec.dimension_name] = spec

        graph_feature_names: dict[str, Any] = {}
        for spec in self._graph_features:
            if spec.anchor_line == pat.entity_line:
                for feat in spec.features:
                    graph_feature_names[feat] = spec

        # Categorize dimensions by type for efficient processing
        static_dims: list[tuple[int, dict, Any]] = []  # (j, rel_meta, rel)
        derived_dims: list[tuple[int, dict, Any, Any]] = []  # (j, rel_meta, rel, spec)
        graph_dims: list[tuple[int, dict, Any, Any]] = []  # (j, rel_meta, rel, spec)

        for j, rel_meta in enumerate(relations_meta):
            direction = rel_meta.get("direction", "in")
            line_id = rel_meta.get("line_id", "")

            if direction == "self":
                # Self-reference: constant 1.0 across all windows
                shape_tensor[:, :, j] = 1.0
                continue

            # Find matching RelationSpec
            matching_rel = None
            for rel in pat.relations:
                if rel.line_id == line_id and rel.direction == direction:
                    matching_rel = rel
                    break
            if matching_rel is None or matching_rel.fk_col is None:
                continue

            fk_col_name = matching_rel.fk_col
            if fk_col_name in derived_dim_names:
                derived_dims.append(
                    (j, rel_meta, matching_rel, derived_dim_names[fk_col_name]),
                )
            elif fk_col_name in graph_feature_names:
                graph_dims.append(
                    (j, rel_meta, matching_rel, graph_feature_names[fk_col_name]),
                )
            else:
                static_dims.append((j, rel_meta, matching_rel))

        # --- A. Static dims: fill once and broadcast across all buckets ---
        if static_dims:
            anchor_table = _anchor_tbl
            for j, rel_meta, rel in static_dims:
                edge_max = rel_meta.get("edge_max")
                fk_col_name = rel.fk_col
                if fk_col_name not in anchor_table.schema.names:
                    continue
                if edge_max is not None:
                    col = anchor_table[fk_col_name]
                    count_arr = pc.fill_null(col, 0).to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float32)
                    static_vals = np.clip(count_arr, 0, edge_max) / edge_max
                else:
                    col_arrow = anchor_table[fk_col_name]
                    valid_mask = pc.fill_null(
                        pc.and_(
                            pc.is_valid(col_arrow),
                            pc.not_equal(col_arrow, ""),
                        ),
                        False,
                    )
                    static_vals = valid_mask.to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float32)
                # Broadcast: same values for all buckets
                shape_tensor[:, :, j] = static_vals[:, np.newaxis]

        # --- B. Property fill: static, broadcast across all buckets ---
        if n_prop > 0:
            anchor_table = _anchor_tbl
            for p_idx, prop in enumerate(prop_columns):
                col_idx = n_rel + p_idx
                if prop in anchor_table.schema.names:
                    col = anchor_table[prop]
                    fill_vec = pc.is_valid(col).to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float32)
                    shape_tensor[:, :, col_idx] = fill_vec[:, np.newaxis]

        # --- C. Derived dims: batched groupby(anchor_fk, bucket) ---
        # Group derived dims by FK column, one multi-aggregate group_by per FK
        if derived_dims:
            from collections import defaultdict as _ddict
            fk_batches: dict[str, list] = _ddict(list)
            for j, rel_meta, rel, spec in derived_dims:
                if spec.metric.startswith("iet_"):
                    continue
                anchor_fk = spec.anchor_fk
                fk_key = "|".join(anchor_fk) if isinstance(anchor_fk, list) else anchor_fk
                fk_batches[fk_key].append((j, rel_meta, rel, spec))

            _agg_map = {
                "count": lambda _mc: ("primary_key", "count"),
                "count_distinct": lambda mc: (mc, "count_distinct"),
                "sum": lambda mc: (mc, "sum"),
                "max": lambda mc: (mc, "max"),
                "mean": lambda mc: (mc, "mean"),
                "std": lambda mc: (mc, "stddev"),
            }

            for _fk_key, batch_dims in fk_batches.items():
                sample_spec = batch_dims[0][3]
                anchor_fk = sample_spec.anchor_fk

                # Build dim→result mapping (needed for both paths)
                agg_exprs: list[tuple[str, str]] = []
                seen_exprs: set[tuple[str, str]] = set()
                dim_to_result: dict[int, tuple[str, float]] = {}

                for j, rel_meta, _rel, spec in batch_dims:
                    agg_col, agg_func = _agg_map[spec.metric](spec.metric_col)
                    result_col = f"{agg_col}_{agg_func}"
                    em = rel_meta.get("edge_max") or 1
                    dim_to_result[j] = (result_col, float(em))

                    expr_key = (agg_col, agg_func)
                    if expr_key not in seen_exprs:
                        seen_exprs.add(expr_key)
                        agg_exprs.append(expr_key)

                # Use pre-computed grouped table or compute fresh
                if pre_grouped is not None and _fk_key in pre_grouped:
                    grouped = pre_grouped[_fk_key]
                    fk_group_col = (
                        "_composite_fk"
                        if isinstance(anchor_fk, list) else anchor_fk
                    )
                else:
                    bucket_pa = pa.array(bucket_np, type=pa.int64())
                    work_table = event_table.append_column("_bucket", bucket_pa)

                    if isinstance(anchor_fk, list):
                        separator = "→"
                        for cs in self._composite_lines:
                            if cs.line_id == sample_spec.anchor_line:
                                separator = cs.separator
                                break
                        str_cols = [
                            pc.cast(work_table[col], pa.string())
                            for col in anchor_fk
                        ]
                        composite_fk = pc.binary_join_element_wise(
                            *str_cols, separator,
                        )
                        gb_table = work_table.append_column(
                            "_composite_fk", composite_fk,
                        )
                        fk_group_col = "_composite_fk"
                    else:
                        gb_table = work_table
                        fk_group_col = anchor_fk

                    grouped = gb_table.group_by(
                        [fk_group_col, "_bucket"],
                    ).aggregate(agg_exprs)

                # Vectorized scatter — Arrow pc.index_in + numpy fancy indexing
                from hypertopos.builder._scatter import vectorized_scatter

                anchor_keys_arr = pa.array(anchor_keys_list)
                gk_col = grouped[fk_group_col]
                gb_col = grouped["_bucket"]

                for j, (result_col, em) in dim_to_result.items():
                    vectorized_scatter(
                        tensor=shape_tensor,
                        dim_idx=j,
                        edge_max=em,
                        anchor_keys_arr=anchor_keys_arr,
                        grouped_fk_col=gk_col,
                        grouped_bucket_col=gb_col,
                        grouped_values_col=grouped[result_col],
                    )

        # --- D. Graph features: batched across all windows ---
        if graph_dims:
            from hypertopos.builder.derived import compute_graph_features_temporal

            spec_to_dims: dict[int, list[tuple[int, dict, Any]]] = {}
            spec_map: dict[int, Any] = {}
            for j, rel_meta, rel, gf_spec in graph_dims:
                spec_key = id(gf_spec)
                if spec_key not in spec_to_dims:
                    spec_to_dims[spec_key] = []
                    spec_map[spec_key] = gf_spec
                spec_to_dims[spec_key].append((j, rel_meta, rel))

            for spec_key, dims_list in spec_to_dims.items():
                gf_spec = spec_map[spec_key]
                all_features = [rel.fk_col for _, _, rel in dims_list]

                if pre_graph is not None and spec_key in pre_graph:
                    tensor_block = pre_graph[spec_key]
                else:
                    tensor_block = compute_graph_features_temporal(
                        event_table, anchor_keys,
                        gf_spec.from_col, gf_spec.to_col,
                        all_features, bucket_np, n_buckets,
                    )

                for f_idx, (j, rel_meta, rel) in enumerate(dims_list):
                    edge_max = rel_meta.get("edge_max")
                    em = edge_max if edge_max is not None else 1
                    shape_tensor[:, :, j] = np.clip(
                        tensor_block[:, :, f_idx], 0, em,
                    ) / em

        return shape_tensor

    def _validate(self) -> None:
        for pat_id, pat in self._patterns.items():
            if pat.entity_line not in self._lines:
                raise ValueError(
                    f"Pattern '{pat_id}' references entity_line '{pat.entity_line}' "
                    f"which was not registered via add_line()"
                )
            entity_table = self._lines[pat.entity_line].table
            for rel in pat.relations:
                if rel.line_id not in self._lines:
                    raise ValueError(
                        f"Pattern '{pat_id}' relation '{rel.line_id}' "
                        f"was not registered via add_line()"
                    )
                if rel.direction != "self" and rel.fk_col is None:
                    raise ValueError(
                        f"Pattern '{pat_id}' relation '{rel.line_id}': "
                        f"fk_col must not be None when direction='{rel.direction}'"
                    )
                if rel.direction != "self" and rel.fk_col is not None:
                    if rel.fk_col not in entity_table.schema.names:
                        raise ValueError(
                            f"Pattern '{pat_id}' relation '{rel.line_id}': "
                            f"fk_col '{rel.fk_col}' not found in '{pat.entity_line}' columns"
                        )
                    if rel.edge_max is not None:
                        if rel.edge_max <= 0:
                            raise ValueError(
                                f"Pattern '{pat_id}' relation '{rel.line_id}': "
                                f"edge_max must be >= 1, got {rel.edge_max}."
                            )
                        col_type = entity_table.schema.field(rel.fk_col).type
                        if not (
                            pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
                        ):
                            raise ValueError(
                                f"Pattern '{pat_id}' relation '{rel.line_id}': "
                                f"edge_max requires a numeric count column, "
                                f"got '{col_type}' for '{rel.fk_col}'."
                            )

            # Validate mixed edge_max: all non-self relations must agree
            non_self = [r for r in pat.relations if r.direction != "self"]
            if non_self:
                has_continuous = any(
                    r.edge_max is not None for r in non_self
                )
                has_binary = any(
                    r.edge_max is None for r in non_self
                )
                if has_continuous and has_binary:
                    raise ValueError(
                        f"Pattern '{pat_id}': mixed edge_max — "
                        "set edge_max for all non-self relations or none."
                    )

            # Validate event dimensions
            for edim in pat.event_dimensions:
                if edim.column not in entity_table.schema.names:
                    raise ValueError(
                        f"Pattern '{pat_id}' event dimension: "
                        f"column '{edim.column}' not found in "
                        f"'{pat.entity_line}' columns"
                    )
                col_type = entity_table.schema.field(edim.column).type
                if not (pa.types.is_integer(col_type) or pa.types.is_floating(col_type)):
                    raise ValueError(
                        f"Pattern '{pat_id}' event dimension: "
                        f"column '{edim.column}' must be numeric, got '{col_type}'"
                    )

            # A pattern with zero declared dimensions has no geometry to compute —
            # delta vectors collapse to width 0. Reject up front: silent acceptance
            # used to be possible under Lance format 2.0, but format 2.1+ panics on
            # fixed_size_list[0] columns (lance-format/lance#5102), and an anomaly
            # geometry without dimensions is meaningless regardless of the storage
            # format. Tracked properties count as dimension sources because they
            # become prop_columns in the resolved delta width.
            if (
                not pat.relations
                and not pat.event_dimensions
                and not pat.tracked_properties
            ):
                # Chain anchor with zero extracted chains lands here because
                # `_resolve_chain_dims` had no `_chain_dims` tuples to inject
                # (the empty-chains short-circuit in `add_chain_line` returns
                # before populating them). The generic "declare at least one
                # relation..." message is misleading for chains because the
                # user did declare features — chain extraction just produced
                # nothing.
                if pat.entity_line in self._chain_lines:
                    raise ValueError(
                        f"Pattern '{pat_id}': chain extraction returned 0 "
                        f"chains for line '{pat.entity_line}'; the auto-"
                        f"injected chain feature relations are therefore "
                        f"empty and the pattern has no geometry to compute. "
                        f"Either adjust chain seed criteria in "
                        f"chain_lines.{pat.entity_line!r} (lower "
                        f"seed_percentile_*, increase max_hops range) so "
                        f"extraction yields chains, or remove the chain_line "
                        f"declaration if no chains are expected for this "
                        f"sphere."
                    )
                raise ValueError(
                    f"Pattern '{pat_id}' has no dimensions: declare at least one "
                    f"relation, event dimension, derived dimension, or "
                    f"tracked_properties entry. A pattern with zero dimensions "
                    f"cannot produce a meaningful geometry."
                )
