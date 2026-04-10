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

logger = logging.getLogger(__name__)

MIN_FILL_RATE: float = 0.05  # props with lower fill rate excluded from delta
MAX_FILL_RATE: float = 0.999  # props with higher fill rate excluded (zero-variance)
GEOMETRY_CHUNK_SIZE: int = 500_000  # entities above this threshold trigger chunked writes

_INTERNAL_COLUMNS = {"version", "status", "created_at", "changed_at"}


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


def compute_entity_geometry(
    entity_table: pa.Table,
    mu: np.ndarray,
    sigma: np.ndarray,
    relations_meta: list[dict],
    event_dimensions_meta: list[dict] | None = None,
    dimension_weights: np.ndarray | None = None,
    prop_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute deltas, norms, and shape vectors for entities using existing pattern stats.

    Builds shape vectors from entity_table columns based on relations, event
    dimensions, and prop_columns metadata (from sphere.json), then z-scores
    against the provided mu/sigma to produce delta vectors.

    Returns:
        (deltas, delta_norms, shape_vectors) all as float32 arrays.
    """
    n = len(entity_table)
    D_rel = len(relations_meta)
    D_event = len(event_dimensions_meta) if event_dimensions_meta else 0
    D_prop = len(prop_columns) if prop_columns else 0

    shape_vectors = np.zeros((n, D_rel + D_event + D_prop), dtype=np.float32)

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

    # Prop columns — binary fill (0/1 based on is_valid)
    if prop_columns:
        D_base = D_rel + D_event
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
        self._precomputed_dims: list = []  # PrecomputedDimSpec list
        self._aliases: dict[str, _AliasReg] = {}
        self._no_edges: bool = False  # set by CLI --no-edges

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
                Default: hop_count, is_cyclic, n_distinct_categories, amount_decay.
        """
        if features is None:
            features = ["hop_count", "is_cyclic", "n_distinct_categories", "amount_decay"]

        if not chains:
            # Empty chains → empty line
            cols: dict[str, list] = {"primary_key": []}
            for f in features:
                cols[f] = []
            self.add_line(line_id, pa.table(cols), key_col="primary_key", source_id=line_id, role="anchor")
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

        # 3. Concatenate edge + event dims + prop fill into full shape matrix
        parts = [shape_vectors]
        if event_dim_matrix.shape[1] > 0:
            parts.append(event_dim_matrix)
        if prop_fill_matrix.shape[1] > 0:
            parts.append(prop_fill_matrix)
        full_shape_vectors = (
            np.concatenate(parts, axis=1) if len(parts) > 1
            else shape_vectors
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
                full_shape_vectors, pat.anomaly_percentile
            )

            # Per-group stats
            group_stats_dict, deltas, delta_norms = compute_stats_grouped(
                full_shape_vectors, group_ids, pat.anomaly_percentile
            )

            # Pre-compute boolean masks once — reused in all three group loops
            unique_groups = list(group_stats_dict.keys())
            group_mask_map: dict[str, np.ndarray] = {
                gid: group_ids == gid for gid in unique_groups
            }

            # Prop_columns sigma override applies to grouped stats too
            n_rel = len(pat.relations)
            if prop_columns:
                from hypertopos.builder._stats import SIGMA_EPS_PROP

                sigma[n_rel:] = np.maximum(sigma[n_rel:], SIGMA_EPS_PROP)
                # Re-run global stats with corrected sigma
                global_deltas = ((full_shape_vectors - mu) / sigma).astype(np.float32)
                global_norms = np.sqrt(
                    np.einsum('ij,ij->i', global_deltas, global_deltas),
                ).astype(np.float32)
                theta_scalar = float(np.percentile(global_norms, pat.anomaly_percentile))
                D_full = full_shape_vectors.shape[1]
                component = theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
                theta = np.full(D_full, component, dtype=np.float32)

                # Re-run per-group with corrected sigma for prop columns
                for gid, (mu_g, sigma_g, _theta_g, pop_g) in group_stats_dict.items():
                    sigma_g[n_rel:] = np.maximum(sigma_g[n_rel:], SIGMA_EPS_PROP)
                    mask = group_mask_map[gid]
                    group_shape = full_shape_vectors[mask]
                    g_deltas = ((group_shape - mu_g) / sigma_g).astype(np.float32)
                    g_norms = np.sqrt(np.einsum('ij,ij->i', g_deltas, g_deltas)).astype(np.float32)
                    deltas[mask] = g_deltas
                    delta_norms[mask] = g_norms
                    g_theta_scalar = (
                        float(np.percentile(g_norms, pat.anomaly_percentile))
                        if len(g_norms) > 1 else 0.0
                    )
                    D_full = full_shape_vectors.shape[1]
                    g_comp = g_theta_scalar / np.sqrt(D_full) if D_full > 0 else 0.0
                    g_theta_new = np.full(D_full, g_comp, dtype=np.float32)
                    group_stats_dict[gid] = (mu_g, sigma_g, g_theta_new, pop_g)
        else:
            mu, sigma, theta, deltas, delta_norms, mah_cov_inv = compute_stats(
                full_shape_vectors, pat.anomaly_percentile,
                use_mahalanobis=bool(pat.use_mahalanobis),
            )

            # Prop_columns are binary (0/1) — use higher sigma floor
            n_rel = len(pat.relations)
            if prop_columns:
                from hypertopos.builder._stats import SIGMA_EPS_PROP

                sigma[n_rel:] = np.maximum(sigma[n_rel:], SIGMA_EPS_PROP)
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
            for c_idx, (mu_c, sigma_c, theta_c, pop_c) in enumerate(gmm_components_result):
                mask = gmm_assignments == c_idx
                if mask.sum() == 0:
                    continue
                # Apply SIGMA_EPS_PROP floor to prop dims in cluster sigma
                if prop_columns:
                    sigma_c = sigma_c.copy()
                    sigma_c[n_rel_gmm:] = np.maximum(sigma_c[n_rel_gmm:], SIGMA_EPS_PROP)
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

        # 5. Compute delta_rank_pct (population-level percentile rank)
        sorted_norms = np.sort(delta_norms)
        ranks = np.searchsorted(sorted_norms, delta_norms, side="left")
        delta_rank_pcts = (ranks / n * 100).astype(np.float32)

        # 6. Compute conformal p-values (reuse sorted_norms from step 5)
        conformal_p = compute_conformal_p(delta_norms, sorted_norms=sorted_norms)

        # 7. Compute is_anomaly (per-cluster if GMM, per-group if grouped, else global)
        is_anomaly_arr: np.ndarray | None = None
        if gmm_components_result is not None:
            is_anomaly_arr = np.zeros(n, dtype=bool)
            for c_idx, (_, _, c_theta, _) in enumerate(gmm_components_result):
                c_theta_norm = float(np.linalg.norm(c_theta))
                mask = gmm_assignments == c_idx
                is_anomaly_arr[mask] = (c_theta_norm > 0.0) & (delta_norms[mask] >= c_theta_norm)
        elif group_stats_dict:
            is_anomaly_arr = np.zeros(n, dtype=bool)
            for gid, (_, _, g_theta, _) in group_stats_dict.items():
                g_theta_norm = float(np.linalg.norm(g_theta))
                mask = group_mask_map[gid]
                is_anomaly_arr[mask] = (g_theta_norm > 0.0) & (delta_norms[mask] >= g_theta_norm)

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
    ) -> pa.Table:
        """Build a geometry Arrow table for entities [start:end).

        Uses pre-computed deltas/norms/ranks from full population stats.
        """
        entity_table = self._lines[pat.entity_line].table
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

        common_cols = {
            "primary_key":     pk_str,
            "scale":           _const_i32(1),
            "delta":           delta_col,
            "delta_norm":      pa.array(chunk_norms, type=pa.float32()),
            "delta_rank_pct":  pa.array(chunk_ranks, type=pa.float32()),
            "is_anomaly":      pa.array(is_anomaly_arr, type=pa.bool_()),
            "conformal_p":     chunk_conformal,
            "n_anomalous_dims": (
                pa.array(n_anom_dims[start:end], type=pa.int32())
                if n_anom_dims is not None
                else pa.array(np.zeros(cn, dtype=np.int32), type=pa.int32())
            ),
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
    ) -> tuple[pa.Table, PopulationStats]:
        """Build geometry Arrow table for a pattern (single-pass, in-memory).

        Returns:
            (geometry_table, population_stats)
        """
        n = len(self._lines[pat.entity_line].table)
        ps = self._compute_population_stats(pat)

        theta_norm = float(np.linalg.norm(ps.theta))
        table = self._build_geometry_slice(
            pat, 0, n, ps.deltas, ps.delta_norms,
            ps.delta_rank_pcts, theta_norm, ps.fk_arrays,
            ps.conformal_p, ps.is_anomaly_arr, ps.n_anom_dims,
        )
        return table, ps

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
            "timestamp", "ts", "event_time", "created_at", "tx_date", "date",
        )
        amt_candidates = (
            "amount_received", "amount", "amount_paid", "value",
            "total", "amt",
        )

        ts_col = next(
            (c for c in ts_candidates if c in schema_names), None,
        )
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

        # Timestamp
        ts_arr: pa.Array
        ts_col = cfg.timestamp_col
        if ts_col and ts_col in schema_names:
            ts_arr = self._to_epoch_seconds(event_table[ts_col])
        else:
            # Try common column names
            for name in ("timestamp", "ts", "date", "created_at", "tx_date"):
                if name in schema_names:
                    ts_arr = self._to_epoch_seconds(event_table[name])
                    break
            else:
                ts_arr = pa.array(
                    [0.0] * len(event_table), type=pa.float64(),
                )

        # Amount
        amt_arr: pa.Array
        amt_col = cfg.amount_col
        if amt_col and amt_col in schema_names:
            amt_arr = pc.cast(
                pc.fill_null(event_table[amt_col], 0.0), pa.float64(),
            )
        else:
            # Try common column names
            for name in ("amount", "value", "total", "amt"):
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

            # Read geometry deltas from Lance
            import lance

            geo_path = (
                self.output_path / "geometry" / areg.base_pattern_id
                / "v=1" / "data.lance"
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
            deltas = delta_col.values.to_numpy(
                zero_copy_only=False,
            ).reshape(-1, list_size).astype(np.float32)

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

    def _compute_dim_percentiles(
        self, entity_line_id: str,
    ) -> dict[str, dict[str, float]] | None:
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
        return percentiles if percentiles else None

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
            }
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
            if pbr.dim_percentiles:
                pat_dict["dim_percentiles"] = pbr.dim_percentiles
            if pat.description:
                pat_dict["description"] = pat.description
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
            patterns_dict[pat_id] = pat_dict

        sphere_dict: dict[str, Any] = {
            "sphere_id": self.sphere_id,
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

    def build(self) -> str:
        """Validate, compute stats, write all files. Returns output_path as string."""
        from concurrent.futures import ThreadPoolExecutor

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
            # Skip internal dummy dimension lines (_d_*) — single-row placeholders
            if line_id.startswith("_d_"):
                return
            # Resolve FTS columns: None → auto (anchor=all, event=none)
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

        # Parallel write for independent lines
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

        # 2. Build and write geometry for each pattern (parallel when >1)
        _stats_writer = GDSWriter(str(self.output_path))

        def _build_and_write(
            pat_id: str, pat: _PatternReg,
        ) -> tuple[str, PatternBuildResult]:
            n = len(self._lines[pat.entity_line].table)

            if n > GEOMETRY_CHUNK_SIZE:
                # Complex modes (group_by, GMM, Mahalanobis) need full
                # population in memory; use chunked fallback.
                # Simple mode uses streaming three-pass with O(chunk) RAM.
                if (pat.group_by_property
                        or pat.gmm_n_components
                        or pat.use_mahalanobis):
                    return self._build_and_write_chunked(
                        pat_id, pat, _stats_writer,
                        write_geometry_chunk, finalize_geometry_chunks,
                    )
                return self._build_and_write_streaming(
                    pat_id, pat, _stats_writer,
                    write_geometry_chunk, finalize_geometry_chunks,
                )

            geom_table, ps = self._build_geometry_table(pat)
            write_geometry(
                self.output_path, pat_id, geom_table, version=1,
            )

            # Persist geometry stats cache
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

            # Emit edge table for event patterns with from/to structure
            edge_cfg = self._resolve_edge_table_config(pat)
            if edge_cfg is not None:
                edge_table = self._extract_edge_table(pat, edge_cfg)
                if edge_table.num_rows > 0:
                    _stats_writer.write_edges(pat_id, edge_table)
                    # Cache edge stats
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
                ),
            )

        if len(self._patterns) > 1:
            with ThreadPoolExecutor(max_workers=len(self._patterns)) as pool:
                futures = [
                    pool.submit(_build_and_write, pid, p)
                    for pid, p in self._patterns.items()
                ]
                for fut in futures:
                    pat_id, stats = fut.result()
                    pattern_stats[pat_id] = stats
        else:
            for pat_id, pat in self._patterns.items():
                _, stats = _build_and_write(pat_id, pat)
                pattern_stats[pat_id] = stats

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

        # Compaction not needed — temporal uses Lance append mode (one dataset per pattern)

        return str(self.output_path)

    def incremental_update(
        self,
        pattern_id: str,
        changed_entities: pa.Table | None = None,
        deleted_keys: list[str] | None = None,
        recalibrate: str = "auto",
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

        # 3. Resolve geometry Lance path
        lance_path = str(
            self.output_path / "geometry" / pattern_id / f"v={version}" / "data.lance"
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
            deltas, delta_norms, shape_vectors = compute_entity_geometry(
                changed_entities, mu, sigma,
                enriched_relations, event_dims_meta, dim_weights,
                prop_columns=prop_cols,
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
                "n_anomalous_dims": pa.array(n_anom_dims, type=pa.int32()),
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

        # 10. Recompute delta_rank_pct (O(N) global)
        writer = GDSWriter(str(self.output_path))
        writer.recompute_delta_rank_pct(pattern_id, version=version)

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

        theta_norm = float(np.linalg.norm(ps.theta))

        # 2. Write geometry in chunks
        for start in range(0, n, GEOMETRY_CHUNK_SIZE):
            end = min(start + GEOMETRY_CHUNK_SIZE, n)
            chunk_table = self._build_geometry_slice(
                pat, start, end,
                ps.deltas, ps.delta_norms, ps.delta_rank_pcts,
                theta_norm, ps.fk_arrays, ps.conformal_p,
                ps.is_anomaly_arr, ps.n_anom_dims,
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
            ),
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

        entity_line = self._lines[pat.entity_line]
        entity_table = entity_line.table
        n = len(entity_table)
        D_rel = len(pat.relations)
        D_event = len(pat.event_dimensions)
        chunk_size = GEOMETRY_CHUNK_SIZE

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
        D_max = D_rel + D_event + len(tracked)
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
        # Dims order: [relations] + [event dims] + [included props]
        D_base = D_rel + D_event
        if prop_columns:
            keep_dims = list(range(D_base)) + [D_base + j for j in prop_indices]
            mu = mu_full[keep_dims]
            sigma = sigma_full[keep_dims]
        else:
            keep_dims = list(range(D_base))
            mu = mu_full[:D_base]
            sigma = sigma_full[:D_base]

        D_full = len(mu)

        # Apply SIGMA_EPS_PROP for binary prop columns
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
        # _build_geometry_slice reads entity_table via self._lines and
        # slices [start:end]. We temporarily swap the entity table to
        # the current chunk so that start=0, end=cn addresses the chunk.

        orig_entity_table = entity_line.table

        try:
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_entity_table = orig_entity_table.slice(
                    start, end - start,
                )
                cn = end - start
                shapes, fk_slices = self._build_shape_chunk(
                    pat, chunk_entity_table,
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

                # is_anomaly
                is_anomaly_arr = (
                    (theta_norm > 0.0) & (chunk_norms >= theta_norm)
                )

                # Per-dim anomaly count using reservoir thresholds
                abs_chunk_deltas = np.abs(chunk_deltas)
                exceeds = (
                    abs_chunk_deltas > per_dim_thresholds
                ).astype(np.int32)
                chunk_n_anom_dims = exceeds.sum(axis=1).astype(np.int32)

                # Swap entity table to chunk for _build_geometry_slice
                entity_line.table = chunk_entity_table
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
                )

                write_chunk_fn(
                    self.output_path, pat_id, geom_table, version=1,
                )
        finally:
            entity_line.table = orig_entity_table

        # Finalize: compact fragments, build indices
        finalize_fn(self.output_path, pat_id, version=1)

        # Persist geometry stats cache — read stored is_anomaly from finalized Lance
        import lance as _lance
        _geom_ds = _lance.dataset(
            str(self.output_path / "geometry" / pat_id / "v=1" / "data.lance"),
        )
        all_is_anomaly = _geom_ds.to_table(
            columns=["is_anomaly"],
        )["is_anomaly"].to_numpy(zero_copy_only=False)
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
            ),
        )

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
        from hypertopos.builder.derived import _parse_time_window
        from hypertopos.storage.writer import _write_lance

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

        # 4. Parse timestamps to epoch seconds
        ts_col = event_table[time_col]
        if pa.types.is_timestamp(ts_col.type):
            unit = ts_col.type.unit
            divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
            divisor = divisors.get(unit, 1e6)
            epoch_arr = pc.divide(
                pc.cast(pc.cast(ts_col, pa.int64()), pa.float64()), divisor,
            )
        elif pa.types.is_floating(ts_col.type) or pa.types.is_integer(ts_col.type):
            epoch_arr = pc.cast(ts_col, pa.float64())
        else:
            from hypertopos.engine.chains import parse_timestamps_to_epoch
            epoch_arr = pa.chunked_array([pa.array(
                parse_timestamps_to_epoch(ts_col.to_pylist()),
                type=pa.float64(),
            )])

        # 5. Compute bucket IDs
        window_secs = _parse_time_window(time_window)
        min_ts = pc.min(epoch_arr).as_py()
        if min_ts is None:
            return {}
        diff = pc.subtract(epoch_arr, min_ts)
        bucket_arr = pc.cast(
            pc.floor(pc.divide(pc.cast(diff, pa.float64()), window_secs)),
            pa.int64(),
        )
        bucket_np = bucket_arr.to_numpy(zero_copy_only=False).astype(np.int64)
        n_buckets = int(bucket_np.max()) + 1

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

        from hypertopos.storage.writer import _TEMPORAL_SCHEMA, GDSWriter

        writer = GDSWriter(str(self.output_path))
        result: dict[str, int] = {}

        def _build_temporal_for_pattern(
            pat_id: str, pat: _PatternReg,
        ) -> tuple[str, int]:
            pat_meta = sphere_data["patterns"].get(pat_id)
            if pat_meta is None:
                return pat_id, 0

            relations_meta = pat_meta.get("relations", [])
            prop_columns = pat_meta.get("prop_columns", [])

            anchor_line_reg = self._lines[pat.entity_line]
            anchor_keys = anchor_line_reg.table["primary_key"]
            anchor_keys_list = anchor_keys.to_pylist()
            n_anchor = len(anchor_keys_list)

            lance_dir = self.output_path / "temporal" / pat_id
            lance_dir.mkdir(parents=True, exist_ok=True)
            lance_path = lance_dir / "data.lance"

            # Pre-compute the full shape tensor: (n_anchor, n_buckets, D)
            shape_tensor = self._precompute_shape_tensor(
                pat_id, event_table, bucket_np, n_buckets,
                anchor_keys, anchor_keys_list, n_anchor,
                relations_meta, prop_columns,
            )

            # Build all bucket tables then write in a single Lance append
            D = shape_tensor.shape[2]
            pk_col = pa.array(anchor_keys_list, type=pa.string())
            _zeros = pa.array(np.zeros(n_anchor, dtype=np.int32))
            deform_broadcast = pa.array(
                ["window_snapshot"], type=pa.string(),
            ).take(_zeros)
            ver_broadcast = pa.array([1], type=pa.int32()).take(_zeros)
            null_str_col = pa.array([None] * n_anchor, type=pa.string())

            all_tables: list[pa.Table] = []
            for bucket_idx in range(n_buckets):
                mask = bucket_np == bucket_idx
                if not mask.any():
                    continue

                window_start_epoch = min_ts + bucket_idx * window_secs
                window_start = datetime.fromtimestamp(
                    window_start_epoch, tz=UTC,
                )

                # Shape column from pre-computed tensor via FixedSizeListArray
                shape_data = shape_tensor[:, bucket_idx, :]  # (n_anchor, D)
                if D > 0:
                    shape_col = pa.FixedSizeListArray.from_arrays(
                        pa.array(shape_data.ravel(), type=pa.float32()),
                        list_size=D,
                    ).cast(pa.list_(pa.float32()))
                else:
                    shape_col = pa.array(
                        [[] for _ in range(n_anchor)],
                        type=pa.list_(pa.float32()),
                    )

                # Broadcast constant columns
                slice_col = pa.array(
                    [int(bucket_idx)], type=pa.int32(),
                ).take(_zeros)
                ts_col_out = pa.array(
                    [window_start], type=pa.timestamp("us", tz="UTC"),
                ).take(_zeros)

                all_tables.append(pa.table(
                    {
                        "primary_key": pk_col,
                        "slice_index": slice_col,
                        "timestamp": ts_col_out,
                        "deformation_type": deform_broadcast,
                        "shape_snapshot": shape_col,
                        "pattern_ver": ver_broadcast,
                        "changed_property": null_str_col,
                        "changed_line_id": null_str_col,
                    },
                    schema=_TEMPORAL_SCHEMA,
                ))

            slices_written = len(all_tables)
            if all_tables:
                combined = pa.concat_tables(all_tables)
                mode = "append" if lance_path.exists() else "create"
                _write_lance(combined, str(lance_path), mode=mode)

            # 8. Compute per-window population centroids for pi11/pi12
            if slices_written > 0:
                mu = np.array(pat_meta["mu"], dtype=np.float32)
                sigma = np.array(pat_meta["sigma_diag"], dtype=np.float32)
                sigma_safe = np.where(sigma < 1e-6, 1.0, sigma)
                theta_norm = float(np.linalg.norm(
                    np.array(pat_meta["theta"], dtype=np.float32),
                ))

                centroids: list[dict] = []
                for b in range(n_buckets):
                    window_shapes = shape_tensor[:, b, :]
                    active_mask = np.any(window_shapes != 0, axis=1)
                    n_active = int(active_mask.sum())
                    if n_active == 0:
                        continue
                    active_shapes = window_shapes[active_mask]
                    active_deltas = (active_shapes - mu) / sigma_safe
                    centroid = active_deltas.mean(axis=0).tolist()

                    anom_rate = 0.0
                    if theta_norm > 0:
                        norms = np.sqrt(np.einsum('ij,ij->i', active_deltas, active_deltas))
                        anom_rate = float(
                            (norms > theta_norm).sum() / len(norms),
                        )

                    ws_epoch = min_ts + b * window_secs
                    we_epoch = min_ts + (b + 1) * window_secs
                    centroids.append({
                        "window_start": datetime.fromtimestamp(
                            ws_epoch, tz=UTC,
                        ),
                        "window_end": datetime.fromtimestamp(
                            we_epoch, tz=UTC,
                        ),
                        "centroid": centroid,
                        "entity_count": n_active,
                        "anomaly_rate": anom_rate,
                    })

                if centroids:
                    writer.write_temporal_centroids(pat_id, centroids)

            # 9. Compute rolling z-scores from shape tensor
            if slices_written >= 3:
                max_rolling_z = self._compute_max_rolling_z(
                    shape_tensor, n_anchor, n_buckets,
                )
                # Write max_rolling_z to geometry (update existing Lance)
                self._write_max_rolling_z(
                    pat_id, anchor_keys_list, max_rolling_z,
                )

            # 10. Build trajectory vectors directly from shape tensor
            if slices_written >= 2:
                active_mask = np.any(
                    shape_tensor.reshape(n_anchor, -1) != 0, axis=1,
                )
                if active_mask.any():
                    active_tensor = shape_tensor[active_mask]  # (n_active, n_buckets, D)
                    active_pks = [
                        anchor_keys_list[i] for i in np.flatnonzero(active_mask)
                    ]
                    _mu = (
                        np.array(pat_meta["mu"], dtype=np.float32)
                        if pat_meta.get("mu") else None
                    )
                    _sig = (
                        np.array(pat_meta["sigma_diag"], dtype=np.float32)
                        if pat_meta.get("sigma_diag") else None
                    )
                    writer.write_trajectory_from_tensor(
                        pat_id, active_tensor, active_pks,
                        bucket_timestamps=[
                            datetime.fromtimestamp(min_ts + b * window_secs, tz=UTC)
                            for b in range(n_buckets)
                        ],
                        mu=_mu, sigma_diag=_sig,
                    )

            # 11. Finalize — compact + BTREE index
            if slices_written > 0:
                writer.compact_temporal(pat_id)
                writer.build_temporal_index(pat_id)

            return pat_id, slices_written

        # 7. Process each anchor pattern — parallel when >1
        if len(patterns_to_process) > 1:
            with ThreadPoolExecutor(max_workers=len(patterns_to_process)) as pool:
                futures = [
                    pool.submit(_build_temporal_for_pattern, pid, p)
                    for pid, p in patterns_to_process.items()
                ]
                for fut in futures:
                    pid, n_slices = fut.result()
                    result[pid] = n_slices
        else:
            for pat_id, pat in patterns_to_process.items():
                pid, n_slices = _build_temporal_for_pattern(pat_id, pat)
                result[pid] = n_slices

        return result

    @staticmethod
    def _compute_max_rolling_z(
        shape_tensor: np.ndarray,
        n_anchor: int,
        n_buckets: int,
    ) -> np.ndarray:
        """Compute max rolling z-score across temporal windows.

        For each entity, for each time step t >= 2, compute:
          z[t] = |shape[t] - mean(shape[0..t-1])| / std(shape[0..t-1])
        Return the maximum z-score across all (t, dim) per entity.
        """
        max_z = np.zeros(n_anchor, dtype=np.float32)
        for t in range(2, n_buckets):
            # Skip empty buckets (all-zero shape = no data in this window)
            current = shape_tensor[:, t, :]
            if current.sum() == 0:
                continue
            window = shape_tensor[:, :t, :]
            mu_w = window.mean(axis=1)
            std_w = np.maximum(window.std(axis=1), 0.01)
            z = np.abs((current - mu_w) / std_w)
            z_max = z.max(axis=1)
            max_z = np.maximum(max_z, z_max)
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
            self.output_path / "geometry" / pat_id / "v=1" / "data.lance"
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
    ) -> np.ndarray:
        """Pre-compute shape tensor (n_anchor, n_buckets, D) in single-pass.

        For derived dims: one groupby(anchor_fk, bucket) per dim fills
        all windows at once. For graph features: one batched call per
        window. Static dims and props are filled once and broadcast.

        Returns (n_anchor, n_buckets, D) float32 array.
        """
        from hypertopos.builder.derived import (
            compute_graph_features,
        )

        pat = self._patterns[pat_id]
        n_rel = len(relations_meta)
        n_prop = len(prop_columns)
        D = n_rel + n_prop

        shape_tensor = np.zeros((n_anchor, n_buckets, D), dtype=np.float32)

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
            anchor_table = self._lines[pat.entity_line].table
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
            anchor_table = self._lines[pat.entity_line].table
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
            bucket_pa = pa.array(bucket_np, type=pa.int64())
            work_table = event_table.append_column("_bucket", bucket_pa)

            # Group by FK column for batching
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
                # Resolve FK column
                sample_spec = batch_dims[0][3]
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

                # Build multi-aggregate expression list (dedup identical cols)
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

                # ONE group_by for all dims sharing this FK
                grouped = gb_table.group_by(
                    [fk_group_col, "_bucket"],
                ).aggregate(agg_exprs)

                # Materialize FK + bucket columns once
                gk = grouped[fk_group_col].to_pylist()
                gb_col = grouped["_bucket"].to_numpy(
                    zero_copy_only=False,
                ).astype(np.int64)

                # Scatter each dim's results
                for j, (result_col, em) in dim_to_result.items():
                    gv = grouped[result_col].to_numpy(
                        zero_copy_only=False,
                    ).astype(np.float64)
                    for row_idx in range(len(gk)):
                        idx = key_to_idx.get(gk[row_idx])
                        if idx is None:
                            continue
                        b = int(gb_col[row_idx])
                        if 0 <= b < n_buckets:
                            v = gv[row_idx]
                            if not np.isnan(v):
                                shape_tensor[idx, b, j] = (
                                    max(0.0, min(v, em)) / em
                                )

        # --- D. Graph features: batch all features per window ---
        if graph_dims:
            # Group graph dims by spec (same from_col/to_col) for batching
            spec_to_dims: dict[int, list[tuple[int, dict, Any]]] = {}
            spec_map: dict[int, Any] = {}
            for j, rel_meta, rel, gf_spec in graph_dims:
                spec_key = id(gf_spec)
                if spec_key not in spec_to_dims:
                    spec_to_dims[spec_key] = []
                    spec_map[spec_key] = gf_spec
                spec_to_dims[spec_key].append((j, rel_meta, rel))

            for bucket_idx in range(n_buckets):
                mask = bucket_np == bucket_idx
                if not mask.any():
                    continue

                indices = np.where(mask)[0]
                filtered_events = event_table.take(
                    pa.array(indices, type=pa.int64()),
                )

                for spec_key, dims_list in spec_to_dims.items():
                    gf_spec = spec_map[spec_key]
                    # Batch all features for this spec in one call
                    all_features = [rel.fk_col for _, _, rel in dims_list]
                    feature_results = compute_graph_features(
                        filtered_events, anchor_keys,
                        gf_spec.from_col, gf_spec.to_col, all_features,
                    )
                    for j, rel_meta, rel in dims_list:
                        feat_name = rel.fk_col
                        if feat_name in feature_results:
                            values, _em = feature_results[feat_name]
                            edge_max = rel_meta.get("edge_max")
                            em = edge_max if edge_max is not None else 1
                            shape_tensor[
                                :, bucket_idx, j
                            ] = np.clip(values, 0, em) / em

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
