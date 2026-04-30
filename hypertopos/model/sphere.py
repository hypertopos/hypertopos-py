# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np


@dataclass
class LayerStorage:
    format: Literal["lance"] = "lance"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageConfig:
    points: LayerStorage = field(default_factory=LayerStorage)
    geometry: LayerStorage = field(default_factory=LayerStorage)
    temporal: LayerStorage = field(default_factory=LayerStorage)
    invalidation_log: LayerStorage = field(default_factory=LayerStorage)
    forecast: LayerStorage = field(default_factory=lambda: LayerStorage(format="lance"))


@dataclass
class PartitionConfig:
    mode: Literal["static", "liquid"]
    columns: list[str]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupStats:
    """Per-group population statistics for segmented anomaly detection."""

    mu: np.ndarray
    sigma_diag: np.ndarray
    theta: np.ndarray
    population_size: int


@dataclass
class GMMComponent:
    """Single Gaussian mixture component for subpopulation-aware anomaly detection."""

    mu: np.ndarray
    sigma_diag: np.ndarray
    theta: np.ndarray
    population_size: int


@dataclass
class RelationDef:
    line_id: str
    direction: Literal["in", "out", "self"]
    required: bool
    display_name: str | None = None
    interpretation: str | None = None


@dataclass
class EventDimDef:
    """Continuous dimension definition parsed from sphere.json."""
    column: str
    edge_max: float
    display_name: str | None = None


@dataclass
class Pattern:
    pattern_id: str
    entity_type: str
    pattern_type: Literal["anchor", "event"]
    relations: list[RelationDef]
    mu: np.ndarray
    sigma_diag: np.ndarray
    theta: np.ndarray
    population_size: int
    computed_at: datetime
    version: int
    status: Literal["prerelease", "production", "deprecated", "orphaned"]
    edge_max: np.ndarray | None = None
    description: str | None = None
    last_calibrated_at: datetime | None = None
    prop_columns: list[str] = field(default_factory=list)
    excluded_properties: list[str] = field(default_factory=list)
    group_stats: dict[str, GroupStats] | None = None
    group_by_property: str | None = None
    dimension_weights: np.ndarray | None = None
    gmm_components: list[GMMComponent] | None = None
    cholesky_inv: np.ndarray | None = None
    entity_line_id: str | None = None
    event_dimensions: list[EventDimDef] = field(default_factory=list)
    dim_percentiles: dict[str, dict[str, float]] | None = None
    timestamp_col: str | None = None
    dimension_kinds: list[str] | None = None

    def delta_dim(self) -> int:
        return len(self.relations) + len(self.event_dimensions) + len(self.prop_columns)

    @property
    def theta_norm(self) -> float:
        """L2 norm of the anomaly threshold vector."""
        return float(np.linalg.norm(self.theta))

    @property
    def dim_labels(self) -> list[str]:
        """Human-readable dimension labels: relation display_names + event dims + prop_columns."""
        labels = [r.display_name if r.display_name else r.line_id for r in self.relations]
        labels.extend(
            ed.display_name or ed.column for ed in self.event_dimensions
        )
        labels.extend(self.prop_columns)
        return labels

    @property
    def max_hub_score(self) -> float | None:
        """Theoretical maximum hub score (sum of edge_max). None if binary mode."""
        if self.edge_max is None:
            return None
        return sum(float(v) for v in self.edge_max)

    @property
    def is_continuous(self) -> bool:
        """True if pattern uses continuous edge encoding (edge_max is set)."""
        return self.edge_max is not None

    def effective_sample_size(self, sample_pct: float) -> int:
        """Convert sample_pct to absolute sample_size based on population_size."""
        return max(1, int(self.population_size * sample_pct))

    def dim_index(self, dim_name: str) -> int:
        """Resolve dimension name to delta vector index.

        Searches in order: relations (line_id, display_name),
        event dimensions (column, display_name), prop_columns.
        Raises ValueError if dim_name is not found.
        """
        k = len(self.relations)
        for i, rel in enumerate(self.relations):
            if rel.line_id == dim_name or (rel.display_name and rel.display_name == dim_name):
                return i
        k2 = k + len(self.event_dimensions)
        for j, ed in enumerate(self.event_dimensions):
            if ed.column == dim_name or (ed.display_name and ed.display_name == dim_name):
                return k + j
        for j, prop in enumerate(self.prop_columns):
            if prop == dim_name:
                return k2 + j
        available = [
            rel.line_id + (f" ({rel.display_name})" if rel.display_name else "")
            for rel in self.relations
        ] + [
            ed.column + (f" ({ed.display_name})" if ed.display_name else "")
            for ed in self.event_dimensions
        ] + self.prop_columns
        raise ValueError(
            f"Dimension '{dim_name}' not found in pattern relations. "
            f"Available: {available}"
        )


@dataclass
class ColumnSchema:
    name: str
    type: str


@dataclass
class Line:
    line_id: str
    entity_type: str
    line_role: Literal["anchor", "event"]
    pattern_id: str
    partitioning: PartitionConfig
    versions: list[int]
    description: str | None = None
    columns: list[ColumnSchema] | None = None
    fts_columns: list[str] | str | None = None
    source_id: str | None = None

    def current_version(self) -> int:
        return max(self.versions)

    def has_fts(self) -> bool:
        """Return True if this line has any FTS columns configured."""
        if self.fts_columns is None:
            return self.line_role != "event"
        if self.fts_columns == "all":
            return True
        if isinstance(self.fts_columns, list):
            return len(self.fts_columns) > 0
        return False


@dataclass
class CuttingPlane:
    """Hyperplane in delta-space defining segment membership geometrically.

    w·delta >= b → entity is "in segment".
    """

    normal: list[float]  # w — one weight per delta dimension
    bias: float  # b — threshold

    def signed_distance(self, delta: np.ndarray) -> float:
        w = np.array(self.normal, dtype=np.float32)
        norm_w = float(np.linalg.norm(w))
        if norm_w == 0.0:
            raise ValueError(
                "CuttingPlane normal vector has zero norm — cannot compute signed distance"
            )
        return float((np.dot(w, delta) - self.bias) / norm_w)

    def signed_distances_batch(self, deltas: np.ndarray) -> np.ndarray:
        """Vectorized signed distance for a (n, d) delta matrix. Returns shape (n,)."""
        w = np.array(self.normal, dtype=np.float32)
        norm_w = float(np.linalg.norm(w))
        if norm_w == 0.0:
            raise ValueError(
                "CuttingPlane normal vector has zero norm — cannot compute signed distance"
            )
        return (deltas @ w - self.bias) / norm_w

    def contains(self, delta: np.ndarray) -> bool:
        w = np.array(self.normal, dtype=np.float32)
        return float(np.dot(w, delta)) >= self.bias


@dataclass
class AliasFilter:
    include_relations: list[str]
    edge_conditions: dict[str, Any] = field(default_factory=dict)
    cutting_plane: CuttingPlane | None = None


@dataclass
class DerivedPattern:
    mu: np.ndarray
    sigma_diag: np.ndarray
    theta: np.ndarray
    population_size: int
    computed_at: datetime


@dataclass
class Alias:
    alias_id: str
    base_pattern_id: str
    filter: AliasFilter
    derived_pattern: DerivedPattern
    version: int
    status: Literal["prerelease", "production", "deprecated", "orphaned"]


@dataclass
class Sphere:
    sphere_id: str
    name: str
    base_path: str
    lines: dict[str, Line] = field(default_factory=dict)
    patterns: dict[str, Pattern] = field(default_factory=dict)
    aliases: dict[str, Alias] = field(default_factory=dict)
    storage: StorageConfig = field(default_factory=StorageConfig)
    description: str | None = None
    reverse_index: dict[str, list[str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        idx: dict[str, list[str]] = defaultdict(list)
        for pattern in self.patterns.values():
            for rel in pattern.relations:
                idx[rel.line_id].append(pattern.pattern_id)
        self.reverse_index = dict(idx)

        # Build source_id → [line_ids] index for sibling discovery
        # Skip derived dimension lines (_d_*) — each has unique source_id
        self._source_groups: dict[str, list[str]] = defaultdict(list)
        for lid, line in self.lines.items():
            if line.source_id and not lid.startswith("_d_"):
                self._source_groups[line.source_id].append(lid)

    def sibling_lines(self, line_id: str) -> list[str]:
        """Return line_ids sharing the same source_id (excluding self)."""
        line = self.lines.get(line_id)
        if not line or not line.source_id:
            return []
        return [lid for lid in self._source_groups[line.source_id] if lid != line_id]

    def entity_line(self, pattern_id: str) -> str | None:
        """Return the line_id of the anchor line for the given pattern, or None if not found."""
        pat = self.patterns.get(pattern_id)
        if pat and pat.entity_line_id:
            return pat.entity_line_id
        # Fallback for spheres without entity_line_id on pattern
        for line_id, line in self.lines.items():
            if line.pattern_id == pattern_id and line.line_role == "anchor":
                return line_id
        return None

    def event_line(self, pattern_id: str) -> str | None:
        """Return the line_id of the event-role line for this pattern, or None if not found."""
        pat = self.patterns.get(pattern_id)
        if pat and pat.entity_line_id:
            return pat.entity_line_id
        # Fallback for spheres without entity_line_id on pattern
        for line_id, line in self.lines.items():
            if line.pattern_id == pattern_id and line.line_role == "event":
                return line_id
        return None


@dataclass(frozen=True)
class CalibrationFit:
    """Frozen snapshot of a pattern's statistical fit at one calibration epoch.

    Captures everything that changes when a pattern is re-fitted on the same
    schema: population statistics (mu/sigma/theta), dimension metadata that
    depends on the fit (kinds, weights, percentiles), and segmented variants
    (group_stats, gmm_components). Structural definition (relations,
    event_dimensions, prop_columns) lives in the parent Pattern and is not
    duplicated here. Schema change invalidates and discards all prior
    CalibrationFit entries for a pattern.
    """

    pattern_id: str
    calibration_epoch: int
    schema_version: int
    schema_hash: str
    mu: np.ndarray
    sigma_diag: np.ndarray
    theta: np.ndarray
    population_size: int
    dimension_weights: np.ndarray | None
    dimension_kinds: list[str] | None
    dim_percentiles: dict[str, dict[str, float]] | None
    group_stats: dict[str, GroupStats] | None
    gmm_components: list[GMMComponent] | None
    edge_max: np.ndarray | None
    computed_at: datetime
    last_calibrated_at: datetime


@dataclass(frozen=True)
class DimensionDrift:
    """Per-dimension drift between two calibration epochs of the same pattern.

    `mu_delta_normalized` is `(mu_to - mu_from) / sigma_from` — the z-score
    of the centroid shift. For dimensions with `sigma_from == 0`
    (degenerate), the normalization falls back to raw `mu_delta`.
    """

    dim_index: int
    dim_kind: str | None
    mu_from: float
    mu_to: float
    mu_delta: float
    mu_delta_normalized: float
    sigma_from: float
    sigma_to: float
    sigma_delta: float
    theta_from: float
    theta_to: float
    theta_delta: float


@dataclass(frozen=True)
class CalibrationDriftReport:
    """Drift report between two calibration epochs of one pattern.

    `overall_drift_rms` is `||mu_delta_normalized||_2 / sqrt(D)` — RMS drift
    per dimension in σ units, comparable across patterns of different
    dimensionality. `top_drifted` is sorted by `|mu_delta_normalized|` desc.
    `per_dimension` is populated only when the caller passes verbose=True.
    """

    pattern_id: str
    v_from: int
    v_to: int
    schema_hash: str
    population_size_from: int
    population_size_to: int
    overall_drift_rms: float
    top_drifted: list[DimensionDrift]
    per_dimension: list[DimensionDrift] | None


@dataclass(frozen=True)
class DimensionDecomposition:
    """Per-dimension intrinsic vs extrinsic split for one entity drift.

    `total` = `delta_b - delta_a` per-dim z-score change.
    `intrinsic` = `(shape_b - shape_a) / sigma_v1[i]` — entity's own structural change
    measured under the earlier-epoch sigma.
    `extrinsic` = `total - intrinsic` — residual attributable to population recalibration.
    `intrinsic_fraction` = `intrinsic^2 / (intrinsic^2 + extrinsic^2)`, bounded [0, 1].
    Falls back to 0.0 when total is exactly zero (no change either component).
    """

    dim_index: int
    dim_kind: str | None
    dim_label: str | None
    total: float
    intrinsic: float
    extrinsic: float
    intrinsic_fraction: float


@dataclass(frozen=True)
class IntrinsicExtrinsicReport:
    """M3 decomposition report for a single entity drift between two epochs.

    `intrinsic_displacement` = `||intrinsic||_2` (entity-driven L2 magnitude).
    `extrinsic_displacement` = `||extrinsic||_2` (population-driven L2 magnitude).
    `total_displacement` = `||total||_2`.
    `intrinsic_fraction` = `||intrinsic||^2 / (||intrinsic||^2 + ||extrinsic||^2)`,
    bounded [0, 1] — what proportion of the squared change is entity-caused.
    `top_dimensions` is sorted by `|total|` desc.
    `per_dimension` is populated only when the caller passes `verbose=True`.
    """

    pattern_id: str
    entity_key: str
    v_from: int
    v_to: int
    schema_hash: str
    timestamp_from: datetime
    timestamp_to: datetime
    intrinsic_displacement: float
    extrinsic_displacement: float
    total_displacement: float
    intrinsic_fraction: float
    top_dimensions: list[DimensionDecomposition]
    per_dimension: list[DimensionDecomposition] | None


@dataclass(frozen=True)
class DimensionContribution:
    """Per-dimension breakdown of one entity's (or group's) influence on the
    coordinate system.

    Hidden-influencer matrix. See
    `engine.geometry._compute_leave_one_out_impact` for math.
    """
    dim_index: int
    dim_kind: str | None
    dim_label: str | None
    mu_shift: float
    sigma_shift: float
    contribution: float


@dataclass(frozen=True)
class InfluenceEntry:
    """Per-entity influence record returned by find_calibration_influencers."""
    entity_key: str
    mu_impact: float
    sigma_impact: float
    total_impact: float
    delta_norm: float
    classification: str
    top_dim_contributions: list[DimensionContribution]
    cascading_flip_count: int | None = None


@dataclass(frozen=True)
class InfluenceReport:
    """Aggregate report from find_calibration_influencers."""
    pattern_id: str
    pattern_version: int
    population_size: int
    high_threshold_pct: float
    total_impact_threshold: float
    theta_norm: float
    classify_filter: str
    cell_counts: dict[str, int]
    entries: list[InfluenceEntry]


@dataclass(frozen=True)
class GroupInfluenceReport:
    """Per-group influence record from find_group_influence (caller-supplied form)."""
    pattern_id: str
    pattern_version: int
    group_index: int
    member_count: int
    members: list[str]
    mu_impact_set: float
    sigma_impact_set: float
    total_impact_set: float
    sum_individual_impacts: float
    reinforcing_factor: float
    top_dim_contributions: list[DimensionContribution]


@dataclass(frozen=True)
class DimPairLeadLag:
    """One (dim_a, dim_b) cross-pattern lead-lag entry from find_lead_lag.

    `lag` is the peak lag in epochs (positive: dim_a in pattern A leads
    dim_b in pattern B). `correlation` is the Pearson correlation of the
    differenced centroid coordinate series at that lag. `q_value` is BH
    or Storey-adjusted across all D_A * D_B pairs at the same `fdr_alpha`;
    `is_significant` is `q_value < fdr_alpha`.
    """
    dim_index_a: int
    dim_index_b: int
    dim_label_a: str | None
    dim_label_b: str | None
    lag: int
    correlation: float
    p_value: float
    q_value: float
    is_significant: bool


@dataclass(frozen=True)
class LeadLagReport:
    """Cross-pattern lead-lag report.

    Three answer levels live in one report:
      1. Headline (population, scalar): `lag`, `correlation` from differenced
         centroid drift series of P_A vs P_B.
      2. Per-dim drill-down: `top_dim_pairs` (top 10 by ascending q_value);
         full matrix in `per_dim_pairs` when verbose=True.
      3. Per-entity drill-down: when `entity_key` is given, the population
         centroid is replaced by that entity's own delta trajectory.

    Significance: peak Bonferroni-adjusted threshold (`max_corr_threshold`)
    for the population peak; BH or Storey FDR across D_A * D_B for per-dim.
    `bartlett_ci_95` reports the unadjusted single-test 95% CI for
    transparency; `is_significant` uses `max_corr_threshold` (peak-adjusted).
    """
    pattern_a: str
    pattern_b: str
    entity_key: str | None
    n_epochs_used: int
    n_dropped_a: int
    n_dropped_b: int
    cohort_size: int
    cohort_dropped: int | None
    timestamp_from: datetime
    timestamp_to: datetime
    schema_hash_a: str
    schema_hash_b: str
    lag: int
    correlation: float
    centroid_drift_series_a: list[float]
    centroid_drift_series_b: list[float]
    lag_volatility: int
    correlation_volatility: float
    volatility_series_a: list[float]
    volatility_series_b: list[float]
    agreement: str
    bartlett_ci_95: float
    max_corr_threshold: float
    is_significant: bool
    fdr_alpha: float
    fdr_method: str
    n_dim_pairs: int
    n_significant_pairs: int
    top_dim_pairs: list[DimPairLeadLag]
    per_dim_pairs: list[DimPairLeadLag] | None
    reliability: str
    max_lag: int
    correlation_by_lag: list[float]
    coverage_warning: bool
    degenerate_signal: bool


@dataclass(frozen=True)
class HopPredicate:
    """Per-hop constraints for the declarative motif API.

    Used by ``GDSNavigator.find_motif_by_hops`` to describe a custom motif
    as a list of per-hop predicates instead of a closed-vocabulary
    ``motif_type``. The bounded MVP (0.6.0) supports ``amount_min`` /
    ``amount_max`` / ``time_delta_max_hours`` / ``direction`` /
    ``edge_dim_predicates``; ``amount_ratio_to_prev`` and
    ``require_anomalous_entity`` ship in a follow-up release.
    """

    amount_min: float | None = None
    amount_max: float | None = None
    time_delta_max_hours: float | None = None
    direction: Literal["forward", "reverse", "any"] = "forward"
    # ``edge_dim_predicates`` maps a dim name (e.g. ``pair_edge_count``) to
    # a (operator, value) tuple. Operators: ``"<"``, ``"<="``, ``">"``,
    # ``">="``, ``"=="``. Dim names must exist in the event pattern's
    # edge_features sidecar (declared via the ``edge_dimensions:`` YAML).
    edge_dim_predicates: dict[str, tuple[str, float]] = field(
        default_factory=dict,
    )
