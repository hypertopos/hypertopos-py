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
