# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from hypertopos.model.sphere import Pattern


@dataclass
class Point:
    primary_key: str
    line_id: str
    version: int
    status: Literal["active", "expired", "ghost"]
    properties: dict[str, Any]
    created_at: datetime
    changed_at: datetime


@dataclass
class Edge:
    line_id: str
    point_key: str
    status: Literal["alive", "dead"]
    direction: Literal["in", "out", "self"]
    is_jumpable: bool = True  # False for continuous-mode (edge_max) edges with empty point_key

    def is_alive(self) -> bool:
        return self.status == "alive"


@dataclass
class Polygon:
    primary_key: str
    pattern_id: str
    pattern_ver: int
    pattern_type: Literal["anchor", "event"]
    scale: int
    delta: np.ndarray
    delta_norm: float
    is_anomaly: bool
    edges: list[Edge]
    last_refresh_at: datetime
    updated_at: datetime
    alias_id: str | None = None
    alias_ver: int | None = None
    delta_alias: np.ndarray | None = None
    is_anomaly_alias: bool | None = None
    delta_rank_pct: float | None = None

    def is_event(self) -> bool:
        return self.pattern_type == "event"

    def is_anchor(self) -> bool:
        return not self.is_event()

    def edges_for_line(self, line_id: str) -> list[Edge]:
        return [e for e in self.edges if e.line_id == line_id]

    def alive_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.is_alive()]

    def count_alive_edges_to(self, line_id: str) -> int:
        """Count alive edges to a specific line."""
        return sum(1 for e in self.edges if e.line_id == line_id and e.is_alive())


@dataclass
class SolidSlice:
    slice_index: int
    timestamp: datetime
    deformation_type: Literal["internal", "edge", "structural"]
    delta_snapshot: np.ndarray
    delta_norm_snapshot: float
    pattern_ver: int
    changed_property: str | None
    changed_line_id: str | None
    added_edge: Edge | None

    def prop_column_states(self, pattern: Pattern) -> dict[str, bool]:
        """Inverse z-score: recover boolean property states from delta_snapshot."""
        n_rel = len(pattern.relations)
        return {
            name: (
                float(self.delta_snapshot[n_rel + j])
                * float(pattern.sigma_diag[n_rel + j])
                + float(pattern.mu[n_rel + j])
            ) > 0.5
            for j, name in enumerate(pattern.prop_columns)
            if (n_rel + j) < len(self.delta_snapshot)
        }

    def delta_relations(self, pattern: Pattern) -> list[float]:
        """Return only the relation dimensions of delta_snapshot (exclude prop columns)."""
        n_rel = len(pattern.relations)
        return [round(float(x), 4) for x in self.delta_snapshot[:n_rel]]


@dataclass
class Solid:
    primary_key: str
    pattern_id: str
    base_polygon: Polygon
    slices: list[SolidSlice]
    _timestamps: list[datetime] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._timestamps = [s.timestamp for s in self.slices]

    def slice_at(self, timestamp: datetime) -> SolidSlice | None:
        idx = bisect.bisect_right(self._timestamps, timestamp) - 1
        if idx < 0:
            return None
        return self.slices[idx]
