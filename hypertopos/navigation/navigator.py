# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import dataclasses
import logging
import math
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hypertopos.model.objects import Edge, Point, Polygon, Solid
from hypertopos.model.sphere import (
    CalibrationDriftReport,
    CalibrationFit,
    DimensionDrift,
    IntrinsicExtrinsicReport,
)
from hypertopos.utils.arrow import delta_matrix_from_arrow

# Numeric stability bounds for motif-score products.
# ``_MOTIF_SCORE_EPSILON`` clamps non-zero underflow (product < 1e-30) UP
# so sorting stays stable; ``_MOTIF_SCORE_MAX`` clamps overflow (product
# > 1e300 or +inf) DOWN so JSON serialisation stays finite. The
# ``log_score`` companion field remains informative past both clamps.
_MOTIF_SCORE_EPSILON = 1e-30
_MOTIF_SCORE_MAX = 1e300

logger = logging.getLogger(__name__)

_NAVIGATION_RECOVERABLE_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    AttributeError,
    pa.ArrowInvalid,
    pa.ArrowTypeError,
)

if TYPE_CHECKING:
    from hypertopos.engine.adjacency import AdjacencyIndex
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Contract, Manifest
    from hypertopos.storage.reader import GDSReader


# ---------------------------------------------------------------------------
# GDSError hierarchy
# ---------------------------------------------------------------------------

class GDSError(Exception):
    """Base error for the entire hypertopos / GDS framework."""
    pass


class GDSNavigationError(GDSError):
    """Navigation-related errors (π operations, goto, position queries)."""
    pass


class GDSNoAliveEdgeError(GDSNavigationError):
    """π2 fails because no alive edge connects to the target line."""
    pass


class GDSPositionError(GDSNavigationError):
    """Current position type is incompatible with the requested operation."""
    pass


class GDSEntityNotFoundError(GDSNavigationError):
    """A primary-key / entity was not found in the specified line."""
    pass


class GDSStorageError(GDSError):
    """Storage-layer errors (files, I/O)."""
    pass


class GDSMissingFileError(GDSStorageError):
    """An expected data file was not found."""
    pass


class GDSCorruptedFileError(GDSStorageError):
    """A data file exists but its content is invalid or unreadable."""
    pass


class GDSVersionError(GDSError):
    """Version mismatch or requested version not found."""
    pass


# ---------------------------------------------------------------------------
# Similarity result container (backward-compatible list subclass)
# ---------------------------------------------------------------------------

class SimilarityResult(list):
    """List of (primary_key, distance) tuples with an optional degenerate_warning.

    Extends ``list`` so all existing callers that iterate, slice, or convert to
    ``dict()`` continue to work unchanged.  The extra ``degenerate_warning``
    attribute is ``None`` when the result set is healthy, or a descriptive
    string when >50 % of neighbors have distance = 0.
    """

    degenerate_warning: str | None

    def __init__(self, items: list[tuple[str, float]], *, degenerate_warning: str | None = None):
        super().__init__(items)
        self.degenerate_warning = degenerate_warning


# ---------------------------------------------------------------------------
# Root-cause trace — DAG node container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RootCauseNode:
    """One node in a root-cause trace DAG returned by ``trace_root_cause``.

    role: "root" | "structural_witness" | "edge_counterparty" | "hub" | "neighbor_contamination"
    evidence: free-form per-role evidence dict (top_dimensions, delta_norm, contagion_score, ...)
    children: downstream nodes (may be empty for leaves / bounded branches)
    """

    entity_key: str
    role: str
    severity: str
    evidence: dict[str, Any]
    children: list[RootCauseNode]


# ---------------------------------------------------------------------------
# Witness cohort discovery — config and result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WitnessCohortWeights:
    """Composite score weights for ``find_witness_cohort``.

    Defaults are sensible for fraud/AML use cases. Weights should sum to 1.0
    so the composite score stays in [0, 1].
    """

    delta: float = 0.40
    witness: float = 0.30
    trajectory: float = 0.20
    anomaly: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "witness": self.witness,
            "trajectory": self.trajectory,
            "anomaly": self.anomaly,
        }


@dataclasses.dataclass(frozen=True)
class WitnessCohortConfig:
    """Tunable parameters for ``find_witness_cohort``.

    Groups configuration that does not change call to call from the call
    arguments that do (primary_key, pattern_id, top_n). Construct once and
    reuse across multiple calls when running batches.
    """

    candidate_pool: int = 100
    min_witness_overlap: float = 0.0
    min_score: float = 0.0
    weights: WitnessCohortWeights = dataclasses.field(
        default_factory=WitnessCohortWeights,
    )
    use_trajectory: bool | None = None
    bidirectional_check: bool = True
    timestamp_cutoff: float | None = None


@dataclasses.dataclass(frozen=True)
class CohortMember:
    """A single member of a witness cohort — an entity geometrically peer
    of the target with explainable per-component scores.

    Returned inside ``WitnessCohortResult.members``.
    """

    primary_key: str
    score: float                            # final composite ∈ [0, 1]
    delta_similarity: float                 # exp(-distance / theta_norm)
    witness_overlap: float                  # Jaccard ∈ [0, 1]
    trajectory_alignment: float | None      # cos sim remapped to [0, 1], or None
    is_anomaly: bool
    delta_rank_pct: float
    explanation: str
    component_scores: dict[str, float]


@dataclasses.dataclass(frozen=True)
class WitnessCohortResult:
    """Ranked geometric peers (cohort) of a target entity.

    Returned by ``find_witness_cohort``. Members are entities that share the
    target's witness signature, are geometrically close in delta space, and
    are NOT already connected via the resolved edge table. This is an
    investigative ranking — not a forecast of future edges.
    """

    primary_key: str
    pattern_id: str
    edge_pattern_id: str
    members: list[CohortMember]
    excluded_existing_edges: int
    excluded_low_score: int
    candidate_pool_size: int
    weights_used: dict[str, float]
    summary: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class MotifSpec:
    """Dispatcher entry for a named structural motif.

    Enumerator signature: ``enumerate(nav, seed, pattern_id, time_window_hours,
    **kwargs) -> list[dict]`` — returns a list of motif instances, each with
    ``edges: list[tuple[str,str]]`` plus motif-specific fields.
    """

    enumerate: Any
    default_window_hours: int
    min_instances: int


def _derive_edge_line_ids(edges_list: list[dict] | None) -> list[str]:
    """Derive alive edge line_ids from a single row's edges struct list."""
    return [e["line_id"] for e in (edges_list or []) if e.get("status") == "alive"]


def _derive_edge_point_keys(edges_list: list[dict] | None) -> list[str]:
    """Derive alive edge point_keys from a single row's edges struct list."""
    return [e["point_key"] for e in (edges_list or []) if e.get("status") == "alive"]


def _derive_edge_line_ids_list(edges_col: list[list[dict] | None]) -> list[list[str]]:
    """Derive alive edge line_ids for each row from an edges column (to_pylist)."""
    return [_derive_edge_line_ids(row) for row in edges_col]


def _derive_edge_point_keys_list(edges_col: list[list[dict] | None]) -> list[list[str]]:
    """Derive alive edge point_keys for each row from an edges column (to_pylist)."""
    return [_derive_edge_point_keys(row) for row in edges_col]


def _derive_edge_line_ids_from_table(table: Any) -> list[list[str]]:
    """Derive alive edge line_ids from an Arrow table with edges column."""
    return _derive_edge_line_ids_list(table.column("edges").to_pylist())


# ---------------------------------------------------------------------------
# Entity-keys reconstruction helpers (for event geometry without edges column)
# ---------------------------------------------------------------------------

def _derive_line_ids_from_entity_keys(
    entity_keys: list[str] | None,
    relations: list,
) -> list[str]:
    """Derive alive edge line_ids from entity_keys + relations for one row."""
    keys = entity_keys or []
    return [
        rel.line_id for i, rel in enumerate(relations)
        if i < len(keys) and keys[i]
    ]


def _derive_point_keys_from_entity_keys(
    entity_keys: list[str] | None,
    relations: list,
) -> list[str]:
    """Derive alive edge point_keys from entity_keys + relations for one row."""
    keys = entity_keys or []
    return [
        keys[i] for i, rel in enumerate(relations)
        if i < len(keys) and keys[i]
    ]


def _derive_line_ids_from_entity_keys_list(
    entity_keys_col: list[list[str] | None],
    relations: list,
) -> list[list[str]]:
    """Derive alive edge line_ids for each row from entity_keys column."""
    return [_derive_line_ids_from_entity_keys(ek, relations) for ek in entity_keys_col]


def _derive_point_keys_from_entity_keys_list(
    entity_keys_col: list[list[str] | None],
    relations: list,
) -> list[list[str]]:
    """Derive alive edge point_keys for each row from entity_keys column."""
    return [_derive_point_keys_from_entity_keys(ek, relations) for ek in entity_keys_col]


def _table_edge_line_ids(table: Any, relations: list | None = None) -> list[list[str]]:
    """Derive alive edge line_ids from table — edges column or entity_keys fallback."""
    if "edges" in table.schema.names:
        return _derive_edge_line_ids_from_table(table)
    if "entity_keys" in table.schema.names and relations:
        return _derive_line_ids_from_entity_keys_list(
            table.column("entity_keys").to_pylist(), relations,
        )
    return [[] for _ in range(table.num_rows)]


def _table_edge_point_keys(table: Any, relations: list | None = None) -> list[list[str]]:
    """Derive alive edge point_keys from table — edges column or entity_keys fallback."""
    if "edges" in table.schema.names:
        return _derive_edge_point_keys_list(table.column("edges").to_pylist())
    if "entity_keys" in table.schema.names and relations:
        return _derive_point_keys_from_entity_keys_list(
            table.column("entity_keys").to_pylist(), relations,
        )
    return [[] for _ in range(table.num_rows)]


def _table_edge_line_and_point_keys(
    table: Any, relations: list | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Derive both alive edge line_ids and point_keys from table."""
    if "edges" in table.schema.names:
        edges_col = table.column("edges").to_pylist()
        return (
            _derive_edge_line_ids_list(edges_col),
            _derive_edge_point_keys_list(edges_col),
        )
    if "entity_keys" in table.schema.names and relations:
        ek_col = table.column("entity_keys").to_pylist()
        return (
            _derive_line_ids_from_entity_keys_list(ek_col, relations),
            _derive_point_keys_from_entity_keys_list(ek_col, relations),
        )
    empty: list[list[str]] = [[] for _ in range(table.num_rows)]
    return empty, empty


def _reconstruct_edges_from_row(
    row: dict, relations: list,
) -> list[Edge]:
    """Reconstruct Edge objects from a row dict using edges or entity_keys."""
    if row.get("edges"):
        return [
            Edge(
                line_id=e["line_id"],
                point_key=e["point_key"],
                status=e["status"],
                direction=e["direction"],
                is_jumpable=bool(e["point_key"]),
            )
            for e in row["edges"]
        ]
    # Fallback: reconstruct from entity_keys + relations
    from hypertopos.engine.geometry import _reconstruct_edges_from_entity_keys
    return _reconstruct_edges_from_entity_keys(row.get("entity_keys"), relations)


def _classify_calibration_health(anomaly_rate: float, total_entities: int) -> str:
    """Classify anomaly_rate into calibration health label.

    Returns "good", "suspect", or "poor" based on the following thresholds:
    - Empty pattern (total_entities == 0) → "good"
    - anomaly_rate < 0.001 or anomaly_rate > 0.30  → "poor"
    - anomaly_rate < 0.01  or anomaly_rate > 0.20  → "suspect"
    - Otherwise (1%–20% inclusive)                 → "good"
    """
    if total_entities == 0:
        return "good"
    if anomaly_rate < 0.001 or anomaly_rate > 0.30:
        return "poor"
    if anomaly_rate < 0.01 or anomaly_rate > 0.20:
        return "suspect"
    return "good"


def _classify_trajectory(delta_norms: list[float]) -> str:
    """Classify temporal trajectory shape from a sequence of delta norms.

    Returns one of: "arch", "v_shape", "spike_recovery", "linear_drift",
    "flat", "insufficient_data", or "other".
    """
    if len(delta_norms) < 3:
        return "insufficient_data"
    n = len(delta_norms)
    arr = np.array(delta_norms, dtype=np.float32)
    span = float(arr.max() - arr.min())
    if span < 1e-4:
        return "flat"
    norm = (arr - arr.min()) / span
    peak_idx = int(np.argmax(norm))
    trough_idx = int(np.argmin(norm))
    if 0.2 < peak_idx / n < 0.8 and norm[0] < 0.6 and norm[-1] < 0.6:
        return "arch"
    if 0.2 < trough_idx / n < 0.8 and norm[0] > 0.4 and norm[-1] > 0.4:
        return "v_shape"
    if peak_idx < n * 0.3 and norm[0] < 0.5 and norm[-1] < 0.4:
        return "spike_recovery"
    diffs = np.diff(arr)
    if np.sum(diffs > 0) > 0.7 * len(diffs):
        return "linear_drift"
    return "other"


_SIGMA_SAFE_FLOOR = 1e-12


def _compute_calibration_drift(
    fit_from: CalibrationFit,
    fit_to: CalibrationFit,
    top_n: int,
    verbose: bool,
) -> CalibrationDriftReport:
    """Pure math: compute drift between two CalibrationFit instances.

    Caller is responsible for verifying schema_hash agreement before calling.
    """
    mu_a = fit_from.mu.astype(np.float64)
    mu_b = fit_to.mu.astype(np.float64)
    sigma_a = fit_from.sigma_diag.astype(np.float64)
    sigma_b = fit_to.sigma_diag.astype(np.float64)
    theta_a = fit_from.theta.astype(np.float64)
    theta_b = fit_to.theta.astype(np.float64)

    D = mu_a.shape[0]

    mu_delta = mu_b - mu_a
    sigma_safe = np.where(sigma_a > _SIGMA_SAFE_FLOOR, sigma_a, 1.0)
    mu_delta_normalized = mu_delta / sigma_safe

    sigma_delta = sigma_b - sigma_a
    theta_delta = theta_b - theta_a

    overall_drift_rms = float(np.linalg.norm(mu_delta_normalized) / np.sqrt(D))

    kinds_a = fit_from.dimension_kinds
    per_dim: list[DimensionDrift] = []
    for i in range(D):
        per_dim.append(
            DimensionDrift(
                dim_index=i,
                dim_kind=kinds_a[i] if kinds_a is not None else None,
                mu_from=float(mu_a[i]),
                mu_to=float(mu_b[i]),
                mu_delta=float(mu_delta[i]),
                mu_delta_normalized=float(mu_delta_normalized[i]),
                sigma_from=float(sigma_a[i]),
                sigma_to=float(sigma_b[i]),
                sigma_delta=float(sigma_delta[i]),
                theta_from=float(theta_a[i]),
                theta_to=float(theta_b[i]),
                theta_delta=float(theta_delta[i]),
            )
        )

    ranked = sorted(per_dim, key=lambda d: abs(d.mu_delta_normalized), reverse=True)
    top_drifted = ranked[: min(top_n, D)]

    # Edge-dim threshold drift — surface per-source-dim p95 threshold change
    # across epochs when the anchor pattern declared `edge_dim_aggregations:`.
    # Both fits' thresholds are populated by the builder at calibration write
    # time; missing on either side means that epoch did not declare aggregations.
    threshold_drift: dict[str, dict[str, float]] | None = None
    thr_from = fit_from.edge_dim_thresholds
    thr_to = fit_to.edge_dim_thresholds
    if thr_from is not None and thr_to is not None:
        all_dims = sorted(set(thr_from) | set(thr_to))
        threshold_drift = {}
        for d in all_dims:
            tf = thr_from.get(d)
            tt = thr_to.get(d)
            if tf is None or tt is None:
                continue
            threshold_drift[d] = {
                "from": float(tf),
                "to": float(tt),
                "delta": float(tt - tf),
            }
        if not threshold_drift:
            threshold_drift = None

    return CalibrationDriftReport(
        pattern_id=fit_from.pattern_id,
        v_from=fit_from.calibration_epoch,
        v_to=fit_to.calibration_epoch,
        schema_hash=fit_from.schema_hash,
        population_size_from=fit_from.population_size,
        population_size_to=fit_to.population_size,
        overall_drift_rms=overall_drift_rms,
        top_drifted=top_drifted,
        per_dimension=per_dim if verbose else None,
        edge_dim_threshold_drift=threshold_drift,
    )


class GDSNavigator:
    def __init__(
        self,
        engine: GDSEngine,
        storage: GDSReader,
        manifest: Manifest,
        contract: Contract,
    ) -> None:
        self._engine = engine
        self._storage = storage
        self._manifest = manifest
        self._contract = contract
        self._position: Point | Polygon | Solid | None = None
        self._last_total_pre_geometry_filter: int | None = None
        self._dead_dim_cache: dict[tuple[str, int], list[int]] = {}
        self._cross_pattern_map: dict[str, dict[str, str]] = {}
        self._chain_reverse_index: dict[tuple[str, int], dict[str, list[str]]] = {}

    @property
    def position(self) -> Point | Polygon | Solid | None:
        return self._position

    def goto(self, primary_key: str, line_id: str) -> GDSNavigator:
        version = self._manifest.line_version(line_id) or 1
        table = self._storage.read_points(line_id, version)
        mask = pc.equal(table["primary_key"], primary_key)
        rows = table.filter(mask)
        if rows.num_rows == 0:
            raise GDSEntityNotFoundError(f"Point {primary_key} not found in {line_id}")
        row = {col: rows[col][0].as_py() for col in rows.schema.names}
        self._position = Point(
            primary_key=row["primary_key"],
            line_id=line_id,
            version=row["version"],
            status=row["status"],
            properties={k: row[k] for k in row if k not in
                        {"primary_key", "version", "status", "created_at", "changed_at"}},
            created_at=row["created_at"],
            changed_at=row["changed_at"],
        )
        return self

    def search_entities_fts(
        self, line_id: str, query: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Full-text search across all string properties of a line.

        Returns up to *limit* entities whose string columns contain *query*,
        ranked by BM25 relevance (best match first). Each result is a dict with
        'primary_key', 'status', and 'properties' keys — same format as the MCP
        search_entities tool.

        Requires INVERTED indices to be present on the Lance points dataset.
        GDSBuilder builds these automatically at write time. Returns an empty list
        when no match is found.
        """
        version = self._manifest.line_version(line_id) or 1
        table = self._storage.search_points_fts(line_id, version, query, limit=limit)
        if table.num_rows == 0:
            return []
        # Drop _score column before converting to dicts
        if "_score" in table.schema.names:
            table = table.drop("_score")
        results: list[dict[str, Any]] = []
        for row in table.to_pylist():
            results.append({
                "primary_key": row["primary_key"],
                "status": row.get("status", "unknown"),
                "properties": {
                    k: v
                    for k, v in row.items()
                    if k not in {"primary_key", "status"} and v is not None
                },
            })
        return results

    def _search_fts_scored(
        self, line_id: str, query: str, limit: int = 20
    ) -> list[tuple[str, float]]:
        """Return (primary_key, bm25_score) pairs for FTS query.

        Internal helper for search_hybrid — preserves BM25 scores before they are
        dropped. Higher score = better match. Returns empty list when no matches.
        """
        version = self._manifest.line_version(line_id) or 1
        table = self._storage.search_points_fts(line_id, version, query, limit=limit)
        if table.num_rows == 0:
            return []
        if "_score" not in table.schema.names:
            raise ValueError(
                f"FTS result for line '{line_id}' missing '_score' column — unexpected schema"
            )
        keys = table["primary_key"].to_pylist()
        raw_scores = table["_score"].to_pylist()
        if any(s is None for s in raw_scores):
            raise ValueError(
                f"FTS result for line '{line_id}' contains null _score values — "
                "Lance FTS index may be missing or corrupt"
            )
        scores: list[float] = [float(s) for s in raw_scores]
        return list(zip(keys, scores, strict=False))

    def search_hybrid(
        self,
        primary_key: str,
        pattern_id: str,
        line_id: str,
        query: str,
        alpha: float = 0.7,
        top_n: int = 10,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """Fuse ANN (cosine) and BM25 scores into a single entity ranking.

        Candidates = union of ANN top-(top_n*2) and FTS top-(top_n*5) results.
        Scores normalised to [0, 1]:
          vector_score = 1 - dist/max_dist  (ANN cosine distance)
          text_score   = bm25 / max_bm25
        Fusion: final_score = alpha * vector_score + (1 - alpha) * text_score
        Returns dict with "results" (up to top_n, sorted by final_score desc),
        "ann_active" (True when ANN returned at least one candidate), and
        "fts_candidates" (number of unique FTS matches in the pool — useful to
        diagnose text_score=0.0: if fts_candidates=0, the query matched nothing).
        """
        ann_fetch_n = top_n * 2
        fts_fetch_n = top_n * 5  # wider pool — low-IDF terms rank many candidates equally

        # Gather ANN candidates: (key -> distance)
        # primary_key is already excluded by find_similar_entities
        ann_pairs = self.find_similar_entities(
            primary_key, pattern_id, top_n=ann_fetch_n, filter_expr=filter_expr
        )
        ann_dist: dict[str, float] = dict(ann_pairs)

        # Gather FTS candidates: (key -> bm25_score)
        fts_pairs = self._search_fts_scored(line_id, query, limit=fts_fetch_n)
        fts_score: dict[str, float] = dict(fts_pairs)

        # Union of both key sets, excluding the reference entity
        candidates = (set(ann_dist.keys()) | set(fts_score.keys())) - {primary_key}

        if not candidates:
            return {"results": [], "ann_active": bool(ann_dist)}

        # Normalise ANN distances to similarity [0, 1]
        max_dist = max(ann_dist.values()) if ann_dist else 1.0
        if max_dist == 0.0:
            max_dist = 1.0

        # Normalise BM25 scores to [0, 1]
        max_bm25 = max(fts_score.values()) if fts_score else 1.0
        if max_bm25 == 0.0:
            max_bm25 = 1.0

        results: list[dict[str, Any]] = []
        for key in candidates:
            dist = ann_dist.get(key)
            bm25 = fts_score.get(key)

            vector_score = round(1.0 - dist / max_dist, 4) if dist is not None else 0.0
            text_score = round(bm25 / max_bm25, 4) if bm25 is not None else 0.0
            final_score = round(alpha * vector_score + (1.0 - alpha) * text_score, 4)

            results.append({
                "primary_key": key,
                "vector_score": vector_score,
                "text_score": text_score,
                "final_score": final_score,
            })

        results.sort(key=lambda r: r["final_score"], reverse=True)
        return {
            "results": results[:top_n],
            "ann_active": bool(ann_dist),
            "fts_candidates": len(fts_score),
        }

    def π1_walk_line(
        self, line_id: str, direction: Literal["+", "-"] = "+"
    ) -> GDSNavigator:
        if not isinstance(self._position, Point):
            raise GDSPositionError("π1 requires position to be a Point")
        version = self._manifest.line_version(line_id) or 1
        table = self._storage.read_points(line_id, version)
        keys = table["primary_key"].to_pylist()
        try:
            idx = keys.index(self._position.primary_key)
        except ValueError as exc:
            raise GDSEntityNotFoundError(
                f"{self._position.primary_key} not in {line_id}"
            ) from exc
        next_idx = idx + (1 if direction == "+" else -1)
        if next_idx < 0 or next_idx >= len(keys):
            raise GDSNavigationError("No adjacent point in that direction")
        return self.goto(keys[next_idx], line_id)

    def π2_jump_polygon(
        self, polygon: Polygon, target_line_id: str, edge_index: int = 0
    ) -> GDSNavigator:
        alive_targets = [
            e for e in polygon.edges
            if e.line_id == target_line_id and e.is_alive()
        ]
        if not alive_targets:
            raise GDSNoAliveEdgeError(
                f"No alive edge to {target_line_id} in polygon {polygon.primary_key}"
            )
        if edge_index < 0 or edge_index >= len(alive_targets):
            raise GDSNavigationError(
                f"edge_index {edge_index} out of range — only {len(alive_targets)} "
                f"alive edge(s) to '{target_line_id}' in polygon {polygon.primary_key}"
            )
        target_edge = alive_targets[edge_index]
        if not target_edge.point_key:
            raise ValueError(
                f"Cannot jump to '{target_line_id}': edge uses continuous mode "
                f"(edge_max pattern) which stores edge counts, not entity keys. "
                f"Use get_centroid_map(group_by_property=...) or aggregate() instead."
            )
        return self.goto(target_edge.point_key, target_line_id)

    def π3_dive_solid(
        self,
        primary_key: str,
        pattern_id: str,
        timestamp: datetime | None = None,
    ) -> GDSNavigator:
        solid = self._engine.build_solid(
            primary_key, pattern_id, self._manifest, timestamp=timestamp
        )
        self._position = solid
        return self

    def π4_emerge(self) -> GDSNavigator:
        if self._position is None:
            raise GDSPositionError("π4 requires active position")
        if isinstance(self._position, (Polygon, Solid)):
            self._position = Point(
                primary_key=self._position.primary_key,
                line_id="emerged",
                version=0, status="active", properties={},
                created_at=datetime.now(), changed_at=datetime.now(),
            )
        return self

    def _resolve_version(self, pattern_id: str) -> int:
        version = self._manifest.pattern_version(pattern_id)
        if version is None:
            raise GDSNavigationError(
                f"No geometry version for pattern '{pattern_id}' in manifest."
            )
        return version

    def dead_dim_indices(self, pattern_id: str) -> list[int]:
        """Return dimension indices with near-zero variance (< 0.01) in the population.

        Samples up to 200 geometry rows. Cached per (pattern_id, version).
        """
        version = self._resolve_version(pattern_id)
        key = (pattern_id, version)
        if key in self._dead_dim_cache:
            return self._dead_dim_cache[key]
        geo = self._storage.read_geometry(pattern_id, version, columns=["delta"])
        deltas_raw = geo["delta"].to_pylist()
        if len(deltas_raw) > 200:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(deltas_raw), 200, replace=False)
            deltas_raw = [deltas_raw[int(i)] for i in idx]
        if not deltas_raw:
            self._dead_dim_cache[key] = []
            return []
        deltas = np.array(deltas_raw, dtype=np.float32)
        variances = np.var(deltas, axis=0)
        dead = [int(i) for i in range(len(variances)) if variances[i] < 0.01]
        self._dead_dim_cache[key] = dead
        return dead

    _LIGHT_COLUMNS = ["primary_key", "delta", "delta_norm", "is_anomaly", "delta_rank_pct"]
    _CONTRAST_COLUMNS: list[str] = [
        "primary_key",
        "delta",
        "is_anomaly",
        "edges",
        "entity_keys",
    ]
    _CENTROID_COLUMNS: list[str] = [
        "primary_key",
        "delta",
        "edges",
        "entity_keys",
    ]
    # Columns needed for constructing full Polygon objects from event geometry.
    # Reader silently drops columns missing from the Lance schema (e.g. "edges"
    # on event geometry), so it is safe to include both "edges" and "entity_keys".
    _POLYGON_COLUMNS: list[str] = [
        "primary_key", "scale", "delta", "delta_norm", "delta_rank_pct",
        "is_anomaly", "last_refresh_at", "updated_at",
        "edges", "entity_keys",
    ]
    _CLUSTER_COLUMNS: list[str] = [
        "primary_key", "delta", "delta_norm", "is_anomaly",
    ]
    _CENTROID_AUTO_SAMPLE: int = 100_000

    @staticmethod
    def _apply_fdr_select_polygons(
        polygons: list[Polygon],
        *,
        fdr_alpha: float | None,
        select: str,
        top_n: int,
        fdr_method: str = "bh",
        p_value_method: str = "rank",
        pattern_df: int | None = None,
    ) -> list[Polygon]:
        """Apply FDR filtering and diverse selection to a list of Polygons.

        Mutates Polygon objects in-place (sets q_value / representativeness)
        and returns the filtered/reordered list.

        p_value_method: "rank" (default, uniform-by-construction) or "chi2"
          (upper-tail χ²(df) on ||delta||²). Rank-based p-values carry no null
          vs alternative signal for the Storey estimator; "chi2" is the option
          that makes fdr_method="storey" actually shrink q-values. Requires
          pattern_df to be supplied by the caller.
        """
        if p_value_method not in ("rank", "chi2"):
            raise ValueError(
                f"p_value_method must be 'rank' or 'chi2', got {p_value_method!r}"
            )
        if p_value_method == "chi2" and pattern_df is None:
            raise ValueError("p_value_method='chi2' requires pattern_df")

        # --- FDR filtering (opt-in) ---
        if fdr_alpha is not None and len(polygons) > 0:
            from hypertopos.engine.fdr import (
                benjamini_hochberg,
                empirical_p_values_from_rank,
                parametric_p_values_chi2,
            )
            if p_value_method == "rank":
                rank_pct = np.array(
                    [p.delta_rank_pct if p.delta_rank_pct is not None else 0.0
                     for p in polygons],
                    dtype=np.float64,
                )
                p_values = empirical_p_values_from_rank(rank_pct)
            else:  # chi2
                delta_norms = np.array(
                    [float(p.delta_norm) for p in polygons],
                    dtype=np.float64,
                )
                p_values = parametric_p_values_chi2(delta_norms, df=int(pattern_df))
            rejected, q_values = benjamini_hochberg(p_values, fdr_alpha, method=fdr_method)
            for poly, q in zip(polygons, q_values):
                poly.q_value = float(q)  # type: ignore[attr-defined]
            polygons = [p for p, keep in zip(polygons, rejected) if keep]

        # --- Diverse selection (opt-in) ---
        if select not in ("top_norm", "diverse"):
            raise ValueError(f"unknown select mode: {select!r}")
        if select == "diverse" and len(polygons) > 0:
            from hypertopos.engine.selection import lazy_greedy_facility_location
            delta_vectors = np.array(
                [p.delta for p in polygons], dtype=np.float64,
            )
            k = min(top_n, len(polygons))
            selected_idx, representativeness = lazy_greedy_facility_location(
                delta_vectors, k,
            )
            out: list[Polygon] = []
            for i, idx in enumerate(selected_idx):
                poly = polygons[int(idx)]
                poly.representativeness = int(representativeness[i])  # type: ignore[attr-defined]
                out.append(poly)
            polygons = out

        return polygons

    def π5_attract_anomaly(
        self,
        pattern_id: str,
        radius: float | None = None,
        top_n: int = 10,
        offset: int = 0,
        missing_edge_to: str | None = None,
        include_emerging: bool = False,
        rank_by_property: str | None = None,
        property_filters: dict | None = None,
        fdr_alpha: float | None = None,
        fdr_method: str = "bh",
        p_value_method: str = "rank",
        select: str = "top_norm",
        metric: str = "L2",
        min_confidence: float = 0.0,
    ) -> tuple[list[Polygon], int, list[dict] | None, dict | None]:
        """Find the most anomalous polygons in a pattern.

        Returns ``(polygons, total_found, emerging, meta)`` where *emerging*
        is a list of not-yet-anomalous entities trending toward theta (only
        when *include_emerging=True* and *offset == 0*), and *meta* is a dict
        with ``total_anomalies_unfiltered`` when *property_filters* is set,
        or ``None`` otherwise.
        """
        if fdr_method not in ("bh", "storey"):
            raise ValueError(
                f"fdr_method must be 'bh' or 'storey', got {fdr_method!r}"
            )
        if metric not in ("L2", "Linf"):
            raise ValueError(f"metric must be 'L2' or 'Linf', got '{metric}'")
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]

        if missing_edge_to:
            if pattern.pattern_type == "event":
                raise ValueError(
                    "missing_edge_to is not supported for event patterns — "
                    "use missing_edge_to at the aggregate level instead"
                )
            if missing_edge_to not in sphere.lines:
                raise ValueError(
                    f"Unknown line '{missing_edge_to}' in missing_edge_to. "
                    f"Available: {sorted(sphere.lines)}"
                )

        theta_norm = float(np.linalg.norm(np.array(pattern.theta, dtype=np.float32)))
        threshold = theta_norm if radius is None else radius * theta_norm

        # ------------------------------------------------------------------
        # Subprocess fast path for large geometry tables (>500K rows).
        # Offloads the delta_norm filter + top-N sort to the persistent
        # worker process, avoiding a full in-process Arrow scan.
        # ------------------------------------------------------------------
        try:
            _geo_count = (
                int(self._storage.count_geometry_rows(pattern_id, version))
                if not missing_edge_to and not property_filters else 0
            )
        except _NAVIGATION_RECOVERABLE_ERRORS:
            _geo_count = 0
        if _geo_count > 500_000 and rank_by_property is None and metric == "L2":
            from hypertopos.engine.lance_sql_agg import (
                find_anomalies as _lance_sql_find_anomalies,
            )
            geo_lance_path = str(
                self._storage._base.resolve()
                / "geometry" / pattern_id / f"v={version}" / "data.lance"
            )
            resp = _lance_sql_find_anomalies(
                geo_lance_path,
                threshold=float(threshold),
                top_n=offset + top_n,
                offset=0,
                min_confidence=min_confidence,
            )
            total_found = resp["total_found"]
            # Build norm lookup from full response first (handles duplicates)
            _all_norm_lookup = dict(zip(resp["keys"], resp["delta_norms"]))
            sub_keys = list(dict.fromkeys(resp["keys"][offset:]))
            if sub_keys:
                light_cols = [
                    "primary_key", "scale", "delta", "delta_norm",
                    "delta_rank_pct", "is_anomaly",
                    "last_refresh_at", "updated_at",
                    "bregman_divergence", "anomaly_confidence",
                    "edges", "entity_keys",
                ]
                full_geo = self._storage.read_geometry(
                    pattern_id, version, point_keys=sub_keys,
                    columns=light_cols,
                )
                norm_lookup = {k: _all_norm_lookup[k] for k in sub_keys if k in _all_norm_lookup}
                results = self._engine.geometry_to_polygons(
                    full_geo, norm_lookup=norm_lookup, top_n=top_n,
                    pattern=pattern,
                    pattern_id=pattern_id,
                    pattern_type=pattern.pattern_type,
                    pattern_ver=version,
                )
            else:
                results = []
            results = self._apply_fdr_select_polygons(
                results, fdr_alpha=fdr_alpha, select=select, top_n=top_n,
                fdr_method=fdr_method, p_value_method=p_value_method,
                pattern_df=len(pattern.theta) if p_value_method == "chi2" else None,
            )
            emerging = self._find_emerging(
                pattern_id, version, pattern, include_emerging, offset, top_n,
            )
            return results, total_found, emerging, None

        # ------------------------------------------------------------------
        # In-process full-scan path
        # ------------------------------------------------------------------

        # Pass 1: light scan — push delta_norm >= threshold to Lance scanner
        # Linf metric: can't use pre-computed delta_norm, read all geometry
        cols = list(self._LIGHT_COLUMNS)
        if missing_edge_to:
            cols.append("edges")
        if min_confidence > 0.0:
            cols.append("anomaly_confidence")
        if metric == "bregman":
            cols.append("bregman_divergence")
        if metric in ("Linf", "bregman"):
            # Linf/bregman: can't use pre-computed delta_norm filter, read all
            light = self._storage.read_geometry(
                pattern_id, version, columns=cols,
            )
        else:
            light = self._storage.read_geometry(
                pattern_id, version, columns=cols,
                filter=f"delta_norm >= {threshold}",
            )
        if light.num_rows == 0:
            emerging = self._find_emerging(
                pattern_id, version, pattern, include_emerging, offset, top_n,
            )
            return [], 0, emerging, None

        # Post-filter: keep only entities WITHOUT an edge to the target line
        if missing_edge_to:
            eli = _derive_edge_line_ids_from_table(light)
            mask = [missing_edge_to not in (row or []) for row in eli]
            light = light.filter(pa.array(mask))
            if light.num_rows == 0:
                emerging = self._find_emerging(
                    pattern_id, version, pattern, include_emerging, offset, top_n,
                )
                return [], 0, emerging, None

        # Vectorized recompute + rank
        delta_matrix = delta_matrix_from_arrow(light)
        if metric == "Linf":
            norms = np.max(np.abs(delta_matrix), axis=1).astype(np.float32)
        elif metric == "bregman" and "bregman_divergence" in light.schema.names:
            norms = light["bregman_divergence"].to_numpy(zero_copy_only=False).astype(np.float32)
        else:
            norms = np.sqrt(np.einsum('ij,ij->i', delta_matrix, delta_matrix)).astype(np.float32)
        valid = norms >= threshold
        if not np.any(valid):
            emerging = self._find_emerging(
                pattern_id, version, pattern, include_emerging, offset, top_n,
            )
            return [], 0, emerging, None
        valid_idx = np.where(valid)[0]
        valid_norms = norms[valid_idx]
        total_found = len(valid_norms)

        # Capture unfiltered count BEFORE any confidence / property filtering
        total_anomalies_unfiltered = total_found

        # min_confidence filter — secondary filter on anomaly_confidence column
        if min_confidence > 0.0 and pattern.dimension_kinds is None:
            logger.warning(
                "min_confidence filter ignored — pattern '%s' lacks "
                "dimension_kinds (rebuild sphere with Bregman calibration)",
                pattern_id,
            )
            min_confidence = 0.0
        if min_confidence > 0.0 and "anomaly_confidence" in light.schema.names:
            conf_values = light["anomaly_confidence"].to_numpy(zero_copy_only=False).astype(np.float32)
            conf_mask = conf_values[valid_idx] >= min_confidence
            valid_idx = valid_idx[conf_mask]
            valid_norms = norms[valid_idx]
            total_found = len(valid_norms)

        # Property filters — narrow anomalous set by entity line properties
        _pre_loaded_pts = None
        if property_filters:
            all_keys = [str(light["primary_key"][int(i)].as_py()) for i in valid_idx]
            entity_line_id = pattern.entity_line_id
            line_ver = self._manifest.line_version(entity_line_id) if entity_line_id else None
            if not (entity_line_id and line_ver is not None):
                raise GDSNavigationError(
                    f"property_filters requires an entity line — "
                    f"event patterns don't have entity properties"
                )
            read_cols = list(property_filters.keys())
            if rank_by_property and rank_by_property not in read_cols:
                read_cols.append(rank_by_property)
            pts = self._storage.read_points_batch(
                entity_line_id, line_ver, all_keys,
                columns=read_cols,
            )
            from hypertopos.engine.aggregation import _apply_event_filters
            for col in property_filters:
                if col not in pts.column_names:
                    raise GDSNavigationError(
                        f"property_filters column '{col}' not found on entity line "
                        f"'{entity_line_id}'. Available: {sorted(c for c in pts.column_names if c != 'primary_key')}"
                    )
            pts = _apply_event_filters(pts, property_filters)
            surviving = set(pts["primary_key"].to_pylist())
            keep_mask = np.array([all_keys[j] in surviving for j in range(len(all_keys))])
            valid_idx = valid_idx[keep_mask]
            valid_norms = norms[valid_idx]
            total_found = len(valid_norms)
            if rank_by_property:
                _pre_loaded_pts = pts

        if rank_by_property is not None:
            # Optimised path: sort by property BEFORE building Polygon objects.
            # 1. Read only primary_key + property for ALL anomalous entities (lightweight)
            # 2. Sort by property, take top_n keys
            # 3. Build Polygon objects only for those top_n keys
            all_keys = [str(light["primary_key"][int(i)].as_py()) for i in valid_idx]
            entity_line_id = pattern.entity_line_id
            line_ver = self._manifest.line_version(entity_line_id) if entity_line_id else None
            if not (entity_line_id and line_ver is not None):
                raise GDSNavigationError(
                    f"rank_by_property is not supported on pattern '{pattern_id}' — "
                    f"no entity line found (event patterns don't have entity properties)"
                )
            pts = _pre_loaded_pts if _pre_loaded_pts is not None else self._storage.read_points_batch(
                entity_line_id, line_ver, all_keys,
                columns=["primary_key", rank_by_property],
            )
            if rank_by_property not in pts.column_names:
                raise GDSNavigationError(
                    f"rank_by_property='{rank_by_property}' not found in "
                    f"entity line '{entity_line_id}'. Available columns visible "
                    f"via get_sphere_info."
                )
            prop_pairs: list[tuple[str, float]] = []
            for i in range(pts.num_rows):
                pk = pts["primary_key"][i].as_py()
                val = pts[rank_by_property][i].as_py()
                try:
                    prop_pairs.append((pk, float(val) if val is not None else float("-inf")))
                except (TypeError, ValueError):
                    prop_pairs.append((pk, float("-inf")))
            prop_pairs.sort(key=lambda x: x[1], reverse=True)
            top_keys = [pk for pk, _ in prop_pairs[offset:offset + top_n]]
            # Build norm lookup for these keys
            key_to_norm = {str(light["primary_key"][int(i)].as_py()): float(norms[i]) for i in valid_idx}
        else:
            if len(valid_norms) == 0:
                emerging = self._find_emerging(
                    pattern_id, version, pattern, include_emerging, offset, top_n,
                )
                meta = {"total_anomalies_unfiltered": total_anomalies_unfiltered} if property_filters else None
                return [], total_found, emerging, meta
            # Full sort for deterministic pagination (population is pre-filtered
            # to anomalies above threshold, typically 5-20% of total)
            sorted_local = np.argsort(valid_norms)[::-1]
            n = min(offset + top_n, len(sorted_local))
            # Include all entities tied with the boundary norm to prevent
            # pagination duplicates when tied entities shift positions
            if n < len(sorted_local):
                boundary_norm = valid_norms[sorted_local[n - 1]]
                while n < len(sorted_local) and valid_norms[sorted_local[n]] == boundary_norm:
                    n += 1
            top_idx = valid_idx[sorted_local[:n]]
            top_keys = [str(light["primary_key"][int(i)].as_py()) for i in top_idx]
            key_to_norm = {str(light["primary_key"][int(i)].as_py()): float(norms[i]) for i in top_idx}

        # Pass 2: light geometry read for selected keys only
        escaped = [k.replace("'", "''") for k in top_keys]
        pk_in = ", ".join(f"'{k}'" for k in escaped)
        _anomaly_light_cols = [
            "primary_key", "scale", "delta", "delta_norm",
            "delta_rank_pct", "is_anomaly",
            "last_refresh_at", "updated_at",
            "bregman_divergence", "anomaly_confidence",
            "edges", "entity_keys",
        ]
        full = self._storage.read_geometry(
            pattern_id, version, filter=f"primary_key IN ({pk_in})",
            columns=_anomaly_light_cols,
        )
        from hypertopos.engine.geometry import _reconstruct_edges_from_entity_keys

        results: list[Polygon] = []
        for i in range(full.num_rows):
            row = {col: full[col][i].as_py() for col in full.schema.names}
            pk = row["primary_key"]
            recomputed_norm = key_to_norm.get(
                pk, float(np.linalg.norm(np.array(row["delta"], dtype=np.float32)))
            )
            # Decode edges from struct column or reconstruct from entity_keys
            if row.get("edges"):
                edges = [
                    Edge(
                        line_id=e["line_id"],
                        point_key=e["point_key"],
                        status=e["status"],
                        direction=e["direction"],
                        is_jumpable=bool(e["point_key"]),
                    )
                    for e in row["edges"]
                ]
            elif row.get("entity_keys") and pattern.relations:
                edges = _reconstruct_edges_from_entity_keys(
                    row["entity_keys"], pattern.relations,
                )
            else:
                edges = []
            results.append(Polygon(
                primary_key=pk,
                pattern_id=row.get("pattern_id", pattern_id),
                pattern_ver=row.get("pattern_ver", version),
                pattern_type=row.get("pattern_type", pattern.pattern_type),
                scale=row["scale"],
                delta=np.array(row["delta"], dtype=np.float32),
                delta_norm=recomputed_norm,
                is_anomaly=bool(row["is_anomaly"]),
                edges=edges,
                last_refresh_at=row["last_refresh_at"],
                updated_at=row["updated_at"],
                delta_rank_pct=(
                    None if row.get("delta_rank_pct") is None
                    else float(row["delta_rank_pct"])
                ),
                bregman_divergence=(
                    None if row.get("bregman_divergence") is None
                    else float(row["bregman_divergence"])
                ),
                anomaly_confidence=(
                    None if row.get("anomaly_confidence") is None
                    else float(row["anomaly_confidence"])
                ),
            ))
        if rank_by_property is not None:
            # Preserve property-based order (Pass 2 may scramble it)
            key_order = {k: i for i, k in enumerate(top_keys)}
            results.sort(key=lambda p: key_order.get(p.primary_key, 999999))
        else:
            results.sort(key=lambda p: p.delta_norm, reverse=True)

        # Deduplicate on primary_key — Lance may have duplicate rows from
        # interrupted incremental_update. Keep highest delta_norm per key.
        seen: dict[str, Polygon] = {}
        for p in results:
            if p.primary_key not in seen or p.delta_norm > seen[p.primary_key].delta_norm:
                seen[p.primary_key] = p
        # Sort by (-delta_norm, primary_key) for deterministic pagination on tied norms
        results = sorted(seen.values(), key=lambda p: (-p.delta_norm, p.primary_key))
        results = results[offset:offset + top_n]
        results = self._apply_fdr_select_polygons(
            results, fdr_alpha=fdr_alpha, select=select, top_n=top_n,
            fdr_method=fdr_method, p_value_method=p_value_method,
            pattern_df=len(pattern.theta) if p_value_method == "chi2" else None,
        )

        emerging = self._find_emerging(
            pattern_id, version, pattern, include_emerging, offset, top_n,
        )
        meta = {"total_anomalies_unfiltered": total_anomalies_unfiltered} if property_filters else None
        return results, total_found, emerging, meta

    def _find_emerging(
        self,
        pattern_id: str,
        version: int,
        pattern: Any,
        include_emerging: bool,
        offset: int,
        top_n: int,
    ) -> list[dict] | None:
        """Scan non-anomalous entities for emerging anomaly trajectories.

        Returns a sorted list of dicts or None when *include_emerging* is
        False or *offset > 0*.
        """
        if not include_emerging or offset != 0:
            return None

        from hypertopos.engine.forecast import forecast_anomaly_status

        non_anom_geo = self._storage.read_geometry(
            pattern_id, version,
            filter="is_anomaly = false",
            columns=["primary_key"],
        )
        candidate_keys = non_anom_geo["primary_key"].to_pylist()[:100]

        emerging: list[dict] = []
        for pk in candidate_keys:
            try:
                solid = self._engine.build_solid(pk, pattern_id, self._manifest)
                if len(solid.slices) >= 3:
                    deltas = [s.delta_snapshot for s in solid.slices]
                    base_delta_norm = float(solid.base_polygon.delta_norm)
                    af = forecast_anomaly_status(
                        deltas, pattern.theta_norm, horizon=1,
                        current_delta_norm=base_delta_norm,
                    )
                    if af.forecast_is_anomaly and not af.current_is_anomaly:
                        emerging.append({
                            "primary_key": pk,
                            "current_delta_norm": round(base_delta_norm, 4),
                            "predicted_delta_norm": round(
                                af.predicted_delta_norm, 4,
                            ),
                            "reliability": af.reliability,
                        })
            except _NAVIGATION_RECOVERABLE_ERRORS:
                pass

        emerging.sort(key=lambda e: e["predicted_delta_norm"], reverse=True)
        return emerging[:top_n]

    def anomaly_summary(self, pattern_id: str, max_clusters: int = 20) -> dict[str, Any]:
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        theta_norm = float(np.linalg.norm(pattern.theta))

        bucket_labels = ["0–0.25θ", "0.25θ–0.5θ", "0.5θ–0.75θ", "0.75θ–θ", "θ–1.5θ", "1.5θ+"]

        # Try geometry stats cache — avoids loading the large delta column for
        # percentile and count fields (O(1) vs O(n) full scan on 1M datasets).
        stats_cache = self._storage.read_geometry_stats(pattern_id, version)

        if stats_cache is not None:
            total = stats_cache["total_entities"]
            cached_pcts = stats_cache["percentiles"]
            percentiles = {
                "p50":  round(float(cached_pcts["p50"]),  4),
                "p75":  round(float(cached_pcts["p75"]),  4),
                "p90":  round(float(cached_pcts["p90"]),  4),
                "p95":  round(float(cached_pcts["p95"]),  4),
                "p99":  round(float(cached_pcts["p99"]),  4),
                "max":  round(float(cached_pcts["max"]),  4),
            }

            if total == 0:
                return {
                    "pattern_id": pattern_id,
                    "total_entities": 0,
                    "total_anomalies": 0,
                    "anomaly_rate": 0.0,
                    "theta_norm": round(theta_norm, 4),
                    "clusters": [],
                    "delta_norm_percentiles": dict.fromkeys(
                        ["p50", "p75", "p90", "p95", "p99", "max"], 0.0
                    ),
                    "delta_norm_distribution": dict.fromkeys(bucket_labels, 0),
                }

            # Distribution scan: skip large delta column — only delta_norm needed.
            dist_table = self._storage.read_geometry(
                pattern_id, version,
                columns=["primary_key", "delta_norm", "is_anomaly"],
            )
            norms_arr = dist_table["delta_norm"].to_numpy().astype(np.float64)

            # Recount from stored delta_norm — matches cache-miss path and
            # find_anomalies semantics.
            anomaly_count = int((norms_arr >= theta_norm).sum()) if theta_norm > 0.0 else 0

            # Theta-relative distribution from the dist_table norms
            if theta_norm > 0:
                bucket_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]) * theta_norm
                bin_indices = np.clip(np.digitize(norms_arr, bucket_edges) - 1, 0, 5)
                counts = np.bincount(bin_indices.astype(np.intp), minlength=6).tolist()
                distribution = dict(zip(bucket_labels, [int(c) for c in counts], strict=False))
            else:
                distribution = dict.fromkeys(bucket_labels, 0)
                distribution["0–0.25θ"] = total

            # Load delta only for anomalous rows (typically <5% of total) for cluster breakdown.
            anomaly_table = self._storage.read_geometry(
                pattern_id, version,
                columns=["primary_key", "delta"],
                filter=f"delta_norm >= {theta_norm}",
            )
        else:
            # Cache miss: full scan including delta column (backwards compatibility)
            table = self._storage.read_geometry(
                pattern_id, version,
                columns=["primary_key", "delta", "delta_norm"],
            )
            total = table.num_rows

            if total == 0:
                return {
                    "pattern_id": pattern_id,
                    "total_entities": 0,
                    "total_anomalies": 0,
                    "anomaly_rate": 0.0,
                    "theta_norm": round(theta_norm, 4),
                    "clusters": [],
                    "delta_norm_percentiles": dict.fromkeys(
                        ["p50", "p75", "p90", "p95", "p99", "max"], 0.0
                    ),
                    "delta_norm_distribution": dict.fromkeys(bucket_labels, 0),
                }

            norms_arr = table["delta_norm"].to_numpy().astype(np.float64)
            anomaly_count = int((norms_arr >= theta_norm).sum()) if theta_norm > 0.0 else 0
            percentiles = {
                "p50": round(float(np.percentile(norms_arr, 50)), 4),
                "p75": round(float(np.percentile(norms_arr, 75)), 4),
                "p90": round(float(np.percentile(norms_arr, 90)), 4),
                "p95": round(float(np.percentile(norms_arr, 95)), 4),
                "p99": round(float(np.percentile(norms_arr, 99)), 4),
                "max": round(float(norms_arr.max()), 4),
            }

            # Theta-relative adaptive distribution
            if theta_norm > 0:
                bucket_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]) * theta_norm
                bin_indices = np.clip(np.digitize(norms_arr, bucket_edges) - 1, 0, 5)
                counts = np.bincount(bin_indices.astype(np.intp), minlength=6).tolist()
                distribution = dict(zip(bucket_labels, [int(c) for c in counts], strict=False))
            else:
                distribution = dict.fromkeys(bucket_labels, 0)
                distribution["0–0.25θ"] = total

            if theta_norm > 0.0:
                anomaly_mask = pc.greater_equal(table["delta_norm"], theta_norm)
                anomaly_table = table.filter(anomaly_mask)
            else:
                anomaly_table = table.slice(0, 0)  # empty — no anomalies when theta=0

        # Build cluster breakdown from anomalous rows (uses delta column)
        clusters: dict[tuple, list[str]] = {}
        if anomaly_table.num_rows > 0:
            anom_deltas = delta_matrix_from_arrow(anomaly_table)
            anom_rounded = np.round(anom_deltas, 1)
            anom_pks = anomaly_table["primary_key"].to_pylist()
            for i in range(anomaly_table.num_rows):
                key = tuple(anom_rounded[i].tolist())
                clusters.setdefault(key, []).append(str(anom_pks[i]))
        cluster_list = []
        for delta_key, pks in sorted(clusters.items(), key=lambda x: -len(x[1])):
            k = len(pattern.relations)
            missing_edges = [
                pattern.relations[j].line_id
                for j, v in enumerate(delta_key)
                if j < k and v < 0
            ]
            missing_props = [
                pattern.prop_columns[j - k]
                for j, v in enumerate(delta_key)
                if j >= k and j < k + len(pattern.prop_columns) and v < 0
            ]
            all_missing = missing_edges + [f"prop:{p}" for p in missing_props]
            label = f"missing: {', '.join(all_missing)}" if all_missing else "excess edges"
            cluster_list.append({
                "delta": list(delta_key),
                "label": label,
                "count": len(pks),
                "examples": pks[:3],
            })

        total_clusters = len(cluster_list)
        truncated = max_clusters > 0 and total_clusters > max_clusters
        if truncated:
            cluster_list = cluster_list[:max_clusters]

        # Compute top_driving_dimensions from cluster data
        labels = pattern.dim_labels
        dim_sq_totals = np.zeros(len(labels), dtype=np.float64)
        total_weight = 0.0
        for cluster in cluster_list:
            delta = np.array(cluster["delta"], dtype=np.float32)
            sq = delta ** 2
            count = cluster["count"]
            delta_norm_c = float(np.linalg.norm(delta))
            weight = delta_norm_c * count
            dim_sq_totals[:len(sq)] += sq * weight
            total_weight += weight
        if total_weight > 1e-10:
            pcts = dim_sq_totals / dim_sq_totals.sum() * 100
            top_idx = np.argsort(pcts)[::-1]
            top_driving_dimensions = [
                {
                    "dim": int(i),
                    "label": labels[i],
                    "mean_contribution_pct": round(float(pcts[i]), 1),
                }
                for i in top_idx if pcts[i] > 3.0
            ]
        else:
            top_driving_dimensions = []

        return {
            "pattern_id": pattern_id,
            "total_entities": total,
            "total_anomalies": anomaly_count,
            "anomaly_rate": round(anomaly_count / total, 4),
            "theta_norm": round(theta_norm, 4),
            "clusters": cluster_list,
            "total_clusters": total_clusters,
            "clusters_truncated": truncated,
            "delta_norm_percentiles": percentiles,
            "delta_norm_distribution": distribution,
            "top_driving_dimensions": top_driving_dimensions,
        }

    def aggregate_anomalies(
        self,
        pattern_id: str,
        group_by: str,
        top_n: int = 50,
        sample_size: int | None = None,
        include_keys: bool = False,
        keys_per_group: int = 5,
        property_filters: dict | None = None,
    ) -> dict[str, Any]:
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type == "event":
            raise ValueError(
                "aggregate_anomalies is for anchor/composite patterns. "
                "For event patterns, use aggregate(geometry_filters={\"is_anomaly\": true})."
            )
        entity_line_id = pattern.entity_line_id
        if entity_line_id is None:
            entity_line_id = sphere.entity_line(pattern_id)
        if entity_line_id is None:
            raise ValueError(
                f"Cannot resolve entity line for pattern '{pattern_id}'"
            )

        line = sphere.lines.get(entity_line_id)
        if line and line.columns:
            col_names = {c.name for c in line.columns}
            if group_by not in col_names:
                raise ValueError(
                    f"Column '{group_by}' not found on entity line "
                    f"'{entity_line_id}'. Available: {sorted(col_names)}"
                )

        # Use delta_norm >= theta_norm to match anomaly_summary / find_anomalies
        # semantics. The is_anomaly column uses per-group thetas for
        # grouped patterns, which diverges from the global threshold.
        theta_norm = float(np.linalg.norm(pattern.theta))
        geom = self._storage.read_geometry(
            pattern_id, version,
            columns=["primary_key", "delta_norm"],
            filter=f"delta_norm >= {theta_norm}" if theta_norm > 0.0 else "is_anomaly = true",
            sample_size=sample_size,
        )
        total = self._storage.count_geometry_rows(pattern_id, version)
        if geom.num_rows == 0:
            return {
                "pattern_id": pattern_id,
                "group_by": group_by,
                "total_entities": total,
                "total_anomalies": 0,
                "anomaly_rate": 0.0,
                "groups_returned": 0,
                "groups": [],
            }

        anom_pks = geom["primary_key"].to_pylist()
        anom_norms = geom["delta_norm"].to_numpy(zero_copy_only=False)

        # Read entity points — combine group_by + filter columns in single read
        read_cols = [group_by]
        if property_filters:
            read_cols = list(dict.fromkeys(
                read_cols + list(property_filters.keys())
            ))
        pts_tbl = self._storage.read_points_batch(
            entity_line_id, version, anom_pks,
            columns=read_cols,
        )

        # Apply property_filters if set
        if property_filters:
            from hypertopos.engine.aggregation import _apply_event_filters
            pts_tbl = _apply_event_filters(pts_tbl, property_filters)
            # Narrow anom_pks/norms to surviving keys
            surviving = set(pts_tbl["primary_key"].to_pylist())
            keep = [i for i, pk in enumerate(anom_pks) if pk in surviving]
            anom_pks = [anom_pks[i] for i in keep]
            anom_norms = anom_norms[keep]

        pk_to_norm = dict(zip(anom_pks, anom_norms.tolist()))

        from collections import defaultdict
        groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
        pks = pts_tbl["primary_key"].to_pylist()
        gvs = pts_tbl[group_by].to_pylist()
        seen_pks: set[str] = set()
        for pk, gv in zip(pks, gvs):
            if pk not in pk_to_norm or pk in seen_pks:
                continue
            seen_pks.add(pk)
            gk = str(gv) if gv is not None else "(null)"
            groups[gk].append((pk, pk_to_norm[pk]))

        group_list = []
        for gk, members in groups.items():
            norms = [m[1] for m in members]
            entry: dict[str, Any] = {
                "group_key": gk,
                "anomaly_count": len(members),
                "mean_delta_norm": round(float(np.mean(norms)), 4),
            }
            if include_keys:
                entry["entity_keys"] = [m[0] for m in members[:keys_per_group]]
            group_list.append(entry)

        group_list.sort(key=lambda g: g["anomaly_count"], reverse=True)

        grouped_count = sum(g["anomaly_count"] for g in group_list)
        ungrouped_count = len(anom_pks) - grouped_count

        result = {
            "pattern_id": pattern_id,
            "group_by": group_by,
            "total_entities": total,
            "total_anomalies": len(anom_pks),
            "ungrouped_anomalies": ungrouped_count,
            "anomaly_rate": round(len(anom_pks) / total, 4) if total > 0 else 0.0,
            "groups_returned": min(len(group_list), top_n),
            "groups": group_list[:top_n],
        }
        if ungrouped_count > 0:
            result["ungrouped_note"] = (
                f"{ungrouped_count} anomalous entities have null/missing "
                f"'{group_by}' or are absent from entity line — not in any group."
            )
        return result

    def current_polygon(self, pattern_id: str) -> Polygon:
        if not isinstance(self._position, Point):
            raise GDSPositionError("current_polygon requires position to be a Point")
        try:
            polygon = self._engine.build_polygon(
                self._position.primary_key, pattern_id, self._manifest
            )
        except KeyError as exc:
            sphere = self._storage.read_sphere()
            pattern = sphere.patterns.get(pattern_id)
            if pattern is not None:
                relation_line_ids = {r.line_id for r in pattern.relations}
                if self._position.line_id in relation_line_ids:
                    raise GDSNavigationError(
                        f"No geometry for {self._position.primary_key} in {pattern_id}"
                        f" — '{self._position.line_id}' is a relation line in this pattern,"
                        f" not the pattern entity line."
                        f" Use get_event_polygons or aggregate to explore"
                        f" {self._position.primary_key}'s connections."
                    ) from exc
            raise GDSNavigationError(str(exc)) from exc

        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version,
            primary_key=self._position.primary_key,
            columns=["delta", "delta_norm", "is_anomaly", "delta_rank_pct"],
        )
        if geo.num_rows > 0:
            stored_delta = np.array(geo["delta"][0].as_py(), dtype=np.float32)
            stored_delta_norm = float(geo["delta_norm"][0].as_py())
            stored_is_anomaly = bool(geo["is_anomaly"][0].as_py())
            pct_val = geo["delta_rank_pct"][0].as_py()
            stored_delta_rank_pct = float(pct_val) if pct_val is not None else None
            polygon = dataclasses.replace(
                polygon,
                delta=stored_delta,
                delta_norm=stored_delta_norm,
                is_anomaly=stored_is_anomaly,
                delta_rank_pct=stored_delta_rank_pct,
            )
        return polygon

    def current_solid(
        self, pattern_id: str, filters: dict[str, str | list[str]] | None = None
    ) -> Solid:
        if not isinstance(self._position, Point):
            raise GDSPositionError("current_solid requires position to be a Point")
        try:
            return self._engine.build_solid(
                self._position.primary_key, pattern_id, self._manifest, filters=filters
            )
        except KeyError as exc:
            sphere = self._storage.read_sphere()
            pattern = sphere.patterns.get(pattern_id)
            if pattern is not None:
                relation_line_ids = {r.line_id for r in pattern.relations}
                if self._position.line_id in relation_line_ids:
                    raise GDSNavigationError(
                        f"No geometry for {self._position.primary_key} in {pattern_id}"
                        f" — '{self._position.line_id}' is a relation line in this pattern,"
                        f" not the pattern entity line."
                        f" Use get_event_polygons or aggregate to explore"
                        f" {self._position.primary_key}'s connections."
                    ) from exc
            raise GDSNavigationError(str(exc)) from exc

    def event_polygons_for(
        self,
        entity_key: str,
        event_pattern_id: str,
        filters: list[dict[str, str]] | None = None,
        geometry_filters: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
        sample_size: int | None = None,
        sample_pct: float | None = None,
        seed: int | None = None,
    ) -> list[Polygon]:
        """Return event polygons whose edges reference *entity_key*.

        When *filters* are provided and geometry has entity_keys with a
        LABEL_LIST index, uses it for O(log n) lookup instead of a full scan.
        Falls back to full scan + Python filter if index unavailable.

        *filters* is a list of ``{line, key}`` dicts (AND semantics): only
        polygons whose edge list contains an entry matching every filter are
        returned.
        """
        version = self._resolve_version(event_pattern_id)

        # Build column projection for Polygon construction.
        # Extend with delta_dim_* columns when geometry_filters["delta_dim"] is set.
        _epf_cols: list[str] = list(self._POLYGON_COLUMNS)
        if geometry_filters and "delta_dim" in geometry_filters:
            sphere_tmp = self._storage.read_sphere()
            _ep_tmp = sphere_tmp.patterns[event_pattern_id]
            for dim_name in geometry_filters["delta_dim"]:
                idx = _ep_tmp.dim_index(dim_name)
                _epf_cols.append(f"delta_dim_{idx}")

        if filters:
            # Try entity_keys index path: resolve PKs for entity_key + each filter
            entity_pks = self._storage.resolve_primary_keys_by_edge(
                event_pattern_id, version, None, entity_key
            )
            if entity_pks is not None:
                # Index available — resolve each filter and intersect
                pk_sets: list[set[str]] = [set(entity_pks)]
                for f in filters:
                    filter_pks = self._storage.resolve_primary_keys_by_edge(
                        event_pattern_id, version, f["line"], f["key"]
                    )
                    pk_sets.append(set(filter_pks or []))

                final_pks = pk_sets[0]
                for s in pk_sets[1:]:
                    final_pks &= s

                if not final_pks:
                    # No matching rows — return empty table immediately
                    table = self._storage.read_geometry(
                        event_pattern_id, version,
                        columns=_epf_cols,
                        filter="primary_key = '__no_match__'",
                    ).slice(0, 0)
                else:
                    escaped_pks = [p.replace("'", "''") for p in final_pks]
                    pk_filter = "primary_key IN ('" + "', '".join(escaped_pks) + "')"
                    table = self._storage.read_geometry(
                        event_pattern_id, version, columns=_epf_cols,
                        filter=pk_filter,
                    )
            else:
                # entity_keys index unavailable — fall back to full scan + Python filter
                table = self._storage.read_geometry(
                    event_pattern_id, version, point_keys=[entity_key],
                    columns=_epf_cols,
                )
                sphere = self._storage.read_sphere()
                _ep = sphere.patterns[event_pattern_id]
                line_ids_col, pt_keys_col = _table_edge_line_and_point_keys(
                    table, _ep.relations,
                )
                keep = [
                    i for i, (row_lids, row_pks) in enumerate(
                        zip(line_ids_col, pt_keys_col, strict=False)
                    )
                    if all(
                        any(
                            lid == f["line"] and pk == f["key"]
                            for lid, pk in zip(row_lids or [], row_pks or [], strict=False)
                        )
                        for f in filters
                    )
                ]
                table = (
                    table.take(pa.array(keep, type=pa.int64()))
                    if keep
                    else table.slice(0, 0)
                )
        else:
            # No filters — use entity_keys scan
            table = self._storage.read_geometry(
                event_pattern_id, version, point_keys=[entity_key],
                columns=_epf_cols,
            )
        # Record total before geometry_filters are applied so callers can
        # obtain total_unfiltered without a second scan.
        self._last_total_pre_geometry_filter = table.num_rows if geometry_filters else None
        if geometry_filters:
            _supported = {"is_anomaly", "delta_rank_pct", "delta_dim"}
            unknown = set(geometry_filters) - _supported
            if unknown:
                raise ValueError(
                    f"Unknown geometry_filters keys: {unknown}. Supported: {_supported}."
                )
            if "is_anomaly" in geometry_filters:
                table = table.filter(
                    pc.equal(table["is_anomaly"], bool(geometry_filters["is_anomaly"]))
                )
            if "delta_rank_pct" in geometry_filters:
                spec = geometry_filters["delta_rank_pct"]
                _ops = {"gt": pc.greater, "gte": pc.greater_equal,
                        "lt": pc.less, "lte": pc.less_equal, "eq": pc.equal}
                if isinstance(spec, dict):
                    mask = None
                    for op_name, value in spec.items():
                        if op_name not in _ops:
                            raise ValueError(
                                f"Unknown comparison op '{op_name}'. Supported: {list(_ops)}"
                            )
                        pred = _ops[op_name](table["delta_rank_pct"], value)
                        mask = pred if mask is None else pc.and_(mask, pred)
                    table = table.filter(mask)
                else:
                    table = table.filter(pc.equal(table["delta_rank_pct"], float(spec)))
            if "delta_dim" in geometry_filters and table.num_rows > 0:
                sphere = self._storage.read_sphere()
                event_pattern = sphere.patterns[event_pattern_id]
                delta_dim_spec = geometry_filters["delta_dim"]
                _pc_ops = {
                    "gt": pc.greater, "gte": pc.greater_equal,
                    "lt": pc.less, "lte": pc.less_equal, "eq": pc.equal,
                }
                mask = None
                for dim_name, predicates in delta_dim_spec.items():
                    idx = event_pattern.dim_index(dim_name)
                    col_name = f"delta_dim_{idx}"
                    col = table[col_name]
                    for op_name, threshold in predicates.items():
                        if op_name not in _pc_ops:
                            raise ValueError(
                                f"Unknown comparison op '{op_name}'. "
                                f"Supported: {list(_pc_ops)}"
                            )
                        pred = _pc_ops[op_name](col, float(threshold))
                        mask = pred if mask is None else pc.and_(mask, pred)
                if mask is not None:
                    table = table.filter(mask)
        # Store total (post-filters, pre-pagination) for callers
        self._last_total_post_geometry_filter = table.num_rows
        if table.num_rows == 0:
            return []

        # Apply limit/offset at Arrow level before Polygon construction
        if offset > 0 or limit is not None:
            end = table.num_rows if limit is None else min(offset + limit, table.num_rows)
            if offset >= table.num_rows:
                return []
            table = table.slice(offset, end - offset)

        sphere = self._storage.read_sphere()
        _ep = sphere.patterns[event_pattern_id]

        results: list[Polygon] = []
        for i in range(table.num_rows):
            row = {col: table[col][i].as_py() for col in table.schema.names}
            edges = _reconstruct_edges_from_row(row, _ep.relations)
            results.append(Polygon(
                primary_key=row["primary_key"],
                pattern_id=row.get("pattern_id", event_pattern_id),
                pattern_ver=row.get("pattern_ver", version),
                pattern_type=row.get("pattern_type", _ep.pattern_type),
                scale=row["scale"],
                delta=np.array(row["delta"], dtype=np.float32),
                delta_norm=float(row["delta_norm"]),
                is_anomaly=bool(row["is_anomaly"]),
                edges=edges,
                last_refresh_at=row["last_refresh_at"],
                updated_at=row["updated_at"],
                delta_rank_pct=(
                    None if row.get("delta_rank_pct") is None
                    else float(row["delta_rank_pct"])
                ),
            ))

        # Apply random sampling if requested (post-construction, pre-return)
        if sample_size is not None or sample_pct is not None:
            total_polygons = len(results)
            if sample_pct is not None:
                n = max(1, min(int(total_polygons * sample_pct), total_polygons))
            else:
                n = min(sample_size, total_polygons)  # type: ignore[arg-type]
            if n < total_polygons:
                rng = np.random.default_rng(seed)
                chosen = sorted(rng.choice(total_polygons, size=n, replace=False))
                results = [results[i] for i in chosen]

        return results

    def π6_attract_boundary(
        self,
        alias_id: str,
        pattern_id: str,
        direction: Literal["in", "out", "both"] = "both",
        top_n: int = 10,
        fdr_alpha: float | None = None,
        fdr_method: str = "bh",
        p_value_method: str = "rank",
        select: str = "top_norm",
    ) -> list[tuple[Polygon, float]]:
        """Find entities closest to an alias segment boundary (cutting plane).

        Returns (polygon, signed_distance) pairs sorted by |signed_distance|.
        signed_distance >= 0 → inside segment, < 0 → outside segment.
        """
        if fdr_method not in ("bh", "storey"):
            raise ValueError(
                f"fdr_method must be 'bh' or 'storey', got {fdr_method!r}"
            )
        sphere = self._storage.read_sphere()
        alias = sphere.aliases.get(alias_id)
        if alias is None:
            raise GDSNavigationError(f"Alias '{alias_id}' not found")
        cp = alias.filter.cutting_plane
        if cp is None:
            raise GDSNavigationError(
                f"Alias '{alias_id}' has no cutting_plane — π6 requires a geometric boundary"
            )

        version = self._resolve_version(pattern_id)

        # Pass 1: light scan — only delta needed for boundary ranking
        light = self._storage.read_geometry(
            pattern_id, version, columns=["primary_key", "delta"],
        )
        if light.num_rows == 0:
            return []

        delta_matrix = delta_matrix_from_arrow(light)  # (n, d)
        scores = cp.signed_distances_batch(delta_matrix)  # (n,)

        # direction filter (vectorized)
        if direction == "in":
            mask = scores >= 0
        elif direction == "out":
            mask = scores < 0
        else:
            mask = np.ones(len(scores), dtype=bool)

        filtered_idx = np.where(mask)[0]
        filtered_scores = scores[filtered_idx]

        if len(filtered_idx) == 0:
            return []
        abs_scores = np.abs(filtered_scores)
        if len(abs_scores) > top_n:
            part = np.argpartition(abs_scores, top_n)[:top_n]
            rank_order = part[np.argsort(abs_scores[part])]
        else:
            rank_order = np.argsort(abs_scores)

        top_orig_idx = [int(filtered_idx[li]) for li in rank_order]
        top_keys = [str(light["primary_key"][i].as_py()) for i in top_orig_idx]
        score_lookup = {
            str(light["primary_key"][int(filtered_idx[li])].as_py()): float(filtered_scores[li])
            for li in rank_order
        }
        delta_lookup = {
            str(light["primary_key"][i].as_py()): delta_matrix[i].copy()
            for i in top_orig_idx
        }

        # Pass 2: full rows for top-N only (Lance pushdown on primary_key)
        escaped = [k.replace("'", "''") for k in top_keys]
        pk_in = ", ".join(f"'{k}'" for k in escaped)
        full = self._storage.read_geometry(
            pattern_id, version, filter=f"primary_key IN ({pk_in})",
            columns=self._POLYGON_COLUMNS,
        )

        pattern = sphere.patterns[pattern_id]
        results: list[tuple[Polygon, float]] = []
        for i in range(full.num_rows):
            row = {col: full[col][i].as_py() for col in full.schema.names}
            pk = str(row["primary_key"])
            signed_dist = score_lookup.get(pk, 0.0)
            delta = delta_lookup.get(pk, np.array(row["delta"], dtype=np.float32))
            edges = _reconstruct_edges_from_row(row, pattern.relations)
            polygon = Polygon(
                primary_key=pk,
                pattern_id=row.get("pattern_id", pattern_id),
                pattern_ver=row.get("pattern_ver", version),
                pattern_type=row.get("pattern_type", pattern.pattern_type),
                scale=row["scale"],
                delta=delta,
                delta_norm=float(row["delta_norm"]),
                is_anomaly=bool(row["is_anomaly"]),
                edges=edges,
                last_refresh_at=row["last_refresh_at"],
                updated_at=row["updated_at"],
                delta_rank_pct=(
                    None if row.get("delta_rank_pct") is None
                    else float(row["delta_rank_pct"])
                ),
            )
            results.append((polygon, signed_dist))
        results.sort(key=lambda x: abs(x[1]))

        # --- FDR + diverse selection post-processing ---
        if fdr_alpha is not None or select != "top_norm":
            polygons = [p for p, _ in results]
            polygons = self._apply_fdr_select_polygons(
                polygons, fdr_alpha=fdr_alpha, select=select, top_n=top_n,
                fdr_method=fdr_method, p_value_method=p_value_method,
                pattern_df=len(pattern.theta) if p_value_method == "chi2" else None,
            )
            kept_keys = {p.primary_key for p in polygons}
            dist_lookup = {p.primary_key: d for p, d in results}
            results = [(p, dist_lookup[p.primary_key]) for p in polygons]

        return results

    def contrast_populations(
        self,
        pattern_id: str,
        group_a: dict,
        group_b: dict | None = None,
    ) -> list[dict]:
        """Compare two entity groups dimension-by-dimension.

        Returns per-dimension contrast sorted by |effect_size| descending,
        answering "why are these groups different?".

        group_a / group_b accept three spec formats:
          {"anomaly": bool}          — select by is_anomaly flag
          {"keys": ["K-1", "K-2"]}  — explicit business key list
          {"alias": "id", "side": "in"|"out"}  — cutting-plane segment

        When group_b is None the complement of group_a is used.
        """
        version = self._resolve_version(pattern_id)
        table = self._storage.read_geometry(pattern_id, version, columns=self._CONTRAST_COLUMNS)
        if table.num_rows == 0:
            return []

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]

        keys = table["primary_key"].to_pylist()
        delta_matrix = delta_matrix_from_arrow(table)

        mask_a = self._resolve_group_mask(group_a, table, keys, delta_matrix, pattern_id)
        mask_b = (
            ~mask_a
            if group_b is None
            else self._resolve_group_mask(group_b, table, keys, delta_matrix, pattern_id)
        )

        dim_labels = (
            [rel.display_name or rel.line_id for rel in pattern.relations]
            + list(pattern.prop_columns)
        )
        return self._engine.contrast_populations(delta_matrix, mask_a, mask_b, dim_labels)

    def _resolve_group_mask(
        self,
        group_spec: dict,
        table: Any,
        keys: list[str],
        delta_matrix: np.ndarray,
        pattern_id: str = "",
    ) -> np.ndarray:
        """Resolve a group spec dict to a boolean mask over the geometry table rows."""
        if "anomaly" in group_spec:
            target = bool(group_spec["anomaly"])
            anomaly_arr = table["is_anomaly"].combine_chunks().to_numpy(zero_copy_only=False)
            return anomaly_arr == target
        if "keys" in group_spec:
            key_set = set(group_spec["keys"])
            mask = np.array([k in key_set for k in keys])
            if not np.any(mask):
                sample = sorted(key_set)[:3]
                raise GDSNavigationError(
                    f"None of the {len(key_set)} specified keys found in '{pattern_id}' geometry. "
                    f"These entities may not have geometry in this pattern — "
                    f"check that the keys belong to the primary entity of '{pattern_id}'. "
                    f"Keys checked (sample): {sample}"
                )
            return mask
        if "alias" in group_spec:
            alias_id = str(group_spec["alias"])
            side = str(group_spec.get("side", "in"))
            sphere = self._storage.read_sphere()
            alias = sphere.aliases.get(alias_id)
            if alias is None:
                raise GDSNavigationError(f"Alias '{alias_id}' not found")
            if alias.base_pattern_id != pattern_id:
                pat = sphere.patterns[pattern_id]
                base_pat = sphere.patterns.get(alias.base_pattern_id)
                base_dim = len(base_pat.relations) if base_pat is not None else "unknown"
                raise ValueError(
                    f"Alias '{alias_id}' has base_pattern_id='{alias.base_pattern_id}' "
                    f"(delta_dim={base_dim}) "
                    f"but contrast_populations was called with pattern_id='{pattern_id}' "
                    f"(delta_dim={pat.delta_dim()}). "
                    f"Use the alias's base pattern or choose a compatible alias."
                )
            cp = alias.filter.cutting_plane
            if cp is None:
                raise GDSNavigationError(
                    f"Alias '{alias_id}' has no cutting_plane"
                    " — alias mode requires a geometric boundary"
                )
            if side == "in":
                return np.array([cp.contains(delta_matrix[i]) for i in range(len(keys))])
            else:
                return np.array([not cp.contains(delta_matrix[i]) for i in range(len(keys))])
        if "edge" in group_spec:
            edge_spec = group_spec["edge"]
            target_key = str(edge_spec["key"])
            target_line = edge_spec.get("line_id")
            sphere = self._storage.read_sphere()
            _pat = sphere.patterns.get(pattern_id)
            _rels = _pat.relations if _pat else None
            # Derive alive-only line_ids and point_keys — edges struct or entity_keys fallback
            line_ids_col, pt_keys_col = _table_edge_line_and_point_keys(
                table, _rels,
            )
            # Guard: detect continuous-mode patterns (edge_max) — all point_keys are ""
            # Sample first rows for the target line; if all keys are "" → continuous mode.
            # If no edge to target_line in the sample, guard is skipped (conservative: avoids
            # false positives on sparse lines).
            _sample_keys = [
                pk
                for lids, pks in zip(line_ids_col[:5], pt_keys_col[:5], strict=False)
                for lid, pk in zip(lids or [], pks or [], strict=False)
                if target_line is None or lid == target_line
            ]
            if _sample_keys and all(k == "" for k in _sample_keys):
                raise GDSNavigationError(
                    f"Cannot filter by edge to '{target_line}': pattern '{pattern_id}' uses "
                    f"continuous mode (edge_max) — edges store counts, not entity keys. "
                    f"Specify the group by 'anomaly', 'keys', or 'alias' instead."
                )
            mask = []
            for i in range(len(keys)):
                row_lines = line_ids_col[i] or []
                row_keys = pt_keys_col[i] or []
                matched = any(
                    pk == target_key and (target_line is None or lid == target_line)
                    for lid, pk in zip(row_lines, row_keys, strict=False)
                )
                mask.append(matched)
            return np.array(mask)
        raise GDSNavigationError(
            "Unknown group spec: expected 'anomaly', 'keys', 'alias', or 'edge' key, "
            f"got {list(group_spec.keys())}"
        )

    def centroid_map(
        self,
        pattern_id: str,
        group_by_line: str,
        group_by_property: str | None = None,
        sample_size: int | None = None,
        include_drift: bool = True,
    ) -> dict:
        """Compute centroid map — meta-geometry of entity groups.

        Groups entities by their edge to ``group_by_line`` (or by a property
        of that line when ``group_by_property`` is given), then computes
        global + per-group centroids in delta-space.

        Args:
            pattern_id: Pattern to analyse.
            group_by_line: Line ID whose edges define group membership.
            group_by_property: Optional ``"line_id:property"`` — use property
                value as group label instead of edge key.

        Returns:
            Dict with global_centroid, group_centroids, inter_centroid_distances,
            structural_outlier, dimensions.  Empty dict when geometry is empty.
        """
        version = self._resolve_version(pattern_id)

        # Early detection: continuous-mode patterns have edge_max set — all edges store
        # counts not entity keys, so edge-based grouping is impossible.  read_sphere() is
        # fast (cached sphere.json), so this check saves the expensive read_geometry call.
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.edge_max is not None and group_by_property is None:
            raise ValueError(
                f"Cannot group by line '{group_by_line}': all edges use continuous mode "
                f"(edge_max) — no entity keys stored. "
                f"Use group_by_property='<line_id>:<property>' instead."
            )

        # Guard: detect self-referential grouping — when group_by_line is the entity's own
        # line, all labels are None and compute_centroid_map returns {}, yielding the
        # confusing "No geometry" error.  Raise early with an actionable message.
        # Exception: continuous-mode patterns with group_by_property set — the entity
        # groups by its own primary_key, then property lookup maps keys to values.
        entity_line_id = sphere.entity_line(pattern_id)
        _self_group = entity_line_id is not None and group_by_line == entity_line_id
        # Continuous self-group: entity PKs are the grouping keys; property lookup maps them to
        # values. This is the only valid use of group_by_property on a continuous-mode pattern.
        _use_pk_as_label = _self_group and pattern.edge_max is not None
        if _self_group and not _use_pk_as_label:
            raise ValueError(
                f"Cannot group by '{group_by_line}': this is the entity's own line. "
                f"Use group_by_property='{group_by_line}:<property_name>' to group "
                f"by a property of the entity itself "
                f"(e.g. group_by_property='{group_by_line}:loyalty_tier')."
            )
        # Continuous-mode + property but not self-group: edge keys are always empty,
        # so group membership can never be resolved. Raise before the expensive read_geometry.
        if pattern.edge_max is not None and group_by_property is not None and not _self_group:
            raise ValueError(
                f"Cannot use group_by_property with continuous-mode pattern '{pattern_id}' "
                f"when group_by_line='{group_by_line}' is not the entity's own line "
                f"('{entity_line_id}'). Set group_by_line='{entity_line_id}' to use "
                f"self-grouping."
            )

        table = self._storage.read_geometry(pattern_id, version, columns=self._CENTROID_COLUMNS)
        if table.num_rows == 0:
            return {}

        # Explicit sampling — agent decides via sample_size param
        _sampled = False
        _total_before_sample = table.num_rows
        if sample_size is not None and table.num_rows > sample_size:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(
                table.num_rows, size=sample_size, replace=False,
            ))
            table = table.take(pa.array(idx, type=pa.int64()))
            _sampled = True

        keys = table["primary_key"].to_pylist()
        delta_matrix = delta_matrix_from_arrow(table)

        # Extract per-row group labels from edges
        if _use_pk_as_label:
            # Continuous self-group: entity PK is the grouping key; property lookup maps to values
            raw_labels = keys
        else:
            raw_labels = self._extract_group_labels(
                table, keys, group_by_line, pattern.relations,
            )

        # Fallback guard: detect continuous-mode patterns (all edge keys empty → no FK stored).
        # Should never trigger when edge_max is set (caught above), but kept as a safety net
        # for edge cases where edge_max is unset yet all point_keys happen to be empty.
        present = [lb for lb in raw_labels if lb is not None]
        if present and all(lb == "" for lb in present):
            raise ValueError(
                f"Cannot group by line '{group_by_line}': all edges use continuous mode "
                f"(edge_max) — no entity keys stored. "
                f"Use group_by_property='<line_id>:<property>' instead."
            )

        # Optionally map edge keys to property values
        if group_by_property:
            if ":" not in group_by_property:
                raise GDSNavigationError(
                    f"group_by_property must be 'line_id:property_name', got '{group_by_property}'"
                )
            prop_line_id, prop_name = group_by_property.split(":", 1)
            prop_version = self._manifest.line_version(prop_line_id) or 1
            prop_table = self._storage.read_points(prop_line_id, prop_version)
            prop_keys = prop_table["primary_key"].to_pylist()
            prop_vals = prop_table[prop_name].to_pylist()
            prop_lookup = {
                pk: (str(pv) if pv is not None else None)
                for pk, pv in zip(prop_keys, prop_vals, strict=False)
            }
            labels: list[str | None] = [prop_lookup.get(rl) if rl else None for rl in raw_labels]
        else:
            labels = raw_labels

        # Filter out entities without a label
        mask = np.array([lb is not None for lb in labels])
        if not mask.any():
            return {}
        delta_matrix = delta_matrix[mask]
        filtered_keys = [k for k, lb in zip(keys, labels, strict=False) if lb is not None]
        labels = [lb for lb in labels if lb is not None]

        dim_labels = (
            [rel.display_name or rel.line_id for rel in pattern.relations]
            + list(pattern.prop_columns)
        )
        result = self._engine.compute_centroid_map(
            delta_matrix, labels, dim_labels, entity_keys=filtered_keys
        )
        if _sampled:
            result["sampled"] = True
            result["sample_size"] = sample_size
            result["total_entities"] = _total_before_sample

        if include_drift and pattern.pattern_type == "anchor":
            from hypertopos.engine.forecast import extrapolate_trajectory

            for g in result.get("group_centroids", []):
                member_keys = g.pop("member_samples", [])
                if not member_keys:
                    continue
                drift_vectors: list[np.ndarray] = []
                n_samples = 0
                for pk in member_keys:
                    try:
                        solid = self._engine.build_solid(
                            pk, pattern_id, self._manifest,
                        )
                    except _NAVIGATION_RECOVERABLE_ERRORS:
                        continue
                    if len(solid.slices) < 3:
                        continue
                    deltas_arr = [s.delta_snapshot for s in solid.slices]
                    traj = extrapolate_trajectory(deltas_arr, horizon=1)
                    drift = (
                        traj.predicted_delta
                        - deltas_arr[-1].astype(np.float32)
                    )
                    drift_vectors.append(drift)
                    n_samples += 1
                if drift_vectors:
                    avg_drift = np.mean(drift_vectors, axis=0)
                    g["centroid_drift"] = {
                        "direction": [
                            round(float(v), 6) for v in avg_drift
                        ],
                        "magnitude": round(
                            float(np.linalg.norm(avg_drift)), 6,
                        ),
                        "reliability": (
                            "medium" if n_samples >= 3 else "low"
                        ),
                    }

        return result

    def _extract_group_labels(
        self,
        table: Any,
        keys: list[str],
        group_by_line: str,
        relations: list | None = None,
    ) -> list[str | None]:
        """Extract group label per row from edges or entity_keys for a given line_id.

        Derives alive ``line_id`` / ``point_key`` from the ``edges`` struct column,
        falling back to ``entity_keys`` + ``relations`` when ``edges`` is absent
        (event geometry without edges column).
        Returns the ``point_key`` of the first alive edge matching ``group_by_line``,
        or ``None`` if the entity has no edge to that line.
        """
        line_ids_col, pt_keys_col = _table_edge_line_and_point_keys(
            table, relations,
        )
        labels: list[str | None] = []
        for i in range(len(keys)):
            row_lines = line_ids_col[i] or []
            row_keys = pt_keys_col[i] or []
            label = None
            for lid, pk in zip(row_lines, row_keys, strict=False):
                if lid == group_by_line:
                    label = pk
                    break
            labels.append(label)
        return labels

    def find_similar_entities(
        self,
        primary_key: str,
        pattern_id: str,
        top_n: int = 5,
        filter_expr: str | None = None,
        missing_edge_to: str | None = None,
        dim_mask: list[str] | None = None,
        metric: str = "L2",
    ) -> SimilarityResult:
        """Find top_n entities most similar to primary_key by delta vector distance.

        Returns ``SimilarityResult`` — a list subclass of (primary_key, distance)
        tuples sorted ascending.  Carries an optional ``degenerate_warning`` when
        >50 % of neighbors have distance = 0 (many inactive entities).

        Works for both anchor and event patterns — event polygons have delta vectors
        and support ANN search.

        filter_expr: optional Lance SQL predicate applied at ANN time, enabling single-pass
        ANN + scalar filter (e.g. 'is_anomaly = true', 'delta_rank_pct > 95').

        missing_edge_to: optional line_id — post-filter that keeps only entities WITHOUT
        an edge to the target line. Over-fetches 5x from ANN to compensate for attrition.
        """
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()

        if missing_edge_to:
            pattern = sphere.patterns[pattern_id]
            if pattern.pattern_type == "event":
                raise ValueError(
                    "missing_edge_to is not supported for event patterns — "
                    "use missing_edge_to at the aggregate level instead"
                )
            if missing_edge_to not in sphere.lines:
                raise ValueError(
                    f"Unknown line '{missing_edge_to}' in missing_edge_to. "
                    f"Available: {sorted(sphere.lines)}"
                )

        table = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key, columns=["delta"],
        )
        if table.num_rows == 0:
            raise KeyError(f"Entity '{primary_key}' not found in {pattern_id} v{version}")
        ref_delta = np.array(table["delta"][0].as_py(), dtype=np.float32)

        # Resolve dim_mask names → indices
        dim_mask_indices: list[int] | None = None
        if dim_mask is not None:
            if not dim_mask:
                raise ValueError("dim_mask must be non-empty")
            pattern = sphere.patterns[pattern_id]
            labels = pattern.dim_labels
            dim_mask_indices = []
            for name in dim_mask:
                if name in labels:
                    dim_mask_indices.append(labels.index(name))
                else:
                    raise ValueError(
                        f"Unknown dimension '{name}' in dim_mask. "
                        f"Available: {labels}"
                    )
        if metric not in ("L2", "cosine"):
            raise ValueError(f"metric must be 'L2' or 'cosine', got '{metric}'")

        # Over-fetch when post-filter will remove some results
        fetch_n = top_n * 5 if missing_edge_to else top_n

        results = self._engine.find_nearest(
            ref_delta=ref_delta,
            pattern_id=pattern_id,
            version=version,
            top_n=fetch_n,
            exclude_keys={primary_key},
            filter_expr=filter_expr,
            dim_mask_indices=dim_mask_indices,
            metric=metric,
        )

        if missing_edge_to and results:
            # Read edges/entity_keys for candidate keys — not the full table
            candidate_keys = {k for k, _ in results}
            _pat = sphere.patterns[pattern_id]
            geo = self._storage.read_geometry(
                pattern_id, version,
                columns=["primary_key", "edges", "entity_keys"],
            )
            key_col = geo.column("primary_key")
            candidate_mask = pc.is_in(key_col, pa.array(list(candidate_keys)))
            geo = geo.filter(candidate_mask)
            pk_list = geo.column("primary_key").to_pylist()
            eli_list = _table_edge_line_ids(geo, _pat.relations)
            eli_map = dict(zip(pk_list, eli_list, strict=False))
            results = [
                (k, d) for k, d in results
                if missing_edge_to not in (eli_map.get(k) or [])
            ][:top_n]

        # B17: detect degenerate ANN results (many identical delta vectors)
        degenerate_warning = None
        if results:
            zero_count = sum(1 for _, d in results if d == 0.0)
            if zero_count >= 2 and zero_count > len(results) // 2:
                degenerate_warning = (
                    f"Degenerate: {zero_count}/{len(results)} neighbors at distance=0 "
                    f"(inactive entities). Results may be misleading."
                )

        return SimilarityResult(results, degenerate_warning=degenerate_warning)

    def get_entity_geometry_meta(
        self,
        primary_key: str,
        pattern_id: str,
    ) -> dict:
        """Read stored geometry metadata for a single entity.

        Returns dict with delta_norm (float), is_anomaly (bool), delta_rank_pct (float | None).
        Raises KeyError if entity not found in the geometry dataset.
        """
        version = self._resolve_version(pattern_id)
        cols = ["delta_norm", "is_anomaly", "delta_rank_pct"]
        table = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key, columns=cols,
        )
        if table.num_rows == 0:
            raise KeyError(f"Entity '{primary_key}' not found in {pattern_id} v{version}")
        pct = table["delta_rank_pct"][0].as_py()
        return {
            "delta_norm": float(table["delta_norm"][0].as_py()),
            "is_anomaly": bool(table["is_anomaly"][0].as_py()),
            "delta_rank_pct": float(pct) if pct is not None else None,
        }

    def compare_entities_intraclass(
        self,
        key_a: str,
        key_b: str,
        pattern_id: str,
    ) -> dict:
        """Compare two entities by their stored delta vectors (direct key lookup).

        Reads delta from the geometry Lance dataset via primary_key BTREE index —
        never uses ANN. This guarantees correctness even when the ANN index is stale.

        Returns a dict with keys: distance, delta_norm_a, delta_rank_pct_a,
        is_anomaly_a, delta_norm_b, delta_rank_pct_b, is_anomaly_b.
        Raises KeyError if either entity is not found in the geometry dataset.
        """
        version = self._resolve_version(pattern_id)
        cols = ["primary_key", "delta", "delta_norm", "delta_rank_pct", "is_anomaly"]

        table_a = self._storage.read_geometry(
            pattern_id, version, primary_key=key_a, columns=cols,
        )
        if table_a.num_rows == 0:
            raise KeyError(f"Entity '{key_a}' not found in {pattern_id} v{version}")

        table_b = self._storage.read_geometry(
            pattern_id, version, primary_key=key_b, columns=cols,
        )
        if table_b.num_rows == 0:
            raise KeyError(f"Entity '{key_b}' not found in {pattern_id} v{version}")

        delta_a = np.array(table_a["delta"][0].as_py(), dtype=np.float32)
        delta_b = np.array(table_b["delta"][0].as_py(), dtype=np.float32)
        distance = float(np.linalg.norm(delta_a - delta_b))

        def _pct(tbl: Any) -> float | None:
            col_names = tbl.schema.names
            if "delta_rank_pct" not in col_names:
                return None
            val = tbl["delta_rank_pct"][0].as_py()
            return float(val) if val is not None else None

        interpretation = (
            "identical shapes" if distance == 0.0
            else "very similar" if distance < 0.1
            else "similar" if distance < 0.5
            else "moderately different" if distance < 1.0
            else "very different"
        )

        return {
            "distance": distance,
            "interpretation": interpretation,
            "delta_norm_a": float(table_a["delta_norm"][0].as_py()),
            "delta_rank_pct_a": _pct(table_a),
            "is_anomaly_a": bool(table_a["is_anomaly"][0].as_py()),
            "delta_norm_b": float(table_b["delta_norm"][0].as_py()),
            "delta_rank_pct_b": _pct(table_b),
            "is_anomaly_b": bool(table_b["is_anomaly"][0].as_py()),
        }

    def compare_entities_temporal(
        self,
        key_a: str,
        key_b: str,
        pattern_id: str,
    ) -> dict[str, Any]:
        """Compare two entities by temporal trajectory (DTW distance).

        Builds solids for both entities and computes DTW distance between
        their deformation histories. Lower distance = more similar trajectories.
        """
        solid_a = self._engine.build_solid(key_a, pattern_id, self._manifest)
        solid_b = self._engine.build_solid(key_b, pattern_id, self._manifest)
        dist = self._engine.compute_distance_temporal(solid_a, solid_b)
        interpretation = (
            "identical history" if dist == 0.0
            else "similar history" if dist < 1.0
            else "divergent history" if dist < 3.0
            else "very different history"
        )
        return {
            "distance": round(float(dist), 4),
            "slices_a": len(solid_a.slices),
            "slices_b": len(solid_b.slices),
            "interpretation": interpretation,
        }

    def find_common_relations(
        self,
        key_a: str,
        key_b: str,
        pattern_id: str,
    ) -> dict[str, Any]:
        """Find common polygon relations (shared alive edges) between two entities.

        Returns a dict with keys: common (set of (line_id, point_key) tuples),
        edges_a, edges_b (alive edge counts for each polygon).
        """
        poly_a = self._engine.build_polygon(key_a, pattern_id, self._manifest)
        poly_b = self._engine.build_polygon(key_b, pattern_id, self._manifest)
        common = self._engine._find_common_polygons(poly_a, poly_b)
        return {
            "common": common,
            "edges_a": len(poly_a.alive_edges()),
            "edges_b": len(poly_b.alive_edges()),
        }

    def _find_counterparties_via_edges(
        self,
        primary_key: str,
        line_id: str,
        pattern_id: str,
        top_n: int = 20,
        *,
        timestamp_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Fast counterparty lookup via adjacency index.

        Returns same structure as find_counterparties but with ``amount_sum``
        and ``amount_max`` per counterparty entry.  Anomaly enrichment uses
        the resolved anchor pattern's geometry.

        ``timestamp_cutoff`` restricts the lookup to edges with
        ``timestamp <= timestamp_cutoff``.
        """
        adj = self._storage.get_adjacency(pattern_id)
        fwd_edges = adj.neighbors_out(primary_key, ts_to=timestamp_cutoff)
        rev_edges = adj.neighbors_in(primary_key, ts_to=timestamp_cutoff)

        def _group(edges: list, top_n: int) -> list[dict[str, Any]]:
            if not edges:
                return []
            from collections import defaultdict
            agg: dict[str, list[float]] = defaultdict(list)
            for target, _ts, amount, _ek in edges:
                agg[target].append(amount)
            pairs = sorted(
                [
                    (k, len(amounts), sum(amounts), max(amounts))
                    for k, amounts in agg.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]
            return [
                {
                    "key": k,
                    "tx_count": c,
                    "amount_sum": round(float(s), 2),
                    "amount_max": round(float(m), 2),
                }
                for k, c, s, m in pairs
            ]

        outgoing = _group(fwd_edges, top_n)
        incoming = _group(rev_edges, top_n)

        # Anomaly enrichment via anchor pattern geometry
        scoring_pattern = (
            self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        )
        all_cp_keys = {e["key"] for e in outgoing} | {e["key"] for e in incoming}
        geo_lookup: dict[str, dict[str, Any]] = {}
        _enrichment_warning: str | None = None

        if all_cp_keys:
            try:
                geo_version = self._resolve_version(scoring_pattern)

                # Detect composite-key patterns by sampling one geometry PK
                _composite_map: dict[str, str] = {}
                _geo_sample = self._storage.read_geometry(
                    scoring_pattern, geo_version,
                    columns=["primary_key"], sample_size=1,
                )
                if _geo_sample.num_rows > 0:
                    _sample_pk = str(_geo_sample["primary_key"][0].as_py())
                    for sep in ("→", "|"):
                        if sep in _sample_pk:
                            for cpk in all_cp_keys:
                                _composite_map[f"{primary_key}{sep}{cpk}"] = cpk
                            break

                # Read geometry — use point_keys for direct match, full scan for composite
                if _composite_map:
                    geo = self._storage.read_geometry(
                        scoring_pattern, geo_version,
                        columns=["primary_key", "is_anomaly", "delta_rank_pct"],
                    )
                else:
                    geo = self._storage.read_geometry(
                        scoring_pattern, geo_version,
                        point_keys=list(all_cp_keys),
                        columns=["primary_key", "is_anomaly", "delta_rank_pct"],
                    )

                for i in range(geo.num_rows):
                    pk = geo["primary_key"][i].as_py()
                    _data = {
                        "is_anomaly": bool(geo["is_anomaly"][i].as_py()),
                        "delta_rank_pct": round(
                            float(geo["delta_rank_pct"][i].as_py()), 2,
                        ),
                    }
                    if pk in all_cp_keys:
                        geo_lookup[pk] = _data
                    elif pk in _composite_map:
                        geo_lookup[_composite_map[pk]] = _data

                if all_cp_keys and not geo_lookup:
                    _enrichment_warning = (
                        f"Enrichment returned 0 matches for '{scoring_pattern}'. "
                        f"Pattern may use composite keys that don't match "
                        f"counterparty keys directly."
                    )
            except GDSNavigationError:
                pass  # no geometry available — skip enrichment

        for entry in (*outgoing, *incoming):
            if entry["key"] in geo_lookup:
                entry.update(geo_lookup[entry["key"]])

        anomalous_out = sum(1 for e in outgoing if e.get("is_anomaly"))
        anomalous_in = sum(1 for e in incoming if e.get("is_anomaly"))

        result: dict[str, Any] = {
            "primary_key": primary_key,
            "line_id": line_id,
            "outgoing": outgoing,
            "incoming": incoming,
            "summary": {
                "total_outgoing": len(outgoing),
                "total_incoming": len(incoming),
                "anomalous_outgoing": anomalous_out,
                "anomalous_incoming": anomalous_in,
            },
        }
        if _enrichment_warning:
            result["enrichment_warning"] = _enrichment_warning
        return result

    def find_counterparties(
        self,
        primary_key: str,
        line_id: str,
        from_col: str,
        to_col: str,
        pattern_id: str | None = None,
        top_n: int = 20,
        use_edge_table: bool = True,
        *,
        timestamp_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Find transaction counterparties of an entity with anomaly enrichment.

        When *pattern_id* is given and its edge table exists, uses BTREE-indexed
        lookup for O(log n) performance and includes ``amount_sum``/``amount_max``
        per counterparty.  Falls back to full points scan otherwise.

        If *pattern_id* is provided, each counterparty is enriched with ``is_anomaly``
        and ``delta_rank_pct`` from that pattern's geometry.

        ``timestamp_cutoff`` restricts the lookup to edges with
        ``timestamp <= timestamp_cutoff``. **Edge-table fast path only.** The
        points-scan fallback has no timestamp column available and cannot
        honor the cutoff, so passing it without an edge-table-eligible
        configuration raises ``GDSNavigationError`` to fail loudly instead of
        silently returning unfiltered results.

        Returns ``{primary_key, outgoing, incoming, summary}``.
        """
        # Fast path: edge table BTREE lookup
        if (
            pattern_id
            and use_edge_table
            and self._storage.has_edge_table(pattern_id)
        ):
            return self._find_counterparties_via_edges(
                primary_key, line_id, pattern_id, top_n,
                timestamp_cutoff=timestamp_cutoff,
            )

        # Cutoff is meaningless on the points-scan fallback — fail loudly.
        if timestamp_cutoff is not None:
            raise GDSNavigationError(
                "find_counterparties: timestamp_cutoff is only supported on "
                "the edge-table fast path. Provide a pattern_id whose edge "
                "table exists and leave use_edge_table=True, or omit "
                "timestamp_cutoff."
            )

        sphere = self._storage.read_sphere()
        if line_id not in sphere.lines:
            raise GDSNavigationError(
                f"Line '{line_id}' not found. Available: {sorted(sphere.lines)}"
            )
        line = sphere.lines[line_id]
        version = self._manifest.line_version(line_id) or line.current_version()
        needed_cols = {from_col, to_col, "primary_key"}
        points = self._storage.read_points(line_id, version)
        available = set(points.schema.names)
        for col in (from_col, to_col):
            if col not in available:
                raise GDSNavigationError(
                    f"Column '{col}' not found in line '{line_id}'. "
                    f"Available: {sorted(available)}"
                )
        points = points.select([c for c in needed_cols if c in available])

        # Outgoing: rows where from_col == primary_key → group by to_col
        out_mask = pc.equal(points[from_col], primary_key)
        out_rows = points.filter(out_mask)
        out_grouped = out_rows.group_by(to_col).aggregate(
            [("primary_key", "count")]
        )
        out_keys_col = out_grouped[to_col].to_pylist()
        out_counts = out_grouped["primary_key_count"].to_pylist()
        out_pairs = sorted(
            zip(out_keys_col, out_counts, strict=False), key=lambda x: x[1], reverse=True
        )[:top_n]

        # Incoming: rows where to_col == primary_key → group by from_col
        in_mask = pc.equal(points[to_col], primary_key)
        in_rows = points.filter(in_mask)
        in_grouped = in_rows.group_by(from_col).aggregate(
            [("primary_key", "count")]
        )
        in_keys_col = in_grouped[from_col].to_pylist()
        in_counts = in_grouped["primary_key_count"].to_pylist()
        in_pairs = sorted(
            zip(in_keys_col, in_counts, strict=False), key=lambda x: x[1], reverse=True
        )[:top_n]

        # Anomaly enrichment — handles both direct and composite-key patterns
        geo_lookup: dict[str, dict[str, Any]] = {}
        _enrichment_warning: str | None = None
        if pattern_id:
            if pattern_id not in sphere.patterns:
                raise GDSNavigationError(
                    f"Pattern '{pattern_id}' not found. "
                    f"Available: {sorted(sphere.patterns)}"
                )
            geo_version = self._resolve_version(pattern_id)
            all_cp_keys = {k for k, _ in out_pairs} | {k for k, _ in in_pairs}

            # Detect composite-key patterns by sampling one geometry PK
            _composite_map: dict[str, str] = {}
            _geo_sample = self._storage.read_geometry(
                pattern_id, geo_version,
                columns=["primary_key"], sample_size=1,
            )
            if _geo_sample.num_rows > 0:
                _sample_pk = str(_geo_sample["primary_key"][0].as_py())
                for sep in ("→", "|"):
                    if sep in _sample_pk:
                        for cpk in all_cp_keys:
                            _composite_map[f"{primary_key}{sep}{cpk}"] = cpk
                        break

            geo = self._storage.read_geometry(
                pattern_id, geo_version,
                columns=["primary_key", "is_anomaly", "delta_rank_pct"],
            )
            geo_pks = geo["primary_key"].to_pylist()
            geo_anom = geo["is_anomaly"].to_pylist()
            geo_rank = geo["delta_rank_pct"].to_pylist()
            for pk, anom, rank in zip(geo_pks, geo_anom, geo_rank, strict=False):
                _data = {
                    "is_anomaly": bool(anom),
                    "delta_rank_pct": (
                        round(float(rank), 2) if rank is not None else None
                    ),
                }
                if pk in all_cp_keys:
                    geo_lookup[pk] = _data
                elif pk in _composite_map:
                    geo_lookup[_composite_map[pk]] = _data

            if all_cp_keys and not geo_lookup:
                _enrichment_warning = (
                    f"Enrichment returned 0 matches for '{pattern_id}'. "
                    f"Pattern may use composite keys that don't match "
                    f"counterparty keys directly."
                )

        def _build_entry(key: str, tx_count: int) -> dict[str, Any]:
            entry: dict[str, Any] = {"key": key, "tx_count": tx_count}
            if pattern_id and key in geo_lookup:
                entry.update(geo_lookup[key])
            return entry

        outgoing = [_build_entry(k, c) for k, c in out_pairs]
        incoming = [_build_entry(k, c) for k, c in in_pairs]

        anomalous_out = sum(1 for e in outgoing if e.get("is_anomaly"))
        anomalous_in = sum(1 for e in incoming if e.get("is_anomaly"))

        result: dict[str, Any] = {
            "primary_key": primary_key,
            "line_id": line_id,
            "outgoing": outgoing,
            "incoming": incoming,
            "summary": {
                "total_outgoing": len(outgoing),
                "total_incoming": len(incoming),
                "anomalous_outgoing": anomalous_out,
                "anomalous_incoming": anomalous_in,
            },
        }
        if _enrichment_warning:
            result["enrichment_warning"] = _enrichment_warning
        return result

    def entity_flow(
        self,
        primary_key: str,
        pattern_id: str,
        top_n: int = 20,
        *,
        timestamp_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Net flow analysis per counterparty via edge table.

        Two edge lookups (outgoing + incoming), sum amounts, compute
        per-counterparty net flow.

        When ``timestamp_cutoff`` is set (Unix seconds as float), only edges
        with ``timestamp <= timestamp_cutoff`` are considered — used for
        as-of evaluation of flow history up to a given point in time.

        Returns ``{outgoing_total, incoming_total, net_flow, flow_direction,
        counterparties: [{key, net_flow, direction}]}``.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "entity_flow requires an edge table."
            )
        adj = self._storage.get_adjacency(pattern_id)
        fwd_edges = adj.neighbors_out(primary_key, ts_to=timestamp_cutoff)
        rev_edges = adj.neighbors_in(primary_key, ts_to=timestamp_cutoff)

        from collections import defaultdict
        out_by_cp: dict[str, float] = defaultdict(float)
        for target, _ts, amount, _ek in fwd_edges:
            out_by_cp[target] += amount

        in_by_cp: dict[str, float] = defaultdict(float)
        for source, _ts, amount, _ek in rev_edges:
            in_by_cp[source] += amount

        outgoing_total = sum(out_by_cp.values())
        incoming_total = sum(in_by_cp.values())
        net_flow = outgoing_total - incoming_total

        # Per-counterparty net flow
        all_cps = set(out_by_cp) | set(in_by_cp)
        cp_flows: list[dict[str, Any]] = []
        for cp in all_cps:
            cp_out = out_by_cp.get(cp, 0.0)
            cp_in = in_by_cp.get(cp, 0.0)
            cp_net = cp_out - cp_in
            cp_flows.append({
                "key": cp,
                "net_flow": round(cp_net, 2),
                "direction": "outgoing" if cp_net > 0 else "incoming" if cp_net < 0 else "balanced",
            })
        cp_flows.sort(key=lambda x: abs(x["net_flow"]), reverse=True)
        cp_flows = cp_flows[:top_n]

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "outgoing_total": round(outgoing_total, 2),
            "incoming_total": round(incoming_total, 2),
            "net_flow": round(net_flow, 2),
            "flow_direction": "outgoing" if net_flow > 0 else "incoming" if net_flow < 0 else "balanced",
            "counterparties": cp_flows,
        }

    def contagion_score(
        self,
        primary_key: str,
        pattern_id: str,
        *,
        timestamp_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Score how many of an entity's counterparties are anomalous.

        Edge lookup for counterparties, batch geometry check on the anchor
        pattern.  Score = anomalous_counterparties / total_counterparties.

        When ``timestamp_cutoff`` is set (Unix seconds as float), only edges
        with ``timestamp <= timestamp_cutoff`` are considered — used for
        as-of evaluation, reproducing the state of the graph at a given
        point in time.

        Returns ``{score: 0.0-1.0, total_counterparties,
        anomalous_counterparties, interpretation}``.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "contagion_score requires an edge table."
            )
        adj = self._storage.get_adjacency(pattern_id)
        fwd_edges = adj.neighbors_out(primary_key, ts_to=timestamp_cutoff)
        rev_edges = adj.neighbors_in(primary_key, ts_to=timestamp_cutoff)

        cp_keys: set[str] = {e[0] for e in fwd_edges} | {e[0] for e in rev_edges}
        cp_keys.discard(primary_key)

        total = len(cp_keys)
        if total == 0:
            return {
                "primary_key": primary_key,
                "pattern_id": pattern_id,
                "score": 0.0,
                "total_counterparties": 0,
                "anomalous_counterparties": 0,
                "interpretation": "No counterparties found.",
            }

        scoring_pattern = (
            self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        )
        try:
            batch = self.check_anomaly_batch(
                scoring_pattern, list(cp_keys), max_keys=500,
            )
            anomalous = batch["anomalous_count"]
        except GDSNavigationError:
            anomalous = 0
        score = round(anomalous / total, 4)

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "score": score,
            "total_counterparties": total,
            "anomalous_counterparties": anomalous,
            "interpretation": (
                f"{anomalous}/{total} counterparties are anomalous "
                f"(contagion score {score:.2f})."
            ),
        }

    def contagion_score_batch(
        self,
        primary_keys: list[str],
        pattern_id: str,
        max_keys: int = 200,
        *,
        timestamp_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Contagion score for multiple entities.

        When ``timestamp_cutoff`` is set, forwards it to each per-entity
        contagion_score call so that only edges with
        ``timestamp < timestamp_cutoff`` are considered.

        Returns per-entity scores plus a summary with mean/max.
        """
        keys = primary_keys[:max_keys]
        results: list[dict[str, Any]] = []
        for pk in keys:
            results.append(
                self.contagion_score(
                    pk, pattern_id, timestamp_cutoff=timestamp_cutoff,
                )
            )

        scores = [r["score"] for r in results]
        mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        return {
            "pattern_id": pattern_id,
            "total": len(results),
            "results": results,
            "summary": {
                "mean_score": mean_score,
                "max_score": max_score,
                "high_contagion_count": sum(1 for s in scores if s >= 0.5),
            },
        }

    def degree_velocity(
        self,
        primary_key: str,
        pattern_id: str,
        n_buckets: int = 4,
        *,
        timestamp_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Temporal connection velocity via edge table.

        Buckets edges by timestamp, counts unique counterparties per bucket.
        Velocity = degree in last bucket / degree in first bucket.

        When ``timestamp_cutoff`` is set (Unix seconds as float), only edges
        with ``timestamp <= timestamp_cutoff`` are considered. Buckets are
        computed from the filtered edge set, so the last bucket endpoint is
        naturally <= cutoff.

        Returns ``{buckets: [{period, out_degree, in_degree}],
        velocity_out, velocity_in, interpretation}``.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "degree_velocity requires an edge table."
            )
        fwd = self._storage.read_edges(
            pattern_id, from_keys=[primary_key], timestamp_to=timestamp_cutoff,
        )
        rev = self._storage.read_edges(
            pattern_id, to_keys=[primary_key], timestamp_to=timestamp_cutoff,
        )

        # Collect all timestamps
        fwd_ts = fwd["timestamp"].to_pylist() if fwd.num_rows > 0 else []
        rev_ts = rev["timestamp"].to_pylist() if rev.num_rows > 0 else []
        fwd_to = fwd["to_key"].to_pylist() if fwd.num_rows > 0 else []
        rev_from = rev["from_key"].to_pylist() if rev.num_rows > 0 else []

        all_ts = fwd_ts + rev_ts
        # Degenerate: no edges, all timestamps 0, or no temporal spread
        if not all_ts or min(all_ts) == max(all_ts):
            return {
                "primary_key": primary_key,
                "pattern_id": pattern_id,
                "buckets": [],
                "velocity_out": None,
                "velocity_in": None,
                "warning": "Insufficient temporal spread to compute velocity "
                "(no edges, uniform timestamps, or all zeros).",
            }

        ts_min = min(all_ts)
        ts_max = max(all_ts)
        bucket_width = (ts_max - ts_min) / n_buckets

        def _bucket_idx(ts: float) -> int:
            idx = int((ts - ts_min) / bucket_width)
            return min(idx, n_buckets - 1)

        # Count unique counterparties per bucket
        out_buckets: list[set[str]] = [set() for _ in range(n_buckets)]
        for ts, to_k in zip(fwd_ts, fwd_to, strict=False):
            out_buckets[_bucket_idx(ts)].add(to_k)

        in_buckets: list[set[str]] = [set() for _ in range(n_buckets)]
        for ts, from_k in zip(rev_ts, rev_from, strict=False):
            in_buckets[_bucket_idx(ts)].add(from_k)

        buckets = []
        for i in range(n_buckets):
            period_start = ts_min + i * bucket_width
            period_end = period_start + bucket_width
            buckets.append({
                "period": f"{period_start:.0f}-{period_end:.0f}",
                "out_degree": len(out_buckets[i]),
                "in_degree": len(in_buckets[i]),
            })

        first_out = len(out_buckets[0]) or 1
        last_out = len(out_buckets[-1])
        first_in = len(in_buckets[0]) or 1
        last_in = len(in_buckets[-1])
        velocity_out = round(last_out / first_out, 4)
        velocity_in = round(last_in / first_in, 4)

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "buckets": buckets,
            "velocity_out": velocity_out,
            "velocity_in": velocity_in,
            "interpretation": (
                f"Out-degree velocity {velocity_out:.2f} "
                f"({'accelerating' if velocity_out > 1 else 'decelerating' if velocity_out < 1 else 'stable'}), "
                f"in-degree velocity {velocity_in:.2f} "
                f"({'accelerating' if velocity_in > 1 else 'decelerating' if velocity_in < 1 else 'stable'})."
            ),
        }

    def investigation_coverage(
        self,
        primary_key: str,
        pattern_id: str,
        explored_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """Agent guidance: how much of an entity's edge neighborhood has been explored.

        Looks up all counterparties via edge table, splits into explored vs
        unexplored based on *explored_keys*, and runs a batch anomaly check
        on the unexplored set.

        Returns ``{total_edges, explored, unexplored, unexplored_anomalous,
        coverage_pct, summary}``.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "investigation_coverage requires an edge table."
            )
        if explored_keys is None:
            explored_keys = set()

        fwd = self._storage.read_edges(pattern_id, from_keys=[primary_key])
        rev = self._storage.read_edges(pattern_id, to_keys=[primary_key])

        all_cp: set[str] = set()
        if fwd.num_rows > 0:
            all_cp.update(fwd["to_key"].to_pylist())
        if rev.num_rows > 0:
            all_cp.update(rev["from_key"].to_pylist())
        all_cp.discard(primary_key)

        explored = all_cp & explored_keys
        unexplored = all_cp - explored_keys
        total = len(all_cp)
        coverage_pct = round(len(explored) / total, 4) if total > 0 else None

        # Batch anomaly check on unexplored counterparties
        unexplored_anomalous: list[dict[str, Any]] = []
        if unexplored:
            scoring_pattern = (
                self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
            )
            try:
                batch = self.check_anomaly_batch(
                    scoring_pattern, list(unexplored), max_keys=500,
                )
                unexplored_anomalous = [
                    r for r in batch["results"] if r["is_anomaly"]
                ]
            except GDSNavigationError:
                pass  # no geometry — skip enrichment

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "total_edges": total,
            "explored": len(explored),
            "unexplored": len(unexplored),
            "unexplored_anomalous": unexplored_anomalous,
            "coverage_pct": coverage_pct,
            "summary": (
                "No counterparties found."
                if total == 0
                else (
                    f"{len(explored)}/{total} counterparties explored "
                    f"({coverage_pct:.0%} coverage). "
                    f"{len(unexplored_anomalous)} unexplored anomalous entities."
                )
            ),
        }

    def propagate_influence(
        self,
        seed_keys: list[str],
        pattern_id: str,
        max_depth: int = 3,
        decay: float = 0.7,
        min_threshold: float = 0.001,
        max_affected: int = 10_000,
        *,
        timestamp_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """BFS influence propagation from seed entities with geometric decay.

        At each hop, influence_score = parent_score * decay * geometric_coherence.
        Stops expanding when score falls below *min_threshold* or when
        *max_affected* entities have been reached.

        When ``timestamp_cutoff`` is set (Unix seconds as float), the BFS only
        follows edges with ``timestamp <= timestamp_cutoff``. Use this to
        reconstruct what influence propagation would have surfaced at a prior
        point in time — e.g. "what was reachable from this seed on the day
        of the incident?".

        Returns ``{seeds, affected_entities, summary}``.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "propagate_influence requires an edge table."
            )
        scoring_pattern = (
            self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        )
        seed_set = set(seed_keys)

        # Build weighted adjacency directly from edges (not _build_adjacency,
        # which deduplicates to one edge per pair — influence needs tx_count).
        # adj[key] = {neighbor: tx_count}
        adj: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        expanded_keys: set[str] = set()

        seen_edges: set[str] = set()  # event_key dedup across fwd/rev reads

        def _expand_neighbors(keys: set[str]) -> None:
            """Read edges for keys and aggregate tx_count per neighbor."""
            key_list = list(keys)
            fwd = self._storage.read_edges(
                pattern_id, from_keys=key_list, timestamp_to=timestamp_cutoff,
            )
            rev = self._storage.read_edges(
                pattern_id, to_keys=key_list, timestamp_to=timestamp_cutoff,
            )
            for tbl in (fwd, rev):
                from_arr = tbl["from_key"].to_pylist()
                to_arr = tbl["to_key"].to_pylist()
                ek_arr = tbl["event_key"].to_pylist()
                for f, t, ek in zip(from_arr, to_arr, ek_arr, strict=False):
                    if f == t or ek in seen_edges:
                        continue
                    seen_edges.add(ek)
                    adj[f][t] += 1
                    adj[t][f] += 1

        # BFS: frontier = [(key, score, depth)]
        frontier: list[tuple[str, float, int]] = [(k, 1.0, 0) for k in seed_keys]
        visited: dict[str, tuple[float, int, int]] = {}  # key → (score, depth, tx_count)

        while frontier:
            if len(visited) >= max_affected:
                break
            next_frontier: list[tuple[str, float, int]] = []
            # Expand edges for new tips
            tips = {k for k, _, _ in frontier} - expanded_keys
            if tips:
                _expand_neighbors(tips)
                expanded_keys |= tips
                # Prefetch deltas for scoring
                neighbor_keys = set()
                for tip in tips:
                    neighbor_keys |= set(adj.get(tip, {}).keys())
                self._prefetch_deltas(neighbor_keys | tips, scoring_pattern)

            for key, score, depth in frontier:
                if depth >= max_depth:
                    continue
                for neighbor, tx_count in adj.get(key, {}).items():
                    if neighbor in seed_set:
                        continue
                    coherence = self._score_hop(key, neighbor, scoring_pattern, "geometric")
                    # Clamp to prevent sign flips from negative cosine
                    coherence = max(coherence, 0.01)
                    # Weight by log(tx_count) — more transactions = stronger influence
                    tx_weight = 1.0 + float(np.log1p(tx_count - 1)) if tx_count > 1 else 1.0
                    new_score = score * decay * coherence * tx_weight
                    if new_score < min_threshold:
                        continue
                    # Keep highest score if revisited
                    if neighbor in visited and visited[neighbor][0] >= new_score:
                        continue
                    visited[neighbor] = (new_score, depth + 1, tx_count)
                    next_frontier.append((neighbor, new_score, depth + 1))

            frontier = next_frontier

        # Anomaly enrichment
        affected_keys = list(visited.keys())
        anomaly_map: dict[str, bool] = {}
        if affected_keys:
            try:
                batch = self.check_anomaly_batch(
                    scoring_pattern, affected_keys, max_keys=500,
                )
                for r in batch["results"]:
                    anomaly_map[r["primary_key"]] = r["is_anomaly"]
            except GDSNavigationError:
                pass

        affected = sorted(
            [
                {
                    "key": k,
                    "depth": d,
                    "influence_score": round(s, 4),
                    "tx_count": tc,
                    "is_anomaly": anomaly_map.get(k, False),
                }
                for k, (s, d, tc) in visited.items()
            ],
            key=lambda x: x["influence_score"],
            reverse=True,
        )

        max_depth_reached = max((d for _, (_, d, _) in visited.items()), default=0)
        anomalous_affected = sum(1 for a in affected if a["is_anomaly"])

        return {
            "seeds": seed_keys,
            "pattern_id": pattern_id,
            "affected_entities": affected,
            "summary": {
                "total_affected": len(affected),
                "max_depth_reached": max_depth_reached,
                "anomalous_affected": anomalous_affected,
            },
        }

    def cluster_bridges(
        self,
        pattern_id: str,
        n_clusters: int = 5,
        top_n_bridges: int = 10,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        """Find entities that bridge geometric clusters via edge table.

        1. Run π8 clustering on the anchor pattern to get cluster membership.
        2. Read full edge table to find cross-cluster edges.
        3. Identify top bridge entities connecting different clusters.

        Warning: reads the full edge table into memory. When *sample_size* is set,
        only sampled entities appear in the cluster map — bridge counts will be
        systematically underreported for non-sampled entities.

        Returns ``{clusters, bridges, summary}``.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "cluster_bridges requires an edge table."
            )
        scoring_pattern = (
            self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        )

        # Step 1: cluster the anchor pattern — get full membership
        clusters = self.π8_attract_cluster(
            scoring_pattern, n_clusters=n_clusters,
            top_n=None, sample_size=sample_size,
        )

        # Build entity → cluster_id map
        entity_cluster: dict[str, int] = {}
        for c in clusters:
            cid = c["cluster_id"]
            for key in c.get("member_keys", []):
                entity_cluster[key] = cid

        # Step 2: read edge table — vectorized with pandas for speed on millions of rows
        import pandas as pd

        edges = self._storage.read_edges(
            pattern_id, columns=["from_key", "to_key"],
        )
        df = pd.DataFrame({
            "f": edges["from_key"].to_pandas(),
            "t": edges["to_key"].to_pandas(),
        })
        df = df[df["f"] != df["t"]]  # drop self-loops
        df["c_f"] = df["f"].map(entity_cluster)
        df["c_t"] = df["t"].map(entity_cluster)
        cross = df.dropna(subset=["c_f", "c_t"])
        cross = cross[cross["c_f"] != cross["c_t"]]

        # Normalize cluster pairs (min, max) and count edges
        cross["c_lo"] = cross[["c_f", "c_t"]].min(axis=1).astype(int)
        cross["c_hi"] = cross[["c_f", "c_t"]].max(axis=1).astype(int)

        bridge_edges: dict[tuple[int, int], int] = {}
        if not cross.empty:
            pair_counts = cross.groupby(["c_lo", "c_hi"]).size()
            bridge_edges = {(int(a), int(b)): int(c) for (a, b), c in pair_counts.items()}

        # Count bridge appearances per entity
        bridge_entity_count: dict[str, int] = defaultdict(int)
        if not cross.empty:
            for col in ("f", "t"):
                for entity, cnt in cross[col].value_counts().items():
                    bridge_entity_count[entity] += int(cnt)

        # Step 3: batch anomaly check on top bridge entities
        top_bridge_entities = sorted(
            bridge_entity_count.items(), key=lambda x: x[1], reverse=True,
        )
        bridge_keys = [k for k, _ in top_bridge_entities[:200]]
        anomaly_map: dict[str, bool] = {}
        if bridge_keys:
            try:
                batch = self.check_anomaly_batch(
                    scoring_pattern, bridge_keys, max_keys=200,
                )
                for r in batch["results"]:
                    anomaly_map[r["primary_key"]] = r["is_anomaly"]
            except GDSNavigationError:
                pass

        # Format bridges
        bridges_out = sorted(
            [
                {
                    "cluster_a": a,
                    "cluster_b": b,
                    "edge_count": cnt,
                    "bridge_entities": [
                        {
                            "key": k,
                            "is_anomaly": anomaly_map.get(k, False),
                        }
                        for k, _ in sorted(
                            [(e, c) for e, c in bridge_entity_count.items()
                             if entity_cluster.get(e) in (a, b)],
                            key=lambda x: x[1], reverse=True,
                        )[:5]
                    ],
                }
                for (a, b), cnt in bridge_edges.items()
            ],
            key=lambda x: x["edge_count"],
            reverse=True,
        )[:top_n_bridges]

        # Format clusters
        clusters_out = [
            {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "anomaly_rate": c.get("anomaly_rate", 0.0),
            }
            for c in clusters
        ]

        total_bridge_entities = len(bridge_entity_count)
        top_bridge = top_bridge_entities[0][0] if top_bridge_entities else None

        return {
            "pattern_id": pattern_id,
            "clusters": clusters_out,
            "bridges": bridges_out,
            "summary": {
                "total_clusters": len(clusters),
                "total_bridge_edges": sum(bridge_edges.values()),
                "total_bridge_entities": total_bridge_entities,
                "top_bridge_entity": top_bridge,
            },
        }

    def anomalous_edges(
        self,
        from_key: str,
        to_key: str,
        pattern_id: str,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Find edges between two entities enriched with event-level geometry.

        Unlike path/chain tools which resolve anchor pattern geometry, this reads
        geometry from the EVENT pattern itself — ``event_key`` is the primary key
        in event geometry.  Sorts by ``delta_norm`` descending.

        Returns ``{from_key, to_key, edges, summary}``.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "anomalous_edges requires an edge table."
            )
        adj = self._storage.get_adjacency(pattern_id)
        # A→B edges
        fwd_edges = [
            (from_key, tgt, ts, amt, ek)
            for tgt, ts, amt, ek in adj.neighbors_out(from_key)
            if tgt == to_key
        ]
        # B→A edges
        rev_edges = [
            (to_key, tgt, ts, amt, ek)
            for tgt, ts, amt, ek in adj.neighbors_out(to_key)
            if tgt == from_key
        ]

        all_edges: list[dict[str, Any]] = []
        seen_event_keys: set[str] = set()
        for f, t, ts, amt, ek in fwd_edges + rev_edges:
            if ek in seen_event_keys:
                continue
            seen_event_keys.add(ek)
            all_edges.append({
                "event_key": ek,
                "from_key": f,
                "to_key": t,
                "amount": round(float(amt), 2),
                "timestamp": float(ts),
            })

        if not all_edges:
            return {
                "from_key": from_key,
                "to_key": to_key,
                "pattern_id": pattern_id,
                "edges": [],
                "summary": {
                    "total_edges": 0,
                    "returned": 0,
                    "anomalous": 0,
                    "max_delta_norm": 0.0,
                },
            }

        # Enrich with EVENT geometry (not anchor!)
        event_keys = [e["event_key"] for e in all_edges]
        try:
            version = self._resolve_version(pattern_id)
            geo = self._storage.read_geometry(
                pattern_id, version,
                point_keys=event_keys,
                columns=["primary_key", "delta_norm", "is_anomaly", "delta_rank_pct"],
            )
            geo_map: dict[str, dict[str, Any]] = {}
            for i in range(geo.num_rows):
                pk = geo["primary_key"][i].as_py()
                geo_map[pk] = {
                    "delta_norm": round(float(geo["delta_norm"][i].as_py()), 4),
                    "is_anomaly": bool(geo["is_anomaly"][i].as_py()),
                    "delta_rank_pct": round(float(geo["delta_rank_pct"][i].as_py()), 2),
                }
        except GDSNavigationError:
            geo_map = {}

        for edge in all_edges:
            geo_data = geo_map.get(edge["event_key"], {})
            edge["delta_norm"] = geo_data.get("delta_norm", 0.0)
            edge["is_anomaly"] = geo_data.get("is_anomaly", False)
            edge["delta_rank_pct"] = geo_data.get("delta_rank_pct", 0.0)

        # Sort by delta_norm desc, cap at top_n
        all_edges.sort(key=lambda e: e["delta_norm"], reverse=True)
        total = len(all_edges)
        anomalous = sum(1 for e in all_edges if e["is_anomaly"])
        max_dn = all_edges[0]["delta_norm"] if all_edges else 0.0
        returned = all_edges[:top_n]

        return {
            "from_key": from_key,
            "to_key": to_key,
            "pattern_id": pattern_id,
            "edges": returned,
            "summary": {
                "total_edges": total,
                "returned": len(returned),
                "anomalous": anomalous,
                "max_delta_norm": round(max_dn, 4),
            },
        }

    def _resolve_edge_pattern_for_anchor(
        self, anchor_pattern_id: str,
    ) -> str | None:
        """Find an event pattern with edge table whose relations cover this anchor.

        Inverse of ``_resolve_anchor_pattern_for_scoring``: given an anchor pattern,
        find the event pattern whose edge table connects entities of this anchor.
        Picks the event pattern with the most relations to this anchor's entity line
        (best graph coverage). Returns None when no such event pattern exists.
        """
        if not hasattr(self, "_edge_pattern_cache"):
            self._edge_pattern_cache: dict[str, str | None] = {}
        if anchor_pattern_id in self._edge_pattern_cache:
            return self._edge_pattern_cache[anchor_pattern_id]

        sphere = self._storage.read_sphere()
        anchor_pat = sphere.patterns.get(anchor_pattern_id)
        if anchor_pat is None or anchor_pat.pattern_type != "anchor":
            self._edge_pattern_cache[anchor_pattern_id] = None
            return None

        anchor_line = anchor_pat.entity_line_id
        best: tuple[str, int] | None = None
        for pid, pat in sphere.patterns.items():
            if pat.pattern_type != "event":
                continue
            if not self._storage.has_edge_table(pid):
                continue
            relevance = sum(
                1 for rel in pat.relations if rel.line_id == anchor_line
            )
            if relevance == 0:
                continue
            if best is None or relevance > best[1]:
                best = (pid, relevance)

        result = best[0] if best else None
        self._edge_pattern_cache[anchor_pattern_id] = result
        return result

    def _pair_count_for_pattern(
        self, graph_pid: str, version: int,
    ) -> dict[tuple[str, str], int]:
        """Delegate to AdjacencyIndex.pair_counts() — shared with motif ranking."""
        return self._storage.get_adjacency(graph_pid).pair_counts()

    def _pair_count_direct(
        self, graph_pid: str,
    ) -> dict[tuple[str, str], int]:
        """Compute pair counts from edge table via PyArrow groupby.

        Reads only from_key/to_key (2 columns), bypassing AdjacencyIndex.
        10–20× faster than get_adjacency().pair_counts() on large edge tables
        because it skips the 5-column full-table to_pylist() in AdjacencyIndex.
        """
        tbl = self._storage.read_edges(graph_pid, columns=["from_key", "to_key"])
        if tbl.num_rows == 0:
            return {}
        agg = tbl.group_by(["from_key", "to_key"]).aggregate(
            [("from_key", "count")]
        )
        return {
            (agg["from_key"][i].as_py(), agg["to_key"][i].as_py()): int(
                agg["from_key_count"][i].as_py()
            )
            for i in range(agg.num_rows)
        }

    _EDGE_POTENTIAL_PAIR_CAP = 1000

    def edge_potential(
        self,
        from_key: str,
        to_key: str,
        pattern_id: str,
    ) -> dict[str, Any]:
        """Geometric anomaly score for the edge (from_key → to_key).

        Formula: ||δ_from − δ_to||₂ × (1 / min(pair_tx_count, 1000)).

        High score = endpoints are structurally distant AND the pair is rare.
        Empirically (AUROC 0.918 on AML HI-small is_laundering), the rarity
        prior carries most of the signal — recurring high-volume pairs between
        two "extreme" accounts are usually legitimate (e.g. corporate payroll),
        while a single transaction between two geometrically divergent accounts
        is the classic layering signature.

        Raises GDSNavigationError when either endpoint is missing from the
        anchor pattern's geometry.
        """
        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )
        pattern = sphere.patterns[pattern_id]
        graph_pid: str | None = None
        if pattern.pattern_type == "anchor":
            graph_pid = self._resolve_edge_pattern_for_anchor(pattern_id)
        elif pattern.pattern_type == "event" and self._storage.has_edge_table(pattern_id):
            graph_pid = pattern_id
        if graph_pid is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no graph companion; edge_potential "
                f"requires an event pattern with an edge table."
            )

        version = self._resolve_version(pattern_id)
        # Load endpoint delta vectors — fail loudly if either missing.
        def _load_delta(key: str) -> np.ndarray:
            geo = self._storage.read_geometry(
                pattern_id, version, primary_key=key, columns=["primary_key", "delta"],
            )
            if geo.num_rows == 0:
                raise GDSNavigationError(
                    f"Entity '{key}' not found in pattern '{pattern_id}' geometry."
                )
            return np.asarray(geo["delta"][0].as_py(), dtype=np.float64)

        d_from = _load_delta(from_key)
        d_to = _load_delta(to_key)
        distance = float(np.linalg.norm(d_from - d_to))

        counts = self._pair_count_for_pattern(graph_pid, version)
        if from_key == to_key:
            # Self-loop: use the stored count once; do not double by adding the
            # reverse direction (which is the same pair in this dict).
            pair_tx_count = counts.get((from_key, to_key), 0)
        else:
            pair_tx_count = counts.get((from_key, to_key), 0) + counts.get((to_key, from_key), 0)
        # Cap the weight denominator so a single 10 000-tx pair doesn't underflow
        # to 0; raw count is still reported separately so the agent sees the cap.
        effective_count = max(1, min(pair_tx_count, self._EDGE_POTENTIAL_PAIR_CAP))
        score = round(distance * (1.0 / effective_count), 6)

        # Pattern-local normalisation: compute rank_pct + is_high_potential
        # against the full ranking. Uses attract_edge_potential's cache, so
        # only the first call pays the full-population scoring cost.
        score_rank_pct: float | None = None
        is_high_potential: bool | None = None
        try:
            full_ranking = self.attract_edge_potential(
                pattern_id, top_n=10**9, min_pair_count=1,
            )
            if full_ranking:
                better = sum(1 for r in full_ranking if r["score"] > score)
                n = len(full_ranking)
                score_rank_pct = round(100.0 * (n - better) / n, 2)
                p95_idx = max(0, int(n * 0.05) - 1)
                p95_threshold = full_ranking[p95_idx]["score"] if p95_idx < n else full_ranking[0]["score"]
                is_high_potential = bool(score >= p95_threshold)
        except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
            pass

        return {
            "from_key": from_key,
            "to_key": to_key,
            "pattern_id": pattern_id,
            "score": score,
            "delta_distance": round(distance, 4),
            "pair_tx_count": pair_tx_count,
            "effective_weight": round(1.0 / effective_count, 6),
            "score_rank_pct": score_rank_pct,
            "is_high_potential": is_high_potential,
            "interpretation": (
                f"Edge score {round(score, 4)} = distance {round(distance, 4)} "
                f"× weight {round(1.0 / effective_count, 4)} "
                f"(pair_tx_count={pair_tx_count})."
            ),
        }

    def attract_edge_potential(
        self,
        pattern_id: str,
        top_n: int = 10,
        from_key: str | None = None,
        to_key: str | None = None,
        min_pair_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Rank edges by geometric edge potential, highest first.

        Scans the companion event pattern's edge table, vectorised by a single
        batch geometry read of all endpoint entities, then ranks by
        ``||δ_from − δ_to||₂ × (1 / min(pair_tx_count, 1000))``. Use ``from_key``
        or ``to_key`` to scope the ranking to edges touching a specific entity —
        useful for per-entity investigation within ``trace_root_cause``.

        Each result also carries ``score_rank_pct`` (percentile within the
        pattern, 0–100) and ``is_high_potential`` (True when score ≥ p95 of
        the scored population). Results are cached per
        ``(pattern_id, version, from_key, to_key, min_pair_count)`` at navigator
        instance level — repeat calls with the same params return in O(1).

        ``min_pair_count`` filters out pairs appearing fewer times than the
        threshold (default 1 — keeps everything). Raise to 3+ on very large
        edge tables where one-off pairs dominate.
        """
        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"Pattern '{pattern_id}' not found in sphere.")
        pattern = sphere.patterns[pattern_id]
        graph_pid: str | None = None
        if pattern.pattern_type == "anchor":
            graph_pid = self._resolve_edge_pattern_for_anchor(pattern_id)
        elif pattern.pattern_type == "event" and self._storage.has_edge_table(pattern_id):
            graph_pid = pattern_id
        if graph_pid is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no graph companion; attract_edge_potential "
                f"requires an event pattern with an edge table."
            )

        version = self._resolve_version(pattern_id)

        # ---- Base ranking cache (the whole pattern, min_pair_count=1) ----
        # Computed once per (pattern_id, version) — all filtered calls reuse it.
        # Each entry carries `score_rank_pct_global` and `is_high_potential`
        # (against pattern-wide p95). Filter queries add `score_rank_pct_in_filter`
        # relative to their own subset without recomputing the O(pairs) scoring loop.
        if not hasattr(self, "_attract_edge_base_cache"):
            self._attract_edge_base_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        base_key = (pattern_id, version)
        base_sorted = self._attract_edge_base_cache.get(base_key)

        if base_sorted is None:
            # For event patterns: bail BEFORE the 7.5 M-edge groupby when the
            # cached edge_stats reveal that the unique-endpoint upper bound is
            # < 1 % of the geometry rowcount.  In that regime the endpoints
            # belong to a different entity type (e.g. zone IDs in a trip-edge
            # pattern whose geometry holds trips) and no key will ever match
            # the geometry primary keys.  Reading the build-time JSON cache
            # is O(1); falling through pays for a 2-column scan of the full
            # edge table.  Skip the live-scan branch of edge_table_stats —
            # if the cache is absent, defer to the late-stage guard below.
            if pattern.pattern_type == "event":
                edge_stats = self._storage.edge_stats_cached(graph_pid)
                if (
                    edge_stats is not None
                    and "unique_from" in edge_stats
                    and "unique_to" in edge_stats
                ):
                    n_endpoints_upper = (
                        int(edge_stats["unique_from"]) + int(edge_stats["unique_to"])
                    )
                    # Use pattern.population_size (already loaded from sphere.json)
                    # as the geometry-row denominator — avoids opening the Lance
                    # dataset cold just to count rows.  population_size is
                    # geometry rowcount at calibration time and matches the
                    # current dataset within the freshness window we accept here.
                    geo_count = int(pattern.population_size or 0)
                    if (
                        geo_count > 0
                        and n_endpoints_upper / geo_count < 0.01
                    ):
                        self._attract_edge_base_cache[base_key] = []
                        base_sorted = []

            if base_sorted is None:
                # Use direct PyArrow groupby (reads only 2 columns) instead of
                # AdjacencyIndex.from_lance() which reads 5 columns and calls
                # to_pylist() on the full edge table — 10–20× slower on large
                # event patterns (e.g. 7.5 M trip edges = 132 s vs ~5 s).
                counts = self._pair_count_direct(graph_pid)
                if not counts:
                    self._attract_edge_base_cache[base_key] = []
                    base_sorted = []
                else:
                    # Batch-load ALL endpoint delta vectors in one geometry read.
                    # We intentionally do NOT push endpoints into Lance as a
                    # point_keys filter here, even though on first glance that
                    # looks like free pruning. On endpoint-dense patterns (e.g.
                    # AML-shaped transaction graphs where every account appears
                    # in at least one edge) Lance BTREE predicate pushdown with
                    # hundreds of thousands of keys is ~1000-1500× slower than a
                    # full-population scan of a cached file. The full scan plus
                    # Python-side membership check is consistently fast across
                    # the endpoint-density regimes we see in practice.
                    all_keys = {k for pair in counts.keys() for k in pair}
                    # Late-stage ratio guard — fires on spheres without the
                    # build-time edge_stats cache (older builds).  Same predicate
                    # as the early bail above; here we already paid for the
                    # groupby, so we skip only the geometry scan.
                    geo_row_count = self._storage.count_geometry_rows(pattern_id, version)
                    if geo_row_count > 0 and len(all_keys) / geo_row_count < 0.01:
                        self._attract_edge_base_cache[base_key] = []
                        base_sorted = []
                    else:
                        geo = self._storage.read_geometry(
                            pattern_id, version, columns=["primary_key", "delta"],
                        )
                        delta_by_key: dict[str, np.ndarray] = {}
                        for i in range(geo.num_rows):
                            k = geo["primary_key"][i].as_py()
                            if k in all_keys:
                                delta_by_key[k] = np.asarray(geo["delta"][i].as_py(), dtype=np.float64)

                        scored: list[dict[str, Any]] = []
                        for (f, t), cnt in counts.items():
                            if f not in delta_by_key or t not in delta_by_key:
                                continue
                            distance = float(np.linalg.norm(delta_by_key[f] - delta_by_key[t]))
                            effective_count = max(1, min(cnt, self._EDGE_POTENTIAL_PAIR_CAP))
                            score = distance * (1.0 / effective_count)
                            scored.append({
                                "from_key": f,
                                "to_key": t,
                                "score": round(score, 6),
                                "delta_distance": round(distance, 4),
                                "pair_tx_count": cnt,
                            })
                        scored.sort(key=lambda r: -r["score"])

                        # Assign GLOBAL rank_pct + is_high_potential (pattern-wide p95).
                        n = len(scored)
                        if n > 0:
                            p95_idx = max(0, int(n * 0.05) - 1)
                            p95_threshold = scored[p95_idx]["score"] if p95_idx < n else scored[0]["score"]
                            for i, r in enumerate(scored):
                                r["score_rank_pct_global"] = round(100.0 * (n - i) / n, 2)
                                r["is_high_potential"] = bool(r["score"] >= p95_threshold)
                        base_sorted = scored
                        self._attract_edge_base_cache[base_key] = base_sorted

        # ---- Fast-path: unfiltered default call returns base directly ----
        if from_key is None and to_key is None and min_pair_count == 1:
            # Back-compat field: also surface `score_rank_pct` (alias of global).
            for r in base_sorted:
                r.setdefault("score_rank_pct", r.get("score_rank_pct_global"))
            return base_sorted[:top_n]

        # ---- Filter path: O(m) pass over base_sorted, no rescoring ----
        filter_key = (pattern_id, version, from_key, to_key, min_pair_count)
        if not hasattr(self, "_attract_edge_filter_cache"):
            self._attract_edge_filter_cache: dict[tuple, list[dict[str, Any]]] = {}
        filtered = self._attract_edge_filter_cache.get(filter_key)
        if filtered is None:
            filtered = []
            for r in base_sorted:
                if from_key is not None and r["from_key"] != from_key:
                    continue
                if to_key is not None and r["to_key"] != to_key:
                    continue
                if r["pair_tx_count"] < min_pair_count:
                    continue
                filtered.append(r)
            # Reassign `score_rank_pct` (filter-local) and preserve global fields.
            m = len(filtered)
            for i, r in enumerate(filtered):
                # Clone to avoid mutating base_sorted with filter-local pct.
                cloned = dict(r)
                cloned["score_rank_pct_in_filter"] = round(100.0 * (m - i) / m, 2) if m else 100.0
                cloned["score_rank_pct"] = cloned.get("score_rank_pct_global")
                filtered[i] = cloned
            self._attract_edge_filter_cache[filter_key] = filtered

        return filtered[:top_n]

    # ------------------------------------------------------------------
    # find_motif — structural motif scoring (0.5.0)
    # ------------------------------------------------------------------

    # _MOTIF_SCORE_EPSILON mirrored as class attribute for legacy
    # `self._MOTIF_SCORE_EPSILON` subclass overrides — the actual scorers
    # reference the module-level constant. _MOTIF_SCORE_MAX is module-level
    # only (no historical subclass override to preserve).
    _MOTIF_SCORE_EPSILON = 1e-30

    @property
    def _motif_registry(self) -> dict[str, MotifSpec]:
        """Dispatch table for named AML motifs.

        Lazy-initialised so subclasses or monkey-patched test doubles can
        swap enumerators without fighting a class-level attribute.
        """
        cached = getattr(self, "_motif_registry_cache", None)
        if cached is not None:
            return cached
        registry: dict[str, MotifSpec] = {
            "fan_out": MotifSpec(
                enumerate=GDSNavigator._enumerate_fan_out,
                default_window_hours=168,
                min_instances=3,
            ),
            "cycle_2": MotifSpec(
                enumerate=GDSNavigator._enumerate_cycle_2,
                default_window_hours=24,
                min_instances=1,
            ),
            "cycle_3": MotifSpec(
                enumerate=GDSNavigator._enumerate_cycle_3,
                default_window_hours=72,
                min_instances=1,
            ),
            "structuring": MotifSpec(
                enumerate=GDSNavigator._enumerate_structuring,
                default_window_hours=1,
                min_instances=1,
            ),
            "fan_in": MotifSpec(
                enumerate=GDSNavigator._enumerate_fan_in,
                default_window_hours=168,
                min_instances=3,
            ),
            "chain_k": MotifSpec(
                enumerate=GDSNavigator._enumerate_chain_k,
                default_window_hours=168,
                min_instances=1,
            ),
            "split_recombine": MotifSpec(
                enumerate=GDSNavigator._enumerate_split_recombine,
                default_window_hours=168,
                min_instances=1,
            ),
            "bipartite_burst": MotifSpec(
                enumerate=GDSNavigator._enumerate_bipartite_burst,
                default_window_hours=24,
                min_instances=1,
            ),
        }
        self._motif_registry_cache = registry
        return registry

    def _score_motif_from_edges(
        self,
        edges: list[tuple[str, str]],
        pattern_id: str,
        *,
        event_keys: list[str] | None = None,
        event_pattern_id: str | None = None,
    ) -> dict[str, Any]:
        """Score a motif as the product of edge_potential across its edges.

        Batched implementation: pre-fetches all unique endpoint deltas in a
        single filtered Lance scan, then reuses the warm AdjacencyIndex
        pair_counts cache. Cuts geometry I/O from O(num_edges) per-endpoint
        Lance reads to O(1) batched scan over the endpoint subset.

        Event-aware scoring (opt-in via ``event_keys`` + ``event_pattern_id``):
        when both are supplied, each edge's potential is additionally
        weighted by ``(1 + ||event_polygon[event_key]||)`` from the event
        pattern's per-event geometry. This breaks the score-collapse where
        multiple distinct events between the same ``(from, to)`` node pair
        would otherwise produce identical scores (delta_distance and
        pair_count both depend only on the node pair). The ``+ 1`` keeps
        normal events (event_norm = 0 at population centroid) at parity
        with the node-pair-only score; anomalous events boost above. When
        either argument is omitted scoring reduces to the legacy
        node-pair-only formula — preserving backward compat for existing
        single-motif callers (``score_motif``, ``find_high_potential_motifs``).

        A zero edge_potential (identical-delta endpoints) is semantically
        distinct from underflow — it collapses the motif score to exactly 0.0
        to surface "structurally a motif, geometrically indistinguishable
        endpoints" cases. Non-zero products below 1e-30 are clamped to 1e-30
        to keep sorting stable; products above 1e300 (or +inf) are clamped
        DOWN to 1e300 and the ``score_clamped`` flag is set. ``log_score``
        = sum of log(ep) across non-zero edges (or ``-inf`` when any edge is
        zero) stays informative past both clamps and is what the ranking
        sort actually keys on.
        """
        if not edges:
            raise GDSNavigationError(
                "_score_motif_from_edges requires a non-empty edges list.",
            )
        if event_keys is not None and len(event_keys) != len(edges):
            raise GDSNavigationError(
                f"event_keys length {len(event_keys)} does not match edges "
                f"length {len(edges)} — invariant violation.",
            )
        endpoint_keys: set[str] = set()
        for (u, v) in edges:
            endpoint_keys.add(u)
            endpoint_keys.add(v)
        version = self._resolve_version(pattern_id)
        delta_map = self._batch_read_deltas(pattern_id, version, endpoint_keys)

        event_factors: list[float] | None = None
        if event_keys is not None and event_pattern_id is not None:
            event_version = self._resolve_version(event_pattern_id)
            event_delta_map = self._batch_read_deltas(
                event_pattern_id, event_version, set(event_keys),
            )
            event_factors = []
            for ek in event_keys:
                ed = event_delta_map.get(ek)
                if ed is None:
                    # Event polygon missing — neutral factor (no boost,
                    # no penalty). Keeps ranking stable when event-pattern
                    # geometry is partial.
                    event_factors.append(1.0)
                else:
                    event_factors.append(1.0 + float(np.linalg.norm(ed)))

        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        pair_counts = adj.pair_counts()
        result = self._lean_score_motif(
            edges, delta_map, pair_counts,
            event_factors=event_factors,
        )
        if result is None:
            # _lean_score_motif returned None — at least one endpoint is absent from the batched delta read.
            missing = next(
                (u, v) for (u, v) in edges
                if u not in delta_map or v not in delta_map
            )
            raise GDSNavigationError(
                f"Entity {missing[0]!r} or {missing[1]!r} not found in "
                f"pattern {pattern_id!r} geometry.",
            )
        return result

    _MOTIF_RANKING_CACHE_MAX = 8

    def score_motif(
        self,
        entity_key: str,
        motif_type: str,
        pattern_id: str,
        time_window_hours: int | None = None,
        amt1_min: float = 10000.0,
        amt2_max: float = 10000.0,
        min_k: int | None = None,
        k: int = 4,
        direction: str = "forward",
        min_m: int = 3,
    ) -> dict[str, Any]:
        """Score the best ``motif_type`` instance seeded at ``entity_key``.

        When multiple motif instances are found around ``entity_key``, the one
        with the highest product-of-edge_potential score wins. Returns a dict
        with ``found`` flag, ``score``, ``breakdown`` (per-edge scores), and
        motif-specific identifying fields (``counterparty`` for cycle_2,
        ``ring`` for cycle_3, ``k`` for fan_out / fan_in / chain_k /
        split_recombine / bipartite_burst, ``path`` for structuring /
        chain_k, ``source`` / ``sink`` / ``intermediaries`` for
        split_recombine, ``sources`` / ``sinks`` / ``m`` for bipartite_burst).
        ``amt1_min`` and ``amt2_max`` gate the three hops of a
        ``structuring`` motif; ``k`` sets the chain length for ``chain_k``
        (3 ≤ k ≤ 8, default 4); ignored for other motif types. ``min_k``
        overrides the distinct-neighbour threshold for ``fan_out`` /
        ``fan_in`` / ``split_recombine`` / ``bipartite_burst`` (default 3
        when ``None``). ``direction`` ("forward" or "backward") chooses
        which side of a ``split_recombine`` diamond ``entity_key`` plays;
        ignored for other motif types. ``min_m`` sets the second cardinality
        of a ``bipartite_burst`` K_{k,m} subgraph (default 3); ignored for
        other motif types.
        """
        if motif_type not in self._motif_registry:
            valid = ", ".join(sorted(self._motif_registry.keys()))
            raise GDSNavigationError(
                f"Unknown motif_type '{motif_type}'. Valid: {valid}.",
            )
        if time_window_hours is not None and time_window_hours <= 0:
            raise GDSNavigationError(
                f"time_window_hours must be positive, got {time_window_hours}.",
            )
        if min_k is not None and min_k < 1:
            raise GDSNavigationError(
                f"min_k must be ≥ 1 when provided, got {min_k}.",
            )
        if motif_type == "structuring":
            if amt1_min <= 0 or amt2_max <= 0:
                raise GDSNavigationError(
                    "amt1_min and amt2_max must be positive for structuring motif.",
                )
        if motif_type == "chain_k":
            if k < 3 or k > 8:
                raise GDSNavigationError(
                    f"chain_k requires 3 ≤ k ≤ 8, got k={k}.",
                )
        if motif_type == "split_recombine":
            if direction not in ("forward", "backward"):
                raise GDSNavigationError(
                    f"split_recombine direction must be 'forward' or 'backward', got {direction!r}.",
                )
            if min_k is not None and min_k < 2:
                raise GDSNavigationError(
                    f"split_recombine min_k must be >= 2, got {min_k}.",
                )
        if motif_type == "bipartite_burst":
            effective_mk = min_k if min_k is not None else 3
            if effective_mk < 2 or min_m < 2:
                raise GDSNavigationError(
                    f"bipartite_burst requires min_k >= 2 and min_m >= 2, got min_k={effective_mk}, min_m={min_m}.",
                )
        spec = self._motif_registry[motif_type]
        window = time_window_hours if time_window_hours is not None else spec.default_window_hours

        enum_kwargs: dict[str, Any] = {
            "amt1_min": amt1_min, "amt2_max": amt2_max, "k": k,
            "direction": direction, "min_m": min_m,
        }
        if min_k is not None:
            enum_kwargs["min_k"] = min_k
        instances = spec.enumerate(
            self, entity_key, pattern_id, window, **enum_kwargs,
        )
        if not instances:
            return {
                "motif_type": motif_type,
                "seed": entity_key,
                "pattern_id": pattern_id,
                "found": False,
                "score": 0.0,
                "time_window_hours": window,
                "reason": f"no {motif_type} motif around entity in window {window}h",
            }
        best: dict[str, Any] | None = None
        best_score = -1.0
        for inst in instances:
            scored = self._score_motif_from_edges(inst["edges"], pattern_id)
            if scored["score"] > best_score:
                best_score = scored["score"]
                best = {**inst, **scored}
        assert best is not None
        best["pattern_id"] = pattern_id
        best["found"] = True
        best["time_window_hours"] = window
        return best

    def find_high_potential_motifs(
        self,
        pattern_id: str,
        motif_type: str,
        top_n: int = 10,
        time_window_hours: int | None = None,
        seeds: list[str] | None = None,
        min_k: int | None = None,
        amt1_min: float = 10000.0,
        amt2_max: float = 10000.0,
        k: int = 4,
        direction: str = "forward",
        min_m: int = 3,
    ) -> list[dict[str, Any]]:
        """Rank ``motif_type`` instances across the pattern by score, desc.

        LRU-cached per ``(pattern_id, version, motif_type, time_window_hours,
        amt1_min, amt2_max, k, direction, min_m)``. Filters by ``seeds``
        (post-cache) if provided. For ``cycle_3`` results are deduplicated
        by canonical ring (sorted tuple of primary keys) so each physical
        cycle appears once regardless of which of 3 seeds surfaces it. For
        ``structuring`` results are deduplicated by canonical path (tuple
        of 4 primary keys). For ``split_recombine`` results are
        deduplicated by ``(direction, source, sink, sorted intermediaries)``.
        For ``bipartite_burst`` results are deduplicated by
        ``(frozenset sources, frozenset sinks)``. Amount thresholds
        ``amt1_min`` / ``amt2_max`` only affect ``structuring``; other
        motif types ignore them.
        """
        if motif_type not in self._motif_registry:
            valid = ", ".join(sorted(self._motif_registry.keys()))
            raise GDSNavigationError(
                f"Unknown motif_type '{motif_type}'. Valid: {valid}.",
            )
        if top_n <= 0:
            raise GDSNavigationError(f"top_n must be positive, got {top_n}.")
        if time_window_hours is not None and time_window_hours <= 0:
            raise GDSNavigationError(
                f"time_window_hours must be positive, got {time_window_hours}.",
            )
        if motif_type == "structuring":
            if amt1_min <= 0 or amt2_max <= 0:
                raise GDSNavigationError(
                    "amt1_min and amt2_max must be positive for structuring motif.",
                )
        if motif_type == "chain_k":
            if k < 3 or k > 8:
                raise GDSNavigationError(
                    f"chain_k requires 3 ≤ k ≤ 8, got k={k}.",
                )
        if motif_type == "split_recombine":
            if direction not in ("forward", "backward"):
                raise GDSNavigationError(
                    f"split_recombine direction must be 'forward' or 'backward', got {direction!r}.",
                )
            if min_k is not None and min_k < 2:
                raise GDSNavigationError(
                    f"split_recombine min_k must be >= 2, got {min_k}.",
                )
        if motif_type == "bipartite_burst":
            effective_mk = min_k if min_k is not None else 3
            if effective_mk < 2 or min_m < 2:
                raise GDSNavigationError(
                    f"bipartite_burst requires min_k >= 2 and min_m >= 2, got min_k={effective_mk}, min_m={min_m}.",
                )
        spec = self._motif_registry[motif_type]
        window = time_window_hours if time_window_hours is not None else spec.default_window_hours
        version = self._resolve_version(pattern_id)

        cache_key = (
            pattern_id, version, motif_type, window,
            amt1_min, amt2_max, k, direction, min_m,
        )
        if not hasattr(self, "_motif_ranking_cache"):
            from collections import OrderedDict
            self._motif_ranking_cache: OrderedDict[tuple, list[dict[str, Any]]] = OrderedDict()
        cache = self._motif_ranking_cache
        if cache_key in cache:
            cache.move_to_end(cache_key)
            ranked = cache[cache_key]
        else:
            # Discover all seeds from the pattern's geometry primary_key column.
            geo = self._storage.read_geometry(
                pattern_id, version, columns=["primary_key"],
            )
            all_seeds = geo["primary_key"].to_pylist() if geo.num_rows else []
            ranked = self._rank_motifs(
                all_seeds, pattern_id, motif_type, window, min_k,
                amt1_min=amt1_min, amt2_max=amt2_max, k=k,
                direction=direction, min_m=min_m,
            )
            cache[cache_key] = ranked
            while len(cache) > self._MOTIF_RANKING_CACHE_MAX:
                cache.popitem(last=False)

        if seeds is not None:
            seed_set = set(seeds)
            filtered = [r for r in ranked if r["seed"] in seed_set]
        else:
            filtered = ranked
        return filtered[:top_n]

    def _batch_read_deltas(
        self, pattern_id: str, version: int, keys: set[str],
    ) -> dict[str, np.ndarray]:
        """Filtered geometry read → ``{primary_key: delta_vector}`` for ``keys``.

        Uses Lance ``point_keys`` BTREE filter so only rows for the endpoints
        actually participating in motif instances are materialised — cuts
        geometry I/O from full-population (often 500k+ rows) down to the
        endpoint subset (typically <20% of the population on AML-shaped
        patterns where most accounts are isolated or low-degree).
        """
        if not keys:
            return {}
        geo = self._storage.read_geometry(
            pattern_id, version,
            point_keys=list(keys),
            columns=["primary_key", "delta"],
        )
        pk_col = geo["primary_key"].to_pylist()
        delta_col = geo["delta"]
        result: dict[str, np.ndarray] = {}
        for i, pk in enumerate(pk_col):
            result[pk] = np.asarray(delta_col[i].as_py(), dtype=np.float64)
        return result

    def _lean_score_motif(
        self,
        edges: list[tuple[str, str]],
        delta_map: dict[str, np.ndarray],
        pair_counts: dict[tuple[str, str], int],
        *,
        event_factors: list[float] | None = None,
    ) -> dict[str, Any] | None:
        """Product of edge_potential scores using only the pre-fetched caches.

        ``event_factors`` (optional, length must equal ``edges`` when provided):
        per-edge multiplicative boost from event-pattern geometry — see
        ``_score_motif_from_edges`` docstring for semantics. Surfaced in the
        per-edge breakdown when supplied.

        Returns ``None`` when any endpoint is missing from ``delta_map``
        (skipped from the ranking).
        """
        breakdown: list[dict[str, Any]] = []
        product = 1.0
        log_score = 0.0
        saw_zero = False
        for i, (u, v) in enumerate(edges):
            if u not in delta_map or v not in delta_map:
                return None
            distance = float(np.linalg.norm(delta_map[u] - delta_map[v]))
            if u == v:
                cnt = pair_counts.get((u, v), 0)
            else:
                cnt = pair_counts.get((u, v), 0) + pair_counts.get((v, u), 0)
            effective = max(1, min(cnt, self._EDGE_POTENTIAL_PAIR_CAP))
            event_factor = (
                event_factors[i] if event_factors is not None else 1.0
            )
            edge_score = distance * (1.0 / effective) * event_factor
            entry: dict[str, Any] = {
                "edge": (u, v),
                "edge_potential": round(edge_score, 6),
                "delta_distance": round(distance, 4),
                "pair_tx_count": cnt,
                "effective_weight": round(1.0 / effective, 6),
            }
            if event_factors is not None:
                entry["event_factor"] = round(event_factor, 6)
            breakdown.append(entry)
            if edge_score == 0.0:
                saw_zero = True
            else:
                log_score += math.log(edge_score)
            product *= edge_score
        score_clamped = False
        if saw_zero:
            product_out: float = 0.0
            log_score_out: float = -math.inf
        else:
            if product < _MOTIF_SCORE_EPSILON:
                product_out = _MOTIF_SCORE_EPSILON
            elif math.isinf(product) or product > _MOTIF_SCORE_MAX:
                product_out = _MOTIF_SCORE_MAX
                score_clamped = True
            else:
                product_out = product
            log_score_out = log_score
        return {
            "score": product_out,
            "log_score": log_score_out,
            "score_clamped": score_clamped,
            "breakdown": breakdown,
        }

    def _rank_motifs(
        self,
        all_seeds: list[str],
        pattern_id: str,
        motif_type: str,
        window: int,
        min_k: int | None,
        amt1_min: float = 10000.0,
        amt2_max: float = 10000.0,
        k: int = 4,
        direction: str = "forward",
        min_m: int = 3,
    ) -> list[dict[str, Any]]:
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        version = self._resolve_version(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        if adj.edge_count() == 0:
            return []
        pair_counts = adj.pair_counts()
        window_sec = float(window) * 3600.0
        effective_min_k = (
            min_k if (
                motif_type in {"fan_out", "fan_in", "split_recombine", "bipartite_burst"}
                and min_k is not None
            ) else 3
        )
        effective_min_m = min_m if min_m is not None else 3

        active_seeds = self._active_seeds_for_motif(
            adj, all_seeds, motif_type, effective_min_k,
            amt1_min=amt1_min,
            direction=direction, min_m=effective_min_m,
        )

        instances: list[dict[str, Any]] = []
        seen_rings: set = set()
        for seed in active_seeds:
            if motif_type == "fan_out":
                found = self._enum_fan_out_via_adj(seed, adj, window_sec, effective_min_k)
            elif motif_type == "fan_in":
                found = self._enum_fan_in_via_adj(
                    seed, adj, window_sec, effective_min_k,
                )
            elif motif_type == "cycle_2":
                found = self._enum_cycle_2_via_adj(seed, adj, window_sec)
            elif motif_type == "cycle_3":
                found = self._enum_cycle_3_via_adj(seed, adj, window_sec)
            elif motif_type == "structuring":
                found = self._enum_structuring_via_adj(
                    seed, adj, window_sec, amt1_min, amt2_max,
                )
            elif motif_type == "chain_k":
                _chain_k_frontier = self._CHAIN_K_MAX_FRONTIER_PER_K.get(k, self._CHAIN_K_MAX_FRONTIER)
                found = self._enum_chain_k_via_adj(
                    seed, adj, window_sec, k,
                    _chain_k_frontier, self._CHAIN_K_MAX_RESULTS,
                )
            elif motif_type == "split_recombine":
                found = self._enum_split_recombine_via_adj(
                    seed, adj, window_sec, effective_min_k, direction,
                )
            elif motif_type == "bipartite_burst":
                found = self._enum_bipartite_burst_via_adj(
                    seed, adj, window_sec, effective_min_k, effective_min_m,
                )
            else:
                found = []
            for inst in found:
                if motif_type == "cycle_3":
                    ring = tuple(sorted(inst["ring"]))
                    if ring in seen_rings:
                        continue
                    seen_rings.add(ring)
                elif motif_type == "cycle_2":
                    pair = tuple(sorted([inst["seed"], inst["counterparty"]]))
                    if pair in seen_rings:
                        continue
                    seen_rings.add(pair)
                elif motif_type in {"structuring", "chain_k"}:
                    canonical = tuple(inst["path"])
                    if canonical in seen_rings:
                        continue
                    seen_rings.add(canonical)
                elif motif_type == "split_recombine":
                    canonical_sr = (
                        inst["direction"], inst["source"], inst["sink"],
                        tuple(sorted(inst["intermediaries"])),
                    )
                    if canonical_sr in seen_rings:
                        continue
                    seen_rings.add(canonical_sr)
                elif motif_type == "bipartite_burst":
                    canonical_bb = (
                        frozenset(inst["sources"]),
                        frozenset(inst["sinks"]),
                    )
                    if canonical_bb in seen_rings:
                        continue
                    seen_rings.add(canonical_bb)
                instances.append(inst)

        all_endpoint_keys: set[str] = set()
        for inst in instances:
            for (u, v) in inst["edges"]:
                all_endpoint_keys.add(u)
                all_endpoint_keys.add(v)
        delta_map = self._batch_read_deltas(pattern_id, version, all_endpoint_keys)

        scored: list[dict[str, Any]] = []
        for inst in instances:
            sc = self._lean_score_motif(inst["edges"], delta_map, pair_counts)
            if sc is None:
                continue
            scored.append({**inst, **sc})
        # Sort by log_score DESC rather than raw score — stays correct past
        # the overflow clamp (_MOTIF_SCORE_MAX). log_score is monotonic with
        # product over the finite positive range, so order below the clamp
        # is unchanged; above the clamp log_score breaks ties that the
        # clamped raw score can't distinguish. -inf log_score (zero product)
        # naturally sorts last under DESC.
        scored.sort(key=lambda r: -r["log_score"])
        n = len(scored)
        if n > 0:
            p95_idx = max(0, int(n * 0.05) - 1)
            p95_threshold = scored[p95_idx]["score"] if p95_idx < n else scored[0]["score"]
            for i, r in enumerate(scored):
                r["score_rank_pct"] = round(100.0 * (n - i) / n, 2)
                r["is_high_potential"] = bool(r["score"] >= p95_threshold)
        return scored

    @staticmethod
    def _active_seeds_for_motif(
        adj: "AdjacencyIndex",
        all_seeds: list[str],
        motif_type: str,
        effective_min_k: int,
        amt1_min: float = 10000.0,
        direction: str = "forward",
        min_m: int = 3,
    ) -> list[str]:
        all_seeds_set = set(all_seeds)
        if motif_type in {"cycle_2", "cycle_3"}:
            return [
                s for s in (set(adj._out.keys()) & set(adj._in.keys()))
                if s in all_seeds_set
            ]
        if motif_type == "fan_out":
            return [
                s for s in all_seeds_set
                if adj.distinct_neighbors_out(s) >= effective_min_k
            ]
        if motif_type == "fan_in":
            return [
                s for s in all_seeds_set
                if adj.distinct_neighbors_in(s) >= effective_min_k
            ]
        if motif_type == "split_recombine":
            if direction == "forward":
                return [
                    s for s in all_seeds_set
                    if adj.distinct_neighbors_out(s) >= effective_min_k
                ]
            return [
                s for s in all_seeds_set
                if adj.distinct_neighbors_in(s) >= effective_min_k
            ]
        if motif_type == "bipartite_burst":
            return [
                s for s in all_seeds_set
                if adj.distinct_neighbors_out(s) >= min_m
                or adj.distinct_neighbors_in(s) >= effective_min_k
            ]
        if motif_type == "structuring":
            return [
                s for s in all_seeds_set
                if adj.max_amount_out_excl_self(s) >= amt1_min
            ]
        if motif_type == "chain_k":
            return [
                s for s in all_seeds_set
                if adj.distinct_neighbors_out(s) > 0
            ]
        return [s for s in all_seeds if s in adj._out]

    @staticmethod
    def _enum_fan_out_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
        min_k: int,
    ) -> list[dict[str, Any]]:
        nbr = adj.neighbors_out_window(seed)
        timestamps = nbr["timestamp"]
        if not timestamps:
            return []
        to_keys = nbr["to_key"]
        max_ts = max(timestamps)
        recent = [
            (t, ts) for t, ts in zip(to_keys, timestamps, strict=True)
            if t != seed and ts >= max_ts - window_sec
        ]
        unique = sorted({t for (t, _) in recent})
        if len(unique) < min_k:
            return []
        return [{
            "motif_type": "fan_out",
            "seed": seed,
            "k": len(unique),
            "edges": [(seed, t) for t in unique],
        }]

    @staticmethod
    def _enum_fan_in_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
        min_k: int,
    ) -> list[dict[str, Any]]:
        """Mirror of _enum_fan_out_via_adj: seed = sink, collect distinct sources."""
        nbr = adj.neighbors_in_window(seed)
        timestamps = nbr["timestamp"]
        if not timestamps:
            return []
        from_keys = nbr["from_key"]
        max_ts = max(timestamps)
        recent = [
            (f, ts) for f, ts in zip(from_keys, timestamps, strict=True)
            if f != seed and ts >= max_ts - window_sec
        ]
        unique = sorted({f for (f, _) in recent})
        if len(unique) < min_k:
            return []
        return [{
            "motif_type": "fan_in",
            "seed": seed,
            "k": len(unique),
            "edges": [(f, seed) for f in unique],
        }]

    @staticmethod
    def _enum_split_recombine_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
        min_k: int,
        direction: str = "forward",
    ) -> list[dict[str, Any]]:
        """Adjacency-path diamond enumerator; mirrors _enumerate_split_recombine."""
        if direction not in ("forward", "backward"):
            return []
        if direction == "forward":
            out_edges = adj.neighbors_out(seed)
            if not out_edges:
                return []
            max_ts = max(ts for (_n, ts, *_r) in out_edges)
            latest_split_in: dict[str, float] = {}
            for (m, ts, *_r) in out_edges:
                if m == seed or ts < max_ts - window_sec:
                    continue
                if ts > latest_split_in.get(m, float("-inf")):
                    latest_split_in[m] = ts
            if len(latest_split_in) < min_k:
                return []
            sink_to_inter: dict[str, set[str]] = {}
            for m, s_in_ts in latest_split_in.items():
                for (d, t, *_r) in adj.neighbors_out(m):
                    if d == seed or d == m:
                        continue
                    if t <= s_in_ts or t - s_in_ts > window_sec:
                        continue
                    sink_to_inter.setdefault(d, set()).add(m)
            best_sink: str | None = None
            best_inter: set[str] = set()
            for d, inter in sorted(sink_to_inter.items()):
                if len(inter) >= min_k and len(inter) > len(best_inter):
                    best_sink, best_inter = d, inter
            if best_sink is None:
                return []
            inter_sorted = sorted(best_inter)
            edges = [(seed, m) for m in inter_sorted] + [
                (m, best_sink) for m in inter_sorted
            ]
            return [{
                "motif_type": "split_recombine",
                "direction": "forward",
                "seed": seed,
                "source": seed,
                "sink": best_sink,
                "k": len(inter_sorted),
                "intermediaries": inter_sorted,
                "edges": edges,
            }]

        # direction == "backward"
        in_edges = adj.neighbors_in(seed)
        if not in_edges:
            return []
        max_ts = max(ts for (_n, ts, *_r) in in_edges)
        earliest_recomb_out: dict[str, float] = {}
        for (m, ts, *_r) in in_edges:
            if m == seed or ts < max_ts - window_sec:
                continue
            if ts < earliest_recomb_out.get(m, float("inf")):
                earliest_recomb_out[m] = ts
        if len(earliest_recomb_out) < min_k:
            return []
        source_to_inter: dict[str, set[str]] = {}
        for m, r_out_ts in earliest_recomb_out.items():
            for (s, t, *_r) in adj.neighbors_in(m):
                if s == seed or s == m:
                    continue
                if t >= r_out_ts or r_out_ts - t > window_sec:
                    continue
                source_to_inter.setdefault(s, set()).add(m)
        best_source: str | None = None
        best_inter = set()
        for s, inter in sorted(source_to_inter.items()):
            if len(inter) >= min_k and len(inter) > len(best_inter):
                best_source, best_inter = s, inter
        if best_source is None:
            return []
        inter_sorted = sorted(best_inter)
        edges = [(best_source, m) for m in inter_sorted] + [
            (m, seed) for m in inter_sorted
        ]
        return [{
            "motif_type": "split_recombine",
            "direction": "backward",
            "seed": seed,
            "source": best_source,
            "sink": seed,
            "k": len(inter_sorted),
            "intermediaries": inter_sorted,
            "edges": edges,
        }]

    @staticmethod
    def _enum_bipartite_burst_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
        min_k: int,
        min_m: int,
    ) -> list[dict[str, Any]]:
        """Adjacency K_{k,m} enumerator with fused source/sink dispatcher.

        Branch B optimisation: fetches out_edges / in_edges once, pre-checks
        distinct degree before invoking the heavyweight inner logic.  Seeds that
        only qualify on one side skip the other side entirely, eliminating the
        fruitless `_try_source` fallback that dominated Phase 1 cProfile output.

        Nested helpers (_try_source_inner / _try_sink_inner) receive the
        per-column dict windows produced by `neighbors_out_window` /
        `neighbors_in_window` so no second adj lookup is needed.  Inner logic
        iterates `zip(keys, timestamps)` instead of unpacking 4-tuples — the
        `amount` and `event_key` columns are not consumed at this layer and
        skipping their materialization is the C1 win.  Small-set-first
        intersection ordering is preserved.
        """

        def _try_source_inner(
            seed_: str,
            out_window_: dict[str, list[Any]],
        ) -> dict[str, Any] | None:
            timestamps = out_window_["timestamp"]
            if not timestamps:
                return None
            to_keys = out_window_["to_key"]
            max_ts = max(timestamps)
            sinks = sorted({
                d for d, ts in zip(to_keys, timestamps, strict=True)
                if d != seed_ and ts >= max_ts - window_sec
            })
            if len(sinks) < min_m:
                return None
            sources_per_sink: dict[str, set[str]] = {}
            window_min = max_ts - window_sec
            for d in sinks:
                nbr = adj.neighbors_in_window(d, ts_min=window_min, columns=("from_key",))
                for f in nbr["from_key"]:
                    if f == d:
                        continue
                    sources_per_sink.setdefault(d, set()).add(f)
            sinks = [d for d in sinks if d in sources_per_sink]
            if len(sinks) < min_m:
                return None
            sinks_by_size = sorted(sinks, key=lambda d: len(sources_per_sink[d]))
            common = set(sources_per_sink[sinks_by_size[0]])
            for d in sinks_by_size[1:]:
                common &= sources_per_sink[d]
                if len(common) < min_k:
                    break
            if seed_ not in common:
                return None
            if len(common) < min_k:
                return None
            sources_sorted = sorted(common)
            sources_set = set(sources_sorted)
            sinks_final = [
                d for d in sorted(sinks)
                if sources_set <= sources_per_sink[d]
            ]
            if len(sinks_final) < min_m:
                return None
            edges = [(s, d) for s in sources_sorted for d in sinks_final]
            return {
                "motif_type": "bipartite_burst",
                "seed": seed_,
                "k": len(sources_sorted),
                "m": len(sinks_final),
                "sources": sources_sorted,
                "sinks": sinks_final,
                "edges": edges,
            }

        def _try_sink_inner(
            seed_: str,
            in_window_: dict[str, list[Any]],
        ) -> dict[str, Any] | None:
            timestamps = in_window_["timestamp"]
            if not timestamps:
                return None
            from_keys = in_window_["from_key"]
            max_ts = max(timestamps)
            sources = sorted({
                f for f, ts in zip(from_keys, timestamps, strict=True)
                if f != seed_ and ts >= max_ts - window_sec
            })
            if len(sources) < min_k:
                return None
            sinks_per_source: dict[str, set[str]] = {}
            window_min = max_ts - window_sec
            for s in sources:
                nbr = adj.neighbors_out_window(s, ts_min=window_min, columns=("to_key",))
                for d in nbr["to_key"]:
                    if d == s:
                        continue
                    sinks_per_source.setdefault(s, set()).add(d)
            sources = [s for s in sources if s in sinks_per_source]
            if len(sources) < min_k:
                return None
            sources_by_size = sorted(sources, key=lambda s: len(sinks_per_source[s]))
            common = set(sinks_per_source[sources_by_size[0]])
            for s in sources_by_size[1:]:
                common &= sinks_per_source[s]
                if len(common) < min_m:
                    break
            if seed_ not in common:
                return None
            if len(common) < min_m:
                return None
            sinks_sorted = sorted(common)
            sinks_set = set(sinks_sorted)
            sources_final = [
                s for s in sorted(sources)
                if sinks_set <= sinks_per_source[s]
            ]
            if len(sources_final) < min_k:
                return None
            edges = [(s, d) for s in sources_final for d in sinks_sorted]
            return {
                "motif_type": "bipartite_burst",
                "seed": seed_,
                "k": len(sources_final),
                "m": len(sinks_sorted),
                "sources": sources_final,
                "sinks": sinks_sorted,
                "edges": edges,
            }

        out_window = adj.neighbors_out_window(seed)
        in_window = adj.neighbors_in_window(seed)
        out_to_keys = out_window["to_key"]
        in_from_keys = in_window["from_key"]
        if not out_to_keys and not in_from_keys:
            return []

        distinct_out = len({t for t in out_to_keys if t != seed})
        distinct_in = len({f for f in in_from_keys if f != seed})

        candidates: list = []
        if distinct_out >= min_m:
            candidates.append(lambda: _try_source_inner(seed, out_window))
        if distinct_in >= min_k:
            candidates.append(lambda: _try_sink_inner(seed, in_window))

        for fn in candidates:
            hit = fn()
            if hit is not None:
                return [hit]
        return []

    @staticmethod
    def _enum_cycle_2_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
    ) -> list[dict[str, Any]]:
        out_nbr = adj.neighbors_out_window(seed)
        in_nbr = adj.neighbors_in_window(seed)
        if not out_nbr["timestamp"] or not in_nbr["timestamp"]:
            return []
        out_by_cp: dict[str, list[float]] = defaultdict(list)
        for t, ts in zip(out_nbr["to_key"], out_nbr["timestamp"], strict=True):
            if t != seed:
                out_by_cp[t].append(ts)
        in_by_cp: dict[str, list[float]] = defaultdict(list)
        for f, ts in zip(in_nbr["from_key"], in_nbr["timestamp"], strict=True):
            if f != seed:
                in_by_cp[f].append(ts)
        results: list[dict[str, Any]] = []
        for cp in set(out_by_cp) & set(in_by_cp):
            closest = min(abs(a - b) for a in out_by_cp[cp] for b in in_by_cp[cp])
            if closest > window_sec:
                continue
            results.append({
                "motif_type": "cycle_2",
                "seed": seed,
                "counterparty": cp,
                "edges": [(seed, cp), (cp, seed)],
            })
        return results

    @staticmethod
    def _enum_cycle_3_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
        max_triads: int = 50,
    ) -> list[dict[str, Any]]:
        out_from_seed = adj.neighbors_out(seed)
        if not out_from_seed:
            return []
        seed_to_b: dict[str, list[float]] = defaultdict(list)
        for (b, ts, *_r) in out_from_seed:
            if b != seed:
                seed_to_b[b].append(ts)
        # Pre-filter: keep only b-nodes that have at least one out-neighbor already
        # pointing back to seed.  Computed once outside the loop — O(|in(seed)|) build,
        # then O(|out(b)|) per b vs O(|out(b)| × |in(seed)|) inside.
        seed_in_set = {f for (f, *_r) in adj.neighbors_in(seed) if f != seed}
        seed_to_b = {
            b: ts_list
            for b, ts_list in seed_to_b.items()
            if any(c in seed_in_set for (c, *_r) in adj.neighbors_out(b))
        }
        triads: list[dict[str, Any]] = []
        for b, ts_ab_list in seed_to_b.items():
            for (c, ts_bc, *_r) in adj.neighbors_out(b):
                if c == seed or c == b:
                    continue
                closing = [ts for (x, ts, *_r) in adj.neighbors_out(c) if x == seed]
                if not closing:
                    continue
                best: tuple[float, float, float] | None = None
                for t_ab in ts_ab_list:
                    if ts_bc <= t_ab:
                        continue
                    for t_ca in closing:
                        if t_ca <= ts_bc:
                            continue
                        if t_ca - t_ab > window_sec:
                            continue
                        if best is None or (t_ca - t_ab) < (best[2] - best[0]):
                            best = (t_ab, ts_bc, t_ca)
                if best is None:
                    continue
                triads.append({
                    "motif_type": "cycle_3",
                    "seed": seed,
                    "ring": [seed, b, c],
                    "edges": [(seed, b), (b, c), (c, seed)],
                    "timestamps": list(best),
                })
                if len(triads) >= max_triads:
                    return triads
        return triads

    @staticmethod
    def _enum_structuring_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
        amt1_min: float,
        amt2_max: float,
        max_instances: int = 50,
    ) -> list[dict[str, Any]]:
        """Open 3-hop chain A→B→C→D with amount gating and strict temporal ordering.

        hop1 (A→B) amount >= amt1_min, hop2 (B→C) and hop3 (C→D) amount <= amt2_max.
        ts_ab < ts_bc < ts_cd, total span ts_cd - ts_ab <= window_sec. Self-visits
        rejected (D, C ∉ visited path prefix).
        """
        out1 = adj.neighbors_out(seed)
        # Guard NULL / non-positive amounts. EDGE_TABLE_SCHEMA declares amount
        # as nullable pa.float64, and some producers emit signed amounts
        # (refunds/reversals as negative). Structuring is a positive money
        # flow by definition — treating NULL or ≤ 0 as if zero would falsely
        # pass the "≤ amt2_max" small-hop predicate. Skip both at every hop.
        large_first = [
            (b, ts, amt)
            for (b, ts, amt, _ek) in out1
            if b != seed
            and amt is not None and amt > 0
            and amt >= amt1_min
        ]
        if not large_first:
            return []

        results: list[dict[str, Any]] = []
        for (b, ts_ab, amt_ab) in large_first:
            for (c, ts_bc, amt_bc, _ek_bc) in adj.neighbors_out(b):
                if c == seed or c == b:
                    continue
                if ts_bc <= ts_ab or ts_bc - ts_ab > window_sec:
                    continue
                if amt_bc is None or amt_bc <= 0 or amt_bc > amt2_max:
                    continue
                for (d, ts_cd, amt_cd, _ek_cd) in adj.neighbors_out(c):
                    if d == seed or d == b or d == c:
                        continue
                    if ts_cd <= ts_bc or ts_cd - ts_ab > window_sec:
                        continue
                    if amt_cd is None or amt_cd <= 0 or amt_cd > amt2_max:
                        continue
                    results.append({
                        "motif_type": "structuring",
                        "seed": seed,
                        "path": [seed, b, c, d],
                        "edges": [(seed, b), (b, c), (c, d)],
                        "timestamps": [ts_ab, ts_bc, ts_cd],
                        "amounts": [amt_ab, amt_bc, amt_cd],
                    })
                    if len(results) >= max_instances:
                        return results
        return results

    @staticmethod
    def _enum_chain_k_via_adj(
        seed: str,
        adj: "AdjacencyIndex",
        window_sec: float,
        k: int,
        max_frontier: int = 1000,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Adjacency-based DFS for open chain of length k from seed.

        Constraints: no node revisit (also blocks cycle closure), strict
        monotone timestamps, total span ≤ window_sec. Frontier capped per
        level at max_frontier; final results capped at max_results.
        """
        if k < 3 or k > 8:
            return []
        # Partial path: (path_tuple, edges_list, ts_first, ts_last)
        _Partial = tuple[tuple[str, ...], list[tuple[str, str]], float | None, float | None]
        frontier: list[_Partial] = [((seed,), [], None, None)]
        truncated = False
        for _hop in range(k - 1):
            next_frontier: list[_Partial] = []
            for (path, edge_list, ts_first, ts_last) in frontier:
                tail = path[-1]
                for (nxt, ts_new, _amt, _ek) in adj.neighbors_out(tail):
                    if nxt in path:
                        continue
                    if ts_last is not None and ts_new <= ts_last:
                        continue
                    new_first = ts_first if ts_first is not None else ts_new
                    if ts_new - new_first > window_sec:
                        continue
                    next_frontier.append((
                        path + (nxt,),
                        edge_list + [(tail, nxt)],
                        new_first,
                        ts_new,
                    ))
                    if len(next_frontier) >= max_frontier:
                        truncated = True
                        break
                if len(next_frontier) >= max_frontier:
                    truncated = True
                    break
            frontier = next_frontier
            if not frontier:
                return []
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for (path, edge_list, _ts_first, _ts_last) in frontier:
            if path in seen:
                continue
            seen.add(path)
            results.append({
                "motif_type": "chain_k",
                "seed": seed,
                "k": k,
                "path": list(path),
                "edges": edge_list,
                "frontier_truncated": truncated,
            })
            if len(results) >= max_results:
                break
        return results

    def _resolve_motif_graph_pid(self, pattern_id: str) -> str:
        """Return the graph companion pattern_id for motif enumeration."""
        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"Pattern '{pattern_id}' not found in sphere.")
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type == "anchor":
            graph_pid = self._resolve_edge_pattern_for_anchor(pattern_id)
            if graph_pid is None:
                raise GDSNavigationError(
                    f"Anchor pattern '{pattern_id}' has no event pattern companion "
                    f"with an edge table; motif analysis requires one.",
                )
            return graph_pid
        if pattern.pattern_type == "event" and self._storage.has_edge_table(pattern_id):
            return pattern_id
        raise GDSNavigationError(
            f"Pattern '{pattern_id}' has no graph companion; motif analysis "
            f"requires an event pattern with an edge table.",
        )

    def _enumerate_fan_out(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        min_k: int = 3,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Hub → k distinct targets within a sliding time window.

        Delegates to the in-memory adjacency enumerator. Single-seed cost on
        a high-out-degree hub is dominated by ``adj.neighbors_out`` dict
        lookups instead of a Lance ``read_edges`` scan.
        """
        if min_k < 2:
            raise GDSNavigationError(
                f"fan_out min_k must be >= 2, got {min_k}.",
            )
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        return self._enum_fan_out_via_adj(seed, adj, window_sec, min_k)

    def _enumerate_cycle_2(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        counterparty: str | None = None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Bidirectional pair (A ↔ B) with both directions within ``time_window_hours``.

        Delegates to the in-memory adjacency enumerator. ``counterparty`` filter
        applied post-delegate to preserve the public signature.
        """
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        results = self._enum_cycle_2_via_adj(seed, adj, window_sec)
        if counterparty is not None:
            results = [r for r in results if r.get("counterparty") == counterparty]
        return results

    def _enumerate_cycle_3(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        max_triads: int = 50,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Directed triad A→B→C→A with strictly monotonic timestamps within window.

        Delegates to the in-memory adjacency enumerator.
        """
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        return self._enum_cycle_3_via_adj(seed, adj, window_sec, max_triads)

    def _enumerate_structuring(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        max_instances: int = 50,
        amt1_min: float = 10000.0,
        amt2_max: float = 10000.0,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Single-seed enumeration of structuring A→B→C→D.

        Delegates to the in-memory adjacency enumerator.
        """
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        return self._enum_structuring_via_adj(
            seed, adj, window_sec, amt1_min, amt2_max, max_instances,
        )

    def _existing_neighbors(
        self,
        primary_key: str,
        edge_pattern_id: str,
        bidirectional: bool = True,
        timestamp_max: float | None = None,
    ) -> set[str]:
        """Return set of entities already connected to ``primary_key`` via edge table.

        BTREE-indexed lookup. When ``bidirectional`` is True (default), inspects both
        outgoing and incoming edges.

        ``timestamp_max`` restricts the lookup to edges with
        ``timestamp <= timestamp_max`` — used by hold-out evaluation to
        reproduce the as-of state of the graph at a given point in time.
        """
        existing: set[str] = set()
        fwd = self._storage.read_edges(
            edge_pattern_id,
            from_keys=[primary_key],
            timestamp_to=timestamp_max,
            columns=["to_key"],
        )
        if fwd.num_rows > 0:
            existing.update(fwd["to_key"].to_pylist())
        if bidirectional:
            rev = self._storage.read_edges(
                edge_pattern_id,
                to_keys=[primary_key],
                timestamp_to=timestamp_max,
                columns=["from_key"],
            )
            if rev.num_rows > 0:
                existing.update(rev["from_key"].to_pylist())
        existing.discard(primary_key)
        return existing

    def _load_trajectory_vector(
        self, primary_key: str, pattern_id: str,
    ) -> np.ndarray | None:
        """Load a single trajectory summary vector from the trajectory ANN index.

        Returns None when the index does not exist or the entity is missing.
        """
        result = self._load_trajectory_vectors_batch([primary_key], pattern_id)
        return result.get(primary_key)

    def _load_trajectory_vectors_batch(
        self, primary_keys: list[str], pattern_id: str,
    ) -> dict[str, np.ndarray]:
        """Batch-load trajectory summary vectors via single Lance scan.

        Returns a dict ``{primary_key: vector}``. Missing keys (no temporal
        history) are absent from the result. Returns an empty dict when the
        trajectory index does not exist.
        """
        if not primary_keys:
            return {}
        import lance as _lance_local

        traj_path = (
            self._storage._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        )
        if not traj_path.exists():
            return {}
        escaped = ", ".join(
            f"'{k.replace(chr(39), chr(39)*2)}'" for k in primary_keys
        )
        try:
            ds = _lance_local.dataset(str(traj_path))
            scanner = ds.scanner(
                columns=["primary_key", "trajectory_vector"],
                filter=f"primary_key IN ({escaped})",
            )
            tbl = scanner.to_table()
        except _NAVIGATION_RECOVERABLE_ERRORS:
            return {}
        if tbl.num_rows == 0:
            return {}
        keys = tbl["primary_key"].to_pylist()
        vectors = tbl["trajectory_vector"].to_pylist()
        return {
            keys[i]: np.asarray(vectors[i], dtype=np.float64)
            for i in range(tbl.num_rows)
        }

    def _enumerate_fan_in(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        min_k: int = 3,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """k distinct sources → sink within a sliding time window.

        Delegates to the in-memory adjacency enumerator.
        """
        if min_k < 2:
            raise GDSNavigationError(
                f"fan_in min_k must be >= 2, got {min_k}.",
            )
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        return self._enum_fan_in_via_adj(seed, adj, window_sec, min_k)

    # Per-k adaptive frontier cap. k=3,4 use generous cap; k>=5 tighten
    # progressively to bound worst-case latency on hub seeds without losing
    # recall on small-k investigations. Tuned against AML HI-Small FHPM
    # k-sweep measurements (benchmark/ibm-aml/profiling/2026-04-27-115-*).
    _CHAIN_K_MAX_FRONTIER_PER_K = {3: 1000, 4: 1000, 5: 500, 6: 250, 7: 125, 8: 100}
    _CHAIN_K_MAX_FRONTIER = 1000  # legacy fallback for k outside the table
    _CHAIN_K_MAX_RESULTS = 50

    def _enumerate_chain_k(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        k: int = 4,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Open directed chain A → B → ... of length k, no cycle closure.

        Delegates to the in-memory adjacency enumerator.
        """
        if k < 3 or k > 8:
            raise GDSNavigationError(
                f"chain_k requires 3 ≤ k ≤ 8, got k={k}.",
            )
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        max_frontier = self._CHAIN_K_MAX_FRONTIER_PER_K.get(k, self._CHAIN_K_MAX_FRONTIER)
        return self._enum_chain_k_via_adj(
            seed, adj, window_sec, k,
            max_frontier, self._CHAIN_K_MAX_RESULTS,
        )

    def _enumerate_split_recombine(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        min_k: int = 3,
        direction: str = "forward",
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Diamond topology: source → k intermediaries → single sink, stacked-bipartite.

        Delegates to the in-memory adjacency-path enumerator. Single-seed cost
        on a high-out-degree hub is dominated by ``adj.neighbors_out`` lookups
        (O(degree) dict iterations) instead of a Lance batched scan over every
        intermediary's out-edges, which collapses tail latency on hub seeds.
        """
        if direction not in ("forward", "backward"):
            raise GDSNavigationError(
                f"split_recombine direction must be 'forward' or 'backward', got {direction!r}.",
            )
        if min_k < 2:
            raise GDSNavigationError(
                f"split_recombine min_k must be >= 2, got {min_k}.",
            )
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        return self._enum_split_recombine_via_adj(
            seed, adj, window_sec, min_k, direction,
        )

    def _enumerate_bipartite_burst(
        self,
        seed: str,
        pattern_id: str,
        time_window_hours: int,
        min_k: int = 3,
        min_m: int = 3,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Complete K_{k,m} bipartite subgraph in a tight time window.

        Delegates to the in-memory adjacency-path enumerator. Single-seed cost
        is dominated by ``adj.neighbors_in`` / ``adj.neighbors_out`` lookups
        instead of Lance batched scans over every candidate source's out-edges,
        which collapses tail latency on hub seeds with high in/out degree.
        """
        if min_k < 2 or min_m < 2:
            raise GDSNavigationError(
                f"bipartite_burst requires min_k >= 2 and min_m >= 2, got min_k={min_k}, min_m={min_m}.",
            )
        graph_pid = self._resolve_motif_graph_pid(pattern_id)
        adj = self._storage.get_adjacency(graph_pid)
        window_sec = float(time_window_hours) * 3600.0
        return self._enum_bipartite_burst_via_adj(
            seed, adj, window_sec, min_k, min_m,
        )

    def find_witness_cohort(
        self,
        primary_key: str,
        pattern_id: str,
        top_n: int = 10,
        *,
        config: WitnessCohortConfig | None = None,
        edge_pattern_id: str | None = None,
    ) -> WitnessCohortResult:
        """Rank entities that share ``primary_key``'s witness signature.

        **Investigative peer ranking, not edge forecasting.** Surfaces
        entities that share the target's anomaly signature and are likely
        to belong to the same investigative cohort. The function does NOT
        forecast which entities will form future edges.

        Combines four signals into a composite score in [0, 1]:

        * ``delta_similarity``: ``exp(-distance / theta_norm)`` — absolute,
          population-scaled mapping from ANN distance, independent of pool size
        * ``witness_overlap``: Jaccard on the two witness dimension label sets
        * ``trajectory_alignment``: cosine similarity (remapped to [0, 1]) on
          trajectory vectors. The whole component is enabled or disabled once
          per call: when the reference entity has a trajectory vector, every
          candidate gets a number (0.5 when its own trajectory is missing).
          When the reference has no vector, the trajectory component is removed
          for the entire call and weights renormalize across the remaining
          three signals.
        * ``anomaly_bonus``: graded by ``delta_rank_pct / 100`` — a candidate
          at the 99th percentile contributes much more than one at the 90th

        Candidates already connected to ``primary_key`` via the resolved edge
        table are excluded. This filter is the function's main contribution
        over plain ANN: existing counterparties (often legitimate) are removed
        so the cohort is denser in unknown peers worth investigating. When
        ``config.bidirectional_check`` is True (default) both outgoing and
        incoming edges count as existing. ``config.timestamp_cutoff`` further
        restricts the existing-edge filter to edges with
        ``timestamp <= cutoff`` — used by hold-out evaluation to reproduce
        the as-of state of the graph at a given point in time.

        ``edge_pattern_id`` overrides the auto-resolved event pattern; use this
        when multiple event patterns share the same anchor and you want a
        specific one. The override is validated to actually point at an
        existing edge table.

        When the target entity is not anomalous, its witness set is empty and
        the witness component degrades to 0 for every candidate. The summary
        carries ``target_witness_size`` so callers can detect this situation.

        Raises:
            KeyError: ``primary_key`` is not present in ``pattern_id``.
            ValueError: ``pattern_id`` is not an anchor pattern, no event
                pattern with an edge table covers this anchor, or an explicit
                ``edge_pattern_id`` does not have an edge table.
        """
        cfg = config or WitnessCohortConfig()
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise ValueError(f"Pattern '{pattern_id}' not found in sphere")
        if pattern.pattern_type != "anchor":
            raise ValueError(
                f"find_witness_cohort requires an anchor pattern, "
                f"but '{pattern_id}' is type '{pattern.pattern_type}'"
            )

        if edge_pattern_id is not None:
            if not self._storage.has_edge_table(edge_pattern_id):
                raise ValueError(
                    f"Explicit edge_pattern_id '{edge_pattern_id}' has no edge "
                    "table. Either pass a different pattern or omit the override "
                    "and let auto-resolution pick one."
                )
            resolved_edge_pattern = edge_pattern_id
        else:
            resolved_edge_pattern = self._resolve_edge_pattern_for_anchor(pattern_id)

        if resolved_edge_pattern is None:
            raise ValueError(
                f"No event pattern with edge table covers anchor '{pattern_id}'. "
                "Build a sphere with an event pattern referencing this anchor "
                "and edge_table config, or pass edge_pattern_id explicitly."
            )

        version = self._resolve_version(pattern_id)
        dim_labels = pattern.dim_labels
        theta_norm = (
            float(np.linalg.norm(pattern.theta))
            if pattern.theta is not None else 0.0
        )
        # Guard against zero theta — fall back to 1.0 to keep delta_sim defined
        theta_scale = theta_norm if theta_norm > 1e-9 else 1.0

        # Reference entity: delta + witness
        ref_table = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key,
            columns=["delta", "delta_norm", "is_anomaly", "delta_rank_pct"],
        )
        if ref_table.num_rows == 0:
            raise KeyError(
                f"Entity '{primary_key}' not found in {pattern_id} v{version}"
            )
        ref_delta = np.asarray(ref_table["delta"][0].as_py(), dtype=np.float64)
        ref_is_anomaly = bool(ref_table["is_anomaly"][0].as_py())
        ref_witness_struct = self._engine.witness_set(
            ref_delta, theta_norm, dim_labels,
        )
        ref_witness = {d["label"] for d in ref_witness_struct.get("witness_dims", [])}
        target_witness_size = len(ref_witness)

        # Trajectory feature for the reference entity (auto-detect when None)
        trajectory_index_present = (
            self._storage._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        ).exists()
        if cfg.use_trajectory is None:
            use_trajectory_requested = trajectory_index_present
        else:
            use_trajectory_requested = cfg.use_trajectory and trajectory_index_present

        ref_trajectory: np.ndarray | None = None
        if use_trajectory_requested:
            ref_trajectory = self._load_trajectory_vector(primary_key, pattern_id)

        # The trajectory component is in for the entire call iff we successfully
        # loaded a reference trajectory vector. Per-candidate decisions could
        # otherwise produce mixed renormalization and incomparable scores.
        trajectory_active = ref_trajectory is not None

        # ANN candidates — over-fetch the configured pool
        ann_results = self._engine.find_nearest(
            ref_delta=np.asarray(ref_delta, dtype=np.float32),
            pattern_id=pattern_id,
            version=version,
            top_n=cfg.candidate_pool,
            exclude_keys={primary_key},
        )
        candidate_pool_size = len(ann_results)

        # Edge exclusion via BTREE lookup
        existing = self._existing_neighbors(
            primary_key,
            resolved_edge_pattern,
            bidirectional=cfg.bidirectional_check,
            timestamp_max=cfg.timestamp_cutoff,
        )
        ann_results = [(k, d) for k, d in ann_results if k not in existing]
        excluded_existing_edges = candidate_pool_size - len(ann_results)

        weights = cfg.weights.as_dict()

        if not ann_results:
            return WitnessCohortResult(
                primary_key=primary_key,
                pattern_id=pattern_id,
                edge_pattern_id=resolved_edge_pattern,
                members=[],
                excluded_existing_edges=excluded_existing_edges,
                excluded_low_score=0,
                candidate_pool_size=candidate_pool_size,
                weights_used=weights,
                summary={
                    "max_score": 0.0, "mean_score": 0.0,
                    "anomaly_count": 0,
                    "trajectory_used": trajectory_active,
                    "target_witness_size": target_witness_size,
                    "target_is_anomaly": ref_is_anomaly,
                },
            )

        candidate_keys = [k for k, _ in ann_results]
        distance_map = dict(ann_results)

        # Batch fetch candidate geometry
        cand_geo = self._storage.read_geometry(
            pattern_id, version,
            point_keys=candidate_keys,
            columns=["primary_key", "delta", "delta_norm", "is_anomaly", "delta_rank_pct"],
        )
        cand_pk = cand_geo["primary_key"].to_pylist()
        cand_delta = cand_geo["delta"].to_pylist()
        cand_norm = cand_geo["delta_norm"].to_pylist()
        cand_anom = cand_geo["is_anomaly"].to_pylist()
        cand_rank = cand_geo["delta_rank_pct"].to_pylist()
        geo_row = {
            cand_pk[i]: (cand_delta[i], cand_norm[i], cand_anom[i], cand_rank[i])
            for i in range(cand_geo.num_rows)
        }

        # Batch trajectory load — single Lance scan instead of per-candidate
        cand_trajectories: dict[str, np.ndarray] = {}
        if trajectory_active:
            cand_trajectories = self._load_trajectory_vectors_batch(
                candidate_keys, pattern_id,
            )

        scored: list[CohortMember] = []
        excluded_low_score = 0

        for cand_key in candidate_keys:
            row = geo_row.get(cand_key)
            if row is None:
                continue
            delta_y, delta_norm_y, is_anom_y, rank_y = row
            delta_arr = np.asarray(delta_y, dtype=np.float64)

            # Component 1 — absolute delta similarity (no pool dependency)
            distance = float(distance_map.get(cand_key, 0.0))
            delta_sim = float(np.exp(-distance / theta_scale))

            # Component 2 — witness overlap
            cand_witness_struct = self._engine.witness_set(
                delta_arr, theta_norm, dim_labels,
            )
            cand_witness = {
                d["label"] for d in cand_witness_struct.get("witness_dims", [])
            }
            witness_overlap = self._engine.witness_jaccard(ref_witness, cand_witness)

            if witness_overlap < cfg.min_witness_overlap:
                excluded_low_score += 1
                continue

            # Component 3 — trajectory alignment.
            # When trajectory_active is True, every candidate gets a number
            # (neutral 0.5 when its own trajectory is missing) so weights are
            # consistent across the whole result set. When False, all candidates
            # uniformly skip the trajectory component and weights renormalize.
            if trajectory_active:
                cand_traj = cand_trajectories.get(cand_key)
                if cand_traj is None:
                    traj_align = 0.5
                else:
                    traj_align = self._engine.trajectory_cosine(
                        ref_trajectory, cand_traj,
                    )
            else:
                traj_align = None

            # Component 4 — graded anomaly bonus by percentile rank
            anomaly_bonus = max(0.0, min(1.0, float(rank_y) / 100.0))

            score, components = self._engine.composite_link_score(
                delta_similarity=delta_sim,
                witness_overlap=witness_overlap,
                trajectory_alignment=traj_align,
                anomaly_bonus=anomaly_bonus,
                weights=weights,
            )

            if score < cfg.min_score:
                excluded_low_score += 1
                continue

            shared = ", ".join(sorted(ref_witness & cand_witness)) or "—"
            traj_phrase = (
                f", trajectory alignment {traj_align:.2f}"
                if traj_align is not None else ""
            )
            anom_phrase = " (anomalous)" if is_anom_y else ""
            explanation = (
                f"delta similarity {delta_sim:.2f}, "
                f"witness overlap {witness_overlap:.2f} (shared: {shared})"
                f"{traj_phrase}{anom_phrase}"
            )

            scored.append(CohortMember(
                primary_key=cand_key,
                score=round(score, 4),
                delta_similarity=round(delta_sim, 4),
                witness_overlap=round(witness_overlap, 4),
                trajectory_alignment=(
                    round(traj_align, 4) if traj_align is not None else None
                ),
                is_anomaly=bool(is_anom_y),
                delta_rank_pct=round(float(rank_y), 2),
                explanation=explanation,
                component_scores={k: round(v, 4) for k, v in components.items()},
            ))

        scored.sort(key=lambda m: m.score, reverse=True)
        top_members = scored[:top_n]

        anomaly_count = sum(1 for m in top_members if m.is_anomaly)
        max_score = top_members[0].score if top_members else 0.0
        mean_score = (
            sum(m.score for m in top_members) / len(top_members)
            if top_members else 0.0
        )

        return WitnessCohortResult(
            primary_key=primary_key,
            pattern_id=pattern_id,
            edge_pattern_id=resolved_edge_pattern,
            members=top_members,
            excluded_existing_edges=excluded_existing_edges,
            excluded_low_score=excluded_low_score,
            candidate_pool_size=candidate_pool_size,
            weights_used=weights,
            summary={
                "max_score": round(max_score, 4),
                "mean_score": round(mean_score, 4),
                "anomaly_count": anomaly_count,
                "trajectory_used": trajectory_active,
                "target_witness_size": target_witness_size,
                "target_is_anomaly": ref_is_anomaly,
            },
        )

    def solid_forecast(
        self,
        primary_key: str,
        pattern_id: str,
        current_delta_norm: float | None = None,
    ) -> dict[str, Any] | None:
        """Forecast anomaly status for an entity's solid.

        Returns None if the solid has fewer than 3 slices (insufficient history).
        When current_delta_norm is provided, it is used for current_is_anomaly
        instead of the last slice norm.
        """
        solid = self._engine.build_solid(primary_key, pattern_id, self._manifest)
        if len(solid.slices) < 3:
            return None

        from hypertopos.engine.forecast import check_stale_forecast, forecast_anomaly_status

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        deltas = [s.delta_snapshot for s in solid.slices]
        af = forecast_anomaly_status(
            deltas,
            pattern.theta_norm,
            horizon=1,
            current_delta_norm=current_delta_norm,
        )
        forecast: dict[str, Any] = {
            "horizon": af.horizon,
            "predicted_delta_norm": round(af.predicted_delta_norm, 4),
            "current_delta_norm": (
                round(float(current_delta_norm), 4)
                if current_delta_norm is not None else None
            ),
            "forecast_is_anomaly": af.forecast_is_anomaly,
            "current_is_anomaly": af.current_is_anomaly,
            "reliability": af.reliability,
        }
        last_ts = solid.slices[-1].timestamp
        if last_ts:
            forecast = check_stale_forecast(last_ts, forecast)
        return forecast

    def solid_reputation(self, primary_key: str, pattern_id: str) -> dict | None:
        """Compute reputation from entity's temporal history.

        Reads only delta_norm_snapshot from temporal data (avoids full build_solid).
        Returns None for event patterns or entities without temporal slices.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type != "anchor":
            return None
        theta_norm = float(np.linalg.norm(pattern.theta)) if pattern.theta is not None else 0.0
        try:
            temporal = self._storage.read_temporal(pattern_id, primary_key)
        except _NAVIGATION_RECOVERABLE_ERRORS:
            return None
        if temporal.num_rows == 0 or "delta_norm_snapshot" not in temporal.column_names:
            return None
        slice_norms = temporal.column("delta_norm_snapshot").to_numpy().astype(np.float32)
        from hypertopos.engine.geometry import GDSEngine as _GE
        return _GE.compute_reputation(slice_norms, theta_norm)

    def classify_anomalies(
        self, polygons: list[Polygon], pattern_id: str,
    ) -> list[dict]:
        """Classify anomalous polygons into labeled clusters."""
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        return self._engine.classify_anomalies(polygons, pattern)

    def _compute_hub_scores(
        self,
        table: Any,
        pattern: Any,
        line_id_filter: str | None,
    ) -> np.ndarray:
        """Compute hub scores from a geometry table. Returns float64 scores array."""
        if pattern.edge_max is not None:
            deltas = delta_matrix_from_arrow(table)
            sigma = np.maximum(pattern.sigma_diag, 1e-2)
            shape_matrix = np.clip(deltas * sigma + pattern.mu, 0.0, 1.0)
            if line_id_filter is not None:
                dim_idx = next(
                    (i for i, r in enumerate(pattern.relations)
                     if r.line_id == line_id_filter),
                    None,
                )
                if dim_idx is None:
                    raise GDSNavigationError(
                        f"line_id_filter '{line_id_filter}' not found in "
                        f"pattern '{pattern.pattern_id}' relations."
                    )
                return (shape_matrix[:, dim_idx] * pattern.edge_max[dim_idx]).astype(np.float64)
            return np.sum(shape_matrix * pattern.edge_max, axis=1).astype(np.float64)
        elif "edges" in table.schema.names:
            return np.array(
                [
                    sum(
                        1 for e in (table["edges"][i].as_py() or [])
                        if e.get("status") == "alive"
                        and (line_id_filter is None or e.get("line_id") == line_id_filter)
                    )
                    for i in range(table.num_rows)
                ],
                dtype=np.float64,
            )
        else:
            # Fallback: reconstruct from entity_keys + relations
            line_ids_col, _ = _table_edge_line_and_point_keys(
                table, pattern.relations,
            )
            return np.array(
                [
                    sum(
                        1 for lid in (row_lids or [])
                        if line_id_filter is None or lid == line_id_filter
                    )
                    for row_lids in line_ids_col
                ],
                dtype=np.float64,
            )

    def π7_attract_hub(
        self,
        pattern_id: str,
        top_n: int = 10,
        line_id_filter: str | None = None,
        fdr_alpha: float | None = None,
        fdr_method: str = "bh",
        p_value_method: str = "rank",
        select: str = "top_norm",
    ) -> list[tuple[str, int, float]]:
        """π7 — Find entities with highest geometric connectivity (hub score).

        Scans geometry and ranks entities by shape-vector footprint.
        Returns list of (primary_key, alive_edge_count, hub_score) sorted DESC.

        Continuous path (edge_max defined): numpy vectorized — shape = delta*sigma + mu,
        score = shape * edge_max → raw alive count.
        Binary fallback (no edge_max): parse edges struct → count alive edges.

        Use line_id_filter to rank by edges to a specific line only.
        """
        if fdr_method not in ("bh", "storey"):
            raise ValueError(
                f"fdr_method must be 'bh' or 'storey', got {fdr_method!r}"
            )
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]

        # Continuous path only needs primary_key + delta
        hub_columns: list[str] = (
            ["primary_key", "delta"] if pattern.edge_max is not None
            else ["primary_key", "edges", "entity_keys"]
        )
        table = self._storage.read_geometry(
            pattern_id, version, columns=hub_columns,
        )
        if table.num_rows == 0:
            return []

        keys = table["primary_key"].to_pylist()
        _hub_deltas: np.ndarray | None = None

        if pattern.edge_max is not None:
            # --- Continuous path: numpy vectorized ---
            deltas = delta_matrix_from_arrow(table)
            _hub_deltas = deltas
            sigma = np.maximum(pattern.sigma_diag, 1e-2)
            shape_matrix = np.clip(deltas * sigma + pattern.mu, 0.0, 1.0)

            if line_id_filter is not None:
                dim_idx = next(
                    (i for i, r in enumerate(pattern.relations)
                     if r.line_id == line_id_filter),
                    None,
                )
                if dim_idx is None:
                    raise GDSNavigationError(
                        f"line_id_filter '{line_id_filter}' not found in "
                        f"pattern '{pattern_id}' relations."
                    )
                scores = shape_matrix[:, dim_idx] * pattern.edge_max[dim_idx]
            else:
                scores = np.sum(shape_matrix * pattern.edge_max, axis=1)

            # Round to int for alive_edge_count
            edge_counts = np.rint(scores).astype(int)
            n = min(top_n, len(scores))
            top_indices = np.argpartition(scores, -n)[-n:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            results: list[tuple[str, int, float]] = [
                (keys[i], int(edge_counts[i]), float(scores[i]))
                for i in top_indices
            ]
        elif "edges" in table.schema.names:
            # --- Binary fallback: JSON edge count ---
            results = []
            for i in range(table.num_rows):
                bk = keys[i]
                edges = table["edges"][i].as_py() or []
                count = sum(
                    1 for e in edges
                    if e.get("status") == "alive"
                    and (line_id_filter is None or e.get("line_id") == line_id_filter)
                )
                results.append((bk, count, float(count)))
            results.sort(key=lambda r: r[2], reverse=True)
            results = results[:top_n]
        else:
            # --- entity_keys fallback: reconstruct from relations ---
            line_ids_col, _ = _table_edge_line_and_point_keys(
                table, pattern.relations,
            )
            results = []
            for i in range(table.num_rows):
                bk = keys[i]
                count = sum(
                    1 for lid in (line_ids_col[i] or [])
                    if line_id_filter is None or lid == line_id_filter
                )
                results.append((bk, count, float(count)))
            results.sort(key=lambda r: r[2], reverse=True)
            results = results[:top_n]

        # --- FDR filtering (opt-in) ---
        if fdr_alpha is not None and len(results) > 0:
            from hypertopos.engine.fdr import (
                benjamini_hochberg,
                empirical_p_values_from_rank,
            )
            N = len(results)
            # p-value from hub_score ranking: rank 1 (highest) → lowest p
            p_values = np.array(
                [(N - i) / N for i in range(N)], dtype=np.float64,
            )
            rejected, q_values = benjamini_hochberg(p_values, fdr_alpha, method=fdr_method)
            results = [
                r for r, keep in zip(results, rejected) if keep
            ]

        # --- Diverse selection (opt-in) ---
        if select not in ("top_norm", "diverse"):
            raise ValueError(f"unknown select mode: {select!r}")
        if select == "diverse" and len(results) > 0 and _hub_deltas is not None:
            from hypertopos.engine.selection import lazy_greedy_facility_location
            # Build delta matrix for the result keys
            key_to_idx = {k: i for i, k in enumerate(keys)}
            result_indices = [key_to_idx[r[0]] for r in results if r[0] in key_to_idx]
            if result_indices:
                delta_vectors = _hub_deltas[result_indices].astype(np.float64)
                k = min(top_n, len(delta_vectors))
                selected_idx, _ = lazy_greedy_facility_location(delta_vectors, k)
                results = [results[int(idx)] for idx in selected_idx]

        return results

    def hub_score_stats(
        self, pattern_id: str, line_id_filter: str | None = None
    ) -> dict:
        """Compute hub score distribution statistics.

        When line_id_filter is provided, scores are computed for that line only
        (same filtering logic as π7_attract_hub). Returns stats on the filtered
        score distribution so agents can correctly interpret top-N results.
        """
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        stats_columns: list[str] = (
            ["primary_key", "delta"] if pattern.edge_max is not None
            else ["primary_key", "edges", "entity_keys"]
        )
        table = self._storage.read_geometry(
            pattern_id, version, columns=stats_columns,
        )
        if table.num_rows == 0:
            return {
                "mean": 0.0, "std": 0.0,
                "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0,
                "max": 0.0, "total_entities": 0,
            }

        if pattern.edge_max is not None:
            deltas = delta_matrix_from_arrow(table)
            sigma = np.maximum(pattern.sigma_diag, 1e-2)
            shape_matrix = np.clip(deltas * sigma + pattern.mu, 0.0, 1.0)
            if line_id_filter is not None:
                dim_idx = next(
                    (i for i, r in enumerate(pattern.relations)
                     if r.line_id == line_id_filter),
                    None,
                )
                if dim_idx is None:
                    raise GDSNavigationError(
                        f"line_id_filter '{line_id_filter}' not found in "
                        f"pattern '{pattern_id}' relations."
                    )
                scores = (shape_matrix[:, dim_idx] * pattern.edge_max[dim_idx]).astype(np.float64)
            else:
                scores = np.sum(shape_matrix * pattern.edge_max, axis=1).astype(np.float64)
        elif "edges" in table.schema.names:
            scores = np.array(
                [
                    sum(
                        1 for e in (table["edges"][i].as_py() or [])
                        if e.get("status") == "alive"
                        and (line_id_filter is None or e.get("line_id") == line_id_filter)
                    )
                    for i in range(table.num_rows)
                ],
                dtype=np.float64,
            )
        else:
            # Fallback: reconstruct from entity_keys + relations
            line_ids_col, _ = _table_edge_line_and_point_keys(
                table, pattern.relations,
            )
            scores = np.array(
                [
                    sum(
                        1 for lid in (row_lids or [])
                        if line_id_filter is None or lid == line_id_filter
                    )
                    for row_lids in line_ids_col
                ],
                dtype=np.float64,
            )

        return {
            "mean": round(float(np.mean(scores)), 3),
            "std": round(float(np.std(scores)), 3),
            "p25": round(float(np.percentile(scores, 25)), 3),
            "p50": round(float(np.percentile(scores, 50)), 3),
            "p75": round(float(np.percentile(scores, 75)), 3),
            "p90": round(float(np.percentile(scores, 90)), 3),
            "p95": round(float(np.percentile(scores, 95)), 3),
            "max": round(float(np.max(scores)), 3),
            "total_entities": int(table.num_rows),
        }

    def π7_attract_hub_and_stats(
        self,
        pattern_id: str,
        top_n: int = 10,
        line_id_filter: str | None = None,
        fdr_alpha: float | None = None,
        fdr_method: str = "bh",
        p_value_method: str = "rank",
        select: str = "top_norm",
    ) -> tuple[list[tuple[str, int, float, float | None]], dict]:
        """π7 variant — returns (top_n_results, score_stats) in ONE geometry scan.

        Avoids the two-scan overhead of calling π7_attract_hub + hub_score_stats separately.

        Each result tuple is (primary_key, alive_edge_count, hub_score, hub_score_pct).
        hub_score_pct is the score as a percentage of max_hub_score (None in binary mode).
        stats dict includes max_hub_score for continuous patterns.
        """
        if fdr_method not in ("bh", "storey"):
            raise ValueError(
                f"fdr_method must be 'bh' or 'storey', got {fdr_method!r}"
            )
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        max_hub_score = pattern.max_hub_score
        hub_columns: list[str] = (
            ["primary_key", "delta"] if pattern.edge_max is not None
            else ["primary_key", "edges", "entity_keys"]
        )
        table = self._storage.read_geometry(pattern_id, version, columns=hub_columns)

        if table.num_rows == 0:
            empty_stats: dict[str, Any] = {
                "mean": 0.0, "std": 0.0,
                "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0,
                "max": 0.0, "total_entities": 0,
                "max_hub_score": max_hub_score,
            }
            return [], empty_stats

        keys = table["primary_key"].to_pylist()
        scores = self._compute_hub_scores(table, pattern, line_id_filter)
        _hub_deltas: np.ndarray | None = None
        if pattern.edge_max is not None:
            _hub_deltas = delta_matrix_from_arrow(table)

        # Top-N results
        if pattern.edge_max is not None:
            edge_counts = np.rint(scores).astype(int)
        else:
            edge_counts = scores.astype(int)
        n = min(top_n, len(scores))
        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        results = [
            (
                keys[i],
                int(edge_counts[i]),
                float(scores[i]),
                round(float(scores[i]) / max_hub_score * 100, 1) if max_hub_score else None,
            )
            for i in top_indices
        ]

        # --- FDR filtering (opt-in) ---
        if fdr_alpha is not None and len(results) > 0:
            from hypertopos.engine.fdr import (
                benjamini_hochberg,
                empirical_p_values_from_rank,
            )

            # Population rank percentile from the full scores array
            total = len(scores)
            rank_pcts = np.array(
                [float(np.sum(scores <= r[2])) / total * 100
                 for r in results],
                dtype=np.float64,
            )
            p_values = empirical_p_values_from_rank(rank_pcts)
            rejected, q_values = benjamini_hochberg(p_values, fdr_alpha, method=fdr_method)
            results = [
                r for r, keep in zip(results, rejected) if keep
            ]

        # --- Diverse selection (opt-in) ---
        if select not in ("top_norm", "diverse"):
            raise ValueError(f"unknown select mode: {select!r}")
        if select == "diverse" and len(results) > 0 and _hub_deltas is not None:
            from hypertopos.engine.selection import lazy_greedy_facility_location

            key_to_idx = {k: i for i, k in enumerate(keys)}
            result_indices = [key_to_idx[r[0]] for r in results if r[0] in key_to_idx]
            if result_indices:
                delta_vectors = _hub_deltas[result_indices].astype(np.float64)
                k = min(top_n, len(delta_vectors))
                selected_idx, _ = lazy_greedy_facility_location(delta_vectors, k)
                results = [results[int(idx)] for idx in selected_idx]

        # Stats (full population)
        stats: dict[str, Any] = {
            "mean": round(float(np.mean(scores)), 3),
            "std": round(float(np.std(scores)), 3),
            "p25": round(float(np.percentile(scores, 25)), 3),
            "p50": round(float(np.percentile(scores, 50)), 3),
            "p75": round(float(np.percentile(scores, 75)), 3),
            "p90": round(float(np.percentile(scores, 90)), 3),
            "p95": round(float(np.percentile(scores, 95)), 3),
            "max": round(float(np.max(scores)), 3),
            "total_entities": int(table.num_rows),
            "max_hub_score": max_hub_score,
        }
        return results, stats

    def hub_score_history(self, primary_key: str, pattern_id: str) -> list[dict]:
        """Hub score evolution per temporal slice. Returns [] in binary mode."""
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.edge_max is None:
            return []

        solid = self._engine.build_solid(primary_key, pattern_id, self._manifest)

        sigma = np.maximum(pattern.sigma_diag, 1e-2)

        def _score(delta: np.ndarray) -> tuple[float, int]:
            s = float(np.sum((delta * sigma + pattern.mu) * pattern.edge_max))
            return round(max(0.0, s), 3), max(0, int(round(s)))

        history = []
        for sl in solid.slices:
            sc, alive = _score(sl.delta_snapshot)
            history.append({
                "timestamp": sl.timestamp.isoformat(),
                "hub_score": sc,
                "alive_edges_est": alive,
                "deformation_type": sl.deformation_type,
                "changed_line_id": sl.changed_line_id,
                "delta_norm": round(float(sl.delta_norm_snapshot), 4),
            })

        base = solid.base_polygon
        # Read stored delta directly — build_polygon recomputes delta from Edge structs
        # which in continuous mode have point_key="" and yield wrong alive counts.
        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key,
            columns=["delta", "delta_norm"],
        )
        if geo.num_rows > 0:
            base_delta = np.array(geo["delta"][0].as_py(), dtype=np.float32)
            base_delta_norm = float(geo["delta_norm"][0].as_py())
        else:
            base_delta = base.delta
            base_delta_norm = base.delta_norm
        sc, alive = _score(base_delta)
        history.append({
            "timestamp": base.updated_at.isoformat(),
            "hub_score": sc,
            "alive_edges_est": alive,
            "deformation_type": "current",
            "changed_line_id": None,
            "delta_norm": round(base_delta_norm, 4),
        })

        return history

    @staticmethod
    def _us_to_iso(us: int) -> str:
        """Convert microseconds-since-epoch to ISO 8601 UTC string."""
        from datetime import datetime
        return datetime.fromtimestamp(us / 1_000_000, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )

    def π9_attract_drift(
        self,
        pattern_id: str,
        top_n: int = 10,
        sample_size: int | None = None,
        filters: dict[str, str | list[str]] | None = None,
        forecast_horizon: int | None = None,
        rank_by_dimension: str | None = None,
        fdr_alpha: float | None = None,
        fdr_method: str = "bh",
        p_value_method: str = "rank",
        select: str = "top_norm",
    ) -> list[dict]:
        """π9 — Find entities with highest geometric drift (temporal velocity and direction).

        Scans anchor pattern population, reads all temporal history in one pass,
        computes displacement (||delta_last - delta_first||), path length
        (Σ ||delta[i+1] - delta[i]||), and direction of drift relative to the
        null centre. Returns top_n ranked by displacement DESC.

        Two direction fields are attached per entity:
          - gradient_alignment ∈ [-1, 1] — radially-inward component of the
            drift vector. +1 means the entity is moving perfectly toward the
            null centre (normalising), -1 means moving perfectly away
            (deteriorating), 0 means tangential motion at constant radius.
          - drift_direction ∈ {"normalizing", "deteriorating", "neutral"} —
            label derived from gradient_alignment with soft cutoffs at ±0.3.

        Only works on anchor patterns — event patterns have no temporal history.
        Use filters={"timestamp_from": "2024-01-01", "timestamp_to": "2026-01-01"}
        to restrict to a time window.

        NOTE: displacement, displacement_current, path_length, and TAC are computed
        over structural dimensions only (pattern.relations), excluding prop_columns.
        Prop_columns (e.g. fashion_news_frequency) encode property presence as 0/1
        raw shape values; when a customer acquires a property between the first temporal
        slice and the current geometry, the resulting delta difference dominates the
        norm and produces artificially large displacement_current.  Excluding prop_columns
        keeps all four metrics focused on geometric/behavioural drift.
        dimension_diffs and dimension_diffs_current still include prop_columns as
        informational context so agents can see property acquisition separately.
        """
        if fdr_method not in ("bh", "storey"):
            raise ValueError(
                f"fdr_method must be 'bh' or 'storey', got {fdr_method!r}"
            )
        import random

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]

        if pattern.pattern_type == "event":
            raise ValueError(
                f"π9 requires anchor pattern — event patterns have no temporal "
                f"history. Got pattern '{pattern_id}' with type 'event'."
            )

        dim_names_early = [r.line_id for r in pattern.relations]
        rank_dim_index: int | None = None
        if rank_by_dimension is not None:
            if rank_by_dimension not in dim_names_early:
                raise GDSNavigationError(
                    f"rank_by_dimension='{rank_by_dimension}' not found in "
                    f"structural dimensions: {dim_names_early}"
                )
            rank_dim_index = dim_names_early.index(rank_by_dimension)

        _theta_norm = float(np.linalg.norm(pattern.theta)) if pattern.theta is not None else 0.0
        version = self._resolve_version(pattern_id)
        geo_table = self._storage.read_geometry(
            pattern_id, version, columns=["primary_key", "delta"]
        )
        if geo_table.num_rows == 0:
            return []

        # Extract base deltas via vectorized matrix build — avoids per-scalar .as_py()
        pk_col = geo_table["primary_key"].to_pylist()
        delta_matrix_geo = delta_matrix_from_arrow(geo_table)
        base_deltas: dict[str, np.ndarray] = dict(zip(pk_col, delta_matrix_geo, strict=False))

        keys = pk_col
        if sample_size is not None and sample_size < len(keys):
            keys = random.sample(keys, sample_size)

        dim_names = pattern.dim_labels or (
            [r.line_id for r in pattern.relations] + list(pattern.prop_columns)
        )

        # True streaming — consume batches without list() materialisation.
        # path_length requires all intermediate slice deltas, so a single Arrow
        # table is still built, but from a generator to avoid the double-buffer
        # overhead of list() + from_batches(list).
        import itertools

        # When sample_size is given, pass only the sampled keys to the temporal
        # reader so the Lance scanner skips rows for unsampled entities entirely
        # (avoids a full table scan that can cost 30 s on large spheres).
        temporal_keys = keys if (sample_size is not None and sample_size < len(pk_col)) else None
        batch_iter = self._storage.read_temporal_batched(
            pattern_id,
            timestamp_from=filters.get("timestamp_from") if filters else None,
            timestamp_to=filters.get("timestamp_to") if filters else None,
            keys=temporal_keys,
        )
        try:
            first_batch = next(batch_iter)
        except StopIteration:
            return []
        temporal_table = pa.Table.from_batches(
            itertools.chain([first_batch], batch_iter)
        )
        # Only apply remaining filters (e.g. "year") — timestamp bounds already
        # handled by read_temporal_batched predicate pushdown above.
        remaining_filters = {
            k: v for k, v in (filters or {}).items()
            if k not in ("timestamp_from", "timestamp_to")
        }
        if remaining_filters and temporal_table.num_rows > 0:
            temporal_table = self._storage._apply_temporal_filters(
                temporal_table, remaining_filters
            )

        if temporal_table.num_rows == 0:
            return []

        if "shape_snapshot" not in temporal_table.schema.names:
            raise GDSNavigationError(
                f"Temporal data for pattern '{pattern_id}' uses legacy schema "
                "(delta_snapshot). Run GDSWriter.migrate_temporal_to_shape_snapshot() "
                "to upgrade."
            )

        # Pre-sort once by (primary_key, timestamp) so groups are contiguous slices
        sorted_t = temporal_table.sort_by(
            [("primary_key", "ascending"), ("timestamp", "ascending")]
        )

        # Pre-extract all columns as Python lists — avoids slow per-scalar .as_py()
        # on timezone-aware timestamps (expensive on Windows without tzdata)
        t_pk: list[str] = sorted_t["primary_key"].to_pylist()
        t_shape: list[list[float]] = sorted_t["shape_snapshot"].to_pylist()
        # Cast timestamps to int64 (μs since epoch) before to_pylist — avoids tz lookup
        t_ts_us: list[int] = pc.cast(sorted_t["timestamp"], pa.int64()).to_pylist()

        # Build {pk: (start, end)} index ranges in the pre-sorted list
        groups_range: dict[str, tuple[int, int]] = {}
        prev_pk: str | None = None
        seg_start = 0
        for i, pk in enumerate(t_pk):
            if pk != prev_pk:
                if prev_pk is not None:
                    groups_range[prev_pk] = (seg_start, i)
                prev_pk = pk
                seg_start = i
        if prev_pk is not None:
            groups_range[prev_pk] = (seg_start, len(t_pk))

        # Structural dimension count — prop_columns excluded from all norm calculations
        # (displacement, displacement_current, path_length, TAC).  See docstring.
        n_rel = len(pattern.relations)

        from hypertopos.engine.geometry import GDSEngine as _GE
        from hypertopos.engine.geometry import _decomposition_scalars
        _compute_rep = _GE.compute_reputation

        # M3 decomposition pre-flight: load oldest+latest calibration epochs once
        # so the inline inner-loop computation just needs already-loaded shapes.
        # Failure here (single epoch, schema mismatch, missing fit, or storage
        # backend without multi-epoch retention) → 3 None additive fields per
        # entry, batch response stays intact.
        _decomp_fit_v1: CalibrationFit | None = None
        _decomp_fit_v2: CalibrationFit | None = None
        if hasattr(self._storage, "list_calibration_versions") and hasattr(
            self._storage, "read_calibration_fit"
        ):
            try:
                _decomp_versions = self._storage.list_calibration_versions(pattern_id)
                if len(_decomp_versions) >= 2:
                    _f1 = self._storage.read_calibration_fit(
                        pattern_id, version=_decomp_versions[0],
                    )
                    _f2 = self._storage.read_calibration_fit(
                        pattern_id, version=_decomp_versions[-1],
                    )
                    if _f1.schema_hash == _f2.schema_hash:
                        _decomp_fit_v1 = _f1
                        _decomp_fit_v2 = _f2
            except (GDSError, KeyError, OSError, ValueError):
                pass

        results: list[dict] = []

        for bk in keys:
            rng = groups_range.get(bk)
            if rng is None:
                continue
            start, end = rng
            n = end - start
            if n < 2:
                continue

            # Slice pre-extracted lists — no Arrow table creation in the hot loop
            _sigma = np.maximum(pattern.sigma_diag, 1e-2)
            all_shapes = np.array(t_shape[start:end], dtype=np.float32)  # shape (n, d)
            all_deltas = (all_shapes - pattern.mu) / _sigma
            delta_first = all_deltas[0]
            delta_last = all_deltas[-1]
            diff = delta_last - delta_first

            displacement = float(np.linalg.norm(diff[:n_rel]))
            base_delta = base_deltas[bk]
            diff_current = base_delta - delta_first
            displacement_current = float(np.linalg.norm(diff_current[:n_rel]))

            step_diffs = np.diff(all_deltas[:, :n_rel], axis=0)  # shape (n-1, n_rel)
            path_length = float(np.sqrt(np.einsum('ij,ij->i', step_diffs, step_diffs)).sum())
            ratio = displacement / path_length if path_length > 0 else 0.0

            # Gradient alignment: radially-inward component of the drift vector.
            # +1 → drift aimed at null centre (normalising),
            #  0 → tangential (constant radius),
            # -1 → pure outward drift (deteriorating).
            _first_norm = float(np.linalg.norm(delta_first[:n_rel]))
            if displacement > 1e-9 and _first_norm > 1e-9:
                _gradient_alignment = float(
                    -np.dot(diff[:n_rel], delta_first[:n_rel])
                    / (displacement * _first_norm)
                )
            else:
                _gradient_alignment = 0.0
            if _gradient_alignment > 0.3:
                _drift_direction = "normalizing"
            elif _gradient_alignment < -0.3:
                _drift_direction = "deteriorating"
            else:
                _drift_direction = "neutral"

            _rel_deltas = all_deltas[:, :n_rel]
            delta_norms = np.sqrt(np.einsum('ij,ij->i', _rel_deltas, _rel_deltas))
            if n >= 3:
                _mean = float(np.mean(delta_norms))
                tac: float | None = round(
                    1.0 - float(np.std(delta_norms)) / max(_mean, 1e-4), 4
                )
                tac = max(0.0, min(1.0, tac))
            else:
                tac = None

            rep = _compute_rep(delta_norms, _theta_norm)

            if _decomp_fit_v1 is not None and _decomp_fit_v2 is not None:
                _intr, _extr, _frac = _decomposition_scalars(
                    all_shapes[0], all_shapes[-1],
                    _decomp_fit_v1, _decomp_fit_v2,
                )
                _intrinsic_displacement = round(_intr, 4)
                _extrinsic_displacement = round(_extr, 4)
                _intrinsic_fraction: float | None = round(_frac, 4)
            else:
                _intrinsic_displacement = None
                _extrinsic_displacement = None
                _intrinsic_fraction = None

            results.append({
                "primary_key": bk,
                "displacement": round(displacement, 4),
                "displacement_current": round(displacement_current, 4),
                "dimension_diffs_current": {
                    name: round(float(diff_current[i]), 4)
                    for i, name in enumerate(dim_names[:n_rel])
                    if i < len(diff_current)
                },
                "prop_column_changes": {
                    name: (abs(float(diff_current[n_rel + j])) > 0.5)
                    for j, name in enumerate(pattern.prop_columns)
                    if (n_rel + j) < len(diff_current)
                },
                "path_length": round(path_length, 4),
                "ratio": round(ratio, 4),
                "gradient_alignment": round(_gradient_alignment, 4),
                "drift_direction": _drift_direction,
                "num_slices": n,
                "first_timestamp": self._us_to_iso(t_ts_us[start]),
                "last_timestamp": self._us_to_iso(t_ts_us[end - 1]),
                "delta_norm_first": round(float(np.linalg.norm(all_deltas[0])), 4),
                "delta_norm_last": round(float(np.linalg.norm(all_deltas[-1])), 4),
                "tac": tac,
                "reputation": rep["reputation"],
                "anomaly_tenure": rep["anomaly_tenure"],
                "dimension_diffs": {
                    name: round(float(diff[i]), 4)
                    for i, name in enumerate(dim_names[:n_rel])
                    if i < len(diff)
                },
                "intrinsic_displacement": _intrinsic_displacement,
                "extrinsic_displacement": _extrinsic_displacement,
                "intrinsic_fraction": _intrinsic_fraction,
            })

        if rank_dim_index is not None:
            results.sort(
                key=lambda r: abs(r["dimension_diffs"].get(rank_by_dimension, 0)),
                reverse=True,
            )
        else:
            results.sort(key=lambda r: r["displacement"], reverse=True)
        total_drift_population = len(results)
        results = results[:top_n]

        # Add slice_window_days to each entry
        for entry in results:
            from datetime import datetime as _dt

            first = _dt.fromisoformat(entry["first_timestamp"])
            last = _dt.fromisoformat(entry["last_timestamp"])
            entry["slice_window_days"] = (last - first).days

        # Optional forecast
        if forecast_horizon is not None:
            from hypertopos.engine.forecast import (
                check_stale_forecast,
                extrapolate_trajectory,
                forecast_segment_crossing,
            )

            planes: dict[str, Any] = {}
            if hasattr(sphere, "aliases"):
                planes = {
                    aid: a.filter.cutting_plane
                    for aid, a in sphere.aliases.items()
                    if a.base_pattern_id == pattern_id
                    and a.filter.cutting_plane is not None
                }
            for entry in results:
                pk = entry["primary_key"]
                try:
                    solid = self._engine.build_solid(
                        pk, pattern_id, self._manifest,
                    )
                except _NAVIGATION_RECOVERABLE_ERRORS:
                    continue
                if len(solid.slices) < 3:
                    continue
                deltas_arr = [s.delta_snapshot for s in solid.slices]
                traj = extrapolate_trajectory(
                    deltas_arr, horizon=forecast_horizon,
                )
                crossings = (
                    forecast_segment_crossing(
                        deltas_arr, planes, horizon=forecast_horizon,
                    )
                    if planes
                    else []
                )
                time_to_boundary = min(
                    (c.time_to_boundary for c in crossings
                     if c.time_to_boundary is not None),
                    default=None,
                )
                forecast = {
                    "predicted_delta_norm": round(
                        traj.predicted_delta_norm, 4,
                    ),
                    "time_to_boundary": time_to_boundary,
                    "reliability": traj.reliability,
                }
                from datetime import datetime as _dt2

                last_ts = _dt2.fromisoformat(entry["last_timestamp"])
                forecast = check_stale_forecast(last_ts, forecast)
                entry["drift_forecast"] = forecast

        # --- FDR filtering (opt-in) ---
        if fdr_alpha is not None and len(results) > 0:
            from hypertopos.engine.fdr import (
                benjamini_hochberg,
                empirical_p_values_from_rank,
            )
            # Population rank percentile: rank i (0-based) in sorted top
            # corresponds to position i in total_drift_population
            rank_pcts = np.array(
                [(total_drift_population - i) / total_drift_population * 100
                 for i in range(len(results))],
                dtype=np.float64,
            )
            p_values = empirical_p_values_from_rank(rank_pcts)
            rejected, q_values = benjamini_hochberg(p_values, fdr_alpha, method=fdr_method)
            for row, q in zip(results, q_values):
                row["q_value"] = float(q)
            results = [row for row, keep in zip(results, rejected) if keep]

        # --- Diverse selection (opt-in) ---
        if select not in ("top_norm", "diverse"):
            raise ValueError(f"unknown select mode: {select!r}")
        if select == "diverse" and len(results) > 0:
            from hypertopos.engine.selection import lazy_greedy_facility_location
            dim_keys = list(results[0].get("dimension_diffs", {}).keys())
            if dim_keys:
                delta_vectors = np.array(
                    [[r["dimension_diffs"].get(d, 0.0) for d in dim_keys] for r in results],
                    dtype=np.float64,
                )
                k = min(top_n, len(results))
                selected_idx, representativeness = lazy_greedy_facility_location(
                    delta_vectors, k,
                )
                out: list[dict] = []
                for i, idx in enumerate(selected_idx):
                    row = dict(results[int(idx)])
                    row["representativeness"] = int(representativeness[i])
                    out.append(row)
                results = out

        return results

    def find_drifting_similar(
        self,
        primary_key: str,
        pattern_id: str,
        top_n: int = 5,
    ) -> list[dict]:
        """Find entities that changed in a geometrically similar way.

        Uses ANN search over trajectory summary vectors (mean + std of all temporal
        deformations). Only meaningful for anchor patterns with temporal history.

        Returns list of dicts: {primary_key, distance, displacement, num_slices,
        first_timestamp, last_timestamp}, sorted by distance ascending.

        Raises ValueError if pattern is not an anchor type.
        Raises ValueError if trajectory index has not been built yet.
        """
        import lance as _lance_local

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise ValueError(f"Pattern '{pattern_id}' not found in sphere")
        if pattern.pattern_type != "anchor":
            raise ValueError(
                f"find_drifting_similar requires an anchor pattern, "
                f"but '{pattern_id}' is type '{pattern.pattern_type}'"
            )

        traj_path = self._storage._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        if not traj_path.exists():
            raise ValueError(
                f"Trajectory index not found for pattern '{pattern_id}'. "
                f"Run GDSWriter.build_trajectory_index('{pattern_id}') first."
            )

        traj_ds = _lance_local.dataset(str(traj_path))
        escaped = primary_key.replace("'", "''")
        query_row = traj_ds.scanner(
            filter=f"primary_key = '{escaped}'",
            columns=["trajectory_vector", "num_slices"],
        ).to_table()
        if query_row.num_rows == 0:
            raise ValueError(
                f"Entity '{primary_key}' has no trajectory data in pattern '{pattern_id}'. "
                f"Ensure the entity has at least one temporal deformation."
            )

        num_slices = int(query_row["num_slices"][0].as_py())
        if num_slices < 2:
            raise ValueError(
                f"insufficient_temporal_history: entity '{primary_key}' has {num_slices} "
                f"temporal slice — minimum 2 required (need a start and end point to define "
                f"a direction). To find entities with similar current shape use "
                f"find_similar_entities('{primary_key}', '{pattern_id}'). "
                f"To inspect the available slice use get_solid()."
            )

        query_vector = np.array(query_row["trajectory_vector"][0].as_py(), dtype=np.float32)

        results = self._storage.find_nearest_trajectory(
            pattern_id,
            query_vector,
            k=top_n + 1,  # +1 to account for self
            exclude_keys={primary_key},
        )
        if results is None:
            raise ValueError(
                f"Trajectory index not found for pattern '{pattern_id}'."
            )

        return results[:top_n]

    def π10_attract_trajectory(
        self,
        primary_key: str,
        pattern_id: str,
        top_n: int = 5,
    ) -> list[dict]:
        """π10 — Find entities with similar temporal trajectory (ANN search).

        Alias for ``find_drifting_similar``.
        """
        return self.find_drifting_similar(primary_key, pattern_id, top_n)

    def π8_attract_cluster(
        self,
        pattern_id: str,
        n_clusters: int = 5,
        top_n: int = 10,
        sample_size: int | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Discover intrinsic geometric archetypes in delta-space via k-means++.

        Returns cluster dicts sorted by size descending. Each dict:
        cluster_id, size, anomaly_rate, centroid_delta, delta_norm_mean,
        delta_norm_std, representative_key, dim_profile,
        member_keys (trimmed to top_n closest to centroid).
        """
        version = self._resolve_version(pattern_id)
        table = self._storage.read_geometry(
            pattern_id, version, sample_size=sample_size,
            columns=self._CLUSTER_COLUMNS,
        )
        if table.num_rows == 0:
            return []

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]

        keys: list[str] = table["primary_key"].to_pylist()
        is_anomaly_flags: list[bool] = table["is_anomaly"].to_pylist()
        delta_norms: list[float] = [float(v) for v in table["delta_norm"].to_pylist()]

        delta_matrix = delta_matrix_from_arrow(table)
        dim_names = pattern.dim_labels or (
            [r.line_id for r in pattern.relations] + list(pattern.prop_columns)
        )

        clusters = self._engine.find_clusters(
            delta_matrix=delta_matrix,
            keys=keys,
            is_anomaly_flags=is_anomaly_flags,
            delta_norms=delta_norms,
            n_clusters=n_clusters,
            dim_names=dim_names,
            seed=seed,
        )

        # Annotate auto-k metadata if auto-detection was used
        if n_clusters == 0:
            for cluster in clusters:
                cluster["auto_k"] = True

        for cluster in clusters:
            cluster["member_keys"] = cluster["member_keys"][:top_n]

        return clusters

    # ------------------------------------------------------------------
    # Observability methods
    # ------------------------------------------------------------------

    def _detect_geometry_mode(
        self, pattern_id: str, version: int, total: int
    ) -> str:
        """Detect whether geometry delta vectors are binary, continuous, or mixed.

        Samples up to 200 rows from the geometry dataset (delta column only)
        and counts unique values per dimension.

        Classification rules:
        - ALL dims have ≤ 3 unique values → "binary"
        - ALL dims have > 5 unique values → "continuous"
        - Otherwise → "mixed"

        Falls back to "continuous" if total == 0 or read fails.
        """
        if total == 0:
            return "continuous"
        try:
            tbl = self._storage.read_geometry(
                pattern_id, version, columns=["delta"], sample_size=200
            )
            if tbl is None or len(tbl) == 0:
                return "continuous"
            delta_col = tbl["delta"].combine_chunks()
            flat = delta_col.values.to_numpy(zero_copy_only=False)
            d = len(flat) // len(delta_col)
            if d == 0:
                return "continuous"
            matrix = flat.reshape(-1, d)
            unique_counts = [len(np.unique(matrix[:, i])) for i in range(d)]
            if all(u <= 3 for u in unique_counts):
                return "binary"
            if all(u > 5 for u in unique_counts):
                return "continuous"
            return "mixed"
        except _NAVIGATION_RECOVERABLE_ERRORS:
            return "continuous"

    def sphere_overview(self, pattern_id: str | None = None) -> list[dict]:
        """Return population-level summary for one or all patterns.

        The recommended first call for any agent entering a sphere cold.
        Uses pre-computed geometry stats when available (O(1)); falls back
        to count_geometry_rows scan (O(log n) index reads).

        Returns per pattern: pattern_id, pattern_type, total_entities,
        anomaly_rate, theta_norm, calibration_health, geometry_mode.
        """
        sphere = self._storage.read_sphere()
        pattern_ids = [pattern_id] if pattern_id else list(sphere.patterns.keys())
        results: list[dict] = []
        for pid in pattern_ids:
            version = self._resolve_version(pid)
            pattern = sphere.patterns[pid]
            theta_norm = round(float(np.linalg.norm(pattern.theta)), 4)
            stats = self._storage.read_geometry_stats(pid, version)
            if stats:
                total = stats["total_entities"]
            else:
                total = self._storage.count_geometry_rows(pid, version)
            # Count anomalies by delta_norm >= theta_norm to match
            # anomaly_summary / find_anomalies semantics.
            # The is_anomaly column uses per-group thetas for grouped
            # patterns, which diverges from the global theta_norm used
            # by find_anomalies — leading to contradictory counts.
            theta_norm_raw = float(np.linalg.norm(pattern.theta))
            if theta_norm_raw > 0.0:
                anomaly_count = self._storage.count_geometry_rows(
                    pid, version, filter=f"delta_norm >= {theta_norm_raw}"
                )
            else:
                anomaly_count = 0
            anomaly_rate = round(anomaly_count / total, 4) if total > 0 else 0.0
            anomaly_rate_source = "delta_norm_scan"
            calibration_health = _classify_calibration_health(anomaly_rate, total)
            geometry_mode = self._detect_geometry_mode(pid, version, total)
            entry: dict[str, Any] = {
                "pattern_id": pid,
                "pattern_type": pattern.pattern_type,
                "total_entities": total,
                "anomaly_rate": anomaly_rate,
                "anomaly_rate_source": anomaly_rate_source,
                "theta_norm": theta_norm,
                "calibration_health": calibration_health,
                "geometry_mode": geometry_mode,
            }

            # dimension_kinds summary — compact view of divergence families
            if pattern.dimension_kinds:
                from hypertopos.builder._bregman import format_kinds_summary
                entry["dimension_kinds"] = format_kinds_summary(pattern.dimension_kinds)

            # inactive_ratio from geometry_stats cache (no geometry scan)
            if (
                entry["pattern_type"] == "anchor"
                and stats
                and stats.get("inactive_ratio") is not None
            ):
                entry["inactive_ratio"] = stats["inactive_ratio"]

            # has_temporal — O(1) path existence check
            if (
                entry["pattern_type"] == "anchor"
                and hasattr(self._storage, "_base")
            ):
                temporal_path = (
                    self._storage._base / "temporal" / pid / "data.lance"
                )
                if temporal_path.exists():
                    entry["has_temporal"] = True

            results.append(entry)
        return results

    def temporal_quality_summary(
        self,
        pattern_id: str,
        max_sample: int = 1000,
    ) -> dict | None:
        """Compute temporal anomaly persistence metrics for a pattern.

        Measures how stable anomaly status is across consecutive temporal
        slices.  Returns None if pattern has no temporal data or is an
        event pattern.

        Returns dict with:
          persistence_rate  — fraction of anomaly→anomaly transitions
          transition_rate   — fraction of anomaly→normal transitions
          signal_quality    — "persistent" | "volatile" | "mixed"
          n_entities_sampled — how many entities were evaluated
          n_anomaly_transitions — total anomaly→X transitions counted
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]

        if pattern.pattern_type == "event":
            return None

        theta_norm = float(np.linalg.norm(pattern.theta))
        if theta_norm <= 0:
            return {
                "persistence_rate": 0.0,
                "transition_rate": 0.0,
                "signal_quality": "no_anomalies",
                "n_entities_sampled": 0,
                "n_anomaly_transitions": 0,
            }

        temporal_table = self._storage.read_temporal_batch(pattern_id)
        if temporal_table.num_rows == 0:
            return None

        if "shape_snapshot" not in temporal_table.schema.names:
            return None  # Legacy schema, can't compute

        sigma = np.maximum(
            np.array(pattern.sigma_diag, dtype=np.float64), 1e-2,
        )
        mu = np.array(pattern.mu, dtype=np.float64)

        sorted_t = temporal_table.sort_by([
            ("primary_key", "ascending"),
            ("timestamp", "ascending"),
        ])
        pks = sorted_t["primary_key"].to_pylist()
        shapes = sorted_t["shape_snapshot"].to_pylist()

        # Group by primary_key (data is sorted)
        groups: dict[str, list[list[float]]] = {}
        prev_pk = None
        seg_start = 0
        for i, pk in enumerate(pks):
            if pk != prev_pk:
                if prev_pk is not None and (i - seg_start) >= 2:
                    groups[prev_pk] = shapes[seg_start:i]
                prev_pk = pk
                seg_start = i
        if prev_pk is not None and (len(pks) - seg_start) >= 2:
            groups[prev_pk] = shapes[seg_start:]

        if not groups:
            return None

        # Sample up to max_sample entities
        keys = list(groups.keys())
        if len(keys) > max_sample:
            rng = np.random.default_rng(42)
            chosen = rng.choice(len(keys), max_sample, replace=False)
            keys = [keys[i] for i in chosen]

        persist_count = 0
        flip_count = 0

        for key in keys:
            shape_list = groups[key]
            arr = np.array(shape_list, dtype=np.float64)
            deltas = (arr - mu) / sigma
            norms = np.sqrt(np.einsum('ij,ij->i', deltas, deltas))
            is_anom = norms >= theta_norm

            for j in range(len(is_anom) - 1):
                if is_anom[j]:
                    if is_anom[j + 1]:
                        persist_count += 1
                    else:
                        flip_count += 1

        all_pairs = persist_count + flip_count
        persistence_rate = persist_count / all_pairs if all_pairs > 0 else 0.0
        transition_rate = flip_count / all_pairs if all_pairs > 0 else 0.0

        if persistence_rate > 0.7:
            quality = "persistent"
        elif persistence_rate < 0.3:
            quality = "volatile"
        else:
            quality = "mixed"

        return {
            "persistence_rate": round(persistence_rate, 4),
            "transition_rate": round(transition_rate, 4),
            "signal_quality": quality,
            "n_entities_sampled": len(keys),
            "n_anomaly_transitions": all_pairs,
        }

    def _compute_event_rate_divergence(self) -> list[dict]:
        """Find entities with high event anomaly rate but below-theta static geometry.

        For each (anchor, event) pattern pair sharing a line:
        - Reads a single event geometry sample (columns: is_anomaly, entity_keys)
        - Accumulates per-anchor-entity: total event count and anomalous event count
        - Flags entities where event_anomaly_rate > 15% AND anchor delta_norm < theta
          (entities invisible to find_anomalies but with concentrated temporal anomalies)

        Uses one geometry read per (event_pid, anchor_line) pair to ensure rates are
        consistent (anom/total from the same sample, never > 1.0).

        Returns at most 20 alerts sorted by event_anomaly_rate descending.
        """
        _RATE_THRESHOLD = 0.15
        _SAMPLE_SIZE = 200000
        _MIN_EVENTS = 5  # skip entities with too few sampled events to avoid noise
        _MAX_ALERTS = 20

        sphere = self._storage.read_sphere()
        alerts: list[dict] = []

        # Build (event_pid, anchor_pid, anchor_line, anchor_idx) pairs
        pairs: list[tuple[str, str, str, int]] = []
        for anchor_pid, anchor_pat in sphere.patterns.items():
            if anchor_pat.pattern_type != "anchor":
                continue
            anchor_line = sphere.entity_line(anchor_pid)
            if not anchor_line:
                continue
            for event_pid, event_pat in sphere.patterns.items():
                if event_pat.pattern_type != "event":
                    continue
                relation_lines = [r.line_id for r in event_pat.relations]
                if anchor_line not in relation_lines:
                    continue
                pairs.append((event_pid, anchor_pid, anchor_line, relation_lines.index(anchor_line)))

        for event_pid, anchor_pid, anchor_line, anchor_idx in pairs:
            anchor_pat = sphere.patterns[anchor_pid]
            theta_norm = round(float(np.linalg.norm(anchor_pat.theta)), 4)
            if theta_norm <= 0:
                continue

            try:
                anchor_version = self._resolve_version(anchor_pid)
                event_version = self._resolve_version(event_pid)
            except (KeyError, ValueError):
                continue

            # Single geometry read — consistent anom/total from same sample
            try:
                geo = self._storage.read_geometry(
                    event_pid, event_version,
                    sample_size=_SAMPLE_SIZE,
                    columns=["is_anomaly", "entity_keys"],
                )
            except (FileNotFoundError, OSError, KeyError):
                continue

            if geo.num_rows == 0:
                continue

            total_counts: dict[str, int] = {}
            anom_counts: dict[str, int] = {}
            for ek_val, is_anom_val in zip(
                geo["entity_keys"].to_pylist(),
                geo["is_anomaly"].to_pylist(),
            ):
                if not ek_val or len(ek_val) <= anchor_idx or not ek_val[anchor_idx]:
                    continue
                key = ek_val[anchor_idx]
                total_counts[key] = total_counts.get(key, 0) + 1
                if is_anom_val:
                    anom_counts[key] = anom_counts.get(key, 0) + 1

            high_rate: dict[str, float] = {}
            for key, total in total_counts.items():
                if total < _MIN_EVENTS:
                    continue
                rate = anom_counts.get(key, 0) / total
                if rate > _RATE_THRESHOLD:
                    high_rate[key] = round(rate, 4)

            if not high_rate:
                continue

            # Cross-reference with anchor geometry — only flag non-static-anomalies
            high_rate_keys = list(high_rate.keys())
            try:
                geo_table = self._storage.read_geometry(
                    anchor_pid, anchor_version,
                    point_keys=high_rate_keys,
                    columns=["primary_key", "delta_norm", "is_anomaly"],
                )
            except (FileNotFoundError, OSError, KeyError):
                continue

            for i in range(geo_table.num_rows):
                pk = geo_table["primary_key"][i].as_py()
                delta_norm = float(geo_table["delta_norm"][i].as_py())
                is_anomaly = bool(geo_table["is_anomaly"][i].as_py())
                if is_anomaly:
                    continue
                rate = high_rate.get(pk, 0.0)
                alerts.append({
                    "pattern_id": anchor_pid,
                    "event_pattern_id": event_pid,
                    "entity_key": pk,
                    "event_anomaly_rate": rate,
                    "delta_norm": round(delta_norm, 4),
                    "theta_norm": theta_norm,
                    "alert": (
                        f"high event anomaly rate ({int(rate * 100)}%) but normal static"
                        " geometry — investigate temporal"
                    ),
                })

        alerts.sort(key=lambda a: a["event_anomaly_rate"], reverse=True)
        return alerts[:_MAX_ALERTS]

    def _pi11_from_cache(
        self,
        pattern_id: str,
        cached: list[dict],
        window_a_from: str,
        window_a_to: str,
        window_b_from: str,
        window_b_to: str,
    ) -> dict:
        """Build pi11 result from pre-computed temporal centroid cache.

        Note: uses entity_count-weighted mean of pre-computed window centroids —
        an approximation of the exact per-entity mean from full temporal scan.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        n_rel = len(pattern.relations)
        dim_names = pattern.dim_labels or (
            [r.line_id for r in pattern.relations] + list(pattern.prop_columns)
        )

        def _parse_ts(s: str) -> datetime:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt

        wa_from = _parse_ts(window_a_from)
        wa_to = _parse_ts(window_a_to)
        wb_from = _parse_ts(window_b_from)
        wb_to = _parse_ts(window_b_to)

        def _weighted_centroid(
            ts_from: datetime, ts_to: datetime,
        ) -> tuple[np.ndarray | None, int]:
            """Weighted mean of cached centroids overlapping [ts_from, ts_to)."""
            total_weight = 0
            weighted_sum: np.ndarray | None = None
            for c in cached:
                ws = c["window_start"]
                we = c["window_end"]
                # Overlap check: [ws, we) intersects [ts_from, ts_to)
                if we <= ts_from or ws >= ts_to:
                    continue
                w = c["entity_count"]
                vec = np.asarray(c["centroid"], dtype=np.float64)
                if weighted_sum is None:
                    weighted_sum = vec * w
                else:
                    weighted_sum += vec * w
                total_weight += w
            if weighted_sum is None or total_weight == 0:
                return None, 0
            return weighted_sum / total_weight, total_weight

        c_a, n_a = _weighted_centroid(wa_from, wa_to)
        c_b, n_b = _weighted_centroid(wb_from, wb_to)

        if c_a is None or c_b is None:
            return {
                "pattern_id": pattern_id,
                "centroid_shift": None,
                "warning": "one or both windows have no temporal data",
                "window_a": {
                    "from": window_a_from, "to": window_a_to,
                    "entry_count": n_a,
                },
                "window_b": {
                    "from": window_b_from, "to": window_b_to,
                    "entry_count": n_b,
                },
                "cached": True,
            }

        shift = float(np.linalg.norm((c_b - c_a)[:n_rel]))
        dim_diffs = sorted(
            [
                {
                    "dimension": (
                        dim_names[i] if i < len(dim_names) else f"dim_{i}"
                    ),
                    "mean_a": round(float(c_a[i]), 6),
                    "mean_b": round(float(c_b[i]), 6),
                    "diff": round(float(c_b[i] - c_a[i]), 6),
                }
                for i in range(n_rel)
            ],
            key=lambda x: abs(x["diff"]),
            reverse=True,
        )
        return {
            "pattern_id": pattern_id,
            "window_a": {
                "from": window_a_from,
                "to": window_a_to,
                "entry_count": n_a,
            },
            "window_b": {
                "from": window_b_from,
                "to": window_b_to,
                "entry_count": n_b,
            },
            "centroid_shift": round(shift, 6),
            "top_changed_dimensions": dim_diffs[:5],
            "interpretation": (
                "significant drift" if shift > 0.5
                else "minor shift" if shift > 0.05
                else "stable"
            ),
            "cached": True,
        }

    def π11_attract_population_compare(
        self,
        pattern_id: str,
        window_a_from: str,
        window_a_to: str,
        window_b_from: str,
        window_b_to: str,
    ) -> dict:
        """π11 — Compare population geometry between two time windows.

        For each window, collects temporal deformation delta_snapshots and computes
        the population centroid. Returns centroid shift (L2), anomaly rate change,
        and per-dimension breakdown sorted by |diff| descending.

        Use for batch monitoring: 'did last ingestion change the population shape?'
        windows: ISO-8601 strings, half-open [from, to).

        Partition pruning is automatic — the reader derives year/month hints
        from timestamp_from/timestamp_to by inspecting the directory structure, so agents
        do not need to pass year/month keys explicitly.
        For sub-year precision (month, quarter, specific date range), use ISO-8601 bounds
        (half-open range: from inclusive, to exclusive):
          timestamp_from="2024-06-01", timestamp_to="2024-10-01"
        """
        # Fast path: use pre-computed centroid cache if available
        cached = self._storage.read_temporal_centroids(pattern_id)
        if cached is not None:
            return self._pi11_from_cache(
                pattern_id, cached,
                window_a_from, window_a_to, window_b_from, window_b_to,
            )

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        dim_names = pattern.dim_labels or (
            [r.line_id for r in pattern.relations] + list(pattern.prop_columns)
        )

        def _read_window(ts_from: str, ts_to: str) -> pa.Table:
            import itertools as _itertools
            batch_iter = self._storage.read_temporal_batched(
                pattern_id, timestamp_from=ts_from, timestamp_to=ts_to
            )
            try:
                first = next(batch_iter)
            except StopIteration:
                return pa.table({})
            return pa.Table.from_batches(_itertools.chain([first], batch_iter))

        def _centroid_stats(tbl: pa.Table) -> tuple:
            if tbl.num_rows == 0:
                return None, 0
            if "shape_snapshot" not in tbl.schema.names:
                raise GDSNavigationError(
                    f"Temporal data for pattern '{pattern_id}' uses legacy schema "
                    "(delta_snapshot). Run GDSWriter.migrate_temporal_to_shape_snapshot() "
                    "to upgrade."
                )
            shapes = tbl["shape_snapshot"].to_pylist()
            _sigma = np.maximum(pattern.sigma_diag, 1e-2)
            mat = (np.array(shapes, dtype=np.float32) - pattern.mu) / _sigma
            centroid = mat.mean(axis=0)
            return centroid, int(tbl.num_rows)

        tbl_a = _read_window(window_a_from, window_a_to)
        tbl_b = _read_window(window_b_from, window_b_to)
        c_a, n_a = _centroid_stats(tbl_a)
        c_b, n_b = _centroid_stats(tbl_b)

        if c_a is None or c_b is None:
            return {
                "pattern_id": pattern_id,
                "centroid_shift": None,
                "warning": "one or both windows have no temporal data",
                "window_a": {"from": window_a_from, "to": window_a_to, "entry_count": n_a},
                "window_b": {"from": window_b_from, "to": window_b_to, "entry_count": n_b},
            }

        n_rel = len(pattern.relations)
        shift = float(np.linalg.norm((c_b - c_a)[:n_rel]))
        dim_diffs = sorted(
            [
                {
                    "dimension": dim_names[i] if i < len(dim_names) else f"dim_{i}",
                    "mean_a": round(float(c_a[i]), 6),
                    "mean_b": round(float(c_b[i]), 6),
                    "diff": round(float(c_b[i] - c_a[i]), 6),
                }
                for i in range(n_rel)
            ],
            key=lambda x: abs(x["diff"]),
            reverse=True,
        )
        return {
            "pattern_id": pattern_id,
            "window_a": {
                "from": window_a_from,
                "to": window_a_to,
                "entry_count": n_a,
            },
            "window_b": {
                "from": window_b_from,
                "to": window_b_to,
                "entry_count": n_b,
            },
            "centroid_shift": round(shift, 6),
            "top_changed_dimensions": dim_diffs[:5],
            "interpretation": (
                "significant drift" if shift > 0.5
                else "minor shift" if shift > 0.05
                else "stable"
            ),
        }

    def detect_data_quality_issues(
        self, pattern_id: str, sample_size: int | None = None,
    ) -> list[dict]:
        """Scan for data quality problems in a pattern's geometry.

        Checks:
        1. Coverage gaps — required relation lines with < 50% coverage
        2. Optional lines with near-zero coverage (< 5%)
        3. Degenerate polygons — delta_norm ~ 0 on > 10% of entities
        4. High anomaly rate — rate > 30% suggests miscalibrated theta or corrupted data
        5. Zero anomaly rate — anomaly_rate == 0% on > 1000 entities suggests theta miscalibration
        6. Theta ceiling — >50% of entities have delta_norm >= 0.75*theta
           (distribution massed at theta)
        7. Delta-norm mismatch — stored delta_norm != ||delta||
        8. Zero-variance prop columns — prop dims with sigma < 0.01

        Returns findings[] sorted by severity (HIGH first).
        Empty list = no issues detected.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        version = self._resolve_version(pattern_id)

        total = self._storage.count_geometry_rows(pattern_id, version)
        if total == 0:
            return [
                {
                    "issue_type": "empty_pattern",
                    "severity": "HIGH",
                    "count": 0,
                    "pct": 0.0,
                    "description": "Pattern has no geometry rows",
                }
            ]

        findings: list[dict] = []

        # ── 1/2. Coverage — scan edges/entity_keys and derive alive line_ids
        # When sample_size is set, read full table then sample for coverage scan.
        # Event patterns may lack the 'edges' column — filter to existing cols.
        _all_cov = ["edges", "entity_keys"]
        _schema_names = self._storage.geometry_column_names(pattern_id, version)
        _cov_cols = [c for c in _all_cov if c in _schema_names]
        _sampled = False
        _scan_total = total
        if sample_size is not None and total > sample_size:
            geo_table = self._storage.read_geometry(
                pattern_id, version, columns=_cov_cols,
            )
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(geo_table.num_rows, size=sample_size, replace=False))
            geo_table = geo_table.take(pa.array(idx, type=pa.int64()))
            _sampled = True
            _scan_total = sample_size

        rel_covered: dict[str, int] = {r.line_id: 0 for r in pattern.relations}
        if _sampled:
            for row_lines in _table_edge_line_ids(geo_table, pattern.relations):
                seen: set[str] = set()
                for lid in row_lines:
                    if lid in rel_covered and lid not in seen:
                        rel_covered[lid] += 1
                        seen.add(lid)
        else:
            for batch in self._storage.read_geometry_batched(
                pattern_id, version, columns=_cov_cols
            ):
                for row_lines in _table_edge_line_ids(batch, pattern.relations):
                    seen: set[str] = set()
                    for lid in row_lines:
                        if lid in rel_covered and lid not in seen:
                            rel_covered[lid] += 1
                            seen.add(lid)

        for rel in pattern.relations:
            covered = rel_covered.get(rel.line_id, 0)
            pct = covered / _scan_total
            if rel.required and pct < 0.5:
                findings.append({
                    "issue_type": "low_coverage",
                    "severity": "HIGH",
                    "line_id": rel.line_id,
                    "count": covered,
                    "pct": round(pct, 4),
                    "description": (
                        f"Required line '{rel.line_id}' has only {pct:.1%} coverage "
                        f"({covered}/{total} entities)"
                    ),
                })
            elif not rel.required and pct < 0.05:
                findings.append({
                    "issue_type": "low_coverage",
                    "severity": "MEDIUM",
                    "line_id": rel.line_id,
                    "count": covered,
                    "pct": round(pct, 4),
                    "description": (
                        f"Optional line '{rel.line_id}' has near-zero coverage ({pct:.1%})"
                    ),
                })

        # ── 3. Degenerate polygons — BTREE index on delta_norm (O(log n))
        degenerate = self._storage.count_geometry_rows(
            pattern_id, version, filter="delta_norm < 0.0001"
        )
        if degenerate > 0 and degenerate / total > 0.1:
            findings.append({
                "issue_type": "degenerate_polygons",
                "severity": "MEDIUM",
                "count": degenerate,
                "pct": round(degenerate / total, 4),
                "description": (
                    f"{degenerate} entities ({degenerate / total:.1%}) have delta_norm ~ 0 "
                    "— possible zero-variance or missing data"
                ),
            })

        # ── 4. High anomaly rate — delta_norm >= theta_norm
        theta_norm = float(np.linalg.norm(pattern.theta))
        anomaly_count = self._storage.count_geometry_rows(
            pattern_id, version,
            filter=f"delta_norm >= {theta_norm}" if theta_norm > 0.0 else "is_anomaly = true",
        )
        anomaly_rate = anomaly_count / total
        if anomaly_rate > 0.3:
            findings.append({
                "issue_type": "high_anomaly_rate",
                "severity": "HIGH" if anomaly_rate > 0.5 else "MEDIUM",
                "count": anomaly_count,
                "pct": round(anomaly_rate, 4),
                "description": (
                    f"Anomaly rate is {anomaly_rate:.1%} — "
                    "suggests miscalibrated theta or corrupted data"
                ),
            })

        # ── 5. Zero anomaly rate on large population (theta miscalibration)
        if anomaly_count == 0 and total > 1000:
            findings.append({
                "issue_type": "zero_anomaly_rate",
                "severity": "MEDIUM",
                "count": 0,
                "pct": 0.0,
                "description": (
                    f"Anomaly rate is 0% on {total} entities — "
                    "theta may be set at population maximum, disabling anomaly detection. "
                    "Consider recalibrating with a lower percentile cutoff."
                ),
            })

        # ── 6. Theta ceiling distribution (>50% entities massed near theta)
        theta_norm = float(np.linalg.norm(pattern.theta))
        if theta_norm > 0:
            ceiling_threshold = 0.75 * theta_norm
            near_ceiling = self._storage.count_geometry_rows(
                pattern_id, version, filter=f"delta_norm >= {ceiling_threshold:.6f}"
            )
            if near_ceiling / total > 0.5:
                findings.append({
                    "issue_type": "theta_ceiling",
                    "severity": "MEDIUM",
                    "count": near_ceiling,
                    "pct": round(near_ceiling / total, 4),
                    "description": (
                        f"{near_ceiling / total:.1%} of entities have delta_norm >= 0.75*theta "
                        f"({near_ceiling}/{total}) — distribution is massed against theta. "
                        "Consider recalibrating at a lower percentile."
                    ),
                })

        # ── 7. Delta-norm mismatch — verify stored delta_norm == ||delta||.
        # Sample up to 10 entities; a mismatch means recompute_delta_rank_pct() was not
        # called after the geometry was written, leaving ANN search results unreliable.
        _SAMPLE_SIZE = 10
        sample_cols = ["primary_key", "delta", "delta_norm"]
        first_batch = next(
            self._storage.read_geometry_batched(pattern_id, version, columns=sample_cols),
            None,
        )
        if first_batch is not None and first_batch.num_rows > 0:
            for idx in range(min(_SAMPLE_SIZE, first_batch.num_rows)):
                pk = first_batch.column("primary_key")[idx].as_py()
                stored_norm = float(first_batch.column("delta_norm")[idx].as_py())
                delta_vec = np.array(first_batch.column("delta")[idx].as_py(), dtype=np.float32)
                actual_norm = float(np.linalg.norm(delta_vec))
                # Threshold 0.01 is absolute (not relative): a stale delta_norm column will
                # differ from ||delta|| by the full magnitude of the change, making relative
                # vs absolute moot. Float32 precision errors are < 1e-4, well below this threshold.
                if abs(actual_norm - stored_norm) > 0.01:
                    findings.append({
                        "issue_type": "delta_norm_mismatch",
                        "severity": "HIGH",
                        "entity": pk,
                        "stored_delta_norm": round(stored_norm, 4),
                        "actual_delta_norm": round(actual_norm, 4),
                        "message": (
                            f"stored delta_norm {stored_norm:.4f} does not match "
                            f"||delta|| {actual_norm:.4f} — geometry may have been written "
                            "without calling recompute_delta_rank_pct(). "
                            "ANN search results will be unreliable."
                        ),
                    })
                    break  # One example is enough to flag the issue

        # ── 8. Zero-variance prop columns — check actual delta variance on prop dims
        # NOTE: sigma metadata is floored by SIGMA_EPS_PROP (0.2), so checking
        # sigma < 0.01 never triggers for prop columns. Instead, sample actual
        # delta vectors and measure variance on each prop dimension.
        if pattern.prop_columns:
            n_rel = len(pattern.relations)
            _prop_sample_size = min(total, 500)
            try:
                _prop_geo = self._storage.read_geometry(
                    pattern_id, version, columns=["delta"],
                )
                if _prop_geo.num_rows > _prop_sample_size:
                    _rng = np.random.default_rng(42)
                    _idx = _rng.choice(
                        _prop_geo.num_rows, size=_prop_sample_size, replace=False,
                    )
                    _prop_geo = _prop_geo.take(pa.array(_idx, type=pa.int64()))
                _deltas = np.array(
                    _prop_geo["delta"].to_pylist(), dtype=np.float32,
                )
                for j, prop_name in enumerate(pattern.prop_columns):
                    dim_idx = n_rel + j
                    if dim_idx < _deltas.shape[1]:
                        dim_var = float(np.var(_deltas[:, dim_idx]))
                        if dim_var < 0.01:
                            findings.append({
                                "issue_type": "zero_variance_prop_column",
                                "severity": "MEDIUM",
                                "dimension": prop_name,
                                "dim_index": dim_idx,
                                "delta_variance": round(dim_var, 6),
                                "description": (
                                    f"Property column '{prop_name}' has near-zero "
                                    f"delta variance ({dim_var:.4f}) — this dimension "
                                    f"contributes no discriminative signal to delta "
                                    f"vectors. Consider removing it from "
                                    f"pattern.prop_columns to reduce noise."
                                ),
                            })
            except _NAVIGATION_RECOVERABLE_ERRORS:
                pass

        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda f: severity_order.get(f["severity"], 9))
        return findings

    def _pi12_from_cache(
        self,
        pattern_id: str,
        cached: list[dict],
        timestamp_from: str | None,
        timestamp_to: str | None,
        n_regimes: int,
    ) -> list[dict]:
        """Build pi12 result from pre-computed temporal centroid cache."""
        def _parse_ts(s: str) -> datetime:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt

        filtered = list(cached)
        if timestamp_from:
            dt_from = _parse_ts(timestamp_from)
            filtered = [c for c in filtered if c["window_start"] >= dt_from]
        if timestamp_to:
            dt_to = _parse_ts(timestamp_to)
            filtered = [c for c in filtered if c["window_end"] <= dt_to]

        if len(filtered) < 2:
            return []

        filtered.sort(key=lambda c: c["window_start"])

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        n_rel = len(pattern.relations)

        shifts: list[tuple[datetime, float, np.ndarray]] = []
        for i in range(1, len(filtered)):
            c_prev = np.array(filtered[i - 1]["centroid"][:n_rel], dtype=np.float64)
            c_curr = np.array(filtered[i]["centroid"][:n_rel], dtype=np.float64)
            diff = c_curr - c_prev
            mag = float(np.linalg.norm(diff))
            shifts.append((filtered[i]["window_start"], mag, diff))

        if not shifts:
            return []

        mags = [s[1] for s in shifts]
        mean_m = float(np.mean(mags))
        std_m = float(np.std(mags)) if len(mags) > 2 else mean_m * 0.5
        threshold = mean_m + 1.5 * std_m

        changepoints: list[dict] = []
        for ts, mag, diff in shifts:
            if mag <= threshold:
                continue
            top_dims = sorted(
                [
                    {
                        "dimension": pattern.relations[j].line_id,
                        "diff": round(float(diff[j]), 6),
                    }
                    for j in range(n_rel)
                ],
                key=lambda d: abs(d["diff"]),
                reverse=True,
            )
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            changepoints.append({
                "timestamp": ts_str,
                "magnitude": round(mag, 6),
                "top_changed_dimensions": top_dims[:3],
                "description": (
                    f"Population centroid shifted {mag:.3f} "
                    f"(threshold {threshold:.3f})"
                ),
            })

        changepoints.sort(key=lambda c: c["magnitude"], reverse=True)
        return changepoints[:n_regimes]

    def π12_attract_regime_change(
        self,
        pattern_id: str,
        timestamp_from: str | None = None,
        timestamp_to: str | None = None,
        n_regimes: int = 3,
    ) -> list[dict]:
        """π12 — Detect when population geometry shifted significantly (changepoint detection).

        Aggregates temporal deformation entries into time buckets, computes rolling
        population centroid per bucket, and returns buckets where shift > mean + 1.5σ.

        anchor patterns only — event patterns have no temporal history.
        Returns list of {timestamp, magnitude, top_changed_dimensions, description},
        sorted by magnitude descending, capped at n_regimes.

        timestamp_from / timestamp_to: optional ISO-8601 bounds to limit the scan range.
        Partition pruning is automatic — no need to pass year/month keys explicitly.
        For sub-year precision (month, quarter, specific date range), use ISO-8601 bounds
        (half-open range: from inclusive, to exclusive).
        """
        # Fast path: use pre-computed centroid cache if available
        cached = self._storage.read_temporal_centroids(pattern_id)
        if cached is not None:
            return self._pi12_from_cache(
                pattern_id, cached, timestamp_from, timestamp_to, n_regimes,
            )

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type == "event":
            raise ValueError(
                f"pi12 requires anchor pattern — '{pattern_id}' has type 'event'."
            )

        dim_names = pattern.dim_labels or (
            [r.line_id for r in pattern.relations] + list(pattern.prop_columns)
        )
        n_rel = len(pattern.relations)

        import itertools
        batch_iter = self._storage.read_temporal_batched(
            pattern_id,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
        )
        try:
            first_batch = next(batch_iter)
        except StopIteration:
            return [{"warning": "no temporal data found for the given range"}]
        temporal_table = pa.Table.from_batches(itertools.chain([first_batch], batch_iter))
        if temporal_table.num_rows < 4:
            n = temporal_table.num_rows
            return [{"warning": f"insufficient temporal data: {n} entries (minimum 4 required)"}]

        if "shape_snapshot" not in temporal_table.schema.names:
            raise GDSNavigationError(
                f"Temporal data for pattern '{pattern_id}' uses legacy schema "
                "(delta_snapshot). Run GDSWriter.migrate_temporal_to_shape_snapshot() "
                "to upgrade."
            )

        sorted_t = temporal_table.sort_by([("timestamp", "ascending")])
        timestamps = sorted_t["timestamp"].to_pylist()
        _sigma = np.maximum(pattern.sigma_diag, 1e-2)
        shapes = sorted_t["shape_snapshot"].to_pylist()
        _shapes_mat = np.array(shapes, dtype=np.float32)  # (n, d)
        _deltas_mat = (_shapes_mat - pattern.mu) / _sigma  # vectorised
        deltas = _deltas_mat.tolist()  # list of lists for downstream code

        # Time-based bucketing: divide the temporal span into equal-duration intervals.
        # Count-based bucketing (n // (n_regimes+1)) produces buckets that are too
        # coarse when the data spans a long history — a concentrated changepoint in a
        # short window gets diluted into a large bucket and its signal falls below the
        # detection threshold.
        n_buckets = max(4, n_regimes * 4)
        t_min = timestamps[0].timestamp()
        t_max = timestamps[-1].timestamp()
        if t_max == t_min:
            return [{"warning": "all temporal entries share the same timestamp — cannot detect changes"}]  # noqa: E501

        bucket_width = (t_max - t_min) / n_buckets

        # Pre-compute float timestamps once (data is already sorted via sorted_t).
        # Use searchsorted to locate bucket boundaries in O(log n) per bucket instead
        # of O(n) boolean masking — total O(n_buckets * log n) vs O(n * n_buckets).
        ts_floats = np.array([ts.timestamp() for ts in timestamps], dtype=np.float64)

        bucket_centroids: list[np.ndarray] = []
        bucket_timestamps: list = []
        for b in range(n_buckets):
            t_start = t_min + b * bucket_width
            t_end = t_min + (b + 1) * bucket_width
            # Include the upper bound only for the last bucket so every entry is captured.
            upper = t_max + 1.0 if b == n_buckets - 1 else t_end
            left_idx = int(np.searchsorted(ts_floats, t_start, side="left"))
            right_idx = int(np.searchsorted(ts_floats, upper, side="right"))
            idxs = list(range(left_idx, right_idx))
            if not idxs:
                continue
            mat = np.array([deltas[i] for i in idxs], dtype=np.float32)
            bucket_centroids.append(mat.mean(axis=0))
            bucket_timestamps.append(timestamps[idxs[-1]])

        if len(bucket_centroids) < 2:
            n_b = len(bucket_centroids)
            msg = f"only {n_b} non-empty time bucket(s) — need at least 2 to compute shifts"
            return [{"warning": msg}]

        shifts = [
            float(np.linalg.norm(bucket_centroids[i + 1] - bucket_centroids[i]))
            for i in range(len(bucket_centroids) - 1)
        ]
        mean_shift = float(np.mean(shifts))
        std_shift = float(np.std(shifts))
        threshold = mean_shift + 1.5 * std_shift

        changes: list[dict] = []
        for i, shift in enumerate(shifts):
            if shift <= threshold:
                continue
            diff_vec = bucket_centroids[i + 1] - bucket_centroids[i]
            dim_diffs = sorted(
                [
                    {
                        "dimension": dim_names[j] if j < len(dim_names) else f"dim_{j}",
                        "diff": round(float(diff_vec[j]), 6),
                    }
                    for j in range(n_rel)
                ],
                key=lambda x: abs(x["diff"]),
                reverse=True,
            )
            ts = bucket_timestamps[i + 1]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            changes.append(
                {
                    "timestamp": ts_str,
                    "magnitude": round(shift, 6),
                    "top_changed_dimensions": dim_diffs[:3],
                    "description": (
                        f"Population centroid shifted {shift:.3f} "
                        f"(threshold {threshold:.3f})"
                    ),
                }
            )

        changes.sort(key=lambda x: x["magnitude"], reverse=True)
        if not changes:
            return [{
                "warning": (
                    "no_regime_changes_detected: all bucket shifts fell below the detection "
                    f"threshold (mean+1.5\u03c3 = {threshold:.4f}). Temporal data may be too sparse "  # noqa: E501
                    "for reliable changepoint detection. Use compare_time_windows() for "
                    "aggregate shift comparison."
                )
            }]
        return changes[:n_regimes]

    def line_geometry_stats(
        self, line_id: str, pattern_id: str, sample_size: int | None = None,
    ) -> dict:
        """Return geometric statistics for one relation line within a pattern.

        coverage_pct: fraction of entities with >= 1 alive edge to this line.
        edge_distribution: {0, 1, 2, 3+} — how many entities have exactly N alive edges.
        mean_delta_contribution: mean z-scored delta on this line's dimension,
            averaged over entities whose delta vector includes this dimension.
        required: from pattern definition.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        rel = next((r for r in pattern.relations if r.line_id == line_id), None)
        if rel is None:
            raise ValueError(
                f"Line '{line_id}' is not a relation in pattern '{pattern_id}'"
            )
        version = self._resolve_version(pattern_id)
        total = self._storage.count_geometry_rows(pattern_id, version)
        if total == 0:
            return {
                "line_id": line_id,
                "pattern_id": pattern_id,
                "total_entities": 0,
                "coverage_pct": 0.0,
                "edge_distribution": {"0": 0, "1": 0, "2": 0, "3+": 0},
                "mean_delta_contribution": 0.0,
                "required": rel.required,
            }

        rel_idx = next(i for i, r in enumerate(pattern.relations) if r.line_id == line_id)
        dist: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3+": 0}

        # Pass 1: coverage + edge distribution — edges or entity_keys
        _cov_cols = ["edges", "entity_keys"]  # reader drops missing columns
        _sampled = False
        _scan_total = total

        def _count_line(row_line_ids: list[str]) -> None:
            c = row_line_ids.count(line_id)
            if c == 0:
                dist["0"] += 1
            elif c == 1:
                dist["1"] += 1
            elif c == 2:
                dist["2"] += 1
            else:
                dist["3+"] += 1

        if sample_size is not None and total > sample_size:
            geo_table = self._storage.read_geometry(
                pattern_id, version, columns=_cov_cols,
            )
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(geo_table.num_rows, size=sample_size, replace=False))
            geo_table = geo_table.take(pa.array(idx, type=pa.int64()))
            _sampled = True
            _scan_total = sample_size
            for row_line_ids in _table_edge_line_ids(geo_table, pattern.relations):
                _count_line(row_line_ids)
        else:
            for batch in self._storage.read_geometry_batched(
                pattern_id, version, columns=_cov_cols
            ):
                for row_line_ids in _table_edge_line_ids(batch, pattern.relations):
                    _count_line(row_line_ids)

        # Pass 2: mean_delta_contribution — sampled from first batch only (cheap)
        delta_sum = 0.0
        delta_n = 0
        first_batch = next(
            self._storage.read_geometry_batched(
                pattern_id, version, columns=["delta"]
            ),
            None,
        )
        if first_batch is not None:
            for d_vec in first_batch.column("delta").to_pylist():
                if d_vec is not None and rel_idx < len(d_vec):
                    delta_sum += abs(float(d_vec[rel_idx]))
                    delta_n += 1

        covered = _scan_total - dist["0"]
        mean_delta = (delta_sum / delta_n) if delta_n > 0 else 0.0

        result = {
            "line_id": line_id,
            "pattern_id": pattern_id,
            "total_entities": total,
            "coverage_pct": round(covered / _scan_total, 4),
            "edge_distribution": dist,
            "mean_delta_contribution": round(mean_delta, 6),
            "required": rel.required,
        }
        if _sampled:
            result["sampled"] = True
            result["sample_size"] = sample_size
        return result

    # ===================================================================
    # check_alerts — implicit geometric health checks
    # ===================================================================

    def check_alerts(
        self, pattern_id: str | None = None
    ) -> dict[str, Any]:
        """Evaluate implicit geometric health checks across patterns.

        Runs 6 checks per pattern and returns a sorted list of alerts
        (HIGH first, then MEDIUM). Designed for batch monitoring: call
        with pattern_id=None to scan every pattern in the manifest.

        Returns::

            {
                "alerts": [...],          # list of alert dicts
                "patterns_checked": int,
            }

        Each alert::

            {
                "severity": "HIGH" | "MEDIUM",
                "check_type": str,
                "pattern_id": str,
                "message": str,
                "details": dict,
            }
        """
        sphere = self._storage.read_sphere()
        if pattern_id is not None:
            pattern_ids = [pattern_id]
        else:
            pattern_ids = [
                pid
                for pid in sphere.patterns
                if self._manifest.pattern_version(pid) is not None
            ]

        all_alerts: list[dict[str, Any]] = []
        for pid in pattern_ids:
            version = self._manifest.pattern_version(pid)
            if version is None:
                continue
            current_stats = self._storage.read_geometry_stats(pid, version)
            if current_stats is None:
                continue
            prev_stats = self._storage.read_geometry_stats(pid, version - 1)

            all_alerts.extend(
                self._check_anomaly_rate_spike(pid, current_stats, prev_stats)
            )
            all_alerts.extend(
                self._check_population_size_shock(pid, current_stats, prev_stats)
            )
            all_alerts.extend(self._check_data_quality(pid, current_stats))
            all_alerts.extend(
                self._check_regime_changepoint(pid, current_stats)
            )
            all_alerts.extend(self._check_calibration_staleness(pid))

        severity_order = {"HIGH": 0, "MEDIUM": 1}
        all_alerts.sort(key=lambda a: severity_order.get(a["severity"], 2))
        return {
            "alerts": all_alerts,
            "patterns_checked": len(pattern_ids),
            "limitations": [
                "coverage_gap and theta_ceiling checks require a full geometry scan"
                " — use detect_data_quality_issues() for the complete diagnostic",
            ],
        }

    # -- private check helpers ------------------------------------------

    def _check_anomaly_rate_spike(
        self,
        pattern_id: str,
        current: dict,
        prev: dict | None,
    ) -> list[dict[str, Any]]:
        """HIGH if anomaly rate increased > 5 pp vs previous version."""
        if prev is None:
            return []
        cur_total = current.get("total_entities", 0)
        prev_total = prev.get("total_entities", 0)
        if cur_total == 0 or prev_total == 0:
            return []
        cur_rate = current.get("total_anomalies", 0) / cur_total
        prev_rate = prev.get("total_anomalies", 0) / prev_total
        diff_pp = cur_rate - prev_rate
        if diff_pp > 0.05:
            return [
                {
                    "severity": "HIGH",
                    "check_type": "anomaly_rate_spike",
                    "pattern_id": pattern_id,
                    "message": (
                        f"Anomaly rate jumped from {prev_rate:.1%} to "
                        f"{cur_rate:.1%} (+{diff_pp:.1f} pp)"
                    ),
                    "details": {
                        "current_rate": round(cur_rate, 4),
                        "previous_rate": round(prev_rate, 4),
                        "diff_pp": round(diff_pp, 4),
                    },
                }
            ]
        return []

    def _check_population_size_shock(
        self,
        pattern_id: str,
        current: dict,
        prev: dict | None,
    ) -> list[dict[str, Any]]:
        """HIGH if |population change| > 10% vs previous version."""
        if prev is None:
            return []
        cur_total = current.get("total_entities", 0)
        prev_total = prev.get("total_entities", 0)
        if prev_total == 0:
            return []
        change_pct = (cur_total - prev_total) / prev_total
        if abs(change_pct) > 0.10:
            direction = "grew" if change_pct > 0 else "shrank"
            return [
                {
                    "severity": "HIGH",
                    "check_type": "population_size_shock",
                    "pattern_id": pattern_id,
                    "message": (
                        f"Population {direction} by {abs(change_pct):.1%} "
                        f"({prev_total} -> {cur_total})"
                    ),
                    "details": {
                        "current_total": cur_total,
                        "previous_total": prev_total,
                        "change_pct": round(change_pct, 4),
                    },
                }
            ]
        return []

    def _check_data_quality(
        self, pattern_id: str, current_stats: dict
    ) -> list[dict[str, Any]]:
        """Check geometric health using pre-computed geometry_stats.

        Uses current_stats (from geometry_stats.json) to avoid full geometry
        scans.  Coverage-gap checks (which require per-entity edge data) are
        skipped here — use detect_data_quality_issues() for the full audit.

        NOTE: Coverage-gap and theta_ceiling checks are not performed here
        (require per-entity geometry scan). Use detect_data_quality_issues()
        for the full diagnostic.
        """
        alerts: list[dict[str, Any]] = []
        total_entities = current_stats.get("total_entities", 0)
        total_anomalies = current_stats.get("total_anomalies", 0)
        theta_norm = current_stats.get("theta_norm", 0.0)

        # HIGH: anomaly rate > 50%; MEDIUM: > 30% (mirrors detect_data_quality_issues)
        if total_entities > 0:
            anomaly_rate = total_anomalies / total_entities
            if anomaly_rate > 0.30:
                severity = "HIGH" if anomaly_rate > 0.50 else "MEDIUM"
                alerts.append(
                    {
                        "severity": severity,
                        "check_type": "data_quality_high_anomaly_rate",
                        "pattern_id": pattern_id,
                        "message": (
                            f"High anomaly rate {anomaly_rate:.1%} "
                            f"({total_anomalies}/{total_entities} entities)"
                        ),
                        "details": {
                            "issue_type": "high_anomaly_rate",
                            "anomaly_rate": round(anomaly_rate, 4),
                            "total_anomalies": total_anomalies,
                            "total_entities": total_entities,
                        },
                    }
                )

        # MEDIUM: theta miscalibration (zero anomalies or theta=0) on large population
        # Note: zero_anomaly_rate is only meaningful when theta>0 — if theta=0, Bug 1 fix
        # guarantees zero anomalies (is_anomaly requires theta>0), so the finding would
        # be misleading noise. The theta_miscalibration finding already covers that case.
        theta_findings: list[dict] = []
        if total_entities > 1000 and total_anomalies == 0 and theta_norm > 0:
            theta_findings.append(
                {
                    "issue_type": "zero_anomaly_rate",
                    "severity": "MEDIUM",
                    "description": (
                        "No anomalies detected on a large population — "
                        "theta may be miscalibrated or recalibration is needed"
                    ),
                }
            )
        actual_rate = total_anomalies / total_entities if total_entities > 0 else 0
        if total_entities > 1000 and actual_rate > 0.20:
            theta_findings.append(
                {
                    "issue_type": "high_anomaly_rate",
                    "severity": "HIGH",
                    "description": (
                        f"Anomaly rate {actual_rate:.1%} exceeds 20% — "
                        f"theta is likely undercalibrated. Consider increasing "
                        f"anomaly_percentile or recalibrating."
                    ),
                    "actual_rate": round(actual_rate, 4),
                }
            )
        if total_entities > 1000 and theta_norm == 0:
            theta_findings.append(
                {
                    "issue_type": "theta_miscalibration",
                    "severity": "MEDIUM",
                    "description": (
                        "theta_norm is 0 on a large population — "
                        "geometry calibration has not been run or has reset"
                    ),
                }
            )
        if theta_findings:
            alerts.append(
                {
                    "severity": "MEDIUM",
                    "check_type": "theta_miscalibration",
                    "pattern_id": pattern_id,
                    "message": (
                        f"{len(theta_findings)} theta-related issue(s) detected"
                    ),
                    "details": {"findings": theta_findings},
                }
            )
        return alerts


    def _check_regime_changepoint(
        self,
        pattern_id: str,
        current: dict,
    ) -> list[dict[str, Any]]:
        """HIGH if any regime changepoints detected in the last 90 days.

        Bounded to the last 90 days so the check is fast on large spheres:
        spheres built from a single historical batch return no temporal entries
        in that window and pi12 exits immediately (< 4 rows).
        """
        cutoff = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        try:
            changes = self.π12_attract_regime_change(
                pattern_id, n_regimes=2, timestamp_from=cutoff, timestamp_to=None,
            )
        except _NAVIGATION_RECOVERABLE_ERRORS:
            return []
        if not changes:
            return []
        # pi12 returns [{"warning": "..."}] when no temporal data exists —
        # filter these out so they don't become false HIGH alerts.
        real_changepoints = [c for c in changes if "warning" not in c]
        if not real_changepoints:
            return []
        return [
            {
                "severity": "HIGH",
                "check_type": "regime_changepoint",
                "pattern_id": pattern_id,
                "message": (
                    f"{len(real_changepoints)} regime changepoint(s) detected"
                ),
                "details": {"changepoints": real_changepoints},
            }
        ]

    # ------------------------------------------------------------------
    # B8 / B13 / B14 / B15 — new utility methods
    # ------------------------------------------------------------------

    def suggest_grouping_properties(self, pattern_id: str) -> list[str]:
        """Return string property columns available for group_by_property on this pattern."""
        import pyarrow.types as pat

        sphere = self._storage.read_sphere()
        entity_line_id = sphere.entity_line(pattern_id)
        if not entity_line_id:
            return []
        version = self._manifest.line_version(entity_line_id)
        if version is None:
            return []
        points = self._storage.read_points(entity_line_id, version)
        skip = {"primary_key", "version", "created_at", "changed_at", "status"}
        result: list[str] = []
        for col_name in points.column_names:
            if col_name in skip:
                continue
            col_type = points.schema.field(col_name).type
            if pat.is_string(col_type) or pat.is_large_string(col_type):
                result.append(col_name)
        return result

    def temporal_hint(self, primary_key: str, pattern_id: str) -> dict | None:
        """Return temporal summary: num_slices, last_timestamp. None if no data."""
        temporal = self._storage.read_temporal(pattern_id, primary_key)
        if temporal is None or temporal.num_rows == 0:
            return None
        last_ts = pc.max(temporal["timestamp"]).as_py()
        return {
            "num_slices": temporal.num_rows,
            "last_timestamp": last_ts.isoformat() if last_ts else None,
        }

    def search_entities(
        self, line_id: str, property_name: str, value: str, limit: int = 20,
    ) -> dict:
        """Search for entities by property value. Returns {total, returned, entities}."""
        sphere = self._storage.read_sphere()
        line = sphere.lines.get(line_id)
        if not line:
            raise GDSNavigationError(f"Line '{line_id}' not found")
        version = line.versions[-1]
        table = self._storage.read_points(line_id, version)

        if property_name not in table.column_names:
            raise GDSNavigationError(
                f"Property '{property_name}' not found. "
                f"Available: {table.column_names}"
            )

        col_type = table.schema.field(property_name).type

        if pa.types.is_boolean(col_type):
            cast_value = value.lower() in ("true", "1", "yes")
            mask = pc.equal(table[property_name], pa.scalar(cast_value))
        else:
            mask = pc.equal(table[property_name], pa.scalar(value, type=col_type))

        filtered = table.filter(mask)
        total = len(filtered)

        entities: list[dict] = []
        for row in filtered.slice(0, limit).to_pylist():
            pk = row.pop("primary_key", None)
            row.pop("version", None)
            row.pop("created_at", None)
            row.pop("changed_at", None)
            entities.append({
                "primary_key": pk,
                "status": row.pop("status", "active"),
                "properties": row,
            })

        return {"total": total, "returned": len(entities), "entities": entities}

    def alias_population_count(self, alias_id: str) -> int | None:
        """Count entities inside an alias segment. Returns None if no cutting_plane."""
        sphere = self._storage.read_sphere()
        alias = sphere.aliases.get(alias_id)
        if not alias:
            return None
        cp = alias.filter.cutting_plane if alias.filter else None
        if cp is None:
            return None
        pid = alias.base_pattern_id
        version = self._resolve_version(pid)
        geo = self._storage.read_geometry(pid, version, columns=["delta"])
        return self._engine.count_inside_alias(alias, geo)

    def _check_calibration_staleness(
        self, pattern_id: str,
    ) -> list[dict[str, Any]]:
        """Check if calibration drift exceeds thresholds."""
        tracker = self._storage.read_calibration_tracker(pattern_id)
        if tracker is None:
            return []
        alerts: list[dict[str, Any]] = []
        if tracker.is_blocked:
            alerts.append({
                "severity": "HIGH",
                "check_type": "calibration_drift",
                "pattern_id": pattern_id,
                "message": (
                    f"Calibration drift {tracker.drift_pct:.1%} exceeds hard "
                    f"threshold {tracker.hard_threshold:.1%}. Appends are "
                    f"blocked. Call recalibrate('{pattern_id}') to fix."
                ),
                "details": {
                    "drift_pct": round(tracker.drift_pct, 4),
                    "hard_threshold": tracker.hard_threshold,
                    "running_n": tracker.running_n,
                    "calibrated_n": tracker.calibrated_n,
                },
            })
        elif tracker.is_stale:
            alerts.append({
                "severity": "MEDIUM",
                "check_type": "calibration_drift",
                "pattern_id": pattern_id,
                "message": (
                    f"Calibration drift {tracker.drift_pct:.1%} exceeds soft "
                    f"threshold {tracker.soft_threshold:.1%}. Consider "
                    f"recalibrating."
                ),
                "details": {
                    "drift_pct": round(tracker.drift_pct, 4),
                    "soft_threshold": tracker.soft_threshold,
                    "running_n": tracker.running_n,
                },
            })
        return alerts

    # ------------------------------------------------------------------
    # Aggregation (delegates to engine.aggregation)
    # ------------------------------------------------------------------

    def aggregate(
        self,
        event_pattern_id: str,
        group_by_line: str,
        group_by_line_2: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Aggregate event polygons. Thin delegate to engine.aggregation.

        Accepts all parameters of ``engine.aggregation.aggregate`` as keyword
        arguments (metric, filters, geometry_filters, event_filters, etc.).
        """
        from hypertopos.engine.aggregation import aggregate as _agg

        sphere = self._storage.read_sphere()
        return _agg(
            self._storage,
            self._engine,
            sphere,
            self._manifest,
            event_pattern_id=event_pattern_id,
            group_by_line=group_by_line,
            group_by_line_2=group_by_line_2,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Cross-pattern entity profile
    # ------------------------------------------------------------------

    def _discover_pattern_map(
        self, home_line_id: str,
    ) -> dict[str, str]:
        """Classify each pattern as direct/composite/chain/none for this entity.

        Returns {pattern_id: key_type} where key_type is one of:
        - "direct": entity is primary key of this pattern's anchor line
        - "sibling": pattern's anchor line shares source_id with home line
        - "composite": entity appears in entity_keys of this pattern's geometry
        - "chain": entity appears in chain_keys of this pattern's points table
        """
        if home_line_id in self._cross_pattern_map:
            return self._cross_pattern_map[home_line_id]

        sphere = self._storage.read_sphere()
        result: dict[str, str] = {}

        sibling_ids = set(sphere.sibling_lines(home_line_id))

        for pat_id, pattern in sphere.patterns.items():
            entity_line_id = sphere.entity_line(pat_id)
            if entity_line_id == home_line_id:
                result[pat_id] = "direct"
                continue

            # Sibling lines share the same source_id (same primary keys)
            if entity_line_id in sibling_ids:
                result[pat_id] = "sibling"
                continue

            try:
                version = self._resolve_version(pat_id)
            except GDSNavigationError:
                continue

            # For event patterns, entity_line returns None — check event_line
            if not entity_line_id:
                entity_line_id = sphere.event_line(pat_id)

            if not entity_line_id:
                continue

            # Check if home_line is a declared relation of this pattern
            # (entity appears as edge in event/anchor polygons)
            has_relation = any(
                r.line_id == home_line_id for r in pattern.relations
            )
            if has_relation:
                result[pat_id] = "event_edge"
                continue

            # Check chain: anchor line has chain_keys column
            line_ver = self._manifest.line_version(entity_line_id) or 1
            try:
                pts = self._storage.read_points(entity_line_id, line_ver)
                if "chain_keys" in pts.schema.names:
                    result[pat_id] = "chain"
                    continue
            except (ValueError, FileNotFoundError):
                pass

            # Check composite: sample 1 row and check if primary_key
            # contains separator (e.g. "→"), indicating composite keys
            try:
                sample = self._storage.read_geometry(
                    pat_id, version, columns=["primary_key"],
                    sample_size=1,
                )
                if sample.num_rows > 0:
                    sample_pk = sample["primary_key"][0].as_py()
                    if "\u2192" in sample_pk:
                        result[pat_id] = "composite"
                        continue
            except _NAVIGATION_RECOVERABLE_ERRORS:
                logger.debug("Pattern check failed for %s", pat_id)

        self._cross_pattern_map[home_line_id] = result
        return result

    def _get_chain_reverse_index(
        self, line_id: str, version: int,
    ) -> dict[str, list[str]]:
        """Build or return cached entity_key → [chain_pks] mapping."""
        cache_key = (line_id, version)
        if cache_key in self._chain_reverse_index:
            return self._chain_reverse_index[cache_key]

        pts = self._storage.read_points(line_id, version)
        idx: dict[str, list[str]] = defaultdict(list)
        pk_col = pts["primary_key"].to_pylist()
        ck_col = pts["chain_keys"].to_pylist()
        for pk, ck in zip(pk_col, ck_col, strict=False):
            if ck:
                for k in ck.split(","):
                    idx[k].append(pk)
        result = dict(idx)
        self._chain_reverse_index[cache_key] = result
        return result

    def cross_pattern_profile(
        self,
        primary_key: str,
        line_id: str | None = None,
        max_related: int = 5,
    ) -> dict:
        """Gather anomaly status from all patterns this entity participates in.

        Returns dict with:
            primary_key: str
            line_id: str
            source_count: int — number of patterns flagging at least one anomaly
            total_patterns: int — number of patterns the entity participates in
            signals: {pattern_id: {key_type, is_anomaly, delta_norm,
                      delta_rank_pct, conformal_p, related_count,
                      anomalous_count, anomalous_keys}}
        """
        # Validate entity key - reject separator or SQL metacharacters
        if "\u2192" in primary_key or "'" in primary_key or "--" in primary_key:
            raise GDSNavigationError(
                f"Invalid characters in primary_key: {primary_key!r}"
            )

        sphere = self._storage.read_sphere()

        # Discover home line
        if line_id is None:
            for lid, line in sphere.lines.items():
                if line.line_role != "anchor" or lid.startswith("_d_"):
                    continue
                ver = self._manifest.line_version(lid) or 1
                try:
                    pts = self._storage.read_points(lid, ver, primary_key=primary_key)
                    if pts.num_rows > 0:
                        line_id = lid
                        break
                except (ValueError, FileNotFoundError):
                    continue
            if line_id is None:
                raise GDSEntityNotFoundError(
                    f"Entity '{primary_key}' not found in any anchor line"
                )

        # Discover which patterns this entity participates in
        pattern_map = self._discover_pattern_map(line_id)

        signals: dict[str, dict] = {}
        source_count = 0

        for pat_id, key_type in pattern_map.items():
            try:
                version = self._resolve_version(pat_id)
            except GDSNavigationError:
                continue

            if key_type in ("direct", "sibling"):
                signal = self._profile_direct(
                    primary_key, pat_id, version,
                )
                if signal is not None:
                    signal["key_type"] = key_type
            elif key_type == "event_edge":
                signal = self._profile_event_edge(
                    primary_key, pat_id, version, max_related,
                )
            elif key_type == "composite":
                signal = self._profile_composite(
                    primary_key, pat_id, version, max_related,
                )
            elif key_type == "chain":
                entity_line_id = sphere.entity_line(pat_id)
                if not entity_line_id:
                    continue
                line_ver = self._manifest.line_version(entity_line_id) or 1
                signal = self._profile_chain(
                    primary_key, pat_id, version,
                    entity_line_id, line_ver, max_related,
                )
            else:
                continue

            if signal:
                signals[pat_id] = signal
                if signal.get("anomalous_count", 0) > 0:
                    source_count += 1

        # Weighted risk score: each pattern contributes its anomaly density
        # (anomalous_count / related_count). Continuous 0.0-N scale.
        risk_score = 0.0
        for sig in signals.values():
            related = max(sig.get("related_count", 1), 1)
            anom = sig.get("anomalous_count", 0)
            risk_score += anom / related

        # Connected risk: mean delta_rank_pct of counterparties in
        # the direct pattern. Measures how anomalous the entity's
        # immediate network is (1-hop risk propagation).
        connected_risk = self._compute_connected_risk(
            primary_key, line_id, signals, pattern_map,
        )

        return {
            "primary_key": primary_key,
            "line_id": line_id,
            "source_count": source_count,
            "risk_score": round(risk_score, 4),
            "connected_risk": round(connected_risk, 2) if connected_risk is not None else None,
            "total_patterns": len(signals),
            "signals": signals,
        }

    def composite_risk(
        self,
        primary_key: str,
        line_id: str | None = None,
        max_related: int = 10,
    ) -> dict:
        """Compose anomaly p-values across patterns via Fisher's method.

        Uses conformal_p from each pattern's cross_pattern_profile signal.
        """
        from hypertopos.engine.composition import fisher_combine_pvalues

        profile = self.cross_pattern_profile(primary_key, line_id, max_related)
        p_values = []
        per_pattern: dict[str, dict] = {}
        for pat_id, signal in profile.get("signals", {}).items():
            cp = signal.get("conformal_p")
            if cp is not None and cp > 0:
                p_values.append(cp)
                per_pattern[pat_id] = {
                    "conformal_p": cp,
                    "is_anomaly": signal.get("is_anomaly", False),
                    "delta_norm": signal.get("delta_norm"),
                }
        if not p_values:
            return {
                "primary_key": primary_key,
                "combined_p": None,
                "per_pattern": per_pattern,
                "n_patterns": 0,
            }
        fisher = fisher_combine_pvalues(p_values)
        return {
            "primary_key": primary_key,
            "combined_p": fisher["combined_p"],
            "chi2": fisher["chi2"],
            "df": fisher["df"],
            "n_patterns": fisher["k"],
            "per_pattern": per_pattern,
        }

    def composite_risk_batch(
        self,
        primary_keys: list[str],
        line_id: str | None = None,
        max_keys: int = 200,
    ) -> dict:
        """Batch composite risk — Fisher combination for multiple entities.

        Returns per-key combined_p + summary counts at p<0.10 and p<0.05.
        Hard cap: max_keys (default 200).
        """
        keys = primary_keys[:max_keys]
        results = []
        for key in keys:
            try:
                cr = self.composite_risk(key, line_id)
                if cr.get("n_patterns", 0) == 0:
                    cr["error"] = "not_found"
                results.append(cr)
            except _NAVIGATION_RECOVERABLE_ERRORS:
                results.append({
                    "primary_key": key,
                    "combined_p": None,
                    "n_patterns": 0,
                    "error": "not_found",
                })
        valid = [r for r in results if r.get("combined_p") is not None]
        caught_010 = sum(1 for r in valid if r["combined_p"] < 0.10)
        caught_005 = sum(1 for r in valid if r["combined_p"] < 0.05)
        return {
            "total_requested": len(primary_keys),
            "total_checked": len(keys),
            "caught_p010": caught_010,
            "caught_p005": caught_005,
            "results": sorted(results, key=lambda r: r.get("combined_p") or 999),
        }

    def _compute_connected_risk(
        self,
        primary_key: str,
        line_id: str,
        signals: dict[str, dict],
        pattern_map: dict[str, str],
    ) -> float | None:
        """Compute mean delta_rank_pct of counterparties (1-hop risk).

        Uses the composite pattern's anomalous_keys to find counterparty
        account keys, then looks up their delta_rank_pct in the direct
        pattern. Returns mean rank (0-100) or None if no composite signal.
        """
        # Find the direct pattern for this entity's line
        direct_pat_id = None
        for pat_id, key_type in pattern_map.items():
            if key_type == "direct":
                direct_pat_id = pat_id
                break
        if direct_pat_id is None:
            return None

        # Collect counterparty keys from composite signals
        counterparty_keys: set[str] = set()
        sep = "\u2192"
        for _pat_id, sig in signals.items():
            if sig.get("key_type") != "composite":
                continue
            for anom_key in sig.get("anomalous_keys", []):
                parts = anom_key.split(sep)
                for p in parts:
                    if p != primary_key:
                        counterparty_keys.add(p)

        if not counterparty_keys:
            return None

        # Sample up to 50 counterparties for efficiency
        sample = list(counterparty_keys)[:50]
        try:
            self._resolve_version(direct_pat_id)
        except GDSNavigationError:
            return None

        ranks: list[float] = []
        for ck in sample:
            try:
                meta = self.get_entity_geometry_meta(ck, direct_pat_id)
                if meta.get("delta_rank_pct") is not None:
                    ranks.append(meta["delta_rank_pct"])
            except KeyError:
                continue

        if not ranks:
            return None
        return float(np.mean(ranks))

    def _profile_direct(
        self, primary_key: str, pattern_id: str, version: int,
    ) -> dict | None:
        """Profile entity in a direct pattern (same key space)."""
        cols = ["delta_norm", "is_anomaly", "delta_rank_pct", "conformal_p"]
        geo = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key, columns=cols,
        )
        if geo.num_rows == 0:
            return None

        is_anom = bool(geo["is_anomaly"][0].as_py())
        dn = float(geo["delta_norm"][0].as_py())
        pct = geo["delta_rank_pct"][0].as_py()
        result = {
            "key_type": "direct",
            "is_anomaly": is_anom,
            "delta_norm": round(dn, 4),
            "delta_rank_pct": round(float(pct), 2) if pct is not None else None,
            "related_count": 1,
            "anomalous_count": 1 if is_anom else 0,
            "anomalous_keys": [primary_key] if is_anom else [],
        }
        if "conformal_p" in geo.column_names:
            cp = geo["conformal_p"][0].as_py()
            if cp is not None:
                result["conformal_p"] = round(float(cp), 6)
        return result

    def _profile_event_edge(
        self, entity_key: str, pattern_id: str, version: int,
        max_related: int,
    ) -> dict | None:
        """Profile entity via edge lookup in an event/anchor pattern.

        Uses point_keys (LABEL_LIST index) to find polygons referencing this entity.
        """
        cols = ["primary_key", "is_anomaly", "delta_norm"]
        geo = self._storage.read_geometry(
            pattern_id, version, point_keys=[entity_key], columns=cols,
        )
        if geo.num_rows == 0:
            return None

        total = geo.num_rows
        anom_pks = []
        for i in range(geo.num_rows):
            if geo["is_anomaly"][i].as_py():
                anom_pks.append(geo["primary_key"][i].as_py())

        return {
            "key_type": "event_edge",
            "related_count": total,
            "anomalous_count": len(anom_pks),
            "anomalous_keys": anom_pks[:max_related],
        }

    def _profile_composite(
        self, entity_key: str, pattern_id: str, version: int,
        max_related: int,
    ) -> dict | None:
        """Profile entity in a composite pattern (e.g. pair_pattern).

        Two-pass approach for performance:
        1. Count total related rows (lightweight, primary_key only)
        2. Read only anomalous rows (few rows, with delta_norm)
        """
        ek_esc = entity_key.replace("'", "''")
        sep = "\u2192"
        base_filt = (
            f"starts_with(primary_key, '{ek_esc}{sep}') "
            f"OR ends_with(primary_key, '{sep}{ek_esc}')"
        )

        # Pass 1: count total (lightweight — primary_key column only)
        total_geo = self._storage.read_geometry(
            pattern_id, version, columns=["primary_key"],
            filter=base_filt,
        )
        total = total_geo.num_rows
        if total == 0:
            return None

        # Pass 2: read only anomalous rows (much fewer)
        anom_filt = f"({base_filt}) AND is_anomaly = true"
        anom_geo = self._storage.read_geometry(
            pattern_id, version,
            columns=["primary_key", "delta_norm"],
            filter=anom_filt,
        )
        anom_count = anom_geo.num_rows
        max_norm = 0.0
        anom_keys: list[str] = []
        if anom_count > 0:
            norms = anom_geo["delta_norm"].to_pylist()
            max_norm = max(norms)
            anom_keys = anom_geo["primary_key"].to_pylist()[:max_related]

        return {
            "key_type": "composite",
            "is_anomaly": anom_count > 0,
            "delta_norm": round(float(max_norm), 4),
            "delta_rank_pct": None,
            "related_count": total,
            "anomalous_count": anom_count,
            "anomalous_keys": anom_keys,
        }

    def _profile_chain(
        self, entity_key: str, pattern_id: str, version: int,
        chain_line_id: str, chain_line_version: int,
        max_related: int,
    ) -> dict | None:
        """Profile entity in a chain pattern.

        Uses reverse index (entity→chain_pks) then reads anomaly status
        from geometry. Reads full geometry once and filters in-memory
        (cheaper than building huge SQL OR filter).
        """
        rev_idx = self._get_chain_reverse_index(
            chain_line_id, chain_line_version,
        )
        chain_pks = rev_idx.get(entity_key, [])
        if not chain_pks:
            return None

        chain_pk_set = set(chain_pks)

        # Read full chain geometry (is_anomaly + delta_norm only — lightweight)
        # and filter in-memory by chain_pk_set. Faster than building
        # OR filter with 10K+ PKs.
        geo = self._storage.read_geometry(
            pattern_id, version,
            columns=["primary_key", "is_anomaly", "delta_norm"],
        )
        # In-memory filter
        pks = geo["primary_key"].to_pylist()
        anoms = geo["is_anomaly"].to_pylist()
        norms = geo["delta_norm"].to_pylist()

        anom_count = 0
        max_norm = 0.0
        anom_keys: list[str] = []
        matched = 0
        for pk, anom, norm in zip(pks, anoms, norms, strict=False):
            if pk not in chain_pk_set:
                continue
            matched += 1
            if anom:
                anom_count += 1
                if norm > max_norm:
                    max_norm = norm
                if len(anom_keys) < max_related:
                    anom_keys.append(pk)

        if matched == 0:
            return None

        return {
            "key_type": "chain",
            "is_anomaly": anom_count > 0,
            "delta_norm": round(float(max_norm), 4),
            "delta_rank_pct": None,
            "related_count": matched,
            "anomalous_count": anom_count,
            "anomalous_keys": anom_keys,
        }

    def find_neighborhood(
        self,
        primary_key: str,
        pattern_id: str,
        max_hops: int = 2,
        max_entities: int = 100,
    ) -> dict[str, Any]:
        """BFS from entity through jumpable polygon edges. Returns reachable entities.

        Only works for patterns with jumpable edges (binary FK mode).
        Continuous-mode patterns have point_key="" — not jumpable.
        For continuous mode, use find_counterparties instead.
        """
        version = self._resolve_version(pattern_id)

        sphere = self._storage.read_sphere()
        _pat = sphere.patterns[pattern_id]

        if _pat.is_continuous:
            raise GDSNavigationError(
                f"find_neighborhood requires binary FK mode. "
                f"Pattern '{pattern_id}' uses continuous edge encoding "
                f"(edge_max={_pat.edge_max}). "
                f"Use find_counterparties instead."
            )

        visited: set[str] = {primary_key}
        entities: list[dict[str, Any]] = []
        queue: deque[tuple[str, int]] = deque()

        # Seed BFS from center entity
        center_geo = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key,
            columns=["primary_key", "edges", "entity_keys"],
        )
        if center_geo.num_rows == 0:
            return {
                "center": primary_key,
                "pattern_id": pattern_id,
                "max_hops": max_hops,
                "entities": [],
                "summary": {
                    "total": 0,
                    "anomalous": 0,
                    "max_hop_reached": 0,
                    "capped": False,
                },
            }

        center_row = {c: center_geo[c][0].as_py() for c in center_geo.schema.names}
        center_edge_objs = _reconstruct_edges_from_row(center_row, _pat.relations)
        for edge in center_edge_objs:
            pk = edge.point_key
            if edge.is_alive() and pk and pk not in visited:
                visited.add(pk)
                queue.append((pk, 1))

        max_hop_reached = 0
        capped = False

        while queue and not capped:
            entity_key, hop = queue.popleft()
            if hop > max_hops:
                continue

            if hop > max_hop_reached:
                max_hop_reached = hop

            # Read geometry for this entity to get edges/entity_keys + anomaly info
            geo = self._storage.read_geometry(
                pattern_id, version, primary_key=entity_key,
                columns=["primary_key", "edges", "entity_keys", "is_anomaly", "delta_rank_pct"],
            )
            if geo.num_rows == 0:
                # Entity has no geometry row in this pattern — record with
                # unknown anomaly status but do not expand further.
                entities.append({
                    "key": entity_key,
                    "hop": hop,
                    "is_anomaly": None,
                    "delta_rank_pct": None,
                })
                if len(entities) >= max_entities:
                    capped = True
                continue

            is_anomaly = bool(geo["is_anomaly"][0].as_py())
            rank_val = geo["delta_rank_pct"][0].as_py()
            delta_rank_pct = (
                round(float(rank_val), 2) if rank_val is not None else None
            )

            entities.append({
                "key": entity_key,
                "hop": hop,
                "is_anomaly": is_anomaly,
                "delta_rank_pct": delta_rank_pct,
            })

            if len(entities) >= max_entities:
                capped = True
                break

            # Expand neighbors if we haven't hit max_hops
            if hop < max_hops:
                row = {c: geo[c][0].as_py() for c in geo.schema.names}
                row_edge_objs = _reconstruct_edges_from_row(row, _pat.relations)
                for edge in row_edge_objs:
                    pk = edge.point_key
                    if (
                        edge.is_alive()
                        and pk
                        and pk not in visited
                    ):
                        visited.add(pk)
                        queue.append((pk, hop + 1))

        return {
            "center": primary_key,
            "pattern_id": pattern_id,
            "max_hops": max_hops,
            "entities": entities,
            "summary": {
                "total": len(entities),
                "anomalous": sum(
                    1 for e in entities if e.get("is_anomaly") is True
                ),
                "max_hop_reached": max_hop_reached,
                "capped": capped,
            },
        }

    def find_chains_for_entity(
        self,
        primary_key: str,
        pattern_id: str,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Find transaction chains involving a specific entity.

        Uses the chain_keys reverse index to discover which chains the entity
        participates in, then enriches each chain with anomaly information
        from the pattern's geometry.

        Returns dict with:
            primary_key: str
            pattern_id: str
            chains: list[{chain_id, is_anomaly, delta_norm, delta_rank_pct}]
            summary: {total, anomalous}
        """
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        entity_line_id = sphere.entity_line(pattern_id)
        if entity_line_id is None:
            raise GDSNavigationError(
                f"No anchor line found for pattern '{pattern_id}'"
            )

        line_ver = self._manifest.line_version(entity_line_id) or 1

        try:
            rev_idx = self._get_chain_reverse_index(entity_line_id, line_ver)
        except KeyError as exc:
            raise GDSNavigationError(
                f"Line '{entity_line_id}' has no chain_keys column — "
                f"pattern '{pattern_id}' is not a chain pattern"
            ) from exc
        chain_pks = rev_idx.get(primary_key, [])

        if not chain_pks:
            return {
                "primary_key": primary_key,
                "pattern_id": pattern_id,
                "chains": [],
                "summary": {"total": 0, "anomalous": 0},
            }

        chain_pk_set = set(chain_pks)

        # Read geometry with anomaly columns, push-down filter via point_keys
        geo = self._storage.read_geometry(
            pattern_id, version,
            point_keys=chain_pks,
            columns=["primary_key", "is_anomaly", "delta_norm",
                     "delta_rank_pct"],
        )

        pks = geo["primary_key"].to_pylist()
        anoms = geo["is_anomaly"].to_pylist()
        norms = geo["delta_norm"].to_pylist()
        ranks = geo["delta_rank_pct"].to_pylist()

        chains: list[dict[str, Any]] = []
        for pk, anom, norm, rank in zip(pks, anoms, norms, ranks, strict=False):
            if pk not in chain_pk_set:
                continue
            chains.append({
                "chain_id": pk,
                "is_anomaly": bool(anom),
                "delta_norm": round(float(norm), 4) if norm is not None else 0.0,
                "delta_rank_pct": (
                    round(float(rank), 2) if rank is not None else None
                ),
            })

        # Sort by delta_norm descending, limit to top_n
        chains.sort(key=lambda c: c["delta_norm"], reverse=True)
        chains = chains[:top_n]

        anomalous = sum(1 for c in chains if c["is_anomaly"])

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "chains": chains,
            "summary": {
                "total": len(chains),
                "anomalous": anomalous,
            },
        }

    # ── Edge table: geometric path finding & lazy chains ──────

    def _resolve_anchor_pattern_for_scoring(self, event_pattern_id: str) -> str | None:
        """Find the anchor pattern that holds geometry for entities in this event pattern.

        Edge table lives on event patterns (tx_pattern) but entities (accounts)
        have their geometry in anchor patterns (account_pattern). Scoring needs
        the anchor pattern's deltas, not the event pattern's.

        Resolution strategy:
        1. Direct match: event relation line_id == anchor entity_line_id.
        2. Sibling match: event relation line shares source_id with anchor
           entity_line (e.g. "zones" and "zones_pickup" are siblings).
        """
        if not hasattr(self, "_anchor_pattern_cache"):
            self._anchor_pattern_cache: dict[str, str | None] = {}
        if event_pattern_id in self._anchor_pattern_cache:
            return self._anchor_pattern_cache[event_pattern_id]
        sphere = self._storage.read_sphere()
        event_pat = sphere.patterns.get(event_pattern_id)
        if event_pat is None or event_pat.pattern_type != "event":
            # Not an event pattern — use itself
            self._anchor_pattern_cache[event_pattern_id] = event_pattern_id
            return event_pattern_id

        rel_line_ids = {rel.line_id for rel in event_pat.relations}

        # Pass 1: direct match — relation line_id == anchor entity_line
        for pid, pat in sphere.patterns.items():
            if pat.pattern_type == "anchor" and pid != event_pattern_id:
                entity_line = pat.entity_line_id
                if entity_line and entity_line in rel_line_ids:
                    self._anchor_pattern_cache[event_pattern_id] = pid
                    return pid

        # Pass 2: sibling match — relation line shares source_id with entity_line
        for pid, pat in sphere.patterns.items():
            if pat.pattern_type == "anchor" and pid != event_pattern_id:
                entity_line = pat.entity_line_id
                if entity_line:
                    siblings = set(sphere.sibling_lines(entity_line))
                    if siblings & rel_line_ids:
                        self._anchor_pattern_cache[event_pattern_id] = pid
                        return pid

        self._anchor_pattern_cache[event_pattern_id] = None
        return None


    def _get_cached_delta(
        self,
        primary_key: str,
        pattern_id: str,
    ) -> np.ndarray | None:
        """Get delta vector for entity. Caches in _delta_cache."""
        if not hasattr(self, "_delta_cache"):
            self._delta_cache: dict[tuple[str, str], np.ndarray] = {}
        cache_key = (primary_key, pattern_id)
        if cache_key in self._delta_cache:
            return self._delta_cache[cache_key]
        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key,
            columns=["primary_key", "delta"],
        )
        if geo.num_rows == 0:
            return None
        delta = np.array(geo["delta"][0].as_py(), dtype=np.float32)
        self._delta_cache[cache_key] = delta
        return delta

    def _prefetch_deltas(
        self,
        keys: set[str],
        pattern_id: str,
    ) -> None:
        """Batch-prefetch delta vectors for a set of entities."""
        if not hasattr(self, "_delta_cache"):
            self._delta_cache = {}
        missing = [k for k in keys if (k, pattern_id) not in self._delta_cache]
        if not missing:
            return
        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version,
            point_keys=missing,
            columns=["primary_key", "delta"],
        )
        for i in range(geo.num_rows):
            pk = geo["primary_key"][i].as_py()
            delta = geo["delta"][i].as_py()
            if delta is not None:
                self._delta_cache[(pk, pattern_id)] = np.array(delta, dtype=np.float32)

    def _get_cached_theta(self, pattern_id: str) -> np.ndarray:
        """Get theta vector for pattern, cached to avoid repeated read_sphere()."""
        if not hasattr(self, "_theta_cache"):
            self._theta_cache: dict[str, np.ndarray] = {}
        if pattern_id not in self._theta_cache:
            sphere = self._storage.read_sphere()
            pat = sphere.patterns[pattern_id]
            self._theta_cache[pattern_id] = np.array(pat.theta, dtype=np.float32)
        return self._theta_cache[pattern_id]

    def _score_hop(
        self,
        from_key: str,
        to_key: str,
        pattern_id: str,
        scoring: str,
        amount: float = 0.0,
        max_amount: float = 1.0,
    ) -> float:
        """Score a single hop by the chosen strategy.

        When *scoring* is ``"amount"``, the geometric score is modulated by
        ``(1 + log1p(amount) / log1p(max_amount))``.
        """
        if scoring == "shortest":
            return 1.0
        delta_from = self._get_cached_delta(from_key, pattern_id)
        delta_to = self._get_cached_delta(to_key, pattern_id)
        if delta_from is None or delta_to is None:
            return 0.0
        if scoring == "anomaly":
            return float(np.linalg.norm(delta_to))
        # "geometric" base scoring
        norm_f = float(np.linalg.norm(delta_from))
        norm_t = float(np.linalg.norm(delta_to))
        # 1. Delta direction alignment (cosine similarity)
        denom = norm_f * norm_t + 1e-10
        alignment = float(np.dot(delta_from, delta_to) / denom)
        # 2. Witness overlap (shared anomalous dimensions)
        theta = self._get_cached_theta(pattern_id)
        witness_from = set(np.where(np.abs(delta_from) > theta)[0])
        witness_to = set(np.where(np.abs(delta_to) > theta)[0])
        if witness_from or witness_to:
            overlap = len(witness_from & witness_to) / len(witness_from | witness_to)
        else:
            overlap = 0.0
        # 3. Anomaly signal preservation
        preservation = min(norm_t, norm_f) / (max(norm_t, norm_f) + 1e-10)
        geo_score = 0.4 * alignment + 0.3 * overlap + 0.3 * preservation

        if scoring == "amount":
            # Modulate by transaction amount
            log_max = float(np.log1p(max_amount)) if max_amount > 0 else 1.0
            amount_factor = 1.0 + float(np.log1p(amount)) / (log_max + 1e-10)
            return geo_score * amount_factor

        return geo_score

    def _score_path(
        self,
        path_keys: list[str],
        pattern_id: str,
        scoring: str,
        adj_index: Any = None,
        max_amount: float = 1.0,
    ) -> float:
        total = 0.0
        for i in range(len(path_keys) - 1):
            amount = 0.0
            if scoring == "amount" and adj_index is not None:
                amount = self._lookup_edge_amount(
                    adj_index, path_keys[i], path_keys[i + 1],
                )
            total += self._score_hop(
                path_keys[i], path_keys[i + 1], pattern_id, scoring,
                amount=amount, max_amount=max_amount,
            )
        return total

    @staticmethod
    def _lookup_edge_amount(adj_index: Any, from_key: str, to_key: str) -> float:
        for tgt, _ts, amt, _ek in adj_index.neighbors_out(from_key):
            if tgt == to_key:
                return amt
        for src, _ts, amt, _ek in adj_index.neighbors_in(from_key):
            if src == to_key:
                return amt
        return 0.0

    @staticmethod
    def _reconstruct_bidir_path(
        fwd_parent: dict[str, str | None],
        bwd_parent: dict[str, str | None],
        meeting: str,
    ) -> list[str]:
        fwd: list[str] = []
        node: str | None = meeting
        while node is not None:
            fwd.append(node)
            node = fwd_parent[node]
        fwd.reverse()
        node = bwd_parent[meeting]
        while node is not None:
            fwd.append(node)
            node = bwd_parent[node]
        return fwd

    def find_geometric_path(
        self,
        from_key: str,
        to_key: str,
        pattern_id: str,
        max_depth: int = 5,
        beam_width: int = 50,
        scoring: str = "geometric",
    ) -> dict[str, Any]:
        """Find paths between two entities scored by geometric coherence.

        Uses bidirectional BFS for reliable path finding, then scores
        found paths by geometric coherence post-hoc.

        Args:
            from_key: Source entity primary key.
            to_key: Target entity primary key.
            pattern_id: Event pattern with edge table.
            max_depth: Maximum hops to search.
            beam_width: Maximum paths returned (top-K by score).
            scoring: "geometric" | "anomaly" | "shortest" | "amount".

        Returns dict with paths, each scored, plus summary.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "Rebuild sphere with edge table support."
            )
        scoring_pattern = self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        adj_index = self._storage.get_adjacency(pattern_id)

        # Phase 1: Bidirectional BFS — expand from both ends
        fwd_parent: dict[str, str | None] = {from_key: None}
        bwd_parent: dict[str, str | None] = {to_key: None}
        fwd_depth: dict[str, int] = {from_key: 0}
        bwd_depth: dict[str, int] = {to_key: 0}
        fwd_frontier: set[str] = {from_key}
        bwd_frontier: set[str] = {to_key}

        half = (max_depth + 1) // 2
        for d in range(1, half + 1):
            new_fwd: set[str] = set()
            for node in fwd_frontier:
                for tgt, _ts, _amt, _ek in adj_index.neighbors_out(node):
                    if tgt != node and tgt not in fwd_parent:
                        fwd_parent[tgt] = node
                        fwd_depth[tgt] = d
                        new_fwd.add(tgt)
                for src, _ts, _amt, _ek in adj_index.neighbors_in(node):
                    if src != node and src not in fwd_parent:
                        fwd_parent[src] = node
                        fwd_depth[src] = d
                        new_fwd.add(src)
            fwd_frontier = new_fwd

            new_bwd: set[str] = set()
            for node in bwd_frontier:
                for tgt, _ts, _amt, _ek in adj_index.neighbors_out(node):
                    if tgt != node and tgt not in bwd_parent:
                        bwd_parent[tgt] = node
                        bwd_depth[tgt] = d
                        new_bwd.add(tgt)
                for src, _ts, _amt, _ek in adj_index.neighbors_in(node):
                    if src != node and src not in bwd_parent:
                        bwd_parent[src] = node
                        bwd_depth[src] = d
                        new_bwd.add(src)
            bwd_frontier = new_bwd

        # Phase 2: Find valid meeting points (0 < total hops ≤ max_depth)
        meetings = fwd_parent.keys() & bwd_parent.keys()
        valid = sorted(
            ((fwd_depth[m] + bwd_depth[m], m) for m in meetings
             if 0 < fwd_depth[m] + bwd_depth[m] <= max_depth),
        )
        valid = valid[:1000]

        # Phase 3: Reconstruct paths, reject cycles
        raw_paths: list[list[str]] = []
        for _, m in valid:
            path = self._reconstruct_bidir_path(fwd_parent, bwd_parent, m)
            if len(path) == len(set(path)):
                raw_paths.append(path)

        # Phase 4: Score paths
        if raw_paths and scoring != "shortest":
            all_keys: set[str] = set()
            for path in raw_paths:
                all_keys.update(path)
            self._prefetch_deltas(all_keys, scoring_pattern)

        max_amt = 1.0
        if scoring == "amount":
            for path in raw_paths:
                for i in range(len(path) - 1):
                    amt = self._lookup_edge_amount(adj_index, path[i], path[i + 1])
                    if amt > max_amt:
                        max_amt = amt

        scored: list[tuple[list[str], float]] = []
        for path in raw_paths:
            score = self._score_path(path, scoring_pattern, scoring, adj_index, max_amt)
            scored.append((path, score))
        if scoring == "shortest":
            scored.sort(key=lambda x: len(x[0]))  # prefer fewer hops
        else:
            scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:beam_width]

        # Format results
        paths = []
        for path_keys, score in scored:
            paths.append({
                "keys": path_keys,
                "hops": len(path_keys) - 1,
                "geometric_score": round(score, 4),
            })

        return {
            "from_key": from_key,
            "to_key": to_key,
            "pattern_id": pattern_id,
            "scoring": scoring,
            "paths": paths,
            "summary": {
                "paths_found": len(paths),
                "best_score": round(paths[0]["geometric_score"], 4) if paths else 0.0,
                "max_depth": max_depth,
                "beam_width": beam_width,
                "score_interpretation": (
                    "geometric: 0=no coherence, 1=perfect alignment. "
                    "Components: delta alignment (40%), witness overlap (30%), anomaly preservation (30%)"
                    if scoring == "geometric"
                    else "amount: geometric score modulated by log(amount). "
                    "Higher = geometrically coherent path through high-value transactions"
                    if scoring == "amount"
                    else "anomaly: higher = more anomalous intermediaries"
                    if scoring == "anomaly"
                    else "shortest: all hops score 1.0"
                ),
            },
        }

    def discover_chains(
        self,
        primary_key: str,
        pattern_id: str,
        time_window_hours: int = 168,
        max_hops: int = 10,
        min_hops: int = 2,
        max_chains: int = 100,
        direction: str = "forward",
    ) -> dict[str, Any]:
        """Discover transaction chains from entity via temporal BFS on edge table.

        Unlike find_chains_for_entity() which looks up pre-computed chains,
        this performs runtime BFS — works without build-time chain extraction.

        **Note:** total_amount is the sum of hop amounts, not a tracked money flow.
        See "Chain Interpretation" in concepts.md for details.

        Args:
            primary_key: Starting entity.
            pattern_id: Event pattern with edge table.
            time_window_hours: Max gap between consecutive hops.
            max_hops: Maximum chain length.
            min_hops: Minimum chain length (filter shorter).
            max_chains: Output cap.
            direction: "forward" | "backward" | "both".

        Returns dict with chains, each scored by geometric coherence.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table."
            )
        window_secs = time_window_hours * 3600.0

        adj_index = self._storage.get_adjacency(pattern_id)

        def _load_temporal_adj(keys: list[str]) -> tuple[dict, dict]:
            fwd_adj: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
            bwd_adj: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
            for k in keys:
                fwd_adj[k] = [
                    (tgt, ts, amt) for tgt, ts, amt, _ek in adj_index.neighbors_out(k) if tgt != k
                ]
                bwd_adj[k] = [
                    (src, ts, amt) for src, ts, amt, _ek in adj_index.neighbors_in(k) if src != k
                ]
            return fwd_adj, bwd_adj

        fwd, bwd = _load_temporal_adj([primary_key])

        _QUEUE_CAP = 500_000  # prevent unbounded memory on dense graphs

        chains: list[dict[str, Any]] = []
        chain_id_counter = 0

        expanded_keys: set[str] = set()
        seen_chains: set[tuple[str, ...]] = set()

        def _bfs(is_forward: bool, start: str) -> None:
            nonlocal chain_id_counter
            adj = fwd if is_forward else bwd
            # Queue: (current_key, path_keys, first_timestamp, last_timestamp, total_amount)
            queue: deque[tuple[str, list[str], float, float, float]] = deque()
            for neighbor, ts, amt in adj.get(start, []):
                queue.append((neighbor, [start, neighbor], ts, ts, amt))

            while queue and len(chains) < max_chains and len(queue) < _QUEUE_CAP:
                current, path, first_ts, last_ts, total_amt = queue.popleft()
                # Record if >= min_hops
                if len(path) - 1 >= min_hops:
                    chain_key = tuple(path)
                    if chain_key in seen_chains:
                        continue
                    seen_chains.add(chain_key)
                    chain_id_counter += 1
                    is_cyclic = path[-1] == path[0]
                    time_span = last_ts - first_ts
                    chains.append({
                        "chain_id": f"chain_{chain_id_counter:05d}",
                        "keys": list(path),
                        "hop_count": len(path) - 1,
                        "is_cyclic": is_cyclic,
                        "time_span_hours": round(time_span / 3600.0, 2) if time_span else 0.0,
                        "total_amount": round(total_amt, 2),
                    })
                # Expand if under max_hops — lazy load adjacency for new keys
                if len(path) - 1 < max_hops:
                    if current not in expanded_keys:
                        expanded_keys.add(current)
                        new_fwd, new_bwd = _load_temporal_adj([current])
                        for k, v in new_fwd.items():
                            fwd[k].extend(v)
                            fwd[k].sort(key=lambda x: x[1])
                        for k, v in new_bwd.items():
                            bwd[k].extend(v)
                            bwd[k].sort(key=lambda x: x[1])
                    for neighbor, ts, amt in adj.get(current, []):
                        if ts >= last_ts and ts <= last_ts + window_secs:
                            if neighbor not in set(path):  # no revisit
                                queue.append((
                                    neighbor,
                                    path + [neighbor],
                                    first_ts,
                                    ts,
                                    total_amt + amt,
                                ))

        if direction in ("forward", "both"):
            _bfs(True, primary_key)
        if direction in ("backward", "both"):
            _bfs(False, primary_key)

        # Score chains geometrically — resolve anchor pattern for delta lookups
        scoring_pattern = self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        if chains:
            all_keys = set()
            for c in chains:
                all_keys.update(c["keys"])
            self._prefetch_deltas(all_keys, scoring_pattern)
            for c in chains:
                keys = c["keys"]
                if len(keys) < 2:
                    c["geometric_score"] = 0.0
                    continue
                total = 0.0
                for i in range(len(keys) - 1):
                    total += self._score_hop(keys[i], keys[i + 1], scoring_pattern, "geometric")
                c["geometric_score"] = round(total / (len(keys) - 1), 4)

        # Sort by geometric_score desc
        chains.sort(key=lambda c: c.get("geometric_score", 0.0), reverse=True)
        chains = chains[:max_chains]

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "chains": chains,
            "summary": {
                "total": len(chains),
                "cyclic": sum(1 for c in chains if c["is_cyclic"]),
                "avg_hops": (
                    round(sum(c["hop_count"] for c in chains) / len(chains), 1)
                    if chains else 0.0
                ),
            },
        }

    def explain_anomaly(
        self,
        primary_key: str,
        pattern_id: str,
    ) -> dict:
        """Structured investigation explanation for an anomalous entity.

        Combines: severity, witness set, repair set, top dimensions,
        conformal p-value, temporal context, reputation, and composite risk.
        """
        from hypertopos.engine.investigation import build_explanation

        polygon = self._engine.build_polygon(primary_key, pattern_id, self._manifest)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        dim_labels = pattern.dim_labels

        # Read conformal_p from geometry table (not on Polygon dataclass)
        conformal_p = None
        try:
            version = self._resolve_version(pattern_id)
            geo_row = self._storage.read_geometry(
                pattern_id, version,
                primary_key=primary_key,
                columns=["conformal_p"],
            )
            if geo_row.num_rows > 0 and "conformal_p" in geo_row.schema.names:
                conformal_p = float(geo_row.column("conformal_p")[0].as_py())
        except _NAVIGATION_RECOVERABLE_ERRORS:
            pass

        # Temporal context + reputation
        temporal_slices = None
        reputation = self.solid_reputation(primary_key, pattern_id)
        if pattern.pattern_type == "anchor":
            try:
                solid = self._engine.build_solid(primary_key, pattern_id, self._manifest)
                temporal_slices = len(solid.slices)
            except _NAVIGATION_RECOVERABLE_ERRORS:
                pass

        theta_norm = float(np.linalg.norm(pattern.theta)) if pattern.theta is not None else 0.0
        explanation = build_explanation(
            delta=polygon.delta,
            dim_labels=dim_labels,
            theta_norm=theta_norm,
            delta_norm=polygon.delta_norm,
            conformal_p=conformal_p,
            temporal_slices=temporal_slices,
            reputation=reputation,
            dimension_kinds=pattern.dimension_kinds,
            sigma=pattern.sigma_diag,
            mu=pattern.mu,
            dimension_weights=pattern.dimension_weights,
        )
        explanation["primary_key"] = primary_key
        explanation["pattern_id"] = pattern_id

        # Cross-pattern composite risk — skip when ≤1 direct pattern (saves ~2s)
        if polygon.is_anomaly:
            home_line = sphere.entity_line(pattern_id)
            if home_line is None and pattern.relations:
                home_line = pattern.relations[0].line_id
            if home_line:
                pmap = self._discover_pattern_map(home_line)
                n_direct = sum(1 for v in pmap.values() if v == "direct")
                if n_direct >= 2:
                    try:
                        composite = self.composite_risk(primary_key, home_line)
                        explanation["composite_risk"] = composite
                    except _NAVIGATION_RECOVERABLE_ERRORS:
                        pass

        return explanation

    def trace_root_cause(
        self,
        primary_key: str,
        pattern_id: str,
        max_depth: int = 2,
        max_branches: int = 3,
        *,
        hub_pop_limit: int = 50_000,
        contagion_min_threshold: float = 0.10,
        contagion_min_counterparties: int = 3,
        max_total_nodes: int = 50,
        edge_counterparty_top_n: int = 1,
        branches_enabled: list[str] | None = None,
    ) -> dict[str, Any]:
        """Multi-hop root-cause DAG for an anomalous entity.

        Composes ``explain_anomaly`` (root + top dimensions),
        ``find_counterparties`` (edge-derived witness follow — sorted by
        anomaly, not by transaction volume), ``contagion_score`` (neighbour
        anomaly fraction with explicit anomalous counterparty keys), and
        ``π7_attract_hub`` (hub concentration signal) into one bounded DAG.
        Replaces the former linear ``explain_anomaly_chain``.

        Severity uses one scale for every node:
        ``"normal" < "low" < "moderate" < "high" < "critical" < "extreme"``.
        Contagion grading: < ``contagion_min_threshold`` → no branch, then
        ``low`` (>= threshold), ``moderate`` (>= 0.25), ``high`` (>= 0.50),
        ``critical`` (>= 0.75).

        Per-node candidates (contagion, edge-counterparty, hub) are collected,
        scored by unified severity strength, then the top ``max_branches`` are
        kept — the tree is not FIFO-ordered.  ``truncated`` is set only when at
        least one candidate was dropped because of the cap.  A hard
        ``max_total_nodes`` cap guards against recursion blowups.

        Args:
            primary_key: anomalous entity to trace.
            pattern_id: pattern it lives in.
            max_depth: hops away from the root to expand. 0 = root only.
            max_branches: max children kept per node after priority sort.
            hub_pop_limit: skip hub branch when the pattern has more than this
                many entities (π7 is O(n), not worth it on 500k+ populations).
            contagion_min_threshold: minimum contagion score for the branch to
                be attached at all. Set to 0.0 to always attach when the entity
                has counterparties; set above 0.5 to keep only high-signal.
            max_total_nodes: hard cap on nodes expanded across the whole DAG.

        Returns a wrapper dict:
            {
                "root": nested RootCauseNode dict,
                "summary": str,
                "hop_count": int,
                "branches_explored": int,
                "truncated": bool,
            }
        """
        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"Pattern '{pattern_id}' not found in sphere.")
        pattern = sphere.patterns[pattern_id]

        # Resolve graph companion — the event pattern carrying the actual edge
        # table for an anchor pattern. Continuous-mode anchors cannot answer
        # counterparty / contagion queries directly; their graph lives in a
        # paired event pattern. Fall back to the pattern itself when it is an
        # event pattern with its own edges, or None when no graph is usable.
        graph_pid: str | None = None
        try:
            if pattern.pattern_type == "anchor":
                graph_pid = self._resolve_edge_pattern_for_anchor(pattern_id)
            elif pattern.pattern_type == "event" and self._storage.has_edge_table(pattern_id):
                graph_pid = pattern_id
        except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
            graph_pid = None
        home_line = sphere.entity_line(pattern_id) if hasattr(sphere, "entity_line") else None

        # Version-keyed hub cache — invalidates automatically across rebuilds.
        try:
            pattern_version = self._resolve_version(pattern_id)
        except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
            pattern_version = 0

        # Unified severity scale used by every node in the DAG.
        # The legacy "medium" label from explain_anomaly is mapped to "moderate"
        # so the whole tree speaks one vocabulary.
        _SEVERITY_STRENGTH = {
            "normal": 0,
            "low": 1,
            "medium": 2,
            "moderate": 2,
            "high": 3,
            "critical": 4,
            "extreme": 5,
        }

        def _normalise_severity(sev: str) -> str:
            return "moderate" if sev == "medium" else sev

        # Session-scope (navigator-instance) caches keyed by (pattern_version, entity).
        # Survive across multiple trace_root_cause calls within one session — critical
        # for agent investigation flows that hit the same counterparty repeatedly.
        # Hub cache already lives at self._trace_hub_cache with version keying.
        # LRU cap — 2000 entries per cache, evict oldest on overflow. Protects
        # against long agent sessions touching 10k+ entities.
        _LRU_MAX = 2000
        if not hasattr(self, "_trace_contagion_cache"):
            self._trace_contagion_cache: dict[tuple[str, int, str], dict[str, Any] | None] = {}
        if not hasattr(self, "_trace_cps_cache"):
            self._trace_cps_cache: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        if not hasattr(self, "_trace_cp_ledger"):
            # Cross-call ledger: entity_key -> set of entity_keys that listed it
            # as an anomalous counterparty in their trace. Exposes inter-call
            # clique signals without requiring the agent to diff multiple trees.
            self._trace_cp_ledger: dict[tuple[str, int, str], set[str]] = {}

        def _lru_put(cache: dict, key: Any, value: Any) -> None:
            if key in cache:
                return  # already present
            if len(cache) >= _LRU_MAX:
                # Evict oldest — dict preserves insertion order in py3.7+.
                cache.pop(next(iter(cache)))
            cache[key] = value

        _contagion_cache = self._trace_contagion_cache
        _cps_cache = self._trace_cps_cache
        _cp_ledger = self._trace_cp_ledger
        # Cache key format: (graph_pid, version, entity_key) — version invalidates
        # automatically on rebuild.

        # Validate branches_enabled — typos like ["kubelek"] would silently
        # disable all branches, which is a user-hostile failure mode.
        _VALID_BRANCHES = {"edge_counterparty", "neighbor_contamination", "hub"}
        if branches_enabled is not None:
            invalid = set(branches_enabled) - _VALID_BRANCHES
            if invalid:
                raise GDSNavigationError(
                    f"branches_enabled contains invalid values: {sorted(invalid)}. "
                    f"Valid options: {sorted(_VALID_BRANCHES)}."
                )
        _enabled_branches = set(branches_enabled) if branches_enabled else _VALID_BRANCHES

        def _grade_contagion(score: float) -> str | None:
            if score < contagion_min_threshold:
                return None
            if score >= 0.75:
                return "critical"
            if score >= 0.50:
                return "high"
            if score >= 0.25:
                return "moderate"
            return "low"

        visited: set[str] = set()
        stats = {"hop_count": 0, "branches_explored": 0, "truncated": False}

        def _node_to_dict(node: RootCauseNode) -> dict[str, Any]:
            return {
                "entity_key": node.entity_key,
                "role": node.role,
                "severity": node.severity,
                "evidence": node.evidence,
                "children": [_node_to_dict(c) for c in node.children],
            }

        def _expand(
            entity_key: str,
            depth_remaining: int,
            role: str,
            inherited_evidence: dict[str, Any] | None = None,
        ) -> RootCauseNode:
            if entity_key in visited:
                return RootCauseNode(
                    entity_key=entity_key,
                    role=role,
                    severity="normal",
                    evidence={"cycle": True, **(inherited_evidence or {})},
                    children=[],
                )
            visited.add(entity_key)
            stats["hop_count"] += 1

            try:
                explanation = self.explain_anomaly(entity_key, pattern_id)
            except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
                return RootCauseNode(
                    entity_key=entity_key,
                    role=role,
                    severity="normal",
                    evidence={"explain_failed": True, **(inherited_evidence or {})},
                    children=[],
                )

            severity = _normalise_severity(explanation.get("severity", "normal"))
            top_dims = explanation.get("top_dimensions", [])[:max_branches]
            evidence: dict[str, Any] = {
                "top_dimensions": top_dims,
                "delta_norm": explanation.get("delta_norm"),
                "conformal_p": explanation.get("conformal_p"),
            }
            if inherited_evidence:
                evidence.update(inherited_evidence)

            if (
                severity == "normal"
                or not top_dims
                or depth_remaining <= 0
                or stats["hop_count"] >= max_total_nodes
            ):
                if stats["hop_count"] >= max_total_nodes:
                    stats["truncated"] = True
                return RootCauseNode(
                    entity_key=entity_key,
                    role=role,
                    severity=severity,
                    evidence=evidence,
                    children=[],
                )

            # ---- Build candidate list (not yet filtered by max_branches) ----
            # Each candidate is (strength: int, builder: Callable[[], RootCauseNode]).
            # Callers that actually recurse (edge_counterparty) are deferred so
            # recursion only runs for selected candidates — prevents wasted work.
            candidates: list[tuple[int, Any]] = []

            # --- Candidate: neighbor_contamination ---
            contagion_data: dict[str, Any] | None = None
            cps_data: list[dict[str, Any]] = []
            if graph_pid is not None:
                # contagion_score cache: one 27s scan per entity max per session.
                cache_key = (graph_pid, pattern_version, entity_key)
                if cache_key in _contagion_cache:
                    contagion_data = _contagion_cache[cache_key]
                else:
                    try:
                        contagion_data = self.contagion_score(entity_key, graph_pid)
                    except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
                        contagion_data = None
                    _lru_put(_contagion_cache, cache_key, contagion_data)

                # find_counterparties cache: same rationale.
                # Note: find_counterparties returns {outgoing: [...], incoming: [...]}
                # (not a flat "counterparties" list). Merge and dedupe self-edges.
                if home_line is not None:
                    if cache_key in _cps_cache:
                        cps_data = _cps_cache[cache_key]
                    else:
                        try:
                            cp_result = self.find_counterparties(
                                entity_key,
                                line_id=home_line,
                                from_col="from_key",
                                to_col="to_key",
                                pattern_id=graph_pid,
                                top_n=50,
                            )
                            if isinstance(cp_result, dict):
                                raw_cps = (
                                    list(cp_result.get("outgoing") or [])
                                    + list(cp_result.get("incoming") or [])
                                )
                                seen_keys: set[str] = set()
                                merged: list[dict[str, Any]] = []
                                for c in raw_cps:
                                    k = c.get("primary_key") or c.get("key")
                                    if not k or k == entity_key or k in seen_keys:
                                        continue
                                    seen_keys.add(k)
                                    merged.append(c)
                                cps_data = merged
                        except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
                            cps_data = []
                        _lru_put(_cps_cache, cache_key, cps_data)
                    # find_counterparties fast path (edge-table BTREE lookup)
                    # already enriches is_anomaly via _resolve_anchor_pattern_for_scoring
                    # — the flag is already an account-level signal, not TX-level,
                    # so no second-pass enrichment is needed.

            if (
                contagion_data is not None
                and "neighbor_contamination" in _enabled_branches
            ):
                cscore = float(contagion_data.get("score", 0.0) or 0.0)
                total_cp = int(contagion_data.get("total_counterparties") or 0)
                c_severity = _grade_contagion(cscore)
                # Small-N guard: a score of 1.0 from 1/1 counterparty is statistical
                # noise, not a meaningful contamination signal. Require a minimum
                # counterparty pool before attaching the branch.
                if (
                    total_cp >= contagion_min_counterparties
                    and c_severity is not None
                ):
                    # Populate anomalous_cp_keys from the counterparty list —
                    # saves the agent a follow-up call.
                    anom_keys = [
                        c.get("primary_key") or c.get("key")
                        for c in cps_data
                        if c.get("is_anomaly") and (c.get("primary_key") or c.get("key"))
                    ]
                    # Update session-scope ledger: record that `entity_key` listed
                    # each anom_key as its anomalous counterparty. Read back when
                    # building other branches to surface inter-call clique signals.
                    for ak in anom_keys:
                        ledger_key = (graph_pid, pattern_version, ak)
                        _cp_ledger.setdefault(ledger_key, set()).add(entity_key)
                    # Revisits-root list: the subset of anomalous counterparties
                    # that equals the trace root — a self-documenting clique signal.
                    revisits_root_keys = [k for k in anom_keys if k == primary_key]
                    # Inter-call clique: any entity previously seen listing THIS
                    # entity as their anomalous counterparty.
                    prev_seen_key = (graph_pid, pattern_version, entity_key)
                    seen_as_cp_of = sorted(_cp_ledger.get(prev_seen_key, set()) - {entity_key})
                    cscore_rounded = round(cscore, 4)
                    c_severity_final = c_severity
                    anom_keys_sample = anom_keys[:10]
                    contagion_anom_count = contagion_data.get("anomalous_counterparties")

                    def _build_contagion(
                        _key: str = entity_key,
                        _sev: str = c_severity_final,
                        _score: float = cscore_rounded,
                        _total: int = total_cp,
                        _anom_count: Any = contagion_anom_count,
                        _keys: list[str] = anom_keys_sample,
                        _revisits_keys: list[str] = revisits_root_keys,
                        _seen_as_cp: list[str] = seen_as_cp_of,
                    ) -> RootCauseNode:
                        ev: dict[str, Any] = {
                            "contagion_score": _score,
                            "total_counterparties": _total,
                            "anomalous_counterparties": _anom_count,
                            "anomalous_cp_keys": _keys,
                        }
                        if _revisits_keys:
                            ev["revisits_root"] = _revisits_keys
                        if _seen_as_cp:
                            ev["previously_seen_as_cp_of"] = _seen_as_cp
                        return RootCauseNode(
                            entity_key=_key,
                            role="neighbor_contamination",
                            severity=_sev,
                            evidence=ev,
                            children=[],
                        )

                    candidates.append((_SEVERITY_STRENGTH.get(c_severity_final, 0), _build_contagion))

            # --- Candidate(s): edge_counterparty (sort-by-anomaly, not volume) ---
            # Each anomalous counterparty up to edge_counterparty_top_n becomes a
            # separate candidate. If more anomalous CPs exist than the cap, mark
            # truncated — the extras ARE informative but we chose not to recurse
            # on them to keep the tree bounded.
            if (
                depth_remaining > 1
                and graph_pid is not None
                and cps_data
                and stats["hop_count"] < max_total_nodes
                and "edge_counterparty" in _enabled_branches
            ):
                # Rank counterparties: anomalous first (by delta_rank_pct desc
                # within each group). The critical fix over sorting purely by
                # amount_sum, which repeatedly missed the actual anomalous
                # neighbours on AML-style continuous patterns.
                def _cp_sort_key(c: dict[str, Any]) -> tuple[int, float]:
                    anom_flag = 1 if c.get("is_anomaly") else 0
                    rank_pct = float(c.get("delta_rank_pct") or 0.0)
                    return (-anom_flag, -rank_pct)

                sorted_cps = sorted(cps_data, key=_cp_sort_key)
                anomalous_cps = [c for c in sorted_cps if c.get("is_anomaly")]
                if len(anomalous_cps) > edge_counterparty_top_n:
                    stats["truncated"] = True
                picked_cps = anomalous_cps[:edge_counterparty_top_n]
                first_dim = top_dims[0] if top_dims else {}
                via_dim = first_dim.get("label") or first_dim.get("dim")

                for anom_cp in picked_cps:
                    cp_key = anom_cp.get("primary_key") or anom_cp.get("key")
                    if not cp_key:
                        continue
                    rank_pct = float(anom_cp.get("delta_rank_pct") or 0.0)
                    if rank_pct >= 99.9:
                        cp_sev_hint = "extreme"
                    elif rank_pct >= 99.0:
                        cp_sev_hint = "critical"
                    elif rank_pct >= 95.0:
                        cp_sev_hint = "high"
                    else:
                        cp_sev_hint = "moderate"
                    edge_pot_evidence: dict[str, Any] | None = None
                    try:
                        edge_pot_evidence = self.edge_potential(
                            entity_key, cp_key, pattern_id,
                        )
                    except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
                        edge_pot_evidence = None
                    inherited = {
                        "via_dim": via_dim,
                        "witness_counterparty_delta_rank_pct": rank_pct,
                    }
                    if edge_pot_evidence is not None:
                        inherited["edge_potential"] = {
                            "score": edge_pot_evidence.get("score"),
                            "delta_distance": edge_pot_evidence.get("delta_distance"),
                            "pair_tx_count": edge_pot_evidence.get("pair_tx_count"),
                            "effective_weight": edge_pot_evidence.get("effective_weight"),
                        }

                    best_motif: dict[str, Any] | None = None
                    for mt in ("cycle_2", "cycle_3", "fan_out"):
                        try:
                            scored = self.score_motif(
                                entity_key, motif_type=mt, pattern_id=pattern_id,
                            )
                        except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
                            continue
                        if not scored.get("found"):
                            continue
                        if mt == "fan_out" and cp_key not in {
                            e[1] for e in scored.get("edges", [])
                        }:
                            continue
                        if mt in {"cycle_2", "cycle_3"} and cp_key not in scored.get(
                            "ring", [scored.get("counterparty")],
                        ):
                            continue
                        if best_motif is None or scored["score"] > best_motif["score"]:
                            best_motif = {
                                "motif_type": mt,
                                "score": scored["score"],
                                "time_window_hours": scored.get("time_window_hours"),
                            }
                            if mt == "cycle_3":
                                best_motif["ring"] = scored.get("ring")
                            elif mt == "cycle_2":
                                best_motif["counterparty"] = scored.get("counterparty")
                            elif mt == "fan_out":
                                best_motif["k"] = scored.get("k")
                    if best_motif is not None:
                        inherited["motif_potential"] = best_motif

                    def _build_edge_cp(
                        _key: str = cp_key,
                        _depth: int = depth_remaining - 1,
                        _inh: dict[str, Any] = inherited,
                    ) -> RootCauseNode:
                        return _expand(
                            _key,
                            _depth,
                            role="edge_counterparty",
                            inherited_evidence=_inh,
                        )

                    candidates.append((_SEVERITY_STRENGTH.get(cp_sev_hint, 0), _build_edge_cp))

            # --- Candidate: hub (gated by population size) ---
            try:
                pop_size = int(getattr(pattern, "population_size", 0) or 0)
            except (TypeError, ValueError):
                pop_size = 0
            if pop_size <= hub_pop_limit and "hub" in _enabled_branches:
                if not hasattr(self, "_trace_hub_cache"):
                    self._trace_hub_cache: dict[tuple[str, int], set[str]] = {}
                cache_key = (pattern_id, pattern_version)
                hub_keys = self._trace_hub_cache.get(cache_key)
                if hub_keys is None:
                    try:
                        hubs = self.π7_attract_hub(pattern_id, top_n=20)
                        hub_keys = {h[0] for h in hubs}
                    except (*_NAVIGATION_RECOVERABLE_ERRORS, GDSError):
                        hub_keys = set()
                    self._trace_hub_cache[cache_key] = hub_keys
                if entity_key in hub_keys:
                    hub_severity_final = severity  # inherit root anomaly severity
                    hub_entity_final = entity_key

                    def _build_hub(
                        _sev: str = hub_severity_final,
                        _entity: str = hub_entity_final,
                    ) -> RootCauseNode:
                        return RootCauseNode(
                            entity_key=_entity,
                            role="hub",
                            severity=_sev,
                            evidence={"is_top_hub": True, "hub_top_n": 20},
                            children=[],
                        )

                    candidates.append((_SEVERITY_STRENGTH.get(hub_severity_final, 0), _build_hub))

            # ---- Priority selection: sort desc by strength, take top-K ----
            candidates.sort(key=lambda c: -c[0])
            selected = candidates[:max_branches]
            if len(candidates) > max_branches:
                stats["truncated"] = True

            children: list[RootCauseNode] = []
            for _, builder in selected:
                if stats["hop_count"] >= max_total_nodes:
                    stats["truncated"] = True
                    break
                children.append(builder())
                stats["branches_explored"] += 1

            return RootCauseNode(
                entity_key=entity_key,
                role=role,
                severity=severity,
                evidence=evidence,
                children=children,
            )

        root_node = _expand(primary_key, max_depth, role="root")

        severity_root = root_node.severity
        if severity_root == "normal":
            summary = (
                f"Entity {primary_key} is not anomalous in pattern {pattern_id} — "
                f"no root cause to trace."
            )
        else:
            top_witness = None
            td = root_node.evidence.get("top_dimensions") or []
            if td and isinstance(td[0], dict):
                top_witness = td[0].get("label") or td[0].get("dim")

            # Walk the whole tree to collect the set of branch roles actually
            # emitted — saves the agent a traversal pass.
            found_roles: list[str] = []
            def _collect_roles(node: RootCauseNode) -> None:
                for c in node.children:
                    if c.role not in found_roles:
                        found_roles.append(c.role)
                    _collect_roles(c)
            _collect_roles(root_node)
            roles_str = ", ".join(found_roles) if found_roles else "none"

            summary = (
                f"Entity {primary_key} is {severity_root} in {pattern_id}"
                + (f"; primary witness: {top_witness}" if top_witness else "")
                + f"; branches found: {roles_str}"
                + f"; {stats['branches_explored']} nodes"
                + (" (truncated)" if stats["truncated"] else "")
                + "."
            )

        return {
            "root": _node_to_dict(root_node),
            "summary": summary,
            "hop_count": stats["hop_count"],
            "branches_explored": stats["branches_explored"],
            "truncated": stats["truncated"],
        }

    # ------------------------------------------------------------------
    # line_profile — direct points-table column profiling
    # ------------------------------------------------------------------

    def line_profile(
        self,
        line_id: str,
        property_name: str,
        *,
        limit: int = 20,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        """Profile a single column from a line's points table.

        Returns categorical value-counts for string/bool columns, descriptive
        statistics for numeric columns, or min/max for temporal columns.
        When *group_by* is supplied, numeric stats are broken down per group.
        """
        import pyarrow.types as pat

        # -- resolve line ------------------------------------------------
        sphere = self._storage.read_sphere()
        if line_id not in sphere.lines:
            raise GDSNavigationError(
                f"Line '{line_id}' not found in sphere. "
                f"Available: {sorted(sphere.lines)}"
            )
        version = self._manifest.line_version(line_id) or 1
        table = self._storage.read_points(line_id, version)

        # -- resolve available columns (sphere metadata, not raw Lance) ----
        _line_meta = sphere.lines[line_id]
        _meta_cols = (
            [c.name for c in _line_meta.columns] if _line_meta.columns else []
        )
        _available = _meta_cols or [
            n for n in table.schema.names
            if n not in {"version", "status", "created_at", "changed_at"}
        ]

        # -- resolve property column -------------------------------------
        if property_name not in table.schema.names:
            raise GDSNavigationError(
                f"Property '{property_name}' not found in line '{line_id}'. "
                f"Available: {_available}"
            )
        col = table[property_name]
        col_type = col.type

        # -- resolve group_by column if requested ------------------------
        if group_by is not None:
            if group_by not in table.schema.names:
                raise GDSNavigationError(
                    f"Group-by column '{group_by}' not found in line "
                    f"'{line_id}'. Available: {_available}"
                )
            gb_type = table.schema.field(group_by).type
            if not (pat.is_string(gb_type) or pat.is_large_string(gb_type)
                    or pat.is_boolean(gb_type)):
                raise GDSNavigationError(
                    f"group_by column '{group_by}' must be categorical "
                    f"(string or bool), got {gb_type}"
                )

        # -- categorical (string / bool) ---------------------------------
        if pat.is_string(col_type) or pat.is_large_string(col_type) or pat.is_boolean(col_type):
            if group_by is not None:
                raise GDSNavigationError(
                    f"group_by requires a numeric property column, "
                    f"but '{property_name}' is categorical"
                )
            return self._profile_categorical(col, limit)

        # -- numeric -----------------------------------------------------
        if pat.is_integer(col_type) or pat.is_floating(col_type) or pat.is_decimal(col_type):
            if group_by is not None:
                return self._profile_numeric_grouped(table, property_name, group_by, col)
            return self._profile_numeric(col)

        # -- temporal (date / timestamp) ---------------------------------
        if pat.is_date(col_type) or pat.is_timestamp(col_type):
            if group_by is not None:
                raise GDSNavigationError(
                    f"group_by requires a numeric property column, "
                    f"but '{property_name}' is temporal"
                )
            return self._profile_temporal(col)

        # -- fallback: treat as categorical ------------------------------
        if group_by is not None:
            raise GDSNavigationError(
                f"group_by requires a numeric property column, "
                f"but '{property_name}' has unsupported type {col_type}"
            )
        return self._profile_categorical(col, limit)

    # -- profile helpers ------------------------------------------------

    @staticmethod
    def _profile_categorical(col: pa.ChunkedArray, limit: int) -> dict[str, Any]:
        total = len(col)
        null_count = col.null_count
        vc = pc.value_counts(col)
        distinct = len(vc)
        # sort descending by counts
        counts_arr = pc.struct_field(vc, "counts")
        indices = pc.sort_indices(counts_arr, sort_keys=[("not_used", "descending")])
        top_values = []
        for i in indices[:limit].to_pylist():
            entry = vc[i].as_py()
            top_values.append({"value": entry["values"], "count": entry["counts"]})
        return {
            "type": "categorical",
            "total": total,
            "null_count": null_count,
            "distinct": distinct,
            "top_values": top_values,
        }

    @staticmethod
    def _profile_numeric(col: pa.ChunkedArray) -> dict[str, Any]:
        total = len(col)
        null_count = col.null_count
        valid = pc.drop_null(col)
        if len(valid) == 0:
            return {
                "type": "numeric", "total": total, "null_count": null_count,
                "min": None, "max": None, "mean": None, "std": None,
                "median": None, "p25": None, "p75": None,
            }
        quantiles = pc.quantile(valid, q=[0.25, 0.5, 0.75]).to_pylist()
        return {
            "type": "numeric",
            "total": total,
            "null_count": null_count,
            "min": pc.min(valid).as_py(),
            "max": pc.max(valid).as_py(),
            "mean": pc.mean(valid).as_py(),
            "std": pc.stddev(valid).as_py(),
            "median": quantiles[1],
            "p25": quantiles[0],
            "p75": quantiles[2],
        }

    @staticmethod
    def _profile_temporal(col: pa.ChunkedArray) -> dict[str, Any]:
        total = len(col)
        null_count = col.null_count
        return {
            "type": "temporal",
            "total": total,
            "null_count": null_count,
            "min": pc.min(col).as_py(),
            "max": pc.max(col).as_py(),
        }

    @staticmethod
    def _profile_numeric_grouped(
        table: pa.Table,
        property_name: str,
        group_by: str,
        col: pa.ChunkedArray,
    ) -> dict[str, Any]:
        total = len(col)
        gb_col = table[group_by]
        groups_out: list[dict[str, Any]] = []
        # get unique group values
        unique_groups = pc.unique(gb_col).to_pylist()
        for gval in sorted(unique_groups, key=lambda x: (x is None, str(x))):
            mask = pc.is_null(gb_col) if gval is None else pc.equal(gb_col, gval)
            subset = pc.filter(col, mask)
            valid = pc.drop_null(subset)
            count = len(valid)
            if count == 0:
                groups_out.append({
                    "group": gval, "count": 0,
                    "min": None, "max": None, "mean": None, "std": None,
                    "median": None, "p25": None, "p75": None,
                })
                continue
            quantiles = pc.quantile(valid, q=[0.25, 0.5, 0.75]).to_pylist()
            groups_out.append({
                "group": gval,
                "count": count,
                "min": pc.min(valid).as_py(),
                "max": pc.max(valid).as_py(),
                "mean": pc.mean(valid).as_py(),
                "std": pc.stddev(valid).as_py(),
                "median": quantiles[1],
                "p25": quantiles[0],
                "p75": quantiles[2],
            })
        return {
            "type": "numeric_grouped",
            "total": total,
            "group_by": group_by,
            "groups": groups_out,
        }

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def detect_cross_pattern_discrepancy(
        self,
        entity_line: str,
        top_n: int = 50,
    ) -> list[dict]:
        """Find entities anomalous in exactly one pattern but normal elsewhere.

        Uses PassiveScanner to screen the population, then cross_pattern_profile
        to identify which single pattern flags the entity.  Returns up to top_n
        results sorted by anomalous delta_norm descending.

        Requires entity_line to be covered by 2+ patterns.  NB-Split spheres
        (patterns on isolated sibling lines with same source) are NOT detected —
        use passive_scan + composite_risk for cross-line comparisons instead.
        """
        from hypertopos.navigation.scanner import PassiveScanner

        sphere = self._storage.read_sphere()
        scanner = PassiveScanner(self._storage, sphere, self._manifest)
        # Skip graph sources — this detector measures geometry disagreement
        # between patterns, not graph contagion. Registering graph sources
        # here triggers full edge-table reads per pattern (~37s on 5M-edge
        # spheres, compounding for multi-pattern lines) with zero signal
        # contribution to the downstream single-source hit check.
        scanner.auto_discover(entity_line, include_graph=False)

        if len(scanner._sources) < 2:
            return []

        result = scanner.scan(entity_line, scoring="count", threshold=1, top_n=top_n)

        # Keep only hits flagged by exactly one source, limit to top_n
        # before the expensive per-entity cross_pattern_profile loop.
        single_source_hits = [h for h in result.hits if h.score == 1]
        single_source_hits.sort(key=lambda h: h.weighted_score, reverse=True)
        single_source_hits = single_source_hits[:top_n]

        output: list[dict] = []
        skipped_errors = 0
        for hit in single_source_hits:
            try:
                profile = self.cross_pattern_profile(
                    hit.primary_key, line_id=entity_line,
                )
            except (GDSNavigationError, GDSEntityNotFoundError, KeyError):
                skipped_errors += 1
                continue

            signals = profile.get("signals", {})
            anomalous_pattern: str | None = None
            normal_patterns: list[str] = []
            delta_norm_anomalous = 0.0
            delta_rank_pct_anomalous = 0.0

            for pat_id, sig in signals.items():
                if sig.get("is_anomaly"):
                    anomalous_pattern = pat_id
                    delta_norm_anomalous = sig.get("delta_norm", 0.0) or 0.0
                    delta_rank_pct_anomalous = sig.get("delta_rank_pct", 0.0) or 0.0
                else:
                    normal_patterns.append(pat_id)

            if anomalous_pattern is None:
                continue

            interpretation = (
                f"Entity {hit.primary_key} is anomalous only in "
                f"{anomalous_pattern} (delta_norm={delta_norm_anomalous:.3f}, "
                f"rank_pct={delta_rank_pct_anomalous:.1f}%) but normal in "
                f"{len(normal_patterns)} other pattern(s)."
            )
            output.append({
                "entity_key": hit.primary_key,
                "anomalous_pattern": anomalous_pattern,
                "normal_patterns": normal_patterns,
                "delta_norm_anomalous": delta_norm_anomalous,
                "delta_rank_pct_anomalous": delta_rank_pct_anomalous,
                "interpretation": interpretation,
            })

        output.sort(key=lambda d: d["delta_norm_anomalous"], reverse=True)
        results = output[:top_n]
        if skipped_errors > 0 and results:
            results[0]["_skipped_errors"] = skipped_errors
        return results

    def detect_neighbor_contamination(
        self,
        pattern_id: str,
        k: int = 10,
        sample_size: int = 20,
        contamination_threshold: float = 0.5,
    ) -> list[dict]:
        """Find normal entities whose geometric neighborhood is dominated by anomalies.

        Inverted search: starts from anomalous entities, finds their normal neighbors,
        then checks each normal neighbor's full neighborhood contamination rate.
        This guarantees exploration of the anomaly boundary where contaminated
        entities live, rather than random sampling that misses sparse targets.
        """
        import random

        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version,
            columns=["primary_key", "is_anomaly"],
        )
        if geo.num_rows == 0:
            return []

        pks_list = geo["primary_key"].to_pylist()
        is_anom_list = geo["is_anomaly"].to_pylist()
        anomaly_map: dict[str, bool] = {pk: bool(ia) for pk, ia in zip(pks_list, is_anom_list)}
        anomalous_keys: list[str] = [pk for pk, ia in zip(pks_list, is_anom_list) if ia]

        if not anomalous_keys:
            return []

        # Phase 1: Sample anomalous entities, find their neighbors
        anom_sample = random.sample(anomalous_keys, min(sample_size, len(anomalous_keys)))
        normal_candidates: set[str] = set()
        for key in anom_sample:
            try:
                neighbors = self.find_similar_entities(key, pattern_id, top_n=k)
                for nk, _ in neighbors:
                    if not anomaly_map.get(nk, False):
                        normal_candidates.add(nk)
            except (KeyError, GDSNavigationError):
                pass

        if not normal_candidates:
            return []

        # Phase 2: For each normal candidate, check ITS neighborhood contamination
        output: list[dict] = []
        for target_key in normal_candidates:
            try:
                neighbors = self.find_similar_entities(target_key, pattern_id, top_n=k)
                nkeys = [n[0] for n in neighbors]
            except (KeyError, GDSNavigationError):
                continue
            if not nkeys:
                continue

            # Use anomaly_map for known keys, batch-read unknowns
            unknown = [nk for nk in nkeys if nk not in anomaly_map]
            if unknown:
                unk_geo = self._storage.read_geometry(
                    pattern_id, version,
                    point_keys=unknown,
                    columns=["primary_key", "is_anomaly"],
                )
                for unk_pk, unk_ia in zip(
                    unk_geo["primary_key"].to_pylist(),
                    unk_geo["is_anomaly"].to_pylist(),
                ):
                    anomaly_map[unk_pk] = bool(unk_ia)

            anomalous_count = sum(1 for nk in nkeys if anomaly_map.get(nk, False))
            rate = anomalous_count / len(nkeys)
            if rate >= contamination_threshold:
                output.append({
                    "target_key": target_key,
                    "is_anomaly_target": False,
                    "contamination_rate": round(rate, 3),
                    "anomalous_neighbor_count": anomalous_count,
                    "total_neighbors": len(nkeys),
                    "neighbor_keys": nkeys,
                })

        output.sort(key=lambda d: d["contamination_rate"], reverse=True)
        return output

    def detect_trajectory_anomaly(
        self,
        pattern_id: str,
        displacement_ranks: list[int] | None = None,
        top_n_per_range: int = 5,
        sample_size: int = 10_000,
    ) -> list[dict]:
        """Find entities with unusual temporal trajectory shapes.

        Classifies every entity's trajectory shape (arch, v_shape,
        spike_recovery, linear_drift, flat) and returns only non-trivial
        shapes. Arch/V-shape trajectories have near-zero displacement and
        are invisible to drift ranking — hence the dedicated scan.

        sample_size: max number of distinct entities streamed before stopping.
            Defaults to 10,000. Pass None to scan the full population
            (may be slow on large spheres).
        displacement_ranks: deprecated, ignored (kept for API compatibility).
        top_n_per_range: max results returned (repurposed as top_n).

        Only works on anchor patterns with temporal history.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )
        if pattern.pattern_type == "event":
            raise ValueError(
                f"detect_trajectory_anomaly requires anchor pattern — "
                f"event patterns have no temporal history. "
                f"Got pattern '{pattern_id}' with type 'event'."
            )

        _interesting = {"arch", "v_shape", "spike_recovery"}
        _sigma = np.maximum(pattern.sigma_diag, 1e-2) if pattern.sigma_diag is not None else None
        n_rel = len(pattern.relations)
        top_n = top_n_per_range  # repurposed

        # Phase 1: Full temporal scan — stream all data, group by entity
        entity_slices: dict[str, list[tuple]] = defaultdict(list)
        try:
            for batch in self._storage.read_temporal_batched(pattern_id):
                table = pa.Table.from_batches([batch])
                if "shape_snapshot" not in table.schema.names:
                    continue
                pks = table["primary_key"].to_pylist()
                snapshots = table["shape_snapshot"].to_pylist()
                timestamps = pc.cast(table["timestamp"], pa.int64()).to_pylist()
                for pk, snap, ts in zip(pks, snapshots, timestamps, strict=True):
                    entity_slices[pk].append((ts, snap))
                if sample_size is not None and len(entity_slices) >= sample_size:
                    break
        except StopIteration:
            return []

        if not entity_slices:
            return []

        # Phase 2: Classify every entity's trajectory
        candidates: list[dict] = []
        for entity_key, slices in entity_slices.items():
            if len(slices) < 3:
                continue
            slices.sort(key=lambda x: x[0])  # sort by timestamp
            shapes_arr = np.array([s[1] for s in slices], dtype=np.float32)
            deltas = (shapes_arr - pattern.mu) / _sigma if _sigma is not None else shapes_arr
            rel_deltas = deltas[:, :n_rel]
            delta_norms = np.sqrt(
                np.einsum("ij,ij->i", rel_deltas, rel_deltas)
            ).tolist()

            shape = _classify_trajectory(delta_norms)
            if shape not in _interesting:
                continue

            first_ts = self._us_to_iso(slices[0][0])
            last_ts = self._us_to_iso(slices[-1][0])

            # Compute displacement and path_length from delta_norms
            norms_arr = np.array(delta_norms)
            displacement = abs(float(norms_arr[-1] - norms_arr[0]))
            path_length = float(np.sum(np.abs(np.diff(norms_arr))))

            # Wasted motion: path that doesn't contribute to net displacement.
            # Arch (path=10, disp=1) → wasted=9.  Flat (path=1, disp=0.5) → wasted=0.5.
            wasted_motion = path_length - displacement

            candidates.append({
                "entity_key": entity_key,
                "trajectory_shape": shape,
                "displacement": round(displacement, 3),
                "path_length": round(path_length, 3),
                "wasted_motion": round(wasted_motion, 3),
                "num_slices": len(slices),
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
            })

        # Sort by wasted_motion descending — most non-linear trajectories first
        candidates.sort(key=lambda d: d["wasted_motion"], reverse=True)
        results = candidates[:top_n]

        # Phase 3: Enrich top results with cohort info
        for entry in results:
            cohort_keys: list[str] = []
            try:
                similar = self.find_drifting_similar(
                    entry["entity_key"], pattern_id, top_n=20,
                )
                cohort_keys = [s["primary_key"] for s in similar]
            except (ValueError, GDSNavigationError):
                pass

            interpretation = (
                f"Entity {entry['entity_key']} shows '{entry['trajectory_shape']}' "
                f"trajectory over {entry['num_slices']} slices "
                f"(path_length={entry['path_length']:.3f}). "
            )
            if cohort_keys:
                interpretation += (
                    f"Cohort of {len(cohort_keys)} entities share a similar "
                    f"temporal trajectory."
                )

            entry["cohort_size"] = len(cohort_keys)
            entry["cohort_keys"] = cohort_keys
            entry["interpretation"] = interpretation

        return results

    def detect_segment_shift(
        self,
        pattern_id: str,
        max_cardinality: int = 50,
        min_shift_ratio: float = 2.0,
        top_n: int = 20,
    ) -> list[dict]:
        """Find entity segments with disproportionate anomaly rates.

        Scans string-type columns on the entity line, computes per-value anomaly
        rate, and compares to the population baseline.  Returns segments where
        shift_ratio >= min_shift_ratio, sorted descending.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )

        entity_line_id = sphere.entity_line(pattern_id)
        if not entity_line_id:
            raise GDSNavigationError(
                f"No entity line found for pattern '{pattern_id}'."
            )

        # Population anomaly rate
        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version,
            columns=["primary_key", "is_anomaly"],
        )
        if geo.num_rows == 0:
            return []

        total_population = geo.num_rows
        total_anomalies = int(pc.sum(geo["is_anomaly"]).as_py())
        population_rate = total_anomalies / total_population if total_population > 0 else 0.0

        if population_rate < 1e-6:
            return []

        # Find string columns on entity line
        line = sphere.lines.get(entity_line_id)
        if line is None or not line.columns:
            return []

        string_columns = [
            col.name for col in line.columns
            if col.type in ("string", "utf8", "str")
            and col.name != "primary_key"
        ]
        if not string_columns:
            return []

        # Read entity points for the string columns
        line_ver = self._manifest.line_version(entity_line_id) or 1
        pts = self._storage.read_points(
            entity_line_id, line_ver,
            columns=["primary_key"] + string_columns,
        )

        # Vectorized is_anomaly mask aligned with pts — single pc.is_in pass
        anomalous_pk_arr = (
            geo.filter(geo["is_anomaly"]).column("primary_key").combine_chunks()
        )
        pts_pk_arr = pts["primary_key"].combine_chunks()
        is_anom_mask = pc.is_in(pts_pk_arr, value_set=anomalous_pk_arr)

        output: list[dict] = []

        for col_name in string_columns:
            if col_name not in pts.column_names:
                continue

            col_arr = pts[col_name]

            # Early cardinality check — skip high-cardinality cols without Python loop
            n_unique = pc.count_distinct(col_arr.drop_null()).as_py()
            if n_unique > max_cardinality:
                continue

            # Vectorized groupby: total and anomaly count per segment value
            not_null_mask = pc.is_valid(col_arr)
            grp_tbl = pa.table({
                "seg": col_arr.filter(not_null_mask),
                "is_anomaly": is_anom_mask.filter(not_null_mask),
                "_pk": pts_pk_arr.filter(not_null_mask),
            })
            agg = grp_tbl.group_by("seg").aggregate([
                ("_pk", "count"),
                ("is_anomaly", "sum"),
            ])

            for i in range(agg.num_rows):
                val = agg["seg"][i].as_py()
                entity_count = int(agg["_pk_count"][i].as_py() or 0)
                anomalous_count = int(agg["is_anomaly_sum"][i].as_py() or 0)
                if entity_count == 0:
                    continue

                anomaly_rate = anomalous_count / entity_count
                shift_ratio = anomaly_rate / population_rate if population_rate > 0 else 0.0
                if shift_ratio < min_shift_ratio:
                    continue

                output.append({
                    "segment_property": col_name,
                    "segment_value": str(val),
                    "anomaly_rate": round(anomaly_rate, 4),
                    "population_rate": round(population_rate, 4),
                    "shift_ratio": round(shift_ratio, 2),
                    "entity_count": entity_count,
                    "anomalous_count": anomalous_count,
                    "interpretation": (
                        f"Segment {col_name}='{val}' has anomaly rate "
                        f"{anomaly_rate:.1%} vs population {population_rate:.1%} "
                        f"({shift_ratio:.1f}x overrepresented)."
                    ),
                })

        output.sort(key=lambda d: d["shift_ratio"], reverse=True)
        return output[:top_n]

    # ------------------------------------------------------------------
    # Detection methods — Phase 2 (false-positive, event-rate, chain,
    # hub-anomaly, composite-subgroup, collective-drift, temporal-burst)
    # ------------------------------------------------------------------

    def assess_false_positive(
        self,
        primary_key: str,
        pattern_id: str,
        n_perturbations: int = 20,
        perturbation_pct: float = 0.05,
    ) -> dict[str, Any]:
        """Assess whether an anomalous entity is a stable anomaly or borderline.

        Perturbs theta_norm by ±perturbation_pct N times and counts how many
        perturbations flip the anomaly verdict.  High stability → real anomaly,
        low stability → likely false positive near the decision boundary.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )

        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version,
            primary_key=primary_key,
            columns=["primary_key", "delta_norm", "is_anomaly"],
        )
        if geo.num_rows == 0:
            raise GDSEntityNotFoundError(
                f"Entity '{primary_key}' not found in geometry for '{pattern_id}'."
            )

        delta_norm = float(geo["delta_norm"][0].as_py())
        is_anomaly = bool(geo["is_anomaly"][0].as_py())
        theta_norm = (
            float(np.linalg.norm(pattern.theta))
            if pattern.theta is not None else 0.0
        )

        if not is_anomaly:
            return {
                "primary_key": primary_key,
                "verdict": "not_anomaly",
                "delta_norm": round(delta_norm, 4),
                "theta_norm": round(theta_norm, 4),
                "interpretation": f"Entity {primary_key} is not flagged as anomaly.",
            }

        margin = delta_norm - theta_norm
        rng = np.random.default_rng()
        offsets = rng.uniform(-perturbation_pct, perturbation_pct, size=n_perturbations)
        perturbed_thetas = theta_norm * (1.0 + offsets)
        flips = int(np.sum(delta_norm < perturbed_thetas))
        stability_score = 1.0 - (flips / n_perturbations)

        if stability_score > 0.8:
            verdict = "stable_anomaly"
        elif stability_score > 0.4:
            verdict = "borderline"
        else:
            verdict = "likely_false_positive"

        interpretation = (
            f"Entity {primary_key}: delta_norm={delta_norm:.4f}, "
            f"theta={theta_norm:.4f}, margin={margin:.4f}. "
            f"Stability={stability_score:.2f} ({flips}/{n_perturbations} flips) "
            f"→ {verdict}."
        )
        return {
            "primary_key": primary_key,
            "delta_norm": round(delta_norm, 4),
            "theta_norm": round(theta_norm, 4),
            "margin": round(margin, 4),
            "stability_score": round(stability_score, 4),
            "flips": flips,
            "verdict": verdict,
            "interpretation": interpretation,
        }

    def detect_event_rate_anomaly(
        self,
        pattern_id: str,
        threshold: float = 0.15,
        top_n: int = 20,
        min_events: int = 5,
        sample_size: int = 200_000,
    ) -> list[dict[str, Any]]:
        """Find entities with high event anomaly rate but normal anchor geometry.

        Extends _compute_event_rate_divergence by accepting an explicit anchor
        pattern_id and configurable thresholds.  For each event pattern sharing
        the anchor's entity line, reads event geometry, computes per-entity
        event anomaly rate, and cross-references with anchor geometry to find
        entities invisible to static anomaly detection.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )
        if pattern.pattern_type != "anchor":
            raise ValueError(
                f"detect_event_rate_anomaly requires anchor pattern, "
                f"got '{pattern.pattern_type}'."
            )

        anchor_line = sphere.entity_line(pattern_id)
        if not anchor_line:
            return []

        anchor_version = self._resolve_version(pattern_id)
        theta_norm = float(np.linalg.norm(pattern.theta)) if pattern.theta is not None else 0.0

        # Find event patterns sharing this anchor line
        pairs: list[tuple[str, int]] = []
        for event_pid, event_pat in sphere.patterns.items():
            if event_pat.pattern_type != "event":
                continue
            relation_lines = [r.line_id for r in event_pat.relations]
            if anchor_line in relation_lines:
                pairs.append((event_pid, relation_lines.index(anchor_line)))

        if not pairs:
            return []

        alerts: list[dict[str, Any]] = []
        for event_pid, anchor_idx in pairs:
            try:
                event_version = self._resolve_version(event_pid)
                geo = self._storage.read_geometry(
                    event_pid, event_version,
                    sample_size=sample_size,
                    columns=["is_anomaly", "entity_keys"],
                )
            except (FileNotFoundError, OSError, KeyError, GDSNavigationError):
                continue
            if geo.num_rows == 0:
                continue

            total_counts: dict[str, int] = {}
            anom_counts: dict[str, int] = {}
            ek_list = geo["entity_keys"].to_pylist()
            anom_list = geo["is_anomaly"].to_pylist()
            for ek_val, is_anom in zip(ek_list, anom_list, strict=True):
                if not ek_val or len(ek_val) <= anchor_idx or not ek_val[anchor_idx]:
                    continue
                key = ek_val[anchor_idx]
                total_counts[key] = total_counts.get(key, 0) + 1
                if is_anom:
                    anom_counts[key] = anom_counts.get(key, 0) + 1

            high_rate_keys = [
                k for k, total in total_counts.items()
                if total >= min_events and anom_counts.get(k, 0) / total > threshold
            ]
            if not high_rate_keys:
                continue

            # Cross-reference with anchor geometry
            try:
                anchor_geo = self._storage.read_geometry(
                    pattern_id, anchor_version,
                    point_keys=high_rate_keys,
                    columns=["primary_key", "delta_norm", "is_anomaly"],
                )
            except (FileNotFoundError, OSError, KeyError):
                continue

            for i in range(anchor_geo.num_rows):
                pk = anchor_geo["primary_key"][i].as_py()
                is_anchor_anomaly = bool(anchor_geo["is_anomaly"][i].as_py())
                if is_anchor_anomaly:
                    continue
                total = total_counts.get(pk, 0)
                rate = anom_counts.get(pk, 0) / total if total > 0 else 0.0
                delta_norm = float(anchor_geo["delta_norm"][i].as_py())
                interpretation = (
                    f"Entity {pk}: event anomaly rate {rate:.0%} in "
                    f"{event_pid} ({anom_counts.get(pk, 0)}/{total} events) "
                    f"but normal in anchor (delta_norm={delta_norm:.4f} < "
                    f"theta={theta_norm:.4f})."
                )
                alerts.append({
                    "entity_key": pk,
                    "event_pattern_id": event_pid,
                    "event_anomaly_rate": round(rate, 4),
                    "event_total": total,
                    "event_anomalous": anom_counts.get(pk, 0),
                    "anchor_delta_norm": round(delta_norm, 4),
                    "theta_norm": round(theta_norm, 4),
                    "interpretation": interpretation,
                })

        alerts.sort(key=lambda d: d["event_anomaly_rate"], reverse=True)
        return alerts[:top_n]

    def detect_hub_anomaly_concentration(
        self,
        pattern_id: str,
        top_n: int = 20,
        min_anomaly_ratio: float = 0.5,
        hub_top_n: int = 20,
        neighbor_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find hubs whose geometric neighborhood is dominated by anomalies.

        Gets top hubs via π7, then for each hub checks the anomaly ratio
        among its nearest neighbors.  Returns hubs where neighbor anomaly
        ratio >= min_anomaly_ratio, sorted by ratio descending.
        """
        version = self._resolve_version(pattern_id)

        # Get top hubs
        hubs = self.π7_attract_hub(pattern_id, top_n=hub_top_n)
        if not hubs:
            return []

        # Preload anomaly map
        geo = self._storage.read_geometry(
            pattern_id, version,
            columns=["primary_key", "is_anomaly"],
        )
        anomaly_map = dict(zip(
            geo["primary_key"].to_pylist(),
            [bool(v) for v in geo["is_anomaly"].to_pylist()],
        ))

        output: list[dict[str, Any]] = []
        for pk, edge_count, hub_score in hubs:
            try:
                neighbors = self.find_similar_entities(
                    pk, pattern_id, top_n=neighbor_k,
                )
            except (GDSNavigationError, GDSEntityNotFoundError, KeyError):
                continue

            if not neighbors:
                continue

            nkeys = [nk for nk, _ in neighbors]
            anom_count = sum(1 for nk in nkeys if anomaly_map.get(nk, False))
            ratio = anom_count / len(nkeys)

            if ratio < min_anomaly_ratio:
                continue

            is_hub_anomaly = anomaly_map.get(pk, False)
            interpretation = (
                f"Hub {pk} (score={hub_score:.3f}, edges={edge_count}): "
                f"{anom_count}/{len(nkeys)} neighbors anomalous ({ratio:.0%}). "
                f"Hub itself {'IS' if is_hub_anomaly else 'is NOT'} anomalous."
            )
            output.append({
                "hub_key": pk,
                "hub_score": round(hub_score, 4),
                "edge_count": edge_count,
                "is_hub_anomaly": is_hub_anomaly,
                "neighbor_anomaly_ratio": round(ratio, 4),
                "anomalous_neighbor_count": anom_count,
                "total_neighbors": len(nkeys),
                "interpretation": interpretation,
            })

        output.sort(key=lambda d: d["neighbor_anomaly_ratio"], reverse=True)
        return output[:top_n]

    def detect_composite_subgroup_inflation(
        self,
        entity_line: str,
        group_by: str,
        top_n: int = 10,
        min_inflation: float = 1.5,
        sample_per_group: int = 10,
    ) -> list[dict[str, Any]]:
        """Find subgroups with inflated composite risk vs population baseline.

        Uses aggregate_anomalies for group breakdown, then composite_risk_batch
        on sampled keys per group.  Compares mean group composite risk to the
        population baseline and returns groups with inflation >= min_inflation.
        """
        import random

        sphere = self._storage.read_sphere()

        # Find anchor patterns for this entity line
        anchor_pids = [
            pid for pid, pat in sphere.patterns.items()
            if pat.pattern_type == "anchor"
            and sphere.entity_line(pid) == entity_line
        ]
        if not anchor_pids:
            return []

        # Use first anchor for grouping
        anchor_pid = anchor_pids[0]
        agg = self.aggregate_anomalies(
            anchor_pid, group_by,
            include_keys=True,
            keys_per_group=sample_per_group,
        )

        groups = agg.get("groups", [])
        if not groups:
            return []

        # Population baseline — sample random keys for composite risk
        version = self._resolve_version(anchor_pid)
        geo = self._storage.read_geometry(
            anchor_pid, version, columns=["primary_key"],
            sample_size=min(sample_per_group * 20, 1000),
        )
        all_keys = geo["primary_key"].to_pylist()
        pop_sample = random.sample(all_keys, min(sample_per_group * 10, len(all_keys)))
        pop_risk = self.composite_risk_batch(pop_sample, line_id=entity_line)
        pop_scores = [
            v["combined_p"] for v in pop_risk.get("results", [])
            if isinstance(v, dict) and v.get("combined_p") is not None
        ]
        pop_mean = float(np.mean(pop_scores)) if pop_scores else 0.5

        output: list[dict[str, Any]] = []
        for grp in groups:
            keys = grp.get("entity_keys", [])
            if not keys:
                continue
            sample_keys = keys[:sample_per_group]
            try:
                grp_risk = self.composite_risk_batch(
                    sample_keys, line_id=entity_line,
                )
            except (GDSNavigationError, ValueError):
                continue

            grp_scores = [
                v["combined_p"] for v in grp_risk.get("results", [])
                if isinstance(v, dict) and v.get("combined_p") is not None
            ]
            if not grp_scores:
                continue

            grp_mean = float(np.mean(grp_scores))
            # Lower combined_p = higher risk; inflation = pop/group
            inflation = pop_mean / grp_mean if grp_mean > 1e-6 else 0.0

            if inflation < min_inflation:
                continue

            interpretation = (
                f"Subgroup {group_by}='{grp.get('value', '?')}': "
                f"mean composite p={grp_mean:.4f} vs population p={pop_mean:.4f} "
                f"({inflation:.1f}x inflation). "
                f"Sampled {len(sample_keys)} of {grp.get('count', len(keys))} entities."
            )
            output.append({
                "group_value": grp.get("value"),
                "group_count": grp.get("count", len(keys)),
                "group_anomaly_count": grp.get("anomaly_count", 0),
                "group_mean_p": round(grp_mean, 4),
                "population_mean_p": round(pop_mean, 4),
                "inflation_ratio": round(inflation, 2),
                "sampled_keys": len(sample_keys),
                "interpretation": interpretation,
            })

        output.sort(key=lambda d: d["inflation_ratio"], reverse=True)
        return output[:top_n]

    def detect_collective_drift(
        self,
        pattern_id: str,
        top_n: int = 100,
        n_clusters: int = 5,
        min_cluster_size: int = 5,
        sample_size: int = 5000,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Find clusters of entities drifting in the same geometric direction.

        Reads drift data via π9, extracts dimension_diffs as drift vectors,
        normalizes to unit vectors, and clusters via k-means.  Returns clusters
        where entities share a coherent drift direction (high mean cosine
        similarity), sorted by cluster size descending.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )
        if pattern.pattern_type == "event":
            raise ValueError(
                "detect_collective_drift requires anchor pattern — "
                "event patterns have no temporal history."
            )

        # Over-fetch to get enough drift data for clustering
        drift_results = self.π9_attract_drift(
            pattern_id, top_n=sample_size, sample_size=sample_size,
        )
        if len(drift_results) < min_cluster_size:
            return []

        # Build drift vector matrix from dimension_diffs
        dim_keys: list[str] = []
        if drift_results:
            dim_keys = list(drift_results[0].get("dimension_diffs", {}).keys())
        if not dim_keys:
            return []

        keys: list[str] = []
        vectors: list[list[float]] = []
        for entry in drift_results:
            dd = entry.get("dimension_diffs", {})
            vec = [dd.get(d, 0.0) for d in dim_keys]
            keys.append(entry["primary_key"])
            vectors.append(vec)

        mat = np.array(vectors, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        unit_mat = mat / norms

        # K-means clustering on unit drift vectors
        rng = np.random.default_rng(seed)
        n = unit_mat.shape[0]
        k = min(n_clusters, n)
        # K-means++ init
        centers = np.empty((k, unit_mat.shape[1]), dtype=np.float64)
        idx = rng.integers(0, n)
        centers[0] = unit_mat[idx]
        for c in range(1, k):
            dists = np.min(
                np.array([np.sum((unit_mat - centers[j]) ** 2, axis=1) for j in range(c)]),
                axis=0,
            )
            probs = dists / (dists.sum() + 1e-12)
            idx = rng.choice(n, p=probs)
            centers[c] = unit_mat[idx]

        # Iterate
        for _ in range(50):
            dists_all = np.array([
                np.sum((unit_mat - centers[j]) ** 2, axis=1) for j in range(k)
            ])  # shape (k, n)
            labels = np.argmin(dists_all, axis=0)
            new_centers = np.empty_like(centers)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centers[j] = unit_mat[mask].mean(axis=0)
                    norm_j = np.linalg.norm(new_centers[j])
                    if norm_j > 1e-8:
                        new_centers[j] /= norm_j
                else:
                    new_centers[j] = centers[j]
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        # Build cluster results
        output: list[dict[str, Any]] = []
        for j in range(k):
            mask = labels == j
            cluster_size = int(mask.sum())
            if cluster_size < min_cluster_size:
                continue

            cluster_vecs = unit_mat[mask]
            centroid = centers[j]
            cosines = cluster_vecs @ centroid
            mean_cosine = float(np.mean(cosines))

            cluster_keys = [keys[i] for i in range(n) if labels[i] == j]
            drift_direction = {
                dim_keys[d]: round(float(centroid[d]), 4)
                for d in range(len(dim_keys))
            }

            interpretation = (
                f"Cluster of {cluster_size} entities drifting in coherent "
                f"direction (mean cosine={mean_cosine:.3f}). "
                f"Dominant drift dimensions: "
                + ", ".join(
                    f"{d}={v:+.3f}" for d, v in
                    sorted(drift_direction.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                )
            )
            output.append({
                "cluster_id": j,
                "cluster_size": cluster_size,
                "mean_cosine_similarity": round(mean_cosine, 4),
                "drift_direction": drift_direction,
                "entity_keys": cluster_keys[:50],
                "interpretation": interpretation,
            })

        output.sort(key=lambda d: d["cluster_size"], reverse=True)
        return output[:top_n]

    def detect_temporal_burst(
        self,
        pattern_id: str,
        window_days: int = 30,
        z_threshold: float = 3.0,
        top_n: int = 20,
        sample_size: int = 50_000,
    ) -> list[dict[str, Any]]:
        """Find entities with bursty event patterns via z-score on rolling windows.

        Reads event timestamps grouped by entity, computes rolling window event
        counts, and flags entities whose peak window count exceeds z_threshold
        standard deviations above the population mean.
        """
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )
        if pattern.pattern_type != "event":
            raise ValueError(
                f"detect_temporal_burst requires event pattern, "
                f"got '{pattern.pattern_type}'."
            )

        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id, version,
            sample_size=sample_size,
            columns=["primary_key", "entity_keys"],
        )
        if geo.num_rows == 0:
            return []

        # Determine the anchor line index (first relation)
        if not pattern.relations:
            return []
        anchor_idx = 0

        # Read timestamps from temporal data or geometry
        # For event patterns, use geometry timestamps if available
        # Probe schema for timestamp columns (no data read)
        ts_col = None
        schema = self._storage.read_geometry(
            pattern_id, version, sample_size=0,
        ).schema
        for col_name in ("timestamp", "event_timestamp"):
            if col_name in schema.names:
                ts_col = col_name
                break
        if ts_col:
            geo = self._storage.read_geometry(
                pattern_id, version,
                sample_size=sample_size,
                columns=["entity_keys", ts_col],
            )

        if ts_col is None:
            # Fallback: count events per entity without temporal windowing
            entity_counts: dict[str, int] = {}
            for ek_val in geo["entity_keys"].to_pylist():
                if not ek_val or len(ek_val) <= anchor_idx or not ek_val[anchor_idx]:
                    continue
                key = ek_val[anchor_idx]
                entity_counts[key] = entity_counts.get(key, 0) + 1

            if not entity_counts:
                return []

            counts_arr = np.array(list(entity_counts.values()), dtype=np.float64)
            mean_count = float(np.mean(counts_arr))
            std_count = float(np.std(counts_arr))
            if std_count < 1e-6:
                return []

            output: list[dict[str, Any]] = []
            for key, count in entity_counts.items():
                z = (count - mean_count) / std_count
                if z < z_threshold:
                    continue
                interpretation = (
                    f"Entity {key}: {count} events (z={z:.2f}, "
                    f"mean={mean_count:.1f}, std={std_count:.1f}). "
                    f"Burst detected via event count z-score."
                )
                output.append({
                    "entity_key": key,
                    "event_count": count,
                    "z_score": round(z, 4),
                    "population_mean": round(mean_count, 2),
                    "population_std": round(std_count, 2),
                    "interpretation": interpretation,
                })
            output.sort(key=lambda d: d["z_score"], reverse=True)
            return output[:top_n]

        # Temporal windowing path
        window_us = window_days * 86_400 * 1_000_000
        entity_timestamps: dict[str, list[int]] = defaultdict(list)

        ek_list = geo["entity_keys"].to_pylist()
        ts_raw = pc.cast(geo[ts_col], pa.int64()).to_pylist()

        for ek_val, ts_val in zip(ek_list, ts_raw, strict=True):
            if not ek_val or len(ek_val) <= anchor_idx or not ek_val[anchor_idx]:
                continue
            if ts_val is None:
                continue
            entity_timestamps[ek_val[anchor_idx]].append(ts_val)

        if not entity_timestamps:
            return []

        # Compute max rolling window count per entity
        peak_counts: dict[str, int] = {}
        for key, timestamps in entity_timestamps.items():
            if len(timestamps) < 2:
                continue
            ts_arr = np.sort(np.array(timestamps, dtype=np.int64))
            max_count = 0
            left = 0
            for right in range(len(ts_arr)):
                while ts_arr[right] - ts_arr[left] > window_us:
                    left += 1
                max_count = max(max_count, right - left + 1)
            peak_counts[key] = max_count

        if not peak_counts:
            return []

        peaks_arr = np.array(list(peak_counts.values()), dtype=np.float64)
        mean_peak = float(np.mean(peaks_arr))
        std_peak = float(np.std(peaks_arr))
        if std_peak < 1e-6:
            return []

        output = []
        for key, peak in peak_counts.items():
            z = (peak - mean_peak) / std_peak
            if z < z_threshold:
                continue
            interpretation = (
                f"Entity {key}: peak {peak} events in {window_days}d window "
                f"(z={z:.2f}, mean={mean_peak:.1f}, std={std_peak:.1f}). "
                f"Temporal burst detected."
            )
            output.append({
                "entity_key": key,
                "peak_window_count": peak,
                "window_days": window_days,
                "total_events": len(entity_timestamps[key]),
                "z_score": round(z, 4),
                "population_mean": round(mean_peak, 2),
                "population_std": round(std_peak, 2),
                "interpretation": interpretation,
            })

        output.sort(key=lambda d: d["z_score"], reverse=True)
        return output[:top_n]

    # ------------------------------------------------------------------
    # Batch / scan helpers (moved from MCP step handlers)
    # ------------------------------------------------------------------

    def check_anomaly_batch(
        self,
        pattern_id: str,
        primary_keys: list[str],
        max_keys: int = 500,
    ) -> dict[str, Any]:
        """Check anomaly status for multiple entities in one geometry read."""
        version = self._resolve_version(pattern_id)
        geo = self._storage.read_geometry(
            pattern_id,
            version,
            point_keys=primary_keys[:max_keys],
            columns=["primary_key", "is_anomaly", "delta_rank_pct"],
        )
        results: list[dict[str, Any]] = [
            {
                "primary_key": geo["primary_key"][i].as_py(),
                "is_anomaly": bool(geo["is_anomaly"][i].as_py()),
                "delta_rank_pct": round(float(geo["delta_rank_pct"][i].as_py()), 2),
            }
            for i in range(geo.num_rows)
        ]
        anomalous_count = sum(1 for r in results if r["is_anomaly"])
        return {
            "total": len(results),
            "anomalous_count": anomalous_count,
            "results": results,
            "interpretation": f"Checked {len(results)} entities: {anomalous_count} anomalous.",
        }

    def passive_scan(
        self,
        home_line_id: str,
        threshold: int = 2,
        top_n: int = 100,
    ) -> dict[str, Any]:
        """Multi-source anomaly screening via PassiveScanner."""
        from hypertopos.navigation.scanner import PassiveScanner

        scanner = PassiveScanner(
            self._storage,
            self._storage.read_sphere(),
            self._manifest,
        )
        scanner.auto_discover(home_line_id)
        result = scanner.scan(
            home_line_id, scoring="count", threshold=threshold, top_n=top_n,
        )
        return {
            "total_flagged": result.total_flagged,
            "hits": [
                {
                    "primary_key": h.primary_key,
                    "score": h.score,
                    "weighted_score": h.weighted_score,
                }
                for h in result.hits
            ],
            "interpretation": f"Passive scan flagged {result.total_flagged} entities at threshold={threshold}.",
        }

    def extract_chains(
        self,
        event_pattern_id: str,
        from_col: str,
        to_col: str,
        time_col: str | None = None,
        category_col: str | None = None,
        amount_col: str | None = None,
        time_window_hours: int = 168,
        max_hops: int = 15,
        min_hops: int = 2,
        top_n: int = 20,
        sort_by: str = "hop_count",
        sample_size: int | None = 50_000,
        max_chains: int = 100_000,
        seed_nodes: list[str] | None = None,
        bidirectional: bool = False,
    ) -> dict[str, Any]:
        """Extract transaction chains by following from_col->to_col links.

        Reads the event points table, builds adjacency from from_col/to_col,
        then delegates to :func:`hypertopos.engine.chains.extract_chains`.
        """
        sphere = self._storage.read_sphere()

        # Resolve to points table
        if event_pattern_id in sphere.patterns:
            version = self._resolve_version(event_pattern_id)
            line_id = sphere.entity_line(event_pattern_id)
            if not line_id:
                line_id = sphere.event_line(event_pattern_id)
            if not line_id:
                raise GDSNavigationError(
                    f"Cannot resolve line for pattern '{event_pattern_id}'. "
                    f"Available lines: {sorted(sphere.lines)}"
                )
            points_table = self._storage.read_points(line_id, version)
        elif event_pattern_id in sphere.lines:
            line = sphere.lines[event_pattern_id]
            version = line.versions[-1] if line.versions else 1
            points_table = self._storage.read_points(event_pattern_id, version)
        else:
            raise GDSNavigationError(
                f"'{event_pattern_id}' is neither a pattern nor a line. "
                f"Available patterns: {sorted(sphere.patterns)}, "
                f"lines: {sorted(sphere.lines)}"
            )

        # Validate required columns
        schema_names = {col.name for col in points_table.schema}
        for col_name, col_label in [(from_col, "from_col"), (to_col, "to_col")]:
            if col_name not in schema_names:
                raise GDSNavigationError(
                    f"{col_label}='{col_name}' not found in line schema. "
                    f"Available columns: {sorted(schema_names)}"
                )

        # Select only needed columns
        needed_cols = ["primary_key", from_col, to_col]
        if time_col and time_col in points_table.schema.names:
            needed_cols.append(time_col)
        if category_col and category_col in points_table.schema.names:
            needed_cols.append(category_col)
        if amount_col and amount_col in points_table.schema.names:
            needed_cols.append(amount_col)
        points_table = points_table.select(needed_cols)

        from_keys = points_table[from_col].to_pylist()
        to_keys = points_table[to_col].to_pylist()
        event_pks = points_table["primary_key"].to_pylist()

        timestamps = None
        if time_col and time_col in points_table.schema.names:
            from hypertopos.engine.chains import parse_timestamps_to_epoch

            ts_raw = points_table[time_col].to_pylist()
            timestamps = parse_timestamps_to_epoch(ts_raw)

        categories = None
        if category_col and category_col in points_table.schema.names:
            categories = points_table[category_col].to_pylist()

        amounts = None
        if amount_col and amount_col in points_table.schema.names:
            amounts = [
                float(v) if v is not None else 0.0
                for v in points_table[amount_col].to_pylist()
            ]

        from hypertopos.engine.chains import extract_chains as _core_extract

        chains = _core_extract(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            timestamps=timestamps,
            categories=categories,
            amounts=amounts,
            time_window_hours=time_window_hours,
            max_hops=max_hops,
            min_hops=min_hops,
            sample_size=sample_size,
            max_chains=max_chains,
            seed_nodes=seed_nodes,
            bidirectional=bidirectional,
        )

        # Sort
        if sort_by == "hop_count":
            chains.sort(key=lambda c: c.hop_count, reverse=True)
        elif sort_by == "amount_decay":
            chains.sort(key=lambda c: c.amount_decay)

        result_chains = [c.to_dict() for c in chains[:top_n]]

        resp: dict[str, Any] = {
            "event_pattern_id": event_pattern_id,
            "total_chains": len(chains),
            "returned": len(result_chains),
            "sort_by": sort_by,
            "chains": result_chains,
        }

        if chains:
            hops = [c.hop_count for c in chains]
            cyclic_count = sum(1 for c in chains if c.is_cyclic)
            resp["summary"] = {
                "total_chains": len(chains),
                "cyclic_chains": cyclic_count,
                "hop_count_mean": round(float(np.mean(hops)), 1),
                "hop_count_max": max(hops),
            }

        resp["interpretation"] = (
            f"Extracted {len(chains)} chains ({len(result_chains)} returned), "
            f"max_hops={max_hops}, min_hops={min_hops}."
        )

        return resp

    # ------------------------------------------------------------------
    # Geometric heredity — novelty scoring
    # ------------------------------------------------------------------

    def find_novel_entities(
        self,
        pattern_id: str,
        top_n: int = 10,
        *,
        sample_size: int = 5000,
    ) -> list[dict]:
        """Find entities whose geometry diverges most from their graph neighbors.

        For each entity the *expected* delta is the mean of its neighbors'
        deltas (looked up via the edge table). The novelty score is the L2
        distance between the entity's actual delta and that expected delta.

        Parameters
        ----------
        pattern_id:
            Pattern whose edge table defines the neighborhood graph. May be
            an event pattern — the anchor pattern is resolved automatically
            for geometry lookup.
        top_n:
            Number of top-scoring entities to return.
        sample_size:
            Max entities to evaluate. When the population exceeds this,
            a random sample is drawn.

        Returns
        -------
        list[dict]
            Sorted descending by ``novelty_score``. Each entry contains
            ``{primary_key, novelty_score, n_neighbors}``.

        Raises
        ------
        GDSNavigationError
            If *pattern_id* has no edge table.
        """
        if not self._storage.has_edge_table(pattern_id):
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' has no edge table. "
                "find_novel_entities requires an edge table."
            )

        from hypertopos.engine.heredity import (
            compute_expected_delta,
            compute_novelty_score,
        )

        # Resolve anchor pattern that holds entity geometry
        scoring_pattern = (
            self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        )
        version = self._resolve_version(scoring_pattern)

        # Load geometry (sampled if large)
        geo = self._storage.read_geometry(
            scoring_pattern, version,
            columns=["primary_key", "delta"],
            sample_size=sample_size,
        )
        if geo.num_rows == 0:
            return []

        # Build delta lookup: primary_key → np.ndarray
        pk_list = geo["primary_key"].to_pylist()
        delta_list = geo["delta"].to_pylist()
        delta_lookup: dict[str, np.ndarray] = {}
        for pk, d in zip(pk_list, delta_list, strict=False):
            if d is not None:
                delta_lookup[pk] = np.array(d, dtype=np.float32)

        if not delta_lookup:
            return []

        entity_keys = set(delta_lookup.keys())

        # Build adjacency for sampled entities
        adj_index = self._storage.get_adjacency(pattern_id)

        # Score each entity
        scored: list[tuple[str, float, int]] = []
        for pk, actual_delta in delta_lookup.items():
            neighbors = adj_index.neighbors_out(pk) + adj_index.neighbors_in(pk)
            # Collect neighbor deltas — only those present in our delta_lookup
            neighbor_deltas_list = []
            for nb_key, _ts, _amt, _ek in neighbors:
                nb_delta = delta_lookup.get(nb_key)
                if nb_delta is not None:
                    neighbor_deltas_list.append(nb_delta)
            n_neighbors = len(neighbor_deltas_list)
            if n_neighbors == 0:
                continue
            neighbor_deltas = np.array(neighbor_deltas_list, dtype=np.float32)
            expected = compute_expected_delta(neighbor_deltas)
            score = compute_novelty_score(actual_delta, expected)
            scored.append((pk, score, n_neighbors))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "primary_key": pk,
                "novelty_score": round(score, 6),
                "n_neighbors": n_nb,
            }
            for pk, score, n_nb in scored[:top_n]
        ]

    def compare_calibrations(
        self,
        pattern_id: str,
        v_from: int | None = None,
        v_to: int | None = None,
        top_n: int = 10,
        verbose: bool = False,
    ) -> CalibrationDriftReport:
        """Per-dimension μ/σ/θ drift between two calibration epochs of the same pattern.

        Auto-resolve:
          - v_from=None and v_to=None → (versions[-2], versions[-1])
          - v_from=N and v_to=None → (N, latest)

        Raises ValueError on v_from == v_to, schema_hash mismatch, or single-epoch
        auto-resolve. CalibrationNotFoundError bubbles up from missing versions.
        """
        reader = self._storage
        versions = reader.list_calibration_versions(pattern_id)

        if v_from is None and v_to is None:
            if len(versions) < 2:
                raise ValueError(
                    f"compare_calibrations requires at least 2 epochs on disk for "
                    f"pattern={pattern_id!r}; found {len(versions)}"
                )
            v_from = versions[-2]
            v_to = versions[-1]
        elif v_to is None:
            if not versions:
                raise ValueError(
                    f"compare_calibrations: no epochs on disk for pattern={pattern_id!r}"
                )
            v_to = versions[-1]
        elif v_from is None:
            raise ValueError(
                "compare_calibrations: v_from cannot be None when v_to is explicit"
            )

        if v_from == v_to:
            raise ValueError(
                f"compare_calibrations: v_from and v_to must differ (both={v_from})"
            )

        fit_from = reader.read_calibration_fit(pattern_id, version=v_from)
        fit_to = reader.read_calibration_fit(pattern_id, version=v_to)

        if fit_from.schema_hash != fit_to.schema_hash:
            raise ValueError(
                f"compare_calibrations: schema_hash mismatch between v={v_from} "
                f"({fit_from.schema_hash[:12]}...) and v={v_to} "
                f"({fit_to.schema_hash[:12]}...) — mu vectors are not "
                f"dimensionally comparable across schema changes"
            )

        return _compute_calibration_drift(fit_from, fit_to, top_n=top_n, verbose=verbose)

    def decompose_drift(
        self,
        entity_key: str,
        pattern_id: str,
        v_from: int | None = None,
        v_to: int | None = None,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
        top_n: int = 10,
        verbose: bool = False,
    ) -> IntrinsicExtrinsicReport:
        """Decompose an entity's drift between two temporal slices into intrinsic
        (entity-driven, σ_v1-normalised shape change) and extrinsic (residual,
        population-recalibration-driven) components.

        Auto-resolve:
          - v_from=None and v_to=None → (versions[0] is oldest retained; v_to=versions[-1])
          - v_from=N and v_to=None → (N, latest)
          - timestamp_from=None → first slice in window
          - timestamp_to=None → last slice in window

        Raises ValueError on:
          - <2 retained calibration epochs (auto-resolve)
          - v_from == v_to
          - schema_hash mismatch between v_from and v_to
          - <2 temporal slices in the window
          - event pattern (M3 requires anchor — temporal data is anchor-only)

        CalibrationNotFoundError bubbles up from the underlying reader if a
        requested version was GC'd by M1 retention policy.
        """
        from hypertopos.engine.geometry import (
            _compute_intrinsic_extrinsic_decomposition,
        )

        reader = self._storage
        sphere = reader.read_sphere()
        pattern = sphere.patterns[pattern_id]

        if pattern.pattern_type == "event":
            raise ValueError(
                f"decompose_drift: pattern_type 'event' has no temporal history; "
                f"M3 requires anchor pattern (got pattern_id={pattern_id!r})"
            )

        versions = reader.list_calibration_versions(pattern_id)

        if v_from is None and v_to is None:
            if len(versions) < 2:
                raise ValueError(
                    f"decompose_drift requires at least 2 epochs on disk for "
                    f"pattern={pattern_id!r}; found {len(versions)}"
                )
            v_from = versions[0]
            v_to = versions[-1]
        elif v_to is None:
            if not versions:
                raise ValueError(
                    f"decompose_drift: no epochs on disk for pattern={pattern_id!r}"
                )
            v_to = versions[-1]
        elif v_from is None:
            raise ValueError(
                "decompose_drift: v_from cannot be None when v_to is explicit"
            )

        if v_from == v_to:
            raise ValueError(
                f"decompose_drift: v_from and v_to must differ (both={v_from})"
            )

        fit_v1 = reader.read_calibration_fit(pattern_id, version=v_from)
        fit_v2 = reader.read_calibration_fit(pattern_id, version=v_to)

        if fit_v1.schema_hash != fit_v2.schema_hash:
            raise ValueError(
                f"decompose_drift: schema_hash mismatch between v={v_from} "
                f"({fit_v1.schema_hash[:12]}...) and v={v_to} "
                f"({fit_v2.schema_hash[:12]}...) — mu vectors are not "
                f"dimensionally comparable across schema changes"
            )

        table = reader.read_temporal(
            pattern_id, entity_key, agent_id=self._manifest.agent_id,
        )
        if table.num_rows == 0:
            raise ValueError(
                f"decompose_drift: no temporal slices for entity={entity_key!r} "
                f"in pattern={pattern_id!r}"
            )

        timestamps = table["timestamp"].to_pylist()
        shapes = table["shape_snapshot"].to_pylist()
        rows = sorted(zip(timestamps, shapes), key=lambda r: r[0])

        if timestamp_from is not None:
            rows = [r for r in rows if r[0] >= timestamp_from]
        if timestamp_to is not None:
            rows = [r for r in rows if r[0] <= timestamp_to]

        if len(rows) < 2:
            raise ValueError(
                f"decompose_drift requires at least 2 temporal slices for "
                f"entity={entity_key!r} pattern={pattern_id!r} in window "
                f"({timestamp_from!r}..{timestamp_to!r}); found {len(rows)}"
            )

        ts_a, shape_a_list = rows[0]
        ts_b, shape_b_list = rows[-1]
        shape_a = np.asarray(shape_a_list, dtype=np.float32)
        shape_b = np.asarray(shape_b_list, dtype=np.float32)

        return _compute_intrinsic_extrinsic_decomposition(
            shape_a=shape_a, shape_b=shape_b,
            fit_v1=fit_v1, fit_v2=fit_v2,
            entity_key=entity_key, pattern_id=pattern_id,
            timestamp_from=ts_a, timestamp_to=ts_b,
            dim_labels=pattern.dim_labels,
            top_n=top_n, verbose=verbose,
        )

    def find_calibration_influencers(
        self,
        pattern_id: str,
        top_n: int = 10,
        classify: str = "hidden",
        high_threshold_pct: float = 90.0,
        sample_size: int | None = None,
        verbose: bool = False,
    ):
        """M4: find entities with high influence on coordinate system calibration.

        See `docs/plans/2026-04-28-m4-hidden-influencer-design.md`.
        """
        from hypertopos.engine.geometry import (
            _classify_influence,
            _compute_leave_one_out_impact,
            _count_cascading_flips,
        )
        from hypertopos.model.sphere import (
            DimensionContribution,
            InfluenceEntry,
            InfluenceReport,
        )
        from hypertopos.utils.arrow import delta_matrix_from_arrow

        valid_classify = {"hidden", "distorter", "standard_anomaly", "normal", "all"}
        if classify not in valid_classify:
            raise ValueError(
                f"find_calibration_influencers: classify must be one of "
                f"{sorted(valid_classify)}; got {classify!r}"
            )
        if not 0.0 < high_threshold_pct < 100.0:
            raise ValueError(
                f"find_calibration_influencers: high_threshold_pct must be "
                f"in (0, 100); got {high_threshold_pct}"
            )
        if top_n < 1 or top_n > 50:
            raise ValueError(
                f"find_calibration_influencers: top_n must be in [1, 50]; "
                f"got {top_n}"
            )

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type == "event":
            raise ValueError(
                f"find_calibration_influencers: pattern_type 'event' has no "
                f"population statistics; M4 requires anchor pattern "
                f"(got pattern_id={pattern_id!r})"
            )

        version = self._manifest.pattern_versions[pattern_id]
        geo_table = self._storage.read_geometry(
            pattern_id, version,
            columns=["primary_key", "delta"],
            sample_size=sample_size,
        )
        if geo_table.num_rows == 0:
            raise ValueError(
                f"find_calibration_influencers: pattern={pattern_id!r} has no entities"
            )
        if geo_table.num_rows < 2:
            raise ValueError(
                f"find_calibration_influencers: leave-one-out requires N >= 2 "
                f"entities; pattern={pattern_id!r} has N={geo_table.num_rows}"
            )

        deltas = delta_matrix_from_arrow(geo_table).astype(np.float64)
        pk_col = geo_table["primary_key"].to_pylist()

        sigma_diag = np.asarray(pattern.sigma_diag, dtype=np.float64)
        mu = np.asarray(pattern.mu, dtype=np.float64)
        sigma_floor = np.maximum(sigma_diag, 1e-2)
        shapes = deltas * sigma_floor + mu
        N, D = shapes.shape

        mu_full = shapes.mean(axis=0)
        sigma_full = shapes.std(axis=0, ddof=0)
        delta_norm = np.linalg.norm(deltas, axis=1)
        theta_norm = float(pattern.theta_norm)

        mu_imp, sigma_imp, total_imp, contrib = _compute_leave_one_out_impact(
            shapes, mu_full, sigma_full,
        )
        classes = _classify_influence(
            total_imp, delta_norm, theta_norm, high_threshold_pct=high_threshold_pct,
        )
        impact_threshold = float(np.percentile(total_imp, high_threshold_pct))

        cell_counts = {
            "hidden": classes.count("hidden"),
            "distorter": classes.count("distorter"),
            "standard_anomaly": classes.count("standard_anomaly"),
            "normal": classes.count("normal"),
        }

        if classify == "all":
            candidates = list(range(N))
        else:
            candidates = [i for i, c in enumerate(classes) if c == classify]
        candidates.sort(key=lambda i: total_imp[i], reverse=True)
        top_candidates = candidates[:top_n]

        sum_s = shapes.sum(axis=0)
        sum_s_sq = (shapes ** 2).sum(axis=0)
        sigma_full_safe = np.maximum(sigma_full, 1e-12)
        if verbose:
            is_anom_full = delta_norm >= theta_norm

        kinds_v = pattern.dimension_kinds
        dim_labels_v = pattern.dim_labels

        entries: list[InfluenceEntry] = []
        for i in top_candidates:
            contrib_i = contrib[i]
            ranked_dim_idx = np.argsort(-np.abs(contrib_i))[: min(5, D)]
            mu_without_i = (sum_s - shapes[i]) / (N - 1)
            var_without_i = (sum_s_sq - shapes[i] ** 2) / (N - 1) - mu_without_i ** 2
            var_without_i = np.maximum(var_without_i, 0.0)
            sigma_without_i = np.sqrt(var_without_i)
            mu_shift_i = (mu_full - mu_without_i) / sigma_full_safe
            sigma_shift_i = (sigma_full - sigma_without_i) / sigma_full_safe

            top_contribs = [
                DimensionContribution(
                    dim_index=int(d),
                    dim_kind=(
                        kinds_v[d]
                        if kinds_v is not None and d < len(kinds_v) else None
                    ),
                    dim_label=(
                        dim_labels_v[d]
                        if dim_labels_v is not None and d < len(dim_labels_v) else None
                    ),
                    mu_shift=float(mu_shift_i[d]),
                    sigma_shift=float(sigma_shift_i[d]),
                    contribution=float(contrib_i[d]),
                )
                for d in ranked_dim_idx
            ]

            cascading: int | None = None
            if verbose:
                cascading = _count_cascading_flips(
                    shape_E=shapes[i],
                    sum_s=sum_s,
                    sum_s_sq=sum_s_sq,
                    shapes=shapes,
                    is_anomaly_full=is_anom_full,
                    e_idx=i,
                    theta_norm=theta_norm,
                )

            entries.append(InfluenceEntry(
                entity_key=pk_col[i],
                mu_impact=float(mu_imp[i]),
                sigma_impact=float(sigma_imp[i]),
                total_impact=float(total_imp[i]),
                delta_norm=float(delta_norm[i]),
                classification=classes[i],
                top_dim_contributions=top_contribs,
                cascading_flip_count=cascading,
            ))

        return InfluenceReport(
            pattern_id=pattern_id,
            pattern_version=version,
            population_size=N,
            high_threshold_pct=high_threshold_pct,
            total_impact_threshold=impact_threshold,
            theta_norm=theta_norm,
            classify_filter=classify,
            cell_counts=cell_counts,
            entries=entries,
        )

    def find_group_influence(
        self,
        pattern_id: str,
        groups: list[list[str]],
    ):
        """Caller-supplied per-group leave-set-out impact +
        reinforcing/canceling factor."""
        from hypertopos.engine.geometry import (
            _compute_leave_one_out_impact,
            _compute_leave_set_out_impact,
        )
        from hypertopos.model.sphere import (
            DimensionContribution,
            GroupInfluenceReport,
        )
        from hypertopos.utils.arrow import delta_matrix_from_arrow

        if len(groups) == 0:
            raise ValueError(
                "find_group_influence: groups list must be non-empty"
            )

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type == "event":
            raise ValueError(
                f"find_group_influence: pattern_type 'event' has no "
                f"population statistics; M4 requires anchor pattern "
                f"(got pattern_id={pattern_id!r})"
            )

        version = self._manifest.pattern_versions[pattern_id]
        geo_table = self._storage.read_geometry(pattern_id, version)
        N = geo_table.num_rows
        if N < 3:
            raise ValueError(
                f"find_group_influence: pattern={pattern_id!r} has N={N}; "
                f"need at least 3 entities total to leave a non-trivial "
                f"population after removing a group of 2"
            )

        deltas = delta_matrix_from_arrow(geo_table).astype(np.float64)
        pk_col = geo_table["primary_key"].to_pylist()
        pk_to_idx = {pk: i for i, pk in enumerate(pk_col)}
        sigma_diag = np.asarray(pattern.sigma_diag, dtype=np.float64)
        mu = np.asarray(pattern.mu, dtype=np.float64)
        sigma_floor = np.maximum(sigma_diag, 1e-2)
        shapes = deltas * sigma_floor + mu

        mu_full = shapes.mean(axis=0)
        sigma_full = shapes.std(axis=0, ddof=0)

        _, _, total_imp_per_entity, _ = _compute_leave_one_out_impact(
            shapes, mu_full, sigma_full,
        )

        sigma_full_safe = np.maximum(sigma_full, 1e-12)
        kinds_v = pattern.dimension_kinds
        dim_labels_v = pattern.dim_labels

        reports: list[GroupInfluenceReport] = []
        for g_idx, group in enumerate(groups):
            k = len(group)
            if k < 2:
                raise ValueError(
                    f"find_group_influence: group at index {g_idx} has {k} "
                    f"member(s); single-entity groups duplicate "
                    f"find_calibration_influencers — use that primitive instead"
                )
            if len(set(group)) != k:
                raise ValueError(
                    f"find_group_influence: group at index {g_idx} contains "
                    f"duplicate entity (group={group})"
                )
            if k >= N:
                raise ValueError(
                    f"find_group_influence: group at index {g_idx} has {k} "
                    f"members but population N={N}; cannot leave empty population"
                )
            members_idx_list = []
            for key in group:
                if key not in pk_to_idx:
                    raise ValueError(
                        f"find_group_influence: entity {key!r} in group "
                        f"{g_idx} not found in pattern={pattern_id!r}"
                    )
                members_idx_list.append(pk_to_idx[key])
            members_idx = np.asarray(members_idx_list, dtype=np.int64)

            mu_imp_set, sigma_imp_set, total_imp_set, contributions = (
                _compute_leave_set_out_impact(
                    shapes=shapes,
                    members_idx=members_idx,
                    mu_full=mu_full,
                    sigma_full=sigma_full,
                )
            )

            sum_individual = float(total_imp_per_entity[members_idx].sum())
            if sum_individual <= 0.0:
                raise ValueError(
                    f"find_group_influence: group at index {g_idx} sum of "
                    f"individual impacts is 0.0; reinforcing_factor undefined. "
                    f"Group is at population centroid — use a non-degenerate group."
                )
            reinforcing = total_imp_set / sum_individual

            D = shapes.shape[1]
            ranked_dim_idx = np.argsort(-np.abs(contributions))[: min(5, D)]
            set_shapes = shapes[members_idx]
            set_sum = set_shapes.sum(axis=0)
            set_sum_sq = (set_shapes ** 2).sum(axis=0)
            mu_without_set = (shapes.sum(axis=0) - set_sum) / (N - k)
            var_without_set = (
                ((shapes ** 2).sum(axis=0) - set_sum_sq) / (N - k)
                - mu_without_set ** 2
            )
            var_without_set = np.maximum(var_without_set, 0.0)
            sigma_without_set = np.sqrt(var_without_set)
            mu_shift = (mu_full - mu_without_set) / sigma_full_safe
            sigma_shift = (sigma_full - sigma_without_set) / sigma_full_safe

            top_contribs = [
                DimensionContribution(
                    dim_index=int(d),
                    dim_kind=(
                        kinds_v[d]
                        if kinds_v is not None and d < len(kinds_v) else None
                    ),
                    dim_label=(
                        dim_labels_v[d]
                        if dim_labels_v is not None and d < len(dim_labels_v) else None
                    ),
                    mu_shift=float(mu_shift[d]),
                    sigma_shift=float(sigma_shift[d]),
                    contribution=float(contributions[d]),
                )
                for d in ranked_dim_idx
            ]

            reports.append(GroupInfluenceReport(
                pattern_id=pattern_id,
                pattern_version=version,
                group_index=g_idx,
                member_count=k,
                members=list(group),
                mu_impact_set=mu_imp_set,
                sigma_impact_set=sigma_imp_set,
                total_impact_set=total_imp_set,
                sum_individual_impacts=sum_individual,
                reinforcing_factor=reinforcing,
                top_dim_contributions=top_contribs,
            ))

        return reports

    def find_lead_lag(
        self,
        pattern_a: str,
        pattern_b: str,
        *,
        timestamp_from: str | None = None,
        timestamp_to: str | None = None,
        cohort: str = "fixed",
        min_epochs: int = 8,
        max_lag: int | None = None,
        fdr_alpha: float = 0.05,
        fdr_method: str = "storey",
        verbose: bool = False,
        entity_key: str | None = None,
    ):
        """M5: cross-pattern temporal lead-lag in population-relative coordinates.

        Both patterns must be ``pattern_type='anchor'``. Time alignment uses
        the intersection of pattern timestamp sets; raises if intersection
        size is below ``min_epochs``. Cohort defaults to ``'fixed'`` —
        entities present at every common epoch in both patterns — for
        clean panel-style centroid signals.

        Returns a :class:`LeadLagReport` with three nested answer levels:
          1. Population scalar lead-lag (``lag``, ``correlation``,
             ``is_significant`` — Bonferroni-adjusted peak threshold).
          2. Per-dim D_A x D_B matrix (``top_dim_pairs`` and full
             ``per_dim_pairs`` when ``verbose=True``).
          3. Per-entity drill-down via ``entity_key`` (population centroid
             replaced by that entity's delta trajectory).
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        from hypertopos.engine.geometry import _compute_lead_lag_report

        # ── 1. Validation ─────────────────────────────────────────────
        if cohort not in ("fixed", "all"):
            raise ValueError(
                f"find_lead_lag: cohort must be 'fixed' or 'all'; got {cohort!r}"
            )
        if fdr_method not in ("bh", "storey"):
            raise ValueError(
                f"find_lead_lag: fdr_method must be 'bh' or 'storey'; got {fdr_method!r}"
            )
        if pattern_a == pattern_b:
            raise ValueError(
                f"find_lead_lag: pattern_a and pattern_b must differ (both={pattern_a!r})"
            )
        if min_epochs < 4:
            raise ValueError(
                f"find_lead_lag: min_epochs must be >= 4; got {min_epochs}"
            )
        if not 0.0 < fdr_alpha < 1.0:
            raise ValueError(
                f"find_lead_lag: fdr_alpha must be in (0, 1); got {fdr_alpha}"
            )

        sphere = self._storage.read_sphere()
        if pattern_a not in sphere.patterns:
            raise KeyError(f"pattern not found: {pattern_a!r}")
        if pattern_b not in sphere.patterns:
            raise KeyError(f"pattern not found: {pattern_b!r}")
        pat_a = sphere.patterns[pattern_a]
        pat_b = sphere.patterns[pattern_b]
        if pat_a.pattern_type != "anchor":
            raise ValueError(
                f"find_lead_lag requires anchor patterns; got pattern_type="
                f"{pat_a.pattern_type!r} for {pattern_a!r}"
            )
        if pat_b.pattern_type != "anchor":
            raise ValueError(
                f"find_lead_lag requires anchor patterns; got pattern_type="
                f"{pat_b.pattern_type!r} for {pattern_b!r}"
            )

        # ── 2. Preflight cohort-size budget via geometry count ────────
        # Geometry has one row per entity for anchor patterns (Lance
        # count_rows is metadata-only, no scan). Lets us reject huge
        # disjoint-line cross-pattern queries WITHOUT reading temporal.
        D_a_pre = len(pat_a.mu)
        D_b_pre = len(pat_b.mu)
        _budget_bytes = 1_000_000_000  # 1 GB total across both tensors
        if entity_key is None:
            try:
                ver_a = self._manifest.pattern_versions[pattern_a]
                ver_b = self._manifest.pattern_versions[pattern_b]
                n_ent_a = self._storage.count_geometry_rows(pattern_a, ver_a)
                n_ent_b = self._storage.count_geometry_rows(pattern_b, ver_b)
                # cohort='all' upper bound = union; cohort='fixed' ≤ min.
                cohort_upper = (
                    n_ent_a + n_ent_b if cohort == "all"
                    else min(n_ent_a, n_ent_b)
                )
                # Use min_epochs as a conservative N (real N may be larger;
                # if it isn't, the call would fail elsewhere on min_epochs).
                bound_bytes = 4 * min_epochs * cohort_upper * (D_a_pre + D_b_pre)
                if bound_bytes > _budget_bytes:
                    raise ValueError(
                        f"find_lead_lag: cohort upper bound would build at "
                        f"least {bound_bytes / 1e9:.2f} GB of shape tensors "
                        f"(pattern_a entities={n_ent_a}, pattern_b entities="
                        f"{n_ent_b}, D_a={D_a_pre}, D_b={D_b_pre}, "
                        f"min_epochs={min_epochs}); budget is "
                        f"{_budget_bytes / 1e9:.1f} GB. Patterns are likely "
                        f"over disjoint entity_lines — cross-pattern lead-lag "
                        f"requires shared entity space. Try cohort='fixed' "
                        f"(panel-clean intersection), rebuild the sphere with "
                        f"both patterns over the same entity_line, or pass "
                        f"entity_key=<id> for per-entity drill-down."
                    )
            except (KeyError, OSError, FileNotFoundError):
                # Geometry not available — fall through to temporal scan.
                pass

        # ── 3. Phase 1 scan — meta only ([primary_key, timestamp]) ────
        # Skip the heavy shape_snapshot list column on this pass so we
        # can decide cohort + budget guard before reading gigabytes.
        keys_filter = [entity_key] if entity_key is not None else None

        def _read_meta_table(pid: str) -> pa.Table:
            batches = list(self._storage.read_temporal_batched(
                pid,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                keys=keys_filter,
                columns=["primary_key", "timestamp"],
            ))
            if not batches:
                return pa.table({})
            return pa.Table.from_batches(batches)

        ta_meta = _read_meta_table(pattern_a)
        tb_meta = _read_meta_table(pattern_b)
        if ta_meta.num_rows == 0:
            raise ValueError(
                f"find_lead_lag: no temporal slices for pattern={pattern_a!r}"
                + (f" entity={entity_key!r}" if entity_key else "")
            )
        if tb_meta.num_rows == 0:
            raise ValueError(
                f"find_lead_lag: no temporal slices for pattern={pattern_b!r}"
                + (f" entity={entity_key!r}" if entity_key else "")
            )

        # ── 3. Time alignment via timestamp intersection ──────────────
        ts_a_unique = pc.unique(ta_meta["timestamp"]).to_pylist()
        ts_b_unique = pc.unique(tb_meta["timestamp"]).to_pylist()
        T_a = sorted(set(ts_a_unique))
        T_b = sorted(set(ts_b_unique))
        T_common = sorted(set(T_a) & set(T_b))
        N = len(T_common)
        if N < min_epochs:
            raise ValueError(
                f"find_lead_lag: intersection of pattern timestamps has only "
                f"N={N} epochs; need >= {min_epochs}. If patterns use different "
                "event_line/window, lead-lag is not well-defined; rebuild with "
                f"shared temporal config. (T_a count={len(T_a)}, T_b count={len(T_b)})"
            )

        max_lag_resolved = max(1, (N - 1) // 4) if max_lag is None else int(max_lag)
        if (N - 1) - 2 * max_lag_resolved < 2:
            raise ValueError(
                f"find_lead_lag: trimmed window L={(N - 1) - 2 * max_lag_resolved} < 2; "
                f"max_lag={max_lag_resolved} too large for N={N}"
            )

        # Restrict meta tables to T_common
        T_common_arr = pa.array(T_common, type=ta_meta["timestamp"].type)
        ta_meta = ta_meta.filter(
            pc.is_in(ta_meta["timestamp"], value_set=T_common_arr),
        )
        tb_meta = tb_meta.filter(
            pc.is_in(tb_meta["timestamp"], value_set=T_common_arr),
        )

        # ── 4. Cohort selection (uses meta table only) ────────────────
        cohort_dropped: int | None = None
        if entity_key is not None:
            ts_a_for_e = sorted(set(ta_meta["timestamp"].to_pylist()))
            ts_b_for_e = sorted(set(tb_meta["timestamp"].to_pylist()))
            common_for_e = sorted(set(ts_a_for_e) & set(ts_b_for_e))
            if len(common_for_e) < min_epochs:
                raise ValueError(
                    f"find_lead_lag: entity={entity_key!r} present in only "
                    f"{len(common_for_e)} common epochs; need >= {min_epochs}"
                )
            T_common = common_for_e
            N = len(T_common)
            if (N - 1) - 2 * max_lag_resolved < 2:
                raise ValueError(
                    f"find_lead_lag: entity={entity_key!r} N={N} too small for "
                    f"max_lag={max_lag_resolved}"
                )
            T_common_arr = pa.array(T_common, type=ta_meta["timestamp"].type)
            cohort_E = [entity_key]
            cohort_dropped = None
        elif cohort == "fixed":
            # Vectorised cohort intersection via Arrow group-by COUNT.
            # An entity present at every t ∈ T_common in pattern A has exactly
            # |T_common| (pk, t) rows after the T_common filter (and at least
            # one if rows are duplicated, which builder writes never produce).
            n_common = len(T_common)
            counts_a = ta_meta.group_by("primary_key").aggregate(
                [("timestamp", "count")]
            )
            counts_b = tb_meta.group_by("primary_key").aggregate(
                [("timestamp", "count")]
            )
            full_in_a = set(
                counts_a.filter(
                    pc.equal(counts_a["timestamp_count"], n_common)
                )["primary_key"].to_pylist()
            )
            full_in_b = set(
                counts_b.filter(
                    pc.equal(counts_b["timestamp_count"], n_common)
                )["primary_key"].to_pylist()
            )
            cohort_E_set = full_in_a & full_in_b
            if not cohort_E_set:
                raise ValueError(
                    "find_lead_lag: cohort='fixed' produced empty cohort "
                    "(no entities present at every epoch in both patterns). "
                    "If both patterns track different entity populations "
                    "(e.g. accounts vs account_pairs), cross-pattern lead-lag "
                    "is not applicable — there are no shared entities. "
                    "Otherwise try cohort='all', entity_key=<id> for per-entity "
                    "drill-down, or shorten the time window."
                )
            cohort_E = sorted(cohort_E_set)
            cohort_dropped = (
                (counts_a.num_rows + counts_b.num_rows) // 2 - len(cohort_E)
            )
        else:
            cohort_E = sorted(
                set(ta_meta["primary_key"].to_pylist())
                | set(tb_meta["primary_key"].to_pylist())
            )
            cohort_dropped = None

        # ── 5. Memory budget guard before full read ───────────────────
        D_a = len(pat_a.mu)
        D_b = len(pat_b.mu)

        # Refuse to allocate giant tensors. Hits cohort='all' between
        # patterns over disjoint entity_lines (e.g. AML accounts vs
        # account_pairs vs tx_chains), where the union covers every
        # entity in the sphere and the (N, |cohort|, D) tensors blow
        # gigabytes before any compute starts.
        _budget_bytes = 1_000_000_000  # 1 GB total across both tensors
        _tensor_bytes = 4 * N * len(cohort_E) * (D_a + D_b)
        if _tensor_bytes > _budget_bytes:
            raise ValueError(
                f"find_lead_lag: cohort would build "
                f"{_tensor_bytes / 1e9:.2f} GB of shape tensors "
                f"(cohort={len(cohort_E)}, N={N}, D_a={D_a}, D_b={D_b}); "
                f"budget is {_budget_bytes / 1e9:.1f} GB. Patterns are "
                f"likely over disjoint entity_lines — cross-pattern "
                f"lead-lag requires shared entity space. Try cohort='fixed' "
                f"(panel-clean intersection) or rebuild the sphere with "
                f"both patterns over the same entity_line. For per-entity "
                f"drill-down, pass entity_key=<id>."
            )

        # ── 6. Phase 2 scan — full read, filter in memory ────────────
        # Cohort + budget already validated; key pushdown for thousands
        # of IN-list entries is slower in Lance than a single full scan
        # followed by an Arrow is_in mask. For entity_key mode (1 key)
        # we DO push down — that's the fast point lookup path.
        cohort_pa_arr = pa.array(cohort_E)
        push_keys = (
            [entity_key] if entity_key is not None
            else (cohort_E if len(cohort_E) <= 64 else None)
        )

        def _read_full_table(pid: str, keys: list[str] | None) -> pa.Table:
            batches = list(self._storage.read_temporal_batched(
                pid,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                keys=keys,
            ))
            if not batches:
                return pa.table({})
            return pa.Table.from_batches(batches)

        ta_filt = _read_full_table(pattern_a, push_keys)
        tb_filt = _read_full_table(pattern_b, push_keys)
        if push_keys is None:
            if ta_filt.num_rows > 0:
                ta_filt = ta_filt.filter(
                    pc.is_in(ta_filt["primary_key"], value_set=cohort_pa_arr),
                )
            if tb_filt.num_rows > 0:
                tb_filt = tb_filt.filter(
                    pc.is_in(tb_filt["primary_key"], value_set=cohort_pa_arr),
                )
        if ta_filt.num_rows > 0:
            ta_filt = ta_filt.filter(
                pc.is_in(ta_filt["timestamp"], value_set=T_common_arr),
            )
        if tb_filt.num_rows > 0:
            tb_filt = tb_filt.filter(
                pc.is_in(tb_filt["timestamp"], value_set=T_common_arr),
            )

        T_common_pa_arr = T_common_arr

        def _build_shape_tensor(table: pa.Table, D: int) -> np.ndarray:
            """Vectorised shape tensor build via Arrow index_in + list_flatten.

            Returns (N, |cohort|, D) float32 with NaN where (pk, t) cell has
            no row in the temporal table (entity missing at that epoch under
            cohort='all').
            """
            tens = np.full((N, len(cohort_E), D), np.nan, dtype=np.float32)
            if table.num_rows == 0:
                return tens
            # Vectorised index lookup — index_in returns null when not in value_set;
            # we filter those out with a single Arrow mask.
            pk_idx_pa = pc.index_in(table["primary_key"], value_set=cohort_pa_arr)
            ts_idx_pa = pc.index_in(table["timestamp"], value_set=T_common_pa_arr)
            keep_mask = pc.and_(pc.is_valid(pk_idx_pa), pc.is_valid(ts_idx_pa))
            if not pc.any(keep_mask).as_py():
                return tens
            kept_table = table.filter(keep_mask)
            kept_pk_idx = pc.index_in(
                kept_table["primary_key"], value_set=cohort_pa_arr,
            ).to_numpy(zero_copy_only=False).astype(np.int64)
            kept_ts_idx = pc.index_in(
                kept_table["timestamp"], value_set=T_common_pa_arr,
            ).to_numpy(zero_copy_only=False).astype(np.int64)
            # Flatten shape_snapshot list column → contiguous float32 (n_kept, D_actual).
            shape_chunked = kept_table["shape_snapshot"]
            shape_arr = (
                shape_chunked.combine_chunks()
                if hasattr(shape_chunked, "combine_chunks") else shape_chunked
            )
            flat_vals = shape_arr.values.to_numpy(zero_copy_only=False).astype(np.float32)
            n_kept = kept_table.num_rows
            if n_kept == 0:
                return tens
            D_actual = flat_vals.shape[0] // n_kept
            shape_matrix = flat_vals.reshape(n_kept, D_actual)
            d_use = min(D, D_actual)
            # Vectorised scatter into the (T, |cohort|, D) tensor
            tens[kept_ts_idx, kept_pk_idx, :d_use] = shape_matrix[:, :d_use]
            return tens

        shapes_a = _build_shape_tensor(ta_filt, D_a)
        shapes_b = _build_shape_tensor(tb_filt, D_b)

        # Replace NaN cells with the present-entity per-(t, dim) mean so that
        # the population mean at epoch t equals the mean of present entities
        # at that epoch (works for both 'all' and 'fixed' modes, with 'fixed'
        # being NaN-free by construction except for rare missing rows).
        def _impute_to_present_mean(tens: np.ndarray) -> np.ndarray:
            mask = ~np.isnan(tens)
            sums = np.where(mask, tens, np.float32(0.0)).sum(
                axis=1, dtype=np.float32,
            )
            counts = mask.sum(axis=1).astype(np.float32)
            counts_safe = np.where(counts > 0, counts, np.float32(1.0))
            mean_per_td = (sums / counts_safe).astype(np.float32, copy=False)
            return np.where(
                mask, tens, mean_per_td[:, None, :],
            ).astype(np.float32, copy=False)

        _has_missing = (
            cohort == "all"
            or np.isnan(shapes_a).any()
            or np.isnan(shapes_b).any()
        )
        if entity_key is None and _has_missing:
            shapes_a = _impute_to_present_mean(shapes_a)
            shapes_b = _impute_to_present_mean(shapes_b)
        elif entity_key is not None:
            # Per-entity mode: NaN rows mean entity is missing at that epoch.
            # Fill with the entity's nearest-neighbour value — for single-entity
            # cohort interpolating to "present-mean" reduces to copying.
            mask_a = ~np.isnan(shapes_a).any(axis=(1, 2))
            mask_b = ~np.isnan(shapes_b).any(axis=(1, 2))
            if not (mask_a.all() and mask_b.all()):
                # Drop missing epochs entirely — re-validate length
                keep = np.where(mask_a & mask_b)[0]
                if keep.shape[0] < N:
                    if (keep.shape[0] - 1) - 2 * max_lag_resolved < 2:
                        raise ValueError(
                            f"find_lead_lag: entity={entity_key!r} after dropping "
                            f"missing epochs has N={keep.shape[0]} too small for "
                            f"max_lag={max_lag_resolved}"
                        )
                    shapes_a = shapes_a[keep]
                    shapes_b = shapes_b[keep]
                    T_common = [T_common[i] for i in keep]
                    N = len(T_common)

        # ── 6. Resolve schema_hash for traceability (best-effort) ─────
        def _safe_schema_hash(pid: str) -> str:
            try:
                versions = self._storage.list_calibration_versions(pid)
                if versions:
                    fit = self._storage.read_calibration_fit(pid, version=versions[-1])
                    return fit.schema_hash
            except Exception:
                pass
            return ""

        schema_hash_a = _safe_schema_hash(pattern_a)
        schema_hash_b = _safe_schema_hash(pattern_b)

        # ── 7. Build LeadLagReport via engine orchestrator ────────────
        n_dropped_a = len(T_a) - N
        n_dropped_b = len(T_b) - N
        return _compute_lead_lag_report(
            pattern_a=pattern_a,
            pattern_b=pattern_b,
            entity_key=entity_key,
            shapes_a=shapes_a,
            shapes_b=shapes_b,
            mu_a=np.asarray(pat_a.mu, dtype=np.float32),
            sigma_a=np.asarray(pat_a.sigma_diag, dtype=np.float32),
            mu_b=np.asarray(pat_b.mu, dtype=np.float32),
            sigma_b=np.asarray(pat_b.sigma_diag, dtype=np.float32),
            dim_labels_a=pat_a.dim_labels,
            dim_labels_b=pat_b.dim_labels,
            timestamps=T_common,
            n_dropped_a=n_dropped_a,
            n_dropped_b=n_dropped_b,
            cohort_size=len(cohort_E),
            cohort_dropped=cohort_dropped,
            schema_hash_a=schema_hash_a,
            schema_hash_b=schema_hash_b,
            max_lag=max_lag_resolved,
            fdr_alpha=fdr_alpha,
            fdr_method=fdr_method,
            verbose=verbose,
        )

    def _attach_influence_fields_to_anomaly_entries(
        self,
        entries: list[dict],
        pattern_id: str,
    ) -> list[dict]:
        """M4 additive: attach total_impact + classification per π5 entry.

        Resolves to None per-entry when pattern is event-type, N<2, or storage
        backend lacks shape reconstruction prerequisites."""
        from hypertopos.engine.geometry import (
            _classify_influence,
            _compute_leave_one_out_impact,
        )
        from hypertopos.utils.arrow import delta_matrix_from_arrow

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type == "event":
            for e in entries:
                e["total_impact"] = None
                e["classification"] = None
            return entries

        version = self._manifest.pattern_versions[pattern_id]
        try:
            geo_table = self._storage.read_geometry(pattern_id, version)
        except (KeyError, OSError, ValueError):
            for e in entries:
                e["total_impact"] = None
                e["classification"] = None
            return entries

        N = geo_table.num_rows
        if N < 2:
            for e in entries:
                e["total_impact"] = None
                e["classification"] = None
            return entries

        deltas = delta_matrix_from_arrow(geo_table).astype(np.float64)
        pk_col = geo_table["primary_key"].to_pylist()
        pk_to_idx = {pk: i for i, pk in enumerate(pk_col)}
        sigma_diag = np.asarray(pattern.sigma_diag, dtype=np.float64)
        mu = np.asarray(pattern.mu, dtype=np.float64)
        sigma_floor = np.maximum(sigma_diag, 1e-2)
        shapes = deltas * sigma_floor + mu
        mu_full = shapes.mean(axis=0)
        sigma_full = shapes.std(axis=0, ddof=0)
        delta_norm = np.linalg.norm(deltas, axis=1)
        theta_norm = float(pattern.theta_norm)

        try:
            _, _, total_imp, _ = _compute_leave_one_out_impact(
                shapes, mu_full, sigma_full,
            )
            classes = _classify_influence(
                total_imp, delta_norm, theta_norm, high_threshold_pct=90.0,
            )
        except (ValueError, FloatingPointError):
            for e in entries:
                e["total_impact"] = None
                e["classification"] = None
            return entries

        for e in entries:
            ekey = e.get("primary_key")
            if ekey is None or ekey not in pk_to_idx:
                e["total_impact"] = None
                e["classification"] = None
                continue
            i = pk_to_idx[ekey]
            e["total_impact"] = float(round(total_imp[i], 4))
            e["classification"] = classes[i]
        return entries

    def find_density_gaps(
        self,
        pattern_id: str,
        *,
        top_n: int = 10,
        dim_pairs: list[tuple[str, str]] | None = None,
        bins: int = 10,
        alpha: float = 0.05,
        r_min: float = 0.1,
        r_max: float = 0.7,
        sample_size: int = 100_000,
    ) -> dict[str, Any]:
        """Joint density gap detection via PIT + independence null + BH chi^2.

        For each selected dim pair build a uniform-marginal 2D
        histogram (probability integral transform normalises every dim
        kind) and flag bins whose observed count is significantly below
        the uniform-independence expectation. Each flagged bin maps
        back to a named raw-feature range with a BH-corrected q-value.

        sample_size: max entities to read (random sample when sphere is
            larger). Defaults to 100,000. Pass None to read all.
        """
        from hypertopos.engine.density_gaps import (
            ECDFEntry,
            compute_density_gaps_for_pair,
            is_usable_for_gap,
            select_pairs_by_corr,
        )

        if not (0.0 < alpha < 1.0):
            raise GDSNavigationError(
                f"alpha must be in (0, 1); got {alpha}",
            )
        if not (4 <= bins <= 50):
            raise GDSNavigationError(
                f"bins must be in [4, 50]; got {bins}",
            )
        if not (0.0 <= r_min < r_max <= 1.0):
            raise GDSNavigationError(
                f"r_min/r_max must satisfy 0 <= r_min < r_max <= 1; "
                f"got r_min={r_min}, r_max={r_max}",
            )
        if top_n < 1:
            raise GDSNavigationError(
                f"top_n must be >= 1; got {top_n}",
            )

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"pattern not found: {pattern_id!r}",
            )
        pattern = sphere.patterns[pattern_id]
        version = self._manifest.pattern_versions.get(pattern_id, 1)
        geo = self._storage.read_geometry(
            pattern_id, version,
            columns=["delta"],
            sample_size=sample_size if sample_size is not None else None,
        )
        n = geo.num_rows
        if n < 100:
            raise GDSNavigationError(
                f"pattern too small for density gap detection "
                f"(need >= 100, got {n})",
            )

        flat = geo["delta"].combine_chunks().values.to_numpy(zero_copy_only=False)
        d = flat.shape[0] // n
        x_matrix = flat.reshape(n, d).astype(np.float32)
        dim_labels = list(
            pattern.dim_labels or [f"dim_{i}" for i in range(d)],
        )

        excluded: list[dict[str, str]] = []
        usable_idx: list[int] = []
        for i in range(d):
            ok, reason = is_usable_for_gap(x_matrix[:, i])
            if ok:
                usable_idx.append(i)
            else:
                excluded.append({"dim": dim_labels[i], "reason": reason})

        if len(usable_idx) < 2:
            return {
                "pattern_id": pattern_id,
                "n_entities": n,
                "gaps": [],
                "excluded_dims": excluded,
                "n_pairs_tested": 0,
                "reason": "fewer than 2 usable dims",
            }

        ecdf_cache: dict[int, ECDFEntry] = {
            i: ECDFEntry.from_values(x_matrix[:, i]) for i in usable_idx
        }
        u_matrix = np.column_stack(
            [ecdf_cache[i].transform(x_matrix[:, i]) for i in usable_idx],
        )

        if dim_pairs is not None:
            label_to_local = {
                dim_labels[i]: idx for idx, i in enumerate(usable_idx)
            }
            unknown = [
                name for pair in dim_pairs for name in pair
                if name not in label_to_local
            ]
            if unknown:
                raise GDSNavigationError(
                    f"unknown dim names in dim_pairs: "
                    f"{sorted(set(unknown))}",
                )
            pair_local: list[tuple[int, int, float]] = []
            for (a, b) in dim_pairs:
                la, lb = label_to_local[a], label_to_local[b]
                r = float(abs(np.corrcoef(
                    u_matrix[:, la], u_matrix[:, lb],
                )[0, 1]))
                pair_local.append((la, lb, r))
        else:
            corr = np.corrcoef(u_matrix.T)
            pair_local = select_pairs_by_corr(
                corr, r_min=r_min, r_max=r_max, top_k=20,
            )

        if not pair_local:
            return {
                "pattern_id": pattern_id,
                "n_entities": n,
                "gaps": [],
                "excluded_dims": excluded,
                "n_pairs_tested": 0,
                "reason": "no pairs in correlation window",
            }

        all_cells: list[dict[str, Any]] = []
        for (i_local, j_local, r) in pair_local:
            i_global = usable_idx[i_local]
            j_global = usable_idx[j_local]
            cells = compute_density_gaps_for_pair(
                u_matrix[:, i_local], u_matrix[:, j_local],
                n=n, bins=bins, alpha=alpha,
            )
            for c in cells:
                if not c["is_gap"]:
                    continue
                delta_lo_i, delta_hi_i = ecdf_cache[i_global].inverse(
                    np.array(c["u_range_i"]),
                )
                delta_lo_j, delta_hi_j = ecdf_cache[j_global].inverse(
                    np.array(c["u_range_j"]),
                )
                ratio = (
                    float(c["expected"] / max(c["observed"], 1e-9))
                    if c["observed"] > 0 else float("inf")
                )
                all_cells.append({
                    "dim_i": dim_labels[i_global],
                    "dim_j": dim_labels[j_global],
                    "u_range_i": list(c["u_range_i"]),
                    "u_range_j": list(c["u_range_j"]),
                    "delta_range_i": [float(delta_lo_i), float(delta_hi_i)],
                    "delta_range_j": [float(delta_lo_j), float(delta_hi_j)],
                    "observed": c["observed"],
                    "expected": c["expected"],
                    "ratio": ratio,
                    "p_value": c["p_value"],
                    "q_value": c["q_value"],
                    "is_gap": True,
                    "correlation": float(r),
                })

        all_cells.sort(key=lambda c: -c["ratio"])
        return {
            "pattern_id": pattern_id,
            "n_entities": n,
            "gaps": all_cells[:top_n],
            "excluded_dims": excluded,
            "n_pairs_tested": len(pair_local),
        }

    def _build_scoped_adjacency_for_motif(
        self,
        pattern_id: str,
        hops: list[Any],
        seed_keys: list[str],
    ) -> Any:
        """Build a partial AdjacencyIndex covering only the BFS frontier
        reachable from ``seed_keys`` within ``len(hops)`` levels.

        Uses Lance BTREE-pushdown reads (``from_keys=`` / ``to_keys=`` IN
        filters) to fetch only the edges actually visited.  On a 5 M-tx
        edge table seeded by one anchor entity, this typically reads tens
        to low hundreds of rows instead of the full table.
        """
        import pyarrow as pa
        from hypertopos.engine.adjacency import AdjacencyIndex

        needs_forward = any(
            getattr(h, "direction", "forward") in ("forward", "any") for h in hops
        )
        needs_reverse = any(
            getattr(h, "direction", "forward") in ("reverse", "any") for h in hops
        )
        if not (needs_forward or needs_reverse):
            needs_forward = True

        visited: set[str] = set(seed_keys)
        frontier: set[str] = set(seed_keys)
        edge_tables: list[pa.Table] = []

        for _ in range(len(hops)):
            if not frontier:
                break
            new_keys: set[str] = set()
            if needs_forward:
                t = self._storage.read_edges(
                    pattern_id, from_keys=list(frontier),
                )
                if t.num_rows > 0:
                    edge_tables.append(t)
                    new_keys.update(t["to_key"].to_pylist())
            if needs_reverse:
                t = self._storage.read_edges(
                    pattern_id, to_keys=list(frontier),
                )
                if t.num_rows > 0:
                    edge_tables.append(t)
                    new_keys.update(t["from_key"].to_pylist())
            next_frontier = new_keys - visited
            visited |= new_keys
            frontier = next_frontier

        if not edge_tables:
            return AdjacencyIndex._empty()

        combined = pa.concat_tables(edge_tables)
        # Forward + reverse reads of overlapping subgraphs surface the same
        # transaction twice; dedup on event_key before handing to from_edge_lists.
        if combined.num_rows > 0 and "event_key" in combined.column_names:
            seen: set[str] = set()
            keep: list[int] = []
            eks = combined["event_key"].to_pylist()
            for i, ek in enumerate(eks):
                if ek not in seen:
                    seen.add(ek)
                    keep.append(i)
            if len(keep) < combined.num_rows:
                combined = combined.take(keep)

        return AdjacencyIndex.from_edge_lists(
            combined["from_key"].to_pylist(),
            combined["to_key"].to_pylist(),
            combined["timestamp"].to_pylist(),
            combined["amount"].to_pylist(),
            combined["event_key"].to_pylist(),
        )

    def find_motif_by_hops(
        self,
        pattern_id: str,
        hops: list[Any],
        *,
        seed_keys: list[str] | None = None,
        max_results: int = 100,
        score: bool = False,
        time_window_hours: float | None = None,
    ) -> dict[str, Any]:
        """Match motifs declaratively via per-hop ``HopPredicate``s.

        Power-user escape hatch from the closed-vocab ``find_motif`` registry.
        Caller passes a list of ``HopPredicate``s describing per-hop
        amount / temporal / direction / edge-dim constraints; the navigator
        walks the edge table looking for matching chains.

        Supports ``amount_min`` / ``amount_max`` /
        ``time_delta_max_hours`` / ``direction`` (``forward`` / ``reverse``
        / ``any``) / ``amount_ratio_to_prev`` (decreasing-chain ratio in
        ``(0, 1.0]``; rejects edge unless
        ``current_amount / prev_hop_amount <= ratio``; must be ``None`` on
        ``hops[0]``) / ``edge_dim_predicates``.

        ``time_window_hours`` (optional, default ``None``): global
        total-chain-span cap, measured from the first hop's edge
        timestamp. When set, every hop after the first must satisfy
        ``abs(current_edge_ts - first_edge_ts) <= time_window_hours``.
        Independent of per-hop ``time_delta_max_hours``; both apply when
        both are set. Must be strictly positive when not ``None``.

        ``score`` (optional, default ``False``): when ``True``, score each
        motif as the product of event-aware ``edge_potential`` across its
        edges. The scoring kernel resolves the anchor-companion pattern
        whose ``entity_line`` covers the edge endpoints
        (``_resolve_anchor_pattern_for_scoring``) and combines its
        per-entity geometry with the event pattern's per-event polygon
        norms — formula
        ``delta_distance × (1/effective_pair_count) × (1 + event_norm)``
        per edge. The event-norm factor breaks ties between motifs that
        share a node sequence but use different transactions (without it,
        all such motifs would collapse to identical scores). Each scored
        motif gains ``score``, ``score_breakdown`` (per-edge entries
        carry ``edge_potential``, ``delta_distance``, ``pair_tx_count``,
        ``effective_weight``, ``event_factor``), and
        ``anchor_pattern_id`` fields together; output is sorted descending
        on score with unscored motifs (endpoint missing from anchor
        geometry) at the tail. Raises ``GDSNavigationError`` when no
        anchor companion is configured for ``pattern_id``.

        ``require_anomalous_entity`` (per-hop bool, default ``False``):
        when ``True`` on hop ``i``, the destination entity (``nodes[i+1]``
        of the resulting motif) must satisfy ``is_anomaly=True`` in the
        resolved anchor companion pattern's geometry. Multiple hops may
        set this independently; constraints AND across hops. Filter runs
        after BFS, before scoring — saves scoring work on motifs that
        get dropped. ``max_results`` applies AFTER the filter, so a
        restrictive filter can return fewer than ``max_results`` motifs.
        Seed (``nodes[0]``) is never checked — pre-filter ``seed_keys``
        upfront to cover it. Raises ``GDSNavigationError`` when no
        anchor companion is configured.
        """
        from hypertopos.engine.hop_predicate import (
            enumerate_motifs_by_hops,
            validate_hops,
        )

        if max_results < 1:
            raise GDSNavigationError(
                f"max_results must be >= 1; got {max_results}",
            )
        try:
            validate_hops(hops, time_window_hours=time_window_hours)
        except ValueError as exc:
            raise GDSNavigationError(str(exc)) from exc

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"pattern not found: {pattern_id!r}",
            )
        pattern = sphere.patterns[pattern_id]
        if pattern.pattern_type != "event":
            raise GDSNavigationError(
                f"find_motif_by_hops requires an event pattern; "
                f"got pattern_type={pattern.pattern_type!r} for "
                f"{pattern_id!r}",
            )

        # When seed_keys are provided, build a scoped AdjacencyIndex via
        # Lance BTREE-pushdown reads expanding the BFS frontier hop-by-hop.
        # On large edge tables (multi-million rows) this avoids the cold-cache
        # full-table to_pylist() rebuild that AdjacencyIndex.from_lance pays
        # on the first global call — for a 2-hop seeded query on AML HI-small
        # (5 M tx, ~22 outgoing edges per seed) the scoped build reads ~70
        # edges vs. 5 M.  The unseeded path keeps the cached global index.
        if seed_keys:
            adj = self._build_scoped_adjacency_for_motif(
                pattern_id, hops, seed_keys,
            )
        else:
            # AdjacencyIndex is built once per pattern at storage level
            # (cached inside GDSReader._adjacency_cache).  Subsequent calls
            # reuse it; cold first call on a multi-million-edge table can
            # take tens of seconds — that is the price of full enumeration.
            adj = self._storage.get_adjacency(pattern_id)
        if adj is None or not adj._out:
            return {
                "pattern_id": pattern_id,
                "n_results": 0,
                "motifs": [],
                "reason": "pattern has no edge table",
            }

        # edge_features sidecar — only loaded when at least one hop
        # references edge_dim_predicates.
        needs_features = any(
            bool(getattr(h, "edge_dim_predicates", {})) for h in hops
        )
        edge_features = (
            self._storage.read_edge_features(pattern_id)
            if needs_features else None
        )
        if needs_features and (edge_features is None or edge_features.num_rows == 0):
            raise GDSNavigationError(
                f"edge_dim_predicates require an edge_features sidecar; "
                f"pattern {pattern_id!r} has none — declare "
                f"edge_dimensions: in YAML and rebuild the sphere",
            )

        try:
            instances = enumerate_motifs_by_hops(
                adj._out,
                adj._in,
                hops=hops,
                seed_keys=seed_keys,
                max_results=max_results,
                edge_features=edge_features,
                time_window_hours=time_window_hours,
            )
        except ValueError as exc:
            raise GDSNavigationError(str(exc)) from exc

        # F4 require_anomalous_entity — drop motifs where any flagged
        # hop's destination is not anomalous in the anchor companion.
        # Filter runs after BFS, before scoring — saves scoring work on
        # motifs that will be dropped, and `n_results` reflects the
        # post-filter count. `max_results` applies AFTER this filter; if
        # the filter is restrictive the call returns fewer than
        # max_results motifs.
        require_flags = [
            bool(getattr(h, "require_anomalous_entity", False))
            for h in hops
        ]
        if instances and any(require_flags):
            anchor_pid_for_filter = (
                self._resolve_anchor_pattern_for_scoring(pattern_id)
            )
            if (
                anchor_pid_for_filter is None
                or anchor_pid_for_filter == pattern_id
            ):
                raise GDSNavigationError(
                    f"require_anomalous_entity requires an anchor "
                    f"pattern whose entity_line covers edge endpoints "
                    f"of {pattern_id!r}; none found.",
                )
            flagged_hop_indices = [
                i for i, f in enumerate(require_flags) if f
            ]
            candidate_keys: set[str] = set()
            for inst in instances:
                for i in flagged_hop_indices:
                    candidate_keys.add(inst["nodes"][i + 1])
            anchor_version = self._resolve_version(anchor_pid_for_filter)
            geo = self._storage.read_geometry(
                anchor_pid_for_filter, anchor_version,
                point_keys=list(candidate_keys),
                columns=["primary_key", "is_anomaly"],
            )
            if "is_anomaly" not in geo.column_names:
                raise GDSNavigationError(
                    f"anchor pattern {anchor_pid_for_filter!r} has no "
                    f"is_anomaly column — calibration must run first",
                )
            is_anomaly_map: dict[str, bool] = {
                geo["primary_key"][i].as_py(): bool(
                    geo["is_anomaly"][i].as_py(),
                )
                for i in range(geo.num_rows)
            }
            filtered: list[dict[str, Any]] = []
            for inst in instances:
                if all(
                    is_anomaly_map.get(inst["nodes"][i + 1], False)
                    for i in flagged_hop_indices
                ):
                    filtered.append(inst)
            instances = filtered

        # Scoring uses _score_motif_from_edges which requires anchor-pattern
        # geometry (entity-level polygons). Event patterns store
        # per-transaction polygons keyed by event_key, so motif edges
        # (account, account) must use the anchor pattern's per-entity
        # geometry instead. _resolve_anchor_pattern_for_scoring picks the
        # anchor whose entity_line covers the edge endpoints.
        if score and instances:
            anchor_pid = self._resolve_anchor_pattern_for_scoring(pattern_id)
            if anchor_pid is None or anchor_pid == pattern_id:
                raise GDSNavigationError(
                    f"score=True requires an anchor pattern whose "
                    f"entity_line covers edge endpoints of {pattern_id!r}; "
                    f"none found. Declare an anchor pattern in the "
                    f"sphere config.",
                )
            # Hoist BOTH delta reads out of the per-motif loop:
            #   * anchor pattern (per-account geometry, ~500k rows)
            #   * event pattern (per-transaction polygons, ~5M rows on AML)
            # Union all motifs' endpoints / event_keys once and pay a
            # single Lance scan per pattern. Without this, each motif
            # pays its own pair of reads, multiplying I/O linearly with
            # max_results — measured 3.5× warm regression vs baseline
            # before this hoist; with both reads hoisted F5's
            # event-aware scoring fits within the perf gate.
            all_anchor_keys: set[str] = set()
            all_event_keys: set[str] = set()
            for inst in instances:
                for node in inst["nodes"]:
                    all_anchor_keys.add(node)
                all_event_keys.update(inst["edges"])
            anchor_version = self._resolve_version(anchor_pid)
            anchor_delta_map = self._batch_read_deltas(
                anchor_pid, anchor_version, all_anchor_keys,
            )
            event_version = self._resolve_version(pattern_id)
            event_delta_map = self._batch_read_deltas(
                pattern_id, event_version, all_event_keys,
            )
            graph_pid = self._resolve_motif_graph_pid(anchor_pid)
            anchor_adj = self._storage.get_adjacency(graph_pid)
            pair_counts = anchor_adj.pair_counts()

            scored: list[dict[str, Any]] = []
            for inst in instances:
                edge_pairs = [
                    (inst["nodes"][i], inst["nodes"][i + 1])
                    for i in range(len(inst["edges"]))
                ]
                event_factors_for_motif: list[float] = []
                for ek in inst["edges"]:
                    ed = event_delta_map.get(ek)
                    event_factors_for_motif.append(
                        1.0 if ed is None
                        else 1.0 + float(np.linalg.norm(ed)),
                    )
                score_result = self._lean_score_motif(
                    edge_pairs, anchor_delta_map, pair_counts,
                    event_factors=event_factors_for_motif,
                )
                if score_result is None:
                    # Endpoint missing in anchor geometry — silent skip.
                    scored.append(dict(inst))
                    continue
                inst_with_score = dict(inst)
                inst_with_score["score"] = score_result.get("score", 0.0)
                inst_with_score["score_breakdown"] = score_result.get(
                    "breakdown", [],
                )
                inst_with_score["anchor_pattern_id"] = anchor_pid
                scored.append(inst_with_score)
            scored.sort(key=lambda m: -m.get("score", 0.0))
            instances = scored

        return {
            "pattern_id": pattern_id,
            "n_results": len(instances),
            "motifs": instances,
        }

