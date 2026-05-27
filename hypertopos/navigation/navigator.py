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

# Lower clamp for per-detector p-values entering the harmonic-mean
# combiner. A single saturated detector (e.g. delta_norm clamped to 1e-300
# when anomaly_confidence ~= 1.0, or Fisher exact under-flowing on a
# hub bank with 500 accounts at 25 % anomaly vs 5 % population) would
# otherwise dominate ``sum(w_i / p_i)`` so completely that orthogonal
# detector signal is erased and entities collapse onto a single HMP
# value (~ 3e-300 for L=3 detectors). 1e-12 sits well below typical
# operating significance levels yet stays inside float64's reciprocal
# round-trip range so the harmonic mean still discriminates between
# entities whose dominant detector saturated.
_HMP_INPUT_P_FLOOR = 1e-12

logger = logging.getLogger(__name__)

_NAVIGATION_RECOVERABLE_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    AttributeError,
    pa.ArrowInvalid,
    pa.ArrowTypeError,
)


def _auroc_rank_sum(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via the Mann-Whitney U / rank-sum identity.

    Equivalent to ``sklearn.metrics.roc_auc_score(labels, scores)`` for
    binary labels. Vendored to avoid adding sklearn to the main runtime
    deps — it currently lives in the ``[topology]`` extra. Uses average
    ranks to handle ties correctly.

    AUROC = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg),
    where R_pos is the sum of ranks of positive-class scores after a
    stable sort assigning ties the average of the tied positions.

    Caller must ensure ``n_pos > 0`` AND ``n_neg > 0``.
    """
    n = scores.shape[0]
    if n == 0:
        return float("nan")
    # Average ranks for tie handling.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Replace tied groups with their mean rank.
    sorted_scores = scores[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            mean_rank = sorted_ranks[i:j].mean()
            sorted_ranks[i:j] = mean_rank
        i = j
    ranks[order] = sorted_ranks
    n_pos = int(labels.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

if TYPE_CHECKING:
    from hypertopos.engine.adjacency import AdjacencyIndex
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Contract, Manifest
    from hypertopos.storage.reader import GDSReader


# ---------------------------------------------------------------------------
# GDSError hierarchy
# ---------------------------------------------------------------------------

from hypertopos.storage.exceptions import (  # noqa: E402
    GDSCorruptedFileError as GDSCorruptedFileError,
    GDSError as GDSError,
    GDSMissingFileError as GDSMissingFileError,
    GDSStorageError as GDSStorageError,
    GDSVersionError as GDSVersionError,
)


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


# ---------------------------------------------------------------------------
# Similarity result container (backward-compatible list subclass)
# ---------------------------------------------------------------------------

class SimilarityResult(list):
    """List of (primary_key, distance) tuples with optional sidecar attributes.

    Extends ``list`` so all existing callers that iterate, slice, or convert to
    ``dict()`` continue to work unchanged. Sidecar attributes:

    - ``degenerate_warning`` — descriptive string when >50% of neighbors have
      distance = 0, else ``None``.
    - ``is_anomaly_map`` — ``{primary_key: is_anomaly}`` for the neighbours,
      populated when the caller asked the engine for stored anomaly flags;
      ``None`` when the lookup was skipped.
    """

    degenerate_warning: str | None
    is_anomaly_map: dict[str, bool] | None

    def __init__(
        self,
        items: list[tuple[str, float]],
        *,
        degenerate_warning: str | None = None,
        is_anomaly_map: dict[str, bool] | None = None,
    ):
        super().__init__(items)
        self.degenerate_warning = degenerate_warning
        self.is_anomaly_map = is_anomaly_map


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

# Population-tail share threshold above which a single dim is judged to
# dominate the geometric anomaly score. Aligned with the per-polygon
# ``compute_reliability_flags`` ``single_dim_driven`` default so the
# pattern-level audit and the per-polygon flag share the same notion of
# "one dim drives the score".
_DOMINANT_DIM_MASS_SHARE_THRESHOLD = 0.7

# Brown-Forsythe (median-centred Levene) p-value below which delta_norm
# variance is taken to differ meaningfully across the levels of a
# pattern's ``group_by_property`` — i.e. the global θ assumption is
# statistically violated and the agent should treat a single global
# threshold with caution. 0.01 keeps the false-positive rate low on
# multi-group spheres (typical agent surfaces 4-12 patterns ⇒ a 0.05
# threshold would fire spuriously on healthy spheres).
_HETEROSCEDASTICITY_P_THRESHOLD = 0.01

# Per-dim normality alpha — Shapiro-Wilk / KS p-value below this fires
# the ``non_normal_dim`` warning on gaussian dims. 0.01 is intentionally
# stricter than the classical 0.05 because pattern dims are tested in
# batch (one test per dim) and the calibration scan happens on the full
# population (large N) — both push raw p-values down, so a tighter alpha
# keeps the warning surface focused on dims with material distributional
# departure rather than near-normal dims with statistically significant
# but practically negligible skew.
_NON_NORMAL_DIM_PVALUE_THRESHOLD = 0.01

# ``kind_mismatch`` gates. The warning fires on a gaussian-declared dim
# when the Fisher LDA direction component is near zero (the dim does
# not carry the label-discriminating signal in the global axis) AND the
# raw class moments still show separation. The two-sided gate guards
# against (a) noise dims with near-zero direction and near-zero
# cohens_d (Dim C in the engineered fixture — both terms zero, nothing
# to surface) and (b) genuine high-direction dims that happen to have
# moderate cohens_d (Dim B — already captured by the Fisher axis, no
# mismatch). The 0.05 direction threshold mirrors the
# ``investigate_drift`` boundary used by the MCP ``audit_pattern_dims``
# tool so the two surfaces tell the same story.
_KIND_MISMATCH_DIRECTION_THRESHOLD = 0.05
_KIND_MISMATCH_COHENS_D_THRESHOLD = 0.3

# ``signed_tail_concentration`` gates. Fires when the persisted
# ``Pattern.signed_percentiles`` shows a one-sided extreme tail —
# ``|p99| / max(|p50|, 1e-9) > 50`` — on the Fisher LDA-projected delta
# distribution, signalling that the label-discriminating direction is
# driven by a tiny outlier subgroup rather than the global label split.
# Suppressed when the positive class is undersampled
# (``label_aware_n_pos < _SIGNED_TAIL_MIN_N_POS``) — with few labelled
# positives the LDA fit is itself unstable and the warning would mostly
# echo small-sample noise.
_SIGNED_TAIL_RATIO_THRESHOLD = 50.0
_SIGNED_TAIL_MIN_N_POS = 30


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


def _empty_coherent_diagnostics(
    *,
    n_chains_total: int,
    elapsed_ms: float,
) -> dict[str, Any]:
    """Diagnostics block returned by ``find_chains_with_coherent_anomaly``
    on the empty-result early-exit paths. Mirrors the populated shape so
    that downstream summarisers can read the same keys regardless of
    whether the population had any anomalous entities."""
    return {
        "n_chains_total": n_chains_total,
        "n_anomaly_entities": 0,
        "n_runs_total_pre_truncation": 0,
        "top_dim_counts_full": {},
        "run_length_distribution_full": {
            "min": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0, "mean": 0.0,
        },
        "all_coherent_chain_ids": set(),
        "elapsed_ms": elapsed_ms,
    }


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
        self._anomaly_map_cache: dict[tuple[str, int], dict[str, bool]] = {}

    def _get_anomaly_map(self, pattern_id: str, version: int) -> dict[str, bool]:
        """Lazy session-scoped {primary_key: is_anomaly} cache per (pattern, version).

        First lookup runs one full geometry scan over (primary_key, is_anomaly);
        subsequent lookups are O(1) dict reads. Memory cost is bounded by the
        pattern's row count — typically <10 MB for 1 M entities.
        """
        key = (pattern_id, version)
        cached = self._anomaly_map_cache.get(key)
        if cached is not None:
            return cached
        table = self._storage.read_geometry(
            pattern_id, version, columns=["primary_key", "is_anomaly"],
        )
        pk_list = table["primary_key"].to_pylist()
        anom_list = table["is_anomaly"].to_pylist()
        full_map = {
            pk: bool(flag) for pk, flag in zip(pk_list, anom_list, strict=False)
            if pk is not None and flag is not None
        }
        self._anomaly_map_cache[key] = full_map
        return full_map

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
        *,
        counterfactual_frozen_population: bool = False,
    ) -> GDSNavigator:
        """Dive into an entity's temporal Solid.

        When ``counterfactual_frozen_population=True``, each returned
        ``SolidSlice`` carries an additional ``delta_norm_frozen_pop`` field —
        the per-slice L2 norm recomputed against the FIRST slice's raw shape
        as the entity-relative reference epoch (sigma stays at the current
        pattern's diagonal). Answers "is this entity's apparent normalisation
        a real change, or just population drift around a stationary entity?"
        — a stationary entity yields ``delta_norm_frozen_pop = 0`` across all
        slices while ``delta_norm_snapshot`` reflects the drifting population.
        Default ``False`` keeps ``delta_norm_frozen_pop = None`` and preserves
        the existing return shape.
        """
        solid = self._engine.build_solid(
            primary_key,
            pattern_id,
            self._manifest,
            timestamp=timestamp,
            counterfactual_frozen_population=counterfactual_frozen_population,
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
        fdr_axis: str = "entity",
    ) -> list[Polygon]:
        """Apply FDR filtering and diverse selection to a list of Polygons.

        Mutates Polygon objects in-place (sets q_value / representativeness)
        and returns the filtered/reordered list.

        p_value_method: "rank" (default, uniform-by-construction) or "chi2"
          (upper-tail χ²(df) on ||delta||²). Rank-based p-values carry no null
          vs alternative signal for the Storey estimator; "chi2" is the option
          that makes fdr_method="storey" actually shrink q-values. Requires
          pattern_df to be supplied by the caller.

        fdr_axis: "entity" (default, current behaviour — flat FDR over per-
          entity anomaly p-values), "per_dim" (independent BH/Storey per
          dim using chi²(1) univariate per-cell p-values; keep entity iff
          any dim's q ≤ alpha), "both" (entity-level survival AND ≥1 dim
          surviving per-dim FDR). Per-dim mode reduces inflation when one
          dim drives many anomalies.
        """
        if p_value_method not in ("rank", "chi2"):
            raise ValueError(
                f"p_value_method must be 'rank' or 'chi2', got {p_value_method!r}"
            )
        if p_value_method == "chi2" and pattern_df is None:
            raise ValueError("p_value_method='chi2' requires pattern_df")
        if fdr_axis not in ("entity", "per_dim", "both"):
            raise ValueError(
                f"fdr_axis must be 'entity', 'per_dim', or 'both', got "
                f"{fdr_axis!r}",
            )

        # --- FDR filtering (opt-in) ---
        if fdr_alpha is not None and len(polygons) > 0:
            from hypertopos.engine.fdr import (
                benjamini_hochberg,
                empirical_p_values_from_rank,
                fdr_per_dimension,
                parametric_p_values_chi2,
                per_dim_p_values_chi2_univariate,
            )

            # Entity-axis FDR (always computed when fdr_alpha is set so that
            # poly.q_value reflects the entity-level test, even in per_dim
            # mode where the keep-decision is dim-driven).
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
            rejected_entity, q_values = benjamini_hochberg(
                p_values, fdr_alpha, method=fdr_method,
            )
            for poly, q in zip(polygons, q_values, strict=True):
                poly.q_value = float(q)  # type: ignore[attr-defined]

            # Per-dim FDR — opt-in via fdr_axis.
            if fdr_axis in ("per_dim", "both"):
                delta_matrix = np.array(
                    [p.delta for p in polygons], dtype=np.float64,
                )
                p_per_dim = per_dim_p_values_chi2_univariate(delta_matrix)
                rejected_per_dim, q_per_dim = fdr_per_dimension(
                    p_per_dim, alpha=fdr_alpha, method=fdr_method,
                )
                # Per-entity decision: ≥ 1 dim survives.
                rejected_any_dim = rejected_per_dim.any(axis=1)
                # Attach per-dim q-values + most-significant-dim summary.
                for i, poly in enumerate(polygons):
                    poly.q_values_per_dim = q_per_dim[i, :].tolist()  # type: ignore[attr-defined]
                    poly.min_q_per_dim = float(q_per_dim[i, :].min())  # type: ignore[attr-defined]
                    poly.dominant_q_dim_idx = int(  # type: ignore[attr-defined]
                        q_per_dim[i, :].argmin(),
                    )
                if fdr_axis == "per_dim":
                    rejected_final = rejected_any_dim
                else:  # both
                    rejected_final = rejected_entity & rejected_any_dim
            else:
                rejected_final = rejected_entity
            polygons = [
                p for p, keep in zip(polygons, rejected_final, strict=True)
                if keep
            ]

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

    @staticmethod
    def _stratified_sample_light(
        light: pa.Table,
        *,
        sample_size: int | None,
        boundary_aware: bool,
        theta_norm: float,
    ) -> pa.Table:
        """Sample ``light`` down to ``sample_size`` rows.

        When ``boundary_aware`` is True, half the budget goes to entities with
        ``delta_norm`` in ``[0.8 * theta_norm, 1.2 * theta_norm]`` (or all of
        them if fewer); the remainder is drawn uniformly from outside the
        band. Otherwise: uniform random sample without replacement.
        """
        n = light.num_rows
        if sample_size is None or n <= sample_size:
            return light
        rng = np.random.default_rng(0)
        if not boundary_aware:
            idx = rng.choice(n, size=sample_size, replace=False)
            idx.sort()
            return light.take(pa.array(idx))
        norms = light["delta_norm"].to_numpy(zero_copy_only=False).astype(np.float64)
        lo = 0.8 * theta_norm
        hi = 1.2 * theta_norm
        boundary_mask = (norms >= lo) & (norms <= hi)
        boundary_idx = np.where(boundary_mask)[0]
        non_boundary_idx = np.where(~boundary_mask)[0]
        budget_boundary = sample_size // 2
        budget_other = sample_size - budget_boundary
        if boundary_idx.size <= budget_boundary:
            pick_boundary = boundary_idx
            # Spill leftover to the other stratum so total budget is preserved.
            budget_other = sample_size - int(pick_boundary.size)
        else:
            pick_boundary = rng.choice(boundary_idx, size=budget_boundary, replace=False)
        if non_boundary_idx.size <= budget_other:
            pick_other = non_boundary_idx
        else:
            pick_other = rng.choice(non_boundary_idx, size=budget_other, replace=False)
        chosen = np.concatenate([pick_boundary, pick_other])
        chosen.sort()
        return light.take(pa.array(chosen))

    @staticmethod
    def _attach_reliability_flags(
        polygons: list[Polygon],
        *,
        pattern: Any,
    ) -> None:
        """Attach ``reliability_flags`` runtime attr on each polygon.

        Uses the same per-dim contribution machinery as
        ``explain_anomaly.top_dimensions`` so the dominant-dim attribution
        is consistent across surfaces. Sanitises ``anomaly_confidence``
        (``NaN`` / ``±inf`` → ``None``) per the strict-JSON convention.
        """
        from hypertopos.engine.geometry import compute_reliability_flags
        for poly in polygons:
            poly.reliability_flags = compute_reliability_flags(  # type: ignore[attr-defined]
                poly.delta,
                pattern=pattern,
                anomaly_confidence=poly.anomaly_confidence,
            )

    @staticmethod
    def _attach_signed_confidence_fields(
        polygons: list[Polygon],
        *,
        pattern: Any,
    ) -> None:
        """Compute and attach the signed-confidence triad per polygon.

        Composes three already-shipped signals into one confidence-weighted
        score:

            score = delta_norm_signed × |lda_alignment| × (1 − penalty)

        where ``delta_norm_signed = delta · direction`` (sign-preserving
        projection onto the Fisher LDA axis), ``lda_alignment = (delta /
        ||delta||) · direction`` (cosine on the LDA axis, in ``[-1, 1]``),
        and ``penalty = 0.5 · single_dim_driven + 0.5 ·
        low_confidence_bucket``. Pre-condition: caller verified that
        ``pattern.label_aware_calibration`` is populated and
        ``reliability_flags`` is already attached on each polygon.
        Zero-norm polygons receive ``lda_alignment = 0`` to avoid
        div-by-zero; non-finite scores are sanitised to ``0.0`` so
        downstream JSON serialisation stays clean.
        """
        cal = pattern.label_aware_calibration
        direction_vec = np.array(
            [float(cal[label].direction) for label in pattern.dim_labels],
            dtype=np.float64,
        )
        for poly in polygons:
            delta = np.asarray(poly.delta, dtype=np.float64)
            norm = float(np.linalg.norm(delta))
            signed = float(delta @ direction_vec)
            alignment = float(signed / norm) if norm > 0.0 else 0.0
            # Clip into [-1, 1] to absorb float noise; the direction is
            # unit-norm by construction but |cosine| occasionally slips
            # past 1.0 by ~1e-7.
            if alignment > 1.0:
                alignment = 1.0
            elif alignment < -1.0:
                alignment = -1.0
            r_flags = getattr(poly, "reliability_flags", None) or {}
            sdd = 1.0 if r_flags.get("single_dim_driven") else 0.0
            lcb = 1.0 if r_flags.get("low_confidence_bucket") else 0.0
            penalty = 0.5 * sdd + 0.5 * lcb
            score = signed * abs(alignment) * (1.0 - penalty)
            if not np.isfinite(score):
                score = 0.0
            poly.signed_confidence_score = score  # type: ignore[attr-defined]
            poly.lda_alignment = alignment  # type: ignore[attr-defined]
            poly.reliability_penalty = penalty  # type: ignore[attr-defined]

    def _apply_fdr_multi_resolution_gate(
        self,
        polygons: list[Polygon],
        *,
        pattern: Any,
        pattern_id: str,
        version: int,
        fdr_resolution: str | None,
        fdr_temporal_resolution: str | None,
        fdr_method: str,
        fdr_alpha: float,
    ) -> list[Polygon]:
        """Per-level BH/Storey FDR gating on a spatial / temporal hierarchy.

        Reads a minimal anomaly-indicator + hierarchy slice of geometry,
        computes per-cell hypergeometric p-values, applies per-level FDR via
        ``fdr_multi_resolution``, then filters ``polygons`` to those whose
        cell-tuple cleared every named level. Survivors are annotated with
        ``cell_q_spatial`` / ``cell_q_temporal`` / ``cell_path``.

        Pre-condition: caller validated that ``fdr_resolution`` /
        ``fdr_temporal_resolution`` exist on the pattern's hierarchies and
        that ``fdr_alpha`` is set.
        """
        from hypertopos.engine.fdr import (
            cell_p_values_from_anomaly_indicator,
            fdr_multi_resolution,
        )

        # Truncate spatial walk at user-named level
        spatial_levels_for_engine: list[str] = []
        hierarchy_dims: list[str] = []
        if fdr_resolution is not None:
            for lvl in pattern.fdr_hierarchy:
                spatial_levels_for_engine.append(lvl.level)
                hierarchy_dims.append(lvl.from_dimension)
                if lvl.level == fdr_resolution:
                    break

        # Truncate temporal walk at user-named level
        temporal_levels_for_engine: list[str] = []
        temporal_dim: str | None = None
        if fdr_temporal_resolution is not None:
            for lvl in pattern.fdr_temporal_hierarchy:
                temporal_levels_for_engine.append(lvl.level)
                temporal_dim = lvl.slice_dimension
                if lvl.level == fdr_temporal_resolution:
                    break

        # Minimal-cost read of the population indicator + cell-defining cols
        read_cols = ["primary_key", "is_anomaly", *hierarchy_dims]
        if temporal_dim is not None:
            read_cols.append(temporal_dim)
        try:
            geo_slice = self._storage.read_geometry(
                pattern_id, version, columns=read_cols,
            )
        except _NAVIGATION_RECOVERABLE_ERRORS as exc:
            raise GDSNavigationError(
                f"fdr_resolution / fdr_temporal_resolution gating failed to "
                f"read geometry for pattern {pattern_id!r}: {exc}",
            ) from exc

        missing = [c for c in read_cols if c not in geo_slice.schema.names]
        if missing:
            raise GDSNavigationError(
                f"fdr_resolution / fdr_temporal_resolution requires geometry "
                f"columns {missing!r} — rebuild the sphere after declaring "
                f"fdr_hierarchy / fdr_temporal_hierarchy on pattern "
                f"{pattern_id!r}",
            )

        # Pure engine math: per-cell p-values + per-level FDR
        cell_p = cell_p_values_from_anomaly_indicator(
            geo_slice,
            hierarchy_dims=hierarchy_dims or None,
            temporal_dim=temporal_dim,
            anomaly_col="is_anomaly",
        )
        cell_q, surviving_cells = fdr_multi_resolution(
            cell_p,
            hierarchy=spatial_levels_for_engine or None,
            temporal_levels=temporal_levels_for_engine or None,
            method=fdr_method,
            alpha=fdr_alpha,
        )

        # Build per-entity cell-tuple lookup from the same slice
        cell_dims = [*hierarchy_dims]
        if temporal_dim is not None:
            cell_dims.append(temporal_dim)
        pks = geo_slice["primary_key"].to_pylist()
        cell_value_cols = [geo_slice[d].to_pylist() for d in cell_dims]
        cell_by_entity: dict[str, tuple] = {}
        for i, pk in enumerate(pks):
            cell_by_entity[pk] = tuple(col[i] for col in cell_value_cols)

        # Filter survivors (set membership preserves order)
        filtered = [
            p for p in polygons
            if cell_by_entity.get(p.primary_key) in surviving_cells
        ]

        # Annotate survivors
        spatial_n = len(hierarchy_dims)
        for p in filtered:
            cell = cell_by_entity.get(p.primary_key)
            if cell is None:
                continue
            q_value = cell_q.get(cell)
            spatial_part = cell[:spatial_n]
            temporal_part = cell[spatial_n:]
            if hierarchy_dims:
                p.cell_q_spatial = q_value  # type: ignore[attr-defined]
            if temporal_dim is not None:
                p.cell_q_temporal = q_value  # type: ignore[attr-defined]
            spatial_pairs = list(
                zip(spatial_levels_for_engine, spatial_part, strict=True),
            )
            temporal_pairs = list(
                zip(temporal_levels_for_engine, temporal_part, strict=True),
            )
            p.cell_path = tuple(spatial_pairs + temporal_pairs)  # type: ignore[attr-defined]

        return filtered

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
        fdr_method: str | None = None,
        p_value_method: str | None = None,
        fdr_axis: str = "entity",
        fdr_resolution: str | None = None,
        fdr_temporal_resolution: str | None = None,
        rank_by: str = "delta_norm",
        select: str = "top_norm",
        metric: str = "L2",
        min_confidence: float = 0.0,
        dimension_weights: dict[str, float] | None = None,
        sample_size: int | None = None,
        boundary_aware: bool = False,
    ) -> tuple[list[Polygon], int, list[dict] | None, dict | None]:
        """Find the most anomalous polygons in a pattern.

        Returns ``(polygons, total_found, emerging, meta)`` where *emerging*
        is a list of not-yet-anomalous entities trending toward theta (only
        when *include_emerging=True* and *offset == 0*), and *meta* is a dict
        with ``total_anomalies_unfiltered`` when *property_filters* is set,
        or ``None`` otherwise.

        ``dimension_weights`` (optional) — per-dimension multipliers applied
        to the delta vector before computing the rank score. Default ``None``
        leaves behaviour unchanged. When set, dims missing from the dict
        default to ``1.0``; explicit ``0.0`` silences a dim. An empty dict
        ``{}`` is equivalent to ``None`` (no weights). Requires
        ``metric in ('L2', 'Linf')`` — Bregman divergence is precomputed and
        cannot be reweighted post-hoc. Forces the in-process scan path
        (the subprocess fast path relies on Lance's stored ``delta_norm``).
        Connects Theme E correlation-gate verdicts to runtime ranking
        (NOISE-classified dim → weight ``0.0``; HEAVY-TAIL dim → ``0.5``).

        Note: when weights are active, only ``delta_norm`` on the returned
        polygons reflects the weighted rank score. ``delta`` is the raw
        unweighted vector; ``is_anomaly``, ``bregman_divergence``,
        ``delta_rank_pct``, and ``emerging[]`` come from storage and reflect
        the unweighted calibration. The threshold (``theta_norm``) is also
        unweighted — silencing dims naturally reduces the surviving count.

        ``rank_by`` (default ``"delta_norm"``) selects the final ranking key
        for survivors:
          - ``"delta_norm"`` — sort by ``||delta||`` descending (default).
          - ``"min_q_per_dim"`` — sort by per-dim FDR q-value ascending
            (smallest q first). Re-ranks the same delta_norm top-N pool by
            the dimension that produced the strongest single-dim signal;
            requires ``fdr_alpha`` set and ``fdr_axis in {"per_dim", "both"}``;
            incompatible with ``select="diverse"``.
          - ``"signed_confidence"`` — sort by a confidence-weighted signed
            score ``delta_norm_signed × |lda_alignment| × (1 - penalty)``
            descending, where ``delta_norm_signed = delta · direction`` and
            ``lda_alignment = (delta / ||delta||) · direction`` project onto
            the Fisher LDA axis declared by ``label_audit:``, and
            ``penalty = 0.5 · single_dim_driven + 0.5 ·
            low_confidence_bucket``. Each surviving polygon carries
            ``signed_confidence_score`` / ``lda_alignment`` /
            ``reliability_penalty`` for transparency. The sort is sign-
            preserving — anti-aligned polygons (negative signed score)
            land at the bottom by design, not the top. Requires
            ``pattern.label_aware_calibration`` to be populated (raises
            ``GDSNavigationError`` otherwise — no silent fallback to
            ``delta_norm``); incompatible with ``select="diverse"``.

        Warning: ``fdr_axis="per_dim"`` and ``rank_by="min_q_per_dim"`` use
        chi²(1) two-sided survival on ``delta_i,d``, which is *direction-
        agnostic* — both extreme-positive and extreme-negative deviations
        produce small p-values. On spheres with anti-signal dims (high
        ``|delta|`` correlated with non-fraud — see ``engine.dim_audit``
        per-dim label AUROC), per-dim FDR can flag both wings. When ground-
        truth labels exist, combine with
        ``engine.dim_audit.compute_per_dim_label_auroc`` to identify and
        down-weight anti-signal dims via ``dimension_weights`` before
        ranking.

        ``fdr_resolution`` (optional) — when set, filter survivors to those
        in cells that clear per-level BH/Storey FDR on the spatial hierarchy
        declared on the pattern (``Pattern.fdr_hierarchy``). Value must be a
        ``level`` name declared on the pattern's ``fdr_hierarchy``. Requires
        ``fdr_alpha`` set. Survivors gain ``cell_q_spatial`` and ``cell_path``
        attributes.

        ``fdr_temporal_resolution`` (optional) — same, for the temporal
        hierarchy declared via ``Pattern.fdr_temporal_hierarchy``. Survivors
        gain ``cell_q_temporal`` and ``cell_path`` attributes.

        When either ``fdr_resolution`` or ``fdr_temporal_resolution`` is set
        on the entity axis (the default), the entity-level FDR parameters
        switch to ``p_value_method='chi2'`` and ``fdr_method='storey'`` —
        but only when the caller left those values implicit. Implementation
        uses the ``None`` sentinel: ``p_value_method=None`` (and analogously
        ``fdr_method=None``) means "pick the context-correct value", which
        is ``'chi2'`` / ``'storey'`` when ``fdr_resolution`` is active and
        ``'rank'`` / ``'bh'`` otherwise. A caller who explicitly passes
        ``p_value_method='rank'`` keeps rank semantics — useful for
        reproducing pre-upgrade behaviour, validating migrations, or
        benchmarking the degenerate path on purpose. ``fdr_axis`` in
        ``{'per_dim', 'both'}`` skips the upgrade since per-dim chi²(1)
        is independent of ``p_value_method``. Rank-based p-values are
        uniform-by-construction; BH at any reasonable alpha rejects nothing,
        which would collapse the entity-level FDR to a no-op and leave
        survivors ordered by ``delta_norm`` alone — that's the failure mode
        the context-sensitive default avoids by default.

        When both are set, intersection-FDR applies: an entity survives iff
        its cell cleared every named spatial level AND every named temporal
        level. The gate uses ``hypergeom`` upper-tail on ``is_anomaly`` per
        cell, then per-level BH/Storey FDR via
        ``hypertopos.engine.fdr.fdr_multi_resolution``. ``cell_q_spatial`` and
        ``cell_q_temporal`` both carry the conservative joint q (element-wise
        max across the cell's projections at each level) when both axes are
        active, since ``fdr_multi_resolution`` returns a single per-cell q.

        ``sample_size`` (optional, default ``None`` = scan full population) —
        cap on the number of geometry rows scored. When set below the
        population size, a random sample is drawn before threshold filtering
        and ranking. Forces the in-process scan path (the Lance pushdown fast
        path operates on the full population).

        ``boundary_aware`` (optional, default ``False``) — stratified sampling
        mode for ``sample_size``. When ``True``, half the budget is drawn
        from entities in the ``[0.8 * theta_norm, 1.2 * theta_norm]`` band
        around the decision threshold (boundary cases), the other half from
        the rest of the population. Useful for calibration audits — uniform
        sampling under-represents boundary entities since they make up a
        minority of the population. Requires ``sample_size`` to be set; raises
        ``ValueError`` otherwise. Forces the in-process scan path.
        """
        # Resolve sentinel-None defaults BEFORE the fdr_method / p_value_method
        # value-set validation runs (the check directly below expects a string).
        # Upgrade applies only when fdr_resolution / fdr_temporal_resolution is
        # set on the entity axis (per_dim / both compute chi2(1) per-dim
        # regardless of p_value_method). Explicit non-None values pass through
        # — the sentinel distinguishes "caller left this implicit" from
        # "caller asked for this specific value".
        _needs_resolution_upgrade = (
            fdr_axis == "entity"
            and (fdr_resolution is not None or fdr_temporal_resolution is not None)
        )
        if p_value_method is None:
            p_value_method = "chi2" if _needs_resolution_upgrade else "rank"
        if fdr_method is None:
            fdr_method = "storey" if _needs_resolution_upgrade else "bh"
        if fdr_method not in ("bh", "storey"):
            raise ValueError(
                f"fdr_method must be 'bh' or 'storey', got {fdr_method!r}"
            )
        if metric not in ("L2", "Linf"):
            raise ValueError(f"metric must be 'L2' or 'Linf', got '{metric}'")
        if boundary_aware and sample_size is None:
            raise ValueError(
                "boundary_aware=True requires sample_size to be set — "
                "stratified sampling needs an explicit budget."
            )
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")
        if rank_by not in ("delta_norm", "min_q_per_dim", "signed_confidence"):
            raise ValueError(
                f"rank_by must be 'delta_norm', 'min_q_per_dim', or "
                f"'signed_confidence', got {rank_by!r}"
            )
        if rank_by == "min_q_per_dim":
            if fdr_alpha is None:
                raise ValueError(
                    "rank_by='min_q_per_dim' requires fdr_alpha to be set",
                )
            if fdr_axis not in ("per_dim", "both"):
                raise ValueError(
                    "rank_by='min_q_per_dim' requires fdr_axis in {'per_dim', 'both'} "
                    f"(got fdr_axis={fdr_axis!r}) — without per-dim FDR no q-values "
                    "are computed to sort by",
                )
            if select != "top_norm":
                raise ValueError(
                    "rank_by='min_q_per_dim' is incompatible with select='diverse' — "
                    "diverse selection re-orders by representativeness, not q-values",
                )
        if rank_by == "signed_confidence" and select != "top_norm":
            raise ValueError(
                "rank_by='signed_confidence' is incompatible with select='diverse' — "
                "diverse selection re-orders by representativeness, not signed scores",
            )
        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        # signed_confidence ranking requires label-aware calibration —
        # fail fast with a structured error instead of silently degrading
        # to delta_norm ranking, so callers know they need to rebuild with
        # ``label_audit:`` enabled.
        if rank_by == "signed_confidence" and pattern.label_aware_calibration is None:
            raise GDSNavigationError(
                "signed_confidence ranking requires a label_audit:-enabled "
                "pattern; this pattern has no label-aware calibration. "
                "Rebuild with label_audit: pointing at this pattern, or use "
                "rank_by='delta_norm' (default).",
            )
        # Multi-resolution FDR — validate against pattern-declared hierarchy
        if fdr_resolution is not None or fdr_temporal_resolution is not None:
            if fdr_alpha is None:
                raise ValueError(
                    "fdr_resolution / fdr_temporal_resolution require "
                    "fdr_alpha to be set",
                )
            if fdr_resolution is not None:
                spatial_levels = [lvl.level for lvl in pattern.fdr_hierarchy]
                if fdr_resolution not in spatial_levels:
                    raise ValueError(
                        f"fdr_resolution={fdr_resolution!r} not in pattern's "
                        f"fdr_hierarchy levels {spatial_levels!r}",
                    )
            if fdr_temporal_resolution is not None:
                temporal_levels_decl = [
                    lvl.level for lvl in pattern.fdr_temporal_hierarchy
                ]
                if fdr_temporal_resolution not in temporal_levels_decl:
                    raise ValueError(
                        f"fdr_temporal_resolution={fdr_temporal_resolution!r} "
                        f"not in pattern's fdr_temporal_hierarchy levels "
                        f"{temporal_levels_decl!r}",
                    )
        weight_vector = (
            self._build_dimension_weight_vector(pattern, dimension_weights, metric)
            if dimension_weights
            else None
        )

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
                int(self._storage.count_geometry_rows(pattern_id))
                if not missing_edge_to and not property_filters else 0
            )
        except _NAVIGATION_RECOVERABLE_ERRORS:
            _geo_count = 0
        if (
            _geo_count > 500_000
            and rank_by_property is None
            and metric == "L2"
            and weight_vector is None
            and rank_by != "signed_confidence"
            and sample_size is None
            and not boundary_aware
        ):
            from hypertopos.engine.lance_sql_agg import (
                find_anomalies as _lance_sql_find_anomalies,
            )
            geo_lance_path = str(
                self._storage._base.resolve()
                / "geometry" / pattern_id / "data.lance"
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
                fdr_axis=fdr_axis,
            )
            if rank_by == "min_q_per_dim":
                results.sort(key=lambda p: p.min_q_per_dim)
            if fdr_resolution is not None or fdr_temporal_resolution is not None:
                results = self._apply_fdr_multi_resolution_gate(
                    results,
                    pattern=pattern,
                    pattern_id=pattern_id,
                    version=version,
                    fdr_resolution=fdr_resolution,
                    fdr_temporal_resolution=fdr_temporal_resolution,
                    fdr_method=fdr_method,
                    fdr_alpha=fdr_alpha,  # type: ignore[arg-type]
                )
            emerging = self._find_emerging(
                pattern_id, version, pattern, include_emerging, offset, top_n,
            )
            self._attach_reliability_flags(results, pattern=pattern)
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
        # Sampling needs visibility into entities BELOW the threshold (the
        # boundary band straddles theta_norm and the uniform-sampling acceptance
        # ratio assumes the full population is in scope). Drop the pre-filter
        # in that case — the threshold check still happens post-sampling below.
        _drop_norm_prefilter = sample_size is not None or boundary_aware
        if metric in ("Linf", "bregman") or _drop_norm_prefilter:
            # Linf/bregman: can't use pre-computed delta_norm filter, read all
            light = self._storage.read_geometry(
                pattern_id, version, columns=cols,
            )
        else:
            light = self._storage.read_geometry(
                pattern_id, version, columns=cols,
                filter=f"delta_norm >= {threshold}",
            )
        if _drop_norm_prefilter and light.num_rows > 0:
            light = self._stratified_sample_light(
                light, sample_size=sample_size, boundary_aware=boundary_aware,
                theta_norm=theta_norm,
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
        if weight_vector is not None:
            delta_matrix = delta_matrix * weight_vector
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
                    "property_filters requires an entity line — "
                    "event patterns don't have entity properties"
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
            if rank_by == "signed_confidence":
                # signed_confidence re-ranks the candidate pool by a score that
                # composes delta_norm_signed, LDA alignment, and reliability
                # flags — top_n by delta_norm is too narrow to surface anti-
                # aligned demotions, so we widen the pool to the full anomalous
                # population. The post-truncation top_n still bounds the
                # returned envelope.
                n = len(sorted_local)
            else:
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
        # signed_confidence defers the top_n truncation until after the
        # score is computed so the full anomalous pool is re-ranked rather
        # than only the delta_norm top_n.
        if rank_by != "signed_confidence":
            results = results[offset:offset + top_n]
        results = self._apply_fdr_select_polygons(
            results, fdr_alpha=fdr_alpha, select=select, top_n=top_n,
            fdr_method=fdr_method, p_value_method=p_value_method,
            pattern_df=len(pattern.theta) if p_value_method == "chi2" else None,
            fdr_axis=fdr_axis,
        )
        if rank_by == "min_q_per_dim":
            results.sort(key=lambda p: p.min_q_per_dim)
        if fdr_resolution is not None or fdr_temporal_resolution is not None:
            results = self._apply_fdr_multi_resolution_gate(
                results,
                pattern=pattern,
                pattern_id=pattern_id,
                version=version,
                fdr_resolution=fdr_resolution,
                fdr_temporal_resolution=fdr_temporal_resolution,
                fdr_method=fdr_method,
                fdr_alpha=fdr_alpha,  # type: ignore[arg-type]
            )

        emerging = self._find_emerging(
            pattern_id, version, pattern, include_emerging, offset, top_n,
        )
        meta = {"total_anomalies_unfiltered": total_anomalies_unfiltered} if property_filters else None
        self._attach_reliability_flags(results, pattern=pattern)
        if rank_by == "signed_confidence":
            self._attach_signed_confidence_fields(results, pattern=pattern)
            # Sort by signed_confidence_score descending; sign of
            # delta_norm_signed is preserved, so anti-aligned polygons
            # (negative scores) land at the bottom by design.
            results.sort(
                key=lambda p: (
                    -float(getattr(p, "signed_confidence_score", 0.0) or 0.0),
                    p.primary_key,
                ),
            )
            results = results[offset:offset + top_n]
        return results, total_found, emerging, meta

    @staticmethod
    def _build_dimension_weight_vector(
        pattern: Any,
        weights: dict[str, float],
        metric: str,
    ) -> np.ndarray:
        """Resolve a ``{dim_name: weight}`` mapping to a per-dim weight vector.

        Accepts the same dim-name vocabulary as ``Pattern.dim_index``
        (relation ``line_id`` OR ``display_name``; event-dim ``column`` OR
        ``display_name``; ``prop_columns``) plus aggregated edge-dim names
        (e.g. ``amount_mean``) which only surface via ``dim_labels``.

        Validates that every supplied dim exists on the pattern and that
        every weight is a non-negative finite number. Missing dims default
        to ``1.0``. Raises ``ValueError`` for ``metric='bregman'`` because
        Bregman divergence is precomputed per-row and cannot be reweighted.
        """
        if metric == "bregman":
            raise ValueError(
                "dimension_weights requires metric in ('L2', 'Linf'); "
                "Bregman divergence is precomputed per-row and cannot be "
                "reweighted post-hoc"
            )
        if not isinstance(weights, dict):
            raise ValueError(
                f"dimension_weights must be a dict[str, float], got "
                f"{type(weights).__name__}"
            )
        labels = pattern.dim_labels
        # Edge-dim aggregations live at the tail of dim_labels and are not
        # resolvable via Pattern.dim_index — build a fallback map for them.
        edge_agg_names = pattern._edge_dim_aggregation_names()
        n_non_edge = len(labels) - len(edge_agg_names)
        edge_agg_to_idx = {
            name: n_non_edge + i for i, name in enumerate(edge_agg_names)
        }
        vec = np.ones(len(labels), dtype=np.float32)
        for name, raw in weights.items():
            # Try Pattern.dim_index first — it accepts both line_id and
            # display_name on relations / event_dimensions, plus prop_columns.
            try:
                idx = pattern.dim_index(name)
            except ValueError:
                idx = edge_agg_to_idx.get(name)
                if idx is None:
                    raise ValueError(
                        f"dimension_weights: unknown dimension {name!r}. "
                        f"Valid dims for pattern '{pattern.pattern_id}': "
                        f"{labels}"
                    ) from None
            try:
                w = float(raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"dimension_weights[{name!r}] must be a number, got "
                    f"{raw!r}"
                ) from e
            if not np.isfinite(w) or w < 0:
                raise ValueError(
                    f"dimension_weights[{name!r}] must be a non-negative "
                    f"finite number, got {w}"
                )
            vec[idx] = w
        return vec

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
            dim_sq_totals += sq * weight
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
        total = self._storage.count_geometry_rows(pattern_id)
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
                fdr_axis="entity",
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
        with_neighbor_anomaly: bool = False,
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

        with_neighbor_anomaly: when True, look up stored ``is_anomaly`` per neighbour
        from the geometry dataset and populate ``SimilarityResult.is_anomaly_map``.
        Cost is one BTREE point lookup over top_n keys — typically <1 ms for
        top_n <= 50. Default ``False`` keeps existing callers unaffected.
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

        is_anomaly_map: dict[str, bool] | None = None
        if with_neighbor_anomaly and results:
            full_map = self._get_anomaly_map(pattern_id, version)
            is_anomaly_map = {
                k: full_map[k] for k, _ in results if k in full_map
            }
        elif with_neighbor_anomaly:
            is_anomaly_map = {}

        return SimilarityResult(
            results,
            degenerate_warning=degenerate_warning,
            is_anomaly_map=is_anomaly_map,
        )

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

        When the AdjacencyIndex is not yet cached for ``pattern_id``, the
        index build pays the full sort-+-2×groupby cost over the entire edge
        table (~13 s on a 5 M-edge AML pattern) before the per-entity
        lookup runs.  For a single-entity query that cold-build is wasted
        work — we route through Lance BTREE-pushdown filters on
        ``from_key``/``to_key`` instead (single-key equality scan,
        ~100 ms on the same dataset).  The full AdjacencyIndex is still
        built lazily on the first call that legitimately needs it (e.g.
        ``entity_flow``, ``contagion_score``).
        """
        adj_cache = getattr(self._storage, "_adjacency_cache", None)
        adj_warm = isinstance(adj_cache, dict) and pattern_id in adj_cache

        if adj_warm:
            adj = self._storage.get_adjacency(pattern_id)
            fwd_edges = adj.neighbors_out(primary_key, ts_to=timestamp_cutoff)
            rev_edges = adj.neighbors_in(primary_key, ts_to=timestamp_cutoff)
        else:
            # Cold path: direct BTREE-pushdown filter on the edge table.
            # ``read_edges(from_keys=[pk])`` uses the Lance BTREE index on
            # ``from_key`` for O(log n) lookup — sub-second even on 5 M-row
            # edge tables.  Convert each row to the (target, ts, amount,
            # event_key) tuple shape the downstream ``_group`` helper
            # expects so the two paths produce identical results.
            from_tbl = self._storage.read_edges(
                pattern_id,
                from_keys=[primary_key],
                timestamp_to=timestamp_cutoff,
                columns=["to_key", "timestamp", "amount", "event_key"],
            )
            to_tbl = self._storage.read_edges(
                pattern_id,
                to_keys=[primary_key],
                timestamp_to=timestamp_cutoff,
                columns=["from_key", "timestamp", "amount", "event_key"],
            )
            fwd_edges = list(zip(
                from_tbl["to_key"].to_pylist(),
                from_tbl["timestamp"].to_pylist(),
                from_tbl["amount"].to_pylist(),
                from_tbl["event_key"].to_pylist(),
                strict=True,
            ))
            rev_edges = list(zip(
                to_tbl["from_key"].to_pylist(),
                to_tbl["timestamp"].to_pylist(),
                to_tbl["amount"].to_pylist(),
                to_tbl["event_key"].to_pylist(),
                strict=True,
            ))

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
                if pattern.pattern_type == "event":
                    entity_hint = (
                        f"Pattern '{pattern_id}' is an event pattern — "
                        f"from_key / to_key must be event (transaction) keys "
                        f"that index its geometry, not anchor / account keys. "
                        f"Use search_entities(line_id=<event line>) to list "
                        f"valid event keys, or pass an anchor pattern_id "
                        f"(e.g. the corresponding account_pattern) if you want "
                        f"to score an account-to-account edge."
                    )
                else:
                    entity_hint = (
                        f"Pattern '{pattern_id}' is an anchor pattern — "
                        f"from_key / to_key must be primary keys present in "
                        f"its geometry. Use search_entities(line_id=<anchor "
                        f"entity line>) or find_anomalies(pattern_id='{pattern_id}') "
                        f"to discover valid keys."
                    )
                raise GDSNavigationError(
                    f"Entity '{key}' not found in pattern '{pattern_id}' "
                    f"geometry. {entity_hint}"
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

                # Endpoint-namespace probe: when the edge endpoint keys live
                # in a different namespace than the geometry primary_key
                # (e.g. event pattern whose geometry holds transactions but
                # whose edge table joins accounts), no pair can ever match
                # the geometry primary keys.  Sample one edge endpoint and
                # check whether it exists in the geometry — a single BTREE
                # lookup (~10 ms) replaces a 24 s full-edge groupby followed
                # by a 25 s geometry scan when the namespaces are disjoint.
                # Ratio guard above (0.01) misses this regime when the
                # endpoint count is comparable to the geometry rowcount but
                # the keys themselves are still disjoint (e.g. AML tx_pattern
                # with 917 k account endpoints vs 5 M transaction rows).
                if base_sorted is None and pattern.pattern_type == "event":
                    try:
                        sample_edge = self._storage.read_edges(
                            graph_pid, columns=["from_key"],
                        ).slice(0, 1)
                        if sample_edge.num_rows > 0:
                            sample_endpoint = sample_edge["from_key"][0].as_py()
                            probe = self._storage.read_geometry(
                                pattern_id, version,
                                primary_key=sample_endpoint,
                                columns=["primary_key"],
                            )
                            if probe.num_rows == 0:
                                self._attract_edge_base_cache[base_key] = []
                                base_sorted = []
                    except (GDSNavigationError, GDSError):
                        # Probe failure (e.g. edge table missing) — fall
                        # through to the legacy scoring path which will
                        # raise the canonical error in context.
                        pass

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
                    geo_row_count = self._storage.count_geometry_rows(pattern_id)
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
        self._require_anchor_pattern_for_motif(pattern_id)
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
        self._require_anchor_pattern_for_motif(pattern_id)
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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
        adj: AdjacencyIndex,
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

    def _require_anchor_pattern_for_motif(self, pattern_id: str) -> None:
        """Reject event ``pattern_id`` on agent-facing motif primitives.

        Motif edges live in the event pattern but the SEEDS and the per-entity
        geometry that drives ``edge_potential`` scoring live in the ANCHOR
        pattern. When an agent passes the event pattern directly, the
        geometry's ``primary_key`` column carries event keys (e.g. transaction
        ids) while the adjacency index carries entity keys — every seed fails
        the active-seed gate and the call returns an empty list after burning
        the full enumeration cost. Detect the case early and point the agent
        at the anchor companion.

        Best-effort: this helper is a pre-check that ENRICHES the error message
        when an obvious redirect exists. Pattern-not-found and pattern-type
        edge cases fall through to ``_resolve_motif_graph_pid`` which raises
        its own, more authoritative diagnostic.
        """
        try:
            sphere = self._storage.read_sphere()
        except Exception:  # noqa: BLE001
            return
        pattern = sphere.patterns.get(pattern_id) if hasattr(
            sphere.patterns, "get",
        ) else None
        if pattern is None or getattr(pattern, "pattern_type", None) != "event":
            return
        suggestion: str | None = None
        for anchor_id, anchor in sphere.patterns.items():
            if anchor.pattern_type != "anchor":
                continue
            agg = anchor.edge_dim_aggregations
            if agg is not None and agg.from_event_pattern == pattern_id:
                suggestion = anchor_id
                break
        if suggestion is None:
            # No anchor pattern points at this event pattern — legacy event-only
            # sphere. Allow the call; the per-event geometry's primary_key may
            # itself index the entities (engineered fixtures, single-pattern
            # spheres). Raising here would block valid edge cases without an
            # actionable redirect.
            return
        raise GDSNavigationError(
            f"Pattern '{pattern_id}' is an event pattern — motif primitives "
            f"require an anchor pattern_id (their seeds and per-entity "
            f"geometry index entities, not events): use the anchor companion "
            f"'{suggestion}' instead.",
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
            _ew = len(pattern.edge_max)
            return np.sum(shape_matrix[:, :_ew] * pattern.edge_max, axis=1).astype(np.float64)
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
                # edge_max covers structural relations only; edge_dim_aggregations
                # dims have no edge_max entry. Slice shape_matrix to edge_max width
                # before the per-relation multiply.
                _ew = len(pattern.edge_max)
                scores = np.sum(shape_matrix[:, :_ew] * pattern.edge_max, axis=1)

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
                _ew = len(pattern.edge_max)
                scores = np.sum(shape_matrix[:, :_ew] * pattern.edge_max, axis=1).astype(np.float64)
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
        _ew = len(pattern.edge_max)

        def _score(delta: np.ndarray) -> tuple[float, int]:
            shape_vec = delta[:_ew] * sigma[:_ew] + pattern.mu[:_ew]
            s = float(np.sum(shape_vec * pattern.edge_max))
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
            all_shapes = np.array(t_shape[start:end], dtype=np.float32)  # shape (n, d)
            # Temporal shape_snapshot carries structural dims only; aggregated
            # edge_dim dims have no temporal history. Slice calibration arrays
            # to the snapshot width before the broadcast.
            _w = all_shapes.shape[-1]
            _sigma = np.maximum(pattern.sigma_diag[:_w], 1e-2)
            all_deltas = (all_shapes - pattern.mu[:_w]) / _sigma
            delta_first = all_deltas[0]
            delta_last = all_deltas[-1]
            diff = delta_last - delta_first

            displacement = float(np.linalg.norm(diff[:n_rel]))
            # base_delta comes from geometry (full delta_dim incl. aggregations);
            # delta_first is shape_snapshot width (structural only). Slice
            # base_delta to match before the diff.
            base_delta = base_deltas[bk][:_w]
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
                total = self._storage.count_geometry_rows(pid)
            # Count anomalies by delta_norm >= theta_norm to match
            # anomaly_summary / find_anomalies semantics.
            # The is_anomaly column uses per-group thetas for grouped
            # patterns, which diverges from the global theta_norm used
            # by find_anomalies — leading to contradictory counts.
            theta_norm_raw = float(np.linalg.norm(pattern.theta))
            if theta_norm_raw > 0.0:
                anomaly_count = self._storage.count_geometry_rows(
                    pid, filter=f"delta_norm >= {theta_norm_raw}"
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

            # theta_sensitivity_summary — compact diagnostic from latest
            # calibration epoch's populated theta_sensitivity field.
            # Skipped silently for pre-T2 spheres (no field on disk).
            ts_summary = self._build_theta_sensitivity_summary(pid)
            if ts_summary is not None:
                entry["theta_sensitivity_summary"] = ts_summary

            # dim_quality_warnings — dead dims (sigma_diag near zero, z-score
            # undefined) and sparse dims (median zero with rare nonzero,
            # gaussian assumption wrong). Both are silent build-time failures
            # of the "trust the space" frame: the dim sits in the delta
            # vector contributing nothing or contributing wrong signal,
            # and the investigator has no way to know without scrolling
            # the calibration log. Surfacing here makes the failure
            # auditable from sphere_overview.
            warnings = self._compute_dim_quality_warnings(pattern)
            if warnings:
                entry["dim_quality_warnings"] = warnings

            results.append(entry)
        return results

    @staticmethod
    def _compute_dim_quality_warnings(pattern: Any) -> list[dict[str, Any]]:
        """Surface build-time dim-quality issues that silently break z-score
        / delta_norm semantics.

        Five classes flagged:

        - **dead_dim**: ``sigma_diag[i] < 1e-10`` — the dim has zero
          variance across the population, so the z-score `(x - mu) / sigma`
          is undefined / explodes. The dim contributes nothing meaningful
          to ``delta_norm`` and silently dilutes any other dim's signal.

        - **sparse_dim**: ``dim_percentiles[d]['p50'] == 0`` AND
          ``dim_percentiles[d]['p99'] > 0`` — the dim is mostly-zero with
          rare nonzero values, so the gaussian z-score assumption is
          wrong (the empirical distribution is point-mass-at-zero plus a
          tail). Bregman divergence with a poisson / bernoulli kind tag
          is the correct distance.

        - **dominant_dim_mass**: one dim contributes ``>=70%`` of the
          population's p99-tail variance mass (``z_p99 ** 2`` summed
          across dims with percentile coverage). Top-N anomaly ranking
          is then effectively a one-dim detector and the geometric
          framing is misleading.

        - **negative_space**: a dim declared with ``kind='gaussian'``
          whose empirical median is exactly zero — the gaussian z-score
          sits at the mode of the distribution rather than the tail, so
          every anomaly score on this dim is geometry noise. The
          existing ``sparse_dim`` rule fires regardless of declared kind
          and asks for a re-declare; this rule narrows the action to
          gaussian-only cases where the kind itself is the bug.

        - **non_normal_dim**: a dim declared with ``kind='gaussian'``
          whose build-time Shapiro-Wilk / KS p-value is below
          ``_NON_NORMAL_DIM_PVALUE_THRESHOLD`` — the gaussian z-score
          assumes the dim distribution is approximately normal. When
          the empirical distribution is heavy-tailed (Pareto, log-normal
          income / amount data), z-scoring concentrates the delta mass
          in a few extreme rows and poorly discriminates the bulk of
          the population. Test family is picked per-dim at build time
          by sample size (Shapiro-Wilk for N <= 5000, KS otherwise).
          Suppressed when ``negative_space`` already fires on the same
          dim — that warning's remediation (re-declare kind) supersedes
          the normality complaint, and double-flagging is noise.

        - **kind_mismatch**: a dim declared with ``kind='gaussian'``
          whose Fisher LDA direction component is below
          ``_KIND_MISMATCH_DIRECTION_THRESHOLD`` (the global axis
          assigns near-zero weight to the dim) WHILE the raw per-class
          moments still separate
          (``|cohens_d_pos_neg| >= _KIND_MISMATCH_COHENS_D_THRESHOLD``).
          The combination means the dim's variance is captured by
          another dim's Fisher axis — re-declaring kind or splitting
          the dim into binary + continuous components is the right
          remediation. Pre-requisite: ``Pattern.label_aware_calibration``
          must be non-None; patterns built without the
          ``label_audit:`` block in sphere.yaml are skipped silently.
          Suppressed when ``negative_space`` already fires on the same
          dim — same dedup convention as ``non_normal_dim``.

        - **signed_tail_concentration**: pattern-level warning fired
          when ``Pattern.signed_percentiles`` shows
          ``|p99| / max(|p50|, 1e-9) > _SIGNED_TAIL_RATIO_THRESHOLD``
          on the Fisher LDA-projected delta distribution. Indicates
          the label-discriminating axis is being driven by a tiny
          outlier subgroup rather than the broad positive / negative
          class split. Suppressed when
          ``Pattern.label_aware_n_pos < _SIGNED_TAIL_MIN_N_POS``
          (positive-class undersampled — LDA fit itself unstable).
          ``dim_label`` is ``<pattern_id>:signed_percentiles`` to
          signal pattern-level rather than dim-level scope.

        All warnings are computed from cached pattern state
        (``sigma_diag``, ``dim_percentiles``, ``mu``,
        ``dimension_kinds``, ``dim_normality_pvalues``,
        ``label_aware_calibration``, ``signed_percentiles``,
        ``label_aware_n_pos``) — sub-millisecond, no storage scan.
        """
        def _build_raw_dim_name_to_index(pattern: Any) -> dict[str, int]:
            """Map dim_percentiles raw keys (column / line_id-without-_d_
            prefix / prop name) to their delta-vector index. Mirrors the
            sparse_dim auditor's dim_columns set construction so the
            pattern-level mass + negative-space auditors look up percentile
            entries by the same key the existing sparse_dim path uses.
            """
            mapping: dict[str, int] = {}
            for i, rel in enumerate(pattern.relations):
                if rel.line_id.startswith("_d_"):
                    stripped = rel.line_id[3:]
                    mapping[stripped] = i
                    # Some builders also strip a leading namespace segment
                    # from the column when persisting dim_percentiles —
                    # e.g. `_d_chain_hop_count` line_id appears in the
                    # percentile cache as `hop_count`, not `chain_hop_count`.
                    # Map the drop-first-segment alias too (first match wins
                    # via setdefault so the exact full-strip name remains
                    # canonical for relations whose full stripped name IS
                    # the percentile key).
                    parts = stripped.split("_", 1)
                    if len(parts) > 1:
                        mapping.setdefault(parts[1], i)
                else:
                    mapping[rel.line_id] = i
            k = len(pattern.relations)
            for j, ed in enumerate(pattern.event_dimensions):
                mapping[ed.column] = k + j
            k2 = k + len(pattern.event_dimensions)
            for j, prop in enumerate(pattern.prop_columns):
                mapping[prop] = k2 + j
            return mapping

        def _compute_dominant_dim_mass_warning(
            pattern: Any,
        ) -> dict[str, Any] | None:
            """Detect single-dim domination of the p99 tail variance mass.

            For every dim with percentile coverage the squared z-score at
            p99 is the dim's contribution to the population's tail mass;
            if one dim's share crosses the threshold the geometric
            anomaly score collapses into a one-dim detector. Returns
            None when no dim crosses the threshold, when mu / sigma are
            unavailable, or when no dim survives the per-dim
            preconditions (zero variance, no positive tail, no
            percentile cache).
            """
            mu = getattr(pattern, "mu", None)
            sigma_diag = getattr(pattern, "sigma_diag", None)
            if mu is None or sigma_diag is None:
                return None

            dim_percentiles = getattr(pattern, "dim_percentiles", None) or {}
            if not dim_percentiles:
                return None
            raw_to_idx = _build_raw_dim_name_to_index(pattern)
            mu_arr = np.asarray(mu, dtype=np.float64)
            sigma_arr = np.asarray(sigma_diag, dtype=np.float64)

            survivors: list[tuple[int, str, float]] = []
            for raw_name, stats in dim_percentiles.items():
                i = raw_to_idx.get(raw_name)
                if i is None or i >= len(mu_arr) or i >= len(sigma_arr):
                    # Aggregated edge dim (no slot in relations) or
                    # length mismatch — exclude from tail-mass computation.
                    continue
                sigma_i = float(sigma_arr[i])
                if sigma_i < 1e-10:
                    # Dead dim — already surfaced by the dead_dim
                    # auditor; including it would divide-by-zero.
                    continue
                mu_i = float(mu_arr[i])
                p99 = float(stats.get("p99", 0))
                if not np.isfinite(p99) or p99 <= mu_i:
                    continue
                z_p99 = (p99 - mu_i) / sigma_i
                mass_i = z_p99 ** 2
                survivors.append((i, raw_name, mass_i))

            if not survivors:
                return None
            total_mass = sum(m for _, _, m in survivors)
            if total_mass < 1e-12:
                return None

            shares = [(i, label, m / total_mass) for i, label, m in survivors]
            dom_idx, dom_label, dom_share = max(shares, key=lambda t: t[2])
            threshold = _DOMINANT_DIM_MASS_SHARE_THRESHOLD
            if dom_share < threshold:
                return None

            return {
                "type": "dominant_dim_mass",
                "dim_label": dom_label,
                "reason": (
                    f"p99-tail mass share = {dom_share:.4f} (dim drives "
                    f">{int(threshold * 100)}% of population tail variance "
                    f"across {len(survivors)} dims with percentile coverage)"
                ),
                "advice": (
                    "Top-N anomaly ranking is likely single-dim-driven on "
                    "this dim. Cross-check by inspecting "
                    "reliability_flags.single_dim_driven incidence among "
                    "find_anomalies results. If high, the sphere is "
                    "effectively a one-dim detector; rebalance by splitting "
                    "the dim into a saturating component + a tail "
                    "component, or adding correlated dims that capture "
                    "independent signal."
                ),
                "evidence_value": round(dom_share, 4),
                "threshold": threshold,
            }

        def _compute_negative_space_warnings(
            pattern: Any,
        ) -> list[dict[str, Any]]:
            """Detect gaussian-kind dims whose empirical median is zero.

            The gaussian z-score on such a dim sits at the mode of the
            distribution rather than the tail, so every anomaly score on
            this dim is geometry noise. Returns an empty list when the
            pattern lacks dimension_kinds (legacy build), when no dim is
            both gaussian and zero-median, or when percentile coverage
            is absent for the gaussian dims.
            """
            dim_labels = pattern.dim_labels if pattern.dim_labels else []
            dimension_kinds = getattr(pattern, "dimension_kinds", None)
            if dimension_kinds is None or len(dimension_kinds) != len(dim_labels):
                return []

            dim_percentiles = getattr(pattern, "dim_percentiles", None) or {}
            if not dim_percentiles:
                return []
            raw_to_idx = _build_raw_dim_name_to_index(pattern)
            out: list[dict[str, Any]] = []
            for raw_name, stats in dim_percentiles.items():
                i = raw_to_idx.get(raw_name)
                if i is None or i >= len(dimension_kinds):
                    continue
                kind = dimension_kinds[i]
                if kind != "gaussian":
                    continue
                label = raw_name
                p50 = float(stats.get("p50", 0))
                if p50 != 0:
                    continue
                p99 = float(stats.get("p99", 0))
                p75 = float(stats.get("p75", 0))
                if p99 == 0:
                    fraction_zero_estimate = "all_zero"
                elif p75 == 0:
                    fraction_zero_estimate = ">=0.75"
                else:
                    fraction_zero_estimate = "0.50-0.75"
                out.append({
                    "type": "negative_space",
                    "dim_label": label,
                    "reason": (
                        f"p50=0 with declared gaussian kind (empirical "
                        f"distribution is point-mass-at-zero, approx "
                        f"fraction zero {fraction_zero_estimate}; gaussian "
                        f"z-score is wrong for this dim — it sits at the "
                        f"mode of the distribution, not the tail)"
                    ),
                    "advice": (
                        "Re-declare this dim with kind='bernoulli' "
                        "(presence/absence) or kind='poisson' (count), or "
                        "split into a binary 'is_active' dim + a "
                        "continuous 'value_when_active' dim. As long as "
                        "it's gaussian, every anomaly score on this dim "
                        "is geometry noise."
                    ),
                })
            return out

        warnings: list[dict[str, Any]] = []
        labels = pattern.dim_labels if pattern.dim_labels else []

        # Dead dims — indexed against dim_labels via sigma_diag position.
        # Threshold chosen well below float32 noise floor (~1.19e-7) so
        # sigma values that survive at 1e-10 or above are genuine
        # population spread, not numerical noise. Strict `<` (not `<=`)
        # is intentional: 1e-10 itself is on the boundary and we
        # prefer one false negative over one false positive on a
        # numerically-defined dim.
        sigma_diag = getattr(pattern, "sigma_diag", None)
        if sigma_diag is not None:
            sigma_arr = np.asarray(sigma_diag, dtype=np.float64)
            for i, s in enumerate(sigma_arr):
                if s < 1e-10:
                    label = labels[i] if i < len(labels) else f"dim_{i}"
                    warnings.append({
                        "type": "dead_dim",
                        "dim_label": label,
                        "reason": (
                            f"sigma_diag = {float(s):.2e} (below 1e-10 — "
                            f"zero variance across population)"
                        ),
                        "advice": (
                            "z-score is undefined / explodes; this dim "
                            "contributes nothing meaningful to delta_norm "
                            "and silently dilutes other dims' signal. "
                            "Drop the dim from the pattern, or investigate "
                            "the data source for missing values / "
                            "constant column."
                        ),
                    })

        # Sparse dims — keyed by dim name, not delta-vector index.
        # `dim_percentiles` holds {dim_name: {p25, p50, p75, p99, max}}.
        dim_percentiles = getattr(pattern, "dim_percentiles", None) or {}
        for dim_name, stats in dim_percentiles.items():
            p50 = stats.get("p50", 0)
            p99 = stats.get("p99", 0)
            if p50 == 0 and p99 > 0:
                p25 = stats.get("p25", 0)
                p75 = stats.get("p75", 0)
                if p75 == 0:
                    fraction_zero_estimate = ">=0.75"
                elif p25 == 0:
                    fraction_zero_estimate = "0.50-0.75"
                else:
                    fraction_zero_estimate = "0.25-0.50"
                warnings.append({
                    "type": "sparse_dim",
                    "dim_label": dim_name,
                    "reason": (
                        f"median = 0 with p99 = {p99} "
                        f"(mostly-zero with rare nonzero; approx "
                        f"fraction zero {fraction_zero_estimate})"
                    ),
                    "advice": (
                        "Gaussian z-score assumption is wrong for this "
                        "dim. Use Bregman divergence with a "
                        "poisson / bernoulli kind tag instead, or split "
                        "into a binary 'is_active' dim + a continuous "
                        "'value_when_active' dim."
                    ),
                })

        # Pattern-level audits. Helpers return None / [] for legacy
        # spheres missing mu / sigma_diag / dimension_kinds / dim_percentiles —
        # no outer try/except so a genuine bug surfaces instead of
        # silently dropping the warning (the keying-mismatch bug in the
        # first cut of this code was invisible for an hour exactly
        # because a blanket except swallowed the KeyError).
        dom_warning = _compute_dominant_dim_mass_warning(pattern)
        if dom_warning is not None:
            warnings.append(dom_warning)

        negative_space_warnings = _compute_negative_space_warnings(pattern)
        warnings.extend(negative_space_warnings)

        # Per-dim normality test outcome. Gaussian-only (z-score assumes
        # normality, so the test is meaningless on bernoulli / poisson
        # kinds). Suppress for dims already flagged by `negative_space`
        # because that warning's remediation ("re-declare kind") is the
        # right action and the normality complaint is downstream noise.
        negative_space_labels = {
            w["dim_label"] for w in negative_space_warnings
        }
        warnings.extend(
            GDSNavigator._compute_non_normal_dim_warnings(
                pattern, suppress_labels=negative_space_labels,
            ),
        )

        # Kind-tag mismatch — gaussian dims with near-zero Fisher LDA
        # direction component but raw class separation, signalling that
        # the dim's variance is captured by another dim's axis. Skipped
        # silently when ``label_aware_calibration`` is None (pattern
        # built without the ``label_audit:`` block). Suppressed by INDEX
        # for dims already covered by ``negative_space`` — the two
        # auditors key their dim_labels off slightly different naming
        # schemes (negative_space uses the raw ``dim_percentiles`` key,
        # ``label_aware_calibration`` uses the storage-layout
        # ``dim_labels`` entry), so index-based dedup is the only
        # reliable join.
        warnings.extend(
            GDSNavigator._compute_kind_mismatch_warnings(
                pattern,
                negative_space_indices=(
                    GDSNavigator._negative_space_indices(
                        pattern, negative_space_warnings,
                    )
                ),
            ),
        )

        # Signed-tail concentration — pattern-level warning fired when
        # the Fisher LDA projection of polygon deltas
        # (``Pattern.signed_percentiles`` populated at build time from
        # the ``delta_norm_signed`` Lance column) shows a one-sided
        # extreme tail: ``|p99| / max(|p50|, 1e-9) > 50``. Indicates the
        # label-discriminating axis is being driven by a tiny outlier
        # subgroup rather than the broad positive / negative class
        # split. Skipped when ``signed_percentiles`` is absent (pattern
        # built without label-aware calibration) or when the positive
        # class count is below ``_SIGNED_TAIL_MIN_N_POS`` — small-N LDA
        # is unstable on its own and the warning would echo noise.
        signed_tail = GDSNavigator._compute_signed_tail_concentration_warning(
            pattern,
        )
        if signed_tail is not None:
            warnings.append(signed_tail)

        # Heteroscedasticity — Brown-Forsythe p-value persisted by the
        # builder when the pattern carries `group_by_property`. The
        # dim_label here references the grouping variable (a categorical
        # line property), not a δ-dim — the warning's role is to flag
        # that the global θ / Cohen's d pooled-σ / APS global-percentile
        # assumptions are violated for this pattern, not that any one
        # δ-dim is misbehaving.
        het_diag = getattr(pattern, "heteroscedasticity_diagnostic", None) or {}
        for prop_name, entry in het_diag.items():
            p_value = entry.get("p_value")
            # None: < 2 qualifying groups (low-N skip exhausted the test).
            # NaN: every qualifying group had zero residual variance
            # (degenerate distribution — typical of event-line shapes
            # with one in-relation, but also a bug-screen surface). In
            # neither case is the global-θ assumption "violated" — the
            # diagnostic simply could not be computed.
            if p_value is None or not np.isfinite(p_value):
                continue
            if p_value >= _HETEROSCEDASTICITY_P_THRESHOLD:
                continue
            w_stat = entry.get("W_statistic")
            warnings.append({
                "type": "heteroscedasticity",
                "dim_label": prop_name,
                "reason": (
                    f"Levene W={float(w_stat):.2f} p={float(p_value):.2e} "
                    f"on group_by={prop_name} — global θ assumption "
                    f"violated"
                ),
                "advice": (
                    "Per-group θ calibration is statistically warranted "
                    "on this grouping; if per-group θ is undesirable "
                    "downstream consider a variance-stabilizing "
                    "transform (log1p) on delta_norm before "
                    "thresholding."
                ),
                "evidence_value": float(p_value),
                "threshold": _HETEROSCEDASTICITY_P_THRESHOLD,
            })

        return warnings

    @staticmethod
    def _compute_non_normal_dim_warnings(
        pattern: Any,
        *,
        suppress_labels: set[str],
    ) -> list[dict[str, Any]]:
        """Emit ``non_normal_dim`` warnings for gaussian dims that failed
        the build-time normality test (Shapiro-Wilk for N <= 5000, KS
        otherwise). Returns an empty list when the pattern lacks
        ``dim_normality_pvalues`` (pre-diagnostic build), when no
        gaussian dim has a p-value below the alpha threshold, or when
        every below-threshold dim is already covered by a
        ``negative_space`` warning.
        """
        pvalues = getattr(pattern, "dim_normality_pvalues", None)
        if not pvalues:
            return []

        dimension_kinds = getattr(pattern, "dimension_kinds", None)
        dim_labels = pattern.dim_labels if pattern.dim_labels else []
        if dimension_kinds is None or len(dimension_kinds) != len(dim_labels):
            # Legacy / mis-shaped pattern — no reliable way to confirm
            # the dim is gaussian, so do not emit.
            return []

        # Reuse the navigator's keying convention: raw column / line_id
        # without `_d_` prefix / prop name → delta-vector index. Same
        # build as the `negative_space` auditor so a dim's normality
        # entry resolves to the same kind slot.
        def _raw_to_idx(pattern: Any) -> dict[str, int]:
            mapping: dict[str, int] = {}
            for i, rel in enumerate(pattern.relations):
                if rel.line_id.startswith("_d_"):
                    stripped = rel.line_id[3:]
                    mapping[stripped] = i
                    parts = stripped.split("_", 1)
                    if len(parts) > 1:
                        mapping.setdefault(parts[1], i)
                else:
                    mapping[rel.line_id] = i
            k = len(pattern.relations)
            for j, ed in enumerate(pattern.event_dimensions):
                mapping[ed.column] = k + j
            k2 = k + len(pattern.event_dimensions)
            for j, prop in enumerate(pattern.prop_columns):
                mapping[prop] = k2 + j
            return mapping

        raw_to_idx = _raw_to_idx(pattern)
        out: list[dict[str, Any]] = []
        for raw_name, p_value in pvalues.items():
            if raw_name in suppress_labels:
                continue
            i = raw_to_idx.get(raw_name)
            if i is None or i >= len(dimension_kinds):
                continue
            if dimension_kinds[i] != "gaussian":
                continue
            if not np.isfinite(p_value):
                continue
            if p_value >= _NON_NORMAL_DIM_PVALUE_THRESHOLD:
                continue
            out.append({
                "type": "non_normal_dim",
                "dim_label": raw_name,
                "reason": (
                    f"normality test p={p_value:.2e} < "
                    f"{_NON_NORMAL_DIM_PVALUE_THRESHOLD} — z-score "
                    f"assumes normality, this dim's distribution is "
                    f"heavy-tailed or otherwise non-normal"
                ),
                "advice": (
                    "Consider a log1p / sqrt / rank transform of the "
                    "raw shape values before mu/sigma computation, or "
                    "re-declare kind as 'poisson' (discrete counts) or "
                    "'bernoulli' (binary presence) if applicable. "
                    "Untransformed, the gaussian z-score concentrates "
                    "the delta mass in a few extreme rows and poorly "
                    "discriminates the bulk of the population."
                ),
                "evidence_value": float(p_value),
                "threshold": _NON_NORMAL_DIM_PVALUE_THRESHOLD,
            })
        return out

    @staticmethod
    def _negative_space_indices(
        pattern: Any,
        negative_space_warnings: list[dict[str, Any]],
    ) -> set[int]:
        """Translate ``negative_space`` warning labels to delta-vector
        indices.

        ``negative_space`` keys its ``dim_label`` off the
        ``dim_percentiles`` raw column name; the kind_mismatch auditor
        keys off the storage-layout ``dim_labels`` entry. The two names
        can differ (e.g. relations whose ``line_id`` is ``_d_foo`` show
        up as ``foo`` in the percentile cache but as ``_d_foo`` in
        ``dim_labels``). Resolving via the shared raw-to-index mapping
        gives a single source of truth for suppression.
        """
        if not negative_space_warnings:
            return set()
        # Inline the mapping logic — the outer helper
        # ``_build_raw_dim_name_to_index`` is scoped inside
        # ``_compute_dim_quality_warnings`` and not reachable here.
        mapping: dict[str, int] = {}
        for i, rel in enumerate(pattern.relations):
            if rel.line_id.startswith("_d_"):
                stripped = rel.line_id[3:]
                mapping[stripped] = i
                parts = stripped.split("_", 1)
                if len(parts) > 1:
                    mapping.setdefault(parts[1], i)
            else:
                mapping[rel.line_id] = i
        k = len(pattern.relations)
        for j, ed in enumerate(pattern.event_dimensions):
            mapping[ed.column] = k + j
        k2 = k + len(pattern.event_dimensions)
        for j, prop in enumerate(pattern.prop_columns):
            mapping[prop] = k2 + j

        out: set[int] = set()
        for w in negative_space_warnings:
            label = w.get("dim_label")
            if label is None:
                continue
            idx = mapping.get(label)
            if idx is not None:
                out.add(idx)
        return out

    @staticmethod
    def _compute_kind_mismatch_warnings(
        pattern: Any,
        *,
        negative_space_indices: set[int],
    ) -> list[dict[str, Any]]:
        """Emit ``kind_mismatch`` warnings for gaussian-declared dims
        whose Fisher LDA direction component is near zero AND whose raw
        per-class moments still separate.

        Returns an empty list when the pattern lacks
        ``label_aware_calibration`` (no Fisher direction to test
        against), when ``dimension_kinds`` is missing or mis-shaped, or
        when no dim crosses both gates.
        """
        lac = getattr(pattern, "label_aware_calibration", None)
        if not lac:
            return []

        dimension_kinds = getattr(pattern, "dimension_kinds", None)
        dim_labels = pattern.dim_labels if pattern.dim_labels else []
        if dimension_kinds is None or len(dimension_kinds) != len(dim_labels):
            return []

        out: list[dict[str, Any]] = []
        for i, label in enumerate(dim_labels):
            if i in negative_space_indices:
                continue
            if i >= len(dimension_kinds):
                continue
            if dimension_kinds[i] != "gaussian":
                continue
            dim_cal = lac.get(label) if isinstance(lac, dict) else None
            if dim_cal is None:
                continue
            mu_pos = float(getattr(dim_cal, "mu_pos", 0.0))
            sigma_pos = float(getattr(dim_cal, "sigma_pos", 0.0))
            mu_neg = float(getattr(dim_cal, "mu_neg", 0.0))
            sigma_neg = float(getattr(dim_cal, "sigma_neg", 0.0))
            direction = float(getattr(dim_cal, "direction", 0.0))

            if not np.isfinite(direction):
                continue
            denom_sq = (sigma_pos * sigma_pos + sigma_neg * sigma_neg) / 2.0
            if denom_sq <= 0.0:
                cohens_d = 0.0
            else:
                cohens_d = abs(mu_pos - mu_neg) / float(np.sqrt(denom_sq))
            if not np.isfinite(cohens_d):
                continue

            abs_dir = abs(direction)
            if abs_dir >= _KIND_MISMATCH_DIRECTION_THRESHOLD:
                continue
            if cohens_d < _KIND_MISMATCH_COHENS_D_THRESHOLD:
                continue

            out.append({
                "type": "kind_mismatch",
                "dim_label": label,
                "reason": (
                    f"kind=gaussian declared but Fisher direction shows "
                    f"|direction|={abs_dir:.3f} < "
                    f"{_KIND_MISMATCH_DIRECTION_THRESHOLD} while raw "
                    f"classes show cohens_d={cohens_d:.3f} >= "
                    f"{_KIND_MISMATCH_COHENS_D_THRESHOLD} — likely "
                    f"confounded with another dim"
                ),
                "advice": (
                    "consider kind='bernoulli' / 'poisson' "
                    "re-declaration, or split the dim into binary + "
                    "continuous components"
                ),
                "evidence_value": abs_dir,
                "threshold": _KIND_MISMATCH_DIRECTION_THRESHOLD,
            })
        return out

    @staticmethod
    def _compute_signed_tail_concentration_warning(
        pattern: Any,
    ) -> dict[str, Any] | None:
        """Pattern-level warning when ``signed_percentiles`` p99 / p50
        ratio crosses ``_SIGNED_TAIL_RATIO_THRESHOLD``.

        Returns ``None`` when:

        - ``pattern.signed_percentiles`` is absent (no label-aware
          calibration available — nothing to test),
        - ``pattern.label_aware_n_pos`` is below
          ``_SIGNED_TAIL_MIN_N_POS`` (positive class undersampled —
          LDA fit unstable, the ratio echoes noise rather than signal),
        - p99 / p50 ratio does not cross the threshold (no concentration).

        The ``dim_label`` is set to ``<pattern_id>:signed_percentiles``
        to mark the warning as pattern-level rather than dim-keyed; the
        ``evidence_value`` carries the ratio for downstream inspection.
        """
        sp = getattr(pattern, "signed_percentiles", None)
        if not sp:
            return None
        n_pos = getattr(pattern, "label_aware_n_pos", None)
        if n_pos is None or int(n_pos) < _SIGNED_TAIL_MIN_N_POS:
            return None

        p50 = float(sp.get("p50", 0.0))
        p99 = float(sp.get("p99", 0.0))
        denom = max(abs(p50), 1e-9)
        ratio = abs(p99) / denom
        if not np.isfinite(ratio) or ratio <= _SIGNED_TAIL_RATIO_THRESHOLD:
            return None

        return {
            "type": "signed_tail_concentration",
            "dim_label": f"{pattern.pattern_id}:signed_percentiles",
            "reason": (
                f"|p99|/|p50| = {ratio:.1f} > "
                f"{_SIGNED_TAIL_RATIO_THRESHOLD:.0f} on the Fisher LDA "
                f"projection (n_pos={int(n_pos)}) — signed-delta tail "
                f"one-sided extreme, likely driven by a tiny outlier "
                f"subgroup rather than the broad positive / negative "
                f"class split"
            ),
            "advice": (
                "re-calibrate with stratified sampling on the label "
                "column, or collect a larger positive class — the "
                "current LDA direction is dominated by a few outlier "
                "rows and may not generalise"
            ),
            "evidence_value": round(ratio, 4),
            "threshold": _SIGNED_TAIL_RATIO_THRESHOLD,
        }

    def _build_theta_sensitivity_summary(self, pattern_id: str) -> dict | None:
        """Compact theta_sensitivity diagnostic for sphere_overview.

        Returns None when the calibration epoch lacks the diagnostic field
        (pre-T2 sphere) or when no calibration epochs exist on disk.
        Otherwise returns a small dict with stable_band shape, cliff count,
        and the production p95 theta value for at-a-glance inspection.

        Cost: one JSON read of calibration_history v=latest (~few KB) plus
        an O(P=10) derivation. Sub-millisecond per pattern.
        """
        from hypertopos.builder._theta_sensitivity import (
            derive_stable_band_and_cliffs,
        )
        from hypertopos.storage.calibration_history import (
            CalibrationNotFoundError,
        )

        try:
            versions = self._storage.list_calibration_versions(pattern_id)
            if not versions:
                return None
            fit = self._storage.read_calibration_fit(
                pattern_id, version=versions[-1],
            )
        except (CalibrationNotFoundError, FileNotFoundError):
            # Race between list_versions and read_fit (GC trimmed the
            # version between the two calls), or the file vanished.
            # Other exceptions propagate so reviewers see them.
            return None

        if fit.theta_sensitivity is None:
            return None

        derived = derive_stable_band_and_cliffs(fit.theta_sensitivity)
        p95 = fit.theta_sensitivity.get("p95")
        return {
            "stable_band_from": derived["stable_band"]["from"],
            "stable_band_to": derived["stable_band"]["to"],
            "stable_band_length": derived["stable_band_length"],
            "n_cliffs": derived["n_cliffs"],
            "theta_at_p95": (
                round(float(p95["theta_mean"]), 4) if p95 else None
            ),
        }

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

    def _compute_cross_pattern_discrepancy(self) -> dict | None:
        """Pairwise Jaccard overlap of anomalous primary_keys across patterns
        that cover the same entity line.

        Two patterns are "cover-overlapping" when ``sphere.entity_line(pat_id)``
        returns the same non-null line_id — their geometry tables index the
        same primary_key space, so the anomalous-key sets are comparable.

        For each unordered pair (A, B), reads ``primary_key`` + ``is_anomaly``
        from both pattern geometry tables (two cheap columns, Lance bitmap
        index on ``is_anomaly``), takes the set difference / intersection /
        union on the union of all keys observed across both tables, and
        emits the four bucket counts plus ``jaccard_anomaly_overlap``.

        Returns ``None`` when fewer than two patterns share an entity line.
        Output schema::

            {
              "pairs": [
                {
                  "pattern_a": str,
                  "pattern_b": str,
                  "shared_line": str,
                  "n_anomalous_only_in_a": int,
                  "n_anomalous_only_in_b": int,
                  "n_anomalous_in_both": int,
                  "n_anomalous_in_neither": int,
                  "jaccard_anomaly_overlap": float | None,
                }
              ]
            }

        ``jaccard_anomaly_overlap`` is ``None`` when both anomaly sets are
        empty (``|A ∪ B| == 0``).

        Threshold caveat: each pattern's ``is_anomaly`` column was materialised
        with that pattern's own ``theta``. Comparing two patterns therefore
        compares two thresholds on the same key space — Jaccard near zero is
        the expected null case (different detectors with different calibration
        rarely agree on the exact anomalous subset), and large overlap is the
        notable signal.
        """
        sphere = self._storage.read_sphere()

        # Group patterns by their entity_line. Pairs only form within a group.
        groups: dict[str, list[str]] = defaultdict(list)
        for pid in sphere.patterns:
            line_id = sphere.entity_line(pid)
            if not line_id:
                continue
            groups[line_id].append(pid)

        # Drop groups with fewer than two patterns.
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if not groups:
            return None

        import pyarrow.compute as pc

        # Cache anomalous-key sets per (pid, version) to avoid re-reads.
        anom_cache: dict[str, set[str]] = {}

        def _anom_keys(pid: str) -> set[str]:
            if pid in anom_cache:
                return anom_cache[pid]
            try:
                version = self._resolve_version(pid)
                tbl = self._storage.read_geometry(
                    pid, version, columns=["primary_key", "is_anomaly"],
                )
            except _NAVIGATION_RECOVERABLE_ERRORS:
                anom_cache[pid] = set()
                return anom_cache[pid]
            mask = pc.equal(tbl["is_anomaly"], True)
            keys = tbl.filter(mask)["primary_key"].to_pylist()
            anom_cache[pid] = set(keys)
            return anom_cache[pid]

        # Cache full-key (universe) sets per pid to compute n_anomalous_in_neither.
        universe_cache: dict[str, set[str]] = {}

        def _universe(pid: str) -> set[str]:
            if pid in universe_cache:
                return universe_cache[pid]
            try:
                version = self._resolve_version(pid)
                tbl = self._storage.read_geometry(
                    pid, version, columns=["primary_key"],
                )
            except _NAVIGATION_RECOVERABLE_ERRORS:
                universe_cache[pid] = set()
                return universe_cache[pid]
            universe_cache[pid] = set(tbl["primary_key"].to_pylist())
            return universe_cache[pid]

        pairs: list[dict[str, Any]] = []
        for shared_line, pids in groups.items():
            pids_sorted = sorted(pids)
            for i, pat_a in enumerate(pids_sorted):
                for pat_b in pids_sorted[i + 1:]:
                    anom_a = _anom_keys(pat_a)
                    anom_b = _anom_keys(pat_b)
                    only_a = anom_a - anom_b
                    only_b = anom_b - anom_a
                    both = anom_a & anom_b
                    union_universe = _universe(pat_a) | _universe(pat_b)
                    neither = union_universe - anom_a - anom_b
                    union_anom_size = len(anom_a | anom_b)
                    jaccard: float | None
                    if union_anom_size == 0:
                        jaccard = None
                    else:
                        jaccard = round(len(both) / union_anom_size, 4)
                    pairs.append({
                        "pattern_a": pat_a,
                        "pattern_b": pat_b,
                        "shared_line": shared_line,
                        "n_anomalous_only_in_a": len(only_a),
                        "n_anomalous_only_in_b": len(only_b),
                        "n_anomalous_in_both": len(both),
                        "n_anomalous_in_neither": len(neither),
                        "jaccard_anomaly_overlap": jaccard,
                    })

        return {"pairs": pairs}

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

        total = self._storage.count_geometry_rows(pattern_id)
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
            pattern_id, filter="delta_norm < 0.0001"
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
            pattern_id,
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
                pattern_id, filter=f"delta_norm >= {ceiling_threshold:.6f}"
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

    def find_conformance_violations(
        self,
        pattern_id: str,
        *,
        rule_id: str | None = None,
        severity_min: Literal["low", "medium", "high", "critical"] = "low",
        top_n: int = 100,
    ) -> dict[str, Any]:
        """Return entities violating one or more declared conformance rules.

        Reads the build-time sidecar Lance dataset under
        ``_gds_meta/conformance/violations/{pattern_id}/v={N}.lance`` with
        Lance filter pushdown on ``rule_id`` and post-scan filtering on
        ``severity_min``. The sidecar is written by the builder when a
        pattern declares ``conformance_rules``; otherwise this method
        returns an empty result with ``manifest=None``.

        Detects a rule-set hash mismatch between the sidecar manifest and
        the pattern's currently declared rules — surfaced as a warning,
        never raised, because the navigator is read-only and the builder
        is the authoritative re-evaluator.
        """
        from hypertopos.engine.conformance import read_violations
        from hypertopos.model.sphere import compute_rule_set_hash

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere",
            )
        pattern = sphere.patterns[pattern_id]
        version = self._resolve_version(pattern_id)
        base_path = self._storage._base

        violations, manifest = read_violations(
            base_path=base_path,
            pattern_id=pattern_id,
            version=version,
            rule_id=rule_id,
            severity_min=severity_min,
            top_n=top_n,
        )

        warnings: list[str] = []
        if manifest is not None:
            current_hash = compute_rule_set_hash(pattern.conformance_rules)
            if current_hash != manifest.get("rule_set_hash"):
                warnings.append(
                    "rule_set_hash_mismatch: sidecar was built against a "
                    "different ruleset; rebuild the sphere to re-evaluate",
                )

        return {
            "pattern_id": pattern_id,
            "n_violations": len(violations),
            "violations": violations,
            "rules_evaluated": [
                r.rule_id for r in pattern.conformance_rules
            ],
            "manifest": manifest,
            "warnings": warnings,
            "follow_up": (
                "Use investigate_entity(primary_key) on top violators to "
                "drill into the geometric anomaly that may accompany the "
                "compliance rule break."
                if violations else None
            ),
        }

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

        shifts: list[tuple[datetime, float, np.ndarray, int]] = []
        for i in range(1, len(filtered)):
            c_prev = np.array(filtered[i - 1]["centroid"][:n_rel], dtype=np.float64)
            c_curr = np.array(filtered[i]["centroid"][:n_rel], dtype=np.float64)
            diff = c_curr - c_prev
            mag = float(np.linalg.norm(diff))
            shifts.append((filtered[i]["window_start"], mag, diff, i))

        if not shifts:
            return []

        mags = [s[1] for s in shifts]
        mean_m = float(np.mean(mags))
        std_m = float(np.std(mags)) if len(mags) > 2 else mean_m * 0.5
        threshold = mean_m + 1.5 * std_m

        n_total = len(filtered)
        boundary_idx = {0, 1, n_total - 1, n_total - 2}

        changepoints: list[dict] = []
        for ts, mag, diff, idx in shifts:
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
                "near_data_boundary": idx in boundary_idx,
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

        n_buckets = len(bucket_timestamps)
        boundary_idx = {0, 1, n_buckets - 1, n_buckets - 2}

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
                    "near_data_boundary": (i + 1) in boundary_idx,
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
        total = self._storage.count_geometry_rows(pattern_id)
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

    def audit_label_alignment(
        self, pattern_id: str, *, top_n: int = 10
    ) -> dict[str, Any]:
        """Report Fisher LDA direction alignment as an AUROC against labels.

        Sibling to the MCP-only ``audit_pattern_dims`` calibration audit:
        ``audit_pattern_dims`` describes the per-dim moments and the Fisher
        axis components; ``audit_label_alignment`` answers the orthogonal
        question — "does the projection of polygons onto that axis actually
        separate the two labelled classes?" — by computing AUROC of
        ``delta_norm_signed`` against the binary label column declared in
        ``sphere.yaml``'s ``label_audit:`` block.

        Returns ``{pattern_id, auroc, n_pos, n_neg, top_dims,
        label_aware_available, elapsed_ms}`` on the full-field path. The
        ``top_dims`` list carries the ``top_n`` most label-discriminating
        dims, sorted by ``|direction_component|`` desc, each row carrying
        ``{dim_label, direction_component, abs_direction, cohens_d_pos_neg}``.

        Returns a fallback shape with ``auroc: null`` /
        ``label_aware_available: False`` / a top-level ``reason`` when:

        - the pattern was built without a ``label_audit:`` block (no
          ``label_aware_calibration`` on the Pattern), or
        - the sphere has no top-level ``label_audit`` block (no labels to
          join against), or
        - the joined sample has zero positives or zero negatives
          (degenerate label distribution).
        """
        t0 = datetime.now(UTC)
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise KeyError(f"unknown pattern_id '{pattern_id}'")

        def _elapsed_ms() -> float:
            return (datetime.now(UTC) - t0).total_seconds() * 1000.0

        def _fallback(reason: str) -> dict[str, Any]:
            return {
                "pattern_id": pattern_id,
                "auroc": None,
                "n_pos": None,
                "n_neg": None,
                "top_dims": [],
                "label_aware_available": False,
                "reason": reason,
                "elapsed_ms": _elapsed_ms(),
            }

        lac = getattr(pattern, "label_aware_calibration", None)
        if not lac:
            return _fallback(
                "no label-aware calibration available — pattern was built "
                "without the 'label_audit:' block in sphere.yaml"
            )

        label_audit = getattr(sphere, "label_audit", None)
        if not isinstance(label_audit, dict):
            return _fallback(
                "sphere has no top-level label_audit block — labels "
                "unavailable for AUROC computation"
            )
        label_column = label_audit.get("label_column")
        label_positive_value = label_audit.get("label_positive_value")
        if not label_column or label_positive_value is None:
            return _fallback(
                "label_audit block missing label_column / "
                "label_positive_value"
            )

        entity_line_id = pattern.entity_line_id
        if not entity_line_id:
            return _fallback(
                "pattern has no entity_line — cannot resolve label column"
            )

        version = self._manifest.pattern_version(pattern_id)
        if version is None:
            return _fallback(
                f"pattern '{pattern_id}' has no active version in manifest"
            )

        # Read signed projection — keep nulls so we can drop them with the
        # label-side join (patterns without label-aware calibration emit
        # all-null even when the column exists).
        geo = self._storage.read_geometry(
            pattern_id, version, columns=["primary_key", "delta_norm_signed"],
        )
        if "delta_norm_signed" not in geo.schema.names:
            return _fallback(
                "geometry dataset has no 'delta_norm_signed' column — "
                "sphere predates label-aware calibration support"
            )

        line_version = self._manifest.line_version(entity_line_id) or 1
        try:
            points = self._storage.read_points(
                entity_line_id, line_version,
                columns=["primary_key", label_column],
            )
        except (KeyError, FileNotFoundError, OSError, pa.ArrowInvalid) as exc:
            return _fallback(
                f"failed to read label column '{label_column}' from line "
                f"'{entity_line_id}': {exc!r}"
            )
        if label_column not in points.schema.names:
            return _fallback(
                f"label column '{label_column}' not found on entity line "
                f"'{entity_line_id}'"
            )

        # Join geometry signed projection ←→ labels on primary_key.
        joined = geo.join(
            points, keys="primary_key", join_type="inner",
        )
        if joined.num_rows == 0:
            return _fallback(
                "no rows after joining geometry to labels on primary_key"
            )

        signed_arr = joined["delta_norm_signed"]
        label_arr = joined[label_column]
        # Drop rows where either side is null (all-null signed → AUROC
        # cannot be computed; null labels → no class to score against).
        mask = pc.and_(
            pc.is_valid(signed_arr), pc.is_valid(label_arr),
        )
        signed_arr = signed_arr.filter(mask)
        label_arr = label_arr.filter(mask)
        if len(signed_arr) == 0:
            return _fallback(
                "all delta_norm_signed values are null — pattern was not "
                "calibrated label-aware (column populated requires the "
                "build flag and the label_audit block)"
            )

        signed_np = signed_arr.to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False,
        )
        # Binarize against positive value via element-wise equality.
        labels_list = label_arr.to_pylist()
        labels_np = np.asarray(
            [1 if v == label_positive_value else 0 for v in labels_list],
            dtype=np.int32,
        )
        n_pos = int(labels_np.sum())
        n_neg = int(len(labels_np) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return _fallback(
                f"degenerate label distribution: n_pos={n_pos}, "
                f"n_neg={n_neg} after joining on primary_key"
            )

        auroc = _auroc_rank_sum(signed_np, labels_np)

        # Top-N dims by |direction_component| desc; carry Cohen's d so
        # the report stands alone without a second audit_pattern_dims call.
        dim_labels = pattern.dim_labels
        top_rows: list[dict[str, Any]] = []
        for label in dim_labels:
            dim_cal = lac.get(label) if isinstance(lac, dict) else None
            if dim_cal is None:
                continue
            direction = float(dim_cal.direction)
            mu_pos = float(dim_cal.mu_pos)
            sigma_pos = float(dim_cal.sigma_pos)
            mu_neg = float(dim_cal.mu_neg)
            sigma_neg = float(dim_cal.sigma_neg)
            denom_sq = (sigma_pos * sigma_pos + sigma_neg * sigma_neg) / 2.0
            cohens_d = (
                abs(mu_pos - mu_neg) / math.sqrt(denom_sq)
                if denom_sq > 0.0
                else 0.0
            )
            top_rows.append({
                "dim_label": label,
                "direction_component": direction,
                "abs_direction": abs(direction),
                "cohens_d_pos_neg": cohens_d,
            })
        top_rows.sort(key=lambda r: r["abs_direction"], reverse=True)
        top_rows = top_rows[:top_n]

        return {
            "pattern_id": pattern_id,
            "auroc": auroc,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "top_dims": top_rows,
            "label_aware_available": True,
            "elapsed_ms": _elapsed_ms(),
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
        *,
        include_reliability_flags: bool = True,
    ) -> dict:
        """Compose anomaly p-values across patterns via Wilson harmonic-mean p.

        Uses conformal_p from each pattern's cross_pattern_profile signal.
        Replaces the prior Fisher combination — HMP is robust to positive
        dependence between detectors (Wilson 2019), which is the AML regime
        where multiple detectors fire on the same entity.
        """
        from hypertopos.engine.composition import harmonic_mean_p

        profile = self.cross_pattern_profile(primary_key, line_id, max_related)
        p_values: dict[str, float] = {}
        per_pattern: dict[str, dict] = {}
        for pat_id, signal in profile.get("signals", {}).items():
            cp = signal.get("conformal_p")
            if cp is not None and cp > 0:
                p_values[pat_id] = float(cp)
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
        combined_p = harmonic_mean_p(p_values)
        result = {
            "primary_key": primary_key,
            "combined_p": round(float(combined_p), 6),
            "n_patterns": len(p_values),
            "per_pattern": per_pattern,
        }
        # Reliability flags for the entity's home (direct) pattern. One
        # scalar reliability_flags per composite_risk call so a high
        # combined_p carries the same caveat metadata as the find_anomalies
        # row would. Picks the first "direct" key_type pattern from the
        # cross-pattern profile signals — that's the anchor pattern of the
        # entity's home line (matches the "this entity IS the primary key
        # of pattern X" semantic). Skipped when no direct pattern exists.
        # Caller can pass include_reliability_flags=False to skip the
        # extra read_sphere + build_polygon — that's the path composite_risk_batch
        # takes by default to keep the bulk-loop cost bounded.
        home_pattern_id: str | None = None
        for pat_id, sig in profile.get("signals", {}).items():
            if sig.get("key_type") == "direct":
                home_pattern_id = pat_id
                break
        if include_reliability_flags and home_pattern_id:
            try:
                from hypertopos.engine.geometry import compute_reliability_flags
                sphere = self._storage.read_sphere()
                home_pat = sphere.patterns.get(home_pattern_id)
                home_poly = self._engine.build_polygon(
                    primary_key, home_pattern_id, self._manifest,
                )
                if home_pat is not None and home_poly is not None:
                    result["reliability_flags"] = compute_reliability_flags(
                        home_poly.delta,
                        pattern=home_pat,
                        anomaly_confidence=home_poly.anomaly_confidence,
                    )
            except _NAVIGATION_RECOVERABLE_ERRORS:
                pass
        return result

    def composite_risk_batch(
        self,
        primary_keys: list[str],
        line_id: str | None = None,
        max_keys: int = 200,
        *,
        include_reliability_flags: bool = False,
    ) -> dict:
        """Batch composite risk — harmonic-mean p-value across patterns.

        Returns per-key combined_p + summary counts at p<0.10 and p<0.05.
        Hard cap: max_keys (default 200). ``include_reliability_flags``
        defaults False on the batch path — every per-entity attachment
        triggers a read_sphere + build_polygon, so 200 entities is 200
        fresh polygon builds. Investigators who need per-entity reliability
        metadata in bulk should call ``composite_risk`` individually for
        the keys of interest.

        Per-pattern geometry reads are memoised across the batch so each
        pattern is read at most once instead of once per key.  The largest
        wins come from event-edge patterns (one batched ``point_keys``
        lookup over all keys replaces ``len(keys)`` separate scans, each
        of which pulls ~10^5 rows on AML-shaped graphs) and chain patterns
        (the full chain geometry is read once and filtered in-memory for
        every key).  The memo is local to this call — it does not leak
        into the navigator instance and never survives the batch.
        """
        keys = primary_keys[:max_keys]
        # Per-batch memo: caches the heavy reads inside the helper
        # branches of ``_profile_event_edge`` / ``_profile_chain`` so the
        # same Lance scan is not paid once per key.  Cleared at the end
        # of the batch — no cross-call state.  ``_batch_keys`` carries
        # the union of keys so helpers can issue a single batched read
        # on first hit instead of waiting for every key to arrive
        # individually.
        self._batch_profile_cache = {"_batch_keys": list(keys)}
        try:
            results = []
            for key in keys:
                try:
                    cr = self.composite_risk(
                        key, line_id,
                        include_reliability_flags=include_reliability_flags,
                    )
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
        finally:
            self._batch_profile_cache = None
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

    def combine_anomaly_pvalues(
        self,
        pattern_id: str,
        *,
        detectors: tuple[str, ...] = (
            "delta_norm",
            "neighbor_contamination",
            "segment_shift",
            "trajectory_continuous",
            "density_gap",
        ),
        weights: dict[str, float] | None = None,
        sample_size: int | None = 10_000,
        top_n: int = 50,
    ) -> list[dict]:
        """Multi-detector anomaly consensus via Wilson harmonic-mean p-value.

        Calibrates each named detector to a per-entity p-value, combines them
        across detectors via the harmonic-mean p (`engine/composition.py`),
        and returns the ranked list ascending by HMP. The default detector
        set spans the five orthogonal anomaly axes:

            * ``delta_norm`` — population-relative geometry deviation
            * ``neighbor_contamination`` — graph-neighbour anomaly density
            * ``segment_shift`` — categorical-segment anomaly rate shift
            * ``trajectory_continuous`` — DTW distance vs median trajectory
            * ``density_gap`` — local density-gap detector q-values

        Detectors that fail to produce data (e.g. no temporal solid for
        ``trajectory_continuous``, no string columns for ``segment_shift``,
        ``density_gap`` which is structurally aggregate and has no
        per-entity attribution) are silently skipped per entity — the HMP
        is then computed from the remaining detectors. ``delta_norm`` is
        the always-available primary path.

        Args:
            pattern_id: Pattern to score.
            detectors: Subset of detector names to include.
            weights: Per-detector weight; defaults to uniform across the
                detectors that produced a p-value for the given entity.
            sample_size: Cap on geometry rows used per detector
                (delta_norm always honors this cap).
            top_n: Maximum entries returned in the ranked list.

        Returns:
            List of ``{primary_key, hmp, p_per_detector, rank}`` ascending by
            ``hmp``. ``p_per_detector`` only contains detectors that produced
            a valid p-value for the entity.
        """
        from hypertopos.engine.composition import harmonic_mean_p
        from hypertopos.engine.p_value_calibration import (
            detector_p_value_delta_norm,
            detector_p_value_density_gap,
            detector_p_value_neighbor_contamination,
            detector_p_value_segment_shift,
            detector_p_value_trajectory_continuous,
        )

        version = self._resolve_version(pattern_id)
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere.",
            )

        # Pattern dimensionality is the chi2 fallback df for delta_norm.
        df = int(getattr(pattern, "mu", np.zeros(1)).shape[0])
        if df <= 0:
            df = max(len(getattr(pattern, "relations", []) or []), 1)

        # Read the geometry once (delta_norm + anomaly_confidence path).
        geo_cols = [
            "primary_key",
            "delta_norm",
            "is_anomaly",
            "anomaly_confidence",
        ]
        try:
            geo = self._storage.read_geometry(
                pattern_id, version,
                columns=geo_cols,
                sample_size=sample_size if sample_size and sample_size > 0 else None,
            )
        except TypeError:
            # Some mock storages do not accept sample_size — fall back.
            geo = self._storage.read_geometry(
                pattern_id, version, columns=geo_cols,
            )
        if geo.num_rows == 0:
            return []

        primary_keys: list[str] = geo["primary_key"].to_pylist()
        # Pre-compute per-entity p-values for each enabled detector.
        per_detector_ps: dict[str, dict[str, float]] = {}

        if "delta_norm" in detectors:
            per_detector_ps["delta_norm"] = detector_p_value_delta_norm(
                geo, primary_keys, df=df,
            )

        if "neighbor_contamination" in detectors:
            per_detector_ps["neighbor_contamination"] = (
                self._collect_neighbor_contamination_p(
                    pattern_id, geo,
                )
            )

        if "segment_shift" in detectors:
            per_detector_ps["segment_shift"] = (
                self._collect_segment_shift_p(pattern_id)
            )

        if "trajectory_continuous" in detectors:
            per_detector_ps["trajectory_continuous"] = (
                self._collect_trajectory_continuous_p(
                    pattern_id, primary_keys,
                )
            )

        if "density_gap" in detectors:
            per_detector_ps["density_gap"] = (
                self._collect_density_gap_p(pattern_id)
            )

        # Apply a uniform lower-floor on every per-detector p before
        # combining via HMP. A single detector saturating at the
        # numeric floor (delta_norm clamped to 1e-300 when
        # anomaly_confidence ~ 1.0; Fisher exact at p < 1e-300 on hub
        # segments) would otherwise dominate ``sum(w_i / p_i)`` so
        # completely that the harmonic mean collapses to ~3e-300
        # for every saturated entity, erasing orthogonal signal and
        # tying ranks within the saturated cohort. The
        # ``_HMP_INPUT_P_FLOOR`` clip preserves cross-detector
        # discrimination: entities still rank by combined evidence,
        # not by which detector saturated first.
        for det_name in per_detector_ps:
            per_detector_ps[det_name] = {
                k: max(float(v), _HMP_INPUT_P_FLOOR)
                for k, v in per_detector_ps[det_name].items()
            }

        # Combine per-entity via HMP across the detectors that produced a value.
        combined: list[dict] = []
        for pk in primary_keys:
            ps_for_entity: dict[str, float] = {}
            for det_name, det_dict in per_detector_ps.items():
                if pk in det_dict:
                    ps_for_entity[det_name] = float(det_dict[pk])
            if not ps_for_entity:
                continue
            entity_weights: dict[str, float] | None = None
            if weights is not None:
                entity_weights = {
                    k: float(weights.get(k, 0.0)) for k in ps_for_entity
                }
                if sum(entity_weights.values()) <= 0.0:
                    entity_weights = None
            hmp_value = float(harmonic_mean_p(ps_for_entity, weights=entity_weights))
            if not math.isfinite(hmp_value):
                hmp_value = 1.0
            combined.append({
                "primary_key": pk,
                "hmp": hmp_value,
                # 15-decimal display covers the float64 mantissa range —
                # matches scipy.stats convention and preserves ranking
                # discrimination when HMP saturates near the input
                # p-value floor (sub-1e-9 values would otherwise
                # collapse to 0.0 under 9-decimal rounding even though
                # the unrounded sort order is correct).
                "p_per_detector": {
                    k: round(float(v), 15) for k, v in ps_for_entity.items()
                },
            })

        combined.sort(key=lambda d: d["hmp"])
        if top_n is not None and top_n > 0:
            combined = combined[:top_n]
        for rank, entry in enumerate(combined, start=1):
            entry["rank"] = rank
            entry["hmp"] = round(entry["hmp"], 15)

        # Attach reliability_flags to the top-N (post-truncation) only.
        # Re-fetches delta + anomaly_confidence for the surviving keys —
        # avoids materialising delta vectors for every scanned row. Wrapped
        # in a permissive try-block: storage backends or test mocks that
        # don't carry a ``delta`` column simply skip the attachment, no
        # call abort.
        if combined:
            from hypertopos.engine.geometry import compute_reliability_flags
            top_keys = {entry["primary_key"] for entry in combined}
            delta_rows = None
            try:
                delta_rows = self._storage.read_geometry(
                    pattern_id, version,
                    columns=["primary_key", "delta", "anomaly_confidence"],
                    point_keys=list(top_keys),
                )
            except (TypeError, KeyError, *_NAVIGATION_RECOVERABLE_ERRORS):
                delta_rows = None
            if (
                delta_rows is not None
                and delta_rows.num_rows > 0
                and "delta" in delta_rows.schema.names
            ):
                pk_col = delta_rows["primary_key"].to_pylist()
                d_col = delta_rows["delta"].to_pylist()
                conf_col = (
                    delta_rows["anomaly_confidence"].to_pylist()
                    if "anomaly_confidence" in delta_rows.schema.names
                    else [None] * len(pk_col)
                )
                by_key = {
                    pk: (d, c) for pk, d, c in zip(pk_col, d_col, conf_col)
                }
                for entry in combined:
                    pair = by_key.get(entry["primary_key"])
                    if pair is None or pair[0] is None:
                        continue
                    d, c = pair
                    entry["reliability_flags"] = compute_reliability_flags(
                        d, pattern=pattern, anomaly_confidence=c,
                    )
        return combined

    def classify_detector_consensus(
        self,
        pattern_id: str,
        *,
        detectors: tuple[str, ...] = (
            "delta_norm",
            "neighbor_contamination",
            "segment_shift",
            "trajectory_continuous",
            "density_gap",
        ),
        sample_size: int | None = 10_000,
        top_n: int = 50,
        anomaly_threshold: float = 0.01,
        normal_threshold: float = 0.5,
    ) -> list[dict]:
        """Categorical multi-detector consensus typology — investigator-actionable
        alternative to the scalar HMP ranking from `combine_anomaly_pvalues`.

        Where `combine_anomaly_pvalues` collapses per-detector evidence to a single
        ranked HMP score, `classify_detector_consensus` surfaces the *pattern of
        agreement* between detectors as a categorical label. Two detectors firing
        in opposite directions ("anomalous globally but normal in segment") tells
        the investigator something the combined HMP score hides.

        Band-gap thresholding (anomaly_threshold < normal_threshold) — each
        detector's per-entity p-value is split into three buckets:
            ``p < anomaly_threshold``                     -> clearly anomalous
            ``p > normal_threshold``                      -> clearly normal
            ``anomaly_threshold <= p <= normal_threshold`` -> borderline (excluded)

        The borderline band rules out the classification cliff where p=0.0500006
        (just-over a single threshold) silently flips the entity from
        ``anomalous_consensus`` to ``mixed_signal``. To be flagged as a clear
        anomaly or clear normal a detector's p-value must be on its respective
        side of the band — borderline detectors contribute to ``n_detectors_fired``
        but do not vote in the classification.

        Entity classification:

            * ``mixed_signal`` — at least one detector clearly anomalous AND at
              least one detector clearly normal. Most actionable for
              investigators: this is the "hidden mule" / "legitimate-but-extreme"
              surface where detectors genuinely disagree.
            * ``anomalous_consensus`` — at least two detectors clearly anomalous,
              zero clearly normal (borderlines allowed). Clear investigation
              target.
            * ``single_detector_signal`` — exactly one detector lands clearly on
              either side, the rest are borderline or did not fire. Needs
              corroboration before acting.
            * ``normal_consensus`` — at least two detectors clearly normal, zero
              clearly anomalous (borderlines allowed). Deprioritise.
            * ``insufficient_data`` — zero detectors land clearly on either side.

        The ranking is by classification priority (mixed_signal > anomalous_consensus
        > single_detector_signal > normal_consensus > insufficient_data); within
        each class entries are sorted by HMP ascending so the most-anomalous mixed
        signals surface first.

        Args:
            pattern_id: Pattern to score. Same as `combine_anomaly_pvalues`.
            detectors: Subset of detector names to include.
            sample_size: Cap on geometry rows scored (delegated to
                `combine_anomaly_pvalues`).
            top_n: Maximum entries returned in the ranked list.
            anomaly_threshold: lower band edge. ``p < anomaly_threshold`` flags
                clear anomaly. Default 0.01 (was 0.05 before band-gap; tighter
                cutoff prevents borderline-anomalous detectors from voting).
            normal_threshold: upper band edge. ``p > normal_threshold`` flags
                clear normal. Default 0.5. Must be > anomaly_threshold.

        Returns:
            List of `{primary_key, classification, anomalous_detectors,
            normal_detectors, borderline_detectors, n_detectors_fired, hmp,
            p_per_detector, rank}` sorted by classification priority then HMP
            ascending. ``rank`` reflects position in the returned list.

        Raises:
            ValueError: if ``normal_threshold <= anomaly_threshold``.
        """
        if normal_threshold <= anomaly_threshold:
            raise ValueError(
                f"normal_threshold ({normal_threshold}) must be > "
                f"anomaly_threshold ({anomaly_threshold}).",
            )
        # Reuse the full machinery — single source of truth for collection +
        # saturation guard + sanitisation.
        full_results = self.combine_anomaly_pvalues(
            pattern_id,
            detectors=detectors,
            weights=None,
            sample_size=sample_size,
            top_n=sample_size if sample_size and sample_size > 0 else None,
        )

        classified: list[dict] = []
        for r in full_results:
            p_per_det = r.get("p_per_detector", {}) or {}
            anomalous = sorted(
                d for d, p in p_per_det.items() if p < anomaly_threshold
            )
            normal = sorted(
                d for d, p in p_per_det.items() if p > normal_threshold
            )
            borderline = sorted(
                d for d, p in p_per_det.items()
                if anomaly_threshold <= p <= normal_threshold
            )
            # n_detectors_fired counts every detector that returned a p-value,
            # including borderline ones (they ARE firing, just not voting).
            n_fired = len(anomalous) + len(normal) + len(borderline)
            n_clear = len(anomalous) + len(normal)

            if n_clear == 0:
                classification = "insufficient_data"
            elif n_clear == 1:
                classification = "single_detector_signal"
            elif anomalous and normal:
                classification = "mixed_signal"
            elif anomalous:
                classification = "anomalous_consensus"
            else:
                classification = "normal_consensus"

            # reliability_flags carries delta_norm from combine_anomaly_pvalues
            # — pull it out as a deterministic tiebreaker for the HMP-collapse
            # case (where per-detector p-values saturate at the float floor
            # and the top-N ordering becomes meaningless without a secondary
            # key independent of the p-distribution).
            r_flags = r.get("reliability_flags") or {}
            delta_norm_for_tiebreak = float(r_flags.get("delta_norm") or 0.0)
            classified.append({
                "primary_key": r["primary_key"],
                "classification": classification,
                "anomalous_detectors": anomalous,
                "normal_detectors": normal,
                "borderline_detectors": borderline,
                "n_detectors_fired": n_fired,
                "hmp": r.get("hmp"),
                "p_per_detector": p_per_det,
                "_delta_norm_for_tiebreak": delta_norm_for_tiebreak,
            })

        # Classification priority — mixed_signal first, then anomalous, then
        # weaker signals, then normal, then no-data.
        priority = {
            "mixed_signal": 0,
            "anomalous_consensus": 1,
            "single_detector_signal": 2,
            "normal_consensus": 3,
            "insufficient_data": 4,
        }

        def _consensus_sort_key(
            entry: dict,
        ) -> tuple[int, float, tuple[float, ...], float]:
            # Tiebreak the HMP collapse — when per-detector p-values saturate at
            # the float floor (1e-12), the HMP collapses too and the sort
            # becomes meaningless within the top-N. Use the per-detector
            # p-value vector (ascending) as a third key so entities with
            # MORE detectors at the floor outrank ones with only one; fall
            # through to ``-delta_norm`` so entities with stronger raw
            # geometric anomaly outrank ones with weaker anomaly when the
            # p-vector is identical too (the AML HI-small saturation case).
            p_vec = tuple(sorted(
                float(p) for p in (entry.get("p_per_detector") or {}).values()
            ))
            return (
                priority.get(entry["classification"], 5),
                entry.get("hmp") or 1.0,
                p_vec,
                -float(entry.get("_delta_norm_for_tiebreak") or 0.0),
            )

        classified.sort(key=_consensus_sort_key)
        for entry in classified:
            entry.pop("_delta_norm_for_tiebreak", None)

        if top_n is not None and top_n > 0:
            classified = classified[:top_n]
        for rank, entry in enumerate(classified, start=1):
            entry["rank"] = rank
        return classified

    def _collect_neighbor_contamination_p(
        self,
        pattern_id: str,
        geo: pa.Table,
    ) -> dict[str, float]:
        """Per-entity graph-neighbor anomaly p-values keyed by primary_key.

        Computes neighbor-contamination directly from the AdjacencyIndex
        of the edge-bearing pattern that connects entities of ``pattern_id``.
        For each entity in ``geo`` with at least one graph neighbor:

            (k_obs, x_obs) = (neighbor_count, anomalous_neighbor_count)
            p = P(X >= x_obs | X ~ Hypergeom(N=pop, K=anom_pop, n=k_obs))

        Yields a dense per-entity dict (p ≈ 1.0 for non-elevated entities,
        p << 1.0 for entities with disproportionately many anomalous
        graph neighbors). Returns ``{}`` when no edge-bearing pattern
        can be resolved (anchor without graph) or when the population
        has zero anomalies.
        """
        from hypertopos.engine.p_value_calibration import (
            detector_p_value_neighbor_contamination,
        )

        try:
            sphere = self._storage.read_sphere()
            pattern = sphere.patterns.get(pattern_id)
        except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS):
            return {}
        if pattern is None:
            return {}

        # Resolve which pattern carries the graph edges for these entities.
        graph_pid: str | None = None
        if pattern.pattern_type == "event" and self._storage.has_edge_table(
            pattern_id,
        ):
            graph_pid = pattern_id
        elif pattern.pattern_type == "anchor":
            try:
                graph_pid = self._resolve_edge_pattern_for_anchor(pattern_id)
            except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS):
                return {}
        if graph_pid is None:
            return {}

        # The hypergeometric null compares observed anomalous-neighbor
        # counts against a uniform draw from the FULL graph population
        # (neighbors come from the unsampled graph, so the anomaly map
        # must cover every node — sampling only ``geo`` would silently
        # under-count anomalous neighbors).
        try:
            version = self._resolve_version(pattern_id)
            full_geo = self._storage.read_geometry(
                pattern_id, version,
                columns=["primary_key", "is_anomaly"],
            )
        except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS):
            return {}
        if full_geo.num_rows == 0:
            return {}
        try:
            full_pks = full_geo["primary_key"].to_pylist()
            full_is_anom = full_geo["is_anomaly"].to_pylist()
        except (KeyError, pa.ArrowInvalid):
            return {}
        anomaly_map: dict[str, bool] = {
            str(pk): bool(ia)
            for pk, ia in zip(full_pks, full_is_anom, strict=False)
        }
        total_population = full_geo.num_rows
        total_anomalies = sum(1 for v in full_is_anom if v)
        if total_anomalies <= 0 or total_population <= 0:
            return {}

        # Iterate over the entities in the (possibly sampled) geo —
        # the HMP combiner only needs p-values for these primary_keys.
        try:
            pks_col = geo["primary_key"].to_pylist()
        except (KeyError, pa.ArrowInvalid):
            return {}

        try:
            adj = self._storage.get_adjacency(graph_pid)
        except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS):
            return {}

        observations: dict[str, tuple[int, int]] = {}
        for pk in pks_col:
            spk = str(pk)
            try:
                out_edges = adj.neighbors_out(spk)
                in_edges = adj.neighbors_in(spk)
            except (KeyError, AttributeError):
                continue
            # Distinct neighbor keys (drop self-loops, dedupe across out+in).
            neighbor_keys: set[str] = set()
            for tup in out_edges:
                nk = tup[0] if tup else None
                if nk is None or nk == spk:
                    continue
                neighbor_keys.add(str(nk))
            for tup in in_edges:
                nk = tup[0] if tup else None
                if nk is None or nk == spk:
                    continue
                neighbor_keys.add(str(nk))
            k_obs = len(neighbor_keys)
            if k_obs == 0:
                continue
            x_obs = sum(
                1 for nk in neighbor_keys if anomaly_map.get(nk, False)
            )
            observations[spk] = (k_obs, x_obs)

        if not observations:
            return {}

        return detector_p_value_neighbor_contamination(
            observations,
            total_population=total_population,
            total_anomalies=total_anomalies,
            k=0,
        )

    def _collect_segment_shift_p(
        self,
        pattern_id: str,
        *,
        min_segment_size: int = 30,
    ) -> dict[str, float]:
        """Per-entity segment-shift Fisher p-values keyed by primary_key.

        Computes Fisher exact 2x2 (in-segment vs out-of-segment anomaly
        rate) for every (string column, segment value) pair where the
        segment has at least ``min_segment_size`` entities. The
        ``detect_segment_shift`` public detector applies a stricter
        ``min_shift_ratio=2.0`` and ``max_cardinality=50`` filter to keep
        agent output readable; this collector deliberately bypasses both
        because:

            1. The HMP combiner WANTS dense per-entity p-values, even when
               the shift is mild — Fisher will assign p≈1.0 to weak
               segments and only the elevated ones drag the HMP down.

            2. High-cardinality columns (e.g. ``bank_id`` with 30k unique
               values on AML HI-small) carry the strongest segment signal
               in financial-graph spheres — laundering hubs concentrate
               in a handful of small banks. Filtering by
               ``min_segment_size`` keeps Fisher cells well-conditioned
               while still surfacing those hubs.

        Each entity inherits the most significant Fisher p-value across
        the segments it belongs to. Returns ``{}`` when the entity line
        carries no string columns or population anomaly rate is zero.
        """
        from hypertopos.engine.p_value_calibration import (
            detector_p_value_segment_shift,
        )

        # Population totals over the FULL geometry (Fisher null is
        # population-wide, not sampled).
        try:
            version = self._resolve_version(pattern_id)
            geo = self._storage.read_geometry(
                pattern_id, version,
                columns=["primary_key", "is_anomaly"],
            )
        except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS):
            return {}
        if geo.num_rows == 0:
            return {}
        pop_size = geo.num_rows
        try:
            pop_anom = int(pc.sum(geo["is_anomaly"]).as_py() or 0)
        except (KeyError, pa.ArrowInvalid):
            pop_anom = 0
        if pop_anom <= 0:
            return {}

        try:
            sphere = self._storage.read_sphere()
            entity_line_id = sphere.entity_line(pattern_id)
        except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS, AttributeError):
            return {}
        if not entity_line_id:
            return {}
        line = sphere.lines.get(entity_line_id)
        if line is None or not line.columns:
            return {}

        string_columns = [
            col.name for col in line.columns
            if col.type in ("string", "utf8", "str")
            and col.name != "primary_key"
        ]
        if not string_columns:
            return {}

        try:
            line_ver = self._manifest.line_version(entity_line_id) or 1
            pts = self._storage.read_points(
                entity_line_id, line_ver,
                columns=["primary_key"] + string_columns,
            )
        except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS):
            return {}
        if pts is None or pts.num_rows == 0:
            return {}

        # Vectorized is_anomaly mask aligned with pts.
        anomalous_pk_arr = (
            geo.filter(geo["is_anomaly"]).column("primary_key").combine_chunks()
        )
        pts_pk_arr = pts["primary_key"].combine_chunks()
        is_anom_mask = pc.is_in(pts_pk_arr, value_set=anomalous_pk_arr)

        # First pass: aggregate per segment (column, value) -> (count, anom).
        # Second pass: Fisher per qualifying segment, then back-project.
        per_entity: dict[str, float] = {}

        for col_name in string_columns:
            if col_name not in pts.column_names:
                continue
            col_arr = pts[col_name]
            not_null_mask = pc.is_valid(col_arr)
            grp_tbl = pa.table({
                "seg": col_arr.filter(not_null_mask),
                "is_anom": is_anom_mask.filter(not_null_mask),
                "_pk": pts_pk_arr.filter(not_null_mask),
            })
            try:
                agg = grp_tbl.group_by("seg").aggregate([
                    ("_pk", "count"),
                    ("is_anom", "sum"),
                ])
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                continue

            # Build observations for Fisher in one batch — one entry per
            # qualifying segment value.
            observations: dict[str, dict[str, int]] = {}
            for i in range(agg.num_rows):
                val = agg["seg"][i].as_py()
                if val is None:
                    continue
                in_t = int(agg["_pk_count"][i].as_py() or 0)
                if in_t < min_segment_size:
                    continue
                in_a = int(agg["is_anom_sum"][i].as_py() or 0)
                seg_key = str(val)
                observations[seg_key] = {
                    "in_segment_anomalous": in_a,
                    "in_segment_total": in_t,
                    "out_segment_anomalous": max(pop_anom - in_a, 0),
                    "out_segment_total": max(pop_size - in_t, 0),
                }
            if not observations:
                continue

            seg_p_values = detector_p_value_segment_shift(observations)
            if not seg_p_values:
                continue

            # Back-project: assign each entity the lowest p across its
            # eligible segments (lower p == stronger evidence of shift).
            # The combiner clips at ``_HMP_INPUT_P_FLOOR`` (see
            # ``combine_anomaly_pvalues``); the back-projection here is
            # raw to preserve the segment-level Fisher signal for callers
            # that consume this collector outside the HMP path.
            pk_col = pts["primary_key"].to_pylist()
            val_col = pts[col_name].to_pylist()
            for pk, val in zip(pk_col, val_col, strict=False):
                if pk is None or val is None:
                    continue
                p = seg_p_values.get(str(val))
                if p is None:
                    continue
                spk = str(pk)
                prev = per_entity.get(spk)
                if prev is None or float(p) < prev:
                    per_entity[spk] = float(p)

        return per_entity

    def _collect_trajectory_continuous_p(
        self,
        pattern_id: str,
        primary_keys: list[str],
    ) -> dict[str, float]:
        """Best-effort trajectory DTW p-values keyed by primary_key."""
        from hypertopos.engine.p_value_calibration import (
            detector_p_value_trajectory_continuous,
        )
        from hypertopos.engine.topology import trajectory_continuous_score

        # Read solid table if available — best-effort.
        try:
            solid_tbl = self._storage.read_solid_table(pattern_id)  # type: ignore[attr-defined]
        except (AttributeError, NotImplementedError):
            return {}
        except (GDSError, *_NAVIGATION_RECOVERABLE_ERRORS):
            return {}
        if solid_tbl is None or solid_tbl.num_rows == 0:
            return {}
        try:
            scores = trajectory_continuous_score(solid_tbl)
        except (ValueError, KeyError):
            return {}
        if not scores:
            return {}
        return detector_p_value_trajectory_continuous(scores)

    def _collect_density_gap_p(
        self,
        pattern_id: str,
    ) -> dict[str, float]:
        """Density-gap detector — structurally aggregate, no per-entity p-value.

        ``find_density_gaps`` identifies anomalous **bins** where entities are
        unexpectedly *absent* (joint-marginal density holes). The signal is
        about missing population, not present anomalies, so it has no natural
        per-entity attribution and is silently skipped in the per-entity HMP
        consensus. The adapter `detector_p_value_density_gap` remains
        available for callers that wish to operate on per-entity gap rankings
        (e.g. when a future detector re-projects gaps back to nearest-neighbor
        entities). Always returns ``{}``.
        """
        return {}

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

        When called inside a ``composite_risk_batch`` (i.e. the
        batch memo dict ``self._batch_profile_cache`` is set), the
        first key to hit a given ``(pattern_id, version)`` triggers a
        union read over every key in the batch that participates in
        this pattern; subsequent keys filter the cached table
        in-memory.  Outside a batch this branch is a no-op and the
        legacy per-key indexed read fires as before.
        """
        cache = getattr(self, "_batch_profile_cache", None)
        if isinstance(cache, dict):
            cache_key = ("event_edge", pattern_id, version)
            entry = cache.get(cache_key)
            if entry is None:
                # First batch hit on this pattern — read the union of all
                # batch keys in one indexed scan.  ``entity_keys`` is
                # included so we can attribute each polygon row to the
                # specific input key it matched on.
                batch_keys = list(cache.get("_batch_keys", [entity_key]))
                if entity_key not in batch_keys:
                    batch_keys.append(entity_key)
                cols = ["primary_key", "is_anomaly", "delta_norm", "entity_keys"]
                geo = self._storage.read_geometry(
                    pattern_id, version,
                    point_keys=batch_keys, columns=cols,
                )
                # Vectorised attribution: flatten the entity_keys list
                # column, compute per-target masks via pyarrow compute,
                # and read off matching row indices through
                # ``list_parent_indices``.  This replaces the
                # row-by-row Python loop (a hot path at 10^5–10^6
                # rows) with C++ kernel work plus a small numpy
                # gather.  ``np.unique`` guards against the
                # pathological case where a single row's
                # ``entity_keys`` list contains the same target key
                # more than once — the single-key path counts each
                # row at most once via the BTREE lookup, so we mirror
                # that semantic here.
                ek_col = geo["entity_keys"]
                flat_values = pc.list_flatten(ek_col)
                parent_indices = pc.list_parent_indices(ek_col).to_numpy(
                    zero_copy_only=False,
                )
                anoms_arr = geo["is_anomaly"].to_numpy(zero_copy_only=False)
                pks = geo["primary_key"].to_pylist()
                bucket: dict[str, dict[str, Any]] = {}
                for k in batch_keys:
                    flat_mask = pc.equal(
                        flat_values, pa.scalar(k),
                    ).to_numpy(zero_copy_only=False)
                    matched_rows = np.unique(parent_indices[flat_mask])
                    related = int(matched_rows.size)
                    if related == 0:
                        bucket[k] = {"related_count": 0, "anom_pks": []}
                        continue
                    anom_rows = matched_rows[anoms_arr[matched_rows]]
                    bucket[k] = {
                        "related_count": related,
                        "anom_pks": [pks[int(r)] for r in anom_rows],
                    }
                entry = bucket
                cache[cache_key] = entry
            b = entry.get(entity_key)
            if b is None or b["related_count"] == 0:
                return None
            anom_pks = b["anom_pks"]
            return {
                "key_type": "event_edge",
                "related_count": b["related_count"],
                "anomalous_count": len(anom_pks),
                "anomalous_keys": anom_pks[:max_related],
            }

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

        When called inside a ``composite_risk_batch`` the full chain
        geometry is memoised on ``self._batch_profile_cache`` and
        re-used across every key in the batch — one ~1 s read replaces
        ``len(keys)`` reads.  Outside a batch the legacy path runs
        unchanged.
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
        cache = getattr(self, "_batch_profile_cache", None)
        cached_columns: tuple[list[str], list[bool], list[float]] | None = None
        if isinstance(cache, dict):
            cache_key = ("chain_geo_cols", pattern_id, version)
            cached_columns = cache.get(cache_key)
            if cached_columns is None:
                geo = self._storage.read_geometry(
                    pattern_id, version,
                    columns=["primary_key", "is_anomaly", "delta_norm"],
                )
                cached_columns = (
                    geo["primary_key"].to_pylist(),
                    geo["is_anomaly"].to_pylist(),
                    geo["delta_norm"].to_pylist(),
                )
                cache[cache_key] = cached_columns
        if cached_columns is None:
            geo = self._storage.read_geometry(
                pattern_id, version,
                columns=["primary_key", "is_anomaly", "delta_norm"],
            )
            pks = geo["primary_key"].to_pylist()
            anoms = geo["is_anomaly"].to_pylist()
            norms = geo["delta_norm"].to_pylist()
        else:
            pks, anoms, norms = cached_columns

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

        # Cyclic / self-revisiting chains insert the entity primary_key
        # multiple times into chain_keys (e.g. a chain A->B->A->C contains
        # A twice). The reverse index therefore lists the same chain_id
        # twice for entity A. Per-chain dedup is provided by the geometry
        # table being keyed by chain_id (one row per chain), so the loop
        # below appends each chain at most once regardless of how many
        # times the entity revisits it. set(chain_pks) here is the
        # O(1) membership filter for the loop below — the dedup itself
        # is a geometry-storage invariant.
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

    def find_chains_with_coherent_anomaly(
        self,
        pattern_id: str,
        *,
        anchor_pattern_id: str,
        min_hops: int = 3,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """Find chains where ≥min_hops consecutive entity-anchor positions
        are individually anomalous AND share the same dominant delta dim.

        Targets the "coherent anomaly cascade" signal: chains hopping
        through entities individually flagged for the same structural
        reason (e.g. AML structuring chains routed through accounts that
        all show high pass-through ratio or fan-asymmetry).

        Distinct from find_anomalies(chain_pattern), which scores chains
        on chain-level features (hop_count, amount_decay, time_span);
        this primitive scores chain composition, not chain shape.

        Args:
            pattern_id: chain anchor pattern id (built from chain_lines:).
            anchor_pattern_id: entity anchor pattern whose primary_keys
                match the chain hops (e.g. account_pattern when chains
                hop through accounts).
            min_hops: strict consecutive run length; must be >= 2.
            max_results: cap on returned runs.

        Returns dict with chains list (each entry: chain_id,
        run_start_idx, run_length, top_dim, run_keys, max_delta_norm)
        and diagnostics (n_chains_total, n_anomaly_entities, elapsed_ms).

        Sorting: (run_length DESC, max_delta_norm DESC).
        """
        import time as _time
        t0 = _time.perf_counter()

        if min_hops < 2:
            raise ValueError(
                f"min_hops must be >= 2 (coherence undefined for single "
                f"hop); got {min_hops}",
            )
        if max_results < 0:
            raise ValueError(f"max_results must be >= 0; got {max_results}")

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        if anchor_pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"anchor pattern not found: {anchor_pattern_id!r}",
            )

        chain_pat = sphere.patterns[pattern_id]
        if chain_pat.pattern_type != "anchor":
            raise GDSNavigationError(
                f"pattern_id must be a chain anchor pattern; got "
                f"pattern_type={chain_pat.pattern_type!r} for "
                f"{pattern_id!r}",
            )

        anchor_pat = sphere.patterns[anchor_pattern_id]
        if anchor_pat.pattern_type != "anchor":
            raise GDSNavigationError(
                f"anchor_pattern_id must be an anchor pattern; got "
                f"pattern_type={anchor_pat.pattern_type!r} for "
                f"{anchor_pattern_id!r}",
            )

        chain_line_id = sphere.entity_line(pattern_id)
        if chain_line_id is None:
            raise GDSNavigationError(
                f"chain anchor pattern {pattern_id!r} has no entity_line",
            )
        chain_line_ver = self._manifest.line_version(chain_line_id) or 1
        pts = self._storage.read_points(
            chain_line_id, chain_line_ver,
            columns=["primary_key", "chain_keys"],
        )
        if "chain_keys" not in pts.schema.names:
            raise GDSNavigationError(
                f"line {chain_line_id!r} has no chain_keys column — "
                f"pattern {pattern_id!r} is not a chain pattern",
            )

        # n_chains_total = non-empty chain entries (the actual sweep
        # surface). Computed once here so both the early-return paths
        # and the main sweep share the same semantic.
        chain_keys_strs_all = pts["chain_keys"].to_pylist()
        n_chains_total = sum(1 for ck in chain_keys_strs_all if ck)

        anchor_version = self._resolve_version(anchor_pattern_id)
        try:
            anchor_geo = self._storage.read_geometry(
                anchor_pattern_id, anchor_version,
                columns=["primary_key", "is_anomaly", "delta", "delta_norm"],
            )
        except (KeyError, ValueError) as exc:
            raise GDSNavigationError(
                f"anchor pattern {anchor_pattern_id!r} cannot serve "
                f"is_anomaly / delta — calibration must run first",
            ) from exc

        if anchor_geo.num_rows == 0:
            return {
                "pattern_id": pattern_id,
                "anchor_pattern_id": anchor_pattern_id,
                "n_results": 0,
                "chains": [],
                "diagnostics": _empty_coherent_diagnostics(
                    n_chains_total=n_chains_total,
                    elapsed_ms=round(
                        (_time.perf_counter() - t0) * 1000.0, 2,
                    ),
                ),
            }

        pks_arr = anchor_geo["primary_key"].to_pylist()
        is_anom_arr = np.asarray(
            anchor_geo["is_anomaly"].to_pylist(), dtype=bool,
        )
        delta_norm_arr = anchor_geo["delta_norm"].to_pylist()
        delta_2d = delta_matrix_from_arrow(anchor_geo)
        n_dims = delta_2d.shape[1]

        dim_labels = (
            list(anchor_pat.dim_labels)
            if anchor_pat.dim_labels else
            [f"dim_{i}" for i in range(n_dims)]
        )

        sigma = getattr(anchor_pat, "sigma", None)
        if sigma is not None:
            sigma_arr = np.asarray(sigma, dtype=np.float32)
            if sigma_arr.shape[0] != n_dims:
                sigma_safe = np.ones(n_dims, dtype=np.float32)
            else:
                sigma_safe = np.where(sigma_arr > 1e-12, sigma_arr, 1.0)
        else:
            sigma_safe = np.ones(n_dims, dtype=np.float32)

        anom_idxs = np.flatnonzero(is_anom_arr)
        if anom_idxs.size == 0:
            return {
                "pattern_id": pattern_id,
                "anchor_pattern_id": anchor_pattern_id,
                "n_results": 0,
                "chains": [],
                "diagnostics": _empty_coherent_diagnostics(
                    n_chains_total=n_chains_total,
                    elapsed_ms=round(
                        (_time.perf_counter() - t0) * 1000.0, 2,
                    ),
                ),
            }

        anomaly_delta = delta_2d[anom_idxs]
        normalized = np.abs(anomaly_delta) / sigma_safe[None, :]
        top_idxs = np.argmax(normalized, axis=1)
        anomaly_top_dim: dict[str, str] = {}
        pk_to_norm: dict[str, float] = {}
        for j, i in enumerate(anom_idxs):
            pk = pks_arr[i]
            anomaly_top_dim[pk] = dim_labels[int(top_idxs[j])]
            n_val = delta_norm_arr[i]
            pk_to_norm[pk] = float(n_val) if n_val is not None else 0.0

        chain_pks = pts["primary_key"].to_pylist()
        chain_keys_strs = chain_keys_strs_all

        runs: list[dict[str, Any]] = []
        for chain_pk, ck in zip(chain_pks, chain_keys_strs, strict=False):
            if not ck:
                continue
            keys = ck.split(",")
            n = len(keys)
            best_len = 0
            best_start = 0
            best_dim: str | None = None
            i = 0
            while i < n:
                td = anomaly_top_dim.get(keys[i])
                if td is None:
                    i += 1
                    continue
                j = i + 1
                while j < n and anomaly_top_dim.get(keys[j]) == td:
                    j += 1
                run_len = j - i
                if run_len > best_len:
                    best_len = run_len
                    best_start = i
                    best_dim = td
                i = j
            if best_len >= min_hops and max_results > 0:
                run_keys = keys[best_start:best_start + best_len]
                max_norm = max(
                    (pk_to_norm.get(k, 0.0) for k in run_keys),
                    default=0.0,
                )
                runs.append({
                    "chain_id": chain_pk,
                    "run_start_idx": best_start,
                    "run_length": best_len,
                    "top_dim": best_dim,
                    "run_keys": run_keys,
                    "max_delta_norm": round(max_norm, 4),
                })

        # Compute population aggregates BEFORE sorting + truncating runs[]
        # so that downstream summarisers (e.g. chain_investigation_summary)
        # see the full distribution, not a top_K slice biased toward long
        # runs and dominant top_dim labels. The runs[] field stays
        # truncated for callers that just want the headline list.
        from collections import Counter as _Counter
        n_runs_total_pre_truncation = len(runs)
        top_dim_counts_full: dict[str, int] = dict(
            _Counter(r["top_dim"] for r in runs if r.get("top_dim"))
        )
        if runs:
            run_lengths_arr = np.fromiter(
                (int(r["run_length"]) for r in runs),
                dtype=np.int32, count=len(runs),
            )
            run_length_distribution_full: dict[str, float] = {
                "min": int(run_lengths_arr.min()),
                "p50": int(np.percentile(run_lengths_arr, 50)),
                "p75": int(np.percentile(run_lengths_arr, 75)),
                "p90": int(np.percentile(run_lengths_arr, 90)),
                "max": int(run_lengths_arr.max()),
                "mean": round(float(run_lengths_arr.mean()), 2),
            }
            all_coherent_chain_ids: set[str] = {r["chain_id"] for r in runs}
        else:
            run_length_distribution_full = {
                "min": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0, "mean": 0.0,
            }
            all_coherent_chain_ids = set()

        runs.sort(
            key=lambda r: (r["run_length"], r["max_delta_norm"]),
            reverse=True,
        )
        runs = runs[:max_results]

        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        return {
            "pattern_id": pattern_id,
            "anchor_pattern_id": anchor_pattern_id,
            "n_results": len(runs),
            "chains": runs,
            "diagnostics": {
                "n_chains_total": n_chains_total,
                "n_anomaly_entities": int(anom_idxs.size),
                "n_runs_total_pre_truncation": n_runs_total_pre_truncation,
                "top_dim_counts_full": top_dim_counts_full,
                "run_length_distribution_full": run_length_distribution_full,
                "all_coherent_chain_ids": all_coherent_chain_ids,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        }

    def anomaly_propagation_in_chain(
        self,
        chain_id: str,
        pattern_id: str,
        *,
        anchor_pattern_id: str,
    ) -> dict[str, Any]:
        """Per-hop anomaly progression for a single chain.

        Inspector primitive complementary to
        `find_chains_with_coherent_anomaly`: the latter sweeps the
        population of chains; this primitive takes one chain_id and
        returns its hop-by-hop anomaly trace — for each entity in the
        chain's keys sequence, returns is_anomaly + delta_norm + top
        dominant dim (sigma-normalised argmax over delta) +
        delta_rank_pct.

        Use case: after a population sweep flags a chain as having a
        coherent anomaly run, drill into that chain's full hop
        progression to see how the anomaly accumulates and where it
        breaks.

        Args:
            chain_id: primary_key of the chain in the chain anchor pattern.
            pattern_id: chain anchor pattern id (built from chain_lines:).
            anchor_pattern_id: entity anchor pattern whose primary_keys
                match the chain hops.

        Returns dict with hops[] (per-hop progression) and summary stats
        (n_hops, n_anomalous, max_run_length_same_top_dim,
        dominant_top_dim). On ties for dominant_top_dim,
        ``Counter.most_common`` returns the dim that first appeared in
        chain order (Python 3.7+ insertion-ordered dict guarantee), not
        the lexicographically smallest.

        Raises GDSNavigationError when chain_id is not found, pattern_id
        is not a chain anchor, anchor_pattern_id is not an anchor, or
        the anchor lacks calibration.
        """
        import time as _time
        from collections import Counter
        t0 = _time.perf_counter()

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        if anchor_pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"anchor pattern not found: {anchor_pattern_id!r}",
            )

        chain_pat = sphere.patterns[pattern_id]
        if chain_pat.pattern_type != "anchor":
            raise GDSNavigationError(
                f"pattern_id must be a chain anchor pattern; got "
                f"pattern_type={chain_pat.pattern_type!r} for "
                f"{pattern_id!r}",
            )

        anchor_pat = sphere.patterns[anchor_pattern_id]
        if anchor_pat.pattern_type != "anchor":
            raise GDSNavigationError(
                f"anchor_pattern_id must be an anchor pattern; got "
                f"pattern_type={anchor_pat.pattern_type!r} for "
                f"{anchor_pattern_id!r}",
            )

        chain_line_id = sphere.entity_line(pattern_id)
        if chain_line_id is None:
            raise GDSNavigationError(
                f"chain anchor pattern {pattern_id!r} has no entity_line",
            )
        chain_line_ver = self._manifest.line_version(chain_line_id) or 1

        pts = self._storage.read_points(
            chain_line_id, chain_line_ver,
            columns=["primary_key", "chain_keys"],
            primary_key=chain_id,
        )
        if pts.num_rows == 0:
            raise GDSNavigationError(
                f"chain not found: {chain_id!r} in pattern {pattern_id!r}",
            )
        if "chain_keys" not in pts.schema.names:
            raise GDSNavigationError(
                f"line {chain_line_id!r} has no chain_keys column — "
                f"pattern {pattern_id!r} is not a chain pattern",
            )
        # Defend against a builder-side chain_id collision: multiple chain
        # rows with identical primary_key but distinct chain_keys. Until the
        # extractor guarantees uniqueness, a duplicate is ambiguous and we
        # cannot pick the "correct" chain — raise rather than silently
        # return one variant's hops.
        if pts.num_rows > 1:
            distinct_keys = sorted({
                str(pts["chain_keys"][i].as_py())
                for i in range(pts.num_rows)
                if pts["chain_keys"][i].as_py()
            })
            raise GDSNavigationError(
                f"ambiguous chain_id {chain_id!r} in pattern "
                f"{pattern_id!r}: {pts.num_rows} rows share this id with "
                f"{len(distinct_keys)} distinct chain_keys variants. "
                f"Chain extraction prior to the parallel-worker id "
                f"collision fix produced colliding chain_ids; rebuild "
                f"this sphere's chain pattern to restore primary_key "
                f"uniqueness, then retry the inspector.",
            )
        ck = pts["chain_keys"][0].as_py()
        if not ck:
            return {
                "chain_id": chain_id,
                "pattern_id": pattern_id,
                "anchor_pattern_id": anchor_pattern_id,
                "hops": [],
                "summary": {
                    "n_hops": 0,
                    "n_anomalous": 0,
                    "max_run_length_same_top_dim": 0,
                    "dominant_top_dim": None,
                },
                "elapsed_ms": round(
                    (_time.perf_counter() - t0) * 1000.0, 2,
                ),
            }
        keys = ck.split(",")

        anchor_version = self._resolve_version(anchor_pattern_id)
        try:
            anchor_geo = self._storage.read_geometry(
                anchor_pattern_id, anchor_version,
                columns=["primary_key", "is_anomaly", "delta",
                         "delta_norm", "delta_rank_pct"],
                point_keys=keys,
            )
        except (KeyError, ValueError) as exc:
            raise GDSNavigationError(
                f"anchor pattern {anchor_pattern_id!r} cannot serve "
                f"is_anomaly / delta — calibration must run first",
            ) from exc

        entity_info: dict[str, dict[str, Any]] = {}
        if anchor_geo.num_rows > 0:
            pks_arr = anchor_geo["primary_key"].to_pylist()
            is_anom_arr = anchor_geo["is_anomaly"].to_pylist()
            dn_arr = anchor_geo["delta_norm"].to_pylist()
            drp_arr = anchor_geo["delta_rank_pct"].to_pylist()
            delta_2d = delta_matrix_from_arrow(anchor_geo)
            n_dims = delta_2d.shape[1]

            dim_labels = (
                list(anchor_pat.dim_labels)
                if anchor_pat.dim_labels else
                [f"dim_{i}" for i in range(n_dims)]
            )

            sigma = getattr(anchor_pat, "sigma", None)
            if sigma is not None:
                sigma_arr = np.asarray(sigma, dtype=np.float32)
                if sigma_arr.shape[0] != n_dims:
                    sigma_safe = np.ones(n_dims, dtype=np.float32)
                else:
                    sigma_safe = np.where(
                        sigma_arr > 1e-12, sigma_arr, 1.0,
                    )
            else:
                sigma_safe = np.ones(n_dims, dtype=np.float32)

            normalized = np.abs(delta_2d) / sigma_safe[None, :]
            top_idxs = np.argmax(normalized, axis=1)

            for i, pk in enumerate(pks_arr):
                is_anom = bool(is_anom_arr[i])
                top_dim = (
                    dim_labels[int(top_idxs[i])] if is_anom else None
                )
                entity_info[pk] = {
                    "is_anomaly": is_anom,
                    "delta_norm": (
                        round(float(dn_arr[i]), 4)
                        if dn_arr[i] is not None else 0.0
                    ),
                    "top_dim": top_dim,
                    "delta_rank_pct": (
                        round(float(drp_arr[i]), 2)
                        if drp_arr[i] is not None else None
                    ),
                }

        hops: list[dict[str, Any]] = []
        for i, key in enumerate(keys):
            info = entity_info.get(key)
            if info is None:
                hops.append({
                    "hop_idx": i,
                    "primary_key": key,
                    "is_anomaly": False,
                    "delta_norm": 0.0,
                    "top_dim": None,
                    "delta_rank_pct": None,
                })
            else:
                hops.append({"hop_idx": i, "primary_key": key, **info})

        n_anomalous = sum(1 for h in hops if h["is_anomaly"])
        max_run = 0
        cur_run = 0
        cur_dim: str | None = None
        for h in hops:
            if (
                h["is_anomaly"] and h["top_dim"] is not None
                and h["top_dim"] == cur_dim
            ):
                cur_run += 1
            elif h["is_anomaly"]:
                cur_dim = h["top_dim"]
                cur_run = 1
            else:
                cur_dim = None
                cur_run = 0
            if cur_run > max_run:
                max_run = cur_run

        dim_counter: Counter[str] = Counter(
            h["top_dim"] for h in hops
            if h["is_anomaly"] and h["top_dim"] is not None
        )
        dominant = dim_counter.most_common(1)[0][0] if dim_counter else None

        return {
            "chain_id": chain_id,
            "pattern_id": pattern_id,
            "anchor_pattern_id": anchor_pattern_id,
            "hops": hops,
            "summary": {
                "n_hops": len(hops),
                "n_anomalous": n_anomalous,
                "max_run_length_same_top_dim": max_run,
                "dominant_top_dim": dominant,
            },
            "elapsed_ms": round(
                (_time.perf_counter() - t0) * 1000.0, 2,
            ),
        }

    def chain_witness_intersection(
        self,
        chain_id: str,
        *,
        chain_pattern: str,
        member_pattern: str,
        min_jaccard: float = 0.5,
        top_k_witness: int = 5,
    ) -> dict[str, Any]:
        """Intersect top witness dimensions across a chain's members.

        Pure composition over `explain_anomaly`: resolves a chain anchor's
        member keys via the `chain_keys` column, calls `explain_anomaly`
        per unique member on `member_pattern`, then computes the
        intersection / union / mean pairwise Jaccard of their top-k witness
        dimension labels. Members sharing >=`min_jaccard` witness sets
        imply a coordinated anomaly mechanism — a single geometric
        diagnosis for the structural object.

        Members whose `explain_anomaly` raises (member key not present in
        `member_pattern`'s geometry) are reported as skipped; non-anomalous
        members are explained successfully with an empty witness set, which
        drags pairwise jaccard down naturally.

        Args:
            chain_id: chain anchor primary key.
            chain_pattern: anchor pattern id whose points table carries
                ``chain_keys``.
            member_pattern: pattern id whose ``explain_anomaly`` is called
                per member.
            min_jaccard: threshold for ``coordinated=True``. Default 0.5.
            top_k_witness: per-member top-k witness dims to intersect.
                Default 5.

        Returns dict with chain_id, chain_pattern, member_pattern,
        n_members, n_members_explained, n_members_skipped,
        intersected_witness_dims (alphabetical), union_witness_dims
        (alphabetical), mean_pairwise_witness_jaccard (None when every
        pair has empty union), coordinated (bool), interpretation (str),
        per_member_top_dims (sorted by primary_key).

        Raises ValueError when chain_pattern is not an anchor, when its
        points lack the chain_keys column, when chain_id is not present,
        or when fewer than two unique members can be explained.
        """
        sphere = self._storage.read_sphere()
        if chain_pattern not in sphere.patterns:
            raise ValueError(f"chain_pattern not found: {chain_pattern!r}")
        if member_pattern not in sphere.patterns:
            raise ValueError(f"member_pattern not found: {member_pattern!r}")

        chain_pat = sphere.patterns[chain_pattern]
        if chain_pat.pattern_type != "anchor":
            raise ValueError(
                f"chain_pattern must be an anchor pattern; got "
                f"pattern_type={chain_pat.pattern_type!r} for "
                f"{chain_pattern!r}",
            )

        chain_line_id = sphere.entity_line(chain_pattern)
        if chain_line_id is None:
            raise ValueError(
                f"chain_pattern {chain_pattern!r} has no entity_line",
            )
        chain_line_ver = self._manifest.line_version(chain_line_id) or 1

        pts = self._storage.read_points(
            chain_line_id, chain_line_ver,
            columns=["primary_key", "chain_keys"],
            primary_key=chain_id,
        )
        if "chain_keys" not in pts.schema.names:
            raise ValueError(
                f"line {chain_line_id!r} has no chain_keys column — "
                f"pattern {chain_pattern!r} is not a chain pattern",
            )
        if pts.num_rows == 0:
            raise ValueError(
                f"chain not found: {chain_id!r} in pattern "
                f"{chain_pattern!r}",
            )

        ck = pts["chain_keys"][0].as_py() or ""
        raw_keys = [k.strip() for k in ck.split(",") if k.strip()]
        # Dedupe preserving order: a member appearing twice in chain_keys
        # (e.g. self-loop A->B->A) is a single witness contributor.
        unique_members = list(dict.fromkeys(raw_keys))

        per_member: list[dict[str, Any]] = []
        n_skipped = 0
        for key in unique_members:
            try:
                explanation = self.explain_anomaly(
                    key, pattern_id=member_pattern,
                )
            except (GDSError, ValueError):
                n_skipped += 1
                continue
            top_dims_raw = explanation.get("top_dimensions", []) or []
            labels: list[str] = []
            for entry in top_dims_raw[:top_k_witness]:
                # Legacy GDSEngine.anomaly_dimensions emits {"dim": int,
                # "label": str, ...}; Bregman path emits {"dim": str,
                # "kind": str, ...}. Prefer "label", fall back to "dim".
                lbl = entry.get("label") or entry.get("dim")
                if isinstance(lbl, str) and lbl:
                    labels.append(lbl)
            per_member.append({
                "primary_key": key,
                "top_dims": labels,
            })

        n_explained = len(per_member)
        if n_explained < 2:
            raise ValueError(
                f"chain {chain_id!r}: only {n_explained} unique members "
                f"resolvable in pattern {member_pattern!r} "
                f"({n_skipped} skipped) — cannot compute pairwise jaccard",
            )

        member_sets = [set(m["top_dims"]) for m in per_member]
        union_dims: set[str] = set().union(*member_sets)
        intersect_dims: set[str] = set(member_sets[0]).intersection(*member_sets[1:])

        # Mean pairwise jaccard over all unordered pairs. Empty-union pair
        # contributes 0.0 (no signal), keeping the metric in [0, 1].
        n_pairs = 0
        total = 0.0
        for i in range(len(member_sets)):
            for j in range(i + 1, len(member_sets)):
                a, b = member_sets[i], member_sets[j]
                u = a | b
                if not u:
                    total += 0.0
                else:
                    total += len(a & b) / len(u)
                n_pairs += 1

        if n_pairs == 0:
            mean_jaccard: float | None = None
        else:
            raw_mean = total / n_pairs
            if math.isnan(raw_mean) or math.isinf(raw_mean):
                mean_jaccard = None
            else:
                mean_jaccard = round(raw_mean, 4)

        coordinated = (
            mean_jaccard is not None and mean_jaccard >= min_jaccard
        )

        intersected_sorted = sorted(intersect_dims)
        union_sorted = sorted(union_dims)
        per_member_sorted = sorted(per_member, key=lambda m: m["primary_key"])

        if coordinated:
            interpretation = (
                f"{n_explained} of {len(unique_members)} members coordinated on "
                f"dims {intersected_sorted} "
                f"(mean pairwise jaccard "
                f"{mean_jaccard:.2f})"
                if intersected_sorted
                else (
                    f"{n_explained} members share witness mass above "
                    f"threshold (mean pairwise jaccard "
                    f"{mean_jaccard:.2f}) but no dim appears in every "
                    f"member's top-{top_k_witness}"
                )
            )
        else:
            jval = mean_jaccard if mean_jaccard is not None else 0.0
            interpretation = (
                f"Disjoint witness mechanisms — chain members anomalous "
                f"via independent dim sets (mean pairwise jaccard "
                f"{jval:.2f})"
            )

        return {
            "chain_id": chain_id,
            "chain_pattern": chain_pattern,
            "member_pattern": member_pattern,
            "n_members": len(unique_members),
            "n_members_explained": n_explained,
            "n_members_skipped": n_skipped,
            "intersected_witness_dims": intersected_sorted,
            "union_witness_dims": union_sorted,
            "mean_pairwise_witness_jaccard": mean_jaccard,
            "coordinated": bool(coordinated),
            "interpretation": interpretation,
            "per_member_top_dims": per_member_sorted,
        }

    def chain_signed_confidence_rollup(
        self,
        chain_id: str,
        *,
        chain_pattern: str,
        anchor_pattern: str,
    ) -> dict[str, Any]:
        """Aggregate per-member signed-confidence into a chain-level rollup.

        For each unique member of ``chain_id`` (resolved via the
        ``chain_keys`` column on the chain anchor's points table), reads
        the member's polygon geometry on ``anchor_pattern``, attaches
        reliability flags + the signed-confidence triad
        (``signed_confidence_score`` / ``lda_alignment`` /
        ``reliability_penalty``), then aggregates four chain-level fields
        and a verdict.

        Returns a dict with:
          - ``chain_id``, ``chain_pattern``, ``anchor_pattern``
          - ``n_members``: count of unique deduped chain members
          - ``chain_mean_signed_confidence``: mean of per-member
            ``signed_confidence_score`` across resolved members (``None``
            when label_aware_calibration is absent OR chain is empty)
          - ``chain_n_low_confidence_members``: count of members whose
            ``reliability_penalty >= 0.5`` (``None`` when unavailable)
          - ``chain_n_single_dim_driven_members``: count of members whose
            ``reliability_flags["single_dim_driven"]`` is True (``None``
            when unavailable)
          - ``chain_confidence_verdict``: one of ``"high"`` / ``"medium"``
            / ``"low"`` / ``"label-aware-unavailable"`` (``None`` when
            chain is empty)
          - ``n_members_resolved``: count of members whose geometry was
            successfully read (members absent from geometry are silently
            skipped)

        Verdict thresholds (deterministic):
          - ``"label-aware-unavailable"`` when ``anchor_pattern`` lacks
            ``label_aware_calibration`` (all four numeric fields = None)
          - ``"low"`` when
            ``chain_n_low_confidence_members >= 0.5 * n_members``
          - ``"medium"`` when ``chain_mean_signed_confidence < 1.0`` and
            not ``"low"``. Note: anti-aligned chains (negative mean) fall
            in this bucket by the literal threshold.
          - ``"high"`` otherwise

        Args:
            chain_id: chain anchor primary key.
            chain_pattern: anchor pattern id whose points carry
                ``chain_keys``.
            anchor_pattern: entity anchor pattern whose primary_keys
                match the chain hops.

        Raises ValueError when chain_pattern or anchor_pattern is unknown,
        when chain_pattern is not an anchor, when its points lack
        ``chain_keys``, or when ``chain_id`` is not present.
        """
        sphere = self._storage.read_sphere()
        if chain_pattern not in sphere.patterns:
            raise ValueError(f"chain_pattern not found: {chain_pattern!r}")
        if anchor_pattern not in sphere.patterns:
            raise ValueError(f"anchor_pattern not found: {anchor_pattern!r}")

        chain_pat = sphere.patterns[chain_pattern]
        if chain_pat.pattern_type != "anchor":
            raise ValueError(
                f"chain_pattern must be an anchor pattern; got "
                f"pattern_type={chain_pat.pattern_type!r} for "
                f"{chain_pattern!r}",
            )

        chain_line_id = sphere.entity_line(chain_pattern)
        if chain_line_id is None:
            raise ValueError(
                f"chain_pattern {chain_pattern!r} has no entity_line",
            )
        chain_line_ver = self._manifest.line_version(chain_line_id) or 1

        pts = self._storage.read_points(
            chain_line_id, chain_line_ver,
            columns=["primary_key", "chain_keys"],
            primary_key=chain_id,
        )
        if "chain_keys" not in pts.schema.names:
            raise ValueError(
                f"line {chain_line_id!r} has no chain_keys column — "
                f"pattern {chain_pattern!r} is not a chain pattern",
            )
        if pts.num_rows == 0:
            raise ValueError(
                f"chain not found: {chain_id!r} in pattern "
                f"{chain_pattern!r}",
            )

        ck = pts["chain_keys"][0].as_py() or ""
        raw_keys = [k.strip() for k in ck.split(",") if k.strip()]
        unique_members = list(dict.fromkeys(raw_keys))
        n_members = len(unique_members)

        # Empty-chain short-circuit — all None including verdict.
        if n_members == 0:
            return {
                "chain_id": chain_id,
                "chain_pattern": chain_pattern,
                "anchor_pattern": anchor_pattern,
                "n_members": 0,
                "n_members_resolved": 0,
                "chain_mean_signed_confidence": None,
                "chain_n_low_confidence_members": None,
                "chain_n_single_dim_driven_members": None,
                "chain_confidence_verdict": None,
            }

        anchor_pat = sphere.patterns[anchor_pattern]
        # Label-aware calibration unavailable — fail soft with verdict label.
        if anchor_pat.label_aware_calibration is None:
            return {
                "chain_id": chain_id,
                "chain_pattern": chain_pattern,
                "anchor_pattern": anchor_pattern,
                "n_members": n_members,
                "n_members_resolved": 0,
                "chain_mean_signed_confidence": None,
                "chain_n_low_confidence_members": None,
                "chain_n_single_dim_driven_members": None,
                "chain_confidence_verdict": "label-aware-unavailable",
            }

        # Read per-member polygon geometry for the anchor pattern and
        # build Polygon objects so the existing _attach_* static helpers
        # (used by π5_attract_anomaly) can decorate them with reliability
        # flags + signed-confidence triad.
        anchor_version = self._resolve_version(anchor_pattern)
        light_cols = [
            "primary_key", "scale", "delta", "delta_norm",
            "delta_rank_pct", "is_anomaly",
            "last_refresh_at", "updated_at",
            "bregman_divergence", "anomaly_confidence",
            "edges", "entity_keys",
        ]
        geo = self._storage.read_geometry(
            anchor_pattern, anchor_version,
            point_keys=unique_members,
            columns=light_cols,
        )
        polygons = self._engine.geometry_to_polygons(
            geo,
            pattern=anchor_pat,
            pattern_id=anchor_pattern,
            pattern_type=anchor_pat.pattern_type,
            pattern_ver=anchor_version,
        )
        # Same ordering as π5_attract_anomaly: reliability flags FIRST,
        # signed-confidence triad SECOND (the latter reads
        # reliability_flags via the penalty term).
        self._attach_reliability_flags(polygons, pattern=anchor_pat)
        self._attach_signed_confidence_fields(polygons, pattern=anchor_pat)

        n_resolved = len(polygons)
        if n_resolved == 0:
            # All members absent from geometry — no signal to aggregate.
            # Distinct from the "label-aware-unavailable" shape above:
            # label-aware IS configured here, the data gap is in members
            # missing from the anchor pattern's geometry. Surface verdict
            # = None so consumers can distinguish "calibration missing"
            # from "members missing".
            return {
                "chain_id": chain_id,
                "chain_pattern": chain_pattern,
                "anchor_pattern": anchor_pattern,
                "n_members": n_members,
                "n_members_resolved": 0,
                "chain_mean_signed_confidence": None,
                "chain_n_low_confidence_members": None,
                "chain_n_single_dim_driven_members": None,
                "chain_confidence_verdict": None,
            }

        scores = [
            float(getattr(p, "signed_confidence_score", 0.0) or 0.0)
            for p in polygons
        ]
        n_low_conf = sum(
            1 for p in polygons
            if float(getattr(p, "reliability_penalty", 0.0) or 0.0) >= 0.5
        )
        n_single_dim = sum(
            1 for p in polygons
            if (getattr(p, "reliability_flags", None) or {}).get(
                "single_dim_driven", False,
            )
        )
        mean_signed = sum(scores) / float(n_resolved)
        # Sanitize ±inf / NaN → None for strict-JSON wire format.
        if not np.isfinite(mean_signed):
            mean_signed_out: float | None = None
        else:
            mean_signed_out = float(mean_signed)

        # Verdict thresholds use n_members (the structural denominator)
        # so a chain with many unresolved members lands on "low" when
        # half-or-more of the structural population could not be cleared.
        if n_low_conf >= 0.5 * n_members:
            verdict = "low"
        elif mean_signed_out is not None and mean_signed_out < 1.0:
            verdict = "medium"
        else:
            verdict = "high"

        return {
            "chain_id": chain_id,
            "chain_pattern": chain_pattern,
            "anchor_pattern": anchor_pattern,
            "n_members": n_members,
            "n_members_resolved": n_resolved,
            "chain_mean_signed_confidence": mean_signed_out,
            "chain_n_low_confidence_members": n_low_conf,
            "chain_n_single_dim_driven_members": n_single_dim,
            "chain_confidence_verdict": verdict,
        }

    def chain_drift_trajectory(
        self,
        chain_id: str,
        *,
        chain_pattern: str,
        member_pattern: str,
        n_windows: int = 4,
    ) -> dict[str, Any]:
        """Per-position member delta_norm trajectory over n_windows time slices.

        Resolves ``chain_id`` into member primary keys via the
        ``chain_keys`` column convention on ``chain_pattern``'s points
        table. For each unique member, reads its temporal history via
        ``engine.build_solid``, stride-samples the slices into
        ``n_windows`` contiguous buckets (tail remainder dropped),
        computes each window's mean ``delta_norm``, and fits a
        least-squares slope across windows. The slope's sign and
        magnitude relative to ``0.05 * member_pattern.theta_norm``
        classify each member's regime as ``normalizing`` /
        ``deteriorating`` / ``neutral``. Sign convention is opposite to
        π9_attract_drift: here a positive slope means delta_norm grows
        over time (drifting AWAY from null = deteriorating); π9
        measures radial alignment where positive means drift TOWARD
        null (normalizing).

        Members partition into four counters that sum to
        ``n_members``:

        - ``n_members_with_history``: enough slices for n_windows;
          included in ``per_position_trajectory``.
        - ``n_members_skipped``: ``engine.build_solid`` raised
          (member not in geometry / temporal layer).
        - ``n_members_short_history``: ``build_solid`` succeeded but
          ``len(slices) < n_windows`` — soft-skipped so partial chain
          signal survives.

        Args:
            chain_id: chain anchor primary key.
            chain_pattern: anchor pattern id whose points carry
                ``chain_keys``.
            member_pattern: pattern id whose temporal history is
                consumed per member.
            n_windows: number of time slices per member. Default 4.
                Must be >= 2.

        Returns dict with chain_id, chain_pattern, member_pattern,
        n_members, n_members_with_history, n_members_skipped,
        n_members_short_history, n_windows, per_position_trajectory
        (sorted by original deduped chain index — gaps preserved when
        members are skipped or short),  chain_level_regime
        (``neutral`` / ``normalizing`` / ``deteriorating`` / ``mixed``),
        and chain_drift_score (mean |slope| weighted by per-member
        last-window delta_norm; ``None`` when no finite signal).

        Regime labels align with π9_attract_drift: the neutral band
        (slope within ±cutoff) uses ``neutral``, matching π9's
        ``drift_direction`` vocabulary so agents can safely compare
        labels across primitives.

        Raises ValueError when chain_pattern is not an anchor, when
        its points lack the chain_keys column, when chain_id is not
        present, when n_windows < 2, or when no member has sufficient
        temporal history (``n_members_with_history < 1``).
        """
        if n_windows < 2:
            raise ValueError(
                f"n_windows must be >= 2, got {n_windows}",
            )

        sphere = self._storage.read_sphere()
        if chain_pattern not in sphere.patterns:
            raise ValueError(f"chain_pattern not found: {chain_pattern!r}")
        if member_pattern not in sphere.patterns:
            raise ValueError(f"member_pattern not found: {member_pattern!r}")

        chain_pat = sphere.patterns[chain_pattern]
        if chain_pat.pattern_type != "anchor":
            raise ValueError(
                f"chain_pattern must be an anchor pattern; got "
                f"pattern_type={chain_pat.pattern_type!r} for "
                f"{chain_pattern!r}",
            )

        chain_line_id = sphere.entity_line(chain_pattern)
        if chain_line_id is None:
            raise ValueError(
                f"chain_pattern {chain_pattern!r} has no entity_line",
            )
        chain_line_ver = self._manifest.line_version(chain_line_id) or 1

        pts = self._storage.read_points(
            chain_line_id, chain_line_ver,
            columns=["primary_key", "chain_keys"],
            primary_key=chain_id,
        )
        if "chain_keys" not in pts.schema.names:
            raise ValueError(
                f"line {chain_line_id!r} has no chain_keys column — "
                f"pattern {chain_pattern!r} is not a chain pattern",
            )
        if pts.num_rows == 0:
            raise ValueError(
                f"chain not found: {chain_id!r} in pattern "
                f"{chain_pattern!r}",
            )

        ck = pts["chain_keys"][0].as_py() or ""
        raw_keys = [k.strip() for k in ck.split(",") if k.strip()]
        unique_members = list(dict.fromkeys(raw_keys))

        member_pat = sphere.patterns[member_pattern]
        theta_norm = float(member_pat.theta_norm)
        cutoff = 0.05 * theta_norm

        per_position: list[dict[str, Any]] = []
        n_skipped = 0
        n_short = 0
        for position, key in enumerate(unique_members):
            try:
                solid = self._engine.build_solid(
                    key, member_pattern, self._manifest,
                )
            except (GDSError, ValueError, KeyError):
                n_skipped += 1
                continue
            slices = solid.slices
            if len(slices) < n_windows:
                n_short += 1
                continue

            stride = len(slices) // n_windows
            window_means: list[float | None] = []
            for w in range(n_windows):
                start = w * stride
                end = start + stride
                vals = [float(s.delta_norm_snapshot) for s in slices[start:end]]
                if any(not math.isfinite(v) for v in vals):
                    window_means.append(None)
                else:
                    window_means.append(sum(vals) / len(vals))

            # Slope via least-squares fit. Any None in window_means
            # contaminates the fit -> sanitise slope to None.
            if any(v is None for v in window_means):
                slope: float | None = None
            else:
                xs = np.arange(n_windows, dtype=np.float64)
                ys = np.asarray(window_means, dtype=np.float64)
                raw_slope = float(np.polyfit(xs, ys, 1)[0])
                slope = raw_slope if math.isfinite(raw_slope) else None

            # Regime classification. Zero theta_norm => no scale, no
            # signal => label all members 'neutral' as a defensive fallback.
            if slope is None or theta_norm == 0.0:
                regime = "neutral"
            elif slope >= cutoff:
                regime = "deteriorating"
            elif slope <= -cutoff:
                regime = "normalizing"
            else:
                regime = "neutral"

            per_position.append({
                "position": position,
                "member_key": key,
                "delta_norms_over_time": [
                    None if v is None else round(v, 6) for v in window_means
                ],
                "slope": None if slope is None else round(slope, 6),
                "regime": regime,
            })

        n_with_history = len(per_position)
        if n_with_history < 1:
            raise ValueError(
                f"chain {chain_id!r}: n_members_with_history < 1 "
                f"({n_skipped} skipped, {n_short} short_history) — "
                f"no member has at least {n_windows} temporal slices; "
                f"retry with a smaller n_windows or a chain with denser "
                f"temporal history",
            )

        regimes = [e["regime"] for e in per_position]
        regime_set = set(regimes)
        if regime_set == {"neutral"}:
            chain_level_regime = "neutral"
        elif regime_set == {"normalizing"}:
            chain_level_regime = "normalizing"
        elif regime_set == {"deteriorating"}:
            chain_level_regime = "deteriorating"
        else:
            chain_level_regime = "mixed"

        # Chain drift score: |slope| weighted by last-window delta_norm
        # (clipped to >= 0). When all weights are zero or all slopes
        # are None, fall back to uniform mean over finite |slope|s.
        finite_entries = [
            e for e in per_position if e["slope"] is not None
            and e["delta_norms_over_time"][-1] is not None
        ]
        if not finite_entries:
            chain_drift_score: float | None = None
        else:
            weights = [max(0.0, e["delta_norms_over_time"][-1]) for e in finite_entries]
            abs_slopes = [abs(e["slope"]) for e in finite_entries]
            if sum(weights) > 0.0:
                raw_score = (
                    sum(s * w for s, w in zip(abs_slopes, weights, strict=True))
                    / sum(weights)
                )
            else:
                raw_score = sum(abs_slopes) / len(abs_slopes)
            chain_drift_score = (
                round(raw_score, 6) if math.isfinite(raw_score) else None
            )

        return {
            "chain_id": chain_id,
            "chain_pattern": chain_pattern,
            "member_pattern": member_pattern,
            "n_members": len(unique_members),
            "n_members_with_history": n_with_history,
            "n_members_skipped": n_skipped,
            "n_members_short_history": n_short,
            "n_windows": n_windows,
            "per_position_trajectory": per_position,
            "chain_level_regime": chain_level_regime,
            "chain_drift_score": chain_drift_score,
        }

    def classify_chain_typology(
        self,
        chain_id: str,
        pattern_id: str,
        *,
        anchor_pattern_id: str,
    ) -> dict[str, Any]:
        """Five-dimensional typology classification for a single chain.

        Wraps `anomaly_propagation_in_chain` and applies classification
        rules over the per-hop trace to label the chain along five
        operational axes:

          - shape: monotone-rising / monotone-falling / peak-in-middle /
            peak-at-start / peak-at-end / flat / no-anomalous-run
          - peak_position: at-start / early / middle / late / at-end /
            single-hop / no-run
          - position_in_chain: leading / transit / terminal / full-chain /
            no-run
          - extension_signals: backward (pre-run rank in elevated band) /
            forward (next-hop rank in elevated band)
          - dominant_top_dim across the entire chain

        Run band buckets ("low" / "median" / "elevated" /
        "borderline-anomalous") are the same as
        `find_chains_with_coherent_anomaly`'s breakpoint analysis.
        """
        import time as _time
        t0 = _time.perf_counter()

        trace = self.anomaly_propagation_in_chain(
            chain_id, pattern_id, anchor_pattern_id=anchor_pattern_id,
        )
        hops = trace["hops"]
        n_hops = len(hops)

        # Identify the longest same-top-dim run (matches the C1 sweep).
        run_start: int | None = None
        run_length = 0
        run_top_dim: str | None = None
        cur_dim: str | None = None
        cur_start = 0
        cur_len = 0
        for i, h in enumerate(hops):
            td = h["top_dim"] if h["is_anomaly"] else None
            if td is not None and td == cur_dim:
                cur_len += 1
            elif td is not None:
                cur_dim = td
                cur_start = i
                cur_len = 1
            else:
                cur_dim = None
                cur_len = 0
            if cur_len > run_length:
                run_length = cur_len
                run_start = cur_start
                run_top_dim = cur_dim

        if run_start is None or run_length == 0:
            return {
                "chain_id": chain_id,
                "pattern_id": pattern_id,
                "anchor_pattern_id": anchor_pattern_id,
                "typology": {
                    "shape": "no-anomalous-run",
                    "peak_position": "no-run",
                    "position_in_chain": "no-run",
                    "extension_signals": {"backward": False, "forward": False},
                    "pre_run_rank_bucket": "no-pre-run",
                    "breakpoint_rank_bucket": "no-breakpoint",
                    "dominant_top_dim": trace["summary"]["dominant_top_dim"],
                    "run_length": 0,
                    "run_start_idx": None,
                    "run_top_dim": None,
                },
                "elapsed_ms": round(
                    (_time.perf_counter() - t0) * 1000.0, 2,
                ),
            }

        run_end_excl = run_start + run_length
        run_deltas = [h["delta_norm"] for h in hops[run_start:run_end_excl]]

        # Shape classification.
        if len(run_deltas) < 2:
            shape = "single-hop"
        else:
            diffs = [run_deltas[i + 1] - run_deltas[i] for i in range(len(run_deltas) - 1)]
            n_up = sum(1 for d in diffs if d > 0.5)
            n_down = sum(1 for d in diffs if d < -0.5)
            if n_up == 0 and n_down == 0:
                shape = "flat"
            elif n_down == 0 and n_up > 0:
                shape = "monotone-rising"
            elif n_up == 0 and n_down > 0:
                shape = "monotone-falling"
            else:
                peak_local = run_deltas.index(max(run_deltas))
                if 0 < peak_local < len(run_deltas) - 1:
                    shape = "peak-in-middle"
                elif peak_local == 0:
                    shape = "peak-at-start"
                else:
                    shape = "peak-at-end"

        # Peak position bucket within the run.
        if run_deltas:
            peak_local = run_deltas.index(max(run_deltas))
            n = len(run_deltas)
            if n == 1:
                peak_position = "single-hop"
            elif peak_local == 0:
                peak_position = "at-start"
            elif peak_local == n - 1:
                peak_position = "at-end"
            elif peak_local < n / 3:
                peak_position = "early"
            elif peak_local > 2 * n / 3:
                peak_position = "late"
            else:
                peak_position = "middle"
        else:
            peak_position = "no-run"

        # Position class.
        is_at_start = run_start == 0
        is_terminal = run_end_excl >= n_hops
        if is_at_start and is_terminal:
            position_in_chain = "full-chain"
        elif is_at_start:
            position_in_chain = "leading"
        elif is_terminal:
            position_in_chain = "terminal"
        else:
            position_in_chain = "transit"

        def _bucket(rank: float | None) -> str:
            if rank is None:
                return "no-rank"
            if rank >= 95:
                return "borderline-anomalous"
            if rank >= 80:
                return "elevated"
            if rank >= 50:
                return "median"
            return "low"

        pre_run_rank = (
            hops[run_start - 1].get("delta_rank_pct")
            if run_start > 0 else None
        )
        breakpoint_rank = (
            hops[run_end_excl].get("delta_rank_pct")
            if run_end_excl < n_hops else None
        )
        pre_run_bucket = (
            _bucket(pre_run_rank) if run_start > 0 else "no-pre-run"
        )
        breakpoint_bucket = (
            _bucket(breakpoint_rank)
            if run_end_excl < n_hops else "no-breakpoint"
        )
        backward_signal = pre_run_bucket in ("elevated", "borderline-anomalous")
        forward_signal = breakpoint_bucket in (
            "elevated", "borderline-anomalous",
        )

        return {
            "chain_id": chain_id,
            "pattern_id": pattern_id,
            "anchor_pattern_id": anchor_pattern_id,
            "typology": {
                "shape": shape,
                "peak_position": peak_position,
                "position_in_chain": position_in_chain,
                "extension_signals": {
                    "backward": backward_signal,
                    "forward": forward_signal,
                },
                "pre_run_rank_bucket": pre_run_bucket,
                "breakpoint_rank_bucket": breakpoint_bucket,
                "dominant_top_dim": trace["summary"]["dominant_top_dim"],
                "run_length": run_length,
                "run_start_idx": run_start,
                "run_top_dim": run_top_dim,
            },
            "elapsed_ms": round(
                (_time.perf_counter() - t0) * 1000.0, 2,
            ),
        }

    def chain_investigation_summary(
        self,
        chain_pattern_id: str,
        *,
        anchor_pattern_id: str,
        min_hops: int = 2,
        max_runs: int = 10_000,
    ) -> dict[str, Any]:
        """Pre-investigation triage for a chain pattern.

        Returns a population-level summary that lets an agent decide
        whether to commit investigation budget to the chain-coherent loop
        on this sphere before drilling in. Aggregates from a single
        ``find_chains_with_coherent_anomaly`` sweep + a chain-pattern
        geometry scan, so the cost is one coherent-anomaly sweep — what
        the agent would pay anyway as the first triage step.

        Args:
            chain_pattern_id: chain anchor pattern id.
            anchor_pattern_id: entity anchor pattern that supplies the
                ``is_anomaly`` / ``delta`` columns referenced by chain
                hops (e.g. ``account_pattern`` when chains hop accounts).
            min_hops: minimum coherent-run length to count; passed
                through to ``find_chains_with_coherent_anomaly``.
            max_runs: cap on coherent runs read for the typology
                aggregation; defaults large enough to capture the full
                tail on most spheres.

        Returns dict with: chain_pattern_id, anchor_pattern_id,
        n_chains_total, n_chains_with_coherent_anomaly_run,
        coherent_run_rate, n_chains_with_shape_anomaly,
        shape_anomaly_rate, cross_pattern_overlap (n_both,
        n_coherent_only, n_shape_only, jaccard),
        top_dims_in_coherent_runs (top 10 dim labels by run count),
        run_length_distribution (min, p50, p75, p90, max, mean),
        recommended_min_hops, elapsed_ms.
        """
        import time as _time
        from collections import Counter
        t0 = _time.perf_counter()

        if min_hops < 2:
            raise ValueError(
                f"min_hops must be >= 2 (coherence undefined for single "
                f"hop); got {min_hops}",
            )
        if max_runs < 0:
            raise ValueError(f"max_runs must be >= 0; got {max_runs}")

        sphere = self._storage.read_sphere()
        if chain_pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"pattern not found: {chain_pattern_id!r}",
            )
        if anchor_pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"anchor pattern not found: {anchor_pattern_id!r}",
            )

        coh = self.find_chains_with_coherent_anomaly(
            chain_pattern_id,
            anchor_pattern_id=anchor_pattern_id,
            min_hops=min_hops,
            max_results=max_runs,
        )
        diag = coh["diagnostics"]
        n_chains_total = int(diag["n_chains_total"])
        # Use pre-truncation aggregates from the underlying sweep so that
        # callers passing a small max_runs still get the FULL population
        # distribution. The sweep sorts (run_length DESC, max_delta_norm
        # DESC) before truncating, so reading aggregates from chains[]
        # would bias every metric toward long-run / high-delta tails.
        n_coherent = int(diag["n_runs_total_pre_truncation"])
        coherent_run_rate = (
            round(n_coherent / n_chains_total, 6)
            if n_chains_total > 0 else 0.0
        )
        coh_pks: set[str] = set(diag["all_coherent_chain_ids"])

        chain_version = self._resolve_version(chain_pattern_id)
        try:
            chain_geo = self._storage.read_geometry(
                chain_pattern_id, chain_version,
                columns=["primary_key", "is_anomaly"],
            )
        except (KeyError, ValueError):
            chain_anom_pks: set[str] = set()
        else:
            chain_geo_pks = chain_geo["primary_key"].to_pylist()
            chain_geo_anoms = chain_geo["is_anomaly"].to_pylist()
            chain_anom_pks = {
                pk for pk, a in zip(chain_geo_pks, chain_geo_anoms, strict=False)
                if a
            }

        n_shape = len(chain_anom_pks)
        shape_anomaly_rate = (
            round(n_shape / n_chains_total, 6)
            if n_chains_total > 0 else 0.0
        )

        intersect = coh_pks & chain_anom_pks
        union = coh_pks | chain_anom_pks
        jaccard = round(len(intersect) / len(union), 6) if union else 0.0

        top_dim_counts: Counter[str] = Counter(diag["top_dim_counts_full"])
        top_dims = [
            {"top_dim": d, "count": c}
            for d, c in top_dim_counts.most_common(10)
        ]

        run_length_distribution = dict(diag["run_length_distribution_full"])
        if n_coherent >= 50:
            recommended_min_hops = max(
                int(run_length_distribution["p75"]),
                min_hops,
            )
        else:
            recommended_min_hops = min_hops

        elapsed_ms = round((_time.perf_counter() - t0) * 1000.0, 2)
        return {
            "chain_pattern_id": chain_pattern_id,
            "anchor_pattern_id": anchor_pattern_id,
            "n_chains_total": n_chains_total,
            "n_chains_with_coherent_anomaly_run": n_coherent,
            "coherent_run_rate": coherent_run_rate,
            "n_chains_with_shape_anomaly": n_shape,
            "shape_anomaly_rate": shape_anomaly_rate,
            "cross_pattern_overlap": {
                "n_both": len(intersect),
                "n_coherent_only": len(coh_pks - chain_anom_pks),
                "n_shape_only": len(chain_anom_pks - coh_pks),
                "jaccard": jaccard,
            },
            "top_dims_in_coherent_runs": top_dims,
            "run_length_distribution": run_length_distribution,
            "recommended_min_hops": recommended_min_hops,
            "elapsed_ms": elapsed_ms,
        }

    def extend_chain(
        self,
        chain_id: str,
        pattern_id: str,
        *,
        anchor_pattern_id: str,
        direction: str = "forward",
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Suggest candidate extension entities at the boundary of a
        chain's anomalous run.

        Forward: at the run's end, find entities that follow the
        boundary entity in OTHER chains in the same chain pattern, and
        rank them by their own anchor anomaly status. These are
        candidates for extending the investigation forward into the
        laundering ring.

        Backward: same logic at the run's start, returning entities
        that PRECEDE the boundary entity in other chains.

        Returns dict with `boundary_key`, `boundary_position`
        (start/end of run within original chain), `candidates[]`
        (entity_key, is_anomaly, delta_norm, delta_rank_pct,
        source_chain_id), and `summary` (`n_candidates`,
        `n_anomalous_candidates`, `n_unique_keys`).
        """
        import time as _time
        from collections import defaultdict
        t0 = _time.perf_counter()

        if direction not in ("forward", "backward"):
            raise ValueError(
                f"direction must be 'forward' or 'backward'; "
                f"got {direction!r}",
            )
        if max_results < 0:
            raise ValueError(f"max_results must be >= 0; got {max_results}")

        trace = self.anomaly_propagation_in_chain(
            chain_id, pattern_id, anchor_pattern_id=anchor_pattern_id,
        )
        hops = trace["hops"]
        n_hops = len(hops)

        # Locate boundary: end of the longest anomalous-same-top-dim run
        # for forward, start for backward.
        run_start: int | None = None
        run_end_excl = 0
        cur_dim: str | None = None
        cur_start = 0
        cur_len = 0
        best_len = 0
        for i, h in enumerate(hops):
            td = h["top_dim"] if h["is_anomaly"] else None
            if td is not None and td == cur_dim:
                cur_len += 1
            elif td is not None:
                cur_dim = td
                cur_start = i
                cur_len = 1
            else:
                cur_dim = None
                cur_len = 0
            if cur_len > best_len:
                best_len = cur_len
                run_start = cur_start
                run_end_excl = i + 1

        if run_start is None or best_len == 0:
            return {
                "chain_id": chain_id,
                "pattern_id": pattern_id,
                "anchor_pattern_id": anchor_pattern_id,
                "direction": direction,
                "boundary_key": None,
                "boundary_position": None,
                "candidates": [],
                "summary": {
                    "n_candidates": 0,
                    "n_anomalous_candidates": 0,
                    "n_unique_keys": 0,
                },
                "elapsed_ms": round(
                    (_time.perf_counter() - t0) * 1000.0, 2,
                ),
            }

        if direction == "forward":
            boundary_idx = run_end_excl - 1
            boundary_key = hops[boundary_idx]["primary_key"]
            boundary_position = "run-end"
        else:
            boundary_idx = run_start
            boundary_key = hops[boundary_idx]["primary_key"]
            boundary_position = "run-start"

        # Use the chain reverse index to find OTHER chains containing
        # the boundary key. Then walk those chains to extract the
        # candidate "next-after-boundary" or "prev-before-boundary"
        # entity keys.
        sphere = self._storage.read_sphere()
        chain_line_id = sphere.entity_line(pattern_id)
        chain_line_ver = self._manifest.line_version(chain_line_id) or 1

        # Read full chain points (filtered to chains containing
        # boundary_key via the existing reverse index).
        rev_idx = self._get_chain_reverse_index(chain_line_id, chain_line_ver)
        chain_pks_with_boundary = rev_idx.get(boundary_key, [])
        if not chain_pks_with_boundary:
            return {
                "chain_id": chain_id,
                "pattern_id": pattern_id,
                "anchor_pattern_id": anchor_pattern_id,
                "direction": direction,
                "boundary_key": boundary_key,
                "boundary_position": boundary_position,
                "candidates": [],
                "summary": {
                    "n_candidates": 0,
                    "n_anomalous_candidates": 0,
                    "n_unique_keys": 0,
                },
                "elapsed_ms": round(
                    (_time.perf_counter() - t0) * 1000.0, 2,
                ),
            }

        # Targeted batch read of the chains containing the boundary key.
        # The reverse index narrowed us to a small subset; reading the
        # full points table just to filter in Python materialises all
        # 290 k+ string columns and dominates wall-clock at ~700 ms warm.
        # read_points_batch uses BTREE-pushed equality scans, sub-100 ms.
        pts = self._storage.read_points_batch(
            chain_line_id, chain_line_ver,
            primary_keys=sorted(set(chain_pks_with_boundary)),
            columns=["primary_key", "chain_keys"],
        )
        all_pks = pts["primary_key"].to_pylist()
        all_cks = pts["chain_keys"].to_pylist()

        # Collect candidate entity keys per source chain.
        candidate_to_sources: dict[str, list[str]] = defaultdict(list)
        for pk, ck in zip(all_pks, all_cks, strict=False):
            if pk == chain_id or not ck:
                continue
            keys = ck.split(",")
            for j, k in enumerate(keys):
                if k != boundary_key:
                    continue
                if direction == "forward" and j + 1 < len(keys):
                    candidate_to_sources[keys[j + 1]].append(pk)
                elif direction == "backward" and j > 0:
                    candidate_to_sources[keys[j - 1]].append(pk)

        if not candidate_to_sources:
            return {
                "chain_id": chain_id,
                "pattern_id": pattern_id,
                "anchor_pattern_id": anchor_pattern_id,
                "direction": direction,
                "boundary_key": boundary_key,
                "boundary_position": boundary_position,
                "candidates": [],
                "summary": {
                    "n_candidates": 0,
                    "n_anomalous_candidates": 0,
                    "n_unique_keys": 0,
                },
                "elapsed_ms": round(
                    (_time.perf_counter() - t0) * 1000.0, 2,
                ),
            }

        # Look up anchor anomaly + delta_norm + delta_rank_pct for
        # candidate keys.
        candidate_keys = list(candidate_to_sources.keys())
        anchor_version = self._resolve_version(anchor_pattern_id)
        try:
            anchor_geo = self._storage.read_geometry(
                anchor_pattern_id, anchor_version,
                columns=["primary_key", "is_anomaly", "delta_norm",
                         "delta_rank_pct"],
                point_keys=candidate_keys,
            )
        except (KeyError, ValueError) as exc:
            raise GDSNavigationError(
                f"anchor pattern {anchor_pattern_id!r} cannot serve "
                f"is_anomaly / delta — calibration must run first",
            ) from exc

        info_by_key: dict[str, dict[str, Any]] = {}
        if anchor_geo.num_rows > 0:
            for i, pk in enumerate(anchor_geo["primary_key"].to_pylist()):
                info_by_key[pk] = {
                    "is_anomaly": bool(
                        anchor_geo["is_anomaly"][i].as_py(),
                    ),
                    "delta_norm": (
                        round(float(anchor_geo["delta_norm"][i].as_py()), 4)
                        if anchor_geo["delta_norm"][i].as_py() is not None
                        else 0.0
                    ),
                    "delta_rank_pct": (
                        round(
                            float(
                                anchor_geo["delta_rank_pct"][i].as_py()
                            ), 2,
                        )
                        if anchor_geo["delta_rank_pct"][i].as_py() is not None
                        else None
                    ),
                }

        candidates: list[dict[str, Any]] = []
        for k, sources in candidate_to_sources.items():
            info = info_by_key.get(k, {
                "is_anomaly": False,
                "delta_norm": 0.0,
                "delta_rank_pct": None,
            })
            candidates.append({
                "entity_key": k,
                "is_anomaly": info["is_anomaly"],
                "delta_norm": info["delta_norm"],
                "delta_rank_pct": info["delta_rank_pct"],
                "n_source_chains": len(sources),
                "source_chain_ids": sources[:5],
            })

        candidates.sort(
            key=lambda c: (
                c["is_anomaly"],
                c["delta_norm"],
                c["n_source_chains"],
            ),
            reverse=True,
        )
        candidates = candidates[:max_results]

        n_anom = sum(1 for c in candidates if c["is_anomaly"])
        return {
            "chain_id": chain_id,
            "pattern_id": pattern_id,
            "anchor_pattern_id": anchor_pattern_id,
            "direction": direction,
            "boundary_key": boundary_key,
            "boundary_position": boundary_position,
            "candidates": candidates,
            "summary": {
                "n_candidates": len(candidates),
                "n_anomalous_candidates": n_anom,
                "n_unique_keys": len(candidate_to_sources),
            },
            "elapsed_ms": round(
                (_time.perf_counter() - t0) * 1000.0, 2,
            ),
        }

    def investigate_chain(
        self,
        chain_id: str,
        pattern_id: str,
        *,
        anchor_pattern_id: str,
        extension_max_results: int = 20,
    ) -> dict[str, Any]:
        """One-shot orchestrator that runs the full R9 investigative loop
        on a single chain and aggregates the per-step outputs into a
        single SAR-ready report.

        Composes (in order):
          1. ``anomaly_propagation_in_chain`` — per-hop progression + run
             summary
          2. ``classify_chain_typology`` — five-axis operational tag
          3. chain-pattern geometry lookup for the chain_id — `is_anomaly`,
             `delta_norm`, `delta_rank_pct` (one row read; equivalent to
             asking ``find_anomalies(<chain_pattern>)`` for this chain
             only)
          4. ``extend_chain(direction='forward')`` — boundary candidates
             past the run end
          5. ``extend_chain(direction='backward')`` — boundary candidates
             before the run start

        Each step's status is reported individually in the output so a
        partial failure (e.g. extension lookup raises because the chain
        has no anomalous run) does not abort the whole investigation.

        ``summary`` derives an investigation strength + recommended
        action from the trace + typology + shape + extension signals.
        Strength is a coarse triage signal — the per-step blocks remain
        the source of truth for any nuanced reading.

        Returns dict with keys: chain_id, pattern_id, anchor_pattern_id,
        trace, typology, shape_anomaly, extension_forward,
        extension_backward, summary, elapsed_ms.
        """
        import time as _time
        t0 = _time.perf_counter()

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        if anchor_pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"anchor pattern not found: {anchor_pattern_id!r}",
            )

        def _safe(call):
            try:
                return {"ok": True, "data": call()}
            except (GDSNavigationError, ValueError, KeyError) as e:
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        trace_block = _safe(lambda: self.anomaly_propagation_in_chain(
            chain_id, pattern_id, anchor_pattern_id=anchor_pattern_id,
        ))
        typology_block = _safe(lambda: self.classify_chain_typology(
            chain_id, pattern_id, anchor_pattern_id=anchor_pattern_id,
        ))

        chain_version = self._resolve_version(pattern_id)

        def _shape_lookup() -> dict[str, Any]:
            chain_geo = self._storage.read_geometry(
                pattern_id, chain_version,
                point_keys=[chain_id],
                columns=["primary_key", "is_anomaly", "delta_norm",
                         "delta_rank_pct"],
            )
            if chain_geo.num_rows == 0:
                raise GDSNavigationError(
                    f"chain_id {chain_id!r} not found in "
                    f"{pattern_id!r} geometry"
                )
            return {
                "chain_id": chain_id,
                "is_anomaly": bool(chain_geo["is_anomaly"][0].as_py()),
                "delta_norm": (
                    None if chain_geo["delta_norm"][0].as_py() is None
                    else round(
                        float(chain_geo["delta_norm"][0].as_py()), 4,
                    )
                ),
                "delta_rank_pct": (
                    None if chain_geo["delta_rank_pct"][0].as_py() is None
                    else round(
                        float(chain_geo["delta_rank_pct"][0].as_py()), 2,
                    )
                ),
            }

        shape_anomaly = _safe(_shape_lookup)

        extension_forward = _safe(lambda: self.extend_chain(
            chain_id, pattern_id, anchor_pattern_id=anchor_pattern_id,
            direction="forward", max_results=extension_max_results,
        ))
        extension_backward = _safe(lambda: self.extend_chain(
            chain_id, pattern_id, anchor_pattern_id=anchor_pattern_id,
            direction="backward", max_results=extension_max_results,
        ))

        summary = self._derive_investigation_summary(
            trace_block, typology_block, shape_anomaly,
            extension_forward, extension_backward,
        )

        elapsed_ms = round((_time.perf_counter() - t0) * 1000.0, 2)
        return {
            "chain_id": chain_id,
            "pattern_id": pattern_id,
            "anchor_pattern_id": anchor_pattern_id,
            "trace": trace_block,
            "typology": typology_block,
            "shape_anomaly": shape_anomaly,
            "extension_forward": extension_forward,
            "extension_backward": extension_backward,
            "summary": summary,
            "elapsed_ms": elapsed_ms,
        }

    @staticmethod
    def _derive_investigation_summary(
        trace_block: dict,
        typology_block: dict,
        shape_anomaly: dict,
        extension_forward: dict,
        extension_backward: dict,
    ) -> dict[str, Any]:
        """Aggregate per-step outcomes into a coarse triage signal.

        Strength scoring — four 0/1 signals (chain-composition focused):
          - trace.summary.max_run_length_same_top_dim >= 3
          - typology.position_in_chain in {leading, transit, terminal,
            full-chain} (i.e. NOT no-run)
          - extension_forward.summary.n_anomalous_candidates >= 1
          - extension_backward.summary.n_anomalous_candidates >= 1

        Strength buckets: 3-4 = strong → escalate to SAR;
        2 = moderate → continue investigation; 0-1 = weak →
        false-positive candidate.

        ``shape_anomaly`` is intentionally NOT in the score. R9's value
        proposition is catching what ``find_anomalies(<chain_pattern>)``
        misses — composition-anomalous-but-shape-normal is the textbook
        R9 sweet spot. Awarding score for shape agreement would
        mechanically cap that case at moderate. The shape block stays
        in the report as evidence (and as a one-line rationale add-on
        when it agrees) but does not drive the verdict.

        Rationale concatenates the contributing signals as a single
        SAR-ready paragraph for paste into investigator notes; when the
        chain shape ALSO flags, that's surfaced as additional evidence.
        """
        signals: list[str] = []
        score = 0

        if trace_block.get("ok"):
            t = trace_block["data"]
            run_len = int(t.get("summary", {}).get(
                "max_run_length_same_top_dim", 0,
            ))
            top_dim = t.get("summary", {}).get("dominant_top_dim")
            if run_len >= 3:
                score += 1
                signals.append(
                    f"Coherent anomaly run of length {run_len} on "
                    f"`{top_dim}`."
                )

        if typology_block.get("ok"):
            ty = typology_block["data"].get("typology", {})
            position = ty.get("position_in_chain")
            shape_label = ty.get("shape")
            if position and position != "no-run":
                score += 1
                signals.append(
                    f"Typology: {shape_label} run in {position} position."
                )

        if extension_forward.get("ok"):
            ef = extension_forward["data"].get("summary", {})
            if int(ef.get("n_anomalous_candidates", 0)) >= 1:
                score += 1
                signals.append(
                    f"Forward extension surfaces "
                    f"{ef['n_anomalous_candidates']} anomalous "
                    f"boundary candidate(s)."
                )

        if extension_backward.get("ok"):
            eb = extension_backward["data"].get("summary", {})
            if int(eb.get("n_anomalous_candidates", 0)) >= 1:
                score += 1
                signals.append(
                    f"Backward extension surfaces "
                    f"{eb['n_anomalous_candidates']} anomalous "
                    f"boundary candidate(s)."
                )

        # Shape evidence as rationale add-on only — does NOT contribute
        # to the score (see docstring).
        if shape_anomaly.get("ok") and shape_anomaly["data"].get("is_anomaly"):
            rank = shape_anomaly["data"].get("delta_rank_pct")
            signals.append(
                f"Chain-shape anomaly also flags "
                f"(delta_rank_pct={rank})."
            )

        if score >= 3:
            strength = "strong"
            action = "escalate to SAR"
        elif score >= 2:
            strength = "moderate"
            action = "continue investigation"
        else:
            strength = "weak"
            action = "false-positive candidate"

        rationale = (
            " ".join(signals) if signals else
            "No coherent investigative signal across the R9 surfaces."
        )
        return {
            "investigation_strength": strength,
            "recommended_action": action,
            "score": score,
            "rationale": rationale,
        }

    def generate_sar_rationale(
        self,
        chain_id: str,
        pattern_id: str,
        *,
        anchor_pattern_id: str,
        evidence: dict[str, Any] | None = None,
        regulatory_template: str = "FinCEN SAR",
    ) -> dict[str, Any]:
        """Compose a SAR-ready narrative from R9 evidence on a single chain.

        Template-based composition over the structured per-step output of
        ``investigate_chain``. When ``evidence`` is None, runs the R9 loop
        server-side first; when supplied, the dict must match the
        ``investigate_chain`` return shape (trace + typology + shape_anomaly
        + extension_forward + extension_backward + summary, each per-step
        wrapped in ``{ok, data | error}``). Caller-supplied evidence is the
        cheap path for repeated narrative generation on the same chain.

        The narrative is composed from structured templates — no LLM call.
        Each paragraph fills placeholders from the corresponding evidence
        block; failed evidence blocks (``ok: False``) are silently skipped
        in the narrative but surface in ``evidence_anchors`` as ``null``.

        Returns dict with ``sar_narrative`` (3-5 paragraph string),
        ``evidence_anchors`` (structured pointers to the source data per
        narrative claim), ``regulatory_template_hint`` (echoes the input
        parameter), ``confidence`` (``high`` / ``moderate`` / ``low``
        derived from investigation_strength + evidence completeness),
        ``chain_id``, ``pattern_id``, ``anchor_pattern_id``, ``elapsed_ms``.

        Honesty discipline: language is "evidence indicates" / "the
        per-hop trace shows" — never "confirms". The narrative is a
        starting point for the investigator's draft, not a final verdict.
        """
        import time as _time
        t0 = _time.perf_counter()

        if evidence is None:
            evidence = self.investigate_chain(
                chain_id, pattern_id,
                anchor_pattern_id=anchor_pattern_id,
            )

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        if anchor_pattern_id not in sphere.patterns:
            raise GDSNavigationError(
                f"anchor pattern not found: {anchor_pattern_id!r}",
            )

        narrative, anchors = self._compose_sar_narrative(
            chain_id, pattern_id, anchor_pattern_id, evidence,
        )
        confidence = self._derive_sar_confidence(evidence)

        elapsed_ms = round((_time.perf_counter() - t0) * 1000.0, 2)
        return {
            "chain_id": chain_id,
            "pattern_id": pattern_id,
            "anchor_pattern_id": anchor_pattern_id,
            "sar_narrative": narrative,
            "evidence_anchors": anchors,
            "regulatory_template_hint": regulatory_template,
            "confidence": confidence,
            "elapsed_ms": elapsed_ms,
        }

    @staticmethod
    def _compose_sar_narrative(
        chain_id: str,
        pattern_id: str,
        anchor_pattern_id: str,
        evidence: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Build the paragraph-structured SAR narrative + structured
        evidence_anchors block from an investigate_chain output dict.

        Returns ``(narrative, anchors)`` where narrative is a
        newline-separated 3-5 paragraph string and anchors is a dict of
        structured pointers to the source data per narrative claim
        (each pointer null when the corresponding R9 surface failed).
        """
        paragraphs: list[str] = []
        anchors: dict[str, Any] = {}

        # P1 — opening + chain identification + typology one-liner.
        typology_block = evidence.get("typology", {})
        if typology_block.get("ok"):
            ty = typology_block["data"].get("typology", {})
            shape = ty.get("shape", "unspecified")
            position = ty.get("position_in_chain", "unspecified")
            run_length = ty.get("run_length", 0)
            anchors["typology_axes"] = {
                "shape": shape,
                "position_in_chain": position,
                "peak_position": ty.get("peak_position"),
                "run_length": run_length,
                "dominant_top_dim": ty.get("dominant_top_dim"),
            }
            paragraphs.append(
                f"Chain {chain_id} in pattern '{pattern_id}' is "
                f"classified as a {shape} run in {position} position "
                f"with run length {run_length}, member entities drawn "
                f"from anchor pattern '{anchor_pattern_id}'."
            )
        else:
            anchors["typology_axes"] = None
            paragraphs.append(
                f"Chain {chain_id} in pattern '{pattern_id}' (member "
                f"entities from '{anchor_pattern_id}'). Typology "
                f"classification was not available."
            )

        # P2 — per-hop trace evidence.
        trace_block = evidence.get("trace", {})
        if trace_block.get("ok"):
            t = trace_block["data"]
            tsum = t.get("summary", {})
            n_hops = int(tsum.get("n_hops", 0))
            n_anom = int(tsum.get("n_anomalous", 0))
            run_len = int(tsum.get("max_run_length_same_top_dim", 0))
            top_dim = tsum.get("dominant_top_dim", "unspecified")
            anchors["per_hop_trace"] = {
                "n_hops": n_hops,
                "n_anomalous": n_anom,
                "max_run_length_same_top_dim": run_len,
                "dominant_top_dim": top_dim,
            }
            paragraphs.append(
                f"The per-hop trace covers {n_hops} member entities; "
                f"{n_anom} are individually flagged as anomalous on "
                f"the entity anchor pattern. The longest contiguous "
                f"run of consecutive members sharing the same dominant "
                f"delta dimension is {run_len} hop(s) on "
                f"`{top_dim}` — the operative coherent-cascade signal "
                f"for this chain."
            )
        else:
            anchors["per_hop_trace"] = None

        # P3 — boundary extension evidence (forward + backward combined).
        ext_fwd = evidence.get("extension_forward", {})
        ext_bwd = evidence.get("extension_backward", {})
        ext_pieces: list[str] = []
        ext_anchors: dict[str, Any] = {}
        for direction, block in (("forward", ext_fwd), ("backward", ext_bwd)):
            if block.get("ok"):
                d = block["data"]
                bsum = d.get("summary", {})
                n_cands = int(bsum.get("n_candidates", 0))
                n_anom_c = int(bsum.get("n_anomalous_candidates", 0))
                ext_anchors[direction] = {
                    "boundary_key": d.get("boundary_key"),
                    "boundary_position": d.get("boundary_position"),
                    "n_candidates": n_cands,
                    "n_anomalous_candidates": n_anom_c,
                }
                if n_cands > 0:
                    ext_pieces.append(
                        f"the {direction} boundary at "
                        f"{d.get('boundary_key', '?')} surfaces "
                        f"{n_cands} extension candidate(s) "
                        f"({n_anom_c} individually anomalous)"
                    )
            else:
                ext_anchors[direction] = None
        anchors["boundary_extensions"] = ext_anchors
        if ext_pieces:
            paragraphs.append(
                "Boundary analysis: " + "; ".join(ext_pieces) + "."
            )

        # P4 — chain-shape corroboration.
        shape_block = evidence.get("shape_anomaly", {})
        if shape_block.get("ok"):
            s = shape_block["data"]
            is_anom = bool(s.get("is_anomaly"))
            rank = s.get("delta_rank_pct")
            anchors["chain_shape_anomaly"] = {
                "is_anomaly": is_anom,
                "delta_norm": s.get("delta_norm"),
                "delta_rank_pct": rank,
            }
            if is_anom:
                paragraphs.append(
                    f"The chain-level shape (chain-feature delta) is "
                    f"independently anomalous at delta_rank_pct={rank} "
                    f"— corroborating evidence that the structural "
                    f"profile of the chain itself, not only the "
                    f"composition of its members, deviates from the "
                    f"chain population."
                )
            else:
                paragraphs.append(
                    f"The chain-level shape (chain-feature delta) is "
                    f"NOT independently anomalous "
                    f"(delta_rank_pct={rank}); the investigative signal "
                    f"comes from the composition of member entities, "
                    f"not the chain's own feature profile. This is the "
                    f"orthogonal-detector regime — chains that "
                    f"`find_anomalies(<chain_pattern>)` would miss."
                )
        else:
            anchors["chain_shape_anomaly"] = None

        # P5 — strength + recommended action, OR untriaged guard.
        # Count R9 surfaces that actually returned evaluable evidence.
        # When ALL fail, the strength/action derived from the empty
        # summary would render as "weak / false-positive candidate" —
        # in a SAR context that text reads "we evaluated and found
        # the chain clear", which is the worst silent-error class.
        # Replace with an explicit untriaged guard so the investigator
        # cannot paste "false-positive" on a chain that was never
        # actually checked.
        ok_count = sum(
            1 for k in ("trace", "typology", "shape_anomaly",
                        "extension_forward", "extension_backward")
            if evidence.get(k, {}).get("ok")
        )
        summary_block = evidence.get("summary")
        if ok_count == 0:
            anchors["summary"] = None
            paragraphs.append(
                "Investigation could not complete — none of the R9 "
                "surfaces returned evaluable evidence (see "
                "`evidence_anchors` for per-step error status). Treat "
                "this chain as **untriaged**, not as cleared. The "
                "narrative above is the chain identification only; no "
                "investigative finding was produced."
            )
        else:
            if not summary_block:
                anchors["summary"] = None
                strength = "weak"
                action = "false-positive candidate"
                score = 0
            else:
                strength = summary_block.get("investigation_strength", "weak")
                action = summary_block.get(
                    "recommended_action", "false-positive candidate",
                )
                score = int(summary_block.get("score", 0))
                rationale = summary_block.get("rationale", "")
                anchors["summary"] = {
                    "investigation_strength": strength,
                    "recommended_action": action,
                    "score": score,
                    "rationale": rationale,
                }
            paragraphs.append(
                f"Aggregating across the R9 surfaces, the investigation "
                f"strength is {strength} (composition score {score}/4), "
                f"recommended action: {action}."
            )

        narrative = "\n\n".join(paragraphs)
        return narrative, anchors

    @staticmethod
    def _derive_sar_confidence(evidence: dict[str, Any]) -> str:
        """Map investigation_strength + evidence completeness to a
        coarse confidence band.

        Rules:
          - strong + 5 R9 surfaces ok               → high
          - strong + 4 surfaces ok                  → moderate
          - moderate + 4-5 surfaces ok              → moderate
          - everything else (weak strength, or
            strong/moderate with too-few surfaces)  → low

        The strong+4 rule prevents the contradiction "confidence=low,
        recommended_action='escalate to SAR'" — a chain scoring
        composition signals at strong but with one R9 surface failing
        (e.g. extension_backward errored on a chain whose run starts
        at hop 0 — normal) gets confidence=moderate, not low.

        Evidence completeness: count of per-step blocks where ok=True.
        """
        summary = evidence.get("summary", {})
        strength = summary.get("investigation_strength", "weak")
        ok_count = sum(
            1 for k in ("trace", "typology", "shape_anomaly",
                        "extension_forward", "extension_backward")
            if evidence.get(k, {}).get("ok")
        )
        if strength == "strong" and ok_count == 5:
            return "high"
        if strength == "strong" and ok_count == 4:
            return "moderate"
        if strength == "moderate" and ok_count >= 4:
            return "moderate"
        return "low"

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

        # Reliability flags — surface single_dim_driven + low_confidence_bucket
        # using the same per-dim contribution machinery that top_dimensions
        # routes through, so dominant_dim agrees with explain output.
        from hypertopos.engine.geometry import compute_reliability_flags
        explanation["reliability_flags"] = compute_reliability_flags(
            polygon.delta,
            pattern=pattern,
            anomaly_confidence=polygon.anomaly_confidence,
        )

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

    def find_diverse_explanations(
        self,
        primary_key: str,
        *,
        pattern_id: str,
        n_hypotheses: int = 3,
        min_contribution_pct: float = 0.10,
        validate: bool = False,
    ) -> dict[str, Any]:
        """Diverse-cover explanation: up to K disjoint hypotheses, each
        a small set of dimensions that jointly explain at least
        ``min_contribution_pct`` of the entity's anomaly mass.

        Where ``explain_anomaly`` ranks every dim by Bregman
        contribution and returns the full ordered list,
        ``find_diverse_explanations`` partitions the contribution
        budget into ``n_hypotheses`` strictly disjoint subsets via a
        greedy max-K cover. Useful when the investigator wants several
        alternative narratives instead of one ranking — e.g. "is this
        account anomalous because of structuring volume, or because of
        unusual counterparty mix, or both?".

        Hypotheses are strictly disjoint: each dim appears in at most
        one hypothesis. The post-hoc ``diversity_score`` is the mean
        pairwise ``1 - jaccard`` over the emitted dim sets — under the
        disjoint constraint this is always ``1.0`` when at least two
        hypotheses are returned (the field is kept for cross-call
        comparability and for the degenerate single-hypothesis case,
        where it is ``None``).

        Each hypothesis carries the dim labels it covers, the joint
        contribution percentage, and a short narrative built from its
        top-2 dims. When ``validate=True`` each hypothesis is replayed
        through ``simulate_dimension_change`` (setting every dim in the
        hypothesis to its population mu) so the caller can see whether
        neutralising that subset alone would clear the anomaly flag.

        Greedy cover is deterministic: the same ``contributions`` /
        ``min_contribution_pct`` pair always produces the same dim
        sets.

        Args:
            primary_key: anomalous entity to explain.
            pattern_id: pattern whose geometry holds the entity row.
            n_hypotheses: requested number of hypotheses ``K``.
            min_contribution_pct: joint-share floor a hypothesis must
                clear before it is emitted, in ``[0, 1]``.
            validate: when ``True``, replay each hypothesis through
                ``simulate_dimension_change`` and attach a
                ``validation`` block with the post-override
                ``delta_norm`` and the anomaly-flag transition.

        Returns:
            Dict echoing ``primary_key`` / ``pattern_id`` /
            ``delta_norm`` / ``theta_norm``, the requested vs returned
            hypothesis counts, the ``hypotheses`` list (each entry
            with ``hypothesis_id`` / ``dim_labels`` /
            ``joint_contribution_pct`` / ``narrative``, an optional
            ``validation`` block, and — for graceful-degrade
            secondaries — ``is_degraded=True``), a
            ``diversity_score`` mean pairwise ``1 - jaccard``
            (``None`` when fewer than two hypotheses were returned —
            no pair to compare), and a ``degraded_reason`` field
            (``None`` / ``"insufficient_diverse_mass"`` /
            ``"capped_to_dim_count"`` /
            ``"diversity_unavailable_top1_only"``).

            ``"diversity_unavailable_top1_only"`` fires when a single
            dim covers so much mass that no second hypothesis can
            clear ``min_contribution_pct``; the response carries one
            primary hypothesis (passes the floor) plus one secondary
            (``is_degraded=True``, ``joint_contribution_pct`` below
            the floor) so the agent at least sees the runner-up dim.
        """
        from hypertopos.engine.explanation import (
            _jaccard,
            submodular_diverse_cover,
        )
        from hypertopos.engine.geometry import _per_dim_anomaly_contributions

        if n_hypotheses < 1:
            raise GDSNavigationError(
                f"n_hypotheses must be >= 1; got {n_hypotheses}",
            )
        if min_contribution_pct < 0.0 or min_contribution_pct > 1.0:
            raise GDSNavigationError(
                f"min_contribution_pct must be in [0, 1]; got {min_contribution_pct}",
            )

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        pattern = sphere.patterns[pattern_id]

        version = self._resolve_version(pattern_id)
        tbl = self._storage.read_geometry(
            pattern_id,
            version,
            primary_key=primary_key,
            columns=["primary_key", "delta", "delta_norm"],
        )
        if tbl.num_rows == 0:
            raise GDSNavigationError(
                f"entity {primary_key!r} not found in {pattern_id!r} v{version}",
            )

        delta_arr = np.asarray(tbl["delta"][0].as_py(), dtype=np.float64)
        delta_norm = float(tbl["delta_norm"][0].as_py())
        theta_norm = (
            float(np.linalg.norm(pattern.theta))
            if pattern.theta is not None
            else 0.0
        )

        n_hypotheses_original = n_hypotheses
        n_dims = len(pattern.dim_labels)
        cap_degraded_reason: str | None = None
        if n_hypotheses > n_dims:
            n_hypotheses = n_dims
            cap_degraded_reason = "capped_to_dim_count"

        contributions = _per_dim_anomaly_contributions(
            delta_arr,
            dimension_kinds=pattern.dimension_kinds,
            sigma=pattern.sigma_diag,
            mu=pattern.mu,
            dimension_weights=pattern.dimension_weights,
        )

        dim_sets, alg_degraded_reason = submodular_diverse_cover(
            contributions,
            pattern.dim_labels,
            n_hypotheses=n_hypotheses,
            min_contribution_pct=min_contribution_pct,
        )

        # Graceful-degrade fallback: when a single dim dominates so much
        # of the mass that no second hypothesis can clear the floor, the
        # cover returns exactly one hypothesis and the agent never sees
        # the next-rank dim at all.  Emit a *secondary* hypothesis that
        # contains the highest-contribution dim from outside the
        # primary, flagged as degraded with its actual joint share
        # (which is below the floor by construction — that's why the
        # cover dropped it).  This preserves the disjoint-set invariant
        # of the primary cover (each dim still appears in at most one
        # hypothesis) and lets the agent at least name the runner-up
        # dim instead of silently truncating to top-1.
        #
        # Trigger conditions:
        #   * exactly one primary hypothesis was emitted
        #   * the user asked for at least two
        #   * cap-degradation did not fire (capped-to-dim-count is a
        #     separate, more upstream problem)
        #   * at least one dim outside the primary carries positive mass
        diversity_unavailable_secondary: dict[str, Any] | None = None
        if (
            len(dim_sets) == 1
            and n_hypotheses >= 2
            and cap_degraded_reason is None
        ):
            primary_dims = dim_sets[0]
            remaining = [
                d for d in range(len(contributions))
                if d not in primary_dims and contributions[d] > 0.0
            ]
            if remaining:
                best_remaining = max(remaining, key=lambda d: contributions[d])
                diversity_unavailable_secondary = {
                    "dim_index": best_remaining,
                    "contribution": float(contributions[best_remaining]),
                }
                # Override the algorithm's degradation flag to signal
                # the more specific failure mode.
                alg_degraded_reason = "diversity_unavailable_top1_only"

        # Cap-degradation takes priority over alg-degradation when both
        # would apply — the user asked for more hypotheses than dims,
        # which is the more upstream problem to flag.
        if cap_degraded_reason is not None:
            degraded_reason = cap_degraded_reason
        else:
            degraded_reason = alg_degraded_reason

        total_mass = float(contributions.sum()) or 1.0
        hypotheses: list[dict[str, Any]] = []
        for h_idx, dim_set in enumerate(dim_sets, start=1):
            indices = sorted(dim_set)
            joint_mass = float(contributions[indices].sum())
            joint_pct = round(100.0 * joint_mass / total_mass, 4)
            labels = [pattern.dim_labels[i] for i in indices]
            top_2_labels = [
                pattern.dim_labels[i]
                for i in sorted(indices, key=lambda j: -contributions[j])[:2]
            ]
            narrative = f"driven by {', '.join(top_2_labels)}"
            entry: dict[str, Any] = {
                "hypothesis_id": h_idx,
                "dim_labels": labels,
                "joint_contribution_pct": joint_pct,
                "narrative": narrative,
            }
            if validate:
                # Override every dim in the hypothesis to its
                # population mu and re-check the anomaly flag. The
                # call can raise GDSNavigationError on legitimate
                # input-shape mismatches (e.g. event patterns whose
                # primary_key is not the line_id we expect), so we
                # catch narrowly and attach the error string rather
                # than letting it sink the whole response.
                set_dim = {
                    pattern.dim_labels[i]: float(pattern.mu[i])
                    for i in indices
                }
                try:
                    sim = self.simulate_dimension_change(
                        primary_key,
                        pattern_id=pattern_id,
                        line_id=primary_key,
                        set_dimension=set_dim,
                    )
                    entry["validation"] = {
                        "delta_norm_after_override": sim["delta_norm_after"],
                        "neutralizes_anomaly": not sim["is_anomaly_after"],
                    }
                except GDSNavigationError as exc:
                    entry["validation"] = {"error": str(exc)}
            hypotheses.append(entry)

        # Attach the graceful-degrade secondary hypothesis (see the
        # block above ``submodular_diverse_cover``).  It is emitted
        # *after* the disjoint primary cover so the primary's
        # joint_contribution_pct math is not perturbed.  The secondary
        # always carries ``is_degraded=True`` and its own
        # ``joint_contribution_pct`` (below floor by construction).
        if diversity_unavailable_secondary is not None:
            dim_idx = diversity_unavailable_secondary["dim_index"]
            sec_mass = diversity_unavailable_secondary["contribution"]
            sec_pct = round(100.0 * sec_mass / total_mass, 4)
            sec_label = pattern.dim_labels[dim_idx]
            sec_entry: dict[str, Any] = {
                "hypothesis_id": len(hypotheses) + 1,
                "dim_labels": [sec_label],
                "joint_contribution_pct": sec_pct,
                "narrative": (
                    f"driven by {sec_label} "
                    f"(below {min_contribution_pct * 100:.0f}% floor — "
                    f"reported for visibility)"
                ),
                "is_degraded": True,
            }
            hypotheses.append(sec_entry)
            # Mirror the secondary dim into ``dim_sets`` so the
            # diversity_score branch below treats it as a real
            # hypothesis for pair-distance accounting.
            dim_sets = list(dim_sets) + [{dim_idx}]

        # Diversity score = mean pairwise (1 − Jaccard) over the
        # emitted dim sets. With fewer than two hypotheses there is no
        # pair to compare, so we return ``None`` rather than a numeric
        # default — 0.0 would falsely imply "perfectly identical" and
        # 1.0 would falsely imply "maximally diverse", both lies about
        # a degenerate single-hypothesis case.
        if len(dim_sets) < 2:
            diversity_score: float | None = None
        else:
            distances: list[float] = []
            for i in range(len(dim_sets)):
                for j in range(i + 1, len(dim_sets)):
                    distances.append(1.0 - _jaccard(dim_sets[i], dim_sets[j]))
            diversity_score = round(float(np.mean(distances)), 4)

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "delta_norm": round(delta_norm, 4),
            "theta_norm": round(theta_norm, 4),
            "n_hypotheses_requested": n_hypotheses_original,
            "n_hypotheses_returned": len(hypotheses),
            "hypotheses": hypotheses,
            "diversity_score": diversity_score,
            "degraded_reason": degraded_reason,
        }

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
            # Temporal shape_snapshot carries structural dims only; aggregated
            # edge_dim dims have no temporal history. Slice calibration arrays
            # to the snapshot width before the broadcast.
            if pattern.sigma_diag is not None:
                _w = shapes_arr.shape[-1]
                _sigma = np.maximum(pattern.sigma_diag[:_w], 1e-2)
                deltas = (shapes_arr - pattern.mu[:_w]) / _sigma
            else:
                deltas = shapes_arr
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

    def classify_trajectory(
        self,
        primary_key: str,
        pattern_id: str,
        sample_size: int = 10_000,
    ) -> dict[str, Any]:
        """Categorise one entity's temporal trajectory vs the population.

        Combines the DTW distance from
        ``engine.topology.classify_trajectory`` (population-median reference)
        with a first-derivative slope comparison to surface one of
        ``outlier`` / ``lagging`` / ``leading`` / ``typical``.

        Args:
            primary_key: entity to classify.
            pattern_id: anchor pattern with temporal history.
            sample_size: cap on entities sampled for the median trajectory and
                DTW threshold estimation. Default 10 000.

        Returns:
            ``{primary_key, dtw_distance, category, category_evidence}`` for
            the requested entity. ``category`` is ``"unknown"`` when the
            entity is absent from temporal storage.
        """
        from hypertopos.engine.topology import (
            classify_trajectory as _engine_classify_trajectory,
        )

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            raise GDSNavigationError(
                f"Pattern '{pattern_id}' not found in sphere."
            )
        if pattern.pattern_type == "event":
            raise ValueError(
                f"classify_trajectory requires anchor pattern — "
                f"event patterns have no temporal history. "
                f"Got pattern '{pattern_id}' with type 'event'."
            )

        # Stream temporal data, convert shape_snapshot to delta_snapshot via
        # pattern calibration (same path as detect_trajectory_anomaly).
        entity_slices: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
        try:
            for batch in self._storage.read_temporal_batched(pattern_id):
                table = pa.Table.from_batches([batch])
                if "shape_snapshot" not in table.schema.names:
                    continue
                pks = table["primary_key"].to_pylist()
                snapshots = table["shape_snapshot"].to_pylist()
                timestamps = pc.cast(table["timestamp"], pa.int64()).to_pylist()
                for pk, snap, ts in zip(pks, snapshots, timestamps, strict=True):
                    if pk is None or snap is None:
                        continue
                    entity_slices[pk].append((ts, snap))
                if len(entity_slices) >= sample_size:
                    break
        except StopIteration:
            pass

        if not entity_slices:
            return {
                "primary_key": primary_key,
                "dtw_distance": 0.0,
                "category": "unknown",
                "category_evidence": 0.0,
            }

        # Convert shape_snapshot → delta_snapshot via pattern (mu, sigma_diag).
        sigma = (
            np.maximum(pattern.sigma_diag, 1e-2)
            if pattern.sigma_diag is not None else None
        )
        mu = pattern.mu

        pks_flat: list[str] = []
        snaps_flat: list[list[float]] = []
        for pk, slices in entity_slices.items():
            slices.sort(key=lambda x: x[0])
            for _ts, snap in slices:
                arr = np.asarray(snap, dtype=np.float32)
                if sigma is not None:
                    w = arr.shape[0]
                    delta = ((arr - mu[:w]) / sigma[:w]).astype(np.float32)
                else:
                    delta = arr
                pks_flat.append(pk)
                snaps_flat.append(delta.tolist())

        solid_tbl = pa.table({
            "primary_key": pks_flat,
            "delta_snapshot": snaps_flat,
        })
        results = _engine_classify_trajectory(solid_tbl, sample_size=sample_size)
        for entry in results:
            if entry["primary_key"] == primary_key:
                return entry
        return {
            "primary_key": primary_key,
            "dtw_distance": 0.0,
            "category": "unknown",
            "category_evidence": 0.0,
        }

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

    def find_topological_anomalies(
        self,
        pattern_id: str,
        *,
        top_n: int = 20,
        force: bool = False,
        sample_size: int = 50_000,
        k_neighbors: int = 50,
        homology_dim: int = 1,
        pca_dim: int = 10,
    ) -> list[dict[str, Any]]:
        """Per-entity local k-NN VR-filtration H_1 persistence anomaly ranking.

        Surfaces entities whose local geometric neighborhood carries a cycle
        signature that the population-level delta_norm rank misses. Results
        are cached per ``(pattern_id, pattern_version)`` in a sidecar Lance
        store under ``_gds_meta/topology_cache/anomalies/{pid}/v={N}.lance``;
        a pattern re-calibration writes a fresh version file and the stale
        one is collected by the sphere GC pass.

        Args:
            pattern_id: pattern whose entities are scored. Event patterns
                resolve to their anchor pattern for geometry lookup.
            top_n: number of top-score entities returned.
            force: when True, bypass the cache and recompute.
            sample_size: cap on entities loaded from geometry and scored.
            k_neighbors: size of each entity's local cloud passed to ripser.
            homology_dim: max homology dimension computed (default 1).
            pca_dim: PCA target dim when geometry dim is larger.

        Returns:
            ``top_n`` entries sorted by ``topo_score`` descending, each with
            ``primary_key``, ``topo_score``, ``h1_max_persistence``,
            ``h0_mean_death``, ``n_h1_features``, ``computed_at``.
        """
        from hypertopos.engine.topology import (
            find_topological_anomalies as _engine_find_topological_anomalies,
        )
        from hypertopos.storage.topology_cache import (
            ANOMALIES_SCHEMA,
            cache_path,
            read_cache,
            write_cache,
        )

        scoring_pattern = (
            self._resolve_anchor_pattern_for_scoring(pattern_id) or pattern_id
        )
        version = self._resolve_version(scoring_pattern)
        cpath = cache_path(
            self._storage._base, "anomalies", scoring_pattern, version,
        )

        if not force:
            cached = read_cache(cpath)
            if cached is not None and cached.num_rows >= top_n:
                sorted_tbl = cached.sort_by([("h1_max_persistence", "descending")])
                return sorted_tbl.slice(0, top_n).to_pylist()

        geo = self._storage.read_geometry(
            scoring_pattern, version,
            columns=["primary_key", "delta"],
            sample_size=sample_size,
        )
        if geo.num_rows == 0:
            return []

        delta_np = np.asarray(geo["delta"].to_pylist(), dtype=np.float64)

        flat_cols: dict[str, pa.Array] = {"primary_key": geo["primary_key"]}
        for j in range(delta_np.shape[1]):
            flat_cols[f"d{j}"] = pa.array(delta_np[:, j], type=pa.float64())
        flat = pa.table(flat_cols)

        rows = _engine_find_topological_anomalies(
            flat,
            top_n=top_n,
            sample_size=sample_size,
            k_neighbors=k_neighbors,
            homology_dim=homology_dim,
            pca_dim=pca_dim,
        )

        write_cache(cpath, rows, ANOMALIES_SCHEMA)
        return rows

    def simulate_edge_removal(
        self,
        primary_key: str,
        *,
        pattern_id: str,
        line_id: str,
        top_n: int = 5,
        edge_ids: list[str] | None = None,
        max_edges_loaded: int = 2000,
    ) -> list[dict[str, Any]]:
        """Per-edge counterfactual: rank the entity's edges by their
        contribution to ``delta_norm``.

        Covers two dim classes:

        - ``relations`` — count-based closed-form (each matching edge
          contributes ``+1`` to its relation dim).
        - ``edge_dim_aggregations`` — pyarrow-aligned aggregation rescan
          across the four supported aggregations (``mean`` / ``max`` /
          ``std`` / ``p95``). ``count_above_threshold`` aggregations
          require population-level threshold lookup and are reported in
          ``dimensions_skipped`` (held constant).

        ``event_dimensions`` and ``prop_columns`` dim classes are
        unchanged-by-design (no per-edge contribution by construction —
        event_dim is scalar-per-event, prop_columns are static entity
        attributes).
        """
        from hypertopos.engine.counterfactual import (
            simulate_edge_removal_with_aggregations,
        )

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        pattern = sphere.patterns[pattern_id]

        version = self._resolve_version(pattern_id)
        tbl = self._storage.read_geometry(
            pattern_id, version,
            primary_key=primary_key,
            columns=["primary_key", "delta", "delta_norm"],
        )
        if tbl.num_rows == 0:
            raise GDSNavigationError(
                f"entity {primary_key!r} not found in {pattern_id!r} v{version}",
            )

        delta = np.asarray(tbl["delta"][0].as_py(), dtype=np.float32)
        delta_norm = float(tbl["delta_norm"][0].as_py())

        relations_for_engine: list[dict[str, Any]] = []
        for rel in pattern.relations:
            relations_for_engine.append({
                "line_id": rel.line_id,
                "direction": rel.direction,
            })

        # Recover full shape vector from delta + mu + sigma_diag.
        if delta.shape[0] != len(pattern.mu):
            raise GDSNavigationError(
                f"entity delta length {delta.shape[0]} does not match "
                f"pattern mu length {len(pattern.mu)}; "
                "geometry may be corrupted or pattern version mismatched",
            )
        delta_full = delta.astype(np.float64)
        mu_full = np.asarray(pattern.mu, dtype=np.float64)
        sigma_full = np.asarray(pattern.sigma_diag, dtype=np.float64)
        shape_full = delta_full * sigma_full + mu_full

        # Load entity's events from event-pattern edge table + edge_features
        # sidecar for the edge_dim_aggregations dispatch.
        edge_agg_specs: list[tuple[str, str]] = []
        edge_agg_dim_offset = (
            len(pattern.relations)
            + len(pattern.event_dimensions)
            + len(pattern.prop_columns)
        )
        event_source_values: dict[str, dict[str, float]] = {}
        edges_for_engine: list[dict[str, Any]] = []

        if pattern.edge_dim_aggregations is not None:
            agg = pattern.edge_dim_aggregations
            event_pattern_id = agg.from_event_pattern
            per_dim = agg.aggregates_per_dim or {}
            for source_dim in (agg.dims or ()):
                for agg_kind in per_dim.get(source_dim, ()):
                    edge_agg_specs.append((source_dim, agg_kind))

            # Read event-pattern edge table for this entity's events.
            event_pattern = sphere.patterns.get(event_pattern_id)
            if event_pattern is not None:
                try:
                    event_adj = self._storage.get_adjacency(event_pattern_id)
                    for edge_tuple in event_adj.neighbors_out(primary_key):
                        partner = str(edge_tuple[0])
                        event_key = (
                            str(edge_tuple[3]) if len(edge_tuple) >= 4
                            else f"{primary_key}->{partner}"
                        )
                        edges_for_engine.append({
                            "edge_id": event_key,
                            "partner_key": partner,
                            "direction": "out",
                            "line_id": event_pattern_id,
                        })
                    for edge_tuple in event_adj.neighbors_in(primary_key):
                        partner = str(edge_tuple[0])
                        event_key = (
                            str(edge_tuple[3]) if len(edge_tuple) >= 4
                            else f"{partner}->{primary_key}"
                        )
                        edges_for_engine.append({
                            "edge_id": event_key,
                            "partner_key": partner,
                            "direction": "in",
                            "line_id": event_pattern_id,
                        })
                except Exception:  # noqa: BLE001
                    edges_for_engine = []

            # Truncate BEFORE the sidecar IN-clause — for a hub entity with
            # 168 k transactions the sidecar SQL `event_key IN (...)` string
            # alone is multi-megabyte and Lance's filter parser bogs down.
            if len(edges_for_engine) > max_edges_loaded:
                edges_for_engine = edges_for_engine[:max_edges_loaded]

            # Read per-event source-dim values from edge_features sidecar +
            # compute per-source-dim population p95 thresholds for the
            # count_above_threshold aggregation. The thresholds are cached
            # per (event_pattern_id) on the navigator instance — repeated
            # calls for entities under the same pattern reuse them.
            if edges_for_engine and (agg.dims or ()):
                try:
                    sidecar_path = (
                        self._storage._base / "_gds_meta" / "edge_features"
                        / event_pattern_id / "data.lance"
                    )
                    if sidecar_path.exists():
                        import lance as _lance_local
                        import pyarrow.compute as _pc_local
                        ds = _lance_local.dataset(str(sidecar_path))
                        event_keys = [e["edge_id"] for e in edges_for_engine]
                        escaped = ", ".join(
                            f"'{k.replace(chr(39), chr(39)*2)}'"
                            for k in event_keys
                        )
                        cols = ["event_key", *agg.dims]
                        scanner = ds.scanner(
                            columns=cols,
                            filter=f"event_key IN ({escaped})",
                        )
                        sidecar_tbl = scanner.to_table()
                        for i in range(sidecar_tbl.num_rows):
                            ek = sidecar_tbl["event_key"][i].as_py()
                            event_source_values[ek] = {
                                d: float(sidecar_tbl[d][i].as_py() or 0.0)
                                for d in agg.dims
                            }

                        # Joint cache: per-event-pattern p95 thresholds AND
                        # sorted population samples (ECDF) for significance
                        # p-values. One full-column scan per source_dim
                        # amortised across both downstream consumers.
                        cache = self.__dict__.setdefault(
                            "_counterfactual_population_cache", {},
                        )
                        if event_pattern_id not in cache:
                            per_dim_thresholds: dict[str, float] = {}
                            per_dim_ecdfs: dict[str, np.ndarray] = {}
                            for source_dim in agg.dims:
                                col_tbl = ds.scanner(columns=[source_dim]).to_table()
                                col = col_tbl[source_dim]
                                # Threshold via pc.quantile (exact).
                                q_arr = _pc_local.quantile(col, q=0.95)
                                raw = (
                                    q_arr.as_py()
                                    if hasattr(q_arr, "as_py")
                                    else q_arr.to_pylist()
                                )
                                q = raw[0] if isinstance(raw, list) else raw
                                if q is not None:
                                    qf = float(q)
                                    if math.isfinite(qf):
                                        per_dim_thresholds[source_dim] = qf
                                # ECDF: sample down to <=100k for resolution
                                # 1e-5 in p-value, then ascending-sort.
                                col_np = np.asarray(col, dtype=np.float64)
                                col_np = col_np[np.isfinite(col_np)]
                                if col_np.size > 100_000:
                                    rng = np.random.default_rng(seed=0)
                                    col_np = rng.choice(
                                        col_np, size=100_000, replace=False,
                                    )
                                per_dim_ecdfs[source_dim] = np.sort(col_np)
                            cache[event_pattern_id] = {
                                "thresholds": per_dim_thresholds,
                                "ecdfs": per_dim_ecdfs,
                            }
                        cached = cache[event_pattern_id]
                        thresholds_for_engine = cached["thresholds"]
                        population_ecdfs_for_engine = cached["ecdfs"]
                    else:
                        thresholds_for_engine = {}
                        population_ecdfs_for_engine = {}
                except Exception:  # noqa: BLE001
                    thresholds_for_engine = {}
                    population_ecdfs_for_engine = {}
            else:
                thresholds_for_engine = {}
                population_ecdfs_for_engine = {}
        else:
            thresholds_for_engine = {}
            population_ecdfs_for_engine = {}

        # Fall back to line_id adjacency when no edge-pattern derived
        # edges were collected (backwards compatibility for patterns
        # without edge_dim_aggregations).
        if not edges_for_engine:
            adj = self._storage.get_adjacency(line_id)
            for edge_tuple in adj.neighbors_out(primary_key):
                partner = str(edge_tuple[0])
                event_key = (
                    str(edge_tuple[3]) if len(edge_tuple) >= 4
                    else f"{primary_key}->{partner}"
                )
                edges_for_engine.append({
                    "edge_id": event_key,
                    "partner_key": partner,
                    "direction": "out",
                    "line_id": line_id,
                })
            for edge_tuple in adj.neighbors_in(primary_key):
                partner = str(edge_tuple[0])
                event_key = (
                    str(edge_tuple[3]) if len(edge_tuple) >= 4
                    else f"{partner}->{primary_key}"
                )
                edges_for_engine.append({
                    "edge_id": event_key,
                    "partner_key": partner,
                    "direction": "in",
                    "line_id": line_id,
                })

        # Hard cap on candidate edges to keep per-call latency bounded on
        # hub entities (single accounts can have hundreds of thousands of
        # transactions in the AML HI-small sphere; without this cap the
        # downstream Lance sidecar IN-clause and engine evaluation each
        # scale O(n_edges) and push the call past several minutes).
        if len(edges_for_engine) > max_edges_loaded:
            edges_for_engine = edges_for_engine[:max_edges_loaded]

        # Evaluate ALL candidate edges in the engine — the navigator-side
        # tie-break by min_pvalue needs the full result set, not just the
        # engine's |drop_pct|-truncated top_n. Truncation happens after
        # the tie-break sort below.
        rows = simulate_edge_removal_with_aggregations(
            shape=shape_full,
            mu=mu_full,
            sigma_diag=sigma_full,
            delta_norm=delta_norm,
            edges=edges_for_engine,
            relations=relations_for_engine,
            edge_agg_dim_offset=edge_agg_dim_offset,
            edge_agg_specs=edge_agg_specs,
            event_source_values=event_source_values,
            candidate_edge_ids=edge_ids,
            top_n=10**9,
            thresholds=thresholds_for_engine,
        )

        # Attach dim_label for dominant_dim_idx.
        dim_labels = pattern.dim_labels
        for row in rows:
            idx = row.get("dominant_dim_idx")
            if idx is not None and 0 <= idx < len(dim_labels):
                row["dominant_dim_label"] = dim_labels[idx]
            else:
                row["dominant_dim_label"] = None

        # Phase 2.D — per-edge source-value ECDF p-values. Breaks the
        # within-tied-drop_pct degeneracy on high-volume entities where the
        # raw drop_pct ranking is flat but source values still discriminate.
        if (
            pattern.edge_dim_aggregations is not None
            and population_ecdfs_for_engine
            and edges_for_engine
        ):
            from hypertopos.engine.counterfactual import (
                compute_per_edge_source_value_pvalues,
            )
            source_dims = list(pattern.edge_dim_aggregations.dims or ())
            pvalues_by_edge = compute_per_edge_source_value_pvalues(
                edges=edges_for_engine,
                event_source_values=event_source_values,
                population_ecdfs=population_ecdfs_for_engine,
                source_dims=source_dims,
            )
            for row in rows:
                pdata = pvalues_by_edge.get(row.get("edge_id"))
                if pdata is not None:
                    row["source_value_pvalues"] = {
                        d: pdata[d] for d in source_dims
                    }
                    row["min_pvalue"] = pdata["min_pvalue"]
                    row["dominant_significance_dim"] = pdata[
                        "dominant_significance_dim"
                    ]
                else:
                    row["source_value_pvalues"] = None
                    row["min_pvalue"] = None
                    row["dominant_significance_dim"] = None
        else:
            for row in rows:
                row["source_value_pvalues"] = None
                row["min_pvalue"] = None
                row["dominant_significance_dim"] = None

        # Composite sort by harmonic-style score combining counterfactual
        # impact (`|drop_pct|`) with source-value extremeness (`1 -
        # min_pvalue`). This is NOT a tie-break — it's a co-primary
        # ranking. The simple drop_pct-only sort fails on high-volume
        # entities where many edges share identical |drop_pct| AND
        # identical source values (same value → same p-value), leaving
        # the top-N decorated with one shared p-value across all ties.
        # The composite surfaces edges that have BOTH non-trivial
        # counterfactual impact AND extreme source values vs population.
        # Fallback when p-values unavailable: pure |drop_pct| desc.
        def _composite_score(r: dict[str, Any]) -> float:
            abs_drop = abs(r.get("drop_pct", 0.0))
            mp = r.get("min_pvalue")
            if mp is None:
                return -abs_drop
            extremeness = 1.0 - float(mp)
            # Negative because Python sort is ascending; we want largest
            # composite first. Harmonic-style combiner keeps a high
            # value only when BOTH factors are non-trivial.
            if abs_drop <= 0.0 or extremeness <= 0.0:
                return -abs_drop  # one factor zero → fall back to drop only
            return -2.0 * abs_drop * extremeness / (abs_drop + extremeness)
        rows.sort(key=_composite_score)
        return rows[:top_n]

    def simulate_dimension_change(
        self,
        primary_key: str,
        *,
        pattern_id: str,
        line_id: str,  # noqa: ARG002
        set_dimension: dict[str, float],
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Per-dimension counterfactual: hold the entity fixed in
        shape-space except for one or more overridden dimensions, then
        recompute ``delta``/``delta_norm`` and re-check the anomaly flag.

        Mirrors ``compute_delta`` (cholesky vs diagonal path, optional
        ``dimension_weights``) exactly so the simulated values are
        directly comparable to the persisted geometry row.

        ``line_id`` is kept for signature parity with
        ``simulate_edge_removal``; it is not consumed by this method.

        Returned dict echoes the override, reports
        ``delta_norm_before`` / ``delta_norm_after`` / their percent
        change, the anomaly flag transition, the top witness dims after
        the override ranked by squared-delta contribution, and the
        per-dim before/after audit row for every override key.
        """
        from hypertopos.engine.geometry import GDSEngine as _GE

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        pattern = sphere.patterns[pattern_id]

        version = self._resolve_version(pattern_id)
        tbl = self._storage.read_geometry(
            pattern_id, version,
            primary_key=primary_key,
            columns=["primary_key", "delta", "delta_norm", "is_anomaly"],
        )
        if tbl.num_rows == 0:
            raise GDSNavigationError(
                f"entity {primary_key!r} not found in {pattern_id!r} v{version}",
            )

        delta_before = np.asarray(tbl["delta"][0].as_py(), dtype=np.float64)
        delta_norm_before = float(tbl["delta_norm"][0].as_py())
        is_anomaly_before = bool(tbl["is_anomaly"][0].as_py())

        if not set_dimension:
            raise GDSNavigationError(
                "set_dimension is empty — supply at least one "
                "{dim_label: value} entry to override",
            )

        # Validate dim labels (raise GDSNavigationError listing available
        # labels) and override values (raise listing offenders).
        # ``pattern.dim_index`` covers relations, event_dimensions,
        # prop_columns, and edge_dim_aggregations labels — a single
        # canonical resolution path.
        resolved_indices: dict[str, int] = {}
        for label in set_dimension:
            try:
                resolved_indices[label] = pattern.dim_index(label)
            except ValueError:
                raise GDSNavigationError(
                    f"unknown dim_label {label!r} for pattern "
                    f"{pattern_id!r}; available: {pattern.dim_labels}",
                ) from None

        bad_values = [
            (label, value)
            for label, value in set_dimension.items()
            if not math.isfinite(float(value))
        ]
        if bad_values:
            offenders = ", ".join(
                f"{label}={value!r}" for label, value in bad_values
            )
            raise GDSNavigationError(
                f"set_dimension contains non-finite value(s): {offenders}",
            )

        # Reconstruct shape-space vector that produced delta_before.
        # ``compute_delta`` (engine/geometry.py) is the inverse target;
        # we mirror its two paths.
        mu = np.asarray(pattern.mu, dtype=np.float64)
        if pattern.cholesky_inv is not None:
            # cholesky path: delta = L_inv @ (shape - mu)  ⇒
            # shape = mu + inv(L_inv) @ delta. Reverse dimension_weights
            # first if present (compute_delta multiplies after the
            # whitening) so we recover the pre-weighting delta.
            unweighted_delta = delta_before.copy()
            if pattern.dimension_weights is not None:
                weights = np.asarray(
                    pattern.dimension_weights, dtype=np.float64,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    unweighted_delta = np.where(
                        weights != 0.0,
                        delta_before / weights,
                        0.0,
                    )
            cholesky_forward = np.linalg.inv(
                np.asarray(pattern.cholesky_inv, dtype=np.float64),
            )
            shape_before = mu + cholesky_forward @ unweighted_delta
        else:
            sigma = np.maximum(
                np.asarray(pattern.sigma_diag, dtype=np.float64),
                _GE.SIGMA_EPSILON,
            )
            unweighted_delta = delta_before.copy()
            if pattern.dimension_weights is not None:
                weights = np.asarray(
                    pattern.dimension_weights, dtype=np.float64,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    unweighted_delta = np.where(
                        weights != 0.0,
                        delta_before / weights,
                        0.0,
                    )
            shape_before = unweighted_delta * sigma + mu

        shape_after = shape_before.copy()
        for label, value in set_dimension.items():
            shape_after[resolved_indices[label]] = float(value)

        # Recompute delta_after via the exact compute_delta formula.
        if pattern.cholesky_inv is not None:
            delta_after = np.asarray(
                pattern.cholesky_inv, dtype=np.float64,
            ) @ (shape_after - mu)
        else:
            sigma = np.maximum(
                np.asarray(pattern.sigma_diag, dtype=np.float64),
                _GE.SIGMA_EPSILON,
            )
            delta_after = (shape_after - mu) / sigma
        if pattern.dimension_weights is not None:
            delta_after = delta_after * np.asarray(
                pattern.dimension_weights, dtype=np.float64,
            )

        delta_norm_after = float(np.linalg.norm(delta_after))
        theta_norm = float(
            np.linalg.norm(np.asarray(pattern.theta, dtype=np.float64)),
        )
        is_anomaly_after = theta_norm > 0.0 and delta_norm_after >= theta_norm

        if delta_norm_before > 0.0:
            pct_change = (delta_norm_after - delta_norm_before) / delta_norm_before * 100.0
        else:
            pct_change = float("inf") if delta_norm_after > 0.0 else 0.0

        # Top witness dims after, ranked by squared-delta contribution.
        delta_sq = delta_after ** 2
        total = float(delta_sq.sum())
        order = np.argsort(-delta_sq)[: max(0, int(top_n))]
        dim_labels = pattern.dim_labels
        top_witness_dims_after: list[dict[str, Any]] = []
        for i in order:
            idx = int(i)
            top_witness_dims_after.append({
                "dim_label": dim_labels[idx],
                "dim_index": idx,
                "contribution_pct": round(
                    float(delta_sq[idx]) / max(total, 1e-12) * 100.0,
                    4,
                ),
                "delta": round(float(delta_after[idx]), 4),
            })

        # Audit row for each overridden dim, in original key order.
        dimensions_overridden: list[dict[str, Any]] = []
        for label, value in set_dimension.items():
            idx = resolved_indices[label]
            dimensions_overridden.append({
                "dim_label": label,
                "dim_index": idx,
                "old_value": round(float(shape_before[idx]), 4),
                "new_value": round(float(value), 4),
                "old_delta": round(float(delta_before[idx]), 4),
                "new_delta": round(float(delta_after[idx]), 4),
            })

        return {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "set_dimension": dict(set_dimension),
            "delta_norm_before": delta_norm_before,
            "delta_norm_after": delta_norm_after,
            "delta_norm_pct_change": pct_change,
            "is_anomaly_before": is_anomaly_before,
            "is_anomaly_after": is_anomaly_after,
            "is_anomaly_change": is_anomaly_before != is_anomaly_after,
            "top_witness_dims_after": top_witness_dims_after,
            "dimensions_overridden": dimensions_overridden,
        }

    def select_minimal_joint_edge_removal(
        self,
        primary_key: str,
        *,
        pattern_id: str,
        line_id: str,
        target_drop_pct: float = 50.0,
        k_max: int = 10,
        max_candidates: int = 500,
    ) -> dict[str, Any]:
        """Greedy joint counterfactual: find the smallest set of edges
        whose joint removal drops the entity's ``delta_norm`` by at least
        ``target_drop_pct`` percent.

        Reveals coordinated edge groups (AML laundering rings, motif
        structuring) that single-edge counterfactuals cannot detect.

        Args:
            primary_key: entity to investigate.
            pattern_id: anchor pattern carrying the polygon.
            line_id: line whose adjacency carries the entity's edges
                (back-compat fallback when pattern has no
                ``edge_dim_aggregations`` block).
            target_drop_pct: stop when joint drop reaches this percent
                (default 50%).
            k_max: hard cap on selected set size.
            max_candidates: hard cap on candidate edges before greedy
                search. Default 500 — keeps a hub entity with thousands
                of edges from blowing the per-call wall clock. When the
                entity's adjacency exceeds the cap the surplus edges are
                truncated (adjacency order) and ``candidates_truncated``
                is set in the result.

        Returns:
            ``{primary_key, selected_edge_ids, selected_partner_keys,
            achieved_drop_pct, selection_sequence, target_reached,
            k_max_reached, candidates_truncated, n_candidates_seen,
            n_candidates_used}``.
        """
        from hypertopos.engine.counterfactual import (
            select_minimal_joint_removal,
        )

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise GDSNavigationError(f"pattern not found: {pattern_id!r}")
        pattern = sphere.patterns[pattern_id]
        version = self._resolve_version(pattern_id)
        tbl = self._storage.read_geometry(
            pattern_id, version,
            primary_key=primary_key,
            columns=["primary_key", "delta", "delta_norm"],
        )
        if tbl.num_rows == 0:
            raise GDSNavigationError(
                f"entity {primary_key!r} not found in {pattern_id!r} v{version}",
            )
        delta = np.asarray(tbl["delta"][0].as_py(), dtype=np.float32)
        delta_norm = float(tbl["delta_norm"][0].as_py())
        mu_full = np.asarray(pattern.mu, dtype=np.float64)
        sigma_full = np.asarray(pattern.sigma_diag, dtype=np.float64)
        shape_full = delta.astype(np.float64) * sigma_full + mu_full

        relations_for_engine: list[dict[str, Any]] = [
            {"line_id": rel.line_id, "direction": rel.direction}
            for rel in pattern.relations
        ]
        edge_agg_specs: list[tuple[str, str]] = []
        edge_agg_dim_offset = (
            len(pattern.relations)
            + len(pattern.event_dimensions)
            + len(pattern.prop_columns)
        )
        event_source_values: dict[str, dict[str, float]] = {}
        thresholds_for_engine: dict[str, float] = {}
        edges_for_engine: list[dict[str, Any]] = []

        # Reuse the same data-collection path as simulate_edge_removal —
        # delegate to it via a thin loop. The per-edge call already
        # populates the population cache; here we just call into
        # simulate_edge_removal with top_n large enough to evaluate every
        # candidate, then extract the edge inventory + threshold cache
        # off the navigator instance.
        _ = self.simulate_edge_removal(
            primary_key,
            pattern_id=pattern_id,
            line_id=line_id,
            top_n=10**9,
        )
        # Rebuild context from the warm cache + a fresh adjacency walk.
        # (Engine call below needs raw edges list, event_source_values,
        # and thresholds — all populated when simulate_edge_removal ran.)
        if pattern.edge_dim_aggregations is not None:
            agg = pattern.edge_dim_aggregations
            event_pattern_id = agg.from_event_pattern
            per_dim = agg.aggregates_per_dim or {}
            for source_dim in (agg.dims or ()):
                for agg_kind in per_dim.get(source_dim, ()):
                    edge_agg_specs.append((source_dim, agg_kind))
            cache = self.__dict__.get("_counterfactual_population_cache", {})
            cached = cache.get(event_pattern_id, {})
            thresholds_for_engine = cached.get("thresholds", {})
            # Reload edges + event_source_values for the engine.
            try:
                event_adj = self._storage.get_adjacency(event_pattern_id)
                for et in event_adj.neighbors_out(primary_key):
                    partner = str(et[0])
                    ek = (
                        str(et[3]) if len(et) >= 4
                        else f"{primary_key}->{partner}"
                    )
                    edges_for_engine.append({
                        "edge_id": ek, "partner_key": partner,
                        "direction": "out", "line_id": event_pattern_id,
                    })
                for et in event_adj.neighbors_in(primary_key):
                    partner = str(et[0])
                    ek = (
                        str(et[3]) if len(et) >= 4
                        else f"{partner}->{primary_key}"
                    )
                    edges_for_engine.append({
                        "edge_id": ek, "partner_key": partner,
                        "direction": "in", "line_id": event_pattern_id,
                    })
                sidecar_path = (
                    self._storage._base / "_gds_meta" / "edge_features"
                    / event_pattern_id / "data.lance"
                )
                if sidecar_path.exists() and edges_for_engine and agg.dims:
                    import lance as _lance_local
                    ds = _lance_local.dataset(str(sidecar_path))
                    event_keys = [e["edge_id"] for e in edges_for_engine]
                    escaped = ", ".join(
                        f"'{k.replace(chr(39), chr(39)*2)}'"
                        for k in event_keys
                    )
                    sidecar_tbl = ds.scanner(
                        columns=["event_key", *agg.dims],
                        filter=f"event_key IN ({escaped})",
                    ).to_table()
                    for i in range(sidecar_tbl.num_rows):
                        ek = sidecar_tbl["event_key"][i].as_py()
                        event_source_values[ek] = {
                            d: float(sidecar_tbl[d][i].as_py() or 0.0)
                            for d in agg.dims
                        }
            except Exception:  # noqa: BLE001
                pass

        if not edges_for_engine:
            adj = self._storage.get_adjacency(line_id)
            for et in adj.neighbors_out(primary_key):
                partner = str(et[0])
                ek = str(et[3]) if len(et) >= 4 else f"{primary_key}->{partner}"
                edges_for_engine.append({
                    "edge_id": ek, "partner_key": partner,
                    "direction": "out", "line_id": line_id,
                })
            for et in adj.neighbors_in(primary_key):
                partner = str(et[0])
                ek = str(et[3]) if len(et) >= 4 else f"{partner}->{primary_key}"
                edges_for_engine.append({
                    "edge_id": ek, "partner_key": partner,
                    "direction": "in", "line_id": line_id,
                })

        n_candidates_seen = len(edges_for_engine)
        candidates_truncated = n_candidates_seen > max_candidates
        if candidates_truncated:
            edges_for_engine = edges_for_engine[:max_candidates]

        result = select_minimal_joint_removal(
            shape=shape_full, mu=mu_full, sigma_diag=sigma_full,
            delta_norm=delta_norm,
            candidate_edges=edges_for_engine,
            relations=relations_for_engine,
            edge_agg_dim_offset=edge_agg_dim_offset,
            edge_agg_specs=edge_agg_specs,
            event_source_values=event_source_values,
            target_drop_pct=target_drop_pct,
            k_max=k_max,
            thresholds=thresholds_for_engine,
        )
        result["primary_key"] = primary_key
        result["delta_norm_before"] = delta_norm
        result["n_candidates_seen"] = n_candidates_seen
        result["n_candidates_used"] = len(edges_for_engine)
        result["candidates_truncated"] = candidates_truncated
        return result

    def simulate_counterparty_removal(
        self,
        primary_key: str,
        *,
        pattern_id: str,
        line_id: str,
        top_n: int = 5,
        edge_top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Per-counterparty rollup of per-edge counterfactual contributions.

        Investigator-facing primitive: AML / fraud analysts think
        per-counterparty, not per-transaction. This runs
        ``simulate_edge_removal`` with a large internal cap (so the rollup
        sees the entity's complete edge inventory, not just the top-N),
        then groups by ``edge_partner_key`` and surfaces the partners
        whose collective edges concentrate the most anomaly contribution.

        Args:
            primary_key: entity to investigate.
            pattern_id: anchor pattern carrying the polygon.
            line_id: line whose adjacency carries the entity's edges.
            top_n: cap on returned counterparties (sorted by
                ``sum_abs_drop_pct`` descending).
            edge_top_n: internal cap on per-edge results pre-rollup.
                Default ``None`` = score every candidate edge so the rollup
                is exhaustive over the entity's adjacency.

        Returns:
            List of dicts sorted by ``sum_abs_drop_pct`` descending. Each
            entry: ``partner_key``, ``n_edges``, ``sum_drop_pct``,
            ``sum_abs_drop_pct``, ``max_abs_drop_pct``,
            ``dominant_dim_label``, ``edge_ids``.
        """
        from hypertopos.engine.counterfactual import (
            aggregate_edge_removals_by_counterparty,
        )
        per_edge = self.simulate_edge_removal(
            primary_key,
            pattern_id=pattern_id,
            line_id=line_id,
            top_n=edge_top_n if edge_top_n is not None else 10**9,
        )
        return aggregate_edge_removals_by_counterparty(per_edge, top_n=top_n)

    def find_graph_geometry_tension(
        self,
        primary_key: str,
        *,
        pattern_id: str,
        line_id: str,
        k_geometric: int = 20,
        top_n_hidden: int = 5,
        top_n_suspicious: int = 5,
    ) -> dict[str, Any]:
        """Cross-tabulate behavioural k-NN with graph adjacency for one entity.

        Surfaces two cells of the 2×2 contingency table that scalar anomaly
        detectors cannot separate:

        - **hidden_cluster**: behavioural k-NN entities that do **not** have a
          graph edge to the anchor — "suspicious cohort never seen together",
          the lookalike-fraud signature.
        - **suspicious_links**: entities with a graph edge to the anchor that
          are **not** in the behavioural k-NN — "transacts outside its peer
          group".

        The behavioural k-NN is computed via :py:meth:`find_similar_entities`
        on the entity's delta vector; the graph cohort is the union of incoming
        and outgoing edges in the line's adjacency index.

        Args:
            primary_key: anchor entity.
            pattern_id: pattern whose delta vector defines behavioural similarity.
            line_id: line whose adjacency index defines the graph cohort.
            k_geometric: behavioural k-NN size.
            top_n_hidden: cap on returned hidden_cluster entries.
            top_n_suspicious: cap on returned suspicious_links entries.

        Returns:
            ``{primary_key, hidden_cluster, suspicious_links, tension_score}``
            where ``tension_score = (|hidden_cluster| + |suspicious_links|) /
            k_geometric``.
        """
        similar = self.find_similar_entities(
            primary_key, pattern_id, top_n=k_geometric,
        )
        behav_neighbors: dict[str, float] = {
            str(pk): float(dist) for pk, dist in similar
        }

        adj = self._storage.get_adjacency(line_id)
        edge_counts: dict[str, int] = {}
        for edge in adj.neighbors_out(primary_key):
            nb = str(edge[0])
            edge_counts[nb] = edge_counts.get(nb, 0) + 1
        for edge in adj.neighbors_in(primary_key):
            nb = str(edge[0])
            edge_counts[nb] = edge_counts.get(nb, 0) + 1

        hidden_cluster: list[dict[str, Any]] = []
        for nb_key, dist in behav_neighbors.items():
            if nb_key == primary_key or nb_key in edge_counts:
                continue
            hidden_cluster.append({
                "neighbor_key": nb_key,
                "geometric_distance": dist,
                "edge_present": False,
            })
        n_hidden_total = len(hidden_cluster)
        hidden_cluster.sort(key=lambda r: r["geometric_distance"])
        hidden_cluster = hidden_cluster[:top_n_hidden]

        suspicious_links: list[dict[str, Any]] = []
        for nb_key, n_edges in edge_counts.items():
            if nb_key == primary_key or nb_key in behav_neighbors:
                continue
            suspicious_links.append({
                "neighbor_key": nb_key,
                "geometric_distance": float("inf"),
                "edge_present": True,
                "edge_count": int(n_edges),
            })
        n_suspicious_total = len(suspicious_links)
        suspicious_links.sort(key=lambda r: -r["edge_count"])
        suspicious_links = suspicious_links[:top_n_suspicious]

        if k_geometric <= 0:
            tension_score = 0.0
        else:
            n_total = n_hidden_total + n_suspicious_total
            tension_score = float(n_total) / float(k_geometric)

        return {
            "primary_key": primary_key,
            "hidden_cluster": hidden_cluster,
            "suspicious_links": suspicious_links,
            "tension_score": tension_score,
        }

    def investigate_entity(
        self,
        primary_key: str,
        *,
        pattern_id: str,
        line_id: str,
        chain_pattern_id: str | None = None,
        include_polygon: bool = True,
        include_explain: bool = True,
        include_witness_cohort: bool = True,
        include_chains: bool = True,
        include_root_cause: bool = True,
        include_graph_geometry_tension: bool = True,
        include_per_edge_counterfactual: bool = False,
        include_reliability_flags: bool = True,
        top_n_witnesses: int = 5,
        top_n_chains: int = 3,
        top_n_edges: int = 5,
    ) -> dict[str, Any]:
        """One-call entity investigation orchestrator.

        Chains the existing entity-side primitives (polygon shape lookup,
        explain_anomaly, find_witness_cohort, find_chains_for_entity,
        trace_root_cause, find_graph_geometry_tension) into one aggregated
        report. Each step is wrapped in a safe-call envelope so a partial
        failure on one primitive does not abort the whole investigation —
        the caller sees ``steps_status[step].ok = False`` with the error
        string instead.

        Mirror of ``investigate_chain`` (0.6.7) for the entity-side
        investigation surface.

        Args:
            primary_key: anchor entity.
            pattern_id: anchor pattern for polygon, witness cohort, root cause.
            line_id: edge-bearing pattern for graph geometry tension and the
                eventual per-edge counterfactual.
            chain_pattern_id: optional chain pattern for chain membership
                lookup; when omitted the chains block reports
                ``skipped: no chain_pattern_id provided``.
            include_*: per-step opt-in flags.
            include_per_edge_counterfactual: opt-in for the M1.2 C3
                counterfactual block. The underlying ``simulate_edge_removal``
                primitive is not yet shipped, so when this flag is True the
                block reports ``ok: False`` with an
                ``"simulate_edge_removal not yet available"`` error rather
                than crashing.

        Returns:
            Dict with one block per included step plus ``primary_key``,
            ``pattern_id``, ``line_id``, ``steps_status`` mapping step name
            to ``{ok, error}``, and ``elapsed_ms``.
        """
        import time as _time
        t0 = _time.perf_counter()

        steps_status: dict[str, dict[str, Any]] = {}
        out: dict[str, Any] = {
            "primary_key": primary_key,
            "pattern_id": pattern_id,
            "line_id": line_id,
        }

        import dataclasses as _dc

        def _safe(name: str, fn):
            try:
                data = fn()
                # Unwrap dataclass instances → dict so downstream JSON
                # serialisation doesn't fall through to repr (e.g.
                # WitnessCohortResult / CohortMember leak as a single
                # string otherwise).
                if _dc.is_dataclass(data) and not isinstance(data, type):
                    data = _dc.asdict(data)
                steps_status[name] = {"ok": True, "error": None}
                return data
            except (GDSNavigationError, ValueError, KeyError, AttributeError,
                    NotImplementedError) as exc:
                steps_status[name] = {
                    "ok": False, "error": f"{type(exc).__name__}: {exc}",
                }
                return None

        if include_polygon:
            def _polygon_lookup() -> dict[str, Any]:
                version = self._resolve_version(pattern_id)
                tbl = self._storage.read_geometry(
                    pattern_id, version,
                    primary_key=primary_key,
                    columns=["primary_key", "delta_norm", "is_anomaly",
                             "delta_rank_pct"],
                )
                if tbl.num_rows == 0:
                    raise KeyError(
                        f"entity {primary_key!r} not found in {pattern_id!r}",
                    )
                return {
                    "delta_norm": float(tbl["delta_norm"][0].as_py()),
                    "is_anomaly": bool(tbl["is_anomaly"][0].as_py()),
                    "delta_rank_pct": float(tbl["delta_rank_pct"][0].as_py()),
                }
            out["polygon"] = _safe("polygon", _polygon_lookup)

        # Reliability flags — independent step (not gated on include_explain
        # so callers who skip explain_anomaly still get the dominant_dim /
        # single_dim_driven / low_confidence_bucket caveat metadata). The
        # step builds a fresh polygon; pass include_reliability_flags=False
        # to skip when running investigate_entity in a tight loop where the
        # build cost matters and the polygon block (which carries the
        # already-cheap delta_norm/is_anomaly summary) is enough.
        if include_reliability_flags:
            def _reliability_flags_lookup() -> dict[str, Any]:
                from hypertopos.engine.geometry import compute_reliability_flags
                sphere = self._storage.read_sphere()
                pat = sphere.patterns.get(pattern_id)
                if pat is None:
                    raise KeyError(f"pattern {pattern_id!r} not in sphere")
                poly = self._engine.build_polygon(
                    primary_key, pattern_id, self._manifest,
                )
                return compute_reliability_flags(
                    poly.delta,
                    pattern=pat,
                    anomaly_confidence=poly.anomaly_confidence,
                )
            out["reliability_flags"] = _safe(
                "reliability_flags", _reliability_flags_lookup,
            )

        if include_explain:
            out["explain_anomaly"] = _safe(
                "explain_anomaly",
                lambda: self.explain_anomaly(primary_key, pattern_id),
            )

        if include_witness_cohort:
            out["witness_cohort"] = _safe(
                "witness_cohort",
                lambda: self.find_witness_cohort(
                    primary_key, pattern_id, top_n=top_n_witnesses,
                ),
            )

        if include_chains:
            if chain_pattern_id is None:
                steps_status["chains"] = {
                    "ok": True,
                    "error": None,
                    "skipped": "no chain_pattern_id provided",
                }
                out["chains"] = []
            else:
                out["chains"] = _safe(
                    "chains",
                    lambda: self.find_chains_for_entity(
                        primary_key, chain_pattern_id, top_n=top_n_chains,
                    ).get("chains", []),
                )

        if include_root_cause:
            out["root_cause"] = _safe(
                "root_cause",
                lambda: self.trace_root_cause(primary_key, pattern_id),
            )

        if include_graph_geometry_tension:
            out["graph_geometry_tension"] = _safe(
                "graph_geometry_tension",
                lambda: self.find_graph_geometry_tension(
                    primary_key,
                    pattern_id=pattern_id, line_id=line_id,
                ),
            )

        if include_per_edge_counterfactual:
            out["per_edge_counterfactual"] = _safe(
                "per_edge_counterfactual",
                lambda: self.simulate_edge_removal(
                    primary_key, pattern_id=pattern_id, line_id=line_id,
                    top_n=top_n_edges,
                ),
            )

        out["steps_status"] = steps_status
        out["elapsed_ms"] = (_time.perf_counter() - t0) * 1000.0
        return out

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

    def theta_sensitivity(
        self,
        pattern_id: str,
        version: int | None = None,
    ) -> "ThetaSensitivityReport":
        """Calibration-quality diagnostic surfacing `theta_sensitivity` for one
        pattern epoch.

        Reads the populated `theta_sensitivity` field on `CalibrationFit` and
        derives `stable_band` + `cliffs` so agents see which `anomaly_percentile`
        recalibration moves are safe and which destabilise the population.

        Args:
            pattern_id: which pattern to inspect.
            version: calibration epoch (None → latest on disk).

        Returns:
            ThetaSensitivityReport with the per-percentile sweep, stable band,
            and cliff list.

        Raises ValueError when the calibration epoch lacks the
        `theta_sensitivity` field — pre-T2 spheres need a rebuild before
        the diagnostic is available. CalibrationNotFoundError bubbles
        up from missing versions.
        """
        from hypertopos.builder._theta_sensitivity import (
            derive_stable_band_and_cliffs,
        )
        from hypertopos.model.sphere import ThetaSensitivityReport

        reader = self._storage
        if version is None:
            versions = reader.list_calibration_versions(pattern_id)
            if not versions:
                raise ValueError(
                    f"theta_sensitivity: no calibration epochs on disk for "
                    f"pattern={pattern_id!r}"
                )
            version = versions[-1]

        fit = reader.read_calibration_fit(pattern_id, version=version)
        if fit.theta_sensitivity is None:
            raise ValueError(
                f"theta_sensitivity: calibration epoch v={version} for "
                f"pattern={pattern_id!r} was built before the diagnostic was "
                f"wired in. Rebuild the pattern (or the full sphere) to "
                f"populate the field."
            )

        derived = derive_stable_band_and_cliffs(fit.theta_sensitivity)
        return ThetaSensitivityReport(
            pattern_id=fit.pattern_id,
            calibration_epoch=fit.calibration_epoch,
            population_size=fit.population_size,
            theta_sensitivity=fit.theta_sensitivity,
            stable_band=derived["stable_band"],
            cliffs=derived["cliffs"],
            n_cliffs=derived["n_cliffs"],
            stable_band_length=derived["stable_band_length"],
        )

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
        *,
        auto_discover: bool = False,
        auto_k: int = 10,
    ):
        """M4: find entities with high influence on coordinate system calibration.

        Default mode scans every entity in the pattern with leave-one-out and
        returns the ``top_n`` candidates filtered by ``classify``. In auto-
        discovery mode (``auto_discover=True``) the population is first
        partitioned by k-means++ into ``auto_k`` clusters; the nearest entity
        to each cluster centroid is selected as the cluster's representative
        and only those K representatives are ranked. Each entry then carries
        ``cluster_size`` and ``cluster_centroid_distance`` so callers can see
        which population segment the influencer represents. A write-through
        side effect populates per-influencer epoch caches under
        ``_gds_meta/calibration_history/<pattern_id>/influencer_<pk>.json`` so
        that ``calibration_influencer_history`` returns chronological history
        without recomputing.
        """
        from hypertopos.engine.geometry import (
            GDSEngine,
            _classify_influence,
            _compute_leave_one_out_impact,
            _count_cascading_flips,
        )
        from hypertopos.model.sphere import (
            DimensionContribution,
            InfluenceEntry,
            InfluenceReport,
        )
        from hypertopos.storage.calibration_history import (
            upsert_influencer_history_entry,
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
        if auto_discover and (auto_k < 1 or auto_k > 50):
            raise ValueError(
                f"find_calibration_influencers: auto_k must be in [1, 50]; "
                f"got {auto_k}"
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

        # Per-entity cluster metadata (populated only in auto_discover mode).
        cluster_size_by_idx: dict[int, int] = {}
        cluster_centroid_distance_by_idx: dict[int, float] = {}

        if auto_discover:
            # Partition the population with k-means++ on the delta matrix and
            # pick the entity nearest to each cluster centroid as the
            # cluster's representative. Only those K representatives are
            # ranked, so classify acts as a post-filter on cluster reps.
            effective_k = max(1, min(auto_k, N))
            labels, centroids = GDSEngine._kmeans(deltas, effective_k, seed=42)
            representatives: list[int] = []
            for k in range(centroids.shape[0]):
                member_mask = labels == k
                member_idx = np.where(member_mask)[0]
                if member_idx.size == 0:
                    continue
                member_deltas = deltas[member_idx]
                dists = np.linalg.norm(member_deltas - centroids[k], axis=1)
                rep_local = int(np.argmin(dists))
                rep_global = int(member_idx[rep_local])
                representatives.append(rep_global)
                cluster_size_by_idx[rep_global] = int(member_idx.size)
                cluster_centroid_distance_by_idx[rep_global] = float(dists[rep_local])
            candidates = representatives
            if classify != "all":
                candidates = [
                    i for i in candidates if classes[i] == classify
                ]
            candidates.sort(key=lambda i: total_imp[i], reverse=True)
            top_candidates = candidates[:top_n]
        else:
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
                cluster_size=cluster_size_by_idx.get(i),
                cluster_centroid_distance=cluster_centroid_distance_by_idx.get(i),
            ))

        # Write-through cache: upsert each surfaced entity's (epoch,
        # mu_impact, delta_norm) into its per-influencer history file so
        # calibration_influencer_history can replay impact across rebuilds
        # without recomputing the leave-one-out scan.
        try:
            fit = self._storage.read_calibration_fit(pattern_id, version=version)
            calibrated_at_iso = fit.last_calibrated_at.isoformat()
            for entry in entries:
                upsert_influencer_history_entry(
                    self._storage._base,
                    pattern_id,
                    entry.entity_key,
                    epoch=int(version),
                    calibrated_at=calibrated_at_iso,
                    mu_impact=float(entry.mu_impact),
                    delta_norm_impact=float(entry.delta_norm),
                )
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            # Cache writing is best-effort — never fail the primary analysis
            # because a sidecar file could not be written. TypeError covers
            # mocked-storage call sites where MagicMock leaks into the
            # serialiser path.
            logger.debug(
                "find_calibration_influencers: write-through cache failed",
                exc_info=True,
            )

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
            auto_discovered=auto_discover,
        )

    def calibration_influencer_history(
        self,
        primary_key: str,
        *,
        pattern_id: str,
    ):
        """Return chronological per-epoch μ-impact for a known influencer.

        Reads ``_gds_meta/calibration_history/<pattern_id>/influencer_<pk>.json``,
        written lazily by ``find_calibration_influencers``. When the cache is
        absent for the requested ``(primary_key, pattern_id)``, returns an
        empty history together with a ``hint`` explaining how to populate it.
        """
        import time

        from hypertopos.model.sphere import (
            InfluencerHistoryEntry,
            InfluencerHistoryReport,
        )
        from hypertopos.storage.calibration_history import read_influencer_history

        t0 = time.perf_counter()

        sphere = self._storage.read_sphere()
        if pattern_id not in sphere.patterns:
            raise ValueError(
                f"calibration_influencer_history: pattern_id={pattern_id!r} "
                f"not found on this sphere"
            )

        records = read_influencer_history(
            self._storage._base, pattern_id, primary_key,
        )
        entries = [
            InfluencerHistoryEntry(
                epoch=int(r["epoch"]),
                calibrated_at=str(r["calibrated_at"]),
                mu_impact=float(r["mu_impact"]),
                delta_norm_impact=float(r["delta_norm_impact"]),
            )
            for r in records
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hint: str | None = None
        if not entries:
            hint = (
                "no impact history recorded for this entity yet — call "
                "find_calibration_influencers(pattern_id=..., classify='all') "
                "in this sphere to populate the cache, then retry"
            )

        return InfluencerHistoryReport(
            primary_key=primary_key,
            pattern_id=pattern_id,
            history=entries,
            n_epochs=len(entries),
            elapsed_ms=elapsed_ms,
            hint=hint,
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
                n_ent_a = self._storage.count_geometry_rows(pattern_a)
                n_ent_b = self._storage.count_geometry_rows(pattern_b)
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
        if min_epochs > N:
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

    def find_motif_by_hops(
        self,
        pattern_id: str,
        hops: list[Any],
        *,
        seed_keys: list[str] | None = None,
        max_results: int = 100,
        score: bool = False,
        time_window_hours: float | None = None,
        anomaly_seed_filter: bool = False,
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

        ``anomaly_seed_filter`` (default ``False``): pre-filter the BFS
        starting frontier to entities with ``is_anomaly=True`` in the
        resolved anchor companion pattern. When ``seed_keys=None``,
        replaces the implicit "all keys" frontier with the anomaly
        subset; when ``seed_keys=<list>`` is provided, intersects the
        list with the anomaly subset. On large populations (e.g. 515 k
        anchor entities with ~ 5 % anomaly rate) this collapses the BFS
        traversal to ~ 1/20 of the work, typically 5–15 × wall-clock
        improvement on population-sweep motif queries. Raises
        ``GDSNavigationError`` when the resolved anchor pattern has no
        ``is_anomaly`` column (calibration must run first). The result
        dict gains ``seed_filter_summary`` (``{requested, anomaly,
        filtered}``) so the caller can verify the prune.
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

        # anomaly_seed_filter pre-prune: collapse the BFS starting
        # frontier to anomaly entities of the resolved anchor companion.
        # Either replaces the implicit "all keys" frontier (when
        # seed_keys=None) or intersects with an explicit caller-supplied
        # list. Captured for the diagnostics block.
        seed_filter_summary: dict[str, Any] | None = None
        if anomaly_seed_filter:
            anchor_pid = (
                self._resolve_anchor_pattern_for_scoring(pattern_id)
            )
            if anchor_pid is None or anchor_pid == pattern_id:
                raise GDSNavigationError(
                    f"anomaly_seed_filter requires an anchor companion "
                    f"pattern whose entity_line covers edge endpoints "
                    f"of {pattern_id!r}; none found.",
                )
            anchor_version = self._resolve_version(anchor_pid)
            try:
                anom_geo = self._storage.read_geometry(
                    anchor_pid, anchor_version,
                    columns=["primary_key", "is_anomaly"],
                    filter="is_anomaly = true",
                )
            except (KeyError, ValueError) as exc:
                raise GDSNavigationError(
                    f"anchor pattern {anchor_pid!r} cannot serve "
                    f"is_anomaly — calibration must run first",
                ) from exc
            anom_keys = anom_geo["primary_key"].to_pylist()
            anom_set = set(anom_keys)
            requested = (
                len(seed_keys) if seed_keys is not None else None
            )
            if seed_keys is None:
                # sorted() — list(set) ordering is nondeterministic across
                # processes (PYTHONHASHSEED randomises string hashing) and
                # would produce different BFS visit orders → different
                # max_results trims across runs.
                filtered_seeds = sorted(anom_set)
            else:
                filtered_seeds = [k for k in seed_keys if k in anom_set]
            seed_filter_summary = {
                "requested": requested,
                "anomaly": len(anom_set),
                "filtered": len(filtered_seeds),
            }
            seed_keys = filtered_seeds

        # AdjacencyIndex is built once per pattern at storage level
        # (cached inside GDSReader._adjacency_cache). The cold first call
        # on a multi-million-edge table pays the full O(E) build; warm
        # calls amortise to BFS-only cost. The BFS uses seed_keys to
        # filter the starting frontier when provided, so the global
        # adjacency works correctly for both seeded and unseeded queries.
        adj = self._storage.get_adjacency(pattern_id)
        if adj is None or not adj._out:
            empty: dict[str, Any] = {
                "pattern_id": pattern_id,
                "n_results": 0,
                "motifs": [],
                "reason": "pattern has no edge table",
            }
            if seed_filter_summary is not None:
                empty["seed_filter_summary"] = seed_filter_summary
            return empty

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

        result: dict[str, Any] = {
            "pattern_id": pattern_id,
            "n_results": len(instances),
            "motifs": instances,
        }
        if seed_filter_summary is not None:
            result["seed_filter_summary"] = seed_filter_summary
        return result

