# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Per-edge transaction counterfactual — pure-math primitive.

For an entity sitting on a polygon shape vector and a set of edges in the
underlying graph, simulate "what would the entity's delta_norm be if edge X
were removed?" for each candidate edge. Rank by the magnitude of the drop.

v0 covers the relations-dim class only (count-based, closed-form). The
event_dim and edge_dim_aggregation classes need deeper builder integration
and ship in a follow-up patch.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "simulate_edge_removal_naive",
    "simulate_edge_removal_with_aggregations",
    "aggregate_edge_removals_by_counterparty",
    "ecdf_pvalue_upper_tail",
    "compute_per_edge_source_value_pvalues",
    "simulate_joint_edge_removal",
    "select_minimal_joint_removal",
]


_SIGMA_ZERO_FLOOR = 1e-10
_AGGS_INTRINSIC = ("mean", "max", "std", "p95")  # no extra inputs needed
_AGG_COUNT_ABOVE = "count_above_threshold"  # requires per-source-dim threshold


def _aggregate(
    values: np.ndarray,
    agg: str,
    threshold: float | None = None,
) -> float:
    """Match the builder's aggregation math exactly (see engine.edge_features)."""
    if values.size == 0:
        return 0.0
    if agg == "mean":
        return float(values.mean())
    if agg == "max":
        return float(values.max())
    if agg == "std":
        # builder uses pc.VarianceOptions(ddof=0) → numpy default
        return float(values.std(ddof=0))
    if agg == "p95":
        # builder: lexsort + index at (size-1) * 0.95 (truncating int) — exact
        n = values.size
        sorted_vals = np.sort(values)
        idx = int((n - 1) * 0.95)
        return float(sorted_vals[idx])
    if agg == _AGG_COUNT_ABOVE:
        if threshold is None:
            raise ValueError(
                "count_above_threshold requires a threshold value",
            )
        return float(int((values > threshold).sum()))
    raise ValueError(f"unsupported aggregation: {agg!r}")


def simulate_edge_removal_naive(
    *,
    shape: np.ndarray,
    mu: np.ndarray,
    sigma_diag: np.ndarray,
    delta_norm: float,
    edges: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    candidate_edge_ids: list[str] | None,
    top_n: int,
) -> list[dict[str, Any]]:
    """Per-edge counterfactual on relations dims.

    For each candidate edge in ``edges``, identify which relations dims it
    contributes to (line_id + direction match against ``relations``),
    subtract the per-dim contribution from ``shape``, re-normalise via
    ``(shape - mu) / sigma_diag`` to a new delta vector, take the L2 norm,
    and report the drop.

    Args:
        shape: entity's raw shape vector (pre-normalisation), shape (D,).
        mu: pattern population mean per dim, shape (D,).
        sigma_diag: pattern population std per dim, shape (D,). Dims with
            ``sigma < 1e-10`` are reported in ``dimensions_skipped`` and
            held constant.
        delta_norm: entity's current delta_norm (pre-removal).
        edges: list of edge records, each ``{edge_id, partner_key,
            direction, line_id}``.
        relations: list of relation defs in the pattern's dim order, each
            ``{line_id, direction}``. Length must equal len(shape).
        candidate_edge_ids: if not None, restrict simulation to edges whose
            ``edge_id`` is in this list. Otherwise simulate all edges.
        top_n: cap on returned entries (sorted by ``|drop_pct|`` desc).

    Returns:
        List of dicts sorted by ``|drop_pct|`` descending. Each entry:
        ``{edge_id, edge_partner_key, edge_direction,
        delta_norm_before, delta_norm_after, drop_pct,
        dominant_dim_idx, dominant_dim_label,
        dimensions_simulated, dimensions_skipped}``.

        Empty list if no edge produces a simulatable contribution (e.g.
        pattern has no relations matching any edge's line_id + direction).
    """
    if shape.shape != mu.shape or shape.shape != sigma_diag.shape:
        raise ValueError(
            f"shape / mu / sigma_diag must have the same length; got "
            f"{shape.shape} / {mu.shape} / {sigma_diag.shape}",
        )
    if len(relations) != len(shape):
        raise ValueError(
            f"len(relations) ({len(relations)}) must equal len(shape) "
            f"({len(shape)}) — one relation def per pattern dim",
        )

    shape = shape.astype(np.float64, copy=True)
    mu = mu.astype(np.float64, copy=False)
    sigma_diag = sigma_diag.astype(np.float64, copy=False)

    sigma_dead_mask = sigma_diag < _SIGMA_ZERO_FLOOR
    sigma_safe = np.where(sigma_dead_mask, 1.0, sigma_diag)
    dims_skipped = [int(i) for i, dead in enumerate(sigma_dead_mask) if dead]
    dims_simulated = [int(i) for i, dead in enumerate(sigma_dead_mask) if not dead]

    # Per-dim contribution: +1 for each edge that matches the dim's
    # (line_id, direction). Pre-compute the lookup so the per-edge loop is
    # O(D) not O(D × n_relations) per edge.
    rel_by_key: dict[tuple[str, str], list[int]] = {}
    for dim_idx, rel in enumerate(relations):
        key = (rel["line_id"], rel["direction"])
        rel_by_key.setdefault(key, []).append(dim_idx)

    # Filter edges by candidate_edge_ids.
    if candidate_edge_ids is None:
        candidates = list(edges)
    else:
        candidate_set = set(candidate_edge_ids)
        candidates = [e for e in edges if e["edge_id"] in candidate_set]

    rows: list[dict[str, Any]] = []
    for edge in candidates:
        contribution = np.zeros(len(shape), dtype=np.float64)
        key = (edge["line_id"], edge["direction"])
        for dim_idx in rel_by_key.get(key, []):
            if dim_idx not in dims_skipped:
                contribution[dim_idx] = 1.0

        new_shape = shape - contribution
        new_delta = np.where(
            sigma_dead_mask, 0.0, (new_shape - mu) / sigma_safe,
        )
        new_delta_norm = float(np.linalg.norm(new_delta))
        if not np.isfinite(new_delta_norm):
            new_delta_norm = float(delta_norm)
        drop_pct = (
            (delta_norm - new_delta_norm) / delta_norm * 100.0
            if delta_norm > 0
            else 0.0
        )

        # Dominant dim: argmax of |new_delta - old_delta|. If contribution is
        # all zero, the dominant_dim_idx is undefined; default to 0.
        old_delta = np.where(
            sigma_dead_mask, 0.0, (shape - mu) / sigma_safe,
        )
        delta_diff = np.abs(new_delta - old_delta)
        dominant_dim_idx = int(np.argmax(delta_diff))

        rows.append({
            "edge_id": edge["edge_id"],
            "edge_partner_key": edge.get("partner_key"),
            "edge_direction": edge.get("direction"),
            "edge_line_id": edge.get("line_id"),
            "delta_norm_before": float(delta_norm),
            "delta_norm_after": new_delta_norm,
            "drop_pct": float(drop_pct),
            "dominant_dim_idx": dominant_dim_idx,
            "dimensions_simulated": dims_simulated,
            "dimensions_skipped": dims_skipped,
        })

    # If the pattern has NO relations matching ANY edge's line_id (regardless
    # of direction), the counterfactual surface is structurally empty —
    # nothing in this pattern can be simulated against these edges. Surface
    # that as an empty list. (Distinct from "line matches but direction
    # mismatches" — that case keeps the edge in the result with drop_pct=0
    # so the investigator sees the edge was evaluated.)
    if candidate_edge_ids is None:
        pattern_line_ids = {rel["line_id"] for rel in relations}
        any_line_match = any(edge["line_id"] in pattern_line_ids for edge in edges)
        if not any_line_match:
            return []

    rows.sort(key=lambda r: abs(r["drop_pct"]), reverse=True)
    return rows[:top_n]


def simulate_edge_removal_with_aggregations(
    *,
    shape: np.ndarray,
    mu: np.ndarray,
    sigma_diag: np.ndarray,
    delta_norm: float,
    edges: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    edge_agg_dim_offset: int,
    edge_agg_specs: list[tuple[str, str]],
    event_source_values: dict[str, dict[str, float]],
    candidate_edge_ids: list[str] | None,
    top_n: int,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Per-edge counterfactual covering relations + edge_dim_aggregations.

    Beyond the relations support of ``simulate_edge_removal_naive``, this
    function also handles the ``edge_dim_aggregations`` dim class — the
    AGGREGATE_NAMES family (``mean`` / ``max`` / ``std`` / ``p95`` /
    ``count_above_threshold``) of per-anchor aggregations over the joined
    edges sidecar.

    For each candidate edge ``e``:
        1. Subtract per-relation contributions (closed-form +1) — same as
           the naive path.
        2. For each ``(source_dim_name, agg_kind)`` in ``edge_agg_specs``,
           recompute the aggregate on the entity's event source-values
           EXCLUDING the candidate ``e`` and write the new value into
           ``new_shape[edge_agg_dim_offset + i]``.
        3. Re-normalise via ``(new_shape - mu) / sigma_diag`` and report
           the delta_norm drop.

    ``count_above_threshold`` requires population-level threshold lookup
    that the builder resolves from sidecar p95; if a caller cannot supply
    it, those (dim, agg) pairs are recorded in ``dimensions_skipped`` and
    held constant — see ``_SUPPORTED_AGGS``.

    Args:
        shape, mu, sigma_diag: per-pattern population statistics.
        delta_norm: entity's current delta_norm.
        edges: entity's edges, each with ``edge_id`` + relation tags.
        relations: pattern's relation defs (line_id + direction per dim),
            same order as the first ``len(relations)`` dims of the shape
            vector.
        edge_agg_dim_offset: starting index of the edge_dim_aggregations
            block in the shape vector (typically
            ``len(relations) + len(event_dimensions) + len(prop_columns)``).
        edge_agg_specs: ordered list of ``(source_dim_name, agg_kind)``
            tuples, one per shape position starting at ``edge_agg_dim_offset``.
        event_source_values: per-event source-dim values keyed by
            ``event_key`` then ``source_dim_name``. Holds the values that
            the builder's group_by + aggregate would consume.
        candidate_edge_ids: filter.
        top_n: cap on returned entries.

    Returns:
        Per-edge dict list — same shape as ``simulate_edge_removal_naive``
        plus ``dimensions_simulated`` covers BOTH relations and the
        edge_dim_aggregations the implementation supports.
    """
    if shape.shape != mu.shape or shape.shape != sigma_diag.shape:
        raise ValueError(
            f"shape / mu / sigma_diag must have the same length; got "
            f"{shape.shape} / {mu.shape} / {sigma_diag.shape}",
        )

    shape = shape.astype(np.float64, copy=True)
    mu = mu.astype(np.float64, copy=False)
    sigma_diag = sigma_diag.astype(np.float64, copy=False)

    sigma_dead_mask = sigma_diag < _SIGMA_ZERO_FLOOR
    sigma_safe = np.where(sigma_dead_mask, 1.0, sigma_diag)
    dims_skipped = [int(i) for i, dead in enumerate(sigma_dead_mask) if dead]

    # Relations lookup (same as naive path).
    rel_by_key: dict[tuple[str, str], list[int]] = {}
    for dim_idx, rel in enumerate(relations):
        key = (rel["line_id"], rel["direction"])
        rel_by_key.setdefault(key, []).append(dim_idx)

    # Edge-dim-agg lookup: for each (source_dim_name, agg) compute
    # population value vector across entity's events. agg held constant if
    # not in _SUPPORTED_AGGS.
    n_edges_total = len(edges)
    event_keys = [e["edge_id"] for e in edges]
    source_value_arrays: dict[str, np.ndarray] = {}
    for source_dim in {s[0] for s in edge_agg_specs}:
        vals = np.array(
            [event_source_values.get(ek, {}).get(source_dim, 0.0)
             for ek in event_keys],
            dtype=np.float64,
        )
        source_value_arrays[source_dim] = vals

    thresholds_map = thresholds or {}

    def _agg_is_supported(source_dim: str, agg: str) -> bool:
        if agg in _AGGS_INTRINSIC:
            return True
        if agg == _AGG_COUNT_ABOVE:
            return source_dim in thresholds_map
        return False

    unsupported_agg_indices: list[int] = []
    for i, (source_dim, agg) in enumerate(edge_agg_specs):
        if not _agg_is_supported(source_dim, agg):
            unsupported_agg_indices.append(edge_agg_dim_offset + i)
    dims_skipped.extend(unsupported_agg_indices)
    dims_skipped = sorted(set(dims_skipped))
    dims_simulated = [
        i for i in range(len(shape)) if i not in dims_skipped
    ]

    # Filter candidates.
    if candidate_edge_ids is None:
        candidates = list(edges)
    else:
        candidate_set = set(candidate_edge_ids)
        candidates = [e for e in edges if e["edge_id"] in candidate_set]

    # If the pattern has NO relations matching ANY edge's line_id AND there
    # are no edge_dim_aggregations to simulate either, the surface is empty.
    pattern_line_ids = {rel["line_id"] for rel in relations}
    any_relation_line_match = any(
        edge["line_id"] in pattern_line_ids for edge in edges
    )
    has_supported_aggs = any(
        _agg_is_supported(source_dim, agg)
        for source_dim, agg in edge_agg_specs
    )
    if (
        candidate_edge_ids is None
        and not any_relation_line_match
        and not has_supported_aggs
    ):
        return []

    rows: list[dict[str, Any]] = []
    for edge in candidates:
        contribution = np.zeros(len(shape), dtype=np.float64)
        # Relations contribution
        key = (edge["line_id"], edge["direction"])
        for dim_idx in rel_by_key.get(key, []):
            if dim_idx not in dims_skipped:
                contribution[dim_idx] = 1.0
        new_shape = shape - contribution

        # Edge-dim-agg contribution: REPLACE shape value at agg-dim with the
        # recomputed aggregate over remaining events (not subtraction).
        edge_idx_in_events: int | None = None
        for j, ek in enumerate(event_keys):
            if ek == edge["edge_id"]:
                edge_idx_in_events = j
                break
        if edge_idx_in_events is not None and n_edges_total > 0:
            for i, (source_dim, agg) in enumerate(edge_agg_specs):
                dim_idx = edge_agg_dim_offset + i
                if dim_idx in dims_skipped:
                    continue
                vals = source_value_arrays[source_dim]
                if vals.size <= 1:
                    new_agg = 0.0
                else:
                    remaining = np.delete(vals, edge_idx_in_events)
                    new_agg = _aggregate(
                        remaining, agg,
                        threshold=thresholds_map.get(source_dim),
                    )
                new_shape[dim_idx] = new_agg

        new_delta = np.where(
            sigma_dead_mask, 0.0, (new_shape - mu) / sigma_safe,
        )
        new_delta_norm = float(np.linalg.norm(new_delta))
        if not np.isfinite(new_delta_norm):
            new_delta_norm = float(delta_norm)
        drop_pct = (
            (delta_norm - new_delta_norm) / delta_norm * 100.0
            if delta_norm > 0
            else 0.0
        )

        old_delta = np.where(
            sigma_dead_mask, 0.0, (shape - mu) / sigma_safe,
        )
        delta_diff = np.abs(new_delta - old_delta)
        dominant_dim_idx = int(np.argmax(delta_diff))

        rows.append({
            "edge_id": edge["edge_id"],
            "edge_partner_key": edge.get("partner_key"),
            "edge_direction": edge.get("direction"),
            "edge_line_id": edge.get("line_id"),
            "delta_norm_before": float(delta_norm),
            "delta_norm_after": new_delta_norm,
            "drop_pct": float(drop_pct),
            "dominant_dim_idx": dominant_dim_idx,
            "dimensions_simulated": dims_simulated,
            "dimensions_skipped": dims_skipped,
        })

    rows.sort(key=lambda r: abs(r["drop_pct"]), reverse=True)
    return rows[:top_n]


def aggregate_edge_removals_by_counterparty(
    per_edge_results: list[dict[str, Any]],
    *,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Group per-edge counterfactual results by counterparty partner key.

    Investigator-facing rollup: AML analysts think per-counterparty, not
    per-transaction. Given the per-edge ranking, this surfaces which
    PARTNER concentrates the entity's anomaly contribution.

    Aggregation per partner:
        - ``n_edges``: distinct edge_ids attributed to this partner.
        - ``sum_drop_pct``: directional sum (positive = removing all
          partner's edges would lower delta_norm; negative = raise).
        - ``sum_abs_drop_pct``: total magnitude of contribution.
        - ``max_abs_drop_pct``: worst single-edge contribution.
        - ``dominant_dim_label``: dim label of that worst single edge.
        - ``edge_ids``: distinct edge_ids comprising this partner.

    Dedup discipline: the same ``edge_id`` may appear multiple times in
    ``per_edge_results`` when an event surfaces in both directions of
    adjacency (bidirectional / self-referencing edges). Counted ONCE per
    partner. The dedup key is ``(partner, edge_id)`` — values from the
    first occurrence win.

    Sort: ``sum_abs_drop_pct`` descending — the partner whose collective
    edges contribute the most absolute magnitude rises to the top.
    Truncated to ``top_n``.

    ``edge_partner_key=None`` (self-edges / missing) bucketed under
    ``"__unknown__"``.
    """
    by_partner: dict[str, dict[str, Any]] = {}
    seen_edges: set[tuple[str, str]] = set()

    for row in per_edge_results:
        partner = row.get("edge_partner_key") or "__unknown__"
        edge_id = row.get("edge_id")
        if edge_id is None:
            continue
        dedup_key = (partner, edge_id)
        if dedup_key in seen_edges:
            continue
        seen_edges.add(dedup_key)

        drop = float(row.get("drop_pct", 0.0))
        bucket = by_partner.setdefault(partner, {
            "partner_key": partner,
            "n_edges": 0,
            "sum_drop_pct": 0.0,
            "sum_abs_drop_pct": 0.0,
            "max_abs_drop_pct": 0.0,
            "dominant_dim_label": None,
            "edge_ids": [],
        })
        bucket["n_edges"] += 1
        bucket["sum_drop_pct"] += drop
        bucket["sum_abs_drop_pct"] += abs(drop)
        bucket["edge_ids"].append(edge_id)
        if abs(drop) > bucket["max_abs_drop_pct"]:
            bucket["max_abs_drop_pct"] = abs(drop)
            bucket["dominant_dim_label"] = row.get("dominant_dim_label")

    rows = list(by_partner.values())
    rows.sort(key=lambda r: r["sum_abs_drop_pct"], reverse=True)
    return rows[:top_n]


def ecdf_pvalue_upper_tail(
    value: float,
    population_sorted: np.ndarray,
) -> float:
    """Upper-tail empirical p-value for ``value`` against a sorted population.

    Returns ``P(V >= value)`` evaluated on the empirical CDF of
    ``population_sorted`` (a 1-D ascending-sorted numpy array). Equivalent to
    the rank-based one-sided significance:

        p = #(pop >= value) / N

    Phipson-Smyth-style floor applied: when ``value`` exceeds every
    population sample (would otherwise return 0), returns ``1 / (N + 1)``
    so HMP / log combiners downstream stay finite.

    Args:
        value: edge's source-dim observation.
        population_sorted: ascending-sorted 1-D array of population values
            for the same source dim. Caller is responsible for the sort.
    """
    n = population_sorted.size
    if n == 0:
        return 1.0
    # searchsorted with side='left' returns the index where `value` would be
    # inserted to keep order, i.e. number of pop entries strictly < value.
    # Then #(pop >= value) = n - that_index.
    idx = int(np.searchsorted(population_sorted, value, side="left"))
    rank_geq = n - idx
    if rank_geq <= 0:
        return 1.0 / (n + 1)
    return float(rank_geq) / float(n)


def compute_per_edge_source_value_pvalues(
    *,
    edges: list[dict[str, Any]],
    event_source_values: dict[str, dict[str, float]],
    population_ecdfs: dict[str, np.ndarray],
    source_dims: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-edge per-source-dim extremeness p-value against population ECDFs.

    For each edge ``e`` and each ``source_dim d`` in ``source_dims``:
        p_{e, d} = P(V >= v_{e, d}) on the population sample for ``d``

    The MIN across dims is reported as ``min_pvalue`` — the edge's most
    extreme source-value dim. ``dominant_significance_dim`` carries the
    label of that dim.

    Designed to **break the within-tied-`drop_pct` degeneracy** that the
    drop_pct-only ranking inherits when an entity's edges contribute
    uniformly to a robust-tail aggregation (e.g. `p95` with duplicates):
    even when raw ``drop_pct`` is constant across edges, the source-value
    extremeness still discriminates because edges differ in source values.

    Args:
        edges: ordered list of edge dicts (each carries ``edge_id``).
        event_source_values: per-event source-dim values, same shape as
            in ``simulate_edge_removal_with_aggregations``.
        population_ecdfs: per-source-dim ascending-sorted population
            samples. Caller pre-sorts. Sample size affects p-value
            resolution (`1/N` floor).
        source_dims: ordered list of source-dim names to evaluate.

    Returns:
        ``{edge_id: {source_dim: p_value, ..., "min_pvalue": float,
        "dominant_significance_dim": str}}``. Missing edges in
        ``event_source_values`` default to value=0.0 for each source_dim.
    """
    out: dict[str, dict[str, Any]] = {}
    for edge in edges:
        edge_id = edge["edge_id"]
        per_dim_pvalues: dict[str, Any] = {}
        min_p = 1.0
        min_dim: str | None = None
        edge_vals = event_source_values.get(edge_id, {})
        for d in source_dims:
            pop_sorted = population_ecdfs.get(d)
            if pop_sorted is None or pop_sorted.size == 0:
                per_dim_pvalues[d] = 1.0
                continue
            v = float(edge_vals.get(d, 0.0))
            p = ecdf_pvalue_upper_tail(v, pop_sorted)
            per_dim_pvalues[d] = p
            if p < min_p:
                min_p = p
                min_dim = d
        per_dim_pvalues["min_pvalue"] = min_p
        per_dim_pvalues["dominant_significance_dim"] = min_dim
        out[edge_id] = per_dim_pvalues
    return out


def _joint_new_shape(
    *,
    shape: np.ndarray,
    edges_to_remove: list[dict[str, Any]],
    all_edges: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    edge_agg_dim_offset: int,
    edge_agg_specs: list[tuple[str, str]],
    event_source_values: dict[str, dict[str, float]],
    sigma_dead_mask: np.ndarray,
    thresholds_map: dict[str, float],
    supported_aggs_mask: list[bool],
) -> np.ndarray:
    """Compute new shape vector when ``edges_to_remove`` are jointly gone.

    Relations dims: subtract count of matching (line_id, direction) edges
    in ``edges_to_remove``. Aggregation dims: recompute over surviving
    events (all_edges \\ edges_to_remove) for supported aggs.
    """
    new_shape = shape.astype(np.float64, copy=True)
    remove_ids = {e["edge_id"] for e in edges_to_remove}

    # Relations contribution (subtractive).
    rel_by_key: dict[tuple[str, str], list[int]] = {}
    for dim_idx, rel in enumerate(relations):
        rel_by_key.setdefault(
            (rel["line_id"], rel["direction"]), [],
        ).append(dim_idx)
    for edge in edges_to_remove:
        for dim_idx in rel_by_key.get(
            (edge["line_id"], edge["direction"]), [],
        ):
            if not sigma_dead_mask[dim_idx]:
                new_shape[dim_idx] -= 1.0

    # Aggregation dims: rescan over surviving events.
    if edge_agg_specs:
        surviving_events = [
            e for e in all_edges if e["edge_id"] not in remove_ids
        ]
        if surviving_events:
            for i, (source_dim, agg) in enumerate(edge_agg_specs):
                dim_idx = edge_agg_dim_offset + i
                if sigma_dead_mask[dim_idx] or not supported_aggs_mask[i]:
                    continue
                vals = np.array(
                    [event_source_values.get(e["edge_id"], {})
                     .get(source_dim, 0.0)
                     for e in surviving_events],
                    dtype=np.float64,
                )
                new_shape[dim_idx] = _aggregate(
                    vals, agg, threshold=thresholds_map.get(source_dim),
                )
    return new_shape


def simulate_joint_edge_removal(
    *,
    shape: np.ndarray,
    mu: np.ndarray,
    sigma_diag: np.ndarray,
    delta_norm: float,
    edges_to_remove: list[dict[str, Any]],
    all_edges: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    edge_agg_dim_offset: int,
    edge_agg_specs: list[tuple[str, str]],
    event_source_values: dict[str, dict[str, float]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Joint counterfactual: remove a SET of edges simultaneously and
    report the resulting ``delta_norm``.

    Unlike the per-edge primitive, the joint primitive captures
    coordination: removing one edge of a structured group may shift
    delta_norm by 0.3%; removing all 5 of them may shift by 8%. AML
    laundering is the canonical example — single-edge removal of a
    pattern member doesn't move the needle; coordinated removal does.

    Args:
        shape, mu, sigma_diag, delta_norm: entity's polygon state.
        edges_to_remove: set of edges to remove jointly.
        all_edges: full edge inventory (needed to rescan aggregations
            over the surviving events).
        relations, edge_agg_dim_offset, edge_agg_specs,
            event_source_values, thresholds: identical to the per-edge
            primitive.

    Returns:
        ``{removed_edge_ids, delta_norm_before, delta_norm_after,
        joint_drop_pct, dominant_dim_idx}`` where ``joint_drop_pct`` is
        ``(before - after) / before * 100``.
    """
    if shape.shape != mu.shape or shape.shape != sigma_diag.shape:
        raise ValueError(
            f"shape / mu / sigma_diag must have the same length; got "
            f"{shape.shape} / {mu.shape} / {sigma_diag.shape}",
        )

    shape = shape.astype(np.float64, copy=False)
    mu = mu.astype(np.float64, copy=False)
    sigma_diag = sigma_diag.astype(np.float64, copy=False)

    sigma_dead_mask = sigma_diag < _SIGMA_ZERO_FLOOR
    sigma_safe = np.where(sigma_dead_mask, 1.0, sigma_diag)
    thresholds_map = thresholds or {}

    def _agg_is_supported(source_dim: str, agg: str) -> bool:
        if agg in _AGGS_INTRINSIC:
            return True
        if agg == _AGG_COUNT_ABOVE:
            return source_dim in thresholds_map
        return False
    supported_aggs_mask = [
        _agg_is_supported(s, a) for s, a in edge_agg_specs
    ]

    new_shape = _joint_new_shape(
        shape=shape, edges_to_remove=edges_to_remove, all_edges=all_edges,
        relations=relations, edge_agg_dim_offset=edge_agg_dim_offset,
        edge_agg_specs=edge_agg_specs,
        event_source_values=event_source_values,
        sigma_dead_mask=sigma_dead_mask, thresholds_map=thresholds_map,
        supported_aggs_mask=supported_aggs_mask,
    )
    new_delta = np.where(
        sigma_dead_mask, 0.0, (new_shape - mu) / sigma_safe,
    )
    new_delta_norm = float(np.linalg.norm(new_delta))
    if not np.isfinite(new_delta_norm):
        new_delta_norm = float(delta_norm)
    drop_pct = (
        (delta_norm - new_delta_norm) / delta_norm * 100.0
        if delta_norm > 0 else 0.0
    )
    old_delta = np.where(
        sigma_dead_mask, 0.0, (shape - mu) / sigma_safe,
    )
    dominant_dim_idx = int(np.argmax(np.abs(new_delta - old_delta)))
    return {
        "removed_edge_ids": [e["edge_id"] for e in edges_to_remove],
        "delta_norm_before": float(delta_norm),
        "delta_norm_after": new_delta_norm,
        "joint_drop_pct": float(drop_pct),
        "dominant_dim_idx": dominant_dim_idx,
    }


def select_minimal_joint_removal(
    *,
    shape: np.ndarray,
    mu: np.ndarray,
    sigma_diag: np.ndarray,
    delta_norm: float,
    candidate_edges: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    edge_agg_dim_offset: int,
    edge_agg_specs: list[tuple[str, str]],
    event_source_values: dict[str, dict[str, float]],
    target_drop_pct: float = 50.0,
    k_max: int = 10,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Greedy submodular selection of the smallest edge set whose joint
    removal achieves at least ``target_drop_pct`` reduction in
    ``delta_norm``.

    At each step picks the candidate edge whose addition to the
    currently-selected set maximises the joint drop. Stops when
    ``joint_drop_pct >= target_drop_pct`` (target_reached) or when
    ``len(selected) == k_max`` (k_max_reached) or no candidate improves
    the joint drop (plateau).

    Designed for AML coordination detection: single-edge counterfactual
    on a structured laundering group returns near-zero drop_pct per edge
    (the group's contribution is non-decomposable to individual edges);
    the joint selection reveals which 3-10 edges form the coordinated
    set.

    Args:
        target_drop_pct: stop when joint reduction reaches this percent.
        k_max: hard cap on selected set size (greedy compute cost is
            ``O(k_max × n_candidates)`` evaluations).

    Returns:
        ``{selected_edge_ids, achieved_drop_pct, selection_sequence,
        target_reached, k_max_reached}`` where ``selection_sequence`` is
        a per-step record ``[{step, picked_edge_id, joint_drop_pct}, ...]``.
    """
    # Use absolute magnitude as the selection criterion: AML entities can
    # have all-positive OR all-negative per-edge drop_pct depending on
    # whether the entity is currently anomalous (negative drop = removing
    # edge raises anomaly) or close-to-population (positive drop = removing
    # edge lowers anomaly). The investigator question is "which set of
    # edges most influences this entity's polygon position" — direction-
    # agnostic. Target threshold is on |joint_drop_pct| likewise.
    selected: list[dict[str, Any]] = []
    remaining = list(candidate_edges)
    sequence: list[dict[str, Any]] = []
    achieved = 0.0
    achieved_signed = 0.0

    for step in range(k_max):
        best_abs = abs(achieved_signed)
        best_signed = achieved_signed
        best_edge: dict[str, Any] | None = None
        for cand in remaining:
            trial = [*selected, cand]
            r = simulate_joint_edge_removal(
                shape=shape, mu=mu, sigma_diag=sigma_diag,
                delta_norm=delta_norm,
                edges_to_remove=trial, all_edges=candidate_edges,
                relations=relations, edge_agg_dim_offset=edge_agg_dim_offset,
                edge_agg_specs=edge_agg_specs,
                event_source_values=event_source_values,
                thresholds=thresholds,
            )
            cand_signed = r["joint_drop_pct"]
            if abs(cand_signed) > best_abs:
                best_abs = abs(cand_signed)
                best_signed = cand_signed
                best_edge = cand
        if best_edge is None:
            break  # plateau — no candidate improves |joint_drop_pct|
        selected.append(best_edge)
        remaining = [
            e for e in remaining if e["edge_id"] != best_edge["edge_id"]
        ]
        achieved = best_abs
        achieved_signed = best_signed
        sequence.append({
            "step": step + 1,
            "picked_edge_id": best_edge["edge_id"],
            "picked_partner_key": best_edge.get("partner_key"),
            "picked_direction": best_edge.get("direction"),
            "joint_drop_pct": achieved_signed,
            "abs_joint_drop_pct": achieved,
        })
        if achieved >= target_drop_pct:
            break

    target_reached = achieved >= target_drop_pct
    k_max_reached = len(selected) >= k_max
    return {
        "selected_edge_ids": [e["edge_id"] for e in selected],
        "selected_partner_keys": [e.get("partner_key") for e in selected],
        "achieved_drop_pct": achieved_signed,
        "achieved_abs_drop_pct": achieved,
        "selection_sequence": sequence,
        "target_reached": target_reached,
        "k_max_reached": k_max_reached,
    }
