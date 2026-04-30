"""Declarative motif enumeration via per-hop predicates.

Walks the existing in-memory ``AdjacencyIndex`` (cached at storage level
so repeat calls are O(1) per node lookup, no 5M-edge dict rebuild).
For each candidate seed (single-seed list or all unique ``from_key``s),
applies ``hops[0]`` predicates to find hop-0 candidates, then walks
forward through hops applying temporal window, amount, direction and
optional edge-dim predicates.

Bounded MVP — ``amount_ratio_to_prev`` and ``require_anomalous_entity``
are NOT supported in this module; full PBL vectorisation lands in a follow-up release.
"""
from __future__ import annotations

from typing import Any

import pyarrow as pa  # noqa: TC002 — pyarrow.Table is exposed in signatures

_VALID_DIRECTIONS: frozenset[str] = frozenset({"forward", "reverse", "any"})
_VALID_OPERATORS: frozenset[str] = frozenset({"<", "<=", ">", ">=", "=="})


def _evaluate_predicate(value: float, op: str, threshold: float) -> bool:
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unknown operator: {op!r}")


def _passes_amount(
    amt: float | None,
    amount_min: float | None,
    amount_max: float | None,
) -> bool:
    if amt is None or amt <= 0:
        return False
    if amount_min is not None and amt < amount_min:
        return False
    if amount_max is not None and amt > amount_max:  # noqa: SIM103
        return False
    return True


class _FeatureLookup:
    """Lazy ek → dim values lookup.

    Materialises one ek→idx dict + one parallel list per requested dim
    instead of an N-row dict-of-dicts. On 5M-row sidecars this is ~10×
    smaller in heap memory because we skip the per-event nested dict
    allocation.
    """
    __slots__ = ("_idx", "_cols", "_dim_values_cache")

    def __init__(self, edge_features: pa.Table, requested_dims: set[str]) -> None:
        eks = edge_features["event_key"].to_pylist()
        self._idx: dict[str, int] = {ek: i for i, ek in enumerate(eks)}
        cols = [c for c in edge_features.column_names if c in requested_dims]
        self._cols: dict[str, list[float]] = {
            c: edge_features[c].to_pylist() for c in cols
        }
        self._dim_values_cache: dict[str, dict[str, float]] = {}

    def passes(
        self,
        event_key: str,
        predicates: dict[str, tuple[str, float]],
    ) -> bool:
        if not predicates:
            return True
        i = self._idx.get(event_key)
        if i is None:
            return False
        for dim_name, (op, threshold) in predicates.items():
            col = self._cols.get(dim_name)
            if col is None:
                return False
            val = col[i]
            if val is None or not _evaluate_predicate(val, op, threshold):
                return False
        return True

    def values_at(self, event_key: str) -> dict[str, float]:
        cached = self._dim_values_cache.get(event_key)
        if cached is not None:
            return cached
        i = self._idx.get(event_key)
        if i is None:
            return {}
        row = {c: col[i] for c, col in self._cols.items()}
        self._dim_values_cache[event_key] = row
        return row


def _passes_edge_dim(
    event_key: str,
    predicates: dict[str, tuple[str, float]],
    feature_lookup: _FeatureLookup | None,
) -> bool:
    if not predicates:
        return True
    if feature_lookup is None:
        return False
    return feature_lookup.passes(event_key, predicates)


def _build_feature_lookup(
    edge_features: pa.Table | None,
    requested_dims: set[str],
) -> _FeatureLookup | None:
    if edge_features is None or edge_features.num_rows == 0 or not requested_dims:
        return None
    return _FeatureLookup(edge_features, requested_dims)


def _candidates(
    out_map: dict[str, list[tuple[str, float, float, str]]],
    in_map: dict[str, list[tuple[str, float, float, str]]],
    current: str,
    direction: str,
) -> list[tuple[str, float, float, str]]:
    if direction == "forward":
        return out_map.get(current, [])
    if direction == "reverse":
        return in_map.get(current, [])
    # "any" — concat
    fwd = out_map.get(current, [])
    rev = in_map.get(current, [])
    if not rev:
        return fwd
    if not fwd:
        return rev
    return [*fwd, *rev]


def enumerate_motifs_by_hops(
    out_map: dict[str, list[tuple[str, float, float, str]]],
    in_map: dict[str, list[tuple[str, float, float, str]]],
    hops: list[Any],
    *,
    seed_keys: list[str] | None = None,
    max_results: int = 100,
    edge_features: pa.Table | None = None,
) -> list[dict[str, Any]]:
    """Enumerate motif instances matching the per-hop predicate list.

    ``out_map`` / ``in_map`` are AdjacencyIndex._out / _in style dicts —
    pre-sorted by timestamp ascending. Built once per pattern at storage
    level; reused across calls.

    Returns motif dicts with ``nodes``, ``edges``, ``timestamps``,
    ``amounts``, ``dim_values_per_hop`` (only when edge_dim_predicates
    were used). Stops at ``max_results``.
    """
    if not hops:
        raise ValueError("hops must be non-empty")
    if not 1 <= len(hops) <= 6:
        raise ValueError(f"hop count must be 1..6; got {len(hops)}")
    if hops[0].time_delta_max_hours is not None:
        raise ValueError(
            "hops[0].time_delta_max_hours must be None — first hop has no "
            "previous timestamp to compare against; place the time-window "
            "constraint on hops[1..] instead",
        )
    for i, hp in enumerate(hops):
        if hp.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"hops[{i}].direction must be one of {sorted(_VALID_DIRECTIONS)}; "
                f"got {hp.direction!r}",
            )
        if (
            hp.time_delta_max_hours is not None
            and hp.time_delta_max_hours <= 0
        ):
            raise ValueError(
                f"hops[{i}].time_delta_max_hours must be positive; "
                f"got {hp.time_delta_max_hours!r}",
            )
        for dim, (op, _v) in hp.edge_dim_predicates.items():
            if op not in _VALID_OPERATORS:
                raise ValueError(
                    f"hops[{i}].edge_dim_predicates[{dim!r}] operator must "
                    f"be one of {sorted(_VALID_OPERATORS)}; got {op!r}",
                )

    requested_dims: set[str] = set()
    for hp in hops:
        requested_dims.update(hp.edge_dim_predicates.keys())
    feature_lookup = _build_feature_lookup(edge_features, requested_dims)
    if requested_dims and feature_lookup is None:
        raise ValueError(
            f"edge_dim_predicates reference {sorted(requested_dims)} but no "
            f"edge_features sidecar is available for this pattern",
        )
    if requested_dims and edge_features is not None:
        sidecar_dims = set(edge_features.column_names) - {"event_key"}
        unknown = requested_dims - sidecar_dims
        if unknown:
            raise ValueError(
                f"edge_dim_predicates reference unknown dims "
                f"{sorted(unknown)}; available: {sorted(sidecar_dims)}",
            )

    candidate_seeds: list[str] = (
        seed_keys if seed_keys is not None else sorted(out_map.keys())
    )

    results: list[dict[str, Any]] = []

    def _walk(
        prev_nodes: list[str],
        prev_edges: list[str],
        prev_ts: list[float],
        prev_amts: list[float],
        prev_dim_values: list[dict[str, float]],
        hop_idx: int,
    ) -> None:
        if len(results) >= max_results:
            return
        if hop_idx == len(hops):
            inst: dict[str, Any] = {
                "nodes": list(prev_nodes),
                "edges": list(prev_edges),
                "timestamps": list(prev_ts),
                "amounts": list(prev_amts),
            }
            if requested_dims:
                inst["dim_values_per_hop"] = list(prev_dim_values)
            results.append(inst)
            return

        hp = hops[hop_idx]
        current = prev_nodes[-1]
        cands = _candidates(out_map, in_map, current, hp.direction)
        for (nxt, ts, amt, ek) in cands:
            if nxt in prev_nodes:
                continue
            if not _passes_amount(amt, hp.amount_min, hp.amount_max):
                continue
            if hop_idx > 0 and prev_ts:
                # Direction-aware temporal monotonicity:
                #   forward → strict-increasing time
                #   reverse → strict-decreasing time (predecessor edges
                #             precede current edge in causal order)
                #   any     → no monotonic constraint
                last_ts = prev_ts[-1]
                if hp.direction == "forward" and ts <= last_ts:
                    continue
                if hp.direction == "reverse" and ts >= last_ts:
                    continue
                if hp.time_delta_max_hours is not None:
                    delta = abs(ts - last_ts)
                    if delta > hp.time_delta_max_hours * 3600.0:
                        continue
            if not _passes_edge_dim(
                ek, hp.edge_dim_predicates, feature_lookup,
            ):
                continue

            row_dims = (
                feature_lookup.values_at(ek) if feature_lookup is not None else {}
            )
            _walk(
                [*prev_nodes, nxt],
                [*prev_edges, ek],
                [*prev_ts, ts],
                [*prev_amts, amt],
                [*prev_dim_values, row_dims] if requested_dims else prev_dim_values,
                hop_idx + 1,
            )
            if len(results) >= max_results:
                return

    for seed in candidate_seeds:
        _walk([seed], [], [], [], [], 0)
        if len(results) >= max_results:
            break

    return results
