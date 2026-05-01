"""Declarative motif enumeration via per-hop predicates.

Walks the existing in-memory ``AdjacencyIndex`` (cached at storage level
so repeat calls are O(1) per node lookup, no full edge-table dict
rebuild). For each candidate seed (single-seed list or all unique
``from_key``s), iterates BFS-by-level: at each hop level, every partial
chain in the frontier is extended by one edge subject to the hop's
predicates (amount window, direction-aware temporal monotonicity,
per-hop ``time_delta_max_hours``, optional ``amount_ratio_to_prev``,
optional global ``time_window_hours`` chain-span cap, optional
``edge_dim_predicates``).

Design note: this is a pragmatic level-synchronous BFS enumerator, not
the Paranjape-Benson-Leskovec delta-temporal motif state machine. PBL
counts fixed-template motifs (k=2, k=3) on temporal edge streams
without per-edge predicates; this enumerator targets k=1..8 chains
with arbitrary HopPredicate constraints, which is a different surface.
PBL is cited as prior art for sliding-window enumeration; the
``time_window_hours`` parameter expresses the analogous total-span cap
without adopting the paper's algorithm.

``require_anomalous_entity`` is enforced at the navigator layer
post-BFS — see ``GDSNavigator.find_motif_by_hops``. The engine here is
not aware of anchor-pattern anomaly status; the navigator filters
motifs by destination ``is_anomaly`` after this enumerator returns.
"""
from __future__ import annotations

from typing import Any

import pyarrow as pa  # noqa: TC002 — pyarrow.Table is exposed in signatures

_VALID_DIRECTIONS: frozenset[str] = frozenset({"forward", "reverse", "any"})
_VALID_OPERATORS: frozenset[str] = frozenset({"<", "<=", ">", ">=", "=="})

# Cap on intermediate frontier size relative to ``max_results`` — keeps
# enough survivors alive so the final hop produces ``max_results``
# results with high probability, without exponential blow-up on dense
# graphs.
_FRONTIER_SLACK_FACTOR = 4


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


def validate_hops(
    hops: list[Any],
    *,
    time_window_hours: float | None = None,
) -> None:
    """Pure-input validation of a per-hop predicate list.

    Extracted so callers (navigator early-return paths, MCP tool layer)
    can validate predicates BEFORE any sphere-state-dependent early-return
    fires — closes the silent-accept failure class on edge-table-less
    spheres.

    ``time_window_hours`` is the optional global chain-span cap; when
    not None it must be strictly positive.

    Raises ``ValueError`` on any rule violation.
    """
    if not hops:
        raise ValueError("hops must be non-empty")
    if not 1 <= len(hops) <= 8:
        raise ValueError(f"hop count must be 1..8; got {len(hops)}")
    if time_window_hours is not None and time_window_hours <= 0:
        raise ValueError(
            f"time_window_hours must be positive; got {time_window_hours!r}",
        )
    if hops[0].time_delta_max_hours is not None:
        raise ValueError(
            "hops[0].time_delta_max_hours must be None — first hop has no "
            "previous timestamp to compare against; place the time-window "
            "constraint on hops[1..] instead",
        )
    if hops[0].amount_ratio_to_prev is not None:
        raise ValueError(
            "hops[0].amount_ratio_to_prev must be None — first hop has no "
            "previous amount to compare against; place the ratio constraint "
            "on hops[1..] instead",
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
        if hp.amount_ratio_to_prev is not None:
            if not (0 < hp.amount_ratio_to_prev <= 1.0):
                raise ValueError(
                    f"hops[{i}].amount_ratio_to_prev must be in (0, 1]; "
                    f"got {hp.amount_ratio_to_prev!r}",
                )
        for dim, (op, _v) in hp.edge_dim_predicates.items():
            if op not in _VALID_OPERATORS:
                raise ValueError(
                    f"hops[{i}].edge_dim_predicates[{dim!r}] operator must "
                    f"be one of {sorted(_VALID_OPERATORS)}; got {op!r}",
                )


def _expand_one_level(
    state: list[dict[str, Any]],
    out_map: dict[str, list[tuple[str, float, float, str]]],
    in_map: dict[str, list[tuple[str, float, float, str]]],
    hp: Any,
    hop_idx: int,
    feature_lookup: _FeatureLookup | None,
    requested_dims: set[str],
    time_window_hours: float | None,
    max_results: int,
) -> list[dict[str, Any]]:
    """Extend every partial chain in ``state`` by one hop.

    Returns the new frontier. Predicate evaluation order matches the
    previous DFS implementation so backward-compat is preserved.
    """
    next_state: list[dict[str, Any]] = []
    frontier_cap = max(max_results * _FRONTIER_SLACK_FACTOR, max_results)
    window_seconds = (
        time_window_hours * 3600.0 if time_window_hours is not None else None
    )

    for chain in state:
        prev_nodes: list[str] = chain["nodes"]
        prev_edges: list[str] = chain["edges"]
        prev_ts: list[float] = chain["ts"]
        prev_amts: list[float] = chain["amts"]
        prev_dims: list[dict[str, float]] | None = chain["dims"]

        last_node = prev_nodes[-1]
        cands = _candidates(out_map, in_map, last_node, hp.direction)
        # Local refs avoid attribute lookups inside the per-edge loop.
        amount_min = hp.amount_min
        amount_max = hp.amount_max
        time_delta_max_hours = hp.time_delta_max_hours
        amount_ratio_to_prev = hp.amount_ratio_to_prev
        direction = hp.direction
        edge_dim_predicates = hp.edge_dim_predicates

        for (nxt, ts, amt, ek) in cands:
            if nxt in prev_nodes:
                continue
            if not _passes_amount(amt, amount_min, amount_max):
                continue
            # Inter-hop predicates (only fire from hop_idx >= 1).
            if hop_idx > 0 and prev_ts:
                last_ts = prev_ts[-1]
                # Direction-aware temporal monotonicity:
                #   forward → strict-increasing time
                #   reverse → strict-decreasing time (predecessor edges
                #             precede current edge in causal order)
                #   any     → no monotonic constraint
                if direction == "forward" and ts <= last_ts:
                    continue
                if direction == "reverse" and ts >= last_ts:
                    continue
                if time_delta_max_hours is not None:
                    delta = abs(ts - last_ts)
                    if delta > time_delta_max_hours * 3600.0:
                        continue
                if amount_ratio_to_prev is not None:
                    prev_amt = prev_amts[-1]
                    if prev_amt <= 0 or amt <= 0:
                        continue
                    if amt / prev_amt > amount_ratio_to_prev:
                        continue
            # Global chain-span cap — independent of per-hop windows.
            # Compares current edge ts to the chain's first hop ts.
            if window_seconds is not None and prev_ts:
                if abs(ts - prev_ts[0]) > window_seconds:
                    continue
            if not _passes_edge_dim(
                ek, edge_dim_predicates, feature_lookup,
            ):
                continue

            row_dims = (
                feature_lookup.values_at(ek) if feature_lookup is not None else {}
            )
            new_chain: dict[str, Any] = {
                "nodes": [*prev_nodes, nxt],
                "edges": [*prev_edges, ek],
                "ts": [*prev_ts, ts],
                "amts": [*prev_amts, amt],
                "dims": (
                    [*prev_dims, row_dims] if prev_dims is not None else None
                ),
            }
            next_state.append(new_chain)
            if len(next_state) >= frontier_cap:
                return next_state

    return next_state


def _state_to_motif(
    chain: dict[str, Any],
    requested_dims: set[str],
) -> dict[str, Any]:
    inst: dict[str, Any] = {
        "nodes": list(chain["nodes"]),
        "edges": list(chain["edges"]),
        "timestamps": list(chain["ts"]),
        "amounts": list(chain["amts"]),
    }
    if requested_dims:
        inst["dim_values_per_hop"] = list(chain["dims"] or [])
    return inst


def enumerate_motifs_by_hops(
    out_map: dict[str, list[tuple[str, float, float, str]]],
    in_map: dict[str, list[tuple[str, float, float, str]]],
    hops: list[Any],
    *,
    seed_keys: list[str] | None = None,
    max_results: int = 100,
    edge_features: pa.Table | None = None,
    time_window_hours: float | None = None,
) -> list[dict[str, Any]]:
    """Enumerate motif instances matching the per-hop predicate list.

    ``out_map`` / ``in_map`` are AdjacencyIndex._out / _in style dicts —
    pre-sorted by timestamp ascending. Built once per pattern at storage
    level; reused across calls.

    ``time_window_hours`` is an optional total-chain-span cap. When set,
    every chain instance must satisfy
    ``abs(current_edge_ts - first_edge_ts) <= time_window_hours * 3600``
    on every hop after the first. This is independent of the per-hop
    ``time_delta_max_hours`` constraint (consecutive-hop window) — both
    apply when both are set.

    Returns motif dicts with ``nodes``, ``edges``, ``timestamps``,
    ``amounts``, ``dim_values_per_hop`` (only when edge_dim_predicates
    were used). Stops at ``max_results``.
    """
    validate_hops(hops, time_window_hours=time_window_hours)

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

    # Initial frontier: one trivial partial chain per seed (no edges yet).
    state: list[dict[str, Any]] = [
        {
            "nodes": [s],
            "edges": [],
            "ts": [],
            "amts": [],
            "dims": [] if requested_dims else None,
        }
        for s in candidate_seeds
    ]

    # Apply each hop level-synchronously. Empty frontier short-circuits.
    for hop_idx in range(len(hops)):
        if not state:
            break
        state = _expand_one_level(
            state,
            out_map,
            in_map,
            hops[hop_idx],
            hop_idx,
            feature_lookup,
            requested_dims,
            time_window_hours,
            max_results,
        )

    return [_state_to_motif(c, requested_dims) for c in state[:max_results]]
