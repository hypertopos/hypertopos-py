# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Chain extraction from event data — temporal BFS over transaction graphs.

Finds sequences of linked events (A→B→C→D) within a time window.
Used for: money laundering chain detection, supply chain path analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hypertopos.utils.gc_pause import without_cyclic_gc
from hypertopos.utils.time import to_epoch_seconds

_DEDUP_CAP = 500_000  # max entries in visited set to prevent unbounded memory

# DFS stack frame: (current, path_keys, path_keys_set, path_events,
# path_events_set, path_categories, path_amounts, first_ts, last_ts)
_DfsFrame = tuple[
    int,
    tuple[int, ...],
    set[int],
    tuple[int, ...],
    set[int],
    tuple[Any, ...],
    tuple[Any, ...],
    float | None,
    float | None,
]


class ChainResult(list["Chain"]):
    """List[Chain] subclass carrying an optional *hint* for the caller.

    Behaves exactly like ``list`` (len, iteration, indexing, slicing …)
    so every existing caller that does ``chains = extract_chains(…)``
    keeps working unchanged.  The *hint* attribute is only consumed by
    callers that know about it (e.g. the MCP wrapper).
    """

    hint: str | None

    def __init__(self, chains: list[Chain] | None = None, *, hint: str | None = None):
        super().__init__(chains or [])
        self.hint = hint


def _scalar_to_epoch(value: Any) -> float:
    """Per-value fallback; naive datetimes are UTC, unparseable values 0.0."""
    if value is None:
        return 0.0
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    parsed = to_epoch_seconds(pa.array([str(value)]), errors="null")[0].as_py()
    return 0.0 if parsed is None else float(parsed)


def parse_timestamps_to_epoch(ts_list: list[Any]) -> list[float]:
    """Convert a list of timestamps to epoch seconds (naive = UTC, null or unparseable = 0.0).

    Vectorised through :func:`hypertopos.utils.time.to_epoch_seconds`; only a
    list that pyarrow cannot type as one column (e.g. mixed datetimes and
    strings) falls back to per-value conversion.
    """
    if not ts_list:
        return []
    try:
        epochs = to_epoch_seconds(pa.array(ts_list), errors="null")
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        return [_scalar_to_epoch(v) for v in ts_list]
    return cast("list[float]", pc.fill_null(epochs, 0.0).to_pylist())


@dataclass
class Chain:
    """A sequence of linked entities with temporal ordering."""

    chain_id: str
    keys: list[str]  # entity keys in order [A, B, C, D]
    event_keys: list[str]  # event PKs per hop
    hop_count: int
    is_cyclic: bool  # last key == first key
    time_span_hours: float
    categories: list[str]  # per-hop categorical values (e.g. currency, type)
    amounts: list[float]  # per-hop numeric values
    amount_decay: float  # last_amount / first_amount (0 if first=0)
    # Per-hop bank pair (from_bank, to_bank). Empty when bank data is
    # not available at extraction time. Populated by callers passing
    # `event_banks=[(from_bank, to_bank), ...]` to extract_chains.
    banks: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Mutable default safe via post_init.
        if self.banks is None:
            self.banks = []

    def cross_bank_count(self) -> int:
        """Number of distinct banks the chain transits (across all
        from_bank / to_bank values per hop). High counts flag
        jurisdictional layering — textbook AML indicator."""
        if not self.banks:
            return 0
        seen: set[str] = set()
        for fb, tb in self.banks:
            seen.add(fb)
            seen.add(tb)
        return len(seen)

    def amount_monotone_decreasing(self) -> bool:
        """True iff amounts strictly decrease at every hop AND the
        chain is not cyclic. Textbook structuring pattern — funds
        split into smaller amounts at each layering step. Cyclic
        chains can also produce strictly decreasing amounts (e.g.
        decay round-trip) but those are not structuring; gate on
        is_cyclic to keep the flag meaning specific to layering."""
        if self.is_cyclic or len(self.amounts) < 2:
            return False
        return all(self.amounts[i + 1] < self.amounts[i] for i in range(len(self.amounts) - 1))

    def to_dict(
        self,
        *,
        delta_by_key: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Serialise the chain to a JSON-safe dict.

        ``delta_by_key`` is an optional pre-fetched mapping from entity key
        to that entity's polygon delta vector against the chain's anchor
        pattern. When supplied, the output gains an ``edge_potentials``
        field — one Euclidean distance ``||delta(keys[i]) - delta(keys[i+1])||``
        per consecutive-pair hop. Element is ``None`` when either endpoint
        lacks a delta entry, the vectors have mismatched or zero length,
        or the resulting distance is non-finite (NaN / inf — strict-JSON
        sanitised). When ``delta_by_key`` is None, ``edge_potentials`` is
        a list of ``None`` of the appropriate length.
        """
        amt_std = float(np.std(self.amounts)) if self.amounts else 0.0
        amt_mean = float(np.mean(self.amounts)) if self.amounts else 0.0
        return {
            "chain_id": self.chain_id,
            "keys": self.keys,
            "event_keys": self.event_keys,
            "hop_count": self.hop_count,
            "is_cyclic": self.is_cyclic,
            "time_span_hours": self.time_span_hours,
            "n_distinct_categories": len(set(self.categories)),
            "categories": sorted(set(self.categories)),
            "amount_decay": self.amount_decay,
            "total_amount": sum(self.amounts),
            "amount_std": amt_std,
            "amount_cv": (amt_std / amt_mean) if amt_mean > 0 else 0.0,
            "amount_max": max(self.amounts) if self.amounts else 0.0,
            "amount_min": min(self.amounts) if self.amounts else 0.0,
            "n_unique_keys": len(set(self.keys)),
            "cross_bank_count": self.cross_bank_count(),
            "amount_monotone_decreasing": (self.amount_monotone_decreasing()),
            "edge_potentials": _compute_edge_potentials(
                self.keys,
                delta_by_key,
            ),
        }


def _compute_edge_potentials(
    keys: list[str],
    delta_by_key: dict[str, np.ndarray] | None,
) -> list[float | None]:
    """Euclidean distance ``||delta(keys[i]) - delta(keys[i+1])||`` per hop.

    Length is ``max(0, len(keys) - 1)``. Element is ``None`` when:
      - ``delta_by_key`` is None (the caller did not pre-fetch deltas);
      - either endpoint's key is missing from ``delta_by_key``;
      - the two delta vectors have mismatched or zero length;
      - the computed distance is not finite (NaN / inf).

    NaN / inf protection keeps the result strict-JSON-safe: any non-finite
    distance is replaced by ``None`` before serialisation.
    """
    n_hops = max(0, len(keys) - 1)
    if n_hops == 0:
        return []
    if delta_by_key is None:
        return [None] * n_hops

    out: list[float | None] = []
    for i in range(n_hops):
        a = delta_by_key.get(keys[i])
        b = delta_by_key.get(keys[i + 1])
        if a is None or b is None:
            out.append(None)
            continue
        if a.shape != b.shape or a.size == 0:
            out.append(None)
            continue
        d = float(np.linalg.norm(a - b))
        if not math.isfinite(d):
            out.append(None)
            continue
        out.append(d)
    return out


@without_cyclic_gc
def extract_chains(
    from_keys: list[str],
    to_keys: list[str],
    event_pks: list[str],
    timestamps: list[float] | None = None,
    categories: list[str] | None = None,
    amounts: list[float] | None = None,
    time_window_hours: int = 168,
    max_hops: int = 15,
    min_hops: int = 2,
    sample_size: int | None = None,
    max_chains: int = 100_000,
    seed_nodes: list[str] | None = None,
    bidirectional: bool = False,
    event_banks: list[tuple[str, str]] | None = None,
) -> list[Chain]:
    """Extract transaction chains via temporal BFS.

    Args:
        from_keys: Source entity key per event.
        to_keys: Destination entity key per event.
        event_pks: Event primary keys.
        timestamps: Unix timestamps (seconds). None = ignore temporal ordering.
        categories: Categorical value per event (e.g. currency, type). None = not tracked.
        amounts: Numeric value per event (e.g. amount). None = not tracked.
        time_window_hours: Max gap between consecutive hops.
        max_hops: Maximum chain length.
        min_hops: Minimum chain length (filter shorter chains).
        sample_size: Limit starting nodes for BFS.
        max_chains: Global limit on chains produced. Dense graphs can produce
            millions of chains; this cap prevents hang/OOM. Default 100,000.
        seed_nodes: If provided, restrict BFS starting nodes to this list.
            The sample_size limit still applies after seed_nodes filtering.
        bidirectional: If True, each edge A→B also creates a reverse edge
            B→A in the adjacency graph. Useful for undirected relationship
            analysis (e.g. communication networks). Default False.
        event_banks: Optional list of (from_bank, to_bank) per event,
            same length as from_keys. When provided, the resulting
            chains carry per-hop bank pairs in `Chain.banks`, enabling
            `cross_bank_count` derivation (textbook AML jurisdictional
            layering indicator). Defaults to None (no bank tracking).

    Returns:
        List of Chain objects sorted by hop_count descending.
    """
    n = len(from_keys)
    window_secs = time_window_hours * 3600.0

    # If sample_size is set, subsample events to bound adjacency memory.
    # Factor of 20 allows enough events per starting node for multi-hop chains.
    # Seeded runs keep every event: a dropped event would cut a seed's chain.
    if sample_size is not None and not seed_nodes:
        max_events = sample_size * 20
        if n > max_events:
            rng = np.random.default_rng(42)
            indices = np.sort(rng.choice(n, size=max_events, replace=False))
            from_keys = [from_keys[i] for i in indices]
            to_keys = [to_keys[i] for i in indices]
            event_pks = [event_pks[i] for i in indices]
            if timestamps:
                timestamps = [timestamps[i] for i in indices]
            if categories:
                categories = [categories[i] for i in indices]
            if amounts:
                amounts = [amounts[i] for i in indices]
            if event_banks:
                event_banks = [event_banks[i] for i in indices]
            n = max_events

    # Build adjacency using integer IDs for speed (string→int mapping)
    _key_to_id: dict[str, int] = {}
    _id_to_key: list[str] = []

    def _intern(k: str) -> int:
        kid = _key_to_id.get(k)
        if kid is None:
            kid = len(_id_to_key)
            _key_to_id[k] = kid
            _id_to_key.append(k)
        return kid

    adj: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for i in range(n):
        fk = from_keys[i]
        tk = to_keys[i]
        if fk is None or tk is None:
            continue
        fk_id = _intern(fk)
        tk_id = _intern(tk)
        ts = timestamps[i] if timestamps else 0.0
        cur = categories[i] if categories else ""
        amt = amounts[i] if amounts else 0.0
        adj[fk_id].append((tk_id, ts, i, cur, amt))  # use index i instead of event_pk string
        if bidirectional:
            adj[tk_id].append((fk_id, ts, i, cur, amt))

    # Overlap check: chains require to_key of one event to match from_key
    # of another.  If the two sets are completely disjoint, no chain can
    # ever form — return early with an explanatory hint.
    from_set = {fk for fk in from_keys if fk is not None}
    to_set = {tk for tk in to_keys if tk is not None}
    if not (from_set & to_set):
        return ChainResult(
            hint=(
                "No chains possible: from_col and to_col value sets "
                "are completely disjoint (no overlap). Chains require "
                "a to_key in one event to appear as a from_key in "
                "another event. Check that from_col and to_col are "
                "correct — they should represent the same entity "
                "namespace (e.g. both are account IDs)."
            )
        )

    # Sort adjacency lists by timestamp
    if timestamps:
        for k in adj:
            adj[k].sort(key=lambda x: x[1])

    # Select starting nodes (as integer IDs)
    if seed_nodes is not None:
        start_nodes = [
            _key_to_id[k] for k in seed_nodes if k in _key_to_id and _key_to_id[k] in adj
        ]
    else:
        start_nodes = list(adj.keys())
    if sample_size and len(start_nodes) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(start_nodes), size=sample_size, replace=False)
        start_nodes = [start_nodes[i] for i in idx]

    # Per-seed chain budget ensures all seeds get explored, not just the
    # first few hubs.  Default: distribute max_chains evenly across seeds
    # with a minimum of 3 per seed.
    per_seed_budget = max(3, max_chains // max(len(start_nodes), 1))

    has_ts = bool(timestamps)
    chains = _bfs_seed_batch(
        start_nodes,
        adj,
        max_hops,
        min_hops,
        window_secs,
        has_ts,
        per_seed_budget,
        max_chains,
    )
    # Drop chains whose entity path is a strict prefix of a longer chain
    # (investigatively redundant); chain_ids are reassigned in the new order.
    chains, _ = _dedup_strict_prefixes(chains, max_chains=max_chains)

    # Translate int IDs back to string keys. Capture raw event indices
    # BEFORE event_keys translation so per-event bank lookup works on
    # the integer indices.
    for chain in chains:
        # keys / event_keys still hold int IDs at this stage (see _emit_chain)
        raw_event_indices = cast("list[int]", chain.event_keys)
        chain.keys = [_id_to_key[k] for k in cast("list[int]", chain.keys)]
        chain.event_keys = [event_pks[i] for i in raw_event_indices]
        if event_banks is not None:
            chain.banks = [event_banks[i] for i in raw_event_indices]

    # Sort by hop_count descending
    chains.sort(key=lambda c: c.hop_count, reverse=True)
    return ChainResult(chains[:max_chains])


def _bfs_seed_batch(
    start_nodes: list[int],
    adj: dict[int, list[tuple[Any, ...]]],
    max_hops: int,
    min_hops: int,
    window_secs: float,
    has_timestamps: bool,
    per_seed_budget: int,
    max_chains: int,
) -> list[Chain]:
    """Run DFS from a batch of seed nodes.

    The stack holds one cursor per expanded node, ``[frame, neighbors, i]``,
    not one built frame per child: the per-seed budget usually ends a seed
    long before its children are visited, so a child's path tuples and sets
    are built only when the cursor reaches it. Cursors walk their neighbors
    from the last to the first, which is the order a stack of pushed children
    would pop them in.
    """
    from hypertopos.engine.adjacency import temporal_bisect

    chains: list[Chain] = []
    visited_chains: set[frozenset[int]] = set()

    for start in start_nodes:
        seed_start_count = len(chains)
        stack: list[list[Any]] = []
        frame: _DfsFrame | None = (start, (start,), {start}, (), set(), (), (), None, None)

        while True:
            if frame is None:
                # Next unvisited child of the innermost node that has one.
                while stack:
                    cursor = stack[-1]
                    parent, neighbors, i = cursor
                    keys_set, events_set = parent[2], parent[4]
                    while i >= 0:
                        to_key, _ts, epk, _cur, _amt = neighbors[i]
                        if epk not in events_set and to_key not in keys_set:
                            break
                        i -= 1
                    if i < 0:
                        stack.pop()
                        continue
                    cursor[2] = i - 1
                    to_key, ts, epk, cur, amt = neighbors[i]
                    frame = (
                        to_key,
                        parent[1] + (to_key,),
                        keys_set | {to_key},
                        parent[3] + (epk,),
                        events_set | {epk},
                        parent[5] + (cur,),
                        parent[6] + (amt,),
                        parent[7] if parent[7] is not None else ts,
                        ts,
                    )
                    break
                if frame is None:
                    break

            (
                current,
                path_keys,
                path_keys_set,
                path_events,
                path_events_set,
                path_categories,
                path_amounts,
                first_ts,
                last_ts,
            ) = expanding = frame
            frame = None

            if len(path_keys) - 1 >= max_hops:
                _emit_chain(
                    chains,
                    len(chains),
                    path_keys,
                    path_events,
                    path_categories,
                    path_amounts,
                    visited_chains,
                    min_hops,
                    first_ts,
                    last_ts,
                )
                if len(chains) >= max_chains or len(chains) - seed_start_count >= per_seed_budget:
                    break
                continue

            extended = False
            has_child = False
            neighbors = adj.get(current, [])
            if has_timestamps and last_ts is not None and neighbors:
                neighbors = temporal_bisect(
                    neighbors, ts_from=last_ts, ts_to=last_ts + window_secs, ts_index=1
                )
            for to_key, ts, epk, cur, amt in neighbors:
                if epk in path_events_set:
                    continue

                if to_key == start and len(path_keys) >= min_hops:
                    _emit_chain(
                        chains,
                        len(chains),
                        path_keys + (to_key,),
                        path_events + (epk,),
                        path_categories + (cur,),
                        path_amounts + (amt,),
                        visited_chains,
                        min_hops,
                        first_ts if first_ts is not None else ts,
                        ts,
                    )
                    if (
                        len(chains) >= max_chains
                        or len(chains) - seed_start_count >= per_seed_budget
                    ):
                        break
                    extended = True
                    continue

                if to_key in path_keys_set:
                    continue

                extended = has_child = True

            if len(chains) >= max_chains or len(chains) - seed_start_count >= per_seed_budget:
                break

            if has_child:
                stack.append([expanding, neighbors, len(neighbors) - 1])

            if not extended and len(path_keys) - 1 >= min_hops:
                _emit_chain(
                    chains,
                    len(chains),
                    path_keys,
                    path_events,
                    path_categories,
                    path_amounts,
                    visited_chains,
                    min_hops,
                    first_ts,
                    last_ts,
                )
                if len(chains) >= max_chains or len(chains) - seed_start_count >= per_seed_budget:
                    break
        if len(chains) >= max_chains:
            break

    return chains


def _dedup_strict_prefixes(
    chains: list[Chain],
    *,
    max_chains: int = 100_000,
) -> tuple[list[Chain], int]:
    """Drop chains whose entity-key sequence (`keys`) is a strict
    ordered prefix of another chain's `keys`, then reassign
    chain_ids from the surviving order.

    Strict prefix means chain X.keys equals chain Y.keys[:len(X)]
    AND len(X) < len(Y). Y traverses every entity X traverses plus
    more — investigatively redundant since chain-coherent loop tools
    key off entity sequence, not event sequence.

    Why entity sequence (`keys`) not event sequence (`event_keys`):
    two chains may visit the same entity path through different
    transactions (e.g. parallel edges between the same accounts).
    Entity-path is what investigators reason about and what `chain_keys`
    in the points table stores; event_keys is the underlying tx
    sequence persisted only in chain features. The investigatively-
    redundant cleanup operates on the entity basis.

    Algorithm: sort longest-first, register every proper prefix in
    a hash set per chain. A shorter chain whose entity path matches
    one of the registered prefixes is subsumed. O(N x L_avg) work.

    Note: this changes chain_id ordering relative to raw extraction
    order — surviving chains carry sequential CHAIN-NNNNNN ids
    reflecting post-dedup order, with the longest chains assigned
    first. Chain_ids are opaque per architecture; downstream code
    must not rely on extraction-order semantics.
    """
    import dataclasses

    # Stable sort longest-first so prefixes get registered before any
    # shorter chain reaches the dedup check.
    sorted_chains = sorted(chains, key=lambda c: -len(c.keys))

    extended_prefixes: set[tuple[str, ...]] = set()
    deduped: list[Chain] = []
    removed = 0
    for c in sorted_chains:
        ek = tuple(c.keys)
        if ek in extended_prefixes:
            removed += 1
            continue
        deduped.append(
            dataclasses.replace(
                c,
                chain_id=f"CHAIN-{len(deduped):06d}",
            ),
        )
        if len(deduped) >= max_chains:
            break
        for k in range(1, len(ek)):
            extended_prefixes.add(ek[:k])
    return deduped, removed


def _emit_chain(
    chains: list[Chain],
    counter: int,
    keys: tuple[Any, ...],
    events: tuple[Any, ...],
    categories: tuple[Any, ...],
    amounts: tuple[Any, ...],
    visited: set[frozenset[int]],
    min_hops: int,
    first_ts: float | None = None,
    last_ts: float | None = None,
) -> None:
    """Create and append a chain if it passes dedup and min_hops.

    keys and events use integer IDs during BFS; translated to strings
    after all chains are collected (in extract_chains).
    """
    hop_count = len(keys) - 1
    if hop_count < min_hops:
        return

    # Dedup by frozenset of event indices — O(L) construction vs O(L log L) sort
    if len(visited) < _DEDUP_CAP:
        event_sig = frozenset(events)
        if event_sig in visited:
            return
        visited.add(event_sig)
        if len(visited) == _DEDUP_CAP:
            import warnings

            warnings.warn(
                f"chain dedup stopped after {_DEDUP_CAP} distinct event sequences; "
                "later chains may repeat an earlier one",
                stacklevel=2,
            )

    is_cyclic = keys[-1] == keys[0] if len(keys) >= 3 else False
    if first_ts is not None and last_ts is not None:
        time_span = (last_ts - first_ts) / 3600.0
    else:
        time_span = 0.0

    amount_decay = amounts[-1] / amounts[0] if amounts and amounts[0] != 0 else 0.0

    chains.append(
        Chain(
            chain_id=f"CHAIN-{counter:06d}",
            keys=list(keys),
            event_keys=list(events),
            hop_count=hop_count,
            is_cyclic=is_cyclic,
            time_span_hours=time_span,
            categories=list(categories),
            amounts=list(amounts),
            amount_decay=amount_decay,
        )
    )
