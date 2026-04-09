# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Chain extraction from event data — temporal BFS over transaction graphs.

Finds sequences of linked events (A→B→C→D) within a time window.
Used for: money laundering chain detection, supply chain path analysis.
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

_DEDUP_CAP = 500_000  # max entries in visited set to prevent unbounded memory


class ChainResult(list):
    """List[Chain] subclass carrying an optional *hint* for the caller.

    Behaves exactly like ``list`` (len, iteration, indexing, slicing …)
    so every existing caller that does ``chains = extract_chains(…)``
    keeps working unchanged.  The *hint* attribute is only consumed by
    callers that know about it (e.g. the MCP wrapper).
    """

    hint: str | None

    def __init__(self, chains: list | None = None, *, hint: str | None = None):
        super().__init__(chains or [])
        self.hint = hint


def parse_timestamps_to_epoch(ts_list: list) -> list[float]:
    """Convert a list of mixed-type timestamps to epoch seconds."""
    if not ts_list:
        return []
    sample = ts_list[0]
    if hasattr(sample, "timestamp"):
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            arr = pa.array(ts_list)
            if pa.types.is_timestamp(arr.type):
                epoch_us = pc.cast(arr, pa.int64())
                unit = arr.type.unit
                divisors = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
                d = divisors.get(unit, 1e6)
                return (epoch_us.to_numpy(zero_copy_only=False).astype(np.float64) / d).tolist()
        except Exception:
            pass
    elif isinstance(sample, (int, float)):
        return [float(t) for t in ts_list]
    result = []
    for t in ts_list:
        if hasattr(t, "timestamp"):
            result.append(t.timestamp())
        elif isinstance(t, (int, float)):
            result.append(float(t))
        else:
            try:
                dt = datetime.fromisoformat(str(t).replace("/", "-"))
                result.append(dt.timestamp())
            except Exception:
                result.append(0.0)
    return result


@dataclass
class Chain:
    """A sequence of linked entities with temporal ordering."""

    chain_id: str
    keys: list[str]        # entity keys in order [A, B, C, D]
    event_keys: list[str]  # event PKs per hop
    hop_count: int
    is_cyclic: bool        # last key == first key
    time_span_hours: float
    categories: list[str]  # per-hop categorical values (e.g. currency, type)
    amounts: list[float]   # per-hop numeric values
    amount_decay: float    # last_amount / first_amount (0 if first=0)

    def to_dict(self) -> dict:
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
            "categories": list(set(self.categories)),
            "amount_decay": self.amount_decay,
            "total_amount": sum(self.amounts),
            "amount_std": amt_std,
            "amount_cv": (amt_std / amt_mean) if amt_mean > 0 else 0.0,
            "amount_max": max(self.amounts) if self.amounts else 0.0,
            "amount_min": min(self.amounts) if self.amounts else 0.0,
            "n_unique_keys": len(set(self.keys)),
        }


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

    Returns:
        List of Chain objects sorted by hop_count descending.
    """
    n = len(from_keys)
    window_secs = time_window_hours * 3600.0

    # If sample_size is set, subsample events to bound adjacency memory.
    # Factor of 20 allows enough events per starting node for multi-hop chains.
    if sample_size is not None:
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

    adj: dict[int, list[tuple]] = defaultdict(list)
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
        return ChainResult(hint=(
            "No chains possible: from_col and to_col value sets "
            "are completely disjoint (no overlap). Chains require "
            "a to_key in one event to appear as a from_key in "
            "another event. Check that from_col and to_col are "
            "correct — they should represent the same entity "
            "namespace (e.g. both are account IDs)."
        ))

    # Sort adjacency lists by timestamp
    if timestamps:
        for k in adj:
            adj[k].sort(key=lambda x: x[1])

    # Select starting nodes (as integer IDs)
    if seed_nodes is not None:
        start_nodes = [
            _key_to_id[k] for k in seed_nodes
            if k in _key_to_id and _key_to_id[k] in adj
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

    n_workers = min(4, max(1, len(start_nodes) // 100))
    has_ts = bool(timestamps)

    if n_workers <= 1 or len(start_nodes) < 400:
        chains = _bfs_seed_batch(
            start_nodes, adj, max_hops, min_hops,
            window_secs, has_ts, per_seed_budget, max_chains,
        )
    else:
        chains = _parallel_bfs(
            start_nodes, adj, max_hops, min_hops,
            window_secs, has_ts, per_seed_budget, max_chains,
            n_workers,
        )

    # Translate int IDs back to string keys
    for chain in chains:
        chain.keys = [_id_to_key[k] for k in chain.keys]
        chain.event_keys = [event_pks[i] for i in chain.event_keys]

    # Sort by hop_count descending
    chains.sort(key=lambda c: c.hop_count, reverse=True)
    return ChainResult(chains[:max_chains])


def _worker_fn(args: tuple) -> list[Chain]:
    """Multiprocessing worker — loads adj from disk, runs BFS on seed chunk."""
    (adj_path, seeds, max_hops, min_hops,
     window_secs, has_ts, per_seed_budget, chunk_budget) = args
    import pickle
    with open(adj_path, "rb") as f:
        adj = pickle.load(f)
    return _bfs_seed_batch(
        seeds, adj, max_hops, min_hops,
        window_secs, has_ts, per_seed_budget, chunk_budget,
    )


def _parallel_bfs(
    start_nodes: list[int],
    adj: dict[int, list[tuple]],
    max_hops: int,
    min_hops: int,
    window_secs: float,
    has_ts: bool,
    per_seed_budget: int,
    max_chains: int,
    n_workers: int,
) -> list[Chain]:
    """Disk-based parallel BFS — pickle adj once, workers load from file.

    Each worker loads adj from a temp file (avoids pickle-through-pipe overhead).
    Post-merge dedup ensures correctness across worker boundaries.
    """
    import pickle
    import tempfile
    import warnings
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    # Serialize adj to temp file (one write, N reads)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            pickle.dump(adj, tmp)
            adj_path = tmp.name
            tmp_path = tmp.name

        chunk_size = (len(start_nodes) + n_workers - 1) // n_workers
        chunk_budget = max_chains // n_workers + 1
        tasks = [
            (adj_path, start_nodes[i:i + chunk_size], max_hops, min_hops,
             window_secs, has_ts, per_seed_budget, chunk_budget)
            for i in range(0, len(start_nodes), chunk_size)
        ]

        chains: list[Chain] = []
        # ProcessPoolExecutor can fail to start on some Windows environments
        # (for example, restricted handle creation under pytest). Fall back to
        # the serial BFS path instead of failing the whole extraction.
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for batch in pool.map(_worker_fn, tasks):
                    chains.extend(batch)
        except (OSError, BrokenProcessPool):
            return _bfs_seed_batch(
                start_nodes, adj, max_hops, min_hops,
                window_secs, has_ts, per_seed_budget, max_chains,
            )

        # Post-merge dedup via frozenset — ensures correctness even when
        # per-worker dedup caps (_DEDUP_CAP) are hit.
        seen: set[frozenset] = set()
        deduped: list[Chain] = []
        pre_dedup = len(chains)
        for c in chains:
            sig = frozenset(c.event_keys)
            if sig not in seen:
                seen.add(sig)
                deduped.append(c)
                if len(deduped) >= max_chains:
                    break
        if pre_dedup - len(deduped) > pre_dedup * 0.1:
            warnings.warn(
                f"Chain dedup removed {pre_dedup - len(deduped)} duplicates "
                f"({(pre_dedup - len(deduped)) / pre_dedup:.0%}) across workers.", stacklevel=2
            )
        return deduped
    finally:
        import os
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def _bfs_seed_batch(
    start_nodes: list[int],
    adj: dict[int, list[tuple]],
    max_hops: int,
    min_hops: int,
    window_secs: float,
    has_timestamps: bool,
    per_seed_budget: int,
    max_chains: int,
) -> list[Chain]:
    """Run DFS from a batch of seed nodes. Designed for multiprocessing."""
    chains: list[Chain] = []
    visited_chains: set[frozenset] = set()

    for start in start_nodes:
        seed_start_count = len(chains)
        stack = [(start, (start,), {start}, (), set(), (), (), None, None)]

        while stack:
            (current, path_keys, path_keys_set,
             path_events, path_events_set,
             path_categories, path_amounts,
             first_ts, last_ts) = stack.pop()

            if len(path_keys) - 1 >= max_hops:
                _emit_chain(
                    chains, len(chains), path_keys, path_events,
                    path_categories, path_amounts, visited_chains,
                    min_hops, first_ts, last_ts,
                )
                if (len(chains) >= max_chains
                        or len(chains) - seed_start_count >= per_seed_budget):
                    break
                continue

            extended = False
            for to_key, ts, epk, cur, amt in adj.get(current, []):
                if has_timestamps and last_ts is not None:
                    if ts < last_ts:
                        continue
                    if (ts - last_ts) > window_secs:
                        continue

                if epk in path_events_set:
                    continue

                new_keys = path_keys + (to_key,)
                new_events = path_events + (epk,)
                new_categories = path_categories + (cur,)
                new_amounts = path_amounts + (amt,)
                new_first_ts = first_ts if first_ts is not None else ts
                new_last_ts = ts

                if to_key == start and len(new_keys) >= min_hops + 1:
                    _emit_chain(
                        chains, len(chains), new_keys, new_events,
                        new_categories, new_amounts, visited_chains,
                        min_hops, new_first_ts, new_last_ts,
                    )
                    if (len(chains) >= max_chains
                            or len(chains) - seed_start_count >= per_seed_budget):
                        break
                    extended = True
                    continue

                if to_key in path_keys_set:
                    continue

                new_keys_set = path_keys_set | {to_key}
                new_events_set = path_events_set | {epk}
                stack.append((
                    to_key, new_keys, new_keys_set, new_events,
                    new_events_set, new_categories, new_amounts,
                    new_first_ts, new_last_ts,
                ))
                extended = True

            if (len(chains) >= max_chains
                    or len(chains) - seed_start_count >= per_seed_budget):
                break

            if not extended and len(path_keys) - 1 >= min_hops:
                _emit_chain(
                    chains, len(chains), path_keys, path_events,
                    path_categories, path_amounts, visited_chains,
                    min_hops, first_ts, last_ts,
                )
                if (len(chains) >= max_chains
                        or len(chains) - seed_start_count >= per_seed_budget):
                    break
        if len(chains) >= max_chains:
            break

    return chains


def _emit_chain(
    chains: list[Chain],
    counter: int,
    keys: tuple,
    events: tuple,
    categories: tuple,
    amounts: tuple,
    visited: set[frozenset],
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

    is_cyclic = keys[-1] == keys[0] if len(keys) >= 3 else False
    if first_ts is not None and last_ts is not None:
        time_span = (last_ts - first_ts) / 3600.0
    else:
        time_span = 0.0

    amount_decay = amounts[-1] / amounts[0] if amounts and amounts[0] != 0 else 0.0

    chains.append(Chain(
        chain_id=f"CHAIN-{counter:06d}",
        keys=list(keys),
        event_keys=list(events),
        hop_count=hop_count,
        is_cyclic=is_cyclic,
        time_span_hours=time_span,
        categories=list(categories),
        amounts=list(amounts),
        amount_decay=amount_decay,
    ))
