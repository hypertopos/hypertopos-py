# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Structuring motif enumerator — A→B→C→D with amount gating + temporal ordering.

Single source of truth for structuring detection. Used by:
- ``navigation.navigator._enumerate_structuring`` (single-seed runtime path)
- ``engine.edge_features.compute_find_motif_structuring`` (build-time sweep)

Algorithm: open 3-hop chains with
  hop1 (A→B) amount >= amt1_min,
  hop2 (B→C), hop3 (C→D) amount <= amt2_max,
  ts_ab < ts_bc < ts_cd, ts_cd - ts_ab <= time_window_sec,
  D, C ∉ {A, B} and D != C, no null/non-positive amounts at any hop.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pyarrow as pa


def _build_outgoing_index(
    edges: pa.Table,
) -> dict[str, list[tuple[str, float, float, str]]]:
    out: dict[str, list[tuple[str, float, float, str]]] = defaultdict(list)
    from_keys = edges["from_key"].to_pylist()
    to_keys = edges["to_key"].to_pylist()
    timestamps = edges["timestamp"].to_pylist()
    amounts = edges["amount"].to_pylist()
    event_keys = edges["event_key"].to_pylist()
    for fk, tk, ts, amt, ek in zip(
        from_keys, to_keys, timestamps, amounts, event_keys, strict=False,
    ):
        out[fk].append((tk, ts, amt, ek))
    return out


def enumerate_structuring_for_seed(
    seed: str,
    edges: pa.Table,
    *,
    time_window_sec: float,
    amt1_min: float,
    amt2_max: float,
    max_instances: int = 50,
) -> list[dict[str, Any]]:
    """Single-seed enumeration of structuring A→B→C→D.

    Returns list of motif dicts with keys ``edges``, ``timestamps``, ``amounts``.
    """
    out_idx = _build_outgoing_index(edges)
    out1 = out_idx.get(seed, [])
    large_first = [
        (b, ts, amt, ek)
        for (b, ts, amt, ek) in out1
        if b != seed and amt is not None and amt > 0 and amt >= amt1_min
    ]
    if not large_first:
        return []

    results: list[dict[str, Any]] = []
    for (b, ts_ab, amt_ab, ek_ab) in large_first:
        for (c, ts_bc, amt_bc, ek_bc) in out_idx.get(b, []):
            if c in (seed, b):
                continue
            if ts_bc <= ts_ab or ts_bc - ts_ab > time_window_sec:
                continue
            if amt_bc is None or amt_bc <= 0 or amt_bc > amt2_max:
                continue
            for (d, ts_cd, amt_cd, ek_cd) in out_idx.get(c, []):
                if d in (seed, b, c):
                    continue
                if ts_cd <= ts_bc or ts_cd - ts_ab > time_window_sec:
                    continue
                if amt_cd is None or amt_cd <= 0 or amt_cd > amt2_max:
                    continue
                results.append({
                    "edges": [
                        (seed, b, ek_ab),
                        (b, c, ek_bc),
                        (c, d, ek_cd),
                    ],
                    "timestamps": [ts_ab, ts_bc, ts_cd],
                    "amounts": [amt_ab, amt_bc, amt_cd],
                })
                if len(results) >= max_instances:
                    return results
    return results


def enumerate_structuring_event_keys(
    edges: pa.Table,
    *,
    time_window_sec: float,
    amt1_min: float,
    amt2_max: float,
) -> set[str]:
    """All-seeds sweep: return event_keys participating in ANY structuring motif.

    Build-time helper for the ``find_motif_structuring`` edge-derived dimension.
    """
    out_idx = _build_outgoing_index(edges)
    flagged: set[str] = set()
    for seed, out1 in out_idx.items():
        large_first = [
            (b, ts, amt, ek)
            for (b, ts, amt, ek) in out1
            if b != seed and amt is not None and amt > 0 and amt >= amt1_min
        ]
        for (b, ts_ab, _amt_ab, ek_ab) in large_first:
            for (c, ts_bc, amt_bc, ek_bc) in out_idx.get(b, []):
                if c in (seed, b):
                    continue
                if ts_bc <= ts_ab or ts_bc - ts_ab > time_window_sec:
                    continue
                if amt_bc is None or amt_bc <= 0 or amt_bc > amt2_max:
                    continue
                for (d, ts_cd, amt_cd, ek_cd) in out_idx.get(c, []):
                    if d in (seed, b, c):
                        continue
                    if ts_cd <= ts_bc or ts_cd - ts_ab > time_window_sec:
                        continue
                    if amt_cd is None or amt_cd <= 0 or amt_cd > amt2_max:
                        continue
                    flagged.add(ek_ab)
                    flagged.add(ek_bc)
                    flagged.add(ek_cd)
    return flagged
