# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""extract_chains output is pinned: traversal changes must not change which chains come out."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest

from hypertopos.engine.chains import extract_chains


def _hub_graph(n_events: int = 6000, n_nodes: int = 300, seed: int = 7):
    """Random transfers plus two hubs, so per-seed budgets cut exploration short.

    Integer-valued stdlib randomness keeps the input identical on every platform.
    """
    rng = random.Random(seed)
    src = [
        rng.choice((0, 1)) if rng.random() < 0.25 else rng.randrange(n_nodes)
        for _ in range(n_events)
    ]
    dst = [rng.randrange(n_nodes) for _ in range(n_events)]
    keys = [f"N{i}" for i in range(n_nodes)]
    from_keys = [keys[i] for i in src]
    to_keys = [keys[i] for i in dst]
    event_pks = [f"E{i}" for i in range(n_events)]
    timestamps = sorted(float(rng.randrange(30 * 86400)) for _ in range(n_events))
    categories = [rng.choice(("USD", "EUR", "PLN")) for _ in range(n_events)]
    amounts = [float(rng.randrange(10, 50_000)) for _ in range(n_events)]
    banks = [(f"B{i % 5}", f"B{j % 5}") for i, j in zip(src, dst, strict=True)]
    return from_keys, to_keys, event_pks, timestamps, categories, amounts, banks


def _fingerprint(chains) -> str:
    rows = [dataclasses.asdict(c) for c in chains]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


# Recorded from the implementation that pushed one fully built frame per child.
_EXPECTED = {
    "temporal": "c444bcaaf583a9723b1f15b7c755973c4669788f5301a253c1092a11de678c21",
    "no_timestamps": "f9d1a0c3af0fb0b0bc7c7711b462733562d67397142f08a912b82690c29850a8",
    "bidirectional_short": "1aa9f8cf74e7a0cb099b3a06b48612cb9c7fde7198166372cbb61060baae6091",
    "seeded": "a64060f0ef718b49c64cdd53d12b6b5c8312be22311746bc7959b14f9849c6c0",
}


def _run(case: str):
    fk, tk, pks, ts, cats, amts, banks = _hub_graph()
    if case == "temporal":
        return extract_chains(
            fk,
            tk,
            pks,
            ts,
            cats,
            amts,
            time_window_hours=48,
            max_hops=6,
            max_chains=2000,
            event_banks=banks,
        )
    if case == "no_timestamps":
        return extract_chains(fk, tk, pks, None, cats, amts, max_hops=4, max_chains=1500)
    if case == "bidirectional_short":
        return extract_chains(
            fk,
            tk,
            pks,
            ts,
            None,
            None,
            time_window_hours=24,
            max_hops=3,
            min_hops=1,
            max_chains=1000,
            bidirectional=True,
        )
    return extract_chains(
        fk,
        tk,
        pks,
        ts,
        cats,
        amts,
        time_window_hours=72,
        max_hops=8,
        max_chains=600,
        seed_nodes=["N0", "N1", "N5", "N9"],
    )


@pytest.mark.parametrize("case", sorted(_EXPECTED))
def test_extract_chains_output_is_unchanged(case: str) -> None:
    chains = _run(case)
    assert len(chains) > 50, "the case must produce enough chains to pin traversal order"
    assert _fingerprint(chains) == _EXPECTED[case]
