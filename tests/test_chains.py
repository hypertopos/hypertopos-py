# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for chain extraction engine."""

import pytest

from hypertopos.engine.chains import extract_chains


class TestExtractChains:
    def test_cross_bank_count_zero_when_banks_unset(self):
        """When event_banks is not passed, chains carry empty banks
        list and cross_bank_count returns 0."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
        )
        assert len(chains) == 1
        assert chains[0].banks == []
        assert chains[0].cross_bank_count() == 0

    def test_cross_bank_count_counts_distinct_banks(self):
        """When event_banks is passed, cross_bank_count counts
        distinct bank ids across all from_bank / to_bank pairs."""
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            event_banks=[
                ("BANK1", "BANK2"),
                ("BANK2", "BANK3"),
                ("BANK3", "BANK1"),
            ],
            min_hops=3,
        )
        # Chain A→B→C→D traversing 3 distinct banks (BANK1/2/3)
        relevant = [c for c in chains if c.keys == ["A", "B", "C", "D"]]
        assert relevant
        c = relevant[0]
        assert len(c.banks) == 3
        assert c.cross_bank_count() == 3

    def test_amount_monotone_decreasing_true(self):
        """Strictly decreasing amounts → True."""
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            amounts=[1000.0, 500.0, 200.0],
            min_hops=3,
        )
        relevant = [c for c in chains if c.keys == ["A", "B", "C", "D"]]
        assert relevant
        assert relevant[0].amount_monotone_decreasing() is True

    def test_amount_monotone_decreasing_false_on_increase(self):
        """Any non-decreasing pair → False."""
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            amounts=[1000.0, 500.0, 800.0],  # increase at last hop
            min_hops=3,
        )
        relevant = [c for c in chains if c.keys == ["A", "B", "C", "D"]]
        assert relevant
        assert relevant[0].amount_monotone_decreasing() is False

    def test_amount_monotone_decreasing_false_on_equal(self):
        """Equal amounts at any hop are not strictly decreasing → False."""
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            amounts=[1000.0, 1000.0, 500.0],
            min_hops=3,
        )
        relevant = [c for c in chains if c.keys == ["A", "B", "C", "D"]]
        assert relevant
        assert relevant[0].amount_monotone_decreasing() is False

    def test_to_dict_includes_new_features(self):
        """Chain.to_dict() exposes cross_bank_count and
        amount_monotone_decreasing — these get baked into polygon
        delta via the chain anchor pattern build."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            amounts=[1000.0, 500.0],
            event_banks=[("BANK1", "BANK2"), ("BANK2", "BANK3")],
            min_hops=2,
        )
        d = chains[0].to_dict()
        assert "cross_bank_count" in d
        assert "amount_monotone_decreasing" in d
        assert d["cross_bank_count"] == 3
        assert d["amount_monotone_decreasing"] is True

    def test_simple_chain(self):
        """A→B→C extracted as 2-hop chain."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
        )
        assert len(chains) == 1
        assert chains[0].keys == ["A", "B", "C"]
        assert chains[0].hop_count == 2
        assert chains[0].is_cyclic is False

    def test_dedup_strict_prefixes_removes_subsumed_chain(self):
        """Direct test of the prefix dedup helper. Chain X with
        event_keys [E1,E2] is a strict prefix of chain Y with
        event_keys [E1,E2,E3] — Y carries every hop of X plus more,
        so X is redundant and should be dropped."""
        from hypertopos.engine.chains import Chain, _dedup_strict_prefixes

        x = Chain(
            chain_id="CHAIN-X",
            keys=["A", "B", "C"],
            event_keys=["E1", "E2"],
            hop_count=2,
            is_cyclic=False,
            time_span_hours=0.0,
            categories=[],
            amounts=[],
            amount_decay=0.0,
        )
        y = Chain(
            chain_id="CHAIN-Y",
            keys=["A", "B", "C", "D"],
            event_keys=["E1", "E2", "E3"],
            hop_count=3,
            is_cyclic=False,
            time_span_hours=0.0,
            categories=[],
            amounts=[],
            amount_decay=0.0,
        )
        deduped, removed = _dedup_strict_prefixes([x, y])
        assert len(deduped) == 1
        assert removed == 1
        # The longer chain Y survives; chain_id reassigned to CHAIN-000000
        assert deduped[0].keys == ["A", "B", "C", "D"]
        assert deduped[0].event_keys == ["E1", "E2", "E3"]
        assert deduped[0].chain_id == "CHAIN-000000"

    def test_dedup_strict_prefixes_keeps_diverging_tails(self):
        """Two chains sharing a common prefix but diverging in their
        tails (different events at the divergence point) are siblings,
        not in a prefix relationship — both must survive."""
        from hypertopos.engine.chains import Chain, _dedup_strict_prefixes

        x = Chain(
            chain_id="CHAIN-X",
            keys=["A", "B", "C", "D"],
            event_keys=["E1", "E2", "E3"],
            hop_count=3,
            is_cyclic=False,
            time_span_hours=0.0,
            categories=[],
            amounts=[],
            amount_decay=0.0,
        )
        y = Chain(
            chain_id="CHAIN-Y",
            keys=["A", "B", "C", "E"],
            event_keys=["E1", "E2", "E4"],
            hop_count=3,
            is_cyclic=False,
            time_span_hours=0.0,
            categories=[],
            amounts=[],
            amount_decay=0.0,
        )
        deduped, removed = _dedup_strict_prefixes([x, y])
        assert len(deduped) == 2
        assert removed == 0

    def test_dedup_strict_prefixes_handles_empty(self):
        from hypertopos.engine.chains import _dedup_strict_prefixes

        deduped, removed = _dedup_strict_prefixes([])
        assert deduped == []
        assert removed == 0

    def test_subchain_prefix_dedup_does_not_drop_siblings_from_extract(self):
        """Integration check: when extract_chains produces multiple
        chains starting from different seeds (A and B), BFS produces
        sibling chains that share suffixes but differ in starting
        node — these are NOT in a strict prefix relationship and must
        all survive. Concretely: from edges A→B, B→C, C→D, C→E the
        BFS emits chains starting from A AND B, with branches at C.
        None is a strict prefix of another (different leading event)."""
        chains = extract_chains(
            from_keys=["A", "B", "C", "C"],
            to_keys=["B", "C", "D", "E"],
            event_pks=["TX-1", "TX-2", "TX-3", "TX-4"],
            min_hops=2,
            max_hops=10,
        )
        # No two chains may have entity-key sequences in a strict-prefix
        # relationship (would mean dedup over-aggressively kept a
        # redundant chain or under-aggressively missed dropping one).
        key_seqs = [tuple(c.keys) for c in chains]
        for i, seq_i in enumerate(key_seqs):
            for j, seq_j in enumerate(key_seqs):
                if i == j:
                    continue
                if len(seq_i) < len(seq_j) and seq_j[: len(seq_i)] == seq_i:
                    raise AssertionError(
                        f"chain {i} {seq_i} is a strict prefix of "
                        f"chain {j} {seq_j} — dedup missed it",
                    )

    def test_chain_ids_unique(self):
        """Extraction must produce unique chain_ids: downstream point-table
        reads are keyed by chain_id."""
        # 50 seeds, several chains per seed.
        from_keys: list[str] = []
        to_keys: list[str] = []
        event_pks: list[str] = []
        for seed_idx in range(50):
            seed = f"SEED-{seed_idx}"
            for hop in range(5):
                from_keys.append(seed if hop == 0 else f"M-{seed_idx}-{hop - 1}")
                to_keys.append(f"M-{seed_idx}-{hop}")
                event_pks.append(f"TX-{seed_idx}-{hop}")
        chains = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
            max_hops=10,
        )
        ids = [c.chain_id for c in chains]
        assert len(ids) == len(set(ids)), (
            f"chain_ids are not unique: {len(ids) - len(set(ids))} duplicates in {len(ids)} chains"
        )

    def test_cycle_detection(self):
        """A→B→C→A extracted with is_cyclic=True."""
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "A"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            min_hops=2,
        )
        cyclic = [c for c in chains if c.is_cyclic]
        assert len(cyclic) >= 1
        assert cyclic[0].keys[0] == cyclic[0].keys[-1]

    def test_fan_out_multiple_chains(self):
        """A→B and A→C produce separate chains when continued."""
        chains = extract_chains(
            from_keys=["A", "A", "B", "C"],
            to_keys=["B", "C", "D", "E"],
            event_pks=["TX-1", "TX-2", "TX-3", "TX-4"],
            min_hops=2,
        )
        # A→B→D and A→C→E
        assert len(chains) >= 2

    def test_min_hops_filter(self):
        """Single-hop chains filtered out with min_hops=2."""
        chains = extract_chains(
            from_keys=["A"],
            to_keys=["B"],
            event_pks=["TX-1"],
            min_hops=2,
        )
        assert len(chains) == 0

    def test_max_hops_limit(self):
        """Chain truncated at max_hops."""
        # Long chain: A→B→C→D→E→F
        chains = extract_chains(
            from_keys=["A", "B", "C", "D", "E"],
            to_keys=["B", "C", "D", "E", "F"],
            event_pks=[f"TX-{i}" for i in range(5)],
            min_hops=2,
            max_hops=3,
        )
        for c in chains:
            assert c.hop_count <= 3

    def test_time_window_breaks_chain(self):
        """Gap > time_window splits chains."""
        # A→B at t=0, B→C at t=8days — beyond 7d window
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            timestamps=[0.0, 8 * 24 * 3600.0],
            time_window_hours=168,  # 7 days
            min_hops=2,
        )
        assert len(chains) == 0  # gap too large

    def test_time_window_allows_chain(self):
        """Within time_window → chain extracted."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            timestamps=[0.0, 3 * 24 * 3600.0],  # 3 days gap
            time_window_hours=168,
            min_hops=2,
        )
        assert len(chains) == 1

    def test_amount_decay(self):
        """amount_decay = last_amount / first_amount."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            amounts=[1000.0, 950.0],
            min_hops=2,
        )
        assert len(chains) == 1
        assert chains[0].amount_decay == pytest.approx(0.95)

    def test_categories_tracked(self):
        """Distinct categories in chain are tracked."""
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            categories=["USD", "EUR", "USD"],
            min_hops=2,
        )
        assert len(chains) >= 1
        # Should have USD and EUR
        assert set(chains[0].categories) == {"USD", "EUR"} or len(chains[0].categories) >= 2

    def test_sample_size(self):
        """sample_size limits starting nodes."""
        # 100 disconnected chains A0→B0, A1→B1, ...
        from_keys = [f"A{i}" for i in range(100)] + [f"B{i}" for i in range(100)]
        to_keys = [f"B{i}" for i in range(100)] + [f"C{i}" for i in range(100)]
        event_pks = [f"TX-{i}" for i in range(200)]

        chains_all = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
        )
        chains_sampled = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
            sample_size=10,
        )
        assert len(chains_sampled) <= len(chains_all)
        assert len(chains_sampled) <= 10

    def test_dedup_chains(self):
        """Same event set doesn't produce duplicate chains."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
        )
        event_sets = [tuple(sorted(c.event_keys)) for c in chains]
        assert len(event_sets) == len(set(event_sets))

    def test_empty_input(self):
        """Empty event data returns empty chains."""
        chains = extract_chains(
            from_keys=[],
            to_keys=[],
            event_pks=[],
            min_hops=2,
        )
        assert chains == []

    def test_to_dict(self):
        """Chain.to_dict() returns expected structure."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            categories=["USD", "EUR"],
            amounts=[100.0, 95.0],
            min_hops=2,
        )
        d = chains[0].to_dict()
        assert "chain_id" in d
        assert "hop_count" in d
        assert "is_cyclic" in d
        assert "n_distinct_categories" in d
        assert "amount_decay" in d

    def test_time_span_computed(self):
        """time_span_hours is computed from first/last timestamps."""
        # A→B at t=0h, B→C at t=24h, C→D at t=72h
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            timestamps=[0.0, 24 * 3600.0, 72 * 3600.0],
            time_window_hours=168,
            min_hops=2,
        )
        assert len(chains) >= 1
        # The longest chain A→B→C→D spans 72 hours
        longest = max(chains, key=lambda c: c.hop_count)
        assert longest.time_span_hours == pytest.approx(72.0)

    def test_time_span_zero_without_timestamps(self):
        """time_span_hours is 0 when no timestamps provided."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
        )
        assert len(chains) == 1
        assert chains[0].time_span_hours == 0.0

    def test_enriched_dict_fields(self):
        """to_dict() includes amount_std, amount_cv, amount_max, amount_min, n_unique_keys."""
        chains = extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            amounts=[100.0, 200.0, 150.0],
            min_hops=2,
        )
        assert len(chains) >= 1
        d = chains[0].to_dict()
        # All enriched fields must be present
        assert "amount_std" in d
        assert "amount_cv" in d
        assert "amount_max" in d
        assert "amount_min" in d
        assert "n_unique_keys" in d
        # Verify correctness for the longest chain
        longest = max(chains, key=lambda c: c.hop_count)
        ld = longest.to_dict()
        assert ld["amount_max"] == 200.0
        assert ld["amount_min"] == 100.0
        assert ld["amount_std"] > 0.0
        assert ld["amount_cv"] > 0.0
        assert ld["n_unique_keys"] == len(set(longest.keys))

    def test_enriched_dict_empty_amounts(self):
        """Enriched fields default to 0 when amounts is empty."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
        )
        assert len(chains) == 1
        d = chains[0].to_dict()
        assert d["amount_std"] == 0.0
        assert d["amount_cv"] == 0.0
        assert d["amount_max"] == 0.0
        assert d["amount_min"] == 0.0

    def test_seed_nodes_restricts_starts(self):
        """seed_nodes restricts BFS to start only from specified nodes."""
        # Graph: A→B→C, D→E→F
        chains = extract_chains(
            from_keys=["A", "B", "D", "E"],
            to_keys=["B", "C", "E", "F"],
            event_pks=["TX-1", "TX-2", "TX-3", "TX-4"],
            min_hops=2,
            seed_nodes=["A"],
        )
        # Only chains starting from A should be present
        assert len(chains) >= 1
        for c in chains:
            assert c.keys[0] == "A"

    def test_seed_nodes_with_sample_size(self):
        """seed_nodes + sample_size composes correctly."""
        # 50 disconnected 2-hop chains: Ai→Bi→Ci
        from_keys = [f"A{i}" for i in range(50)] + [f"B{i}" for i in range(50)]
        to_keys = [f"B{i}" for i in range(50)] + [f"C{i}" for i in range(50)]
        event_pks = [f"TX-{i}" for i in range(100)]
        seeds = [f"A{i}" for i in range(20)]  # 20 seeds

        chains = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
            seed_nodes=seeds,
            sample_size=5,
        )
        # sample_size=5 should further limit the 20 seeds
        assert len(chains) <= 5
        # All chains must start from one of the seed nodes
        for c in chains:
            assert c.keys[0] in seeds

    def test_seed_nodes_nonexistent(self):
        """seed_nodes with keys not in adjacency returns empty."""
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
            seed_nodes=["Z"],
        )
        assert len(chains) == 0

    def test_bidirectional_finds_reverse(self):
        """With bidirectional=True, BFS can traverse edges in reverse."""
        # Directed edges: A→B, B→C
        # Without bidirectional, seed_nodes=["C"] finds nothing (C has no outgoing edges)
        # With bidirectional, C→B and B→A become available, so C can reach A
        chains_no_bidir = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
            seed_nodes=["C"],
            bidirectional=False,
        )
        assert len(chains_no_bidir) == 0

        chains_bidir = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
            seed_nodes=["C"],
            bidirectional=True,
        )
        assert len(chains_bidir) >= 1
        assert chains_bidir[0].keys[0] == "C"

    def test_bidirectional_default_false(self):
        """Default bidirectional=False preserves directed-only behavior."""
        # Directed edge: A→B only. seed_nodes=["B"] should find nothing
        # because B has no outgoing edges in directed mode.
        chains = extract_chains(
            from_keys=["A", "B"],
            to_keys=["B", "C"],
            event_pks=["TX-1", "TX-2"],
            min_hops=2,
            seed_nodes=["C"],
        )
        assert len(chains) == 0


class TestParallelBFS:
    def test_no_duplicate_event_sets_on_dense_graph(self):
        """Seeds that reach the same chain do not emit it twice."""
        # Dense graph: many seeds find the same chains
        n = 500
        from_keys = [f"N-{i % 50:03d}" for i in range(n)]
        to_keys = [f"N-{(i + 1) % 50:03d}" for i in range(n)]
        event_pks = [f"TX-{i:06d}" for i in range(n)]

        chains = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
            max_chains=5000,
        )
        # Verify no duplicate event_keys sets
        sigs = [frozenset(c.event_keys) for c in chains]
        assert len(sigs) == len(set(sigs))


def test_chain_to_dict_lists_distinct_categories_in_sorted_order() -> None:
    from hypertopos.engine.chains import Chain

    chain = Chain(
        chain_id="CHAIN-000000",
        keys=["A", "B", "C", "D"],
        event_keys=["e1", "e2", "e3"],
        hop_count=3,
        is_cyclic=False,
        time_span_hours=1.0,
        categories=["PLN", "EUR", "PLN"],
        amounts=[3.0, 2.0, 1.0],
        amount_decay=1 / 3,
    )
    assert chain.to_dict()["categories"] == ["EUR", "PLN"]


def test_extract_chains_restores_the_garbage_collector_state() -> None:
    import gc

    from hypertopos.engine.chains import extract_chains

    args = (["A", "B"], ["B", "A"], ["e1", "e2"], [0.0, 10.0])
    try:
        gc.enable()
        extract_chains(*args)
        assert gc.isenabled()
        gc.disable()
        extract_chains(*args)
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_dedup_cap_overflow_is_reported(monkeypatch):
    """When the dedup set fills up, later duplicate chains are no longer
    removed; that must be visible, not silent."""
    import hypertopos.engine.chains as chains_mod

    monkeypatch.setattr(chains_mod, "_DEDUP_CAP", 1)
    with pytest.warns(UserWarning, match="chain dedup stopped"):
        extract_chains(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            event_pks=["TX-1", "TX-2", "TX-3"],
            min_hops=2,
        )
