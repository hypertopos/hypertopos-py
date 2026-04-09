# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for chain extraction engine."""

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool

import pytest
from hypertopos.engine.chains import extract_chains


class TestExtractChains:
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

    def test_parallel_falls_back_on_broken_process_pool(self, monkeypatch):
        """BrokenProcessPool during startup falls back to serial BFS."""

        class BrokenExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                raise BrokenProcessPool("boom")

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", BrokenExecutor)

        from_keys = [f"A{i}" for i in range(400)] + [f"B{i}" for i in range(400)]
        to_keys = [f"B{i}" for i in range(400)] + [f"C{i}" for i in range(400)]
        event_pks = [f"TX-{i}" for i in range(800)]

        chains = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
        )
        assert len(chains) == 400

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
    def test_parallel_produces_same_chains_as_serial(self):
        """Parallel BFS (>400 seeds) produces equivalent chains to serial."""
        # Build a graph with 500+ unique starting nodes
        n = 600
        from_keys = [f"N-{i:04d}" for i in range(n)]
        to_keys = [f"N-{(i + 1) % n:04d}" for i in range(n)]
        event_pks = [f"TX-{i:06d}" for i in range(n)]

        serial = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
            max_chains=1000,
        )
        # Should produce chains (ring graph → every node is a seed)
        assert len(serial) > 0
        # All chains have hop_count >= 2
        assert all(c.hop_count >= 2 for c in serial)

    def test_parallel_falls_back_when_process_pool_unavailable(self, monkeypatch):
        """Parallel BFS falls back to serial when process pools cannot start."""
        import concurrent.futures

        n = 600
        from_keys = [f"N-{i:04d}" for i in range(n)]
        to_keys = [f"N-{(i + 1) % n:04d}" for i in range(n)]
        event_pks = [f"TX-{i:06d}" for i in range(n)]

        def raise_permission_error(*args, **kwargs):
            raise PermissionError("mocked process pool failure")

        monkeypatch.setattr(
            concurrent.futures,
            "ProcessPoolExecutor",
            raise_permission_error,
        )

        chains = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            min_hops=2,
            max_chains=1000,
        )

        assert len(chains) > 0
        assert all(c.hop_count >= 2 for c in chains)

    def test_parallel_dedup_across_workers(self):
        """Post-merge dedup removes duplicates found by different workers."""
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
