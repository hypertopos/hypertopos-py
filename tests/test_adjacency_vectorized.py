"""Parity + correctness tests for the vectorized AdjacencyIndex.

Verifies the new pyarrow group_by + lazy materialization implementation produces
identical observable output to the legacy from_edge_lists Python-loop path.
"""
from __future__ import annotations

from hypertopos.engine.adjacency import AdjacencyIndex


def _synth_5() -> tuple[list, list, list, list, list]:
    """5-edge synthetic fixture used in multiple tests.

    A→B @ 10:00 ($100, e1), A→C @ 11:00 ($200, e2), B→C @ 12:00 ($50, e3),
    C→D @ 13:00 ($300, e4), A→D @ 14:00 ($400, e5).
    """
    return (
        ["A", "A", "B", "C", "A"],
        ["B", "C", "C", "D", "D"],
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [100.0, 200.0, 50.0, 300.0, 400.0],
        ["e1", "e2", "e3", "e4", "e5"],
    )


def test_empty_edge_lists_produces_empty_index():
    adj = AdjacencyIndex.from_edge_lists([], [], [], [], [])
    assert adj.edge_count() == 0
    assert adj.node_count() == 0
    assert adj.neighbors_out("anything") == []
    assert adj.neighbors_in("anything") == []
    assert adj.degree_out("anything") == 0
    assert adj.degree_in("anything") == 0


def test_from_edge_lists_basic_neighbors_out():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    out_a = adj.neighbors_out("A")
    assert [e[0] for e in out_a] == ["B", "C", "D"]
    assert [e[1] for e in out_a] == [10.0, 11.0, 14.0]
    assert [e[2] for e in out_a] == [100.0, 200.0, 400.0]
    assert [e[3] for e in out_a] == ["e1", "e2", "e5"]


def test_from_edge_lists_basic_neighbors_in():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    in_c = adj.neighbors_in("C")
    assert [e[0] for e in in_c] == ["A", "B"]  # A→C @ 11, B→C @ 12
    assert [e[1] for e in in_c] == [11.0, 12.0]


def test_temporal_filter_neighbors_out():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    # A's edges: 10, 11, 14. Filter [11, 13] → only 11
    filtered = adj.neighbors_out("A", ts_from=11.0, ts_to=13.0)
    assert [e[1] for e in filtered] == [11.0]


def test_temporal_filter_neighbors_in():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    in_c = adj.neighbors_in("C", ts_from=12.0)
    assert [e[1] for e in in_c] == [12.0]


def test_neighbors_all():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    # C's outgoing: C→D @ 13. C's incoming: A→C @ 11, B→C @ 12.
    all_c = adj.neighbors_all("C")
    assert len(all_c) == 3
    timestamps = sorted(e[1] for e in all_c)
    assert timestamps == [11.0, 12.0, 13.0]


def test_degree_out_in():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    assert adj.degree_out("A") == 3  # A→B, A→C, A→D
    assert adj.degree_out("B") == 1
    assert adj.degree_in("C") == 2  # A→C, B→C
    assert adj.degree_in("D") == 2
    assert adj.degree_in("A") == 0
    assert adj.degree_out("D") == 0


def test_all_nodes():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    assert adj.all_nodes() == {"A", "B", "C", "D"}


def test_edge_count_node_count():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    assert adj.edge_count() == 5
    assert adj.node_count() == 4


def test_all_edges_yields_every_edge():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    edges = list(adj.all_edges())
    assert len(edges) == 5
    # Verify all 5 input edges appear as (src, tgt, ts, amt, ek) tuples
    expected = {
        ("A", "B", 10.0, 100.0, "e1"),
        ("A", "C", 11.0, 200.0, "e2"),
        ("B", "C", 12.0, 50.0, "e3"),
        ("C", "D", 13.0, 300.0, "e4"),
        ("A", "D", 14.0, 400.0, "e5"),
    }
    assert set(edges) == expected


def test_pair_counts_simple():
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    pc = adj.pair_counts()
    assert pc[("A", "B")] == 1
    assert pc[("A", "C")] == 1
    assert pc[("A", "D")] == 1
    assert pc[("B", "C")] == 1
    assert pc[("C", "D")] == 1


def test_pair_counts_with_duplicate_pair():
    """Two edges with same (from, to) but different timestamps → pair count = 2."""
    adj = AdjacencyIndex.from_edge_lists(
        ["A", "A", "B"], ["B", "B", "C"], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0],
        ["e1", "e2", "e3"],
    )
    pc = adj.pair_counts()
    assert pc[("A", "B")] == 2
    assert pc[("B", "C")] == 1


def test_caching_pair_counts():
    """pair_counts() second call returns cached result."""
    fk, tk, ts, amt, ek = _synth_5()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)
    pc1 = adj.pair_counts()
    pc2 = adj.pair_counts()
    assert pc1 is pc2  # same object — cache hit


# ── Parity test against a synthetic 200-edge fixture ──────────────────────
def _build_random_fixture(n: int = 200, seed: int = 42):
    """Build a deterministic random edge fixture with overlapping pairs and ties."""
    import random
    rng = random.Random(seed)
    nodes = [f"N{i}" for i in range(20)]
    fk = [rng.choice(nodes) for _ in range(n)]
    tk = [rng.choice(nodes) for _ in range(n)]
    # Some intentional timestamp ties
    ts = [float(rng.randint(0, 50)) for _ in range(n)]
    amt = [float(rng.randint(1, 1000)) for _ in range(n)]
    ek = [f"e{i:05d}" for i in range(n)]
    return fk, tk, ts, amt, ek


def test_parity_random_fixture():
    """200-edge random fixture: every neighbors_out / neighbors_in / pair_counts
    must match what an independent reference Python loop produces."""
    fk, tk, ts, amt, ek = _build_random_fixture()
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, amt, ek)

    # Build reference dict the slow way
    ref_out: dict[str, list] = {}
    ref_in: dict[str, list] = {}
    ref_pairs: dict[tuple, int] = {}
    for i in range(len(fk)):
        ref_out.setdefault(fk[i], []).append((tk[i], ts[i], amt[i], ek[i]))
        ref_in.setdefault(tk[i], []).append((fk[i], ts[i], amt[i], ek[i]))
        ref_pairs[(fk[i], tk[i])] = ref_pairs.get((fk[i], tk[i]), 0) + 1
    # Sort reference per-key lists by timestamp (stable — ties keep input order)
    for v in ref_out.values():
        v.sort(key=lambda e: e[1])
    for v in ref_in.values():
        v.sort(key=lambda e: e[1])

    # Compare neighbors_out for every key
    for k in ref_out:
        actual = adj.neighbors_out(k)
        # Compare timestamp-tag sequences (ties can differ in tuple order
        # depending on stable-sort detail; if the test fails on ties, use
        # multiset comparison).
        assert sorted(actual) == sorted(ref_out[k]), (
            f"out mismatch for {k}: {actual} vs {ref_out[k]}"
        )
    # Compare neighbors_in for every key
    for k in ref_in:
        actual = adj.neighbors_in(k)
        assert sorted(actual) == sorted(ref_in[k]), f"in mismatch for {k}"
    # Compare pair_counts
    assert adj.pair_counts() == ref_pairs
    # Counts
    assert adj.edge_count() == len(fk)
    assert adj.node_count() == len(set(fk) | set(tk))
