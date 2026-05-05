"""Parity + correctness tests for distinct_neighbors_out/in (OPI-116 hotfix)."""
from __future__ import annotations

from hypertopos.engine.adjacency import AdjacencyIndex


def test_distinct_neighbors_out_basic():
    adj = AdjacencyIndex.from_edge_lists(
        ["A", "A", "B", "A"], ["B", "C", "C", "B"],  # A→B (×2), A→C, B→C
        [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0],
        ["e1", "e2", "e3", "e4"],
    )
    assert adj.distinct_neighbors_out("A") == 2  # B and C
    assert adj.distinct_neighbors_out("B") == 1  # C
    assert adj.distinct_neighbors_out("C") == 0  # nothing
    assert adj.distinct_neighbors_out("missing") == 0


def test_distinct_neighbors_in_basic():
    adj = AdjacencyIndex.from_edge_lists(
        ["A", "A", "B", "A"], ["B", "C", "C", "B"],
        [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0],
        ["e1", "e2", "e3", "e4"],
    )
    assert adj.distinct_neighbors_in("A") == 0
    assert adj.distinct_neighbors_in("B") == 1  # A
    assert adj.distinct_neighbors_in("C") == 2  # A and B
    assert adj.distinct_neighbors_in("missing") == 0


def test_distinct_neighbors_excludes_self_loops():
    adj = AdjacencyIndex.from_edge_lists(
        ["A", "A", "B", "B"], ["A", "B", "B", "A"],  # A→A, A→B, B→B, B→A
        [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0],
        ["e1", "e2", "e3", "e4"],
    )
    assert adj.distinct_neighbors_out("A") == 1  # B only (self-loop excluded)
    assert adj.distinct_neighbors_out("B") == 1  # A only
    assert adj.distinct_neighbors_in("A") == 1  # B only
    assert adj.distinct_neighbors_in("B") == 1  # A only


def test_distinct_neighbors_empty():
    adj = AdjacencyIndex.from_edge_lists([], [], [], [], [])
    assert adj.distinct_neighbors_out("anything") == 0
    assert adj.distinct_neighbors_in("anything") == 0


def test_parity_with_legacy_set_comprehension():
    """For every key, distinct_neighbors_out(k) must equal
    len({t for (t, *_r) in adj._out[k] if t != k}).
    """
    import random
    rng = random.Random(42)
    nodes = [f"N{i}" for i in range(20)]
    n = 200
    fk = [rng.choice(nodes) for _ in range(n)]
    tk = [rng.choice(nodes) for _ in range(n)]
    ts = [float(rng.randint(0, 50)) for _ in range(n)]
    am = [float(rng.randint(1, 1000)) for _ in range(n)]
    ek = [f"e{i:05d}" for i in range(n)]
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, am, ek)
    for k in adj.all_nodes():
        legacy_out = len({t for (t, *_r) in adj._out.get(k, []) if t != k})
        legacy_in = len({f for (f, *_r) in adj._in.get(k, []) if f != k})
        got_out = adj.distinct_neighbors_out(k)
        got_in = adj.distinct_neighbors_in(k)
        assert got_out == legacy_out, f"out mismatch on {k}: {got_out} vs {legacy_out}"
        assert got_in == legacy_in, f"in mismatch on {k}: {got_in} vs {legacy_in}"


def test_max_amount_out_excl_self_basic():
    # A->B(100), A->C(200), B->C(50), A->A(999 self-loop), C->C(999 self-loop)
    adj = AdjacencyIndex.from_edge_lists(
        ["A", "A", "B", "A", "C"], ["B", "C", "C", "A", "C"],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [100.0, 200.0, 50.0, 999.0, 999.0],
        ["e1", "e2", "e3", "e4", "e5"],
    )
    # max of (B:100, C:200), self-loop A->A excluded
    assert adj.max_amount_out_excl_self("A") == 200.0
    assert adj.max_amount_out_excl_self("B") == 50.0    # only B->C
    assert adj.max_amount_out_excl_self("C") == 0.0     # only self-loop C->C, excluded
    assert adj.max_amount_out_excl_self("missing") == 0.0


def test_max_amount_out_excl_self_handles_null_amounts():
    """Null amounts must be ignored (matches legacy `amt is not None` predicate)."""
    import pyarrow as pa
    # Build a table with null amount on one edge
    tbl = pa.table({
        "from_key": ["A", "A", "B"],
        "to_key": ["B", "C", "C"],
        "timestamp": [1.0, 2.0, 3.0],
        "amount": pa.array([None, 200.0, 50.0], type=pa.float64()),
        "event_key": ["e1", "e2", "e3"],
    })
    adj = AdjacencyIndex._from_table(tbl)
    assert adj.max_amount_out_excl_self("A") == 200.0  # null ignored
    assert adj.max_amount_out_excl_self("B") == 50.0


def test_max_amount_out_excl_self_parity_random():
    """Parity vs legacy `any(amt is not None AND amt >= threshold AND t != s)` on random fixture."""
    import random
    rng = random.Random(43)
    nodes = [f"N{i}" for i in range(15)]
    n = 150
    fk = [rng.choice(nodes) for _ in range(n)]
    tk = [rng.choice(nodes) for _ in range(n)]
    ts = [float(rng.randint(0, 50)) for _ in range(n)]
    # Mix of high and low amounts; some self-loops
    am = [float(rng.randint(1, 20000)) for _ in range(n)]
    ek = [f"e{i:05d}" for i in range(n)]
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, am, ek)

    threshold = 10000.0
    for k in adj.all_nodes():
        # Legacy any-predicate
        legacy = any(
            amt is not None and amt >= threshold and t != k
            for (t, _ts, amt, _ek) in adj._out.get(k, [])
        )
        # New max-predicate
        new = adj.max_amount_out_excl_self(k) >= threshold
        max_val = adj.max_amount_out_excl_self(k)
        assert legacy == new, (
            f"parity fail on {k}: legacy={legacy}, new={new}, max={max_val}"
        )


def test_pair_counts_includes_self_loops():
    """pair_counts() must include self-loops (legacy semantic)."""
    adj = AdjacencyIndex.from_edge_lists(
        ["A", "A", "B", "B"], ["A", "B", "B", "A"],  # A->A, A->B, B->B, B->A
        [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0],
        ["e1", "e2", "e3", "e4"],
    )
    pc = adj.pair_counts()
    assert pc[("A", "A")] == 1  # self-loop counted
    assert pc[("A", "B")] == 1
    assert pc[("B", "B")] == 1  # self-loop counted
    assert pc[("B", "A")] == 1


def test_pair_counts_aggregates_duplicate_edges():
    """Multiple edges with same (from, to) but different timestamps -> count = N."""
    adj = AdjacencyIndex.from_edge_lists(
        ["A", "A", "A"], ["B", "B", "B"], [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0], ["e1", "e2", "e3"],
    )
    assert adj.pair_counts()[("A", "B")] == 3


def test_pair_counts_eagerly_populated_from_table():
    """After _from_table, _pair_counts must already be populated (not None).

    No lazy materialize on first call.
    """
    adj = AdjacencyIndex.from_edge_lists(
        ["A"], ["B"], [1.0], [10.0], ["e1"],
    )
    assert adj._pair_counts is not None
    assert adj._pair_counts == {("A", "B"): 1}


def test_pair_counts_parity_random_fixture():
    """200-edge random fixture: pair_counts must match a reference Python loop on the same input."""
    import random
    rng = random.Random(44)
    nodes = [f"N{i}" for i in range(15)]
    n = 200
    fk = [rng.choice(nodes) for _ in range(n)]
    tk = [rng.choice(nodes) for _ in range(n)]
    ts = [float(rng.randint(0, 50)) for _ in range(n)]
    am = [float(rng.randint(1, 1000)) for _ in range(n)]
    ek = [f"e{i:05d}" for i in range(n)]
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, am, ek)

    # Reference: count edges per (from, to) directly
    ref: dict[tuple[str, str], int] = {}
    for i in range(n):
        ref[(fk[i], tk[i])] = ref.get((fk[i], tk[i]), 0) + 1

    assert adj.pair_counts() == ref


def test_pair_counts_empty():
    """Empty AdjacencyIndex.pair_counts() returns {}."""
    adj = AdjacencyIndex.from_edge_lists([], [], [], [], [])
    assert adj.pair_counts() == {}
