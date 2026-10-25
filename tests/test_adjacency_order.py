"""Per-key neighbour lists are in timestamp order, ties in edge-table order,
on every build — whatever order the grouping hands rows back in."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from hypertopos.engine.adjacency import AdjacencyIndex


def _edge_table(n: int = 6000, n_keys: int = 40, n_chunks: int = 6) -> pa.Table:
    rng = np.random.default_rng(11)
    keys = np.array([f"K{i:02d}" for i in range(n_keys)])
    m = n // n_chunks
    batches = [
        pa.record_batch(
            {
                "from_key": pa.array(keys[rng.integers(0, n_keys, m)]),
                "to_key": pa.array(keys[rng.integers(0, n_keys, m)]),
                # Coarse timestamps: many ties.
                "timestamp": pa.array(rng.integers(0, 50, m).astype(np.float64)),
                "amount": pa.array(rng.random(m)),
                "event_key": pa.array([f"E{c}-{j}" for j in range(m)]),
            }
        )
        for c in range(n_chunks)
    ]
    return pa.Table.from_batches(batches)


def _expected(table: pa.Table, key_col: str) -> dict[str, list[str]]:
    """event_key order per key: timestamp ascending, ties in table row order."""
    rows = table.to_pylist()
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["timestamp"], i))
    out: dict[str, list[str]] = {}
    for i in order:
        out.setdefault(rows[i][key_col], []).append(rows[i]["event_key"])
    return out


def _actual(adj: AdjacencyIndex) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    out = {k: [e[3] for e in adj.neighbors_out(k)] for k in adj._out_key_to_idx}
    inn = {k: [e[3] for e in adj.neighbors_in(k)] for k in adj._in_key_to_idx}
    return out, inn


@pytest.fixture
def scrambling_group_by(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grouping hands rows back in an arbitrary order, as threaded grouping can."""
    import pyarrow.acero as acero

    original = acero._group_by
    rng = np.random.default_rng(5)

    def group_by(table: pa.Table, *args, **kwargs):  # type: ignore[no-untyped-def]
        shuffled = table.take(pa.array(rng.permutation(table.num_rows)))
        return original(shuffled, *args, **kwargs)

    monkeypatch.setattr(acero, "_group_by", group_by)


def test_lists_ordered_even_when_grouping_scrambles_rows(scrambling_group_by: None) -> None:
    table = _edge_table()
    out, inn = _actual(AdjacencyIndex.from_table(table))
    assert out == _expected(table, "from_key")
    assert inn == _expected(table, "to_key")


def test_repeated_builds_identical_and_time_ordered() -> None:
    table = _edge_table()
    expected = (_expected(table, "from_key"), _expected(table, "to_key"))
    for _ in range(5):
        adj = AdjacencyIndex.from_table(table)
        assert _actual(adj) == expected
        for grouped in (adj._out_grouped, adj._in_grouped):
            ts = grouped["timestamp_list"].combine_chunks()
            vals = pc.list_flatten(ts).to_numpy()
            parents = pc.list_parent_indices(ts).to_numpy()
            assert not ((np.diff(vals) < 0) & (np.diff(parents) == 0)).any()
