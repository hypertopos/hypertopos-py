"""Tests for AdjacencyIndex.neighbors_out_window / neighbors_in_window."""
from __future__ import annotations

import random

import pytest

from hypertopos.engine.adjacency import AdjacencyIndex


def _synth_5() -> AdjacencyIndex:
    """5-edge fixture: A->B@10, A->C@11, B->C@12, C->D@13, A->D@14."""
    return AdjacencyIndex.from_edge_lists(
        ["A", "A", "B", "C", "A"], ["B", "C", "C", "D", "D"],
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [100.0, 200.0, 50.0, 300.0, 400.0],
        ["e1", "e2", "e3", "e4", "e5"],
    )


def test_neighbors_out_window_no_filter_default_columns():
    adj = _synth_5()
    result = adj.neighbors_out_window("A")
    assert set(result.keys()) == {"to_key", "timestamp"}
    assert result["to_key"] == ["B", "C", "D"]  # sorted by timestamp
    assert result["timestamp"] == [10.0, 11.0, 14.0]


def test_neighbors_out_window_filter_applied():
    adj = _synth_5()
    result = adj.neighbors_out_window("A", ts_min=11.0)
    assert result["to_key"] == ["C", "D"]
    assert result["timestamp"] == [11.0, 14.0]


def test_neighbors_out_window_custom_columns():
    adj = _synth_5()
    result = adj.neighbors_out_window("A", columns=("to_key", "amount"))
    assert set(result.keys()) == {"to_key", "amount"}
    assert result["to_key"] == ["B", "C", "D"]
    assert result["amount"] == [100.0, 200.0, 400.0]


def test_neighbors_out_window_missing_key():
    adj = _synth_5()
    result = adj.neighbors_out_window("MISSING")
    assert result == {"to_key": [], "timestamp": []}


def test_neighbors_in_window_basic():
    adj = _synth_5()
    result = adj.neighbors_in_window("C")
    assert result["from_key"] == ["A", "B"]
    assert result["timestamp"] == [11.0, 12.0]


def test_neighbors_in_window_filter():
    adj = _synth_5()
    result = adj.neighbors_in_window("D", ts_min=14.0)
    assert result["from_key"] == ["A"]
    assert result["timestamp"] == [14.0]


def test_window_parity_with_neighbors_out():
    """For window-filtered subset, neighbors_out_window matches neighbors_out + manual filter."""
    rng = random.Random(46)
    nodes = [f"N{i}" for i in range(15)]
    n = 200
    fk = [rng.choice(nodes) for _ in range(n)]
    tk = [rng.choice(nodes) for _ in range(n)]
    ts = [float(rng.randint(0, 100)) for _ in range(n)]
    am = [float(rng.randint(1, 1000)) for _ in range(n)]
    ek = [f"e{i:05d}" for i in range(n)]
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, am, ek)

    threshold = 50.0
    for k in adj.all_nodes():
        legacy = [(d, t) for (d, t, _a, _e) in adj.neighbors_out(k) if t >= threshold]
        windowed = adj.neighbors_out_window(k, ts_min=threshold)
        zipped = list(zip(windowed["to_key"], windowed["timestamp"], strict=True))
        assert zipped == legacy, f"window parity fail on {k}: {zipped} vs {legacy}"


def test_window_parity_with_neighbors_in():
    """Symmetric parity for in-direction."""
    rng = random.Random(47)
    nodes = [f"N{i}" for i in range(15)]
    n = 200
    fk = [rng.choice(nodes) for _ in range(n)]
    tk = [rng.choice(nodes) for _ in range(n)]
    ts = [float(rng.randint(0, 100)) for _ in range(n)]
    am = [float(rng.randint(1, 1000)) for _ in range(n)]
    ek = [f"e{i:05d}" for i in range(n)]
    adj = AdjacencyIndex.from_edge_lists(fk, tk, ts, am, ek)

    threshold = 30.0
    for k in adj.all_nodes():
        legacy = [(f, t) for (f, t, _a, _e) in adj.neighbors_in(k) if t >= threshold]
        windowed = adj.neighbors_in_window(k, ts_min=threshold)
        zipped = list(zip(windowed["from_key"], windowed["timestamp"], strict=True))
        assert zipped == legacy


def test_neighbors_out_window_rejects_invalid_column():
    """Misuse case: out_window doesn't have from_key column (out_grouped is keyed by from_key)."""
    adj = _synth_5()
    with pytest.raises(ValueError, match="unknown columns"):
        adj.neighbors_out_window("A", columns=("from_key",))
    with pytest.raises(ValueError, match="unknown columns"):
        adj.neighbors_out_window("A", columns=("to_key", "bogus_column"))


def test_neighbors_in_window_rejects_invalid_column():
    """Misuse case: in_window doesn't have to_key column (in_grouped is keyed by to_key)."""
    adj = _synth_5()
    with pytest.raises(ValueError, match="unknown columns"):
        adj.neighbors_in_window("C", columns=("to_key",))
    with pytest.raises(ValueError, match="unknown columns"):
        adj.neighbors_in_window("C", columns=("from_key", "totally_bogus"))
