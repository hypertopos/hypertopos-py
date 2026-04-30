"""Tests for engine/structuring.py — extracted structuring enumerator."""
from __future__ import annotations

import pyarrow as pa

from hypertopos.engine.structuring import (
    enumerate_structuring_event_keys,
    enumerate_structuring_for_seed,
)


def _edges(rows: list[tuple[str, str, str, float, float]]) -> pa.Table:
    return pa.table({
        "from_key":  [r[0] for r in rows],
        "to_key":    [r[1] for r in rows],
        "event_key": [r[2] for r in rows],
        "timestamp": [r[3] for r in rows],
        "amount":    [r[4] for r in rows],
    })


def test_enumerate_structuring_event_keys_simple_chain():
    edges = _edges([
        ("A", "B", "ek1", 0.0,    20000.0),
        ("B", "C", "ek2", 10.0,   5000.0),
        ("C", "D", "ek3", 20.0,   5000.0),
    ])
    keys = enumerate_structuring_event_keys(
        edges, time_window_sec=3600.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    assert keys == {"ek1", "ek2", "ek3"}


def test_enumerate_structuring_event_keys_no_match():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  5000.0),
        ("B", "C", "ek2", 10.0, 1000.0),
        ("C", "D", "ek3", 20.0, 1000.0),
    ])
    keys = enumerate_structuring_event_keys(
        edges, time_window_sec=3600.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    assert keys == set()


def test_enumerate_structuring_event_keys_multiple_motifs_share_event():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  20000.0),
        ("B", "C", "ek2", 10.0, 5000.0),
        ("C", "D", "ek3", 20.0, 5000.0),
        ("C", "E", "ek4", 25.0, 5000.0),
    ])
    keys = enumerate_structuring_event_keys(
        edges, time_window_sec=3600.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    assert keys == {"ek1", "ek2", "ek3", "ek4"}


def test_enumerate_structuring_event_keys_window_breach():
    edges = _edges([
        ("A", "B", "ek1", 0.0,    20000.0),
        ("B", "C", "ek2", 1800.0, 5000.0),
        ("C", "D", "ek3", 7300.0, 5000.0),
    ])
    keys = enumerate_structuring_event_keys(
        edges, time_window_sec=3600.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    assert keys == set()


def test_enumerate_structuring_event_keys_self_visit_rejected():
    edges = _edges([
        ("A", "B", "ek1", 0.0,  20000.0),
        ("B", "A", "ek2", 10.0, 5000.0),
        ("A", "D", "ek3", 20.0, 5000.0),
    ])
    keys = enumerate_structuring_event_keys(
        edges, time_window_sec=3600.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    assert keys == set()


def test_enumerate_structuring_event_keys_negative_amount_skipped():
    edges = _edges([
        ("A", "B", "ek1", 0.0,    20000.0),
        ("B", "C", "ek2", 10.0,   -5000.0),
        ("C", "D", "ek3", 20.0,   5000.0),
    ])
    keys = enumerate_structuring_event_keys(
        edges, time_window_sec=3600.0, amt1_min=10000.0, amt2_max=10000.0,
    )
    assert keys == set()


def test_enumerate_structuring_for_seed_returns_motif_dicts():
    edges = _edges([
        ("A", "B", "ek1", 0.0,    20000.0),
        ("B", "C", "ek2", 10.0,   5000.0),
        ("C", "D", "ek3", 20.0,   5000.0),
    ])
    motifs = enumerate_structuring_for_seed(
        seed="A", edges=edges,
        time_window_sec=3600.0, amt1_min=10000.0, amt2_max=10000.0,
        max_instances=10,
    )
    assert len(motifs) == 1
    assert motifs[0]["edges"] == [
        ("A", "B", "ek1"),
        ("B", "C", "ek2"),
        ("C", "D", "ek3"),
    ]
