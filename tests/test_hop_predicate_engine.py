"""Engine-level tests for declarative motif enumeration via HopPredicate."""
from __future__ import annotations

import pyarrow as pa
import pytest

from hypertopos import HopPredicate
from hypertopos.engine.adjacency import AdjacencyIndex
from hypertopos.engine.hop_predicate import enumerate_motifs_by_hops


def _maps_from_rows(
    rows: list[tuple[str, str, str, float, float]],
) -> tuple[
    dict[str, list[tuple[str, float, float, str]]],
    dict[str, list[tuple[str, float, float, str]]],
]:
    """Build (out_map, in_map) the same way AdjacencyIndex does."""
    adj = AdjacencyIndex.from_edge_lists(
        from_keys=[r[0] for r in rows],
        to_keys=[r[1] for r in rows],
        timestamps=[r[3] for r in rows],
        amounts=[r[4] for r in rows],
        event_keys=[r[2] for r in rows],
    )
    return adj._out, adj._in


def _features(rows: list[tuple[str, dict[str, float]]]) -> pa.Table:
    eks = [r[0] for r in rows]
    cols: dict[str, list[float]] = {}
    for _, props in rows:
        for k in props:
            cols.setdefault(k, [])
    for ek, props in rows:
        for k in cols:
            cols[k].append(float(props.get(k, 0.0)))
    table_dict = {"event_key": eks, **cols}
    return pa.table(table_dict)


def test_single_hop_amount_min():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0, 100.0),
        ("A", "C", "ek2", 1.0, 50.0),
        ("A", "D", "ek3", 2.0, 200.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[HopPredicate(amount_min=100.0)],
        seed_keys=["A"],
    )
    assert len(motifs) == 2
    sorted_eks = sorted(m["edges"][0] for m in motifs)
    assert sorted_eks == ["ek1", "ek3"]


def test_three_hop_structuring_equivalent():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0,    20000.0),
        ("B", "C", "ek2", 100.0,  5000.0),
        ("C", "D", "ek3", 200.0,  5000.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[
            HopPredicate(amount_min=10000.0),
            HopPredicate(amount_max=10000.0, time_delta_max_hours=1.0),
            HopPredicate(amount_max=10000.0, time_delta_max_hours=1.0),
        ],
        seed_keys=["A"],
    )
    assert len(motifs) == 1
    assert motifs[0]["nodes"] == ["A", "B", "C", "D"]
    assert motifs[0]["edges"] == ["ek1", "ek2", "ek3"]


def test_temporal_window_breach():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0,    20000.0),
        ("B", "C", "ek2", 7300.0, 5000.0),  # 2.03h > 1h window
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[
            HopPredicate(amount_min=10000.0),
            HopPredicate(amount_max=10000.0, time_delta_max_hours=1.0),
        ],
        seed_keys=["A"],
    )
    assert motifs == []


def test_self_visit_rejected():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0,  100.0),
        ("B", "A", "ek2", 1.0,  100.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[HopPredicate(), HopPredicate()],
        seed_keys=["A"],
    )
    assert motifs == []


def test_all_seeds_when_seed_keys_none():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0,  100.0),
        ("X", "Y", "ek2", 0.0,  100.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map, hops=[HopPredicate()], seed_keys=None,
    )
    sorted_seeds = sorted(m["nodes"][0] for m in motifs)
    assert sorted_seeds == ["A", "X"]


def test_max_results_cap():
    rows = [("A", f"B{i}", f"ek{i}", float(i), 100.0) for i in range(50)]
    out_map, in_map = _maps_from_rows(rows)
    motifs = enumerate_motifs_by_hops(
        out_map, in_map, hops=[HopPredicate()], seed_keys=["A"], max_results=10,
    )
    assert len(motifs) == 10


def test_direction_reverse():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0, 100.0),
        ("X", "B", "ek2", 1.0, 100.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[HopPredicate(direction="reverse")],
        seed_keys=["B"],
    )
    sorted_seeds = sorted(m["nodes"][1] for m in motifs)
    assert sorted_seeds == ["A", "X"]


def test_direction_any_includes_both():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0, 100.0),
        ("B", "C", "ek2", 1.0, 100.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[HopPredicate(direction="any")],
        seed_keys=["B"],
    )
    nxt_nodes = sorted(m["nodes"][1] for m in motifs)
    assert nxt_nodes == ["A", "C"]


def test_edge_dim_predicate_filter():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0, 100.0),
        ("A", "C", "ek2", 1.0, 100.0),
    ])
    feats = _features([
        ("ek1", {"pair_edge_count": 5.0}),
        ("ek2", {"pair_edge_count": 25.0}),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[HopPredicate(
            edge_dim_predicates={"pair_edge_count": (">=", 20.0)},
        )],
        seed_keys=["A"],
        edge_features=feats,
    )
    assert len(motifs) == 1
    assert motifs[0]["edges"] == ["ek2"]
    assert motifs[0]["dim_values_per_hop"][0]["pair_edge_count"] == 25.0


def test_edge_dim_predicate_unknown_dim_raises():
    out_map, in_map = _maps_from_rows([("A", "B", "ek1", 0.0, 100.0)])
    feats = _features([("ek1", {"pair_edge_count": 5.0})])
    with pytest.raises(ValueError, match="unknown dims"):
        enumerate_motifs_by_hops(
            out_map, in_map,
            hops=[HopPredicate(
                edge_dim_predicates={"nonexistent_dim": (">=", 1.0)},
            )],
            seed_keys=["A"],
            edge_features=feats,
        )


def test_edge_dim_predicate_no_sidecar_raises():
    out_map, in_map = _maps_from_rows([("A", "B", "ek1", 0.0, 100.0)])
    with pytest.raises(ValueError, match="no edge_features sidecar"):
        enumerate_motifs_by_hops(
            out_map, in_map,
            hops=[HopPredicate(
                edge_dim_predicates={"pair_edge_count": (">=", 1.0)},
            )],
            seed_keys=["A"],
            edge_features=None,
        )


def test_invalid_operator_raises():
    out_map, in_map = _maps_from_rows([("A", "B", "ek1", 0.0, 100.0)])
    feats = _features([("ek1", {"pair_edge_count": 5.0})])
    with pytest.raises(ValueError, match="operator must be"):
        enumerate_motifs_by_hops(
            out_map, in_map,
            hops=[HopPredicate(
                edge_dim_predicates={"pair_edge_count": ("~", 1.0)},
            )],
            seed_keys=["A"],
            edge_features=feats,
        )


def test_invalid_direction_raises():
    out_map, in_map = _maps_from_rows([("A", "B", "ek1", 0.0, 100.0)])
    # Bypass frozen dataclass to inject an invalid direction at runtime —
    # the engine's ValueError branch is the safety net.
    bad = HopPredicate(amount_min=None)
    object.__setattr__(bad, "direction", "sideways")
    with pytest.raises(ValueError, match="direction must be one of"):
        enumerate_motifs_by_hops(
            out_map, in_map, hops=[bad], seed_keys=["A"],
        )


def test_empty_hops_raises():
    out_map, in_map = _maps_from_rows([("A", "B", "ek1", 0.0, 100.0)])
    with pytest.raises(ValueError, match="hops must be non-empty"):
        enumerate_motifs_by_hops(out_map, in_map, hops=[], seed_keys=["A"])


def test_hop_count_above_six_raises():
    out_map, in_map = _maps_from_rows([("A", "B", "ek1", 0.0, 100.0)])
    with pytest.raises(ValueError, match="hop count must be 1..6"):
        enumerate_motifs_by_hops(
            out_map, in_map, hops=[HopPredicate()] * 7, seed_keys=["A"],
        )


def test_temporal_ordering_strict():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 100.0, 100.0),
        ("B", "C", "ek2", 100.0, 100.0),  # ts == prev → must be rejected
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map, hops=[HopPredicate(), HopPredicate()], seed_keys=["A"],
    )
    assert motifs == []


def test_hop0_time_delta_max_hours_rejected():
    # First hop has no previous timestamp — putting time_delta there is a
    # silent footgun, must be rejected at validation time.
    out_map, in_map = _maps_from_rows([("A", "B", "ek1", 0.0, 100.0)])
    with pytest.raises(ValueError, match="hops\\[0\\].time_delta_max_hours must be None"):
        enumerate_motifs_by_hops(
            out_map, in_map,
            hops=[HopPredicate(time_delta_max_hours=1.0)],
            seed_keys=["A"],
        )


def test_time_delta_max_hours_must_be_positive():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek1", 0.0, 100.0),
        ("B", "C", "ek2", 1.0, 100.0),
    ])
    with pytest.raises(ValueError, match="time_delta_max_hours must be positive"):
        enumerate_motifs_by_hops(
            out_map, in_map,
            hops=[HopPredicate(), HopPredicate(time_delta_max_hours=0.0)],
            seed_keys=["A"],
        )
    with pytest.raises(ValueError, match="time_delta_max_hours must be positive"):
        enumerate_motifs_by_hops(
            out_map, in_map,
            hops=[HopPredicate(), HopPredicate(time_delta_max_hours=-1.0)],
            seed_keys=["A"],
        )


def test_direction_reverse_multi_hop_walks_backwards_in_time():
    # Z→Y at t=10, Y→X at t=20.  Walking reverse from X must surface
    # the predecessor chain X←Y←Z with strict-decreasing timestamps.
    out_map, in_map = _maps_from_rows([
        ("Y", "X", "ek_yx", 20.0, 100.0),
        ("Z", "Y", "ek_zy", 10.0, 100.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[
            HopPredicate(direction="reverse"),
            HopPredicate(direction="reverse"),
        ],
        seed_keys=["X"],
    )
    assert len(motifs) == 1
    assert motifs[0]["nodes"] == ["X", "Y", "Z"]
    assert motifs[0]["edges"] == ["ek_yx", "ek_zy"]
    # Timestamps strictly DECREASING across hops, by causal ordering.
    assert motifs[0]["timestamps"] == [20.0, 10.0]


def test_direction_reverse_rejects_anti_temporal():
    # Same shape but Z→Y happens AFTER Y→X — not a valid causal predecessor.
    out_map, in_map = _maps_from_rows([
        ("Y", "X", "ek_yx", 10.0, 100.0),
        ("Z", "Y", "ek_zy", 20.0, 100.0),  # later than the X-incoming edge
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[
            HopPredicate(direction="reverse"),
            HopPredicate(direction="reverse"),
        ],
        seed_keys=["X"],
    )
    assert motifs == []


def test_direction_any_no_monotonic_constraint():
    # Mixed-direction walk: A→B at t=5, then B←C at t=10. With direction="any"
    # on hop 1, we want both candidates available regardless of monotonicity.
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek_ab", 5.0, 100.0),
        ("C", "B", "ek_cb", 10.0, 100.0),
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[HopPredicate(), HopPredicate(direction="any")],
        seed_keys=["A"],
    )
    # A→B (forward, t=5), then from B walk "any": predecessor C exists
    # at t=10 (later than 5 — not strictly decreasing nor strictly
    # increasing, but allowed under direction="any").
    nxt = sorted(m["nodes"][2] for m in motifs)
    assert nxt == ["C"]


def test_direction_any_window_uses_absolute_delta():
    out_map, in_map = _maps_from_rows([
        ("A", "B", "ek_ab", 1000.0, 100.0),
        ("C", "B", "ek_cb",  500.0, 100.0),  # 500s before A→B
    ])
    motifs = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[
            HopPredicate(),
            HopPredicate(direction="any", time_delta_max_hours=1.0),  # 3600s window
        ],
        seed_keys=["A"],
    )
    # |1000 - 500| = 500s < 3600s → match.
    assert len(motifs) == 1
    motifs_tight = enumerate_motifs_by_hops(
        out_map, in_map,
        hops=[
            HopPredicate(),
            HopPredicate(direction="any", time_delta_max_hours=0.1),  # 360s window
        ],
        seed_keys=["A"],
    )
    # |1000 - 500| = 500s > 360s → reject.
    assert motifs_tight == []
