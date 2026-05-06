from __future__ import annotations

import pyarrow as pa
import pytest

from hypertopos.engine.edge_features import (
    AGGREGATE_NAMES,
    aggregate_edge_dims_for_anchor,
    aggregate_kind,
)


def _sidecar(rows: list[tuple[str, float, float]]) -> pa.Table:
    return pa.table({
        "event_key": [r[0] for r in rows],
        "pair_edge_count": pa.array([r[1] for r in rows], type=pa.float32()),
        "find_motif_structuring": pa.array([r[2] for r in rows], type=pa.float32()),
    })


def _edges(rows: list[tuple[str, str, str]]) -> pa.Table:
    return pa.table({
        "event_key": [r[0] for r in rows],
        "from_key":  [r[1] for r in rows],
        "to_key":    [r[2] for r in rows],
    })


def test_AGGREGATE_NAMES_constant():
    assert AGGREGATE_NAMES == (
        "mean", "max", "std", "p95", "count_above_threshold",
    )


def test_aggregate_kind_count_above_threshold_is_poisson():
    """count_above_threshold counts edges crossing a per-dim threshold —
    a Poisson rate by construction, regardless of source kind."""
    assert aggregate_kind("poisson", "count_above_threshold") == "poisson"
    assert aggregate_kind("gaussian", "count_above_threshold") == "poisson"
    assert aggregate_kind("bernoulli", "count_above_threshold") == "poisson"


def test_count_above_threshold_default_population_p95():
    """Default threshold = population p95 of source dim from sidecar."""
    edges = _edges([
        ("e1", "A", "B"), ("e2", "A", "B"), ("e3", "A", "B"),
        ("e4", "A", "B"), ("e5", "A", "C"),
    ])
    side = _sidecar([
        ("e1", 1.0, 0.0), ("e2", 2.0, 0.0), ("e3", 3.0, 0.0),
        ("e4", 4.0, 0.0), ("e5", 100.0, 0.0),
    ])
    # Population p95 of pair_edge_count = ~80 (between 4 and 100); only e5 (100)
    # crosses. e5 → A→C anchor, so A→C count_above_threshold = 1, A→B = 0.
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B", "A→C"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
    )
    pks = out["primary_key"].to_pylist()
    assert (
        out["pair_edge_count_count_above_threshold"][pks.index("A→C")].as_py()
        == pytest.approx(1.0)
    )
    assert (
        out["pair_edge_count_count_above_threshold"][pks.index("A→B")].as_py()
        == pytest.approx(0.0)
    )


def test_count_above_threshold_user_override():
    """User-provided threshold overrides the default population p95."""
    edges = _edges([
        ("e1", "A", "B"), ("e2", "A", "B"), ("e3", "A", "B"),
    ])
    side = _sidecar([
        ("e1", 1.0, 0.0), ("e2", 5.0, 0.0), ("e3", 10.0, 0.0),
    ])
    # Threshold=4: e2 (5) and e3 (10) cross → A→B count = 2
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
        thresholds={"pair_edge_count": 4.0},
    )
    assert (
        out["pair_edge_count_count_above_threshold"][0].as_py()
        == pytest.approx(2.0)
    )


def test_aggregate_single_key_anchor_takes_union_of_in_out():
    edges = _edges([("e1", "A", "B"), ("e2", "A", "C"), ("e3", "B", "A")])
    side  = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0), ("e3", 1.0, 0.0)])
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A", "B", "C", "D"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count", "find_motif_structuring"],
        anchor_kind="single",
    )
    pks = out["primary_key"].to_pylist()
    a = pks.index("A")
    b = pks.index("B")
    c = pks.index("C")
    d = pks.index("D")
    assert out["pair_edge_count_mean"][a].as_py() == pytest.approx((2 + 4 + 1) / 3)
    assert out["pair_edge_count_max"][a].as_py() == pytest.approx(4.0)
    assert out["find_motif_structuring_mean"][a].as_py() == pytest.approx(1 / 3)
    assert out["find_motif_structuring_max"][a].as_py() == pytest.approx(1.0)
    # B touches e1 (in) + e3 (out) → mean=(2+1)/2=1.5, max=2
    assert out["pair_edge_count_mean"][b].as_py() == pytest.approx(1.5)
    assert out["pair_edge_count_max"][b].as_py() == pytest.approx(2.0)
    # C touches e2 only → mean=max=4
    assert out["pair_edge_count_mean"][c].as_py() == pytest.approx(4.0)
    assert out["pair_edge_count_max"][c].as_py() == pytest.approx(4.0)
    # D has zero edges → all zeros
    assert out["pair_edge_count_mean"][d].as_py() == 0.0
    assert out["pair_edge_count_max"][d].as_py() == 0.0
    assert out["find_motif_structuring_mean"][d].as_py() == 0.0
    assert out["find_motif_structuring_max"][d].as_py() == 0.0


def test_aggregate_composite_key_anchor_default_separator():
    """Default separator is '→' (matches GDSBuilder.add_composite_line default)."""
    edges = _edges([("e1", "A", "B"), ("e2", "A", "C"), ("e3", "B", "A"), ("e4", "A", "B")])
    side  = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0), ("e3", 1.0, 0.0), ("e4", 6.0, 1.0)])
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B", "A→C", "B→A"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
    )
    pks = out["primary_key"].to_pylist()
    assert out["pair_edge_count_mean"][pks.index("A→B")].as_py() == pytest.approx(4.0)
    assert out["pair_edge_count_max"][pks.index("A→B")].as_py() == pytest.approx(6.0)
    assert out["pair_edge_count_mean"][pks.index("A→C")].as_py() == pytest.approx(4.0)
    assert out["pair_edge_count_mean"][pks.index("B→A")].as_py() == pytest.approx(1.0)


def test_aggregate_composite_key_anchor_custom_separator():
    """Custom pair_separator must propagate to anchor PK construction.

    Discriminator test: if the engine ignored the parameter (e.g. hard-coded
    '__' bug pre-fix), all aggregates would be zero because the anchor_keys
    list passed in (with '|' separator) would not match the engine-generated
    PKs (with '__' separator).
    """
    edges = _edges([("e1", "A", "B"), ("e2", "A", "C"), ("e3", "B", "A")])
    side  = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0), ("e3", 1.0, 0.0)])
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A|B", "A|C", "B|A"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
        pair_separator="|",
    )
    pks = out["primary_key"].to_pylist()
    # If separator handling were broken, these would all be 0.0
    assert out["pair_edge_count_mean"][pks.index("A|B")].as_py() == pytest.approx(2.0)
    assert out["pair_edge_count_mean"][pks.index("A|C")].as_py() == pytest.approx(4.0)
    assert out["pair_edge_count_mean"][pks.index("B|A")].as_py() == pytest.approx(1.0)


def test_aggregate_kind_mapping():
    # _mean → gaussian via CLT, regardless of source kind.
    assert aggregate_kind("poisson", "mean") == "gaussian"
    assert aggregate_kind("gaussian", "mean") == "gaussian"
    assert aggregate_kind("bernoulli", "mean") == "gaussian"  # proportion in [0,1]
    # _max → gaussian (Gumbel-ish, better approx by gaussian than source) for
    # poisson/gaussian; bernoulli stays bernoulli (max is 0/1).
    assert aggregate_kind("poisson", "max") == "gaussian"
    assert aggregate_kind("gaussian", "max") == "gaussian"
    assert aggregate_kind("bernoulli", "max") == "bernoulli"


def test_aggregate_unknown_dim_raises():
    edges = _edges([("e1", "A", "B")])
    side  = _sidecar([("e1", 2.0, 0.0)])
    with pytest.raises(ValueError, match="unknown edge dimension"):
        aggregate_edge_dims_for_anchor(
            anchor_keys=["A"],
            edges=edges,
            sidecar=side,
            dims=["bogus_dim"],
            anchor_kind="single",
        )


def test_aggregate_dim_missing_from_sidecar_raises():
    edges = _edges([("e1", "A", "B")])
    # sidecar without find_motif_structuring column
    side = pa.table({
        "event_key": ["e1"],
        "pair_edge_count": pa.array([2.0], type=pa.float32()),
    })
    with pytest.raises(ValueError, match="not present in sidecar"):
        aggregate_edge_dims_for_anchor(
            anchor_keys=["A"],
            edges=edges,
            sidecar=side,
            dims=["find_motif_structuring"],
            anchor_kind="single",
        )


def test_aggregate_invalid_anchor_kind_raises():
    edges = _edges([("e1", "A", "B")])
    side  = _sidecar([("e1", 2.0, 0.0)])
    with pytest.raises(ValueError, match="anchor_kind must be"):
        aggregate_edge_dims_for_anchor(
            anchor_keys=["A"],
            edges=edges,
            sidecar=side,
            dims=["pair_edge_count"],
            anchor_kind="bogus",
        )


def test_aggregate_empty_edges_returns_zeros():
    edges = pa.table({
        "event_key": pa.array([], type=pa.string()),
        "from_key":  pa.array([], type=pa.string()),
        "to_key":    pa.array([], type=pa.string()),
    })
    side = pa.table({
        "event_key": pa.array([], type=pa.string()),
        "pair_edge_count": pa.array([], type=pa.float32()),
    })
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A", "B"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="single",
    )
    assert out["pair_edge_count_mean"].to_pylist() == [0.0, 0.0]
    assert out["pair_edge_count_max"].to_pylist() == [0.0, 0.0]


def test_aggregate_chain_kind_with_chain_events_works():
    """Chain regime aggregates dim values per chain via explosion+groupby."""
    sidecar = pa.table({
        "event_key": ["evt1", "evt2", "evt3", "evt4", "evt5"],
        "find_motif_structuring": pa.array(
            [1.0, 0.0, 1.0, 0.0, 1.0], type=pa.float32(),
        ),
    })
    edges = pa.table({
        "event_key": pa.array([], type=pa.string()),
        "from_key": pa.array([], type=pa.string()),
        "to_key":   pa.array([], type=pa.string()),
    })

    result = aggregate_edge_dims_for_anchor(
        anchor_keys=["chain_a", "chain_b", "chain_c"],
        edges=edges,
        sidecar=sidecar,
        dims=["find_motif_structuring"],
        anchor_kind="chain",
        chain_events=["evt1,evt2,evt3", "evt2,evt4", "evt5"],
    )

    pks = result["primary_key"].to_pylist()
    means = result["find_motif_structuring_mean"].to_numpy()
    maxs  = result["find_motif_structuring_max"].to_numpy()
    assert pks == ["chain_a", "chain_b", "chain_c"]
    assert means == pytest.approx([2.0 / 3.0, 0.0, 1.0], abs=1e-6)
    assert maxs  == pytest.approx([1.0, 0.0, 1.0], abs=1e-6)


def test_aggregate_chain_kind_requires_chain_events_arg():
    """Engine raises when chain regime is requested without chain_events list."""
    sidecar = pa.table({
        "event_key": ["evt1"],
        "find_motif_structuring": pa.array([1.0], type=pa.float32()),
    })
    edges = pa.table({
        "event_key": pa.array([], type=pa.string()),
        "from_key": pa.array([], type=pa.string()),
        "to_key":   pa.array([], type=pa.string()),
    })
    with pytest.raises(ValueError, match="chain regime requires chain_events"):
        aggregate_edge_dims_for_anchor(
            anchor_keys=["chain_a"],
            edges=edges,
            sidecar=sidecar,
            dims=["find_motif_structuring"],
            anchor_kind="chain",
            chain_events=None,
        )


def test_aggregate_chain_kind_chain_events_length_mismatch():
    """Engine raises when len(chain_events) != len(anchor_keys)."""
    sidecar = pa.table({
        "event_key": ["evt1"],
        "find_motif_structuring": pa.array([1.0], type=pa.float32()),
    })
    edges = pa.table({
        "event_key": pa.array([], type=pa.string()),
        "from_key": pa.array([], type=pa.string()),
        "to_key":   pa.array([], type=pa.string()),
    })
    with pytest.raises(ValueError, match="must match"):
        aggregate_edge_dims_for_anchor(
            anchor_keys=["chain_a", "chain_b"],
            edges=edges,
            sidecar=sidecar,
            dims=["find_motif_structuring"],
            anchor_kind="chain",
            chain_events=["evt1"],
        )


def test_aggregate_chain_kind_empty_chain_events_string_returns_zero():
    """Per-chain empty chain_events string defaults to 0.0 aggregates."""
    sidecar = pa.table({
        "event_key": ["evt1", "evt2"],
        "find_motif_structuring": pa.array([1.0, 1.0], type=pa.float32()),
    })
    edges = pa.table({
        "event_key": pa.array([], type=pa.string()),
        "from_key": pa.array([], type=pa.string()),
        "to_key":   pa.array([], type=pa.string()),
    })
    result = aggregate_edge_dims_for_anchor(
        anchor_keys=["chain_with_edges", "chain_empty"],
        edges=edges,
        sidecar=sidecar,
        dims=["find_motif_structuring"],
        anchor_kind="chain",
        chain_events=["evt1,evt2", ""],
    )
    means = result["find_motif_structuring_mean"].to_numpy()
    maxs  = result["find_motif_structuring_max"].to_numpy()
    assert means == pytest.approx([1.0, 0.0])
    assert maxs  == pytest.approx([1.0, 0.0])


def test_aggregate_chain_kind_unknown_event_keys_skipped():
    """Chain event_keys not in sidecar contribute nothing to aggregates."""
    sidecar = pa.table({
        "event_key": ["evt1", "evt2"],
        "find_motif_structuring": pa.array([1.0, 1.0], type=pa.float32()),
    })
    edges = pa.table({
        "event_key": pa.array([], type=pa.string()),
        "from_key": pa.array([], type=pa.string()),
        "to_key":   pa.array([], type=pa.string()),
    })
    result = aggregate_edge_dims_for_anchor(
        anchor_keys=["chain_with_unknown"],
        edges=edges,
        sidecar=sidecar,
        dims=["find_motif_structuring"],
        anchor_kind="chain",
        chain_events=["evt1,evt_missing,evt2"],
    )
    assert result["find_motif_structuring_mean"][0].as_py() == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# k>2 composite anchor regime — F1.b
# -----------------------------------------------------------------------------

def test_aggregate_composite_kgt2_tripartite_uses_positional_key_cols():
    """Tripartite anchor: key_cols=[from_key, to_key, currency] builds 3-tuple PK."""
    edges = pa.table({
        "event_key": ["e1", "e2", "e3", "e4"],
        "from_key":  ["A", "A", "B", "A"],
        "to_key":    ["B", "C", "A", "B"],
        "currency":  ["USD", "USD", "EUR", "EUR"],
    })
    side = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0), ("e3", 1.0, 0.0), ("e4", 6.0, 1.0)])
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B→USD", "A→B→EUR", "A→C→USD", "B→A→EUR"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count", "find_motif_structuring"],
        anchor_kind="pair",
        key_cols=["from_key", "to_key", "currency"],
    )
    pks = out["primary_key"].to_pylist()
    # A→B→USD: e1 only → mean=max=2
    assert out["pair_edge_count_mean"][pks.index("A→B→USD")].as_py() == pytest.approx(2.0)
    assert out["pair_edge_count_max"][pks.index("A→B→USD")].as_py() == pytest.approx(2.0)
    # A→B→EUR: e4 only → mean=max=6
    assert out["pair_edge_count_mean"][pks.index("A→B→EUR")].as_py() == pytest.approx(6.0)
    assert out["pair_edge_count_max"][pks.index("A→B→EUR")].as_py() == pytest.approx(6.0)
    # A→C→USD: e2 only → mean=max=4
    assert out["pair_edge_count_mean"][pks.index("A→C→USD")].as_py() == pytest.approx(4.0)
    # B→A→EUR: e3 only → mean=max=1
    assert out["pair_edge_count_mean"][pks.index("B→A→EUR")].as_py() == pytest.approx(1.0)
    assert out["find_motif_structuring_max"][pks.index("A→B→EUR")].as_py() == pytest.approx(1.0)


def test_aggregate_composite_kgt2_custom_separator():
    """Tripartite with custom separator '|' instead of default '→'."""
    edges = pa.table({
        "event_key": ["e1", "e2"],
        "from_key":  ["A", "A"],
        "to_key":    ["B", "B"],
        "currency":  ["USD", "USD"],
    })
    side = _sidecar([("e1", 3.0, 0.0), ("e2", 5.0, 1.0)])
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A|B|USD"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
        pair_separator="|",
        key_cols=["from_key", "to_key", "currency"],
    )
    assert out["pair_edge_count_mean"][0].as_py() == pytest.approx(4.0)
    assert out["pair_edge_count_max"][0].as_py() == pytest.approx(5.0)


def test_aggregate_composite_kgt2_aggregates_across_groups():
    """4-tuple anchor with multiple events per group — verify mean/max correct."""
    edges = pa.table({
        "event_key": ["e1", "e2", "e3", "e4", "e5"],
        "from_key":  ["A", "A", "A", "B", "A"],
        "to_key":    ["B", "B", "B", "C", "C"],
        "currency":  ["USD", "USD", "USD", "EUR", "USD"],
        "channel":   ["wire", "wire", "wire", "ach", "ach"],
    })
    side = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0), ("e3", 6.0, 0.0),
                     ("e4", 1.0, 0.0), ("e5", 8.0, 1.0)])
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B→USD→wire", "A→C→USD→ach", "B→C→EUR→ach"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
        key_cols=["from_key", "to_key", "currency", "channel"],
    )
    pks = out["primary_key"].to_pylist()
    # A→B→USD→wire: e1+e2+e3 → mean=4, max=6
    assert out["pair_edge_count_mean"][pks.index("A→B→USD→wire")].as_py() == pytest.approx(4.0)
    assert out["pair_edge_count_max"][pks.index("A→B→USD→wire")].as_py() == pytest.approx(6.0)
    # A→C→USD→ach: e5 only → mean=max=8
    assert out["pair_edge_count_mean"][pks.index("A→C→USD→ach")].as_py() == pytest.approx(8.0)
    # B→C→EUR→ach: e4 only → mean=max=1
    assert out["pair_edge_count_mean"][pks.index("B→C→EUR→ach")].as_py() == pytest.approx(1.0)


def test_aggregate_composite_k2_backward_compat_no_key_cols():
    """k=2 callers without key_cols still work (default = from_key, to_key)."""
    edges = _edges([("e1", "A", "B"), ("e2", "A", "B"), ("e3", "B", "A")])
    side  = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0), ("e3", 1.0, 0.0)])
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B", "B→A"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
    )
    pks = out["primary_key"].to_pylist()
    assert out["pair_edge_count_mean"][pks.index("A→B")].as_py() == pytest.approx(3.0)
    assert out["pair_edge_count_max"][pks.index("A→B")].as_py() == pytest.approx(4.0)
    assert out["pair_edge_count_mean"][pks.index("B→A")].as_py() == pytest.approx(1.0)


def test_aggregate_composite_kgt2_missing_key_col_raises_without_event_table():
    """k>2 with property key_col not in joined and event_table=None raises."""
    edges = _edges([("e1", "A", "B")])
    side  = _sidecar([("e1", 2.0, 0.0)])
    with pytest.raises(ValueError, match="not present in joined.*event_table=None"):
        aggregate_edge_dims_for_anchor(
            anchor_keys=["A→B→USD"],
            edges=edges,
            sidecar=side,
            dims=["pair_edge_count"],
            anchor_kind="pair",
            key_cols=["from_key", "to_key", "currency"],
        )


def test_aggregate_composite_kgt2_property_from_event_table():
    """k>2 property column missing from edges is joined from event_table."""
    edges = _edges([("e1", "A", "B"), ("e2", "A", "B")])
    side  = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0)])
    event_table = pa.table({
        "event_key": ["e1", "e2"],
        "currency":  ["USD", "EUR"],
    })
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B→USD", "A→B→EUR"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
        key_cols=["from_key", "to_key", "currency"],
        event_table=event_table,
    )
    pks = out["primary_key"].to_pylist()
    assert out["pair_edge_count_mean"][pks.index("A→B→USD")].as_py() == pytest.approx(2.0)
    assert out["pair_edge_count_mean"][pks.index("A→B→EUR")].as_py() == pytest.approx(4.0)


def test_aggregate_composite_kgt2_event_table_uses_primary_key_alias():
    """event_table with primary_key (event line convention) instead of event_key works."""
    edges = _edges([("e1", "A", "B"), ("e2", "A", "B")])
    side  = _sidecar([("e1", 2.0, 0.0), ("e2", 4.0, 1.0)])
    event_table = pa.table({
        "primary_key": ["e1", "e2"],  # event line uses primary_key, not event_key
        "currency":    ["USD", "EUR"],
    })
    out = aggregate_edge_dims_for_anchor(
        anchor_keys=["A→B→USD", "A→B→EUR"],
        edges=edges,
        sidecar=side,
        dims=["pair_edge_count"],
        anchor_kind="pair",
        key_cols=["from_key", "to_key", "currency"],
        event_table=event_table,
    )
    pks = out["primary_key"].to_pylist()
    assert out["pair_edge_count_mean"][pks.index("A→B→USD")].as_py() == pytest.approx(2.0)
    assert out["pair_edge_count_mean"][pks.index("A→B→EUR")].as_py() == pytest.approx(4.0)
