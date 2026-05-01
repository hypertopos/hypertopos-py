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
    assert AGGREGATE_NAMES == ("mean", "max")


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


def test_aggregate_chain_kind_raises_not_implemented():
    edges = _edges([("e1", "A", "B")])
    side  = _sidecar([("e1", 2.0, 0.0)])
    with pytest.raises(NotImplementedError, match="0.6.2"):
        aggregate_edge_dims_for_anchor(
            anchor_keys=["chain1"],
            edges=edges,
            sidecar=side,
            dims=["pair_edge_count"],
            anchor_kind="chain",
        )


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
