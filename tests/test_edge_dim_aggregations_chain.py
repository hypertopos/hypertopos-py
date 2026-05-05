"""End-to-end integration tests for chain-anchor edge_dim_aggregations regime.

Mirrors the structure of test_edge_dim_aggregations_builder.py: a single
synthetic-sphere fixture builder, then a battery of GDSReader-driven static
checks (sphere.json, dimension_kinds, geometry delta length) plus
HyperSphere-driven behavior smoke (anomaly_summary, π5_attract_anomaly)
exercising the chain regime added in 0.6.2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.builder.builder import EdgeTableConfig
from hypertopos.builder.mapping import (
    EdgeDimAggregationsConfig,
    EdgeDimensionsConfig,
)
from hypertopos.storage.reader import GDSReader


_CHAIN_FEATURES = ["hop_count", "is_cyclic"]


def _make_chain_sphere(
    out_root: Path,
    *,
    with_chain_aggregations: bool,
) -> GDSBuilder:
    """Build a 5-chain synthetic sphere with chain-anchor aggregation toggle.

    Layout:
      - accounts (anchor): A, B, C, D, E
      - transactions (event): 6 events stitching the accounts in time
      - tx_pattern (event pattern w/ edge_dimensions):
            find_motif_structuring on the (from→to, ts, amount) edge table
      - tx_chains (chain anchor line, 5 chains: long, cyclic, short,
            overlap, isolated)
      - tx_chains_pattern (anchor pattern, optionally aggregating
            find_motif_structuring from tx_pattern)
    """
    b = GDSBuilder("test_chain_eda", str(out_root))
    b.add_line(
        "accounts",
        [
            {"acct_id": "A"},
            {"acct_id": "B"},
            {"acct_id": "C"},
            {"acct_id": "D"},
            {"acct_id": "E"},
        ],
        key_col="acct_id", source_id="t",
    )
    b.add_line(
        "transactions",
        [
            {"tx_id": "evt1", "from_acct": "A", "to_acct": "B",
             "ts": 1000.0, "amount": 10000.0},
            {"tx_id": "evt2", "from_acct": "B", "to_acct": "C",
             "ts": 2000.0, "amount": 5000.0},
            {"tx_id": "evt3", "from_acct": "C", "to_acct": "D",
             "ts": 3000.0, "amount": 2500.0},
            {"tx_id": "evt4", "from_acct": "A", "to_acct": "E",
             "ts": 4000.0, "amount": 8000.0},
            {"tx_id": "evt5", "from_acct": "D", "to_acct": "E",
             "ts": 5000.0, "amount": 1200.0},
            {"tx_id": "evt6", "from_acct": "E", "to_acct": "A",
             "ts": 6000.0, "amount": 600.0},
        ],
        key_col="tx_id", source_id="t",
    )

    edge_dims = EdgeDimensionsConfig(dims={
        "find_motif_structuring": {
            "time_window_hours": 24.0,
            "amt1_min": 5000.0,
            "amt2_max": 7500.0,
        },
    })

    # Event pattern declared FIRST so its sidecar exists when chain anchor
    # tries to aggregate from it.
    b.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="transactions",
        relations=[
            RelationSpec(
                "accounts", fk_col="from_acct", direction="in", required=True,
            ),
        ],
        edge_table=EdgeTableConfig(
            from_col="from_acct", to_col="to_acct",
            timestamp_col="ts", amount_col="amount",
        ),
        edge_dimensions=edge_dims,
    )

    chains = [
        {
            "chain_id": "ch_long",
            "keys": ["A", "B", "C", "D"],
            "event_keys": ["evt1", "evt2", "evt3"],
            "hop_count": 3, "is_cyclic": 0.0,
        },
        {
            "chain_id": "ch_cycle",
            "keys": ["A", "E", "A"],
            "event_keys": ["evt4", "evt6"],
            "hop_count": 2, "is_cyclic": 1.0,
        },
        {
            "chain_id": "ch_short",
            "keys": ["D", "E"],
            "event_keys": ["evt5"],
            "hop_count": 1, "is_cyclic": 0.0,
        },
        {
            "chain_id": "ch_overlap",
            "keys": ["A", "B", "C"],
            "event_keys": ["evt1", "evt2"],
            "hop_count": 2, "is_cyclic": 0.0,
        },
        {
            "chain_id": "ch_isolated",
            "keys": ["E", "A"],
            "event_keys": ["evt6"],
            "hop_count": 1, "is_cyclic": 0.0,
        },
    ]
    b.add_chain_line("tx_chains", chains=chains, features=_CHAIN_FEATURES)

    eda_cfg = (
        EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern",
            dims=("find_motif_structuring",),
        )
        if with_chain_aggregations else None
    )
    b.add_pattern(
        "tx_chains_pattern",
        pattern_type="anchor",
        entity_line="tx_chains",
        relations=[],
        edge_dim_aggregations=eda_cfg,
    )
    return b


# ---------------------------------------------------------------------------
# 1. dimension_kinds grows by 2 (one per agg) when aggregation is declared.
# ---------------------------------------------------------------------------
def test_chain_anchor_aggregations_extends_dimension_kinds(tmp_path: Path):
    out_with = tmp_path / "with"
    _make_chain_sphere(out_with, with_chain_aggregations=True).build()
    out_without = tmp_path / "without"
    _make_chain_sphere(out_without, with_chain_aggregations=False).build()

    sphere_with = GDSReader(str(out_with)).read_sphere()
    sphere_without = GDSReader(str(out_without)).read_sphere()
    pat_with = sphere_with.patterns["tx_chains_pattern"]
    pat_without = sphere_without.patterns["tx_chains_pattern"]
    kinds_with = pat_with.dimension_kinds or []
    kinds_without = pat_without.dimension_kinds or []
    # 1 source dim × 2 aggs (mean + max) = 2 new entries appended at the tail.
    assert len(kinds_with) == len(kinds_without) + 2


# ---------------------------------------------------------------------------
# 2. Polygon delta length matches dimension_kinds length (i.e. aggregates
#    are baked into the geometry on disk).
# ---------------------------------------------------------------------------
def test_chain_anchor_aggregations_baked_into_geometry_delta(tmp_path: Path):
    out_with = tmp_path / "with"
    _make_chain_sphere(out_with, with_chain_aggregations=True).build()
    out_without = tmp_path / "without"
    _make_chain_sphere(out_without, with_chain_aggregations=False).build()

    reader_with = GDSReader(str(out_with))
    reader_without = GDSReader(str(out_without))
    geo_with = reader_with.read_geometry("tx_chains_pattern", version=1)
    geo_without = reader_without.read_geometry("tx_chains_pattern", version=1)

    sphere_with = reader_with.read_sphere()
    sphere_without = reader_without.read_sphere()
    pat_with = sphere_with.patterns["tx_chains_pattern"]
    pat_without = sphere_without.patterns["tx_chains_pattern"]

    # All 5 chains should be present in geometry (every regime).
    assert geo_with.num_rows == 5
    assert geo_without.num_rows == 5

    # Per-row delta length matches the pattern's dimension_kinds count.
    delta_len_with = len(geo_with["delta"][0].as_py())
    delta_len_without = len(geo_without["delta"][0].as_py())
    assert delta_len_with == len(pat_with.dimension_kinds or [])
    assert delta_len_without == len(pat_without.dimension_kinds or [])
    # And with-aggregations is exactly 2 longer than without.
    assert delta_len_with == delta_len_without + 2


# ---------------------------------------------------------------------------
# 3. dim_labels include human-readable aggregated names (no placeholder
#    `dim_N` leakage). Verifies F6 path through Pattern.dim_labels.
# ---------------------------------------------------------------------------
def test_chain_pattern_dim_labels_include_aggregated_names(tmp_path: Path):
    out = tmp_path / "with"
    _make_chain_sphere(out, with_chain_aggregations=True).build()
    pat = GDSReader(str(out)).read_sphere().patterns["tx_chains_pattern"]
    labels = pat.dim_labels
    assert any("_mean" in lbl or "_max" in lbl for lbl in labels), (
        f"dim_labels must include aggregated names; got {labels}"
    )
    for lbl in labels:
        assert not lbl.startswith("dim_"), (
            f"placeholder label leaked: {lbl!r}"
        )


# ---------------------------------------------------------------------------
# 4. anomaly_summary works on aggregated chain pattern (F6 broadcast bug
#    regression check).
# ---------------------------------------------------------------------------
def test_anomaly_summary_works_on_aggregated_chain_pattern(tmp_path: Path):
    from hypertopos.sphere import HyperSphere

    out = tmp_path / "with"
    _make_chain_sphere(out, with_chain_aggregations=True).build()
    hs = HyperSphere.open(str(out))
    nav = hs.session("chain-agg-test").navigator()
    summary = nav.anomaly_summary("tx_chains_pattern")
    assert "total_entities" in summary
    assert "top_driving_dimensions" in summary


# ---------------------------------------------------------------------------
# 5. sphere.json roundtrip for edge_dim_aggregations on a chain pattern.
# ---------------------------------------------------------------------------
def test_chain_anchor_aggregations_in_sphere_json(tmp_path: Path):
    out = tmp_path / "with"
    _make_chain_sphere(out, with_chain_aggregations=True).build()
    raw = json.loads(
        (out / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"),
    )
    pat_node = raw["patterns"]["tx_chains_pattern"]
    assert pat_node["edge_dim_aggregations"]["from"] == "tx_pattern"
    assert pat_node["edge_dim_aggregations"]["dims"] == ["find_motif_structuring"]


# ---------------------------------------------------------------------------
# 6. Cyclic chain present in geometry (is_cyclic=1.0 anchor flows through).
# ---------------------------------------------------------------------------
def test_chain_with_cyclic_chains_aggregates_normally(tmp_path: Path):
    out = tmp_path / "with"
    _make_chain_sphere(out, with_chain_aggregations=True).build()
    geo = GDSReader(str(out)).read_geometry("tx_chains_pattern", version=1)
    pks = geo["primary_key"].to_pylist()
    assert "ch_cycle" in pks
    # Delta vector must have full length on the cyclic row too.
    cycle_idx = pks.index("ch_cycle")
    pat = GDSReader(str(out)).read_sphere().patterns["tx_chains_pattern"]
    assert len(geo["delta"][cycle_idx].as_py()) == len(pat.dimension_kinds or [])


# ---------------------------------------------------------------------------
# 7. Short chain (1 hop, ch_short / ch_isolated) aggregates without crash.
# ---------------------------------------------------------------------------
def test_chain_short_2hop_aggregation(tmp_path: Path):
    out = tmp_path / "with"
    _make_chain_sphere(out, with_chain_aggregations=True).build()
    geo = GDSReader(str(out)).read_geometry("tx_chains_pattern", version=1)
    pks = geo["primary_key"].to_pylist()
    assert "ch_short" in pks
    assert "ch_isolated" in pks


# ---------------------------------------------------------------------------
# 8. Multi-edge chain (3 edges) aggregates over all hops.
# ---------------------------------------------------------------------------
def test_chain_long_multihop_aggregation(tmp_path: Path):
    out = tmp_path / "with"
    _make_chain_sphere(out, with_chain_aggregations=True).build()
    reader = GDSReader(str(out))
    geo = reader.read_geometry("tx_chains_pattern", version=1)
    pat = reader.read_sphere().patterns["tx_chains_pattern"]
    pks = geo["primary_key"].to_pylist()
    assert "ch_long" in pks
    long_idx = pks.index("ch_long")
    # ch_long covers evt1 (10000) → evt2 (5000) → evt3 (2500). evt1 trips
    # find_motif_structuring (amount >= amt1_min=5000), evt3 trips it
    # (amount <= amt2_max=7500). The chain regime aggregates these into
    # find_motif_structuring_mean / find_motif_structuring_max on the chain
    # anchor's delta. Look up by label rather than positional slice — the
    # delta layout interleaves base dims, prop fills, and dim blocks
    # depending on pattern config, and positional assertions silently
    # check the wrong slots if any of those grow later.
    labels = pat.dim_labels
    mean_idx = labels.index("find_motif_structuring_mean")
    max_idx  = labels.index("find_motif_structuring_max")
    delta = geo["delta"][long_idx].as_py()
    assert any(abs(v) > 1e-6 for v in (delta[mean_idx], delta[max_idx])), (
        f"chain aggregates appear all-zero on ch_long — chain regime did "
        f"not fire; mean={delta[mean_idx]} max={delta[max_idx]}"
    )


# ---------------------------------------------------------------------------
# 9. Without aggregations, chain pattern has only base dims (regression).
# ---------------------------------------------------------------------------
def test_chain_without_aggregations_unchanged(tmp_path: Path):
    out = tmp_path / "without"
    _make_chain_sphere(out, with_chain_aggregations=False).build()
    raw = json.loads(
        (out / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"),
    )
    pat_node = raw["patterns"]["tx_chains_pattern"]
    assert "edge_dim_aggregations" not in pat_node
    pat = GDSReader(str(out)).read_sphere().patterns["tx_chains_pattern"]
    for lbl in pat.dim_labels:
        assert "_mean" not in lbl
        assert "_max" not in lbl


# ---------------------------------------------------------------------------
# 10. Zero-chain extraction with declared aggregations cannot build:
#     `add_chain_line(chains=[])` produces a zero-row chain table with no
#     auto-derived chain dims, so the chain anchor pattern has zero
#     dimensions and `_validate()` fails before chain dispatch.
#     Either way the build is rejected at build() dispatch — this test pins
#     that contract so a future relaxation (e.g. allowing zero-dim chain
#     anchors) is caught.
# ---------------------------------------------------------------------------
def test_chain_with_zero_chains_extracted_raises(tmp_path: Path):
    out = tmp_path / "gds_zero_chains"
    b = GDSBuilder("zero", str(out))
    b.add_line(
        "accounts",
        [{"acct_id": "A"}, {"acct_id": "B"}],
        key_col="acct_id", source_id="t",
    )
    b.add_line(
        "transactions",
        [
            {"tx_id": "evt1", "from_acct": "A", "to_acct": "B",
             "ts": 1000.0, "amount": 100.0},
        ],
        key_col="tx_id", source_id="t",
    )
    edge_dims = EdgeDimensionsConfig(dims={
        "find_motif_structuring": {
            "time_window_hours": 24.0,
            "amt1_min": 5000.0, "amt2_max": 7500.0,
        },
    })
    b.add_pattern(
        "tx_pattern", pattern_type="event", entity_line="transactions",
        relations=[
            RelationSpec(
                "accounts", fk_col="from_acct", direction="in", required=True,
            ),
        ],
        edge_table=EdgeTableConfig(
            from_col="from_acct", to_col="to_acct",
            timestamp_col="ts", amount_col="amount",
        ),
        edge_dimensions=edge_dims,
    )
    # Zero chains → empty chain line (no auto-derived chain dims, no
    # chain_events column emitted).
    b.add_chain_line("tx_chains", chains=[], features=["hop_count"])
    b.add_pattern(
        "tx_chains_pattern", pattern_type="anchor", entity_line="tx_chains",
        relations=[],
        edge_dim_aggregations=EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern",
            dims=("find_motif_structuring",),
        ),
    )
    # Build raises a chain-aware error at _validate() pinning the actual
    # cause (zero chains extracted), not the generic "has no dimensions"
    # message that would have surfaced before the chain-aware check landed.
    with pytest.raises(ValueError, match="chain extraction returned 0 chains"):
        b.build()
