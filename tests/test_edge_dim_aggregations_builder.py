"""Builder integration: edge_dim_aggregations on an anchor pattern bakes
mean/max aggregates of per-edge sidecar dims into the anchor polygon."""
from __future__ import annotations

import json
from pathlib import Path

from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.builder.builder import EdgeTableConfig
from hypertopos.builder.mapping import (
    EdgeDimAggregationsConfig,
    EdgeDimensionsConfig,
)
from hypertopos.storage.reader import GDSReader


def _make_sphere(
    out_root: Path,
    *,
    with_aggregations: bool,
) -> GDSBuilder:
    b = GDSBuilder("test_s1_ext", str(out_root))
    b.add_line(
        "accounts",
        [
            {"acct_id": "A", "name": "alpha"},
            {"acct_id": "B", "name": "beta"},
            {"acct_id": "C", "name": "gamma"},
            {"acct_id": "D", "name": "delta"},
        ],
        key_col="acct_id", source_id="t",
    )
    b.add_line(
        "transactions",
        [
            {"tx_id": "ek1", "from_acct": "A", "to_acct": "B",
             "ts": 0.0,    "amount": 20000.0},
            {"tx_id": "ek2", "from_acct": "B", "to_acct": "C",
             "ts": 100.0,  "amount": 5000.0},
            {"tx_id": "ek3", "from_acct": "C", "to_acct": "D",
             "ts": 200.0,  "amount": 5000.0},
            {"tx_id": "ek4", "from_acct": "A", "to_acct": "B",
             "ts": 1000.0, "amount": 1000.0},
            {"tx_id": "ek5", "from_acct": "C", "to_acct": "D",
             "ts": 5000.0, "amount": 2000.0},
            {"tx_id": "ek6", "from_acct": "D", "to_acct": "A",
             "ts": 8000.0, "amount": 3000.0},
        ],
        key_col="tx_id", source_id="t",
    )

    edge_dims = EdgeDimensionsConfig(dims={
        "pair_edge_count": {},
        "find_motif_structuring": {
            "time_window_hours": 1.0,
            "amt1_min": 10000.0,
            "amt2_max": 10000.0,
        },
    })

    # Event pattern declared FIRST so its sidecar is built before the
    # anchor pattern that aggregates from it.
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

    eda_cfg = (
        EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern",
            dims=("pair_edge_count",),
        )
        if with_aggregations else None
    )

    b.add_pattern(
        "account_pattern",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[],
        edge_dim_aggregations=eda_cfg,
    )
    b.add_derived_dimension(
        anchor_line="accounts",
        event_line="transactions",
        anchor_fk="from_acct",
        metric="count",
        metric_col=None,
        dimension_name="tx_out_count",
        edge_max="auto",
    )
    return b


def test_account_pattern_with_aggregations_extends_dimension_kinds(
    tmp_path: Path,
):
    out_baseline = tmp_path / "gds_baseline"
    _make_sphere(out_baseline, with_aggregations=False).build()
    sphere_b = GDSReader(str(out_baseline)).read_sphere()
    base_kinds = sphere_b.patterns["account_pattern"].dimension_kinds or []
    base_dim = len(base_kinds)

    out_with = tmp_path / "gds_with"
    _make_sphere(out_with, with_aggregations=True).build()
    sphere_w = GDSReader(str(out_with)).read_sphere()
    pat = sphere_w.patterns["account_pattern"]
    new_kinds = pat.dimension_kinds or []

    # 1 source dim × 5 aggs (mean / max / std / p95 / count_above_threshold)
    # = 5 new kinds appended at the tail. pair_edge_count source kind is
    # poisson; mean/max/std/p95 map to gaussian; count_above_threshold maps
    # to poisson (count of edges above the per-dim threshold).
    assert len(new_kinds) == base_dim + 5
    assert new_kinds[-5:] == [
        "gaussian", "gaussian", "gaussian", "gaussian", "poisson",
    ]


def test_account_pattern_aggregations_baked_into_geometry_delta(
    tmp_path: Path,
):
    out = tmp_path / "gds_with"
    _make_sphere(out, with_aggregations=True).build()
    reader = GDSReader(str(out))
    geo = reader.read_geometry("account_pattern", version=1)
    sphere = reader.read_sphere()
    pat = sphere.patterns["account_pattern"]

    assert geo.num_rows == 4  # 4 accounts
    delta_len = len(geo["delta"][0].as_py())
    assert delta_len == len(pat.dimension_kinds)


def test_account_pattern_edge_dim_aggregations_in_sphere_json(tmp_path: Path):
    out = tmp_path / "gds_with"
    _make_sphere(out, with_aggregations=True).build()
    raw = json.loads(
        (out / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"),
    )
    pat_node = raw["patterns"]["account_pattern"]
    assert pat_node["edge_dim_aggregations"]["from"] == "tx_pattern"
    assert pat_node["edge_dim_aggregations"]["dims"] == ["pair_edge_count"]


def test_anomaly_summary_works_on_aggregated_pattern(tmp_path: Path):
    """Integration: navigator.anomaly_summary on a freshly-built sphere
    with edge_dim_aggregations declared must NOT raise the
    'operands could not be broadcast together' error. Pre-fix the
    Pattern.dim_labels was 33-long but cluster delta vectors had length
    37 (33 base + 4 aggregated), leading to a (33,) (37,) (33,)
    broadcast error in the top_driving_dimensions accumulator. The
    Pattern.dim_labels fix in this PR sizes the array correctly."""
    from hypertopos.sphere import HyperSphere

    out = tmp_path / "gds_with_agg"
    _make_sphere(out, with_aggregations=True).build()
    hs = HyperSphere.open(str(out))
    nav = hs.session("agg-test").navigator()
    summary = nav.anomaly_summary("account_pattern")
    assert "total_entities" in summary
    assert "top_driving_dimensions" in summary
    sphere = nav._storage.read_sphere()
    pat = sphere.patterns["account_pattern"]
    assert pat.delta_dim() > len(pat.relations), (
        "delta_dim must include aggregated edge-dim count"
    )
    assert any("_mean" in lbl or "_max" in lbl for lbl in pat.dim_labels), (
        f"dim_labels must include aggregated names; got {pat.dim_labels}"
    )


def test_anchor_without_aggregations_unchanged(tmp_path: Path):
    out = tmp_path / "gds_baseline"
    _make_sphere(out, with_aggregations=False).build()
    raw = json.loads(
        (out / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"),
    )
    pat_node = raw["patterns"]["account_pattern"]
    assert "edge_dim_aggregations" not in pat_node


def _make_sphere_with_pair_anchor(out_root: Path) -> GDSBuilder:
    """Toy sphere with a real composite (pair) anchor pattern via add_composite_line.

    Discriminator: this exercises the actual code path that resolves the
    composite-line separator at the builder call site. If the engine were
    still hard-coding "__" (pre-fix), every pair aggregate would be 0.0 and
    test_pair_anchor_aggregations_nonzero would catch it.
    """
    b = GDSBuilder("test_pair", str(out_root))
    b.add_line(
        "accounts",
        [
            {"acct_id": "A", "name": "alpha"},
            {"acct_id": "B", "name": "beta"},
            {"acct_id": "C", "name": "gamma"},
        ],
        key_col="acct_id", source_id="t",
    )
    b.add_line(
        "transactions",
        [
            {"tx_id": "ek1", "from_acct": "A", "to_acct": "B",
             "ts": 0.0,    "amount": 1000.0},
            {"tx_id": "ek2", "from_acct": "A", "to_acct": "B",
             "ts": 100.0,  "amount": 1500.0},
            {"tx_id": "ek3", "from_acct": "A", "to_acct": "B",
             "ts": 200.0,  "amount": 2000.0},
            {"tx_id": "ek4", "from_acct": "A", "to_acct": "C",
             "ts": 300.0,  "amount": 500.0},
            {"tx_id": "ek5", "from_acct": "B", "to_acct": "A",
             "ts": 400.0,  "amount": 800.0},
        ],
        key_col="tx_id", source_id="t",
    )

    edge_dims = EdgeDimensionsConfig(dims={"pair_edge_count": {}})

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

    b.add_composite_line(
        "account_pairs",
        event_line="transactions",
        key_cols=["from_acct", "to_acct"],
    )

    b.add_pattern(
        "account_pairs_pattern",
        pattern_type="anchor",
        entity_line="account_pairs",
        relations=[],
        edge_dim_aggregations=EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern",
            dims=("pair_edge_count",),
        ),
    )
    b.add_derived_dimension(
        anchor_line="account_pairs",
        event_line="transactions",
        anchor_fk=["from_acct", "to_acct"],
        metric="count",
        metric_col=None,
        dimension_name="pair_count",
        edge_max="auto",
    )
    return b


def test_chain_or_unknown_anchor_kind_raises_not_implemented(tmp_path: Path):
    """Discriminator: when an anchor entity_line is neither a relation of the
    source event pattern nor a registered composite_line, the dispatch must
    raise NotImplementedError instead of silently coercing to 'pair' kind
    (which produced all-zero aggregates pre-fix)."""
    import pytest

    out_root = tmp_path / "gds_chain"
    b = GDSBuilder("test_chain", str(out_root))
    b.add_line(
        "accounts",
        [{"acct_id": "A"}, {"acct_id": "B"}],
        key_col="acct_id", source_id="t",
    )
    b.add_line(
        "transactions",
        [
            {"tx_id": "ek1", "from_acct": "A", "to_acct": "B",
             "ts": 0.0, "amount": 100.0},
        ],
        key_col="tx_id", source_id="t",
    )
    b.add_line(
        "tx_chains",
        [{"chain_id": "c1", "hop_count": 1}],
        key_col="chain_id", source_id="t", role="anchor",
    )
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
        edge_dimensions=EdgeDimensionsConfig(dims={"pair_edge_count": {}}),
    )
    b.add_pattern(
        "chain_pattern",
        pattern_type="anchor",
        entity_line="tx_chains",
        relations=[],
        edge_dim_aggregations=EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern",
            dims=("pair_edge_count",),
        ),
    )
    b.add_precomputed_dimension(
        anchor_line="tx_chains",
        dimension_name="hop_count",
        edge_max="auto",
    )
    with pytest.raises(NotImplementedError, match="0.6.2|chain"):
        b.build()


def test_pair_anchor_aggregations_nonzero(tmp_path: Path):
    """End-to-end discriminator: pair anchor with real composite-line
    separator must produce non-zero aggregates on the (A→B) pair (3 edges,
    pair_edge_count = 3 across all three).

    Pre-fix engine hard-coded "__", AML's CompositeLineSpec defaults to "→",
    so every pair anchor would silently get all-zero aggregates and look
    "fine" at runtime while doing nothing.
    """
    out = tmp_path / "gds_pair"
    _make_sphere_with_pair_anchor(out).build()
    reader = GDSReader(str(out))
    geo = reader.read_geometry("account_pairs_pattern", version=1)
    sphere = reader.read_sphere()
    pat = sphere.patterns["account_pairs_pattern"]

    # 3 unique pairs: A→B, A→C, B→A
    assert geo.num_rows == 3
    pks = geo["primary_key"].to_pylist()
    assert "A→B" in pks
    assert "A→C" in pks
    assert "B→A" in pks

    # Find pair_edge_count_mean column index in delta
    kinds = pat.dimension_kinds or []
    # Layout: relations + derived_dim_count + agg_mean + agg_max
    # The two trailing dims are the aggregates.
    assert len(kinds) >= 2

    deltas = [geo["delta"][i].as_py() for i in range(geo.num_rows)]
    pair_idx = pks.index("A→B")
    # A→B has 3 edges; pair_edge_count column for those edges is 3 each.
    # _mean and _max of a constant 3.0 column = 3.0, but z-scored against
    # population (mu, sigma). What we really need to check is that the
    # delta values are NOT all zero — ie. the aggregation actually fired.
    last_two_a_b = deltas[pair_idx][-2:]
    assert any(abs(v) > 1e-6 for v in last_two_a_b), (
        f"pair regime aggregates appear all-zero on (A→B) — separator "
        f"plumbing likely broken; delta tail was {last_two_a_b}"
    )


def test_edge_dim_thresholds_persisted_into_calibration_history(
    tmp_path: Path,
):
    """End-to-end: anchor pattern with `edge_dim_aggregations:` writes
    per-source-dim `_count_above_threshold` cutoffs into the calibration
    epoch JSON alongside mu/sigma/theta. Builds a tiny sphere, then reads
    `_gds_meta/calibration_history/account_pattern/v=1.json` directly off
    disk to confirm the field is populated and the persisted value matches
    what `_resolve_count_above_thresholds` would compute on the same sidecar.
    Closes the build → write → read wiring gap that round-trip serializer
    tests cannot catch."""
    out = tmp_path / "gds_persist"
    _make_sphere(out, with_aggregations=True).build()

    epoch_path = (
        out / "_gds_meta" / "calibration_history" / "account_pattern"
        / "v=1.json"
    )
    assert epoch_path.exists(), (
        f"calibration epoch JSON missing at {epoch_path}"
    )
    blob = json.loads(epoch_path.read_text())

    assert "edge_dim_thresholds" in blob, (
        "edge_dim_thresholds key missing from epoch JSON — builder did not "
        "persist thresholds even though pattern declares edge_dim_aggregations:"
    )
    thr = blob["edge_dim_thresholds"]
    assert thr is not None
    assert "pair_edge_count" in thr, (
        f"expected pair_edge_count in thresholds, got keys: {list(thr)}"
    )
    # Population p95 of pair_edge_count over the 6-edge sidecar — counts are
    # small (each pair has 1-3 edges), so threshold is finite and >= 0.
    pec_thr = thr["pair_edge_count"]
    assert isinstance(pec_thr, (int, float))
    assert pec_thr >= 0.0
    # NaN/Inf guard: threshold MUST be finite — degenerate sidecars are
    # mapped to 0.0 by `_resolve_count_above_thresholds`.
    import math
    assert math.isfinite(pec_thr)


def test_edge_dim_thresholds_absent_when_pattern_has_no_aggregations(
    tmp_path: Path,
):
    """Anchor patterns built without `edge_dim_aggregations:` must NOT carry
    `edge_dim_thresholds` in their calibration epoch JSON — `None` after
    deserialization, omitted on disk by the serializer."""
    out = tmp_path / "gds_no_agg"
    _make_sphere(out, with_aggregations=False).build()

    epoch_path = (
        out / "_gds_meta" / "calibration_history" / "account_pattern"
        / "v=1.json"
    )
    assert epoch_path.exists()
    blob = json.loads(epoch_path.read_text())
    # Either key absent OR explicitly null — both shapes deserialize as None.
    assert blob.get("edge_dim_thresholds") is None


def _make_sphere_with_per_dim_aggregates(
    out_root: Path,
    *,
    aggregates_per_dim: dict[str, tuple[str, ...]],
) -> GDSBuilder:
    """Variant of `_make_sphere(with_aggregations=True)` that drives the
    per-dim aggregate-subset selector via `aggregates_per_dim`."""
    b = GDSBuilder("test_aggsel", str(out_root))
    b.add_line(
        "accounts",
        [
            {"acct_id": "A", "name": "alpha"},
            {"acct_id": "B", "name": "beta"},
            {"acct_id": "C", "name": "gamma"},
            {"acct_id": "D", "name": "delta"},
        ],
        key_col="acct_id", source_id="t",
    )
    b.add_line(
        "transactions",
        [
            {"tx_id": "ek1", "from_acct": "A", "to_acct": "B",
             "ts": 0.0,    "amount": 20000.0},
            {"tx_id": "ek2", "from_acct": "B", "to_acct": "C",
             "ts": 100.0,  "amount": 5000.0},
            {"tx_id": "ek3", "from_acct": "C", "to_acct": "D",
             "ts": 200.0,  "amount": 5000.0},
            {"tx_id": "ek4", "from_acct": "A", "to_acct": "B",
             "ts": 1000.0, "amount": 1000.0},
        ],
        key_col="tx_id", source_id="t",
    )

    edge_dims = EdgeDimensionsConfig(dims={
        "pair_edge_count": {},
        "find_motif_structuring": {
            "time_window_hours": 1.0,
            "amt1_min": 10000.0,
            "amt2_max": 10000.0,
        },
    })

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

    b.add_pattern(
        "account_pattern",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[],
        edge_dim_aggregations=EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern",
            dims=tuple(aggregates_per_dim.keys()),
            aggregates_per_dim=aggregates_per_dim,
        ),
    )
    # Anchor needs at least one base dimension before aggregates can extend it.
    b.add_derived_dimension(
        anchor_line="accounts",
        event_line="transactions",
        anchor_fk="from_acct",
        metric="count",
        metric_col=None,
        dimension_name="tx_out_count",
        edge_max="auto",
    )
    return b


def test_per_dim_subset_emits_only_selected_aggregate_columns(tmp_path: Path):
    """`aggregates_per_dim` drives polygon-dim count: source dims × user-
    selected aggregates produces exactly that many extra columns, instead
    of source-dims × 5. Closes the polygon-dim balloon for narrow-intent
    users."""
    out = tmp_path / "gds_subset"
    _make_sphere_with_per_dim_aggregates(
        out,
        aggregates_per_dim={
            "pair_edge_count": ("count_above_threshold",),
            "find_motif_structuring": ("mean", "max"),
        },
    ).build()

    sphere = GDSReader(str(out)).read_sphere()
    pat = sphere.patterns["account_pattern"]
    kinds = pat.dimension_kinds or []
    # 1 derived (tx_out_count) + (1 + 2) per-dim aggregates = 4 dims
    assert len(kinds) == 4, (
        f"expected 4 polygon dims (1 derived + 3 selected aggregates), "
        f"got {len(kinds)}"
    )
    labels = pat.dim_labels
    assert labels[-3:] == [
        "pair_edge_count_count_above_threshold",
        "find_motif_structuring_mean",
        "find_motif_structuring_max",
    ]


def test_per_dim_subset_round_trips_through_sphere_json(tmp_path: Path):
    """The per-dim selector survives the build → sphere.json → reader path,
    so `Pattern.dim_labels` reconstructs the same shape after a cold read."""
    out = tmp_path / "gds_roundtrip"
    _make_sphere_with_per_dim_aggregates(
        out,
        aggregates_per_dim={
            "pair_edge_count": ("count_above_threshold",),
            "find_motif_structuring": ("mean", "max", "p95"),
        },
    ).build()

    sj = json.loads(
        (out / "_gds_meta" / "sphere.json").read_text(),
    )
    eda_node = sj["patterns"]["account_pattern"]["edge_dim_aggregations"]
    assert eda_node["aggregates_per_dim"] == {
        "pair_edge_count": ["count_above_threshold"],
        "find_motif_structuring": ["mean", "max", "p95"],
    }

    sphere = GDSReader(str(out)).read_sphere()
    pat = sphere.patterns["account_pattern"]
    assert pat.edge_dim_aggregations is not None
    assert pat.edge_dim_aggregations.aggregates_per_dim == {
        "pair_edge_count": ("count_above_threshold",),
        "find_motif_structuring": ("mean", "max", "p95"),
    }


def test_per_dim_subset_flips_schema_hash_vs_all_five(tmp_path: Path):
    """Central design claim: changing the per-dim selector changes
    `dimension_kinds` length, which changes the calibration `schema_hash`,
    which in turn auto-wipes `calibration_history` on the next build —
    no separate persistence-layer wiping logic needed. Two builds with
    different selectors must produce different `schema_hash` values."""
    out_full = tmp_path / "gds_full"
    _make_sphere_with_per_dim_aggregates(
        out_full,
        aggregates_per_dim={
            "pair_edge_count": ("mean", "max", "std", "p95",
                                "count_above_threshold"),
            "find_motif_structuring": ("mean", "max", "std", "p95",
                                        "count_above_threshold"),
        },
    ).build()

    out_subset = tmp_path / "gds_subset"
    _make_sphere_with_per_dim_aggregates(
        out_subset,
        aggregates_per_dim={
            "pair_edge_count": ("count_above_threshold",),
            "find_motif_structuring": ("mean", "max"),
        },
    ).build()

    sj_full = json.loads(
        (out_full / "_gds_meta" / "sphere.json").read_text(),
    )
    sj_subset = json.loads(
        (out_subset / "_gds_meta" / "sphere.json").read_text(),
    )
    hash_full = sj_full["patterns"]["account_pattern"]["schema_hash"]
    hash_subset = sj_subset["patterns"]["account_pattern"]["schema_hash"]
    assert hash_full != hash_subset, (
        f"schema_hash MUST flip when the per-dim selector subset "
        f"changes — auto-wipe of calibration_history depends on this. "
        f"Both builds produced {hash_full!r}"
    )

    # Sanity: dimension_kinds length reflects the selector difference.
    kinds_full = (
        sj_full["patterns"]["account_pattern"].get("dimension_kinds") or []
    )
    kinds_subset = (
        sj_subset["patterns"]["account_pattern"].get("dimension_kinds") or []
    )
    # full: 1 derived + 2 dims × 5 aggs = 11; subset: 1 derived + 1 + 2 = 4
    assert len(kinds_full) == 11
    assert len(kinds_subset) == 4


def test_dim_percentiles_cover_aggregated_edge_dims(tmp_path: Path):
    """Builder emits dim_percentiles entries for every aggregated edge dim,
    keyed by the canonical ``{source_dim}_{aggregate}`` label from
    ``Pattern._edge_dim_aggregation_names``. Pre-fix the cache covered
    only event_dims / prop_cols, so percentile-keyed consumers
    (sphere_overview profiling_alerts, audit_pattern_dims thresholds)
    had no signal for the aggregation block — the new aggregated
    dimensions were invisible to those auditors.
    """
    out = tmp_path / "gds_with"
    _make_sphere(out, with_aggregations=True).build()
    sphere = GDSReader(str(out)).read_sphere()
    pat = sphere.patterns["account_pattern"]

    agg_names = pat._edge_dim_aggregation_names()
    assert agg_names, "fixture must declare at least one agg dim"
    # 1 source dim × 5 aggs (mean / max / std / p95 / count_above_threshold)
    expected_aggs = {
        "pair_edge_count_mean",
        "pair_edge_count_max",
        "pair_edge_count_std",
        "pair_edge_count_p95",
        "pair_edge_count_count_above_threshold",
    }
    assert set(agg_names) == expected_aggs

    dp = pat.dim_percentiles or {}
    missing = set(agg_names) - set(dp)
    assert not missing, (
        f"dim_percentiles must contain an entry for every aggregated edge "
        f"dim; missing: {sorted(missing)}; have: {sorted(dp)}"
    )

    # Schema parity with event_dims / prop_cols path: same six keys per
    # entry. Any drift here breaks consumers that read the cache uniformly.
    expected_keys = {"min", "p25", "p50", "p75", "p99", "max"}
    for label in agg_names:
        assert set(dp[label]) == expected_keys, (
            f"dim_percentiles[{label!r}] schema {set(dp[label])} != "
            f"expected {expected_keys}"
        )
        # Percentile invariants: monotone non-decreasing, finite.
        entry = dp[label]
        for k in ("min", "p25", "p50", "p75", "p99", "max"):
            assert entry[k] == entry[k], f"NaN at {label!r}[{k!r}]"
        assert entry["min"] <= entry["p50"] <= entry["max"], (
            f"non-monotone percentiles for {label!r}: {entry}"
        )


def test_dim_percentiles_absent_for_anchor_without_aggregations(
    tmp_path: Path,
):
    """Discriminator: when the anchor has no edge_dim_aggregations
    block, no agg-label entries leak into ``dim_percentiles``. Catches a
    bug class where the new code path would always run regardless of
    whether the pattern declares aggregations.
    """
    out = tmp_path / "gds_baseline"
    _make_sphere(out, with_aggregations=False).build()
    sphere = GDSReader(str(out)).read_sphere()
    pat = sphere.patterns["account_pattern"]

    assert pat._edge_dim_aggregation_names() == []
    dp = pat.dim_percentiles or {}
    # No "<dim>_<agg>" entries should appear when no aggregations declared.
    agg_suffixes = ("_mean", "_max", "_std", "_p95", "_count_above_threshold")
    leaked = [k for k in dp if any(k.endswith(s) for s in agg_suffixes)]
    # tx_out_count is a derived dim with edge_max — it gets a percentile entry
    # but doesn't end in any of the agg suffixes, so the filter is clean.
    assert not leaked, (
        f"aggregation labels leaked into dim_percentiles for an anchor "
        f"without aggregations: {leaked}"
    )
