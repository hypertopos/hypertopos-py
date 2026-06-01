# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for incremental geometry updates."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from hypertopos.builder.builder import (
    GDSBuilder,
    IncrementalUpdateResult,
    compute_entity_geometry,
)
from hypertopos.storage.reader import GDSReader


@pytest.fixture(scope="module")
def base_sphere(tmp_path_factory):
    """Build a small anchor-only sphere for incremental update tests."""
    tmp = tmp_path_factory.mktemp("incremental")
    out = str(tmp / "gds_inc")

    rng = np.random.default_rng(42)
    n = 100
    custs = pa.table(
        {
            "primary_key": pa.array([f"C{i}" for i in range(n)], type=pa.string()),
            "region": pa.array(
                [rng.choice(["A", "B"]) for _ in range(n)],
                type=pa.string(),
            ),
        }
    )
    events = pa.table(
        {
            "primary_key": pa.array(
                [f"E{i}" for i in range(n * 5)],
                type=pa.string(),
            ),
            "cust_fk": pa.array(
                [f"C{rng.integers(0, n)}" for _ in range(n * 5)],
                type=pa.string(),
            ),
            "amount": pa.array(
                rng.uniform(10, 100, size=n * 5).tolist(),
                type=pa.float64(),
            ),
        }
    )

    builder = GDSBuilder("test_inc", out)
    builder.add_line("customers", custs, key_col="primary_key", source_id="test", role="anchor")
    builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")
    builder.add_derived_dimension(
        "customers",
        "events",
        "cust_fk",
        "count",
        None,
        "event_count",
    )
    builder.add_pattern(
        "cust_pattern",
        "anchor",
        "customers",
        relations=[],
    )
    builder.build()
    return out


def _copy_sphere(base_sphere, tmp_path_factory, name):
    """Clone sphere to a fresh dir so tests don't interfere."""
    from tests.conftest import clone_sphere

    tmp = tmp_path_factory.mktemp(name)
    dest = tmp / "gds_inc"
    clone_sphere(base_sphere, dest)
    return str(dest)


@pytest.fixture
def sphere_for_add(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_add")


@pytest.fixture
def sphere_for_delete(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_delete")


@pytest.fixture
def sphere_for_modify(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_modify")


@pytest.fixture
def sphere_for_drift(base_sphere, tmp_path_factory):
    return _copy_sphere(base_sphere, tmp_path_factory, "inc_drift")


def _read_sphere_json(sphere_path):
    return json.loads((Path(sphere_path) / "_gds_meta" / "sphere.json").read_text())


def _get_pattern_meta(sphere_path, pattern_id="cust_pattern"):
    sj = _read_sphere_json(sphere_path)
    return sj["patterns"][pattern_id]


def test_incremental_add_new_entities(sphere_for_add):
    """Add 10 new customers, verify geometry grows."""
    sphere_path = sphere_for_add
    old_pop = _get_pattern_meta(sphere_path)["population_size"]

    # Build entity table with event_count column (matches derived dim)
    new_custs = pa.table(
        {
            "primary_key": pa.array(
                [f"C{100 + i}" for i in range(10)],
                type=pa.string(),
            ),
            "event_count": pa.array([5.0] * 10, type=pa.float64()),
        }
    )

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update("cust_pattern", changed_entities=new_custs)

    assert isinstance(result, IncrementalUpdateResult)
    assert result.added == 10
    assert result.modified == 0
    assert result.deleted == 0
    assert result.population_size == old_pop + 10

    # Verify sphere.json was updated
    new_pop = _get_pattern_meta(sphere_path)["population_size"]
    assert new_pop == old_pop + 10


def test_incremental_delete_entities(sphere_for_delete):
    """Delete 5 entities, verify population shrinks."""
    sphere_path = sphere_for_delete
    old_pop = _get_pattern_meta(sphere_path)["population_size"]

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update(
        "cust_pattern",
        deleted_keys=["C0", "C1", "C2", "C3", "C4"],
    )

    assert result.deleted == 5
    assert result.added == 0
    assert result.modified == 0
    assert result.population_size == old_pop - 5

    # Verify sphere.json was updated
    new_pop = _get_pattern_meta(sphere_path)["population_size"]
    assert new_pop == old_pop - 5


def test_incremental_modify_entities(sphere_for_modify):
    """Modify existing entities, verify geometry updated."""
    sphere_path = sphere_for_modify
    old_pop = _get_pattern_meta(sphere_path)["population_size"]

    modified = pa.table(
        {
            "primary_key": pa.array(["C10", "C11"], type=pa.string()),
            "event_count": pa.array([999.0, 999.0], type=pa.float64()),
        }
    )

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update("cust_pattern", changed_entities=modified)

    assert result.modified == 2
    assert result.added == 0
    # Population size unchanged for modifications
    assert result.population_size == old_pop


def test_incremental_returns_drift_pct(sphere_for_drift):
    """Verify drift_pct is reported."""
    sphere_path = sphere_for_drift

    new_custs = pa.table(
        {
            "primary_key": pa.array(["C200"], type=pa.string()),
            "event_count": pa.array([5.0], type=pa.float64()),
        }
    )

    builder = GDSBuilder("test_inc", sphere_path)
    result = builder.incremental_update("cust_pattern", changed_entities=new_custs)

    assert isinstance(result.drift_pct, float)
    assert result.drift_pct >= 0.0
    assert isinstance(result.theta_norm, float)
    assert result.theta_norm > 0.0


# ── Unit tests for helpers ──


# ── Edge-derived-dim (edge_dim_aggregations) incremental support ──


def _build_agg_sphere(out_root):
    """Tiny 4-account anchor sphere whose account_pattern carries 1 derived
    relation (tx_out_count) + 5 aggregated edge dims (pair_edge_count_*),
    giving a 6-wide geometry. Mirrors the AML account_pattern shape where the
    mu/sigma width exceeds the relation count by the edge_dim_aggregations
    block."""
    from hypertopos.builder import GDSBuilder, RelationSpec
    from hypertopos.builder.builder import EdgeTableConfig
    from hypertopos.builder.mapping import (
        EdgeDimAggregationsConfig,
        EdgeDimensionsConfig,
    )

    b = GDSBuilder("test_inc_agg", str(out_root))
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
        "account_pattern",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[],
        edge_dim_aggregations=EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern",
            dims=("pair_edge_count",),
        ),
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
    b.build()
    return str(out_root)


@pytest.fixture
def agg_sphere(tmp_path):
    return _build_agg_sphere(tmp_path / "gds_inc_agg")


def test_build_refuses_tracked_properties_with_edge_blocks(tmp_path):
    """A pattern declaring BOTH tracked_properties and an edge_dimensions /
    edge_dim_aggregations block is refused at build: the geometry concat and the
    dim-label layer order these blocks differently, so per-dimension stats would
    be silently mis-attributed (dim_index resolves prop/agg labels to the wrong
    delta slot). No shipped sphere pairs the blocks — guard until the dimension
    order is unified."""
    from hypertopos.builder import GDSBuilder, RelationSpec
    from hypertopos.builder.builder import EdgeTableConfig
    from hypertopos.builder.mapping import (
        EdgeDimAggregationsConfig,
        EdgeDimensionsConfig,
    )

    b = GDSBuilder("test_guard", str(tmp_path / "gds_guard"))
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
            {"tx_id": f"ek{i}", "from_acct": "A", "to_acct": "B",
             "ts": float(i), "amount": 100.0}
            for i in range(6)
        ],
        key_col="tx_id", source_id="t",
    )
    b.add_pattern(
        "tx_pattern", pattern_type="event", entity_line="transactions",
        relations=[
            RelationSpec("accounts", fk_col="from_acct", direction="in", required=True),
        ],
        edge_table=EdgeTableConfig(
            from_col="from_acct", to_col="to_acct",
            timestamp_col="ts", amount_col="amount",
        ),
        edge_dimensions=EdgeDimensionsConfig(dims={"pair_edge_count": {}}),
    )
    b.add_pattern(
        "account_pattern", pattern_type="anchor", entity_line="accounts",
        relations=[],
        tracked_properties=["name"],  # prop block ...
        edge_dim_aggregations=EdgeDimAggregationsConfig(  # ... + edge-agg block
            from_event_pattern="tx_pattern", dims=("pair_edge_count",),
        ),
    )
    with pytest.raises(ValueError, match="tracked_properties"):
        b.build()


def test_incremental_update_edge_dim_aggregation_pattern(agg_sphere):
    """incremental_update on an anchor pattern with an edge_dim_aggregations
    block must NOT raise the (N, n_relations) (mu_width,) broadcast error.
    The new entity's geometry must be the full mu-width (1 derived relation +
    5 aggregated edge dims = 6), and the new key must surface through the
    find_anomalies scan path on a fresh HyperSphere session."""
    from hypertopos.sphere import HyperSphere

    pat_meta = _get_pattern_meta(agg_sphere, "account_pattern")
    expected_width = len(pat_meta["mu"])
    assert expected_width == 6, (
        f"fixture sanity: expected 6-wide mu (1 derived + 5 aggs), "
        f"got {expected_width}"
    )
    old_pop = pat_meta["population_size"]

    # The caller supplies precomputed feature columns — the derived relation
    # (tx_out_count) plus the 5 aggregated edge-dim columns — exactly as it
    # would for any graph feature (pagerank, betweenness). Values set extreme
    # so the new account is unambiguously anomalous.
    new_accts = pa.table(
        {
            "primary_key": pa.array(["NEW_ACCT"], type=pa.string()),
            "tx_out_count": pa.array([999.0], type=pa.float64()),
            "pair_edge_count_mean": pa.array([999.0], type=pa.float64()),
            "pair_edge_count_max": pa.array([999.0], type=pa.float64()),
            "pair_edge_count_std": pa.array([999.0], type=pa.float64()),
            "pair_edge_count_p95": pa.array([999.0], type=pa.float64()),
            "pair_edge_count_count_above_threshold": pa.array(
                [999.0], type=pa.float64(),
            ),
        }
    )

    builder = GDSBuilder("test_inc_agg", agg_sphere)
    result = builder.incremental_update(
        "account_pattern", changed_entities=new_accts, recalibrate="never",
    )

    assert result.added == 1
    assert result.population_size == old_pop + 1

    # New geometry row must be full mu-width — not truncated to relation count
    # — and flagged anomalous (its extreme feature values exceed theta).
    reader = GDSReader(agg_sphere)
    geo = reader.read_geometry("account_pattern", version=1)
    new_rows = geo.filter(pc.equal(geo["primary_key"], "NEW_ACCT"))
    assert new_rows.num_rows == 1
    assert len(new_rows["delta"][0].as_py()) == expected_width
    assert new_rows["is_anomaly"][0].as_py() is True, (
        "extreme incrementally-added account must be flagged is_anomaly=True"
    )

    # find_anomalies (π5 scan) must surface the new extreme account, and rank
    # it at the top (select=top_norm) — a population of 5 means top_n=10 would
    # return everyone, so use top_n=1 to make surfacing non-trivial.
    hs = HyperSphere.open(agg_sphere)
    nav = hs.session("inc-agg-test").navigator()
    polygons, _total, _emerging, _meta = nav.π5_attract_anomaly(
        "account_pattern", top_n=1,
    )
    surfaced = {p.primary_key for p in polygons}
    assert "NEW_ACCT" in surfaced, (
        f"new incrementally-added anomalous account not the top anomaly via "
        f"find_anomalies; got {surfaced}"
    )


def test_incremental_update_triggers_reindex(agg_sphere, monkeypatch):
    """incremental_update must call _maybe_reindex_geometry so appended rows
    can enter the IVF index. reindex=True forces a rebuild (threshold=0.0);
    the default uses the standard 10% threshold. A 4-row fixture is below the
    256-row IVF minimum, so the actual rebuild no-ops — this test asserts the
    WIRING (the call and its threshold), not a rebuilt index."""
    from hypertopos.storage.writer import GDSWriter

    calls: list[float] = []
    original = GDSWriter._maybe_reindex_geometry

    def _spy(self, pattern_id, threshold=0.1, version=1):
        calls.append(threshold)
        return original(self, pattern_id, threshold=threshold, version=version)

    monkeypatch.setattr(GDSWriter, "_maybe_reindex_geometry", _spy)

    new_accts = pa.table(
        {
            "primary_key": pa.array(["RX_ACCT"], type=pa.string()),
            "tx_out_count": pa.array([5.0], type=pa.float64()),
            "pair_edge_count_mean": pa.array([1.0], type=pa.float64()),
            "pair_edge_count_max": pa.array([1.0], type=pa.float64()),
            "pair_edge_count_std": pa.array([0.0], type=pa.float64()),
            "pair_edge_count_p95": pa.array([1.0], type=pa.float64()),
            "pair_edge_count_count_above_threshold": pa.array(
                [0.0], type=pa.float64(),
            ),
        }
    )

    builder = GDSBuilder("test_inc_agg", agg_sphere)
    builder.incremental_update(
        "account_pattern", changed_entities=new_accts,
        recalibrate="never", reindex=True,
    )

    assert calls == [0.0], (
        f"reindex=True must force a rebuild via threshold=0.0; got {calls}"
    )


def _build_anchor_sphere(out_root, n):
    """Synthetic anchor-only sphere with one derived count dimension and *n*
    customers. Large enough (n >= 256) to exercise the IVF index path."""
    rng = np.random.default_rng(7)
    custs = pa.table(
        {
            "primary_key": pa.array(
                [f"C{i}" for i in range(n)], type=pa.string(),
            ),
        }
    )
    # ~6 events per customer with skewed counts so delta_norms are diverse.
    n_events = n * 6
    events = pa.table(
        {
            "primary_key": pa.array(
                [f"E{i}" for i in range(n_events)], type=pa.string(),
            ),
            "cust_fk": pa.array(
                [f"C{int(rng.integers(0, n))}" for _ in range(n_events)],
                type=pa.string(),
            ),
        }
    )
    b = GDSBuilder("test_inc_perf", str(out_root))
    b.add_line(
        "customers", custs, key_col="primary_key",
        source_id="t", role="anchor",
    )
    b.add_line(
        "events", events, key_col="primary_key", source_id="t", role="event",
    )
    b.add_derived_dimension(
        "customers", "events", "cust_fk", "count", None, "event_count",
        edge_max="auto",
    )
    b.add_pattern("cust_pattern", "anchor", "customers", relations=[])
    b.build()
    return str(out_root)


def _read_rank(sphere_path, key, pattern_id="cust_pattern"):
    reader = GDSReader(sphere_path)
    geo = reader.read_geometry(pattern_id, version=1)
    row = geo.filter(pc.equal(geo["primary_key"], key))
    assert row.num_rows == 1
    return float(row["delta_rank_pct"][0].as_py())


def _read_conformal_p(sphere_path, key, pattern_id="cust_pattern"):
    reader = GDSReader(sphere_path)
    geo = reader.read_geometry(pattern_id, version=1)
    row = geo.filter(pc.equal(geo["primary_key"], key))
    assert row.num_rows == 1
    return float(row["conformal_p"][0].as_py())


def test_incremental_conformal_p_is_population_relative_not_batch_local(tmp_path):
    """An entity ingested via incremental_update must get a population-relative
    conformal_p (right-tail: lower = more anomalous), not the old batch-local /
    polarity-inverted value.

    Regression: incremental_update wrote conformal_p = (rank+1)/(n_new+1) over
    the ingest batch only, with left-rank polarity — so the most-anomalous
    ingested row got the LARGEST p (looked benign) and a single-entity append
    always got exactly 0.5, contradicting the documented 'lower = more
    anomalous' contract that assess_anomaly_certainty / composite_risk rely on.
    The fix recomputes conformal_p globally (recompute_conformal_p), mirroring
    delta_rank_pct."""
    sphere = _build_anchor_sphere(tmp_path / "gds_cp", 300)

    # Append ONE extreme outlier: event_count far above the population.
    outlier = pa.table({
        "primary_key": pa.array(["C_OUTLIER"], type=pa.string()),
        "event_count": pa.array([99999.0], type=pa.float64()),
    })
    b = GDSBuilder("test_inc_perf", sphere)
    b.incremental_update("cust_pattern", changed_entities=outlier, recalibrate="never")

    cp = _read_conformal_p(sphere, "C_OUTLIER")
    # Right-tail population p: an extreme outlier sits at the top of the
    # population, so very few rows have a >= norm → p near 1/N (small).
    # The old bug gave ~0.5 (single-batch (0+1)/(1+1)) or an inverted-large p.
    assert cp < 0.05, (
        f"extreme ingested outlier must get a small (population-relative) "
        f"conformal_p, got {cp} — batch-local/inverted regression"
    )
    # Sanity: it must NOT be the tell-tale single-batch 0.5.
    assert abs(cp - 0.5) > 0.01

    # And it agrees with the population oracle on the full post-ingest set.
    from hypertopos.builder._stats import compute_conformal_p
    geo = GDSReader(sphere).read_geometry("cust_pattern", version=1)
    norms = geo["delta_norm"].to_numpy(zero_copy_only=False)
    keys = geo["primary_key"].to_pylist()
    oracle = compute_conformal_p(norms)
    idx = keys.index("C_OUTLIER")
    assert abs(cp - float(oracle[idx])) < 1e-3, (
        f"stored conformal_p {cp} must match the full-population oracle "
        f"{float(oracle[idx])}"
    )


def test_recompute_rank_stats_matches_separate_recomputes(tmp_path):
    """recompute_rank_stats (single-pass) produces the same delta_rank_pct and
    conformal_p as calling recompute_delta_rank_pct + recompute_conformal_p
    separately — at half the append-path scan/merge I/O. Corrupts both columns
    first so the recompute is observably doing work, not a no-op."""
    import lance
    from pathlib import Path

    from hypertopos.storage.writer import GDSWriter

    sphere = _build_anchor_sphere(tmp_path / "gds_fuse", 300)
    extra = pa.table({
        "primary_key": pa.array([f"C_X{i}" for i in range(5)], type=pa.string()),
        "event_count": pa.array([10.0, 50.0, 99999.0, 1.0, 500.0], type=pa.float64()),
    })
    GDSBuilder("test_inc_perf", sphere).incremental_update(
        "cust_pattern", changed_entities=extra, recalibrate="never",
    )

    geo_path = str(Path(sphere) / "geometry" / "cust_pattern" / "data.lance")
    pks = lance.dataset(geo_path).to_table(columns=["primary_key"])["primary_key"]
    sentinel = pa.array([-1.0] * len(pks), type=pa.float32())

    def _corrupt():
        lance.dataset(geo_path).merge_insert(
            "primary_key"
        ).when_matched_update_all().execute(pa.table({
            "primary_key": pks,
            "delta_rank_pct": sentinel,
            "conformal_p": sentinel,
        }))

    def _read():
        geo = GDSReader(sphere).read_geometry("cust_pattern", version=1)
        return {
            r["primary_key"]: (r["delta_rank_pct"], r["conformal_p"])
            for r in geo.to_pylist()
        }

    writer = GDSWriter(str(sphere))
    _corrupt()
    writer.recompute_rank_stats("cust_pattern")
    fused = _read()
    _corrupt()
    writer.recompute_delta_rank_pct("cust_pattern")
    writer.recompute_conformal_p("cust_pattern")
    separate = _read()

    assert fused == separate
    # The fused recompute actually ran — no sentinel survived.
    assert all(rank != -1.0 and cp != -1.0 for rank, cp in fused.values())


def test_incremental_recompute_ranks_false_defers_then_finalize(tmp_path):
    """recompute_ranks=False keeps existing rows' delta_rank_pct stale during
    batched ingestion; finalize_incremental restores them to the same values
    recompute_ranks=True would produce. Proves both the skip and the
    finalize-correctness in one test (no copytree, ~300 rows)."""
    # Reference sphere: per-call recompute (always correct).
    ref = _build_anchor_sphere(tmp_path / "gds_ref", 300)
    # Deferred sphere: identical build, then recompute_ranks=False appends.
    deferred = _build_anchor_sphere(tmp_path / "gds_def", 300)

    # Two batches of new customers with EXTREME counts that reshuffle ranks:
    # being top-anomalous pushes existing high-norm rows down in percentile.
    def _batch(start):
        return pa.table(
            {
                "primary_key": pa.array(
                    [f"C{start + i}" for i in range(20)], type=pa.string(),
                ),
                "event_count": pa.array([9999.0] * 20, type=pa.float64()),
            }
        )

    rank_before = _read_rank(deferred, "C0")

    b_ref = GDSBuilder("test_inc_perf", ref)
    b_def = GDSBuilder("test_inc_perf", deferred)
    for batch_start in (300, 320):
        b_ref.incremental_update(
            "cust_pattern", changed_entities=_batch(batch_start),
            recalibrate="never",
        )
        b_def.incremental_update(
            "cust_pattern", changed_entities=_batch(batch_start),
            recalibrate="never", recompute_ranks=False,
        )

    # Mid-batch (deferred): C0's rank is stale — unchanged from the original
    # build even though the population grew by 40 high-norm rows.
    rank_deferred_mid = _read_rank(deferred, "C0")
    assert rank_deferred_mid == rank_before, (
        "recompute_ranks=False must leave existing rows' delta_rank_pct stale "
        f"until finalize; was {rank_before}, now {rank_deferred_mid}"
    )

    # The per-call-recompute reference has moved C0's rank (population shifted).
    rank_ref = _read_rank(ref, "C0")
    assert rank_ref != rank_before, (
        "reference (recompute_ranks=True) should have shifted C0's rank after "
        "40 extreme rows were added — fixture not discriminating"
    )

    # Finalize the deferred sphere → C0's rank must now match the reference.
    b_def.finalize_incremental("cust_pattern")
    rank_deferred_final = _read_rank(deferred, "C0")
    assert abs(rank_deferred_final - rank_ref) < 1e-3, (
        f"finalize_incremental must restore correct population percentile; "
        f"got {rank_deferred_final}, reference {rank_ref}"
    )


def test_incremental_update_rejects_grouped_pattern(tmp_path):
    """incremental_update must refuse a group_by_property / GMM / FDR pattern
    rather than silently recomputing its geometry against the GLOBAL (not
    per-group) mu/sigma/theta — which would write delta_norm / is_anomaly that
    disagree with build() and drop the FDR carrier columns the dataset carries."""
    import json as _json

    sphere = _build_anchor_sphere(tmp_path / "gds_grouped", 300)
    sj_path = Path(sphere) / "_gds_meta" / "sphere.json"
    sj = _json.loads(sj_path.read_text())
    sj["patterns"]["cust_pattern"]["group_by_property"] = "region"
    sj_path.write_text(_json.dumps(sj))

    extra = pa.table({
        "primary_key": pa.array(["C_NEW"], type=pa.string()),
        "event_count": pa.array([5.0], type=pa.float64()),
    })
    with pytest.raises(ValueError, match="group_by_property"):
        GDSBuilder("test_inc_grouped", sphere).incremental_update(
            "cust_pattern", changed_entities=extra, recalibrate="never",
        )


def test_incremental_reindex_covers_all_rows_after_recompute(tmp_path):
    """B-ordering regression: incremental_update(reindex=True) must leave the
    IVF index covering ALL rows. recompute_delta_rank_pct's merge_insert
    rewrites matched rows into new fragments — if reindex ran BEFORE the
    recompute, the rebuilt index would cover zero current rows. This asserts
    reindex runs AFTER recompute by checking num_rows_indexed == total."""
    import lance

    sphere = _build_anchor_sphere(tmp_path / "gds_idx", 300)

    new_custs = pa.table(
        {
            "primary_key": pa.array(
                [f"C{300 + i}" for i in range(40)], type=pa.string(),
            ),
            "event_count": pa.array([50.0] * 40, type=pa.float64()),
        }
    )
    b = GDSBuilder("test_inc_perf", sphere)
    b.incremental_update(
        "cust_pattern", changed_entities=new_custs,
        recalibrate="never", reindex=True,
    )

    lance_path = Path(sphere) / "geometry" / "cust_pattern" / "data.lance"
    ds = lance.dataset(str(lance_path))
    total = ds.count_rows()
    indexed = None
    for idx in ds.describe_indices():
        if "delta" in idx.field_names:
            indexed = idx.num_rows_indexed
            break
    assert indexed is not None, "delta vector index missing after reindex=True"
    assert indexed == total, (
        f"IVF index must cover all {total} rows after reindex; covers "
        f"{indexed}. reindex likely ran BEFORE recompute_delta_rank_pct, "
        f"whose merge_insert rewrote rows into fresh fragments."
    )


def test_incremental_default_path_indexes_new_rows_after_recompute(tmp_path):
    """Default path (reindex=False, recompute_ranks=True): the rank recompute's
    merge_insert drops the vector index, so the subsequent threshold-gated
    reindex sees 100% unindexed and rebuilds — leaving the index covering all
    rows. Guards against a regression where the default path silently leaves
    new rows outside the index (π10 would miss them)."""
    import lance

    sphere = _build_anchor_sphere(tmp_path / "gds_def_idx", 300)
    # Append > 10% of population so the threshold would matter even if the
    # index were preserved.
    new_custs = pa.table(
        {
            "primary_key": pa.array(
                [f"C{300 + i}" for i in range(40)], type=pa.string(),
            ),
            "event_count": pa.array([10.0] * 40, type=pa.float64()),
        }
    )
    b = GDSBuilder("test_inc_perf", sphere)
    b.incremental_update(
        "cust_pattern", changed_entities=new_custs, recalibrate="never",
        # defaults: reindex=False, recompute_ranks=True
    )
    lance_path = Path(sphere) / "geometry" / "cust_pattern" / "data.lance"
    ds = lance.dataset(str(lance_path))
    total = ds.count_rows()
    indexed = None
    for idx in ds.describe_indices():
        if "delta" in idx.field_names:
            indexed = idx.num_rows_indexed
            break
    assert indexed == total, (
        f"default-path incremental_update must leave all {total} rows indexed; "
        f"covers {indexed}. New rows would be invisible to ANN navigation."
    )


def test_incremental_width_guard_rejects_dim_block_pattern(tmp_path):
    """A pattern carrying a generalized dimension block (metric_properties) has
    a mu wider than the blocks compute_entity_geometry can reconstruct.
    incremental_update must raise a clear ValueError naming the width mismatch
    instead of a cryptic Lance append error."""
    n = 20
    custs = pa.table(
        {
            "primary_key": pa.array(
                [f"C{i}" for i in range(n)], type=pa.string(),
            ),
            "balance": pa.array(
                np.linspace(0.0, 1000.0, n).tolist(), type=pa.float64(),
            ),
            "age_days": pa.array(
                np.linspace(1.0, 500.0, n).tolist(), type=pa.float64(),
            ),
        }
    )
    events = pa.table(
        {
            "primary_key": pa.array(
                [f"E{i}" for i in range(n * 3)], type=pa.string(),
            ),
            "cust_fk": pa.array(
                [f"C{i % n}" for i in range(n * 3)], type=pa.string(),
            ),
        }
    )
    out = str(tmp_path / "gds_block")
    b = GDSBuilder("test_inc_block", out)
    b.add_line(
        "customers", custs, key_col="primary_key",
        source_id="t", role="anchor",
    )
    b.add_line(
        "events", events, key_col="primary_key", source_id="t", role="event",
    )
    b.add_derived_dimension(
        "customers", "events", "cust_fk", "count", None, "event_count",
        edge_max="auto",
    )
    # 1 derived dim (reconstructable) + 2 metric-block dims (NOT
    # reconstructable by incremental_update) → mu is 3-wide.
    b.add_pattern(
        "cust_pattern", "anchor", "customers", relations=[],
        metric_properties=["balance", "age_days"],
    )
    b.build()

    pat_meta = _get_pattern_meta(out, "cust_pattern")
    assert len(pat_meta["mu"]) == 3, (
        f"fixture sanity: 1 derived + 2 metric-block dims; got "
        f"{len(pat_meta['mu'])}"
    )

    new_custs = pa.table(
        {
            "primary_key": pa.array(["C100"], type=pa.string()),
            "event_count": pa.array([3.0], type=pa.float64()),
            "balance": pa.array([500.0], type=pa.float64()),
            "age_days": pa.array([250.0], type=pa.float64()),
        }
    )
    builder = GDSBuilder("test_inc_block", out)
    with pytest.raises(ValueError, match="computed .* but the pattern mu"):
        builder.incremental_update(
            "cust_pattern", changed_entities=new_custs, recalibrate="never",
        )


def test_compute_entity_geometry_basic():
    """Test compute_entity_geometry with simple relation metadata."""
    entity_table = pa.table(
        {
            "primary_key": pa.array(["A", "B", "C"], type=pa.string()),
            "fk_line1": pa.array(["X", "", "Y"], type=pa.string()),
        }
    )
    mu = np.array([0.5], dtype=np.float32)
    sigma = np.array([0.3], dtype=np.float32)
    relations = [{"line_id": "line1", "direction": "in", "fk_col": "fk_line1"}]

    deltas, norms, shapes = compute_entity_geometry(
        entity_table,
        mu,
        sigma,
        relations,
    )

    assert deltas.shape == (3, 1)
    assert norms.shape == (3,)
    assert shapes.shape == (3, 1)
    # A has FK "X" → shape 1.0, B has "" → shape 0.0, C has "Y" → shape 1.0
    assert shapes[0, 0] == 1.0
    assert shapes[1, 0] == 0.0
    assert shapes[2, 0] == 1.0


def test_compute_entity_geometry_raises_on_absent_edge_dim_agg_column():
    """A declared edge_dim_aggregation column absent from the ingested table
    must RAISE, not silently zero-fill.

    Regression for the silent-wrong-result: an absent {dim}_{agg} column left
    the shape at 0.0, and the z-score turned that into delta = (0 - mu)/sigma —
    non-zero for any aggregation whose population mean is non-zero (e.g. a
    transaction-amount mean), inflating delta_norm and able to flip is_anomaly
    with no diagnostic. The guard now fails loud instead.
    """
    entity_table = pa.table({"primary_key": pa.array(["A"], type=pa.string())})
    mu = np.array([50.0], dtype=np.float32)   # nonzero population mean
    sigma = np.array([10.0], dtype=np.float32)

    with pytest.raises(ValueError, match="declared value column"):
        compute_entity_geometry(
            entity_table, mu, sigma, [],
            edge_dim_agg_labels=["amount_mean"],
        )


def test_compute_entity_geometry_absent_edge_dim_agg_would_corrupt_geometry():
    """Pin the underlying divergence the guard prevents: WITHOUT the guard an
    absent column produces a spurious delta, whereas the same entity carrying
    the column at the population mean produces delta 0. Asserting the present-
    column path is correct (delta_norm == 0 at mu) documents what the absent
    path would otherwise have wrongly reported as anomalous (delta_norm == 5)."""
    mu = np.array([50.0], dtype=np.float32)
    sigma = np.array([10.0], dtype=np.float32)
    present = pa.table(
        {
            "primary_key": pa.array(["A"], type=pa.string()),
            "amount_mean": pa.array([50.0], type=pa.float64()),
        }
    )
    _d, norms, _s = compute_entity_geometry(
        present, mu, sigma, [], edge_dim_agg_labels=["amount_mean"],
    )
    # Entity at the population mean is NOT anomalous: delta_norm ~ 0.
    assert float(norms[0]) == pytest.approx(0.0, abs=1e-6)
