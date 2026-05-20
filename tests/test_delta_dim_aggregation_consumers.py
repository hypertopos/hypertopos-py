"""Regression tests for the shape-mismatch crash family when a Pattern has
delta-vector widths that differ between calibration arrays (mu/sigma/theta)
and downstream consumer assumptions.

Three distinct mismatches are covered, each crashing one or more navigator
methods on real spheres:

  Class A — π7_attract_hub: when ``edge_dim_aggregations`` is declared,
    ``pattern.mu`` / ``sigma_diag`` extend to ``delta_dim`` but
    ``pattern.edge_max`` stays at ``len(relations)`` (aggregation dims
    have no edge_max). The shape-matrix × edge_max multiply broadcasts
    ``(N, delta_dim) * (n_rel,)`` and raises.

  Class B — build_solid / π9_attract_drift / detect_trajectory_anomaly:
    the temporal layer's ``shape_snapshot`` carries structural dims only
    (no aggregation history), but ``pattern.mu`` is the full delta_dim.
    ``shape - pattern.mu`` raises a broadcast error.

  Class C — anomaly_summary: when ``pattern.dim_labels`` undercounts the
    on-disk delta width (event patterns with stored mu wider than
    ``delta_dim()``), the ``dim_sq_totals`` accumulator sized by
    ``len(labels)`` cannot absorb the wider per-cluster ``sq`` vector.

Each test engineers a fixture that reproduces exactly one of the three
mismatches so a future regression is caught at the unit level instead of
only at integration-time on a real sphere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.model.sphere import (
    EdgeDimAggregationsRef,
    Pattern,
    RelationDef,
    Sphere,
)
from hypertopos.navigation.navigator import GDSNavigator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anchor_with_aggregations(
    n_relations: int = 3,
    n_source_dims: int = 2,
    aggregates: tuple[str, ...] = (
        "mean", "max", "std", "p95", "count_above_threshold",
    ),
) -> Pattern:
    """Build an anchor pattern with edge_dim_aggregations.

    Width budget:
      - relations: n_relations
      - edge_dim aggregations: n_source_dims × len(aggregates)
      - total delta_dim: n_relations + n_source_dims * len(aggregates)

    mu/sigma/theta are sized to delta_dim (full width including aggregations).
    edge_max stays at n_relations (relations-only, the real-sphere invariant).
    """
    n_agg = n_source_dims * len(aggregates)
    delta_dim = n_relations + n_agg
    relations = [
        RelationDef(line_id=f"line_{i}", direction="out", required=False)
        for i in range(n_relations)
    ]
    return Pattern(
        pattern_id="anchor_with_aggs",
        entity_type="entities",
        pattern_type="anchor",
        relations=relations,
        mu=np.zeros(delta_dim, dtype=np.float32),
        sigma_diag=np.ones(delta_dim, dtype=np.float32),
        theta=np.ones(delta_dim, dtype=np.float32),
        edge_max=np.ones(n_relations, dtype=np.float32) * 10.0,
        population_size=100,
        computed_at=datetime.now(UTC),
        version=1,
        status="production",
        edge_dim_aggregations=EdgeDimAggregationsRef(
            from_event_pattern="ev_pattern",
            dims=tuple(f"src_{j}" for j in range(n_source_dims)),
            aggregates_per_dim={
                f"src_{j}": aggregates for j in range(n_source_dims)
            },
        ),
    )


def _make_event_with_undercount(
    n_relations: int = 4,
    n_event_dims: int = 8,
    n_stored_extra: int = 5,
) -> Pattern:
    """Build an event pattern that mimics tx_pattern's storage undercount.

    On real spheres `tx_pattern` carries 17-wide mu/sigma/theta on disk but
    ``delta_dim()`` returns 4 + 8 + 0 = 12. The five extra stored dims have
    no counterpart in ``dim_labels``. This fixture reproduces the same gap
    by sizing mu/sigma/theta wider than the labelled-dim count.
    """
    stored_width = n_relations + n_event_dims + n_stored_extra
    relations = [
        RelationDef(line_id=f"rel_{i}", direction="in", required=False)
        for i in range(n_relations)
    ]
    from hypertopos.model.sphere import EventDimDef
    event_dimensions = [
        EventDimDef(column=f"col_{j}", edge_max=1.0)
        for j in range(n_event_dims)
    ]
    return Pattern(
        pattern_id="event_undercount",
        entity_type="events",
        pattern_type="event",
        relations=relations,
        mu=np.zeros(stored_width, dtype=np.float32),
        sigma_diag=np.ones(stored_width, dtype=np.float32),
        theta=np.ones(stored_width, dtype=np.float32) * 2.0,
        edge_max=np.ones(n_relations + n_event_dims, dtype=np.float32),
        population_size=100,
        computed_at=datetime.now(UTC),
        version=1,
        status="production",
        event_dimensions=event_dimensions,
    )


def _make_navigator_with_sphere(pattern: Pattern) -> GDSNavigator:
    """Wrap a single Pattern in a mocked sphere/storage and return a
    GDSNavigator. Geometry / temporal reads are stubbed per-test."""
    sphere = MagicMock(spec=Sphere)
    sphere.patterns = {pattern.pattern_id: pattern}
    sphere.lines = {}
    sphere.aliases = {}
    sphere.entity_line = MagicMock(return_value=None)

    storage = MagicMock()
    storage.read_sphere = MagicMock(return_value=sphere)
    storage.read_geometry_stats = MagicMock(return_value=None)

    manifest = MagicMock()
    manifest.pattern_version = MagicMock(return_value=pattern.version)
    manifest.agent_id = "test"
    manifest.line_version = MagicMock(return_value=1)

    nav = GDSNavigator(MagicMock(), storage, manifest, MagicMock())
    return nav


# ---------------------------------------------------------------------------
# Class A — π7_attract_hub broadcast against edge_max
# ---------------------------------------------------------------------------


def test_attract_hub_with_aggregation_dims_does_not_broadcast_crash():
    """π7_attract_hub must slice shape_matrix to edge_max width before the
    per-relation multiply. Aggregation dims are not edges; they have no
    edge_max counterpart on real spheres."""
    pat = _make_anchor_with_aggregations(n_relations=3, n_source_dims=2)
    nav = _make_navigator_with_sphere(pat)

    # Build a geometry table with the full delta_dim (= 13) width
    delta_dim = pat.delta_dim()
    assert delta_dim == 13
    assert len(pat.edge_max) == 3  # relations only

    # 5 rows of full-width deltas — values irrelevant to the broadcast check
    rng = np.random.default_rng(0)
    deltas = rng.standard_normal((5, delta_dim)).astype(np.float32) * 0.1
    table = pa.table({
        "primary_key": [f"e{i}" for i in range(5)],
        "delta": pa.array(
            [d.tolist() for d in deltas],
            type=pa.list_(pa.float32()),
        ),
    })
    nav._storage.read_geometry = MagicMock(return_value=table)

    # Must not raise ValueError("operands could not be broadcast together")
    results = nav.π7_attract_hub(pattern_id="anchor_with_aggs", top_n=3)
    assert len(results) == 3
    # Each result is (primary_key, alive_edge_count, hub_score)
    for pk, alive, score in results:
        assert isinstance(pk, str)
        assert isinstance(alive, int)
        assert isinstance(score, float)


def test_attract_hub_and_stats_full_pattern_with_aggregation_dims_does_not_broadcast_crash():
    """π7_attract_hub_and_stats (full-pattern path) routes through
    _compute_hub_scores + the inline full-population stats block. Both
    must slice shape_matrix to edge_max width before the per-relation
    multiply when the pattern carries edge_dim_aggregations (delta_dim >
    len(relations))."""
    pat = _make_anchor_with_aggregations(n_relations=3, n_source_dims=2)
    nav = _make_navigator_with_sphere(pat)

    delta_dim = pat.delta_dim()
    assert delta_dim == 13
    assert len(pat.edge_max) == 3  # relations only

    # max_hub_score is derived from edge_max in the and_stats path;
    # the fixture's edge_max = [10, 10, 10] → max_hub_score = 30.0.
    max_hub_score = pat.max_hub_score
    assert max_hub_score is not None and max_hub_score > 0

    rng = np.random.default_rng(3)
    deltas = rng.standard_normal((7, delta_dim)).astype(np.float32) * 0.1
    table = pa.table({
        "primary_key": [f"H{i}" for i in range(7)],
        "delta": pa.array(
            [d.tolist() for d in deltas],
            type=pa.list_(pa.float32()),
        ),
    })
    nav._storage.read_geometry = MagicMock(return_value=table)

    # Must not raise — full-pattern path (line_id_filter=None) hits
    # both _compute_hub_scores' aggregated branch and the post-loop
    # np.mean/np.percentile stats over the same scores array.
    results, stats = nav.π7_attract_hub_and_stats(
        pattern_id="anchor_with_aggs", top_n=5,
    )
    assert len(results) == 5
    # Each result is (primary_key, alive_edge_count, hub_score, hub_score_pct)
    for pk, alive, score, pct in results:
        assert isinstance(pk, str)
        assert isinstance(alive, int)
        assert isinstance(score, float)
        assert np.isfinite(score), (
            f"hub_score must be finite, got {score!r} — broadcast bug "
            f"would yield NaN or raise"
        )
        # pct is a percentage of max_hub_score (finite, non-negative)
        assert pct is not None
        assert isinstance(pct, float)
        assert np.isfinite(pct)
    # Stats must be finite numbers across the full population
    for key in ("mean", "std", "p25", "p50", "p75", "p90", "p95", "max"):
        assert np.isfinite(stats[key]), (
            f"stats[{key!r}] must be finite, got {stats[key]!r}"
        )
    assert stats["total_entities"] == 7


def test_hub_score_history_with_aggregation_dims_does_not_broadcast_crash():
    """hub_score_history slices mu/sigma/edge_max to a common ``_ew``
    width before the per-slice shape-vector multiply. Pre-fix, the inner
    ``_score`` helper did ``delta * sigma + pattern.mu`` then ``shape_vec
    * pattern.edge_max`` — width mismatch (delta_dim wide on base delta,
    relations-wide on temporal snapshots, edge_max narrower than both)
    raised a broadcast error on both code paths.

    Stubs ``engine.build_solid`` to return a Solid whose slices carry
    structural-only ``delta_snapshot`` (width = n_rel). Stubs
    ``read_geometry`` to return a base delta at the full delta_dim width.
    Both paths must run without raising and produce one history entry per
    slice plus one ``current`` entry for the base polygon.
    """
    from hypertopos.model.objects import Polygon, Solid, SolidSlice

    pat = _make_anchor_with_aggregations(n_relations=3, n_source_dims=2)
    nav = _make_navigator_with_sphere(pat)

    n_rel = len(pat.relations)
    delta_dim = pat.delta_dim()
    assert delta_dim > n_rel  # the invariant that triggers the bug

    # Two temporal slices — structural-width delta_snapshot only.
    ts1 = datetime(2026, 1, 1, tzinfo=UTC)
    ts2 = datetime(2026, 2, 1, tzinfo=UTC)
    slices = [
        SolidSlice(
            slice_index=0,
            timestamp=ts1,
            deformation_type="structural",
            delta_snapshot=np.zeros(n_rel, dtype=np.float32),
            delta_norm_snapshot=0.0,
            pattern_ver=pat.version,
            changed_property=None,
            changed_line_id=None,
            added_edge=None,
        ),
        SolidSlice(
            slice_index=1,
            timestamp=ts2,
            deformation_type="edge",
            delta_snapshot=np.array(
                [0.1, 0.2, 0.3], dtype=np.float32,
            ),
            delta_norm_snapshot=float(np.linalg.norm([0.1, 0.2, 0.3])),
            pattern_ver=pat.version,
            changed_property=None,
            changed_line_id="line_0",
            added_edge=None,
        ),
    ]
    base_polygon = Polygon(
        primary_key="E1",
        pattern_id="anchor_with_aggs",
        pattern_ver=pat.version,
        pattern_type="anchor",
        scale=1.0,
        delta=np.zeros(delta_dim, dtype=np.float32),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[],
        last_refresh_at=datetime.now(UTC),
        updated_at=datetime(2026, 3, 1, tzinfo=UTC),
    )
    solid = Solid(
        primary_key="E1",
        pattern_id="anchor_with_aggs",
        base_polygon=base_polygon,
        slices=slices,
    )
    nav._engine.build_solid = MagicMock(return_value=solid)

    # Base delta read — full delta_dim width on disk.
    base_delta = np.zeros(delta_dim, dtype=np.float32)
    base_delta[:n_rel] = [0.15, 0.25, 0.35]
    base_table = pa.table({
        "delta": pa.array(
            [base_delta.tolist()], type=pa.list_(pa.float32()),
        ),
        "delta_norm": pa.array(
            [float(np.linalg.norm(base_delta))], type=pa.float32(),
        ),
    })
    nav._storage.read_geometry = MagicMock(return_value=base_table)

    # Must not raise the (n_rel,) vs (delta_dim,) broadcast crash.
    history = nav.hub_score_history(primary_key="E1", pattern_id="anchor_with_aggs")
    # Expected: one entry per slice + one "current" entry for base polygon.
    assert len(history) == len(slices) + 1
    deformations = [h["deformation_type"] for h in history]
    assert deformations[-1] == "current"
    for entry in history:
        assert isinstance(entry["hub_score"], float)
        assert np.isfinite(entry["hub_score"]), (
            f"hub_score must be finite, got {entry['hub_score']!r}"
        )
        assert isinstance(entry["alive_edges_est"], int)
        assert entry["alive_edges_est"] >= 0


# ---------------------------------------------------------------------------
# Class B — temporal shape_snapshot width < pattern.mu width
# ---------------------------------------------------------------------------


def test_build_solid_with_aggregations_handles_narrow_shape_snapshot():
    """build_solid must slice mu/sigma to the shape_snapshot width before
    the (shape - mu) / sigma broadcast. Temporal storage carries only
    structural dims; aggregation dims have no per-slice history.

    Stubs build_polygon to isolate the temporal-loop broadcast — the
    polygon-construction path requires every geometry column and is
    covered by integration tests elsewhere.
    """
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.objects import Polygon

    pat = _make_anchor_with_aggregations(n_relations=3, n_source_dims=2)
    nav = _make_navigator_with_sphere(pat)

    n_rel = len(pat.relations)
    delta_dim = pat.delta_dim()

    # Temporal read — shape_snapshot is structural-only (width = n_rel)
    ts1 = datetime(2026, 1, 1, tzinfo=UTC)
    ts2 = datetime(2026, 2, 1, tzinfo=UTC)
    temporal_table = pa.table({
        "primary_key": ["E1", "E1"],
        "slice_index": pa.array([0, 1], type=pa.int32()),
        "timestamp": pa.array([ts1, ts2], type=pa.timestamp("us", tz="UTC")),
        "deformation_type": ["base", "edge_added"],
        "shape_snapshot": pa.array(
            [
                np.zeros(n_rel, dtype=np.float32).tolist(),
                np.ones(n_rel, dtype=np.float32).tolist(),
            ],
            type=pa.list_(pa.float32()),
        ),
        "pattern_ver": pa.array([1, 1], type=pa.int32()),
        "changed_property": [None, None],
        "changed_line_id": [None, "line_0"],
    })
    nav._storage.read_temporal = MagicMock(return_value=temporal_table)

    cache_stub = MagicMock()
    cache_stub.get_polygon = MagicMock(return_value=None)
    cache_stub.set_polygon = MagicMock(return_value=None)
    engine = GDSEngine(nav._storage, cache_stub)

    # Bypass build_polygon — the broadcast bug is in the temporal-slice
    # loop, not in polygon construction.
    stub_polygon = Polygon(
        primary_key="E1",
        pattern_id="anchor_with_aggs",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1.0,
        delta=np.zeros(delta_dim, dtype=np.float32),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[],
        last_refresh_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    engine.build_polygon = MagicMock(return_value=stub_polygon)

    # Must not raise — and slice widths must match shape_snapshot, not mu
    solid = engine.build_solid("E1", "anchor_with_aggs", nav._manifest)
    assert len(solid.slices) == 2
    for slc in solid.slices:
        assert len(slc.delta_snapshot) == n_rel, (
            f"delta_snapshot must keep structural width ({n_rel}), got "
            f"{len(slc.delta_snapshot)} — padding with zero aggregation "
            f"dims would hide the storage invariant"
        )


def test_attract_drift_with_aggregations_handles_narrow_shape_snapshot():
    """π9_attract_drift consumes temporal shape_snapshot (structural only)
    against pattern.mu (full delta_dim). Must slice mu/sigma before the
    broadcast; base_delta from geometry needs the same slice."""
    pat = _make_anchor_with_aggregations(n_relations=3, n_source_dims=2)
    nav = _make_navigator_with_sphere(pat)

    n_rel = len(pat.relations)
    delta_dim = pat.delta_dim()

    # Geometry table — full delta_dim width
    rng = np.random.default_rng(1)
    deltas = rng.standard_normal((4, delta_dim)).astype(np.float32) * 0.1
    geo_table = pa.table({
        "primary_key": [f"E{i}" for i in range(4)],
        "delta": pa.array(
            [d.tolist() for d in deltas],
            type=pa.list_(pa.float32()),
        ),
    })
    nav._storage.read_geometry = MagicMock(return_value=geo_table)

    # Temporal batch — narrow shape_snapshot
    ts_seq = [
        datetime(2026, 1, m, tzinfo=UTC) for m in (1, 2, 3, 4, 5)
    ]
    pks = ["E0"] * 5
    snapshots = [
        np.array([0.1 * i, 0.2 * i, 0.3 * i], dtype=np.float32).tolist()
        for i in range(5)
    ]
    temporal_table = pa.table({
        "primary_key": pks,
        "timestamp": pa.array(ts_seq, type=pa.timestamp("us", tz="UTC")),
        "shape_snapshot": pa.array(
            snapshots, type=pa.list_(pa.float32()),
        ),
    })
    nav._storage.read_temporal_batched = MagicMock(
        return_value=iter([temporal_table.to_batches()[0]])
    )

    # Calibration-version listing for the M3 decomposition pre-flight —
    # absence drops the optional decomposition branch but the function
    # must still complete.
    nav._storage.list_calibration_versions = MagicMock(return_value=[])

    results = nav.π9_attract_drift(
        pattern_id="anchor_with_aggs", sample_size=4, top_n=1,
    )
    assert isinstance(results, list)
    # Pre-fix this raises ValueError on the broadcast — passing is the
    # gate. If results return, every shape-snapshot derived field must
    # exist.
    if results:
        for r in results:
            assert "displacement" in r
            assert "primary_key" in r


def test_detect_trajectory_anomaly_with_aggregations_handles_narrow_snapshot():
    """detect_trajectory_anomaly reads shape_snapshot (structural width)
    and computes (shape - mu) / sigma. Must slice mu/sigma to snapshot
    width before the broadcast."""
    pat = _make_anchor_with_aggregations(n_relations=3, n_source_dims=2)
    nav = _make_navigator_with_sphere(pat)

    n_rel = len(pat.relations)

    # An arch-shaped trajectory: norms go up then down across 5 slices.
    ts_seq = [datetime(2026, 1, m, tzinfo=UTC) for m in (1, 2, 3, 4, 5)]
    norms_target = [0.1, 0.5, 1.2, 0.6, 0.15]
    snapshots = [
        (np.ones(n_rel, dtype=np.float32) * norms_target[i]).tolist()
        for i in range(5)
    ]
    temporal_table = pa.table({
        "primary_key": ["E0"] * 5,
        "timestamp": pa.array(ts_seq, type=pa.timestamp("us", tz="UTC")),
        "shape_snapshot": pa.array(
            snapshots, type=pa.list_(pa.float32()),
        ),
    })
    nav._storage.read_temporal_batched = MagicMock(
        return_value=iter([temporal_table.to_batches()[0]])
    )

    # Must not raise — even when no entity is classified as anomalous, the
    # function must return without a broadcast crash.
    results = nav.detect_trajectory_anomaly(
        pattern_id="anchor_with_aggs", sample_size=10, top_n_per_range=3,
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Class C — anomaly_summary dim_sq_totals undercount
# ---------------------------------------------------------------------------


def test_anomaly_summary_with_undercounted_dim_labels_does_not_crash():
    """anomaly_summary must size dim_sq_totals by the on-disk delta width,
    not by len(pattern.dim_labels), and must tolerate labels shorter than
    the delta. Reproduces the tx_pattern crash where stored mu/delta width
    is 17 but dim_labels reports 12."""
    pat = _make_event_with_undercount(
        n_relations=4, n_event_dims=8, n_stored_extra=5,
    )
    nav = _make_navigator_with_sphere(pat)

    stored_width = len(pat.mu)
    labels_width = len(pat.dim_labels)
    assert labels_width < stored_width, (
        "fixture must reproduce the undercount invariant"
    )

    theta_norm = float(np.linalg.norm(pat.theta))

    # Build a small geometry table with deltas at the on-disk (stored)
    # width and at least one anomalous row (delta_norm >= theta_norm).
    rng = np.random.default_rng(2)
    deltas = rng.standard_normal((6, stored_width)).astype(np.float32) * 0.1
    # Make first 2 entries clearly anomalous
    deltas[0] = np.ones(stored_width, dtype=np.float32) * 3.0
    deltas[1] = np.ones(stored_width, dtype=np.float32) * 2.5
    delta_norms = np.linalg.norm(deltas, axis=1)

    geo_table = pa.table({
        "primary_key": [f"E{i}" for i in range(6)],
        "delta": pa.array(
            [d.tolist() for d in deltas],
            type=pa.list_(pa.float32()),
        ),
        "delta_norm": pa.array(delta_norms, type=pa.float32()),
        "is_anomaly": pa.array(
            (delta_norms >= theta_norm).tolist(), type=pa.bool_(),
        ),
    })
    nav._storage.read_geometry = MagicMock(return_value=geo_table)

    # Must not raise the (12,) (17,) (12,) broadcast error
    result = nav.anomaly_summary(pattern_id="event_undercount")
    assert result["total_entities"] == 6
    assert result["total_anomalies"] >= 1
    # top_driving_dimensions can reference dim indices beyond len(labels);
    # those must fall back to a synthesized label, not crash.
    for entry in result["top_driving_dimensions"]:
        assert isinstance(entry["label"], str)
        assert entry["label"]  # not empty


# ---------------------------------------------------------------------------
# Sanity helper — fixture invariants
# ---------------------------------------------------------------------------


def test_aggregation_fixture_invariants():
    """Guard the fixtures themselves: anchor-with-aggs must have
    edge_max narrower than mu, and event-with-undercount must have
    dim_labels narrower than mu."""
    pat_a = _make_anchor_with_aggregations()
    assert len(pat_a.edge_max) < len(pat_a.mu)
    assert len(pat_a.edge_max) == len(pat_a.relations)
    assert pat_a.delta_dim() == len(pat_a.mu)

    pat_c = _make_event_with_undercount()
    assert len(pat_c.dim_labels) < len(pat_c.mu)
    assert pat_c.delta_dim() == len(pat_c.dim_labels)
