"""Navigator-level integration for find_motif_by_hops.

Uses the bundled AML HI-small sphere (event tx_pattern with edge_table +
edge_dimensions sidecar) since Berka's tx_pattern lacks an explicit
edge_table block.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypertopos import HopPredicate, HyperSphere


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AML_PATH = (
    PROJECT_ROOT / "benchmark" / "ibm-aml" / "hi_small_sphere"
    / "gds_aml_hi_small"
)


def _aml_sphere_unavailable() -> bool:
    sphere_json = AML_PATH / "_gds_meta" / "sphere.json"
    if not sphere_json.exists():
        return True
    import json
    return json.loads(sphere_json.read_text()).get("format_version") != "3.0"


pytestmark = pytest.mark.skipif(
    _aml_sphere_unavailable(),
    reason="AML HI-small sphere not built at format 3.0",
)


@pytest.fixture(scope="module")
def aml_nav():
    hs = HyperSphere.open(AML_PATH)
    return hs.session("hops-test").navigator()


def test_returns_dict_with_motifs_block(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=5,
    )
    assert isinstance(out, dict)
    assert "motifs" in out
    assert "n_results" in out
    assert out["pattern_id"] == "tx_pattern"


def test_anomaly_seed_filter_prunes_frontier(aml_nav):
    """anomaly_seed_filter=True must intersect the BFS frontier with
    the anomaly subset of the resolved anchor companion. Verifies the
    seed_filter_summary diagnostics surface the prune count."""
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate()],
        anomaly_seed_filter=True,
        max_results=5,
    )
    assert "seed_filter_summary" in out
    summary = out["seed_filter_summary"]
    assert summary["requested"] is None
    assert summary["anomaly"] > 0
    assert summary["filtered"] > 0
    assert summary["filtered"] <= summary["anomaly"]
    # Verify the Lance filter actually filtered — anomaly count must be
    # a proper SUBSET of all anchor entities, not the full population.
    # AML HI-small has ~5 % anomaly rate on account_pattern, so anomaly
    # count must be well under the total row count. If the
    # `is_anomaly = true` filter silently failed open (Lance returning
    # all rows), summary["anomaly"] would equal the full anchor row
    # count and this assertion would catch it.
    total_anchor_rows = aml_nav._storage.read_geometry(
        "account_pattern",
        aml_nav._resolve_version("account_pattern"),
        columns=["primary_key"],
    ).num_rows
    assert summary["anomaly"] < total_anchor_rows * 0.5, (
        f"anomaly subset {summary['anomaly']} not meaningfully smaller "
        f"than total {total_anchor_rows} — filter may have failed open"
    )


def test_anomaly_seed_filter_intersects_explicit_seeds(aml_nav):
    """When seed_keys is provided AND anomaly_seed_filter=True, result
    is the intersection — both an explicit seed list and the anomaly
    subset must contain the key."""
    geo = aml_nav._storage.read_geometry(
        "account_pattern",
        aml_nav._resolve_version("account_pattern"),
        columns=["primary_key"],
    )
    seeds = geo["primary_key"][:50].to_pylist()
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate()],
        seed_keys=seeds,
        anomaly_seed_filter=True,
        max_results=5,
    )
    summary = out["seed_filter_summary"]
    assert summary["requested"] == 50
    assert summary["filtered"] <= 50
    assert summary["filtered"] <= summary["anomaly"]


def test_anomaly_seed_filter_omitted_no_summary(aml_nav):
    """seed_filter_summary key must be ABSENT when anomaly_seed_filter
    is False (default), preserving prior return shape."""
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate()],
        max_results=3,
    )
    assert "seed_filter_summary" not in out


def test_large_seed_list_does_not_trigger_full_edge_read(aml_nav):
    """Regression test: a 1000+ seed_keys query must not re-read the
    edge table per call.

    Prior to dropping the per-call scoped adjacency build, this case ran
    the full edge table read on every call (~58s on AML HI-small). After
    routing through the cached global AdjacencyIndex with seed-frontier
    filter in BFS, warm calls amortise to BFS-only cost. Asserts a very
    lenient wall-clock budget (5s) that catches a regression to the
    scoped path without being flaky on slow CI machines.
    """
    import time

    # Fetch 1000 anchor primary_keys from account_pattern as seeds.
    # Module-scope nav fixture means the adjacency cache is warm by the
    # time this test runs (ordering-independent — all tests share it).
    geo = aml_nav._storage.read_geometry(
        "account_pattern",
        aml_nav._resolve_version("account_pattern"),
        columns=["primary_key"],
    )
    seeds = geo["primary_key"][:1000].to_pylist()

    # First call may pay cold adjacency cost (~28s on this sphere); not
    # asserted. Second call must be warm and fast.
    aml_nav.find_motif_by_hops(
        "tx_pattern", hops=[HopPredicate()],
        seed_keys=seeds, max_results=10,
    )
    t0 = time.perf_counter()
    out = aml_nav.find_motif_by_hops(
        "tx_pattern", hops=[HopPredicate()],
        seed_keys=seeds, max_results=10,
    )
    elapsed = time.perf_counter() - t0
    assert isinstance(out, dict)
    assert "n_results" in out
    assert elapsed < 5.0, (
        f"warm-cache call with 1000 seed_keys took {elapsed:.2f}s — "
        f"regression toward per-call edge-table read suspected"
    )


def test_anchor_pattern_raises(aml_nav):
    with pytest.raises(Exception, match="event pattern"):
        aml_nav.find_motif_by_hops(
            "account_pattern", hops=[HopPredicate()],
        )


def test_unknown_pattern_raises(aml_nav):
    with pytest.raises(Exception, match="pattern not found"):
        aml_nav.find_motif_by_hops(
            "nonexistent", hops=[HopPredicate()],
        )


def test_empty_hops_raises(aml_nav):
    with pytest.raises(Exception, match="hops"):
        aml_nav.find_motif_by_hops("tx_pattern", hops=[])


def test_too_many_hops_raises(aml_nav):
    with pytest.raises(Exception, match="hop count"):
        aml_nav.find_motif_by_hops(
            "tx_pattern", hops=[HopPredicate()] * 9,
        )


def test_invalid_max_results_raises(aml_nav):
    with pytest.raises(Exception, match="max_results"):
        aml_nav.find_motif_by_hops(
            "tx_pattern", hops=[HopPredicate()], max_results=0,
        )


def test_edge_dim_predicate_filter_runs(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(
            amount_min=10000.0,
            edge_dim_predicates={"pair_edge_count": (">=", 5.0)},
        )],
        max_results=3,
        score=False,
    )
    assert isinstance(out["motifs"], list)
    for m in out["motifs"]:
        assert m["dim_values_per_hop"][0]["pair_edge_count"] >= 5.0


def test_score_false_omits_score_field(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=3,
        score=False,
    )
    if out["motifs"]:
        assert "score" not in out["motifs"][0]


def test_score_true_returns_score_field(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=5,
        score=True,
    )
    assert out["motifs"], "AML HI-small must return at least one motif"
    for m in out["motifs"]:
        assert "score" in m, f"motif missing score field: {m}"
        assert "score_breakdown" in m
        assert "anchor_pattern_id" in m
        assert isinstance(m["score"], float)
        assert isinstance(m["score_breakdown"], list)
        assert isinstance(m["anchor_pattern_id"], str)


def test_score_true_sorts_by_score_desc(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=10,
        score=True,
    )
    scores = [m["score"] for m in out["motifs"] if "score" in m]
    assert scores == sorted(scores, reverse=True), (
        f"motifs must be sorted descending on score, got {scores}"
    )


def test_anchor_pattern_id_matches_resolved_companion(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=3,
        score=True,
    )
    expected = aml_nav._resolve_anchor_pattern_for_scoring("tx_pattern")
    assert expected is not None and expected != "tx_pattern"
    for m in out["motifs"]:
        if "anchor_pattern_id" in m:
            assert m["anchor_pattern_id"] == expected


def test_score_false_default_no_score_fields(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=3,
        score=False,
    )
    for m in out["motifs"]:
        assert "score" not in m
        assert "score_breakdown" not in m
        assert "anchor_pattern_id" not in m


def test_score_true_no_anchor_companion_raises(aml_nav, monkeypatch):
    monkeypatch.setattr(
        aml_nav,
        "_resolve_anchor_pattern_for_scoring",
        lambda pid: None,
    )
    with pytest.raises(Exception, match="anchor pattern"):
        aml_nav.find_motif_by_hops(
            "tx_pattern",
            hops=[HopPredicate(amount_min=10000.0)],
            max_results=3,
            score=True,
        )


def test_score_event_aware_distinguishes_same_node_pair(aml_nav):
    """Discriminator: two motifs sharing the SAME node-pair sequence but
    DIFFERENT event_keys must produce DIFFERENT scores under event-aware
    scoring. Pre-kernel-fix this collapsed because edge_potential
    aggregated only by (u, v) — distinct events between the same pair
    produced identical structural scores.
    """
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0), HopPredicate(amount_min=10000.0)],
        max_results=20,
        score=True,
    )
    motifs = out["motifs"]
    by_node_seq: dict[tuple[str, ...], list[dict]] = {}
    for m in motifs:
        if "score" not in m:
            continue
        key = tuple(m["nodes"])
        by_node_seq.setdefault(key, []).append(m)

    same_seq_groups = [g for g in by_node_seq.values() if len(g) >= 2]
    assert same_seq_groups, (
        "AML 2-hop max_results=20 must contain at least one group of "
        "motifs sharing the same node sequence (multi-event between same "
        "account pair)"
    )
    for group in same_seq_groups:
        edges_per_motif = {tuple(m["edges"]) for m in group}
        assert len(edges_per_motif) == len(group), (
            f"Within-group event_keys must be distinct; got {edges_per_motif}"
        )
        scores = {round(m["score"], 6) for m in group}
        assert len(scores) >= 2, (
            f"Same node sequence {group[0]['nodes']} produced collapsed "
            f"scores {scores} across {len(group)} distinct events — "
            f"event-aware scoring kernel did not break the tie"
        )


def test_score_breakdown_carries_event_factor(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=3,
        score=True,
    )
    for m in out["motifs"]:
        if "score_breakdown" not in m:
            continue
        for entry in m["score_breakdown"]:
            assert "event_factor" in entry, (
                f"event-aware breakdown must carry event_factor field; "
                f"got {entry}"
            )
            assert entry["event_factor"] >= 1.0, (
                f"event_factor must be >= 1.0 (centroid event = 1.0); "
                f"got {entry['event_factor']}"
            )


def test_score_event_aware_missing_event_polygon_fallback(aml_nav, monkeypatch):
    """When the event pattern's batch_read_deltas returns empty (e.g.
    sphere with edge_table but no per-event polygons written), each
    edge's event_factor must fall back to 1.0 — equivalent to legacy
    node-pair-only scoring — without raising."""
    real_batch_read = aml_nav._batch_read_deltas

    def selective_empty(pattern_id, version, keys):
        if pattern_id == "tx_pattern":
            return {}
        return real_batch_read(pattern_id, version, keys)

    monkeypatch.setattr(aml_nav, "_batch_read_deltas", selective_empty)

    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=3,
        score=True,
    )
    for m in out["motifs"]:
        if "score_breakdown" not in m:
            continue
        for entry in m["score_breakdown"]:
            assert entry["event_factor"] == 1.0, (
                f"missing event polygon must yield neutral factor 1.0; "
                f"got {entry['event_factor']}"
            )


def test_score_legacy_signature_backward_compat(aml_nav):
    """When _score_motif_from_edges is called WITHOUT event_keys (legacy
    callers like score_motif), output must NOT carry event_factor and
    behavior must match the pre-kernel-fix node-pair-only formula."""
    edges = [("ACCT-A", "ACCT-B"), ("ACCT-B", "ACCT-C")]
    delta_map = {
        "ACCT-A": __import__("numpy").array([1.0, 0.0], dtype=__import__("numpy").float32),
        "ACCT-B": __import__("numpy").array([0.0, 1.0], dtype=__import__("numpy").float32),
        "ACCT-C": __import__("numpy").array([1.0, 1.0], dtype=__import__("numpy").float32),
    }
    pair_counts = {("ACCT-A", "ACCT-B"): 5, ("ACCT-B", "ACCT-C"): 5}
    sc = aml_nav._lean_score_motif(edges, delta_map, pair_counts)
    assert sc is not None
    for entry in sc["breakdown"]:
        assert "event_factor" not in entry


def test_score_per_motif_failure_silent_skip(aml_nav, monkeypatch):
    real_lean = aml_nav._lean_score_motif
    call_count = {"n": 0}

    def flaky(edges, delta_map, pair_counts, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None  # signals endpoint missing — caller silent-skips
        return real_lean(edges, delta_map, pair_counts, **kwargs)

    monkeypatch.setattr(aml_nav, "_lean_score_motif", flaky)
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=5,
        score=True,
    )
    if len(out["motifs"]) >= 2:
        unscored = [m for m in out["motifs"] if "score" not in m]
        scored = [m for m in out["motifs"] if "score" in m]
        assert len(unscored) >= 1
        assert len(scored) >= 1
        for m in unscored:
            assert "score_breakdown" not in m
            assert "anchor_pattern_id" not in m
        for m in scored:
            assert "score_breakdown" in m
            assert "anchor_pattern_id" in m


def test_require_anomalous_default_no_filter(aml_nav):
    """Default require_anomalous_entity=False is a no-op — pre-F4 result
    shape and motif count match."""
    base = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=5,
        score=False,
    )
    explicit = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(
            amount_min=10000.0, require_anomalous_entity=False,
        )],
        max_results=5,
        score=False,
    )
    assert base["n_results"] == explicit["n_results"]
    assert [m["nodes"] for m in base["motifs"]] == [
        m["nodes"] for m in explicit["motifs"]
    ]


def test_require_anomalous_filters_destination(aml_nav):
    """With require_anomalous_entity=True on hop[0], every returned
    motif's nodes[1] must be is_anomaly=True in the anchor companion."""
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(
            amount_min=10000.0, require_anomalous_entity=True,
        )],
        max_results=20,
        score=False,
    )
    if not out["motifs"]:
        return  # acceptable if AML happens to have no qualifying motif
    anchor_pid = aml_nav._resolve_anchor_pattern_for_scoring("tx_pattern")
    version = aml_nav._resolve_version(anchor_pid)
    keys = list({m["nodes"][1] for m in out["motifs"]})
    geo = aml_nav._storage.read_geometry(
        anchor_pid, version, point_keys=keys,
        columns=["primary_key", "is_anomaly"],
    )
    is_anomaly = {
        geo["primary_key"][i].as_py(): bool(geo["is_anomaly"][i].as_py())
        for i in range(geo.num_rows)
    }
    for m in out["motifs"]:
        assert is_anomaly.get(m["nodes"][1], False), (
            f"motif destination {m['nodes'][1]} not anomalous in anchor"
        )


def test_require_anomalous_per_hop_independence(aml_nav):
    """hop[0] flag set, hop[1] not — filter applies to nodes[1] only."""
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[
            HopPredicate(amount_min=10000.0, require_anomalous_entity=True),
            HopPredicate(amount_min=10000.0, require_anomalous_entity=False),
        ],
        max_results=10,
        score=False,
    )
    if not out["motifs"]:
        return
    anchor_pid = aml_nav._resolve_anchor_pattern_for_scoring("tx_pattern")
    version = aml_nav._resolve_version(anchor_pid)
    dest_keys = list({m["nodes"][1] for m in out["motifs"]})
    geo = aml_nav._storage.read_geometry(
        anchor_pid, version, point_keys=dest_keys,
        columns=["primary_key", "is_anomaly"],
    )
    is_anomaly = {
        geo["primary_key"][i].as_py(): bool(geo["is_anomaly"][i].as_py())
        for i in range(geo.num_rows)
    }
    for m in out["motifs"]:
        assert is_anomaly.get(m["nodes"][1], False), (
            f"hop[0] flag must enforce nodes[1] anomalous"
        )


def test_require_anomalous_no_anchor_companion_raises(aml_nav, monkeypatch):
    monkeypatch.setattr(
        aml_nav, "_resolve_anchor_pattern_for_scoring", lambda pid: None,
    )
    with pytest.raises(Exception, match="anchor pattern"):
        aml_nav.find_motif_by_hops(
            "tx_pattern",
            hops=[HopPredicate(
                amount_min=10000.0, require_anomalous_entity=True,
            )],
            max_results=3,
            score=False,
        )


def test_require_anomalous_anchor_missing_is_anomaly_column_raises(
    aml_nav, monkeypatch,
):
    """Anchor pattern without is_anomaly column → raise GDSNavigationError."""
    real_read = aml_nav._storage.read_geometry

    def stripped_read(pattern_id, version, *args, **kwargs):
        if "is_anomaly" in (kwargs.get("columns") or ()):
            from pyarrow import Table
            return Table.from_pydict({"primary_key": []})
        return real_read(pattern_id, version, *args, **kwargs)

    monkeypatch.setattr(aml_nav._storage, "read_geometry", stripped_read)
    with pytest.raises(Exception, match="is_anomaly"):
        aml_nav.find_motif_by_hops(
            "tx_pattern",
            hops=[HopPredicate(
                amount_min=10000.0, require_anomalous_entity=True,
            )],
            max_results=3,
            score=False,
        )


def test_require_anomalous_dest_missing_in_geometry_dropped(
    aml_nav, monkeypatch,
):
    """If anchor geometry returns no row for a destination key, treat as
    not anomalous and drop the motif."""
    real_read = aml_nav._storage.read_geometry

    def empty_geo_read(pattern_id, version, *args, **kwargs):
        if "is_anomaly" in (kwargs.get("columns") or ()):
            from pyarrow import Table
            return Table.from_pydict({"primary_key": [], "is_anomaly": []})
        return real_read(pattern_id, version, *args, **kwargs)

    monkeypatch.setattr(aml_nav._storage, "read_geometry", empty_geo_read)
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(
            amount_min=10000.0, require_anomalous_entity=True,
        )],
        max_results=5,
        score=False,
    )
    assert out["n_results"] == 0


def test_unknown_dim_raises(aml_nav):
    with pytest.raises(Exception, match="unknown dims"):
        aml_nav.find_motif_by_hops(
            "tx_pattern",
            hops=[HopPredicate(
                edge_dim_predicates={"nonexistent_dim": (">=", 1.0)},
            )],
        )
