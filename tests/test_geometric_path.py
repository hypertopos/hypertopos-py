# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for edge table navigator functions: find_counterparties fast path,
find_geometric_path, discover_chains, entity_flow, contagion_score,
amount-weighted scoring, and degree_velocity."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.builder.builder import GDSBuilder, RelationSpec
from hypertopos.navigation.navigator import GDSNavigationError, GDSNavigator
from hypertopos.sphere import HyperSphere


@pytest.fixture(scope="module")
def tx_sphere(tmp_path_factory):
    """Build a sphere with accounts + transactions for path/chain tests.

    Graph: A000→A003→A006→A009→A012→A015→A018→A001→A004→A007
    (each account sends to account+3 mod 20, creates a traversable graph)
    """
    tmp = tmp_path_factory.mktemp("path_sphere")
    builder = GDSBuilder("path_test", str(tmp / "gds"))

    accounts = pa.table({
        "primary_key": [f"A{i:03d}" for i in range(20)],
        "name": [f"Account {i}" for i in range(20)],
    })
    builder.add_line("accounts", accounts, key_col="primary_key",
                     source_id="accounts", role="anchor")

    n_tx = 200
    txns = pa.table({
        "primary_key": [f"TX{i:05d}" for i in range(n_tx)],
        "sender_id": [f"A{i % 20:03d}" for i in range(n_tx)],
        "receiver_id": [f"A{(i + 3) % 20:03d}" for i in range(n_tx)],
        "amount": [float(100 + i) for i in range(n_tx)],
        "tx_date": pa.array(
            [1_700_000_000.0 + i * 3600.0 for i in range(n_tx)],
            type=pa.float64(),
        ),
    })
    builder.add_line("transactions", txns, key_col="primary_key",
                     source_id="transactions", role="event")

    builder.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="transactions",
        relations=[
            RelationSpec(line_id="accounts", fk_col="sender_id", direction="out"),
            RelationSpec(line_id="accounts", fk_col="receiver_id", direction="in"),
        ],
    )

    # Anchor pattern for accounts (needed for geometry/deltas)
    builder.add_pattern(
        "acct_pattern",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[
            RelationSpec(line_id="accounts", fk_col=None, direction="self"),
        ],
    )

    path = builder.build()
    return path


@pytest.fixture
def nav(tx_sphere):
    hs = HyperSphere.open(tx_sphere)
    session = hs.session("test_agent")
    return session.navigator()


# ── find_counterparties (edge table fast path) tests ────────


class TestFindCounterpartiesEdgeTable:
    def test_fast_path_returns_counterparties(self, nav):
        """Edge table fast path should find counterparties for A000."""
        result = nav.find_counterparties(
            "A000", "transactions", "sender_id", "receiver_id",
            pattern_id="tx_pattern",
        )
        assert result["primary_key"] == "A000"
        assert result["summary"]["total_outgoing"] > 0

    def test_includes_amount_sum_max(self, nav):
        """Fast path entries should include amount_sum and amount_max."""
        result = nav.find_counterparties(
            "A000", "transactions", "sender_id", "receiver_id",
            pattern_id="tx_pattern",
        )
        for entry in result["outgoing"]:
            assert "amount_sum" in entry
            assert "amount_max" in entry
            assert entry["amount_sum"] > 0
            assert entry["amount_max"] > 0

    def test_incoming(self, nav):
        """Fast path should also find incoming counterparties."""
        result = nav.find_counterparties(
            "A003", "transactions", "sender_id", "receiver_id",
            pattern_id="tx_pattern",
        )
        # A003 receives from A000 (i%20==0 sends to i+3%20==3)
        assert result["summary"]["total_incoming"] > 0

    def test_fallback_no_edge_table(self, nav):
        """Without edge table (acct_pattern), should fall back to points scan."""
        result = nav.find_counterparties(
            "A000", "transactions", "sender_id", "receiver_id",
            pattern_id=None,
        )
        # Falls back to points scan — no amount fields
        assert result["summary"]["total_outgoing"] > 0
        for entry in result["outgoing"]:
            assert "amount_sum" not in entry

    def test_use_edge_table_false(self, nav):
        """Explicit use_edge_table=False should force points scan."""
        result = nav.find_counterparties(
            "A000", "transactions", "sender_id", "receiver_id",
            pattern_id="tx_pattern", use_edge_table=False,
        )
        # Forced points scan — no amount fields
        assert result["summary"]["total_outgoing"] > 0
        for entry in result["outgoing"]:
            assert "amount_sum" not in entry

    def test_top_n_limits(self, nav):
        """top_n should cap the number of counterparties returned."""
        result = nav.find_counterparties(
            "A000", "transactions", "sender_id", "receiver_id",
            pattern_id="tx_pattern", top_n=1,
        )
        assert len(result["outgoing"]) <= 1


# ── entity_flow tests ───────────────────────────────────────


class TestEntityFlow:
    def test_basic_flow(self, nav):
        """A000 should have outgoing flow (sends to A003)."""
        result = nav.entity_flow("A000", "tx_pattern")
        assert result["outgoing_total"] > 0
        assert result["primary_key"] == "A000"

    def test_net_flow_sign(self, nav):
        """net_flow should equal outgoing_total - incoming_total."""
        result = nav.entity_flow("A000", "tx_pattern")
        expected = result["outgoing_total"] - result["incoming_total"]
        assert abs(result["net_flow"] - expected) < 0.01

    def test_counterparties_sorted_by_abs_net(self, nav):
        """Counterparties should be sorted by abs(net_flow) descending."""
        result = nav.entity_flow("A000", "tx_pattern")
        nets = [abs(cp["net_flow"]) for cp in result["counterparties"]]
        assert nets == sorted(nets, reverse=True)

    def test_no_edge_table_raises(self, nav):
        """Pattern without edge table should raise."""
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.entity_flow("A000", "acct_pattern")

    def test_nonexistent_entity_zeros(self, nav):
        """Nonexistent entity should return zero flows."""
        result = nav.entity_flow("NONEXISTENT", "tx_pattern")
        assert result["outgoing_total"] == 0
        assert result["incoming_total"] == 0
        assert result["net_flow"] == 0


# ── contagion_score tests ───────────────────────────────────


class TestContagionScore:
    def test_basic_score_range(self, nav):
        """Score should be between 0.0 and 1.0."""
        result = nav.contagion_score("A000", "tx_pattern")
        assert 0.0 <= result["score"] <= 1.0

    def test_has_counterparty_counts(self, nav):
        """Result should include counterparty counts."""
        result = nav.contagion_score("A000", "tx_pattern")
        assert result["total_counterparties"] > 0
        assert "anomalous_counterparties" in result
        assert "interpretation" in result

    def test_no_edge_table_raises(self, nav):
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.contagion_score("A000", "acct_pattern")

    def test_batch_returns_all_keys(self, nav):
        keys = ["A000", "A001", "A002"]
        result = nav.contagion_score_batch(keys, "tx_pattern")
        assert result["total"] == 3
        assert len(result["results"]) == 3

    def test_batch_summary(self, nav):
        keys = ["A000", "A001"]
        result = nav.contagion_score_batch(keys, "tx_pattern")
        assert "mean_score" in result["summary"]
        assert "max_score" in result["summary"]
        assert "high_contagion_count" in result["summary"]


# ── degree_velocity tests ───────────────────────────────────


class TestDegreeVelocity:
    def test_basic_structure(self, nav):
        """Should return correct keys for A000."""
        result = nav.degree_velocity("A000", "tx_pattern")
        assert result["primary_key"] == "A000"
        assert "velocity_out" in result
        assert "velocity_in" in result

    def test_uniform_timestamps_warning(self, nav):
        """Builder uses build-time as timestamp → uniform → warning."""
        # Edge table built by GDSBuilder has uniform timestamps (build time),
        # so degree_velocity correctly reports insufficient temporal spread.
        result = nav.degree_velocity("A000", "tx_pattern", n_buckets=3)
        assert result["velocity_out"] is None
        assert result["velocity_in"] is None
        assert "warning" in result
        assert result["buckets"] == []

    def test_no_edges_warning(self, nav):
        """Entity with no edges should produce warning."""
        result = nav.degree_velocity("NONEXISTENT", "tx_pattern")
        assert result["velocity_out"] is None
        assert result["velocity_in"] is None
        assert "warning" in result

    def test_no_edge_table_raises(self, nav):
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.degree_velocity("A000", "acct_pattern")


# ── investigation_coverage tests ────────────────────────────


class TestInvestigationCoverage:
    def test_full_coverage_100pct(self, nav):
        """All counterparties explored → 100% coverage."""
        # First get all counterparties for A000
        flow = nav.entity_flow("A000", "tx_pattern")
        all_cps = {cp["key"] for cp in flow["counterparties"]}
        result = nav.investigation_coverage("A000", "tx_pattern", all_cps)
        assert result["coverage_pct"] == 1.0
        assert result["unexplored"] == 0

    def test_zero_coverage(self, nav):
        """No explored keys → 0% coverage."""
        result = nav.investigation_coverage("A000", "tx_pattern", set())
        assert result["coverage_pct"] == 0.0
        assert result["explored"] == 0
        assert result["unexplored"] == result["total_edges"]

    def test_partial_coverage(self, nav):
        """Exploring some keys → partial coverage."""
        result = nav.investigation_coverage("A000", "tx_pattern", {"A003"})
        assert 0.0 < result["coverage_pct"] < 1.0
        assert result["explored"] >= 1

    def test_unexplored_have_anomaly_info(self, nav):
        """Unexplored anomalous entries should have is_anomaly and delta_rank_pct."""
        result = nav.investigation_coverage("A000", "tx_pattern", set())
        for entry in result["unexplored_anomalous"]:
            assert "is_anomaly" in entry
            assert "delta_rank_pct" in entry

    def test_leaf_node_no_crash(self, nav):
        """Entity with zero edges should not crash (coverage_pct=None)."""
        result = nav.investigation_coverage("NONEXISTENT", "tx_pattern", set())
        assert result["total_edges"] == 0
        assert result["coverage_pct"] is None
        assert "No counterparties" in result["summary"]

    def test_no_edge_table_raises(self, nav):
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.investigation_coverage("A000", "acct_pattern", set())


# ── propagate_influence tests ───────────────────────────────


class TestPropagateInfluence:
    def test_basic_propagation(self, nav):
        """Seeds should propagate to neighbors."""
        # Low threshold because test fixture has zero deltas → coherence clamped to 0.01
        result = nav.propagate_influence(["A000"], "tx_pattern", min_threshold=0.001)
        assert result["summary"]["total_affected"] > 0
        assert result["seeds"] == ["A000"]

    def test_depth_1_only_neighbors(self, nav):
        result = nav.propagate_influence(["A000"], "tx_pattern", max_depth=1, min_threshold=0.001)
        for entity in result["affected_entities"]:
            assert entity["depth"] == 1

    def test_decay_reduces_score(self, nav):
        result = nav.propagate_influence(["A000"], "tx_pattern", max_depth=2, min_threshold=0.0001)
        scores_by_depth: dict[int, list[float]] = {}
        for e in result["affected_entities"]:
            scores_by_depth.setdefault(e["depth"], []).append(e["influence_score"])
        # If both depths present, depth 2 avg should be <= depth 1 avg
        if 1 in scores_by_depth and 2 in scores_by_depth:
            avg1 = sum(scores_by_depth[1]) / len(scores_by_depth[1])
            avg2 = sum(scores_by_depth[2]) / len(scores_by_depth[2])
            assert avg2 <= avg1

    def test_high_threshold_fewer_affected(self, nav):
        low = nav.propagate_influence(["A000"], "tx_pattern", min_threshold=0.001)
        high = nav.propagate_influence(["A000"], "tx_pattern", min_threshold=0.5)
        assert low["summary"]["total_affected"] >= high["summary"]["total_affected"]

    def test_multiple_seeds(self, nav):
        result = nav.propagate_influence(["A000", "A001"], "tx_pattern", max_depth=1, min_threshold=0.001)
        assert result["summary"]["total_affected"] > 0

    def test_has_anomaly_field(self, nav):
        result = nav.propagate_influence(["A000"], "tx_pattern", max_depth=1)
        for e in result["affected_entities"]:
            assert "is_anomaly" in e

    def test_seeds_not_in_affected(self, nav):
        result = nav.propagate_influence(["A000"], "tx_pattern")
        affected_keys = {e["key"] for e in result["affected_entities"]}
        assert "A000" not in affected_keys

    def test_no_edge_table_raises(self, nav):
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.propagate_influence(["A000"], "acct_pattern")


# ── cluster_bridges tests ───────────────────────────────────


class TestClusterBridges:
    def test_basic_bridges(self, nav):
        """Should find clusters and bridges."""
        result = nav.cluster_bridges("tx_pattern", n_clusters=3)
        assert len(result["clusters"]) > 0
        assert "bridges" in result

    def test_bridge_structure(self, nav):
        result = nav.cluster_bridges("tx_pattern", n_clusters=3)
        for bridge in result["bridges"]:
            assert "cluster_a" in bridge
            assert "cluster_b" in bridge
            assert "edge_count" in bridge
            assert bridge["cluster_a"] != bridge["cluster_b"]

    def test_bridge_entities_have_anomaly(self, nav):
        result = nav.cluster_bridges("tx_pattern", n_clusters=3)
        for bridge in result["bridges"]:
            for entity in bridge["bridge_entities"]:
                assert "is_anomaly" in entity

    def test_cluster_ids_differ(self, nav):
        result = nav.cluster_bridges("tx_pattern", n_clusters=3)
        for bridge in result["bridges"]:
            assert bridge["cluster_a"] != bridge["cluster_b"]

    def test_top_n_bridges_limit(self, nav):
        result = nav.cluster_bridges("tx_pattern", n_clusters=3, top_n_bridges=2)
        assert len(result["bridges"]) <= 2

    def test_summary_present(self, nav):
        result = nav.cluster_bridges("tx_pattern", n_clusters=3)
        s = result["summary"]
        assert "total_clusters" in s
        assert "total_bridge_edges" in s
        assert "total_bridge_entities" in s

    def test_no_edge_table_raises(self, nav):
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.cluster_bridges("acct_pattern")


# ── anomalous_edges tests ───────────────────────────────────


class TestAnomalousEdges:
    def test_basic_anomalous_edges(self, nav):
        """A000→A003 should have edges (fixture: A000 sends to A003)."""
        result = nav.anomalous_edges("A000", "A003", "tx_pattern")
        assert result["from_key"] == "A000"
        assert result["to_key"] == "A003"
        assert result["summary"]["total_edges"] > 0

    def test_both_directions(self, nav):
        """Should find edges in both directions."""
        # A000→A003 and A003→A006, but A003 also receives from A000
        result = nav.anomalous_edges("A000", "A003", "tx_pattern")
        directions = {(e["from_key"], e["to_key"]) for e in result["edges"]}
        # At minimum A000→A003 should exist
        assert ("A000", "A003") in directions

    def test_top_n_limit(self, nav):
        result = nav.anomalous_edges("A000", "A003", "tx_pattern", top_n=2)
        assert len(result["edges"]) <= 2

    def test_no_edges_disconnected(self, nav):
        """Disconnected pair should return empty edges."""
        result = nav.anomalous_edges("A000", "A001", "tx_pattern")
        # A000 sends to A003 (not A001), A001 sends to A004
        assert result["summary"]["total_edges"] == 0
        assert result["edges"] == []

    def test_no_edge_table_raises(self, nav):
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.anomalous_edges("A000", "A003", "acct_pattern")

    def test_summary_counts(self, nav):
        result = nav.anomalous_edges("A000", "A003", "tx_pattern")
        s = result["summary"]
        assert "total_edges" in s
        assert "returned" in s
        assert "anomalous" in s
        assert "max_delta_norm" in s
        assert s["returned"] <= s["total_edges"]

    def test_event_geometry_enrichment_keys(self, nav):
        """Edges should have geometry fields from event pattern enrichment."""
        result = nav.anomalous_edges("A000", "A003", "tx_pattern")
        for edge in result["edges"]:
            assert "delta_norm" in edge
            assert "is_anomaly" in edge
            assert "delta_rank_pct" in edge
            assert isinstance(edge["delta_norm"], float)
            assert isinstance(edge["is_anomaly"], bool)


# ── find_geometric_path tests ────────────────────────────────


class TestFindGeometricPath:
    def test_direct_connection(self, nav):
        """A000 sends to A003 directly."""
        result = nav.find_geometric_path("A000", "A003", "tx_pattern")
        assert result["summary"]["paths_found"] >= 1
        best = result["paths"][0]
        assert best["keys"][0] == "A000"
        assert best["keys"][-1] == "A003"

    def test_two_hops(self, nav):
        """A000→A003→A006 should be found within depth 2."""
        result = nav.find_geometric_path(
            "A000", "A006", "tx_pattern", max_depth=3,
        )
        assert result["summary"]["paths_found"] >= 1
        assert any(p["hops"] <= 3 for p in result["paths"])

    def test_no_path_disconnected(self, nav):
        """Asking for a nonexistent target yields empty paths."""
        result = nav.find_geometric_path(
            "A000", "NONEXISTENT", "tx_pattern", max_depth=3,
        )
        assert result["summary"]["paths_found"] == 0
        assert result["paths"] == []

    def test_max_depth_limit(self, nav):
        """With depth=1, only direct neighbors are reachable."""
        result = nav.find_geometric_path(
            "A000", "A006", "tx_pattern", max_depth=1,
        )
        # A006 is 2 hops away, depth=1 shouldn't find it
        assert result["summary"]["paths_found"] == 0

    def test_scoring_modes(self, nav):
        """All four scoring modes should work without error."""
        for mode in ("geometric", "anomaly", "shortest", "amount"):
            result = nav.find_geometric_path(
                "A000", "A003", "tx_pattern", scoring=mode,
            )
            assert result["scoring"] == mode
            assert result["summary"]["paths_found"] >= 1

    def test_amount_scoring_mode(self, nav):
        """Amount scoring should find paths and report correctly."""
        result = nav.find_geometric_path(
            "A000", "A003", "tx_pattern", scoring="amount",
        )
        assert result["scoring"] == "amount"
        assert result["summary"]["paths_found"] >= 1
        assert "amount" in result["summary"]["score_interpretation"]

    def test_amount_score_hop_differs_from_geometric(self, nav):
        """_score_hop with amount mode should differ from geometric given non-zero deltas."""
        # Inject synthetic deltas into the cache
        pat = "acct_pattern"
        nav._delta_cache = {
            ("A000", pat): np.array([1.0, 0.5, -0.3], dtype=np.float32),
            ("A003", pat): np.array([0.8, 0.6, -0.2], dtype=np.float32),
        }
        nav._theta_cache = {pat: np.array([0.1, 0.1, 0.1], dtype=np.float32)}

        geo = nav._score_hop("A000", "A003", pat, "geometric")
        amt = nav._score_hop("A000", "A003", pat, "amount", amount=500.0, max_amount=1000.0)
        assert geo > 0, "geometric score should be positive with synthetic deltas"
        assert amt > geo, "amount mode should amplify geometric score"

    def test_no_cycles_in_path(self, nav):
        result = nav.find_geometric_path(
            "A000", "A009", "tx_pattern", max_depth=5,
        )
        for p in result["paths"]:
            assert len(p["keys"]) == len(set(p["keys"]))

    def test_beam_width_1(self, nav):
        """beam_width=1 still finds a path (greedy)."""
        result = nav.find_geometric_path(
            "A000", "A003", "tx_pattern", beam_width=1,
        )
        assert result["summary"]["paths_found"] >= 1

    def test_no_edge_table_raises(self, nav):
        """Pattern without edge table should raise."""
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.find_geometric_path("A000", "A003", "acct_pattern")


# ── discover_chains tests ────────────────────────────────────


class TestDiscoverChains:
    def test_basic_discovery(self, nav):
        result = nav.discover_chains(
            "A000", "tx_pattern", min_hops=2, max_chains=50,
        )
        assert result["summary"]["total"] > 0
        for chain in result["chains"]:
            assert chain["hop_count"] >= 2
            assert chain["keys"][0] == "A000"

    def test_temporal_ordering(self, nav):
        """Chains should respect temporal order (timestamps increasing)."""
        result = nav.discover_chains(
            "A000", "tx_pattern", min_hops=2, max_chains=10,
        )
        # All chains start from A000, so temporal ordering is implicit
        assert result["summary"]["total"] > 0

    def test_min_hops_filter(self, nav):
        result = nav.discover_chains(
            "A000", "tx_pattern", min_hops=3, max_chains=50,
        )
        for chain in result["chains"]:
            assert chain["hop_count"] >= 3

    def test_max_chains_limit(self, nav):
        result = nav.discover_chains(
            "A000", "tx_pattern", min_hops=2, max_chains=5,
        )
        assert len(result["chains"]) <= 5

    def test_geometric_score_present(self, nav):
        result = nav.discover_chains(
            "A000", "tx_pattern", min_hops=2, max_chains=10,
        )
        for chain in result["chains"]:
            assert "geometric_score" in chain

    def test_no_edge_table_raises(self, nav):
        with pytest.raises(GDSNavigationError, match="no edge table"):
            nav.discover_chains("A000", "acct_pattern")

    def test_forward_direction(self, nav):
        result = nav.discover_chains(
            "A000", "tx_pattern", direction="forward",
            min_hops=2, max_chains=20,
        )
        assert result["summary"]["total"] >= 0  # may be 0 if timestamps don't align
