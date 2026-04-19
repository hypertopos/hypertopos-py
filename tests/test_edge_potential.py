# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Unit + integration tests for geometric edge potential."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.navigation.navigator import GDSNavigator


def _mock_edges_table(pairs: list[tuple[str, str]]) -> pa.Table:
    """Build a fake Arrow edges table with from_key/to_key columns."""
    return pa.table({
        "from_key": [p[0] for p in pairs],
        "to_key": [p[1] for p in pairs],
    })


def _make_adjacency(pairs: list[tuple[str, str]]):
    from hypertopos.engine.adjacency import AdjacencyIndex
    if not pairs:
        return AdjacencyIndex(_out={}, _in={}, _nodes=set(), _edge_count=0)
    return AdjacencyIndex.from_edge_lists(
        from_keys=[p[0] for p in pairs],
        to_keys=[p[1] for p in pairs],
        timestamps=[float(i) for i in range(len(pairs))],
        amounts=[1.0] * len(pairs),
        event_keys=[f"e{i}" for i in range(len(pairs))],
    )


def _make_nav_with_edges(pairs: list[tuple[str, str]]) -> GDSNavigator:
    storage = MagicMock()
    storage.read_edges = MagicMock(return_value=_mock_edges_table(pairs))
    storage.get_adjacency = MagicMock(return_value=_make_adjacency(pairs))
    sphere = MagicMock()
    sphere.patterns = {
        "tx_pattern": MagicMock(pattern_type="event"),
        "account_pattern": MagicMock(pattern_type="anchor"),
    }
    storage.read_sphere = MagicMock(return_value=sphere)
    return GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())


class TestPairCountCache:
    def test_counts_distinct_ordered_pairs(self):
        nav = _make_nav_with_edges([
            ("A", "B"), ("A", "B"), ("A", "C"), ("B", "A"),
        ])
        counts = nav._pair_count_for_pattern("tx_pattern", version=1)
        assert counts == {("A", "B"): 2, ("A", "C"): 1, ("B", "A"): 1}

    def test_cache_is_shared_with_adjacency(self):
        # After the AdjacencyIndex reuse refactor, pair_counts share the
        # same cache lifetime as the adjacency itself (one instance per
        # pattern_id on the reader). Version parameter is cosmetic —
        # invalidation happens when the adjacency is rebuilt.
        nav = _make_nav_with_edges([("A", "B")])
        _ = nav._pair_count_for_pattern("tx_pattern", version=1)
        first = nav._storage.get_adjacency.call_count
        _ = nav._pair_count_for_pattern("tx_pattern", version=1)
        _ = nav._pair_count_for_pattern("tx_pattern", version=2)
        # Navigator delegates unconditionally; reader-level cache suppresses
        # the actual read_edges. We assert the delegation pattern works:
        assert nav._storage.get_adjacency.call_count == first + 2


class TestEdgePotentialScoring:
    def _make_nav_with_geometry(
        self,
        deltas: dict[str, np.ndarray],
        edges: list[tuple[str, str]],
        home_line: str = "accounts",
    ) -> GDSNavigator:
        """Build a nav where geometry lookup returns fixed delta vectors."""
        storage = MagicMock()
        storage.read_edges = MagicMock(return_value=_mock_edges_table(edges))
        storage.get_adjacency = MagicMock(return_value=_make_adjacency(edges))
        sphere = MagicMock()
        anchor = MagicMock(pattern_type="anchor")
        sphere.patterns = {"account_pattern": anchor, "tx_pattern": MagicMock(pattern_type="event")}
        sphere.entity_line = MagicMock(return_value=home_line)
        storage.read_sphere = MagicMock(return_value=sphere)

        def fake_read_geometry(pid, version, primary_key=None, columns=None, **kw):
            if primary_key in deltas:
                return pa.table({
                    "primary_key": [primary_key],
                    "delta": [deltas[primary_key].tolist()],
                })
            return pa.table({"primary_key": [], "delta": []})

        storage.read_geometry = fake_read_geometry
        nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())

        # Stub companion resolver so edge_potential finds the graph pid
        nav._resolve_edge_pattern_for_anchor = lambda pid: "tx_pattern"
        nav._resolve_version = lambda pid: 1
        return nav

    def test_identical_endpoints_score_zero(self):
        """||δ_A - δ_A|| = 0 → edge_potential = 0 regardless of pair count."""
        nav = self._make_nav_with_geometry(
            deltas={"A": np.array([1.0, 2.0, 3.0])},
            edges=[("A", "A")],
        )
        result = nav.edge_potential("A", "A", "account_pattern")
        assert result["score"] == 0.0
        assert result["delta_distance"] == 0.0
        assert result["pair_tx_count"] == 1

    def test_distant_rare_pair_scores_high(self):
        """Distant endpoints + singleton pair → high score."""
        nav = self._make_nav_with_geometry(
            deltas={
                "A": np.array([10.0, 0.0, 0.0]),
                "B": np.array([0.0, 10.0, 0.0]),
            },
            edges=[("A", "B")],
        )
        result = nav.edge_potential("A", "B", "account_pattern")
        expected_dist = float(np.linalg.norm(np.array([10.0, -10.0, 0.0])))
        # pair_count=1 → weight 1.0 → score == distance
        assert result["score"] == pytest.approx(expected_dist, rel=1e-4)
        assert result["pair_tx_count"] == 1

    def test_frequent_pair_gets_weight_shrink(self):
        """Distant endpoints + frequent pair → score shrunk by 1/count."""
        pairs = [("A", "B")] * 50
        nav = self._make_nav_with_geometry(
            deltas={
                "A": np.array([10.0, 0.0, 0.0]),
                "B": np.array([0.0, 10.0, 0.0]),
            },
            edges=pairs,
        )
        result = nav.edge_potential("A", "B", "account_pattern")
        expected_dist = float(np.linalg.norm(np.array([10.0, -10.0, 0.0])))
        assert result["score"] == pytest.approx(expected_dist / 50.0, rel=1e-4)
        assert result["pair_tx_count"] == 50

    def test_pair_count_cap(self):
        """pair_tx_count is capped at 1000 — weight 1/1000, no underflow on 10k-tx pairs."""
        pairs = [("A", "B")] * 10_000
        nav = self._make_nav_with_geometry(
            deltas={
                "A": np.array([10.0, 0.0, 0.0]),
                "B": np.array([0.0, 10.0, 0.0]),
            },
            edges=pairs,
        )
        result = nav.edge_potential("A", "B", "account_pattern")
        expected_dist = float(np.linalg.norm(np.array([10.0, -10.0, 0.0])))
        assert result["score"] == pytest.approx(expected_dist / 1000.0, rel=1e-4)
        assert result["pair_tx_count"] == 10_000  # raw count reported, but weight capped

    def test_missing_endpoint_raises(self):
        """A key without geometry cannot be scored — raise GDSNavigationError."""
        from hypertopos.navigation.navigator import GDSNavigationError
        nav = self._make_nav_with_geometry(
            deltas={"A": np.array([1.0, 2.0])},
            edges=[("A", "B")],
        )
        with pytest.raises(GDSNavigationError, match="not found"):
            nav.edge_potential("A", "B", "account_pattern")


class TestAttractEdgePotential:
    def _nav(
        self,
        deltas: dict[str, np.ndarray],
        edges: list[tuple[str, str]],
    ) -> GDSNavigator:
        storage = MagicMock()
        storage.read_edges = MagicMock(return_value=_mock_edges_table(edges))
        storage.get_adjacency = MagicMock(return_value=_make_adjacency(edges))
        sphere = MagicMock()
        sphere.patterns = {
            "account_pattern": MagicMock(pattern_type="anchor"),
            "tx_pattern": MagicMock(pattern_type="event"),
        }
        sphere.entity_line = MagicMock(return_value="accounts")
        storage.read_sphere = MagicMock(return_value=sphere)

        def fake_read_geometry(pid, version, primary_key=None, columns=None, **kw):
            if primary_key and primary_key in deltas:
                return pa.table({
                    "primary_key": [primary_key],
                    "delta": [deltas[primary_key].tolist()],
                })
            keys = sorted(deltas.keys())
            return pa.table({
                "primary_key": keys,
                "delta": [deltas[k].tolist() for k in keys],
            })

        storage.read_geometry = fake_read_geometry
        nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())
        nav._resolve_edge_pattern_for_anchor = lambda pid: "tx_pattern"
        nav._resolve_version = lambda pid: 1
        return nav

    def test_returns_sorted_desc_by_score(self):
        deltas = {
            "A": np.array([10.0, 0.0]),
            "B": np.array([0.0, 10.0]),
            "C": np.array([5.0, 0.0]),
            "D": np.array([0.0, 0.0]),
        }
        edges = [("A", "B"), ("C", "D"), ("C", "D"), ("A", "C")]
        nav = self._nav(deltas, edges)
        results = nav.attract_edge_potential("account_pattern", top_n=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_cap(self):
        deltas = {k: np.array([float(i), 0.0]) for i, k in enumerate("ABCDE")}
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("A", "E")]
        nav = self._nav(deltas, edges)
        results = nav.attract_edge_potential("account_pattern", top_n=3)
        assert len(results) == 3

    def test_cold_start_does_not_push_endpoint_filter_into_lance(self):
        # Anti-regression guard. Blanket pin — asserts the cold-start call
        # NEVER passes point_keys, regardless of endpoint density.
        # Rationale: an earlier attempt passed `point_keys=list(endpoints)`
        # to read_geometry to skip non-endpoint rows. Empirically this is
        # ~1000–1500× slower on endpoint-dense patterns (measured on
        # 2026-04-19: 85s with point_keys vs 0.05s full scan on a ~500k-
        # account AML graph). A future conditional re-attempt ("use
        # point_keys if endpoints < threshold") would pass this test on
        # mocks with few endpoints — if you plan such a re-attempt,
        # re-benchmark empirically on a real sphere with your expected
        # endpoint density before flipping this test to a regime-specific
        # assertion.
        deltas = {k: np.array([float(i), 0.0]) for i, k in enumerate("ABCDE")}
        edges = [("A", "B"), ("B", "C"), ("A", "C")]

        storage = MagicMock()
        storage.read_edges = MagicMock(return_value=_mock_edges_table(edges))
        storage.get_adjacency = MagicMock(return_value=_make_adjacency(edges))
        sphere = MagicMock()
        sphere.patterns = {
            "account_pattern": MagicMock(pattern_type="anchor"),
            "tx_pattern": MagicMock(pattern_type="event"),
        }
        sphere.entity_line = MagicMock(return_value="accounts")
        storage.read_sphere = MagicMock(return_value=sphere)

        seen_point_keys: list[list[str] | None] = []

        def spy_read_geometry(pid, version, point_keys=None, columns=None, **kw):
            seen_point_keys.append(list(point_keys) if point_keys is not None else None)
            keys = sorted(deltas.keys())
            return pa.table({
                "primary_key": keys,
                "delta": [deltas[k].tolist() for k in keys],
            })

        storage.read_geometry = spy_read_geometry
        nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())
        nav._resolve_edge_pattern_for_anchor = lambda pid: "tx_pattern"
        nav._resolve_version = lambda pid: 1

        nav.attract_edge_potential("account_pattern", top_n=10)

        # The cold-start read MUST NOT receive point_keys — full scan is
        # faster on the endpoint-dense patterns that hypertopos actually
        # sees in production AML data.
        assert any(p is None for p in seen_point_keys), (
            "attract_edge_potential cold-start passed point_keys into "
            "read_geometry — causes ~1000-1500× slowdown on dense-endpoint "
            "patterns."
        )

    def test_min_pair_count_filter(self):
        deltas = {
            "A": np.array([10.0, 0.0]),
            "B": np.array([0.0, 10.0]),
            "C": np.array([5.0, 0.0]),
        }
        edges = [("A", "B"), ("A", "C"), ("A", "C"), ("A", "C")]
        nav = self._nav(deltas, edges)
        results = nav.attract_edge_potential(
            "account_pattern", top_n=10, min_pair_count=2,
        )
        pairs = {(r["from_key"], r["to_key"]) for r in results}
        assert ("A", "B") not in pairs
        assert ("A", "C") in pairs

    def test_scoped_by_from_key(self):
        deltas = {k: np.array([float(i), 0.0]) for i, k in enumerate("ABCD")}
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        nav = self._nav(deltas, edges)
        results = nav.attract_edge_potential(
            "account_pattern", top_n=10, from_key="A",
        )
        for r in results:
            assert r["from_key"] == "A"


class TestEdgePotentialFixtureSmoke:
    """Real fixture sphere (sales_sphere) integration — protects against
    shape mismatches and missing-endpoint handling that mock tests can't catch."""

    def test_graceful_when_no_companion(self, sphere_path):
        """Real fixture sphere (sales_sphere) has only a customer_pattern with no
        event companion. `attract_edge_potential` must raise GDSNavigationError
        loudly — NOT return empty list silently.

        This is the graceful-empty path: if the fixture is later extended with
        an event pattern, add a separate test `test_full_pipeline_on_extended_fixture`
        that exercises the non-empty path.
        """
        from hypertopos.sphere import HyperSphere
        from hypertopos.navigation.navigator import GDSNavigationError

        hs = HyperSphere.open(sphere_path)
        with hs.session("test-agent") as session:
            sphere = session._reader.read_sphere()
            anchor_pids = [
                pid for pid, pat in sphere.patterns.items()
                if pat.pattern_type == "anchor"
            ]
            assert anchor_pids, (
                "Fixture regression: sales_sphere has no anchor pattern — "
                "update the fixture or the test's pattern-selection strategy."
            )
            pid = anchor_pids[0]
            nav = session.navigator()
            companion = nav._resolve_edge_pattern_for_anchor(pid)
            # Expectation: sales_sphere fixture has NO companion. If that
            # changes, this test must be updated — don't let it silently skip.
            assert companion is None, (
                f"Fixture now has a companion for {pid!r}: {companion!r}. "
                "Extend this test to exercise the populated path."
            )

            with pytest.raises(GDSNavigationError, match="no graph companion"):
                nav.attract_edge_potential(pid, top_n=5)

    def test_edge_potential_single_lookup_graceful_when_no_companion(self, sphere_path):
        """score_edge on sales_sphere must raise loudly — same fail-loud contract."""
        from hypertopos.sphere import HyperSphere
        from hypertopos.navigation.navigator import GDSNavigationError

        hs = HyperSphere.open(sphere_path)
        with hs.session("test-agent") as session:
            sphere = session._reader.read_sphere()
            anchor_pids = [
                pid for pid, pat in sphere.patterns.items()
                if pat.pattern_type == "anchor"
            ]
            assert anchor_pids
            pid = anchor_pids[0]
            nav = session.navigator()
            with pytest.raises(GDSNavigationError, match="no graph companion"):
                nav.edge_potential("ANY-KEY", "ANOTHER-KEY", pid)


class TestGlobalVsFilterRankPct:
    """score_rank_pct_global is pattern-wide (immutable across filters);
    score_rank_pct_in_filter reflects rank WITHIN the filtered subset.
    is_high_potential uses the GLOBAL p95 threshold — not filter-local."""

    def _nav(
        self,
        deltas: dict[str, np.ndarray],
        edges: list[tuple[str, str]],
    ) -> GDSNavigator:
        storage = MagicMock()
        storage.read_edges = MagicMock(return_value=_mock_edges_table(edges))
        storage.get_adjacency = MagicMock(return_value=_make_adjacency(edges))
        sphere = MagicMock()
        sphere.patterns = {
            "account_pattern": MagicMock(pattern_type="anchor"),
            "tx_pattern": MagicMock(pattern_type="event"),
        }
        sphere.entity_line = MagicMock(return_value="accounts")
        storage.read_sphere = MagicMock(return_value=sphere)

        def fake_read_geometry(pid, version, primary_key=None, columns=None, **kw):
            keys = sorted(deltas.keys())
            return pa.table({
                "primary_key": keys,
                "delta": [deltas[k].tolist() for k in keys],
            })

        storage.read_geometry = fake_read_geometry
        nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())
        nav._resolve_edge_pattern_for_anchor = lambda pid: "tx_pattern"
        nav._resolve_version = lambda pid: 1
        return nav

    def test_global_pct_stable_across_filter_calls(self):
        deltas = {
            "A": np.array([10.0, 0.0]),
            "B": np.array([0.0, 10.0]),
            "C": np.array([5.0, 0.0]),
            "D": np.array([0.0, 0.0]),
        }
        # Mix of singleton and recurring pairs
        edges = [("A", "B")] + [("C", "D")] * 3 + [("A", "C")] * 2
        nav = self._nav(deltas, edges)
        base = nav.attract_edge_potential("account_pattern", top_n=10)
        # Same top entry's global pct should match in filter query
        top_entry = base[0]
        filtered = nav.attract_edge_potential(
            "account_pattern", top_n=10, from_key=top_entry["from_key"],
        )
        # The same (from, to) entry should have identical global pct
        match = [r for r in filtered if r["to_key"] == top_entry["to_key"]]
        assert match
        assert match[0]["score_rank_pct_global"] == top_entry["score_rank_pct_global"]

    def test_filter_local_pct_is_100_for_filter_top(self):
        deltas = {
            "A": np.array([10.0, 0.0]),
            "B": np.array([0.0, 10.0]),
            "C": np.array([5.0, 0.0]),
        }
        edges = [("A", "B"), ("A", "B"), ("A", "B"), ("A", "C"), ("B", "C")]
        nav = self._nav(deltas, edges)
        filtered = nav.attract_edge_potential(
            "account_pattern", top_n=5, min_pair_count=3,
        )
        # Filter-local pct must exist and start at 100 for the top entry
        assert filtered
        assert filtered[0]["score_rank_pct_in_filter"] == 100.0

    def test_is_high_potential_uses_global_p95_not_filter_local(self):
        """The boolean is stable against filter changes — a pair below global
        p95 stays False even if it's the top of a narrow filter subset."""
        deltas = {
            "A": np.array([10.0, 0.0]),
            "B": np.array([0.0, 10.0]),
            "C": np.array([5.0, 0.0]),
            "D": np.array([0.0, 0.0]),
        }
        # 100 singleton pairs with high distance (top of distribution)
        singleton_edges = []
        for i in range(20):
            singleton_edges.append((f"S{i}", f"T{i}"))
            deltas[f"S{i}"] = np.array([float(i), 0.0])
            deltas[f"T{i}"] = np.array([0.0, float(i + 10)])
        # A frequent low-score pair
        freq_edges = [("A", "C")] * 10
        nav = self._nav(deltas, singleton_edges + freq_edges)
        # When filtered to min_pair_count=10 only (A,C) survives — it's
        # filter-local top (pct=100) but globally below p95 of the huge
        # singleton pool. is_high_potential should be False.
        filtered = nav.attract_edge_potential(
            "account_pattern", top_n=5, min_pair_count=10,
        )
        assert filtered
        assert filtered[0]["score_rank_pct_in_filter"] == 100.0
        # Global pct is lower because the frequent pair has score << singletons
        assert filtered[0]["score_rank_pct_global"] < 100.0
