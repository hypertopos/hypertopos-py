# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Unit + integration tests for find_motif primitives.

Test classes cover: motif scoring, motif registry dispatch, per-type
enumerators (fan_out / cycle_2 / cycle_3 / structuring), ranking cache
behaviour, and fixture-based end-to-end smoke.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hypertopos.navigation.navigator import (
    GDSNavigationError,
    GDSNavigator,
    MotifSpec,
)


# -----------------------------------------------------------------------------
# Task 1: Foundation — MotifSpec + _motif_registry + _score_motif_from_edges
# -----------------------------------------------------------------------------


class TestMotifRegistryContract:
    """The dispatcher registry must expose each motif as a MotifSpec."""

    def test_registry_has_four_motifs(self) -> None:
        nav = GDSNavigator(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        keys = set(nav._motif_registry.keys())
        assert keys == {"fan_out", "cycle_2", "cycle_3", "structuring"}

    def test_each_motif_has_positive_default_window(self) -> None:
        nav = GDSNavigator(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        for mt, spec in nav._motif_registry.items():
            assert spec.default_window_hours > 0, f"{mt} has non-positive window"

    def test_each_motif_exposes_callable_enumerator(self) -> None:
        nav = GDSNavigator(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        for mt, spec in nav._motif_registry.items():
            assert callable(spec.enumerate), f"{mt} enumerator is not callable"


class TestMotifScoring:
    """_score_motif_from_edges returns the product of edge_potentials."""

    def _nav_with_edge_pots(
        self, edge_pot_map: dict[tuple[str, str], float],
    ) -> GDSNavigator:
        nav = GDSNavigator(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        # Patch edge_potential to a deterministic lookup for unit testing.
        def _ep(from_key: str, to_key: str, pattern_id: str):  # type: ignore[no-untyped-def]
            return {
                "from_key": from_key,
                "to_key": to_key,
                "pattern_id": pattern_id,
                "score": edge_pot_map[(from_key, to_key)],
                "delta_distance": edge_pot_map[(from_key, to_key)],
                "pair_tx_count": 1,
                "effective_weight": 1.0,
            }

        nav.edge_potential = _ep  # type: ignore[assignment]
        return nav

    def test_score_is_product_of_edge_potentials(self) -> None:
        nav = self._nav_with_edge_pots(
            {("A", "B"): 2.0, ("B", "C"): 3.0, ("C", "A"): 4.0}
        )
        result = nav._score_motif_from_edges(
            [("A", "B"), ("B", "C"), ("C", "A")], "p"
        )
        assert result["score"] == pytest.approx(24.0)
        assert len(result["breakdown"]) == 3

    def test_score_breakdown_order_matches_input(self) -> None:
        nav = self._nav_with_edge_pots(
            {("X", "Y"): 1.5, ("Y", "X"): 2.5}
        )
        result = nav._score_motif_from_edges([("X", "Y"), ("Y", "X")], "p")
        edges = [entry["edge"] for entry in result["breakdown"]]
        assert edges == [("X", "Y"), ("Y", "X")]

    def test_score_underflow_clamped_to_epsilon(self) -> None:
        nav = self._nav_with_edge_pots(
            {("A", "B"): 1e-20, ("B", "C"): 1e-20, ("C", "A"): 1e-20}
        )
        result = nav._score_motif_from_edges(
            [("A", "B"), ("B", "C"), ("C", "A")], "p"
        )
        # Raw product = 1e-60, should be clamped to 1e-30 to keep sortability.
        assert result["score"] >= 1e-30

    def test_score_zero_when_any_edge_potential_is_zero(self) -> None:
        """Identical-delta endpoints produce edge_potential=0 → motif score=0."""
        nav = self._nav_with_edge_pots(
            {("A", "B"): 0.0, ("B", "C"): 5.0, ("C", "A"): 5.0}
        )
        result = nav._score_motif_from_edges(
            [("A", "B"), ("B", "C"), ("C", "A")], "p"
        )
        # Zero dominates product → score == 0 (before underflow clamp, because
        # zero is semantically distinct from underflow).
        assert result["score"] == 0.0

    def test_score_empty_edges_raises(self) -> None:
        nav = self._nav_with_edge_pots({})
        with pytest.raises(GDSNavigationError):
            nav._score_motif_from_edges([], "p")

    def test_motif_spec_is_frozen_dataclass(self) -> None:
        spec = MotifSpec(
            enumerate=lambda *a, **kw: None,
            default_window_hours=24,
            min_instances=1,
        )
        with pytest.raises(dataclasses_frozen_error()):
            spec.default_window_hours = 48  # type: ignore[misc]


def dataclasses_frozen_error() -> type[Exception]:
    """Dataclass frozen mutation raises FrozenInstanceError at runtime."""
    import dataclasses
    return dataclasses.FrozenInstanceError


# -----------------------------------------------------------------------------
# Task 2/3/4: Enumerators (fan_out, cycle_2, cycle_3)
# -----------------------------------------------------------------------------

import pyarrow as pa


def _hour_us(h: float) -> int:
    return int(h * 3600 * 1_000_000)


def _mock_edges(rows: list[tuple[str, str, int]]) -> pa.Table:
    """Build a fake edge table with (from_key, to_key, timestamp_us)."""
    return pa.table({
        "from_key": [r[0] for r in rows],
        "to_key": [r[1] for r in rows],
        "timestamp": [r[2] for r in rows],
    })


def _enum_nav(edges_rows: list[tuple[str, str, int]]) -> GDSNavigator:
    """Build a navigator with a mock storage, sphere, and graph pattern."""
    storage = MagicMock()
    storage.has_edge_table = MagicMock(return_value=True)

    def _read_edges(
        pattern_id: str,
        from_keys: list[str] | None = None,
        to_keys: list[str] | None = None,
        timestamp_from: float | None = None,
        timestamp_to: float | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        filtered = edges_rows
        if from_keys is not None:
            filtered = [r for r in filtered if r[0] in from_keys]
        if to_keys is not None:
            filtered = [r for r in filtered if r[1] in to_keys]
        if timestamp_from is not None:
            filtered = [r for r in filtered if r[2] >= timestamp_from]
        if timestamp_to is not None:
            filtered = [r for r in filtered if r[2] <= timestamp_to]
        return _mock_edges(filtered)

    storage.read_edges = MagicMock(side_effect=_read_edges)

    sphere = MagicMock()
    event_pattern = MagicMock()
    event_pattern.pattern_type = "event"
    sphere.patterns = {"tx": event_pattern}
    storage.read_sphere = MagicMock(return_value=sphere)

    nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())
    return nav


class TestFanOutEnumerator:

    def test_fan_out_below_min_k_returns_empty(self) -> None:
        nav = _enum_nav([
            ("H", "T1", _hour_us(0)),
            ("H", "T2", _hour_us(10)),
        ])
        result = nav._enumerate_fan_out("H", "tx", time_window_hours=168, min_k=3)
        assert result == []

    def test_fan_out_meets_min_k_returns_one_motif(self) -> None:
        nav = _enum_nav([
            ("H", "T1", _hour_us(0)),
            ("H", "T2", _hour_us(1)),
            ("H", "T3", _hour_us(2)),
        ])
        result = nav._enumerate_fan_out("H", "tx", time_window_hours=168, min_k=3)
        assert len(result) == 1
        assert result[0]["motif_type"] == "fan_out"
        assert result[0]["seed"] == "H"
        assert result[0]["k"] == 3
        edges = result[0]["edges"]
        assert set(edges) == {("H", "T1"), ("H", "T2"), ("H", "T3")}

    def test_fan_out_filters_outside_time_window(self) -> None:
        # Window = 24h. Targets T1,T2 in window, T3 5 days back.
        nav = _enum_nav([
            ("H", "T1", _hour_us(100)),
            ("H", "T2", _hour_us(105)),
            ("H", "T3", _hour_us(0)),
        ])
        result = nav._enumerate_fan_out("H", "tx", time_window_hours=24, min_k=2)
        assert len(result) == 1
        edges_to = {e[1] for e in result[0]["edges"]}
        assert edges_to == {"T1", "T2"}

    def test_fan_out_deduplicates_repeated_tx_to_same_target(self) -> None:
        nav = _enum_nav([
            ("H", "T1", _hour_us(0)),
            ("H", "T1", _hour_us(1)),
            ("H", "T1", _hour_us(2)),
            ("H", "T2", _hour_us(3)),
            ("H", "T3", _hour_us(4)),
        ])
        result = nav._enumerate_fan_out("H", "tx", time_window_hours=168, min_k=3)
        assert result[0]["k"] == 3


class TestCycle2Enumerator:

    def test_cycle_2_requires_both_directions(self) -> None:
        nav = _enum_nav([("A", "B", _hour_us(0))])
        result = nav._enumerate_cycle_2("A", "tx", time_window_hours=24)
        assert result == []

    def test_cycle_2_finds_bidirectional_pair_in_window(self) -> None:
        nav = _enum_nav([
            ("A", "B", _hour_us(0)),
            ("B", "A", _hour_us(5)),
        ])
        result = nav._enumerate_cycle_2("A", "tx", time_window_hours=24)
        assert len(result) == 1
        assert result[0]["motif_type"] == "cycle_2"
        assert result[0]["seed"] == "A"
        assert result[0]["counterparty"] == "B"
        assert set(result[0]["edges"]) == {("A", "B"), ("B", "A")}

    def test_cycle_2_time_window_enforced(self) -> None:
        nav = _enum_nav([
            ("A", "B", _hour_us(0)),
            ("B", "A", _hour_us(48)),
        ])
        result = nav._enumerate_cycle_2("A", "tx", time_window_hours=24)
        assert result == []

    def test_cycle_2_self_loop_excluded(self) -> None:
        nav = _enum_nav([("A", "A", _hour_us(0))])
        result = nav._enumerate_cycle_2("A", "tx", time_window_hours=24)
        assert result == []

    def test_cycle_2_with_counterparty_filter(self) -> None:
        nav = _enum_nav([
            ("A", "B", _hour_us(0)), ("B", "A", _hour_us(1)),
            ("A", "C", _hour_us(2)), ("C", "A", _hour_us(3)),
        ])
        result = nav._enumerate_cycle_2(
            "A", "tx", time_window_hours=24, counterparty="C",
        )
        assert len(result) == 1
        assert result[0]["counterparty"] == "C"


class TestCycle3Enumerator:

    def test_cycle_3_finds_triad_with_temporal_ordering(self) -> None:
        nav = _enum_nav([
            ("A", "B", _hour_us(0)),
            ("B", "C", _hour_us(1)),
            ("C", "A", _hour_us(2)),
        ])
        result = nav._enumerate_cycle_3("A", "tx", time_window_hours=72)
        assert len(result) == 1
        assert result[0]["motif_type"] == "cycle_3"
        assert result[0]["ring"] == ["A", "B", "C"]
        assert result[0]["edges"] == [("A", "B"), ("B", "C"), ("C", "A")]

    def test_cycle_3_rejects_non_monotonic_timestamps(self) -> None:
        # A→B at t=10, B→C at t=5, C→A at t=20 — not strictly increasing.
        nav = _enum_nav([
            ("A", "B", _hour_us(10)),
            ("B", "C", _hour_us(5)),
            ("C", "A", _hour_us(20)),
        ])
        result = nav._enumerate_cycle_3("A", "tx", time_window_hours=72)
        assert result == []

    def test_cycle_3_enforces_window_span(self) -> None:
        # Span = 100h > window 72h.
        nav = _enum_nav([
            ("A", "B", _hour_us(0)),
            ("B", "C", _hour_us(50)),
            ("C", "A", _hour_us(100)),
        ])
        result = nav._enumerate_cycle_3("A", "tx", time_window_hours=72)
        assert result == []

    def test_cycle_3_rejects_self_loop_second_hop(self) -> None:
        # A→B, B→A, A→A would be "cycle" — reject (self-loop at step 3 is not a triad).
        nav = _enum_nav([
            ("A", "B", _hour_us(0)),
            ("B", "A", _hour_us(1)),
        ])
        result = nav._enumerate_cycle_3("A", "tx", time_window_hours=72)
        # This is a cycle_2 pattern, not cycle_3. Should return empty.
        assert result == []

    def test_cycle_3_max_triads_cap(self) -> None:
        # A fans out to many B's, each has one C that closes back.
        rows: list[tuple[str, str, int]] = []
        for i in range(100):
            b = f"B{i}"
            c = f"C{i}"
            rows.append(("A", b, _hour_us(0 + i * 0.01)))
            rows.append((b, c, _hour_us(1 + i * 0.01)))
            rows.append((c, "A", _hour_us(2 + i * 0.01)))
        nav = _enum_nav(rows)
        result = nav._enumerate_cycle_3("A", "tx", time_window_hours=72, max_triads=50)
        assert len(result) == 50


# -----------------------------------------------------------------------------
# Task 5: Public score_motif dispatch
# -----------------------------------------------------------------------------


def _nav_with_mocked_pipeline(
    edges_rows: list[tuple[str, str, int]],
    edge_pot_map: dict[tuple[str, str], float],
    amounts: dict[tuple[str, str], float] | None = None,
) -> GDSNavigator:
    nav = _enum_nav(edges_rows)

    # Stub storage.get_adjacency with a real AdjacencyIndex built from
    # edges_rows. Test timestamps are int-microseconds (via _hour_us);
    # AdjacencyIndex + _rank_motifs both operate in float-seconds at
    # production, so convert μs → seconds here. Per-edge amounts honored
    # when provided via `amounts` (keyed by (from_key, to_key)); default 1.0.
    from hypertopos.engine.adjacency import AdjacencyIndex
    if edges_rows:
        edge_amounts = [
            (amounts.get((r[0], r[1]), 1.0) if amounts else 1.0)
            for r in edges_rows
        ]
        adj = AdjacencyIndex.from_edge_lists(
            from_keys=[r[0] for r in edges_rows],
            to_keys=[r[1] for r in edges_rows],
            timestamps=[r[2] / 1_000_000.0 for r in edges_rows],
            amounts=edge_amounts,
            event_keys=[f"e{i}" for i in range(len(edges_rows))],
        )
    else:
        adj = AdjacencyIndex(_out={}, _in={}, _nodes=set(), _edge_count=0)
    nav._storage.get_adjacency = MagicMock(return_value=adj)

    def _ep(from_key: str, to_key: str, pattern_id: str):  # type: ignore[no-untyped-def]
        return {
            "from_key": from_key,
            "to_key": to_key,
            "pattern_id": pattern_id,
            "score": edge_pot_map.get((from_key, to_key), 1.0),
            "delta_distance": 1.0,
            "pair_tx_count": 1,
            "effective_weight": 1.0,
        }
    nav.edge_potential = _ep  # type: ignore[assignment]

    # The ranking path uses a lean score via _batch_read_deltas +
    # _pair_count_for_pattern, bypassing edge_potential. Stub both so the lean
    # math reproduces the edge_pot_map scores: edge_score = dist × (1/cnt).
    # Choose dist = sqrt(edge_pot_map[(u,v)]) and cnt = 1 so score lines up.
    def _batch_deltas(pattern_id: str, version: int, keys: set[str]):  # type: ignore[no-untyped-def]
        result = {}
        for (u, v), s in edge_pot_map.items():
            # assign deltas on a 1-D axis such that |δ_u − δ_v| == s
            # Use u,v names as offsets to avoid collisions.
            if u not in result:
                result[u] = __import__("numpy").asarray([0.0])
            if v not in result:
                result[v] = __import__("numpy").asarray([float(s)])
        return {k: result[k] for k in keys if k in result}
    nav._batch_read_deltas = _batch_deltas  # type: ignore[assignment]

    # Pair counts: one each direction so effective bidirectional sum is 1
    # when only one direction exists, 2 when both exist. For cycle_2 edges
    # the bidirectional sum = 2, so effective = 2, weight = 0.5, edge_score
    # = distance × 0.5. To match edge_pot_map[(u,v)] = s: distance = 2s.
    # Rebuild _batch_deltas accordingly:
    def _batch_deltas_v2(pattern_id: str, version: int, keys: set[str]):  # type: ignore[no-untyped-def]
        import numpy as _np
        result = {}
        for (u, v), s in edge_pot_map.items():
            if u not in result:
                result[u] = _np.asarray([0.0])
            if v not in result:
                result[v] = _np.asarray([float(s) * 2.0])
        return {k: result[k] for k in keys if k in result}
    nav._batch_read_deltas = _batch_deltas_v2  # type: ignore[assignment]

    return nav


class TestScoreMotifDispatch:

    def test_invalid_motif_type_raises_with_valid_values(self) -> None:
        nav = _nav_with_mocked_pipeline([], {})
        with pytest.raises(GDSNavigationError) as exc:
            nav.score_motif("A", motif_type="pentagon", pattern_id="tx")
        msg = str(exc.value)
        assert "fan_out" in msg
        assert "cycle_2" in msg
        assert "cycle_3" in msg
        assert "structuring" in msg

    def test_cycle_2_score_matches_product_of_edge_potentials(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("B", "A", _hour_us(1))],
            {("A", "B"): 3.0, ("B", "A"): 4.0},
        )
        result = nav.score_motif("A", motif_type="cycle_2", pattern_id="tx")
        assert result["found"] is True
        assert result["score"] == pytest.approx(12.0)
        assert result["motif_type"] == "cycle_2"

    def test_no_motif_instance_returns_graceful_not_found(self) -> None:
        nav = _nav_with_mocked_pipeline([("A", "B", _hour_us(0))], {})
        result = nav.score_motif("A", motif_type="cycle_2", pattern_id="tx")
        assert result["found"] is False
        assert result["score"] == 0.0
        assert "reason" in result

    def test_score_motif_uses_default_window_when_unspecified(self) -> None:
        # cycle_2 default = 24h. ts_ab=0, ts_ba=20h → within 24h default.
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("B", "A", _hour_us(20))],
            {("A", "B"): 2.0, ("B", "A"): 2.0},
        )
        result = nav.score_motif("A", motif_type="cycle_2", pattern_id="tx")
        assert result["found"] is True

    def test_score_motif_picks_best_instance_when_multiple(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [
                ("A", "B", _hour_us(0)), ("B", "A", _hour_us(1)),
                ("A", "C", _hour_us(2)), ("C", "A", _hour_us(3)),
            ],
            {
                ("A", "B"): 2.0, ("B", "A"): 3.0,       # score = 6
                ("A", "C"): 10.0, ("C", "A"): 10.0,     # score = 100
            },
        )
        result = nav.score_motif("A", motif_type="cycle_2", pattern_id="tx")
        assert result["score"] == pytest.approx(100.0)
        assert result["counterparty"] == "C"

    def test_score_motif_structuring_single_seed(self) -> None:
        # Single-seed path uses _enumerate_structuring via storage.read_edges
        # BTREE, NOT the AdjacencyIndex fast-path used by _rank_motifs.
        # Mock read_edges to include an `amount` column (unlike the default
        # _mock_edges helper which drops amount — _enumerate_structuring
        # raises if the column is missing). Timestamps are float-seconds to
        # match the production EDGE_TABLE_SCHEMA (pa.float64 Unix seconds).
        rows_with_amt = [
            ("A", "B", 0.0, 15000.0),
            ("B", "C", 720.0, 5000.0),       # 12 min after A→B
            ("C", "D", 1440.0, 5000.0),      # 24 min after B→C (total 24 min within 1h window)
        ]
        storage = MagicMock()
        storage.has_edge_table = MagicMock(return_value=True)

        def _read_edges(
            pattern_id, from_keys=None, to_keys=None,
            timestamp_from=None, timestamp_to=None, columns=None,
        ):
            filtered = rows_with_amt
            if from_keys is not None:
                filtered = [r for r in filtered if r[0] in from_keys]
            if to_keys is not None:
                filtered = [r for r in filtered if r[1] in to_keys]
            return pa.table({
                "from_key": [r[0] for r in filtered],
                "to_key": [r[1] for r in filtered],
                "timestamp": [r[2] for r in filtered],
                "amount": [r[3] for r in filtered],
            })
        storage.read_edges = MagicMock(side_effect=_read_edges)

        sphere = MagicMock()
        event_pattern = MagicMock()
        event_pattern.pattern_type = "event"
        sphere.patterns = {"tx": event_pattern}
        storage.read_sphere = MagicMock(return_value=sphere)

        nav = GDSNavigator(MagicMock(), storage, MagicMock(), MagicMock())

        # Stub edge_potential so the product composes cleanly.
        def _ep(from_key, to_key, pattern_id):  # type: ignore[no-untyped-def]
            return {
                "from_key": from_key, "to_key": to_key, "pattern_id": pattern_id,
                "score": 2.0, "delta_distance": 1.0,
                "pair_tx_count": 1, "effective_weight": 1.0,
            }
        nav.edge_potential = _ep  # type: ignore[assignment]

        result = nav.score_motif(
            "A", motif_type="structuring", pattern_id="tx",
            time_window_hours=1,
        )
        assert result["found"] is True
        assert result["motif_type"] == "structuring"
        assert result["path"] == ["A", "B", "C", "D"]
        assert result["edges"] == [("A", "B"), ("B", "C"), ("C", "D")]
        assert result["amounts"] == [15000.0, 5000.0, 5000.0]
        # score = product of 2.0 across 3 edges = 8.0
        assert result["score"] == pytest.approx(8.0)

    def test_score_motif_structuring_requires_amount_column(self) -> None:
        # If the edge table has no amount column, _enumerate_structuring must
        # raise GDSNavigationError with a clear message.
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("B", "C", _hour_us(1)), ("C", "D", _hour_us(2))],
            {},
        )
        with pytest.raises(GDSNavigationError) as exc:
            nav.score_motif("A", motif_type="structuring", pattern_id="tx")
        assert "amount" in str(exc.value).lower()


# -----------------------------------------------------------------------------
# Task 6: find_high_potential_motifs + LRU ranking cache
# -----------------------------------------------------------------------------


class TestFindHighPotentialMotifs:

    def test_global_ranking_returns_sorted_desc(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [
                ("A", "B", _hour_us(0)), ("B", "A", _hour_us(1)),
                ("C", "D", _hour_us(0)), ("D", "C", _hour_us(1)),
            ],
            {
                ("A", "B"): 1.0, ("B", "A"): 1.0,   # score = 1
                ("C", "D"): 5.0, ("D", "C"): 5.0,   # score = 25
            },
        )
        # Patch geometry read for seed discovery; return all unique from_keys.
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B", "C", "D"]})
        )
        # And resolve_version stub.
        nav._resolve_version = MagicMock(return_value=1)
        result = nav.find_high_potential_motifs(
            "tx", motif_type="cycle_2", top_n=10,
        )
        assert len(result) >= 2
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0]["score"] == pytest.approx(25.0)

    def test_ranking_cache_is_version_keyed(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("B", "A", _hour_us(1))],
            {("A", "B"): 1.0, ("B", "A"): 1.0},
        )
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        nav.find_high_potential_motifs("tx", motif_type="cycle_2")
        # Spy on get_adjacency — a ranking-cache hit skips the body entirely
        # (no adjacency fetch, no scoring). A miss re-executes the body.
        first_calls = nav._storage.get_adjacency.call_count
        nav.find_high_potential_motifs("tx", motif_type="cycle_2")
        assert nav._storage.get_adjacency.call_count == first_calls

        # New version invalidates the ranking cache.
        nav._resolve_version = MagicMock(return_value=2)
        nav.find_high_potential_motifs("tx", motif_type="cycle_2")
        assert nav._storage.get_adjacency.call_count > first_calls

    def test_ranking_lru_caps_at_eight(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("B", "A", _hour_us(1))],
            {("A", "B"): 1.0, ("B", "A"): 1.0},
        )
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        for window in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            nav.find_high_potential_motifs(
                "tx", motif_type="cycle_2", time_window_hours=window,
            )
        assert len(nav._motif_ranking_cache) == 8

    def test_ranking_adds_score_rank_pct_and_is_high_potential(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [
                ("A", "B", _hour_us(0)), ("B", "A", _hour_us(1)),
                ("C", "D", _hour_us(0)), ("D", "C", _hour_us(1)),
            ],
            {
                ("A", "B"): 1.0, ("B", "A"): 1.0,
                ("C", "D"): 5.0, ("D", "C"): 5.0,
            },
        )
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B", "C", "D"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        result = nav.find_high_potential_motifs("tx", motif_type="cycle_2", top_n=10)
        assert all("score_rank_pct" in r for r in result)
        assert all("is_high_potential" in r for r in result)
        assert result[0]["score_rank_pct"] == 100.0

    def test_invalid_motif_type_in_ranking_raises(self) -> None:
        nav = _nav_with_mocked_pipeline([], {})
        with pytest.raises(GDSNavigationError):
            nav.find_high_potential_motifs("tx", motif_type="heptagon")

    def test_cycle_2_ranking_dedupes_pair_across_seed_directions(self) -> None:
        # A<->B bidirectional pair should appear once, not twice (once per seed).
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("B", "A", _hour_us(1))],
            {("A", "B"): 1.0, ("B", "A"): 1.0},
        )
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        result = nav.find_high_potential_motifs("tx", motif_type="cycle_2", top_n=10)
        assert len(result) == 1
        pair = tuple(sorted([result[0]["seed"], result[0]["counterparty"]]))
        assert pair == ("A", "B")


class TestAdjacencyReuseRegression:
    """Regression guards for the 0.5.0 AdjacencyIndex reuse refactor."""

    def test_rank_motifs_uses_adjacency_not_direct_read_edges(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("B", "A", _hour_us(1))],
            {("A", "B"): 1.0, ("B", "A"): 1.0},
        )
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        nav._storage.read_edges.reset_mock()
        nav.find_high_potential_motifs("tx", motif_type="cycle_2", top_n=10)
        assert nav._storage.get_adjacency.call_count >= 1
        assert nav._storage.read_edges.call_count == 0

    def test_edge_potential_reuses_adjacency_pair_counts(self) -> None:
        nav = _nav_with_mocked_pipeline(
            [("A", "B", _hour_us(0)), ("A", "B", _hour_us(1)), ("C", "D", _hour_us(2))],
            {},
        )
        nav._resolve_version = MagicMock(return_value=1)
        counts = nav._pair_count_for_pattern("tx", 1)
        assert counts == {("A", "B"): 2, ("C", "D"): 1}
        assert nav._storage.get_adjacency.call_count >= 1

    def test_rank_motifs_inherits_adj_temporal_window(self) -> None:
        # Window = 24h; 10 outgoing edges spread 20h apart.
        # Only edges with ts >= max_ts - 24h survive: T8 (h=160) and T9 (h=180).
        import numpy as _np
        rows = [
            ("H", f"T{i}", _hour_us(float(i * 20)))
            for i in range(10)
        ]
        nav = _nav_with_mocked_pipeline(rows, {})
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["H"] + [f"T{i}" for i in range(10)]})
        )
        nav._resolve_version = MagicMock(return_value=1)

        def _deltas(pattern_id, version, keys):  # type: ignore[no-untyped-def]
            return {k: _np.asarray([0.0, 1.0]) for k in keys}
        nav._batch_read_deltas = _deltas  # type: ignore[assignment]

        result = nav.find_high_potential_motifs(
            "tx", motif_type="fan_out", time_window_hours=24, min_k=2, top_n=5,
        )
        assert len(result) == 1
        targets = {v for (_u, v) in result[0]["edges"]}
        assert targets == {"T8", "T9"}

    def test_rank_motifs_empty_adjacency_returns_empty_without_crash(self) -> None:
        nav = _nav_with_mocked_pipeline([], {})
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": []}, schema=pa.schema(
                [("primary_key", pa.string())]
            ))
        )
        nav._resolve_version = MagicMock(return_value=1)
        result = nav.find_high_potential_motifs("tx", motif_type="cycle_2", top_n=10)
        assert result == []


class TestStructuringEnumerator:
    """Runtime structuring motif — A→B→C→D with amount predicates.

    Autoresearch E4 target on AML HI-small: +180 TP / +5.25 pp recall
    when cycle_3 caught 0/0. Structuring is open-chain 3-hop with
    hop1 amount ≥ amt1_min and hop2, hop3 amount ≤ amt2_max.
    """

    def _mk_nav(self, amounts, hours=(0.0, 0.2, 0.4), extra_rows=None):
        import numpy as _np
        rows = [
            ("A", "B", _hour_us(hours[0])),
            ("B", "C", _hour_us(hours[1])),
            ("C", "D", _hour_us(hours[2])),
        ]
        if extra_rows:
            rows.extend(extra_rows)
        nav = _nav_with_mocked_pipeline(rows, {}, amounts=amounts)
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B", "C", "D"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        def _deltas(pattern_id, version, keys):  # type: ignore[no-untyped-def]
            return {k: _np.asarray([0.0, 1.0]) for k in keys}
        nav._batch_read_deltas = _deltas  # type: ignore[assignment]
        return nav

    def test_structuring_basic_match(self) -> None:
        nav = self._mk_nav({("A", "B"): 15000, ("B", "C"): 5000, ("C", "D"): 5000})
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert len(result) == 1
        assert result[0]["motif_type"] == "structuring"
        assert result[0]["seed"] == "A"
        assert result[0]["path"] == ["A", "B", "C", "D"]
        assert result[0]["edges"] == [("A", "B"), ("B", "C"), ("C", "D")]
        assert result[0]["amounts"] == [15000, 5000, 5000]

    def test_structuring_rejects_small_first_hop(self) -> None:
        # hop1 < amt1_min=10000 → no match
        nav = self._mk_nav({("A", "B"): 5000, ("B", "C"): 5000, ("C", "D"): 5000})
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result == []

    def test_structuring_rejects_large_later_hop(self) -> None:
        # hop2 above amt2_max=10000 → no match
        nav = self._mk_nav({("A", "B"): 15000, ("B", "C"): 15000, ("C", "D"): 5000})
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result == []

    def test_structuring_temporal_ordering_strict(self) -> None:
        # ts_bc < ts_ab → reject (violates monotonic hop ordering)
        nav = self._mk_nav(
            {("A", "B"): 15000, ("B", "C"): 5000, ("C", "D"): 5000},
            hours=(0.5, 0.1, 0.4),  # ts_bc=0.1h < ts_ab=0.5h
        )
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result == []

    def test_structuring_time_window_enforced(self) -> None:
        # total span 3h > time_window_hours=1 → reject
        nav = self._mk_nav(
            {("A", "B"): 15000, ("B", "C"): 5000, ("C", "D"): 5000},
            hours=(0.0, 1.5, 3.0),
        )
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result == []

    def test_structuring_dedup_on_canonical_path(self) -> None:
        # Two (A,B) edge rows with same endpoints at different ts produce
        # the same canonical path A→B→C→D; dedup keeps 1.
        import numpy as _np
        rows = [
            ("A", "B", _hour_us(0.0)),
            ("A", "B", _hour_us(0.1)),  # duplicate pair at different ts
            ("B", "C", _hour_us(0.3)),
            ("C", "D", _hour_us(0.5)),
        ]
        amounts = {("A", "B"): 15000, ("B", "C"): 5000, ("C", "D"): 5000}
        nav = _nav_with_mocked_pipeline(rows, {}, amounts=amounts)
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B", "C", "D"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        def _deltas(pattern_id, version, keys):  # type: ignore[no-untyped-def]
            return {k: _np.asarray([0.0, 1.0]) for k in keys}
        nav._batch_read_deltas = _deltas  # type: ignore[assignment]
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert len(result) == 1

    def test_structuring_custom_thresholds(self) -> None:
        # amounts (60000, 5000, 5000): matches with default (amt1_min=10000).
        # With amt1_min=50000 should still match. With amt1_min=100000 should reject.
        nav = self._mk_nav({("A", "B"): 60000, ("B", "C"): 5000, ("C", "D"): 5000})
        with_default = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert len(with_default) == 1

        # Fresh nav for re-run to bypass motif ranking LRU cache
        nav2 = self._mk_nav({("A", "B"): 60000, ("B", "C"): 5000, ("C", "D"): 5000})
        with_high_min = nav2.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1,
            amt1_min=100000.0, top_n=10,
        )
        assert with_high_min == []

    def test_structuring_no_self_visits(self) -> None:
        # D = A (loop back to seed) — reject per guard
        import numpy as _np
        rows = [
            ("A", "B", _hour_us(0.0)),
            ("B", "C", _hour_us(0.2)),
            ("C", "A", _hour_us(0.4)),  # closes back to seed — rejected
        ]
        amounts = {("A", "B"): 15000, ("B", "C"): 5000, ("C", "A"): 5000}
        nav = _nav_with_mocked_pipeline(rows, {}, amounts=amounts)
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B", "C"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        def _deltas(pattern_id, version, keys):  # type: ignore[no-untyped-def]
            return {k: _np.asarray([0.0, 1.0]) for k in keys}
        nav._batch_read_deltas = _deltas  # type: ignore[assignment]
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result == []

    def test_structuring_null_amount_skipped_not_crashed(self) -> None:
        # EDGE_TABLE_SCHEMA amount column is nullable. Inject a NULL amount
        # on hop 2 (B→C) — the enumerator must SKIP the edge, not:
        #   (a) crash on `None >= amt1_min` / `None <= amt2_max`
        #   (b) treat None as 0.0 and falsely pass `<= amt2_max`.
        # Only the valid non-null path should survive.
        import numpy as _np
        from hypertopos.engine.adjacency import AdjacencyIndex

        rows = [
            ("A", "B", _hour_us(0.0)),
            ("B", "C", _hour_us(0.2)),  # NULL amount — must be skipped
            ("C", "D", _hour_us(0.4)),
        ]
        # Build adjacency manually so we can inject None on one hop — the
        # helper zeroes NULLs to 1.0 so can't exercise this via the shortcut.
        adj = AdjacencyIndex.from_edge_lists(
            from_keys=["A", "B", "C"],
            to_keys=["B", "C", "D"],
            timestamps=[_hour_us(0.0) / 1_000_000.0,
                        _hour_us(0.2) / 1_000_000.0,
                        _hour_us(0.4) / 1_000_000.0],
            amounts=[15000.0, None, 5000.0],  # type: ignore[list-item]
            event_keys=["e0", "e1", "e2"],
        )
        nav = _enum_nav(rows)
        nav._storage.get_adjacency = MagicMock(return_value=adj)
        nav._storage.read_geometry = MagicMock(
            return_value=pa.table({"primary_key": ["A", "B", "C", "D"]})
        )
        nav._resolve_version = MagicMock(return_value=1)
        def _deltas(pattern_id, version, keys):  # type: ignore[no-untyped-def]
            return {k: _np.asarray([0.0, 1.0]) for k in keys}
        nav._batch_read_deltas = _deltas  # type: ignore[assignment]

        # Should not crash. Should return 0 matches because the B→C hop
        # is NULL-amount and gets skipped, breaking the chain.
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result == []

    def test_structuring_negative_threshold_rejected(self) -> None:
        # Validation: amt1_min or amt2_max ≤ 0 → GDSNavigationError.
        nav = self._mk_nav({("A", "B"): 15000, ("B", "C"): 5000, ("C", "D"): 5000})
        with pytest.raises(GDSNavigationError):
            nav.find_high_potential_motifs(
                "tx", motif_type="structuring", time_window_hours=1,
                amt1_min=0.0, top_n=10,
            )
        with pytest.raises(GDSNavigationError):
            nav.find_high_potential_motifs(
                "tx", motif_type="structuring", time_window_hours=1,
                amt2_max=-100.0, top_n=10,
            )

    def test_structuring_rejects_nonpositive_hop_amounts(self) -> None:
        # Refund/reversal amount (≤ 0) on hop 2 must be rejected even though
        # it would otherwise satisfy "≤ amt2_max". Structuring is a positive
        # money flow by definition.
        nav = self._mk_nav(
            {("A", "B"): 15000, ("B", "C"): -500, ("C", "D"): 5000}
        )
        result = nav.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result == []

        # Same for zero-amount hop 3.
        nav2 = self._mk_nav(
            {("A", "B"): 15000, ("B", "C"): 5000, ("C", "D"): 0}
        )
        result2 = nav2.find_high_potential_motifs(
            "tx", motif_type="structuring", time_window_hours=1, top_n=10,
        )
        assert result2 == []
