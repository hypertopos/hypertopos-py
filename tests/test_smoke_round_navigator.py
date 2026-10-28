"""Navigator fixes from the live smoke round on a large AML sphere."""

from __future__ import annotations

from hypertopos.navigation.navigator import GDSNavigator

_FAILED = {"ok": False, "error": "GDSNavigationError: ambiguous chain_id"}


def _shape_ok() -> dict:
    return {"ok": True, "data": {"is_anomaly": True, "score_rank_pct": 99.2}}


class TestInvestigationSummaryWithFailedSteps:
    def test_failed_steps_are_inconclusive_not_false_positive(self) -> None:
        summary = GDSNavigator._derive_investigation_summary(
            trace_block=_FAILED,
            typology_block=_FAILED,
            shape_anomaly=_shape_ok(),
            extension_forward=_FAILED,
            extension_backward=_FAILED,
        )
        assert summary["investigation_strength"] == "inconclusive"
        assert summary["recommended_action"] != "false-positive candidate"
        assert summary["failed_steps"] == [
            "trace",
            "typology",
            "extension_forward",
            "extension_backward",
        ]

    def test_all_steps_ok_without_signal_stays_weak(self) -> None:
        empty = {"ok": True, "data": {"summary": {"n_anomalous_candidates": 0}}}
        summary = GDSNavigator._derive_investigation_summary(
            trace_block={"ok": True, "data": {"summary": {"max_run_length_same_top_dim": 1}}},
            typology_block={"ok": True, "data": {"typology": {"position_in_chain": "no-run"}}},
            shape_anomaly=_shape_ok(),
            extension_forward=empty,
            extension_backward=empty,
        )
        assert summary["investigation_strength"] == "weak"
        assert summary["recommended_action"] == "false-positive candidate"
        assert "failed_steps" not in summary

    def test_strong_score_stands_despite_one_failed_step(self) -> None:
        hit = {"ok": True, "data": {"summary": {"n_anomalous_candidates": 2}}}
        summary = GDSNavigator._derive_investigation_summary(
            trace_block={
                "ok": True,
                "data": {"summary": {"max_run_length_same_top_dim": 4, "dominant_top_dim": "x"}},
            },
            typology_block={
                "ok": True,
                "data": {"typology": {"position_in_chain": "transit", "shape": "peak"}},
            },
            shape_anomaly=_shape_ok(),
            extension_forward=hit,
            extension_backward=_FAILED,
        )
        assert summary["investigation_strength"] == "strong"
        assert summary["failed_steps"] == ["extension_backward"]


class TestReaderKeepsRelationNames:
    def test_relation_display_name_and_interpretation_are_read(self, tmp_path) -> None:
        import json
        from pathlib import Path

        from hypertopos.storage.reader import GDSReader

        src = Path(__file__).parent / "fixtures" / "gds" / "sales_sphere" / "_gds_meta"
        raw = json.loads((src / "sphere.json").read_text(encoding="utf-8"))
        pid, pattern = next((k, p) for k, p in raw["patterns"].items() if p["relations"])
        pattern["relations"][0]["display_name"] = "sender"
        pattern["relations"][0]["interpretation"] = "who pays"
        meta = tmp_path / "_gds_meta"
        meta.mkdir()
        (meta / "sphere.json").write_text(json.dumps(raw), encoding="utf-8")

        rel = GDSReader(str(tmp_path)).read_sphere().patterns[pid].relations[0]
        # The declared name is the relation's role; display_name stays the
        # semantics overlay, so the dim label of a unique line is unchanged.
        assert rel.role == "sender"
        assert rel.display_name is None
        assert rel.interpretation == "who pays"


class TestRelationDimLabels:
    def test_role_labels_only_a_repeated_line_and_overlay_always_labels(self) -> None:
        from hypertopos.model.sphere import RelationDef, relation_dim_labels

        rels = [
            RelationDef("accounts", "in", True, role="sender"),
            RelationDef("accounts", "in", False, role="receiver"),
            RelationDef("_d_amount_sum", "in", False, role="amount_sum"),
            RelationDef("products", "in", False, display_name="bought products"),
            RelationDef("currencies", "in", False),
        ]
        assert relation_dim_labels(rels) == [
            "sender",
            "receiver",
            "_d_amount_sum",
            "bought products",
            "currencies",
        ]


class TestSegmentShiftMultipleTesting:
    def test_entity_p_is_corrected_for_every_segment_tested(self, monkeypatch) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import pyarrow as pa

        from hypertopos.engine.p_value_calibration import detector_p_value_segment_shift

        n = 200
        pks = [f"E{i:03d}" for i in range(n)]
        # Segment "hot" (first 50) holds 30 anomalies, "cold" holds 5.
        is_anom = [i < 30 or 100 <= i < 105 for i in range(n)]
        col_a = ["hot" if i < 50 else "cold" for i in range(n)]
        col_b = ["left" if i % 2 else "right" for i in range(n)]
        storage = MagicMock()
        storage.read_geometry.return_value = pa.table({"primary_key": pks, "is_anomaly": is_anom})
        line = SimpleNamespace(
            columns=[
                SimpleNamespace(name="seg_a", type="string"),
                SimpleNamespace(name="seg_b", type="string"),
            ]
        )
        sphere = MagicMock()
        sphere.entity_line.return_value = "acc"
        sphere.lines = {"acc": line}
        storage.read_sphere.return_value = sphere
        storage.read_points.return_value = pa.table(
            {"primary_key": pks, "seg_a": col_a, "seg_b": col_b}
        )
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        nav._manifest = MagicMock()
        nav._manifest.line_version.return_value = 1
        monkeypatch.setattr(GDSNavigator, "_resolve_version", lambda self, pid: 1)

        out = nav._collect_segment_shift_p("p")

        pop_anom = sum(is_anom)
        raw_hot = detector_p_value_segment_shift(
            {
                "hot": {
                    "in_segment_anomalous": 30,
                    "in_segment_total": 50,
                    "out_segment_anomalous": pop_anom - 30,
                    "out_segment_total": n - 50,
                }
            }
        )["hot"]
        # Four segments tested over two columns.
        assert out["E000"] == min(1.0, raw_hot * 4)


class TestLabelAwareUnavailableReason:
    def test_reason_names_the_actual_gap(self) -> None:
        from types import SimpleNamespace

        from hypertopos.navigation.navigator import label_aware_unavailable_reason

        no_block = SimpleNamespace(label_audit=None)
        assert "no 'label_audit:' block" in label_aware_unavailable_reason(no_block, "acc")
        other = SimpleNamespace(label_audit={"patterns": ["tx"]})
        reason = label_aware_unavailable_reason(other, "acc")
        assert "does not list" in reason
        assert "tx" in reason
        covered = SimpleNamespace(label_audit={"patterns": ["acc"]})
        assert "rebuild" in label_aware_unavailable_reason(covered, "acc")


class TestAmbiguousChainIdEverywhere:
    def test_witness_intersection_refuses_an_ambiguous_chain_id(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import pyarrow as pa
        import pytest

        sphere = MagicMock()
        sphere.patterns = {
            "ch": SimpleNamespace(pattern_type="anchor"),
            "acc": SimpleNamespace(pattern_type="anchor"),
        }
        sphere.entity_line.return_value = "chains"
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        storage.read_points.return_value = pa.table(
            {"primary_key": ["C1", "C1"], "chain_keys": ["A,B,C", "D,E,F"]}
        )
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        nav._manifest = MagicMock()
        nav._manifest.line_version.return_value = 1
        with pytest.raises(ValueError, match="ambiguous chain_id"):
            nav.chain_witness_intersection("C1", chain_pattern="ch", member_pattern="acc")


class TestChainSummaryCounts:
    def test_chain_counts_are_distinct_chains_and_add_up(self, monkeypatch) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import pyarrow as pa

        diag = {
            "n_chains_total": 10,
            # C1 carries two coherent runs: 3 runs over 2 chains.
            "n_runs_total_pre_truncation": 3,
            "all_coherent_chain_ids": ["C1", "C1", "C2"],
            "top_dim_counts_full": {"x": 3},
            "run_length_distribution_full": {"p75": 3},
        }
        monkeypatch.setattr(
            GDSNavigator,
            "find_chains_with_coherent_anomaly",
            lambda self, *a, **k: {"diagnostics": diag, "chains": []},
        )
        monkeypatch.setattr(GDSNavigator, "_resolve_version", lambda self, pid: 1)
        storage = MagicMock()
        storage.read_sphere.return_value = SimpleNamespace(patterns={"ch": 1, "acc": 1})
        storage.read_geometry.return_value = pa.table(
            {"primary_key": ["C2", "C3"], "is_anomaly": [True, True]}
        )
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        out = nav.chain_investigation_summary("ch", anchor_pattern_id="acc")
        overlap = out["cross_pattern_overlap"]
        assert out["n_chains_with_coherent_anomaly_run"] == 2
        assert out["n_coherent_runs"] == 3
        assert overlap["n_both"] + overlap["n_coherent_only"] == 2
        assert overlap["n_both"] + overlap["n_shape_only"] == out["n_chains_with_shape_anomaly"]


class TestActiveSliceMask:
    def test_marks_windows_with_activity_on_varying_dims(self) -> None:
        import numpy as np

        from hypertopos.navigation.navigator import active_slice_mask

        # Second column is static (broadcast into every window).
        shapes = np.array([[1.0, 0.5], [0.0, 0.5], [0.0, 0.5]])
        assert active_slice_mask(shapes).tolist() == [True, False, False]
        assert active_slice_mask(np.ones((3, 2))).tolist() == [True, True, True]
        assert active_slice_mask(np.zeros((3, 2))).tolist() == [True, True, True]


class TestPairCountColdPath:
    def test_cold_adjacency_counts_one_pair_from_its_own_edges(self) -> None:
        from unittest.mock import MagicMock

        import pyarrow as pa

        storage = MagicMock()
        storage._adjacency_cache = {}

        def read_edges(pid, from_keys=None, columns=None, **kw):
            if from_keys == ["A"]:
                return pa.table({"to_key": ["B", "B", "C"]})
            if from_keys == ["B"]:
                return pa.table({"to_key": ["A", "D"]})
            return pa.table({"to_key": pa.array([], type=pa.string())})

        storage.read_edges.side_effect = read_edges
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        assert nav._pair_tx_count("tx", "A", "B") == 3  # A→B twice, B→A once
        assert nav._pair_tx_count("tx", "A", "A") == 0
        storage.get_adjacency.assert_not_called()


class TestContagionColdPath:
    def test_cold_adjacency_reads_only_the_entitys_edges(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        import pyarrow as pa

        storage = MagicMock()
        storage._adjacency_cache = {}
        storage.has_edge_table.return_value = True

        def read_edges(pid, from_keys=None, to_keys=None, timestamp_to=None, columns=None):
            if from_keys:
                return pa.table({"to_key": ["B", "C", "A"]})
            return pa.table({"from_key": ["C", "D"]})

        storage.read_edges.side_effect = read_edges
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        monkeypatch.setattr(
            GDSNavigator, "_resolve_anchor_pattern_for_scoring", lambda self, pid: "acc"
        )
        monkeypatch.setattr(
            GDSNavigator,
            "check_anomaly_batch",
            lambda self, pid, keys, max_keys=500: {"anomalous_count": 1},
        )
        out = nav.contagion_score("A", "tx")
        storage.get_adjacency.assert_not_called()
        assert out["total_counterparties"] == 3  # B, C, D — A itself excluded
        assert out["anomalous_counterparties"] == 1


class TestCounterpartiesSelfTransfers:
    def test_self_transfers_are_not_listed_as_a_counterparty(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        from hypertopos.engine.adjacency import AdjacencyIndex
        from hypertopos.navigation.navigator import GDSNavigationError

        adj = AdjacencyIndex.from_edge_lists(
            ["A", "A", "A", "B"],
            ["A", "B", "B", "A"],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 1.0, 1.0, 1.0],
            ["E1", "E2", "E3", "E4"],
        )
        storage = MagicMock()
        storage._adjacency_cache = {"tx": adj}
        storage.get_adjacency.return_value = adj
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        monkeypatch.setattr(
            GDSNavigator, "_resolve_anchor_pattern_for_scoring", lambda self, pid: None
        )

        def _no_geometry(self, pid):
            raise GDSNavigationError("no geometry")

        monkeypatch.setattr(GDSNavigator, "_resolve_version", _no_geometry)
        out = nav._find_counterparties_via_edges("A", "accounts", "tx")
        assert [e["key"] for e in out["outgoing"]] == ["B"]
        assert [e["key"] for e in out["incoming"]] == ["B"]
        assert out["summary"]["self_transfers"] == 1


class TestSolidForecastSubspace:
    def test_sliced_dims_are_compared_with_their_own_threshold(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import numpy as np

        # Slices carry 2 dims; the pattern has 3 (the third is edge-derived).
        slices = [
            SimpleNamespace(delta_snapshot=np.array([v, v]), timestamp=None)
            for v in (1.0, 1.5, 2.0, 2.5)
        ]
        engine = MagicMock()
        engine.build_solid.return_value = SimpleNamespace(slices=slices)
        pattern = SimpleNamespace(theta=np.array([1.0, 1.0, 10.0]), theta_norm=10.1)
        storage = MagicMock()
        storage.read_sphere.return_value = SimpleNamespace(patterns={"p": pattern})
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._engine = engine
        nav._storage = storage
        nav._manifest = MagicMock()

        out = nav.solid_forecast("E", "p", current_delta_norm=4.0)
        assert out["forecast_dims"] == 2
        assert out["forecast_theta_norm"] == round(float(np.sqrt(2.0)), 4)
        assert out["forecast_is_anomaly"] is True
        # The current state is judged in the full space.
        assert out["current_is_anomaly"] is False


class TestGeometricPathDedupe:
    def test_same_key_path_found_through_two_meeting_points_is_returned_once(
        self, monkeypatch
    ) -> None:
        from unittest.mock import MagicMock

        from hypertopos.engine.adjacency import AdjacencyIndex

        adj = AdjacencyIndex.from_edge_lists(
            ["A", "B", "C"], ["B", "C", "D"], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], ["E1", "E2", "E3"]
        )
        storage = MagicMock()
        storage.has_edge_table.return_value = True
        storage.get_adjacency.return_value = adj
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        monkeypatch.setattr(
            GDSNavigator, "_resolve_anchor_pattern_for_scoring", lambda self, pid: None
        )
        out = nav.find_geometric_path("A", "D", "tx", max_depth=4, scoring="shortest")
        keys = [tuple(p["keys"]) for p in out["paths"]]
        assert keys == [("A", "B", "C", "D")]


class TestLeadLagBudgetMessage:
    def test_budget_error_does_not_suggest_the_cohort_already_in_use(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import numpy as np
        import pytest

        pat = SimpleNamespace(pattern_type="anchor", mu=np.zeros(40))
        sphere = MagicMock()
        sphere.patterns = {"a": pat, "b": pat}
        storage = MagicMock()
        storage.read_sphere.return_value = sphere
        storage.count_geometry_rows.return_value = 5_000_000
        nav = GDSNavigator.__new__(GDSNavigator)
        nav._storage = storage
        nav._manifest = SimpleNamespace(pattern_versions={"a": 1, "b": 1})
        with pytest.raises(ValueError, match="budget") as exc:
            nav.find_lead_lag("a", "b", cohort="fixed")
        assert "cohort='fixed'" not in str(exc.value)
        assert "entity_key=" not in str(exc.value)


class TestConsensusIncludesFocalEntity:
    @staticmethod
    def _nav():
        from test_navigation import (
            Contract,
            _make_multi_detector_manifest,
            _MockEngine,
            _MockStorageMultiDetector,
        )

        class _SamplingStorage(_MockStorageMultiDetector):
            """Honours sample_size by keeping the first rows, like a sample that
            happened to miss the last entity."""

            def read_geometry(self, *args, sample_size=None, point_keys=None, **kwargs):
                tbl = super().read_geometry(*args, point_keys=point_keys, **kwargs)
                if sample_size and point_keys is None:
                    tbl = tbl.slice(0, sample_size)
                return tbl

        return GDSNavigator(
            engine=_MockEngine(),
            storage=_SamplingStorage(),
            manifest=_make_multi_detector_manifest(),
            contract=Contract("m-multi", []),
        )

    def test_focal_entity_outside_sample_is_scored(self) -> None:
        nav = self._nav()
        without = nav.classify_detector_consensus("acc_pat", sample_size=2, top_n=50)
        assert "E" not in {r["primary_key"] for r in without}
        with_focal = nav.classify_detector_consensus(
            "acc_pat", sample_size=2, top_n=50, include_keys=["E"]
        )
        assert "E" in {r["primary_key"] for r in with_focal}
