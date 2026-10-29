# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Population counts on a calibrated pattern follow the stored ``is_anomaly``.

``find_anomalies`` selects a calibrated pattern by its stored ``is_anomaly`` (the
alarm budget); every other count of "anomalous entities" must agree with it,
not with ``delta_norm >= θ``.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from test_builder_robust import _builder

from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb

PID = "account_pattern"


@pytest.fixture(scope="module")
def sphere_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("calcounts") / "sphere"
    _builder(out).build()
    lb.create_branch(out, "exp")
    return out


@pytest.fixture(scope="module")
def flagged(sphere_dir: Path) -> np.ndarray:
    geo = lance.dataset(str(sphere_dir / "geometry" / PID / "data.lance")).to_table(
        columns=["is_anomaly", "delta_norm"]
    )
    is_anomaly = geo["is_anomaly"].to_numpy(zero_copy_only=False).astype(bool)
    return is_anomaly


@pytest.fixture(scope="module")
def nav(sphere_dir: Path):  # type: ignore[no-untyped-def]
    return HyperSphere.open(str(sphere_dir)).session("t").navigator()


def test_fixture_separates_the_two_rules(sphere_dir: Path, flagged: np.ndarray, nav) -> None:  # type: ignore[no-untyped-def]
    pattern = nav._storage.read_sphere().patterns[PID]
    theta_norm = float(np.linalg.norm(pattern.theta))
    dn = (
        lance.dataset(str(sphere_dir / "geometry" / PID / "data.lance"))
        .to_table(columns=["delta_norm"])["delta_norm"]
        .to_numpy()
    )
    assert flagged.any()
    assert int((dn >= theta_norm).sum()) != int(flagged.sum())


def test_find_anomalies_total_is_the_stored_flag(flagged: np.ndarray, nav) -> None:  # type: ignore[no-untyped-def]
    _, total, _, _ = nav.π5_attract_anomaly(PID, top_n=5)
    assert total == int(flagged.sum())


def test_anomaly_summary_counts_the_stored_flag(flagged: np.ndarray, nav) -> None:  # type: ignore[no-untyped-def]
    summary = nav.anomaly_summary(PID)
    assert summary["total_anomalies"] == int(flagged.sum())
    assert sum(c["count"] for c in summary["clusters"]) == int(flagged.sum())


def test_aggregate_anomalies_counts_the_stored_flag(flagged: np.ndarray, nav) -> None:  # type: ignore[no-untyped-def]
    out = nav.aggregate_anomalies(PID, group_by="segment")
    assert out["total_anomalies"] == int(flagged.sum())


def test_sphere_overview_rate_is_the_stored_flag(flagged: np.ndarray, nav) -> None:  # type: ignore[no-untyped-def]
    (entry,) = nav.sphere_overview(PID)
    assert entry["anomaly_rate"] == round(float(flagged.mean()), 4)
    assert entry["anomaly_rate_source"] == "is_anomaly"


def test_data_quality_skips_the_theta_ceiling_check(nav) -> None:  # type: ignore[no-untyped-def]
    issues = nav.detect_data_quality_issues(PID)
    assert all(i["issue_type"] != "theta_ceiling" for i in issues)


def test_compare_branches_counts_the_stored_flag(flagged: np.ndarray, nav) -> None:  # type: ignore[no-untyped-def]
    report = nav.compare_branches(PID, "exp")
    assert report.anomalies_from == report.anomalies_to == int(flagged.sum())
    assert report.flipped_in == report.flipped_out == 0
    assert report.jaccard == 1.0


def test_compare_branches_movers_report_the_calibrated_score(nav) -> None:  # type: ignore[no-untyped-def]
    report = nav.compare_branches(PID, "exp", top_n=3)
    assert report.top_movers
    for mover in report.top_movers:
        assert {"anomaly_score_from", "anomaly_score_to", "evidence_change"} <= set(mover)


def test_compare_branches_movers_rank_by_evidence_not_score_ratio(nav, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A tail entity gaining evidence must outrank a bottom entity whose small
    # anomaly_score moved by a large ratio.
    from hypertopos.storage.reader import GDSReader

    real_read = GDSReader.read_geometry

    def shifted(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        table = real_read(self, *args, **kwargs)
        if self.branch != "exp" or "conformal_p" not in table.column_names:
            return table
        keys = table["primary_key"].to_pylist()
        score = np.array(table["anomaly_score"].to_numpy(zero_copy_only=False), dtype=np.float64)
        p = np.array(table["conformal_p"].to_numpy(zero_copy_only=False), dtype=np.float64)
        low, high = int(np.argmin(score)), int(np.argmax(score))
        score[low] *= 3.0  # bottom: big ratio, no evidence
        p[high] /= 4.0  # tail: two more bits
        table = table.set_column(
            table.schema.get_field_index("anomaly_score"), "anomaly_score", pa.array(score)
        )
        table = table.set_column(
            table.schema.get_field_index("conformal_p"),
            "conformal_p",
            pa.array(p.astype(np.float32)),
        )
        shifted.keys = (keys[low], keys[high])  # type: ignore[attr-defined]
        return table

    monkeypatch.setattr(GDSReader, "read_geometry", shifted)
    report = nav.compare_branches(PID, "exp", top_n=1)
    _, tail = shifted.keys  # type: ignore[attr-defined]
    assert report.top_movers[0]["primary_key"] == tail
    assert report.top_movers[0]["evidence_change"] == pytest.approx(2.0, abs=1e-3)


def _stored(sphere_dir: Path, key: str) -> dict:  # type: ignore[type-arg]
    geo = lance.dataset(str(sphere_dir / "geometry" / PID / "data.lance")).to_table(
        columns=["primary_key", "conformal_p", "anomaly_score", "is_anomaly"],
        filter=f"primary_key = '{key}'",
    )
    return geo.to_pylist()[0]


def test_simulate_dimension_change_rescores_the_calibrated_score(sphere_dir: Path, nav) -> None:  # type: ignore[no-untyped-def]
    from test_builder_robust import DIMS, _natural_x

    key = "A010"
    keys = lance.dataset(str(sphere_dir / "geometry" / PID / "data.lance")).to_table(
        columns=["primary_key"]
    )["primary_key"]
    row = keys.to_pylist().index(key)
    natural = _natural_x(sphere_dir, keys)[row]
    stored = _stored(sphere_dir, key)

    same = nav.simulate_dimension_change(
        key, pattern_id=PID, line_id="accounts", set_dimension={DIMS[1]: float(natural[1])}
    )
    assert same["conformal_p_after"] == pytest.approx(stored["conformal_p"])
    assert same["is_anomaly_after"] == stored["is_anomaly"]
    assert same["held_axes"] == ["current"]
    assert "delta_norm_after" not in same  # one unit system: the calibrated layer
    (row_out,) = same["dimensions_overridden"]
    assert row_out["old_value"] == pytest.approx(natural[1])
    assert row_out["new_z"] == pytest.approx(row_out["old_z"])

    far = nav.simulate_dimension_change(
        key, pattern_id=PID, line_id="accounts", set_dimension={DIMS[1]: 1e9}
    )
    assert far["conformal_p_after"] < stored["conformal_p"]
    assert far["anomaly_score_after"] > stored["anomaly_score"]
    assert far["conformal_p_before"] == pytest.approx(stored["conformal_p"])
    assert far["top_dimensions_after"][0]["dim"] == DIMS[1]

    aliased = nav.simulate_dimension_change(
        key, pattern_id=PID, line_id="accounts", set_dimension={f"_d_{DIMS[1]}": 1e9}
    )
    assert aliased["conformal_p_after"] == pytest.approx(far["conformal_p_after"])


def test_diverse_explanations_validate_on_the_calibrated_score(nav) -> None:  # type: ignore[no-untyped-def]
    polygons, _, _, _ = nav.π5_attract_anomaly(PID, top_n=1)
    out = nav.find_diverse_explanations(polygons[0].primary_key, pattern_id=PID, validate=True)
    validations = [h["validation"] for h in out["hypotheses"] if "validation" in h]
    assert validations
    assert all("conformal_p_after_override" in v for v in validations)


def test_calibration_influencers_reject_a_calibrated_pattern(nav) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="calibrated"):
        nav.find_calibration_influencers(PID)


def test_find_anomalies_entries_carry_no_theta_influence_class(nav) -> None:  # type: ignore[no-untyped-def]
    entries = nav._attach_influence_fields_to_anomaly_entries([{"primary_key": "A000"}], PID)
    assert entries == [{"primary_key": "A000"}]
    assert not nav._influence_cache


def test_freshly_built_sphere_raises_no_alarm_rate_alert(nav, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Nothing was written since the calibration: the check reads no geometry.
    def _no_scan(*a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("geometry scanned")

    monkeypatch.setattr(nav._storage, "read_geometry", _no_scan)
    assert nav._check_alarm_rate_drift(PID) == []


def _ingested(nav, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Make the pattern look ingested into after its calibration."""
    sphere = nav._storage.read_sphere()
    pattern = sphere.patterns[PID]
    later = dataclasses.replace(pattern, computed_at=pattern.last_calibrated_at + timedelta(days=3))
    patched = dataclasses.replace(sphere, patterns={**sphere.patterns, PID: later})
    monkeypatch.setattr(nav._storage, "read_sphere", lambda *a, **k: patched)


def test_rows_since_calibration_flagged_above_the_budget_raise_an_alert(nav, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pattern = nav._storage.read_sphere().patterns[PID]
    later = pattern.last_calibrated_at.replace(tzinfo=None) + timedelta(days=3)
    n = 400
    flagged = np.zeros(n, dtype=bool)
    flagged[:40] = True  # 10 % against a 5 % budget
    table = pa.table(
        {
            "updated_at": pa.array([later] * n, type=pa.timestamp("us")),
            "is_anomaly": pa.array(flagged),
        }
    )
    _ingested(nav, monkeypatch)
    monkeypatch.setattr(nav._storage, "read_geometry", lambda *a, **k: table)
    (alert,) = nav._check_alarm_rate_drift(PID)
    assert alert["check_type"] == "alarm_rate_drift"
    assert alert["details"]["n_new"] == n
    assert alert["details"]["flagged"] == 40
    assert alert["details"]["p_value"] < 0.01
    within = table.set_column(1, "is_anomaly", pa.array(np.arange(n) < 20))
    monkeypatch.setattr(nav._storage, "read_geometry", lambda *a, **k: within)
    assert nav._check_alarm_rate_drift(PID) == []


def test_alarm_rate_alert_is_quiet_on_a_population_that_did_not_move(nav, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Many more new rows than the calibration holds: their flags share one
    # calibration, so a test that treats them as independent fires far too often.
    from hypertopos.engine.robust import fit_reference, is_anomaly, score

    _ingested(nav, monkeypatch)
    pattern = nav._storage.read_sphere().patterns[PID]
    later = pattern.last_calibrated_at.replace(tzinfo=None) + timedelta(days=3)
    fires = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        ref, _ = fit_reference(rng.normal(size=(500, 10)), tuple(f"d{i}" for i in range(10)))
        p = score(ref, rng.normal(size=(5_000, 10))).conformal_p
        table = pa.table(
            {
                "updated_at": pa.array([later] * len(p), type=pa.timestamp("us")),
                "is_anomaly": pa.array(is_anomaly(p, pattern.anomaly_percentile)),
            }
        )
        monkeypatch.setattr(nav._storage, "read_geometry", lambda *a, t=table, **k: t)
        monkeypatch.setattr(nav._storage, "read_reference", lambda *a, r=ref, **k: r)
        fires += bool(nav._check_alarm_rate_drift(PID))
    assert fires <= 3  # nominal 1 % of 40; a binomial test fires on ~18 %


def test_alarm_rate_alert_skips_a_reference_without_out_of_fold_calibration(
    nav, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    # Built before out-of-fold calibration: its new rows were calibrated in
    # sample and over-flag, so the budget test would fire on an unmoved population.
    _ingested(nav, monkeypatch)
    pattern = nav._storage.read_sphere().patterns[PID]
    later = pattern.last_calibrated_at.replace(tzinfo=None) + timedelta(days=3)
    n = 400
    table = pa.table(
        {
            "updated_at": pa.array([later] * n, type=pa.timestamp("us")),
            "is_anomaly": pa.array(np.arange(n) < 40),
        }
    )
    ref = nav._storage.read_reference(PID)
    assert ref.mahal_sorted_oof is not None
    old = dataclasses.replace(
        ref, mahal_sorted_oof=None, cal_sorted_oof=None, cal_sorted_no_current_oof=None
    )
    monkeypatch.setattr(nav._storage, "read_geometry", lambda *a, **k: table)
    monkeypatch.setattr(nav._storage, "read_reference", lambda *a, **k: old)
    assert nav._check_alarm_rate_drift(PID) == []


def test_find_anomalies_reports_no_theta_forecast_as_emerging(nav) -> None:  # type: ignore[no-untyped-def]
    _, _, emerging, _ = nav.π5_attract_anomaly(PID, top_n=3, include_emerging=True)
    assert emerging == []


def test_find_anomalies_dimensions_agree_with_explain(nav) -> None:  # type: ignore[no-untyped-def]
    polygons, _, _, _ = nav.π5_attract_anomaly(PID, top_n=3)
    for poly in polygons:
        explained = nav.explain_anomaly(poly.primary_key, PID)["top_dimensions"]
        assert poly.reliability_flags["dominant_dim"] == explained[0]["dim"]
        assert poly.anomaly_dimensions[0]["dim"] == explained[0]["dim"]
        assert sum(d["mahal_share"] for d in poly.anomaly_dimensions) <= 1.0 + 1e-9
