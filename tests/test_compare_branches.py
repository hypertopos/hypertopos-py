# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""GDSNavigator.compare_branches: anomaly and calibration diff between refs."""

from __future__ import annotations

from pathlib import Path

import lance
import pytest

from hypertopos.model.sphere import BranchDiffReport
from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.exceptions import GDSError
from tests.conftest import FIXTURES_PATH, clone_sphere

PID = "customer_pattern"


@pytest.fixture
def sphere_dir(tmp_path: Path) -> Path:
    base = clone_sphere(FIXTURES_PATH, tmp_path / "s")
    lb.create_branch(base, "exp")
    return base


def _keys(base: Path) -> list[str]:
    geo = base / "geometry" / PID / "data.lance"
    return lance.dataset(str(geo)).to_table(columns=["primary_key"])["primary_key"].to_pylist()


def test_identical_branch_has_no_diff(sphere_dir: Path) -> None:
    nav = HyperSphere.open(str(sphere_dir)).session("t").navigator()
    report = nav.compare_branches(PID, "exp")
    assert isinstance(report, BranchDiffReport)
    assert (report.from_ref, report.to_ref) == ("main", "exp")
    assert report.n_from == report.n_to == len(_keys(sphere_dir))
    assert report.flipped_in == report.flipped_out == 0
    assert report.jaccard == 1.0
    assert report.only_in_from == report.only_in_to == 0
    assert all(m["delta_norm_from"] == m["delta_norm_to"] for m in report.top_movers)
    assert report.calibration_drift is not None
    assert report.calibration_drift.overall_drift_rms == 0.0
    assert report.recipe == []


def test_branch_recalibration_shows_up_as_movers_and_drift(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    hs.session("b", branch="exp").recalibrate(PID, exclude_keys=[_keys(sphere_dir)[0]])
    report = hs.session("t").navigator().compare_branches(PID, "exp", top_n=1)
    assert len(report.top_movers) == 1
    mover = report.top_movers[0]
    assert set(mover) == {
        "primary_key",
        "delta_norm_from",
        "delta_norm_to",
        "relative_change",
        "is_anomaly_from",
        "is_anomaly_to",
        "rank_pct_from",
        "rank_pct_to",
    }
    assert mover["delta_norm_from"] != mover["delta_norm_to"]
    assert report.calibration_drift is not None
    assert report.calibration_drift.overall_drift_rms > 0.0
    assert [r["op"] for r in report.recipe] == ["recalibrate"]


def test_branch_vs_branch(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    hs.create_branch("other")
    hs.session("b", branch="exp").recalibrate(PID, exclude_keys=[_keys(sphere_dir)[0]])
    report = hs.session("t").navigator().compare_branches(PID, "exp", other="other")
    assert (report.from_ref, report.to_ref) == ("other", "exp")
    assert report.calibration_drift is not None
    assert report.calibration_drift.overall_drift_rms > 0.0


def test_keys_on_one_side_only_are_counted(sphere_dir: Path) -> None:
    geo = sphere_dir / "geometry" / PID / "data.lance"
    branch_ds = lance.dataset(str(geo)).checkout_version(("exp", None))
    lance.write_dataset(branch_ds.to_table().slice(0, 1), branch_ds, mode="overwrite")
    report = HyperSphere.open(str(sphere_dir)).session("t").navigator().compare_branches(PID, "exp")
    assert report.n_from == 2 and report.n_to == 1
    assert report.only_in_from == 1 and report.only_in_to == 0


def test_unknown_branch_raises(sphere_dir: Path) -> None:
    nav = HyperSphere.open(str(sphere_dir)).session("t").navigator()
    with pytest.raises(GDSError, match="no branch"):
        nav.compare_branches(PID, "nope")


def test_unknown_pattern_raises(sphere_dir: Path) -> None:
    nav = HyperSphere.open(str(sphere_dir)).session("t").navigator()
    with pytest.raises(GDSError, match="nope_pattern"):
        nav.compare_branches("nope_pattern", "exp")


def test_duplicate_keys_count_once(sphere_dir: Path) -> None:
    geo = sphere_dir / "geometry" / PID / "data.lance"
    branch_ds = lance.dataset(str(geo)).checkout_version(("exp", None))
    table = branch_ds.to_table()
    lance.write_dataset(table.slice(0, 1), branch_ds, mode="append")
    report = HyperSphere.open(str(sphere_dir)).session("t").navigator().compare_branches(PID, "exp")
    assert report.n_to == report.n_from == table.num_rows
    assert report.only_in_from == report.only_in_to == 0
    assert 0.0 <= report.jaccard <= 1.0


def test_unknown_branch_is_reported_before_any_geometry_read(
    sphere_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hypertopos.storage.reader import GDSReader

    def no_reads(*_a: object, **_k: object) -> None:
        raise AssertionError("geometry read before the refs were checked")

    monkeypatch.setattr(GDSReader, "read_geometry", no_reads)
    nav = HyperSphere.open(str(sphere_dir)).session("t").navigator()
    with pytest.raises(GDSError, match="no branch"):
        nav.compare_branches(PID, "nope")


def test_anomaly_counts_follow_the_global_threshold_rule(sphere_dir: Path) -> None:
    import numpy as np

    hs = HyperSphere.open(str(sphere_dir))
    report = hs.session("t").navigator().compare_branches(PID, "exp")
    pattern = hs.session("u").sphere().patterns[PID]
    theta_norm = float(np.linalg.norm(pattern.theta))
    table = lance.dataset(str(sphere_dir / "geometry" / PID / "data.lance")).to_table(
        columns=["delta_norm"]
    )
    expected = int(np.count_nonzero(table["delta_norm"].to_numpy() >= theta_norm))
    assert report.anomalies_from == report.anomalies_to == expected


def test_report_lists_flipped_entities(sphere_dir: Path) -> None:
    keys = _keys(sphere_dir)
    hs = HyperSphere.open(str(sphere_dir))
    hs.session("b", branch="exp").recalibrate(PID, exclude_keys=[keys[0]])
    report = hs.session("t").navigator().compare_branches(PID, "exp")
    assert len(report.flipped_in_top) == min(report.flipped_in, 20)
    assert len(report.flipped_out_top) == min(report.flipped_out, 20)
    for row in report.flipped_in_top:
        assert row["is_anomaly_to"] and not row["is_anomaly_from"]
    for row in report.flipped_out_top:
        assert row["is_anomaly_from"] and not row["is_anomaly_to"]


def test_top_movers_rank_by_relative_change(sphere_dir: Path) -> None:
    hs = HyperSphere.open(str(sphere_dir))
    hs.session("b", branch="exp").recalibrate(PID, exclude_keys=[_keys(sphere_dir)[0]])
    movers = hs.session("t").navigator().compare_branches(PID, "exp").top_movers
    rel = [
        abs(m["delta_norm_to"] - m["delta_norm_from"]) / max(abs(m["delta_norm_from"]), 1e-12)
        for m in movers
    ]
    assert rel == sorted(rel, reverse=True)
    assert all("relative_change" in m for m in movers)
