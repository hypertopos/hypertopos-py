# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""recalibrate records a new calibration epoch, like a rebuild does."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.builder import GDSBuilder
from hypertopos.sphere import HyperSphere

PID = "cust_pattern"


def _sphere_with_outliers(tmp_path: Path) -> str:
    from tests.test_incremental import _build_anchor_sphere

    base = str(_build_anchor_sphere(tmp_path / "anchor", 300))
    extra = pa.table(
        {
            "primary_key": pa.array([f"X{i}" for i in range(60)], type=pa.string()),
            "event_count": pa.array([5000.0] * 60, type=pa.float64()),
        }
    )
    GDSBuilder("other", base).incremental_update(PID, changed_entities=extra, recalibrate="never")
    return base


def _epoch(base: str) -> int:
    meta = json.loads((Path(base) / "_gds_meta" / "sphere.json").read_text())
    return int(meta["patterns"][PID]["calibration_epoch"])


def test_recalibrate_writes_the_next_epoch(tmp_path):
    base = _sphere_with_outliers(tmp_path)
    session = HyperSphere.open(base).session("epoch")
    reader = session.navigator()._storage
    assert reader.list_calibration_versions(PID) == [1]
    fit_before = reader.read_calibration_fit(PID, 1)

    result = session.recalibrate(PID)

    assert _epoch(base) == 2
    assert reader.list_calibration_versions(PID) == [1, 2]
    fit = reader.read_calibration_fit(PID)
    assert fit.calibration_epoch == 2
    pattern = session.sphere().patterns[PID]
    np.testing.assert_array_equal(fit.mu, pattern.mu)
    np.testing.assert_array_equal(fit.sigma_diag, pattern.sigma_diag)
    np.testing.assert_array_equal(fit.theta, pattern.theta)
    assert fit.population_size == result["records_recalibrated"] == 360
    assert fit.schema_hash == fit_before.schema_hash
    assert fit.theta_sensitivity != fit_before.theta_sensitivity
    np.testing.assert_array_equal(reader.read_calibration_fit(PID, 1).mu, fit_before.mu)


def test_compare_calibrations_sees_the_recalibration(tmp_path):
    base = _sphere_with_outliers(tmp_path)
    session = HyperSphere.open(base).session("compare")
    session.recalibrate(PID)
    report = session.navigator().compare_calibrations(PID, 1, 2)
    assert report.v_from == 1 and report.v_to == 2


def test_recalibrate_keeps_only_last_k_epochs(tmp_path):
    from hypertopos.storage.writer import write_sphere_json

    base = _sphere_with_outliers(tmp_path)
    meta_dir = Path(base) / "_gds_meta"
    data = json.loads((meta_dir / "sphere.json").read_text())
    data["calibration_history_policy"] = {"last_k": 2}
    write_sphere_json(meta_dir, data)
    session = HyperSphere.open(base).session("gc")
    for _ in range(3):
        session.recalibrate(PID)
    assert _epoch(base) == 4
    assert session.navigator()._storage.list_calibration_versions(PID) == [3, 4]


def test_reader_without_snapshot_reads_the_new_epoch(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = _sphere_with_outliers(tmp_path)
    HyperSphere.open(base).session("writer").recalibrate(PID)
    assert GDSReader(base).read_calibration_fit(PID).calibration_epoch == 2


@pytest.mark.parametrize("field", ["dim_percentiles", "edge_max"])
def test_fields_computed_from_entity_data_carry_over(tmp_path, field):
    base = _sphere_with_outliers(tmp_path)
    session = HyperSphere.open(base).session("carry")
    reader = session.navigator()._storage
    before = getattr(reader.read_calibration_fit(PID, 1), field)
    session.recalibrate(PID)
    after = getattr(reader.read_calibration_fit(PID, 2), field)
    if isinstance(before, np.ndarray):
        np.testing.assert_array_equal(after, before)
    else:
        assert after == before


def test_first_recalibrate_without_history_keeps_epoch_one(tmp_path):
    import shutil

    base = _sphere_with_outliers(tmp_path)
    shutil.rmtree(Path(base) / "_gds_meta" / "calibration_history")
    session = HyperSphere.open(base).session("legacy")
    reader = session.navigator()._storage
    reconstructed = reader.read_calibration_fit(PID)

    session.recalibrate(PID)

    assert reader.list_calibration_versions(PID) == [1, 2]
    np.testing.assert_array_equal(reader.read_calibration_fit(PID, 1).mu, reconstructed.mu)
    report = session.navigator().compare_calibrations(PID)
    assert report.v_from == 1 and report.v_to == 2
