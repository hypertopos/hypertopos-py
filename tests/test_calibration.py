# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from pathlib import Path

import numpy as np
import pytest
from hypertopos.engine.calibration import (
    CalibrationTracker,
)


class TestCalibrationTracker:
    def test_create_from_stats(self):
        mu = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        sigma = np.array([0.05, 0.2, 0.2], dtype=np.float32)
        theta = np.array([4.49, 4.49, 4.49], dtype=np.float32)
        tracker = CalibrationTracker.from_stats(mu, sigma, theta, n=2000)
        assert tracker.running_n == 0
        assert tracker.drift_pct == 0.0
        assert not tracker.is_stale
        assert not tracker.is_blocked

    def test_welford_update_single(self):
        mu = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        sigma = np.array([0.05, 0.2, 0.2], dtype=np.float32)
        theta = np.array([4.49, 4.49, 4.49], dtype=np.float32)
        tracker = CalibrationTracker.from_stats(mu, sigma, theta, n=2000)
        new_shape = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        tracker.update(new_shape.reshape(1, -1))
        assert tracker.running_n == 1
        assert tracker.drift_pct == pytest.approx(0.0, abs=1e-4)

    def test_welford_update_batch_drift(self):
        mu = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        sigma = np.array([0.05, 0.2, 0.2], dtype=np.float32)
        theta = np.array([4.49, 4.49, 4.49], dtype=np.float32)
        tracker = CalibrationTracker.from_stats(mu, sigma, theta, n=100)
        shifted = np.tile([1.1, 0.9, 0.8], (100, 1)).astype(np.float32)
        tracker.update(shifted)
        assert tracker.running_n == 100
        assert tracker.drift_pct > 0.0

    def test_soft_threshold(self):
        mu = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        sigma = np.array([0.05, 0.2, 0.2], dtype=np.float32)
        theta = np.array([4.49, 4.49, 4.49], dtype=np.float32)
        tracker = CalibrationTracker.from_stats(
            mu,
            sigma,
            theta,
            n=10,
            soft_threshold=0.01,
        )
        shifted = np.tile([1.01, 0.7, 0.6], (10, 1)).astype(np.float32)
        tracker.update(shifted)
        assert tracker.is_stale
        assert not tracker.is_blocked

    def test_hard_threshold(self):
        mu = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        sigma = np.array([0.05, 0.2, 0.2], dtype=np.float32)
        theta = np.array([4.49, 4.49, 4.49], dtype=np.float32)
        tracker = CalibrationTracker.from_stats(
            mu,
            sigma,
            theta,
            n=10,
            hard_threshold=0.01,
        )
        shifted = np.tile([1.1, 0.9, 0.8], (100, 1)).astype(np.float32)
        tracker.update(shifted)
        assert tracker.is_blocked

    def test_reset(self):
        mu = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        sigma = np.array([0.05, 0.2, 0.2], dtype=np.float32)
        theta = np.array([4.49, 4.49, 4.49], dtype=np.float32)
        tracker = CalibrationTracker.from_stats(mu, sigma, theta, n=100)
        shifted = np.tile([1.1, 0.9, 0.8], (100, 1)).astype(np.float32)
        tracker.update(shifted)
        assert tracker.drift_pct > 0
        new_mu = np.array([1.05, 0.8, 0.7], dtype=np.float32)
        new_sigma = np.array([0.06, 0.21, 0.19], dtype=np.float32)
        new_theta = np.array([4.5, 4.5, 4.5], dtype=np.float32)
        tracker.reset(new_mu, new_sigma, new_theta, n=200)
        assert tracker.running_n == 0
        assert tracker.drift_pct == 0.0
        assert not tracker.is_stale
        assert not tracker.is_blocked


class TestCalibrationTrackerIO:
    def test_roundtrip(self, tmp_path: Path):
        from hypertopos.storage.reader import GDSReader
        from hypertopos.storage.writer import GDSWriter

        mu = np.array([1.0, 0.7, 0.6], dtype=np.float32)
        sigma = np.array([0.05, 0.2, 0.2], dtype=np.float32)
        theta = np.array([4.49, 4.49, 4.49], dtype=np.float32)
        tracker = CalibrationTracker.from_stats(mu, sigma, theta, n=2000)
        tracker.update(np.tile([1.01, 0.71, 0.61], (50, 1)).astype(np.float32))

        base = tmp_path / "gds" / "test_sphere"
        writer = GDSWriter(str(base))
        writer.write_calibration_tracker("test_pattern", tracker)

        reader = GDSReader(str(base))
        loaded = reader.read_calibration_tracker("test_pattern")
        assert loaded is not None
        assert loaded.running_n == 50
        assert loaded.calibrated_n == 2000
        np.testing.assert_array_almost_equal(loaded.calibrated_mu, mu)
        assert loaded.drift_pct == pytest.approx(tracker.drift_pct, abs=1e-4)

    def test_read_missing_returns_none(self, tmp_path: Path):
        from hypertopos.storage.reader import GDSReader

        base = tmp_path / "gds" / "test_sphere"
        base.mkdir(parents=True)
        reader = GDSReader(str(base))
        assert reader.read_calibration_tracker("nonexistent") is None


class TestBuilderInitializesTracker:
    def test_build_creates_tracker(self, tmp_path: Path):
        """After GDSBuilder.build(), calibration tracker exists for each pattern."""
        import pyarrow as pa
        from hypertopos.builder.builder import GDSBuilder
        from hypertopos.storage.reader import GDSReader

        rng = np.random.default_rng(42)
        builder = GDSBuilder(
            sphere_id="test_sphere",
            output_path=str(tmp_path / "test_sphere"),
        )

        # Add a minimal line with 10 entities, 2 properties
        table = pa.table(
            {
                "id": [f"entity_{i}" for i in range(10)],
                "amount": rng.normal(100, 10, 10).tolist(),
                "quantity": rng.normal(50, 5, 10).tolist(),
            }
        )
        from hypertopos.builder.builder import RelationSpec

        builder.add_line("test_line", table, key_col="id", source_id="test")
        builder.add_pattern(
            "test_pattern",
            pattern_type="anchor",
            entity_line="test_line",
            relations=[
                RelationSpec(line_id="test_line", fk_col=None, direction="self"),
            ],
        )

        sphere_path = builder.build()

        reader = GDSReader(sphere_path)
        tracker = reader.read_calibration_tracker("test_pattern")
        assert tracker is not None
        assert tracker.calibrated_n == 10
        assert tracker.running_n == 0
        assert tracker.drift_pct == 0.0
        assert len(tracker.calibrated_mu) >= 0


class TestRecalibrate:
    def test_recalibrate_resets_drift(self, tmp_path: Path):
        """recalibrate() recomputes stats and resets drift to zero."""
        from hypertopos.sphere import HyperSphere
        from tests.conftest import clone_sphere

        # Clone fixture to tmp_path to avoid corrupting shared fixture
        src = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gds" / "sales_sphere"
        fixture_path = tmp_path / "sales_sphere"
        clone_sphere(src, fixture_path)

        hs = HyperSphere.open(str(fixture_path))
        session = hs.session("test")

        # Get first pattern_id
        pid = list(session._manifest.pattern_versions.keys())[0]

        result = session.recalibrate(pid)
        assert result["pattern_id"] == pid
        assert result["records_recalibrated"] > 0
        assert "new_theta_norm" in result
        assert "old_theta_norm" in result
        assert isinstance(result["new_theta_norm"], float)
        assert isinstance(result["old_theta_norm"], float)

        # Tracker should be reset
        tracker_after = session._reader.read_calibration_tracker(pid)
        assert tracker_after is not None
        assert tracker_after.running_n == 0
        assert tracker_after.drift_pct == 0.0

    def test_recalibrate_keeps_zero_thresholds(self, tmp_path: Path):
        """Explicit 0.0 thresholds must not fall back to defaults."""
        from hypertopos.sphere import HyperSphere
        from tests.conftest import clone_sphere

        src = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gds" / "sales_sphere"
        fixture_path = tmp_path / "sales_sphere"
        clone_sphere(src, fixture_path)

        hs = HyperSphere.open(str(fixture_path))
        session = hs.session("test")
        pid = list(session._manifest.pattern_versions.keys())[0]

        session.recalibrate(pid, soft_threshold=0.0, hard_threshold=0.0)

        tracker_after = session._reader.read_calibration_tracker(pid)
        assert tracker_after is not None
        assert tracker_after.soft_threshold == 0.0
        assert tracker_after.hard_threshold == 0.0

    def test_recalibrate_prop_sigma_override(self, tmp_path: Path):
        """After recalibrate, prop_column sigma dims must be >= SIGMA_EPS_PROP."""
        import json

        import pyarrow as pa
        from hypertopos.builder._stats import SIGMA_EPS_PROP
        from hypertopos.builder.builder import GDSBuilder
        from hypertopos.sphere import HyperSphere

        rng = np.random.default_rng(42)
        n_entities = 50

        builder = GDSBuilder(
            sphere_id="prop_test",
            output_path=str(tmp_path / "prop_test"),
        )

        # Create a line with a low-variance property column.
        # amount: ~96% fill (2 nulls) → included as prop_column, low variance
        # status_flag: ~96% fill (2 nulls) → included, near-constant sigma << 0.2
        amounts = rng.normal(100, 10, n_entities).tolist()
        amounts[0] = None
        amounts[1] = None
        status_flags = [1.0] * n_entities
        status_flags[0] = None
        status_flags[1] = None
        table = pa.table(
            {
                "id": [f"e_{i}" for i in range(n_entities)],
                "amount": amounts,
                "status_flag": status_flags,
            }
        )
        builder.add_line("items", table, key_col="id", source_id="test")
        builder.add_pattern(
            "item_pattern",
            pattern_type="anchor",
            entity_line="items",
            relations=[],
            tracked_properties=["amount", "status_flag"],
        )

        sphere_path = builder.build()

        # Open and recalibrate
        hs = HyperSphere.open(sphere_path)
        session = hs.session("test")
        result = session.recalibrate("item_pattern")
        assert result["records_recalibrated"] == n_entities

        # Read updated sphere.json and check sigma for prop dims
        sphere_json = json.loads(
            (Path(sphere_path) / "_gds_meta" / "sphere.json").read_text(),
        )
        sigma = np.array(sphere_json["patterns"]["item_pattern"]["sigma_diag"])

        # prop_columns start after relation dims (0 relations here)
        # so ALL dims are prop dims — each must be >= SIGMA_EPS_PROP
        assert np.all(sigma >= SIGMA_EPS_PROP), (
            f"Expected all prop sigma >= {SIGMA_EPS_PROP}, got {sigma}"
        )
