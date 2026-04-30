# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Reader tests for read_calibration_fit + list_calibration_versions on
hand-built sphere fixtures (2.3-shaped and 2.4-shaped)."""
from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pytest


def _write_minimal_sphere_json(base: Path, format_version: str, pattern_extra: dict | None = None) -> None:
    pattern_node = {
        "pattern_id": "p",
        "version": 1,
        "relations": [{"line_id": "tx", "event_columns": ["amount"]}],
        "event_dimensions": ["amount_zscore"],
        "prop_columns": [],
        "dimension_kinds": ["gaussian"],
        "mu": [0.1],
        "sigma_diag": [0.05],
        "theta": [3.0],
        "population_size": 100,
        "dimension_weights": None,
        "dim_percentiles": None,
        "group_stats": None,
        "gmm_components": None,
        "edge_max": None,
        "computed_at": "2026-04-27T12:00:00+00:00",
        "last_calibrated_at": "2026-04-27T12:00:00+00:00",
    }
    if pattern_extra:
        pattern_node.update(pattern_extra)

    sphere_meta = {
        "format_version": format_version,
        "patterns": {"p": pattern_node},
    }
    if format_version == "2.4":
        sphere_meta["calibration_history_policy"] = {"last_k": 5}

    meta_dir = base / "_gds_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sphere.json").write_text(json.dumps(sphere_meta, indent=2), encoding="utf-8")


def _write_history_epoch(base: Path, pattern_id: str, epoch: int, mu_value: float) -> None:
    pdir = base / "_gds_meta" / "calibration_history" / pattern_id
    pdir.mkdir(parents=True, exist_ok=True)
    blob = {
        "pattern_id": pattern_id,
        "calibration_epoch": epoch,
        "schema_version": 1,
        "schema_hash": "a" * 64,
        "mu": [mu_value],
        "sigma_diag": [0.05],
        "theta": [3.0],
        "population_size": 100,
        "dimension_weights": None,
        "dimension_kinds": ["gaussian"],
        "dim_percentiles": None,
        "group_stats": None,
        "gmm_components": None,
        "edge_max": None,
        "computed_at": "2026-04-27T12:00:00+00:00",
        "last_calibrated_at": "2026-04-27T12:00:00+00:00",
    }
    (pdir / f"v={epoch}.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")


def test_read_calibration_fit_2_4_explicit_version(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.4", pattern_extra={"calibration_epoch": 2, "schema_hash": "a" * 64})
    _write_history_epoch(base, "p", 1, mu_value=0.1)
    _write_history_epoch(base, "p", 2, mu_value=0.2)

    reader = GDSReader(base)
    fit_v1 = reader.read_calibration_fit("p", version=1)
    fit_v2 = reader.read_calibration_fit("p", version=2)
    assert fit_v1.calibration_epoch == 1
    assert fit_v2.calibration_epoch == 2
    np.testing.assert_array_equal(fit_v1.mu, np.array([0.1], dtype=np.float32))
    np.testing.assert_array_equal(fit_v2.mu, np.array([0.2], dtype=np.float32))


def test_read_calibration_fit_2_4_version_none_resolves_latest(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.4", pattern_extra={"calibration_epoch": 2, "schema_hash": "a" * 64})
    _write_history_epoch(base, "p", 1, mu_value=0.1)
    _write_history_epoch(base, "p", 2, mu_value=0.2)

    reader = GDSReader(base)
    fit_latest = reader.read_calibration_fit("p")
    assert fit_latest.calibration_epoch == 2
    np.testing.assert_array_equal(fit_latest.mu, np.array([0.2], dtype=np.float32))


def test_read_calibration_fit_raises_when_version_missing(tmp_path):
    from hypertopos import CalibrationNotFoundError
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.4", pattern_extra={"calibration_epoch": 2, "schema_hash": "a" * 64})
    _write_history_epoch(base, "p", 2, mu_value=0.2)

    reader = GDSReader(base)
    with pytest.raises(CalibrationNotFoundError):
        reader.read_calibration_fit("p", version=1)


def test_list_calibration_versions_2_4(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.4", pattern_extra={"calibration_epoch": 3, "schema_hash": "a" * 64})
    _write_history_epoch(base, "p", 1, mu_value=0.1)
    _write_history_epoch(base, "p", 2, mu_value=0.2)
    _write_history_epoch(base, "p", 3, mu_value=0.3)

    reader = GDSReader(base)
    assert reader.list_calibration_versions("p") == [1, 2, 3]


def test_read_calibration_fit_2_3_reconstructs_from_inline_fields(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.3")  # no calibration_epoch, no schema_hash, no history dir

    reader = GDSReader(base)
    fit = reader.read_calibration_fit("p")
    assert fit.calibration_epoch == 1
    assert fit.schema_version == 1
    assert len(fit.schema_hash) == 64  # sha256 reconstructed from inline fields
    np.testing.assert_array_equal(fit.mu, np.array([0.1], dtype=np.float32))


def test_read_calibration_fit_2_3_explicit_v1_same_as_none(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.3")

    reader = GDSReader(base)
    fit_none = reader.read_calibration_fit("p")
    fit_v1 = reader.read_calibration_fit("p", version=1)
    assert fit_none.schema_hash == fit_v1.schema_hash
    np.testing.assert_array_equal(fit_none.mu, fit_v1.mu)


def test_read_calibration_fit_2_3_v2_raises(tmp_path):
    from hypertopos import CalibrationNotFoundError
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.3")

    reader = GDSReader(base)
    with pytest.raises(CalibrationNotFoundError):
        reader.read_calibration_fit("p", version=2)


def test_list_calibration_versions_2_3_returns_one(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.3")

    reader = GDSReader(base)
    assert reader.list_calibration_versions("p") == [1]


def test_sphere_load_rejects_last_k_zero(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.4", pattern_extra={"calibration_epoch": 1, "schema_hash": "a" * 64})
    sphere_path = base / "_gds_meta" / "sphere.json"
    sphere = json.loads(sphere_path.read_text())
    sphere["calibration_history_policy"] = {"last_k": 0}
    sphere_path.write_text(json.dumps(sphere))

    reader = GDSReader(base)
    with pytest.raises(ValueError, match="last_k must be >= 1"):
        reader.read_calibration_history_policy()


def test_sphere_load_rejects_last_k_negative(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    _write_minimal_sphere_json(base, "2.4", pattern_extra={"calibration_epoch": 1, "schema_hash": "a" * 64})
    sphere_path = base / "_gds_meta" / "sphere.json"
    sphere = json.loads(sphere_path.read_text())
    sphere["calibration_history_policy"] = {"last_k": -3}
    sphere_path.write_text(json.dumps(sphere))

    reader = GDSReader(base)
    with pytest.raises(ValueError, match="last_k must be >= 1"):
        reader.read_calibration_history_policy()


def test_sphere_load_accepts_last_k_one_and_default(tmp_path):
    from hypertopos.storage.reader import GDSReader

    base1 = tmp_path / "k1"
    base1.mkdir()
    _write_minimal_sphere_json(base1, "2.4", pattern_extra={"calibration_epoch": 1, "schema_hash": "a" * 64})
    sphere_path = base1 / "_gds_meta" / "sphere.json"
    sphere = json.loads(sphere_path.read_text())
    sphere["calibration_history_policy"] = {"last_k": 1}
    sphere_path.write_text(json.dumps(sphere))
    assert GDSReader(base1).read_calibration_history_policy() == {"last_k": 1}

    base_default = tmp_path / "default"
    base_default.mkdir()
    _write_minimal_sphere_json(base_default, "2.4", pattern_extra={"calibration_epoch": 1, "schema_hash": "a" * 64})
    sphere_path = base_default / "_gds_meta" / "sphere.json"
    sphere = json.loads(sphere_path.read_text())
    sphere.pop("calibration_history_policy", None)
    sphere_path.write_text(json.dumps(sphere))
    assert GDSReader(base_default).read_calibration_history_policy() == {"last_k": 5}


def test_read_calibration_fit_2_3_with_null_last_calibrated_at(tmp_path):
    """2.3 sphere may have last_calibrated_at: null on uncalibrated patterns.
    The fallback must not crash — falls back to computed_at."""
    from hypertopos.storage.reader import GDSReader

    base = tmp_path
    meta_dir = base / "_gds_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sphere.json").write_text(
        json.dumps(
            {
                "format_version": "2.3",
                "patterns": {
                    "p": {
                        "pattern_id": "p",
                        "version": 1,
                        "relations": [{"line_id": "tx", "event_columns": ["amount"]}],
                        "event_dimensions": ["amount_zscore"],
                        "prop_columns": [],
                        "dimension_kinds": ["gaussian"],
                        "mu": [0.1],
                        "sigma_diag": [0.05],
                        "theta": [3.0],
                        "population_size": 100,
                        "computed_at": "2026-04-27T12:00:00+00:00",
                        "last_calibrated_at": None,  # this is the test point
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reader = GDSReader(base)
    fit = reader.read_calibration_fit("p")
    # Falls back to computed_at when last_calibrated_at is null
    assert fit.last_calibrated_at.isoformat() == "2026-04-27T12:00:00+00:00"
