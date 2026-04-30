"""Unit tests for the multi-epoch calibration retention building blocks.

Covers: CalibrationFit dataclass, CalibrationNotFoundError, schema_hash helpers,
JSON serialization helpers, history-write/GC helpers. Integration tests against
real builder runs live in test_calibration_history_builder.py.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

import numpy as np
import pytest


def test_calibration_fit_construction_and_frozen():
    from hypertopos.model.sphere import CalibrationFit

    fit = CalibrationFit(
        pattern_id="account_pattern",
        calibration_epoch=1,
        schema_version=1,
        schema_hash="0" * 64,
        mu=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        sigma_diag=np.array([0.05, 0.06, 0.07], dtype=np.float32),
        theta=np.array([3.0, 3.0, 3.0], dtype=np.float32),
        population_size=1000,
        dimension_weights=None,
        dimension_kinds=["gaussian", "gaussian", "gaussian"],
        dim_percentiles=None,
        group_stats=None,
        gmm_components=None,
        edge_max=None,
        computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert fit.pattern_id == "account_pattern"
    assert fit.calibration_epoch == 1
    assert fit.schema_hash == "0" * 64
    # frozen: reassignment must raise
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.calibration_epoch = 2  # type: ignore[misc]


def test_calibration_not_found_error_extends_gds_error():
    from hypertopos import GDSError
    from hypertopos.storage.calibration_history import CalibrationNotFoundError

    err = CalibrationNotFoundError("pattern=foo version=42")
    assert isinstance(err, GDSError)
    assert "pattern=foo version=42" in str(err)


def test_compute_pattern_schema_hash_deterministic():
    from hypertopos.storage.calibration_history import compute_pattern_schema_hash

    payload = {
        "relations": [{"line_id": "tx", "event_columns": ["amount", "ts"]}],
        "event_dimensions": ["amount_zscore", "interevent_seconds"],
        "prop_columns": ["country"],
        "dimension_kinds": ["gaussian", "gaussian", "bernoulli"],
    }
    h1 = compute_pattern_schema_hash(payload)
    h2 = compute_pattern_schema_hash(payload)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex
    assert all(c in "0123456789abcdef" for c in h1)


def test_compute_pattern_schema_hash_changes_on_schema_drift():
    from hypertopos.storage.calibration_history import compute_pattern_schema_hash

    base = {
        "relations": [{"line_id": "tx", "event_columns": ["amount"]}],
        "event_dimensions": ["amount_zscore"],
        "prop_columns": [],
        "dimension_kinds": ["gaussian"],
    }
    h_base = compute_pattern_schema_hash(base)

    # add a prop column → different hash
    drift_a = {**base, "prop_columns": ["country"]}
    assert compute_pattern_schema_hash(drift_a) != h_base

    # change a dimension_kind (Gaussian → Poisson) → different hash
    drift_b = {**base, "dimension_kinds": ["poisson"]}
    assert compute_pattern_schema_hash(drift_b) != h_base

    # reorder relations → different hash (order matters because dim order matters)
    drift_c = {
        **base,
        "relations": [
            {"line_id": "tx", "event_columns": ["amount"]},
            {"line_id": "tx2", "event_columns": []},
        ],
    }
    drift_c_reordered = {
        **base,
        "relations": [
            {"line_id": "tx2", "event_columns": []},
            {"line_id": "tx", "event_columns": ["amount"]},
        ],
    }
    h_c = compute_pattern_schema_hash(drift_c)
    h_c_rev = compute_pattern_schema_hash(drift_c_reordered)
    assert h_c != h_base  # adding a relation must change the hash
    assert h_c != h_c_rev  # reordering relations must change the hash


def test_schema_hash_2_3_reconstructor_agrees_with_compute():
    from hypertopos.storage.calibration_history import (
        _compute_schema_hash_from_pattern_node,
        compute_pattern_schema_hash,
    )

    pattern_node = {
        "relations": [
            {"line_id": "tx", "event_columns": ["amount", "ts"]},
        ],
        "event_dimensions": ["amount_zscore"],
        "prop_columns": ["country"],
        "dimension_kinds": ["gaussian", "gaussian", "bernoulli"],
        # plus extra fields that should NOT influence the hash:
        "mu": [0.1, 0.2, 0.3],
        "version": 1,
    }
    payload = {
        "relations": [
            {"line_id": "tx", "event_columns": ["amount", "ts"]},
        ],
        "event_dimensions": ["amount_zscore"],
        "prop_columns": ["country"],
        "dimension_kinds": ["gaussian", "gaussian", "bernoulli"],
    }
    assert _compute_schema_hash_from_pattern_node(pattern_node) == compute_pattern_schema_hash(payload)


def test_calibration_fit_json_round_trip():
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import deserialize_fit, serialize_fit

    original = CalibrationFit(
        pattern_id="p",
        calibration_epoch=3,
        schema_version=1,
        schema_hash="a" * 64,
        mu=np.array([0.1, 0.2], dtype=np.float32),
        sigma_diag=np.array([0.05, 0.06], dtype=np.float32),
        theta=np.array([3.0, 3.0], dtype=np.float32),
        population_size=2000,
        dimension_weights=np.array([1.0, 0.5], dtype=np.float32),
        dimension_kinds=["gaussian", "poisson"],
        dim_percentiles={"0": [0.0, 0.5, 1.0]},
        group_stats={"US": {"mu": [0.1, 0.2], "sigma_diag": [0.05, 0.06], "theta": [3.0, 3.0], "population_size": 500}},
        gmm_components=[{"weight": 1.0, "mu": [0.1, 0.2], "sigma_diag": [0.05, 0.06]}],
        edge_max=np.array([10.0, 20.0], dtype=np.float32),
        computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
    )

    blob = serialize_fit(original)
    encoded = json.dumps(blob)
    decoded = json.loads(encoded)
    restored = deserialize_fit(decoded)

    assert restored.pattern_id == original.pattern_id
    assert restored.calibration_epoch == original.calibration_epoch
    assert restored.schema_hash == original.schema_hash
    np.testing.assert_array_equal(restored.mu, original.mu)
    np.testing.assert_array_equal(restored.sigma_diag, original.sigma_diag)
    np.testing.assert_array_equal(restored.theta, original.theta)
    np.testing.assert_array_equal(restored.dimension_weights, original.dimension_weights)
    assert restored.dimension_kinds == original.dimension_kinds
    np.testing.assert_array_equal(restored.edge_max, original.edge_max)
    assert restored.dim_percentiles == original.dim_percentiles
    assert restored.group_stats == original.group_stats
    assert restored.gmm_components == original.gmm_components
    assert restored.computed_at == original.computed_at
    assert restored.last_calibrated_at == original.last_calibrated_at


def test_calibration_fit_round_trip_with_optionals_none():
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import deserialize_fit, serialize_fit

    original = CalibrationFit(
        pattern_id="p",
        calibration_epoch=1,
        schema_version=1,
        schema_hash="b" * 64,
        mu=np.array([0.1], dtype=np.float32),
        sigma_diag=np.array([0.05], dtype=np.float32),
        theta=np.array([3.0], dtype=np.float32),
        population_size=100,
        dimension_weights=None,
        dimension_kinds=None,
        dim_percentiles=None,
        group_stats=None,
        gmm_components=None,
        edge_max=None,
        computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    restored = deserialize_fit(json.loads(json.dumps(serialize_fit(original))))
    assert restored.dimension_weights is None
    assert restored.dimension_kinds is None
    assert restored.dim_percentiles is None
    assert restored.group_stats is None
    assert restored.gmm_components is None
    assert restored.edge_max is None


def test_write_calibration_history_epoch_creates_file(tmp_path):
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import (
        list_calibration_versions,
        write_calibration_history_epoch,
    )

    base = tmp_path
    fit = CalibrationFit(
        pattern_id="p",
        calibration_epoch=1,
        schema_version=1,
        schema_hash="a" * 64,
        mu=np.array([0.1], dtype=np.float32),
        sigma_diag=np.array([0.05], dtype=np.float32),
        theta=np.array([3.0], dtype=np.float32),
        population_size=100,
        dimension_weights=None,
        dimension_kinds=None,
        dim_percentiles=None,
        group_stats=None,
        gmm_components=None,
        edge_max=None,
        computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    write_calibration_history_epoch(base, fit, last_k=5)

    expected = base / "_gds_meta" / "calibration_history" / "p" / "v=1.json"
    assert expected.exists()
    assert list_calibration_versions(base, "p") == [1]


def test_write_calibration_history_epoch_gc_trims_oldest(tmp_path):
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import (
        list_calibration_versions,
        write_calibration_history_epoch,
    )

    base = tmp_path

    def _make_fit(n: int) -> CalibrationFit:
        return CalibrationFit(
            pattern_id="p",
            calibration_epoch=n,
            schema_version=1,
            schema_hash="a" * 64,
            mu=np.array([0.1], dtype=np.float32),
            sigma_diag=np.array([0.05], dtype=np.float32),
            theta=np.array([3.0], dtype=np.float32),
            population_size=100,
            dimension_weights=None,
            dimension_kinds=None,
            dim_percentiles=None,
            group_stats=None,
            gmm_components=None,
            edge_max=None,
            computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
            last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        )

    for n in range(1, 7):
        write_calibration_history_epoch(base, _make_fit(n), last_k=5)

    assert list_calibration_versions(base, "p") == [2, 3, 4, 5, 6]


def test_list_calibration_versions_returns_empty_list_when_no_history_dir(tmp_path):
    from hypertopos.storage.calibration_history import list_calibration_versions

    assert list_calibration_versions(tmp_path, "missing_pattern") == []


def test_write_calibration_history_epoch_last_k_one_is_replace(tmp_path):
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import (
        list_calibration_versions,
        write_calibration_history_epoch,
    )

    base = tmp_path
    for n in range(1, 5):
        fit = CalibrationFit(
            pattern_id="p",
            calibration_epoch=n,
            schema_version=1,
            schema_hash="a" * 64,
            mu=np.array([0.1], dtype=np.float32),
            sigma_diag=np.array([0.05], dtype=np.float32),
            theta=np.array([3.0], dtype=np.float32),
            population_size=100,
            dimension_weights=None,
            dimension_kinds=None,
            dim_percentiles=None,
            group_stats=None,
            gmm_components=None,
            edge_max=None,
            computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
            last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        )
        write_calibration_history_epoch(base, fit, last_k=1)

    assert list_calibration_versions(base, "p") == [4]


def test_write_calibration_history_epoch_rejects_invalid_last_k(tmp_path):
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import write_calibration_history_epoch

    fit = CalibrationFit(
        pattern_id="p",
        calibration_epoch=1,
        schema_version=1,
        schema_hash="a" * 64,
        mu=np.array([0.1], dtype=np.float32),
        sigma_diag=np.array([0.05], dtype=np.float32),
        theta=np.array([3.0], dtype=np.float32),
        population_size=100,
        dimension_weights=None,
        dimension_kinds=None,
        dim_percentiles=None,
        group_stats=None,
        gmm_components=None,
        edge_max=None,
        computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="last_k must be >= 1"):
        write_calibration_history_epoch(tmp_path, fit, last_k=0)
    with pytest.raises(ValueError, match="last_k must be >= 1"):
        write_calibration_history_epoch(tmp_path, fit, last_k=-3)
