"""Unit tests for the M2 compare_calibrations primitive.

Covers: DimensionDrift + CalibrationDriftReport dataclasses, _compute_calibration_drift
math helper, GDSNavigator.compare_calibrations orchestration (auto-resolve, edge
cases, error semantics).
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


def test_dimension_drift_construction_and_frozen():
    from hypertopos.model.sphere import DimensionDrift

    d = DimensionDrift(
        dim_index=3,
        dim_kind="gaussian",
        mu_from=0.1,
        mu_to=0.4,
        mu_delta=0.3,
        mu_delta_normalized=6.0,
        sigma_from=0.05,
        sigma_to=0.06,
        sigma_delta=0.01,
        theta_from=3.0,
        theta_to=3.2,
        theta_delta=0.2,
    )
    assert d.dim_index == 3
    assert d.mu_delta_normalized == 6.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.mu_delta = 999.0  # type: ignore[misc]


def test_calibration_drift_report_construction_and_frozen():
    from hypertopos.model.sphere import CalibrationDriftReport, DimensionDrift

    dd = DimensionDrift(
        dim_index=0,
        dim_kind="gaussian",
        mu_from=0.0,
        mu_to=0.0,
        mu_delta=0.0,
        mu_delta_normalized=0.0,
        sigma_from=1.0,
        sigma_to=1.0,
        sigma_delta=0.0,
        theta_from=3.0,
        theta_to=3.0,
        theta_delta=0.0,
    )
    report = CalibrationDriftReport(
        pattern_id="p",
        v_from=1,
        v_to=2,
        schema_hash="a" * 64,
        population_size_from=100,
        population_size_to=110,
        overall_drift_rms=0.0,
        top_drifted=[dd],
        per_dimension=None,
    )
    assert report.pattern_id == "p"
    assert report.top_drifted[0].dim_index == 0
    assert report.per_dimension is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.v_from = 99  # type: ignore[misc]


def test_calibration_drift_report_public_re_exports():
    from hypertopos import CalibrationDriftReport, DimensionDrift  # noqa: F401


def _make_fit(
    *,
    mu: list[float],
    sigma: list[float] | None = None,
    theta: list[float] | None = None,
    pattern_id: str = "p",
    calibration_epoch: int = 1,
    schema_hash: str = "a" * 64,
    population_size: int = 100,
    dimension_kinds: list[str] | None = None,
):
    from hypertopos.model.sphere import CalibrationFit

    D = len(mu)
    return CalibrationFit(
        pattern_id=pattern_id,
        calibration_epoch=calibration_epoch,
        schema_version=1,
        schema_hash=schema_hash,
        mu=np.asarray(mu, dtype=np.float32),
        sigma_diag=np.asarray(sigma if sigma is not None else [1.0] * D, dtype=np.float32),
        theta=np.asarray(theta if theta is not None else [3.0] * D, dtype=np.float32),
        population_size=population_size,
        dimension_weights=None,
        dimension_kinds=dimension_kinds,
        dim_percentiles=None,
        group_stats=None,
        gmm_components=None,
        edge_max=None,
        computed_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
        last_calibrated_at=datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_compute_calibration_drift_identical_fits_returns_zero():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit = _make_fit(mu=[0.1, 0.2, 0.3], sigma=[0.05, 0.05, 0.05])
    report = _compute_calibration_drift(fit, fit, top_n=10, verbose=False)

    assert report.overall_drift_rms == 0.0
    assert len(report.top_drifted) == 3
    for d in report.top_drifted:
        assert d.mu_delta == 0.0
        assert d.mu_delta_normalized == 0.0
        assert d.sigma_delta == 0.0


def test_compute_calibration_drift_single_dim_shifted_ranks_first():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit_a = _make_fit(mu=[0.0, 0.0, 0.0, 0.0], sigma=[0.1, 0.1, 0.1, 0.1])
    fit_b = _make_fit(
        mu=[0.0, 0.0, 0.5, 0.0],
        sigma=[0.1, 0.1, 0.1, 0.1],
        calibration_epoch=2,
    )
    report = _compute_calibration_drift(fit_a, fit_b, top_n=4, verbose=False)

    assert report.top_drifted[0].dim_index == 2
    assert report.top_drifted[0].mu_delta == pytest.approx(0.5, abs=1e-6)
    assert report.top_drifted[0].mu_delta_normalized == pytest.approx(5.0, abs=1e-6)
    for d in report.top_drifted[1:]:
        assert d.mu_delta == pytest.approx(0.0, abs=1e-6)


def test_compute_calibration_drift_overall_rms_one_when_all_dims_shift_by_one_sigma():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit_a = _make_fit(mu=[0.0, 0.0, 0.0, 0.0], sigma=[0.1, 0.1, 0.1, 0.1])
    fit_b = _make_fit(
        mu=[0.1, 0.1, 0.1, 0.1],
        sigma=[0.1, 0.1, 0.1, 0.1],
        calibration_epoch=2,
    )
    report = _compute_calibration_drift(fit_a, fit_b, top_n=4, verbose=False)

    # ||[1, 1, 1, 1]||_2 = 2.0; / sqrt(4) = 1.0
    assert report.overall_drift_rms == pytest.approx(1.0, abs=1e-6)


def test_compute_calibration_drift_sigma_safe_guard_on_degenerate_dim():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit_a = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.0])
    fit_b = _make_fit(mu=[0.05, 0.7], sigma=[0.1, 0.0], calibration_epoch=2)
    report = _compute_calibration_drift(fit_a, fit_b, top_n=2, verbose=False)

    dim_1 = next(d for d in report.top_drifted if d.dim_index == 1)
    assert np.isfinite(dim_1.mu_delta_normalized)
    assert dim_1.mu_delta_normalized == pytest.approx(0.7, abs=1e-6)


def test_compute_calibration_drift_verbose_returns_full_table():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit_a = _make_fit(mu=[0.0] * 5, sigma=[0.1] * 5)
    fit_b = _make_fit(mu=[0.05] * 5, sigma=[0.1] * 5, calibration_epoch=2)
    report = _compute_calibration_drift(fit_a, fit_b, top_n=2, verbose=True)

    assert len(report.top_drifted) == 2
    assert report.per_dimension is not None
    assert len(report.per_dimension) == 5


def test_compute_calibration_drift_top_n_larger_than_d_returns_all():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit_a = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1])
    fit_b = _make_fit(mu=[0.05, 0.05], sigma=[0.1, 0.1], calibration_epoch=2)
    report = _compute_calibration_drift(fit_a, fit_b, top_n=99, verbose=False)

    assert len(report.top_drifted) == 2


def test_compute_calibration_drift_carries_dim_kind_when_present():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit_a = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], dimension_kinds=["gaussian", "poisson"])
    fit_b = _make_fit(
        mu=[0.0, 0.05],
        sigma=[0.1, 0.1],
        calibration_epoch=2,
        dimension_kinds=["gaussian", "poisson"],
    )
    report = _compute_calibration_drift(fit_a, fit_b, top_n=2, verbose=False)

    by_index = {d.dim_index: d for d in report.top_drifted}
    assert by_index[0].dim_kind == "gaussian"
    assert by_index[1].dim_kind == "poisson"


def test_compute_calibration_drift_dim_kind_none_when_fits_have_no_kinds():
    from hypertopos.navigation.navigator import _compute_calibration_drift

    fit_a = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], dimension_kinds=None)
    fit_b = _make_fit(mu=[0.0, 0.05], sigma=[0.1, 0.1], calibration_epoch=2, dimension_kinds=None)
    report = _compute_calibration_drift(fit_a, fit_b, top_n=2, verbose=False)
    for d in report.top_drifted:
        assert d.dim_kind is None


def _write_minimal_2_4_sphere(base: Path, epochs: dict[int, dict]) -> None:
    """Write a minimal 2.4 sphere with a single pattern 'p' and the given
    epochs on disk under calibration_history/p/.
    `epochs` is {N: {"mu": [...], "sigma": [...], "schema_hash": "...", ...}}.
    """
    meta_dir = base / "_gds_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    versions = sorted(epochs.keys())
    latest = versions[-1]
    latest_blob = epochs[latest]
    pattern_node = {
        "pattern_id": "p",
        "version": 1,
        "relations": [{"line_id": "tx", "event_columns": ["amount"]}],
        "event_dimensions": ["amount_zscore"],
        "prop_columns": [],
        "dimension_kinds": latest_blob.get("dimension_kinds", ["gaussian"]),
        "mu": latest_blob["mu"],
        "sigma_diag": latest_blob["sigma"],
        "theta": latest_blob.get("theta", [3.0] * len(latest_blob["mu"])),
        "population_size": latest_blob.get("population_size", 100),
        "dimension_weights": None,
        "dim_percentiles": None,
        "group_stats": None,
        "gmm_components": None,
        "edge_max": None,
        "computed_at": "2026-04-27T12:00:00+00:00",
        "last_calibrated_at": "2026-04-27T12:00:00+00:00",
        "calibration_epoch": latest,
        "schema_hash": latest_blob.get("schema_hash", "a" * 64),
    }
    sphere_meta = {
        "format_version": "2.4",
        "calibration_history_policy": {"last_k": 5},
        "patterns": {"p": pattern_node},
    }
    (meta_dir / "sphere.json").write_text(json.dumps(sphere_meta, indent=2), encoding="utf-8")

    history_dir = meta_dir / "calibration_history" / "p"
    history_dir.mkdir(parents=True, exist_ok=True)
    for n, blob in epochs.items():
        D = len(blob["mu"])
        epoch_blob = {
            "pattern_id": "p",
            "calibration_epoch": n,
            "schema_version": 1,
            "schema_hash": blob.get("schema_hash", "a" * 64),
            "mu": blob["mu"],
            "sigma_diag": blob["sigma"],
            "theta": blob.get("theta", [3.0] * D),
            "population_size": blob.get("population_size", 100),
            "dimension_weights": None,
            "dimension_kinds": blob.get("dimension_kinds", ["gaussian"] * D),
            "dim_percentiles": None,
            "group_stats": None,
            "gmm_components": None,
            "edge_max": None,
            "computed_at": "2026-04-27T12:00:00+00:00",
            "last_calibrated_at": "2026-04-27T12:00:00+00:00",
        }
        (history_dir / f"v={n}.json").write_text(
            json.dumps(epoch_blob, indent=2), encoding="utf-8"
        )


def _make_navigator(base: Path):
    """Construct GDSNavigator directly bound to a GDSReader rooted at `base`.

    Bypasses HyperSphere.open() because it parses the full sphere.json
    (all pattern fields) which would force the minimal fixture above to
    duplicate every Pattern/Line/Storage field. compare_calibrations only
    touches reader.list_calibration_versions + reader.read_calibration_fit,
    so engine/manifest/contract are unused on this code path and are
    safely mocked.
    """
    from unittest.mock import MagicMock

    from hypertopos.navigation.navigator import GDSNavigator
    from hypertopos.storage.reader import GDSReader

    reader = GDSReader(str(base))
    return GDSNavigator(
        engine=MagicMock(),
        storage=reader,
        manifest=MagicMock(),
        contract=MagicMock(),
    )


def test_compare_calibrations_explicit_versions(tmp_path):
    _write_minimal_2_4_sphere(
        tmp_path,
        {
            1: {"mu": [0.0, 0.0], "sigma": [0.1, 0.1]},
            2: {"mu": [0.05, 0.0], "sigma": [0.1, 0.1]},
        },
    )
    nav = _make_navigator(tmp_path)
    report = nav.compare_calibrations("p", v_from=1, v_to=2, top_n=2, verbose=False)
    assert report.v_from == 1
    assert report.v_to == 2
    assert report.overall_drift_rms == pytest.approx(0.5 / np.sqrt(2), abs=1e-5)
    by_index = {d.dim_index: d for d in report.top_drifted}
    assert by_index[0].mu_delta_normalized == pytest.approx(0.5, abs=1e-5)
    assert by_index[1].mu_delta_normalized == pytest.approx(0.0, abs=1e-5)


def test_compare_calibrations_auto_resolves_to_two_latest_when_both_none(tmp_path):
    _write_minimal_2_4_sphere(
        tmp_path,
        {
            1: {"mu": [0.0], "sigma": [0.1]},
            2: {"mu": [0.05], "sigma": [0.1]},
            3: {"mu": [0.10], "sigma": [0.1]},
        },
    )
    nav = _make_navigator(tmp_path)
    report = nav.compare_calibrations("p", top_n=1, verbose=False)
    assert report.v_from == 2
    assert report.v_to == 3


def test_compare_calibrations_v_to_none_resolves_to_latest(tmp_path):
    _write_minimal_2_4_sphere(
        tmp_path,
        {
            1: {"mu": [0.0], "sigma": [0.1]},
            2: {"mu": [0.05], "sigma": [0.1]},
            3: {"mu": [0.10], "sigma": [0.1]},
        },
    )
    nav = _make_navigator(tmp_path)
    report = nav.compare_calibrations("p", v_from=1, top_n=1, verbose=False)
    assert report.v_from == 1
    assert report.v_to == 3


def test_compare_calibrations_raises_when_versions_equal(tmp_path):
    _write_minimal_2_4_sphere(
        tmp_path,
        {
            1: {"mu": [0.0], "sigma": [0.1]},
            2: {"mu": [0.05], "sigma": [0.1]},
        },
    )
    nav = _make_navigator(tmp_path)
    with pytest.raises(ValueError, match="must differ"):
        nav.compare_calibrations("p", v_from=2, v_to=2)


def test_compare_calibrations_raises_when_only_one_epoch_and_auto(tmp_path):
    _write_minimal_2_4_sphere(tmp_path, {1: {"mu": [0.0], "sigma": [0.1]}})
    nav = _make_navigator(tmp_path)
    with pytest.raises(ValueError, match="at least 2 epochs"):
        nav.compare_calibrations("p")


def test_compare_calibrations_raises_on_schema_mismatch(tmp_path):
    _write_minimal_2_4_sphere(
        tmp_path,
        {
            1: {"mu": [0.0], "sigma": [0.1], "schema_hash": "a" * 64},
            2: {"mu": [0.05], "sigma": [0.1], "schema_hash": "b" * 64},
        },
    )
    nav = _make_navigator(tmp_path)
    with pytest.raises(ValueError, match="schema_hash"):
        nav.compare_calibrations("p", v_from=1, v_to=2)


def test_compare_calibrations_raises_on_missing_version(tmp_path):
    from hypertopos import CalibrationNotFoundError

    _write_minimal_2_4_sphere(tmp_path, {2: {"mu": [0.0], "sigma": [0.1]}})
    nav = _make_navigator(tmp_path)
    with pytest.raises(CalibrationNotFoundError):
        nav.compare_calibrations("p", v_from=1, v_to=2)
