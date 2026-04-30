"""Unit tests for the M3 decompose_drift primitive.

Covers: DimensionDecomposition + IntrinsicExtrinsicReport dataclasses,
_compute_intrinsic_extrinsic_decomposition math helper, GDSNavigator.decompose_drift
orchestration (auto-resolve, edge cases, error semantics).
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest


def test_dimension_decomposition_construction_and_frozen():
    from hypertopos.model.sphere import DimensionDecomposition

    d = DimensionDecomposition(
        dim_index=2,
        dim_kind="gaussian",
        dim_label="amount",
        total=1.5,
        intrinsic=0.2,
        extrinsic=1.3,
        intrinsic_fraction=0.0231,
    )
    assert d.dim_index == 2
    assert d.dim_label == "amount"
    assert d.total == pytest.approx(1.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.intrinsic = 99.0  # type: ignore[misc]


def test_intrinsic_extrinsic_report_construction_and_frozen():
    from hypertopos.model.sphere import (
        DimensionDecomposition,
        IntrinsicExtrinsicReport,
    )

    dd = DimensionDecomposition(
        dim_index=0,
        dim_kind="gaussian",
        dim_label="d0",
        total=0.0,
        intrinsic=0.0,
        extrinsic=0.0,
        intrinsic_fraction=0.0,
    )
    report = IntrinsicExtrinsicReport(
        pattern_id="p",
        entity_key="E1",
        v_from=1,
        v_to=2,
        schema_hash="a" * 64,
        timestamp_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        timestamp_to=datetime(2026, 4, 1, tzinfo=timezone.utc),
        intrinsic_displacement=0.5,
        extrinsic_displacement=1.2,
        total_displacement=1.3,
        intrinsic_fraction=0.148,
        top_dimensions=[dd],
        per_dimension=None,
    )
    assert report.entity_key == "E1"
    assert report.top_dimensions[0].dim_index == 0
    assert report.per_dimension is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.v_from = 99  # type: ignore[misc]


def test_decompose_drift_public_re_exports():
    from hypertopos import DimensionDecomposition, IntrinsicExtrinsicReport  # noqa: F401


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


def _ts(month: int) -> datetime:
    return datetime(2026, month, 1, tzinfo=timezone.utc)


def test_compute_decomposition_identical_shape_identical_calibration_returns_zero():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit = _make_fit(mu=[0.1, 0.2, 0.3], sigma=[0.05, 0.05, 0.05])
    shape = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape, shape_b=shape, fit_v1=fit, fit_v2=fit,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=["d0", "d1", "d2"],
        top_n=10, verbose=False,
    )
    assert report.total_displacement == 0.0
    assert report.intrinsic_displacement == 0.0
    assert report.extrinsic_displacement == 0.0
    assert report.intrinsic_fraction == 0.0
    assert len(report.top_dimensions) == 3


def test_compute_decomposition_shape_changed_calibration_unchanged_is_all_intrinsic():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1])
    shape_a = np.asarray([0.0, 0.0], dtype=np.float32)
    shape_b = np.asarray([0.0, 0.5], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape_a, shape_b=shape_b, fit_v1=fit, fit_v2=fit,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=None, top_n=2, verbose=False,
    )
    assert report.intrinsic_displacement == pytest.approx(5.0, abs=1e-5)
    assert report.extrinsic_displacement == pytest.approx(0.0, abs=1e-5)
    assert report.intrinsic_fraction == pytest.approx(1.0, abs=1e-5)


def test_compute_decomposition_shape_unchanged_calibration_changed_is_all_extrinsic():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit_v1 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1])
    fit_v2 = _make_fit(mu=[0.05, 0.0], sigma=[0.1, 0.1], calibration_epoch=2)
    shape = np.asarray([0.5, 0.0], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape, shape_b=shape, fit_v1=fit_v1, fit_v2=fit_v2,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=None, top_n=2, verbose=False,
    )
    assert report.intrinsic_displacement == pytest.approx(0.0, abs=1e-5)
    assert report.extrinsic_displacement == pytest.approx(0.5, abs=1e-5)
    assert report.intrinsic_fraction == pytest.approx(0.0, abs=1e-5)


def test_compute_decomposition_mixed_split_correct_ratio():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit_v1 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1])
    fit_v2 = _make_fit(mu=[0.1, 0.0], sigma=[0.1, 0.1], calibration_epoch=2)
    shape_a = np.asarray([0.0, 0.0], dtype=np.float32)
    shape_b = np.asarray([0.0, 0.1], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape_a, shape_b=shape_b, fit_v1=fit_v1, fit_v2=fit_v2,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=["d0", "d1"], top_n=2, verbose=False,
    )
    assert report.intrinsic_displacement == pytest.approx(1.0, abs=1e-5)
    assert report.extrinsic_displacement == pytest.approx(1.0, abs=1e-5)
    assert report.intrinsic_fraction == pytest.approx(0.5, abs=1e-5)
    by_index = {d.dim_index: d for d in report.top_dimensions}
    assert by_index[0].intrinsic == pytest.approx(0.0, abs=1e-5)
    assert by_index[0].extrinsic == pytest.approx(-1.0, abs=1e-5)
    assert by_index[1].intrinsic == pytest.approx(1.0, abs=1e-5)
    assert by_index[1].extrinsic == pytest.approx(0.0, abs=1e-5)


def test_compute_decomposition_sigma_safe_guard_on_degenerate_dim():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit_v1 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.0])
    fit_v2 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], calibration_epoch=2)
    shape_a = np.asarray([0.0, 0.0], dtype=np.float32)
    shape_b = np.asarray([0.0, 0.7], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape_a, shape_b=shape_b, fit_v1=fit_v1, fit_v2=fit_v2,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=None, top_n=2, verbose=False,
    )
    by_index = {d.dim_index: d for d in report.top_dimensions}
    assert np.isfinite(by_index[1].intrinsic)
    assert np.isfinite(by_index[1].extrinsic)
    assert np.isfinite(report.intrinsic_displacement)


def test_compute_decomposition_verbose_returns_full_table():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit = _make_fit(mu=[0.0] * 5, sigma=[0.1] * 5)
    shape_a = np.asarray([0.0] * 5, dtype=np.float32)
    shape_b = np.asarray([0.05] * 5, dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape_a, shape_b=shape_b, fit_v1=fit, fit_v2=fit,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=None, top_n=2, verbose=True,
    )
    assert len(report.top_dimensions) == 2
    assert report.per_dimension is not None
    assert len(report.per_dimension) == 5


def test_compute_decomposition_top_n_larger_than_d_returns_all():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1])
    shape_a = np.asarray([0.0, 0.0], dtype=np.float32)
    shape_b = np.asarray([0.05, 0.05], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape_a, shape_b=shape_b, fit_v1=fit, fit_v2=fit,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=None, top_n=99, verbose=False,
    )
    assert len(report.top_dimensions) == 2


def test_compute_decomposition_dim_labels_propagate():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], dimension_kinds=["gaussian", "poisson"])
    shape_a = np.asarray([0.0, 0.0], dtype=np.float32)
    shape_b = np.asarray([0.05, 0.0], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape_a, shape_b=shape_b, fit_v1=fit, fit_v2=fit,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=["amount", "tx_count"], top_n=2, verbose=False,
    )
    by_index = {d.dim_index: d for d in report.top_dimensions}
    assert by_index[0].dim_label == "amount"
    assert by_index[0].dim_kind == "gaussian"
    assert by_index[1].dim_label == "tx_count"
    assert by_index[1].dim_kind == "poisson"


def test_compute_decomposition_dim_labels_none_uses_index_only():
    from hypertopos.engine.geometry import _compute_intrinsic_extrinsic_decomposition

    fit = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1])
    shape_a = np.asarray([0.0, 0.0], dtype=np.float32)
    shape_b = np.asarray([0.05, 0.0], dtype=np.float32)
    report = _compute_intrinsic_extrinsic_decomposition(
        shape_a=shape_a, shape_b=shape_b, fit_v1=fit, fit_v2=fit,
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        dim_labels=None, top_n=2, verbose=False,
    )
    for d in report.top_dimensions:
        assert d.dim_label is None


def _make_navigator_with_mocks(reader_mock):
    from unittest.mock import MagicMock

    from hypertopos.navigation.navigator import GDSNavigator

    return GDSNavigator(
        engine=MagicMock(),
        storage=reader_mock,
        manifest=MagicMock(agent_id="test"),
        contract=MagicMock(),
    )


def _build_temporal_table(rows: list[dict]):
    """Build a pyarrow Table with the same schema read_temporal returns."""
    import pyarrow as pa

    return pa.table(
        {
            "slice_index": pa.array([r["slice_index"] for r in rows], type=pa.int32()),
            "timestamp": pa.array([r["timestamp"] for r in rows]),
            "deformation_type": pa.array(
                [r.get("deformation_type", "window_snapshot") for r in rows]
            ),
            "shape_snapshot": pa.array([r["shape_snapshot"] for r in rows]),
            "pattern_ver": pa.array([r.get("pattern_ver", 1) for r in rows], type=pa.int32()),
            "changed_property": pa.array(
                [r.get("changed_property") for r in rows], type=pa.string(),
            ),
            "changed_line_id": pa.array(
                [r.get("changed_line_id") for r in rows], type=pa.string(),
            ),
        }
    )


def _make_reader_mock(
    *,
    versions,
    fits,
    temporal_rows,
    pattern_type="anchor",
    dim_labels=None,
):
    from unittest.mock import MagicMock

    reader = MagicMock()
    reader.list_calibration_versions.return_value = versions
    reader.read_calibration_fit.side_effect = (
        lambda pattern_id, version=None: fits[version if version is not None else versions[-1]]
    )
    reader.read_temporal.return_value = _build_temporal_table(temporal_rows)

    sphere = MagicMock()
    pattern = MagicMock()
    pattern.pattern_type = pattern_type
    pattern.dim_labels = dim_labels
    sphere.patterns = {"p": pattern}
    reader.read_sphere.return_value = sphere
    return reader


def test_decompose_drift_explicit_versions_and_timestamps():
    fit_v1 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], calibration_epoch=1)
    fit_v2 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], calibration_epoch=2)
    reader = _make_reader_mock(
        versions=[1, 2],
        fits={1: fit_v1, 2: fit_v2},
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0, 0.0]},
            {"slice_index": 1, "timestamp": _ts(2), "shape_snapshot": [0.5, 0.0]},
        ],
        dim_labels=["dim0", "dim1"],
    )
    nav = _make_navigator_with_mocks(reader)
    report = nav.decompose_drift(
        entity_key="E1", pattern_id="p", v_from=1, v_to=2,
        timestamp_from=_ts(1), timestamp_to=_ts(2),
        top_n=2, verbose=False,
    )
    assert report.entity_key == "E1"
    assert report.v_from == 1
    assert report.v_to == 2
    assert report.intrinsic_displacement == pytest.approx(5.0, abs=1e-5)
    assert report.extrinsic_displacement == pytest.approx(0.0, abs=1e-5)
    assert report.intrinsic_fraction == pytest.approx(1.0, abs=1e-5)


def test_decompose_drift_auto_resolves_to_oldest_and_current():
    fits = {
        1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1),
        2: _make_fit(mu=[0.05], sigma=[0.1], calibration_epoch=2),
        3: _make_fit(mu=[0.10], sigma=[0.1], calibration_epoch=3),
    }
    reader = _make_reader_mock(
        versions=[1, 2, 3],
        fits=fits,
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0]},
            {"slice_index": 1, "timestamp": _ts(3), "shape_snapshot": [0.10]},
        ],
    )
    nav = _make_navigator_with_mocks(reader)
    report = nav.decompose_drift(entity_key="E1", pattern_id="p", top_n=1)
    assert report.v_from == 1
    assert report.v_to == 3


def test_decompose_drift_v_to_none_resolves_to_latest():
    fits = {
        1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1),
        2: _make_fit(mu=[0.05], sigma=[0.1], calibration_epoch=2),
        3: _make_fit(mu=[0.10], sigma=[0.1], calibration_epoch=3),
    }
    reader = _make_reader_mock(
        versions=[1, 2, 3],
        fits=fits,
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0]},
            {"slice_index": 1, "timestamp": _ts(3), "shape_snapshot": [0.10]},
        ],
    )
    nav = _make_navigator_with_mocks(reader)
    report = nav.decompose_drift(entity_key="E1", pattern_id="p", v_from=1, top_n=1)
    assert report.v_from == 1
    assert report.v_to == 3


def test_decompose_drift_raises_when_versions_equal():
    fits = {
        1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1),
        2: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=2),
    }
    reader = _make_reader_mock(
        versions=[1, 2],
        fits=fits,
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0]},
            {"slice_index": 1, "timestamp": _ts(2), "shape_snapshot": [0.05]},
        ],
    )
    nav = _make_navigator_with_mocks(reader)
    with pytest.raises(ValueError, match="must differ"):
        nav.decompose_drift(entity_key="E1", pattern_id="p", v_from=2, v_to=2)


def test_decompose_drift_raises_when_only_one_epoch_and_auto():
    fits = {1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1)}
    reader = _make_reader_mock(
        versions=[1],
        fits=fits,
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0]},
            {"slice_index": 1, "timestamp": _ts(2), "shape_snapshot": [0.05]},
        ],
    )
    nav = _make_navigator_with_mocks(reader)
    with pytest.raises(ValueError, match="at least 2 epochs"):
        nav.decompose_drift(entity_key="E1", pattern_id="p")


def test_decompose_drift_raises_on_schema_mismatch():
    fits = {
        1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1, schema_hash="a" * 64),
        2: _make_fit(mu=[0.05], sigma=[0.1], calibration_epoch=2, schema_hash="b" * 64),
    }
    reader = _make_reader_mock(
        versions=[1, 2],
        fits=fits,
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0]},
            {"slice_index": 1, "timestamp": _ts(2), "shape_snapshot": [0.05]},
        ],
    )
    nav = _make_navigator_with_mocks(reader)
    with pytest.raises(ValueError, match="schema_hash"):
        nav.decompose_drift(entity_key="E1", pattern_id="p", v_from=1, v_to=2)


def test_decompose_drift_raises_on_event_pattern():
    fits = {
        1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1),
        2: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=2),
    }
    reader = _make_reader_mock(
        versions=[1, 2],
        fits=fits,
        temporal_rows=[],
        pattern_type="event",
    )
    nav = _make_navigator_with_mocks(reader)
    with pytest.raises(ValueError, match="event"):
        nav.decompose_drift(entity_key="E1", pattern_id="p")


def test_decompose_drift_raises_on_too_few_slices():
    fits = {
        1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1),
        2: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=2),
    }
    reader = _make_reader_mock(
        versions=[1, 2],
        fits=fits,
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0]},
        ],
    )
    nav = _make_navigator_with_mocks(reader)
    with pytest.raises(ValueError, match="at least 2"):
        nav.decompose_drift(entity_key="E1", pattern_id="p")


def test_decompose_drift_window_filter_slices_outside_range():
    fits = {
        1: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1),
        2: _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=2),
    }
    reader = _make_reader_mock(
        versions=[1, 2],
        fits=fits,
        temporal_rows=[
            {"slice_index": 0, "timestamp": _ts(1), "shape_snapshot": [0.0]},
            {"slice_index": 1, "timestamp": _ts(2), "shape_snapshot": [0.05]},
            {"slice_index": 2, "timestamp": _ts(3), "shape_snapshot": [0.10]},
        ],
    )
    nav = _make_navigator_with_mocks(reader)
    report = nav.decompose_drift(
        entity_key="E1", pattern_id="p",
        timestamp_from=_ts(2), timestamp_to=_ts(3),
    )
    assert report.timestamp_from == _ts(2)
    assert report.timestamp_to == _ts(3)


def test_decomposition_scalars_pure_intrinsic_drift():
    """When mu/sigma stay fixed and shape changes, intrinsic carries the whole budget."""
    from hypertopos.engine.geometry import _decomposition_scalars

    fit_v1 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], calibration_epoch=1)
    fit_v2 = _make_fit(mu=[0.0, 0.0], sigma=[0.1, 0.1], calibration_epoch=2)
    shape_a = np.array([0.0, 0.0], dtype=np.float32)
    shape_b = np.array([0.5, 0.0], dtype=np.float32)
    intr, extr, frac = _decomposition_scalars(shape_a, shape_b, fit_v1, fit_v2)
    assert intr == pytest.approx(5.0, abs=1e-5)
    assert extr == pytest.approx(0.0, abs=1e-5)
    assert frac == pytest.approx(1.0, abs=1e-5)


def test_decomposition_scalars_pure_extrinsic_drift():
    """When shape stays fixed and mu/sigma move, extrinsic carries the whole budget."""
    from hypertopos.engine.geometry import _decomposition_scalars

    fit_v1 = _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1)
    fit_v2 = _make_fit(mu=[0.5], sigma=[0.1], calibration_epoch=2)
    shape_a = np.array([0.0], dtype=np.float32)
    shape_b = np.array([0.0], dtype=np.float32)
    intr, extr, frac = _decomposition_scalars(shape_a, shape_b, fit_v1, fit_v2)
    assert intr == pytest.approx(0.0, abs=1e-5)
    assert extr == pytest.approx(5.0, abs=1e-5)
    assert frac == pytest.approx(0.0, abs=1e-5)


def test_decomposition_scalars_zero_drift_yields_zero_fraction():
    """Identical shapes and identical fits → zeros all around, fraction = 0.0 (not NaN)."""
    from hypertopos.engine.geometry import _decomposition_scalars

    fit_v1 = _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=1)
    fit_v2 = _make_fit(mu=[0.0], sigma=[0.1], calibration_epoch=2)
    shape_a = np.array([0.0], dtype=np.float32)
    shape_b = np.array([0.0], dtype=np.float32)
    intr, extr, frac = _decomposition_scalars(shape_a, shape_b, fit_v1, fit_v2)
    assert intr == pytest.approx(0.0, abs=1e-9)
    assert extr == pytest.approx(0.0, abs=1e-9)
    assert frac == 0.0


def test_decomposition_scalars_sigma_safe_on_degenerate_dim():
    """sigma_v1[i] == 0 must not blow up; helper substitutes 1.0 in the safe vector."""
    from hypertopos.engine.geometry import _decomposition_scalars

    fit_v1 = _make_fit(mu=[0.0, 0.0], sigma=[0.0, 0.1], calibration_epoch=1)
    fit_v2 = _make_fit(mu=[0.0, 0.0], sigma=[0.0, 0.1], calibration_epoch=2)
    shape_a = np.array([0.0, 0.0], dtype=np.float32)
    shape_b = np.array([1.0, 0.0], dtype=np.float32)
    intr, extr, frac = _decomposition_scalars(shape_a, shape_b, fit_v1, fit_v2)
    assert np.isfinite(intr)
    assert np.isfinite(extr)
    assert 0.0 <= frac <= 1.0
