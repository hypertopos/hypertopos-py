# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Cluster A — label-aware calibration depth extensions.

Three additive surfaces over the existing ``label_aware_calibration``
foundation:

- A1: per-dim closed-form Gaussian AUROC field on ``audit_pattern_dims``.
- A2: ``Pattern.signed_percentiles`` + ``signed_tail_concentration``
  dim_quality warning.
- A3: ``GDSEngine.decompose_displacement`` + per-pattern
  ``intrinsic_displacement_mean`` / ``extrinsic_displacement_mean``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.cli.schema import LabelAuditConfig
from hypertopos.engine.geometry import GDSEngine
from hypertopos.navigation.navigator import GDSNavigator
from hypertopos.storage.reader import GDSReader


# ─────────────────────────────────────────────────────────────────────
# A1 — per-dim AUROC field on audit_pattern_dims
# ─────────────────────────────────────────────────────────────────────


def _make_dim_cal(mu_pos: float, sigma_pos: float, mu_neg: float, sigma_neg: float, direction: float) -> SimpleNamespace:
    return SimpleNamespace(
        mu_pos=mu_pos,
        sigma_pos=sigma_pos,
        mu_neg=mu_neg,
        sigma_neg=sigma_neg,
        direction=direction,
    )


def _gaussian_auroc(mu_pos: float, sigma_pos: float, mu_neg: float, sigma_neg: float) -> float:
    """Closed-form Phi((mu_pos - mu_neg) / sqrt(sigma_pos^2 + sigma_neg^2))."""
    from scipy.special import ndtr
    sum_sq = sigma_pos ** 2 + sigma_neg ** 2
    if sum_sq <= 0.0:
        return 0.5
    return float(ndtr((mu_pos - mu_neg) / math.sqrt(sum_sq)))


def _call_audit_pattern_dims(pattern: object, *, pattern_id: str = "synth_pat", top_k: int = 10) -> dict:
    """Invoke audit_pattern_dims against an in-memory pattern stand-in."""
    import hypertopos_mcp.tools.observability  # noqa: F401 — register tool
    from hypertopos_mcp.server import _state
    from hypertopos_mcp.tools.observability import audit_pattern_dims

    saved_nav = _state.get("navigator")
    saved_sphere = _state.get("sphere")
    try:
        sphere_wrapper = MagicMock()
        sphere_wrapper._sphere = SimpleNamespace(patterns={pattern_id: pattern})
        _state["navigator"] = MagicMock()
        _state["sphere"] = sphere_wrapper
        body = audit_pattern_dims(pattern_id=pattern_id, top_k=top_k)
        return json.loads(body)
    finally:
        _state["navigator"] = saved_nav
        _state["sphere"] = saved_sphere


def test_a1_auroc_per_dim_matches_closed_form_three_dims():
    """Three engineered dims with distinct (mu_pos, mu_neg, sigma_pos,
    sigma_neg) tuples — auroc_per_dim must equal
    Phi((mu_pos - mu_neg) / sqrt(sigma_pos^2 + sigma_neg^2)) to 6 dp.
    """
    dim_labels = ["dim_a", "dim_b", "dim_c"]
    tuples = [
        (2.0, 0.5, 0.0, 0.5),    # strong separation, small spread
        (1.0, 1.0, 0.0, 1.5),    # moderate separation, asymmetric spread
        (0.1, 2.0, 0.0, 2.0),    # weak separation, large spread
    ]
    lac = {
        label: _make_dim_cal(mp, sp, mn, sn, direction=0.5)
        for label, (mp, sp, mn, sn) in zip(dim_labels, tuples, strict=True)
    }

    pattern = SimpleNamespace(
        pattern_id="synth_pat",
        dim_labels=dim_labels,
        mu=np.array([1.0, 0.5, 0.05], dtype=np.float32),
        sigma_diag=np.array([0.5, 1.25, 2.0], dtype=np.float32),
        relations=[],
        event_dimensions=[],
        prop_columns=[],
        label_aware_calibration=lac,
        dimension_kinds=None,
        intrinsic_displacement_mean=None,
        extrinsic_displacement_mean=None,
    )

    parsed = _call_audit_pattern_dims(pattern)
    assert parsed["label_aware_available"] is True
    by_label = {row["dim_label"]: row for row in parsed["dims"]}
    for label, (mp, sp, mn, sn) in zip(dim_labels, tuples, strict=True):
        expected = _gaussian_auroc(mp, sp, mn, sn)
        got = by_label[label]["auroc_per_dim"]
        assert got == pytest.approx(expected, abs=1e-6), (
            f"{label}: got {got}, expected {expected}"
        )


def test_a1_auroc_per_dim_degenerate_zero_sigmas():
    """When both class sigmas are zero, AUROC is undefined — report 0.5.

    Guards the closed-form against divide-by-zero blowing up the response.
    """
    lac = {"only_dim": _make_dim_cal(1.0, 0.0, 0.0, 0.0, direction=0.0)}
    pattern = SimpleNamespace(
        pattern_id="degen_pat",
        dim_labels=["only_dim"],
        mu=np.array([0.5], dtype=np.float32),
        sigma_diag=np.array([0.5], dtype=np.float32),
        relations=[],
        event_dimensions=[],
        prop_columns=[],
        label_aware_calibration=lac,
        dimension_kinds=None,
        intrinsic_displacement_mean=None,
        extrinsic_displacement_mean=None,
    )

    parsed = _call_audit_pattern_dims(pattern, pattern_id="degen_pat")
    assert parsed["dims"][0]["auroc_per_dim"] == pytest.approx(0.5, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────
# A2 — signed_tail_concentration warning
# ─────────────────────────────────────────────────────────────────────


def _pattern_with_signed_percentiles(
    *,
    p1: float,
    p5: float,
    p50: float,
    p95: float,
    p99: float,
    n_pos: int,
    pattern_id: str = "tail_pat",
) -> SimpleNamespace:
    """Minimal Pattern stand-in for the warning computation."""
    return SimpleNamespace(
        pattern_id=pattern_id,
        signed_percentiles={
            "p1": p1, "p5": p5, "p50": p50, "p95": p95, "p99": p99,
        },
        label_aware_n_pos=n_pos,
    )


def test_a2_signed_tail_concentration_fires_above_threshold():
    """Ratio 100 (p99=100, p50=1) with n_pos >> 30 → warning fires."""
    pat = _pattern_with_signed_percentiles(
        p1=-1.0, p5=-0.5, p50=1.0, p95=10.0, p99=100.0, n_pos=120,
    )
    w = GDSNavigator._compute_signed_tail_concentration_warning(pat)
    assert w is not None
    assert w["type"] == "signed_tail_concentration"
    assert w["dim_label"] == "tail_pat:signed_percentiles"
    assert w["evidence_value"] == pytest.approx(100.0, rel=1e-3)
    assert w["threshold"] == 50.0


def test_a2_signed_tail_concentration_does_not_fire_below_threshold():
    """Ratio 10 (p99=10, p50=1) — under threshold (50), no warning."""
    pat = _pattern_with_signed_percentiles(
        p1=-1.0, p5=-0.5, p50=1.0, p95=5.0, p99=10.0, n_pos=120,
    )
    w = GDSNavigator._compute_signed_tail_concentration_warning(pat)
    assert w is None


def test_a2_signed_tail_concentration_suppressed_when_n_pos_low():
    """n_pos < 30 — warning suppressed even when ratio crosses 50."""
    pat = _pattern_with_signed_percentiles(
        p1=-1.0, p5=-0.5, p50=1.0, p95=10.0, p99=100.0, n_pos=10,
    )
    w = GDSNavigator._compute_signed_tail_concentration_warning(pat)
    assert w is None


def test_a2_signed_tail_concentration_skipped_without_signed_percentiles():
    """Pattern lacking signed_percentiles → no warning regardless of n_pos."""
    pat = SimpleNamespace(
        pattern_id="legacy_pat",
        signed_percentiles=None,
        label_aware_n_pos=120,
    )
    assert GDSNavigator._compute_signed_tail_concentration_warning(pat) is None


# ─────────────────────────────────────────────────────────────────────
# A3 — decompose_displacement + intrinsic/extrinsic identity
# ─────────────────────────────────────────────────────────────────────


def test_a3_decompose_displacement_perpendicular():
    """Delta perpendicular to label_direction → intrinsic=0, extrinsic=||delta||."""
    delta = np.array([0.0, 3.0, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    r = GDSEngine.decompose_displacement(delta, direction)
    assert r["intrinsic"] == pytest.approx(0.0, abs=1e-12)
    assert r["extrinsic"] == pytest.approx(3.0, abs=1e-12)


def test_a3_decompose_displacement_parallel():
    """Delta parallel to label_direction → intrinsic=||delta||, extrinsic=0."""
    delta = np.array([4.0, 0.0, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    r = GDSEngine.decompose_displacement(delta, direction)
    assert r["intrinsic"] == pytest.approx(4.0, abs=1e-12)
    assert r["extrinsic"] == pytest.approx(0.0, abs=1e-12)


def test_a3_decompose_displacement_antiparallel_uses_absolute_value():
    """Negative dot product still yields non-negative intrinsic magnitude."""
    delta = np.array([-4.0, 0.0, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    r = GDSEngine.decompose_displacement(delta, direction)
    assert r["intrinsic"] == pytest.approx(4.0, abs=1e-12)
    assert r["extrinsic"] == pytest.approx(0.0, abs=1e-12)


def test_a3_decompose_displacement_preservation_identity():
    """intrinsic^2 + extrinsic^2 == ||delta||^2 to 6 decimals — random vectors."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        d = rng.normal(0.0, 1.0, size=8).astype(np.float64)
        ld = rng.normal(0.0, 1.0, size=8).astype(np.float64)
        r = GDSEngine.decompose_displacement(d, ld)
        identity_lhs = r["intrinsic"] ** 2 + r["extrinsic"] ** 2
        identity_rhs = float(np.dot(d, d))
        assert identity_lhs == pytest.approx(identity_rhs, abs=1e-6), (
            f"identity violated: lhs={identity_lhs} rhs={identity_rhs} "
            f"d={d}, ld={ld}"
        )


def test_a3_decompose_displacement_normalises_non_unit_direction():
    """Non-unit direction must be normalised internally; result invariant
    under positive scalar rescaling of direction.
    """
    delta = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    ld_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ld_scaled = ld_unit * 7.5
    r_unit = GDSEngine.decompose_displacement(delta, ld_unit)
    r_scaled = GDSEngine.decompose_displacement(delta, ld_scaled)
    assert r_unit["intrinsic"] == pytest.approx(r_scaled["intrinsic"], abs=1e-12)
    assert r_unit["extrinsic"] == pytest.approx(r_scaled["extrinsic"], abs=1e-12)


def test_a3_decompose_displacement_zero_direction_returns_zeros():
    """Zero-norm label direction — no axis to project onto; both magnitudes 0."""
    delta = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    direction = np.zeros(3, dtype=np.float64)
    r = GDSEngine.decompose_displacement(delta, direction)
    assert r["intrinsic"] == 0.0
    assert r["extrinsic"] == 0.0


def test_a3_decompose_displacement_shape_mismatch_raises():
    """Mismatched shapes — ValueError, not silent garbage."""
    with pytest.raises(ValueError, match="shape mismatch"):
        GDSEngine.decompose_displacement(
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0, 0.0]),
        )


# ─────────────────────────────────────────────────────────────────────
# Round-trip — build → reader → Pattern carries new fields
# ─────────────────────────────────────────────────────────────────────


def _build_two_class_sphere_for_round_trip(
    tmp_path: Path,
    *,
    n_per_class: int = 80,
    out_dir_name: str = "gds_cluster_a_round_trip",
) -> str:
    """Minimal label-aware sphere — identical shape to test_m1_1 fixture."""
    rng = np.random.RandomState(17)
    n = 2 * n_per_class
    labels_pyarr = ["anom"] * n_per_class + ["norm"] * n_per_class
    sep_pos = rng.normal(2.5, 1.0, n_per_class).astype(np.float32)
    sep_neg = rng.normal(0.0, 1.0, n_per_class).astype(np.float32)
    sep_score = np.concatenate([sep_pos, sep_neg])
    noise_score = rng.normal(0.0, 1.0, n).astype(np.float32)

    pks = [f"T-{i:04d}" for i in range(n)]
    tx = pa.table({
        "tx_id": pks,
        "account_id": ["A-shared"] * n,
        "sep_score": sep_score,
        "noise_score": noise_score,
        "label": labels_pyarr,
    })
    accounts = pa.table({"account_id": ["A-shared"]})

    out_path = tmp_path / out_dir_name
    b = GDSBuilder("two_class_round_trip", str(out_path))
    b.add_line(
        "accounts", accounts, key_col="account_id", source_id="test",
    )
    b.add_line(
        "tx", tx, key_col="tx_id", source_id="test", role="event",
    )
    b.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="tx",
        relations=[
            RelationSpec(
                line_id="accounts", fk_col="account_id",
                direction="in", required=True,
            ),
        ],
        anomaly_percentile=95.0,
    )
    b.add_event_dimension("tx_pattern", column="sep_score", edge_max="auto")
    b.add_event_dimension("tx_pattern", column="noise_score", edge_max="auto")

    b._label_aware_calibration = True
    b._label_audit_block = LabelAuditConfig(
        label_column="label",
        label_positive_value="anom",
        patterns=["tx_pattern"],
    )
    return b.build()


def test_round_trip_signed_percentiles_and_decomposition_means(tmp_path):
    """Build → reader round trip preserves signed_percentiles +
    intrinsic/extrinsic displacement means + n_pos / n_neg counts.
    """
    out = _build_two_class_sphere_for_round_trip(tmp_path)
    reader = GDSReader(out)
    pat = reader.read_sphere().patterns["tx_pattern"]

    # n_pos / n_neg are persisted with the calibration result.
    assert pat.label_aware_n_pos == 80
    assert pat.label_aware_n_neg == 80

    # signed_percentiles populated with the five canonical keys.
    assert pat.signed_percentiles is not None
    assert set(pat.signed_percentiles.keys()) == {"p1", "p5", "p50", "p95", "p99"}
    # Sanity ordering: p1 <= p5 <= p50 <= p95 <= p99.
    sp = pat.signed_percentiles
    assert sp["p1"] <= sp["p5"] <= sp["p50"] <= sp["p95"] <= sp["p99"]

    # Intrinsic + extrinsic means are non-negative finite floats.
    assert pat.intrinsic_displacement_mean is not None
    assert pat.extrinsic_displacement_mean is not None
    assert pat.intrinsic_displacement_mean >= 0.0
    assert pat.extrinsic_displacement_mean >= 0.0
    assert np.isfinite(pat.intrinsic_displacement_mean)
    assert np.isfinite(pat.extrinsic_displacement_mean)


def test_dim_quality_warnings_returns_list_with_signed_percentiles(tmp_path):
    """End-to-end wiring smoke: a Pattern carrying ``signed_percentiles``
    must round-trip through ``_compute_dim_quality_warnings`` without
    raising. Guards the navigator-side append at the call site
    (typo / shape mismatch would surface here rather than at MCP smoke).
    """
    out = _build_two_class_sphere_for_round_trip(
        tmp_path, out_dir_name="gds_cluster_a_warnings_wiring",
    )
    pat = GDSReader(out).read_sphere().patterns["tx_pattern"]
    assert pat.signed_percentiles is not None
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    # Smoke contract: returns a list (possibly empty), never raises.
    assert isinstance(warnings, list)


def test_audit_pattern_dims_surfaces_decomposition_means(tmp_path):
    """End-to-end: build → audit_pattern_dims returns pattern-level
    intrinsic/extrinsic displacement means in the top-level response.
    """
    out = _build_two_class_sphere_for_round_trip(
        tmp_path, out_dir_name="gds_cluster_a_audit_e2e",
    )
    sphere = GDSReader(out).read_sphere()
    pat = sphere.patterns["tx_pattern"]

    parsed = _call_audit_pattern_dims(pat, pattern_id="tx_pattern")
    assert parsed["label_aware_available"] is True
    assert parsed["intrinsic_displacement_mean"] is not None
    assert parsed["extrinsic_displacement_mean"] is not None
    assert parsed["intrinsic_displacement_mean"] == pytest.approx(
        pat.intrinsic_displacement_mean, abs=1e-6,
    )
    assert parsed["extrinsic_displacement_mean"] == pytest.approx(
        pat.extrinsic_displacement_mean, abs=1e-6,
    )
    # Per-row auroc_per_dim populated when label-aware fired.
    for row in parsed["dims"]:
        assert row["auroc_per_dim"] is not None
        assert 0.0 <= row["auroc_per_dim"] <= 1.0
