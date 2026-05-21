# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``kind_mismatch`` warning emitted by
``_compute_dim_quality_warnings``.

Fires when a dim declared with ``kind='gaussian'`` shows a near-zero
Fisher LDA direction component (the dim does not carry the
label-discriminating signal in the global axis) while the raw per-class
moments still separate (``|cohens_d_pos_neg| >= 0.3``). The combination
means the dim's variance is captured by another dim's Fisher axis —
re-declaring kind or splitting the dim is recommended.

Suppressed by index when ``negative_space`` already fires on the same
dim — the negative_space remediation supersedes the kind_mismatch
complaint (same pattern as ``non_normal_dim``).
"""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest  # noqa: F401 — keeps parity with the sibling test module
from hypertopos.model.sphere import Pattern, RelationDef
from hypertopos.navigation.navigator import GDSNavigator


def _dim_cal(*, mu_pos, sigma_pos, mu_neg, sigma_neg, direction):
    """Minimal stand-in for ``engine.calibration_label_aware.DimCalibration``.

    Attribute access only — keeps the test decoupled from the dataclass.
    """
    return SimpleNamespace(
        mu_pos=mu_pos,
        sigma_pos=sigma_pos,
        mu_neg=mu_neg,
        sigma_neg=sigma_neg,
        direction=direction,
    )


def _make_pattern(
    *,
    sigma_diag: list[float],
    dim_percentiles: dict | None = None,
    relations: list[RelationDef] | None = None,
    dimension_kinds: list[str] | None = None,
    label_aware_calibration: dict | None = None,
) -> Pattern:
    rels = relations or [
        RelationDef(line_id=f"line_{i}", direction="in", required=True)
        for i in range(len(sigma_diag))
    ]
    return Pattern(
        pattern_id="p_test",
        entity_type="test",
        pattern_type="anchor",
        relations=rels,
        mu=np.zeros(len(sigma_diag), dtype=np.float32),
        sigma_diag=np.asarray(sigma_diag, dtype=np.float32),
        theta=np.ones(len(sigma_diag), dtype=np.float32),
        population_size=1000,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        dim_percentiles=dim_percentiles,
        dimension_kinds=dimension_kinds,
        label_aware_calibration=label_aware_calibration,
    )


# Cohen's d ≈ 0.5 / sqrt((1+1)/2) = 0.5 — well above the 0.3 gate.
_HIGH_D_CAL = {"mu_pos": 0.5, "sigma_pos": 1.0, "mu_neg": 0.0, "sigma_neg": 1.0}
# Cohen's d ≈ 0.0 — below the 0.3 gate.
_ZERO_D_CAL = {"mu_pos": 0.0, "sigma_pos": 1.0, "mu_neg": 0.0, "sigma_neg": 1.0}


def test_kind_mismatch_fires_when_gaussian_high_d_low_direction():
    """Dim A: cohens_d ≈ 0.5, |direction| = 0.02, kind='gaussian' → FIRES."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=["gaussian"],
        label_aware_calibration={
            "line_0": _dim_cal(direction=0.02, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    mismatch = [w for w in warnings if w["type"] == "kind_mismatch"]
    assert len(mismatch) == 1
    w = mismatch[0]
    assert w["dim_label"] == "line_0"
    assert "kind=gaussian" in w["reason"]
    assert "direction" in w["reason"]
    assert "cohens_d" in w["reason"]
    assert "bernoulli" in w["advice"] or "poisson" in w["advice"]
    # |direction_component| surfaced as evidence_value, threshold = 0.05
    assert w["evidence_value"] == pytest.approx(0.02, abs=1e-9)
    assert w["threshold"] == 0.05


def test_kind_mismatch_silent_when_direction_above_threshold():
    """Dim B: cohens_d ≈ 0.5, |direction| = 0.6, kind='gaussian' → NOT FIRED."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=["gaussian"],
        label_aware_calibration={
            "line_0": _dim_cal(direction=0.6, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "kind_mismatch" for w in warnings)


def test_kind_mismatch_silent_when_cohens_d_below_threshold():
    """Dim C: cohens_d = 0, |direction| = 0.02, kind='gaussian' → NOT FIRED."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=["gaussian"],
        label_aware_calibration={
            "line_0": _dim_cal(direction=0.02, **_ZERO_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "kind_mismatch" for w in warnings)


def test_kind_mismatch_suppressed_when_negative_space_fires():
    """Dim D: gaussian + zero-direction + high cohens_d AND p50=0 →
    negative_space fires; kind_mismatch is suppressed (same dim,
    negative_space remediation supersedes)."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=["gaussian"],
        # p50=0 with positive p99 → negative_space fires on this dim.
        dim_percentiles={
            "line_0": {"p25": 0.0, "p50": 0.0, "p75": 0.0,
                       "p99": 5.0, "max": 100.0},
        },
        label_aware_calibration={
            "line_0": _dim_cal(direction=0.02, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    types_on_dim = [w["type"] for w in warnings if w["dim_label"] == "line_0"]
    assert "negative_space" in types_on_dim
    assert "kind_mismatch" not in types_on_dim


def test_kind_mismatch_skipped_when_no_label_aware_calibration():
    """Pre-requisite: ``Pattern.label_aware_calibration`` must be
    non-None. Without it the navigator silently skips the audit (no
    direction_component to test against)."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=["gaussian"],
        label_aware_calibration=None,
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "kind_mismatch" for w in warnings)


def test_kind_mismatch_skipped_for_bernoulli_kind():
    """Bernoulli dim is silently ignored even with high cohens_d + low
    direction — the warning's remediation does not apply to binary
    kinds."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=["bernoulli"],
        label_aware_calibration={
            "line_0": _dim_cal(direction=0.02, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "kind_mismatch" for w in warnings)


def test_kind_mismatch_uses_absolute_direction():
    """A strongly negative direction (e.g. -0.6) does NOT fire — the
    threshold is on |direction|, mirroring the existing
    investigate_drift action on the MCP tool."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=["gaussian"],
        label_aware_calibration={
            "line_0": _dim_cal(direction=-0.6, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "kind_mismatch" for w in warnings)


def test_kind_mismatch_multiple_dims_independent():
    """Four-dim engineered fixture from the M1.4 brief: dim A fires,
    dims B / C / D don't (with D's reason being the negative_space
    suppression)."""
    pat = _make_pattern(
        sigma_diag=[1.0, 1.0, 1.0, 1.0],
        dimension_kinds=["gaussian", "gaussian", "gaussian", "gaussian"],
        # Only dim D has the zero-p50 percentile pattern that triggers
        # negative_space — dims A / B / C have no percentile coverage at
        # all, so the negative_space auditor returns nothing for them.
        dim_percentiles={
            "line_3": {"p25": 0.0, "p50": 0.0, "p75": 0.0,
                       "p99": 5.0, "max": 100.0},
        },
        label_aware_calibration={
            # A: HIGH cohens_d, low direction, gaussian → FIRES
            "line_0": _dim_cal(direction=0.02, **_HIGH_D_CAL),
            # B: HIGH cohens_d, HIGH direction → does NOT fire
            "line_1": _dim_cal(direction=0.6, **_HIGH_D_CAL),
            # C: ZERO cohens_d, low direction → does NOT fire
            "line_2": _dim_cal(direction=0.02, **_ZERO_D_CAL),
            # D: HIGH cohens_d, low direction, BUT negative_space → suppressed
            "line_3": _dim_cal(direction=0.02, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    mismatch_labels = [
        w["dim_label"] for w in warnings if w["type"] == "kind_mismatch"
    ]
    assert mismatch_labels == ["line_0"]


def test_kind_mismatch_skipped_when_dimension_kinds_absent():
    """Legacy pattern without ``dimension_kinds`` set → no kind_mismatch
    warning fires (no way to confirm the dim is gaussian)."""
    pat = _make_pattern(
        sigma_diag=[1.0],
        dimension_kinds=None,
        label_aware_calibration={
            "line_0": _dim_cal(direction=0.02, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    assert all(w["type"] != "kind_mismatch" for w in warnings)


def test_kind_mismatch_suppression_by_index_when_naming_schemes_differ():
    """Suppression must resolve via the shared raw-to-index mapping,
    not by direct string match on dim_label.

    Sets up a pattern where ``negative_space`` and ``label_aware_calibration``
    key the same dim under different names:

    - relation ``line_id="_d_foo"`` → ``dim_labels[0] = "_d_foo"``
      (consumed by the kind_mismatch auditor and the label-aware
      calibration map)
    - ``dim_percentiles["foo"]`` → ``negative_space.dim_label = "foo"``
      (consumed by the negative_space auditor via the
      ``_d_``-stripping rule in ``_build_raw_dim_name_to_index``)

    If suppression were keyed on dim_label strings the two paths would
    not join and kind_mismatch would fire alongside negative_space.
    With index-based suppression both auditors land on i=0 and
    kind_mismatch is silenced.
    """
    from hypertopos.model.sphere import RelationDef as _RD
    pat = _make_pattern(
        sigma_diag=[1.0],
        relations=[_RD(line_id="_d_foo", direction="in", required=True)],
        dimension_kinds=["gaussian"],
        # negative_space keys off raw name "foo" (after stripping the
        # "_d_" prefix on the line_id).
        dim_percentiles={
            "foo": {"p25": 0.0, "p50": 0.0, "p75": 0.0,
                    "p99": 5.0, "max": 100.0},
        },
        # label-aware calibration keys off the full dim_label "_d_foo"
        # (this is what the builder emits via _dim_labels_for_pattern).
        label_aware_calibration={
            "_d_foo": _dim_cal(direction=0.02, **_HIGH_D_CAL),
        },
    )
    warnings = GDSNavigator._compute_dim_quality_warnings(pat)
    by_type = {w["type"] for w in warnings}
    assert "negative_space" in by_type
    assert "kind_mismatch" not in by_type
