# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Signed-confidence ranking on ``find_anomalies`` (M2.1, O2).

The new ``rank_by="signed_confidence"`` mode composes three already-shipped
signals — the ``delta_norm_signed`` Lance column, the Fisher LDA direction
alignment, and the reliability flags — into one confidence-weighted score:

    score = delta_norm_signed × |lda_alignment| × (1 − reliability_penalty)

These tests cover:
- A synthetic 2-class sphere where one dim carries the label signal and a
  second carries a misleading anti-aligned high-magnitude signature. Top-N
  signed_confidence ranking captures more positive-class polygons than the
  delta_norm baseline.
- An engineered deterministic test on the
  ``_attach_signed_confidence_fields`` primitive — three polygons with
  hand-picked deltas + reliability flags assert the ranking matches the
  formula bit-for-bit.
- Pattern without ``label_aware_calibration`` → ``find_anomalies(rank_by=
  "signed_confidence")`` raises ``GDSNavigationError`` (not ``ValueError``)
  with the explicit message documented in the brief.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.cli.schema import LabelAuditConfig
from hypertopos.model.objects import Polygon
from hypertopos.navigation.navigator import GDSNavigationError, GDSNavigator
from hypertopos.sphere import HyperSphere


def _build_signed_confidence_fixture(
    tmp_path: Path,
    *,
    n_per_class: int = 80,
    n_anti_aligned: int = 25,
    sep_mean: float = 3.0,
    noise_outlier_mean: float = 6.0,
    seed: int = 1234,
    out_dir_name: str = "gds_signed_conf_fixture",
) -> tuple[str, list[str], np.ndarray]:
    """2-class fixture with anti-aligned high-magnitude outliers.

    Layout:
    - ``n_per_class`` positive-class transactions with ``sep_score ~
      N(sep_mean, 1)`` and ``noise_score ~ N(0, 1)``.
    - ``n_per_class`` negative-class transactions with ``sep_score ~
      N(0, 1)`` and ``noise_score ~ N(0, 1)``.
    - ``n_anti_aligned`` extra negative-class transactions with
      ``sep_score ~ N(0, 1)`` and ``noise_score ~ N(noise_outlier_mean,
      1)``. These have high ``||delta||`` (dominated by the noise dim)
      but minimal projection on the Fisher LDA direction (which points
      along sep_score). Under ``rank_by="delta_norm"`` they crowd the
      top-N, displacing genuine positive-class polygons. Under ``rank_by=
      "signed_confidence"`` they get demoted (low ``|lda_alignment|``)
      and the positive-class polygons are recovered.

    Returns ``(sphere_path, pks, labels)`` where ``labels[i]`` is 1 for
    positive-class and 0 for negative-class (including anti-aligned
    outliers).
    """
    rng = np.random.RandomState(seed)
    n_pos = n_per_class
    n_neg = n_per_class + n_anti_aligned
    n = n_pos + n_neg

    sep_pos = rng.normal(sep_mean, 1.0, n_pos).astype(np.float32)
    sep_neg = rng.normal(0.0, 1.0, n_per_class).astype(np.float32)
    sep_anti = rng.normal(0.0, 1.0, n_anti_aligned).astype(np.float32)
    sep_score = np.concatenate([sep_pos, sep_neg, sep_anti])

    noise_pos = rng.normal(0.0, 1.0, n_pos).astype(np.float32)
    noise_neg = rng.normal(0.0, 1.0, n_per_class).astype(np.float32)
    noise_anti = rng.normal(noise_outlier_mean, 1.0, n_anti_aligned).astype(
        np.float32,
    )
    noise_score = np.concatenate([noise_pos, noise_neg, noise_anti])

    pks = [f"T-{i:04d}" for i in range(n)]
    labels_pyarr = (
        ["anom"] * n_pos + ["norm"] * n_per_class + ["norm"] * n_anti_aligned
    )
    labels_arr = np.array(
        [1] * n_pos + [0] * n_per_class + [0] * n_anti_aligned, dtype=np.int32,
    )

    tx = pa.table({
        "tx_id": pks,
        "account_id": ["A-shared"] * n,
        "sep_score": sep_score,
        "noise_score": noise_score,
        "label": labels_pyarr,
    })
    accounts = pa.table({"account_id": ["A-shared"]})

    out_path = tmp_path / out_dir_name
    b = GDSBuilder("signed_conf_fixture", str(out_path))
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
        anomaly_percentile=80.0,
    )
    b.add_event_dimension(
        "tx_pattern", column="sep_score", edge_max="auto",
    )
    b.add_event_dimension(
        "tx_pattern", column="noise_score", edge_max="auto",
    )

    b._label_aware_calibration = True
    b._label_audit_block = LabelAuditConfig(
        label_column="label",
        label_positive_value="anom",
        patterns=["tx_pattern"],
    )

    out = b.build()
    return out, pks, labels_arr


def _open_navigator(sphere_path: str) -> tuple[HyperSphere, GDSNavigator]:
    sphere = HyperSphere.open(sphere_path)
    session = sphere.session("test-agent")
    return sphere, session.navigator()


def test_signed_confidence_lifts_positive_class_recall(tmp_path):
    """signed_confidence top-N captures more positives than delta_norm top-N.

    The anti-aligned outlier polygons have high ``||delta||`` (dominated
    by the ``noise_score`` dim) but tiny ``|lda_alignment|`` (the LDA
    direction lives almost entirely on ``sep_score``). Under the default
    ``rank_by="delta_norm"`` they crowd the top-N, hiding the true
    positive-class polygons. Under ``rank_by="signed_confidence"`` they
    get demoted and recall on the positive class lifts measurably.
    """
    out, pks, labels = _build_signed_confidence_fixture(
        tmp_path,
        n_per_class=80,
        n_anti_aligned=30,
        sep_mean=3.5,
        noise_outlier_mean=8.0,
        out_dir_name="gds_signed_conf_recall",
    )
    pk_to_label = dict(zip(pks, labels, strict=True))

    _sphere, nav = _open_navigator(out)
    top_n = 20
    polys_delta, _, _, _ = nav.π5_attract_anomaly(
        "tx_pattern", top_n=top_n, rank_by="delta_norm",
    )
    polys_signed, _, _, _ = nav.π5_attract_anomaly(
        "tx_pattern", top_n=top_n, rank_by="signed_confidence",
    )

    recall_delta = sum(
        1 for p in polys_delta if pk_to_label.get(p.primary_key) == 1
    )
    recall_signed = sum(
        1 for p in polys_signed if pk_to_label.get(p.primary_key) == 1
    )
    # Acceptance: signed_confidence captures at least 0.1 (in absolute
    # recall@top_n) more positive-class polygons than the delta_norm
    # baseline on the same data.
    lift = (recall_signed - recall_delta) / float(top_n)
    assert lift >= 0.1, (
        f"signed_confidence recall@{top_n}={recall_signed} "
        f"delta_norm recall@{top_n}={recall_delta} "
        f"absolute lift={lift:.3f} (expected >= 0.1)"
    )

    # Each surviving polygon carries the triad fields.
    for p in polys_signed:
        assert hasattr(p, "signed_confidence_score")
        assert hasattr(p, "lda_alignment")
        assert hasattr(p, "reliability_penalty")
        assert -1.0 <= float(p.lda_alignment) <= 1.0
        assert 0.0 <= float(p.reliability_penalty) <= 1.0


def _make_polygon(
    pk: str,
    delta: list[float],
    *,
    pattern_id: str = "tx_pattern",
) -> Polygon:
    arr = np.array(delta, dtype=np.float32)
    return Polygon(
        primary_key=pk,
        pattern_id=pattern_id,
        pattern_ver=1,
        pattern_type="event",
        scale=0,
        delta=arr,
        delta_norm=float(np.linalg.norm(arr)),
        is_anomaly=True,
        edges=[],
        last_refresh_at=None,
        updated_at=None,
    )


def test_signed_confidence_formula_deterministic_ranking():
    """Engineered polygons hit each branch of the score formula.

    Three polygons with known deltas + injected ``reliability_flags`` —
    the helper computes ``signed_confidence_score`` per polygon and the
    ranking matches the algebraically-expected ordering.
    """
    # 2-D delta space; LDA direction points along dim 0.
    # dim_labels match the pattern stub below.
    direction = {
        "sep": SimpleNamespace(direction=1.0),
        "noise": SimpleNamespace(direction=0.0),
    }
    pattern = SimpleNamespace(
        label_aware_calibration=direction,
        dim_labels=["sep", "noise"],
    )

    # P1: aligned, large, no reliability penalty.
    #     delta = [10, 0] → signed = 10, alignment = 1.0, penalty = 0.0
    #     score = 10 * 1.0 * 1.0 = 10.0
    p1 = _make_polygon("P1", [10.0, 0.0])
    p1.reliability_flags = {
        "single_dim_driven": False, "low_confidence_bucket": False,
    }
    # P2: aligned but single-dim-driven penalty.
    #     delta = [10, 0] → signed = 10, alignment = 1.0, penalty = 0.5
    #     score = 10 * 1.0 * 0.5 = 5.0
    p2 = _make_polygon("P2", [10.0, 0.0])
    p2.reliability_flags = {
        "single_dim_driven": True, "low_confidence_bucket": False,
    }
    # P3: high magnitude but orthogonal to LDA direction.
    #     delta = [0, 10] → signed = 0, alignment = 0.0, penalty = 0.0
    #     score = 0
    p3 = _make_polygon("P3", [0.0, 10.0])
    p3.reliability_flags = {
        "single_dim_driven": False, "low_confidence_bucket": False,
    }

    polys = [p1, p2, p3]
    GDSNavigator._attach_signed_confidence_fields(polys, pattern=pattern)

    assert p1.signed_confidence_score == pytest.approx(10.0, abs=1e-6)
    assert p2.signed_confidence_score == pytest.approx(5.0, abs=1e-6)
    assert p3.signed_confidence_score == pytest.approx(0.0, abs=1e-6)

    assert p1.lda_alignment == pytest.approx(1.0, abs=1e-6)
    assert p2.lda_alignment == pytest.approx(1.0, abs=1e-6)
    assert p3.lda_alignment == pytest.approx(0.0, abs=1e-6)

    assert p1.reliability_penalty == pytest.approx(0.0, abs=1e-6)
    assert p2.reliability_penalty == pytest.approx(0.5, abs=1e-6)
    assert p3.reliability_penalty == pytest.approx(0.0, abs=1e-6)

    # Sort by score descending — ordering matches the formula.
    polys_sorted = sorted(
        polys,
        key=lambda p: (
            -float(getattr(p, "signed_confidence_score", 0.0) or 0.0),
            p.primary_key,
        ),
    )
    assert [p.primary_key for p in polys_sorted] == ["P1", "P2", "P3"]


def test_signed_confidence_sign_preserves_anti_aligned():
    """Anti-aligned polygons get negative scores and sort to the bottom.

    The brief is explicit — the formula keeps the sign of
    ``delta_norm_signed`` so polygons pushed away from the positive
    centroid receive negative scores. The sort by ``-score`` then puts
    them at the bottom, not the top.
    """
    direction = {
        "sep": SimpleNamespace(direction=1.0),
        "noise": SimpleNamespace(direction=0.0),
    }
    pattern = SimpleNamespace(
        label_aware_calibration=direction,
        dim_labels=["sep", "noise"],
    )
    # Positive-aligned polygon.
    pos = _make_polygon("POS", [5.0, 0.0])
    pos.reliability_flags = {
        "single_dim_driven": False, "low_confidence_bucket": False,
    }
    # Anti-aligned polygon (delta on negative side of the LDA axis).
    neg = _make_polygon("NEG", [-5.0, 0.0])
    neg.reliability_flags = {
        "single_dim_driven": False, "low_confidence_bucket": False,
    }
    GDSNavigator._attach_signed_confidence_fields([pos, neg], pattern=pattern)
    assert pos.signed_confidence_score == pytest.approx(5.0, abs=1e-6)
    assert neg.signed_confidence_score == pytest.approx(-5.0, abs=1e-6)


def test_signed_confidence_zero_norm_safe():
    """Zero-norm polygons receive ``lda_alignment = 0`` (no div-by-zero)."""
    direction = {
        "sep": SimpleNamespace(direction=1.0),
        "noise": SimpleNamespace(direction=0.0),
    }
    pattern = SimpleNamespace(
        label_aware_calibration=direction,
        dim_labels=["sep", "noise"],
    )
    zero = _make_polygon("Z0", [0.0, 0.0])
    zero.reliability_flags = {
        "single_dim_driven": False, "low_confidence_bucket": False,
    }
    GDSNavigator._attach_signed_confidence_fields([zero], pattern=pattern)
    assert zero.lda_alignment == 0.0
    assert zero.signed_confidence_score == 0.0


def test_signed_confidence_requires_label_aware_calibration(tmp_path):
    """Pattern without ``label_aware_calibration`` → ``GDSNavigationError``.

    Brief is explicit: fail fast, no silent degrade to ``delta_norm``.
    The error message must point the agent at rebuilding with
    ``label_audit:`` enabled.
    """
    # Build a sphere WITHOUT the label_audit block.
    n = 60
    rng = np.random.RandomState(0)
    pks = [f"T-{i:04d}" for i in range(n)]
    tx = pa.table({
        "tx_id": pks,
        "account_id": ["A-shared"] * n,
        "sep_score": rng.normal(0, 1, n).astype(np.float32),
    })
    accounts = pa.table({"account_id": ["A-shared"]})
    out_path = tmp_path / "gds_no_label_audit"
    b = GDSBuilder("no_label_audit", str(out_path))
    b.add_line("accounts", accounts, key_col="account_id", source_id="test")
    b.add_line("tx", tx, key_col="tx_id", source_id="test", role="event")
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
        anomaly_percentile=80.0,
    )
    b.add_event_dimension("tx_pattern", column="sep_score", edge_max="auto")
    out = b.build()

    _sphere, nav = _open_navigator(out)
    with pytest.raises(GDSNavigationError) as excinfo:
        nav.π5_attract_anomaly(
            "tx_pattern", top_n=5, rank_by="signed_confidence",
        )
    msg = str(excinfo.value)
    assert "signed_confidence ranking requires" in msg
    assert "label_audit:" in msg
    assert "rank_by='delta_norm'" in msg


def test_signed_confidence_rejects_diverse_select():
    """``select="diverse"`` is incompatible with ``signed_confidence``."""
    # Use a fake navigator + arg validation entry to avoid building a
    # full sphere; the value-set check fires before any sphere I/O.
    nav = GDSNavigator.__new__(GDSNavigator)
    with pytest.raises(ValueError, match="incompatible with select='diverse'"):
        # Direct value-check via the static gate — the call mirrors the
        # navigator's argument-validation prologue. Use the public method
        # path by mocking the minimum needed.
        nav.π5_attract_anomaly(  # type: ignore[misc]
            "tx_pattern",
            rank_by="signed_confidence",
            select="diverse",
        )
