# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Navigator-level tests for ``GDSNavigator.audit_label_alignment``.

Covers:

- Full-field path on a synthetic 2-class sphere (AUROC ≥ 0.95).
- Fallback shape when the pattern carries no ``label_aware_calibration``.
- ``top_n`` cap on the returned ``top_dims`` list.
- Vendored ``_auroc_rank_sum`` rank-based AUROC matches the textbook
  definition on a tied / untied toy input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.cli.schema import LabelAuditConfig
from hypertopos.navigation.navigator import _auroc_rank_sum
from hypertopos.sphere import HyperSphere


def _build_two_class_sphere(
    tmp_path: Path,
    *,
    enable_label_aware: bool,
    register_block: bool,
    n_per_class: int = 120,
    sep_mean: float = 3.0,
    seed: int = 17,
    out_dir_name: str = "gds_align_two_class",
    n_extra_dims: int = 0,
) -> tuple[str, list[str], np.ndarray]:
    """Synthetic 2-class event-pattern sphere with optional label-aware wiring.

    Mirrors the fixture used by ``test_m1_1_builder_wiring`` — one
    ``sep_score`` event dim shifts between classes (carries the label
    signal); one ``noise_score`` event dim is drawn from the same
    distribution in both classes. ``n_extra_dims`` extra noise dims are
    added so the ``top_n`` cap can be exercised.
    """
    rng = np.random.RandomState(seed)
    n = 2 * n_per_class
    labels_pyarr = ["anom"] * n_per_class + ["norm"] * n_per_class
    sep_pos = rng.normal(sep_mean, 1.0, n_per_class).astype(np.float32)
    sep_neg = rng.normal(0.0, 1.0, n_per_class).astype(np.float32)
    sep_score = np.concatenate([sep_pos, sep_neg])
    noise_score = rng.normal(0.0, 1.0, n).astype(np.float32)

    pks = [f"T-{i:04d}" for i in range(n)]
    cols: dict[str, list | np.ndarray] = {
        "tx_id": pks,
        "account_id": ["A-shared"] * n,
        "sep_score": sep_score,
        "noise_score": noise_score,
        "label": labels_pyarr,
    }
    for k in range(n_extra_dims):
        cols[f"extra_{k}"] = rng.normal(0.0, 1.0, n).astype(np.float32)
    tx = pa.table(cols)
    accounts = pa.table({"account_id": ["A-shared"]})

    out_path = tmp_path / out_dir_name
    b = GDSBuilder("two_class_align", str(out_path))
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
    b.add_event_dimension(
        "tx_pattern", column="sep_score", edge_max="auto",
    )
    b.add_event_dimension(
        "tx_pattern", column="noise_score", edge_max="auto",
    )
    for k in range(n_extra_dims):
        b.add_event_dimension(
            "tx_pattern", column=f"extra_{k}", edge_max="auto",
        )

    if enable_label_aware:
        b._label_aware_calibration = True
    if register_block:
        b._label_audit_block = LabelAuditConfig(
            label_column="label",
            label_positive_value="anom",
            patterns=["tx_pattern"],
        )

    out = b.build()
    labels_arr = np.array(
        [1] * n_per_class + [0] * n_per_class, dtype=np.int32,
    )
    return out, pks, labels_arr


def test_auroc_rank_sum_matches_textbook_formula():
    """Vendored AUROC reproduces the textbook AUROC on small inputs.

    Untied case: positive class entirely above negatives → AUROC = 1.0.
    Perfectly anticorrelated case → 0.0. Ties at the median use the
    average-rank convention (Mann-Whitney U / 2).
    """
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    assert _auroc_rank_sum(scores, labels) == pytest.approx(1.0)

    labels_flip = np.array([1, 1, 0, 0])
    assert _auroc_rank_sum(scores, labels_flip) == pytest.approx(0.0)

    # Tied scores: two positives and two negatives at identical values
    # → expected AUROC = 0.5 (no separation).
    scores_tied = np.array([0.5, 0.5, 0.5, 0.5])
    labels_split = np.array([1, 1, 0, 0])
    assert _auroc_rank_sum(scores_tied, labels_split) == pytest.approx(0.5)


def test_full_field_path_auroc_high(tmp_path):
    """End-to-end: label-aware build → AUROC ≥ 0.95 on the separating sphere."""
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=True,
        n_per_class=120,
        sep_mean=3.0,
        out_dir_name="gds_align_full",
    )

    session = HyperSphere.open(out).session("test-agent")
    nav = session.navigator()
    result = nav.audit_label_alignment("tx_pattern", top_n=10)

    assert result["label_aware_available"] is True
    assert result["pattern_id"] == "tx_pattern"
    assert result["auroc"] is not None
    assert result["auroc"] >= 0.95, (
        f"separating sphere AUROC={result['auroc']:.3f} should exceed 0.95"
    )
    assert result["n_pos"] == 120
    assert result["n_neg"] == 120
    assert result["elapsed_ms"] >= 0.0
    # top_dims has entries for every dim in the calibration map (≤ top_n);
    # for this fixture there are 3 dims (accounts + sep_score + noise_score).
    assert 1 <= len(result["top_dims"]) <= 10
    # The separating dim must rank ahead of the noise dim by |direction|.
    rows = {row["dim_label"]: row for row in result["top_dims"]}
    assert rows["sep_score"]["abs_direction"] > rows["noise_score"]["abs_direction"]
    # Each row carries the full field set.
    for row in result["top_dims"]:
        assert set(row.keys()) == {
            "dim_label", "direction_component", "abs_direction",
            "cohens_d_pos_neg",
        }


def test_fallback_when_no_label_aware_calibration(tmp_path):
    """Pattern without ``label_aware_calibration`` returns fallback shape."""
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=False,
        register_block=False,
        out_dir_name="gds_align_no_lac",
    )

    session = HyperSphere.open(out).session("test-agent")
    nav = session.navigator()
    result = nav.audit_label_alignment("tx_pattern", top_n=10)

    assert result["label_aware_available"] is False
    assert result["auroc"] is None
    assert result["n_pos"] is None
    assert result["n_neg"] is None
    assert result["top_dims"] == []
    assert "reason" in result


def test_top_n_truncation(tmp_path):
    """``top_n`` caps the returned dim rows exactly."""
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=True,
        n_per_class=80,
        sep_mean=2.5,
        n_extra_dims=5,  # 2 base + 5 extra + 1 anchor relation = 8 dims total
        out_dir_name="gds_align_topn",
    )

    session = HyperSphere.open(out).session("test-agent")
    nav = session.navigator()
    result = nav.audit_label_alignment("tx_pattern", top_n=3)

    assert result["label_aware_available"] is True
    assert len(result["top_dims"]) == 3
    # Returned dims must be sorted by abs_direction desc.
    abs_dirs = [row["abs_direction"] for row in result["top_dims"]]
    assert abs_dirs == sorted(abs_dirs, reverse=True)


def test_top_n_below_one_raises(tmp_path):
    """``top_n < 1`` is a programmer error — surfaced as ``ValueError``."""
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=True,
        n_per_class=40,
        out_dir_name="gds_align_topn_err",
    )
    session = HyperSphere.open(out).session("test-agent")
    nav = session.navigator()
    with pytest.raises(ValueError, match="top_n must be >= 1"):
        nav.audit_label_alignment("tx_pattern", top_n=0)


def test_unknown_pattern_raises(tmp_path):
    """Unknown pattern_id surfaces as ``KeyError`` (MCP wrapper formats it)."""
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=True,
        n_per_class=40,
        out_dir_name="gds_align_unknown",
    )
    session = HyperSphere.open(out).session("test-agent")
    nav = session.navigator()
    with pytest.raises(KeyError, match="unknown pattern_id"):
        nav.audit_label_alignment("does_not_exist", top_n=5)
