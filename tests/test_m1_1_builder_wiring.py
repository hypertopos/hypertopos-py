# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Builder wiring of ``calibrate_label_aware`` (forward port of M1.1).

The builder previously initialised ``_label_aware_directions`` but never
called ``calibrate_label_aware``, so every pattern's ``delta_norm_signed``
column was full-null and the MCP ``audit_pattern_dims`` tool always
fell back to the no-label-aware shape. These tests cover the new
end-to-end wiring:

- Pattern listed under ``label_audit.patterns`` AND
  ``_label_aware_calibration = True`` → per-dim calibration persisted on
  ``Pattern.label_aware_calibration``, signed direction registered, and
  ``delta_norm_signed`` populated with non-null per-row values that
  correlate with class separation.
- Pattern NOT listed (or flag off) → ``Pattern.label_aware_calibration``
  remains ``None`` and ``delta_norm_signed`` stays all-null
  (backward-compat with format 3.0 spheres).
- Sphere written → reader hydrates the field with attribute-access
  semantics expected by ``audit_pattern_dims``.
"""
from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.cli.schema import LabelAuditConfig
from hypertopos.storage.reader import GDSReader


def _build_two_class_sphere(
    tmp_path: Path,
    *,
    enable_label_aware: bool,
    register_block: bool,
    n_per_class: int = 80,
    sep_mean: float = 2.5,
    seed: int = 17,
    out_dir_name: str = "gds_two_class_wired",
) -> tuple[str, list[str], np.ndarray]:
    """Construct a 2-class event-pattern sphere with optional label-aware wiring.

    One ``sep_score`` event dim shifts between classes (carries the
    label signal); one ``noise_score`` event dim is drawn the same in
    both classes. The label column ``label`` is "anom" for the positive
    class and "norm" for the negative class.
    """
    rng = np.random.RandomState(seed)
    n = 2 * n_per_class
    labels_pyarr = ["anom"] * n_per_class + ["norm"] * n_per_class
    sep_pos = rng.normal(sep_mean, 1.0, n_per_class).astype(np.float32)
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
    b = GDSBuilder("two_class_wired", str(out_path))
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


def test_label_aware_wired_populates_pattern_and_geometry(tmp_path):
    """End-to-end: builder hook wires LDA fit into pattern + Lance column.

    With label_audit registered AND the CLI flag set, the build pipeline
    fits ``calibrate_label_aware`` on the pattern's deltas, persists the
    per-dim moments + direction component, and projects each polygon's
    delta onto the registered direction so ``delta_norm_signed`` is
    populated (no null rows). The separating dim's direction component
    must carry the bulk of the unit-norm axis (>0.5 abs) while the noise
    dim's component must be small (<0.1 abs).
    """
    out, pks, labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=True,
        n_per_class=120,
        sep_mean=3.0,
        out_dir_name="gds_wired_on",
    )

    # 1. Pattern.label_aware_calibration populated and attribute-accessible.
    reader = GDSReader(out)
    sphere = reader.read_sphere()
    pat = sphere.patterns["tx_pattern"]
    assert pat.label_aware_calibration is not None
    assert set(pat.label_aware_calibration.keys()) == {
        "accounts", "sep_score", "noise_score",
    }
    sep_cal = pat.label_aware_calibration["sep_score"]
    noise_cal = pat.label_aware_calibration["noise_score"]
    # Attribute access (not dict access) is the contract the MCP audit
    # tool depends on.
    assert hasattr(sep_cal, "mu_pos")
    assert hasattr(sep_cal, "direction")
    assert sep_cal.mu_pos > sep_cal.mu_neg
    # Separating dim carries the bulk of the unit Fisher axis; noise
    # dim's component stays small.
    assert abs(sep_cal.direction) > 0.5, (
        f"separating dim |direction|={abs(sep_cal.direction):.3f} "
        f"should exceed 0.5"
    )
    assert abs(noise_cal.direction) < 0.1, (
        f"noise dim |direction|={abs(noise_cal.direction):.3f} "
        f"should be below 0.1"
    )

    # 2. delta_norm_signed populated (no nulls when label-aware fires).
    geo_path = Path(out) / "geometry" / "tx_pattern" / "data.lance"
    tbl = lance.dataset(str(geo_path)).to_table()
    assert "delta_norm_signed" in tbl.schema.names
    assert tbl["delta_norm_signed"].null_count == 0

    # 3. Per-class means of signed projection follow LDA orientation.
    pk_to_label = dict(zip(pks, labels, strict=True))
    geo_pks = tbl["primary_key"].to_pylist()
    row_labels = np.array(
        [pk_to_label[pk] for pk in geo_pks], dtype=np.int32,
    )
    signed = tbl["delta_norm_signed"].to_numpy(zero_copy_only=False)
    assert signed[row_labels == 1].mean() > signed[row_labels == 0].mean()


def test_label_aware_off_when_block_missing(tmp_path):
    """No ``label_audit:`` block → pattern carries no calibration field.

    Flag flipped on without a block is a no-op (matches the existing
    ``_run_label_aware_calibration`` short-circuit). ``delta_norm_signed``
    must stay null on every row, preserving format 3.0 backward compat.
    """
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=False,
        out_dir_name="gds_wired_off_block",
    )
    reader = GDSReader(out)
    sphere = reader.read_sphere()
    pat = sphere.patterns["tx_pattern"]
    assert pat.label_aware_calibration is None

    geo_path = Path(out) / "geometry" / "tx_pattern" / "data.lance"
    tbl = lance.dataset(str(geo_path)).to_table()
    assert tbl["delta_norm_signed"].null_count == tbl.num_rows


def test_label_aware_off_when_flag_missing(tmp_path):
    """``label_audit:`` block alone (no flag) → calibration does not run."""
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=False,
        register_block=True,
        out_dir_name="gds_wired_off_flag",
    )
    reader = GDSReader(out)
    sphere = reader.read_sphere()
    pat = sphere.patterns["tx_pattern"]
    assert pat.label_aware_calibration is None

    geo_path = Path(out) / "geometry" / "tx_pattern" / "data.lance"
    tbl = lance.dataset(str(geo_path)).to_table()
    assert tbl["delta_norm_signed"].null_count == tbl.num_rows


def test_round_trip_preserves_label_aware_calibration(tmp_path):
    """Build → reader round trip preserves moments + attribute access.

    Builder writes per-dim values to ``sphere.json`` as a JSON-safe
    dict-of-dicts. Reader hydrates back to attribute-access objects so
    ``audit_pattern_dims``' ``dim_cal.mu_pos`` path keeps working.
    Values must round-trip within float32 noise.
    """
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=True,
        n_per_class=100,
        out_dir_name="gds_wired_round_trip",
    )

    # First reader (just-built path)
    reader1 = GDSReader(out)
    pat1 = reader1.read_sphere().patterns["tx_pattern"]
    cal1 = pat1.label_aware_calibration
    assert cal1 is not None

    # Second reader (cold reload — same on-disk sphere.json)
    reader2 = GDSReader(out)
    pat2 = reader2.read_sphere().patterns["tx_pattern"]
    cal2 = pat2.label_aware_calibration
    assert cal2 is not None
    assert set(cal1.keys()) == set(cal2.keys())
    for label in cal1:
        d1, d2 = cal1[label], cal2[label]
        for attr in ("mu_pos", "sigma_pos", "mu_neg", "sigma_neg", "direction"):
            v1, v2 = getattr(d1, attr), getattr(d2, attr)
            assert v1 == v2, (
                f"{label}.{attr} round-trip mismatch: {v1} != {v2}"
            )


def test_audit_pattern_dims_returns_full_field_path_end_to_end(tmp_path):
    """End-to-end chain: build → reader → MCP audit_pattern_dims.

    With M1.1 wired, calling ``audit_pattern_dims`` on a sphere built
    with the label_audit block must return the full per-dim path
    (``label_aware_available: True``, non-null ``cohens_d_pos_neg`` and
    ``direction_component`` per dim) — not the fallback shape. This is
    the user-visible deliverable the brief targets.
    """
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import hypertopos_mcp.tools.observability  # noqa: F401 — register tool
    from hypertopos_mcp.server import _state
    from hypertopos_mcp.tools.observability import audit_pattern_dims

    out, _pks, _labels = _build_two_class_sphere(
        tmp_path,
        enable_label_aware=True,
        register_block=True,
        n_per_class=120,
        sep_mean=3.0,
        out_dir_name="gds_audit_e2e",
    )
    sphere = GDSReader(out).read_sphere()

    saved_nav = _state.get("navigator")
    saved_sphere = _state.get("sphere")
    try:
        # Install the built sphere into MCP state. The tool reads
        # ``_state["sphere"]._sphere``, so we wrap with a stand-in that
        # exposes the Pattern dict on `.patterns` via the inner attribute.
        sphere_wrapper = MagicMock()
        sphere_wrapper._sphere = SimpleNamespace(patterns=sphere.patterns)
        _state["navigator"] = MagicMock()
        _state["sphere"] = sphere_wrapper

        body = audit_pattern_dims(pattern_id="tx_pattern", top_k=10)
        parsed = json.loads(body)

        assert parsed["label_aware_available"] is True, (
            f"M1.1 wired but tool still returned fallback shape: {parsed}"
        )
        assert "reason" not in parsed
        # Each returned dim row carries the full field set; cohens_d and
        # direction_component must be present and non-null per dim.
        for row in parsed["dims"]:
            assert row["cohens_d_pos_neg"] is not None, row
            assert row["direction_component"] is not None, row
            assert {
                "mu_pos", "sigma_pos", "mu_neg", "sigma_neg",
            }.issubset(row.keys())
        # The dim with the separating signal must out-rank the noise dim
        # by absolute Cohen's d.
        rows_by_label = {row["dim_label"]: row for row in parsed["dims"]}
        assert abs(rows_by_label["sep_score"]["cohens_d_pos_neg"]) > abs(
            rows_by_label["noise_score"]["cohens_d_pos_neg"],
        )
    finally:
        _state["navigator"] = saved_nav
        _state["sphere"] = saved_sphere


def test_label_aware_skipped_when_label_column_missing(tmp_path, caplog):
    """Missing label column in entity_line → hook short-circuits, no crash.

    Guards against accidental builds where the YAML block references a
    column that doesn't exist on the entity line (typo in column name).
    Builder must keep going and produce a sphere with no label-aware
    calibration rather than aborting the whole build.
    """
    # Build a small synthetic sphere where the label column is "wrong_col"
    # but the entity_line has no such column.
    n = 40
    pks = [f"T-{i:04d}" for i in range(n)]
    tx = pa.table({
        "tx_id": pks,
        "account_id": ["A-shared"] * n,
        "sep_score": np.arange(n, dtype=np.float32),
    })
    accounts = pa.table({"account_id": ["A-shared"]})

    out_path = tmp_path / "gds_missing_label_col"
    b = GDSBuilder("missing_label_col", str(out_path))
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
    )
    b.add_event_dimension("tx_pattern", column="sep_score", edge_max="auto")
    b._label_aware_calibration = True
    b._label_audit_block = LabelAuditConfig(
        label_column="does_not_exist",
        label_positive_value="anom",
        patterns=["tx_pattern"],
    )
    out = b.build()

    reader = GDSReader(out)
    pat = reader.read_sphere().patterns["tx_pattern"]
    # Hook short-circuited — no calibration persisted.
    assert pat.label_aware_calibration is None
