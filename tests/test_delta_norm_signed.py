# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``delta_norm_signed`` Lance column — builder geometry pass coverage.

Acceptance signal (plan §M1.3):
- Column written iff pattern has a registered label-aware direction
  vector; reader doesn't break for patterns without it (nullable).
- 2-class synthetic: signed delta separation between classes >
  unsigned delta separation by ≥ 20%.
"""
from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.engine.calibration_label_aware import calibrate_label_aware


def _make_two_class_sphere_inputs(
    *,
    n_per_class: int = 80,
    sep_mean: float = 2.5,
    seed: int = 17,
) -> tuple[pa.Table, pa.Table, list[str], np.ndarray]:
    """Construct a 2-class event-pattern table with one separating numeric
    dim and one noise numeric dim.

    Event-pattern ``tx`` has two ``event_dimensions``:
    - ``sep_score``: positive class drawn N(``sep_mean``, 1.0),
      negative class drawn N(0.0, 1.0). Carries the label signal.
    - ``noise_score``: both classes drawn N(0.0, 1.0). No label signal.

    Each transaction also links to a single ``account`` anchor — the
    relation is required but identical across both classes, so it
    contributes no separating dim. The two event_dimensions produce
    per-entity numeric variation that survives z-scoring, so LDA fit
    on per-event deltas does NOT collapse to the "class means
    identical" degenerate case that pure categorical FK patterns hit.

    Returns the accounts and tx Arrow tables, the ordered list of
    primary keys, and the label vector aligned with that primary-key
    order.
    """
    rng = np.random.RandomState(seed)
    n = 2 * n_per_class
    labels = np.array([1] * n_per_class + [0] * n_per_class, dtype=np.int32)
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
    })
    accounts = pa.table({"account_id": ["A-shared"]})
    return accounts, tx, pks, labels


def _build_two_class_sphere(
    tmp_path: Path,
    *,
    n_per_class: int = 80,
    seed: int = 17,
    direction: np.ndarray | None = None,
    out_dir_name: str = "gds_two_class",
) -> tuple[str, list[str], np.ndarray]:
    """Build a 2-class event-pattern sphere; optionally inject a direction.

    Returns the sphere build path, the ordered list of primary keys, and
    the label vector aligned with that primary-key order.
    """
    accounts, tx, pks, labels = _make_two_class_sphere_inputs(
        n_per_class=n_per_class, seed=seed,
    )
    b = GDSBuilder("two_class_sphere", str(tmp_path / out_dir_name))
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
    if direction is not None:
        b._label_aware_directions["tx_pattern"] = direction
    out = b.build()
    return out, pks, labels


def _fit_direction_for_pattern(
    out_path: str, pattern_id: str, pks: list[str], labels: np.ndarray,
) -> np.ndarray:
    """Read the freshly built geometry, fit a Fisher LDA direction.

    Returns the unit-norm signed direction vector. Used to populate the
    builder's in-build direction registry on a second build so the
    geometry pass can project deltas onto it.
    """
    geo_path = Path(out_path) / "geometry" / pattern_id / "data.lance"
    geo = lance.dataset(str(geo_path)).to_table()
    pk_to_idx = {pk: i for i, pk in enumerate(pks)}
    delta_col = geo["delta"].combine_chunks()
    geo_pks = geo["primary_key"].to_pylist()
    n_dims = len(delta_col[0])
    deltas = np.zeros((len(geo_pks), n_dims), dtype=np.float32)
    for row_idx, pk in enumerate(geo_pks):
        deltas[pk_to_idx[pk]] = np.asarray(
            delta_col[row_idx].as_py(), dtype=np.float32,
        )
    result = calibrate_label_aware(
        deltas=deltas, labels=labels,
    )
    return np.asarray(result.signed_direction_vector, dtype=np.float32)


def test_delta_norm_signed_emits_null_without_registered_direction(tmp_path):
    """Patterns without a registered direction get a full-null column.

    Acceptance: column is in the Lance schema (so readers don't need a
    presence check) AND every row is null.
    """
    out, _pks, _labels = _build_two_class_sphere(tmp_path)
    geo_path = Path(out) / "geometry" / "tx_pattern" / "data.lance"
    tbl = lance.dataset(str(geo_path)).to_table()
    assert "delta_norm_signed" in tbl.schema.names
    assert tbl["delta_norm_signed"].null_count == tbl.num_rows


def test_delta_norm_signed_separates_classes_better_than_unsigned(tmp_path):
    """Plan acceptance: signed class separation ≥ 1.2 × unsigned.

    Two-class synthetic with a separating prop_column (``sep_score``
    shifted between classes) and a noise prop_column. Build once
    without a registered direction to obtain stable deltas, fit the
    Fisher LDA direction on those deltas, then build a second sphere
    with the same data + registered direction so the geometry pass
    emits ``delta_norm_signed``. Compare per-class means of
    ``delta_norm_signed`` (signed projection) against per-class means
    of ``delta_norm`` (unsigned magnitude) — the signed projection
    must discriminate substantially more than the unsigned magnitude.

    The unsigned magnitude collapses sign, so positive- and negative-
    class polygons sitting on opposite sides of mu look similar; the
    signed projection keeps them on opposite sides of zero. The 1.2×
    margin is the plan threshold.
    """
    out, pks, labels = _build_two_class_sphere(tmp_path)
    direction = _fit_direction_for_pattern(out, "tx_pattern", pks, labels)

    # Rebuild with the direction registered so the geometry pass
    # projects each polygon's delta vector onto the unit-norm axis.
    rebuild_out, _pks2, _labels2 = _build_two_class_sphere(
        tmp_path, direction=direction, out_dir_name="gds_two_class_signed",
    )
    geo_path = Path(rebuild_out) / "geometry" / "tx_pattern" / "data.lance"
    tbl = lance.dataset(str(geo_path)).to_table()

    # The column must be present, fully populated (no nulls) when a
    # direction is registered, and a float32 scalar per polygon.
    assert "delta_norm_signed" in tbl.schema.names
    signed_field = tbl.schema.field("delta_norm_signed")
    assert signed_field.type == pa.float32()
    assert tbl["delta_norm_signed"].null_count == 0

    # Align label vector against geometry row order.
    pk_to_label = dict(zip(pks, labels, strict=True))
    geo_pks = tbl["primary_key"].to_pylist()
    row_labels = np.array(
        [pk_to_label[pk] for pk in geo_pks], dtype=np.int32,
    )

    signed = tbl["delta_norm_signed"].to_numpy(zero_copy_only=False)
    unsigned = tbl["delta_norm"].to_numpy(zero_copy_only=False)

    # Per-class means — separation is the absolute mean diff.
    pos_mask = row_labels == 1
    neg_mask = row_labels == 0
    signed_sep = abs(signed[pos_mask].mean() - signed[neg_mask].mean())
    unsigned_sep = abs(unsigned[pos_mask].mean() - unsigned[neg_mask].mean())

    # Diagnostic: both terms must be meaningfully positive — guards
    # against the trivial pass where unsigned_sep ≈ 0 makes the ratio
    # explode without proving anything about signed_sep itself.
    assert signed_sep > 0.5, (
        f"signed separation too small to be meaningful: {signed_sep:.4f}"
    )
    assert unsigned_sep > 0.05, (
        f"unsigned separation must be non-trivially positive to make "
        f"the 1.2× comparison meaningful, got {unsigned_sep:.4f}"
    )

    # Plan threshold: signed separation must be at least 20% larger
    # than unsigned.
    assert signed_sep >= 1.2 * unsigned_sep, (
        f"signed/unsigned separation ratio "
        f"{signed_sep / max(unsigned_sep, 1e-9):.3f} below 1.2 plan "
        f"threshold (signed={signed_sep:.4f}, "
        f"unsigned={unsigned_sep:.4f})"
    )


def test_delta_norm_signed_dim_mismatch_falls_back_to_null(tmp_path):
    """A registered direction with wrong dim count must NOT corrupt rows.

    Defensive contract for the in-build registry — the geometry pass
    silently emits a full-null column when ``direction.shape[0]`` does
    not match ``chunk_deltas.shape[1]``. This mirrors the "no
    calibration registered" path and prevents a malformed registry
    entry from blowing up an otherwise-valid build.
    """
    # Deliberately wrong-length direction (99 dims vs the pattern's
    # actual relations + prop_columns count, which is well under 99).
    bogus_direction = np.zeros(99, dtype=np.float32)
    bogus_direction[0] = 1.0
    out, _pks, _labels = _build_two_class_sphere(
        tmp_path, direction=bogus_direction,
        out_dir_name="gds_two_class_mismatch",
    )
    geo_path = Path(out) / "geometry" / "tx_pattern" / "data.lance"
    tbl = lance.dataset(str(geo_path)).to_table()
    assert "delta_norm_signed" in tbl.schema.names
    assert tbl["delta_norm_signed"].null_count == tbl.num_rows


def test_delta_norm_signed_skipped_for_unrelated_pattern(tmp_path):
    """A registered direction for one pattern leaves other patterns null.

    Builds two event patterns over disjoint event lines; only ``tx_pattern``
    has a registered direction. ``other_pattern`` emits the column as
    full-nulls regardless of what's in the registry under another key.
    """
    accounts, tx, _pks, _labels = _make_two_class_sphere_inputs(
        n_per_class=20, seed=23,
    )
    other_event_pks = [f"O-{i:04d}" for i in range(10)]
    other_events = pa.table({
        "other_id": other_event_pks,
        "account_id": ["A-shared"] * 10,
        "value": np.arange(10, dtype=np.float32),
    })

    b = GDSBuilder("two_class_partial", str(tmp_path / "gds_partial"))
    b.add_line(
        "accounts", accounts, key_col="account_id", source_id="test",
    )
    b.add_line(
        "tx", tx, key_col="tx_id", source_id="test", role="event",
    )
    b.add_line(
        "other_events", other_events, key_col="other_id",
        source_id="test", role="event",
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

    b.add_pattern(
        "other_pattern",
        pattern_type="event",
        entity_line="other_events",
        relations=[
            RelationSpec(
                line_id="accounts", fk_col="account_id",
                direction="in", required=True,
            ),
        ],
        anomaly_percentile=95.0,
    )
    b.add_event_dimension("other_pattern", column="value", edge_max="auto")

    # Register a direction only for tx_pattern. other_pattern has no
    # registered direction → must stay null. The direction shape here
    # is intentionally wrong for tx_pattern (2 dims instead of the
    # 1 relation + 2 event dims = 3 dims it actually carries), so
    # tx_pattern's column falls back to null too — the test only
    # asserts the per-pattern boundary, not what tx_pattern emits.
    b._label_aware_directions["tx_pattern"] = np.array(
        [1.0, 0.0], dtype=np.float32,
    )
    b.build()

    tx_tbl = lance.dataset(
        str(tmp_path / "gds_partial" / "geometry" / "tx_pattern" / "data.lance"),
    ).to_table()
    other_tbl = lance.dataset(
        str(tmp_path / "gds_partial" / "geometry" / "other_pattern" / "data.lance"),
    ).to_table()

    assert "delta_norm_signed" in tx_tbl.schema.names
    assert "delta_norm_signed" in other_tbl.schema.names

    # ``other_pattern`` has no registered direction, so its column must
    # be full-null regardless of what happened on ``tx_pattern``.
    assert other_tbl["delta_norm_signed"].null_count == other_tbl.num_rows


def test_delta_norm_signed_preserves_sign(tmp_path):
    """Positive class lands on the positive side of zero, negative on negative.

    LDA direction is sign-oriented by ``fit_lda_direction`` so projecting
    onto it places mean(delta_pos) on the positive side. Each individual
    polygon may sit on either side, but the per-class mean must follow
    the orientation.
    """
    out, pks, labels = _build_two_class_sphere(tmp_path, n_per_class=60)
    direction = _fit_direction_for_pattern(out, "tx_pattern", pks, labels)

    rebuild_out, _pks2, _labels2 = _build_two_class_sphere(
        tmp_path, n_per_class=60, direction=direction,
        out_dir_name="gds_two_class_sign",
    )
    geo_path = Path(rebuild_out) / "geometry" / "tx_pattern" / "data.lance"
    tbl = lance.dataset(str(geo_path)).to_table()
    pk_to_label = dict(zip(pks, labels, strict=True))
    geo_pks = tbl["primary_key"].to_pylist()
    row_labels = np.array(
        [pk_to_label[pk] for pk in geo_pks], dtype=np.int32,
    )
    signed = tbl["delta_norm_signed"].to_numpy(zero_copy_only=False)
    pos_mean = signed[row_labels == 1].mean()
    neg_mean = signed[row_labels == 0].mean()
    assert pos_mean > neg_mean, (
        f"sign-oriented direction must place positive class above "
        f"negative class on average — pos_mean={pos_mean:.4f}, "
        f"neg_mean={neg_mean:.4f}"
    )
