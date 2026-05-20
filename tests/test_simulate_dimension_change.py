# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for ``GDSNavigator.simulate_dimension_change``.

Per-dimension counterfactual: override one or more shape-space dims for a
single entity, then recompute ``delta`` / ``delta_norm`` / anomaly flag and
report the per-dim audit. Math is hand-verified against the engine's
``compute_delta`` (cholesky vs diagonal path, optional ``dimension_weights``).
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.model.sphere import Pattern, RelationDef
from hypertopos.navigation.navigator import GDSNavigationError, GDSNavigator


# ── Synthetic pattern helpers ──────────────────────────────────────────────


def _make_relations(line_ids: list[str]) -> list[RelationDef]:
    return [
        RelationDef(line_id=lid, direction="out", required=False)
        for lid in line_ids
    ]


def _make_pattern(
    *,
    line_ids: list[str],
    mu: list[float],
    sigma_diag: list[float],
    theta: list[float],
    dimension_weights: list[float] | None = None,
    cholesky_inv: np.ndarray | None = None,
) -> Pattern:
    return Pattern(
        pattern_id="test_pattern",
        entity_type="account",
        pattern_type="anchor",
        relations=_make_relations(line_ids),
        mu=np.asarray(mu, dtype=np.float64),
        sigma_diag=np.asarray(sigma_diag, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        population_size=1000,
        computed_at=datetime(2026, 5, 17, tzinfo=UTC),
        version=1,
        status="production",
        dimension_weights=(
            np.asarray(dimension_weights, dtype=np.float64)
            if dimension_weights is not None
            else None
        ),
        cholesky_inv=cholesky_inv,
    )


def _make_nav(pattern: Pattern, *, delta: list[float], is_anomaly: bool) -> GDSNavigator:
    """Build a GDSNavigator with a mocked storage layer that returns a
    single geometry row matching ``delta`` (and the corresponding
    ``delta_norm`` computed inline)."""
    delta_np = np.asarray(delta, dtype=np.float64)
    delta_norm = float(np.linalg.norm(delta_np))

    nav = GDSNavigator.__new__(GDSNavigator)
    nav._storage = MagicMock()
    nav._storage.read_sphere.return_value = MagicMock(
        patterns={"test_pattern": pattern},
    )
    nav._storage.read_geometry.return_value = pa.table({
        "primary_key": ["E1"],
        "delta": pa.array([delta_np.tolist()], type=pa.list_(pa.float64())),
        "delta_norm": [delta_norm],
        "is_anomaly": [is_anomaly],
    })
    nav._resolve_version = MagicMock(return_value=1)
    return nav


# ── Tests ──────────────────────────────────────────────────────────────────


def test_set_dim_to_mu_shrinks_norm_proportional():
    """mu=[1,1,1], sigma=[1,1,1], shape=[3,2,1] ⇒ delta=[2,1,0], norm²=5.
    Override dim 0 to mu (1.0) ⇒ shape=[1,2,1], delta=[0,1,0], norm=1.
    Expected pct_change ≈ (1 - sqrt(5)) / sqrt(5) * 100 ≈ -55.279%."""
    pattern = _make_pattern(
        line_ids=["A", "B", "C"],
        mu=[1.0, 1.0, 1.0],
        sigma_diag=[1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0],
    )
    # delta = (shape - mu) / sigma = ([3,2,1] - [1,1,1]) / [1,1,1] = [2,1,0]
    nav = _make_nav(pattern, delta=[2.0, 1.0, 0.0], is_anomaly=False)

    result = nav.simulate_dimension_change(
        "E1",
        pattern_id="test_pattern",
        line_id="line_A",
        set_dimension={"A": 1.0},
    )

    assert result["delta_norm_before"] == pytest.approx(math.sqrt(5.0), abs=1e-9)
    assert result["delta_norm_after"] == pytest.approx(1.0, abs=1e-9)
    expected_pct = (1.0 - math.sqrt(5.0)) / math.sqrt(5.0) * 100.0
    assert result["delta_norm_pct_change"] == pytest.approx(expected_pct, abs=1e-6)
    # Override audit row for dim "A".
    assert len(result["dimensions_overridden"]) == 1
    row = result["dimensions_overridden"][0]
    assert row["dim_label"] == "A"
    assert row["dim_index"] == 0
    assert row["old_value"] == pytest.approx(3.0)
    assert row["new_value"] == pytest.approx(1.0)
    assert row["old_delta"] == pytest.approx(2.0)
    assert row["new_delta"] == pytest.approx(0.0)


def test_spike_low_dim_creates_anomaly():
    """Engineer non-anomalous entity; spike a low dim to mu+5σ; assert
    anomaly flag flips False → True."""
    pattern = _make_pattern(
        line_ids=["A", "B"],
        mu=[2.0, 2.0],
        sigma_diag=[1.0, 1.0],
        theta=[3.0, 3.0],  # theta_norm = sqrt(18) ≈ 4.243
    )
    # shape=[2,2] ⇒ delta=[0,0], norm=0 — clearly below theta.
    nav = _make_nav(pattern, delta=[0.0, 0.0], is_anomaly=False)

    # Override "A" to mu + 5σ = 7.0 ⇒ new delta = [5, 0], norm = 5 > theta_norm.
    result = nav.simulate_dimension_change(
        "E1",
        pattern_id="test_pattern",
        line_id="line_A",
        set_dimension={"A": 7.0},
    )

    assert result["is_anomaly_before"] is False
    assert result["is_anomaly_after"] is True
    assert result["is_anomaly_change"] is True
    assert result["delta_norm_after"] == pytest.approx(5.0, abs=1e-9)


def test_zero_anomalous_dim_clears_anomaly():
    """Engineer anomalous entity dominated by one dim; zero that dim's
    shape value to mu; assert anomaly flag flips True → False."""
    pattern = _make_pattern(
        line_ids=["A", "B"],
        mu=[0.0, 0.0],
        sigma_diag=[1.0, 1.0],
        theta=[3.0, 3.0],  # theta_norm = sqrt(18) ≈ 4.243
    )
    # delta=[5, 0.1], norm ≈ 5.001 > theta_norm → anomaly
    nav = _make_nav(pattern, delta=[5.0, 0.1], is_anomaly=True)

    # Override "A" to mu (0.0) ⇒ new delta = [0, 0.1], norm = 0.1 < theta
    result = nav.simulate_dimension_change(
        "E1",
        pattern_id="test_pattern",
        line_id="line_A",
        set_dimension={"A": 0.0},
    )

    assert result["is_anomaly_before"] is True
    assert result["is_anomaly_after"] is False
    assert result["is_anomaly_change"] is True


def test_unknown_pattern_id_raises():
    pattern = _make_pattern(
        line_ids=["A"], mu=[0.0], sigma_diag=[1.0], theta=[3.0],
    )
    nav = _make_nav(pattern, delta=[0.0], is_anomaly=False)

    with pytest.raises(GDSNavigationError, match="pattern not found"):
        nav.simulate_dimension_change(
            "E1",
            pattern_id="does_not_exist",
            line_id="line_A",
            set_dimension={"A": 1.0},
        )


def test_unknown_primary_key_raises():
    pattern = _make_pattern(
        line_ids=["A"], mu=[0.0], sigma_diag=[1.0], theta=[3.0],
    )
    nav = _make_nav(pattern, delta=[0.0], is_anomaly=False)
    # Force empty geometry table.
    nav._storage.read_geometry.return_value = pa.table({
        "primary_key": pa.array([], type=pa.string()),
        "delta": pa.array([], type=pa.list_(pa.float64())),
        "delta_norm": pa.array([], type=pa.float64()),
        "is_anomaly": pa.array([], type=pa.bool_()),
    })

    with pytest.raises(GDSNavigationError, match="not found in"):
        nav.simulate_dimension_change(
            "missing_entity",
            pattern_id="test_pattern",
            line_id="line_A",
            set_dimension={"A": 1.0},
        )


def test_unknown_dim_label_raises_with_available_list():
    pattern = _make_pattern(
        line_ids=["A", "B", "C"],
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=[0.0, 0.0, 0.0], is_anomaly=False)

    with pytest.raises(GDSNavigationError) as exc_info:
        nav.simulate_dimension_change(
            "E1",
            pattern_id="test_pattern",
            line_id="line_A",
            set_dimension={"ZZZ_does_not_exist": 1.0},
        )
    msg = str(exc_info.value)
    assert "ZZZ_does_not_exist" in msg
    # At least one valid label appears in the available list.
    assert "A" in msg
    assert "available" in msg.lower()


def test_aggregation_label_resolves_via_dim_labels_fallback():
    """Labels exposed only through `pattern.dim_labels` (e.g.
    `edge_dim_aggregations` aggregate names) must resolve via the
    dim_labels.index() fallback when `pattern.dim_index` rejects them.
    The public contract is "any label in `pattern.dim_labels` is overridable".

    Real reproduction: AML HI-small `account_pattern` exposes
    `pair_edge_count_*` and `find_motif_structuring_*` labels in
    `dim_labels` but `dim_index` only searches relations / event_dims /
    prop_columns — without the fallback these labels raise unknown."""
    extra_label = "edge_count_max_aggregation"
    pattern_real = _make_pattern(
        line_ids=["A", "B", extra_label],  # carry 3 slots in relations…
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0],
        theta=[2.0, 2.0, 2.0],
    )
    # …then wrap to make dim_index reject extra_label (simulating that
    # it is NOT a relation, only an aggregation entry in dim_labels).
    def fake_dim_index(name):
        if name in ("A", "B"):
            return ["A", "B"].index(name)
        raise ValueError(
            f"Dimension {name!r} not found in pattern relations. "
            f"Available: ['A', 'B']"
        )

    wrapper = MagicMock(wraps=pattern_real)
    wrapper.dim_labels = ["A", "B", extra_label]
    wrapper.dim_index.side_effect = fake_dim_index
    wrapper.mu = pattern_real.mu
    wrapper.sigma_diag = pattern_real.sigma_diag
    wrapper.theta = pattern_real.theta
    wrapper.cholesky_inv = pattern_real.cholesky_inv
    wrapper.dimension_weights = pattern_real.dimension_weights
    wrapper.pattern_type = pattern_real.pattern_type

    delta_np = np.asarray([0.5, 0.3, 0.7], dtype=np.float64)
    nav = GDSNavigator.__new__(GDSNavigator)
    nav._storage = MagicMock()
    nav._storage.read_sphere.return_value = MagicMock(
        patterns={"test_pattern": wrapper},
    )
    nav._storage.read_geometry.return_value = pa.table({
        "primary_key": ["E1"],
        "delta": pa.array([delta_np.tolist()], type=pa.list_(pa.float64())),
        "delta_norm": [float(np.linalg.norm(delta_np))],
        "is_anomaly": [False],
    })
    nav._resolve_version = MagicMock(return_value=1)

    # Override the aggregation label — must NOT raise; resolves to idx=2
    # via dim_labels.index() fallback after dim_index ValueError.
    result = nav.simulate_dimension_change(
        "E1",
        pattern_id="test_pattern",
        line_id="line_A",
        set_dimension={extra_label: 0.0},
    )
    assert result["dimensions_overridden"][0]["dim_label"] == extra_label
    assert result["dimensions_overridden"][0]["dim_index"] == 2


def test_nan_value_raises():
    pattern = _make_pattern(
        line_ids=["A", "B"],
        mu=[0.0, 0.0],
        sigma_diag=[1.0, 1.0],
        theta=[3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=[0.0, 0.0], is_anomaly=False)

    with pytest.raises(GDSNavigationError) as exc_info:
        nav.simulate_dimension_change(
            "E1",
            pattern_id="test_pattern",
            line_id="line_A",
            set_dimension={"A": float("nan")},
        )
    assert "A" in str(exc_info.value)
    assert "non-finite" in str(exc_info.value).lower()


def test_inf_value_raises():
    pattern = _make_pattern(
        line_ids=["A", "B"],
        mu=[0.0, 0.0],
        sigma_diag=[1.0, 1.0],
        theta=[3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=[0.0, 0.0], is_anomaly=False)

    with pytest.raises(GDSNavigationError) as exc_info:
        nav.simulate_dimension_change(
            "E1",
            pattern_id="test_pattern",
            line_id="line_A",
            set_dimension={"B": float("inf")},
        )
    assert "B" in str(exc_info.value)
    assert "non-finite" in str(exc_info.value).lower()


def test_empty_set_dimension_raises():
    pattern = _make_pattern(
        line_ids=["A"], mu=[0.0], sigma_diag=[1.0], theta=[3.0],
    )
    nav = _make_nav(pattern, delta=[0.0], is_anomaly=False)

    with pytest.raises(GDSNavigationError, match="empty"):
        nav.simulate_dimension_change(
            "E1",
            pattern_id="test_pattern",
            line_id="line_A",
            set_dimension={},
        )


def test_multi_dim_override_applies_all():
    """Override two dims jointly; verify both audit rows and hand-computed
    math. mu=[0,0,0], sigma=[1,1,1], shape_before=[2,1,3] ⇒ delta=[2,1,3].
    Override A→0, C→0 ⇒ shape_after=[0,1,0] ⇒ delta_after=[0,1,0].
    Order of dimensions_overridden must match insertion order of set_dimension."""
    pattern = _make_pattern(
        line_ids=["A", "B", "C"],
        mu=[0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=[2.0, 1.0, 3.0], is_anomaly=False)

    result = nav.simulate_dimension_change(
        "E1",
        pattern_id="test_pattern",
        line_id="line_A",
        # Insertion order: A, then C.
        set_dimension={"A": 0.0, "C": 0.0},
    )

    assert result["delta_norm_after"] == pytest.approx(1.0, abs=1e-9)
    assert len(result["dimensions_overridden"]) == 2
    # Insertion order preserved (Python dict literal order).
    labels = [row["dim_label"] for row in result["dimensions_overridden"]]
    assert labels == ["A", "C"]
    # Dim A audit row.
    row_a = result["dimensions_overridden"][0]
    assert row_a["dim_index"] == 0
    assert row_a["old_value"] == pytest.approx(2.0)
    assert row_a["new_value"] == pytest.approx(0.0)
    assert row_a["old_delta"] == pytest.approx(2.0)
    assert row_a["new_delta"] == pytest.approx(0.0)
    # Dim C audit row.
    row_c = result["dimensions_overridden"][1]
    assert row_c["dim_index"] == 2
    assert row_c["old_value"] == pytest.approx(3.0)
    assert row_c["new_value"] == pytest.approx(0.0)
    assert row_c["old_delta"] == pytest.approx(3.0)
    assert row_c["new_delta"] == pytest.approx(0.0)


def test_top_witness_dims_after_sorted_by_abs_delta_desc():
    """Engineer a 4-dim pattern with varying contributions and assert the
    ordering of ``top_witness_dims_after`` (by |delta| desc) and that
    contribution_pct sums to ≤100."""
    pattern = _make_pattern(
        line_ids=["A", "B", "C", "D"],
        mu=[0.0, 0.0, 0.0, 0.0],
        sigma_diag=[1.0, 1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0, 3.0],
    )
    # Before override: delta = shape = [1, 4, 2, 3] (mu=0, sigma=1).
    nav = _make_nav(pattern, delta=[1.0, 4.0, 2.0, 3.0], is_anomaly=False)

    # No-op override (set "A" to its current shape value) so delta_after
    # equals delta_before. Expected ranking by |delta|: B(4) > D(3) > C(2) > A(1).
    result = nav.simulate_dimension_change(
        "E1",
        pattern_id="test_pattern",
        line_id="line_A",
        set_dimension={"A": 1.0},
        top_n=4,
    )

    ordered_labels = [row["dim_label"] for row in result["top_witness_dims_after"]]
    assert ordered_labels == ["B", "D", "C", "A"]
    abs_deltas = [abs(row["delta"]) for row in result["top_witness_dims_after"]]
    assert abs_deltas == sorted(abs_deltas, reverse=True)
    total_pct = sum(row["contribution_pct"] for row in result["top_witness_dims_after"])
    assert total_pct == pytest.approx(100.0, abs=0.01)


def test_cholesky_pattern_recomputes_correctly():
    """Engineer a 2-dim pattern with a non-diagonal cholesky_inv so cross-
    coupling between dims is visible. Overriding a single shape dim must
    move multiple delta dims (because L_inv is dense)."""
    # Choose L_inv with explicit cross-term so dim 0 of shape touches BOTH
    # delta dims.
    cholesky_inv = np.array(
        [[2.0, 0.0],
         [1.0, 0.5]],
        dtype=np.float64,
    )
    pattern = _make_pattern(
        line_ids=["A", "B"],
        mu=[1.0, 2.0],
        sigma_diag=[1.0, 1.0],  # ignored when cholesky_inv present
        theta=[3.0, 3.0],
        cholesky_inv=cholesky_inv,
    )

    # Choose shape_before = [3.0, 4.0]. Hand-compute delta_before:
    # delta_before = L_inv @ (shape - mu)
    #              = L_inv @ [2, 2]
    #              = [4, 3]
    delta_before = (cholesky_inv @ np.array([2.0, 2.0], dtype=np.float64)).tolist()
    assert delta_before == [4.0, 3.0]

    nav = _make_nav(pattern, delta=delta_before, is_anomaly=False)

    # Override dim "A" (index 0) to 5.0 ⇒ shape_after = [5.0, 4.0]
    # delta_after = L_inv @ ([5,4] - [1,2]) = L_inv @ [4, 2]
    #             = [8.0, 5.0]
    result = nav.simulate_dimension_change(
        "E1",
        pattern_id="test_pattern",
        line_id="line_A",
        set_dimension={"A": 5.0},
        top_n=2,
    )

    expected_delta_after = (
        cholesky_inv @ np.array([4.0, 2.0], dtype=np.float64)
    ).tolist()
    assert expected_delta_after == [8.0, 5.0]
    assert result["delta_norm_after"] == pytest.approx(math.sqrt(64.0 + 25.0), abs=1e-9)
    # Both delta entries moved despite overriding only one shape dim.
    row_a = result["dimensions_overridden"][0]
    assert row_a["old_delta"] == pytest.approx(4.0, abs=1e-9)
    assert row_a["new_delta"] == pytest.approx(8.0, abs=1e-9)
    # Witness dim B (index 1) reflects the cross-coupled new delta (5.0).
    witness_by_label = {
        row["dim_label"]: row for row in result["top_witness_dims_after"]
    }
    assert witness_by_label["B"]["delta"] == pytest.approx(5.0, abs=1e-9)
    assert witness_by_label["A"]["delta"] == pytest.approx(8.0, abs=1e-9)


def test_determinism():
    """Calling the method twice with identical inputs must return
    numerically identical results (tighter than 1e-12)."""
    pattern = _make_pattern(
        line_ids=["A", "B", "C"],
        mu=[1.0, 1.0, 1.0],
        sigma_diag=[1.0, 1.0, 1.0],
        theta=[3.0, 3.0, 3.0],
    )
    nav = _make_nav(pattern, delta=[2.0, 1.0, 0.0], is_anomaly=False)

    args = {
        "pattern_id": "test_pattern",
        "line_id": "line_A",
        "set_dimension": {"A": 0.5, "C": 0.25},
        "top_n": 3,
    }
    r1 = nav.simulate_dimension_change("E1", **args)
    r2 = nav.simulate_dimension_change("E1", **args)

    assert r1["delta_norm_after"] == pytest.approx(r2["delta_norm_after"], abs=1e-15)
    assert r1["delta_norm_pct_change"] == pytest.approx(
        r2["delta_norm_pct_change"], abs=1e-15,
    )
    assert r1["top_witness_dims_after"] == r2["top_witness_dims_after"]
    assert r1["dimensions_overridden"] == r2["dimensions_overridden"]
