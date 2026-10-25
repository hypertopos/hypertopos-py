# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""FrozenCalibration: fit on an entity table, score against the frozen population."""

from __future__ import annotations

import pickle

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.scoring import POOLED_GROUP, FrozenCalibration

DIMS = ("a", "b", "c")


def _table(n: int = 2000, seed: int = 0, groups: list[str] | None = None) -> pa.Table:
    rng = np.random.default_rng(seed)
    cols: dict[str, object] = {
        "entity": [f"E{i}" for i in range(n)],
        "a": rng.lognormal(0, 1, n),
        "b": rng.poisson(4, n).astype(np.int64),
        "c": rng.normal(10, 3, n),
    }
    if groups is not None:
        cols["segment"] = groups
    return pa.table(cols)


def _oracle(x: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray, float]:
    mu = x.mean(axis=0)
    sigma = np.maximum(x.std(axis=0), 1e-2)
    norm = np.linalg.norm((x - mu) / sigma, axis=1)
    return norm, (x - mu) / sigma, float(np.percentile(norm, percentile))


def _matrix(table: pa.Table) -> np.ndarray:
    return np.column_stack([table[d].to_numpy().astype(np.float32) for d in DIMS])


def test_fit_and_score_match_a_numpy_oracle():
    table = _table()
    scores = FrozenCalibration.fit(table, DIMS, percentile=99.0).score(table)
    norm, delta, theta = _oracle(_matrix(table).astype(np.float64), 99.0)
    np.testing.assert_allclose(scores.delta_norm, norm, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scores.delta, delta, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(scores.is_anomaly, norm >= theta)
    assert scores.is_anomaly.mean() == pytest.approx(0.01, abs=0.002)
    smaller = (norm[None, :] < norm[:, None]).sum(axis=1)
    np.testing.assert_allclose(scores.rank_pct, smaller / len(norm) * 100, atol=1e-3)
    np.testing.assert_allclose(scores.conformal_p, (len(norm) - smaller) / len(norm), atol=1e-6)
    assert scores.calibration_scope is None


def test_extreme_values_keep_their_distance():
    # No saturation: an entity far beyond the 99th percentile must stay farther
    # than one just beyond it (a p99 clip would put both at the same ceiling).
    base = _table(seed=1)
    a = base["a"].to_numpy()
    p99 = float(np.percentile(a, 99))
    extra = pa.table(
        {"entity": ["NEAR", "FAR"], "a": [1.1 * p99, 10 * p99], "b": [4, 4], "c": [10.0, 10.0]}
    )
    calibration = FrozenCalibration.fit(base, DIMS)
    near, far = calibration.score(extra).delta_norm
    assert far > near * 5


def test_score_uses_the_frozen_population_without_refitting():
    calibration = FrozenCalibration.fit(_table(seed=2), DIMS)
    one = pa.table({"entity": ["X"], "a": [1.0], "b": [4], "c": [10.0]})
    mu, sigma = calibration.raw_params(one)
    expected = np.linalg.norm((np.array([1.0, 4.0, 10.0]) - mu[0]) / sigma[0])
    assert calibration.score(one).delta_norm[0] == pytest.approx(expected, rel=1e-5)


def test_null_dimension_counts_as_zero():
    table = _table(n=500)
    with_null = pa.table(
        {"entity": ["N"], "a": pa.array([None], pa.float64()), "b": [4], "c": [10.0]}
    )
    as_zero = pa.table({"entity": ["N"], "a": [0.0], "b": [4], "c": [10.0]})
    calibration = FrozenCalibration.fit(table, DIMS)
    assert calibration.score(with_null).delta_norm[0] == calibration.score(as_zero).delta_norm[0]


def test_groups_are_calibrated_separately():
    n = 3000
    segment = ["big" if i < 2000 else "small" for i in range(n)]
    table = _table(n=n, groups=segment)
    shifted = table.set_column(
        table.schema.get_field_index("c"),
        "c",
        pa.array(np.where(np.array(segment) == "small", table["c"].to_numpy() + 50, table["c"])),
    )
    calibration = FrozenCalibration.fit(shifted, DIMS, group_by="segment")
    scores = calibration.score(shifted)
    for name in ("big", "small"):
        mask = np.array(segment) == name
        norm, _, theta = _oracle(_matrix(shifted)[mask].astype(np.float64), 99.0)
        np.testing.assert_allclose(scores.delta_norm[mask], norm, rtol=1e-5, atol=1e-5)
        assert scores.is_anomaly[mask].mean() == pytest.approx(0.01, abs=0.005)
    assert set(scores.calibration_scope) == {"group"}


def test_small_groups_are_pooled_and_unseen_groups_use_global():
    segment = ["big"] * 1900 + [f"tiny{i}" for i in range(100)]
    table = _table(n=2000, groups=segment)
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment", min_group_size=100)
    assert set(calibration.groups or {}) == {"big", POOLED_GROUP}
    scores = calibration.score(table)
    scope = scores.calibration_scope
    assert set(scope[:1900]) == {"group"}
    assert set(scope[1900:]) == {"pooled"}
    # The pooled group has 100 members, so its threshold can flag someone.
    assert calibration.groups[POOLED_GROUP].theta_norm > 0
    unseen = pa.table(
        {"entity": ["U"], "a": [1.0], "b": [4], "c": [10.0], "segment": ["never-seen"]}
    )
    assert calibration.score(unseen).calibration_scope.tolist() == ["global"]


@pytest.mark.parametrize("grouped", [False, True], ids=["global", "grouped"])
def test_calibration_matches_the_sphere_built_from_it(tmp_path, grouped):
    """The contract: a sphere built from the calibration reports the same geometry."""
    import lance

    from hypertopos.spark._build import PATTERN_ID, build_sphere

    rng = np.random.default_rng(3)
    n = 2600
    segment = ["big"] * 2000 + ["mid"] * 550 + [f"tiny{i % 5}" for i in range(50)]
    table = pa.table(
        {
            "entity": [f"E{i}" for i in range(n)],
            "a": rng.lognormal(0, 1.5, n),
            "b": rng.poisson(4, n).astype(np.int64),
            "c": np.where(rng.random(n) < 0.97, 0.0, rng.uniform(0, 0.05, n)),  # near-constant
            "segment": segment,
        }
    )
    table = table.append_column("n", pa.array(rng.integers(1, 40, n)))
    calibration = FrozenCalibration.fit(
        table, DIMS, group_by="segment" if grouped else None, shrink={"c": "n"}
    )
    scores = calibration.score(table)
    build_sphere(table, calibration, str(tmp_path / "s"))
    geo = lance.dataset(str(tmp_path / "s" / "geometry" / PATTERN_ID / "data.lance")).to_table()
    order = np.argsort([int(k[1:]) for k in geo["primary_key"].to_pylist()])
    np.testing.assert_allclose(
        geo["delta_norm"].to_numpy()[order], scores.delta_norm, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(geo["is_anomaly"].to_numpy()[order], scores.is_anomaly)
    np.testing.assert_allclose(geo["delta_rank_pct"].to_numpy()[order], scores.rank_pct, atol=1e-4)
    np.testing.assert_allclose(geo["conformal_p"].to_numpy()[order], scores.conformal_p, atol=1e-6)
    if grouped:
        assert set(scores.calibration_scope[2550:]) == {"pooled"}


def test_small_counts_are_shrunk_toward_the_population():
    # A share measured on one event is pulled to the population; on many events it stays.
    rng = np.random.default_rng(5)
    n = 3000
    count = rng.integers(5, 60, n)
    share = np.clip(rng.normal(0.3, 0.08, n), 0, 1)
    count[:2], share[:2] = [1, 200], [1.0, 1.0]  # one payment vs two hundred, both at 100 %
    table = pa.table({"entity": [f"E{i}" for i in range(n)], "share": share, "n": count})
    plain = FrozenCalibration.fit(table, ["share"]).score(table)
    shrunk_cal = FrozenCalibration.fit(table, ["share"], shrink={"share": "n"})
    shrunk = shrunk_cal.score(table)
    mean = share.mean()
    assert shrunk_cal.shapes(table)[0, 0] == pytest.approx((1.0 + 3 * mean) / (1 + 3), rel=1e-5)
    assert shrunk.delta_norm[0] < plain.delta_norm[0] / 2  # one event is no longer extreme
    # 200 events: not discounted (less small-count noise in the population makes it stand out more)
    assert shrunk.delta_norm[1] >= plain.delta_norm[1]
    assert shrunk.is_anomaly[1]
    assert not shrunk.is_anomaly[0]


def test_shrinkage_uses_the_frozen_prior_when_scoring_new_rows():
    table = pa.table({"entity": ["A", "B", "C", "D"], "share": [0.2, 0.4, 0.6, 0.8], "n": [5] * 4})
    calibration = FrozenCalibration.fit(table, ["share"], shrink={"share": "n"})
    new = pa.table({"entity": ["X"], "share": [1.0], "n": [1]})
    assert calibration.shapes(new)[0, 0] == pytest.approx((1.0 + 3 * 0.5) / 4, rel=1e-6)


def test_pickle_round_trip():
    table = _table(groups=["x" if i % 2 else "y" for i in range(2000)])
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment")
    restored = pickle.loads(pickle.dumps(calibration))
    np.testing.assert_array_equal(
        restored.score(table).delta_norm, calibration.score(table).delta_norm
    )
