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

from hypertopos.engine.robust import dim_shares_rows, fit_reference, is_anomaly
from hypertopos.engine.robust import score as score_reference
from hypertopos.scoring import (
    POOLED_GROUP,
    FrozenCalibration,
    effective_min_group_size,
    score_shares,
)

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
    norm, delta, _ = _oracle(_matrix(table).astype(np.float64), 99.0)
    np.testing.assert_allclose(scores.delta_norm, norm, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scores.delta, delta, rtol=1e-4, atol=1e-5)
    assert scores.calibration_scope is None


def test_score_layer_is_the_calibrated_evidence_score():
    # anomaly_score, p and rank come from the robust reference fitted on the same
    # values a sphere build scores; is_anomaly is the conformal p at the percentile.
    table = _table()
    calibration = FrozenCalibration.fit(table, DIMS, percentile=99.0)
    scores = calibration.score(table, members=True)
    _, fitted = fit_reference(_matrix(table).astype(np.float64), DIMS)
    assert calibration.reference.axes == ("mahal",)  # no events: no current axis
    np.testing.assert_array_equal(scores.anomaly_score, fitted.anomaly_score)
    np.testing.assert_array_equal(scores.conformal_p, fitted.conformal_p)
    np.testing.assert_array_equal(scores.score_rank_pct, fitted.score_rank_pct)
    np.testing.assert_array_equal(scores.is_anomaly, is_anomaly(fitted.conformal_p, 99.0))
    assert scores.is_anomaly.mean() == pytest.approx(0.01, abs=0.002)


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
        norm, _, _ = _oracle(_matrix(shifted)[mask].astype(np.float64), 99.0)
        np.testing.assert_allclose(scores.delta_norm[mask], norm, rtol=1e-5, atol=1e-5)
    assert set(scores.calibration_scope) == {"group"}
    # The score is global, as in a sphere build: groups change only the deviations.
    overall = FrozenCalibration.fit(shifted, DIMS).score(shifted)
    np.testing.assert_array_equal(scores.anomaly_score, overall.anomaly_score)
    np.testing.assert_array_equal(scores.is_anomaly, overall.is_anomaly)


def test_small_groups_are_pooled_and_unseen_groups_use_global():
    segment = ["big"] * 1900 + [f"tiny{i}" for i in range(100)]
    table = _table(n=2000, groups=segment)
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment", min_group_size=100)
    assert set(calibration.groups or {}) == {"big", POOLED_GROUP}
    scores = calibration.score(table)
    scope = scores.calibration_scope
    assert set(scope[:1900]) == {"group"}
    assert set(scope[1900:]) == {"pooled"}
    assert calibration.groups[POOLED_GROUP].size == 100
    unseen = pa.table(
        {"entity": ["U"], "a": [1.0], "b": [4], "c": [10.0], "segment": ["never-seen"]}
    )
    assert calibration.score(unseen).calibration_scope.tolist() == ["global"]


@pytest.mark.parametrize(
    ("segment", "min_group_size", "scopes"),
    [
        (
            ["big"] * 2000 + ["mid"] * 600 + [f"tiny{i % 5}" for i in range(250)],
            None,
            {"group", "pooled"},
        ),
        (
            ["big"] * 2000 + ["mid"] * 550 + [f"tiny{i % 5}" for i in range(50)],
            None,
            {"group", "global"},
        ),
        (
            ["big"] * 2000 + ["mid"] * 550 + [f"tiny{i % 5}" for i in range(50)],
            600,
            {"group", "pooled"},
        ),
    ],
    ids=["pool", "small-pool", "explicit-size"],
)
def test_per_group_calibration_matches_the_sphere_built_from_it(
    tmp_path, segment, min_group_size, scopes
):
    import lance

    from hypertopos.spark._build import PATTERN_ID, build_sphere

    n = len(segment)
    rng = np.random.default_rng(5)
    table = pa.table(
        {
            "entity": [f"E{i}" for i in range(n)],
            "a": rng.lognormal(0, 1.5, n),
            "b": rng.poisson(4, n).astype(np.int64),
            "c": rng.gamma(2.0, 1.0, n),
            "segment": segment,
        }
    )
    calibration = FrozenCalibration.fit(
        table, DIMS, group_by="segment", score_per_group=True, min_group_size=min_group_size
    )
    scores = calibration.score(table, members=True)
    assert set(scores.calibration_scope) == scopes
    build_sphere(table, calibration, str(tmp_path / "s"))
    geo = lance.dataset(str(tmp_path / "s" / "geometry" / PATTERN_ID / "data.lance")).to_table()
    order = np.argsort([int(k[1:]) for k in geo["primary_key"].to_pylist()])
    np.testing.assert_allclose(geo["conformal_p"].to_numpy()[order], scores.conformal_p, atol=1e-6)
    np.testing.assert_array_equal(geo["is_anomaly"].to_numpy()[order], scores.is_anomaly)
    np.testing.assert_allclose(
        geo["anomaly_score"].to_numpy()[order], scores.anomaly_score, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        geo["score_rank_pct"].to_numpy()[order], scores.score_rank_pct, atol=1e-4
    )
    np.testing.assert_allclose(
        geo["delta_norm"].to_numpy()[order], scores.delta_norm, rtol=1e-6, atol=1e-6
    )
    stored = np.array(geo["score_group"].to_pylist(), dtype=object)[order]
    assert ((stored == None) == (scores.calibration_scope == "global")).all()  # noqa: E711


@pytest.mark.parametrize("grouped", [False, True], ids=["global", "grouped"])
def test_calibration_matches_the_sphere_built_from_it(tmp_path, grouped):
    """The contract: a sphere built from the calibration reports the same scores."""
    import json

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
    scores = calibration.score(table, members=True)
    build_sphere(table, calibration, str(tmp_path / "s"))
    geo = lance.dataset(str(tmp_path / "s" / "geometry" / PATTERN_ID / "data.lance")).to_table()
    order = np.argsort([int(k[1:]) for k in geo["primary_key"].to_pylist()])
    np.testing.assert_allclose(
        geo["delta_norm"].to_numpy()[order], scores.delta_norm, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(geo["is_anomaly"].to_numpy()[order], scores.is_anomaly)
    np.testing.assert_allclose(
        geo["anomaly_score"].to_numpy()[order], scores.anomaly_score, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        geo["score_rank_pct"].to_numpy()[order], scores.score_rank_pct, atol=1e-4
    )
    np.testing.assert_allclose(geo["conformal_p"].to_numpy()[order], scores.conformal_p, atol=1e-6)
    meta = json.loads((tmp_path / "s" / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"))
    assert meta["patterns"][PATTERN_ID]["score_dims"] == list(DIMS)
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
    np.testing.assert_array_equal(
        restored.score(table).anomaly_score, calibration.score(table).anomaly_score
    )


def test_p_value_and_rank_are_exact_ratios():
    # Ratios of counts: a float32 round trip printed 0.001 as 0.0010000000474974513.
    table = _table()
    scores = FrozenCalibration.fit(table, DIMS).score(table, members=True)
    n = len(scores.anomaly_score)
    reference = np.sort(scores.anomaly_score)
    above = n - np.searchsorted(reference, scores.anomaly_score, side="left")
    assert scores.conformal_p.dtype == np.float64
    assert scores.score_rank_pct.dtype == np.float64
    np.testing.assert_array_equal(scores.conformal_p, (1.0 + above) / (n + 1.0))
    np.testing.assert_array_equal(scores.score_rank_pct, 100.0 * (n - above) / n)


def _per_group_table(tiny: int) -> tuple[pa.Table, np.ndarray]:
    # big and mid are different kinds of entity (c shifted); ``tiny`` groups of 50 pool.
    segment = np.array(
        ["big"] * 2000 + ["mid"] * 600 + [f"tiny{i % tiny}" for i in range(50 * tiny)]
    )
    table = _table(n=len(segment), seed=7, groups=segment.tolist())
    c = table["c"].to_numpy() + np.where(segment == "mid", 40.0, 0.0)
    return table.set_column(table.schema.get_field_index("c"), "c", pa.array(c)), segment


def test_per_group_score_is_each_groups_own_reference():
    table, segment = _per_group_table(tiny=5)  # pool of 250: it gets its own reference
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment", score_per_group=True)
    scores = calibration.score(table, members=True)
    x = _matrix(table).astype(np.float64)
    pool = np.char.startswith(segment, "tiny")
    for name, mask in (("big", segment == "big"), ("mid", segment == "mid"), (POOLED_GROUP, pool)):
        _, fitted = fit_reference(x[mask], DIMS)
        np.testing.assert_array_equal(scores.conformal_p[mask], fitted.conformal_p)
        np.testing.assert_array_equal(scores.anomaly_score[mask], fitted.anomaly_score)
        np.testing.assert_array_equal(scores.score_rank_pct[mask], fitted.score_rank_pct)
        assert calibration.groups[name].reference is not None
    np.testing.assert_array_equal(scores.is_anomaly, is_anomaly(scores.conformal_p, 99.0))
    assert set(scores.calibration_scope[segment == "big"]) == {"group"}
    assert set(scores.calibration_scope[pool]) == {"pooled"}
    # The population reference stays the one a global fit gives.
    population, fitted = fit_reference(x, DIMS)
    np.testing.assert_array_equal(calibration.reference.cal_sorted, population.cal_sorted)
    assert not np.array_equal(scores.conformal_p, fitted.conformal_p)


def test_per_group_undersized_pool_and_unseen_group_score_globally():
    table, segment = _per_group_table(tiny=3)  # pool of 150: below the reference size
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment", score_per_group=True)
    scores = calibration.score(table, members=True)
    pool = np.char.startswith(segment, "tiny")
    assert calibration.groups[POOLED_GROUP].reference is None
    assert set(scores.calibration_scope[pool]) == {"global"}
    x = _matrix(table).astype(np.float64)
    _, population = fit_reference(x, DIMS)
    np.testing.assert_array_equal(scores.conformal_p[pool], population.conformal_p[pool])
    # The deviations still come from the pool's mean and spread, pooled at the same size.
    off = FrozenCalibration.fit(table, DIMS, group_by="segment", min_group_size=199)
    np.testing.assert_array_equal(scores.delta, off.score(table, members=True).delta)
    unseen = pa.table(
        {"entity": ["U"], "a": [1.0], "b": [4], "c": [10.0], "segment": ["never-seen"]}
    )
    new = calibration.score(unseen)
    assert new.calibration_scope.tolist() == ["global"]
    expected = score_reference(calibration.reference, _matrix(unseen).astype(np.float64))
    np.testing.assert_array_equal(new.conformal_p, expected.conformal_p)


def test_per_group_new_rows_use_the_groups_out_of_fold_calibration():
    table, segment = _per_group_table(tiny=5)
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment", score_per_group=True)
    mid = table.filter(pa.array(segment == "mid"))
    ref = calibration.groups["mid"].reference
    assert ref.mahal_sorted_oof is not None
    expected = score_reference(ref, _matrix(mid).astype(np.float64))
    np.testing.assert_array_equal(calibration.score(mid).conformal_p, expected.conformal_p)


def test_per_group_min_group_size_below_the_reference_size_raises():
    table, _ = _per_group_table(tiny=5)
    with pytest.raises(ValueError, match="below 199"):
        FrozenCalibration.fit(
            table, DIMS, group_by="segment", score_per_group=True, min_group_size=100
        )
    with pytest.raises(ValueError, match="group_by"):
        FrozenCalibration.fit(table, DIMS, score_per_group=True)
    # Off: a small explicit value pools as before.
    off = FrozenCalibration.fit(table, DIMS, group_by="segment", min_group_size=10)
    assert off.pooled == frozenset()


def test_effective_min_group_size():
    assert effective_min_group_size(None, 99.0, score_per_group=False) == 100
    assert effective_min_group_size(10, 99.0, score_per_group=False) == 10
    assert effective_min_group_size(None, 99.0, score_per_group=True) == 199
    assert effective_min_group_size(500, 99.0, score_per_group=True) == 500
    with pytest.raises(ValueError, match="below 199"):
        effective_min_group_size(150, 99.0, score_per_group=True)


def test_per_group_score_shares_use_the_groups_reference():
    table, segment = _per_group_table(tiny=5)
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment", score_per_group=True)
    scores = calibration.score(table, members=True)
    shares = calibration.score_shares(table, scores.score_delta)
    mid = segment == "mid"
    expected = dim_shares_rows(calibration.groups["mid"].reference, scores.score_delta[mid])
    np.testing.assert_array_equal(shares[mid], expected)
    # Off: the population reference, as the module-level function gives.
    off = FrozenCalibration.fit(table, DIMS, group_by="segment")
    off_scores = off.score(table, members=True)
    np.testing.assert_array_equal(
        off.score_shares(table, off_scores.score_delta),
        score_shares(off.reference, off_scores.score_delta),
    )


def test_per_group_score_of_an_empty_table():
    table, _ = _per_group_table(tiny=5)
    calibration = FrozenCalibration.fit(table, DIMS, group_by="segment", score_per_group=True)
    empty = table.slice(0, 0)
    scores = calibration.score(empty)
    assert len(scores.conformal_p) == 0
    assert calibration.score_shares(empty, scores.score_delta).shape == (0, len(DIMS))
