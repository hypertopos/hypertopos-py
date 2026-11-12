from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.scoring import FrozenCalibration

DIMS = ("a", "b", "c")


def _table(n: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    seg = np.where(np.arange(n) % 2 == 0, "x", "y")
    shift = np.where(seg == "x", 0.0, 5.0)
    cols = {d: pa.array(rng.normal(size=n) + shift) for d in DIMS}
    return pa.table({**cols, "seg": pa.array(seg)})


def test_rank_is_the_share_of_the_population_below_the_value():
    table = _table(4_000, 1)
    cal = FrozenCalibration.fit(table, DIMS)
    ranks = cal.dimension_rank_pct(table, cal.score(table, members=True).score_delta)
    assert ranks.shape == (table.num_rows, len(DIMS))
    assert ranks.min() > 0.0
    assert ranks.max() < 100.0
    for d, name in enumerate(DIMS):
        values = table[name].to_numpy()
        empirical = 100.0 * (np.argsort(np.argsort(values)) + 0.5) / len(values)
        np.testing.assert_allclose(ranks[:, d], empirical, atol=0.05)


def test_rank_is_within_the_entitys_own_group_with_score_per_group():
    table = _table(4_000, 2)
    cal = FrozenCalibration.fit(table, DIMS, group_by="seg", score_per_group=True)
    ranks = cal.dimension_rank_pct(table, cal.score(table, members=True).score_delta)
    seg = np.asarray(table["seg"].to_pylist())
    # group y sits 5 sd above group x: pooled, every y value would rank in the top half
    for group in ("x", "y"):
        assert abs(float(np.median(ranks[seg == group, 0])) - 50.0) < 2.0


def test_a_new_row_gets_its_rank_in_the_fitted_population():
    table = _table(4_000, 3)
    cal = FrozenCalibration.fit(table, DIMS)
    low, high = np.quantile(table["a"].to_numpy(), [0.10, 0.90])
    new = pa.table({"a": [low, high], "b": [0.0, 0.0], "c": [0.0, 0.0], "seg": ["x", "x"]})
    ranks = cal.dimension_rank_pct(new, cal.score(new).score_delta)
    np.testing.assert_allclose(ranks[:, 0], [10.0, 90.0], atol=0.2)


def _shares(n: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    share = np.where(rng.random(n) < 0.04, rng.random(n), 0.0)
    seg = np.where(np.arange(n) % 2 == 0, "x", "y")
    return pa.table(
        {
            "share": pa.array(share + np.where((seg == "y") & (share > 0), 1.0, 0.0)),
            "other": pa.array(rng.normal(size=n)),
            "n_out": pa.array(rng.integers(1, 200, n).astype(np.float64)),
            "seg": pa.array(seg),
        }
    )


def test_an_adjusted_dimension_ranks_its_measured_value():
    # A share measured on few events is pulled toward the population mean before it
    # is scored. The rank a user filters on is of the measured value: entities with
    # the same measured share have the same rank whatever their event counts.
    table = _shares(5_000, 4)
    cal = FrozenCalibration.fit(table, ("share", "other"), shrink={"share": "n_out"})
    ranks = cal.dimension_rank_pct(table, cal.score(table, members=True).score_delta)[:, 0]
    share = table["share"].to_numpy()
    zeros = ranks[share == 0.0]
    assert zeros.max() - zeros.min() < 1e-6
    assert zeros[0] == pytest.approx(50.0 * np.mean(share == 0.0), abs=0.1)
    order = np.argsort(share)
    assert np.all(np.diff(ranks[order]) >= -1e-6)
    assert ranks[share > 0].min() > zeros[0]


def test_the_measured_rank_is_within_the_group_with_score_per_group():
    table = _shares(8_000, 5)
    cal = FrozenCalibration.fit(
        table, ("share", "other"), shrink={"share": "n_out"}, group_by="seg", score_per_group=True
    )
    ranks = cal.dimension_rank_pct(table, cal.score(table, members=True).score_delta)[:, 0]
    share = table["share"].to_numpy()
    seg = np.asarray(table["seg"].to_pylist())
    # group y's nonzero shares are all above group x's, yet each group's own top ranks near 100
    for group in ("x", "y"):
        top = ranks[(seg == group) & (share == share[seg == group].max())]
        assert top[0] > 99.0
        assert ranks[(seg == group) & (share == 0.0)].max() < 50.0


def test_a_calibration_without_measured_grids_is_refused():
    table = _shares(3_000, 6)
    cal = FrozenCalibration.fit(table, ("share", "other"), shrink={"share": "n_out"})
    old = dataclasses.replace(cal, raw_grid=None)
    with pytest.raises(ValueError, match="recalibrate"):
        old.dimension_rank_pct(table, old.score(table, members=True).score_delta)


def test_a_nan_in_an_adjusted_dimension_counts_as_zero_everywhere():
    rng = np.random.default_rng(7)
    n = 1_000
    share = rng.uniform(size=n)
    share[3] = np.nan
    table = pa.table(
        {
            "a": pa.array(rng.normal(size=n)),
            "share": pa.array(share),
            "count": pa.array(rng.integers(1, 50, n).astype(np.float64)),
        }
    )
    cal = FrozenCalibration.fit(table, ("a", "share"), shrink={"share": "count"})
    assert np.isfinite(cal.shrink["share"][1])  # the population mean the adjustment pulls toward
    assert np.isfinite(cal.raw_grid).all()
    ranks = cal.dimension_rank_pct(table, cal.score(table, members=True).score_delta)
    assert np.isfinite(ranks).all()
    assert ranks[np.argmax(np.nan_to_num(share)), 1] > 99.0
