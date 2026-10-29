from __future__ import annotations

import io

import numpy as np
import pytest
from scipy.stats import norm, rankdata, spearmanr

from hypertopos.engine import robust
from hypertopos.engine.robust import (
    AXES_ALL,
    EventArrays,
    attribute,
    conformal_p,
    counterfactual,
    fit_cdf,
    fit_reference,
    is_anomaly,
    ledoit_wolf,
    reference_from_arrays,
    reference_to_arrays,
    score,
    to_gauss,
)

DAY = 86400.0


def _population(n: int, seed: int, drift: float = 0.0) -> tuple[np.ndarray, EventArrays]:
    """Normal dimensions plus 4 events per entity, two in each calendar half."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    code = np.repeat(np.arange(n), 4)
    ts = np.tile([0.0, DAY, 3 * DAY, 4 * DAY], n)
    direction = np.tile([0, 1, 0, 1], n).astype(np.int8)
    base = rng.lognormal(size=n)
    amount = np.repeat(base, 4) * np.tile([1.0, 1.0, 1.0 + drift, 1.0 + drift], n)
    return x, EventArrays(code, direction, ts, amount)


def _subset_events(ev: EventArrays, mask: np.ndarray) -> EventArrays:
    new_index = np.cumsum(mask) - 1
    keep = mask[ev.code]
    return EventArrays(new_index[ev.code[keep]], ev.direction[keep], ev.ts[keep], ev.amount[keep])


def _reverse_entity(ev: EventArrays, entity: int, factor: float = 0.25) -> EventArrays:
    amount = ev.amount.copy()
    second = (ev.code == entity) & (ev.ts >= 3 * DAY)
    amount[second] *= factor
    return EventArrays(ev.code, ev.direction, ev.ts, amount)


def test_conformal_p_formula():
    ref = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
    p = conformal_p(ref, np.array([4.0, 2.5, 0.0, 9.0]))
    np.testing.assert_allclose(p, [2 / 5, 3 / 5, 5 / 5, 1 / 5])


def test_conformal_p_uniform_on_exchangeable_rows():
    rng = np.random.default_rng(0)
    ref = np.sort(rng.normal(size=20_000))
    p = conformal_p(ref, rng.normal(size=20_000))
    assert 0.008 <= (p <= 0.01).mean() <= 0.012


def test_is_anomaly_threshold():
    assert is_anomaly(np.array([0.004, 0.006]), 99.5).tolist() == [True, False]


def test_ledoit_wolf_matches_reference_estimator():
    covariance = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(10)
    z = rng.normal(size=(3_000, 6)) @ rng.normal(size=(6, 6))
    location, precision = ledoit_wolf(z)
    lw = covariance.LedoitWolf().fit(z)
    np.testing.assert_allclose(location, lw.location_, rtol=1e-10)
    np.testing.assert_allclose(precision, lw.precision_, rtol=1e-6, atol=1e-9)


def test_sketch_matches_exact_rank_gauss_ranking():
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.lognormal(size=50_000), rng.poisson(2, 50_000).astype(float)])
    grid, levels = fit_cdf(x)
    z = to_gauss(grid, levels, x)
    exact = norm.ppf((rankdata(x[:, 0]) - 0.5) / len(x))
    assert spearmanr(z[:, 0], exact).statistic > 0.9999
    assert np.isfinite(z).all()


def test_constant_dimension_is_finite():
    x = np.column_stack([np.zeros(500), np.arange(500.0)])
    _, sc = fit_reference(x, ("a", "b"), axes=("mahal",))
    assert np.isfinite(sc.delta_norm).all()
    assert np.isfinite(sc.conformal_p).all()


def test_out_of_range_rows_score_finite():
    x = np.random.default_rng(2).normal(size=(2_000, 3))
    ref, _ = fit_reference(x, ("a", "b", "c"), axes=("mahal",))
    s = score(ref, np.full((1, 3), 1e9))
    assert np.isfinite(s.anomaly_score).all()
    assert s.conformal_p[0] == pytest.approx(1 / 2001)


def test_delta_is_rank_gauss_per_named_dimension():
    x = np.random.default_rng(7).lognormal(size=(3_000, 2))
    ref, sc = fit_reference(x, ("a", "b"), axes=("mahal",))
    z = to_gauss(ref.cdf_grid, ref.cdf_levels, x)
    np.testing.assert_allclose(sc.delta, (z - ref.lw_location).astype(np.float32), atol=1e-6)


def test_inactive_entity_gets_median_current_rank():
    n = 200
    rng = np.random.default_rng(4)
    code = np.repeat(np.arange(n), 4)
    ts = np.tile([0.0, DAY, 3 * DAY, 4 * DAY], n)
    ts[code == n - 1] = [0.0, 1.0, 2.0, 3.0]
    ev = EventArrays(code, np.tile([0, 1, 0, 1], n).astype(np.int8), ts, rng.lognormal(size=4 * n))
    ref, sc = fit_reference(rng.normal(size=(n, 3)), ("a", "b", "c"), events=ev)
    j = ref.axes.index("current")
    assert np.isnan(sc.axis_values["current"][n - 1])
    assert sc.axis_ranks[n - 1, j] == 0.5


def test_current_axis_dropped_without_events():
    ref, _ = fit_reference(np.random.default_rng(8).normal(size=(500, 3)), ("a", "b", "c"))
    assert ref.axes == ("mahal",)


def test_small_population_clips_k():
    x = np.random.default_rng(3).normal(size=(30, 4))
    _, sc = fit_reference(x, tuple("abcd"), axes=("mahal", "knn"))
    assert np.isfinite(sc.anomaly_score).all()


def test_fit_rows_scored_inductively_match_for_non_sample_rows(monkeypatch):
    monkeypatch.setattr(robust, "MAX_REFERENCE_ROWS", 1_000)
    x, ev = _population(n=3_000, seed=5)
    ref, fit = fit_reference(x, ("a", "b", "c"), axes=AXES_ALL, events=ev)
    assert fit.sample_index is not None
    out = ~np.isin(np.arange(3_000), fit.sample_index)
    again = score(ref, x[out], _subset_events(ev, out), members=True)
    np.testing.assert_allclose(fit.anomaly_score[out], again.anomaly_score, atol=1e-9)
    np.testing.assert_allclose(fit.conformal_p[out], again.conformal_p, atol=1e-12)


def test_planted_mover_ranks_top():
    x, ev = _population(n=2_000, seed=6, drift=1.0)
    ev = _reverse_entity(ev, entity=0)
    _, sc = fit_reference(x, ("a", "b", "c"), events=ev)
    assert int(np.nanargmax(sc.axis_values["current"])) == 0


def test_rows_without_events_are_scored_on_the_other_axes():
    x, ev = _population(n=4_000, seed=11)
    rng = np.random.default_rng(13)
    ev = EventArrays(ev.code, ev.direction, ev.ts, rng.lognormal(sigma=1.5, size=len(ev.code)))
    ref, fit = fit_reference(x, ("a", "b", "c"), events=ev)
    assert np.unique(fit.axis_ranks[:, 1]).size > 1000  # the current axis really varies
    assert ref.axes == ("mahal", "current")
    extreme = score(ref, np.full((1, 3), 50.0))  # no events: e.g. an ingested entity
    assert extreme.conformal_p[0] == pytest.approx(1 / 4001)
    assert np.isnan(extreme.axis_values["current"][0])
    fresh = np.random.default_rng(12).normal(size=(4_000, 3))
    p = score(ref, fresh).conformal_p
    assert 0.007 <= (p <= 0.01).mean() <= 0.013  # still calibrated without the current axis


def test_attribution_sums_exactly_to_the_bits():
    x, ev = _population(n=3_000, seed=14)
    rng = np.random.default_rng(15)
    ev = EventArrays(ev.code, ev.direction, ev.ts, rng.lognormal(sigma=1.5, size=len(ev.code)))
    ref, fit = fit_reference(x, ("a", "b", "c"), events=ev)
    for i in np.argsort(fit.conformal_p)[:5]:
        a = attribute(ref, fit.delta[i], fit.axis_ranks[i], float(fit.conformal_p[i]))
        assert a.bits == pytest.approx(-np.log2(fit.conformal_p[i]))
        total = sum(a.axis_bits.values())
        assert total == pytest.approx(a.bits, abs=1e-9)
        assert sum(a.dim_bits.values()) == pytest.approx(a.axis_bits["mahal"], abs=1e-9)


def test_attribution_names_the_planted_dimension():
    rng = np.random.default_rng(16)
    x = rng.normal(size=(5_000, 4))
    x[0, 2] = 12.0  # one dimension far out, the others ordinary
    ref, fit = fit_reference(x, ("a", "b", "c", "d"), axes=("mahal",))
    a = attribute(ref, fit.delta[0], fit.axis_ranks[0], float(fit.conformal_p[0]))
    assert max(a.dim_bits, key=a.dim_bits.get) == "c"
    assert a.dim_bits["c"] / a.axis_bits["mahal"] > 0.7


def test_counterfactual_without_change_reproduces_the_stored_score():
    x, ev = _population(n=2_000, seed=17)
    ref, fit = fit_reference(x, ("a", "b", "c"), events=ev)
    for i in (0, 7, int(np.argmin(fit.conformal_p))):
        cf = counterfactual(ref, fit.delta[i], fit.axis_ranks[i], {})
        assert cf.anomaly_score == pytest.approx(fit.anomaly_score[i])
        assert cf.conformal_p == pytest.approx(fit.conformal_p[i])
        np.testing.assert_allclose(cf.axis_ranks, fit.axis_ranks[i], atol=1e-6)


def test_counterfactual_rescoring_matches_scoring_the_changed_row():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(3_000, 3))
    ref, fit = fit_reference(x, ("a", "b", "c"), axes=("mahal",))
    i = int(np.argmax(fit.conformal_p))  # an ordinary row, not in a reference sample
    changed = x[i : i + 1].copy()
    changed[0, 1] = 6.0
    cf = counterfactual(ref, fit.delta[i], fit.axis_ranks[i], {1: 6.0})
    direct = score(ref, changed, members=True)  # a member with a value overridden
    assert cf.anomaly_score == pytest.approx(direct.anomaly_score[0])
    assert cf.conformal_p == pytest.approx(direct.conformal_p[0])
    assert cf.conformal_p < fit.conformal_p[i]


def test_counterfactual_keeps_an_absent_current_axis_absent():
    x, ev = _population(n=2_000, seed=19)
    ref, _ = fit_reference(x, ("a", "b", "c"), events=ev)
    fresh = score(ref, x[:5])  # rows without events: current rank is NaN
    cf = counterfactual(ref, fresh.delta[0], fresh.axis_ranks[0], {}, members=False)
    assert np.isnan(cf.axis_ranks[ref.axes.index("current")])
    assert cf.conformal_p == pytest.approx(fresh.conformal_p[0])


def test_reference_roundtrip_scores_identically():
    x, ev = _population(n=1_500, seed=9)
    ref, _ = fit_reference(x, ("a", "b", "c"), axes=AXES_ALL, events=ev)
    buf = io.BytesIO()
    np.savez(buf, **reference_to_arrays(ref))
    buf.seek(0)
    back = reference_from_arrays(dict(np.load(buf, allow_pickle=False)))
    a, b = score(ref, x, ev), score(back, x, ev)
    np.testing.assert_array_equal(a.anomaly_score, b.anomaly_score)
    np.testing.assert_array_equal(a.conformal_p, b.conformal_p)
    assert back.axes == ref.axes
    assert back.dims == ref.dims


def test_shapley_shares_survive_a_singular_covariance() -> None:
    # A constant or duplicated score dimension leaves the covariance only
    # positive semi-definite; the shares must still come out and sum to d2.
    cov = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    rows = np.array([[1.0, 1.0, 2.0], [0.5, 0.5, -1.0]])
    shares = robust._shapley_d2(rows, cov, robust.SHAPLEY_ORDERINGS)
    assert np.all(np.isfinite(shares))
    assert np.all(shares >= 0.0)
    d2 = np.einsum("ij,jk,ik->i", rows, np.linalg.pinv(cov), rows)
    assert shares.sum(axis=1) == pytest.approx(d2, rel=1e-6)
