# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Rows scored against a stored reference keep the alarm budget.

The reference is fitted on the population it calibrates, so a row the fit has
not seen scores higher than a member of the same distribution. New rows are
calibrated against out-of-fold scores of the population; members keep their
in-sample p-values.
"""

from __future__ import annotations

import numpy as np

from hypertopos.engine.robust import EventArrays, fit_reference, is_anomaly, score

DAY = 86_400.0
BUDGET_PCT = 99.0


def _correlated(rng: np.random.Generator, n: int, d: int, cov: np.ndarray) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(d), cov, size=n)


def _cov(d: int, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(d, d))
    c = a @ a.T + d * np.eye(d)
    s = np.sqrt(np.diag(c))
    return c / np.outer(s, s)


def _events(n: int, rng: np.random.Generator) -> EventArrays:
    code = np.repeat(np.arange(n), 4)
    ts = np.tile([0.0, DAY, 3 * DAY, 4 * DAY], n)
    direction = np.tile([0, 1, 0, 1], n).astype(np.int8)
    amount = np.repeat(rng.lognormal(size=n), 4) * rng.lognormal(sigma=0.3, size=4 * n)
    return EventArrays(code, direction, ts, amount)


def _alarm_rates(repeats: int, n: int, d: int, with_events: bool) -> np.ndarray:
    cov = _cov(d, seed=7)
    rates = []
    for r in range(repeats):
        rng = np.random.default_rng(1_000 + r)
        x = _correlated(rng, n, d, cov)
        events = _events(n, rng) if with_events else None
        ref, _ = fit_reference(x, tuple(f"d{i}" for i in range(d)), events=events)
        fresh = _correlated(rng, 2_000, d, cov)
        p = score(ref, fresh).conformal_p  # no events: as an ingested entity
        rates.append(is_anomaly(p, BUDGET_PCT).mean())
    return np.asarray(rates)


def _assert_within_budget(rates: np.ndarray) -> None:
    budget = 1.0 - BUDGET_PCT / 100.0
    se = rates.std(ddof=1) / np.sqrt(len(rates))
    assert rates.mean() <= budget + 2 * se, (rates.mean(), se)
    assert rates.mean() >= 0.5 * budget, rates.mean()


def test_new_rows_keep_the_alarm_budget_on_a_small_reference() -> None:
    _assert_within_budget(_alarm_rates(repeats=40, n=500, d=15, with_events=False))


def test_rows_without_events_keep_the_budget_of_an_event_reference() -> None:
    # The ingest path: fitted with the current axis, new rows carry no events.
    _assert_within_budget(_alarm_rates(repeats=30, n=500, d=10, with_events=True))


def test_members_rescored_as_members_reproduce_the_fit() -> None:
    rng = np.random.default_rng(3)
    x = _correlated(rng, 800, 6, _cov(6, seed=4))
    ref, fit = fit_reference(x, tuple(f"d{i}" for i in range(6)))
    again = score(ref, x, members=True)
    np.testing.assert_array_equal(again.conformal_p, fit.conformal_p)
    np.testing.assert_array_equal(again.score_rank_pct, fit.score_rank_pct)
