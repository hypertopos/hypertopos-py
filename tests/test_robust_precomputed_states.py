# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The ``current`` axis from per-entity half states computed elsewhere (e.g. Spark executors)."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from hypertopos.engine.robust import (
    EventArrays,
    HalfStates,
    RobustScores,
    fit_reference,
    half_states,
    score,
)

DAY = 86400.0
DIMS = ("a", "b", "c")


def _population(n: int, seed: int) -> tuple[np.ndarray, EventArrays]:
    rng = np.random.default_rng(seed)
    m = 12 * n
    code = rng.integers(0, n, m)
    ts = 1.7e9 + rng.uniform(0, 30 * DAY, m)
    direction = rng.integers(0, 2, m).astype(np.int8)
    amount = rng.lognormal(4.0, 1.0, m)
    amount[::17] = 0.0
    x = np.column_stack([np.bincount(code, minlength=n), rng.normal(size=n), rng.normal(size=n)])
    return x.astype(np.float64), EventArrays(code, direction, ts, amount)


def _equal(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


def _assert_same(a: object, b: object) -> None:
    for f in fields(a):  # type: ignore[arg-type]
        assert _equal(getattr(a, f.name), getattr(b, f.name)), f.name


def test_precomputed_states_fit_the_same_reference_as_the_events() -> None:
    x, events = _population(400, seed=1)
    ref_events, fit_events = fit_reference(x, DIMS, events=events)
    states = half_states(events, len(x))
    ref_states, fit_states = fit_reference(x, DIMS, states=states)
    assert ref_events.axes == ("mahal", "current")
    _assert_same(ref_events, ref_states)
    _assert_same(fit_events, fit_states)


def test_half_states_split_at_the_middle_day_of_the_events() -> None:
    x, events = _population(50, seed=2)
    states = half_states(events, len(x))
    ref, _ = fit_reference(x, DIMS, events=events)
    assert (states.t0, states.t_mid) == (ref.t0, ref.t_mid)
    assert isinstance(states, HalfStates)
    assert states.first.shape == states.second.shape == (len(x), 5)


def test_new_rows_score_the_same_from_states_as_from_events() -> None:
    x, events = _population(400, seed=3)
    ref, _ = fit_reference(x, DIMS, events=events)
    x_new, new_events = _population(120, seed=4)
    assert ref.t_mid is not None
    states = half_states(new_events, len(x_new), t_mid=ref.t_mid)
    _assert_same(score(ref, x_new, new_events), score(ref, x_new, states=states))


def test_members_rescored_from_states_reproduce_the_fit() -> None:
    # Each sampled member was left out of its own velocity neighbours at fit time.
    x, events = _population(400, seed=5)
    states = half_states(events, len(x))
    ref, fitted = fit_reference(x, DIMS, states=states)
    assert fitted.sample_index is not None
    vel_pos = np.full(len(x), -1)
    vel_pos[fitted.sample_index] = np.arange(len(fitted.sample_index))
    again = score(ref, x, states=states, members=True, vel_pos=vel_pos)
    for name in ("anomaly_score", "conformal_p", "score_rank_pct", "axis_ranks"):
        assert np.array_equal(getattr(again, name), getattr(fitted, name), equal_nan=True), name
    assert isinstance(again, RobustScores)
