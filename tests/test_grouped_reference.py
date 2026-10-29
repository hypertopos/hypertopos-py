"""Per-group calibrated references: each group's evidence score against its own population."""

from __future__ import annotations

import numpy as np
import pytest

from hypertopos.engine.robust import (
    MAX_SCORE_GROUPS,
    POOLED_GROUP,
    EventArrays,
    fit_grouped_reference,
    fit_reference,
    grouped_reference_from_arrays,
    grouped_reference_to_arrays,
    is_anomaly,
    min_reference_size,
    pool_groups,
    score,
    subset_events,
)

DIMS = ("a", "b", "c")


def _population(sizes: dict[str, int], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ids = [], []
    for k, (gid, n) in enumerate(sizes.items()):
        xs.append(rng.lognormal(mean=k, sigma=0.5 + 0.2 * k, size=(n, len(DIMS))))
        ids.extend([gid] * n)
    return np.vstack(xs), np.array(ids, dtype=object)


def test_min_reference_size_at_percentile() -> None:
    assert min_reference_size(99.0) == 199
    assert min_reference_size(95.0) == 39
    # In-sample, a member's smallest p is 2/(n+1): one short of the bound never flags.
    assert not is_anomaly(np.array([2 / 199]), 99.0)[0]
    assert is_anomaly(np.array([2 / 200]), 99.0)[0]


def test_pool_groups_folds_groups_below_the_bound() -> None:
    ids = np.array(["big"] * 300 + ["mid"] * 199 + ["tiny"] * 50, dtype=object)
    pooled_ids, pooled = pool_groups(ids, 99.0)
    assert pooled == frozenset({"tiny"})
    assert set(pooled_ids[-50:]) == {POOLED_GROUP}
    assert set(pooled_ids[:499]) == {"big", "mid"}


def test_pool_groups_rejects_a_min_group_size_below_the_bound() -> None:
    ids = np.array(["g"] * 300, dtype=object)
    with pytest.raises(ValueError, match="199"):
        pool_groups(ids, 99.0, min_group_size=100)
    _, pooled = pool_groups(ids, 99.0, min_group_size=400)
    assert pooled == frozenset({"g"})


def test_grouped_reference_matches_per_group_fit() -> None:
    x, ids = _population({"retail": 900, "business": 400})
    grouped, scores, keys = fit_grouped_reference(x, DIMS, ids, percentile=99.0, axes=("mahal",))
    assert set(grouped.groups) == {"retail", "business"}
    for gid in ("retail", "business"):
        m = ids == gid
        _, own = fit_reference(x[m], DIMS, axes=("mahal",))
        np.testing.assert_array_equal(scores.conformal_p[m], own.conformal_p)
        np.testing.assert_array_equal(scores.anomaly_score[m], own.anomaly_score)
        assert set(keys[m]) == {gid}


def test_pool_at_bound_gets_its_own_reference() -> None:
    x, ids = _population({"big": 600, **{f"t{i}": 50 for i in range(4)}})
    grouped, scores, keys = fit_grouped_reference(x, DIMS, ids, percentile=99.0, axes=("mahal",))
    assert POOLED_GROUP in grouped.groups
    assert grouped.pooled == frozenset({"t0", "t1", "t2", "t3"})
    m = ids != "big"
    _, own = fit_reference(x[m], DIMS, axes=("mahal",))
    np.testing.assert_array_equal(scores.conformal_p[m], own.conformal_p)
    assert set(keys[m]) == {POOLED_GROUP}
    assert grouped.resolve("t2")[1:] == (POOLED_GROUP, "pooled")


def test_pool_below_bound_is_scored_with_the_population() -> None:
    x, ids = _population({"big": 600, "t0": 50, "t1": 50})
    grouped, scores, keys = fit_grouped_reference(x, DIMS, ids, percentile=99.0, axes=("mahal",))
    assert POOLED_GROUP not in grouped.groups
    m = ids != "big"
    assert all(k is None for k in keys[m])
    _, population = fit_reference(x, DIMS, axes=("mahal",))
    np.testing.assert_array_equal(scores.conformal_p[m], population.conformal_p[m])
    assert grouped.resolve("t0")[1:] == (None, "global")


def test_unseen_group_resolves_to_the_population() -> None:
    x, ids = _population({"retail": 900, "business": 400})
    grouped, _, _ = fit_grouped_reference(x, DIMS, ids, percentile=99.0, axes=("mahal",))
    ref, key, scope = grouped.resolve("never-seen")
    assert ref is grouped.population
    assert (key, scope) == (None, "global")
    assert grouped.resolve("retail")[1:] == ("retail", "group")


def test_each_group_gets_its_own_out_of_fold_calibration() -> None:
    x, ids = _population({"small": 300, "huge": 7000})
    x1 = x[:, :1]
    grouped, _, _ = fit_grouped_reference(x1, ("a",), ids, percentile=99.0, axes=("mahal",))
    # One dimension: the population (7300 rows) is above the out-of-fold cut-off,
    # the small group is not.
    assert grouped.population.mahal_sorted_oof is None
    assert grouped.groups["small"].mahal_sorted_oof is not None


def test_events_are_split_per_group() -> None:
    x, ids = _population({"retail": 300, "business": 250})
    rng = np.random.default_rng(1)
    m = 4000
    events = EventArrays(
        code=rng.integers(0, len(x), m).astype(np.int64),
        direction=rng.integers(0, 2, m).astype(np.int8),
        ts=np.sort(rng.uniform(0, 86400 * 60, m)),
        amount=rng.lognormal(3, 1, m),
    )
    grouped, scores, _ = fit_grouped_reference(
        x, DIMS, ids, percentile=99.0, axes=("mahal", "current"), events=events
    )
    mask = ids == "business"
    _, own = fit_reference(
        x[mask], DIMS, axes=("mahal", "current"), events=subset_events(events, mask)
    )
    np.testing.assert_array_equal(scores.conformal_p[mask], own.conformal_p)
    assert grouped.groups["business"].axes == ("mahal", "current")


def test_grouped_arrays_round_trip() -> None:
    x, ids = _population({"a/b": 400, "C:\\x": 300, "t0": 50, "t1": 50, "t2": 50, "t3": 50})
    grouped, _, _ = fit_grouped_reference(x, DIMS, ids, percentile=99.0, axes=("mahal",))
    back = grouped_reference_from_arrays(grouped_reference_to_arrays(grouped))
    assert set(back.groups) == set(grouped.groups) == {"a/b", "C:\\x", POOLED_GROUP}
    assert back.pooled == grouped.pooled
    for gid, ref in grouped.groups.items():
        np.testing.assert_array_equal(back.groups[gid].cal_sorted, ref.cal_sorted)
    np.testing.assert_array_equal(back.population.cal_sorted, grouped.population.cal_sorted)
    probe = x[:20]
    np.testing.assert_array_equal(
        score(back.resolve("a/b")[0], probe).conformal_p,
        score(grouped.resolve("a/b")[0], probe).conformal_p,
    )


def test_too_many_groups_raise() -> None:
    n = MAX_SCORE_GROUPS + 1
    ids = np.repeat(np.array([f"g{i}" for i in range(n)], dtype=object), 199)
    x = np.random.default_rng(0).normal(size=(len(ids), 2))
    with pytest.raises(ValueError, match="coarser"):
        fit_grouped_reference(x, ("a", "b"), ids, percentile=99.0, axes=("mahal",))


def test_a_group_without_events_keeps_the_current_axis_neutral() -> None:
    x, ids = _population({"active": 300, "dormant": 300})
    rng = np.random.default_rng(2)
    m = 3000
    events = EventArrays(
        code=rng.integers(0, 300, m).astype(np.int64),
        direction=rng.integers(0, 2, m).astype(np.int8),
        ts=np.sort(rng.uniform(0, 86400 * 60, m)),
        amount=rng.lognormal(3, 1, m),
    )
    grouped, scores, _ = fit_grouped_reference(
        x, DIMS, ids, percentile=95.0, axes=("mahal", "current"), events=events
    )
    assert grouped.groups["dormant"].axes == ("mahal", "current")
    dormant = ids == "dormant"
    np.testing.assert_array_equal(scores.axis_ranks[dormant, 1], 0.5)
    assert np.isfinite(scores.conformal_p).all()


def test_a_group_named_like_the_pool_is_refused() -> None:
    ids = np.array(["A"] * 100 + [POOLED_GROUP] * 100 + ["C"] * 5, dtype=object)
    with pytest.raises(ValueError, match="small groups"):
        pool_groups(ids, 95.0)
    # Labels pooled once already pass through unchanged.
    pooled_ids, _ = pool_groups(np.array(["A"] * 100 + ["C"] * 5, dtype=object), 95.0)
    again, pooled = pool_groups(pooled_ids, 95.0)
    np.testing.assert_array_equal(again, pooled_ids)
    assert pooled == frozenset({POOLED_GROUP})


def test_stored_labels_merge_a_shrunk_group_into_the_existing_pool() -> None:
    # Labels read back from a sphere may already hold the pool; a group that an
    # exclusion shrank below the size joins it instead of raising.
    ids = np.array(["A"] * 100 + [POOLED_GROUP] * 10 + ["B"] * 38, dtype=object)
    labels, pooled = pool_groups(ids, 95.0, stored_labels=True)
    assert set(labels[-48:]) == {POOLED_GROUP}
    assert pooled == frozenset({POOLED_GROUP, "B"})
    x = np.random.default_rng(0).normal(size=(len(ids), 3))
    grouped, _, _ = fit_grouped_reference(
        x, DIMS, ids, percentile=95.0, axes=("mahal",), stored_labels=True
    )
    assert set(grouped.groups) == {"A", POOLED_GROUP}
