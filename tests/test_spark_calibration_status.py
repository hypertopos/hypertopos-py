# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""score(dimensions=True) and calibration_status: is a frozen calibration still valid?"""

from __future__ import annotations

import math
import pickle
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.datasets import synthetic_aml  # noqa: E402
from hypertopos.scoring import POOLED_GROUP, FrozenCalibration  # noqa: E402
from hypertopos.spark import (  # noqa: E402
    _status,  # noqa: E402
    calibration_status,
    find_anomalies,
    score,
    sphere_of,
)
from hypertopos.spark._api import CALIBRATION_FILE, ENTITIES_FILE, PLAN_FILE  # noqa: E402
from hypertopos.spark._drift_stats import flag_count_p  # noqa: E402
from hypertopos.spark._plan import window_days  # noqa: E402
from hypertopos.spark._status import window_matches  # noqa: E402

ROLES = {"entity": "from_account", "counterparty": "to_account", "amount": "amount", "time": "ts"}
DAY = 86_400


@pytest.fixture(scope="module")
def data():
    return synthetic_aml(n_accounts=1_000, seed=1)


@pytest.fixture(scope="module")
def epoch_seconds(data):
    return data.transactions["ts"].cast("int64").to_numpy() / 1_000_000


@pytest.fixture(scope="module")
def events(spark, data):
    tx = spark.createDataFrame(data.transactions.to_pandas())
    accounts = spark.createDataFrame(data.accounts.to_pandas()).select("account", "segment")
    return tx.join(accounts, tx.from_account == accounts.account).drop("account").cache()


@pytest.fixture(scope="module")
def result(events, tmp_path_factory):
    out = find_anomalies(events, **ROLES, sphere_root=str(tmp_path_factory.mktemp("s"))).cache()
    out.count()
    return out


@pytest.fixture(scope="module")
def scored(events, result):
    out = score(events, like=result, dimensions=True).cache()
    out.count()
    return out


def test_without_dimensions_the_result_is_unchanged(events, result, scored):
    plain = score(events, like=result)
    extra = ["dimension_values", "first_event", "last_event"]
    assert plain.columns == result.columns
    assert scored.columns == [*result.columns, *extra]
    assert plain.schema == scored.drop(*extra).schema
    by_entity = lambda rows: sorted(rows, key=lambda r: r["entity"])  # noqa: E731
    assert by_entity(plain.collect()) == by_entity(scored.drop(*extra).collect())


def test_first_and_last_event_are_epoch_seconds_per_entity(data, scored):
    tx = data.transactions
    t = tx["ts"].cast("int64").to_numpy() / 1_000_000
    keys = np.concatenate([tx["from_account"].to_numpy(), tx["to_account"].to_numpy()])
    times = pd.Series(np.concatenate([t, t])).groupby(keys)
    first, last = times.min().to_dict(), times.max().to_dict()
    rows = scored.select("entity", "first_event", "last_event").collect()
    assert len(rows) == len(first)
    for r in rows:
        assert r["first_event"] == pytest.approx(first[r["entity"]])
        assert r["last_event"] == pytest.approx(last[r["entity"]])


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        (0.0, 0.0, 1),
        (0.0, 3_600.0, 1),
        (0.0, DAY, 1),
        (0.0, DAY + 1, 2),
        (100.0, 100.0 + 29.5 * DAY, 30),
        (None, 5.0, None),
        (5.0, None, None),
    ],
)
def test_window_days_is_the_calendar_span_rounded_up(first, last, expected):
    assert window_days(first, last) == expected


@pytest.mark.parametrize(
    ("window", "matches"),
    [(60, True), (57, True), (56, False), (61, False), (30, False), (120, False)],
)
def test_a_scored_window_matches_at_95_to_100_percent_of_the_calibration(window, matches):
    assert window_matches(window, 60) is matches


def test_the_calibration_keeps_its_window_and_not_the_event_times(result, epoch_seconds):
    sphere = sphere_of(result)
    span = (epoch_seconds.max() - epoch_seconds.min()) / DAY
    assert sphere.calibration().window_days == math.ceil(span)
    assert not {"first_event", "last_event"} & set(sphere.entities().column_names)


def test_scoring_the_calibration_window_reports_both_windows(result, scored):
    status = calibration_status(scored, result)
    assert status["status"] == "ok", status
    assert status["calibration_window_days"] == sphere_of(result).calibration().window_days
    assert status["window_days"] == status["calibration_window_days"]
    assert "reason" not in status


def test_a_shorter_scored_window_is_a_window_mismatch(events, result, epoch_seconds):
    start = float(epoch_seconds.min())
    half = events.where(F.col("ts") < F.timestamp_seconds(F.lit(start + 30 * DAY)))
    status = calibration_status(score(half, like=result, dimensions=True), result)
    calibration = sphere_of(result).calibration().window_days
    kept = epoch_seconds[epoch_seconds < start + 30 * DAY]
    assert status["status"] == "window_mismatch", status
    assert status["window_days"] == math.ceil((kept.max() - kept.min()) / DAY)
    assert status["calibration_window_days"] == calibration
    assert f"{status['window_days']} days" in status["reason"]
    assert f"{calibration} days" in status["reason"]
    assert "same length" in status["reason"]
    # The statistics are still measured.
    assert status["p_distribution"] is not None
    assert set(status["p_dimensions"]) == set(sphere_of(result).calibration().dims)


def test_too_few_rows_wins_over_a_window_mismatch(events, result, epoch_seconds):
    start = float(epoch_seconds.min())
    day_one = events.where(F.col("ts") < F.timestamp_seconds(F.lit(start + DAY))).limit(50)
    status = calibration_status(score(day_one, like=result, dimensions=True), result)
    assert status["status"] == "too_few_rows"
    assert status["window_days"] == 1


def test_a_calibration_saved_without_a_window_still_scores(events, result, tmp_path):
    source = sphere_of(result).path
    target = tmp_path / source.name
    target.mkdir()
    for name in (ENTITIES_FILE, PLAN_FILE):
        shutil.copy2(source / name, target / name)
    old = sphere_of(result).calibration()
    vars(old).pop("window_days")  # a calibration pickled before the field existed
    with open(target / CALIBRATION_FILE, "wb") as fh:
        pickle.dump(old, fh)
    sphere = sphere_of(target)
    assert "window_days" not in vars(sphere.calibration())
    assert sphere.calibration().window_days is None
    status = calibration_status(score(events, like=sphere, dimensions=True), sphere)
    assert status["status"] == "ok", status
    assert status["calibration_window_days"] is None


def test_dimension_values_are_the_values_the_calibration_scores(result, scored):
    sphere = sphere_of(result)
    entities = sphere.entities()
    shapes = sphere.calibration().shapes(entities).astype(np.float64)
    expected = dict(zip(entities["entity"].to_pylist(), shapes.tolist(), strict=True))
    got = {r["entity"]: r["dimension_values"] for r in scored.collect()}
    assert got == expected  # exact: the sketch grid is compared value for value


def test_rows_from_the_fitted_population_are_ok(result, scored):
    # Rescored as new rows: a fresh synthetic draw is a different population at
    # this size (its in-degree structure differs measurably), the fitted one is not.
    status = calibration_status(scored, result)
    assert status["status"] == "ok", status
    assert status["rows"] == scored.count()
    assert status["flagged"] == scored.where("is_anomaly").count()
    assert status["expected_alarm_rate"] == pytest.approx(0.01)
    assert status["drifted_dimensions"] == []
    assert set(status["p_dimensions"]) == set(sphere_of(result).calibration().dims)
    assert 0.0 <= status["p_flag_count"] <= 1.0


def test_a_shifted_amount_is_drift_on_the_amount_dimensions(events, result):
    # +0.5 standard deviation of the log mean amount, applied to every amount.
    sd = float(np.std(sphere_of(result).entities()["amount_mean_out"].to_numpy()))
    shifted = events.withColumn("amount", F.col("amount") * float(np.exp(0.5 * sd)))
    status = calibration_status(score(shifted, like=result, dimensions=True), result)
    assert status["status"] == "drift", status
    assert status["drifted_dimensions"]
    # Scaling every amount moves the amount dimensions (and the share of round
    # amounts), never the counts or the timing.
    amount_dims = {d for d in status["p_dimensions"] if d.startswith("amount_")}
    assert set(status["drifted_dimensions"]) <= amount_dims | {"round_amount_share"}
    assert set(status["drifted_dimensions"]) & amount_dims
    assert all(status["p_dimensions"][d] < 0.025 / 2 for d in status["drifted_dimensions"])


def test_the_status_names_the_axes_of_the_scores_and_of_the_calibration(result, scored):
    # find_anomalies with time defaults to mahal + current; score uses mahal alone.
    status = calibration_status(scored, result)
    assert status["calibration_axes"] == ["mahal", "current"]
    assert status["score_axes"] == ["mahal"]


def test_too_few_rows(result, scored):
    status = calibration_status(scored.limit(50), result)
    assert status["status"] == "too_few_rows"
    assert status["rows"] == 50
    assert status["calibration_window_days"] == sphere_of(result).calibration().window_days


@pytest.fixture(scope="module")
def grouped_fits(events, tmp_path_factory):
    """``find_anomalies`` with ``group_by="segment"``, per ``score_per_group``, fitted once."""
    fits: dict[bool, object] = {}

    def get(score_per_group):
        if score_per_group not in fits:
            fits[score_per_group] = find_anomalies(
                events,
                **ROLES,
                group_by="segment",
                score_per_group=score_per_group,
                sphere_root=str(tmp_path_factory.mktemp("g")),
            )
        return fits[score_per_group]

    return get


@pytest.mark.parametrize("score_per_group", [True, False])
def test_grouped_calibrations_report_a_status_per_group(events, grouped_fits, score_per_group):
    grouped = grouped_fits(score_per_group)
    scored = score(events, like=grouped, dimensions=True).cache()
    status = calibration_status(scored, grouped)
    assert status["status"] == "ok", status
    assert status["rows"] == scored.count()
    table = status["groups"]
    assert set(table) == {"retail", "business"}
    assert sum(g["rows"] for g in table.values()) == status["rows"]
    assert sum(g["flagged"] for g in table.values()) == status["flagged"]
    scopes = {r["segment"]: r["calibration_scope"] for r in scored.collect()}
    for label, entry in table.items():
        assert entry["scope"] == scopes[label] == "group"
        assert entry["status"] == ("ok" if score_per_group else "untested")
    assert status["drifted_groups"] == ([] if score_per_group else None)
    assert set(status["p_dimensions"]) == set(sphere_of(grouped).calibration().dims)


@pytest.mark.parametrize(
    ("score_axes", "calibration_axes"),
    [(None, ["mahal"]), (("mahal", "current"), ["mahal", "current"])],
)
def test_score_per_group_status_names_the_axes(events, tmp_path, score_axes, calibration_axes):
    # With score_per_group the default is mahal alone; current is opt-in.
    grouped = find_anomalies(
        events,
        **ROLES,
        group_by="segment",
        score_per_group=True,
        score_axes=score_axes,
        sphere_root=str(tmp_path),
    )
    status = calibration_status(score(events, like=grouped, dimensions=True), grouped)
    assert status["calibration_axes"] == calibration_axes
    assert status["score_axes"] == ["mahal"]


def test_a_grouped_scored_result_without_the_group_column_is_refused(events, grouped_fits):
    grouped = grouped_fits(False)
    scored = score(events, like=grouped, dimensions=True).drop("segment")
    with pytest.raises(ValueError, match="segment"):
        calibration_status(scored, grouped)


def test_a_find_anomalies_result_is_refused(result):
    with pytest.raises(ValueError, match=r"score\(\.\.\., dimensions=True\)"):
        calibration_status(result, result)


def test_a_scored_result_without_the_event_times_is_refused(result, scored):
    with pytest.raises(ValueError, match=r"score\(\.\.\., dimensions=True\)"):
        calibration_status(scored.drop("first_event", "last_event"), result)


# Grouped calibrations, driven without Spark through the executor counter and the driver test.

GROUP_DIMS = tuple(f"d{i}" for i in range(4))
# (share, location, scale) per group: the groups differ from each other and from the population.
GROUP_SHAPES = {"A": (0.45, 0.0, 1.0), "B": (0.35, 0.5, 1.2), "C": (0.12, -0.4, 0.8)}
SMALL_GROUPS = {f"s{k}": (0.016, 0.8 + 0.1 * k, 1.0) for k in range(5)}


def _group_table(rng, n, layout, *, shift=0.0, shifted=None):
    names = list(layout)
    shares = np.array([layout[g][0] for g in names])
    seg = rng.choice(names, n, p=shares / shares.sum())
    y = rng.standard_normal((n, len(GROUP_DIMS)))
    for g, (_, loc, sd) in layout.items():
        m = seg == g
        y[m] = sd * y[m] + loc + (shift if g == shifted else 0.0)
    columns = {d: np.exp(y[:, i]) for i, d in enumerate(GROUP_DIMS)}
    return pa.table({**columns, "seg": pa.array(seg.tolist(), pa.string())})


def _scored_batch(calibration, table):
    scores = calibration.score(table)
    values = calibration.shapes(table).astype(np.float64)
    offsets = pa.array(np.arange(len(values) + 1, dtype=np.int32) * values.shape[1])
    return pa.record_batch(
        {
            "p_value": pa.array(scores.conformal_p.astype(np.float64)),
            "is_anomaly": pa.array(scores.is_anomaly),
            "dimension_values": pa.ListArray.from_arrays(offsets, pa.array(values.ravel())),
            "first_event": pa.array(np.zeros(len(values))),
            "last_event": pa.array(np.zeros(len(values))),
            "seg": table["seg"].combine_chunks(),
        }
    )


def _local_status(calibration, batch, partitions=1):
    units = _status.status_units(calibration)
    bounds = np.linspace(0, batch.num_rows, partitions + 1).astype(int)
    parts = [
        row
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        for out in _status.count_partition(units, [batch.slice(lo, hi - lo)])
        for row in out.to_pylist()
    ]
    return _status.status_from_counts(calibration, units, parts)


@pytest.fixture(scope="module")
def group_layout():
    return {**GROUP_SHAPES, **SMALL_GROUPS}


@pytest.fixture(scope="module")
def group_population(group_layout):
    return _group_table(np.random.default_rng(11), 5_000, group_layout)


@pytest.fixture(scope="module")
def per_group(group_population):
    return FrozenCalibration.fit(group_population, GROUP_DIMS, group_by="seg", score_per_group=True)


def test_group_by_alone_is_the_population_test_with_a_table_per_group(
    group_layout, group_population
):
    grouped = FrozenCalibration.fit(group_population, GROUP_DIMS, group_by="seg")
    plain = FrozenCalibration.fit(group_population, GROUP_DIMS)
    batch = _group_table(np.random.default_rng(12), 3_000, group_layout)
    got = _local_status(grouped, _scored_batch(grouped, batch))
    want = _local_status(plain, _scored_batch(plain, batch))
    for key in ("rows", "flagged", "p_distribution", "p_dimensions", "p_flag_count", "status"):
        assert got[key] == want[key], key
    assert got["status"] == "ok", got
    assert got["p_flag_count"] == flag_count_p(
        got["flagged"], got["rows"], grouped.reference.n_fitted, grouped.percentile
    )
    assert got["drifted_groups"] is None
    table = got["groups"]
    assert set(table) == {"A", "B", "C", POOLED_GROUP}
    assert sum(g["rows"] for g in table.values()) == got["rows"]
    assert sum(g["flagged"] for g in table.values()) == got["flagged"]
    assert table[POOLED_GROUP]["scope"] == "pooled"
    assert table["A"]["scope"] == "group"
    assert {g["status"] for g in table.values()} == {"untested"}
    assert all(g["p_distribution"] is None for g in table.values())
    assert "groups" not in want


def test_group_by_alone_tests_unseen_groups_like_the_population_test(
    group_layout, group_population
):
    # With group_by alone every row is scored against the population, so the status
    # is the status without groups, also for rows of a group the calibration never
    # saw (a renamed group here): the whole batch is tested.
    grouped = FrozenCalibration.fit(group_population, GROUP_DIMS, group_by="seg")
    plain = FrozenCalibration.fit(group_population, GROUP_DIMS)
    batch = _group_table(np.random.default_rng(20), 3_000, group_layout)
    renamed = pc.if_else(pc.equal(batch["seg"], "A"), "A-renamed", batch["seg"])
    batch = batch.set_column(batch.schema.get_field_index("seg"), "seg", renamed)
    got = _local_status(grouped, _scored_batch(grouped, batch))
    want = _local_status(plain, _scored_batch(plain, batch))
    for key in ("rows", "flagged", "p_distribution", "p_dimensions", "p_flag_count", "status"):
        assert got[key] == want[key], key
    assert got["groups"]["A-renamed"]["scope"] == "global"
    assert got["groups"]["A-renamed"]["status"] == "untested"


def test_group_by_alone_reads_a_shifted_new_segment_as_drift(group_layout, group_population):
    grouped = FrozenCalibration.fit(group_population, GROUP_DIMS, group_by="seg")
    layout = {**group_layout, "new": (0.3, 1.5, 1.0)}
    batch = _group_table(np.random.default_rng(21), 4_000, layout)
    status = _local_status(grouped, _scored_batch(grouped, batch))
    assert status["status"] == "drift", status
    assert status["groups"]["new"]["scope"] == "global"


def test_score_per_group_with_recoded_group_values_is_too_few_rows(group_layout, per_group):
    batch = _group_table(np.random.default_rng(22), 3_000, group_layout)
    recoded = pc.binary_join_element_wise(batch["seg"], "-recoded", "")
    batch = batch.set_column(batch.schema.get_field_index("seg"), "seg", recoded)
    status = _local_status(per_group, _scored_batch(per_group, batch))
    assert status["rows"] == 3_000
    assert status["status"] == "too_few_rows", status
    assert {g["scope"] for g in status["groups"].values()} == {"global"}
    assert {g["status"] for g in status["groups"].values()} == {"untested"}


def test_score_per_group_tests_each_reference_with_bonferroni_over_the_units(
    group_layout, per_group
):
    batch = _group_table(np.random.default_rng(13), 4_000, group_layout)
    status = _local_status(per_group, _scored_batch(per_group, batch))
    table = status["groups"]
    assert set(table) == {"A", "B", "C", POOLED_GROUP}
    assert table[POOLED_GROUP]["scope"] == "pooled"
    tested = [g for g in table.values() if g["status"] in ("ok", "drift")]
    k = len(tested)
    assert k == 4
    assert status["p_distribution"] == pytest.approx(
        min(1.0, k * min(g["p_distribution"] for g in tested))
    )
    for d in GROUP_DIMS:
        assert status["p_dimensions"][d] == pytest.approx(
            min(1.0, k * min(g["p_dimensions"][d] for g in tested))
        )
    assert status["p_flag_count"] is None
    assert all(0.0 <= g["p_flag_count"] <= 1.0 for g in tested)
    assert status["status"] == "ok", status
    assert status["drifted_groups"] == []


def test_score_per_group_finds_the_one_shifted_group(group_layout, per_group):
    batch = _group_table(np.random.default_rng(14), 4_000, group_layout, shift=0.5, shifted="B")
    status = _local_status(per_group, _scored_batch(per_group, batch))
    assert status["status"] == "drift", status
    assert status["drifted_groups"] == ["B"]
    assert status["groups"]["B"]["status"] == "drift"
    assert status["groups"]["B"]["drifted_dimensions"]
    assert set(status["drifted_dimensions"]) <= set(status["groups"]["B"]["drifted_dimensions"])


def test_rows_scored_by_the_population_are_counted_and_not_tested(group_layout, per_group):
    unseen = {**group_layout, "new": (0.05, 2.0, 1.0)}
    batch = _group_table(np.random.default_rng(15), 4_000, unseen)
    status = _local_status(per_group, _scored_batch(per_group, batch))
    new = status["groups"]["new"]
    assert new["scope"] == "global"
    assert new["status"] == "untested"
    assert new["p_distribution"] is None
    assert status["rows"] == 4_000
    assert sum(g["rows"] for g in status["groups"].values()) == 4_000
    assert status["status"] == "ok", status  # a group far from the population is not drift


def test_a_unit_under_the_minimum_rows_is_excluded(group_layout, per_group):
    few = {**group_layout, "C": (0.01, *group_layout["C"][1:])}
    batch = _group_table(np.random.default_rng(16), 4_000, few)
    status = _local_status(per_group, _scored_batch(per_group, batch))
    c = status["groups"]["C"]
    assert c["rows"] < _status.MIN_ROWS
    assert c["status"] == "too_few_rows"
    assert c["p_distribution"] is None
    tested = [g for g in status["groups"].values() if g["status"] in ("ok", "drift")]
    assert len(tested) == 3
    assert status["p_distribution"] == pytest.approx(
        min(1.0, len(tested) * min(g["p_distribution"] for g in tested))
    )


def test_no_tested_unit_is_too_few_rows(group_layout, per_group):
    batch = _group_table(np.random.default_rng(17), 180, group_layout)
    status = _local_status(per_group, _scored_batch(per_group, batch))
    assert status["rows"] == 180
    assert status["status"] == "too_few_rows"
    assert status["p_distribution"] is None
    assert {g["status"] for g in status["groups"].values()} == {"too_few_rows"}


@pytest.mark.parametrize("score_per_group", [True, False])
def test_counts_merge_over_partitions(group_layout, group_population, score_per_group):
    calibration = FrozenCalibration.fit(
        group_population, GROUP_DIMS, group_by="seg", score_per_group=score_per_group
    )
    table = _group_table(np.random.default_rng(18), 2_000, group_layout)
    batch = _scored_batch(calibration, table)
    assert _local_status(calibration, batch) == _local_status(calibration, batch, partitions=7)


def test_the_unit_scope_is_the_scope_that_scored_the_row(group_layout, per_group):
    table = _group_table(
        np.random.default_rng(19), 3_000, {**group_layout, "new": (0.05, 0.0, 1.0)}
    )
    scopes = per_group.score(table).calibration_scope
    status = _local_status(per_group, _scored_batch(per_group, table))
    seg = np.asarray(table["seg"].to_pylist(), dtype=object)
    labels = np.where(np.isin(seg, list(per_group.pooled)), POOLED_GROUP, seg)
    for label, entry in status["groups"].items():
        assert set(scopes[labels == label]) == {entry["scope"]}, label
