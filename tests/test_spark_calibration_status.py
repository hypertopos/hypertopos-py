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
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.datasets import synthetic_aml  # noqa: E402
from hypertopos.spark import calibration_status, find_anomalies, score, sphere_of  # noqa: E402
from hypertopos.spark._api import CALIBRATION_FILE, ENTITIES_FILE, PLAN_FILE  # noqa: E402
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


def test_too_few_rows(result, scored):
    status = calibration_status(scored.limit(50), result)
    assert status["status"] == "too_few_rows"
    assert status["rows"] == 50
    assert status["calibration_window_days"] == sphere_of(result).calibration().window_days


@pytest.mark.parametrize("score_per_group", [True, False])
def test_grouped_calibrations_are_unsupported(events, tmp_path_factory, score_per_group):
    grouped = find_anomalies(
        events,
        **ROLES,
        group_by="segment",
        score_per_group=score_per_group,
        sphere_root=str(tmp_path_factory.mktemp("g")),
    )
    status = calibration_status(score(events, like=grouped, dimensions=True), grouped)
    assert status["status"] == "unsupported"
    assert status["reason"]


def test_a_find_anomalies_result_is_refused(result):
    with pytest.raises(ValueError, match=r"score\(\.\.\., dimensions=True\)"):
        calibration_status(result, result)


def test_a_scored_result_without_the_event_times_is_refused(result, scored):
    with pytest.raises(ValueError, match=r"score\(\.\.\., dimensions=True\)"):
        calibration_status(scored.drop("first_event", "last_event"), result)
