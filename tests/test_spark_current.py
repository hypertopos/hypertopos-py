# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The ``current`` axis on the Spark path: half states on the executors, same score as a sphere."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.builder.builder import EdgeTableConfig, GDSBuilder, RelationSpec  # noqa: E402
from hypertopos.engine.robust import EventArrays, half_states, halves  # noqa: E402
from hypertopos.scoring import POOLED_GROUP, STATE_COLUMNS  # noqa: E402
from hypertopos.spark import find_anomalies, score, sphere_of  # noqa: E402
from hypertopos.spark._api import _score_axes  # noqa: E402
from hypertopos.spark._build import LINE_ID, PATTERN_ID  # noqa: E402
from hypertopos.spark._plan import aggregate, build_plan  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}
SCHEMA = "src string, dst string, amt double, ts timestamp"
START = datetime(2026, 1, 1, tzinfo=UTC)


def _events(n_entities: int = 60, n_events: int = 1500, seed: int = 7) -> list[tuple]:
    # Self-transfers included: a build counts them as an out and an in event.
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_events):
        src = f"E{int(rng.integers(n_entities)):03d}"
        dst = src if i % 40 == 0 else f"E{int(rng.integers(n_entities)):03d}"
        amt = 0.0 if i % 53 == 0 else float(np.round(rng.lognormal(5, 1.2), 2))
        ts = START + timedelta(seconds=int(rng.integers(0, 45 * 86400)))
        rows.append((src, dst, amt, ts))
    return rows


@pytest.fixture(scope="module")
def rows() -> list[tuple]:
    return _events()


@pytest.fixture(scope="module")
def events(spark, rows):
    df = spark.createDataFrame(rows, SCHEMA).cache()
    df.count()
    return df


AXES = ("mahal", "current")


@pytest.fixture(scope="module")
def result(events, tmp_path_factory):
    root = tmp_path_factory.mktemp("current")
    return find_anomalies(
        events, **ROLES, score_axes=AXES, sphere_root=str(root), cache=False
    ).cache()


@pytest.fixture(scope="module")
def grouped(events, tmp_path_factory):
    # Groups set the deviations only: the score and its velocity field stay population-wide.
    segment = F.when(F.col("src") < "E030", "low").otherwise("high")
    root = tmp_path_factory.mktemp("current_grouped")
    return find_anomalies(
        events.withColumn("segment", segment),
        **ROLES,
        group_by="segment",
        min_group_size=10,
        score_axes=AXES,
        sphere_root=str(root),
        cache=False,
    ).cache()


@pytest.fixture(scope="module")
def wide_rows() -> list[tuple]:
    return _events(n_entities=480, n_events=12000, seed=8)


@pytest.fixture(scope="module")
def wide_events(spark, wide_rows):
    # Two groups above the reference size, one under it (scored against the population).
    segment = (
        F.when(F.col("src") < "E240", "low").when(F.col("src") < "E450", "high").otherwise("tiny")
    )
    df = spark.createDataFrame(wide_rows, SCHEMA).withColumn("segment", segment).cache()
    df.count()
    return df


@pytest.fixture(scope="module")
def per_group(wide_events, tmp_path_factory):
    # Each group's own reference, velocity field included.
    root = tmp_path_factory.mktemp("current_per_group")
    return find_anomalies(
        wide_events,
        **ROLES,
        group_by="segment",
        score_per_group=True,
        score_axes=AXES,
        sphere_root=str(root),
        cache=False,
    ).cache()


def _event_arrays(rows: list[tuple], keys: list[str]) -> EventArrays:
    """The event rows a build's ``current`` axis reads: one out row per event, one in row."""
    index = {k: i for i, k in enumerate(keys)}
    code, direction, ts, amount = [], [], [], []
    for src, dst, amt, when in rows:
        for d, key in ((0, src), (1, dst)):
            code.append(index[key])
            direction.append(d)
            ts.append(when.timestamp())
            amount.append(amt)
    return EventArrays(
        np.array(code), np.array(direction, dtype=np.int8), np.array(ts), np.array(amount)
    )


def test_state_columns_are_the_half_states_of_a_build(events, rows):
    epochs = [r[3].timestamp() for r in rows]
    t0, t_mid = halves(min(epochs), max(epochs))
    table = aggregate(events, build_plan(**ROLES), t_mid=t_mid).toPandas().sort_values("entity")
    keys = table["entity"].tolist()
    expected = half_states(_event_arrays(rows, keys), len(keys))
    assert (expected.t0, expected.t_mid) == (t0, t_mid)
    raw = table[list(STATE_COLUMNS)].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(np.log1p(raw[:, :5]), expected.first, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.log1p(raw[:, 5:]), expected.second, rtol=1e-12, atol=1e-12)


def _built_from_events(
    entities: pa.Table,
    rows: list[tuple],
    dims: list[str],
    out: Path,
    group_by: str | None = None,
    score_per_group: bool = False,
) -> dict:
    events = pa.table(
        {
            "event_id": [f"T{i:05d}" for i in range(len(rows))],
            "src": [r[0] for r in rows],
            "dst": [r[1] for r in rows],
            "amt": [r[2] for r in rows],
            "ts": pa.array([r[3] for r in rows], pa.timestamp("us", tz="UTC")),
        }
    )
    b = GDSBuilder("contract", str(out))
    b.add_line(LINE_ID, entities, key_col="entity", source_id="t")
    b.add_line("events", events, key_col="event_id", source_id="t", role="event")
    b.add_pattern(
        "event_pattern",
        "event",
        "events",
        relations=[
            RelationSpec(LINE_ID, fk_col="src", direction="in", required=True),
            RelationSpec(LINE_ID, fk_col="dst", direction="in", required=False),
        ],
        edge_table=EdgeTableConfig(
            from_col="src", to_col="dst", timestamp_col="ts", amount_col="amt"
        ),
    )
    b.add_pattern(
        PATTERN_ID,
        "anchor",
        LINE_ID,
        relations=[],
        anomaly_percentile=99.0,
        group_by_property=group_by,
        metric_properties=dims,
        score_axes=["mahal", "current"],
        score_per_group=score_per_group,
    )
    b.build()
    return _geometry(out)


def _geometry(out: Path) -> dict:
    t = lance.dataset(str(out / "geometry" / PATTERN_ID / "data.lance")).to_table(
        columns=["primary_key", "anomaly_score", "conformal_p", "is_anomaly"]
    )
    return {r["primary_key"]: r for r in t.to_pylist()}


def _assert_same_scores(result, geometry: dict) -> None:
    got = {r["entity"]: r for r in result.collect()}
    assert got.keys() == geometry.keys()
    for key, row in got.items():
        assert row["anomaly_score"] == pytest.approx(geometry[key]["anomaly_score"], abs=1e-6), key
        assert row["p_value"] == pytest.approx(geometry[key]["conformal_p"], abs=1e-6), key
        assert row["is_anomaly"] == geometry[key]["is_anomaly"], key


@pytest.mark.parametrize(
    ("which", "events_of"), [("result", "rows"), ("grouped", "rows"), ("per_group", "wide_rows")]
)
def test_find_anomalies_scores_as_a_sphere_built_from_the_events(
    which, events_of, request, tmp_path
):
    result = request.getfixturevalue(which)
    rows = request.getfixturevalue(events_of)
    sphere = sphere_of(result)
    calibration = sphere.calibration()
    assert calibration.reference.axes == ("mahal", "current")
    entities = sphere.entities()
    shapes = calibration.shapes(entities)
    for j, dim in enumerate(calibration.dims):
        idx = entities.schema.get_field_index(dim)
        entities = entities.set_column(idx, dim, pa.array(shapes[:, j].astype("float64")))
    group_by = None
    if calibration.group_by is not None:
        group_by = "calibration_group"
        raw = entities[calibration.group_by].to_pylist()
        pooled = [POOLED_GROUP if g in calibration.pooled else g for g in raw]
        entities = entities.append_column(group_by, pa.array(pooled, pa.string()))
    geometry = _built_from_events(
        entities,
        rows,
        list(calibration.dims),
        tmp_path / "s",
        group_by=group_by,
        score_per_group=calibration.score_per_group,
    )
    _assert_same_scores(result, geometry)


@pytest.mark.parametrize("which", ["result", "grouped", "per_group"])
def test_the_navigable_sphere_reports_the_same_scores(which, request):
    result = request.getfixturevalue(which)
    _assert_same_scores(result, _geometry(sphere_of(result).lance_path()))


def test_new_events_are_scored_without_current(events, result):
    # A scored batch carries no half states: mahal alone, against the population's
    # scores without current, so it keeps the alarm budget.
    sphere = sphere_of(result)
    calibration = sphere.calibration()
    entities = sphere.entities().drop_columns(list(STATE_COLUMNS))
    expected = calibration.score(entities)
    want = dict(zip(entities["entity"].to_pylist(), expected.conformal_p.tolist(), strict=True))
    got = {r["entity"]: r["p_value"] for r in score(events, like=result).collect()}
    assert got.keys() == want.keys()
    for key, p in got.items():
        assert p == pytest.approx(want[key], abs=1e-9), key


def test_a_fresh_fit_is_deterministic(events, result, tmp_path):
    again = find_anomalies(events, **ROLES, score_axes=AXES, sphere_root=str(tmp_path), cache=False)
    first = {r["entity"]: r["p_value"] for r in result.collect()}
    assert {r["entity"]: r["p_value"] for r in again.collect()} == first


def test_score_axes_mahal_leaves_current_out(events, result, tmp_path):
    out = find_anomalies(events, **ROLES, score_axes=("mahal",), sphere_root=str(tmp_path))
    calibration = sphere_of(out).calibration()
    assert calibration.reference.axes == ("mahal",)
    assert not set(STATE_COLUMNS) & set(sphere_of(out).entities().column_names)
    assert sphere_of(out).fingerprint != sphere_of(result).fingerprint


def test_current_needs_time(events, tmp_path):
    no_time = {k: v for k, v in ROLES.items() if k != "time"}
    with pytest.raises(ValueError, match="time"):
        find_anomalies(events, **no_time, score_axes=("mahal", "current"))
    with pytest.raises(ValueError, match="score_axes"):
        find_anomalies(events, **ROLES, score_axes=("mahal", "knn"))
    out = find_anomalies(events, **ROLES, score_axes=("mahal",), sphere_root=str(tmp_path))
    assert sphere_of(out).calibration().reference.axes == ("mahal",)
    keys = sphere_of(out).entities()["entity"]
    assert pc.all(pc.greater_equal(keys.slice(1), keys.slice(0, len(keys) - 1))).as_py()


@pytest.mark.parametrize(
    ("score_axes", "time", "score_per_group", "expected"),
    [
        (None, "ts", False, ("mahal", "current")),
        (None, None, False, ("mahal",)),
        (None, "ts", True, ("mahal",)),
        (("mahal",), "ts", False, ("mahal",)),
        (("mahal", "current"), "ts", False, ("mahal", "current")),
        (("mahal", "current"), "ts", True, ("mahal", "current")),
        (("mahal",), "ts", True, ("mahal",)),
    ],
)
def test_default_axes_add_current_when_time_is_given(score_axes, time, score_per_group, expected):
    assert _score_axes(score_axes, time, score_per_group) == expected


def test_a_bare_string_score_axes_asks_for_a_tuple():
    with pytest.raises(TypeError, match=r"\('mahal',\)"):
        _score_axes("mahal", "ts", False)


def test_the_default_with_time_is_the_current_calibration(events, result, tmp_path):
    # The axes resolve before the fingerprint: the default reuses a calibration made
    # with explicit (mahal, current), never one made with mahal alone.
    out = find_anomalies(events, **ROLES, sphere_root=str(tmp_path), cache=False)
    assert sphere_of(out).calibration().reference.axes == ("mahal", "current")
    assert sphere_of(out).fingerprint == sphere_of(result).fingerprint
    mahal = find_anomalies(events, **ROLES, score_axes=("mahal",), sphere_root=str(tmp_path))
    assert sphere_of(mahal).fingerprint != sphere_of(out).fingerprint


def test_score_per_group_fits_current_in_every_own_group(per_group):
    sphere = sphere_of(per_group)
    calibration = sphere.calibration()
    assert calibration.groups is not None
    assert calibration.reference.axes == AXES
    for gid in ("low", "high"):
        ref = calibration.groups[gid].reference
        assert ref is not None and ref.axes == AXES
    assert calibration.groups.get("tiny") is None or calibration.groups["tiny"].reference is None
    got = {r["entity"]: r for r in per_group.collect()}
    assert {got[f"E{i}"]["calibration_scope"] for i in range(450, 480)} == {"global"}
    assert {got["E000"]["calibration_scope"], got["E300"]["calibration_scope"]} == {"group"}
    # The saved calibration rescores its members to the fit's own scores (explain, cache).
    geometry = sphere.geometry()
    p = dict(zip(geometry["entity"].to_pylist(), geometry["p_value"].to_pylist(), strict=True))
    assert p == {k: r["p_value"] for k, r in got.items()}


def test_score_per_group_fingerprint_carries_the_axes(wide_events, per_group, tmp_path):
    mahal = find_anomalies(
        wide_events,
        **ROLES,
        group_by="segment",
        score_per_group=True,
        score_axes=("mahal",),
        sphere_root=str(tmp_path),
    )
    assert sphere_of(mahal).calibration().reference.axes == ("mahal",)
    assert sphere_of(mahal).fingerprint != sphere_of(per_group).fingerprint


def test_a_later_window_is_scored_per_group_without_current(wide_events, per_group):
    sphere = sphere_of(per_group)
    calibration = sphere.calibration()
    entities = sphere.entities().drop_columns(list(STATE_COLUMNS))
    expected = calibration.score(entities)
    want = dict(zip(entities["entity"].to_pylist(), expected.conformal_p.tolist(), strict=True))
    got = {r["entity"]: r["p_value"] for r in score(wide_events, like=per_group).collect()}
    assert got.keys() == want.keys()
    for key, p in got.items():
        assert p == pytest.approx(want[key], abs=1e-9), key
