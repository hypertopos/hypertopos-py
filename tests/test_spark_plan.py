"""Column roles -> per-entity aggregation plan on Spark (hypertopos.spark._plan).

The hand-computed fixture below is the specification of every dimension:
self-loops, null counterparties, null entities, null and negative amounts and
equal timestamps are all in it.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import types as T  # noqa: E402

from hypertopos.spark._plan import DIMS, aggregate, build_plan, describe_plan  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "spark_plan_full.txt"

SCHEMA = T.StructType(
    [
        T.StructField("src", T.StringType()),
        T.StructField("dst", T.StringType()),
        T.StructField("amt", T.DoubleType()),
        T.StructField("ts", T.TimestampType()),
    ]
)


def _frame(spark, rows, schema):
    # A DataFrame built from Python rows is re-read by every branch of the plan, and on
    # Windows each read spawns Python workers; caching keeps the tests at seconds.
    df = spark.createDataFrame(rows, schema).cache()
    df.count()
    return df


def _ts(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, minute, tzinfo=UTC)


# 2026-01-05 is a Monday; 01-10 / 01-11 are Saturday / Sunday.
ROWS = [
    ("A", "B", 100.0, _ts(5, 10)),
    ("A", "B", 200.0, _ts(5, 12)),
    ("A", "C", 300.0, _ts(6, 2)),
    ("B", "A", 50.0, _ts(10, 3)),
    ("B", "C", 150.0, _ts(10, 15)),
    ("C", "A", 400.0, _ts(11, 9)),
    ("A", "A", 10.0, _ts(7, 8)),  # self-loop: one out row only
    ("D", None, 70.0, _ts(8, 1)),  # null counterparty: out row only
    (None, "A", 90.0, _ts(8, 5)),  # null entity: dropped
    ("C", "B", -20.0, _ts(12, 18)),  # negative amount
    ("B", "A", None, _ts(10, 3)),  # null amount, same instant as row 4
    ("A", "D", 300.0, _ts(6, 2)),  # same instant as row 3
]

FULL_ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}
ALL_DIMS = [d.name for d in DIMS]


@pytest.fixture(scope="module")
def events(spark):
    return _frame(spark, ROWS, SCHEMA)


@pytest.fixture(scope="module")
def full(events):
    plan = build_plan(**FULL_ROLES, dims=ALL_DIMS, log_amounts=False)
    rows = aggregate(events, plan).collect()
    return {r["entity"]: r.asDict() for r in rows}


def test_one_row_per_entity_and_null_entity_dropped(full):
    assert sorted(full) == ["A", "B", "C", "D"]


def test_counts_for_entity_a(full):
    a = full["A"]
    assert a["n_out"] == 5
    assert a["n_in"] == 3  # the null-entity row does not reach A
    assert a["n_events"] == 8
    assert a["n_distinct_out"] == 4  # B, C, A (self), D
    assert a["n_distinct_in"] == 2  # B, C
    assert a["top_counterparty_share_out"] == pytest.approx(2 / 5)
    assert a["in_out_ratio"] == pytest.approx(3 / 8)


def test_amounts_for_entity_a(full):
    a = full["A"]
    out = [100.0, 200.0, 300.0, 10.0, 300.0]
    assert a["amount_sum_out"] == pytest.approx(910.0)
    assert a["amount_mean_out"] == pytest.approx(182.0)
    assert a["amount_std_out"] == pytest.approx(float(np.std(out)))  # population std
    assert a["amount_max_out"] == pytest.approx(300.0)
    assert a["amount_min_out"] == pytest.approx(10.0)
    assert a["amount_sum_in"] == pytest.approx(450.0)  # 50 + 400; the null amount is skipped
    assert a["amount_mean_in"] == pytest.approx(225.0)
    assert a["amount_std_in"] == pytest.approx(175.0)
    assert a["amount_max_in"] == pytest.approx(400.0)


def test_time_dims_for_entity_a(full):
    a = full["A"]
    gaps = [7200, 50400, 0, 108000, 241200, 0, 108000]
    assert a["active_days"] == 5
    assert a["span_days"] == pytest.approx(5 + 23 / 24)
    assert a["burst_max_daily"] == 2
    assert a["hour_share_night"] == pytest.approx(4 / 8)
    assert a["weekend_share"] == pytest.approx(3 / 8)
    assert a["iet_mean"] == pytest.approx(float(np.mean(gaps)))
    assert a["iet_std"] == pytest.approx(float(np.std(gaps)))


def test_single_out_event_and_null_counterparty_for_entity_d(full):
    d = full["D"]
    assert (d["n_out"], d["n_in"], d["n_events"]) == (1, 1, 2)
    assert d["n_distinct_out"] == 0
    assert d["top_counterparty_share_out"] == 0.0  # the only out row has no counterparty
    assert d["amount_std_out"] == 0.0  # a single event has no spread
    assert d["amount_mean_out"] == pytest.approx(70.0)
    assert d["iet_mean"] == pytest.approx(47 * 3600)
    assert d["iet_std"] == 0.0


def test_empty_side_fills_zero(spark):
    rows = [("X", "Y", 5.0, _ts(5, 10))]
    plan = build_plan(**FULL_ROLES, log_amounts=False)
    out = {r["entity"]: r.asDict() for r in aggregate(_frame(spark, rows, SCHEMA), plan).collect()}
    y = out["Y"]
    assert y["n_out"] == 0
    assert y["amount_mean_out"] == 0.0
    assert y["amount_min_out"] == 0.0
    assert y["top_counterparty_share_out"] == 0.0
    assert y["iet_mean"] == 0.0


def test_log_amounts_is_log1p_of_non_negative_part(events):
    # Heavy-tailed magnitudes (amounts, their spread, gaps in seconds) are log-scaled;
    # negatives count as 0 so the log stays defined.
    plan = build_plan(**FULL_ROLES)  # log_amounts defaults to True
    out = {r["entity"]: r.asDict() for r in aggregate(events, plan).collect()}
    a = out["A"]
    gaps = [7200, 50400, 0, 108000, 241200, 0, 108000]
    assert a["amount_sum_out"] == pytest.approx(math.log1p(910.0))
    assert out["C"]["amount_min_out"] == 0.0
    assert a["amount_std_out"] == pytest.approx(math.log1p(np.std([100, 200, 300, 10, 300])))
    assert a["iet_mean"] == pytest.approx(math.log1p(np.mean(gaps)))
    assert a["iet_std"] == pytest.approx(math.log1p(np.std(gaps)))
    assert a["span_days"] == pytest.approx(5 + 23 / 24)  # not log-scaled


def test_round_amount_share(full):
    # A's out amounts 100, 200, 300, 10, 300: four are positive multiples of 100.
    assert full["A"]["round_amount_share"] == pytest.approx(4 / 5)
    # C's out amounts 400 and -20.
    assert full["C"]["round_amount_share"] == pytest.approx(1 / 2)


def test_sample_sizes_of_shrunk_dims_travel_with_the_table(events):
    # n_events is not a dimension with a counterparty, but the shares and the
    # gap spread that are shrunk by it need it in the entity table.
    plan = build_plan(**FULL_ROLES)
    assert "n_events" not in [d.name for d in plan.dims]
    out = {r["entity"]: r.asDict() for r in aggregate(events, plan).collect()}
    assert out["A"]["n_events"] == 8
    assert {d.count for d in plan.dims if d.count} == {"n_out", "n_in", "n_events"}


def test_auto_leaves_out_n_events_when_a_counterparty_is_given():
    names = [d.name for d in build_plan(**FULL_ROLES).dims]
    assert "n_events" not in names
    assert "n_out" in names
    assert [d.name for d in build_plan(entity="src").dims] == ["n_events"]
    assert [d.name for d in build_plan(**FULL_ROLES, dims=["n_events"]).dims] == ["n_events"]


def test_entity_only_role(events):
    plan = build_plan(entity="src")
    rows = aggregate(events, plan).collect()
    assert rows[0].asDict().keys() == {"entity", "n_events"}
    by = {r["entity"]: r["n_events"] for r in rows}
    assert by == {"A": 5, "B": 3, "C": 2, "D": 1}


def test_dims_subset_and_order(events):
    plan = build_plan(**FULL_ROLES, dims=["amount_sum_out", "n_out"])
    rows = aggregate(events, plan)
    assert rows.columns == [
        "entity",
        "n_out",
        "amount_sum_out",
    ]  # catalogue order, not request order


def test_unknown_dim_is_rejected():
    with pytest.raises(ValueError, match="unknown dims.*nope"):
        build_plan(**FULL_ROLES, dims=["n_out", "nope"])


def test_dim_needing_a_missing_role_is_rejected():
    with pytest.raises(ValueError, match="amount_sum_out.*amount"):
        build_plan(entity="src", counterparty="dst", dims=["amount_sum_out"])


def test_extra_dims(events):
    plan = build_plan(
        **FULL_ROLES,
        dims=["n_out"],
        extra_dims={"big_out": "sum(CASE WHEN side = 'out' AND amount > 250 THEN 1 ELSE 0 END)"},
    )
    rows = {r["entity"]: r.asDict() for r in aggregate(events, plan).collect()}
    assert aggregate(events, plan).columns == ["entity", "n_out", "big_out"]
    assert rows["A"]["big_out"] == 2


def test_extra_dim_name_collision_is_rejected():
    with pytest.raises(ValueError, match="n_out"):
        build_plan(**FULL_ROLES, extra_dims={"n_out": "count(*)"})


@pytest.mark.parametrize(
    "name", ["first_event", "last_event", "dimension_values", "first_half_n", "key"]
)
def test_extra_dims_named_like_an_output_column_are_rejected(name):
    with pytest.raises(ValueError, match=name):
        build_plan(**FULL_ROLES, extra_dims={name: "count(1)"})


def test_string_time_is_rejected(spark):
    df = spark.createDataFrame(
        [("A", "B", 1.0, "2026-01-05 10:00:00")], ["src", "dst", "amt", "ts"]
    )
    with pytest.raises(TypeError, match="ts"):
        aggregate(df, build_plan(**FULL_ROLES))


def test_time_dims_do_not_depend_on_session_time_zone(spark, events):
    plan = build_plan(**FULL_ROLES, log_amounts=False)
    ntz_schema = T.StructType([*SCHEMA.fields[:3], T.StructField("ts", T.TimestampNTZType())])
    ntz_rows = [(s, d, a, t.replace(tzinfo=None)) for s, d, a, t in ROWS]
    epoch_rows = [(s, d, a, t.timestamp()) for s, d, a, t in ROWS]
    epoch_schema = T.StructType([*SCHEMA.fields[:3], T.StructField("ts", T.DoubleType())])
    previous = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    try:
        results = [
            {r["entity"]: r.asDict() for r in aggregate(frame, plan).collect()}
            for frame in (
                events,
                _frame(spark, ntz_rows, ntz_schema),
                _frame(spark, epoch_rows, epoch_schema),
            )
        ]
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous)
    keys = (
        "active_days",
        "span_days",
        "burst_max_daily",
        "hour_share_night",
        "weekend_share",
        "iet_mean",
    )
    for entity in "ABCD":
        expected = [results[0][entity][k] for k in keys]
        for other in results[1:]:
            assert [other[entity][k] for k in keys] == pytest.approx(expected), entity


def test_date_time_column(spark):
    schema = T.StructType([*SCHEMA.fields[:3], T.StructField("ts", T.DateType())])
    rows = [("A", "B", 1.0, date(2026, 1, 5)), ("A", "B", 1.0, date(2026, 1, 7))]
    plan = build_plan(**FULL_ROLES, log_amounts=False)
    out = aggregate(_frame(spark, rows, schema), plan).collect()
    a = next(r for r in out if r["entity"] == "A")
    assert a["active_days"] == 2
    assert a["iet_mean"] == pytest.approx(2 * 86400)


def test_plan_description_golden():
    text = describe_plan(build_plan(**FULL_ROLES))
    if os.environ.get("HYPERTOPOS_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(text, encoding="utf-8", newline="\n")
    assert text == GOLDEN.read_text(encoding="utf-8")


def test_random_events_match_a_pandas_oracle(spark):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    n = 1000
    keys = [f"k{i}" for i in range(40)]
    src = rng.choice(keys + [None], size=n, p=[0.98 / 40] * 40 + [0.02])
    dst = rng.choice(keys + [None], size=n, p=[0.97 / 40] * 40 + [0.03])
    amt = np.where(rng.random(n) < 0.05, np.nan, rng.normal(100, 80, n))
    ts = 1_767_225_600 + rng.integers(0, 30 * 86400, n)
    frame = pd.DataFrame({"src": src, "dst": dst, "amt": amt, "ts": ts.astype("float64")})
    schema = T.StructType([*SCHEMA.fields[:3], T.StructField("ts", T.DoubleType())])
    rows = [
        (
            None if s is None else str(s),
            None if d is None else str(d),
            None if math.isnan(a) else float(a),
            float(t),
        )
        for s, d, a, t in zip(src.tolist(), dst.tolist(), amt.tolist(), ts.tolist(), strict=True)
    ]
    got = {
        r["entity"]: r.asDict()
        for r in aggregate(
            _frame(spark, rows, schema), build_plan(**FULL_ROLES, log_amounts=False)
        ).collect()
    }

    valid = frame[frame["src"].notna()]
    out_rows = valid.assign(key=valid["src"], other=valid["dst"], side="out")
    inn = valid[valid["dst"].notna() & (valid["dst"] != valid["src"])]
    in_rows = inn.assign(key=inn["dst"], other=inn["src"], side="in")
    long = pd.concat([out_rows, in_rows], ignore_index=True)
    assert sorted(got) == sorted(long["key"].unique())
    for key, g in long.groupby("key"):
        o = g[g["side"] == "out"]
        i = g[g["side"] == "in"]
        row = got[key]
        assert row["n_out"] == len(o)
        assert row["n_in"] == len(i)
        assert row["n_distinct_out"] == o["other"].nunique()
        oa = o["amt"].dropna()
        assert row["amount_sum_out"] == pytest.approx(oa.sum() if len(oa) else 0.0)
        assert row["amount_std_out"] == pytest.approx(float(np.std(oa)) if len(oa) else 0.0)
        t = np.sort(g["ts"].to_numpy())
        gaps = np.diff(t)
        assert row["iet_mean"] == pytest.approx(float(gaps.mean()) if len(gaps) else 0.0)
        assert row["active_days"] == len(np.unique(np.floor(t / 86400)))
