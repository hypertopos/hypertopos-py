# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Column roles -> per-entity aggregation plan, compiled to Spark.

The dimension catalogue is data (``DIMS``); ``describe_plan`` renders it as
text, which is what the plan golden locks, so the meaning of every dimension
cannot change silently with the Spark version or the compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from hypertopos.scoring import SHRINK_WEIGHT

Kind = Literal["poisson", "gaussian"]
Stage = Literal["events", "pairs", "days", "gaps"]

_ROLE_ORDER = ("entity", "counterparty", "amount", "time")
_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class DimSpec:
    """One entity dimension: what it needs, where it is computed and what it means."""

    name: str
    kind: Kind
    requires: tuple[str, ...]
    stage: Stage
    definition: str
    # The dimension holding this one's sample size, for shares, spreads and ratios:
    # their value at one or two events is noise, so it is shrunk toward the population.
    count: str | None = None


_E: tuple[str, ...] = ("entity",)
_EC: tuple[str, ...] = ("entity", "counterparty")
_EA: tuple[str, ...] = ("entity", "amount")
_ECA: tuple[str, ...] = ("entity", "counterparty", "amount")
_ET: tuple[str, ...] = ("entity", "time")

DIMS: tuple[DimSpec, ...] = (
    DimSpec("n_events", "poisson", _E, "events", "rows of the entity in the long table (out + in)"),
    DimSpec("n_out", "poisson", _EC, "events", "out rows"),
    DimSpec("n_in", "poisson", _EC, "events", "in rows"),
    DimSpec(
        "n_distinct_out", "poisson", _EC, "events", "distinct non-null counterparties of out rows"
    ),
    DimSpec("n_distinct_in", "poisson", _EC, "events", "distinct entities of in rows"),
    DimSpec(
        "top_counterparty_share_out",
        "gaussian",
        _EC,
        "pairs",
        "largest out-row count to one non-null counterparty / n_out; 0 when n_out = 0",
        count="n_out",
    ),
    DimSpec("in_out_ratio", "gaussian", _EC, "events", "n_in / n_events", count="n_events"),
    DimSpec(
        "amount_sum_out",
        "gaussian",
        _ECA,
        "events",
        "sum of non-null out amounts; 0 when none; log",
    ),
    DimSpec(
        "amount_sum_in", "gaussian", _ECA, "events", "sum of non-null in amounts; 0 when none; log"
    ),
    DimSpec(
        "amount_mean_out",
        "gaussian",
        _ECA,
        "events",
        "mean of non-null out amounts; 0 when none; log",
    ),
    DimSpec(
        "amount_mean_in",
        "gaussian",
        _ECA,
        "events",
        "mean of non-null in amounts; 0 when none; log",
    ),
    DimSpec(
        "amount_std_out",
        "gaussian",
        _ECA,
        "events",
        "population std (ddof 0) of out amounts; 0 when < 2; log",
        count="n_out",
    ),
    DimSpec(
        "amount_std_in",
        "gaussian",
        _ECA,
        "events",
        "population std (ddof 0) of in amounts; 0 when < 2; log",
        count="n_in",
    ),
    DimSpec(
        "amount_max_out",
        "gaussian",
        _ECA,
        "events",
        "max of non-null out amounts; 0 when none; log",
    ),
    DimSpec(
        "amount_max_in", "gaussian", _ECA, "events", "max of non-null in amounts; 0 when none; log"
    ),
    DimSpec(
        "amount_min_out",
        "gaussian",
        _ECA,
        "events",
        "min of non-null out amounts; 0 when none; log",
    ),
    DimSpec(
        "round_amount_share",
        "gaussian",
        _ECA,
        "events",
        "share of non-null out amounts that are a positive multiple of 100; 0 when none",
        count="n_out",
    ),
    DimSpec("active_days", "poisson", _ET, "events", "distinct UTC days with an event"),
    DimSpec("span_days", "gaussian", _ET, "events", "(last - first event time) in days"),
    DimSpec(
        "iet_mean",
        "gaussian",
        _ET,
        "gaps",
        "mean gap in seconds between consecutive events; 0 when < 2; log",
    ),
    DimSpec(
        "iet_std",
        "gaussian",
        _ET,
        "gaps",
        "population std (ddof 0) of the gaps in seconds; 0 when < 2 gaps; log",
        count="n_events",
    ),
    DimSpec("burst_max_daily", "poisson", _ET, "days", "largest number of events on one UTC day"),
    DimSpec(
        "hour_share_night",
        "gaussian",
        _ET,
        "events",
        "share of events with UTC hour in [0, 6)",
        count="n_events",
    ),
    DimSpec(
        "weekend_share",
        "gaussian",
        _ET,
        "events",
        "share of events on a UTC Saturday or Sunday",
        count="n_events",
    ),
)

_BY_NAME = {d.name: d for d in DIMS}
_LOG_DIMS = frozenset(
    {
        "amount_sum_out",
        "amount_sum_in",
        "amount_mean_out",
        "amount_mean_in",
        "amount_max_out",
        "amount_max_in",
        "amount_min_out",
        "amount_std_out",
        "amount_std_in",
        "iet_mean",
        "iet_std",
    }
)
# n_events = n_out + n_in once a counterparty is given; with a diagonal sigma it would
# count the same signal twice, so auto leaves it out (it stays selectable by name).
_REDUNDANT_WITH_COUNTERPARTY = frozenset({"n_events"})
_LOG_TEXT = "log1p(max(x, 0)) on amounts and gaps, negatives count as 0"

LONG_TABLE = (
    "one out row per event with a non-null entity (key = entity, other = counterparty); "
    "one in row per event whose counterparty is non-null and differs from the entity "
    "(key = counterparty, other = entity); events with a null entity are dropped; "
    "keys are cast to string; time is epoch seconds in UTC "
    "(timestamp = instant, timestamp_ntz = wall clock read as UTC, date = UTC midnight, "
    "numeric = epoch seconds)"
)


@dataclass(frozen=True)
class Plan:
    """A resolved aggregation plan: roles, the dims in catalogue order, extra dims."""

    roles: dict[str, str]
    dims: tuple[DimSpec, ...]
    extra_dims: tuple[tuple[str, str], ...]
    log_amounts: bool


def build_plan(
    entity: str,
    counterparty: str | None = None,
    amount: str | None = None,
    time: str | None = None,
    *,
    dims: Sequence[str] | Literal["auto"] = "auto",
    extra_dims: Mapping[str, str] | None = None,
    log_amounts: bool = True,
) -> Plan:
    """Resolve column roles into a plan.

    ``dims="auto"`` takes every catalogue dimension whose roles are present (but not
    ``n_events`` when a counterparty is given: it equals ``n_out + n_in``);
    a list selects dimensions by name (output order stays the catalogue order).
    ``extra_dims`` maps new dimension names to Spark SQL aggregate expressions
    evaluated per entity over the long table, whose columns are
    ``key, other, side ('out' | 'in'), amount, ts`` (``ts`` in epoch seconds).
    """
    given = {"entity": entity, "counterparty": counterparty, "amount": amount, "time": time}
    roles = {r: c for r, c in given.items() if c is not None}
    present = set(roles)
    if dims == "auto":
        chosen = tuple(
            d
            for d in DIMS
            if set(d.requires) <= present
            and not ("counterparty" in present and d.name in _REDUNDANT_WITH_COUNTERPARTY)
        )
    else:
        unknown = [n for n in dims if n not in _BY_NAME]
        if unknown:
            raise ValueError(f"unknown dims {unknown}; available: {[d.name for d in DIMS]}")
        missing = {n: [r for r in _BY_NAME[n].requires if r not in present] for n in dims}
        missing = {n: m for n, m in missing.items() if m}
        if missing:
            detail = ", ".join(f"{n} needs {m}" for n, m in missing.items())
            raise ValueError(f"dims need roles that were not given: {detail}")
        wanted = set(dims)
        chosen = tuple(d for d in DIMS if d.name in wanted)
    extras = tuple((extra_dims or {}).items())
    clash = [n for n, _ in extras if n in _BY_NAME or n == "entity"]
    if clash:
        raise ValueError(f"extra_dims names collide with built-in names: {clash}")
    return Plan(roles=roles, dims=chosen, extra_dims=extras, log_amounts=log_amounts)


def describe_plan(plan: Plan) -> str:
    """Deterministic text form of the plan (locked by a golden test)."""
    lines = [
        "roles: " + " ".join(f"{r}={plan.roles[r]}" for r in _ROLE_ORDER if r in plan.roles),
        f"log_amounts: {_LOG_TEXT if plan.log_amounts else 'off'}",
        f"long table: {LONG_TABLE}",
        "dims:",
    ]
    for d in plan.dims:
        text = d.definition
        if d.name in _LOG_DIMS and not plan.log_amounts:
            text = text.removesuffix("; log")
        if d.count:
            text += f"; shrunk toward the population mean, weight {SHRINK_WEIGHT:g} vs {d.count}"
        lines.append(f"  {d.name} [{d.kind}] stage={d.stage}: {text}")
    for name, expr in plan.extra_dims:
        lines.append(f"  {name} [extra] stage=events: {expr}")
    return "\n".join(lines) + "\n"


def _epoch_seconds(df: DataFrame, column: str) -> Column:
    dtype = df.schema[column].dataType
    col = F.col(column)
    if isinstance(dtype, T.TimestampType):
        return col.cast("double")
    if isinstance(dtype, T.TimestampNTZType | T.DateType):
        day = F.datediff(F.to_date(col), F.lit("1970-01-01"))
        if isinstance(dtype, T.DateType):
            return (day * _SECONDS_PER_DAY).cast("double")
        clock = F.hour(col) * 3600 + F.minute(col) * 60 + F.second(col)
        return (day * _SECONDS_PER_DAY + clock).cast("double")
    if isinstance(dtype, T.NumericType):
        return col.cast("double")
    raise TypeError(
        f"time column {column!r} has type {dtype.simpleString()}; "
        "cast it to timestamp, timestamp_ntz, date or epoch seconds first"
    )


def _long_table(df: DataFrame, plan: Plan) -> DataFrame:
    roles = plan.roles
    entity = F.col(roles["entity"]).cast("string")
    other = (
        F.col(roles["counterparty"]).cast("string")
        if "counterparty" in roles
        else F.lit(None).cast("string")
    )
    amount = (
        F.col(roles["amount"]).cast("double") if "amount" in roles else F.lit(None).cast("double")
    )
    ts = _epoch_seconds(df, roles["time"]) if "time" in roles else F.lit(None).cast("double")
    base = df.select(entity.alias("e"), other.alias("o"), amount.alias("amount"), ts.alias("ts"))
    base = base.where(F.col("e").isNotNull())
    out_rows = base.select(
        F.col("e").alias("key"),
        F.col("o").alias("other"),
        F.lit("out").alias("side"),
        "amount",
        "ts",
    )
    if "counterparty" not in roles:
        return out_rows
    in_rows = base.where(F.col("o").isNotNull() & (F.col("o") != F.col("e"))).select(
        F.col("o").alias("key"),
        F.col("e").alias("other"),
        F.lit("in").alias("side"),
        "amount",
        "ts",
    )
    return out_rows.unionByName(in_rows)


def _side_amount(side: str) -> Column:
    return F.when(F.col("side") == side, F.col("amount"))


def _log1p_non_negative(col: Column) -> Column:
    # The geometry clips shapes at 0, so a negative log value would silently read as 0.
    return F.log1p(F.greatest(col, F.lit(0.0)))


def aggregate(df: DataFrame, plan: Plan) -> DataFrame:
    """Compile the plan to Spark: one row per entity, ``entity`` + the plan's dims."""
    long = _long_table(df, plan)
    names = {d.name for d in plan.dims}
    out = F.col("side") == "out"
    inn = F.col("side") == "in"
    has_ts = F.col("ts").isNotNull()
    day = F.floor(F.col("ts") / _SECONDS_PER_DAY)
    hour = F.floor(F.pmod(F.col("ts"), F.lit(_SECONDS_PER_DAY)) / 3600)
    weekday = F.pmod(day + 3, F.lit(7))  # 1970-01-01 was a Thursday; 0 = Monday

    exprs: dict[str, Column] = {
        "n_events": F.count(F.lit(1)),
        "n_out": F.sum(F.when(out, 1).otherwise(0)),
        "n_in": F.sum(F.when(inn, 1).otherwise(0)),
        "n_distinct_out": F.countDistinct(F.when(out, F.col("other"))),
        "n_distinct_in": F.countDistinct(F.when(inn, F.col("other"))),
        "amount_sum_out": F.sum(_side_amount("out")),
        "amount_sum_in": F.sum(_side_amount("in")),
        "amount_mean_out": F.avg(_side_amount("out")),
        "amount_mean_in": F.avg(_side_amount("in")),
        "amount_std_out": F.stddev_pop(_side_amount("out")),
        "amount_std_in": F.stddev_pop(_side_amount("in")),
        "amount_max_out": F.max(_side_amount("out")),
        "amount_max_in": F.max(_side_amount("in")),
        "amount_min_out": F.min(_side_amount("out")),
        "round_amount_share": F.avg(
            F.when(
                _side_amount("out").isNotNull(),
                F.when((F.col("amount") > 0) & (F.col("amount") % 100 == 0), 1.0).otherwise(0.0),
            )
        ),
        "active_days": F.countDistinct(F.when(has_ts, day)),
        "span_days": (F.max("ts") - F.min("ts")) / _SECONDS_PER_DAY,
        "hour_share_night": F.avg(F.when(has_ts, F.when(hour < 6, 1.0).otherwise(0.0))),
        "weekend_share": F.avg(F.when(has_ts, F.when(weekday >= 5, 1.0).otherwise(0.0))),
    }
    base_names = [n for n in exprs if n in names]
    counts = count_columns(plan)
    needs_n = bool({"in_out_ratio", "top_counterparty_share_out"} & names) or bool(counts)
    aggs = [exprs[n].alias(n) for n in base_names]
    if needs_n:
        for helper in ("n_events", "n_out", "n_in"):
            if helper not in base_names:
                aggs.append(exprs[helper].alias(f"_{helper}"))
    aggs.extend(F.expr(expr).alias(name) for name, expr in plan.extra_dims)
    result = long.groupBy("key").agg(*aggs)

    def ref(name: str) -> Column:
        return F.col(name) if name in base_names else F.col(f"_{name}")

    if "in_out_ratio" in names:
        result = result.withColumn("in_out_ratio", ref("n_in") / ref("n_events"))
    if "top_counterparty_share_out" in names:
        top = (
            long.where(out & F.col("other").isNotNull())
            .groupBy("key", "other")
            .count()
            .groupBy("key")
            .agg(F.max("count").alias("_top"))
        )
        result = result.join(top, "key", "left").withColumn(
            "top_counterparty_share_out",
            F.when(ref("n_out") > 0, F.coalesce(F.col("_top"), F.lit(0)) / ref("n_out")).otherwise(
                0.0
            ),
        )
    if "burst_max_daily" in names:
        burst = (
            long.where(has_ts)
            .groupBy("key", day.alias("_day"))
            .count()
            .groupBy("key")
            .agg(F.max("count").alias("burst_max_daily"))
        )
        result = result.join(burst, "key", "left")
    if {"iet_mean", "iet_std"} & names:
        gap = F.col("ts") - F.lag("ts").over(Window.partitionBy("key").orderBy("ts"))
        gaps = (
            long.where(has_ts)
            .select("key", gap.alias("_gap"))
            .where(F.col("_gap").isNotNull())
            .groupBy("key")
            .agg(F.avg("_gap").alias("iet_mean"), F.stddev_pop("_gap").alias("iet_std"))
        )
        result = result.join(gaps, "key", "left")

    columns = [F.col("key").alias("entity")]
    for d in plan.dims:
        value = F.coalesce(F.col(d.name).cast("double"), F.lit(0.0))
        if plan.log_amounts and d.name in _LOG_DIMS:
            value = _log1p_non_negative(value)
        if d.kind == "poisson":
            value = value.cast("long")
        columns.append(value.alias(d.name))
    columns.extend(F.col(name) for name, _ in plan.extra_dims)
    # Sample sizes of shrunk dimensions travel with the table even when not selected.
    columns.extend(
        F.coalesce(ref(c), F.lit(0)).cast("long").alias(c) for c in counts if c not in names
    )
    return result.select(*columns)


def count_columns(plan: Plan) -> tuple[str, ...]:
    """Sample-size columns the plan's shrunk dimensions need, in catalogue order."""
    wanted = {d.count for d in plan.dims if d.count}
    return tuple(d.name for d in DIMS if d.name in wanted)
