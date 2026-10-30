# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The one-call Spark API: a DataFrame of events in, a DataFrame of entities out."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyspark.errors import AnalysisException, PySparkException
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from hypertopos.engine.robust import halves
from hypertopos.scoring import FrozenCalibration, Scores, effective_min_group_size
from hypertopos.spark._build import KEY, collect_entities, fingerprint, sphere_dir
from hypertopos.spark._plan import (
    EVENT_TIMES,
    FIRST_EVENT,
    LAST_EVENT,
    Plan,
    aggregate,
    build_plan,
    count_entities,
    event_span,
    window_days,
)
from hypertopos.spark._result import TOP_REASONS, result_table

MAX_ENTITIES = 5_000_000
NO_GROUP = "(none)"
ENTITIES_FILE = "entities.parquet"
CALIBRATION_FILE = "calibration.pkl"
PLAN_FILE = "plan.pkl"
METADATA_FINGERPRINT = "hypertopos.fingerprint"
METADATA_PATH = "hypertopos.path"
METADATA_SCORED = (
    "hypertopos.scored"  # set on score() results: their entities are not in the calibration
)
SPARK_AXES = ("mahal", "current")


def _sphere_root(df: DataFrame, sphere_root: str | None) -> str:
    if sphere_root:
        return sphere_root
    try:
        configured = df.sparkSession.conf.get("hypertopos.sphere_root", None)
    except PySparkException:
        # Serverless refuses to read a conf it does not know (CONFIG_NOT_AVAILABLE)
        # instead of returning the default; the error class depends on the client.
        configured = None
    if configured:
        return configured
    return os.environ.get("HYPERTOPOS_SPHERE_ROOT") or str(
        Path(tempfile.gettempdir()) / "hypertopos"
    )


def find_anomalies(
    df: DataFrame,
    entity: str,
    counterparty: str | None = None,
    amount: str | None = None,
    time: str | None = None,
    *,
    group_by: str | None = None,
    percentile: float = 99.0,
    dims: str | Sequence[str] = "auto",
    extra_dims: Mapping[str, str] | None = None,
    log_amounts: bool = True,
    min_group_size: int | None = None,
    score_per_group: bool = False,
    score_axes: Sequence[str] | None = None,
    sphere_root: str | None = None,
    cache: bool = True,
    max_entities: int = MAX_ENTITIES,
) -> DataFrame:
    """Score every entity of an event table against its population.

    Args:
        df: Events, one row per transaction / call / shipment.
        entity: Column of the entity each event belongs to.
        counterparty: Column of the other side of the event.
        amount: Numeric column with the size of the event.
        time: Timestamp, timestamp_ntz, date or epoch-seconds column.
        group_by: Column of an entity property (e.g. segment, country); the
            deviations behind ``delta_norm`` and the reasons' ``delta`` are
            computed per group. By default the score, its p-value, flag and the
            reasons' shares stay population-wide; see ``score_per_group``. An
            entity takes the most frequent value in its own events (ties: the
            smallest); entities seen only as a counterparty go to ``"(none)"``.
        percentile: Population percentile above which an entity is flagged.
        dims: ``"auto"`` or the dimension names to use (see ``docs/spark.md``).
        extra_dims: ``{name: Spark SQL aggregate}`` over the event rows.
        log_amounts: ``log1p`` of amount and gap magnitudes.
        min_group_size: Groups smaller than this are calibrated together.
            Default 100; with ``score_per_group`` the reference size (199 at
            the 99th percentile), and a smaller value raises.
        score_per_group: Score each ``group_by`` group against its own
            population instead of the whole one, so an entity is flagged when
            it is rare within its group. Use it when the groups are distinct
            kinds of entity, each with at least about 1,000 members. Groups
            under ``min_group_size`` are pooled; a pool still under it, and a
            group not seen at calibration, are scored against the whole
            population.
        score_axes: The axes of ``anomaly_score``: ``("mahal",)``, the
            rarity of the entity's combination of dimensions, or
            ``("mahal", "current")``, which adds how unusually the entity's
            activity changed between the two calendar halves of the events
            compared with entities in the same state (needs ``time``; with
            ``score_per_group`` each group fits its own velocity field).
            Default ``("mahal", "current")`` when ``time`` is given, else
            ``("mahal",)``; with ``score_per_group`` ``("mahal",)``.
        sphere_root: Where calibrations are kept; default is the Spark conf
            ``hypertopos.sphere_root``, then ``HYPERTOPOS_SPHERE_ROOT``, then
            the driver's temp directory.
        cache: Reuse the calibration made earlier from the same input and settings.
        max_entities: Refuse to calibrate above this many entities (driver memory).

    Returns:
        One row per entity: ``entity``, ``anomaly_score`` (the calibrated
        evidence score, 0-1: how rare the entity's combination of dimensions
        is in the population), ``is_anomaly`` (``p_value`` at or below
        ``1 - percentile / 100``), ``p_value`` (conformal: share of the
        population scoring at least as high), ``score_rank_pct`` (0-100, share
        of the population with a lower score), ``delta_norm`` (distance from
        the population centre, in standard deviations), ``top_reasons`` (up to
        three ``{dim, delta, pct_of_total}``: the dimension's deviation in
        standard deviations and its share in percent of the Mahalanobis
        evidence, the Shapley split ``explain_anomaly`` reports on a sphere;
        empty for an entity at the centre of the score). With ``current`` the
        score and p-value equal those of a sphere built from the same events
        with an event pattern. With ``group_by``: ``delta_norm`` and
        the reasons' ``delta`` are measured against the entity's group, while
        the score and its shares are over the whole population unless
        ``score_per_group``, which measures them within the group too; the
        result adds that column and ``calibration_scope`` (``group``,
        ``pooled``, or with ``score_per_group`` ``global`` for an entity
        scored against the whole population). The
        ``entity`` column's metadata carries the fingerprint and the
        calibration path.
    """
    if score_per_group and group_by is None:
        raise ValueError("score_per_group needs group_by")
    group_size = effective_min_group_size(min_group_size, percentile, score_per_group)
    axes = _score_axes(score_axes, time, score_per_group)
    plan = build_plan(
        entity,
        counterparty,
        amount,
        time,
        dims=dims,
        extra_dims=extra_dims,
        log_amounts=log_amounts,
    )
    fp = fingerprint(
        df,
        plan,
        percentile=percentile,
        group_by=group_by,
        min_group_size=group_size,
        score_per_group=score_per_group,
        score_axes=axes,
    )
    path = sphere_dir(_sphere_root(df, sphere_root), fp)
    scores: Scores | None = None  # a cached entry holds no member scores: they are computed below
    if cache and (path / CALIBRATION_FILE).exists():
        entities = pq.read_table(path / ENTITIES_FILE)
        with open(path / CALIBRATION_FILE, "rb") as fh:
            calibration = pickle.load(fh)
    else:
        split = None
        if "current" in axes:
            first, last = event_span(df, plan)
            if first is None or last is None:
                raise ValueError(f"the time column {time!r} holds no values")
            split = halves(first, last)
        entities = _entity_table(
            df, plan, group_by, max_entities, t_mid=None if split is None else split[1]
        )
        # Key order makes the fit independent of the order Spark returns rows in.
        entities = entities.sort_by(KEY)
        window = window_days(
            pc.min(entities[FIRST_EVENT]).as_py(), pc.max(entities[LAST_EVENT]).as_py()
        )
        # The event times only measure the window: the saved entity table (and the
        # sphere built from it) holds the dimensions alone.
        entities = entities.drop_columns(list(EVENT_TIMES))
        dim_names = [d.name for d in plan.dims] + [name for name, _ in plan.extra_dims]
        # The fit's own member scores are the result's: no second scoring pass.
        calibration, scores = FrozenCalibration.fit_scored(
            entities,
            dim_names,
            percentile=percentile,
            group_by=group_by,
            min_group_size=group_size,
            shrink={d.name: d.count for d in plan.dims if d.count},
            score_per_group=score_per_group,
            window_days=window,
            halves=split,
            key=KEY if split is not None else None,
        )
        _save(path, entities, plan, calibration)

    table = result_table(entities, calibration, members=True, scores=scores)
    flat = df.sparkSession.createDataFrame(table.to_pandas())
    return shape_result(flat, len(calibration.dims), fp, path, group_by)


def _entity_table(
    df: DataFrame,
    plan: Plan,
    group_by: str | None,
    max_entities: int,
    *,
    t_mid: float | None = None,
) -> pa.Table:
    entity_df = entity_frame(df, plan, group_by, event_times=True, t_mid=t_mid)
    # Cached, counting the frame computes the aggregation once and the collect
    # reads the cache. Where caching is refused (serverless on older client
    # environments) every action recomputes it, so the count comes from the keys
    # alone and the full aggregation runs only for the collect.
    try:
        entity_df = entity_df.persist()
        cached = True
    except AnalysisException:
        cached = False
    try:
        n_entities = entity_df.count() if cached else count_entities(df, plan)
        if n_entities > max_entities:
            raise ValueError(
                f"{n_entities} entities exceed max_entities={max_entities}. Raise "
                "max_entities on a driver with more memory."
            )
        return collect_entities(entity_df, n_rows=n_entities)
    finally:
        if cached:
            entity_df.unpersist()


def entity_frame(
    df: DataFrame,
    plan: Plan,
    group_by: str | None,
    *,
    event_times: bool = False,
    t_mid: float | None = None,
) -> DataFrame:
    """Per-entity dimensions, plus the entity's ``group_by`` value when given.

    ``event_times=True`` adds each entity's ``first_event`` / ``last_event``;
    ``t_mid`` the half-state columns of the ``current`` axis.
    """
    entity_df = aggregate(df, plan, event_times=event_times, t_mid=t_mid)
    if group_by is None:
        return entity_df
    groups = _group_values(df, plan.roles["entity"], group_by)
    return entity_df.join(groups, KEY, "left").fillna({group_by: NO_GROUP})


def _score_axes(
    score_axes: Sequence[str] | None, time: str | None, score_per_group: bool
) -> tuple[str, ...]:
    if score_axes is None:
        # As a sphere's default: current whenever there are times to split, except
        # per group, where current is opt-in.
        return SPARK_AXES if time is not None and not score_per_group else ("mahal",)
    if isinstance(score_axes, str):
        raise TypeError(f"score_axes takes a tuple of axis names, e.g. ({score_axes!r},)")
    unknown = [a for a in score_axes if a not in SPARK_AXES]
    if unknown or "mahal" not in score_axes:
        raise ValueError(f"score_axes must be ('mahal',) or {SPARK_AXES}, got {tuple(score_axes)}")
    if "current" not in score_axes:
        return ("mahal",)
    if time is None:
        raise ValueError("the current axis needs time")
    return SPARK_AXES


def _group_values(df: DataFrame, entity: str, group_by: str) -> DataFrame:
    """Most frequent group value of each entity's own events; ties go to the smallest."""
    counts = (
        df.where(F.col(entity).isNotNull() & F.col(group_by).isNotNull())
        .groupBy(
            F.col(entity).cast("string").alias(KEY), F.col(group_by).cast("string").alias(group_by)
        )
        .count()
    )
    order = Window.partitionBy(KEY).orderBy(F.col("count").desc(), F.col(group_by).asc())
    return (
        counts.withColumn("_rank", F.row_number().over(order))
        .where(F.col("_rank") == 1)
        .select(KEY, group_by)
    )


def _save(path: Path, entities: pa.Table, plan: Plan, calibration: FrozenCalibration) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(entities, path / ENTITIES_FILE)
    with open(path / PLAN_FILE, "wb") as fh:
        pickle.dump(plan, fh)
    # The calibration is written last: its presence marks a complete cache entry.
    tmp = path / (CALIBRATION_FILE + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(calibration, fh)
    os.replace(tmp, path / CALIBRATION_FILE)


def shape_result(
    flat: DataFrame,
    n_dims: int,
    fp: str | None,
    path: Path | None,
    group_by: str | None,
    *,
    scored: bool = False,
    dimensions: bool = False,
) -> DataFrame:
    """Flat reason columns -> the result contract; metadata only when ``fp`` is given."""
    k = min(TOP_REASONS, n_dims)
    reasons = F.array(
        *[
            F.struct(
                F.col(f"reason_{i}_dim").alias("dim"),
                F.col(f"reason_{i}_delta").alias("delta"),
                F.col(f"reason_{i}_pct").alias("pct_of_total"),
            )
            for i in range(1, k + 1)
        ]
    )
    metadata = {METADATA_FINGERPRINT: fp, METADATA_PATH: str(path)}
    if scored:
        metadata[METADATA_SCORED] = "true"
    key = F.col(KEY).alias(KEY, metadata=metadata) if fp is not None else F.col(KEY)
    columns: list[Column | str] = [
        key,
        "anomaly_score",
        "is_anomaly",
        "p_value",
        "score_rank_pct",
        "delta_norm",
        F.filter(reasons, lambda r: r["pct_of_total"] > 0).alias("top_reasons"),
    ]
    if group_by is not None:
        columns += [group_by, "calibration_scope"]
    if dimensions:
        columns += ["dimension_values", *EVENT_TIMES]
    return flat.select(*columns)
