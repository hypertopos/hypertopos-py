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
import pyarrow.parquet as pq
from pyspark.errors import AnalysisException
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from hypertopos.scoring import MIN_GROUP_SIZE, FrozenCalibration
from hypertopos.spark._build import KEY, collect_entities, fingerprint, sphere_dir
from hypertopos.spark._plan import Plan, aggregate, build_plan
from hypertopos.spark._result import TOP_REASONS, result_table

MAX_ENTITIES = 5_000_000
NO_GROUP = "(none)"
ENTITIES_FILE = "entities.parquet"
CALIBRATION_FILE = "calibration.pkl"
PLAN_FILE = "plan.pkl"
METADATA_FINGERPRINT = "hypertopos.fingerprint"
METADATA_PATH = "hypertopos.path"


def _sphere_root(df: DataFrame, sphere_root: str | None) -> str:
    if sphere_root:
        return sphere_root
    configured = df.sparkSession.conf.get("hypertopos.sphere_root", None)
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
    min_group_size: int = MIN_GROUP_SIZE,
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
            population and the anomaly threshold are computed per group. An
            entity takes the most frequent value in its own events (ties: the
            smallest); entities seen only as a counterparty go to ``"(none)"``.
        percentile: Population percentile above which an entity is flagged.
        dims: ``"auto"`` or the dimension names to use (see ``docs/spark.md``).
        extra_dims: ``{name: Spark SQL aggregate}`` over the event rows.
        log_amounts: ``log1p`` of amount and gap magnitudes.
        min_group_size: Groups smaller than this are calibrated together.
        sphere_root: Where calibrations are kept; default is the Spark conf
            ``hypertopos.sphere_root``, then ``HYPERTOPOS_SPHERE_ROOT``, then
            the driver's temp directory.
        cache: Reuse the calibration made earlier from the same input and settings.
        max_entities: Refuse to calibrate above this many entities (driver memory).

    Returns:
        One row per entity: ``entity``, ``delta_norm`` (distance from the
        population centre, in standard deviations), ``is_anomaly``, ``p_value``
        (share of the population at least as far out), ``rank_pct`` (0-100,
        share of the population with a smaller ``delta_norm``), ``top_reasons``
        (up to three ``{dim, delta, pct_of_total}``, the share of
        ``delta_norm**2`` in percent; empty for an entity at the centre). With
        ``group_by``: that column and ``calibration_scope`` (``group`` or
        ``pooled``). The ``entity`` column's metadata carries the fingerprint
        and the calibration path.
    """
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
        df, plan, percentile=percentile, group_by=group_by, min_group_size=min_group_size
    )
    path = sphere_dir(_sphere_root(df, sphere_root), fp)
    if cache and (path / CALIBRATION_FILE).exists():
        entities = pq.read_table(path / ENTITIES_FILE)
        with open(path / CALIBRATION_FILE, "rb") as fh:
            calibration = pickle.load(fh)
    else:
        entities = _entity_table(df, plan, group_by, max_entities)
        dim_names = [d.name for d in plan.dims] + [name for name, _ in plan.extra_dims]
        calibration = FrozenCalibration.fit(
            entities,
            dim_names,
            percentile=percentile,
            group_by=group_by,
            min_group_size=min_group_size,
            shrink={d.name: d.count for d in plan.dims if d.count},
        )
        _save(path, entities, plan, calibration)

    table = result_table(entities, calibration)
    flat = df.sparkSession.createDataFrame(table.to_pandas())
    return shape_result(flat, len(calibration.dims), fp, path, group_by)


def _entity_table(
    df: DataFrame,
    plan: Plan,
    group_by: str | None,
    max_entities: int,
) -> pa.Table:
    entity_df = entity_frame(df, plan, group_by)
    # Caching saves recomputing the aggregation between count and collect;
    # serverless compute on older client environments refuses it.
    try:
        entity_df = entity_df.persist()
        cached = True
    except AnalysisException:
        cached = False
    try:
        n_entities = entity_df.count()
        if n_entities > max_entities:
            raise ValueError(
                f"{n_entities} entities exceed max_entities={max_entities}. Raise "
                "max_entities on a driver with more memory."
            )
        return collect_entities(entity_df)
    finally:
        if cached:
            entity_df.unpersist()


def entity_frame(df: DataFrame, plan: Plan, group_by: str | None) -> DataFrame:
    """Per-entity dimensions, plus the entity's ``group_by`` value when given."""
    entity_df = aggregate(df, plan)
    if group_by is None:
        return entity_df
    groups = _group_values(df, plan.roles["entity"], group_by)
    return entity_df.join(groups, KEY, "left").fillna({group_by: NO_GROUP})


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
    flat: DataFrame, n_dims: int, fp: str | None, path: Path | None, group_by: str | None
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
    key = (
        F.col(KEY).alias(KEY, metadata={METADATA_FINGERPRINT: fp, METADATA_PATH: str(path)})
        if fp is not None
        else F.col(KEY)
    )
    columns: list[Column | str] = [
        key,
        "delta_norm",
        "is_anomaly",
        "p_value",
        "rank_pct",
        F.filter(reasons, lambda r: r["pct_of_total"] > 0).alias("top_reasons"),
    ]
    if group_by is not None:
        columns += [group_by, "calibration_scope"]
    return flat.select(*columns)
