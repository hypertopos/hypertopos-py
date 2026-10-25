# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``score``: new events against a frozen calibration, on the executors."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Iterator

import pyarrow as pa
from pyspark.sql import DataFrame

from hypertopos.scoring import FrozenCalibration
from hypertopos.spark._api import PLAN_FILE, entity_frame, shape_result
from hypertopos.spark._build import KEY
from hypertopos.spark._explain import Sphere, sphere_of
from hypertopos.spark._plan import Plan
from hypertopos.spark._result import TOP_REASONS, result_table


def score(df: DataFrame, like: DataFrame | Sphere) -> DataFrame:
    """Score new events against the population ``like`` was calibrated on.

    Args:
        df: New events with the same columns as the events behind ``like``.
        like: A ``find_anomalies`` result, or its ``Sphere``.

    Returns:
        The ``find_anomalies`` columns for the entities in ``df``, measured
        against the frozen population: no recalibration, so ``p_value`` and
        ``rank_pct`` place each entity within the original population. The
        aggregation and the scoring both run on the executors; only the
        calibration (a few arrays) travels with the tasks. The result carries no
        calibration metadata: ``explain`` describes ``like``, not this result.
    """
    sphere = like if isinstance(like, Sphere) else sphere_of(like)
    calibration = sphere.calibration()
    with open(sphere.path / PLAN_FILE, "rb") as fh:
        plan: Plan = pickle.load(fh)
    entities = entity_frame(df, plan, calibration.group_by)
    flat = entities.mapInArrow(_scorer(calibration), _result_schema(calibration))
    return shape_result(flat, len(calibration.dims), None, None, calibration.group_by)


def _scorer(
    calibration: FrozenCalibration,
) -> Callable[[Iterable[pa.RecordBatch]], Iterable[pa.RecordBatch]]:
    def run(batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        for batch in batches:
            if batch.num_rows:
                yield from result_table(pa.Table.from_batches([batch]), calibration).to_batches()

    return run


def _result_schema(calibration: FrozenCalibration) -> str:
    fields = [
        f"{KEY} string",
        "delta_norm double",
        "is_anomaly boolean",
        "p_value double",
        "rank_pct double",
    ]
    for i in range(1, min(TOP_REASONS, len(calibration.dims)) + 1):
        fields += [f"reason_{i}_dim string", f"reason_{i}_delta double", f"reason_{i}_pct double"]
    if calibration.group_by is not None:
        fields += [f"`{calibration.group_by}` string", "calibration_scope string"]
    return ", ".join(fields)
