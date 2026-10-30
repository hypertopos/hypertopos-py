# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``score``: new events against a frozen calibration, on the executors."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace

import pyarrow as pa
from pyspark.sql import DataFrame

from hypertopos.engine.robust import Reference
from hypertopos.scoring import FrozenCalibration
from hypertopos.spark._api import PLAN_FILE, entity_frame, shape_result
from hypertopos.spark._build import KEY
from hypertopos.spark._explain import Sphere, sphere_of
from hypertopos.spark._plan import EVENT_TIMES, Plan
from hypertopos.spark._result import TOP_REASONS, result_table


def score(df: DataFrame, like: DataFrame | Sphere, *, dimensions: bool = False) -> DataFrame:
    """Score new events against the population ``like`` was calibrated on.

    Args:
        df: New events with the same columns as the events behind ``like``.
        like: A ``find_anomalies`` result, or its ``Sphere``.
        dimensions: Add ``dimension_values``, the values each entity was scored
            on in ``sphere_of(like).calibration().dims`` order; this is the
            input ``calibration_status`` needs, together with ``first_event``
            and ``last_event``, each entity's first and last event time in
            epoch seconds (UTC).

    Returns:
        The ``find_anomalies`` columns for the entities in ``df``, measured
        against the frozen population: no recalibration, so ``p_value`` and
        ``score_rank_pct`` place each entity within the original population. The
        aggregation and the scoring both run on the executors; the calibration
        travels with the tasks, and its sorted reference scores grow with the
        fitted population. With the ``current`` axis
        (``score_axes=("mahal", "current")``) new entities are scored on
        ``mahal`` alone, against the population's scores without ``current``,
        so a scored batch keeps the alarm budget. ``sphere_of`` on the
        result gives the calibration; ``explain(result, events=df)`` explains its
        entities from the events they were scored on.
    """
    sphere = like if isinstance(like, Sphere) else sphere_of(like)
    calibration = sphere.calibration()
    with open(sphere.path / PLAN_FILE, "rb") as fh:
        plan: Plan = pickle.load(fh)
    entities = entity_frame(df, plan, calibration.group_by, event_times=dimensions)
    # New entities carry no half states: they are scored without ``current``, so
    # the velocity field stays on the driver.
    shipped = replace(
        calibration,
        vel_keys=None,
        reference=_without_velocity(calibration.reference),
        groups=None
        if calibration.groups is None
        else {
            gid: replace(
                cal,
                vel_keys=None,
                reference=None if cal.reference is None else _without_velocity(cal.reference),
            )
            for gid, cal in calibration.groups.items()
        },
    )
    flat = entities.mapInArrow(
        _scorer(shipped, dimensions), _result_schema(calibration, dimensions)
    )
    return shape_result(
        flat,
        len(calibration.dims),
        sphere.fingerprint,
        sphere.path,
        calibration.group_by,
        scored=True,
        dimensions=dimensions,
    )


def _without_velocity(reference: Reference) -> Reference:
    return replace(reference, vel_s0=None, vel_u=None)


def _scorer(
    calibration: FrozenCalibration, dimensions: bool
) -> Callable[[Iterable[pa.RecordBatch]], Iterable[pa.RecordBatch]]:
    def run(batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        for batch in batches:
            if batch.num_rows:
                yield from result_table(
                    pa.Table.from_batches([batch]), calibration, dimensions=dimensions
                ).to_batches()

    return run


def _result_schema(calibration: FrozenCalibration, dimensions: bool) -> str:
    fields = [
        f"{KEY} string",
        "anomaly_score double",
        "is_anomaly boolean",
        "p_value double",
        "score_rank_pct double",
        "delta_norm double",
    ]
    for i in range(1, min(TOP_REASONS, len(calibration.dims)) + 1):
        fields += [f"reason_{i}_dim string", f"reason_{i}_delta double", f"reason_{i}_pct double"]
    if calibration.group_by is not None:
        fields += [f"`{calibration.group_by}` string", "calibration_scope string"]
    if dimensions:
        fields += ["dimension_values array<double>", *(f"{c} double" for c in EVENT_TIMES)]
    return ", ".join(fields)
