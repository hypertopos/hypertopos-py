# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``score``: new events against a frozen calibration, on the executors."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace

import numpy as np
import pyarrow as pa
from pyspark.sql import DataFrame

from hypertopos.engine.robust import Reference
from hypertopos.scoring import FrozenCalibration
from hypertopos.spark._api import PLAN_FILE, entity_frame, shape_result
from hypertopos.spark._build import KEY
from hypertopos.spark._explain import Sphere, sphere_of
from hypertopos.spark._plan import EVENT_TIMES, Plan
from hypertopos.spark._result import RANK_PREFIX, TOP_REASONS, result_table


def score(
    df: DataFrame,
    like: DataFrame | Sphere,
    *,
    dimensions: bool = False,
    dimension_ranks: bool = False,
) -> DataFrame:
    """Score new events against the population ``like`` was calibrated on.

    Args:
        df: New events with the same columns as the events behind ``like``.
        like: A ``find_anomalies`` result, or its ``Sphere``.
        dimensions: Add ``dimension_values``, the values each entity was scored
            on in ``sphere_of(like).calibration().dims`` order; this is the
            input ``calibration_status`` needs, together with ``first_event``
            and ``last_event``, each entity's first and last event time in
            epoch seconds (UTC).
        dimension_ranks: Add ``dimension_rank_pct``, a map from dimension name
            to where the entity's value sits in the calibrated population
            (its group's with ``score_per_group``), 0-100: the share of that
            population below the value as measured (shares and spreads before
            the small-count adjustment the score applies).

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
    if dimension_ranks:
        calibration.require_rank_grid()  # here, not in the first action on the executors
    with open(sphere.path / PLAN_FILE, "rb") as fh:
        plan: Plan = pickle.load(fh)
    entities = entity_frame(df, plan, calibration.group_by, event_times=dimensions)
    flat = entities.mapInArrow(
        _scorer(_shipped_calibration(calibration), dimensions, dimension_ranks),
        _result_schema(calibration, dimensions, dimension_ranks),
    )
    return shape_result(
        flat,
        len(calibration.dims),
        sphere.fingerprint,
        sphere.path,
        calibration.group_by,
        scored=True,
        dimensions=dimensions,
        rank_dims=calibration.dims if dimension_ranks else None,
    )


def _shipped_calibration(calibration: FrozenCalibration) -> FrozenCalibration:
    """What the executors need to score new entities.

    New entities carry no half states, so they are scored without ``current``:
    the velocity field, the in-sample fused scores and the ``current`` axis'
    sorted values stay on the driver. That keeps the shipped calibration under
    the Spark Connect message limit for a larger population.
    """
    return replace(
        calibration,
        vel_keys=None,
        reference=_for_new_rows(calibration.reference),
        groups=None
        if calibration.groups is None
        else {
            gid: replace(
                cal,
                vel_keys=None,
                reference=None if cal.reference is None else _for_new_rows(cal.reference),
            )
            for gid, cal in calibration.groups.items()
        },
    )


def _for_new_rows(reference: Reference) -> Reference:
    ref = replace(reference, vel_s0=None, vel_u=None)
    if ref.cal_sorted_no_current is None:
        # Without current every row is scored on the full calibration.
        return ref
    # A row without current takes rank 0.5 on that axis whatever its sorted values.
    return replace(
        ref,
        cal_sorted=np.empty(0),
        cal_sorted_oof=None,
        axis_sorted={**ref.axis_sorted, "current": np.empty(0)},
    )


def _scorer(
    calibration: FrozenCalibration, dimensions: bool, dimension_ranks: bool
) -> Callable[[Iterable[pa.RecordBatch]], Iterable[pa.RecordBatch]]:
    def run(batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        for batch in batches:
            if batch.num_rows:
                yield from result_table(
                    pa.Table.from_batches([batch]),
                    calibration,
                    dimensions=dimensions,
                    dimension_ranks=dimension_ranks,
                ).to_batches()

    return run


def _result_schema(calibration: FrozenCalibration, dimensions: bool, dimension_ranks: bool) -> str:
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
    if dimension_ranks:
        fields += [f"{RANK_PREFIX}{j} double" for j in range(len(calibration.dims))]
    return ", ".join(fields)
