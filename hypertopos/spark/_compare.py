# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``compare``: which entities entered or left the flag between two results."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from hypertopos.spark._api import METADATA_SCORED
from hypertopos.spark._build import KEY
from hypertopos.spark._explain import sphere_of

_CARRIED = ("p_value", "is_anomaly", "anomaly_score")


def compare(previous: DataFrame, current: DataFrame) -> DataFrame:
    """Which entities entered or left the flag between two results of one calibration.

    Args:
        previous: An earlier ``score`` result.
        current: A later ``score`` result, scored ``like`` the same sphere. To
            compare with the calibration window itself, score its events first:
            ``score(events, like=result)``. A ``find_anomalies`` result is
            refused: it holds each entity's verdict inside its own reference,
            and with the ``current`` axis on other axes than ``score`` uses.

    Returns:
        One row per entity present in either result:

        - ``entity``;
        - ``transition``: ``entered`` (flagged now, not before), ``left``,
          ``stayed_flagged``, ``stayed_unflagged``, ``appeared`` (only in
          ``current``) or ``disappeared`` (only in ``previous``);
        - ``p_value_previous`` / ``p_value_current``, ``is_anomaly_previous`` /
          ``is_anomaly_current`` and ``anomaly_score_previous`` /
          ``anomaly_score_current``: the verdicts as each result holds them,
          null on the side the entity is absent from.

        It says that a verdict changed, not why: read the reasons of the current
        run in its ``top_reasons``, or with ``explain``.

    Raises:
        ValueError: If the two results were judged against different
            calibrations, one of them carries none, or one of them is a
            ``find_anomalies`` result.
    """
    try:
        before, now = sphere_of(previous), sphere_of(current)
    except ValueError:
        raise ValueError(
            "this DataFrame does not carry a hypertopos calibration: pass score() results "
            "(selecting or renaming the 'entity' column drops its metadata)"
        ) from None
    if before.fingerprint[:16] != now.fingerprint[:16]:
        raise ValueError(
            "compare needs two results judged against the same calibration: score both "
            "batches like one sphere (score(events, like=sphere_of(result)))"
        )
    for result in (previous, current):
        if not result.schema[KEY].metadata.get(METADATA_SCORED):
            raise ValueError(
                "compare needs two score() results; for the calibration window itself "
                "pass score(events, like=result), not the find_anomalies result"
            )
    old = previous.select(
        F.col(KEY).alias("_previous"), *(F.col(c).alias(f"{c}_previous") for c in _CARRIED)
    )
    new = current.select(
        F.col(KEY).alias("_current"), *(F.col(c).alias(f"{c}_current") for c in _CARRIED)
    )
    was, is_now = F.col("is_anomaly_previous"), F.col("is_anomaly_current")
    transition = (
        F.when(F.col("_previous").isNull(), "appeared")
        .when(F.col("_current").isNull(), "disappeared")
        .when(~was & is_now, "entered")
        .when(was & ~is_now, "left")
        .when(was & is_now, "stayed_flagged")
        .otherwise("stayed_unflagged")
    )
    return old.join(new, F.col("_previous") == F.col("_current"), "full_outer").select(
        F.coalesce("_previous", "_current").alias(KEY),
        transition.alias("transition"),
        *(f"{c}_{side}" for c in _CARRIED for side in ("previous", "current")),
    )
