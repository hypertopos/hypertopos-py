# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``evidence_record``: one self-contained row per entity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pyspark.sql import DataFrame

import hypertopos
from hypertopos.engine.robust import attribute, min_reference_size
from hypertopos.spark._build import KEY
from hypertopos.spark._card import one_sided, tied_dimensions
from hypertopos.spark._explain import _explained, sphere_of


def evidence_record(
    result: DataFrame,
    entities: Sequence[str] | None = None,
    top_n: int = 50,
    events: DataFrame | None = None,
) -> DataFrame:
    """Everything behind an entity's verdict in one row: the evidence, its limits, its source.

    Args:
        result: The DataFrame ``find_anomalies`` or ``score`` returned.
        entities: Entity keys; default is the ``top_n`` highest scores.
        top_n: How many of the highest-scoring entities, when ``entities`` is None.
        events: For a ``score`` result, the events it scored (as for ``explain``).

    Returns:
        One row per entity, in the order of ``entities`` when given, otherwise by
        score:

        - the verdict as the result holds it: ``entity``, ``p_value``,
          ``is_anomaly``, ``anomaly_score``, and with ``group_by`` the group
          column and ``calibration_scope``;
        - ``bits`` (``-log2 p_value``) with ``bits_mahal`` and ``bits_current``
          as :func:`axis_evidence` gives them;
        - ``reasons``: the result's ``top_reasons``, each with the entity's
          measured ``value`` of the dimension (log-scaled where the dimension
          is) and its ``rank_pct`` (0-100, the
          share of the population that scored the entity below that value). For
          shares and spreads ``explain`` shows the value after the small-count
          adjustment the score applies; here it is the value as measured;
        - ``notes``: what limits this entity's evidence. ``no_incoming_events``:
          no event names it as the counterparty (when the entity table has
          ``n_in``), so every incoming count, sum, mean and extreme is 0;
          ``tied_dimension:<dim>``:
          a reason rests on a dimension where one value holds at least 95 % of
          the entities; ``reference_cannot_flag``: the population that scored it
          is too small for any of its entities to reach the threshold;
          ``scope_pooled`` / ``scope_global``: with ``score_per_group``, it was
          scored against the pool of small groups or the whole population;
        - where the verdict comes from: ``reference_size`` (entities in the
          population that scored it), ``percentile``, ``score_axes``,
          ``window_days``, ``calibration`` (the fingerprint ``sphere_of`` shows)
          and ``hypertopos_version``.
    """
    sphere = sphere_of(result)
    sphere.calibration().require_rank_grid()  # both before any Spark action
    population = sphere.entities()
    calibration, table, keys, scored, stored = _explained(
        result, entities, top_n, events, None, verdict=True
    )
    group_by = calibration.group_by

    evidence, refs = calibration.row_evidence(table, members=not scored)
    measured = {
        dim: np.nan_to_num(table[dim].to_numpy(zero_copy_only=False).astype(np.float64))
        for dim in calibration.dims
    }
    ranks = calibration.dimension_rank_pct(table, evidence.delta)
    tied = tied_dimensions(calibration, population)
    sides = one_sided(table)
    alpha = 1.0 - calibration.percentile / 100.0
    smallest = min_reference_size(calibration.percentile)
    column = {name: j for j, name in enumerate(calibration.dims)}
    names = table[KEY].to_pylist()
    order = {k: i for i, k in enumerate(keys)}

    rows: list[tuple[Any, ...]] = []
    for i in sorted(range(len(names)), key=lambda i: order[names[i]]):
        row = stored[names[i]]
        ref = refs[i]
        size = ref.n_fitted
        split = attribute(ref, evidence.delta[i], evidence.axis_ranks[i], float(row["p_value"]))
        reasons = [
            (
                r["dim"],
                float(measured[r["dim"]][i]),
                float(ranks[i, column[r["dim"]]]),
                r["delta"],
                r["pct_of_total"],
            )
            for r in row["top_reasons"]
        ]
        notes = [note for note, mask in sides.items() if mask[i]]
        notes += [f"tied_dimension:{r['dim']}" for r in row["top_reasons"] if r["dim"] in tied]
        # A new row's smallest p is 1 / (n + 1); a fitted entity counts itself: 2 / (n + 1).
        if (1.0 / (size + 1) > alpha) if scored else (size < smallest):
            notes.append("reference_cannot_flag")
        if calibration.score_per_group and row["calibration_scope"] in ("pooled", "global"):
            notes.append(f"scope_{row['calibration_scope']}")
        rows.append(
            (
                names[i],
                row["p_value"],
                row["is_anomaly"],
                row["anomaly_score"],
                split.bits,
                split.axis_bits.get("mahal", 0.0),
                split.axis_bits.get("current"),
                reasons,
                notes,
                *((row[group_by], row["calibration_scope"]) if group_by is not None else ()),
                size,
                float(calibration.percentile),
                ["mahal"] if scored else list(ref.axes),  # a scored batch has no current axis
                None if calibration.window_days is None else float(calibration.window_days),
                sphere.fingerprint,
                hypertopos.__version__,
            )
        )
    group_fields = f"`{group_by}` string, calibration_scope string, " if group_by else ""
    schema = (
        f"{KEY} string, p_value double, is_anomaly boolean, anomaly_score double, "
        "bits double, bits_mahal double, bits_current double, "
        "reasons array<struct<dim:string,value:double,rank_pct:double,delta:double,"
        "pct_of_total:double>>, notes array<string>, "
        f"{group_fields}reference_size long, percentile double, score_axes array<string>, "
        "window_days double, calibration string, hypertopos_version string"
    )
    return result.sparkSession.createDataFrame(rows, schema)
