# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``calibration_status``: do newly scored rows still come from the calibrated population?"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyspark.sql import DataFrame

from hypertopos.spark import _drift_stats as ds
from hypertopos.spark._explain import Sphere, sphere_of
from hypertopos.spark._plan import EVENT_TIMES, FIRST_EVENT, LAST_EVENT, window_days

MIN_ROWS = 100
ALPHA = 0.025
WINDOW_RATIO_MIN = 0.95
"""Shortest scored window, as a share of the calibration window, that keeps the alarm budget."""
WINDOW_RATIO_MAX = 1.0
"""Longest scored window, as a share of the calibration window (a 5 % longer one already fails)."""
_COUNTS_DDL = (
    "rows long, flagged long, p_hist array<long>, dim_counts array<long>, "
    "first_event double, last_event double"
)
_COUNTS = pa.schema(
    [
        ("rows", pa.int64()),
        ("flagged", pa.int64()),
        ("p_hist", pa.list_(pa.int64())),
        ("dim_counts", pa.list_(pa.int64())),
        (FIRST_EVENT, pa.float64()),
        (LAST_EVENT, pa.float64()),
    ]
)


def calibration_status(scored: DataFrame, like: DataFrame | Sphere) -> dict[str, Any]:
    """Test whether the rows of a ``score`` result still match the calibration.

    Two tests at a combined false-alarm rate of about 2.5 %: the rows'
    ``p_value`` against uniform (Kolmogorov-Smirnov), and each dimension's
    values against the calibrated population's distribution of it (one
    Kolmogorov-Smirnov test per dimension, Bonferroni over the dimensions).
    Both account for the calibration being a sample itself. The flagged count
    is also tested against the alarm budget (beta-binomial) and reported in
    ``p_flag_count``; it does not set ``status``. The batch's calendar span,
    first to last event, must be between ``WINDOW_RATIO_MIN`` and
    ``WINDOW_RATIO_MAX`` of the calibration window's: the window-length
    dimensions of a shorter or longer window come from another distribution.

    The executors count, the driver tests the summed counts: one Spark action
    over ``scored``. Persist or write ``scored`` first, or the action recomputes
    the scoring.

    Args:
        scored: A ``score(..., dimensions=True)`` result.
        like: The ``find_anomalies`` result ``scored`` was scored against, or its ``Sphere``.

    Returns:
        ``rows``, ``flagged``, ``alarm_rate``, ``expected_alarm_rate``,
        ``p_distribution``, ``p_dimensions`` (``{dim: p}``),
        ``drifted_dimensions``, ``p_flag_count``, ``window_days`` (the batch's
        span in whole days, rounded up), ``calibration_window_days`` (None for a
        calibration saved without it: the window is then not checked) and
        ``status``: ``"ok"``, ``"drift"``, ``"window_mismatch"`` (with a
        ``reason``; the statistics are still reported), or ``"too_few_rows"``
        under 100 rows (the p-values are then None), in reverse order of
        precedence. Calibrations with ``group_by`` give only ``status``
        ``"unsupported"`` and a ``reason``.

    Raises:
        ValueError: ``scored`` has no ``dimension_values``, ``first_event`` or
            ``last_event`` column.
    """
    sphere = like if isinstance(like, Sphere) else sphere_of(like)
    calibration = sphere.calibration()
    if calibration.score_per_group:
        return {
            "status": "unsupported",
            "reason": "score_per_group: each group has its own reference and alarm budget",
        }
    if calibration.group_by is not None:
        return {
            "status": "unsupported",
            "reason": "group_by: the status is measured for calibrations without groups only",
        }
    if not {"dimension_values", *EVENT_TIMES} <= set(scored.columns):
        raise ValueError(
            "calibration_status needs the dimension values and event times: pass a "
            "score(..., dimensions=True) result (a find_anomalies result ranks "
            "the calibrated entities themselves)"
        )
    ref = calibration.reference
    grids = [ds.grid_levels(ref.cdf_grid[d], ref.cdf_levels) for d in range(len(calibration.dims))]
    values = [g[0] for g in grids]
    parts = (
        scored.select("p_value", "is_anomaly", "dimension_values", *EVENT_TIMES)
        .mapInArrow(_counter(values), _COUNTS_DDL)
        .collect()
    )
    rows = sum(int(r["rows"]) for r in parts)
    flagged = sum(int(r["flagged"]) for r in parts)
    budget = 1.0 - calibration.percentile / 100.0
    firsts = [r[FIRST_EVENT] for r in parts if r[FIRST_EVENT] is not None]
    lasts = [r[LAST_EVENT] for r in parts if r[LAST_EVENT] is not None]
    window = window_days(min(firsts, default=None), max(lasts, default=None))
    calibration_window = calibration.window_days
    out: dict[str, Any] = {
        "rows": rows,
        "flagged": flagged,
        "alarm_rate": flagged / rows if rows else None,
        "expected_alarm_rate": budget,
        "p_distribution": None,
        "p_dimensions": None,
        "drifted_dimensions": [],
        "p_flag_count": None,
        "window_days": window,
        "calibration_window_days": calibration_window,
        "status": "too_few_rows",
    }
    if rows < MIN_ROWS:
        return out
    hist = np.sum([np.asarray(r["p_hist"], dtype=np.int64) for r in parts], axis=0)
    counts = np.sum([np.asarray(r["dim_counts"], dtype=np.int64) for r in parts], axis=0)
    n_eff = ds.effective_n(ref.n_fitted, rows)
    p_dims: dict[str, float] = {}
    start = 0
    for dim, (v, low, high) in zip(calibration.dims, grids, strict=True):
        stop = start + 2 * len(v) + 1
        p_dims[dim] = ds.ks_p(ds.grid_ks(counts[start:stop], low, high), n_eff)
        start = stop
    p_distribution = ds.ks_p(ds.uniform_ks_binned(hist), n_eff)
    drifted = [d for d, p in p_dims.items() if p < ALPHA / (2 * len(p_dims))]
    out.update(
        p_distribution=p_distribution,
        p_dimensions=p_dims,
        drifted_dimensions=drifted,
        p_flag_count=ds.flag_count_p(flagged, rows, ref.n_fitted, calibration.percentile),
        status="drift" if p_distribution < ALPHA / 2 or drifted else "ok",
    )
    if (
        window is not None
        and calibration_window is not None
        and not window_matches(window, calibration_window)
    ):
        out.update(
            status="window_mismatch",
            reason=(
                f"the scored window spans {window} days, the calibration window "
                f"{calibration_window} days: calibrate on a window of the same length "
                "as the scored one"
            ),
        )
    return out


def window_matches(window: int, calibration_window: int) -> bool:
    """Whether a scored window of ``window`` days keeps a ``calibration_window``-day budget."""
    return WINDOW_RATIO_MIN <= window / calibration_window <= WINDOW_RATIO_MAX


def _counter(
    values: list[np.ndarray],
) -> Callable[[Iterable[pa.RecordBatch]], Iterable[pa.RecordBatch]]:
    """Per partition: one row of mergeable counts (nothing for an empty partition)."""

    def run(batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        rows = flagged = 0
        hist = np.zeros(ds.P_BINS, dtype=np.int64)
        counts = [np.zeros(2 * len(v) + 1, dtype=np.int64) for v in values]
        first: float | None = None
        last: float | None = None
        for batch in batches:
            if not batch.num_rows:
                continue
            rows += batch.num_rows
            lo = pc.min(batch.column(FIRST_EVENT)).as_py()
            hi = pc.max(batch.column(LAST_EVENT)).as_py()
            if lo is not None:
                first = lo if first is None else min(first, lo)
            if hi is not None:
                last = hi if last is None else max(last, hi)
            flagged += int(batch.column("is_anomaly").to_numpy(zero_copy_only=False).sum())
            hist += ds.p_histogram(batch.column("p_value").to_numpy(zero_copy_only=False))
            x = (
                batch.column("dimension_values")
                .flatten()
                .to_numpy()
                .reshape(batch.num_rows, len(values))
            )
            for d, v in enumerate(values):
                counts[d] += ds.grid_counts(v, x[:, d])
        if rows:
            yield pa.RecordBatch.from_pylist(
                [
                    {
                        "rows": rows,
                        "flagged": flagged,
                        "p_hist": hist.tolist(),
                        "dim_counts": np.concatenate(counts).tolist(),
                        FIRST_EVENT: first,
                        LAST_EVENT: last,
                    }
                ],
                schema=_COUNTS,
            )

    return run
