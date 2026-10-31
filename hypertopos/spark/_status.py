# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``calibration_status``: do newly scored rows still come from the calibrated population?"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyspark.sql import DataFrame

from hypertopos.engine.robust import Reference
from hypertopos.scoring import POOLED_GROUP, FrozenCalibration
from hypertopos.spark import _drift_stats as ds
from hypertopos.spark._explain import Sphere, sphere_of
from hypertopos.spark._plan import EVENT_TIMES, FIRST_EVENT, LAST_EVENT, window_days

MIN_ROWS = 100
"""Fewest rows a test runs on: the batch, and each group's reference with ``score_per_group``."""
ALPHA = 0.025
WINDOW_RATIO_MIN = 0.95
"""Shortest scored window, as a share of the calibration window, that keeps the alarm budget."""
WINDOW_RATIO_MAX = 1.0
"""Longest scored window, as a share of the calibration window (a 5 % longer one already fails)."""
_COUNTS_DDL = (
    "unit string, rows long, flagged long, idx array<int>, cnt array<long>, "
    "first_event double, last_event double"
)
_COUNTS = pa.schema(
    [
        ("unit", pa.string()),
        ("rows", pa.int64()),
        ("flagged", pa.int64()),
        ("idx", pa.list_(pa.int32())),
        ("cnt", pa.list_(pa.int64())),
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

    With ``group_by`` every row is scored against the population, so the
    tests run on the whole batch as without groups, and ``groups`` adds each
    group's rows and flagged count; rows of a group unseen at calibration are
    tested with the rest, so a new or renamed group value does not change the
    status. With ``score_per_group`` each reference is tested on the rows it
    scored: each group with its own reference, and the pooled small groups
    together; Bonferroni over these groups keeps the combined rate. A group
    under ``MIN_ROWS`` rows is not tested. With ``score_per_group``, rows scored
    against the population (a group unseen at calibration, a pool too small for
    its own reference) are counted but not tested: they are a part of the
    population, whose distribution is not theirs. If the group values were
    recoded, every row is such a row: check ``groups`` for scope ``"global"``.

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
        calibration saved without it: the window is then not checked),
        ``score_axes`` (the axes of the batch's scores: ``["mahal"]``, as
        ``score`` ranks new entities), ``calibration_axes`` (the axes of the
        ``find_anomalies`` result's own scores) and ``status``: ``"ok"``,
        ``"drift"``, ``"window_mismatch"`` (with a ``reason``; the statistics
        are still reported), or ``"too_few_rows"`` under 100 rows (the p-values
        are then None), in reverse order of precedence. With
        ``score_per_group``, ``p_distribution`` and ``p_dimensions`` are the
        smallest group p-values times the number of tested groups (at most 1),
        ``p_flag_count`` is per group only, and ``"too_few_rows"`` means that
        no group with its own reference reached 100 rows.

        With ``group_by`` also ``groups``, ``{group: {...}}`` with ``scope``
        (the reference ``calibration_scope`` names for the group's rows:
        ``"group"``, ``"pooled"`` for the small groups calibrated together,
        ``"global"`` for the whole population's), ``rows``,
        ``flagged``, ``alarm_rate``, ``p_distribution``, ``p_dimensions``,
        ``drifted_dimensions``, ``p_flag_count`` and ``status`` (``"ok"``,
        ``"drift"``, ``"too_few_rows"``, or ``"untested"`` for a group not
        tested on its own), and ``drifted_groups`` (None without
        ``score_per_group``: the groups are not tested on their own).

    Raises:
        ValueError: ``scored`` has no ``dimension_values``, ``first_event``,
            ``last_event`` or (with ``group_by``) group column.
    """
    sphere = like if isinstance(like, Sphere) else sphere_of(like)
    calibration = sphere.calibration()
    group = [] if calibration.group_by is None else [calibration.group_by]
    missing = {"dimension_values", *EVENT_TIMES, *group} - set(scored.columns)
    if missing:
        raise ValueError(
            "calibration_status needs the dimension values, event times and group column "
            f"(missing: {', '.join(sorted(missing))}): pass a score(..., dimensions=True) "
            "result (a find_anomalies result ranks the calibrated entities themselves)"
        )
    units = status_units(calibration)
    columns = ["p_value", "is_anomaly", "dimension_values", *EVENT_TIMES, *group]
    parts = (
        scored.select(*(f"`{c}`" for c in columns))
        .mapInArrow(lambda batches: count_partition(units, batches), _COUNTS_DDL)
        .collect()
    )
    return status_from_counts(calibration, units, [r.asDict() for r in parts])


def window_matches(window: int, calibration_window: int) -> bool:
    """Whether a scored window of ``window`` days keeps a ``calibration_window``-day budget."""
    return WINDOW_RATIO_MIN <= window / calibration_window <= WINDOW_RATIO_MAX


@dataclass(frozen=True)
class StatusUnits:
    """What the executors need to count a scored batch per tested reference.

    ``values``: the unique grid values per dimension of each tested reference,
    keyed by unit: None for the population (without ``score_per_group``), the
    group label otherwise. Without ``score_per_group`` the population's test
    takes every row, of every group value: each row was scored against the
    population.
    """

    dims: int
    values: Mapping[str | None, list[np.ndarray]]
    group_by: str | None
    pooled: frozenset[str]
    per_group: bool


def _references(calibration: FrozenCalibration) -> dict[str | None, Reference]:
    """The tested references: the population's, or each group's own with ``score_per_group``."""
    if not calibration.score_per_group or calibration.groups is None:
        return {None: calibration.reference}
    return {
        gid: cal.reference for gid, cal in calibration.groups.items() if cal.reference is not None
    }


def _grids(ref: Reference, dims: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [ds.grid_levels(ref.cdf_grid[d], ref.cdf_levels) for d in range(dims)]


def status_units(calibration: FrozenCalibration) -> StatusUnits:
    dims = len(calibration.dims)
    return StatusUnits(
        dims=dims,
        values={
            unit: [g[0] for g in _grids(ref, dims)]
            for unit, ref in _references(calibration).items()
        },
        group_by=calibration.group_by,
        pooled=calibration.pooled,
        per_group=calibration.score_per_group,
    )


def _labels(units: StatusUnits, column: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Each row's group label as calibrated: its group, or the pool for a pooled group."""
    # The same string form the calibration's group ids take.
    raw: np.ndarray = pc.cast(column, pa.string()).to_numpy(zero_copy_only=False)
    ids = raw.astype(str).astype(object)
    if units.pooled:
        ids[np.isin(ids, list(units.pooled))] = POOLED_GROUP
    return ids


def _vector(values: list[np.ndarray], p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The p-value histogram and each dimension's grid counts, concatenated."""
    return np.concatenate(
        [ds.p_histogram(p), *(ds.grid_counts(v, x[:, d]) for d, v in enumerate(values))]
    )


def count_partition(
    units: StatusUnits, batches: Iterable[pa.RecordBatch]
) -> Iterator[pa.RecordBatch]:
    """Per partition: one row of mergeable counts per unit present (nothing when empty).

    A tested unit's counts are sparse (``idx``, ``cnt``); an untested group
    carries its rows and flagged count only.
    """
    acc: dict[str | None, list[Any]] = {}
    first: float | None = None
    last: float | None = None
    for batch in batches:
        if not batch.num_rows:
            continue
        lo = pc.min(batch.column(FIRST_EVENT)).as_py()
        hi = pc.max(batch.column(LAST_EVENT)).as_py()
        if lo is not None:
            first = lo if first is None else min(first, lo)
        if hi is not None:
            last = hi if last is None else max(last, hi)
        p = batch.column("p_value").to_numpy(zero_copy_only=False)
        flag = batch.column("is_anomaly").to_numpy(zero_copy_only=False).astype(bool)
        x = (
            batch.column("dimension_values")
            .flatten()
            .to_numpy()
            .reshape(batch.num_rows, units.dims)
        )
        parts: list[tuple[str | None, np.ndarray | None]] = []
        labels = None if units.group_by is None else _labels(units, batch.column(units.group_by))
        if not units.per_group:
            parts.append((None, None))
        if labels is not None:
            parts += [(str(label), labels == label) for label in np.unique(labels)]
        for unit, mask in parts:
            n = batch.num_rows if mask is None else int(mask.sum())
            values = units.values.get(unit)
            vec = None
            if values is not None:
                vec = _vector(values, p, x) if mask is None else _vector(values, p[mask], x[mask])
            k = int(flag.sum()) if mask is None else int(flag[mask].sum())
            slot = acc.get(unit)
            if slot is None:
                acc[unit] = [n, k, vec]
            else:
                slot[0] += n
                slot[1] += k
                if vec is not None:
                    slot[2] += vec
    if not acc:
        return
    rows = []
    for unit, (n, k, vec) in acc.items():
        nz = np.zeros(0, dtype=np.int32) if vec is None else np.flatnonzero(vec).astype(np.int32)
        rows.append(
            {
                "unit": unit,
                "rows": n,
                "flagged": k,
                "idx": nz.tolist(),
                "cnt": [] if vec is None else vec[nz].tolist(),
                FIRST_EVENT: first,
                LAST_EVENT: last,
            }
        )
    yield pa.RecordBatch.from_pylist(rows, schema=_COUNTS)


def _unit_tests(
    vec: np.ndarray,
    grids: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    dims: tuple[str, ...],
    n_fitted: int,
    rows: int,
) -> tuple[float, dict[str, float]]:
    """``p_distribution`` and ``p_dimensions`` of one unit's summed counts."""
    n_eff = ds.effective_n(n_fitted, rows)
    p_dims: dict[str, float] = {}
    start = ds.P_BINS
    for dim, (v, low, high) in zip(dims, grids, strict=True):
        stop = start + 2 * len(v) + 1
        p_dims[dim] = ds.ks_p(ds.grid_ks(vec[start:stop], low, high), n_eff)
        start = stop
    return ds.ks_p(ds.uniform_ks_binned(vec[: ds.P_BINS]), n_eff), p_dims


def _scope(calibration: FrozenCalibration, unit: str, tested: bool) -> str:
    """The ``calibration_scope`` of the rows of ``unit``, as scoring labels them."""
    if calibration.score_per_group and not tested:
        return "global"
    if unit == POOLED_GROUP:
        return "pooled"
    return "group" if calibration.groups is not None and unit in calibration.groups else "global"


def status_from_counts(
    calibration: FrozenCalibration, units: StatusUnits, parts: list[dict[str, Any]]
) -> dict[str, Any]:
    """The status from the partitions' summed counts (driver side)."""
    totals: dict[str | None, list[Any]] = {}
    for r in parts:
        unit = r["unit"]
        slot = totals.setdefault(unit, [0, 0, None])
        slot[0] += int(r["rows"])
        slot[1] += int(r["flagged"])
        if unit in units.values:
            if slot[2] is None:
                slot[2] = np.zeros(
                    ds.P_BINS + sum(2 * len(v) + 1 for v in units.values[unit]), dtype=np.int64
                )
            np.add.at(slot[2], np.asarray(r["idx"], dtype=np.int64), np.asarray(r["cnt"]))
    labels = sorted(u for u in totals if u is not None)
    if units.group_by is not None:
        rows = sum(totals[u][0] for u in labels)
        flagged = sum(totals[u][1] for u in labels)
    else:
        rows, flagged = totals[None][:2] if None in totals else (0, 0)
    firsts = [r[FIRST_EVENT] for r in parts if r[FIRST_EVENT] is not None]
    lasts = [r[LAST_EVENT] for r in parts if r[LAST_EVENT] is not None]
    window = window_days(min(firsts, default=None), max(lasts, default=None))
    calibration_window = calibration.window_days
    out: dict[str, Any] = {
        "rows": rows,
        "flagged": flagged,
        "alarm_rate": flagged / rows if rows else None,
        "expected_alarm_rate": 1.0 - calibration.percentile / 100.0,
        "p_distribution": None,
        "p_dimensions": None,
        "drifted_dimensions": [],
        "p_flag_count": None,
        "window_days": window,
        "calibration_window_days": calibration_window,
        # New entities carry no half states, so ``score`` ranks them on mahal alone.
        "score_axes": ["mahal"],
        "calibration_axes": list(calibration.reference.axes),
        "status": "too_few_rows",
    }
    refs = _references(calibration)
    dims = calibration.dims
    tested: dict[str | None, tuple[float, dict[str, float]]] = {}
    if rows >= MIN_ROWS:
        for unit, (n, _, vec) in totals.items():
            if vec is not None and n >= MIN_ROWS:
                ref = refs[unit]
                tested[unit] = _unit_tests(vec, _grids(ref, len(dims)), dims, ref.n_fitted, n)
    k = len(tested)
    if calibration.group_by is not None:
        out["groups"] = {}
        out["drifted_groups"] = [] if units.per_group else None
        for unit in labels:
            n, f, vec = totals[unit]
            entry: dict[str, Any] = {
                "scope": _scope(calibration, unit, unit in units.values),
                "rows": n,
                "flagged": f,
                "alarm_rate": f / n if n else None,
                "p_distribution": None,
                "p_dimensions": None,
                "drifted_dimensions": [],
                "p_flag_count": None,
                "status": "untested" if vec is None else "too_few_rows",
            }
            if unit in tested:
                p_dist, p_dims = tested[unit]
                drifted = [d for d, p in p_dims.items() if p < ALPHA / (2 * k * len(dims))]
                is_drift = p_dist < ALPHA / (2 * k) or bool(drifted)
                entry.update(
                    p_distribution=p_dist,
                    p_dimensions=p_dims,
                    drifted_dimensions=drifted,
                    p_flag_count=ds.flag_count_p(f, n, refs[unit].n_fitted, calibration.percentile),
                    status="drift" if is_drift else "ok",
                )
                if is_drift:
                    out["drifted_groups"].append(unit)
            out["groups"][unit] = entry
    if not tested:
        return out
    # Bonferroni over the tested units: the smallest p of each test times their count.
    p_distribution = min(1.0, k * min(t[0] for t in tested.values()))
    p_dims = {d: min(1.0, k * min(t[1][d] for t in tested.values())) for d in dims}
    drifted = [d for d, p in p_dims.items() if p < ALPHA / (2 * len(dims))]
    out.update(
        p_distribution=p_distribution,
        p_dimensions=p_dims,
        drifted_dimensions=drifted,
        p_flag_count=None
        if units.per_group
        else ds.flag_count_p(
            totals[None][1], totals[None][0], refs[None].n_fitted, calibration.percentile
        ),
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
