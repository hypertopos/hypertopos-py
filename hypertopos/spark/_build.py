# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Entity table hand-off from Spark to the driver, and the sphere build over it."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import hypertopos
from hypertopos.builder import GDSBuilder
from hypertopos.builder.builder import _PatternReg
from hypertopos.engine.robust import EventArrays
from hypertopos.scoring import (
    GROUP_STATE_COLUMNS,
    POOLED_GROUP,
    STATE_COLUMNS,
    FrozenCalibration,
)
from hypertopos.spark._plan import EVENT_TIMES, Plan, describe_plan

SPHERE_ID = "spark"
LINE_ID = "entities"
PATTERN_ID = "entity_pattern"
KEY = "entity"
CALIBRATION_GROUP = "calibration_group"

_DAY = 86400.0

# spark.driver.maxResultSize is 1 GB by default and fixed on serverless; stay below it.
MAX_COLLECT_BYTES = 700_000_000
_KEY_BYTES = 32


def estimate_bytes(n_rows: int, n_numeric: int) -> int:
    """Upper-bound size of the entity table in Arrow (8 bytes per number + a short key)."""
    return n_rows * (8 * n_numeric + _KEY_BYTES)


def collect_entities(
    entity_df: DataFrame,
    *,
    n_rows: int | None = None,
    max_collect_bytes: int = MAX_COLLECT_BYTES,
) -> pa.Table:
    """Bring the per-entity table to the driver as Arrow.

    A table larger than ``max_collect_bytes`` comes over in chunks (entities
    split by a hash of the key), so no single collect reaches
    ``spark.driver.maxResultSize`` and nothing is written to storage the driver
    may not be able to read back (serverless, Volumes). ``entity_df`` should be
    persisted: it is read once per chunk. ``n_rows``, when known, saves a count
    that recomputes the aggregation where the frame cannot be cached.
    """
    if n_rows is None:
        n_rows = entity_df.count()
    if n_rows == 0:
        raise ValueError("the aggregation produced no entities; check the entity column")
    size = estimate_bytes(n_rows, len(entity_df.columns) - 1)
    chunks = max(1, math.ceil(size / max_collect_bytes))
    if chunks == 1:
        return _to_arrow(entity_df)
    bucket = F.pmod(F.crc32(F.col(KEY)), F.lit(chunks))
    parts = [_to_arrow(entity_df.where(bucket == i)) for i in range(chunks)]
    return pa.concat_tables([p for p in parts if p.num_rows])


def _to_arrow(frame: DataFrame) -> pa.Table:
    # toPandas exists on classic and Spark Connect sessions (serverless, shared
    # clusters); with Arrow enabled it transfers columnar batches.
    return _plain_types(pa.Table.from_pandas(frame.toPandas(), preserve_index=False))


def _plain_types(table: pa.Table) -> pa.Table:
    # pandas may hand strings back as large_string and adds its own schema metadata;
    # the entity table keeps plain string keys whichever way it reached the driver.
    fields = [
        pa.field(f.name, pa.string()) if pa.types.is_large_string(f.type) else f
        for f in table.schema
    ]
    return table.cast(pa.schema(fields))


def fingerprint(
    df: DataFrame,
    plan: Plan,
    *,
    percentile: float,
    group_by: str | None,
    min_group_size: int,
    score_per_group: bool = False,
    score_axes: tuple[str, ...] = ("mahal",),
) -> str:
    """Same input, same settings, same hypertopos version -> same fingerprint.

    The input is its semantic plan hash, the files it reads and its schema. A file
    or table source keeps its hash across re-reads and changes files with a new
    Delta version. An in-memory frame has no data identity Spark can give: its hash
    changes with every created frame, so the same frame reuses the sphere and a
    re-created one rebuilds. A plain file rewritten in place under the same name is
    not detected. Uses only APIs that Spark Connect sessions also have.
    """
    parts = [
        str(df.semanticHash()),
        "\n".join(sorted(df.inputFiles())),
        df.schema.json(),
        describe_plan(plan),
        f"percentile={percentile!r}",
        f"group_by={group_by!r}",
        f"min_group_size={min_group_size!r}",
        f"hypertopos={hypertopos.__version__}",
    ]
    if score_per_group:
        parts.append("score_per_group=True")
    if score_axes != ("mahal",):
        parts.append(f"score_axes={score_axes!r}")
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def sphere_dir(root: str, fp: str) -> Path:
    # Short paths: Windows MAX_PATH and Volume paths both favour them.
    return Path(root) / fp[:16]


def build_sphere(entities: pa.Table, calibration: FrozenCalibration, path: str) -> str:
    """Build the sphere agents navigate, with the geometry ``calibration`` reports.

    One anchor pattern whose dimensions are a metric block over the entity
    table, holding the values the calibration scores (small counts already
    shrunk). With ``group_by``, the build groups by the calibration's own groups
    (small ones already pooled), and with ``score_per_group`` scores each group
    against its own reference, so flags agree with ``find_anomalies``. With
    the ``current`` axis the build reads it from the entities' half-state
    columns, so the scores agree too.
    """
    shapes = calibration.shapes(entities)
    for j, dim in enumerate(calibration.dims):
        entities = entities.set_column(
            entities.schema.get_field_index(dim), dim, pa.array(shapes[:, j].astype("float64"))
        )
    group_column: str | None = None
    group_t_mid = None
    if calibration.group_by is not None:
        group_column = CALIBRATION_GROUP
        raw = entities[calibration.group_by].cast(pa.string()).to_pylist()
        pooled = [POOLED_GROUP if g in calibration.pooled else g for g in raw]
        entities = entities.append_column(group_column, pa.array(pooled, pa.string()))
        if calibration.halves is not None and set(GROUP_STATE_COLUMNS) <= set(
            entities.column_names
        ):
            group_t_mid = np.array([_t_mid_of(calibration, g) for g in pooled])
    builder = (
        GDSBuilder(SPHERE_ID, path)
        if calibration.halves is None
        else _HalfStateBuilder(SPHERE_ID, path, entities, calibration.halves, group_t_mid)
    )
    builder.add_line(LINE_ID, entities, key_col=KEY, source_id="spark")
    builder.add_pattern(
        PATTERN_ID,
        "anchor",
        LINE_ID,
        relations=[],
        anomaly_percentile=calibration.percentile,
        group_by_property=group_column,
        metric_properties=list(calibration.dims),
        score_per_group=calibration.score_per_group,
        score_min_group_size=calibration.min_group_size,
        score_axes=list(calibration.reference.axes),
    )
    return builder.build()


def _t_mid_of(calibration: FrozenCalibration, label: str) -> float:
    assert calibration.halves is not None and calibration.groups is not None
    group = calibration.groups.get(label)
    ref = None if group is None else group.reference
    return calibration.halves[1] if ref is None or ref.t_mid is None else ref.t_mid


class _HalfStateBuilder(GDSBuilder):
    """A build whose ``current`` axis reads the entities' half-state columns.

    The entity table holds no events, only their per-half counts and log-amount
    sums: each (entity, half, side) becomes that many event rows of equal
    log-amount, placed so the build splits the halves at the same ``t_mid``.
    With ``group_t_mid`` (each row's score group's ``t_mid``) the rows also
    carry ``GROUP_STATE_COLUMNS`` and their first / last event: the events then
    fill the intervals the two splits cut, and every row's events start and end
    at its own first and last event, so each group's own halves come out too.
    """

    def __init__(
        self,
        sphere_id: str,
        path: str,
        entities: pa.Table,
        halves: tuple[float, float],
        group_t_mid: np.ndarray | None = None,
    ) -> None:
        super().__init__(sphere_id, path)
        self._states = entities
        self._halves = halves
        self._group_t_mid = group_t_mid

    def _score_events(self, pat: _PatternReg, entity_table: pa.Table) -> EventArrays | None:
        if "current" not in pat.score_axes:
            return None
        rows = pc.index_in(
            pc.cast(entity_table["primary_key"], pa.string()), value_set=self._states[KEY]
        ).to_numpy(zero_copy_only=False)
        raw = self._raw(STATE_COLUMNS)[rows]
        if self._group_t_mid is not None:
            return self._split_events(raw, rows)
        t0, t_mid = self._halves
        days = round((t_mid - t0) / _DAY)
        # The build puts t_mid at t0 + ((last day + 1) // 2) days: a last event half
        # a day into day 2 * days - 1 gives the same t_mid.
        times = (t0, t0 + (2 * days - 0.5) * _DAY if days else t0)
        code, direction, ts, amount = [], [], [], []
        for half in (0, 1):
            for side in (0, 1):
                col = 5 * half + 2 * side
                n = raw[:, col].astype(np.int64)
                each = np.expm1(raw[:, col + 1] / np.maximum(n, 1))
                code.append(np.repeat(np.arange(len(raw)), n))
                amount.append(np.repeat(each, n))
                direction.append(np.full(int(n.sum()), side, dtype=np.int8))
                ts.append(np.full(int(n.sum()), times[half]))
        return EventArrays(
            np.concatenate(code),
            np.concatenate(direction),
            np.concatenate(ts),
            np.concatenate(amount),
        )

    def _raw(self, columns: tuple[str, ...]) -> np.ndarray:
        return np.column_stack(
            [
                pc.fill_null(pc.cast(self._states[c], pa.float64()), np.nan).to_numpy(
                    zero_copy_only=False
                )
                for c in columns
            ]
        )

    def _split_events(self, pop: np.ndarray, rows: np.ndarray) -> EventArrays:
        assert self._group_t_mid is not None
        own = self._raw(GROUP_STATE_COLUMNS)[rows]
        first, last = self._raw(EVENT_TIMES)[rows].T
        t_mid = self._halves[1]
        t_group = self._group_t_mid[rows]
        later = t_group >= t_mid  # the group splits later: the population's first half is A
        # Intervals A < min(t_mid, t_group) <= B < max(...) <= C; [count, log-sum] per side.
        parts = np.empty((3, 2, 2, len(rows)))
        for side in (0, 1):
            for j in (0, 1):
                col = 2 * side + j
                p1, p2, g1, g2 = pop[:, col], pop[:, 5 + col], own[:, col], own[:, 5 + col]
                parts[0, side, j] = np.where(later, p1, g1)
                parts[1, side, j] = np.where(later, g1 - p1, p1 - g1)
                parts[2, side, j] = np.where(later, g2, p2)
        n = np.rint(parts[:, :, 0]).astype(np.int64)  # (3, 2, rows)
        each = np.expm1(np.maximum(parts[:, :, 1], 0.0) / np.maximum(n, 1))
        busy = n.sum(axis=1) > 0  # (3, rows)
        lo = np.argmax(busy, axis=0)
        hi = 2 - np.argmax(busy[::-1], axis=0)
        times = np.tile(np.minimum(t_mid, t_group), (3, 1))  # a middle B
        r = np.arange(len(rows))
        times[lo, r] = first
        times[hi, r] = last
        # One interval only: one of its events moves to the first event, so the row
        # still starts and ends where its events do.
        moved = np.zeros_like(n)
        single = (lo == hi) & busy.any(axis=0)
        pick = np.where(n[lo, 0, r] > 0, 0, 1)
        moved[lo[single], pick[single], r[single]] = 1
        code, direction, ts, amount = [], [], [], []
        for i in range(3):
            for d in (0, 1):
                for count, when in ((n[i, d] - moved[i, d], times[i]), (moved[i, d], first)):
                    code.append(np.repeat(r, count))
                    amount.append(np.repeat(each[i, d], count))
                    direction.append(np.full(int(count.sum()), d, dtype=np.int8))
                    ts.append(np.repeat(when, count))
        return EventArrays(
            np.concatenate(code),
            np.concatenate(direction),
            np.concatenate(ts),
            np.concatenate(amount),
        )
