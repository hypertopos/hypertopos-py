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
from hypertopos.scoring import POOLED_GROUP, STATE_COLUMNS, FrozenCalibration
from hypertopos.spark._plan import Plan, describe_plan

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
    if calibration.group_by is not None:
        group_column = CALIBRATION_GROUP
        raw = entities[calibration.group_by].cast(pa.string()).to_pylist()
        pooled = [POOLED_GROUP if g in calibration.pooled else g for g in raw]
        entities = entities.append_column(group_column, pa.array(pooled, pa.string()))
    builder = (
        GDSBuilder(SPHERE_ID, path)
        if calibration.halves is None
        else _HalfStateBuilder(SPHERE_ID, path, entities, calibration.halves)
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


class _HalfStateBuilder(GDSBuilder):
    """A build whose ``current`` axis reads the entities' half-state columns.

    The entity table holds no events, only their per-half counts and log-amount
    sums: each (entity, half, side) becomes that many event rows of equal
    log-amount, placed so the build splits the halves at the same ``t_mid``.
    """

    def __init__(
        self, sphere_id: str, path: str, entities: pa.Table, halves: tuple[float, float]
    ) -> None:
        super().__init__(sphere_id, path)
        self._states = entities
        self._halves = halves

    def _score_events(self, pat: _PatternReg, entity_table: pa.Table) -> EventArrays | None:
        if "current" not in pat.score_axes:
            return None
        rows = pc.index_in(
            pc.cast(entity_table["primary_key"], pa.string()), value_set=self._states[KEY]
        ).to_numpy(zero_copy_only=False)
        raw = np.column_stack(
            [self._states[c].to_numpy(zero_copy_only=False) for c in STATE_COLUMNS]
        ).astype(np.float64)[rows]
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
