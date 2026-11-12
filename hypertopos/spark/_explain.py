# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``explain`` and the ``Sphere`` handle: what a ``find_anomalies`` result rests on."""

from __future__ import annotations

import importlib.util
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F

from hypertopos.scoring import FrozenCalibration
from hypertopos.spark._api import (
    CALIBRATION_FILE,
    ENTITIES_FILE,
    METADATA_PATH,
    METADATA_SCORED,
    PLAN_FILE,
    entity_frame,
)
from hypertopos.spark._build import KEY, build_sphere, collect_entities

SPHERE_DIR = "sphere"
_MCP_ARGS = ("-m", "hypertopos_mcp.main")


@dataclass(frozen=True)
class Sphere:
    """The calibration behind a ``find_anomalies`` result: entities, calibration, geometry."""

    path: Path
    fingerprint: str

    def entities(self) -> pa.Table:
        """The per-entity table the calibration was fitted on."""
        if not (self.path / ENTITIES_FILE).exists():
            raise ValueError(
                "this calibration was exported without its entities (export_reference): "
                "new events can be scored and checked against it (score, calibration_status, "
                "explain with events=), but it cannot show the population it was fitted on"
            )
        return pq.read_table(self.path / ENTITIES_FILE)

    def export_reference(self, root: str | os.PathLike[str]) -> Sphere:
        """Save what scoring against this population needs, and nothing about its entities.

        Writes the calibration and the aggregation plan to a directory of the
        same name under ``root`` and returns that ``Sphere``. The copy holds no
        entity table, no entity keys and no entity states: ``score``,
        ``calibration_status`` and
        ``explain(..., events=)`` work against it; ``calibration_card``,
        ``evidence_record``, ``entities`` and ``geometry`` need the original. It
        does hold each dimension's quantiles, extremes included, and the
        population's sorted scores, so it describes the population closely.
        """
        from hypertopos.spark._score import _shipped_calibration

        target = Path(root) / self.path.name
        if target.resolve() == self.path.resolve():
            raise ValueError(
                "export_reference would overwrite this calibration: pass a directory "
                "other than the one it is kept in"
            )
        if (target / ENTITIES_FILE).exists():
            raise ValueError(
                f"{target} holds a calibration with its entities: export_reference would "
                "replace that calibration and leave its entity table in the export; pass "
                "another directory"
            )
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path / PLAN_FILE, target / PLAN_FILE)
        tmp = target / (CALIBRATION_FILE + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(_shipped_calibration(self.calibration()), fh)
        os.replace(tmp, target / CALIBRATION_FILE)  # written last: marks the export complete
        return Sphere(path=target, fingerprint=self.fingerprint)

    def calibration(self) -> FrozenCalibration:
        with open(self.path / CALIBRATION_FILE, "rb") as fh:
            calibration: FrozenCalibration = pickle.load(fh)
        return calibration

    def geometry(self) -> pa.Table:
        """Per entity: every dimension's value and deviation, plus the scores."""
        entities = self.entities()
        calibration = self.calibration()
        scores = calibration.score(entities, members=True)
        columns: dict[str, pa.Array] = {KEY: entities[KEY].combine_chunks()}
        shapes = calibration.shapes(entities)
        for j, dim in enumerate(calibration.dims):
            columns[dim] = pa.array(shapes[:, j].astype(np.float64))
            columns[f"delta_{dim}"] = pa.array(scores.delta[:, j].astype(np.float64))
        columns["anomaly_score"] = pa.array(scores.anomaly_score.astype(np.float64))
        columns["is_anomaly"] = pa.array(scores.is_anomaly)
        columns["p_value"] = pa.array(scores.conformal_p.astype(np.float64))
        columns["score_rank_pct"] = pa.array(scores.score_rank_pct.astype(np.float64))
        columns["delta_norm"] = pa.array(scores.delta_norm.astype(np.float64))
        if calibration.group_by is not None and scores.calibration_scope is not None:
            columns[calibration.group_by] = entities[calibration.group_by].combine_chunks()
            columns["calibration_scope"] = pa.array(scores.calibration_scope.tolist(), pa.string())
        return pa.table(columns)

    def lance_path(self) -> Path:
        """The navigable Lance sphere for agents, built on first use and reused after.

        Its geometry is the one ``find_anomalies`` reports: same scores, flags,
        ranks, p-values and ``delta_norm``. It is built on the driver's local disk and
        copied next to the calibration, since object-store mounts such as
        Databricks Volumes do not take the random writes a build makes.
        """
        target = self.path / SPHERE_DIR
        marker = Path("_gds_meta") / "sphere.json"
        if not (target / marker).exists():
            with tempfile.TemporaryDirectory(prefix="hypertopos_") as local:
                built = Path(local) / SPHERE_DIR
                build_sphere(self.entities(), self.calibration(), str(built))
                # sphere.json goes last: a copy cut short has none and is rebuilt.
                # No directory rename, which Windows refuses while freshly
                # copied files are still held open by indexing or antivirus.
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(
                    built,
                    target,
                    ignore=lambda d, _: ["sphere.json"] if Path(d) == built / "_gds_meta" else [],
                )
                shutil.copy2(built / marker, target / marker)
        return target

    def mcp_config(self) -> dict[str, object]:
        """An MCP client entry that serves this sphere (a ``mcpServers`` value)."""
        return {
            "command": sys.executable,
            "args": list(_MCP_ARGS),
            "env": {"HYPERTOPOS_SPHERE_PATH": str(self.lance_path())},
        }

    def serve(self) -> None:
        """Run the hypertopos MCP server on this sphere (stdio; blocks until it exits)."""
        if importlib.util.find_spec("hypertopos_mcp") is None:
            raise ImportError("serving agents needs the MCP server: pip install hypertopos-mcp")
        env = {**os.environ, "HYPERTOPOS_SPHERE_PATH": str(self.lance_path())}
        subprocess.run([sys.executable, *_MCP_ARGS], env=env, check=True)

    def to_delta(self, spark: SparkSession, table: str, mode: str = "overwrite") -> None:
        """Write ``geometry()`` to a table (Delta on Databricks) with ``saveAsTable``."""
        spark.createDataFrame(self.geometry().to_pandas()).write.mode(mode).saveAsTable(table)


def sphere_of(result: DataFrame | str | os.PathLike[str]) -> Sphere:
    """The ``Sphere`` behind a ``find_anomalies`` result, or saved at a path.

    A path is the ``Sphere.path`` of an earlier result kept on persistent storage
    (``sphere_root``), so a later job, in a new process, can score against it.
    """
    if isinstance(result, (str, os.PathLike)):
        path = Path(result)
        if not (path / CALIBRATION_FILE).exists():
            raise FileNotFoundError(
                f"no hypertopos calibration at {path}: pass the path of sphere_of(result) "
                "from a find_anomalies call whose sphere_root is persistent storage"
            )
        return Sphere(path=path, fingerprint=path.resolve().name)
    metadata = result.schema[KEY].metadata if KEY in result.columns else {}
    if METADATA_PATH not in metadata:
        raise ValueError(
            "this DataFrame does not carry a hypertopos calibration: pass the result of "
            "find_anomalies (selecting or renaming the 'entity' column drops its metadata)"
        )
    path = Path(metadata[METADATA_PATH])
    if not (path / CALIBRATION_FILE).exists():
        raise FileNotFoundError(
            f"the calibration at {path} is gone (driver temp directories do not survive a "
            "restart); run find_anomalies again, or set sphere_root to a persistent location"
        )
    # The directory name is the identifier: a path gives the same one in a later process.
    return Sphere(path=path, fingerprint=path.resolve().name)


def _scored_entities(
    events: DataFrame, sphere: Sphere, keys: list[str], group_by: str | None
) -> pa.Table:
    """The entity table of ``keys``, aggregated from their own events only."""
    with open(sphere.path / PLAN_FILE, "rb") as fh:
        plan = pickle.load(fh)
    roles = plan.roles
    touches = F.col(roles["entity"]).cast("string").isin(keys)
    if "counterparty" in roles:
        touches = touches | F.col(roles["counterparty"]).cast("string").isin(keys)
    frame = entity_frame(events.where(touches), plan, group_by)
    return collect_entities(frame.where(F.col(KEY).isin(keys)))


def explain(
    result: DataFrame,
    entities: Sequence[str] | None = None,
    top_n: int = 50,
    dims: Sequence[str] | None = None,
    events: DataFrame | None = None,
) -> DataFrame:
    """Why entities stand out: one row per entity and dimension.

    Args:
        result: The DataFrame ``find_anomalies`` or ``score`` returned.
        entities: Entity keys to explain; default is the ``top_n`` highest scores.
        top_n: How many of the highest-scoring entities, when ``entities`` is None.
        dims: Restrict to these dimensions; default is all.
        events: For a ``score`` result, the events it scored: their entities are
            not in the calibration, so their dimensions are recomputed from the
            events of the explained entities only.

    Returns:
        ``entity, dim, value, mu, sigma, delta, pct_of_total, direction``: the
        entity's value of the dimension (shares and spreads after the
        small-count adjustment ``find_anomalies`` applies), the population mean and standard
        deviation it is compared with (its group's, with ``group_by``), the
        signed deviation in standard deviations, the dimension's share of the
        entity's ``anomaly_score`` in percent (the Shapley split of its
        Mahalanobis evidence that ``explain_anomaly`` reports on a sphere, so
        correlated dimensions share a deviation they make together), and
        ``above`` / ``below`` / ``at`` the mean. Entities come in the order of
        ``entities`` when given, otherwise by score; their rows by share.
    """
    calibration, table, keys, scored, _ = _explained(result, entities, top_n, events, dims)
    scores = calibration.score(table, members=not scored)
    shapes = calibration.shapes(table)
    mu, sigma = calibration.raw_params(table)
    delta = scores.delta.astype(np.float64)
    shares = 100.0 * calibration.score_shares(table, scores.score_delta)
    wanted = [calibration.dims.index(d) for d in (dims or calibration.dims)]
    names = table[KEY].to_pylist()
    rank = {k: i for i, k in enumerate(keys)}
    rows = []
    for i in sorted(range(len(names)), key=lambda i: rank[names[i]]):
        for j in sorted(wanted, key=lambda j: -shares[i, j]):
            d = float(delta[i, j])
            rows.append(
                (
                    names[i],
                    calibration.dims[j],
                    float(shapes[i, j]),
                    float(mu[i, j]),
                    float(sigma[i, j]),
                    d,
                    float(shares[i, j]),
                    "above" if d > 0 else "below" if d < 0 else "at",
                )
            )
    schema = (
        f"{KEY} string, dim string, value double, mu double, sigma double, "
        "delta double, pct_of_total double, direction string"
    )
    return result.sparkSession.createDataFrame(rows, schema)


def axis_evidence(
    result: DataFrame,
    entities: Sequence[str] | None = None,
    top_n: int = 50,
    events: DataFrame | None = None,
) -> DataFrame:
    """How much of an entity's evidence is extreme values, how much is change: one row per entity.

    Args:
        result: The DataFrame ``find_anomalies`` or ``score`` returned.
        entities: Entity keys; default is the ``top_n`` highest scores.
        top_n: How many of the highest-scoring entities, when ``entities`` is None.
        events: For a ``score`` result, the events it scored (as for ``explain``).

    Returns:
        ``entity, p_value, is_anomaly, bits, bits_mahal, bits_current``:
        ``bits`` is the entity's evidence ``-log2 p_value``, split over the
        score axes as ``explain_anomaly`` splits it on a sphere. ``bits_mahal``
        is the part from how far its values lie from the population (the part
        ``explain`` divides over dimensions), ``bits_current`` the part from how
        unusually its activity changed between the two halves of the period.
        ``bits_current`` is null when the result has no ``current`` axis
        (without a ``time`` role, and by default with ``score_per_group``,
        where ``score_axes`` must name it); then ``bits_mahal`` equals ``bits``.
        Rows come in the order of ``entities`` when given, otherwise by score.
    """
    from hypertopos.engine.robust import attribute

    # The result's own p-value and flag: rescoring a few rows can land one rank
    # step away from the score the whole table got (floating-point summation order).
    calibration, table, keys, scored, stored = _explained(
        result, entities, top_n, events, None, verdict=True
    )
    evidence, refs = calibration.row_evidence(table, members=not scored)
    names = table[KEY].to_pylist()
    rank = {k: i for i, k in enumerate(keys)}
    rows = []
    for i in sorted(range(len(names)), key=lambda i: rank[names[i]]):
        p = float(stored[names[i]]["p_value"])
        split = attribute(refs[i], evidence.delta[i], evidence.axis_ranks[i], p)
        rows.append(
            (
                names[i],
                p,
                bool(stored[names[i]]["is_anomaly"]),
                split.bits,
                split.axis_bits.get("mahal", 0.0),
                split.axis_bits.get("current"),
            )
        )
    schema = (
        f"{KEY} string, p_value double, is_anomaly boolean, bits double, bits_mahal double, "
        "bits_current double"
    )
    return result.sparkSession.createDataFrame(rows, schema)


def _explained(
    result: DataFrame,
    entities: Sequence[str] | None,
    top_n: int,
    events: DataFrame | None,
    dims: Sequence[str] | None,
    *,
    verdict: bool = False,
) -> tuple[FrozenCalibration, pa.Table, list[str], bool, dict[str, Row]]:
    """The calibration, the entity table of the entities to explain, their keys in
    order, whether ``result`` came from ``score``, and with ``verdict`` the
    result's own rows for those keys, read in the same single action that picks them
    (a result that is not persisted is recomputed by every action on it)."""
    sphere = sphere_of(result)
    calibration = sphere.calibration()
    if dims is not None:
        unknown = [d for d in dims if d not in calibration.dims]
        if unknown:
            raise ValueError(f"unknown dims {unknown}; calibrated: {list(calibration.dims)}")
    scored = bool(KEY in result.columns and result.schema[KEY].metadata.get(METADATA_SCORED))
    if scored and events is None:
        # before any action: reading a score result that is not persisted rescores it
        raise ValueError(
            "this is a score() result: its entities are not in the calibration; "
            "pass the events it scored, explain(result, events=df)"
        )
    rows: list[Row] = []
    if entities is None:
        rows = result.orderBy("anomaly_score", ascending=False).limit(top_n).collect()
        keys = [r[KEY] for r in rows]
    else:
        keys = [str(k) for k in entities]
        if verdict:
            rows = result.where(F.col(KEY).isin(keys)).collect()
    stored = {r[KEY]: r for r in rows} if verdict else {}
    if verdict and set(keys) - set(stored):
        raise ValueError(f"not in this result: {sorted(set(keys) - set(stored))[:10]}")
    if scored:
        assert events is not None
        table = _scored_entities(events, sphere, keys, calibration.group_by)
    else:
        table = sphere.entities()
        table = table.filter(pc.is_in(table[KEY], value_set=pa.array(keys, pa.string())))
    missing = set(keys) - set(table[KEY].to_pylist())
    if missing:
        raise ValueError(f"not in this result: {sorted(missing)[:10]}")
    return calibration, table, keys, scored, stored
