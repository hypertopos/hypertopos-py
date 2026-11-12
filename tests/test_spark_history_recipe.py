# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The entity-history recipe of docs/spark.md: appended runs, onset and newly flagged entities."""

from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import find_anomalies, score, sphere_of  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}

ONSET = """
SELECT entity, MIN(run_date) AS first_flagged, MAX(run_date) AS last_flagged,
       COUNT(*) AS runs_flagged
FROM entity_history WHERE is_anomaly GROUP BY entity
"""
NEWLY_FLAGGED = """
SELECT entity, run_date FROM (
  SELECT *, LAG(is_anomaly) OVER (PARTITION BY entity ORDER BY run_date) AS flagged_before
  FROM entity_history)
WHERE is_anomaly AND NOT COALESCE(flagged_before, false)
"""


def _history(scored, like, run):
    return (
        scored.select("entity", "p_value", "is_anomaly", "anomaly_score", "top_reasons")
        .withColumn("run_date", F.lit(run))
        .withColumn("calibration", F.lit(like.fingerprint))
    )


def test_history_table_gives_onset_and_newly_flagged(spark, tmp_path):
    events = (
        spark.range(12_000)
        .select(
            F.concat(F.lit("E"), (F.col("id") % 1200).cast("string")).alias("src"),
            F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 1200).cast("string")).alias("dst"),
            (F.exp(F.randn(seed=11)) * 100).alias("amt"),
            F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
        )
        .cache()
    )
    like = sphere_of(find_anomalies(events, **ROLES, sphere_root=str(tmp_path)))
    day_1, day_2 = date(2026, 3, 1), date(2026, 3, 2)

    first = score(events, like=like).cache()
    quiet = first.where("NOT is_anomaly AND p_value > 0.5").orderBy("entity").first()["entity"]
    # on day 2 the quiet entity pays 300 counterparties in one burst
    burst = spark.range(300).select(
        F.lit(quiet).alias("src"),
        F.concat(F.lit("E"), F.col("id").cast("string")).alias("dst"),
        F.lit(5_000.0).alias("amt"),
        F.timestamp_seconds(F.lit(1767225600) + F.col("id") * 60).alias("ts"),
    )
    broken = events.unionByName(burst)
    second = score(broken, like=like)

    _history(first, like, day_1).unionByName(_history(second, like, day_2)).createOrReplaceTempView(
        "entity_history"
    )
    onset = {r["entity"]: r for r in spark.sql(ONSET).collect()}
    assert onset[quiet]["first_flagged"] == day_2
    assert onset[quiet]["runs_flagged"] == 1
    newly = {(r["entity"], r["run_date"]) for r in spark.sql(NEWLY_FLAGGED).collect()}
    assert (quiet, day_2) in newly
    # an entity flagged on both days is new only on the first
    both = [e for e, r in onset.items() if r["runs_flagged"] == 2]
    assert both
    assert all((e, day_2) not in newly for e in both)
    assert spark.table("entity_history").select("calibration").distinct().count() == 1
