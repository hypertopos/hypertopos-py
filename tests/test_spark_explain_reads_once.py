# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""axis_evidence and evidence_record read the result with one Spark job.

A result that is not persisted is recomputed by every action on it, so a second
read of a ``score`` result is a second pass over the events.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import axis_evidence, evidence_record, find_anomalies  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


@pytest.fixture(scope="module")
def result(spark, tmp_path_factory):
    events = spark.range(6_000).select(
        F.concat(F.lit("E"), (F.col("id") % 600).cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 600).cast("string")).alias("dst"),
        (F.exp(F.randn(seed=11)) * 100).alias("amt"),
        F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
    )
    out = find_anomalies(events, **ROLES, sphere_root=str(tmp_path_factory.mktemp("once"))).cache()
    out.count()
    return out


def _jobs(spark, call) -> int:
    sc = spark.sparkContext
    group = f"reads-once-{uuid.uuid4()}"
    sc.setJobGroup(group, "")
    try:
        call()
    finally:
        sc.setJobGroup("", "")
    return len(sc.statusTracker().getJobIdsForGroup(group))


def test_axis_evidence_reads_the_result_once(spark, result):
    assert _jobs(spark, lambda: axis_evidence(result, top_n=5)) == 1
    keys = [r["entity"] for r in result.limit(3).collect()]
    assert _jobs(spark, lambda: axis_evidence(result, entities=keys)) == 1


def test_evidence_record_reads_the_result_once(spark, result):
    assert _jobs(spark, lambda: evidence_record(result, top_n=5)) == 1
    keys = [r["entity"] for r in result.limit(3).collect()]
    assert _jobs(spark, lambda: evidence_record(result, entities=keys)) == 1


def test_a_scored_result_without_events_is_refused_before_any_job(spark, result):
    from hypertopos.spark import score

    events = spark.range(600).select(
        F.concat(F.lit("E"), F.col("id").cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") + 1) % 600).cast("string")).alias("dst"),
        F.lit(100.0).alias("amt"),
        F.timestamp_seconds(F.lit(1767225600) + F.col("id") * 60).alias("ts"),
    )
    scored = score(events, like=result)

    def refused():
        with pytest.raises(ValueError, match="pass the events"):
            evidence_record(scored, entities=["E1"])

    assert _jobs(spark, refused) == 0
