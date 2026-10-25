# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""score: new events against a frozen calibration, on the executors."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import find_anomalies, score, sphere_of  # noqa: E402
from hypertopos.spark._api import METADATA_PATH  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


def _events(spark, n: int, seed: int):
    return spark.range(n).select(
        F.concat(F.lit("E"), (F.col("id") % 1000).cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 1000).cast("string")).alias("dst"),
        (F.exp(F.randn(seed=seed)) * 100).alias("amt"),
        F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
        F.when(F.col("id") % 1000 < 400, "retail").otherwise("business").alias("segment"),
    )


@pytest.fixture(scope="module")
def events(spark):
    df = _events(spark, 10_000, seed=4).cache()
    df.count()
    return df


@pytest.fixture(scope="module")
def result(events, tmp_path_factory):
    out = find_anomalies(
        events, **ROLES, group_by="segment", sphere_root=str(tmp_path_factory.mktemp("s"))
    ).cache()
    out.count()
    return out


def test_scoring_the_calibration_data_reproduces_the_result(events, result):
    # Four partitions on the executors give what the driver computed.
    scored = score(events.repartition(4), like=result)
    assert scored.columns == result.columns
    got = {r["entity"]: r for r in scored.collect()}
    for r in result.collect():
        s = got[r["entity"]]
        assert s["delta_norm"] == pytest.approx(r["delta_norm"], rel=1e-6, abs=1e-6)
        assert s["is_anomaly"] == r["is_anomaly"]
        assert s["p_value"] == pytest.approx(r["p_value"], abs=1e-6)
        assert s["calibration_scope"] == r["calibration_scope"]
        assert [x["dim"] for x in s["top_reasons"]] == [x["dim"] for x in r["top_reasons"]]


def test_new_events_are_measured_against_the_frozen_population(spark, result):
    shifted = _events(spark, 10_000, seed=4).withColumn("amt", F.col("amt") * 50)
    scored = score(shifted, like=sphere_of(result))
    flagged = scored.where("is_anomaly").count() / scored.count()
    assert flagged > 0.5  # a 50x amount shift stands out against the old population


def test_unseen_group_falls_back_to_the_global_calibration(spark, result):
    other = _events(spark, 2_000, seed=9).withColumn("segment", F.lit("wholesale"))
    scopes = {r[0] for r in score(other, like=result).select("calibration_scope").collect()}
    assert scopes == {"global"}


def test_explain_compares_with_the_entitys_own_group(result):
    from hypertopos.spark import explain

    rows = result.select("entity", "segment").collect()
    retail = next(r["entity"] for r in rows if r["segment"] == "retail")
    business = next(r["entity"] for r in rows if r["segment"] == "business")
    out = explain(result, entities=[retail, business], dims=["amount_sum_out"]).collect()
    by_entity = {r["entity"]: r for r in out}
    assert by_entity[retail]["mu"] != pytest.approx(by_entity[business]["mu"])
    for r in out:
        assert r["delta"] == pytest.approx((r["value"] - r["mu"]) / r["sigma"], rel=1e-4, abs=1e-5)


def test_scored_result_carries_no_calibration_metadata(events, result):
    scored = score(events, like=result)
    assert METADATA_PATH not in scored.schema["entity"].metadata
