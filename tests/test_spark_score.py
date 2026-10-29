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
        assert s["anomaly_score"] == pytest.approx(r["anomaly_score"], abs=1e-6)
        # score() treats its rows as new entities: the p-value comes from the
        # out-of-fold calibration, so it is checked as a distribution below.
        assert 0.0 < s["p_value"] <= 1.0
        assert s["score_rank_pct"] == pytest.approx(r["score_rank_pct"], abs=1e-4)
        assert s["calibration_scope"] == r["calibration_scope"]
        assert [x["dim"] for x in s["top_reasons"]] == [x["dim"] for x in r["top_reasons"]]
    flagged_fit = sum(r["is_anomaly"] for r in result.collect())
    flagged_new = sum(s["is_anomaly"] for s in got.values())
    assert abs(flagged_new - flagged_fit) <= max(2, flagged_fit // 2)


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


def test_scored_result_points_to_its_calibration_and_is_marked_scored(events, result):
    # Its entities are not in the calibration's entity table, so explain needs the events.
    metadata = score(events, like=result).schema["entity"].metadata
    assert metadata[METADATA_PATH] == str(sphere_of(result).path)
    assert metadata["hypertopos.scored"] == "true"


def test_a_saved_calibration_scores_new_data_by_its_path(events, result):
    # A scheduled job runs in a fresh process: no result object, only the path.
    path = str(sphere_of(result).path)
    from_path = score(events, like=sphere_of(path))
    expected = {r["entity"]: r["delta_norm"] for r in score(events, like=result).collect()}
    got = {r["entity"]: r["delta_norm"] for r in from_path.collect()}
    assert got == pytest.approx(expected)


def test_a_path_without_a_calibration_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no hypertopos calibration"):
        sphere_of(str(tmp_path))


def test_explain_a_scored_result_from_the_events_it_scored(events, result):
    # Scoring the calibration data reproduces it, so the explanations must match.
    from hypertopos.spark import explain

    scored = score(events, like=result)
    assert sphere_of(scored).path == sphere_of(result).path
    top = result.orderBy("anomaly_score", ascending=False).limit(3).collect()
    keys = [r["entity"] for r in top]
    expected = explain(result, entities=keys).collect()
    got = explain(scored, entities=keys, events=events).collect()
    assert [(r["entity"], r["dim"]) for r in got] == [(r["entity"], r["dim"]) for r in expected]
    for g, e in zip(got, expected, strict=True):
        assert g["value"] == pytest.approx(e["value"], rel=1e-6, abs=1e-6)
        assert g["delta"] == pytest.approx(e["delta"], rel=1e-5, abs=1e-5)


def test_explain_a_scored_result_needs_its_events(events, result):
    from hypertopos.spark import explain

    with pytest.raises(ValueError, match="events="):
        explain(score(events, like=result), top_n=2)
