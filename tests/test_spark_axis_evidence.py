# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""axis_evidence (hypertopos.spark)."""

from __future__ import annotations

import dataclasses
import math

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import axis_evidence, explain, find_anomalies, score  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


def _events(spark, n: int, seed: int):
    return spark.range(n).select(
        F.concat(F.lit("E"), (F.col("id") % 1200).cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 1200).cast("string")).alias("dst"),
        (F.exp(F.randn(seed=seed)) * 100).alias("amt"),
        F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
    )


@pytest.fixture(scope="module")
def events(spark):
    return _events(spark, 12_000, 11).cache()


@pytest.fixture(scope="module")
def result(events, tmp_path_factory):
    out = find_anomalies(events, **ROLES, sphere_root=str(tmp_path_factory.mktemp("s"))).cache()
    out.count()
    return out


def test_one_row_per_entity_in_score_order(result):
    top = [r["entity"] for r in result.orderBy("anomaly_score", ascending=False).limit(5).collect()]
    rows = axis_evidence(result, top_n=5).collect()
    assert [r["entity"] for r in rows] == top
    stored = {r["entity"]: r for r in result.where(F.col("entity").isin(top)).collect()}
    for r in rows:
        assert r["p_value"] == pytest.approx(stored[r["entity"]]["p_value"], rel=1e-9)
        assert r["is_anomaly"] == stored[r["entity"]]["is_anomaly"]
        assert r["bits"] == pytest.approx(-math.log2(r["p_value"]))


def test_p_value_and_flag_are_the_results_own(result, monkeypatch):
    # Rescoring a handful of rows may differ from the stored score by a rank step.
    from hypertopos.scoring import FrozenCalibration

    real = FrozenCalibration.row_evidence

    def one_rank_off(self, table, *, members=False):
        evidence, refs = real(self, table, members=members)
        return dataclasses.replace(evidence, conformal_p=evidence.conformal_p * 0.5), refs

    monkeypatch.setattr(FrozenCalibration, "row_evidence", one_rank_off)
    rows = axis_evidence(result, top_n=5).collect()
    stored = {r["entity"]: r for r in result.collect()}
    for r in rows:
        assert r["p_value"] == stored[r["entity"]]["p_value"]
        assert r["is_anomaly"] == stored[r["entity"]]["is_anomaly"]


def test_bits_split_over_mahal_and_current(result):
    rows = axis_evidence(result, top_n=10).collect()
    assert all(r["bits_current"] is not None for r in rows)  # time role: current axis
    for r in rows:
        assert r["bits"] == pytest.approx(r["bits_mahal"] + r["bits_current"])
        assert r["bits_mahal"] >= 0 and r["bits_current"] >= 0


def test_without_current_all_evidence_is_mahal(events, tmp_path_factory):
    out = find_anomalies(
        events, **ROLES, score_axes=("mahal",), sphere_root=str(tmp_path_factory.mktemp("m"))
    )
    for r in axis_evidence(out, top_n=3).collect():
        assert r["bits_current"] is None
        assert r["bits_mahal"] == pytest.approx(r["bits"])


def test_explain_is_unchanged(result):
    before = explain(result, top_n=3).collect()
    axis_evidence(result, top_n=3).collect()
    assert explain(result, top_n=3).collect() == before


def test_score_result_needs_its_events(result, spark):
    fresh = _events(spark, 3_000, 12)
    scored = score(fresh, like=result)
    with pytest.raises(ValueError, match="pass the events it scored"):
        axis_evidence(scored, top_n=2)
    assert len(axis_evidence(scored, top_n=2, events=fresh).collect()) == 2
