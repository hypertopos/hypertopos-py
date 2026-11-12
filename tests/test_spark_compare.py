# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""compare: which entities entered or left the flag between two results of one calibration."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import (  # noqa: E402
    compare,
    evidence_record,
    find_anomalies,
    score,
    sphere_of,
)

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


def _events(spark, seed):
    return spark.range(12_000).select(
        F.concat(F.lit("E"), (F.col("id") % 1200).cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 1200).cast("string")).alias("dst"),
        (F.exp(F.randn(seed=seed)) * 100).alias("amt"),
        F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
    )


@pytest.fixture(scope="module")
def runs(spark, tmp_path_factory):
    first = _events(spark, 11).cache()
    like = sphere_of(find_anomalies(first, **ROLES, sphere_root=str(tmp_path_factory.mktemp("c"))))
    # The later batch: other amounts, E5..E9 are in no event, N0..N4 are new senders.
    gone = [f"E{i}" for i in range(5, 10)]
    later = _events(spark, 12).where(~F.col("src").isin(gone) & ~F.col("dst").isin(gone))
    new = later.where(F.col("src").isin([f"E{i}" for i in range(5)])).withColumn(
        "src", F.concat(F.lit("N"), F.substring("src", 2, 9))
    )
    previous = score(first, like=like).cache()
    current = score(later.unionByName(new), like=like).cache()
    return like, previous, current


def test_every_entity_of_either_run_gets_one_row_with_its_transition(runs):
    _, previous, current = runs
    before = {r["entity"]: r for r in previous.collect()}
    now = {r["entity"]: r for r in current.collect()}
    rows = compare(previous, current).collect()
    assert sorted(r["entity"] for r in rows) == sorted(before.keys() | now.keys())
    seen = set()
    for r in rows:
        a, b = before.get(r["entity"]), now.get(r["entity"])
        if a is None:
            expected = "appeared"
        elif b is None:
            expected = "disappeared"
        elif a["is_anomaly"] != b["is_anomaly"]:
            expected = "entered" if b["is_anomaly"] else "left"
        else:
            expected = "stayed_flagged" if b["is_anomaly"] else "stayed_unflagged"
        assert r["transition"] == expected
        seen.add(expected)
        for side, row in (("previous", a), ("current", b)):
            for name in ("p_value", "is_anomaly", "anomaly_score"):
                assert r[f"{name}_{side}"] == (None if row is None else row[name])
    assert {"appeared", "disappeared", "entered", "left", "stayed_unflagged"} <= seen
    assert {f"N{i}" for i in range(5)} == {
        r["entity"] for r in rows if r["transition"] == "appeared"
    }
    # Besides the five, an account whose only events were with them is gone too.
    assert {f"E{i}" for i in range(5, 10)} <= {
        r["entity"] for r in rows if r["transition"] == "disappeared"
    }


def test_results_of_two_calibrations_are_refused(spark, runs, tmp_path_factory):
    _, previous, _ = runs
    other = find_anomalies(
        _events(spark, 13), **ROLES, sphere_root=str(tmp_path_factory.mktemp("o"))
    )
    with pytest.raises(ValueError, match="same calibration"):
        compare(previous, other)


def test_a_frame_without_a_calibration_is_refused(runs):
    _, previous, current = runs
    with pytest.raises(ValueError, match="does not carry a hypertopos calibration"):
        compare(previous, current.select("p_value", "is_anomaly"))


def test_a_batch_scored_against_the_exported_reference_compares_with_the_original(
    spark, runs, tmp_path_factory
):
    like, previous, _ = runs
    shared = like.export_reference(str(tmp_path_factory.mktemp("shared")))
    elsewhere = score(_events(spark, 12), like=shared)
    assert compare(previous, elsewhere).where("transition = 'disappeared'").count() == 0


def test_a_find_anomalies_result_is_refused(spark, runs):
    like, previous, _ = runs
    fitted = find_anomalies(_events(spark, 11), **ROLES, sphere_root=str(like.path.parent))
    assert sphere_of(fitted).fingerprint == like.fingerprint
    for pair in ((fitted, previous), (previous, fitted)):
        with pytest.raises(ValueError, match="two score"):
            compare(*pair)


def test_one_calibration_has_one_fingerprint(runs):
    like, previous, _ = runs
    assert sphere_of(previous).fingerprint == like.fingerprint
    assert sphere_of(str(like.path)).fingerprint == like.fingerprint


def test_a_scored_record_names_the_axis_its_verdict_used(spark, runs):
    like, previous, _ = runs
    assert "current" in like.calibration().reference.axes
    row = evidence_record(previous, top_n=1, events=_events(spark, 11)).first()
    assert row["score_axes"] == ["mahal"]
    assert row["bits_current"] is None
    assert row["calibration"] == sphere_of(str(like.path)).fingerprint
