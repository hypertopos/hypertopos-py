# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Sphere.export_reference: a calibration that can be scored against, without the entities."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import (  # noqa: E402
    calibration_card,
    calibration_status,
    evidence_record,
    explain,
    find_anomalies,
    score,
    sphere_of,
)

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


@pytest.fixture(scope="module")
def events(spark):
    return (
        spark.range(12_000)
        .select(
            F.concat(F.lit("E"), (F.col("id") % 1200).cast("string")).alias("src"),
            F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 1200).cast("string")).alias("dst"),
            (F.exp(F.randn(seed=11)) * 100).alias("amt"),
            F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
        )
        .cache()
    )


@pytest.fixture(scope="module")
def full(events, tmp_path_factory):
    return sphere_of(find_anomalies(events, **ROLES, sphere_root=str(tmp_path_factory.mktemp("f"))))


@pytest.fixture(scope="module")
def reference(full, tmp_path_factory):
    return full.export_reference(str(tmp_path_factory.mktemp("shared")))


def test_the_export_holds_no_entity_table_and_no_entity_keys(full, reference):
    assert sorted(os.listdir(reference.path)) == ["calibration.pkl", "plan.pkl"]
    assert reference.fingerprint == full.fingerprint
    assert reference.path.name == full.path.name
    assert sphere_of(str(reference.path)).calibration().dims == full.calibration().dims
    calibration = reference.calibration()
    assert calibration.vel_keys is None
    assert calibration.reference.vel_s0 is None
    assert full.calibration().vel_keys is not None  # the original keeps its sample


def test_scoring_against_the_export_equals_scoring_against_the_original(events, full, reference):
    a = {r["entity"]: r for r in score(events, like=full, dimension_ranks=True).collect()}
    b = {r["entity"]: r for r in score(events, like=reference, dimension_ranks=True).collect()}
    assert a.keys() == b.keys()
    for key, row in a.items():
        assert b[key]["p_value"] == row["p_value"]
        assert b[key]["is_anomaly"] == row["is_anomaly"]
        assert b[key]["dimension_rank_pct"] == row["dimension_rank_pct"]


def test_the_calibration_check_works_on_the_export(events, full, reference):
    scored = score(events, like=reference, dimensions=True).cache()
    status = calibration_status(scored, like=reference)
    original = calibration_status(score(events, like=full, dimensions=True), like=full)
    assert status["status"] == original["status"]
    assert status["rows"] == original["rows"]
    assert status["p_distribution"] == pytest.approx(original["p_distribution"])


def test_what_needs_the_entities_says_so(events, reference):
    scored = score(events, like=reference)
    # a scored batch is explained from its own events: no entity table needed
    assert explain(scored, top_n=3, events=events).count() > 0
    for call in (
        lambda: reference.entities(),
        lambda: calibration_card(reference),
        lambda: evidence_record(scored, top_n=3, events=events),
    ):
        with pytest.raises(ValueError, match="exported without its entities"):
            call()


def test_the_export_cannot_overwrite_the_calibration_it_comes_from(full):
    with pytest.raises(ValueError, match="overwrite"):
        full.export_reference(str(full.path.parent))
    assert (full.path / "entities.parquet").exists()
    assert full.calibration().vel_keys is not None


def test_an_export_into_a_directory_that_holds_a_calibration_with_entities_is_refused(
    events, full, tmp_path
):
    # The same events and settings under another root give a directory of the same name.
    twin = sphere_of(find_anomalies(events, **ROLES, sphere_root=str(tmp_path)))
    assert twin.path.name == full.path.name
    with pytest.raises(ValueError, match="holds a calibration with its entities"):
        full.export_reference(str(tmp_path))
    assert twin.entities().num_rows == full.entities().num_rows
    assert twin.calibration().vel_keys is not None  # the calibration there is untouched


@pytest.fixture(scope="module")
def earlier(full, tmp_path_factory):
    """A calibration as an earlier version saved it: no distribution of measured values."""
    import dataclasses
    import pickle
    import shutil

    path = tmp_path_factory.mktemp("earlier") / full.path.name
    shutil.copytree(full.path, path)
    calibration = dataclasses.replace(full.calibration(), raw_grid=None)
    with open(path / "calibration.pkl", "wb") as fh:
        pickle.dump(calibration, fh)
    return sphere_of(str(path))


def test_score_with_ranks_refuses_an_earlier_calibration_at_the_call(events, earlier):
    with pytest.raises(ValueError, match="recalibrate"):
        score(events, like=earlier, dimension_ranks=True)
    assert score(events, like=earlier).count() > 0  # scoring itself still works


def test_evidence_record_refuses_an_earlier_calibration(events, earlier):
    scored = score(events, like=earlier)
    with pytest.raises(ValueError, match="recalibrate"):
        evidence_record(scored, top_n=3, events=events)


def test_a_record_on_an_export_is_refused_before_any_spark_job(spark, events, reference):
    scored = score(events, like=reference)
    sc = spark.sparkContext
    sc.setJobGroup("record-on-export", "")
    try:
        with pytest.raises(ValueError, match="exported without its entities"):
            evidence_record(scored, top_n=3, events=events)
    finally:
        sc.setJobGroup("", "")
    assert len(sc.statusTracker().getJobIdsForGroup("record-on-export")) == 0


def test_find_anomalies_does_not_calibrate_into_an_export(events, reference):
    # The same events and settings under the export's root resolve to the export's directory.
    for cache in (True, False):
        with pytest.raises(ValueError, match="exported reference"):
            find_anomalies(events, **ROLES, sphere_root=str(reference.path.parent), cache=cache)
    assert sorted(os.listdir(reference.path)) == ["calibration.pkl", "plan.pkl"]
