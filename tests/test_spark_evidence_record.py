# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""evidence_record (hypertopos.spark)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

import hypertopos  # noqa: E402
from hypertopos.datasets import synthetic_aml  # noqa: E402
from hypertopos.spark import (  # noqa: E402
    calibration_card,
    evidence_record,
    find_anomalies,
    score,
    sphere_of,
)


@pytest.fixture(scope="module")
def events(spark):
    data = synthetic_aml(n_accounts=1_000, seed=1)
    tx = spark.createDataFrame(data.transactions.to_pandas())
    accounts = spark.createDataFrame(data.accounts.to_pandas())
    return tx.join(accounts.select("account", "segment"), tx.from_account == accounts.account)


@pytest.fixture(scope="module")
def readme(events, tmp_path_factory):
    out = find_anomalies(
        events,
        entity="from_account",
        counterparty="to_account",
        amount="amount",
        time="ts",
        group_by="segment",
        score_per_group=True,
        sphere_root=str(tmp_path_factory.mktemp("rec")),
    ).cache()
    out.count()
    return out


@pytest.fixture(scope="module")
def records(readme):
    return evidence_record(readme, top_n=20).collect()


def test_one_row_per_entity_with_the_results_own_verdict(readme, records):
    top = readme.orderBy("anomaly_score", ascending=False).limit(20).collect()
    assert [r["entity"] for r in records] == [r["entity"] for r in top]
    for rec, row in zip(records, top, strict=True):
        assert rec["p_value"] == row["p_value"]
        assert rec["is_anomaly"] == row["is_anomaly"]
        assert rec["anomaly_score"] == row["anomaly_score"]
        assert rec["bits"] == pytest.approx(-math.log2(row["p_value"]))
        assert rec["segment"] == row["segment"]
        assert rec["calibration_scope"] == row["calibration_scope"]


def test_reasons_carry_the_value_and_its_place_in_the_population(readme, records):
    stored = {r["entity"]: r["top_reasons"] for r in readme.collect()}
    measured = sphere_of(readme).entities().to_pandas().set_index("entity")
    for rec in records:
        assert [x["dim"] for x in rec["reasons"]] == [x["dim"] for x in stored[rec["entity"]]]
        for reason, kept in zip(rec["reasons"], stored[rec["entity"]], strict=True):
            assert reason["delta"] == kept["delta"]
            assert reason["pct_of_total"] == kept["pct_of_total"]
            assert reason["value"] == pytest.approx(measured.loc[rec["entity"], reason["dim"]])
            assert 0.0 < reason["rank_pct"] < 100.0


def test_notes_say_what_limits_this_entitys_evidence(readme, records):
    card = calibration_card(readme)
    for rec in records:
        expected = {f"tied_dimension:{x['dim']}" for x in rec["reasons"]} & {
            f"tied_dimension:{d}" for d in card["tied_dimensions"]
        }
        if rec["calibration_scope"] != "group":
            expected.add(f"scope_{rec['calibration_scope']}")
        if not card["groups"][rec["segment"]]["can_flag"]:
            expected.add("reference_cannot_flag")
        assert set(rec["notes"]) == expected


def test_provenance_names_the_calibration(readme, records):
    card = calibration_card(readme)
    for rec in records:
        assert rec["calibration"] == sphere_of(readme).fingerprint
        assert rec["hypertopos_version"] == hypertopos.__version__
        assert rec["percentile"] == card["percentile"]
        assert rec["reference_size"] == card["groups"][rec["segment"]]["reference_size"]
        assert rec["score_axes"] == ["mahal"]
        assert rec["bits_current"] is None
        assert rec["bits_mahal"] == pytest.approx(rec["bits"])


def test_a_scored_batch_is_recorded_from_its_events(events, readme):
    # cached: the test collects the scored frame as well
    scored = score(events, like=readme).cache()
    recs = evidence_record(scored, top_n=5, events=events).collect()
    kept = {r["entity"]: r for r in scored.collect()}
    assert len(recs) == 5
    for rec in recs:
        assert rec["p_value"] == kept[rec["entity"]]["p_value"]
        assert rec["calibration"] == sphere_of(readme).fingerprint


def test_a_population_too_small_to_flag_says_so(spark, tmp_path):
    rng = np.random.default_rng(0)
    n, per = 150, 8
    src = np.repeat(np.arange(n), per)
    small = pd.DataFrame(
        {
            "src": [f"a{i}" for i in src],
            "dst": [f"a{(i + 1) % n}" for i in src],
            "amount": rng.lognormal(3.0, 1.0, n * per),
        }
    )
    out = find_anomalies(
        spark.createDataFrame(small),
        entity="src",
        counterparty="dst",
        amount="amount",
        sphere_root=str(tmp_path),
    )
    recs = evidence_record(out, top_n=3).collect()
    assert all("reference_cannot_flag" in r["notes"] for r in recs)
    assert all(r["reference_size"] == n for r in recs)
    assert "segment" not in recs[0].asDict()


def test_a_record_notes_an_entity_with_no_incoming_events(spark, tmp_path):
    events = spark.range(6_000).selectExpr(
        "CASE WHEN id % 200 = 0 THEN concat('S', id % 3) ELSE concat('E', id % 600) END AS src",
        "concat('E', (id * 7919 + 13) % 600) AS dst",
        "exp(randn(5)) * 100 AS amt",
        "timestamp_seconds(1767225600 + (id * 7907) % 5184000) AS ts",
    )
    result = find_anomalies(
        events, entity="src", counterparty="dst", amount="amt", time="ts", sphere_root=str(tmp_path)
    ).cache()
    table = sphere_of(result).entities().to_pandas().set_index("entity")
    silent = sorted(table.index[table["n_in"] == 0])
    both = table.index[table["n_in"] > 0].tolist()[:1]
    assert silent == ["S0", "S1", "S2"] and both
    notes = {
        r["entity"]: r["notes"] for r in evidence_record(result, entities=silent + both).collect()
    }
    for entity in silent:
        assert "no_incoming_events" in notes[entity]
    assert "no_incoming_events" not in notes[both[0]]
    assert not any("no_outgoing_events" in n for n in notes.values())


def test_notes_name_a_scope_outside_the_entitys_own_group(spark, tmp_path):
    events = spark.range(9_000).selectExpr(
        "concat('E', id % 900) AS src",
        "exp(randn(5)) * 100 AS amt",
        "CASE WHEN id % 900 < 20 THEN 'a' WHEN id % 900 < 50 THEN 'b' ELSE 'big' END AS seg",
    )
    result = find_anomalies(
        events,
        entity="src",
        amount="amt",
        group_by="seg",
        score_per_group=True,
        sphere_root=str(tmp_path),
    )
    recs = evidence_record(result, entities=["E0", "E30", "E500"]).collect()
    scopes = {r["entity"]: r["calibration_scope"] for r in recs}
    assert scopes["E500"] == "group"
    assert scopes["E0"] in ("pooled", "global") and scopes["E30"] in ("pooled", "global")
    for rec in recs:
        outside = [n for n in rec["notes"] if n.startswith("scope_")]
        expected = [] if scopes[rec["entity"]] == "group" else [f"scope_{scopes[rec['entity']]}"]
        assert outside == expected


def test_notes_say_when_the_reference_cannot_flag(spark, tmp_path):
    events = spark.range(1_200).selectExpr("concat('E', id % 150) AS src", "rand(1) * 100 AS amt")
    result = find_anomalies(events, entity="src", amount="amt", sphere_root=str(tmp_path))
    assert not calibration_card(result)["can_flag"]
    for rec in evidence_record(result, top_n=3).collect():
        assert "reference_cannot_flag" in rec["notes"]
