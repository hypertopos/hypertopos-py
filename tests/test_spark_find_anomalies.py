# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The one-call API: hypertopos.spark.find_anomalies."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import find_anomalies  # noqa: E402
from hypertopos.spark._api import METADATA_FINGERPRINT, METADATA_PATH  # noqa: E402

N_ENTITIES = 3000
ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


@pytest.fixture(scope="module")
def events(spark):
    # Generated in the JVM: 40 k events over 3 k entities, two segments.
    df = (
        spark.range(40_000)
        .select(
            F.concat(F.lit("E"), (F.col("id") % N_ENTITIES).cast("string")).alias("src"),
            F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % N_ENTITIES).cast("string")).alias(
                "dst"
            ),
            (F.exp(F.randn(seed=3)) * 100).alias("amt"),
            F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
            F.when(F.col("id") % N_ENTITIES < 1000, "retail")
            .otherwise("business")
            .alias("segment"),
        )
        .cache()
    )
    df.count()
    return df


@pytest.fixture(scope="module")
def result(events, tmp_path_factory):
    root = tmp_path_factory.mktemp("spheres")
    out = find_anomalies(events, **ROLES, sphere_root=str(root)).cache()
    out.count()
    return out


def test_one_row_per_entity_with_the_contract_columns(result):
    assert result.columns == [
        "entity",
        "delta_norm",
        "is_anomaly",
        "p_value",
        "rank_pct",
        "top_reasons",
    ]
    assert result.count() == N_ENTITIES
    assert result.select("entity").distinct().count() == N_ENTITIES
    reason = result.schema["top_reasons"].dataType.elementType
    assert [f.name for f in reason.fields] == ["dim", "delta", "pct_of_total"]


def test_anomaly_rate_matches_percentile(result):
    rate = result.where("is_anomaly").count() / N_ENTITIES
    assert rate == pytest.approx(0.01, abs=0.002)


def test_reasons_are_the_largest_shares_in_order(result):
    rows = result.select("top_reasons").collect()
    for row in rows:
        shares = [r["pct_of_total"] for r in row["top_reasons"]]
        assert 1 <= len(shares) <= 3
        assert all(s > 0 for s in shares)
        assert shares == sorted(shares, reverse=True)
        assert 0.0 <= sum(shares) <= 100.0 + 1e-6


def test_single_dimension_reason_takes_the_whole_distance(events, tmp_path):
    out = find_anomalies(events, entity="src", sphere_root=str(tmp_path)).collect()
    for row in out:
        if row["delta_norm"] == 0:
            assert row["top_reasons"] == []  # at the centre: no reason to report
            continue
        (reason,) = row["top_reasons"]
        assert reason["dim"] == "n_events"
        assert reason["pct_of_total"] == pytest.approx(100.0)


def test_p_value_and_rank_are_consistent_with_the_distance(result):
    rows = sorted(result.select("delta_norm", "p_value", "rank_pct").collect(), key=lambda r: r[0])
    p_values = [r["p_value"] for r in rows]
    ranks = [r["rank_pct"] for r in rows]
    assert all(0.0 <= p <= 1.0 for p in p_values)
    assert p_values == sorted(p_values, reverse=True)
    assert ranks == sorted(ranks)
    assert ranks[0] >= 0.0
    assert ranks[-1] <= 100.0


def test_entity_metadata_points_to_the_calibration(result):
    meta = result.schema["entity"].metadata
    assert len(meta[METADATA_FINGERPRINT]) == 64
    assert meta[METADATA_PATH].endswith(meta[METADATA_FINGERPRINT][:16])


def test_second_call_reuses_the_calibration(events, result):
    path = result.schema["entity"].metadata[METADATA_PATH]
    import pathlib

    sphere_json = pathlib.Path(path) / "calibration.pkl"
    before = sphere_json.stat().st_mtime_ns
    root = str(pathlib.Path(path).parent)
    find_anomalies(events, **ROLES, sphere_root=root).count()
    assert sphere_json.stat().st_mtime_ns == before
    find_anomalies(events, **ROLES, sphere_root=root, cache=False).count()
    assert sphere_json.stat().st_mtime_ns != before


def test_group_by_flags_about_the_percentile_in_each_group(events, tmp_path):
    out = find_anomalies(events, **ROLES, group_by="segment", sphere_root=str(tmp_path))
    assert out.columns[-2:] == ["segment", "calibration_scope"]
    assert {r[0] for r in out.select("calibration_scope").distinct().collect()} == {"group"}
    per_group = {
        r["segment"]: (r["n"], r["flagged"])
        for r in out.groupBy("segment")
        .agg(F.count("*").alias("n"), F.sum(F.col("is_anomaly").cast("int")).alias("flagged"))
        .collect()
    }
    assert set(per_group) == {"retail", "business"}
    for n, flagged in per_group.values():
        assert flagged / n == pytest.approx(0.01, abs=0.005)


def test_max_entities_refuses_a_too_large_build(events, tmp_path):
    with pytest.raises(ValueError, match="exceed max_entities=100"):
        find_anomalies(events, **ROLES, sphere_root=str(tmp_path), max_entities=100)


def test_group_value_is_the_most_frequent_one_and_tiny_groups_are_pooled(spark, tmp_path):
    # E1 has two "b" events and one "a": its group is "b", whatever the row order.
    rows = [("E1", "X", 1.0, "a"), ("E1", "Y", 1.0, "b"), ("E1", "Z", 1.0, "b")]
    rows += [(f"F{i}", "X", float(i % 7), "big") for i in range(300)]
    df = spark.createDataFrame(rows, "src string, dst string, amt double, seg string").cache()
    out = {
        r["entity"]: r.asDict()
        for r in find_anomalies(
            df,
            entity="src",
            counterparty="dst",
            amount="amt",
            group_by="seg",
            sphere_root=str(tmp_path),
        ).collect()
    }
    assert out["E1"]["seg"] == "b"
    assert out["E1"]["calibration_scope"] == "pooled"  # a one-entity group cannot set a threshold
    assert out["F1"]["calibration_scope"] == "group"
    assert out["X"]["seg"] == "(none)"  # seen only as a counterparty
