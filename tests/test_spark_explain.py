# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""explain and the Sphere handle (hypertopos.spark)."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import explain, find_anomalies, sphere_of  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


@pytest.fixture(scope="module")
def result(spark, tmp_path_factory):
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
    out = find_anomalies(events, **ROLES, sphere_root=str(tmp_path_factory.mktemp("s"))).cache()
    out.count()
    return out


def test_explain_matches_top_reasons(result):
    top = result.orderBy("anomaly_score", ascending=False).limit(5).collect()
    rows = explain(result, top_n=5).collect()
    by_entity: dict[str, list] = {}
    for r in rows:
        by_entity.setdefault(r["entity"], []).append(r)
    assert list(by_entity) == [t["entity"] for t in top]  # highest score first
    for t in top:
        explained = by_entity[t["entity"]]
        assert sum(r["pct_of_total"] for r in explained) == pytest.approx(100.0)
        for reason, row in zip(t["top_reasons"], explained, strict=False):
            assert row["dim"] == reason["dim"]
            assert row["delta"] == pytest.approx(reason["delta"], rel=1e-6)
            assert row["pct_of_total"] == pytest.approx(reason["pct_of_total"], rel=1e-6)


def test_explain_row_arithmetic(result):
    for r in explain(result, top_n=3).collect():
        assert r["delta"] == pytest.approx((r["value"] - r["mu"]) / r["sigma"], rel=1e-4, abs=1e-5)
        assert r["direction"] == (
            "above" if r["delta"] > 0 else "below" if r["delta"] < 0 else "at"
        )


def test_explain_named_entities_and_dims(result):
    rows = explain(result, entities=["E1", "E2"], dims=["n_out", "amount_sum_out"]).collect()
    assert {(r["entity"], r["dim"]) for r in rows} == {
        (e, d) for e in ("E1", "E2") for d in ("n_out", "amount_sum_out")
    }
    with pytest.raises(ValueError, match="not in this result"):
        explain(result, entities=["nobody"])
    with pytest.raises(ValueError, match="unknown dims"):
        explain(result, dims=["nope"])


def test_sphere_geometry_reproduces_the_result(result):
    geometry = sphere_of(result).geometry()
    norms = dict(
        zip(geometry["entity"].to_pylist(), geometry["delta_norm"].to_pylist(), strict=True)
    )
    for r in result.select("entity", "delta_norm").collect():
        assert norms[r["entity"]] == pytest.approx(r["delta_norm"], rel=1e-6)
    assert "delta_n_out" in geometry.column_names
    assert "n_out" in geometry.column_names


def test_agent_sphere_is_built_once_with_the_result_geometry(result):
    import lance

    sphere = sphere_of(result)
    path = sphere.lance_path()
    meta = path / "_gds_meta" / "sphere.json"
    built_at = meta.stat().st_mtime_ns
    assert sphere.lance_path() == path
    assert meta.stat().st_mtime_ns == built_at  # reused, not rebuilt
    geo = lance.dataset(str(path / "geometry" / "entity_pattern" / "data.lance")).to_table(
        columns=["primary_key", "delta_norm", "is_anomaly"]
    )
    stored = dict(zip(geo["primary_key"].to_pylist(), geo["delta_norm"].to_pylist(), strict=True))
    for r in result.select("entity", "delta_norm").collect():
        assert stored[r["entity"]] == pytest.approx(r["delta_norm"], rel=1e-6, abs=1e-6)
    config = sphere.mcp_config()
    assert config["args"] == ["-m", "hypertopos_mcp.main"]
    assert config["env"] == {"HYPERTOPOS_SPHERE_PATH": str(path)}


def test_lost_metadata_gives_a_clear_error(result):
    renamed = result.withColumnRenamed("entity", "account")
    with pytest.raises(ValueError, match="does not carry a hypertopos calibration"):
        sphere_of(renamed)


@pytest.mark.skipif(sys.platform == "win32", reason="saveAsTable needs winutils on Windows")
def test_to_delta_writes_the_geometry_table(spark, result):
    sphere_of(result).to_delta(spark, "hypertopos_geometry_test")
    written = spark.table("hypertopos_geometry_test")
    assert written.count() == result.count()
    assert {
        "entity",
        "anomaly_score",
        "is_anomaly",
        "p_value",
        "score_rank_pct",
        "delta_norm",
    } <= set(written.columns)
