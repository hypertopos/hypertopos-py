# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""dimension_rank_pct on find_anomalies and score results (hypertopos.spark)."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.spark import find_anomalies, score, sphere_of  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}
N = 1200


@pytest.fixture(scope="module")
def events(spark):
    return (
        spark.range(12_000)
        .select(
            F.concat(F.lit("E"), (F.col("id") % N).cast("string")).alias("src"),
            F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % N).cast("string")).alias("dst"),
            (F.exp(F.randn(seed=11)) * 100).alias("amt"),
            F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
        )
        .cache()
    )


@pytest.fixture(scope="module")
def ranked(events, tmp_path_factory):
    out = find_anomalies(
        events, **ROLES, dimension_ranks=True, sphere_root=str(tmp_path_factory.mktemp("r"))
    ).cache()
    out.count()
    return out


def test_default_result_has_no_rank_column(events, tmp_path):
    out = find_anomalies(events, **ROLES, sphere_root=str(tmp_path))
    assert "dimension_rank_pct" not in out.columns


def test_ranks_are_named_by_dimension(ranked):
    dims = list(sphere_of(ranked).calibration().dims)
    row = ranked.select("dimension_rank_pct").first()[0]
    assert sorted(row) == sorted(dims)
    assert all(0.0 < v < 100.0 for v in row.values())


def test_a_rank_filter_selects_that_share_of_the_population(ranked):
    # amount_sum_out is continuous here: no ties, so the top 5 % is 5 % of the entities
    top = ranked.where("dimension_rank_pct['amount_sum_out'] >= 95").count()
    assert top == pytest.approx(0.05 * N, abs=3)


def test_score_gives_the_same_ranks_for_the_same_events(events, ranked):
    rescored = score(events, like=ranked, dimension_ranks=True)
    assert "dimension_rank_pct" in rescored.columns
    a = {r["entity"]: r["dimension_rank_pct"] for r in ranked.limit(50).collect()}
    b = {
        r["entity"]: r["dimension_rank_pct"]
        for r in rescored.where(F.col("entity").isin(list(a))).collect()
    }
    for key, ranks in a.items():
        assert b[key] == pytest.approx(ranks, abs=1e-3)


def test_equal_measured_values_rank_equal_on_adjusted_dimensions(ranked):
    # shares and spreads are scored after a small-count adjustment; the rank is of
    # the value as measured, so entities with one measured value share one rank
    sphere = sphere_of(ranked)
    measured = sphere.entities().to_pandas().set_index("entity")
    ranks = {r["entity"]: r["dimension_rank_pct"] for r in ranked.collect()}
    adjusted = list(sphere.calibration().shrink)
    assert adjusted
    repeated = 0
    for dim in adjusted:
        by_value: dict[float, set[float]] = {}
        for entity, value in measured[dim].items():
            by_value.setdefault(float(value), set()).add(round(ranks[entity][dim], 6))
        assert all(len(found) == 1 for found in by_value.values()), dim
        repeated += sum(1 for v in measured[dim].value_counts() if v > 1)
    assert repeated > 0
