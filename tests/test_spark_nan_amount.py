# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""A NaN amount is a missing amount: the event counts, its amount does not, on every path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from hypertopos.scoring import STATE_COLUMNS  # noqa: E402
from hypertopos.spark import compare, find_anomalies, score, sphere_of  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}


def test_a_nan_amount_leaves_finite_states_and_a_sphere_that_builds(spark, tmp_path):
    events = spark.range(12_000).select(
        F.concat(F.lit("E"), (F.col("id") % 1200).cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 1200).cast("string")).alias("dst"),
        F.when(F.col("id") % 1200 < 40, F.lit(float("nan")))
        .otherwise(F.exp(F.randn(seed=11)) * 100)
        .alias("amt"),
        F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
    )
    result = find_anomalies(events, **ROLES, sphere_root=str(tmp_path)).cache()
    sphere = sphere_of(result)
    assert "current" in sphere.calibration().reference.axes
    table = sphere.entities()
    for column in STATE_COLUMNS:
        assert np.isfinite(table[column].to_numpy(zero_copy_only=False).astype(float)).all(), column
    p_values = [r["p_value"] for r in result.select("p_value").collect()]
    assert np.isfinite(p_values).all()
    assert sphere.lance_path()  # the sphere build reads the same states


def test_compare_names_score_when_a_side_lost_its_calibration(spark, tmp_path):
    events = spark.range(6_000).select(
        F.concat(F.lit("E"), (F.col("id") % 600).cast("string")).alias("src"),
        (F.exp(F.randn(seed=3)) * 100).alias("amt"),
    )
    like = sphere_of(find_anomalies(events, entity="src", amount="amt", sphere_root=str(tmp_path)))
    scored = score(events, like=like)
    with pytest.raises(ValueError, match=r"score\(\) results"):
        compare(scored, scored.withColumn("entity", F.concat("entity", F.lit(""))))


def test_a_nan_amount_gives_the_entity_table_a_null_amount_gives(spark, tmp_path):
    def table(missing):
        events = spark.range(6_000).select(
            F.concat(F.lit("E"), (F.col("id") % 600).cast("string")).alias("src"),
            F.concat(F.lit("E"), ((F.col("id") * 7919 + 13) % 600).cast("string")).alias("dst"),
            F.when(F.col("id") % 97 == 0, missing)
            .otherwise(F.exp(F.randn(seed=5)) * 100)
            .alias("amt"),
            F.timestamp_seconds(F.lit(1767225600) + (F.col("id") * 7907) % 5_184_000).alias("ts"),
        )
        root = tmp_path / ("nan" if missing is not None and "NaN" in str(missing) else "null")
        result = find_anomalies(events, **ROLES, sphere_root=str(root))
        return sphere_of(result).entities().sort_by("entity")

    with_nan = table(F.lit(float("nan")))
    with_null = table(F.lit(None).cast("double"))
    assert with_nan.column_names == with_null.column_names
    for name in with_nan.column_names:
        a, b = with_nan[name].to_pylist(), with_null[name].to_pylist()
        if name == "entity":
            assert a == b
        else:
            np.testing.assert_array_equal(
                np.array(a, dtype=float), np.array(b, dtype=float), err_msg=name
            )
