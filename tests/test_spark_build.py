# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Entity table hand-off and sphere build (hypertopos.spark._build)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

pytest.importorskip("pyspark")

from hypertopos.cli.build import build_from_config  # noqa: E402
from hypertopos.cli.schema import parse_config  # noqa: E402
from hypertopos.scoring import FrozenCalibration  # noqa: E402
from hypertopos.spark._build import (  # noqa: E402
    CALIBRATION_GROUP,
    LINE_ID,
    PATTERN_ID,
    build_sphere,
    collect_entities,
    estimate_bytes,
    fingerprint,
    sphere_dir,
)
from hypertopos.spark._plan import aggregate, build_plan  # noqa: E402
from hypertopos.storage.reader import GDSReader  # noqa: E402

ROLES = {"entity": "src", "counterparty": "dst", "amount": "amt", "time": "ts"}
SCHEMA = "src string, dst string, amt double, ts timestamp"


def _events(n_entities: int = 60, n_events: int = 900, seed: int = 7) -> list[tuple]:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for _ in range(n_events):
        src = f"E{int(rng.integers(n_entities)):03d}"
        dst = f"E{int(rng.integers(n_entities)):03d}"
        amt = float(np.round(rng.lognormal(5, 1.2), 2))
        ts = start + timedelta(seconds=int(rng.integers(0, 60 * 86400)))
        rows.append((src, dst, amt, ts))
    return rows


@pytest.fixture(scope="module")
def events(spark):
    df = spark.createDataFrame(_events(), SCHEMA).cache()
    df.count()
    return df


@pytest.fixture(scope="module")
def plan():
    return build_plan(**ROLES)


@pytest.fixture(scope="module")
def entity_df(events, plan):
    df = aggregate(events, plan).cache()
    df.count()
    return df


def _calibration(table: pa.Table, plan) -> FrozenCalibration:
    names = [d.name for d in plan.dims] + [name for name, _ in plan.extra_dims]
    return FrozenCalibration.fit(table, names)


def _geometry(path: Path) -> dict[str, float]:
    t = lance.dataset(str(path / "geometry" / PATTERN_ID / "data.lance")).to_table(
        columns=["primary_key", "delta_norm"]
    )
    return dict(zip(t["primary_key"].to_pylist(), t["delta_norm"].to_pylist(), strict=True))


def test_collect_entities_returns_one_row_per_entity(entity_df, tmp_path):
    table = collect_entities(entity_df)
    assert table.num_rows == entity_df.count()
    assert table.column_names == entity_df.columns
    assert pa.types.is_string(table.schema.field("entity").type)


def test_large_entity_table_comes_over_in_chunks(entity_df):
    # Above the collect limit the table arrives in hash-split chunks: nothing is
    # written to storage, and the result equals a single collect.
    direct = collect_entities(entity_df)
    size = estimate_bytes(direct.num_rows, len(entity_df.columns) - 1)
    chunked = collect_entities(entity_df, max_collect_bytes=size // 4)
    assert chunked.sort_by("entity").equals(direct.sort_by("entity"))


def test_estimate_bytes_counts_numbers_and_key():
    assert estimate_bytes(1_000_000, 23) == 1_000_000 * (8 * 23 + 32)


def test_spark_path_matches_file_path(entity_df, plan, tmp_path):
    """The Spark hand-off and the YAML/CLI path give the same geometry."""
    spark_table = collect_entities(entity_df)
    build_sphere(spark_table, _calibration(spark_table, plan), str(tmp_path / "spark_sphere"))

    # File path: an independent hand-off (pandas -> Parquet) declared in YAML.
    pq.write_table(
        pa.Table.from_pandas(entity_df.toPandas(), preserve_index=False),
        tmp_path / "entities.parquet",
    )
    config = {
        "version": "0.1.0",
        "sphere_id": "spark",
        "sources": {"entities": {"path": "entities.parquet"}},
        "lines": {LINE_ID: {"source": "entities", "key": "entity", "role": "anchor"}},
        "patterns": {
            PATTERN_ID: {
                "type": "anchor",
                "entity_line": LINE_ID,
                "anomaly_percentile": 99.0,
                "metric_properties": [d.name for d in plan.dims],
            }
        },
    }
    (tmp_path / "sphere.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    build_from_config(
        parse_config(tmp_path / "sphere.yaml"), tmp_path / "file_sphere", source_dir=tmp_path
    )

    spark_geo = _geometry(tmp_path / "spark_sphere")
    file_geo = _geometry(tmp_path / "file_sphere")
    assert spark_geo.keys() == file_geo.keys()
    for key, norm in spark_geo.items():
        assert norm == pytest.approx(file_geo[key], abs=1e-6), key

    a = GDSReader(str(tmp_path / "spark_sphere")).read_sphere().patterns[PATTERN_ID]
    b = GDSReader(str(tmp_path / "file_sphere")).read_sphere().patterns[PATTERN_ID]
    assert a.dimension_kinds == b.dimension_kinds
    np.testing.assert_allclose(a.mu, b.mu, atol=1e-6)
    np.testing.assert_allclose(a.sigma_diag, b.sigma_diag, atol=1e-6)


def test_sphere_labels_are_the_catalogue_names(entity_df, plan, tmp_path):
    table = collect_entities(entity_df)
    build_sphere(table, _calibration(table, plan), str(tmp_path / "s"))
    pattern = GDSReader(str(tmp_path / "s")).read_sphere().patterns[PATTERN_ID]
    assert pattern.dim_labels == [f"t:{d.name}" for d in plan.dims]


def test_group_by_builds_one_population_per_group(entity_df, plan, tmp_path):
    table = collect_entities(entity_df)
    segment = pa.array(
        ["even" if int(k[1:]) % 2 == 0 else "odd" for k in table["entity"].to_pylist()]
    )
    table = table.append_column("segment", segment)
    calibration = FrozenCalibration.fit(
        table, [d.name for d in plan.dims], group_by="segment", min_group_size=10
    )
    build_sphere(table, calibration, str(tmp_path / "s"))
    pattern = GDSReader(str(tmp_path / "s")).read_sphere().patterns[PATTERN_ID]
    assert pattern.group_by_property == CALIBRATION_GROUP
    assert set(pattern.group_stats or {}) == {"even", "odd"}


def test_extra_dims_become_dimensions(events, tmp_path):
    plan = build_plan(
        **ROLES,
        extra_dims={"big_out": "sum(CASE WHEN side = 'out' AND amount > 500 THEN 1 ELSE 0 END)"},
    )
    table = collect_entities(aggregate(events, plan))
    build_sphere(table, _calibration(table, plan), str(tmp_path / "s"))
    pattern = GDSReader(str(tmp_path / "s")).read_sphere().patterns[PATTERN_ID]
    assert pattern.dim_labels[-1] == "t:big_out"


def test_fingerprint_of_in_memory_frame_is_per_object(spark, plan):
    # Spark's canonical plan of an in-memory frame does not include its rows, so
    # two frames with different rows must never share a fingerprint.
    a = spark.createDataFrame([("A", "B", 1.0, datetime(2026, 1, 1))], SCHEMA)
    b = spark.createDataFrame([("A", "B", 2.0, datetime(2026, 1, 1))], SCHEMA)
    base = fingerprint(a, plan, percentile=99.0, group_by=None, min_group_size=100)
    assert base == fingerprint(a, plan, percentile=99.0, group_by=None, min_group_size=100)
    assert base != fingerprint(b, plan, percentile=99.0, group_by=None, min_group_size=100)
    assert base != fingerprint(a, plan, percentile=95.0, group_by=None, min_group_size=100)
    assert base != fingerprint(a, plan, percentile=99.0, group_by="segment", min_group_size=100)
    assert base != fingerprint(
        a, build_plan(entity="src"), percentile=99.0, group_by=None, min_group_size=100
    )
    # A different pooling of small groups is a different calibration.
    assert base != fingerprint(a, plan, percentile=99.0, group_by=None, min_group_size=10)


@pytest.mark.skipif(sys.platform == "win32", reason="Spark file reads need winutils on Windows")
def test_fingerprint_of_file_source_follows_its_files(spark, plan, tmp_path):
    rows = pa.table({"src": ["A"], "dst": ["B"], "amt": [1.0], "ts": [datetime(2026, 1, 1)]})
    v1 = tmp_path / "v1"
    v1.mkdir()
    pq.write_table(rows, v1 / "part-0.parquet")
    first = fingerprint(
        spark.read.parquet(str(v1)), plan, percentile=99.0, group_by=None, min_group_size=100
    )
    again = fingerprint(
        spark.read.parquet(str(v1)), plan, percentile=99.0, group_by=None, min_group_size=100
    )
    assert first == again  # a re-read of the same files reuses the sphere
    pq.write_table(rows, v1 / "part-1.parquet")  # a new table version adds files
    grown = fingerprint(
        spark.read.parquet(str(v1)), plan, percentile=99.0, group_by=None, min_group_size=100
    )
    assert grown != first


def test_sphere_dir_is_short():
    assert sphere_dir("/tmp/hypertopos", "a" * 64) == Path("/tmp/hypertopos") / ("a" * 16)
