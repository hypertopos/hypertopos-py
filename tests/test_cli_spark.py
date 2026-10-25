# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``hypertopos spark``: CLI front of the Spark API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hypertopos.cli.main import main
from hypertopos.cli.spark_cmd import read_input


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["hypertopos", *argv])
    main()


def test_root_help_lists_spark(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "--help")
    assert "spark" in capsys.readouterr().out


def test_find_anomalies_help_names_the_roles_and_the_docs(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "spark", "find-anomalies", "--help")
    text = capsys.readouterr().out
    for flag in ("--entity", "--counterparty", "--amount", "--time", "--group-by", "--show"):
        assert flag in text
    assert "docs/spark.md" in text


def test_an_output_is_required(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "spark", "find-anomalies", "--input", "t", "--entity", "src")
    assert "--output" in capsys.readouterr().err


class _Reader:
    def __init__(self, calls: list[tuple[str, ...]]) -> None:
        self.calls = calls

    def format(self, name: str) -> _Reader:
        self.calls.append(("format", name))
        return self

    def load(self, path: str) -> str:
        self.calls.append(("load", path))
        return "delta"

    def csv(self, path: str, **_: object) -> str:
        return "csv"

    def parquet(self, path: str) -> str:
        return "parquet"


class _Spark:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.read = _Reader(self.calls)

    def table(self, name: str) -> str:
        return f"table:{name}"


def test_read_input_routes_by_source(tmp_path):
    spark = _Spark()
    assert read_input(spark, "main.aml.transactions") == "table:main.aml.transactions"
    delta = tmp_path / "delta_tx"
    (delta / "_delta_log").mkdir(parents=True)
    assert read_input(spark, str(delta)) == "delta"
    csv = tmp_path / "tx.csv"
    csv.write_text("a\n1\n")
    assert read_input(spark, str(csv)) == "csv"
    parquet = tmp_path / "tx.parquet"
    parquet.write_bytes(b"")
    assert read_input(spark, str(parquet)) == "parquet"
    assert read_input(spark, "/Volumes/main/aml/raw/tx") == "parquet"


@pytest.mark.skipif(sys.platform == "win32", reason="Spark file writes need winutils on Windows")
def test_find_anomalies_writes_parquet(spark, monkeypatch, tmp_path):
    pytest.importorskip("pyspark")
    from pyspark.sql import functions as F

    spark.range(2000).select(
        F.concat(F.lit("E"), (F.col("id") % 200).cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") * 31 + 7) % 200).cast("string")).alias("dst"),
    ).createOrReplaceTempView("cli_tx_out")
    out = tmp_path / "result.parquet"
    _run(
        monkeypatch,
        "spark",
        "find-anomalies",
        "--input",
        "cli_tx_out",
        "--entity",
        "src",
        "--counterparty",
        "dst",
        "--sphere-root",
        str(tmp_path / "s"),
        "--output",
        str(out),
    )
    written = spark.read.parquet(str(out))
    assert written.count() == 200
    assert "top_reasons" in written.columns


def test_find_anomalies_on_a_view_prints_the_top_entities(spark, monkeypatch, capsys, tmp_path):
    pytest.importorskip("pyspark")
    from pyspark.sql import functions as F

    spark.range(3000).select(
        F.concat(F.lit("E"), (F.col("id") % 300).cast("string")).alias("src"),
        F.concat(F.lit("E"), ((F.col("id") * 31 + 7) % 300).cast("string")).alias("dst"),
        (F.exp(F.randn(seed=5)) * 100).alias("amt"),
    ).createOrReplaceTempView("cli_tx")
    _run(
        monkeypatch,
        "spark",
        "find-anomalies",
        "--input",
        "cli_tx",
        "--entity",
        "src",
        "--counterparty",
        "dst",
        "--amount",
        "amt",
        "--sphere-root",
        str(Path(tmp_path)),
        "--show",
        "5",
    )
    out = capsys.readouterr().out
    assert "delta_norm" in out
    assert "top_reasons" in out
