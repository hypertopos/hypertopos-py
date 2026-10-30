# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``hypertopos spark``: the Spark API from the command line (local, Databricks wheel tasks)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DOCS = "https://github.com/hypertopos/hypertopos-py/blob/main/docs/spark.md"


def add_parser(sub: Any) -> argparse.ArgumentParser:
    spark_p: argparse.ArgumentParser = sub.add_parser(
        "spark",
        help="Score entities of an event table on Spark (needs hypertopos[spark])",
        description=(
            "Run the hypertopos Spark API from the command line: locally on a "
            "local Spark session, or as a Databricks Python wheel task on the "
            f"cluster's session. Docs: {DOCS}"
        ),
    )
    spark_sub = spark_p.add_subparsers(dest="spark_command", metavar="subcommand")
    fa = spark_sub.add_parser(
        "find-anomalies",
        help="One row per entity: distance from its population, flag, p-value, reasons",
        description=(
            "Aggregate the events per entity on Spark, calibrate the population and "
            "score every entity. Give the column roles; only --entity is required."
        ),
        epilog=(
            "example:\n  hypertopos spark find-anomalies --input tx.parquet --entity from_account "
            "\\\n      --counterparty to_account --amount amount --time ts --show 20\n\n"
            f"docs: {DOCS}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fa.add_argument(
        "--input",
        required=True,
        help="Parquet/CSV file or directory, Delta directory, or a table / view name",
    )
    fa.add_argument("--entity", required=True, help="column of the entity each event belongs to")
    fa.add_argument("--counterparty", help="column of the other side of the event")
    fa.add_argument("--amount", help="numeric column with the size of the event")
    fa.add_argument("--time", help="timestamp, timestamp_ntz, date or epoch-seconds column")
    fa.add_argument("--group-by", help="entity property column: one population per group")
    fa.add_argument(
        "--percentile", type=float, default=99.0, help="flag above this percentile (99)"
    )
    fa.add_argument("--dims", help="comma-separated dimension names (default: auto)")
    fa.add_argument(
        "--score-axes",
        help="comma-separated score axes: mahal or mahal,current "
        "(default: mahal,current with --time, else mahal)",
    )
    fa.add_argument("--sphere-root", help="where calibrations are kept (default: temp dir)")
    fa.add_argument("--no-cache", action="store_true", help="recalibrate even if cached")
    out = fa.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", help="write the result as Parquet to this path")
    out.add_argument("--output-table", help="write the result to this table (saveAsTable)")
    out.add_argument(
        "--show", type=int, metavar="N", help="print the N most distant entities instead"
    )
    sc = spark_sub.add_parser(
        "score",
        help="Score new events against a saved calibration (no recalibration)",
        description=(
            "Aggregate new events per entity and score them against the population a "
            "find-anomalies run calibrated, as a scheduled job does every day. "
            "find-anomalies prints the calibration path; give it with --calibration."
        ),
        epilog=(
            "example:\n  hypertopos spark score --input main.aml.transactions_today "
            "\\\n      --calibration /Volumes/main/aml/hypertopos/1bb767d5fc6a53ec "
            "\\\n      --anomalies-only --append --output-table main.aml.daily_alerts\n\n"
            f"docs: {DOCS}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sc.add_argument(
        "--input",
        required=True,
        help="Parquet/CSV file or directory, Delta directory, or a table / view name",
    )
    sc.add_argument(
        "--calibration", required=True, help="calibration path printed by find-anomalies"
    )
    sc.add_argument("--anomalies-only", action="store_true", help="keep flagged entities only")
    sc.add_argument("--append", action="store_true", help="append to the output, not overwrite")
    sc_out = sc.add_mutually_exclusive_group(required=True)
    sc_out.add_argument("--output", help="write the result as Parquet to this path")
    sc_out.add_argument("--output-table", help="write the result to this table (saveAsTable)")
    sc_out.add_argument(
        "--show", type=int, metavar="N", help="print the N most distant entities instead"
    )
    return spark_p


def run(args: argparse.Namespace, spark_p: argparse.ArgumentParser) -> None:
    if args.spark_command not in ("find-anomalies", "score"):
        spark_p.print_help()
        sys.exit(1)
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        print('PySpark is not installed: pip install "hypertopos[spark]"', file=sys.stderr)
        sys.exit(2)
    from hypertopos.spark import find_anomalies, score, sphere_of

    spark = SparkSession.builder.appName("hypertopos").getOrCreate()
    if args.spark_command == "score":
        result = score(read_input(spark, args.input), like=sphere_of(args.calibration))
        if args.anomalies_only:
            result = result.where("is_anomaly")
        _write(result, args, "append" if args.append else "overwrite")
        return
    result = find_anomalies(
        read_input(spark, args.input),
        entity=args.entity,
        counterparty=args.counterparty,
        amount=args.amount,
        time=args.time,
        group_by=args.group_by,
        percentile=args.percentile,
        dims=args.dims.split(",") if args.dims else "auto",
        score_axes=tuple(args.score_axes.split(",")) if args.score_axes else None,
        sphere_root=args.sphere_root,
        cache=not args.no_cache,
    )
    # A scheduled score job reads the calibration from this path.
    print(f"calibration: {sphere_of(result).path}")
    _write(result, args, "overwrite")


def _write(result: Any, args: argparse.Namespace, mode: str) -> None:
    if args.show is not None:
        result.orderBy("anomaly_score", ascending=False).show(args.show, truncate=False)
    elif args.output_table:
        result.write.mode(mode).saveAsTable(args.output_table)
    else:
        result.write.mode(mode).parquet(args.output)


def read_input(spark: Any, source: str) -> Any:
    """A path is read by its format (Delta directory, CSV, else Parquet); a name is a table."""
    path = Path(source)
    if not path.exists() and "://" not in source and not source.startswith("/Volumes/"):
        return spark.table(source)
    if (path / "_delta_log").is_dir():
        return spark.read.format("delta").load(source)
    if source.lower().endswith(".csv"):
        return spark.read.csv(source, header=True, inferSchema=True)
    return spark.read.parquet(source)
