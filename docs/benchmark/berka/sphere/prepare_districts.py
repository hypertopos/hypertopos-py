# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 prepare script — builds districts table from district.csv."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "berka"


def prepare() -> pa.Table:
    opts = pcsv.ParseOptions(delimiter=";")
    df = pcsv.read_csv(DATA_DIR / "district.csv", parse_options=opts)
    return pa.table({
        "primary_key": pa.array([str(df["A1"][i].as_py()) for i in range(df.num_rows)], type=pa.string()),
        "district_id": pa.array([str(df["A1"][i].as_py()) for i in range(df.num_rows)], type=pa.string()),
        "name": pa.array([df["A2"][i].as_py() for i in range(df.num_rows)], type=pa.string()),
        "region": pa.array([df["A3"][i].as_py() for i in range(df.num_rows)], type=pa.string()),
        "population": pa.array([df["A4"][i].as_py() for i in range(df.num_rows)], type=pa.int64()),
        "avg_salary": pa.array([df["A11"][i].as_py() for i in range(df.num_rows)], type=pa.int64()),
    })
