# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 prepare script — builds transactions table from trans.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "berka"


def _parse_berka_date(date_col: pa.Array) -> pa.Array:
    dates = []
    for v in date_col.to_pylist():
        if v is None:
            dates.append(None)
            continue
        s = str(int(v))
        yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
        dates.append(f"{1900 + yy}-{mm:02d}-{dd:02d}")
    return pa.array(dates, type=pa.string())


def _col_to_str(col: pa.Array) -> pa.Array:
    vals = []
    for v in col.to_pylist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append("")
        else:
            vals.append(str(v))
    return pa.array(vals, type=pa.string())


def prepare() -> pa.Table:
    opts = pcsv.ParseOptions(delimiter=";")
    trans_df = pcsv.read_csv(DATA_DIR / "trans.csv", parse_options=opts)
    n = trans_df.num_rows
    return pa.table({
        "primary_key": pa.array([f"TX-{i:07d}" for i in range(n)], type=pa.string()),
        "account_id": pa.array([str(v) for v in trans_df["account_id"].to_pylist()], type=pa.string()),
        "date": _parse_berka_date(trans_df["date"]),
        "type": _col_to_str(trans_df["type"]),
        "operation": _col_to_str(trans_df["operation"]),
        "amount": trans_df["amount"].cast(pa.float64()),
        "balance": trans_df["balance"].cast(pa.float64()),
        "k_symbol": _col_to_str(trans_df["k_symbol"]),
        "bank": _col_to_str(trans_df["bank"]),
    })
