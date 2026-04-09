# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 prepare script — builds orders table from order.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "berka"


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
    order_df = pcsv.read_csv(DATA_DIR / "order.csv", parse_options=opts)
    return pa.table({
        "primary_key": pa.array(
            [f"ORD-{order_df['order_id'][i].as_py()}" for i in range(order_df.num_rows)],
            type=pa.string(),
        ),
        "account_id": pa.array(
            [str(order_df["account_id"][i].as_py()) for i in range(order_df.num_rows)],
            type=pa.string(),
        ),
        "bank_to": _col_to_str(order_df["bank_to"]),
        "k_symbol": _col_to_str(order_df["k_symbol"]),
        "amount": order_df["amount"].cast(pa.float64()),
    })
