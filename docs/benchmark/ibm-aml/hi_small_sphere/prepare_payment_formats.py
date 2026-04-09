# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Extract unique payment formats from transactions."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ibm-aml"


def prepare() -> pa.Table:
    tx = pcsv.read_csv(
        DATA_DIR / "HI-Small_Trans.csv",
        read_options=pcsv.ReadOptions(
            column_names=["timestamp", "from_bank", "from_account", "to_bank",
                          "to_account", "amount_received", "receiving_currency",
                          "amount_paid", "payment_currency", "payment_format",
                          "is_laundering"],
            skip_rows=1,
        ),
        convert_options=pcsv.ConvertOptions(
            include_columns=["payment_format"],
        ),
    )
    vals = sorted({v for v in tx["payment_format"].to_pylist() if v})
    return pa.table({"primary_key": pa.array(vals, type=pa.string())})