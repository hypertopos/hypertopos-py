# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Extract unique counterparty banks from trans.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "berka"


def prepare() -> pa.Table:
    opts = pcsv.ParseOptions(delimiter=";")
    trans_df = pcsv.read_csv(DATA_DIR / "trans.csv", parse_options=opts)
    vals = sorted({str(v) for v in trans_df["bank"].to_pylist()
                   if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v)})
    return pa.table({"primary_key": pa.array(vals, type=pa.string())})
