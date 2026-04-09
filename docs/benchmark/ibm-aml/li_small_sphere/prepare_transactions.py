# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 — builds enriched transactions table with context features."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ibm-aml"
TRANS_FILE = DATA_DIR / "LI-Small_Trans.csv"


def prepare() -> pa.Table:
    tx_table = pcsv.read_csv(
        TRANS_FILE,
        read_options=pcsv.ReadOptions(
            column_names=[
                "timestamp", "from_bank", "from_account", "to_bank",
                "to_account", "amount_received", "receiving_currency",
                "amount_paid", "payment_currency", "payment_format",
                "is_laundering",
            ],
            skip_rows=1,
        ),
    )
    n_tx = tx_table.num_rows
    tx_pks = pa.array([f"TX-{i:08d}" for i in range(n_tx)])
    tx_table = tx_table.append_column("primary_key", tx_pks)

    amt = pc.cast(tx_table["amount_received"], pa.float64())

    # --- Sender z-score: (amount - sender_mean) / sender_std ---
    sender_stats = tx_table.group_by("from_account").aggregate([
        ("amount_received", "mean"), ("amount_received", "stddev"),
    ])
    # Join stats back to tx_table via from_account
    _joined = tx_table.join(sender_stats, "from_account")
    _s_mean = pc.fill_null(pc.cast(_joined["amount_received_mean"], pa.float64()), 0.0)
    _s_std = pc.fill_null(pc.cast(_joined["amount_received_stddev"], pa.float64()), 1.0)
    _s_std = pc.max_element_wise(_s_std, pa.scalar(1e-6, pa.float64()))
    amount_z_sender = pc.divide(pc.subtract(amt, _s_mean), _s_std)

    # --- Receiver z-score ---
    recv_stats = tx_table.group_by("to_account").aggregate([
        ("amount_received", "mean"), ("amount_received", "stddev"),
    ])
    _joined_r = tx_table.join(recv_stats, keys="to_account")
    _r_mean = pc.fill_null(pc.cast(_joined_r["amount_received_mean"], pa.float64()), 0.0)
    _r_std = pc.fill_null(pc.cast(_joined_r["amount_received_stddev"], pa.float64()), 1.0)
    _r_std = pc.max_element_wise(_r_std, pa.scalar(1e-6, pa.float64()))
    delta_z_recv = pc.divide(pc.subtract(amt, _r_mean), _r_std)

    # --- Pair z-score ---
    pair_stats = tx_table.group_by(["from_account", "to_account"]).aggregate([
        ("amount_received", "mean"), ("amount_received", "stddev"),
        ("amount_received", "count"),
    ])
    _joined_p = tx_table.join(pair_stats, keys=["from_account", "to_account"])
    _p_mean = pc.fill_null(pc.cast(_joined_p["amount_received_mean"], pa.float64()), 0.0)
    _p_std = pc.fill_null(pc.cast(_joined_p["amount_received_stddev"], pa.float64()), 1.0)
    _p_std = pc.max_element_wise(_p_std, pa.scalar(1e-6, pa.float64()))
    amount_z_pair = pc.divide(pc.subtract(amt, _p_mean), _p_std)

    # --- Cross-currency flag ---
    is_cross = pc.and_(
        pc.not_equal(tx_table["receiving_currency"], tx_table["payment_currency"]),
        pc.and_(pc.is_valid(tx_table["receiving_currency"]),
                pc.is_valid(tx_table["payment_currency"])),
    )
    is_cross_currency = pc.cast(is_cross, pa.float64())

    # --- New counterparty: pair count <= 1 ---
    _p_count = pc.fill_null(
        pc.cast(_joined_p["amount_received_count"], pa.float64()), 0.0,
    )
    is_new_counterparty = pc.cast(pc.less_equal(_p_count, 1.0), pa.float64())

    # --- Time interval ratio: 1.0 placeholder ---
    # Full per-sender gap tracking handled by sphere's iet_mean/iet_std dims.
    time_interval_ratio = pa.array([1.0] * n_tx, type=pa.float64())

    tx_table = tx_table.append_column("amount_z_sender", amount_z_sender)
    tx_table = tx_table.append_column("delta_z_recv", delta_z_recv)
    tx_table = tx_table.append_column("amount_z_pair", amount_z_pair)
    tx_table = tx_table.append_column("is_cross_currency", is_cross_currency)
    tx_table = tx_table.append_column("is_new_counterparty", is_new_counterparty)
    tx_table = tx_table.append_column("time_interval_ratio", time_interval_ratio)

    return tx_table