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
                "timestamp",
                "from_bank",
                "from_account",
                "to_bank",
                "to_account",
                "amount_received",
                "receiving_currency",
                "amount_paid",
                "payment_currency",
                "payment_format",
                "is_laundering",
            ],
            skip_rows=1,
        ),
    )
    n_tx = tx_table.num_rows
    tx_pks = pa.array([f"TX-{i:08d}" for i in range(n_tx)])
    tx_table = tx_table.append_column("primary_key", tx_pks)

    amt = pc.cast(tx_table["amount_received"], pa.float64())

    # Group stats are joined back by key. Table.join does not keep row order,
    # so each joined table is sorted back to tx_table order via _row.
    keyed = tx_table.append_column("_row", pa.array(range(n_tx), type=pa.int64()))

    def _group_stats(keys: list[str]) -> pa.Table:
        stats = tx_table.group_by(keys).aggregate(
            [
                ("amount_received", "mean"),
                ("amount_received", "stddev"),
                ("amount_received", "count"),
            ]
        )
        return keyed.join(stats, keys=keys).sort_by("_row")

    def _robust_z(joined: pa.Table) -> pa.Array:
        """(amount - mean) / std, 0 where the group has < 2 rows or ~zero spread.

        A group that repeats one amount has std ~ 0 (float noise); dividing by
        it gives meaningless z-scores, so the z is 0 when std is below
        1e-9 * max(1, |mean|).
        """
        mean = pc.fill_null(pc.cast(joined["amount_received_mean"], pa.float64()), 0.0)
        std = pc.fill_null(pc.cast(joined["amount_received_stddev"], pa.float64()), 0.0)
        count = pc.fill_null(joined["amount_received_count"], 0)
        floor = pc.multiply(pc.max_element_wise(pc.abs(mean), 1.0), 1e-9)
        valid = pc.and_(pc.greater_equal(count, 2), pc.greater(std, floor))
        safe_std = pc.if_else(valid, std, 1.0)
        z = pc.divide(pc.subtract(amt, mean), safe_std)
        return pc.if_else(valid, z, 0.0)

    # --- Sender z-score: (amount - sender_mean) / sender_std ---
    amount_z_sender = _robust_z(_group_stats(["from_account"]))

    # --- Receiver z-score ---
    delta_z_recv = _robust_z(_group_stats(["to_account"]))

    # --- Pair z-score ---
    _joined_p = _group_stats(["from_account", "to_account"])
    amount_z_pair = _robust_z(_joined_p)

    # --- Cross-currency flag ---
    is_cross = pc.and_(
        pc.not_equal(tx_table["receiving_currency"], tx_table["payment_currency"]),
        pc.and_(
            pc.is_valid(tx_table["receiving_currency"]), pc.is_valid(tx_table["payment_currency"])
        ),
    )
    is_cross_currency = pc.cast(is_cross, pa.float64())

    # --- New counterparty: pair count <= 1 ---
    _p_count = pc.fill_null(
        pc.cast(_joined_p["amount_received_count"], pa.float64()),
        0.0,
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
