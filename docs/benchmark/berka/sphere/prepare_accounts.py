# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 prepare script — builds enriched accounts table from 7 Berka CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "berka"


def _decode_birth_number(bn: int) -> tuple[int, str]:
    s = str(bn)
    yy, mm = int(s[:2]), int(s[2:4])
    gender = "F" if mm > 12 else "M"
    return 1900 + yy, gender


def prepare() -> pa.Table:
    opts = pcsv.ParseOptions(delimiter=";")
    account_df = pcsv.read_csv(DATA_DIR / "account.csv", parse_options=opts)
    client_df = pcsv.read_csv(DATA_DIR / "client.csv", parse_options=opts)
    disp_df = pcsv.read_csv(DATA_DIR / "disp.csv", parse_options=opts)
    loan_df = pcsv.read_csv(DATA_DIR / "loan.csv", parse_options=opts)
    card_df = pcsv.read_csv(DATA_DIR / "card.csv", parse_options=opts)
    order_df = pcsv.read_csv(DATA_DIR / "order.csv", parse_options=opts)
    district_df = pcsv.read_csv(DATA_DIR / "district.csv", parse_options=opts)

    acct_map: dict[int, dict] = {}
    for i in range(account_df.num_rows):
        aid = account_df["account_id"][i].as_py()
        acct_map[aid] = {
            "frequency": account_df["frequency"][i].as_py(),
            "district_id": account_df["district_id"][i].as_py(),
        }

    dist_region = {district_df["A1"][i].as_py(): district_df["A3"][i].as_py()
                   for i in range(district_df.num_rows)}
    for info in acct_map.values():
        info["region"] = dist_region.get(info["district_id"], "unknown")

    disp_by_account: dict[int, list[dict]] = {}
    for i in range(disp_df.num_rows):
        aid = disp_df["account_id"][i].as_py()
        disp_by_account.setdefault(aid, []).append({
            "disp_id": disp_df["disp_id"][i].as_py(),
            "client_id": disp_df["client_id"][i].as_py(),
            "type": disp_df["type"][i].as_py(),
        })

    client_info = {}
    for i in range(client_df.num_rows):
        cid = client_df["client_id"][i].as_py()
        year, gender = _decode_birth_number(int(client_df["birth_number"][i].as_py()))
        client_info[cid] = {"birth_year": year, "gender": gender}

    for aid, info in acct_map.items():
        disps = disp_by_account.get(aid, [])
        info["n_clients"] = len(disps)
        owner = next((d for d in disps if d["type"] == "OWNER"), None)
        if owner and owner["client_id"] in client_info:
            ci = client_info[owner["client_id"]]
            info["owner_gender"] = ci["gender"]
            info["owner_birth_year"] = ci["birth_year"]
        else:
            info["owner_gender"] = None
            info["owner_birth_year"] = 0

    loan_by_account = {}
    for i in range(loan_df.num_rows):
        aid = loan_df["account_id"][i].as_py()
        loan_by_account[aid] = {
            "status": loan_df["status"][i].as_py(),
            "amount": loan_df["amount"][i].as_py(),
            "duration": loan_df["duration"][i].as_py(),
        }
    for aid, info in acct_map.items():
        loan = loan_by_account.get(aid)
        if loan:
            info["has_loan"] = "yes"
            info["loan_status"] = loan["status"]
            info["loan_amount"] = loan["amount"]
            info["loan_duration"] = loan["duration"]
        else:
            info["has_loan"] = None
            info["loan_status"] = None
            info["loan_amount"] = 0
            info["loan_duration"] = 0

    card_by_disp = {card_df["disp_id"][i].as_py(): card_df["type"][i].as_py()
                    for i in range(card_df.num_rows)}
    for aid, info in acct_map.items():
        disps = disp_by_account.get(aid, [])
        card_type = None
        for d in disps:
            if d["disp_id"] in card_by_disp:
                card_type = card_by_disp[d["disp_id"]]
                break
        info["has_card"] = "yes" if card_type else None
        info["card_type"] = card_type

    order_count: dict[int, int] = {}
    for i in range(order_df.num_rows):
        aid = order_df["account_id"][i].as_py()
        order_count[aid] = order_count.get(aid, 0) + 1
    for aid, info in acct_map.items():
        info["n_standing_orders"] = order_count.get(aid, 0)

    # --- Credit risk dimensions (precomputed from transactions) ---
    trans_df = pcsv.read_csv(DATA_DIR / "trans.csv", parse_options=opts)
    # Group transactions by account
    acct_balances: dict[int, list[float]] = {}
    acct_incomes: dict[int, float] = {}
    acct_months: dict[int, set[str]] = {}
    acct_penalties: dict[int, int] = {}
    for i in range(trans_df.num_rows):
        aid = trans_df["account_id"][i].as_py()
        bal = trans_df["balance"][i].as_py()
        amt = trans_df["amount"][i].as_py()
        tx_type = str(trans_df["type"][i].as_py()).strip('"')
        k_symbol = str(trans_df["k_symbol"][i].as_py() or "").strip('"').strip()
        date_raw = str(int(trans_df["date"][i].as_py()))
        month_key = date_raw[:4]  # YYMM

        acct_balances.setdefault(aid, []).append(float(bal))
        acct_months.setdefault(aid, set()).add(month_key)
        if tx_type == "PRIJEM":  # income/credit
            acct_incomes[aid] = acct_incomes.get(aid, 0.0) + float(amt)
        if k_symbol == "SANKC. UROK":  # penalty interest
            acct_penalties[aid] = acct_penalties.get(aid, 0) + 1

    for aid, info in acct_map.items():
        balances = acct_balances.get(aid, [0.0])
        loan_amt = info["loan_amount"]
        n_months = max(len(acct_months.get(aid, {1})), 1)
        total_income = acct_incomes.get(aid, 0.0)
        monthly_income = total_income / n_months

        # balance_to_loan: mean_balance / loan_amount (0 if no loan)
        mean_bal = float(np.mean(balances))
        info["balance_to_loan"] = mean_bal / loan_amt if loan_amt > 0 else 0.0

        # min_balance: lowest balance ever seen
        info["min_balance"] = float(np.min(balances))

        # balance_volatility: std(balance) / mean(balance) — coefficient of variation
        std_bal = float(np.std(balances))
        info["balance_volatility"] = std_bal / mean_bal if mean_bal > 0 else 0.0

        # income_coverage: monthly_income / monthly_payment (0 if no loan)
        if loan_amt > 0 and info["loan_duration"] > 0:
            monthly_payment = loan_amt / info["loan_duration"]
            info["income_coverage"] = monthly_income / monthly_payment if monthly_payment > 0 else 0.0
        else:
            info["income_coverage"] = 0.0

        # balance_trend: sign of (last 10% of balances mean - first 10% mean)
        n = len(balances)
        if n >= 10:
            first_chunk = balances[:max(n // 10, 1)]
            last_chunk = balances[-max(n // 10, 1):]
            info["balance_trend"] = float(np.mean(last_chunk) - np.mean(first_chunk))
        else:
            info["balance_trend"] = 0.0

        # penalty_interest_count: number of SANKC. UROK transactions
        info["penalty_interest_count"] = acct_penalties.get(aid, 0)

    sorted_aids = sorted(acct_map.keys())
    return pa.table({
        "primary_key": pa.array([str(a) for a in sorted_aids], type=pa.string()),
        "account_id": pa.array([str(a) for a in sorted_aids], type=pa.string()),
        "frequency": pa.array([acct_map[a]["frequency"] for a in sorted_aids], type=pa.string()),
        "district_id": pa.array([str(acct_map[a]["district_id"]) for a in sorted_aids], type=pa.string()),
        "region": pa.array([acct_map[a]["region"] for a in sorted_aids], type=pa.string()),
        "has_loan": pa.array([acct_map[a]["has_loan"] for a in sorted_aids], type=pa.string()),
        "loan_status": pa.array([acct_map[a]["loan_status"] for a in sorted_aids], type=pa.string()),
        "has_card": pa.array([acct_map[a]["has_card"] for a in sorted_aids], type=pa.string()),
        "card_type": pa.array([acct_map[a]["card_type"] for a in sorted_aids], type=pa.string()),
        "owner_gender": pa.array([acct_map[a]["owner_gender"] for a in sorted_aids], type=pa.string()),
        "owner_birth_year": pa.array([acct_map[a]["owner_birth_year"] for a in sorted_aids], type=pa.int64()),
        "n_clients": pa.array([acct_map[a]["n_clients"] for a in sorted_aids], type=pa.int64()),
        "n_standing_orders": pa.array([acct_map[a]["n_standing_orders"] for a in sorted_aids], type=pa.int64()),
        "loan_amount": pa.array([acct_map[a]["loan_amount"] for a in sorted_aids], type=pa.int64()),
        "loan_duration": pa.array([acct_map[a]["loan_duration"] for a in sorted_aids], type=pa.int64()),
        "balance_to_loan": pa.array([acct_map[a]["balance_to_loan"] for a in sorted_aids], type=pa.float64()),
        "min_balance": pa.array([acct_map[a]["min_balance"] for a in sorted_aids], type=pa.float64()),
        "balance_volatility": pa.array([acct_map[a]["balance_volatility"] for a in sorted_aids], type=pa.float64()),
        "income_coverage": pa.array([acct_map[a]["income_coverage"] for a in sorted_aids], type=pa.float64()),
        "balance_trend": pa.array([acct_map[a]["balance_trend"] for a in sorted_aids], type=pa.float64()),
        "penalty_interest_count": pa.array([acct_map[a]["penalty_interest_count"] for a in sorted_aids], type=pa.int64()),
    })
