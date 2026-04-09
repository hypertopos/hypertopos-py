# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tier 3 — builds enriched accounts table with graph features + ratio columns."""
from __future__ import annotations

import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ibm-aml"
TRANS_FILE = DATA_DIR / "HI-Small_Trans.csv"
ACCTS_FILE = DATA_DIR / "HI-Small_accounts.csv"


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

    # Bulk extract — single to_pydict instead of 3x to_pylist
    _bulk = tx_table.select(["from_account", "to_account", "amount_received"]).to_pydict()
    _from_col = _bulk["from_account"]
    _to_col = _bulk["to_account"]
    _amounts = _bulk["amount_received"]

    # Unique accounts — set union instead of list concat
    all_accts = sorted(set(_from_col) | set(_to_col) - {None})

    # Load account metadata — vectorized via to_pydict
    acct_meta = pcsv.read_csv(ACCTS_FILE)
    _meta = acct_meta.to_pydict()
    acct_dict: dict[str, dict] = {}
    for acct_id, entity_name, bank_id in zip(
        _meta["Account Number"], _meta["Entity Name"], _meta["Bank ID"],
    ):
        aid = str(acct_id or "")
        ename = str(entity_name or "")
        entity_type = ename.rsplit(" #", 1)[0] if " #" in ename else ename
        acct_dict[aid] = {
            "bank_id": str(bank_id or ""),
            "entity_type": entity_type,
        }

    # Per-account outgoing/incoming stats — Arrow groupby, dict via to_pydict
    out_grouped = tx_table.group_by("from_account").aggregate([
        ("to_account", "count_distinct"),
        ("to_bank", "count_distinct"),
        ("receiving_currency", "count_distinct"),
        ("amount_received", "count"),
    ])
    _og = out_grouped.to_pydict()
    out_map: dict[str, dict] = {}
    for k, cnt, tgt, bnk, cur in zip(
        _og["from_account"], _og["amount_received_count"],
        _og["to_account_count_distinct"], _og["to_bank_count_distinct"],
        _og["receiving_currency_count_distinct"],
    ):
        out_map[k] = {"count": cnt, "targets": tgt, "banks": bnk, "currencies": cur}

    in_grouped = tx_table.group_by("to_account").aggregate([
        ("from_account", "count_distinct"),
        ("amount_received", "count"),
    ])
    _ig = in_grouped.to_pydict()
    in_map: dict[str, dict] = {}
    for k, cnt, src in zip(
        _ig["to_account"], _ig["amount_received_count"],
        _ig["from_account_count_distinct"],
    ):
        in_map[k] = {"count": cnt, "sources": src}

    # Ratio features — vectorized groupby
    amt_grouped = tx_table.group_by("from_account").aggregate([
        ("amount_received", "stddev"),
        ("amount_received", "mean"),
    ])
    _ag = amt_grouped.to_pydict()
    amt_map = {}
    for k, std, mean in zip(
        _ag["from_account"], _ag["amount_received_stddev"],
        _ag["amount_received_mean"],
    ):
        amt_map[k] = {"std": std or 0.0, "mean": mean or 0.0}

    intermediary_scores = []
    fan_asymmetries = []
    amount_uniformities = []
    for k in all_accts:
        o = out_map.get(k, {}).get("count", 0)
        i_cnt = in_map.get(k, {}).get("count", 0)
        intermediary_scores.append(min(o, i_cnt) / max(o, i_cnt) if max(o, i_cnt) > 0 else 0.0)
        fan_asymmetries.append(o / (o + i_cnt) if o + i_cnt > 0 else 0.5)
        am = amt_map.get(k, {"std": 0.0, "mean": 0.0})
        if am["mean"] > 0:
            cv = am["std"] / am["mean"]
            amount_uniformities.append(1.0 - min(cv, 1.0))
        else:
            amount_uniformities.append(0.0)

    # Community detection — connected components (O(V+E), fast even on 7M edges)
    # Cached to avoid igraph import overhead on subsequent runs.
    cache_dir = DATA_DIR / ".cache"
    cache_dir.mkdir(exist_ok=True)
    comm_cache = cache_dir / "community_hi_small.pkl"
    if comm_cache.exists():
        acct_to_comm = pickle.loads(comm_cache.read_bytes())
    else:
        import igraph as ig
        G = ig.Graph.TupleList(zip(_from_col, _to_col), directed=False)
        components = G.connected_components()
        acct_to_comm: dict[str, tuple[int, int]] = {}
        for comp_id, members in enumerate(components):
            size = len(members)
            for v_idx in members:
                acct_to_comm[G.vs[v_idx]["name"]] = (comp_id, size)
        comm_cache.write_bytes(pickle.dumps(acct_to_comm))

    # Structuring pct — Arrow groupby for total counts, Python for near-round filter
    import pyarrow.compute as pc

    # Total tx per sender
    _tx_count = tx_table.group_by("from_account").aggregate([("to_account", "count")])
    _txc = _tx_count.to_pydict()
    acct_tx_total = dict(zip(_txc["from_account"], _txc["to_account_count"]))

    # Near-round amounts: filter → groupby
    amt_arr = pc.cast(tx_table["amount_received"], pa.float64())
    nk_arr = pc.multiply(pc.round(pc.divide(amt_arr, 1000.0)), 1000.0)
    abs_diff = pc.abs(pc.subtract(amt_arr, nk_arr))
    is_near_round = pc.and_(
        pc.greater(amt_arr, 100.0),
        pc.and_(pc.greater(nk_arr, 0.0), pc.less(pc.divide(abs_diff, nk_arr), 0.02)),
    )
    near_round_tx = tx_table.filter(is_near_round)
    _nr = near_round_tx.group_by("from_account").aggregate([("to_account", "count")])
    _nrd = _nr.to_pydict()
    acct_near_round = dict(zip(_nrd["from_account"], _nrd["to_account_count"]))

    structuring_pcts = [
        acct_near_round.get(k, 0) / acct_tx_total[k] if acct_tx_total.get(k, 0) > 0 else 0.0
        for k in all_accts
    ]

    # Return ratio + incoming concentration — Arrow groupby for pair sums
    pair_sums = tx_table.select(
        ["from_account", "to_account", "amount_received"],
    ).group_by(["from_account", "to_account"]).aggregate([
        ("amount_received", "sum"),
    ])
    _ps = pair_sums.to_pydict()
    sent_to: dict[str, dict[str, float]] = {}
    recv_from: dict[str, dict[str, float]] = {}
    for f, t, s in zip(_ps["from_account"], _ps["to_account"], _ps["amount_received_sum"]):
        if f and t and s is not None:
            sent_to.setdefault(f, {})[t] = float(s)
            recv_from.setdefault(t, {})[f] = float(s)

    return_ratios = []
    for k in all_accts:
        total_sent = sum(sent_to.get(k, {}).values())
        if total_sent > 0:
            returned = sum(sent_to.get(tgt, {}).get(k, 0.0) for tgt in sent_to.get(k, {}))
            return_ratios.append(min(returned / total_sent, 1.0))
        else:
            return_ratios.append(0.0)

    incoming_concentrations = []
    for k in all_accts:
        senders = recv_from.get(k, {})
        total = sum(senders.values())
        if total > 0 and len(senders) > 1:
            hhi = sum((v / total) ** 2 for v in senders.values())
            incoming_concentrations.append(hhi)
        else:
            incoming_concentrations.append(1.0)

    return pa.table({
        "primary_key": pa.array(all_accts, type=pa.string()),
        "bank_id": pa.array(
            [acct_dict.get(k, {}).get("bank_id") or None for k in all_accts],
            type=pa.string()),
        "entity_type": pa.array(
            [acct_dict.get(k, {}).get("entity_type") or None for k in all_accts],
            type=pa.string()),
        "community_id": pa.array(
            [str(acct_to_comm[k][0]) if k in acct_to_comm else None
             for k in all_accts], type=pa.string()),
        "community_size": pa.array(
            [acct_to_comm[k][1] if k in acct_to_comm else None
             for k in all_accts], type=pa.int64()),
        "intermediary_score": pa.array(intermediary_scores, type=pa.float64()),
        "fan_asymmetry": pa.array(fan_asymmetries, type=pa.float64()),
        "amount_uniformity": pa.array(amount_uniformities, type=pa.float64()),
        "structuring_pct": pa.array(structuring_pcts, type=pa.float64()),
        "return_ratio": pa.array(return_ratios, type=pa.float64()),
        "incoming_concentration": pa.array(incoming_concentrations, type=pa.float64()),
    })