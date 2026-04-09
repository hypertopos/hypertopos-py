# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""XGBoost precision booster — 3 streams on full pipeline stages.

Stream A: Stage 1 (192K) → XGBoost → high-recall / regulatory
Stream B: Stage 3 (35K)  → XGBoost → daily-ops / case-team
Stream C: Stage 4 (14K)  → XGBoost → high-confidence / escalation

Each stream: geometry provides recall pool, XGBoost provides precision filtering.
Features: 24 raw account aggregates + 15 geometry-output (delta_rank_pct, conformal_p,
is_anomaly, multi_source_score, n_anom_pairs, max_pair_dn, n_chains, n_anom_chains,
in_cyclic, tx_density, connected_risk, n_anom_cp, cp_anom_rate, community_anom_rate,
n_counterparties).
"""
import sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages" / "hypertopos"))
from hypertopos import HyperSphere
from hypertopos.navigation.scanner import PassiveScanner
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import pyarrow.csv as pcsv

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ibm-aml"
SPHERE = str(Path(__file__).resolve().parent / "hi_small_sphere" / "gds_aml_hi_small")
t0 = time.time()

# === GT ===
gt_pat = set()
with open(DATA_DIR / "HI-Small_Patterns.txt") as f:
    ct = None
    for line in f:
        line = line.strip()
        if line.startswith("BEGIN"): ct = True
        elif line.startswith("END"): ct = None
        elif ct and line:
            parts = line.split(",")
            if len(parts) >= 5: gt_pat.add(parts[2]); gt_pat.add(parts[4])
gt_full = set()
with open(DATA_DIR / "HI-Small_laundering_accounts.txt") as f:
    for line in f: gt_full.add(line.strip())
print("GT known: %d, GT full: %d" % (len(gt_pat), len(gt_full)))

# === SPHERE DATA ===
hs = HyperSphere.open(SPHERE)
with hs.session("ml") as s:
    reader = s._reader
    sphere = reader.read_sphere()

    print("Loading points...", "%.0fs" % (time.time()-t0))
    pts = reader.read_points("accounts", 1)
    pk_list = pts["primary_key"].to_pylist()
    n = len(pk_list)
    pk_idx = {pk: i for i, pk in enumerate(pk_list)}
    def PV(col, i):
        v = pts[col][i].as_py()
        return float(v) if v is not None else 0.0

    raw_cols = ["tx_out_count","n_out_targets","n_dest_banks","n_currencies_out",
                "n_payment_formats","sum_out","max_out","mean_out",
                "tx_in_count","n_in_sources","n_source_banks","sum_in",
                "burst_tx_out","burst_tx_in","in_degree","out_degree",
                "reciprocity","counterpart_overlap","intermediary_score",
                "fan_asymmetry","amount_uniformity","structuring_pct","return_ratio",
                "n_currencies_in"]

    # Account geometry
    print("Loading geometry...", "%.0fs" % (time.time()-t0))
    geo = reader.read_geometry("account_pattern", 1,
        columns=["primary_key","delta_norm","delta_rank_pct","conformal_p","is_anomaly"])
    geo_map = {}
    for i in range(geo.num_rows):
        pk = geo["primary_key"][i].as_py()
        geo_map[pk] = {
            "rk": geo["delta_rank_pct"][i].as_py() or 0,
            "cp": geo["conformal_p"][i].as_py() or 1.0,
            "an": 1 if geo["is_anomaly"][i].as_py() else 0,
        }

    # Geometry multi-source score (sources 1-3)
    scanner = PassiveScanner(reader, sphere, s._manifest)
    scanner.add_source("account_pattern", "account_pattern", key_type="direct")
    scanner.add_source("account_pairs_pattern", "account_pairs_pattern", key_type="composite")
    scanner.add_source("tx_chains_pattern", "tx_chains_pattern", key_type="chain")
    result = scanner.scan("accounts", scoring="count", threshold=1)
    geo_score = {h.primary_key: len(h.sources) for h in result.hits}
    src_acct = {h.primary_key for h in result.hits if "account_pattern" in h.sources}
    src_pair = {h.primary_key for h in result.hits if "account_pairs_pattern" in h.sources}
    src_chain = {h.primary_key for h in result.hits if "tx_chains_pattern" in h.sources}

    # Pair geometry
    print("Loading pairs...", "%.0fs" % (time.time()-t0))
    pg = reader.read_geometry("account_pairs_pattern", 1,
        columns=["primary_key","is_anomaly","delta_norm"])
    acct_anom_pairs = defaultdict(int)
    acct_max_pair_dn = defaultdict(float)
    acct_counterparties = defaultdict(set)
    for i in range(pg.num_rows):
        pk = pg["primary_key"][i].as_py()
        if not pk or "::" not in pk: continue
        a, b = pk.split("::")
        acct_counterparties[a].add(b)
        acct_counterparties[b].add(a)
        if pg["is_anomaly"][i].as_py():
            dn = pg["delta_norm"][i].as_py() or 0
            for p in (a, b):
                acct_anom_pairs[p] += 1
                if dn > acct_max_pair_dn[p]: acct_max_pair_dn[p] = dn

    # Chain geometry
    print("Loading chains...", "%.0fs" % (time.time()-t0))
    cp = reader.read_points("tx_chains", 1)
    cg = reader.read_geometry("tx_chains_pattern", 1,
        columns=["primary_key","entity_keys","is_anomaly"])
    acct_n_chains = defaultdict(int)
    acct_n_anom_chains = defaultdict(int)
    acct_in_cyclic = set()
    cf = {}
    for i in range(cp.num_rows):
        cf[cp["primary_key"][i].as_py()] = cp["is_cyclic"][i].as_py() or False
    for i in range(cg.num_rows):
        cpk = cg["primary_key"][i].as_py()
        ek = cg["entity_keys"][i].as_py()
        anom = cg["is_anomaly"][i].as_py()
        if not ek: continue
        for k in ek:
            if not k: continue
            acct_n_chains[k] += 1
            if anom: acct_n_anom_chains[k] += 1
            if cf.get(cpk, False): acct_in_cyclic.add(k)

    # Tx density
    print("Loading tx density...", "%.0fs" % (time.time()-t0))
    tx_density = {}
    if "tx_pattern" in sphere.patterns:
        tac = {}; ttc = {}
        for ek in reader.read_geometry("tx_pattern", 1, columns=["entity_keys"],
                                        filter="is_anomaly = true")["entity_keys"].to_pylist():
            if ek:
                for k in ek:
                    if k: tac[k] = tac.get(k, 0) + 1
        for ek in reader.read_geometry("tx_pattern", 1, columns=["entity_keys"])["entity_keys"].to_pylist():
            if ek:
                for k in ek:
                    if k: ttc[k] = ttc.get(k, 0) + 1
        for a, c in tac.items(): tx_density[a] = c / ttc.get(a, 1)
    high_d = {a for a, r in tx_density.items() if r >= 0.05}

    # Connected risk
    print("Computing connected risk...", "%.0fs" % (time.time()-t0))
    acct_connected_risk = {}
    acct_n_anom_cp = {}
    acct_cp_anom_rate = {}
    for acct, cps in acct_counterparties.items():
        ranks = []; na = 0
        for c in cps:
            if c in geo_map:
                ranks.append(geo_map[c]["rk"])
                if geo_map[c]["an"]: na += 1
        acct_connected_risk[acct] = np.mean(ranks) if ranks else 0
        acct_n_anom_cp[acct] = na
        acct_cp_anom_rate[acct] = na / len(cps) if cps else 0

    # Community anomaly rate
    comm_col = pts["community_id"].to_pylist() if "community_id" in pts.schema.names else [None]*n
    cc = defaultdict(lambda: [0, 0])
    for i, pk in enumerate(pk_list):
        cid = comm_col[i]
        if cid is None: continue
        cc[cid][0] += 1
        if pk in geo_map and geo_map[pk]["an"]: cc[cid][1] += 1
    comm_anom_rate = {cid: a/t if t > 0 else 0 for cid, (t, a) in cc.items()}

    # === HEURISTIC SOURCES 4-7 (exact, matching test_passive_scan.py) ===
    print("Loading transactions for heuristic sources...", "%.0fs" % (time.time()-t0))
    tx = pcsv.read_csv(str(DATA_DIR / "HI-Small_Trans.csv"), read_options=pcsv.ReadOptions(
        column_names=["timestamp","from_bank","from_account","to_bank","to_account",
                      "amount_received","receiving_currency","amount_paid","payment_currency",
                      "payment_format","is_laundering"], skip_rows=1))
    from_list = tx["from_account"].to_pylist()
    to_list = tx["to_account"].to_pylist()
    amounts = [float(v) if v is not None else 0.0 for v in tx["amount_received"].to_pylist()]

    # Source 4: multi-cur cross-border (exact CSV logic)
    sent_to = defaultdict(lambda: defaultdict(float))
    acct_currencies = defaultdict(set)
    acct_dest_banks = defaultdict(set)
    for f, t, amt, cur, tb in zip(from_list, to_list, amounts,
                                   tx["receiving_currency"].to_pylist(), tx["to_bank"].to_pylist()):
        if f is None or t is None: continue
        sent_to[f][t] += amt
        if cur: acct_currencies[f].add(cur)
        if tb: acct_dest_banks[f].add(tb)
    src_mcur = set()
    for acct in sent_to:
        if len(acct_currencies.get(acct, set())) < 2: continue
        total = sum(sent_to[acct].values())
        if total > 0:
            ret = sum(sent_to[tgt].get(acct, 0) for tgt in sent_to[acct] if acct in sent_to.get(tgt, {}))
            if ret / total > 0.3: src_mcur.add(acct); continue
        if len(acct_dest_banks.get(acct, set())) >= 3: src_mcur.add(acct)

    # Source 5: borderline (from geometry)
    src_border = set()
    for i in range(geo.num_rows):
        if not geo["is_anomaly"][i].as_py():
            pct = geo["delta_rank_pct"][i].as_py()
            if pct is not None and pct >= 80:
                src_border.add(geo["primary_key"][i].as_py())

    # Source 6: reputation — skipped (0 unique TP in benchmark)
    src_rep = set()

    # Source 7: structuring (exact CSV logic)
    acct_near = defaultdict(int); acct_total_tx = defaultdict(int)
    for f, amt in zip(from_list, amounts):
        if f is None: continue
        acct_total_tx[f] += 1
        if amt > 100:
            nk = round(amt / 1000) * 1000
            if nk > 0 and abs(amt - nk) / nk < 0.02: acct_near[f] += 1
    src_struct = set()
    for acct, near in acct_near.items():
        total = acct_total_tx.get(acct, 1)
        if total >= 5 and near / total > 0.5: src_struct.add(acct)

    # === STREAMS C, D (behavioral typology sources for Stage 1) ===
    print("Computing streams C, D...", "%.0fs" % (time.time()-t0))

    # Stream C: CYCLE/round-trip (return_ratio >= 0.20 AND in any chain)
    chain_accounts = set()
    for i in range(cg.num_rows):
        ek = cg["entity_keys"][i].as_py()
        if ek:
            for k in ek:
                if k: chain_accounts.add(k)
    src_stream_c = set()
    for i in range(n):
        pk = pk_list[i]
        if PV("return_ratio", i) >= 0.20 and pk in chain_accounts:
            src_stream_c.add(pk)

    # Stream D: STRUCTURING behavioral (structuring_pct >= 0.50 AND amount_uniformity >= 0.5
    #           AND tx_out_count >= population p50)
    tx_out_vals = [PV("tx_out_count", i) for i in range(n)]
    tx_out_p50 = np.percentile(tx_out_vals, 50)
    src_stream_d = set()
    for i in range(n):
        if (PV("structuring_pct", i) >= 0.50 and PV("amount_uniformity", i) >= 0.5
                and PV("tx_out_count", i) >= tx_out_p50):
            src_stream_d.add(pk_list[i])

    # === FULL 9-SOURCE SCORING ===
    print("Computing full pipeline scores...", "%.0fs" % (time.time()-t0))
    core_sources = [src_acct, src_pair, src_chain, src_mcur, src_border, src_rep, src_struct]
    all_9_sources = core_sources + [src_stream_c, src_stream_d]

    # Stage 1 = OR of all 9 sources
    all_accounts_9 = set()
    for src in all_9_sources: all_accounts_9 |= src

    # Stage 2-4 scoring uses 7 core sources only (C/D don't boost scoring)
    all_accounts_7 = set()
    for src in core_sources: all_accounts_7 |= src
    full_score = {}
    for acct in all_accounts_7:
        full_score[acct] = sum(1 for src in core_sources if acct in src)

    # Boosted scores (density boost)
    boosted = {}
    for acct in all_accounts_7 | high_d:
        boosted[acct] = full_score.get(acct, 0) + (1 if acct in high_d else 0)

    # Full pipeline stages
    stage1_pks = sorted(all_accounts_9)  # OR all 9 sources
    stage3_pks = sorted(pk for pk, sc in boosted.items() if sc >= 3)
    stage4_pks = sorted(pk for pk, sc in full_score.items() if sc >= 3 and pk in high_d)

    print("Stage 1: %d suspects" % len(stage1_pks))
    print("Stage 3: %d suspects" % len(stage3_pks))
    print("Stage 4: %d suspects" % len(stage4_pks))

    # === BUILD FEATURES ===
    geo_cols = ["delta_rank_pct","conformal_p","is_anomaly","multi_source_score",
                "n_anom_pairs","max_pair_dn","n_chains","n_anom_chains","in_cyclic",
                "tx_density","connected_risk","n_anom_cp","cp_anom_rate",
                "community_anom_rate","n_counterparties"]
    feat_names = raw_cols + geo_cols
    go = len(raw_cols)

    def build_features(pks):
        X = np.zeros((len(pks), len(feat_names)), dtype=np.float32)
        yp = np.zeros(len(pks), dtype=np.int32)
        yf = np.zeros(len(pks), dtype=np.int32)
        for j, pk in enumerate(pks):
            if pk not in pk_idx: continue
            i = pk_idx[pk]
            for k, col in enumerate(raw_cols): X[j, k] = PV(col, i)
            gm = geo_map.get(pk, {"rk": 0, "cp": 1.0, "an": 0})
            X[j, go+0] = gm["rk"]; X[j, go+1] = gm["cp"]; X[j, go+2] = gm["an"]
            X[j, go+3] = geo_score.get(pk, 0)
            X[j, go+4] = acct_anom_pairs.get(pk, 0)
            X[j, go+5] = acct_max_pair_dn.get(pk, 0)
            X[j, go+6] = acct_n_chains.get(pk, 0)
            X[j, go+7] = acct_n_anom_chains.get(pk, 0)
            X[j, go+8] = 1.0 if pk in acct_in_cyclic else 0.0
            X[j, go+9] = tx_density.get(pk, 0)
            X[j, go+10] = acct_connected_risk.get(pk, 0)
            X[j, go+11] = acct_n_anom_cp.get(pk, 0)
            X[j, go+12] = acct_cp_anom_rate.get(pk, 0)
            cid = comm_col[pk_idx[pk]]
            X[j, go+13] = comm_anom_rate.get(cid, 0) if cid else 0
            X[j, go+14] = len(acct_counterparties.get(pk, set()))
            yp[j] = 1 if pk in gt_pat else 0
            yf[j] = 1 if pk in gt_full else 0
        return X, yp, yf

    # === RUN 3 STREAMS ===
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for stream_name, pks, thresholds in [
        ("Stream A: Stage 1 (high-recall)", stage1_pks, [0.05, 0.10, 0.15, 0.20, 0.30]),
        ("Stream B: Stage 3 (daily-ops)", stage3_pks, [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]),
        ("Stream C: Stage 4 (escalation)", stage4_pks, [0.10, 0.20, 0.30, 0.40, 0.50]),
    ]:
        X, yp, yf = build_features(pks)
        print()
        print("=" * 80)
        print("%s — %d suspects, %d TP known, %d TP full" % (stream_name, len(pks), yp.sum(), yf.sum()))
        print("=" * 80)

        # Train on known GT ONLY, measure on both GTs
        print()
        tp_base_k = yp.sum(); tp_base_f = yf.sum()
        fp_base_k = len(pks) - tp_base_k; fp_base_f = len(pks) - tp_base_f
        print("%-12s %8s %8s %8s %8s %8s" % ("Threshold", "Flagged", "rec(kn)", "FP/TP(kn)", "rec(fl)", "FP/TP(fl)"))
        print("-" * 60)
        print("%-12s %8d %7.1f%% %9.1f %7.1f%% %9.1f" % (
            "baseline", len(pks),
            tp_base_k/len(gt_pat)*100, fp_base_k/tp_base_k if tp_base_k else 999,
            tp_base_f/len(gt_full)*100, fp_base_f/tp_base_f if tp_base_f else 999))

        for thresh in thresholds:
            flagged_idx = set()
            for tri, tei in skf.split(X, yp):
                m = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=len(yp[yp==0])/max(len(yp[yp==1]), 1),
                    eval_metric="logloss", verbosity=0, random_state=42)
                m.fit(X[tri], yp[tri])
                pr = m.predict_proba(X[tei])[:, 1]
                for i, idx in enumerate(tei):
                    if pr[i] >= thresh:
                        flagged_idx.add(idx)
            flagged_pks = {pks[i] for i in flagged_idx}
            tp_k = len(flagged_pks & gt_pat); fp_k = len(flagged_pks) - tp_k
            tp_f = len(flagged_pks & gt_full); fp_f = len(flagged_pks) - tp_f
            af = len(flagged_pks)
            fpt_k = fp_k / tp_k if tp_k else 999
            fpt_f = fp_f / tp_f if tp_f else 999
            print("%-12s %8d %7.1f%% %9.1f %7.1f%% %9.1f" % (
                ">= %.2f" % thresh, af,
                tp_k/len(gt_pat)*100, fpt_k,
                tp_f/len(gt_full)*100, fpt_f))

    # Feature importance (Stream B, known GT)
    print()
    print("Feature importance (Stream B, known GT):")
    X_b, yp_b, _ = build_features(stage3_pks)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=len(yp_b[yp_b==0])/max(len(yp_b[yp_b==1]), 1),
        eval_metric="logloss", verbosity=0, random_state=42)
    m.fit(X_b, yp_b)
    imp = m.feature_importances_
    for rank, idx in enumerate(np.argsort(imp)[::-1][:15]):
        src = "raw" if idx < len(raw_cols) else "GEO"
        print("  %2d. %-25s %.4f [%s]" % (rank+1, feat_names[idx], imp[idx], src))

    print("\nElapsed: %.0fs" % (time.time()-t0))
