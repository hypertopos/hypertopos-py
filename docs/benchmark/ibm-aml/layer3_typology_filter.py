# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Typology filter on XGBoost stream outputs. Sphere-only, all 3 streams."""
import sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path("packages/hypertopos").resolve()))
from hypertopos import HyperSphere
from hypertopos.navigation.scanner import PassiveScanner
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

DATA_DIR = Path("benchmark/data/ibm-aml")
SPHERE = "benchmark/ibm-aml/hi_small_sphere/gds_aml_hi_small"
t0 = time.time()

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

hs = HyperSphere.open(SPHERE)
with hs.session("ag") as s:
    reader = s._reader; sphere = reader.read_sphere()
    print("Points...", "%.0fs" % (time.time()-t0))
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

    # Geometry
    print("Geometry...", "%.0fs" % (time.time()-t0))
    geo = reader.read_geometry("account_pattern", 1,
        columns=["primary_key","delta_rank_pct","conformal_p","is_anomaly"])
    geo_map = {}
    for i in range(geo.num_rows):
        pk = geo["primary_key"][i].as_py()
        geo_map[pk] = {"rk": geo["delta_rank_pct"][i].as_py() or 0,
                        "cp": geo["conformal_p"][i].as_py() or 1.0,
                        "an": 1 if geo["is_anomaly"][i].as_py() else 0}

    scanner = PassiveScanner(reader, sphere, s._manifest)
    scanner.add_source("account_pattern", "account_pattern", key_type="direct")
    scanner.add_source("account_pairs_pattern", "account_pairs_pattern", key_type="composite")
    scanner.add_source("tx_chains_pattern", "tx_chains_pattern", key_type="chain")
    result = scanner.scan("accounts", scoring="count", threshold=1)
    geo_score = {h.primary_key: len(h.sources) for h in result.hits}

    # Pairs — just anomaly membership
    print("Pairs...", "%.0fs" % (time.time()-t0))
    pg = reader.read_geometry("account_pairs_pattern", 1, columns=["primary_key","is_anomaly","delta_norm"])
    acct_anom_pairs = defaultdict(int); acct_max_pair_dn = defaultdict(float)
    acct_counterparties = defaultdict(set)
    in_anom_pair = set()
    for i in range(pg.num_rows):
        pk = pg["primary_key"][i].as_py()
        if not pk or "::" not in pk: continue
        a, b = pk.split("::")
        acct_counterparties[a].add(b); acct_counterparties[b].add(a)
        if pg["is_anomaly"][i].as_py():
            dn = pg["delta_norm"][i].as_py() or 0
            in_anom_pair.add(a); in_anom_pair.add(b)
            for p in (a, b):
                acct_anom_pairs[p] += 1
                acct_max_pair_dn[p] = max(acct_max_pair_dn[p], dn)

    # Chains — dict lookup, NOT nested loop
    print("Chains...", "%.0fs" % (time.time()-t0))
    cp = reader.read_points("tx_chains", 1)
    chain_feat = {}
    for i in range(cp.num_rows):
        chain_feat[cp["primary_key"][i].as_py()] = {
            "cyc": cp["is_cyclic"][i].as_py() or False,
            "hops": cp["hop_count"][i].as_py() or 0,
            "tsh": cp["time_span_hours"][i].as_py() or 0,
            "ndc": cp["n_distinct_categories"][i].as_py() or 0,
            "cv": cp["amount_cv"][i].as_py() or 0,
        }
    cg = reader.read_geometry("tx_chains_pattern", 1, columns=["primary_key","entity_keys","is_anomaly"])
    acct_nc = defaultdict(int); acct_nac = defaultdict(int)
    in_cyclic = set(); max_hops = defaultdict(int); min_ts = defaultdict(lambda: 9999.0)
    max_ndc = defaultdict(int); low_cv = set(); chain_accts = set()
    for i in range(cg.num_rows):
        cpk = cg["primary_key"][i].as_py()
        ek = cg["entity_keys"][i].as_py()
        anom = cg["is_anomaly"][i].as_py()
        if not ek or cpk not in chain_feat: continue
        cf = chain_feat[cpk]
        for k in ek:
            if not k: continue
            chain_accts.add(k); acct_nc[k] += 1
            if anom: acct_nac[k] += 1
            if cf["cyc"]: in_cyclic.add(k)
            if cf["hops"] > max_hops[k]: max_hops[k] = cf["hops"]
            if 0 < cf["tsh"] < min_ts[k]: min_ts[k] = cf["tsh"]
            if cf["ndc"] > max_ndc[k]: max_ndc[k] = cf["ndc"]
            if cf["cv"] < 0.3 and cf["hops"] >= 3: low_cv.add(k)

    # Tx density
    print("Tx density...", "%.0fs" % (time.time()-t0))
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
    print("Connected risk...", "%.0fs" % (time.time()-t0))
    acct_cr = {}; acct_nacp = {}; acct_car = {}
    for acct, cps in acct_counterparties.items():
        ranks = []; na = 0
        for c in cps:
            if c in geo_map:
                ranks.append(geo_map[c]["rk"])
                if geo_map[c]["an"]: na += 1
        acct_cr[acct] = np.mean(ranks) if ranks else 0
        acct_nacp[acct] = na; acct_car[acct] = na / len(cps) if cps else 0
    comm_col = pts["community_id"].to_pylist() if "community_id" in pts.schema.names else [None]*n
    ccc = defaultdict(lambda: [0, 0])
    for i, pk in enumerate(pk_list):
        cid = comm_col[i]
        if cid is None: continue
        ccc[cid][0] += 1
        if pk in geo_map and geo_map[pk]["an"]: ccc[cid][1] += 1
    car = {cid: a/t if t > 0 else 0 for cid, (t, a) in ccc.items()}

    # Heuristic sources (points-based approx for scoring)
    print("Scoring...", "%.0fs" % (time.time()-t0))
    src_acct = {h.primary_key for h in result.hits if "account_pattern" in h.sources}
    src_pair = {h.primary_key for h in result.hits if "account_pairs_pattern" in h.sources}
    src_chain = {h.primary_key for h in result.hits if "tx_chains_pattern" in h.sources}
    src_mcur = {pk_list[i] for i in range(n)
                if PV("n_currencies_out", i) >= 2 and (PV("return_ratio", i) > 0.3 or PV("n_dest_banks", i) >= 3)}
    src_border = {geo["primary_key"][i].as_py() for i in range(geo.num_rows)
                  if not geo["is_anomaly"][i].as_py() and (geo["delta_rank_pct"][i].as_py() or 0) >= 80}
    src_struct = {pk_list[i] for i in range(n)
                  if PV("structuring_pct", i) > 0.5 and PV("tx_out_count", i) >= 5}
    core = [src_acct, src_pair, src_chain, src_mcur, src_border, set(), src_struct]
    all_7 = set()
    for src in core: all_7 |= src
    full_score = {a: sum(1 for src in core if a in src) for a in all_7}
    boosted = {a: full_score.get(a, 0) + (1 if a in high_d else 0) for a in all_7 | high_d}

    # Streams C/D for Stage 1
    src_c = {pk_list[i] for i in range(n)
             if PV("return_ratio", i) >= 0.20 and pk_list[i] in chain_accts}
    tx_p50 = np.percentile([PV("tx_out_count", i) for i in range(n)], 50)
    src_d = {pk_list[i] for i in range(n)
             if PV("structuring_pct", i) >= 0.50 and PV("amount_uniformity", i) >= 0.5
             and PV("tx_out_count", i) >= tx_p50}
    all_9 = all_7 | src_c | src_d

    stage1_pks = sorted(all_9)
    stage3_pks = sorted(pk for pk, sc in boosted.items() if sc >= 3)
    stage4_pks = sorted(pk for pk, sc in full_score.items() if sc >= 3 and pk in high_d)

    # Typology check
    burst_p90 = np.percentile([PV("burst_tx_out", i) for i in range(n)], 90)
    def check_typo(pk):
        if pk not in pk_idx: return 0
        i = pk_idx[pk]; h = 0
        if PV("amount_uniformity",i) > 0.7 and PV("n_currencies_out",i) >= 2 and PV("n_dest_banks",i) >= 3: h+=1
        if PV("burst_tx_out",i) > burst_p90 and PV("n_currencies_out",i) >= 2 and pk in in_cyclic and min_ts.get(pk,9999) < 24: h+=1
        if pk in in_cyclic and min_ts.get(pk,9999) <= 24 and max_ndc.get(pk,0) >= 2: h+=1
        if pk in in_anom_pair and PV("counterpart_overlap",i) > 0: h+=1
        if max_hops.get(pk,0) >= 4 and max_ndc.get(pk,0) >= 2: h+=1
        if PV("intermediary_score",i) > 0.3 and PV("n_currencies_out",i) >= 2: h+=1
        if PV("return_ratio",i) > 0.3: h+=1
        if PV("n_payment_formats",i) >= 4 and PV("n_currencies_out",i) >= 2: h+=1
        if PV("in_degree",i) >= 10 and PV("fan_asymmetry",i) < 0.3: h+=1
        if PV("n_dest_banks",i) >= 5 and PV("n_currencies_out",i) >= 3: h+=1
        if pk in low_cv: h+=1
        if PV("counterpart_overlap",i) > 0 and PV("return_ratio",i) > 0.2: h+=1
        if PV("tx_out_count",i) > 50 and PV("amount_uniformity",i) > 0.5 and PV("n_currencies_out",i) >= 2: h+=1
        if PV("tx_out_count",i) > 100 and PV("mean_out",i) < 500 and PV("n_out_targets",i) >= 10 and PV("n_currencies_out",i) >= 2: h+=1
        return h

    # Build features
    go = len(raw_cols)
    geo_cols = ["delta_rank_pct","conformal_p","is_anomaly","multi_source_score",
                "n_anom_pairs","max_pair_dn","n_chains","n_anom_chains","in_cyclic",
                "tx_density","connected_risk","n_anom_cp","cp_anom_rate",
                "community_anom_rate","n_counterparties"]
    def build(pks):
        X = np.zeros((len(pks), len(raw_cols)+len(geo_cols)), dtype=np.float32)
        yp = np.zeros(len(pks), dtype=np.int32)
        yf = np.zeros(len(pks), dtype=np.int32)
        for j, pk in enumerate(pks):
            if pk not in pk_idx: continue
            i = pk_idx[pk]
            for k, col in enumerate(raw_cols): X[j,k]=PV(col,i)
            gm=geo_map.get(pk,{"rk":0,"cp":1.0,"an":0})
            X[j,go]=gm["rk"]; X[j,go+1]=gm["cp"]; X[j,go+2]=gm["an"]
            X[j,go+3]=geo_score.get(pk,0); X[j,go+4]=acct_anom_pairs.get(pk,0)
            X[j,go+5]=acct_max_pair_dn.get(pk,0); X[j,go+6]=acct_nc.get(pk,0)
            X[j,go+7]=acct_nac.get(pk,0); X[j,go+8]=1.0 if pk in in_cyclic else 0.0
            X[j,go+9]=tx_density.get(pk,0); X[j,go+10]=acct_cr.get(pk,0)
            X[j,go+11]=acct_nacp.get(pk,0); X[j,go+12]=acct_car.get(pk,0)
            cid=comm_col[pk_idx[pk]]; X[j,go+13]=car.get(cid,0) if cid else 0
            X[j,go+14]=len(acct_counterparties.get(pk,set()))
            yp[j]=1 if pk in gt_pat else 0; yf[j]=1 if pk in gt_full else 0
        return X, yp, yf

    # Run streams B/C + filter
    print("XGBoost + filter...", "%.0fs" % (time.time()-t0))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, pks, thresholds in [
        ("Stream A (Stage 1)", stage1_pks, [0.10, 0.20]),
        ("Stream B (Stage 3)", stage3_pks, [0.10, 0.20]),
        ("Stream C (Stage 4)", stage4_pks, [0.20, 0.50]),
    ]:
        X, yp, yf = build(pks)
        print()
        print("=== %s: %d suspects, %d TP known, %d TP full ===" % (name, len(pks), yp.sum(), yf.sum()))
        for thresh in thresholds:
            flagged = set()
            for tri, tei in skf.split(X, yp):
                m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=len(yp[yp==0])/max(len(yp[yp==1]),1),
                    eval_metric="logloss", verbosity=0, random_state=42)
                m.fit(X[tri], yp[tri])
                pr = m.predict_proba(X[tei])[:, 1]
                for idx, p in zip(tei, pr):
                    if p >= thresh: flagged.add(pks[idx])
            tp_x = len(flagged & gt_pat); fp_x = len(flagged - gt_pat)
            tp_xf = len(flagged & gt_full); fp_xf = len(flagged - gt_full)
            with_t = {pk for pk in flagged if check_typo(pk) >= 1}
            tp_a = len(with_t & gt_pat); fp_a = len(with_t - gt_pat)
            tp_af = len(with_t & gt_full); fp_af = len(with_t - gt_full)
            no_t = flagged - with_t; tp_n = len(no_t & gt_pat)
            print("  p>=%.2f:" % thresh)
            print("    XGBoost:       %5d  rec(kn)=%.1f%% FP/TP=%.1f  rec(fl)=%.1f%% FP/TP=%.1f" % (
                len(flagged), tp_x/len(gt_pat)*100, fp_x/tp_x if tp_x else 999,
                tp_xf/len(gt_full)*100, fp_xf/tp_xf if tp_xf else 999))
            print("    + filter:       %5d  rec(kn)=%.1f%% FP/TP=%.1f  rec(fl)=%.1f%% FP/TP=%.1f" % (
                len(with_t), tp_a/len(gt_pat)*100, fp_a/tp_a if tp_a else 999,
                tp_af/len(gt_full)*100, fp_af/tp_af if tp_af else 999))
            print("    no typology:   %5d  TP(kn)=%4d  TP(fl)=%4d" % (len(no_t), tp_n, len(no_t & gt_full)))

            # 2 example profiles (1 TP + 1 FP)
            if flagged:
                tp_ex = sorted(flagged & gt_pat)[:1]
                fp_ex = sorted(flagged - gt_pat)[:1]
                for ex_pk in tp_ex + fp_ex:
                    if ex_pk not in pk_idx: continue
                    ei = pk_idx[ex_pk]
                    gm = geo_map.get(ex_pk, {"rk": 0, "an": 0})
                    label = "TP" if ex_pk in gt_pat else "FP"
                    typo_names = []
                    if PV("n_dest_banks", ei) >= 5 and PV("n_currencies_out", ei) >= 3: typo_names.append("T14")
                    if PV("in_degree", ei) >= 10 and PV("fan_asymmetry", ei) < 0.3: typo_names.append("T13")
                    if PV("intermediary_score", ei) > 0.3 and PV("n_currencies_out", ei) >= 2: typo_names.append("T9")
                    if PV("return_ratio", ei) > 0.3: typo_names.append("T11")
                    if PV("n_payment_formats", ei) >= 4 and PV("n_currencies_out", ei) >= 2: typo_names.append("T12")
                    if ex_pk in in_anom_pair and PV("counterpart_overlap", ei) > 0: typo_names.append("T4")
                    in_ch = "yes" if ex_pk in chain_accts else "no"
                    print("    Example [%s]: %s  geo_score=%d rank=%.0f chain=%s n_cur_out=%.0f n_banks=%.0f typos=%s" % (
                        label, ex_pk, geo_score.get(ex_pk, 0), gm["rk"], in_ch,
                        PV("n_currencies_out", ei), PV("n_dest_banks", ei),
                        ",".join(typo_names) if typo_names else "none"))

    print("\nElapsed: %.0fs" % (time.time()-t0))
