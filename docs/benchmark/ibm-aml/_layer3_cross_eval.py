# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Layer 3 cross-eval: train on HI-Small, apply + typology filter to LI-Small."""
import sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages" / "hypertopos"))
from hypertopos import HyperSphere
from hypertopos.navigation.scanner import PassiveScanner
import xgboost as xgb
import pyarrow.csv as pcsv

t0 = time.time()
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ibm-aml"
SPHERE_HI = str(Path(__file__).resolve().parent / "hi_small_sphere" / "gds_aml_hi_small")
SPHERE_LI = str(Path(__file__).resolve().parent / "li_small_sphere" / "gds_aml_li_small")

raw_cols = ["tx_out_count","n_out_targets","n_dest_banks","n_currencies_out",
            "n_payment_formats","sum_out","max_out","mean_out",
            "tx_in_count","n_in_sources","n_source_banks","sum_in",
            "burst_tx_out","burst_tx_in","in_degree","out_degree",
            "reciprocity","counterpart_overlap","intermediary_score",
            "fan_asymmetry","amount_uniformity","structuring_pct","return_ratio",
            "n_currencies_in"]
geo_cols = ["delta_rank_pct","conformal_p","is_anomaly","multi_source_score",
            "n_anom_pairs","max_pair_dn","n_chains","n_anom_chains","in_cyclic",
            "tx_density","connected_risk","n_anom_cp","cp_anom_rate",
            "community_anom_rate","n_counterparties"]
feat_names = raw_cols + geo_cols
go = len(raw_cols)


def load_gt(prefix):
    gt_pat = set()
    with open(DATA_DIR / f"{prefix}_Patterns.txt") as f:
        ct = None
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN"): ct = True
            elif line.startswith("END"): ct = None
            elif ct and line:
                parts = line.split(",")
                if len(parts) >= 5: gt_pat.add(parts[2]); gt_pat.add(parts[4])
    gt_full = set()
    with open(DATA_DIR / f"{prefix}_laundering_accounts.txt") as f:
        for line in f: gt_full.add(line.strip())
    return gt_pat, gt_full


def load_sphere(sphere_path, csv_prefix):
    hs = HyperSphere.open(sphere_path)
    with hs.session("ml") as s:
        reader = s._reader; sphere = reader.read_sphere()
        print(f"  [{csv_prefix}] points...", "%.0fs" % (time.time()-t0))
        pts = reader.read_points("accounts", 1)
        pk_list = pts["primary_key"].to_pylist(); n = len(pk_list)
        pk_idx = {pk: i for i, pk in enumerate(pk_list)}
        rv = {col: [float(v) if v is not None else 0.0 for v in pts[col].to_pylist()] for col in raw_cols}
        comm_col = pts["community_id"].to_pylist() if "community_id" in pts.schema.names else [None]*n

        print(f"  [{csv_prefix}] geometry...", "%.0fs" % (time.time()-t0))
        geo = reader.read_geometry("account_pattern", 1,
            columns=["primary_key","delta_norm","delta_rank_pct","conformal_p","is_anomaly"])
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
        src_acct = {h.primary_key for h in result.hits if "account_pattern" in h.sources}
        src_pair = {h.primary_key for h in result.hits if "account_pairs_pattern" in h.sources}
        src_chain = {h.primary_key for h in result.hits if "tx_chains_pattern" in h.sources}

        print(f"  [{csv_prefix}] pairs...", "%.0fs" % (time.time()-t0))
        pg = reader.read_geometry("account_pairs_pattern", 1,
            columns=["primary_key","is_anomaly","delta_norm"])
        acct_anom_pairs = defaultdict(int); acct_max_pair_dn = defaultdict(float)
        acct_counterparties = defaultdict(set); in_anom_pair = set()
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
                    if dn > acct_max_pair_dn[p]: acct_max_pair_dn[p] = dn

        print(f"  [{csv_prefix}] chains...", "%.0fs" % (time.time()-t0))
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
        cg = reader.read_geometry("tx_chains_pattern", 1,
            columns=["primary_key","entity_keys","is_anomaly"])
        acct_n_chains = defaultdict(int); acct_n_anom_chains = defaultdict(int)
        acct_in_cyclic = set(); max_hops = defaultdict(int)
        min_ts = defaultdict(lambda: 9999.0); max_ndc = defaultdict(int)
        low_cv = set(); chain_accounts = set()
        for i in range(cg.num_rows):
            cpk = cg["primary_key"][i].as_py(); ek = cg["entity_keys"][i].as_py()
            anom = cg["is_anomaly"][i].as_py()
            if not ek or cpk not in chain_feat: continue
            cf = chain_feat[cpk]
            for k in ek:
                if not k: continue
                chain_accounts.add(k); acct_n_chains[k] += 1
                if anom: acct_n_anom_chains[k] += 1
                if cf["cyc"]: acct_in_cyclic.add(k)
                if cf["hops"] > max_hops[k]: max_hops[k] = cf["hops"]
                if 0 < cf["tsh"] < min_ts[k]: min_ts[k] = cf["tsh"]
                if cf["ndc"] > max_ndc[k]: max_ndc[k] = cf["ndc"]
                if cf["cv"] < 0.3 and cf["hops"] >= 3: low_cv.add(k)

        print(f"  [{csv_prefix}] tx density...", "%.0fs" % (time.time()-t0))
        tx_density = {}
        if "tx_pattern" in sphere.patterns:
            tac = {}; ttc = {}
            for ek in reader.read_geometry("tx_pattern", 1, columns=["entity_keys"],
                    filter="is_anomaly = true")["entity_keys"].to_pylist():
                if ek:
                    for k in ek:
                        if k: tac[k] = tac.get(k, 0) + 1
            for ek in reader.read_geometry("tx_pattern", 1,
                    columns=["entity_keys"])["entity_keys"].to_pylist():
                if ek:
                    for k in ek:
                        if k: ttc[k] = ttc.get(k, 0) + 1
            for a, c in tac.items(): tx_density[a] = c / ttc.get(a, 1)
        high_d = {a for a, r in tx_density.items() if r >= 0.05}

        print(f"  [{csv_prefix}] connected risk...", "%.0fs" % (time.time()-t0))
        acct_cr = {}; acct_nacp = {}; acct_car = {}
        for acct, cps in acct_counterparties.items():
            ranks = []; na = 0
            for c in cps:
                if c in geo_map:
                    ranks.append(geo_map[c]["rk"])
                    if geo_map[c]["an"]: na += 1
            acct_cr[acct] = np.mean(ranks) if ranks else 0
            acct_nacp[acct] = na; acct_car[acct] = na / len(cps) if cps else 0
        cc = defaultdict(lambda: [0, 0])
        for i, pk in enumerate(pk_list):
            cid = comm_col[i]
            if cid is None: continue
            cc[cid][0] += 1
            if pk in geo_map and geo_map[pk]["an"]: cc[cid][1] += 1
        comm_anom_rate = {cid: a/t if t > 0 else 0 for cid, (t, a) in cc.items()}

        print(f"  [{csv_prefix}] transactions...", "%.0fs" % (time.time()-t0))
        tx = pcsv.read_csv(str(DATA_DIR / f"{csv_prefix}_Trans.csv"),
            read_options=pcsv.ReadOptions(
                column_names=["timestamp","from_bank","from_account","to_bank","to_account",
                              "amount_received","receiving_currency","amount_paid",
                              "payment_currency","payment_format","is_laundering"], skip_rows=1))
        from_list = tx["from_account"].to_pylist()
        to_list = tx["to_account"].to_pylist()
        amounts = [float(v) if v is not None else 0.0 for v in tx["amount_received"].to_pylist()]
        sent_to = defaultdict(lambda: defaultdict(float))
        acct_cur = defaultdict(set); acct_db = defaultdict(set)
        for f, t_a, amt, cur, tb in zip(from_list, to_list, amounts,
                                        tx["receiving_currency"].to_pylist(),
                                        tx["to_bank"].to_pylist()):
            if f is None or t_a is None: continue
            sent_to[f][t_a] += amt
            if cur: acct_cur[f].add(cur)
            if tb: acct_db[f].add(tb)
        src_mcur = set()
        for acct in sent_to:
            if len(acct_cur.get(acct, set())) < 2: continue
            total = sum(sent_to[acct].values())
            if total > 0:
                ret = sum(sent_to[tgt].get(acct, 0) for tgt in sent_to[acct]
                          if acct in sent_to.get(tgt, {}))
                if ret / total > 0.3: src_mcur.add(acct); continue
            if len(acct_db.get(acct, set())) >= 3: src_mcur.add(acct)
        src_border = {geo["primary_key"][i].as_py() for i in range(geo.num_rows)
                      if not geo["is_anomaly"][i].as_py()
                      and (geo["delta_rank_pct"][i].as_py() or 0) >= 80}
        src_rep = set()
        acct_near = defaultdict(int); acct_tot = defaultdict(int)
        for f, amt in zip(from_list, amounts):
            if f is None: continue
            acct_tot[f] += 1
            if amt > 100:
                nk = round(amt / 1000) * 1000
                if nk > 0 and abs(amt - nk) / nk < 0.02: acct_near[f] += 1
        src_struct = {acct for acct, near in acct_near.items()
                      if acct_tot.get(acct, 1) >= 5 and near / acct_tot.get(acct, 1) > 0.5}
        src_c = {pk_list[i] for i in range(n)
                 if rv["return_ratio"][i] >= 0.20 and pk_list[i] in chain_accounts}
        tx_p50 = np.percentile(rv["tx_out_count"], 50)
        src_d = {pk_list[i] for i in range(n)
                 if rv["structuring_pct"][i] >= 0.50 and rv["amount_uniformity"][i] >= 0.5
                 and rv["tx_out_count"][i] >= tx_p50}
        core = [src_acct, src_pair, src_chain, src_mcur, src_border, src_rep, src_struct]
        all9 = set()
        for src in core + [src_c, src_d]: all9.update(src)
        all7 = set()
        for src in core: all7.update(src)
        full_score = {a: sum(1 for src in core if a in src) for a in all7}
        boosted = {a: full_score.get(a, 0) + (1 if a in high_d else 0) for a in all7 | high_d}
        stage1 = sorted(all9)
        stage3 = sorted(pk for pk, sc in boosted.items() if sc >= 3)
        stage4 = sorted(pk for pk, sc in full_score.items() if sc >= 3 and pk in high_d)
        burst_p90 = np.percentile(rv["burst_tx_out"], 90)
        print(f"  [{csv_prefix}] Stage1={len(stage1)} Stage3={len(stage3)} Stage4={len(stage4)}")
        return {"pk_idx": pk_idx, "rv": rv, "comm_col": comm_col, "geo_map": geo_map,
                "geo_score": geo_score, "acct_anom_pairs": dict(acct_anom_pairs),
                "acct_max_pair_dn": dict(acct_max_pair_dn), "acct_n_chains": dict(acct_n_chains),
                "acct_n_anom_chains": dict(acct_n_anom_chains), "acct_in_cyclic": acct_in_cyclic,
                "tx_density": tx_density, "acct_cr": acct_cr, "acct_nacp": acct_nacp,
                "acct_car": acct_car, "comm_anom_rate": comm_anom_rate,
                "acct_counterparties": dict(acct_counterparties),
                "in_anom_pair": in_anom_pair, "max_hops": dict(max_hops),
                "min_ts": dict(min_ts), "max_ndc": dict(max_ndc), "low_cv": low_cv,
                "burst_p90": burst_p90,
                "stage1": stage1, "stage3": stage3, "stage4": stage4}


def build_features(pks, d, gt_pat, gt_full):
    pk_idx = d["pk_idx"]; rv = d["rv"]; geo_map = d["geo_map"]; geo_score = d["geo_score"]
    comm_col = d["comm_col"]; comm_anom_rate = d["comm_anom_rate"]
    X = np.zeros((len(pks), len(feat_names)), dtype=np.float32)
    yp = np.zeros(len(pks), dtype=np.int32); yf = np.zeros(len(pks), dtype=np.int32)
    for j, pk in enumerate(pks):
        if pk not in pk_idx: continue
        i = pk_idx[pk]
        for k, col in enumerate(raw_cols): X[j, k] = rv[col][i]
        gm = geo_map.get(pk, {"rk": 0, "cp": 1.0, "an": 0})
        X[j, go+0] = gm["rk"]; X[j, go+1] = gm["cp"]; X[j, go+2] = gm["an"]
        X[j, go+3] = geo_score.get(pk, 0)
        X[j, go+4] = d["acct_anom_pairs"].get(pk, 0)
        X[j, go+5] = d["acct_max_pair_dn"].get(pk, 0)
        X[j, go+6] = d["acct_n_chains"].get(pk, 0)
        X[j, go+7] = d["acct_n_anom_chains"].get(pk, 0)
        X[j, go+8] = 1.0 if pk in d["acct_in_cyclic"] else 0.0
        X[j, go+9] = d["tx_density"].get(pk, 0)
        X[j, go+10] = d["acct_cr"].get(pk, 0)
        X[j, go+11] = d["acct_nacp"].get(pk, 0)
        X[j, go+12] = d["acct_car"].get(pk, 0)
        cid = comm_col[i]
        X[j, go+13] = comm_anom_rate.get(cid, 0) if cid else 0
        X[j, go+14] = len(d["acct_counterparties"].get(pk, set()))
        yp[j] = 1 if pk in gt_pat else 0
        yf[j] = 1 if pk in gt_full else 0
    return X, yp, yf


def check_typo(pk, d):
    pk_idx = d["pk_idx"]; rv = d["rv"]
    if pk not in pk_idx: return 0
    i = pk_idx[pk]; h = 0
    def V(col): return rv[col][i]
    burst_p90 = d["burst_p90"]
    in_cyclic = d["acct_in_cyclic"]; in_anom_pair = d["in_anom_pair"]
    min_ts = d["min_ts"]; max_ndc = d["max_ndc"]; max_hops = d["max_hops"]
    low_cv = d["low_cv"]
    if V("amount_uniformity") > 0.7 and V("n_currencies_out") >= 2 and V("n_dest_banks") >= 3: h += 1
    if V("burst_tx_out") > burst_p90 and V("n_currencies_out") >= 2 and pk in in_cyclic and min_ts.get(pk, 9999) < 24: h += 1
    if pk in in_cyclic and min_ts.get(pk, 9999) <= 24 and max_ndc.get(pk, 0) >= 2: h += 1
    if pk in in_anom_pair and V("counterpart_overlap") > 0: h += 1
    if max_hops.get(pk, 0) >= 4 and max_ndc.get(pk, 0) >= 2: h += 1
    if V("intermediary_score") > 0.3 and V("n_currencies_out") >= 2: h += 1
    if V("return_ratio") > 0.3: h += 1
    if V("n_payment_formats") >= 4 and V("n_currencies_out") >= 2: h += 1
    if V("in_degree") >= 10 and V("fan_asymmetry") < 0.3: h += 1
    if V("n_dest_banks") >= 5 and V("n_currencies_out") >= 3: h += 1
    if pk in low_cv: h += 1
    if V("counterpart_overlap") > 0 and V("return_ratio") > 0.2: h += 1
    if V("tx_out_count") > 50 and V("amount_uniformity") > 0.5 and V("n_currencies_out") >= 2: h += 1
    if V("tx_out_count") > 100 and V("mean_out") < 500 and V("n_out_targets") >= 10 and V("n_currencies_out") >= 2: h += 1
    return h


gt_hi_pat, gt_hi_full = load_gt("HI-Small")
gt_li_pat, gt_li_full = load_gt("LI-Small")
print("HI-Small GT known=%d full=%d" % (len(gt_hi_pat), len(gt_hi_full)))
print("LI-Small GT known=%d full=%d" % (len(gt_li_pat), len(gt_li_full)))

print("Loading HI-Small (train)...")
hi = load_sphere(SPHERE_HI, "HI-Small")
print("Loading LI-Small (test)...")
li = load_sphere(SPHERE_LI, "LI-Small")

for stream_name, hi_pks, li_pks, thresholds in [
    ("Stream A: Stage 1 (high-recall)", hi["stage1"], li["stage1"], [0.10, 0.20]),
    ("Stream B: Stage 3 (daily-ops)",   hi["stage3"], li["stage3"], [0.10, 0.20]),
    ("Stream C: Stage 4 (escalation)",  hi["stage4"], li["stage4"], [0.20, 0.50]),
]:
    X_train, yp_train, _ = build_features(hi_pks, hi, gt_hi_pat, gt_hi_full)
    X_test, yp_test, yf_test = build_features(li_pks, li, gt_li_pat, gt_li_full)
    print()
    print("=== %s: %d suspects, %d TP known, %d TP full ===" % (
        stream_name, len(li_pks), int(yp_test.sum()), int(yf_test.sum())))
    for thresh in thresholds:
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=len(yp_train[yp_train==0])/max(len(yp_train[yp_train==1]), 1),
            eval_metric="logloss", verbosity=0, random_state=42)
        m.fit(X_train, yp_train)
        pr = m.predict_proba(X_test)[:, 1]
        flagged = {li_pks[i] for i, p in enumerate(pr) if p >= thresh}
        with_t = {pk for pk in flagged if check_typo(pk, li) >= 1}
        no_t = flagged - with_t
        tp_x = len(flagged & gt_li_pat); fp_x = len(flagged) - tp_x
        tp_xf = len(flagged & gt_li_full); fp_xf = len(flagged) - tp_xf
        tp_a = len(with_t & gt_li_pat); fp_a = len(with_t) - tp_a
        tp_af = len(with_t & gt_li_full); fp_af = len(with_t) - tp_af
        tp_n = len(no_t & gt_li_pat)
        print("  p>=%.2f:" % thresh)
        print("    XGBoost:       %5d  rec(kn)=%.1f%% FP/TP=%.1f  rec(fl)=%.1f%% FP/TP=%.1f" % (
            len(flagged), tp_x/len(gt_li_pat)*100, fp_x/tp_x if tp_x else 999,
            tp_xf/len(gt_li_full)*100, fp_xf/tp_xf if tp_xf else 999))
        print("    + filter:       %5d  rec(kn)=%.1f%% FP/TP=%.1f  rec(fl)=%.1f%% FP/TP=%.1f" % (
            len(with_t), tp_a/len(gt_li_pat)*100, fp_a/tp_a if tp_a else 999,
            tp_af/len(gt_li_full)*100, fp_af/tp_af if tp_af else 999))
        print("    no typology:   %5d  TP(kn)=%4d  TP(fl)=%4d" % (len(no_t), tp_n, len(no_t & gt_li_full)))

print("\nElapsed: %.0fs" % (time.time()-t0))
