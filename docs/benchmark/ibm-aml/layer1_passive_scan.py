# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
"""Multi-source passive scan v2 — uses extended PassiveScanner for all sources.

v1 → v2 migration:
  Sources 1-3: unchanged (PassiveScanner direct/composite/chain)
  Source 4: multi-cur cross-border → compound (return_ratio + n_currencies_out)
  Source 5: borderline → add_borderline_source
  Source 6: reputation → UNCHANGED (requires temporal scan)
  Source 7: structuring → add_points_source (structuring_pct precomputed)

Usage:
    .venv/Scripts/python benchmark/ibm-aml/layer1_passive_scan_v2.py
    .venv/Scripts/python benchmark/ibm-aml/layer1_passive_scan_v2.py \
        --sphere benchmark/ibm-aml/li_small_sphere/gds_aml_li_small --dataset LI-Small
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hypertopos"))

from hypertopos import HyperSphere
from hypertopos.navigation.scanner import PassiveScanner

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ibm-aml"
DEFAULT_SPHERE = str(
    Path(__file__).resolve().parent / "hi_small_sphere" / "gds_aml_hi_small"
)


def load_ground_truth(dataset: str):
    gt_patterns = set()
    patterns_by_type = defaultdict(set)
    current_type = None
    with open(DATA_DIR / f"{dataset}_Patterns.txt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                current_type = line.split("-")[-1].strip().split(":")[0].strip()
            elif line.startswith("END"):
                current_type = None
            elif current_type and line:
                parts = line.split(",")
                if len(parts) >= 5:
                    patterns_by_type[current_type].add(parts[2])
                    patterns_by_type[current_type].add(parts[4])
                    gt_patterns.add(parts[2])
                    gt_patterns.add(parts[4])

    gt_full_file = DATA_DIR / f"{dataset}_laundering_accounts.txt"
    gt_full = None
    if gt_full_file.exists():
        gt_full = set()
        with open(gt_full_file) as f:
            for line in f:
                gt_full.add(line.strip())

    return gt_patterns, gt_full, patterns_by_type


def _print_step(label, src, cum_set, gt, prev_tp_count):
    tp = len(cum_set & gt)
    fp = len(cum_set - gt)
    new = tp - prev_tp_count
    fp_per_tp = fp / tp if tp > 0 else 0
    print(
        f"  {label:<32} {len(src):>8,} suspects  "
        f"+{new:<4} TP  cum={tp:<5}  "
        f"recall={tp/len(gt)*100:.1f}%  FP/TP={fp_per_tp:.1f}"
    )
    return tp


def run_scan(sphere_path: str, dataset: str = "HI-Small"):
    t0 = time.time()
    gt_patterns, gt_full, patterns_by_type = load_ground_truth(dataset)
    name_map = {
        "Out": "FAN-OUT", "In": "FAN-IN", "CYCLE": "CYCLE",
        "GATHER": "GATHER", "BIPARTITE": "BIPARTITE",
        "STACK": "STACK", "RANDOM": "RANDOM",
    }

    print(f"Dataset: {dataset}")
    print(f"GT (Patterns.txt): {len(gt_patterns):,} accounts")
    if gt_full:
        print(f"GT (is_laundering=1): {len(gt_full):,} accounts")
    print()

    gt = gt_patterns

    hs = HyperSphere.open(sphere_path)
    with hs.session("scan") as s:
        reader = s._reader
        sphere = reader.read_sphere()

        scanner = PassiveScanner(reader, sphere, s._manifest)

        # Sources 1-3: geometry (same as v1)
        scanner.add_source("account_pattern", "account_pattern", key_type="direct")
        scanner.add_source(
            "account_pairs_pattern", "account_pairs_pattern", key_type="composite",
        )
        scanner.add_source(
            "tx_chains_pattern", "tx_chains_pattern", key_type="chain",
        )

        # Source 4: multi-cur cross-border (points rules — no geometry needed)
        # 4a: return_ratio > 0.3 AND n_currencies_out >= 2
        scanner.add_points_source(
            "multi_cur_rt",
            line_id="accounts",
            rules={
                "return_ratio": (">", 0.3),
                "n_currencies_out": (">=", 2),
            },
            combine="AND",
        )
        # 4b: n_dest_banks >= 3 AND n_currencies_out >= 2
        scanner.add_points_source(
            "multi_cur_banks",
            line_id="accounts",
            rules={
                "n_dest_banks": (">=", 3),
                "n_currencies_out": (">=", 2),
            },
            combine="AND",
        )

        # Source 5: borderline
        scanner.add_borderline_source(
            "borderline", "account_pattern", rank_threshold=80,
        )

        # Source 7: structuring — must match v1: total_tx >= 5 AND pct > 0.5
        scanner.add_points_source(
            "structuring", "accounts",
            rules={
                "structuring_pct": (">", 0.5),
                "tx_out_count": (">=", 5),
            },
            combine="AND",
        )

        # Stream C: CYCLE/round-trip = return_ratio >= 0.40 AND account in any chain
        scanner.add_compound_source(
            "stream_c_cycle",
            geometry_pattern_id="tx_chains_pattern",
            line_id="accounts",
            rules={"return_ratio": (">=", 0.2)},
            geometry_key_type="chain",
            geometry_filter_expr="",  # all chains, not just anomalous
        )

        # Stream D: structuring_pct >= 0.5 AND amount_uniformity >= 0.5
        #           AND tx_out_count >= p50 (compute p50 dynamically)
        import pyarrow.compute as pc
        pts_table = reader.read_points("accounts", sphere.lines["accounts"].current_version())
        if "tx_out_count" in pts_table.schema.names:
            tx_out_arr = pc.fill_null(pts_table["tx_out_count"], 0)
            tx_out_p50 = float(
                pc.quantile(tx_out_arr, q=[0.5])[0].as_py()
            )
        else:
            tx_out_p50 = 5.0  # fallback

        scanner.add_points_source(
            "stream_d_struct",
            line_id="accounts",
            rules={
                "structuring_pct": (">=", 0.5),
                "amount_uniformity": (">=", 0.5),
                "tx_out_count": (">=", tx_out_p50),
            },
            combine="AND",
        )

        # Run all scanner sources
        t_scan = time.time()
        scan_result = scanner.scan("accounts", scoring="count", threshold=1)
        print(f"  PassiveScanner: {time.time() - t_scan:.1f}s")

        # Extract per-source sets
        source_sets: dict[str, set] = {}
        for src_name in scan_result.sources_summary:
            source_sets[src_name] = {
                h.primary_key for h in scan_result.hits
                if src_name in h.sources
            }

        # Step-by-step reporting (all 9 sources)
        core_sources = [
            ("account_pattern", "1. Account anomaly"),
            ("account_pairs_pattern", "2. Pair anomaly"),
            ("tx_chains_pattern", "3. Chain anomaly"),
            ("multi_cur_rt", "4a. Multi-cur RT>=0.3"),
            ("multi_cur_banks", "4b. Multi-cur banks>=3"),
            ("borderline", "5. Borderline (rank>=80)"),
            ("structuring", "7. Structuring (pct>=0.5)"),
        ]
        stream_sources = [
            ("stream_c_cycle", "C. CYCLE/round-trip"),
            ("stream_d_struct", "D. STRUCTURING compound"),
        ]
        cum = set()
        prev_tp = 0
        for name, label in core_sources + stream_sources:
            src = source_sets.get(name, set())
            cum |= src
            prev_tp = _print_step(label, src, cum, gt, prev_tp)

        # Source 6: reputation (still manual — requires temporal scan)
        src_reputation = set()
        temporal_dir = Path(sphere_path) / "temporal" / "account_pattern" / "data.lance"
        if temporal_dir.exists():
            import lance
            from hypertopos.engine.geometry import GDSEngine
            pattern = sphere.patterns["account_pattern"]
            theta_norm = float(np.linalg.norm(pattern.theta))
            _sigma = np.maximum(pattern.sigma_diag, 1e-2)
            n_rel = len(pattern.relations)
            ds = lance.dataset(str(temporal_dir))
            t_table = ds.to_table(columns=["primary_key", "shape_snapshot"])
            pk_norms: dict[str, list[float]] = defaultdict(list)
            for pk, shape in zip(
                t_table["primary_key"].to_pylist(),
                t_table["shape_snapshot"].to_pylist(),
            ):
                if pk is None or shape is None:
                    continue
                sa = np.array(shape, dtype=np.float32)
                delta = (sa - pattern.mu) / _sigma
                dn = float(np.linalg.norm(delta[:n_rel]))
                pk_norms[pk].append(dn)
            for pk, norms in pk_norms.items():
                if len(norms) < 3:
                    continue
                arr = np.array(norms, dtype=np.float32)
                rep = GDSEngine.compute_reputation(arr, theta_norm)
                if rep["reputation"] >= 0.5 and rep["anomaly_tenure"] >= 2:
                    src_reputation.add(pk)
        cum |= src_reputation
        prev_tp = _print_step("6. Reputation (chronic)", src_reputation, cum, gt, prev_tp)

        # Results
        print(f"\n{'Pattern':<15} {'GT':>5} {'Found':>6} {'Recall':>8}")
        print("-" * 38)
        for pt in ["Out", "In", "CYCLE", "GATHER", "BIPARTITE", "STACK", "RANDOM"]:
            gs = patterns_by_type[pt]
            if not gs:
                continue
            found = gs & cum
            print(
                f"{name_map[pt]:<15} {len(gs):>5} {len(found):>6} "
                f"{len(found)/len(gs)*100:>7.1f}%"
            )
        print("-" * 38)

        for gt_name, gt_set in [
            ("Patterns.txt", gt_patterns),
            ("Full (is_laundering)", gt_full),
        ]:
            if gt_set is None:
                continue
            tp = cum & gt_set
            fp = cum - gt_set
            fpt = len(fp) / len(tp) if tp else 0
            print(f"\nGT: {gt_name} ({len(gt_set):,} accounts)")
            print(
                f"  TP: {len(tp):,}  FP: {len(fp):,}  "
                f"Suspects: {len(cum):,}"
            )
            print(
                f"  Recall: {len(tp)/len(gt_set)*100:.1f}%  "
                f"Precision: {len(tp)/len(cum)*100:.2f}%  "
                f"FP/TP: {fpt:.1f}"
            )

        # === TX DENSITY: per-account anomalous tx metrics ===
        tx_density: dict[str, float] = {}
        if "tx_pattern" in sphere.patterns:
            tx_geo_anom = reader.read_geometry(
                "tx_pattern", 1,
                columns=["entity_keys", "delta_norm"],
                filter="is_anomaly = true",
            )
            tx_anom_count: dict[str, int] = {}
            for ek, dn in zip(
                tx_geo_anom["entity_keys"].to_pylist(),
                tx_geo_anom["delta_norm"].to_pylist(),
            ):
                if not ek or dn is None:
                    continue
                for acct_key in ek:
                    if acct_key:
                        tx_anom_count[acct_key] = (
                            tx_anom_count.get(acct_key, 0) + 1
                        )

            tx_geo_all = reader.read_geometry(
                "tx_pattern", 1, columns=["entity_keys"],
            )
            tx_total_count: dict[str, int] = {}
            for ek in tx_geo_all["entity_keys"].to_pylist():
                if ek:
                    for acct_key in ek:
                        if acct_key:
                            tx_total_count[acct_key] = (
                                tx_total_count.get(acct_key, 0) + 1
                            )

            for acct, anom in tx_anom_count.items():
                total = tx_total_count.get(acct, 1)
                tx_density[acct] = anom / total

        # Multi-source scoring — 7 core sources only (excl streams C/D)
        core_source_names = [n for n, _ in core_sources]
        core_sets = [source_sets.get(n, set()) for n in core_source_names]
        core_sets.append(src_reputation)  # source 6
        all_accounts = set()
        for src in core_sets:
            all_accounts |= src
        acct_scores = {
            a: sum(1 for src in core_sets if a in src) for a in all_accounts
        }
        # Stage 1 includes C/D streams too
        all_sources_incl_streams = core_sets + [
            source_sets.get(n, set()) for n, _ in stream_sources
        ]
        stage1_accounts = set()
        for src in all_sources_incl_streams:
            stage1_accounts |= src

        # === 4 STAGES ===
        print("\n=== Operating Stages ===")

        def _stage_report(label, suspects):
            for gt_name, gt_set in [
                ("Patterns", gt_patterns), ("Full", gt_full),
            ]:
                if gt_set is None:
                    continue
                tp = suspects & gt_set
                fp = suspects - gt_set
                fpt = len(fp) / len(tp) if tp else 0
                print(
                    f"  {label:<30} {len(suspects):>7,} suspects  "
                    f"TP={len(tp):>5,}  recall={len(tp)/len(gt_set)*100:.1f}%  "
                    f"FP/TP={fpt:.1f}  ({gt_name})"
                )

        # Stage 1: Surveillance — all 9 sources in OR
        stage1 = stage1_accounts
        _stage_report("Stage 1: Surveillance (9-src)", stage1)

        # Stage 2: Confirmed — score >= 2
        stage2 = {a for a, sc in acct_scores.items() if sc >= 2}
        _stage_report("Stage 2: Confirmed (score>=2)", stage2)

        # Stage 3: Investigation — boost(density>=0.05) + score >= 3
        boosted = {}
        high_density = {a for a, d in tx_density.items() if d >= 0.05}
        for acct in all_accounts | high_density:
            base = acct_scores.get(acct, 0)
            boost = 1 if acct in high_density else 0
            boosted[acct] = base + boost
        stage3 = {a for a, sc in boosted.items() if sc >= 3}
        _stage_report("Stage 3: Investigation", stage3)

        # Stage 4: Critical — score >= 3 + density filter >= 0.05
        stage4 = {
            a for a, sc in acct_scores.items()
            if sc >= 3 and tx_density.get(a, 0) >= 0.05
        }
        _stage_report("Stage 4: Critical", stage4)

        # Score distribution
        print("\n--- Multi-source score distribution ---")
        for min_score in [1, 2, 3, 4]:
            scored = {a for a, sc in acct_scores.items() if sc >= min_score}
            for gt_name, gt_set in [("Patterns", gt_patterns), ("Full", gt_full)]:
                if gt_set is None:
                    continue
                tp = scored & gt_set
                fp = scored - gt_set
                fpt = len(fp) / len(tp) if tp else 0
                print(
                    f"  score>={min_score}: {len(scored):>7,} suspects  "
                    f"TP={len(tp):>5,}  recall={len(tp)/len(gt_set)*100:.1f}%  "
                    f"FP/TP={fpt:.1f}  ({gt_name})"
                )

        # Per-pattern recall
        print(f"\n{'Pattern':<15} {'GT':>5} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6}")
        print("-" * 50)
        for pt in ["Out", "In", "CYCLE", "GATHER", "BIPARTITE", "STACK", "RANDOM"]:
            gs = patterns_by_type[pt]
            if not gs:
                continue
            print(
                f"{name_map[pt]:<15} {len(gs):>5} "
                f"{len(gs & stage1):>6} {len(gs & stage2):>6} "
                f"{len(gs & stage3):>6} {len(gs & stage4):>6}"
            )

        print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sphere", default=DEFAULT_SPHERE)
    parser.add_argument("--dataset", default="HI-Small")
    args = parser.parse_args()
    run_scan(args.sphere, args.dataset)
