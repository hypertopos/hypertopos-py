# Passive Scan Calibration — Layer 1 (Geometry)

Calibration report for the unsupervised geometry layer on IBM AML HI-Small.
For full 3-layer pipeline results see [`README.md`](README.md).

| | |
|---|---|
| **Dataset** | IBM AML HI-Small — 5M tx, 515K accounts, 3,170 known laundering |
| **Sphere** | `benchmark/ibm-aml/hi_small_sphere/gds_aml_hi_small` |
| **Builder** | `hypertopos build --config benchmark/ibm-aml/hi_small_sphere/sphere.yaml` |
| **Scan** | [`layer1_passive_scan.py`](layer1_passive_scan.py) |

> **Reproducibility note:** Per-pattern recall is sensitive to theta recalibration.
> A full `--force` sphere rebuild recomputes population statistics from scratch — anomaly
> boundaries shift, and individual accounts move in/out of geometry sources. Expect
> per-pattern recall variance of ±10pp across rebuilds; Stage 1 overall recall is more
> stable (±3pp); Stages 2-4 are stable (multi-source scoring averages out the noise).
> Treat all numbers as indicative ranges, not exact fixed values.

---

## Sphere

| Pattern | Dims | Entities | Description |
|---------|------|----------|-------------|
| `account_pattern` | 27D | 515,080 | 9 outgoing + 4 incoming + 2 burst + 4 graph + 5 ratio + 1 currency + 2 IET |
| `account_pairs_pattern` | 6D | 1,015,736 | Per from→to relationship geometry |
| `tx_chains_pattern` | 9D | 300,000 | Bidirectional BFS chains |
| `tx_pattern` | 8D event | 5,078,345 | Per-tx: amount, z-scores, cross-currency, new-counterparty, time-interval |
| Temporal | 9 windows × 2d | all anchor patterns | Deformation history |

> **Chain coverage note:** Only 300K chains were generated for this benchmark run. Full BFS
> expansion would produce significantly more chains, increasing entity coverage for chain-based
> sources (source 3, stream C). Expect chain anomaly recall to improve with higher chain limits —
> particularly FAN-OUT (passive receivers rely on chain membership) and CYCLE patterns.

---

## Detection Sources

### Core geometry sources (1-3)

| # | Source | Logic | Suspects | +TP |
|---|--------|-------|----------|-----|
| 1 | Account anomaly | `is_anomaly=True` (per-group theta, 95th pct per entity_type) | 25,757 | +619 |
| 2 | Pair anomaly | `is_anomaly=True` on composite A→B pairs, split to accounts | 68,979 | +411 |
| 3 | Chain anomaly | `is_anomaly=True` on chains, expand via entity_keys | 18,682 | +656 |

### Heuristic sources (4-7)

| # | Source | Logic | Suspects | +TP |
|---|--------|-------|----------|-----|
| 4 | Multi-cur cross-border | return_ratio >30% + multi-cur OR ≥3 dest banks + multi-cur | 7,823 | +473 |
| 5 | Borderline | `delta_rank_pct >= 80` AND NOT `is_anomaly` | 77,259 | +176 |
| 6 | Reputation (chronic) | `reputation >= 0.5` AND `anomaly_tenure >= 2` (max-consecutive-run) | 785 | +0 |
| 7 | Structuring | >50% of amounts within 2% of round thousands, ≥5 tx | 30,926 | +32 |

### Behavioral typology streams (C, D)

| # | Source | Logic | Suspects | Unique TP vs 1-7 |
|---|--------|-------|----------|:----------------:|
| C | CYCLE/round-trip | `return_ratio >= 0.20 AND account in any chain` | 74,389 | +136 |
| D | STRUCTURING | `structuring_pct >= 0.50 AND amount_uniformity >= 0.5 AND tx_out >= p50` | 11,331 | +46 |

### Tx event density (scoring booster)

Per-account metric: `anomalous_tx_ratio = anomalous_transactions / total_transactions`.
134,132 accounts with anomalous tx. Distribution: p50=0.097, p90=0.500.
Used as +1 score boost when density >= 0.05.

---

## Four-Stage Operating Model

Each stage is a strict subset of the previous.

| Stage | Config | Recall | FP/TP | Suspects | Purpose |
|-------|--------|:------:|:-----:|:--------:|---------|
| **1. Surveillance** | OR(sources 1-7 + C + D) | **80.4%** | 74.4 | 192K | Regulatory monitoring, back-testing, SAR post-hoc |
| **2. Confirmed** | score >= 2 on sources 1-7 | **59.6%** | 29.0 | ~57K | Scored alert queue, pre-filtering |
| **3. Investigation** | boost(d>=0.05) + score >= 3 on sources 1-7 | **50.8%** | 21.2 | ~35K | Daily case-team, escalation |
| **4. Critical** | score >= 3 + density filter >= 0.05 on sources 1-7 | **30.2%** | 14.2 | ~14K | Immediate escalation, high-confidence SAR |

**Stage 1** uses all 9 sources in OR — maximum coverage. Streams C and D add +182
accounts invisible to geometry (cycle participants and structuring accounts with
moderate overall activity).

**Stages 2-4** use only 7 core sources (1-7, excluding streams C/D). Streams C/D are
included only in Stage 1 OR — their unique TP rarely overlap with other sources
and don't reach score >= 3. Tx density booster (Stage 3) or hard filter (Stage 4)
separates legit high-volume accounts from structurally anomalous ones.

**Exact stage definitions:**

```
Stage 1: OR(all 9 sources incl. C, D)  — any source flags the account
Stage 2: score >= 2 on 7 core sources (1-7) — multi-source confirmation
Stage 3: boost(density >= 0.05, +1 score) then score >= 3 on 7 core sources
          density = anomalous_tx / total_tx per account from tx_pattern geometry
Stage 4: score >= 3 on 7 core sources AND density >= 0.05
          accounts WITHOUT anomalous tx data are EXCLUDED (no pass-through)
```

**Score computation:** for each account, count how many of the 7 core sources
(1. account anomaly, 2. pair anomaly, 3. chain anomaly, 4. multi-cur, 5. borderline,
6. reputation, 7. structuring) flag it. Streams C/D contribute to Stage 1 only.

**Tx density:** `anomalous_tx_ratio = count(is_anomaly=true in tx_pattern geometry for
this account) / count(all tx for this account)`. Computed from tx_pattern geometry
entity_keys expansion. Accounts with 0 anomalous tx have density=0.

### Per-pattern recall (Stage 1)

| Pattern | GT | Recall |
|---------|---:|-------:|
| FAN-OUT | 359 | 65.2% |
| FAN-IN | 1,007 | 81.2% |
| CYCLE | 271 | 90.4% |
| GATHER | 369 | 92.1% |
| BIPARTITE | 491 | 77.4% |
| STACK | 663 | 80.1% |
| RANDOM | 211 | 91.5% |

Hardest: FAN-OUT (passive destination leaves) and BIPARTITE (peripheral 2-3 tx accounts).
Easiest: GATHER (92.1%) and RANDOM (91.5%) — strongest geometric signatures via currency
diversity and chain membership.

---

## Layer 1 Results (geometry only, no labels)

| Stage | Suspects | Recall (known) | FP/TP |
|-------|:--------:|:--------------:|:-----:|
| 1. Surveillance (OR 9-src) | 192K | 80.4% | 74.4 |
| 2. Confirmed (score>=2) | ~57K | 59.6% | 29.0 |
| 3. Investigation (boost+score>=3) | ~35K | 50.8% | 21.2 |
| 4. Critical (score>=3+density) | ~14K | 30.2% | 14.2 |

This is the passive detection layer only. Layer 2 (XGBoost) and Layer 3 (agent typology)
operate on top — see [`README.md`](README.md) for full pipeline comparison.

---

## Calibration

Steps explored to reach the current operating model. Each was measured against
known patterns ground truth on HI-Small.

### Engine-level scoring (15 configs tested)

| Change | Impact | Verdict |
|--------|--------|---------|
| Per-group theta (entity_type) | Changes who is flagged, not how many | Enabled, marginal |
| Kurtosis dimension weights | Amplifies noise dims (amount, count) | Rejected |
| GMM k=5 per-cluster theta | Changes cluster assignment, not FP/TP | Rejected |
| Mahalanobis distance | Correlations too weak in 27D | Enabled, no measurable impact |
| IET features (iet_mean, iet_std) | Signal already captured by count/burst | Included, marginal |
| Rolling z-scores | Doesn't distinguish fraud from legit activity changes | Rejected |
| Embedded raw data dims (structuring, return_ratio) | Already caught by heuristic sources | Included for geometry completeness |
| Per-dim anomaly count (n_anomalous_dims) | Strict subset of is_anomaly | Rejected |
| Conformal p filtering | Cuts TP and FP proportionally | Not useful standalone |
| anomaly_percentile=99 | Only affects source 1 | Rejected |

**Conclusion:** At 95th percentile, 5% of each population is flagged. More sophisticated
scoring changes WHICH 5% but not HOW MANY.

### Source selection (10 tested, 7 retained + 2 streams)

| Source | Result | Status |
|--------|--------|--------|
| Account anomaly (is_anomaly) | 619 TP, primary geometric source | **Retained** |
| Pair anomaly (95th pct) | +411 TP from relationship-level anomalies | **Retained** |
| Chain anomaly | +656 TP from chain membership expansion | **Retained** |
| Multi-cur cross-border | +473 TP, top heuristic source | **Retained** |
| Borderline (rank>=80) | +176 TP near-threshold accounts | **Retained** |
| Reputation (chronic) | 785 suspects, 0 new TP (subset of geometry) | **Retained** (temporal signal) |
| Structuring (>50% round amounts) | +32 TP | **Retained** |
| Session anomaly (burst in 4h window) | 0 unique TP at any threshold config | **Removed** |
| Currency imbalance (dominant >80%) | 0 unique TP vs sources 1-7 | **Removed** |
| Bidir+multi-cur / multi-fmt / acceleration | 0 TP on HI-Small | **Removed** |
| Stream C: CYCLE/round-trip | +136 unique TP | **Retained** (Stage 1 only) |
| Stream D: STRUCTURING behavioral | +46 unique TP | **Retained** (Stage 1 only) |
| Stream A: FAN-OUT behavioral | 49K suspects, 39 unique TP — too noisy | **Rejected** |
| Stream B: FAN-IN/GATHER behavioral | 49K suspects, 75 unique TP — too noisy | **Rejected** |
| Stream E: HIGH-VOLUME cross-border | 0 unique TP (redundant with source 4) | **Rejected** |

### Dimension exploration (27D final)

| Dimension group | Dims | Impact on recall | Status |
|-----------------|------|:----------------:|--------|
| Outgoing (tx_out, n_targets, n_banks, n_cur, n_fmt, amount_std, sum, max, mean) | 9 | Core signal | Included |
| Incoming (tx_in, n_sources, n_src_banks, sum_in) | 4 | Moderate | Included |
| Temporal burst (burst_tx_out, burst_tx_in) | 2 | Marginal | Included |
| Graph features (in_degree, out_degree, reciprocity, counterpart_overlap) | 4 | Moderate | Included |
| Ratio features (intermediary, fan_asymmetry, amount_uniformity, structuring_pct, return_ratio) | 5 | Marginal | Included |
| n_currencies_in | 1 | Moderate | Included |
| IET (avg_gap, tx_regularity) | 2 | Marginal — signal in count/burst | Included |

### Per-typology detector analysis (rejected)

Tested dedicated 3-5 dim detectors per pattern type. Finding: top dims overlap across
ALL 7 patterns — `_d_n_currencies_out` dominates everything (2.8-9.7x separation ratio).
No distinct per-typology geometric signatures.

Union of all per-typology p95 detectors: 45.4% recall, FP/TP=133 — worse than
multi-source pipeline.

### Diagnostic findings

| Finding | Detail |
|---------|--------|
| FAN-OUT destination blindspot | 316 passive receivers, 1-2 outgoing tx, median rank=25. Source-side recall 76%, destination 15%. |
| Pair FP explosion | theta=3.30 too low, 51% anomalies barely above threshold. Hub accounts inflate via 1000+ pairs. |
| Reputation temporal structure | Laundering stops at slice 5/9. Fixed tenure to max-consecutive-run. 785 suspects, 0 new unique TP. |
| `_d_n_currencies_out` dominance | Single dim drives detection across all 7 pattern types. Univariate threshold nearly matches 5-dim detectors. |
| Reciprocity degenerate | p80=p90=p95=1.0 — zero discriminating power in this dataset. |
| tx_regularity degenerate | p10=0.0 — >10% of accounts have regularity=0. |

### Scoring strategy evolution

| Approach | FP/TP | Recall | Suspects | → Stage |
|----------|:-----:|:------:|:--------:|---------|
| **OR 9-src (current 27D sphere)** | **74.4** | **80.4%** | **192K** | **Stage 1 — Surveillance** |
| **Multi-source score >= 2** | **29.0** | **59.6%** | **~57K** | **Stage 2 — Confirmed** |
| Multi-source score >= 3 | 14.2 | 40.1% | ~19K | Too aggressive alone |
| Density as hard filter (score>=2 + density>=0.05) | 23.5 | 44.9% | ~35K | Cuts TP proportionally |
| **Density as score booster (boost+score>=3)** | **21.2** | **50.8%** | **~35K** | **Stage 3 — Investigation** |
| **Score >= 3 + density filter >= 0.05** | **14.2** | **30.2%** | **~14K** | **Stage 4 — Critical** |

Key insight: density works as BOOSTER (+1 to score) not FILTER. Account with score=2
plus high tx density gets promoted to score=3. Legit high-volume accounts without
tx anomalies stay at score=2 and drop out.