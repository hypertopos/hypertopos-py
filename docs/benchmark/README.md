# Validation Suite

Autonomous agent validation experiments for hypertopos GDS. Each experiment
answers a specific question about the system — not a leaderboard competition.

**Goals:**
- **GDS capability** — can an AI agent autonomously explore a sphere, find anomalies, and produce an actionable report?
- **Skill calibration** — do hypertopos-skills workflows improve agent performance over cold-start?
- **MCP interface quality** — how well do MCP tool descriptions alone guide autonomous investigation?
- **Domain generalization** — does the geometric approach work beyond the financial data it was developed on?
- **Detection pipeline** — can geometry + ML + rules produce operationally useful suspect lists? (IBM AML only — the one true benchmark with ground truth and baselines.)

**Agent model:** Claude Sonnet 4.6 (200K context) for all experiments.
**Scorer model:** Claude Opus 4.6 (1M context) — independent, receives only the report.

## Spheres

| Sphere | Path | Scale | Patterns | Temporal | Data |
|--------|------|-------|----------|----------|------|
| **Berka Banking** | `benchmark/berka/sphere/gds_berka_banking` | 1M tx, 4.5K accounts | 5 | 3 patterns | Real (Czech bank, PKDD 1999) |
| **NYC Yellow Taxi** | `benchmark/nyc-taxi/sphere/gds_nyc_taxi` | 7.6M trips, 265 zones | 3 | 2 patterns | Real (TLC, Jan 2019) |
| **IBM AML HI-Small** | `benchmark/ibm-aml/hi_small_sphere/gds_aml_hi_small` | 5M tx, 515K accounts | 3 + tx | 9 windows | Synthetic (IBM) |
| **IBM AML LI-Small** | `benchmark/ibm-aml/li_small_sphere/gds_aml_li_small` | 6.9M tx, 706K accounts | 3 + tx | 9 windows | Synthetic (IBM) |

## Experiments

### Berka Banking — Skill calibration study

Agent explores a Czech banking sphere (76 real loan defaults) across 6 runs.
Measures: does skill iteration improve recall? What is the MCP-only baseline?

| Experiment | Agent setup | Scoring | Version |
|-----------|------------|---------|---------|
| **A — Cold-start** | Manual mode, MCP tools only, no skills | Max 24 | 0.1.0 |
| **B — Manual + skills** | Manual mode, MCP tools + hypertopos-skills | Max 33 | 0.1.0 |

Details: [`benchmark/berka/README.md`](berka/README.md)

### NYC Yellow Taxi — Domain generalization test + MCP calibration

Agent explores NYC taxi data (7.6M trips) — a non-financial, non-synthetic dataset.
Tests whether geometric navigation generalizes beyond the financial data GDS was
developed on. No ground truth labels; findings are empirically verifiable in raw
TLC data. Secondary: MCP interface quality on unfamiliar domain (cold-start Exp A).

| Experiment | Agent setup | Scoring |
|-----------|------------|---------|
| **A — Smart mode** | `detect_pattern(query)` primary | Max 39 |
| **B — Manual mode** | 40+ granular tools, no detect_pattern | Max 30 |

Smart mode (Exp A) also tests `detect_pattern` routing vs manual tool selection.

Details: [`benchmark/nyc-taxi/README.md`](nyc-taxi/README.md)

### IBM AML — Detection pipeline benchmark

The only experiment with labeled ground truth and published baselines — a true benchmark.
Three-layer AML detection pipeline. Layer 1: unsupervised passive scan. Layer 2: XGBoost (39 features).
Layer 3: typology filter (14 rules). Validated on HI-Small (calibration) and LI-Small (cross-validation,
HI-Small model applied without retraining).

**Layer 1 — HI-Small (unsupervised)**

| Stage | Recall (known) | Recall (full) | FP/TP (known) |
|-------|:--------------:|:-------------:|:-------------:|
| Surveillance (OR, 9 sources) | 80.4% | 64.4% | 74.4 |
| Confirmed (score≥2) | 59.6% | 38.6% | 29.0 |
| Investigation (boost+score≥3) | 50.8% | 31.3% | 21.2 |
| Critical (score≥3+density) | 30.2% | 16.8% | 14.2 |

Known = 3,170 accounts in structured laundering patterns (8 types). Full = 6,357 accounts
with is_laundering=1 (includes unstructured activity with no geometric signature).

**Layer 2+3 — HI-Small (supervised, best per stream)**

| Stream | Suspects | Recall (known) | Recall (full) | FP/TP (known) |
|--------|:--------:|:--------------:|:-------------:|:-------------:|
| A: Regulatory (p>=0.20) + filter | 10,813 | 52.0% | 27.8% | 5.6 |
| **B: Daily-ops (p>=0.20) + filter** | **2,025** | **40.2%** | **20.3%** | **0.6** |
| C: Escalation (p>=0.50) + filter | 709 | 19.7% | 9.8% | 0.1 |

**Cross-validation (HI-Small → LI-Small, no retrain)**

| Metric | HI-Small | LI-Small |
|--------|:--------:|:--------:|
| Layer 1 recall (known) | 80.4% | 80.2% |
| Stream B suspects | 2,025 | 2,325 |
| Stream B + filter FP/TP | 0.6 | 5.0 |
| Stream B + filter F1 (full GT) | 31% | 10% |

Details: [`benchmark/ibm-aml/README.md`](ibm-aml/README.md)

## Important note

All experiments measure **fully autonomous** agent exploration — no human guidance
during runs. In real-world usage, a human analyst directs the agent, which is expected
to improve results (not measured in this suite). Autonomous performance is the floor,
not the ceiling.

## Status

| Experiment | Type | Status | Best Score | Recall/Findings | Runs |
|-----------|------|--------|:----------:|:------:|:----:|
| Berka A — Cold-start | Skill calibration | **done** | 23/24 (96%) | 30–85.5% | 6 |
| Berka B — Skill-augmented | Skill calibration | **done** | 32/33 (97%) | 85.5–90.8% | 6 |
| NYC Taxi A — Smart mode | Domain generalization | **done** | 31/39 (79%) | 10–16 findings | 3 |
| NYC Taxi B — Manual + skills | Domain generalization | **done** | 28/30 (93%) | 10–12 findings | 3 |
| NYC Taxi C — Production reference | Domain generalization | **done** | unscored | 21 findings | 3 |
| AML HI-Small — Full pipeline | **Benchmark** | **done** | L1: 80.4% recall, L3 Stream B: FP/TP=0.6 | 3-layer | 1 |
| AML LI-Small — Cross-validation | **Benchmark** | **done** | L1: 80.2% recall, L3 Stream B: FP/TP=5.0 | no retrain | 1 |

**Detailed summaries:**
- Berka 0.1.0 — 6 runs, skill evolution, ML comparison: [`berka/results/0.1.0/SUMMARY.md`](berka/results/0.1.0/SUMMARY.md)
- NYC Taxi 0.1.0 — 3 runs, smart mode evolution, Exp C comparison: [`nyc-taxi/results/0.1.0/SUMMARY.md`](nyc-taxi/results/0.1.0/SUMMARY.md)
- IBM AML — 3-layer pipeline, LI-Small cross-validation, calibration history: [`ibm-aml/README.md`](ibm-aml/README.md)

### Findings

1. **Unsupervised recall on labeled datasets.** Berka: 85.5% default recall (best run; range 30–85.5% across 6 cold-start runs — 30% from one run where agent focused on wrong pattern; 5 of 6 runs achieved 65–85.5%). AML: 80.4% fraud recall on known patterns (64.4% on full GT). Both with zero labeled training data. AML Layer 1 recall generalizes across datasets (80.4% HI-Small → 80.2% LI-Small).

2. **Domain generalization (N=3, preliminary).** Berka and AML are financial datasets. NYC taxi (transportation) surfaces ghost trips, fare fraud, speed artifacts, and zone structural anomalies via the same navigation primitives. Suggests the geometric approach is not finance-specific, though more domains are needed to confirm generality.

3. **Smart mode quality depends on server-side planning.** NYC taxi r1→r2: +5 points with zero instruction change — purely from server upgrades (follow-up hints, profiling alerts, auto-temporal). The agent's prompt is a constant; the planner is the variable.

4. **Smart mode and manual mode find different things.** Smart mode specializes in data integrity anomalies (billing errors, GPS artifacts, impossible physics). Manual + skills specializes in operational patterns (holidays, rate code semantics, payment behavior). Neither is complete alone — production setup (Exp C) covers both.

5. **Skills consistently add 10–15 rubric points (N=2 datasets).** NYC taxi A (no skills): 79% → B (skills): 93% (+14 pts). Berka r1 A (no skills): 83% → B (skills): 97% (+14 pts). These are rubric scores measuring exploration quality (explanation depth, tool coverage, temporal awareness), not detection recall directly. Skills encode investigation methodology — temporal event detection, exhaustive tool enumeration, cross-pattern validation — that neither smart routing nor cold-start agents discover on their own.

6. **Geometry features boost supervised ML (AML).** XGBoost trained on geometry suspects with geometry-derived features achieves FP/TP=0.6 at 40% recall — ~80x improvement over geometry alone (FP/TP=74). Standalone XGBoost without geometry features reaches FP/TP=11.9. However: Layer 2/3 precision does not generalize without recalibration — Stream B FP/TP degrades from 0.6 (HI-Small) to 5.0 (LI-Small) when applying HI-Small thresholds to a dataset with 1.6x lower illicit ratio. Only Layer 1 recall generalizes.
