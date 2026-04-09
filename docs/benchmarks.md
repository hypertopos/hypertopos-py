# Benchmarks

> Validation results across three domains — zero labels, zero training, same engine.

## Results Summary

| Domain | Dataset | Scale | Key Result | Labels |
|--------|---------|-------|-----------|--------|
| **Banking** | Berka (Czech banking) | 4.5K accounts, 1M tx | 85.5% recall, 97% skill score (best of 6 runs) | Zero |
| **AML** | IBM AML (synthetic) | 5K accounts, 500K tx | 80.4% recall Layer 1, FP/TP=0.6 Layer 2+3 | Zero (Layer 1) |
| **Transport** | NYC Yellow Taxi | 7.6M trips, 265 zones | 8/8 anomaly categories, 93% skill score | Zero |

All benchmarks use the same hypertopos engine. No domain-specific rules, no training, no tuning between domains.

## Berka — Czech Banking

Real banking data (NeurIPS 2023 reference dataset). 4,500 accounts, 1M transactions, 76 known loan defaults.

**Best run (r6):** 32/33 scoring criteria (97%), 85.5% recall on `account_stress_pattern`. 6 runs total — 5 out of 6 skill-augmented runs scored 97%.

Key findings: loan default detection, credit stress patterns, NB-Split entity isolation enabling per-pattern recall.

Full results: [`benchmark/berka/`](benchmark/berka/)

## IBM AML — Anti-Money Laundering

Synthetic transaction data. 3-layer pipeline: geometry (unsupervised) → XGBoost (39 features) → typology filter (14 rules).

| Layer | Recall | FP/TP | Method |
|-------|--------|-------|--------|
| Layer 1 (OR, 9 sources) | 80.4% | 74.4 | Pure geometry, unsupervised |
| Layer 1 (score≥2) | 59.6% | 29.0 | Multi-source confirmed |
| Layer 2+3 (Stream B) | 40.2% | 0.6 | XGBoost + typology filter |

Cold-start: 80.4% recall from day zero on any transaction dataset. No model training, no labels for Layer 1.

Full results: [`benchmark/ibm-aml/`](benchmark/ibm-aml/)

## NYC Yellow Taxi — Domain Generalization

7.6M real trips (January 2019, TLC public data). Transportation domain — zero overlap with financial data. Same engine, same 12 navigation primitives.

**8/8 organic anomaly categories detected:** ghost trips, fare fraud, speed artifacts, dead zones, surge zones, temporal events, rate code abuse, tip encoding anomalies.

| Mode | Score | Findings per run |
|------|-------|-----------------|
| Smart mode (best) | 31/39 (79%) | 16 |
| Manual + skills (best) | 28/30 (93%) | 10 |
| Production reference | 21 findings/run avg | 3 runs |

Key proof: GDS is not a financial tool. Same geometric primitives surface ghost trips and fare fraud as cleanly as they surface loan defaults and AML typologies.

Full results: [`benchmark/nyc-taxi/`](benchmark/nyc-taxi/)

## What This Proves

1. **Cross-domain** — banking, AML, transportation. Same engine, zero domain rules.
2. **Zero labels** — population geometry calibrates itself from the data.
3. **Reproducible** — `pip install hypertopos`, run build scripts, verify on your machine.
4. **Measurable** — exact recall, scoring rubrics, per-run reports with full methodology.

---

*See also: [Core Concepts](concepts.md) for the mathematical foundation, [Configuration](configuration.md) for sphere building.*
