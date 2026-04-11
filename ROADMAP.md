# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback. 

## Current: 0.2.2

As-of graph reconstruction across edge-table graph primitives + latency fix on `detect_cross_pattern_discrepancy`. See CHANGELOG.

---

## 0.2.1

Witness cohort discovery — investigative peer ranking. See CHANGELOG.

---

## 0.2.0

Graph meets geometry — edge table, runtime traversal, contagion/influence, +11 navigator functions. See CHANGELOG.

---

## 0.1.0

First public release — full GDS stack, π1–π12, builder, MCP server, validation suite. See CHANGELOG.

---

## Plan: 0.3.0

**Storage / performance:**
- Upgrade `pylance 2.0.1 → 3.0.1` — format 2.2, ~300% faster scans, LZ4/zstd compression
- Replace `shutil.copytree` with `lance.dataset.clone()` in test fixtures (O(1) metadata clone)
- Builder: incremental rebuild (geometry-only without `--force` wipe)

**Code refactoring:**
- Break up oversized modules into smaller, domain-focused components
- Reduce coupling between core layers by replacing private cross-layer access with explicit interfaces where practical
- Tighten error handling in hot paths so failures are surfaced more consistently
- Consolidate repeated orchestration logic into shared helpers instead of duplicating it across modules
- Make the refactor pass incremental: preserve behavior, improve structure, then revisit deeper architectural boundaries

**PassiveScanner evolution:**
- Native temporal source support for direct temporal inputs, without requiring manual dataset plumbing in benchmark scripts
- Optional weighted scoring mode that uses continuous intensity instead of binary counts
- **SphereProfiler** — autonomous sphere scanner that profiles all patterns, runs calibration sweeps across source combinations, proposes optimal PassiveScanner composition for Layer 1 surveillance. Core loop: enumerate patterns → single-source anomaly scans → greedy multi-source combination → ranked composition report. No labeled GT needed; optional GT file unlocks supervised calibration.

**Anomaly detection quality:**
- Confidence scoring, robust estimators, multi-scale resolution — improve anomaly precision and reduce false positives on heavy-tail and multi-modal populations

---

## Future

- Cross-sphere comparison — dimensionless metrics across independently calibrated coordinate spaces
- What-if analysis — hypothetical edge changes producing modified coordinate vectors
- Dimension access control — per-agent visibility constraints on delta dimensions
- Runtime latency benchmarks in package docs
