# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback. 

## Current: 0.2.0

**Graph meets geometry.** Lance-based **edge table** per event pattern with BTREE indexes, auto-emitted at build time, gives runtime graph traversal at O(log n) scale. 11 new navigator functions (paths, chains, flow, contagion, influence, bridges, anomalous edges, …) and matching MCP tools available in Phase 2. PassiveScanner gains a `"graph"` contagion source. 7 fraud-investigation recipes added to `gds-fraud-investigator`, all 8 skills refreshed for the new tools. Major perf wins (`passive_scan` 84 s → <10 s, `detect_pattern` 86 s → <15 s, `cluster_bridges` 31 s → <5 s). 66 MCP tools total (55 → 66).

---

## 0.1.0

**First public release.** Complete GDS stack: declarative YAML sphere builder with CLI, navigation primitives π1–π12 (walk/jump/dive/attract/drift/trajectory/regime), aliases with cutting planes, temporal solids, investigation tools (`explain_anomaly`, witness/anti-witness, reputation), forecasting, FTS + hybrid search, online calibration tracker, MVCC sessions, multi-source PassiveScanner, incremental builder updates, MCP server (55 tools, smart detection, 3-phase visibility), Lance/Arrow storage, validation suite (Berka, NYC Taxi, IBM AML), full docs.

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
