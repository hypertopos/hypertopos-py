# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback. 

## Current: 0.1.0

Core GDS stack — sphere build, geometry, navigation, MCP server, storage/performance, and maintainability.

- Sphere builder — declarative YAML config, CLI (`build`, `validate`, `info`), Tier 1-3 sources, derived/precomputed/graph dimensions, composite lines, chain lines
- Navigation primitives π1–π12 — walk, jump, dive, emerge, attract (anomaly, boundary, hub, cluster), drift, trajectory, population compare, regime change
- Aliases — cutting-plane sub-populations with derived geometry (W vector, bias, direction)
- Temporal snapshots — deformation history, solid construction, temporal centroids
- Investigation — `explain_anomaly`, witness set, anti-witness, reputation scoring
- Forecasting — trajectory extrapolation, anomaly forecast, segment crossing prediction, pluggable `ForecastProvider` protocol
- Full-text search — FTS index on points, `search_entities_fts`, hybrid search (semantic + FTS with reciprocal rank fusion)
- CalibrationTracker — online Welford drift detection with soft/hard thresholds
- MVCC sessions — Manifest/Contract version pinning, isolated reads per agent
- PassiveScanner — multi-source batch screening (geometry, borderline, points, compound), auto-discover, density boost
- Incremental update — `GDSBuilder.incremental_update()` for appending without full rebuild
- MCP server (55 tools, smart detection mode, 3-phase tool visibility)
- Validation suite — Berka (skill calibration), NYC Taxi (domain generalization), IBM AML (3-layer benchmark)
- Storage — Lance vector index, Arrow IPC, LRU cache, append-only writes
- Docs — quickstart, concepts, architecture, configuration, API reference, data format

---

## Plan: 0.2.0

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

- On-demand chain discovery — replace expensive build-time BFS with per-entity runtime extraction
- Geometric path finding — multi-hop traversal scored by delta geometry
- Cross-sphere comparison — dimensionless metrics across independently calibrated coordinate spaces
- What-if analysis — hypothetical edge changes producing modified coordinate vectors
- Dimension access control — per-agent visibility constraints on delta dimensions
- Runtime latency benchmarks in package docs
