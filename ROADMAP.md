# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback. 

## Current: 0.2.0

**Graph meets geometry.** The main theme is the unified **edge table** — a Lance-based adjacency index per event pattern with BTREE lookups, auto-emitted at build time, giving runtime graph traversal at O(log n) scale.

- **Edge table** — one Lance dataset per event pattern, BTREE indexes on `from_key`/`to_key`, MVCC session pinning, YAML `edge_table` config (auto-detected from `graph_features`/relations), `--no-edges` CLI flag
- **11 new navigation functions** — `find_geometric_path` (beam search with geometric/anomaly/shortest/amount scoring), `discover_chains` (runtime temporal BFS, no build-time extraction), `entity_flow` (net flow per counterparty), `contagion_score` / `contagion_score_batch` (anomaly neighborhood scoring), `degree_velocity`, `investigation_coverage`, `propagate_influence` (BFS with geometric decay), `cluster_bridges` (geometry+graph fusion), `anomalous_edges` (event-level scoring)
- **`find_counterparties` fast path** — edge table BTREE lookup with amount aggregates when `pattern_id` given
- **PassiveScanner graph source** — `"graph"` source type for contagion scoring, `add_graph_source()`, `auto_discover()` auto-detects edge tables
- **11 new MCP tools** — edge table tools available in Phase 2 (immediately after `open_sphere`), no `sphere_overview` needed. Total: 66 MCP tools (55 → 66)
- **7 fraud investigation recipes** added to `gds-fraud-investigator` skill — Mirror Transaction, Pass-Through, Burst Detection, Weighted Reciprocity, Financial Profile, Concentration Risk, Benford's Law
- **All 8 skills updated** — edge table tools integrated across `gds-analyst`, `gds-detective`, `gds-explorer`, `gds-fraud-investigator`, `gds-investigator`, `gds-monitor`, `gds-scanner`, `gds-sphere-designer`
- **Builder improvements** — adjacency deduplication, self-loop filtering, edge stats cached at build time (`_gds_meta/edge_stats/`), timestamp string parsing with 6-format sample-based detection, Windows timezone database fallback
- **Performance** — `passive_scan` 84s → <10s, `detect_pattern` 86s → <15s, `cluster_bridges` 31s → <5s
- **Fixes** — `find_anomalies` Lance duplicate row deduplication, zero-variance `tracked_properties` cleanup in AML benchmark spheres

---

## 0.1.0

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
