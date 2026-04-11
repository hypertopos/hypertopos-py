# Changelog

All notable changes to hypertopos will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-04-11

### Added

**Temporal as-of reconstruction across edge-table graph primitives**
- `contagion_score()`, `contagion_score_batch()` — optional keyword-only `timestamp_cutoff: float | None` parameter. When set, only edges with `timestamp <= timestamp_cutoff` are considered. Batch forwards the cutoff to every per-entity call.
- `entity_flow()` — same `timestamp_cutoff` parameter; both outgoing and incoming edge reads honor the cutoff, so net flow reflects the as-of graph state.
- `degree_velocity()` — same parameter; buckets derive from the filtered edge set so the last bucket endpoint is naturally `<= timestamp_cutoff`.
- `propagate_influence()` — same parameter; threaded through the BFS `_expand_neighbors` closure so expansion follows only edges with `timestamp <= cutoff`.
- `find_counterparties()` — same parameter on the edge-table fast path via `_find_counterparties_via_edges`. The points-scan fallback has no timestamp column and **raises `GDSNavigationError`** when `timestamp_cutoff` is supplied — fail loudly instead of silently returning unfiltered results.

The semantic mirrors the existing `WitnessCohortConfig.timestamp_cutoff` from 0.2.1: edges with `timestamp <= timestamp_cutoff` are included. Enables agents to reconstruct contagion, flow, connection velocity, and influence propagation state at a prior point in time — useful for incident forensics ("what did this neighborhood look like on the day the alert fired?"), retroactive detection validation, and historical-snapshot comparisons.

### Fixed

- `detect_cross_pattern_discrepancy` no longer triggers full edge-table scans through `PassiveScanner.auto_discover`. The scanner gains an `include_graph: bool = True` keyword-only parameter; `detect_cross_pattern_discrepancy` calls it with `include_graph=False` because graph contagion plays no role in the downstream geometry-disagreement check. Eliminates the dominant latency regression on multi-pattern spheres with edge tables — the discrepancy detector previously paid one full edge-table read per event pattern with no signal benefit. Every other `auto_discover` caller (`composite_risk`, explicit scanner use) stays at the graph-enabled default.

## [0.2.1] — 2026-04-11

### Added

**Witness cohort discovery**
- `find_witness_cohort()` — rank entities that share the target's witness signature. **Investigative peer ranking, not edge forecasting.**
- Combines four signals: `exp(-distance/theta)` delta similarity (absolute, pool-independent), witness Jaccard overlap, trajectory cosine alignment (optional), graded anomaly bonus from `delta_rank_pct / 100`
- Excludes already-connected entities via BTREE edge lookup; bidirectional check by default; `timestamp_cutoff` for as-of evaluation in temporal hold-out
- Auto-resolves the event pattern with edge table covering an anchor's entity line; explicit `edge_pattern_id` override is validated
- Trajectory branch decided once per call (not per candidate) — when ref has trajectory, missing candidates get neutral 0.5 instead of mixed renormalization
- Batch trajectory load via single Lance scan instead of per-candidate (~11× speedup over the per-candidate path)
- `WitnessCohortConfig` and `WitnessCohortWeights` dataclasses group all tunable parameters; navigator API takes a single `config=` keyword
- `CohortMember` and `WitnessCohortResult` frozen dataclasses with per-component scores, exclusion counts, and reproducibility metadata
- `GDSEngine.witness_jaccard()`, `GDSEngine.trajectory_cosine()`, `GDSEngine.composite_link_score()` — pure scoring helpers exposed for reuse

## [0.2.0] — 2026-04-10

### Added

**Edge Table**
- Lance-based edge table per event pattern — auto-emitted at build time from FK data
- BTREE indexes on `from_key`/`to_key` — O(log n) lookups at any scale
- `GDSWriter.write_edges()`, `append_edges()`, `create_edge_indexes()`
- `GDSReader.read_edges()`, `has_edge_table()`, `edge_table_stats()`
- MVCC session pinning for edge tables
- YAML `edge_table` config (optional, auto-detected from graph_features/relations)
- `--no-edges` CLI flag

**Navigation**
- `find_geometric_path()` — beam search with geometric/anomaly/shortest/amount scoring
- `discover_chains()` — runtime temporal BFS on edge table (no build-time extraction needed)
- `find_counterparties()` — edge table fast path with BTREE lookup and amount aggregates
- `entity_flow()` — net flow per counterparty via edge table
- `contagion_score()` / `contagion_score_batch()` — anomaly neighborhood scoring via edge table
- `degree_velocity()` — temporal connection velocity (degree change over time buckets)
- `investigation_coverage()` — agent guidance: explored vs unexplored counterparty coverage
- `propagate_influence()` — BFS influence propagation with geometric decay and tx_count weighting
- `cluster_bridges()` — geometry+graph fusion: find entities bridging geometric clusters
- `anomalous_edges()` — event-level scoring between entity pairs (uses event geometry, not anchor)
- Amount-weighted scoring mode for `find_geometric_path` — `scoring="amount"`
- Lazy adjacency expansion — never loads full edge table into memory
- Anchor pattern resolution for geometric scoring (event pattern edge table → anchor pattern deltas)
- Score interpretation hint in `find_geometric_path` summary

**PassiveScanner**
- `"graph"` source type — contagion scoring via edge table + geometry anomaly check
- `add_graph_source()` — register graph contagion source with configurable threshold
- `auto_discover()` — auto-detects graph sources for event patterns with edge tables

**MCP Tools**
- `find_geometric_path` — path finding with geometric coherence scoring (+ amount mode)
- `discover_chains` — runtime chain discovery without pre-built chain lines
- `edge_stats` — edge table statistics (row count, degree, timestamp/amount range)
- `entity_flow` — net flow analysis per counterparty
- `contagion_score` / `contagion_score_batch` — anomaly neighborhood scoring
- `degree_velocity` — temporal connection velocity
- `investigation_coverage` — agent guidance for investigation coverage
- `propagate_influence` — BFS influence propagation with geometric decay and tx_count weighting
- `cluster_bridges` — geometry+graph fusion cluster bridge analysis
- `anomalous_edges` — event-level edge scoring between entity pairs
- Output cap (top 20 paths / top 100 influenced) with warning when truncated

**Builder**
- Edge table emission in all build paths (standard, streaming, chunked)
- Adjacency deduplication — one entry per unique neighbor
- Self-loop filtering in graph traversal and temporal chain BFS
- Edge stats cached at build time (`_gds_meta/edge_stats/`) for instant reads
- Timestamp string parsing with sample-based format detection (6 formats supported)
- Windows timezone database fallback in timestamp parsing
- Edge table auto-detect infers `timestamp_col` and `amount_col` from common column names (`timestamp`/`ts`/`event_time`/`created_at`/`tx_date`/`date` and `amount_received`/`amount`/`amount_paid`/`value`/`total`/`amt`) when not explicitly configured
- `sphere.json` edge_table metadata persists full config (`from_col`, `to_col`, plus `timestamp_col`/`amount_col` when set)

---

## [0.1.0] — 2026-04-07

First public release. Core GDS stack.

### Added

**Sphere Builder**
- Declarative YAML config (`sphere.yaml`) with CLI: `hypertopos build`, `validate`, `info`
- Three source tiers: single file (CSV/Parquet), multi-file join, Python script
- Derived dimensions (count, sum, avg, windowed metrics, IET)
- Precomputed dimensions with `edge_max` continuous mode
- Graph features: `in_degree`, `out_degree`, `reciprocity`, `counterpart_overlap`
- Composite lines (multi-key entities)
- Chain lines (temporal BFS extraction with parallel processing)
- Aliases with cutting-plane sub-populations
- Temporal snapshot builder
- `dimension_weights: kurtosis` automatic weighting
- Incremental update (`GDSBuilder.incremental_update()`)

**Navigation (π1–π12)**
- π1 `walk_line` — step along a line
- π2 `jump_polygon` — cross to related line via edge
- π3 `dive_solid` — enter temporal history
- π4 `emerge` — return to surface
- π5 `attract_anomaly` — find outliers in population
- π6 `attract_boundary` — find entities near alias boundary
- π7 `attract_hub` — find most connected entities
- π8 `attract_cluster` — discover geometric archetypes (k-means++)
- π9 `attract_drift` — find entities with highest temporal drift
- π10 `attract_trajectory` — find entities with similar trajectory (ANN)
- π11 `attract_population_compare` — compare geometry across time windows
- π12 `attract_regime_change` — detect structural shifts in population

**Analysis & Investigation**
- `explain_anomaly` — structured explanation with witness set, repair set, severity, reputation
- `contrast_populations` — dimension-by-dimension comparison (Cohen's d)
- `find_similar_entities` — ANN search in delta-space
- `centroid_map` — group centroids for sub-population positioning
- `composite_risk` — cross-pattern risk scoring (Fisher's method)
- `cross_pattern_profile` — multi-pattern risk view for one entity
- Full-text search (`search_entities_fts`) and hybrid search (semantic + FTS with RRF)
- 10 detection recipes: cross-pattern discrepancy, neighbor contamination, trajectory anomaly, segment shift, event rate anomaly, hub concentration, subgroup inflation, collective drift, temporal burst, data quality

**Forecasting**
- Trajectory extrapolation (exponentially-weighted linear regression)
- `forecast_anomaly_status` — predict future anomaly state
- `forecast_segment_crossing` — predict boundary crossings
- Pluggable `ForecastProvider` protocol for external backends

**Model**
- Point, Edge, Polygon, Solid, SolidSlice
- Line, Pattern, Alias, Manifest, Contract
- CalibrationTracker (online Welford drift detection)
- MVCC sessions (version-pinned reads per agent)

**Engine**
- Delta vector computation with z-score normalization
- Anomaly detection: theta threshold + conformal p-values
- Mahalanobis variant (ellipsoidal boundary via Cholesky decomposition)
- K-means++ clustering with automatic k selection (silhouette)
- DTW trajectory comparison
- Reputation scoring (Beta distribution posterior)
- Investigation engine (witness set, anti-witness, severity classification)
- Composition: Fisher's method for p-value combination, co-dispersion (Spearman)

**Storage**
- Arrow IPC format with Lance vector index (IVF-PQ)
- BTREE, BITMAP, FTS indices on points
- Append-only writes, LRU polygon cache
- Geometry stats cache, temporal centroid cache, trajectory ANN index
- Optional DataFusion SQL aggregation (~30x speedup on 5M+ events)

**PassiveScanner**
- Multi-source batch screening: geometry, borderline, points, compound sources
- `auto_discover` — automatic source registration from sphere structure
- Density boost, weighted scoring mode
- 4 operating stages

**Validation**
- Berka banking benchmark (skill calibration, 6 runs)
- NYC Yellow Taxi benchmark (domain generalization, 3 runs)
- IBM AML benchmark (3-layer pipeline with cross-validation)

**Documentation**
- Quick Start guide
- Core Concepts with mathematical foundation
- API Reference with navigation primitive families
- Configuration YAML reference with aliases
- Physical data format reference
- Architecture overview
