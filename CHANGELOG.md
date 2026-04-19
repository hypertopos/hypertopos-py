# Changelog

All notable changes to hypertopos will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-04-19

### Added
- Storey adaptive π₀ FDR: `benjamini_hochberg(..., method="storey")` and `fdr_method="storey"` on `π5_attract_anomaly`, `π6_attract_boundary`, `π7_attract_hub`, `π9_attract_drift`. The LSL π̂₀ estimator scales BH q-values, recovering discoveries vanilla BH leaves on the table when the population carries a meaningful null mass. Default `fdr_method` remains `"bh"`. Power recovery is regime-dependent — spheres with an over-compressed delta_norm distribution or an extreme super-anomaly tail see zero uplift because BH already separates every entity from the null.
- Parametric chi-squared p-values: `parametric_p_values_chi2(delta_norms, df)` computes upper-tail χ²(df) survival — the mathematically correct p-value under the `N(0, 1)` null. Navigator gains `p_value_method` parameter (`"rank"` default, `"chi2"` opt-in). `"chi2"` is what lets `fdr_method="storey"` actually shrink q-values; rank-based p-values are uniform by construction and defeat the Storey estimator.
- Drift direction: `π9_attract_drift` (and MCP `find_drifting_entities`) returns `gradient_alignment ∈ [-1, 1]` and `drift_direction ∈ {"normalizing", "deteriorating", "neutral"}` for every entity — the radially-inward component of the drift vector, distinguishing entities moving toward the population centre from ones moving away without extra tool calls.
- Multi-hop root-cause tracing: new `GDSNavigator.trace_root_cause(primary_key, pattern_id, max_depth=2, max_branches=3, …)` returns a bounded DAG of evidence around an anomalous entity — root witness dimensions, edge-counterparty branch (sorted by anomaly not transaction volume), neighbour-contamination branch with `anomalous_cp_keys` evidence, and hub membership branch. All nodes share one severity scale, candidate branches are priority-ordered by severity strength, and `truncated=true` flags when a real candidate was dropped. One call replaces the manual `explain_anomaly → find_counterparties → contagion_score → π7 hub` chain.
- Geometric edge potential: new `GDSNavigator.edge_potential(from_key, to_key, pattern_id)` and `attract_edge_potential(pattern_id, top_n, …)` primitives score transaction edges (not just node footprints) as `||δ_from − δ_to|| × (1/pair_tx_count)` — catches one-off transactions between geometrically divergent accounts, a classic AML layering signature that entity-level `delta_norm` misses. Pair-count histogram is sourced from the shared `AdjacencyIndex.pair_counts()` cache used by eight graph primitives, so the rarity prior is O(1) after any graph primitive has warmed adjacency in the session. Auto-enriches the `edge_counterparty` branch of `trace_root_cause`.
- Structural motif scoring: new `GDSNavigator.score_motif(entity_key, motif_type, pattern_id, time_window_hours=None, amt1_min=10000.0, amt2_max=10000.0)` and `find_high_potential_motifs(pattern_id, motif_type, top_n, …)` compose edge_potential across motif edges via product. Closed vocabulary of 4 atoms: `fan_out` (hub → k targets), `cycle_2` (A↔B round-trip), `cycle_3` (directed A→B→C→A triad with strict temporal ordering), and `structuring` (open A→B→C→D chain with amount-gated hops — hop1 ≥ `amt1_min`, hops 2-3 ≤ `amt2_max`, flash window). Amount thresholds default to the USD reporting threshold and are configurable per jurisdiction. Smart per-motif default time windows (168h / 24h / 72h / 1h). Motif ranking and edge_potential both enumerate off the shared `AdjacencyIndex` cache — adjacency build cost is paid once per pattern per session and reused by eight graph primitives. LRU cache caps 8 ranking entries per navigator instance with `structuring` thresholds part of the cache key. Covers structural atoms of 25 documented AML typologies.

### Removed
- `GDSNavigator.explain_anomaly_chain` — superseded by `trace_root_cause`. Previous linear same-similar walk is recoverable via `find_similar_entities(..., filter_expr="is_anomaly = true")`; the DAG tracer solves the intended root-cause use case.

## [0.4.1] — 2026-04-16

### Added
- Native graph algorithms as build-time geometry dimensions: `pagerank` (importance), `betweenness` (brokerage, edge-sampled Brandes), `community` (label-propagation membership), `clustering_coefficient` (local triangle density), `connected_component` (disconnected subgraph id). Computed via igraph C backend on the in-memory adjacency index introduced in 0.4.0 — no external services, no Python graph library beyond igraph bindings. Add to `graph_features.features` in sphere YAML and the metrics flow through `explain_anomaly`, `find_anomalies`, and every delta-based primitive like any other dimension.

### Changed
- `find_geometric_path` switched from beam search to bidirectional BFS. Guarantees a path is found when one exists within `max_depth` — beam search could miss valid paths in sparse or loosely connected graphs. `beam_width` now caps the number of top-scored paths returned after discovery, not search width.

### Fixed
- `hypertopos.__version__` reports the installed package version (was frozen at `0.3.0`).

## [0.4.0] — 2026-04-15

### Added
- Distribution-aware Bregman divergence with per-dimension kind tags (gaussian/poisson/bernoulli). Auto-detected from dimension type; YAML `kind:` override per dimension.
- Per-dimension anomaly threshold (hyper-ellipsoid boundary) replacing uniform decomposition (hyper-sphere).
- `anomaly_confidence` per entity via stratified bootstrap resampling. Configurable `bootstrap_iterations` (default 200, 0 to skip). Skipped for populations > 50K (use `conformal_p`), `group_by_property`, and `use_mahalanobis` patterns.
- `bregman_divergence` stored alongside `delta_norm` in geometry.
- `dimension_kinds` per pattern in sphere.json metadata.
- `find_anomalies` returns edges on polygons (both fast path and in-process path).
- `find_neighborhood` raises `GDSNavigationError` for continuous-mode patterns instead of silently returning empty results.
- In-memory adjacency index for graph operations — lazy build on first graph call, cached on session reader.
- Build pipeline per-pattern parallelism: geometry → temporal as concurrent pipeline (up to 4 threads).

### Changed
- `explain_anomaly` returns per-dimension Bregman contributions (additive, kind-labeled) instead of absolute z-score deltas when dimension_kinds available.
- Sphere format bumped to 2.3. Old spheres must be rebuilt.
- Chain extraction uses temporal bisect for neighbor filtering instead of linear timestamp scan.

### Fixed
- `find_anomalies` pagination: deterministic ordering on tied `delta_norm` values prevents entity duplication across page boundaries.
- `hub_score_history` clamps negative `hub_score`/`alive_edges_est` to zero for early temporal slices.
- `anomalous_edges` deduplicates by `event_key` for self-loops (`from_key == to_key`).
- `find_clusters`, `find_drifting_entities`, `compare_time_windows`, `find_regime_changes` use `pattern.dim_labels` for dimension names, including event dimensions.
- `find_geometric_path` default `beam_width` increased from 10 to 50; auto-scales for deep searches.
- `aggregate` raises explicit error when `group_by_line` matches multiple relations on the same pattern.
- `search_entities` JSON serialization handles datetime columns.
- `sphere_overview` mode detection uses `sample_size=200` instead of full geometry table read.
- Weighted Bregman norms for `metric="bregman"` in `find_anomalies`.

## [0.3.3] — 2026-04-13

### Added

- `dim_mask` parameter on `find_similar_entities` — compute distance only on named dimensions. Focuses similarity search on specific aspects of geometry (e.g., only the dimensions driving an anomaly).
- `metric` parameter on `find_similar_entities` — `"L2"` (default Euclidean) or `"cosine"` (shape similarity ignoring magnitude). Cosine compares anomaly profile direction, not severity.
- `metric` parameter on `find_anomalies` (π5) — `"L2"` (default, pre-computed) or `"Linf"` (max single-dimension |delta|, runtime scan). Linf catches single-dimension spikes that L2 dilutes.
- `push-mirrors.sh --from-ref` mode for incremental mirror pushes between tagged releases, with pre/post author verification guard.

## [0.3.2] — 2026-04-13

### Added

- Generalized dimension blocks (g/t/s). `geo_properties`, `metric_properties`, and `semantic_dim` optional fields on pattern config. Geographic and metric blocks use empirical mu/sigma normalization; semantic block applies SVD-based PCA dimensionality reduction. Dimension names are prefixed (`g:lat`, `t:balance`, `s:pc0`) for identification in `explain_anomaly` output. All optional — existing configs unaffected.
- `find_novel_entities(pattern_id)` navigator method. Ranks entities by geometric deviation from neighbor-expected position using edge table adjacency. High novelty_score = entity doesn't behave like its neighborhood. Requires a pattern with an edge table.
- `engine/heredity.py` module: `compute_expected_delta`, `compute_novelty_score`, `compute_novelty_decomposition` — pure NumPy scoring functions for geometric heredity.
- `builder/dim_blocks.py` module: `normalize_metric_block`, `normalize_geo_block`, `normalize_semantic_block` — normalization helpers for generalized dimension blocks.

### Changed

- Temporal build: `reciprocity` and `counterpart_overlap` graph features replaced per-window Arrow fallback with NumPy integer-encoded set operations. One-time entity encoding via `pc.index_in`, then per-bucket `np.unique` + `np.intersect1d` + `np.bincount`. Eliminates Arrow table construction and hash joins per temporal window.
- Temporal build: `in_degree`/`out_degree` use single-pass Arrow `group_by([key, bucket]).aggregate([count_distinct])` across all buckets. Eliminates O(n_buckets) per-window iterations.
- Temporal build: chunked path pre-computes derived-dim groupby tables and graph feature tensors once before the entity chunk loop. Eliminates redundant full event-table scans per chunk.
- Geometry build: per-dimension BTREE scalar indices (`delta_dim_0` through `delta_dim_N`) replaced with Lance zone-map filtering. Saves sequential index builds proportional to dimension count.
- Geometry build: streaming path keeps `is_anomaly` in memory instead of re-reading from Lance after finalization.
- Lance compaction `target_rows_per_fragment` raised from 1M to 4M for temporal, geometry, and geometry chunk finalization — fewer fragments to merge during compaction.

## [0.3.1] — 2026-04-12

### Added

- Benjamini-Hochberg FDR control on attract primitives. `fdr_alpha` parameter on `pi5_attract_anomaly`, `pi6_attract_boundary`, `pi7_attract_hub`, and `pi9_attract_drift` applies BH multiple-testing correction to empirical-null p-values derived from delta rank percentile. Entities with `q_value > alpha` are removed before the top-N step. Guarantees `E[FDR] <= alpha`. Each retained entity carries a `q_value` — the minimum alpha at which it would still be retained. Off by default; legacy behavior preserved when omitted.
- Diverse selection via submodular facility location. `select="diverse"` on the same four primitives replaces top-N ranking with a lazy-greedy maximisation of the facility location objective over pairwise cosine similarity of delta vectors. Achieves `(1 - 1/e)` approximation to the optimal K-subset. Each selected entity reports a `representativeness` count — how many population members are nearest to it among the selected set. Covers the geometric space of the result set instead of clustering around a single extreme region.
- `engine/fdr.py` module: `benjamini_hochberg`, `empirical_p_values_from_rank`, `q_values_from_p_values` — pure NumPy, no state, no I/O.
- `engine/selection.py` module: `lazy_greedy_facility_location`, `compute_similarity_matrix` — pure NumPy, Nemhauser-Wolsey-Fisher optimality guarantee.

### Changed

- Build pipeline: derived dimension scatter loops replaced with Arrow `pc.index_in` + numpy fancy indexing. Eliminates per-row Python iteration in both static (`compute_derived_batch`) and temporal (`_precompute_shape_tensor`) build paths.
- Build pipeline: contagion stats computation (`_build_contagion_stats`) rewritten from Python loops over edge table to Arrow group_by + join. Eliminates O(E+N) Python iterations.
- Build pipeline: rolling z-score (`_compute_max_rolling_z`) uses Welford online algorithm. Reduces time complexity from O(n_buckets² × N × D) to O(n_buckets × N × D) and memory from O(N × n_buckets × D) to O(N × D).
- Build pipeline: temporal Arrow assembly uses single numpy reshape instead of per-bucket table construction loop.
- Build pipeline: graph dims temporal computation uses pre-sorted event table with `searchsorted` for O(1) per-bucket slicing instead of O(E) scan per bucket.
- Build pipeline: `build_temporal` adaptively chunks entities when the shape tensor would exceed available RAM. Auto-detects memory via platform APIs, falls back to 4 GB budget. Single-pass path (zero overhead) when tensor fits.

### Fixed

- `empirical_p_values_from_rank` used a `1/N` floor where N was the input array length. When the input was a small `top_n` slice (e.g. 5 entities), the floor clipped all p-values to 0.2, causing BH to reject everything at `alpha=0.05`. Now uses a fixed `1e-10` floor — the rank percentile is already computed against the full population, so the floor only needs to prevent exact zero for numerical stability.
- `π7_attract_hub` and `π9_attract_drift` FDR computed p-values as `(N-i)/N` where N was the `top_n` result count (e.g. 10), making the best possible p-value `1/top_n` — mathematically unable to pass BH at any `alpha < 1/top_n`. Now uses population-level rank percentiles: hubs use the empirical CDF from the full scores array, drift uses the rank position relative to the total pre-truncation population.

## [0.3.0] — 2026-04-12

> **Theme:** Lance 3.x perf upgrade. Aggregate engine moves from a persistent subprocess worker plus an optional external DataFusion package to the Lance scanner's built-in DataFusion executor via `LanceDataset.sql(...)`. The build-time contagion stats table replaces the runtime edge-table replay in the graph contagion scanner. Format 2.2 is the new write default.
>
> **Mandatory rebuild for spheres built before this release.** See the Migration section below.

### Changed

- `pylance` minimum bumped to 4.x. New writes target Lance format 2.2 with structural-decode parallelism, BSS float compression, cascading codecs, and block-level RLE on top of the previous baseline. Older spheres written with format 2.0 / 2.1 remain readable transparently.
- The aggregate engine now pushes GROUP BY computation directly into the Lance scanner. The new `lance_sql_agg` module hosts `aggregate_count`, `aggregate_metric` (sum / avg / min / max), `aggregate_filtered_metric`, `aggregate_pivot`, `aggregate_property`, `aggregate_percentile`, and `find_anomalies`. Each helper streams the columns it needs from the geometry / event-line / property-line Lance datasets through `LanceDataset.sql(...)` and finishes the join + aggregate in pyarrow on the small post-projection result. The full geometry table is never loaded into Python memory and the persistent subprocess worker is no longer involved for any of these paths.
- Test fixture cloning uses `lance.LanceDataset.shallow_clone` for every Lance dataset directory inside a sphere tree, with a thin `clone_sphere` helper in `tests/conftest.py` that walks the directory and falls back to `shutil.copy2` for non-Lance files. Shallow clone copies only metadata and references — measurably faster on Windows where deep `copytree` over a sphere built from many small Lance fragment files is dominated by per-file NTFS overhead.

### Added

- `GDSBuilder` rejects patterns with zero declared dimensions (no relations, no event dimensions, no derived dimensions, no tracked properties) at validation time. A pattern with width-0 delta vectors had no meaningful geometry to compute even before this release; format 2.0 silently allowed the construct, format 2.1+ refuses to encode the resulting `fixed_size_list[0]` columns at all. The validation surfaces a typed error up front instead of letting the call descend into an opaque write-time panic.
- Builder precomputes per-entity contagion statistics for every pattern with an edge table and writes them to `_gds_meta/contagion_stats/{pattern_id}.lance` with a BTREE index on `primary_key`. The table holds `(primary_key, neighbor_count, anomalous_neighbor_count, contagion_ratio)` snapshotted at build time against the just-calibrated anchor geometry.

### Fixed

- `_scan_graph` (the graph contagion source feeding `passive_scan` and `composite_risk`) reads the precomputed `_gds_meta/contagion_stats/{pattern_id}.lance` table directly and threshold-filters inline. The legacy edge-table-replay path is gone; spheres built before 0.3.0 must be rebuilt to get graph contagion hits.
- `contagion_score` anchor pattern resolution now correctly resolves via sibling lines when the anchor pattern is not directly associated with the target entity's line. Previously crashed with `KeyError` on spheres where the anchor pattern entity line differed from the navigated line.
- Edge table auto-detect excludes metadata columns (`created_at`, `changed_at`, `version`) from the timestamp type-based fallback, preventing incorrect timestamp column selection on entity lines with versioning columns. Amount candidate list widened to cover `fare_amount`, `total_amount`, `amount_received`, `amount_paid`.

### Security

- `lance_sql_agg` validates user-controlled SQL inputs up front. Lance's `LanceDataset.sql(...)` does not expose parameter binding, so values and identifiers are inlined into the query string. Filter values (entity primary keys) are escaped with `_escape_sql_string`, which doubles single quotes and rejects backslash and ASCII control characters. Column-name identifiers (`metric_col`, `pivot_field`, `prop_name`) are validated against `^[A-Za-z_][A-Za-z0-9_]*$` via `_validate_sql_identifier` before being inlined into SQL strings — anything outside that pattern raises immediately. Both helpers fail loud rather than silently produce wrong results or pass injected SQL through to the parser.

### Removed

- `hypertopos.engine.subprocess_agg` (the persistent subprocess worker and its JSON command protocol) is deleted; every entry point it served is now routed through `lance_sql_agg`.
- `hypertopos.engine.datafusion_agg` (the in-memory external-DataFusion fast path for count / metric / pivot / gbp / percentile) is deleted along with the optional `datafusion` extra in `pyproject.toml`. Lance 4.x's built-in DataFusion executor covers every query the external package was carrying.
- The matching `tests/test_subprocess_agg.py` and `tests/test_datafusion_agg.py` are deleted. Aggregate semantics are exercised through `tests/test_aggregation_engine.py` and the MCP tool tests.

### Migration

- **Rebuild required.** Spheres built before 0.3.0 are still openable but degrade gracefully on graph contagion — the precomputed `contagion_stats` table is created at build time only, and the runtime scanner returns zero hits without it. Rebuild with `hypertopos build --config sphere.yaml --force`. There are no in-place upgrade paths.
- The optional `datafusion` package is no longer a dependency of any aggregate path. If your environment installed it via `pip install hypertopos[datafusion]`, that extra now resolves to nothing — you can drop it from your install command.

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
