# Changelog

All notable changes to hypertopos will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
