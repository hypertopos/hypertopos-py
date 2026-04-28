# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback.

## Plan: 0.6.0

Theme: **architecture-emergent analytics, properly enabled.**

- Multi-epoch calibration retention in the builder (versioned μ/σ/θ per pattern) — prerequisite infrastructure that gates the three patent-aligned analytics below.
- Intrinsic/extrinsic decomposition — split an entity's temporal position change into entity-caused vs population-caused components.
- Entity influence on coordinate system — leave-one-out influence score; hidden-influencer detection.
- Cross-pattern temporal lead-lag — cross-correlation between per-entity displacement series across independently calibrated patterns.
- Edge-derived dimensions on event patterns — five build-time edge dimensions empirically validated for AML recall.
- Joint density gap detection — distribution-free "anomaly by absence" via probability integral transform + independence null.

Any item may slip to 0.7.0 if scope pressure requires. The multi-epoch retention infrastructure is non-negotiable for the theme.

---

## 0.5.0

Detection-quality and investigation-workflow push.

- **0.5.0** — Storey adaptive π₀ FDR + chi² p-values (power recovery in moderate-super-anomaly regimes without chasing the Storey estimator off a rank-based null). Drift direction on `π9_attract_drift` — `gradient_alignment` and `drift_direction ∈ {normalizing, deteriorating, neutral}` on every entity. Multi-hop root-cause tracing (`trace_root_cause` DAG) replacing `explain_anomaly_chain`. Geometric edge potential — per-edge anomaly score via `||δ_from − δ_to|| × (1/pair_tx_count)` sourced from the shared `AdjacencyIndex.pair_counts()` cache. Structural motif scoring — `score_motif` and `find_high_potential_motifs` over four closed-vocabulary atoms (`fan_out`, `cycle_2`, `cycle_3`, `structuring`) composing `edge_potential` via product, with 0.5.0 AdjacencyIndex-reuse refactor consolidating motif ranking / edge_potential / eight graph primitives onto one shared adjacency build per session. The "architecture-emergent analytics" side of the original plan was re-scoped to 0.6.0 because the prerequisite infrastructure (multi-epoch calibration retention in the builder) was larger than the advertised features.
- **0.5.1** — Two new closed-vocabulary motifs extending the 0.5.0 catalog: `fan_in` (mirror of `fan_out`, sink-centric, covers T12 destination-side parallel layering and T13 concentrator/sink) and `chain_k` (open directed chain of parametric length 3 ≤ k ≤ 8, no cycle closure, no node revisit, covers T5 multi-stage layering and T18 multi-jurisdiction latency). Reuses the 0.5.0 `MotifSpec` / `_score_motif_from_edges` / AdjacencyIndex machinery without touching the format. `score_motif` and `find_high_potential_motifs` grow a `k` parameter (default 4, validated 3–8 for `chain_k`, ignored otherwise); `min_k` now applies to `fan_in` in addition to `fan_out`. `split_recombine` and `bipartite_burst` remain under consideration for 0.5.2.
- **0.5.2** — Closes the structural motif catalog with two bipartite atoms — `split_recombine` (diamond scatter-gather S → k intermediaries → D, stacked-bipartite temporal order, `direction ∈ {"forward","backward"}` for source vs sink seed anchoring) and `bipartite_burst` (complete K_{k,m} bipartite subgraph in a tight window, asymmetric `min_k` / `min_m` cardinality controls) — covering T1 / T12 / T13 / T16 atoms unreachable by the 0.5.0 / 0.5.1 vocabulary. Registry grows from six to eight types. Same release ships a perf-hardening pass on the existing catalog: single-seed enumeration unified onto the in-memory adjacency path, `_score_motif_from_edges` batched, `cycle_3` two-step pre-filter on hub seeds, `chain_k` per-k adaptive frontier cap, `bipartite_burst` size-ascending K-set intersection, and `_enumerate_structuring` consistency fix. No format change.

---

## 0.4.0

Distribution-aware geometry, calibration fidelity, and graph acceleration.

- **0.4.0** — Distribution-aware Bregman divergence with per-dimension kind tags (gaussian/poisson/bernoulli). Per-dimension anomaly threshold (hyper-ellipsoid). Theta calibration fix. Anomaly confidence via stratified bootstrap, `min_confidence` filter on `find_anomalies`. Interpretive MCP output (`explain_anomaly` returns additive per-dimension Bregman contributions). Sphere format 2.3. Per-pattern pipeline parallelism (geometry → temporal overlap, up to 4 threads). In-memory adjacency index — full graph in a dict with O(1) neighbour lookups replacing Lance BTREE scans across all graph operations. Temporal bisect in chain extraction. Stress-test fixes: `find_anomalies` edges and pagination, `hub_score_history` clamp, `detect_pattern` type guard, aggregate ambiguity guard, `dim_labels` in `find_clusters`.
- **0.4.1** — Native graph algorithms as build-time geometry dimensions (igraph C backend): PageRank, betweenness centrality (edge-sampled Brandes), community detection (label propagation), clustering coefficient, connected components — all reading the in-memory adjacency index from 0.4.0 at zero additional build cost. Bidirectional BFS rewrite of `find_geometric_path` for reliable path discovery in sparse graphs (replaces beam search). `hypertopos.__version__` fix.

---

## 0.3.0

Lance perf upgrade, FDR control, builder intelligence.

- **0.3.0** — Aggregate engine rewritten around Lance SQL, precomputed contagion stats, format 2.2, edge table auto-detect fix.
- **0.3.1** — Benjamini-Hochberg FDR control, submodular facility location, vectorized build with adaptive memory chunking.
- **0.3.2** — NumPy graph features, chunked pre-computation, Lance compact tuning, per-dim index removal. Generalized dimension blocks (g/t/s), geometric heredity (`find_novel_entities`).
- **0.3.3** — Agent navigation policy (investigation memory, failure guards, decision scoring). Dimension-selective similarity (`dim_mask`, `metric="cosine"`, `metric="Linf"`). Sphere-specific hardcoding removed from 6 skill guidance texts.

---

## 0.2.0

Graph meets geometry — edge table, runtime traversal, contagion/influence.

- **0.2.0** — Edge table, +11 navigator functions, contagion/influence primitives.
- **0.2.1** — Witness cohort discovery, investigative peer ranking.
- **0.2.2** — As-of graph reconstruction, `detect_cross_pattern_discrepancy` latency fix.

---

## 0.1.0

First public release — full GDS stack, π1–π12, builder, MCP server, validation suite.

---

## Future

**Detection quality**
- Declarative motif API with per-hop predicates (amount, time-delta, edge-dimension predicates).
- Robust estimators, multi-scale resolution — trimmed means / MAD-based σ, hierarchical build at daily/weekly/full granularity.
- Chains as first-class geometric entities — external chains declared as anchor lines with a membership table, plus chain-aware geometric primitives (anomaly propagation inside a chain, coherent-anomaly filter, chain drift trajectory).

**Builder evolution**
- Incremental rebuild — geometry-only without `--force` wipe.

**PassiveScanner evolution**
- Native temporal source support for direct temporal inputs, without requiring manual dataset plumbing in benchmark scripts.
- Optional weighted scoring mode that uses continuous intensity instead of binary counts.
- **SphereProfiler** — autonomous sphere scanner that profiles all patterns, runs calibration sweeps across source combinations, proposes optimal PassiveScanner composition for Layer 1 surveillance.

**Code refactoring**
- Break up oversized modules into smaller, domain-focused components (post-0.5.0 `navigator.py` is over 11k lines and is the obvious first target; per-pattern parallelism and AdjacencyIndex reuse in 0.4.0/0.5.0 reduced some coupling but the module itself did not shrink).
- Consolidate repeated orchestration logic into shared helpers.

**Cross-sphere capabilities**
- Cross-sphere comparison — dimensionless metrics across independently calibrated coordinate spaces.
- What-if analysis — hypothetical edge changes producing modified coordinate vectors.

**Enterprise / governance**
- Dimension access control — per-agent visibility constraints on delta dimensions.

**Tooling**
- Runtime latency benchmarks in package docs.
