# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback.

## 0.6.0

Architecture-emergent analytics, properly enabled — calibration is now versioned, and patent-aligned analytics ride on that versioning.

- **0.6.0** — multi-epoch calibration retention, `compare_calibrations`, intrinsic / extrinsic drift decomposition, hidden-influencer matrix (`find_calibration_influencers`), cross-pattern temporal lead-lag (`find_lead_lag`), edge-derived event-pattern dimensions, joint density gap detection (`find_density_gaps`), declarative motif API (`find_motif_by_hops`, bounded MVP). Sphere format 2.4.
- **0.6.1** — anchor-pattern aggregation of edge-derived dimensions (`edge_dim_aggregations:`); BFS-by-level enumerator on `find_motif_by_hops` with hop count cap raised 6→8 and global `time_window_hours`; declarative motif API completion via `HopPredicate.amount_ratio_to_prev` and `require_anomalous_entity`; functional anchor-companion + event-aware scoring on `find_motif_by_hops(score=True)`; broadcast-safe `Pattern.dim_labels` on patterns with aggregated edge-dim names. Additive only — no sphere format bump, no new tools.
- **0.6.2** — chain-anchor `edge_dim_aggregations:` regime closes the third `anchor_kind` after single (account-style) and pair (k=2 composite); vectorized `AdjacencyIndex` load path (pyarrow `group_by` + lazy per-key materialization) for cold-start latency cleanup; column-selective `neighbors_*_window` accessors with motif enumerator migration for `bipartite_burst`, `fan_out`, `fan_in`, `cycle_2`; new project skill `profile-before-plan` gating hot-path refactors on real-data cProfile. Additive only — sphere format stays at 2.4 (the originally proposed persisted-artifact path was empirically refuted; vectorized loader ships without any on-disk change).
- **0.6.3** — `edge_dim_aggregations:` expansion along four axes: three additional canonical aggregates per source dim (`_std`, `_p95`, `_count_above_threshold` with population p95 cutoff); k>2 composite anchor support (tripartite and beyond) via positional `key_cols` to edge endpoints; per-source-dim subset selector (`dims:` accepts list form for back-compat all-five sugar or mapping form `{dim: [agg, ...]}` for explicit per-dim subset); `_count_above_threshold` cutoffs persisted into the calibration epoch JSON with cross-epoch drift surfacing through `compare_calibrations.edge_dim_threshold_drift`. Spheres declaring `edge_dim_aggregations:` must rebuild; sphere format stays at 2.4.
- **0.6.4** — chain-coherent investigative loop: `find_chains_with_coherent_anomaly` flags chains where consecutive entity-anchor positions are individually anomalous on the same dominant delta dim (chain composition vs chain shape — orthogonal to `find_anomalies` on the chain pattern); `anomaly_propagation_in_chain` returns the per-chain hop-by-hop anomaly trace; `classify_chain_typology` labels each chain along five operational axes (shape, peak_position, position_in_chain, extension_signals, dominant_top_dim); `extend_chain` suggests boundary extension entities via the chain reverse index. Plus `find_motif_by_hops.anomaly_seed_filter` parameter to prune the BFS frontier to anomalous entities, and a routing fix that drops the per-call scoped edge-table read in favour of the cached global adjacency. Critical correctness fix in parallel chain extraction: worker-local `chain_id` counters were producing colliding ids in the merged output — post-merge dedup now reassigns from the global merged index, restoring `primary_key` uniqueness in the chain pattern points table. Spheres built before the fix carry colliding chain_ids; rebuild required. Sphere format stays at 2.4.
- **0.6.5** — sharpen chain generation: chain anchor pattern gains `cross_bank_count` (distinct banks the chain transits — textbook jurisdictional layering signal) and `amount_monotone_decreasing` (boolean, true when amounts strictly decrease at every hop — textbook structuring pattern), both auto-populated from `from_bank` / `to_bank` columns when present and gated on next chain pattern rebuild. `extract_chains` post-merge dedup adds a strict-prefix subsumption pass — chains whose entity sequence is a strict prefix of another chain's are dropped, eliminating near-duplicate rows for the same investigative finding. Sphere format stays at 2.4.

See [CHANGELOG](CHANGELOG.md) for the full feature list.

---

## 0.5.0

Detection-quality and investigation-workflow push.

- **0.5.0** — Storey adaptive π₀ FDR; drift direction on `π9_attract_drift`; multi-hop root-cause tracing (`trace_root_cause` DAG); geometric edge potential; structural motif scoring (`fan_out`, `cycle_2`, `cycle_3`, `structuring`).
- **0.5.1** — `fan_in` and `chain_k` motifs.
- **0.5.2** — `split_recombine` and `bipartite_burst` motifs; perf-hardening pass on the catalog.

---

## 0.4.0

Distribution-aware geometry, calibration fidelity, and graph acceleration.

- **0.4.0** — Distribution-aware Bregman divergence with per-dim kind tags; per-dimension anomaly threshold; in-memory adjacency index.
- **0.4.1** — Native graph algorithms as build-time dimensions (PageRank, betweenness, community, clustering, components); bidirectional BFS rewrite of `find_geometric_path`.

---

## 0.3.0

Lance perf upgrade, FDR control, builder intelligence.

- **0.3.0** — Aggregate engine rewrite on Lance SQL; precomputed contagion stats; format 2.2.
- **0.3.1** — Benjamini-Hochberg FDR control; submodular facility location; vectorized build with adaptive memory chunking.
- **0.3.2** — NumPy graph features; chunked pre-computation; geometric heredity (`find_novel_entities`).
- **0.3.3** — Agent navigation policy; dimension-selective similarity.

---

## 0.2.0

Graph meets geometry — edge table, runtime traversal, contagion/influence.

- **0.2.0** — Edge table; +11 navigator functions; contagion/influence primitives.
- **0.2.1** — Witness cohort discovery; investigative peer ranking.
- **0.2.2** — As-of graph reconstruction; `detect_cross_pattern_discrepancy` latency fix.

---

## 0.1.0

First public release — full GDS stack, π1–π12, builder, MCP server, validation suite.

---

## Future

**Detection quality**
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
