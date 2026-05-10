# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback.

## 0.6.0

Architecture-emergent analytics, properly enabled — calibration is now versioned, and patent-aligned analytics ride on that versioning.

- **0.6.0** — multi-epoch calibration retention with `compare_calibrations`; intrinsic / extrinsic drift decomposition; declarative motif API (`find_motif_by_hops`).
- **0.6.1** — anchor-pattern aggregation of edge-derived dimensions; declarative motif API completion (`HopPredicate`, anchor-companion scoring).
- **0.6.2** — chain-anchor `edge_dim_aggregations:` support; vectorized `AdjacencyIndex` load path.
- **0.6.3** — `edge_dim_aggregations:` expanded with `_std`, `_p95`, `_count_above_threshold`; k>2 composite anchors via positional `key_cols`.
- **0.6.4** — chain-coherent investigative loop primitives: `find_chains_with_coherent_anomaly`, `anomaly_propagation_in_chain`, `classify_chain_typology`, `extend_chain`.
- **0.6.5** — chain anchor pattern gains `cross_bank_count` and `amount_monotone_decreasing`; `extract_chains` strict-prefix dedup.
- **0.6.6** — agent-side close of the chain investigative loop: smart-mode chain routing, `sphere_overview.suggested_queries` entry-point, R9 cheatsheet; `theta_sensitivity` calibration introspection; length- and volume-stratified correlation gates.
- **0.6.7** — close the investigation→SAR pipeline: `dimension_weights` runtime override on `find_anomalies`, `chain_investigation_summary` triage, `investigate_chain` one-shot R9 orchestrator, `generate_sar_rationale` template-based SAR narrative draft; `dim_quality_warnings` (dead / sparse dim) on `sphere_overview`; external-chains-as-anchor-line cookbook.

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
