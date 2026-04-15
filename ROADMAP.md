# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback.

## Plan: 0.5.0

Investigation depth, detection quality, edge-derived geometry, and advanced analytics.

- **Lazy chain geometry** — on-demand chain delta vectors via sampled population calibration; `discover_chains(include_geometry=True)` supplements build-time `chain_lines` for per-entity investigation without rebuild cost.
- **Multi-hop anomaly explanation chains** — automate the trace-to-root-cause workflow: anomalous entity → top dimension → edge entity → hub check → neighbor check → explanation chain with per-hop evidence. Collapses 4–6 tool calls into one.
- **Geometric edge potential** — relationship anomaly scoring via endpoint geometric distance; flags edges connecting geometrically distant entities.
- **Geometric dark matter** — population density gap detection; identifies coordinate-space regions that should have entities but are empty (anomaly by absence).
- **Gradient alignment on drift** — drift direction signal (normalizing vs deteriorating) on `find_drifting_entities` and `dive_solid`.
- **Edge-derived dimensions on event patterns** — pair-level and node-derived features computed at build time as geometric dimensions.
- **`find_motif` — structural pattern matching on edge table** — per-hop predicates for subgraph patterns invisible to flat queries.
- **Advanced coordinate space analytics** — new methods leveraging properties unique to population-relative coordinate construction.

---

## 0.4.0

Bregman geometry, anomaly confidence, distribution-aware calibration, build pipeline, graph acceleration.

- **0.4.0** — Distribution-aware Bregman divergence with per-dimension kind tags. Per-dimension anomaly threshold (hyper-ellipsoid). Theta calibration fix. Anomaly confidence via stratified bootstrap. Interpretive MCP output (`explain_anomaly` returns Bregman contributions). Sphere format 2.3. Per-pattern pipeline parallelism (geometry → temporal overlap, up to 4 threads). In-memory adjacency index — full graph in dict, O(1) neighbor lookups replacing Lance BTREE scans across all graph operations. Temporal bisect in chain extraction.
- **0.4.x** — Native graph algorithms as dimensions — PageRank, community detection (label propagation), betweenness centrality, triangle count, connected components computed at build time via adjacency index. Continuous features as derived dimensions, categorical features as group_by_property.

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
- Edge-derived dimensions + temporal motif matcher
- Confidence scoring, robust estimators, multi-scale resolution — improve anomaly precision and reduce false positives on heavy-tail and multi-modal populations

**Builder evolution**
- Incremental rebuild — geometry-only without `--force` wipe

**PassiveScanner evolution**
- Native temporal source support for direct temporal inputs, without requiring manual dataset plumbing in benchmark scripts
- Optional weighted scoring mode that uses continuous intensity instead of binary counts
- **SphereProfiler** — autonomous sphere scanner that profiles all patterns, runs calibration sweeps across source combinations, proposes optimal PassiveScanner composition for Layer 1 surveillance.

**Code refactoring**
- Break up oversized modules into smaller, domain-focused components
- Reduce coupling between core layers
- Consolidate repeated orchestration logic into shared helpers

**Cross-sphere capabilities**
- Cross-sphere comparison — dimensionless metrics across independently calibrated coordinate spaces
- What-if analysis — hypothetical edge changes producing modified coordinate vectors

**Enterprise / governance**
- Dimension access control — per-agent visibility constraints on delta dimensions

**Tooling**
- Runtime latency benchmarks in package docs
