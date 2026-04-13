# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback.

## Plan: 0.3.3

Agent navigation policy — skill-level changes to make agents effective navigators of geometric space.

- **Graph confirmation flow** — add witness_cohort + contagion confirmation step to investigator and fraud-investigator skills after explain_anomaly.
- **Investigation memory** — instruct agents to track checked entities, dead ends, and promising leads in conversation state.
- **Failure mode guards** — depth limits, anomaly strength thresholds, and force-switch after N consecutive calls on same entity.
- **Decision scoring** — priority queue heuristic (anomaly_strength + graph_support + temporal_signal + novelty) for iterative re-ranking of investigation targets.

---

## Plan: 0.4.0

Edge-derived geometry — the edge table becomes a first-class geometric citizen.

- **Edge-derived dimensions on event patterns** — degree, flow, velocity computed at build time as geometric dimensions. Edges feed geometry.
- **`find_motif` — structural pattern matching on edge table** — per-hop predicates, new navigation primitive for subgraph patterns invisible to flat queries.
- **Geometric edge potential** — relationship anomaly scoring via endpoint geometric distance. Geometry feeds edge scoring.

---

## 0.3.0

Lance perf upgrade, FDR control, builder intelligence.

- **0.3.0** — Aggregate engine rewritten around Lance SQL, precomputed contagion stats, format 2.2, edge table auto-detect fix.
- **0.3.1** — Benjamini-Hochberg FDR control, submodular facility location, vectorized build with adaptive memory chunking.
- **0.3.2** — NumPy graph features, chunked pre-computation, Lance compact tuning, per-dim index removal. Generalized dimension blocks (g/t/s), geometric heredity (`find_novel_entities`).

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
- Lazy chain geometry — on-demand chain delta vectors via sampled population calibration; supplements the build-time `chain_lines` path

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
