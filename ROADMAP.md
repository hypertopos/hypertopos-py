# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback. 

## Current: 0.3.1

FDR control, diverse anomaly summarisation, and build pipeline optimization — Benjamini-Hochberg multiple-testing correction, submodular facility location, vectorized build with adaptive memory chunking. See CHANGELOG.

---

## 0.3.0

Lance 3.x perf upgrade — aggregate engine rewritten around Lance SQL, precomputed contagion stats, format 2.2, contagion anchor resolution fix, edge table auto-detect fix. See CHANGELOG.

---

## 0.2.2

As-of graph reconstruction across edge-table graph primitives + latency fix on `detect_cross_pattern_discrepancy`. See CHANGELOG.

---

## 0.2.1

Witness cohort discovery — investigative peer ranking. See CHANGELOG.

---

## 0.2.0

Graph meets geometry — edge table, runtime traversal, contagion/influence, +11 navigator functions. See CHANGELOG.

---

## 0.1.0

First public release — full GDS stack, π1–π12, builder, MCP server, validation suite. See CHANGELOG.

---

## Plan: 0.3.2

Builder intelligence — smarter geometry, faster builds.

- **Temporal build phase optimization** — pre-aggregated event matrix eliminates O(n × windows) rescans; 5–10× speedup on temporal phase, enabling practical iteration on richer sphere configurations.
- **Generalized (d+m+g+t+s) dimension blocks** — geographic, metric, and semantic dimension support in the builder beyond the current anchor/event model.
- **Geometric heredity for cold-start entities** — inherited geometric positioning for entities without sufficient history, using population priors and neighbor geometry.

---

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
- **Temporal build phase optimization** — pre-aggregated event matrix eliminates O(n × windows) rescans, enabling practical rebuilds with richer edge-derived dimensions.

---

## Future

**Detection quality**
- Edge-derived dimensions + temporal motif matcher
- Confidence scoring, robust estimators, multi-scale resolution — improve anomaly precision and reduce false positives on heavy-tail and multi-modal populations
- Lazy chain geometry — on-demand chain delta vectors via sampled population calibration; supplements the build-time `chain_lines` path

**Builder evolution**
- Incremental rebuild — geometry-only without `--force` wipe
- Generalized (d+m+g+t+s) dimension blocks — geographic / metric / semantic dimension support

**PassiveScanner evolution**
- Native temporal source support for direct temporal inputs, without requiring manual dataset plumbing in benchmark scripts
- Optional weighted scoring mode that uses continuous intensity instead of binary counts
- **SphereProfiler** — autonomous sphere scanner that profiles all patterns, runs calibration sweeps across source combinations, proposes optimal PassiveScanner composition for Layer 1 surveillance. Core loop: enumerate patterns → single-source anomaly scans → greedy multi-source combination → ranked composition report. No labeled GT needed; optional GT file unlocks supervised calibration.

**Code refactoring**
- Break up oversized modules into smaller, domain-focused components
- Reduce coupling between core layers by replacing private cross-layer access with explicit interfaces where practical
- Tighten error handling in hot paths so failures are surfaced more consistently
- Consolidate repeated orchestration logic into shared helpers instead of duplicating it across modules
- Incremental refactor: preserve behavior, improve structure, then revisit deeper architectural boundaries

**Cross-sphere capabilities**
- Cross-sphere comparison — dimensionless metrics across independently calibrated coordinate spaces
- What-if analysis — hypothetical edge changes producing modified coordinate vectors

**Enterprise / governance**
- Dimension access control — per-agent visibility constraints on delta dimensions

**Tooling**
- Runtime latency benchmarks in package docs
