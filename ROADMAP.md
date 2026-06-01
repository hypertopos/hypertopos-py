# hypertopos — Roadmap

> Planned direction. Priorities may change based on feedback.
> See [CHANGELOG](CHANGELOG.md) for the full feature list per release.

## 0.8.0

Lance 7.0 single-process performance, incremental ingest, agent-correctness composers, and operations tooling — on the existing sphere format 3.x, no rebuild required.

- **0.8.0** — Lance 7.0 storage with native batched reads; incremental sphere ingest (`hypertopos sphere ingest`) without a full rebuild; agent-correctness composers (`assess_anomaly_certainty`, `consensus_classification`, `calibration_drift_report`, `diverse_explanations`, `theta_sensitivity_report`); vector-index health in `audit_pattern_dims`; cloud-ops CLI (`sphere health` / `validate` / `diff`); trajectory ANN index auto-skipped on small populations.
- **0.8.1** — incremental-ingest correctness (population-relative `conformal_p`, complete `{line, key}` polygon filtering, missing edge-dim-aggregation column guard); `π7_attract_hub` FDR polarity fix; `discover_chains` cyclic round-trip detection; build-time guards refusing grouped/GMM/FDR incremental ingest and `tracked_properties` + edge-block patterns; parameter-aware topological anomaly cache; single-pass population-statistic recompute.

---

## 0.7.0

Detector composition and counterfactual primitives on native Lance MVCC — multi-detector consensus, per-edge / joint / per-counterparty influence math, topological cycle persistence, multi-resolution FDR, and the entity-side investigation orchestrator land on a modernized Lance 6.0 storage substrate with sphere format 3.0.

- **0.7.0** — Lance 6.0 + sphere format 3.0; detector consensus (Wilson HMP), counterfactual family, topological cycle persistence, multi-resolution FDR, `investigate_entity` orchestrator.
- **0.7.1** — label-aware calibration foundation: `label_audit:` YAML, per-class Cohen's d + Fisher LDA on `Pattern`, `delta_norm_signed` Lance column.
- **0.7.2** — composition orchestrators: `find_anomalies(rank_by="signed_confidence")`, `chain_full_loop_summary`, `audit_label_alignment`, counterfactual frozen-population on `dive_solid`.
- **0.7.3** — investigation-orchestrator surface: per-dim AUROC, intrinsic/extrinsic decomposition, chain reliability rollup, DTW trajectory classifier, Louvain `community_id`, calibration-influencer auto-discovery, MCP ergonomics on `passive_scan` / `find_similar_entities` / `dive_solid`.

---

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

> Next major direction (planned): **0.9.0 — cloud foundations**, making hypertopos hostable. Priorities may change based on feedback.

**Cloud foundations (0.9.0 direction)**
- Session-keyed MCP server state with per-session isolation, so concurrent HTTP clients and multi-tenant hosting are safe.
- URI-addressable object storage (S3 / ADLS / GCS) for spheres, with a credential profile and a read-path prefetch cache.
- Serializable cold-open caches over the existing on-disk scalar indices, cutting first-query latency with no format change.
- Production-path architecture decision records (data freshness, multi-tenancy, permission passthrough) and a committed surface model: MCP for agents, library API for embedding, CLI for ops / batch / cloud-compute.

**Adoption & product**
- `scan <file>` auto-builder for one-command onboarding, plus a deterministic (no-LLM) interpretation layer that consolidates today's per-method explanations into a single standalone, audit-ready API.

**Detection quality**
- Multi-scale resolution — hierarchical build at daily/weekly/full granularity.

**PassiveScanner evolution**
- Native temporal source support for direct temporal inputs, without requiring manual dataset plumbing in benchmark scripts.
- **SphereProfiler** — autonomous sphere scanner that profiles all patterns, runs calibration sweeps across source combinations, proposes optimal PassiveScanner composition for Layer 1 surveillance.

**Code refactoring**
- Break up oversized modules into smaller, domain-focused components — `navigator.py` is now well over 20k lines and is the primary target for a mixin-based split that preserves the public single-class API.
- Consolidate repeated orchestration logic into shared helpers.

**Cross-sphere capabilities**
- Cross-sphere comparison — dimensionless metrics across independently calibrated coordinate spaces.

**Enterprise / governance**
- Dimension access control — per-agent visibility constraints on delta dimensions.

**Tooling**
- Runtime-latency reference tables in package docs, beyond the currently-documented `elapsed_ms` response field.
