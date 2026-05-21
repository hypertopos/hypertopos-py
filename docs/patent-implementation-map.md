# Patent implementation map

This document maps the architectural claims behind hypertopos to the shipped implementation surfaces in the current release.

Each cluster names a load-bearing idea, the navigator or MCP surface that ships it, and the method one-line. The point is to make it easy for a reviewer or external auditor to walk from "this is what the project claims to do" to "this is the function that does it and the test that verifies it".

There are no claim numbers (`IC-XX`, `DC-XX.Y`) in this document — those are internal tracker IDs. Each cluster is named by its descriptive function.

---

## Per-entity geometric position derived from a population

**What it is.** Every entity carries a position in a multi-dimensional space whose axes are not raw column values but population-relative statistics — z-scores against the population mean, ratios against population percentiles, signed projections onto label-discriminating axes. The position is computed once at build time and stored as a Lance column.

**Surfaces.** `Pattern.mu_diag`, `Pattern.sigma_diag`, `Pattern.theta_diag` (population-level statistics persisted on every pattern). `Pattern.label_aware_calibration` (per-class moments + Fisher LDA direction, opt-in via `sphere.yaml`'s `label_audit:` block). `delta_norm` and `delta_norm_signed` Lance columns on every geometry table.

**Method.** Welford one-pass for mu / sigma. Quantile sketches for theta. Linear discriminant analysis with diagonal-covariance approximation for label-aware direction. All computations are deterministic and reproducible from the source data plus `sphere.yaml`.

**Tests.** `tests/test_calibration.py`, `tests/test_calibration_label_aware.py`, `tests/test_m1_1_builder_wiring.py`.

---

## Counterfactual closure — what changes if X is removed

**What it is.** For an anomalous entity, ask "would it still be anomalous if this edge, this counterparty, or this dim were removed?" — and answer with a closed-form recomputation of `delta_norm` against the same population statistics, no model retraining.

**Surfaces.** `GDSNavigator.simulate_edge_removal`, `simulate_counterparty_removal`, `select_minimal_joint_edge_removal`, `simulate_dimension_change`. Each surfaces as an MCP tool with the same name.

**Method.** Closed-form exp-family delta recomputation. For each candidate removal, recompute the entity's polygon contribution to `delta_norm` as `mu_diag - new_centroid`, take the ratio against `sigma_diag`, project against `theta_diag` percentile — same calibration, different inputs. Greedy selection for the minimal joint set.

**Tests.** `tests/test_counterfactual_*.py`.

---

## Multi-resolution false-discovery-rate control

**What it is.** Standard Benjamini-Hochberg FDR on a per-entity anomaly score answers one question — "is this entity anomalous?" — but ignores that entities live in nested cells (region × time × kind). Multi-resolution FDR aggregates per-cell p-values up a hierarchy with Tippett min-p and reports `q`-values at every level, surviving the more conservative join.

**Surfaces.** `Pattern.fdr_hierarchy` and `Pattern.fdr_temporal_hierarchy` (declarative schema in `sphere.yaml`). `find_anomalies(fdr_resolution, fdr_temporal_resolution)` MCP tool returns `cell_q_spatial` and `cell_q_temporal` per survivor. `engine.fdr.fdr_multi_resolution` (pure-math primitive).

**Method.** Per-cell Fisher exact 2×2 on an anomaly indicator vs background. BH or Storey correction within each cell. Tippett min-p aggregation up each declared level. Independent per-axis adjustment for spatial vs temporal hierarchies.

**Tests.** `tests/test_fdr_*.py`.

---

## Persistent-homology cycle persistence at the entity level

**What it is.** Treat each entity's k-nearest-neighbour ball in the polygon space as a small point cloud and ask whether that cloud carries persistent topological cycles. An entity with high `h1_max_persistence` sits inside a non-trivial topological structure — a ring, a hub, a cluster boundary — that single-entity z-scores cannot see.

**Surfaces.** `find_topological_anomalies` MCP tool. `engine.topology` module (Vietoris-Rips persistent homology backed by `ripser` when the optional `[topology]` extra is installed). `storage.topology_cache` for memoised per-entity persistence values.

**Method.** Per-entity k-NN ball (k ≈ 50 by default), PCA reduction to 10 dims, Vietoris-Rips H_1 persistence computed with `ripser`. Score = max `h1` death-birth interval. Cached because the computation is expensive; invalidated when calibration epochs bump.

**Tests.** `tests/test_topology_*.py`.

---

## Declarative compliance rules as a first-class build-time predicate language

**What it is.** Compliance officers write rules in YAML using a small declarative AST (`and / or / not` over `==, !=, <, <=, >, >=, in`); the builder evaluates them at build time and materialises a violations sidecar Lance dataset; a runtime MCP tool reads the sidecar with rule-id and severity pushdown. Separate from `delta_norm` anomaly flags — an entity can be one, the other, or both.

**Surfaces.** `Pattern.conformance_rules` (model field). `builder.conformance_mapping.parse_conformance_rules` (YAML loader with column-existence validation). `find_conformance_violations` MCP tool.

**Method.** Vectorised PyArrow expression evaluation. Severity literals `low < medium < high < critical`. Rule-id stamping. Sidecar Lance dataset alongside the geometry table.

**Tests.** `tests/test_conformance_mapping.py`, `tests/test_find_conformance_violations_mcp.py`.

---

## Detector composition via harmonic-mean p-value

**What it is.** Five independent detectors — `delta_norm`, `neighbor_contamination`, `segment_shift`, `trajectory_continuous`, `density_gap` — score each entity. A correct composition treats these as multiple hypothesis tests on the same null and combines them via the Wilson harmonic-mean p-value (HMP), not via score averaging. The combined `hmp` is rank-stable across detector subsets and exact under the null.

**Surfaces.** `combine_anomaly_pvalues`, `classify_detector_consensus`, `composite_risk`, `composite_risk_batch` MCP tools. `engine.composition.harmonic_mean_p` (pure-math primitive). `engine.p_value_calibration` (per-detector score-to-p-value adapters).

**Method.** Per-detector calibrated p-values. Wilson HMP combiner. Null-simulation lookup for the `L`-test threshold at level `alpha`. Per-detector weighting optional.

**Tests.** `tests/test_composition_*.py`, `tests/test_detector_consensus_*.py`.

---

## Chain-coherent investigation primitives

**What it is.** A chain is a multi-hop path through an event graph (`from_account → to_account` over time). Investigating a chain means asking whether the chain itself carries anomaly signal (`investigate_chain`), whether its members move together in regime space (`chain_drift_trajectory`), whether their explanatory witness sets intersect (`chain_witness_intersection`), and whether the chain's typology matches a known pattern (`classify_chain_typology`). The chain becomes a first-class investigation entity, not just a list of vertices.

**Surfaces.** `chain_lines:` YAML block. `extract_chains` build pass. `chain_pattern` derived geometry. `find_chains_with_coherent_anomaly`, `chain_witness_intersection`, `chain_drift_trajectory`, `classify_chain_typology`, `extend_chain`, `investigate_chain`, `generate_sar_rationale` MCP tools.

**Method.** Seeded BFS over the event graph with multi-criteria seed selection (`fan_out`, `cross_bank`, `multi_currency`, `pass_through`). Chain feature extraction at build time (`hop_count`, `is_cyclic`, `time_span_hours`, `amount_decay`, etc.). Same pattern-level calibration applied to chains as to entities, so `delta_norm` carries the same population-relative semantics.

**Tests.** `tests/test_chain_*.py`.

---

## Reliability triage on every flagged polygon

**What it is.** A polygon flagged anomalous can be flagged because one dim dominates the score (an artefact of that dim's calibration, not a real anomaly) or because the bootstrap confidence on `is_anomaly` is fragile (the flag would flip if the population were resampled). Both surfaces are reported as boolean flags alongside the score, so an investigator can corroborate before opening a case.

**Surfaces.** `reliability_flags` field on `find_anomalies`, `explain_anomaly`, `composite_risk`, `combine_anomaly_pvalues`, `investigate_entity`. Shape `{single_dim_driven, dominant_dim, dominant_dim_share, low_confidence_bucket, confidence, flags}`.

**Method.** `single_dim_driven` fires when one dim's attribution share exceeds 70 %. `low_confidence_bucket` fires when bootstrap-resampled `anomaly_confidence` falls below 0.5. Both computed at flag time, no extra MCP roundtrip.

**Tests.** `tests/test_reliability_flags.py`.

---

## Build-time dim quality auditing — silent calibration failure modes surfaced

**What it is.** Five build-time auditors catch the silent failure modes that break z-score / `delta_norm` semantics: dead dims (zero variance), sparse dims (mostly zero), dominant-dim mass (one dim accounts for ≥70 % of tail variance), negative-space dims (gaussian-declared but empirically zero-modal), non-normal dims (gaussian-declared but Shapiro-Wilk / KS `p < 0.01`), and per-pattern heteroscedasticity (Brown-Forsythe Levene `p < 0.01` on the grouping variable).

**Surfaces.** `sphere_overview.dim_quality_warnings[]` block. `engine.diagnostics` and `engine.dim_audit` modules.

**Method.** Build-time statistical tests on the calibrated geometry table. Each auditor returns a structured warning with `dim_label`, `reason`, `advice`, `evidence_value`, `threshold` — enough for a downstream consumer to either raise a calibration ticket or apply the recommended transform.

**Tests.** `tests/test_dim_quality_warnings.py`, `tests/test_diagnostics.py`.

---

## Notes

This document is a living artefact. As new architectural surfaces ship, they are added here. As surfaces are deprecated or refactored, the entries update with the cycle they shipped in.

Internal tracker IDs (`IC-XX`, `DC-XX.Y`, `OPF-XX`, `IDEA-XXX`) are kept out of this document by project rule — they are private bookkeeping and would look like dead hyperlinks to anyone reading from the public mirror. Each cluster is named by what it does, not by which ticket tracks it.
