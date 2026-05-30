# Changelog

All notable changes to hypertopos will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] — 2026-05-30

### Added

- `hypertopos sphere health <path>` — composes the population summary with the geometric health checks into a single status (`ok` / `warning` / `critical`); `--exit-code-on-critical` exits `2` on a critical status so a CI shell gate can fail a deploy, and `--json` emits a `{status, sphere_path, overview, alerts}` document.
- `hypertopos sphere validate <path> [--strict] [--json]` — structural integrity check over a built sphere (sphere.json parses, each materialized line has a points directory, each pattern has a geometry directory); `--strict` promotes calibration-health and dimension-quality warnings to errors, exiting `1` when invalid.
- `hypertopos sphere diff <old> <new> [--json]` — pre-deploy diff reporting the pattern-inventory delta (added / removed / common) and per-shared-pattern calibration drift between two built spheres; patterns whose calibration schema differs are marked `not_comparable`.
- `hypertopos sphere ingest <sphere> --points <file>` — incremental ingest CLI: append a new-/changed-entities table (Arrow IPC, Parquet, or CSV) to one pattern's geometry without a full rebuild. `--pattern` selects the target pattern (optional when the sphere has a single one); `--reindex` forces an immediate ANN index rebuild; `--finalize` recomputes the global rank percentile and rebuilds the ANN index once at the end of a batched ingest session; `--json` emits an `{added, modified, deleted, population_size, drift_pct, geometry_version_before, geometry_version_after, reindexed, finalized}` summary for pipeline consumption.
- Production-ready incremental ingest: `GDSBuilder.incremental_update` adds, modifies, and deletes entities on an existing sphere without a full rebuild, and newly added entities are immediately visible to anomaly search. Patterns whose dimensions are relations, event dimensions, edge-dim aggregations, and tracked properties are now fully supported (previously such patterns raised a width-mismatch error). The appended rows now also enter the ANN vector index — pass `reindex=True` to rebuild it immediately. For batched ingestion of many small appends, pass `recompute_ranks=False` to defer the population percentile recompute and call the new `GDSBuilder.finalize_incremental` once at the end of the session.

### Changed

- Per-tag calibration-epoch timestamps now resolve directly from the dataset tag listing (a single lookup) instead of scanning version history, with read-side fallback for spheres written by older tooling.
- Batch point reads (`read_points_batch`) issue a single indexed lookup over the whole key set on a reused dataset handle, replacing the per-key access loop — repeated lookups of the same entities are dramatically faster.
- `pylance` dependency floor bumped to 7.x — Lance dataset reads, writes, scalar-index builds, and `merge_insert` upsert behavior verified equivalent to the prior 6.x floor.
- Builder skips the trajectory ANN index when a pattern has too few rows to train it, shortening sphere builds for patterns with few entities; `attract_trajectory` (π10) and `find_drifting_similar` still return correct nearest-trajectory results via an exact full-scan fallback when no index is present.

## [0.7.3] — 2026-05-27

### Added

- `GDSNavigator.π12_attract_regime_change` (and its cache fast-path) — each detected changepoint now carries a `near_data_boundary: bool` field. Fires `True` when the changepoint timestamp index falls in the first two or last two temporal buckets of the filtered range — agents can suppress these as low-confidence detections sitting on insufficient one-sided context (the buckets at the data range edges have asymmetric neighbours, which inflates the shift magnitude). Applies to both the cache fast-path (`_pi12_from_cache`) and the cold-scan path; no schema change.
- `Pattern.signed_percentiles` — per-pattern percentile cache (`p1`, `p5`, `p50`, `p95`, `p99`) over the `delta_norm_signed` Lance column, populated at build time when label-aware calibration fires. Drives the new `signed_tail_concentration` dim_quality warning that flags one-sided extreme tails (`|p99| / max(|p50|, 1e-9) > 50`) on the Fisher LDA-projected delta distribution; suppressed when `Pattern.label_aware_n_pos < 30` to avoid noise from undersampled positive classes.
- `GDSEngine.decompose_displacement(delta, label_direction)` — pure-math helper that splits a polygon delta vector along (`intrinsic`) and orthogonal to (`extrinsic`) any label-discriminating axis, satisfying `intrinsic^2 + extrinsic^2 == ||delta||^2`. Pattern-level means `intrinsic_displacement_mean` / `extrinsic_displacement_mean` are persisted on `Pattern` at build time when label-aware calibration fires.
- `Pattern.label_aware_n_pos` / `Pattern.label_aware_n_neg` — positive / negative labelled sample counts captured at build time, co-populated with `label_aware_calibration`.
- `GDSNavigator.chain_signed_confidence_rollup(chain_id, chain_pattern=..., anchor_pattern=...)` — chain-level reliability rollup derived from per-member `signed_confidence` ranking on the anchor pattern. Returns `chain_mean_signed_confidence`, `chain_n_low_confidence_members`, `chain_n_single_dim_driven_members`, and `chain_confidence_verdict` ∈ {high, medium, low, label-aware-unavailable}. Falls back cleanly with verdict `"label-aware-unavailable"` and null numeric fields when the anchor pattern lacks `label_aware_calibration`.
- `GDSNavigator.aggregate` per-entity result rows gain `anomaly_rate: float | null` — fraction of an entity's events flagged anomalous (`delta_norm >= theta_diag`). `null` when entity has zero events or pattern lacks anomaly threshold.
- `GDSNavigator.sphere_overview` gains `cross_pattern_discrepancy: dict | null` — pairwise Jaccard overlap of anomalous primary_keys across cover-overlapping patterns. Lists pairs with `pattern_a`, `pattern_b`, `shared_line`, the four anomaly-bucket counts, and `jaccard_anomaly_overlap`. `null` on spheres with fewer than two cover-overlapping patterns.
- `GDSNavigator.find_anomalies(..., sample_size: int | None = None, boundary_aware: bool = False)` — opt-in sampling for the in-process scan path. `sample_size` caps geometry rows scored; `boundary_aware=True` allocates half the budget to entities within the `[0.8 * theta_norm, 1.2 * theta_norm]` band around the decision threshold, the other half drawn from the rest of the population. Defaults preserve the existing full-scan behaviour. `boundary_aware=True` requires `sample_size` to be set.
- `engine.topology.classify_trajectory(solid_table, *, sample_size=10000)` — wraps `trajectory_continuous_score` with a deterministic 4-category classifier (`outlier` / `lagging` / `leading` / `typical`). Combines DTW distance against the population-median trajectory with a first-derivative slope comparison via `scipy.stats.linregress`. Returns a list of `{primary_key, dtw_distance, category, category_evidence}` per entity, with `category_evidence` carrying the signed deviation that drove the classification.
- `GDSNavigator.classify_trajectory(primary_key, pattern_id, sample_size=10000)` — per-entity navigator wrapper that streams temporal `shape_snapshot` rows, projects to `delta_snapshot` via the pattern's `(mu, sigma_diag)`, and returns the classification for the requested entity.
- `sphere.yaml` `graph_features.features:` accepts a new token — `community_id` — Louvain modularity-optimizing community detection on the anchor-pattern graph. `igraph.Graph.community_multilevel` when igraph is installed; `networkx.algorithms.community.louvain_communities` otherwise; gracefully emits an empty mapping (caller fills nulls) when neither backend is available. Community IDs are renumbered so the largest community is `0`.
- `graph_features.features:` `pagerank` and `connected_component` now route through pure-numpy helpers (damped PageRank with damping `0.85`, max `100` iterations, L2 convergence threshold `1e-6`; Union-Find connected components with path compression and union-by-rank). No external graph library is required for these two; igraph remains optional and is used only for the other features in the block.
- `engine.graph.compute_pagerank`, `compute_louvain_community`, `compute_connected_components`, `compute_from_adjacency` — pure helpers exposed for direct use on an `AdjacencyIndex`. PageRank parameters (`damping`, `max_iter`, `tol`) are configurable per call.
- `GDSNavigator.find_calibration_influencers(..., auto_discover: bool = False, auto_k: int = 10)` — auto-discovery mode runs k-means on the pattern's geometry, surfaces top-K candidate influencers from cluster centroids, then ranks each by existing μ-impact algorithm. Returns the existing shape extended with per-candidate `cluster_size` and `cluster_centroid_distance`. Default `auto_discover=False` preserves manual mode requiring caller-supplied candidate_keys.
- `GDSNavigator.calibration_influencer_history(primary_key, *, pattern_id)` — per-epoch μ-impact history for a known influencer. Lazily populated write-through cache at `_gds_meta/calibration_history/<pattern>/influencer_<primary_key>.json`. Returns chronological list of `{epoch, calibrated_at, mu_impact, delta_norm_impact}` entries. Empty list + warning when cache absent.
- `engine.topology.local_trajectory_shape(delta_norms)` — pure-function classifier that categorises an entity's own `delta_norm` time-series as `arch` / `V` / `linear` / `flat` without requiring a population reference. Returns `None` when fewer than three samples are available. Used by `dive_solid` to surface trajectory shape on the entity's temporal slices.
- `Navigator.find_similar_entities(..., with_neighbor_anomaly=False)` — opt-in flag triggers a single point lookup over the returned top-N neighbour keys and populates the `SimilarityResult.is_anomaly_map` sidecar with stored anomaly flags. Existing 2-tuple iteration is preserved.

## [0.7.2] — 2026-05-21

### Added
- `GDSNavigator.π5_attract_anomaly` / `find_anomalies` accept `rank_by="signed_confidence"` — composes `delta_norm_signed`, the Fisher LDA alignment, and the reliability flags into one confidence-weighted score: `score = delta_norm_signed × |lda_alignment| × (1 − reliability_penalty)` where `reliability_penalty = 0.5 × single_dim_driven + 0.5 × low_confidence_bucket`. Survivors carry `signed_confidence_score`, `lda_alignment`, `reliability_penalty` per polygon. Pattern without `label_aware_calibration` raises `GDSNavigationError` — no silent fallback to `delta_norm` ranking.
- New per-dim `kind_mismatch` warning class on `sphere_overview.dim_quality_warnings`. Fires when a `kind='gaussian'` dim shows `|direction_component| < 0.05` (LDA assigns near-zero weight) AND `cohens_d_pos_neg >= 0.3` (raw class stats separate) — the dim's variance is captured by another dim's Fisher axis, suggesting kind re-declaration or split. Composes with label-aware calibration; suppressed for dims already flagged `negative_space`. Pattern-level prerequisite: `Pattern.label_aware_calibration` must be non-None.
- `GDSNavigator.audit_label_alignment(pattern_id, *, top_n=10)` — reports Fisher LDA direction's discrimination power as AUROC of `delta_norm_signed` against the binary label column declared in `sphere.yaml`'s `label_audit:` block. Returns `{pattern_id, auroc, n_pos, n_neg, top_dims}` where `top_dims` carries the `top_n` dims sorted by `|direction_component|` desc (most label-discriminating axes first). Returns fallback shape with `auroc: null` and `label_aware_available: False` on patterns built without `label_audit:`.
- `Chain.to_dict()` carries an additional `edge_potentials: list[float | null]` field — one Euclidean distance `||delta(keys[i]) - delta(keys[i+1])||` per consecutive-pair hop, computed against a caller-supplied anchor pattern delta lookup. `null` on hops with missing polygon, mismatched delta shapes, or non-finite distance (NaN / inf strict-JSON sanitised). Length equals `len(keys) - 1` (empty list for single-entity chains). Surfaces high-distance behavioural jumps inside multi-hop chains for downstream chain investigation primitives.
- `GDSNavigator.π3_dive_solid` accepts `counterfactual_frozen_population: bool = False` — when `True`, each `SolidSlice` on the returned solid carries an additional `delta_norm_frozen_pop` field reporting the per-slice L2 norm recomputed against the FIRST slice's raw shape as the entity-relative reference epoch (sigma stays at the current pattern's diagonal). Answers "is this entity's apparent normalization a real shift, or just population drift around a stationary entity?" — a stationary entity yields `delta_norm_frozen_pop = 0` across all slices. Default `False` preserves the existing return shape (`delta_norm_frozen_pop = None`).
- `engine.counterfactual.recompute_delta_norm_against_frozen(shape, mu_frozen, sigma)` — pure-NumPy helper used by the frozen-trajectory path. Sigma-dead dims contribute zero (same convention as the per-edge counterfactual primitives).
- `SolidSlice.delta_norm_frozen_pop: float | None` — new optional field on the model dataclass populated by the frozen-trajectory path; `None` for the default build path.
- `docs/refutations.md` — record of tested-and-closed hypotheses: pattern-level σ-estimator swap, per-dim heuristic σ-swap, lazy-chain sampled calibration, Bregman-based `is_anomaly` flag, plus a methodology refutation on cycle-compression. Each entry states the hypothesis, dataset, numeric verdict, root cause, and closure rule.
- `docs/patent-implementation-map.md` — descriptive map of nine architectural-claim clusters (per-entity geometric position, counterfactual closure, multi-resolution FDR, persistent-homology cycle persistence, declarative compliance, HMP detector composition, chain-coherent investigation, reliability triage, build-time dim quality auditing) to their shipped surfaces. No internal tracker IDs.

### Changed
- README adds CI test workflow badge linking to the public-mirror Actions run.

## [0.7.1] — 2026-05-20

### Added

- `GDSBuilder` now invokes `engine.calibration_label_aware.calibrate_label_aware` per pattern listed in `sphere.yaml`'s `label_audit:` block when the `--label-aware-calibration` build flag is set. The result is persisted onto `Pattern.label_aware_calibration` (new field, type `dict[str, dict[str, float]] | None`) and `signed_direction_vector` populates the `delta_norm_signed` Lance column. Closes the wiring gap that left `audit_pattern_dims` MCP tool in fallback mode on every sphere prior to this build path.
- `Pattern.heteroscedasticity_diagnostic` — optional per-pattern dict, keyed by the `group_by_property` column name, persisted at build time when a pattern carries `group_by_property`. Each entry is a Brown-Forsythe (median-centred Levene) test result on `delta_norm` partitioned by the grouping column: `{W_statistic, p_value, k_groups, per_group_variance, per_group_n, skipped_groups_low_n}`. Groups with N < 30 are dropped from the test (low Levene power) and counted in `skipped_groups_low_n`. When all qualifying groups have zero residual variance Levene returns NaN; the diagnostic encodes that as `null` for `W_statistic` / `p_value` so the JSON is well-formed and consumers can treat "computed-but-degenerate" the same way as "fewer than two qualifying groups".
- `sphere_overview.dim_quality_warnings[]` gains a new pattern-level type `"heteroscedasticity"` — fires when the persisted Levene `p_value < 0.01` on a pattern's grouping variable. The `dim_label` carries the grouping column name (a categorical line property), not a δ-dim — the warning's role is to flag that the global θ / Cohen's d pooled-σ / APS global-percentile assumptions are violated for this pattern. Carries `evidence_value` (p-value) and `threshold` (0.01) so the agent can rank patterns by severity. Advice: keep the per-group θ calibration the pattern already carries, or apply a variance-stabilizing transform (`log1p`) on `delta_norm` before thresholding if per-group θ is undesirable downstream.
- `engine.diagnostics.levene_test_per_group(values, group_ids)` — pure-math primitive (NumPy + scipy.stats) returning `{W_statistic, p_value, k_groups, per_group_variance, per_group_n, skipped_groups_low_n}`. Uses Brown-Forsythe (median-centred) variant of Levene for robustness on skewed distributions. Independent of any builder state — callable on any `(values, group_ids)` pair.
- `engine.dim_audit.normality_test_per_dim(values)` — per-dim normality test primitive. Selects Shapiro-Wilk for samples up to 5000 observations and Kolmogorov-Smirnov against a fitted normal for larger samples; returns `{test_name, statistic, p_value, n}`. Strips NaN before testing and returns a `nan` p-value for samples with fewer than three finite values or zero variance, letting callers treat those cases as "insufficient data" without pre-filtering.
- `Pattern.dim_normality_pvalues` and `CalibrationFit.dim_normality_pvalues` — optional `{dim_name: p_value}` mapping populated at build time by walking the same float columns the `dim_percentiles` cache covers (entity-line columns plus aggregated edge dims). Persisted to `sphere.json` and the calibration history; legacy spheres lacking the field continue to load unchanged.
- New `non_normal_dim` warning class on `sphere_overview.dim_quality_warnings`. Fires when a dim declared with `kind='gaussian'` has a build-time normality p-value below 0.01 — surfaces dims where the z-score `(x - mu) / sigma` is a poor anomaly scorer because the empirical distribution is heavy-tailed. The warning carries `reason`, `advice` (suggesting `log1p` / `sqrt` / `rank` transform or kind re-declaration), `evidence_value` (the p-value), and `threshold` (0.01). Suppressed for dims already flagged by `negative_space` to avoid double-flagging when the kind itself is the bug. Bernoulli and poisson dims are silently skipped — normality testing does not apply to binary or discrete-count kinds.
- `Pattern.dim_percentiles` now covers aggregated edge dimensions on anchor patterns. Each `{source_dim}_{aggregate}` label declared via `edge_dim_aggregations` receives the same six-key entry (`min / p25 / p50 / p75 / p99 / max`) that the cache already provided for `event_dimensions` and `prop_columns`. Consumers of the percentile cache — `dim_quality_warnings` in `sphere_overview`, `dominant_dim_mass` / `negative_space` auditors, the percentile-driven scoring path in `find_anomalies` — now see aggregated edge dims uniformly alongside the other dim families.
- `engine.calibration_label_aware.calibrate_label_aware(deltas, labels, dim_labels=None, regularization=1e-6)` — per-dim label-aware calibration producing `{mu_pos, sigma_pos, mu_neg, sigma_neg, direction}` per dim plus the unit-norm Fisher LDA direction vector across all dims; opt-in calibration path layered on top of standard mu/sigma stats.
- `hypertopos build --label-aware-calibration` CLI flag — opt-in switch that activates label-aware calibration during build. No-op until the sphere config declares a `label_audit` block selecting patterns; behavior unchanged on unlabeled spheres.
- `delta_norm_signed` Lance column in the geometry dataset — per-polygon projection of the delta vector onto the label-aware Fisher LDA direction, signed so positive values mean the polygon is pushed toward the positive-labelled centroid and negative values toward the negative. Nullable float32; populated only for patterns whose builder calibration produced a `signed_direction_vector`. Patterns without label-aware calibration emit the column as all-null, preserving the existing unsigned `delta_norm` magnitude untouched.
- `sphere.yaml` now accepts a top-level `label_audit:` block — `{label_column, label_positive_value, patterns: [...]}` — naming the binary-label column on the entity line, the value treated as the positive class, and the patterns to calibrate. The parser rejects malformed blocks (missing `label_column`, missing `label_positive_value`, missing or empty `patterns`, unknown pattern names). The block is persisted into `sphere.json` and surfaced to consumers via `Sphere.label_audit`.
- Sphere format minor `3.1` — stamped on `sphere.json` only when the build registers a `label_audit` block; otherwise spheres continue to stamp `3.0`. Readers compare on the major component only, so older 3.x readers transparently load newer 3.x spheres and ignore the new optional field.
- `Pattern.edge_dim_names: list[str]` — edge-derived dim names persisted on event patterns that declared an `edge_dimensions:` block in `sphere.yaml`. The block is written into `pattern.json` and surfaced via `Pattern.dim_labels` and `Pattern.dim_index(name)` in storage order (after event dims, before prop columns), so every stored delta dim now has a labelled name. `Pattern.delta_dim()` matches the stored mu / sigma / theta width for these patterns; empty list for patterns without `edge_dimensions:`. Rebuild required to surface labels on existing spheres.

### Changed
- `storage.reader.GDSReader.read_sphere` accepts any sphere with format major 3; pre-3.x and malformed `format_version` strings continue to raise `GDSVersionError` with a rebuild hint.
- `builder.conformance_mapping.parse_conformance_rules(raw_block, *, pattern_id, available_columns=None)` — YAML loader for the per-pattern `conformance_rules:` block. Accepts the declarative predicate AST (`and`, `or`, `not`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`) and the `severity` literals `low`, `medium`, `high`, `critical`. Validates structural shape (logical compounds require non-empty `terms`, `not` requires exactly one term, `in` requires an iterable `value`), assigns `rule_id` of the form `r0`, `r1`, … when absent, rejects duplicate `rule_id`s, and — when `available_columns` is supplied — fails fast on any leaf `prop` that is not a column on the entity points table. Returns a list of `ConformanceRule` ready to attach to `Pattern.conformance_rules`.
- `sphere.yaml` patterns now accept a `conformance_rules:` block. The block is a list of entries shaped `{rule_id?, severity, description?, violates_when}` where `violates_when` is the predicate AST mirroring the on-disk JSON shape. The build pipeline parses, validates against the entity-line points table, attaches the rule set to the pattern, and writes the violations sidecar alongside the geometry — every entity matching `violates_when` lands in the sidecar tagged with the rule's `rule_id` and `severity`.
- `builder.add_pattern(..., conformance_rules=...)` keyword — programmatic equivalent of the YAML block, accepting a list of `ConformanceRule` directly for callers using the builder API without YAML.

### Fixed
- `anomaly_summary` no longer falls back to synthesized `dim_{i}` labels for the `top_driving_dimensions` block on event patterns that declared `edge_dimensions:`. Every reported dim now resolves to the labelled name persisted on the pattern; the per-cluster `dim_sq_totals` accumulator is sized by `len(pattern.dim_labels)` and the round-trip is consistent with `Pattern.delta_dim()`.

## [0.7.0] — 2026-05-18

### Added

#### Sphere format 3.0 + Lance 6.0 modernization (breaking)
- **Sphere format bumped 2.4 → 3.0 (breaking).** Geometry is stored as a single Lance dataset per pattern with calibration epochs tracked as native Lance dataset versions tagged `epoch_<N>`; the public navigator API is unchanged. **Spheres on format 2.4 do not open on this release** — `GDSVersionError` points to a clean rebuild.
- Lance dependency bumped 4.0.0 → 6.0.0; auto-applied SIMD distance kernels, zonemap improvements, manifest interning, eager I/O scheduling, and FTS prewarm wins on the cold read path.
- `storage.reader.read_geometry_batched` and `storage.reader.read_temporal_batched` now batch by bytes instead of rows; the `batch_size` parameter is renamed `batch_size_bytes`.

#### Entity-side investigation orchestrator
- `GDSNavigator.investigate_entity(primary_key, *, pattern_id, line_id, chain_pattern_id=None, include_polygon=True, include_explain=True, include_witness_cohort=True, include_chains=True, include_root_cause=True, include_graph_geometry_tension=True, include_per_edge_counterfactual=False, top_n_witnesses=5, top_n_chains=3, top_n_edges=5)` — one-call entity investigation orchestrator returning one block per included step plus `steps_status` mapping step name to `{ok, error}` and `elapsed_ms`.

#### Detector composition
- `GDSNavigator.combine_anomaly_pvalues(pattern_id, *, detectors, weights, sample_size, top_n)` — multi-detector anomaly consensus across `delta_norm`, `neighbor_contamination`, `segment_shift`, `trajectory_continuous`, and `density_gap` (skipped silently); returns ranked `{primary_key, hmp, p_per_detector, rank}`.
- `GDSNavigator.classify_detector_consensus(pattern_id, *, detectors, sample_size, top_n, anomaly_threshold, normal_threshold)` — categorical detector-agreement typology with band-gap thresholding returning `{primary_key, classification, anomalous_detectors, normal_detectors, borderline_detectors, n_detectors_fired, hmp, p_per_detector, rank}`.
- `engine.composition.harmonic_mean_p(p_values, *, weights=None)` — Wilson harmonic-mean p-value combiner with optional per-detector weights.
- `engine.composition.hmp_threshold_at_alpha(L, alpha, *, n_simulation_draws=1_000_000)` — null-simulation HMP threshold lookup at level `alpha` for `L` tests.
- `engine.topology.trajectory_continuous_score(solid, *, sample_size)` — per-entity DTW distance against the population-median trajectory.
- `engine.p_value_calibration` module — five adapter functions mapping detector scores to per-entity p-values: `detector_p_value_delta_norm`, `detector_p_value_neighbor_contamination`, `detector_p_value_segment_shift`, `detector_p_value_density_gap`, `detector_p_value_trajectory_continuous`.
- `composite_risk` and `composite_risk_batch` now combine cross-pattern p-values via the Wilson harmonic-mean p-value (HMP) instead of Fisher's method; `combined_p`, `n_patterns`, `per_pattern{}` retained, `chi2` and `df` removed.

#### Counterfactual suite
- `GDSNavigator.simulate_edge_removal(primary_key, *, pattern_id, line_id, top_n=5, edge_ids=None)` — per-edge counterfactual returning `(edge_id, delta_norm_before, delta_norm_after, drop_pct, dominant_dim_label, source_value_pvalues, min_pvalue, dominant_significance_dim, dimensions_skipped)` ranked by composite of `|drop_pct|` and source-value extremeness.
- `GDSNavigator.simulate_counterparty_removal(primary_key, *, pattern_id, line_id, top_n=5, edge_top_n=None)` — per-counterparty rollup returning `{partner_key, n_edges, sum_drop_pct, sum_abs_drop_pct, max_abs_drop_pct, dominant_dim_label, edge_ids}` sorted by `sum_abs_drop_pct`.
- `GDSNavigator.select_minimal_joint_edge_removal(primary_key, *, pattern_id, line_id, target_drop_pct=50.0, k_max=10)` — greedy joint counterfactual returning `{selected_edge_ids, selected_partner_keys, achieved_drop_pct, achieved_abs_drop_pct, selection_sequence, target_reached, k_max_reached, delta_norm_before}`.
- `GDSNavigator.simulate_dimension_change(primary_key, *, pattern_id, line_id, set_dimension, top_n=5)` — what-if dimension override returning `delta_norm_before/after`, `delta_norm_pct_change`, `is_anomaly_before/after/change`, `top_witness_dims_after`, `dimensions_overridden`.
- `engine.counterfactual` module — pure-math primitives for direct embedding: `simulate_joint_edge_removal`, `select_minimal_joint_removal`, `aggregate_edge_removals_by_counterparty`, `ecdf_pvalue_upper_tail`, `compute_per_edge_source_value_pvalues`.

#### Multi-hypothesis explanation
- `GDSNavigator.find_diverse_explanations(primary_key, *, pattern_id, n_hypotheses=3, min_contribution_pct=0.10, validate=False)` — K diverse disjoint hypotheses for why an entity is anomalous returning `{primary_key, pattern_id, delta_norm, theta_norm, n_hypotheses_requested, n_hypotheses_returned, hypotheses, diversity_score, degraded_reason}`.

#### FDR upgrades
- `engine.fdr.fdr_multi_resolution(cell_p_values, *, hierarchy, temporal_levels, method, alpha)` — multi-resolution FDR over a cell-tuple lattice with Tippett min-p aggregation up each declared level.
- `engine.fdr.cell_p_values_from_anomaly_indicator(geometry, *, hierarchy_dims, temporal_dim, anomaly_col)` — per-cell Fisher exact 2×2 upper-tail p-value on an anomaly indicator.
- `Pattern.fdr_hierarchy: list[FDRHierarchyLevel]` and `Pattern.fdr_temporal_hierarchy: list[FDRTemporalLevel]` — declarative sphere.yaml schema for spatial and temporal FDR hierarchies; `GDSNavigator.π5_attract_anomaly` and `find_anomalies` gain `fdr_resolution` and `fdr_temporal_resolution` parameters, survivors carry `cell_q_spatial`, `cell_q_temporal`, `cell_path`. When a resolution is set, unspecified `p_value_method` defaults to `"chi2"` and unspecified `fdr_method` to `"storey"`.
- `builder.temporal_bucket` — opt-in per-anchor centroid-timestamp bucketing materialiser triggered automatically when `fdr_temporal_hierarchy:` declares a `slice_dimension` missing from the geometry table.
- `engine.fdr.per_dim_p_values_chi2_univariate(deltas)` and `engine.fdr.fdr_per_dimension(p_values_per_dim, *, alpha, method)` — per-entity-per-dim chi²(1) two-sided p-values with BH/Storey correction applied independently per column.
- `GDSNavigator.π5_attract_anomaly` gains `fdr_axis: "entity" | "per_dim" | "both"` and `rank_by: "delta_norm" | "min_q_per_dim"`; per-dim mode attaches `q_values_per_dim`, `min_q_per_dim`, `dominant_q_dim_idx` to each survivor.

#### Chain extensions
- `GDSNavigator.chain_witness_intersection(chain_id, *, chain_pattern, member_pattern, min_jaccard=0.5, top_k_witness=5)` — coordinated-witness diagnosis returning `{chain_id, chain_pattern, member_pattern, n_members, n_members_explained, n_members_skipped, intersected_witness_dims, union_witness_dims, mean_pairwise_witness_jaccard, coordinated, interpretation, per_member_top_dims}`.
- `GDSNavigator.chain_drift_trajectory(chain_id, *, chain_pattern, member_pattern, n_windows=4)` — per-member regime + chain-level drift score returning `{chain_id, chain_pattern, member_pattern, n_members, n_members_with_history, n_members_skipped, n_members_short_history, n_windows, per_position_trajectory, chain_level_regime, chain_drift_score}`.

#### Graph-geometry primitives
- `GDSNavigator.find_graph_geometry_tension(primary_key, *, pattern_id, line_id, k_geometric=20, top_n_hidden=5, top_n_suspicious=5)` — per-entity 2×2 cross-tab of behavioural k-NN vs graph adjacency returning `{primary_key, hidden_cluster, suspicious_links, tension_score}`.
- `edge_curvature_frc` — opt-in build-time edge dimension materialising combinatorial Forman-Ricci curvature per transaction edge via `edge_dimensions: [edge_curvature_frc]`.
- `engine.dim_audit.fit_lda_direction(*, deltas, labels, regularization=1e-6)` — Fisher LDA direction fit over labeled delta vectors returning `{direction, fisher_score, n_anom, n_normal}`.

#### Reliability triage
- `engine.geometry.compute_reliability_flags(delta, *, pattern, anomaly_confidence, dominant_dim_threshold=0.7, confidence_threshold=0.5)` — per-polygon reliability triage returning `{single_dim_driven, dominant_dim, dominant_dim_share, low_confidence_bucket, confidence, flags}`. Surfaced on `π5_attract_anomaly`, `explain_anomaly`, `composite_risk`, `combine_anomaly_pvalues`, and `investigate_entity`; additive on every surface.

#### Persistent-homology stack
- `GDSNavigator.find_topological_anomalies(pattern_id, *, top_n=20, force=False, sample_size=50_000, k_neighbors=50, homology_dim=1, pca_dim=10)` — per-entity local k-NN Vietoris–Rips H_1 cycle-persistence ranking returning `{primary_key, topo_score, h1_max_persistence, h0_mean_death, n_h1_features, computed_at}`; optional dependency `pip install hypertopos[topology]`.
- `engine.topology.find_topological_anomalies(geometry_table, *, k_neighbors, homology_dim, pca_dim, sample_size, top_n)` — engine primitive accepting any Arrow table with `primary_key` plus numeric feature columns.
- `engine.topology.find_topological_trajectory_anomalies(solid_table, *, homology_dim, min_timesteps, pca_dim, sample_size, top_n)` — per-entity trajectory-PH anomaly score returning `trajectory_topo_score`, `dominant_feature_birth`, `dominant_feature_death`.
- `storage.topology_cache` module — sidecar cache helpers `cache_path`, `read_cache`, `write_cache` with `ANOMALIES_SCHEMA` and `TRAJECTORY_SCHEMA`.

#### Sphere-validation cheap-tier
- `sphere_overview.dim_quality_warnings` gains `dominant_dim_mass` (one dim accounts for ≥70% of population p99-tail variance) and `negative_space` (gaussian-declared dims with empirical `p50==0`); each warning carries `type`, `dim_label`, `reason`, `advice`, with `evidence_value` and `threshold` on `dominant_dim_mass`.

#### Declarative compliance rules
- `Pattern.conformance_rules: list[ConformanceRule]` — declarative compliance rules in a safe AST predicate language (`and` / `or` / `not` over `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`); builder evaluates rules vectorised via PyArrow and persists violations alongside `rule_set_hash`. Helpers `compile_predicate` and `compute_rule_set_hash`.
- `GDSNavigator.find_conformance_violations(pattern_id, *, rule_id=None, severity_min="low", top_n=100)` — query primitive returning `{pattern_id, n_violations, violations, rules_evaluated, manifest, warnings, follow_up}`; detects rule-set hash mismatch as a warning.

### Fixed

#### Stress-test follow-up hardening
- `GDSNavigator.π7_attract_hub_and_stats` and `GDSNavigator.hub_score_history` no longer raise a broadcast error on anchor patterns whose geometry dim count exceeds the relation count (the synthetic conformance / motif dims tail at the end of `pattern.delta`). Hub-scoring code paths now slice the shape matrix to `len(pattern.edge_max)` before the per-relation weight multiply, matching the fix already applied to `π7_attract_hub`. `find_hubs` and `hub_history` are now callable on these patterns without `line_id_filter` as a workaround.
- `GDSNavigator.simulate_edge_removal` no longer scans the entity's full adjacency on hub entities — a new `max_edges_loaded` parameter (default 2000) truncates the candidate edge list before the sidecar IN-clause and the engine evaluation. Per-call latency on entities with tens of thousands of edges drops from indefinite to seconds.
- `GDSNavigator.select_minimal_joint_edge_removal` accepts a new `max_candidates` parameter (default 500) capping the greedy search input; results carry `n_candidates_seen`, `n_candidates_used`, and `candidates_truncated` so the agent sees when truncation occurred.
- `GDSNavigator.edge_potential` (backing `score_edge` / `score_motif`) now names the pattern type and the expected key shape in its `Entity not found` error — event patterns require event keys, anchor patterns require anchor keys, and the message points the agent at `search_entities` for discovery.
- `GDSNavigator.classify_detector_consensus` ranking is deterministic on the HMP-saturation case — `delta_norm` (pulled from the reliability flags attached by `combine_anomaly_pvalues`) is used as a final tiebreaker when the per-detector p-value vector is identical across entities. The previous behaviour returned an order determined by floating-point sort tie-break.
- `GDSNavigator.find_high_potential_motifs` and `GDSNavigator.score_motif` reject event `pattern_id` early with a `GDSNavigationError` that names the correct anchor companion. The previous behaviour silently returned an empty result after burning the full enumeration cost (the geometry's `primary_key` column carries event keys, but the adjacency index is keyed on entities; no seed ever passed the active-seed gate).

## [0.6.7] — 2026-05-10

### Added
- `sphere_overview` per-pattern entry gains an optional `dim_quality_warnings` block surfacing two silent build-time failure modes that break z-score / `delta_norm` semantics: **dead_dim** (`sigma_diag[i] < 1e-10` — zero variance, z-score undefined, the dim contributes nothing meaningful and silently dilutes other dims' signal) and **sparse_dim** (`dim_percentiles[d]['p50'] == 0 AND p99 > 0` — mostly-zero with rare nonzero, gaussian z-score assumption is wrong; Bregman divergence with poisson / bernoulli kind tag is the correct distance). Each warning carries `type`, `dim_label`, `reason` (with the offending value), and `advice` (concrete remediation: drop dim / fix data source for dead, switch to Bregman / split into is_active+value_when_active for sparse). Computed from cached pattern state (`sigma_diag`, `dim_percentiles`) — sub-millisecond, no storage scan; skipped silently when neither field is populated. Surfacing through `sphere_overview` makes both classes auditable on a fresh sphere read instead of being buried in the calibration log.

### Added
- `GDSNavigator.generate_sar_rationale(chain_id, pattern_id, *, anchor_pattern_id, evidence=None, regulatory_template="FinCEN SAR")` — template-based composition (no LLM call) of a SAR-ready narrative from R9 evidence on a single chain. Takes the structured per-step output of `investigate_chain` (or runs the loop server-side when `evidence=None`) and produces a 3-5 paragraph draft covering chain identification + typology, per-hop trace evidence, boundary extension candidates, chain-shape corroboration, and aggregated strength + recommended action. Returns `sar_narrative` (paragraph-separated string), `evidence_anchors` (structured pointers per narrative claim — null when the corresponding R9 surface failed), `regulatory_template_hint` (passthrough), and `confidence` (`high` / `moderate` / `low` derived from `investigation_strength` + evidence completeness; `high` requires `strong` strength AND all 5 R9 surfaces ok). Closes the investigation→SAR pipeline: investigators now get a starting draft instead of a blank page after `investigate_chain` completes. Honesty discipline: language is "evidence indicates" / "the per-hop trace shows" — never "confirms" — and the narrative is positioned as a starting point for the investigator's draft, not a final verdict.

### Added
- New documentation page `external-chains-as-anchor-line.md` covering the workflow for ingesting chains discovered outside hypertopos (SAR typology engines, ERP supply-chain workflows, EHR clinical pathways, customer-journey platforms). Documents the chain anchor line schema convention — required key + chain features, optional `chain_keys` column (comma-joined member primary_keys in chain order) that unlocks the chain-coherent investigative loop on externally-curated chains. Walks through declaration in `sphere.yaml`, lists every standard anchor primitive that works out-of-the-box (`find_anomalies`, drift / trajectory / population-compare / regime-change, density gaps, lead-lag, trace_root_cause, explain_anomaly, similar entities, decompose_drift), and the R9 family that the convention column unlocks (`find_chains_with_coherent_anomaly`, `anomaly_propagation_in_chain`, `classify_chain_typology`, `extend_chain`, `chain_investigation_summary`, `investigate_chain`). No code change — pure documentation of an already-supported integration path. `concepts.md` chain-interpretation section gets a cross-link to the new page.

### Added
- `GDSNavigator.investigate_chain(chain_id, pattern_id, *, anchor_pattern_id, extension_max_results=20)` — one-shot orchestrator that runs the full R9 investigative loop on a single chain and aggregates the per-step outputs into a single SAR-ready report. Composes `anomaly_propagation_in_chain` (per-hop trace), `classify_chain_typology` (five-axis tag), a chain-pattern geometry lookup for the chain_id (`shape_anomaly`), and `extend_chain` in both forward and backward directions (extension candidates). Each per-step output is wrapped in `{ok: True, data: ...}` or `{ok: False, error: ...}` so a partial failure does not abort the whole report. Summary derives `investigation_strength` from four chain-composition signals (run length >= 3, typology position not "no-run", forward extension has an anomalous candidate, backward extension has an anomalous candidate); chain-shape anomaly is reported as evidence but intentionally not scored so the R9 sweet spot (composition anomalous, chain shape normal) reaches `strong` without needing chain-shape agreement. Buckets: score >= 3 → `strong` → `escalate to SAR`; score 2 → `moderate` → `continue investigation`; score 0-1 → `weak` → `false-positive candidate`. Rationale concatenates firing signals as a single paragraph. Saves the round-trip cost of running the four R9 primitives sequentially when the investigator already knows which chain to drill into; the granular tools remain available when per-step control is needed.

### Added
- `GDSNavigator.π5_attract_anomaly(..., dimension_weights=None)` — optional `{dim_name: float}` mapping that multiplies each dimension's contribution to the rank score before computing `delta_norm`. Default `None` leaves behaviour unchanged. Missing dims default to weight `1.0`; explicit `0.0` silences a dim. Requires `metric in ('L2', 'Linf')` — Bregman divergence is precomputed per-row and cannot be reweighted post-hoc. When supplied, forces the in-process scan path (the >500K-row Lance subprocess fast path relies on the stored `delta_norm`). Connects stratified correlation-gate verdicts to runtime ranking — discount NOISE-classified dims via weight `0.0`, down-weight HEAVY-TAIL dims via `0.5`. Validates dim names against `pattern.dim_labels`, rejects negative / non-finite / non-numeric weights with `ValueError`.
- `GDSNavigator.chain_investigation_summary(chain_pattern_id, anchor_pattern_id, *, min_hops=2, max_runs=10000)` — pre-investigation triage diagnostic for a chain pattern. Aggregates one `find_chains_with_coherent_anomaly` sweep + a chain-pattern geometry scan into population-level metrics: `n_chains_total`, `n_chains_with_coherent_anomaly_run`, `coherent_run_rate`, `n_chains_with_shape_anomaly`, `shape_anomaly_rate`, `cross_pattern_overlap` (`n_both`, `n_coherent_only`, `n_shape_only`, `jaccard`), `top_dims_in_coherent_runs` (top 10), `run_length_distribution` (`min`, `p50`, `p90`, `max`, `mean`), `recommended_min_hops`. Cost is one coherent-anomaly sweep — the same call an investigator would issue as the first triage step, with the aggregates surfaced for free. Lets agents decide whether to commit budget to the chain-coherent investigative loop on a sphere before drilling into individual chains.

## [0.6.6] — 2026-05-09

### Added
- `GDSNavigator.sphere_overview()` per-pattern entry gains an optional `theta_sensitivity_summary` block when the pattern's latest calibration epoch has a populated `theta_sensitivity` field. The block carries `stable_band_from`, `stable_band_to`, `stable_band_length`, `n_cliffs`, and `theta_at_p95` so an agent reading the population overview sees at a glance whether the production threshold sits in a smooth region or near a heavy-tail jump. Calibration epochs from older builds that lack the underlying field continue to render `sphere_overview` entries as before — the block is silently skipped. Cost: one calibration-history JSON read plus an O(P=10) derivation per pattern, sub-millisecond per call.
- `GDSNavigator.theta_sensitivity(pattern_id, version=None)` — new navigator method surfacing the calibration-quality diagnostic. Reads the populated `theta_sensitivity` field on the resolved `CalibrationFit` and returns a `ThetaSensitivityReport` with the per-percentile sweep plus derived `stable_band` (`from`, `to`, `length`) and `cliffs[]` (`from`, `to`, `ratio`). Resolves the latest epoch on disk by default; `version=N` selects an explicit historical epoch. Raises `ValueError` when the calibration epoch lacks the diagnostic field. Companion to the existing `compare_calibrations` and `decompose_drift` calibration-history surfaces.
- Calibration epoch JSON gains a `theta_sensitivity` field — per-percentile sweep over the population's `delta_norm` distribution at p90..p99. For each percentile reports `theta_mean` (the threshold value at that percentile), `anomaly_count_mean` (entities at or above that threshold), and `anomaly_rate`. Lets investigators inspect how stable the chosen `anomaly_percentile` is to perturbation: contiguous percentiles where adjacent-pair anomaly_count ratios stay below ~1.30 form the safe recalibration zone; ratios at or above ~1.50 mark cliffs where moving the threshold changes population materially. Populated automatically at build time on every pattern; field is `None` on calibration epochs from prior builds and surfaces on the next pattern rebuild. Reads back via `read_calibration_fit` on `CalibrationFit.theta_sensitivity`.

## [0.6.5] — 2026-05-08

### Added
- Chain anchor pattern's per-chain feature surface gains two new derived columns: `cross_bank_count` (number of distinct banks the chain transits across all from_bank / to_bank pairs — textbook AML jurisdictional layering indicator) and `amount_monotone_decreasing` (boolean, true when amounts strictly decrease at every hop — textbook structuring pattern). Both flow into the chain anchor pattern's polygon delta and surface in `find_anomalies` / chain-coherent loop tools / `classify_chain_typology` `dominant_top_dim`. Populated automatically when the source event line declares `from_bank` / `to_bank` columns (or `source_bank` / `destination_bank`); chains built without bank columns carry `cross_bank_count = 0` (additive, no breakage). Effect is gated on next chain pattern rebuild — existing spheres carry the prior feature set until rebuild.

### Fixed
- `extract_chains` post-merge dedup gains a strict-prefix subsumption pass: chains whose entity sequence is a strict ordered prefix of another chain's entity sequence are dropped (the longer chain traverses every entity the shorter one does plus more, so the shorter is investigatively redundant). Effect is gated on next chain pattern rebuild — existing spheres carry the prior chain count until rebuild. Chain-coherent loop tools see slightly fewer redundant chain entries on rebuilt spheres.

## [0.6.4] — 2026-05-07

### Fixed
- Parallel chain extraction (`extract_chains` via `ProcessPoolExecutor`) emitted `chain_id`s that collided across workers. Each worker built its chain list locally and used `len(chains)` as the id counter, so the merged output assigned the same `CHAIN-XXXXXX` value to up to N_workers distinct chains. Chains in the points table were then no longer keyed uniquely by `primary_key`. Post-merge dedup now reassigns `chain_id` from the global merged index, restoring uniqueness. **Spheres built before this fix carry colliding chain_ids; rebuild the chain pattern (or the full sphere) to restore unique ids.** A defensive raise in `anomaly_propagation_in_chain` surfaces the ambiguity on legacy spheres so investigations don't silently return the wrong chain's hops. Chain extraction now also asserts post-dedup chain_id uniqueness as a build-time safety net so any future regression in the merge logic fails loudly instead of silently corrupting the chain pattern points table.

### Added
- `find_motif_by_hops` gains an `anomaly_seed_filter: bool = False` parameter. When `True`, the BFS starting frontier is intersected with the anomaly subset of the resolved anchor companion pattern (entities with `is_anomaly=True`) — replaces the implicit "all keys" frontier when `seed_keys=None`, intersects with the explicit list otherwise. On large populations with a small anomaly fraction, the BFS traversal is collapsed proportionally to the prune ratio, reducing wall-clock for population-sweep motif queries that are bounded on frontier size. Result dict gains `seed_filter_summary` with `requested`, `anomaly`, and `filtered` counts so callers can verify the prune. Raises `GDSNavigationError` when no anchor companion pattern is configured.
- `classify_chain_typology(chain_id, pattern_id, *, anchor_pattern_id)` — new navigator inspector that wraps `anomaly_propagation_in_chain` and labels the chain along five operational axes: `shape` (monotone-rising / monotone-falling / peak-in-middle / peak-at-start / peak-at-end / flat / single-hop / no-anomalous-run), `peak_position` within the run, `position_in_chain` (leading / transit / terminal / full-chain), `extension_signals` (`forward` / `backward` booleans indicating whether the next-hop or pre-run hop is in an elevated rank band), `pre_run_rank_bucket` and `breakpoint_rank_bucket`, plus the chain-wide `dominant_top_dim`. Returns the chain's longest-run summary alongside the typology block.
- `extend_chain(chain_id, pattern_id, *, anchor_pattern_id, direction="forward"|"backward", max_results=20)` — new navigator inspector that suggests extension entities at the boundary of a chain's anomalous run. Forward looks at entities that follow the run-end key in OTHER chains in the same chain pattern (via the chain reverse index); backward looks at entities that precede the run-start key. Candidates are returned with `is_anomaly`, `delta_norm`, `delta_rank_pct`, source chain ids, and ranked by `(is_anomaly DESC, delta_norm DESC, n_source_chains DESC)`.
- `anomaly_propagation_in_chain(chain_id, pattern_id, *, anchor_pattern_id)` — new navigator inspector primitive complementary to `find_chains_with_coherent_anomaly`. Takes a single chain id and returns the chain's hop-by-hop anomaly trace: for each entity in the chain's keys sequence, returns `is_anomaly`, `delta_norm`, the dominant delta dimension (sigma-normalised argmax over `|delta|`) when anomalous, and `delta_rank_pct`. Plus a summary with `n_hops`, `n_anomalous`, `max_run_length_same_top_dim`, and `dominant_top_dim`. Use after a population sweep flags a chain to drill into how the anomaly accumulates and where it breaks. Pure query-side, additive within sphere format 2.4.

### Changed
- `find_motif_by_hops` no longer routes seeded queries (`seed_keys=<list>`) through a scoped per-call edge-table read path. Scoped routing was uniformly slower than the cached global `AdjacencyIndex` + per-call seed filter at every measured seed count from 10 up; the BFS enumerator already filtered the starting frontier by `seed_keys`, so the scoped branch added cost without benefit. After the change all queries — seeded or unseeded — use the cached global adjacency. The cold first call still pays the full O(E) adjacency build; warm calls amortise to BFS-only cost. The internal `_build_scoped_adjacency_for_motif` helper is removed.

### Added
- `find_chains_with_coherent_anomaly(pattern_id, *, anchor_pattern_id, min_hops=3, max_results=100)` — new navigator primitive that surfaces chains where `>=min_hops` strictly consecutive entity-anchor positions are individually flagged as `is_anomaly=True` AND share the same dominant delta dimension (argmax over sigma-normalised |delta|). Returns ranked runs sorted by `(run_length DESC, max_delta_norm DESC)` with `chain_id`, `run_start_idx`, `run_length`, `top_dim`, `run_keys`, `max_delta_norm` per match plus diagnostics (`n_chains_total`, `n_anomaly_entities`, `elapsed_ms`). Pure query-side, additive within sphere format 2.4. Distinct from `find_anomalies` on a chain pattern, which scores chain-level features (hop count, time span, amount decay); the new primitive scores chain composition — coherent cascades through structurally-similar anomalous entities. Useful when chain shape looks normal but consecutive hops go through entities that all share the same anomaly driver.

## [0.6.3] — 2026-05-06

### Changed
- Build-time compute paths for `edge_dim_aggregations:` and the edge feature
  catalog (`compute_pair_edge_count`, `compute_time_since_pair_last_edge`,
  `compute_pair_amount_zscore`, `parse_timestamps_to_epoch`) optimised —
  pure-Python per-row loops replaced with vectorised pyarrow / numpy
  paths. The `_p95` aggregate now returns the exact 95-th percentile
  element (previously a t-digest sketch approximation).

### Added
- `aggregates:` per-source-dim subset selector on `edge_dim_aggregations:`.
  Source dims can now declare which aggregates to emit (`mean` / `max` /
  `std` / `p95` / `count_above_threshold`) instead of always emitting all
  five. The existing `dims: [a, b]` list form continues to expand to all
  five aggregates per dim; the new mapping form
  `dims: {a: [count_above_threshold], b: [mean, max]}` selects subsets.
  Polygon-dim layout follows source-dim insertion order × canonical
  aggregate order, so reordering the user-supplied aggregate list does
  not flip `schema_hash`. Spheres rebuilt with subset selectors carry an
  `aggregates_per_dim` block under each pattern's `edge_dim_aggregations:`
  in `sphere.json`; pre-selector spheres keep emitting the full five-tuple
  on read.
- Per-source-dim `_count_above_threshold` thresholds are now persisted into
  the calibration epoch JSON under `_gds_meta/calibration_history/<pid>/v={N}.json`
  alongside `mu` / `sigma` / `theta`. Each calibration epoch carries the
  thresholds used during its build, so re-builds are deterministic and
  multi-epoch comparisons see threshold drift. `compare_calibrations`
  output gains a new `edge_dim_threshold_drift` field — a per-source-dim
  `{from, to, delta}` map populated when both compared epochs declared
  `edge_dim_aggregations:`. Legacy epochs (built before this release)
  deserialize with `edge_dim_thresholds=None` and silently omit the
  drift block.
- `edge_dim_aggregations:` on composite anchors with `len(key_cols) > 2`
  (tripartite and beyond). The previous k=2 (pair) restriction is lifted —
  composite anchors registered via `composite_lines:` with three or more
  key columns now compose with `edge_dim_aggregations:` the same way k=2
  composite anchors did. The anchor primary key is constructed positionally
  from `key_cols` joined by the `composite_lines.<id>.separator`, matching
  the convention used to register the composite line itself. Aggregated
  per-anchor `<source_dim>_mean` / `<source_dim>_max` columns surface in
  every existing anchor-pattern primitive (`find_anomalies`,
  `explain_anomaly`, `find_clusters`, `find_calibration_influencers`,
  `decompose_drift`) via the same `dimension_kinds` source-of-truth as the
  earlier `single` / `pair` / `chain` regimes. No new tools, no MCP API
  change, no sphere format bump.
- Three new aggregates on `edge_dim_aggregations:` — `_std`
  (per-anchor standard deviation of the source dim across the anchor's
  edges), `_p95` (95th percentile, tail-mass tracker without `_max`'s
  extreme-outlier sensitivity), and `_count_above_threshold` (number
  of edges where the source dim crosses a per-dim threshold). Extend
  the prior `_mean` / `_max` set on every anchor regime (single, pair,
  k>2, chain). Surface automatically alongside `_mean` / `_max` on
  every anchor primitive (`find_anomalies`, `explain_anomaly`,
  `find_clusters`, `find_calibration_influencers`, `decompose_drift`).
  Per-dim threshold for `_count_above_threshold` defaults to the
  population p95 of the source dim; user can override by passing a
  `thresholds` dict keyed by source dim name. `_std` adds variance
  signal that `_mean` / `_max` don't capture; `_p95` separates
  tail-mass from extreme outliers; `_count_above_threshold` flags
  anchors with anomalous concentration of high-value edges. No new
  YAML keywords, no new tools, no sphere format bump.

## [0.6.2] — 2026-05-05

### Added
- Chain-anchor `edge_dim_aggregations:` regime — third `anchor_kind`
  alongside `single` (account-style) and `pair` (k=2 composite). Chain
  anchor patterns auto-emitted from `chain_lines:` config can declare
  `edge_dim_aggregations: { from: <event_pid>, dims: [...] }` directly
  in the `chain_lines.<id>:` block; the builder forwards it to the
  auto-emitted `<id>_pattern`. Aggregation reads the source event
  pattern's edge_features sidecar and bakes per-chain `<source_dim>_mean`
  and `<source_dim>_max` columns into the chain anchor polygon's
  `shape_snapshot`. New `_chain_lines: set[str]` registry on `GDSBuilder`
  for regime detection. `aggregate_edge_dims_for_anchor` gains a
  `chain_events: list[str] | None` keyword for the chain regime
  (comma-joined event_keys per anchor; engine performs explosion + join
  + groupby). Cross-block parse-time validation enforces
  `chain_lines.<id>.event_line == src_pat.edge_table.event_line`;
  zero-extracted-chains with declared aggregations raise loudly at
  build dispatch. Sphere format stays at 2.4 — `chain_events` property
  column on the chain anchor line is the existing membership source.
  Aggregated dims surface automatically in every existing anchor-pattern
  primitive (`find_anomalies`, `explain_anomaly`, `find_clusters`,
  `find_calibration_influencers`, `decompose_drift`) via
  `dimension_kinds` source-of-truth — no new tools, no MCP API change.

### Added
- `AdjacencyIndex.neighbors_out_window` and
  `AdjacencyIndex.neighbors_in_window` — column-selective neighbor
  accessors. Return a per-column dict (`{col: list[Any]}`) for any subset
  of `("to_key" / "from_key", "timestamp", "amount", "event_key")` with
  an optional `ts_min` window predicate applied at the pyarrow C++ layer
  before materialization. Useful for callers that need only key + timestamp
  without paying the cost of materializing `amount` / `event_key`.
- `AdjacencyIndex.distinct_neighbors_out`,
  `AdjacencyIndex.distinct_neighbors_in`, and
  `AdjacencyIndex.max_amount_out_excl_self` — O(1) public accessors over
  precomputed per-`from_key` aggregates (distinct-out-neighbor count,
  distinct-in-neighbor count, max-amount-excluding-self). Self-loop
  semantics preserved; null amounts skipped.

### Fixed
- Cold-call latency on motif enumeration tools (`find_high_potential_motifs`,
  `score_motif`, `find_motif_by_hops`) reduced across `bipartite_burst`,
  `fan_out`, `fan_in`, and `cycle_2` motif types. No public API change.
- `AdjacencyIndex.pair_counts()` first-call cost on hub-graph workloads
  reduced. The aggregate is now precomputed at adjacency build time and
  surfaced in O(1) on first call. Self-loop semantics preserved
  (`pair_counts` includes self-pairs; distinct counts skip them).

### Changed
- `AdjacencyIndex` cold load path is faster on edge-table-backed
  patterns; motif tools' first-call latency drops materially. No format
  bump, no persisted artifact, no public API change — every public method
  (`neighbors_out`, `neighbors_in`, `degree_out`, `degree_in`, `all_nodes`,
  `all_edges`, `node_count`, `edge_count`, `pair_counts`) preserves its
  signature and observable behavior.

## [0.6.1] — 2026-05-01

### Fixed
- `find_motif_by_hops` `score=True` now functional on event patterns. The
  scoring branch was previously dead code (gated on
  `pattern_type != "event"`, but the function rejects non-event patterns
  upstream), so passing `score=True` had no observable effect since the
  declarative motif API shipped. Scoring now resolves the event pattern's
  anchor companion via `_resolve_anchor_pattern_for_scoring` and uses the
  anchor pattern's per-entity geometry (entity-level deltas) rather than
  the event pattern's per-transaction polygons. Each scored motif carries
  a new `anchor_pattern_id` provenance field. Raises `GDSNavigationError`
  when no anchor companion is configured for the event pattern; per-motif
  scoring failures (endpoint missing in anchor geometry) leave the motif
  unscored without breaking the call. Output ordering: descending on
  score, unscored motifs at tail.
- Motif scoring kernel (`_score_motif_from_edges`, `_lean_score_motif`)
  now event-aware via opt-in `event_keys` + `event_pattern_id` arguments.
  Previously `edge_potential = delta_distance × (1/effective_pair_count)`
  depended only on the node pair, so multiple distinct transactions
  between the same `(from, to)` accounts produced identical motif scores
  — turning ranked `find_motif_by_hops(score=True)` output into "5–10
  distinct ranks with the rest as ties" whenever motifs shared a node
  sequence. With the new arguments the kernel batch-reads the event
  pattern's per-event polygons and multiplies each edge's potential by
  `(1 + ||event_polygon[event_key]||)`; events sitting at the population
  centroid (norm 0) keep the legacy score, anomalous events boost above.
  Per-edge `score_breakdown` carries a new `event_factor` field when
  event-aware. Backward compatible: legacy callers (`score_motif`,
  `find_high_potential_motifs`, any external caller of
  `_score_motif_from_edges` without event_keys) see unchanged scores.
  `find_motif_by_hops` passes both arguments automatically and hoists
  both anchor and event delta reads out of the per-motif loop (one
  batched scan per pattern across the union of all motifs, vs.
  per-motif reads pre-hoist) so the per-call I/O cost stays O(1) in
  motif count.
- `Pattern.dim_labels` and `Pattern.delta_dim()` now include aggregated
  edge-dim names from `edge_dim_aggregations` so callers don't see a
  stale 33-element view of a 37-element delta. Previously the property
  returned only `relations + event_dimensions + prop_columns` — but the
  builder appends `(source_dim)_mean` and `(source_dim)_max` columns to
  the polygon delta when an anchor pattern declares
  `edge_dim_aggregations: { from: <event>, dims: [...] }`. The mismatch
  produced two visible failure classes on patterns with aggregations
  declared:
    - `anomaly_summary` raised `operands could not be broadcast together
      with shapes (33,) (37,) (33,)` because `dim_sq_totals` was sized
      from `len(pattern.dim_labels)` (33) but cluster delta vectors had
      length 37 (33 base + 4 aggregated).
    - `find_clusters.dim_profile`, `find_anomalies.anomaly_dimensions`,
      `explain_anomaly` top-dim labels, `contrast_populations`
      effect-size labels and other label-resolving paths returned
      `dim_35` / `dim_36` placeholders for the aggregated dims instead
      of the build-time names (`find_motif_structuring_mean`,
      `pair_edge_count_max`, etc.).
  Both classes are resolved by the model property fix; the aggregated
  name convention follows `f"{source_dim}_{agg}"` for each
  `agg ∈ AGGREGATE_NAMES = ("mean", "max")` in
  `engine/edge_features.py`. `parse_edge_dim_aggregations` now requires
  `dims` to be a non-empty list — the previous `dims=None` shorthand
  ("aggregate every applicable dim") was a latent build-vs-runtime
  inconsistency: builder appended columns from the source event
  pattern, but the runtime model had no record of which columns. Users
  must now declare dims explicitly. Backward compatible for spheres
  with explicit `dims:` on disk (AML HI-small etc.); spheres that
  somehow built with `dims=None` fail loudly at re-load and need a
  rebuild with explicit declaration. Closes the F1.d follow-up
  (human-readable aggregated-dim labels) and the F6 stress-test
  wildcard for the broadcast regression.

### Changed
- `find_motif_by_hops` engine: replaced the recursive DFS walk with an
  iterative BFS-by-level enumerator. Lifts the per-call hop count cap
  from 6 to 8 (matches the `chain_k` motif vocabulary). Adds an optional
  `time_window_hours: float | None` top-level parameter for total-
  chain-span cap (independent semantic from per-hop
  `time_delta_max_hours`; both apply when both are set). API surface
  preserved: same enumeration output shape, all existing tests pass.
  Cites the Paranjape-Benson-Leskovec delta-temporal motif algorithm as
  prior art for sliding-window enumeration; the implementation is a
  pragmatic level-synchronous BFS (PBL counts fixed-template motifs
  without per-edge predicates, which does not fit the enumerate-with-
  arbitrary-predicates use case).

### Added
- `HopPredicate.require_anomalous_entity: bool` — new per-hop predicate
  field for the declarative motif API (closes the X1 predicate set
  alongside `amount_ratio_to_prev`). When `True` on hop `i`, the
  destination entity (`nodes[i+1]` of the resulting motif) must satisfy
  `is_anomaly=True` in the resolved anchor companion pattern's geometry.
  Multiple hops can set this independently; constraints AND across hops.
  Filter runs at the navigator layer after BFS enumeration but before
  scoring (saves scoring work on motifs that get dropped). The seed
  (`nodes[0]`) is never checked — pre-filter `seed_keys` upfront if seed
  coverage is needed. `max_results` applies AFTER the filter, so a
  restrictive filter can return fewer than `max_results` motifs. Raises
  `GDSNavigationError` when no anchor companion is configured for the
  queried event pattern, or when the anchor pattern has no `is_anomaly`
  column (calibration must run first). Default `False` is a no-op,
  preserving prior behavior. Reuses the same anchor-companion lookup
  wired by F5; no sphere format change, no rebuild required.
- `HopPredicate.amount_ratio_to_prev: float | None` — new per-hop predicate
  field for the declarative motif API. When set on hop `i ≥ 1`, the engine
  rejects a candidate edge unless `current_amount / prev_hop_amount ≤
  ratio`. Bounds `0 < ratio ≤ 1.0` enforced at validation; `hops[0]` must
  leave the field None (no previous amount). Edges where either amount is
  ≤ 0 are silently skipped, matching the existing `find_motif_structuring`
  convention. Use case: declarative structuring / layering chain detection
  without baking absolute amount thresholds (`amount_min` / `amount_max`).
- Anchor-pattern aggregation of edge-derived dimensions (S1 extension).
  An anchor pattern can declare `edge_dim_aggregations: { from: <event_pid>, dims: [...] }`
  in its YAML stanza; the builder reads the event pattern's edge_features
  sidecar and bakes per-anchor-entity `_mean` and `_max` aggregates of
  the named source dims into the anchor polygon's `shape_snapshot`.
  Aggregation handles two anchor regimes — `single` (anchor PK matches
  edge `from_key` OR `to_key`, account-style) and `pair` (anchor PK
  encoded as `<from>__<to>`, composite-pair-style); chain anchors
  raise `NotImplementedError` and ship in 0.6.2. New
  `EdgeDimAggregationsConfig` builder dataclass, `EdgeDimAggregationsRef`
  runtime dataclass on `Pattern`, and
  `engine.edge_features.aggregate_edge_dims_for_anchor` engine entry
  point. Builder switches to a two-phase build (event patterns first,
  anchor patterns second) when any pattern declares
  `edge_dim_aggregations` so the sidecar dependency holds without a
  topo sort. New aggregated dims surface automatically in every
  existing anchor-pattern primitive (`find_anomalies`,
  `explain_anomaly`, `find_clusters`, `find_calibration_influencers`,
  `decompose_drift`) via the `dimension_kinds` source-of-truth — no
  new tools, no MCP API change. Sphere format stays at 2.4; spheres
  without the new YAML block are byte-identical.

## [0.6.0] — 2026-04-30

### Theme

**Architecture-emergent analytics, properly enabled.** Calibration is
now versioned (multi-epoch retention, sphere format 2.4) and four
patent-aligned analytics ride on that versioning:
`compare_calibrations`, intrinsic / extrinsic drift decomposition,
hidden-influencer matrix, cross-pattern temporal lead-lag. Companion
features round out the release: edge-derived build-time dimensions,
joint density gap detection, and a declarative `HopPredicate` motif
API as power-user escape hatch from the closed-vocab `find_motif`
registry.

### Added
- `GDSNavigator.find_motif_by_hops(pattern_id, hops, *, seed_keys, max_results, score)`
  — declarative motif API. Caller passes a list of `HopPredicate`s
  describing per-hop constraints (`amount_min` / `amount_max` /
  `time_delta_max_hours` / `direction` (`"forward"` / `"reverse"` /
  `"any"`) / `edge_dim_predicates` referencing the per-edge sidecar from
  the edge_dimensions YAML block) and the navigator walks the edge table
  for matching chains of length 1..6. Power-user escape hatch from the
  closed-vocab `find_motif` registry — composes cleanly with
  `pair_edge_count`, `position_in_chain`, `time_since_pair_last_edge`,
  `pair_amount_zscore`, `find_motif_structuring`. New `HopPredicate`
  frozen dataclass in `hypertopos.model.sphere` (re-exported from the
  package root). Seeded queries (`seed_keys` provided) build a scoped
  `AdjacencyIndex` via Lance BTREE-pushdown reads expanding the BFS
  frontier hop-by-hop — the visited subgraph alone is materialised, not
  the full edge table. Unseeded full enumeration uses the cached global
  `AdjacencyIndex` so repeat calls are O(1) per node lookup. Bounded
  MVP — `amount_ratio_to_prev` and
  `require_anomalous_entity` predicates plus k>6 land in a follow-up.
  Scoring path uses the existing `_score_motif_from_edges` infrastructure
  on anchor patterns; on event patterns the score is silently skipped
  (anchor-companion scoring is the follow-up wiring); `score` defaults to
  `False` since only event patterns are accepted today. Direction-aware
  temporal monotonicity: `direction="forward"` enforces strict-increasing
  timestamps, `direction="reverse"` enforces strict-decreasing
  (causal-predecessor chain), `direction="any"` drops the monotonic
  constraint and treats the time window as `|Δt|`. `hops[0].time_delta_max_hours`
  must be `None` (rejected at validation — first hop has no previous
  timestamp); `time_delta_max_hours` on subsequent hops must be positive.
  Sidecar lookup is materialised lazily as `ek→idx` plus per-dim parallel
  lists (rather than a 5M-row dict-of-dicts) — avoids the per-event
  nested-dict allocation when `edge_dim_predicates` are used.
- `GDSNavigator.find_density_gaps(pattern_id, *, top_n, dim_pairs, bins, alpha, r_min, r_max, sample_size)`
  — joint density gap detection via probability integral transform plus
  independence null on dim pairs. For each pattern dim that survives the
  usability check (≥30 finite values, σ > 0, more than 2 unique levels)
  the empirical CDF transform produces a uniform marginal on `[0, 1]`.
  Pairs whose Pearson `|r|` lands inside the configurable
  `[r_min, r_max]` window become the test set. Per pair the `bins x bins`
  joint histogram is compared against the uniform-independence
  expectation `N / bins²` via per-cell chi² residual; only
  under-populated cells are kept and Benjamini-Hochberg correction is
  applied across all under-populated cells to control FDR. Each flagged
  cell maps back to a named delta-space (z-score) range via
  `ECDFEntry.inverse` — values live on geometry-delta units; raw
  property unit mapping is deferred follow-up.
  Returns a dict with `gaps` (sorted by `expected/observed` ratio
  descending), `excluded_dims` reporting (with reason), and
  `n_pairs_tested`. Anchor patterns only (event patterns are not the
  primary use case); patterns with `< 100` entities raise. Geometry is
  read with column projection narrowed to `["delta"]` and Lance-side
  random sampling via `sample_size` (default `100,000`; pass `0` to
  scan the full population). Delta matrix is built via
  `combine_chunks().values.to_numpy()` reshape rather than
  `to_pylist()` + `np.array`. New `engine.density_gaps` module exposes
  `ECDFEntry`, `compute_density_gaps_for_pair`, `is_usable_for_gap`,
  and `select_pairs_by_corr` as building blocks.
- Edge-derived dimensions on event patterns:
  `engine.edge_features` exposes five build-time per-edge dim functions
  (`pair_edge_count`, `position_in_chain` with default `min_position=5`
  and parse-time reject below 3, `time_since_pair_last_edge`,
  `pair_amount_zscore` LOW_VAR pairs only, `find_motif_structuring`)
  plus an orchestrator `compute_all_edge_dims(edges, config)` that runs
  every dim listed in a config dict and returns a single Arrow table
  keyed by `event_key`. `EDGE_DIM_KINDS` table publishes the kind tag
  (poisson / gaussian / bernoulli) for each dim so builder calibration
  can pick the right normaliser. New `engine.structuring` module hosts
  the structuring motif enumerator (single-seed +
  all-seeds sweep returning the set of `event_key`s in any motif) for
  the build-time `find_motif_structuring` dim; the navigator runtime
  `_enumerate_structuring` path is unchanged. New
  `EDGE_FEATURES_SCHEMA` constant in `storage._schemas` describes the
  per-edge sidecar Lance dataset layout. New
  `GDSReader.read_edge_features(pattern_id)` reads
  `_gds_meta/edge_features/{pid}/data.lance` (empty table when sidecar
  absent), forward-compatible with the
  HopPredicate.edge_dim_predicates query API planned in a future
  release. YAML surface for the new dims: `patterns.<pid>.edge_dimensions`
  list of either bare dim names or single-key dicts with overrides;
  parsed into `EdgeDimensionsConfig` and attached to `PatternMapping`.
  Builder integration baked the per-event dim values into the event
  polygon `shape_snapshot` (one extra dim per declared entry) and
  extends `dimension_kinds` so downstream Bregman / theta-norm
  calibration treats the new dims natively. Sidecar Lance dataset at
  `_gds_meta/edge_features/{pid}/data.lance` is written alongside,
  forward-compatible with the planned HopPredicate.edge_dim_predicates
  query API. Anchor patterns and event patterns without edge_dimensions
  are byte-identical to builds without the new YAML block.
- Multi-epoch calibration retention: builder writes per-pattern historical
  fits to `_gds_meta/calibration_history/{pid}/v={N}.json` on every full
  build; sphere format bumps to 2.4 with `pattern.calibration_epoch`,
  `pattern.schema_hash`, and root `calibration_history_policy`. New reader
  API: `read_calibration_fit`, `list_calibration_versions`,
  `read_calibration_history_policy`. New types: `CalibrationFit`,
  `CalibrationNotFoundError`. GC keeps the most-recent K epochs (default 5).
  Schema drift wipes history; 2.3 spheres open read-only and migrate on
  first 0.6.0 build.
- `GDSNavigator.compare_calibrations(pattern_id, v_from, v_to, top_n, verbose)`
  — per-dimension μ/σ/θ drift between two calibration epochs, with auto-resolve
  defaults and aggregate RMS scalar. New types: `CalibrationDriftReport`,
  `DimensionDrift`. Diagnostic for inspecting calibration shifts after a
  builder rebuild; reads via the multi-epoch retention API added earlier in
  this release.
- `GDSNavigator.decompose_drift(entity_key, pattern_id, v_from, v_to, timestamp_from, timestamp_to, top_n, verbose)`
  — per-entity intrinsic vs extrinsic decomposition of geometric drift between
  two temporal slices viewed across two calibration epochs. Returns
  `IntrinsicExtrinsicReport` with aggregate L2 displacements, sum-of-squares
  `intrinsic_fraction` bounded `[0, 1]`, and ranked per-dimension breakdown.
  New types: `IntrinsicExtrinsicReport`, `DimensionDecomposition`.
- `find_drifting_entities` per-entity dict gains 3 additive scalar fields:
  `intrinsic_displacement`, `extrinsic_displacement`, `intrinsic_fraction`.
  Auto-defaults to (oldest retained, current) calibration epochs;
  resolves to `null` per-entity when decomposition isn't computable.
- `GDSNavigator.find_calibration_influencers(pattern_id, top_n, classify, high_threshold_pct, sample_size, verbose)`
  — per-entity exact leave-one-out impact on coordinate system calibration with
  4-cell classification matrix (hidden / distorter / standard_anomaly / normal).
  Math uses exact leave-one-out via rolling Σs/Σs² (NOT first-order μ-only
  approximation — that approximation makes the "hidden influencer" cell empty
  by construction; corrected during brainstorm). New types:
  `DimensionContribution`, `InfluenceEntry`, `InfluenceReport`. `verbose=True`
  attaches `cascading_flip_count` per entry — count of OTHER entities
  flipping is_anomaly after this entity's removal.
- `GDSNavigator.find_group_influence(pattern_id, groups)` — caller-
  supplied leave-set-out impact with `reinforcing_factor = total_impact_set
  / Σ_individuals`. Reinforcing > 1 detects coordinated population-shift
  attacks (collusion rings, duplicate-record contamination); < 1 detects
  canceling sets. New type: `GroupInfluenceReport`.
- `find_anomalies` MCP per-entity dict gains 2 additive scalar fields:
  `total_impact`, `classification`. Hooked at the MCP-layer polygon→dict
  conversion via `_attach_influence_fields_to_anomaly_entries`; resolves to
  `null` per-entity when pattern is event-type, N<2, or storage backend
  lacks shape-reconstruction prerequisites.
- `GDSNavigator.find_lead_lag(pattern_a, pattern_b, *, timestamp_from, timestamp_to, cohort, min_epochs, max_lag, fdr_alpha, fdr_method, verbose, entity_key)`
  — cross-pattern temporal lead-lag in population-relative coordinates.
  Population-aggregated centroid drift cross-correlation (primary signal,
  patent-line) plus mean step-magnitude volatility confirmation series with
  `agreement` label. Per-dim D_A × D_B matrix with BH or Storey FDR over
  Bonferroni-over-lag-adjusted p-values (`top_dim_pairs` ranked by ascending
  q-value, full matrix in `per_dim_pairs` when `verbose=True`). Per-entity
  drill-down via `entity_key` parameter. Time alignment by intersection of
  pattern timestamp sets, hard floor `min_epochs=8`, default
  `cohort="fixed"` (panel-clean centroid). Significance: peak Bonferroni
  adjustment for the population peak (`max_corr_threshold` cut-off,
  `bartlett_ci_95` reported alongside, `is_significant` boolean,
  `degenerate_signal` flag set when either centroid drift series has zero
  variance — agreement forced to `"divergent"`). Reliability label mirrors
  `engine.forecast.reliability_label` convention (high N-1 ≥ 24, medium
  ≥ 12, else low). New types: `LeadLagReport`, `DimPairLeadLag`. Two-phase
  Lance scan: a column-projected meta read picks the cohort and
  pre-validates a 1 GB tensor budget against `count_geometry_rows` upper
  bounds, so cross-pattern queries over disjoint entity_lines (e.g. AML
  accounts vs account_pairs) raise immediately with an actionable error
  pointing the agent at `cohort='fixed'` or `entity_key=<id>` instead of
  allocating gigabytes. `read_temporal_batched` gains optional `columns=`
  projection.

### Fixed
- `find_anomalies` polygon construction crashed with
  `TypeError: float() argument must be a string or a real number, not 'NoneType'`
  on rows whose `delta_rank_pct` / `bregman_divergence` / `anomaly_confidence`
  column existed but stored a null Arrow value. The reconstruction code
  used `float(row.get(field, 0.0))` guarded by `if field in row`, but
  `row.get` returns `None` (not the default) for null cells, so the
  default never fired. Replaced with explicit
  `None if row.get(field) is None else float(row[field])` across both
  `engine.geometry._reconstruct_polygons_from_geometry_table` and the
  two find_anomalies branches in `navigation.navigator` (full-rank and
  rank-by-property). No agent-visible API change.
- `aggregate_anomalies`: negative `ungrouped_anomalies` when
  `read_points_batch` returns duplicate rows per entity across Lance
  partitions. Added dedup via `seen_pks` set — each entity is now counted
  in exactly one group regardless of partition layout. `ungrouped_anomalies`
  is now guaranteed non-negative.
- `detect_neighbor_contamination`: replaced row-by-row `as_py()` loop
  (O(N) PyArrow element access) with vectorised `to_pylist()` for the
  initial geometry scan. Also vectorised the small batch-read in the
  unknown-keys path.
- `detect_segment_shift`: removed internal `π12_attract_regime_change`
  call used only to populate the non-essential `changepoint_date` output
  field. Calling full changepoint detection (O(N×buckets)) for a single
  optional output field dominated the tool's wall-clock time. Field
  removed from output; callers needing changepoint context should call
  `find_regime_changes` separately. Per-segment Python loop over the
  points table also replaced with vectorised PyArrow ops — `pc.is_in`
  for the anomalous-PK mask, `pc.count_distinct` for the high-
  cardinality early exit, and
  `group_by(...).aggregate([("_pk","count"),("is_anomaly","sum")])` for
  the per-segment counts.
- `find_high_potential_edges` (`attract_edge_potential`) on event
  patterns with large edge tables: replaced the
  `AdjacencyIndex.from_lance()` rebuild — which read all five edge
  columns and called `to_pylist()` over the full table — with a direct
  two-column PyArrow `group_by(["from_key","to_key"])` aggregate.
  Eliminates the per-call full-table materialisation that dominated
  cost on multi-million-edge event patterns. Added an entity-type ratio
  guard that fires BEFORE the groupby: read the cached
  `_gds_meta/edge_stats/<pid>.json` (built at sphere-write time) for
  `unique_from + unique_to` and divide by `pattern.population_size`
  (already loaded from `sphere.json`). When the ratio is < 1 % the
  edge endpoints belong to a different entity type than the host
  geometry (e.g. zone IDs in a trip-edge pattern whose geometry holds
  trips); the tool returns an empty ranking without opening either the
  Lance edge table or the geometry dataset. New
  `GDSReader.edge_stats_cached(pid)` helper that reads the build-time
  JSON only — never falls back to the live full-table scan that
  `edge_table_stats` performs as a recovery path — so primitives that
  need a cheap guard cannot accidentally trigger a scan when the cache
  is missing.
- `detect_trajectory_anomaly`: changed `sample_size` default from
  `None` (full population scan) to `10,000`. Full-population temporal
  streaming on large spheres caused multi-minute latency; the default
  now caps at 10k entities. Pass `sample_size=0` to restore the full
  scan.

## [0.5.2] — 2026-04-28

### Added
- Two new closed-vocabulary motifs extending the 0.5.1 catalog:
  - `split_recombine` — diamond scatter-gather: source S → k distinct intermediaries M = {m₁,…,mₖ} → single sink D, with stacked-bipartite temporal order (all split-hops precede all recombine-hops within the window). Seed picks whether to enumerate forward from the source (`direction="forward"`) or backward from the sink (`direction="backward"`). Default window 24h, `min_k` default 3. Covers AML scatter-gather smurfing, parallel layering, and concentrator/sink atoms (forward from source vs backward from sink) without amount gating — for amount-gated chains use `structuring`. Canonical definition follows Starnini et al. 2021 and the IBM AMLSim stacked-bipartite spec.
  - `bipartite_burst` — complete K_{k,m} bipartite subgraph within a tight time window: k distinct sources each send to every one of m distinct sinks. Greedy single-core enumeration (not maximal): enumerate seed-as-source first, fall back to seed-as-sink. Parameters `min_k` (sources, default 3), `min_m` (sinks, default 3), default window 24h. Covers coordinated-burst and parallel-collusion atoms; complements `fan_out` + `fan_in` by requiring completeness on both sides rather than single-anchor density.
- `score_motif` and `find_high_potential_motifs` accept `direction` (`split_recombine`) and `min_m` (`bipartite_burst`) parameters. `min_k` override now applies to `split_recombine` and `bipartite_burst` in addition to `fan_out` / `fan_in`. Registry size grows from 6 to 8.

### Changed
- chain_k motif enumerator now uses per-k adaptive frontier cap (k=3,4: 1000; k=5: 500; k=6: 250; k=7: 125; k=8: 100) instead of a static 1000-cap. Bounds worst-case FHPM cold latency at higher k while preserving the generous cap for k=3,4 where measurements show no fragility. Public API unchanged (k=3..8 still accepted); `frontier_truncated=true` may surface more often at higher k as a result.
- Single-seed motif enumeration (fan_out, fan_in, cycle_2, cycle_3, structuring, chain_k) now delegates to the in-memory adjacency-path enumerator instead of issuing per-call Lance read_edges scans. Single source of truth per motif type — single-seed dispatch follows the same hot path as find_high_potential_motifs ranking.
- `_score_motif_from_edges` batches endpoint delta reads into a single filtered Lance scan and reuses the warm AdjacencyIndex pair_counts cache, mirroring the fast-path used by find_high_potential_motifs. Cuts found=true scoring tail latency from O(num_edges × per-endpoint Lance read) to O(1) batched scan. trace_root_cause's motif_potential auto-attach branch benefits transitively.
- bipartite_burst K-set intersection now starts from the smallest neighbour-set (size-ascending sink/source ordering) instead of alphabetic order, bounding the running intersection by the smallest set. Result identical (set intersection is commutative); protects against worst-case dense graphs where one sink dominates.
- `_enumerate_structuring` no longer raises `GDSNavigationError` when the underlying edge table lacks an amount column. The post-refactor adjacency path always carries amounts (defaulting to 1.0 when unspecified), so the predicate `amt >= amt1_min` returns an empty result cleanly. Now consistent with how the other 7 motifs handle no-match.
- cycle_3 motif enumerator now pre-filters intermediary candidates via 2-step adjacency traversal before the inner pair loop, skipping intermediaries that have no return-edge to the seed within the time window. Result equivalence preserved (semantic no-op on the candidate space).
- bipartite_burst K_{k,m} dispatcher now pre-checks seed in/out degree before invoking the per-side enumerator. Correctness-preserving; skips fruitless fail-fast work for seeds that don't qualify on either side.

## [0.5.1] — 2026-04-21

### Added
- Two new closed-vocabulary motifs extending the 0.5.0 structural catalog:
  - `fan_in` — k distinct sources → one sink within a sliding time window, mirror of `fan_out`. Seed = sink. Surfaces concentrator/sink accounts (T13) and destination-side parallel layering (T12) as a single-call atomic query, so the agent no longer has to invert the graph by hand.
  - `chain_k` — open directed chain A→B→…→Z of length `k` (3 ≤ k ≤ 8, k-1 edges), no cycle closure, no node revisit, strict monotone timestamps, total span ≤ window. Covers multi-stage layering typologies (T5 Long-Cycle Multi-Stage, T18 Multi-Jurisdiction Latency) that `cycle_3` and `structuring` could not express — `cycle_3` requires closure back to the seed and `structuring` is amount-gated with a 1h window; `chain_k` is the open-chain, amount-free variant that matches layering-over-days semantics. Each `chain_k` instance carries a `frontier_truncated: bool` flag — truthy when `_CHAIN_K_MAX_FRONTIER = 1000` was hit while expanding partial paths on at least one hop, signalling that the returned chains are real but the ranking may miss longer chains (recommended agent response: retry with a tighter `time_window_hours` or a lower `k`). Surfaces on both `score_motif` and `find_high_potential_motifs` result dicts.
- `score_motif` and `find_high_potential_motifs` gain a `k` parameter (default 4, validated 3–8 for `chain_k`, ignored for other types). `k` participates in the ranking cache key so different chain lengths are cached separately. `score_motif` additionally gains a `min_k: int | None = None` override on the single-seed path — when provided (must be ≥ 1), it plumbs through to the `fan_out` / `fan_in` enumerators, so callers can check e.g. "is this sink fed by ≥ 10 distinct sources?" without falling back on `find_high_potential_motifs` and triggering a pattern-wide cold cache. When None (default), enumerators fall back to their built-in `min_k = 3`, preserving the original behaviour exactly. Ignored for motif types that don't accept it (`cycle_2`, `cycle_3`, `structuring`, `chain_k`).
- `find_high_potential_motifs`'s `min_k` parameter now applies to `fan_in` in addition to `fan_out` (default 3).
- Numeric stability for the motif catalog: every motif scorer returns `log_score` (sum of `log(edge_potential)` over non-zero edges, `-inf` when any edge is zero) and `score_clamped: bool` alongside `score`. The raw `score` is clamped at `_MOTIF_SCORE_MAX = 1e300` when the edge-potential product would overflow (mirror of the existing `_MOTIF_SCORE_EPSILON = 1e-30` underflow clamp), and the ranking sort key in `_rank_motifs` switched from `-score` to `-log_score` so large-`k` motifs rank correctly regardless of clamp. Observed on AML HI-small: `fan_out` / `fan_in` hubs with k in the low thousands produced `score = +inf` that survived JSON serialisation, tied `score_rank_pct` to `100.0` across every affected result, and destroyed ordering at the top of the ranking.

### Fixed
- `passive_scan` chain and composite sources report entity-specific `related_count` instead of the total pattern population count.
- `detect_trajectory_anomaly` navigator method accepts `sample_size` to cap entity streaming on large patterns.
- Builder writes `null` (not `0.0`) for `anomaly_confidence` when bootstrap is skipped, preventing agents from misinterpreting zero as a computed confidence score.
- Single-seed motif enumerators for `fan_out`, `cycle_2`, and `cycle_3` computed their time-window threshold in microseconds against float-seconds timestamps (`EDGE_TABLE_SCHEMA` declares `timestamp: float64` = epoch seconds, as produced by `builder._to_epoch_seconds`). The µs-scaled threshold could never be exceeded by seconds-scale deltas, silently disabling the window filter in production — `score_motif(entity, motif_type, pattern_id, time_window_hours=H)` returned results as if no window was set. Flipped all three enumerators to seconds arithmetic, matching `_enumerate_structuring`, every `_enum_*_via_adj` adjacency path, and the new `_enumerate_fan_in` / `_enumerate_chain_k`. **Observable behaviour change on `score_motif`** for these three motif types: users who relied on the broken "all-time" behaviour may see fewer results on narrow windows. `find_high_potential_motifs` is unaffected (its adjacency ranking path was already correct). Test helper `_hour_us` retired in favour of `_hour_sec` so fixture semantics align with the production edge table; five tight-window parity regression guards (one per migrated enumerator plus the two shipping with the new motifs) prevent future unit reversion.

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
