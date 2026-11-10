# Refutations

This document records hypotheses that were tested, found wrong, and closed. Each entry follows the discipline: state the hypothesis, name the dataset, give the numeric evidence, and link the commit or PR that closed the line of work.

The point is not to celebrate failures. It is to leave a record so the same idea is not re-investigated six months later under a different name, and so a reviewer can see that the project takes negative results seriously.

---

## Pattern-level σ-estimator swap (refuted 0.6.6, PR #435)

**Hypothesis.** Replacing the population standard deviation σ used in `delta_norm` calibration with a robust estimator (MAD, IQR, Qn) would lift downstream anomaly detection by being less sensitive to outlier leak.

**What was tested.** Four σ estimators (`std`, `mad`, `iqr`, `qn`) plugged into `Pattern.sigma_diag` calibration. Evaluation on a money-laundering account-level pattern with binary `is_laundering` labels.

**Verdict.** AUROC and top-K precision differences across the four estimators sat inside the bootstrap noise band (±0.005 on AUROC). MAD and IQR did not deliver any robustness lift on this distribution. The "robust σ" line of work was closed.

**Why it failed.** Not because the inputs were normalised: account-level features reach calibration as raw aggregates — neither the builder nor the benchmark preparation applies a log or rank transform. The ranking at the top is dominated by a few heavy-tailed raw aggregates, and swapping the scale estimator rescales those dimensions without changing which entities sit in the tail. A different σ does not tame a heavy tail; a change of coordinates (quantile or copula transform of the dimension itself) is the open question this result points to.

**Citation.** PR #435 retrospective + `MEMORY.md` entry `project_population_robust_sigma_refuted`.

---

## Per-dim heuristic σ-estimator swap (refuted 0.6.6, PR #436)

**Hypothesis.** σ-swap might work if applied per-dim rather than population-wide — pick the estimator per column based on a heuristic (skewness, kurtosis, sparsity).

**What was tested.** Per-dim heuristic dispatcher selecting `std` / `mad` / `iqr` based on column statistics. Same evaluation setup as PR #435.

**Verdict.** Same outcome — ΔAUROC < 0.005, inside bootstrap noise. The heuristic dispatcher added complexity without measurable lift.

**Why it failed.** Same cause as above (#435): the inputs are raw, heavy-tailed aggregates, and choosing a different scale estimator per dimension still only rescales them — the tail that drives the ranking is unchanged. The dispatcher changed the estimator where the problem is the coordinate.

**Closure rule.** The entire "calibration robustness via σ-estimator swap" theme is closed: scale-estimator choice alone does not move detection on heavy-tailed raw inputs. Work on heavy tails continues as coordinate changes (per-dimension quantile or copula transforms), not as another σ estimator. The closure rule is enforced via the session memory `project_population_robust_sigma_refuted` — re-opening requires a different dataset or methodology.

---

## Lazy-chain geometry sampled calibration (refuted 0.5.x, retired 2026-04-19)

**Hypothesis.** Computing chain-pattern calibration on a sampled subset of chains rather than the full population would scale linearly with sphere size and preserve detection quality.

**What was tested.** Sampled-calibration variant against full-calibration baseline on the money-laundering chain pattern. Compared `delta_norm` rankings and downstream chain-level anomaly recall.

**Verdict.** Sampled calibration produced a sub-null `delta_norm` distribution — calibration estimates were biased low because the sampled tail underrepresented the true variance. Bias roughly 24 % on the most affected dims. Chain recall on labelled bad chains dropped materially.

**Why it failed.** Chains are a long-tail distribution by structure. Uniform sampling under-weights long chains, which carry the discriminating signal. A weighted sampler would have helped in theory but the engineering cost outweighed the expected gain.

**Closure.** The lazy-chain-geometry document carries an in-doc retired banner dated 2026-04-19. The work was superseded by treating chains as first-class geometric entities (the `chain_lines:` YAML block and the chain investigation primitives shipped in 0.6.x and 0.7.0).

---

## Bregman-based `is_anomaly` flag (refuted 2026-04-16)

**Hypothesis.** Replacing the Mahalanobis-distance-based anomaly flag with a Bregman-divergence-based flag would lift top-K precision on money-laundering account-level detection by giving exp-family-correct distance semantics for non-gaussian dims (counts, ratios).

**What was tested.** Bregman-flag variant evaluated against Mahalanobis baseline at K ∈ {50, 100, 200, 500}, top-K precision and recall on labelled laundering accounts.

**Verdict.** ΔF1 ≤ +0.005 across all K values — inside bootstrap noise. The Bregman variant did not deliver measurable lift on this data shape.

**Why it failed.** The dimension-kind tags in production are mostly `gaussian` or `mahalanobis` with a few `bernoulli` flags. Bregman's correctness advantage only matters when a substantial fraction of dims are explicitly `poisson` or `bernoulli` and carry top-of-rank load. On the tested pattern, no such dim dominated.

**Closure.** The `is_anomaly` flag stays Mahalanobis-based. Bregman remains a per-dim choice via `dimension_kinds: poisson` / `bernoulli` declarations but is not promoted to the global flag computation. Tracked as a closed direction in session memory.

---

## Process refutation — emergent cycle-compression (0.7.0 retrospective)

This one is not algorithmic. The 0.7.0 cycle landed five PRs in a 24-hour window, four of which had been originally planned as 0.7.1 milestones. The result was a tag that shipped on the planned date but with a release-prep state that drifted across packages.

**Lesson.** Items that emerge during a cycle and feel ready to ship should not skip cycle boundaries unless the truth-in-CHANGELOG cost of leaving them is concrete. The hard cycle-window rule is enforced going forward: feature PRs land outside the seven-day pre-tag window; the release-prep PR is the literal last PR before the tag; emerged-during-execution items defer to the next cycle by default.

**Why this entry exists.** Methodology refutations are as load-bearing as algorithmic ones — they keep the cycle cadence honest. The 0.7.1 cycle deliberately sized smaller to test the rule under real conditions.

---

## Geometry research round — what did not hold (refuted 2026-10-26/27)

A pre-registered research round tested candidate foundations for the entity geometry on a labelled synthetic money-laundering benchmark (two packs from one generator, development and locked halves kept apart) and on a second, non-template synthetic money-laundering benchmark. Each line below is a closed direction; reopening one needs a different dataset or a different construction, not a re-run.

**Variance-stabilising (Anscombe-type) count coordinates.** Hypothesis: treating counts as Poisson and transforming them with the Anscombe square root gives an honest χ² null and better ranking. Verdict: worse than a plain `sign·log1p` transform (paired ΔAUROC in favour of the logarithm, CI excluding 0 on one pack, tie on the other). Why: the population's daily event counts follow a Taylor law with exponent ≈ 1.75, not 1; at an exponent near 2 the variance-stabilising transform is the logarithm. Coordinates should follow the measured scaling law, not an assumed data kind.

**Correlation-aware distance as the headline.** Hypothesis: a shrunk-covariance (Ledoit–Wolf) Mahalanobis distance on the account features beats the best simple baseline. Verdict: it clearly improves on the flat per-dimension distance, but on the first benchmark it does not beat the best single raw column, and on the second it is one robust axis among several rather than a winner. It stays as an axis, not as the claim.

**Per-context quotas.** Hypothesis: calibrating p-values separately per context (Mondrian conformal), or ranking within each context, gives "unusual for entities like me". Verdict: both act as a quota per context and cost 0.07–0.09 AUROC against the same score calibrated globally. Calibrate globally; condition inside the score. Amendment: the finding holds for arbitrary group splits. When the population genuinely consists of distinct kinds of entity — each kind having different normal behaviour, not just different scale — fitting a reference per kind (opt-in via `score_per_group`) is a supported configuration; detection quality on arbitrary group splits is not assessed and should not be assumed to hold.

**χ² p-values after parametric fixes.** Full covariance and rank-Gaussianisation both leave the χ² tail miscalibrated (several times the nominal share of negatives below p = 0.01). Split-conformal calibration of the same scores is exact to within a tenth of a percentage point.

**One-hop neighbour residual in σ-space.** Hypothesis: an entity's distance from the mean position of its counterparties carries the relational signal. Verdict: close to chance (AUROC ≈ 0.55). A neighbourhood-averaged e-value ("entity and its neighbourhood are ordinary") is below the entity's own score on both benchmarks; on the non-template benchmark the neighbourhood is anti-informative.

**Designed-process constructs on template data.** Subset-sum closure of outgoing amounts, value retention around loops, effective-resistance leverage and a motif replication index were all at or below chance overall and on the typologies they target. Value retention is not measurable when amounts are recorded in different payment currencies without conversion; it needs single-currency or converted amounts before it can be tested.

**Local-dimension and atlas variants of density.** Putting the local intrinsic dimension into the kNN volume formula, and scoring rarity inside or distance off local PCA charts, are both below plain kNN density on both benchmarks.

**New-foundation constructions.** A forward/reverse divergence plane ("mimicry"), broken-symmetry scores, a Helmholtz circulation share of the entity's trajectory, proper-time (evidence-age) drift and observer-resolution blurring carry no signal once the number of events is controlled; most of their raw versions were the event count in disguise.

**What held, for contrast.** Coordinates from rank or log transforms, global split-conformal calibration, Shapley attribution over named coordinates without redundant pairs, and a residual against the population's current carried signal; supervised models trained on the same features remain far ahead whenever labels are plentiful.

---

## Extensions of the calibrated score — what did not hold (refuted 2026-10-29 to 2026-11-01)

Each line below was pre-registered with a pass rule before it ran and stays closed. The data are the same two labelled synthetic money-laundering benchmarks as the geometry round (development halves only), the labelled synthetic generator shipped in `hypertopos.datasets`, and simulated populations where the question is about error rates.

**False-discovery-rate alert sets for scored batches.** Hypothesis: a Benjamini–Hochberg selection over a batch's conformal p-values gives a better alert list than the fixed alarm budget. Verdict: the selection is valid (every null cell stays within its bound, and the false share stays under q in every contaminated cell) and a quiet day gives almost no alerts, but the registered rule asked for fewer false alerts than the budget in every contaminated cell, and that fails once more than about a tenth of a batch is anomalous: the selection threshold then exceeds the budget's own. Two registrations, two fails; no alert-set feature ships. `calibration_status` remains the report for a batch that has moved as a whole.

**Counterfactual explanation ("the smallest change that would make this entity ordinary").** Hypothesis: the first dimensions such a search moves name the mechanism of the entity's typology. Verdict: they contain the mechanism dimension for 27 % of flagged laundering accounts against a required 60 %, barely above shuffled typologies (21 %). A search on the raw distance instead of the saturated p-value does not change that (25 %). The reasons `explain` already gives name the mechanism dimension in their top two for 40 % of the same accounts. Only the per-axis evidence split (`axis_evidence`) ships, described as what it computes.

**Group masking on the `current` axis.** Hypothesis: a group that moves together forms its own neighbourhood in the population's flow, hides its members' motion, and a leave-group-out gain exposes it. Verdict: the mechanism is real on synthetic data (a co-moving group of about a hundred entities is fully masked, and leaving it out restores the signal), but on the benchmark's known laundering rings the gain is exactly zero for all 34 rings: ring members do not sit near each other in state space, so nothing is hidden. No tool ships. It reopens only for data where co-moving groups with similar states exist.

**Other ways to fuse the axes.** Hypothesis: a Cauchy combination, a weighted mean or the maximum of the axis ranks keeps strong single-axis anomalies that the mean of ranks dilutes. Verdict: at a 1 % false-positive rate none improves on the mean of ranks on either benchmark: all three lose on the second, and the Cauchy combination and the maximum also lose on the first. The fusion rule stays.

**Kurtosis dimension weights.** Hypothesis: weighting dimensions by excess kurtosis (`dimension_weights: kurtosis`) improves ranking on a `delta_norm`-scored event pattern. Verdict: paired AUROC difference −0.054 against uniform weights, with the interval excluding zero. The option stays available; the documentation no longer recommends it.

**Calibrated evidence as features for a supervised model.** Hypothesis: adding the label-free evidence and the calibrated coordinates to a gradient-boosted model's inputs lifts it. Verdict: average precision drops slightly on the benchmark (interval excluding zero) and on each of three synthetic seeds. The evidence summarises the same columns, and a model with labels recovers it by itself. hypertopos is not a feature store for supervised models.

**An agent with the tool surface against an agent with SQL.** Hypothesis: on the same investigation tasks an agent using the MCP tools beats an agent with a SQL engine and an agent with a Python sandbox on screening, typology naming and ring membership, and gives stable answers to paraphrased questions. Verdict: fail. The tool agent is ahead on screening, by using the calibrated score the SQL agent ignored, but it is behind on naming the typology, its answers change under paraphrase more than the rule allows (so do the other agents'), and it costs more than twice as much per run. A plain SQL graph rule bounds all three agents from far above on the screening task. The registered consequence is the fixed tool profiles: a short default surface instead of a growing one.

**What held, for contrast.** The conformal p-value keeps its false-alarm budget on new rows; the population test behind `calibration_status` and the `population_drift` alert detects a moved population where the flag count alone does not; per-group references help when the groups are genuinely different kinds of entity.
