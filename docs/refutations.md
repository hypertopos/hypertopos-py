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
