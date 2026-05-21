# Refutations

This document records hypotheses that were tested, found wrong, and closed. Each entry follows the discipline: state the hypothesis, name the dataset, give the numeric evidence, and link the commit or PR that closed the line of work.

The point is not to celebrate failures. It is to leave a record so the same idea is not re-investigated six months later under a different name, and so a reviewer can see that the project takes negative results seriously.

---

## Pattern-level σ-estimator swap (refuted 0.6.6, PR #435)

**Hypothesis.** Replacing the population standard deviation σ used in `delta_norm` calibration with a robust estimator (MAD, IQR, Qn) would lift downstream anomaly detection by being less sensitive to outlier leak.

**What was tested.** Four σ estimators (`std`, `mad`, `iqr`, `qn`) plugged into `Pattern.sigma_diag` calibration. Evaluation on a money-laundering account-level pattern with binary `is_laundering` labels.

**Verdict.** AUROC and top-K precision differences across the four estimators sat inside the bootstrap noise band (±0.005 on AUROC). MAD and IQR did not deliver any robustness lift on this distribution. The "robust σ" line of work was closed.

**Why it failed.** Account-level features are already log-transformed or rank-normalised upstream of calibration. Outliers no longer drive σ when the input is approximately normal post-transform. The σ-swap intervention had no surface to act on.

**Citation.** PR #435 retrospective + `MEMORY.md` entry `project_population_robust_sigma_refuted`.

---

## Per-dim heuristic σ-estimator swap (refuted 0.6.6, PR #436)

**Hypothesis.** σ-swap might work if applied per-dim rather than population-wide — pick the estimator per column based on a heuristic (skewness, kurtosis, sparsity).

**What was tested.** Per-dim heuristic dispatcher selecting `std` / `mad` / `iqr` based on column statistics. Same evaluation setup as PR #435.

**Verdict.** Same outcome — ΔAUROC < 0.005, inside bootstrap noise. The heuristic dispatcher added complexity without measurable lift.

**Why it failed.** Same root cause as PR #435 — when upstream features are normalised, σ-estimator choice does not materially change the calibration. The dispatcher was solving a problem that did not exist on this data shape.

**Closure rule.** The entire "calibration robustness via σ-estimator swap" theme is closed for the project unless a new dataset surfaces with un-normalised, heavy-tailed inputs where the assumption is testable from scratch. The closure rule is enforced via the session memory `project_population_robust_sigma_refuted` — re-opening requires a different dataset or methodology.

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
