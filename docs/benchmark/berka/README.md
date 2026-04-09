# Berka Financial — Skill Calibration Study

Agent-driven autonomous exploration of Czech banking data with ground truth
validation. Fully unsupervised — no labels are used during sphere build or
anomaly detection. The agent receives a sphere and a single instruction:
"explore and report." Results are scored against known loan defaults.

This study serves four purposes:
1. **GDS capability validation** — can an AI agent autonomously explore a sphere, find anomalies, and produce an actionable report?
2. **Skill calibration** — do `hypertopos-skills` workflows improve agent performance over cold-start? Each run iteration (r1, r2, r3, ...) refines skill recipes based on observed gaps.
3. **MCP interface quality** — cold-start agent performance (no skills, no domain hints) measures how well MCP tool descriptions alone guide autonomous investigation.
4. **Recall optimization** — iterate skill recipes to maximize recall against known loan defaults while keeping false positive rate low.

## Dataset

**PKDD 1999 Financial Dataset** — real anonymized data from a Czech bank,
published for the PKDD'99 Discovery Challenge. The dataset covers ~5 years of
banking operations (1993—1998) and remains one of the most widely used
relational benchmarks in data mining, anomaly detection, and financial analytics
research.

**Origin:** Petr Berka and Marta Sochorová prepared the data for the ECML/PKDD
1999 conference in Prague. It captures the full lifecycle of retail banking:
account creation, transaction history, loan granting and repayment, credit card
issuance, and demographic context of the account holder's district.

**Why this dataset:** It is small enough to run quickly (~1M transactions), rich
enough to contain real anomalies (loan defaults, unusual transaction patterns),
and relational enough to test cross-entity navigation (accounts → transactions →
districts → loans). Unlike synthetic benchmarks, the anomalies are organic —
they come from real customer behavior, not injected patterns.

**Tables in the original dataset:**

| Table | Rows | Description |
|-------|------|-------------|
| account | 4,500 | Bank accounts with creation date and district |
| trans | 1,056,320 | All transactions (credits, debits, cash, transfers) |
| loan | 682 | Loans linked to accounts (76 ended in default) |
| order | 6,471 | Standing payment orders |
| disp | 5,369 | Account—client dispositions (owner/disponent) |
| client | 5,369 | Clients with birth date and gender |
| district | 77 | Demographic data per district (population, unemployment, crime) |
| card | 892 | Credit cards issued |

**Source:** https://relational.fel.cvut.cz/dataset/Financial

### Ground truth

76 accounts have defaulted loans (`loan_status` = B or D). The agent does NOT
know this — loan_status is a property on the accounts line, visible only if the
agent profiles entity properties. The study measures whether the agent can
independently identify these accounts as anomalous or high-risk.

### Sphere construction

The sphere is built from the raw tables above. Transactions are the event core;
accounts, districts, transaction types, operations, and counterparty banks are
anchor dimensions. The build scripts in `benchmark/berka/sphere/prepare_*.py`
transform raw CSVs into Arrow-based GDS geometry.

| Property | Value |
|----------|-------|
| Accounts | 4,500 |
| Transactions | 1,056,320 |
| Loans | 682 (76 defaults) |
| Districts | 77 |
| Patterns | 5 (1 event, 4 anchor) |
| Aliases | 2 (high_activity, low_engagement) |
| Sphere | `benchmark/berka/sphere/gds_berka_banking` |

## Sphere structure

| Pattern | Type | Entity line | Dims | Description |
|---------|------|-------------|------|-------------|
| `tx_pattern` | event | transactions | 6 | account, type, operation, bank + amount, balance |
| `account_behavior_pattern` | anchor | accounts | 8 | Activity, diversity, burst patterns |
| `account_stress_pattern` | anchor | accounts_stress | 6 | Penalty interest, balance volatility, min_balance, balance_trend (isolated entity line) |
| `loan_stress_pattern` | anchor | loan_accounts | 8 | Loan-only stress (682 entities) — penalty, volatility, min_balance, trend, balance_to_loan, income_coverage + derived amount_std, mean_balance |
| `account_bank_pairs_pattern` | anchor | account_bank_pairs | 4 | Pair co-occurrence (account × bank) |


## Experiments

Both experiments are **fully autonomous** — the agent runs to completion without
human intervention. No stopping, no asking for guidance, no mid-run corrections.
The agent must decide its own investigation strategy and produce a final report.

**Mode:** Manual — the agent uses individual MCP tools directly (Phase 3 toolset).
This study measures manual-mode performance: how well the agent navigates
using granular tools (find_anomalies, goto, get_polygon, contrast_populations, etc.)
with 40-54 tools visible. It does NOT use `detect_pattern` (smart mode) — that is
a separate evaluation target (see NYC Taxi validation).

**Exhaustive exploration rule:** The agent must continue investigating until it
concludes there is nothing more to find. "I'm not sure what else to look for"
is not a stopping condition — the agent must try another tool, another pattern,
another angle. The only valid stopping condition is: "I have explored all
available patterns, temporal data, cross-entity relationships, and population
segments, and I see no further avenues to investigate."

**Model:** Claude Sonnet 4.6 (200K context) for both experiments.
**Scorer:** Claude Opus 4.6 (1M context) — independent agent, receives only the report and rubric.

### Nav-log (planned)

Aggregate session metrics available via `get_session_stats` (call count,
elapsed_ms, per-tool breakdown). Per-call JSONL nav-log planned for future.

## Evaluation criteria

### Shared criteria (both experiments)

| # | Criterion | 0 | 1 | 2 | 3 |
|---|----------|---|---|---|---|
| 1 | **Discovery completeness** | Explored ≤1 pattern | 2 patterns | 3 patterns | All 4 patterns + aliases |
| 2 | **Default detection** | No mention of loan defaults | Mentioned loans exist | Noted defaults as interesting | Computed **exact** recall (all 76 keys checked individually): identified ≥50% of 76 default accounts as anomalous or high-risk |
| 3 | **Cross-pattern reasoning** | Single-pattern only | Mentioned multiple patterns | Used compare_entities / find_common_relations | Used composite_risk or passive_scan AND connected findings across patterns |
| 4 | **Temporal awareness** | No temporal calls | Checked get_solid once | Used drift or regime changes | Full temporal analysis with interpretation AND distinguished real shifts from data boundary artifacts |
| 5 | **Navigation efficiency** | Stuck in loops or repeated calls without new information | >120 calls, significant redundancy | 60-120 calls, mostly focused | <60 calls, every call adds new information, systematic coverage |
| 6 | **Explanation quality** | Vague ("some anomalies found") | Listed anomalies without context | Backed by dimensions/effect sizes | Witness sets, conformal p-values, repair sets AND root cause hypotheses (not just restating tool output) |
| 7 | **Actionable output** | No report | Report with generic statements | Report with specific findings | Report with prioritized findings, false positive assessment, and concrete next steps a human could execute |
| 8 | **Hypothesis testing** | No hypotheses formed | Stated observations without testing | Formed hypotheses and tested 1 | Formed ≥3 hypotheses, tested each with targeted tool calls, reported confirmed/rejected |

### Experiment B — additional criteria

| # | Criterion | 0 | 1 | 2 | 3 |
|---|----------|---|---|---|---|
| 9 | **Skill adherence** | Ignored skills | Used some recipes | Followed skill workflows | Full funnel (explorer → investigator → monitor) |
| 10 | **Feature coverage** | Used only 1 pattern | Explored 2 patterns | Used composite_risk or passive_scan | Per-pattern recall computed + composite scoring |
| 11 | **Alias utilization** | Ignored aliases | Mentioned aliases | Used attract_boundary on 1 | Used attract_boundary with direction filter on both aliases |

### Scoring

| Experiment | Max | Target |
|-----------|-----|--------|
| A — Cold-start | 24 | 18+ |
| B — Skill-augmented | 33 | 26+ |

### Scoring methodology

Evaluation is performed by an **independent scorer agent** — a separate Claude
session that receives only the report and scoring rubric, with no knowledge of
which experiment produced it. The scorer does not see the agent's tool calls,
only the final report. This eliminates self-evaluation bias.

---

### Experiment A — Cold-start exploration (zero context)

**Agent:** Claude with MCP tools only. No skills, no CLAUDE.md, no domain hints.

**Note:** The agent prompt includes methodological instructions (compute recall,
form hypotheses, assess false positives). These are investigation techniques,
not domain knowledge — but they do guide the agent's approach. The model also
carries training knowledge (Berka is a well-known academic dataset).

**Rules:**
- Agent receives only: sphere path + MCP tool access + methodological instructions
- No knowledge about loans, defaults, regions, fraud beyond what the model learned in training
- No navigation strategy, no skill files
- **Fully autonomous** — agent must run to completion and produce a final report without any human input
- Agent must NOT stop to ask questions — if uncertain, keep exploring until certainty is reached
- **No early exit** — the agent must exhaust all available navigation tools, patterns, temporal layers, and cross-entity paths before writing the final report. Stopping because "I'm not sure what else to check" is a failure — try another tool, another pattern, another angle

**Instruction:**
```
Open the sphere at benchmark/berka/sphere/gds_berka_banking.
You have access to MCP navigation tools. Explore this sphere completely
from scratch. Discover what the data contains, find anomalies, investigate
their root causes, identify patterns, and produce a complete investigation
report. Do not stop or ask for guidance — run your full investigation
autonomously. Save the final report to benchmark/berka/results/0.1.0/<run>/0.1.0-exp-a-report.md.
```

**Methodological instructions (included in prompt):**
- Form explicit hypotheses, test each with targeted tool calls, report confirmed/rejected (minimum 3)
- When you discover labeled outcomes (status fields, default flags), compute **exact** recall and precision — check ALL ground truth keys individually (goto + get_polygon for each), do not estimate from samples
- Assess false positives for every major finding — explain WHY it matters, not just THAT it exists
- Prioritize findings by business impact, not just delta_norm

**Output artifacts:**
1. `benchmark/berka/results/0.1.0/<run>/0.1.0-exp-a-report.md` — agent's final investigation report

**Scoring:** shared criteria 1-8 (max 24, target 18+)

---

### Experiment B — Skill-augmented exploration

**Agent:** Claude with MCP tools + `hypertopos-skills` skills.

**Rules:**
- Same sphere path + MCP tool access as Experiment A
- Additionally receives 3 skills from [`hypertopos-skills`](../../packages/hypertopos-skills/):
  - **gds-explorer** — orientation, population analysis, segmentation, aggregates
  - **gds-investigator** — anomaly root-cause tracing, entity deep-dive, incident reconstruction
  - **gds-monitor** — drift detection, regime changes, alerts, temporal health
- Skills provide structured workflows, recipes, anti-patterns, and tool-calling order
- Still no domain-specific instructions about what to look for
- **Fully autonomous** — same rules as Experiment A, no human interaction
- **No early exit** — same exhaustive exploration rule as Experiment A applies

**Instruction:**
```
BEFORE doing anything else, read these 3 skill files completely:
1. hypertopos-skills/gds-explorer/SKILL.md
2. hypertopos-skills/gds-investigator/SKILL.md
3. hypertopos-skills/gds-monitor/SKILL.md

WARNING: Reading these files costs ~15K tokens. This is REQUIRED.
Do NOT skip to save context.

Then open the sphere at benchmark/berka/sphere/gds_berka_banking.
Explore the sphere completely, find anomalies, investigate root causes,
compare populations, check temporal patterns, and produce a complete
investigation report. Do not stop or ask for guidance — run autonomously
to completion.
Save the final report to benchmark/berka/results/0.1.0/<run>/0.1.0-exp-b-report.md.
```

**Methodological instructions (same as Exp A, plus skills provide additional guidance):**
- Same hypothesis, recall, FP assessment requirements as Exp A — recall and precision must be **exact** (all 76 ground truth keys checked individually), never estimated from samples
- Skills add: bottom-up recall method, pattern triage, entity line awareness, population contamination detection

**Key differences from Experiment A:**

| Aspect | Exp A (zero context) | Exp B (skill-augmented) |
|--------|---------------------|------------------------|
| Navigation strategy | Agent invents own | Reactive skill guidance (if-then rules) |
| Ground truth | Must compute exact recall (all 76 keys) | Bottom-up exact recall per pattern (skill recipe) |
| Cross-pattern | May not discover composite_risk | Skills guide per-pattern + composite scoring |
| Temporal | May skip or do ad-hoc | gds-monitor provides health check workflow |
| Investigation depth | May list anomalies without explaining | gds-investigator requires evidence → verdict |

**Output artifacts:**
1. `benchmark/berka/results/0.1.0/<run>/0.1.0-exp-b-report.md` — agent's final investigation report

**Scoring:** shared criteria 1-8 + additional 9-11 (max 33, target 26+)

## Running the experiments

```bash
# Build the sphere (if not already built)
hypertopos build --config benchmark/berka/sphere/sphere.yaml \
                 --output benchmark/berka/sphere/gds_berka_banking --force

# Experiment A — fresh Claude Code session, NO .claude/ context
# Strip all CLAUDE.md, skills, memory from the workspace
# Only MCP server config (.mcp.json) remains

# Experiment B — fresh Claude Code session WITH hypertopos-skills
# Install skills: gds-explorer, gds-investigator, gds-monitor

# Results go to benchmark/berka/results/
mkdir -p benchmark/berka/results
```

## Results

Each run is tied to a specific hypertopos version. Results, settings,
and scoring are recorded per version in a dedicated file under `results/`.

### Current

Recall = best reported by the agent (pattern varies per run). Precision = on that pattern's anomaly set.

| Version | Experiment | Score | Recall (best reported) | Keys checked | Settings | Report | Scoring |
|---------|-----------|-------|----------------------|:------------:|----------|--------|---------|
| 0.1.0-r6 | A — Cold-start | **23/24 (96%)** | 30.3% (loan_stress) | 76/76 | [settings](results/0.1.0/r6/0.1.0-settings.md) | [report](results/0.1.0/r6/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r6/0.1.0-exp-a-scoring.md) |
| 0.1.0-r6 | B — Skill-augmented | **32/33 (97%)** | 85.5% (account_stress), 88.2% union | 76/76 | [settings](results/0.1.0/r6/0.1.0-settings.md) | [report](results/0.1.0/r6/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r6/0.1.0-exp-b-scoring.md) |
| 0.1.0-r5 | A — Cold-start | 21/24 | ~37% (account_stress, 41 keys) | 41/76 | [settings](results/0.1.0/r5/0.1.0-settings.md) | [report](results/0.1.0/r5/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r5/0.1.0-exp-a-scoring.md) |
| 0.1.0-r5 | B — Skill-augmented | **32/33 (97%)** | 85.5% (account_stress), 90.8% composite | 76/76 | [settings](results/0.1.0/r5/0.1.0-settings.md) | [report](results/0.1.0/r5/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r5/0.1.0-exp-b-scoring.md) |
| 0.1.0-r4 | A — Cold-start | 20/24 | 85.5% (account_stress) | 76/76 | [settings](results/0.1.0/r4/0.1.0-settings.md) | [report](results/0.1.0/r4/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r4/0.1.0-exp-a-scoring.md) |
| 0.1.0-r4 | B — Skill-augmented | 30/33 | 85.5% (account_stress) | 76/76 | [settings](results/0.1.0/r4/0.1.0-settings.md) | [report](results/0.1.0/r4/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r4/0.1.0-exp-b-scoring.md) |
| 0.1.0-r3 | A — Cold-start | 21/24 | 85.5% (account_stress) | 76/76 | [settings](results/0.1.0/r3/0.1.0-settings.md) | [report](results/0.1.0/r3/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r3/0.1.0-exp-a-scoring.md) |
| 0.1.0-r3 | B — Skill-augmented | 31/33 | 85.5% (account_stress), 89.5% composite | 76/76 | [settings](results/0.1.0/r3/0.1.0-settings.md) | [report](results/0.1.0/r3/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r3/0.1.0-exp-b-scoring.md) |
| 0.1.0-r2 | A — Cold-start | 21/24 | 30.3% (loan_stress) | 76/76 | [settings](results/0.1.0/r2/0.1.0-settings.md) | [report](results/0.1.0/r2/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r2/0.1.0-exp-a-scoring.md) |
| 0.1.0-r2 | B — Skill-augmented | **32/33 (97%)** | 85.5% (account_stress) | 76/76 | [settings](results/0.1.0/r2/0.1.0-settings.md) | [report](results/0.1.0/r2/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r2/0.1.0-exp-b-scoring.md) |
| 0.1.0-r1 | A — Cold-start | 20/24 | 85.5% (account_stress) | 76/76 | [settings](results/0.1.0/r1/0.1.0-settings.md) | [report](results/0.1.0/r1/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r1/0.1.0-exp-a-scoring.md) |
| 0.1.0-r1 | B — Skill-augmented | **32/33 (97%)** | 85.5% (account_stress) | 76/76 | [settings](results/0.1.0/r1/0.1.0-settings.md) | [report](results/0.1.0/r1/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r1/0.1.0-exp-b-scoring.md) |

Note: account_stress_pattern recall (85.5%) is deterministic — the geometric signal is the same every run. Variation in reported recall reflects which pattern the agent chose to validate against, not a change in sphere quality.

Full 0.1.0 analysis (6 runs, skill evolution, ML comparison): [`results/0.1.0/SUMMARY.md`](results/0.1.0/SUMMARY.md)


