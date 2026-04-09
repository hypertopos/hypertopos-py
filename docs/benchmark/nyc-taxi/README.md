# NYC Yellow Taxi — Domain Generalization Test

Agent-driven autonomous exploration of NYC taxi trip data with organic anomaly
detection. Fully unsupervised — no labels injected, no synthetic patterns.
The agent receives a sphere and a single instruction: "explore and report."
Anomalies are real: ghost trips, fare outliers, speed impossibilities, tip
fraud patterns, dead zones, temporal bursts.

This validation serves four purposes:
1. **Real-world anomaly detection** — can an AI agent find genuine anomalies in messy, real transportation data using geometric navigation?
2. **Smart mode validation** — does `detect_pattern` correctly route queries and produce actionable results on non-financial, non-synthetic data?
3. **Domain generalization** — hypertopos was developed on financial data (Berka, AML). NYC taxi tests whether the geometric approach generalizes to transportation/logistics.
4. **MCP calibration** — how well do 55 MCP tool descriptions guide an agent on unfamiliar data? Cold-start performance (Exp A, no skills) measures interface quality independent of skill recipes.

## Dataset

**NYC TLC Yellow Taxi Trip Records** — real taxi trip data published by the
New York City Taxi and Limousine Commission. January 2019, ~7.6M trips after
cleaning.

**Source:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

**Why this dataset:** It is large enough to stress-test performance (7.6M events),
rich enough to contain real organic anomalies (fare fraud, ghost trips, speed
impossibilities), and relational enough to test cross-entity navigation
(zones x vendors x rate codes x payment types). Unlike TPC-H, anomalies are
organic — they come from real driver/passenger behavior, not injected patterns.

### Data after cleaning

| Property | Value |
|----------|-------|
| Trips | 7,556,807 |
| Zones | 265 (TLC taxi zones) |
| Vendors | 2 (Creative Mobile, VeriFone) |
| Rate codes | 6 (Standard, JFK, Newark, Nassau, Negotiated, Group) |
| Payment types | 6 (Credit card, Cash, No charge, Dispute, Unknown, Voided) |
| Period | January 2019 (1 month) |
| Avg fare | $12.24 |
| Avg distance | 2.83 mi |
| Avg duration | 12.5 min |
| Avg speed | 12.6 mph |

### Organic anomaly categories (ground truth)

No injection. These are real patterns the agent should discover:

| Category | Description | Expected signal |
|----------|-------------|----------------|
| **Ghost trips** | Zero passengers, minimal distance, non-zero fare | High delta_norm on distance + passenger dims |
| **Fare outliers** | Extreme fares ($200+ on short trips, $0.01 fares) | rank_by_property on fare_amount |
| **Speed impossibilities** | Trips reporting 80+ mph in NYC | speed_mph dim dominates delta |
| **Tip anomalies** | Cash trips always tip=0 (cash tips unreported), credit tip distributions | tip_pct dim, payment_type relation |
| **Dead zones** | Zones with near-zero pickup activity | Zone activity pattern, trip_count near 0 |
| **Surge zones** | Airport zones (JFK, LaGuardia, Newark) with extreme fare/distance patterns | Zone pattern outliers, rate_code=2/3 |
| **Temporal bursts** | New Year's Day (Jan 1), snowstorms, weekend vs weekday shifts | Temporal drift in zone patterns |
| **Rate code abuse** | Negotiated fare (rate_code=5) on short Manhattan trips | Rate code relation + fare/distance mismatch |

### Sphere construction

| Property | Value |
|----------|-------|
| Trips (event) | 7,556,807 |
| Zones (anchor) | 265 |
| Vendors (anchor) | 2 |
| Rate codes (anchor) | 6 |
| Payment types (anchor) | 6 |
| Patterns | 4 (1 event, 3 anchor) |
| Aliases | 1 (high_volume_zones) |
| Temporal | Weekly windows on zone patterns |
| Sphere | `benchmark/nyc-taxi/sphere/gds_nyc_taxi` |

## Sphere structure

| Pattern | Type | Entity line | Dims | Description |
|---------|------|-------------|------|-------------|
| `trip_pattern` | event | trips | 11 | 5 relational (pu_zone, do_zone, vendor, rate_code, pay_type) + 6 continuous (fare, distance, duration, tip%, speed, passengers) |
| `zone_activity_pattern` | anchor | zones_pickup | 8 | Pickup zone profile: trip volume, fare stats, vendor mix, tip rates |
| `zone_destination_pattern` | anchor | zones_dropoff | 6 | Dropoff zone profile: destination volume, fare, distance, duration |
| `vendor_pattern` | anchor | vendors | 6 | Vendor operational profile (only 2 entities — limited) |

## Experiments

Two experiments, both **fully autonomous** — the agent runs to completion
without human intervention. Same sphere, same task, different tools.

**What this validation measures:** does smart mode (`detect_pattern`) produce
better, faster, cheaper results than manual mode (40+ granular tools)?
Secondary: does the geometric approach generalize to non-financial data?

**Model:** Claude Sonnet 4.6 (200K context) for both experiment agents.
**Scorer:** Claude Opus 4.6 (1M context) — independent agent, receives only the report and rubric.

---

### Exp A — Smart mode

**Agent:** Claude with `detect_pattern` as primary tool. Manual tools available
after `sphere_overview()` for drill-down only.

**Instruction:**
```
Open the sphere at benchmark/nyc-taxi/sphere/gds_nyc_taxi using MCP tools.

This is a NYC Yellow Taxi dataset. Explore it and find anything unusual.

Produce a final report with your findings — what anomalies exist, why
they matter, and how confident you are in each.

Do not stop or ask for guidance — run autonomously to completion.
Save report to benchmark/nyc-taxi/results/0.1.0/<run>/0.1.0-exp-a-report.md.
```

---

### Exp B — Manual mode + skills

**Agent:** Claude with full MCP toolset (40+ tools) + hypertopos-skills.
No `detect_pattern` — agent must choose which tools to call, guided by skills.

**Instruction:**
```
BEFORE doing anything else, read these skill files completely:
1. packages/hypertopos-skills/gds-analyst/SKILL.md — investigation driver
2. packages/hypertopos-skills/gds-scanner/SKILL.md — advanced detection
3. packages/hypertopos-skills/gds-detective/SKILL.md — detection recipes
4. packages/hypertopos-skills/gds-investigator/SKILL.md — root cause
5. packages/hypertopos-skills/gds-monitor/SKILL.md — temporal monitoring
6. packages/hypertopos-skills/gds-explorer/SKILL.md — orientation

Then open the sphere at benchmark/nyc-taxi/sphere/gds_nyc_taxi using MCP tools.

After open_sphere, call sphere_overview() to unlock the full toolset.
The skills are your toolbox — use what fits the task.

This is a NYC Yellow Taxi dataset (January 2019, 7.6M trips). Explore it
and find anomalies. Look for:
- Unusual trip patterns (ghost trips, extreme fares, impossible speeds)
- Zone-level anomalies (dead zones, surge zones, unusual pickup/dropoff patterns)
- Temporal patterns (New Year's Day effects, weekday vs weekend shifts)
- Payment and tip anomalies (cash vs credit patterns, rate code misuse)

After investigating, produce a final report with:
- Findings per category (entity keys, mechanism, confidence)
- Cross-validation between categories
- False positive assessment

Do not stop or ask for guidance — run autonomously to completion.
Save report to benchmark/nyc-taxi/results/0.1.0/<run>/0.1.0-exp-b-report.md.
```

---

### Exp C — Smart mode + skills (production reference)

Not scored — this is the **recommended real-world workflow**. Combines
`detect_pattern` for automated detection with skills for structured reporting
and manual drill-down guidance. Not benchmarked because mixing modes makes
comparison unfair — but this is how a real user should work.

**Instruction:**
```
BEFORE doing anything else, read these skill files completely:
1. packages/hypertopos-skills/gds-analyst/SKILL.md — investigation driver
2. packages/hypertopos-skills/gds-scanner/SKILL.md — advanced detection
3. packages/hypertopos-skills/gds-detective/SKILL.md — detection recipes
4. packages/hypertopos-skills/gds-investigator/SKILL.md — root cause
5. packages/hypertopos-skills/gds-monitor/SKILL.md — temporal monitoring
6. packages/hypertopos-skills/gds-explorer/SKILL.md — orientation

Open the sphere at benchmark/nyc-taxi/sphere/gds_nyc_taxi using MCP tools.

PHASE 1 — RECONNAISSANCE (detect_pattern)
Use the suggested_queries and patterns from open_sphere response.
Run detect_pattern for each suggestion. Read follow_up after each
call and execute them. This gives you a map of the sphere.

PHASE 2 — SKILL-GUIDED DEEP INVESTIGATION
Call sphere_overview() to unlock the full toolset. Using skills as your
guide, systematically investigate EVERY pattern:
- Per pattern: find_anomalies, anomaly_summary, detect_segment_shift, find_clusters
- Temporal: find_regime_changes, find_drifting_entities, compare_time_windows
- Cross-pattern: detect_cross_pattern_discrepancy, passive_scan
- Root cause: explain_anomaly on top findings, assess_false_positive
- Aliases: attract_boundary on each alias

Do NOT stop until every pattern, every temporal window, every segment,
every alias, and every dimension has been checked. The more findings
the better — 10 is not enough, aim for 20+.

PHASE 3 — STRUCTURED REPORT
Produce a final report with:
- Executive summary (3-5 sentences, key message)
- Findings ranked by severity (critical -> high -> medium -> info)
- For each: entity keys, mechanism, dimensions involved, business impact
- Cross-validation table (entities flagged in multiple categories)
- False positive assessment per finding
- Recommendations (what should the data owner do about each finding?)

Do not stop or ask for guidance — run autonomously to completion.
Do NOT delegate to subagents — run all MCP tool calls directly in this session.
Save report to benchmark/nyc-taxi/results/0.1.0/<run>/0.1.0-exp-c-report.md.
```

---

### Key differences

| Aspect | Exp A — Smart mode | Exp B — Manual + skills | Exp C — Production reference |
|--------|-------------------|------------------------|-------------------------------|
| Primary tool | `detect_pattern(query)` | Individual tools (40+) | `detect_pattern` + manual drill-down |
| Step selection | Server decides | Agent decides (skills advise) | Server for detection, agent for drill-down |
| Skills | None | 6 skills read upfront | 6 skills for reporting + drill-down |
| Hints | None ("find anything unusual") | 4 anomaly categories listed | None — skills guide |
| Report quality | Raw findings | Skill-structured | Executive summary + severity ranking + recommendations |

## Evaluation criteria

### Shared criteria

| # | Criterion | 0 | 1 | 2 | 3 |
|---|----------|---|---|---|---|
| 1 | **Anomaly discovery** | 0 categories found | 1-2 categories | 3-4 categories | 5+ distinct anomaly categories with entity examples |
| 2 | **Discovery completeness** | ≤1 pattern explored | 2 patterns | 3 patterns | All patterns + alias + temporal |
| 3 | **Characterization quality** | "Found anomalies" | Listed entity keys | Keys + mechanism (which dimensions) | Keys + mechanism + business interpretation |
| 4 | **Cross-entity reasoning** | Single-entity findings | Zone-level patterns | Zone + trip cross-reference | Multi-pattern triangulation (zone pattern confirms trip anomaly) |
| 5 | **Temporal awareness** | No temporal | Mentioned time | Weekly pattern identified | Specific temporal events (Jan 1, snowstorm) with entity impact |
| 6 | **Query quality** | Vague queries | Targeted but imprecise | Well-targeted, cover 3+ categories | Precise queries with dimension-specific language |
| 7 | **Explanation quality** | Vague | Listed anomalies | Backed by dimensions/metrics | Mechanism + business interpretation + false positive assessment |
| 8 | **Actionable output** | No report | Generic | Specific findings | Prioritized, actionable, includes next steps |
| 9 | **Precision** | Mostly noise | >50% meaningful | >70% meaningful | >90% genuine anomalies with business relevance |
| 10 | **Token efficiency** | >500K tokens | 200-500K | 100-200K | <100K tokens for complete investigation |

### Smart mode criteria

| # | Criterion | 0 | 1 | 2 | 3 |
|---|----------|---|---|---|---|
| 11 | **Smart mode utilization** | Never used detect_pattern | Used once | Primary tool, manual for drill-down | detect_pattern for all categories, refined queries |
| 12 | **Query refinement** | No refinement | Tried 1 refinement | Systematic refinement per category | Iterative queries narrowing from broad to specific |
| 13 | **Dimension awareness** | No dimension-specific queries | Mentioned dimensions | Queries include dimension names | Queries trigger rank_by_property for targeted detection |

### Scoring

| Experiment | Max | Target |
|-----------|-----|--------|
| A — Smart mode | 39 | 28+ |
| B — Manual + skills | 30 | 22+ |

Exp A has 3 extra criteria (smart mode utilization, query refinement, dimension awareness).
Exp B max is 30 (criteria 1-10 only). Skills provide workflow guidance but no smart routing.

## Results

| Version | Experiment | Score | Normalized | Findings | Model | Report | Scoring | Settings |
|---------|-----------|-------|-----------|----------|-------|--------|---------|----------|
| 0.1.0-r1 | A — Smart mode | 26/39 | 67% | 10 | Sonnet 4.6 | [report](results/0.1.0/r1/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r1/0.1.0-exp-a-scoring.md) | [settings](results/0.1.0/r1/0.1.0-settings.md) |
| 0.1.0-r1 | B — Manual + skills | 27/30 | 90% | 10 | Sonnet 4.6 | [report](results/0.1.0/r1/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r1/0.1.0-exp-b-scoring.md) | [settings](results/0.1.0/r1/0.1.0-settings.md) |
| 0.1.0-r2 | A — Smart mode | **31/39** | **79%** | 16 | Sonnet 4.6 | [report](results/0.1.0/r2/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r2/0.1.0-exp-a-scoring.md) | [settings](results/0.1.0/r2/0.1.0-settings.md) |
| 0.1.0-r2 | B — Manual + skills | **28/30** | **93%** | 10 | Sonnet 4.6 | [report](results/0.1.0/r2/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r2/0.1.0-exp-b-scoring.md) | [settings](results/0.1.0/r2/0.1.0-settings.md) |
| 0.1.0-r3 | A — Smart mode | 30/39 | 77% | 15 | Sonnet 4.6 | [report](results/0.1.0/r3/0.1.0-exp-a-report.md) | [scoring](results/0.1.0/r3/0.1.0-exp-a-scoring.md) | [settings](results/0.1.0/r3/0.1.0-settings.md) |
| 0.1.0-r3 | B — Manual + skills | 27/30 | 90% | 12 | Sonnet 4.6 | [report](results/0.1.0/r3/0.1.0-exp-b-report.md) | [scoring](results/0.1.0/r3/0.1.0-exp-b-scoring.md) | [settings](results/0.1.0/r3/0.1.0-settings.md) |

### Exp C — Production reference

| Version | Assessment | Findings | Model | Report |
|---------|-----------|----------|-------|--------|
| 0.1.0-r1 | Zone-centric sweep; 2 CRITICAL (fare fraud, temporal corruption¹); strong dead-zone and airport detection; trip-level data quality gaps (speed cap, ghost vendor, tip encoding missed) | 21 | Sonnet 4.6 | [report](results/0.1.0/r1/0.1.0-exp-c-report.md) |
| 0.1.0-r2 | Trip-first data quality; 4 criticals (fare fraud, 100 mph speed cap, impossible 831-mi trip, non-TLC vendor); temporal artifacts correctly classified; trajectory findings cross-referenced to primary anomalies; tip encoding artifact missed | 21 | Sonnet 4.6 | [report](results/0.1.0/r2/0.1.0-exp-c-report.md) |
| 0.1.0-r3 | Deepest data-quality characterization; 2 CRITICAL (fare fraud + $0.01/4500%-tip encoding artifact in flat-rate trips²); temporal artifacts correctly classified; explicit 15% FP rate; strongest cross-validation | 21 | Sonnet 4.6 | [report](results/0.1.0/r3/0.1.0-exp-c-report.md) |

¹ r1 flagged timestamp range 2008–2088 as CRITICAL data corruption — r2/r3 correctly identified this as a temporal boundary artifact (informational).
² r3 identified that flat-rate trips to zone 265 (Out-of-NYC) systematically record fare=$0.01 with full amount as tip, producing 4,500% tip rates and contaminating zone 265's geometry.

Full 0.1.0 analysis (3 runs, smart mode evolution, Exp C comparison): [`results/0.1.0/SUMMARY.md`](results/0.1.0/SUMMARY.md)

## Building the sphere

```bash
# 1. Download data (if not already present)
curl -L -o benchmark/data/nyc-taxi/yellow_tripdata_2019-01.parquet \
  "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet"
curl -L -o benchmark/data/nyc-taxi/taxi_zone_lookup.csv \
  "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# 2. Prepare data
.venv/Scripts/python benchmark/nyc-taxi/sphere/prepare_trips.py

# 3. Build sphere
hypertopos build --config benchmark/nyc-taxi/sphere/sphere.yaml \
                 --output benchmark/nyc-taxi/sphere/gds_nyc_taxi --force
```
