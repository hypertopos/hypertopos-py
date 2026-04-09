# NYC Taxi — Domain Generalization Test + MCP Calibration (0.1.0)

**Goals:** (1) Test whether geometric navigation generalizes beyond financial data.
(2) Measure MCP interface quality on an unfamiliar domain — cold-start Exp A
performance reflects how well tool descriptions alone guide investigation.
hypertopos was developed on banking (Berka) and AML (IBM) datasets. NYC taxi is the
first non-financial, non-synthetic validation — real operational data with no ground
truth labels. Findings are empirically verifiable in raw TLC data.

**Sphere:** `benchmark/nyc-taxi/sphere/gds_nyc_taxi`
**Dataset:** NYC TLC Yellow Taxi — 7.56M trips, 265 zones, January 2019. No labels, no injection.
**Agent:** Claude Sonnet 4.6 (200K context) | **Scorer:** Claude Opus 4.6 (1M context)
**Runs:** 3 (r1–r3), 2026-03-31
**Sphere version:** 0.1.0

---

## Final Numbers

| Metric | Exp A (smart mode) | Exp B (manual + skills) |
|--------|:------------------:|:----------------------:|
| Best score | **31/39** (r2) | **28/30** (r2) |
| Score range | 26–31 | 27–28 |
| Best normalized | **79%** | **93%** |
| Temporal awareness | 1–2 / 3 | **3/3** (every run) |
| Token range | ~82–91K | ~119–200K+ |
| Tool call range | 24–70 | 71–113 |
| Findings per run | 10–16 | 10–12 |

---

## Score History

| Run | Exp A | Exp B | Findings A | Findings B | What changed |
|-----|:-----:|:-----:|:----------:|:----------:|--------------|
| r1 | 26 | 27 | 10 | 10 | baseline — basic smart mode, standard skills |
| r2 | **31** | **28** | 16 | 10 | +13 smart mode features; skills updated post-r1 |
| r3 | 30 | 27 | 15 | 12 | same smart mode as r2; B skills unchanged |

---

## Per-Criterion Breakdown

### Exp A — Smart mode

| Criterion | r1 | r2 | r3 | Verdict |
|-----------|:--:|:--:|:--:|---------|
| 1. Anomaly discovery | 3 | 3 | 3 | **always 3/3** |
| 2. Discovery completeness | 2 | 3 | 2 | oscillates — vendor_pattern / alias coverage |
| 3. Characterization quality | 3 | 3 | 3 | **always 3/3** |
| 4. Cross-entity reasoning | 3 | 3 | 3 | **always 3/3** |
| 5. Temporal awareness | 1 | 1 | 2 | weak — improved r3 via auto compare_time_windows |
| 6. Query quality | 2 | 3 | 2 | oscillates — speed/distance coverage inconsistent |
| 7. Explanation quality | 3 | 3 | 3 | **always 3/3** |
| 8. Actionable output | 3 | 3 | 3 | **always 3/3** |
| 9. Precision | 3 | 3 | 3 | **always 3/3** |
| 10. Token efficiency | 2 | 2 | 2 | **always 2/3** — sub-100K not confirmed |
| 11. Smart mode utilization | 2 | 3 | 2 | oscillates — r2 covered all categories, r3 regressed |
| 12. Query refinement | 2 | 2 | 2 | **always 2/3** — no iterative deepening within detect_pattern |
| 13. Dimension awareness | 2 | 2 | 2 | **always 2/3** — dimension analysis post-hoc, not query-time |
| **Total** | **26** | **31** | **30** | |

**7 of 13 criteria stable across all 3 runs.** The 3 structurally stuck criteria (10, 12, 13) reflect smart mode architectural limits — query refinement and dimension-targeted routing happen server-side, not via agent iteration.

### Exp B — Manual + skills

| Criterion | r1 | r2 | r3 | Verdict |
|-----------|:--:|:--:|:--:|---------|
| 1. Anomaly discovery | 3 | 3 | 3 | **always 3/3** |
| 2. Discovery completeness | 2 | 3 | 3 | r1 missed alias; r2/r3 covered all patterns + alias |
| 3. Characterization quality | 3 | 3 | 3 | **always 3/3** |
| 4. Cross-entity reasoning | 3 | 3 | 3 | **always 3/3** |
| 5. Temporal awareness | 3 | 3 | 3 | **always 3/3** — skills guide explicit temporal comparison |
| 6. Query quality | 3 | 3 | 3 | **always 3/3** |
| 7. Explanation quality | 3 | 3 | 3 | **always 3/3** |
| 8. Actionable output | 3 | 3 | 2 | r3 lacked explicit remediation steps per finding |
| 9. Precision | 3 | 3 | 3 | **always 3/3** |
| 10. Token efficiency | 1 | 1 | 1 | **always 1/3** — 6 skills + exhaustive scanning = 200K+ |
| **Total** | **27** | **28** | **27** | |

**9 of 10 criteria stable across all 3 runs.** Token efficiency is a structural ceiling: 6 skills loaded upfront + exhaustive tool enumeration per skill guidance is inherently expensive.

---

## Smart Mode Evolution

### r1: Baseline — core features only

**detect_pattern features:** basic routing, no follow-up hints, no auto-temporal, no profiling alerts, no alias boundary.

**Result:** 26/39. Found 10 findings — strong on data integrity (fare fraud, impossible distances, ghost vendor) but missed zone structural patterns and temporal events. Temporal awareness 1/3: no New Year's Day, no snowstorm.

**Gap identified:** smart mode routes to anomaly scans but doesn't automatically follow up with temporal, cluster, or alias steps. Agent stops at first results.

### r2: Major server upgrade — +13 features

**Changes added to detect_pattern:**
- Auto-temporal bonus (temporal step added when temporal hints present)
- Auto-alias boundary (attract_boundary on any available alias)
- Auto-cross-pattern discrepancy
- Follow-up hints after each result
- Profiling alerts at open_sphere time
- Zero-results hint and auto-retry
- Dimension-aware planner
- Multi-pattern scan
- Per-relation keywords
- Coverage tracking
- Query decomposition
- Step summaries

**Score impact:** 26 → **31** (+5 points). Findings: 10 → 16. Discovery completeness 2→3. Query quality 2→3. Smart mode utilization 2→3. Token cost stayed in ~91K range — the intelligence is server-side, not agent-side.

**Temporal still 1/3:** profiling alerts and follow-up hints help find structural anomalies, but keyword-based routing cannot parse "New Year's Day" or understand seasonal demand patterns. Date comparison requires explicit domain knowledge.

### r3: Same features, fewer calls

**Score:** 31 → 30 (-1). Tool calls: 50 → 24. Same smart mode features, same skills.

**Observation:** With 24 calls, the agent still found 15 findings. Coverage dropped slightly (discovery 3→2, smart mode utilization 3→2) but efficiency improved — findings per call ratio was highest in r3. Temporal improved 1→2 (auto compare_time_windows triggered).

---

## What Smart Mode Finds vs Manual Mode

Across 3 runs, the two approaches consistently find different things:

| Category | Exp A (smart) | Exp B (manual) | Notes |
|----------|:---:|:---:|-------|
| Extreme fare outliers | ✓ | ✓ | Both find $623K every time |
| Extreme tip anomalies | ✓ | ✓ | Both find tip corruption |
| Impossible distances / speeds | ✓ (r2,r3) | — | Smart finds via geometric clustering |
| Ghost trips (zero-passenger) | ✓ | ✓ | Both find via aggregation |
| Unregistered vendor (VendorID=4) | ✓ (r1,r2) | — | Smart-only |
| Zone 264 GPS failures (288K) | ✓ | — | Smart geometric detection |
| Billing encoding errors | ✓ | — | Smart via anomaly clustering |
| Ghost zones (inactive) | ✓ | ✓ | Both |
| Airport rate code misuse | ✓ (r2) | ✓ | Skills guide rate code check explicitly |
| Invalid rate code 99 | ✓ (r1) | ✓ | |
| Zone 265 structural outlier | ✓ | ✓ | Both |
| New Year's Day demand drop | — | ✓ | Manual only — requires domain query |
| Fare structure corruption ($1.95) | — | ✓ | Skills guide congestion surcharge check |
| Cash tip zero-inflation | — | ✓ | Skills guide payment type analysis |
| Neighbor contamination | — | ✓ | Skills explicitly include this tool |
| Staten Island structural | ✓ (r3) | ✓ | Geometry catches it; skills confirm |
| Temporal regime change | ✓ (r2,r3) | ✓ | Both find Feb 19 shift |
| Trajectory anomalies (arch/V) | ✓ (r3) | ✓ | |

**A+B combined across runs: 22 unique findings.** Neither mode alone is complete.

**Smart mode specialization:** data integrity anomalies (billing errors, GPS artifacts, timestamp corruption, impossible physics). These surface as geometric outliers — extreme delta_norm — regardless of domain knowledge.

**Manual + skills specialization:** operational patterns requiring domain-aware queries (holidays, rate code semantics, payment type behavior, neighbor contamination). Skills encode the "what to look for" that smart routing doesn't have.

---

## Exp C — Production Reference

Three unscored runs using smart mode + skills together. No numeric target — investigated to tool exhaustion.

### What each run found distinctively

| Finding | r1 | r2 | r3 |
|---------|:--:|:--:|:--:|
| $623K fare (Zone 237) | ✓ | ✓ | ✓ |
| Temporal range 2008–2088 | CRITICAL¹ | INFO | INFO |
| 100 mph speed cap (systematic) | — | ✓ | — |
| 831-mi impossible trip | — | ✓ | — |
| VendorID=4 non-TLC vendor | — | ✓ | — |
| $0.01 / 4,500% tip encoding artifact | — | — | ✓ |
| Ghost zones (103, 110) | ✓ | ✓ | ✓ |
| Airport zones (JFK, LGA) | ✓ | ✓ | ✓ |
| Zone 265 (Out-of-NYC) structural | ✓ | ✓ | ✓ |
| Cross-pattern asymmetry (13 zones) | ✓ | ✓ | ✓ |
| Temporal trajectory anomalies | ✓ | ✓ | ✓ |
| Explicit FP rate | — | — | **15%** |

¹ r1 flagged the synthetic temporal range (2008–2088) as CRITICAL data corruption. r2 and r3 correctly identified it as a sphere construction artifact — informational.

### Per-run character

**r1 — Zone-centric sweep.** Strong on dead-zone detection and airport zone analysis. Weakest on trip-level data quality: speed cap, ghost vendor, and tip encoding missed. One false CRITICAL (temporal window).

**r2 — Trip-first data quality.** Uniquely caught: speed cap at 100 mph as a systematic GPS artifact, the 831-mi impossible trip, and VendorID=4. Temporal artifacts correctly classified. Trajectory findings cross-referenced to primary anomalies. Missed the tip encoding artifact.

**r3 — Deepest characterization.** Uniquely caught: the $0.01/4,500% tip encoding artifact as a second CRITICAL — flat-rate Out-of-NYC trips post full fare as "tip" due to meter encoding logic. The only run that explicitly stated an overall false positive rate (15%) and assessed FP status per finding. Best cross-validation discipline. Missed speed cap and ghost vendor.

### Convergence across runs

No single Exp C run caught everything. Three different agents, same sphere, same instruction — each found different second-order anomalies. The findings that appear in all 3 runs (fare fraud, ghost zones, airport zones, zone 265, cross-pattern asymmetry, trajectory anomalies) are the sphere's geometric ground truth: large enough signal to be inescapable. The run-specific findings (speed cap, ghost vendor, tip encoding) require either lucky query formulation or domain-specific investigation paths.

**Combined Exp C across r1–r3: ~28 distinct findings.** The production setup outperforms any single scored run.

---

## Organic Anomaly Detection — Ground Truth Coverage

| Category | Expected signal | Caught in A | Caught in B | Caught in C |
|----------|----------------|:-----------:|:-----------:|:-----------:|
| Ghost trips (zero passengers) | High delta on passenger dim | ✓ | ✓ | ✓ |
| Fare outliers ($623K, $8K) | rank_by_property fare_amount | ✓ | ✓ | ✓ |
| Speed impossibilities (80+ mph) | speed_mph dim | ✓ (r2) | ✓ (r1) | ✓ (r2) |
| Tip anomalies (cash=0, 4500%) | tip_pct dim, payment relation | ✓ | ✓ | ✓ |
| Dead zones (zero activity) | Zone pattern outliers | ✓ | ✓ | ✓ |
| Surge zones (JFK, LGA, EWR) | Zone pattern, rate_code relation | ✓ | ✓ | ✓ |
| Temporal bursts (New Year's Day) | Temporal drift in zone patterns | — | ✓ | ✓ |
| Rate code abuse (negotiated) | Rate code relation + fare/distance | ✓ (r2) | ✓ | ✓ |

**8/8 organic anomaly categories detected across the full validation** — but no single experiment or run catches all 8.

- **Smart mode misses:** New Year's Day (no temporal domain query), cash tip zero-inflation (requires payment type filter)
- **Manual mode misses:** VendorID=4 (surfaces geometrically but needs fare fraud angle), encoding-level billing errors
- **Exp C catches all:** production setup with both modes + skills covers the full ground truth

---

## Dataset Character — Why NYC Taxi Differs from Berka

Berka is a credit risk dataset: anomalies are rare (11% bad loan rate), ground truth is known, and the detection task is recall-measurable. NYC taxi is unsupervised with organic real-world noise:

| Property | Berka | NYC Taxi |
|----------|-------|----------|
| Ground truth | 76 labeled defaults | None — organic anomalies |
| Anomaly type | Behavioral (stress patterns) | Mixed: data quality + structural + operational |
| Temporal window | 5 years (1993–1998) | 1 month (Jan 2019) |
| Entities | 4,500 accounts | 265 zones |
| Events | 1M transactions | 7.56M trips |
| Scoring | Recall-measurable | Qualitative (no ground truth) |
| Smart mode ceiling | Temporal artifacts in short window | Temporal event detection (no date parsing) |

**The temporal problem is fundamentally different.** In Berka, temporal is about 5-year arc detection — geometric drift is visible. In NYC taxi, temporal is about named events within a single month — New Year's Day, Polar Vortex, congestion surcharge rollout. Smart mode routing cannot parse "January 1" as an anomaly-relevant date from keyword queries alone.

**Precision is naturally higher in NYC taxi.** With 7.56M trips, data quality errors produce enormous geometric signal — a single $623K fare contaminates an entire zone's delta_norm. Geometric anomaly detection is sensitive to these. All 3 smart mode runs achieved precision 3/3 (zero noise).

---

## Conclusions

### Detection

- Smart mode: 26–31/39 (67–79%). Stable on data integrity, variable on operational patterns.
- Manual + skills: 27–28/30 (90–93%). Temporal always 3/3; structural always 3/3. Token cost always high.
- Combined (Exp C): 8/8 organic categories covered across 3 runs. ~28 distinct findings across r1–r3.
- Persistent smart mode gap: temporal event detection requires named-event domain knowledge that keyword routing cannot provide.

### Smart Mode Improvements

- r1→r2: +5 points from server-side intelligence alone (auto-temporal, profiling alerts, follow-up hints). No instruction change.
- The 5-point gap is the clearest evidence that smart mode quality is a function of server-side planning, not prompt engineering.
- Structural ceiling: dimension-targeted routing (criterion 13, always 2/3) requires rank_by_property to be triggered by the planner, not discovered by the agent.

### Skills

- Manual + skills is 2–3x more expensive (~119–200K tokens vs ~82–91K).
- Skills deliver temporal awareness reliably (3/3 every run). This is the one category smart mode cannot replicate without explicit domain event queries.
- Skills update post-r1 (exhaustive enumeration, mandatory clusters, temporal fallthrough) improved B by +1 point (r2: 28/30).
- r3 Exp B lost 1 point on actionable output — "findings + mechanism" without explicit remediation steps per finding. Skills should enforce this.

### Exp C as Production Reference

- No single run found everything — each found different second-order anomalies.
- r3 is the strongest single run: explicit FP rate, unique tip encoding discovery, best cross-validation.
- The tip encoding artifact ($0.01/4,500% tip on flat-rate Out-of-NYC trips) is a real systematic data quality issue — only r3 characterized it correctly as CRITICAL.
- r1's temporal CRITICAL (2008–2088 window) is a false positive — demonstrates that production setup still requires informed interpretation.

### Limitations

- No labeled ground truth — anomaly category coverage is the proxy for recall.
- Single-month dataset limits temporal analysis — Jan 2019 only provides New Year's Day as a named event.
- Sphere construction creates synthetic temporal range (2008–2088). All temporal findings must be interpreted against this artifact.
- Validation is self-authored: sphere design, rubric, scoring by the same author.
- NYC taxi is a well-known public dataset. Agent (Sonnet 4.6) may carry domain knowledge (congestion surcharge rollout date, airport rate code semantics).

---

## File Index

| Run | Settings | Exp A Report | Exp A Scoring | Exp B Report | Exp B Scoring | Exp C Report |
|-----|----------|-------------|--------------|-------------|--------------|--------------|
| r1 | [settings](r1/0.1.0-settings.md) | [report](r1/0.1.0-exp-a-report.md) | [scoring](r1/0.1.0-exp-a-scoring.md) | [report](r1/0.1.0-exp-b-report.md) | [scoring](r1/0.1.0-exp-b-scoring.md) | [report](r1/0.1.0-exp-c-report.md) |
| r2 | [settings](r2/0.1.0-settings.md) | [report](r2/0.1.0-exp-a-report.md) | [scoring](r2/0.1.0-exp-a-scoring.md) | [report](r2/0.1.0-exp-b-report.md) | [scoring](r2/0.1.0-exp-b-scoring.md) | [report](r2/0.1.0-exp-c-report.md) |
| r3 | [settings](r3/0.1.0-settings.md) | [report](r3/0.1.0-exp-a-report.md) | [scoring](r3/0.1.0-exp-a-scoring.md) | [report](r3/0.1.0-exp-b-report.md) | [scoring](r3/0.1.0-exp-b-scoring.md) | [report](r3/0.1.0-exp-c-report.md) |
