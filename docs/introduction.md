# Introduction

## The problem

Relational databases store entities and their relationships. SQL retrieves the records that match a filter. That works when you know what you are looking for.

Some questions are not filters:

- Which entities behave differently from the rest of the population?
- How is that difference changing over time?
- Which structural patterns exist that nobody thought to query for?
- Would the picture change if a few suspect entities were left out of the baseline?

These are exploration questions. They need context — not one row, but how that row relates to every other row. SQL does not naturally express this, and an agent working through SQL has to rebuild that context on every question.

## The idea

hypertopos maps every entity in a relational dataset into a shared geometric space.

The coordinate of each entity is derived from its typed relationships — who it connects to, through which channels, how often, which properties are filled or missing — measured against the population it belongs to. The result is a position vector: where the entity sits relative to the population norm, dimension by dimension.

From there, geometric operations become analytical operations:

- **Distance from the centre** — anomaly detection, with the threshold set by the population itself
- **Proximity to other entities** — similarity and clustering
- **Movement over time** — drift tracking
- **Group differences** — population comparison
- **Direction of the offset** — explanation: which named dimensions carry the anomaly, and how much of it each one carries

No models are trained. No labels are required. The geometry is computed directly from the relational structure of the data. hypertopos complements SQL rather than replacing it: the database answers *which rows match*; hypertopos answers *which entities are out of place, how that is changing, and why*.

## How it works

**Build.** You describe your data sources and their relationships in a YAML configuration. hypertopos reads the data, computes population statistics (mean, spread and an empirical threshold per dimension — the calibration), and produces a sphere: a directory of pre-computed geometry stored as Lance datasets. Later batches of data are ingested into the same sphere without a full rebuild.

**Navigate.** An AI agent (or Python code) opens a session on the sphere and moves through it. Twelve navigation primitives cover the common exploratory operations: walking along entity collections, jumping across relationships, finding clusters, detecting outliers, comparing groups, and tracking temporal change. Each step is stateful — where you are determines what you see next. Opening a sphere reads a few kilobytes of metadata; everything else is loaded as navigation requires it.

**Stay consistent.** A session reads one snapshot of the sphere — the state current when it opened. Ingests and recalibrations from other processes do not change its answers mid-investigation. The session can ask whether a newer state exists and move to it when the agent chooses, not before.

**Try things on a branch.** A branch of the sphere is a what-if workspace created without copying data. A recalibration that leaves suspected influencers out of the baseline, or a trial ingest, runs on the branch while main stays untouched. The branch is compared with main entity by entity — who became anomalous, who stopped being anomalous, how far the threshold moved — and is then promoted to main or deleted.

Agents reach all of this through the MCP server, which runs over stdio for a single local agent or over HTTP with an independent session per connected agent. Responses are shaped for an agent's context window: labelled values instead of positional vectors, one properties table per result list, and an explicit note of anything left out.

## What you can do with it

<p align="center">
  <img src="images/demo-drift-solid.svg" width="900"
       alt="find_drifting_entities ranks drift; get_solid replays the top entity's slices back inside θ.">
</p>
<p align="center"><sub>Each slice's distance is its real <code>delta_norm_snapshot</code>.</sub></p>

- Detect anomalies without training ML models or labeling data
- Explain each anomaly in the data's own dimensions, with shares that sum to 100%
- Discover clusters and structural archetypes in a population
- Track how entities drift from their historical behaviour, and in which direction
- Compare groups, time windows, or two calibrations of the same population
- Follow chains of linked events and score structural motifs by geometric rarity
- Test a recalibration or a new batch of data on a branch before it touches main
- Let AI agents explore unknown datasets by navigating rather than querying

## Where this stands

hypertopos is a research-stage project with working code and reproducible benchmarks.

It has been validated on three domains — retail banking, anti-money laundering, and urban transport — using the same engine with zero domain-specific rules. Benchmark scripts and data preparation are included in the repository; see [Benchmarks](benchmarks.md).

The library is under active development. The API may change. Contributions and feedback are welcome.

For the mathematical foundation, see [Core Concepts](concepts.md). To try it, see [Quick Start](quickstart.md).
