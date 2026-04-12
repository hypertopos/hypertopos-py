# Introduction

## The problem

Relational databases store entities and their relationships. SQL retrieves specific records that match filters. That works when you know what you're looking for.

But some questions aren't filters:

- Which entities are behaving differently from the rest of the population?
- How is that difference changing over time?
- What structural patterns exist that nobody thought to query for?

These are exploration questions. They require context — not just a row, but how that row relates to every other row in the system. SQL doesn't naturally express this.

## The idea

hypertopos maps every entity in a relational dataset into a shared geometric space.

The coordinate of each entity is derived from its typed relationships — who it connects to, through which channels, how often, which properties are filled or missing. This produces a position vector that captures the entity's structural identity relative to the population.

From there, geometric operations become analytical operations:

- **Distance from center** — anomaly detection
- **Proximity to other entities** — similarity and clustering
- **Movement over time** — drift tracking
- **Group differences** — population comparison

No models are trained. No labels are required. The geometry is computed directly from the relational structure of the data.

## How it works

The process has two phases:

**Build.** You describe your data sources and their relationships in a YAML configuration. hypertopos reads the data, computes population statistics (mean, standard deviation, empirical thresholds per dimension), and produces a sphere — a directory of pre-computed geometry stored in Apache Arrow format.

**Navigate.** An AI agent (or Python code) opens the sphere and moves through it. Twelve navigation primitives cover the common exploratory operations: walking along entity collections, jumping across relationships, finding clusters, detecting outliers, comparing groups, and tracking temporal change. Each step is stateful — where you are determines what you see next.

The sphere is designed for on-demand loading. Opening it reads a few kilobytes of metadata. Everything else is loaded as navigation requires it.

## What you can do with it

- Detect anomalies without training ML models or labeling data
- Discover clusters and structural archetypes in a population
- Track how entities drift from their historical behavior
- Compare groups to find which dimensions differentiate them
- Explore unknown datasets with AI agents that navigate rather than query
- Build investigation workflows that chain geometric operations

## Where this is now

hypertopos is a research-stage project with working code and reproducible benchmarks.

It has been validated on three domains — banking (Berka), anti-money laundering (IBM AML), and transportation (NYC Taxi) — using the same engine with zero domain-specific rules. Benchmark scripts and data preparation are included in the repository.

The library is under active development. The API may change. Contributions and feedback are welcome.

For the mathematical foundation, see [Core Concepts](concepts.md). To try it, see [Quick Start](quickstart.md).
