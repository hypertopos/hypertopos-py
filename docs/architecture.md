# Architecture

> Compact orientation to the hypertopos package: layers, runtime flow, and component responsibilities.

hypertopos implements the Geometric Data Sphere (GDS) concept.

For the conceptual model, see [concepts.md](concepts.md). For API details, see
[api-reference.md](api-reference.md). For the physical storage format, see
[data-format.md](data-format.md). For getting started, see [quickstart.md](quickstart.md).

## Navigation Primitives

Navigation is expressed through a small set of primitives rather than direct query strings.
The point is not to replace every analytical operation with a custom DSL, but to make the
most common exploratory moves explicit, stateful, and composable.

The `π1`–`π12` family covers movement (π1–π4), population-level attraction (π5–π8), and temporal analysis (π9–π12). Navigation is stateful — each step depends on the agent's current position. Primitives are intentionally small and composable: chain them to build investigation workflows.

For the full list with method signatures and return types, see [API Reference — Navigation Primitives](api-reference.md#navigation-primitives).

## Package Layout

The core package is split into a few layers:

- **Model**: sphere, line, pattern, alias, point, edge, polygon, solid, manifest, contract
- **Storage**: readers, writers, caches, and physical layout handling
- **Engine**: geometry, anomaly scoring, similarity, drift, forecasting, aggregation (Lance SQL pushdown via `lance_sql_agg`), calibration tracking
- **Navigation**: stateful step-by-step movement through the sphere
- **PassiveScanner**: multi-source batch screening over navigation primitives
- **Builder**: transforms source data into a complete sphere
- **CLI**: builds and inspects spheres from the command line

The `hypertopos-mcp` package sits on top of this core library and exposes the same capabilities to agents as MCP tools, over stdio or streamable HTTP.

The split is deliberate:

- the **model** defines the vocabulary of the system, including the Manifest/Contract pair that records which line and pattern versions a session reads
- **storage** owns physical reads and writes on Lance datasets; every read goes through the session's snapshot pins, and sphere branches live here
- the **engine** computes derived geometry and analysis, with CalibrationTracker for online drift monitoring
- **navigation** orchestrates stateful movement through that geometry
- **PassiveScanner** runs batch screening across multiple geometric sources without manual primitive chaining
- the **builder** performs ingestion-time work that should not happen on every session open
- the **CLI** is an operational entry point for humans and automation

This keeps expensive work in the build phase and keeps the runtime path focused on reading, navigating, and summarizing already-built data.

## Physical Layout

A sphere is stored as a directory tree. The exact storage rules are defined by the core code and package configuration, but the mental model is:

- metadata describes the sphere
- points store source entities
- geometry stores derived vectors and anomaly labels
- edges store entity-to-entity relationships (Lance, BTREE indexed) for graph traversal and chain discovery
- temporal data stores shape history
- auxiliary caches store precomputed summaries
- branch metadata (`_gds_meta/branches/<name>/`) records each sphere branch: its provenance, its copy of the metadata files and its recipe log

Every dataset is a Lance dataset. Lance versions every commit, which is what makes snapshot sessions and copy-free branches possible: a session pins versions instead of copying data, and a branch is a set of Lance references and recorded versions instead of a second copy of the sphere.

The important point is that the system is designed for on-demand loading. The session does not read every file at startup.

Most of the package logic assumes this storage model:

- the sphere can be opened quickly from metadata
- derived data is read lazily when a primitive needs it
- repeated reads should benefit from cache and precomputed summaries
- build-time artifacts should be stable enough that sessions do not need to recompute them

That means the architecture is optimized for repeated inspection of a built sphere, not for re-deriving the full dataset on every request.

For the full directory structure and Arrow schemas, see [data-format.md](data-format.md).

## Level Design

The package architecture is easier to read when viewed as a few stacked layers:

```mermaid
flowchart TB
    A[Source data<br/>tables / DataFrames] --> B[Builder]
    B --> C[Sphere storage]
    C --> D[Core model<br/>Sphere, Line, Pattern, Polygon, Solid, Manifest]
    D --> E[Engine<br/>geometry, scoring, analysis, calibration]
    E --> F[Navigation<br/>stateful primitives]
    F --> PS[PassiveScanner<br/>batch screening]
    F --> G[MCP tools]
    PS --> G
    G --> H[Agent]
```

The runtime flow is similarly layered:

```mermaid
sequenceDiagram
    participant A as Agent
    participant M as MCP
    participant S as Session
    participant N as Navigator
    participant E as Engine
    participant R as Reader

    A->>M: open_sphere()
    M->>S: create session
    A->>M: navigation or analysis tool
    M->>N: execute primitive
    N->>R: read physical data
    N->>E: compute derived geometry
    E-->>N: polygon / solid / summary
    N-->>M: formatted result
    M-->>A: JSON response
```

These diagrams are intentionally high-level. They are meant to show the shape of the system, not the implementation details.

For a denser top-down view, this stack is a useful shorthand:

```text
┌─────────────────────────────────────┐
│  MCP Server (hypertopos-mcp)        │  stdio / HTTP, smart detection mode
├─────────────────────────────────────┤
│  PassiveScanner                     │  multi-source batch screening
├─────────────────────────────────────┤
│  Navigation (navigator.py)          │  π1–π12 primitives + edge table graph traversal
│                                     │  (find_geometric_path, extract_chains, entity_flow,
│                                     │   contagion_score, propagate_influence, cluster_bridges, ...)
├─────────────────────────────────────┤
│  Engine (geometry.py)               │  delta vectors, metrics, scoring, clustering
│  CalibrationTracker                 │  online Welford drift stats
├─────────────────────────────────────┤
│  Storage (reader.py, writer.py)     │  Lance storage (points, geometry, edges, temporal),
│                                     │  PyArrow transport, partition filtering, BTREE edge indexes
│  Snapshot pins, sphere branches     │  per-session Lance version pins + metadata;
│                                     │  copy-free branches with a recipe log
├─────────────────────────────────────┤
│  Model (sphere.py, objects.py)      │  Point, Edge, Polygon, Solid, Pattern, Alias
├─────────────────────────────────────┤
│  Builder (builder.py)               │  YAML config → sphere build
└─────────────────────────────────────┘
```

This view is intentionally compact. It shows the execution stack from ingestion at the bottom to agent-facing tools at the top.

## Core Model Map

```mermaid
flowchart LR
    Sphere --> Line
    Sphere --> Pattern
    Sphere --> Alias
    Sphere --> Manifest
    Line --> Point
    Point --> Polygon
    Polygon --> Edge
    Polygon --> Solid
    Solid --> SolidSlice
    Manifest --> Contract
```

A session's Manifest records the line and pattern versions of the snapshot it reads; its
Contract lists the patterns that snapshot offers. The snapshot itself — which Lance version of
each dataset the session reads — is held by the session's reader (see [MVCC Model](#mvcc-model)).

```mermaid
flowchart TB
    Point[Point]
    Polygon[Polygon]
    Solid[Solid]

    Point -->|current shape| Polygon
    Polygon -->|time expands into| Solid
    Solid -->|slice_at| Polygon
```

## Responsibility Split

```mermaid
graph LR
    B["Builder\nYAML → Sphere"] --> ST["Storage\nLance read/write"]
    ST --> E["Engine\ngeometry, delta,\nclustering, DTW"]
    E --> N["Navigator\nπ1–π12, detect_*"]
    N --> PS["PassiveScanner\nmulti-source\nbatch screening"]
    N --> MCP["MCP Server\nstdio / HTTP transport"]
    MCP --> Agent["AI Agent"]
    CT["CalibrationTracker\nonline drift\nWelford stats"] -.-> E
    M["Snapshot pin\nper session"] -.-> ST
    BR["Sphere branches\ncopy-free what-if"] -.-> ST
```

## MVCC Model

Multiple agents can open sessions against the same sphere concurrently. A session reads one
consistent snapshot of the sphere — the state current when it opened — while ingests and
recalibrations from other processes commit new Lance versions underneath it.

**What is pinned.** `HyperSphere.session(agent_id)` gives the session its own reader and pins,
in one step, the latest Lance version of the geometry, temporal and edge datasets and of the
derived sidecar datasets (trajectory index, temporal centroids, contagion stats, edge
features, population forecasts), together with the sphere metadata files under `_gds_meta`
(`sphere.json`: pattern statistics, line version lists). The versions are read before and
after the metadata; when a writer commits in between, the snapshot is retaken. Every Lance
read in the storage layer goes through one resolver that applies the pin — scans, the Lance
SQL pushdowns that large patterns use for anomaly search and aggregation, vector-neighbour
search, entity-key lookups and temporal history. Points are versioned by directory, and the
session's manifest records which line version it reads.

**Who repins.** Nothing moves a session automatically. `HyperSession.is_stale()` returns a
`SnapshotStatus` (`pinned_at`, `stale`, `metadata_changed`, `pinned_versions`,
`latest_versions`) that reports whether a newer state exists, a rebuilt sidecar included;
`HyperSession.refresh()` drops everything read under the old snapshot and pins the latest
state. A session sees its own `recalibrate` immediately. In the MCP server,
`get_session_stats` carries a `snapshot` block and `open_sphere(path)` reopens on the latest
state.

```mermaid
sequenceDiagram
    participant A1 as Agent A
    participant A2 as Agent B
    participant S as Sphere

    A1->>S: session("agent-a") pins v3
    Note over S: an ingest commits v4
    A2->>S: session("agent-b") pins v4
    A1->>S: read geometry (sees v3)
    A2->>S: read geometry (sees v4)
    A1->>S: is_stale() → stale, then refresh() pins v4
```

![Sessions and branches](images/sessions-and-branches.svg)

*Sessions never change under you; branches never touch main.*

## Sphere Branches

A sphere branch is a copy-free what-if workspace: a recalibration without suspected
influencers, or a trial ingest, runs on the branch while main stays as it is. Writes on a
branch never touch main; the default ref is `main`.

**Layout.** `HyperSphere.create_branch(name)` copies no data file:

- the datasets a branch write can rewrite — geometry and temporal centroids — get a native
  Lance branch named after the sphere branch, created from their latest version;
- every other dataset (points, temporal, edges, trajectory, other sidecars) is pinned at the
  version current when the branch was created;
- `_gds_meta/branches/<name>/` holds a copy of every metadata file and `branch.json` —
  provenance, the pinned versions and the recipe log.

A full build is the only writer of the pinned datasets, so a build into the sphere directory
(`hypertopos build`, `GDSBuilder.build()`) is refused while the sphere has branches.

**Lifecycle.** A branch is read with `HyperSphere.open(path, branch=name)` or
`HyperSphere.session(agent_id, branch=name)`; those sessions are pinned like any other.
`HyperSession.recalibrate(pattern_id, branch=name, exclude_keys=[...])` writes the new fit on
the branch (creating it when absent) and appends a recipe to its log;
`GDSBuilder.incremental_update(..., branch=name)` ingests into the branch only.
`GDSNavigator.compare_branches(pattern_id, branch)` returns a `BranchDiffReport` — anomaly
counts on both sides, `flipped_in` / `flipped_out`, `jaccard`, `top_movers` and the
calibration drift between the two fits. `HyperSphere.promote_branch(name)` replays the
branch's recipes on main, then deletes the branch; it is refused unless main is still at the
state the branch was created from and every branch commit has a recipe, and each replayed
recalibration must reproduce the branch's fit exactly. `HyperSphere.delete_branch(name)`
discards it. Delete and promote are refused while a session in the same process reads the
branch. The CLI exposes the same operations as `hypertopos sphere branch
{list,create,delete,promote}` and `hypertopos sphere ingest --branch`.

## PassiveScanner

PassiveScanner provides multi-source batch screening on top of the navigation primitives.
Instead of manually chaining individual attract calls, an agent can request a scan across
multiple geometric sources in a single operation. The scanner draws candidates from five
source types:

- **geometry** — anomaly scores from delta vectors (via attract_anomaly)
- **borderline** — boundary proximity scores (via attract_boundary)
- **points** — direct point-level attribute filters
- **compound** — combinations of the above, with configurable weighting
- **graph** — anomaly contagion through the edge table (an entity is flagged when its
  graph neighborhood contains anomalous counterparties beyond a configurable threshold)

The scanner collects, deduplicates, and ranks results into a unified candidate list. This
is the primary entry point for broad screening workflows where the agent does not yet know
which specific primitive to focus on. `auto_discover()` registers the appropriate sources
automatically — graph sources are added for every event pattern that has an edge table.
Callers that do not need graph contagion signal (e.g. `detect_cross_pattern_discrepancy`,
which measures geometry disagreement between patterns, not neighbor contamination) pass
`include_graph=False` to skip graph source registration and avoid the per-pattern edge
table read that would otherwise dominate scan latency on large spheres.

## Build Flow

Building a sphere is an ingestion-time task.

At a high level, the builder:

1. reads structured source tables or DataFrames
2. maps them into lines and patterns
3. computes geometry and population statistics
4. writes the physical datasets
5. builds indexes and auxiliary summaries
6. stores the result as a self-contained sphere

This is where expensive work should happen. The runtime session should mostly read already-prepared data.

In practice, the builder is the place where the package turns raw source data into a reusable geometric asset. If something can be computed once and reused many times, it belongs here rather than in the navigation path.

After the first build, later batches arrive through `GDSBuilder.incremental_update` (CLI `hypertopos sphere ingest`), and `HyperSession.recalibrate` refits a pattern's population statistics on the stored shapes. Both commit new Lance versions; open sessions keep reading their snapshot. A rebuild and a recalibration each record a calibration epoch, so the calibration tools can compare the fit before and after either one.

## Session and Navigation

A session is the agent's working context: open a sphere → create a session (pinned to a snapshot of main or of a branch) → navigate → refresh when the agent chooses → close.

Sessions are stateful. The agent starts with a position and moves from there — jump to an entity, walk a line, inspect a polygon, dive into temporal history, compare populations. Primitives compose: some move the position, some inspect it, some expand it into a population-level view.

Opening a session is lightweight (no rebuild, no data copied): it pins versions and reads a few metadata files. Navigators created from a session share its reader, so they read the same snapshot and follow it on `refresh()`. The session is the working memory of the investigation flow.

## Engine Responsibilities

The engine layer computes the derived signals that make the sphere useful.

Typical responsibilities include:

- geometry construction from population statistics
- clustering and archetype discovery
- similarity search
- anomaly scoring and threshold calibration
- hub and connectivity analysis
- temporal drift and regime analysis
- population comparisons
- aggregation over geometric structure
- **online calibration tracking** via CalibrationTracker, which maintains running Welford
  statistics for drift monitoring without requiring a full population rescan

These computations are what make the GDS representation more than just a storage format.

The engine should stay focused on derived meaning, not on transport or CLI concerns. If a result can be expressed as geometry, score, comparison, or summary, that is usually engine territory.

## MCP Role

The MCP package is a transport and tool layer.

It does not define the core GDS model. Instead, it:

- opens and closes spheres, on main or on a branch (`open_sphere(path, branch=...)`), holding one `HyperSession` per client session
- exposes the navigation, analysis and branch operations as tools (`manage_branch`, `compare_branches`, `recalibrate(..., branch=..., exclude_keys=...)`)
- formats results as JSON in a compact response format: labelled `{dim: value}` maps, one `properties` table per entity list, and an `omitted` map naming what was left out and how to get it (`open_sphere(response_format="full")` turns it off)
- manages session state for an agentic client

The server runs over stdio for one local agent, or over streamable HTTP
(`python -m hypertopos_mcp.main --transport http`, with `--port`, `--session-idle-timeout` and
`--max-sessions`). Over HTTP each client session — identified by the `Mcp-Session-Id` the server assigns —
keeps its own open sphere, tool phases and call statistics,
and calls from different sessions run concurrently.

The MCP layer should stay thinner than the core library. Most behavior belongs in `hypertopos`, not in the transport wrapper.

That boundary matters for maintainability:

- core changes should not force transport logic to become aware of storage internals
- transport changes should not leak into the geometric model
- tool outputs should be formatted results, not a second implementation of the domain logic

The MCP layer is valuable because it exposes the model cleanly, not because it redefines it.

## Design Principles

- **Population first**: start with the sphere before drilling down to entities
- **Geometry over rows**: represent structure as vectors, thresholds, and boundaries
- **Persist derived signals**: expensive work should be computed once and reused
- **Keep sessions stateful**: navigation should build on prior context
- **Separate core from transport**: MCP is a wrapper, not the place for core logic
- **Prefer readable defaults**: the system should be useful before the user learns every primitive
- **Version isolation**: concurrent sessions must not interfere with each other

Two more rules are worth keeping in mind:

- **Make the runtime path cheap**: sessions should mostly read, not build
- **Prefer explicit steps over hidden orchestration**: the system should be understandable from the primitives it exposes

## What This Document Is Not

This page is not:

- a full API reference (see [api-reference.md](api-reference.md))
- a storage-format specification (see [data-format.md](data-format.md))
- a getting-started guide (see [quickstart.md](quickstart.md))
- a list of every module in the package
- a detailed design-decision archive

It is a compact orientation document for the package.

## Short Version

hypertopos is a stateful geometric analytics system.

It turns business data into a sphere, preserves the derived structure needed for navigation, and gives agents a way to explore that structure step by step instead of flattening everything into a single query result.
