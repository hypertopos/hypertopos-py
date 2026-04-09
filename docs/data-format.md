# Physical Data Format

> How hypertopos stores geometric data on disk: directory layout, sphere.json config, Arrow schemas, and what gets read when.

---

## Directory Layout

```mermaid
mindmap
  root((Sphere))
    _gds_meta/
      sphere.json — config & stats
      geometry_stats/ — precomputed summaries
      trajectory/ — ANN index
      temporal_centroids/ — centroid cache
    points/
      line_id/v=n/ — entity data per line
      Lance with BTREE + FTS indices
    geometry/
      pattern_id/v=n/ — delta vectors & edges
      Lance with IVF-PQ ANN index for similarity
    temporal/
      pattern_id/ — deformation history
      shape snapshots over time
```

Each sphere is a self-contained directory:

```
gds_{sphere_id}/
├── _gds_meta/
│   ├── sphere.json              # central config (lines, patterns, aliases, storage)
│   ├── geometry_stats/          # precomputed population summaries
│   ├── trajectory/              # ANN index for trajectory similarity search
│   └── temporal_centroids/      # cached population centroids per time window
├── points/
│   ├── {line_id}/v={n}/
│   │   └── data.lance           # entity records for this line version
│   └── ...
├── geometry/
│   ├── {pattern_id}/v={n}/
│   │   └── data.lance           # delta vectors and edges for this pattern version
│   └── ...
└── temporal/
    ├── {pattern_id}/
    │   └── data.lance           # shape/delta snapshots over time
    └── ...
```

All data files use [Lance](https://github.com/lancedb/lance) format (columnar, versioned, with native ANN index support). The `v={n}` directories are Hive-style version partitions.

---

## sphere.json

The central config file. Loaded once on `open_sphere` (typically a few KB). Contains everything the agent needs to understand the sphere without touching any data files.

| Field | Type | Description |
|-------|------|-------------|
| `sphere_id` | string | Unique sphere identifier |
| `lines` | dict | Line definitions: versions, columns, partition config, descriptions |
| `patterns` | dict | Pattern stats: `mu`, `sigma_diag`, `theta`, `edge_max`, `dimension_weights`, `group_stats` |
| `aliases` | dict | Alias definitions: `base_pattern`, cutting plane (`normal` vector, `bias`) |
| `storage` | dict | Storage config per layer (format, partition mode) |

Example (abridged pattern entry):

```json
{
  "patterns": {
    "tx_pattern": {
      "pattern_id": "tx_pattern",
      "entity_line": "transactions",
      "pattern_type": "event",
      "version": 1,
      "status": "production",
      "relations": [
        {"line_id": "accounts", "direction": "in", "required": true},
        {"line_id": "tx_types", "direction": "in", "required": true}
      ],
      "mu": [1.0, 1.0, 0.827, 0.259, 0.130, 0.356],
      "sigma_diag": [0.01, 0.01, 0.379, 0.437, 0.209, 0.204],
      "theta": [1.284, 1.284, 1.284, 1.284, 1.284, 1.284],
      "population_size": 1056320
    }
  }
}
```

`mu` is the population mean per dimension, `sigma_diag` is the standard deviation used for z-scoring, and `theta` is the per-dimension anomaly threshold vector (entities whose z-scored delta exceeds theta on any dimension are flagged).

---

## Arrow Schemas

### Points (`points/{line_id}/v={n}/data.lance`)

| Column | Type | Description |
|--------|------|-------------|
| `primary_key` | string | Unique entity identifier |
| *(domain columns)* | various | Business data (name, amount, date, etc.) |

Domain columns vary per line. The `primary_key` column is always present and always string-typed.

### Geometry (`geometry/{pattern_id}/v={n}/data.lance`)

| Column | Type | Description |
|--------|------|-------------|
| `primary_key` | string | Entity identifier |
| `delta` | fixed_size_list\<float32\> | Z-scored delta vector (deviation from mu) |
| `delta_norm` | float32 | L2 norm of delta (distance from population center -- used for scoring, clustering, and similarity) |
| `delta_rank_pct` | float32 | Percentile rank of delta_norm (0--100) |
| `edges` | list\<struct\> | Edge list: line_id, point_key, direction, status |

The `delta` vector length equals the number of dimensions in the pattern. Geometry datasets carry an IVF-PQ ANN index on the `delta` column for trajectory similarity search.

### Temporal (`temporal/{pattern_id}/data.lance`)

| Column | Type | Description |
|--------|------|-------------|
| `primary_key` | string | Entity identifier |
| `shape_snapshot` | fixed_size_list\<float32\> | Shape vector at this point in time |
| `delta_snapshot` | fixed_size_list\<float32\> | Delta vector at this point in time |
| `timestamp` | timestamp | When this snapshot was taken |
| `deformation_type` | string | How the shape changed (`internal` / `edge` / `structural`) |

Each row is one entity at one point in time. Multiple rows per entity form the temporal solid.

---

## Data Flow

What gets read at each stage of navigation:

```mermaid
sequenceDiagram
    participant Agent
    participant sphere.json
    participant geometry/
    participant points/
    participant temporal/

    Agent->>sphere.json: open_sphere (few KB)
    Note over Agent: Has all patterns, stats, thresholds

    Agent->>geometry/: π5 attract_anomaly
    Note over geometry/: delta vectors, delta_norm, edges

    Agent->>points/: goto("CUST-001")
    Note over points/: entity properties, raw data

    Agent->>temporal/: π3 dive_solid
    Note over temporal/: shape_snapshot[], timestamp[]
```

The key principle: the agent reads only `sphere.json` on startup. Everything else loads on-demand during navigation. A session that never inspects temporal data never touches `temporal/`.

---

## Version Lifecycle *(concept — not yet finalized)*

```mermaid
stateDiagram-v2
    [*] --> prerelease: build starts
    prerelease --> production: build completes
    production --> deprecated: new version built
    deprecated --> orphaned: GC after grace period
```

- **prerelease** -- data is being written; not yet available for navigation.
- **production** -- active version; all navigation reads from this version.
- **deprecated** -- superseded by a newer version; kept for grace period.
- **orphaned** -- no longer referenced; eligible for garbage collection after the grace period.

Only one version per line or pattern is in `production` at any time.

---

## See Also

- [concepts.md](concepts.md) -- core objects, geometry vocabulary, population statistics
- [configuration.md](configuration.md) -- YAML builder reference for defining spheres
- [api-reference.md](api-reference.md) -- Python API and navigation primitives
