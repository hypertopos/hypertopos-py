# Quick Start

> Build a sphere from your data, then let AI agents navigate it.

## Install

```bash
pip install hypertopos
```

## Build Your First Sphere

Create a `sphere.yaml` that describes your data:

```yaml
version: "0.1.0"
sphere_id: my_sphere

sources:
  customers:
    path: customers.csv
  orders:
    path: orders.csv

lines:
  customers:
    source: customers
    key: customer_id
    role: anchor
  orders:
    source: orders
    key: order_id
    role: event

patterns:
  order_pattern:
    type: event
    entity_line: orders
    relations:
      - line: customers
        direction: in
```

Build:

```bash
hypertopos build sphere.yaml --output my_sphere
```

This produces a sphere on disk — `sphere.json` + geometry + points — ready for navigation.

```
my_sphere/
├── _gds_meta/sphere.json    # population statistics (μ, σ, θ)
├── points/                   # entity data (Lance)
├── geometry/                 # delta vectors, edges (Lance)
└── edges/                    # adjacency table per event pattern (Lance, BTREE-indexed)
```

To skip edge table emission entirely (faster iterative builds), use `--no-edges`.

## Validate (optional)

```bash
hypertopos validate sphere.yaml     # check config without building
hypertopos info my_sphere/          # inspect a built sphere
```

## Navigate with AI agents

The sphere is designed to be navigated by AI agents via MCP. Install the MCP server:

```bash
pip install hypertopos-mcp
```

Configure your MCP client (e.g. Claude Desktop):

```json
{
  "mcpServers": {
    "hypertopos": {
      "command": "python",
      "args": ["-m", "hypertopos_mcp.main"],
      "env": {
        "HYPERTOPOS_SPHERE_PATH": "my_sphere"
      }
    }
  }
}
```

Your agent can now explore the sphere — discover clusters, find outliers, compare populations, track drift — using natural language.

For the full MCP tool reference, see **[hypertopos-mcp documentation](https://github.com/hypertopos/hypertopos-mcp)**.

## What's in the sphere

| Component | What it contains |
|-----------|-----------------|
| `sphere.json` | Population statistics: μ (mean), σ (spread), θ (threshold) per pattern |
| `points/` | Entity data per line (Lance, indexed) |
| `geometry/` | Delta vectors — each entity's position in population-relative space |
| `edges/` | Adjacency table per event pattern (`from_key`, `to_key`, `event_key`, `timestamp`, `amount`) — powers runtime graph traversal: `find_geometric_path`, `discover_chains`, `entity_flow`, `contagion_score`, `propagate_influence`, `cluster_bridges` |

The delta vector `δ = (s − μ) ⊘ σ` is the core coordinate. It tells you where an entity sits relative to its population — basis for clustering, similarity, comparison, hub analysis, boundary exploration, drift tracking, and anomaly detection.

## Next steps

- [Core Concepts](concepts.md) — the GDS mental model and mathematical foundation
- [Configuration](configuration.md) — full YAML reference for sphere building
- [API Reference](api-reference.md) — Python API for programmatic access
- [Data Format](data-format.md) — physical storage layout
