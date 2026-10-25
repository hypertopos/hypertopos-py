# Quick Start

> From zero to a navigable geometric space.

This guide walks you through building a sphere from a relational dataset, opening it from Python, trying a recalibration on a branch, and connecting it to an AI agent. You will define entity types and their relationships in YAML, compute population-relative coordinates for every entity, and produce a sphere that agents can explore.

For the idea behind this, see [Introduction](introduction.md).

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
        key_on_entity: customer_id
    event_dimensions:
      - column: amount
```

Each order is described by the customer it belongs to and its `amount`. `key_on_entity` names the column on `orders` that points at a customer.

Validate the configuration, then build:

```bash
hypertopos validate --config sphere.yaml
hypertopos build --config sphere.yaml --output my_sphere
hypertopos info my_sphere
```

This produces a sphere on disk — `sphere.json` + geometry + points — ready for navigation.

```
my_sphere/
├── _gds_meta/
│   ├── sphere.json            # population statistics (μ, σ, θ) per pattern
│   ├── calibration/           # the current fit per pattern
│   ├── calibration_history/   # one epoch per build or recalibrate
│   └── branches/              # what-if branches, empty until you create one
├── points/                    # entity data per line (Lance)
└── geometry/                  # delta vectors, delta_norm, edges per pattern (Lance)
```

Event patterns with two or more relations to the same anchor line (sender and receiver accounts, for example) also get an `edges/` adjacency table (Lance, BTREE-indexed) for graph traversal. To skip edge table emission (faster iterative builds), use `--no-edges`. Rebuilding into an existing directory needs `--force`.

## Navigate from Python

Open the sphere, start a session and ask for the most anomalous orders:

```python
from hypertopos import HyperSphere

hs = HyperSphere.open("my_sphere")
with hs.session("analyst") as session:
    nav = session.navigator()

    polygons, total, _, _ = nav.π5_attract_anomaly("order_pattern", top_n=5)
    print(f"{total} anomalous orders")
    for p in polygons:
        print(p.primary_key, round(p.delta_norm, 2))

    top = polygons[0].primary_key
    for d in nav.explain_anomaly(top, "order_pattern")["top_dimensions"]:
        print(d["dim"], d["pct_of_total"])
```

`delta_norm` is the order's distance from the population centre; `explain_anomaly` splits it into each dimension's share of `delta_norm²`.

### Stay on one snapshot

A session reads one consistent snapshot of the sphere — the state current when it opened. An ingest or recalibration by another process does not change its answers. The session asks whether a newer state exists and moves to it only when you say so (this and the next snippet continue inside the `with` block):

```python
    status = session.is_stale()
    if status.stale:
        session.refresh()
```

A session sees its own `recalibrate` immediately, without a refresh.

### Try a recalibration on a branch

A sphere branch is a what-if workspace created without copying data; writes on a branch never touch main. Recalibrate the pattern on a branch without the top anomaly in the baseline — every order is still scored — and compare the branch with main:

```python
    result = session.recalibrate("order_pattern", branch="trial", exclude_keys=[top])
    print(result["n_excluded"], result["old_theta_norm"], result["new_theta_norm"])

    diff = nav.compare_branches("order_pattern", "trial")
    print(diff.flipped_in, diff.flipped_out, diff.jaccard, diff.theta_from, diff.theta_to)
```

`flipped_in` counts entities anomalous on the branch but not on main, `flipped_out` the reverse, and `jaccard` compares the two anomaly sets. The branch is created by the call when it does not exist. A recalibration on a branch leaves main — and this main session — unchanged, so `compare_branches` is where its effect shows.

Then keep the result or drop it. Close the sessions that read the branch first:

```python
hs.promote_branch("trial")   # replay the branch's recalibration on main, then delete the branch
# hs.delete_branch("trial")  # or discard it instead; main is unchanged
```

`hs.branches()` lists the branches of a sphere; the CLI equivalent is `hypertopos sphere branch list my_sphere`.

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

The agent opens it with `open_sphere()` (`HYPERTOPOS_SPHERE_PATH` is the default path) and can then explore the sphere — discover clusters, find outliers, compare populations, track drift — using natural language. The branch workflow above is available to the agent as `recalibrate(pattern_id, branch=..., exclude_keys=[...])`, `compare_branches` and `manage_branch`.

This configuration runs the server over stdio for one local agent. To serve several agents on the same machine, run it over streamable HTTP; each client session keeps its own open sphere, state and call statistics:

```bash
python -m hypertopos_mcp.main --transport http --port 8080
```

The HTTP transport is for local use only: it binds to the loopback interface, has no authentication, and any session can open any sphere path the server process can read. Do not expose it to a network.

For the full MCP tool reference, see **[hypertopos-mcp documentation](https://github.com/hypertopos/hypertopos-mcp)**.

## What's in the sphere

| Component | What it contains |
|-----------|-----------------|
| `sphere.json` | Population statistics: μ (mean), σ (spread), θ (threshold) per pattern |
| `points/` | Entity data per line (Lance, indexed) |
| `geometry/` | Delta vectors — each entity's position in population-relative space |
| `edges/` | Adjacency table per event pattern (`from_key`, `to_key`, `event_key`, `timestamp`, `amount`) — powers runtime graph traversal: `find_geometric_path`, `entity_flow`, `contagion_score`, `propagate_influence`, `cluster_bridges` |

The delta vector `δ = (s − μ) ⊘ σ` is the core coordinate. It tells you where an entity sits relative to its population — basis for clustering, similarity, comparison, hub analysis, boundary exploration, drift tracking, and anomaly detection.

## Next steps

- [Core Concepts](concepts.md) — the GDS mental model and mathematical foundation
- [Architecture](architecture.md) — sessions, snapshots, branches and the MCP layer
- [Configuration](configuration.md) — full YAML reference for sphere building
- [API Reference](api-reference.md) — Python API for programmatic access
- [Data Format](data-format.md) — physical storage layout, including branches
