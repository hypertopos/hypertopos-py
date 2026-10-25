# hypertopos Documentation

hypertopos turns relational data into navigable geometric space — a coordinate system where structure, anomalies, and change become directly observable. A sphere is built once from YAML and stored as Lance datasets; agents open it through the MCP server or Python, each session reads one consistent snapshot, and what-if work happens on copy-free sphere branches.

**New here?** Read the [Introduction](introduction.md) for the idea, then [Quick Start](quickstart.md) to try it.

**Want depth?** [Core Concepts](concepts.md) covers the mathematical foundation, objects, calibration, and vocabulary; [Architecture](architecture.md) covers how sessions, snapshots and branches work.

**Building something?** Jump to the reference docs below.

## Documents

| Document | What it covers |
|----------|----------------|
| [Introduction](introduction.md) | The idea, how it works, where it stands |
| [Quick Start](quickstart.md) | Install, build a sphere, navigate from Python, try a branch, connect an agent |
| [Core Concepts](concepts.md) | Mathematical foundation, objects, population model, calibration epochs, snapshots and branches |
| [Architecture](architecture.md) | Package layers, snapshot sessions, sphere branches, MCP layer (stdio / HTTP) |
| [Configuration](configuration.md) | Sphere builder YAML reference |
| [API Reference](api-reference.md) | Python API — classes, methods, errors |
| [Data Format](data-format.md) | On-disk layout, Lance schemas, branches, sphere.json |
| [External Chains as Anchor Lines](external-chains-as-anchor-line.md) | Ingesting chains found by other systems and navigating them with the same primitives |
| [Container Deployment](container-deployment.md) | Running the CLI as one-shot container jobs |
| [Benchmarks](benchmarks.md) | Validation results — banking, AML, transportation |
