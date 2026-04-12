# hypertopos

> Understand the structure of your data — without training machine learning models.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-orange.svg)](LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19482069.svg)](https://doi.org/10.5281/zenodo.19482069)
[![PyArrow](https://img.shields.io/badge/format-PyArrow-red.svg)](https://arrow.apache.org/docs/python/)
[![Lance](https://img.shields.io/badge/storage-Lance-blueviolet.svg)](https://github.com/lance-format/lance)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![Version](https://img.shields.io/badge/version-0.3.1-%235500FF.svg)](pyproject.toml)

hypertopos transforms relational data into a geometric space where every entity gets a coordinate derived from its relationships. Distance from the population center reveals anomalies. Proximity between entities reveals similarity. Movement over time reveals drift. No training, no labels — geometry is computed from the data.

```bash
pip install hypertopos
```

![hypertopos overview](docs/images/hypertopos-overview.svg)

## What you can do with it

- Detect anomalies without training ML models or labeling data
- Discover clusters and structural archetypes
- Track behavioral drift over time
- Compare populations to find what differentiates them
- Explore datasets with AI agents that navigate rather than query

## How it works

You describe your data in YAML — entity types, sources, relationships. hypertopos computes population statistics and produces a sphere: pre-computed geometry stored in Apache Arrow format.

Agents (or Python code) open the sphere and navigate it using twelve primitives that cover movement, clustering, anomaly detection, population comparison, and temporal analysis. Each step is stateful — where you are determines what you see next.

For the full picture: [Introduction](docs/introduction.md) · [Core Concepts](docs/concepts.md) · [Quick Start](docs/quickstart.md)

## Benchmarks

Validated on three domains with the same engine, zero domain rules, zero labels:

| Domain | Dataset | Key result |
|--------|---------|------------|
| Banking | Berka (Czech, real data) | 85.5% recall on loan defaults |
| AML | IBM AML (synthetic) | 80.4% recall, zero labels |
| Transport | NYC Yellow Taxi (7.6M trips) | 8/8 anomaly categories detected |

Benchmark scripts and data preparation are included. Results are reproducible. Numbers are from the pre-0.1.0 validation run and have not been re-evaluated against recent releases.

Full results: [Benchmarks](docs/benchmarks.md)

## Documentation

| | |
|---|---|
| [Introduction](docs/introduction.md) | The idea and where it stands |
| [Quick Start](docs/quickstart.md) | Install, build, navigate |
| [Core Concepts](docs/concepts.md) | Mathematical foundation |
| [Configuration](docs/configuration.md) | Sphere builder YAML reference |
| [API Reference](docs/api-reference.md) | Python API |
| [Data Format](docs/data-format.md) | On-disk storage format |
| [Architecture](docs/architecture.md) | Package layers and design |

## Status

Research-stage project. Working code, reproducible benchmarks, active development. API may change.

## License

[Business Source License 1.1](LICENSE.md). Free for internal use, development, testing, and research. See LICENSE.md for details.
