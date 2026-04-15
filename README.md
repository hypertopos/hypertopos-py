# hypertopos

> A behavioral feature layer for graph and temporal data — turning behavior into coordinates, trajectories, and explanations.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-orange.svg)](LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19482069.svg)](https://doi.org/10.5281/zenodo.19482069)
[![PyArrow](https://img.shields.io/badge/format-PyArrow-red.svg)](https://arrow.apache.org/docs/python/)
[![Lance](https://img.shields.io/badge/storage-Lance-blueviolet.svg)](https://github.com/lance-format/lance)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![Version](https://img.shields.io/badge/version-0.4.0-%235500FF.svg)](pyproject.toml)

hypertopos is not a database, and not a machine learning model. It is a layer that turns relational data into a coordinate system where every entity gets a position derived from its relationships and the population around it.

```
typical:    data → storage → features → feature store → ML → serving → decision
hypertopos: data → hypertopos (WIP) → ML
```

```bash
pip install hypertopos
```

![hypertopos overview](docs/images/hypertopos-overview.svg)

## How it works

You describe your data in YAML — entity types, sources, relationships. hypertopos computes population statistics and produces a sphere: pre-computed geometry stored in Apache Arrow format.

Agents (or Python code) open the sphere and navigate it using twelve primitives that cover movement, clustering, anomaly detection, population comparison, and temporal analysis. Each step is stateful — where you are determines what you see next.

For the full picture: [Introduction](docs/introduction.md) · [Core Concepts](docs/concepts.md) · [Quick Start](docs/quickstart.md)

## What's different

Each capability below emerges from treating entities as points in a shared, population-calibrated space.

| Capability | What it does | Why it matters | What others do instead | Since |
|---|---|---|---|---|
| Population-relative coordinates | Represents each entity as its deviation from the population (in sigma units). `delta = (shape - mu) / sigma`, computed in one aggregation pass. | Anomaly detection, clustering, and comparison all work on the same coordinate — no separate pipelines. | node2vec/GNN: learned embeddings, latent dims, retraining on shift. Feature stores: static columns, no population calibration. | `0.1.0` |
| Self-calibrating threshold | Anomaly when `\|\|delta\|\| > theta`, where `theta = percentile(\|\|delta\|\|, 95)`. Adapts to any population shape. | Works on banking, logistics, healthcare without parameter tuning or labeled data. | PyOD/Isolation Forest: fit model, choose contamination rate. Sigma rules: assume Gaussian. | `0.1.0` |
| Named dimension attribution | `explain_anomaly` ranks dimensions by contribution: `loan_count: +3.2 sigma`, `has_email: -1.8 sigma`. Contributions sum to the total anomaly score. | Every anomaly is immediately interpretable — analyst sees which structural property drives it. | SHAP/LIME: explain a model's prediction on latent features. Here there is no model — dimensions ARE the explanation. | `0.1.0` |
| Temporal deformation log | Append-only log of delta vectors at each structural change. Three drift metrics: displacement (net), path length (cumulative), directionality (displacement/path). | Behavioral change is geometric movement — "how far", "how fast", and "in which direction" are one query. | Time-series DBs store metric values. Here the time series is the entity's trajectory through coordinate space. | `0.1.0` |
| Stateful navigation | 12 typed primitives (`walk_line`, `jump_polygon`, `dive_solid`, `attract_anomaly`, ...). Position type (None/Point/Polygon/Solid) constrains valid operations at runtime. | AI agents navigate structure with guardrails — invalid operations are rejected by the type system, not by error handling. | SQL/GraphQL: stateless queries. REST APIs: stateless endpoints. | `0.1.0` |
| Cross-sphere comparison | `\|\|delta\|\|` is dimensionless (units of sigma). 4.2 in a banking sphere = 4.2 in a logistics sphere. No shared schema or feature alignment required. | Compare structural deviation across domains without joint training or manual normalization. | Cross-domain comparison typically requires shared features or joint embeddings. | `0.1.0` |
| Counterfactual simulation | `simulate_edges(entity, add=[(type, 5)])` recomputes delta against fixed mu/sigma (population baseline unchanged). Reports if entity crosses threshold. | Answers "what if this entity had 5 more transactions?" deterministically, without building a causal model. | Causal inference: requires explicit DAG. What-if in SQL: manual re-aggregation. | `0.1.0` |
| Graph contagion scoring | `contagion_score` measures mean `\|\|delta\|\|` of connected entities. `propagate_influence` computes Cohen's d between connected and control groups. | Measures how anomaly propagates through relationships — not who is central, but whose behavior spreads. | PageRank/betweenness measure graph topology, not behavioral propagation through edges. | `0.2.0` |
| Witness cohort | `find_witness_cohort` finds entities with similar deltas that are NOT directly connected. Combines delta cosine, witness Jaccard, trajectory cosine. | Validates an anomaly is a pattern rather than a one-off — "others behave the same way." | No direct equivalent in common tools. Nearest: k-NN on features, but witness cohort explicitly excludes graph neighbors. | `0.2.1` |
| FDR-controlled detection | `find_anomalies(fdr_alpha=0.05)` applies Benjamini-Hochberg to rank-based p-values of `\|\|delta\|\|`. Returns per-entity q-values. | At alpha=0.05, at most 5% of flagged entities are false discoveries. Statistical guarantee instead of heuristic thresholding. | BH correction exists in biostatistics, but not combined with geometric anomaly detection in one system. | `0.3.1` |
| Regime change detection | `attract_regime_change` computes per-bucket centroids, flags intervals where centroid shift exceeds `mean + 2*std`. Self-calibrating. | Detects shifts in population structure, not just individual entity drift — systemic changes visible. | Evidently/NannyML monitor model prediction drift, not coordinate-space population structure. | `0.1.0` |
| Diverse anomaly selection | `find_anomalies(select="diverse")` uses facility location (Nemhauser-Wolsey-Fisher) to cover distinct anomaly regions. | Returns representative anomalies across the anomaly space, not redundant extremes from one cluster. | Top-N by score returns extremes from the same region. | `0.3.1` |
| Distribution-aware scoring | Each dimension gets a `kind` tag (gaussian/poisson/bernoulli). Anomaly score uses the matching Bregman divergence per dimension — KL for counts, KL for binary, squared z-score for continuous. Contributions are exactly additive (Pythagorean property). | A count dimension with value 1 vs population mean 50 is scored differently than a continuous amount with the same z-score. Attribution sums to 100% — no approximation. | PyOD/sklearn: uniform distance metric across all features. SHAP: approximate, model-dependent attribution. | `0.4.0` |
| Anomaly confidence | `anomaly_confidence: 0-1` per entity via bootstrap resampling. Measures how stable the anomaly verdict is if the population composition changes. `min_confidence` filter on `find_anomalies`. | Agent distinguishes solid anomalies (confidence=0.95) from borderline cases (confidence=0.3) without manual threshold exploration. | No equivalent in standard anomaly detectors — binary verdict with no stability signal. | `0.4.0` |
| In-memory graph index | All graph operations use a lazily-built adjacency index with O(1) neighbor lookups. Cached on the session reader after first graph call. | `find_counterparties`, `discover_chains`, `contagion_score`, `find_geometric_path` run without per-call disk reads — interactive latency on million-edge graphs. | Neo4j/NetworkX: separate graph DB or in-memory load. Here the index is part of the sphere session. | `0.4.0` |

### What changes in practice

The same problems look different when graph, time, and statistics are unified:

| Problem | Typical approach | With hypertopos |
|---|---|---|
| Detect anomalies | Train model, engineer features, choose contamination rate, retrain on shift | `hypertopos build` from YAML, `find_anomalies()` — threshold auto-calibrated from population |
| Explain an anomaly | SHAP on trained model — feature importance for latent dimensions | `explain_anomaly(entity)` — ranked real dimensions: `loan_count: +3.2 sigma (42%)` |
| Compare across domains | Align schemas, build shared features, normalize units | Compare `\|\|delta\|\|` directly — 4.2 sigma means the same in any sphere |
| Track behavioral drift | Export to time-series DB, build dashboard, set manual thresholds | `attract_drift(window)` — displacement, path length, directionality per entity |
| Validate anomaly is real | Manual investigation, ask domain expert | `find_witness_cohort(entity)` — similar non-connected entities confirming the pattern |
| Understand propagation | PageRank, manual path tracing, cross-table joins | `propagate_influence(source)` — Cohen's d between connected vs control group |
| Trust an anomaly verdict | Re-run with different thresholds, manual sensitivity analysis | `find_anomalies(min_confidence=0.8)` — only entities stable under population perturbation |
| Understand why anomalous | SHAP on black-box model, approximate feature importance | `explain_anomaly` — per-dimension Bregman contribution with distribution kind, sums to 100% |

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
