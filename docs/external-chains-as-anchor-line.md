# External Chains as Anchor Lines

> Bring chains discovered outside hypertopos into the same geometric machinery: find_anomalies, drift, density gaps, lead-lag — and the full chain-coherent investigative loop (R9 family) when the convention column is populated.

Real-world chains usually come from somewhere else. SAR typology engines emit candidate laundering chains; supply-chain ERPs maintain canonical product flows; EHR systems publish clinical pathways; customer-journey platforms ship session sequences. These chains have stable identifiers, curated features, and domain semantics that hypertopos's BFS-based `chain_lines:` extractor cannot reproduce. The right way to use them: **declare the external chain table as an anchor line**, let hypertopos compute the population-relative geometry over your chain features, and navigate with the same primitives that work on any anchor pattern.

## When to use this

- You have a parquet / CSV / SQL table where each row is one chain (one `chain_id`) plus chain-level features (length, total amount, time span, jurisdictional flags, ...).
- The chains were produced by an external system (SAR engine, ERP, EHR, audit pipeline) — not by hypertopos's BFS extractor.
- You want geometric anomaly detection / drift tracking / similarity search / lead-lag analysis over those chains, and optionally the R9 chain-coherent investigative loop.

If your chains DON'T exist yet — e.g. you're starting from a transaction edge table — the built-in `chain_lines:` BFS extractor is the right tool. This guide is for the case where chains are already curated upstream.

## Schema convention

A chain anchor line follows the standard hypertopos anchor-line schema, with one optional column to enable the chain-coherent loop:

| column | required | type | meaning |
|---|---|---|---|
| `<key_col>` | yes | string | The primary key — your `chain_id`. |
| `<feature columns>` | yes | numeric / boolean / categorical | Chain-level features that go into the polygon delta vector (hop count, total amount, time span, cross-bank count, etc.). |
| `chain_keys` | optional | string | **Comma-joined member entity primary_keys**, in chain order. Populating this column unlocks the R9 chain-coherent investigative loop on top of the standard anchor primitives. Convention: `"A1,A2,A3,A4"` for a 4-hop chain. |

The chain order encoded in `chain_keys` is what `anomaly_propagation_in_chain` walks; `find_chains_with_coherent_anomaly` reads it to detect coherent runs of consecutive anomalous members on the same dominant dim.

## Minimal example

Source table `sar_chains.parquet`:

| chain_id | hop_count | total_amount | cross_bank_count | chain_keys |
|---|---|---|---|---|
| `SAR-CHAIN-001` | 4 | 380000.0 | 3 | `ACC-1001,ACC-1002,ACC-1003,ACC-1004` |
| `SAR-CHAIN-002` | 3 | 95000.0 | 1 | `ACC-2001,ACC-2002,ACC-2003` |
| ... | ... | ... | ... | ... |

`sphere.yaml`:

```yaml
version: "0.1.0"
sphere_id: aml_with_external_chains

sources:
  accounts:
    path: accounts.parquet
  sar_chains:
    path: sar_chains.parquet

lines:
  accounts:
    source: accounts
    key: account_id
    role: anchor
  sar_chains:
    source: sar_chains
    key: chain_id
    role: anchor          # external chain table = anchor line

patterns:
  account_pattern:
    type: anchor
    entity_line: accounts
    relations: []
    # ... your account features

  sar_chain_pattern:
    type: anchor
    entity_line: sar_chains
    relations: []
    precomputed_dimensions:
      - column: hop_count
      - column: total_amount
      - column: cross_bank_count
    anomaly_percentile: 95
    # Membership lookup is via the chain_keys column on the sar_chains
    # line (see "Chain-coherent loop" below). chain_keys is NOT a
    # geometry dimension — it's a string column the R9 primitives read
    # to walk member entities.
```

`precomputed_dimensions` is the explicit declaration of which columns on the entity line feed the polygon delta vector. Each entry is a dict with `column` (required) and optional `edge_max` (cap for binary mode). Numeric and boolean columns are accepted; string columns like `chain_keys` are NOT eligible — they live on the line as properties and are read by primitives that need them.

Build:

```bash
hypertopos build sphere.yaml --output aml_sphere
```

That's it. The chains now have population-relative coordinates against each other.

## What works out-of-the-box

After the build, every standard anchor primitive operates on `sar_chain_pattern`:

| Primitive | What it gives you |
|---|---|
| `find_anomalies("sar_chain_pattern", top_n=N)` | Top-N chains by `delta_norm` — chains whose feature profile is unusual relative to the chain population. |
| `π9_attract_drift("sar_chain_pattern", ...)` | Chains drifting toward / away from anomaly over the temporal slices, when the line carries timestamps. |
| `π10_attract_trajectory(chain_id, ...)` | Chains with similar temporal trajectories. |
| `π11_attract_population_compare(...)` | Two-window comparison of the chain population (e.g. before / after a regulatory change). |
| `π12_attract_regime_change(...)` | Changepoint detection on the chain population's geometric centroid. |
| `find_density_gaps("sar_chain_pattern")` | Combinations of feature values the independence null says should exist but don't (anomaly by absence). |
| `find_lead_lag("sar_chain_pattern", "other_pattern")` | Centroid-drift cross-correlation if you have a second related pattern. |
| `trace_root_cause(chain_id, "sar_chain_pattern")` | Branching DAG of why this chain is anomalous, via per-dim contribution. |
| `explain_anomaly(chain_id, "sar_chain_pattern")` | Per-dim contribution breakdown for one chain. |
| `find_similar_entities(chain_id, "sar_chain_pattern", k=20)` | k-NN in delta space — chains that look like this one. |
| `decompose_drift(chain_id, "sar_chain_pattern")` | Intrinsic vs extrinsic drift split for one chain over its temporal slices. |

None of these require `chain_keys`. They treat `sar_chain_pattern` as any other anchor pattern.

## Chain-coherent loop (R9 family)

To unlock `find_chains_with_coherent_anomaly`, `anomaly_propagation_in_chain`, `classify_chain_typology`, `extend_chain`, `investigate_chain`, `chain_witness_intersection`, and `chain_drift_trajectory`, the chain anchor line must carry the `chain_keys` column populated per the convention (comma-joined member primary_keys in chain order).

When `chain_keys` is present:

| Primitive | What it gives you |
|---|---|
| `find_chains_with_coherent_anomaly("sar_chain_pattern", anchor_pattern_id="account_pattern", min_hops=3)` | Chains where ≥ `min_hops` consecutive members are individually anomalous AND share the same dominant delta dim — the structuring-cascade signal. |
| `anomaly_propagation_in_chain(chain_id, "sar_chain_pattern", anchor_pattern_id="account_pattern")` | Per-hop anomaly progression — see WHERE the anomaly intensity peaks and WHERE it breaks within the chain. |
| `classify_chain_typology(chain_id, "sar_chain_pattern", anchor_pattern_id="account_pattern")` | Five-axis operational tag: shape (rising / falling / peak), peak position, position in chain (leading / transit / terminal), extension signals, dominant top dim. |
| `extend_chain(chain_id, "sar_chain_pattern", anchor_pattern_id="account_pattern", direction="forward")` | Boundary candidates — entities that follow the chain's anomalous run in OTHER chains and are themselves anomalous. |
| `chain_investigation_summary("sar_chain_pattern", anchor_pattern_id="account_pattern")` | Population-level triage: `coherent_run_rate`, `cross_pattern_overlap`, `recommended_min_hops`. |
| `investigate_chain(chain_id, "sar_chain_pattern", anchor_pattern_id="account_pattern")` | One-shot orchestrator — runs trace + typology + shape lookup + extension forward + extension backward and returns a SAR-ready summary. |
| `chain_witness_intersection(chain_id, "sar_chain_pattern", member_pattern="account_pattern")` | Intersect the top witness dims of the chain's members — `coordinated=True` when their mean pairwise Jaccard clears the threshold, indicating a single anomaly mechanism rather than independent member-level reasons. |
| `chain_drift_trajectory(chain_id, "sar_chain_pattern", member_pattern="account_pattern", n_windows=4)` | Per-member regime (`normalizing` / `deteriorating` / `neutral`) over time-bucketed `delta_norm`, rolled up to a chain-level regime + drift score; spots chains drifting toward anomaly before any single hop crosses the threshold. |

The R9 loop expects each member key in `chain_keys` to exist as a `primary_key` on the corresponding member-anchor line (e.g. `account_pattern`'s entity line). Members not present on that line are silently skipped during the per-hop trace.

## Worked example — SAR typology engine output

A typical SAR typology engine emits:

```python
# sar_engine_output.parquet
chain_id           : str
typology_label     : str   # e.g. "structuring", "layering", "smurfing"
hop_count          : int
total_amount       : float
time_span_hours    : float
cross_bank_count   : int
member_account_ids : str   # comma-joined account ids in chain order
```

Map it to the hypertopos schema. The `lines.<name>.columns:` block is a `{NEW_NAME: source_name}` rename map — point `chain_keys` at the upstream `member_account_ids` column. Then declare:

```yaml
sources:
  sar_typology:
    path: sar_engine_output.parquet

lines:
  sar_typology_chains:
    source: sar_typology
    key: chain_id
    role: anchor
    columns:
      typology_label: typology_label
      hop_count: hop_count
      total_amount: total_amount
      time_span_hours: time_span_hours
      cross_bank_count: cross_bank_count
      chain_keys: member_account_ids   # rename source column to the convention name

patterns:
  sar_typology_pattern:
    type: anchor
    entity_line: sar_typology_chains
    relations: []
    precomputed_dimensions:
      - column: hop_count
      - column: total_amount
      - column: time_span_hours
      - column: cross_bank_count
    anomaly_percentile: 95
```

Build, then run:

```python
nav.investigate_chain(
    "SAR-CHAIN-XXX",
    "sar_typology_pattern",
    anchor_pattern_id="account_pattern",
)
```

You get the full R9 report — trace, typology, shape anomaly, forward/backward extension, SAR-ready summary — over external typology-engine chains, with zero changes to the chain-coherent primitives.

## What this does NOT do

- **Build chains for you.** If your starting point is a transaction edge table with no chain identifiers, use `chain_lines:` (BFS extraction) instead.
- **Validate the chains.** External chains are trusted — hypertopos won't audit your typology engine's output.
- **Re-derive chain-level features.** Whatever columns you put on the chain anchor line ARE the geometry. If you want hop-count / amount-decay / time-span, populate them upstream.
- **Span multiple chain anchor lines.** One chain belongs to one anchor pattern. If the same `chain_id` exists in multiple chain tables (e.g. SAR typology + ERP workflow), declare them as separate anchor patterns; the deferred `cross_pattern_chain_consistency` primitive will eventually correlate them.

## See also

- [Core Concepts](concepts.md) — anchor / event / pattern / line semantics
- [Configuration](configuration.md) — full `sphere.yaml` reference
- [Data Format](data-format.md) — physical storage layout including the `chain_keys` column
- `gds-fraud-investigator` skill — R9 investigative-loop recipe (now external-chain-friendly)
