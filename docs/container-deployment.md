# Container Deployment

> Running hypertopos CLI verbs as one-shot container jobs.

The `hypertopos` CLI is designed to run as a one-shot command inside a
container — build a sphere, gate a deploy on its health, or diff two
spheres before promoting one to production. This page covers the
`ENTRYPOINT ["hypertopos"]` contract and the cloud-ops verbs that pair
with it.

## ENTRYPOINT contract

Set the console script as the image entrypoint and pass the verb plus its
arguments as the container command:

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir hypertopos

ENTRYPOINT ["hypertopos"]
```

With `ENTRYPOINT ["hypertopos"]`, the container command becomes the verb
and its flags:

```bash
# Build a sphere from a mounted config
docker run --rm -v "$PWD:/work" -w /work my-image \
  build --config sphere.yaml --output gds_sales/

# Health-check a built sphere, failing the run on a critical status
docker run --rm -v "$PWD:/work" -w /work my-image \
  sphere health gds_sales/ --exit-code-on-critical
```

Because each invocation is a single short-lived process, no long-running
server is started — the container runs the verb, prints its result, and
exits with a meaningful status code.

## Exit codes

The cloud-ops verbs return process exit codes that shell gates and
orchestrators (`set -e`, Kubernetes Job `backoffLimit`, Airflow
`BashOperator`) can act on directly:

| Verb | Exit 0 | Exit 1 | Exit 2 |
|------|--------|--------|--------|
| `sphere health … --exit-code-on-critical` | status `ok` / `warning` | path is not a sphere | status `critical` (HIGH alert) |
| `sphere validate …` | sphere valid | sphere invalid or path not a sphere | — |
| `sphere diff <old> <new>` | diff produced | either path not a sphere | — |
| `sphere ingest <path> --points <file>` | rows ingested | path not a sphere, points file missing/unsupported, unknown pattern, or unreconstructable geometry | — |

Without `--exit-code-on-critical`, `sphere health` always exits 0 on a
readable sphere (a non-sphere path still exits 1) — useful when you only
want to capture the JSON report and decide what to do with it yourself.

## JSON output for pipelines

Every cloud-ops verb accepts `--json`. In JSON mode, stdout carries only
the JSON document — human-readable diagnostics and error messages go to
stderr — so a pipeline step can parse the result without scraping log
lines:

```bash
# Capture the health report, branch on the status field
status=$(hypertopos sphere health gds_sales/ --json | jq -r .status)
if [ "$status" = "critical" ]; then
  echo "Refusing to deploy a critical sphere" >&2
  exit 1
fi
```

```bash
# Pre-deploy: diff the candidate sphere against the live one
hypertopos sphere diff gds_live/ gds_candidate/ --json > diff.json
jq '.pattern_inventory' diff.json
```

```bash
# Incremental ingest: append a changed-entities table to one pattern's
# geometry, then finalize the rank recompute + ANN rebuild once at the end
# of the batch. --pattern is optional when the sphere has a single pattern.
hypertopos sphere ingest gds_sales/ \
  --points changed_customers.arrow --finalize --json | jq '{added, population_size}'
```

The points table may be Arrow IPC (`.arrow`/`.arrows`), Parquet
(`.parquet`/`.pq`), or CSV (`.csv`/`.csv.gz`) and must carry a
`primary_key` column. The JSON summary reports `added`, `modified`,
`deleted`, `population_size`, the geometry Lance dataset version
before/after, and whether the ANN index was rebuilt (`reindexed`) and the
batch finalized (`finalized`).

## CI gate example

A health check as a deploy gate inside a CI job:

```yaml
- name: Gate on sphere health
  run: hypertopos sphere health gds_sales/ --exit-code-on-critical
```

The step fails (exit 2) and stops the pipeline when the sphere has a
critical alert, and passes otherwise.

## See also

- [Configuration](configuration.md) — the full CLI command reference,
  including `build`, `validate`, `info`, and the `sphere` cloud-ops verbs.
- [Quick Start](quickstart.md) — building a sphere from a relational
  dataset.
