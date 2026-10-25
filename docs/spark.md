# hypertopos on Spark

Point hypertopos at a Spark DataFrame of events (transactions, calls, shipments) and say which column plays which role. hypertopos turns the events into one shape vector per entity on the executors and calibrates it against the whole population. Only the per-entity table reaches the driver; the events stay in Spark. There is no YAML and no sphere path to manage.

In a notebook, `help(find_anomalies)` prints the full parameter list.

## Install

```bash
pip install "hypertopos[spark]"
```

The extra pulls in `pyspark>=3.5,<5`, plus `setuptools` on Python 3.12+ because PySpark 3.5 imports `distutils`. On Databricks, PySpark is already present: `%pip install hypertopos` is enough. hypertopos needs Python 3.12, so on Databricks use Runtime 16 or later, or serverless environment version 3 or later. Classic clusters and Spark Connect sessions (serverless, shared access mode) both work.

> [!NOTE]
> `No matching distribution found for hypertopos`, with versions ignored for `Requires-Python >=3.12`, means the compute runs an older Python, not that the package is missing. In a serverless notebook, open the **Environment** side panel and pick environment version 3 or later. In a job, set `environment_version` to 3 or later in the task's environment spec. A job task without an environment spec can still start on an older default version.

> [!TIP]
> PySpark needs a JVM (Java 11 or 17 for Spark 3.5). Locally, set `PYSPARK_PYTHON` to the interpreter that has hypertopos installed. Otherwise the Python workers start from a different environment. On a laptop, also lower the shuffle partitions, `SparkSession.builder.master("local[2]").config("spark.sql.shuffle.partitions", "4")`. The default of 200 is sized for a cluster and turns every aggregation into hundreds of small tasks.

## Try it on synthetic data

`hypertopos.datasets.synthetic_aml` generates a labelled payments dataset (retail and business accounts, with fan-out, fan-in, cycle, scatter-gather, structuring and pass-through laundering patterns). It is deterministic per seed and needs no download:

```python
from hypertopos.datasets import synthetic_aml
from hypertopos.spark import explain, find_anomalies

data = synthetic_aml(n_accounts=5_000, seed=1)
tx = spark.createDataFrame(data.transactions.to_pandas())
accounts = spark.createDataFrame(data.accounts.to_pandas())
events = tx.join(accounts.select("account", "segment"), tx.from_account == accounts.account)

result = find_anomalies(events, entity="from_account", counterparty="to_account",
                        amount="amount", time="ts", group_by="segment")
result.orderBy("delta_norm", ascending=False).show(10, truncate=False)
explain(result, top_n=3).show(15, truncate=False)
```

`data.accounts` carries the labels (`is_laundering_account`, `typology`), so you can check which flagged accounts were laundering, and compare with other detectors on the same data. The [quickstart notebook](examples/spark-quickstart.ipynb) runs this flow end to end: find, compare with the labels, explain, score a later period and hand the result to an agent.

## Find anomalies

```python
from hypertopos.spark import find_anomalies

result = find_anomalies(
    spark.table("main.aml.transactions"),
    entity="from_account",
    counterparty="to_account",
    amount="amount",
    time="ts",
)
result.orderBy("delta_norm", ascending=False).show(5, truncate=False)
```

One row per entity, sender or receiver:

| Column | Meaning |
|---|---|
| `entity` | The entity key, as a string |
| `delta_norm` | Distance from the population centre, in standard deviations across all dimensions |
| `is_anomaly` | `true` for entities beyond the population's `percentile` (99 by default, about 1 % of entities) |
| `p_value` | Share of the population at least as far from the centre (a conformal p-value: 0.01 means 1 % of entities are as unusual or more) |
| `rank_pct` | Share of the population with a smaller `delta_norm`, 0–100 |
| `top_reasons` | Up to three `{dim, delta, pct_of_total}`: the dimensions that move the entity most, with the signed deviation and each one's share of `delta_norm²` in percent. Empty for an entity exactly at the centre |

`top_reasons` is an `array<struct>`, so it works in `display`, in SQL (`top_reasons[0].dim`) and when written to Delta:

```python
result.write.mode("overwrite").saveAsTable("main.aml.entity_anomalies")
```

### Why an entity stands out

`explain` turns a result into one row per entity and dimension, for the most distant entities or the ones you name:

```python
from hypertopos.spark import explain

explain(result, top_n=20).show(truncate=False)
explain(result, entities=["ACC-1029", "ACC-4471"], dims=["amount_sum_out", "n_distinct_out"])
```

| Column | Meaning |
|---|---|
| `entity`, `dim` | The entity and one of its dimensions |
| `value` | The entity's value of the dimension (log-scaled where the dimension is) |
| `mu`, `sigma` | The population mean and standard deviation it is compared with (its group's, with `group_by`) |
| `delta` | `(value − mu) / sigma`: the signed deviation in standard deviations |
| `pct_of_total` | The dimension's share of `delta_norm²`, in percent; an entity's rows sum to 100 |
| `direction` | `above`, `below` or `at` the mean |

The first three rows of an entity are its `top_reasons`. `explain` runs on the driver over the cached calibration; it needs the result DataFrame itself, because renaming or selecting away the `entity` column drops the metadata that points to the calibration.

### The calibration behind a result

`sphere_of(result)` returns the handle to what a result rests on, without a path to type:

```python
from hypertopos.spark import sphere_of

sphere = sphere_of(result)
sphere.path                                   # where the entity table and calibration are kept
sphere.geometry()                             # pyarrow table: every dimension's value and delta, plus the scores
sphere.to_delta(spark, "main.aml.entity_geometry")   # the same table, written with saveAsTable
```

### Hand the result to an agent

The same calibration becomes a navigable sphere for the [hypertopos MCP server](https://github.com/hypertopos/hypertopos-mcp), so an agent can investigate the entities `find_anomalies` flagged. The agent then sees the same `delta_norm`, flags, ranks and p-values.

```python
sphere.lance_path()     # builds the sphere on first use, then reuses it
sphere.mcp_config()     # {"command": ..., "args": ["-m", "hypertopos_mcp.main"], "env": {"HYPERTOPOS_SPHERE_PATH": ...}}
sphere.serve()          # runs the MCP server on it over stdio (needs pip install hypertopos-mcp)
```

Put `mcp_config()` under `mcpServers` in an MCP client's configuration, or call `open_sphere(path=...)` with `lance_path()` from an agent that already runs the server. In the sphere, each dimension is labelled `t:<dimension>`, and with `group_by` the pattern is grouped by `calibration_group` (the group, or `(small groups)` for pooled ones).

### Score new data without recalibrating

`score` measures new events against the population a result was calibrated on, for example today's transactions against last quarter's:

```python
from hypertopos.spark import score

today = score(spark.table("main.aml.transactions_today"), like=result)
today.where("is_anomaly").write.mode("append").saveAsTable("main.aml.daily_alerts")
```

It returns the same columns as `find_anomalies`. The aggregation and the scoring both run on the executors, and only the calibration travels with the tasks (the per-dimension statistics plus the population's sorted distances, about 4 MB per million calibrated entities), so the new data never reaches the driver. Nothing is recalibrated: `p_value` and `rank_pct` place each entity within the original population. With `group_by`, an entity whose group did not exist at calibration time is scored against the whole population, with `calibration_scope` set to `global`. `like` accepts the result or its `sphere_of(result)`. The scored DataFrame does not point back to a calibration, so `explain` it through the original result.

### Per-group populations

`group_by` names an entity property, such as a segment or a country. Each group gets its own population, so a large business account is compared with businesses, not with retail customers:

```python
find_anomalies(tx, entity="from_account", counterparty="to_account", amount="amount",
               time="ts", group_by="segment")
```

An entity takes the most frequent value of the column in its own events, and the smallest one on a tie. Entities seen only as a counterparty go to the group `"(none)"`. Groups with fewer than `min_group_size` entities (100 by default) are calibrated together: a 99th-percentile threshold cannot flag anyone in a group of five. The result gains the `group_by` column and `calibration_scope`, which is `group` or `pooled`.

With groups, `delta_norm` and `is_anomaly` are measured against the entity's own group: its mean, its spread and its threshold. `p_value` and `rank_pct` then place that group-relative distance within the whole population, so every entity is on one scale. An entity at the 99th percentile of a quiet group can therefore be flagged while its `rank_pct` is lower than that of an unflagged entity in a noisy group. To compare within a group, filter the result by the `group_by` column.

### Where the work runs, and the cache

Spark computes the per-entity table on the executors. Only that table, one row per entity, reaches the driver. There hypertopos calibrates the population and scores every entity.

The entity table and its calibration are stored under `sphere_root`. The default is the Spark conf `hypertopos.sphere_root`, then the environment variable `HYPERTOPOS_SPHERE_ROOT`, then the driver's temp directory. A second call on the same input with the same settings reuses it instead of recalibrating:

- **A table or file source** is the same input while it reads the same files. A new Delta version reads new files, so it recalibrates.
- **A DataFrame built in memory** is reused only as the same object. A re-created frame always recalibrates.
- **A plain file rewritten in place under the same name** is not detected. Pass `cache=False` to force a fresh calibration.

The driver temp directory does not survive a cluster restart. Set `sphere_root` to a persistent location to keep calibrations.

> [!TIP]
> The driver holds the per-entity table, its calibration and the result. With all dimensions, the calibration step alone takes under 1 GB per million entities, and the whole call takes several times that, so size the driver at a few GB per million entities. `max_entities` (5 million by default) stops a larger run with a clear error before it starts; raise it on a bigger driver.

### From the command line

The same call runs from the shell on a local Spark session, or as a Databricks Python wheel task (entry point `hypertopos`) on the cluster's session:

```bash
hypertopos spark find-anomalies --input tx.parquet --entity from_account \
    --counterparty to_account --amount amount --time ts --show 20
hypertopos spark find-anomalies --input main.aml.transactions --entity from_account \
    --counterparty to_account --amount amount --time ts --output-table main.aml.entity_anomalies
```

`--input` takes a Parquet or CSV file or directory, a Delta directory, or a table or view name. Exactly one of `--output` (Parquet path), `--output-table` or `--show N` is required. `hypertopos spark find-anomalies --help` lists every option.

## Column roles

| Role | Required | Meaning | Accepted types |
|---|---|---|---|
| `entity` | yes | Who the event belongs to: account, customer, device | any; cast to string |
| `counterparty` | no | The other side of the event: receiving account, merchant | any; cast to string |
| `amount` | no | Size of the event | numeric |
| `time` | no | When it happened | timestamp, timestamp_ntz, date, or epoch seconds |

Each role you give unlocks a family of dimensions. With only `entity` you get event counts. Adding `counterparty`, `amount` and `time` adds direction, money and rhythm.

## How events become entities

Every event is read from both sides:

- **Out row.** One per event with a non-null entity. The key is the entity and the other side is the counterparty.
- **In row.** One per event whose counterparty is non-null and differs from the entity. The key is the counterparty and the other side is the entity.

So an account that only receives money still gets a shape. Two edge cases:

- An event with a null entity is dropped.
- A self-transfer (entity = counterparty) counts once, as an out row.

## Dimensions

With all four roles, `dims="auto"` gives 23 dimensions. `n_events` is in it only when no counterparty is given: with one, it is `n_out + n_in` and would count the same signal twice.


| Family | Dimension | Definition |
|---|---|---|
| Counts | `n_events` | out + in rows |
| | `n_out`, `n_in` | out rows, in rows |
| | `n_distinct_out` | distinct non-null counterparties of out rows |
| | `n_distinct_in` | distinct senders of in rows |
| Direction | `top_counterparty_share_out` | largest number of out rows to one counterparty / `n_out` (0 when `n_out` = 0) |
| | `in_out_ratio` | `n_in / n_events` |
| Amounts | `amount_sum_out`, `amount_sum_in` | sum of non-null amounts on that side |
| | `amount_mean_out`, `amount_mean_in` | mean of non-null amounts |
| | `amount_std_out`, `amount_std_in` | population std (ddof 0); 0 with fewer than two amounts |
| | `amount_max_out`, `amount_max_in`, `amount_min_out` | max / min of non-null amounts |
| | `round_amount_share` | share of out amounts that are a positive multiple of 100 |
| Rhythm | `active_days` | distinct UTC days with an event |
| | `span_days` | last minus first event time, in days |
| | `iet_mean`, `iet_std` | mean and population std of the gaps between consecutive events, in seconds; 0 with fewer than two gaps |
| | `burst_max_daily` | most events on one UTC day |
| | `hour_share_night` | share of events with UTC hour in [0, 6) |
| | `weekend_share` | share of events on a UTC Saturday or Sunday |

Rules that apply to every dimension:

- **Empty sides give 0.** An entity with no amounts on a side, or no in rows, gets 0 for those dimensions, never null.
- **Few events, little weight.** A share, spread or ratio measured on one or two events is mostly noise: one payment makes the top-counterparty share 100 % by itself. These dimensions are pulled toward the population mean in proportion to how few events they rest on: `(n × value + 3 × mean) / (n + 3)`. Here `n` is `n_out` for `top_counterparty_share_out`, `round_amount_share` and `amount_std_out`; `n_in` for `amount_std_in`; and `n_events` for `in_out_ratio`, `iet_std`, `hour_share_night` and `weekend_share`. These counts are in the entity table even when they are not dimensions. With 30 events the value barely moves; with one it moves three quarters of the way to the mean. `explain` shows the adjusted value.
- **Magnitudes are log-scaled.** Amount sums, means, max, min and std, and the gaps between events, go through `log1p(max(x, 0))` by default, so heavy tails do not dominate the spread. Negative amounts, such as refunds, count as 0. Pass `log_amounts=False` to keep raw values.

### Choosing dimensions

- `dims=[...]` selects by name. The output keeps the catalogue order. An unknown name, or a dimension whose role was not given, fails with a message listing what is available.
- `extra_dims={"name": "<Spark SQL aggregate>"}` adds your own aggregates over the event rows. They can use the columns `key`, `other`, `side` (`"out"` / `"in"`), `amount` and `ts` (epoch seconds), for example `{"big_out": "sum(CASE WHEN side = 'out' AND amount >= 10000 THEN 1 ELSE 0 END)"}`. A name that collides with a built-in dimension is rejected.

## Time

Times are converted to UTC epoch seconds arithmetically, so results do not depend on `spark.sql.session.timeZone`:

| Column type | Read as |
|---|---|
| `timestamp` | the instant it stores |
| `timestamp_ntz` | the wall-clock value, taken as UTC |
| `date` | midnight UTC |
| numeric | epoch seconds |
| string | rejected; cast it first (`F.to_timestamp(...)`) |

> [!TIP]
> If your timestamps are local wall-clock times stored as strings, cast them to `timestamp_ntz`. Days and hours then follow the clock the data was recorded in.

## Limits

- Dimensions are per entity. Pair-level and chain-level patterns (counterparty graphs, chains, motifs) are built by the file-based builder; see [Configuration](configuration.md).
- Each dimension is standardised against the population (mean and standard deviation, with a floor of 0.01 on the deviation) and never clipped, so an entity far beyond the usual range stays far. Dimensions are treated as independent: two dimensions that move together each count in `delta_norm`.
- `find_anomalies` calibrates on the data it is given; to measure new data against an earlier calibration, use `score`.
