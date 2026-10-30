# hypertopos on Spark

Point hypertopos at a Spark DataFrame of events (transactions, calls, shipments) and say which column plays which role. hypertopos turns the events into one shape vector per entity on the executors and calibrates it against the whole population. Only the per-entity table reaches the driver; the events stay in Spark. There is no YAML and no sphere path to manage.

In a notebook, `help(find_anomalies)` prints the full parameter list.

## Install

```bash
pip install "hypertopos[spark]"
```

The extra pulls in `pyspark>=3.5,<5`, plus `setuptools` on Python 3.12+ because PySpark 3.5 imports `distutils`. On Databricks, PySpark is already present: `%pip install hypertopos` is enough. hypertopos needs Python 3.12, so on Databricks use Runtime 16 or later, or serverless environment version 3 or later. It is verified on Databricks serverless (a Spark Connect session) and on classic local Spark sessions; on Databricks, classic clusters run the same classic code path.

> [!NOTE]
> `No matching distribution found for hypertopos`, with versions ignored for `Requires-Python >=3.12`, means the compute runs an older Python, not that the package is missing. In a serverless notebook, open the **Environment** side panel and pick environment version 3 or later. In a job, set `environment_version` to 3 or later in the task's environment spec. A job task without an environment spec can still start on an older default version.

> [!TIP]
> PySpark needs a JVM (Java 11 or 17 for Spark 3.5). Locally, set `PYSPARK_PYTHON` to the interpreter that has hypertopos installed. Otherwise the Python workers start from a different environment.

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
                        amount="amount", time="ts", group_by="segment", score_per_group=True)
result.orderBy("anomaly_score", ascending=False).show(10, truncate=False)
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
result.orderBy("anomaly_score", ascending=False).show(5, truncate=False)
```

One row per entity, sender or receiver:

| Column | Meaning |
|---|---|
| `entity` | The entity key, as a string |
| `anomaly_score` | The calibrated evidence score — the same score the sphere builder gives continuous anchor patterns; higher = more unusual. Rank by it. It averages the entity's population ranks on its axes (see [Score axes](#score-axes)) |
| `is_anomaly` | `true` when `p_value` is at most `1 − percentile/100` (99 by default): the alarm budget — at most 1 % of the population is flagged; entities scored later against the stored calibration keep that budget while the population does not move |
| `p_value` | Conformal p-value of `anomaly_score`: the share of the calibration population scoring at least as high (0.01 means 1 % score as high or higher) |
| `score_rank_pct` | Rank of `anomaly_score` in the population, 0–100 |
| `delta_norm` | Distance from the (group's) centre, in standard deviations across all dimensions — the shape, for reference; it does not decide the flag |
| `top_reasons` | Up to three `{dim, delta, pct_of_total}`: the dimensions that drive `anomaly_score` most, with the signed deviation and each one's share in percent of the Mahalanobis part of the score — the same Shapley split of the Mahalanobis evidence that `explain_anomaly` reports on a sphere, so dimensions that move together share the deviation they make together, and a dimension can lead with a moderate `delta`. Empty for an entity exactly at the centre of the score |

`find_anomalies` scores the fitted table with its in-sample p-values (`FrozenCalibration.score(table, members=True)`), reproducing the values a sphere build would store. `score` for new data uses `members=False`: on smaller calibrations (up to about 2,000 entities per dimension) the alarm budget is kept via out-of-fold scores; on larger ones the in-sample calibration is used because the difference is within noise at that scale. `score` treats every row as a new entity, also when it rescores the calibration's own events: without `current` (`score_axes=("mahal",)`) `anomaly_score`, `score_rank_pct` and the reasons repeat the `find_anomalies` values, while `p_value` and `is_anomaly` can differ slightly.

`top_reasons` is an `array<struct>`, so it works in `display`, in SQL (`top_reasons[0].dim`) and when written to Delta:

```python
result.write.mode("overwrite").saveAsTable("main.aml.entity_anomalies")
```

### Score axes

`anomaly_score` is the mean of the entity's population ranks on its axes, set with `score_axes`:

- **`mahal`** — how rare the entity's combination of dimensions is: a shrunk-covariance Mahalanobis distance on rank-Gaussian coordinates of the dimensions. Always on.
- **`current`** — how unusually the entity's activity changed over the period. The events are split into two calendar halves at the middle day of the whole window (the time role decides the halves); each entity gets a state per half (out and in event counts, the sum of `log1p(amount)` on each side, all events) and moves from the first state to the second. `current` is how far that move is from the average move of the 50 entities whose first-half state is nearest to its own. An entity with no event in one of the halves has no `current` and is scored on `mahal` alone. Needs `time`.

The default is `("mahal", "current")` when `time` is given, and `mahal` alone without `time` or with `score_per_group=True`. `current` costs about a fifth more time per call (the half states and the velocity field); pass `score_axes=("mahal",)` to leave it out. `current` needs events on both sides of the window's middle day, so on a one-day window (a daily calibration) it changes no p-value, rank or flag; pass `score_axes=("mahal",)` there to save its cost. `current` helps when the entities you look for change their activity over the window (an account that turns busy, goes quiet or changes counterparties); when the unusual entities do not shift between the two halves, it only adds noise to the ranking, so leave it out. With `score_per_group=True`, `current` is opt-in: `score_axes=("mahal", "current")` gives each group's reference its own velocity field, fitted on that group's entities, and groups too small for their own reference are pooled: the pool gets its own reference and velocity field when it is large enough, otherwise its entities are scored against the population's, `current` included. The halves are split at the middle day of the whole window in every group; a sphere build splits each group at the middle day of that group's own events, which is the same day whenever the group's events span the whole window. Under the default, a `time` column that holds no values raises `ValueError`, as it does with an explicit `current`. With `current`, `anomaly_score`, `p_value`, `score_rank_pct` and `is_anomaly` equal those of a sphere built by the file-based builder from the same events with an event pattern (an anchor pattern on the same dimensions with `score_axes: [mahal, current]`). The half states count a self-transfer as an out and an in event, as a build does; an event without a counterparty counts as an out event, and an event without a time is in neither half. The half-state columns (`first_half_n_out` … `second_half_n`) are kept in the entity table, and the per-entity work runs on the executors in the same aggregation as the dimensions; the driver reads the first and last event time once more to place the halves.

`score` scores new entities on `mahal` alone, also against a calibration with `current`, and calibrates them against the population's scores without `current`, so the alarm budget holds for a scored batch as without `current`. Rescoring a calibration's own events with `score` then gives a different `anomaly_score` than `find_anomalies`, which scores its members on both axes.

```python
result = find_anomalies(events, entity="from_account", counterparty="to_account",
                        amount="amount", time="ts", score_axes=("mahal",))  # without current
```

### Why an entity stands out

`explain` turns a result into one row per entity and dimension, for the highest-scoring entities or the ones you name:

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
| `pct_of_total` | The dimension's share of the Mahalanobis part of `anomaly_score`, in percent (`explain_anomaly` gives the same share as a fraction); an entity's rows sum to 100. The `current` axis has no per-dimension split |
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

The same calibration becomes a navigable sphere for the [hypertopos MCP server](https://github.com/hypertopos/hypertopos-mcp), so an agent can investigate the entities `find_anomalies` flagged. The agent then sees the same scores, flags, ranks and p-values.

```python
sphere.lance_path()     # builds the sphere on first use, then reuses it
sphere.mcp_config()     # {"command": ..., "args": ["-m", "hypertopos_mcp.main"], "env": {"HYPERTOPOS_SPHERE_PATH": ...}}
sphere.serve()          # runs the MCP server on it over stdio (needs pip install hypertopos-mcp)
```

Put `mcp_config()` under `mcpServers` in an MCP client's configuration, or call `open_sphere(path=...)` with `lance_path()` from an agent that already runs the server. In the sphere, each dimension is labelled `t:<dimension>`, and with `group_by` the pattern is grouped by `calibration_group` (the group, or `(small groups)` for pooled ones).

### Score new data without recalibrating

`score` measures new events against the population a result was calibrated on, for example today's transactions against yesterday's. **Calibrate on a window of the same length as the one you score**: the counts, sums, distinct counterparties and active days of an entity grow with the window, so a day scored against a quarter flags almost everyone (a window a third shorter than its calibration already flags over 20 %).

```python
from hypertopos.spark import score

today = score(spark.table("main.aml.transactions_today"), like=result)
today.where("is_anomaly").write.mode("append").saveAsTable("main.aml.daily_alerts")
```

It returns the same columns as `find_anomalies`. The aggregation and the scoring both run on the executors, and only the calibration travels with the tasks (the per-dimension statistics plus the frozen score reference), so the new data never reaches the driver. Nothing is recalibrated: `anomaly_score`, `p_value` and `score_rank_pct` place each entity within the original population. With `group_by`, an entity whose group did not exist at calibration time is scored against the whole population, with `calibration_scope` set to `global`. `like` accepts the result or its `sphere_of(result)`. To see why a scored entity stands out, pass the events it was scored on: `explain(today, events=new_events)`. Its entities are not in the calibration, so `explain` recomputes their dimensions from their own events, on the executors, and compares them with the frozen population.

#### In a scheduled job

A daily job runs in a new process, so the result of the calibration run is gone. Keep the calibration on persistent storage and score against it by its path:

```python
# Whenever the baseline should move: calibrate on a reference window of the same length
# as the one you score — here yesterday, for scoring today.
reference = spark.table("main.aml.transactions").where("ts >= '2026-10-28' AND ts < '2026-10-29'")
result = find_anomalies(reference, entity="from_account", counterparty="to_account",
                        amount="amount", time="ts", sphere_root="/Volumes/main/aml/hypertopos")
calibration_path = str(sphere_of(result).path)    # keep it, e.g. as a job parameter

# Every day: score the new events against that baseline and append the alerts.
today = score(spark.table("main.aml.transactions_today"), like=sphere_of(calibration_path))
today.where("is_anomaly").write.mode("append").saveAsTable("main.aml.daily_alerts")
```

The baseline stays the same until you run the calibration again, so the scores stay comparable. Keep the two windows the same length: a daily job scores one day against a one-day calibration (yesterday, or the same weekday a week earlier when weekdays differ), a monthly job scores a fixed number of days against the same number. On a Databricks Volume, the entity table and the calibration are written as plain files, which a Volume accepts.

#### Is the calibration still valid?

A frozen baseline keeps its alarm budget only while new data comes from the population it was calibrated on. `calibration_status` tests that on a scored batch:

```python
from hypertopos.spark import calibration_status, score

today = score(new_events, like=result, dimensions=True).persist()
status = calibration_status(today, like=result)
status["status"], status["drifted_dimensions"]
```

Pass every scored row, not the alerts or another subset: the tests compare the whole batch with the whole population, so a filtered frame reads as drift. `dimensions=True` adds three columns: `dimension_values`, the values each entity was scored on, in `sphere_of(result).calibration().dims` order, and `first_event` / `last_event`, the time of the entity's first and last event in epoch seconds (UTC; a `timestamp_ntz` column is its wall clock read as UTC). Without it `score` returns exactly the columns above. `calibration_status` runs one Spark action over the scored frame: the executors count, and only the counts reach the driver. Persist or write the scored frame first, or that action computes the scoring again.

It runs two tests, with a combined false-alarm rate of about 2.5 % when nothing has changed:

- the batch's `p_value` against uniform (a Kolmogorov-Smirnov test, `p_distribution`), which catches a shift of the joint score;
- each dimension against the calibrated population's distribution of that dimension (`p_dimensions`, one Kolmogorov-Smirnov test per dimension with a Bonferroni correction), which names the dimensions that moved in `drifted_dimensions`. Tied values, such as a dimension that is mostly zero, are compared correctly.

Both tests account for the calibration itself being a sample, so a small calibration does not raise false alarms. The result also reports `rows`, `flagged`, `alarm_rate`, `expected_alarm_rate` (1 − `percentile` / 100) and `p_flag_count`, the tail probability of that many flags under the frozen calibration (beta-binomial). That last test is reported for reference and does not set `status`.

It also checks the length of the scored window. `window_days` is the batch's calendar span, from its first to its last event, in whole days rounded up (at least 1); `calibration_window_days` is the same span of the events the calibration was fitted on, stored with the calibration. When the scored window is not between 95 % and 100 % of the calibration window's length, `status` is `window_mismatch` with a `reason` naming both windows: calibrate on a window of the same length as the scored one. Window lengths matter more than they look: a window 10 % shorter than its calibration already doubles the alarm rate, and a 5 % longer one quadruples it. Use fixed-length windows (e.g. 28 or 30 days), not calendar months: February against January is 28 / 31 = 0.90 and fails. The tests above are still computed and reported. A calibration saved before this check existed has `calibration_window_days` `None` and skips it until you calibrate again.

`status` is `ok`, `drift` (recalibrate with `find_anomalies` on a recent reference window), `window_mismatch`, or `too_few_rows` below 100 rows, where the p-values are `None`. When more than one applies, `too_few_rows` comes first, then `window_mismatch`, then `drift`.

Only calibrations without groups are supported. With `group_by` the result is `{"status": "unsupported", "reason": ...}`. A `find_anomalies` result has no `dimension_values`, and its p-values rank the calibrated entities themselves, so `calibration_status` raises `ValueError` for it, as for a scored frame without `dimension_values`, `first_event` or `last_event`.

### Per-group populations

`group_by` names an entity property, such as a segment or a country. With groups, `delta_norm` and the reasons' `delta` are always measured against the entity's own group. Two modes decide what happens to the score and the flag:

**Default (`score_per_group=False`):** the score, `p_value` and `is_anomaly` stay global — every entity is scored against the whole population, so `group_by` does not change who is flagged. Use it when the groups differ in scale but not in kind, so you want group-relative deviations alongside a population-wide flag.

**Per-group (`score_per_group=True`):** each group is scored against its own calibrated reference, so `anomaly_score`, `p_value`, `score_rank_pct` and `is_anomaly` all mean "unusual within this group"; each group keeps its own alarm budget. Use it when the groups are distinct kinds of entity, each with at least about 1,000 members — groups too small to flag a member (199 at the 99th percentile) are pooled, and entities whose group was not seen at calibration time are scored against the whole population. `calibration_scope` says which reference scored each entity (see below).

```python
find_anomalies(tx, entity="from_account", counterparty="to_account", amount="amount",
               time="ts", group_by="segment", score_per_group=True)
```

An entity takes the most frequent value of the column in its own events, and the smallest one on a tie. Entities seen only as a counterparty go to the group `"(none)"`. The result gains the `group_by` column and `calibration_scope`: `"group"` when the entity was scored by its own group's reference, `"pooled"` when by the pool's, and `"global"` when by the whole population (unseen group or undersized pool — only with `score_per_group`).

### Where the work runs, and the cache

Spark computes the per-entity table on the executors. Only that table, one row per entity, reaches the driver. There hypertopos calibrates the population and scores every entity.

The entity table and its calibration are stored under `sphere_root`. The default is the Spark conf `hypertopos.sphere_root`, then the environment variable `HYPERTOPOS_SPHERE_ROOT`, then the driver's temp directory. A second call on the same input with the same settings reuses it instead of recalibrating:

- **A table or file source** is the same input while it reads the same files. A new Delta version reads new files, so it recalibrates.
- **A DataFrame built in memory** is reused only as the same object. A re-created frame always recalibrates.
- **A plain file rewritten in place under the same name** is not detected. Pass `cache=False` to force a fresh calibration.

The driver temp directory does not survive a cluster restart. Set `sphere_root` to a persistent location, such as a Unity Catalog Volume, to keep calibrations. On serverless compute a custom Spark conf cannot be set, so pass `sphere_root=` or set `HYPERTOPOS_SPHERE_ROOT`.

> [!TIP]
> The driver holds the per-entity table, its calibration and the result. With the `current` axis, `lance_path()` also rebuilds one row per event side from the half states on the driver (about 25 bytes each) while it builds the sphere. With all dimensions, the calibration step alone takes under 1 GB per million entities, and the whole call takes several times that, so size the driver at a few GB per million entities. `max_entities` (5 million by default) stops a larger run with a clear error before it starts; raise it on a bigger driver.

### From the command line

The same call runs from the shell on a local Spark session, or as a Databricks Python wheel task (entry point `hypertopos`) on the cluster's session:

```bash
hypertopos spark find-anomalies --input tx.parquet --entity from_account \
    --counterparty to_account --amount amount --time ts --show 20
hypertopos spark find-anomalies --input main.aml.transactions --entity from_account \
    --counterparty to_account --amount amount --time ts --output-table main.aml.entity_anomalies
```

`--input` takes a Parquet or CSV file or directory, a Delta directory, or a table or view name. Exactly one of `--output` (Parquet path), `--output-table` or `--show N` is required. `--score-axes` takes the axes comma-separated (`mahal` or `mahal,current`); unset, it keeps the library default, so pass `--score-axes mahal` to leave `current` out. `hypertopos spark find-anomalies --help` lists every option.

`find-anomalies` prints the path of its calibration (`calibration: ...`). A scheduled job scores each new batch against it with `score`, which adds no recalibration and can append only the flagged entities to an alerts table. Calibrate on a window of the same length as each batch (a day for a daily batch):

```bash
hypertopos spark score --input main.aml.transactions_today \
    --calibration /Volumes/main/aml/hypertopos/1bb767d5fc6a53ec \
    --anomalies-only --append --output-table main.aml.daily_alerts
```

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
- Each dimension is standardised against the population (mean and standard deviation, with a floor of 0.01 on the deviation) and never clipped, so an entity far beyond the usual range stays far. `delta_norm` treats dimensions as independent (two dimensions that move together each count in it); the anomaly score does not, because it measures distance on rank-Gaussian coordinates with a shrunk covariance. The distance is measured from one centre per population, so a population made of several distinct kinds of entity is described poorly without `group_by`.
- `find_anomalies` calibrates on the data it is given; to measure new data against an earlier calibration, use `score`.
- The smallest possible `p_value` is `1/(n+1)` for a population of n entities, so at `percentile=99` a population needs about 100 entities before anything can be flagged.
