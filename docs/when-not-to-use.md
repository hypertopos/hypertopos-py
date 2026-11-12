# When not to use hypertopos

hypertopos scores each entity against its population without labels and returns a p-value, the named reasons, and a calibration that can be reused and monitored. That is a narrow job. This page lists the cases where another tool is the better choice; each one comes from a test recorded in [Refutations](refutations.md) or from a stated limit of the engine.

## Use something else when

| Your situation | Why hypertopos is the wrong tool | What to use |
|---|---|---|
| You have plenty of reliable labels | A supervised model trained on the same per-entity aggregates ranks far better than any label-free score, and adding the hypertopos evidence to its inputs does not help it | A gradient-boosted model on your feature table |
| The signal is a structure spanning several entities (a ring, a chain, a scatter-gather scheme) | The score describes one entity against its population. Tests of neighbour and motif-based scores stayed near chance, and explicit graph rules found structures the score cannot rank | Graph rules in SQL, or a graph database; the motif and chain tools here enumerate and describe such structures, but the calibrated score's false-alarm budget does not cover them |
| One known column already separates the cases | A population-relative score over two dozen dimensions dilutes a signal that a single threshold captures | A filter or a rule on that column |
| You must catch any single value that is wildly off (a unit error, a slipped decimal) | The score caps how far one value counts and treats dimensions that move together as one piece of evidence, so an entity extreme on one family of dimensions can stay unflagged | A range check on the column. `dimension_rank_pct` (a result made with `dimension_ranks=True`) lists the entities at the top of a dimension, but a rank does not say by how much |
| The population is small | A fitted entity's smallest p-value is `2 / (n + 1)`, so a population (or a group with its own reference) below `2 / alpha - 1` entities can never flag a fitted entity; `calibration_card` reports this as `can_flag` | A manual review, or a pooled reference over more entities |
| You need the best raw ranking on one specific dataset | The shipped score is chosen for the worst case across datasets; a nearest-neighbour detector wins on some of them and collapses on others | Benchmark several detectors on your own labelled sample |
| You need a decision per event, in real time | Scoring runs as a batch over entities aggregated from a window of events | A streaming rule engine; use hypertopos for the daily or hourly entity screen behind it |
| The entity table does not fit the Spark driver | Aggregation runs on the executors, but the calibration is fitted on the driver; see the sizing section of [hypertopos on Spark](spark.md) | Sample or partition the population, or score per segment |
| You want an agent to answer ad-hoc questions about the data | In a controlled comparison an agent with the tool surface did not beat an agent with SQL beyond using the score, and cost more per run | A SQL agent, with the hypertopos result table as one of its inputs |

## What the explanations are, and are not

The reasons attached to a flagged entity say which named dimensions carry its distance from the population. They explain the score. They do not name a typology or establish intent: in tests the top reasons pointed at the mechanism of a synthetic scheme less than half of the time.

## What it is for

- A first screen over entities when there are no labels, with a false-alarm budget that holds on new rows.
- A reason list per flagged entity that sums to its evidence, and a split of that evidence between "extreme values" and "changed behaviour".
- A calibration that is saved, reused for later batches, and reports when the population it was fitted on has moved (`calibration_status`, the `population_drift` alert).
