# Golden outputs (hypertopos-py)

Canonical JSON snapshots of key call-site results, checked by
`tests/test_golden_core.py`. They guard refactors that must not change
behaviour: a golden diff is either an intended change or a regression.

**Rule: every golden diff in a PR must be explained in the PR description
(what changed and why it is intended). An unexplained golden diff is a
regression.**

## What is covered

Spheres:

- **built**: a small sphere built in the test session by `GDSBuilder` from
  seeded synthetic rows (anchor + event pattern, event dim, derived dims incl.
  a 7-day window, 14-day temporal slices). Builder output (calibration,
  geometry, temporal) is under test through it.
- **synthetic_chain_sphere** and **sales_sphere**: the hand-set fixtures from
  `tests/fixtures/`.

| File | Call site |
|------|-----------|
| `built_calibration.json` | `sphere.json` per pattern: `mu`, `sigma_diag`, `theta`, `population_size`, `relations` |
| `built_sphere_overview.json` | `GDSNavigator.sphere_overview()` |
| `built_attract_anomaly_accounts.json` | `π5_attract_anomaly` on the anchor pattern: ranking, `delta`, `delta_norm`, rank pct, confidence |
| `built_attract_anomaly_transactions.json` | `π5_attract_anomaly` on the event pattern |
| `built_explain_anomaly.json` | `explain_anomaly` (per-dimension attribution) for the top anomaly |
| `built_attract_drift.json` | `π9_attract_drift` on the temporal slices |
| `built_aggregate.json` | `aggregate()`: count by anchor line, `sum:amount` grouped by an anchor property |
| `built_check_alerts.json` | `check_alerts()` |
| `extract_chains_engine.json` | `engine.chains.extract_chains` on the seeded transactions |
| `chain_coherent_anomaly.json` | `find_chains_with_coherent_anomaly` |
| `chain_propagation_and_typology.json` | `anomaly_propagation_in_chain` + `classify_chain_typology` for four chains |
| `chain_for_entity.json` | `find_chains_for_entity` for an anomalous and a clean entity |
| `sales_calibration_and_anomaly.json` | calibration, `π5_attract_anomaly`, `explain_anomaly` on `sales_sphere` |

## Updating

```bash
HYPERTOPOS_UPDATE_GOLDEN=1 .venv/Scripts/python -m pytest packages/hypertopos-py/tests/test_golden_core.py
```

Add `::test_name` to regenerate one file. Review the resulting `git diff` of
`tests/golden/` and explain it in the PR. On a mismatch the failing test
prints the differing JSON paths (first 20) and the exact command for that
test.

## Normalisation

The snapshot is strict about structure and lenient only about float noise:

- JSON is written with sorted keys, 2-space indent, ASCII escapes, `\n` line
  endings. Comparison is on the parsed documents, not bytes.
- **List order is kept**: ranking order is part of the contract.
- Dict keys, strings, ints, bools and `null` must match exactly, including
  type (an int turning into a float is a diff).
- Floats are stored rounded to 10 significant digits. That keeps a float32
  value (about 7 significant digits) intact and drops float64 summation-order
  noise. Floats are compared with a relative tolerance of `1e-6` (absolute
  `1e-9`), which absorbs SIMD / BLAS / platform differences around the 7th
  digit while still catching any real change.
- A golden float printed with at most 6 decimals is treated as already
  rounded by the code under test; a difference of one unit in its last
  decimal is accepted (a value sitting on a rounding boundary can flip
  between platforms). Because trailing zeros are not printed (`0.0500` is
  stored as `0.05`), that unit is never coarser than `1e-4`.
- NaN and infinities are stored as the strings `"NaN"`, `"Infinity"`,
  `"-Infinity"`; `-0.0` becomes `0.0`; numpy scalars and arrays become plain
  numbers and lists; datetimes become ISO strings; sets are sorted.
- Volatile keys are dropped at any depth: `elapsed_ms`, `elapsed_s`,
  `elapsed_seconds`, `computed_at`, `created_at`, `updated_at`,
  `last_refresh_at`, `built_at`, `generated_at` (wall-clock timings and
  build/refresh timestamps). Data timestamps are kept.
- Absolute sphere paths inside strings are replaced by `<SPHERE>`.

Helpers live in `tests/_golden_py.py`.
