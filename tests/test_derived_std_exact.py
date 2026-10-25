# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Batched std derived dimensions: exact per-group population std, every run."""

from __future__ import annotations

import random
import statistics

import numpy as np
import pyarrow as pa

from hypertopos.builder.derived import DerivedDimSpec, compute_derived_batch


def _spec(fk: str | list[str]) -> DerivedDimSpec:
    return DerivedDimSpec(
        anchor_line="accounts",
        event_line="tx",
        anchor_fk=fk,
        metric="std",
        metric_col="amount",
        dimension_name="amount_std",
        edge_max="auto",
        percentile=99.0,
    )


def _events(seed: int = 2) -> tuple[pa.Table, list[str]]:
    rng = random.Random(seed)
    anchors = [f"A{i}" for i in range(400)]
    rows_k, rows_v = [], []
    for i, a in enumerate(anchors):
        kind = i % 6
        n = rng.randrange(1, 12)
        if kind == 0:  # constant huge amounts -> std exactly 0
            vals = [1_000_000_000_000.0 + 0.37] * n
        elif kind == 1:  # huge mixed with small
            vals = [rng.choice((1e12, 3.5, 1e12 + 1.0)) for _ in range(n)]
        elif kind == 2:  # nulls mixed in
            vals = [None if rng.random() < 0.4 else rng.uniform(0, 1e6) for _ in range(n)]
        elif kind == 3:  # only nulls
            vals = [None] * n
        elif kind == 4:  # single row
            vals = [rng.uniform(0, 100)]
        else:
            continue  # anchor with no events
        rows_k += [a] * len(vals)
        rows_v += vals
    rows_k += ["NOT-AN-ANCHOR"] * 3
    rows_v += [1.0, 2.0, 3.0]
    order = list(range(len(rows_k)))
    rng.shuffle(order)
    table = pa.table(
        {
            "from": [rows_k[i] for i in order],
            "amount": pa.array([rows_v[i] for i in order], type=pa.float64()),
        }
    )
    return table, anchors


def _reference(table: pa.Table, anchors: list[str]) -> np.ndarray:
    by_key: dict[str, list[float]] = {}
    for k, v in zip(table["from"].to_pylist(), table["amount"].to_pylist(), strict=True):
        if v is not None:
            by_key.setdefault(k, []).append(v)
    return np.array([statistics.pstdev(by_key[a]) if a in by_key else 0.0 for a in anchors])


def test_batched_std_is_the_exact_population_std() -> None:
    table, anchors = _events()
    values, _ = compute_derived_batch(table, pa.array(anchors), "from", [_spec("from")])[
        "amount_std"
    ]
    expected = _reference(table, anchors)
    np.testing.assert_allclose(values, expected, rtol=1e-9, atol=0)
    constant = [i for i in range(len(anchors)) if i % 6 == 0]
    assert (values[constant] == 0.0).all(), "a constant group has std exactly 0"


def test_batched_std_repeats_exactly() -> None:
    table, anchors = _events()
    runs = [
        compute_derived_batch(table, pa.array(anchors), "from", [_spec("from")])["amount_std"][0]
        for _ in range(5)
    ]
    assert all(np.array_equal(r, runs[0]) for r in runs[1:])


def test_batched_std_on_composite_keys() -> None:
    table, anchors = _events()
    table = table.append_column("to", pa.array(["X"] * table.num_rows))
    composite = [f"{a}→X" for a in anchors]
    values, _ = compute_derived_batch(
        table, pa.array(composite), ["from", "to"], [_spec(["from", "to"])]
    )["amount_std"]
    np.testing.assert_allclose(values, _reference(table, anchors), rtol=1e-9, atol=0)


def test_batched_float_sum_and_mean_match_per_group_values() -> None:
    import math

    table, anchors = _events()
    table = table.append_column(
        "amount2", pa.array([v if v is None else v * 1.000001 for v in table["amount"].to_pylist()])
    )
    specs = [
        DerivedDimSpec("accounts", "tx", "from", "sum", "amount", "s", "auto", 99.0),
        DerivedDimSpec("accounts", "tx", "from", "mean", "amount2", "m", "auto", 99.0),
    ]
    out = compute_derived_batch(table, pa.array(anchors), "from", specs)
    by_key: dict[str, list[float]] = {}
    by_key2: dict[str, list[float]] = {}
    for k, v, v2 in zip(
        table["from"].to_pylist(),
        table["amount"].to_pylist(),
        table["amount2"].to_pylist(),
        strict=True,
    ):
        if v is not None:
            by_key.setdefault(k, []).append(v)
            by_key2.setdefault(k, []).append(v2)
    exp_sum = [math.fsum(by_key[a]) if a in by_key else 0.0 for a in anchors]
    exp_mean = [math.fsum(by_key2[a]) / len(by_key2[a]) if a in by_key2 else 0.0 for a in anchors]
    np.testing.assert_allclose(out["s"][0], exp_sum, rtol=1e-12)
    np.testing.assert_allclose(out["m"][0], exp_mean, rtol=1e-12)


def test_batched_float_sum_nan_becomes_zero_and_inf_stays() -> None:
    table = pa.table(
        {"from": ["A", "A", "B", "C"], "amount": [1.0, float("nan"), float("inf"), 2.0]}
    )
    spec = DerivedDimSpec("accounts", "tx", "from", "sum", "amount", "s", 1, 99.0)
    values = compute_derived_batch(table, pa.array(["A", "B", "C", "D"]), "from", [spec])["s"][0]
    assert values[0] == 0.0 and values[1] == float("inf") and values[2] == 2.0 and values[3] == 0.0
