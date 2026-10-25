"""group_aggregate: float sum / mean do not depend on row order; other ops match Arrow."""

from __future__ import annotations

import math
import random

import pyarrow as pa
import pytest

from hypertopos.engine.lance_sql_agg import group_aggregate


def _rows(seed: int, n: int = 4000):
    rng = random.Random(seed)
    k1 = [rng.choice(["a", "b", "c", None]) for _ in range(n)]
    k2 = [rng.choice(["x", "y"]) for _ in range(n)]
    v = [
        None if rng.random() < 0.05 else rng.uniform(-1e9, 1e9) * 10 ** rng.randint(-6, 6)
        for _ in range(n)
    ]
    return k1, k2, v


def _table(k1, k2, v, order):
    return pa.table(
        {
            "k1": pa.array([k1[i] for i in order], pa.string()),
            "k2": pa.array([k2[i] for i in order], pa.string()),
            "v": pa.array([v[i] for i in order], pa.float64()),
        }
    )


def _as_map(tbl: pa.Table, keys: list[str], op: str) -> dict:
    return {
        tuple(row[k] for k in keys): (row[f"v_{op}"], row["v_count"]) for row in tbl.to_pylist()
    }


@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("keys", [["k1"], ["k1", "k2"]])
def test_float_sum_and_mean_are_bit_identical_across_row_orders(op, keys) -> None:
    k1, k2, v = _rows(0)
    orders = [list(range(len(v)))]
    for seed in (1, 2):
        shuffled = list(range(len(v)))
        random.Random(seed).shuffle(shuffled)
        orders.append(shuffled)
    results = [
        _as_map(group_aggregate(_table(k1, k2, v, o), keys, "v", op), keys, op) for o in orders
    ]
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("keys", [["k1"], ["k1", "k2"]])
def test_float_sum_is_the_correctly_rounded_exact_sum(keys) -> None:
    k1, k2, v = _rows(3)
    got = _as_map(group_aggregate(_table(k1, k2, v, range(len(v))), keys, "v", "sum"), keys, "sum")
    expected: dict = {}
    for i, value in enumerate(v):
        expected.setdefault(tuple((k1[i], k2[i])[: len(keys)]), []).append(value)
    for key, values in expected.items():
        present = [x for x in values if x is not None]
        assert got[key] == (math.fsum(present) if present else None, len(present))


def test_all_null_group_has_null_sum_and_zero_count() -> None:
    tbl = pa.table({"k1": ["a", "a", "b"], "v": pa.array([None, None, 2.5], pa.float64())})
    got = _as_map(group_aggregate(tbl, ["k1"], "v", "sum"), ["k1"], "sum")
    assert got == {("a",): (None, 0), ("b",): (2.5, 1)}


@pytest.mark.parametrize("op", ["min", "max", "sum", "mean"])
def test_other_cases_delegate_to_arrow(op) -> None:
    tbl = pa.table({"k1": ["a", "b", "a", "c"], "v": pa.array([3, 4, 5, 6], pa.int64())})
    expected = tbl.group_by(["k1"]).aggregate([("v", op), ("v", "count")])
    assert _as_map(group_aggregate(tbl, ["k1"], "v", op), ["k1"], op) == _as_map(
        expected, ["k1"], op
    )


@pytest.mark.parametrize(
    "values, expected",
    [
        ([math.inf, 1.0], math.inf),
        ([-math.inf, 2.0], -math.inf),
        ([1e308, 1e308], math.inf),
        ([math.inf, -math.inf], math.nan),
        ([math.nan, 1.0], math.nan),
    ],
)
def test_non_finite_sums_match_arrow_instead_of_raising(values, expected) -> None:
    tbl = pa.table({"k1": ["a"] * len(values), "v": pa.array(values, pa.float64())})
    got = group_aggregate(tbl, ["k1"], "v", "sum").to_pylist()[0]["v_sum"]
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == expected
