# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Spark reasons attribute the calibrated score, as explain_anomaly does on a sphere."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("pyspark")

from hypertopos.engine import robust  # noqa: E402
from hypertopos.scoring import FrozenCalibration  # noqa: E402
from hypertopos.spark._result import TOP_REASONS, result_table  # noqa: E402

DIMS = ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def fitted() -> tuple[pa.Table, FrozenCalibration]:
    # a and b move together, c and d are independent: the score discounts a
    # deviation shared by a and b, the diagonal delta**2 does not.
    rng = np.random.default_rng(7)
    n = 2000
    a = rng.normal(size=n)
    b = a + 0.15 * rng.normal(size=n)
    c = rng.normal(size=n)
    d = rng.exponential(size=n)
    table = pa.table({"entity": [f"E{i}" for i in range(n)], "a": a, "b": b, "c": c, "d": d})
    return table, FrozenCalibration.fit(table, DIMS)


def _flat_reasons(out: pa.Table, i: int) -> list[tuple[str, float]]:
    return [
        (out[f"reason_{k}_dim"][i].as_py(), out[f"reason_{k}_pct"][i].as_py())
        for k in range(1, TOP_REASONS + 1)
    ]


def test_reasons_are_the_shapley_shares_of_the_score(fitted):
    table, calibration = fitted
    out = result_table(table, calibration)
    scores = calibration.score(table)
    ref = calibration.reference
    cov = np.linalg.pinv(ref.lw_precision, hermitian=True)
    order = np.argsort(-scores.anomaly_score, kind="stable")[:40]
    diagonal_differs = 0
    for i in order:
        shares = robust.dim_shares(ref, np.asarray(scores.score_delta[i], np.float64), cov)
        expected_top = np.argsort(-shares, kind="stable")[:TOP_REASONS]
        got = _flat_reasons(out, int(i))
        assert [g[0] for g in got] == [DIMS[j] for j in expected_top]
        np.testing.assert_allclose([g[1] for g in got], 100.0 * shares[expected_top], rtol=1e-9)
        squared = scores.delta[i].astype(np.float64) ** 2
        if int(np.argmax(squared)) != int(expected_top[0]):
            diagonal_differs += 1
    # The data must tell the two attributions apart, or the test proves nothing.
    assert diagonal_differs > 0


def test_score_shares_match_dim_shares_row_by_row(fitted):
    from hypertopos.scoring import score_shares

    table, calibration = fitted
    scores = calibration.score(table.slice(0, 50))
    ref = calibration.reference
    batch = score_shares(ref, scores.score_delta)
    for i in range(50):
        single = robust.dim_shares(ref, np.asarray(scores.score_delta[i], np.float64))
        np.testing.assert_allclose(batch[i], single, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(batch.sum(axis=1), 1.0)
