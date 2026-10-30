# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""A fresh ``find_anomalies`` reuses the fit's own scores: no second scoring, same result."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyspark")

from hypertopos.datasets import synthetic_aml  # noqa: E402
from hypertopos.scoring import FrozenCalibration  # noqa: E402
from hypertopos.spark import find_anomalies  # noqa: E402

ROLES = {"entity": "from_account", "counterparty": "to_account", "amount": "amount", "time": "ts"}

CASES = {
    "mahal": {"score_axes": ("mahal",)},
    "mahal_grouped": {"group_by": "segment", "score_axes": ("mahal",)},
    "current": {},
    "current_grouped": {"group_by": "segment", "score_axes": ("mahal", "current")},
    "quickstart": {"group_by": "segment", "score_per_group": True},
}


@pytest.fixture(scope="module")
def events(spark):
    # The quickstart's events, on a smaller population.
    data = synthetic_aml(n_accounts=600, seed=1)
    tx = spark.createDataFrame(data.transactions.to_pandas())
    accounts = spark.createDataFrame(data.accounts.to_pandas())
    df = tx.join(accounts.select("account", "segment"), tx.from_account == accounts.account)
    df = df.cache()
    df.count()
    return df


@pytest.mark.parametrize("case", list(CASES))
def test_a_fresh_fit_scores_once_and_matches_the_rescored_result(
    events, tmp_path, monkeypatch, case
):
    calls = []
    original = FrozenCalibration.score

    def counted(self, table, *, members=False):
        calls.append(members)
        return original(self, table, members=members)

    monkeypatch.setattr(FrozenCalibration, "score", counted)
    kwargs = {**ROLES, **CASES[case], "sphere_root": str(tmp_path)}
    fresh = find_anomalies(events, **kwargs, cache=False).toPandas()
    # The fit's own scores are the result: the members are not scored again.
    assert calls == []
    # The cached entry rescores the saved members with ``score(members=True)``.
    cached = find_anomalies(events, **kwargs, cache=True).toPandas()
    assert calls == [True]
    assert list(fresh.columns) == list(cached.columns)
    assert list(fresh.dtypes) == list(cached.dtypes)
    assert len(fresh) > 0
    for column in fresh.columns:
        if column == "top_reasons":
            # Arrays of structs: repr keeps every float exact (and NaN equal to NaN).
            assert [repr(list(v)) for v in fresh[column]] == [repr(list(v)) for v in cached[column]]
            continue
        np.testing.assert_array_equal(
            fresh[column].to_numpy(), cached[column].to_numpy(), err_msg=column
        )
