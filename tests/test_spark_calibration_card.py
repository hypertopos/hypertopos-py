# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""calibration_card (hypertopos.spark)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from hypertopos.datasets import synthetic_aml  # noqa: E402
from hypertopos.engine.robust import min_reference_size  # noqa: E402
from hypertopos.spark import calibration_card, find_anomalies, sphere_of  # noqa: E402


@pytest.fixture(scope="module")
def readme(spark, tmp_path_factory):
    data = synthetic_aml(n_accounts=1_000, seed=1)
    tx = spark.createDataFrame(data.transactions.to_pandas())
    accounts = spark.createDataFrame(data.accounts.to_pandas())
    events = tx.join(accounts.select("account", "segment"), tx.from_account == accounts.account)
    out = find_anomalies(
        events,
        entity="from_account",
        counterparty="to_account",
        amount="amount",
        time="ts",
        group_by="segment",
        score_per_group=True,
        sphere_root=str(tmp_path_factory.mktemp("card")),
    ).cache()
    out.count()
    return out


def test_card_states_what_the_calibration_can_flag(readme):
    card = calibration_card(readme)
    n = readme.count()
    assert card["entities"] == n
    assert card["alpha"] == pytest.approx(0.01)
    assert card["flagged"] == readme.where("is_anomaly").count()
    assert set(card["groups"]) == {
        r["segment"] for r in readme.select("segment").distinct().collect()
    }
    for g in card["groups"].values():
        assert g["p_floor"] == pytest.approx(2.0 / (g["reference_size"] + 1))
        assert g["can_flag"] is (g["reference_size"] >= min_reference_size(card["percentile"]))
        assert g["scored_by"] in ("group", "global")


def test_card_names_tied_mass_dimensions(readme):
    card = calibration_card(readme)
    assert "round_amount_share" in card["tied_dimensions"]
    for share in card["tied_dimensions"].values():
        assert 0.95 <= share <= 1.0


def test_card_accepts_the_sphere(readme):
    assert calibration_card(sphere_of(readme))["entities"] == calibration_card(readme)["entities"]


def test_a_population_too_small_for_its_own_members_cannot_flag(spark, tmp_path):
    # 150 fitted entities at the 99th percentile: the smallest p a member can get
    # is 2 / 151 > 0.01, although 1 / 151 (the floor of a new row) is below it.
    rng = np.random.default_rng(0)
    n, per = 150, 8
    src = np.repeat(np.arange(n), per)
    events = pd.DataFrame(
        {
            "src": [f"a{i}" for i in src],
            "dst": [f"a{(i + 1) % n}" for i in src],
            "amount": rng.lognormal(3.0, 1.0, n * per),
            "ts": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(rng.integers(0, 30 * 24 * 3600, n * per), unit="s"),
        }
    )
    out = find_anomalies(
        spark.createDataFrame(events),
        entity="src",
        counterparty="dst",
        amount="amount",
        time="ts",
        sphere_root=str(tmp_path),
    )
    card = calibration_card(out)
    assert card["entities"] == n
    assert card["p_floor"] == pytest.approx(2.0 / (n + 1))
    assert card["can_flag"] is False
    assert card["flagged"] == 0
