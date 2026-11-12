# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""calibration_card (hypertopos.spark)."""

from __future__ import annotations

from collections import Counter

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


def test_card_says_which_dimensions_lead_the_flags(readme):
    card = calibration_card(readme)
    lead = Counter(r["top_reasons"][0]["dim"] for r in readme.where("is_anomaly").collect())
    total = sum(lead.values())
    assert card["flag_drivers"] == pytest.approx({d: c / total for d, c in lead.items()})
    shares = list(card["flag_drivers"].values())
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0)


def test_flag_drivers_split_only_the_flagged_entities(readme, monkeypatch):
    # The split over dimensions costs far more per row than the score: do it for
    # the flagged entities, not for the population.
    from hypertopos.scoring import FrozenCalibration

    seen = []
    real = FrozenCalibration.score_shares

    def counting(self, table, score_delta):
        seen.append(table.num_rows)
        return real(self, table, score_delta)

    monkeypatch.setattr(FrozenCalibration, "score_shares", counting)
    card = calibration_card(readme)
    assert seen == [card["flagged"]]


def test_card_counts_the_entities_with_no_incoming_events(spark, tmp_path):
    events = spark.range(6_000).selectExpr(
        "CASE WHEN id % 200 = 0 THEN concat('S', id % 3) ELSE concat('E', id % 600) END AS src",
        "concat('E', (id * 7919 + 13) % 600) AS dst",
        "exp(randn(5)) * 100 AS amt",
        "timestamp_seconds(1767225600 + (id * 7907) % 5184000) AS ts",
    )
    result = find_anomalies(
        events, entity="src", counterparty="dst", amount="amt", time="ts", sphere_root=str(tmp_path)
    ).cache()
    table = sphere_of(result).entities().to_pandas().set_index("entity")
    flagged = {r["entity"] for r in result.where("is_anomaly").select("entity").collect()}
    silent = set(table.index[table["n_in"] == 0])
    assert silent == {"S0", "S1", "S2"}  # they only send
    assert calibration_card(result)["one_sided"] == {
        "no_incoming_events": {"entities": 3, "flagged": len(silent & flagged)}
    }


def test_card_has_no_one_sided_count_without_a_counterparty(spark, tmp_path):
    events = spark.range(3_000).selectExpr("concat('E', id % 300) AS src", "rand(1) * 100 AS amt")
    result = find_anomalies(events, entity="src", amount="amt", sphere_root=str(tmp_path))
    assert "one_sided" not in calibration_card(result)
