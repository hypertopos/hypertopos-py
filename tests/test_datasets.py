# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""hypertopos.datasets.synthetic_aml: deterministic, labelled, quick."""

from __future__ import annotations

import time

import numpy as np
import pyarrow.compute as pc
import pytest

from hypertopos.datasets import TYPOLOGIES, synthetic_aml


@pytest.fixture(scope="module")
def data():
    return synthetic_aml(n_accounts=5_000, seed=1)


def test_same_seed_same_tables():
    a = synthetic_aml(n_accounts=500, seed=7)
    b = synthetic_aml(n_accounts=500, seed=7)
    assert a.transactions.equals(b.transactions)
    assert a.accounts.equals(b.accounts)
    assert not a.transactions.equals(synthetic_aml(n_accounts=500, seed=8).transactions)


def test_schema(data):
    assert data.transactions.column_names == [
        "tx_id",
        "ts",
        "from_account",
        "to_account",
        "amount",
        "currency",
        "payment_format",
        "is_laundering",
        "typology",
    ]
    assert data.accounts.column_names == ["account", "segment", "is_laundering_account", "typology"]
    assert str(data.transactions.schema.field("ts").type) == "timestamp[us, tz=UTC]"


def test_laundering_accounts_and_typologies(data):
    accounts = data.accounts
    share = pc.mean(accounts["is_laundering_account"].cast("int8")).as_py()
    assert share == pytest.approx(0.01, abs=0.002)
    assert set(pc.unique(accounts["typology"]).drop_null().to_pylist()) == set(TYPOLOGIES)
    tx = data.transactions
    labelled = tx.filter(tx["is_laundering"])
    assert set(pc.unique(labelled["typology"]).to_pylist()) == set(TYPOLOGIES)
    assert pc.all(pc.is_null(tx.filter(pc.invert(tx["is_laundering"]))["typology"])).as_py()
    # Every laundering account takes part in a labelled transaction.
    members = set(accounts.filter(accounts["is_laundering_account"])["account"].to_pylist())
    touched = set(labelled["from_account"].to_pylist()) | set(labelled["to_account"].to_pylist())
    assert members <= touched


def test_structuring_stays_under_the_threshold(data):
    tx = data.transactions
    s = tx.filter(pc.equal(tx["typology"], "structuring"))
    amounts = s["amount"].to_numpy()
    assert len(amounts) > 0
    assert (amounts < 10_000).all()
    assert (amounts > 8_000).all()


def test_population_is_heterogeneous_and_mostly_normal(data):
    tx = data.transactions
    assert pc.mean(tx["is_laundering"].cast("int8")).as_py() < 0.01
    amounts = tx["amount"].to_numpy()
    assert np.percentile(amounts, 99) / np.percentile(amounts, 50) > 10  # heavy tail
    assert set(data.accounts["segment"].to_pylist()) == {"retail", "business"}


def test_zero_laundering_share_has_no_laundering():
    data = synthetic_aml(n_accounts=2_000, laundering_share=0, seed=3)
    assert data.transactions.num_rows > 0
    assert not pc.any(data.transactions["is_laundering"]).as_py()
    assert pc.all(pc.is_null(data.transactions["typology"])).as_py()
    assert not pc.any(data.accounts["is_laundering_account"]).as_py()
    assert data.transactions.schema == synthetic_aml(n_accounts=2_000, seed=3).transactions.schema


def test_builds_quickly():
    start = time.perf_counter()
    synthetic_aml(n_accounts=50_000, seed=2)
    assert time.perf_counter() - start < 60
