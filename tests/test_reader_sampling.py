"""Sampled geometry reads are reproducible: the same sample_size returns the same rows."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pytest

from hypertopos.storage.reader import GDSReader

_N = 50


@pytest.fixture(scope="module")
def sampled_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from hypertopos.builder import GDSBuilder

    path = tmp_path_factory.mktemp("sampling") / "sphere"
    accounts = [f"A-{i:03d}" for i in range(_N)]
    rows = [
        {
            "tx_id": f"TX-{i}",
            "from_account": accounts[i % _N],
            "amount": float(i % 37 + 1),
            "timestamp": datetime(2024, 1, 1 + i % 3),
        }
        for i in range(4 * _N)
    ]
    b = GDSBuilder("test", str(path))
    b.add_line(
        "transactions",
        pa.table({k: [r[k] for r in rows] for k in rows[0]}),
        key_col="tx_id",
        source_id="test",
        role="event",
    )
    b.add_line(
        "accounts",
        pa.table({"primary_key": accounts}),
        key_col="primary_key",
        source_id="test",
        role="anchor",
    )
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "n")
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", "s")
    b.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
    b.build()
    return path


def _sample(path: Path, k: int) -> list[str]:
    reader = GDSReader(str(path))
    version = reader.read_sphere().patterns["acct_pattern"].version
    table = reader.read_geometry("acct_pattern", version, columns=["primary_key"], sample_size=k)
    return table["primary_key"].to_pylist()


def test_sampled_read_does_not_depend_on_global_random_state(sampled_sphere: Path) -> None:
    random.seed(1)
    first = _sample(sampled_sphere, _N // 2)
    random.seed(2)
    second = _sample(sampled_sphere, _N // 2)
    assert len(first) == _N // 2
    assert len(set(first)) == _N // 2
    assert first == second


def test_sampled_rows_are_a_subset_in_storage_order(sampled_sphere: Path) -> None:
    reader = GDSReader(str(sampled_sphere))
    version = reader.read_sphere().patterns["acct_pattern"].version
    full = reader.read_geometry("acct_pattern", version, columns=["primary_key"])
    order = {k: i for i, k in enumerate(full["primary_key"].to_pylist())}
    positions = [order[k] for k in _sample(sampled_sphere, _N // 3)]
    assert positions == sorted(positions)
