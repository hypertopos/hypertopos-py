"""Navigator-level integration for find_density_gaps.

Runs on a small sphere built in the test (200 accounts, four derived
dimensions from seeded transactions), so it needs no external data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from hypertopos import HyperSphere
from hypertopos.builder.builder import GDSBuilder

_N_ACCOUNTS = 200
_PATTERN = "account_behavior_pattern"


def _build_sphere(path: Path) -> None:
    rng = np.random.default_rng(7)
    accounts = [f"A{i:03d}" for i in range(_N_ACCOUNTS)]
    tx_per_account = rng.integers(1, 30, size=_N_ACCOUNTS)
    senders: list[str] = []
    amounts: list[float] = []
    for acc, n_tx in zip(accounts, tx_per_account, strict=True):
        scale = float(rng.lognormal(4.0, 1.0))
        senders.extend([acc] * int(n_tx))
        amounts.extend(rng.exponential(scale, size=int(n_tx)).round(2).tolist())
    txns = pa.table(
        {
            "primary_key": [f"TX{i:05d}" for i in range(len(senders))],
            "account_id": senders,
            "amount": amounts,
        }
    )
    builder = GDSBuilder("density_gaps_test", str(path))
    builder.add_line("accounts", pa.table({"primary_key": accounts}), "primary_key", "t")
    builder.add_line("transactions", txns, "primary_key", "t", role="event")
    for metric, col, name in (
        ("count", None, "tx_count"),
        ("sum", "amount", "amount_sum"),
        ("max", "amount", "amount_max"),
        ("std", "amount", "amount_std"),
    ):
        builder.add_derived_dimension("accounts", "transactions", "account_id", metric, col, name)
    builder.add_pattern(_PATTERN, "anchor", "accounts", relations=[])
    builder.build()


@pytest.fixture(scope="module")
def dg_nav(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("density_gaps") / "sphere"
    _build_sphere(path)
    hs = HyperSphere.open(str(path))
    return hs.session("dg-test").navigator()


def test_find_density_gaps_returns_dict_with_gaps(dg_nav):
    out = dg_nav.find_density_gaps(
        _PATTERN,
        top_n=10,
    )
    assert isinstance(out, dict)
    assert "gaps" in out
    assert "excluded_dims" in out
    assert "n_pairs_tested" in out
    assert out["n_pairs_tested"] > 0
    assert out["n_entities"] >= 100


def test_find_density_gaps_rejects_unknown_pattern(dg_nav):
    with pytest.raises(Exception, match="pattern not found"):
        dg_nav.find_density_gaps("nonexistent")


def test_find_density_gaps_rejects_invalid_alpha(dg_nav):
    with pytest.raises(Exception, match="alpha"):
        dg_nav.find_density_gaps(
            _PATTERN,
            alpha=0.0,
        )


def test_find_density_gaps_rejects_invalid_bins(dg_nav):
    with pytest.raises(Exception, match="bins"):
        dg_nav.find_density_gaps(
            _PATTERN,
            bins=2,
        )


def test_find_density_gaps_rejects_invalid_r_window(dg_nav):
    with pytest.raises(Exception, match="r_min"):
        dg_nav.find_density_gaps(
            _PATTERN,
            r_min=0.5,
            r_max=0.4,
        )


def test_find_density_gaps_rejects_invalid_top_n(dg_nav):
    with pytest.raises(Exception, match="top_n"):
        dg_nav.find_density_gaps(
            _PATTERN,
            top_n=0,
        )


# Coarse bins and the full correlation window give the 200-entity sphere
# enough power to flag gaps (the defaults find none at this size).
_GAP_KWARGS = {"bins": 4, "r_max": 1.0}


def test_find_density_gaps_returns_sorted_by_ratio(dg_nav):
    out = dg_nav.find_density_gaps(
        _PATTERN,
        top_n=20,
        **_GAP_KWARGS,
    )
    ratios = [g["ratio"] for g in out["gaps"]]
    assert len(ratios) > 1
    assert ratios == sorted(ratios, reverse=True)
    # Empty cells keep a finite ratio (infinity serialises as null).
    assert all(np.isfinite(r) for r in ratios)


def test_find_density_gaps_unknown_dim_in_user_pairs(dg_nav):
    with pytest.raises(Exception, match="unknown dim names"):
        dg_nav.find_density_gaps(
            _PATTERN,
            dim_pairs=[("nope_x", "nope_y")],
        )


def test_find_density_gaps_includes_raw_ranges(dg_nav):
    out = dg_nav.find_density_gaps(
        _PATTERN,
        top_n=5,
        **_GAP_KWARGS,
    )
    assert out["gaps"]
    for gap in out["gaps"]:
        assert "delta_range_i" in gap
        assert "delta_range_j" in gap
        assert len(gap["delta_range_i"]) == 2
        assert len(gap["delta_range_j"]) == 2
        assert gap["dim_i"] != gap["dim_j"]
