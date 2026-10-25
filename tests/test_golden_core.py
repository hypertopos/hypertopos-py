# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Golden-output regression tests for core call sites.

Each test snapshots one call-site result as canonical JSON under
``tests/golden/`` and fails with a path-level diff when it changes.
Regenerate with ``HYPERTOPOS_UPDATE_GOLDEN=1`` (see ``tests/golden/README.md``).

Three spheres are used:

- ``built``: a small sphere built here by ``GDSBuilder`` from seeded
  synthetic rows (event + anchor pattern, derived dims, temporal slices),
  so builder output (calibration, geometry, temporal) is under test;
- ``synthetic_chain_sphere``: the hand-set chain fixture (chain primitives);
- ``sales_sphere``: the hand-set base fixture.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from _golden_py import canonicalize, check_golden

from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.engine.chains import extract_chains
from hypertopos.navigation.navigator import GDSNavigator
from hypertopos.sphere import HyperSphere

TESTS_DIR = Path(__file__).parent
SALES_SPHERE = TESTS_DIR / "fixtures" / "gds" / "sales_sphere"
CHAIN_SPHERE = TESTS_DIR / "fixtures" / "gds" / "synthetic_chain_sphere"

_SEED = 20_261_021
_N_ACCOUNTS = 40
_N_TX = 1200
_REGIONS = ("north", "south", "east", "west")
_CHANNELS = ("wire", "card", "cash")
_START = datetime(2024, 1, 1)


def _built_tables() -> tuple[pa.Table, pa.Table]:
    """Seeded accounts + transactions; three accounts get a burst of large wires."""
    rng = np.random.default_rng(_SEED)
    keys = [f"A{i:03d}" for i in range(_N_ACCOUNTS)]
    accounts = pa.table(
        {
            "primary_key": keys,
            "region": [_REGIONS[i % len(_REGIONS)] for i in range(_N_ACCOUNTS)],
        }
    )
    src = rng.integers(0, _N_ACCOUNTS, _N_TX)
    dst = (src + rng.integers(1, _N_ACCOUNTS, _N_TX)) % _N_ACCOUNTS
    amount = np.round(rng.lognormal(5.0, 0.8, _N_TX), 2)
    minutes = np.sort(rng.integers(0, 90 * 24 * 60, _N_TX))
    channel = rng.integers(0, len(_CHANNELS), _N_TX)
    # Burst accounts: their last 30 transactions become large wires in one week.
    for acct in (3, 17, 29):
        idx = np.flatnonzero(src == acct)[-30:]
        amount[idx] = np.round(amount[idx] * 20.0, 2)
        channel[idx] = 0
        minutes[idx] = 80 * 24 * 60 + np.arange(len(idx)) * 60
    transactions = pa.table(
        {
            "primary_key": [f"TX-{j:05d}" for j in range(_N_TX)],
            "from_account": [keys[i] for i in src],
            "to_account": [keys[i] for i in dst],
            "amount": amount,
            "channel": [_CHANNELS[c] for c in channel],
            "timestamp": [_START + timedelta(minutes=int(m)) for m in minutes],
        }
    )
    return accounts, transactions


@pytest.fixture(scope="session")
def built_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    accounts, transactions = _built_tables()
    out = tmp_path_factory.mktemp("golden") / "gds_golden_built"
    b = GDSBuilder("golden_built", str(out))
    b.add_line("accounts", accounts, key_col="primary_key", source_id="golden", role="anchor")
    b.add_line(
        "transactions", transactions, key_col="primary_key", source_id="golden", role="event"
    )
    b.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="transactions",
        relations=[
            RelationSpec("accounts", fk_col="from_account", direction="in", required=True),
        ],
    )
    b.add_event_dimension("tx_pattern", "amount")
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "tx_count")
    b.add_derived_dimension(
        "accounts", "transactions", "from_account", "sum", "amount", "amount_sum"
    )
    b.add_derived_dimension(
        "accounts", "transactions", "from_account", "count_distinct", "to_account", "n_cpty"
    )
    b.add_derived_dimension(
        "accounts",
        "transactions",
        "from_account",
        "count",
        None,
        "burst_7d",
        time_col="timestamp",
        time_window="7d",
    )
    b.add_pattern("account_pattern", pattern_type="anchor", entity_line="accounts", relations=[])
    b.build()
    b.build_temporal("timestamp", "14d", event_line="transactions")
    return out


def _nav(path: Path) -> GDSNavigator:
    return HyperSphere.open(str(path)).session("golden").navigator()


@pytest.fixture(scope="session")
def built_nav(built_sphere: Path) -> GDSNavigator:
    return _nav(built_sphere)


@pytest.fixture(scope="session")
def chain_nav() -> GDSNavigator:
    return _nav(CHAIN_SPHERE)


@pytest.fixture(scope="session")
def sales_nav() -> GDSNavigator:
    return _nav(SALES_SPHERE)


def _snap(name: str, payload: Any, sphere: Path | None = None) -> None:
    tokens = {str(sphere): "<SPHERE>"} if sphere is not None else None
    check_golden(name, canonicalize(payload, path_tokens=tokens))


def _pattern_calibration(sphere: Path) -> dict[str, Any]:
    meta = json.loads((sphere / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"))
    keep = ("pattern_type", "mu", "sigma_diag", "theta", "population_size", "relations")
    return {pid: {k: pat.get(k) for k in keep} for pid, pat in sorted(meta["patterns"].items())}


def _anomalies(
    nav: GDSNavigator, pattern_id: str, top_n: int, radius: float | None = None
) -> dict[str, Any]:
    polygons, total, clusters, meta = nav.π5_attract_anomaly(pattern_id, radius=radius, top_n=top_n)
    return {
        "total_found": total,
        "ranking": [
            {
                "primary_key": p.primary_key,
                "delta_norm": p.delta_norm,
                "delta": p.delta,
                "delta_rank_pct": p.delta_rank_pct,
                "is_anomaly": p.is_anomaly,
                "anomaly_confidence": p.anomaly_confidence,
            }
            for p in polygons
        ],
        "clusters": clusters,
        "meta": meta,
    }


# ---------------------------------------------------------------- built sphere


def test_built_calibration(built_sphere: Path) -> None:
    _snap("built_calibration", _pattern_calibration(built_sphere))


def test_built_sphere_overview(built_nav: GDSNavigator, built_sphere: Path) -> None:
    _snap("built_sphere_overview", built_nav.sphere_overview(), built_sphere)


def test_built_attract_anomaly_accounts(built_nav: GDSNavigator) -> None:
    _snap(
        "built_attract_anomaly_accounts", _anomalies(built_nav, "account_pattern", 10, radius=0.5)
    )


def test_built_attract_anomaly_transactions(built_nav: GDSNavigator) -> None:
    _snap("built_attract_anomaly_transactions", _anomalies(built_nav, "tx_pattern", 10))


def test_built_explain_anomaly(built_nav: GDSNavigator) -> None:
    polygons, _, _, _ = built_nav.π5_attract_anomaly("account_pattern", top_n=1)
    top = polygons[0].primary_key
    _snap("built_explain_anomaly", built_nav.explain_anomaly(top, "account_pattern"))


def test_built_attract_drift(built_nav: GDSNavigator) -> None:
    _snap("built_attract_drift", built_nav.π9_attract_drift("account_pattern", top_n=5))


def test_built_aggregate(built_nav: GDSNavigator) -> None:
    payload = {
        "count_by_account": built_nav.aggregate("tx_pattern", "accounts", metric="count", limit=8),
        "sum_amount_by_region": built_nav.aggregate(
            "tx_pattern",
            "accounts",
            metric="sum:amount",
            group_by_property="accounts:region",
        ),
    }
    _snap("built_aggregate", payload)


def test_built_check_alerts(built_nav: GDSNavigator) -> None:
    _snap("built_check_alerts", built_nav.check_alerts())


def test_extract_chains_engine() -> None:
    _, tx = _built_tables()
    ts = [t.timestamp() for t in tx["timestamp"].to_pylist()]
    chains = extract_chains(
        from_keys=tx["from_account"].to_pylist(),
        to_keys=tx["to_account"].to_pylist(),
        event_pks=tx["primary_key"].to_pylist(),
        timestamps=ts,
        categories=tx["channel"].to_pylist(),
        amounts=tx["amount"].to_pylist(),
        time_window_hours=24,
        max_hops=4,
        min_hops=3,
        max_chains=20,
    )
    _snap("extract_chains_engine", {"n_chains": len(chains), "chains": chains})


# ---------------------------------------------------------------- chain sphere

_CHAIN_KW = {"pattern_id": "chain_pattern", "anchor_pattern_id": "account_pattern"}


def test_chain_coherent_anomaly(chain_nav: GDSNavigator) -> None:
    out = chain_nav.find_chains_with_coherent_anomaly(
        "chain_pattern", anchor_pattern_id="account_pattern", min_hops=2, max_results=20
    )
    _snap("chain_coherent_anomaly", out)


def test_chain_propagation_and_typology(chain_nav: GDSNavigator) -> None:
    payload = {
        cid: {
            "propagation": chain_nav.anomaly_propagation_in_chain(chain_id=cid, **_CHAIN_KW),
            "typology": chain_nav.classify_chain_typology(chain_id=cid, **_CHAIN_KW),
        }
        for cid in ("CH-001", "CH-002", "CH-009", "CH-012")
    }
    _snap("chain_propagation_and_typology", payload)


def test_chain_for_entity(chain_nav: GDSNavigator) -> None:
    payload = {
        key: chain_nav.find_chains_for_entity(key, "chain_pattern", top_n=20)
        for key in ("A1", "B3")
    }
    _snap("chain_for_entity", payload)


# ---------------------------------------------------------------- sales sphere


def test_sales_calibration_and_anomaly(sales_nav: GDSNavigator) -> None:
    payload = {
        "calibration": _pattern_calibration(SALES_SPHERE),
        "anomaly": _anomalies(sales_nav, "customer_pattern", 5),
        "explain": sales_nav.explain_anomaly("CUST-001", "customer_pattern"),
    }
    _snap("sales_calibration_and_anomaly", payload, SALES_SPHERE)
