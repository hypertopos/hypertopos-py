"""The trajectory_continuous detector produces p-values on a sphere with temporal history."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pytest

_ACCOUNTS = [f"A-{i}" for i in range(1, 9)]


@pytest.fixture(scope="module")
def temporal_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from hypertopos.builder import GDSBuilder

    path = tmp_path_factory.mktemp("trajectory") / "sphere"
    rows = [
        {
            "tx_id": f"TX-{day}-{n}",
            "from_account": _ACCOUNTS[(day * 5 + n) % len(_ACCOUNTS)],
            "amount": float((day * 11 + n * 7) % 40 + 1) * (25.0 if n == 0 and day == 4 else 1.0),
            "timestamp": datetime(2024, 1, day),
        }
        for day in (1, 2, 3, 4)
        for n in range(16)
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
        pa.table({"primary_key": _ACCOUNTS}),
        key_col="primary_key",
        source_id="test",
        role="anchor",
    )
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "n")
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", "s")
    b.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
    b.build()
    b.build_temporal("timestamp", "1d")
    return path


def test_trajectory_detector_scores_entities_with_history(temporal_sphere: Path) -> None:
    from hypertopos.sphere import HyperSphere

    session = HyperSphere.open(str(temporal_sphere)).session(agent_id="trajectory_test")
    try:
        nav = session.navigator()
        p_values = nav._collect_trajectory_continuous_p("acct_pattern", list(_ACCOUNTS))
    finally:
        session.close()

    assert p_values, "trajectory_continuous returned no p-values on a sphere with history"
    assert set(p_values) <= set(_ACCOUNTS)
    assert all(0.0 < p <= 1.0 for p in p_values.values())


def test_trajectory_detector_is_opt_in_for_combine_anomaly_pvalues(temporal_sphere: Path) -> None:
    from hypertopos.sphere import HyperSphere

    session = HyperSphere.open(str(temporal_sphere)).session(agent_id="trajectory_test")
    try:
        nav = session.navigator()
        default = nav.combine_anomaly_pvalues("acct_pattern")
        explicit = nav.combine_anomaly_pvalues(
            "acct_pattern", detectors=("delta_norm", "trajectory_continuous")
        )
    finally:
        session.close()

    assert all("trajectory_continuous" not in r["p_per_detector"] for r in default)
    assert any("trajectory_continuous" in r["p_per_detector"] for r in explicit)
