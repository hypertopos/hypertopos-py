"""Single-entity motif scoring and root-cause tracing on a cold session read only
the entity's neighbourhood, and give the same answers as with the full graph."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from hypertopos.builder.builder import GDSBuilder, RelationSpec

_ACCOUNTS = [f"A{i:02d}" for i in range(30)]
_MOTIFS = ("cycle_2", "cycle_3", "fan_out")


@pytest.fixture(scope="module")
def sphere_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = random.Random(7)
    n = 900
    senders, receivers = [], []
    for _ in range(n):
        s = rng.choice(_ACCOUNTS[:8]) if rng.random() < 0.4 else rng.choice(_ACCOUNTS)
        r = rng.choice(_ACCOUNTS)
        senders.append(s)
        receivers.append(r)  # self-loops allowed
    txns = pa.table(
        {
            "primary_key": [f"TX{i:05d}" for i in range(n)],
            "sender_id": senders,
            "receiver_id": receivers,
            "amount": [float(rng.randint(1, 20_000)) for _ in range(n)],
            # Coarse timestamps: many ties, so neighbour order rests on row order.
            "tx_date": pa.array([1_700_000_000.0 + (i // 4) * 1800.0 for i in range(n)]),
        }
    )
    b = GDSBuilder("seed_scoped", str(tmp_path_factory.mktemp("seed_scoped") / "gds"))
    b.add_line(
        "accounts",
        pa.table({"primary_key": _ACCOUNTS}),
        key_col="primary_key",
        source_id="accounts",
        role="anchor",
    )
    b.add_line("transactions", txns, key_col="primary_key", source_id="transactions", role="event")
    b.add_derived_dimension("accounts", "transactions", "sender_id", "count", None, "n_out")
    b.add_derived_dimension("accounts", "transactions", "sender_id", "sum", "amount", "amt_out")
    b.add_derived_dimension("accounts", "transactions", "receiver_id", "count", None, "n_in")
    b.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="transactions",
        relations=[
            RelationSpec(line_id="accounts", fk_col="sender_id", direction="out"),
            RelationSpec(line_id="accounts", fk_col="receiver_id", direction="in"),
        ],
    )
    # 30 accounts: a calibrated 95th-percentile budget flags at most one, so the
    # traced roots would carry no anomaly; a low percentile keeps several anomalous.
    b.add_pattern("acct_pattern", "anchor", "accounts", relations=[], anomaly_percentile=50.0)
    return Path(b.build())


def _run(sphere_path: Path, warm: bool, call: Any) -> tuple[Any, bool]:
    from hypertopos.sphere import HyperSphere

    session = HyperSphere.open(str(sphere_path)).session(agent_id="seed_scoped")
    try:
        nav = session.navigator()
        graph_pid = nav._resolve_motif_graph_pid("acct_pattern")
        if warm:
            nav._storage.get_adjacency(graph_pid)
        out = call(nav)
        full_built = graph_pid in nav._storage._adjacency_cache
    finally:
        session.close()
    return out, full_built


def _canon(obj: Any) -> str:
    return json.dumps(obj, default=str, sort_keys=True)


@pytest.mark.parametrize("motif_type", _MOTIFS)
def test_score_motif_cold_matches_full_graph(sphere_path: Path, motif_type: str) -> None:
    def call(nav: Any) -> list[Any]:
        # The busy senders and a few quiet accounts.
        return [nav.score_motif(k, motif_type, "acct_pattern") for k in _ACCOUNTS[4:14]]

    cold, full_built = _run(sphere_path, False, call)
    warm, _ = _run(sphere_path, True, call)
    assert not full_built, "a single-entity motif call built the whole-graph index"
    assert any(r["found"] for r in warm), f"fixture has no {motif_type} instance"
    assert _canon(cold) == _canon(warm)


def test_trace_root_cause_cold_matches_full_graph(sphere_path: Path) -> None:
    def call(nav: Any) -> list[Any]:
        return [nav.trace_root_cause(k, "acct_pattern") for k in _ACCOUNTS[:12]]

    cold, full_built = _run(sphere_path, False, call)
    warm, _ = _run(sphere_path, True, call)
    assert not full_built, "a single-entity trace built the whole-graph index"
    assert "motif_type" in _canon(warm), "fixture traces carry no motif evidence"
    assert _canon(cold) == _canon(warm)
