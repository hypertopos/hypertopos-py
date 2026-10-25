# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""aggregate() on an event pattern with two relations to the same line.

Transactions relate to ``accounts`` twice (sender, receiver). Grouping by
``accounts`` alone is ambiguous; ``group_by_relation`` picks one relation by
its display name, and every execution path must then count only that slot.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest

import hypertopos.engine.aggregation as aggregation_mod
from hypertopos.engine.aggregation import aggregate
from hypertopos.model.sphere import Line, RelationDef
from tests.test_aggregation_engine import (
    _MANIFEST,
    _PARTITION,
    _make_pattern,
    _make_sphere,
    _MockEngine,
    _MockReader,
)

# (sender, receiver, currency, amount)
_TXS = [
    ("A", "B", "USD", 100.0),
    ("A", "C", "USD", 200.0),
    ("B", "A", "EUR", 50.0),
    ("C", "A", "USD", 10.0),
    ("A", "A", "USD", 1.0),
    ("B", "C", "EUR", 5.0),
]
_PKS = [f"TX-{i:04d}" for i in range(len(_TXS))]
_SENDER_COUNTS = {"A": 3, "B": 2, "C": 1}
_RECEIVER_COUNTS = {"A": 3, "B": 1, "C": 2}
_SENDER_SUMS = {"A": 301.0, "B": 55.0, "C": 10.0}
_BANKS = {"A": "X", "B": "Y", "C": "X"}


def _pattern():
    return _make_pattern(
        relations=[
            RelationDef(line_id="accounts", direction="in", required=True, display_name="sender"),
            RelationDef(
                line_id="accounts", direction="in", required=False, display_name="receiver"
            ),
            RelationDef(line_id="currencies", direction="in", required=False),
        ]
    )


def _sphere():
    extra = {
        name: Line(
            line_id=name,
            entity_type=name,
            line_role="anchor",
            pattern_id="",
            partitioning=_PARTITION,
            versions=[1],
        )
        for name in ("accounts", "currencies")
    }
    return _make_sphere(pattern=_pattern(), extra_lines=extra)


def _geometry() -> pa.Table:
    return pa.table(
        {
            "primary_key": _PKS,
            "entity_keys": pa.array([[s, r, c] for s, r, c, _ in _TXS], type=pa.list_(pa.string())),
        }
    )


def _points() -> dict[str, pa.Table]:
    return {
        "accounts": pa.table({"primary_key": list(_BANKS), "bank": list(_BANKS.values())}),
        "currencies": pa.table({"primary_key": ["USD", "EUR"]}),
        "transactions": pa.table({"primary_key": _PKS, "amount": [t[3] for t in _TXS]}),
    }


def _run(reader=None, **kwargs):
    return aggregate(
        reader or _MockReader(geometry=_geometry(), points=_points()),
        _MockEngine(),
        _sphere(),
        _MANIFEST,
        event_pattern_id="tx_pattern",
        group_by_line="accounts",
        limit=100,
        **kwargs,
    )


def _values(result) -> dict:
    return {r["key"]: r["value"] for r in result["results"]}


def test_ambiguous_line_error_names_the_selector():
    with pytest.raises(RuntimeError) as exc:
        _run()
    msg = str(exc.value)
    assert "ambiguous" in msg
    assert "group_by_relation" in msg
    assert "'sender'" in msg and "'receiver'" in msg


def test_unknown_relation_name_is_rejected():
    with pytest.raises(RuntimeError) as exc:
        _run(group_by_relation="payer")
    assert "'sender'" in str(exc.value)


def test_count_by_sender_and_receiver():
    sender = _run(group_by_relation="sender")
    receiver = _run(group_by_relation="receiver")
    assert _values(sender) == _SENDER_COUNTS
    assert _values(receiver) == _RECEIVER_COUNTS
    assert sender["group_by_line"] == "accounts"
    assert sender["group_by_relation"] == "sender"


def test_sum_by_sender():
    assert _values(_run(group_by_relation="sender", metric="sum:amount")) == _SENDER_SUMS


def test_sampled_count_by_sender():
    result = _run(group_by_relation="sender", sample_size=len(_TXS), seed=1)
    assert _values(result) == _SENDER_COUNTS


def test_group_by_property_of_the_grouped_relation():
    result = _run(group_by_relation="sender", group_by_property="accounts:bank")
    got = {(r["key"], r["bank"]): r["value"] for r in result["results"]}
    assert got == {(k, _BANKS[k]): v for k, v in _SENDER_COUNTS.items()}


def test_count_distinct_counterparties_per_sender():
    result = _run(group_by_relation="sender", metric="count_distinct:accounts")
    assert _values(result) == {"A": 3, "B": 2, "C": 1}


def test_filters_on_relation_without_selector_still_work():
    result = aggregate(
        _MockReader(geometry=_geometry(), points=_points()),
        _MockEngine(),
        _sphere(),
        _MANIFEST,
        event_pattern_id="tx_pattern",
        group_by_line="currencies",
    )
    assert _values(result) == {"USD": 4, "EUR": 2}


class _LanceReader(_MockReader):
    """Mock reader backed by real Lance datasets so the SQL pushdown paths run."""

    def __init__(self, base: Path):
        super().__init__(geometry=_geometry(), points=_points())
        self._base = base
        lance.write_dataset(_geometry(), str(base / "geometry" / "tx_pattern" / "data.lance"))
        for line_id, tbl in _points().items():
            lance.write_dataset(tbl, str(base / "points" / line_id / "v=1" / "data.lance"))

    def pinned_geometry_version(self, pattern_id):
        return None


@pytest.fixture(scope="module")
def lance_reader(tmp_path_factory: pytest.TempPathFactory) -> _LanceReader:
    return _LanceReader(tmp_path_factory.mktemp("group_by_relation_lance"))


def test_lance_sql_paths_count_only_the_selected_relation(
    lance_reader: _LanceReader, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(aggregation_mod, "_SUBPROCESS_THRESHOLD", 0)
    assert _values(_run(lance_reader, group_by_relation="sender")) == _SENDER_COUNTS
    assert _values(_run(lance_reader, group_by_relation="receiver")) == _RECEIVER_COUNTS
    assert (
        _values(_run(lance_reader, group_by_relation="sender", metric="sum:amount")) == _SENDER_SUMS
    )
    by_bank = _run(lance_reader, group_by_relation="sender", group_by_property="accounts:bank")
    got = {(r["key"], r["bank"]): r["value"] for r in by_bank["results"]}
    assert got == {(k, _BANKS[k]): v for k, v in _SENDER_COUNTS.items()}


class _NoPushdownReader(_MockReader):
    """The mock's point_keys pushdown filters primary keys; drop it for entity filters."""

    def read_geometry(self, pattern_id, version, *, point_keys=None, **kwargs):
        return super().read_geometry(pattern_id, version, **kwargs)


def test_filters_on_the_same_line_match_the_other_relation():
    reader = _NoPushdownReader(geometry=_geometry(), points=_points())
    result = _run(reader, group_by_relation="sender", filters=[{"line": "accounts", "key": "A"}])
    assert _values(result) == {"A": 1, "B": 1, "C": 1}


def test_group_by_line_2_on_the_same_line_pairs_sender_with_receiver():
    result = _run(group_by_relation="sender", group_by_line_2="accounts")
    got = {(r["key"], r["key_2"]): r["value"] for r in result["results"]}
    assert got == {(s, r): 1 for s, r, _, _ in _TXS}


class _AllColumnsReader(_MockReader):
    """Return every geometry column so the in-process delta_dim post-filter runs."""

    def read_geometry(self, pattern_id, version, *, columns=None, **kwargs):
        return super().read_geometry(pattern_id, version, **kwargs)


def test_delta_dim_filter_resolves_dims_by_declared_relations():
    geo = _geometry()
    geo = geo.append_column("delta_dim_0", pa.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    geo = geo.append_column("delta_dim_1", pa.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    assert _pattern().dim_index("accounts") == 0
    result = _run(
        _AllColumnsReader(geometry=geo, points=_points()),
        group_by_relation="sender",
        geometry_filters={"delta_dim": {"accounts": {"gt": 0.5}}},
    )
    assert _values(result) == {"A": 2}


def _currency_by_amount() -> dict:
    return {(c, a): 1 for _, _, c, a in _TXS}


def test_group_by_property_on_the_event_line():
    result = aggregate(
        _MockReader(geometry=_geometry(), points=_points()),
        _MockEngine(),
        _sphere(),
        _MANIFEST,
        event_pattern_id="tx_pattern",
        group_by_line="currencies",
        group_by_property="transactions:amount",
        limit=100,
    )
    got = {(r["key"], r["amount"]): r["value"] for r in result["results"]}
    assert got == _currency_by_amount()


def test_lance_group_by_property_on_the_event_line(
    lance_reader: _LanceReader, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(aggregation_mod, "_SUBPROCESS_THRESHOLD", 0)
    result = aggregate(
        lance_reader,
        _MockEngine(),
        _sphere(),
        _MANIFEST,
        event_pattern_id="tx_pattern",
        group_by_line="currencies",
        group_by_property="transactions:amount",
        limit=100,
    )
    got = {(r["key"], float(r["amount"])): r["value"] for r in result["results"]}
    assert got == _currency_by_amount()
