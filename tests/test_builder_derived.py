# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for GDSBuilder derived dimensions, composite lines, and graph features."""

import json

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.builder.builder import GDSBuilder

# ---------- Helpers ----------


def _build_basic_events():
    """10 transactions between 3 accounts."""
    return pa.table(
        {
            "primary_key": [f"TX-{i}" for i in range(10)],
            "from_account": ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "to_account": ["B", "C", "B", "C", "A", "C", "A", "A", "B", "A"],
            "currency": ["USD", "EUR", "USD", "EUR", "USD", "USD", "EUR", "USD", "EUR", "USD"],
            "amount": [100, 200, 150, 250, 300, 50, 75, 125, 175, 225],
        }
    )


def _build_accounts():
    return pa.table({"primary_key": ["A", "B", "C"]})


def _make_builder(tmp_path, events=None, accounts=None):
    builder = GDSBuilder("test", str(tmp_path / "sphere"))
    builder.add_line(
        "transactions", events or _build_basic_events(), key_col="primary_key", source_id="test", role="event"
    )
    builder.add_line(
        "accounts", accounts or _build_accounts(), key_col="primary_key", source_id="test", role="anchor"
    )
    return builder


# ---------- add_derived_dimension ----------


class TestAddDerivedDimension:
    def test_count_metric(self, tmp_path):
        """Count events per anchor key."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_out_count",
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        sphere_json = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())
        # A=4, B=3, C=3
        assert sphere_json["patterns"]["account_pattern"]["population_size"] == 3
        # Pattern should have 1 relation (auto-created)
        assert len(sphere_json["patterns"]["account_pattern"]["relations"]) == 1

    def test_count_distinct_metric(self, tmp_path):
        """Count distinct values per anchor key."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count_distinct",
            "to_account",
            "n_counterparties",
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        # A sends to {B, C} → 2, B sends to {A, C} → 2, C sends to {A, B} → 2
        sphere_json = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())
        rel = sphere_json["patterns"]["account_pattern"]["relations"][0]
        assert rel["edge_max"] is not None
        assert rel["edge_max"] >= 1

    def test_sum_metric(self, tmp_path):
        """Sum aggregates numeric column per group."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "sum",
            "amount",
            "total_sent",
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        acct_table = builder._lines["accounts"].table
        assert "total_sent" in acct_table.schema.names
        vals = acct_table["total_sent"].to_pylist()
        # A sends: 100+200+150+250=700, B sends: 300+50+75=425, C sends: 125+175+225=525
        assert vals == [700.0, 425.0, 525.0]

    def test_std_metric(self, tmp_path):
        """Std computes standard deviation per group."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "std",
            "amount",
            "amount_std",
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        acct_table = builder._lines["accounts"].table
        assert "amount_std" in acct_table.schema.names
        vals = acct_table["amount_std"].to_pylist()
        # A amounts: [100, 200, 150, 250] → std ≈ 55.9
        assert vals[0] == pytest.approx(np.std([100, 200, 150, 250]), abs=0.1)

    def test_max_metric(self, tmp_path):
        """Max picks maximum value per group."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "max",
            "amount",
            "max_sent",
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        vals = builder._lines["accounts"].table["max_sent"].to_pylist()
        assert vals == [250.0, 300.0, 225.0]  # A=250, B=300, C=225

    def test_mean_metric(self, tmp_path):
        """Mean computes average per group."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "mean",
            "amount",
            "avg_sent",
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        vals = builder._lines["accounts"].table["avg_sent"].to_pylist()
        assert vals[0] == pytest.approx(175.0)  # A: (100+200+150+250)/4
        assert vals[1] == pytest.approx(141.667, abs=0.01)  # B: (300+50+75)/3

    def test_auto_edge_max(self, tmp_path):
        """edge_max='auto' uses p99 — with 3 accounts it equals max count."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
            edge_max="auto",
            percentile=99.0,
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        rel = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())[
            "patterns"
        ]["account_pattern"]["relations"][0]
        assert rel["edge_max"] >= 3  # p99 of [4, 3, 3] — at least 3

    def test_fixed_edge_max(self, tmp_path):
        """edge_max=10 uses fixed value."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
            edge_max=10,
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        rel = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())[
            "patterns"
        ]["account_pattern"]["relations"][0]
        assert rel["edge_max"] == 10

    def test_missing_event_line_raises(self, tmp_path):
        """Referencing non-existent event line raises ValueError."""
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("accounts", _build_accounts(), key_col="primary_key", source_id="test")
        builder.add_derived_dimension(
            "accounts",
            "nonexistent",
            "from_account",
            "count",
            None,
            "x",
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        with pytest.raises(ValueError, match="nonexistent"):
            builder.build()

    def test_missing_anchor_line_raises(self, tmp_path):
        """Referencing non-existent anchor line raises ValueError."""
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", _build_basic_events(), key_col="primary_key", source_id="test", role="event")
        builder.add_derived_dimension(
            "nonexistent",
            "transactions",
            "from_account",
            "count",
            None,
            "x",
        )
        with pytest.raises(ValueError, match="nonexistent"):
            builder.build()

    def test_multiple_derived_dims(self, tmp_path):
        """Multiple derived dimensions create multiple RelationSpecs."""
        builder = _make_builder(tmp_path)
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_out",
        )
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count_distinct",
            "to_account",
            "n_targets",
        )
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count_distinct",
            "currency",
            "n_distinct_categories",
        )
        builder.add_pattern("account_pattern", "anchor", "accounts", relations=[])
        builder.build()

        rels = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())[
            "patterns"
        ]["account_pattern"]["relations"]
        assert len(rels) == 3


# ---------- add_composite_line ----------


class TestTemporalWindow:
    def test_max_window(self, tmp_path):
        """time_window='7d' with max picks the most active 7-day window."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)
        # Account A: 1 tx/week for 4 weeks, then 5 tx in week 5
        timestamps = [
            (base + timedelta(days=0)).isoformat(),
            (base + timedelta(days=7)).isoformat(),
            (base + timedelta(days=14)).isoformat(),
            (base + timedelta(days=21)).isoformat(),
            (base + timedelta(days=28)).isoformat(),
            (base + timedelta(days=29)).isoformat(),
            (base + timedelta(days=30)).isoformat(),
            (base + timedelta(days=31)).isoformat(),
            (base + timedelta(days=32)).isoformat(),
        ]
        events = pa.table(
            {
                "primary_key": [f"TX-{i}" for i in range(9)],
                "from_account": ["A"] * 9,
                "to_account": ["B", "C", "D", "E", "F", "G", "H", "I", "J"],
                "timestamp": timestamps,
            }
        )
        accounts = pa.table({"primary_key": ["A"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "burst_tx_count",
            time_col="timestamp",
            time_window="7d",
            window_aggregation="max",
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()

        vals = builder._lines["accounts"].table["burst_tx_count"].to_pylist()
        # Week 5 has 5 tx → max window = 5
        assert vals[0] == 5.0

    def test_mean_window(self, tmp_path):
        """window_aggregation='mean' averages across windows."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)
        # 2 tx in week 1, 4 tx in week 2
        timestamps = [
            base.isoformat(),
            (base + timedelta(days=1)).isoformat(),
            (base + timedelta(days=7)).isoformat(),
            (base + timedelta(days=8)).isoformat(),
            (base + timedelta(days=9)).isoformat(),
            (base + timedelta(days=10)).isoformat(),
        ]
        events = pa.table(
            {
                "primary_key": [f"TX-{i}" for i in range(6)],
                "from_account": ["A"] * 6,
                "to_account": ["B"] * 6,
                "timestamp": timestamps,
            }
        )
        accounts = pa.table({"primary_key": ["A"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "avg_weekly_tx",
            time_col="timestamp",
            time_window="7d",
            window_aggregation="mean",
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()

        vals = builder._lines["accounts"].table["avg_weekly_tx"].to_pylist()
        # Mean of [2, 4] = 3.0
        assert vals[0] == pytest.approx(3.0)


class TestAddCompositeLine:
    def test_creates_pair_line(self, tmp_path):
        """Composite line creates anchor with composite keys."""
        from hypertopos.builder.builder import RelationSpec

        builder = _make_builder(tmp_path)
        builder.add_composite_line(
            "account_pairs",
            "transactions",
            key_cols=["from_account", "to_account"],
        )
        builder.add_pattern(
            "pair_pattern", "anchor", "account_pairs",
            relations=[
                RelationSpec(line_id="account_pairs", fk_col=None, direction="self"),
            ],
        )
        builder.build()

        pairs = builder._lines["account_pairs"].table
        keys = sorted(pairs["primary_key"].to_pylist())
        # Unique (from, to) pairs: (A,B), (A,C), (B,A), (B,C), (C,A), (C,B)
        assert len(keys) == 6
        assert "A→B" in keys
        assert "C→A" in keys

    def test_composite_stores_components(self, tmp_path):
        """Composite line stores component columns as properties."""
        builder = _make_builder(tmp_path)
        builder.add_composite_line(
            "pairs",
            "transactions",
            key_cols=["from_account", "to_account"],
        )
        builder.add_derived_dimension(
            "pairs",
            "transactions",
            ["from_account", "to_account"],
            "count",
            None,
            "pair_tx_count",
        )
        builder.add_pattern("p", "anchor", "pairs", relations=[])
        builder.build()

        pairs = builder._lines["pairs"].table
        assert "from_account" in pairs.schema.names
        assert "to_account" in pairs.schema.names

    def test_derived_on_composite(self, tmp_path):
        """Derived dimensions work with composite anchor lines."""
        builder = _make_builder(tmp_path)
        builder.add_composite_line(
            "pairs",
            "transactions",
            key_cols=["from_account", "to_account"],
        )
        builder.add_derived_dimension(
            "pairs",
            "transactions",
            ["from_account", "to_account"],
            "count",
            None,
            "pair_tx_count",
        )
        builder.add_pattern("pair_pattern", "anchor", "pairs", relations=[])
        builder.build()

        pairs = builder._lines["pairs"].table
        assert "pair_tx_count" in pairs.schema.names
        # (A,B) appears twice in events → count=2
        keys = pairs["primary_key"].to_pylist()
        vals = pairs["pair_tx_count"].to_pylist()
        idx = keys.index("A→B")
        assert vals[idx] == 2.0

    def test_composite_custom_separator(self, tmp_path):
        """Composite line with custom separator works with derived dims."""
        builder = _make_builder(tmp_path)
        builder.add_composite_line(
            "pairs",
            "transactions",
            key_cols=["from_account", "to_account"],
            separator="/",
        )
        builder.add_derived_dimension(
            "pairs",
            "transactions",
            ["from_account", "to_account"],
            "count",
            None,
            "pair_count",
        )
        builder.add_pattern("pair_pattern", "anchor", "pairs", relations=[])
        builder.build()

        pairs = builder._lines["pairs"].table
        keys = pairs["primary_key"].to_pylist()
        vals = pairs["pair_count"].to_pylist()
        # Keys use "/" separator
        assert "A/B" in keys
        idx = keys.index("A/B")
        assert vals[idx] == 2.0  # (A,B) appears twice in events

    def test_missing_event_line_raises(self, tmp_path):
        """Composite referencing non-existent event line raises."""
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_composite_line("pairs", "nonexistent", ["a", "b"])
        builder.add_pattern("p", "anchor", "pairs", relations=[])
        with pytest.raises(ValueError, match="nonexistent"):
            builder.build()


# ---------- add_graph_features ----------


class TestAddGraphFeatures:
    def test_reciprocity(self, tmp_path):
        """reciprocity=1 for accounts that are both sender and receiver."""
        events = pa.table(
            {
                "primary_key": ["TX-1", "TX-2", "TX-3"],
                "from_account": ["A", "B", "C"],
                "to_account": ["B", "A", "D"],
            }
        )
        accounts = pa.table({"primary_key": ["A", "B", "C", "D"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_graph_features(
            "accounts",
            "transactions",
            "from_account",
            "to_account",
            features=["reciprocity"],
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()

        accts = builder._lines["accounts"].table
        keys = accts["primary_key"].to_pylist()
        vals = accts["reciprocity"].to_pylist()
        # A: sends to B, receives from B → reciprocal
        assert vals[keys.index("A")] == 1.0
        # B: sends to A, receives from A → reciprocal
        assert vals[keys.index("B")] == 1.0
        # C: sends only → not reciprocal
        assert vals[keys.index("C")] == 0.0
        # D: receives only → not reciprocal
        assert vals[keys.index("D")] == 0.0

    def test_in_out_degree(self, tmp_path):
        """in_degree and out_degree computed correctly."""
        events = pa.table(
            {
                "primary_key": ["TX-1", "TX-2", "TX-3", "TX-4"],
                "from_account": ["A", "A", "B", "C"],
                "to_account": ["B", "C", "C", "A"],
            }
        )
        accounts = pa.table({"primary_key": ["A", "B", "C"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_graph_features(
            "accounts",
            "transactions",
            "from_account",
            "to_account",
            features=["in_degree", "out_degree"],
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()

        accts = builder._lines["accounts"].table
        keys = accts["primary_key"].to_pylist()
        out_vals = accts["out_degree"].to_pylist()
        in_vals = accts["in_degree"].to_pylist()

        # A: sends to {B, C} → out=2, receives from {C} → in=1
        assert out_vals[keys.index("A")] == 2.0
        assert in_vals[keys.index("A")] == 1.0
        # B: sends to {C} → out=1, receives from {A} → in=1
        assert out_vals[keys.index("B")] == 1.0
        assert in_vals[keys.index("B")] == 1.0
        # C: sends to {A} → out=1, receives from {A, B} → in=2
        assert out_vals[keys.index("C")] == 1.0
        assert in_vals[keys.index("C")] == 2.0

    def test_all_features(self, tmp_path):
        """Default features = all four, creates 4 relations."""
        builder = _make_builder(tmp_path)
        builder.add_graph_features(
            "accounts",
            "transactions",
            "from_account",
            "to_account",
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()

        rels = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())[
            "patterns"
        ]["p"]["relations"]
        assert len(rels) == 4


class TestCounterpartOverlap:
    def test_full_overlap(self, tmp_path):
        """A sends to {B,C}, receives from {B,C} -> overlap=1.0"""
        events = pa.table(
            {
                "primary_key": ["TX-1", "TX-2", "TX-3", "TX-4"],
                "from_account": ["A", "A", "B", "C"],
                "to_account": ["B", "C", "A", "A"],
            }
        )
        accounts = pa.table({"primary_key": ["A", "B", "C"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_graph_features(
            "accounts",
            "transactions",
            "from_account",
            "to_account",
            features=["counterpart_overlap"],
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()
        accts = builder._lines["accounts"].table
        keys = accts["primary_key"].to_pylist()
        vals = accts["counterpart_overlap"].to_pylist()
        assert vals[keys.index("A")] == pytest.approx(1.0)

    def test_no_overlap(self, tmp_path):
        """A sends to {B}, receives from {C} -> overlap=0.0"""
        events = pa.table(
            {
                "primary_key": ["TX-1", "TX-2"],
                "from_account": ["A", "C"],
                "to_account": ["B", "A"],
            }
        )
        accounts = pa.table({"primary_key": ["A", "B", "C"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_graph_features(
            "accounts",
            "transactions",
            "from_account",
            "to_account",
            features=["counterpart_overlap"],
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()
        accts = builder._lines["accounts"].table
        keys = accts["primary_key"].to_pylist()
        vals = accts["counterpart_overlap"].to_pylist()
        assert vals[keys.index("A")] == pytest.approx(0.0)

    def test_partial_overlap(self, tmp_path):
        """A sends to {B,C,D}, receives from {C,D,E} -> Jaccard=2/4=0.5"""
        events = pa.table(
            {
                "primary_key": ["TX-1", "TX-2", "TX-3", "TX-4", "TX-5", "TX-6"],
                "from_account": ["A", "A", "A", "C", "D", "E"],
                "to_account": ["B", "C", "D", "A", "A", "A"],
            }
        )
        accounts = pa.table({"primary_key": ["A", "B", "C", "D", "E"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_graph_features(
            "accounts",
            "transactions",
            "from_account",
            "to_account",
            features=["counterpart_overlap"],
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()
        accts = builder._lines["accounts"].table
        keys = accts["primary_key"].to_pylist()
        vals = accts["counterpart_overlap"].to_pylist()
        assert vals[keys.index("A")] == pytest.approx(0.5)

    def test_default_features_include_overlap(self, tmp_path):
        """add_graph_features() with no explicit features must compute counterpart_overlap."""
        events = pa.table(
            {
                "primary_key": ["TX-1", "TX-2"],
                "from_account": ["A", "B"],
                "to_account": ["B", "A"],
            }
        )
        accounts = pa.table({"primary_key": ["A", "B"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_graph_features(
            "accounts", "transactions", "from_account", "to_account",
        )
        builder.add_pattern("p", "anchor", "accounts", relations=[])
        builder.build()
        accts = builder._lines["accounts"].table
        assert "counterpart_overlap" in accts.column_names


# ---------- Integration: full sphere build ----------


class TestAddChainLine:
    def test_creates_chain_line(self, tmp_path):
        """add_chain_line creates anchor line from chain dicts."""
        chains = [
            {
                "chain_id": "C-1",
                "keys": ["A", "B", "C"],
                "event_keys": ["TX-1", "TX-2"],
                "hop_count": 2,
                "is_cyclic": False,
                "n_distinct_categories": 1,
                "amount_decay": 0.95,
            },
            {
                "chain_id": "C-2",
                "keys": ["X", "Y", "Z", "X"],
                "event_keys": ["TX-3", "TX-4", "TX-5"],
                "hop_count": 3,
                "is_cyclic": True,
                "n_distinct_categories": 2,
                "amount_decay": 0.8,
            },
        ]
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_chain_line("tx_chains", chains)
        builder.add_pattern("chain_pattern", "anchor", "tx_chains", relations=[])
        builder.build()

        sphere = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())
        assert sphere["patterns"]["chain_pattern"]["population_size"] == 2
        assert len(sphere["patterns"]["chain_pattern"]["relations"]) == 4  # 4 default features

    def test_chain_stores_keys(self, tmp_path):
        """Chain line stores chain_keys and chain_events as properties."""
        chains = [
            {
                "chain_id": "C-1",
                "keys": ["A", "B"],
                "event_keys": ["TX-1"],
                "hop_count": 1,
                "is_cyclic": False,
                "n_distinct_categories": 1,
                "amount_decay": 1.0,
            },
        ]
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_chain_line("chains", chains)
        builder.add_pattern("p", "anchor", "chains", relations=[])
        builder.build()

        tbl = builder._lines["chains"].table
        assert "chain_keys" in tbl.schema.names
        assert tbl["chain_keys"][0].as_py() == "A,B"

    def test_empty_chains(self, tmp_path):
        """Empty chain list creates empty line — build raises on missing dimensions."""
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_chain_line("chains", [])
        builder.add_pattern("p", "anchor", "chains", relations=[])
        with pytest.raises(ValueError, match="non-empty|no dimensions"):
            builder.build()

    def test_missing_chain_keys_raises(self, tmp_path):
        """Chain dicts missing required keys raise ValueError."""
        chains = [{"chain_id": "C-1", "hop_count": 2}]  # missing is_cyclic etc
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        with pytest.raises(ValueError, match="missing keys"):
            builder.add_chain_line("chains", chains)

    def test_custom_features(self, tmp_path):
        """Custom feature list creates matching relations."""
        chains = [
            {
                "chain_id": "C-1",
                "hop_count": 3,
                "is_cyclic": True,
                "n_distinct_categories": 2,
                "amount_decay": 0.9,
                "total_amount": 5000,
                "keys": ["A", "B", "C", "A"],
                "event_keys": ["TX-1", "TX-2", "TX-3"],
            },
        ]
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_chain_line(
            "chains", chains, features=["hop_count", "is_cyclic", "total_amount"]
        )
        builder.add_pattern("p", "anchor", "chains", relations=[])
        builder.build()

        rels = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())[
            "patterns"
        ]["p"]["relations"]
        assert len(rels) == 3


# ---------- build_temporal ----------


class TestBuildTemporal:
    def _make_temporal_events(self, n_days=3):
        """Build events spread across n_days with explicit timestamps."""
        from datetime import datetime

        rows = []
        tx_id = 0
        accounts = ["A", "B", "C"]
        for day in range(1, n_days + 1):
            for _ in range(10):  # 10 events per day
                rows.append(
                    {
                        "primary_key": f"TX-{tx_id}",
                        "from_account": accounts[tx_id % 3],
                        "to_account": accounts[(tx_id + 1) % 3],
                        "amount": (tx_id + 1) * 10,
                        "timestamp": datetime(2024, 1, day),
                    }
                )
                tx_id += 1
        return pa.table({k: [r[k] for r in rows] for k in rows[0]})

    def _build_sphere_with_temporal(self, tmp_path, n_days=3, time_window="1d"):
        events = self._make_temporal_events(n_days)
        accounts = pa.table({"primary_key": ["A", "B", "C"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
        )
        builder.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
        builder.build()
        result = builder.build_temporal("timestamp", time_window)
        return builder, result

    def test_build_temporal_creates_dataset(self, tmp_path):
        """Temporal build creates Lance dataset with expected row count."""
        _builder, result = self._build_sphere_with_temporal(tmp_path, n_days=3)

        assert "acct_pattern" in result
        assert result["acct_pattern"] == 3  # 3 days → 3 slices

        lance_path = tmp_path / "sphere" / "temporal" / "acct_pattern" / "data.lance"
        assert lance_path.exists()

        import lance

        ds = lance.dataset(str(lance_path))
        # 3 accounts × 3 day-slices = 9 rows
        assert ds.count_rows() == 9

    def test_build_temporal_shape_range(self, tmp_path):
        """All shape_snapshot values are in [0.0, 1.0]."""
        _builder, _result = self._build_sphere_with_temporal(tmp_path, n_days=3)

        import lance

        lance_path = tmp_path / "sphere" / "temporal" / "acct_pattern" / "data.lance"
        ds = lance.dataset(str(lance_path))
        tbl = ds.to_table()
        shapes = tbl["shape_snapshot"].to_pylist()

        for shape_vec in shapes:
            for val in shape_vec:
                assert 0.0 <= val <= 1.0, f"shape value {val} out of [0,1] range"

    def test_build_temporal_multiple_slices(self, tmp_path):
        """Slices are sequential (0, 1, 2, ...) for multi-day data."""
        _builder, result = self._build_sphere_with_temporal(
            tmp_path,
            n_days=5,
            time_window="1d",
        )
        assert result["acct_pattern"] == 5

        import lance

        lance_path = tmp_path / "sphere" / "temporal" / "acct_pattern" / "data.lance"
        ds = lance.dataset(str(lance_path))
        tbl = ds.to_table()
        slice_indices = sorted(set(tbl["slice_index"].to_pylist()))
        assert slice_indices == [0, 1, 2, 3, 4]

    def test_build_temporal_requires_build_first(self, tmp_path):
        """Calling build_temporal without build() raises ValueError."""
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        events = self._make_temporal_events(3)
        accounts = pa.table({"primary_key": ["A", "B", "C"]})
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
        )
        builder.add_pattern("acct_pattern", "anchor", "accounts", relations=[])

        with pytest.raises(ValueError, match="build\\(\\) must be called"):
            builder.build_temporal("timestamp", "1d")


class TestIntegrationBuild:
    def test_full_build_with_all_features(self, tmp_path):
        """Complete sphere with derived dims + composite + graph features."""
        builder = _make_builder(tmp_path)

        # Derived dimensions
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_out",
        )
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count_distinct",
            "currency",
            "n_distinct_categories",
        )

        # Graph features
        builder.add_graph_features(
            "accounts",
            "transactions",
            "from_account",
            "to_account",
            features=["reciprocity"],
        )

        # Composite line with derived dim
        builder.add_composite_line(
            "pairs",
            "transactions",
            key_cols=["from_account", "to_account"],
        )
        builder.add_derived_dimension(
            "pairs",
            "transactions",
            ["from_account", "to_account"],
            "count",
            None,
            "pair_count",
        )

        # Patterns
        builder.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
        builder.add_pattern("pair_pattern", "anchor", "pairs", relations=[])

        builder.build()

        sphere = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())
        assert sphere["sphere_id"] == "test"

        # Account pattern: 3 dims (tx_out, n_distinct_categories, reciprocity)
        acct_rels = sphere["patterns"]["acct_pattern"]["relations"]
        assert len(acct_rels) == 3

        # Pair pattern: 1 dim (pair_count)
        pair_rels = sphere["patterns"]["pair_pattern"]["relations"]
        assert len(pair_rels) == 1

        # Sphere has geometry for both patterns
        geo_path = tmp_path / "sphere" / "geometry"
        assert (geo_path / "acct_pattern").exists()
        assert (geo_path / "pair_pattern").exists()


# ---------- Chunked geometry write ----------


class TestChunkedGeometryWrite:
    def test_chunked_write_produces_valid_geometry(self, tmp_path, monkeypatch):
        """When entity count > GEOMETRY_CHUNK_SIZE, chunked path is used and geometry is valid."""
        import hypertopos.builder.builder as builder_mod

        # Patch chunk size to something tiny so we can test with ~500 entities
        monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 200)

        n_entities = 500
        accounts = pa.table({"primary_key": [f"A-{i}" for i in range(n_entities)]})
        # Simple events: each account sends 1-3 transactions
        n_tx = n_entities * 2
        tx_keys = [f"TX-{i}" for i in range(n_tx)]
        from_accts = [f"A-{i % n_entities}" for i in range(n_tx)]
        to_accts = [f"A-{(i + 1) % n_entities}" for i in range(n_tx)]

        events = pa.table(
            {
                "primary_key": tx_keys,
                "from_account": from_accts,
                "to_account": to_accts,
            }
        )

        builder = GDSBuilder("test_chunked", str(tmp_path / "sphere"))
        builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
        )
        builder.add_pattern("acct_pat", "anchor", "accounts", relations=[])
        builder.build()

        # Verify sphere.json
        sphere = json.loads((tmp_path / "sphere" / "_gds_meta" / "sphere.json").read_text())
        assert sphere["patterns"]["acct_pat"]["population_size"] == n_entities

        # Verify geometry Lance dataset has correct row count
        import lance

        geo_path = tmp_path / "sphere" / "geometry" / "acct_pat" / "v=1" / "data.lance"
        assert geo_path.exists()
        ds = lance.dataset(str(geo_path))
        assert ds.count_rows() == n_entities

    def test_chunked_matches_unchunked_results(self, tmp_path, monkeypatch):
        """Chunked and unchunked paths produce identical mu/sigma/theta."""
        import hypertopos.builder.builder as builder_mod

        n_entities = 100
        accounts = pa.table({"primary_key": [f"A-{i}" for i in range(n_entities)]})
        n_tx = n_entities * 2
        events = pa.table(
            {
                "primary_key": [f"TX-{i}" for i in range(n_tx)],
                "from_account": [f"A-{i % n_entities}" for i in range(n_tx)],
                "to_account": [f"A-{(i + 1) % n_entities}" for i in range(n_tx)],
            }
        )

        # Build unchunked (chunk_size > n_entities)
        monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 999_999)
        b1 = GDSBuilder("test_unchunked", str(tmp_path / "sphere1"))
        b1.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        b1.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        b1.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
        )
        b1.add_pattern("p", "anchor", "accounts", relations=[])
        b1.build()

        sphere1 = json.loads((tmp_path / "sphere1" / "_gds_meta" / "sphere.json").read_text())

        # Build chunked (chunk_size < n_entities)
        monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 30)
        b2 = GDSBuilder("test_chunked", str(tmp_path / "sphere2"))
        b2.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
        b2.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
        b2.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
        )
        b2.add_pattern("p", "anchor", "accounts", relations=[])
        b2.build()

        sphere2 = json.loads((tmp_path / "sphere2" / "_gds_meta" / "sphere.json").read_text())

        # mu, sigma, theta must be identical (computed from full population)
        p1, p2 = sphere1["patterns"]["p"], sphere2["patterns"]["p"]
        np.testing.assert_allclose(p1["mu"], p2["mu"], atol=1e-6)
        np.testing.assert_allclose(p1["sigma_diag"], p2["sigma_diag"], atol=1e-6)
        np.testing.assert_allclose(p1["theta"], p2["theta"], atol=1e-6)
        assert p1["population_size"] == p2["population_size"]


# ---------- find_counterparties ----------


def _build_counterparty_sphere(tmp_path):
    """Build a sphere with transactions for counterparty tests."""
    events = pa.table(
        {
            "primary_key": [f"TX-{i}" for i in range(8)],
            "from_account": ["A", "A", "A", "B", "B", "C", "C", "C"],
            "to_account": ["B", "B", "C", "A", "C", "A", "A", "B"],
        }
    )
    accounts = pa.table({"primary_key": ["A", "B", "C"]})
    builder = GDSBuilder("test_cp", str(tmp_path / "sphere"))
    builder.add_line("transactions", events, key_col="primary_key", source_id="test", role="event")
    builder.add_line("accounts", accounts, key_col="primary_key", source_id="test", role="anchor")
    builder.add_derived_dimension(
        "accounts",
        "transactions",
        "from_account",
        "count",
        None,
        "tx_out_count",
    )
    builder.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
    builder.build()
    return str(tmp_path / "sphere")


class TestFindCounterparties:
    def test_outgoing_counterparties(self, tmp_path):
        """Finds accounts that entity sends to, sorted by count desc."""
        from hypertopos.sphere import HyperSphere

        sphere_path = _build_counterparty_sphere(tmp_path)
        hs = HyperSphere.open(sphere_path)
        nav = hs.session("test").navigator()

        result = nav.find_counterparties(
            "A",
            "transactions",
            "from_account",
            "to_account",
        )
        assert result["primary_key"] == "A"
        out = result["outgoing"]
        # A sends to B(2x) and C(1x)
        assert len(out) == 2
        assert out[0]["key"] == "B"
        assert out[0]["tx_count"] == 2
        assert out[1]["key"] == "C"
        assert out[1]["tx_count"] == 1

    def test_incoming_counterparties(self, tmp_path):
        """Finds accounts that send to entity, sorted by count desc."""
        from hypertopos.sphere import HyperSphere

        sphere_path = _build_counterparty_sphere(tmp_path)
        hs = HyperSphere.open(sphere_path)
        nav = hs.session("test").navigator()

        result = nav.find_counterparties(
            "A",
            "transactions",
            "from_account",
            "to_account",
        )
        inc = result["incoming"]
        # A receives from: B(1x), C(2x) → sorted: C(2), B(1)
        assert len(inc) == 2
        assert inc[0]["key"] == "C"
        assert inc[0]["tx_count"] == 2
        assert inc[1]["key"] == "B"
        assert inc[1]["tx_count"] == 1

    def test_summary_counts(self, tmp_path):
        """Summary includes total counts per direction."""
        from hypertopos.sphere import HyperSphere

        sphere_path = _build_counterparty_sphere(tmp_path)
        hs = HyperSphere.open(sphere_path)
        nav = hs.session("test").navigator()

        result = nav.find_counterparties(
            "A",
            "transactions",
            "from_account",
            "to_account",
        )
        s = result["summary"]
        assert s["total_outgoing"] == 2
        assert s["total_incoming"] == 2
        # Without pattern_id, anomalous counts are 0
        assert s["anomalous_outgoing"] == 0
        assert s["anomalous_incoming"] == 0

    def test_anomaly_enrichment(self, tmp_path):
        """With pattern_id, counterparties get is_anomaly + delta_rank_pct."""
        from hypertopos.sphere import HyperSphere

        sphere_path = _build_counterparty_sphere(tmp_path)
        hs = HyperSphere.open(sphere_path)
        nav = hs.session("test").navigator()

        result = nav.find_counterparties(
            "A",
            "transactions",
            "from_account",
            "to_account",
            pattern_id="acct_pattern",
        )
        out = result["outgoing"]
        # With enrichment, entries should have is_anomaly and delta_rank_pct
        for entry in out:
            assert "is_anomaly" in entry
            assert "delta_rank_pct" in entry
            assert isinstance(entry["is_anomaly"], bool)

    def test_top_n_limits(self, tmp_path):
        """top_n limits results per direction."""
        from hypertopos.sphere import HyperSphere

        sphere_path = _build_counterparty_sphere(tmp_path)
        hs = HyperSphere.open(sphere_path)
        nav = hs.session("test").navigator()

        result = nav.find_counterparties(
            "A",
            "transactions",
            "from_account",
            "to_account",
            top_n=1,
        )
        # A sends to 2 different accounts but top_n=1
        assert len(result["outgoing"]) == 1
        assert len(result["incoming"]) == 1

    def test_unknown_line_raises(self, tmp_path):
        """Referencing unknown line raises GDSNavigationError."""
        from hypertopos.navigation.navigator import GDSNavigationError
        from hypertopos.sphere import HyperSphere

        sphere_path = _build_counterparty_sphere(tmp_path)
        hs = HyperSphere.open(sphere_path)
        nav = hs.session("test").navigator()

        with pytest.raises(GDSNavigationError, match="nonexistent"):
            nav.find_counterparties(
                "A",
                "nonexistent",
                "from_account",
                "to_account",
            )

    def test_unknown_column_raises(self, tmp_path):
        """Referencing unknown column raises GDSNavigationError."""
        from hypertopos.navigation.navigator import GDSNavigationError
        from hypertopos.sphere import HyperSphere

        sphere_path = _build_counterparty_sphere(tmp_path)
        hs = HyperSphere.open(sphere_path)
        nav = hs.session("test").navigator()

        with pytest.raises(GDSNavigationError, match="bad_col"):
            nav.find_counterparties(
                "A",
                "transactions",
                "bad_col",
                "to_account",
            )
