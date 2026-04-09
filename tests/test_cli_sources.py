# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for hypertopos.cli.sources — all source loading tiers."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypertopos.cli.schema import JoinSpec, SourceConfig, TransformSpec
from hypertopos.cli.sources import load_source

# ── Tier 1: single file ─────────────────────────────────────────────


class TestTier1SingleFile:
    def test_load_parquet(self, tmp_path):
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        path = tmp_path / "data.parquet"
        pq.write_table(table, str(path))

        cfg = SourceConfig(path="data.parquet")
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 3
        assert result.schema.names == ["id", "name"]

    def test_load_csv_comma(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("id,name\n1,alice\n2,bob\n")

        cfg = SourceConfig(path="data.csv")
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 2
        assert "id" in result.schema.names
        assert "name" in result.schema.names

    def test_load_arrow_ipc(self, tmp_path):
        table = pa.table({"x": [10, 20], "y": [1.5, 2.5]})
        path = tmp_path / "data.arrow"
        with pa.ipc.new_file(str(path), table.schema) as writer:
            writer.write_table(table)

        cfg = SourceConfig(path="data.arrow")
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 2
        assert result.column("x").to_pylist() == [10, 20]

    def test_file_not_found_raises(self, tmp_path):
        cfg = SourceConfig(path="nonexistent.parquet")
        with pytest.raises(FileNotFoundError, match="not found"):
            load_source(cfg, tmp_path)


# ── Tier 1b: CSV with options ───────────────────────────────────────


class TestTier1bCSVOptions:
    def test_semicolon_delimiter(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("id;amount;name\n1;100.5;alice\n2;200.3;bob\n")

        cfg = SourceConfig(path="data.csv", format="csv", delimiter=";")
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 2
        assert result.column("amount").to_pylist() == [100.5, 200.3]

    def test_tab_delimiter(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("id\tname\n1\talice\n2\tbob\n")

        cfg = SourceConfig(path="data.tsv", format="csv", delimiter="\t")
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 2
        assert result.column("name").to_pylist() == ["alice", "bob"]


# ── Tier 1c: transforms ─────────────────────────────────────────────


class TestTier1cTransforms:
    def test_cast_type(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table({"id": [1, 2, 3], "amount": ["100", "200", "300"]})
        pq.write_table(table, str(path))

        cfg = SourceConfig(
            path="data.parquet",
            transform={"amount": TransformSpec(type="float64")},
        )
        result = load_source(cfg, tmp_path)

        assert result.schema.field("amount").type == pa.float64()
        assert result.column("amount").to_pylist() == [100.0, 200.0, 300.0]

    def test_fill_null_string(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table(
            {
                "id": [1, 2, 3],
                "bank": pa.array(["AB", None, "CD"], type=pa.string()),
            }
        )
        pq.write_table(table, str(path))

        cfg = SourceConfig(
            path="data.parquet",
            transform={"bank": TransformSpec(fill_null="")},
        )
        result = load_source(cfg, tmp_path)

        assert result.column("bank").to_pylist() == ["AB", "", "CD"]

    def test_fill_null_numeric(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table(
            {
                "id": [1, 2, 3],
                "amount": pa.array([10.0, None, 30.0], type=pa.float64()),
            }
        )
        pq.write_table(table, str(path))

        cfg = SourceConfig(
            path="data.parquet",
            transform={"amount": TransformSpec(fill_null=0.0)},
        )
        result = load_source(cfg, tmp_path)

        assert result.column("amount").to_pylist() == [10.0, 0.0, 30.0]

    def test_cast_and_fill_null_combined(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table(
            {
                "id": [1, 2, 3],
                "score": pa.array([10, None, 30], type=pa.int64()),
            }
        )
        pq.write_table(table, str(path))

        cfg = SourceConfig(
            path="data.parquet",
            transform={"score": TransformSpec(type="float64", fill_null=0)},
        )
        result = load_source(cfg, tmp_path)

        assert result.schema.field("score").type == pa.float64()
        assert result.column("score").to_pylist() == [10.0, 0.0, 30.0]

    def test_transform_unknown_column_ignored(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table({"id": [1, 2]})
        pq.write_table(table, str(path))

        cfg = SourceConfig(
            path="data.parquet",
            transform={"nonexistent": TransformSpec(type="float64")},
        )
        result = load_source(cfg, tmp_path)
        assert result.num_rows == 2

    def test_unknown_cast_type_raises(self, tmp_path):
        path = tmp_path / "data.parquet"
        table = pa.table({"id": [1, 2], "x": [1, 2]})
        pq.write_table(table, str(path))

        cfg = SourceConfig(
            path="data.parquet",
            transform={"x": TransformSpec(type="bogus")},
        )
        with pytest.raises(ValueError, match="Unknown cast type"):
            load_source(cfg, tmp_path)


# ── Tier 2: multi-file join ─────────────────────────────────────────


class TestTier2Join:
    def test_left_join(self, tmp_path):
        base = pa.table(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
            }
        )
        other = pa.table(
            {
                "id": [1, 3],
                "score": [100, 300],
            }
        )
        pq.write_table(base, str(tmp_path / "base.parquet"))
        pq.write_table(other, str(tmp_path / "other.parquet"))

        cfg = SourceConfig(
            path="base.parquet",
            join=[JoinSpec(file="other.parquet", on="id", type="left")],
        )
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 3
        assert "score" in result.schema.names
        rows = {
            result.column("id")[i].as_py(): result.column("score")[i].as_py()
            for i in range(result.num_rows)
        }
        assert rows[1] == 100
        assert rows[2] is None
        assert rows[3] == 300

    def test_inner_join(self, tmp_path):
        base = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        other = pa.table({"id": [1, 3], "score": [100, 300]})
        pq.write_table(base, str(tmp_path / "base.parquet"))
        pq.write_table(other, str(tmp_path / "other.parquet"))

        cfg = SourceConfig(
            path="base.parquet",
            join=[JoinSpec(file="other.parquet", on="id", type="inner")],
        )
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 2
        assert result.column("name").to_pylist() == ["a", "c"]

    def test_join_with_column_selection(self, tmp_path):
        base = pa.table({"id": [1, 2], "name": ["a", "b"]})
        other = pa.table(
            {
                "id": [1, 2],
                "score": [100, 200],
                "rank": [1, 2],
                "extra": ["x", "y"],
            }
        )
        pq.write_table(base, str(tmp_path / "base.parquet"))
        pq.write_table(other, str(tmp_path / "other.parquet"))

        cfg = SourceConfig(
            path="base.parquet",
            join=[
                JoinSpec(
                    file="other.parquet",
                    on="id",
                    type="left",
                    columns=["score"],
                )
            ],
        )
        result = load_source(cfg, tmp_path)

        assert "score" in result.schema.names
        assert "rank" not in result.schema.names
        assert "extra" not in result.schema.names

    def test_multiple_joins(self, tmp_path):
        base = pa.table({"id": [1, 2, 3]})
        loans = pa.table({"id": [1, 2], "amount": [1000, 2000]})
        cards = pa.table({"id": [2, 3], "card_type": ["gold", "silver"]})
        pq.write_table(base, str(tmp_path / "base.parquet"))
        pq.write_table(loans, str(tmp_path / "loans.parquet"))
        pq.write_table(cards, str(tmp_path / "cards.parquet"))

        cfg = SourceConfig(
            path="base.parquet",
            join=[
                JoinSpec(file="loans.parquet", on="id", type="left"),
                JoinSpec(file="cards.parquet", on="id", type="left"),
            ],
        )
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 3
        assert "amount" in result.schema.names
        assert "card_type" in result.schema.names

    def test_join_then_transform(self, tmp_path):
        base = pa.table({"id": [1, 2]})
        other = pa.table(
            {
                "id": [1, 2],
                "amount": pa.array([100, None], type=pa.int64()),
            }
        )
        pq.write_table(base, str(tmp_path / "base.parquet"))
        pq.write_table(other, str(tmp_path / "other.parquet"))

        cfg = SourceConfig(
            path="base.parquet",
            join=[JoinSpec(file="other.parquet", on="id", type="left")],
            transform={"amount": TransformSpec(type="float64", fill_null=0.0)},
        )
        result = load_source(cfg, tmp_path)

        assert result.schema.field("amount").type == pa.float64()
        assert result.column("amount").to_pylist() == [100.0, 0.0]


# ── Tier 3: script ──────────────────────────────────────────────────


class TestTier3Script:
    def test_load_script(self, tmp_path):
        script = tmp_path / "prep.py"
        script.write_text(
            "import pyarrow as pa\n"
            "def prepare() -> pa.Table:\n"
            "    return pa.table({'pk': ['a', 'b'], 'val': [1, 2]})\n"
        )

        cfg = SourceConfig(script="prep.py")
        result = load_source(cfg, tmp_path)

        assert result.num_rows == 2
        assert result.column("pk").to_pylist() == ["a", "b"]

    def test_script_missing_prepare_raises(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("x = 1\n")

        cfg = SourceConfig(script="bad.py")
        with pytest.raises(ValueError, match="prepare"):
            load_source(cfg, tmp_path)

    def test_script_wrong_return_type_raises(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("def prepare(): return [1, 2, 3]\n")

        cfg = SourceConfig(script="bad.py")
        with pytest.raises(ValueError, match="pyarrow.Table"):
            load_source(cfg, tmp_path)

    def test_script_not_found_raises(self, tmp_path):
        cfg = SourceConfig(script="nonexistent.py")
        with pytest.raises(FileNotFoundError, match="not found"):
            load_source(cfg, tmp_path)
