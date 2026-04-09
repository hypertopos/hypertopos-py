# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for full-text search via Lance INVERTED index."""

from __future__ import annotations

import pyarrow as pa
import pytest
from hypertopos.builder.builder import GDSBuilder
from hypertopos.storage.reader import GDSReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fts_builder(tmp_path):
    """Build a minimal sphere with multi-word customer names for FTS testing."""
    b = GDSBuilder("fts_sphere", str(tmp_path / "gds_fts"))
    b.add_line(
        "customers",
        [
            {"cust_id": "C-001", "name": "Acme Corp", "segment": "enterprise"},
            {"cust_id": "C-002", "name": "Bob Industries", "segment": "mid-market"},
            {"cust_id": "C-003", "name": "Acme Ltd", "segment": "enterprise"},
            {"cust_id": "C-004", "name": "Delta Software", "segment": "smb"},
            {"cust_id": "C-005", "name": "Gamma Services", "segment": "enterprise"},
        ],
        key_col="cust_id",
        source_id="test",
    )
    b.add_pattern(
        "customer_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[],
    )
    return b


# ---------------------------------------------------------------------------
# GDSReader.search_points_fts tests
# ---------------------------------------------------------------------------


def test_search_points_fts_partial_match(tmp_path):
    """search_points_fts returns rows where query token appears in any string column."""
    b = _make_fts_builder(tmp_path)
    out = b.build()

    reader = GDSReader(out)
    result = reader.search_points_fts("customers", 1, "acme", limit=10)
    assert isinstance(result, pa.Table)
    assert result.num_rows > 0
    names = result["name"].to_pylist()
    assert any("acme" in str(n).lower() for n in names)


def test_search_points_fts_no_match(tmp_path):
    """search_points_fts returns empty table when no entity matches the query."""
    b = _make_fts_builder(tmp_path)
    out = b.build()

    reader = GDSReader(out)
    result = reader.search_points_fts("customers", 1, "xyzzy_no_match_99", limit=10)
    assert result.num_rows == 0


def test_search_points_fts_respects_limit(tmp_path):
    """search_points_fts respects the limit parameter."""
    b = _make_fts_builder(tmp_path)
    out = b.build()

    reader = GDSReader(out)
    # "a" is a common token — should match many but limit=2 caps output
    result = reader.search_points_fts("customers", 1, "a", limit=2)
    assert result.num_rows <= 2


def test_search_points_fts_raises_on_missing_dataset(tmp_path):
    """search_points_fts raises ValueError when Lance dataset does not exist."""
    reader = GDSReader(str(tmp_path))
    with pytest.raises(ValueError, match="FTS requires Lance points"):
        reader.search_points_fts("nonexistent_line", 1, "query")


def test_search_points_fts_returns_score_column(tmp_path):
    """search_points_fts result includes _score column (BM25 relevance)."""
    b = _make_fts_builder(tmp_path)
    out = b.build()

    reader = GDSReader(out)
    result = reader.search_points_fts("customers", 1, "acme", limit=10)
    assert "_score" in result.schema.names


def test_search_points_fts_searches_all_string_columns(tmp_path):
    """search_points_fts matches tokens in any indexed string column, not only name."""
    b = _make_fts_builder(tmp_path)
    out = b.build()

    reader = GDSReader(out)
    # "enterprise" only appears in the segment column, not in name
    result = reader.search_points_fts("customers", 1, "enterprise", limit=10)
    assert result.num_rows > 0
    segments = result["segment"].to_pylist()
    assert any("enterprise" in str(s).lower() for s in segments)


# ---------------------------------------------------------------------------
# Navigator.search_entities_fts tests
# ---------------------------------------------------------------------------


def test_navigator_search_entities_fts_partial_match(tmp_path):
    """search_entities_fts returns matching entities as list[dict]."""
    from hypertopos.sphere import HyperSphere

    b = _make_fts_builder(tmp_path)
    out = b.build()

    sphere = HyperSphere.open(out)
    session = sphere.session("test_agent_fts")
    nav = session.navigator()

    results = nav.search_entities_fts("customers", "acme", limit=10)
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("acme" in str(r.get("properties", {}).get("name", "")).lower() for r in results)
    session.close()


def test_navigator_search_entities_fts_no_match(tmp_path):
    """search_entities_fts returns empty list when query has no match."""
    from hypertopos.sphere import HyperSphere

    b = _make_fts_builder(tmp_path)
    out = b.build()

    sphere = HyperSphere.open(out)
    session = sphere.session("test_agent_fts_empty")
    nav = session.navigator()

    results = nav.search_entities_fts("customers", "xyzzy_no_match_99")
    assert results == []
    session.close()


def test_navigator_search_entities_fts_respects_limit(tmp_path):
    """search_entities_fts respects the limit parameter."""
    from hypertopos.sphere import HyperSphere

    b = _make_fts_builder(tmp_path)
    out = b.build()

    sphere = HyperSphere.open(out)
    session = sphere.session("test_agent_fts_limit")
    nav = session.navigator()

    results = nav.search_entities_fts("customers", "a", limit=2)
    assert len(results) <= 2
    session.close()


def test_navigator_search_entities_fts_result_format(tmp_path):
    """search_entities_fts results have primary_key, status, and properties keys."""
    from hypertopos.sphere import HyperSphere

    b = _make_fts_builder(tmp_path)
    out = b.build()

    sphere = HyperSphere.open(out)
    session = sphere.session("test_agent_fts_format")
    nav = session.navigator()

    results = nav.search_entities_fts("customers", "acme", limit=5)
    assert len(results) > 0
    for r in results:
        assert "primary_key" in r
        assert "status" in r
        assert "properties" in r
        # _score must NOT appear in top-level result or properties
        assert "_score" not in r
        assert "_score" not in r.get("properties", {})
    session.close()


# ---------------------------------------------------------------------------
# write_points — INVERTED index built at write time
# ---------------------------------------------------------------------------


def test_write_points_creates_inverted_index(tmp_path):
    """write_points builds INVERTED indices on string columns."""
    import lance
    import pyarrow as pa  # noqa: F811
    from hypertopos.builder._writer import write_points

    table = pa.table(
        {
            "primary_key": ["A-001", "A-002", "A-003"],
            "version": pa.array([1, 1, 1], type=pa.int32()),
            "status": ["active", "active", "active"],
            "name": ["Acme Corp", "Beta Ltd", "Gamma Inc"],
        }
    )
    write_points(tmp_path, "items", table, version=1, partition_col=None)

    lance_path = tmp_path / "points" / "items" / "v=1" / "data.lance"
    assert lance_path.exists()
    ds = lance.dataset(str(lance_path))
    indices = ds.list_indices()
    index_names = [(idx["name"] if isinstance(idx, dict) else idx.name) for idx in indices]
    # At least one INVERTED index should exist on the 'name' or 'status' column
    assert any(
        "name" in str(n).lower() or "status" in str(n).lower() or "inverted" in str(n).lower()
        for n in index_names
    ), f"No INVERTED index found. Indices: {index_names}"


def test_write_points_fts_search_after_build(tmp_path):
    """After write_points with INVERTED index, FTS search returns correct results."""
    import lance
    import pyarrow as pa  # noqa: F811
    from hypertopos.builder._writer import write_points

    table = pa.table(
        {
            "primary_key": ["A-001", "A-002", "A-003"],
            "version": pa.array([1, 1, 1], type=pa.int32()),
            "status": ["active", "active", "active"],
            "name": ["Acme Corp", "Beta Ltd", "Acme Industries"],
        }
    )
    write_points(tmp_path, "items", table, version=1, partition_col=None)

    lance_path = tmp_path / "points" / "items" / "v=1" / "data.lance"
    ds = lance.dataset(str(lance_path))
    result = ds.scanner(full_text_query="acme", limit=10).to_table()
    assert result.num_rows == 2
    names = result["name"].to_pylist()
    assert all("acme" in n.lower() for n in names)


# ---------------------------------------------------------------------------
# fts_columns parameter — opt-in FTS per line
# ---------------------------------------------------------------------------


def _get_inverted_columns(lance_path) -> set[str]:
    """Return set of column names that have an INVERTED index."""
    import lance

    ds = lance.dataset(str(lance_path))
    result = set()
    for idx in ds.list_indices():
        idx_type = idx.get("type", "") if isinstance(idx, dict) else getattr(idx, "index_type", "")
        fields = idx.get("fields", []) if isinstance(idx, dict) else getattr(idx, "columns", [])
        if "inverted" in str(idx_type).lower() or "inverted" in str(idx).lower():
            for f in fields:
                result.add(f)
    return result


def test_write_points_fts_columns_all(tmp_path):
    """fts_columns='all' creates INVERTED on all string columns (except primary_key)."""
    import lance
    from hypertopos.builder._writer import write_points

    table = pa.table(
        {
            "primary_key": ["A-001", "A-002"],
            "version": pa.array([1, 1], type=pa.int32()),
            "status": ["active", "active"],
            "name": ["Acme Corp", "Beta Ltd"],
            "segment": ["enterprise", "smb"],
        }
    )
    write_points(tmp_path, "items", table, version=1, partition_col=None, fts_columns="all")

    lance_path = tmp_path / "points" / "items" / "v=1" / "data.lance"
    ds = lance.dataset(str(lance_path))
    # FTS search should work
    result = ds.scanner(full_text_query="acme", limit=10).to_table()
    assert result.num_rows >= 1


def test_write_points_fts_columns_empty_list(tmp_path):
    """fts_columns=[] creates NO INVERTED indices — FTS search raises."""
    import lance
    from hypertopos.builder._writer import write_points

    table = pa.table(
        {
            "primary_key": ["A-001", "A-002"],
            "version": pa.array([1, 1], type=pa.int32()),
            "status": ["active", "active"],
            "name": ["Acme Corp", "Beta Ltd"],
        }
    )
    write_points(tmp_path, "items", table, version=1, partition_col=None, fts_columns=[])

    lance_path = tmp_path / "points" / "items" / "v=1" / "data.lance"
    ds = lance.dataset(str(lance_path))
    # No INVERTED index — FTS query should fail
    with pytest.raises(ValueError):
        ds.scanner(full_text_query="acme", limit=10).to_table()


def test_write_points_fts_columns_selective(tmp_path):
    """fts_columns=['name'] creates INVERTED only on 'name', not on other string cols."""
    import lance
    from hypertopos.builder._writer import write_points

    table = pa.table(
        {
            "primary_key": ["A-001", "A-002"],
            "version": pa.array([1, 1], type=pa.int32()),
            "status": ["active", "active"],
            "name": ["Acme Corp", "Beta Ltd"],
            "segment": ["enterprise", "smb"],
        }
    )
    write_points(
        tmp_path,
        "items",
        table,
        version=1,
        partition_col=None,
        fts_columns=["name"],
    )

    lance_path = tmp_path / "points" / "items" / "v=1" / "data.lance"
    ds = lance.dataset(str(lance_path))
    # FTS on 'name' should work (Acme appears in name)
    result = ds.scanner(full_text_query="acme", limit=10).to_table()
    assert result.num_rows >= 1
    # FTS on 'segment' only term should NOT match (no INVERTED on segment)
    result_seg = ds.scanner(full_text_query="enterprise", limit=10).to_table()
    assert result_seg.num_rows == 0


def test_builder_event_line_no_fts_by_default(tmp_path):
    """Event lines get no INVERTED indices by default (fts_columns=None resolves to [])."""
    import json
    from pathlib import Path

    import lance

    b = GDSBuilder("fts_test", str(tmp_path / "gds"))
    b.add_line(
        "transactions",
        [
            {"tx_id": "T-001", "description": "payment", "amount": 100.0},
            {"tx_id": "T-002", "description": "refund", "amount": -50.0},
        ],
        key_col="tx_id",
        source_id="test",
        role="event",
    )
    b.add_line(
        "customers",
        [
            {"cust_id": "C-001", "name": "Acme Corp"},
        ],
        key_col="cust_id",
        source_id="test",
        role="anchor",
    )
    b.add_pattern(
        "cust_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[],
    )
    out = Path(b.build())

    # Event line should have no FTS
    tx_lance = out / "points" / "transactions" / "v=1" / "data.lance"
    ds_tx = lance.dataset(str(tx_lance))
    with pytest.raises(ValueError):
        ds_tx.scanner(full_text_query="payment", limit=10).to_table()

    # Anchor line should have FTS
    cust_lance = out / "points" / "customers" / "v=1" / "data.lance"
    ds_cust = lance.dataset(str(cust_lance))
    result = ds_cust.scanner(full_text_query="acme", limit=10).to_table()
    assert result.num_rows >= 1

    # sphere.json should record fts_columns
    sphere_json = json.loads((out / "_gds_meta" / "sphere.json").read_text())
    assert sphere_json["lines"]["transactions"]["fts_columns"] == []
    assert sphere_json["lines"]["customers"]["fts_columns"] == "all"


def test_builder_explicit_fts_columns_on_event(tmp_path):
    """Explicit fts_columns on event line overrides the auto-none default."""
    import json
    from pathlib import Path

    import lance

    b = GDSBuilder("fts_test", str(tmp_path / "gds"))
    b.add_line(
        "transactions",
        [
            {"tx_id": "T-001", "description": "payment", "amount": 100.0},
            {"tx_id": "T-002", "description": "refund", "amount": -50.0},
        ],
        key_col="tx_id",
        source_id="test",
        role="event",
        fts_columns=["description"],
    )
    b.add_line(
        "customers",
        [
            {"cust_id": "C-001", "name": "Acme Corp"},
        ],
        key_col="cust_id",
        source_id="test",
        role="anchor",
    )
    b.add_pattern(
        "cust_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[],
    )
    out = Path(b.build())

    # Event line should have FTS on 'description' only
    tx_lance = out / "points" / "transactions" / "v=1" / "data.lance"
    ds_tx = lance.dataset(str(tx_lance))
    result = ds_tx.scanner(full_text_query="payment", limit=10).to_table()
    assert result.num_rows >= 1

    # sphere.json should record the explicit columns
    sphere_json = json.loads((out / "_gds_meta" / "sphere.json").read_text())
    assert sphere_json["lines"]["transactions"]["fts_columns"] == ["description"]


def test_line_model_has_fts():
    """Line.has_fts() reflects fts_columns correctly."""
    from hypertopos.model.sphere import Line, PartitionConfig

    base = {
        "line_id": "x",
        "entity_type": "x",
        "pattern_id": "",
        "partitioning": PartitionConfig(mode="static", columns=[]),
        "versions": [1],
    }
    # anchor with fts_columns=None → auto → True
    line_anchor = Line(**base, line_role="anchor", fts_columns=None)
    assert line_anchor.has_fts() is True

    # event with fts_columns=None → auto → False
    line_event = Line(**base, line_role="event", fts_columns=None)
    assert line_event.has_fts() is False

    # explicit "all" → True
    line_all = Line(**base, line_role="event", fts_columns="all")
    assert line_all.has_fts() is True

    # explicit list with items → True
    line_list = Line(**base, line_role="event", fts_columns=["name"])
    assert line_list.has_fts() is True

    # explicit empty list → False
    line_empty = Line(**base, line_role="anchor", fts_columns=[])
    assert line_empty.has_fts() is False


def test_reader_parses_fts_columns(tmp_path):
    """GDSReader parses fts_columns from sphere.json into Line model."""

    b = GDSBuilder("fts_test", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [
            {"cust_id": "C-001", "name": "Acme Corp"},
        ],
        key_col="cust_id",
        source_id="test",
        role="anchor",
    )
    b.add_line(
        "events",
        [
            {"ev_id": "E-001", "desc": "sale"},
        ],
        key_col="ev_id",
        source_id="test",
        role="event",
        fts_columns=["desc"],
    )
    b.add_pattern(
        "cust_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[],
    )
    out = b.build()

    reader = GDSReader(out)
    sphere = reader.read_sphere()
    assert sphere.lines["customers"].fts_columns == "all"
    assert sphere.lines["events"].fts_columns == ["desc"]
