# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for hybrid ANN+BM25 search."""

from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from hypertopos.navigation.navigator import GDSNavigator


def _make_navigator(storage=None, engine=None):
    """Minimal navigator for unit tests — mirrors pattern from test_navigation.py."""
    nav = object.__new__(GDSNavigator)
    nav._storage = storage or MagicMock()
    nav._engine = engine or MagicMock()
    nav._manifest = MagicMock()
    nav._manifest.line_version.return_value = 1
    nav._contract = MagicMock()
    nav._position = None
    nav._last_total_pre_geometry_filter = None
    return nav


# ── _search_fts_scored ──────────────────────────────────────────────────────


def test_search_fts_scored_returns_key_score_pairs():
    """_search_fts_scored returns (primary_key, bm25_score) tuples."""
    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": ["A", "B"],
            "name": ["Acme", "Beta"],
            "_score": [3.5, 1.2],
            "status": ["active", "active"],
        }
    )

    nav = _make_navigator(storage=storage)

    result = nav._search_fts_scored("entities", "acme", limit=10)

    assert len(result) == 2
    assert result[0][0] == "A"
    assert result[0][1] == pytest.approx(3.5)
    assert result[1][0] == "B"
    assert result[1][1] == pytest.approx(1.2)


def test_search_fts_scored_empty():
    """Returns empty list when no FTS results."""
    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array([], type=pa.string()),
            "_score": pa.array([], type=pa.float64()),
            "status": pa.array([], type=pa.string()),
        }
    )
    nav = _make_navigator(storage=storage)
    result = nav._search_fts_scored("entities", "zzz", limit=10)
    assert result == []


def test_search_fts_scored_preserves_order():
    """Preserves order from storage (already BM25-ranked)."""
    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": ["X", "Y", "Z"],
            "_score": [10.0, 5.0, 2.0],
            "status": ["active"] * 3,
        }
    )
    nav = _make_navigator(storage=storage)
    result = nav._search_fts_scored("entities", "q", limit=10)
    keys = [r[0] for r in result]
    assert keys == ["X", "Y", "Z"]


def test_search_fts_scored_null_score_raises():
    """Raises ValueError when Lance returns null BM25 score."""
    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array(["A"], type=pa.string()),
            "_score": pa.array([None], type=pa.float32()),
            "status": pa.array(["active"]),
        }
    )
    nav = _make_navigator(storage=storage)
    with pytest.raises(ValueError, match="_score"):
        nav._search_fts_scored("entities", "query", limit=10)


def test_search_fts_scored_missing_score_column_raises():
    """Raises ValueError when _score column is absent from FTS result."""
    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array(["A"]),
            "status": pa.array(["active"]),
            # no _score column
        }
    )
    nav = _make_navigator(storage=storage)
    with pytest.raises(ValueError, match="_score"):
        nav._search_fts_scored("entities", "query", limit=10)


# ── search_hybrid ────────────────────────────────────────────────────────────


def test_search_hybrid_fuses_both_scores():
    """search_hybrid ranks by alpha*vector + (1-alpha)*text."""
    # ANN: A=0.0 (closest), B=0.5 distance
    # max_dist=0.5 → vector(A)=1.0, vector(B)=0.0
    # FTS: A=4.0, B=1.0 → max_bm25=4.0 → text(A)=1.0, text(B)=0.25
    # alpha=0.7 → final(A)=0.7+0.3=1.0, final(B)=0.0+0.075=0.075
    import numpy as np

    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array(["A", "B"]),
            "_score": pa.array([4.0, 1.0], type=pa.float32()),
            "status": pa.array(["active", "active"]),
        }
    )
    engine = MagicMock()
    engine.build_polygon.return_value = MagicMock(delta=np.zeros(4))
    engine.find_nearest.return_value = [("A", 0.0), ("B", 0.5)]

    nav = _make_navigator(storage=storage, engine=engine)

    out = nav.search_hybrid(
        primary_key="REF",
        pattern_id="sale_pattern",
        line_id="customers",
        query="acme",
        alpha=0.7,
        top_n=5,
    )
    results = out["results"]

    assert out["ann_active"] is True
    assert len(results) == 2
    assert results[0]["primary_key"] == "A"
    assert results[0]["final_score"] == pytest.approx(1.0)
    assert results[1]["primary_key"] == "B"
    assert results[1]["final_score"] == pytest.approx(0.075, abs=1e-3)
    for r in results:
        assert "vector_score" in r
        assert "text_score" in r
        assert "final_score" in r


def test_search_hybrid_entity_only_in_ann():
    """Entity in ANN but absent from FTS gets text_score=0."""
    import numpy as np

    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array([], type=pa.string()),
            "_score": pa.array([], type=pa.float32()),
            "status": pa.array([], type=pa.string()),
        }
    )
    engine = MagicMock()
    engine.build_polygon.return_value = MagicMock(delta=np.zeros(4))
    engine.find_nearest.return_value = [("A", 0.3)]

    nav = _make_navigator(storage=storage, engine=engine)
    out = nav.search_hybrid("REF", "p", "line", "zzz", alpha=0.7)
    results = out["results"]

    assert out["ann_active"] is True
    assert len(results) == 1
    assert results[0]["primary_key"] == "A"
    assert results[0]["text_score"] == pytest.approx(0.0)
    # Single ANN candidate: max_dist=0.3, so vector_score = 1 - dist/max_dist = 1 - 0.3/0.3 = 0.0.
    # Min-max normalisation with one candidate collapses to 0, not 1 — sole candidate is both
    # min and max, giving (dist - min) / (max - min) = 0/0 which the implementation resolves to 0.
    assert results[0]["vector_score"] == pytest.approx(0.0)


def test_search_hybrid_entity_only_in_fts():
    """Entity in FTS but absent from ANN gets vector_score=0."""
    import numpy as np

    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array(["X"]),
            "_score": pa.array([2.0], type=pa.float32()),
            "status": pa.array(["active"]),
        }
    )
    engine = MagicMock()
    engine.build_polygon.return_value = MagicMock(delta=np.zeros(4))
    engine.find_nearest.return_value = []

    nav = _make_navigator(storage=storage, engine=engine)
    out = nav.search_hybrid("REF", "p", "line", "acme", alpha=0.7)
    results = out["results"]

    assert out["ann_active"] is False
    assert len(results) == 1
    assert results[0]["primary_key"] == "X"
    assert results[0]["vector_score"] == pytest.approx(0.0)
    assert results[0]["text_score"] == pytest.approx(1.0)
    assert results[0]["final_score"] == pytest.approx(0.3, abs=1e-3)


def test_search_hybrid_top_n_limits_results():
    """Returns at most top_n results."""
    import numpy as np

    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array(["A", "B", "C"]),
            "_score": pa.array([3.0, 2.0, 1.0], type=pa.float32()),
            "status": pa.array(["active"] * 3),
        }
    )
    engine = MagicMock()
    engine.build_polygon.return_value = MagicMock(delta=np.zeros(4))
    engine.find_nearest.return_value = [("A", 0.1), ("B", 0.2), ("C", 0.3)]

    nav = _make_navigator(storage=storage, engine=engine)
    out = nav.search_hybrid("REF", "p", "line", "q", top_n=2)
    assert len(out["results"]) == 2


def test_search_hybrid_excludes_reference_entity():
    """primary_key is never returned in results."""
    import numpy as np

    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array(["REF", "A"]),
            "_score": pa.array([5.0, 1.0], type=pa.float32()),
            "status": pa.array(["active", "active"]),
        }
    )
    engine = MagicMock()
    engine.build_polygon.return_value = MagicMock(delta=np.zeros(4))
    engine.find_nearest.return_value = [("A", 0.1)]  # find_nearest already excludes REF

    nav = _make_navigator(storage=storage, engine=engine)
    out = nav.search_hybrid("REF", "p", "line", "q")

    pks = [r["primary_key"] for r in out["results"]]
    assert "REF" not in pks


def test_search_hybrid_no_candidates_returns_empty():
    """Returns [] when both ANN and FTS find nothing."""
    import numpy as np

    storage = MagicMock()
    storage.search_points_fts.return_value = pa.table(
        {
            "primary_key": pa.array([], type=pa.string()),
            "_score": pa.array([], type=pa.float32()),
            "status": pa.array([], type=pa.string()),
        }
    )
    engine = MagicMock()
    engine.build_polygon.return_value = MagicMock(delta=np.zeros(4))
    engine.find_nearest.return_value = []

    nav = _make_navigator(storage=storage, engine=engine)
    out = nav.search_hybrid("REF", "p", "line", "q")
    assert out["results"] == []
    assert out["ann_active"] is False
