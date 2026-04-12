# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for Arrow-native contagion stats computation."""
from __future__ import annotations

import pyarrow as pa
import pytest

from hypertopos.builder.builder import GDSBuilder


class TestContagionStatsArrow:
    def test_basic_counts(self):
        edges = pa.table({
            "from_key": ["A", "A", "B", "C"],
            "to_key":   ["B", "C", "C", "A"],
        })
        geom = pa.table({
            "primary_key": ["A", "B", "C"],
            "is_anomaly":  [True, False, True],
        })

        result = GDSBuilder._compute_contagion_arrow(edges, geom)
        pk = result["primary_key"].to_pylist()
        nc = result["neighbor_count"].to_pylist()
        ac = result["anomalous_neighbor_count"].to_pylist()
        cr = result["contagion_ratio"].to_pylist()
        idx = {k: i for i, k in enumerate(pk)}

        assert nc[idx["A"]] == 2
        assert nc[idx["B"]] == 2
        assert nc[idx["C"]] == 2

        assert ac[idx["A"]] == 1  # neighbors: B(no), C(yes)
        assert ac[idx["B"]] == 2  # neighbors: A(yes), C(yes)
        assert ac[idx["C"]] == 1  # neighbors: A(yes), B(no)

        assert cr[idx["A"]] == pytest.approx(0.5)
        assert cr[idx["B"]] == pytest.approx(1.0)
        assert cr[idx["C"]] == pytest.approx(0.5)

    def test_self_loops_excluded(self):
        edges = pa.table({
            "from_key": ["A", "A"],
            "to_key":   ["A", "B"],
        })
        geom = pa.table({
            "primary_key": ["A", "B"],
            "is_anomaly":  [False, False],
        })

        result = GDSBuilder._compute_contagion_arrow(edges, geom)
        pk = result["primary_key"].to_pylist()
        nc = result["neighbor_count"].to_pylist()
        idx = {k: i for i, k in enumerate(pk)}
        assert nc[idx["A"]] == 1
        assert nc[idx["B"]] == 1

    def test_empty_edges(self):
        edges = pa.table({
            "from_key": pa.array([], type=pa.string()),
            "to_key": pa.array([], type=pa.string()),
        })
        geom = pa.table({
            "primary_key": ["A"],
            "is_anomaly": [False],
        })

        result = GDSBuilder._compute_contagion_arrow(edges, geom)
        assert result.num_rows == 0

    def test_no_anomalous(self):
        edges = pa.table({
            "from_key": ["A", "B"],
            "to_key":   ["B", "A"],
        })
        geom = pa.table({
            "primary_key": ["A", "B"],
            "is_anomaly":  [False, False],
        })

        result = GDSBuilder._compute_contagion_arrow(edges, geom)
        ac = result["anomalous_neighbor_count"].to_pylist()
        assert all(c == 0 for c in ac)

    def test_all_self_loops(self):
        edges = pa.table({
            "from_key": ["A", "B"],
            "to_key":   ["A", "B"],
        })
        geom = pa.table({
            "primary_key": ["A", "B"],
            "is_anomaly":  [True, True],
        })

        result = GDSBuilder._compute_contagion_arrow(edges, geom)
        assert result.num_rows == 0
