from hypertopos.engine.adjacency import AdjacencyIndex


def test_from_edge_lists_neighbors_out():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "A", "B"],
        to_keys=["B", "C", "C"],
        timestamps=[1.0, 2.0, 3.0],
        amounts=[100.0, 200.0, 300.0],
        event_keys=["e1", "e2", "e3"],
    )
    out = idx.neighbors_out("A")
    assert len(out) == 2
    assert out[0] == ("B", 1.0, 100.0, "e1")
    assert out[1] == ("C", 2.0, 200.0, "e2")


def test_from_edge_lists_neighbors_in():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "A", "B"],
        to_keys=["B", "C", "C"],
        timestamps=[1.0, 2.0, 3.0],
        amounts=[100.0, 200.0, 300.0],
        event_keys=["e1", "e2", "e3"],
    )
    in_edges = idx.neighbors_in("C")
    assert len(in_edges) == 2
    assert in_edges[0] == ("A", 2.0, 200.0, "e2")
    assert in_edges[1] == ("B", 3.0, 300.0, "e3")


def test_neighbors_sorted_by_timestamp():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "A", "A"],
        to_keys=["B", "C", "D"],
        timestamps=[3.0, 1.0, 2.0],
        amounts=[100.0, 200.0, 300.0],
        event_keys=["e1", "e2", "e3"],
    )
    out = idx.neighbors_out("A")
    assert [e[1] for e in out] == [1.0, 2.0, 3.0]


def test_neighbors_unknown_key_returns_empty():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A"], to_keys=["B"],
        timestamps=[1.0], amounts=[10.0], event_keys=["e1"],
    )
    assert idx.neighbors_out("Z") == []
    assert idx.neighbors_in("Z") == []


def test_pair_counts_matches_defaultdict_baseline():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "A", "B", "A", "C", "C"],
        to_keys=["B", "B", "C", "C", "A", "C"],
        timestamps=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        amounts=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        event_keys=["e1", "e2", "e3", "e4", "e5", "e6"],
    )
    assert idx.pair_counts() == {
        ("A", "B"): 2,
        ("B", "C"): 1,
        ("A", "C"): 1,
        ("C", "A"): 1,
        ("C", "C"): 1,
    }


def test_pair_counts_lazy_and_cached():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "A"], to_keys=["B", "C"],
        timestamps=[1.0, 2.0], amounts=[10.0, 20.0],
        event_keys=["e1", "e2"],
    )
    assert idx._pair_counts is None
    first = idx.pair_counts()
    assert idx._pair_counts is first
    second = idx.pair_counts()
    assert second is first


def test_pair_counts_empty_adjacency():
    idx = AdjacencyIndex(_out={}, _in={}, _nodes=set(), _edge_count=0)
    assert idx.pair_counts() == {}


def test_neighbors_out_temporal_filter():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "A", "A", "A"],
        to_keys=["B", "C", "D", "E"],
        timestamps=[1.0, 2.0, 3.0, 4.0],
        amounts=[10.0, 20.0, 30.0, 40.0],
        event_keys=["e1", "e2", "e3", "e4"],
    )
    result = idx.neighbors_out("A", ts_from=2.5)
    assert len(result) == 2
    assert result[0][0] == "D"

    result = idx.neighbors_out("A", ts_to=2.5)
    assert len(result) == 2
    assert result[-1][0] == "C"

    result = idx.neighbors_out("A", ts_from=1.5, ts_to=3.5)
    assert len(result) == 2
    assert [e[0] for e in result] == ["C", "D"]


def test_neighbors_all_combines_directions():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "B"],
        to_keys=["B", "A"],
        timestamps=[1.0, 2.0],
        amounts=[10.0, 20.0],
        event_keys=["e1", "e2"],
    )
    all_a = idx.neighbors_all("A")
    assert len(all_a) == 2


def test_graph_wide_iteration():
    idx = AdjacencyIndex.from_edge_lists(
        from_keys=["A", "B", "C"],
        to_keys=["B", "C", "A"],
        timestamps=[1.0, 2.0, 3.0],
        amounts=[10.0, 20.0, 30.0],
        event_keys=["e1", "e2", "e3"],
    )
    assert idx.node_count() == 3
    assert idx.edge_count() == 3
    assert idx.all_nodes() == {"A", "B", "C"}
    edges = list(idx.all_edges())
    assert len(edges) == 3


def test_reader_get_adjacency_caches():
    """get_adjacency returns same object on second call (cached)."""
    import json
    import os
    from hypertopos.storage.reader import GDSReader

    sphere_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "benchmark", "berka", "sphere", "gds_berka_banking",
    )
    if not os.path.exists(sphere_path):
        import pytest
        pytest.skip("Berka sphere not available")

    reader = GDSReader(sphere_path)
    sphere_data = json.loads(
        (reader._base / "_gds_meta" / "sphere.json").read_text(),
    )
    edge_patterns = [
        pid for pid, p in sphere_data["patterns"].items()
        if p.get("has_edge_table")
    ]
    if not edge_patterns:
        import pytest
        pytest.skip("No edge table in Berka sphere")

    pat_id = edge_patterns[0]
    idx1 = reader.get_adjacency(pat_id)
    idx2 = reader.get_adjacency(pat_id)
    assert idx1 is idx2
    assert idx1.node_count() > 0
    assert idx1.edge_count() > 0


def test_adjacency_integration_session(tmp_path):
    """Build sphere with edges, open session, verify adjacency caching."""
    from datetime import datetime
    import json
    import pyarrow as pa
    from hypertopos.builder.builder import GDSBuilder, RelationSpec
    from hypertopos.sphere import HyperSphere

    rows = []
    for i in range(30):
        rows.append({
            "primary_key": f"TX-{i}",
            "from_account": ["A", "B", "C"][i % 3],
            "to_account": ["A", "B", "C"][(i + 1) % 3],
            "amount": (i + 1) * 10.0,
            "timestamp": datetime(2024, 1, 1 + i // 10),
        })
    events = pa.table({k: [r[k] for r in rows] for k in rows[0]})
    accounts = pa.table({"primary_key": ["A", "B", "C"]})

    path = tmp_path / "sphere"
    b = GDSBuilder("test", str(path))
    b.add_line("tx", events, key_col="primary_key", source_id="t", role="event")
    b.add_line("accts", accounts, key_col="primary_key", source_id="t", role="anchor")
    b.add_derived_dimension("accts", "tx", "from_account", "count", None, "tx_count")
    b.add_pattern("acct_pat", "anchor", "accts", relations=[])
    b.add_pattern("tx_pat", "event", "tx", relations=[
        RelationSpec(line_id="accts", fk_col="from_account", direction="out"),
        RelationSpec(line_id="accts", fk_col="to_account", direction="out"),
    ])
    b.build()

    hs = HyperSphere.open(str(path))
    sess = hs.session("test")

    sj = json.loads((path / "_gds_meta" / "sphere.json").read_text())
    edge_pats = [p for p, d in sj["patterns"].items() if d.get("has_edge_table")]
    if not edge_pats:
        return

    pat_id = edge_pats[0]
    adj = sess._reader.get_adjacency(pat_id)
    assert adj.node_count() > 0
    assert adj.edge_count() > 0
    # Cached — same object
    assert sess._reader.get_adjacency(pat_id) is adj
