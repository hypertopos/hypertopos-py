"""Regression tests for builder / CLI build defects: chain-line source
resolution and null endpoints, chain-line column selection, YAML key
validation, streaming geometry with edge-dim aggregations, and the
incremental n_anomalous_dims threshold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from hypertopos.cli.build import _add_chain_line
from hypertopos.cli.schema import (
    ChainLineConfig,
    EdgeTableYamlConfig,
    LineConfig,
    PatternConfig,
    SourceConfig,
    SphereConfig,
)


class _RecordingBuilder:
    """Captures the chains handed to add_chain_line."""

    def __init__(self) -> None:
        self.chains: list[dict[str, Any]] = []

    def add_chain_line(self, line_id: str, chains: list[dict[str, Any]], **_: Any) -> None:
        self.chains = chains

    def add_pattern(self, *_: Any, **__: Any) -> None:
        pass


def _events(rows: list[tuple[Any, Any, str, float]], **extra: list[Any]) -> pa.Table:
    data: dict[str, list[Any]] = {
        "primary_key": [f"E{i}" for i in range(len(rows))],
        "from_acct": [r[0] for r in rows],
        "to_acct": [r[1] for r in rows],
        "timestamp": [r[2] for r in rows],
        "amount": [r[3] for r in rows],
    }
    data.update(extra)
    return pa.table(data)


_CHAIN_ROWS: list[tuple[Any, Any, str, float]] = [
    ("A", "B", "2024-01-01T00:00:00", 100.0),
    ("B", "C", "2024-01-01T01:00:00", 90.0),
    ("C", "D", "2024-01-01T02:00:00", 80.0),
    ("A", "C", "2024-01-01T03:00:00", 70.0),
    ("A", "D", "2024-01-01T04:00:00", 60.0),
]


def _cfg(
    tables: dict[str, pa.Table],
    event_line_source: str,
    chain: ChainLineConfig,
    edge_table: EdgeTableYamlConfig | None = None,
) -> SphereConfig:
    lines = {
        "accounts": LineConfig(source=event_line_source, key="from_acct"),
        "transfers": LineConfig(source=event_line_source, key="primary_key", role="event"),
    }
    patterns: dict[str, PatternConfig] = {}
    if edge_table is not None:
        patterns["tx_pattern"] = PatternConfig(
            type="event", entity_line="transfers", edge_table=edge_table
        )
    return SphereConfig(
        sphere_id="t",
        sources={name: SourceConfig(path=f"{name}.csv") for name in tables},
        lines=lines,
        patterns=patterns,
        chain_lines={"chains": chain},
    )


def _chain_cfg(**overrides: Any) -> ChainLineConfig:
    kwargs: dict[str, Any] = {
        "event_line": "transfers",
        "from_col": "from_acct",
        "to_col": "to_acct",
        "seed_percentile_fan_out": 0.0,
        "bidirectional": False,
        "min_hops": 2,
    }
    kwargs.update(overrides)
    return ChainLineConfig(**kwargs)


def _run_chain_line(cfg: SphereConfig, tables: dict[str, pa.Table]) -> list[dict[str, Any]]:
    builder = _RecordingBuilder()
    _add_chain_line(builder, "chains", cfg.chain_lines["chains"], cfg, tables, False)  # type: ignore[arg-type]
    return builder.chains


# ── chain lines: null endpoints ──────────────────────────────────────


def test_chain_line_with_null_sender_builds_without_none_nodes() -> None:
    rows = [*_CHAIN_ROWS, (None, "B", "2024-01-01T05:00:00", 50.0)]
    tables = {"tx": _events(rows)}
    cfg = _cfg(tables, "tx", _chain_cfg())
    chains = _run_chain_line(cfg, tables)
    assert chains
    for c in chains:
        assert None not in c["keys"]
        assert "None" not in c["keys"]


# ── chain lines: event_line selects the source ───────────────────────


def test_chain_line_reads_the_event_line_source_not_the_first_match() -> None:
    decoy = _events(
        [("X", "Y", "2024-01-01T00:00:00", 1.0), ("Y", "Z", "2024-01-01T01:00:00", 1.0)]
    )
    real = _events(_CHAIN_ROWS)
    tables = {"cards": decoy, "tx": real}  # decoy declared first, same column names
    cfg = _cfg(tables, "tx", _chain_cfg())
    chains = _run_chain_line(cfg, tables)
    keys = {k for c in chains for k in c["keys"]}
    assert keys
    assert keys <= {"A", "B", "C", "D"}


# ── chain lines: amount / time / category columns ────────────────────


def test_chain_line_uses_edge_table_amount_and_timestamp_columns() -> None:
    rows = _CHAIN_ROWS
    table = pa.table(
        {
            "primary_key": [f"E{i}" for i in range(len(rows))],
            "from_acct": [r[0] for r in rows],
            "to_acct": [r[1] for r in rows],
            "booked_at": [r[2] for r in rows],
            "value": [r[3] for r in rows],
        }
    )
    tables = {"tx": table}
    cfg = _cfg(
        tables,
        "tx",
        _chain_cfg(),
        edge_table=EdgeTableYamlConfig(
            from_col="from_acct", to_col="to_acct", timestamp_col="booked_at", amount_col="value"
        ),
    )
    chains = _run_chain_line(cfg, tables)
    abc = next(c for c in chains if c["keys"][:3] == ["A", "B", "C"])
    assert abc["total_amount"] > 0
    assert abc["time_span_hours"] > 0


def test_chain_line_explicit_columns_override_edge_table() -> None:
    table = _events(_CHAIN_ROWS, channel=["web", "atm", "web", "atm", "web"])
    table = table.append_column("fee", pa.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    tables = {"tx": table}
    cfg = _cfg(
        tables,
        "tx",
        _chain_cfg(amount_col="fee", category_col="channel"),
        edge_table=EdgeTableYamlConfig(from_col="from_acct", to_col="to_acct", amount_col="amount"),
    )
    chains = _run_chain_line(cfg, tables)
    abcd = next(c for c in chains if c["keys"] == ["A", "B", "C", "D"])
    assert abcd["total_amount"] == pytest.approx(6.0)  # fee 1 + 2 + 3, not amount
    assert abcd["categories"] == ["atm", "web"]


def test_chain_line_unknown_explicit_column_raises() -> None:
    tables = {"tx": _events(_CHAIN_ROWS)}
    cfg = _cfg(tables, "tx", _chain_cfg(amount_col="nope"))
    with pytest.raises(ValueError, match="nope"):
        _run_chain_line(cfg, tables)


# ── streaming geometry build with edge_dim_aggregations ──────────────


def _eda_sphere(out: Path) -> Any:
    from hypertopos.builder import GDSBuilder, RelationSpec
    from hypertopos.builder.builder import EdgeTableConfig
    from hypertopos.builder.mapping import EdgeDimAggregationsConfig, EdgeDimensionsConfig

    b = GDSBuilder("eda_stream", str(out))
    accts = [chr(ord("A") + i) for i in range(10)]
    b.add_line(
        "accounts",
        [{"acct_id": a, "segment": "retail" if i % 3 else None} for i, a in enumerate(accts)],
        key_col="acct_id",
        source_id="t",
    )
    pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "B"), ("C", "D"), ("D", "A")]
    b.add_line(
        "transactions",
        [
            {"tx_id": f"ek{i}", "from_acct": f, "to_acct": t, "ts": 100.0 * i, "amount": 10.0 * i}
            for i, (f, t) in enumerate(pairs)
        ],
        key_col="tx_id",
        source_id="t",
    )
    b.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="transactions",
        relations=[RelationSpec("accounts", fk_col="from_acct", direction="in", required=True)],
        edge_table=EdgeTableConfig(
            from_col="from_acct", to_col="to_acct", timestamp_col="ts", amount_col="amount"
        ),
        edge_dimensions=EdgeDimensionsConfig(dims={"pair_edge_count": {}}),
    )
    b.add_pattern(
        "account_pattern",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[],
        edge_dim_aggregations=EdgeDimAggregationsConfig(
            from_event_pattern="tx_pattern", dims=("pair_edge_count",)
        ),
    )
    b.add_derived_dimension(
        anchor_line="accounts",
        event_line="transactions",
        anchor_fk="from_acct",
        metric="count",
        metric_col=None,
        dimension_name="tx_out_count",
        edge_max="auto",
    )
    return b


def _pattern_node(out: Path, pid: str) -> dict[str, Any]:
    import json

    return json.loads((out / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"))["patterns"][
        pid
    ]


def test_streaming_build_with_edge_dim_aggregations_matches_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    import hypertopos.builder.builder as builder_mod

    in_memory = tmp_path / "in_memory"
    _eda_sphere(in_memory).build()

    # 10 accounts > chunk of 8 >= 6 transactions: only the anchor pattern streams.
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 8)
    streamed = tmp_path / "streamed"
    _eda_sphere(streamed).build()

    ref = _pattern_node(in_memory, "account_pattern")
    got = _pattern_node(streamed, "account_pattern")
    assert got["prop_columns"] == ref["prop_columns"]
    assert len(got["mu"]) == len(ref["mu"])
    np.testing.assert_allclose(got["mu"], ref["mu"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got["sigma_diag"], ref["sigma_diag"], rtol=1e-5, atol=1e-6)
    assert got["dimension_kinds"] == ref["dimension_kinds"]


# ── incremental update: n_anomalous_dims thresholds ──────────────────


def _prop_sphere(tmp_path: Path) -> str:
    """100 customers, a count dim and a property filled for 95% of them."""
    import numpy as np

    from hypertopos.builder.builder import GDSBuilder

    n = 100
    rng = np.random.default_rng(7)
    custs = pa.table(
        {
            "primary_key": [f"C{i}" for i in range(n)],
            "region": [None if i % 20 == 0 else "A" for i in range(n)],
        }
    )
    events = pa.table(
        {
            "primary_key": [f"E{i}" for i in range(n * 5)],
            "cust_fk": [f"C{rng.integers(0, n)}" for _ in range(n * 5)],
        }
    )
    out = str(tmp_path / "gds_inc")
    b = GDSBuilder("inc", out)
    b.add_line("customers", custs, key_col="primary_key", source_id="t", role="anchor")
    b.add_line("events", events, key_col="primary_key", source_id="t", role="event")
    b.add_derived_dimension("customers", "events", "cust_fk", "count", None, "event_count")
    b.add_pattern(
        "cust_pattern", "anchor", "customers", relations=[], tracked_properties=["region"]
    )
    b.build()
    return out


def test_incremental_n_anomalous_dims_uses_build_thresholds(tmp_path: Path) -> None:
    """A property dim filled for 95% of entities puts every unfilled entity at
    |delta| ~ 4.4. That is the population P99, so the full build counts it as
    not anomalous; a fixed 2.576 cut would count it. Re-ingesting the same rows
    must reproduce the build's counts."""
    from hypertopos.builder.builder import GDSBuilder
    from hypertopos.storage.reader import GDSReader

    n = 100
    out = _prop_sphere(tmp_path)

    node = _pattern_node(Path(out), "cust_pattern")
    assert len(node["n_anomalous_dims_thresholds"]) == len(node["mu"])

    geo = GDSReader(out).read_geometry("cust_pattern", version=1).to_pydict()
    built = dict(zip(geo["primary_key"], geo["n_anomalous_dims"], strict=True))
    points = GDSReader(out).read_points("customers", version=1)
    unfilled = [f"C{i}" for i in range(0, n, 20)]
    replay = points.filter(pa.compute.is_in(points["primary_key"], pa.array(unfilled)))

    GDSBuilder("inc", out).incremental_update(
        "cust_pattern", changed_entities=replay, recalibrate="never"
    )
    reader = GDSReader(out)
    geo2 = reader.read_geometry(
        "cust_pattern", version=reader.read_sphere().patterns["cust_pattern"].version
    ).to_pydict()
    after = dict(zip(geo2["primary_key"], geo2["n_anomalous_dims"], strict=True))
    assert [after[k] for k in unfilled] == [built[k] for k in unfilled]


def test_chain_line_edge_table_columns_follow_line_rename_and_epoch_seconds() -> None:
    rows = _CHAIN_ROWS
    table = pa.table(
        {
            "primary_key": [f"E{i}" for i in range(len(rows))],
            "from_acct": [r[0] for r in rows],
            "to_acct": [r[1] for r in rows],
            "src_ts": [3600.0 * i for i in range(len(rows))],
            "src_value": [r[3] for r in rows],
        }
    )
    tables = {"tx": table}
    cfg = _cfg(
        tables,
        "tx",
        _chain_cfg(),
        edge_table=EdgeTableYamlConfig(
            from_col="from_acct", to_col="to_acct", timestamp_col="ts", amount_col="value"
        ),
    )
    cfg.lines["transfers"].columns = {
        "primary_key": "primary_key",
        "from_acct": "from_acct",
        "to_acct": "to_acct",
        "ts": "src_ts",
        "value": "src_value",
    }
    chains = _run_chain_line(cfg, tables)
    abcd = next(c for c in chains if c["keys"] == ["A", "B", "C", "D"])
    assert abcd["total_amount"] == pytest.approx(270.0)
    assert abcd["time_span_hours"] == pytest.approx(2.0)


# ── YAML: unknown keys ───────────────────────────────────────────────


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _minimal_yaml() -> dict[str, Any]:
    return {
        "version": "0.1.0",
        "sphere_id": "t",
        "sources": {"tx": {"path": "tx.csv"}},
        "lines": {"tx": {"source": "tx", "key": "primary_key", "role": "event"}},
        "patterns": {
            "tx_pattern": {
                "type": "event",
                "entity_line": "tx",
                "relations": [],
                "edge_table": {"from_col": "a", "to_col": "b", "timestamp_col": "ts"},
            }
        },
    }


def test_misspelled_yaml_key_warns_with_suggestion(tmp_path: Path) -> None:
    from hypertopos.cli.schema import parse_config

    spec = _minimal_yaml()
    del spec["patterns"]["tx_pattern"]["edge_table"]["timestamp_col"]
    spec["patterns"]["tx_pattern"]["edge_table"]["tiemstamp_col"] = "ts"
    with pytest.warns(UserWarning, match=r"edge_table.*'tiemstamp_col'.*'timestamp_col'"):
        parse_config(_write_yaml(tmp_path / "sphere.yaml", spec))


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "sphere_idd"),
        (("sources", "tx"), "delimeter"),
        (("lines", "tx"), "rol"),
        (("patterns", "tx_pattern"), "anomaly_percentil"),
    ],
)
def test_unknown_key_in_any_section_warns(tmp_path: Path, path: tuple[str, ...], key: str) -> None:
    from hypertopos.cli.schema import parse_config

    spec = _minimal_yaml()
    node: dict[str, Any] = spec
    for p in path:
        node = node[p]
    node[key] = 1
    with pytest.warns(UserWarning, match=key):
        parse_config(_write_yaml(tmp_path / "sphere.yaml", spec))


def test_known_keys_do_not_warn(tmp_path: Path) -> None:
    import warnings

    from hypertopos.cli.schema import parse_config

    spec = _minimal_yaml()
    spec["sources"]["tx"]["join"] = [{"file": "x.csv", "on": "id"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parse_config(_write_yaml(tmp_path / "sphere.yaml", spec))


# ── YAML / programmatic build surface ────────────────────────────────


def test_yaml_bootstrap_iterations_reaches_builder(tmp_path: Path) -> None:
    from hypertopos.builder.builder import GDSBuilder
    from hypertopos.cli.build import _add_pattern
    from hypertopos.cli.schema import parse_config

    spec = _minimal_yaml()
    spec["patterns"]["tx_pattern"]["bootstrap_iterations"] = 0
    cfg = parse_config(_write_yaml(tmp_path / "sphere.yaml", spec))
    builder = GDSBuilder("t", str(tmp_path / "out"))
    _add_pattern(builder, "tx_pattern", cfg.patterns["tx_pattern"], cfg, set())
    assert builder._patterns["tx_pattern"].bootstrap_iterations == 0


def test_build_from_config_passes_no_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hypertopos.cli.build as build_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(build_mod, "_do_build", lambda *a, **kw: seen.update(kw))
    build_mod.build_from_config(SphereConfig(sphere_id="t"), tmp_path / "out", no_edges=True)
    assert seen["no_edges"] is True


def test_recalibrate_refreshes_n_anomalous_dims_thresholds(tmp_path: Path) -> None:
    import numpy as np

    from hypertopos.builder._stats import (
        compute_per_dim_anomaly_count,
        per_dim_anomaly_thresholds,
    )
    from hypertopos.sphere import HyperSphere
    from hypertopos.storage.reader import GDSReader
    from hypertopos.utils.arrow import delta_matrix_from_arrow

    out = _prop_sphere(tmp_path)
    before = _pattern_node(Path(out), "cust_pattern")["n_anomalous_dims_thresholds"]
    excluded = [f"C{i}" for i in range(0, 100, 20)]  # every unfilled entity
    HyperSphere.open(out).session("t").recalibrate("cust_pattern", exclude_keys=excluded)

    after = np.asarray(_pattern_node(Path(out), "cust_pattern")["n_anomalous_dims_thresholds"])
    assert not np.allclose(after, before)
    reader = GDSReader(out)
    geo = reader.read_geometry(
        "cust_pattern",
        version=reader.read_sphere().patterns["cust_pattern"].version,
        columns=["primary_key", "delta", "n_anomalous_dims"],
    )
    deltas = delta_matrix_from_arrow(geo)
    fit = ~np.isin(np.asarray(geo["primary_key"].to_pylist(), dtype=object), excluded)
    np.testing.assert_allclose(after, per_dim_anomaly_thresholds(deltas[fit]), rtol=1e-6)
    np.testing.assert_array_equal(
        geo["n_anomalous_dims"].to_numpy(),
        compute_per_dim_anomaly_count(deltas, thresholds=after),
    )
