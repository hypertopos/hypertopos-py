# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Lance write policy: every dataset hypertopos writes uses one pinned file format.

The pylance "stable" file-format alias moves between releases, so a
``write_dataset`` call that bypasses ``_write_lance`` silently picks up
whatever the installed pylance defaults to. These tests guard both the rule
(no bypassing call sites) and its on-disk outcome (a freshly built sphere and
its sidecars all report the pinned format).
"""

from __future__ import annotations

import ast
from pathlib import Path

import lance
import pyarrow as pa

import hypertopos
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.builder.builder import EdgeTableConfig
from hypertopos.builder.conformance import write_conformance_sidecar
from hypertopos.builder.mapping import EdgeDimensionsConfig
from hypertopos.storage.topology_cache import ANOMALIES_SCHEMA, cache_path, write_cache
from hypertopos.storage.writer import _LANCE_WRITE_DEFAULTS

_PACKAGE_ROOT = Path(hypertopos.__file__).resolve().parent
_WRITER = _PACKAGE_ROOT / "storage" / "writer.py"


def _write_dataset_calls_outside_write_lance(path: Path) -> list[int]:
    """Line numbers of ``write_dataset`` calls in ``path`` not inside ``_write_lance``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed: set[int] = set()
    if path == _WRITER:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_write_lance":
                allowed.update(id(n) for n in ast.walk(node))
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in allowed:
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name == "write_dataset":
            lines.append(node.lineno)
    return lines


def test_all_lance_writes_go_through_write_lance() -> None:
    sources = sorted(_PACKAGE_ROOT.rglob("*.py"))
    assert _WRITER in sources, f"package sources not found under {_PACKAGE_ROOT}"
    offenders = [
        f"{f.relative_to(_PACKAGE_ROOT)}:{line}"
        for f in sources
        for line in _write_dataset_calls_outside_write_lance(f)
    ]
    assert offenders == [], f"write_dataset calls bypassing _write_lance: {offenders}"


def _build_event_sphere(out_root: Path) -> None:
    b = GDSBuilder("test_wp", str(out_root))
    b.add_line(
        "accounts",
        [{"acct_id": k, "name": k.lower()} for k in ("A", "B", "C", "D")],
        key_col="acct_id",
        source_id="t",
    )
    b.add_line(
        "transactions",
        [
            {"tx_id": "ek1", "from_acct": "A", "to_acct": "B", "ts": 0.0, "amount": 20000.0},
            {"tx_id": "ek2", "from_acct": "B", "to_acct": "C", "ts": 100.0, "amount": 5000.0},
            {"tx_id": "ek3", "from_acct": "C", "to_acct": "D", "ts": 200.0, "amount": 5000.0},
            {"tx_id": "ek4", "from_acct": "D", "to_acct": "A", "ts": 8000.0, "amount": 3000.0},
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
            from_col="from_acct",
            to_col="to_acct",
            timestamp_col="ts",
            amount_col="amount",
        ),
        edge_dimensions=EdgeDimensionsConfig(dims={"pair_edge_count": {}}),
    )
    b.build()


def test_built_sphere_datasets_use_pinned_storage_format(tmp_path: Path) -> None:
    out_root = tmp_path / "gds_wp"
    _build_event_sphere(out_root)
    write_conformance_sidecar(
        base_path=out_root,
        pattern_id="tx_pattern",
        version=1,
        violations_table=pa.table(
            {
                "primary_key": pa.array(["ek1"], pa.large_string()),
                "rule_id": pa.array(["r1"], pa.large_string()),
                "severity": pa.array(["high"], pa.large_string()),
            }
        ),
        rule_set_hash="h",
        n_rules=1,
    )
    write_cache(
        cache_path(out_root, "anomalies", "tx_pattern", 1),
        [
            {
                "primary_key": "ek1",
                "topo_score": 1.0,
                "h1_max_persistence": 0.5,
                "h0_mean_death": 0.1,
                "n_h1_features": 1,
                "computed_at": None,
            }
        ],
        ANOMALIES_SCHEMA,
    )

    versions = {
        p.parent.relative_to(out_root).as_posix(): lance.dataset(str(p.parent)).data_storage_version
        for p in out_root.rglob("_versions")
    }
    # The sidecars that used to bypass the pin must be among the checked datasets.
    assert any(r.startswith("_gds_meta/edge_features/") for r in versions), versions
    assert any(r.startswith("_gds_meta/topology_cache/") for r in versions), versions
    assert any("conformance" in r for r in versions), versions

    expected = _LANCE_WRITE_DEFAULTS["data_storage_version"]
    assert all(v == expected for v in versions.values()), versions
