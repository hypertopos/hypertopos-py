# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``kind:`` on precomputed dimensions overrides the auto-detected Bregman kind."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hypertopos.builder import GDSBuilder
from hypertopos.cli.build import build_from_config
from hypertopos.cli.schema import parse_config
from hypertopos.storage.reader import GDSReader

_TABLE = pa.table(
    {
        "acct": [f"A{i}" for i in range(8)],
        "n_tx": [1.0, 3.0, 2.0, 7.0, 4.0, 2.0, 5.0, 40.0],
        "ratio": [0.1, 0.5, 0.2, 0.9, 0.3, 0.4, 0.6, 0.7],
    }
)


def _kinds(path: Path) -> list[str]:
    return list(GDSReader(str(path)).read_sphere().patterns["p"].dimension_kinds or [])


def test_builder_kind_overrides_default(tmp_path):
    b = GDSBuilder("k", str(tmp_path / "s"))
    b.add_line("accounts", _TABLE, key_col="acct", source_id="t")
    b.add_pattern("p", "anchor", "accounts", relations=[])
    b.add_precomputed_dimension("accounts", "n_tx", kind="poisson")
    b.add_precomputed_dimension("accounts", "ratio")
    b.build()
    assert _kinds(tmp_path / "s") == ["poisson", "gaussian"]


def test_builder_rejects_unknown_kind(tmp_path):
    b = GDSBuilder("k", str(tmp_path / "s"))
    b.add_line("accounts", _TABLE, key_col="acct", source_id="t")
    with pytest.raises(ValueError, match="kind 'count'"):
        b.add_precomputed_dimension("accounts", "n_tx", kind="count")


def test_yaml_kind_reaches_the_sphere(tmp_path):
    pq.write_table(_TABLE, tmp_path / "accounts.parquet")
    yaml_path = tmp_path / "sphere.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            version: "0.1.0"
            sphere_id: k
            sources:
              accounts:
                path: accounts.parquet
            lines:
              accounts:
                source: accounts
                key: acct
                role: anchor
            patterns:
              p:
                type: anchor
                entity_line: accounts
                precomputed_dimensions:
                  - column: n_tx
                    kind: poisson
                  - column: ratio
                    edge_max: 1
                    kind: gaussian
            """
        ),
        encoding="utf-8",
    )
    cfg = parse_config(yaml_path)
    assert [pc.kind for pc in cfg.patterns["p"].precomputed_dimensions or []] == [
        "poisson",
        "gaussian",
    ]
    build_from_config(cfg, tmp_path / "s", source_dir=tmp_path)
    # Without the override, edge_max: 1 would give bernoulli.
    assert _kinds(tmp_path / "s") == ["poisson", "gaussian"]


def test_yaml_rejects_unknown_kind(tmp_path):
    yaml_path = tmp_path / "sphere.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            version: "0.1.0"
            sphere_id: k
            sources:
              accounts:
                path: accounts.parquet
            lines:
              accounts:
                source: accounts
                key: acct
                role: anchor
            patterns:
              p:
                type: anchor
                entity_line: accounts
                precomputed_dimensions:
                  - column: n_tx
                    kind: count
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kind 'count'"):
        parse_config(yaml_path)
