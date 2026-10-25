# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import io
import subprocess
import sys
import textwrap
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path

import pytest

from hypertopos.cli.main import main as cli_main


@dataclass
class _DirectResult:
    """Mimics subprocess.CompletedProcess for direct CLI invocation."""

    returncode: int
    stderr: str


def _run_cli_direct(args: list[str], monkeypatch) -> _DirectResult:
    """Run CLI entry point in-process; captures SystemExit and stderr."""
    monkeypatch.setattr(sys, "argv", ["hypertopos"] + args)
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            cli_main()
        return _DirectResult(returncode=0, stderr=buf.getvalue())
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return _DirectResult(returncode=code, stderr=buf.getvalue())


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the CLI via python -m hypertopos.cli (for build tests)."""
    return subprocess.run(
        [sys.executable, "-m", "hypertopos.cli"] + args,
        capture_output=True,
        text=True,
    )


def test_cli_no_args_exits_nonzero(monkeypatch):
    result = _run_cli_direct([], monkeypatch)
    assert result.returncode != 0


def test_cli_build_missing_mapping(monkeypatch):
    result = _run_cli_direct(["build"], monkeypatch)
    assert result.returncode != 0


@pytest.fixture(scope="module")
def default_build(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """One real subprocess ``build`` (YAML ``output_path``, no override).

    Read-only for its consumers: they only inspect the exit code, stdout and
    the written sphere, so a single build serves the whole module.
    """
    workdir = tmp_path_factory.mktemp("cli_default_build")
    (workdir / "items.csv").write_text("item_id,name\nI-1,Widget\nI-2,Gadget\nI-3,Foo\n")
    yaml_file = workdir / "mapping.yaml"
    yaml_file.write_text(
        textwrap.dedent(f"""\
        sphere_id: cli_test
        output_path: {workdir / "gds_cli"}
        lines:
          items:
            source: items.csv
            key_col: item_id
        patterns: {{}}
    """)
    )
    return _run_cli(["build", "--mapping", str(yaml_file)]), workdir


def test_cli_build_creates_sphere(default_build):
    result, workdir = default_build
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (workdir / "gds_cli" / "_gds_meta" / "sphere.json").exists()


def test_cli_build_output_override(tmp_path):
    (tmp_path / "items.csv").write_text("item_id\nI-1\nI-2\n")
    yaml_file = tmp_path / "mapping.yaml"
    yaml_file.write_text(
        textwrap.dedent(f"""\
        sphere_id: override_test
        output_path: {tmp_path / "default"}
        lines:
          items:
            source: items.csv
            key_col: item_id
        patterns: {{}}
    """)
    )
    custom_out = tmp_path / "custom_out"
    result = _run_cli(["build", "--mapping", str(yaml_file), "--output", str(custom_out)])
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (custom_out / "_gds_meta" / "sphere.json").exists()
    assert not (tmp_path / "default").exists()


def test_cli_build_nonexistent_mapping(monkeypatch):
    """CLI exits 1 and prints 'error:' on missing mapping file."""
    result = _run_cli_direct(["build", "--mapping", "/nonexistent/mapping.yaml"], monkeypatch)
    assert result.returncode == 1
    assert "error:" in result.stderr.lower()


def test_cli_build_invalid_yaml(tmp_path, monkeypatch):
    """CLI exits 1 and prints 'error:' on invalid YAML content."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("- this\n- is\n- a list\n")
    result = _run_cli_direct(["build", "--mapping", str(yaml_file)], monkeypatch)
    assert result.returncode == 1
    assert "error:" in result.stderr.lower()


def test_cli_build_prints_output_path(default_build):
    result, _ = default_build
    assert result.returncode == 0
    assert "gds_cli" in result.stdout


def test_add_derived_group_resolves_anchor_fk_from_pattern_relations():
    """The derived-dimension YAML path resolves the anchor FK from the
    ``from_pattern`` relations and always threads the sphere config through."""
    from unittest.mock import MagicMock

    from hypertopos.cli.build import _add_derived_group
    from hypertopos.cli.schema import (
        DerivedDimGroup,
        FeatureSpec,
        PatternConfig,
        RelationConfig,
        SphereConfig,
    )

    anchor = PatternConfig(type="anchor", entity_line="accounts")
    event = PatternConfig(
        type="event",
        entity_line="tx",
        relations=[RelationConfig(line="accounts", key_on_entity="from_acct")],
    )
    cfg = SphereConfig(
        sphere_id="s",
        patterns={"account_pattern": anchor, "tx_pattern": event},
    )
    group = DerivedDimGroup(
        from_pattern="tx_pattern",
        features=[FeatureSpec(dimension_name="tx_count", metric="count")],
    )
    builder = MagicMock()

    _add_derived_group(builder, "account_pattern", group, anchor, cfg)

    kwargs = builder.add_derived_dimension.call_args.kwargs
    assert kwargs["event_line"] == "tx"
    assert kwargs["anchor_fk"] == "from_acct"


def test_add_one_feature_requires_sphere_config():
    """Omitting the sphere config is a call error, not a late crash while
    resolving the anchor FK."""
    from unittest.mock import MagicMock

    from hypertopos.cli.build import _add_one_feature
    from hypertopos.cli.schema import FeatureSpec, PatternConfig

    with pytest.raises(TypeError):
        _add_one_feature(  # type: ignore[call-arg]
            MagicMock(),
            "account_pattern",
            FeatureSpec(dimension_name="tx_count", metric="count"),
            PatternConfig(type="anchor", entity_line="accounts"),
            "tx",
            None,
            from_pattern="tx_pattern",
        )
