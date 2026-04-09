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

from hypertopos.cli import main as cli_main


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


def test_cli_build_creates_sphere(tmp_path):
    (tmp_path / "items.csv").write_text("item_id,name\nI-1,Widget\nI-2,Gadget\nI-3,Foo\n")
    yaml_file = tmp_path / "mapping.yaml"
    yaml_file.write_text(
        textwrap.dedent(f"""\
        sphere_id: cli_test
        output_path: {tmp_path / "gds_cli"}
        lines:
          items:
            source: items.csv
            key_col: item_id
        patterns: {{}}
    """)
    )
    result = _run_cli(["build", "--mapping", str(yaml_file)])
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (tmp_path / "gds_cli" / "_gds_meta" / "sphere.json").exists()


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


def test_cli_build_prints_output_path(tmp_path):
    (tmp_path / "data.csv").write_text("id\nX-1\nX-2\n")
    yaml_file = tmp_path / "mapping.yaml"
    yaml_file.write_text(
        textwrap.dedent(f"""\
        sphere_id: print_test
        output_path: {tmp_path / "gds_print"}
        lines:
          data:
            source: data.csv
            key_col: id
        patterns: {{}}
    """)
    )
    result = _run_cli(["build", "--mapping", str(yaml_file)])
    assert result.returncode == 0
    assert "gds_print" in result.stdout
