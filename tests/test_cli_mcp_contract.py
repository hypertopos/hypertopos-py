# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""CLI/MCP contract harness — ``sphere health`` mirrors the navigator layer.

The ``hypertopos sphere health`` CLI command and the MCP ``sphere_overview``
+ ``check_alerts`` tools report the same underlying data. Both surfaces are
thin wrappers over ``GDSNavigator.sphere_overview`` / ``check_alerts``. This
harness pins that contract: the CLI subprocess JSON's ``overview`` and
``alerts`` blocks must equal the navigator-layer result for the same sphere.

Design notes:

* **Subprocess, never CliRunner.** We invoke the CLI exactly as the
  ``hypertopos`` console script does — ``[sys.executable, "-m",
  "hypertopos.cli.main", ...]`` — so the full ``sys.argv`` -> argparse ->
  dispatcher -> flag-parsing path is exercised. ``CliRunner.invoke(obj=...)``
  shortcuts (or calling ``main()`` with a patched ``sys.argv``) bypass the
  parser and hide flag-position bugs by construction.

* **Navigator-layer comparison, not MCP-layer.** This test lives in the
  ``hypertopos-py`` package; importing ``hypertopos_mcp`` would create a
  cross-package test dependency. "Mirrors the MCP tool" is conceptual — the
  MCP tools are themselves thin wrappers over the same navigator methods, so
  asserting CLI-JSON == navigator-result pins the identical contract without
  reaching across packages.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hypertopos.sphere import HyperSphere


@pytest.fixture(scope="session")
def small_fixture_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a minimal sphere via subprocess, return its directory.

    Mirrors the fixture in ``test_cli_subprocess.py`` — built through the
    canonical ``hypertopos build`` path so the sphere under test is the
    same code path the CLI asserts against. Session-scoped: read-only from
    every consumer.
    """
    workdir = tmp_path_factory.mktemp("contract_fixture_sphere")
    (workdir / "items.csv").write_text(
        "item_id,name\nI-1,Widget\nI-2,Gadget\nI-3,Foo\n",
        encoding="utf-8",
    )
    yaml_file = workdir / "mapping.yaml"
    sphere_out = workdir / "gds_contract_fixture"
    yaml_file.write_text(
        textwrap.dedent(
            f"""\
            sphere_id: contract_fixture
            output_path: {sphere_out}
            lines:
              items:
                source: items.csv
                key_col: item_id
            patterns: {{}}
            """
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hypertopos.cli.main",
            "build",
            "--mapping",
            str(yaml_file),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(workdir),
        check=True,
    )
    assert (sphere_out / "_gds_meta" / "sphere.json").exists()
    return sphere_out


def _navigator_for(sphere_path: Path):
    """Open the sphere via the public API and return a fresh navigator."""
    return HyperSphere.open(str(sphere_path)).session("contract-test").navigator()


def test_health_overview_matches_navigator(small_fixture_sphere: Path) -> None:
    """CLI ``sphere health`` ``overview`` block == ``navigator.sphere_overview()``.

    The CLI embeds the navigator's ``sphere_overview()`` result verbatim
    under the ``overview`` key. Asserting exact equality catches any future
    drift where the CLI starts massaging the payload differently from the
    navigator (the MCP ``sphere_overview`` tool wraps the same navigator
    method, adding its own ``profiling_alerts`` / ``continuous_mode_note``
    enrichments on top — so this pins CLI == navigator, not CLI == MCP).
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hypertopos.cli.main",
            "sphere",
            "health",
            str(small_fixture_sphere),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    cli_payload = json.loads(result.stdout)

    nav = _navigator_for(small_fixture_sphere)
    expected_overview = nav.sphere_overview()

    # Round-trip the navigator result through JSON so both sides compare as
    # plain JSON values (the CLI serializes with default=str). On the
    # zero-pattern fixture both are the empty list.
    expected_overview_json = json.loads(
        json.dumps(expected_overview, default=str)
    )
    assert cli_payload["overview"] == expected_overview_json


def test_health_alerts_matches_navigator(small_fixture_sphere: Path) -> None:
    """CLI ``sphere health`` ``alerts`` block == ``navigator.check_alerts()``.

    The CLI embeds the navigator's ``check_alerts()`` result verbatim under
    the ``alerts`` key (same shape the MCP ``check_alerts`` tool returns).
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hypertopos.cli.main",
            "sphere",
            "health",
            str(small_fixture_sphere),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    cli_payload = json.loads(result.stdout)

    nav = _navigator_for(small_fixture_sphere)
    expected_alerts = nav.check_alerts()
    expected_alerts_json = json.loads(json.dumps(expected_alerts, default=str))

    assert cli_payload["alerts"] == expected_alerts_json


def test_health_status_consistent_with_alerts(
    small_fixture_sphere: Path,
) -> None:
    """The derived ``status`` is consistent with the embedded alert severities.

    ``status`` is "critical" iff a HIGH alert is present, "warning" iff a
    MEDIUM alert is present (and no HIGH), else "ok". This pins the
    derivation rule the ``--exit-code-on-critical`` CI gate relies on.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hypertopos.cli.main",
            "sphere",
            "health",
            str(small_fixture_sphere),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    severities = {a.get("severity") for a in payload["alerts"]["alerts"]}
    if "HIGH" in severities:
        expected = "critical"
    elif "MEDIUM" in severities:
        expected = "warning"
    else:
        expected = "ok"
    assert payload["status"] == expected
