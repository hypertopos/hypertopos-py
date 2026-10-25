# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Subprocess-driven CLI tests.

NEVER bypass the parser with in-process shortcuts (e.g.
``CliRunner.invoke(obj={...})`` for Click, or calling ``main()`` with
patched ``sys.argv``) in this file. Those shortcuts hide bug classes
like ISSUE-001 / ISSUE-002 (flag-position parsing, missing ``__main__``
guard, entry-point .exe shim edge cases under Windows runners) by
construction.

Every test here MUST spawn a real subprocess via
``[sys.executable, "-m", "hypertopos.cli.main", ...]`` so that the
full ``sys.argv`` -> argparse -> dispatcher path is exercised on every
run.

This harness ships in 0.8.0 M8 as the foundation for cloud-ops CLI
extensions (M7: ``sphere health`` / ``sphere validate --strict`` /
``sphere diff``). The 0.7.3 commands (``build`` / ``validate`` /
``info``) are covered here too so the harness itself has live coverage
from day one rather than only skip-marked skeletons.

Scope: this file keeps one real-process smoke per verb plus every
data-writing pipeline (``build`` via the session fixture, ``sphere
ingest``). Help text, usage errors and bad-path exits run in-process in
``test_cli_inprocess.py`` through the same ``main()`` entry point with a
patched ``sys.argv`` (argparse is still exercised, only interpreter
start-up is skipped).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------

# We invoke the CLI as ``python -m hypertopos.cli.main`` rather than the
# installed ``hypertopos`` console script. Rationale:
#   * Portable across Windows / Linux / macOS without depending on the
#     console-script .exe shim being on PATH (the shim has edge cases
#     under the GitHub Actions Windows runner -- per rozkmina R5).
#   * Uses the exact ``sys.executable`` of the test interpreter, so the
#     subprocess sees the same site-packages and pip-installed
#     ``hypertopos`` as the test process.
#   * Routes through the ``if __name__ == "__main__"`` guard in
#     ``hypertopos/cli/main.py`` -- the same code path as the
#     ``hypertopos`` console script entry point.


def _run_cli(
    *args: str,
    expect_exit_code: int = 0,
    timeout: float = 120.0,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the hypertopos CLI via subprocess and assert exit code.

    Parameters
    ----------
    *args:
        Arguments appended after ``python -m hypertopos.cli.main``.
    expect_exit_code:
        Expected process exit code. Defaults to 0. Pass an explicit
        non-zero value when asserting an error path.
    timeout:
        Hard timeout in seconds. 120s default keeps a runaway test
        from wedging CI; tighten per-test if a command is known fast.
    cwd:
        Optional working directory for the subprocess. Relevant for
        commands that resolve YAML ``source:`` paths relative to the
        config file's own directory.
    """
    cmd = [sys.executable, "-m", "hypertopos.cli.main", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
    )
    assert result.returncode == expect_exit_code, (
        f"CLI exited {result.returncode}, expected {expect_exit_code}\n"
        f"cmd: {cmd}\n"
        f"stdout:\n{result.stdout[:2000]}\n"
        f"stderr:\n{result.stderr[:2000]}"
    )
    return result


# ---------------------------------------------------------------------------
# Small fixture sphere — built once per test session via the canonical
# ``hypertopos build`` subprocess path. This means the fixture sphere
# itself is produced by the same code path the tests assert against
# (build is the only way to get a sphere in 0.7.3).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_fixture_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a minimal sphere via subprocess, return its directory.

    Session-scoped because a full CLI build round-trip (Python startup
    + YAML load + Arrow write) costs more than the test logic itself,
    and the resulting sphere is read-only from every consumer.
    """
    workdir = tmp_path_factory.mktemp("m8_fixture_sphere")
    (workdir / "items.csv").write_text(
        "item_id,name\nI-1,Widget\nI-2,Gadget\nI-3,Foo\n",
        encoding="utf-8",
    )
    yaml_file = workdir / "mapping.yaml"
    sphere_out = workdir / "gds_m8_fixture"
    yaml_file.write_text(
        textwrap.dedent(
            f"""\
            sphere_id: m8_fixture
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
    _run_cli("build", "--mapping", str(yaml_file), cwd=workdir)
    assert (sphere_out / "_gds_meta" / "sphere.json").exists(), (
        "fixture sphere build did not produce sphere.json"
    )
    return sphere_out


@pytest.fixture(scope="session")
def ingest_fixture_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny anchor sphere with one pattern for ingest tests.

    ``sphere ingest`` calls ``incremental_update``, which needs a pattern
    with reconstructable geometry — the zero-pattern ``small_fixture_sphere``
    can't exercise it. This builds a 20-customer anchor sphere with one
    derived count dimension (``event_count``) so the pattern mu is 1-wide and
    a changed-entities table carrying ``event_count`` round-trips through the
    builder's incremental geometry path. Built in-process via ``GDSBuilder``
    (the only API that produces a pattern-bearing sphere); the system under
    test is still the CLI subprocess that ingests into it.
    """
    import numpy as np
    import pyarrow as pa

    from hypertopos.builder.builder import GDSBuilder

    out = tmp_path_factory.mktemp("ingest_fixture") / "gds_ingest_fixture"
    n = 20
    rng = np.random.default_rng(0)
    custs = pa.table({"primary_key": pa.array([f"C{i}" for i in range(n)], type=pa.string())})
    events = pa.table(
        {
            "primary_key": pa.array([f"E{i}" for i in range(n * 4)], type=pa.string()),
            "cust_fk": pa.array(
                [f"C{int(rng.integers(0, n))}" for _ in range(n * 4)],
                type=pa.string(),
            ),
        }
    )
    b = GDSBuilder("ingest_fixture", str(out))
    b.add_line(
        "customers",
        custs,
        key_col="primary_key",
        source_id="t",
        role="anchor",
    )
    b.add_line(
        "events",
        events,
        key_col="primary_key",
        source_id="t",
        role="event",
    )
    b.add_derived_dimension(
        "customers",
        "events",
        "cust_fk",
        "count",
        None,
        "event_count",
        edge_max="auto",
    )
    b.add_pattern("cust_pattern", "anchor", "customers", relations=[])
    b.build()
    assert (out / "_gds_meta" / "sphere.json").exists()
    return out


@pytest.fixture
def ingest_sphere_copy(
    ingest_fixture_sphere: Path,
    tmp_path: Path,
) -> Path:
    """Clone the ingest fixture to a fresh dir so each ingest test mutates
    its own copy (ingest writes geometry + sphere.json)."""
    from tests.conftest import clone_sphere

    dest = tmp_path / "gds_ingest_copy"
    clone_sphere(ingest_fixture_sphere, dest)
    return dest


def _make_points_arrow(path: Path) -> None:
    """Write a 5-row changed-entities Arrow IPC table to ``path``.

    Carries ``primary_key`` + ``event_count`` (the derived dim the fixture
    pattern reconstructs), matching ``test_incremental``'s changed-entities
    shape.
    """
    import pyarrow as pa
    import pyarrow.feather as feather

    tbl = pa.table(
        {
            "primary_key": pa.array([f"C{100 + i}" for i in range(5)], type=pa.string()),
            "event_count": pa.array([5.0] * 5, type=pa.float64()),
        }
    )
    feather.write_feather(tbl, str(path))


# ---------------------------------------------------------------------------
# 0.7.3 baseline coverage — sanity that the subprocess driver itself
# works end-to-end against the commands that ship today.
# ---------------------------------------------------------------------------


def test_hypertopos_validate_help_documents_config() -> None:
    """``hypertopos validate --help`` documents the required ``--config``."""
    result = _run_cli("validate", "--help")
    assert "--config" in result.stdout


def test_module_entry_point_emits_no_runpy_warning() -> None:
    """``python -m hypertopos.cli.main`` runs without a runpy RuntimeWarning."""
    cmd = [sys.executable, "-W", "error::RuntimeWarning", "-m", "hypertopos.cli.main", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[:2000]
    assert "RuntimeWarning" not in result.stderr


def test_hypertopos_info_on_built_sphere(
    small_fixture_sphere: Path,
) -> None:
    """``hypertopos info <sphere>`` prints sphere id + line/pattern summary.

    Round-trips the build fixture through the read path. Verifies that
    a sphere built via subprocess is consumable via subprocess.
    """
    result = _run_cli("info", str(small_fixture_sphere))
    assert "m8_fixture" in result.stdout  # sphere_id from the YAML
    assert "Lines" in result.stdout
    assert "Patterns" in result.stdout
    assert "items" in result.stdout  # line id from the YAML


# ---------------------------------------------------------------------------
# M7 cloud-ops CLI extension — sphere health / validate --strict / diff.
# Every test spawns a real subprocess against the zero-pattern fixture
# sphere. The fixture has ``patterns: {}`` so each command's empty path
# is exercised here; pattern-bearing behavior is covered by live smoke
# against benchmark spheres in the PR.
# ---------------------------------------------------------------------------


def test_sphere_health_subprocess_json_output(
    small_fixture_sphere: Path,
) -> None:
    """``hypertopos sphere health <path> --json`` returns parsable JSON.

    On the zero-pattern fixture:
      * exit code 0
      * stdout is valid JSON (and ONLY JSON — json.loads must not choke)
      * payload has ``status`` == "ok" (no alerts on a pattern-less sphere)
      * ``overview`` is the empty list, ``alerts`` reports 0 patterns checked
    """
    result = _run_cli(
        "sphere",
        "health",
        str(small_fixture_sphere),
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["overview"] == []
    assert payload["alerts"]["patterns_checked"] == 0
    assert payload["alerts"]["alerts"] == []


def test_sphere_validate_strict_subprocess(
    small_fixture_sphere: Path,
) -> None:
    """``hypertopos sphere validate --strict --json`` emits structured output.

    Strict mode promotes warnings to errors. The fixture sphere is
    structurally valid with no patterns (hence no warnings), so strict
    mode still passes: exit 0, ``valid`` True, empty error/warning lists.
    JSON output is for Airflow BashOperator paths that parse the result
    without regex on stdout.
    """
    result = _run_cli(
        "sphere",
        "validate",
        str(small_fixture_sphere),
        "--strict",
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["strict"] is True
    assert payload["errors"] == []
    assert payload["warnings"] == []


def test_sphere_diff_subprocess(
    small_fixture_sphere: Path,
) -> None:
    """``hypertopos sphere diff <old> <new> --json`` reports the delta.

    Pre-deploy diff: pattern inventory delta + calibration drift. On two
    identical sphere paths the diff is marked ``identical: true`` with
    empty inventory deltas and (for the zero-pattern fixture) no
    calibration-drift rows.
    """
    result = _run_cli(
        "sphere",
        "diff",
        str(small_fixture_sphere),
        str(small_fixture_sphere),
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["identical"] is True
    assert payload["pattern_inventory"]["added"] == []
    assert payload["pattern_inventory"]["removed"] == []
    assert payload["pattern_inventory"]["common"] == []
    assert payload["calibration_drift"] == []


# ---------------------------------------------------------------------------
# sphere ingest — incremental append of a new-/changed-entities table.
# Each test ingests into its own clone (ingest mutates geometry + sphere.json).
# ---------------------------------------------------------------------------


def test_sphere_ingest_arrow_json_output(
    ingest_sphere_copy: Path,
    tmp_path: Path,
) -> None:
    """``sphere ingest <path> --points <arrow> --json`` ingests + reports added.

    Ingests a 5-row Arrow IPC table into the single-pattern fixture clone:
      * exit 0
      * stdout is valid JSON
      * ``added`` == 5, ``pattern_id`` auto-resolved (sole pattern)
      * ``geometry_version_after`` advances past ``_before`` (rows appended)
    """
    points = tmp_path / "new_custs.arrow"
    _make_points_arrow(points)

    result = _run_cli(
        "sphere",
        "ingest",
        str(ingest_sphere_copy),
        "--points",
        str(points),
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["pattern_id"] == "cust_pattern"
    assert payload["added"] == 5
    assert payload["modified"] == 0
    assert payload["deleted"] == 0
    assert payload["finalized"] is False
    assert payload["geometry_version_before"] is not None
    assert payload["geometry_version_after"] > payload["geometry_version_before"]


def test_sphere_ingest_finalize_flag(
    ingest_sphere_copy: Path,
    tmp_path: Path,
) -> None:
    """``--finalize`` round-trips: ``finalized`` is True in the JSON summary."""
    points = tmp_path / "new_custs.arrow"
    _make_points_arrow(points)

    result = _run_cli(
        "sphere",
        "ingest",
        str(ingest_sphere_copy),
        "--points",
        str(points),
        "--finalize",
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["added"] == 5
    assert payload["finalized"] is True


def test_sphere_ingest_non_json_human_output(
    ingest_sphere_copy: Path,
    tmp_path: Path,
) -> None:
    """``sphere ingest`` without --json prints a human-readable summary."""
    points = tmp_path / "new_custs.arrow"
    _make_points_arrow(points)

    result = _run_cli(
        "sphere",
        "ingest",
        str(ingest_sphere_copy),
        "--points",
        str(points),
    )
    assert "Pattern: cust_pattern" in result.stdout
    assert "Added: 5" in result.stdout
