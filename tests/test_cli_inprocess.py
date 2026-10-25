# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""In-process CLI tests: help text, usage errors and read-only verbs.

These tests call the real console-script entry point
(``hypertopos.cli.main:main``) with a patched ``sys.argv``, so the full
argparse -> dispatcher path still runs on every call. Only the Python
start-up cost of a subprocess is skipped. Do NOT shortcut the parser
(e.g. by calling ``run_sphere_health`` directly) in this file.

Process-level behaviour (real binary exit codes, stdout encoding, the
``__main__`` guard, data-writing pipelines such as ``build`` and
``sphere ingest``) stays covered by ``test_cli_subprocess.py``.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from hypertopos.cli.main import main
from tests.test_cli_subprocess import (  # noqa: F401  (pytest fixtures)
    _make_points_arrow,
    ingest_fixture_sphere,
    ingest_sphere_copy,
)

# ---------------------------------------------------------------------------
# In-process driver
# ---------------------------------------------------------------------------


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *args: str,
    expect_exit_code: int = 0,
) -> tuple[str, str]:
    """Run ``main()`` with ``sys.argv = ["hypertopos", *args]``.

    Returns ``(stdout, stderr)``. A normal return counts as exit code 0;
    ``SystemExit`` is translated to its integer exit code (``None`` -> 0).
    """
    monkeypatch.setattr(sys, "argv", ["hypertopos", *args])
    code: int | str | None
    try:
        main()
        code = 0
    except SystemExit as exc:
        code = exc.code
    if code is None:
        code = 0
    out, err = capsys.readouterr()
    assert code == expect_exit_code, (
        f"CLI exited {code!r}, expected {expect_exit_code}\n"
        f"argv: {args}\n"
        f"stdout:\n{out[:2000]}\n"
        f"stderr:\n{err[:2000]}"
    )
    return out, err


@pytest.fixture(scope="module")
def small_fixture_sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the zero-pattern fixture sphere in-process via ``main(["build", ...])``.

    Same YAML as the subprocess fixture; the subprocess build path itself is
    covered by ``test_cli_subprocess.py``.
    """
    workdir = tmp_path_factory.mktemp("m8_fixture_sphere_inproc")
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
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(sys, "argv", ["hypertopos", "build", "--mapping", str(yaml_file)])
        main()
    finally:
        mp.undo()
    assert (sphere_out / "_gds_meta" / "sphere.json").exists(), (
        "fixture sphere build did not produce sphere.json"
    )
    return sphere_out


# ---------------------------------------------------------------------------
# Top-level help / usage
# ---------------------------------------------------------------------------


def test_hypertopos_help_renders(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sanity: ``hypertopos --help`` exits 0 and renders the verb list."""
    out, _ = _run_main(monkeypatch, capsys, "--help")
    assert "Usage:" in out or "usage:" in out
    assert "build" in out
    assert "validate" in out
    assert "info" in out


def test_hypertopos_build_help_documents_required_args(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hypertopos build --help`` exposes the required config flags."""
    out, _ = _run_main(monkeypatch, capsys, "build", "--help")
    assert "--config" in out
    assert "--mapping" in out


def test_hypertopos_info_help_documents_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hypertopos info --help`` documents the positional sphere path."""
    out, _ = _run_main(monkeypatch, capsys, "info", "--help")
    assert "path" in out.lower()


def test_hypertopos_no_args_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running with no subcommand prints help and exits non-zero."""
    _run_main(monkeypatch, capsys, expect_exit_code=1)


def test_hypertopos_unknown_command_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown subcommand is rejected by argparse with non-zero exit."""
    monkeypatch.setattr(sys, "argv", ["hypertopos", "nonexistent-verb"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code not in (0, None)


def test_hypertopos_info_on_nonexistent_path_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hypertopos info`` on a non-sphere directory exits 1 with 'error:'."""
    out, err = _run_main(
        monkeypatch,
        capsys,
        "info",
        "/definitely/not/a/sphere/anywhere",
        expect_exit_code=1,
    )
    assert "error" in (err + out).lower()


def test_hypertopos_info_reads_roles_rows_and_pattern_types(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fixture_sphere_path: Path,
) -> None:
    """``hypertopos info`` prints each line's role and row count and each
    pattern's type and entity line from the current sphere metadata."""
    from hypertopos.storage.reader import GDSReader

    out, _ = _run_main(monkeypatch, capsys, "info", str(fixture_sphere_path))
    assert "=?" not in out

    reader = GDSReader(str(fixture_sphere_path))
    sphere = reader.read_sphere()
    for line_id, line in sphere.lines.items():
        rows = reader.count_points_rows(line_id, line.current_version())
        row = next(r for r in out.splitlines() if r.strip().startswith(line_id + " "))
        assert f"role={line.line_role}" in row
        assert f"rows={rows}" in row
    for pattern_id, pattern in sphere.patterns.items():
        row = next(r for r in out.splitlines() if r.strip().startswith(pattern_id + " "))
        assert f"type={pattern.pattern_type}" in row
        assert f"entity={sphere.entity_line(pattern_id)}" in row


# ---------------------------------------------------------------------------
# sphere health / validate / diff
# ---------------------------------------------------------------------------


def test_sphere_help_lists_cloudops_verbs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hypertopos sphere --help`` lists health / validate / diff."""
    out, _ = _run_main(monkeypatch, capsys, "sphere", "--help")
    assert "health" in out
    assert "validate" in out
    assert "diff" in out


def test_sphere_no_subcommand_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hypertopos sphere`` with no subcommand prints help, exits non-zero."""
    _run_main(monkeypatch, capsys, "sphere", expect_exit_code=1)


def test_sphere_health_non_json_human_output(
    small_fixture_sphere: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``hypertopos sphere health <path>`` (no --json) prints a status line."""
    out, _ = _run_main(monkeypatch, capsys, "sphere", "health", str(small_fixture_sphere))
    assert "Status: ok" in out


def test_sphere_health_exit_code_on_critical(
    small_fixture_sphere: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--exit-code-on-critical`` exits 0 when no critical alerts are present."""
    _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "health",
        str(small_fixture_sphere),
        "--exit-code-on-critical",
        expect_exit_code=0,
    )


def test_sphere_health_bad_path_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sphere health`` on a non-sphere directory exits 1 with 'error'."""
    out, err = _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "health",
        "/definitely/not/a/sphere/anywhere",
        expect_exit_code=1,
    )
    assert "error" in (err + out).lower()


def test_sphere_validate_non_strict_subprocess(
    small_fixture_sphere: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``hypertopos sphere validate <path> --json`` passes on a valid sphere.

    Name kept from the original subprocess test for history traceability.
    """
    out, _ = _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "validate",
        str(small_fixture_sphere),
        "--json",
    )
    payload = json.loads(out)
    assert payload["valid"] is True
    assert payload["strict"] is False


def test_sphere_validate_bad_path_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sphere validate`` on a non-sphere directory exits 1."""
    out, err = _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "validate",
        "/definitely/not/a/sphere/anywhere",
        expect_exit_code=1,
    )
    assert "error" in (err + out).lower()


def test_sphere_diff_bad_path_exits_one(
    small_fixture_sphere: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``sphere diff`` exits 1 when either path is not a sphere."""
    out, err = _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "diff",
        str(small_fixture_sphere),
        "/definitely/not/a/sphere/anywhere",
        expect_exit_code=1,
    )
    assert "error" in (err + out).lower()


# ---------------------------------------------------------------------------
# sphere ingest — help and error paths (data-writing paths stay subprocess)
# ---------------------------------------------------------------------------


def test_sphere_ingest_help_renders(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hypertopos sphere ingest --help`` documents the points/pattern flags."""
    out, _ = _run_main(monkeypatch, capsys, "sphere", "ingest", "--help")
    assert "--points" in out
    assert "--pattern" in out
    assert "--finalize" in out


def test_sphere_help_lists_ingest_verb(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hypertopos sphere --help`` now lists the ingest subcommand."""
    out, _ = _run_main(monkeypatch, capsys, "sphere", "--help")
    assert "ingest" in out


def test_sphere_ingest_bad_sphere_path_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``sphere ingest`` on a non-sphere directory exits 1 with clean error."""
    points = tmp_path / "new_custs.arrow"
    _make_points_arrow(points)
    out, err = _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "ingest",
        "/definitely/not/a/sphere/anywhere",
        "--points",
        str(points),
        expect_exit_code=1,
    )
    assert "error" in (err + out).lower()


def test_sphere_ingest_missing_points_file_exits_one(
    ingest_sphere_copy: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing --points file exits 1 with a clear 'not found' message."""
    out, err = _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "ingest",
        str(ingest_sphere_copy),
        "--points",
        "/definitely/not/a/file.arrow",
        expect_exit_code=1,
    )
    assert "not found" in (err + out).lower()


def test_sphere_ingest_unknown_pattern_exits_one(
    ingest_sphere_copy: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--pattern`` naming a non-existent pattern exits 1 with a clear error."""
    points = tmp_path / "new_custs.arrow"
    _make_points_arrow(points)
    out, err = _run_main(
        monkeypatch,
        capsys,
        "sphere",
        "ingest",
        str(ingest_sphere_copy),
        "--points",
        str(points),
        "--pattern",
        "no_such_pattern",
        expect_exit_code=1,
    )
    assert "no_such_pattern" in (err + out)


# ---------------------------------------------------------------------------
# Strict-JSON sanitization — navigator math can emit ±inf / NaN (e.g.
# overall_drift_rms over a zero-dimension pattern, or a NaN-tainted
# theta_norm). The empty fixture sphere can never exercise that path, so
# this is a discriminator over the sanitizer the JSON verbs share, feeding
# engineered non-finite floats and asserting the serialized stdout carries
# no ``NaN`` / ``Infinity`` literal that strict parsers (jq, Airflow) would
# reject.
# ---------------------------------------------------------------------------


def test_dump_json_strips_non_finite_floats() -> None:
    """``_dump_json`` replaces ±inf / NaN with null and emits strict JSON.

    Covers both Python ``float`` and ``numpy.floating`` non-finite values —
    the latter would otherwise slip through ``default=str`` as the literal
    string ``"nan"``.
    """
    import numpy as np

    from hypertopos.cli.sphere_ops import _dump_json

    payload = {
        "status": "ok",
        "drift_rms": float("nan"),
        "drift_rms_np": np.float64("nan"),
        "nested": {"pos_inf": float("inf"), "neg_inf": float("-inf")},
        "rows": [{"v": float("nan")}, {"v": 1.5}],
    }
    text = _dump_json(payload)

    # No non-finite literal survives into the output. (The substring check is
    # safe because no key in this payload contains these tokens.)
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "nan" not in text
    # And the document round-trips through a strict parser unchanged in shape.
    parsed = json.loads(text)
    assert parsed["drift_rms"] is None
    assert parsed["drift_rms_np"] is None
    assert parsed["nested"]["pos_inf"] is None
    assert parsed["nested"]["neg_inf"] is None
    assert parsed["rows"][0]["v"] is None
    assert parsed["rows"][1]["v"] == 1.5
