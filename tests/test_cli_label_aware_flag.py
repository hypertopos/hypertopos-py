# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""CLI plumbing for the ``--label-aware-calibration`` flag.

The flag is a passthrough: argparse must accept it, `run_build` must
receive it, and on a builder without a sphere-level ``label_audit``
block it must be a no-op. The YAML loader that populates the block
lands in M1.2 — these tests only cover M1.1's flag-acceptance contract.
"""
from __future__ import annotations

import sys

from hypertopos.cli import main as cli_main


def test_label_aware_flag_threads_into_run_build(monkeypatch):
    """`--label-aware-calibration` flips the ``label_aware_calibration``
    kwarg passed to ``run_build``."""
    captured: dict[str, object] = {}

    def fake_run_build(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "hypertopos.cli.build.run_build", fake_run_build, raising=True,
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "hypertopos", "build",
            "--config", "dummy.yaml",
            "--label-aware-calibration",
        ],
    )
    cli_main()
    assert captured.get("label_aware_calibration") is True


def test_label_aware_flag_defaults_off(monkeypatch):
    """Without the flag ``label_aware_calibration`` defaults to False."""
    captured: dict[str, object] = {}

    def fake_run_build(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "hypertopos.cli.build.run_build", fake_run_build, raising=True,
    )
    monkeypatch.setattr(
        sys, "argv",
        ["hypertopos", "build", "--config", "dummy.yaml"],
    )
    cli_main()
    assert captured.get("label_aware_calibration") is False


def test_builder_default_label_aware_state_is_off():
    """Fresh ``GDSBuilder`` carries the opt-in flag off and no block."""
    import tempfile

    from hypertopos.builder.builder import GDSBuilder

    with tempfile.TemporaryDirectory() as tmp:
        builder = GDSBuilder("test", tmp)
    assert builder._label_aware_calibration is False
    assert builder._label_audit_block is None


def test_run_build_threads_flag_into_builder(monkeypatch, tmp_path):
    """``run_build(..., label_aware_calibration=True)`` sets the
    builder attribute. M1.2 will populate ``_label_audit_block`` from
    the YAML config; until then the flag alone is a no-op gate.
    """
    from hypertopos.cli import build as build_mod

    # Minimal config: 1 source, 1 line, 1 anchor pattern with no
    # relations / event dims. Forces the build path to short-circuit
    # without crashing on missing data.
    captured: dict[str, bool] = {}

    def fake_do_build(*args, **kwargs):
        captured["label_aware_calibration"] = kwargs.get(
            "label_aware_calibration", False,
        )

    monkeypatch.setattr(build_mod, "_do_build", fake_do_build, raising=True)
    cfg_path = tmp_path / "sphere.yaml"
    cfg_path.write_text(
        "sphere_id: x\n"
        "sources: {}\n"
        "lines: {}\n"
        "patterns: {}\n",
    )

    # Stub parse_config so we don't depend on full schema validation.
    from hypertopos.cli.schema import SphereConfig

    monkeypatch.setattr(
        "hypertopos.cli.build.parse_config",
        lambda _p: SphereConfig(sphere_id="x"),
        raising=True,
    )

    build_mod.run_build(
        str(cfg_path), str(tmp_path / "out"),
        force=False, verbose=False,
        label_aware_calibration=True,
    )
    assert captured["label_aware_calibration"] is True
