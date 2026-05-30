# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Regression tests for missing-geometry resilience.

A declared-but-not-materialized pattern (``geometry/<pid>/data.lance`` absent
on a corrupt or partially built sphere) must not crash the readers that
``sphere validate`` / ``sphere health`` compose:

- ``GDSReader.count_geometry_rows`` returns 0 (mirrors the ``.exists()``
  guards on its siblings ``geometry_column_names`` / ``read_edge_features``)
  rather than raising when the Lance dataset is absent.
- ``run_sphere_validate`` always emits its ``{valid, errors, warnings}`` JSON
  document — a failure in the soft-signal overview pass is converted to a
  warning, never a crash that suppresses the report.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from hypertopos.storage.reader import GDSReader


def test_count_geometry_rows_returns_zero_when_data_lance_missing(tmp_path):
    """A pattern whose geometry/<pid>/data.lance is absent yields 0, not a
    crash. The geometry/<pid> directory exists (so the structural validate
    check passes) but the Lance dataset inside it was never written."""
    base = tmp_path / "gds_corrupt"
    (base / "geometry" / "ghost_pattern").mkdir(parents=True)
    # No data.lance is created inside geometry/ghost_pattern/.

    reader = GDSReader(base_path=str(base))

    assert reader.count_geometry_rows("ghost_pattern") == 0
    # The filtered variant takes the same early-return path.
    assert reader.count_geometry_rows("ghost_pattern", filter="delta_norm >= 1.0") == 0


def test_sphere_validate_emits_json_when_overview_raises(tmp_path, monkeypatch):
    """``run_sphere_validate`` must always emit its JSON document. If the
    soft-signal overview pass raises (e.g. a corrupt sphere whose navigator
    cannot summarise a pattern), the failure becomes a warning rather than a
    crash that suppresses the {valid, errors, warnings} report."""
    import hypertopos.cli.sphere_ops as sphere_ops

    # Minimal on-disk sphere: valid sphere.json declaring one line + one
    # pattern, with the matching points/ and geometry/ directories present so
    # the structural checks pass. The geometry dataset itself is empty (no
    # data.lance), which is what makes the overview pass fail downstream.
    base = tmp_path / "gds_overview_fail"
    meta = base / "_gds_meta"
    meta.mkdir(parents=True)
    (base / "points" / "accounts").mkdir(parents=True)
    (base / "geometry" / "account_pattern").mkdir(parents=True)
    (meta / "sphere.json").write_text(
        json.dumps(
            {
                "sphere_id": "overview_fail",
                "lines": {"accounts": {}},
                "patterns": {"account_pattern": {}},
            }
        ),
        encoding="utf-8",
    )

    class _BoomNav:
        def sphere_overview(self):
            raise RuntimeError("geometry dataset for account_pattern is corrupt")

    monkeypatch.setattr(sphere_ops, "_open_navigator", lambda _p: _BoomNav())

    buf = io.StringIO()
    with redirect_stdout(buf):
        sphere_ops.run_sphere_validate(str(base), strict=False, as_json=True)

    payload = json.loads(buf.getvalue())
    # The report was emitted despite the overview failure.
    assert "valid" in payload
    assert "errors" in payload
    assert "warnings" in payload
    # Structural checks passed (dirs present) and the overview failure is a
    # warning, not an error, so the sphere is still structurally valid.
    assert payload["valid"] is True
    assert any("overview unavailable" in w.lower() for w in payload["warnings"])
