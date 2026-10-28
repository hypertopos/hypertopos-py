# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Ratchet: no new hand-written shape ↔ delta arithmetic outside the geometry helpers.

Stored deltas use the pattern's saved metric (per-dimension σ or Mahalanobis
whitening, then dimension weights). Hand-written ``(x - mu) / sigma`` or
``delta * sigma + mu`` silently drops the whitening and the weights, which is
how several shape/delta mismatches entered the codebase. New code must use
``engine.geometry.calibrated_deltas`` / ``slice_deltas`` (shape → delta) and
``calibrated_shapes`` / ``shapes_from_deltas`` (delta → shape).

The counts below are the existing occurrences (calibration internals,
statistics that are genuinely diagonal, docstrings). They may only go down:
when you remove one, lower the number here.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "hypertopos"
PATTERN = re.compile(
    r"\*\s*_?sigma\w*(\[[^\]]*\])?\s*\+\s*[\w.]*mu\b"
    r"|-\s*[\w.]*mu\w*(\[[^\]]*\])?\s*\)\s*/\s*[\w.]*sigma"
)
# Files that define the metric itself.
EXEMPT = {"engine/geometry.py", "builder/_stats.py"}
BASELINE = {
    "builder/builder.py": 8,
    "builder/dim_blocks.py": 2,
    "engine/counterfactual.py": 5,
    "engine/dim_audit.py": 1,
    "model/sphere.py": 1,
    "navigation/navigator.py": 5,
}


def _counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel in EXEMPT:
            continue
        n = len(PATTERN.findall(path.read_text(encoding="utf-8")))
        if n:
            counts[rel] = n
    return counts


def test_pattern_catches_both_directions() -> None:
    assert PATTERN.search("shape = delta * sigma + pattern.mu")
    assert PATTERN.search("x = delta[:w] * sigma_safe[:w] + mu")
    assert PATTERN.search("d = (shape - mu) / sigma")
    assert PATTERN.search("d = (s - pattern.mu[:w]) / sigma[:w]")
    assert not PATTERN.search("d = calibrated_deltas(shape, mu=mu, sigma_diag=sigma)")


def test_no_new_hand_written_metric_arithmetic() -> None:
    grown = {
        rel: (n, BASELINE.get(rel, 0)) for rel, n in _counts().items() if n > BASELINE.get(rel, 0)
    }
    assert not grown, (
        "new hand-written shape/delta arithmetic (found, allowed): "
        f"{grown} — use engine.geometry.calibrated_deltas / calibrated_shapes"
    )


def test_baseline_is_tight() -> None:
    """Keep the ratchet honest: lower BASELINE when occurrences are removed."""
    current = _counts()
    stale = {
        rel: (current.get(rel, 0), n) for rel, n in BASELINE.items() if current.get(rel, 0) < n
    }
    assert not stale, f"occurrences removed — lower BASELINE (found, baseline): {stale}"
