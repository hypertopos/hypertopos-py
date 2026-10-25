# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Query-side randomized checks give the same answer on every call."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from hypertopos.builder.builder import GDSBuilder
from hypertopos.engine.calibration import CalibrationTracker
from hypertopos.sphere import HyperSphere


def test_assess_false_positive_repeats(tmp_path, monkeypatch) -> None:
    builder = GDSBuilder("fp", str(tmp_path / "sphere"))
    builder.add_line(
        "people",
        pa.table(
            {"primary_key": [f"P{i}" for i in range(50)], "name": [f"n{i}" for i in range(50)]}
        ),
        key_col="primary_key",
        source_id="t",
        role="anchor",
    )
    builder.add_pattern("p", "anchor", "people", relations=[], tracked_properties=["name"])
    navigator = HyperSphere.open(builder.build()).session("fp-test").navigator()
    sphere = navigator._storage.read_sphere()
    sphere.patterns["p"].theta = np.array([1.0], dtype=np.float32)  # just below the entity
    borderline = pa.table({"primary_key": ["P1"], "delta_norm": [1.01], "is_anomaly": [True]})
    monkeypatch.setattr(navigator._storage, "read_sphere", lambda: sphere)
    monkeypatch.setattr(navigator._storage, "read_geometry", lambda *a, **k: borderline)

    runs = [navigator.assess_false_positive("P1", "p", n_perturbations=50) for _ in range(6)]
    assert all(r == runs[0] for r in runs[1:])


def test_calibration_reservoir_repeats() -> None:
    norms = np.random.default_rng(1).random(30_000).astype(np.float32)

    def reservoir() -> np.ndarray:
        tracker = CalibrationTracker.from_stats(np.zeros(2), np.ones(2), np.ones(2), n=10)
        tracker.update_norms(norms)
        return tracker.norm_reservoir.copy()

    assert np.array_equal(reservoir(), reservoir())
