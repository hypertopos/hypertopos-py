# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for anomaly rate guard warning in compute_stats."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypertopos.builder._stats import compute_stats


def test_warning_emitted_for_binary_shape_vectors(caplog: pytest.LogCaptureFixture) -> None:
    """Binary (degenerate) shape vectors produce high anomaly rate -> warning."""
    # 99 identical rows + 1 outlier -> few unique delta_norms
    base = np.zeros((99, 4), dtype=np.float32)
    outlier = np.full((1, 4), 10.0, dtype=np.float32)
    shape_vectors = np.vstack([base, outlier])

    with caplog.at_level(logging.WARNING, logger="hypertopos.builder._stats"):
        compute_stats(shape_vectors, anomaly_percentile=95.0)

    assert any("Anomaly rate" in r.message for r in caplog.records), (
        "Expected anomaly rate warning for binary shape vectors"
    )


def test_no_warning_for_well_distributed_vectors(caplog: pytest.LogCaptureFixture) -> None:
    """Well-distributed continuous vectors should not trigger warning."""
    rng = np.random.default_rng(42)
    shape_vectors = rng.standard_normal((1000, 8)).astype(np.float32)

    with caplog.at_level(logging.WARNING, logger="hypertopos.builder._stats"):
        compute_stats(shape_vectors, anomaly_percentile=95.0)

    assert not any("Anomaly rate" in r.message for r in caplog.records), (
        "No anomaly rate warning expected for well-distributed vectors"
    )
