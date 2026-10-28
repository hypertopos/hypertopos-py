"""One non-finite p-value must not erase the FDR result for every other entity."""

from __future__ import annotations

import math

import numpy as np

from hypertopos.engine import fdr


def test_bh_q_values_ignore_a_nan_p_value() -> None:
    q = fdr.q_values_from_p_values(np.array([0.001, 0.02, 0.5, math.nan]))
    assert np.isfinite(q).all()
    clean = fdr.q_values_from_p_values(np.array([0.001, 0.02, 0.5, 1.0]))
    assert np.allclose(q, clean)


def test_bh_rejections_survive_a_nan_p_value() -> None:
    result = fdr.benjamini_hochberg(np.array([0.001, 0.02, 0.5, math.nan]), alpha=0.05)
    rejected = result.rejected if hasattr(result, "rejected") else result[0]
    assert list(rejected) == [True, True, False, False]


def test_storey_q_values_ignore_a_nan_p_value() -> None:
    q = fdr.storey_q_values(np.array([0.001, 0.02, 0.5, math.nan] * 5))
    assert np.isfinite(q).all()
