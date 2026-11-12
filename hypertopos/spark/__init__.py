# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""hypertopos on Spark: one call from a DataFrame of events to a DataFrame of results."""

try:
    import pyspark  # noqa: F401
except ImportError as err:
    raise ImportError(
        'hypertopos.spark needs PySpark: pip install "hypertopos[spark]" '
        "(on Databricks it is already installed)"
    ) from err

from hypertopos.spark._api import find_anomalies
from hypertopos.spark._card import calibration_card
from hypertopos.spark._compare import compare
from hypertopos.spark._explain import Sphere, axis_evidence, explain, sphere_of
from hypertopos.spark._record import evidence_record
from hypertopos.spark._score import score
from hypertopos.spark._status import calibration_status

__all__ = [
    "Sphere",
    "axis_evidence",
    "calibration_card",
    "calibration_status",
    "compare",
    "evidence_record",
    "explain",
    "find_anomalies",
    "score",
    "sphere_of",
]
