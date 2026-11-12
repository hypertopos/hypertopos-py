# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Cached DataFrames must not pile up in the one JVM a test worker keeps."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from tests.conftest import release_spark_cache  # noqa: E402


def test_cached_frames_are_released(spark):
    frame = spark.range(10).cache()
    frame.count()
    manager = spark._jsparkSession.sharedState().cacheManager()
    assert not manager.isEmpty()
    release_spark_cache()
    assert manager.isEmpty()


def test_the_session_keeps_little_execution_history(spark):
    # the driver holds plan text and metrics per retained execution; the default is 1,000
    assert int(spark.conf.get("spark.sql.ui.retainedExecutions")) <= 5
