# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""hypertopos.spark must run on Spark Connect sessions (serverless, shared clusters).

Connect DataFrames have no JVM handle and no RDD API, so any of these names in
the Spark package fails there at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SPARK_PACKAGE = Path(__file__).parents[1] / "hypertopos" / "spark"
CLASSIC_ONLY = re.compile(r"\._jdf\b|\._jvm\b|_collect_as_arrow|\.rdd\b|\bsparkContext\b|\._sc\b")


def test_missing_pyspark_gives_an_install_hint(monkeypatch):
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def no_pyspark(name, *args, **kwargs):
        if name == "pyspark" or name.startswith("pyspark."):
            raise ImportError("No module named 'pyspark'")
        return real_import(name, *args, **kwargs)

    for mod in [
        m for m in sys.modules if m == "hypertopos.spark" or m.startswith("hypertopos.spark.")
    ]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", no_pyspark)
    with pytest.raises(ImportError, match=r'pip install "hypertopos\[spark\]"'):
        importlib.import_module("hypertopos.spark")


@pytest.mark.parametrize("path", sorted(SPARK_PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_classic_only_spark_api(path):
    hits = [
        f"{path.name}:{n}: {line.strip()}"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if CLASSIC_ONLY.search(line)
    ]
    assert not hits, "classic-only Spark API (breaks on Spark Connect):\n" + "\n".join(hits)


def test_entity_table_runs_where_caching_is_refused(monkeypatch):
    """Serverless compute on older client environments rejects persist() and unpersist()."""
    pytest.importorskip("pyspark")
    from pyspark.errors import AnalysisException

    from hypertopos.spark import _api

    class NoCacheFrame:
        def persist(self):
            raise AnalysisException("[CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC]")

        def unpersist(self):
            raise AnalysisException("[CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC]")

        def count(self):
            return 2

    frame = NoCacheFrame()
    monkeypatch.setattr(_api, "entity_frame", lambda df, plan, group_by: frame)
    monkeypatch.setattr(_api, "collect_entities", lambda df: ("collected", df))
    assert _api._entity_table(None, None, None, max_entities=10) == ("collected", frame)


def test_sphere_root_falls_back_where_the_conf_cannot_be_read(monkeypatch, tmp_path):
    """Serverless refuses to read an unknown Spark conf (CONFIG_NOT_AVAILABLE)."""
    pytest.importorskip("pyspark")
    from pyspark.errors import AnalysisException

    from hypertopos.spark import _api

    class Conf:
        def get(self, key, default=None):
            raise AnalysisException(f"[CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION] {key}")

    class Session:
        conf = Conf()

    class Frame:
        sparkSession = Session()

    monkeypatch.setenv("HYPERTOPOS_SPHERE_ROOT", str(tmp_path))
    assert _api._sphere_root(Frame(), None) == str(tmp_path)
