# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Per-tool perf-budget regression harness.

This guards the read hot path against latency regressions on a real,
large sphere. It is deliberately a *regression* gate, not a benchmark:
the absolute PD3 stretch targets (e.g. read_points_batch cold < 400 ms,
warm-p50 < 100 ms) are recorded in the cycle's perf report, not asserted
here, because absolute milliseconds drift with hardware and load.

The fixture sphere shipped with the test suite is too small to exercise
the index-heavy paths meaningfully (everything sits sub-millisecond and
proves nothing about the row-id batching win). So this module SKIPS unless
``HYPERTOPOS_PERF_SPHERE`` points at a real benchmark sphere directory.

Run locally against a benchmark sphere::

    HYPERTOPOS_PERF_SPHERE=benchmark/ibm-aml/hi_small_sphere/gds_aml_hi_small \
        .venv/Scripts/python -m pytest packages/hypertopos-py/tests/test_perf_budget.py -v -s

Each tool records the (elapsed_ms, count, size) triple via the captured
``perf_record`` so a reviewer sees what moved — time alone cannot tell a
zero-result early-bail from a real scan. The asserted ceilings are loose
(generous multiples of the PD3 baseline) and exist only to fail a gross
regression, never to flake on a busy machine.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import pytest

_PERF_SPHERE = os.environ.get("HYPERTOPOS_PERF_SPHERE")

pytestmark = pytest.mark.skipif(
    not _PERF_SPHERE or not Path(_PERF_SPHERE).exists(),
    reason="HYPERTOPOS_PERF_SPHERE not set to an existing sphere directory",
)

# Loose regression ceilings (ms). Generous multiples of the PD3 cold-mean
# baselines (read_points_batch 1127, find_anomalies 567, sphere_overview 265)
# so only a gross regression trips them on a shared machine.
_COLD_CEILING_MS = {
    "read_points_batch": 2500.0,
    "find_anomalies": 2000.0,
    "sphere_overview": 1500.0,
}
_WARM_CEILING_MS = {
    "read_points_batch": 400.0,
    "find_anomalies": 1200.0,
    "sphere_overview": 800.0,
}


def _sphere_meta(base: Path) -> dict:
    return json.loads((base / "_gds_meta" / "sphere.json").read_text())


def _anchor_pattern(meta: dict) -> str:
    """Pick an entity-anchor pattern (one with an entity line)."""
    for pid, pat in meta["patterns"].items():
        if pat.get("pattern_type") == "anchor":
            return pid
    return next(iter(meta["patterns"]))


def _measure(fn, warm_reps: int = 5):
    """Return (cold_ms, warm_p50_ms, result)."""
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    cold = (time.perf_counter() - t0) * 1000.0
    warm = []
    for _ in range(warm_reps):
        t0 = time.perf_counter()
        result = fn()
        warm.append((time.perf_counter() - t0) * 1000.0)
    return cold, sorted(warm)[len(warm) // 2], result


@pytest.fixture(scope="module")
def perf_ctx():
    from hypertopos.storage.reader import GDSReader

    base = Path(_PERF_SPHERE)
    meta = _sphere_meta(base)
    pid = _anchor_pattern(meta)
    pat = meta["patterns"][pid]
    entity_line = pat.get("entity_line") or pat["relations"][0]["line_id"]
    version = meta["lines"][entity_line]["versions"][-1]
    return {
        "base": base,
        "meta": meta,
        "pattern_id": pid,
        "entity_line": entity_line,
        "version": version,
        "reader_cls": GDSReader,
    }


def test_read_points_batch_budget(perf_ctx, capsys):
    import lance

    ctx = perf_ctx
    pts = (
        ctx["base"] / "points" / ctx["entity_line"]
        / f"v={ctx['version']}" / "data.lance"
    )
    keys = (
        lance.dataset(str(pts))
        .scanner(columns=["primary_key"], limit=100)
        .to_table()["primary_key"]
        .to_pylist()
    )
    reader = ctx["reader_cls"](str(ctx["base"]))
    reader.read_sphere()

    def call():
        return reader.read_points_batch(ctx["entity_line"], ctx["version"], keys)

    cold, warm_p50, out = _measure(call)
    with capsys.disabled():
        print(
            f"\n[perf] read_points_batch cold={cold:.1f}ms warm_p50={warm_p50:.1f}ms "
            f"count={out.num_rows} size={out.nbytes}B "
            f"handle_cache={reader.points_cache_stats()}"
        )
    assert cold < _COLD_CEILING_MS["read_points_batch"]
    assert warm_p50 < _WARM_CEILING_MS["read_points_batch"]


def test_find_anomalies_budget(perf_ctx, capsys):
    from hypertopos.sphere import HyperSphere

    ctx = perf_ctx

    def make_call():
        sph = HyperSphere.open(str(ctx["base"]))
        nav = sph.session("perf").navigator()
        return nav

    nav = make_call()

    def call():
        polys, total, _emerging, _meta = nav.π5_attract_anomaly(
            ctx["pattern_id"], top_n=100, sample_size=5000
        )
        return polys, total

    cold, warm_p50, (polys, total) = _measure(call)
    with capsys.disabled():
        print(
            f"\n[perf] find_anomalies cold={cold:.1f}ms warm_p50={warm_p50:.1f}ms "
            f"count={len(polys)} total_found={total}"
        )
    assert cold < _COLD_CEILING_MS["find_anomalies"]
    assert warm_p50 < _WARM_CEILING_MS["find_anomalies"]


def test_sphere_overview_budget(perf_ctx, capsys):
    from hypertopos.sphere import HyperSphere

    ctx = perf_ctx
    nav = HyperSphere.open(str(ctx["base"])).session("perf").navigator()

    def call():
        return nav.sphere_overview()

    cold, warm_p50, out = _measure(call)
    with capsys.disabled():
        print(
            f"\n[perf] sphere_overview cold={cold:.1f}ms warm_p50={warm_p50:.1f}ms "
            f"count={len(out)}"
        )
    assert cold < _COLD_CEILING_MS["sphere_overview"]
    assert warm_p50 < _WARM_CEILING_MS["sphere_overview"]
