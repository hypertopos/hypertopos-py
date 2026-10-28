# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""passive_scan intensity on a calibrated pattern follows the alarm budget.

On a calibrated pattern θ is not the anomaly boundary, so a direct source's
``anomaly_intensity`` is ``log(conformal_p) / log(alpha)``: at least 1 exactly
for the entities the alarm budget flags.
"""

from __future__ import annotations

import math
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from test_builder_robust import _builder

from hypertopos.navigation.scanner import PassiveScanner
from hypertopos.sphere import HyperSphere

PID = "account_pattern"


@pytest.fixture(scope="module")
def sphere_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("scancal") / "sphere"
    _builder(out).build()
    return out


@pytest.fixture(scope="module")
def session(sphere_dir: Path):  # type: ignore[no-untyped-def]
    sess = HyperSphere.open(str(sphere_dir)).session("t")
    yield sess
    sess.__exit__(None, None, None)


def _scanner(session) -> PassiveScanner:  # type: ignore[no-untyped-def]
    return PassiveScanner(
        reader=session._reader,
        sphere=session._reader.read_sphere(),
        manifest=session._manifest,
    )


def _geometry(sphere_dir: Path) -> dict[str, tuple[float, float, bool]]:
    geo = lance.dataset(str(sphere_dir / "geometry" / PID / "data.lance")).to_table(
        columns=["primary_key", "delta_norm", "conformal_p", "is_anomaly"]
    )
    return {
        pk: (float(dn), float(p), bool(flag))
        for pk, dn, p, flag in zip(
            geo["primary_key"].to_pylist(),
            geo["delta_norm"].to_pylist(),
            geo["conformal_p"].to_pylist(),
            geo["is_anomaly"].to_pylist(),
            strict=True,
        )
    }


def test_fixture_flags_an_entity_inside_theta(sphere_dir: Path, session) -> None:  # type: ignore[no-untyped-def]
    pattern = session._reader.read_sphere().patterns[PID]
    assert pattern.score_dims, "fixture pattern must be calibrated"
    rows = _geometry(sphere_dir)
    assert any(flag and dn < pattern.theta_norm for dn, _, flag in rows.values())


def test_direct_intensity_is_the_alarm_budget_scale(sphere_dir: Path, session) -> None:  # type: ignore[no-untyped-def]
    pattern = session._reader.read_sphere().patterns[PID]
    alpha = 1.0 - pattern.anomaly_percentile / 100.0
    rows = _geometry(sphere_dir)

    result = _scanner(session).add_source("acct", PID, key_type="direct").scan("accounts")

    flagged = {pk for pk, (_, _, flag) in rows.items() if flag}
    assert {h.primary_key for h in result.hits} == flagged
    for hit in result.hits:
        intensity = hit.sources["acct"].anomaly_intensity
        p = rows[hit.primary_key][1]
        assert intensity == pytest.approx(math.log(p) / math.log(alpha), rel=1e-6)
        assert intensity >= 1.0


def test_entity_flagged_exactly_at_the_budget_counts_one(session, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # conformal_p is stored as float32: a p exactly at the budget is flagged
    # (decided in float64) and reads back slightly above alpha.
    pattern = session._reader.read_sphere().patterns[PID]
    alpha = 1.0 - pattern.anomaly_percentile / 100.0
    real_read = session._reader.read_geometry

    def at_budget(*args, **kwargs):  # type: ignore[no-untyped-def]
        table = real_read(*args, **kwargs)
        p = pa.array(np.full(table.num_rows, alpha, dtype=np.float32))
        return table.set_column(table.schema.get_field_index("conformal_p"), "conformal_p", p)

    monkeypatch.setattr(session._reader, "read_geometry", at_budget)
    result = _scanner(session).add_source("acct", PID, key_type="direct").scan("accounts")
    assert result.hits
    assert all(h.sources["acct"].anomaly_intensity >= 1.0 for h in result.hits)


def test_borderline_intensity_stays_below_one(session) -> None:  # type: ignore[no-untyped-def]
    scanner = _scanner(session).add_borderline_source("bl", PID, rank_threshold=50)
    result = scanner.scan("accounts")
    assert result.hits, "fixture must have unflagged entities above the rank cut"
    assert all(h.sources["bl"].anomaly_intensity < 1.0 for h in result.hits)
