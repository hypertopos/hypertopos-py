# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""combine_anomaly_pvalues scores every entity over the same detector family.

A detector that cannot score an entity (an account without graph neighbours)
counts as no evidence (p = 1), so it neither rewards nor penalises the entity.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_builder_robust import _builder

from hypertopos.sphere import HyperSphere

PID = "account_pattern"


@pytest.fixture(scope="module")
def nav(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    out: Path = tmp_path_factory.mktemp("hmp_family") / "sphere"
    _builder(out).build()
    return HyperSphere.open(str(out)).session("t").navigator()


def test_entity_a_detector_cannot_score_does_not_outrank_its_twin(nav, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    a, b = "A001", "A002"
    monkeypatch.setattr(nav, "_collect_segment_shift_p", lambda pid: {a: 0.02, b: 0.02})
    monkeypatch.setattr(nav, "_collect_neighbor_contamination_p", lambda pid, geo: {a: 1.0})
    out = nav.combine_anomaly_pvalues(
        PID, detectors=("neighbor_contamination", "segment_shift"), sample_size=None, top_n=None
    )
    hmp = {e["primary_key"]: e["hmp"] for e in out}
    assert hmp[a] == pytest.approx(hmp[b])
    # p_per_detector still lists only the detectors that produced a value.
    by_key = {e["primary_key"]: e for e in out}
    assert set(by_key[b]["p_per_detector"]) == {"segment_shift"}


def test_a_detector_silent_for_everyone_leaves_the_family(nav, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(nav, "_collect_segment_shift_p", lambda pid: {"A001": 0.02})
    monkeypatch.setattr(nav, "_collect_neighbor_contamination_p", lambda pid, geo: {})
    out = nav.combine_anomaly_pvalues(
        PID, detectors=("neighbor_contamination", "segment_shift"), sample_size=None, top_n=None
    )
    (entry,) = [e for e in out if e["primary_key"] == "A001"]
    assert entry["hmp"] == pytest.approx(0.02)
