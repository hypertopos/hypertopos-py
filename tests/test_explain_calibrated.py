from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest
from test_builder_robust import DIMS, _builder

from hypertopos.sphere import HyperSphere


@pytest.fixture(scope="module")
def sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("explain_calibrated") / "sphere"
    _builder(out).build()
    return out


def _ranked_keys(sphere: Path) -> list[str]:
    geo = lance.dataset(str(sphere / "geometry" / "account_pattern" / "data.lance")).to_table(
        columns=["primary_key", "conformal_p"]
    )
    return geo.sort_by([("conformal_p", "ascending")])["primary_key"].to_pylist()


def _explain(sphere: Path, key: str) -> dict:
    session = HyperSphere.open(str(sphere)).session(agent_id="explain_calibrated")
    try:
        return session.navigator().explain_anomaly(key, "account_pattern")
    finally:
        session.close()


def test_top_anomaly_explanation_sums_to_its_evidence(sphere: Path) -> None:
    out = _explain(sphere, _ranked_keys(sphere)[0])
    assert out["severity"] in ("extreme", "high", "medium", "low")
    assert out["bits"] == pytest.approx(-np.log2(out["conformal_p"]), abs=1e-3)
    assert sum(out["axes"][a]["bits"] for a in out["axes"]) == pytest.approx(out["bits"], abs=1e-3)
    dims = {d["dim"] for d in out["top_dimensions"]}
    assert dims <= set(DIMS)
    mahal_bits = sum(d["bits"] for d in out["top_dimensions"])
    assert mahal_bits == pytest.approx(out["axes"]["mahal"]["bits"], abs=1e-3)
    assert "theta_norm" not in out  # theta is not the boundary on a calibrated pattern


def test_ordinary_entity_is_normal(sphere: Path) -> None:
    out = _explain(sphere, _ranked_keys(sphere)[-1])
    assert out["severity"] == "normal"
    assert 0.0 < out["conformal_p"] <= 1.0


def test_tiny_conformal_p_is_not_rounded_away() -> None:
    from hypertopos.engine.investigation import build_calibrated_explanation

    out = build_calibrated_explanation(
        conformal_p=7.765769e-06,
        anomaly_score=0.99981523,
        alpha=0.05,
        axis_ranks={"mahal": 0.9998},
        axis_bits={"mahal": 16.97},
        dim_bits={"a": 16.97},
        score_delta={"a": 4.0},
        delta_norm=3.0,
    )
    assert out["conformal_p"] == 7.765769e-06
    assert out["anomaly_score"] == 0.99981523
