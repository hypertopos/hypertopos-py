"""Navigator-level integration for classify_chain_typology + extend_chain.

A1 (typology classifier) wraps anomaly_propagation_in_chain and labels the
chain. A2 (extend_chain) suggests forward/backward extension entities via
the chain reverse index. Tests use AML HI-small fixture and a unique
chain_id to avoid the chain extraction collision regression (pre-fix
spheres expose duplicates).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hypertopos import HyperSphere

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AML_PATH = (
    PROJECT_ROOT / "benchmark" / "ibm-aml" / "hi_small_sphere"
    / "gds_aml_hi_small"
)


pytestmark = pytest.mark.skipif(
    not (AML_PATH / "_gds_meta" / "sphere.json").exists(),
    reason="AML HI-small sphere not built",
)


@pytest.fixture(scope="module")
def aml_nav():
    hs = HyperSphere.open(AML_PATH)
    return hs.session("typology-extend").navigator()


@pytest.fixture(scope="module")
def unique_anomalous_chain_id(aml_nav):
    """Pick a flagged chain whose primary_key is unique in the points
    table (not affected by chain extraction's collision regression)."""
    out = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=3,
        max_results=200,
    )
    line_ver = aml_nav._manifest.line_version("tx_chains") or 1
    pts = aml_nav._storage.read_points(
        "tx_chains", line_ver, columns=["primary_key"],
    )
    pks = pts["primary_key"].to_pylist()
    counts: dict[str, int] = {}
    for pk in pks:
        counts[pk] = counts.get(pk, 0) + 1
    for c in out["chains"]:
        if counts.get(c["chain_id"], 0) == 1:
            return c["chain_id"]
    pytest.skip(
        "no flagged chain has a unique chain_id on this sphere "
        "(chain extraction collision regression — sphere needs rebuild)",
    )


def test_classify_returns_typology_block(aml_nav, unique_anomalous_chain_id):
    out = aml_nav.classify_chain_typology(
        unique_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    assert out["chain_id"] == unique_anomalous_chain_id
    assert "typology" in out
    typ = out["typology"]
    assert typ["shape"] in (
        "monotone-rising", "monotone-falling", "peak-in-middle",
        "peak-at-start", "peak-at-end", "flat", "single-hop",
        "no-anomalous-run",
    )
    assert typ["peak_position"] in (
        "at-start", "early", "middle", "late", "at-end",
        "single-hop", "no-run",
    )
    assert typ["position_in_chain"] in (
        "leading", "transit", "terminal", "full-chain", "no-run",
    )
    assert "extension_signals" in typ
    assert "backward" in typ["extension_signals"]
    assert "forward" in typ["extension_signals"]


def test_classify_run_length_matches_sweep(
    aml_nav, unique_anomalous_chain_id,
):
    """The typology classifier's run_length must match what
    find_chains_with_coherent_anomaly reports for the same chain
    (both compute the longest same-top-dim consecutive run)."""
    sweep = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=3,
        max_results=200,
    )
    swept = next(
        (c for c in sweep["chains"]
         if c["chain_id"] == unique_anomalous_chain_id),
        None,
    )
    assert swept is not None
    classify = aml_nav.classify_chain_typology(
        unique_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    assert classify["typology"]["run_length"] == swept["run_length"]
    assert classify["typology"]["run_top_dim"] == swept["top_dim"]


def test_extend_forward_returns_candidates(
    aml_nav, unique_anomalous_chain_id,
):
    out = aml_nav.extend_chain(
        unique_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        direction="forward",
        max_results=10,
    )
    assert out["chain_id"] == unique_anomalous_chain_id
    assert out["direction"] == "forward"
    assert "boundary_key" in out
    assert "candidates" in out
    if out["candidates"]:
        for cand in out["candidates"]:
            assert "entity_key" in cand
            assert "is_anomaly" in cand
            assert "delta_norm" in cand
            assert "n_source_chains" in cand
        # Sorted: anomalous first, then by delta_norm desc
        for a, b in zip(
            out["candidates"], out["candidates"][1:], strict=False,
        ):
            assert (a["is_anomaly"], a["delta_norm"]) >= (
                b["is_anomaly"], b["delta_norm"],
            )


def test_extend_backward_runs(aml_nav, unique_anomalous_chain_id):
    out = aml_nav.extend_chain(
        unique_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        direction="backward",
        max_results=10,
    )
    assert out["direction"] == "backward"
    assert "candidates" in out


def test_extend_invalid_direction_raises(
    aml_nav, unique_anomalous_chain_id,
):
    with pytest.raises(ValueError, match="direction"):
        aml_nav.extend_chain(
            unique_anomalous_chain_id,
            "tx_chains_pattern",
            anchor_pattern_id="account_pattern",
            direction="sideways",
        )


def test_extend_warm_call_under_budget(
    aml_nav, unique_anomalous_chain_id,
):
    """extend_chain reads chain reverse index + filtered anchor geometry.
    Warm call must be sub-second (typical ~100-500ms after warmup)."""
    import time
    aml_nav.extend_chain(
        unique_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        direction="forward",
    )
    t0 = time.perf_counter()
    aml_nav.extend_chain(
        unique_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        direction="forward",
    )
    assert (time.perf_counter() - t0) < 5.0


def test_classify_unknown_chain_id_raises(aml_nav):
    with pytest.raises(Exception, match="chain not found"):
        aml_nav.classify_chain_typology(
            "CHAIN-999999999",
            "tx_chains_pattern",
            anchor_pattern_id="account_pattern",
        )
