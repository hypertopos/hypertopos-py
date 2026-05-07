"""Navigator-level integration for anomaly_propagation_in_chain.

Per-chain inspector primitive complementary to
find_chains_with_coherent_anomaly. Uses the bundled AML HI-small sphere
since it's the only sphere with a chain anchor + entity anchor with
calibrated geometry.
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
    return hs.session("propagation-test").navigator()


@pytest.fixture(scope="module")
def known_anomalous_chain_id(aml_nav):
    """Pick a chain that the population sweep flagged as having a
    coherent run AND has a UNIQUE chain_id in the points table (i.e.
    is not affected by the chain extraction's chain_id collision bug).
    """
    out = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=3,
        max_results=200,
    )
    assert out["chains"], "AML HI-small must have at least one coherent chain"
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
        "(chain extraction collision bug)",
    )


@pytest.fixture(scope="module")
def known_duplicate_chain_id(aml_nav):
    """A chain_id that the chain extraction emitted >=2 times.
    Used to verify the defensive raise."""
    line_ver = aml_nav._manifest.line_version("tx_chains") or 1
    pts = aml_nav._storage.read_points(
        "tx_chains", line_ver, columns=["primary_key"],
    )
    pks = pts["primary_key"].to_pylist()
    counts: dict[str, int] = {}
    for pk in pks:
        counts[pk] = counts.get(pk, 0) + 1
    duplicated = [pk for pk, c in counts.items() if c > 1]
    if not duplicated:
        pytest.skip("no duplicated chain_ids on this sphere")
    return duplicated[0]


def test_duplicate_chain_id_raises(aml_nav, known_duplicate_chain_id):
    """Defensive raise: when chain extraction emitted multiple chains
    with the same primary_key, the inspector cannot pick one and must
    surface the ambiguity instead of silently returning a wrong trace.
    """
    with pytest.raises(Exception, match="ambiguous chain_id"):
        aml_nav.anomaly_propagation_in_chain(
            known_duplicate_chain_id,
            "tx_chains_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_returns_dict_with_hops_and_summary(aml_nav, known_anomalous_chain_id):
    out = aml_nav.anomaly_propagation_in_chain(
        known_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    assert isinstance(out, dict)
    assert out["chain_id"] == known_anomalous_chain_id
    assert out["pattern_id"] == "tx_chains_pattern"
    assert out["anchor_pattern_id"] == "account_pattern"
    assert "hops" in out
    assert "summary" in out
    assert out["summary"]["n_hops"] > 0
    assert out["elapsed_ms"] > 0


def test_hop_fields_well_formed(aml_nav, known_anomalous_chain_id):
    out = aml_nav.anomaly_propagation_in_chain(
        known_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    for h in out["hops"]:
        assert "hop_idx" in h
        assert "primary_key" in h
        assert "is_anomaly" in h
        assert "delta_norm" in h
        assert "top_dim" in h
        assert "delta_rank_pct" in h
        assert isinstance(h["hop_idx"], int)
        assert isinstance(h["primary_key"], str)
        assert isinstance(h["is_anomaly"], bool)
        assert isinstance(h["delta_norm"], float)
        # top_dim is None when is_anomaly=False
        if h["is_anomaly"]:
            assert h["top_dim"] is not None


def test_hop_idx_sequential_starting_zero(aml_nav, known_anomalous_chain_id):
    out = aml_nav.anomaly_propagation_in_chain(
        known_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    for i, h in enumerate(out["hops"]):
        assert h["hop_idx"] == i


def test_summary_consistent_with_hops(aml_nav, known_anomalous_chain_id):
    out = aml_nav.anomaly_propagation_in_chain(
        known_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    n_anom_actual = sum(1 for h in out["hops"] if h["is_anomaly"])
    assert out["summary"]["n_anomalous"] == n_anom_actual
    assert out["summary"]["n_hops"] == len(out["hops"])
    if out["summary"]["n_anomalous"] > 0:
        assert out["summary"]["dominant_top_dim"] is not None
    assert out["summary"]["max_run_length_same_top_dim"] >= 0


def test_unknown_chain_id_raises(aml_nav):
    with pytest.raises(Exception, match="chain not found"):
        aml_nav.anomaly_propagation_in_chain(
            "CHAIN-999999999",
            "tx_chains_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_unknown_pattern_id_raises(aml_nav):
    with pytest.raises(Exception, match="pattern not found"):
        aml_nav.anomaly_propagation_in_chain(
            "CHAIN-000000",
            "nonexistent_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_unknown_anchor_pattern_raises(aml_nav):
    with pytest.raises(Exception, match="anchor pattern not found"):
        aml_nav.anomaly_propagation_in_chain(
            "CHAIN-000000",
            "tx_chains_pattern",
            anchor_pattern_id="nonexistent_anchor",
        )


def test_event_pattern_as_chain_pattern_raises(aml_nav):
    with pytest.raises(Exception, match="chain anchor"):
        aml_nav.anomaly_propagation_in_chain(
            "anything",
            "tx_pattern",
            anchor_pattern_id="account_pattern",
        )


def test_event_pattern_as_anchor_raises(aml_nav, known_anomalous_chain_id):
    with pytest.raises(Exception, match="anchor"):
        aml_nav.anomaly_propagation_in_chain(
            known_anomalous_chain_id,
            "tx_chains_pattern",
            anchor_pattern_id="tx_pattern",
        )


def test_inspector_warm_call_under_budget(aml_nav, known_anomalous_chain_id):
    """Inspector primitive on a single chain — warm call must be fast.
    Regression guard against accidental full-population scan paths."""
    import time

    # Warm anchor geometry cache
    aml_nav.anomaly_propagation_in_chain(
        known_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    t0 = time.perf_counter()
    out = aml_nav.anomaly_propagation_in_chain(
        known_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    elapsed = time.perf_counter() - t0
    # Empirical baseline on AML HI-small: cold ~35 ms, warm ~5 ms per
    # call. Budget set 10x over realistic cold to absorb CI noise; any
    # regression toward full-table reads (read_geometry(point_keys=)
    # pushdown breakdown) would blow past this.
    assert elapsed < 0.5, (
        f"warm inspector call took {elapsed * 1000:.0f} ms — "
        f"regression toward full-table read path suspected"
    )
    assert out["summary"]["n_hops"] > 0


def test_chain_with_zero_anomalous_hops(aml_nav):
    """Inspect a chain whose entities happen to be all non-anomalous —
    summary.dominant_top_dim must be None and max_run_length zero."""
    # Find a chain not in any C1 result; skip if all chains are C1-flagged.
    sweep = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=2,
        max_results=500,
    )
    flagged = {c["chain_id"] for c in sweep["chains"]}
    line_ver = aml_nav._manifest.line_version("tx_chains") or 1
    pts = aml_nav._storage.read_points(
        "tx_chains", line_ver, columns=["primary_key", "chain_keys"],
    )
    pk_counts: dict[str, int] = {}
    for pk in pts["primary_key"].to_pylist():
        pk_counts[pk] = pk_counts.get(pk, 0) + 1
    candidate = None
    for pk, ck in zip(
        pts["primary_key"].to_pylist(),
        pts["chain_keys"].to_pylist(),
        strict=False,
    ):
        if pk and pk not in flagged and ck and pk_counts[pk] == 1:
            candidate = pk
            break
    if candidate is None:
        pytest.skip(
            "no non-flagged unique-id chain available on this sphere",
        )

    out = aml_nav.anomaly_propagation_in_chain(
        candidate,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    # The candidate was picked as not-in-flagged-set, but it may still
    # have scattered anomalous hops with mixed top_dims. The contract
    # we test here is the all-non-anomalous case specifically — skip
    # if the picked chain has any anomalous hop, then assert
    # unconditionally.
    if out["summary"]["n_anomalous"] > 0:
        pytest.skip(
            f"chosen candidate has {out['summary']['n_anomalous']} "
            f"anomalous hops; zero-anomalous case not surfaced on "
            f"this sphere",
        )
    assert out["summary"]["dominant_top_dim"] is None
    assert out["summary"]["max_run_length_same_top_dim"] == 0


def test_propagation_complements_population_sweep(
    aml_nav, known_anomalous_chain_id,
):
    """C2 inspector must surface the same anomalous run that C1 sweep
    flagged for the same chain — same primitive on the same data."""
    sweep = aml_nav.find_chains_with_coherent_anomaly(
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
        min_hops=3,
        max_results=100,
    )
    swept = next(
        (c for c in sweep["chains"] if c["chain_id"] == known_anomalous_chain_id),
        None,
    )
    assert swept is not None

    inspect = aml_nav.anomaly_propagation_in_chain(
        known_anomalous_chain_id,
        "tx_chains_pattern",
        anchor_pattern_id="account_pattern",
    )
    # The run reported by C1 must match a contiguous anomalous-same-top-dim
    # span starting at run_start_idx in the C2 hop sequence.
    start = swept["run_start_idx"]
    length = swept["run_length"]
    expected_dim = swept["top_dim"]
    for i in range(start, start + length):
        assert inspect["hops"][i]["is_anomaly"] is True
        assert inspect["hops"][i]["top_dim"] == expected_dim
    # Summary's max_run must be at least the C1-reported run length.
    assert inspect["summary"]["max_run_length_same_top_dim"] >= length
