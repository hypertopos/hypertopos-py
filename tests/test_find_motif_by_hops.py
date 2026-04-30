"""Navigator-level integration for find_motif_by_hops.

Uses the bundled AML HI-small sphere (event tx_pattern with edge_table +
edge_dimensions sidecar) since Berka's tx_pattern lacks an explicit
edge_table block.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypertopos import HopPredicate, HyperSphere


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
    return hs.session("hops-test").navigator()


def test_returns_dict_with_motifs_block(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=5,
    )
    assert isinstance(out, dict)
    assert "motifs" in out
    assert "n_results" in out
    assert out["pattern_id"] == "tx_pattern"


def test_anchor_pattern_raises(aml_nav):
    with pytest.raises(Exception, match="event pattern"):
        aml_nav.find_motif_by_hops(
            "account_pattern", hops=[HopPredicate()],
        )


def test_unknown_pattern_raises(aml_nav):
    with pytest.raises(Exception, match="pattern not found"):
        aml_nav.find_motif_by_hops(
            "nonexistent", hops=[HopPredicate()],
        )


def test_empty_hops_raises(aml_nav):
    with pytest.raises(Exception, match="hops"):
        aml_nav.find_motif_by_hops("tx_pattern", hops=[])


def test_too_many_hops_raises(aml_nav):
    with pytest.raises(Exception, match="hop count"):
        aml_nav.find_motif_by_hops(
            "tx_pattern", hops=[HopPredicate()] * 7,
        )


def test_invalid_max_results_raises(aml_nav):
    with pytest.raises(Exception, match="max_results"):
        aml_nav.find_motif_by_hops(
            "tx_pattern", hops=[HopPredicate()], max_results=0,
        )


def test_edge_dim_predicate_filter_runs(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(
            amount_min=10000.0,
            edge_dim_predicates={"pair_edge_count": (">=", 5.0)},
        )],
        max_results=3,
        score=False,
    )
    assert isinstance(out["motifs"], list)
    for m in out["motifs"]:
        assert m["dim_values_per_hop"][0]["pair_edge_count"] >= 5.0


def test_score_false_omits_score_field(aml_nav):
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=3,
        score=False,
    )
    if out["motifs"]:
        assert "score" not in out["motifs"][0]


def test_score_true_on_event_pattern_skips_silently(aml_nav):
    # tx_pattern is an event pattern — _score_motif_from_edges requires
    # anchor-pattern geometry. The navigator silently skips scoring
    # rather than raising; instances come back without a score field.
    out = aml_nav.find_motif_by_hops(
        "tx_pattern",
        hops=[HopPredicate(amount_min=10000.0)],
        max_results=3,
        score=True,
    )
    if out["motifs"]:
        # Either no score field or all instances unscored.
        assert all(
            "score" not in m or m["score"] == 0.0 for m in out["motifs"]
        )


def test_unknown_dim_raises(aml_nav):
    with pytest.raises(Exception, match="unknown dims"):
        aml_nav.find_motif_by_hops(
            "tx_pattern",
            hops=[HopPredicate(
                edge_dim_predicates={"nonexistent_dim": (">=", 1.0)},
            )],
        )
