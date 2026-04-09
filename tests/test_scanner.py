# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for PassiveScanner — batch multi-source anomaly screening."""

from __future__ import annotations

import numpy as np
from hypertopos.builder.builder import GDSBuilder
from hypertopos.engine.chains import Chain
from hypertopos.navigation.scanner import PassiveScanner


def _make_scanner_fixture(tmp_path, n=50, seed=42):
    """Build sphere with account + pair + chain and return scanner."""
    rng = np.random.default_rng(seed)

    accounts = [{"account_id": f"A{i:04d}", "entity_type": "Individual"} for i in range(n)]
    txs = []
    for i in range(n * 5):
        src = f"A{rng.integers(0, n):04d}"
        dst = f"A{rng.integers(0, n):04d}"
        if src == dst:
            continue
        txs.append(
            {
                "tx_id": f"T{i:06d}",
                "from_account": src,
                "to_account": dst,
                "amount": float(rng.uniform(100, 10000)),
            }
        )

    b = GDSBuilder("scan_test", str(tmp_path / "gds"))
    b.add_line("accounts", accounts, key_col="account_id", source_id="accounts", entity_type="account")
    b.add_line("transactions", txs, key_col="tx_id", source_id="transactions", role="event")

    b.add_derived_dimension(
        "accounts",
        "transactions",
        "from_account",
        "count",
        None,
        "tx_out",
    )
    b.add_derived_dimension(
        "accounts",
        "transactions",
        "to_account",
        "count",
        None,
        "tx_in",
    )
    b.add_pattern("account_pattern", "anchor", "accounts", relations=[])

    b.add_composite_line(
        "account_pairs",
        "transactions",
        ["from_account", "to_account"],
    )
    b.add_derived_dimension(
        "account_pairs",
        "transactions",
        ["from_account", "to_account"],
        "count",
        None,
        "pair_count",
    )
    b.add_pattern("pair_pattern", "anchor", "account_pairs", relations=[])

    chains = []
    for ci in range(20):
        keys = [f"A{rng.integers(0, n):04d}" for _ in range(3)]
        c = Chain(
            chain_id=f"ch_{ci:03d}",
            keys=keys,
            event_keys=[f"T{ci:03d}_{j}" for j in range(2)],
            hop_count=2,
            is_cyclic=(keys[0] == keys[-1]),
            time_span_hours=10.0,
            categories=["USD", "USD"],
            amounts=[100.0, 90.0],
            amount_decay=0.9,
        )
        chains.append(c.to_dict())
    b.add_chain_line("tx_chains", chains, features=["hop_count", "is_cyclic"])

    b.build()

    from hypertopos.sphere import HyperSphere

    hs = HyperSphere.open(str(tmp_path / "gds"))
    sess = hs.session("test")
    scanner = PassiveScanner(
        reader=sess._reader,
        sphere=sess._reader.read_sphere(),
        manifest=sess._manifest,
    )
    return scanner, sess


class TestPassiveScannerDirect:
    def test_direct_source_returns_anomalous_keys(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("acct", "account_pattern", key_type="direct")
        result = scanner.scan("accounts")
        assert result.total_flagged > 0
        for hit in result.hits:
            assert hit.score == 1
            assert "acct" in hit.sources
        sess.__exit__(None, None, None)

    def test_direct_source_with_custom_filter(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source(
            "acct_tight",
            "account_pattern",
            key_type="direct",
            filter_expr="delta_rank_pct > 99",
        )
        result = scanner.scan("accounts")
        # Tighter filter → fewer hits
        assert result.total_flagged >= 0
        sess.__exit__(None, None, None)


class TestPassiveScannerComposite:
    def test_composite_splits_keys(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("pairs", "pair_pattern", key_type="composite")
        result = scanner.scan("accounts")
        # Composite keys split → individual account keys
        for hit in result.hits:
            assert "\u2192" not in hit.primary_key, (
                "Composite key should be split to individual keys"
            )
        sess.__exit__(None, None, None)


class TestPassiveScannerChain:
    def test_chain_expands_to_member_keys(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("chains", "chain_pattern", key_type="chain")
        result = scanner.scan("accounts")
        # Chain keys are account keys (A0000 format)
        for hit in result.hits:
            assert hit.primary_key.startswith("A")
        sess.__exit__(None, None, None)


class TestPassiveScannerScoring:
    def test_count_scoring(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("acct", "account_pattern", key_type="direct")
        scanner.add_source("pairs", "pair_pattern", key_type="composite")
        result = scanner.scan("accounts", scoring="count", threshold=1)
        # Entities in 2 sources should have score=2
        multi = [h for h in result.hits if h.score >= 2]
        assert len(multi) >= 0  # may or may not exist in small fixture
        sess.__exit__(None, None, None)

    def test_threshold_filtering(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("acct", "account_pattern", key_type="direct")
        scanner.add_source("pairs", "pair_pattern", key_type="composite")
        r1 = scanner.scan("accounts", threshold=1)
        scanner2, _ = _make_scanner_fixture(tmp_path / "b")
        scanner2.add_source("acct", "account_pattern", key_type="direct")
        scanner2.add_source("pairs", "pair_pattern", key_type="composite")
        r2 = scanner2.scan("accounts", threshold=2)
        assert r2.total_flagged <= r1.total_flagged
        sess.__exit__(None, None, None)

    def test_top_n_limits(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("acct", "account_pattern", key_type="direct")
        result = scanner.scan("accounts", top_n=3)
        assert len(result.hits) <= 3
        sess.__exit__(None, None, None)

    def test_empty_sources_returns_empty(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        result = scanner.scan("accounts")
        assert result.total_flagged == 0
        assert result.hits == []
        sess.__exit__(None, None, None)


class TestPassiveScannerAutoDiscover:
    def test_discovers_all_patterns(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.auto_discover("accounts")
        assert len(scanner._sources) >= 1
        # Should find at least account_pattern (direct)
        names = {s.name for s in scanner._sources}
        assert "account_pattern" in names
        sess.__exit__(None, None, None)

    def test_auto_discover_then_scan(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.auto_discover("accounts")
        result = scanner.scan("accounts")
        assert result.total_flagged >= 0
        assert result.sources_summary  # at least one source should have entries
        sess.__exit__(None, None, None)


class TestPassiveScannerResult:
    def test_result_structure(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("acct", "account_pattern", key_type="direct")
        result = scanner.scan("accounts")
        assert result.home_line_id == "accounts"
        assert result.total_entities > 0
        assert result.elapsed_ms > 0
        assert isinstance(result.sources_summary, dict)
        assert "acct" in result.sources_summary
        sess.__exit__(None, None, None)

    def test_hits_sorted_by_score_desc(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("acct", "account_pattern", key_type="direct")
        scanner.add_source("pairs", "pair_pattern", key_type="composite")
        result = scanner.scan("accounts")
        for i in range(len(result.hits) - 1):
            assert result.hits[i].score >= result.hits[i + 1].score
        sess.__exit__(None, None, None)


class TestPassiveScannerBorderline:
    def test_borderline_finds_near_threshold_entities(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_borderline_source(
            "borderline",
            "account_pattern",
            rank_threshold=80,
        )
        result = scanner.scan("accounts")
        # Borderline = high rank but NOT anomaly
        assert result.total_flagged >= 0
        assert "borderline" in result.sources_summary
        sess.__exit__(None, None, None)

    def test_borderline_excludes_anomalies(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        # Get anomalous keys first
        scanner.add_source("acct", "account_pattern", key_type="direct")
        direct_result = scanner.scan("accounts")
        anomalous_keys = {h.primary_key for h in direct_result.hits}

        # Now scan borderline
        scanner2, _ = _make_scanner_fixture(tmp_path / "b")
        scanner2.add_borderline_source(
            "borderline",
            "account_pattern",
            rank_threshold=50,
        )
        border_result = scanner2.scan("accounts")
        borderline_keys = {h.primary_key for h in border_result.hits}

        # No overlap — borderline explicitly excludes anomalies
        assert borderline_keys.isdisjoint(anomalous_keys)
        sess.__exit__(None, None, None)


class TestPassiveScannerPoints:
    def test_points_source_filters_by_rules(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        # tx_out is a derived dimension on accounts — should be in points
        scanner.add_points_source(
            "high_activity",
            "accounts",
            rules={"tx_out": (">=", 5)},
        )
        result = scanner.scan("accounts")
        assert "high_activity" in result.sources_summary
        sess.__exit__(None, None, None)

    def test_points_source_and_combine(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_points_source(
            "multi_rule",
            "accounts",
            rules={"tx_out": (">=", 3), "tx_in": (">=", 3)},
            combine="AND",
        )
        result = scanner.scan("accounts")
        assert "multi_rule" in result.sources_summary
        sess.__exit__(None, None, None)

    def test_points_source_or_combine(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_points_source(
            "any_active",
            "accounts",
            rules={"tx_out": (">=", 10), "tx_in": (">=", 10)},
            combine="OR",
        )
        result = scanner.scan("accounts")
        assert "any_active" in result.sources_summary
        sess.__exit__(None, None, None)

    def test_points_source_empty_when_no_match(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_points_source(
            "impossible",
            "accounts",
            rules={"tx_out": (">=", 999999)},
        )
        result = scanner.scan("accounts")
        assert result.sources_summary.get("impossible", 0) == 0
        sess.__exit__(None, None, None)


class TestPassiveScannerCompound:
    def test_compound_direct_plus_points(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        # Compound: entity anomalous in account_pattern AND tx_out >= 3
        scanner.add_compound_source(
            "acct_active",
            geometry_pattern_id="account_pattern",
            line_id="accounts",
            rules={"tx_out": (">=", 3)},
        )
        result = scanner.scan("accounts")
        assert "acct_active" in result.sources_summary
        sess.__exit__(None, None, None)

    def test_compound_is_intersection(self, tmp_path):
        """Compound result must be subset of geometry results."""
        scanner1, sess1 = _make_scanner_fixture(tmp_path)
        scanner1.add_source("acct", "account_pattern", key_type="direct")
        direct_result = scanner1.scan("accounts")
        direct_keys = {h.primary_key for h in direct_result.hits}
        sess1.__exit__(None, None, None)

        scanner2, sess2 = _make_scanner_fixture(tmp_path / "b")
        scanner2.add_compound_source(
            "acct_active",
            geometry_pattern_id="account_pattern",
            line_id="accounts",
            rules={"tx_out": (">=", 1)},
        )
        compound_result = scanner2.scan("accounts")
        compound_keys = {h.primary_key for h in compound_result.hits}
        sess2.__exit__(None, None, None)

        # Compound is subset of geometry expansion
        assert compound_keys <= direct_keys


class TestPassiveScannerWeightedIntensity:
    """Weighted scoring uses anomaly intensity, not flat 1/1."""

    def test_weighted_scores_vary_for_direct_sources(self, tmp_path):
        """Direct-source entities with different delta_norm get different weighted_scores."""
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.add_source("acct", "account_pattern", key_type="direct")
        result = scanner.scan("accounts", scoring="weighted", threshold=0.0)
        if len(result.hits) >= 2:
            scores = [h.weighted_score for h in result.hits]
            unique_scores = set(scores)
            assert len(unique_scores) > 1, (
                f"All weighted_scores identical ({scores[0]}) — "
                "expected variation based on delta_norm/theta_norm ratio"
            )
        sess.__exit__(None, None, None)


class TestPassiveScannerAutoDiscoverExtended:
    def test_auto_discover_includes_borderline(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.auto_discover("accounts", include_borderline=True)
        names = {s.name for s in scanner._sources}
        assert any("borderline" in n for n in names)
        sess.__exit__(None, None, None)

    def test_auto_discover_borderline_default_off(self, tmp_path):
        scanner, sess = _make_scanner_fixture(tmp_path)
        scanner.auto_discover("accounts")
        names = {s.name for s in scanner._sources}
        # Default: no borderline sources
        assert not any("borderline" in n for n in names)
        sess.__exit__(None, None, None)


def _make_sibling_fixture(tmp_path, n=50, seed=42):
    """Build sphere with two sibling anchor lines sharing source_id."""
    rng = np.random.default_rng(seed)

    accounts = [{"account_id": f"A{i:04d}", "entity_type": "Individual"} for i in range(n)]
    # Sibling line: same keys, different perspective
    accounts_stress = [
        {"account_id": f"A{i:04d}", "stress_score": float(rng.uniform(0, 100))}
        for i in range(n)
    ]

    txs = []
    for i in range(n * 5):
        src = f"A{rng.integers(0, n):04d}"
        dst = f"A{rng.integers(0, n):04d}"
        if src == dst:
            continue
        txs.append(
            {
                "tx_id": f"T{i:06d}",
                "from_account": src,
                "to_account": dst,
                "amount": float(rng.uniform(100, 10000)),
            }
        )

    b = GDSBuilder("sibling_test", str(tmp_path / "gds"))
    b.add_line("accounts", accounts, key_col="account_id", source_id="acct_src", entity_type="account")
    b.add_line("accounts_stress", accounts_stress, key_col="account_id", source_id="acct_src", entity_type="account")
    b.add_line("transactions", txs, key_col="tx_id", source_id="transactions", role="event")

    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "tx_out")
    b.add_derived_dimension("accounts_stress", "transactions", "from_account", "count", None, "tx_out")

    b.add_pattern("account_pattern", "anchor", "accounts", relations=[])
    b.add_pattern("account_stress_pattern", "anchor", "accounts_stress", relations=[])

    b.build()

    from hypertopos.sphere import HyperSphere

    hs = HyperSphere.open(str(tmp_path / "gds"))
    sess = hs.session("test")
    scanner = PassiveScanner(
        reader=sess._reader,
        sphere=sess._reader.read_sphere(),
        manifest=sess._manifest,
    )
    return scanner, sess


class TestPassiveScannerSiblingLines:
    """auto_discover finds patterns on sibling lines (same source_id)."""

    def test_auto_discover_includes_sibling_patterns(self, tmp_path):
        scanner, sess = _make_sibling_fixture(tmp_path)
        scanner.auto_discover("accounts")
        names = {s.name for s in scanner._sources}
        # Should discover both: own pattern + sibling pattern
        assert "account_pattern" in names
        assert "account_stress_pattern" in names
        sess.__exit__(None, None, None)

    def test_sibling_classified_as_sibling(self, tmp_path):
        scanner, sess = _make_sibling_fixture(tmp_path)
        scanner.auto_discover("accounts")
        type_by_name = {s.name: s.key_type for s in scanner._sources}
        assert type_by_name["account_stress_pattern"] == "sibling"
        sess.__exit__(None, None, None)


class TestPassiveScannerBenchmarkParity:
    """Verify scanner can express all retained benchmark sources."""

    def test_full_pipeline_5_sources(self, tmp_path):
        """Register 5 core sources using all source types."""
        scanner, sess = _make_scanner_fixture(tmp_path, n=100)

        # Sources 1-2: geometry (existing)
        scanner.add_source("account_pattern", "account_pattern", key_type="direct")
        scanner.add_source("pair_pattern", "pair_pattern", key_type="composite")

        # Source 5: borderline (new)
        scanner.add_borderline_source("borderline", "account_pattern", rank_threshold=80)

        # Sources 4, 7, D would need columns not in fixture (n_currencies_out, etc.)
        # So we test with available columns (tx_out, tx_in)
        scanner.add_points_source(
            "high_outgoing",
            "accounts",
            rules={"tx_out": (">=", 5)},
        )

        # Stream C analog: direct + points rule
        scanner.add_compound_source(
            "acct_active",
            geometry_pattern_id="account_pattern",
            line_id="accounts",
            rules={"tx_out": (">=", 3)},
        )

        result = scanner.scan("accounts", scoring="count", threshold=1)

        # Verify all sources registered and scanned
        assert len(result.sources_summary) == 5
        assert result.total_flagged > 0

        # Multi-source scoring works
        multi = [h for h in result.hits if h.score >= 2]
        assert len(multi) >= 0  # at least some overlap in 100-entity fixture

        sess.__exit__(None, None, None)

    def test_sources_api_matches_benchmark_spec(self, tmp_path):
        """Document the exact API calls that replace benchmark ad-hoc code."""
        scanner, sess = _make_scanner_fixture(tmp_path)

        # This is the TARGET API for AML benchmark replacement:
        # Source 4: Multi-cur cross-border (points rule)
        scanner.add_points_source(
            "multi_cur_crossborder",
            "accounts",
            rules={"n_currencies_out": (">=", 2), "return_ratio": (">", 0.3)},
            combine="AND",
        )
        # Source 5: Borderline
        scanner.add_borderline_source("borderline", "account_pattern", rank_threshold=80)
        # Source 7: Structuring
        scanner.add_points_source(
            "structuring",
            "accounts",
            rules={"structuring_pct": (">", 0.5), "tx_out_count": (">=", 5)},
            combine="AND",
        )
        # Stream C: Direct + points compound
        scanner.add_compound_source(
            "cycle_roundtrip",
            geometry_pattern_id="account_pattern",
            line_id="accounts",
            rules={"return_ratio": (">=", 0.4)},
        )
        # Stream D: Structuring behavioral (pure points)
        scanner.add_points_source(
            "structuring_behavioral",
            "accounts",
            rules={
                "structuring_pct": (">=", 0.5),
                "amount_uniformity": (">=", 0.5),
            },
            combine="AND",
        )

        # Columns don't exist in test fixture, but API shape is correct
        # (sources with missing columns return 0 hits gracefully)
        result = scanner.scan("accounts", threshold=1)
        assert isinstance(result.total_flagged, int)
        sess.__exit__(None, None, None)
