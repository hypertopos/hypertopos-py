from __future__ import annotations

import pytest

from hypertopos.builder.mapping import (
    EdgeDimAggregationsConfig,
    parse_edge_dim_aggregations,
)


def test_parse_edge_dim_aggregations_omitted_dims_rejected():
    """The previous 'dims=None means aggregate everything' shorthand was
    a latent build-vs-runtime inconsistency (builder resolved to avail,
    runtime model lost track) — explicit declaration is now required."""
    with pytest.raises(ValueError, match="dims must be a non-empty list"):
        parse_edge_dim_aggregations(
            {"from": "tx_pattern"}, pattern_type="anchor",
        )


def test_parse_edge_dim_aggregations_empty_dims_rejected():
    with pytest.raises(ValueError, match="non-empty list"):
        parse_edge_dim_aggregations(
            {"from": "tx_pattern", "dims": []}, pattern_type="anchor",
        )


def test_parse_edge_dim_aggregations_explicit_dims():
    cfg = parse_edge_dim_aggregations(
        {
            "from": "tx_pattern",
            "dims": ["pair_edge_count", "find_motif_structuring"],
        },
        pattern_type="anchor",
    )
    assert cfg.dims == ("pair_edge_count", "find_motif_structuring")


def test_parse_edge_dim_aggregations_rejects_event():
    with pytest.raises(ValueError, match="only supported on anchor patterns"):
        parse_edge_dim_aggregations(
            {"from": "tx_pattern"}, pattern_type="event",
        )


def test_parse_edge_dim_aggregations_requires_from():
    with pytest.raises(ValueError, match="must specify 'from"):
        parse_edge_dim_aggregations({}, pattern_type="anchor")


def test_parse_edge_dim_aggregations_unknown_dim():
    with pytest.raises(ValueError, match="unknown edge dimension"):
        parse_edge_dim_aggregations(
            {"from": "tx_pattern", "dims": ["bogus_dim"]},
            pattern_type="anchor",
        )


def test_parse_edge_dim_aggregations_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        parse_edge_dim_aggregations(
            ["tx_pattern"], pattern_type="anchor",  # type: ignore[arg-type]
        )


def test_parse_edge_dim_aggregations_rejects_non_list_dims():
    with pytest.raises(ValueError, match=r"\.dims must be a list"):
        parse_edge_dim_aggregations(
            {"from": "tx_pattern", "dims": "pair_edge_count"},
            pattern_type="anchor",
        )


def test_pattern_dim_labels_extends_with_aggregated_dim_names():
    """Pattern.dim_labels must include aggregated dim names so consumers
    that key into delta vectors (anomaly_summary, find_clusters dim_profile,
    explain_anomaly) can label the new dims correctly. Closes the F1.d
    follow-up and the broadcast bug in anomaly_summary on patterns with
    edge_dim_aggregations declared."""
    from datetime import datetime
    import numpy as np
    from hypertopos.model.sphere import (
        EdgeDimAggregationsRef, Pattern, RelationDef,
    )
    pat = Pattern(
        pattern_id="account_pattern",
        entity_type="accounts",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="_d_tx_out_count", direction="out", required=False),
            RelationDef(line_id="_d_sum_out", direction="out", required=False),
        ],
        mu=np.zeros(2),
        sigma_diag=np.ones(2),
        theta=np.ones(2),
        population_size=100,
        computed_at=datetime.now(),
        version=1,
        status="production",
        edge_dim_aggregations=EdgeDimAggregationsRef(
            from_event_pattern="tx_pattern",
            dims=("pair_edge_count", "find_motif_structuring"),
        ),
    )
    labels = pat.dim_labels
    # 2 relations + 0 event + 0 prop + 2 source dims × 5 aggregates = 12
    assert len(labels) == 12, (
        f"2 relations + 0 event + 0 prop + 10 aggregated = 12, got {len(labels)}"
    )
    assert labels[-10:] == [
        "pair_edge_count_mean",
        "pair_edge_count_max",
        "pair_edge_count_std",
        "pair_edge_count_p95",
        "pair_edge_count_count_above_threshold",
        "find_motif_structuring_mean",
        "find_motif_structuring_max",
        "find_motif_structuring_std",
        "find_motif_structuring_p95",
        "find_motif_structuring_count_above_threshold",
    ]
    assert pat.delta_dim() == 12


def test_pattern_dim_labels_no_aggregations_unchanged():
    """Pattern without edge_dim_aggregations keeps the original
    dim_labels / delta_dim shape — backward compat."""
    from datetime import datetime
    import numpy as np
    from hypertopos.model.sphere import Pattern, RelationDef
    pat = Pattern(
        pattern_id="account_pattern",
        entity_type="accounts",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="_d_tx_out_count", direction="out", required=False),
            RelationDef(line_id="_d_sum_out", direction="out", required=False),
        ],
        mu=np.zeros(2),
        sigma_diag=np.ones(2),
        theta=np.ones(2),
        population_size=100,
        computed_at=datetime.now(),
        version=1,
        status="production",
    )
    assert pat.dim_labels == ["_d_tx_out_count", "_d_sum_out"]
    assert pat.delta_dim() == 2


def test_pattern_dim_labels_aggregations_with_empty_dims():
    """Defensive: the Pattern model (low-level dataclass) tolerates an
    EdgeDimAggregationsRef with dims=None — returns base labels without
    crashing. Parse-level rejection (parse_edge_dim_aggregations)
    enforces the user-facing contract; this guards against in-memory
    construction during tests / loaded-from-older-sphere paths."""
    from datetime import datetime
    import numpy as np
    from hypertopos.model.sphere import (
        EdgeDimAggregationsRef, Pattern, RelationDef,
    )
    pat = Pattern(
        pattern_id="account_pattern",
        entity_type="accounts",
        pattern_type="anchor",
        relations=[RelationDef(line_id="_d_tx_out_count", direction="out", required=False)],
        mu=np.zeros(1),
        sigma_diag=np.ones(1),
        theta=np.ones(1),
        population_size=100,
        computed_at=datetime.now(),
        version=1,
        status="production",
        edge_dim_aggregations=EdgeDimAggregationsRef(
            from_event_pattern="tx_pattern",
            dims=None,
        ),
    )
    assert pat.dim_labels == ["_d_tx_out_count"]
    assert pat.delta_dim() == 1


# --- per-dim aggregate selector (mapping form) ---


def test_parse_aggregations_list_form_expands_to_all_five_per_dim():
    """Form A — list of dim names — materialises to all five canonical
    aggregates per dim. Back-compat with the historical YAML shape that
    predates the per-dim subset selector."""
    from hypertopos.engine.edge_features import AGGREGATE_NAMES

    cfg = parse_edge_dim_aggregations(
        {
            "from": "tx_pattern",
            "dims": ["pair_edge_count", "find_motif_structuring"],
        },
        pattern_type="anchor",
    )
    assert cfg.dims == ("pair_edge_count", "find_motif_structuring")
    assert cfg.aggregates_per_dim == {
        "pair_edge_count": tuple(AGGREGATE_NAMES),
        "find_motif_structuring": tuple(AGGREGATE_NAMES),
    }


def test_parse_aggregations_mapping_form_per_dim_subset():
    """Form B — mapping with per-dim agg subsets — preserves keys in YAML
    order (drives polygon-dim layout) and aggregate values in canonical
    AGGREGATE_NAMES order regardless of user-list order."""
    cfg = parse_edge_dim_aggregations(
        {
            "from": "tx_pattern",
            "dims": {
                "pair_edge_count": ["count_above_threshold"],
                "find_motif_structuring": ["max", "mean"],   # reversed user input
            },
        },
        pattern_type="anchor",
    )
    assert cfg.dims == ("pair_edge_count", "find_motif_structuring")
    # user wrote [max, mean] but canonical order materialises as (mean, max)
    assert cfg.aggregates_per_dim == {
        "pair_edge_count": ("count_above_threshold",),
        "find_motif_structuring": ("mean", "max"),
    }


def test_parse_aggregations_canonical_order_insensitive_to_user_input():
    """Schema-hash stability check: two YAML shapes that differ only in the
    user-supplied aggregate order produce identical aggregates_per_dim, so
    `dimension_kinds` and `schema_hash` do not flip between cosmetic edits."""
    cfg_a = parse_edge_dim_aggregations(
        {
            "from": "tx_pattern",
            "dims": {"pair_edge_count": ["mean", "p95", "max"]},
        },
        pattern_type="anchor",
    )
    cfg_b = parse_edge_dim_aggregations(
        {
            "from": "tx_pattern",
            "dims": {"pair_edge_count": ["max", "mean", "p95"]},
        },
        pattern_type="anchor",
    )
    assert cfg_a.aggregates_per_dim == cfg_b.aggregates_per_dim


def test_parse_aggregations_unknown_agg_name_raises():
    with pytest.raises(ValueError, match="unknown aggregate 'median'"):
        parse_edge_dim_aggregations(
            {
                "from": "tx_pattern",
                "dims": {"pair_edge_count": ["mean", "median"]},
            },
            pattern_type="anchor",
        )


def test_parse_aggregations_empty_per_dim_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        parse_edge_dim_aggregations(
            {
                "from": "tx_pattern",
                "dims": {"pair_edge_count": []},
            },
            pattern_type="anchor",
        )


def test_parse_aggregations_empty_mapping_raises():
    with pytest.raises(ValueError, match="non-empty"):
        parse_edge_dim_aggregations(
            {"from": "tx_pattern", "dims": {}},
            pattern_type="anchor",
        )


def test_parse_aggregations_unknown_dim_in_mapping_raises():
    with pytest.raises(ValueError, match="unknown edge dimension"):
        parse_edge_dim_aggregations(
            {
                "from": "tx_pattern",
                "dims": {"definitely_not_a_dim": ["mean"]},
            },
            pattern_type="anchor",
        )


def test_parse_aggregations_per_dim_value_must_be_list():
    with pytest.raises(ValueError, match="must be a list of"):
        parse_edge_dim_aggregations(
            {
                "from": "tx_pattern",
                "dims": {"pair_edge_count": "mean"},
            },
            pattern_type="anchor",
        )


def test_parse_aggregations_dims_neither_list_nor_mapping_raises():
    with pytest.raises(ValueError, match="must be a list or mapping"):
        parse_edge_dim_aggregations(
            {"from": "tx_pattern", "dims": 7},
            pattern_type="anchor",
        )


def test_edge_dim_aggregations_config_post_init_materialises_default():
    """Direct constructor without `aggregates_per_dim` materialises the
    all-five canonical default at __post_init__ time, so engine-level
    dispatch never sees an empty mapping. Locks the contract that
    `EdgeDimAggregationsConfig(from_event_pattern=..., dims=...)` works
    for direct callers (tests, downstream pipelines)."""
    from hypertopos.engine.edge_features import AGGREGATE_NAMES

    cfg = EdgeDimAggregationsConfig(
        from_event_pattern="tx_pattern",
        dims=("pair_edge_count", "find_motif_structuring"),
    )
    assert cfg.aggregates_per_dim == {
        "pair_edge_count": tuple(AGGREGATE_NAMES),
        "find_motif_structuring": tuple(AGGREGATE_NAMES),
    }


def test_edge_dim_aggregations_ref_post_init_materialises_default():
    """The runtime model `EdgeDimAggregationsRef` (built by the reader)
    materialises the same default in `__post_init__`, so direct
    construction and reader paths converge to one source of truth."""
    from hypertopos.engine.edge_features import AGGREGATE_NAMES
    from hypertopos.model.sphere import EdgeDimAggregationsRef

    ref = EdgeDimAggregationsRef(
        from_event_pattern="tx_pattern",
        dims=("pair_edge_count",),
    )
    assert ref.aggregates_per_dim == {
        "pair_edge_count": tuple(AGGREGATE_NAMES),
    }
