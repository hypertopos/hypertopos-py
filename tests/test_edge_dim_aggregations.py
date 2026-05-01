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
    assert len(labels) == 6, (
        f"2 relations + 0 event + 0 prop + 4 aggregated = 6, got {len(labels)}"
    )
    assert labels[-4:] == [
        "pair_edge_count_mean",
        "pair_edge_count_max",
        "find_motif_structuring_mean",
        "find_motif_structuring_max",
    ]
    assert pat.delta_dim() == 6


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
