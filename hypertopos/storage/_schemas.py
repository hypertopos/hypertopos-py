# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Canonical Arrow schemas shared between builder, reader, and writer."""
from __future__ import annotations

import pyarrow as pa

EDGE_STRUCT_TYPE = pa.struct([
    pa.field("line_id",    pa.string()),
    pa.field("point_key",  pa.string()),
    pa.field("status",     pa.string()),
    pa.field("direction",  pa.string()),
])

GEOMETRY_SCHEMA = pa.schema([
    pa.field("primary_key",    pa.string()),
    pa.field("scale",           pa.int32()),
    pa.field("delta",           pa.list_(pa.float32())),
    pa.field("delta_norm",      pa.float32()),
    pa.field("delta_rank_pct",  pa.float32()),
    pa.field("is_anomaly",      pa.bool_()),
    pa.field("conformal_p",        pa.float32()),
    pa.field("bregman_divergence", pa.float32()),
    pa.field("anomaly_confidence", pa.float32()),
    pa.field("n_anomalous_dims",   pa.int32()),
    # Projection of per-polygon delta onto the label-aware Fisher LDA
    # direction. Populated only for patterns whose builder calibration
    # produced a ``signed_direction_vector``; all-null for patterns
    # without label-aware calibration. Sign preserved — positive values
    # mean the polygon is pushed toward the positive-labelled centroid.
    pa.field("delta_norm_signed",  pa.float32()),
    pa.field("edges",           pa.list_(EDGE_STRUCT_TYPE)),
    pa.field("entity_keys",     pa.list_(pa.string())),
    pa.field("last_refresh_at", pa.timestamp("us", tz="UTC")),
    pa.field("updated_at",      pa.timestamp("us", tz="UTC")),
])

# Minimum columns required for navigation (goto → get_polygon).
# Kept intentionally small to support legacy/test spheres that lack
# metadata columns like pattern_ver, scale, last_refresh_at.
GEOMETRY_REQUIRED_COLUMNS = {
    "primary_key", "delta", "delta_norm", "is_anomaly",
}

# Schema for event geometry (no edges column — reconstruct from entity_keys + relations)
GEOMETRY_EVENT_SCHEMA = pa.schema([
    f for f in GEOMETRY_SCHEMA if f.name != "edges"
])

EDGE_TABLE_SCHEMA = pa.schema([
    pa.field("from_key",   pa.string()),
    pa.field("to_key",     pa.string()),
    pa.field("event_key",  pa.string()),
    pa.field("timestamp",  pa.float64()),
    pa.field("amount",     pa.float64()),
])

# Per-edge derived dimensions sidecar — written at build time per event pattern
# whose YAML declares an edge_dimensions: block. Joined to the polygon table
# on event_key. Forward-compatible with the planned HopPredicate query API
# that will read per-edge predicates directly at runtime.
EDGE_FEATURES_SCHEMA = pa.schema([
    pa.field("event_key",                 pa.string()),
    pa.field("pair_edge_count",           pa.float32()),
    pa.field("position_in_chain",         pa.float32()),
    pa.field("time_since_pair_last_edge", pa.float32()),
    pa.field("pair_amount_zscore",        pa.float32()),
    pa.field("find_motif_structuring",    pa.float32()),
])
