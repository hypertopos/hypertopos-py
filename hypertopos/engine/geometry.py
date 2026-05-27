# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from hypertopos.model.objects import Edge, Polygon, Solid, SolidSlice
from hypertopos.model.sphere import (
    CalibrationFit,
    DimensionDecomposition,
    IntrinsicExtrinsicReport,
)
from hypertopos.navigation.navigator import GDSNavigationError
from hypertopos.utils.arrow import delta_matrix_from_arrow

if TYPE_CHECKING:
    from hypertopos.model.manifest import Manifest
    from hypertopos.model.sphere import Pattern, RelationDef
    from hypertopos.storage.cache import GDSCache
    from hypertopos.storage.reader import GDSReader


def _reconstruct_edges_from_entity_keys(
    entity_keys: list[str] | None,
    relations: list[RelationDef],
) -> list[Edge]:
    """Reconstruct Edge objects from entity_keys + pattern.relations.

    entity_keys[i] corresponds to relations[i].
    Dead edge: entity_keys[i] == "" (empty string) or index out of range.
    """
    keys = entity_keys or []
    edges: list[Edge] = []
    for i, rel in enumerate(relations):
        key = keys[i] if i < len(keys) else ""
        alive = bool(key)
        edges.append(Edge(
            line_id=rel.line_id,
            point_key=key,
            status="alive" if alive else "dead",
            direction=rel.direction,
            is_jumpable=alive,
        ))
    return edges


class GDSEngine:
    def __init__(self, storage: GDSReader | None, cache: GDSCache | None) -> None:
        self._storage = storage
        self._cache = cache

    def build_polygon(
        self,
        primary_key: str,
        pattern_id: str,
        manifest: Manifest,
    ) -> Polygon:
        cached = self._cache.get_polygon(primary_key, pattern_id)
        if cached is not None:
            return cached

        version = manifest.pattern_version(pattern_id)
        if version is None:
            raise GDSNavigationError(
                f"No geometry version for pattern '{pattern_id}' in manifest."
            )
        _poly_cols = [
            "primary_key", "scale", "delta", "delta_norm", "delta_rank_pct",
            "is_anomaly", "last_refresh_at", "updated_at",
            "edges", "entity_keys",
        ]
        table = self._storage.read_geometry(
            pattern_id, version, primary_key=primary_key, columns=_poly_cols,
        )
        if table.num_rows == 0:
            raise KeyError(f"No geometry for {primary_key} in {pattern_id} v{version}")

        row = {col: table[col][0].as_py() for col in table.schema.names}

        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]

        if row.get("edges"):
            edges = [
                Edge(
                    line_id=e["line_id"],
                    point_key=e["point_key"],
                    status=e["status"],
                    direction=e["direction"],
                    is_jumpable=bool(e["point_key"]),
                )
                for e in row["edges"]
            ]
        else:
            # Event geometry: reconstruct from entity_keys + pattern.relations
            edges = _reconstruct_edges_from_entity_keys(
                row.get("entity_keys"), pattern.relations,
            )

        polygon = Polygon(
            primary_key=row["primary_key"],
            pattern_id=row.get("pattern_id", pattern_id),
            pattern_ver=row.get("pattern_ver", version),
            pattern_type=row.get("pattern_type", pattern.pattern_type),
            scale=row["scale"],
            delta=np.array(row["delta"], dtype=np.float32),
            delta_norm=float(row["delta_norm"]),
            is_anomaly=bool(row["is_anomaly"]),
            edges=edges,
            last_refresh_at=row["last_refresh_at"],
            updated_at=row["updated_at"],
            delta_rank_pct=(
                None if row.get("delta_rank_pct") is None
                else float(row["delta_rank_pct"])
            ),
        )
        if pattern.edge_max is None:
            # Discrete mode: recompute delta from edges + prop_fill (prop values may have changed).
            prop_fill: np.ndarray | None = None
            if pattern.prop_columns:
                entity_line_id = sphere.entity_line(pattern_id)
                if entity_line_id:
                    line_ver = manifest.line_versions.get(entity_line_id, 1)
                    pts = self._storage.read_points(
                        entity_line_id, line_ver, primary_key=primary_key
                    )
                    prop_fill = self._prop_fill_vector(pts, pattern.prop_columns)
            polygon.delta = self.compute_delta(polygon, pattern, prop_fill=prop_fill)
            polygon.delta_norm = float(np.linalg.norm(polygon.delta))
            theta_norm = float(np.linalg.norm(pattern.theta))
            polygon.is_anomaly = theta_norm > 0.0 and polygon.delta_norm >= theta_norm
        # Continuous mode (edge_max set): stored delta is ground truth.
        # _polygon_to_shape_vector yields alive_count=1 for every entity (single edge,
        # point_key="") — recomputation would produce identical delta for all entities.

        self._cache.put_polygon(polygon)
        return polygon

    SIGMA_EPSILON = 1e-2

    @staticmethod
    def _prop_fill_vector(pts_table: Any, prop_columns: list[str]) -> np.ndarray:
        """Return float32 fill indicator vector for tracked property columns.

        1.0 = property is present (non-null), 0.0 = null/missing.
        Returns zeros if pts_table is empty (entity not found in points layer).
        """
        v = np.zeros(len(prop_columns), dtype=np.float32)
        if pts_table is None or pts_table.num_rows == 0:
            return v
        schema_names = set(pts_table.schema.names)
        row = {col: pts_table[col][0].as_py() for col in prop_columns if col in schema_names}
        for i, prop in enumerate(prop_columns):
            v[i] = 0.0 if row.get(prop) is None else 1.0
        return v

    def compute_delta(
        self,
        polygon: Polygon,
        pattern: Pattern,
        prop_fill: np.ndarray | None = None,
    ) -> np.ndarray:
        shape_vector = self._polygon_to_shape_vector(polygon, pattern)
        if prop_fill is not None and len(prop_fill) > 0:
            shape_vector = np.concatenate([shape_vector, prop_fill])
        if pattern.cholesky_inv is not None:
            delta = pattern.cholesky_inv @ (shape_vector - pattern.mu)
        else:
            sigma = np.maximum(pattern.sigma_diag, self.SIGMA_EPSILON)
            delta = (shape_vector - pattern.mu) / sigma
        if pattern.dimension_weights is not None:
            delta = delta * pattern.dimension_weights
        return delta

    @staticmethod
    def decompose_displacement(
        delta: np.ndarray,
        label_direction: np.ndarray,
    ) -> dict[str, float]:
        """Decompose a single delta vector along the label-discriminating axis.

        Splits ``delta`` into two scalar magnitudes:

        - ``intrinsic`` = ``|delta . label_direction_unit|`` — magnitude of
          the projection onto the unit-norm label direction. Captures how
          far the entity moved along the axis that separates labelled
          classes.
        - ``extrinsic`` = ``sqrt(||delta||^2 - intrinsic^2)`` — magnitude of
          the residual component orthogonal to the label axis.

        Identity: ``intrinsic^2 + extrinsic^2 == ||delta||^2`` to floating
        precision. ``label_direction`` is normalised internally — callers
        may pass any non-zero vector.

        This is the label-axis decomposition of a single polygon's delta
        vector. It is NOT the calibration-drift intrinsic/extrinsic
        decomposition (see ``_compute_intrinsic_extrinsic_decomposition``)
        which splits drift between two calibration epochs into a
        within-entity-shape component vs a coordinate-system component —
        same vocabulary, different math.

        Args:
            delta: 1-D delta vector for one polygon.
            label_direction: 1-D vector of the same length, must be
                non-zero. Direction sense is irrelevant (``intrinsic``
                is absolute-valued).

        Returns:
            ``{"intrinsic": float, "extrinsic": float}``. Both
            non-negative. When ``label_direction`` is the zero vector
            both values are 0.0 (no axis to project onto, no decomposition
            defined).
        """
        d = np.asarray(delta, dtype=np.float64).ravel()
        ld = np.asarray(label_direction, dtype=np.float64).ravel()
        if d.shape != ld.shape:
            raise ValueError(
                f"delta and label_direction shape mismatch: "
                f"{d.shape} vs {ld.shape}",
            )
        ld_norm = float(np.linalg.norm(ld))
        if ld_norm <= 0.0:
            return {"intrinsic": 0.0, "extrinsic": 0.0}
        ld_unit = ld / ld_norm
        intrinsic = float(abs(np.dot(d, ld_unit)))
        d_sq = float(np.dot(d, d))
        residual_sq = max(d_sq - intrinsic * intrinsic, 0.0)
        extrinsic = float(np.sqrt(residual_sq))
        return {"intrinsic": intrinsic, "extrinsic": extrinsic}

    def _polygon_to_shape_vector(
        self, polygon: Polygon, pattern: Pattern
    ) -> np.ndarray:
        vector = np.zeros(len(pattern.relations), dtype=np.float32)
        for i, relation in enumerate(pattern.relations):
            alive_count = sum(
                1 for e in polygon.edges
                if e.line_id == relation.line_id and e.is_alive()
            )
            if pattern.edge_max is not None:
                max_val = pattern.edge_max[i] if pattern.edge_max[i] > 0 else 1.0
                vector[i] = alive_count / max_val
            else:
                vector[i] = 1.0 if alive_count > 0 else 0.0
        return vector

    def build_solid(
        self,
        primary_key: str,
        pattern_id: str,
        manifest: Manifest,
        filters: dict[str, str | list[str]] | None = None,
        timestamp: datetime | None = None,
        counterfactual_frozen_population: bool = False,
    ) -> Solid:
        base = self.build_polygon(primary_key, pattern_id, manifest)
        table = self._storage.read_temporal(
            pattern_id, primary_key, filters=filters, agent_id=manifest.agent_id,
        )
        if table.num_rows > 0 and "shape_snapshot" not in table.schema.names:
            raise GDSNavigationError(
                f"Temporal data for pattern '{pattern_id}' uses legacy schema "
                "(delta_snapshot). Run GDSWriter.migrate_temporal_to_shape_snapshot() "
                "to upgrade."
            )
        sphere = self._storage.read_sphere()
        pattern = sphere.patterns[pattern_id]
        sigma = np.maximum(pattern.sigma_diag, self.SIGMA_EPSILON)
        slices: list[SolidSlice] = []
        # When counterfactual_frozen_population is requested, capture each
        # slice's raw shape_snapshot so the frozen-trajectory recompute can
        # run after the loop using the chronologically-first slice's shape
        # as the entity-relative reference.
        raw_shapes: list[np.ndarray] = []
        for i in range(table.num_rows):
            row = {col: table[col][i].as_py() for col in table.schema.names}
            shape = np.array(row["shape_snapshot"], dtype=np.float32)
            # Temporal shape_snapshot carries structural dims only; aggregated
            # edge_dim dims have no temporal history. Slice calibration arrays
            # to the snapshot width before the broadcast.
            _w = shape.shape[-1]
            if pattern.cholesky_inv is not None:
                delta = pattern.cholesky_inv[:_w, :_w] @ (shape - pattern.mu[:_w])
            else:
                delta = (shape - pattern.mu[:_w]) / sigma[:_w]
            if pattern.dimension_weights is not None:
                delta = delta * pattern.dimension_weights[:_w]
            delta_norm = float(np.linalg.norm(delta))
            slices.append(SolidSlice(
                slice_index=row["slice_index"],
                timestamp=row["timestamp"],
                deformation_type=row["deformation_type"],
                delta_snapshot=delta,
                delta_norm_snapshot=delta_norm,
                pattern_ver=row["pattern_ver"],
                changed_property=row.get("changed_property"),
                changed_line_id=row.get("changed_line_id"),
                added_edge=None,
            ))
            if counterfactual_frozen_population:
                raw_shapes.append(shape)
        # Pair raw_shapes with their slice and sort jointly so the
        # chronologically-first slice's raw shape becomes the frozen
        # reference even when temporal rows arrive out of order.
        if counterfactual_frozen_population and slices:
            paired = sorted(
                zip(slices, raw_shapes, strict=True),
                key=lambda p: p[0].timestamp,
            )
            slices = [p[0] for p in paired]
            raw_shapes = [p[1] for p in paired]
        else:
            slices.sort(key=lambda s: s.timestamp)
        if timestamp is not None:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            if counterfactual_frozen_population:
                surviving = [
                    (s, r) for s, r in zip(slices, raw_shapes, strict=True)
                    if s.timestamp <= timestamp
                ]
                slices = [p[0] for p in surviving]
                raw_shapes = [p[1] for p in surviving]
            else:
                slices = [s for s in slices if s.timestamp <= timestamp]
        if counterfactual_frozen_population and slices:
            from hypertopos.engine.counterfactual import (
                recompute_delta_norm_against_frozen,
            )
            mu_frozen = raw_shapes[0]
            _w = mu_frozen.shape[-1]
            sigma_frozen = sigma[:_w]
            for slc, raw in zip(slices, raw_shapes, strict=True):
                slc.delta_norm_frozen_pop = recompute_delta_norm_against_frozen(
                    shape=raw,
                    mu_frozen=mu_frozen,
                    sigma=sigma_frozen,
                )
        return Solid(primary_key=primary_key, pattern_id=pattern_id,
                     base_polygon=base, slices=slices)

    def compute_distance_temporal(self, solid_a: Solid, solid_b: Solid) -> float:
        seq_a = [s.delta_snapshot for s in solid_a.slices]
        seq_b = [s.delta_snapshot for s in solid_b.slices]
        if not seq_a or not seq_b:
            return 0.0
        return self._dtw(seq_a, seq_b)

    def _dtw(
        self, seq_a: list[np.ndarray], seq_b: list[np.ndarray]
    ) -> float:
        n, m = len(seq_a), len(seq_b)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        return float(dtw[n, m])

    def _find_common_polygons(
        self, polygon_a: Polygon, polygon_b: Polygon
    ) -> set[tuple[str, str]]:
        """Return alive (line_id, point_key) pairs common to both polygons.

        Uses set intersection — O(|B(Φ_A)| + |B(Φ_B)|).
        """
        alive_a = {
            (e.line_id, e.point_key) for e in polygon_a.edges if e.is_alive()
        }
        alive_b = {
            (e.line_id, e.point_key) for e in polygon_b.edges if e.is_alive()
        }
        return alive_a & alive_b

    def classify_anomalies(
        self, polygons: list[Polygon], pattern: Pattern
    ) -> list[dict]:
        clusters: dict[tuple, list[Polygon]] = {}
        for p in polygons:
            if not p.is_anomaly:
                continue
            key = tuple(np.round(p.delta, 1).tolist())
            clusters.setdefault(key, []).append(p)
        result = []
        k = len(pattern.relations)

        def _dim_name(i: int) -> str:
            if i < k:
                rel = pattern.relations[i]
                return rel.display_name or rel.line_id
            prop_idx = i - k
            if prop_idx < len(pattern.prop_columns):
                return f"prop:{pattern.prop_columns[prop_idx]}"
            return f"dim_{i}"

        for delta_key, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
            dims = sorted(
                [(_dim_name(i), v) for i, v in enumerate(delta_key) if abs(v) > 0.05],
                key=lambda x: -abs(x[1]),
            )
            elevated = [name for name, v in dims if v > 0]
            missing = [name for name, v in dims if v < 0]
            # Put the dominant-sign group first (determined by the top driver)
            if dims and dims[0][1] > 0:
                parts = (
                    ([f"elevated: {', '.join(elevated[:3])}"] if elevated else [])
                    + ([f"missing: {', '.join(missing[:3])}"] if missing else [])
                )
            else:
                parts = (
                    ([f"missing: {', '.join(missing[:3])}"] if missing else [])
                    + ([f"elevated: {', '.join(elevated[:3])}"] if elevated else [])
                )
            label = "; ".join(parts) if parts else "no deviation"
            result.append({
                "delta": list(delta_key),
                "label": label,
                "count": len(members),
                "examples": [p.primary_key for p in members[:3]],
            })
        return result

    def count_inside_alias(self, alias: Any, geo: Any) -> int:
        """Count entities geometrically inside the alias segment (signed_dist > 0).

        geo: Arrow table with a 'delta' column (list<float32>).
        Returns 0 for empty geometry or missing cutting plane.
        """
        if geo.num_rows == 0:
            return 0
        cp = alias.filter.cutting_plane
        if cp is None:
            return 0
        delta_matrix = delta_matrix_from_arrow(geo)
        scores = cp.signed_distances_batch(delta_matrix)
        return int((scores > 0).sum())

    _DELTA_OPS: dict[str, Any] = {
        "gt": np.greater, "gte": np.greater_equal,
        "lt": np.less, "lte": np.less_equal, "eq": np.equal,
    }

    def filter_geometry_inside_alias(self, geo: Any, alias: Any) -> Any:
        """Filter geometry Arrow table to rows geometrically inside the alias segment.

        Keeps rows where signed_distance(delta, cutting_plane) > 0.
        Returns geo unchanged if alias has no cutting_plane or geo is empty.
        """
        import pyarrow as pa
        if geo.num_rows == 0:
            return geo
        cp = alias.filter.cutting_plane
        if cp is None:
            return geo
        delta_matrix = delta_matrix_from_arrow(geo)
        inside_mask = cp.signed_distances_batch(delta_matrix) > 0
        return geo.filter(pa.array(inside_mask))

    def filter_geometry_by_delta_dim(
        self, geo: Any, pattern: Any, delta_dim_spec: dict
    ) -> Any:
        """Filter geometry Arrow table by per-dimension delta values.

        delta_dim_spec: {"dim_name": {"gt": 0.5}, "other_dim": {"lt": -0.2}}
        Multiple dimensions combine with AND semantics.
        Raises ValueError for unknown dimension names or operators.
        """
        import pyarrow as pa
        if geo.num_rows == 0:
            return geo
        delta_matrix = delta_matrix_from_arrow(geo)
        mask = np.ones(delta_matrix.shape[0], dtype=bool)
        for dim_name, predicates in delta_dim_spec.items():
            idx = pattern.dim_index(dim_name)  # raises ValueError if unknown
            dim_values = delta_matrix[:, idx]
            for op_name, threshold in predicates.items():
                if op_name not in self._DELTA_OPS:
                    raise ValueError(
                        f"Unknown comparison op '{op_name}'. Supported: {list(self._DELTA_OPS)}"
                    )
                mask &= self._DELTA_OPS[op_name](dim_values, float(threshold))
        return geo.filter(pa.array(mask))

    @staticmethod
    def contrast_populations(
        delta_matrix: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        dim_labels: list[str] | None = None,
    ) -> list[dict]:
        """Compute per-dimension contrast between two entity groups.

        Returns dimensions ranked by |effect_size| descending, answering
        "why are these two groups different?".

        Parameters
        ----------
        delta_matrix:
            Shape (N, D) — delta vectors for the full population.
        mask_a, mask_b:
            Boolean arrays of length N selecting group A and group B.
        dim_labels:
            Optional semantic labels per dimension. Falls back to "dim_i".

        Returns
        -------
        List of dicts (one per dimension) sorted by |effect_size| descending.
        Keys: dim_index, dim_label, mean_a, mean_b, diff, effect_size.
        """
        if not np.any(mask_a):
            raise ValueError("group_a is empty — no entities selected by mask_a")
        if not np.any(mask_b):
            raise ValueError("group_b is empty — no entities selected by mask_b")

        group_a = delta_matrix[mask_a].astype(np.float64)
        group_b = delta_matrix[mask_b].astype(np.float64)

        mean_a = group_a.mean(axis=0)
        mean_b = group_b.mean(axis=0)
        diff = mean_a - mean_b

        sigma_a = group_a.std(axis=0)
        sigma_b = group_b.std(axis=0)
        pooled = np.sqrt((sigma_a ** 2 + sigma_b ** 2) / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            effect_size = np.where(pooled > 0, diff / pooled, diff)

        n_dims = delta_matrix.shape[1]
        results = []
        for i in range(n_dims):
            label = dim_labels[i] if dim_labels and i < len(dim_labels) else f"dim_{i}"
            results.append({
                "dim_index": i,
                "dim_label": label,
                "mean_a": round(float(mean_a[i]), 6),
                "mean_b": round(float(mean_b[i]), 6),
                "diff": round(float(diff[i]), 6),
                "effect_size": round(float(effect_size[i]), 6),
            })
        results.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
        return results

    def compute_centroid_map(
        self,
        delta_matrix: np.ndarray,
        group_labels: list[str],
        dim_labels: list[str] | None = None,
        entity_keys: list[str] | None = None,
        max_representatives: int = 3,
    ) -> dict:
        """Compute global + per-group centroids from delta matrix.

        Args:
            delta_matrix: (N, D) array of delta vectors.
            group_labels: per-row group label (length N). Entries may not be None.
            dim_labels: optional dimension names (length D).
            entity_keys: optional list of business keys parallel to delta_matrix rows.
                When provided, each group centroid includes ``member_samples`` — a
                reservoir sample of up to ``max_representatives`` entity keys drawn
                uniformly from that group's actual members.
            max_representatives: reservoir size for member_samples (default 3).

        Returns:
            Dict with global_centroid, group_centroids, inter_centroid_distances,
            structural_outlier, dimensions.
        """
        if delta_matrix.shape[0] == 0:
            raise ValueError("Cannot compute centroid map from empty delta matrix.")

        n, d = delta_matrix.shape
        norms = np.sqrt(np.einsum('ij,ij->i', delta_matrix, delta_matrix))

        # Global centroid
        global_vec = delta_matrix.mean(axis=0)
        global_centroid = {
            "vector": global_vec.tolist(),
            "radius": round(float(norms.mean()), 6),
            "spread": round(float(norms.std()), 6),
            "count": n,
        }

        # Reservoir sampling: accumulate member_samples per group in a single pass
        # before the per-label mask loop (O(N) rather than O(N×k)).
        group_reservoirs: dict[str, list[str]] = {}
        group_counts: dict[str, int] = {}
        if entity_keys is not None:
            for _idx, (label, key) in enumerate(zip(group_labels, entity_keys, strict=False)):
                count = group_counts.get(label, 0) + 1
                group_counts[label] = count
                reservoir = group_reservoirs.setdefault(label, [])
                if len(reservoir) < max_representatives:
                    reservoir.append(key)
                else:
                    j = random.randint(0, count - 1)  # noqa: S311
                    if j < max_representatives:
                        reservoir[j] = key

        # Per-group centroids
        unique_labels = sorted(set(group_labels))
        group_centroids = []
        centroid_vectors: dict[str, np.ndarray] = {}

        # Pre-convert to numpy array for vectorized comparison (O(N) once, not O(N×k))
        group_arr = np.array(group_labels, dtype=object)

        for label in unique_labels:
            mask = group_arr == label
            group_deltas = delta_matrix[mask]
            group_norms = norms[mask]
            vec = group_deltas.mean(axis=0)
            dist_to_global = float(np.linalg.norm(vec - global_vec))
            centroid_vectors[label] = vec
            entry: dict = {
                "key": label,
                "vector": vec.tolist(),
                "radius": round(float(group_norms.mean()), 6),
                "spread": round(float(group_norms.std()), 6),
                "count": int(mask.sum()),
                "distance_to_global": round(dist_to_global, 6),
            }
            if label in group_reservoirs:
                entry["member_samples"] = group_reservoirs[label]
            group_centroids.append(entry)

        # Inter-centroid pairwise L2 distances
        inter_distances = []
        for i, la in enumerate(unique_labels):
            for lb in unique_labels[i + 1:]:
                dist = float(np.linalg.norm(centroid_vectors[la] - centroid_vectors[lb]))
                inter_distances.append({
                    "pair": [la, lb],
                    "distance": round(dist, 6),
                })

        # Structural outlier = group with max distance_to_global
        outlier = max(group_centroids, key=lambda g: g["distance_to_global"])

        return {
            "global_centroid": global_centroid,
            "group_centroids": group_centroids,
            "inter_centroid_distances": inter_distances,
            "structural_outlier": {
                "key": outlier["key"],
                "distance_to_global": outlier["distance_to_global"],
            },
            "dimensions": dim_labels or [f"dim_{i}" for i in range(d)],
        }

    def find_nearest(
        self,
        ref_delta: np.ndarray,
        pattern_id: str,
        version: int,
        top_n: int = 5,
        exclude_keys: set[str] | None = None,
        filter_expr: str | None = None,
        dim_mask_indices: list[int] | None = None,
        metric: str = "L2",
    ) -> list[tuple[str, float]]:
        """Find top-n nearest entities. Uses Lance ANN index when available.

        filter_expr: optional Lance SQL predicate passed to ANN.
        dim_mask_indices: compute distance only on these dimension indices.
        metric: "L2" (Euclidean) or "cosine" (1 - cos_sim).
        """
        need_rerank = dim_mask_indices is not None or metric != "L2"

        # Fast path: Lance IVF_FLAT ANN (L2 only, full vector)
        _ann_fn = getattr(self._storage, "find_nearest_lance", None)
        if _ann_fn is not None and not need_rerank:
            ann = _ann_fn(pattern_id, version, ref_delta, top_n, exclude_keys, filter_expr)
            if ann is not None:
                return ann

        # Over-fetch from ANN when we need to re-rank
        if _ann_fn is not None and need_rerank:
            ann = _ann_fn(
                pattern_id, version, ref_delta, top_n * 5,
                exclude_keys, filter_expr,
            )
            if ann is not None:
                ann_keys = {k for k, _ in ann}
                geo = self._storage.read_geometry(
                    pattern_id, version,
                    columns=["primary_key", "delta"],
                    point_keys=list(ann_keys),
                )
                keys = geo["primary_key"].to_pylist()
                deltas = delta_matrix_from_arrow(geo)
                distances = self._compute_distances(
                    ref_delta, deltas, dim_mask_indices, metric,
                )
                if exclude_keys:
                    mask = np.array([k in exclude_keys for k in keys])
                    distances[mask] = np.inf
                return self._top_n_from_distances(keys, distances, top_n)

        # Fallback: brute-force NumPy
        geo = self._storage.read_geometry(
            pattern_id, version, columns=["primary_key", "delta"],
        )
        keys = geo["primary_key"].to_pylist()
        deltas = delta_matrix_from_arrow(geo)

        distances = self._compute_distances(
            ref_delta, deltas, dim_mask_indices, metric,
        )

        if exclude_keys:
            mask = np.array([k in exclude_keys for k in keys])
            distances[mask] = np.inf

        return self._top_n_from_distances(keys, distances, top_n)

    @staticmethod
    def _compute_distances(
        ref: np.ndarray,
        candidates: np.ndarray,
        dim_mask_indices: list[int] | None,
        metric: str,
    ) -> np.ndarray:
        ref_f = ref.astype(np.float32)
        cand = candidates

        if dim_mask_indices is not None:
            idx = np.array(dim_mask_indices, dtype=np.intp)
            ref_f = ref_f[idx]
            cand = cand[:, idx]

        if metric == "cosine":
            ref_norm = np.linalg.norm(ref_f)
            cand_norms = np.linalg.norm(cand, axis=1)
            safe_denom = np.maximum(ref_norm * cand_norms, 1e-10)
            cos_sim = cand @ ref_f / safe_denom
            return 1.0 - cos_sim

        # L2 (default)
        diff = cand - ref_f
        return np.sqrt(np.einsum('ij,ij->i', diff, diff))

    @staticmethod
    def _top_n_from_distances(
        keys: list[str],
        distances: np.ndarray,
        top_n: int,
    ) -> list[tuple[str, float]]:
        finite_count = int(np.sum(np.isfinite(distances)))
        n = min(top_n, finite_count)
        if n == 0:
            return []
        if n >= len(distances):
            top_indices = np.argsort(distances)[:n]
        else:
            top_indices = np.argpartition(distances, n)[:n]
            top_indices = top_indices[np.argsort(distances[top_indices])]
        return [(keys[i], float(distances[i])) for i in top_indices]

    def find_clusters(
        self,
        delta_matrix: np.ndarray,
        keys: list[str],
        is_anomaly_flags: list[bool],
        delta_norms: list[float],
        n_clusters: int,
        dim_names: list[str],
        seed: int = 42,
    ) -> list[dict]:
        """Cluster entities by delta vector shape using k-means++.

        Returns list of cluster dicts sorted by size descending. Each dict:
        cluster_id, size, anomaly_rate, centroid_delta, delta_norm_mean,
        delta_norm_std, representative_key, dim_profile, member_keys.
        """
        N = delta_matrix.shape[0]
        if N == 0:
            return []

        # Auto-k: n_clusters=0 triggers silhouette search
        if n_clusters == 0:
            auto = self.find_optimal_k(delta_matrix, seed=seed)
            n_clusters = auto["best_k"]

        labels, centroids = self._kmeans(delta_matrix, n_clusters, seed=seed)

        result: list[dict] = []
        for k in range(centroids.shape[0]):
            mask = labels == k
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            members = delta_matrix[indices]
            centroid = centroids[k]

            # Sort by distance to centroid (closest first → representative)
            _mc_diff = members - centroid
            dists_to_centroid = np.sqrt(np.einsum('ij,ij->i', _mc_diff, _mc_diff))
            sorted_order = np.argsort(dists_to_centroid)
            sorted_indices = indices[sorted_order]

            anomaly_flags = [is_anomaly_flags[i] for i in sorted_indices]
            dn = [delta_norms[i] for i in sorted_indices]

            result.append({
                "cluster_id": int(k),
                "size": int(len(sorted_indices)),
                "anomaly_rate": float(sum(anomaly_flags) / len(anomaly_flags)),
                "centroid_delta": centroid.tolist(),
                "delta_norm_mean": float(np.mean(dn)),
                "delta_norm_std": float(np.std(dn)),
                "representative_key": str(keys[sorted_indices[0]]),
                "dim_profile": [
                    {"dimension": name, "centroid_value": float(centroid[i])}
                    for i, name in enumerate(dim_names)
                ],
                "member_keys": [str(keys[i]) for i in sorted_indices],
            })

        # Sort by size descending; renumber cluster_id after sort
        result.sort(key=lambda c: c["size"], reverse=True)
        for rank, cluster in enumerate(result):
            cluster["cluster_id"] = rank

        return result

    def find_optimal_k(
        self,
        delta_matrix: np.ndarray,
        k_max: int = 15,
        seed: int = 42,
    ) -> dict:
        """Find optimal cluster count via silhouette search over k=2..k_max.

        Subsamples to 5000 for silhouette computation (O(N^2)).
        Returns dict: best_k, silhouette_per_k, best_silhouette, gap.
        """
        N = delta_matrix.shape[0]
        k_max = min(k_max, N - 1, int(N**0.5), 15)
        if k_max < 2:
            return {
                "best_k": 1,
                "silhouette_per_k": {},
                "best_silhouette": 0.0,
                "gap": 0.0,
            }

        # Subsample for silhouette (O(N^2))
        MAX_SIL_SAMPLE = 5000
        if N > MAX_SIL_SAMPLE:
            rng = np.random.default_rng(seed)
            idx = rng.choice(N, MAX_SIL_SAMPLE, replace=False)
            sil_matrix = delta_matrix[idx]
        else:
            sil_matrix = delta_matrix

        sil_n = sil_matrix.shape[0]
        # Precompute pairwise distances once
        sq_norms = np.sum(sil_matrix**2, axis=1)
        dist_sq = (
            sq_norms[:, None] + sq_norms[None, :] - 2 * sil_matrix @ sil_matrix.T
        )
        np.maximum(dist_sq, 0, out=dist_sq)
        dist_matrix = np.sqrt(dist_sq)

        silhouette_per_k: dict[int, float] = {}
        for k in range(2, k_max + 1):
            labels, _ = self._kmeans(sil_matrix, k, seed=seed)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                silhouette_per_k[k] = 0.0
                continue

            scores = np.zeros(sil_n, dtype=np.float64)
            for i in range(sil_n):
                own_label = labels[i]
                own_mask = labels == own_label
                own_mask[i] = False
                if own_mask.sum() == 0:
                    continue
                a_i = dist_matrix[i, own_mask].mean()
                b_i = np.inf
                for lbl in unique_labels:
                    if lbl == own_label:
                        continue
                    other_mask = labels == lbl
                    if other_mask.sum() == 0:
                        continue
                    b_i = min(b_i, dist_matrix[i, other_mask].mean())
                if b_i < np.inf:
                    scores[i] = (b_i - a_i) / max(a_i, b_i)

            silhouette_per_k[k] = round(float(scores.mean()), 6)

        if not silhouette_per_k:
            return {
                "best_k": 1,
                "silhouette_per_k": {},
                "best_silhouette": 0.0,
                "silhouette_margin": 0.0,
            }

        best_k = max(silhouette_per_k, key=silhouette_per_k.get)  # type: ignore[arg-type]
        best_sil = silhouette_per_k[best_k]
        next_k = best_k + 1
        margin = best_sil - silhouette_per_k.get(next_k, best_sil)

        return {
            "best_k": best_k,
            "silhouette_per_k": silhouette_per_k,
            "best_silhouette": round(best_sil, 6),
            "silhouette_margin": round(margin, 6),
        }

    @staticmethod
    def _kmeans(
        delta_matrix: np.ndarray,
        n_clusters: int,
        max_iter: int = 100,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pure numpy k-means++. Returns (labels, centroids) shapes (N,) and (K, D)."""
        N = delta_matrix.shape[0]
        n_clusters = max(1, min(n_clusters, N))
        rng = np.random.default_rng(seed)

        # k-means++ initialisation with running min-distance (O(N) memory)
        def _sq_norms(d: np.ndarray) -> np.ndarray:
            return np.einsum('ij,ij->i', d, d)

        first = int(rng.integers(0, N))
        centroid_list: list[np.ndarray] = [delta_matrix[first]]
        min_sq_dists = _sq_norms(delta_matrix - centroid_list[0])
        for _ in range(n_clusters - 1):
            total = min_sq_dists.sum()
            probs = min_sq_dists / total if total > 0 else np.ones(N) / N
            idx = int(rng.choice(N, p=probs))
            centroid_list.append(delta_matrix[idx])
            new_sq = _sq_norms(delta_matrix - centroid_list[-1])
            min_sq_dists = np.minimum(min_sq_dists, new_sq)

        centroids = np.vstack(centroid_list).astype(np.float32)
        labels = np.zeros(N, dtype=np.int32)

        for _ in range(max_iter):
            # ||x - c||^2 = ||x||^2 - 2*x·c^T + ||c||^2  →  (N,K) not (N,K,D)
            x_sq = np.einsum('ij,ij->i', delta_matrix, delta_matrix)  # (N,)
            c_sq = np.einsum('ij,ij->i', centroids, centroids)        # (K,)
            cross = delta_matrix @ centroids.T                         # (N, K)
            sq_dists = x_sq[:, None] - 2 * cross + c_sq[None, :]      # (N, K)
            new_labels = np.argmin(sq_dists, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                members = delta_matrix[labels == k]
                if len(members) > 0:
                    centroids[k] = members.mean(axis=0)

        # Final centroid update ensures correctness when convergence happens on first iteration
        # (e.g. n_clusters=1): labels were updated but centroids not yet recomputed.
        for k in range(n_clusters):
            members = delta_matrix[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)

        return labels, centroids

    # ------------------------------------------------------------------
    # geometry_to_polygons — reconstruct Polygon objects from Arrow table
    # ------------------------------------------------------------------

    def geometry_to_polygons(
        self,
        geo: Any,
        norm_lookup: dict[str, float] | None = None,
        top_n: int | None = None,
        pattern: Pattern | None = None,
        *,
        pattern_id: str = "",
        pattern_type: str = "",
        pattern_ver: int = 0,
    ) -> list[Polygon]:
        """Reconstruct Polygon objects from a geometry Arrow table.

        Handles the ``edges`` column (``list<struct>``) encoding.
        When ``edges`` is absent (event geometry), reconstructs from
        ``entity_keys`` + ``pattern.relations``.

        Parameters
        ----------
        geo:
            Arrow table with geometry columns (primary_key, delta, …).
        norm_lookup:
            Optional mapping ``primary_key → delta_norm``.  When a key is
            absent, ``np.linalg.norm(delta)`` is computed on the fly.
        top_n:
            If given, return only the *top_n* polygons with the highest
            ``delta_norm`` (list is always sorted descending).
        pattern:
            Pattern object for edge reconstruction when edges column is absent.
        pattern_id:
            Injected pattern_id for Polygon construction (used when column
            is absent from geometry table).
        pattern_type:
            Injected pattern_type (``"anchor"`` or ``"event"``).
        pattern_ver:
            Injected pattern version.
        """
        if norm_lookup is None:
            norm_lookup = {}

        has_edges_col = "edges" in geo.schema.names
        has_entity_keys_col = "entity_keys" in geo.schema.names
        relations = pattern.relations if pattern else []

        results: list[Polygon] = []
        for i in range(geo.num_rows):
            row = {col: geo[col][i].as_py() for col in geo.schema.names}
            pk = row["primary_key"]
            delta_norm = norm_lookup.get(
                pk,
                float(np.linalg.norm(np.array(row["delta"], dtype=np.float32))),
            )

            # Edge decoding: from edges struct column, or reconstruct from entity_keys
            if has_edges_col and row.get("edges"):
                edges = [
                    Edge(
                        line_id=e["line_id"],
                        point_key=e["point_key"],
                        status=e["status"],
                        direction=e["direction"],
                        is_jumpable=bool(e["point_key"]),
                    )
                    for e in row["edges"]
                ]
            elif has_entity_keys_col and relations:
                edges = _reconstruct_edges_from_entity_keys(
                    row.get("entity_keys"), relations,
                )
            else:
                edges = []

            results.append(Polygon(
                primary_key=pk,
                pattern_id=row.get("pattern_id", pattern_id),
                pattern_ver=row.get("pattern_ver", pattern_ver),
                pattern_type=row.get("pattern_type", pattern_type),
                scale=row["scale"],
                delta=np.array(row["delta"], dtype=np.float32),
                delta_norm=delta_norm,
                is_anomaly=bool(row["is_anomaly"]),
                edges=edges,
                last_refresh_at=row["last_refresh_at"],
                updated_at=row["updated_at"],
                delta_rank_pct=(
                    None if row.get("delta_rank_pct") is None
                    else float(row["delta_rank_pct"])
                ),
                bregman_divergence=(
                    None if row.get("bregman_divergence") is None
                    else float(row["bregman_divergence"])
                ),
                anomaly_confidence=(
                    None if row.get("anomaly_confidence") is None
                    else float(row["anomaly_confidence"])
                ),
            ))

        results.sort(key=lambda p: (-p.delta_norm, p.primary_key))
        if top_n is not None:
            return results[:top_n]
        return results

    # ------------------------------------------------------------------
    # anomaly_dimensions — squared contribution ranking
    # ------------------------------------------------------------------

    @staticmethod
    def anomaly_dimensions(
        delta: list[float] | np.ndarray,
        dim_labels: list[str],
        top_n: int = 3,
    ) -> list[dict]:
        """Top-N dimensions by squared contribution to delta_norm.

        Returns list of ``{dim, label, delta, contribution_pct}`` sorted by
        contribution descending.  Dimensions contributing < 5 % are excluded.
        """
        delta_arr = np.array(delta, dtype=np.float32)
        sq = delta_arr ** 2
        total = sq.sum()
        if total < 1e-10:
            return []
        contributions = sq / total * 100
        top_idx = np.argsort(contributions)[::-1][:top_n]
        return [
            {
                "dim": int(i),
                "label": dim_labels[i] if i < len(dim_labels) else f"dim_{i}",
                "delta": round(float(delta_arr[i]), 4),
                "contribution_pct": round(float(contributions[i]), 1),
            }
            for i in top_idx if contributions[i] > 5.0
        ]

    @staticmethod
    def witness_jaccard(set_a: set[str], set_b: set[str]) -> float:
        """Jaccard index over two witness dimension label sets.

        Returns 0.0 if both sets are empty (no signal), otherwise
        ``|A ∩ B| / |A ∪ B|`` ∈ [0, 1].
        """
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def trajectory_cosine(
        traj_a: np.ndarray | list[float],
        traj_b: np.ndarray | list[float],
    ) -> float:
        """Cosine similarity remapped to [0, 1].

        Identical trajectories return 1.0, opposite return 0.0, orthogonal
        return 0.5. Zero-norm vectors yield 0.5 (neutral, no signal).
        """
        a = np.asarray(traj_a, dtype=np.float64)
        b = np.asarray(traj_b, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.5
        cos = float(np.dot(a, b) / (norm_a * norm_b))
        cos = max(-1.0, min(1.0, cos))
        return (cos + 1.0) / 2.0

    @staticmethod
    def composite_link_score(
        delta_similarity: float,
        witness_overlap: float,
        trajectory_alignment: float | None,
        anomaly_bonus: float,
        weights: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Blend four signals into a single score in [0, 1].

        When ``trajectory_alignment`` is None, the trajectory weight is
        redistributed proportionally across the remaining components, so
        the final score remains in [0, 1] regardless of which signals are
        present.

        Returns ``(score, components)`` where ``components`` is the
        per-signal weighted contribution.
        """
        w_d = float(weights.get("delta", 0.0))
        w_w = float(weights.get("witness", 0.0))
        w_t = float(weights.get("trajectory", 0.0))
        w_a = float(weights.get("anomaly", 0.0))

        if trajectory_alignment is None:
            # Redistribute trajectory weight across the rest, proportional
            # to their original share.
            remaining_total = w_d + w_w + w_a
            if remaining_total == 0.0:
                return 0.0, {}
            w_d_n = w_d / remaining_total
            w_w_n = w_w / remaining_total
            w_a_n = w_a / remaining_total
            components = {
                "delta": w_d_n * float(delta_similarity),
                "witness": w_w_n * float(witness_overlap),
                "anomaly": w_a_n * float(anomaly_bonus),
            }
        else:
            components = {
                "delta": w_d * float(delta_similarity),
                "witness": w_w * float(witness_overlap),
                "trajectory": w_t * float(trajectory_alignment),
                "anomaly": w_a * float(anomaly_bonus),
            }

        score = sum(components.values())
        return float(score), components

    @staticmethod
    def witness_set(
        delta: list[float] | np.ndarray,
        theta_norm: float,
        dim_labels: list[str],
    ) -> dict:
        """Minimal subset of dimensions that certifies the anomaly.

        Greedy: add dimensions in order of |delta[d]|^2 until partial norm > theta_norm.
        """
        delta = np.asarray(delta, dtype=np.float64)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= theta_norm or theta_norm <= 0:
            return {"witness_size": 0, "witness_dims": [], "delta_norm": round(delta_norm, 4)}
        sq = delta ** 2
        order = np.argsort(sq)[::-1]
        cumsum = np.cumsum(sq[order])
        k = int(np.searchsorted(cumsum, theta_norm ** 2, side="left")) + 1
        k = min(k, len(delta))
        dims = []
        for i in range(k):
            idx = int(order[i])
            label = dim_labels[idx] if idx < len(dim_labels) else f"dim_{idx}"
            dims.append({"dim": idx, "label": label, "delta_value": round(float(delta[idx]), 4)})
        return {
            "witness_size": k,
            "witness_dims": dims,
            "delta_norm": round(delta_norm, 4),
        }

    @staticmethod
    def anti_witness(
        delta: list[float] | np.ndarray,
        theta_norm: float,
        dim_labels: list[str],
    ) -> dict:
        """Minimal subset of dimensions to zero-out to make entity non-anomalous.

        Greedy: remove dimensions in order of |delta[d]|^2 until residual norm < theta_norm.
        """
        delta = np.asarray(delta, dtype=np.float64)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= theta_norm:
            return {"repair_size": 0, "repair_dims": [], "residual_norm": round(delta_norm, 4)}
        sq = delta ** 2
        order = np.argsort(sq)[::-1]
        total_sq = float(np.sum(sq))
        removed_sq = 0.0
        k = 0
        for i in range(len(delta)):
            removed_sq += sq[order[i]]
            k += 1
            residual = total_sq - removed_sq
            if residual < theta_norm ** 2:
                break
        dims = []
        for i in range(k):
            idx = int(order[i])
            label = dim_labels[idx] if idx < len(dim_labels) else f"dim_{idx}"
            dims.append({"dim": idx, "label": label, "delta_value": round(float(delta[idx]), 4)})
        residual_norm = float(np.sqrt(max(0.0, total_sq - removed_sq)))
        return {
            "repair_size": k,
            "repair_dims": dims,
            "residual_norm": round(residual_norm, 4),
        }

    @staticmethod
    def compute_reputation(
        delta_norms: np.ndarray,
        theta_norm: float,
    ) -> dict:
        """Beta distribution reputation from longitudinal anomaly history.

        reputation = (alpha + 1) / (alpha + beta + 2) — Bayesian posterior mean
        with Laplace smoothing (uniform prior: Beta(1,1)).
        anomaly_tenure = longest consecutive anomalous streak anywhere in history.
        """
        if len(delta_norms) == 0:
            return {"alpha": 0, "beta": 0, "reputation": 0.5, "anomaly_tenure": 0}
        is_anom = delta_norms >= theta_norm
        alpha = int(np.sum(is_anom))
        beta = len(delta_norms) - alpha
        reputation = round((alpha + 1) / (alpha + beta + 2), 4)
        # Anomaly tenure: longest consecutive anomalous streak anywhere
        tenure = 0
        current_streak = 0
        for a in is_anom:
            if a:
                current_streak += 1
                if current_streak > tenure:
                    tenure = current_streak
            else:
                current_streak = 0
        return {
            "alpha": alpha,
            "beta": beta,
            "reputation": reputation,
            "anomaly_tenure": tenure,
        }


_M3_SIGMA_SAFE_FLOOR = 1e-12


def _per_dim_anomaly_contributions(
    delta: np.ndarray,
    *,
    dimension_kinds: list[str] | None,
    sigma: np.ndarray | None,
    mu: np.ndarray | None,
    dimension_weights: np.ndarray | None,
) -> np.ndarray:
    """Per-dim anomaly contribution vector — matches build_explanation routing.

    When ``dimension_kinds + sigma + mu`` are all available, returns the
    Bregman divergence per dim (the canonical mixed-family attribution used
    by ``explain_anomaly.top_dimensions``). Falls back to ``delta**2`` —
    matches ``GDSEngine.anomaly_dimensions`` and ``witness_set`` — when the
    calibration trio is missing or shapes mismatch.

    Output is non-negative and same length as ``delta``. Reliability-flag
    routing and any future per-dim consumer should call this helper so that
    ``reliability_flags.dominant_dim`` and ``explain_anomaly`` agree on the
    same polygon.
    """
    d = np.asarray(delta, dtype=np.float64)
    if (
        dimension_kinds is not None
        and sigma is not None
        and mu is not None
        and len(dimension_kinds) == len(d)
        and len(sigma) == len(d)
        and len(mu) == len(d)
    ):
        from hypertopos.builder._bregman import bregman_divergence
        if dimension_weights is not None:
            w = np.maximum(
                np.asarray(dimension_weights, dtype=np.float64), 1e-9
            )
            d_unw = d / w
        else:
            d_unw = d
        shape = d_unw * np.asarray(sigma, dtype=np.float64) + np.asarray(
            mu, dtype=np.float64
        )
        return bregman_divergence(
            shape,
            np.asarray(mu, dtype=np.float64),
            np.asarray(sigma, dtype=np.float64),
            dimension_kinds,
        )
    return d ** 2


def compute_reliability_flags(
    delta: np.ndarray | list[float],
    *,
    pattern: Pattern,
    anomaly_confidence: float | None = None,
    dominant_dim_threshold: float = 0.7,
    confidence_threshold: float = 0.5,
) -> dict:
    """Surface per-polygon reliability warnings.

    Two flags fire independently:

    - ``single_dim_driven`` — the dominant dimension contributes more than
      ``dominant_dim_threshold`` of total anomaly attribution.
      Investigator action: a single-dim-driven anomaly is more likely to be
      a data-quality artefact (a saturated counter, an outlier on one
      property) than a multi-dim fraud signal. Worth a sanity check on the
      dominant dim before opening a case.
    - ``low_confidence_bucket`` — ``anomaly_confidence`` is set and below
      ``confidence_threshold``. ``anomaly_confidence`` is the
      bootstrap-derived fraction of resamples in which the entity exceeded
      its (fresh-mu, fresh-sigma, fresh-theta) anomaly threshold; a value
      well below 0.5 means the anomaly flag is fragile to population
      resampling and the entity is borderline.

    Takes the raw ``delta`` vector + scalar ``anomaly_confidence`` (matches
    the ``anomaly_dimensions`` / ``witness_set`` calling convention rather
    than requiring a full ``Polygon`` — callers that have only Arrow rows
    skip the ``Polygon`` construction). Returns a dict with both boolean
    flags, the dominant dim label and share, the sanitised confidence
    value, and ``flags`` — the list of triggered flag names (subset of
    ``{"single_dim_driven", "low_confidence_bucket"}``). ``confidence``
    sanitises ``NaN`` / ``±inf`` to ``None`` per the strict-JSON
    convention. The dominant-dim attribution routes through the same
    per-dim contribution primitive (``_per_dim_anomaly_contributions``)
    that ``explain_anomaly.top_dimensions`` uses, so the two surfaces
    always pick the same dim for the same polygon.
    """
    delta_arr = np.asarray(delta, dtype=np.float64)
    dim_labels = pattern.dim_labels
    contributions = _per_dim_anomaly_contributions(
        delta_arr,
        dimension_kinds=pattern.dimension_kinds,
        sigma=pattern.sigma_diag,
        mu=pattern.mu,
        dimension_weights=pattern.dimension_weights,
    )
    total = float(contributions.sum())
    if total > 0:
        dom_idx = int(np.argmax(contributions))
        dom_share = float(contributions[dom_idx]) / total
        dom_label = (
            dim_labels[dom_idx] if 0 <= dom_idx < len(dim_labels) else None
        )
    else:
        dom_idx = -1
        dom_share = 0.0
        dom_label = None

    single_dim_driven = bool(dom_share > dominant_dim_threshold)

    if anomaly_confidence is None or not np.isfinite(float(anomaly_confidence)):
        conf: float | None = None
    else:
        conf = float(anomaly_confidence)
    low_confidence_bucket = bool(conf is not None and conf < confidence_threshold)

    flags: list[str] = []
    if single_dim_driven:
        flags.append("single_dim_driven")
    if low_confidence_bucket:
        flags.append("low_confidence_bucket")

    return {
        "single_dim_driven": single_dim_driven,
        "dominant_dim": dom_label,
        "dominant_dim_share": round(dom_share, 4),
        "low_confidence_bucket": low_confidence_bucket,
        "confidence": None if conf is None else round(conf, 4),
        "flags": flags,
    }


def _compute_decomposition_vectors(
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    fit_v1: CalibrationFit,
    fit_v2: CalibrationFit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (total, intrinsic, extrinsic) decomposition vectors. Pure math."""
    s_a = shape_a.astype(np.float64)
    s_b = shape_b.astype(np.float64)
    mu_v1 = fit_v1.mu.astype(np.float64)
    mu_v2 = fit_v2.mu.astype(np.float64)
    sigma_v1 = fit_v1.sigma_diag.astype(np.float64)
    sigma_v2 = fit_v2.sigma_diag.astype(np.float64)

    sigma_v1_safe = np.where(sigma_v1 > _M3_SIGMA_SAFE_FLOOR, sigma_v1, 1.0)
    sigma_v2_safe = np.where(sigma_v2 > _M3_SIGMA_SAFE_FLOOR, sigma_v2, 1.0)

    delta_a = (s_a - mu_v1) / sigma_v1_safe
    delta_b = (s_b - mu_v2) / sigma_v2_safe
    total = delta_b - delta_a
    intrinsic = (s_b - s_a) / sigma_v1_safe
    extrinsic = total - intrinsic
    return total, intrinsic, extrinsic


def _decomposition_scalars(
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    fit_v1: CalibrationFit,
    fit_v2: CalibrationFit,
) -> tuple[float, float, float]:
    """Aggregate scalars only — (intrinsic_disp, extrinsic_disp, intrinsic_fraction).

    For batch hot paths (π9 find_drifting_entities) where per-dim breakdown
    and ranking are not needed.
    """
    _, intrinsic, extrinsic = _compute_decomposition_vectors(
        shape_a, shape_b, fit_v1, fit_v2,
    )
    intrinsic_disp = float(np.linalg.norm(intrinsic))
    extrinsic_disp = float(np.linalg.norm(extrinsic))
    denom = intrinsic_disp ** 2 + extrinsic_disp ** 2
    intrinsic_fraction = float(intrinsic_disp ** 2 / denom) if denom > 0.0 else 0.0
    return intrinsic_disp, extrinsic_disp, intrinsic_fraction


def _compute_intrinsic_extrinsic_decomposition(
    *,
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    fit_v1: CalibrationFit,
    fit_v2: CalibrationFit,
    entity_key: str,
    pattern_id: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    dim_labels: list[str] | None,
    top_n: int,
    verbose: bool,
) -> IntrinsicExtrinsicReport:
    """Pure math: decompose drift between two temporal slices given two calibrations.

    Caller verifies schema_hash agreement, distinct versions, anchor pattern type,
    and slice availability before calling. Helper trusts its inputs.
    """
    total, intrinsic, extrinsic = _compute_decomposition_vectors(
        shape_a, shape_b, fit_v1, fit_v2,
    )

    intrinsic_disp = float(np.linalg.norm(intrinsic))
    extrinsic_disp = float(np.linalg.norm(extrinsic))
    total_disp = float(np.linalg.norm(total))

    denom = intrinsic_disp ** 2 + extrinsic_disp ** 2
    intrinsic_fraction = float(intrinsic_disp ** 2 / denom) if denom > 0.0 else 0.0

    kinds_v1 = fit_v1.dimension_kinds
    D = total.shape[0]
    per_dim: list[DimensionDecomposition] = []
    for i in range(D):
        i_sq = float(intrinsic[i]) ** 2
        e_sq = float(extrinsic[i]) ** 2
        per_dim_denom = i_sq + e_sq
        per_dim_frac = i_sq / per_dim_denom if per_dim_denom > 0.0 else 0.0
        per_dim.append(
            DimensionDecomposition(
                dim_index=i,
                dim_kind=kinds_v1[i] if kinds_v1 is not None else None,
                dim_label=dim_labels[i] if dim_labels is not None and i < len(dim_labels) else None,
                total=float(total[i]),
                intrinsic=float(intrinsic[i]),
                extrinsic=float(extrinsic[i]),
                intrinsic_fraction=per_dim_frac,
            )
        )

    ranked = sorted(per_dim, key=lambda d: abs(d.total), reverse=True)
    top = ranked[: min(top_n, D)]

    return IntrinsicExtrinsicReport(
        pattern_id=pattern_id,
        entity_key=entity_key,
        v_from=fit_v1.calibration_epoch,
        v_to=fit_v2.calibration_epoch,
        schema_hash=fit_v1.schema_hash,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        intrinsic_displacement=intrinsic_disp,
        extrinsic_displacement=extrinsic_disp,
        total_displacement=total_disp,
        intrinsic_fraction=intrinsic_fraction,
        top_dimensions=top,
        per_dimension=per_dim if verbose else None,
    )


# ---------------------------------------------------------------------------
# Hidden-influencer / coordinate-system influence
# ---------------------------------------------------------------------------

_M4_SIGMA_SAFE_FLOOR = 1e-12


def _compute_leave_one_out_impact(
    shapes: np.ndarray,
    mu_full: np.ndarray,
    sigma_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-entity exact leave-one-out impact on coordinate system calibration.

    Pure math. Inputs trusted (validated by orchestrator).

    Returns:
      mu_impact:     (N,)    L2 norm of per-entity μ-shift (σ-normalised)
      sigma_impact:  (N,)    L2 norm of per-entity σ-shift (σ-normalised)
      total_impact:  (N,)    sqrt(mu_impact² + sigma_impact²)
      contributions: (N, D)  per-dim sqrt(mu_shift_i² + sigma_shift_i²)
    """
    N, D = shapes.shape
    if N < 2:
        raise ValueError(
            f"_compute_leave_one_out_impact: requires N >= 2; got N={N}"
        )

    sum_s = shapes.sum(axis=0)
    sum_s_sq = (shapes ** 2).sum(axis=0)

    sigma_full_safe = np.maximum(sigma_full, _M4_SIGMA_SAFE_FLOOR)

    mu_without = (sum_s[None, :] - shapes) / (N - 1)
    var_without = (sum_s_sq[None, :] - shapes ** 2) / (N - 1) - mu_without ** 2
    var_without = np.maximum(var_without, 0.0)
    sigma_without = np.sqrt(var_without)

    mu_shift = (mu_full[None, :] - mu_without) / sigma_full_safe[None, :]
    sigma_shift = (sigma_full[None, :] - sigma_without) / sigma_full_safe[None, :]

    contributions = np.sqrt(mu_shift ** 2 + sigma_shift ** 2)
    mu_impact = np.linalg.norm(mu_shift, axis=1)
    sigma_impact = np.linalg.norm(sigma_shift, axis=1)
    total_impact = np.sqrt(mu_impact ** 2 + sigma_impact ** 2)

    if not np.all(np.isfinite(total_impact)):
        raise ValueError(
            "_compute_leave_one_out_impact: produced non-finite values; "
            "Welford precision artefact — investigate input shapes/sigma_full"
        )

    return mu_impact, sigma_impact, total_impact, contributions


def _classify_influence(
    total_impact: np.ndarray,
    delta_norm: np.ndarray,
    theta_norm: float,
    high_threshold_pct: float = 90.0,
) -> list[str]:
    """4-cell classification by (impact percentile, anomaly threshold).

    Returns list[str] of length N with one of:
      "hidden", "distorter", "standard_anomaly", "normal"
    """
    if total_impact.shape != delta_norm.shape:
        raise ValueError(
            f"_classify_influence: shape mismatch total_impact={total_impact.shape} "
            f"vs delta_norm={delta_norm.shape}"
        )
    impact_threshold = float(np.percentile(total_impact, high_threshold_pct))
    high_impact = total_impact >= impact_threshold
    high_anomaly = delta_norm >= theta_norm

    classes: list[str] = []
    for hi, ha in zip(high_impact.tolist(), high_anomaly.tolist(), strict=True):
        if hi and not ha:
            classes.append("hidden")
        elif hi and ha:
            classes.append("distorter")
        elif (not hi) and ha:
            classes.append("standard_anomaly")
        else:
            classes.append("normal")
    return classes


def _count_cascading_flips(
    *,
    shape_E: np.ndarray,
    sum_s: np.ndarray,
    sum_s_sq: np.ndarray,
    shapes: np.ndarray,
    is_anomaly_full: np.ndarray,
    e_idx: int,
    theta_norm: float,
) -> int:
    """Count how many other entities flip is_anomaly classification
    after removing entity E from the population stats."""
    N, _ = shapes.shape
    mu_without = (sum_s - shape_E) / (N - 1)
    var_without = (sum_s_sq - shape_E ** 2) / (N - 1) - mu_without ** 2
    var_without = np.maximum(var_without, 0.0)
    sigma_without = np.sqrt(var_without)
    sigma_without_safe = np.maximum(sigma_without, _M4_SIGMA_SAFE_FLOOR)

    deltas_without = (shapes - mu_without) / sigma_without_safe
    delta_norms_without = np.linalg.norm(deltas_without, axis=1)
    is_anomaly_without = delta_norms_without >= theta_norm

    flipped = is_anomaly_full != is_anomaly_without
    flipped[e_idx] = False
    return int(flipped.sum())


def _compute_leave_set_out_impact(
    *,
    shapes: np.ndarray,
    members_idx: np.ndarray,
    mu_full: np.ndarray,
    sigma_full: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Leave-set-out impact for one group of entity indices."""
    N, D = shapes.shape
    k = len(members_idx)
    if k < 2:
        raise ValueError(
            f"_compute_leave_set_out_impact: group must have >=2 members; got {k}"
        )
    if k >= N:
        raise ValueError(
            f"_compute_leave_set_out_impact: group size {k} >= N={N}; "
            f"cannot leave non-empty population"
        )

    sum_s = shapes.sum(axis=0)
    sum_s_sq = (shapes ** 2).sum(axis=0)
    set_shapes = shapes[members_idx]
    set_sum = set_shapes.sum(axis=0)
    set_sum_sq = (set_shapes ** 2).sum(axis=0)

    mu_without_set = (sum_s - set_sum) / (N - k)
    var_without_set = (sum_s_sq - set_sum_sq) / (N - k) - mu_without_set ** 2
    var_without_set = np.maximum(var_without_set, 0.0)
    sigma_without_set = np.sqrt(var_without_set)

    sigma_full_safe = np.maximum(sigma_full, _M4_SIGMA_SAFE_FLOOR)
    mu_shift = (mu_full - mu_without_set) / sigma_full_safe
    sigma_shift = (sigma_full - sigma_without_set) / sigma_full_safe

    contributions = np.sqrt(mu_shift ** 2 + sigma_shift ** 2)
    mu_impact_set = float(np.linalg.norm(mu_shift))
    sigma_impact_set = float(np.linalg.norm(sigma_shift))
    total_impact_set = float(
        np.sqrt(mu_impact_set ** 2 + sigma_impact_set ** 2)
    )
    return mu_impact_set, sigma_impact_set, total_impact_set, contributions


# ── Cross-pattern lead-lag ────────────────────────────────────────────────────


def _compute_centroid_drift_series(
    shapes: np.ndarray,           # (n_epochs, n_entities, D), float32
    mu: np.ndarray,               # (D,)
    sigma: np.ndarray,             # (D,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population centroid drift + per-entity volatility series + raw mu_pop.

    Returns (centroid_drift, volatility, mu_pop) where:
      centroid_drift : (n_epochs - 1,)  ||mu_pop(t+1) - mu_pop(t)||
      volatility     : (n_epochs - 1,)  mean over entities of step magnitude
      mu_pop         : (n_epochs, D)    population centroid trajectory

    Inputs are raw shapes; delta = (shape - mu) / max(sigma, 1e-2). The
    sigma floor mirrors navigator convention.
    """
    sigma_safe = np.maximum(sigma, 1e-2).astype(np.float32, copy=False)
    delta = (shapes - mu.astype(np.float32, copy=False)) / sigma_safe
    mu_pop = delta.mean(axis=1)
    diff_pop = np.diff(mu_pop, axis=0)
    centroid_drift = np.sqrt(np.einsum("ij,ij->i", diff_pop, diff_pop)).astype(np.float64)
    diff_per_entity = np.diff(delta, axis=0)
    per_entity_step = np.sqrt(
        np.einsum("ijk,ijk->ij", diff_per_entity, diff_per_entity)
    )
    volatility = per_entity_step.mean(axis=1).astype(np.float64)
    return centroid_drift, volatility, mu_pop.astype(np.float64)


def _cross_correlate_with_lag(
    a: np.ndarray,            # (L_full,)
    b: np.ndarray,            # (L_full,)
    max_lag: int,
) -> tuple[np.ndarray, int, float]:
    """Cross-correlate two equal-length series at lags [-max_lag, +max_lag].

    Constant-window convention: at every lag both series are trimmed on
    both sides so the pair entering Pearson has length L = L_full - 2*max_lag.
    Keeps Bartlett SE comparable across lags. Positive lag means `a` leads
    `b` (a[t] correlates with b[t + lag]).

    Returns (corr_by_lag, peak_lag, peak_corr) where corr_by_lag has length
    2*max_lag + 1 and peak_corr is signed.
    """
    L_full = a.shape[0]
    if L_full != b.shape[0]:
        raise ValueError(
            f"_cross_correlate_with_lag: length mismatch {L_full} vs {b.shape[0]}"
        )
    if L_full <= 2 * max_lag:
        raise ValueError(
            f"_cross_correlate_with_lag: series length {L_full} too short for "
            f"max_lag={max_lag}"
        )
    L = L_full - 2 * max_lag
    corr_by_lag = np.empty(2 * max_lag + 1, dtype=np.float64)
    a_centre = a[max_lag : max_lag + L]
    a_std = a_centre.std()
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        b_slice = b[max_lag + lag : max_lag + lag + L]
        if a_std == 0.0 or b_slice.std() == 0.0:
            corr_by_lag[k] = 0.0
            continue
        corr_by_lag[k] = float(np.corrcoef(a_centre, b_slice)[0, 1])
    abs_corrs = np.abs(corr_by_lag)
    peak_idx = int(np.argmax(abs_corrs))
    peak_lag = peak_idx - max_lag
    peak_corr = float(corr_by_lag[peak_idx])
    return corr_by_lag, peak_lag, peak_corr


def _compute_per_dim_lead_lag(
    mu_pop_a: np.ndarray,         # (n_epochs, D_a)
    mu_pop_b: np.ndarray,         # (n_epochs, D_b)
    *,
    max_lag: int,
    fdr_alpha: float,
    fdr_method: str,              # "bh" | "storey"
    dim_labels_a: list[str] | None,
    dim_labels_b: list[str] | None,
) -> tuple[list, np.ndarray, np.ndarray]:
    """D_A x D_B per-dim cross-pattern lead-lag with BH or Storey FDR.

    Differences each population centroid coordinate to a length-(n_epochs-1)
    series, cross-correlates every (i, j) pair, takes the peak |corr| / lag,
    computes a Bartlett single-test two-sided p-value with the trimmed
    window length L = n_epochs - 1 - 2*max_lag, and applies BH/Storey FDR
    across all D_a * D_b p-values.

    Returns (pairs, p_values, q_values). Pairs are sorted ascending by
    q_value with tie-break on descending |correlation|.
    """
    from scipy import stats

    from hypertopos.engine.fdr import benjamini_hochberg
    from hypertopos.model.sphere import DimPairLeadLag

    D_a = mu_pop_a.shape[1]
    D_b = mu_pop_b.shape[1]
    da = np.diff(mu_pop_a, axis=0)
    db = np.diff(mu_pop_b, axis=0)
    L_full = da.shape[0]
    L = L_full - 2 * max_lag
    if L < 2:
        raise ValueError(
            f"_compute_per_dim_lead_lag: trimmed window L={L} < 2; "
            f"max_lag={max_lag} too large for n_epochs-1={L_full}"
        )
    n_pairs = D_a * D_b
    n_lags = 2 * max_lag + 1
    p_values = np.empty(n_pairs, dtype=np.float64)
    raw: list[tuple[int, int, int, float, float]] = []
    for i in range(D_a):
        a_dim = da[:, i]
        for j in range(D_b):
            b_dim = db[:, j]
            _, peak_lag, peak_corr = _cross_correlate_with_lag(a_dim, b_dim, max_lag)
            z = abs(peak_corr) * np.sqrt(L)
            p_single = 2.0 * (1.0 - float(stats.norm.cdf(z)))
            # Bonferroni over the lag grid: peak-of-n_lags single-test → multiply by n_lags
            # (matches the population-level max_corr_threshold philosophy and prevents
            # false-positive inflation when the peak |corr| is taken over the lag grid).
            p_pair = float(min(max(p_single * n_lags, 0.0), 1.0))
            idx = i * D_b + j
            p_values[idx] = p_pair
            raw.append((i, j, int(peak_lag), float(peak_corr), p_pair))
    rejected, q_values = benjamini_hochberg(p_values, fdr_alpha, method=fdr_method)
    pairs: list[DimPairLeadLag] = []
    for idx, (i, j, lag, corr, p) in enumerate(raw):
        pairs.append(
            DimPairLeadLag(
                dim_index_a=i,
                dim_index_b=j,
                dim_label_a=(
                    dim_labels_a[i]
                    if dim_labels_a is not None and i < len(dim_labels_a)
                    else None
                ),
                dim_label_b=(
                    dim_labels_b[j]
                    if dim_labels_b is not None and j < len(dim_labels_b)
                    else None
                ),
                lag=int(lag),
                correlation=round(float(corr), 4),
                p_value=round(float(p), 6),
                q_value=round(float(q_values[idx]), 6),
                is_significant=bool(rejected[idx]),
            )
        )
    pairs.sort(key=lambda x: (x.q_value, -abs(x.correlation)))
    return pairs, p_values, q_values


def _reliability_label_for_lead_lag(n_epochs: int) -> str:
    """Mirror engine.forecast.reliability_label: high >= 24, medium >= 12, else low."""
    n_eff = n_epochs - 1
    if n_eff >= 24:
        return "high"
    if n_eff >= 12:
        return "medium"
    return "low"


def _compute_lead_lag_report(
    *,
    pattern_a: str,
    pattern_b: str,
    entity_key: str | None,
    shapes_a: np.ndarray,           # (n_epochs, n_entities, D_a)
    shapes_b: np.ndarray,           # (n_epochs, n_entities, D_b)
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    mu_b: np.ndarray,
    sigma_b: np.ndarray,
    dim_labels_a: list[str] | None,
    dim_labels_b: list[str] | None,
    timestamps: list,                # list[datetime] of length n_epochs
    n_dropped_a: int,
    n_dropped_b: int,
    cohort_size: int,
    cohort_dropped: int | None,
    schema_hash_a: str,
    schema_hash_b: str,
    max_lag: int,
    fdr_alpha: float,
    fdr_method: str,
    verbose: bool,
):
    """Pure-numpy orchestrator. Builds LeadLagReport from aligned shape tensors.

    Caller is responsible for time alignment, cohort selection, and reading
    the (n_epochs, n_entities, D) tensors from temporal storage.
    """
    from scipy import stats

    from hypertopos.model.sphere import LeadLagReport

    n_epochs = shapes_a.shape[0]
    if shapes_b.shape[0] != n_epochs:
        raise ValueError(
            f"_compute_lead_lag_report: shape mismatch shapes_a[0]={n_epochs} "
            f"shapes_b[0]={shapes_b.shape[0]}"
        )
    L_full = n_epochs - 1
    L = L_full - 2 * max_lag
    if L < 2:
        raise ValueError(
            f"_compute_lead_lag_report: trimmed window L={L} too small; "
            f"max_lag={max_lag} too large for n_epochs={n_epochs}"
        )

    centroid_a, vol_a, mu_pop_a = _compute_centroid_drift_series(
        shapes_a, mu_a, sigma_a,
    )
    centroid_b, vol_b, mu_pop_b = _compute_centroid_drift_series(
        shapes_b, mu_b, sigma_b,
    )

    # Degenerate-signal guard: when either centroid drift series is
    # essentially constant (e.g. AML temporal data with near-identical
    # shapes per epoch), Pearson correlation is undefined and the navigator
    # would silently report corr=0 across all dim pairs. Surface this
    # explicitly so the agent does not over-claim signal absence.
    degenerate_signal = bool(
        float(centroid_a.std()) < 1e-12
        or float(centroid_b.std()) < 1e-12
    )

    corr_centroid_full, peak_lag, peak_corr = _cross_correlate_with_lag(
        centroid_a, centroid_b, max_lag,
    )
    _, peak_lag_vol, peak_corr_vol = _cross_correlate_with_lag(
        vol_a, vol_b, max_lag,
    )

    if degenerate_signal:
        agreement = "divergent"
    elif (
        abs(peak_lag - peak_lag_vol) <= 1
        and (peak_corr * peak_corr_vol > 0)
        and abs(peak_corr) > 0.3
        and abs(peak_corr_vol) > 0.3
    ):
        agreement = "strong"
    elif abs(peak_corr) > 0.2 and abs(peak_corr_vol) > 0.2:
        agreement = "weak"
    else:
        agreement = "divergent"

    bartlett_ci_95 = float(1.96 / np.sqrt(L))
    n_lags = 2 * max_lag + 1
    alpha_per_lag = 0.05 / n_lags
    z_adj = float(stats.norm.isf(alpha_per_lag / 2.0))
    max_corr_threshold = float(z_adj / np.sqrt(L))
    is_significant = bool(abs(peak_corr) > max_corr_threshold)

    pairs, p_values, q_values = _compute_per_dim_lead_lag(
        mu_pop_a, mu_pop_b,
        max_lag=max_lag,
        fdr_alpha=fdr_alpha,
        fdr_method=fdr_method,
        dim_labels_a=dim_labels_a,
        dim_labels_b=dim_labels_b,
    )
    n_significant_pairs = int(sum(1 for p in pairs if p.is_significant))
    top_dim_pairs = list(pairs[:10])
    per_dim_pairs = list(pairs) if verbose else None

    return LeadLagReport(
        pattern_a=pattern_a,
        pattern_b=pattern_b,
        entity_key=entity_key,
        n_epochs_used=int(n_epochs),
        n_dropped_a=int(n_dropped_a),
        n_dropped_b=int(n_dropped_b),
        cohort_size=int(cohort_size),
        cohort_dropped=cohort_dropped,
        timestamp_from=timestamps[0],
        timestamp_to=timestamps[-1],
        schema_hash_a=schema_hash_a,
        schema_hash_b=schema_hash_b,
        lag=int(peak_lag),
        correlation=round(float(peak_corr), 4),
        centroid_drift_series_a=[round(float(x), 4) for x in centroid_a.tolist()],
        centroid_drift_series_b=[round(float(x), 4) for x in centroid_b.tolist()],
        lag_volatility=int(peak_lag_vol),
        correlation_volatility=round(float(peak_corr_vol), 4),
        volatility_series_a=[round(float(x), 4) for x in vol_a.tolist()],
        volatility_series_b=[round(float(x), 4) for x in vol_b.tolist()],
        agreement=agreement,
        bartlett_ci_95=round(bartlett_ci_95, 4),
        max_corr_threshold=round(max_corr_threshold, 4),
        is_significant=is_significant,
        fdr_alpha=float(fdr_alpha),
        fdr_method=str(fdr_method),
        n_dim_pairs=int(mu_pop_a.shape[1] * mu_pop_b.shape[1]),
        n_significant_pairs=n_significant_pairs,
        top_dim_pairs=top_dim_pairs,
        per_dim_pairs=per_dim_pairs,
        reliability=_reliability_label_for_lead_lag(n_epochs),
        max_lag=int(max_lag),
        correlation_by_lag=[round(float(x), 4) for x in corr_centroid_full.tolist()],
        coverage_warning=bool(cohort_size is not None and cohort_size < 30),
        degenerate_signal=degenerate_signal,
    )
