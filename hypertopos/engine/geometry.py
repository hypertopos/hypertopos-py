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
            delta_rank_pct=float(row["delta_rank_pct"]) if "delta_rank_pct" in row else None,
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
        for i in range(table.num_rows):
            row = {col: table[col][i].as_py() for col in table.schema.names}
            shape = np.array(row["shape_snapshot"], dtype=np.float32)
            if pattern.cholesky_inv is not None:
                delta = pattern.cholesky_inv @ (shape - pattern.mu)
            else:
                delta = (shape - pattern.mu) / sigma
            if pattern.dimension_weights is not None:
                delta = delta * pattern.dimension_weights
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
        slices.sort(key=lambda s: s.timestamp)
        if timestamp is not None:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            slices = [s for s in slices if s.timestamp <= timestamp]
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
    ) -> list[tuple[str, float]]:
        """Find top-n nearest entities. Uses Lance ANN index when available.

        filter_expr: optional Lance SQL predicate passed to ANN (e.g. 'is_anomaly = true').
        On the NumPy fallback path the expression is ignored (caller responsibility).
        """
        # Fast path: Lance IVF_FLAT ANN
        _ann_fn = getattr(self._storage, "find_nearest_lance", None)
        if _ann_fn is not None:
            ann = _ann_fn(pattern_id, version, ref_delta, top_n, exclude_keys, filter_expr)
            if ann is not None:
                return ann

        # Fallback: brute-force NumPy (non-indexed Lance)
        geo = self._storage.read_geometry(
            pattern_id, version, columns=["primary_key", "delta"],
        )
        keys = geo["primary_key"].to_pylist()
        deltas = delta_matrix_from_arrow(geo)

        _diff = deltas - ref_delta.astype(np.float32)
        distances = np.sqrt(np.einsum('ij,ij->i', _diff, _diff))

        if exclude_keys:
            mask = np.array([k in exclude_keys for k in keys])
            distances[mask] = np.inf

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
                    float(row["delta_rank_pct"])
                    if "delta_rank_pct" in row else None
                ),
            ))

        results.sort(key=lambda p: p.delta_norm, reverse=True)
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
