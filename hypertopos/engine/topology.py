# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Topology helpers for the multi-detector consensus stack.

Currently exposes a single primitive — `trajectory_continuous_score` — which
emits a per-entity DTW distance against a median reference trajectory. This
replaces the categorical 5-class output of `detect_trajectory_anomaly` when
feeding the harmonic-mean p-value combiner, where a continuous score is
required for ECDF-based calibration.

Future PH primitives (M3 of patch 0.7.0) live in this module too if the
regime-check gate ships them.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pyarrow as pa

__all__ = [
    "find_topological_anomalies",
    "find_topological_trajectory_anomalies",
    "trajectory_continuous_score",
]


def _dtw_pair(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Symmetric DTW distance between two (T_a, D) and (T_b, D) trajectories.

    Standard dynamic-time-warping DP with Euclidean per-step cost. Returns the
    optimal cumulative cost (not normalized by path length).
    """
    n = seq_a.shape[0]
    m = seq_b.shape[0]
    if n == 0 or m == 0:
        return 0.0
    # Pre-compute pairwise Euclidean costs for vectorisation.
    diff = seq_a[:, None, :] - seq_b[None, :, :]
    cost = np.sqrt(np.sum(diff * diff, axis=-1))
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(
                dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1],
            )
    return float(dtw[n, m])


def _median_trajectory(trajectories: list[np.ndarray]) -> np.ndarray:
    """Element-wise median across a stack of (T_i, D) arrays.

    Trajectories of unequal length are linearly interpolated onto the modal
    length so the median is well-defined per-time-step.
    """
    if not trajectories:
        return np.zeros((0, 0), dtype=np.float64)
    if len(trajectories) == 1:
        return trajectories[0].astype(np.float64, copy=True)
    # Modal length keeps the median anchored to the most common observation cadence.
    length_counts: dict[int, int] = defaultdict(int)
    for traj in trajectories:
        length_counts[traj.shape[0]] += 1
    target_len = max(length_counts.items(), key=lambda x: (x[1], x[0]))[0]
    if target_len == 0:
        return np.zeros((0, trajectories[0].shape[1]), dtype=np.float64)
    target_dim = trajectories[0].shape[1]
    resampled = np.empty((len(trajectories), target_len, target_dim), dtype=np.float64)
    for i, traj in enumerate(trajectories):
        if traj.shape[0] == target_len:
            resampled[i] = traj.astype(np.float64, copy=False)
            continue
        # Linear interpolation onto target_len evenly-spaced positions.
        src_idx = np.linspace(0.0, traj.shape[0] - 1, target_len)
        for d in range(target_dim):
            resampled[i, :, d] = np.interp(
                src_idx, np.arange(traj.shape[0]), traj[:, d].astype(np.float64),
            )
    return np.median(resampled, axis=0)


def trajectory_continuous_score(
    solid_table: pa.Table,
    *,
    sample_size: int = 10_000,
) -> dict[str, float]:
    """Per-entity continuous trajectory anomaly score.

    For every entity in `solid_table`, compute the DTW distance between the
    entity's trajectory of `delta_snapshot` vectors and the population's median
    trajectory. Replaces the categorical 5-class output of
    `detect_trajectory_anomaly` when feeding the harmonic-mean p-value combiner.

    Args:
        solid_table: Arrow table with at least `primary_key` and
            `delta_snapshot` columns. `delta_snapshot` is a list<float> column;
            rows for the same entity are grouped (any stable ordering is OK as
            long as it reflects time order — typically `t` ascending).
        sample_size: Maximum number of trajectories used to estimate the
            median reference. Larger populations are randomly subsampled with a
            deterministic seed (default 0). All entities are still scored.

    Returns:
        ``{primary_key: dtw_distance}`` — non-negative floats, no NaN/inf.
        ``0.0`` indicates the entity matches the median trajectory exactly.
    """
    if solid_table.num_rows == 0:
        return {}
    if "primary_key" not in solid_table.schema.names:
        raise ValueError("solid_table must have a 'primary_key' column")
    if "delta_snapshot" not in solid_table.schema.names:
        raise ValueError("solid_table must have a 'delta_snapshot' column")

    pks = solid_table["primary_key"].to_pylist()
    snaps = solid_table["delta_snapshot"].to_pylist()

    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for pk, snap in zip(pks, snaps, strict=False):
        if pk is None or snap is None:
            continue
        grouped[pk].append(list(snap))

    if not grouped:
        return {}

    trajectories: dict[str, np.ndarray] = {}
    for pk, frames in grouped.items():
        if not frames:
            continue
        arr = np.asarray(frames, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        trajectories[pk] = arr

    if not trajectories:
        return {}

    keys = list(trajectories.keys())
    if len(keys) > sample_size:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(keys), size=sample_size, replace=False)
        sample_keys = [keys[i] for i in sample_idx]
    else:
        sample_keys = keys

    median_traj = _median_trajectory([trajectories[k] for k in sample_keys])
    if median_traj.size == 0:
        return {pk: 0.0 for pk in keys}

    out: dict[str, float] = {}
    for pk, traj in trajectories.items():
        score = _dtw_pair(traj, median_traj)
        if not np.isfinite(score) or score < 0.0:
            score = 0.0
        out[pk] = float(score)
    return out


_MIN_ENTITIES = 1000
_RELIABILITY_THRESHOLD = 10_000


def find_topological_anomalies(
    geometry_table: pa.Table,
    *,
    k_neighbors: int = 50,
    homology_dim: int = 1,
    pca_dim: int = 10,
    sample_size: int = 50_000,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Per-entity local k-NN VR-filtration H_1 persistence anomaly score.

    For each scored entity, build the Vietoris-Rips filtration on its
    ``k_neighbors``-nearest neighborhood in the (optionally PCA-projected)
    geometry space and compute ``h1_max_persistence`` (the longest finite
    H_1 cycle lifetime in the local diagram) plus the diagnostic ratio
    ``topo_score = h1_max_persistence / max(eps, h0_mean_death)``. **Results
    are ranked by ``h1_max_persistence`` descending** — multi-sphere AUROC
    validation showed the raw H_1 persistence carries the discriminative
    signal while the H_0-normalised ratio dilutes it. ``topo_score`` is
    retained as an auxiliary diagnostic field, not the ranking key.

    Args:
        geometry_table: Arrow table with a ``primary_key`` column and one or
            more float feature columns.
        k_neighbors: size of the per-entity local cloud passed to ripser.
        homology_dim: max homology dimension computed (default 1 → H_0 + H_1).
        pca_dim: project to this many PCA components if input dim is larger.
        sample_size: cap on entities to score. If input has more, a random
            ``sample_size`` subset is scored.
        top_n: number of top-score entities returned.

    Returns:
        List of ``top_n`` dicts (sorted by topo_score descending) with fields
        ``primary_key``, ``topo_score``, ``h1_max_persistence``,
        ``h0_mean_death``, ``n_h1_features``, ``computed_at``.

    Raises:
        ValueError: if ``primary_key`` column is missing, if there are no
            numeric feature columns, or if ``n_entities < 1000``.

    Warns:
        UserWarning: if ``1000 <= n_entities < 10_000`` ("PH reliability
            degrades for small populations").
    """
    from ripser import ripser
    from sklearn.neighbors import NearestNeighbors

    if "primary_key" not in geometry_table.schema.names:
        raise ValueError("geometry_table must have a 'primary_key' column")

    pks = geometry_table["primary_key"].to_pylist()
    n_entities = len(pks)

    if n_entities < _MIN_ENTITIES:
        raise ValueError(
            f"find_topological_anomalies requires n_entities >= {_MIN_ENTITIES} "
            f"(got {n_entities}); persistent homology on smaller populations "
            "is unreliable",
        )

    if n_entities < _RELIABILITY_THRESHOLD:
        warnings.warn(
            f"PH reliability degrades for n_entities < {_RELIABILITY_THRESHOLD} "
            f"(got {n_entities}); results may be noisy. Consider a larger "
            "sample or a different sphere.",
            UserWarning,
            stacklevel=2,
        )

    feature_cols = [
        field.name
        for field in geometry_table.schema
        if field.name != "primary_key"
        and (pa.types.is_floating(field.type) or pa.types.is_integer(field.type))
    ]
    if not feature_cols:
        raise ValueError("geometry_table has no numeric feature columns")

    coords = np.column_stack([
        np.asarray(geometry_table[c].to_numpy(zero_copy_only=False), dtype=np.float64)
        for c in feature_cols
    ])

    if coords.shape[1] > pca_dim:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=pca_dim, random_state=0).fit_transform(coords)

    if n_entities > sample_size:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_entities, size=sample_size, replace=False)
    else:
        sample_idx = np.arange(n_entities)

    effective_k = min(k_neighbors, n_entities)
    nn = NearestNeighbors(n_neighbors=effective_k).fit(coords)

    eps = 1e-12
    rows: list[tuple[int, float, float, float, int]] = []
    for i in sample_idx:
        idx = nn.kneighbors(coords[i:i + 1], return_distance=False)[0]
        neighborhood = coords[idx]
        try:
            diagrams = ripser(neighborhood, maxdim=homology_dim, thresh=np.inf)["dgms"]
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"ripser failed for entity index {int(i)}: {exc!r}",
                UserWarning,
                stacklevel=2,
            )
            rows.append((int(i), 0.0, 0.0, 0.0, 0))
            continue

        h0 = diagrams[0]
        if h0.size > 0:
            h0_finite = h0[np.isfinite(h0[:, 1])]
            h0_mean_death = float(h0_finite[:, 1].mean()) if h0_finite.size else 0.0
        else:
            h0_mean_death = 0.0

        h1_max = 0.0
        n_h1 = 0
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            h1 = diagrams[1]
            h1_finite = h1[np.isfinite(h1[:, 1])]
            if h1_finite.size > 0:
                lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
                h1_max = float(lifetimes.max())
                n_h1 = int(h1_finite.shape[0])

        topo_score = h1_max / max(eps, h0_mean_death)
        if not np.isfinite(topo_score) or topo_score < 0.0:
            topo_score = 0.0
        rows.append((int(i), float(topo_score), h1_max, h0_mean_death, n_h1))

    # Rank by h1_max_persistence (r[2]); empirical validation showed the
    # H_0-normalised topo_score (r[1]) dilutes discriminative signal.
    rows.sort(key=lambda r: r[2], reverse=True)
    now = datetime.now(timezone.utc)
    return [
        {
            "primary_key": pks[idx],
            "topo_score": score,
            "h1_max_persistence": h1_max,
            "h0_mean_death": h0_mean,
            "n_h1_features": n_h1,
            "computed_at": now,
        }
        for idx, score, h1_max, h0_mean, n_h1 in rows[:top_n]
    ]


def find_topological_trajectory_anomalies(
    solid_table: pa.Table,
    *,
    homology_dim: int = 1,
    min_timesteps: int = 8,
    pca_dim: int = 5,
    sample_size: int = 10_000,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Per-entity trajectory-PH anomaly score over the temporal solid.

    For each entity whose trajectory has at least ``min_timesteps`` deformation
    samples, compute the Vietoris-Rips filtration on its (T, D) trajectory
    matrix (optionally PCA-reduced to ``pca_dim``) and report the total
    finite-lifetime persistence of homology dimension ``homology_dim`` as
    ``trajectory_topo_score``. Entities with closed-loop trajectories surface
    via high H_1 persistence; monotonic or stationary trajectories score low.

    Args:
        solid_table: Arrow table with ``primary_key`` and ``delta_snapshot``
            columns. ``delta_snapshot`` is a list<float> column; rows for one
            entity are grouped (assumed to be in time order — typically ``t``
            ascending).
        homology_dim: max homology dimension passed to ripser (default 1).
        min_timesteps: entities with fewer than this many trajectory points
            are dropped from scoring.
        pca_dim: project the (T, D) trajectory to this many components if
            D is larger.
        sample_size: cap on entities scored. Larger populations are randomly
            sub-sampled with a deterministic seed.
        top_n: number of top-score entities returned.

    Returns:
        List of dicts sorted by ``trajectory_topo_score`` descending, with
        ``primary_key``, ``trajectory_topo_score``, ``n_timesteps``,
        ``h1_total_persistence``, ``dominant_feature_birth``,
        ``dominant_feature_death``, ``computed_at``.
    """
    from ripser import ripser

    if solid_table.num_rows == 0:
        return []
    if "primary_key" not in solid_table.schema.names:
        raise ValueError("solid_table must have a 'primary_key' column")
    if "delta_snapshot" not in solid_table.schema.names:
        raise ValueError("solid_table must have a 'delta_snapshot' column")

    pks_col = solid_table["primary_key"].to_pylist()
    snaps_col = solid_table["delta_snapshot"].to_pylist()

    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for pk, snap in zip(pks_col, snaps_col, strict=False):
        if pk is None or snap is None:
            continue
        grouped[pk].append(list(snap))

    trajectories: dict[str, np.ndarray] = {}
    for pk, frames in grouped.items():
        arr = np.asarray(frames, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < min_timesteps:
            continue
        trajectories[pk] = arr

    if not trajectories:
        return []

    keys = list(trajectories.keys())
    if len(keys) > sample_size:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(keys), size=sample_size, replace=False)
        sample_keys = [keys[i] for i in sample_idx]
    else:
        sample_keys = keys

    rows: list[tuple[str, float, int, float, float, float]] = []
    for pk in sample_keys:
        traj = trajectories[pk]
        n_t = int(traj.shape[0])

        if traj.shape[1] > pca_dim:
            from sklearn.decomposition import PCA

            n_components = min(pca_dim, traj.shape[0], traj.shape[1])
            traj_proj = PCA(n_components=n_components, random_state=0).fit_transform(traj)
        else:
            traj_proj = traj

        try:
            diagrams = ripser(traj_proj, maxdim=homology_dim, thresh=np.inf)["dgms"]
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"ripser failed for entity {pk}: {exc!r}",
                UserWarning,
                stacklevel=2,
            )
            rows.append((pk, 0.0, n_t, 0.0, 0.0, 0.0))
            continue

        total_persistence = 0.0
        dominant_birth = 0.0
        dominant_death = 0.0
        h1_total = 0.0

        if homology_dim < len(diagrams):
            target = diagrams[homology_dim]
            if target.size > 0:
                finite = target[np.isfinite(target[:, 1])]
                if finite.size > 0:
                    lifetimes = finite[:, 1] - finite[:, 0]
                    total_persistence = float(lifetimes.sum())
                    best_idx = int(np.argmax(lifetimes))
                    dominant_birth = float(finite[best_idx, 0])
                    dominant_death = float(finite[best_idx, 1])

        if homology_dim == 1:
            h1_total = total_persistence
        elif len(diagrams) > 1 and diagrams[1].size > 0:
            h1 = diagrams[1]
            h1_finite = h1[np.isfinite(h1[:, 1])]
            if h1_finite.size > 0:
                h1_total = float((h1_finite[:, 1] - h1_finite[:, 0]).sum())

        if not np.isfinite(total_persistence) or total_persistence < 0.0:
            total_persistence = 0.0
        rows.append((pk, float(total_persistence), n_t, float(h1_total),
                     dominant_birth, dominant_death))

    rows.sort(key=lambda r: r[1], reverse=True)
    now = datetime.now(timezone.utc)
    return [
        {
            "primary_key": pk,
            "trajectory_topo_score": score,
            "n_timesteps": n_t,
            "h1_total_persistence": h1_total,
            "dominant_feature_birth": dom_birth,
            "dominant_feature_death": dom_death,
            "computed_at": now,
        }
        for pk, score, n_t, h1_total, dom_birth, dom_death in rows[:top_n]
    ]
