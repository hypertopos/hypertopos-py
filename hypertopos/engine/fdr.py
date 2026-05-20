# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Multiple-testing correction primitives for the navigator layer.

Pure NumPy + scipy.special. No state. No I/O. Closed-form per-batch operations.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray
from scipy.special import gammaincc
from scipy.stats import hypergeom

__all__ = [
    "benjamini_hochberg",
    "cell_p_values_from_anomaly_indicator",
    "empirical_p_values_from_rank",
    "fdr_multi_resolution",
    "fdr_per_dimension",
    "parametric_p_values_chi2",
    "per_dim_p_values_chi2_univariate",
    "q_values_from_p_values",
    "storey_pi0",
    "storey_q_values",
]


def empirical_p_values_from_rank(
    delta_rank_pct: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert delta rank percentile (0-100 or 0-1) into empirical-null p-values.

    delta_rank_pct[i] is the entity's position in the population sorted by
    delta_norm ascending --- 100 (or 1.0) is the most extreme entity.
    p-value = 1 - rank_pct/100 (if input > 1, assume 0-100 scale).

    Returns: 1-D array of p-values in (0, 1].
    """
    p = np.asarray(delta_rank_pct, dtype=np.float64)
    # Auto-detect scale: if max > 1, assume 0-100 range
    if p.size > 0 and np.max(p) > 1.0:
        p = p / 100.0
    p = 1.0 - p
    # clip into (eps, 1) — use a fixed floor (not 1/N) because
    # delta_rank_pct is computed against the FULL population, so
    # the input array may be a small top-N subset with very small p-values.
    return np.clip(p, 1e-10, 1.0)


def parametric_p_values_chi2(
    delta_norms: NDArray[np.float64],
    df: int,
) -> NDArray[np.float64]:
    """Upper-tail p-values under the χ²(df) null for ||delta||².

    If delta_i ~ N(0, 1) iid across df dimensions (the population-relative
    construction), then ||delta||² ~ χ²(df) and the upper-tail survival
    function gives per-entity p-values that concentrate near 0 for genuine
    alternatives — unlike the rank-based p-values, which are uniform by
    construction and carry no null/alternative signal for the Storey estimator.

    Returns: p-values in (0, 1], clipped to [1e-10, 1].
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    chi2 = np.asarray(delta_norms, dtype=np.float64) ** 2
    # gammaincc(df/2, x/2) == 1 - F_χ²(df, x) (upper-tail survival function)
    p = gammaincc(df / 2.0, chi2 / 2.0)
    return np.clip(p, 1e-10, 1.0)


def q_values_from_p_values(
    p_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute BH q-value for each input p-value.

    q-value of entity i = minimum FDR alpha at which i would be rejected.
    Implementation: right-to-left running minimum of (p_(j) * m / j).

    Returns: 1-D array of q-values in [0, 1] aligned with input order.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    n = p_values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    raw_q_sorted = sorted_p * n / ranks
    # right-to-left cumulative min keeps q monotonic
    q_sorted = np.minimum.accumulate(raw_q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_values = np.empty(n, dtype=np.float64)
    q_values[order] = q_sorted
    return q_values


def storey_pi0(
    p_values: NDArray[np.float64],
    lam: float = 0.5,
) -> float:
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    p = np.asarray(p_values, dtype=np.float64)
    n = p.shape[0]
    if n == 0:
        return 1.0
    tail_count = float(np.sum(p > lam))
    pi0 = tail_count / ((1.0 - lam) * n)
    # Floor at 1/n — prevents collapse to 0 when tail count is 0 on small n,
    # preserves q-value ranking. Matches the R qvalue package behaviour.
    return float(min(max(pi0, 1.0 / n), 1.0))


def storey_q_values(
    p_values: NDArray[np.float64],
    lam: float = 0.5,
) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=np.float64)
    n = p.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    q_bh = q_values_from_p_values(p)
    pi0 = storey_pi0(p, lam=lam)
    return np.clip(pi0 * q_bh, 0.0, 1.0)


def per_dim_p_values_chi2_univariate(
    deltas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-entity-per-dim two-sided p-value under univariate N(0,1) null.

    Each cell ``deltas[i, d]`` is treated as a single z-score under the
    population-relative construction (delta = (shape - mu) / sigma_diag).
    The two-sided p-value is ``2 * (1 - Φ(|delta_i,d|))``, equivalent to
    ``chi²(1).sf(delta_i,d²)`` for a single observation.

    Warning — direction-agnostic: the chi²(1) survival on ``delta²``
    collapses sign, so both extreme-positive and extreme-negative deviations
    on a given dim produce equally small p-values. On spheres where some
    dims are *anti-signal* (high ``|delta|`` correlated with the non-target
    class — measured by ``engine.dim_audit.compute_per_dim_label_auroc``),
    flagging both wings inflates the false-positive count on the anti-signal
    side. When labels are available, combine with a per-dim label-AUROC
    filter or pre-multiply ``deltas`` by a sign-aware weight vector before
    passing to this function.

    Returns:
        ``(n_entities, n_dims)`` p-value matrix in [1e-10, 1.0].
    """
    chi2 = np.asarray(deltas, dtype=np.float64) ** 2
    p = gammaincc(0.5, chi2 / 2.0)
    return np.clip(p, 1e-10, 1.0)


def fdr_per_dimension(
    p_values_per_dim: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    method: str = "bh",
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Apply BH or Storey FDR INDEPENDENTLY per column (per dim).

    Unlike a flat FDR over all ``n_entities * n_dims`` p-values (which
    would inflate every dim's threshold when one dim drives many
    discoveries), per-dim FDR corrects each dim's column on its own
    ``n_entities`` test pool.

    Args:
        p_values_per_dim: ``(n_entities, n_dims)`` matrix of per-entity-
            per-dim p-values. Caller is responsible for the p-value
            construction (see ``per_dim_p_values_chi2_univariate``).
        alpha: nominal FDR control level in (0, 1).
        method: ``"bh"`` (Benjamini-Hochberg) or ``"storey"`` (BH scaled
            by Storey π₀ per dim — recovers power when most dims have
            π₀ < 1).

    Returns:
        ``(rejected, q_values)`` both shaped ``(n_entities, n_dims)``.
        ``rejected[i, d] = True`` iff ``q_values[i, d] <= alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method not in ("bh", "storey"):
        raise ValueError(f"method must be 'bh' or 'storey', got {method!r}")
    p = np.asarray(p_values_per_dim, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"p_values_per_dim must be 2-D, got shape {p.shape}")
    n_entities, n_dims = p.shape
    if n_entities == 0:
        return (
            np.empty((0, n_dims), dtype=np.bool_),
            np.empty((0, n_dims), dtype=np.float64),
        )
    q_values = np.empty_like(p)
    for d in range(n_dims):
        col = p[:, d]
        q_col = (
            q_values_from_p_values(col) if method == "bh"
            else storey_q_values(col)
        )
        q_values[:, d] = q_col
    rejected = q_values <= alpha
    return rejected, q_values


def benjamini_hochberg(
    p_values: NDArray[np.float64],
    alpha: float,
    method: str = "bh",
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Apply BH-family FDR procedure at level alpha.

    method: "bh" (default) — Benjamini-Hochberg, assumes π₀ = 1.
            "storey" — BH scaled by Storey π̂₀, recovers power when π₀ < 1.

    Guarantees: E[FDR | rejected] <= alpha
        - under independence or PRDS for method="bh"
        - under independence for method="storey" (Storey 2002)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method not in ("bh", "storey"):
        raise ValueError(f"method must be 'bh' or 'storey', got {method!r}")
    p_values = np.asarray(p_values, dtype=np.float64)
    n = p_values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float64)
    q_values = q_values_from_p_values(p_values) if method == "bh" else storey_q_values(p_values)
    rejected = q_values <= alpha
    return rejected, q_values


def cell_p_values_from_anomaly_indicator(
    geometry: pa.Table,
    *,
    hierarchy_dims: list[str] | None = None,
    temporal_dim: str | None = None,
    anomaly_col: str = "is_anomaly",
) -> dict[tuple, float]:
    """Per-cell Fisher exact 2x2 upper-tail p-value on anomaly indicator.

    Cell = unique tuple of (hierarchy_dims values..., temporal_dim value).
    Test: 'is anomalous fraction in this cell higher than rest-of-population?'

        | anomalous | normal   |
    ----+-----------+----------+
    cell|    k      |  n-k     |
    ~cell|  K-k     | N-n-K+k  |

    Implementation: hypergeometric upper-tail survival hypergeom(N, K, n).sf(k-1)
    is exact and matches scipy.stats.fisher_exact(alternative='greater').

    Single-cell case (cell == whole population): returns p=1.0 (no contrast).

    Assumes anomaly_col has no nulls; null entries skew the hypergeom params.

    Args:
        geometry: Arrow table with anomaly_col and all hierarchy/temporal cols
            of dtype string or int.
        hierarchy_dims: spatial cell-defining columns (in order). None or empty
            list => no spatial axis.
        temporal_dim: single temporal cell-defining column. None => no temporal
            axis.
        anomaly_col: column with bool/int 0-1 anomaly indicator. Default
            'is_anomaly'.

    Returns:
        dict keyed by cell-tuple; values are upper-tail p-values in (0, 1].
        Cell-tuple shape: (*hierarchy_dims_values, temporal_dim_value) when both
        are present, or whichever subset is.

    Raises:
        ValueError: if anomaly_col missing from geometry; or if any
            hierarchy_dims/temporal_dim missing from geometry; or if no axis
            is provided.
    """
    if anomaly_col not in geometry.schema.names:
        raise ValueError(
            f"anomaly_col {anomaly_col!r} not in geometry columns "
            f"{geometry.schema.names!r}",
        )
    cell_dims: list[str] = list(hierarchy_dims or [])
    if temporal_dim is not None:
        cell_dims.append(temporal_dim)
    if not cell_dims:
        raise ValueError(
            "at least one of hierarchy_dims or temporal_dim must be set",
        )
    missing = [d for d in cell_dims if d not in geometry.schema.names]
    if missing:
        raise ValueError(
            f"cell-defining columns missing from geometry: {missing!r}",
        )

    n_total = geometry.num_rows
    # is_anomaly may arrive as bool or int8; coerce to int64 for sum
    anomaly_int = pc.cast(geometry[anomaly_col], pa.int64())
    k_total = int(pc.sum(anomaly_int).as_py() or 0)

    # Groupby cell-tuple -> (n_cell, k_cell)
    grouped = (
        geometry.append_column("__anomaly_int__", anomaly_int)
        .group_by(cell_dims)
        .aggregate([("__anomaly_int__", "sum"), ("__anomaly_int__", "count")])
    )

    n_arr = np.asarray(grouped["__anomaly_int___count"].to_pylist(), dtype=np.int64)
    k_raw = grouped["__anomaly_int___sum"].to_pylist()
    k_arr = np.asarray(
        [0 if v is None else int(v) for v in k_raw], dtype=np.int64,
    )
    cell_value_cols = [grouped[d].to_pylist() for d in cell_dims]

    # Default p=1.0 covers two degenerate branches: single-cell case
    # (n_cell == n_total) and no-contrast population (k_total in {0, n_total}).
    # Real hypergeom.sf is only called on contrast-bearing cells, in one
    # vectorised scipy call — replaces ~30 k scalar calls on AML-scale spheres.
    p_arr = np.ones(grouped.num_rows, dtype=np.float64)
    if 0 < k_total < n_total:
        real_mask = n_arr != n_total
        if real_mask.any():
            p_arr[real_mask] = hypergeom.sf(
                k_arr[real_mask] - 1, n_total, k_total, n_arr[real_mask],
            )
    p_arr = np.clip(p_arr, 1e-10, 1.0)

    result: dict[tuple, float] = {}
    for i in range(grouped.num_rows):
        cell_tuple = tuple(cell_value_cols[j][i] for j in range(len(cell_dims)))
        result[cell_tuple] = float(p_arr[i])
    return result


def _aggregate_min_p_at_level_spatial(
    cell_p_values: dict[tuple, float],
    *,
    spatial_prefix_len: int,
    spatial_n: int,
) -> dict[tuple, float]:
    """Tippett min-p on spatial prefix only (ignores temporal portion)."""
    out: dict[tuple, float] = {}
    for cell, p in cell_p_values.items():
        parent = cell[:spatial_prefix_len]
        if parent not in out or p < out[parent]:
            out[parent] = p
    return out


def _aggregate_min_p_at_level_temporal(
    cell_p_values: dict[tuple, float],
    *,
    spatial_n: int,
    temporal_prefix_len: int,
) -> dict[tuple, float]:
    """Tippett min-p on temporal prefix only (ignores spatial portion)."""
    out: dict[tuple, float] = {}
    for cell, p in cell_p_values.items():
        temp_parent = cell[spatial_n:spatial_n + temporal_prefix_len]
        if temp_parent not in out or p < out[temp_parent]:
            out[temp_parent] = p
    return out


def _apply_fdr_at_level(
    parent_p_values: dict[tuple, float],
    *,
    method: str,
    alpha: float,
) -> tuple[dict[tuple, float], set[tuple]]:
    """BH or Storey FDR on a flat parent-cell p-value dict.

    Returns (q_values_by_parent, parents_surviving_at_alpha).
    """
    if not parent_p_values:
        return {}, set()
    parents = list(parent_p_values.keys())
    ps = np.array([parent_p_values[p] for p in parents], dtype=np.float64)
    q_arr = q_values_from_p_values(ps) if method == "bh" else storey_q_values(ps)
    q_map = {parents[i]: float(q_arr[i]) for i in range(len(parents))}
    surviving = {parents[i] for i in range(len(parents)) if q_arr[i] <= alpha}
    return q_map, surviving


def fdr_multi_resolution(
    cell_p_values: dict[tuple, float],
    *,
    hierarchy: list[str] | None = None,
    temporal_levels: list[str] | None = None,
    method: str = "storey",
    alpha: float = 0.05,
) -> tuple[dict[tuple, float], set[tuple]]:
    """Per-level BH/Storey FDR over a cell-tuple lattice.

    Cell-tuple layout: ``cell = (*hierarchy_values, *temporal_values)`` where
    ``hierarchy_values`` has ``len(hierarchy)`` entries and ``temporal_values``
    has ``len(temporal_levels)`` entries (in declaration order).

    For each spatial level (suffix-aggregated from root):
        - Project cells to coarse parent (e.g. ('US','CA','SF') -> ('US','CA'))
        - Tippett min-p aggregate child p-values per parent
        - Apply BH/Storey at alpha on the parent set
        - Cell eligible at this level iff its parent projection survived

    Same logic for each temporal level on the temporal portion of the tuple.

    Intersection-FDR: cell ∈ surviving iff cleared every spatial AND every
    temporal level.

    Args:
        cell_p_values: dict mapping cell-tuple to p-value in (0, 1].
        hierarchy: spatial level names (in order, root -> leaf). Empty or None
            -> no spatial axis.
        temporal_levels: temporal level names (in order, root -> leaf). Empty
            or None -> no temporal axis.
        method: ``"bh"`` or ``"storey"``.
        alpha: per-level FDR control level in (0, 1).

    Returns:
        ``(q_values_by_cell, surviving_cells)`` —
          q_values_by_cell: element-wise max over each level's q at that
            cell's projection (the most conservative bound, summarises overall
            evidence).
          surviving_cells: set of cells that cleared every named level.

    Raises:
        ValueError: alpha not in (0, 1); method not bh/storey; no axis declared;
            cell-tuple length mismatch vs spatial_n + temporal_n.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method not in ("bh", "storey"):
        raise ValueError(f"method must be 'bh' or 'storey', got {method!r}")
    if not cell_p_values:
        return {}, set()
    spatial_n = len(hierarchy or [])
    temporal_n = len(temporal_levels or [])
    if spatial_n + temporal_n == 0:
        raise ValueError(
            "at least one of hierarchy or temporal_levels must be non-empty",
        )

    # Validate uniform cell-tuple length
    expected_len = spatial_n + temporal_n
    for cell in cell_p_values:
        if len(cell) != expected_len:
            raise ValueError(
                f"cell-tuple length mismatch: cell {cell!r} has len "
                f"{len(cell)}, expected {expected_len} "
                f"(spatial={spatial_n} + temporal={temporal_n})",
            )

    # Track per-cell max q across levels and whether cell survived every level
    cell_max_q: dict[tuple, float] = dict.fromkeys(cell_p_values, 0.0)
    cell_survives: dict[tuple, bool] = dict.fromkeys(cell_p_values, True)

    # Walk spatial levels root -> leaf
    for level_idx in range(spatial_n):
        prefix_len = level_idx + 1
        parent_p = _aggregate_min_p_at_level_spatial(
            cell_p_values, spatial_prefix_len=prefix_len, spatial_n=spatial_n,
        )
        q_map, surviving = _apply_fdr_at_level(
            parent_p, method=method, alpha=alpha,
        )
        for cell in cell_p_values:
            parent = cell[:prefix_len]
            q = q_map.get(parent, 1.0)
            if q > cell_max_q[cell]:
                cell_max_q[cell] = q
            if parent not in surviving:
                cell_survives[cell] = False

    # Walk temporal levels
    for level_idx in range(temporal_n):
        prefix_len = level_idx + 1
        parent_p = _aggregate_min_p_at_level_temporal(
            cell_p_values,
            spatial_n=spatial_n,
            temporal_prefix_len=prefix_len,
        )
        q_map, surviving = _apply_fdr_at_level(
            parent_p, method=method, alpha=alpha,
        )
        for cell in cell_p_values:
            parent = cell[spatial_n:spatial_n + prefix_len]
            q = q_map.get(parent, 1.0)
            if q > cell_max_q[cell]:
                cell_max_q[cell] = q
            if parent not in surviving:
                cell_survives[cell] = False

    surviving_cells = {c for c, ok in cell_survives.items() if ok}
    return cell_max_q, surviving_cells
