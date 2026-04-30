"""Joint density gap detection via PIT + independence null + BH chi^2.

Algorithm (probability integral transform → uniform marginals → chi^2 on
2D histogram against independence-null uniform expected, with BH
multiple-testing correction):

1. Per dim, build empirical CDF cache; transform raw values to uniform
   `[0, 1]` via ``ECDFEntry.transform``.
2. For a pair `(i, j)` of dims, compute a `bins x bins` joint histogram
   of `(u_i, u_j)`. Under the null hypothesis of independence with
   uniform marginals each cell has expected count `N / bins^2`.
3. Per cell, compute the chi^2 residual `(observed - expected)^2 /
   expected`. Only under-populated cells (gaps) are kept; over-populated
   cells are clumps and not the target signal.
4. Apply Benjamini-Hochberg correction across all under-populated cells
   tested. Flag cells with `q <= alpha`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2

from hypertopos.engine.fdr import benjamini_hochberg


@dataclass(frozen=True)
class ECDFEntry:
    """Empirical CDF wrapper with vectorised transform + inverse.

    ``transform`` returns each input's quantile in [0, 1] under the
    empirical distribution defined by the values supplied to
    ``from_values``. ``inverse`` maps a quantile back to a representative
    raw value (right-continuous step function — exact for points in the
    sample, monotone-piecewise-constant elsewhere).
    """

    sorted_values: np.ndarray

    @classmethod
    def from_values(cls, x: np.ndarray) -> ECDFEntry:
        return cls(
            sorted_values=np.sort(np.asarray(x, dtype=np.float64)),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        n = len(self.sorted_values)
        if n == 0:
            return np.zeros_like(x, dtype=np.float64)
        return (
            np.searchsorted(self.sorted_values, x, side="right") / n
        ).astype(np.float64)

    def inverse(self, u: np.ndarray) -> np.ndarray:
        n = len(self.sorted_values)
        if n == 0:
            return np.zeros_like(u, dtype=np.float64)
        idx = np.clip(
            (np.asarray(u) * n).astype(np.int64), 0, n - 1,
        )
        return self.sorted_values[idx]


def is_usable_for_gap(col: np.ndarray) -> tuple[bool, str]:
    """Decide whether a column is admissible for joint gap detection.

    Excludes columns that are too sparse, degenerate (zero variance) or
    bernoulli-like (≤2 unique values) — for those the independence null
    has either no meaningful 2D structure or chi^2 fails.
    """
    finite = col[np.isfinite(col)]
    if len(finite) < 30:
        return False, "too_sparse"
    if np.std(finite, ddof=1) < 1e-12:
        return False, "degenerate"
    if len(np.unique(finite)) <= 2:
        return False, "bernoulli_like"
    return True, "ok"


def select_pairs_by_corr(
    corr: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    top_k: int,
) -> list[tuple[int, int, float]]:
    """Pick dim pairs whose Pearson |r| sits inside the active window.

    Pairs with |r| below ``r_min`` are effectively independent already —
    no interesting joint structure to gap-detect. Pairs with |r| above
    ``r_max`` are so strongly correlated that the off-diagonal is empty
    by construction and would saturate the gap detector with false
    positives. The window keeps the middle.
    """
    d = corr.shape[0]
    cand: list[tuple[int, int, float]] = []
    for i in range(d):
        for j in range(i + 1, d):
            r = float(abs(corr[i, j]))
            if r_min <= r <= r_max:
                cand.append((i, j, r))
    cand.sort(key=lambda x: -x[2])
    return cand[:top_k]


def compute_density_gaps_for_pair(
    u_i: np.ndarray,
    u_j: np.ndarray,
    *,
    n: int,
    bins: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Compute under-populated cells in the (u_i, u_j) joint histogram.

    Returns one dict per under-populated cell with keys ``u_range_i``,
    ``u_range_j``, ``observed``, ``expected``, ``p_value``, ``q_value``,
    ``is_gap`` (BH-rejected at the supplied alpha).
    """
    hist, _, _ = np.histogram2d(
        u_i, u_j, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]],
    )
    expected = n / (bins * bins)
    cells: list[dict[str, Any]] = []
    p_values: list[float] = []
    for (a, b), obs in np.ndenumerate(hist):
        if obs >= expected:
            continue  # over-populated cells are clumps, not gaps
        chi2_comp = (obs - expected) ** 2 / expected
        p = float(chi2.sf(chi2_comp, df=1))
        cells.append({
            "u_range_i": (a / bins, (a + 1) / bins),
            "u_range_j": (b / bins, (b + 1) / bins),
            "observed": int(obs),
            "expected": float(expected),
            "p_value": p,
        })
        p_values.append(p)
    if not p_values:
        return []
    rejected, q_values = benjamini_hochberg(
        np.array(p_values), alpha=alpha,
    )
    for cell, q, rej in zip(cells, q_values, rejected, strict=False):
        cell["q_value"] = float(q)
        cell["is_gap"] = bool(rej)
    return cells
