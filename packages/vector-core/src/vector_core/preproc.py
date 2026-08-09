"""NumPy-only preprocessing primitives shared across the fleet.

The fleet's models standardize on a RealMLP-style RobustScaler (median / IQR,
then clip to a bounded range) rather than a plain z-score, because sports and
finance feature tables are heavy-tailed: a single outlier season should not blow
out the scale of a whole feature. An optional PLE (piecewise-linear encoding)
binning helper is provided for models that want quantile-bucketed inputs.

Everything here is pure NumPy so the package imports and runs without torch.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RobustScaler", "ple_bin_edges", "ple_transform"]


class RobustScaler:
    """RealMLP-style robust scaler: center by median, scale by IQR, clip.

    For each column ``j``::

        x' = clip((x - median_j) / iqr_j, clip_range[0], clip_range[1])

    where ``iqr_j`` is the inter-quartile range (75th - 25th percentile), guarded
    against zero. Clipping to ``[-3, 3]`` by default bounds the influence of
    extreme outliers, matching the fleet's documented input pipeline.
    """

    def __init__(self, clip_range: tuple[float, float] = (-3.0, 3.0), eps: float = 1e-8):
        self.clip_range = clip_range
        self.eps = float(eps)
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> RobustScaler:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected 2D array, got shape {X.shape}")
        self.median_ = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        # Guard degenerate (constant) columns so we never divide by ~0.
        iqr = np.where(iqr < self.eps, 1.0, iqr)
        self.iqr_ = iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("RobustScaler must be fit before transform")
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.median_) / self.iqr_
        lo, hi = self.clip_range
        return np.clip(Z, lo, hi)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def ple_bin_edges(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Compute quantile bin edges for a single feature (PLE input).

    Returns ``n_bins + 1`` monotonically non-decreasing edges. Duplicate edges
    (from ties / low cardinality) are collapsed so the encoding stays well-formed.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.nanquantile(x, qs)
    return np.unique(edges)


def ple_transform(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Piecewise-linear encoding of a single feature given bin ``edges``.

    Produces one column per bin. Within a bin the value ramps linearly from 0 to
    1; bins fully below the value are 1; bins fully above are 0. This is the
    standard PLE numeric embedding (Gorishniy et al., 2022).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    edges = np.asarray(edges, dtype=np.float64).ravel()
    if edges.size < 2:
        raise ValueError("need at least 2 edges (1 bin)")
    n_bins = edges.size - 1
    out = np.zeros((x.size, n_bins), dtype=np.float64)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        width = hi - lo if hi > lo else 1.0
        ramp = (x - lo) / width
        out[:, b] = np.clip(ramp, 0.0, 1.0)
    return out
