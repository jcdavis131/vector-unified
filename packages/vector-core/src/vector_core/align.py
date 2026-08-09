"""NumPy rotation-only orthogonal Procrustes alignment.

Used across the fleet to align one embedding space onto another (e.g. aligning a
re-trained encoder's output onto the previously-shipped geometry, or aligning two
per-sport spaces before comparison) without rescaling or reflecting. Rotation
only means the solution is a proper orthogonal matrix (``det(R) = +1``), so
distances and norms are preserved.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit", "apply"]


def fit(A: np.ndarray, B: np.ndarray, allow_reflection: bool = False) -> np.ndarray:
    """Solve for R minimizing ||A @ R - B||_F over orthogonal R.

    ``A`` and ``B`` are ``(n, d)`` arrays of corresponding rows. Returns the
    ``(d, d)`` rotation matrix ``R`` such that ``A @ R`` best matches ``B``.

    By default the solution is constrained to a pure rotation (no reflection):
    if the raw SVD solution has ``det < 0`` the last singular direction is
    flipped. Set ``allow_reflection=True`` to permit the full orthogonal group.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape, got {A.shape} vs {B.shape}")
    if A.ndim != 2:
        raise ValueError("A and B must be 2D (n, d)")

    # Cross-covariance; R = U V^T from its SVD is the orthogonal Procrustes soln.
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        # Flip the least-significant singular direction to force det(+1).
        U[:, -1] *= -1
        R = U @ Vt
    return R


def apply(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply a rotation ``R`` (from :func:`fit`) to rows of ``X``: returns ``X @ R``."""
    X = np.asarray(X, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if X.shape[-1] != R.shape[0]:
        raise ValueError(f"dim mismatch: X has {X.shape[-1]} cols, R is {R.shape}")
    return X @ R
