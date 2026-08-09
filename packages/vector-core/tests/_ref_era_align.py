"""VENDORED reference copy of the sports era-alignment application.

``align_vector`` and ``align_batch`` are copied VERBATIM from
``vector-hoops/pipeline/era_procrustes_align.py`` (they are path-free and take
``chains`` as an argument). ``load_alignment`` in the hoops source hardcodes the
repo asset path; ``ref_load_alignment`` below reproduces its exact body (read
JSON, take ``chainedToRoot``, build float32 arrays) parameterized by an explicit
path, so parity can be checked against ``vector_core.era_align.load_alignment``.

Parity tests require identical arrays (float32) between these and the port.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ref_load_alignment(path) -> dict:
    data = json.loads(Path(path).read_text())
    chained = data.get("chainedToRoot", {})
    chains = {season: np.array(mat, dtype=np.float32) for season, mat in chained.items()}
    return {"chains": chains, "raw": data}


def align_vector(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Apply rotation Q: v_root = v @ Q"""
    return v @ Q


def align_batch(
    Z: np.ndarray,
    seasons: list[str],
    chains: dict[str, np.ndarray],
    feature_order: list[str] | None = None,
) -> np.ndarray:
    """
    Align batch of z-scored vectors to root frame.
    Z: [N, D] in season-local z
    seasons: list of season strings matching Z rows
    chains: season -> [D,D] rotation
    Returns Z_aligned [N,D] in root frame
    """
    Z_aligned = np.zeros_like(Z, dtype=np.float32)
    for i, season in enumerate(seasons):
        Q = chains.get(str(season))
        if Q is None:
            # fallback identity
            Q = np.eye(Z.shape[1], dtype=np.float32)
        # if Q shape mismatches (feature subset), use identity for missing
        if Q.shape[0] != Z.shape[1]:
            # For partial alignment (only GAME_FEATURES 14-d), align subset
            # Assume first len/features in Q correspond to those dims
            d = min(Q.shape[0], Z.shape[1])
            v = Z[i].copy()
            v[:d] = v[:d] @ Q[:d, :d]
            Z_aligned[i] = v
        else:
            Z_aligned[i] = Z[i] @ Q
    return Z_aligned
