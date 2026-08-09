"""Sports-reference era alignment — apply precomputed Procrustes rotations.

Ported from ``vector-hoops/pipeline/era_procrustes_align.py`` so ``vector-core``
is a true drop-in. Given per-season rotation matrices chained to a root frame
(produced by the hoops ``procrustes_drift.py`` into ``drift.json``), these apply
the rotations to map season-local z-scored vectors into the shared root frame.

Distinct from ``vector_core.align`` (which *solves* a single orthogonal
Procrustes ``fit(A, B) -> R``): this module *applies* precomputed per-season
rotations with subset / identity fallback, matching the sports semantics exactly
(parity-proven in ``tests/test_era_align_parity.py``, ``float32``).

The one intentional generalization over the hoops source: ``load_alignment``
takes an explicit ``source`` (a path to ``drift.json`` or a pre-loaded dict)
instead of hardcoding repo asset paths, so any repo can pass its own location.
Everything else — chaining, shape-mismatch subset handling, identity fallback —
is preserved verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["load_alignment", "align_vector", "align_batch"]


def load_alignment(source: str | Path | dict) -> dict:
    """Load chained per-season rotations from a ``drift.json`` path or a dict.

    ``source`` may be:

    - a path (str / ``Path``) to a ``drift.json`` file, or
    - an already-parsed ``drift.json`` dict.

    Returns ``{"chains": {season: (D,D) float32 array}, "raw": <full dict>}``,
    reading the ``"chainedToRoot"`` mapping (chained rotations 1996-97 -> each
    season). Matches the hoops loader exactly apart from taking an explicit
    source instead of a hardcoded asset path.
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run procrustes_drift.py first")
        data = json.loads(path.read_text())
    chained = data.get("chainedToRoot", {})
    # chained values are list of lists -> np array per season
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
