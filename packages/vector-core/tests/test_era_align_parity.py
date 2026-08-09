"""Parity: vector_core.era_align must match the sports hoops source exactly.

We build a synthetic drift.json (chained-to-root rotations) plus season-local
z-scored vectors, then assert the port's ``load_alignment`` / ``align_batch`` /
``align_vector`` produce arrays identical (float32) to the vendored hoops
reference (``tests/_ref_era_align.py``), including the subset-shape and
identity-fallback branches. All numpy — MUST pass without torch.
"""

from __future__ import annotations

import json

import _ref_era_align as ref
import numpy as np

from vector_core import era_align as port


def _rotation(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _write_drift(tmp_path, d_full: int = 6):
    # Root season = identity; two full-D rotations; one SUBSET rotation (smaller
    # than the vector dim) to exercise the shape-mismatch subset branch.
    chained = {
        "1996-97": np.eye(d_full, dtype=np.float32).tolist(),
        "1997-98": _rotation(d_full, 1).tolist(),
        "1998-99": _rotation(d_full, 2).tolist(),
        "1999-00": _rotation(d_full - 2, 3).tolist(),  # subset (D-2 x D-2)
    }
    drift = {"method": "test", "chainedToRoot": chained}
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(drift))
    return path, drift


def test_load_alignment_from_path_parity(tmp_path):
    path, _ = _write_drift(tmp_path)
    p = port.load_alignment(path)
    r = ref.ref_load_alignment(path)
    assert p["chains"].keys() == r["chains"].keys()
    for season in r["chains"]:
        a, b = p["chains"][season], r["chains"][season]
        assert a.dtype == b.dtype == np.float32
        assert np.array_equal(a, b)


def test_load_alignment_from_dict_matches_path(tmp_path):
    path, drift = _write_drift(tmp_path)
    from_path = port.load_alignment(path)["chains"]
    from_dict = port.load_alignment(drift)["chains"]
    assert from_path.keys() == from_dict.keys()
    for s in from_path:
        assert np.array_equal(from_path[s], from_dict[s])


def test_align_batch_parity_all_branches(tmp_path):
    d_full = 6
    path, _ = _write_drift(tmp_path, d_full)
    chains = port.load_alignment(path)["chains"]
    chains_ref = ref.ref_load_alignment(path)["chains"]

    rng = np.random.default_rng(99)
    n = 40
    Z = rng.normal(size=(n, d_full)).astype(np.float32)
    seasons_pool = [
        "1996-97",   # identity
        "1997-98",   # full rotation
        "1998-99",   # full rotation
        "1999-00",   # subset rotation (D-2)
        "2050-51",   # missing -> identity fallback
    ]
    seasons = [seasons_pool[i % len(seasons_pool)] for i in range(n)]

    Zp = port.align_batch(Z, seasons, chains)
    Zr = ref.align_batch(Z, seasons, chains_ref)
    assert Zp.dtype == Zr.dtype == np.float32
    assert np.array_equal(Zp, Zr), np.abs(Zp - Zr).max()


def test_align_vector_parity():
    d = 5
    Q = _rotation(d, 7)
    v = np.random.default_rng(4).normal(size=d).astype(np.float32)
    assert np.array_equal(port.align_vector(v, Q), ref.align_vector(v, Q))
