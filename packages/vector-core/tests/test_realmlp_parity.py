"""Parity: vector_core.realmlp must be byte-identical to the sports source.

This is the zero-behavior-change guarantee. We run the ported RealMLPPreprocessor
/ RobustScaler / audit_current_scaling AND the vendored sports reference
(``tests/_ref_realmlp.py``, copied verbatim from vector-gridiron) on the SAME
seeded fixture (NaNs + mask + multiple seasons) and assert the outputs are
identical: max abs diff exactly 0.0 and the same float32 dtype.

All numpy — MUST pass without torch.
"""

from __future__ import annotations

import _ref_realmlp as ref
import numpy as np

from vector_core import realmlp as port


def _fixture(seed: int = 20240607):
    rng = np.random.default_rng(seed)
    n, d = 240, 7
    feature_names = [f"feat_{j}" for j in range(d)]
    # Multiple seasons, uneven sizes (some seasons < 10 rows to exercise the
    # <10-valid skip rule inside per-season fit).
    seasons_pool = ["2019-20", "2020-21", "2021-22", "2022-23"]
    seasons = [seasons_pool[rng.integers(0, len(seasons_pool))] for _ in range(n)]
    # Add a tiny season with only a handful of rows.
    seasons[:6] = ["1996-97"] * 6

    Z = rng.normal(loc=2.0, scale=5.0, size=(n, d)).astype(np.float32)
    # Heavy tails / outliers.
    Z[rng.integers(0, n, size=15), rng.integers(0, d, size=15)] = 1e5
    # NaNs scattered in.
    nan_idx = (rng.integers(0, n, size=25), rng.integers(0, d, size=25))
    Z[nan_idx] = np.nan
    # Mask: 0 where NaN plus some extra era-missing families.
    mask = (~np.isnan(Z)).astype(np.float32)
    mask[rng.random((n, d)) < 0.1] = 0.0
    # Replace NaN with 0 in the value array (mask carries validity), matching how
    # the sports pipeline feeds masked blocks.
    Z = np.nan_to_num(Z, nan=0.0).astype(np.float32)
    return Z, seasons, mask, feature_names


def _assert_identical(a: np.ndarray, b: np.ndarray):
    assert a.dtype == b.dtype == np.float32, (a.dtype, b.dtype)
    assert a.shape == b.shape
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
    assert np.array_equal(a, b), f"max abs diff = {diff}"


def test_robust_scaler_parity_with_mask():
    Z, _seasons, mask, _names = _fixture()
    p = port.RobustScaler(clip=3.0).fit_transform(Z, mask)
    r = ref.RobustScaler(clip=3.0).fit_transform(Z, mask)
    _assert_identical(p, r)
    # And the fitted stats themselves.
    ps = port.RobustScaler(clip=3.0).fit(Z, mask)
    rs = ref.RobustScaler(clip=3.0).fit(Z, mask)
    _assert_identical(ps.median_, rs.median_)
    _assert_identical(ps.iqr_, rs.iqr_)


def test_realmlp_preprocessor_by_season_parity():
    Z, seasons, mask, names = _fixture()
    p = port.RealMLPPreprocessor(names, clip=3.0)
    r = ref.RealMLPPreprocessor(names, clip=3.0)
    Zp = p.fit_transform(Z, seasons, mask, by_season=True)
    Zr = r.fit_transform(Z, seasons, mask, by_season=True)
    _assert_identical(Zp, Zr)


def test_realmlp_preprocessor_global_transform_parity():
    Z, seasons, mask, names = _fixture()
    p = port.RealMLPPreprocessor(names, clip=3.0).fit(Z, seasons, mask, by_season=True)
    r = ref.RealMLPPreprocessor(names, clip=3.0).fit(Z, seasons, mask, by_season=True)
    # seasons=None -> global scaler path
    _assert_identical(p.transform(Z), r.transform(Z))


def test_realmlp_unseen_season_falls_back_to_global_identically():
    Z, seasons, mask, names = _fixture()
    p = port.RealMLPPreprocessor(names, clip=3.0).fit(Z, seasons, mask, by_season=True)
    r = ref.RealMLPPreprocessor(names, clip=3.0).fit(Z, seasons, mask, by_season=True)
    unseen = ["2099-00"] * Z.shape[0]
    _assert_identical(p.transform(Z, unseen), r.transform(Z, unseen))


def test_realmlp_save_load_roundtrip_parity(tmp_path):
    Z, seasons, mask, names = _fixture()
    p = port.RealMLPPreprocessor(names, clip=3.0).fit(Z, seasons, mask, by_season=True)
    saved = tmp_path / "preproc.json"
    p.save(saved)
    p2 = port.RealMLPPreprocessor.load(saved)
    _assert_identical(p2.transform(Z, seasons), p.transform(Z, seasons))


def test_audit_current_scaling_parity():
    Z, _seasons, _mask, names = _fixture()
    manifest = {"features": names}
    dp = port.audit_current_scaling(Z, manifest)
    dr = ref.audit_current_scaling(Z, manifest)
    assert dp == dr
