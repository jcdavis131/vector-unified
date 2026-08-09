import numpy as np

from vector_core.preproc import RobustScaler, ple_bin_edges, ple_transform


def test_robust_scaler_shape_and_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    Z = RobustScaler().fit_transform(X)
    assert Z.shape == X.shape
    assert Z.min() >= -3.0 - 1e-9
    assert Z.max() <= 3.0 + 1e-9


def test_robust_scaler_resists_outliers():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 1))
    scaler = RobustScaler().fit(X)
    med_clean, iqr_clean = scaler.median_.copy(), scaler.iqr_.copy()

    # Inject extreme outliers; median/IQR should barely move (robustness).
    X2 = X.copy()
    X2[:5, 0] = 1e6
    scaler2 = RobustScaler().fit(X2)
    assert abs(scaler2.median_[0] - med_clean[0]) < 0.2
    assert abs(scaler2.iqr_[0] - iqr_clean[0]) < 0.2

    # And the transformed outliers are clipped, not exploding the scale.
    Z2 = scaler2.transform(X2)
    assert Z2.max() <= 3.0 + 1e-9


def test_robust_scaler_constant_column_safe():
    X = np.ones((10, 2))
    Z = RobustScaler().fit_transform(X)
    assert np.all(np.isfinite(Z))


def test_ple_transform_shape_and_bounds():
    x = np.linspace(0, 10, 50)
    edges = ple_bin_edges(x, n_bins=4)
    out = ple_transform(x, edges)
    assert out.shape[0] == 50
    assert out.shape[1] == edges.size - 1
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Smallest value: all ramps ~0; largest value: all ramps ~1.
    assert out[0].sum() < out[-1].sum()
