import numpy as np

from vector_core.align import apply, fit


def _random_rotation(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1  # make it a proper rotation
    return Q


def test_recovers_known_rotation():
    d = 6
    rng = np.random.default_rng(42)
    A = rng.normal(size=(300, d))
    R_true = _random_rotation(d, seed=7)
    B = A @ R_true

    R_est = fit(A, B)
    assert np.linalg.norm(R_est - R_true) < 1e-5
    assert np.linalg.norm(apply(A, R_est) - B) < 1e-5


def test_rotation_is_orthogonal_and_proper():
    d = 5
    rng = np.random.default_rng(3)
    A = rng.normal(size=(100, d))
    B = rng.normal(size=(100, d))
    R = fit(A, B)
    assert np.linalg.norm(R.T @ R - np.eye(d)) < 1e-9
    assert np.linalg.det(R) > 0  # rotation-only (no reflection)
