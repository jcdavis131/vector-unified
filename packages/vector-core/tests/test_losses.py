import numpy as np

from vector_core.losses import info_nce_numpy, sup_con_numpy


def test_info_nce_finite_and_lower_for_aligned():
    rng = np.random.default_rng(0)
    d = 16
    base = rng.normal(size=(32, d))

    # Aligned: positive is the anchor plus tiny noise -> easy -> low loss.
    aligned_pos = base + 0.01 * rng.normal(size=base.shape)
    loss_aligned = info_nce_numpy(base, aligned_pos, temperature=0.1)

    # Random positives -> hard -> higher loss.
    random_pos = rng.normal(size=base.shape)
    loss_random = info_nce_numpy(base, random_pos, temperature=0.1)

    assert np.isfinite(loss_aligned)
    assert np.isfinite(loss_random)
    assert loss_aligned < loss_random


def test_sup_con_finite_and_lower_when_clustered():
    rng = np.random.default_rng(1)
    d = 8
    # Two well-separated clusters -> low SupCon loss.
    c0 = rng.normal(loc=+3.0, scale=0.1, size=(20, d))
    c1 = rng.normal(loc=-3.0, scale=0.1, size=(20, d))
    feats_clustered = np.vstack([c0, c1])
    labels = np.array([0] * 20 + [1] * 20)
    loss_clustered = sup_con_numpy(feats_clustered, labels, temperature=0.1)

    # Same labels but scrambled features -> higher loss.
    feats_random = rng.normal(size=(40, d))
    loss_random = sup_con_numpy(feats_random, labels, temperature=0.1)

    assert np.isfinite(loss_clustered)
    assert np.isfinite(loss_random)
    assert loss_clustered < loss_random


def test_sup_con_handles_no_positives():
    feats = np.eye(3)
    labels = np.array([0, 1, 2])  # every label unique -> no positives
    assert sup_con_numpy(feats, labels) == 0.0
