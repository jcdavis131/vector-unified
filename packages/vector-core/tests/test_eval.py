import numpy as np

from vector_core.eval import purity_at_k, recall_at_k, silhouette_cosine


def test_recall_at_k_known_answer():
    # 4 gallery vectors along axes; each query is a noisy copy of one of them.
    gallery = np.eye(4)
    queries = gallery + 0.01 * np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    targets = np.array([0, 1, 2, 3])
    # Each query's nearest gallery item is its own target.
    assert recall_at_k(queries, gallery, targets, k=1) == 1.0

    # A deliberately wrong target set -> recall@1 should drop below 1.
    wrong = np.array([1, 2, 3, 0])
    assert recall_at_k(queries, gallery, wrong, k=1) < 1.0
    # But recall@4 (full gallery) is always 1.
    assert recall_at_k(queries, gallery, wrong, k=4) == 1.0


def test_purity_at_k_perfect_and_mixed():
    # Two tight, well-separated clusters -> purity@k == 1.
    c0 = np.random.default_rng(0).normal(loc=+5, scale=0.01, size=(10, 3))
    c1 = np.random.default_rng(1).normal(loc=-5, scale=0.01, size=(10, 3))
    emb = np.vstack([c0, c1])
    labels = np.array([0] * 10 + [1] * 10)
    assert purity_at_k(emb, labels, k=5) == 1.0

    # Random embeddings with balanced labels -> purity well below 1.
    rng = np.random.default_rng(2)
    emb_rand = rng.normal(size=(40, 5))
    lab = np.array([0, 1] * 20)
    assert purity_at_k(emb_rand, lab, k=5) < 0.9


def test_silhouette_cosine_range_and_sign():
    c0 = np.random.default_rng(3).normal(loc=+5, scale=0.05, size=(15, 4))
    c1 = np.random.default_rng(4).normal(loc=-5, scale=0.05, size=(15, 4))
    emb = np.vstack([c0, c1])
    labels = np.array([0] * 15 + [1] * 15)
    s = silhouette_cosine(emb, labels)
    assert -1.0 <= s <= 1.0
    assert s > 0.5  # clean clusters score high
