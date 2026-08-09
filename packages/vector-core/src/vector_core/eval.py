"""NumPy retrieval / clustering evaluation metrics.

These are the metrics the fleet's eval scoreboards report: recall@k on identity or
adjacent-period retrieval, kNN label purity@k, and cosine silhouette. All pure
NumPy so they run in CI without torch.

Convention: embeddings are compared by cosine similarity. Rows are L2-normalized
internally, so callers can pass raw embeddings.
"""

from __future__ import annotations

import numpy as np

__all__ = ["recall_at_k", "purity_at_k", "silhouette_cosine"]


def _l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _cosine_sim(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return _l2_normalize(X) @ _l2_normalize(Y).T


def recall_at_k(
    queries: np.ndarray,
    gallery: np.ndarray,
    targets: np.ndarray,
    k: int = 10,
    exclude_self: bool = False,
) -> float:
    """Fraction of queries whose target index is within the top-k gallery neighbors.

    ``queries`` ``(nq, d)``, ``gallery`` ``(ng, d)``, ``targets`` ``(nq,)`` giving
    the gallery index that is the correct match for each query. If queries and
    gallery are the same set, pass ``exclude_self=True`` to drop the query's own
    row from its ranking.
    """
    queries = np.asarray(queries, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)
    targets = np.asarray(targets).ravel()
    if queries.shape[0] != targets.shape[0]:
        raise ValueError("queries and targets length mismatch")
    if k < 1:
        raise ValueError("k must be >= 1")

    sim = _cosine_sim(queries, gallery)  # (nq, ng)
    if exclude_self:
        n = min(sim.shape[0], sim.shape[1])
        sim[np.arange(n), np.arange(n)] = -np.inf

    kk = min(k, sim.shape[1])
    # Indices of the top-kk gallery items per query.
    topk = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(hits.mean())


def purity_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """Mean kNN label purity@k.

    For each row, look at its ``k`` nearest neighbors (excluding itself) and
    compute the fraction that share its label; average over all rows.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    n = embeddings.shape[0]
    if n != labels.shape[0]:
        raise ValueError("embeddings and labels length mismatch")
    if k < 1:
        raise ValueError("k must be >= 1")

    sim = _cosine_sim(embeddings, embeddings)
    np.fill_diagonal(sim, -np.inf)  # never a neighbor of itself
    kk = min(k, n - 1)
    if kk < 1:
        return 0.0
    topk = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
    neigh_labels = labels[topk]
    same = neigh_labels == labels[:, None]
    return float(same.mean())


def silhouette_cosine(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score using cosine distance.

    For each point: ``a`` = mean cosine distance to same-cluster points, ``b`` =
    min over other clusters of the mean cosine distance to that cluster;
    silhouette ``= (b - a) / max(a, b)``. Singletons contribute 0. Returns the
    mean over all points; range ``[-1, 1]``.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    n = embeddings.shape[0]
    if n != labels.shape[0]:
        raise ValueError("embeddings and labels length mismatch")
    uniq = np.unique(labels)
    if uniq.size < 2:
        raise ValueError("silhouette needs at least 2 clusters")

    dist = 1.0 - _cosine_sim(embeddings, embeddings)  # cosine distance
    np.clip(dist, 0.0, 2.0, out=dist)

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        n_same = same.sum()
        if n_same == 0:
            scores[i] = 0.0  # singleton cluster
            continue
        a = dist[i, same].mean()
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = labels == c
            b = min(b, dist[i, other].mean())
        denom = max(a, b)
        scores[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(scores.mean())
