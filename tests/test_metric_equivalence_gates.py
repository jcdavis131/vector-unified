"""Equivalence gates for the eval-metric candidates — recording what did NOT swap.

The task's other candidates (retrieval recall@k, kNN purity, silhouette) were
gated before any swap. This file holds the runnable evidence for the retrieval
recall@k decision:

  * vector-unified's ``probe_tennis_retrieval.recall_at_k`` is pure NumPy, so it
    CAN be compared directly (unlike the sklearn-backed silhouette / kNN paths,
    which cannot even run here — sklearn is not installed).
  * It matches ``vector_core.eval.recall_at_k`` EXACTLY in the single-partition
    case, but DIVERGES as soon as there is more than one retrieval partition,
    because the repo function restricts candidates to the query's own tour while
    vector_core ranks over the whole gallery.

The real call site always partitions by tour (ATP vs WTA), so the swap would
change the reported eval number. It is therefore left untouched; this gate proves
the divergence rather than asserting it by hand.
"""

from __future__ import annotations

import numpy as np
from probe_tennis_retrieval import recall_at_k as repo_recall_at_k
from vector_core.eval import recall_at_k as vc_recall_at_k


def test_recall_at_k_matches_in_single_partition():
    """One tour + queries in row order => the two definitions are identical."""
    rng = np.random.default_rng(0)
    n, d, k = 20, 8, 5
    E = rng.standard_normal((n, d))
    tours = np.array(["ATP"] * n)  # single partition
    pairs = [(i, (i + 1) % n) for i in range(n)]  # query index == position

    repo = repo_recall_at_k(E, pairs, tours, k=k)
    targets = np.array([(i + 1) % n for i in range(n)])
    vc = vc_recall_at_k(E, E, targets, k=k, exclude_self=True)

    assert repo == vc  # exact: both are hit-fraction over the same candidate pool


def test_recall_at_k_diverges_across_partitions():
    """Two tours => repo restricts to same-tour candidates, vector_core does not."""
    # rows 0-2 are tour A, rows 3-5 are tour B.
    E = np.array(
        [
            [1.00, 0.00, 0.0],  # 0  A  (query)
            [0.90, 0.10, 0.0],  # 1  A  (its within-tour adjacent-year target)
            [0.00, 1.00, 0.0],  # 2  A
            [1.00, 0.02, 0.0],  # 3  B  (nearest overall to row 0, but WRONG tour)
            [0.00, 0.00, 1.0],  # 4  B
            [-1.0, 0.00, 0.0],  # 5  B
        ],
        dtype=np.float64,
    )
    tours = np.array(["A", "A", "A", "B", "B", "B"])
    pairs = [(0, 1)]

    # Repo: candidates limited to tour A minus self -> row 1 is nearest -> hit.
    repo = repo_recall_at_k(E, pairs, tours, k=1)
    # vector_core: query is row 0, gallery is the whole set; exclude_self drops
    # row 0, so the nearest remaining row is row 3 (tour B) and the tour-A target
    # (row 1) misses.
    vc = vc_recall_at_k(E[[0]], E, np.array([1]), k=1, exclude_self=True)

    assert repo == 1.0
    assert vc == 0.0
    assert repo != vc  # the divergence that blocks a safe swap at the real call site
