"""Contrastive losses: InfoNCE and SupCon.

Two implementations of each:

- NumPy reference versions (``*_numpy``) — always importable, no torch. Useful
  for tests, sanity checks, and CPU-only eval.
- Torch versions (``*_torch``) — import torch lazily *inside* the function so the
  module (and the whole package) imports cleanly without torch installed.

Both InfoNCE and SupCon operate on L2-normalized embeddings and a temperature.
InfoNCE takes explicit (anchor, positive) pairs; SupCon takes a batch with
integer labels and treats all same-label rows as positives.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "info_nce_numpy",
    "sup_con_numpy",
    "info_nce_torch",
    "sup_con_torch",
]

# --------------------------------------------------------------------------- #
# NumPy reference implementations (always available)
# --------------------------------------------------------------------------- #


def _l2_normalize_np(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def info_nce_numpy(
    anchors: np.ndarray,
    positives: np.ndarray,
    temperature: float = 0.07,
    normalize: bool = True,
) -> float:
    """InfoNCE loss (NumPy reference).

    ``anchors`` and ``positives`` are ``(n, d)`` aligned rows: row ``i`` of
    positives is the positive for anchor ``i``; all other positives in the batch
    are in-batch negatives. Returns the mean cross-entropy over the batch.
    """
    A = np.asarray(anchors, dtype=np.float64)
    P = np.asarray(positives, dtype=np.float64)
    if A.shape != P.shape:
        raise ValueError("anchors and positives must have the same shape")
    if normalize:
        A = _l2_normalize_np(A)
        P = _l2_normalize_np(P)
    logits = (A @ P.T) / temperature  # (n, n)
    # log-softmax over each row; target is the diagonal.
    logits = logits - logits.max(axis=1, keepdims=True)
    log_prob = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    n = A.shape[0]
    return float(-np.mean(log_prob[np.arange(n), np.arange(n)]))


def sup_con_numpy(
    features: np.ndarray,
    labels: np.ndarray,
    temperature: float = 0.07,
    normalize: bool = True,
) -> float:
    """Supervised Contrastive loss (NumPy reference).

    ``features`` is ``(n, d)``; ``labels`` is ``(n,)`` integer class ids. For each
    anchor, positives are all *other* rows with the same label. Anchors with no
    same-label partner are skipped. Returns the mean loss over valid anchors.
    """
    F = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).ravel()
    if F.shape[0] != y.shape[0]:
        raise ValueError("features and labels length mismatch")
    if normalize:
        F = _l2_normalize_np(F)
    n = F.shape[0]
    sim = (F @ F.T) / temperature
    sim = sim - sim.max(axis=1, keepdims=True)
    exp = np.exp(sim)
    self_mask = np.eye(n, dtype=bool)
    exp[self_mask] = 0.0  # exclude self from the denominator
    denom = exp.sum(axis=1, keepdims=True)
    log_prob = sim - np.log(np.maximum(denom, 1e-12))

    pos_mask = (y[:, None] == y[None, :]) & ~self_mask
    losses = []
    for i in range(n):
        pos = pos_mask[i]
        n_pos = pos.sum()
        if n_pos == 0:
            continue
        losses.append(-log_prob[i, pos].mean())
    if not losses:
        return 0.0
    return float(np.mean(losses))


# --------------------------------------------------------------------------- #
# Torch implementations (torch imported lazily inside each function)
# --------------------------------------------------------------------------- #


def info_nce_torch(anchors, positives, temperature: float = 0.07, normalize: bool = True):
    """InfoNCE loss (torch). Returns a scalar tensor. Requires torch installed."""
    import torch
    import torch.nn.functional as F

    if normalize:
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
    logits = anchors @ positives.T / temperature
    targets = torch.arange(anchors.shape[0], device=anchors.device)
    return F.cross_entropy(logits, targets)


def sup_con_torch(features, labels, temperature: float = 0.07, normalize: bool = True):
    """Supervised Contrastive loss (torch). Returns a scalar tensor."""
    import torch
    import torch.nn.functional as F

    if normalize:
        features = F.normalize(features, dim=1)
    n = features.shape[0]
    device = features.device
    sim = features @ features.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    exp = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True).clamp_min(1e-12))

    labels = labels.view(-1)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return features.sum() * 0.0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()
