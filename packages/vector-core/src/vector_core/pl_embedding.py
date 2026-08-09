"""Sports-reference periodic-linear (PL) embedding — torch ``nn.Module``.

Ported **verbatim** from the sports repos' ``realmlp_preproc.py`` so
``vector-core`` is a true drop-in. This is DISTINCT from the numpy
``ple_transform`` / ``ple_bin_edges`` in ``vector_core.preproc``:

- ``vector_core.preproc.ple_transform`` — piecewise-linear quantile encoding
  (Gorishniy et al., 2022): pure numpy, deterministic, one column per bin.
- ``vector_core.pl_embedding.PLEmbedding`` (below) — the trainable *periodic*
  linear embedding (RealMLP / FT-Transformer): sin/cos with learnable per-feature
  frequencies, projected to ``d_out``. A torch ``nn.Module`` with parameters.

Both are kept — they are different features. torch is imported lazily so
``import vector_core`` (and ``import vector_core.pl_embedding``) works WITHOUT
torch installed; only constructing ``PLEmbedding`` without torch raises.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np

__all__ = ["HAS_TORCH", "PLEmbedding"]

# Detect torch WITHOUT importing it, so module import never crashes.
HAS_TORCH = find_spec("torch") is not None


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "vector_core.pl_embedding.PLEmbedding requires torch. "
            "Install with `pip install vector-core[torch]`."
        )


def _build_pl_embedding():
    import torch
    import torch.nn as nn

    class _PLEmbedding(nn.Module):
        """
        Periodic Linear embeddings (RealMLP / FT-Transformer):
        For each scalar feature x, produce [sin(2π k x), cos(2π k x)] * linear
        k = learnable frequencies per feature.

        Input: [B, F] scalars
        Output: [B, F, d_out]
        """

        def __init__(self, num_features: int, d_out: int = 16, k: int = 8):
            super().__init__()
            self.num_features = num_features
            self.d_out = d_out
            self.k = k
            # learnable frequencies per feature per k — small init 0.1
            self.freq = nn.Parameter(torch.randn(num_features, k) * 0.1)
            self.proj = nn.Linear(2 * k, d_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, F]
            freq = self.freq.unsqueeze(0)  # [1, F, k]
            x_exp = x.unsqueeze(-1) * freq  # [B, F, k]
            sin_emb = torch.sin(2 * np.pi * x_exp)
            cos_emb = torch.cos(2 * np.pi * x_exp)
            periodic = torch.cat([sin_emb, cos_emb], dim=-1)  # [B, F, 2k]
            return self.proj(periodic)  # [B, F, d_out]

    return _PLEmbedding


if HAS_TORCH:
    PLEmbedding = _build_pl_embedding()
else:

    class PLEmbedding:  # type: ignore[no-redef]
        """Shim so ``import`` works without torch; construction raises clearly."""

        def __init__(self, *args, **kwargs):
            _require_torch()
