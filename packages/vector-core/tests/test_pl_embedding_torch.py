"""Torch-gated parity for the sports-reference PLEmbedding.

Skips entirely if torch is absent. Defines a vendored reference PLEmbedding
copied VERBATIM from vector-gridiron's realmlp_preproc.py, then asserts the port
(``vector_core.pl_embedding.PLEmbedding``), seeded identically, matches it in
forward output shape ((B, F, d_out)) and values (exactly equal).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from vector_core.pl_embedding import PLEmbedding  # noqa: E402


class _RefPLEmbedding(nn.Module):
    """Vendored verbatim from vector-gridiron/pipeline/realmlp_preproc.py."""

    def __init__(self, num_features: int, d_out: int = 16, k: int = 8):
        super().__init__()
        self.num_features = num_features
        self.d_out = d_out
        self.k = k
        self.freq = nn.Parameter(torch.randn(num_features, k) * 0.1)
        self.proj = nn.Linear(2 * k, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.freq.unsqueeze(0)
        x_exp = x.unsqueeze(-1) * freq
        sin_emb = torch.sin(2 * np.pi * x_exp)
        cos_emb = torch.cos(2 * np.pi * x_exp)
        periodic = torch.cat([sin_emb, cos_emb], dim=-1)
        return self.proj(periodic)


def test_pl_embedding_forward_shape():
    num_features, d_out, k = 5, 16, 8
    emb = PLEmbedding(num_features, d_out=d_out, k=k)
    emb.eval()
    B = 4
    x = torch.randn(B, num_features)
    with torch.no_grad():
        out = emb(x)
    assert out.shape == (B, num_features, d_out)


def test_pl_embedding_matches_reference_seeded():
    num_features, d_out, k = 6, 16, 8
    B = 8

    torch.manual_seed(1234)
    port = PLEmbedding(num_features, d_out=d_out, k=k)

    torch.manual_seed(1234)
    reference = _RefPLEmbedding(num_features, d_out=d_out, k=k)

    # Identical parameter init under the same seed.
    assert torch.equal(port.freq, reference.freq)
    assert torch.equal(port.proj.weight, reference.proj.weight)
    assert torch.equal(port.proj.bias, reference.proj.bias)

    x = torch.randn(B, num_features)
    port.eval()
    reference.eval()
    with torch.no_grad():
        out_port = port(x)
        out_ref = reference(x)
    assert out_port.shape == out_ref.shape == (B, num_features, d_out)
    assert torch.equal(out_port, out_ref)
