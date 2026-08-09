"""Torch MTNN model — the canonical architecture the vector-* repos share.

Mirrors the fleet's documented design (see ``docs/UNIFIED_ARCHITECTURE.md``):

- ``MaskedResidualTower`` — a per-family residual MLP encoder whose input is the
  masked feature block concatenated with its own missingness mask,
  ``cat([x * m, m])``, so the tower sees *which* inputs were present.
- ``AttentionGatedFusion`` — combines the per-tower embeddings with a learned
  attention gate (the gated-fusion variant used by gridiron/pitch), as an
  alternative to naive concat.
- ``MTNN`` — towers -> fusion -> projection trunk -> L2-normalized embedding.
- ``MultiTaskHeads`` — the small head bundle: archetype classification (CE),
  profile reconstruction, and position classification.

torch is imported **lazily at call time**, guarded at module load, so that
``import vector_core`` (and even ``import vector_core.model``) succeeds without
torch installed. ``HAS_TORCH`` reports availability; the class factories raise a
clear error if torch is missing.
"""

from __future__ import annotations

from importlib.util import find_spec

__all__ = [
    "HAS_TORCH",
    "MaskedResidualTower",
    "AttentionGatedFusion",
    "MTNN",
    "MultiTaskHeads",
    "build_mtnn",
]

# Detect torch WITHOUT importing it, so module import never crashes.
HAS_TORCH = find_spec("torch") is not None


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "vector_core.model requires torch. Install with `pip install vector-core[torch]`."
        )
    import torch  # noqa: F401

    return torch


# The real nn.Module classes are defined inside a factory that runs only when
# torch is present. At module load (no torch) the public names point at thin
# shims that raise on instantiation but keep `import` working.


def _build_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _MaskedResidualTower(nn.Module):
        """Per-family residual MLP. Input is ``cat([x * m, m])`` of width 2*in_dim."""

        def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32, depth: int = 2):
            super().__init__()
            self.in_dim = in_dim
            self.inp = nn.Linear(in_dim * 2, hidden)
            self.blocks = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(depth)]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
            self.out = nn.Linear(hidden, out_dim)

        def forward(self, x, mask=None):
            if mask is None:
                mask = torch.ones_like(x)
            h = self.inp(torch.cat([x * mask, mask], dim=-1))
            h = F.gelu(h)
            for blk, norm in zip(self.blocks, self.norms, strict=False):
                h = h + F.gelu(blk(norm(h)))
            return self.out(h)

    class _AttentionGatedFusion(nn.Module):
        """Attention-gated fusion of ``n_towers`` embeddings each of dim ``tower_dim``."""

        def __init__(self, n_towers: int, tower_dim: int, out_dim: int):
            super().__init__()
            self.n_towers = n_towers
            self.tower_dim = tower_dim
            self.gate = nn.Linear(tower_dim, 1)
            self.proj = nn.Linear(tower_dim, out_dim)

        def forward(self, tower_embs):
            # tower_embs: (B, n_towers, tower_dim)
            scores = self.gate(tower_embs)  # (B, n_towers, 1)
            weights = torch.softmax(scores, dim=1)
            fused = (weights * tower_embs).sum(dim=1)  # (B, tower_dim)
            return self.proj(fused)

    class _MTNN(nn.Module):
        """Multi-Tower Neural Net -> L2-normalized embedding.

        ``family_dims`` gives the input width of each per-family block. The forward
        pass takes a list of ``(x, mask)`` per family (or a single concatenated
        tensor split by ``family_dims``) and returns a unit-norm embedding.
        """

        def __init__(
            self,
            family_dims: list[int],
            tower_dim: int = 32,
            emb_dim: int = 64,
            tower_hidden: int = 64,
        ):
            super().__init__()
            self.family_dims = list(family_dims)
            self.tower_dim = tower_dim
            self.emb_dim = emb_dim
            self.towers = nn.ModuleList(
                [
                    _MaskedResidualTower(d, hidden=tower_hidden, out_dim=tower_dim)
                    for d in family_dims
                ]
            )
            self.fusion = _AttentionGatedFusion(len(family_dims), tower_dim, emb_dim)

        def _split(self, x):
            out, i = [], 0
            for d in self.family_dims:
                out.append(x[:, i : i + d])
                i += d
            return out

        def forward(self, x, mask=None):
            """``x``: (B, sum(family_dims)) tensor, or a list of per-family tensors."""
            if isinstance(x, (list, tuple)):
                blocks = list(x)
                masks = mask if mask is not None else [None] * len(blocks)
            else:
                blocks = self._split(x)
                masks = self._split(mask) if mask is not None else [None] * len(blocks)
            embs = [
                tower(b, m)
                for tower, b, m in zip(self.towers, blocks, masks, strict=False)
            ]
            stacked = torch.stack(embs, dim=1)  # (B, n_towers, tower_dim)
            z = self.fusion(stacked)  # (B, emb_dim)
            return F.normalize(z, dim=-1)

    class _MultiTaskHeads(nn.Module):
        """Small multi-task head bundle on top of the embedding.

        - archetype: linear -> logits over ``n_archetypes`` (train with CE)
        - profile:   linear -> reconstruct ``profile_dim`` features (train with MSE)
        - position:  linear -> logits over ``n_positions`` (train with CE)
        """

        def __init__(
            self,
            emb_dim: int,
            n_archetypes: int = 8,
            profile_dim: int = 16,
            n_positions: int = 5,
        ):
            super().__init__()
            self.archetype = nn.Linear(emb_dim, n_archetypes)
            self.profile = nn.Linear(emb_dim, profile_dim)
            self.position = nn.Linear(emb_dim, n_positions)

        def forward(self, z):
            return {
                "archetype": self.archetype(z),
                "profile": self.profile(z),
                "position": self.position(z),
            }

    return _MaskedResidualTower, _AttentionGatedFusion, _MTNN, _MultiTaskHeads


if HAS_TORCH:
    (
        MaskedResidualTower,
        AttentionGatedFusion,
        MTNN,
        MultiTaskHeads,
    ) = _build_classes()
else:

    class _TorchMissing:
        def __init__(self, *args, **kwargs):
            _require_torch()

    class MaskedResidualTower(_TorchMissing):  # type: ignore[no-redef]
        pass

    class AttentionGatedFusion(_TorchMissing):  # type: ignore[no-redef]
        pass

    class MTNN(_TorchMissing):  # type: ignore[no-redef]
        pass

    class MultiTaskHeads(_TorchMissing):  # type: ignore[no-redef]
        pass


def build_mtnn(family_dims, emb_dim: int = 64, tower_dim: int = 32):
    """Convenience factory returning an ``MTNN``. Raises if torch is absent."""
    _require_torch()
    return MTNN(family_dims, tower_dim=tower_dim, emb_dim=emb_dim)
