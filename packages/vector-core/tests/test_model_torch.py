import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vector_core.model import MTNN, MultiTaskHeads, build_mtnn  # noqa: E402


def test_mtnn_forward_shape_and_unit_norm():
    family_dims = [4, 6, 5]
    emb_dim = 64
    model = build_mtnn(family_dims, emb_dim=emb_dim)
    model.eval()

    batch = 8
    x = torch.randn(batch, sum(family_dims))
    z = model(x)
    assert z.shape == (batch, emb_dim)

    norms = z.norm(dim=1).detach().numpy()
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mtnn_with_mask_and_heads():
    family_dims = [3, 3]
    model = MTNN(family_dims, emb_dim=32)
    heads = MultiTaskHeads(emb_dim=32, n_archetypes=8, profile_dim=6, n_positions=5)

    x = torch.randn(5, sum(family_dims))
    mask = (torch.rand(5, sum(family_dims)) > 0.3).float()
    z = model(x, mask)
    assert z.shape == (5, 32)
    assert np.allclose(z.norm(dim=1).detach().numpy(), 1.0, atol=1e-5)

    out = heads(z)
    assert out["archetype"].shape == (5, 8)
    assert out["profile"].shape == (5, 6)
    assert out["position"].shape == (5, 5)
