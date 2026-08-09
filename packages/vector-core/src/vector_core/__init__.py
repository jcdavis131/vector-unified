"""vector-core — the canonical MTNN building blocks shared across the vector-* fleet.

The numpy-safe API is always available on import (preproc, align, losses [numpy],
eval, schema). The torch model is exposed lazily: ``import vector_core`` works
WITHOUT torch installed. Access the model via ``vector_core.model`` (or the
re-exported names) — those raise a clear error only if you actually construct a
torch module without torch present.
"""

from __future__ import annotations

from . import align, era_align, eval, losses, preproc, realmlp, schema
from .align import apply as align_apply
from .align import fit as align_fit
from .era_align import align_batch, align_vector, load_alignment
from .eval import purity_at_k, recall_at_k, silhouette_cosine
from .losses import info_nce_numpy, sup_con_numpy
from .preproc import RobustScaler, ple_bin_edges, ple_transform
from .realmlp import RealMLPPreprocessor, audit_current_scaling
from .schema import FleetEntry, validate_entry

__version__ = "0.1.0"

__all__ = [
    # submodules
    "preproc",
    "align",
    "losses",
    "eval",
    "schema",
    "realmlp",
    "era_align",
    # preproc (clean fleet primitives)
    "RobustScaler",
    "ple_bin_edges",
    "ple_transform",
    # realmlp (sports-reference drop-in preproc — numpy-safe)
    "RealMLPPreprocessor",
    "audit_current_scaling",
    # era_align (sports-reference alignment — apply precomputed rotations)
    "load_alignment",
    "align_vector",
    "align_batch",
    # align (rotation-only Procrustes solver)
    "align_fit",
    "align_apply",
    # losses (numpy, always safe)
    "info_nce_numpy",
    "sup_con_numpy",
    # eval
    "recall_at_k",
    "purity_at_k",
    "silhouette_cosine",
    # schema
    "FleetEntry",
    "validate_entry",
    # lazy torch model accessor
    "model",
    "HAS_TORCH",
    # lazy torch PL embedding (sports-reference)
    "pl_embedding",
    "PLEmbedding",
]


def __getattr__(name: str):
    """Lazily expose torch-dependent members without importing torch at load."""
    if name in ("model", "HAS_TORCH", "MTNN", "MaskedResidualTower",
                "AttentionGatedFusion", "MultiTaskHeads", "build_mtnn"):
        import importlib

        # Use import_module (not `from . import model`) so we don't re-enter
        # this __getattr__ and recurse when the attribute isn't set yet.
        _model = importlib.import_module(".model", __name__)
        if name == "model":
            return _model
        return getattr(_model, name)
    if name in ("pl_embedding", "PLEmbedding"):
        import importlib

        _ple = importlib.import_module(".pl_embedding", __name__)
        if name == "pl_embedding":
            return _ple
        return getattr(_ple, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
