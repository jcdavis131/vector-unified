"""vector-core — the canonical MTNN building blocks shared across the vector-* fleet.

The numpy-safe API is always available on import (preproc, align, losses [numpy],
eval, schema). The torch model is exposed lazily: ``import vector_core`` works
WITHOUT torch installed. Access the model via ``vector_core.model`` (or the
re-exported names) — those raise a clear error only if you actually construct a
torch module without torch present.
"""

from __future__ import annotations

from . import align, eval, losses, preproc, schema
from .align import apply as align_apply
from .align import fit as align_fit
from .eval import purity_at_k, recall_at_k, silhouette_cosine
from .losses import info_nce_numpy, sup_con_numpy
from .preproc import RobustScaler, ple_bin_edges, ple_transform
from .schema import FleetEntry, validate_entry

__version__ = "0.1.0"

__all__ = [
    # submodules
    "preproc",
    "align",
    "losses",
    "eval",
    "schema",
    # preproc
    "RobustScaler",
    "ple_bin_edges",
    "ple_transform",
    # align
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
]


def __getattr__(name: str):
    """Lazily expose the torch model module without importing torch at package load."""
    if name in ("model", "HAS_TORCH", "MTNN", "MaskedResidualTower",
                "AttentionGatedFusion", "MultiTaskHeads", "build_mtnn"):
        import importlib

        # Use import_module (not `from . import model`) so we don't re-enter
        # this __getattr__ and recurse when the attribute isn't set yet.
        _model = importlib.import_module(".model", __name__)
        if name == "model":
            return _model
        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
