"""Pytest bootstrap for vector-unified.

Makes the in-repo ``vector_core`` package (``packages/vector-core/src``) importable
during tests without requiring an install step, so ``from vector_core import ...``
resolves CPU-only. Installing the package the clean way still works and takes
precedence — see requirements-dev.txt (``pip install -e packages/vector-core``);
this shim is a no-op when the package is already importable.

The ``pipeline/`` directory is also added to ``sys.path`` because the pipeline
modules import each other by bare module name (``from train_unified import ...``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# 1) vector_core: prefer an installed copy; otherwise resolve the in-repo source.
if importlib.util.find_spec("vector_core") is None:
    _vc_src = _ROOT / "packages" / "vector-core" / "src"
    if _vc_src.is_dir():
        sys.path.insert(0, str(_vc_src))

# 2) pipeline modules import each other by bare name.
_pipeline = _ROOT / "pipeline"
if _pipeline.is_dir() and str(_pipeline) not in sys.path:
    sys.path.insert(0, str(_pipeline))
