"""
vector_fusion — LLM + tabular unified sklearn pipeline
"""
from .text_embedder import TextEmbedder, DEFAULT_MODEL, SHORT_MODEL
from .fused_pipeline import (
    build_fused_pipeline,
    build_vector_unified_pipeline,
    build_equities_pipeline,
)

__all__ = [
    "TextEmbedder",
    "DEFAULT_MODEL",
    "SHORT_MODEL",
    "build_fused_pipeline",
    "build_vector_unified_pipeline",
    "build_equities_pipeline",
]

__version__ = "0.2.0"
