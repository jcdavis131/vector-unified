"""
Vector Fusion — LLM text embedder as sklearn transformer.

Adapted from MachineLearningMastery pattern:
https://machinelearningmastery.com/combining-llm-embeddings-with-tabular-features-in-a-unified-scikit-learn-pipeline/

Key differences for fleet:
- Handles DataFrame/Series/list consistently
- Lazy model init in fit() per sklearn cloning rules (no model in __init__ args)
- Honest 503 if model unavailable offline
- Fallback to transformers AutoModel mean-pool (embed_cultural_text.py pattern) when sentence_transformers broken
- Batched, normalized, CPU/GPU auto
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SHORT_MODEL = "all-MiniLM-L6-v2"


def _extract_texts(X) -> List[str]:
    """Extract first column as list of strings, handling DataFrame/Series/list."""
    if isinstance(X, pd.DataFrame):
        # Article pattern: X.iloc[:,0].astype(str).tolist()
        if X.shape[1] == 0:
            return []
        return X.iloc[:, 0].astype(str).tolist()
    elif isinstance(X, pd.Series):
        return X.astype(str).tolist()
    elif isinstance(X, np.ndarray):
        # 2D array: take first column
        if X.ndim == 2:
            return pd.Series(X[:, 0]).astype(str).tolist()
        return pd.Series(X).astype(str).tolist()
    else:
        # list, tuple, etc.
        try:
            return pd.Series(list(X)).astype(str).tolist()
        except Exception:
            return [str(x) for x in X]


def _mean_pool(last_hidden, attention_mask):
    """Mean-pool for transformers fallback, same as embed_cultural_text.py."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TextEmbedder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible text embedder wrapping sentence-transformers.

    Parameters
    ----------
    model_name: str
        HF model id. Default 'all-MiniLM-L6-v2' (short) or 'sentence-transformers/all-MiniLM-L6-v2' (full).
        Both resolve to 384-d.
    batch_size: int
        Encoding batch size.
    device: Optional[str]
        'cuda' or 'cpu' or None (auto).
    normalize: bool
        L2-normalize embeddings (True for cosine-sim retrieval).
    max_length: int
        Truncation length for fallback tokenizer.
    """

    def __init__(
        self,
        model_name: str = SHORT_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize: bool = True,
        max_length: int = 256,
    ):
        # Only simple params here for sklearn.clone() safety
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.max_length = max_length
        # Non-param state, set in fit()
        self.model = None
        self.tokenizer = None
        self._backend = None
        self._dim = None

    def _resolve_device(self):
        if self.device:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def fit(self, X, y=None):
        """
        Init model in fit() to comply with sklearn cloning rules.
        Article pattern: if self.model is None: self.model = SentenceTransformer(self.model_name)
        """
        if self.model is not None:
            return self

        device = self._resolve_device()

        # Try sentence-transformers first (article's choice)
        try:
            from sentence_transformers import SentenceTransformer

            # Handle both short and full names
            name = self.model_name
            # sentence_transformers accepts both; prefer full for cache hit
            if name == SHORT_MODEL:
                name = DEFAULT_MODEL
            self.model = SentenceTransformer(name, device=device)
            self._backend = "sentence_transformers"
            # Probe dim
            self._dim = self.model.get_sentence_embedding_dimension()
            return self
        except Exception as e_st:
            # Fallback to transformers AutoModel mean-pool (fleet's existing pattern)
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                name = self.model_name
                if name == SHORT_MODEL:
                    name = DEFAULT_MODEL
                tok = AutoTokenizer.from_pretrained(name)
                mdl = AutoModel.from_pretrained(name).to(device)
                mdl.eval()
                self.tokenizer = tok
                self.model = mdl
                self._backend = "transformers"
                self._dim = mdl.config.hidden_size  # 384 for MiniLM
                return self
            except Exception as e_tr:
                # Honest 503 — no internet / no cached model / broken env
                raise RuntimeError(
                    f"503: TextEmbedder model unavailable offline (tried sentence_transformers and transformers). "
                    f"model_name={self.model_name} device={device} "
                    f"st_error={str(e_st)[:200]} tr_error={str(e_tr)[:200]}. "
                    f"Run with internet once to cache, or mount fleet cache."
                ) from e_tr

    def transform(self, X, y=None):
        if self.model is None:
            raise RuntimeError(
                "503: TextEmbedder not fitted. Call fit() first. "
                "If offline and model not cached, this is expected — honest 503, not silent failure."
            )

        texts = _extract_texts(X)
        if len(texts) == 0:
            dim = self._dim or 384
            return np.empty((0, dim), dtype=np.float32)

        if self._backend == "sentence_transformers":
            # Batched, normalized if requested
            embs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
            # Ensure float32
            return np.asarray(embs, dtype=np.float32)

        elif self._backend == "transformers":
            import torch
            import torch.nn.functional as F

            device = self._resolve_device()
            chunks = []
            # Batch loop
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                # Truncate to 1200 chars like fleet does (safety)
                batch = [(t or "")[:1200] for t in batch]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    out = self.model(**enc)
                    emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
                    if self.normalize:
                        emb = F.normalize(emb, p=2, dim=1)
                    chunks.append(emb.cpu().numpy().astype(np.float32))
            if not chunks:
                dim = self._dim or 384
                return np.empty((0, dim), dtype=np.float32)
            return np.concatenate(chunks, axis=0)

        else:
            raise RuntimeError(f"503: Unknown backend {self._backend}")

    def get_feature_names_out(self, input_features=None):
        # sklearn 1.0+ API
        dim = self._dim or 384
        return np.array([f"text_emb_{i}" for i in range(dim)], dtype=object)
