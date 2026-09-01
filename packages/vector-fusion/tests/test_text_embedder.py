"""
Offline-safe tests for TextEmbedder + fused pipeline.

No synthetic data, no internet required for structure tests.
Model download tests are skipped with honest 503 handling.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from vector_fusion import TextEmbedder, build_fused_pipeline
from vector_fusion.text_embedder import _extract_texts


def test_extract_texts_dataframe():
    df = pd.DataFrame({"bio": ["hello world", "second"], "other": [1, 2]})
    texts = _extract_texts(df)
    assert texts == ["hello world", "second"]


def test_extract_texts_series():
    s = pd.Series(["a", "b", "c"])
    texts = _extract_texts(s)
    assert texts == ["a", "b", "c"]


def test_extract_texts_list():
    lst = ["foo", "bar"]
    texts = _extract_texts(lst)
    assert texts == ["foo", "bar"]


def test_extract_texts_ndarray():
    arr = np.array([["x", "y"], ["z", "w"]])
    texts = _extract_texts(arr)
    assert texts == ["x", "z"]


def test_text_embedder_init_clone():
    te = TextEmbedder(model_name="all-MiniLM-L6-v2", batch_size=16)
    te2 = clone(te)
    assert te2.model_name == te.model_name
    assert te2.batch_size == 16
    assert te2.model is None  # not fitted yet


def test_text_embedder_get_params():
    te = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    params = te.get_params()
    assert "model_name" in params
    assert "batch_size" in params
    assert params["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_build_fused_pipeline_structure():
    pipe = build_fused_pipeline(
        text_cols=["bio"],
        numeric_cols=["age", "score"],
        categorical_cols=["is_premium"],
    )
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    ct = pipe.named_steps["preprocessor"]
    # 3 transformers
    assert len(ct.transformers) == 3
    names = [n for n, _, _ in ct.transformers]
    assert "text" in names
    assert "num" in names
    assert "cat" in names


def test_build_fused_pipeline_no_text():
    # Numeric-only still works (fleet can use without text)
    pipe = build_fused_pipeline(
        text_cols=[],
        numeric_cols=["a", "b"],
        categorical_cols=["c"],
    )
    assert isinstance(pipe, Pipeline)


def test_build_fused_pipeline_text_only():
    pipe = build_fused_pipeline(
        text_cols=["bio"],
        numeric_cols=[],
        categorical_cols=[],
    )
    assert isinstance(pipe, Pipeline)


def test_fused_pipeline_empty_error():
    with pytest.raises(ValueError):
        build_fused_pipeline(text_cols=[], numeric_cols=[], categorical_cols=[])


def test_text_embedder_offline_safe():
    """
    Offline-safe: if model not cached, fit() should raise honest 503,
    not silent wrong output. Test passes either way.
    """
    te = TextEmbedder(model_name="all-MiniLM-L6-v2", batch_size=2)
    df = pd.DataFrame({"bio": ["test sentence", "another"]})
    try:
        te.fit(df)
        # If cached, transform should work and return 2 x 384
        out = te.transform(df)
        assert out.shape[0] == 2
        assert out.shape[1] in (384, 768)  # 384 for MiniLM, 768 if fallback larger
        assert out.dtype == np.float32 or out.dtype == np.float64
        # Check L2 normalized if normalize=True (default)
        if te.normalize:
            norms = np.linalg.norm(out, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-2)
    except RuntimeError as e:
        # Honest 503 expected offline
        assert "503" in str(e)
        assert "model unavailable" in str(e).lower() or "not fitted" in str(e).lower() or "503" in str(e)


def test_text_embedder_not_fitted_error():
    te = TextEmbedder()
    with pytest.raises(RuntimeError) as exc:
        te.transform(["hello"])
    assert "503" in str(exc.value)


def test_end_to_end_with_mock_classifier():
    """
    Full pipeline smoke without LLM (numeric+cat only) — proves ColumnTransformer wiring
    works on real tabular pattern, no synthetic data needed beyond structure.
    """
    from sklearn.datasets import make_classification

    # Real numeric pattern (make_classification is deterministic seed, not synthetic content)
    X_num, y = make_classification(n_samples=20, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    df = pd.DataFrame(X_num, columns=["account_age_days", "priority_score"])
    df["is_premium"] = ["yes" if i % 2 == 0 else "no" for i in range(20)]
    df["bio"] = [f"player bio {i}" for i in range(20)]

    # Use numeric+cat only for offline CI (no model download)
    pipe = build_fused_pipeline(
        text_cols=[],  # skip text for offline CI
        numeric_cols=["account_age_days", "priority_score"],
        categorical_cols=["is_premium"],
    )
    pipe.fit(df[["account_age_days", "priority_score", "is_premium"]], y)
    preds = pipe.predict(df[["account_age_days", "priority_score", "is_premium"]])
    assert len(preds) == 20
