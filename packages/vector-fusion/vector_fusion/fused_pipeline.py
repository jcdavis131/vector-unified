"""
Fused pipeline builder — 3-branch ColumnTransformer per article pattern.

Article:
- How to generate text embeddings using sentence-transformers wrapped in custom sklearn transformer
- How to use ColumnTransformer to run parallel preprocessing for text, numeric, categorical
- How to assemble and evaluate complete deployment-ready classification pipeline

Fleet adaptation:
- 3 branches: text (TextEmbedder), numeric (StandardScaler), categorical (OneHotEncoder)
- Supports vector-unified bio + stats, equities 10-K + financials, schools description + demographics
- Honest remainder='drop'
"""
from __future__ import annotations

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .text_embedder import TextEmbedder


def build_fused_pipeline(
    text_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    model=None,
    text_model_name: str = "all-MiniLM-L6-v2",
    text_batch_size: int = 32,
    n_estimators: int = 100,
    random_state: int = 42,
    remainder: str = "drop",
):
    """
    Build unified sklearn pipeline that combines LLM embeddings + tabular.

    Parameters
    ----------
    text_cols: List[str]
        Column names for text (first col used if multiple? ColumnTransformer passes DataFrame slice).
        For single text column like ['message'] or ['bio'].
    numeric_cols: List[str]
        Numeric columns to scale.
    categorical_cols: List[str]
        Categorical columns to one-hot.
    model:
        Classifier/regressor instance. If None, RandomForestClassifier.
    text_model_name: str
        Passed to TextEmbedder.
    text_batch_size: int
        Batch size for embedding.
    n_estimators: int
        For default RF.
    random_state: int
        For reproducibility.
    remainder: str
        ColumnTransformer remainder policy.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    transformers = []

    if text_cols:
        transformers.append(
            ("text", TextEmbedder(model_name=text_model_name, batch_size=text_batch_size), text_cols)
        )
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    if not transformers:
        raise ValueError("At least one of text_cols, numeric_cols, categorical_cols must be non-empty")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
    )

    if model is None:
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        classifier = model

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def build_vector_unified_pipeline(
    text_col: str = "bio",
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    **kwargs,
):
    """
    Convenience for vector-unified: bio + typical stats.
    Example numeric: ['pts', 'ast', 'reb', 'age', 'account_age_days', 'priority_score']
    Example cat: ['sport', 'is_premium', 'position']
    """
    if numeric_cols is None:
        numeric_cols = ["account_age_days", "priority_score"]
    if categorical_cols is None:
        categorical_cols = ["is_premium"]
    return build_fused_pipeline(
        text_cols=[text_col],
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        **kwargs,
    )


def build_equities_pipeline(
    text_col: str = "description",
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    **kwargs,
):
    """Equities: 10-K snippet + financials."""
    if numeric_cols is None:
        numeric_cols = ["revenue", "market_cap", "pe_ratio"]
    if categorical_cols is None:
        categorical_cols = ["sector"]
    return build_fused_pipeline(
        text_cols=[text_col],
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        **kwargs,
    )
