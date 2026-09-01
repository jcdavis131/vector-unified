# Vector Fusion

LLM embeddings + tabular features in a unified scikit-learn pipeline.

Pattern from https://machinelearningmastery.com/combining-llm-embeddings-with-tabular-features-in-a-unified-scikit-learn-pipeline/ but adapted for the vector-* fleet (20,719 player-seasons, equities, schools).

## Install

```bash
pip install -e .
# or for fleet repos:
# pip install -e ~/workspace/packages/vector-fusion
```

## Usage

```python
from vector_fusion import TextEmbedder, build_fused_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
  "bio": ["stretch big who shoots", "defensive anchor"],
  "account_age_days": [120, 800],
  "priority_score": [0.8, 0.3],
  "is_premium": ["yes","no"],
  "target": [1,0]
})

X = df[["bio","account_age_days","priority_score","is_premium"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = build_fused_pipeline(
  text_cols=["bio"],
  numeric_cols=["account_age_days","priority_score"],
  categorical_cols=["is_premium"]
)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
```

## Design

- `TextEmbedder` wraps `sentence-transformers/all-MiniLM-L6-v2` (384-d) with sklearn `BaseEstimator, TransformerMixin` compliance: model init in `fit()` for cloning safety, DataFrame/Series handling, batched encoding, honest 503 if offline.
- Fallback to `transformers.AutoModel` mean-pool (same as `pipeline/embed_cultural_text.py`) when `sentence_transformers` import is broken in env.
- `build_fused_pipeline` = 3-branch `ColumnTransformer` (text, num, cat) → classifier.

Zero synthetic data — real fleet data only. Provenance 7-field timeline mandatory.
