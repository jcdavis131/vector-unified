# vector-core

The canonical MTNN building blocks shared across the `vector-*` fleet
(`vector-hoops`, `vector-pitch`, `vector-gridiron`, `vector-equities`,
`vector-realty`, `vector-unified`). Each of those repos grew its own near-copy of
the same preprocessing, alignment, contrastive-loss, evaluation, and model code.
`vector-core` is the deduplicated, tested reference so the estate has one source of
truth for the parts that are genuinely the same everywhere.

## Why it exists

Five model repos independently re-implemented:

- **RealMLP-style robust scaling** (median / IQR, clip to `[-3, 3]`) as the input
  transform.
- **Orthogonal Procrustes alignment** (rotation-only, SVD) to line one embedding
  space up with another.
- **InfoNCE / SupCon** contrastive losses.
- **Retrieval / clustering metrics** — recall@k, kNN purity@k, cosine silhouette —
  that populate every repo's eval scoreboard.
- **The MTNN itself** — masked residual towers over feature families, gated
  fusion, an L2-normalized embedding, and a small multi-task head bundle.

Duplication meant a fix or improvement in one repo silently diverged from the
others. This package standardizes those pieces.

## What it standardizes

| Module | Contents | torch? |
|---|---|---|
| `vector_core.preproc` | `RobustScaler`, `ple_bin_edges`, `ple_transform` | no |
| `vector_core.realmlp` | `RealMLPPreprocessor`, `RobustScaler`, `audit_current_scaling` (sports-reference drop-in) | no |
| `vector_core.align` | `fit(A, B) -> R`, `apply(X, R)` (rotation-only Procrustes) | no |
| `vector_core.era_align` | `load_alignment`, `align_vector`, `align_batch` (apply precomputed rotations) | no |
| `vector_core.pl_embedding` | `PLEmbedding` (periodic sin/cos, trainable freqs — sports-reference) | required |
| `vector_core.losses` | `info_nce_numpy`, `sup_con_numpy` (+ `*_torch` variants) | lazy |
| `vector_core.eval` | `recall_at_k`, `purity_at_k`, `silhouette_cosine` | no |
| `vector_core.schema` | `FleetEntry`, `validate_entry` (fleet-report entry shape) | no |
| `vector_core.model` | `MTNN`, `MaskedResidualTower`, `AttentionGatedFusion`, `MultiTaskHeads` | required |

### Drop-in superset for the sports repos (`vector-realmlp` / `vector-era_align`)

`vector_core.realmlp`, `vector_core.era_align`, and `vector_core.pl_embedding`
are **exact, parity-proven ports** of the sports repos' shared utilities
(`vector-hoops`, `vector-gridiron`). They let those repos adopt `vector-core`
with **zero behavior change** — no model re-validation — because their numerics
are byte-identical to the originals (parity tests assert max abs diff `0.0`,
same `float32` dtype).

They are deliberately kept SEPARATE from the cleaner primitives so both survive:

- **`preproc.RobustScaler`** (clean, new work): float64, IQR guarded against zero
  columns, no per-season logic. **`realmlp.RobustScaler`** (sports-reference):
  float32 output, `(x - median) / (iqr + eps)` with `eps=1e-6`, the
  `<10 valid values -> median 0 / iqr 1` skip rule, and per-season fitting via
  `RealMLPPreprocessor` (per-season scaler dict + global fallback, `mask`,
  `by_season`, `clip`, `save`/`load`, `from_manifest`).
- **`preproc.ple_transform`** (numpy piecewise-linear quantile encoding,
  Gorishniy 2022) vs **`pl_embedding.PLEmbedding`** (trainable *periodic* sin/cos
  embedding, RealMLP / FT-Transformer). Different features — both kept.
- **`align.fit/apply`** *solves* one orthogonal Procrustes rotation, while
  **`era_align`** *applies* precomputed per-season rotations (chained to a root
  frame in a `drift.json`) with subset / identity fallback. `load_alignment`
  takes an explicit `drift.json` path or a pre-loaded dict (the one intentional
  generalization over the hoops source), so any repo passes its own location.

`PLEmbedding` is torch-gated and exposed lazily: `import vector_core` still works
without torch, and `vector_core.PLEmbedding` only errors on construction if torch
is missing.

**The numpy-safe API is always importable.** `import vector_core` works with torch
NOT installed — the torch model is exposed lazily and only errors if you actually
construct a torch module without torch present. This keeps CI, eval, and data
tooling torch-free while the training path opts in via the `torch` extra.

## Install

```bash
pip install -e .            # numpy-safe core only
pip install -e .[torch]     # + the torch MTNN
pip install -e .[dev]       # + pytest, ruff
```

Requires Python >= 3.11.

## Adoption note (incremental migration)

The `vector-*` repos can migrate onto `vector-core` **one primitive at a time** —
there is no big-bang cutover. A repo can, for example, swap its local
`recall_at_k` for `vector_core.eval.recall_at_k` in a single PR while leaving its
model code untouched, verify the eval numbers are byte-identical, and continue.
The model classes here mirror the fleet's documented architecture
(`docs/UNIFIED_ARCHITECTURE.md`) so a repo can adopt the shared tower/fusion
blocks when it next retrains, rather than being forced to. Nothing here changes an
existing repo until that repo chooses to import it.

## License

MIT — see `LICENSE`.
