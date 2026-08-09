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
| `vector_core.align` | `fit(A, B) -> R`, `apply(X, R)` (rotation-only Procrustes) | no |
| `vector_core.losses` | `info_nce_numpy`, `sup_con_numpy` (+ `*_torch` variants) | lazy |
| `vector_core.eval` | `recall_at_k`, `purity_at_k`, `silhouette_cosine` | no |
| `vector_core.schema` | `FleetEntry`, `validate_entry` (fleet-report entry shape) | no |
| `vector_core.model` | `MTNN`, `MaskedResidualTower`, `AttentionGatedFusion`, `MultiTaskHeads` | required |

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
