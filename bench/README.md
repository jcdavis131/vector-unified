# unified bench — the cross-domain transfer probe on REAL data

Answers the registry's unified question per transfer target: does a shared
embedding trained **without** the held-out domain predict that domain's
forward target better than baselines that never saw other domains?

**Verdict (this run): no — baseline wins both judged targets.** Transfer
carries real signal (the frozen probe is far above chance on both targets)
but never beats the held-out domain's own feature encodings. See
`benchmark_report.json` (schema 1.1) for every number.

| target | held-out domain | primary metric | best baseline | MTNN (frozen probe) | beats? |
|---|---|---|---|---|---|
| transfer_forward_return | equities | spearman_ic | pca16_whiten_ridge = 0.1607 | 0.0671 | **no** (−0.0935) |
| transfer_next_season_per | hoops | spearman_ic | pca16_whiten_ridge = 0.7800 | 0.6328 | **no** (−0.1472) |

Context: the owner lanes' own *supervised* MTNNs also lost on these two
targets (hoops 0.7729 vs 0.7794; equities 0.0926 vs 0.1481), so the transfer
probe's loss is against a bar even in-domain training did not clear.

## Protocol

1. **Inputs** — `bench/data/exchange/<domain>/dataset.npz`: committed
   snapshots of the five sibling lanes' verified real-data exchange artifacts
   (hoops NBA next-season stats, gridiron nflverse weekly, equities
   SEC-EDGAR + Yahoo forward windows, realty BIS property prices, pitch FBref
   WSL windows). Each carries its own datasheet with label construction and
   split spec.
2. **Schema alignment** — per domain: `vector_core` RobustScaler → PCA(16,
   full SVD) → per-component whitening, fit on that domain's train rows only.
3. **Shared embedding** — one trunk (16→64→64→32-d L2-normalized, GELU) +
   one linear head per (domain, wired target), trained jointly on the FOUR
   non-held-out domains (masked MSE on train-z-scored regressions, masked BCE
   on binary targets, per-domain mean) + CORAL ×0.1 pulling the four domains'
   embedding distributions together. Gradient rows = each domain's committed
   `train_idx` only; early stopping on committed `val_idx` only; the held-out
   domain is never loaded during embedding training; training domains' test
   rows are never forward-passed.
4. **Probe** — freeze the trunk; map the held-out domain through its own
   unsupervised RobustScaler+PCA+whiten (fit on the harness train side); fit
   ONLY sklearn `Ridge(alpha=1.0)` on the harness train side; predict the
   harness test side. That is the report's `mtnn` rung.
5. **Gauntlet** — the held-out domain's owner-lane task construction,
   reproduced exactly (equities: fy cut 2022, seed 42, train-only
   impute/standardize; hoops: target_year cut 2026, seed 7, harness-ready X,
   `persistence_current_per` rung), + the default vector-bench prediction
   ladder + a `pca16_whiten_ridge` control rung (the trunk's exact input with
   the probe's exact head — isolating what the cross-domain trunk adds).
   Reproduction check: this run's ridge rungs match the owner lanes'
   committed reports to 6 decimals (hoops ridge ic 0.779299, equities ridge
   ic 0.144842) on identical n_train/n_test.

## Reproduce

```bash
python bench/run_transfer_benchmark.py
```

CPU-only (~1 min), 2 threads, seeded (SEED=0 for training/ladder; task seeds
mirror the owner lanes). Two back-to-back runs in this environment produced
byte-identical metric values and verdicts. Requires torch (CPU), sklearn,
and editable vector-core + vector-bench from the vector-hub monorepo.

## Artifacts

- `run_transfer_benchmark.py` — the full pipeline (this file produced every
  number committed here).
- `benchmark_report.json` — schema-1.1 domain report.
- `training_config.json` — exact architecture/optimizer/early-stop record.
- `data/dataset.npz` + `data/datasheet.json` — the unified exchange artifact:
  32-d frozen embeddings of both held-out task universes, labels, owner
  splits.
- `data/exchange/<domain>/` — the five committed input snapshots.

## Honest reading

The frozen cross-domain embedding retains a usable fraction of each held-out
domain's signal (0.63 of 0.78 attainable IC on hoops; 0.067 of 0.161 on
equities) through a 16-d unsupervised bottleneck and a trunk trained on
other domains. But the `pca16_whiten_ridge` control shows the trunk itself
*subtracts* information relative to its own input: cross-domain structure
learned from four domains did not specialize the representation usefully for
the fifth. A clean negative for the shared-embedding transfer thesis at this
scale of domain count and data.
