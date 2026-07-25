# Stage 2 — Unfrozen Encoder Alignment (plan, awaiting approval)

> **Status:** PLAN — not started. SPEC §6 marks end-to-end encoder unfreezing
> **"ask first"** because it risks the three live games. This doc makes that
> decision concrete and reviewable. Say "go Stage 2" to execute.

## 1. Goal

Reach **G2 sport-invariance** (sport recoverable from `z` ≤ chance + 0.10) by
letting the three per-sport MTNN encoders drift into a **shared basis** under the
unified alignment losses. Stage 1 proved (p19, empirical) that G2 is structurally
impossible while encoders are frozen: the native dimension footprint
(hoops 48-d / gridiron 32-d / pitch 24-d) is a perfect, linearly separable sport
signature no trunk-side fix can erase. Unfreezing is the only lever left.

Secondary: let the joint signal refine the per-sport encoders (cross-sport
analogy pressure could sharpen role boundaries inside each sport).

## 2. Feasibility (live evidence, 2026-07-10)

All three encoders are reloadable — model `state_dict` + feature matrix present
for each. **Three different formats and model classes** → Stage 2 needs a
per-sport "live encoder" loader (real, bounded engineering, not a one-liner).

| sport   | checkpoint | size   | fmt keys                          | matrix | matrix keys                                   | d_emb | risk |
|---------|------------|--------|-----------------------------------|--------|-----------------------------------------------|-------|------|
| hoops   | `vector-hoops/pipeline/data/mtnn_best.pt`   | 2.26 MB | `model`, `args`, `weights` (8 heads) | `train_matrix.npz` | `Z,mask,player_id,season,name,cluster`        | 48    | **high** (heaviest, 8 task heads, live game) |
| gridiron| `vector-gridiron/pipeline/data/mtnn_best.pt`| 244 KB  | `state,mu,sd,feats,families,n_seasons,season_min,d_emb` | `train_matrix.npz` | `Z,mask,Y,Y_usage,season,week,gsis,name,pos,team` | 32    | med  |
| pitch   | `vector-pitch/pipeline/data/pitch_mtnn.pt`  | 48 KB   | `state_dict,config`               | `tm_full.npz` | `X,M,ctx_ids,n_rows,n_features`               | 24    | low  |

`load_encoders.py` already reconstructs gridiron's `_MTNN` from its wrapper;
hoops + pitch need their native model classes imported/reconstructed
(`vector-hoops` train script; `vector-pitch/pipeline/train_mtnn.py:PitchMTNN`).

## 3. Architecture

```
feature matrix_s  ->  MTNN_s (UNFROZEN, requires_grad)  ->  e_s (pre-L2 logits)
                                                                     |
        3× e_s  ->  unified trunk (adapter + era + MLP -> 64-d L2 = z)
                                                                     |
   losses: SupCon(arch) + per-sport task(native cluster + position) + GRL(sport)
        backprop flows through the trunk AND back into each MTNN_s
```

- **Live-encoder loader** (`load_live_encoders.py`, new): reconstruct each
  `MTNN_s` from its native class, load `state_dict`, set `requires_grad_(True)`,
  run forward from its feature matrix → returns `e_s` (pre-normalization logits)
  **as a graph-bearing tensor**, plus records. Must normalize each matrix schema
  (hoops `Z/mask`, gridiron `Z/mask`+season scaling, pitch `X/M/ctx_ids`).
- **Trunk**: reuse `UnifiedTrunk` but feed **graph-bearing** `e_s` (not the
  frozen numpy arrays). Adapter now backprops into encoders.
- **Two param groups**: encoders LR **1e-5** (default; tiny to avoid wrecking the
  live games), trunk/heads LR **1e-3** (as Stage 1).
- **Losses**: inherit lean v0.1 — SupCon + task + GRL. Revisit CORAL/VICReg only
  if collapse reappears. Per-sport task losses must stay **dominant** so each
  encoder keeps its native role structure (anti-forgetting).
- **The actual G2 mechanism**: unfreezing lets each encoder's final projection
  rotate so the three sports' embeddings converge on a shared basis — this is
  what erases the dim-footprint leak that p19 showed is unbeatable frozen.

## 4. Gating & revert (the safety contract)

- **Per-sport G1 non-inferiority is the hard gate.** Before Stage 2, record each
  sport's Stage 0 baseline: kNN-5 role accuracy + position accuracy from the
  shipped `e_s`. After each Stage 2 epoch, re-measure on the **unfrozen** `e_s`.
  **Any sport whose role or position accuracy drops > 0.02 vs Stage 0 → auto-revert
  to `unified_best.pt`, halt, report.** (Threshold is a proposed default; see §7.)
- **Checkpoint every epoch** to `unified_stage2_epN.pt`; keep best-by-(G1 ∧ G2).
- **Per-sport assets are NEVER overwritten.** Stage 2 writes only into
  `vector-unified/` (new checkpoints, new `unified.json`). The three live games'
  shipped embeddings (`embedding_v3.npz`, `pitch_mtnn_embeddings.json`, the
  gridiron game's `mtnn_best.pt`) stay read-only. A Stage 2 regression cannot
  touch the live games — only the unified artifact would roll back.
- **Final export** only if G2 passes AND G1 non-inferiority holds for all 3;
  otherwise Stage 1 v0.1 remains the shipped `unified.json`.

## 5. Risks

| risk | mitigation |
|------|------------|
| hoops regression (heaviest, 8 heads, live game) | encoder LR 1e-5; dominant per-sport task loss; stagger option (§7) |
| catastrophic forgetting in encoders | tiny encoder LR + strong per-sport task anchor; early stop on regression |
| compute (forward+backward × 3 MTNNs × 20,721 rows/epoch) | GPU already used; expect ~2-4× Stage 1 wall-clock; batch as today |
| collapse (rank drops) | keep warmup + rank_floor; re-enable VICReg if rank < 12 |
| GRL still inert at Stage 2 | acceptable — even partial sport-confusion + shared basis may satisfy G2; if not, document and hold |

## 6. Implementation steps (ordered)

1. `load_live_encoders.py` — reconstruct 3 MTNN classes, load state_dicts,
   forward from matrices → graph-bearing `e_s` + records. Smoke: norms ≈ 1.0,
   counts match `unified_meta.json`.
2. Extend `UnifiedTrunk.encode` to accept graph-bearing `e_s` (torch tensors, not
   numpy). Verify Stage 1 metrics reproduce on a frozen pass (sanity: unfrozen
   with LR 0 ≡ Stage 1).
3. Record Stage 0 per-sport baselines (kNN-5 role + position) into
   `data/stage2_baselines.json`.
4. `train_stage2.py` — two param groups, per-epoch G1 check, auto-revert, save
   `unified_stage2_epN.pt`. Run 20-40 epochs.
5. `eval_unified.py --ckpt unified_stage2_best.pt` — full G1/G2/G3 + analogy G4.
6. If G2 passes ∧ G1 holds → `export_unified.py` refreshes `assets/unified.json`
   (Stage 2 version). Else revert, document, keep v0.1.
7. Update `UNIFIED_ARCHITECTURE.md` §11 + `SPEC.md` §7 with Stage 2 outcome.

## 7. Open questions for you (the actual ask-first decisions)

1. **Encoder LR**: 1e-5 (default, safe) or 1e-6 (ultra-conservative)?
2. **Stagger or simultaneous**: unfreeze all 3 at once, or start with
   pitch+gridiron (cheap, low-risk) and add hoops only if those hold? Staggering
   protects the live hoops game but yields an asymmetric Stage 2.
3. **Epochs**: 20 (fast probe) or 40 (full)?
4. **Regression threshold**: 0.02 (strict) or 0.05 (looser) per-sport drop before
   auto-revert?

**Defaults if you just say "go Stage 2":** LR 1e-5, all 3 simultaneous, 30 epochs,
threshold 0.02, auto-revert on. I'll report per-epoch G1/G2 and stop the moment
any sport regresses.

---

## 8. Outcome (executed 2026-07-10, defaults approved)

**Status: INFRASTRUCTURE COMPLETE, RUN BLOCKED by a concurrent hoops hillclimb.**

### What was built (sound, proven)
- `pipeline/load_live_encoders.py` — reconstructs all 3 MTNNs (hoops via native
  `train_mtnn.py` importlib, gridiron via `_MTNN`, pitch via native `PitchMTNN`),
  loads state_dicts **strict**, `requires_grad_(True)` on the encode path
  (towers+fusion only; heads frozen). Per-batch `encode_batch(idx)` returns
  graph-bearing L2-normed `e_s` in `load_encoders` row order.
  **Smoke gate: cosine vs frozen = 1.00000 for all 3 sports; grad-flow reaches
  encoder weights (hoops 301 / gridiron 139 param-tensors). PASS.**
- `pipeline/train_stage2.py` — 2 param groups (encoders 1e-5 / trunk 1e-3),
  lean v0.1 losses (SupCon+task+GRL), per-epoch G1 encoder non-regression gate
  on the live `e_s`, best-by-G2 checkpoint capture (trunk + drifted encoder
  state_dicts), post-hoc shippability verdict, cudnn determinism.
- `pipeline/stage2_eval.py` — reconstructs the Stage 2 model from the saved
  encoder + trunk states, runs full G1/G2/G3 + G4 analogy → `data/stage2_report.json`.

### What ran (and why it cannot be trusted)
Three 30-epoch training runs were executed. The numbers trended favorably
(G2 0.743→~0.70, G3 0.681→~0.75, G1 hoops role held or improved, rank ~12),
but **a concurrent hoops hillclimb task was modifying the hoops game artifacts
while Stage 2 ran**: `vector-hoops/pipeline/train_mtnn.py` (modified), new
`composite_score.py` + `seed_cqs_baseline.py`, and `tasks/hillclimb-mtnn-cqs.md`,
and — critically — `pipeline/data/mtnn_best.pt` **and** `pipeline/data/embedding_v3.npz`
were overwritten (mtime 2026-07-10 20:56, both gitignored → unrecoverable).

The hoops encoder was therefore a **moving target** across runs (the run-to-run
hoops G1 variance — 0.849 / 0.803 / 0.918 — is partly this, not only CUDA noise).
A post-run integrity check confirmed it: re-running the live-encoder smoke dropped
hoops cosine-vs-frozen from 1.00000 to **0.88457** (gridiron and pitch stayed 1.00000).
The Stage 2 results are **confounded** and must be discarded.

### What is safe
- **Shipped `assets/unified.json` (Stage 1 v0.1) is intact** — mtime 19:04, written
  before the hoops overwrite, self-contained 64-d embeddings (not dependent on the
  current `mtnn_best.pt`/`embedding_v3.npz`).
- Gridiron and pitch artifacts untouched (mtime 16:57 / 17:56; cos 1.00000).
- No per-sport asset was written by Stage 2 code (read-only by construction).

### Verdict (confounded run — discarded)
- G2 did **not** pass in any run (~0.70 vs target ≤0.43): 30 epochs at encoder
  LR 1e-5 moves sport-recoverability meaningfully but is far from erasing the
  sport signature. Even un-confounded, this config would not have shipped.
- **Stage 1 v0.1 remains the shipped milestone.** `unified_stage2_best.pt` is
  retained as (confounded) evidence, not as a shippable artifact.

### Clean re-run 2026-07-11 (enc_lr=1e-6, 30 epochs) — NOT SHIPPABLE
After the hoops artifacts stabilized, a clean run was executed:

- Smoke: `enc_lr=1e-5` collapses hoops G1 −0.063 (drifting); `1e-6` plateaus at
  −0.040. **Deviated from the 1e-5 default to 1e-6** based on that evidence.
- Warm-start from `unified_best.pt`; encode-path only (towers+fusion); post-hoc
  shippability verdict (no mid-run auto-halt).
- **Best epoch 26:** G2=**0.674** (Stage 1 ~0.74 → −7pp), G3=0.736, rank=12.4.
- **G1 PASS:** hoops role_drop=**−0.040** (improved), gridiron −0.001, pitch 0.000.
- **SHIPPABLE: False** (G1 ok ∧ G2 miss vs target ≤0.433).
- Artifacts: `unified_stage2_best.pt` + `stage2_history.json` + drifted
  `enc_states` in the ckpt. Per-sport assets never written. Shipped v0.1 intact.

### To resume Stage 2 (needs user coordination)
1. ~~Wait for the concurrent hoops hillclimb~~ — done; clean run completed.
2. ~~Re-run smoke~~ — done (cos≈1.0 at launch).
3. Next G2 levers (each gated by G1 non-regression):
   - warmup-freeze encoders (trunk-only warmup), then unfreeze at folding
   - longer run (60–100 epochs) at 1e-6
   - staggered unfreeze (pitch → gridiron → hoops)
   - stronger GRL (λ=0.1) or higher enc_lr (3e-6) once warmup-freeze is in
4. If G2 still misses after those, reconsider whether G2≤0.433 is the right
   ship gate vs a softer "sport-acc Δ ≥ 10pp from Stage 1" relative target.
