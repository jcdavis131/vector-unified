# Vector Unified — Phased Plan

> **Status:** Draft for single approval · 2026-07-10
> **Spec:** [`docs/SPEC.md`](../docs/SPEC.md) · **Architecture:** [`docs/UNIFIED_ARCHITECTURE.md`](../docs/UNIFIED_ARCHITECTURE.md)

The plan is staged so the proven per-sport models are never at risk: prerequisite (pitch) → frozen
alignment (safe) → fine-tune (gated) → ship. Each phase ends in a gate. No phase touches a live
per-sport asset until G1 is green.

---

## Phase 0 — Foundations (no model work)

- [ ] **0.1** Confirm the three encoder checkpoints load read-only and reproduce their shipped
  embeddings (hoops `mtnn_v5_concat_…` → 48-d; gridiron `mtnn_best.pt` → 32-d; pitch `vectors.json`
  → 16-d z + k-means(8) labels). Record shapes + a 3-player sanity fingerprint per sport.
- [ ] **0.2** Author `data/archetype_map.json` — the §4b alignment table: each sport's native
  clusters → the 12 cross-sport archetypes (A0–A11). Mark pitch A6/A7/A9/A10 `n/a`. This is the
  anchor set; everything downstream depends on it.
- [ ] **0.3** Author `data/analogy_triples.json` — the G4 panel seed: ~40 cross-sport analogy
  triples ("PF X ~ SS Y because role-label"), drawn from recognizable names across eras. This is the
  falsifier; write it before the model exists so it can't be reverse-engineered.
- [ ] **0.4** Read sibling conventions: `vector-gridiron/pipeline/train_mtnn.py` (ResidualTower,
  GatedFusion, family-drop), `vector-hoops/pipeline/train_mtnn.py` (heads, InfoNCE),
  `vector-pitch/pipeline/build_vectors.py` (FEATURES, tournament z). Match naming.

**Gate 0:** encoders load; archetype_map + analogy_triples committed; no code yet.

---

## Phase 1 — Prerequisite: pitch MTNN (so pitch can join)

Pitch is the only sport with no neural net. It cannot align as a PCA artifact.

- [ ] **1.1** `build_pitch_mtnn.py` — 16 features → 3 families (attacking, passing/control,
  defending/dueling) → small ResidualTowers → ~24-d L2 embedding. Heads: archetype CE (k-means(8)
  labels as target) + 16-d profile reconstruction. CPU-only, seed 7.
- [ ] **1.2** Validate against the existing PCA+k-means baseline: held-out-tournament archetype
  purity must not drop. (Leave-one-tournament-out: train 2018, test 2022, and vice-versa.)
- [ ] **1.3** Export `e_p` (24-d) for every pitch player-tournament; keep `vectors.json` contract
  intact (additive `pitch_emb.json` if needed; don't overwrite).

**Gate 1:** pitch MTNN purity ≥ k-means baseline. If it can't beat classical PCA on ~600 rows,
**stop** — pitch stays PCA and the unified model is built hoops+gridiron first, pitch added later.

---

## Phase 2 — Joint matrix + frozen-encoder alignment (Stage 1, the safe core)

- [ ] **2.1** `load_encoders.py` — load hoops + gridiron + pitch encoders; emit `e_h/e_g/e_p` per
  player-season (gridiron: mean of weekly `e_g` over REG weeks with ≥1 snap).
- [ ] **2.2** `build_unified_matrix.py` — assemble all player-seasons (sport, name, season, `e_s`,
  native cluster, cross-sport archetype from `archetype_map.json`, era index) into one leak-free
  matrix. Assert shapes, no-NaN, coverage.
- [ ] **2.3** `train_unified.py` Stage 1 — freeze `e_h/e_g/e_p`; train Pillar 2 (shared trunk,
  64-d, sport token, era ctx) + Pillar 3a (per-sport heads on `z`) + Pillar 3b (archetype InfoNCE
  with modality-aware temp + family CORAL) + Pillar 3c (gradient-reversal sport classifier).
  Balanced batches, pitch up-sampled + family-mask augmentation, effective-rank logged/epoch.
- [ ] **2.4** `eval_unified.py` — G1 (per-sport non-inferiority), G2 (sport-invariance + rank),
  G3 (cross-sport archetype silhouette). Write `unified_report.json`.

**Gate 2 (the big one):** G1 no per-sport regression · G2 sport-acc ≥10pp below no-debias baseline
with rank ≥32 · G3 silhouette > 0. If G1 fails, the joint space is hurting a sport → raise task-loss
weight / lower reversal λ / revert to per-sport heads on `e_s` only. If G2 rank drops → collapse;
raise task weight, retrain.

---

## Phase 3 — Analogy face-validity (the product vision, operationalized)

- [ ] **3.1** `analogy_panel.py` — for each triple in `analogy_triples.json`, compute cross-sport
  top-k=10 NN in `z`, score whether Y lands in top-k, emit the role label of the shared archetype.
- [ ] **3.2** Run the panel; record G4 hit-rate and G5 retrieval sanity (manual review of 20
  "find the NFL player most similar to [NBA player]" queries).
- [ ] **3.3** If G4 < 60%: revise `archetype_map.json` (the taxonomy is the lever, not the losses
  first), retrain Stage 1, re-run. Log taxonomy version per run.

**Gate 3:** G4 ≥ 60% with sensible labels. This is the "real answer" gate — the whole point.

---

## Phase 4 — End-to-end fine-tune (Stage 2, gated)

Only if Gate 2 + Gate 3 pass.

- [ ] **4.1** Unfreeze the three encoders; fine-tune the full stack at LR ≈1e-5 with per-sport task
  losses dominant. Goal: let the joint signal refine encoders, not redefine them.
- [ ] **4.2** Re-run G1–G3. **Hard stop if any sport regresses beyond noise** → revert to Stage 1
  frozen encoders and ship that (Stage 1 is a valid ship state).

**Gate 4:** Stage 2 G1–G3 ≥ Stage 1 numbers, or revert to Stage 1.

---

## Phase 5 — Ship + Methods

- [ ] **5.1** `export_unified.py` → `assets/unified.json` (64-d, all sports, one space, PCA(3) map,
  per-player: sport, name, season, archetype, native archetype).
- [ ] **5.2** Ablation report: drop each alignment loss (contrastive / CORAL / adversarial) → Δ on
  G2/G3/G4. Each must earn its keep (house rule). Document any that don't.
- [ ] **5.3** Methods copy, honest-voiced: "shared role space", "best-guess analogies", "wrong all
  the time"; pitch role-only caveat; no "proven similarity" anywhere.
- [ ] **5.4** `vector-hub` landing page: upgrade the "Where this is going" section from *goal* to
  *shipped, with caveats* — only after G1–G4 pass. (Per `vector-hub/README.md` rule: do not upgrade
  the cross-sport claim to "shipped" until it is.)

**Gate 5:** `unified.json` on the hub; Methods honest; hub copy upgraded from goal to shipped-with-caveats.

---

## Sequencing summary

```
Phase 0  foundations (encoders load, taxonomy, analogy panel seed)   (no model)
Phase 1  pitch MTNN prerequisite                                     (gate: beats PCA)
Phase 2  frozen-encoder alignment — Stage 1                          (gate: G1–G3)
Phase 3  analogy face-validity                                       (gate: G4 ≥ 60%)
Phase 4  end-to-end fine-tune — Stage 2 (gated)                      (gate: ≥ Stage 1)
Phase 5  ship + Methods + hub copy                                   (gate: honest + live)
```

The destructive-ish moment is **5.4** (upgrading the hub claim). Everything before it is additive
and reversible. Stage 1 (Phase 2) is a valid ship state if Stage 2 (Phase 4) doesn't help.
