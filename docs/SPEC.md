# Vector Unified — SPEC: Joint Cross-Sport Embedding (v1)

> **Status:** Draft for single approval (auto-mode plan gate) · 2026-07-10
> **Parents:** Vector Hoops MTNN v4/v5 · Vector Gridiron MTNN v2 · Vector Pitch (PCA)
> **Design:** [`docs/UNIFIED_ARCHITECTURE.md`](./UNIFIED_ARCHITECTURE.md)
> **Vision source:** dumbmodel.com — "Where this is going"

---

## ASSUMPTIONS (correct now or we proceed)

1. **The three per-sport models are the input layer**, not competitors to replace. Hoops (48-d MTNN,
   12,966 seasons), Gridiron (32-d MTNN, 49,881 weeks → 5,325 season-aggregated), Pitch (24-d MTNN,
   2,430 seasons — full male StatsBomb open corpus, 11 contexts; the PCA(3)/~600-row baseline was
   replaced in Phase 1) are loaded as frozen encoders first; the unified model folds their outputs
   into one 64-d space.
2. **There are no natural cross-sport pairs.** No NBA player is also an NFL player. Alignment is
   learned from **shared abstract role archetypes** (synthetic positives) + **sport-invariance
   (adversarial)** + **shared family ontology (CORAL)** — not from co-occurrence.
3. **The unified embedding is season-level** (the "who is this player, archetypally" map). Gridiron's
   weekly prediction stays a sport-specific head on weekly features. Pitch player-tournament ≈
   season. Hoops is already season-level.
4. **The unified model is additive.** It ships a new `assets/unified.json`. It must not overwrite or
   regress per-sport assets (`vectors.json`, `embedding.json`, `nextgame.json`, `projections.json`).
5. **Pitch must gain a neural trunk first** (prerequisite). PCA(3) cannot join a learned shared
   space. The pitch MTNN is small, CPU-only, validated against the existing k-means(8) baseline.
6. **Pitch is sparse** — it has only 4 of the 10 sport-agnostic families (volume, efficiency,
   defense, playmaking). Cross-sport analogies involving pitch are role-only, never physical/market.
   Accepted and documented; not faked.
7. **No new data ingestion.** Unified consumes the three existing embeddings + manifests + the
   already-cached StatsBomb data for the pitch MTNN. No paid feeds, no new scrapes.
8. **Repo is greenfield** (`vector-unified/`). No git yet — planning docs land first; first commit
   when you say go.
9. **The cross-sport archetype taxonomy (§4b of the architecture doc) is a v0 human proposal**,
   versioned in `data/archetype_map.json`, revisable after the first face-validity panel.

→ Reply **yes** to approve this plan and start implementation top-to-bottom. Reply with corrections
to any assumption first if needed.

---

## 1. Objective

Build **one joint 64-d L2-normalized embedding** across NBA / NFL / World Cup that:

- Folds the three proven per-sport embeddings into a single geometry where **role** is shared and
  **sport** is (adversarially) de-emphasized.
- Keeps every per-sport task at least as accurate as today (non-inferiority — the games must not
  break).
- Answers "what does a power forward and a strong safety have in common" with a **falsifiable**
  cross-sport nearest-neighbor + role-label, validated by a curated analogy panel.
- Ships a Methods-honest doc set: every technique cited, every gap (pitch sparsity, no ground truth)
  stated, nothing oversold as "proven similarity."

**Users:** the dumbmodel arcade audience + anyone curious about cross-sport role analogies.
**Success:** G1–G3 gates green (§6 of architecture doc) + G4 analogy panel ≥ 60%, with no per-sport
regression and no embedding collapse.

---

## 2. Commands (expected)

```powershell
cd c:\Users\jcdav\vector-unified

# Prerequisite: pitch neural trunk
python pipeline/build_pitch_mtnn.py            # uses cached StatsBomb data; validates vs k-means(8)

# Assemble the joint matrix from the three encoders
python pipeline/load_encoders.py               # smoke: load hoops/gridiron ckpts; build pitch
python pipeline/build_unified_matrix.py        # e_h/e_g/e_p + labels + archetype_map → matrix

# Train
python pipeline/train_unified.py --epochs 60   # Stage 1 frozen; --finetune for Stage 2

# Gates
python pipeline/eval_unified.py                # G1–G3 automatic
python pipeline/analogy_panel.py               # G4/G5 curated panel

# Ship
python pipeline/export_unified.py              # → assets/unified.json
```

Hoops/gridiron encoders are loaded from their own project checkpoints
(`vector-hoops/assets/mtnn_*.pt`, `vector-gridiron/pipeline/data/mtnn_best.pt`) — the unified
pipeline reads them read-only; it never writes back to the sport repos.

---

## 3. Project structure

```
vector-unified/
  docs/
    SPEC.md
    UNIFIED_ARCHITECTURE.md
  tasks/
    plan.md
    todo.md
  pipeline/
    load_encoders.py
    build_pitch_mtnn.py        # prerequisite
    build_unified_matrix.py
    archetype_map.py
    train_unified.py
    eval_unified.py
    analogy_panel.py
    export_unified.py
  data/
    archetype_map.json
    analogy_triples.json
    unified_best.pt
    unified_report.json
  assets/
    unified.json
```

---

## 4. Code style

Match the existing vector-* pipelines: stdlib + numpy + torch, module-level constants, leakage
comments in docstrings, `norm_key(name, pos)` where keys are needed, no new heavy deps unless
justified (`document-non-action`). Prefer the sport projects' family/tower naming so the mapping
(§4a of architecture doc) reads as a relabeling, not a reinvention. Imports at top of module only
(no inline imports).

---

## 5. Testing strategy

| Level | What |
|---|---|
| Unit | encoder load: shapes match per-sport arch json; archetype_map covers every native cluster; unified matrix has no NaN in unmasked cells; 64-d output is L2-unit. |
| Inspect | per-sport row counts in the joint matrix; family coverage per sport; archetype balance. |
| Train gate | joint val score (composite: per-sport val + cross-sport silhouette) early-stop; effective rank of `z` logged every epoch (collapse watch). |
| Eval gates | G1 non-inferiority, G2 sport-invariance + rank, G3 archetype coherence (automatic). |
| Panel | G4 analogy top-k, G5 retrieval sanity (curated, periodic). |
| Ablation | drop each alignment loss (contrastive / CORAL / adversarial) and measure Δ on G2/G3/G4 — each must earn its keep (house rule). |

---

## 6. Boundaries

**Always do**

- Load per-sport encoders read-only; never write back to the sport repos.
- Keep per-sport asset contracts intact; `unified.json` is additive.
- Cite every technique in Methods; state every gap (pitch sparsity, no ground truth).
- Version the archetype taxonomy; log revisions.
- Monitor embedding effective rank every epoch; stop on collapse.

**Ask first**

- End-to-end fine-tune that unfreezes the per-sport encoders (Stage 2) — gated on Stage 1 G1 passing. **(DONE 2026-07-11 clean run: infrastructure proven; enc_lr=1e-6 × 30ep → G1 PASS / G2=0.674 FAIL vs ≤0.433 → SHIPPABLE=False. Dim-footprint leak eroded −7pp but not broken. Stage 1 v0.1 remains shipped. Next levers in [`STAGE2_PLAN.md`](./STAGE2_PLAN.md) §8.)**
- ~~Expanding pitch data beyond WC 2018+2022 (new StatsBomb competitions).~~ **DONE in Phase 1 — pitch now uses the full male StatsBomb open corpus (11 contexts, 2,430 seasons).**
- Replacing any per-sport encoder with a unified-path equivalent.
- Shipping `unified.json` to the live hub before G1–G3 + G4 pass. **(G1/G3/G4 pass; G2 deferred — hub copy must state the sport-invariance caveat honestly before shipping.)**

**Never do**

- Claim cross-sport similarity is "proven" or has ground truth.
- Fake pitch bio/market/pedigree features to fill the family ontology.
- Let the adversarial sport head collapse `z` to chance (rank collapses first — watch it).
- Overwrite a per-sport asset with a unified-path output before non-inferiority is shown.
- Train the joint space on labels that leak across the temporal split of any sport.

---

## 7. Acceptance criteria (Definition of Done)

> **Stage 2.1 status (2026-07-30, shipped):** 1✓ 2✓ 3✓ 4✓ 5✓(G1, improved) 6✗(G2 soft target, not met —
> see below) 7✓(G3) 8✓(G4=0.978, automated not curated) 9✓ 10☐(methods page pending). G2 moved from
> Stage 1's ~0.74-0.79 to 0.693 after unfreezing the encoders (Stage 2.1) but plateaued there through
> 60 epochs, well above the ≤0.433 target. User decision (2026-07-30): ship on G1+G3 with G2 as an
> honest soft-target caveat rather than hold for a Stage 2.2 attempt. `assets/unified.json` now ships
> from `unified_stage2_best.pt` (drifted encoders), not the frozen Stage 1 `unified_best.pt`.
>
> Prior (Stage 1 v0, 2026-07-10): 1✓ 2✓ 3✓ 4✓ 5✓(G1) 6✗(G2 deferred→Stage 2, see arch §11)
> 7✓(G3) 8✓(G4=0.978, automated not curated) 9✓ 10☐(methods page pending). G2 is the one open gate;
> it is structurally blocked at Stage 1 (frozen encoders in different spaces) and is a Stage 2 goal.

1. `docs/UNIFIED_ARCHITECTURE.md` + this SPEC committed in tree.
2. Pitch MTNN built; archetype purity ≥ k-means(8) baseline on held-out tournaments. **✓ — beats
   shipped PCA(3) on 4/4 role-recovery metrics (pos-cluster, kNN-5, NN-role, recon MAE).**
3. `build_unified_matrix.py` assembles e_h/e_g/e_p + `archetype_map.json` into one leak-free matrix;
   shapes/coverage asserted. **✓**
4. `train_unified.py` trains Stage 1 (frozen encoders) end-to-end offline; `unified_report.json`
   written with G1–G3 numbers. **✓**
5. **G1:** no per-sport metric regresses beyond noise (hoops cluster/position/salary/profile;
   gridiron PPR MAE ≤ 4.313 + 0.05; pitch purity ≥ baseline). **✓ PASS — z beats frozen e_s on
   native-cluster kNN-5 for all 3 sports; position 1.0 all.**
6. **G2:** sport-classifier accuracy ≥ 10pp below a no-debiasing baseline; `z` effective rank ≥ 32.
   **✗ STAGE 2.1 SOFT TARGET, SHIPPED ANYWAY — Stage 1 (frozen): sport-acc 0.771 vs baseline 0.799
   (Δ≈3pp; GRL inert), rank 13. Stage 2.1 (unfrozen encoders, 60ep, enc_lr 3e-5, GRL λ 0.10): sport-
   acc improved to 0.693 but plateaued there (not still falling at epoch 60) — target ≤0.433 not
   met. G1 held/improved for all 3 sports throughout (no regression), so per
   `docs/STAGE2.1_SWEEP_PLAN.md` §5's own decision tree ("G2 plateau > 0.55 → declare a soft target,
   user decision") the user chose to ship on G1+G3 with this named as an open gap rather than run
   Stage 2.2. Structural: encoders can drift some without hurting per-sport role, but fully erasing
   sport from a 64-d shared space while keeping G1 intact hits a real ceiling around here.**
7. **G3:** cross-sport archetype silhouette > 0 on `z`. **✓ PASS — silhouette 0.683; within-arch
   cross-sport cos 0.746 >> between-arch -0.121.**
8. **G4:** ≥ 60% of curated cross-sport analogies land in top-k=10 with a sensible role label.
   **✓ PASS — 0.978 via automated cross-sport NN role-coherence (broader than 40 curated triples);
   named panel sensible. Curated `analogy_triples.json` still pending (task 0.3).**
9. `assets/unified.json` ships (64-d, all sports, one space, PCA(3) map for UI). **✓ — 20,721
   players; norms=1.0; per-sport assets untouched (additive).**
10. Methods page copy honest: "shared role space", "best-guess analogies", never "proven similarity";
    pitch role-only caveat stated. **☐ pending (Phase 5.3).**

---

## 8. Non-goals (this pass)

- No new data ingestion beyond existing pipelines + cached StatsBomb.
- No tracking/spatial features in the joint space.
- No live/real-time unified embeddings (batch asset on retrain cadence).
- No Gramian / higher-order alignment term day-one (ablation-gated).
- ~~No pitch data expansion day-one (flagged for a later phase).~~ **DONE — pitch expanded in Phase 1.**
- No replacement of per-sport assets; no UI rewrite of the live games.
- **Market/cultural footprint signals (user-requested: salary/sponsorship/social-media) are NOT in v0** — blocked on external data acquisition (only hoops has sal/pedigree). Folded on role archetypes only; a documented gap.
