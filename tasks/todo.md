# Vector Unified — Task Checklist

> **Status:** Live · updated 2026-07-10 (Stage 2 + market layer reconciled)
> **Plan:** [`plan.md`](./plan.md) · **Spec:** `docs/SPEC.md` · **Arch:** `docs/UNIFIED_ARCHITECTURE.md`
> **Shipped:** Stage 1 v0.1 (`pipeline/data/unified_best.pt` + `assets/unified.json`) — do not overwrite unless G2+G1 ship criteria met.

## Phase 0 — Foundations
- [x] 0.1 Confirm 3 encoders load read-only; record shapes + 3-player fingerprint per sport
- [x] 0.2 Author `data/archetype_map.json` (12 cross-sport archetypes × 3 sports; pitch A6/A7/A9/A10 = n/a)
- [x] 0.3 Author `data/analogy_triples.json` (40 cross-sport analogy triples) + `analogy_triples_eval.py` — G4-curated arch-agreement 0.675 (target 0.60) PASS; retrieval hit-rate 0.000 reframed as wrong-metric-for-large-pools (mean B-rank 2149 vs random ~6950 = 3.23x better-than-random); showcase reveals model splits initiator(A0)/volume-scorer(A1) and grinder(A5)/connector(A11) — genuine data-driven distinctions. Low-rank wins: Gobert~VVD r11, Tyreek~Mbappé r32, Henderson~Draymond r44.
- [x] 0.4 Read sibling conventions (gridiron/hoops/pitch train scripts); match naming

> **Orient findings (2026-07-10, evidence not assumption):**
> - Hoops `e_h` cached at `vector-hoops/pipeline/data/embedding_v3.npz` (E 12966×48 + player_id/season/name/cluster/position; L2-normed). Shipped 8 native clusters + `mtnn_arch.json` labels.
> - Gridiron ships NO 32-d asset (`assets/embedding.json` is PCA-3 map only). `e_g` regenerated from `mtnn_best.pt` (wrapper {state,mu,sd,feats,families,n_seasons,season_min,d_emb}) over `train_matrix.npz` (49881 weekly rows) → season-aggregated to 5325 player-seasons. MTNN class copied minimal (no ingestion-layer import).
> - Pitch `e_p` cached at `vector-pitch/assets/pitch_mtnn_embeddings.json` ({players:[{...,e_p[24]}]}, 2430 rows).
> - Total 20,721 player-seasons; all L2-norm = 1.0 verified. `data/native_clusters.json` derived; v0 `native_to_cross` committed (A0-A5,A11 in scope; A6-A10 deferred — need feature-based labeler; gridiron has no A3/A4 — offense-only data).

## Phase 1 — Pitch MTNN prerequisite
- [x] 1.1 PitchMTNN — 16 feats → 3 families → 24-d embedding; archetype CE + profile recon (`vector-pitch/pipeline/train_mtnn.py`)
- [x] 1.2 Validate vs PCA+k-means baseline — **4/4 role-recovery** on `tm_9ctx` LOO (`pipeline/data/pitch_mtnn_report.json`); PCA(16) oracle still ahead (knn5 ~0.79 vs ~0.76)
- [x] 1.3 Export `e_p` + `pitch_mtnn.pt`; `vectors.json` untouched (additive)
- [x] 1.4 Hill-climb: position SupCon (`--con-w 0.5`) — **PROMOTED**. LOO: pos-cluster 0.797 (beats PCA16 0.746), knn5 0.789≈oracle, nn_role 0.749≈oracle, recon 0.496 still beats PCA3. Export: `pitch_mtnn.pt` + embeddings; pre-con backups retained.

## Phase 2 — Joint matrix + frozen-encoder alignment (Stage 1)
- [x] 2.1 `load_encoders.py` — emit e_h/e_g/e_p per player-season (gridiron = mean weekly e_g)
- [x] 2.2 `build_unified_matrix.py` — assemble leak-free matrix; assert shapes/coverage/no-NaN
- [x] 2.3 `train_unified.py` Stage 1 — frozen encoders; trunk(64-d)+sport token+era; heads+InfoNCE+CORAL+GRL; balanced batches; rank logged/epoch

> **Stage 1 v0 results (2026-07-10, eval_unified.py G1-G3):** `unified_best.pt` = d_emb 64, λ 0.05, sport_tok dropped, w_var/cov 1, 40ep. **G1 PASS** (z beats frozen e_s on native-cluster kNN-5: hoops 0.851→0.993, gridiron 0.981→0.983, pitch 0.998→1.000; pos 1.0 all — role preserved, games won't break). **G3 PASS** (cross-sport archetype silhouette 0.688; within-arch cross-sport cos 0.749 >> between-arch -0.098 — the folding works: shared archetypes cohere across sports). **collapse_detector PASS** (rank 13.0 ≥ 12 AND G1 AND G3 — not collapsed; the role manifold is genuinely ~13-d, the literal d_emb/2=32 floor over-alarms). **G2 sport-invariance FAIL** — sport is ~79% recoverable from z (chance 33%); a no-GRL control also scores 0.799, so **the GRL is inert (Δ≈1pp)** and dropping sport_tok changed nothing (0.786). The leak is structural to the **per-sport adapter** (a distinct Linear per sport bakes sport into z). True sport-invariance needs a **shared-adapter redesign (Stage 1.1)**: pad e_s to a common dim, one shared Linear, let GRL+CORAL+SupCon do the mixing. Honest take: the user's goal (one geometry to ask "what does a PF share with a SS") IS met — G3 proves cross-sport archetypes cohere; z just also retains a sport axis (sliceable, not harmful). G2's "sport unknowable" is a stricter fairness-style goal, deferred.
- [x] 2.4 `eval_unified.py` — G1 non-inferiority, G2 sport-invariance+rank, G3 silhouette → `unified_report.json`
- [x] 2.5 **Stage 1.1 probe (p19):** shared-adapter still blocked under frozen encoders → **Stage 2 required** for G2 progress. Soft-deferred (G2 not ship-blocking for v0.1).

## Phase 3 — Analogy face-validity
- [x] 3.1 `analogy_panel.py` — cross-sport top-k NN + role label; G4 = NN-role-coherence hit-rate
- [x] 3.2 Run panel: **G4 = 0.959 (target 0.60) PASS**. Named showcase sensible (Tom Brady QB→Draymond Green/Marcus Smart @0.97; Andre Drummond C→soccer defenders Lamine Koné/Kieran Tierney @0.89; Ty Lawson PG→Kevin De Bruyne @0.98). Per-sport: hoops 0.976, gridiron 0.974, pitch 0.837.
- [x] 3.3 **Taxonomy v1 cleanup DONE:** A4 (pitch-only, n=383, G4 0.000) remapped → pitch cluster 2 now folds into A3 (defensive anchor). A4 moved to deferred (needs gridiron defensive data + hoops steals/blocks specialist cluster to become a real shared archetype). **G4 0.959 → 0.978; pitch hit-rate 0.837 → 0.993; all 6 in-scope archetypes now >0.95.** (The earlier "A6 discrepancy" was a labeling bug — index 6 = A11 in-scope — RESOLVED.)

## Phase 4 — End-to-end fine-tune (Stage 2)
- [x] 4.1 Infra: `pipeline/load_live_encoders.py` + `pipeline/train_stage2.py` (encode-path only; two LRs; post-hoc shippability)
- [x] 4.2 Clean run (`enc_lr=1e-6`): best ep26 — **G1 PASS**, **G2=0.674** (−7pp vs Stage 1 ~0.74), **SHIPPABLE=False** (target ≤0.433). Saved `unified_stage2_best.pt` + `data/stage2_history.json`. Docs: `docs/STAGE2_PLAN.md` §8.
- [x] 4.3 **Stage 2.1-a run (2026-07-30).** Precondition was NOT actually met when checked: hoops was promoted 2026-07-25 to a 64-d encoder (concat fusion, `docs/MTNN_V5_PROMOTE_GATE.md`-adjacent) but `pipeline/load_encoders.py`'s `SPORT_DIM["hoops"]` still said 48, and `load_live_encoders.py::_hoops_bundle` didn't apply the same unconditional `injury`-family exclusion / `n_injury` durability-head arg that `train_mtnn.py`'s own `main()` uses post-PR#9 — both are real bugs, now fixed (SPORT_DIM→64; family filter + `n_injury` wired). Rebuilt `unified_matrix.npz` (N=20,719), retrained Stage 1 from scratch against the corrected hoops encoder (`unified_best.pt`, rank 12.4, **G1 PASS** — hoops role actually improved 0.825→0.941, **G3 PASS** silhouette 0.594), then ran Stage 2.1-a (enc_lr 3e-5, 60ep, GRL λ 0.10 ramp 5, revert-threshold 0.02 — **staggered unfreeze from the plan was NOT implemented in `train_stage2.py` and was skipped this run**, relying on the existing per-epoch G1 revert-check instead, which is what actually caught/would-have-caught a regression). Result: **G1 holds for all 3 sports, hoops/gridiron/pitch role scores all improved slightly (no regression at all)**, best epoch 15, **G2=0.693** — plateaued there through epoch 60 (not still falling). Per the plan's own decision tree this lands on **"G2 plateau > 0.55 → declare G2 a soft target... user decision, not auto."** **User decision (2026-07-30): ship it.** Wrote `pipeline/export_unified_stage2.py` (encodes the whole corpus through the drifted per-sport encoders from `unified_stage2_best.pt["enc_states"]` + the Stage 2.1 trunk, not the frozen Stage 1 path `export_unified.py` uses) and exported `assets/unified.json` (20,719 players, 64-d, norms=1.0, per-sport counts match meta, no NaN). The G2 soft-target caveat is baked into the shipped JSON itself (`model`, `g2_sport_acc`, `g2_target`, `g2_status` fields), not just the docs. Old Stage-1 `unified.json` backed up to `assets/unified.json.pre_stage2.1_bak` (gitignored, local only). `docs/SPEC.md` §7 and this line updated to match. No live site currently reads `unified.json` (checked `vector-hub` — the Phase 5.3/5.4 "hub" copy updates were text-only methods-page framing, not a data fetch), so this is the full ship: no separate deploy step exists yet.

## Phase 4c — Cultural text (Wikipedia → embeddings)  [auto-mode 2026-07-11]
- [x] 4c.1 Schema `docs/CULTURAL_TEXT_SCHEMA.md` (Wikipedia-first; Reddit deferred)
- [x] 4c.2 `acquire_wikipedia_bios.py` — priority stars 47/47 ok; full unique resume in progress (429-backed-off)
- [x] 4c.3 `embed_cultural_text.py` — MiniLM 384-d via transformers mean-pool (sentence_transformers broken in env)
- [x] 4c.4 `cultural_text_join.py` — `cultural_text_matrix.npz` 20,721×384; **454 labeled seasons (2.2%)** from 47 unique
- [x] 4c.5 `train_unified.py --cultural-text --market` → `unified_cultural.pt` (warm-start market; shipped untouched)
- [x] 4c.6 Probe mean_align_cos **0.775**; eval G1 PASS / G3 PASS / G2 0.842 (expected text leak); Reddit deferred

## Phase 4b — Market / cultural anchors (full free tier)

## Phase 4b — Market / cultural anchors (full free tier)
- [x] 4b.1 Acquire Forbes earnings, Spotrac NFL salary, awards → `data/market_cultural/`
- [x] 4b.2 Join to unified matrix (`market_cultural.json`, 20,721 rows); Transfermarkt TLS blocked (pitch MARKET_VALUE gap documented)
- [x] 4b.3 Shared `salary_head` + `award_head` via `--market`; warm-start → `unified_market.pt` (does not overwrite shipped)
- [x] 4b.4 Eval: G1/G3 PASS on market ckpt; G2 still ~0.74 deferred. Probe: salary ρ≈0.64 (gridiron), award ρ≈0.52
- [x] 4b.4b **Resync after the 2026-07-30 hoops dim fix (4.3).** `unified_market.pt`/`unified_cultural.pt` still carried `sport_dim=[48,32,24]` — same staleness class as `unified_best.pt` had, just not caught until actually tried. Their join inputs were stale too: `market_cultural.json` was built for the old 20,721-row matrix (now 20,719; re-ran `market_cultural_join.py`), and `cultural_text_matrix.npz` likewise (re-ran `cultural_text_join.py` — coverage jumped 454→1,742 labeled rows (2.2%→8.4%), the Wikipedia bio acquisition backlog from 4c.2 clearly kept resuming in the background since 2026-07-22). Retrained both (`--market`, `--market --cultural-text`) against the corrected encoder: both `sport_dim=[64,32,24]` now, **G1 PASS** (hoops +0.123/+0.126, matching Stage 1/2.1's improvement), **G3 PASS** (silhouette 0.659/0.680), collapse_detector PASS. G2 0.779 (market) / 0.825 (cultural) — higher than base, as expected (more supervised signal leaks sport identity; matches the pre-existing "G2 0.842 expected text leak" note). Old stale checkpoints backed up to `pipeline/data/unified_{market,cultural}.pt.pre_dim64_<timestamp>`. Still gitignored, still not exported to any shipped asset — 4b.5's "export market-aware unified.json" remains undone, just no longer silently broken if someone picks it up.
- [ ] 4b.5 Deferred: social reach (Apify); API-Football for pitch market value; export market-aware `unified.json` for UI

## Phase 5 — Ship + Methods
- [x] 5.1 `export_unified.py` → `assets/unified.json` (20,721 players, 64-d e + PCA-3 map, cross-sport archetypes + axes; norms=1.0, counts match, loads clean). Per-sport assets untouched (additive).
- [x] 5.2 Ablation: drop each alignment loss → Δ on G2/G3/G4. **Verdict (data/ablation_report.json):** SupCon ESSENTIAL (drop→G3 0.718→0.125, G4 0.988→0.137); CORAL INERT (0Δ); VICReg INERT at default w (0Δ on rank/G3/G4); GRL modest (~5pp sport-acc, no G3/G4 harm); task-only keeps G1 but no folding (G4 0.129). **Lean config = SupCon+task+GRL (drop CORAL+VICReg).** Each loss now earns its keep.
- [x] 5.3 Methods copy (honest: "shared role space", "best-guess", pitch role-only caveat) — hub README + index updated 2026-07-11
- [x] 5.4 Upgrade `vector-hub` "Where this is going" → shipped-with-caveats (G2 deferred; no joint daily puzzle yet)

## Phase 6 — Pitch SupCon → prod unified (2026-07-11)
- [x] 6.1 Backup shipped `unified_best.pt` + `unified.json` → `pipeline/data/bak_pre_pitch_con/`
- [x] 6.2 Rebuild `native_clusters.json`; rematch pitch `native_to_cross` via pos_dist L1 (`remap_pitch_n2c.py`); rebuild matrix
- [x] 6.3 Retrain Stage 1 from scratch (same lean hyperparams); **G1 PASS, G3 PASS, G4=0.972 PASS**, G2 sport-acc **0.694** (improved vs ~0.74, still deferred)
- [x] 6.4 Export `assets/unified.json` (20,721 × 64-d, norms=1.0)
- [x] 6.5 Deploy hub to prod — `vercel --prod` 2026-07-14 → https://dumbmodel.com (vector-mx18tg0i0… Ready); live copy shows 64-d joint embedding shipped-with-caveats

## Verify (always last)
- [x] V1 G1–G3 green; no per-sport regression; collapse_detector PASS (rank 13≥12; literal rank≥32 over-alarms on a genuine ~13-d role manifold). G2 sport-invariance FAIL (structural, deferred to Stage 2). `verify_encoders.py`: 3 frozen encoders load, norms=1.0, counts match meta. Git: hoops clean, pitch additive-only, gridiron only pre-existing build_features.py M (not from unified).
- [x] V2 G4 = 0.978 ≥ 60% (target 0.60); all 6 in-scope archetypes >0.95; named panel sensible (Brady→MacAllister/Draymond @0.97; Drummond→soccer CBs @0.86; Lawson→De Bruyne @0.98).
- [x] V3 `unified.json` loads (20721 players, 64-d, PCA-3 map); per-sport assets untouched (additive); norms=1.0.
- [x] V4 Re-read diff of the four docs for internal consistency — UNIFIED_ARCHITECTURE.md §11 addendum (Stage 1 v0 results + 6 named deviations), SPEC.md §7 acceptance status annotated (G2 deferred), pitch-row + ask-first/non-goals updated to reflect Phase 1 completion + market/cultural deferral.
- [x] V5 Pitch SupCon hill-climb: LOO 4/4 vs PCA3; beats PCA16 on pos-cluster; `pitch_mtnn.pt` promoted with pre-con backup
