# HANDOFF — Vector Unified (2026-08-09)

> **Current as of 2026-08-09** — Stage 2.1 shipped, Stage 2.2 hill-climb in progress for sport-blindness. This file is the on-ramp for any new operator or sub-agent picking up mid-flight.

## Where we are

- **Shipped:** `data/unified.json` (20,719 × 64-d L2-norm, 12,966 hoops + 5,323 gridiron + 2,430 pitch) + `assets/` static PWA. Best checkpoint `pipeline/data/unified_stage2_best.pt` epoch 58/60, enc_lr 3e-5, GRL λ 0.10 ramp → 0.30 → 0.50 schedule (warmup 5ep + linear 10ep), `w_coral 0.5` cov + `w_coral_centroid 0.5` centroid, `w_sport 0.5`, `w_task 2.0`, SupCon temp 0.07, VICReg var hinge.
- **Gates (see `data/unified_report.json`):**
  - G1 PASS: hoops Δ -0.0526 (0.7911>0.7385), gridiron 0.0 ceiling, pitch +0.0021 within noise. Shuffled null +0.549/+0.692/+0.562 proves not buggy mask.
  - G2 0.6851 vs majority 0.6258 Δ +0.0593 target ≤0.7258 MET — weak. Retired target 0.433 UNREACHABLE (balanced-class math, majority floor 0.6257 globally shuffled). Experimental projection 0.642 range [0.64,0.65] Δ -0.0431 → Δ vs majority 0.0162.
  - G3 PASS: silhouette 0.683, within 0.746 between -0.121 sep 0.867, rank 12.4 ≥12 floor. Composition gap 8.9pp sport-pair mix confound noted.
  - G4: automated 0.9828 vs 0.1712 lift +0.8116 PASS coarse, curated 0/40 top10 FAIL person-level (mean 2114 vs random 2067 ratio 0.978 indistinguishable) — space knows role, not person.

## Pointers

- **SPEC:** `docs/SPEC.md` — assumptions (3 encoders input, no natural cross-sport pairs, season-level, additive asset, pitch MTNN prereq, sparse families accepted, no new ingestion, greenfield, taxonomy K≈12 versioned), objective 64-d L2, commands, project structure, testing (unit/inspect/train gate/eval/panel/ablation), boundaries, acceptance (Stage 2.1 status), corrections 2026-08-03 (G1 mask-as-index bug 1.0, G2 target arithmetic, rank perm-inv, G3 vacuous test, G4 baseline).
- **Architecture:** `docs/UNIFIED_ARCHITECTURE.md` — 4 pillars (P1 frozen encoders, P2 shared trunk 64-d + sport token dim8 proposal now dropped, P3a per-sport heads anti-collapse, P3b alignment SupCon mod-aware temp + CORAL cov/centroid, P3c GRL sport-invar, P4 asset+eval). Family ontology 10 families, taxonomy A0-A11, imbalance handling, training staged, eval G1-G5, risks, research basis e5-omni/CORAL/Football2Vec v2 GRL/ConFu/GRAM/OT/EventGPT.
- **Stage2 plans:**
  - `docs/STAGE2_PLAN.md` — Stage 1.1 shared-adapter probe 0.759 vs 0.743 FAIL (leak is dim-footprint not adapter), Stage 2 impl (load_live_encoders smoke cos 1.0, enc_lr choice, stagger vs simultaneous, epochs, revert contract), execution 2026-07-10/11 clean 30ep enc_lr 1e-6 G1 PASS G2 0.674 FAIL SHIPPABLE False, ask-first decisions resolved, Next levers §8 (longer/higher enc_lr warmup-freeze/staggered/stronger GRL).
  - `docs/STAGE2.1_SWEEP_PLAN.md` — Stage 2.1 recipe (frozen→unfrozen, GRL 0.10→0.30→0.50, centroid addition, task anchor 2.0, ablations, decision tree: G2 plateau >0.55 → declare soft target ship on G1+G3).
- **Data model:** `docs/DATA_MODEL_unified.md` — joint alignment, CORAL+contrastive+adversarial, dailySeed LCG, archetypes 12, era-honest, sources.
- **Unified report:** `data/unified_report.json` + `data/unified_meta.json` + `data/unified.json` prov. `assets/eval_scoreboard.json` composite, `data/analogy_report.json` + `data/analogy_triples.json` G4 curated 40, `data/gate_nonvacuity.json` nulls calibration (G1 shuffled drops, G2 floor 0.6257, G3 SIL_FLOOR/SEP_FLOOR 0.05, G4 baseline 0.1712 validates nulls both land 0.1712).

## Open tracks

- **Sport-blindness hill-climb (current):** goal G2 0.6851→0.64-0.65 closer to floor 0.6258. Recipe: GRL λ schedule 0.3→0.5 after warmup 5ep, ramp 10ep, `w_sport 0.5` (was 0.05 inert, 0.10→0.30), CORAL cov+centroid `w=0.5+0.5` (cov Frobenius shape, centroid L2 location), task anchor `w_task 2.0`, SupCon temp 0.07, VICReg. House rule each loss earns keep — drop contrastive leakage +0.13 Stage1, drop GRL 0.799, each earns keep but ceiling ~0.68 structural adapter distinct Linear bakes sport + dim footprint. Shared-adapter blocked under frozen — Stage2 unfrozen drift helps but not erase. Code: `pipeline/train_unified.py` adds `coral_centroid_loss(z,sport)`, args `--grl-lambda 0.3`, `--grl-lambda-target 0.5`, `--w-sport 0.5`, `--w-coral-centroid 0.5`; `pipeline/train_stage2.py` `coral_loss_fn` returns cov+centroid, lam 0.3→0.5 linear.

## Verification

```bash
python3 -m json.tool data/unified_report.json > /dev/null
python3 -m json.tool data/unified.json > /dev/null
python3 -m json.tool assets/manifest.json > /dev/null
python3 -m json.tool manifest.json > /dev/null
python3 -m json.tool assets/eval_scoreboard.json > /dev/null
python -m http.server 8000   # open localhost:8000/play.html?daily=20260809
python tools/dashboard/server.py   # one instance → localhost:8000
python pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt  # when caches restored
# caches needed: vector-hoops/pipeline/data/embedding_v3.npz, vector-gridiron/pipeline/data/mtnn_best.pt+train_matrix.npz, vector-pitch/assets/pitch_mtnn_embeddings.json → pipeline/data/
python pipeline/build_unified_matrix.py --smoke   # assemble e_h/e_g/e_p+archetype_map
python pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5
```

## Caveats honest

- Sport invariance still weak — G2 0.685 only 5.93pp above majority guess, any embedding hits 0.6258 by guessing hoops.
- G3 8.9pp composition gap — within-arch vs between-arch sport-pair mix differs, some sep is sport-pair effect.
- G4 person-level retrieval FAIL 0/40 — role not person, mean rank 2114≈random 2067.
- Rank 12.4 same as global shuffle — drop CORAL no-op on rank, perm-invariant not quality, highest ranks collapsed configs 19.6/18.8 vs 12.8.
- G1 mask-as-index historical bug pos_drop 0.0 baked asset now measured true −0.0526/0.0/+0.0021.
- Pitch sparse 4/10 families — role-only cross-sport analogies involving pitch, never physical/market, UI + methods state honestly, no faked bio/market.
- `LOCAL_GPU_G2_RESULT.md` 5 seeds×3 arms measured not promoted — full treatment sport no longer decodable above base rate residual −0.0022 CI [-0.006,+0.0016] control +0.0829 p=0.0304, paired mean wrong summary treatment clamps floor 343x var ratio.
- `data/TENNIS_CITATION_GAP.md` why cited_fields red — six published tennis values cite artifacts git never carried.
- `data/seed_order_audit.json` why 3 ablation artifacts disagree full@seed7 — ablation.py seeds line56 after model line50 seed never controlled weights, one hit in 329 files across six repos.

## Dormant tracks

- Market/cultural footprint (Forbes 150, Spotrac 23k py, Wiki awards 205) wired shared heads salary+award on z masked-MSE z-scored, warm-start `unified_market.pt`, face-val ρ 0.64 gridiron salary, ρ 0.52 award pooled, sparse hoops, Transfermarkt blocked TLS, pitch salary Forbes-only, endorse Forbes-only, social reach Apify deferred, cultural text Wikipedia→MiniLM 47→454 seasons `unified_cultural.pt` mean align cos 0.775 G2 0.842 expected leak Reddit deferred — see `docs/VALUE_SIGNAL_CENSUS.md`, `docs/CULTURAL_TEXT_SCHEMA.md`, `docs/MARKET_CULTURAL_SCHEMA.md`.
- Tennis/golf feasibility `docs/TENNIS_GOLF_FEASIBILITY.md` + tennis probes `data/tennis_*`, direction axes, sector map, gate nonvacuity, archetype map.
- Contacts `CONTACTS.md`, Gotchas `GOTCHAS.md` 12 things cost time, Scheduling `SCHEDULING.md` sweep/dashboard durable, Coordination `COORDINATION.md`, Local GPU `COORDINATION_LOCAL_GPU.md` + blocker.

Last verified: 2026-08-09 — json.tool passing, no push until files pass per task, Vercel cleanUrls true, PWA manifest + assets/manifest.json dup, LICENSE MIT, bundles/manifest.json copied root.
