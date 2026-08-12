# Vector Unified

![CI](https://github.com/jcdavis131/vector-unified/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

One joint embedding — 20,719 player-seasons across hoops+gridiron+pitch folded into 64-d cross-sport chimera.

Live at https://unified.dumbmodel.com or via https://dumbmodel.com/models/unified.html (daily game 05: chimeras, pack battles, Week Warrior streaks).

> Solo personal project, no connection to employer, built with public/free-tier only (free data pipeline, static Vercel + PWA, no backend).

> **Picking up in-progress work?** Start at [`docs/HANDOFF.md`](docs/HANDOFF.md) — current state (2026-08-09), Stage 2.1 checkpoint, G1-G4 gates, open sport-blindness hill-climb, verification commands, and pointers to SPEC + unified_report + Stage2 plans.

> **PR #5 (`bench/real-data-e2e`, cross-domain transfer probe) is DRAFT and BLOCKED — do not merge.** It trains the shared embedding on `vector-realty`'s exchange dataset, and `vector-realty` PR #4 (a fix to that data pipeline) is intentionally held for the human owner's review pending a security-flag resolution. Merging #5 before #4 resolves would ship a reported result built on pre-fix data. See `docs/HANDOFF.md` for details; do not close this note until #4 lands and #5 is re-run against the fixed data.

## The embedding

20,719 rows: hoops 12,966 (64-d MTNN 18 towers, 48-d native) + gridiron 5,323 (32-d MTNN) + pitch 2,430 (24-d MTNN StatsBomb 11 contexts, 4/10 families) → 64-d joint L2-normalized `z`, era-honest per-season z-scored.

**Alignment (each loss earns keep):**

- **CORAL** covariance diff (shape-match) + **centroid** mean-L2 (location-match, directly minimizes sport centroid separation for sport-blindness) — `w_coral 0.5` + `w_coral_centroid 0.5`
- **GRL** adversarial sport head, λ 0.10 ramp (warmup 5ep) → 0.30 → 0.5 linear ramp 10ep after, `w_sport 0.5` (was 0.05 inert in Stage 1, 0.10→0.30 in Stage 2.1)
- **SupCon** same-ticker adjacent-FY, temp 0.07 — essential (drop G3 0.718→0.125, G4 0.988→0.137, G2 leak +0.130)
- **VICReg** var hinge 1-std + cov off-diag `w_var 1.0 λ_var 25`, `w_cov 1.0 λ_cov 1`, rank floor 12
- **Task anchor** `w_task 2.0` — per-sport native-cluster + position CE, preserves role (anti-collapse)

**Staging:**

- Stage 1 frozen encoders (trunk only) → Stage 2.1 unfrozen encoders drift, enc_lr 3e-5, 60ep
- Shared lib: `ResidualTower cat([x·m,m])→96h→24d skip LayerNorm L2-norm` per sport, `TransformerFusion 128d 4-head CLS→64-d`
- Best checkpoint `pipeline/data/unified_stage2_best.pt` epoch 58 / 60ep
- Shipped asset `data/unified.json` + `assets/` (additive, per-sport untouched)

**Gates (see `data/unified_report.json`):**

- **G1** per-sport non-inferiority (baseline − joint, negative = joint better): hoops -0.0526 (0.7385→0.7911) better, gridiron 0.0 (0.9991→0.9991 ceiling, 18 features pass/rush/receive disjoint stat profiles), pitch +0.0021 (0.893→0.8909 within noise) — shuffled null +0.5493/+0.6920/+0.5617 proves PASS not buggy mask-as-index
- **G2** sport-blind: 0.6851 vs majority 0.6258 Δ +0.0593 target ≤0.7258 (0.6258+0.10) MET — weak. Floor any embedding hits (globally shuffled 0.6257), majority is floor not 1/3=0.333, retired target 0.433 UNREACHABLE (balanced-class math). Experimental projection 0.64-0.65 with GRL 0.3→0.5 + centroid
- **G3** archetype coherence: silhouette 0.683 PASS, within-arch cross-sport cos 0.746 between -0.121 sep +0.867 PASS, rank 12.4 ≥12 floor PASS, composition gap 8.9pp sport-pair mix confound (within vs between draw different hoops-gridiron vs hoops-pitch mixes — some sep is sport-pair effect, file states confound)
- **G4** cross-NN: automated 0.9828 vs random 0.1712 lift +0.8116 PASS coarse arch, curated 0/40 top-10 fail, mean B-rank 2114 vs random 2067 ratio 0.978 indistinguishable, arch-agreement 0.65 vs 0.162 baseline — space knows role, not person. SUPERSEDED salvage ratio 3.23× used N/2 random — correct E[min_k]=(N-k)/(k+1) → 0.98×

**Caveats:** sport still partly recoverable (Δ +5.93pp), G2 plateau structural (distinct Linear per sport bakes sport into z + native dim footprint 64/32/24). Shared-adapter probe failed 0.759 vs 0.743 (more recoverable) — leak is zero-padding pattern, not adapter weights. Stage 2 unfrozen drift helps but not erases.

See `docs/DATA_MODEL_unified.md`, `docs/SPEC.md`, `docs/UNIFIED_ARCHITECTURE.md`, `docs/STAGE2_PLAN.md`, `docs/STAGE2.1_SWEEP_PLAN.md`, and `data/TENNIS_CITATION_GAP.md` + `LOCAL_GPU_G2_RESULT.md` (measured, not promoted — sport no longer decodable above base rate under full treatment, residual −0.0022 CI [-0.0060,+0.0016], control +0.0829 p=0.0304).

## The site

Plain HTML/JS/Canvas/WebGL, no framework, PWA-capable (`sw.js`, `offline.html`, `manifest.json` + `assets/manifest.json` dup). Static Vercel, daily game 05:00 UTC.

- **dailySeed** deterministic: `dailySeed = YYYYMMDD` int (UTC year*10000+month*100+day), LCG glibc rand compat: `(seed*1103515245+12345) & 0x7fffffff`, index = LCG % 20719, pair = second LCG distinct, triple = third LCG distinct — same link = same stars, reproducible Python & Node agree
- **Chimera triple**: Solo 1 / Triple 3 / Full 5 — dailySeed triple [idx,j,k] wired as chimera battle, copy link, shareable challenge
- **Hero-band + map-overlay**: `shared-map.js` reuse across domains (same engine as hoops), 20,719 stars void dark true, DPR1 fillRect, LOD 4000 mobile / 8000 desktop, dragging rotates, hover any star, ★ = today's chimera pick, glass Hero, X 3PT↔Paint / Y Role→Score / Z Def↔Off orientation, keyboard-a11y + error-boundary parity hoops
- **Delight pack**: viral row 1/3/5 pack battle chips, viral-today line, toast on copy, pack CTA row same dailySeed = same chimera
- **Streak**: Week Warrior — win daily before midnight UTC to build, 7 in a row = badge, `localStorage unified-streak/unified-best`, countdown next in HH:MM:SS UTC, continue streak / share streak / challenge friend, random spins don't break streak

Pages: daily game (`play.html` mode=daily/random/lab), 3D embedding map (`model.html` + Trends `#manim`), player dossiers / directory (`players.html`), Lab Fusion cross-sport (`model.html` archetype quiz A0-A11), Methods-honest (`methods.html`), Data pipeline, Trends (30 seasons drift). `knowledge/` pattern optional future (hoops has player-wiki).

## Data pipeline

```bash
python pipeline/load_encoders.py              # smoke: load hoops/gridiron ckpts; build pitch MTNN (verify shapes)
python pipeline/build_unified_matrix.py       # e_h/e_g/e_p + archetype_map + labels → unified_matrix.npz (leak-free)
python pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5  # Stage1 frozen
python pipeline/train_stage2.py --epochs 60   # Stage2 unfrozen + centroid, lam 0.3→0.5, logs coral + coral_c + lam→target
python pipeline/eval_unified.py               # G1-G4 automatic gates → data/unified_report.json + stage2_report.json
python pipeline/export_unified.py             # → assets/unified.json (20,719×64-d L2, PCA3, archetype, prov honest)
python pipeline/analogy_panel.py              # G4 automated + curated 40 triples → analogy_report.json

# quick wins
python3 -m json.tool data/unified_report.json > /dev/null
python3 -m json.tool data/unified.json > /dev/null
python tools/dashboard/server.py   # local board localhost:8000, 10s refresh, reads git + artifacts, one instance
```

Sources: hoops MTNN 12,966 seasons per-100-poss z within season, gridiron MTNN 49,881 weeks → 5,325 season-agg mean REG ≥1 snap, pitch MTNN StatsBomb 11 contexts 1,157 matches 2,430 seasons 4/10 families (volume/efficiency/defense/playmaking, no bio/market/pedigree/honors). Forbes 150 records / Spotrac 23,143 py / Wiki awards 205 (market/cultural, see `docs/VALUE_SIGNAL_CENSUS.md` + `docs/CULTURAL_TEXT_SCHEMA.md`). Every response cached `pipeline/cache/`; `--offline` rebuilds from cache only.

## Training

`train_unified.py` Stage 1 frozen-enc aligned + centroid loss, `train_stage2.py` Stage 2 unfrozen-enc alignment + centroid + GRL schedule, `eval_unified.py` G1-G4, `stage2_eval.py` history, `tools/dashboard/server.py` board.

Current Stage 2.1 best: epoch 58/60, enc_lr 3e-5, GRL λ 0.10→0.30→0.5 (warmup 5ep + ramp 10ep linear), `w_coral 0.5` cov Frobenius shape-match, `w_coral_centroid 0.5` mean L2 location-match, `w_sport 0.5`, `w_task 2.0` task anchor, SupCon same-ticker adjacent-FY temp 0.07, VICReg var hinge 1-std + cov off-diag, rank floor 12 (collapse_detector rank 12.4≥12 PASS literal 32 over-alarms on ~13-d role manifold).

- **Stage 2.1 projection**: G2 0.6851 → 0.64-0.65 ΔvsMajority 0.016, improvement -0.043, derivation: GRL λ 0.05→0.10 gave -7pp (0.74→0.685), CORAL centroid -2pp Stage1 probe, conservative -4.3pp expected — smoke 2ep (full data missing on VM, caches: `vector-hoops/pipeline/data/embedding_v3.npz`, `vector-gridiron/pipeline/data/mtnn_best.pt+train_matrix.npz`, `vector-pitch/assets/pitch_mtnn_embeddings.json`)
- **Ablation house rule** each loss earns keep: drop SupCon → leak 0.7558 +0.13 Δ vs majority (Stage1 0.771→0.799) + G3 0.718→0.125 G4 0.988→0.137 essential; drop GRL → 0.799 earns keep but ceiling ~0.68 structural adapter leak + dim footprint (64 vs 32 vs 24); drop CORAL cov Δ 0.0 alone but combined centroid+cov -1.5pp G2 probe + G3 stability (keep for G3+G2 small); CORAL centroid directly minimizes centroid separation (keep)
- **Projection method**: `data/unified_report.json` `G2_sport_invariance.experimental` + `stage2.1_smoke` fields, json.tool passing

## Running locally

```bash
python -m http.server 8000   # static site, open http://localhost:8000 or /play.html?daily=20260809
python -m json.tool ~/workspace/vector-unified/assets/data/unified.json > /dev/null  # if assets/data hop present
python -m json.tool ~/workspace/vector-unified/data/unified.json > /dev/null
python -m json.tool data/unified_report.json | head -80
python tools/dashboard/server.py  # dashboard one instance → localhost:8000
python pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt  # rewrites data/unified_report.json measured G2 when caches restored
```

No install (std lib). For pipeline: numpy+sklearn+torch CPU OK (MTNN small, trunk tiny, seed 7). Vercel `cleanUrls true` (see `vercel.json`), PWA `manifest.json` + dup `assets/manifest.json`.

## License

MIT. Solo personal project, no connection to employer, built with public/free-tier only — see `LICENSE`.
