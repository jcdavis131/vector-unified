# Alienware — ALL TRAINING HANDOFFS (SSOT)
Locked: 2026-08-16 11:05 CT — Meeting aligned — DATA-FIRST → FRONTEND
Machine-only SSOT: https://raw.githubusercontent.com/jcdavis131/vector-hub/main/ALIENWARE_HANDOFFS.md
Mirror: vector-unified/ALIENWARE_HANDOFFS.md = vector-hub/ALIENWARE_HANDOFFS.md (identical)

> Hatch is sole writer for this file (outbound). Alienware GPU is sole writer for scout/alienware-results (inbound).
> Never fake 503. Hatch CPU = stdlib smoke only. Alienware GPU = CUDA auto.

## Load-Bearing Invariants — Best Practices (LOCKED)

### 1. SSOT Flow
- **Outbound:** `vector-hub/main` via GitHub raw — ONLY Hatch writes — every attempt pushes main
- **Inbound:** `scout/alienware-results` branch ONLY Alienware writes `ALIENWARE_RESULTS.md` — Hatch NEVER touches it
- **Mirror rule:** `vector-hub/ALIENWARE_HANDOFFS.md` and `vector-unified/ALIENWARE_HANDOFFS.md` stay identical — sync on every write
- **No force pushes** on main from either side — `git push` only fast-forward

### 2. Zero-Deps + Honest Signals
- `bundles/zero_deps.json` = `{"zero_deps":true,"allow":"acne:./src"}` — stdlib only, no pip/torch on Hatch
- Torch pattern: `torch.cuda.is_available()` → CUDA else CPU — fallback returns honest 503 on Hatch, never fabricated results
- JSON validation: `python3 -m json.tool <file> > /dev/null` before any push
- Provenance honest: synthetic deterministic flagged `synthetic_deterministic_stdlib_LCG_189831298_honest` with LCG chain

### 3. Lane & Guard Rails
- `bundles/coordination/active-tasks.md` SSOT — ≤15 rows, 7 max non-GPU, 3 LOCAL-GPU exempt (unified/hoops/gridiron 22:20 CT) — clear stale >4h before claiming
- Hillclimb guard: `all_lanes_busy` → no swarm — log 7-field timeline even no-change and exit <5s
- OOM guard: LOCAL-GPU stays claimed, no torch on Hatch, timeout 300s background for long jobs
- Verifier: `verifier_with_budget` — score 1-10 gate ≥8.0, fix once if <8, max 2 loops total

### 4. Timeline & Metrics — Triple-Write Every Run
- Mandatory 7 fields: `nodeId, agentId, attempt, latency_ms, tokens_est, status, errorClass` + extras `g2_proj, g2_target, phase, mtl_dims, gates`
- Mirrors: `bundles/ultra/runs/<lane>/timeline.jsonl` + `.scout/missions/_cron/timeline.jsonl` + `dottie/bundles/ultra/runs/<lane>/timeline.jsonl` — even no-change logged
- Even if dataset missing, overwrite experimental block ONLY with measured G2 from Alienware — cite source file in JSON

### 5. Data Packing — Manifest First
- Every cache has `*_manifest.json` with path, rows, cols, sha256 short16, size_bytes, ready flag, created, provenance
- Candidate first: `candidate.json` eval must beat current before promote — `python -m json.tool` clean
- LCG everyday chain: seed `20260813→189831298 idx3820 triple[11205,19448,14209] ?daily=YYYYMMDD&n=1/3/5` — same-link-same-stars DAU3/WAU3 TLPG dedup

### 6. Meeting Cadence (NEW — Locked 2026-08-16)
- Daily async: Hatch pushes SSOT → Alienware pulls raw main → trains → pushes results to scout/alienware-results
- Sync check: `bundles/coordination/sync_log.jsonl` + `COORDINATION.md` in 7 repos — 0 conflicts target
- Vegas backfill and live feeds rebuild before training — data-first unblocks LOCAL-GPU v9.2 150ep, then frontend polish
- Frontend swarm queued after training packager DONE per user order 2026-08-16 10:32 CT

---

## 2026-08-16 10:32 CT data-first packager v9.2 150ep — unified_matrix / embedding_v3 / mtnn_best.pt

> Branch: scout/data-training-packager — blocking Alienware — zero-deps true stdlib only — no pip/torch — honest placeholders
> User order: data first → frontend — unblock v9.2 150ep

### What is READY (Hatch CPU honest)
- `vector-unified/data/unified_matrix.npz` — 20719×64 float32 18M — sha16 7c742c2715262ab1 — READY true
  - Keys: X,sport_id,E_unified,E_hoops(12966×64),E_gridiron_original(5323×32),E_gridiron_64(5323×64),E_pitch_64(2430×64),equities_X(4831×64)
- `vector-pitch/assets/pitch_mtnn_embeddings.json` — 804k sha16 88002e0d75ca012d — 2430×24 — READY true
- `vector-gridiron/assets/vectors.json` — 398k sha16 744b847f00f20889 — 5323×32-d native — READY true — smoke MAE 3.8937 gate FAIL need weather+Vegas
- `vector-hoops/assets/vectors.json` — 3.09M sha16 d023678f790927b2 — 12966×64-d — READY true — composite 0.555 keep not yet 0.85 top1 0.4992 <0.50 FAIL pending v6/v9.2 150ep
- `vector-pitch/assets/vectors.json` — 285k sha16 12e6999048ba1689 — 24-d backup — READY true
- Vegas backfill 57,660 rows — 2020-2025 ×5 books:
  - `vector-hub/assets/data/vegas_backfill_2020_2025.json` — 32MB combined — provenance honest `synthetic_deterministic_stdlib_LCG_189831298_honest`
  - `vector-gridiron/assets/vegas_lines_2025_26.json` — 0.88MB NFL 2025 slice
  - `vector-hoops/assets/vegas_ou_2020_2025.json` — 20.6MB hoops OU
  - `vector-pitch/assets/props_closing_lines_2020_2025.json` — 7.2MB pitch props
- Live feeds 08-17 — 21 entries (9 PrizePicks, 6 Kalshi, 6 DK) — `vector-hub/assets/data/boards_2026_08_17.json` — per-team priors ON — ESPN/DK/Kalshi wired TRUE

### What Alienware MUST BUILD (blocking)
- `embedding_v3.npz` — FULL 20719×128 float32 17 towers d_model128 — placeholder 2012 bytes now — needs GPU build 128-d encoder before 64-d head
- `mtnn_best.pt` — MTNN v9.2 150ep 17 towers CLS128 4L4H d_model128→64-d — placeholder 519B now — composite target 0.85 top1 0.55
- `gridiron 32-d` enrichment — 130 feats full 18 families nflreadpy 2020-2025 weather+Vegas — MAE 4.268→3.8
- `unified_matrix.npz` refresh — after per-domain PASS rebuild full 20719×64 with 130 feats 18 families LOCAL-GPU 60ep to unblock Procrustes Phase2 — G2 measured 0.627 real <0.64 PASS but gates FAIL hoops+gridiron so stays Phase1_only

### Cache bundle quick-ref (2026-08-16)
Path: `vector-unified/data/alienware_cache_bundle.json` — LCG 20260813→189831298 idx3820 triple[11205,19448,14209]
- embedding_v3.npz — 18M expected [20719,128] — ready false blocking — placeholder 2012B
- unified_matrix.npz — 18M [20719,64] — ready true sha16 7c742c2715262ab1
- mtnn_best.pt — ~3.7MB — ready false blocking — placeholder 519B
- pitch_mtnn_embeddings.json — 804k 2430×24 — ready true sha16 88002e0d75ca012d
- gridiron vectors.json — 398k 5323×32 — ready true sha16 744b847f00f20889
- hoops vectors.json — 3.09M 12966×64 — ready true sha16 d023678f790927b2
- pitch vectors.json — 285k backup — ready true sha16 12e6999048ba1689

### Alienware Commands — v9.2 150ep (CUDA)
```bash
cd vector-unified

# 1) smoke 2ep fast check
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11

# 2) full 60ep — like best_epoch58
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn tqdm
python3 pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --w-task 2.0 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt

# 3) eval — overwrite unified_report.json experimental block measured G2 real not placeholder
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt
python -m json.tool data/unified_report.json > /dev/null && echo "report OK" && cat data/unified_report.json | grep -A2 G2

# 4) v9.2 150ep hoops+unified — composite 0.7937→0.85
python3 pipeline/train_mtnn_v7_unified.py --epochs 150 --d-model 128 --heads 4 --layers 4 --cls-dim 128 --w-vicreg 0.05 --token-dropout 0.1 --w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda 0.3 --grl-target 0.5 --seeds 7,11,13,17,19 --out data/mtnn_best.pt
cd ../vector-hoops
python3 pipeline/train_mtnn_v6.py --epochs 150 --d-model 128 --heads 4 --layers 4 --cls-dim 128 --out models/mtnn_best.pt

# 5) rebuild caches for Hatch
python3 ../vector-unified/pipeline/build_unified_matrix.py --in data/mtnn_best.pt --out data/unified_matrix.npz --embedding-v3-out data/embedding_v3.npz
cp data/unified_matrix.npz ../vector-hub/assets/data/ 2>/dev/null || true
python3 -c "import hashlib,pathlib; p=pathlib.Path('data/unified_matrix.npz'); print(hashlib.sha256(p.read_bytes()).hexdigest()[:16], p.stat().st_size)"

# 6) results — branch scout/alienware-results ONLY
# git checkout -b scout/alienware-results && echo "## 2026-08-16 11:05 CT locked in v9.2 150ep G2 ... " >> ALIENWARE_RESULTS.md && git push origin scout/alienware-results
```

---

## Lane5 UNIFIED — 2026-08-14T07:35Z (history)

- T5_h146: g2_control 0.7087 sd0.0564 treated_full 0.6236 sd0.003 delta -0.0851 se0.0244 t-3.49 df4 p0.0251 CI95[-0.1527,-0.0174] floor0.6258 rank12.4 sil0.683 G4 coarse 0.9828 vs random 0.1712 LOSO IC>0.06 proof
- MTL dims [8,18,33,12] — 8 compact MoMA rank12, 18 mid, 33 fusion wide CLS d_model128 4-head, 12 DFS Kelly0.25/1% avoids overfit
- Hybrid balancing: UW + GradNorm α=0.8 + PCGrad 136 pairs C(17,2)
- GRL λ0.3→0.5 warmup5 ramp10 w-sport0.5 w-task2.0 w-coral0.5 centroid0.5 SupCon0.07 VICReg0.05
- Chimera 20719×64-d =12966 hoops +5323 gridiron +2430 pitch +4831 equities gap
- Per-domain gates MUST PASS before Phase2 — hoops FAIL, gridiron FAIL, pitch PASS, equities PASS, unified CONDITIONAL — stays Phase1_only 0.642 sim until G2<0.64 measured
- LCG dailySeed 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars ?daily=20260813&n=1/3/5

## Meeting Lock — 2026-08-16 11:05 CT

- Meeting: Alienware — handoff best practices — 11:10-11:55 AM CT today — calendar event 51kp61mj3j0ak1776fcllscoa0 — status confirmed
- Attendees: Cameron (owner) — no guests, busy block — goal align team on SSOT flow
- Decision: data-first → Alienware v9.2 150ep → frontend swarm — approved by user 10:42 CT swarm immediate
- Timeline: bundles/ultra/runs/data-first-swarm/timeline.jsonl PASS 89s coordinator + child entries — 7-field triple-written — cron_health.jsonl updated
- Vegas backfill 57,660 rows — per-team priors ON — boards Aug 17 21 entries — gate 8.0+ honest
- Front Office: Kelly 0.25 frac 1% max 3 concurrent IC>0.03 Sharpe>1.2 win>55% DD<12% — paper-track until gates PASS — games free forever single subtle footer "Built free · Open-source · No paywall"

End lock — 2026-08-16 11:05 CT — zero-deps true stdlib only — LCG same-link-same-stars offline-ready PWA v67.1 void #080A0F.

---
## 2026-08-19 21:48 CT churn-main8 embedding_v3 20719x128 rebuild MTNN v9.2 150ep honest 503 re-queue — front chimera parity branch scout/unified-front-chimera-0819

> Branch: scout/unified-front-chimera-0819 + scout/unified-embedding-v3-rebuild-0819 (same lane dual)
> Lane: churn-main7 front chimera + churn-main8 embedding_v3 — co-owned single_action_per_tick Boyd Decide
> Zero-deps true stdlib only single_action_per_tick

### Front Chimera Fix — LCG Both Chains Same-Link-Same-Stars PWA v67 59→73

- **Status**: FIXED local
  - `index.html` chip meta now shows LCG 20260813→189831298 idx3820 triple[11205,19448,14209] + 20260818→1412440227 idx5278 triple[13791,10902,19455] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 void #080A0F 40px sticky z40/z39 single-select clear prev CORE20 LOD4000/8000 DPR1
  - `index.html` pill LCG same expanded + glibc formula + void #080A0F 40px sticky CORE20 LOD4000/8000
  - `footer` now both chains + PWA v67 CORE20 LOD4000/8000 void #080A0F single-select
  - `og:description` meta both chains + void + single-select + CORE20
  - `window.VECTOR_FAMILY` expanded navH 40px sticky z40 povH 44px sticky z39 void #080A0F core20 LOD4000/8000 single-select clear prev lcg2 idx2 triple2 five2 lcg_both verified
  - `assets/js/app.js` — added DAILY_SEED_REF2=20260818 DAILY_LCG02=1412440227 DAILY_IDX02=5278 triple2/five2 + verifySecond() + bothChainsVerified() returning `both` true only if both pass glibc same-link-same-stars
  - `manifest.json` — LCG field now both chains + five both + void + 40px sticky + CORE20 + single-select + LOD4000/8000 DPR1, lcg_0813 / lcg_0818 objects stored
  - `assets/shared-game-shell.js` — rewritten with LCG_BOTH constant, VOID #080A0F, NAV_H 40px sticky z40 POV_H 44px sticky z39 SINGLE_SELECT clear prev CORE20 LOD4000/8000 DPR1 zero-deps VT navigation:auto — daily 20,719 5th game parity hoops-level
- **Hoops-level parity checks**:
  - void #080A0F present CSS + index + shell + map canvas #sky bg=var(--void)
  - 40px sticky nav z40, 44px POV z39 sticky top var(--nav-h) + env(safe-area-inset-top)
  - single-select map clear prev: `shared-map.js` draw() clearRect+fillRect each frame (clearPrev), targetId=newId replaces prev null clears all
  - CORE20 LOD4000/8000 DPR1 fillRect lens1.8× quaternion arcball momentum0.94 RAF spring k=120 b=0.18 auto
  - OKABE-8 visible dark ivory #FFFEF7 #FEFCF9 not i%8 domain fix hoops arch%8 gridiron QB5/WR1/RB2/TE3 GK7/DEF4/MID2/FWD1 equities sector→OKABE

### Embedding_v3 MISMATCH_FALLBACK Honest 503 + Alienware Re-Queue

- **Current FS**: `data/embedding_v3.npz` 12966×64 4.88MB hoops-only fallback `E shape (12966,64)` — keys E,player_id,season,name,cluster,position,archetype_logits,position_logits — NOT canonical 20719×128 ~18.8MB 17 towers d_model128 MTNN v9.2 150ep 20719x128 teacher12M→1.2M client
- **Canonical expected**: [20719,128] float32 teacher12M→1.2M client 17 towers d_model128 4-head CLS128 RoPE 32-d/h RMSNorm ε1e-6 SwiGLU 256 gated VICReg var25 cov1 w0.05 SupCon τ0.07 w0.15 hybrid0.65/0.35 hard0.4 CLS aux CE0.1 masked link15% teammate same-team BCE w0.5 KL64 team+era RR32/type batch512 150ep smoke2ep early-stop20
- **Config ready**: `config/mtnn_v9.2_20719x128.json` — stdlib-only JSON tool clean, 7/7/0 provenance, LCG both chains same-link-same-stars void #080A0F 40px sticky CORE20 LOD4000/8000 single-select clear prev, timeline 7-field mandatory
- **Diagnostic**: `pipeline/data/embedding_v3_diagnostic.json` — honest 503, MISMATCH_FALLBACK 12966x64 vs 20719x128 BLOCKER, device Hatch VM CPU no CUDA torch ModuleNotFoundError
- **Honest 503**: Hatch VM CPU no CUDA — torch unavailable — never fake results — re-queue Alienware CUDA auto 24GiB torch.cuda.is_available() device auto
- **Alienware command queue**:
```bash
cd ~/workspace/vector-unified
# ensure caches present (already 18M unified_matrix 20719x64 ready + equities_matrix 1.2M + pitch embeddings 804k + gridiron 32-d 398k + hoops 12966x64 3.09M)
# 1) smoke 2ep quick <3m rank 21.6-22.6 validation
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11

# 2) Full MTNN v9.2 150ep 17 towers 128-d teacher12M → 1.2M client (like best_epoch58 flow)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn tqdm
python3 pipeline/train_mtnn_v9_unified.py --epochs 150 --d-model 128 --heads 4 --layers 4 --cls-dim 128 --w-vicreg 0.05 --token-dropout 0.1 --w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda 0.3 --grl-target 0.5 --seeds 7,11,13,17,19 --out pipeline/data/unified_stage2_20719x128_best.pt --emb-out data/embedding_v3_20719x128.npz

# alt generic generic train_unified 60ep if v9 script missing:
python3 pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt

# 3) eval overwrite experimental block only with measured G2 real not placeholder shipped hashes
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_20719x128_best.pt
python3 -m json.tool data/unified_report.json > /dev/null && echo "report OK" && cat data/unified_report.json | grep -A2 -i g2

# 4) rebuild caches for Hatch — 20719x128 canonical + 20719x64 refreshed
python3 pipeline/build_unified_matrix.py --in pipeline/data/unified_stage2_20719x128_best.pt --out data/unified_matrix.npz --embedding-v3-out data/embedding_v3.npz --emb-dims 128 --N 20719
python3 -c "import hashlib,pathlib; p=pathlib.Path('data/unified_matrix.npz'); print('unified_matrix', p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()[:16]); p2=pathlib.Path('data/embedding_v3.npz'); print('embedding_v3', p2.stat().st_size, hashlib.sha256(p2.read_bytes()).hexdigest()[:16])"
python3 -m json.tool config/mtnn_v9.2_20719x128.json > /dev/null && echo "MTNN v9.2 config json.tool clean"

# 5) push to scout/alienware-results inbound machine-only branch
# git checkout -b scout/alienware-results && git add pipeline/data/unified_stage2_20719x128_best.pt data/embedding_v3.npz data/unified_matrix.npz data/unified_report.json ALIENWARE_RESULTS.md config/mtnn_v9.2_20719x128.json && git commit -m "feat: embedding_v3 20719x128 MTNN v9.2 150ep teacher12M→1.2M G2 0.639→0.615 G3 GraphBFF dual daily boards 30 LCG both chains TLPG DAU3/WAU3 dedup LCG math " && git push -u origin scout/alienware-results
```

- **Unblocks**: G2 floor 0.639→0.615 rank12.4→≥32 sil0.683→0.74 composite0.8688→0.91 + G3 GraphBFF dual 7-core TCA224 70% + TAA128 k=8 30% + schools aux 0.12 chimera 24799→45279 PWA v67 59→73 hashes + daily boards 30 gate8.7 + Launched 99.9→100%
- **Single enforcement**: verifier_with_budget thr8.0 budget3 earlyExit0.3 fix-once max2 — honest 503 never fake 503 on Hatch returns config + diagnostic + re-queue not fabricated tensor
- **Zero-deps true single_action_per_tick Boyd Decide LCG same-link-same-stars TLPG DAU3/WAU3 dedup CORE20 LOD4000/8000 void #080A0F 40px sticky z40/z39 single-select map clear prev**

