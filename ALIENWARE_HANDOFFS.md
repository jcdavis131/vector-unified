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
