# MTNN v9.2 20719×128 Prep — 2026-08-30 12:41 CDT — churn-main8

## Current State (Hatch CPU zero-deps)
- embedding_v3.npz: (12966,64) 4.88MB fallback — FINITE true L2 mean 1.0 erank 24.1 rank strict 64 singular top5 [48.42,44.66,35.29,33.72,30.66] — FAIL shape/size/rank vs canonical
- embedding_v3_20719x64.npz: (20719,64) 5.31MB intermediate READY — not 128
- unified_matrix.npz: (20719,64) 18M READY E_unified L2 max_abs0.90783
- unified_matrix_with_schools_full_27181.npz: (47900,64) 12.26MB FULL chimera READY 27,181 schools 27,181 NCES CCD 2023-24 real 0.832 PASS
- mtnn_best.pt: 4.54MB READY legacy
- torch: missing — ModuleNotFoundError honest 503 — Hatch CPU cannot train, Forge must train
- provenance: 7/7/0 59 hashes — need 73 for PWA v67

## Canonical Target
- 20,719×128 float32 ~18.8MB (10,619,? actually 20719*128*4=10,608,128 bytes ~10.6MB raw + overhead ~18.8MB npz compressed? Spec says ~18.8MB)
- Teacher 12M 224-d → client 1.2M 64-d sphere → 128-d teacher distilled
- MTNN v9.2 17 towers d_model128 4-head CLS128 4L RoPE 32-d/h RMSNorm ε1e-6 SwiGLU 256 gated VICReg var25 cov1 w0.05 SupCon τ0.07 w0.15 hybrid0.65/0.35 hard0.4 CLS aux CE0.1 masked link15% BCE w0.5 KL64 RR32/type batch512 150ep smoke2ep early-stop20

## Gates
- shape: 20719×128 or 20719×64+teacher — FAIL current 12966×64
- size: ~18.8MB — FAIL 4.88MB
- finite: PASS
- L2-normalized: PASS mean 1.0
- rank≥32: FAIL erank 24.1 <32 (fallback rank12.4 per board)
- sil≥0.74: FAIL 0.683 measured vs 0.74 need
- composite≥0.91: FAIL 0.8688 vs 0.91
- G2≤0.615: FAIL 0.639 vs 0.615 floor lock
- freshness 7/7/0: FAIL 59 hashes not 73
- PWA 59→73: FAIL pending canonical
- overall: BLOCKER

## Cameron Pause
> Pause training — Cameron says rebuild MLOps 0→1 first, end-to-end, no train gate yet.

Allowed: MLOps prep, smoke 2ep, cache checks, honesty eval, one real-data forward pass. Do NOT start full G2 60ep gridiron 60ep pitch 60ep.

## Forge Command (single lane hot)
```bash
cd ~/workspace/vector-unified
python3 pipeline/train_unified.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11
# DO NOT run full 60ep until Cameron lifts pause
python3 pipeline/build_unified_matrix.py --with-schools --embed-v3 --d128
python3 pipeline/eval_unified.py --embed data/embedding_v3.npz
```

## MLOps 0→1 Checklist (prep done Hatch CPU)
- [x] ingest: schools 27,181 real NCES verified DONE 2026-08-19 14:36 CDT
- [x] clean: 27,181 rows real_data.json 11M 27,181 verified keys NCESSCH/SCH_NAME/ST
- [x] featurize: 20,719 chimera inputs READY (12966 hoops + 5323 gridiron + 2430 pitch)
- [ ] bundle: embedding_v3 20719×128 missing — needs Forge
- [ ] ship: PWA v67 59→73 blocked by bundle
- [ ] monitor: daily boards 30 gate8.7 LIVE pending bundle
- [x] zero-deps true stdlib only
- [x] honest 503 true no synthetic
- [x] LCG 20260813→189831298 idx3820 triple[11205,19448,14209] + 20260818→1412440227 idx5278 triple[13791,10902,19455] same-link-same-stars

## Timeline 7-Field (even no-change)
- nodeId: churn-main8-embedding-v3
- agentId: scout/unified-embedding-v3-rebuild-0819
- attempt: 1
- latency_ms: 2850
- tokens_est: 4520
- status: 503 (Hatch) / PENDING (Forge)
- errorClass: UpstreamDown (GPU absent intentional)
- Message: Hatch CPU honest 503 torch missing — Forge single lane hot queued MTNN v9.2 150ep canonical 18.8MB — prep done cache verified fallback documented — full G2 60ep paused per Cameron MLOps 0→1

## Branch
scout/unified-embedding-v3-rebuild-0819 — current branch verified, config committed pending push

## Next
- Alienware Forge executes MTNN v9.2 150ep → embedding_v3 20719×128
- Validate gates: shape/size/finite/L2/rank≥32/sil≥0.74/composite≥0.91/G2≤0.615/freshness 7/7/0/PWA 59→73
- After gates PASS → unblock G2 floor lock, G3 dual, chimera 47900, PWA v67, daily boards, Launched 99.9→100%
- Topology 1 main +1 churn-aligner c86e297d MUST stay alive +N=4 churn +3 LOCAL-GPU exempt

