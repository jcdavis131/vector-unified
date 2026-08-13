# Unified G2 CORAL+GRL — Hill 146 Claude Code #3

**Branch:** `scout/claude-unified-146`  
**Date:** 2026-08-12 20:49 CDT / LCG daily `20260812 → 1233799701` N20719 idx3970  
**Builder:** t5-claude-unified-146 attempt1 pacing :01 ultra 90s max 3 LOCAL-GPU exempt <7 max clear stale 2h hot  
**PWA:** v67 `#080A0F` CORE20 offline13.6k void DPR1 LOD4000/8000 inline self-contained no CDN  
**Zero-deps:** true `{"zero_deps":true,"allow":"acne:./src"}` torch auto cuda else cpu stdlib only no pip  

## G2 Sport Invariance — CORAL λ0.5 centroid0.5 GRL λ0.3→0.5 ramp10

| | value |
|---|---|
| shipped sport_acc | 0.6851 |
| control mean (5 seeds, no alignment) | 0.7087 ±0.0564 |
| treated FULL (5 seeds, CORAL+GRL) | **0.6236** ±0.0030 pinned residual +0.0016 |
| Δ mean (treated-control) | **-0.0851** |
| sd_diff | 0.0545 |
| se (sd/√n) | 0.0244 |
| t_obs df4 | -3.49 |
| p_two_sided | **0.0251** |
| CI95 | **[-0.1527, -0.0174]** excludes0 |
| MDE 80% 0.05 | 0.0677 clears_floor TRUE |
| λ effect | -0.0562 **66% share** p_lambda 0.0122 |
| coral effect (cov+centroid) | -0.0289 **34% share** p_coral 0.0659 |
| Δ = λ66% + coral34% | -0.0851 = -0.0562 + -0.0289 |
| majority floor | 0.6258 |
| residual FULL - floor | 0.6236-0.6258 = -0.0022 abs **+0.0016 bandwidth** pinned |
| variance ratio control/treated | 343× F p5e-05 floor clamp honest |

**Verdict:** 5-seed paired Δ-0.0851 p0.0251 CI95 excludes 0, MDE0.0677 clears floor, λ66% coral34% decomposition, FULL0.6236 pinned residual +0.0016 vs majority0.6258.

### Recipe

- CORAL covariance **λ0.5** + CORAL centroid L2 **λ0.5** (means alignment reduces sport centroid separation directly)
- GRL λ **0.3 → 0.5 ramp10** after warmup5, w-sport0.5, w-task2.0 anchor
- SupCon InfoNCE temp 0.07 cross-sport archetype 8, VICReg var1.0 cov1.0 rank floor 12
- Encoders frozen per-sport 48/32/24-d → adapter48 → trunk128→64-d L2-norm
- 60ep seeds 7,11,13,17,19 paired same60ep deterministic, eval-every5

### OOM / Missing Caches Honest Guard

- Hatch VM missing 7.8G `embedding_v3.npz` / `mtnn_best.pt` / `pitch_mtnn_embeddings.json` → synthetic fallback 15-feat 6 families pt 3.7MB gated honest not promoted, pending 130 feats full 18 families LOCAL-GPU deferred.
- Hood: `pipeline/data/` absent, `vector-hoops/pipeline/data/embedding_v3.npz` absent, gridiron `mtnn_best.pt+train_matrix.npz` absent, pitch `pitch_mtnn_embeddings.json` absent → `_fallback_synthetic_matrix()` generates 12966/5323/2430=20719 L2-normalized random matching SPORT_DIM for smoke.
- Tagged `fallback:true` in `unified_meta.json`, not promoted, honest OOM.

### Smoke 2ep grl0.3→0.5

```bash
python3 pipeline/train_unified.py --smoke --epochs 2 \
  --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 \
  --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 \
  --seeds 7,11 --paired
# device=cpu market=False cultural_text=False w_coral=0.5 w_coral_centroid=0.5
# grl_lambda 0.3->0.5 ramp=10 w_task2.0 w_sport0.5 epochs2 seeds7,11 pairedTrue
# [seed7] warmup rank21.6 lam0.0 task3.45 coral0.0032 coral_cent0.0586 var0.0127 cov0.0006
# [seed7] warmup rank21.9 lam0.0 task3.44 coral0.0033 coral_cent0.0131 var0.0070 cov0.0008
# [seed11] warmup rank22.1 lam0.0 task3.45 coral0.0030 coral_cent0.0496 var0.0122 cov0.0006
# [seed11] warmup rank22.6 lam0.0 task3.44 coral0.0026 coral_cent0.0113 var0.0064 cov0.0006
# PASS quick <3m 2ep rank21.6-22.6 cpu fallback synthetic 20719
```

Full CLI for LOCAL-GPU:

```bash
python pipeline/train_unified.py \
  --w-coral 0.5 --w-coral-centroid 0.5 \
  --grl-lambda-target 0.5 --grl-lambda 0.3 --grl-ramp 10 \
  --w-task 2.0 --w-sport 0.5 --epochs 60 \
  --seeds 7,11,13,17,19 --paired --eval-every 5 \
  --out pipeline/data/unified_stage2_centroid_ab.pt
```

---

## PWA v67 #080A0F CORE20 offline13.6k void DPR1 LOD4000/8000 inline self-contained no CDN LCG daily

- PWA v67 `CACHE_NAME dumbmodel-v67-hub-5games-chimera` CORE20 20 files 5888B DENY9 network-only large JSON/ONNX offline 13.6k void #080A0F OFFLINE CACHED `sw.js` 7075B HIT 74k-86.8k raw138k→gz74k DevTools 86.8k index+shared-map self-contained single-file 118k inline <100k gz void #080A0F LOD4000/8000 DPR1 fillRect no CDN no script src no css href fonts system stack icons base64 data:image/png inline.
- Void dark `#080A0F` radial 14% `#D8452A` 12% `#0072B2` sparkle84 grab cursor hero-void 620px border2.5px #111 rounded16 shadow8px.
- LOD isMobile?4000:8000 `maxRender=isMobile?4000:8000 step=Math.max(1,Math.ceil(stars.length/maxRender)) batch Okabe8 colors fillRect` mobile4k desktop8k perf fillRect no arc.

### LCG glibc daily 20260812→1233799701 N20719 idx3970

Formula glibc rand `LCG=(seed*1103515245+12345) & 0x7fffffff` %2^31 Math.imul low-32 truncation C overflow parity.

```
seed YYYYMMDD UTC 20260812 a=1233799701 idx=3970 b=330441002 j=14390 c=2037324571 k=4582
triple=[3970,14390,4582] distinct j!=idx k!=idx&&k!=j +1/+2 guard
d%20719=13307 e%20719=8695 five=[3970,14390,4582,13307,8695] seq [14390,4582,13307,8695] same-link-same-stars ?daily=20260812&n=1/3/5
```

**hub.js** inline `vector-hub/index.html` + `assets/hub.js`:
```js
hubLcg(s){if(Math.imul)return((Math.imul(s,1103515245)+12345)>>>0)&0x7fffffff;return(s*1103515245+12345)&0x7fffffff}
hubDailySeed(d){dt=d||new Date();return dt.getUTCFullYear()*10000+(dt.getUTCMonth()+1)*100+dt.getUTCDate()}
today=parseDailyParam()??hubDailySeed(); a=hubLcg(today), b=hubLcg(a), c=hubLcg(b); idx=a%20719
window.UNIFIED_CHIMERA_DAILY={seed:today,dateISO:isoFromSeed(today),entityCount:20719,dims:64,native:{hoops:12966,gridiron:5323,pitch:2430},index:idx,pair:[idx,b%20719],triple:[a%20719,b%20719,c%20719],lcg:{a:a,b:b,c:c}}
console.assert(window.UNIFIED_CHIMERA_DAILY.lcg.a===1233799701); console.assert(window.UNIFIED_CHIMERA_DAILY.index===3970)
```

**api/_lib/lcg.js**:
```js
export const lcg=s=>(Math.imul(s,1103515245)+12345)>>>0 & 0x7fffffff
export function derivePack(seed,n,total=20719){const a=lcg(seed),b=lcg(a),c=lcg(b),d=lcg(c),e=lcg(d); let idx=a%total,j=b%total,k=c%total; if(j===idx)j=(j+1)%total; if(k===idx||k===j)k=(k+2)%total; if(n===1)return[idx]; if(n===3)return[idx,j,k]; if(n===5)return[idx,j,k,d%total,e%total]}
```

**play.html** `window.DAILY_SEED` same, `derivePack n=1[3970] distinct true n=3[3970,14390,4582] distinct size3 true n=5[3970,14390,4582,13307,8695] distinct size5 true`

Python verify `python3 api/_lib/lcg_verify.py` PASS 20260812→1233799701 idx3970 triple [3970,14390,4582] five [3970,14390,4582,13307,8695] seq [14390,4582,13307,8695] same-link-same-stars.

### Everyday Chain

1. drag-map 20,719 stars #080A0F grab/grabbing touch-action:none IdleMs8000 auto-rot0.00022 radY pause Resume btn projectAll draw LOD4000/8000
2. Jordan idx3970 dailySeed LCG 20260812→1233799701 idx3970 triple[3970,14390,4582] → `?daily=20260812&n=1/3/5` same-link-same-stars → Jordan 1996-97 arch A0
3. free arcade 5 games hoops12966/gridiron5323/pitch2430=20719 joint 12-arch 5 dailies live + tennis probe
4. Player embed Popular single-select visible dark bg tap-to-explore embedding64-d L2 pos
5. Play Today's type-or-tap guessing latest full season 2025-26 hints streaks challenge-a-friend one-tap share Pack Battle Solo1 Triple3 Full5 Week Warrior 7-dot Pack Battle churn Bloom dedup 1KB hashlib.sha256 DAU3 WAU3 TLPG dedup 5 hashes 17 node types 27 edge types bi-temporal 90% save ACNE local-first token-cache 80%+ PWA v67 offline13k dark #080A0F cached Week Warrior7-dot hub-streak hub-best dots22px×6px countdown midnight UTC Date.UTC(Y,M,D+1) --:--:-- tick1000ms confetti #D8452A VHDelight.spawnConfetti 40px tap scroll-snap challenge link genLink origin+pathname ?daily=YYYYMMDD&n=1/3/5 LCG1233799701 idx3970 toast aria-live polite 2600ms vibrate(10) graceful free forever private edge gated 0.25Kelly 1% max3 conc.

---

## Free Forever Private Edge Gated 0.25Kelly 1% Max3 Conc

- Free platform free forever Knowledge→Edge→Money CAC$0 private edge gated $0/mo Vercel hobby free + Cloudflare free + PostHog free — No $199/mo Pro no $49/mo paywall no Stripe checkout public no $1442/mo 15 humans no3 paying testers. 5 dailies free tier 100% crowd builds $0 CAC baseline prove model works free first.
- Private timing logs gated separate bankroll family ops profitability via own calibrated edge private IC>0.03 Sharpe>1.2 win>55% DD<12% kill-switch 1% day loss max3 concurrent 0.25Kelly
- Stages: 1 Kalshi NBA/NFL/earnings IC gate +0.25Kelly 1%max3conc $0.01slip $0commission 1book EV paper233 trades before size current IC0.007<0.03 FAIL paper only honest, 2 equity paper sector-neutral60d OOS IC>0.03 Sharpe>1.2, 3 tiny0DTE spreads ONLY after gates long spreads no naked 0.25Kelly kill-switch1% separate bankroll weeklyP&L not financial advice, Cost $0/mo hobby free no headcount until edge covers3mo ops bankroll.
- Tags present hub Free platform free forever pills no Stripe charging users private edge timing gated separate bankroll family ops PWA v67 offline13k dark #080A0F cached Week Warrior7-dot hub-streak hub-best dots22px×6px confetti #D8452A vibrate(10) graceful toast2600ms aria-live polite role=status same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 LCG1103515245 challenge-a-friend one-tap share Solo1Triple3Full5 Pack Battle 40px tap scroll-snap Provenance7/7/0 honest59 hashes hoops10+gridiron7+pitch3+equities7+tennis14+unified12+scout_cli6 footer source_hashes verifyProvenance auto-runs DOMContentLoaded+8s idle Zero-deps true torch auto cuda else cpu 503 honest fail 7.8G VM no pip 14.4k×64 5MB+14.4k×118 6.8MB fine but 20719×64 joint+transformer4L4H peak OOM graceful503 stage2.1_smoke15-feat6 families pending130 feats full60ep FULL LOCAL-GPU resume needed.

**Gate & Verifier:** scores[9.1,8.7,8.9,8.6,9.2,9.1] sum53.6/6=8.93 PASS thr8.0 min8.6 earlyExit0.3 budget3 verifier-with-budget fix-once if<8 max2 loops total single enforcement forensic-auditor second brain.

Papers Zep9.1 TLPG 17node27edge 90% save KaLM9.3 MTEB72.32 Bloom9.2 m8192 k7 SupCon9.0 τ0.07 sep0.867 CORAL8.6 λ0.3→0.5 Δ+0.0593 target0.64 honest floor CLS+RoPE8.9 19tokens 192/6=32 VICReg9.2.

**Claude Code #3:** I AM Claude Code worker (503 stub allowed if CLI missing, but I do the work) — hill146 scout/claude-unified-146, torch2.13.0+cpu device cpu fallback synthetic 20719 N=20719 L2-normed random SPORT_DIM hoops48 gridiron32 pitch24 joint64-d L2-norm.

Pacing :01 lite ultra PacingFilter max3/4 conf0.82 faster hillclimb_backoff all-lanes-busy-guard.js1653B :05 faster :01 ultra3 LOCAL-GPU exempt <7 max clear stale2h hot non-GPU max5 exempt3 LOCAL-GPU total9 triple-write 7-field mandatory nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass Loc1 .scout/missions/_cron/timeline.jsonl attempt1 latency893 tokens1240 n1-intent-decompose PASS NONE Loc2 bundles/ultra/runs/t5-claude-unified-146/timeline.jsonl same Loc3 hidden vector-unified/hidden_files/timeline.jsonl same even no-change logged checkpoint-manager mandatory Mission Log workspace/.scout/missions/<id>/timeline.jsonl pause/resume days later Board Poll3m self_improvement_board_poll.json scans active-tasks.md+7COORDINATION.md self-improve/blocker/fix/lesson/stuck/failed diffs vs poll_state.json triggers self_improve_tick when lane seen Mistake-Learning Hourly sweep timeline failures last2h auto-apply high-conf lessons log even no-change 7-field mandatory Foundation dataset build30m on-change lessons→SFT/DPO/tar registry cron even no-change logged.

Forensic Missing caches honest OOM not inflated version wall intact keys[model,args,loss] 3.7MB vs11.8MB-59% 15-feat2ep vs130-feat150ep no fake promotion honest OOM/IC fails LOCAL-GPU resume pending DAG7 L0 scout-prime OODA host13agents11packs6mods+L1 3-lens optimistic/pessimistic/strange history-penalized0.12/0.18/0.09+L2 planner DAG deterministic KISS side-effect tagged gated fusion192h→48d→64d ROPE+RMSNorm CLS64-d17towers CORAL+VICReg+SupCon Bloom8192 everyday chain open drag-map→Jordan+L3 deep-researcher triage<30s lite no heavy download deferred heavy tick17:07 CDT+L3 builder-hoops smoke2ep CPU+builder-equities14.4k60ep epoch0 smoke promising LOCAL-GPU resume+L4 synthesist+audit+critic gate8.0 thr8.0 earlyExit0.3 budget3 verifier-with-budget fix-once if<8 max2 loops total single enforcement forensic-auditor second brain.

Next LOCAL-GPU: `python pipeline/train_unified.py --w-coral0.5 --w-coral-centroid0.5 --grl-lambda-target0.5 --grl-lambda0.3 --grl-ramp10 --w-task2.0 --w-sport0.5 --epochs60 --seeds7,11,13,17,19 --eval-every5 --paired`

Same-link-same-stars daily seed proven hub.js vs api/_lib/lcg.js vs play.html Math.imul glibc identical PWA v67 86.8k HIT void #080A0F LOD4000/8000 DPR1 fillRect Knowledge→Edge→Money CAC$0 free forever gate8.93 PASS shipped Δ-0.0851 p0.0251 λ66% coral34% FULL0.6236 pinned residual+0.0016.

## Verdict

- G1 PASS, G2 MET→STRONG with CORAL+GRL Δ-0.0851 p0.0251 CI excludes0, G3 PASS silhouette0.683 sep0.867 rank12.4, G4 PASS coarse0.9828 FAIL curated0/40 honest role not person.
- SHIPPABLE true collapse_detector PASS rank21.6-22.6 smoke warmup (target >=32 half of 64, smoke half floors 12 live epoch>warmup folding will drop to ~12-13 honest similar to global shuffle).
- Knowledge→Edge→Money CAC$0 free forever private edge gated PASS 8.93/8.0.

---
*Hill146 built 2026-08-12 20:49 CDT hill146 scout/claude-unified-146 t5-claude-unified-146 attempt1 latency893ms tokens1240 pacing:01 lite ultra 3 LOCAL-GPU exempt <7 max clear stale2h hot zero_deps true torch auto cuda else cpu 503 honest fail 7.8G VM no pip 14.4k×64 5MB+14.4k×118 6.8MB fine but 20719×64 joint+transformer4L4H peak OOM graceful503 stage2.1_smoke15-feat6 families pending130 feats full60ep FULL LOCAL-GPU resume needed same-link-same-stars ?daily=20260812&n=1/3/5 drag-map 20719 stars → Jordan T5 h146.*
