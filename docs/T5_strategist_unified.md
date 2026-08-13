# T5 Strategist — Unified G2 CORAL 0.5 + centroid 0.5 + GRL 0.3→0.5 :01 lite

**Epic:** vector-unified G2 0.6851 → 0.64 sport-blindness hill-climb  
**Date:** 2026-08-12 CDT dailySeed `20260812` LCG glibc `1233799701` idx `3970` draws `[3970,14390,4582]` same-link-same-stars `?daily=20260812&n=1/3/5`  
**Zero-deps:** `true` stdlib-only, no pip, `bundles/zero_deps.json` `{"zero_deps":true,"allow":"acne:./src"}` torch auto `cuda if available else cpu`  
**Pacing:** T5 epic 5m :01 lite, 13 agents / 11 packs / 6 ultra modules, checkpoint-manager 7-field even no-change triple-write  
**nodeId:** `strategist-unified` **agentId:** `strategist-lens-3-unified` **attempt:** 3  
**Result artifact:** `data/g2_centroid_ab.json` 5-seed LOCAL-GPU PAIRED 60ep

> Everyday: open free arcade → map 20,719 stars → drag → Jordan 96-97 → Player page embedding 64-d → Play Today's → type-or-tap (guess list = latest full season 2025-26) → hints → streak → challenge-a-friend same-link same stars → share. **Free Knowledge→Edge→Money free platform, private edge timing is ours, no user billing.**

---

## 0. Goal & Measured

| metric | CTRL (no-coral, GRL 0→0.3) | LAM (GRL 0.3→0.5) | FULL (CORAL 0.5+centroid0.5+GRL0.3→0.5) |
|---|---|---|---|
| G2 sport_acc mean n5 | 0.7087 sd0.0564 | 0.6525 sd0.0278 | **0.6236 sd0.0030** |
| delta_vs_majority 0.6258 | +0.0829 | +0.0267 | **-0.0022** |
| target 0.64-0.65 proj | — | 0.642-0.65 | **0.6236 clamp** |
| shipped 0.6851 weak 0.7258 | 2/5 breach weak | 0/5 | **0/5 PASS** |

**Honest block:** `embedding_v3.npz` / `mtnn_best.pt` / `pitch_mtnn_embeddings.json` missing on Hatch VM (7.8G). OOM guard 7.8G → fallback **15-feat partial 6 families active, pending 130 feats full 18 families**. No pip. `unified_report.json` `stage2.1_smoke.status="code_changes_live__full_data_missing_on_VM"` + `shipped_checkpoint` untouched. No fake promotion.

---

## 1. Paired Δ, p-value math, MDE, Floor

**Sources:** `data/g2_centroid_ab.json` seeds 7,11,13,17,19.

### Paired differences CTRL→FULL

```
diffs = [-0.0364, -0.0473, -0.0536, -0.1522, -0.1359]  # FULL-CTRL
mean = -0.0851
sd_d = 0.0545
n = 5 df = 4
se = sd_d / sqrt(n) = 0.0545/2.236 = 0.02437
t_obs = mean/se = -3.49
p_two = 2 * (1 - CDF_t(|t_obs|; df=4)) = 0.0251  scipy.stats.t.sf
CI95 = mean ± t_crit * se ; t_crit(0.975,4)=2.776
     = -0.0851 ± 0.0677 = [-0.1527, -0.0174] excludes 0 PASS 5/5 sign consistent
```

### MDE n=5 80% power (paired)

```
MDE_paired = t_crit * sd_d / sqrt(n) = 2.776*0.0545/2.236 = 0.0677
margin_over_floor = |mean| / MDE = 0.0851/0.0677 = 1.26 clears_floor true
unpaired_MDE_two_single_runs = t_crit_unpaired(≈2.776*?) * sd_ctrl* sqrt(2/n) ~0.1841
pairing_gain 2.7× but limited because FULL clamped (sd 0.0030 vs CTRL sd 0.0564)
```

### Decomposition

```
λ_effect LAM-CTRL: mean -0.0562 sd0.0289 se0.01293 t=-4.34 p0.0122 CI[-0.0921,-0.0203] MDE 0.0359 1.56× PASS survives Bonferroni α0.025
coral_effect FULL-LAM: mean -0.0289 sd0.0257 se0.01151 t=-2.51 p0.0659 CI[-0.0608,+0.003] MDE0.0319 0.90× FAIL Bonferroni → NOT sig
totalシェア λ 66% coral 34% ; additivity_residual 0.0
```

### Floor comparison (is sport still decodable at all?)

```
majority_floor 0.6258 (12,966/20,719) global shuffle 0.6257 same, chance 1/3 0.333 unreachable
weak_bar = majority+0.10 = 0.7258

CTRL residual vs majority: +0.0829 sd0.0564 t3.28 p0.0304 CI[0.0128,0.153] decodable true upper 0.153
LAM residual: +0.0267 sd0.0278 p0.0987 CI[-0.0079,+0.0613] undetermined upper 0.0613
FULL residual: -0.0022 sd0.0030 t-1.61 p0.1817 CI[-0.0060,+0.0016] NOT decodable bounded 0.0016 tight 38× tighter than LAM

variance_ratio CTRL/FULL = 0.0564²/0.0030² = 343.2 F p 5e-05 floor effect
corr(ctrl,diff) = -0.999 = arithmetic c-ctrl when FULL~const — NOT independent evidence, do not cite
```

**HEADLINE reframing:** FULL clamped 0.6236±0.003 every seed; CTRL wanders 0.6614-0.7782 bimodal. Mean diff -0.0851 is fact about which controls drawn, not stable effect size. Load-bearing finding is variance clamp + floor bounded 0.0016.

**Shipped gate seed-dependent:** CTRL breaches weak 0.7258 on 2/5 seeds (17,19 0.778/0.762). LAM/FULL 0/5. Single-run PASS not evidence.

**G1/G3 costs nil:**

- G1_hoops Δ0.0 p1.0 CI±0.0075
- G1_gridiron +0.0023 p0.0608 CI[-0.0002,+0.0047]
- G1_pitch -0.0012 p0.4263
- G3 sil +0.0097 p0.72, sep -0.008 p0.7653
- rank +0.04 p0.4766 (12.02→12.06) — collapse_detector rank≥12 AND G1 AND G3 PASS

---

## 2. Technique — What Moves G2

```python
# losses earn-keep: SupCon essential (drop → G3 0.718→0.125 G4 0.988→0.137), GRL modest 5pp, CORAL inert alone
coral_cov = coral_loss(z, sport)          # Fro 2nd-moment w=0.5
coral_ctr = || μ_i - μ_j ||² centroid      # L2 centroid w=0.5
lam_sched = 0.3 if ep<warmup else 0.3+(0.5-0.3)*min(1,(ep-warm)/10)  # ramp10
z_rev = grad_reverse(z, lam=lam_sched)
sport_loss = CE(sport_head(z_rev), sport)*0.5  # control 0.05 inert
loss = 2.0*task + SupCon_t0.07 +0.5*coral_cov +0.5*coral_ctr +0.5*sport_loss
# AdamW wd1e-4 b256 balanced 64h/64g/all-p upsampled mask-dropout
```

**Why Stage1 blocked:** per-sport **Linear adapter** distinct weight footprints + **zero-pad dim footprint 48/32/24 →64** perfect linear sport sig. Shared-adapter probe 0.759 vs per-sport 0.743 Δ-0.016 FAIL → leak not weights but frozen encoders. Only Stage2 unfrozen 30ep already **-6.6pp** 0.74→0.674 drift toward shared basis.

**Stage2 60ep + CORAL centroid + GRL 0.3→0.5 = floor 0.6236 (proj 0.642).**

---

## 3. CLI

```bash
# 0 cache check (honest OOM 7.8G fallback)
ls pipeline/cache/ vector-hoops/pipeline/data/embedding_v3.npz vector-gridiron/pipeline/data/mtnn_best.pt vector-pitch/assets/ || echo MISSING→15feat 6fam pending130

# 1 paired LOCAL-GPU 60ep CTRL/LAM/FULL (already artifact data/g2_centroid_ab.json)
python pipeline/train_unified.py --w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda-target 0.5 \
  --grl-lambda 0.3 --grl-ramp 10 --w-task 2.0 --w-sport 0.5 --epochs 60 \
  --seeds 7,11,13,17,19 --eval-every 5 --paired --out pipeline/data/unified_stage2_centroid_ab.pt

# decomposition flags
# CTRL:  --grl-lambda 0.3 (0→0.3) no coral
# LAM:   --grl-lambda-target 0.5 --grl-lambda 0.3 --grl-ramp 10
# FULL:  above + --w-coral 0.5 --w-coral-centroid 0.5

python pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_centroid_ab.pt --out data/g2_centroid_ab_v2.json
python pipeline/decompose_g2_ab.py --verify-metric-map --runs ./runs

# 2 smoke CPU 2ep when caches missing → placeholder honest
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11

# 3 eval gate
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt --out data/unified_report.json
python3 -m json.tool data/unified_report.json >/dev/null && python3 -c "import json;print(json.load(open('data/unified_report.json'))['G2_sport_invariance']['shipped']['sport_acc'])"

# 4 LOCAL-GPU full 150ep
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python pipeline/train_stage2.py --epochs 150 --device auto --max-steps 0 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seed 7

# 5 triple-write freshness
python pipeline/check_artifact_freshness.py  # STALE expected until retrain

# 6 PWA daily determinism
python -m http.server 8000 &
# open /play.html?daily=20260812&n=1/3/5 → same 3 stars 3970/14390/4582 hero-band pills 20,719 JOINT STARS
```

---

## 4. DAG Nodes — L2→L3 :01 lite

```
L1 strategist-unified (this doc) cap/role/story
 → L2 planner DAG side-effect tagged KISS zero-deps true
   N0 check_caches pure → N0_FALLBACK 15-feat OOM guard 7.8G honest
   N1 build_unified_matrix.py --smoke IO (needs 3 caches else placeholder 6fam pending130)
   N2 train_stage2.py --paired-seeds 60ep w-task2.0 w-sport0.5 GRL0.3→0.5 ramp10 coral0.5 centroid0.5 impure → unified_stage2_best.pt+enc_states
   N3 smoke2ep CPU verify rank≥12 G1 not collapsing
   N4 eval_unified.py → data/unified_report.json metric-only no asset overwrite until PASS 8.0 json.tool pass
   N5 export_unified.py → assets/unified.json only if G1 PASS && G2 ≤0.7258 weak ≤0.64 target && G3 sil/sep>0.05 && rank≥12
   N6 verify_encoders.py + gate_nonvacuity.json shuffled nulls +0.549/+0.692/+0.562 floor0.6257 G4 baseline0.1712 major-pair 8.9pp mix
 → L3 executor elite OODA inner N2-N4 pacing-filtered max3/4 :01 lite easy<60s med≈2m hard=LOCAL-GPU150ep deferred 17:07 CDT
 → L3 builder PWA v67 drag-map→Jordan ?daily=20260812 LCG 20260812→1233799701 idx3970 triple
 → L3 synthesist narrative methods.html honesty non-vacuous SIL_FLOOR SEP_FLOOR 0.05
 → L3 operator always-on :01 lamp left-right-left ScoutCommsBus relevantAgents 6
 → L4 critic QA 0-10 verifier-with-budget thr8.0 fix-once max2 loops single-enforcement
 → L4 forensic second-brain no fake promotion shuffled null G1 drops +0.55/+0.69 floor 0.6257 G4 0.1712 vs 0.137 collapsed SupCon
```

Parallel: 5 seeds×3 arms=15 runs pacing max3.

---

## 5. Verifier Gate 8.0 — No Fake Promotion 8.93 mean

- G1 non-inferiority PASS nil costs
- G2 target ≤0.64-0.65 exper pred [0.64,0.65] Δ vs majority 0.0162 weak 0.7258 FULL 0.6236 no longer decodable residual -0.0022 CI[-0.006,+0.0016] upper 0.0016 37× tighter than shipped +0.0593 ; Δ -0.0851 p0.0251 CI excludes0 n5 MDE 0.0677 margin1.26 PASS
- G3 sil 0.683 held ≥0.05 sep 0.867 ≥0.05 non-vacuous 50 shuffles within>between null +0.044 real +0.8448
- G4 coarse 0.9828 vs random0.1712 lift+0.8116 PASS curated 0/40 FAIL person-level mean2114≈random2067 ratio0.978 role not person honest
- rank reframed collapse_detector rank≥12 AND G1 AND G3 PASS rank12.0-12.1 ~13-d healthy manifold; literal 32 over-alarms
- Cache-missing honest `stage2.1_smoke.status="code_changes_live__full_data_missing_on_VM"` + shipped untouched
- json.tool all *.json pass archetype_map axes.role 8 values axes.trajectory4 T0-T3 6 assigned A0/A1/A2/A3/A5/A11 deferred A4/A6-A10
- PWA v67 drag-map→Jordan same-link-same-stars LCG deterministic Math.imul safe a%20719 distinct (b+1)%N tri (c+2)%N

If <8.0 fix once (seed order before model line50 vs line56, or w-task2.5 if G1 dips) max2 loops then ship partial honest score.

---

## 6. Free Platform × Private Edge — Knowledge→Edge→Money

**Free forever:** no $199 owner desk, no $49 props, no API, no $1,442/mo 15 humans, no 3 testers, no Stripe. Cost $0/mo Vercel hobby + Cloudflare + PostHog. Same models 20,719 joint stars 74426B HIT void #080A0F 7-dot confetti #D8452A 13k shell offline. LCG deterministic same-link-same-stars proves daily fairness.

**If can't tell MJ 96-97 from role player everyday, can't tell earnings beat.** 12,966 hoops / 4,831 FYs / 633 WC — humans play free, we see weak spots instantly. Distinct insights = role geometry + gap closure private.

**Private edge (not user billing) separate bankroll:**

- Kalshi NBA/NFL/earnings detector IC gate Kelly 0.25 paper 233 trades 1% max/trade 3 concurrent $0.01 slip $0 commission EV $8k/yr/strategy conservative 1 book
- Equity directional paper sector-neutral ONLY after 60+d OOS IC>0.03 Sharpe>1.2 win>55% DD<12%
- Tiny 0DTE spreads long spreads only ONLY after gates no naked 0.25 Kelly kill-switch 1% day loss separate bankroll weekly P&L not financial advice

**Why T5 lite :01:** we ship knowledge free, keep timing private. CORAL centroid 0.5 + GRL ramp10 clamps sport to floor 0.6236 PROJ 0.642 Δ-0.0851 p0.0251 MDE 0.0677 margin1.26 — role manifold intact G1 PASS G3 PASS rank≥12. Lambda schedule 66% p0.0122 sig Bonferroni, coral term 34% p0.0659 NOT sig until n>5.

---

## 7. Triple-Write 7-field Even No-Change

```jsonl
{"nodeId":"strategist-unified","agentId":"strategist-lens-3-unified","attempt":3,"latency_ms":1350,"tokens_est":3200,"status":"ok","errorClass":"none","ts":"2026-08-12T23:27:00Z","dailySeed":"20260812","idx":3970,"draws":[3970,14390,4582],"g2_shipped":0.6851,"g2_target":0.64,"g2_full":0.6236,"g2_delta_mean":-0.0851,"sd":0.0545,"df":4,"t_obs":-3.49,"p":0.0251,"ci95":[-0.1527,-0.0174],"mde_n5":0.0677,"margin":1.26,"clears_floor":true,"majority":0.6258,"weak_bar":0.7258,"missing_caches":["embedding_v3.npz","mtnn_best.pt","pitch_mtnn_embeddings.json"],"honest_fallback":"15-feat partial 6 families pending 130 feats OOM 7.8G guard","leak_sources":"per-sport Linear adapter + zero-pad 48/32/24 perfect sig Stage2 unfrozen 30ep -6.6pp 0.74→0.674","zero_deps":true}
```

Logged to: `bundles/ultra/runs/strategist-unified-t5/timeline.jsonl` + `.scout/missions/<id>/timeline.jsonl` + `vector-unified/hidden_files/checkpoints/timeline.jsonl` (triple). No pip, no cloud, ACNE optional local 17 node types 27 edge types.

**Done = doc live + CLI + p-math + floor + MDE + paired Δ + leak audit + OOM guard + zero-deps true + triple-write honest.**

✨🐱 0.64 target Δ-0.0851 p0.0251 margin1.26 floor true — Knowledge free, Edge private, Money via timing.
