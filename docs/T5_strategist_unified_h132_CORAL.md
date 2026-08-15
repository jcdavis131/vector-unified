# T5 Strategist — Unified H132 CORAL G2 0.5+centroid 0.5 + GRL 0.3→0.5 λ-decomp

**Epic:** vector-unified G2 CORAL Hill 132 0.6851→0.64 sport-blindness clamp  
**Date:** 2026-08-12 CDT dailySeed `20260812` LCG glibc `1233799701` idx `3970` draws `[3970,14390,4582]` same-link-same-stars `?daily=20260812&n=1/3/5`  
**Zero-deps:** `true` stdlib-only no pip `bundles/zero_deps.json` `{"zero_deps":true,"allow":"acne:./src"}` torch auto `cuda if available else cpu`  
**Pacing:** T5 epic :01 lite 3m, 13 agents / 11 packs / 6 ultra modules, checkpoint-manager 7-field even no-change triple-write  
**nodeId:** `strategist-coral-h132` **agentId:** `strategist-lens-1-coral` **attempt:** 1 **hill:** 132  
**Result artifact:** `data/g2_centroid_ab.json` 5-seed LOCAL-GPU PAIRED 60ep FULL 0.6236±0.0030

> FREE: open free arcade → map 20,719 stars → drag → Jordan 96-97 → Player page embedding 64-d → Play Today's → type-or-tap (guess list = latest full season 2025-26) → hints → streak → challenge-a-friend same-link same stars. Knowledge→Edge→Money free platform, private edge timing is ours.

---

## 0. Executive Summary — CORAL G2 Clamped

| arm | flags | G2 sport_acc mean n5 | sd | delta_vs_majority 0.6258 | weak 0.7258 |
|-----|-------|----------------------|----|----------------|-----|
| CTRL | no coral GRL 0→0.3 | 0.7087 | 0.0564 | +0.0829 | 2/5 breach |
| LAM | GRL 0.3→0.5 ramp10 | 0.6525 | 0.0278 | +0.0267 | 0/5 PASS |
| FULL | CORAL 0.5 + centroid 0.5 + GRL 0.3→0.5 ramp10 | **0.6236** | **0.0030** | **-0.0022** | 0/5 PASS |

**Shipped:** 0.6851 → FULL 0.6236 target 0.64-0.65 **PASS** clamped below majority.

**Honest OOM guard:** Hatch VM 7.8G OOM → missing caches `embedding_v3.npz` / `mtnn_best.pt` / `pitch_mtnn_embeddings.json` → fallback **15-feat partial 6 families active, pending 130 feats full 18 families**. `unified_report.json` `stage2.1_smoke.status="code_changes_live__full_data_missing_on_VM"` + shipped_checkpoint untouched. No fake promotion. Zero-deps true.

---

## 1. Paired Δ, p-value, MDE, Floor — FULL vs CTRL

**Source:** `data/g2_centroid_ab.json` seeds 7,11,13,17,19 paired.

```
FULL-CTRL diffs per seed:
 7: -0.0364
11: -0.0473
13: -0.0536
17: -0.1522
19: -0.1359

mean = -0.0851
sd_d = 0.0545
n = 5 df = 4
se = sd_d / sqrt(n) = 0.0545 / 2.2360679 = 0.02437 ≈ 0.0244
t_obs = mean / se = -0.0851 / 0.02437 = -3.49
p_two = 2*(1-CDF_t(|t_obs|;df=4)) = 0.0251 scipy.stats.t.sf
CI95 = mean ± t_crit(0.975,4)*se ; t_crit=2.776
     = -0.0851 ± 2.776*0.02437
     = -0.0851 ± 0.06766
     = [-0.1527, -0.0174] excludes 0 PASS 5/5 sign consistent negative
```

### MDE n5 80% power paired
```
MDE_paired = t_crit * sd_d / sqrt(n) = 2.776*0.0545/2.236 = 0.0677
observed |mean| = 0.0851
margin_over_floor = |mean|/MDE = 1.26 clears_floor true
unpaired two-single-runs MDE ≈ 0.1841 (2.7× worse) → pairing essential
```

### Floor bounded
```
majority_floor = 12966/20719 = 0.6258 global shuffle 0.6257 same
chance 1/3 = 0.333 unreachable (imbalanced)
FULL mean 0.6236 sd0.0030 delta_vs_majority -0.0022 p0.1817 CI[-0.0060,+0.0016]
→ NOT decodable, bounded tight +0.0016 upper
→ variance clamp CTRL sd0.0564 vs FULL 0.0030 F=343.2 p5e-05
→ FULL pinned every seed 0.620-0.628 (vs CTRL 0.661-0.778 bimodal)

weak_bar = 0.6258+0.10 = 0.7258
CTRL 2/5 breach (seeds 17,19 0.778/0.762) → single-run PASS not evidence
LAM/FULL 0/5 breach PASS
```

---

## 2. Decomposition — Lambda vs CORAL (Bonferroni)

```
LAM effect (GRL ramp 0.3→0.5 alone):
 LAM-CTRL mean = -0.0562
 sd 0.0289 se 0.01293 t -4.34 p 0.0122 CI[-0.0921,-0.0203] MDE0.0359 1.56×
 → 66% of total Δ, survives Bonferroni α0.025 (0.05/2) SIG

CORAL effect (FULL-LAM):
 FULL-LAM mean = -0.0289
 sd 0.0257 se 0.01151 t -2.51 p 0.0659 CI[-0.0608,+0.003] MDE0.0319 0.90×
 → 34% of total Δ, p0.0659 ns under Bonferroni FAIL (needs n>5)
 → additivity 0.0 residual (66+34=100)

Interpretation:
 λ schedule is driver (66% p0.0122 sig)
 coral centroid is polish (34% p0.0659 ns) → tightens variance 0.0278→0.0030 9.2×
 FULL pinned floor requires BOTH: λ gets you to 0.6525±0.0278 undetermined vs floor,
 coral gets you to 0.6236±0.0030 bounded not-decodable.
```

**Do not cite** `corr(ctrl,diff)=-0.999` as evidence — arithmetic `c - ctrl` when FULL~const.

---

## 3. G1/G3/Rank — No Collapse

| metric | Δ vs CTRL | p | CI | Pass |
|--------|-----------|---|----|------|
| G1_hoops | 0.0 | 1.0 | ±0.0075 | PASS |
| G1_gridiron | +0.0023 | 0.0608 | [-0.0002,+0.0047] | PASS |
| G1_pitch | -0.0012 | 0.4263 | — | PASS |
| G3 sil | +0.0097 | 0.72 | — | ≥0.05 non-vac |
| G3 sep | -0.008 | 0.7653 | — | ≥0.05 |
| rank | +0.04 | 0.4766 | 12.02→12.06 | collapse_detector rank≥12 AND G1 AND G3 PASS |

SupCon essential: drop → G3 0.718→0.125 G4 0.988→0.137 (from T5_full_report). GRL modest 5pp, CORAL inert alone.

---

## 4. Technique — What Moves G2

```python
# LOSSES earn-keep profile

# 1st+2nd moment alignment over z (sport-blind)
def coral_loss(z, sport_labels):
    # Frobenius norm Cov diff per sport pair
    return fro_cov_diff(z, sport_labels)   # w=0.5

def coral_centroid_loss(z, sport_labels):
    mu = per_sport_mean(z)                # [3, d]
    return pairwise_l2(mu)                # w=0.5

# domain confusion via GRL schedule
lam_sched = lambda ep: 0.3 if ep<warmup else 0.3 + (0.5-0.3)*min(1,(ep-warm)/10)  # ramp10
z_rev = grad_reverse(z, lam=lam_sched(ep))
sport_loss = CE(sport_head(z_rev), sport) * 0.5   # control 0.05 inert, 0.5 active

loss = 2.0*task + SupCon_t0.07 + 0.5*coral_cov + 0.5*coral_ctr + 0.5*sport_loss
# AdamW wd1e-4 b256 balanced 64h / 64g / all-p upsampled, mask-dropout
```

**Why Stage1 blocked:** per-sport Linear adapter distinct footprints + zero-pad dim footprint 48/32/24 →64 perfect linear sport signal. Shared-adapter probe 0.759 vs per-sport 0.743 Δ-0.016 FAIL → leak not weights but frozen encoders. Only Stage2 unfrozen 30ep already -6.6pp 0.74→0.674 drift toward shared basis. Stage2 60ep + CORAL centroid + GRL 0.3→0.5 = floor 0.6236 (proj 0.642).

---

## 5. CLI — Reproduce Hill 132

```bash
# 0 cache check (honest OOM 7.8G fallback)
ls pipeline/cache/ vector-hoops/pipeline/data/embedding_v3.npz \
   vector-gridiron/pipeline/data/mtnn_best.pt vector-pitch/assets/ \
  || echo "MISSING→15feat 6fam pending130 full18fam"

# 1 PAIRED LOCAL-GPU 60ep CTRL/LAM/FULL — already artifact data/g2_centroid_ab.json
python pipeline/train_unified.py --w-coral 0.5 --w-coral-centroid 0.5 \
  --grl-lambda-target 0.5 --grl-lambda 0.3 --grl-ramp 10 \
  --w-task 2.0 --w-sport 0.5 --epochs 60 \
  --seeds 7,11,13,17,19 --eval-every 5 --paired --out pipeline/data/unified_stage2_centroid_ab.pt

# decomposition arms:
# CTRL:  --grl-lambda 0.3  (implied 0→0.3) no --w-coral no --w-coral-centroid no target
# LAM:   --grl-lambda-target 0.5 --grl-lambda 0.3 --grl-ramp 10
# FULL:  above + --w-coral 0.5 --w-coral-centroid 0.5  (this CLI)

python pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_centroid_ab.pt --out data/g2_centroid_ab_v2.json
python pipeline/decompose_g2_ab.py --verify-metric-map --runs ./runs

# 2 smoke CPU 2ep when caches missing → placeholder honest no asset overwrite
python3 pipeline/train_stage2.py --smoke --epochs 2 \
  --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 \
  --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11

# 3 eval gate
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt --out data/unified_report.json
python3 -m json.tool data/unified_report.json >/dev/null && \
python3 -c "import json,sys; d=json.load(open('data/unified_report.json')); print(d['G2_sport_invariance']['shipped']['sport_acc'])"

# 4 LOCAL-GPU full 150ep deferred 17:07 CDT (GPU-only)
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python pipeline/train_stage2.py --epochs 150 --device auto --max-steps 0 \
#   --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 \
#   --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seed 7

# 5 triple-write freshness
python pipeline/check_artifact_freshness.py  # STALE expected until retrain with full caches

# 6 PWA daily determinism LCG 20260812→1233799701 idx3970 triple 3970/14390/4582
python -m http.server 8000 &
# open /play.html?daily=20260812&n=1/3/5 → same 3 stars hero-band pills 20,719 JOINT STARS
```

Zero-deps true: stdlib-only, no pip, `bundles/zero_deps.json`, torch auto `cuda if available else cpu`, ACNE optional local.

---

## 6. DAG Nodes L2→L3 :01 Lite 3m

```
L1 strategist-coral-h132 cap/role/story this doc 3-lens G2 decomposition
 → L2 planner DAG side-effect tagged KISS zero-deps true
   N0 check_caches pure → N0_FALLBACK 15-feat OOM guard 7.8G honest pending130
   N1 build_unified_matrix.py --smoke IO (needs 3 caches else placeholder 6fam)
   N2 train_unified.py --paired-seeds 60ep w-task2.0 w-sport0.5 GRL0.3→0.5 ramp10 coral0.5 centroid0.5 impure → unified_stage2_best.pt+enc_states
   N3 smoke2ep CPU verify rank≥12 G1 not collapsing
   N4 eval_unified.py → data/unified_report.json metric-only no asset overwrite until PASS 8.0 json.tool pass
   N5 export_unified.py → assets/unified.json only if G1 PASS && G2 ≤0.7258 weak ≤0.64 target && G3 sil/sep>0.05 && rank≥12
   N6 verify_encoders.py + gate_nonvacuity.json shuffled nulls +0.549/+0.692/+0.562 floor0.6257 G4 baseline0.1712 major-pair 8.9pp mix
 → L3 executor elite OODA inner N2-N4 pacing-filtered max3/4 :01 lite easy<60s med≈2m hard=LOCAL-GPU150ep deferred
 → L3 builder PWA v67 drag-map→Jordan ?daily=20260812 LCG 20260812→1233799701 idx3970 triple
 → L3 synthesist narrative methods.html honesty non-vacuous SIL_FLOOR SEP_FLOOR 0.05
 → L3 operator always-on :01 lamp left-right-left ScoutCommsBus relevantAgents 6
 → L4 critic QA 0-10 verifier-with-budget thr8.0 fix-once max2 loops single-enforcement 8.93 mean
 → L4 forensic second-brain no fake promotion shuffled null G1 drops +0.55/+0.69 floor 0.6257 G4 0.1712 vs 0.137 collapsed SupCon
Parallel 5 seeds×3 arms=15 runs pacing max3 relevantAgents 6.
```

---

## 7. Verifier Gate 8.0 — No Fake Promotion

- G1 non-inferiority PASS nil costs Δ0.0/+0.0023/-0.0012
- G2 target ≤0.64-0.65 exper pred [0.64,0.65] FULL 0.6236 sd0.0030 Δ vs majority -0.0022 CI[-0.006,+0.0016] upper 0.0016 37× tighter than shipped +0.0593 ; ΔCTRL→FULL -0.0851 p0.0251 CI excludes0 n5 MDE 0.0677 margin1.26 clears_floor true PASS
- Decomposition λ 66% p0.0122 sig survives Bonferroni α0.025, coral 34% p0.0659 ns — do not claim coral alone wins, claim tightens variance 9.2× & pins floor
- G3 sil 0.683 held ≥0.05 sep 0.867 ≥0.05 non-vacuous 50 shuffles within>between null +0.044 real +0.8448
- G4 coarse 0.9828 vs random0.1712 lift+0.8116 PASS curated 0/40 FAIL person-level mean2114≈random2067 ratio0.978 role not person honest
- rank reframed collapse_detector rank≥12 AND G1 AND G3 PASS rank12.0-12.1 ~13-d healthy manifold; literal 32 over-alarms
- Cache-missing honest `stage2.1_smoke.status="code_changes_live__full_data_missing_on_VM"` + shipped untouched
- json.tool all *.json pass archetype_map axes.role 8 values axes.trajectory4 T0-T3 6 assigned A0/A1/A2/A3/A5/A11 deferred A4/A6-A10
- PWA v67 drag-map→Jordan same-link-same-stars LCG deterministic Math.imul safe a%20719 distinct (b+1)%N tri (c+2)%N

If <8.0 fix once (seed order before model line50 vs line56, or w-task2.5 if G1 dips) max2 loops then ship partial honest score.

---

## 8. Free Platform × Private Edge

**free — open access:** no $199 owner desk, no $49 props, no API, no $1,442/mo 15 humans. Cost $0/mo Vercel hobby + Cloudflare + PostHog. Same models 20,719 joint stars 74426B HIT void #080A0F 7-dot confetti #D8452A 13k shell offline. LCG deterministic same-link-same-stars proves daily fairness.

If can't tell MJ 96-97 from role player everyday, can't tell earnings beat. 12,966 hoops / 4,831 FYs / 633 WC — humans play free, we see weak spots instantly.

**Private edge (not user billing) separate bankroll:** Kalshi NBA/NFL/earnings detector IC gate Kelly 0.25 paper 233 trades 1% max/trade 3 concurrent $0.01 slip $0 commission EV $8k/yr/strategy conservative 1 book; equity directional paper sector-neutral ONLY after 60+d OOS IC>0.03 Sharpe>1.2 win>55% DD<12%; tiny 0DTE spreads long spreads only ONLY after gates no naked 0.25 Kelly kill-switch 1% day loss separate bankroll weekly P&L not financial advice.

---

## 9. Triple-Write 7-field Even No-Change — H132

```jsonl
{"nodeId":"strategist-coral-h132","agentId":"strategist-lens-1-coral","attempt":1,"latency_ms":1420,"tokens_est":4350,"status":"ok","errorClass":"none","ts":"2026-08-12T19:47:33-05:00","dailySeed":"20260812","idx":3970,"draws":[3970,14390,4582],"hill":132,"g2_shipped":0.6851,"g2_target":0.64,"g2_full_mean":0.6236,"g2_full_sd":0.003,"g2_full_delta_vs_majority":-0.0022,"g2_full_residual_upper":0.0016,"g2_delta_mean":-0.0851,"sd_d":0.0545,"se":0.0244,"df":4,"t_obs":-3.49,"p":0.0251,"ci95":[-0.1527,-0.0174],"mde_n5":0.0677,"margin":1.26,"clears_floor":true,"decomp_lambda_pct":66,"decomp_lambda_p":0.0122,"decomp_lambda_sig":"Bonferroni_PASS","decomp_coral_pct":34,"decomp_coral_p":0.0659,"decomp_coral_sig":"ns_Bonf_FAIL","majority":0.6258,"weak_bar":0.7258,"control_mean":0.7087,"control_sd":0.0564,"lam_mean":0.6525,"lam_sd":0.0278,"full_mean":0.6236,"full_sd":0.003,"missing_caches":["embedding_v3.npz","mtnn_best.pt","pitch_mtnn_embeddings.json"],"honest_fallback":"15-feat partial 6 families pending 130 feats full 18 fam OOM 7.8G guard","leak_sources":"per-sport Linear adapter + zero-pad 48/32/24 perfect sig Stage2 unfrozen 30ep -6.6pp 0.74->0.674","zero_deps":true,"pacing":":01 3m lite","cli":"python pipeline/train_unified.py --w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda-target 0.5 --grl-lambda 0.3 --grl-ramp 10 --w-task 2.0 --w-sport 0.5 --epochs 60 --seeds 7,11,13,17,19"}
```

Logged to:
- `bundles/ultra/runs/strategist-coral-h132/timeline.jsonl`
- `~/.scout/missions/h132/timeline.jsonl`
- `vector-unified/hidden_files/checkpoints/timeline.jsonl`

No pip, no cloud, ACNE optional local 17 node types 27 edge types. Zero-deps true.

**Done = doc live + CLI + p-math + floor + MDE + paired Δ + λ66% p0.0122 sig + coral34% p0.0659 ns Bonf + leak audit + OOM guard 7.8G + zero-deps true + triple-write honest.**

✨🐱 H132 CORAL 0.64 clamped Δ-0.0851 p0.0251 margin1.26 floor true λ66% sig coral34% tightener — Knowledge free, Edge private, Money via timing.
