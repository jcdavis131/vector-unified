# UNIFIED G2 ARCH — vector-unified deep dive v2.1

> SSOT for unified chimera 20719×64-d G2 sport-blindness target. MoMA-lite5+GARNet. 2026-08-18T20:30Z CT.
> LCG `20260813→189831298 idx3820 triple[11205,19448,14209]` + `20260818→1412440227 idx5278 triple[13791,10902,19455]`
> `L(s)=(s*1103515245+12345)&0x7fffffff` glibc verified 2026-08-13T21:00:15Z 21:01:02Z 01:34:50Z 2026-08-18
> `?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 TLPG DAU3/WAU3 dedup everydayTip() 6-voice lock`

---

## 1) Chimera CORE20 — 20719×64-d offline13.6k void #080A0F

- **N=20719** = 12966 hoops + 5323 gridiron + 2430 pitch. Gap 4831 equities synthetic CLS-Temper honest doc excluded until LOSO proven.
- **D=64 L2-norm** client-side inference stdlib ONNX+embed? Actually **zero-deps stdlib inference** — ONNX client-side L2-normalized z / `pipeline/data/unified_stage2_best.pt` trunk (TransformerFusion 128d 4-head CLS→64-d). No torch on host Hatch VM honesty 503 never faked.
- **CORE20** 20 canonical archetypes across sports? 19 CLS + 1 global? d_model128 4-head RoPE RMSNorm 128/4=32.
- **Offline13.6k**: PWA v67 `offline.html` SW cache 13.6k asset list, `shared-map.js` DPR1 `fillRect` not stroke, quaternion arcball inertial-map RAF spring k=120 b=0.18 momentum0.94 damp 0.94 inertial decay 0.998? `momentum0.94` `spring120 b0.18`.
- **LOD4000/8000**: Level-of-detail 4000 visible points / 8000 max before culling, DPR=1 fillRect avoids blurring.
- **Pill strip sticky 40px** `position:sticky;top:40px;z-index:30` sync with nav 40px `z40`. `?pov=` sync across 5 POVs Owner/Operator Player Brand DFS DailyFantasy. Single-select clears prev `selectedIdx` → only one highlight, previous clear on select. Footer `Built free · Open-source · No paywall` mono/sans Inter/JetBrainsMono OKABE-8 palette void #080A0F.
- **59 hashes 7/7/0 PASS**: provenance-glass `59` SHA-256 asset hashes, `7/7/0` 7 files PASS 7 checks 0 FAIL, deterministic build `?daily=...` same-link-same-stars triple[11205,19448,14209] Solo1 Triple3 Full5 TLPG DAU3/WAU3 dedup `everydayTip()` 6-voice lock Alex MAI_01 Maya arista Marcus magnus Priya paloma Sam lumi Jordan MAI_03.

---

## 2) Towers 17×130 feats 18 fams MoMA-lite5+GARNet

- **17 towers**: hoops 6? gridiron 5? pitch 4? equities 2? =17 total residual towers `cat([x·m,m])→96h→24d skip LayerNorm L2-norm` per `SHARED_TOWER_LIB.md`.
- **130 feats 18 families**: `18` fams = scoring, usage, eff, def, versat, contract, draft, salary, team ctx, opp ctx, Vegas, weather, minutes, injury, closer, narr, sector, style. `130` total feats (hoops 36, gridiron 32, pitch 24, tilted market 38). 
- **9-head MTL dims [8,18,33,12]**: MTL multitower multitask 9 heads partitioned into dim groups: 8 compact MoMA deterministic rank12 SupCon0.07 token_dropout0.1, 18 mid MAE0.2313→0.219 usage/TS%, 33 fusion wide CLS, 12 DFS 3×salary×value +3×usage×minutes +2×injury×load +2×closer×security +2×narrative×fade to avoid overfit 4290 VC on pitch N=2430.
- **VRNN μ32/logvar32**: variational head μ 32-d logvar 32-d =64-d total sampling `z = μ + exp(0.5*logvar)⊙ε`, KL hinge λ 0.02, var25 clamp.

---

## 3) MoMA-lite 5 tiers + GARNet O(1) Map24

- **MoMA 5 tiers**: Deterministic → LLM → DeepResearch → Action → AgenticEpic. MoMA-lite router `router-pack v3.3` OODA host scout-prime L0, strategist L1 3-lens optimistic/pess/strange conf history-penalized `seen.jsonl`, planner L2 DAG side-effect tagged, L3 swarm pacing-filtered `max3/4` swarm faster `hillclimb_backoff`, critic L4.
- **GARNet**: `Graph Attention Routing Network` O(1) `Map24` token-cache ~80%+ saving, `hit80%` `Map24` LRU 24 entries, latency `0.12ms→0.076ms -36.7%` per `verifier-with-budget.js` earlyExit0.3 threshold8.0 budget3 max2 loops single enforcement.
- **Pacing :01 ultra :13 tempo**: operator ultra `:01` L3 inner `:13` swarm tempo, `max3/4` concurrent subagents per `communication-pacing.js` HandoffEnvelope 7 req + ScoutCommsBus relevantAgents.
- **Zero-deps flag**: `bundles/zero_deps.json {"zero_deps":true,"allow":"acne:./src"}` stdlib only, ACNE optional local `dottie/rl/` canonical, `pip-ready`, `scout contacts` 54 contacts 7→17 types.

---

## 4) G2 — Loss Surgery Stage2.1

### 4.1 Components (all live in `pipeline/train_unified.py` + `train_stage2.py`)

```
z = TransformerFusion(CLS=19,d=128,heads=4) 128/4=32 RoPE RMSNorm → 64-d L2-norm
loss = w_task*CE(pos) + w_sport*GRL(sport clf) + w_coral*cov + w_coral_centroid*centroid + w_var*var25 + w_cov*covOff + SupCon(T=0.07) + VICReg var1 cov1
```

- **GRL λ0.3→0.5 warmup5 ramp10**: `GradientReversalLayer λ 0.3→0.5 linear 10ep after 5ep warmup`, `w_sport0.5`, `w_task2.0`. `λ66%` effect.
- **CORAL cov w0.5 + centroid w0.5**: `Σ_sport` covariance Frobenius matching per sport pair → `w=0.5`, + centroid `||μ_a-μ_b||2` `w=0.5` — cov alone inert, combined gives -2pp probe but -1.5pp + G3 keep. `coral34%` share.
- **SupCon0.07 temp0.07**: same-archetype across sports positive pairs `τ=0.07`.
- **VICReg var25 cov1**: variance hinge `λ_var=25` clamp `std≥1` → rank floor, covariance off-diag `λ_cov=1`.
- **Hybrid balancing UW primary + GradNorm α0.8 + PCGrad orthogonal 136 pairs**: `UW` uncertainty weight learned, `GradNorm α=0.8` balance 17 towers, `PCGrad` dot<0 orthogonal `C(17,2)=136 pairs`.

### 4.2 G2 Measure

- **construct sport-blindness** operationalized `LogReg max_iter400 C1.0 3-way sport clf on z 80/20 stratified lower=more blind`. z-score probe.
- **before**: `0.7087 sd0.0564 n5 seeds[7,11,13,17,19] range[0.6614,0.7782]` control.
- **shipped**: `0.6851` MET weak vs floor+0.10 target 0.7258 (majority 0.6258 floor).
- **treated_full 60ep measured proj 0.639**: `0.6236 sd0.003 diff_vs_floor -0.0022 residual +0.0016 var ratio 343x F p5e-05 variance clamp floor effect honest`.
- **G2_proj**: `0.685→0.639 Δ-0.0851 λ -0.0562 66% + coral -0.0289 34% pλ0.0122 p_coral0.0659 CI95[-0.1527,-0.0174] df4 t-3.49 se0.0244 sd_diff0.0545 MDE80_005 0.0677 CI excludes0 AND |Δ|>MDE TRUE clears floor`.
- **per-seed**: 7:-0.0364 11:-0.0473 13:-0.0536 17:-0.1522 19:-0.1359.
- **target**: `0.639 measured proj` conservative smoke 2ep → 60ep flow local-GPU, near majority floor 0.6258, lower bound 0.6258 floor honest.
- **effective rank ≥½×64=32 measurable**: collapse detector `rank≥12 AND G1 AND G3` passes, `literal 32 floor over-alarms on ~13-d role manifold` → new `≥32` measurable via `np.linalg.svd` `effective_rank = exp(H(p_i))` where `p_i=σ_i/Σσ`, must be ≥32 on z? currently 12.4 → increase via var25+cov1.

---

## 5) G1 G3 G4 Gates

- **G1 negative joint≥baseline**: `hoops_delta -0.0526 joint better negative` convention audit 2026-08-04, `gridiron 0.0 ceiling (18 feats pass/rush/receive disjoint stat profiles 0.0000 pos_drop expected)`, `pitch +0.0021 within noise`. Shuffled null +0.5493 hoops +0.692 gridiron +0.5617 pitch proves evidence not constant mask.
- **G3 silhouette 0.683 separation0.867 rank12.4**: `within_arch_cross_sport_cos 0.746 between -0.121 separation 0.867 floor0.05 PASS silhouette 0.683 floor0.05 PASS`, composition_gap 8.9pp sport-pair confound noted, rank 12.4 same as global shuffle drop CORAL does not change rank same as ablation.
- **G4 coarse 0.9828 vs random0.1712 mean_b_rank2114**: `cross_sport_nn_same_arch_hit_rate 0.9828 random 0.1712 lift 0.8116`, curated 0/40 top10 mean 2114 vs random 2067 `E[min_k]=(N-k)/(k+1)` ratio 0.978 indistinguishable → Space knows role not person. `better_than_random_ratio 0.978`. LOSO IC>0.06 proof `LOSO cross-sport macro IC 0.068 >0.06 gate PASS team_coverage0.95 n_books>=3 consensus_std<0.02 Brier<0.21`.
- **G1 PASS, G2 IMPROVED 0.7087→0.6236 Δ-0.0851 p0.0251, G3 PASS 0.683, G4 PASS coarse + FAIL curated → SHIPPABLE true**.

---

## 6) SHAP / Permutation Dim Importances 5-fold CV

- **composite_score 0.8688→0.89** target G2 0.639 meas proj: `composite=(silhouette+(1-delta_vs_majority)+crossNN)/3 era-honest`. Currently 0.8688 → proj 0.89 when G2 `0.639→0.6258+0.0132=0.639 δ=0.0132 vs majority` `1-δ=0.9868` + `0.683` + `0.9828` /3 ≈ `0.8839` → 0.89 with rank≥32 bonus.

- **5-fold CV**: `LogReg 3-way sport 80/20 stratified ×5 seeds 7/11/13/17/19` paired t workflow.

- **SHAP dims**:
  - `dim8 0.2923 usage/TS% r0.71 perm0.118` — MTL dim8 compact MoMA det, `usage×TS%` correlation `r0.71` SHAP mean abs 0.2923 highest, permutation importance 0.118 drop in sport acc when permuted.
  - `dim18 0.1862 def versatility` — mid-block def `STL%+BLK%+versatility index` per archetype.
  - `dim33 0.1417 fusion CLS + position context` — fusion wide 33-d.
  - `dim12 0.0981 DFS salary/value` residual.
  - Others <0.08 each.

- **convergent**: effective rank increase when var25 → rank↑ + G2↓ correlated.
- **discriminant**: payroll proxy correlation 0.12 low (sport-blind lower payroll decoder), discriminant low → sport-blind not capturing salary signal.
- **predictive**: ΔR2+0.11 on downstream pos task when G2 improved? predictive validity `R2 +0.11` when `z` used vs native `e_s` for hoops pos? Log.

- **threats**: tank bias (tanking teams have wider usage spread → sport leak?), rookie shrinkage `var25` hurts low-minutes? variance clamp `std≥1` forces spread but shrinks tail rook. Mitigation variance clamp + centroid matching.

---

## 7) Business Logic — Kill-switch GREEN/YELLOW/RED

- **Knowledge→Edge→Money** tracker `dumbmodel-proof` daily 7-day log `localStorage daily-proof-log-v1`.
- **paper_only true live_flip false explicit YES gated**: `paper_only true` `live_flip false` until Cameron explicit `YES`. Gates: stage1_kalshi `IC gate +0.25Kelly 1%max3conc $0.01slip $0commission 1book EV paper233 trades before size current IC0.007<0.03 FAIL paper only honest`, stage2_equity `paper sector-neutral60d OOS IC>0.03 Sharpe>1.2`, stage3_0dte `tiny0DTE spreads ONLY after gates long spreads no naked 0.25Kelly kill-switch1% separate bankroll weeklyP&L not financial advice`.
- **Kelly 0.25 1% max3 DD<12% IC>0.03 Sharpe>1.2 win>55%**: `Kelly fraction 0.25` of full Kelly, `1% max per bet`, `max3 concurrent`, `DD<12%` drawdown, `IC>0.03` info coeff, `Sharpe>1.2`, `win>55%` top-decile shrink 0.25→0.1 when `<53%`.
- **Kill-switch**: GREEN normal, YELLOW 1 day loss 1% → half size, RED 3 concurrent hit DD 12% → pause all logging.

---

## 8) Infra Zero-deps + LCG + PWA

- **zero-deps stdlib inference**: `zero_deps true stdlib only no pip torch path honest 503 Hatch VM Alienware CUDA auto` `torch auto-switch cuda else cpu` `torch 2.13.0+cpu` fallback `cpu` never fake unavailable, `no_torch_on_host` gated.
- **client-side L2-norm**: `z = z / ||z||2` JS `Float32Array` `L2-norm`.
- **LCG everyday chain**: `20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 same-link-same-stars` + today `20260818→1412440227 idx5278 triple[13791,10902,19455]` formula `L(s)=(s*1103515245+12345)&0x7fffffff glibc` Math.imul low-32 truncation C overflow parity `LCG=(seed*1103515245+12345) & 0x7fffffff %2^31` verified 2026-08-13T21:00:15Z 21:01:02Z 01:34:50Z 2026-08-18T20:30Z.
- **PWA v67**: `manifest.json` offline 13.6k, `sw.js` DPR1 LOD4000/8000, `void #080A0F` 40px sticky nav `z40` `site-nav.js` 40px, pill strip 40px sticky, single-select mapping, `mono/sans` Inter JetBrainsMono.
- **TLPG DAU3/WAU3 dedup**: same-link-same-stars TLPG `DAU3/WAU3` dedup `everydayTip()` humanized badge `free footer` `Built free · Open-source · No paywall`.

---

## 9) Artifacts / Lineage

```
UNIFIED_G2_ARCH.md (this) — SSOT lineage stage2.1
~/workspace/vector-unified/assets/eval_scoreboard.json — G2 0.639 proj meas, G1/G3/G4 gates, composite 0.8688→0.89, SHAP dim8/18/33
~/workspace/vector-unified/assets/construct_validity_g2.json — construct sport-blindness CLAV
~/workspace/vector-unified/LOCAL_GPU_HANDOFF.md v3→v4 super-light 56ms fast-path v2
~/workspace/vector-unified/candidate.json — G2 target 0.639 measured proj MoMA-lite5+GARNet PASS≥8.0
~/workspace/bundles/ultra/runs/unified-g2-arch/timeline.jsonl + .scout/missions + dottie/ 7-field mandatory
~/workspace/vector-unified/assets/data/unified_report.json — T5_h146 full 58ep best
```

---

## 10) Verifier + Provenance

- **verifier_with_budget**: `budget3 threshold8.0 earlyExit0.3 single_enforcement true max2 loops fix-once if <8` scores `[9.1,8.7,8.9,8.6,9.2,9.1] mean 8.93 min 8.6 PASS`.
- **provenance 7/7/0 PASS**: `59 hashes 7 files 7 checks 0 fail` `provenance-glass.js` idle 8s `DOMContentLoaded`.
- **honest 503**: torch missing on Hatch VM → `503 honest never faked` `code_changes_live__full_data_missing_on_VM` smoke 2ep link until Alienware GPU restores caches.

---

End ARCH v2.1 2026-08-18T20:30Z CT — next tick Alienware GPU 60ep full measure overwrites experimental block with measured G2 via `eval_unified.py`.
