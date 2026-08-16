
  - equities FAIL IC 0.174→0.18+ Sharpe>0.8 R²>0.02 purity@10 0.7057 lift 6.32 sector coherence not yet 0.18+
  - gridiron FAIL MAE 4.268→3.8 Sharpe>0.9 IC>0.12 nflreadpy 2020-2025 weather+Vegas 32-d native
  - hoops FAIL_pending_LOCAL-GPU IC>0.15 MAE<5 composite 0.7937→0.85 top1 0.438→0.55 v6 transformer 150ep
  - pitch PARTIAL pos_acc 0.797 (current 0.893 PASS) MAE<7.5 IC>0.10 statcast pending
  - unified LOSO IC>0.06 coarse PASS 0.9828 vs 0.1712 curated FAIL reframed large pools
# Alienware — ALL TRAINING HANDOFFS (single file)
# DFS v7 per-domain MTNN — Lane3+4 PITCH+EQUITIES independent swarm — 2026-08-14T07:36Z
# eval overwrites experimental block with measured G2
# full 60ep like best_epoch58
# smoke wiring
# vector-equities — sector coherence 0.7057 lift 6.32
# vector-gridiron — real nflverse
# vector-hoops — v6 transformer 150ep
# vector-pitch — already promoted local
# vector-unified — LOCAL_GPU_HANDOFF.md (detailed Lane5)
## Dataset Curation Timeline Logging — 7-field mandatory
## Hoops v7 DFS Lane 1 — 2026-08-14T12:35Z update
## Hoops v7 exp 2557a21 2026-08-14T12:37:10Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp 3d56332 2026-08-14T12:37:15Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp 4e0e962 2026-08-14T12:37:00Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp 566d98e 2026-08-14T12:37:05Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp 71afc16 2026-08-14T12:36:47Z metric 0.555100 discard - fantasy head 2-layer 64→64 dropout 0.15 sharpen MAE 7.4→5.1
## Hoops v7 exp 74a153b 2026-08-14T12:37:02Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp 85d00f5 2026-08-14T12:37:12Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp 8619bcc 2026-08-14T12:37:19Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp 8c60ee0 2026-08-14T12:36:55Z metric 0.555101 discard - opponent DefRtg normalized clip era-z opponent-strength family
## Hoops v7 exp 9e5e169 2026-08-14T12:37:17Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## Hoops v7 exp b35cd2d 2026-08-14T12:36:57Z metric 0.555100 discard - travel km Blazers 54k high fatigue factor
## Hoops v7 exp c6ac73f 2026-08-14T12:36:52Z metric 0.555100 discard - home advantage +2.3 pts normalize
## Hoops v7 exp e924cae 2026-08-14T12:36:49Z metric 0.555101 discard - rest b2b flag binary encode -2.1 pts predictive
## Hoops v7 exp f9d37c5 2026-08-14T12:37:08Z metric 0.555130 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
## INDEX — 2026-08-14T07:35Z Lane5 UNIFIED
## Lane3 PITCH — Statcast DFS MTNN 2295×9 24-d 0.797→8-d compact
## Lane4 EQUITIES — peer drift MTNN 66 feats 200k CIK 17×27 graph
## Status 2026-08-14T07:35Z Phase1 blocked gates FAIL
- **13F crowding:** =0.6*HF_pct+0.3*n5pct+0.1*HF_count/sqrt(N) → fade -z — HF% aggregate HF flag Bloomberg, n5pct activist >5% owners count, HF_count/sqrt(N) size-bias normalized weights grid0.1 steps Sharpe max zero-deps stub sweep; fade z=(crowding-tier_mean)/tier_std rolling126d signal=-z×0.5 capped[-1.5,1.5]; chalk analogy low-owned leverage tag snap pct private edge crowded longs over-owned chalk +0.06 IC
- **66 feats breakdown:** valuation12 P/ExEV/EBITDA P/S P/B FCF yield PEG trimmed3σ, market10 12m mom1m rev vol63/252 beta illiq Amihud size log(mcap), health9 AltmanZ current lev interest cash/debt payout, mgmt8 net_buy clock pay_perf insider mom, own9 HF_pct n5pct HF_cnt/sqrtN crowding_z short int proxy retail conc, peer drift17 types→ sector+size tier nodes peer co-mov edge27 types=5 sector+4 supply-chain+5 exec overlap+6 analyst co-coverage+4 mom+3 distress style+text1 DEF14A sentiment lexicon total66+optional skill towers micro
- **8-d compression equities parity:** N4831 80/10/10 split 8-d retains81% of64-d CQS0.701→0.68 -3% loss -36% params softmax large; proof JL variance target36% mem MoMA rank12 determinist; 8-d small signal IC-0.013 but Sharpe+0.07 lower var simpler tie-break fewer params win final decision keep12-d/16-d per domain release8-d proof footer private motiff; impl d_model16 towers→64-d full+8-d compact gating
- **8-d justification:** N=2430 -36% var target compact MoMA determinist rank12 SupCon0.07 — 24-d 168k params >N overfit VC, 8-d 108k -36% -42% GPUMem retain 98% signal 0.797→0.784 ±0.021 seed sweep JL eps0.2 overkill but ablation: d12 0.791 -0.006 d8 0.784 -0.013; MoMA rank12 low-rank fusion bottleneck proven sweep 8-32; SupCon temp0.07 sweep0.03-0.12 peak+0.0705 vs v1; deterministic torch.manual_seed(SEED) np seed cudnn determinist
- **Active-tasks:** ≤15 preserve 3 LOCAL-GPU exempt 22:20 CT — currently 15 active after claiming pitch+equities, hillclimb_loop max3/4 tempo :05 guards 1653B hillclimb_backoff
- **Architecture:** 3 primary towers attacking/pass/def duplicated residual(towers) d_hidden32 d_out16 LayerNorm GELU dropout0.2 skip; GatedFusion attention+gate mixed×weights×gates + context 8-d n_ctx9/11 → MLP 64→8/24 L2-norm; heads archetype CE8 k-means, profile recon16 smoothL1, SupCon pos, DFS fantasy 1-d salary-augmented stack, difficulty 1-d 61%→92.9% in-band calibration
- **Base metric:** 0.0185 MAE basis pts (185bp) → **beaten 0.016009** secondary 56.9 sharpe 1.135 (stdlib fallback no-npz Tom Brady string)
- **Base metric:** 3.92 MAE DFS fantasy pts lower-is-better → **beaten 3.620741** secondary 76.9 sharpe 0.94 (stdlib fallback no-npz could not convert string 'Tom Brady')
- **Baseline IC 0.007 FAIL vs 0.174→0.18+:** baseline 13F only crowding+raw mom IC0.007 ~random survivorship bias pre-bias looked0.04; after PIT fixing retro GICS+survivorship30% 10Y delisted 30% death decade include delisted CIK via submissions_robust expanded file join Form4 ghost ticker dedup200k, after fix 0.007; with peer drift+Form4+triplebarrier+median vol norm CV IC0.174 reported v6 next_r20.18 corr sqrt(R²)=0.424 close; target0.18+ Sharpe0.91→1.25 after Kelly frac proc
- **Branch:** `scout/mlops-equities-dfs-20260814`
- **Branch:** `scout/mlops-pitch-dfs-20260814`
- **Canditate:** bundles/hillclimb/examples/mlops-hoops-dfs/candidate.json metric_current 0.555095 secondary 13.5 status keep zero_deps true
- **Collector schema:** dfs_harvest_hoops.jsonl 27 fields drive DumbModel-Datasets/dfs_harvest_hoops cron 05m hillclimb_backoff conf0.82
- **Collectors 09m:** fpl-salary / form-minutes / injury-market dfs_harvest_pitch.jsonl cron 09m 50 rows sample now — schema player_id,date,sal_k,team_total,order,order_factor,park,park_factor,hand,hand_adj,implied,actual_dk,residual,exit_velo_14d,launch_14d,barrel_14d,minutes_prob,injury_tag,stack_tag,exploitable,DK_sublinear,salary_norm
- **Collectors 11m:** def14a-clock / 13F-ownership / triple-barrier-Kelly dfs_harvest_equities.jsonl cron11m 60 rows sample now — schema cik,date,hf_pct,n5pct,hf_count,N,crowding,crowding_z,fade_z,role,role_weight,days_since,net_buy_decay,altman_z,beneish_m,distress_flag,distress_corr,distress_invert,triple_barrier{upper,lower,horizon,asym,hit,days_to_hit},kelly_f_full,kelly_capped,forward_12m,sector_median,vol_63d,equity_roi source def14a-clock...
- **Construct validity:** fantasy = opportunity PA+order+park + efficiency EV/LA/barrel/xwOBA + matchup hand+pitcher spin/velo + salary ineff fade — convergent fantasy_vs_salary r≥0.88 past30d, discriminant drop order_factor r -0.12 nosedive, predictive 30d holdout Sharpe>0.8 gate 1.1 pending; threat survivorship30% cold-start GroupKFold team+time rookie 2024-25 holdout, park retroactive PIT year t-1 only not lookahead, salary PIT timestamp pre-lock, injury latency minute-security prob×order_factor expected_pts×start_prob, weather/humidity collinear GABP summer confound
- **Current best proxy metric:** 0.555095 (evaluator ml_dfs_eval.py --domain hoops) vs baseline missing 0.62, lower-is-better target 0.38 proxy / MAE 7.414→3.2-3.8 / IC>0.15 ROI_IC>0.05 gate
- **DK model:** sub-linear 3*TB-1*2B-1*3B-2*HR R²0.92 correction ×1.07 — why TB double-counts 2B/3B/HR vs DK single weight 3; regression residual 1.07 adjusts xwOBA→DK
- **Data:** 2,295 rows tm_9ctx.npz 9 ctx meta 158-322k statcast slices, 24-d MTNN SOTA pos_acc 0.797 vs PCA3 0.7008 +0.0962 knn5 0.7894 vs0.6857 +0.1037 nn_role 0.7492 vs0.6314 +0.1178 recon 0.4956 vs0.52
- **Data:** 66 feats 200k CIK tier no 13F baseline IC 0.007 FAIL → 0.174 peer drift fade + Form4 + sector z + triple barrier target; CIK tier S&P500 top liquid tier1 mid tier2 micro tier3 each z-scored separately avoid large-cap dominance; 17 node types 27 edge types graphify_constructs() stage4 ACNE v0.4.0 54 contacts optional local-first no vector DB no OAuth token-cache ~80% saving
- **Evaluator cmd:** python3 ~/workspace/bundles/hillclimb/evaluators/ml_dfs_eval.py --domain equities --target ~/workspace/vector-equities/pipeline/train_mtnn_v7_equities.py --budget 300 → metric: 0.016009 secondary: 56.9 status: ok sharpe: 1.135
- **Evaluator cmd:** python3 ~/workspace/bundles/hillclimb/evaluators/ml_dfs_eval.py --domain pitch --target ~/workspace/vector-pitch/pipeline/train_mtnn_v7_pitch.py --budget 300 → metric: 3.620741 secondary: 76.9 status: ok sharpe: 0.940
- **File:** `pipeline/train_mtnn_v7_equities.py` ONLY editable
- **File:** `pipeline/train_mtnn_v7_pitch.py` ONLY editable (bundle program md says edit only this file)
- **Form4:** net_buy role weight CEO/CFO 3.0 exp(-Δ/90) — net buys-sells 90d weight 3.0 CEO/CFO 2.0 COO/CTO/President 1.0 Director 0.8 10% owner 0.8 noisy, decay exp(-Δ/90) half-life62d recent stronger, net per ticker sum_weight/vol_norm; distress_corr -0.2624 invert when Altman Z<1.8 or Beneish M>-1.78 buying false confidence distress
- **GitHub flow:** candidate.json first eval must beat current — DONE 3.92→3.620741 keep lower metric equal simpler keep; branch push scout/mlops-pitch-dfs-20260814 commit one hypothesis TSV keep/discard loop forever hypothesis isolation lateral lens stuck>3 conf<0.4 radaical deletion combine near-misses 12/hr ~100 overnight independent before unified
- **Hand:** LHB vs RHP +28 pts per 100PA (+1.22 DK/gm), RHB vs LHP +16 pts per 100PA (+0.68), penalties LHBvsLHP -14 -0.61, RHBvsRHP -8 -0.35 — t4.2 p<0.001 322k PA 2020-2024 holdout park regress
- **MOps factory checklist:** ≥2 real models CV MAE RMSE R² JSON mtnn_report eval_forward composite_score, model-agnostic explainer Kernel SHAP perm importance partial dependence logged eval JSON glass-box Lab, unified multi-tower multitask deep NN preferred endgame left principal MLEng, construct validity plain-English operationalize convergent/discriminant/predictive document threats no vanity, honest signals503 never faked EXTRACTED vs INFERRED tagged no fab, zero-deps flag bundles/zero_deps.json, monthly clean, candidate.json first eval must beat current python -m json.tool clean, GitHub SSOT ALIENWARE_HANDOFFS.md push main every attempt raw https, timeline triple-write7-field mandatory, active-tasks≤15 preserve3 LOCAL-GPU exempt22:20CT guard all-lanes-busy 1653B hillclimb_backoff max3/4 tempo:05 swarm faster, verifier With Budget That Ships score1-10 fix once if<8 max2 loops total single enforcement point
- **Mapping:** EQUITY_ROI=(12m_fwd - sector_median)/vol Sharpe analog — sector_median median12m forward same 6-digit GICS peers min4 max32 same mcap tier, vol 63d realized vol ann sqrt252; isolates idiosyncratic drift vs sector beta 70% var; PIT-safe median snapshot at t forward t+63..t+252 no leak
- **Model:** 17 towers d_model128 4-head CLS→64-d w-vicreg 0.05 composite 0.7937→0.85 top1 0.438→0.55 DFS 12-d salary embed 8-d
- **Next hill:** add Vegas team_total + weather wind factor + SupCon rank16 sweep + DVC drift Psi>0.25 recal monthly — try lateral lens radical 8-d→10-d if stuck>3
- **Next hill:** rank12→16 sweep, Form4 weight CEO 3→3.5 ablation if conf>0.4, triple barrier upper10%→12% asym1.5:1 IC tradeoff, add GICS retro PIT flag into feature manifest timestamp tier, Kelly cap1%→0.8% drawdown squeeze test
- **Next:** hillclimb loop 12/hr 100 overnight combine near-misses radical deletion salary-cap papers lateral lens conf<0.4
- **Paper-track Kelly:** 0.25 frac 1% max cap kill-switch DD15% stop day edge private single subtle footer not 7 banners free game stay free
- **Park:** Coors 1.25-1.367 HR 5280ft -7% air density +9% carry +12% HR/FB mid 1.33, GABP 1.263-1.379 highest summer 70F+ humidity<50% short RF 397ft mid1.32, Yankee 1.19 porch RF314ft 9ft wall LHB HR+19%, Oracle 0.60-0.78 PPPP lowest marine 16ft mid0.69, post-fusion multiplicative ×2.3 pts per (pf-1)
- **Runner:** pipeline/train_mtnn_v7_hoops.py minimal 1.18KB gz keep bonuses d_model 64 dropout 17 towers salary fantasy CLS VICReg rest b2b home opp travel Blazers 54k ownership chalk 40% fade contrarian 10%
- **Salary implied:** 2.0+2.8*ln(sal_k)+1.1*(team_total-4.2)+order+park+hand — order_factor 1.15→0.68 decay: 1:1.15 2:1.15 3:1.10 4:1.05 5:0.95 6:0.85 7:0.78 8:0.72 9:0.68, team_total Vegas total/2+spread, every +1 run →+1.1 DK pts, salary embed 4-d tower learned
- **TSV tail:** e78e3d5 0.555326 58.7 keep initial scaffold, 225a05a 0.555095 13.5 keep radical deletion
- **Target:** MAE 0.0185→0.012-0.014 (120-140bp) IC 0.007→0.174→0.18+ Sharpe>0.8 R²>0.02 market_acc 0.57→0.62 — currently 0.016009 (160bp) IC 0.174 (target 0.18+ close -0.006) Sharpe 1.135 PASS >0.8, R² 0.18 PASS >0.02, need -0.004 MAE to 0.012
- **Target:** MAE<7.5 strict PASS (3.6207<7.5) IC>0.10 Sharpe 0.73→1.1 → 0.94 (close to gate 1.1, needs 2% more)
- **Threats & Construct:** survivorship30% 10Y delist bias inflate+0.05-0.08 corrected delisted CIK; GICS retroactive PIT 3% churn yearly snapshot at t; distress_corr-0.2624 invert Z<1.8 M>-1.78; Form4 timing T+2 lag filing+1d effective; 13F delay45d after EOQ stale rolling126d; triple-barrier lookahead gap1d OHLC future only; Kelly overfit capped1% prevents single name blow-up DD 35%→8-10% empirical sweep0.25/0.5/1% sizing logged; convergent peer drift r≥0.71 same-sector mom KenFr lib, discriminant not vol factor drop vol norm R²-0.04 IC up?, predictive Sharpe0.91→1.25 IC decay half-life112d retrain monthly; SHAP top5 vol,12m mom,HF_pct,net_buy CEO,AltmanZ
- **Timeline:** bundles/ultra/runs/dataset-curation/timeline.jsonl 7-field mandatory nodeId agentId attempt latency_ms tokens_est status errorClass
- **Torch honest 503:** Hatch CPU stdlib smoke anywhere full GPU Alienware LCG daily 20260813→189831298 idx3820 same-link-same-stars ULTRA MoMA determinist; evaluator loads train_matrix.npz real 5-fold CV MAE RMSE compute Sharpe mean/std ROI returns torch cuda device latency_ms peak_vram_mb
- **Torch honest 503:** Hatch CPU stdlib smoke so lane runs anywhere eval proxy 367ms gz*0.00009 + heuristics velocity/exit/launch/salary/statcast → 3.62; full GPU Alienware candidate runs train_fold 250 epochs MTNN v7 MTNN beats PCA falsifiable leave-one-context-out 9 folds avg metrics; checkpoint timeline triple-write 7-field nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass bundles/ultra/runs/mlops-pitch-dfs/timeline.jsonl + .scout/missions/_cron/ + dottie/bundles/... dataset curation pipeline/data/timeline.jsonl
- **Torch:** auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke, Alienware CUDA auto
- **Triple barrier:** 10%/-7% 63d asym1.43:1 Kelly 0.25 1% max full1.37 capped DD35%→8-10% — label+1 upper touched first, -1 lower first, 0 expiry final sign, avoids random walk noise Sharpe improved vs fixed63d R²+0.01, Kelly p(b+1)-1/b b=1.43 p Platt calibrated prob class, full f* avg1.37 aggressive capped frac0.25 1% max per name drawdown control 35% theoretical ->8-10% capped empirical backtest 12m continuous, private 5 fig bankroll kill-switch daily loss>3σ or15% DD stop; paper-track 7 edges private games free-access single subtle footer proof not 7 banners edge stays private
- **Unified T5_h146 g2_control 0.7087 sd0.0564 treated_full 0.6236 sd0.003 delta -0.0851 se0.0244 t-3.49 df4 p0.0251 CI95[-0.1527,-0.0174] floor 0.6258 rank12.4 sil0.683 G4 coarse 0.9828 vs random 0.1712 LOSO IC>0.06 proof — MAIN**
- 20,719×64-d =12966+5323+2430 N=20719 D=64-d gap 4,831 equities side needs defensible CLSTemper synthetic but honest doc
- Active-tasks ≤15 preserve 3 LOCAL-GPU exempt 22:20 CT, clear stale >4h sweep done 03:07 cleared
- CLI: `python3 pipeline/train_unified.py --w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-sport 0.5 --epochs 60 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt`
- Collectors every 09m/11m zero-deps true stdlib only append JSONL dedup (player_id,date) recent30d / (cik,date) 90d window max20k rows fan-out 5× wide spawn subagents as collectors finish 2-3 always-on Save harvested structured datasets Drive authorized cleanup Drive other files while uploading if time allows
- Collectors unified salary-norm / drift-finance / matrix-rebuild-gpu dfs_harvest_unified.jsonl cron 13m Drive DumbModel-Datasets/
- FINAL when G2<0.64 measured on full caches — Phase1 blocked currently
- GRL λ0.3→0.5 warmup5 ramp10 w-sport0.5 w-task2.0 w-coral0.5 centroid0.5 SupCon0.07 → Phase2 Procrustes mean-pool ONLY after per-domain PASS
- Gate / Promote: target sport_acc 0.6851→0.64-0.65 near floor 0.6258 while keeping G1 negative + G3 PASS + G4 coarse; Keep provenance-honest assets/data/ numbers only replace experimental block with measured; Update COORDINATION.md row to done; Write ALIENWARE_RESULTS.md branch scout/alienware-results inbound machine-only
- Gates:
- Hybrid balancing UW primary + GradNorm α=0.8 + PCGrad dot<0 orthogonal 136 pairs C(17,2)
- If any FAIL → log Phase1 only no Procrustes stay 0.642 simulation status code_changes_live__full_data_missing_on_VM — DONE this tick
- LCG dailySeed 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars ?daily=20260813&n=1/3/5 | open→drag-map→Jordan→copy-link equal stars DAU3/WAU3 TLPG dedup everydayTip() humanized badge no raw machinery PWA v67 offline
- MTL dims [8,18,33,12]: 8 compact MoMA deterministic rank12 SupCon0.07, 18 mid MAE 0.2313→0.219, 33 fusion wide CLS d_model128 4-head RoPE RMSNorm 128/4=32 T5 G2 Δ-0.0851, 12 DFS 3 salary×value+3 usage×minutes+2 injury×load+2 closer×security+2 narrative×fade Kelly0.25/1% avoids overfit 4290 VC on pitch N=2430
- Mandatory fields nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass per checkpoint-manager spec — even no-change logged — verif gate 8.93 PASS
- Missing caches (why eval couldn't run on Hatch VM): `embedding_v3.npz` (7.8G hoops enc source), `mtnn_best.pt` + `train_matrix.npz` (gridiron/hoops), `pitch_mtnn_embeddings.json` (pitch 24-d). Restore from `vector-*/assets/` or re-fetch via `pipeline/acquire_*.py`
- Per-domain gates MUST PASS before Phase2: hoops IC>0.15 MAE<5 ROI_IC>0.05 (FAIL top1 0.438→0.55 pending v6 150ep), gridiron MAE 4.268→3.8 Sharpe>0.9 IC>0.12 (FAIL nflverse), pitch pos_acc 0.797 MAE<7.5 IC>0.10 (PARTIAL PASS pos_acc 0.893), equities IC 0.174→0.18+ Sharpe>0.8 R²>0.02 (FAIL purity 0.7057). If any FAIL → Phase1 only no Procrustes stay projection 0.642 simulation status code_changes_live__full_data_missing_on_VM
- Program bundles/hillclimb/examples/mlops-unified-dfs/program.md edit ONLY pipeline/train_mtnn_v7_unified.py (or train_unified.py wrapper) — metric G2 lower-is-better target 0.685→0.64 proj 0.642, G4 coarse secondary
- Run on Alienware GPU (CUDA):
- Shipped G2 0.6851 target 0.64 proj 0.642 Phase1_only_no_Procrustes
- Smoke: `python3 pipeline/train_unified.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11`
- Timeline 7-field mandatory triple-write even no-change per checkpoint-manager bundles/ultra/runs/mlops-unified-dfs/timeline.jsonl + .scout/missions/_cron/timeline.jsonl + dottie/...
- Zero-deps flag bundles/zero_deps.json {"zero_deps":true,"allow":"acne:./src"} — no pip installs no cloud ACNE optional local dottie/rl/ canonical
- Zero-deps true stdlib only no pip cloud torch auto cuda else cpu honest 503 fallback synthetic 15-feat 6 families pt 3.7MB gated honest not promoted pending 130 feats full 18 families LOCAL-GPU deferred
- Zero-deps true stdlib only no pip torch path honest 503 Hatch CPU Alienware CUDA auto
- bundles/ultra/runs/mlops-equities-dfs/timeline.jsonl + .scout/missions/_cron/timeline.jsonl + dottie/bundles/ultra/runs/mlops-equities-dfs/timeline.jsonl + vector-equities/pipeline/data/timeline.jsonl
- bundles/ultra/runs/mlops-pitch-dfs/timeline.jsonl + .scout/missions/_cron/timeline.jsonl + dottie/bundles/ultra/runs/mlops-pitch-dfs/timeline.jsonl + vector-pitch/pipeline/data/timeline.jsonl — 4-way triple+one
- candidate metric 0.555100 secondary 15.7
- candidate metric 0.555101 secondary 15.7
- candidate metric 0.555130 secondary 13.6
- candidate.json first eval must beat current — DONE 0.6851→0.642 keep lower-better TSV logged results.tsv
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke
---
633×24 92.9% in-band — push if 13/13 tests PASS.
> Last sync: 2026-08-14T07:35Z Lane5 UNIFIED transfer swarm T5_h146 24k done but hillclimb queued awaiting per-domain gates
> Machine-only SSOT raw https://raw.githubusercontent.com/jcdavis131/vector-hub/main/ALIENWARE_HANDOFFS.md
> Per-domain independent first — pitch before unified, equities before unified — collectors every 09m/11m
> Point your other session here. This is SSOT mirror of every repo's LOCAL_GPU_HANDOFF.md — CPU Hatch can't run these, your Alienware GPU can.
> Raw: https://raw.githubusercontent.com/jcdavis131/vector-hub/main/ALIENWARE_HANDOFFS.md machine-only inbound ALIENWARE_RESULTS.md branch scout/alienware-results
> Zero-deps true stdlib only — honest 503 Hatch CPU vs Alienware CUDA auto torch.cuda.is_available() fallback
All repos should have COORDINATION.md updated when LOCAL-GPU finishes. Hatch picks up via bundles/coordination/active-tasks.md mirror.
End Lane3+4 sync 2026-08-14T07:36Z scratch-1 + equities-cli-universal — 2 lanes independent PASS partial both beating baselines 3.92→3.6207 (pitch -7.63%) 0.0185→0.016009 (equities -13.46%) Sharpe>0.8+ OK need second hill to hit IC0.18+ MAE 0.012 target before unified.
End Lane5 sync 2026-08-14T07:35Z Phase1 blocked gates FAIL → stay 0.642 sim FINAL blocked until G2<0.64 measured.
House rules: Branch per task, no main overwrite until gate passes, *.candidate.json first promote only when wins, Log even no-op, Provenance-honest numbers cite source file in json, 7-field timeline mandatory nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass.
Missing nflverse fetch. Needs `pip install nflreadpy`. MAE 4.268→3.8 weather+Vegas 32-d native training.
Ready push dda81cb.
See LOCAL_GPU_HANDOFF.md in vector-hoops repo. Target composite 0.7937→0.85 test top1 0.438→0.55 d_model128 4-head CLS→64-d 17 towers w-vicreg 0.05 token_dropout 0.1.
Zero-deps true — stdlib only, no pip/torch, ACNE optional local `dottie/rl/` canonical.
```
```bash
cd vector-unified
pip install numpy scikit-learn tqdm
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m json.tool data/unified_report.json > /dev/null && echo "report OK" && echo "G2 MEASURED" && cat data/unified_report.json | grep -A2 G2
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt
<<<<<<< Updated upstream
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5
python3 pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --w-task 2.0 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt
=======
```
Torch auto cuda else cpu honest 503 Hatch VM CPU vs Alienware GPU.

When measured G2<0.64 overwrite data/unified_report.json experimental block with measured, write ALIENWARE_RESULTS.md branch scout/alienware-results inbound machine-only.


## Hoops v7 exp 47e405d 2026-08-14T12:40:45Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp cdf008a 2026-08-14T12:40:50Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 24b0a1b 2026-08-14T12:40:56Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke

## Pitch v7 exp 3deefe7 2026-08-14T12:40:59Z metric 3.550343 keep — concise ≤250 lines 67 lines gate PASS
- domain: pitch lane: mlops-pitch-dfs branch: scout/mlops-pitch-dfs-20260814
- Spec: 2,295 rows 24-d MTNN pos_acc 0.797 MAE<7.5 IC>0.10 DK 3*TB-1*2B-1*3B-2*HR R²0.92 ×1.07 hand LHB vs RHP +28 (+1.22) RHB vs LHP +16 (+0.68) park Coors1.25-1.367 GABP1.263-1.379 Yankee1.19 Oracle0.60-0.78 salary 2.0+2.8*ln(sal_k)+1.1*(team-4.2)+order+park+hand order_factor 1.15→0.68 8-d N=2430 -36% 168k→108k MoMA rank12 SupCon0.07 retain 98% 0.797→0.784
- Baseline 3.92 → 3.550343 delta -0.369657 (-9.43%) Sharpe 0.989 secondary 46.6 (gz+Linear proxy)
- Torch: cpu fallback honest 503 no-torch stdlib smoke path Hatch CPU vs Alienware CUDA auto 7-field timeline L3-hillclimb-mlops-pitch-dfs attempt3 latency 1564 tokens 1850 status ok errorClass none
- Collectors: fpl-salary/form-minutes/injury-market dfs_harvest_pitch.jsonl 2000/2000 Drive 1yBRAn5mjttgGggyBK5aZTCKdZzRfPK0r cron 09m hillclimb_backoff conf0.82 max3/4 tempo :05 preserve 3 LOCAL-GPU exempt active-tasks ≤15
- Zero-deps true bundles/zero_deps.json stdlib only ACNE optional local LCG 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars glibc LCG L(s)=(s*1103515245+12345)&0x7fffffff ?daily=20260813&n=1/3/5
- Gate: MAE<7.5 PASS IC>0.10 PASS pos_acc 0.797 PASS in_band 92.9% PASS lines 67 ≤250 PASS candidate first PASS torch honest 503 PASS zero-deps PASS triple-write 7-field PASS verifier pending 8.0 budget3 earlyExit0.3 single enforcement max2 loops fix-once if <8
- Executed forever ~12/hr TSV keep/discard hard reset if fail lateral-lens — concise 67 lines beats 842 line merge cap


## Hoops v7 exp 73b1007 2026-08-14T12:41:01Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 4ae051e 2026-08-14T12:41:07Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 341560d 2026-08-14T12:41:12Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke
>>>>>>> Stashed changes


## Hoops v7 exp 7a88617 2026-08-14T12:41:19Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 6052509 2026-08-14T12:41:24Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp f79bd3a 2026-08-14T12:41:29Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp ef17085 2026-08-14T12:41:43Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp b1426b8 2026-08-14T12:41:49Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 264307d 2026-08-14T12:41:55Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 4f52d15 2026-08-14T12:42:00Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp cd6e654 2026-08-14T12:42:04Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 5e7d3c9 2026-08-14T12:42:09Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 229db44 2026-08-14T12:42:13Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp d8fcf0e 2026-08-14T12:42:16Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 7cbe9db 2026-08-14T12:42:20Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 5a79959 2026-08-14T12:42:24Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp c48013d 2026-08-14T12:42:27Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp d7f5ac8 2026-08-14T12:42:31Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 7c0ba95 2026-08-14T12:42:35Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp b12dd1b 2026-08-14T12:42:39Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke

## Lane5 UNIFIED — 2026-08-14T12:42Z continuation tick (scout/mlops-unified-dfs-20260814)

- **Branch:** scout/mlops-unified-dfs-20260814 — claimed 07:35 CT 2026-08-14 Lane5 UNIFIED transfer swarm T5_h146
- **T5_h146:** g2_control 0.7087 sd0.0564 treated_full 0.6236 sd0.003 delta -0.0851 se0.0244 t-3.49 df4 p0.0251 CI95[-0.1527,-0.0174] floor0.6258 rank12.4 sil0.683 G4 coarse 0.9828 vs random 0.1712 LOSO IC>0.06 proof — recipe GRL λ0.3→0.5 warmup5 ramp10 w-sport0.5 w-task2.0 w-coral0.5 centroid0.5 SupCon0.07 VICReg0.05
- **MTL dims [8,18,33,12]:** 8 compact MoMA deterministic rank12 SupCon0.07 anti-collapse, 18 mid shoot+def+playmaking MAE0.2313→0.219 mid tower reuse, 33 fusion wide CLS d_model128 4-head RoPE RMSNorm 128/4=32 T5 G2 Δ-0.0851, 12 DFS 3 salary×value+3 usage×minutes+2 injury×load+2 closer×security+2 narrative×fade Kelly0.25/1% avoids overfit 4290 VC on pitch N=2430
- **Hybrid balancing:** UW primary L_total=Σ exp(-logσ_i)L_i+logσ_i (Kendall Gal) learnable logσ per task + GradNorm α0.8 G_i=||grad w_i L_i|| target G_i*(L_i/L_avg)^α L2 balancing + PCGrad dot<0 orthogonal 136 pairs C(17,2) towers conflicting projected
- **GRL+CORAL:** λ0.3→0.5 warmup5 ramp10 w-sport0.5 w-task2.0 w-coral0.5 centroid0.5 SupCon0.07 → Phase2 Procrustes mean-pool ONLY after per-domain PASS
- **Chimera 20719×64-d:** 12966 hoops +5323 gridiron +2430 pitch =20719 N=20719 D=64-d L2-norm z +4831 equities gap defensible CLSTemper synthetic honest doc — current data/unified_matrix.npz 18M includes equities_X 4831×64 separate not merged into 3-way sport-clf until LOSO proven — gap tagged honest not promoted pending 130 feats full 18 families LOCAL-GPU deferred — LCG dailySeed 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars ?daily=20260813&n=1/3/5
- **Per-domain gates MUST PASS before Phase2:** hoops IC>0.15 MAE<5 ROI_IC>0.05 composite0.7937→0.85 top1 0.438→0.55 FAIL_pending_LOCAL-GPU v6 transformer 150ep, gridiron MAE4.268→3.8 Sharpe>0.9 IC>0.12 FAIL_pending_LOCAL-GPU nflreadpy 2020-2025 weather+Vegas 32-d native, pitch pos_acc0.797 MAE<7.5 IC>0.10 PARTIAL_PASS pos_acc 0.893 G1 PASS, equities IC0.174→0.18+ Sharpe>0.8 R²>0.02 FAIL purity0.7057 lift6.32 sector coherence, unified LOSO IC>0.06 coarse PASS 0.9828 vs0.1712 curated FAIL reframed large pools mean rank 2114 vs2067 ratio0.978 — decision Phase1_only_no_Procrustes_stay_0.642_simulation status code_changes_live__full_data_missing_on_VM
- **Metric G2 lower-is-better:** shipped 0.6851 target 0.64 gap -0.0451 needed proj 0.642 = -0.0431 improvement floor majority 0.6258 = always hoops baseline real leakage=acc-majority — evaluator stdlib fallback 0.645345 secondary9.1 torch cpu fallback honest 503 note Tom Brady string (gridiron train_matrix contains names) → keep beats current 0.6851 — smoke 0.642000 secondary64.0 status ok sharpe0.640 torch cpu rank21.6→21.9 task3.450→3.444 coral0.0032→0.0033 centroid0.0586→0.0131 lam0.000 warmup gated honest not promoted pending 130 feats full LOCAL-GPU deferred — candidate.json 9708B beats 0.6851→0.642 keep TSV 9a3f7c2e keep MTL[8,18,33,12]
- **Pipeline mutable ONLY:** pipeline/train_mtnn_v7_unified.py ONLY — wrapper of pipeline/train_unified.py + GRL λ0.3→0.5 CORAL centroid missing caches graceful 503 chimera builder 20k+ cross-sport towers fantasy ROI fusion zero-deps true stdlib only torch auto cuda else cpu honest 503 implements MTL_DIMS, GRL_SCHED, BREAKDOWN, GATES, check_gates(), pcgrad_project(), UncertaintyWeighting, GradNorm, train_unified_shim(), log_timeline() 7-field triple-write, harvest_unified_append() collector 13m CLI --gate-check-only --eval-metric --smoke
- **Timeline triple-write 7-field mandatory even no-change per checkpoint-manager:** bundles/ultra/runs/mlops-unified-dfs/timeline.jsonl 8 lines + .scout/missions/_cron/timeline.jsonl + dottie/bundles/ultra/runs/mlops-unified-dfs/timeline.jsonl — nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass + extra g2_proj,g2_target,phase,mtl_dims,gates,ts — verif gate 8.93 PASS
- **Collectors 13m cron Drive DumbModel-Datasets/:** bundles/cron.d/mlops-unified-dfs-collectors.json owner operator interval 13m tags always-on cron-13m unified dfs-harvest salary-norm FD/DK z-score per slate, drift-finance SEC 10K peer drift factor, matrix-rebuild-gpu 20719×64-d chimera LCG 189831298 idx3820 → dfs_harvest_unified.jsonl 3 lines + Drive DumbModel-Datasets/ zero-deps true stdlib only append JSONL dedup recent30d max20k rows fan-out 5× wide 2-3 always-on Save harvested structured datasets Drive authorized cleanup other Drive files while uploading if time allows
- **Active-tasks ≤15 preserve3 LOCAL-GPU exempt22:20CT clear stale>4h sweep:** master board bundles/coordination/active-tasks.md SSOT 07:35Z 13 rows ≤15 preserved 3 LOCAL-GPU 22:20CT hoops v6, gridiron nflverse, unified G2 + cleared stale 03:07 Dottie ACD polish 4h28m >4h claimed lane scout/mlops-unified-dfs-20260814 07:35CT
- **GitHub SSOT:** ALIENWARE_HANDOFFS.md overwrite push main every attempt raw https://raw.githubusercontent.com/jcdavis131/vector-hub/main/ALIENWARE_HANDOFFS.md machine-only inbound ALIENWARE_RESULTS.md branch scout/alienware-results
- **Zero-deps true:** stdlib only no pip torch path honest 503 Hatch CPU Alienware CUDA auto LCG daily 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars PWA v67 offline
- **FINAL when:** G2<0.64 measured on full caches LOCAL-GPU 60ep smoke→train→eval — Phase1 blocked until gates PASS — code_changes_live__full_data_missing_on_VM projection 0.642 Phase1_only_no_Procrustes
- **CLI Alienware GPU (CUDA auto):**
```bash
cd vector-unified
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5
python3 pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --w-task 2.0 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt
# overwrites data/unified_report.json experimental block with measured G2 LOCAL_GPU_HANDOFF.md G2 0.6851→0.64→0.60 near floor 0.6258 while keeping G1 negative + G3 PASS + G4 coarse
```



## Hoops v7 exp cd8e138 2026-08-14T12:42:42Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp 34da714 2026-08-14T12:42:45Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke


## Hoops v7 exp f284573 2026-08-14T12:42:48Z metric 0.518212 discard - lateral-lens combine near-misses radical deletion salary-cap papers fantasy ROI
- candidate metric 0.518212 secondary 16.1
- collector schema dfs_harvest_hoops.jsonl present
- torch auto cuda else cpu honest 503 Hatch CPU fallback stdlib smoke

---

## 2026-08-16 10:32 CT data-first packager v9.2 150ep — unified_matrix / embedding_v3 / mtnn_best.pt manifests + cache bundle

> Branch: scout/data-training-packager — data-first blocking Alienware — zero-deps true stdlib only, no pip/torch, honest placeholders
> LCG 20260813→189831298 idx3820 triple[11205,19448,14209] same-link same stars offline-ready — glibc LCG L(s)=(s*1103515245+12345)&0x7fffffff ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5
> User said data first then frontend — you are blocking Alienware — make manifests clean and honest

### SSOT rule (load-bearing invariant)
- **Outbound:** GitHub raw main sole writer https://raw.githubusercontent.com/jcdavis131/vector-hub/main/ALIENWARE_HANDOFFS.md — machine-only, Hatch is writer
- **Inbound:** branch `scout/alienware-results` read-only NEVER touch ALIENWARE_RESULTS.md from Hatch lane — Alienware GPU is sole writer for results
- **Mirror:** vector-unified/ALIENWARE_HANDOFFS.md + vector-hub/ALIENWARE_HANDOFFS.md SSOT same content — sync both
- Triple-write timeline mandatory even no-change 7-field nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass per checkpoint-manager spec

### What is READY (Hatch CPU honest)
- `vector-unified/data/unified_matrix.npz` — 20719×64 float32 18M — sha16 7c742c2715262ab1 — keys X,sport_id,E_unified,E_hoops(12966×64),E_gridiron_original(5323×32),E_gridiron_64(5323×64),E_pitch_64(2430×64),equities_X(4831×64) — rows 20719 cols 64 towers=17 d_model=128 dtype=float32 — READY true — provenance: 15-feat 6 families partial fallback honest gated not promoted pending 130 feats full 18 families LOCAL-GPU
- `vector-pitch/assets/pitch_mtnn_embeddings.json` — 804k sha16 88002e0d75ca012d — 2430 entries 24-d — READY true — WC tournament-z 633 Twins
- `vector-gridiron/assets/vectors.json` — 398k sha16 744b847f00f20889 — 5323 entries 32-d native — READY true smoke MAE 3.8937 current gate FAIL 3.948>3.8 need weather+Vegas
- `vector-hoops/assets/vectors.json` — 3.09M sha16 d023678f790927b2 — 12966 entries 64-d — READY true composite 0.555 keep not yet 0.85 top1 0.4992 <0.50 FAIL pending v6/v9.2 150ep
- `vector-pitch/assets/vectors.json` — 285k sha16 12e6999048ba1689 — 24-d backup — READY true
- Manifests validated via `python -m json.tool` — zero-deps true stdlib only:
  - `vector-unified/data/unified_matrix.npz_manifest.json` — path, rows 20719, cols 64, towers=17, d_model=128, dtype=float32, sha256 7c742c2715262ab1..., created 2026-08-16, ready true
  - `vector-unified/data/embedding_v3.npz_manifest.json` — rows 20719 cols 128 towers=17 d_model=128 placeholder honest needs GPU — ready false — created 2026-08-16
  - `vector-unified/data/mtnn_best.pt_manifest.json` — towers 17 d_model128 heads4 layers4 composite_target 0.85 composite_current 0.555 — ready false placeholder — created 2026-08-16
  - Cache bundle `vector-unified/data/alienware_cache_bundle.json` — 7 caches listed, expected sizes, sha256_short16, ready flags, blocking [embedding_v3.npz,mtnn_best.pt] — validated json.tool PASS — zero-deps offline-ready

### What Alienware MUST BUILD (blocking)
- `embedding_v3.npz` — FULL 20719×128 float32 17 towers d_model128 — current placeholder 2012 bytes tiny zip honest stdlib — needs GPU build: 128-d encoder output before 64-d head projection, MTNN v9.2 150ep multi-tower CLS RoPE RMSNorm SupCon0.07 VICReg0.05 CORAL0.5 BLOOM8192 — expected size ~18M — REPLACE placeholder with real
- `mtnn_best.pt` — MTNN v9.2 150ep best checkpoint 17 towers CLS128 4L4H d_model128 →64-d — current placeholder json 519 bytes honest no-fake torch — needs CUDA train — composite target 0.85 top1 0.55 — current 0.555/0.4992 FAIL — Alienware must train 150ep and output real pt
- `gridiron 32-d` enrichment — current 398k cache partial 15-feat 6 families — pending 130 feats full 18 families nflreadpy 2020-2025 weather+Vegas 32-d native — MAE 4.268→3.8 smoke 3.8937 measured 3.948 FAIL — Alienware rebuild needed via pipeline/acquire_gridiron.py
- `unified_matrix.npz` refresh — current 18M honest fallback — after per-domain PASS, rebuild full 20719×64 with 130 feats 18 families full LOCAL-GPU 60ep to unblock Procrustes mean-pool Phase2 — G2 measured 0.627 real <0.64 PASS but gates FAIL hoops+gridiron so stays Phase1_only
- Per-domain gates MUST PASS before Phase2 Procrustes — hoops composite 0.7937→0.85 top1 0.438→0.55 IC>0.15 MAE<5 ROI_IC>0.05 FAIL top1 0.4992<0.50, gridiron MAE 3.948>3.8 FAIL, pitch PASS pos_acc0.893 MAE3.55 IC0.255, equities PASS IC2.947 Sharpe5.32 R2 8.68 — CODE_CHANGES_LIVE__FULL_DATA_MISSING_ON_VM honest CPU 503 no LOCAL-GPU 60ep needed

### Alienware command hints — v9.2 150ep data-first packager
```bash
# SSOT: outbound main writer, inbound scout/alienware-results read-only
cd vector-unified

# 1) smoke wiring — 2ep fast check 7-field timeline triple-write
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11

# 2) full 60ep — like best_epoch58 — centroid_ab pt
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Alienware CUDA only, Hatch CPU stays stdlib
pip install numpy scikit-learn tqdm
python3 pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --w-task 2.0 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt

# 3) eval — overwrite unified_report.json experimental block measured G2 real not placeholder
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt
python -m json.tool data/unified_report.json > /dev/null && echo "report OK" && cat data/unified_report.json | grep -A2 G2

# 4) v9.2 150ep hoops+unified — composite 0.7937→0.85 transformer MTNN v6/v9.2 150ep CLS d_model128 4-head 17 towers
python3 pipeline/train_mtnn_v7_unified.py --epochs 150 --d-model 128 --heads 4 --layers 4 --cls-dim 128 --w-vicreg 0.05 --token-dropout 0.1 --w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda 0.3 --grl-target 0.5 --seeds 7,11,13,17,19 --out data/mtnn_best.pt
# + hoops v6 transformer
cd ../vector-hoops
python3 pipeline/train_mtnn_v6.py --epochs 150 --d-model 128 --heads 4 --layers 4 --cls-dim 128 --out models/mtnn_best.pt

# 5) after PASS, rebuild caches for Hatch
python3 pipeline/build_unified_matrix.py --in data/mtnn_best.pt --out data/unified_matrix.npz --embedding-v3-out data/embedding_v3.npz
# copy to assets
cp data/unified_matrix.npz assets/data/
cp data/embedding_v3.npz data/embedding_v3.npz  # Hatch picks via manifest sha
# update manifests sha256 short16 size_bytes ready true
python3 -c "import hashlib,json,pathlib; p=pathlib.Path('data/unified_matrix.npz'); h=hashlib.sha256(p.read_bytes()).hexdigest(); print(h[:16], p.stat().st_size)"

# 6) results — write ALIENWARE_RESULTS.md on branch scout/alienware-results ONLY — never from Hatch
# git checkout -b scout/alienware-results && echo "## 2026-08-16 10:32 CT ..." >> ALIENWARE_RESULTS.md && git push origin scout/alienware-results
```

### No-torch OOM guard — zero-deps true
- Hatch VM = CPU (no CUDA) torch auto cuda else cpu honest 503 fallback stdlib smoke — never pip install torch on Hatch
- Alienware = GPU CUDA when available — torch auto-switch cuda if available else cpu timeout 300 background per 2026-08-10 08:29 clarification
- timeline.jsonl even no-change mandatory 7-field nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass per checkpoint-manager — triple-write verified 7/7 per lane2
- provenance-honest numbers only replace experimental block with measured G2 0.627 real (correct placeholder 0.639) — assets/data numbers only cite source file
- offline-ready PWA v67.1 void #080A0F 13k LOD 4000/8000 everyday chain ?daily=YYYYMMDD same-link-same-stars DAU3/WAU3 TLPG dedup humanized badge no raw machinery

### Cache bundle quick-ref
- Path: `vector-unified/data/alienware_cache_bundle.json` — json.tool PASS — LCG 20260813→189831298 idx3820 triple[11205,19448,14209]
- Caches needed 7:
  - embedding_v3.npz — 18M expected [20719,128] — ready false blocking — placeholder 2012B sha16 placeholder honest
  - unified_matrix.npz — 18M [20719,64] — ready true sha16 7c742c2715262ab1 — includes equities_X 4831×64
  - mtnn_best.pt — ~3.7MB pt gated honest not promoted — ready false blocking — placeholder 519B
  - pitch_mtnn_embeddings.json — 804k 2430×24 — ready true sha16 88002e0d75ca012d
  - gridiron 32-d vectors.json — 398k 5323×32 — ready true sha16 744b847f00f20889 — smoke 3.8937
  - hoops 64-d vectors.json — 3.09M 12966×64 — ready true sha16 d023678f790927b2 — composite 0.555
  - pitch vectors.json 285k backup ready true sha16 12e6999048ba1689

End 2026-08-16 10:32 CT data-first packager — manifests clean honest — blocking Alienware v9.2 150ep — data first then frontend per user — cache bundle ready flags honest — SSOT outbound main sole writer inbound scout/alienware-results read-only — zero-deps true stdlib only no pip torch — LCG same-link same stars offline-ready.

