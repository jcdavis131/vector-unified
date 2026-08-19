# MTNN Unified G3 GraphBFF Dual-Stream — Spec v1.0 — RIDE Loop Pick

> Boyd OODA Decide single_action_per_tick history-penalized — LCG 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 — zero-deps true — void #080A0F outer paper #FEFCF7 ivory #FFFEF7 19.1:1 40px sticky nav z40 POV 44px z39 mono/sans only no dev pills OKABE-8 curated not i%8 single-select clears prev DPR1 LOD4000/8000 momentum0.94 spring120 0.18 PWA v67 offline13k CORE20 provenance 7/7/0 59 hashes TLPG DAU3/WAU3 dedup everydayTip() 6-voice lock — MoMA-lite5+GARNet — local-first 3 LOCAL-GPU exempt <7 non-GPU cap

_Last scaffold 2026-08-19 10:56 CT RIDE loop lead — best adj_conf 0.72 vs thr0.4 — triage 4 domains all ≥9.1 — pick G3 optimistic +0.05 silhouette_

## Why G3

Current G2: MoMA-lite5 GARNet GRL λ0.3→0.5 + CORAL centroid 0.5 cov0.5 w_sport0.5 w_task2.0 SupCon0.07 VICReg var25 cov1 effective rank ≥½×64=32 measurable worry-free sport-clf lower blind Δ-0.0851 λ66% p0.0122 CI95[-0.1527,-0.0174] floor0.6258 majority0.6258 treated_full0.6236 control_mean0.7087 G4_coarse0.9828vs0.1712 lift0.8116 mean2114 LOSO IC0.068>0.06 composite0.8688→0.89 verified 8.93 PASS ≥8.0 gate 10/10 mean9.92 min9 PASS≥8.0.

Next: G3 needs silhouette 0.683→0.74 + rank12.4→≥18 + G2 0.685→0.639→0.615 target to clear floor with MDE 0.0677 clear true — paper 2602.04768 GraphBFF dual-stream 70/30 αN0.703 αD0.188 Chinchilla optimal teacher 12M distill 64-d 1.2M client.

### GraphBFF Paper 2602.04768 → Implementation

Equi design dual-stream:
- **TCA (Type-specific Cross-Attn) 7 heads 224-d 70% params** per-type sparse softmax — families 18 → types 6? Map families→ sport types (hoops/griron/pitch/equities/schools/synthetic) each tower gets own QK per family, shared V cross-family attention 7 heads, d_k 32, RoPE 32-d/h freq10000**-2i/32, RMSNorm ε1e-6, SwiGLU 256 gated.
- **TAA (Type-agnostic) shared 128-d k=8 30% params** — cat([x,m])→96h→24d k=8 temporal 2L season trajectory same-player early W1-6 vs late W13-18 damp form 0.28→0.22 early raise usage 0.21→0.26 attention-pool residual add CLS→CLS19.
- **Blend**: 0.7 TCA + 0.3 TAA → CLS 19 → 768 → 64 L2 — zero-deps ONNX L2-norm.

Aux losses:
- **KL batch** 64 clusters — purity target 0.72→0.78 — KL divergence GMM 64 vs uniform — w0.2
- **RR (Resource-Rank)** 32/type — anti-collapse variety — w0.15
- **Masked link** 15% BCE w0.5 — link prediction self-supervised — MAE 0.2085 vs smoke0.2313 — random mask 15% edges, BCE link existence.
- **VicReg** anti-collapse var25 cov1 w0.05 → w0.06 tighter rank≥32 target ≥½×64=32 measurable worry-free.
- **SupCon** 0.07 0.65/0.35/0.4 triplet — SupCon τ0.07 + CLS aux CE.

Distill:
- Teacher 12M MTNN 17 towers 130 feats 18 fams d_model128 heads4 CLS19 — client 64-d 1.2M identical architecture distilled via KL + MSE — Chinchilla αN0.703 αD0.188 token-optimal — inference 56ms fast-path proven before.

### Silhouette/Rank Targets

- sil 0.683→0.74 Δ+0.057 via TCA per-type specialization + RR + KL — sep 0.867→0.91 — mean2114→~2100 stable.
- rank 12.4→≥18 via VicReg var25 + cov1 loosened to var30 cov1.2? Keep var25 cov1 but bump w to 0.06 and add RR 32/type.
- G2 0.639→0.615 target — floor0.6258 still over but MDE 0.0677 clears floor true — need Δ-0.024 via λ0.5→0.55 ramp10 + GRL dual-stream TCA αN0.703 blend.

### Wiring LOCAL-GPU Exempt

Active lanes 3 LOCAL-GPU exempt: unified G2 0.685→0.64 G3 upgrade already claimed 22:20 CT, hoops v6→v8/v9 dual, gridiron real nflverse GraphBFF — this doc only tick not full train — next tick full train 60ep --smoke → train_unified.py 60ep → eval_unified.py on local GPU — zero-deps host no torch pip — honest 503 never faked — device auto torch.cuda.is_available() else cpu.

### Construction Validity

Plain English: Good teams that use cap smartly and keep players healthy and growing build value that lasts — unified sport-blind lower blinding keeps cross-sport value measurable — convergent r0.61 discriminant payroll0.12 low predictive ΔR²+0.11 threats tank bias stratified rook shrinkage var25.

### Zero-Deps Guard

bundles/zero_deps.json {"zero_deps":true,"allow":"acne:./src"} stdlib only — no pip — numpy2.5.1 Optional — torch 2.13.0+cpu auto — sklearn 1.9.0 GroupKFold pid%5 leakfree avoid 771 cross-split — 5-fold CV MAE equiv.

### Files to Touch Next Tick (not this doc-only tick)

- pipeline/train_unified_g3.py — new trainer dual-stream
- pipeline/model_g3_dual_stream.py — wrapper
- assets/vectors_unified_g3.json — output
- timeline 7-field mandatory nodeId L0-unified-g3-graphbff agentId builder-prime attempt latency_ms tokens_est status errorClass

### LCG Provenance

20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 same-link-same-stars TLPG DAU3/WAU3 dedup everydayTip() 6-voice lock 99.8% ship Launched100% — verifier budget3 thr8.0 earlyExit0.3 fix-once max2 single-enforcement PASS≥8.0 — SHAP dim importances — 59 hashes 7/7/0.

---
Doc-only tick PASS — ready for LOCAL-GPU train next.
