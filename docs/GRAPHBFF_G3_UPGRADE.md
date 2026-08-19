# GraphBFF → G3 Upgrade — applying 2602.04768 to dumbmodel.com

> Paper: Billion-Scale Graph Foundation Models (GraphBFF), 21 May 2026 — 1.4B param GraphBFF Transformer on ~50B nodes/edges, KL + RR batching, scaling laws L(N,D) = a/N^0.703 + b/D^0.188 + c, 31 PRAUC gains few-shot.

**What we already have that rhymes:**
- 20719×64-d chimera, 17 towers ×130 feats ×18 fams, MoMA-lite5+GARNet Map24 hit80% latency 0.12→0.076ms
- G2 sport-blindness GRL λ0.3→0.5 + CORAL centroid+cov + SupCon + VICReg + UW+GradNorm+PCGrad 136 pairs
- L2-norm client ONNX stdlib, honest 503, PWA v67 offline13.6k, LCG 20260813→189831298 idx3820 triple[11205,19448,14209]

**What GraphBFF adds we’re missing:**

## 1) Architecture — Dual-stream TCA + TAA

Current Fusion = single 4-head Transformer d128 CLS→64-d.

GraphBFF says: split into two attentions, both required for strict expressivity gain.

**TCA (Type-Conditioned Attention) — capacity driver 70-80% params:**
- For us, edge types T_E = {teammate, draft-class, same-pos, same-archetype, trade-link, opponent-matchup, salary-tier}. That's 7 types, analogous to their heterogeneous industrial graph.
- Attention restricted to S ⊆ T_E per head: QK^T only over neighbors sharing that edge type.
- Separate W_q, W_k, W_v per type subset (majority of 1.2M → ~0.9M).
- Sparse softmax: softmax per type, not global neighborhood — prevents high-degree nodes (LeBron: 500+ teammate edges) drowning rare types (trade-link k=2-3).
- Our version: 7 heads instead of 4, one per edge-type family, d_head 32 → d_model 224 (7×32) kept RoPE 32-d/h, RMSNorm ε1e-6.

**TAA (Type-Agnostic Attention) — parameter-efficient stabilizer:**
- Shared W_qkv across all types, single set 128→128, ~0.15M params.
- Fixed-degree sampling k=8 per node (their stability trick) — cap neighbor list at 8 most recent by season, uniform sample without replacement.
- Gives general structural signal, prevents TCA overfit to rare types.

**Fusion:**
```
z_tca = TCA(x)   # 224-d → proj 112→64
z_taa = TAA(x)   # 128-d → proj 64→64
z = L2Norm( 0.7*z_tca + 0.3*z_taa + CLS residual )  # same 64-d sphere
loss = same as G2 but TCA+TAA contrastive regularizer proves strictly more expressive per Theorem 1 in paper
```

Why it matters: our G2 probes show dim8 usage/TS% r0.71 biggest SHAP — that's TAA-like (agnostic quality). Dim18 def-versatility is TCA-like (type-conditioned). Their theorem says you need both. Our current 4-head mixed is neither.

Zero-deps path: implement both in numpy stdlib, export ONNX twin-branch concat, L2-norm client same as before. Teacher 12M param on Alienware, distill to 1.2M client via MSE(z_teacher, z_student) — preserves 64-d sphere.

## 2) Batching — KL + Round-Robin fixes our skew

Our collectors harvest 12966 hoops + 5323 gridiron + 2430 pitch, but hoops dominates 62%. Same skew problem GraphBFF calls out: small edge types ignored.

**KL-Batching (storage-level):**
- Partition Drive harvest into 64 disjoint clusters via k-means on season+team
- Compute empirical p_k per cluster (type histogram over 7 edge types)
- Global p_G = mean(p_k)
- KL(p_k || p_G) low = representative → load first in epoch. This ensures early steps not biased to Lakers-only cluster.
- Impl: precompute clusters offline, write `kl_order.json` LCG-shuffled same seed 189831298 for determinism

**Round-Robin Batching (GPU-level):**
- Once batch in VRAM, iterate edge types cyclically: sample 32 supervision edges per type per mini-batch (instead of random 256 dominated by teammate 180/256)
- Ensures rare trade-link gets consistent gradient every step — paper reports stable pre-train
- Our code: `RRB(n_types=7, per_type=32) → 224 edges` same batch size as now 512→224 link + 224 neg

## 3) Pretrain — Masked link prediction > pure contrastive

We now do InfoNCE + VICReg + SupCon. GraphBFF says add:

- Remove E+ 15% positive edges (teammate, same-archetype links)
- Sample E- negatives 1:1 per type (not random global negatives)
- Predict link existence BCE: model learns topology + features

Keep VICReg var25/cov1 anti-collapse (we already have), add BCE weight 0.5. This gives "universal structural understanding" — their embedding vis shows linear separation zero-shot, ours current silhouette 0.683 good but rank 12.4 low because no link objective.

## 4) Scaling Laws — what 10B would give us

Paper exponents: αN 0.703 (model), αD 0.188 (data). Translation: data without capacity saturates fast.

Our current:
- N=1.2M params, D=20719 nodes
- L(N,D) dominated by N term because N small

Extrapolation:
- 10M teacher (8× N): loss ↓ ~ (8^0.703)=4.6× first term improvement
- 50M teacher (40×): ~14× first term, hits c floor (their c = irreducible graph noise ~0.12)
- Larger = more sample-efficient: they show 1.4B sees 3× fewer examples for same loss vs 100M

Practical: train 12M G3 teacher (fistful of towers, 7 TCA heads), 60ep full, then distill to 64-d 1.2M client. Expect effective rank 12.4 → ≥32 measurable (their 512-d ranks collapsed without VICReg var25), G2 0.685→0.639 → G3 0.615 proj, silhouette 0.683→0.74, coarse NN 0.9828 stays.

Budget: Alienware 4090 10M param ~2.4GB, batch 224 links, 60ep ~45min — fits our current pipeline handoff.

## 5) Eval — linear probe proves universal

They use frozen reps + linear probe vs task-specific HGT/HAN — wins 31 PRAUC, 10 shots per class beats full-data baseline.

We can replicate:
- Freeze 64-d z, train LogReg max_iter400 C1.0 on 10 tasks: hoops pos 5-way, archetype 8-way, gridiron QB/WR/RB/TE fantasy tier, pitch difficulty percentile, equities sector 11-way, win total regression
- Their few-shot: 10 samples per class. Our pitch 633 ent = few-shot naturally — expect same gain: GraphBFF few-shot beat full-data HAN, we can show linear probe 10-shot > trained-from-scratch MLPs

Already we have 5-fold CV seed 7/11/13/17/19 paired t — add linear probe suite, report PRAUC per task.

## 6) Concrete G3 Recipe (next branch)

```
g3 = GraphBFF-Chimera v3
- nodes: 20719 + 500 equities LOSO after pass 0.068 IC gate (adds 21219)
- edge types 7: teammate, draft-class, same-pos, same-arch, trade, opponent, salary-tier
- TCA: 7 heads d_head32 224-d 70% params, per-type sparse softmax
- TAA: 1 head shared 128-d 30% params k=8 sampling
- Fusion: CLS 19→20 tokens + RoPE 32-d/h + RMSNorm ε1e-6 + SwiGLU 256 gated (keep v8)
- Pretrain: masked link 15% + neg 1:1 type-balanced + VICReg var25 cov1 w0.05 + SupCon τ0.07 w0.15 + BCE link w0.5
- Batch: KL storage order 64 clusters + RR GPU 32/type
- Optimize: AdamW wd2e-4 OneCycle warm10% lr1.5e-3 60ep val_every5 metric (G2 + composite)/2
- Distill: teacher 12M → student 64-d MSE
- Provenance: 7/7/0 same, 59→73 hashes (add 14 edge type counts), LCG both chains same-link-same-stars
- Target: G2 0.639→0.615 (-0.024), rank 12.4→≥32, silhouette 0.683→0.74, composite 0.8688→0.91
```

**Infra:** zero-deps stdlib inference unchanged, torch optional local only, honest 503 unchanged, PWA v67 unchanged, void #080A0F 40px sticky unchanged, 6-voice lock unchanged.

**When to flip G2→G3:** need measured G2 60ep first to lock floor 0.639 conservative, then G3 smoke2ep check GRL λ+ranks, then 60ep full — same flow as before, no extra gates.

---

Refs:
- 2602.04768 αN0.703 αD0.188 TCA 70-80% params sparse softmax per type TAA shared k-fixed Theorem dual>single
- Our current UNIFIED_G2_ARCH.md v2.1
- MTNN_V8_ARCH.md v8 RoPE RMSNorm SwiGLU VICReg var25 cov1 SupCon 0.07
- Scaling law: L(N,D)=a/N^αN + b/D^αD +c, sample efficiency via larger N

Owner: scout aliased — zero-deps true stdlib only — next: create branch scout/unified-g3-graphbff and run 2ep smoke.
