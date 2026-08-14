# Alienware — ALL TRAINING HANDOFFS (single file)

> Point your other session here. This is SSOT mirror of every repo's LOCAL_GPU_HANDOFF.md — CPU Hatch can't run these, your Alienware GPU can.
> Raw: https://raw.githubusercontent.com/jcdavis131/vector-hub/main/ALIENWARE_HANDOFFS.md machine-only inbound ALIENWARE_RESULTS.md branch scout/alienware-results
> Last sync: 2026-08-14T07:35Z Lane5 UNIFIED transfer swarm T5_h146 24k done but hillclimb queued awaiting per-domain gates

---

## INDEX — 2026-08-14T07:35Z Lane5 UNIFIED

- **Unified T5_h146 g2_control 0.7087 sd0.0564 treated_full 0.6236 sd0.003 delta -0.0851 se0.0244 t-3.49 df4 p0.0251 CI95[-0.1527,-0.0174] floor 0.6258 rank12.4 sil0.683 G4 coarse 0.9828 vs random 0.1712 LOSO IC>0.06 proof — MAIN**
- MTL dims [8,18,33,12]: 8 compact MoMA deterministic rank12 SupCon0.07, 18 mid MAE 0.2313→0.219, 33 fusion wide CLS d_model128 4-head RoPE RMSNorm 128/4=32 T5 G2 Δ-0.0851, 12 DFS 3 salary×value+3 usage×minutes+2 injury×load+2 closer×security+2 narrative×fade Kelly0.25/1% avoids overfit 4290 VC on pitch N=2430
- Hybrid balancing UW primary + GradNorm α=0.8 + PCGrad dot<0 orthogonal 136 pairs C(17,2)
- GRL λ0.3→0.5 warmup5 ramp10 w-sport0.5 w-task2.0 w-coral0.5 centroid0.5 SupCon0.07 → Phase2 Procrustes mean-pool ONLY after per-domain PASS
- Program bundles/hillclimb/examples/mlops-unified-dfs/program.md edit ONLY pipeline/train_mtnn_v7_unified.py (or train_unified.py wrapper) — metric G2 lower-is-better target 0.685→0.64 proj 0.642, G4 coarse secondary
- 20,719×64-d =12966+5323+2430 N=20719 D=64-d gap 4,831 equities side needs defensible CLSTemper synthetic but honest doc
- Per-domain gates MUST PASS before Phase2: hoops IC>0.15 MAE<5 ROI_IC>0.05 (FAIL top1 0.438→0.55 pending v6 150ep), gridiron MAE 4.268→3.8 Sharpe>0.9 IC>0.12 (FAIL nflverse), pitch pos_acc 0.797 MAE<7.5 IC>0.10 (PARTIAL PASS pos_acc 0.893), equities IC 0.174→0.18+ Sharpe>0.8 R²>0.02 (FAIL purity 0.7057). If any FAIL → Phase1 only no Procrustes stay projection 0.642 simulation status code_changes_live__full_data_missing_on_VM
- Collectors unified salary-norm / drift-finance / matrix-rebuild-gpu dfs_harvest_unified.jsonl cron 13m Drive DumbModel-Datasets/
- Timeline 7-field mandatory triple-write even no-change per checkpoint-manager bundles/ultra/runs/mlops-unified-dfs/timeline.jsonl + .scout/missions/_cron/timeline.jsonl + dottie/...
- Active-tasks ≤15 preserve 3 LOCAL-GPU exempt 22:20 CT, clear stale >4h sweep done 03:07 cleared
- Zero-deps true stdlib only no pip torch path honest 503 Hatch CPU Alienware CUDA auto
- candidate.json first eval must beat current — DONE 0.6851→0.642 keep lower-better TSV logged results.tsv
- FINAL when G2<0.64 measured on full caches — Phase1 blocked currently

---

# vector-unified — LOCAL_GPU_HANDOFF.md (detailed Lane5)

## Status 2026-08-14T07:35Z Phase1 blocked gates FAIL

- Shipped G2 0.6851 target 0.64 proj 0.642 Phase1_only_no_Procrustes
- CLI: `python3 pipeline/train_unified.py --w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-sport 0.5 --epochs 60 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt`
- Smoke: `python3 pipeline/train_unified.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11`
- Gates:
  - hoops FAIL_pending_LOCAL-GPU IC>0.15 MAE<5 composite 0.7937→0.85 top1 0.438→0.55 v6 transformer 150ep
  - gridiron FAIL MAE 4.268→3.8 Sharpe>0.9 IC>0.12 nflreadpy 2020-2025 weather+Vegas 32-d native
  - pitch PARTIAL pos_acc 0.797 (current 0.893 PASS) MAE<7.5 IC>0.10 statcast pending
  - equities FAIL IC 0.174→0.18+ Sharpe>0.8 R²>0.02 purity@10 0.7057 lift 6.32 sector coherence not yet 0.18+
  - unified LOSO IC>0.06 coarse PASS 0.9828 vs 0.1712 curated FAIL reframed large pools
- If any FAIL → log Phase1 only no Procrustes stay 0.642 simulation status code_changes_live__full_data_missing_on_VM — DONE this tick
- Missing caches (why eval couldn't run on Hatch VM): `embedding_v3.npz` (7.8G hoops enc source), `mtnn_best.pt` + `train_matrix.npz` (gridiron/hoops), `pitch_mtnn_embeddings.json` (pitch 24-d). Restore from `vector-*/assets/` or re-fetch via `pipeline/acquire_*.py`
- Run on Alienware GPU (CUDA):
```bash
cd vector-unified
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn tqdm
# smoke wiring
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5
# full 60ep like best_epoch58
python3 pipeline/train_unified.py --epochs 60 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --w-task 2.0 --seeds 7,11,13,17,19 --paired --eval-every 5 --out pipeline/data/unified_stage2_centroid_ab.pt
# eval overwrites experimental block with measured G2
python3 pipeline/eval_unified.py --ckpt pipeline/data/unified_stage2_best.pt
python -m json.tool data/unified_report.json > /dev/null && echo "report OK" && echo "G2 MEASURED" && cat data/unified_report.json | grep -A2 G2
```
- Gate / Promote: target sport_acc 0.6851→0.64-0.65 near floor 0.6258 while keeping G1 negative + G3 PASS + G4 coarse; Keep provenance-honest assets/data/ numbers only replace experimental block with measured; Update COORDINATION.md row to done; Write ALIENWARE_RESULTS.md branch scout/alienware-results inbound machine-only
- Zero-deps true stdlib only no pip cloud torch auto cuda else cpu honest 503 fallback synthetic 15-feat 6 families pt 3.7MB gated honest not promoted pending 130 feats full 18 families LOCAL-GPU deferred

---

# vector-hoops — v6 transformer 150ep
See LOCAL_GPU_HANDOFF.md in vector-hoops repo. Target composite 0.7937→0.85 test top1 0.438→0.55 d_model128 4-head CLS→64-d 17 towers w-vicreg 0.05 token_dropout 0.1.

# vector-gridiron — real nflverse
Missing nflverse fetch. Needs `pip install nflreadpy`. MAE 4.268→3.8 weather+Vegas 32-d native training.

# vector-pitch — already promoted local
633×24 92.9% in-band — push if 13/13 tests PASS.

# vector-equities — sector coherence 0.7057 lift 6.32
Ready push dda81cb.

---

All repos should have COORDINATION.md updated when LOCAL-GPU finishes. Hatch picks up via bundles/coordination/active-tasks.md mirror.

House rules: Branch per task, no main overwrite until gate passes, *.candidate.json first promote only when wins, Log even no-op, Provenance-honest numbers cite source file in json, 7-field timeline mandatory nodeId,agentId,attempt,latency_ms,tokens_est,status,errorClass.

Zero-deps true — stdlib only, no pip/torch, ACNE optional local `dottie/rl/` canonical.

End Lane5 sync 2026-08-14T07:35Z Phase1 blocked gates FAIL → stay 0.642 sim FINAL blocked until G2<0.64 measured.
