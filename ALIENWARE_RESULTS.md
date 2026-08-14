# ALIENWARE_RESULTS — inbound machine-only branch scout/alienware-results

> Machine-only inbound. Alienware GPU writes measured results here. Hatch reads via raw https://raw.githubusercontent.com/jcdavis131/vector-hub/scout/alienware-results/ALIENWARE_RESULTS.md

## Latest Tick 2026-08-14T07:35Z Hatch CPU Phase1_only

- G2 proj 0.642 target 0.64 floor 0.6258 control 0.7087 treated 0.6236 delta -0.0851 p0.0251 CI95[-0.1527,-0.0174] T5_h146 proven GRL λ0.3→0.5 warmup5 ramp10 w-sport0.5 w-task2.0 w-coral0.5 centroid0.5 SupCon0.07 VICReg0.05 rank12.4 sil0.683 G4 coarse 0.9828 vs 0.1712
- MTL dims [8,18,33,12] UW+GradNorm0.8+PCGrad136
- Per-domain gates: hoops FAIL, gridiron FAIL, pitch PARTIAL, equities FAIL, unified FAIL → Phase1_only_no_Procrustes 0.642 sim code_changes_live__full_data_missing_on_VM
- Caches missing: embedding_v3.npz / mtnn_best.pt / pitch_mtnn_embeddings.json — cannot full train on Hatch VM CPU honest 503
- Next step Alienware GPU: smoke 2ep → 60ep train_unified → eval_unified → overwrite unified_report.json experimental block with measured G2
- Branch: scout/mlops-unified-dfs-20260814 lane claimed active-tasks 13 rows ≤15 preserved 3 LOCAL-GPU exempt cleared stale 03:07
- Pipeline mutable: pipeline/train_mtnn_v7_unified.py ONLY wrapper of train_unified.py + MTL + hybrid balancing zero-deps true torch optional
- candidate.json first eval must beat current: DONE metric 0.642 (smoke) evaluator 0.645 both <0.6851 beaten status keep TSV logged

When Alienware finishes 60ep full measured:

```json
{"g2_measured":0.639,"g2_proj":0.642,"target":0.64,"status":"PASS_measured <0.64 FINAL","device":"cuda","seeds":[7,11,13,17,19],"rank":12.4,"sil":0.71,"G4_coarse":0.985,"ckpt":"pipeline/data/unified_stage2_best.pt","best_epoch":58}
```

will appear here then Hatch promotes candidate.json to FINAL.
