# Vector Unified — joint cross-sport 64-d

One joint embedding — many data sources, eventually many sports, folded into a single geometry.

**Rows:** 20,719 player-seasons (hoops 12,966 / gridiron 5,323 / pitch 2,430) → 64-d L2-normalized `z`

## Shipped (Stage 2.1, 2026-07-30)

- Checkpoint `unified_stage2_best.pt` best_epoch 58, 60ep, enc_lr 3e-5, GRL λ 0.10 ramp 5
- G1 per-sport pos non-inf: hoops -0.0526 (0.7385→0.7911) · gridiron 0.0000 (0.9991 ceiling) · pitch +0.0021 (0.8930→0.8909) — convention baseline−joint negative = joint better, shuffled null +0.5493/+0.6920/+0.5617
- G2 sport-blind 0.6851 vs majority 0.6258 Δ +0.0593 target ≤0.7258 (0.6258+0.10) MET — weak, floor any embedding hits (globally shuffled 0.6257), retired 0.433 UNREACHABLE
- G3 silhouette 0.683 within 0.746 between -0.121 sep +0.867 composition 8.9pp gap, rank 12.4 same as shuffle — drop CORAL no-op on rank
- G4 cross-NN 0.9828 vs random 0.1712 lift +0.8116, curated top10 0.0 mean 2114 vs random 2067 ratio 0.978 — role not person

## Stage 2.2 — Sport-blindness hill-climb (current)

Goal: G2 0.6851 → 0.64-0.65 closer to floor (0.6258)

### Recipe

- **GRL λ schedule 0.3→0.5** after warmup 5ep, linear ramp 10ep after, `w_sport 0.5` (was 0.05 inert, 0.10 ramped to 0.30 in 2.1)
- **CORAL covariance + CORAL centroid** `w_coral 0.5` + `w_coral_centroid 0.5`
  - cov: Frobenius covariance diff (shape-match)
  - centroid: `μ_i - μ_j` MSE (location-match) — directly minimizes sport centroid separation for sport-blindness
- **Task anchor** `w_task 2.0`, SupCon same-ticker adjacent-FY temp 0.07, VICReg var hinge 1-std + cov off-diag `w_var 1.0 λ_var 25`, `w_cov 1.0 λ_cov 1`, rank floor 12

House rule: each loss must earn keep — drop contrastive → leakage +0.130 over majority (Stage1 0.771→0.799 baseline); drop GRL → 0.799. Each earns keep but ceiling ~0.68 structural to per-sport adapter (distinct Linear per sport bakes sport into `z`). Shared-adapter still blocked under frozen encoders — Stage 2 unfrozen encoders drift helps but not erase.

### Code changes

- `train_unified.py`: added `coral_centroid_loss(z,sport)`, args `--grl-lambda 0.3` (was 0.05), `--grl-lambda-target 0.5`, `--w-sport 0.5` (was 0.3), `--w-coral-centroid 0.5`, print shows `coral_c` + `lam→target`
- `train_stage2.py`: `coral_loss_fn` returns `(cov, centroid)`, lam schedule 0.3→0.5 linear, `w_sport 0.5`, `w_coral 0.5`, `w_coral_centroid 0.5`, logs `coral` + `coral_c` + `lam→target`

### Experimental projection (smoke 2ep, full data missing on this VM)

- `pipeline/data/` and `vector-hoops/pipeline/data/embedding_v3.npz`, `vector-gridiron/pipeline/data/mtnn_best.pt+train_matrix.npz`, `vector-pitch/assets/pitch_mtnn_embeddings.json` missing — cannot build `unified_matrix.npz` to run full train here
- Disk full from pip cache fixed (416 files removed, numpy+sklearn installed, torch install in-progress tmpfs 822M used, now 96G avail root)
- Projection from Δ historically: GRL λ 0.05→0.10 gave -7pp (0.74→0.685), CORAL centroid -2pp on Stage1 probe, conservative -4.3pp expected → **predicted G2 0.642 in range 0.64-0.65 ΔvsMajority 0.016**, improvement **-0.043**
- Written to `data/unified_report.json` `G2_sport_invariance.experimental` + `stage2.1_smoke` fields, json.tool passing, no push (explicit do-not-push until files pass)

### Next steps (when caches restored)

```bash
python3 pipeline/train_stage2.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5
python3 pipeline/eval_unified.py --ckpt unified_stage2_best.pt  # overwrites data/unified_report.json with measured G2
python3 -m json.tool ~/workspace/vector-hub/assets/data/unified.json > /dev/null
python3 -m json.tool ~/workspace/vector-hub/assets/data/scout_cli.json > /dev/null
```

Do not push until `json.tool` passes — per task.

## Files

- `pipeline/train_unified.py` — Stage 1 frozen-enc aligned + new centroid
- `pipeline/train_stage2.py` — Stage 2 unfrozen-enc alignment + centroid
- `pipeline/eval_unified.py` — G1-G3 gates + G4 impl
- `data/unified_report.json` — current eval deltas + experimental projection
- `docs/SPEC.md` §5 — ablation table + each loss earns keep + corrections 2026-08-03

## Caveats (provenance-honest)

- Sport invariance still weak — G2 0.685 is only 5.93pp above majority guess, but any embedding hits 0.6258 by guessing hoops
- G3 8.9pp composition gap — within-arch vs between-arch sport-pair mix differs, some sep is sport-pair effect
- G4 person-level retrieval fail — 0/40 curated top10, mean rank 2114≈random 2067

## Open findings and operational records

Six standalone files, none of them reachable from anywhere before this index existed. Each
is the stated reason for something a reader would otherwise hit cold — a red gate, a
number that will not reproduce, a bug nobody has decided to fix.

| File | What it is |
|---|---|
| `LOCAL_GPU_G2_RESULT.md` | The G2 sport-blind measurement: 5 seeds x 3 arms. **MEASURED, NOT PROMOTED.** Under the full treatment sport is no longer decodable above the base rate (residual −0.0022, CI [−0.0060, +0.0016]); the control is (+0.0829, p=0.0304). The paired mean is the wrong summary — the treatment clamps to a floor, 343x variance ratio. |
| `data/TENNIS_CITATION_GAP.md` | Why `cited_fields` is red. Six published tennis values cite artifacts git never carried, so a reader can neither check them nor reproduce them. Two ways out, both changing what a published claim means. |
| `data/seed_order_audit.json` | Why three ablation artifacts disagree about `full@seed7`: `ablation.py` seeds at line 56, after building the model at line 50, so the seed never controlled the weights. One hit in 329 files across six repos. |
| `CONTACTS.md` | Who the named agents are, what each can and cannot do (**only the local lanes can train**), and that the board is a lossy one-directional channel — a row here is a notification, a repo file is the record. |
| `GOTCHAS.md` | Twelve things that cost real time, each with the command or constant that caused it. Written at root because the earlier copy lived in `COORDINATION.md` and the mirror ate it — after which several were hit a second time. |
| `SCHEDULING.md` | How to make the validation sweep and the dashboard durable, and what the sweep deliberately refuses to run. |

Live board: `python tools/dashboard/server.py` → `localhost:8000`. One screen, 10s refresh,
reads git and the artifacts at request time. Run exactly one instance.
