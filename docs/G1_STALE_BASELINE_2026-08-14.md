# G1 was subtracting two different artifacts

> **Status:** measured 2026-08-14, no optimiser step taken
> **Verdict:** **the gate was broken, not the model.** G1's baseline read a
> *stored* embedding while G1's live number *recomputed* one from the encoder
> checkpoint. Those disagree by **+0.2526** for gridiron before training begins.
> The reported regression of +0.1502 is smaller than the offset baked into it.

## The two halves

```python
# baseline, train_stage2.py:243 (before this fix)
e_froz = M["E"][s].cpu().numpy()                    # STORED, off unified_matrix.npz

# live, train_stage2.py:68
e_live = live[sport].encode_full_numpy(device)      # RECOMPUTED from the checkpoint
```

`G1 = baseline − live`. Those are only comparable if the matrix was built by the
same checkpoints the live loader picks up, and nothing enforced that.

## Measured, with no training step

`pipeline/probe_g1_baseline.py` loads exactly what stage 2 loads and scores both
halves:

```
sport            n    stored      live     delta          dim     cos
hoops        12966    0.8408    0.7992   +0.0416     64 vs 64   0.010
               pos    0.7385    0.6994   +0.0391
gridiron      5323    0.9775    0.7249   +0.2526     32 vs 32   0.692
               pos    0.9991    0.8789   +0.1202
pitch         2430    0.9609    0.9609   +0.0000     24 vs 24   1.000
               pos    0.8930    0.8930   +0.0000
```

The file dates say why:

```
unified_matrix.npz              2026-07-31 06:21    <- stored E built here
vector-pitch/pitch_mtnn.pt      2026-07-11 10:04    OLDER   -> cos 1.000, delta 0.0000
vector-gridiron/mtnn_best.pt    2026-08-06 19:41    NEWER   -> cos 0.692, delta 0.2526
vector-hoops/mtnn_best.pt       2026-08-14 06:48    NEWER   -> cos 0.010, delta 0.0416
```

**Pitch is the control and it is a perfect one.** It is the only sport whose
checkpoint predates the matrix, and it is the only sport with an exactly zero
delta and a cosine of 1.000. The other two were retrained after the matrix was
built and both stored blocks are stale.

Hoops is the strange one: cosine **0.010** — the stored and live embeddings are
essentially orthogonal — yet kNN differs by only 0.042. A retrain landed in a
completely different basis that is a comparably good representation. Cosine and
downstream accuracy are answering different questions, and only one of them
noticed.

## What it cost

Gridiron's reported `role_drop` is +0.1502. The offset present *before training*
is +0.2526. So the reported failure is smaller than the artifact gap it is
measured against, and like-for-like the direction reverses:

```
sport      step-0 live   best-epoch live   what stage 2 actually did
gridiron        0.7249            0.8273   IMPROVES  +0.1024
hoops           0.7992            0.9002   IMPROVES  +0.1010
pitch           0.9609            0.9671   IMPROVES  +0.0062
```

All three sports improve. G1 has been reporting `FAIL` for the entire life of
the gate, and `SHIPPABLE: False` with it.

## Two results that finally make sense

**Gridiron's constancy.** Its `role_drop` sat in 0.1437–0.1549 across 21 runs
and four configurations, including one with the alignment objective *entirely
removed*. A constant offset does not move, because nothing being varied causes
it. I spent a 6-seed panel refuting the theory that invariance was bought out of
gridiron; the refutation was correct and the reason was not the one I proposed.

**The `--enc-lr` screen ran backwards.** Four points, monotone the wrong way,
with pitch flat throughout:

```
enc-lr   G2       gridiron   hoops     pitch
1e-5     0.6585   +0.1502    -0.0594   -0.0062
3e-6     0.6672   +0.1850    -0.0177   -0.0041
1e-6     0.7054   +0.2188    +0.0220   -0.0041
3e-7     0.7181   +0.2432    +0.0308   -0.0021
```

Encoder learning was **repairing** the gap, not causing it. Slowing it down left
more of the artifact mismatch in place, which read as more damage — and pitch,
which had nothing to repair, did not move at any setting. That is the shape of a
stale-artifact problem, not a drift problem.

## This is a correction to the instrument, not a relaxation of the gate

Worth stating plainly, because the opposite move was available and refused twice
already this week. `--revert-threshold` is untouched at 0.02. Nothing about what
counts as a regression changed. What changed is that the two quantities being
subtracted are now the *same measurement* — the live encoder, in eval mode,
before the first optimiser step.

Relaxing a gate so a result can pass is the failure mode this whole exercise
exists to avoid. A wrong baseline making a passing model look broken is a
different thing, and fixing it is not the same act.

The stored embedding is still scored as `stored_role_knn5`, still printed with a
`STALE` flag past 0.02, and still emitted as `stored_vs_live_role` in
`stage2_report.json`. **`unified_matrix.npz` does need a rebuild** — that is a
real problem. It is just not a verdict on stage 2, and one number could not
carry both.

## The repo already knew this, for a different quantity

`eval_unified.py:92-108` states the rule in capitals:

> *"THE ONE PLACE THIS RULE LIVES. Seven modules call `encode_all()`, which uses
> the FROZEN cached per-sport outputs in `M["E"]`. That is correct for Stage 1
> and WRONG for Stage 2, whose premise is that the encoders were unfrozen and
> drifted."*

That was written about **z** — the trunk output — and fixed by centralising it
in `load_and_encode()`. The identical defect in **e_s**, the per-sport encoder
baseline, was never covered by it. Three call sites, three different answers:

| file | how it builds `e_s` | correct? |
|---|---|---|
| `stage2_eval.py:62` | `live[s].encode_full_numpy(device)` | yes |
| `train_stage2.py:243` | `M["E"][s]` | **no** — fixed here |
| `eval_unified.py:166` | `M["E"][s]` | **no** — still open |

So `native_knn5_e_s` and `pos_knn5_e_s` in `eval_unified.py` carry the same
staleness for any Stage 2 checkpoint, by the repo's own argument.

**Not changed here, deliberately.** That file is the evaluation harness and the
scale every historical number was recorded on; moving it silently would rescale
the record rather than improve anything, which is the failure the program doc
names. It needs its own decision and its own re-measurement. Flagged rather than
fixed.

## What survives unchanged

- **The G2 arc, 0.7795 → 0.6856 → 0.6540.** Every arm shared this same broken
  baseline and the same matrix, so comparisons *between* arms were never
  affected. Both keeps stand.
- **The alignment finding.** Gridiron's regression is not caused by the
  alignment objective — that was measured across arms and remains true. What
  changed is that there was no regression to explain.

## Reproduce

```bash
python C:\Users\jcdav\herdmux\gpu\train_local.py vector-unified \
    --entry pipeline/probe_g1_baseline.py
```

No prepare step and no optimiser step; it takes about 12 seconds inside the
container.
