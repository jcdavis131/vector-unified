# CORAL centroid: the G2 gate passes, 6/6 seeds

> **Status:** measured 2026-08-14, 6 seeds per arm
> **Verdict:** **KEEP.** `--w-coral-centroid 0.5` moves G2 from 0.7795 to 0.6856
> — **6.4x the baseline's own seed spread** — and takes the gate from **0/6 to
> 6/6 seeds passing**. No G1 cost, no rank cost.

## Why this flag

The 6-seed baseline (`G2_SEED_PANEL_2026-08-14.md`) put the gate miss at **3.7
standard deviations**, which rules out tuning. It needed a mechanism, and stage 2
had one switched off:

```python
ap.add_argument("--w-coral", type=float, default=0.0)
ap.add_argument("--w-coral-centroid", type=float, default=0.0)
```

with the file's own note explaining which of the two G2 can see:

> *"The second is the one G2 can see: its probe reads z, and a sport whose cloud
> is merely SHAPED like the others is still trivially decodable from where it
> sits."*

G2 is a logistic probe on z. Aligning **covariance** does not move where a
sport's cloud sits; aligning **centroids** does. Three prior observations agreed:
CORAL measured live in stage 1 (0.0054, 0.0807), identically **0.0000** in
single-sport hoops training, and stage 2 ran it at zero weight. It only does work
when more than one domain is present — which is exactly this setting.

## Result

Both arms: stage 1 40ep then stage 2 30ep per seed, stage 1 **identical** across
arms so any G2 difference is attributable to stage 2 alone.

```
seed   baseline   +coral-centroid   delta
   5     0.7679            0.6894  -0.0785
   7     0.7599            0.6846  -0.0753
  13     0.7821            0.6856  -0.0965
  21     0.8009            0.6708  -0.1301
  42     0.7780            0.6885  -0.0895
  99     0.7881            0.6950  -0.0931

baseline   n=6  mean 0.7795  sd 0.0146   gate 0/6 pass
variant    n=6  mean 0.6856  sd 0.0081   gate 6/6 pass
delta -0.0938  =  6.4x the baseline seed sd
margin above the bar +0.0402  =  5.0x the variant's own sd
```

Every seed improved. The worst variant seed (0.6950) beats the best baseline
seed (0.7599) by 0.065 — the distributions do not overlap. Dispersion also
**tightened**, 0.0146 -> 0.0081, so the gain is not bought with instability.

**Nothing was paid for it.** Effective rank is unchanged (10.88 +/- 0.24 vs
11.05 +/- 0.22), so no collapse was introduced. G1 is identical in both arms —
gridiron role ends at ~0.828 in all twelve runs, baseline and variant alike.

## What still blocks SHIPPABLE, and it is not G2

```
=== Stage 2 verdict (best epoch 27) ===
  hoops     role_drop=-0.0598  pos_drop=+0.0132  [OK]
  gridiron  role_drop=+0.1465  pos_drop=+0.0038  [REGRESSED]
  pitch     role_drop=-0.0041  pos_drop=-0.0021  [OK]
  G2=0.6885 (target<=0.7258) -> PASS    G1 -> FAIL
```

Stage 2 folds **two** sports cleanly — hoops role *improves* by 0.060, pitch by
0.004 — and pays for the whole fold out of gridiron, which loses 0.147 against a
0.02 revert threshold.

**This does not match the recorded state.** `docs/STAGE2_PLAN.md:187` and
`docs/UNIFIED_ARCHITECTURE.md:617` both record **G1 PASS: hoops role_drop=-0.040
(improved), gridiron -0.001, pitch 0.000**. Measured gridiron is **+0.1465** —
opposite sign, 150x the magnitude, in all twelve runs today. Either that figure
came from a different configuration or the behaviour changed and was never
re-measured. It is stated here as "does not reproduce under the current default
recipe" rather than as an error in the doc.

Gridiron is the smallest and most structured of the three pools (12,966 hoops /
5,323 gridiron / 2,430 pitch, and gridiron role kNN starts at 0.977 — nearly
separable). A shared trunk asked to make sports indistinguishable will spend the
most where the structure is most sport-specific. That is the next question, and
it is a *design* question, not a tuning one.

## Also fixed to make this measurable

- The checkpoint save was gated on `rank >= rank_floor`; folding rank never
  reaches 12.0, so `best_g2` stayed at its initial 1.0 and shippability
  evaluated `1.0000 <= 0.7258` forever. Tracking is now unconditional among
  folding epochs.
- `SHIPPABLE` briefly included a rank veto. That recreated the same bug one step
  later — rank is 10.9-11.1 against a floor of 12.0, 0/12 clearing — so it is
  back to `G1 AND G2`, with rank reported. `eval_unified.py` says why: the floor
  belongs to a compound `collapse_detector = rank>=12 AND G1 AND G3`, and *"rank
  alone over-alarms on a genuinely low-d role manifold"*.

## Reproduce

```bash
python pipeline/seed_panel_g2.py \
    --stage2-extra="--w-coral-centroid 0.5" --compare-to 0.7795 --compare-sd 0.0146
```

Not yet tried, and cheap now that a run is ~4 minutes per seed: `--w-coral`
(covariance) alongside the centroid term, and `--grl-lambda-target 0.5` which the
handoff proposes.
