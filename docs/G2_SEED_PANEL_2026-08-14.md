# The G2 gate, measured across seeds for the first time

> **Status:** measured 2026-08-14 (6 seeds)
> **Finding:** the gate does not fail marginally — it fails by **3.7x the seed
> spread**. And the `rank_nondeg_floor` of 12.0 sits **4.3 sd above** where the
> model actually lives, so it was never a bar this recipe could clear.

## Why this had never been run

`audit_promotion_gates.py` records it plainly:

> *"UNIFIED STAGE 2 IS THE ONE THAT CANNOT BE CHECKED AT ALL. `train_stage2.py`
> has no `--seed` flag AND no SEED constant to override ... so the unified model
> has never been run at a second seed. Its G2 gate then passes by exactly zero:
> effective_rank 12.0 against rank_nondeg_floor 12, a hardcoded literal ...
> Whether that floor was chosen before or after seeing 12.0 is not recorded, and
> with one run there is no way to tell."*

Both stages now take `--seed`, so the sweep became possible. It also required
fixing a gate that could not pass: the checkpoint save was conditioned on
`rank >= rank_floor`, folding rank never reaches 12.0, so `best_g2` stayed at its
initial `1.0` and shippability evaluated `1.0000 <= 0.7258` forever. See the
`fix(stage2)` commit.

## The panel

Stage 1 (40ep) then stage 2 (30ep) per seed, containerised, sibling encoders
mounted read-only, artifact dirs shadowed.

```
seed   best_g2  best_ep  rank@best  shippable
   5    0.7679       19       11.1      False
   7    0.7599       27       11.1      False
  13    0.7821       25       10.7      False
  21    0.8009       27       10.9      False
  42    0.7780       18       11.3      False
  99    0.7881       25       11.2      False

best_g2   n=6  mean 0.7795  sd 0.0146   min 0.7599  max 0.8009
rank      n=6  mean 11.05   sd 0.22
```

## What it says

**The G2 miss is real, not noise.** The bar is `majority + 0.10 = 0.7258`
(sports are 12,966 / 5,323 / 2,430, so a majority predictor scores 0.6258). The
mean misses it by **0.0537 — 3.7 standard deviations**. `0/6` seeds pass. No
amount of re-rolling gets there; the trunk is not achieving sport-invariance at
this bar and the honest next step is a mechanism change, not another run.

**The rank floor was never reachable.** Effective rank is `11.05 +/- 0.22` and
the floor is `12.0` — **4.3 sd above the mean**, `0/6` clear it. The single
recorded observation of exactly 12.0 was a draw from the tail, and a floor
calibrated to it silently became a wall. Whether to lower it is a real decision,
but it should be made against this distribution rather than against one number.

**The quoted target does not reproduce.** `ALIENWARE_HANDOFFS.md` states the
task as *"G2 0.6851 -> 0.64"*. The measured mean is **0.7795**, worse than the
0.6851 it calls the current state, and 0.6851 is below every one of these six
seeds. Like gridiron's quoted MAE of 4.268 (measured: 4.3502 +/- 0.064), it is a
single lucky draw being carried forward as a level. Anyone climbing toward 0.64
from a claimed 0.6851 is chasing a number this recipe does not hold.

## What would actually move it

Not tuning. The bar is 3.7 sd away, so it needs a mechanism:

- **CORAL is available and unused here.** `--w-coral` and `--w-coral-centroid`
  both default to `0.0` in stage 2, and the docstring says so: *"Stage 2 had NO
  coral term at all."* The centroid term is the one G2's probe can see — a
  sport whose cloud is merely *shaped* like the others is still trivially
  decodable from where it sits. Measured live in stage 1 (0.0054, 0.0807), and
  identically 0.0000 in single-sport hoops training, which confirms it only does
  work when more than one domain is present.
- **GRL lambda** is 0.05 with a ramp; the handoff proposes 0.3 -> 0.5.

Both are one-flag experiments and now cost ~4 minutes per seed. Run them as a
panel, not a single seed — this document exists because the last single seed was
believed for months.

## Reproduce

```bash
python pipeline/seed_panel_g2.py --seeds 5 7 13 21 42 99
```
