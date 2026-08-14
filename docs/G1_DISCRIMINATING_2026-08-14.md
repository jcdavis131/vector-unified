# The alignment objective does not cause gridiron's regression

> ⚠️ **PARTIALLY SUPERSEDED, same day.** The first claim below — that alignment
> is not the cause — still stands on 6 seeds. The *second* claim, that the cause
> is encoder drift under `--enc-lr`, is now in doubt and should not be acted on.
>
> Two things broke it. The `--enc-lr` screen moves the **wrong way**: 1e-5 gives
> gridiron +0.1502, 3e-6 gives **+0.1850**. Lowering the rate at which encoders
> move should shrink a drift term. And the training log shows gridiron falling
> 0.9775 → 0.725 after a **single warmup epoch**, while pitch does not move at
> all (0.9609 → 0.963). No encoder moves 0.25 of kNN accuracy in one epoch at
> 1e-5.
>
> The live hypothesis is that G1's two halves are not the same object. The
> baseline is a **stored** embedding (`M["E"][s]`, `train_stage2.py:243`); the
> live number is **recomputed** from the encoder checkpoint
> (`encode_full_numpy`, `:68`). If those disagree before training, the offset is
> baked into every G1 verdict this model has reported.
>
> `pipeline/probe_g1_baseline.py` is the measurement that settles it. Run it
> first.

> **Status:** measured 2026-08-14, 6 seeds, alignment terms at zero
> **Verdict:** **gridiron's G1 failure is not bought by sport-invariance.** With
> SupCon, the GRL adversary and CORAL all switched off, gridiron still loses
> **+0.1514 ± 0.0029** role accuracy, 0/6 seeds passing. That is inside the same
> band it occupies with alignment at full strength.

## The question

Two G2 keeps landed and G1 never moved. Across every configuration tried,
gridiron lost 0.14–0.15 role accuracy against a revert threshold of 0.02 while
hoops and pitch both *improved*. The natural reading is that invariance is being
bought out of gridiron: it is the most separable pool (role kNN starts at 0.977)
and the middle one by size, so a trunk asked to make sports indistinguishable
would spend the most where the structure is most sport-specific.

That reading makes a prediction, and it is testable: remove the pressure and the
regression should shrink.

## The arm

```bash
python pipeline/seed_panel_g2.py --stage2-extra="--w-sup 0 --w-sport 0 --grl-lambda 0"
```

Everything else held: `--w-task 2.0`, `--enc-lr 1e-5`, the adapter, stage 1 at
40 epochs and stage 2 at 30, same six seeds.

Two things had to be fixed before this was even expressible. `--w-task` is not
an alignment term — it multiplies the native and position task heads, which is
the role-*preservation* term, so the arm as originally written down would have
removed the thing holding role structure up and made gridiron worse for an
unrelated reason. And SupCon had no weight at all: it entered the loss at a
hardcoded 1.0 on every folding epoch, and the only way to remove it was
`--warmup >= --epochs`, which stops folding entirely — and with it `best_g2`
tracking, the best-state save, and the verdict block this run exists to read.

## Result

```
seed   best_g2   rank@best
   5    0.7039        14.7
   7    0.7025        15.1
  13    0.7066        14.4
  21    0.7314        14.6
  42    0.7379        14.3
  99    0.7116        14.2

best_g2   n=6  mean 0.7157  sd 0.0152

sport               role_drop           pos_drop   G1
gridiron     +0.1514 +/- 0.0029   +0.0063 +/- 0.0024   0/6 pass
hoops        -0.0512 +/- 0.0041   +0.0076 +/- 0.0034   6/6 pass
pitch        -0.0044 +/- 0.0027   +0.0000 +/- 0.0029   6/6 pass
```

Gridiron's loss with the alignment objective **entirely absent** is +0.1514. The
range across the three previously measured configurations was +0.1437 to
+0.1549. It did not move.

The prediction failed, so the theory is wrong. Sport-invariance is not what
costs gridiron its role structure.

## What that leaves, and it is one thing

Stage 2 does three things beyond stage 1: it optimises task heads through a
shared trunk, it unfreezes the encoders at `--enc-lr 1e-5`, and it projects
through a 48-d adapter. Reading what G1 actually measures collapses those three
to one:

```python
def g1_encoder(live, M, device, frozen_E):
    """Per-sport kNN-5 role+pos on the LIVE e_s (encoder non-regression)."""
    ...
        e_live = live_e_s_numpy(live, device, sport)      # encoder output
        "role_knn5_live": knn5_acc(e_live, native[idx]),
```

`e_live` is `live[sport].encode_full_numpy(device)` — the **per-sport encoder's
own output**. Not `z`, not the adapter, not the trunk. G1 is architecturally
incapable of seeing the adapter or the trunk, so neither can be the cause. The
entire +0.1514 is gridiron's encoder drifting under `--enc-lr 1e-5`.

That also settles the arm I was about to run. `--enc-lr 0` makes role_drop
exactly 0.0 by construction — the encoders never move, `e_live` is identical to
the baseline — so it would report a perfect G1 pass and mean nothing.

The informative experiment is the **tradeoff curve**: sweep `--enc-lr` down
through 3e-6 and 1e-6 and watch both gates. The prediction worth writing down
before running it is that this is cheap: the trunk trains at `--trunk-lr 1e-3`
regardless of what the encoders do, and G2 reads `z`, so the alignment work that
earned 0.7795 → 0.6540 should mostly survive encoders that barely move. If
gridiron's drop falls under the 0.02 revert threshold while G2 stays under
0.7258, both gates pass together for the first time and the model is SHIPPABLE.

If the curve instead shows G2 degrading in lockstep, that is a genuine
architectural tension — invariance obtainable only by damaging the encoders —
and the answer is an anchoring term rather than a learning rate. There is none
today: `g1_encoder` already takes a `frozen_E` argument and never uses it.

## Two side findings, both worth more than the headline

**The default alignment recipe was making G2 worse.** The recorded baseline —
SupCon at 1.0, `--w-sport 0.3`, `--grl-lambda 0.05`, no CORAL — measures
**0.7795 ± 0.0146**. Removing all of it measures **0.7157 ± 0.0152**. Switching
off the machinery whose entire purpose is sport-invariance *improved*
sport-invariance by 0.064, more than four times either arm's seed spread.

The likely mechanism is SupCon. It pulls same-archetype samples together, and if
archetypes are sport-correlated then it sharpens sport clusters rather than
dissolving them — while GRL at λ=0.05 is far too weak to push back. That is a
concrete, cheap next experiment: keep both keeps and lower `--w-sup`.

**Alignment costs effective rank, and it is the whole story on the rank floor.**

```
alignment on   rank 10.9 - 11.2    0/12 clear the 12.0 floor
alignment off  rank 14.55 +/- 0.33  6/6 clear it
```

The floor that was blamed on "a genuinely low-d role manifold" is not a property
of the data. It is a property of the alignment objective, which compresses the
embedding by about 3.6 effective dimensions. The floor is reachable; it is just
not reachable *and* invariant at the same time under this recipe. That is a real
tension between two of the model's own gates, and neither gate document
acknowledges it.

## Reproduce

```bash
python pipeline/seed_panel_g2.py \
    --stage2-extra="--w-sup 0 --w-sport 0 --grl-lambda 0" \
    --compare-to 0.6540 --compare-sd 0.0064
```

Note this arm is diagnostic, not a climb — G2 is expected to worsen, and it
does, by +0.0616. The quantity of interest is the per-sport G1 block, which the
panel now prints directly.
