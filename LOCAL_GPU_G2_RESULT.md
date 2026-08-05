# LOCAL GPU lane — G2 sport-invariance, measured

**Status: MEASURED, NOT PROMOTED.** The shipped model is untouched and hash-verified
(`sport_acc 0.6851`, ckpt `b055641c03760624`). Promotion is the operator's call.

This file exists because the same result was posted to `COORDINATION.md` twice and lost
both times: the active-tasks mirror overwrites that file wholesale (`dcebea5` says "rebase
on 7367a7d", which was the board commit carrying it). Every mirror commit checked touches
`COORDINATION.md` and nothing else, so a sibling file survives. Three heartbeats
(03:43 / 04:13 / 04:43 CDT 2026-08-05) recorded "no local GPU measured G2 ... predicted
0.64-0.65" while this measurement sat on a pushed branch.

Branch: `local/unified-g2-gpu`. Artifacts: `data/g2_centroid_ab.json`,
`data/stage2_seed_floor.json`, `data/ablation_determinism.json`.

## Design

Three arms, 5 seeds (7/11/13/17/19), 60 epochs, paired — same seeds, same
`--w-task 2.0 --w-sport 0.5 --grl-lambda 0.3 --grl-ramp 10`, only the flags under test differ.

| arm | flags added |
|---|---|
| CTRL | none (lambda ramps 0 -> 0.3) |
| LAM | `--grl-lambda-target 0.5` |
| FULL | `--w-coral 0.5 --w-coral-centroid 0.5 --grl-lambda-target 0.5` |

The handoff's CORAL-centroid patch existed nowhere on this box, so `coral_centroid_loss`
was implemented here: pairwise squared L2 between per-sport **centroids**, applied to `z`.
The existing `coral_loss` aligns second moments only, which leaves sport decodable from
the **mean**.

## The result is a floor result, not a delta

`majority_class_share = 0.6258`. Residual decodability is `sport_acc - majority`:

| arm | sport_acc mean | sd | residual | p | 95% CI |
|---|---|---|---|---|---|
| CTRL | 0.7087 | 0.0564 | +0.0829 | 0.0304 | [+0.0128, +0.1530] |
| LAM | 0.6525 | 0.0278 | +0.0267 | 0.0987 | [-0.0079, +0.0613] |
| FULL | 0.6236 | 0.0030 | -0.0022 | 0.1817 | [-0.0060, +0.0016] |

**Under the full treatment sport is no longer decodable above the base rate**, bounded at
+0.0016. The control is decodable. That is the finding.

**LAM and FULL are not the same null.** Both fail to reject at 0.05 and reading that as
"both reached the floor" would be wrong: LAM's upper bound is +0.0613, FULL's is +0.0016,
38x tighter. LAM is UNDETERMINED, not at the floor.

**The paired mean is the wrong summary.** FULL sd 0.0030 vs CTRL sd 0.0564 — a 343x
variance ratio, F-test p=0.00005. The treatment clamps G2 to the floor; the control is
bimodal (0.661/0.666/0.676 vs 0.762/0.778) and is what varies. So the headline difference
is a fact about which controls were drawn, which is why it moved from -0.0458 at n=3 to
-0.0851 at n=5. `corr(control, difference) = -0.999` is **not** independent evidence: if
the treated arm is a constant `c` then `diff = c - ctrl` and the correlation is -1 by
arithmetic.

## Decomposition — the coral term did not confirm

| effect | mean | p | 95% CI | share |
|---|---|---|---|---|
| lambda schedule | -0.0562 | 0.0122 | [-0.0921, -0.0203] | 66% |
| coral / centroid | -0.0289 | 0.0659 | [-0.0608, +0.0030] | 34% |
| total | -0.0851 | 0.0251 | [-0.1527, -0.0174] | |

At n=3 the coral term looked real (p=0.0298). At n=5 its CI spans zero, **before** any
multiplicity correction. **Do not credit this result to the centroid loss.** The estate's
own 10-seed `no_coral` ablation agrees independently: +0.0008, p=0.9111, CI [-0.0146,
+0.0161]. (Not the same ablation — theirs removes the 2nd-moment `coral_loss`, this adds
`coral_centroid_loss` on top. Related evidence, not identical.)

## Corrections made to previously published numbers

- **The floor constant was wrong.** `2.31` is `t(0.975, df=8)` — the n=9 value — used at
  n=3, across 8 stat blocks. Correct is `t(0.975,2)=4.303`. The margin was 2.12x, not the
  published 3.9x. **No conclusion flipped.** Every constant now comes from
  `scipy.stats.t.ppf(0.975, df)`, keyed on df.
- **G1 source fields were identified, not guessed** — each is the unique candidate of four
  reproducing the artifact's recorded per-seed values. Re-runnable:
  `python pipeline/decompose_g2_ab.py --verify-metric-map --runs DIR`.

## Limits

- n=5, df=4. FULL's floor bound is tight; the coral null is absence of evidence, not proof
  of zero effect.
- Controls are bimodal, so the 3:2 basin split at n=5 is itself a small-sample lottery.
- `train_stage2.py` is reproducible — 34 of 37 numeric report fields come back
  bit-identical on rerun, and the 3 that move are one quantity (the G2 probe) by 0.0002.
  Measured effects are 425x and 145x that. `pipeline/ablation.py` is **not** reproducible
  (three runs of one config at one seed gave 0.6940 / 0.6926 / 0.6827); root cause unknown,
  adding determinism flags did **not** fix it. See `data/ablation_determinism.json`.
- The shipped model's own residual decodability is +0.0593 — it passes the shipped gate
  (`majority + 0.10 = 0.7258`) while still decodable by 37x FULL's upper bound. Passing
  that gate and being sport-blind are different properties, which
  `export_unified_stage2.py`'s own `g2_note` already warns ("a weak bar").
- The CONTROL config breaches that shipped gate on 2 of 5 seeds (0.7782, 0.7616). For the
  untreated config the gate's verdict is seed-dependent.

## One thing to weigh before deciding

`data/stage2_history.json` — the training history of the SHIPPED model — was written
2026-07-31. `pipeline/train_stage2.py` has been committed three times since:

    4cdc0db  2026-08-03  7.20: Stage 2's shippability bar was unreachable
    ecc1efa  2026-08-04  add --seed (the last trainer that could not vary its seed)
    b0a7713  2026-08-05  implement coral_centroid_loss + the GRL lambda-target schedule

So the shipped model's recorded history describes a run produced by code that is no longer
in the tree. `check_artifact_freshness.py` reports it STALE (118.4h) and that is a TRUE
positive, not a checker artifact.

It was ALREADY stale against 4cdc0db before this lane started; the last two commits are
mine and WIDENED the gap. An earlier note in this session called that failure
"pre-existing, not caused by me" — half right, and corrected here.

DELIBERATELY NOT CLEARED. The symbol-level exemption in `data/symbol_dep_registry.json`
does not legitimately apply: that mechanism is for an artifact depending on a few NAMED
SYMBOLS of a module, and this artifact is the output of the entire training procedure. The
only honest ways to clear it are to re-run training — which produces a NEW model and
discards the shipped checkpoint — or for the operator to accept it knowingly. Engineering
a green line here would be the unearned green the rest of this repo exists to prevent.

Practical consequence either way: promoting the G2 result means a retrain, which
regenerates `stage2_history.json` and clears this by itself. Discarding it leaves the entry
red until the shipped model is retrained for some other reason.
