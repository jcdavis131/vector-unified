# The published tennis numbers cite artifacts git does not carry

**Status: OPEN. Not repairable from this lane — see the two options at the end.**

dumbmodel.com's tennis page publishes six values that disagree with the artifacts it cites.
`check_cited_fields.py --check` reports them every run:

    tennis:insights[4]   mtnn_mean      page 0.1168   artifact 0.1157
    tennis:insights[4]   mtnn_sd        page 0.0152   artifact 0.0132
    tennis:insights[5]   ridge16_gain   page 0.0941   artifact 0.0846
    tennis:insights[5]   ridge28_gain   page 0.1012   artifact 0.0896
    tennis:insights[5]   candidates_add_over_16   page 0.0071   artifact 0.0051
    tennis:headline_stats[4]  mtnn_mean page 0.1168   artifact 0.1157

## Why this cannot be fixed by reverting a file

Both cited artifacts are **gitignored**:

    data/tennis_mtnn_report.json         UNTRACKED — no committed baseline
    data/tennis_forward_enriched.json    UNTRACKED — no committed baseline

So there is no version to restore. `git checkout` cannot bring back the numbers the page
quotes, because git never had them.

`data/tennis_forward_report.json` IS tracked and had drifted the same way
(`gain_over_rank_alone` 0.0941 -> 0.0846, `ridge16_all_features_r` 0.8427 -> 0.8332). It
was reverted rather than committed: committing a re-run's output would silently move the
baseline off the value the page cites, which converts a measurement problem into a
cosmetic one. Reverting it did **not** clear the gate, because the other two files carry
the remaining disagreements and cannot be reverted at all.

## How the artifacts moved

`validate.py` registers `tennis_mtnn` as `train_tennis_mtnn.py --check`. That arm
**retrains**. So every full gate run regenerates `tennis_mtnn_report.json`, and the value
it lands on is not the value the page was built from.

This is the same class proven elsewhere in this estate on 2026-08-05: `ablation.py` at one
fixed seed, three consecutive runs, same code — `G2_sport_acc` 0.6940 / 0.6926 / 0.6827.
Seeds are set; determinism controls are not. Evidence in `data/ablation_determinism.json`.

## Why this matters more than a stale number

The site's own fine print says **"every number is recomputable from public sources"**. For
these six, it is not:

- a reader cloning the repo gets neither artifact, so cannot check them at all;
- re-running the producer yields different values, so cannot confirm them either.

`.gitignore` already carries a negation list built for exactly this reason — the four
forward-probe reports are tracked because "dumbmodel.com CITES THEM BY PATH ... a citation
pointing at a file git never carried does not meet that claim". These two were missed by
that sweep.

But they **cannot simply be added to it**, because that list has a stated precondition:
"Safe to track because they are byte-DETERMINISTIC — fixed seeds, no timestamps; all four
re-ran identical before their lines were written. A generated file that churns each run
belongs ignored." These churn. They fail the precondition.

## The two ways out, neither taken here

1. **Make tennis training reproducible**, then track both artifacts under the existing
   rule. `train_stage2.py:166-171` shows the pattern that works — its embedding comes back
   bit-identical across reruns (34 of 37 numeric report fields, the 3 that move being one
   probe by 0.0002). Note that adding those same lines to `ablation.py` did **not** fix it
   there, so this is a real investigation, not a two-line change.
2. **Republish the page from a current run**, and accept that the numbers are a snapshot
   of one draw rather than a reproducible fact — saying so on the page.

Option 2 is cheap and honest. Option 1 is what the fine print actually promises. Choosing
between them is the operator's call: one changes a published claim, the other changes what
the site means by "recomputable".

**Do not** "fix" this by editing the page to match a fresh artifact or vice versa without
recording which was done. That would leave the gate green and the promise still broken.
