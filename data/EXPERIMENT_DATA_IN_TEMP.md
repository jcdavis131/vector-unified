# The A/B experiment record for two repos lived in a session temp directory

Seven tracked scripts in `vector-hoops` and `vector-equities` point at

    C:\Users\jcdav\AppData\Local\Temp\claude\C--Users-jcdav\be69d382-ce38-4d23-b6d1-d92c62546c02\scratchpad\hoops_ab

That id is a **Claude session**. The directory exists while the session is open, which is
exactly why nothing ever failed and nobody noticed.

## Correction: `tempfile.mkdtemp()` is NOT the fix, and an earlier commit said it was

Commit `54b283d` called this "the same defect as sweep_tennis_hparams.py" with a "known and
proven" fix. The defect is the same; **the fix is not**. `sweep_tennis_hparams.py` used its
scratch dir for a backup and a restore inside ONE run, so a fresh directory per run is
strictly safer. These scripts use it as a **shared workspace across scripts and across
runs**:

    measure_seed_floor.py:27    READS  SC/matrix_NEW.npz    a fixed input matrix
    measure_seed_floor.py:29    READS  SC/seedfloor.json    resumes from prior results
    rejudge_paired.py:25,42,43  READS  SC/ab_results.json, SC/seedfloor.json,
                                       SC/relweight04.json  — all written by OTHER scripts
                                       in EARLIER runs
    ab_feature_fill.py:66       WRITES SC/ab_results.json   which rejudge_paired.py reads

`mkdtemp` per run would hand `rejudge_paired.py` an empty directory and it would fail on the
first `open()`. Applying the "proven" fix here would have broken the chain.

## What was actually in there

41 files, 31 MB, three kinds, and two of them are irreplaceable.

**Analysis code that exists in NO repository** — checked against every `vector-*/pipeline/`:

    analyse.py  analyse2.py  byseason.py  floor_report.py  relweight.py  run_ab.py

These produced published numbers and were never version-controlled.

**`.shipped` backups of live artifacts** — the safety net an A/B run takes before it
overwrites the shipped state:

    hoops_ab/  live.json.shipped  mtnn_report.json.shipped  mtnn_centroids.npz.shipped
    ab/        mtnn_best.pt.shipped  embedding.npz.shipped  mtnn_report.json.shipped

If a run had been interrupted, these were the only copy.

**The experiment records themselves**, carrying the metrics the estate publishes:

    hoops_ab/ab_final.json      6 rows: arm, seed, cqs, recall, purity, archetype,
                                position, skill_nn
    hoops_ab/relweight04.json   8 seeds: cqs, test_recall, val_recall, purity, next_r2
    hoops_ab/floor.json         cqs, test_recall, val_recall, purity, archetype, next_r2
    ab/ab_results.json          6 rows: arm, seed, test_recall, purity20, sector_acc,
                                market_acc, next_r2_test

## Then a wider sweep of the same directory found the G2 evidence

The seven scripts pointed at two subdirectories. The session scratchpad holds 451 entries,
and most are throwaway — but five more are experiment records, and one of them backs an
**open operator decision**.

    g2run/        48 files. The 5 seeds x 3 arms behind LOCAL_GPU_G2_RESULT.md:
                  ctrl{7,11,13,17,19}/ lam{...}/ seed{...}/, each holding
                  unified_report.json, plus the training and eval logs.
    stage2_seed/  254 files
    grid_seed/    81 files
    pitch_seed/   13 files
    eq_sector/    6 files

`LOCAL_GPU_G2_RESULT.md` is tracked. **The runs behind it were not.**
`decompose_g2_ab.py` says so in its own docstring — "Reads the per-seed eval reports from
the run directory (--runs), NOT from the repo" — which is why the tool itself is portable
and carries no laptop path, and also why the evidence had nothing holding it anywhere.

Deleting that directory would not have lost a report. It would have made the G2
promote-or-discard decision unreviewable without re-running 15 GPU trainings.

Verified usable from the rescue copy, not merely copied:

    python pipeline/decompose_g2_ab.py --runs C:/Users/jcdav/experiment-rescue-2026-08-05/g2run

    lambda mean -0.0356  t -10.24 df 2  p 0.0094  CI95 [-0.0505,-0.0206]  SIGNIFICANT
    coral  mean -0.0102  t  -5.66 df 2  p 0.0298  CI95 [-0.0180,-0.0024]  SIGNIFICANT
    TOTAL  mean -0.0458  t  -9.11 df 2  p 0.0118  CI95 [-0.0674,-0.0241]  SIGNIFICANT
    lambda is 78% of the total. additivity residual 0.00e+00

## The most consequential file in there was found by accident

`g2run/backup/` is not a G2 artifact. It is the **restore set for the shipped model** —
exactly the six files `restore_shipped.py --backup DIR` declares in its MANIFEST:

    unified_stage2_best.pt        stage2_baselines.json    stage2_history.json
    unified_report.json.pre_eval  gridiron_season_emb.npz  before.json

`restore_shipped.py` exists because a restore-by-guess once invented paths and left
`unified_report.json` at the wrong value while printing RESTORED. It takes `--backup DIR`
and never says where DIR is, because the answer was a session temp directory.

So the recovery path for `sport_acc 0.6851` / ckpt `b055641c03760624` — the state the
whole G2 promote-or-discard decision is measured against — was one cleanup away from gone.
It was rescued as a side effect of copying `g2run` wholesale, before anyone knew what it
was.

Verified against the rescue copy with the tool's own check, not by eye:

    python pipeline/restore_shipped.py --backup C:/Users/jcdav/experiment-rescue-2026-08-05/g2run/backup --verify

    ok       pipeline/data/unified_stage2_best.pt    b055641c03760624
    ok       data/stage2_baselines.json              a7531f8b37271282
    ok       data/stage2_history.json                e22629381c4db5c0
    ok       data/unified_report.json                d2ee1ca2e45c6cbd
    ok       pipeline/data/gridiron_season_emb.npz   7982eab5ff00a51c
    snapshot before.json                             no destination, declared
    no drift                                         exit 0

**That is the `--backup DIR` to pass.** It was worth writing down, because the flag is
required and the docstring never named a directory.

## Rescued, not fixed

Everything was copied out of temp to `C:\Users\jcdav\experiment-rescue-2026-08-05\` —
**443 files**:

    hoops_ab/  ab/                                    41 files, byte-size verified one by one
    g2run/  stage2_seed/  grid_seed/                 402 files, verified by count, 402 of 402
    pitch_seed/  eq_sector/

**This is a copy, not a repair.** The seven
scripts still point at the temp path and still break when the session closes.

## The real fix is per-repo and is the operator's

A durable workspace each repo owns — `vector-hoops/pipeline/experiments/`, gitignored for
the matrices, tracked for the analysis scripts and result JSONs. That is a decision about
what belongs in history, in two repos whose lanes are not this one, and `vector-hoops`
master is a deploy.

The analysis scripts are the part worth arguing about: they are small, they are the only
record of how several published numbers were computed, and right now they exist in exactly
one place that is scheduled for deletion.
