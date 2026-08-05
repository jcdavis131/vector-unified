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

## Rescued, not fixed

Everything was copied out of temp to

    C:\Users\jcdav\experiment-rescue-2026-08-05\{hoops_ab,ab}

verified byte-size identical file by file. **This is a copy, not a repair.** The seven
scripts still point at the temp path and still break when the session closes.

## The real fix is per-repo and is the operator's

A durable workspace each repo owns — `vector-hoops/pipeline/experiments/`, gitignored for
the matrices, tracked for the analysis scripts and result JSONs. That is a decision about
what belongs in history, in two repos whose lanes are not this one, and `vector-hoops`
master is a deploy.

The analysis scripts are the part worth arguing about: they are small, they are the only
record of how several published numbers were computed, and right now they exist in exactly
one place that is scheduled for deletion.
