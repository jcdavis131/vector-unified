# LOCAL GPU lane — BLOCKED, with evidence

**Agent:** Claude-Local (local box, RTX 4080, torch+CUDA working)
**Lane:** LOCAL_GPU_HANDOFF.md task 1 — vector-unified G2 0.6851 -> 0.64
**Status:** BLOCKED. Not started. Nothing edited in the trainers.

## Why

`LOCAL_GPU_HANDOFF.md` says the work is already patched in Hatch and this box only has to
run it. **Those patches are not reachable from here.** Checked before claiming, not after:

| thing the handoff's commands need | local master | origin/master |
|---|---|---|
| `coral_centroid_loss` in train_unified.py | absent (0 hits) | absent |
| `coral_centroid_loss` in train_stage2.py | absent (0 hits) | absent |
| `--grl-lambda-target` | absent | absent |
| `--w-coral-centroid` | absent | absent |
| `--w-coral` (train_stage2) | absent | absent |

`git branch -r` lists exactly one remote branch, `origin/master`. There is no
`scout/unified-g2-blind` branch on the remote to pull the work from.

`data/unified_report.json` also does not match the handoff's description: it has **no**
experimental/projection block, and `G2_sport_invariance.sport_acc` is **0.6851** — the
baseline. The predicted 0.642 / delta -0.043 is not in the file.

So the commands in section 1 of the handoff would run the CURRENT trainers with flags they
do not accept. That is not "the local box finishing the job", it is a different experiment.

## What is NOT the blocker

The caches the handoff worried about are fine on this box:

- `vector-hoops/pipeline/data/embedding_v3.npz` — present, and this session verified it
  matches what `unified_matrix.npz` was built from (cosine 1.0000 over 2,000 rows)
- `vector-pitch/assets/pitch_mtnn_embeddings.json` — present
- torch + CUDA — working; `train_stage2.py --smoke` has been run end-to-end on this box
  today (2 epochs, 2 seeds, backup/restore harness)

GPU and data are ready. The **code** is what is missing.

## To unblock

Push `scout/unified-g2-blind` (or whatever branch holds the coral-centroid work) to
`origin`, and this lane can run the same hour. Until then the honest status is `blocked`,
not `in-progress`.

## Meanwhile

Taking `vector-pitch` from the FREE lanes list instead. Logged rather than left silent.
