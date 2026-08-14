"""Does the G1 baseline measure the same thing G1 measures?

G1 is `baseline - live`, and the two halves come from different objects:

    baseline   train_stage2.py:243   e_froz = M["E"][s]
               a STORED per-sport embedding, read off the unified matrix

    live       train_stage2.py:68    live[sport].encode_full_numpy(device)
               RECOMPUTED by loading the sport's encoder checkpoint and running
               its features through it

Those are only comparable if the stored matrix was produced by the same
checkpoint the live loader picks up. Nothing enforces that, and the training
log says something is wrong:

    Stage 0 baseline   gridiron role = 0.9775
    epoch 1 [warmup]   gridiron role = 0.725

A 0.25 drop in kNN-5 accuracy after ONE epoch at --enc-lr 1e-5, during warmup,
with SupCon and the sport adversary both switched off. An encoder cannot move
that far that fast. Pitch, on the same step, does not move at all (0.9609 ->
0.963) and hoops moves 0.039. If this were drift it would be small and it would
grow; instead it appears instantly and is wildly sport-specific.

The hypothesis is that gridiron's stored embedding and its live encoder are not
the same representation, in which case G1's gridiron regression is measuring a
mismatch between two artifacts rather than damage done by stage 2 -- and the
0.14-0.15 that has survived every configuration tried is mostly an offset that
was never going to move.

This takes NO optimiser step. It loads exactly what stage 2 loads, scores both
halves of the comparison, and prints them side by side.

    python pipeline/probe_g1_baseline.py
"""

from __future__ import annotations

import numpy as np
import torch

from eval_unified import knn5_acc
from load_live_encoders import DEVICE_DEF, load_live
from train_unified import SPORTS, load_matrix


def main() -> int:
    device = torch.device(DEVICE_DEF)
    M = load_matrix(device)
    live = load_live(device)

    sid = M["sport_id"].cpu().numpy()
    native = M["native"].cpu().numpy()
    pos = M["pos_id"].cpu().numpy()
    posm = M["pos_mask"].cpu().numpy()

    print(f"{'sport':<10} {'n':>7} {'stored':>9} {'live':>9} {'delta':>9} "
          f"{'dim':>12} {'cos':>7}")
    for s, sport in enumerate(SPORTS):
        idx = np.where(sid == s)[0]
        e_stored = M["E"][s].cpu().numpy()
        e_live = live[sport].encode_full_numpy(device)

        r_stored = knn5_acc(e_stored, native[idx])
        r_live = knn5_acc(e_live, native[idx])

        # Shape first: a dimension mismatch is a different failure from a
        # values mismatch, and only one of them is fixable by re-exporting.
        dim = f"{e_stored.shape[1]} vs {e_live.shape[1]}"
        if e_stored.shape == e_live.shape:
            a = e_stored / (np.linalg.norm(e_stored, axis=1, keepdims=True) + 1e-9)
            b = e_live / (np.linalg.norm(e_live, axis=1, keepdims=True) + 1e-9)
            cos = float(np.mean(np.sum(a * b, axis=1)))
        else:
            cos = float("nan")
        print(f"{sport:<10} {len(idx):>7} {r_stored:>9.4f} {r_live:>9.4f} "
              f"{r_stored - r_live:>+9.4f} {dim:>12} {cos:>7.3f}")

        # Position too -- gridiron's pos baseline is 0.999, which is the kind of
        # number that usually means the label is recoverable from the artifact
        # rather than that the model is excellent.
        if posm[idx].any():
            p_stored = knn5_acc(e_stored, pos[idx], posm[idx])
            p_live = knn5_acc(e_live, pos[idx], posm[idx])
            print(f"{'':<10} {'pos':>7} {p_stored:>9.4f} {p_live:>9.4f} "
                  f"{p_stored - p_live:>+9.4f}")

    print("\nA nonzero delta here is present BEFORE any training step, so it is")
    print("baked into every G1 verdict this model has ever reported. Compare it")
    print("to the measured role_drop at the best epoch: gridiron +0.1502.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
