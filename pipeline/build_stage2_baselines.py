#!/usr/bin/env python3
"""Stage 0 baselines on their own, so fixing a buggy artifact does not need a training run.

Solo personal project, no connection to employer, built with public/free-tier only

data/stage2_baselines.json is what Stage 2 measures ITSELF against, and it currently reads:

    hoops     pos_knn5 1.0
    gridiron  pos_knn5 1.0
    pitch     pos_knn5 1.0

That is the knn5_acc mask-used-as-an-index bug (7.21). `pos_mask` is int64 {0,1}, so
`emb[mask]` was integer fancy indexing — it selected rows 0 and 1, twice, and kNN-5 scored
exactly 1.0 for every sport, for both arms, and for a globally shuffled embedding. It is
also the direct explanation of `pos_drop = 0.0` in every G1 verdict: the baseline arm and
the model arm were both pinned at 1.0, so their difference had to be zero.

WHY A SEPARATE FILE. The computation is a kNN over the CACHED unified_matrix.npz — no
gradient, no encoder forward pass, seconds. But it lived inside train_stage2.py's training
entry point, so correcting a wrong number in a shipped artifact meant launching a Stage 2
run on a shipped model. That made a cheap fix look like an operator decision. It is not one;
it was a packaging accident.

train_stage2.py is left ALONE and still writes its own baselines at the top of a run. This
is not a second implementation of the rule — both call the same `knn5_acc` from
eval_unified.py, which is the only place the fix lives. Two copies of the CALL are fine;
two copies of the RULE would be the defect this repo keeps finding.

THE CORRECTNESS CHECK IS THE ROLE COLUMN. `role_knn5` never used a mask and was never
affected by the bug, so a standalone recomputation must reproduce the shipped role numbers
exactly. If it does, this script is computing the same quantity as the trainer and the only
thing that moved is the column that was broken. If it does not, this script is wrong and
says so instead of writing.

    python pipeline/build_stage2_baselines.py            # verify + write
    python pipeline/build_stage2_baselines.py --dry-run  # print, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_unified import knn5_acc
from train_unified import SPORTS, load_matrix

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "stage2_baselines.json"
# role_knn5 was never masked and so was never affected by the bug. These are the values the
# shipped file carries; a standalone recomputation must reproduce them.
EXPECT_ROLE = {
    "hoops": 0.8407864302235929,
    "gridiron": 0.9774647887323944,
    "pitch": 0.9609053497942387,
}
ROLE_TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = torch.device("cpu")
    M = load_matrix(device)
    sid = M["sport_id"].cpu().numpy()
    native = M["native"].cpu().numpy()
    pos = M["pos_id"].cpu().numpy()
    posm = M["pos_mask"].cpu().numpy()

    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    baselines, drift = {}, []
    for s, sport in enumerate(SPORTS):
        idx = np.where(sid == s)[0]
        e = M["E"][s].cpu().numpy()
        role = knn5_acc(e, native[idx])
        pos_acc = knn5_acc(e, pos[idx], posm[idx]) if posm[idx].any() else None
        baselines[sport] = {"n": int(len(idx)), "role_knn5": role, "pos_knn5": pos_acc}
        exp = EXPECT_ROLE.get(sport)
        ok = exp is not None and role is not None and abs(role - exp) <= ROLE_TOL
        if not ok:
            drift.append(f"{sport}: role_knn5 {role} != shipped {exp}")
        was = (prev.get(sport) or {}).get("pos_knn5")
        print(f"  {sport:9} n={len(idx):6}  role={role:.10f} {'OK' if ok else 'DRIFT'}" f"   pos {was} -> {pos_acc}")

    if drift:
        print("\nROLE COLUMN DID NOT REPRODUCE — refusing to write:")
        for d in drift:
            print(f"  {d}")
        print(
            "role_knn5 was never masked and so was never touched by the 7.21 fix. If it "
            "moved, this script is measuring something other than what the trainer "
            "measured, and its pos_knn5 cannot be trusted either."
        )
        return 1

    print(
        "\nrole column reproduces the shipped values exactly, so this computes the same "
        "quantity the trainer does; only the masked column moved."
    )
    if args.dry_run:
        print("--dry-run: nothing written")
        return 0
    OUT.write_text(json.dumps(baselines, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
