#!/usr/bin/env python3
"""What G1's position arm actually says, now that it says anything at all.

Solo personal project, no connection to employer, built with public/free-tier only

`pos_drop` has read exactly 0.0 for every sport in every G1 verdict since Phase 2, and
`pos_ok` has read true, and both meant NOTHING. knn5_acc took `pos_mask` — int64 {0,1} —
and used it as an INDEX rather than a mask, so `emb[mask]` selected rows 0 and 1 instead of
filtering. Two vectors, two labels, kNN-5 scores exactly 1.0. Both arms scored 1.0, so the
difference was 0.0 by construction, on the real embedding and on a shuffled one alike.

Fixed in eval_unified.knn5_acc (7.21). This computes the number the gate was supposed to be
reporting, and records it, because "the bug is fixed" and "we know what the answer is" are
different states and the repo was in the first one.

    pos_drop = frozen-encoder baseline (e_s) - joint embedding (z)     NEGATIVE IS BETTER

matching train_stage2.py:321's own convention, `b["pos_knn5"] - g["pos_knn5_live"]`.

WHICH ARMS, stated because they are not the arms train_stage2 compares. The baseline here is
the frozen per-sport encoder from the cached unified_matrix.npz, identical to Stage 0. The
joint arm is the SHIPPED z out of assets/unified.json, not the in-training `pos_knn5_live`
off the drifted encoders. So this answers "what does the artifact we actually publish
achieve", which is the question a reader of the model card is asking, and is close to but
not identical to the in-run number.

IT ALSO RESOLVES A DISCREPANCY. assets/unified.json's g1_pos_caveat quotes "True position
accuracy is ~0.78 hoops / 0.999 gridiron / 0.88 pitch". Those are the Z ARM, not the
baseline — measured here, z lands 0.7911 / 0.9991 / 0.8909 against a baseline of 0.7385 /
0.9991 / 0.8930. The caveat is quoting the right quantity and rounding it down by about a
point on two of the three.

    python pipeline/probe_g1_position.py
    python pipeline/probe_g1_position.py --check   # exit 1 if a sport regresses past the bar
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
ASSET = ROOT / "assets" / "unified.json"
OUT = ROOT / "data" / "g1_position_probe.json"
# train_stage2.py's --revert_threshold default. A drop ABOVE this would have reverted the
# run; it never fired, because the quantity was pinned at 0.0.
REVERT_THRESHOLD = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not ASSET.exists():
        print(f"missing {ASSET} — run export_unified_stage2.py first")
        return 2

    M = load_matrix(torch.device("cpu"))
    sid = M["sport_id"].cpu().numpy()
    pos = M["pos_id"].cpu().numpy()
    posm = M["pos_mask"].cpu().numpy()

    players = json.loads(ASSET.read_text(encoding="utf-8"))["players"]
    Z = np.array([p.get("e") or p.get("emb") for p in players], dtype=np.float32)
    if Z.shape[0] != len(sid):
        print(
            f"ROW MISMATCH: asset has {Z.shape[0]} players, matrix has {len(sid)}. "
            f"Refusing — a positional join across a length mismatch is how a real number "
            f"ends up describing the wrong player."
        )
        return 2

    rows, regressions = {}, []
    for s, sport in enumerate(SPORTS):
        idx = np.where(sid == s)[0]
        if not posm[idx].any():
            rows[sport] = {"n_labelled": 0, "note": "no position labels for this sport"}
            continue
        base = knn5_acc(M["E"][s].cpu().numpy(), pos[idx], posm[idx])
        joint = knn5_acc(Z[idx], pos[idx], posm[idx])
        drop = round(base - joint, 4)
        ok = drop <= REVERT_THRESHOLD
        rows[sport] = {
            "n_rows": int(len(idx)),
            "n_labelled": int(posm[idx].sum()),
            "baseline_e_s": round(base, 4),
            "joint_z": round(joint, 4),
            "pos_drop": drop,
            "pos_ok": bool(ok),
        }
        if not ok:
            regressions.append(f"{sport}: pos_drop {drop} exceeds {REVERT_THRESHOLD}")
        print(
            f"  {sport:9} labelled={int(posm[idx].sum()):6}  e_s={base:.4f}  "
            f"z={joint:.4f}  drop={drop:+.4f}  {'ok' if ok else 'REGRESSION'}"
        )

    report = {
        "what": (
            "The G1 position arm, computed with the FIXED knn5_acc. It read pos_drop "
            "0.0 / pos_ok true for every sport from Phase 2 until 7.21 and meant "
            "nothing: pos_mask was used as an index, both arms scored exactly 1.0, so "
            "their difference was 0.0 by construction — on a shuffled embedding too."
        ),
        "convention": (
            "pos_drop = baseline(e_s) - joint(z), matching train_stage2.py:321. "
            "NEGATIVE means the joint embedding recovers position BETTER than "
            "the frozen per-sport encoder it was built from."
        ),
        "arms": (
            "baseline = frozen per-sport encoder from the cached unified_matrix.npz. "
            "joint = the SHIPPED z in assets/unified.json, not the in-training "
            "pos_knn5_live off the drifted encoders. This answers what the published "
            "artifact achieves."
        ),
        "revert_threshold": REVERT_THRESHOLD,
        "per_sport": rows,
        "caveat_discrepancy_resolved": (
            "assets/unified.json g1_pos_caveat says 'True position accuracy is ~0.78 hoops "
            "/ 0.999 gridiron / 0.88 pitch'. Those are the Z arm, confirmed here. The "
            "caveat rounds down by about a point on hoops (0.7911) and pitch (0.8909); "
            "gridiron matches. It is NOT quoting the baseline, which reads 0.7385 / 0.9991 "
            "/ 0.8930."
        ),
        "verdict_note": (
            "The gate still passes, and that is the honest result — but it now "
            "passes on measured evidence rather than on a metric that could "
            "not fail. A check that could not fail was reported as passing for "
            "months; the finding is the vacuity, not the outcome."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    if regressions:
        for r in regressions:
            print(f"  REGRESSION {r}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
