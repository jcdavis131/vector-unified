#!/usr/bin/env python3
"""Do the 12 new features help PREDICT next year's rank, or only identify the player?

Solo personal project, no connection to employer, built with public/free-tier only

4821c78 validated 12 schedule/shot candidates on RETRIEVAL — finding the same player's
adjacent year — and 47edd8e built an MTNN on them. Retrieval is an identity task. Whether
those same features help FORECASTING is a different question, and assuming the answer
carries over is exactly the kind of transfer this repo keeps catching.

BUILT AS A SEPARATE PROBE ON PURPOSE. build_tennis_forward.py is a registered gate check and
its numbers are published on dumbmodel.com's tennis card (+0.0941 style over rank
persistence, 0.7486 baseline). Changing its feature set would silently move a live figure.
This asks the question beside it and changes nothing.

    BASELINE     next ENTERING_RANK_LOG = this year's                   persistence
    RIDGE-1      ridge on rank alone
    RIDGE-16     rank + the original 15, value AND mask
    RIDGE-28     rank + all 27 others, value AND mask

The null, the temporal split and the estimator are IMPORTED from build_tennis_forward.py,
not re-implemented — one definition of the shuffled-extras null, one ridge, one r().

    python pipeline/probe_tennis_forward_enriched.py
    python pipeline/probe_tennis_forward_enriched.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from build_tennis_forward import null_extras_gain, r, ridge  # noqa: E402

MATRIX = ROOT / "pipeline" / "data" / "tennis_matrix.npz"
META = ROOT / "pipeline" / "data" / "meta_tennis_matrix.json"
OUT = ROOT / "data" / "tennis_forward_enriched.json"
CUT_YEAR = 2022


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from probe_tennis_candidate_features import CANDIDATES, build as build_cand

    a = np.load(MATRIX, allow_pickle=True)
    X, M = a["X"].astype(np.float64), a["M"].astype(np.float64)
    feats = [str(f) for f in a["features"]]
    meta = json.loads(META.read_text(encoding="utf-8"))
    if len(meta) != X.shape[0]:
        print(f"meta {len(meta)} vs matrix {X.shape[0]} — refusing positional join")
        return 2

    cand = build_cand()
    keys = [(m["player"], m["year"], m["tour"]) for m in meta]
    C = np.full((len(keys), len(CANDIDATES)), np.nan)
    for i, k in enumerate(keys):
        v = cand.get(k)
        if v:
            for c, nm in enumerate(CANDIDATES):
                if v.get(nm) is not None:
                    C[i, c] = v[nm]
    CM = (~np.isnan(C)).astype(float)
    C = np.nan_to_num(C)

    rank_j = feats.index("ENTERING_RANK_LOG")
    idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
    pairs = [(i, idx[(m["player"], m["year"] + 1, m["tour"])], m["year"] + 1)
             for i, m in enumerate(meta)
             if (m["player"], m["year"] + 1, m["tour"]) in idx]
    pairs = [(i, k, y) for i, k, y in pairs if M[i, rank_j] == 1 and M[k, rank_j] == 1]

    src = np.array([i for i, _, _ in pairs])
    dst = np.array([k for _, k, _ in pairs])
    ty = np.array([y for _, _, y in pairs])
    y_prev, y_next = X[src, rank_j], X[dst, rank_j]
    tr, te = ty <= CUT_YEAR, ty > CUT_YEAR

    F16 = np.hstack([X, M])[src]
    F28 = np.hstack([X, C, M, CM])[src]
    # rank sits at column rank_j in both blocks — the value half is laid out identically
    print(f"{len(pairs)} pairs   train(target<= {CUT_YEAR}) {tr.sum()}   test {te.sum()}")
    print(f"  F16 {F16.shape[1]} cols   F28 {F28.shape[1]} cols\n")
    if te.sum() < 100:
        print("test split too small to interpret")
        return 2

    persistence = r(y_prev[te], y_next[te])
    r1 = r(ridge(y_prev[tr, None], y_next[tr], y_prev[te, None]), y_next[te])
    r16 = r(ridge(F16[tr], y_next[tr], F16[te]), y_next[te])
    r28 = r(ridge(F28[tr], y_next[tr], F28[te]), y_next[te])

    n16 = null_extras_gain(F16, rank_j, y_prev, y_next, tr, te)
    n28 = null_extras_gain(F28, rank_j, y_prev, y_next, tr, te)
    p16 = float((n16 >= (r16 - r1)).mean())
    p28 = float((n28 >= (r28 - r1)).mean())

    print(f"  persistence (this rank -> next)   r = {persistence:.4f}")
    print(f"  RIDGE-1   rank only               r = {r1:.4f}")
    print(f"  RIDGE-16  original features       r = {r16:.4f}   gain {r16-r1:+.4f}  "
          f"p={p16:.3f}")
    print(f"  RIDGE-28  + 12 candidates         r = {r28:.4f}   gain {r28-r1:+.4f}  "
          f"p={p28:.3f}")
    print(f"\n  the 12 candidates add {r28-r16:+.4f} over the original 16")

    helps = (r28 - r16) > 0.01 and p28 < 0.05
    verdict = (
        f"THE CANDIDATES HELP FORECASTING TOO ({r28-r16:+.4f} over RIDGE-16)" if helps else
        f"IDENTITY ONLY. The 12 features that lifted retrieval 0.0447 -> 0.0584 add "
        f"{r28-r16:+.4f} to rank forecasting — below the 0.01 bar. Knowing WHO a player is "
        f"and knowing WHERE THEY WILL FINISH are different problems, and features that "
        f"solve the first need not touch the second.")
    print(f"\n  verdict: {verdict}")

    OUT.write_text(json.dumps({
        "question": ("Do the 12 schedule/shot candidates validated on RETRIEVAL also "
                     "improve rank FORECASTING?"),
        "why_separate_file": ("build_tennis_forward.py is a registered gate check and its "
                             "numbers are published on dumbmodel.com's tennis card. "
                             "Changing its feature set would silently move a live figure."),
        "persistence_r": round(persistence, 4),
        "ridge1_rank_only_r": round(r1, 4),
        "ridge16_r": round(r16, 4), "ridge16_gain": round(r16 - r1, 4), "ridge16_null_p": p16,
        "ridge28_r": round(r28, 4), "ridge28_gain": round(r28 - r1, 4), "ridge28_null_p": p28,
        "candidates_add_over_16": round(r28 - r16, 4),
        "retrieval_contrast": ("The same 12 features lifted retrieval from 0.0447 to 0.0584 "
                               "(learned linear) and the MTNN to 0.0783. Retrieval is an "
                               "identity task; this is a forecasting task."),
        "n_pairs": len(pairs), "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "split": f"TEMPORAL — train on target year <= {CUT_YEAR}, test strictly after",
        "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    if args.check and te.sum() < 100:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
