#!/usr/bin/env python3
"""What does a stage-2 gate metric do across seeds when NOTHING changes? Measure it.

Solo personal project, no connection to employer, built with public/free-tier only

vector-hoops carries pipeline/seed_floor.json and says, in train_mtnn.py's promotion_gate
note, exactly why:

    "this config's 8-seed mean test recall is 0.7582 (sd 0.0942), which is BELOW S2.
     Individual seeds clear it about half the time by chance, so a single run is not
     evidence the gate is met -- use evaluate_multiseed.py and judge the K-seed mean."

vector-unified has no equivalent for its stage-2 gates. data/stage2_seed_nonvacuity.json
answers a DIFFERENT question -- "does the --seed flag change anything at all", at 2-epoch
smoke, and it says so itself with SMOKE_VALUES_ARE_NOT_THE_MODEL. Nothing here records
what a full 60-epoch metric does across seeds, so nothing tells a reader whether a
single-run difference is a result or a reroll.

MEASURED, NOT ASSUMED. Reads the per-seed eval reports from an arms directory and reports,
per arm and per metric: n, mean, sd, range, and the two numbers that decide whether a
comparison means anything:

  paired_MDE     smallest difference a PAIRED comparison (same seeds, one flag changed)
                 can resolve:   t(0.975, n-1) * sd_of_differences / sqrt(n)
  unpaired_MDE   smallest difference between two INDEPENDENT single runs:
                 t(0.975, 2n-2) * sd * sqrt(2)   <- the one that applies when someone
                 eyeballs two runs from different days

HOW MUCH PAIRING BUYS IS MEASURED, NOT ASSERTED. An earlier draft of this docstring said
pairing is "roughly 7x" tighter, carried over from another experiment. On this data it is
2.7x (paired MDE 0.0677 vs unpaired 0.1841), and the gap is small for a specific reason
worth knowing: pairing cancels variance the two arms SHARE, and here the treated arm is
clamped to the majority-class floor. If the treated arm is a constant c then
diff = c - ctrl and Var(diff) = Var(ctrl) exactly, so pairing cancels nothing and only the
df changes. Measured: sd of control 0.0564, sd of differences 0.0545 -- almost no
cancellation. PAIRING STILL WINS, but quoting a remembered multiplier instead of the
measured one is the same defect this repo keeps cataloguing.

THE CONTROL ARM IS THE FLOOR. Its spread is the spread of the pipeline with no
intervention, which is what "noise" means here. The treated arm's tiny sd is NOT a floor:
it sits on the majority-class base rate and cannot vary downward, so quoting it as the
noise level would understate the floor by two orders of magnitude -- a real number
answering a different question.

    python pipeline/build_seed_floor.py --runs DIR --seeds 7,11,13,17,19
    python pipeline/build_seed_floor.py --runs DIR --check   # exit 1 if a floor is missing

Writes: data/stage2_seed_floor.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stage2_seed_floor.json"

# Reuse the identified map rather than re-deriving it; decompose_g2_ab.py's
# --verify-metric-map proves each G1 entry is the unique field reproducing the artifact.
from decompose_g2_ab import METRICS, dig  # noqa: E402

ARMS = {"ctrl": "lambda 0->0.3, no coral (NO INTERVENTION — this is the floor)",
        "lam": "lambda 0.3->0.5, no coral",
        "seed": "lambda 0.3->0.5, both coral terms"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--seeds", default="7,11,13,17,19")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the control arm has fewer than 3 seeds")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from scipy import stats

    runs = Path(args.runs)
    seeds = [int(s) for s in args.seeds.split(",")]
    reports, missing = {}, []
    for a in ARMS:
        got = {}
        for s in seeds:
            p = runs / f"{a}{s}" / "unified_report.json"
            if p.exists():
                got[s] = json.loads(p.read_text(encoding="utf-8"))
            else:
                missing.append(f"{a}{s}")
        reports[a] = got

    n_ctrl = len(reports["ctrl"])
    if args.check and n_ctrl < 3:
        print(f"FAIL: control arm has {n_ctrl} seeds; a floor from <3 is not a floor",
              file=sys.stderr)
        return 1

    out = {
        "question": "What does each stage-2 gate metric do across seeds with nothing "
                    "changed, and how big must a difference be to mean anything?",
        "why": "vector-hoops has pipeline/seed_floor.json and states that individual "
               "seeds clear its S2 gate 'about half the time by chance'. vector-unified "
               "had no equivalent for full-training stage-2 metrics. "
               "data/stage2_seed_nonvacuity.json answers whether --seed changes anything "
               "at 2-epoch smoke, and disclaims itself with SMOKE_VALUES_ARE_NOT_THE_MODEL.",
        "arms": ARMS,
        "which_arm_is_the_floor": "ctrl. Its spread is the pipeline's spread under no "
            "intervention. The treated arm's sd is NOT a floor: it sits on the "
            "majority-class base rate and cannot vary downward, so quoting it as the "
            "noise level would understate the floor by ~2 orders of magnitude.",
        "seeds_requested": seeds,
        "reports_missing": missing or "none",
        "per_arm": {},
    }

    for a, got in reports.items():
        if len(got) < 2:
            out["per_arm"][a] = {"n": len(got), "note": "fewer than 2 seeds, no spread"}
            continue
        ss = sorted(got)
        n = len(ss)
        block = {"n": n, "seeds": ss, "metrics": {}}
        for name, path in METRICS.items():
            v = [dig(got[s], path) for s in ss]
            sd = st.stdev(v)
            unp = float(stats.t.ppf(0.975, 2 * n - 2)) * sd * math.sqrt(2)
            block["metrics"][name] = {
                "values": [round(x, 4) for x in v],
                "mean": round(st.mean(v), 4), "sd": round(sd, 4),
                "range": [round(min(v), 4), round(max(v), 4)],
                "unpaired_MDE_two_single_runs": round(unp, 4),
                "note_if_degenerate": ("sd is ~0 because this arm sits on a floor; do NOT "
                                       "use this as the noise level")
                if sd < 0.005 and a != "ctrl" else None,
            }
        out["per_arm"][a] = block

    # The headline the reader needs: for the un-intervened pipeline, how big must a
    # single-run G2 difference be before it is worth discussing?
    c = out["per_arm"].get("ctrl", {}).get("metrics", {}).get("G2_sport_acc")
    if c:
        out["HEADLINE_single_run_G2_differences_below_this_are_noise"] = {
            "control_sd_over_%d_seeds" % n_ctrl: c["sd"],
            "unpaired_MDE_two_single_runs": c["unpaired_MDE_two_single_runs"],
            "reading": "Two independent single runs of the untreated config must differ "
                       "by more than %.4f on G2 sport_acc before the difference is "
                       "distinguishable from a seed swap. The control's own range across "
                       "%d seeds is %.4f to %.4f."
                       % (c["unpaired_MDE_two_single_runs"], n_ctrl,
                          c["range"][0], c["range"][1]),
            "pairing_is_cheaper_but_MEASURE_how_much": None,
        }
        # Measured, because the generic "pairing is ~7x tighter" is wrong here and the
        # reason is the finding: pairing cancels only variance the arms SHARE, and a
        # treated arm clamped to a floor shares none of the control's.
        tr = out["per_arm"].get("seed", {}).get("metrics", {}).get("G2_sport_acc")
        if tr and n_ctrl == len(tr["values"]):
            diffs = [t - c for t, c in zip(tr["values"], c["values"])]
            sd_d = st.stdev(diffs)
            pm = float(stats.t.ppf(0.975, n_ctrl - 1)) * sd_d / math.sqrt(n_ctrl)
            out["HEADLINE_single_run_G2_differences_below_this_are_noise"][
                "pairing_is_cheaper_but_MEASURE_how_much"] = {
                "paired_MDE": round(pm, 4),
                "unpaired_MDE": c["unpaired_MDE_two_single_runs"],
                "pairing_gain": f"{c['unpaired_MDE_two_single_runs'] / pm:.1f}x",
                "sd_of_control": c["sd"], "sd_of_differences": round(sd_d, 4),
                "why_the_gain_is_small_here": "Pairing cancels variance the two arms "
                    "SHARE. The treated arm is clamped to the majority-class floor, so it "
                    "shares almost none of the control's spread: sd of differences "
                    f"{sd_d:.4f} vs sd of control {c['sd']:.4f}. In the limit of a perfect "
                    "clamp, diff = constant - control and Var(diff) = Var(control), so "
                    "pairing cancels nothing and only df changes. Pairing still wins; the "
                    "multiplier is measured rather than carried over from another "
                    "experiment.",
            }

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for a, b in out["per_arm"].items():
        if "metrics" not in b:
            print(f"  {a:<5} n={b['n']}  {b['note']}")
            continue
        g = b["metrics"]["G2_sport_acc"]
        print(f"  {a:<5} n={b['n']}  G2 mean {g['mean']:.4f}  sd {g['sd']:.4f}  "
              f"range [{g['range'][0]:.4f},{g['range'][1]:.4f}]  "
              f"unpaired MDE {g['unpaired_MDE_two_single_runs']:.4f}")
    if missing:
        print(f"  missing reports (excluded, not silently dropped): {missing}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
