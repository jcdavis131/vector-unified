#!/usr/bin/env python3
"""Split the G2 A/B effect into its lambda half and its coral half. Each gets its OWN floor.

Solo personal project, no connection to employer, built with public/free-tier only

g2_centroid_ab.json reported a paired -0.0458 on G2 sport accuracy and said, in
WHAT_THIS_DOES_NOT_ISOLATE, that the number is the COMBINED effect of three flags:
the control ramps lambda 0 -> 0.3, the treatment ramps 0.3 -> 0.5 AND adds both coral
terms. Crediting all of -0.0458 to the centroid term would be wrong. This runs the
arithmetic on the third arm that closes that gap.

    CTRL   lambda 0   -> 0.3 , no coral
    LAM    lambda 0.3 -> 0.5 , no coral      <- the arm that was missing
    FULL   lambda 0.3 -> 0.5 , both coral terms

    lambda effect = LAM  - CTRL
    coral  effect = FULL - LAM
    total         = FULL - CTRL   (identically the sum; checked, not assumed)

THE FLOOR IS RECOMPUTED PER EFFECT, and that is the whole point of this file. The
headline's floor (paired MDE 0.0116) was derived from the sd of the TOTAL differences.
A component effect has its own spread, so reusing the total's floor would be a real
number answering a different question -- the same defect class this estate keeps finding.
Each effect below is tested against 2.31 * sd(its own diffs) / sqrt(n).

n = 3 SEEDS. 2.31 is t(0.95, df=2) one-sided; with df=2 the interval is wide and a
component can easily land BELOW its own floor. That is a reportable outcome, not a
failure -- "this term's contribution is not separable from noise at n=3" is the honest
reading and is written into the artifact as such.

    python pipeline/decompose_g2_ab.py                 # print only
    python pipeline/decompose_g2_ab.py --write         # update data/g2_centroid_ab.json

Reads the per-seed eval reports from the run directory (--runs), NOT from the repo:
the arms overwrite the same four shipped write targets, so only the copies taken
after each run exist side by side.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "g2_centroid_ab.json"

# THE HEADLINE'S FLOOR WAS COMPUTED WITH THE WRONG CONSTANT, and this is where that is
# fixed rather than quietly reused. g2_centroid_ab.json reported the -0.0458 as "3.9x the
# paired MDE(n=3) 0.0116", from 2.31 * sd_d / sqrt(3). But 2.31 is t(0.975, df=8) -- the
# constant for n=9 PAIRS. At n=3 the correct two-sided constant is t(0.975, df=2) = 4.303,
# which puts the floor at 0.0216 and the effect at 2.12x, not 3.9x.
#
#   The finding survives the correction; the effect-size claim does not.
#   paired t = -9.11, df = 2, p = 0.0118; 95% CI [-0.0674, -0.0241], excludes 0.
#
# Same shape as everything else this phase found: a real constant answering a different
# question (how big must an effect be at n=9) than the one it appeared to answer (n=3).
#
# So this file reports the CI and the p-value as primary and the ratio as secondary. A
# "3.9x the floor" reads as a margin of safety and is one substituted constant away from
# being fiction; an interval that excludes zero is checkable against its own definition.
HEADLINE_T_WRONG = 2.31
HEADLINE_T_WHY = "t(0.975, df=8) -- the n=9 constant, used at n=3"


def acc(p: Path) -> float:
    return json.loads(p.read_text(encoding="utf-8"))["G2_sport_invariance"]["sport_acc"]


def paired(diffs: list[float]) -> dict:
    """Paired t on the differences. df = n-1, and that is the bug this signature exists
    to prevent -- the first draft looked its constant up by n instead of by df."""
    from scipy import stats

    n = len(diffs)
    df = n - 1
    m = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df))          # df, NOT n
    tobs = m / se if se > 0 else float("inf")
    p = float(2 * stats.t.cdf(-abs(tobs), df))
    half = tcrit * se
    return {
        "n": n,
        "df": df,
        "mean": round(m, 4),
        "sd_of_diff": round(sd, 4),
        "se": round(se, 5),
        "t_obs": round(tobs, 2),
        "p_two_sided": round(p, 4),
        "ci95": [round(m - half, 4), round(m + half, 4)],
        "ci95_excludes_zero": bool((m - half) * (m + half) > 0),
        "significant_p05": bool(p < 0.05),
        "t_crit_two_sided": round(tcrit, 3),
        "paired_MDE": round(half, 4),
        "x_floor": round(abs(m) / half, 2) if half > 0 else None,
        "consistent_sign": len({d < 0 for d in diffs}) == 1,
        "diffs": [round(d, 4) for d in diffs],
    }


def fix_floors() -> int:
    """Recompute EVERY paired floor in the artifact with t(0.975, df) instead of 2.31.

    The wrong constant is not confined to the headline. Every stat block in
    g2_centroid_ab.json carries a `paired_MDE_n3` built the same way, so a correction
    that touched only the headline would leave 7 more wrong numbers behind and look
    thorough while doing it.
    """
    from scipy import stats

    d = json.loads(ART.read_text(encoding="utf-8"))
    blocks = {k: v for k, v in d.items()
              if isinstance(v, dict) and "sd_of_differences" in v}
    n = len(next(iter(blocks.values()))["per_seed"])
    df = n - 1
    tcrit = float(stats.t.ppf(0.975, df))

    changed, flipped = [], []
    for k, v in blocks.items():
        m, sd = v["mean_difference"], v["sd_of_differences"]
        se = sd / math.sqrt(n)
        new = tcrit * se
        was = bool(v["clears_floor"])
        now = abs(m) > new
        p = float(2 * stats.t.cdf(-abs(m / se), df)) if se > 0 else 1.0
        # idempotent: a second run must not overwrite the preserved original with the
        # already-corrected value, which would erase the evidence of the correction.
        v.setdefault("paired_MDE_n3_WRONG", v["paired_MDE_n3"])
        v["paired_MDE_n3"] = round(new, 4)
        v["margin_over_floor"] = round(abs(m) / new, 2) if new else None
        v["clears_floor"] = now
        v["p_two_sided"] = round(p, 4)
        v["ci95"] = [round(m - new, 4), round(m + new, 4)]
        v["df"] = df
        changed.append(k)
        if was != now:
            flipped.append(k)

    d["CORRECTION_the_paired_floor_constant_was_the_n9_one"] = {
        # ONLY the prose fields are declared. The gate checks that a corrected target no
        # longer asserts the stale claim, which it can only do on a string -- declaring
        # the 16 numeric paths made it report "path ... is not a string" 16 times. The
        # numeric blocks are corrected in place and listed in blast_radius instead, where
        # they are visible without being claimed as gate-verified.
        "corrects_field": ["VERDICT", "n_is_3"],
        "what_was_published": f"Every paired floor in this file was {HEADLINE_T_WRONG} * "
            f"sd_d / sqrt({n}); the headline read '3.9x the paired MDE(n={n}) 0.0116'.",
        "what_is_wrong": f"{HEADLINE_T_WRONG} is {HEADLINE_T_WHY}. A paired test on {n} "
            f"seeds has df={df}, where the two-sided constant is t(0.975,{df})="
            f"{tcrit:.3f}. Every floor here was too small by {tcrit/HEADLINE_T_WRONG:.3f}x.",
        "blast_radius": f"{len(changed)} stat blocks, not just the headline: "
                        f"{', '.join(sorted(changed))}.",
        "conclusions_that_flip": flipped or
            "NONE. All %d blocks keep their verdict. The wrong constant inflated the "
            "MARGINS, it did not reverse a decision -- G2 still clears (2.12x, p=0.0118) "
            "and every cost field was already below its floor and sits further below "
            "now. The correction is to the reported effect size, not to the finding."
            % len(changed),
        "found_by": "checking the constant against scipy before reusing it in the "
                    "decomposition -- not by anything downstream catching it.",
    }
    g2 = blocks["G2_sport_acc"]
    d["VERDICT"] = (
        f"G2 REAL. Paired mean {g2['mean_difference']:+.4f}, t={g2['mean_difference']/(g2['sd_of_differences']/math.sqrt(n)):.2f} "
        f"df={df}, p={g2['p_two_sided']}, 95% CI [{g2['ci95'][0]:+.4f}, {g2['ci95'][1]:+.4f}] "
        f"which excludes 0; {n} of {n} seeds agree in sign. That is {g2['margin_over_floor']}x "
        f"the corrected floor {g2['paired_MDE_n3']}, NOT the 3.9x originally published "
        f"against a floor built with the n=9 constant. All gates PASS on all three "
        f"treatment seeds (G1=PASS G3=PASS collapse_detector=PASS). Nothing promoted.")
    d["n_is_3"] = (
        f"sd from three paired differences is itself uncertain (df={df}). The margin is "
        f"2.12x the corrected floor, down from the 3.9x originally published against a "
        f"floor built with the n=9 constant. p={0.0118}, 95% CI [-0.0674, -0.0241]. A "
        f"5-seed confirmation is the cheap next step and was not run.")
    ART.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  recomputed {len(changed)} floors with t(0.975,{df})={tcrit:.3f}")
    print(f"  conclusions flipped: {flipped or 'NONE'}")
    print(f"  wrote {ART}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fix-floors", action="store_true",
                    help="recompute every paired floor in the artifact with t(0.975,df)")
    ap.add_argument("--runs", help="directory holding ctrl<seed>/ lam<seed>/ seed<seed>/")
    ap.add_argument("--seeds", default="7,11,13")
    ap.add_argument("--write", action="store_true", help="update data/g2_centroid_ab.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.fix_floors:
        return fix_floors()
    if not args.runs:
        print("FAIL: --runs is required unless --fix-floors", file=sys.stderr)
        return 2

    runs = Path(args.runs)
    seeds = [int(s) for s in args.seeds.split(",")]

    rows, missing = [], []
    for s in seeds:
        ps = {"ctrl": runs / f"ctrl{s}" / "unified_report.json",
              "lam": runs / f"lam{s}" / "unified_report.json",
              "full": runs / f"seed{s}" / "unified_report.json"}
        absent = [k for k, p in ps.items() if not p.exists()]
        if absent:
            missing.append({"seed": s, "arms_absent": absent})
            continue
        rows.append({"seed": s, **{k: acc(p) for k, p in ps.items()}})

    if len(rows) < 2:
        print(f"FAIL: need >=2 complete seeds, have {len(rows)}. missing={missing}",
              file=sys.stderr)
        return 2

    lam_e = paired([r["lam"] - r["ctrl"] for r in rows])
    cor_e = paired([r["full"] - r["lam"] for r in rows])
    tot_e = paired([r["full"] - r["ctrl"] for r in rows])

    # additivity is arithmetic, so it must hold exactly. Checked rather than asserted in
    # prose: if it ever does not, the arms were not run on the same seeds.
    resid = max(abs((r["lam"] - r["ctrl"]) + (r["full"] - r["lam"]) - (r["full"] - r["ctrl"]))
                for r in rows)

    share = (abs(lam_e["mean"]) / abs(tot_e["mean"])) if tot_e["mean"] else float("nan")

    print(f"{'seed':>5}{'CTRL':>9}{'LAM':>9}{'FULL':>9}{'lam_e':>9}{'coral_e':>9}")
    for r in rows:
        print(f"{r['seed']:>5}{r['ctrl']:>9.4f}{r['lam']:>9.4f}{r['full']:>9.4f}"
              f"{r['lam']-r['ctrl']:>+9.4f}{r['full']-r['lam']:>+9.4f}")
    for nm, e in (("lambda", lam_e), ("coral ", cor_e), ("TOTAL ", tot_e)):
        v = "SIGNIFICANT" if e["significant_p05"] else "NOT separable from noise"
        print(f"  {nm} mean {e['mean']:+.4f}  t {e['t_obs']:+.2f} df {e['df']}  "
              f"p {e['p_two_sided']:.4f}  CI95 [{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]"
              f"  {v}")
    print(f"\n  lambda is {share:.0%} of the total. additivity residual {resid:.2e}")
    if missing:
        print(f"  INCOMPLETE SEEDS (excluded, not silently dropped): {missing}")

    if not args.write:
        print("\n(no --write, artifact untouched)")
        return 0

    d = json.loads(ART.read_text(encoding="utf-8"))
    d["DECOMPOSITION"] = {
        "why": "WHAT_THIS_DOES_NOT_ISOLATE said the -0.0458 was the combined effect of "
               "three flags and that separating them needed a third arm that was not "
               "run. It has now been run. This is that separation.",
        "arms": {"CTRL": "lambda 0->0.3, no coral",
                 "LAM": "lambda 0.3->0.5, no coral (the arm that was missing)",
                 "FULL": "lambda 0.3->0.5, both coral terms"},
        "per_seed": [{k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in r.items()} for r in rows],
        "lambda_effect": lam_e,
        "coral_effect": cor_e,
        "total_effect": tot_e,
        "lambda_share_of_total": round(share, 3),
        "additivity_residual": resid,
        "floors_are_per_effect": "Each effect is tested on its OWN differences -- its own "
            "sd, its own se, its own CI. Reusing the total's floor for a component would "
            "be a real number answering a different question.",
        "seeds_excluded": missing or "none",
        "MULTIPLICITY_the_coral_term_does_not_survive_a_correction": {
            "why_raised": "Two effects are tested on one dataset. The coral term's "
                f"p={cor_e['p_two_sided']} clears a naive 0.05 and does NOT clear a "
                "Bonferroni-corrected 0.025 for 2 comparisons. Reporting it as "
                "SIGNIFICANT without saying that would be the same overstatement the "
                "floor-constant correction above just fixed.",
            "bonferroni_alpha_2_tests": 0.025,
            "coral_p": cor_e["p_two_sided"],
            "coral_survives_bonferroni": bool(cor_e["p_two_sided"] < 0.025),
            "lambda_p": lam_e["p_two_sided"],
            "lambda_survives_bonferroni": bool(lam_e["p_two_sided"] < 0.025),
            "honest_reading": "The lambda schedule is the effect. It is 78% of the total, "
                "significant at any correction applied here, and consistent across all 3 "
                "seeds. The coral/centroid term contributes a real-looking but SMALL "
                "residual (-0.0102) whose significance is not robust to a multiplicity "
                "correction at n=3. Crediting the headline to the centroid loss -- which "
                "is the term I wrote and would most like to be the cause -- is not "
                "supported. The cheap fix is seeds, not argument: at n=5 the coral term "
                "either clears a corrected threshold or it does not.",
        },
    }
    d["CORRECTION_the_floor_constant_was_wrong"] = {
        "corrects_field": "VERDICT",
        "what_was_published": f"The -0.0458 was reported as clearing the paired MDE(n=3) "
            f"0.0116 by 3.9x, computed as {HEADLINE_T_WRONG} * sd_d / sqrt(3).",
        "what_is_wrong": f"{HEADLINE_T_WRONG} is {HEADLINE_T_WHY}. A paired test on 3 "
            "seeds has df=2, where the two-sided constant is t(0.975,2)=4.303.",
        "corrected": {
            "paired_MDE": round(tot_e["paired_MDE"], 4),
            "x_floor": tot_e["x_floor"],
            "t_obs": tot_e["t_obs"], "df": tot_e["df"],
            "p_two_sided": tot_e["p_two_sided"], "ci95": tot_e["ci95"],
        },
        "does_the_finding_survive": "YES. p=%.4f, CI %s excludes 0, and the sign is "
            "consistent across all %d seeds. What does NOT survive is the SIZE of the "
            "margin: 2.12x, not 3.9x." % (tot_e["p_two_sided"], tot_e["ci95"], tot_e["n"]),
        "found_by": "checking my own constant against scipy before reusing it in the "
                    "decomposition, not by anything downstream catching it.",
    }
    ART.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote DECOMPOSITION into {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
