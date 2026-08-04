#!/usr/bin/env python3
"""Does the equities embedding forecast next year, or only describe this one? (7.35)

Solo personal project, no connection to employer, built with public/free-tier only

The third of four forward probes — gridiron came later and also answered no. Tennis found
style adds +0.0941 over rank persistence; hoops found the skill profile adds +0.0625 over
impact persistence. Asked of equities, the same question returns essentially nothing, and
that is the result rather than a failure of the run.

    Profitability     persistence 0.8473   +64-d embedding 0.8516   gain +0.0043
    Balance_Health    persistence 0.9151   +64-d embedding 0.9157   gain +0.0006
    Market_Momentum   persistence 0.8477   +64-d embedding 0.8506   gain +0.0029

THE GAINS ARE REAL, CONSISTENT, AND TOO SMALL TO MATTER — three separate claims, and the
middle one was only established after this file had already shipped twice. Every gain beats
the shuffled-extras null (p = 0.000 / 0.025 / 0.000), so this is not noise; every gain is
positive at all four cut years, so it is not an artefact of where the split fell; and every
gain sits below the 0.01 bar fixed before the run.

    Profitability    2019:+0.0039  2020:+0.0028  2021:+0.0043  2022:+0.0068
    Balance_Health   2019:+0.0008  2020:+0.0008  2021:+0.0006  2022:+0.0016
    Market_Momentum  2019:+0.0025  2020:+0.0043  2021:+0.0029  2022:+0.0016

Quoting only the p-values would turn a null result into a headline; quoting only "no gain"
would overstate it the other way.

THE SWEEP WAS MISSING HERE UNTIL GRIDIRON EXPOSED WHY IT MATTERS. Tennis and hoops required
gain_positive_at_every_cut from the start. This file did not, and nothing false shipped only
because every gain sits below the bar regardless — the verdict was one number away from
resting on a split nobody had varied. build_gridiron_forward.py's first run made that
concrete: it reported TE as an earner at +0.0105 on a single cut, a gain that turns -0.0022
when the boundary moves to 2021. Three of four probes demanding cross-cut stability while
the fourth did not was a fact about my process, not about equities.

THE EMBEDDING IS NOT UNINFORMATIVE — IT IS REDUNDANT, and the difference matters. Scored on
its own, with the company's current score withheld entirely, the 64-d vector recovers next
year's skill about as well as carrying this year's number forward (0.8511 / 0.9141 / 0.8502
against 0.8473 / 0.9151 / 0.8477). So it encodes the present profile well. It just does not
know anything ABOUT NEXT YEAR that the present profile does not already say.

READ THE BASELINE BEFORE READING THE GAIN. Persistence here is 0.85 to 0.92. There is very
little headroom for ANY model, and "gain ~0 against a 0.92 baseline" is a much weaker claim
than "the model is bad". A quality composite that moved enough to leave room would be a
suspicious quality composite.

IT IS NOT CARRY-FORWARD, checked before drawing any conclusion, because persistence that
high is exactly what a stale-data bug looks like:
    51,972 year-over-year deltas, 71 exactly zero (0.14%)
    ZERO rows identical to their prior year
    median |delta| ~0.05 against a value sd of ~0.21
The numbers genuinely move. These composites are simply slow.

WHAT IS AND IS NOT BEING TESTED, corrected twice, because the first correction repeated the
false premise it was correcting. The original version of this file said dumbmodel.com's
"next-year profile" claim COULD NOT BE CONFIRMED, on the grounds that mtnn_report.json
records only recall@10, archetype purity and sector accuracy. The first correction kept
that sentence and called it "true of the report, unfair as an insinuation".

IT WAS NOT TRUE OF THE REPORT. mtnn_report.json has a top-level next_profile block:

    next_profile.val   rows 990  r2 0.262   mae_z 0.2516  rmse 0.525
    next_profile.test  rows 500  r2 0.1965  mae_z 0.3596  rmse 0.6511

and the head that produces it is in the source:

    model.py:274        self.next_profile_head = head(n_game)
    train_mtnn.py:42    "next_profile": 0.10                       # loss weight
    train_mtnn.py:456   smooth_l1_loss(out_a["next_profile"][valid_t], game_z[nxt_t])

I NEVER OPENED THE FILE. Both the claim and its first correction were written from a
remembered summary of what the report contained, treated as a verified fact because it was
my own earlier sentence. That is this repo's defect class turned on itself: a real value —
the summary — answering a different question than the one it appeared to answer.

The evidence was already in my own published work. dumbmodel.com's equities page has
carried insights[4], "Next-year prediction is weak, and the model's own report says so",
citing `pipeline/data/mtnn_report.json -> next_profile.val, next_profile.test` since the
page was built and adversarially verified. I contradicted a live, checked page of my own
while auditing that same site.

SCOPE, stated correctly. This probe tests the SHIPPED EMBEDDING — the trunk output in
assets/real_data.json, what dumbmodel.com serves and what any downstream consumer gets —
under a linear read. IT DOES NOT TEST THE next_profile HEAD, whose weights are not in the
shipped asset.

AND THE TWO NUMBERS ARE NOT COMPARABLE, which is worth stating because the temptation to
line them up is obvious. The head reports r2 0.1965 on 500 test rows; persistence here
reaches r 0.85-0.92, and r2 ~0.72 would look like a rout. They are not measured on the same
quantity: the head is scored on the full z-scored next-year profile vector across all skill
dimensions, this file is scored per-skill on three named composites, over different rows
and a different split. Putting them side by side would be exactly the cross-metric
apples-to-oranges this repo keeps refusing to make.

    python pipeline/build_equities_forward.py
    python pipeline/build_equities_forward.py --check   # exit 1 only if the run is broken
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tennis_forward import null_extras_gain, r, ridge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = Path("C:/Users/jcdav/vector-equities/assets/real_data.json")
OUT = ROOT / "data" / "equities_forward_report.json"
CUT_YEAR = 2021
TARGETS = ("Profitability", "Balance_Health", "Market_Momentum")
EARNS = 0.01     # a gain below this is not worth calling a gain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not REAL.exists():
        print(f"missing {REAL}")
        return 2
    e = json.loads(REAL.read_text(encoding="utf-8"))
    pts, keys = e["points"], e["skill_keys"]
    S = np.array([p["skills"] for p in pts], dtype=np.float32)
    E = np.array([p["emb"] for p in pts], dtype=np.float32)
    if S.shape[0] != E.shape[0] or S.shape[1] != len(keys):
        print("shape mismatch between points, skills and skill_keys — refusing")
        return 2

    idx = {(x["ticker"], int(x["year"])): i for i, x in enumerate(pts)}
    prs = [(idx[(t, y)], idx[(t, y + 1)], y + 1) for (t, y) in idx if (t, y + 1) in idx]
    src = np.array([i for i, _, _ in prs])
    dst = np.array([k for _, k, _ in prs])
    ty = np.array([y for _, _, y in prs])
    tr, te = ty <= CUT_YEAR, ty > CUT_YEAR

    # Carry-forward check. Persistence of 0.9 is what a stale-data bug looks like, so this
    # runs before anything is concluded from the persistence number.
    D = S[dst] - S[src]
    exact_zero = int((D == 0).sum())
    identical_rows = int((np.abs(D).max(axis=1) < 1e-6).sum())

    print(f"{len(prs)} consecutive-year pairs   train {tr.sum()}   test {te.sum()}")
    print(f"  carry-forward check: {exact_zero}/{D.size} deltas exactly zero "
          f"({100*exact_zero/D.size:.2f}%), {identical_rows} identical rows")
    print(f"\n  {'skill':22} {'persist':>8} {'emb ONLY':>9} {'score+emb':>10} {'gain':>8}")

    rows, any_earns = [], False
    for tgt in TARGETS:
        j = keys.index(tgt)
        yp, yn = S[src, j], S[dst, j]
        F = np.hstack([S[src, j:j + 1], E[src]])
        persist = r(yp[te], yn[te])
        only = r(ridge(E[src][tr], yn[tr], E[src][te]), yn[te])
        r1 = r(ridge(yp[tr, None], yn[tr], yp[te, None]), yn[te])
        rb = r(ridge(F[tr], yn[tr], F[te]), yn[te])
        gain = rb - r1
        ndist = null_extras_gain(F, 0, yp, yn, tr, te, reps=40)
        p_val = float((ndist >= gain).mean())

        # CUT-YEAR SWEEP, ADDED LAST — this file was the only one of the four forward
        # probes without one. Tennis and hoops required gain_positive_at_every_cut from the
        # start; gridiron omitted it and its first run reported TE as an earner at +0.0105
        # on a single split, a gain that turns -0.0022 when the boundary moves to 2021.
        # Nothing false shipped here because every equities gain sits below the 0.01 bar
        # anyway — but the verdict was one number away from depending on a split nobody had
        # varied, and three of four probes demanding cross-cut stability while the fourth
        # did not is a fact about my process, not about equities.
        sweep = []
        for cut in (2019, 2020, 2021, 2022):
            a_tr, a_te = ty <= cut, ty > cut
            if a_te.sum() < 200 or a_tr.sum() < 200:
                continue
            g1 = r(ridge(yp[a_tr, None], yn[a_tr], yp[a_te, None]), yn[a_te])
            gb = r(ridge(F[a_tr], yn[a_tr], F[a_te]), yn[a_te])
            sweep.append({"cut_year": cut, "n_test": int(a_te.sum()),
                          "score_only_r": round(g1, 4), "score_plus_embedding_r": round(gb, 4),
                          "gain": round(gb - g1, 4)})
        cg = [s_["gain"] for s_ in sweep]
        all_pos = bool(cg) and all(g > 0 for g in cg)

        earns = gain > EARNS and p_val < 0.05 and all_pos
        any_earns = any_earns or earns
        rows.append({"skill": tgt, "persistence_r": round(persist, 4),
                     "embedding_only_r": round(only, 4), "score_only_r": round(r1, 4),
                     "score_plus_embedding_r": round(rb, 4), "gain": round(gain, 4),
                     "null_p": p_val, "cut_year_sweep": sweep,
                     "gain_positive_at_every_cut": all_pos,
                     "gain_mean_across_cuts": round(float(np.mean(cg)), 4) if cg else None,
                     "earns_its_keep": bool(earns)})
        print(f"  {tgt:22} {persist:>8.4f} {only:>9.4f} {rb:>10.4f} {gain:>+8.4f}"
              f"   p={p_val:.3f} {'EARNS' if earns else 'no'}")
        if sweep:
            print(f"  {'':22} cuts: " + "  ".join(
                f"{c['cut_year']}:{c['gain']:+.4f}" for c in sweep)
                + ("  all positive" if all_pos else "  NOT positive at every cut"))

    print(f"\n  verdict: {'some targets earn it' if any_earns else 'NO on MAGNITUDE, not on significance — every gain beats the null but all sit below the 0.01 bar'}")

    OUT.write_text(json.dumps({
        "question": ("Does the shipped 64-d equities embedding predict next year's skill "
                     "profile beyond what this year's own score already predicts?"),
        "verdict": ("NO, ON MAGNITUDE — NOT ON SIGNIFICANCE. Gains of +0.0043 / +0.0006 / "
                    "+0.0029 against persistence of 0.8473 / 0.9151 / 0.8477. Every one of "
                    "them BEATS the shuffled-extras null (p = 0.000 / 0.025 / 0.000), so the "
                    "embedding does carry a real, detectable increment. It is just far too "
                    "small to matter: below the 0.01 bar this file set before running. "
                    "Statistically distinguishable and practically negligible are different "
                    "findings, and reporting only the p-value would turn a null result into "
                    "a headline." if not any_earns else "some targets earn it"),
        "redundant_not_uninformative": (
            "Scored ALONE, with the current score withheld, the embedding recovers next "
            "year's skill about as well as persistence does. It encodes the present profile "
            "well; it just knows nothing about next year that the present does not say. "
            "'Redundant' and 'uninformative' are different findings and this is the first."),
        "read_the_baseline_first": (
            "Persistence is 0.85-0.92 here. There is very little headroom for any model, and "
            "'gain ~0 against a 0.92 baseline' is a far weaker claim than 'the model is "
            "bad'. A quality composite that moved enough to leave room would be suspicious."),
        "carry_forward_check": {
            "deltas": int(D.size), "exactly_zero": exact_zero,
            "pct_exactly_zero": round(100 * exact_zero / D.size, 2),
            "rows_identical_to_prior_year": identical_rows,
            "why": ("Persistence of 0.9 is what a stale-data bug looks like, so this ran "
                    "before any conclusion was drawn from it. The values genuinely move."),
        },
        "scope_corrected": (
            "Corrected TWICE; the first correction repeated the false premise. This file "
            "claimed mtnn_report.json records no next-year head. It has a top-level "
            "next_profile block (val r2 0.262 over 990 rows, test r2 0.1965 over 500), and "
            "model.py:274 / train_mtnn.py:456 define and train the head at loss weight "
            "0.10. I never opened the report — both the claim and its first correction came "
            "from a remembered summary treated as verified because it was my own earlier "
            "sentence. dumbmodel.com's equities insights[4] has cited those exact fields "
            "since the page was built, so I contradicted my own live verified page."),
        "scope_of_this_probe": (
            "Tests the SHIPPED EMBEDDING — the trunk output in real_data.json, what the "
            "site serves — under a linear read. Does NOT test the next_profile head, whose "
            "weights are not in the shipped asset. The head's r2 0.1965 and this file's "
            "persistence r 0.85-0.92 are NOT comparable: the head is scored on the full "
            "z-scored profile vector across all skill dims, this is scored per-skill on "
            "three named composites, over different rows and a different split. Lining "
            "them up would be the cross-metric apples-to-oranges this repo keeps refusing."),
        "n_pairs": len(prs), "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "split": f"TEMPORAL — train on target year <= {CUT_YEAR}, test strictly after",
        "per_target": rows,
        "sweep_added_late": (
            "Tennis and hoops required gain_positive_at_every_cut from the start; this file "
            "did not. Nothing false shipped, because every equities gain sits below the 0.01 "
            "bar regardless — but the verdict was one number away from resting on a split "
            "nobody had varied. build_gridiron_forward.py's first run made that concrete: TE "
            "reported as an earner at +0.0105 on one cut, -0.0022 when the boundary moved. "
            "All three equities gains ARE positive at every cut, so the finding is unchanged "
            "and now says more: real, consistent, and too small to matter."),
        "vs_other_sports": (
            "tennis +0.0941 over a 0.7486 baseline, hoops +0.0625 over 0.4514, equities ~0 "
            "over 0.85-0.92, gridiron 0 of 4 positions over 0.49-0.77. The GAINS are not "
            "directly comparable — different targets, different baselines, different domains "
            "— but the pattern that headroom tracks baseline is worth noticing rather than "
            "reading as a ranking of the models."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

    # ---- --check HAD NO BODY, AND THIS FILE WAS REGISTERED AS A GATE ------------
    # validate.py ran `build_equities_forward.py --check` as the `equities_forward` check
    # while main() never read args.check. It could not fail under any input. The docstring
    # promised "exit 1 only if the run is broken" — a documented promise the code did not
    # keep — and it sat in the gate reporting PASS for three commits, including the one
    # that derived "one mutation per arm, not per file" from a different vacuous arm.
    #
    # WHAT "BROKEN" MEANS HERE is not "the answer is negative". The negative result IS the
    # finding, so gating on p >= 0.05 (as tennis and hoops do) would be asserting the
    # opposite of what this file concluded. It means the measurement cannot be trusted:
    #
    #   CARRY-FORWARD   persistence of 0.85-0.92 is exactly what stale data looks like. The
    #                   run already computes identical_rows and then only PRINTS it. If the
    #                   composites ever start carrying forward, every number here becomes an
    #                   artifact of duplication and the file would still exit 0 and publish.
    #   DEGENERATE NULL a null with no spread makes every p-value meaningless. This repo
    #                   already shipped one degenerate null (permuted-target, sd 0.1487) and
    #                   the lesson was to measure the null's spread, not assume it.
    fails = []
    frac_identical = identical_rows / max(1, len(prs))
    if frac_identical > 0.01:
        fails.append(f"CARRY-FORWARD: {identical_rows}/{len(prs)} pairs ({100*frac_identical:.1f}%) "
                     f"are identical to their prior year — persistence of "
                     f"{rows[0]['persistence_r']} would be measuring duplication, not skill")
    null_sd = float(ndist.std())
    if null_sd < 1e-6:
        fails.append(f"DEGENERATE NULL: shuffled-extras sd is {null_sd:.2e}; every p-value "
                     f"in this report is uninterpretable")

    if args.check and fails:
        print()
        for f_ in fails:
            print(f"FAIL {f_}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
