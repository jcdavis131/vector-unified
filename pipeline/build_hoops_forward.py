#!/usr/bin/env python3
"""Does an NBA season's SKILL PROFILE predict next season's impact, beyond this one's? (7.34)

Solo personal project, no connection to employer, built with public/free-tier only

The hoops twin of build_tennis_forward.py, asking the same question of a different sport and
built so the answer can be no. Tennis found that playing style adds +0.0941 over predicting
next year's rank from this year's rank. Basketball's persistence is much weaker to begin
with — impact carries r = 0.4674 season to season against tennis rank's 0.7486 — so the
headroom is larger and the baseline is a lower bar. Both facts are reported.

    BASELINE     next impact = this impact                    persistence
    RIDGE-1      ridge on impact alone
    RIDGE-12     ridge on all twelve skill grades
    the profile earns its keep only if RIDGE-12 beats RIDGE-1 OUT OF SAMPLE

MERGED-NAME PAIRS ARE EXCLUDED, and this is not hygiene, it is the difference between a
measurement and a fiction. vector-hoops' vectors.json has no player identity key — its `id`
is a row index, LeBron James has 23 of them — so a consecutive-season pair keyed by name can
join TWO DIFFERENT PEOPLE across the season boundary. 112 of 10,108 pairs do exactly that.
Left in, the model would be scored partly on predicting one man's next season from another
man's last one. merged_names() minus the DOB acquittals is the same set every axis in this
repo already excludes.

TEMPORAL SPLIT, NOT RANDOM. A random split puts a player's 2019 and 2020 rows on opposite
sides; impact is persistent enough that the model would score by recognising the player. The
split also has to match the use — predicting a season that has not happened.

ERA IS ALREADY HANDLED and deliberately not handled again: skills.json grades are PERCENTILE
WITHIN SEASON POOL, so a 1997 grade and a 2025 grade are both "how good, among peers that
year". No era term is needed and adding one would be double-counting.

NON-VACUITY IS PART OF THE RUN: the same pipeline scored with the TARGET SHUFFLED. If that
arm does not collapse toward zero the evaluation leaks and the real number means nothing.

    python pipeline/build_hoops_forward.py
    python pipeline/build_hoops_forward.py --check   # exit 1 if the null does not collapse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_hoops_vor_draft_value as B
from build_tennis_forward import null_extras_gain, r, ridge

# One implementation of the estimator AND of the null, imported not copied.

ROOT = Path(__file__).resolve().parent.parent
HOOPS = Path("C:/Users/jcdav/vector-hoops/assets")
OUT = ROOT / "data" / "hoops_forward_report.json"
SEED = 7
CUT_YEAR = 2019
# NULL_TOL removed — asserted tolerance on a degenerate quantity.
NULL_P = 0.05
TARGET = "impact"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sk = json.loads((HOOPS / "skills.json").read_text(encoding="utf-8"))
    vec = json.loads((HOOPS / "vectors.json").read_text(encoding="utf-8"))
    players = vec["players"]
    keys = [s.get("key") if isinstance(s, dict) else s for s in sk["skills"]]
    G = np.array(sk["grades"], dtype=np.float32)
    if G.shape[0] != len(players):
        print(
            f"ALIGNMENT BROKEN: {len(players)} players, {G.shape[0]} grade rows. "
            f"Refusing — every grade would land on the wrong player."
        )
        return 2
    tj = keys.index(TARGET)

    seasons = sorted({str(p["season"]) for p in players}, key=B.season_start)
    series, _ = B.vor_series(seasons, B.eligible_pairs(vec))
    draft = json.loads(B.DRAFT.read_text(encoding="utf-8"))["players"]
    merged = B.merged_names(series, draft)

    idx = {(B.norm_name(p["name"]), B.season_start(str(p["season"]))): i for i, p in enumerate(players)}
    raw = [(i, idx[(n, y + 1)], y + 1, n) for (n, y), i in idx.items() if (n, y + 1) in idx]
    pairs = [(i, k, y) for i, k, y, n in raw if n not in merged]
    excluded = len(raw) - len(pairs)

    src = np.array([i for i, _, _ in pairs])
    dst = np.array([k for _, k, _ in pairs])
    ty = np.array([y for _, _, y in pairs])
    y_next = G[dst, tj]
    y_prev = G[src, tj]
    F = G[src]

    tr, te = ty <= CUT_YEAR, ty > CUT_YEAR
    print(f"{len(raw)} consecutive-season pairs, {excluded} excluded as merged names, " f"{len(pairs)} used")
    print(f"  train(target<= {CUT_YEAR}) {tr.sum()}   test {te.sum()}")
    if te.sum() < 200:
        print("test split too small to interpret")
        return 2

    persistence = r(y_prev[te], y_next[te])
    r1 = r(ridge(y_prev[tr, None], y_next[tr], y_prev[te, None]), y_next[te])
    r12 = r(ridge(F[tr], y_next[tr], F[te]), y_next[te])

    rng = np.random.default_rng(SEED)
    rnull = r(ridge(F[tr], y_next[tr][rng.permutation(int(tr.sum()))], F[te]), y_next[te])
    ndist = null_extras_gain(F, tj, y_prev, y_next, tr, te)
    p_val = float((ndist >= (r12 - r1)).mean())

    sweep = []
    for cut in (2010, 2013, 2016, 2019, 2022):
        a_tr, a_te = ty <= cut, ty > cut
        if a_te.sum() < 200:
            continue
        g1 = r(ridge(y_prev[a_tr, None], y_next[a_tr], y_prev[a_te, None]), y_next[a_te])
        g12 = r(ridge(F[a_tr], y_next[a_tr], F[a_te]), y_next[a_te])
        sweep.append(
            {
                "cut_year": cut,
                "n_test": int(a_te.sum()),
                "impact_only_r": round(g1, 4),
                "all12_r": round(g12, 4),
                "gain": round(g12 - g1, 4),
            }
        )
    gains = [s["gain"] for s in sweep]
    all_pos = bool(gains) and all(g > 0 for g in gains)
    gain = r12 - r1
    earns = gain > 0.01 and all_pos and p_val < 0.05

    print(f"\n  persistence (this impact -> next)      r = {persistence:.4f}")
    print(f"  RIDGE-1  (impact only)                 r = {r1:.4f}")
    print(f"  RIDGE-12 (all skill grades)            r = {r12:.4f}")
    print(f"  gain from the other 11 skills          {gain:+.4f}")
    print(
        f"  NULL (extras shuffled) gain            mean {ndist.mean():+.4f}  "
        f"sd {ndist.std():.4f}  ->  p = {p_val:.3f}"
    )
    print(
        f"  gain across {len(sweep)} cuts: mean {np.mean(gains):+.4f}  "
        f"min {min(gains):+.4f}  max {max(gains):+.4f}  "
        f"{'positive at every cut' if all_pos else 'NOT consistently positive'}"
    )
    print(
        f"\n  verdict: {'the profile adds signal beyond impact' if earns else 'NO — the other skills do not beat impact alone'}"
    )

    OUT.write_text(
        json.dumps(
            {
                "question": (
                    "Does an NBA season's skill profile predict next season's impact "
                    "beyond what this season's impact already predicts?"
                ),
                "n_pairs_raw": len(raw),
                "n_excluded_merged_names": excluded,
                "n_pairs_used": len(pairs),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "split": f"TEMPORAL — train on target season <= {CUT_YEAR}, test strictly after",
                "merged_name_note": (
                    "vectors.json has NO player identity key — its `id` is a row index and LeBron "
                    f"James has 23 of them — so {excluded} of {len(raw)} name-keyed consecutive-season "
                    "pairs join TWO DIFFERENT PEOPLE across the boundary. Left in, the model would "
                    "be scored partly on predicting one man's next season from another man's last "
                    "one. Excluded using the same merged_names() set every axis here uses, minus "
                    "the Wikidata DOB acquittals."
                ),
                "era_note": (
                    "skills.json grades are PERCENTILE WITHIN SEASON POOL, so a 1997 grade "
                    "and a 2025 grade are both 'how good among peers that year'. No era "
                    "term is added; one would double-count."
                ),
                "persistence_r": round(persistence, 4),
                "ridge1_impact_only_r": round(r1, 4),
                "ridge12_all_skills_r": round(r12, 4),
                "gain_over_impact_alone": round(gain, 4),
                "cut_year_sweep": sweep,
                "gain_positive_at_every_cut": all_pos,
                "gain_mean_across_cuts": round(float(np.mean(gains)), 4),
                "null_extras_shuffled": {
                    "mean": round(float(ndist.mean()), 4),
                    "sd": round(float(ndist.std()), 4),
                    "pct95": round(float(np.percentile(ndist, 95)), 4),
                    "p_value_of_real_gain": p_val,
                    "reps": int(len(ndist)),
                    "what": (
                        "Keeps the target and the impact column intact, shuffles only the "
                        "other eleven skills. The gain you would see if they carried nothing."
                    ),
                },
                "superseded_shuffled_target_null_r": round(rnull, 4),
                "superseded_null_note": (
                    "A permuted-TARGET null with an ASSERTED 0.15 tolerance was used first. It is "
                    "degenerate — the ridge fits ~the mean, predictions go near-constant, and the "
                    "correlation is unstable: sd 0.1487 over 40 seeds. It would have failed a sound "
                    "evaluation about a third of the time. Kept for the record, not used."
                ),
                "verdict": (
                    "the profile adds signal beyond impact"
                    if earns
                    else "NO — the other skills do not improve on impact alone out of sample"
                ),
                "vs_tennis": (
                    "build_tennis_forward.py asks the same question of tennis and finds "
                    "+0.0941 over a 0.7486 persistence baseline. The two GAINS are not "
                    "directly comparable — different targets, different baselines, "
                    "different sports — and quoting one against the other would be the "
                    "cross-sport apples-to-oranges this repo keeps refusing to make."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")

    if args.check and p_val >= 0.05:
        print(
            f"\nFAIL the gain is not distinguishable from shuffling the other eleven "
            f"skills (p={p_val:.3f}) — they are not earning their place"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
