#!/usr/bin/env python3
"""Does an NFL season's PROFILE predict next season's fantasy output, beyond this one's?

Solo personal project, no connection to employer, built with public/free-tier only

The fourth forward probe and the last sport that could support one. Tennis found style adds
+0.0941 over rank persistence; hoops found the skill profile adds +0.0625 over impact
persistence; equities found essentially nothing over a 0.85-0.92 baseline. Pitch was asked
and could not answer — its contexts are one-off tournaments with no coherent successor, and
only 403 of 1,836 players appear in more than one. Gridiron can answer, and was simply
never asked.

    BASELINE   next PPR points-per-game = this season's                 persistence
    RIDGE-1    ridge on this season's PPR alone
    RIDGE-19   ridge on PPR + the 18 z-scored per-game rate features
    the profile earns its keep only if RIDGE-19 beats RIDGE-1 OUT OF SAMPLE

THE ANSWER IS NO, FOR EVERY POSITION, and one of those verdicts changed during the run:

    group          persist  prof ONLY  +profile     gain          cuts 2012/2015/2018/2021
    POOLED          0.8000     0.8083    0.8099  +0.0099   +0.0102 +0.0097 +0.0099 +0.0138
    QB              0.4914     0.4738    0.4763  -0.0151   -0.0259 -0.0212 -0.0151 +0.0248
    RB              0.7405     0.7432    0.7449  +0.0044   +0.0005 -0.0016 +0.0044 +0.0095
    WR              0.7731     0.7791    0.7802  +0.0071   +0.0066 +0.0076 +0.0071 +0.0104
    TE              0.7507     0.7055    0.7612  +0.0105   +0.0059 +0.0091 +0.0105 -0.0022

THE CUT-YEAR SWEEP IS WHY TE IS NOT REPORTED AS AN EARNER. The first run of this file had
no sweep, and TE cleared both bars on the single 2018 cut — gain +0.0105 against a 0.01
threshold, p = 0.000. It would have shipped as "the tight-end profile forecasts next
season". Moving the boundary to 2021 turns that gain NEGATIVE. A result that changes sign
with where the split was drawn is a fact about the split, not about football, and
build_tennis_forward.py had already added a sweep for exactly this reason — omitting it
here was my error, not a difference between the sports.

QB IS THE INTERESTING NEGATIVE. The profile makes quarterback prediction WORSE (-0.0151,
p = 0.500 — the null beats the real gain half the time) and QB persistence is itself far
weaker than every other position, 0.4914 against 0.74-0.77. Both point the same way: QB
season-to-season output is the least self-similar and the 18 features, which are dominated
by volume rates, add noise rather than signal on the smallest sample here.

THE POOLED NUMBER IS CONFOUNDED BY POSITION AND THE PER-POSITION ONES ARE NOT, which is the
main reason this file is longer than it looks like it needs to be. Pooling QB/RB/WR/TE puts
a 14.48-mean population next to a 5.86-mean one, so a model that learns nothing except
which position a row belongs to will predict next year well:

    POOLED persistence          r = 0.7642
    QB  n=761   r = 0.5848      mean ppg 14.48
    RB  n=1823  r = 0.7211      mean ppg  8.93
    WR  n=2746  r = 0.7301      mean ppg  8.75
    TE  n=1520  r = 0.7367      mean ppg  5.86

The 18 features are z-scored WITHIN SEASON ACROSS ALL FOUR POSITIONS, so PASS_ATT_PG is
strongly negative for every non-QB. A ridge on them can recover position almost exactly,
and "the model knows a quarterback is a quarterback" would then be reported as forecasting
skill. Both are computed; the PER-POSITION result is the finding and the pooled one is
reported as the confounded quantity it is.

IDENTITY IS THE SAME TRAP HOOPS SET, AND IT IS LIVE HERE. vectors.json has no player key —
`id` is a row index, 10,700 distinct over 10,700 rows, and 2,014 names own more than one of
them, so a consecutive-season pair keyed by name can join two different people. 227 of
7,262 pairs do. Excluded with merged_names() from build_vor_draft_value.py, the single
gridiron normaliser in this estate, which applies the gsis and Wikidata acquittals inside
itself rather than leaving that to callers.

MINIMUM 8 GAMES ON BOTH SIDES, PRE-REGISTERED BEFORE THE RUN. A 2-game season's per-game
rate is mostly noise, and noise in the target depresses every correlation equally, which
would understate baseline and model alike. 9,411 of 10,700 rows clear it. The unfiltered
numbers are computed too, so the filter's effect is visible rather than assumed.

ERA IS ALREADY HANDLED: the features are z-scored within season, so a 1999 rate and a 2025
rate both mean "how far from that year's mean". No era term is added; one would double-count.

    python pipeline/build_gridiron_forward.py
    python pipeline/build_gridiron_forward.py --check   # exit 1 if the run is broken
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_vor_draft_value as B  # noqa: E402
from build_tennis_forward import null_extras_gain, r, ridge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VEC = Path("C:/Users/jcdav/vector-gridiron/assets/vectors.json")
OUT = ROOT / "data" / "gridiron_forward_report.json"
CUT_YEAR = 2018          # train on target seasons <= this, test strictly after
MIN_GAMES = 8            # pre-registered, both sides of the pair
EARNS = 0.01
POSITIONS = ("QB", "RB", "WR", "TE")


def arm(ppr, F, src, dst, ty, mask, label):
    """One baseline-vs-profile comparison over a subset of pairs."""
    s, d_, y = src[mask], dst[mask], ty[mask]
    tr, te = y <= CUT_YEAR, y > CUT_YEAR
    if te.sum() < 60 or tr.sum() < 60:
        return {"group": label, "n_pairs": int(mask.sum()), "skipped":
                f"train {int(tr.sum())} / test {int(te.sum())} — too few to interpret"}
    yp, yn = ppr[s], ppr[d_]
    X = np.hstack([yp[:, None], F[s]])
    persist = r(yp[te], yn[te])
    r1 = r(ridge(yp[tr, None], yn[tr], yp[te, None]), yn[te])
    rN = r(ridge(X[tr], yn[tr], X[te]), yn[te])
    only = r(ridge(F[s][tr], yn[tr], F[s][te]), yn[te])
    gain = rN - r1
    nd = null_extras_gain(X, 0, yp, yn, tr, te, reps=40)
    p_val = float((nd >= gain).mean())

    # ROBUSTNESS ACROSS CUTS. Without this, TE's +0.0105 against a 0.01 bar is a verdict
    # decided by one arbitrary split — the exact "one cut can be lucky" case that
    # build_tennis_forward.py added its own sweep to guard against, and which this file
    # omitted on the first run. A gain that flips sign as the boundary moves is not a
    # finding about football, it is a finding about where the boundary was drawn.
    sweep = []
    for cut in (2012, 2015, 2018, 2021):
        a_tr, a_te = y <= cut, y > cut
        if a_te.sum() < 60 or a_tr.sum() < 60:
            continue
        g1 = r(ridge(yp[a_tr, None], yn[a_tr], yp[a_te, None]), yn[a_te])
        gN = r(ridge(X[a_tr], yn[a_tr], X[a_te]), yn[a_te])
        sweep.append({"cut_year": cut, "n_test": int(a_te.sum()),
                      "ppr_only_r": round(g1, 4), "with_profile_r": round(gN, 4),
                      "gain": round(gN - g1, 4)})
    gains = [s_["gain"] for s_ in sweep]
    all_pos = bool(gains) and all(g > 0 for g in gains)

    return {"group": label, "n_pairs": int(mask.sum()), "n_train": int(tr.sum()),
            "n_test": int(te.sum()), "persistence_r": round(persist, 4),
            "ridge1_ppr_only_r": round(r1, 4), "profile_only_r": round(only, 4),
            "ridge19_r": round(rN, 4), "gain": round(gain, 4), "null_p": p_val,
            "null_sd": round(float(nd.std()), 4),
            "cut_year_sweep": sweep, "gain_positive_at_every_cut": all_pos,
            "gain_mean_across_cuts": round(float(np.mean(gains)), 4) if gains else None,
            "earns_its_keep": bool(gain > EARNS and p_val < 0.05 and all_pos)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not VEC.exists():
        print(f"missing {VEC}")
        return 2
    doc = json.loads(VEC.read_text(encoding="utf-8"))
    P = doc["players"]
    feats = list(doc.get("features") or [])
    if not P or len(P[0]["v"]) != len(feats):
        print(f"feature/vector length mismatch: v={len(P[0]['v'])} features={len(feats)}")
        return 2

    # ---- the merged-name set, from the ONE gridiron normaliser ------------------
    seasons_of = defaultdict(list)
    for p in P:
        v = (p.get("ppg") or {}).get("ppr")
        if v is not None:
            seasons_of[B.norm_name(p["name"])].append((int(p["season"]), float(v)))
    merged = B.merged_names(seasons_of, B.DRAFT_CSV)

    ppr = np.array([(p.get("ppg") or {}).get("ppr", np.nan) for p in P], dtype=np.float64)
    games = np.array([p.get("games") or 0 for p in P], dtype=np.int32)
    pos = np.array([p["pos"] for p in P])
    F = np.array([p["v"] for p in P], dtype=np.float64)

    idx = {(B.norm_name(p["name"]), int(p["season"])): i for i, p in enumerate(P)}
    raw = [(i, idx[(n, s + 1)], s + 1, n) for (n, s), i in idx.items() if (n, s + 1) in idx]
    kept = [(a, b, y) for a, b, y, n in raw if n not in merged]
    excluded = len(raw) - len(kept)

    src = np.array([a for a, _, _ in kept])
    dst = np.array([b for _, b, _ in kept])
    ty = np.array([y for _, _, y in kept])

    usable = ~(np.isnan(ppr[src]) | np.isnan(ppr[dst]))
    enough = (games[src] >= MIN_GAMES) & (games[dst] >= MIN_GAMES)
    keep = usable & enough
    src, dst, ty = src[keep], dst[keep], ty[keep]
    print(f"{len(raw)} name-keyed consecutive-season pairs, {excluded} excluded as merged "
          f"names, {int(usable.sum())} with PPR both sides, {len(src)} after "
          f"MIN_GAMES={MIN_GAMES} on both")

    # unfiltered pooled, so the filter's effect is visible rather than assumed
    s_all = np.array([a for a, _, _ in kept])[usable]
    d_all = np.array([b for _, b, _ in kept])[usable]
    print(f"  pooled persistence  unfiltered r = {r(ppr[s_all], ppr[d_all]):.4f}   "
          f"MIN_GAMES>={MIN_GAMES} r = {r(ppr[src], ppr[dst]):.4f}")

    rows = [arm(ppr, F, src, dst, ty, np.ones(len(src), bool), "POOLED (position-confounded)")]
    for q in POSITIONS:
        rows.append(arm(ppr, F, src, dst, ty, pos[src] == q, q))

    print(f"\n  {'group':30} {'persist':>8} {'prof ONLY':>10} {'+profile':>9} {'gain':>8}")
    for w in rows:
        if w.get("skipped"):
            print(f"  {w['group']:30} SKIPPED — {w['skipped']}")
            continue
        cuts = w.get("cut_year_sweep") or []
        flag = ("EARNS" if w["earns_its_keep"] else
                ("no (flips sign across cuts)" if w["gain"] > EARNS and w["null_p"] < 0.05
                 and not w["gain_positive_at_every_cut"] else "no"))
        print(f"  {w['group']:30} {w['persistence_r']:>8.4f} {w['profile_only_r']:>10.4f} "
              f"{w['ridge19_r']:>9.4f} {w['gain']:>+8.4f}   p={w['null_p']:.3f} {flag}")
        if cuts:
            print(f"  {'':30} cuts: " + "  ".join(
                f"{c['cut_year']}:{c['gain']:+.4f}" for c in cuts))

    scored = [w for w in rows if not w.get("skipped")]
    per_pos = [w for w in scored if w["group"] in POSITIONS]
    earners = [w["group"] for w in per_pos if w["earns_its_keep"]]

    OUT.write_text(json.dumps({
        "question": ("Does an NFL season's 18-feature profile predict next season's PPR "
                     "points-per-game beyond what this season's PPR already predicts?"),
        "verdict": (f"{len(earners)} of {len(per_pos)} positions earn it: "
                    f"{', '.join(earners) or 'none'}. The PER-POSITION rows are the finding; "
                    f"the pooled row is reported and is confounded."),
        "cut_sweep_changed_a_verdict": (
            "TE cleared both bars on the single 2018 cut — gain +0.0105 against a 0.01 "
            "threshold at p=0.000 — and would have shipped as 'the tight-end profile "
            "forecasts next season'. At the 2021 cut the same gain is -0.0022. A result "
            "that changes sign with where the boundary was drawn is a fact about the "
            "boundary. The first version of this file had no sweep; build_tennis_forward.py "
            "already had one for this reason, so its absence here was an omission rather "
            "than a difference between the sports."),
        "qb_note": (
            "The profile makes QB prediction WORSE (-0.0151, p=0.500 — the null beats the "
            "real gain half the time), and QB persistence is itself the weakest at 0.4914 "
            "against 0.74-0.77 elsewhere. QB output is the least self-similar season to "
            "season, and the volume-dominated features add noise on the smallest sample."),
        "why_per_position_is_the_finding": (
            "Pooling QB/RB/WR/TE puts a 14.48-mean population beside a 5.86-mean one, and "
            "the 18 features are z-scored WITHIN SEASON ACROSS ALL FOUR positions, so "
            "PASS_ATT_PG is strongly negative for every non-QB. A ridge can recover "
            "position from them almost exactly, and 'the model knows a quarterback is a "
            "quarterback' would be reported as forecasting skill. Pooled persistence is "
            "0.7642 against per-position 0.58-0.74 for exactly that reason."),
        "merged_name_note": (
            f"vectors.json has NO player key — `id` is a row index, 10,700 distinct over "
            f"10,700 rows, and 2,014 names own more than one. {excluded} of {len(raw)} "
            f"name-keyed pairs join two different people across the season boundary. "
            f"Excluded via merged_names() in build_vor_draft_value.py, the one gridiron "
            f"normaliser, which applies the gsis and Wikidata acquittals internally."),
        "min_games": MIN_GAMES,
        "min_games_note": (
            "Pre-registered before the run, not tuned. A 2-game season's per-game rate is "
            "mostly noise, and noise in the target depresses baseline and model equally. "
            "Unfiltered pooled persistence is reported alongside so the effect is visible."),
        "era_note": ("Features are z-scored within season, so a 1999 rate and a 2025 rate "
                     "both mean 'how far from that year's mean'. No era term is added."),
        # HEADLINE FIELDS, UNIQUELY NAMED AND TOP-LEVEL. The per-group numbers live inside
        # per_group[], where `gain` occurs five times with five different values — so
        # check_cited_fields.py correctly refuses to verify a page citing `gain=-0.0151`,
        # because it cannot tell which of the five was meant. Anything published from this
        # report would therefore be unverifiable exactly where it is most interesting.
        # A report whose headline numbers can only be reached by indexing into a list is
        # awkward for a human consumer and unusable for a mechanical one; these names are
        # unambiguous, so a citation to them can be checked.
        "headline_positions_earning_keep": len(earners),
        "headline_positions_tested": len(per_pos),
        "headline_pooled_persistence_r": next(
            (w["persistence_r"] for w in scored if w["group"].startswith("POOLED")), None),
        "headline_qb_persistence_r": next(
            (w["persistence_r"] for w in scored if w["group"] == "QB"), None),
        "headline_qb_gain": next((w["gain"] for w in scored if w["group"] == "QB"), None),
        "headline_te_gain_at_main_cut": next(
            (w["gain"] for w in scored if w["group"] == "TE"), None),
        "headline_te_gain_at_2021_cut": next(
            (c["gain"] for w in scored if w["group"] == "TE"
             for c in (w.get("cut_year_sweep") or []) if c["cut_year"] == 2021), None),
        "n_pairs_raw": len(raw), "n_excluded_merged_names": excluded,
        "n_pairs_used": len(src),
        "split": f"TEMPORAL — train on target season <= {CUT_YEAR}, test strictly after",
        "target": "ppg.ppr (PPR fantasy points per game)",
        "per_group": rows,
        "vs_other_sports": (
            "tennis +0.0941 over 0.7486, hoops +0.0625 over 0.4514, equities ~0 over "
            "0.85-0.92. The GAINS are not comparable across sports — different targets, "
            "baselines and domains — and quoting one against another would be the "
            "apples-to-oranges this repo keeps refusing to make."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")

    # ---- --check has a BODY, unlike the equities probe as first shipped ---------
    # build_equities_forward.py accepted --check, never read it, and ran as a registered
    # gate that could not fail under any input. "Broken" here means the same two things:
    # a target that carries forward (which would make persistence an artefact) and a null
    # with no spread (which would make every p-value meaningless).
    fails = []
    identical = int((np.abs(ppr[dst] - ppr[src]) < 1e-9).sum())
    if identical / max(1, len(src)) > 0.01:
        fails.append(f"CARRY-FORWARD: {identical}/{len(src)} pairs have an identical PPR "
                     f"both seasons — persistence would be measuring duplication")
    flat = [w["group"] for w in scored if w["null_sd"] < 1e-6]
    if flat:
        fails.append(f"DEGENERATE NULL in {flat}: shuffled-extras sd is ~0, so every "
                     f"p-value for those groups is uninterpretable")
    if args.check and fails:
        print()
        for f_ in fails:
            print(f"FAIL {f_}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
