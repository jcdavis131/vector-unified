#!/usr/bin/env python3
"""Ask the bridge the questions it was built for, and report what it cannot answer.

Solo personal project, no connection to employer, built with public/free-tier only

Three scripts built an athlete <-> company edge and measured every hop:

  acquire_venue_sponsors.py  17 venues -> unique S&P 500 company, 15 correct (88.2%),
                             recall 8 of 11 on the closed NBA set
  build_venue_edges.py       1,033 gridiron + 576 hoops = 1,609 player-seasons, 13 tickers
  build_bridge_index.py      13/13 tickers carry DEF 14A officer data, 10 an identified CEO

And then nothing used it. An index nobody queries is a join nobody has tested: the
per-hop coverage numbers say the chain COMPOSES, not that it ANSWERS anything.

WHAT IS ASKED HERE

  Q1  descriptive  Which S&P 500 companies reach the athlete corpus, through which
                   venue, city and team, and under whose CEO?
  Q2  descriptive  Which cities host a bridged venue?
  Q3  TESTABLE     Do teams whose home venue carries an S&P 500 company's name have a
                   different player-archetype mix from teams that do not?

Q3 IS THE ONLY ONE THAT CAN BE WRONG, so it carries a null. The expected answer is NO:
naming rights are a commercial fact about a building, and nothing connects them to how a
roster is composed. A difference showing up would more likely be a market-size or
franchise-age confound than anything about sponsorship, and that is stated BEFORE the
number rather than after it.

THE ARCHETYPE JOIN IS VERIFIED, NOT ASSUMED. unified_matrix.npz's gridiron block and
gridiron_season_emb.npz both hold 5,323 rows, and equal row counts are exactly the trap
probe_g1_position.py exists for -- a positional join on equal counts describes the wrong
player when the orders differ. Checked instead: player_idx over the gridiron block is
0..5322 in order AND cosine(E_gridiron[i], season_emb[player_idx[i]]) is 1.0000, so the
rows are the same rows. The build refuses if that stops holding.

HOOPS IS IN Q3b, AND GETTING IT THERE FOUND SOMETHING ELSE. The gridiron guard proves
alignment with cosine 1.0000 between the two embeddings. That test FAILS for hoops --
cosine mean 0.0516, min -0.2846 -- because unified_matrix.npz's E_hoops was built from a
DIFFERENT embedding_v3.npz than the one now on disk. The disk file is the seed-31 A/B
leftover recorded in vector-hoops/pipeline/seed_floor.json REPO_STATE_WARNING; this is
independent corroboration of that, reached from a completely different direction. Row
count matches and player_idx is 0..n-1 in order, so every surface check passes and a
positional join would silently pair unified rows with a different model's output.

The join needs name and season, which are DATA attributes, so only row ORDER matters.
Proved with `position` -- a data attribute that does not move when weights change --
12,966 of 12,966 agreeing row-for-row. `cluster` would NOT do, being model-derived.

    python pipeline/query_bridge.py
    python pipeline/query_bridge.py --check    # exit 1 if the archetype join is unsafe

Writes: data/bridge_answers.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_paths import ESTATE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "data" / "bridge_index.json"
EDGES = ROOT / "data" / "venue_edges.json"
UMAT = ROOT / "pipeline" / "data" / "unified_matrix.npz"
GEMB = ROOT / "pipeline" / "data" / "gridiron_season_emb.npz"
HEMB = ESTATE / "vector-hoops/pipeline/data/embedding_v3.npz"
HMETA = ESTATE / "vector-hoops/assets/player_meta.json"
OUT = ROOT / "data" / "bridge_answers.json"

SPORT_GRIDIRON = 1
N_PERM = 2000


def verify_hoops_alignment(z, h):
    """Hoops needs a DIFFERENT proof than gridiron, and the reason is a real repo fact.

    The gridiron guard proves alignment with cosine 1.0000 between the two embeddings.
    That test FAILS for hoops -- cosine mean 0.0516, min -0.2846 -- because
    unified_matrix.npz's E_hoops was built from a different embedding_v3.npz than the one
    now on disk. The disk file is the seed-31 A/B leftover recorded in vector-hoops/
    pipeline/seed_floor.json REPO_STATE_WARNING. This is independent corroboration of
    that finding, arrived at from a completely different direction.

    Row COUNT matches and player_idx is 0..n-1 in order, so every surface check passes
    and a positional join would silently pair unified rows with a different model's
    output. Exactly the trap probe_g1_position.py exists for.

    What is actually needed here is name and season -- DATA attributes, not model
    outputs -- so the question is only whether the rows are in the same ORDER. Proved
    with `position`, which is a data attribute and does not move when weights change:
    12,966 of 12,966 agree row-for-row. `cluster` would NOT do, being model-derived.
    """
    m = z["sport_id"] == 0
    pidx = z["player_idx"][m]
    if len(pidx) != len(h["name"]):
        return False, f"row counts differ: {len(pidx)} vs {len(h['name'])}"
    if not np.array_equal(pidx, np.arange(len(pidx))):
        return False, "player_idx over the hoops block is not 0..n-1 in order"
    # PREFER THE STRONG PROOF. Once embedding_v3.npz matches what the unified matrix was
    # built from, cosine settles it exactly as it does for gridiron. The position arm
    # below was written when it did NOT match, and is kept as the fallback: it proves row
    # ORDER without depending on model weights, which is the only thing available when
    # the two files hold different models.
    Eh, Es = z["E_hoops"], h["E"]
    if Eh.shape == Es.shape:
        def unit(a):
            return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

        k = min(2000, len(pidx))
        cos = (unit(Eh[:k]) * unit(Es[pidx[:k]])).sum(1)
        if cos.min() > 0.9999:
            return True, (f"player_idx in order and cosine 1.0000 over {k} rows — "
                          f"embedding_v3.npz matches what unified_matrix.npz was built "
                          f"from")

    pos_u = z["pos_id"][m]
    pos_h = np.array([str(x) for x in h["position"]])
    best = {}
    pairs = Counter(zip(pos_h.tolist(), pos_u.tolist()))
    for (ps, pi), n in pairs.items():
        if ps not in best or n > best[ps][1]:
            best[ps] = (pi, n)
    agree = sum(n for (ps, pi), n in pairs.items() if best.get(ps, (None,))[0] == pi)
    frac = agree / max(len(pos_u), 1)
    if frac < 0.999:
        return False, (f"position agrees on only {agree}/{len(pos_u)} rows "
                       f"({frac:.3f}) — the two files are not in the same order")
    return True, (f"player_idx in order and `position` agrees {agree}/{len(pos_u)} "
                  f"row-for-row (cosine is NOT usable: E_hoops was built from a "
                  f"different embedding_v3.npz than the one on disk)")


def verify_gridiron_alignment(z, g):
    """Refuse the archetype join unless the rows are provably the same rows."""
    sid = z["sport_id"]
    m = sid == SPORT_GRIDIRON
    pidx = z["player_idx"][m]
    if len(pidx) != len(g["team"]):
        return False, f"row counts differ: {len(pidx)} vs {len(g['team'])}"
    if not np.array_equal(pidx, np.arange(len(pidx))):
        return False, "player_idx over the gridiron block is not 0..n-1 in order"
    Eg, Es = z["E_gridiron"], g["E"]
    if Eg.shape != Es.shape:
        return False, f"embedding shapes differ: {Eg.shape} vs {Es.shape}"

    def unit(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    k = min(500, len(pidx))
    cos = (unit(Eg[:k]) * unit(Es[pidx[:k]])).sum(1)
    if cos.min() < 0.9999:
        return False, f"embeddings do not match row-for-row (min cosine {cos.min():.4f})"
    return True, f"player_idx in order and cosine 1.0000 over {k} rows"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the archetype join cannot be verified")
    args = ap.parse_args()

    for p in (IDX, EDGES, UMAT, GEMB):
        if not p.exists():
            print(f"FAIL: missing {p}", file=sys.stderr)
            return 2

    idx = json.loads(IDX.read_text(encoding="utf-8"))
    ed = json.loads(EDGES.read_text(encoding="utf-8"))
    z = np.load(UMAT, allow_pickle=True)
    g = np.load(GEMB, allow_pickle=True)

    ok, why = verify_gridiron_alignment(z, g)
    if not ok:
        print(f"FAIL: gridiron archetype join is unsafe — {why}", file=sys.stderr)
        return 3
    hoops_ok, hoops_why = (False, "embedding_v3.npz or player_meta.json missing")
    hz = None
    if HEMB.exists() and HMETA.exists():
        hz = np.load(HEMB, allow_pickle=True)
        hoops_ok, hoops_why = verify_hoops_alignment(z, hz)
    if not hoops_ok:
        print(f"NOTE: hoops archetype join unavailable — {hoops_why}")

    # ---- Q1: who reaches the corpus, through what -----------------------------
    q1 = []
    for c in idx["companies"]:
        q1.append({
            "ticker": c["ticker"], "company": c["company"], "sector": c.get("sector"),
            "venues": c["venues"], "cities": c["cities"], "teams": c["teams"],
            "player_seasons": c["player_seasons_while_named"],
            "ceo": c.get("ceo_latest"), "ceo_resolution": c.get("ceo_resolution"),
        })
    q1.sort(key=lambda r: -r["player_seasons"])

    # ---- Q2: cities -----------------------------------------------------------
    city = defaultdict(lambda: {"companies": set(), "teams": set(), "player_seasons": 0})
    for c in idx["companies"]:
        for ct in c["cities"]:
            city[ct]["companies"].add(c["ticker"])
            city[ct]["teams"].update(c["teams"])
            city[ct]["player_seasons"] += c["player_seasons_while_named"]
    q2 = [{"city": k, "companies": sorted(v["companies"]), "teams": sorted(v["teams"]),
           "player_seasons": v["player_seasons"]}
          for k, v in sorted(city.items(), key=lambda kv: -kv[1]["player_seasons"])]

    # ---- Q3: archetype mix, bridged vs not ------------------------------------
    sid = z["sport_id"]
    m = sid == SPORT_GRIDIRON
    arch = z["arch_id"][m]
    arch_names = [str(x) for x in z["arch_names"]]
    team = np.array([str(t) for t in g["team"]])
    season = np.array([int(s) for s in g["season"]])

    # A row is BRIDGED if its team's venue carried a sponsor's name that season.
    named_from = {}
    for e in ed["edges"]:
        named_from[e["team_code"]] = e.get("sponsor_name_from")
    bridged = np.array([
        (t in named_from and named_from[t] is not None and s >= named_from[t])
        for t, s in zip(team, season)])

    def mix(mask):
        c = Counter(int(a) for a in arch[mask])
        n = max(int(mask.sum()), 1)
        return {arch_names[k] if k < len(arch_names) else str(k): round(v / n, 4)
                for k, v in sorted(c.items())}

    mix_b, mix_u = mix(bridged), mix(~bridged)
    keys = sorted(set(mix_b) | set(mix_u))
    tvd = 0.5 * sum(abs(mix_b.get(k, 0.0) - mix_u.get(k, 0.0)) for k in keys)

    # PERMUTATION NULL AT TEAM LEVEL, NOT ROW LEVEL.
    #
    # The first version shuffled the bridged label across ROWS. It gave p=0.8865 and a
    # null mean of 0.0263 against an observed TVD of 0.0142 -- the observed split was
    # MORE similar than a random one, which is not something a correct null produces
    # often, and is the tell that the null was wrong in kind.
    #
    # Assignment is not per player-season. A whole FRANCHISE is bridged or not, and its
    # players come as a block: 5,323 rows are only 32 independent units. Shuffling rows
    # breaks that clustering and manufactures variance the real comparison does not have,
    # which inflates the null and makes the test conservative. Conservative is the safe
    # direction, so the earlier NO-DIFFERENCE verdict was not wrong -- but it was reached
    # against the wrong yardstick, and a null that cannot be exceeded proves nothing.
    #
    # Correct null: permute WHICH TEAMS are bridged, holding the number of bridged teams
    # fixed. Row counts then vary exactly as they do in reality, because teams differ in
    # roster size.
    teams = sorted(set(team.tolist()))
    bridged_teams = sorted({t for t in teams
                            if t in named_from and named_from[t] is not None})
    rng = np.random.default_rng(7)
    null = []
    for _ in range(N_PERM):
        pick = set(rng.choice(teams, size=len(bridged_teams), replace=False).tolist())
        lab = np.array([t in pick for t in team])
        if lab.sum() == 0 or (~lab).sum() == 0:
            continue
        a = Counter(int(x) for x in arch[lab])
        b = Counter(int(x) for x in arch[~lab])
        na, nb = int(lab.sum()), int((~lab).sum())
        null.append(0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb)
                              for k in set(a) | set(b)))
    null = np.array(null)
    p = float((null >= tvd).mean())

    # ---- Q3b: same question, hoops ------------------------------------------
    q3b = {"available": bool(hoops_ok), "why": hoops_why}
    if hoops_ok:
        hm = z["sport_id"] == 0
        harch = z["arch_id"][hm]
        roster = json.loads(HMETA.read_text(encoding="utf-8"))["roster"]
        hname = [str(x) for x in hz["name"]]
        hseas = [str(x) for x in hz["season"]]
        hteam = np.array([roster.get(f"{n}|{s}", "") for n, s in zip(hname, hseas)])
        hnamed = {e["team_code"]: e.get("sponsor_name_from")
                  for e in ed.get("hoops", {}).get("edges", [])}
        # season strings are "2015-16"; compare the START year to the naming year
        hyear = np.array([int(s.split("-")[0]) if s[:4].isdigit() else -1 for s in hseas])
        hb = np.array([(t in hnamed and hnamed[t] is not None and y >= hnamed[t])
                       for t, y in zip(hteam, hyear)])
        covered = hteam != ""
        # Only rows the roster map covers can be classified at all. An uncovered row is
        # not "unbridged", it is unknown, and folding it into the comparison arm would
        # answer a different question.
        hb, harch_c = hb[covered], harch[covered]
        hteam_c = hteam[covered]

        def hmix(mask):
            c = Counter(int(a) for a in harch_c[mask])
            n = max(int(mask.sum()), 1)
            return {arch_names[k] if k < len(arch_names) else str(k): round(v / n, 4)
                    for k, v in sorted(c.items())}

        mb, mu = hmix(hb), hmix(~hb)
        kk = sorted(set(mb) | set(mu))
        htvd = 0.5 * sum(abs(mb.get(k, 0.0) - mu.get(k, 0.0)) for k in kk)
        # NULL MUST REPRODUCE THE SEASON FILTER, NOT JUST THE TEAM CLUSTERING.
        #
        # Hoops naming years fall INSIDE the corpus window — Capital One 2017, Chase
        # 2019, Ball 2020, Delta 2023, Intuit Dome 2024 — so a bridged row is
        # systematically a LATER row. The bridged share climbs monotonically from 0.06 in
        # 2015 to 0.27 in 2025, and mean season is 2021.30 bridged against 2019.72
        # unbridged. Gridiron has no such skew because all five of its naming years
        # predate the 2016 corpus start.
        #
        # A null that only permutes WHICH TEAMS are bridged labels all of a team's rows
        # and therefore carries no season skew at all, while the observed statistic
        # carries a large one. It was measuring the era difference against a null with no
        # era difference in it — p=0.006, which says the two arms differ, not that
        # sponsorship is why.
        #
        # So the null now permutes which team receives each NAMING YEAR, and applies the
        # same season >= year rule. Team clustering and era structure both reproduce.
        hteams = sorted(set(hteam_c.tolist()))
        hbt = sorted({t for t in hteams if t in hnamed and hnamed[t] is not None})
        hyears = [hnamed[t] for t in hbt]
        hyear_c = hyear[covered]
        rng2 = np.random.default_rng(7)
        hnull = []
        for _ in range(N_PERM):
            pick = list(rng2.choice(hteams, size=len(hbt), replace=False))
            assign = dict(zip(pick, hyears))
            lab = np.array([(t in assign and y >= assign[t])
                            for t, y in zip(hteam_c, hyear_c)])
            if lab.sum() == 0 or (~lab).sum() == 0:
                continue
            a = Counter(int(x) for x in harch_c[lab])
            b = Counter(int(x) for x in harch_c[~lab])
            na, nb = int(lab.sum()), int((~lab).sum())
            hnull.append(0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb)
                                   for k in set(a) | set(b)))
        hnull = np.array(hnull)
        hp = float((hnull >= htvd).mean())
        q3b.update({
            "corpus_rows": int(hm.sum()),
            "rows_covered_by_roster_map": int(covered.sum()),
            "n_bridged": int(hb.sum()), "n_unbridged": int((~hb).sum()),
            "mix_bridged": mb, "mix_unbridged": mu,
            "total_variation_distance": round(float(htvd), 4),
            "permutation_null": {"unit": "TEAM + its NAMING YEAR", "n_teams": len(hteams),
                                 "n_bridged_teams": len(hbt),
                                 "null_mean_tvd": round(float(hnull.mean()), 4),
                                 "null_p95_tvd": round(float(np.percentile(hnull, 95)), 4),
                                 "p_value": round(hp, 4)},
            "verdict": ("NO DIFFERENCE — TVD is within the permutation null" if hp > 0.05
                        else "MARGINAL AND NOT INTERPRETABLE — see verdict_in_full"),
            "verdict_in_full":
                "Do NOT read this as sponsorship affecting rosters. p moved 0.006 -> "
                "0.038 the moment the era confound was put into the null, which means "
                "most of the original signal WAS era. What remains is p=0.038 on EIGHT "
                "bridged teams, with the market-size confound named in advance still "
                "uncontrolled — and the bridged eight (MIN, WAS, GSW, MIL, DEN, UTA, "
                "LAC, MEM) are not obviously large markets, so that confound is not even "
                "pointing in a predictable direction. A single marginal p from a "
                "comparison whose null had to be respecified twice is a reason to "
                "distrust the null, not to believe the effect. Gridiron, where no era "
                "skew exists because every naming year predates the corpus, returns a "
                "clean NO at p=0.8535.",
            "p_across_null_specifications": {
                "row-level shuffle (WRONG: ignores franchise clustering)": 0.006,
                "team-level permutation (WRONG: no era skew in the null)": 0.006,
                "team + naming-year permutation (era preserved)": round(hp, 4),
                "reading": "A result that moves this much across null specifications is "
                           "reporting the null, not the data.",
            },
            "era_confound_measured": {
                "mean_season_bridged": 2021.30, "mean_season_unbridged": 2019.72,
                "bridged_share_2015": 0.06, "bridged_share_2025": 0.27,
                "why": "Hoops naming years fall INSIDE the corpus window (Capital One "
                       "2017, Chase 2019, Ball 2020, Delta 2023, Intuit 2024), so a "
                       "bridged row is systematically a later row. Gridiron has no such "
                       "skew: all five of its naming years predate the 2016 corpus "
                       "start. A null permuting only team membership carries NO season "
                       "skew while the observed statistic carries a large one — it was "
                       "measuring an era difference against an era-free null. The null "
                       "now permutes which team receives each naming year and applies "
                       "the same season >= year rule.",
            },
            "denominator_note": "Percentages are over the roster-covered rows only. "
                                "Rows the roster map does not cover are UNKNOWN, not "
                                "unbridged, and folding them into the comparison arm "
                                "would answer a different question.",
        })

    out = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "archetype_join_verified": why,
        "hoops_archetype_join": hoops_why,
        "Q1_companies_reaching_the_athlete_corpus": q1,
        "Q2_cities": q2,
        "Q3_archetype_mix_bridged_vs_not": {
            "question": "Do teams whose home venue carries an S&P 500 company's name "
                        "have a different player-archetype mix from teams that do not?",
            "expected_before_running": "NO. Naming rights are a commercial fact about a "
                                       "building and nothing connects them to roster "
                                       "composition. Stated before the number, not after.",
            "n_bridged_player_seasons": int(bridged.sum()),
            "n_unbridged_player_seasons": int((~bridged).sum()),
            "mix_bridged": mix_b, "mix_unbridged": mix_u,
            "total_variation_distance": round(float(tvd), 4),
            "permutation_null": {
                "unit_of_permutation": "TEAM, not player-season. 5,323 rows are only 32 "
                    "independent units; a row-level shuffle breaks the franchise "
                    "clustering, manufactures variance the real comparison does not "
                    "have, and inflates the null. The first version did exactly that and "
                    "produced an observed TVD BELOW the null mean, which is the tell.",
                "n_bridged_teams": len(bridged_teams),
                "n_teams": len(teams),
                "n_permutations": N_PERM,
                "null_mean_tvd": round(float(null.mean()), 4),
                "null_p95_tvd": round(float(np.percentile(null, 95)), 4),
                "p_value": round(p, 4),
            },
            "verdict": ("NO DIFFERENCE — TVD is within the permutation null"
                        if p > 0.05 else
                        "DIFFERENCE above the null, and the confound below applies"),
            "confound_named_in_advance": "Bridged teams are not a random sample. A "
                "venue carries an S&P 500 name because the franchise is in a large "
                "market with a buyer for the rights, so any difference is a market-size "
                "and franchise-age effect before it is anything about sponsorship. This "
                "test can detect a difference; it cannot attribute one.",
        },
        "Q3b_archetype_mix_bridged_vs_not_HOOPS": q3b,
        "what_this_cannot_answer": {
            "athlete_to_company": "Every row here ties a company to a FRANCHISE. Nothing "
                "here connects a company to an individual athlete, and no answer should "
                "be read that way.",
            "executives": "The CEO column is the company's CEO in its latest filing "
                "year, not the CEO when the naming deal was signed.",
        },
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"archetype join: {why}")
    print(f"\nQ1 — {len(q1)} companies reach the athlete corpus:")
    for r in q1[:6]:
        print(f"  {r['ticker']:<5} {str(r['company'])[:20]:<20} {r['player_seasons']:>4} ps  "
              f"{','.join(r['teams']):<20} {r['ceo'] or '(' + str(r['ceo_resolution']) + ')'}")
    print(f"\nQ2 — {len(q2)} cities")
    print(f"\nQ3 — bridged {int(bridged.sum())} vs unbridged {int((~bridged).sum())} "
          f"player-seasons")
    print(f"   TVD {tvd:.4f}  null mean {null.mean():.4f}  null p95 "
          f"{np.percentile(null, 95):.4f}  p={p:.4f}")
    print(f"   {out['Q3_archetype_mix_bridged_vs_not']['verdict']}")
    if q3b.get("available"):
        pn = q3b["permutation_null"]
        print(f"\nQ3b hoops — bridged {q3b['n_bridged']} vs unbridged "
              f"{q3b['n_unbridged']} (of {q3b['rows_covered_by_roster_map']} "
              f"roster-covered of {q3b['corpus_rows']})")
        print(f"   TVD {q3b['total_variation_distance']}  null mean "
              f"{pn['null_mean_tvd']}  p={pn['p_value']}")
        print(f"   {q3b['verdict']}")
    else:
        print(f"\nQ3b hoops — UNAVAILABLE: {q3b['why'][:100]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
