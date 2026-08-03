#!/usr/bin/env python3
"""One name, two people. Nobody plays a season before they were drafted.

Solo personal project, no connection to employer, built with public/free-tier only

Operator report, 2026-08-03: "some players are 'jr' but are getting combined with another
player's career eg jaren jackson and jaren jackson jr. also some players have the same exact
name so you need to check dates of birth and team to get unique players consistently."

Correct, and it is worse than a normalisation bug. Every `norm_name()` in this repo strips
`jr|sr|ii|iii|iv|v`, but that is not where the merge happens — **vector-hoops' own
`vectors.json` already carries zero suffixes**, so Sr. and Jr. arrive as one key before this
repo sees them. Measured: `Jaren Jackson` holds thirteen seasons spanning 1996-97 to
2025-26, and `vectors.json`'s `id` field is a ROW index, not a player id — LeBron James has
twenty-three different ones. **There is no player identity key in the source at all.**

WHAT IT COST, and this is why the check exists rather than a note:

    direction_axis_hoops.json   jaren jackson   0.94 -> 10.09   delta +9.15   axis D0

Labelled a rising career. It is Jaren Jackson Sr. followed by Jaren Jackson Jr., and the
axis reports two people as one of the biggest risers in the league. Same shape as the Damion
James defect in 7.8a: a real number answering a different question.

THE TEST IS ARITHMETIC, NOT HEURISTIC. A gap in a career proves nothing — Anthony Parker
played in Europe from 2000 to 2005, Antonio McDyess lost years to a knee. 115 of 2,415
eligible hoops careers carry a >=3-season gap and most are one person. What cannot happen:

    A PLAYER CANNOT RECORD AN NBA SEASON BEFORE THE YEAR HE WAS DRAFTED.

Jaren Jackson's draft entry is (2018, pick 4) — Jr. — and the name has seasons from 1996-97.
Seventeen years of impossible. That is definitive, needs no threshold, and no legitimate
career can trip it.

TWO SIGNALS, reported separately because they carry different weight:

  IMPOSSIBLE   a season strictly before the draft year. Definitive. Two people.
  AMBIGUOUS    the name has more than one entry in draft_history. Definitive that two
               people share the name; which seasons belong to whom is unknown, so the
               draft slot cannot be attributed. 250 of 7,383 draft names.
  REVIEW       a large career gap with neither of the above. Suggestive only, and the
               false-positive rate is high enough that it is never used to exclude.

WHAT THIS DOES NOT DO. It does not split a merged career, because splitting without an
identity key is a guess dressed as a repair. The operator's suggestion — date of birth —
is the real fix and it is free via Wikidata, the same machinery
`probe_pitch_expectation_sources.py` already uses. Until that exists, the honest handling is
to EXCLUDE the definitively-merged from any axis that treats a career as one person's.

    python pipeline/check_merged_careers.py
    python pipeline/check_merged_careers.py --check   # exit 1 if an axis carries one
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_hoops_vor_draft_value as B  # noqa: E402
import build_vor_draft_value as G  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# artifact -> (row key, WHICH SPORT'S merged set applies). Scoped per sport because a name
# merged in gridiron says nothing about hoops: the first version unioned both sets and
# applied them to every artifact, flagging `james jones` and `anthony johnson` in the HOOPS
# table on the strength of a gridiron collision.
AXES = {
    "direction_axis_hoops.json": ("careers", "hoops"),
    "hoops_vor_draft_value.json": ("players", "hoops"),
    "direction_axis_gridiron.json": ("careers", "gridiron"),
    "vor_draft_value.json": ("player_rows", "gridiron"),
}
OUT = ROOT / "data" / "merged_careers.json"
GAP_YEARS = 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if an axis artifact still carries a merged career")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vec = json.loads(B.VECTORS.read_text(encoding="utf-8"))
    seasons = sorted({str(p["season"]) for p in vec["players"]}, key=B.season_start)
    series, _ = B.vor_series(seasons, B.eligible_pairs(vec))
    draft = json.loads(B.DRAFT.read_text(encoding="utf-8"))["players"]

    picks_by_norm: dict[str, list] = collections.defaultdict(list)
    for raw, picks in draft.items():
        picks_by_norm[B.norm_name(raw)].extend(picks)

    impossible, ambiguous, review = {}, {}, {}
    for name, rows in series.items():
        years = sorted({y for y, _ in rows})
        picks = picks_by_norm.get(name) or []
        draft_years = sorted({p["year"] for p in picks if p.get("year")})

        if len(picks) > 1:
            ambiguous[name] = {
                "picks": [(p.get("year"), p.get("overall")) for p in picks],
                "seasons": [years[0], years[-1]], "n_seasons": len(years)}

        if draft_years:
            earliest_draft = min(draft_years)
            before = [y for y in years if y < earliest_draft]
            if before:
                impossible[name] = {
                    "earliest_draft_year": earliest_draft,
                    "seasons_before_draft": before,
                    "span": [years[0], years[-1]], "n_seasons": len(years)}
                continue

        gaps = [(a, b) for a, b in zip(years, years[1:], strict=False) if b - a >= GAP_YEARS]
        if gaps and name not in ambiguous:
            review[name] = {"gaps": gaps, "span": [years[0], years[-1]],
                            "n_seasons": len(years)}

    # GRIDIRON TOO. The operator's report named a hoops pair, but the defect is larger
    # here: 318 draft names carry more than one distinct draft year against hoops' 250, and
    # `antonio brown` was this estate's number-one gridiron D0 example at +6.93 while
    # holding seasons from 2003 and 2005 against a 2010 draft.
    gvec = json.loads(G.GRID_VEC.read_text(encoding="utf-8"))["players"]
    gseries: dict[str, list] = collections.defaultdict(list)
    for p in gvec:
        ppr = (p.get("ppg") or {}).get("ppr")
        if ppr is not None:
            gseries[G.norm_name(p["name"])].append((int(p["season"]), float(ppr)))
    gmerged = G.merged_names(gseries, G.DRAFT_CSV)

    by_sport = {"hoops": set(impossible) | set(ambiguous), "gridiron": gmerged}
    definitive = by_sport["hoops"] | by_sport["gridiron"]

    contaminated = {}
    for fn, (key, sport) in AXES.items():
        p = ROOT / "data" / fn
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        rows = doc.get(key) or doc.get("report", {}).get(key) or []
        bad = [r for r in rows if r.get("name") in by_sport[sport]]
        if bad:
            contaminated[fn] = [
                {"name": r["name"],
                 **{k: r[k] for k in ("delta", "direction", "vor_total", "overall")
                    if k in r}}
                for r in bad[:10]]

    report = {
        "operator_report": (
            "Jaren Jackson and Jaren Jackson Jr. are being combined; same-name players need "
            "date of birth and team to separate. Reported 2026-08-03."),
        "root_cause": (
            "Not the norm_name() suffix strip. vector-hoops' vectors.json already carries "
            "ZERO suffixes, so Sr. and Jr. arrive as one key. Its `id` field is a row index, "
            "not a player id — LeBron James has 23 of them. There is no player identity key "
            "in the source."),
        "eligible_careers": len(series),
        "impossible_count": len(impossible),
        "ambiguous_count": len(ambiguous),
        "review_count": len(review),
        "definitive_count": len(definitive),
        "gridiron_merged_count": len(gmerged),
        "gridiron_note": ("Same two definitive tests applied to gridiron via "
                          "build_vor_draft_value.merged_names. The defect is larger there: "
                          "318 draft names with >1 distinct draft year, and `antonio brown` "
                          "was the top D0 example while carrying pre-draft seasons."),
        "impossible": dict(sorted(impossible.items())),
        "ambiguous": dict(sorted(ambiguous.items())[:40]),
        "review_sample": dict(sorted(review.items())[:20]),
        "contaminated_artifacts": contaminated,
        "test_note": (
            "IMPOSSIBLE is arithmetic: a player cannot record a season before the year he "
            "was drafted. It needs no threshold and no legitimate career trips it. A career "
            "GAP proves nothing by contrast — Anthony Parker played in Europe 2000-2005 and "
            f"{len(review)} careers carry a >={GAP_YEARS}-season gap — so gaps are REVIEW "
            "only and never used to exclude."),
        "the_real_fix": (
            "Date of birth, as the operator said. It is free via Wikidata and this repo "
            "already runs that query in probe_pitch_expectation_sources.py. Until it "
            "exists, definitively-merged careers should be EXCLUDED from any axis that "
            "treats a career as one person's, not split — splitting without an identity key "
            "is a guess dressed as a repair."),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"eligible hoops careers {len(series)}")
    print(f"  IMPOSSIBLE (season before draft year) : {len(impossible)}")
    for n, d in sorted(impossible.items())[:8]:
        print(f"      {n:24} drafted {d['earliest_draft_year']}, seasons from "
              f"{d['span'][0]}  ({len(d['seasons_before_draft'])} impossible)")
    print(f"  AMBIGUOUS  (>1 draft entry for the name) : {len(ambiguous)}")
    print(f"  REVIEW     (gap only, NOT excluded)      : {len(review)}")
    print(f"gridiron definitively-merged names          : {len(gmerged)}")

    if contaminated:
        print(f"\n{sum(len(v) for v in contaminated.values())} contaminated row(s) in "
              f"{len(contaminated)} artifact(s):")
        for fn, rows in contaminated.items():
            for r in rows:
                print(f"  {fn}: {r}")
        print("\nThese treat two people as one career. Exclude them or supply a DOB key.")
        return 1 if args.check else 0
    print("\nno axis artifact carries a definitively-merged career.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
