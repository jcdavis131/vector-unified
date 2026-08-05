"""Close the chain: athlete -> team -> venue -> city -> company -> named executives.

This is the thing the whole sponsor thread was for. Three scripts got here:

  acquire_venue_sponsors.py  does a joinable athlete<->company edge exist?
                             17 venues -> unique S&P 500 company, 15 correct (88.2%),
                             recall 8 of 11 on the closed NBA set.
  build_venue_edges.py       how many player-seasons does it reach?
                             1,033 gridiron + 576 hoops = 1,609, 13 tickers.
  this script                what can the chain actually answer?

EVERY HOP IS ALREADY MEASURED SEPARATELY. What was never checked is whether the hops
COMPOSE -- a chain is only as long as its narrowest join, and four hops each at "good"
coverage can still multiply out to nothing. So this reports per-hop survival, not a
single headline number, and the last hop is the one that could have killed it: all 13
bridge tickers turn out to have officer data, 13 of 13.

WHAT IT DOES NOT CLAIM. A venue-naming edge is a commercial relationship between a
company and a FRANCHISE. It is not a relationship between that company and any
individual athlete, and nothing here should be read as one. "Player X is connected to
JPMorgan" is false; "Player X played a season for a team whose arena carried JPMorgan's
brand" is true and much narrower. The report says which.

THE CEO FIELD. The first version called vector-equities/pipeline/officer_features.py
ceo_of() and printed whatever it returned alongside its ambiguity flag. For FDX that was
"Krishnasamy Sriram", whose title is "EVP CDI Off & CTO/CEO FDW" -- FedEx Dataworks. The
actual FedEx CEO, Subramaniam Rajesh, sits in the same row set as "President/CEO".
ceo_of() flagged the row ambiguous and was right to, but a field named `ceo_latest`
carrying a wrong name is read as the answer whether or not a flag sits beside it. That
is the State Farm -> State Street failure again: a confident-looking single value that
is wrong, where nothing about its shape says to check it.

So `ceo_latest` is now populated ONLY when the string determines it, and is null
otherwise with the candidates listed. See corporate_ceo() below. officer_features.py is
NOT modified -- it feeds trained NEO features in vector-equities, and editing it to fix
a display field in this repo would move a model input. Its verdict is carried alongside
in `equities_ceo_of_would_say` so the two can be compared rather than silently diverge.

    python pipeline/build_bridge_index.py

Reads:  data/venue_edges.json
        vector-equities/pipeline/data/officers.json
        vector-equities/pipeline/data/universe.json
Writes: data/bridge_index.json
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_paths import ESTATE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "data" / "venue_edges.json"
EQ = ESTATE / "vector-equities/pipeline/data"
OFFICERS = EQ / "officers.json"
UNIVERSE = EQ / "universe.json"
OFFICER_FEATURES = EQ.parent / "officer_features.py"
OUT = ROOT / "data" / "bridge_index.json"


# A CORPORATE CEO'S TITLE CONTAINS ONLY ROLE WORDS. A divisional CEO's title names the
# division. That is the whole discriminator, and it needs no list of division names.
#
# officer_features.ceo_of() flags FDX ambiguous and returns Krishnasamy Sriram, whose
# title is "EVP CDI Off & CTO/CEO FDW" -- FedEx Dataworks. The actual FedEx CEO,
# Subramaniam Rajesh, is in the same row set as "President/CEO". ceo_of()'s DIVISIONAL
# list ("insur gr", "reinsur", " group ", "division", ...) was written against insurers
# and does not know "FDW" or "Airline FEC". Extending that list here would be tuning it
# against the cases it judges, and worse, officer_features.py feeds trained NEO features
# in vector-equities -- changing it to fix a display field in this repo would move a
# model input. So the rule lives here and equities is untouched.
#
# This is the SAME rule as the FedEx/FedEx Freight tiebreak in acquire_venue_sponsors.py:
# when several candidates survive, prefer the one the string determines exactly, and
# refuse when nothing does. It resolves JPM (DIMON "Chairman & CEO" over Erdoes "CEO
# Asset & Wealth Management") and FDX, and it declines to invent one for FISV and TGT,
# whose latest filing has no CEO-role row at all.
ROLE_WORDS = {
    "chairman", "chairwoman", "chair", "president", "ceo", "chief", "executive",
    "officer", "co", "and", "the", "of", "&", "-", "director", "founder", "interim",
}


def corporate_ceo(rows):
    """(name, title, how) picking the CEO whose title names no division.

    how is one of: exact, same_person_two_titles, sole_ceo_row (a name is
    returned); transition_or_multiple, all_candidates_divisional, absent (None is).
    Kept identical to vector-equities/pipeline/audit_ceo_resolution.py -- two copies of
    a rule that drift apart is how two numbers that should agree stop agreeing."""
    ceos = [r for r in rows if (r.get("role") or "") == "CEO"]
    if not ceos:
        return None, None, "absent"
    pure = []
    for r in ceos:
        toks = [t for t in re.split(r"[^A-Za-z&\-]+", str(r.get("title", "")).lower())
                if t]
        if toks and all(t in ROLE_WORDS for t in toks):
            pure.append(r)
    if len(pure) == 1:
        return pure[0]["name"], pure[0]["title"], "exact"
    if len(pure) > 1:
        # Several pure-role rows naming ONE person is a title-spelling artifact, not
        # ambiguity: "Chair and CEO" plus "Chairman and CEO" for one human. Refusing
        # those cost 319 ticker-years in the audit. Several pure-role rows naming
        # DIFFERENT people is a CEO TRANSITION -- ACN_2019 carries Nanterme (died in
        # January), Rowland (interim) and Sweet (from September), all real -- and there
        # the refusal is the correct answer, because the year has no single CEO and
        # picking one erases the succession.
        names = {r["name"].strip().upper() for r in pure}
        if len(names) == 1:
            return pure[0]["name"], pure[0]["title"], "same_person_two_titles"
        return None, None, "transition_or_multiple"
    if len(ceos) == 1:
        return ceos[0]["name"], ceos[0]["title"], "sole_ceo_row"
    return None, None, "all_candidates_divisional"


def load_ceo_of():
    """Import ceo_of from vector-equities rather than reimplementing it."""
    if not OFFICER_FEATURES.exists():
        return None
    spec = importlib.util.spec_from_file_location("officer_features", OFFICER_FEATURES)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, "ceo_of", None)


def main() -> int:
    for p in (EDGES, OFFICERS, UNIVERSE):
        if not p.exists():
            print(f"FAIL: missing {p}", file=sys.stderr)
            return 2

    ed = json.loads(EDGES.read_text(encoding="utf-8"))
    officers = json.loads(OFFICERS.read_text(encoding="utf-8"))
    universe = {r["ticker"]: r for r in json.loads(UNIVERSE.read_text(encoding="utf-8"))}
    ceo_of = load_ceo_of()

    all_edges = [dict(e, sport="gridiron") for e in ed["edges"]]
    all_edges += [dict(e, sport="hoops") for e in ed.get("hoops", {}).get("edges", [])]

    by_ticker = defaultdict(list)
    for e in all_edges:
        by_ticker[e["ticker"]].append(e)

    # officers.json is keyed TICKER_YEAR; collect the years available per bridge ticker
    years_for = defaultdict(list)
    for k in officers:
        t, _, y = k.rpartition("_")
        if t in by_ticker and y.isdigit():
            years_for[t].append(int(y))

    rows = []
    for tk, edges in sorted(by_ticker.items()):
        yrs = sorted(years_for.get(tk, []))
        latest = yrs[-1] if yrs else None
        offs = officers.get(f"{tk}_{latest}", []) if latest else []
        ceo_name, ceo_title, how = corporate_ceo(offs)
        eq_name = eq_unambig = None
        if ceo_of and offs:
            eq_name, _eq_title, eq_unambig = ceo_of(offs)
        by_role = defaultdict(list)
        for o in offs:
            by_role[o.get("role") or "UNLABELLED"].append(o["name"])
        rows.append({
            "ticker": tk,
            "company": universe.get(tk, {}).get("company"),
            "sector": universe.get(tk, {}).get("sector"),
            "venues": sorted({e["venue"] for e in edges}),
            "cities": sorted({e["location"] for e in edges if e.get("location")}),
            "teams": sorted({f"{e['sport']}:{e['team_code']}" for e in edges}),
            "player_seasons_while_named": sum(
                e["player_seasons_while_named"] for e in edges),
            "officer_years": [yrs[0], yrs[-1]] if yrs else None,
            "officers_in_latest_year": len(offs),
            "ceo_latest": ceo_name,
            "ceo_title": ceo_title,
            "ceo_resolution": how,
            "ceo_candidates_if_ambiguous": (
                [f"{o['name']} | {o['title']}" for o in offs
                 if (o.get("role") or "") == "CEO"]
                if how in ("transition_or_multiple", "all_candidates_divisional")
                else None),
            "equities_ceo_of_would_say": eq_name,
            "equities_ceo_of_unambiguous": eq_unambig,
            "named_roles_present": sorted(by_role),
        })

    n_ps = sum(r["player_seasons_while_named"] for r in rows)
    with_off = [r for r in rows if r["officers_in_latest_year"]]
    with_ceo = [r for r in rows if r["ceo_latest"]]

    out = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chain": "athlete(player-season) -> team -> venue -> city -> company(ticker) "
                 "-> named executives",
        "per_hop_survival": {
            "note": "A chain is only as long as its narrowest join. Four hops each at "
                    "'good' coverage can still multiply out to nothing, so every hop is "
                    "reported rather than a single headline.",
            "1_venues_resolved_to_a_unique_company": 17,
            "1_of_which_correct": 15,
            "2_venues_that_map_to_a_team_in_a_corpus": len(
                {e["venue"] for e in all_edges}),
            "3_player_seasons_reached": n_ps,
            "4_bridge_tickers": len(rows),
            "5_bridge_tickers_with_officer_data": len(with_off),
            "6_bridge_tickers_with_an_identified_CEO": len(with_ceo),
            "6_note": "identified = exactly one CEO-role title made of pure role words, or a sole CEO row. Never a guess among several.",
            "narrowest_hop": "hop 1 -> 2. Of 17 venues resolved to a company, only "
                             f"{len({e['venue'] for e in all_edges})} sit in a league "
                             "whose corpus this repo holds; the rest are NHL, defunct, "
                             "or foreign (Uber Arena is in Berlin).",
        },
        "what_this_does_not_claim": "A venue-naming edge is a commercial relationship "
            "between a company and a FRANCHISE, not between that company and any "
            "individual athlete. 'Player X is connected to JPMorgan' is false. 'Player X "
            "played a season for a team whose arena carried JPMorgan's brand' is true "
            "and is all this supports.",
        "ceo_field_provenance": "vector-equities/pipeline/officer_features.py ceo_of(), "
            "imported rather than reimplemented. That function already handles the "
            "divisional-CEO case -- JPM lists both a Chairman & CEO and a CEO of Asset & "
            "Wealth Management, and a naive 'first row with role CEO' picks whichever "
            "the parser emitted first. `ceo_unambiguous` carries its verdict.",
        "companies": rows,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"chain: {out['chain']}")
    for k, v in out["per_hop_survival"].items():
        if k[0].isdigit():
            print(f"  {k:<48} {v}")
    print()
    for r in rows:
        ceo = r["ceo_latest"] or f"({r['ceo_resolution']})"
        print(f"  {r['ticker']:<5} {str(r['company'])[:22]:<22} "
              f"{r['player_seasons_while_named']:>4} ps  "
              f"{','.join(r['teams']):<22} {ceo[:30]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
