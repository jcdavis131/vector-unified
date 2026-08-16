#!/usr/bin/env python3
"""Acquire the free tennis corpus and MEASURE it — coverage before modelling.

Solo personal project, no connection to employer, built with public/free-tier only

Follows `acquire_sponsors.py`'s precedent in this repo: a first step that ACQUIRES AND
MEASURES and deliberately trains nothing. Phase 7 spent its length learning that a coverage
number decides whether the next step is worth taking, and that finding it out afterwards is
how survivor-biased denominators and star-only samples get shipped.

WHY NOT THE OBVIOUS SOURCE. `JeffSackmann/tennis_atp` and `tennis_wta` are the canonical
free tennis datasets and every modelling guide on the internet still points at them. They
are **gone**: both repos 404 on `main` and `master`, and `api.github.com/users/JeffSackmann`
reports `public_repos=1`. Measured 2026-08-03. Anyone planning around them is planning
around something that no longer exists.

tennis-data.co.uk is the viable free path. Per-season xlsx, men and women, 2013-2026.

ROBOTS.TXT IS RESPECTED AND THE CHECK IS IN THE CODE, not in a promise. The site disallows
`/stuff/` and `/2000/` through `/2005/`. Everything this fetches is 2013+, which is allowed,
and DISALLOWED_PREFIXES below encodes the rule so a future widening of the year range trips
it instead of quietly crawling a forbidden path.

WHAT MAKES THIS SPORT WORTH ADDING, and it is not the reason the expansion was proposed.
`docs/TENNIS_GOLF_FEASIBILITY.md` measured the sponsor rationale and it fails: Wikidata
records P859 on **6 of 15,896 tennis players**. What tennis actually brings:

  * NO TEAM CONFOUND. The pitch P0/P1 axis had to control for team strength because it
    dominated the signal (+0.2626 against -0.1671). A tennis result is the athlete's own.
  * TWO MARKET-LIKE PRIORS, which is more than any current member has. `WRank`/`LRank` are
    the ATP/WTA ranking at match time — a public valuation made BEFORE the match. `AvgW`/
    `AvgL` are closing odds — literally what the market priced. Both are far closer to a
    draft slot than pitch's age curve, which is why that one is `P0/P1` and not `T0/T1`.
  * LOCATION per tournament, in the data file rather than in Wikidata, where the same
    feasibility check found location on 12% of tennis tournaments and 7 golf tournaments.

    python pipeline/acquire_tennis.py            # fetch missing seasons, then report
    python pipeline/acquire_tennis.py --offline  # re-report from cache, no network
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "pipeline" / "cache" / "tennis"
OUT = ROOT / "data" / "tennis_coverage.json"

BASE = "http://www.tennis-data.co.uk"
UA = "vector-unified/0.1 (personal research; contact via github)"
YEARS = range(2013, 2027)
SLEEP = 1.5  # polite pacing; this is one person's static file host
DISALLOWED_PREFIXES = ("/stuff/",) + tuple(f"/{y}/" for y in range(2000, 2006))

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# The columns that decide whether this corpus can carry an R0/R1 axis at all. Named here
# so a silent schema change shows up as a coverage drop against a stated expectation
# rather than as a KeyError three scripts downstream.
PRIOR_COLS = ("WRank", "LRank", "WPts", "LPts")
MARKET_COLS = ("AvgW", "AvgL")
CONTEXT_COLS = ("Location", "Tournament", "Surface", "Series", "Round")


def url_for(year: int, women: bool) -> str:
    return f"{BASE}/{year}{'w' if women else ''}/{year}.xlsx"


def path_for(year: int, women: bool) -> Path:
    return CACHE / f"{year}{'w' if women else ''}.xlsx"


def fetch(year: int, women: bool) -> tuple[bool, str]:
    u = url_for(year, women)
    tail = u[len(BASE) :]
    if any(tail.startswith(p) for p in DISALLOWED_PREFIXES):
        return False, f"robots.txt disallows {tail} — not fetched"
    p = path_for(year, women)
    if p.exists() and p.stat().st_size > 0:
        return True, "cached"
    try:
        r = requests.get(u, headers={"User-Agent": UA}, timeout=120)
    except Exception as e:
        return False, f"error {str(e)[:60]}"
    if r.status_code != 200 or not r.content.startswith(b"PK"):
        return False, f"HTTP {r.status_code}, {len(r.content)}B"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(r.content)
    time.sleep(SLEEP)
    return True, f"fetched {len(r.content)}B"


def read_sheet(p: Path) -> tuple[list[str], list[list[str]]]:
    """xlsx -> (header, rows), stdlib only.

    Deliberately not pandas+openpyxl: openpyxl is not installed in the CUDA venv this
    estate uses, and adding a dependency to read one static spreadsheet per season is a
    worse trade than forty lines of ElementTree.
    """
    z = zipfile.ZipFile(p)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    sheet = next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))
    rows_xml = ET.fromstring(z.read(sheet)).findall(f".//{NS}row")

    def val(c) -> str:
        v = c.find(f"{NS}v")
        if v is None or v.text is None:
            return ""
        if c.get("t") == "s" and v.text.isdigit():
            return shared[int(v.text)]
        return v.text

    def cells(row) -> list[str]:
        # A blank cell is OMITTED from the XML, not written empty, so positions must come
        # from the r="B7" reference. Zipping raw children against the header silently
        # shifts every column after the first gap — the same class of index-drift the
        # unified matrix's mask bug came from.
        out: list[str] = []
        for c in row:
            ref = c.get("r") or ""
            letters = "".join(ch for ch in ref if ch.isalpha())
            idx = 0
            for ch in letters:
                idx = idx * 26 + (ord(ch) - 64)
            idx -= 1
            while len(out) < idx:
                out.append("")
            out.append(val(c))
        return out

    if not rows_xml:
        return [], []
    header = cells(rows_xml[0])
    body = []
    for r in rows_xml[1:]:
        row = cells(r)
        while len(row) < len(header):
            row.append("")
        body.append(row)
    return header, body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="re-report from cache only")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    seasons = []
    for women in (False, True):
        for y in YEARS:
            if args.offline:
                ok = path_for(y, women).exists()
                note = "cached" if ok else "absent"
            else:
                ok, note = fetch(y, women)
            seasons.append({"year": y, "tour": "wta" if women else "atp", "ok": ok, "note": note})
            print(
                f"  {y}{'w' if women else ' '} {'wta' if women else 'atp'}  {note}",
                flush=True,
            )

    per_season = []
    col_present: collections.Counter = collections.Counter()
    total_rows = 0
    have_prior = have_market = 0
    surfaces: collections.Counter = collections.Counter()
    locations: set[str] = set()
    for s in seasons:
        if not s["ok"]:
            continue
        p = path_for(s["year"], s["tour"] == "wta")
        try:
            hdr, body = read_sheet(p)
        except Exception as e:
            per_season.append({**s, "error": str(e)[:80]})
            continue
        idx = {c: i for i, c in enumerate(hdr)}
        for c in hdr:
            col_present[c] += 1
        n = len(body)
        total_rows += n

        def filled(cols):
            if not all(c in idx for c in cols):
                return 0
            return sum(1 for r in body if all(str(r[idx[c]]).strip() for c in cols))

        pri, mkt = filled(PRIOR_COLS), filled(MARKET_COLS)
        have_prior += pri
        have_market += mkt
        if "Surface" in idx:
            for r in body:
                surfaces[str(r[idx["Surface"]]).strip()] += 1
        if "Location" in idx:
            locations |= {str(r[idx["Location"]]).strip() for r in body}
        per_season.append(
            {
                **s,
                "rows": n,
                "cols": len(hdr),
                "rows_with_full_prior": pri,
                "rows_with_closing_odds": mkt,
            }
        )

    got = [s for s in per_season if s.get("rows")]
    report = {
        "source": f"{BASE} — free per-season xlsx, no key (see the 7.9 no-paid-API rule)",
        "robots_respected": (
            "Disallow list is /stuff/ and /2000/-/2005/. Every year fetched is 2013+, and "
            "DISALLOWED_PREFIXES encodes the rule so widening the range trips it."
        ),
        "sackmann_note": (
            "JeffSackmann/tennis_atp and tennis_wta 404 on main and master; "
            "api.github.com/users/JeffSackmann reports public_repos=1. The canonical free "
            "tennis corpus is no longer published. Measured 2026-08-03."
        ),
        "seasons_requested": len(seasons),
        "seasons_present": len(got),
        "total_matches": total_rows,
        "rows_with_full_ranking_prior": have_prior,
        "pct_with_ranking_prior": round(100.0 * have_prior / max(total_rows, 1), 1),
        "rows_with_closing_odds": have_market,
        "pct_with_closing_odds": round(100.0 * have_market / max(total_rows, 1), 1),
        "prior_cols": list(PRIOR_COLS),
        "market_cols": list(MARKET_COLS),
        "context_cols_present_in_n_seasons": {c: col_present.get(c, 0) for c in CONTEXT_COLS},
        "distinct_locations": len(locations),
        "surface_mix": dict(surfaces.most_common()),
        "per_season": per_season,
        "what_this_does_not_do": (
            "No model, no axis, no join to the unified corpus. This answers one question — "
            "does the free tennis corpus carry a usable pre-match prior and enough context "
            "columns to be worth a tower — and it answers it before anything is built on "
            "the answer."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nseasons {len(got)}/{len(seasons)}   matches {total_rows}")
    print(f"ranking prior (WRank/LRank/WPts/LPts all present): {have_prior} " f"({report['pct_with_ranking_prior']}%)")
    print(f"closing odds  (AvgW/AvgL present)                : {have_market} " f"({report['pct_with_closing_odds']}%)")
    print(f"distinct locations {len(locations)}   surfaces {dict(surfaces.most_common(5))}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
