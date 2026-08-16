"""Vector Unified — Forbes highest-paid athletes pull.

Fetches the Wikipedia article "Forbes list of the world's highest-paid athletes"
(via the MediaWiki action API, parsed HTML) and extracts every yearly table
(2012-2026) into data/market/Forbes_highest_paid.json with the salary/winnings vs
endorsements split, mapped to the unified sport taxonomy.

This is the STAR-tier cross-sport anchor: ~10 athletes/year, ~150 rows total, the
cleanest sport-agnostic $ signal (salary + endorsements in common USD units across
basketball / american football / association football / other sports).

Output rows: {year, rank, name, sport_forbes, sport_unified, out_of_corpus, country,
              total_musd, salary_musd, endorse_musd}
  sport_unified in {hoops, gridiron, pitch, other}; out_of_corpus=True for non-target sports.
  salary_musd/endorse_musd are None where the source table has no split (e.g. 2010-2019 decade table).

Re-run is idempotent (overwrites the JSON). No external auth; Wikipedia only needs a real UA.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "data" / "market"

API = (
    "https://en.wikipedia.org/w/api.php?action=parse&page="
    "Forbes_list_of_the_world%27s_highest-paid_athletes"
    "&format=json&prop=text&disabletoc=1&disableeditsection=1"
)
UA = "VectorUnifiedResearch/0.1 (athlete market/cultural signal research; local build)"

SPORT_MAP = {
    "basketball": "hoops",
    "american football": "gridiron",
    "association football": "pitch",
}


def parse_money(cell_text: str):
    """'$300 million' -> 300.0 ; '$0' -> 0.0 ; '—' / '' -> None."""
    s = cell_text.strip().replace("\u2014", "-")
    if not s or s == "-":
        return None
    m = re.search(r"\$\s*([\d.]+)\s*million", s, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"\$\s*([\d,.]+)", s)
    if m:
        return float(m.group(1).replace(",", "")) / 1e6  # raw $ -> millions
    if re.search(r"\b0\b", s):
        return 0.0
    return None


def norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip().lower())


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    MARKET.mkdir(parents=True, exist_ok=True)
    print("fetching Wikipedia Forbes list...")
    r = requests.get(API, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    html = r.json()["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    # walk elements in document order, tracking the current heading text
    cur_year = None
    for el in soup.find_all(["h2", "h3", "table"]):
        if el.name in ("h2", "h3"):
            txt = el.get_text(" ", strip=True)
            m = re.match(r"^(\d{4})\s*(?:–|-)?\s*\d{0,4}\s*list", txt, re.I)
            if m:
                cur_year = int(m.group(1))
            elif re.search(r"statistics|see also|references|external", txt, re.I):
                cur_year = None  # stop attributing tables to a year
            continue
        if el.name != "table" or cur_year is None:
            continue
        if "wikitable" not in (el.get("class") or []):
            continue
        # parse this year's table
        trs = el.find_all("tr")
        if not trs:
            continue
        header_cells = [norm_header(c.get_text(" ", strip=True)) for c in trs[0].find_all(["th", "td"])]

        # map column indices by header keyword
        def colidx(*keywords):
            for i, h in enumerate(header_cells):
                if any(k in h for k in keywords):
                    return i
            return None

        i_rank = colidx("rank")
        i_name = colidx("name")
        i_sport = colidx("sport")
        i_country = colidx("country", "nation", "nationality")
        i_total = colidx("total", "earnings")
        i_sal = colidx("salary", "winnings")
        i_end = colidx("endorse")
        for tr in trs[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 3:
                continue

            def cell(i):
                return cells[i].get_text(" ", strip=True) if (i is not None and i < len(cells)) else ""

            name = cell(i_name)
            if not name or name.lower() == "name":
                continue
            sport_forbes = cell(i_sport)
            sport_unified = SPORT_MAP.get(sport_forbes.lower(), "other")
            total = parse_money(cell(i_total))
            sal = parse_money(cell(i_sal)) if i_sal is not None else None
            end = parse_money(cell(i_end)) if i_end is not None else None
            # rank may be th or td
            rank_s = cell(i_rank) if i_rank is not None else ""
            rm = re.search(r"\d+", rank_s)
            rank = int(rm.group()) if rm else None
            rows.append(
                {
                    "year": cur_year,
                    "rank": rank,
                    "name": name,
                    "sport_forbes": sport_forbes,
                    "sport_unified": sport_unified,
                    "out_of_corpus": sport_unified == "other",
                    "country": cell(i_country),
                    "total_musd": total,
                    "salary_musd": sal,
                    "endorse_musd": end,
                }
            )

    # dedupe identical (year, name) rows (Wikipedia sometimes lists ties/footnotes twice)
    seen = set()
    deduped = []
    for r_ in rows:
        k = (r_["year"], r_["name"], r_["sport_forbes"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r_)
    rows = deduped

    by_sport = {}
    for r_ in rows:
        by_sport[r_["sport_unified"]] = by_sport.get(r_["sport_unified"], 0) + 1
    years = sorted({r_["year"] for r_ in rows})

    out = {
        "source": "Wikipedia: Forbes list of the world's highest-paid athletes",
        "fetched_via": "MediaWiki action=parse API",
        "n_rows": len(rows),
        "years": years,
        "by_sport_unified": by_sport,
        "note": (
            "STAR-tier cross-sport earnings anchor. ~10 athletes/year. "
            "salary_musd/endorse_musd are the on-field/off-field $ split (millions USD). "
            "out_of_corpus=True for sports outside hoops/gridiron/pitch (kept for context, "
            "not joined to the unified matrix)."
        ),
        "rows": rows,
    }
    (MARKET / "Forbes_highest_paid.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote {MARKET / 'Forbes_highest_paid.json'}: {len(rows)} rows across {len(years)} years ({years[0]}-{years[-1]})"
    )
    print("by sport_unified:", by_sport)
    # show the in-corpus stars
    inc = [r_ for r_ in rows if not r_["out_of_corpus"]]
    print(f"in-corpus star rows: {len(inc)} ({len({r_['name'] for r_ in inc})} unique players)")
    for s in ("hoops", "gridiron", "pitch"):
        names = sorted({r_["name"] for r_ in inc if r_["sport_unified"] == s})
        print(f"  {s:9s} ({len(names)} unique): {', '.join(names[:12])}{' ...' if len(names)>12 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
