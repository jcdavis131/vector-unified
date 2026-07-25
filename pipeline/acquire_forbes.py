"""Vector Unified — Phase 1a market/cultural acquisition: Forbes highest-paid athletes.

Pulls the cross-sport elite earnings anchor from Wikipedia's
"Forbes list of the world's highest-paid athletes" (yearly tables 2012-2026),
which splits each athlete's earnings into on-field (salary/winnings) and off-field
(endorsements). This is the single cleanest cross-sport $ anchor: same units
(dollars), explicit sport label, explicit salary vs endorsement split, ~10
athletes/year across all sports.

Output: data/market_cultural/forbes_earnings.json
  { built, source, years, lists: { year: [ {rank, name, norm, sport_raw, sport,
     country, total_m, salary_m, endorse_m} ... ] } }

sport mapping: Basketball -> hoops, American football -> gridiron,
  Association football -> pitch, everything else -> other (out of our 3 sports).

Cross-ref: loads assets/unified.json, builds (norm, sport) -> row-indices index,
reports how many Forbes athletes land in the unified corpus (per sport, total).
Writes data/market_cultural/forbes_coverage.json.

Run:  python pipeline/acquire_forbes.py [--offline] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "pipeline"
DATA = ROOT / "data" / "market_cultural"
ASSETS = ROOT / "assets"
CACHE = PIPE / "cache"

WIKI_RENDER = ("https://en.wikipedia.org/wiki/"
               "Forbes_list_of_the_world%27s_highest-paid_athletes?action=render")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

SPORT_MAP = {
    "basketball": "hoops",
    "american football": "gridiron",
    "association football": "pitch",
    "soccer": "pitch",
    "football": "pitch",  # cautious default; refined below if needed
}


def norm_name(name: str) -> str:
    """Match hoops fetch_honors.norm_name for cross-sport joins."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def fetch_html(url: str) -> str:
    try:
        from curl_cffi import requests as cr
        r = cr.get(url, impersonate="chrome120", headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        return r.text
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace")


def _strip_tags(s: str) -> str:
    """Remove HTML tags, collapse whitespace, drop footnote refs like [1]."""
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&amp;", "&").replace("&nbsp;", " ")
          .replace("&#160;", " ").replace("&quot;", '"')
          .replace("&#39;", "'").replace("&apos;", "'"))
    s = re.sub(r"\s*\[\d+\]\s*", " ", s)  # wikipedia reference markers
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_money(cell: str) -> float | None:
    """'$300 million' / '$137.8 million' / '$0' / '$0.03 million' -> float (millions)."""
    if not cell:
        return None
    low = cell.lower().replace(",", "")
    if "n/a" in low or low in ("—", "-", "–", ""):
        return None
    m = re.search(r"[\d.]+", low.replace("$", ""))
    if not m:
        return None
    val = float(m.group(0))
    # Forbes tables are uniformly in millions; "million"/"m" is the unit, not a multiplier
    return val


class _TableTextExtractor(HTMLParser):
    """Collect a list of tables; each table is a list of rows; each row is a list of cell-texts."""

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._cur_table: list[list[str]] | None = None
        self._cur_row: list[str] | None = None
        self._cur_cell: list[str] = []
        self._in_cell = False
        self._cell_tag = ""
        # heading positions in the raw html for table->year association
        self._tag_path: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")
        if tag == "table" and "wikitable" in cls:
            self._cur_table = []
        elif self._cur_table is not None and tag == "tr":
            self._cur_row = []
        elif self._cur_row is not None and tag in ("td", "th"):
            self._in_cell = True
            self._cell_tag = tag
            self._cur_cell = []

    def handle_endtag(self, tag):
        if self._in_cell and tag == self._cell_tag:
            self._cur_row.append(_strip_tags("".join(self._cur_cell)))
            self._in_cell = False
        elif self._cur_table is not None and tag == "tr" and self._cur_row is not None:
            if self._cur_row:
                self._cur_table.append(self._cur_row)
            self._cur_row = None
        elif tag == "table" and self._cur_table is not None:
            self.tables.append(self._cur_table)
            self._cur_table = None

    def handle_data(self, data):
        if self._in_cell:
            self._cur_cell.append(data)


def _year_from_heading(html: str, table_start: int) -> int | None:
    """Find the nearest preceding heading before a table's start; return its 4-digit year."""
    chunk = html[:table_start]
    # iterate heading matches from the end backwards
    heads = list(re.finditer(
        r"<h[23][^>]*>(.*?)</h[23]>", chunk, re.DOTALL | re.IGNORECASE))
    for h in reversed(heads):
        text = _strip_tags(h.group(1))
        m = re.search(r"(\d{4})", text)
        if m:
            year = int(m.group(1))
            # skip the decade-aggregate "2010-2019" heading (contains an en/em dash range)
            if re.search(r"\d{4}\s*[–-]\s*\d{4}", text):
                return None
            return year
    return None


def parse_forbes_html(html: str) -> dict[int, list[dict]]:
    """Return {year: [record, ...]} from the rendered Wikipedia HTML."""
    # locate table start offsets for heading association
    table_spans = [(m.start(), m.end()) for m in re.finditer(
        r"<table[^>]*class=\"[^\"]*wikitable[^\"]*\"[^>]*>.*?</table>",
        html, re.DOTALL | re.IGNORECASE)]
    ext = _TableTextExtractor()
    ext.feed(html)
    if not ext.tables:
        return {}
    # map each parsed table to its span by order
    out: dict[int, list[dict]] = {}
    for idx, tbl in enumerate(ext.tables):
        if idx >= len(table_spans):
            break
        year = _year_from_heading(html, table_spans[idx][0])
        if year is None:
            continue  # decade aggregate or no year -> skip
        # find header row to confirm column layout (7 cols expected)
        if not tbl:
            continue
        header = tbl[0]
        # column position semantics: rank, name, sport, country, total, salary, endorse
        # accept any header; we parse by position. skip rows where col0 != integer rank.
        records = []
        for row in tbl[1:]:
            if len(row) < 7:
                continue
            rank_s = row[0].strip()
            if not re.fullmatch(r"\d+", rank_s):
                continue  # header repeat or non-data
            name = row[1].strip()
            if not name:
                continue
            sport_raw = row[2].strip()
            country = row[3].strip()
            total = _parse_money(row[4])
            salary = _parse_money(row[5])
            endorse = _parse_money(row[6])
            sport = SPORT_MAP.get(sport_raw.lower(), "other")
            records.append({
                "rank": int(rank_s),
                "name": name,
                "norm": norm_name(name),
                "sport_raw": sport_raw,
                "sport": sport,
                "country": country,
                "total_m": total,
                "salary_m": salary,
                "endorse_m": endorse,
            })
        if records:
            out.setdefault(year, []).extend(records)
    return out


def build_forbes() -> dict:
    html = fetch_html(WIKI_RENDER)
    lists = parse_forbes_html(html)
    years = sorted(lists.keys())
    return {
        "built": time.strftime("%Y-%m-%d"),
        "source": "en.wikipedia.org/wiki/Forbes_list_of_the_world's_highest-paid_athletes",
        "n_years": len(years),
        "years": years,
        "lists": {str(y): lists[y] for y in years},
    }


def cross_ref(doc: dict) -> dict:
    """Match Forbes athletes to unified.json (norm, sport) -> row indices."""
    upath = ASSETS / "unified.json"
    if not upath.exists():
        return {"unified_json": str(upath), "present": False}
    U = json.loads(upath.read_text(encoding="utf-8"))
    players = U["players"]
    idx: dict[tuple[str, str], list[int]] = {}
    for i, p in enumerate(players):
        idx.setdefault((norm_name(p["name"]), p["sport"]), []).append(i)

    by_sport = {"hoops": 0, "gridiron": 0, "pitch": 0, "other": 0}
    matched_records = 0
    total_records = 0
    matched_athletes: set[tuple[str, str]] = set()
    examples = []
    unmatched = []  # in-sport Forbes records with no unified.json (norm, sport) hit
    # also index unified by norm-only, to diagnose name variants across sports
    norm_any: dict[str, list[str]] = {}
    for p in players:
        norm_any.setdefault(norm_name(p["name"]), []).append(p["sport"])
    for year, recs in doc["lists"].items():
        for r in recs:
            total_records += 1
            if r["sport"] == "other":
                continue
            key = (r["norm"], r["sport"])
            if key in idx:
                by_sport[r["sport"]] += 1
                matched_records += 1
                matched_athletes.add(key)
                if len(examples) < 12:
                    examples.append({"year": int(year), "name": r["name"],
                                     "sport": r["sport"], "total_m": r["total_m"],
                                     "unified_rows": len(idx[key])})
            else:
                sports_with_name = norm_any.get(r["norm"], [])
                unmatched.append({"year": int(year), "name": r["name"],
                                  "sport": r["sport"], "sport_raw": r["sport_raw"],
                                  "name_found_in_sports": sorted(set(sports_with_name))})
    # unique athletes in our 3 sports
    unique_in3 = sum(1 for k in matched_athletes)
    return {
        "unified_json": str(upath),
        "present": True,
        "unified_rows": len(players),
        "forbes_records_total": total_records,
        "forbes_records_matched_in3": matched_records,
        "forbes_unique_athletes_matched_in3": unique_in3,
        "by_sport": by_sport,
        "examples": examples,
        "unmatched_in_sport": unmatched,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="use cached forbes_earnings.json; only run cross-ref")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch even if cache exists")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = DATA / "forbes_earnings.json"

    if args.offline and out.exists() and not args.refresh:
        doc = json.loads(out.read_text(encoding="utf-8"))
        print(f"[offline] forbes_earnings.json: {doc['n_years']} years cached")
    else:
        if out.exists() and not args.refresh:
            doc = json.loads(out.read_text(encoding="utf-8"))
            print(f"forbes_earnings.json cached (use --refresh to re-fetch): "
                  f"{doc['n_years']} years")
        else:
            print("fetching Wikipedia Forbes list ...")
            doc = build_forbes()
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            print(f"saved {out.name}: {doc['n_years']} years "
                  f"({sum(len(v) for v in doc['lists'].values())} records)")
            time.sleep(1.0)

    # summary
    by_sport = {"hoops": 0, "gridiron": 0, "pitch": 0, "other": 0}
    for recs in doc["lists"].values():
        for r in recs:
            by_sport[r["sport"]] = by_sport.get(r["sport"], 0) + 1
    print(f"records by sport: {by_sport}")

    # cross-ref vs unified corpus
    cov = cross_ref(doc)
    (DATA / "forbes_coverage.json").write_text(
        json.dumps(cov, indent=2, ensure_ascii=False), encoding="utf-8")
    if cov.get("present"):
        print(f"\ncross-ref vs unified.json ({cov['unified_rows']} rows):")
        print(f"  Forbes records in our 3 sports matched: "
              f"{cov['forbes_records_matched_in3']}/{cov['forbes_records_total']}  "
              f"({cov['forbes_unique_athletes_matched_in3']} unique athletes)")
        print(f"  by sport: {cov['by_sport']}")
        unmatched = cov.get("unmatched_in_sport", [])
        if unmatched:
            print(f"  unmatched in-sport records: {len(unmatched)} "
                  f"(name-variant or not in corpus):")
            for u in unmatched:
                tag = (f" [name exists as {u['name_found_in_sports']}]" if u["name_found_in_sports"] else " [name not in corpus]")
                print(f"    {u['year']} {u['name']:<22} [{u['sport'][:2]}] "
                      f"({u['sport_raw']}){tag}")
        print("  examples:")
        for e in cov["examples"][:8]:
            print(f"    {e['year']} {e['name']:<22} [{e['sport'][:2]}] "
                  f"total=${e['total_m']}M  unified_rows={e['unified_rows']}")
    else:
        print(f"unified.json not found at {cov['unified_json']} (cross-ref skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
