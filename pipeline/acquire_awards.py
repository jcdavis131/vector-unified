"""Vector Unified — Phase 1d market/cultural: cross-sport awards -> AWARD_PRESTIGE.

Pulls the top individual award winners from Wikipedia for each of our three
sports, then builds a career AWARD_PRESTIGE score (tier-weighted count of wins).

Sources (Wikipedia ?action=render):
  hoops:     NBA Most Valuable Player Award          (tier 1.0)
  gridiron:  List of NFL Most Valuable Player awards (AP column, tier 1.0)
  pitch:     Ballon d'Or                             (tier 1.0)
             FIFA World Player of the Year / The Best FIFA Men's Player (tier 0.9)

Tier weights are sport-agnostic prestige units — an NBA MVP, AP NFL MVP, and
Ballon d'Or are treated as equivalent top-of-sport honors. Secondary awards
(All-NBA / All-Pro / FIFA FIFPro XI) are deferred; hoops already has honors.json
for All-NBA vote lag.

Output: data/market_cultural/awards.json
  { built, sources, winners: [ {sport, award, year, name, norm, tier} ... ],
    prestige: { sport: { norm: {name, wins, prestige, by_award: {...}} } } }

Cross-ref vs assets/unified.json -> data/market_cultural/awards_coverage.json.

Run:  python pipeline/acquire_awards.py [--offline] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_forbes import fetch_html, norm_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_cultural"
ASSETS = ROOT / "assets"
CACHE = ROOT / "pipeline" / "cache"

# (sport, award_key, wiki_title, tier, parser_kind)
SOURCES = [
    ("hoops", "nba_mvp", "NBA_Most_Valuable_Player_Award", 1.0, "nba_mvp"),
    (
        "gridiron",
        "ap_nfl_mvp",
        "List_of_NFL_Most_Valuable_Player_awards",
        1.0,
        "nfl_mvp",
    ),
    ("pitch", "ballon_dor", "Ballon_d%27Or", 1.0, "ballon"),
    ("pitch", "fifa_best", "The_Best_FIFA_Men%27s_Player", 0.9, "fifa_best"),
    ("pitch", "fifa_world_player", "FIFA_World_Player_of_the_Year", 0.9, "fifa_wpoty"),
]


class _Wikitable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._t = None
        self._r = None
        self._c: list[str] = []
        self._in = False
        self._tag = ""

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == "table" and "wikitable" in attrs_d.get("class", ""):
            self._t = []
        elif self._t is not None and tag == "tr":
            self._r = []
        elif self._r is not None and tag in ("td", "th"):
            self._in = True
            self._tag = tag
            self._c = []

    def handle_endtag(self, tag):
        if self._in and tag == self._tag:
            text = re.sub(r"\s+", " ", "".join(self._c)).strip()
            text = re.sub(r"\[\d+\]", "", text).strip()
            self._r.append(text)
            self._in = False
        elif self._t is not None and tag == "tr" and self._r is not None:
            if self._r:
                self._t.append(self._r)
            self._r = None
        elif tag == "table" and self._t is not None:
            self.tables.append(self._t)
            self._t = None

    def handle_data(self, data):
        if self._in:
            self._c.append(data)


def _strip_player(cell: str) -> str:
    """'LeBron James^ (4)' / 'Peyton Manning*' -> clean name."""
    s = cell.strip()
    s = re.sub(r"\s*[\^*†‡§].*$", "", s)  # footnote markers
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)  # (N) win count
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _year_from_season(cell: str) -> int | None:
    """'2023–24' / '2023-24' / '2024' -> end year (award year)."""
    m = re.search(r"(\d{4})\s*[–-]\s*(\d{2,4})", cell)
    if m:
        y1 = int(m.group(1))
        y2s = m.group(2)
        y2 = int(y2s) if len(y2s) == 4 else 2000 + int(y2s) if int(y2s) < 100 else int(y2s)
        # handle 1999-00 -> 2000
        if y2 < y1:
            y2 += 100 if y2 < 100 else 0
        return y2
    m = re.search(r"(\d{4})", cell)
    return int(m.group(1)) if m else None


def parse_nba_mvp(tables: list[list[list[str]]]) -> list[dict]:
    """Winners table: Season | Player | Position | Nationality | Team."""
    out = []
    for tbl in tables:
        if not tbl:
            continue
        header = [c.lower() for c in tbl[0]]
        if not any("season" in h for h in header):
            continue
        if not any("player" in h for h in header):
            continue
        # find col indices
        si = next(i for i, h in enumerate(header) if "season" in h)
        pi = next(i for i, h in enumerate(header) if "player" in h)
        for row in tbl[1:]:
            if len(row) <= max(si, pi):
                continue
            year = _year_from_season(row[si])
            name = _strip_player(row[pi])
            if year and name and not name.lower().startswith("season"):
                out.append({"year": year, "name": name})
        if out:
            break
    return out


def parse_nfl_mvp(tables: list[list[list[str]]]) -> list[dict]:
    """Multi-org table: Year | AP | PFWA | ... — take AP column."""
    out = []
    for tbl in tables:
        if not tbl or len(tbl[0]) < 2:
            continue
        header = [c.lower() for c in tbl[0]]
        if "year" not in header[0] and not any("year" in h for h in header):
            continue
        # prefer AP column
        ap_i = next(
            (i for i, h in enumerate(header) if h.strip() in ("ap", "associated press")),
            None,
        )
        if ap_i is None:
            # sometimes header is just "AP"
            ap_i = next(
                (i for i, h in enumerate(header) if "ap" == h.strip() or h.strip().startswith("ap ")),
                1,
            )
        yi = 0
        for row in tbl[1:]:
            if len(row) <= max(yi, ap_i):
                continue
            year = _year_from_season(row[yi])
            name = _strip_player(row[ap_i])
            if not year or not name or name in ("—", "-", "–", ""):
                continue
            # skip shared awards like "A and B" — take first name
            if " and " in name.lower():
                name = re.split(r"\s+and\s+", name, flags=re.IGNORECASE)[0].strip()
            out.append({"year": year, "name": name})
        if out:
            break
    return out


def parse_ballon(tables: list[list[list[str]]]) -> list[dict]:
    """Ballon d'Or winners: Year | Player | Club | Nationality (layouts vary)."""
    out = []
    for tbl in tables:
        if not tbl:
            continue
        header = [c.lower() for c in tbl[0]]
        if not any("year" in h or "season" in h for h in header):
            continue
        if not any("player" in h or "winner" in h for h in header):
            continue
        yi = next(i for i, h in enumerate(header) if "year" in h or "season" in h)
        pi = next(i for i, h in enumerate(header) if "player" in h or "winner" in h)
        for row in tbl[1:]:
            if len(row) <= max(yi, pi):
                continue
            year = _year_from_season(row[yi])
            name = _strip_player(row[pi])
            if year and name and year >= 1990:  # corpus era
                out.append({"year": year, "name": name})
        if len(out) >= 10:
            break
    return out


def parse_fifa(tables: list[list[list[str]]]) -> list[dict]:
    """FIFA World Player / The Best: Year | Player | ..."""
    return parse_ballon(tables)  # same column shape


PARSERS = {
    "nba_mvp": parse_nba_mvp,
    "nfl_mvp": parse_nfl_mvp,
    "ballon": parse_ballon,
    "fifa_best": parse_fifa,
    "fifa_wpoty": parse_fifa,
}


def fetch_source(sport: str, award: str, title: str, tier: float, kind: str, refresh: bool) -> list[dict]:
    cache = CACHE / f"awards_{award}.html"
    url = f"https://en.wikipedia.org/wiki/{title}?action=render"
    if cache.exists() and not refresh:
        html = cache.read_text(encoding="utf-8")
        fetched = False
    else:
        html = fetch_html(url)
        cache.write_text(html, encoding="utf-8")
        fetched = True
        time.sleep(2.0)
    ext = _Wikitable()
    ext.feed(html)
    rows = PARSERS[kind](ext.tables)
    return [
        {
            "sport": sport,
            "award": award,
            "year": r["year"],
            "name": r["name"],
            "norm": norm_name(r["name"]),
            "tier": tier,
        }
        for r in rows
    ], fetched


def build_prestige(winners: list[dict]) -> dict:
    """sport -> norm -> {name, wins, prestige, by_award}."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for w in winners:
        slot = out[w["sport"]].setdefault(
            w["norm"],
            {
                "name": w["name"],
                "wins": 0,
                "prestige": 0.0,
                "by_award": {},
            },
        )
        slot["wins"] += 1
        slot["prestige"] = round(slot["prestige"] + w["tier"], 2)
        slot["by_award"][w["award"]] = slot["by_award"].get(w["award"], 0) + 1
        # keep the most recent display name
        slot["name"] = w["name"]
    return {s: dict(v) for s, v in out.items()}


def cross_ref(prestige: dict) -> dict:
    upath = ASSETS / "unified.json"
    if not upath.exists():
        return {"present": False}
    U = json.loads(upath.read_text(encoding="utf-8"))
    idx: dict[tuple[str, str], int] = {}
    for p in U["players"]:
        idx[(norm_name(p["name"]), p["sport"])] = idx.get((norm_name(p["name"]), p["sport"]), 0) + 1
    cov = {}
    for sport, athletes in prestige.items():
        matched = []
        missed = []
        for nn, rec in athletes.items():
            if (nn, sport) in idx:
                matched.append(
                    {
                        "name": rec["name"],
                        "prestige": rec["prestige"],
                        "wins": rec["wins"],
                        "unified_rows": idx[(nn, sport)],
                    }
                )
            else:
                missed.append(
                    {
                        "name": rec["name"],
                        "prestige": rec["prestige"],
                        "wins": rec["wins"],
                    }
                )
        cov[sport] = {
            "n_award_winners": len(athletes),
            "matched": len(matched),
            "missed": len(missed),
            "matched_examples": sorted(matched, key=lambda x: -x["prestige"])[:8],
            "missed_examples": sorted(missed, key=lambda x: -x["prestige"])[:8],
        }
    return {"present": True, "by_sport": cov}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    out_path = DATA / "awards.json"
    if args.offline and out_path.exists() and not args.refresh:
        doc = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"[offline] awards.json: {len(doc['winners'])} winner-records")
    else:
        winners: list[dict] = []
        for sport, award, title, tier, kind in SOURCES:
            try:
                rows, fetched = fetch_source(sport, award, title, tier, kind, args.refresh)
                tag = "fetched" if fetched else "cached"
                print(f"  {award:20s} [{sport:8s}] {tag}: {len(rows)} winners")
                winners.extend(rows)
            except Exception as e:
                print(f"  {award:20s} FAILED ({type(e).__name__}: {e})")
        # de-dupe identical (sport, award, year, norm) — FIFA Best + WPOTY can overlap eras
        seen = set()
        uniq = []
        for w in winners:
            key = (w["sport"], w["award"], w["year"], w["norm"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(w)
        prestige = build_prestige(uniq)
        doc = {
            "built": time.strftime("%Y-%m-%d"),
            "sources": [{"sport": s, "award": a, "wiki": t, "tier": tr} for s, a, t, tr, _ in SOURCES],
            "n_winner_records": len(uniq),
            "winners": uniq,
            "prestige": prestige,
        }
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"\nsaved awards.json: {len(uniq)} winner-records, "
            f"{sum(len(v) for v in prestige.values())} unique athletes"
        )

    cov = cross_ref(doc["prestige"])
    (DATA / "awards_coverage.json").write_text(json.dumps(cov, indent=2, ensure_ascii=False), encoding="utf-8")
    if cov.get("present"):
        print("\ncross-ref vs unified.json:")
        for sport, c in cov["by_sport"].items():
            print(f"  {sport:9s} matched {c['matched']}/{c['n_award_winners']} award-winners")
            for e in c["matched_examples"][:4]:
                print(f"    {e['name']:<22} prestige={e['prestige']} wins={e['wins']} " f"rows={e['unified_rows']}")
            if c["missed_examples"]:
                print(
                    "    missed (not in corpus): "
                    + ", ".join(f"{m['name']}({m['prestige']})" for m in c["missed_examples"][:4])
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
