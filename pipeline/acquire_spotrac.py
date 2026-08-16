"""Vector Unified — Phase 1b market/cultural acquisition: Spotrac NFL salaries.

Pulls per-player, per-year NFL cap hit from Spotrac's NFL Cap Hit Rankings pages
(https://www.spotrac.com/nfl/rankings/player/_/year/{YYYY}) for the gridiron
corpus years. Cap hit is a per-season on-field $ proxy -> SALARY_LOG for gridiron
(hoops already has BBREF salary; this completes the salary axis for 2 of 3 sports).

Spotrac is scrape-friendly (HTTP 200, server-rendered div-list, no JS needed).
Each player row is an <li class="list-group-item"> with a /redirect/player/{id}
link (name + spotrac_id), a <small> with team+pos, and a <span class="medium">
holding the $cap hit.

Output: data/market_cultural/spotrac_nfl_salary.json
  { built, source, years, lists: { year: [ {rank, name, norm, spotrac_id,
     team, pos, cap_hit} ... ] } }
Raw HTML cached per year at pipeline/cache/spotrac_nfl_{year}.html (offline re-parse).

Cross-ref: vs assets/unified.json gridiron players (norm, "gridiron").
Writes data/market_cultural/spotrac_coverage.json.

Run:  python pipeline/acquire_spotrac.py [--offline] [--refresh] [--year YYYY]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_forbes import fetch_html, norm_name  # reuse curl_cffi fetch + name norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_cultural"
ASSETS = ROOT / "assets"
CACHE = ROOT / "pipeline" / "cache"

BASE = "https://www.spotrac.com/nfl/rankings/player/_/year/{year}"
# gridiron corpus era: unified era_counts jump from ~2015 -> scrape 2015-2025
YEARS = list(range(2015, 2026))

ROW_RE = re.compile(r'<li class="list-group-item[^"]*"[^>]*>(.*?)</li>', re.DOTALL | re.IGNORECASE)
LINK_RE = re.compile(
    r'href="https://www\.spotrac\.com/redirect/player/(\d+)"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)
SMALL_RE = re.compile(r"/>\s*([^<]+?)</small>", re.DOTALL)
MONEY_RE = re.compile(r"\$([\d,]+)")
RANK_RE = re.compile(r'width:65px;"[^>]*>\s*(\d+)\s*</div>', re.DOTALL)


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


def parse_year_html(html: str) -> list[dict]:
    rows = []
    for block in ROW_RE.findall(html):
        link = LINK_RE.search(block)
        if not link:
            continue
        spotrac_id = int(link.group(1))
        name = _strip(link.group(2))
        if not name:
            continue
        team = pos = ""
        sm = SMALL_RE.search(block)
        if sm:
            tp = _strip(sm.group(1))  # e.g. "KC, QB"
            team, _, pos = tp.partition(",")
            team, pos = team.strip(), pos.strip()
        money = MONEY_RE.search(block)
        cap_hit = int(money.group(1).replace(",", "")) if money else None
        rank_m = RANK_RE.search(block)
        rank = int(rank_m.group(1)) if rank_m else None
        rows.append(
            {
                "rank": rank,
                "name": name,
                "norm": norm_name(name),
                "spotrac_id": spotrac_id,
                "team": team,
                "pos": pos,
                "cap_hit": cap_hit,
            }
        )
    return rows


def build_year(year: int, refresh: bool) -> tuple[list[dict], bool, bool]:
    """Return (rows, cached, fetched). Uses cache unless refresh."""
    cp = CACHE / f"spotrac_nfl_{year}.html"
    if cp.exists() and not refresh:
        html = cp.read_text(encoding="utf-8")
        return parse_year_html(html), True, False
    html = fetch_html(BASE.format(year=year))
    cp.write_text(html, encoding="utf-8")
    return parse_year_html(html), False, True


def cross_ref(doc: dict) -> dict:
    upath = ASSETS / "unified.json"
    if not upath.exists():
        return {"unified_json": str(upath), "present": False}
    U = json.loads(upath.read_text(encoding="utf-8"))
    players = U["players"]
    idx: dict[tuple[str, str], list[int]] = {}
    for i, p in enumerate(players):
        if p["sport"] == "gridiron":
            idx.setdefault((norm_name(p["name"]), "gridiron"), []).append(i)
    grid_n = sum(1 for p in players if p["sport"] == "gridiron")

    matched_records = 0
    total_records = 0
    unique_matched: set[str] = set()
    by_year: dict[int, int] = {}
    examples = []
    unmatched_stars = []  # high-cap-hit players not in corpus (validation)
    for year, recs in doc["lists"].items():
        ym = 0
        for r in recs:
            if r["cap_hit"] is None:
                continue
            total_records += 1
            if (r["norm"], "gridiron") in idx:
                matched_records += 1
                ym += 1
                unique_matched.add(r["norm"])
                if len(examples) < 10:
                    examples.append(
                        {
                            "year": int(year),
                            "name": r["name"],
                            "team": r["team"],
                            "pos": r["pos"],
                            "cap_hit": r["cap_hit"],
                            "unified_rows": len(idx[(r["norm"], "gridiron")]),
                        }
                    )
            elif r["cap_hit"] and r["cap_hit"] > 20_000_000 and len(unmatched_stars) < 15:
                unmatched_stars.append(
                    {
                        "year": int(year),
                        "name": r["name"],
                        "team": r["team"],
                        "pos": r["pos"],
                        "cap_hit": r["cap_hit"],
                    }
                )
        by_year[int(year)] = ym
    return {
        "unified_json": str(upath),
        "present": True,
        "gridiron_rows": grid_n,
        "spotrac_records_with_cap": total_records,
        "matched_records": matched_records,
        "unique_gridiron_athletes_matched": len(unique_matched),
        "by_year": by_year,
        "examples": examples,
        "unmatched_high_cap_stars": unmatched_stars,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--offline",
        action="store_true",
        help="parse only from cached HTML; skip missing years",
    )
    ap.add_argument("--refresh", action="store_true", help="re-fetch even if HTML cache exists")
    ap.add_argument("--year", type=int, default=None, help="single year (debug)")
    args = ap.parse_args()
    years = [args.year] if args.year else YEARS

    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    lists: dict[int, list[dict]] = {}
    for y in years:
        try:
            rows, cached, fetched = build_year(y, args.refresh)
            if args.offline and not fetched and not cached:
                # offline + no cache + couldn't fetch -> skip silently
                continue
            lists[y] = rows
            tag = "fetched" if fetched else "cached"
            n_cap = sum(1 for r in rows if r["cap_hit"] is not None)
            print(f"  {y}: {tag} {len(rows)} rows ({n_cap} with cap hit)")
            if fetched:
                time.sleep(3.5)  # polite Spotrac throttle
        except Exception as e:
            print(f"  {y}: FAILED ({type(e).__name__}: {e})")

    doc = {
        "built": time.strftime("%Y-%m-%d"),
        "source": "spotrac.com/nfl/rankings/player (cap hit, per year)",
        "signal": "cap_hit (per-season NFL salary cap charge; SALARY_LOG proxy)",
        "n_years": len(lists),
        "years": sorted(lists.keys()),
        "lists": {str(y): lists[y] for y in sorted(lists.keys())},
    }
    out = DATA / "spotrac_nfl_salary.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(len(v) for v in lists.values())
    print(f"\nsaved {out.name}: {doc['n_years']} years, {total} player-year records")

    cov = cross_ref(doc)
    (DATA / "spotrac_coverage.json").write_text(json.dumps(cov, indent=2, ensure_ascii=False), encoding="utf-8")
    if cov.get("present"):
        print(f"\ncross-ref vs unified.json gridiron ({cov['gridiron_rows']} rows):")
        print(
            f"  Spotrac cap-hit records matched: {cov['matched_records']}/"
            f"{cov['spotrac_records_with_cap']}  "
            f"({cov['unique_gridiron_athletes_matched']} unique athletes)"
        )
        print(f"  by year: {cov['by_year']}")
        print("  examples:")
        for e in cov["examples"][:6]:
            print(
                f"    {e['year']} {e['name']:<22} {e['team']:>3} {e['pos']:<4} "
                f"cap=${e['cap_hit']:,}  unified_rows={e['unified_rows']}"
            )
        if cov["unmatched_high_cap_stars"]:
            print("  unmatched high-cap (>=$20M) stars " "(validation — should be in gridiron corpus):")
            for u in cov["unmatched_high_cap_stars"][:8]:
                print(f"    {u['year']} {u['name']:<22} {u['team']:>3} {u['pos']:<4} " f"cap=${u['cap_hit']:,}")
    else:
        print("unified.json not found (cross-ref skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
