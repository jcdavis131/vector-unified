"""Vector Unified — join market/cultural sources onto the unified corpus.

Reads:
  assets/unified.json
  data/market_cultural/forbes_earnings.json
  data/market_cultural/spotrac_nfl_salary.json
  data/market_cultural/awards.json
  data/market_cultural/wikipedia_pageviews.json (social reach, keyless -- see
    acquire_wikipedia_pageviews.py; replaces the originally-scoped Apify path)

Writes player-season rows aligned to unified.json order:
  data/market_cultural/market_cultural.json
    { built, n_rows, schema, coverage, rows: [
        { sport, player_id, name, season, year,
          salary_m, endorse_m, earnings_m, award_prestige, reach_views,
          m_salary, m_endorse, m_earnings, m_award, m_reach } ... ] }

Honesty contract (§257): missing signals are masked (m_*=0), never imputed.
Pitch salary is almost entirely Forbes-stars-only (Transfermarkt blocked).
Social reach: Wikipedia pageviews, exact (player, year) match only (no
interpolation across years, same discipline as Forbes) -- only covers the 429
players resolved by acquire_wikipedia_bios.py and only 2015+ seasons (earliest
article-level pageview data). Hoops full salary stays in the native BBREF
pipeline; this join adds the *cross-sport* Forbes/Spotrac/awards/reach layer.

Run:  python pipeline/market_cultural_join.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_forbes import norm_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_cultural"
ASSETS = ROOT / "assets"


def season_to_year(sport: str, season) -> int | None:
    if sport == "gridiron":
        return int(season) if season is not None else None
    if sport == "hoops":
        # "1996-97" -> 1997 (end year, matches Forbes/awards award year)
        m = re.search(r"(\d{4})\s*[-–]\s*(\d{2,4})", str(season))
        if m:
            y2 = m.group(2)
            return int(y2) if len(y2) == 4 else (1900 + int(y2) if int(y2) >= 50 else 2000 + int(y2))
        m = re.search(r"(\d{4})", str(season))
        return int(m.group(1)) if m else None
    if sport == "pitch":
        # "WC 2018" / "EURO 2020" / "Copa America 2019" -> year token
        m = re.search(r"(19|20)\d{2}", str(season))
        return int(m.group(0)) if m else None
    return None


def log1p_m(val_m: float | None) -> float | None:
    """log1p of dollars (millions -> dollars first)."""
    if val_m is None or val_m < 0:
        return None
    return math.log1p(val_m * 1_000_000.0)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    U = json.loads((ASSETS / "unified.json").read_text(encoding="utf-8"))
    forbes = json.loads((DATA / "forbes_earnings.json").read_text(encoding="utf-8"))
    spotrac = json.loads((DATA / "spotrac_nfl_salary.json").read_text(encoding="utf-8"))
    awards = json.loads((DATA / "awards.json").read_text(encoding="utf-8"))
    pageviews_path = DATA / "wikipedia_pageviews.json"
    pageviews = json.loads(pageviews_path.read_text(encoding="utf-8")) if pageviews_path.exists() else {"players": {}}

    # Forbes index: (norm, sport, year) -> {salary_m, endorse_m, total_m}
    # Also career-max endorse per (norm, sport) as a fallback cultural signal
    f_idx: dict[tuple[str, str, int], dict] = {}
    f_endorse_max: dict[tuple[str, str], float] = {}
    for year_s, recs in forbes["lists"].items():
        year = int(year_s)
        for r in recs:
            if r["sport"] == "other":
                continue
            key = (r["norm"], r["sport"], year)
            f_idx[key] = {
                "salary_m": r["salary_m"],
                "endorse_m": r["endorse_m"],
                "total_m": r["total_m"],
            }
            ek = (r["norm"], r["sport"])
            if r["endorse_m"] is not None:
                f_endorse_max[ek] = max(f_endorse_max.get(ek, 0.0), r["endorse_m"])

    # Spotrac: (norm, year) -> cap_hit dollars
    s_idx: dict[tuple[str, int], float] = {}
    for year_s, recs in spotrac["lists"].items():
        year = int(year_s)
        for r in recs:
            if r["cap_hit"] is None:
                continue
            s_idx[(r["norm"], year)] = float(r["cap_hit"])

    # Awards prestige: (norm, sport) -> prestige (career)
    a_idx: dict[tuple[str, str], float] = {}
    for sport, athletes in awards["prestige"].items():
        for nn, rec in athletes.items():
            a_idx[(nn, sport)] = float(rec["prestige"])

    # Pageviews: "sport::norm" (matches acquire_wikipedia_bios.py's key) -> by_year
    pv_idx: dict[str, dict] = pageviews.get("players", {})

    rows = []
    cov = {
        "salary": {"hoops": 0, "gridiron": 0, "pitch": 0},
        "endorse": {"hoops": 0, "gridiron": 0, "pitch": 0},
        "award": {"hoops": 0, "gridiron": 0, "pitch": 0},
        "reach": {"hoops": 0, "gridiron": 0, "pitch": 0},
        "any": {"hoops": 0, "gridiron": 0, "pitch": 0},
        "n": {"hoops": 0, "gridiron": 0, "pitch": 0},
    }
    for p in U["players"]:
        sport = p["sport"]
        nn = norm_name(p["name"])
        year = season_to_year(sport, p["season"])
        cov["n"][sport] += 1

        salary_m = None
        endorse_m = None
        # 1) Forbes year-exact (stars)
        if year is not None:
            fr = f_idx.get((nn, sport, year))
            if fr:
                salary_m = fr["salary_m"]
                endorse_m = fr["endorse_m"]
        # 2) Spotrac fills gridiron salary when Forbes missing
        if sport == "gridiron" and salary_m is None and year is not None:
            cap = s_idx.get((nn, year))
            if cap is not None:
                salary_m = cap / 1_000_000.0  # dollars -> millions
        # 3) career award prestige (same across seasons)
        award_prestige = a_idx.get((nn, sport))
        # 4) social reach: exact (player, year) pageviews match, no interpolation
        reach_views = None
        pv = pv_idx.get(f"{sport}::{nn}")
        if pv and year is not None:
            reach_views = pv["by_year"].get(str(year))

        earnings_m = None
        if salary_m is not None or endorse_m is not None:
            earnings_m = (salary_m or 0.0) + (endorse_m or 0.0)

        m_salary = 1 if salary_m is not None else 0
        m_endorse = 1 if endorse_m is not None else 0
        m_earnings = 1 if earnings_m is not None else 0
        m_award = 1 if award_prestige is not None else 0
        m_reach = 1 if reach_views is not None else 0

        if m_salary:
            cov["salary"][sport] += 1
        if m_endorse:
            cov["endorse"][sport] += 1
        if m_award:
            cov["award"][sport] += 1
        if m_reach:
            cov["reach"][sport] += 1
        if m_salary or m_endorse or m_award or m_reach:
            cov["any"][sport] += 1

        rows.append(
            {
                "sport": sport,
                "player_id": p["player_id"],
                "name": p["name"],
                "season": p["season"],
                "year": year,
                "salary_m": salary_m,
                "endorse_m": endorse_m,
                "earnings_m": earnings_m,
                "award_prestige": award_prestige,
                "reach_views": reach_views,
                "salary_log": log1p_m(salary_m),
                "endorse_log": log1p_m(endorse_m),
                "earnings_log": log1p_m(earnings_m),
                "reach_log": (math.log1p(reach_views) if reach_views is not None else None),
                "m_salary": m_salary,
                "m_endorse": m_endorse,
                "m_earnings": m_earnings,
                "m_award": m_award,
                "m_reach": m_reach,
            }
        )

    # coverage rates
    rates = {}
    for signal in ("salary", "endorse", "award", "reach", "any"):
        rates[signal] = {s: round(cov[signal][s] / max(1, cov["n"][s]), 4) for s in ("hoops", "gridiron", "pitch")}

    doc = {
        "built": time.strftime("%Y-%m-%d"),
        "n_rows": len(rows),
        "aligned_to": "assets/unified.json (same order)",
        "schema": {
            "salary_m": "on-field $ millions (Forbes stars all sports; Spotrac cap-hit gridiron)",
            "endorse_m": "off-field $ millions (Forbes stars only)",
            "earnings_m": "salary_m + endorse_m when either present",
            "award_prestige": "career tier-weighted award wins (NBA MVP / AP NFL MVP / Ballon d'Or / FIFA Best)",
            "reach_views": "Wikipedia pageviews for that exact (player, year) -- keyless social-reach signal, no interpolation",
            "masks": "m_*=1 when present; never impute (§257)",
            "gaps": [
                "pitch salary: Forbes stars only (Transfermarkt market-value blocked)",
                "endorse: Forbes stars only (~23 unique athletes across 3 sports)",
                "social reach: Wikipedia pageviews (keyless), only for the 429 players "
                "resolved so far and only 2015+ seasons (earliest article-level data); "
                "originally scoped via Apify (needs a paid key), swapped for a free source",
                "hoops full salary: remains in native BBREF pipeline, not duplicated here",
            ],
        },
        "coverage_counts": cov,
        "coverage_rates": rates,
        "rows": rows,
    }
    out = DATA / "market_cultural.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved {out.name}: {len(rows)} rows (aligned to unified.json)")
    print("coverage rates (fraction of player-seasons with signal):")
    for signal in ("salary", "endorse", "award", "reach", "any"):
        r = rates[signal]
        print(f"  {signal:8s}  hoops={r['hoops']:.3f}  gridiron={r['gridiron']:.3f}  pitch={r['pitch']:.3f}")
    print("coverage counts:")
    for signal in ("salary", "endorse", "award", "reach", "any"):
        print(f"  {signal:8s}  {cov[signal]}")

    # showcase: top prestige / earnings in corpus
    print("\nshowcase (highest award_prestige in corpus):")
    top = sorted([r for r in rows if r["m_award"]], key=lambda r: -r["award_prestige"])
    seen = set()
    for r in top:
        k = (norm_name(r["name"]), r["sport"])
        if k in seen:
            continue
        seen.add(k)
        print(f"  [{r['sport'][:2]}] {r['name']:<22} prestige={r['award_prestige']}")
        if len(seen) >= 8:
            break
    print("\nshowcase (highest salary_m present):")
    top_s = sorted([r for r in rows if r["m_salary"]], key=lambda r: -(r["salary_m"] or 0))[:6]
    for r in top_s:
        print(
            f"  [{r['sport'][:2]}] {r['year']} {r['name']:<22} salary=${r['salary_m']:.1f}M "
            f"endorse={r['endorse_m']}"
        )
    print("\nshowcase (highest reach_views present):")
    top_r = sorted([r for r in rows if r["m_reach"]], key=lambda r: -(r["reach_views"] or 0))[:6]
    for r in top_r:
        print(f"  [{r['sport'][:2]}] {r['year']} {r['name']:<22} reach_views={r['reach_views']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
