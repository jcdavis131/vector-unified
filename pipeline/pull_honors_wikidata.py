"""Vector Unified — Wikidata honors + social-handles pull (cross-sport, free, no anti-bot).

Queries the Wikidata SPARQL endpoint for athletes in our three sports
(basketball Q5372 / american football Q41323 / association football Q2736) who have:
  - awards received (P166)  -> award ledger -> AWARD_PRESTIGE / AWARD_RECENT
  - Twitter/X handle (P2002) and/or Facebook (P2013) -> social handle seed

Outputs (data/market/):
  honors_wikidata.json   # raw: {qid: {name, sport_unified, country, birthyear, awards:[...]}}
  social_handles.json    # {qid: {name, sport_unified, handles:{twitter,facebook}}}
  award_prestige.json    # matched to native players: {player_idx, AWARD_PRESTIGE, AWARD_RECENT, n_awards, tiers}
  market_cultural_report.json (appended) — coverage stats

Tier scoring: schema §2 award-prestige tier map. AWARD_PRESTIGE = 1 - prod(1 - w_tier)
over career awards (soft-or, saturates). AWARD_RECENT recomputes over awards dated in the
last 2 calendar years (uses P166 qualifier P585 "point in time" where present; else
undated awards count to career prestige only, not recent).

Honesty: unmatched/ambiguous names -> not written to award_prestige (left masked
downstream). Uncategorized awards are kept in the ledger with tier=null and flagged.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import requests
from match_names import match_external
from resolve_names import MARKET

UA = "VectorUnifiedResearch/0.1 (athlete market/cultural signal research; local build)"
ENDPOINT = "https://query.wikidata.org/sparql"
# Q41323 = American football. It was Q9398, which is GRUGLIASCO, AN ITALIAN COMUNE — so
# every gridiron query asked Wikidata for people whose sport is a town near Turin and got
# back nothing. The failure was silent because "0 rows" and "query returned no matches"
# look identical, and the pull reports row counts rather than checking that each sport
# produced any. 7.12 measured the consequence before finding the cause: gridiron carried 4
# of 1,572 athletes in award_prestige, and both raw pulls had zero gridiron rows.
SPORT_Q = {"hoops": "Q5372", "gridiron": "Q41323", "pitch": "Q2736"}
SPORT_Q_REV = {v: k for k, v in SPORT_Q.items()}

# ---- award prestige tiers (schema §2) ----
TIERS = [
    (
        0.85,
        [
            "mvp",
            "most valuable player",
            "dpoy",
            "defensive player of the year",
            "finals mvp",
            "ballon d'or",
            "the best",
            "fifa best",
            "golden ball",
            "kopa trophy",
            "yashin trophy",
            "super bowl mvp",
            "finals most valuable",
        ],
    ),
    (
        0.50,
        [
            "all-nba first",
            "all-nba 1st",
            "first-team all-nba",
            "all-pro first",
            "first-team all-pro",
            "fifpro",
            "fifpro world xi",
            "uefa team of the year",
            "uefa toty",
            "team of the year",
        ],
    ),
    (
        0.30,
        [
            "all-star",
            "all star",
            "pro bowl",
            "team of the tournament",
            "all-star team",
            "world cup all-star",
            "tournament team",
        ],
    ),
    (
        0.20,
        [
            "rookie of the year",
            "roy",
            "golden boy",
            "young player",
            "young player of the",
            "most improved",
            "mip",
            "sixth man",
            "6th man",
            "comeback player",
        ],
    ),
    (
        0.40,
        [
            "championship",
            "champion",
            "nba championship",
            "super bowl champion",
            "world cup winner",
            "world cup champion",
            "league title",
            "continental",
            "cup winner",
            "trophy winner",
            "golden boot",
            "top scorer",
            "assist leader",
            "scoring champion",
            "gold medal",
            "silver medal",
            "bronze medal",
        ],
    ),
]


def classify_tier(award_label: str):
    s = award_label.lower()
    for w, kws in TIERS:
        for kw in kws:
            if kw in s:
                return w
    return None


def prestige_score(tiers: list[float]) -> float:
    if not tiers:
        return 0.0
    prod = 1.0
    for t in tiers:
        prod *= 1.0 - t
    return round(1.0 - prod, 4)


def run_query(q: str):
    r = requests.get(
        ENDPOINT,
        params={"query": q, "format": "json"},
        headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def awards_query(sport_qid: str):
    return f"""
SELECT ?item ?itemLabel ?countryLabel ?birthYear ?awardLabel ?when WHERE {{
  ?item wdt:P641 wd:{sport_qid}.
  ?item p:P166 ?st.
  ?st ps:P166 ?award.
  OPTIONAL {{ ?item wdt:P27 ?country. }}
  OPTIONAL {{ ?item wdt:P569 ?dob. BIND(YEAR(?dob) AS ?birthYear) }}
  OPTIONAL {{ ?st pq:P585 ?when. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def handles_query(sport_qid: str):
    return f"""
SELECT ?item ?itemLabel ?tw ?fb WHERE {{
  ?item wdt:P641 wd:{sport_qid}.
  OPTIONAL {{ ?item wdt:P2002 ?tw. }}
  OPTIONAL {{ ?item wdt:P2013 ?fb. }}
  FILTER(BOUND(?tw) || BOUND(?fb))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 8000"""


def val(b, k):
    return b.get(k, {}).get("value")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    MARKET.mkdir(parents=True, exist_ok=True)
    idx = json.loads((MARKET / "name_index.json").read_text(encoding="utf-8"))

    honors: dict[str, dict] = {}  # qid -> {name, sport, country, birthyear, awards:[{label, when, tier}]}
    handles: dict[str, dict] = {}  # qid -> {name, sport, twitter, facebook}
    n_uncat = 0

    for sport, qid in SPORT_Q.items():
        print(f"[{sport}] awards query ({qid})...")
        try:
            rows = run_query(awards_query(qid))
        except Exception as e:
            print(f"  awards query failed: {e}")
            rows = []
        for b in rows:
            qidu = val(b, "item")
            if not qidu:
                continue
            if qidu not in honors:
                honors[qidu] = {
                    "name": val(b, "itemLabel") or "",
                    "sport_unified": sport,
                    "country": val(b, "countryLabel"),
                    "birthyear": None,
                    "awards": [],
                }
            if val(b, "birthYear"):
                try:
                    honors[qidu]["birthyear"] = int(val(b, "birthYear"))
                except ValueError:
                    pass
            lbl = val(b, "awardLabel") or ""
            when = val(b, "when") or ""
            yr = None
            if when:
                m = re.search(r"(\d{4})", when)
                yr = int(m.group(1)) if m else None
            t = classify_tier(lbl)
            if t is None:
                n_uncat += 1
            honors[qidu]["awards"].append({"label": lbl, "year": yr, "tier": t})
        print(f"  awards: {len(rows)} rows, {len([q for q in honors if honors[q]['sport_unified']==sport])} athletes")

        print(f"[{sport}] handles query...")
        try:
            rows = run_query(handles_query(qid))
        except Exception as e:
            print(f"  handles query failed: {e}")
            rows = []
        for b in rows:
            qidu = val(b, "item")
            if not qidu:
                continue
            if qidu not in handles:
                handles[qidu] = {
                    "name": val(b, "itemLabel") or "",
                    "sport_unified": sport,
                    "twitter": None,
                    "facebook": None,
                }
            if val(b, "tw"):
                handles[qidu]["twitter"] = val(b, "tw")
            if val(b, "fb"):
                handles[qidu]["facebook"] = val(b, "fb")
        print(
            f"  handles: {len(rows)} rows, {len([q for q in handles if handles[q]['sport_unified']==sport])} athletes"
        )

    # REFUSE TO OVERWRITE ON A SILENTLY EMPTY SPORT. A wrong sport QID returns 0 rows and
    # a healthy query for a sport with no award-holders would too — but the second case
    # does not exist for these three leagues, so 0 means the query is broken. Writing
    # anyway is what let a comune in Piedmont stand in for the NFL across two artifacts
    # and a derived one, until 7.12 measured the hole from the far end.
    empty = [
        sp
        for sp in SPORT_Q
        if not any(v["sport_unified"] == sp for v in honors.values())
        or not any(v["sport_unified"] == sp for v in handles.values())
    ]
    if empty:
        raise SystemExit(
            f"REFUSING TO WRITE: no rows for {', '.join(empty)}. Check the sport QID in "
            f"SPORT_Q ({', '.join(f'{k}={v}' for k, v in SPORT_Q.items())}) — a QID that "
            f"is not a sport returns an empty result set, not an error."
        )

    # write raw honors + handles
    (MARKET / "honors_wikidata.json").write_text(json.dumps(honors, indent=2, ensure_ascii=False), encoding="utf-8")
    (MARKET / "social_handles.json").write_text(json.dumps(handles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nraw: {len(honors)} athletes-with-awards, {len(handles)} athletes-with-handles "
        f"(uncategorized awards: {n_uncat})"
    )

    # match to native players + compute AWARD_PRESTIGE / AWARD_RECENT
    import datetime

    this_year = datetime.date.today().year
    prestige = {}
    tiers_used = defaultdict(int)
    match_tiers = defaultdict(int)
    for qidu, h in honors.items():
        pid, tier, ncand = match_external(h["name"], h["sport_unified"], h["country"], idx)
        match_tiers[tier] += 1
        if pid is None:
            continue
        aw = h["awards"]
        career_tiers = [a["tier"] for a in aw if a["tier"] is not None]
        recent_tiers = [
            a["tier"] for a in aw if a["tier"] is not None and a["year"] is not None and a["year"] >= this_year - 2
        ]
        for t in career_tiers:
            tiers_used[t] += 1
        prestige[pid] = {
            "name": idx["players"][pid]["name"],
            "sport": idx["players"][pid]["sport"],
            "native_player_id": idx["players"][pid]["player_id"],
            "qid": qidu,
            "n_awards": len(aw),
            "n_tiered": len(career_tiers),
            "AWARD_PRESTIGE": prestige_score(career_tiers),
            "AWARD_RECENT": prestige_score(recent_tiers),
            "awards": aw,
            "match_tier": tier,
        }
    (MARKET / "award_prestige.json").write_text(json.dumps(prestige, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nmatched to native: {len(prestige)} players (match tiers: {dict(match_tiers)})")
    print(f"tier weights used (career): { {str(k): v for k, v in tiers_used.items()} }")
    # per-sport coverage
    for s in ("hoops", "gridiron", "pitch"):
        n = sum(1 for p in prestige.values() if p["sport"] == s)
        print(f"  {s:9s} {n} players with award-prestige")
    # top prestige showcase
    top = sorted(prestige.values(), key=lambda p: -p["AWARD_PRESTIGE"])[:12]
    print("\ntop-12 by AWARD_PRESTIGE:")
    for p in top:
        print(f"  {p['AWARD_PRESTIGE']:.3f}  {p['sport']:9s} {p['name']}  (n_awards={p['n_awards']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
