"""Vector Unified — acquire English Wikipedia lead extracts for cultural text.

Resolves unique (sport, name_norm) players from assets/unified.json to an enwiki
title + lead extract. Resumable JSON cache. Ambiguous / missing pages stay masked
downstream (honesty §257).

Priority order: Forbes / awards norms first, then remaining unique players.

API: MediaWiki action=query (search + extracts). Polite UA + rate limit.
Output: data/market_cultural/wikipedia_bios.json

Run:
  python pipeline/acquire_wikipedia_bios.py [--limit N] [--sleep 0.2] [--priority-only]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_forbes import norm_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_cultural"
ASSETS = ROOT / "assets"
OUT = DATA / "wikipedia_bios.json"

UA = "VectorUnifiedResearch/0.1 (cultural-text MVP; local research build; contact: local)"
API = "https://en.wikipedia.org/w/api.php"

SPORT_HINT = {
    "hoops": "basketball",
    "gridiron": "American football",
    "pitch": "footballer",
}
SPORT_REJECT = {
    "hoops": [r"american football", r"association football", r"soccer"],
    "gridiron": [r"basketball", r"association football", r"\bfootballer\b"],
    "pitch": [r"basketball", r"american football", r"nfl"],
}


def wiki_get(params: dict, retries: int = 4) -> dict:
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                wait = 15 * (attempt + 1)
                print(f"  rate-limit {e.code}; sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last_err


def search_title(name: str, sport: str) -> str | None:
    # Prefer exact title hit first (1 request)
    direct = name.replace(" ", "_")
    data = wiki_get(
        {
            "action": "query",
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "exchars": 200,
            "redirects": 1,
            "titles": name,
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if int(pid) > 0 and page.get("extract"):
            extract = page["extract"]
            if "may refer to" not in extract.lower()[:80]:
                return page.get("title") or name

    hint = SPORT_HINT.get(sport, "")
    query = f"{name} {hint}".strip()
    data = wiki_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "srnamespace": 0,
        }
    )
    hits = data.get("query", {}).get("search", [])
    if not hits:
        data = wiki_get(
            {
                "action": "query",
                "list": "search",
                "srsearch": name,
                "srlimit": 5,
                "srnamespace": 0,
            }
        )
        hits = data.get("query", {}).get("search", [])
    for h in hits:
        title = h.get("title") or ""
        if "disambiguation" in title.lower():
            continue
        return title
    return None


def fetch_extract(title: str) -> tuple[str | None, str | None, int]:
    """Return (canonical_title, extract, pageid) or (None, None, 0)."""
    data = wiki_get(
        {
            "action": "query",
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "exchars": 1200,
            "redirects": 1,
            "titles": title,
            "inprop": "url",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if int(pid) < 0 or page.get("missing") is not None:
            return None, None, 0
        extract = (page.get("extract") or "").strip()
        if not extract:
            return None, None, 0
        if "may refer to" in extract.lower()[:80]:
            return None, None, 0
        return page.get("title") or title, extract, int(pid)
    return None, None, 0


def extract_ok(sport: str, extract: str) -> bool:
    low = extract.lower()
    for pat in SPORT_REJECT.get(sport, []):
        # only reject if the *wrong* sport dominates the opening and ours is absent
        if re.search(pat, low[:400]) and SPORT_HINT.get(sport, "xyz").lower() not in low[:400]:
            # pitch hint "footballer" often absent; allow "football" / "midfielder" etc.
            if sport == "pitch" and any(
                w in low[:400]
                for w in (
                    "footballer",
                    "football player",
                    "midfielder",
                    "striker",
                    "winger",
                    "goalkeeper",
                    "premier league",
                    "la liga",
                    "serie a",
                    "bundesliga",
                    "world cup",
                    "uefa",
                )
            ):
                continue
            if sport == "gridiron" and any(
                w in low[:400]
                for w in (
                    "nfl",
                    "quarterback",
                    "wide receiver",
                    "running back",
                    "linebacker",
                    "tight end",
                    "american football",
                )
            ):
                continue
            if sport == "hoops" and any(
                w in low[:400]
                for w in (
                    "nba",
                    "basketball",
                    "point guard",
                    "shooting guard",
                    "small forward",
                    "power forward",
                    "center",
                )
            ):
                continue
            return False
    return True


def load_priority_norms() -> set[tuple[str, str]]:
    """(sport, norm) set from Forbes + awards for priority pull."""
    pri: set[tuple[str, str]] = set()
    fb = DATA / "forbes_earnings.json"
    if fb.exists():
        forbes = json.loads(fb.read_text(encoding="utf-8"))
        for recs in forbes.get("lists", {}).values():
            for r in recs:
                if r.get("sport") in SPORT_HINT:
                    pri.add((r["sport"], r["norm"]))
    aw = DATA / "awards.json"
    if aw.exists():
        awards = json.loads(aw.read_text(encoding="utf-8"))
        for sport, by_norm in awards.get("prestige", {}).items():
            for norm in by_norm:
                pri.add((sport, norm))
    return pri


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max new resolves this run (0=all)")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--priority-only", action="store_true")
    ap.add_argument(
        "--refresh-failed",
        action="store_true",
        help="retry entries previously marked failed/unmatched",
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    U = json.loads((ASSETS / "unified.json").read_text(encoding="utf-8"))
    # unique by (sport, norm) keeping a display name
    uniq: dict[tuple[str, str], str] = {}
    for p in U["players"]:
        key = (p["sport"], norm_name(p["name"]))
        uniq.setdefault(key, p["name"])

    pri = load_priority_norms()
    keys = sorted(uniq.keys(), key=lambda k: (0 if k in pri else 1, k[0], k[1]))
    if args.priority_only:
        keys = [k for k in keys if k in pri]

    state = {
        "built": time.strftime("%Y-%m-%d"),
        "n_unique_target": len(keys),
        "ua": UA,
        "players": {},
    }
    if OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        state["players"] = prev.get("players", {})

    done = 0
    new = 0
    for sport, norm in keys:
        pk = f"{sport}::{norm}"
        prev_row = state["players"].get(pk)
        if prev_row and prev_row.get("status") == "ok":
            done += 1
            continue
        if prev_row and prev_row.get("status") in ("unmatched", "reject", "error"):
            if not args.refresh_failed:
                done += 1
                continue
        if args.limit and new >= args.limit:
            break

        name = uniq[(sport, norm)]
        try:
            title = search_title(name, sport)
            time.sleep(args.sleep)
            if not title:
                state["players"][pk] = {
                    "sport": sport,
                    "norm": norm,
                    "name": name,
                    "status": "unmatched",
                    "wiki_title": None,
                    "extract": None,
                }
            else:
                canon, extract, pageid = fetch_extract(title)
                time.sleep(args.sleep)
                if not extract or not extract_ok(sport, extract):
                    state["players"][pk] = {
                        "sport": sport,
                        "norm": norm,
                        "name": name,
                        "status": "reject",
                        "wiki_title": canon or title,
                        "extract": None,
                        "pageid": pageid,
                    }
                else:
                    state["players"][pk] = {
                        "sport": sport,
                        "norm": norm,
                        "name": name,
                        "status": "ok",
                        "wiki_title": canon,
                        "pageid": pageid,
                        "extract": extract,
                        "extract_chars": len(extract),
                    }
            new += 1
        except Exception as e:
            state["players"][pk] = {
                "sport": sport,
                "norm": norm,
                "name": name,
                "status": "error",
                "error": str(e)[:200],
                "wiki_title": None,
                "extract": None,
            }
            new += 1
            time.sleep(max(args.sleep, 1.0))

        if new % 25 == 0:
            ok_n = sum(1 for v in state["players"].values() if v.get("status") == "ok")
            state["coverage"] = {
                "resolved": len(state["players"]),
                "ok": ok_n,
                "ok_rate": round(ok_n / max(1, len(state["players"])), 4),
            }
            OUT.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"  checkpoint new={new} ok={ok_n}/{len(state['players'])}", flush=True)

    ok_n = sum(1 for v in state["players"].values() if v.get("status") == "ok")
    state["coverage"] = {
        "resolved": len(state["players"]),
        "ok": ok_n,
        "ok_rate": round(ok_n / max(1, len(state["players"])), 4),
        "priority_norms": len(pri),
        "target_unique": len(keys),
    }
    state["built"] = time.strftime("%Y-%m-%d")
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(
        f"wrote {OUT} | ok={ok_n}/{len(state['players'])} new_this_run={new}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
