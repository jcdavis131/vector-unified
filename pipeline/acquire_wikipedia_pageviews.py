"""Vector Unified — acquire Wikipedia pageviews as a keyless "social reach" signal.

Phase 4b.5 originally scoped social reach via Apify (paid, needs an API key).
This is a free, keyless substitute using the official Wikimedia REST pageviews
API (no auth, no key, generous public rate limits): for each player already
resolved to an enwiki title in `wikipedia_bios.json`, pull monthly pageviews
from the start of Wikimedia's article-level data (2015-07) through the most
recent complete month, and aggregate to annual sums. This gives a genuine,
verifiable public-attention signal per (player, year) -- not a proxy invented
from nothing, and not blocked the way Transfermarkt (TLS) or a keyed API would
be.

Coverage is honestly bounded: only players with 2015+ playing years get any
signal (pre-2015 seasons stay masked downstream), and only the 447 players
already resolved by acquire_wikipedia_bios.py are queried (extending that
resolution to the full corpus is a separate, larger acquisition run).

Output: data/market_cultural/wikipedia_pageviews.json
  { built, source, window: [start, end], players: {
      "sport::norm": { title, by_year: { "2019": 812345, ... } } } }
Resumable: players already in the output are skipped unless --refresh.

Run:  python pipeline/acquire_wikipedia_pageviews.py [--limit N] [--sleep 0.3] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_cultural"
BIOS = DATA / "wikipedia_bios.json"
OUT = DATA / "wikipedia_pageviews.json"

UA = "VectorUnifiedResearch/0.1 (social-reach MVP; local research build; contact: local)"
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
START = "2015070100"  # earliest available article-level data


def _end_month() -> str:
    """Most recent COMPLETE calendar month (pageviews for the current month are partial)."""
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}{m:02d}0100"


def fetch_monthly(title: str, retries: int = 4) -> list[dict] | None:
    article = title.replace(" ", "_")
    url = f"{API}/en.wikipedia.org/all-access/all-agents/{urllib.parse.quote(article, safe='')}/monthly/{START}/{_end_month()}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8")).get("items", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []  # no data for this article (e.g. created after window, or redirect target mismatch)
            if e.code in (429, 503):
                wait = 15 * (attempt + 1)
                print(f"  rate-limit {e.code}; sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    return None


def to_annual(items: list[dict]) -> dict[str, int]:
    by_year: dict[str, int] = {}
    for it in items:
        year = it["timestamp"][:4]
        by_year[year] = by_year.get(year, 0) + int(it["views"])
    return by_year


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if not BIOS.exists():
        raise SystemExit(f"missing {BIOS} -- run acquire_wikipedia_bios.py first")
    bios = json.loads(BIOS.read_text(encoding="utf-8"))
    candidates = [(k, v) for k, v in bios["players"].items()
                  if v.get("status") == "ok" and v.get("wiki_title")]
    print(f"{len(candidates)} resolved players available for pageview lookup")

    out = {"built": None, "source": "wikimedia pageviews REST API (no key)",
           "window": [START, _end_month()], "players": {}}
    if OUT.exists() and not args.refresh:
        out = json.loads(OUT.read_text(encoding="utf-8"))
        out["window"] = [START, _end_month()]

    todo = [(k, v) for k, v in candidates if k not in out["players"]]
    n_cached = len(candidates) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} to fetch this run ({n_cached} already cached)")

    ok = 0
    for i, (key, v) in enumerate(todo):
        title = v["wiki_title"]
        items = fetch_monthly(title)
        if items is None:
            print(f"  [{i+1}/{len(todo)}] {title}: FAILED (retries exhausted)")
            continue
        by_year = to_annual(items)
        out["players"][key] = {"title": title, "by_year": by_year}
        if by_year:
            ok += 1
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}] {title}: {len(by_year)} years, "
                  f"total={sum(by_year.values()):,}", flush=True)
            OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        time.sleep(args.sleep)

    out["built"] = datetime.now().date().isoformat()
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    n_with_data = sum(1 for p in out["players"].values() if p["by_year"])
    print(f"\nwrote {OUT}: {len(out['players'])} players cached, {n_with_data} with any pageview data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
