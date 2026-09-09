"""Emit the cross-domain as-of join table (Season Clock spine).

Joins, offline, from existing artifacts:

  venue naming window + team + season  ->  sponsor ticker
  market_cultural player-season         ->  endorse_m (masked, never imputed)

Grain: one row per (sport, player_id, season_year) that falls inside a
**named-window** venue->sponsor edge. This is co-occurrence under an honest
naming year, NOT an endorsement deal and NOT a NIL match.

  python pipeline/build_asof_join.py
  python pipeline/build_asof_join.py --data-root PATH --out PATH

Writes (under data/, gitignored):
  asof_join.jsonl          player-season rows
  asof_join_meta.json      coverage + honesty contract
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_paths import ESTATE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_GRID_EMB = ROOT / "pipeline" / "data" / "gridiron_season_emb.npz"
DEFAULT_HOOPS_META = ESTATE / "vector-hoops" / "assets" / "player_meta.json"
DEFAULT_HOOPS_EMB = ESTATE / "vector-hoops" / "pipeline" / "data" / "embedding_v3.npz"

# Honesty contract — repeated in meta so UI cannot quietly overclaim.
DISCLAIMER = (
    "Rows mean: this player-season was on a team whose home venue carried an "
    "S&P sponsor name in that season_year (named-window filter). "
    "They do NOT mean an endorsement contract, NIL deal, or school match."
)

SCHEMA = (
    "sport",
    "player_id",
    "player_name",
    "season_year",
    "season_label",
    "team",
    "venue",
    "ticker",
    "company",
    "sponsor_name_from",
    "naming_year_source",
    "endorse_m",
    "m_endorse",
)


def load_venue_edges(path: Path) -> dict[str, list[dict]]:
    """Return {sport: [edge,...]} using named-window-capable edge lists only."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    if isinstance(doc.get("edges"), list):
        out["gridiron"] = [e for e in doc["edges"] if e.get("sponsor_name_from")]
    hoops = doc.get("hoops") or {}
    if isinstance(hoops.get("edges"), list):
        out["hoops"] = [e for e in hoops["edges"] if e.get("sponsor_name_from")]
    return out


def index_endorse(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    """(sport, player_id, year) -> market_cultural row fields we keep."""
    idx: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        y = r.get("year")
        if y is None:
            continue
        key = (str(r["sport"]), str(r["player_id"]), int(y))
        idx[key] = {
            "endorse_m": r.get("endorse_m"),
            "m_endorse": int(r.get("m_endorse") or 0),
            "player_name": r.get("name"),
            "season_label": r.get("season"),
        }
    return idx


def expand_gridiron(
    edges: list[dict],
    emb_path: Path,
    endorse_idx: dict[tuple[str, str, int], dict],
) -> list[dict]:
    z = np.load(emb_path, allow_pickle=True)
    teams = np.array([str(x) for x in z["team"]])
    seasons = np.array([int(x) for x in z["season"]])
    gsis = np.array([str(x) for x in z["gsis"]])
    names = np.array([str(x) for x in z["name"]])
    rows: list[dict] = []
    for e in edges:
        named = int(e["sponsor_name_from"])
        ab = str(e["team_code"])
        mask = (teams == ab) & (seasons >= named)
        for i in np.flatnonzero(mask):
            y = int(seasons[i])
            pid = str(gsis[i])
            m = endorse_idx.get(("gridiron", pid, y), {})
            rows.append(
                {
                    "sport": "gridiron",
                    "player_id": pid,
                    "player_name": m.get("player_name") or str(names[i]),
                    "season_year": y,
                    "season_label": m.get("season_label") or str(y),
                    "team": ab,
                    "venue": e["venue"],
                    "ticker": e["ticker"],
                    "company": e["company"],
                    "sponsor_name_from": named,
                    "naming_year_source": e.get("naming_year_source") or "unknown",
                    "endorse_m": m.get("endorse_m"),
                    "m_endorse": int(m.get("m_endorse") or 0),
                }
            )
    return rows


def _hoops_start_year(season_label: str) -> int | None:
    """Compare naming years to season *start* year (build_venue_edges discipline)."""
    s = str(season_label)
    try:
        return int(s.split("-", 1)[0])
    except ValueError:
        return None


def expand_hoops(
    edges: list[dict],
    meta_path: Path,
    emb_path: Path,
    endorse_idx: dict[tuple[str, str, int], dict],
) -> list[dict]:
    roster = json.loads(meta_path.read_text(encoding="utf-8"))["roster"]
    hz = np.load(emb_path, allow_pickle=True)
    names = [str(x) for x in hz["name"]]
    seasons = [str(x) for x in hz["season"]]
    pids = [str(x) for x in hz["player_id"]]
    hkeys = [f"{n}|{s}" for n, s in zip(names, seasons, strict=True)]
    collide: dict[str, set[str]] = defaultdict(set)
    for k, pid in zip(hkeys, pids, strict=True):
        collide[k].add(pid)
    ambiguous = {k for k, v in collide.items() if len(v) > 1}
    if ambiguous:
        raise SystemExit(
            f"REFUSING: {len(ambiguous)} Name|Season keys map to >1 player_id "
            f"(e.g. {sorted(ambiguous)[:3]}). Same guard as build_venue_edges.py."
        )

    # end-year for market_cultural join (Forbes/awards use end year)
    def end_year(label: str) -> int | None:
        parts = str(label).split("-")
        if len(parts) == 1:
            try:
                return int(parts[0])
            except ValueError:
                return None
        try:
            start = int(parts[0])
            end = int(parts[1])
            if end < 100:
                end += start - (start % 100)
                if end < start:
                    end += 100
            return end
        except ValueError:
            return None

    rows: list[dict] = []
    for e in edges:
        named = int(e["sponsor_name_from"])
        ab = str(e["team_code"])
        for k, n, s, pid in zip(hkeys, names, seasons, pids, strict=True):
            if roster.get(k) != ab:
                continue
            start = _hoops_start_year(s)
            if start is None or start < named:
                continue
            y = end_year(s)
            if y is None:
                continue
            m = endorse_idx.get(("hoops", pid, y), {})
            rows.append(
                {
                    "sport": "hoops",
                    "player_id": pid,
                    "player_name": m.get("player_name") or n,
                    "season_year": y,
                    "season_label": m.get("season_label") or s,
                    "team": ab,
                    "venue": e["venue"],
                    "ticker": e["ticker"],
                    "company": e["company"],
                    "sponsor_name_from": named,
                    "naming_year_source": e.get("naming_year_source") or "unknown",
                    "endorse_m": m.get("endorse_m"),
                    "m_endorse": int(m.get("m_endorse") or 0),
                }
            )
    return rows


def build_meta(rows: list[dict], sources: dict) -> dict:
    by_sport: dict[str, int] = defaultdict(int)
    endorse_on = 0
    tickers: set[str] = set()
    for r in rows:
        by_sport[r["sport"]] += 1
        if r.get("m_endorse"):
            endorse_on += 1
        tickers.add(str(r["ticker"]))
    return {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "join_key": "season_year",
        "grain": "sport,player_id,season_year (named-window venue sponsor edges)",
        "disclaimer": DISCLAIMER,
        "schema": list(SCHEMA),
        "n_rows": len(rows),
        "n_rows_by_sport": dict(by_sport),
        "n_distinct_tickers": len(tickers),
        "n_rows_with_endorse_m": endorse_on,
        "sources": sources,
    }


def write_outputs(rows: list[dict], meta: dict, out_jsonl: Path, out_meta: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--venue-edges", type=Path, default=None)
    ap.add_argument("--market-cultural", type=Path, default=None)
    ap.add_argument("--grid-emb", type=Path, default=DEFAULT_GRID_EMB)
    ap.add_argument("--hoops-meta", type=Path, default=DEFAULT_HOOPS_META)
    ap.add_argument("--hoops-emb", type=Path, default=DEFAULT_HOOPS_EMB)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--out-meta", type=Path, default=None)
    ap.add_argument("--skip-hoops", action="store_true")
    args = ap.parse_args(argv)

    data = args.data_root
    venue_path = args.venue_edges or (data / "venue_edges.json")
    mc_path = args.market_cultural or (data / "market_cultural" / "market_cultural.json")
    out_jsonl = args.out_jsonl or (data / "asof_join.jsonl")
    out_meta = args.out_meta or (data / "asof_join_meta.json")

    for p in (venue_path, mc_path, args.grid_emb):
        if not p.exists():
            print(f"FAIL: missing {p}", file=sys.stderr)
            return 2

    edges = load_venue_edges(venue_path)
    mc = json.loads(mc_path.read_text(encoding="utf-8"))
    endorse_idx = index_endorse(mc.get("rows") or [])

    rows: list[dict] = []
    sources = {
        "venue_edges": str(venue_path),
        "market_cultural": str(mc_path),
        "gridiron_emb": str(args.grid_emb),
    }
    rows.extend(expand_gridiron(edges.get("gridiron") or [], args.grid_emb, endorse_idx))

    if not args.skip_hoops:
        if args.hoops_meta.exists() and args.hoops_emb.exists():
            sources["hoops_meta"] = str(args.hoops_meta)
            sources["hoops_emb"] = str(args.hoops_emb)
            rows.extend(
                expand_hoops(
                    edges.get("hoops") or [],
                    args.hoops_meta,
                    args.hoops_emb,
                    endorse_idx,
                )
            )
        else:
            sources["hoops"] = "skipped: meta/emb missing"

    meta = build_meta(rows, sources)
    write_outputs(rows, meta, out_jsonl, out_meta)
    print(
        f"Wrote {meta['n_rows']} rows -> {out_jsonl} "
        f"(gridiron={meta['n_rows_by_sport'].get('gridiron', 0)} "
        f"hoops={meta['n_rows_by_sport'].get('hoops', 0)} "
        f"endorse_on={meta['n_rows_with_endorse_m']})"
    )
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
