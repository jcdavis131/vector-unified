#!/usr/bin/env python3
"""Do SCHEDULE features identify a tennis player better than performance rates?

Solo personal project, no connection to employer, built with public/free-tier only

probe_tennis_feature_identity.py established the criterion a retrieval feature must meet —
stable within a player across adjacent years AND not shared across the field — and found
that the existing 16 mostly fail the second half. It also produced a lead worth chasing:

    BIG_EVENT_SHARE   autocorr 0.6170   third highest of the 16

BIG_EVENT_SHARE is a MIX feature. It measures WHAT a player plays, not how well. The
hypothesis here is that a schedule is a fingerprint — which surfaces, which tiers, how many
distinct events, indoor or out — because a player's calendar is idiosyncratic and repeats
year to year, whereas win rates regress and rank is shared.

TWO FAMILIES, tested against each other rather than assumed:

    SCHEDULE / MIX      what you play      surface shares, indoor share, tier shares,
                        (hypothesis)       distinct tournaments, distinct locations
    SHOT SHAPE          how you win        tiebreak rate, bagel rate, comeback rate,
                        (control)          mean games per set

All are derived from pipeline/cache/tennis/*.xlsx, already on disk — 28 files, 2013-2026,
67,081 matches. No fetch, no key.

RATES AND SHARES, NOT COUNTS, matching build_tennis_matrix.py's stated rule that "every
feature is a rate, so a player with three matches and a player with sixty are on the same
scale". The two count-like candidates (distinct tournaments, distinct locations) are kept
BECAUSE they violate it — if a raw count identifies better than a rate, that is worth
knowing rather than excluding by convention.

    python pipeline/probe_tennis_candidate_features.py
    python pipeline/probe_tennis_candidate_features.py --check
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from acquire_tennis import path_for, read_sheet  # noqa: E402

MATRIX = ROOT / "pipeline" / "data" / "tennis_matrix.npz"
META = ROOT / "pipeline" / "data" / "meta_tennis_matrix.json"
OUT = ROOT / "data" / "tennis_candidate_features.json"
YEARS = range(2013, 2027)
K = 10

SCHEDULE = ("SURF_SHARE_HARD", "SURF_SHARE_CLAY", "SURF_SHARE_GRASS", "INDOOR_SHARE",
            "TIER_SHARE_TOP", "N_TOURNAMENTS", "N_LOCATIONS", "MEAN_EVENTS_PER_MONTH")
SHOT = ("TIEBREAK_RATE", "BAGEL_RATE", "COMEBACK_RATE", "MEAN_GAMES_PER_SET")
CANDIDATES = SCHEDULE + SHOT


def num(v, d=None):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return d


def collect(women: bool, acc: dict) -> None:
    tour = "wta" if women else "atp"
    for y in YEARS:
        p = path_for(y, women)
        if not p.exists():
            continue
        hdr, body = read_sheet(p)
        i = {c: k for k, c in enumerate(hdr)}
        if "Winner" not in i or "Loser" not in i:
            continue
        for r in body:
            surf = str(r[i["Surface"]]).strip().lower() if "Surface" in i else ""
            court = str(r[i["Court"]]).strip().lower() if "Court" in i else ""
            tier = str(r[i.get("Series", i.get("Tier", 0))]).strip().lower() if (
                "Series" in i or "Tier" in i) else ""
            loc = str(r[i["Location"]]).strip() if "Location" in i else ""
            trn = str(r[i["Tournament"]]).strip() if "Tournament" in i else ""
            date = str(r[i["Date"]]).strip() if "Date" in i else ""
            ws, ls = num(r[i.get("Wsets", 0)], 0) or 0, num(r[i.get("Lsets", 0)], 0) or 0
            sets_w = [(num(r[i[f"W{k}"]], None), num(r[i[f"L{k}"]], None))
                      for k in range(1, 6) if f"W{k}" in i and f"L{k}" in i]
            sets_w = [(a, b) for a, b in sets_w if a is not None and b is not None
                      and (a > 0 or b > 0)]

            for name, won in ((str(r[i["Winner"]]).strip(), 1),
                              (str(r[i["Loser"]]).strip(), 0)):
                if not name:
                    continue
                d = acc[(name, y, tour)]
                d["m"] += 1
                if surf in ("hard", "clay", "grass"):
                    d[f"surf_{surf}"] += 1
                if court == "indoor":
                    d["indoor"] += 1
                if any(t in tier for t in ("grand slam", "masters", "premier", "wta1000",
                                           "atp1000", "international gold")):
                    d["top_tier"] += 1
                if trn:
                    d["tourneys"].add(trn)
                if loc:
                    d["locs"].add(loc)
                if date:
                    d["dates"].append(date)
                for a, b in sets_w:
                    d["sets"] += 1
                    hi, lo = (a, b) if a >= b else (b, a)
                    if hi == 7 and lo == 6:
                        d["tb"] += 1
                    if lo <= 1 and hi >= 6:
                        d["bagel"] += 1
                    d["games"] += a + b
                # comeback: won the match after losing the first set
                if won and sets_w and sets_w[0][0] < sets_w[0][1]:
                    d["comeback"] += 1
                if won:
                    d["wins"] += 1


def build() -> dict[tuple, dict]:
    acc: dict[tuple, dict] = collections.defaultdict(lambda: {
        "m": 0, "wins": 0, "surf_hard": 0, "surf_clay": 0, "surf_grass": 0,
        "indoor": 0, "top_tier": 0, "tourneys": set(), "locs": set(), "dates": [],
        "sets": 0, "tb": 0, "bagel": 0, "games": 0, "comeback": 0,
    })
    collect(False, acc)
    collect(True, acc)
    out = {}
    for key, d in acc.items():
        m, s = d["m"], d["sets"]
        if m < 3:
            continue
        months = len({str(x)[:6] for x in d["dates"]}) or 1
        out[key] = {
            "SURF_SHARE_HARD": d["surf_hard"] / m,
            "SURF_SHARE_CLAY": d["surf_clay"] / m,
            "SURF_SHARE_GRASS": d["surf_grass"] / m,
            "INDOOR_SHARE": d["indoor"] / m,
            "TIER_SHARE_TOP": d["top_tier"] / m,
            "N_TOURNAMENTS": float(len(d["tourneys"])),
            "N_LOCATIONS": float(len(d["locs"])),
            "MEAN_EVENTS_PER_MONTH": len(d["tourneys"]) / months,
            "TIEBREAK_RATE": (d["tb"] / s) if s else None,
            "BAGEL_RATE": (d["bagel"] / s) if s else None,
            "COMEBACK_RATE": d["comeback"] / m,
            "MEAN_GAMES_PER_SET": (d["games"] / s) if s else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    a = np.load(MATRIX, allow_pickle=True)
    X, M = a["X"].astype(np.float64), a["M"].astype(np.float64)
    feats = [str(f) for f in a["features"]]
    meta = json.loads(META.read_text(encoding="utf-8"))
    if len(meta) != X.shape[0]:
        print(f"meta {len(meta)} vs matrix {X.shape[0]} — refusing positional join")
        return 2

    cand = build()
    keys = [(m["player"], m["year"], m["tour"]) for m in meta]
    hit = sum(1 for k in keys if k in cand)
    print(f"matrix rows {len(keys)}   candidate rows built {len(cand)}   "
          f"joined {hit} ({100*hit/len(keys):.1f}%)")
    if hit < 0.5 * len(keys):
        print("join rate too low — the key convention does not match, refusing")
        return 2

    C = np.full((len(keys), len(CANDIDATES)), np.nan)
    for r, k in enumerate(keys):
        v = cand.get(k)
        if v:
            for c, name in enumerate(CANDIDATES):
                if v.get(name) is not None:
                    C[r, c] = v[name]
    CM = (~np.isnan(C)).astype(float)
    C = np.nan_to_num(C)

    tours = np.array([m["tour"] for m in meta])
    idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
    pairs = [(i, idx[(m["player"], m["year"] + 1, m["tour"])]) for i, m in enumerate(meta)
             if (m["player"], m["year"] + 1, m["tour"]) in idx]
    src = np.array([p for p, _ in pairs])
    dst = np.array([q for _, q in pairs])

    print(f"\n  {'candidate':26} {'family':9} {'pairs':>6} {'autocorr':>9}")
    rows = []
    for c, name in enumerate(CANDIDATES):
        ok = (CM[src, c] > 0) & (CM[dst, c] > 0)
        n = int(ok.sum())
        r = (float(np.corrcoef(C[src[ok], c], C[dst[ok], c])[0, 1])
             if n > 50 and C[src[ok], c].std() > 0 and C[dst[ok], c].std() > 0
             else float("nan"))
        fam = "schedule" if name in SCHEDULE else "shot"
        rows.append({"feature": name, "family": fam, "n_pairs": n,
                     "autocorr": None if np.isnan(r) else round(r, 4)})
        print(f"  {name:26} {fam:9} {n:>6} "
              f"{('n/a' if np.isnan(r) else f'{r:.4f}'):>9}")

    def recall(V, VM):
        F = np.hstack([V, VM])
        mu, sd = F.mean(0), F.std(0)
        sd[sd == 0] = 1.0
        E = (F - mu) / sd
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        h = 0
        for q, t in pairs:
            same = np.where(tours == tours[q])[0]
            same = same[same != q]
            if t in same[np.argsort(-(En[same] @ En[q]))][:K]:
                h += 1
        return h / len(pairs)

    ident = json.loads((ROOT / "data" / "tennis_feature_identity.json")
                       .read_text(encoding="utf-8"))
    ac = {r["feature"]: (r["autocorr"] or 0.0) for r in ident["per_feature"]}
    keep = [j for j in range(len(feats)) if ac[feats[j]] >= 0.25]
    sch = [c for c, n in enumerate(CANDIDATES) if n in SCHEDULE]
    sht = [c for c, n in enumerate(CANDIDATES) if n in SHOT]

    tests = {
        "existing_9 (best known)": (X[:, keep], M[:, keep]),
        "existing_9 + schedule": (np.hstack([X[:, keep], C[:, sch]]),
                                  np.hstack([M[:, keep], CM[:, sch]])),
        "existing_9 + shot": (np.hstack([X[:, keep], C[:, sht]]),
                              np.hstack([M[:, keep], CM[:, sht]])),
        "existing_9 + all candidates": (np.hstack([X[:, keep], C]),
                                        np.hstack([M[:, keep], CM])),
        "schedule ONLY": (C[:, sch], CM[:, sch]),
        "shot ONLY": (C[:, sht], CM[:, sht]),
    }
    print(f"\n  {'feature set':30} {'recall@10':>10}")
    res, base = {}, None
    for name, (V, VM) in tests.items():
        r = recall(V, VM)
        if base is None:
            base = r
        res[name] = round(float(r), 4)
        print(f"  {name:30} {r:>10.4f}  {r-base:+.4f}")

    binom = float(np.sqrt(base * (1 - base) / max(1, len(pairs))))
    best = max(res, key=res.get)
    print(f"\n  best: {best} at {res[best]:.4f}  ({res[best]-base:+.4f} vs existing_9)")
    print(f"  binomial sd at n={len(pairs)}: {binom:.4f} -> "
          f"{abs(res[best]-base)/binom:.1f} sd")

    OUT.write_text(json.dumps({
        "question": ("Do SCHEDULE features (what a player plays) identify better than "
                     "performance rates (how well they play)?"),
        "hypothesis": ("BIG_EVENT_SHARE, a mix feature, autocorrelates at 0.6170 — third "
                       "highest of the existing 16. A player's calendar may be a "
                       "fingerprint: idiosyncratic and repeated, where win rates regress "
                       "and rank is shared across hundreds."),
        "source": ("pipeline/cache/tennis/*.xlsx, already on disk — 28 files, 2013-2026, "
                   "67,081 matches. No fetch, no key."),
        "join_rate_pct": round(100 * hit / len(keys), 1),
        "per_candidate": rows,
        "retrieval": res,
        "baseline_existing_9": round(float(base), 4),
        "binomial_sd": round(binom, 4),
        "n_pairs": len(pairs),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    if args.check and hit < 0.5 * len(keys):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
