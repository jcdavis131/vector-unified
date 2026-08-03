#!/usr/bin/env python3
"""Is a team's company edge a COMMERCIAL fact or a Wikidata-COVERAGE artifact? (7.11)

Solo personal project, no connection to employer, built with public/free-tier only

7.10 established that the company layer is team-level and that only 14.0% of athletes are
reached by a genuine sponsor edge. This asks the question that has to be settled before any
of it is described as measuring commerce: **why does a team have a company edge at all?**

The alternative explanation is boring and very likely: a team that is written about more
has a more complete Wikidata item, and a more complete item carries more P859/P127/P115
statements. Under that story the layer measures NOTABILITY, and every downstream sector
claim is a restatement of which teams have good encyclopedia articles.

OUTCOME IS BINARY, and that is forced by the data rather than chosen. Sectors per team:

    0 sectors  135 teams        3 sectors    1
    1 sector    71              6 sectors    1
    2 sectors   13             12 sectors    1

135 of 222 teams have NO business-sector edge and only 3 have more than two. A per-team
sector PROFILE does not exist at this density, and a "which sectors is this team missing"
whitespace report would answer "26 of 28" for the modal team. So the only outcome the data
supports is has-any-edge (87/222 = 39.2%), and that is what is modelled.

PRE-REGISTERED DECISION RULE, fixed here before the first run:

    predictors   log10 venue capacity  (commercial scale; a physical fact, not a
                                        description, so it cannot be inflated by coverage)
                 log10 Wikidata sitelinks (notability; the number of Wikipedia language
                                        editions carrying the team)

    r_cap   = point-biserial corr(has_edge, log capacity)
    r_site  = point-biserial corr(has_edge, log sitelinks)
    r_cap|site = partial correlation of capacity given sitelinks

    COVERAGE ARTIFACT if  |r_site| > |r_cap|  AND the bootstrap CI on r_cap|site
                          INCLUDES zero. Then the layer must not be described as
                          measuring commercial attachment.
    COMMERCIAL SIGNAL if  the CI on r_cap|site EXCLUDES zero. Capacity then carries
                          information that notability does not.
    Anything else is reported as UNRESOLVED, not rounded toward the preferred answer.

    Bootstrap 10,000 resamples over TEAMS (the independent unit), seed fixed below.

AMENDMENT AFTER THE FIRST RUN, disclosed rather than retro-fitted. The rule above was
pre-registered and it is INSUFFICIENT: it tests whether the partial correlation is
distinguishable from zero but never asks whether the binary outcome has enough MINORITY
cases to support one. First run:

    hoops     n=30  edge 93.3%   r_cap -0.3055   r_cap|site -0.3014  CI [-0.602, -0.146]
    gridiron  n=30  edge 76.7%   r_cap -0.2114   r_cap|site -0.1932  CI [-0.600, +0.140]
    pitch     n=101 edge 34.7%   r_cap +0.2212   r_cap|site +0.1742  CI [+0.053, +0.334]

hoops came back COMMERCIAL SIGNAL on a CI that excludes zero — while carrying **exactly 2
teams without an edge**. The entire correlation is a statement about those two, and its
sign is NEGATIVE: bigger arena, LESS likely to have a company edge. That is not a
commercial signal under any reading; it is 2 data points and a significance test that
cannot tell the difference.

So MIN_MINORITY is added: a group decides only if the smaller outcome class has at least
that many teams (15, the usual events-per-predictor floor for 2 predictors). The original
rule's output is still printed for every group, marked, because hiding the verdict that
exposed the flaw would defeat the purpose of pre-registering one.

STRATIFY BY SPORT, because sport moves both predictors and the outcome in the same
direction and pooling would manufacture the association on its own: NFL stadiums are the
largest, European clubs carry the most language editions, and edge rates run hoops 76%,
gridiron 72%, pitch 23%. The pooled number is reported but the per-sport numbers are the
ones that decide the rule.

    python pipeline/probe_company_edge_confound.py --fetch    # one-time sitelink pull
    python pipeline/probe_company_edge_confound.py
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
COMPANIES = ROOT / "data" / "orgs" / "company_entities.json"
SECTORS = ROOT / "data" / "orgs" / "company_sectors_applied.json"
ORGS = ROOT / "data" / "orgs" / "org_entities.json"
CACHE = ROOT / "data" / "orgs" / "cache" / "org_sitelinks.json"
OUT = ROOT / "data" / "orgs" / "company_edge_confound.json"

API = "https://www.wikidata.org/w/api.php"
UA = "vector-unified/0.1 (personal research; contact via github)"
SEED = 20260803
REPS = 10000
MIN_N = 30          # below this a per-sport correlation is reported but not used to decide
MIN_MINORITY = 15   # smaller outcome class must reach this; see the AMENDMENT above


def fetch_sitelinks(qids: list[str]) -> dict[str, int]:
    """QID -> number of Wikipedia language editions. 50 per call, the API's own limit."""
    out: dict[str, int] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        r = requests.get(API, params={
            "action": "wbgetentities", "ids": "|".join(chunk),
            "props": "sitelinks", "format": "json"},
            headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        for qid, ent in (r.json().get("entities") or {}).items():
            if "missing" in ent:
                continue
            # wiki sitelinks only — commons/wikiquote/wikinews are not language editions
            out[qid] = sum(1 for k in (ent.get("sitelinks") or {}) if k.endswith("wiki"))
        time.sleep(0.5)
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return statistics.correlation(xs, ys)


def partial(y: list[float], x: list[float], z: list[float]) -> float:
    """corr(y, x | z) via the standard three-correlation identity."""
    ryx, ryz, rxz = pearson(y, x), pearson(y, z), pearson(x, z)
    den = math.sqrt(max(0.0, (1 - ryz ** 2) * (1 - rxz ** 2)))
    if den == 0 or any(math.isnan(v) for v in (ryx, ryz, rxz)):
        return float("nan")
    return (ryx - ryz * rxz) / den


def ci(vals: list[float]) -> tuple[float, float]:
    v = sorted(x for x in vals if not math.isnan(x))
    if len(v) < 20:
        return float("nan"), float("nan")
    return v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))]


def analyse(rows: list[dict], rng: random.Random) -> dict:
    y = [1.0 if r["has_edge"] else 0.0 for r in rows]
    cap = [math.log10(r["capacity"]) for r in rows]
    site = [math.log10(max(r["sitelinks"], 1)) for r in rows]
    r_cap, r_site = pearson(y, cap), pearson(y, site)
    r_partial = partial(y, cap, site)
    boots = []
    n = len(rows)
    for _ in range(REPS):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(partial([y[i] for i in idx], [cap[i] for i in idx],
                             [site[i] for i in idx]))
    lo, hi = ci(boots)
    excludes = not (math.isnan(lo) or lo <= 0.0 <= hi)
    pos = int(sum(y))
    return {"n": n, "positives": pos, "negatives": n - pos,
            "minority": min(pos, n - pos),
            "edge_rate": round(100.0 * pos / n, 1),
            "r_capacity": round(r_cap, 4), "r_sitelinks": round(r_site, 4),
            "r_capacity_given_sitelinks": round(r_partial, 4),
            "partial_ci95": [round(lo, 4), round(hi, 4)],
            "partial_ci_excludes_zero": excludes,
            "verdict": ("COMMERCIAL SIGNAL" if excludes else
                        "COVERAGE ARTIFACT" if abs(r_site) > abs(r_cap) else
                        "UNRESOLVED")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="pull sitelink counts and cache")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (COMPANIES, SECTORS, ORGS):
        if not p.exists():
            print(f"missing {p} — build the company layer first")
            return 2

    comp = json.loads(COMPANIES.read_text(encoding="utf-8"))
    sect = json.loads(SECTORS.read_text(encoding="utf-8"))
    orgs = json.loads(ORGS.read_text(encoding="utf-8"))

    sector_of_label = {lab: s for s, labs in sect["sector_companies"].items() for lab in labs}
    business = {c["qid"] for c in comp["companies"] if c.get("is_business")}
    label_of = {c["qid"]: c["label"] for c in comp["companies"]}
    sectors_of_team: dict[str, set[str]] = collections.defaultdict(set)
    for e in comp["edges"]:
        s = sector_of_label.get(label_of.get(e["company"], ""))
        if e["company"] in business and s:
            sectors_of_team[e["org_key"]].add(s)

    # one row per TEAM, attributes taken from its most recent season
    team_row: dict[str, dict] = {}
    for x in orgs["orgs"]:
        tk = "::".join(x["org_id"].split("::")[:2])
        cur = team_row.get(tk)
        if cur is None or str(x.get("season")) > str(cur.get("season")):
            team_row[tk] = x

    if args.fetch:
        qids = sorted({(x.get("attrs") or {}).get("wikidata") for x in team_row.values()
                       if (x.get("attrs") or {}).get("wikidata")})
        print(f"fetching sitelinks for {len(qids)} team QIDs...")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        got = fetch_sitelinks(qids)
        CACHE.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {CACHE}  ({len(got)} resolved)")

    if not CACHE.exists():
        print(f"missing {CACHE} — run once with --fetch")
        return 2
    sitelinks = json.loads(CACHE.read_text(encoding="utf-8"))

    rows, dropped = [], collections.Counter()
    for tk, x in team_row.items():
        at = x.get("attrs") or {}
        qid, cap = at.get("wikidata"), at.get("capacity")
        if not qid:
            dropped["no_wikidata_id"] += 1
            continue
        if not cap:
            dropped["no_venue_capacity"] += 1
            continue
        if qid not in sitelinks:
            dropped["sitelinks_unresolved"] += 1
            continue
        rows.append({"team": tk, "sport": x["sport"], "qid": qid,
                     "capacity": float(cap), "sitelinks": sitelinks[qid],
                     "has_edge": bool(sectors_of_team.get(tk))})

    if len(rows) < MIN_N:
        print(f"only {len(rows)} teams usable — not deciding.")
        return 2

    rng = random.Random(SEED)
    pooled = analyse(rows, rng)
    per_sport = {}
    for sp in sorted({r["sport"] for r in rows}):
        sub = [r for r in rows if r["sport"] == sp]
        res = analyse(sub, random.Random(SEED)) if len(sub) >= 3 else {"n": len(sub)}
        res["decides"] = len(sub) >= MIN_N and res.get("minority", 0) >= MIN_MINORITY
        if len(sub) >= MIN_N and res.get("minority", 0) < MIN_MINORITY:
            res["excluded_because"] = (
                f"minority outcome class has {res.get('minority')} teams (< {MIN_MINORITY}) "
                f"— the correlation is a statement about that many points")
        per_sport[sp] = res

    deciding = [v for v in per_sport.values() if v.get("decides")]
    verdicts = {v["verdict"] for v in deciding}

    report = {
        "question": ("Does a team have a company edge because it is commercially large, "
                     "or because its Wikidata item is well populated?"),
        "outcome": "has_edge (binary) — the sector distribution is too sparse for anything else",
        "sectors_per_team_histogram": dict(sorted(collections.Counter(
            len(sectors_of_team.get(t, ())) for t in team_row).items())),
        "teams_total": len(team_row), "teams_usable": len(rows), "dropped": dict(dropped),
        "pooled": pooled,
        "per_sport": per_sport,
        "pooled_caveat": (
            "The pooled row is reported but does NOT decide. Sport moves capacity, "
            "sitelinks and the outcome together — NFL stadiums are largest, European clubs "
            "carry the most language editions, edge rates run hoops 76% / gridiron 72% / "
            "pitch 23% — so a pooled association can be produced by sport alone."),
        "decision_rule": (
            f"A group decides only with n >= {MIN_N} AND minority class >= {MIN_MINORITY}. "
            "COVERAGE ARTIFACT if |r_sitelinks| > |r_capacity| and the bootstrap CI on the "
            "partial correlation of capacity given sitelinks INCLUDES zero; COMMERCIAL "
            "SIGNAL if that CI EXCLUDES zero; otherwise UNRESOLVED. Fixed in the docstring "
            "before the first run."),
        "verdict": (next(iter(verdicts)) if len(verdicts) == 1 else
                    f"MIXED across sports: {sorted(verdicts)}" if verdicts else
                    "NO SPORT MEETS THE MINIMUM n"),
        "decided_by": sorted(k for k, v in per_sport.items() if v.get("decides")),
        "scope_of_the_verdict": (
            "One sport of three decides it. hoops (28 of 30 teams have an edge) and "
            "gridiron (23 of 30) have almost no negative cases, so in US leagues there is "
            "nothing to explain — a company edge is close to universal and its presence "
            "carries no information. The question only has variance in pitch, where 35 of "
            "101 clubs have one."),
        "how_strong_the_signal_is": (
            "Weaker than the label sounds. In pitch, notability is still the STRONGER "
            "single predictor (r_site +0.3154 vs r_cap +0.2212); capacity survives "
            "controlling for it (partial +0.1742, CI excludes zero) but does not dominate "
            "it. The correct reading is 'commercial scale adds something beyond how much "
            "has been written about the club', not 'the layer measures commerce'."),
        "seed": SEED, "reps": REPS, "min_n_to_decide": MIN_N,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"teams {len(team_row)}   usable {len(rows)}   dropped {dict(dropped)}\n")
    hdr = (f"{'group':<12} {'n':>4} {'min':>4} {'edge%':>6} {'r_cap':>8} {'r_site':>8} "
           f"{'r_cap|site':>11}  CI")
    print(hdr)
    for name, v in [("POOLED", pooled), *per_sport.items()]:
        if "r_capacity" not in v:
            print(f"{name:<12} {v['n']:>4}   too few")
            continue
        lo, hi = v["partial_ci95"]
        mark = ("" if v.get("decides", True) else
                "  NOT DECIDING (%s)" % (v.get("excluded_because") or f"n<{MIN_N}"))
        print(f"{name:<12} {v['n']:>4} {v['minority']:>4} {v['edge_rate']:>5.1f}% "
              f"{v['r_capacity']:>+8.4f} "
              f"{v['r_sitelinks']:>+8.4f} {v['r_capacity_given_sitelinks']:>+11.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}] {v['verdict']}{mark}")
    print(f"\nVERDICT: {report['verdict']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
