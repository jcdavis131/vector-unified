"""Vector Unified — ask the entity graph a question, with the confounds handled.

First query layer over athlete -> org -> attrs. Everything before this built registries;
this is the part that returns an answer.

    python pipeline/query_graph.py                 # all questions
    python pipeline/query_graph.py --q archetype   # one

Q1 archetype x market scale
    Do cross-sport archetypes concentrate in large-market organizations?

TWO METHOD DECISIONS THAT DECIDE WHETHER THE ANSWER MEANS ANYTHING.

1. CAPACITY IS RANKED WITHIN SPORT, never compared raw. NFL stadiums run ~70k and NBA
   arenas ~19k, so a raw cross-sport capacity comparison ranks every gridiron org above
   every hoops org and then reports the sport mix of each archetype as if it were a market
   finding. Percentile-within-sport removes the format offset and leaves the thing actually
   asked: is this archetype in a BIG club FOR ITS SPORT.

2. A SHUFFLE BASELINE RUNS EVERY TIME. Archetype labels are permuted and the same statistic
   recomputed, so the spread you see has something to be compared against. Without it any
   ordering looks like a finding — 12 groups over 20,719 rows will always produce a
   highest and a lowest.

The verdict is printed as a comparison against that baseline, and when the observed spread
does not clear it the output says so instead of ranking the groups anyway.

LIMITS, stated up front because they bound every number below:
  - capacity is a VENUE property standing in for market size. It is not revenue, and
    Wikidata has revenue for 1 of 180 orgs (measured), so this is the available proxy
    rather than the right one.
  - capacity is STATIC per club, so a franchise that moved or rebuilt carries one value
    across all seasons.
  - 86.1% of enriched orgs have capacity; rows without it are dropped, not zero-filled,
    and the dropped count is reported.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIFIED = ROOT / "assets" / "unified.json"
ORGS = ROOT / "data" / "orgs" / "org_entities.json"

SHUFFLES = 200
SEED = 7


def arch_labels() -> dict[str, str]:
    """A0..A11 -> the human-authored label. Opaque IDs make a finding unsellable.

    "A1 carries a star premium" is not actionable; "volume scorers carry a star premium"
    is. Labels come from data/archetype_map.json's hand-authored taxonomy, which is the
    same anchor the cross-sport alignment was built against.

    Worth recording: the gridiron_hint for A0 reads "QB / high-usage RB / WR1". At Q5 I
    called gridiron A0's 20.41% pay share "a quarterback" from the number alone, before
    reading this file. The taxonomy independently agrees.
    """
    p = ROOT / "data" / "archetype_map.json"
    if not p.exists():
        return {}
    try:
        tax = json.loads(p.read_text(encoding="utf-8")).get("taxonomy", [])
    except Exception:
        return {}
    return {t["id"]: t.get("label", "") for t in tax if t.get("id")}


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def load_joined() -> tuple[list[dict], dict]:
    """player-season rows joined to their org, with capacity as a WITHIN-SPORT percentile."""
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    orgs = {o["org_id"]: o for o in doc["orgs"]}

    # (norm, sport, season) -> org_id. A traded player has several; keep them all and let
    # the row count reflect that rather than silently picking one.
    edge_index: dict[tuple, list[str]] = defaultdict(list)
    for e in doc["edges"]:
        if e.get("org_id"):
            edge_index[(e["norm"], e["sport"], str(e["season"]))].append(e["org_id"])

    # Capacity percentile is computed per sport, so the format offset cannot leak in.
    caps_by_sport: dict[str, list[int]] = defaultdict(list)
    for o in orgs.values():
        c = (o.get("attrs") or {}).get("capacity")
        if c:
            caps_by_sport[o["sport"]].append(c)
    for v in caps_by_sport.values():
        v.sort()

    def pct_within_sport(sport: str, cap: int) -> float:
        v = caps_by_sport[sport]
        below = sum(1 for x in v if x < cap)
        return 100.0 * below / max(len(v) - 1, 1)

    rows, dropped = [], {"no_edge": 0, "no_capacity": 0, "no_arch": 0}
    for p in players:
        arch = p.get("cross_arch")
        if arch is None:
            dropped["no_arch"] += 1
            continue
        oids = edge_index.get((norm_name(p["name"]), p["sport"], str(p["season"])))
        if not oids:
            dropped["no_edge"] += 1
            continue
        for oid in oids:
            o = orgs.get(oid)
            cap = (o.get("attrs") or {}).get("capacity") if o else None
            if not cap:
                dropped["no_capacity"] += 1
                continue
            rows.append(
                {
                    "sport": p["sport"],
                    "arch": str(arch),
                    "capacity": cap,
                    "cap_pct": pct_within_sport(p["sport"], cap),
                    "org": o["team"],
                    "city": (o.get("attrs") or {}).get("city"),
                }
            )
    return rows, dropped


def q_archetype(rows: list[dict], dropped: dict) -> None:
    print("Q1  Do cross-sport archetypes concentrate in large-market organizations?")
    print(f"    joined rows {len(rows)}   dropped {dropped}\n")
    if len(rows) < 100:
        print("    too few joined rows to say anything. Not ranking them.")
        return

    by_arch: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_arch[r["arch"]].append(r["cap_pct"])
    means = {a: statistics.mean(v) for a, v in by_arch.items() if len(v) >= 30}
    if len(means) < 2:
        print("    fewer than 2 archetypes with n>=30. Not ranking them.")
        return
    observed_spread = max(means.values()) - min(means.values())

    # Same statistic on permuted labels. If the real spread does not clear this, the
    # ordering below is what 12 groups do by chance and must not be read as a finding.
    rng = random.Random(SEED)
    labels = [r["arch"] for r in rows]
    vals = [r["cap_pct"] for r in rows]
    null_spreads = []
    for _ in range(SHUFFLES):
        rng.shuffle(labels)
        g: dict[str, list[float]] = defaultdict(list)
        for a, v in zip(labels, vals, strict=False):
            g[a].append(v)
        m = [statistics.mean(v) for v in g.values() if len(v) >= 30]
        if len(m) >= 2:
            null_spreads.append(max(m) - min(m))
    null_p95 = sorted(null_spreads)[int(0.95 * len(null_spreads))] if null_spreads else 0.0

    print(f"    {'archetype':12} {'n':>6}  mean capacity percentile within sport")
    for a, m in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"    {a:12} {len(by_arch[a]):6}  {m:5.1f}")
    print()
    print(f"    observed spread  : {observed_spread:5.1f} percentile points")
    print(f"    shuffle p95      : {null_p95:5.1f}   ({SHUFFLES} permutations, seed {SEED})")
    if observed_spread > null_p95:
        print("    VERDICT: spread exceeds the shuffled baseline — archetypes do differ")
        print("             in the market scale of the clubs they sit in.")
    else:
        print("    VERDICT: spread does NOT clear the shuffled baseline. The ordering above")
        print("             is what this many groups produce by chance. No finding.")


def q_roster_mix() -> None:
    """Q2: does an org's ROSTER COMPOSITION relate to how well it performs?

    hoops only, and that is a feature rather than a limitation: it is the one sport whose
    orgs carry a season-varying outcome (NET_RATING), and staying inside one sport removes
    the format confound that made Q1 need a within-sport ranking in the first place.

    Predictor is archetype concentration (HHI over the roster's cross_arch mix): 1.0 means
    every player shares an archetype, low means a broad mix. The question a front office
    actually asks — build around one profile, or diversify.
    """
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    orgs = {o["org_id"]: o for o in doc["orgs"]}

    key_to_arch: dict[tuple, str] = {}
    for p in players:
        if p["sport"] == "hoops" and p.get("cross_arch") is not None:
            key_to_arch[(norm_name(p["name"]), str(p["season"]))] = str(p["cross_arch"])

    roster: dict[str, list[str]] = defaultdict(list)
    for e in doc["edges"]:
        if e["sport"] != "hoops" or not e.get("org_id"):
            continue
        a = key_to_arch.get((e["norm"], str(e["season"])))
        if a is not None:
            roster[e["org_id"]].append(a)

    pts = []
    for oid, archs in roster.items():
        if len(archs) < 8:  # a roster, not a fragment
            continue
        net = (orgs[oid].get("features") or {}).get("NET_RATING")
        if net is None:
            continue
        n = len(archs)
        counts: dict[str, int] = defaultdict(int)
        for a in archs:
            counts[a] += 1
        hhi = sum((c / n) ** 2 for c in counts.values())
        pts.append((hhi, float(net)))

    print("\nQ2  Does roster archetype concentration relate to team performance?")
    print(f"    hoops team-seasons with >=8 archetyped players and a NET_RATING: {len(pts)}\n")
    if len(pts) < 50:
        print("    too few to say anything. Not reporting a correlation.")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    r = statistics.correlation(xs, ys)

    # Same shuffle discipline as Q1: permute the outcome, recompute, see what |r| chance
    # produces at this n. A correlation with no null to beat is a number, not a result.
    rng = random.Random(SEED)
    shuffled = list(ys)
    null = []
    for _ in range(SHUFFLES):
        rng.shuffle(shuffled)
        null.append(abs(statistics.correlation(xs, shuffled)))
    null_p95 = sorted(null)[int(0.95 * len(null))]

    lo = statistics.mean([y for x, y in pts if x <= statistics.quantiles(xs, n=4)[0]])
    hi = statistics.mean([y for x, y in pts if x >= statistics.quantiles(xs, n=4)[2]])
    print(f"    HHI range        : {min(xs):.3f} .. {max(xs):.3f}")
    print(f"    mean NET_RATING  : most-diverse quartile {lo:+.2f}   " f"most-concentrated quartile {hi:+.2f}")
    print(f"    pearson r        : {r:+.3f}")
    print(f"    shuffle p95 |r|  : {null_p95:.3f}   ({SHUFFLES} permutations, seed {SEED})")
    if abs(r) > null_p95:
        direction = "CONCENTRATED" if r > 0 else "DIVERSE"
        print(f"    VERDICT: clears the baseline. More {direction} rosters post better")
        print("             NET_RATING. Association only — this does not establish that")
        print("             composition CAUSES performance; good teams also attract")
        print("             particular player types.")
    else:
        print("    VERDICT: |r| does not clear the shuffled baseline. No finding.")


def _salary_index(sport: str) -> dict:
    """(norm_name, season) -> a pay figure, per sport, from whichever source has coverage.

    Two sources because two separate acquisitions produced them. docs/VALUE_SIGNAL_CENSUS.md
    records what each reaches:

        hoops     vector-hoops salary_market.json  11,408 player-seasons, SALARY_TEAM_PCT
        gridiron  market_cultural.json (Spotrac)   93.27% of rows, salary_m (NFL cap hit)
        pitch     0.58% — returns {} so Q3 prints "skipped" rather than a thin number

    UNITS DIFFER (cap share vs $M) and that is deliberately fine: the statistic is HHI over
    each roster's shares of ITS OWN total, which is scale-invariant. Comparing the raw
    values across sports would NOT be fine, and nothing here does that.
    """
    if sport == "hoops":
        p = ROOT.parent / "vector-hoops" / "pipeline" / "data" / "salary_market.json"
        if not p.exists():
            return {}
        return {
            (norm_name(r["name"]), str(r["season"])): float(r["SALARY_TEAM_PCT"])
            for r in json.loads(p.read_text(encoding="utf-8"))["players"]
            if r.get("SALARY_TEAM_PCT") is not None
        }
    p = ROOT / "data" / "market_cultural" / "market_cultural.json"
    if not p.exists():
        return {}
    return {
        (norm_name(r["name"]), str(r["season"])): float(r["salary_m"])
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]
        if r.get("sport") == sport and r.get("salary_m")
    }


def q_salary_concentration(sport: str = "hoops") -> None:
    """Q3: do larger-market clubs concentrate pay in fewer players?

    Q1 and Q2 were null for a reason worth acting on rather than repeating: the archetype
    layer is a ROLE taxonomy and a role label cannot carry a value question. This uses the
    two real value signals in the estate (see _salary_index).

    WHY CONCENTRATION AND NOT TOTAL PAYROLL. Both leagues have a salary cap, so total
    payroll is near-constant by RULE and correlating it with market size would produce a
    null that says nothing about markets and everything about league regulation.
    Concentration — one star on a max deal versus a balanced roster — is not capped away,
    and it is the actual strategic choice a front office makes.

    RUN ON BOTH LEAGUES ON PURPOSE. A single null is a result; the same null in a HARDER-
    capped league (NFL hard cap vs NBA soft cap + luxury tax) is a replication that could
    have failed and did not.

    Predictor is org capacity percentile WITHIN the sport; outcome is HHI over each
    roster's salary shares.
    """
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    orgs = {o["org_id"]: o for o in doc["orgs"]}
    sal = _salary_index(sport)
    if not sal:
        print(f"\nQ3 [{sport}] skipped: no salary source with usable coverage.")
        return

    caps = sorted(
        o["attrs"]["capacity"] for o in orgs.values() if o["sport"] == sport and (o.get("attrs") or {}).get("capacity")
    )

    def cap_pct(c: int) -> float:
        return 100.0 * sum(1 for x in caps if x < c) / max(len(caps) - 1, 1)

    shares: dict[str, list[float]] = defaultdict(list)
    for e in doc["edges"]:
        if e["sport"] != sport or not e.get("org_id"):
            continue
        s = sal.get((e["norm"], str(e["season"])))
        if s is not None:
            shares[e["org_id"]].append(float(s))

    pts = []
    for oid, vals in shares.items():
        if len(vals) < 8:
            continue
        o = orgs.get(oid)
        cap = (o.get("attrs") or {}).get("capacity") if o else None
        if not cap:
            continue
        tot = sum(vals)
        if tot <= 0:
            continue
        hhi = sum((v / tot) ** 2 for v in vals)
        pts.append((cap_pct(cap), hhi))

    print(f"\nQ3 [{sport}]  Do larger-market clubs concentrate pay in fewer players?")
    print(f"    {sport} team-seasons with >=8 salaried players and a capacity: {len(pts)}\n")
    if len(pts) < 50:
        print("    too few to say anything. Not reporting a correlation.")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    r = statistics.correlation(xs, ys)
    rng = random.Random(SEED)
    shuffled = list(ys)
    null = [abs(statistics.correlation(xs, shuffled)) for _ in range(SHUFFLES) if not rng.shuffle(shuffled)]
    p95 = sorted(null)[int(0.95 * len(null))]

    q = statistics.quantiles(xs, n=4)
    small = statistics.mean([y for x, y in pts if x <= q[0]])
    big = statistics.mean([y for x, y in pts if x >= q[2]])
    print(f"    salary HHI       : smallest-venue quartile {small:.4f}   " f"largest {big:.4f}")
    print(f"    pearson r        : {r:+.3f}")
    print(f"    shuffle p95 |r|  : {p95:.3f}   ({SHUFFLES} permutations, seed {SEED})")
    if abs(r) > p95:
        d = "MORE" if r > 0 else "LESS"
        print(f"    VERDICT: clears the baseline. Larger-venue clubs concentrate pay {d}.")
        print("             Association only. Venue capacity is a market proxy, not market")
        print("             revenue, and roster construction responds to many things.")
    else:
        print("    VERDICT: |r| does not clear the shuffled baseline. No finding.")


def q_league_style() -> None:
    """Q4: do national leagues field different player archetypes?

    Q1-Q3 all tested MARKET-SIZE effects and all came back null. Salary caps exist
    specifically to suppress market-size effects, so four nulls in a row may say more about
    league regulation than about the graph. This asks something a cap does not touch.

    The prior is real rather than fished for: national leagues are widely held to differ in
    style, and if the archetype layer carries anything about how a league plays, a Spanish
    club's archetype mix should differ from an English one. If this is ALSO null, the honest
    reading shifts — it would suggest the archetype layer is thinner than G4 0.978 implies,
    not that every sports question is null.

    pitch only: it is the sport with real country spread (58 org countries) and no cap.
    Statistic is the L1 distance between each country's archetype distribution and the
    global one, averaged over countries with enough rows. Shuffled country labels give the
    null, exactly as in Q1-Q3.
    """
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    orgs = {o["org_id"]: o for o in doc["orgs"]}

    arch_of = {
        (norm_name(p["name"]), str(p["season"])): str(p["cross_arch"])
        for p in players
        if p["sport"] == "pitch" and p.get("cross_arch") is not None
    }
    pairs = []
    for e in doc["edges"]:
        if e["sport"] != "pitch" or not e.get("org_id"):
            continue
        a = arch_of.get((e["norm"], str(e["season"])))
        country = (orgs.get(e["org_id"], {}).get("attrs") or {}).get("country")
        if a and country:
            pairs.append((country, a))

    print("\nQ4 [pitch]  Do national leagues field different player archetypes?")
    print(f"    athlete-org rows with both an archetype and a country: {len(pairs)}\n")
    if len(pairs) < 200:
        print("    too few rows. Not reporting.")
        return

    def mean_l1(rows: list[tuple[str, str]]) -> float:
        glob: dict[str, int] = defaultdict(int)
        by_c: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for c, a in rows:
            glob[a] += 1
            by_c[c][a] += 1
        gtot = sum(glob.values())
        gdist = {a: n / gtot for a, n in glob.items()}
        ds = []
        for c, counts in by_c.items():
            n = sum(counts.values())
            if n < 100:  # a league sample, not a handful of players
                continue
            ds.append(sum(abs(counts.get(a, 0) / n - gdist.get(a, 0)) for a in gdist))
        return statistics.mean(ds) if ds else 0.0

    observed = mean_l1(pairs)
    countries = [c for c, _ in pairs]
    archs = [a for _, a in pairs]
    rng = random.Random(SEED)
    null = []
    for _ in range(SHUFFLES):
        rng.shuffle(countries)
        null.append(mean_l1(list(zip(countries, archs, strict=False))))
    p95 = sorted(null)[int(0.95 * len(null))]

    kept = {c for c in set(c for c, _ in pairs) if sum(1 for x, _ in pairs if x == c) >= 100}
    print(f"    countries with n>=100 : {len(kept)}  {sorted(kept)[:6]}")
    print(f"    observed mean L1      : {observed:.4f}")
    print(f"    shuffle p95           : {p95:.4f}   ({SHUFFLES} permutations, seed {SEED})")
    if observed > p95:
        print("    VERDICT: clears the baseline. Archetype mix DOES vary by league country.")
        print("             Association only — squad composition also reflects transfer")
        print("             markets, academy pipelines and which clubs the corpus covers.")
    elif len(kept) < 5:
        # "Cannot detect" and "detected nothing" are different claims, and reporting them
        # in the same words is how an underpowered test gets cited as a negative result.
        # With 3 groups the permutation null is wide by construction, so a real effect of
        # this size would not clear it either.
        print(f"    VERDICT: UNDERPOWERED, not null. Only {len(kept)} countries clear n>=100,")
        print("             so the permutation null is wide by construction and an effect")
        print("             this size could not have been detected. This is NOT evidence")
        print("             that leagues field the same archetypes — it is evidence that")
        print("             2,108 pitch rows spread over 58 countries cannot answer it.")
        print("             Needs deeper per-league coverage, not a different statistic.")
    else:
        print("    VERDICT: does not clear the shuffled baseline. No finding.")


def q_archetype_pay(sport: str = "hoops") -> None:
    """Q5: does an athlete's archetype predict their share of the team's pay?

    THE POWER Q4 DID NOT HAVE. Q4 was underpowered — 3 country groups over 2,108 rows. This
    is the same kind of question (does the archetype layer carry value information?) on
    11,408 hoops rows and 4,962 gridiron rows, with 6 archetype groups. If the archetype
    layer carries ANY signal about what a player is worth, this is where it shows.

    NO MARKET-SIZE CONFOUND. Q1-Q3 all measured market effects, which caps are designed to
    suppress, so their nulls were partly about league regulation. Pay SHARE is within-team
    by construction: it asks which roles a club spends its money on, not how much money the
    club has. The cap cannot flatten this.

    A POSITIVE HERE IS LOAD-BEARING FOR THE WHOLE SESSION. Four market-size nulls are only
    credible as negatives ABOUT MARKETS if the same graph can be shown to find something
    else. A fifth null would instead point at the archetype layer being thin.
    """
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    sal = _salary_index(sport)
    if not sal:
        print(f"\nQ5 [{sport}] skipped: no salary source with usable coverage.")
        return

    arch_of = {
        (norm_name(p["name"]), str(p["season"])): str(p["cross_arch"])
        for p in players
        if p["sport"] == sport and p.get("cross_arch") is not None
    }

    # Pay is turned into a WITHIN-TEAM share before anything is compared, so a rich club and
    # a poor club contribute on the same scale and cap inflation across 30 years drops out.
    team_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in doc["edges"]:
        if e["sport"] != sport or not e.get("org_id"):
            continue
        k = (e["norm"], str(e["season"]))
        a, v = arch_of.get(k), sal.get(k)
        if a is not None and v:
            team_rows[e["org_id"]].append((a, float(v)))

    pairs: list[tuple[str, float]] = []
    for rows in team_rows.values():
        if len(rows) < 8:
            continue
        tot = sum(v for _, v in rows)
        if tot <= 0:
            continue
        pairs.extend((a, v / tot) for a, v in rows)

    print(f"\nQ5 [{sport}]  Does archetype predict share of team pay?")
    print(f"    player-seasons with an archetype and a pay share: {len(pairs)}\n")
    if len(pairs) < 200:
        print("    too few rows. Not reporting.")
        return

    by_a: dict[str, list[float]] = defaultdict(list)
    for a, s in pairs:
        by_a[a].append(s)
    means = {a: statistics.mean(v) for a, v in by_a.items() if len(v) >= 30}
    if len(means) < 2:
        print("    fewer than 2 archetypes with n>=30. Not ranking them.")
        return
    observed = max(means.values()) - min(means.values())

    rng = random.Random(SEED)
    labels = [a for a, _ in pairs]
    vals = [s for _, s in pairs]
    null = []
    for _ in range(SHUFFLES):
        rng.shuffle(labels)
        g: dict[str, list[float]] = defaultdict(list)
        for a, v in zip(labels, vals, strict=False):
            g[a].append(v)
        m = [statistics.mean(v) for v in g.values() if len(v) >= 30]
        if len(m) >= 2:
            null.append(max(m) - min(m))
    p95 = sorted(null)[int(0.95 * len(null))] if null else 0.0

    print(f"    {'archetype':12} {'n':>6}  mean share of team pay")
    for a, m in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"    {a:12} {len(by_a[a]):6}  {100 * m:5.2f}%")
    print()
    print(f"    observed spread : {100 * observed:.2f} percentage points")
    print(f"    shuffle p95     : {100 * p95:.2f}   ({SHUFFLES} permutations, seed {SEED})")
    if observed > p95:
        print("    VERDICT: clears the baseline. Archetype DOES carry pay information.")
        print("             Association only — role and quality are entangled, and this")
        print("             does not separate 'this role is paid more' from 'better")
        print("             players end up in this role'.")
    elif len(means) < 5:
        print(f"    VERDICT: UNDERPOWERED — only {len(means)} archetype groups clear n>=30.")
    else:
        print("    VERDICT: does not clear the shuffled baseline. No finding.")


def q_pay_within_usage_band() -> None:
    """Q6: does archetype still predict pay INSIDE a narrow band of usage?

    THE TEST THAT DECIDES WHETHER Q5 IS INTERESTING. Q5 found archetype predicts share of
    team pay at 11-14x the shuffled baseline, and it is real but unsurprising: role and
    quality are entangled, so "this role is paid more" and "better players end up in this
    role" produce the same table. Everyone already knows quarterbacks are expensive.

    Stratifying on minutes per game separates them. Inside a band of players who all play
    roughly the same amount, usage is held roughly constant — so if archetype STILL predicts
    pay there, the premium is attached to the ROLE rather than to how much the player plays.
    That is the non-obvious claim.

    hoops only: minutes come from acquire_hoops_rosters.py, which now keeps the MIN column
    LeagueDashPlayerStats was already returning. gridiron has no comparable usage field here.

    HONEST LIMIT, and it is not small. Minutes are a USAGE proxy, not a quality measure. A
    coach gives minutes partly because a player is good, so this holds usage constant, not
    talent. A surviving effect means "not explained by playing time", which is weaker than
    "not explained by quality" and stronger than Q5 alone.
    """
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    sal = _salary_index("hoops")
    if not sal:
        print("\nQ6 skipped: no hoops salary source.")
        return

    arch_of = {
        (norm_name(p["name"]), str(p["season"])): str(p["cross_arch"])
        for p in players
        if p["sport"] == "hoops" and p.get("cross_arch") is not None
    }

    team_rows: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for e in doc["edges"]:
        if e["sport"] != "hoops" or not e.get("org_id") or e.get("min") is None:
            continue
        k = (e["norm"], str(e["season"]))
        a, v = arch_of.get(k), sal.get(k)
        if a is not None and v:
            team_rows[e["org_id"]].append((a, float(v), float(e["min"])))

    rows: list[tuple[str, float, float]] = []
    for tr in team_rows.values():
        if len(tr) < 8:
            continue
        tot = sum(v for _, v, _ in tr)
        if tot <= 0:
            continue
        rows.extend((a, v / tot, mn) for a, v, mn in tr)

    print("\nQ6 [hoops]  Does archetype predict pay INSIDE a usage band?")
    print(f"    player-seasons with archetype, pay share and minutes: {len(rows)}\n")
    if len(rows) < 500:
        print("    too few rows. Not reporting.")
        return

    mins = sorted(r[2] for r in rows)
    cuts = [mins[int(len(mins) * f)] for f in (0.25, 0.5, 0.75)]
    bands = [
        ("bench      <%.0f mpg" % cuts[0], lambda m: m < cuts[0]),
        ("rotation %.0f-%.0f" % (cuts[0], cuts[1]), lambda m: cuts[0] <= m < cuts[1]),
        ("starter  %.0f-%.0f" % (cuts[1], cuts[2]), lambda m: cuts[1] <= m < cuts[2]),
        ("heavy      >=%.0f mpg" % cuts[2], lambda m: m >= cuts[2]),
    ]

    rng = random.Random(SEED)
    cleared = 0
    for label, pred in bands:
        band = [(a, s) for a, s, m in rows if pred(m)]
        by_a: dict[str, list[float]] = defaultdict(list)
        for a, s in band:
            by_a[a].append(s)
        means = {a: statistics.mean(v) for a, v in by_a.items() if len(v) >= 30}
        if len(means) < 2:
            print(f"    {label:22} n={len(band):5}  too few groups")
            continue
        observed = max(means.values()) - min(means.values())
        labels = [a for a, _ in band]
        vals = [s for _, s in band]
        null = []
        for _ in range(SHUFFLES):
            rng.shuffle(labels)
            g: dict[str, list[float]] = defaultdict(list)
            for a, v in zip(labels, vals, strict=False):
                g[a].append(v)
            m = [statistics.mean(v) for v in g.values() if len(v) >= 30]
            if len(m) >= 2:
                null.append(max(m) - min(m))
        p95 = sorted(null)[int(0.95 * len(null))] if null else 0.0
        ok = observed > p95
        cleared += ok
        print(
            f"    {label:22} n={len(band):5}  spread {100 * observed:5.2f}pp  "
            f"p95 {100 * p95:5.2f}  {'CLEARS' if ok else 'no'}"
        )

    print()
    if cleared == len(bands):
        print("    VERDICT: archetype predicts pay in EVERY usage band. The premium is not")
        print("             explained by playing time — a role effect, not just a minutes")
        print("             effect. Still not proof it is not QUALITY: minutes are a usage")
        print("             proxy, and coaches give minutes partly because a player is good.")
    elif cleared:
        print(f"    VERDICT: clears in {cleared} of {len(bands)} bands. Partial — the effect")
        print("             is not uniform across usage levels, which is itself informative.")
    else:
        print("    VERDICT: clears in NO band. Q5's effect is explained by playing time:")
        print("             archetypes differ in minutes, and minutes are what is paid for.")


def q_pay_within_quality_band() -> None:
    """Q7: does archetype still predict pay inside a band of equal PLAYER QUALITY?

    THE CLAIM Q6 COULD NOT MAKE. Q6 held USAGE constant (minutes) and the role premium
    survived, but minutes are not talent — a coach plays someone partly because they are
    good, so "not explained by playing time" was the most that could be earned.

    I ALSO SAID THE ESTATE HAD NO TALENT MEASURE. That was wrong and I checked rather than
    repeating it: vector-hoops/pipeline/cache/dashadvanced_<season>.json covers all 30
    corpus seasons and carries PIE — the NBA's own Player Impact Estimate, a composite
    single-number quality metric computed from box-score contribution and independent of
    salary. Stratifying on PIE holds QUALITY roughly constant, which is the thing Q6 could
    not do.

    If archetype still predicts pay share inside a PIE band, the premium is attached to the
    ROLE and not to how good the player is. That is the sellable claim: two players of equal
    measured impact are paid differently according to the archetype they occupy.

    LIMITS, and they are real. PIE is a box-score composite: it under-credits defence and
    off-ball work, and it is not a market-neutral measure of worth. It is a far better
    talent proxy than minutes, not a perfect one. Association only — this cannot separate
    "clubs overpay this role" from "this role produces value PIE does not capture".
    """
    hoops_cache = ROOT.parent / "vector-hoops" / "pipeline" / "cache"
    files = sorted(hoops_cache.glob("dashadvanced_*.json"))
    if not files:
        print("\nQ7 skipped: no dashadvanced_*.json in vector-hoops cache.")
        return

    pie: dict[tuple, float] = {}
    for f in files:
        season = f.stem.replace("dashadvanced_", "")
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in rows:
            v = r.get("PIE")
            if v is None:
                continue
            pie[(norm_name(str(r.get("PLAYER_NAME") or "")), season)] = float(v)

    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    sal = _salary_index("hoops")
    arch_of = {
        (norm_name(p["name"]), str(p["season"])): str(p["cross_arch"])
        for p in players
        if p["sport"] == "hoops" and p.get("cross_arch") is not None
    }

    team_rows: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for e in doc["edges"]:
        if e["sport"] != "hoops" or not e.get("org_id"):
            continue
        k = (e["norm"], str(e["season"]))
        a, v, q = arch_of.get(k), sal.get(k), pie.get(k)
        if a is not None and v and q is not None:
            team_rows[e["org_id"]].append((a, float(v), q))

    rows: list[tuple[str, float, float]] = []
    for tr in team_rows.values():
        if len(tr) < 8:
            continue
        tot = sum(v for _, v, _ in tr)
        if tot <= 0:
            continue
        rows.extend((a, v / tot, q) for a, v, q in tr)

    print("\nQ7 [hoops]  Does archetype predict pay INSIDE a PIE (quality) band?")
    print(f"    player-seasons with archetype, pay share and PIE: {len(rows)}")
    print(f"    PIE coverage: {len(pie)} player-seasons across {len(files)} season files\n")
    if len(rows) < 500:
        print("    too few rows. Not reporting.")
        return

    qs = sorted(r[2] for r in rows)
    cuts = [qs[int(len(qs) * f)] for f in (0.25, 0.5, 0.75)]
    bands = [
        (f"low PIE    <{cuts[0]:.3f}", lambda q: q < cuts[0]),
        (f"mid-low  {cuts[0]:.3f}-{cuts[1]:.3f}", lambda q: cuts[0] <= q < cuts[1]),
        (f"mid-high {cuts[1]:.3f}-{cuts[2]:.3f}", lambda q: cuts[1] <= q < cuts[2]),
        (f"high PIE   >={cuts[2]:.3f}", lambda q: q >= cuts[2]),
    ]

    rng = random.Random(SEED)
    cleared = 0
    rank_by_band: dict[str, list[str]] = {}
    mean_by_band: dict[str, dict[str, float]] = {}
    for label, pred in bands:
        band = [(a, s) for a, s, q in rows if pred(q)]
        by_a: dict[str, list[float]] = defaultdict(list)
        for a, s in band:
            by_a[a].append(s)
        means = {a: statistics.mean(v) for a, v in by_a.items() if len(v) >= 30}
        if len(means) < 2:
            print(f"    {label:26} n={len(band):5}  too few groups")
            continue
        rank_by_band[label] = [a for a, _ in sorted(means.items(), key=lambda kv: -kv[1])]
        mean_by_band[label] = means
        observed = max(means.values()) - min(means.values())
        labels = [a for a, _ in band]
        vals = [s for _, s in band]
        null = []
        for _ in range(SHUFFLES):
            rng.shuffle(labels)
            g: dict[str, list[float]] = defaultdict(list)
            for a, v in zip(labels, vals, strict=False):
                g[a].append(v)
            m = [statistics.mean(v) for v in g.values() if len(v) >= 30]
            if len(m) >= 2:
                null.append(max(m) - min(m))
        p95 = sorted(null)[int(0.95 * len(null))] if null else 0.0
        ok = observed > p95
        cleared += ok
        print(
            f"    {label:26} n={len(band):5}  spread {100 * observed:5.2f}pp  "
            f"p95 {100 * p95:5.2f}  {'CLEARS' if ok else 'no'}"
        )

    # WHICH roles carry the premium, and whether the ordering survives quality. A role that
    # is top-paid at EVERY quality level is a genuine premium; one that only leads in the
    # high-PIE band is just "stars are expensive" restated.
    if len(rank_by_band) >= 2:
        common = set.intersection(*(set(v) for v in rank_by_band.values()))
        labels = arch_labels()
        print()
        print("    ROLE PREMIUM by quality band (mean share of team pay):")
        hdr = "    " + f"{'archetype':10}" + "".join(f"{lbl.split()[0]:>12}" for lbl in rank_by_band)
        print(hdr)
        overall = {
            a: statistics.mean([mean_by_band[b][a] for b in rank_by_band if a in mean_by_band[b]]) for a in common
        }
        for a in sorted(overall, key=lambda x: -overall[x]):
            row = f"    {a:10}"
            for b in rank_by_band:
                row += f"{100 * mean_by_band[b].get(a, float('nan')):11.2f}%"
            print(row + f"   {labels.get(a, '')}")

        tops = {v[0] for v in rank_by_band.values()}
        bots = {v[-1] for v in rank_by_band.values()}
        print()
        if len(tops) == 1:
            _t = tops.pop()
            print(f"    {_t} ({arch_labels().get(_t,'?')}) is the highest-paid archetype in")
            print("    EVERY quality band —")
            print("    a role premium that holds at every level of measured impact.")
        else:
            _lab = arch_labels()
            print(
                "    top-paid role is NOT stable across bands: "
                + ", ".join(f"{t} ({_lab.get(t,'?')})" for t in sorted(tops))
            )
            print("    The premium ordering is quality-dependent, not a fixed role effect:")
            print("    which role commands the most pay DEPENDS on the quality band.")
        if len(bots) == 1:
            _b = bots.pop()
            print(f"    {_b} ({arch_labels().get(_b,'?')}) is the lowest-paid in every band.")

    print()
    if cleared == len(bands):
        print("    VERDICT: archetype predicts pay in EVERY quality band. Two players of")
        print("             equal measured impact are paid differently by role. This is the")
        print("             claim Q6 could not make — not explained by talent, as PIE")
        print("             measures it.")
        print("             PIE under-credits defence and off-ball work, so 'clubs overpay")
        print("             this role' and 'this role creates value PIE misses' remain")
        print("             indistinguishable here.")
    elif cleared:
        print(f"    VERDICT: clears in {cleared} of {len(bands)} bands. Partial, and where it")
        print("             fails is the informative part.")
    else:
        print("    VERDICT: clears in NO band. Q5/Q6's effect is explained by player")
        print("             quality: archetypes differ in PIE, and PIE is what is paid for.")


def positive_control() -> bool:
    """WIN_PCT vs NET_RATING must come back significant, or every null here is worthless.

    Q1 and Q2 both returned "no finding". Two nulls in a row from a tool nobody has seen
    fire is indistinguishable from a tool that CANNOT fire, and reporting them as results
    would be the vacuous-pass shape this estate keeps finding elsewhere. So the same
    machinery — same n, same shuffle, same seed — is pointed at a relationship that must
    exist, and the nulls only mean something if this clears.
    """
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    pts = [
        (o["features"]["WIN_PCT"], o["features"]["NET_RATING"])
        for o in doc["orgs"]
        if o["sport"] == "hoops"
        and (o.get("features") or {}).get("WIN_PCT") is not None
        and (o.get("features") or {}).get("NET_RATING") is not None
    ]
    if len(pts) < 50:
        print("CONTROL SKIPPED: too few hoops orgs with both fields.")
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    r = statistics.correlation(xs, ys)
    rng = random.Random(SEED)
    shuffled = list(ys)
    null = [abs(statistics.correlation(xs, shuffled)) for _ in range(SHUFFLES) if not rng.shuffle(shuffled)]
    p95 = sorted(null)[int(0.95 * len(null))]
    ok = abs(r) > p95
    print(
        f"CONTROL  WIN_PCT vs NET_RATING  n={len(pts)}  r={r:+.3f}  "
        f"shuffle p95 |r|={p95:.3f}  ->  {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        print("  The machinery cannot detect a relationship that must exist.")
        print("  Treat every 'no finding' below as unproven, not as a negative result.")
    print()
    return ok


def q_company_archetype() -> None:
    """Q8: within sport, does an org's archetype mix differ by whether a COMPANY named
    its venue?

    Pre-registered before running, because the tempting version of this analysis is wrong
    in a way that produces a significant result almost automatically.

    THE UNIT IS THE ORG, NOT THE ATHLETE. The company edge is a property of the club:
    every player on the Crypto.com Arena roster shares the same naming-rights company.
    5,248 athletes reach a company, but that reach varies across only ~142 orgs, so
    athlete-level rows are ~37x replicated within cluster. Testing at athlete level would
    treat 5,248 correlated rows as independent and manufacture significance out of
    roster size. This aggregates to one archetype distribution per org first.

    STRATIFIED BY SPORT, for the reason Q1 had to learn: US venues carry naming rights
    (AT&T, Bank of America) where European clubs carry shirt sponsors instead, and
    archetype mixes also differ by sport. An unstratified test would recover "gridiron vs
    pitch" and call it a company effect.

    THE CONTRAST IS BINARY ON PURPOSE. Industry would be the interesting split, but P452
    is fragmented into 83 values across 103 companies and 93 companies have none at all
    — that is a power problem dressed as a richness problem. Naming rights is the single
    cleanest "this club is sponsored" bit available, and it is the one a sponsorship
    product would actually key on.

    Statistic: L1 distance between the archetype distribution of named-venue orgs and
    non-named orgs, within sport, averaged over sports with enough orgs on both sides.
    Null: shuffle the has-company label among orgs WITHIN sport, same statistic.
    """
    comp_path = ROOT / "data" / "orgs" / "company_entities.json"
    if not comp_path.exists():
        print("\nQ8  no company_entities.json — run build_company_entities.py first.")
        return
    comp = json.loads(comp_path.read_text(encoding="utf-8"))
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    orgs = {o["org_id"]: o for o in doc["orgs"]}

    named_keys = {e["org_key"] for e in comp["edges"] if e["rel"] == "named_after" and e.get("org_key")}
    probed_keys = {e["org_key"] for e in comp["edges"] if e.get("org_key")}

    arch_of = {
        (norm_name(p["name"]), p["sport"], str(p["season"])): str(p["cross_arch"])
        for p in players
        if p.get("cross_arch") is not None
    }

    # org key -> archetype counts, aggregated across seasons
    per_org: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in doc["edges"]:
        oid = e.get("org_id")
        if not oid or oid not in orgs:
            continue
        o = orgs[oid]
        key = f"{o.get('sport')}::{o.get('team')}"
        if key not in probed_keys:
            continue  # only orgs we actually probed can be scored either way
        a = arch_of.get((e["norm"], e["sport"], str(e["season"])))
        if a:
            per_org[(o["sport"], key)][a] += 1

    print("\nQ8  Within sport, does archetype mix differ where a COMPANY named the venue?")
    print(f"    orgs probed: {len(probed_keys)}   with archetyped players: {len(per_org)}")
    print("    unit = ORG (the company edge is a club property; athlete rows are clustered)\n")

    MIN_ORGS = 8
    by_sport: dict[str, list[tuple[bool, dict[str, int]]]] = defaultdict(list)
    for (sport, key), counts in per_org.items():
        if sum(counts.values()) >= 5:  # an org with a real roster, not a stub
            by_sport[sport].append((key in named_keys, counts))

    def l1(group: list[tuple[bool, dict[str, int]]]) -> float | None:
        yes = [c for f, c in group if f]
        no = [c for f, c in group if not f]
        if len(yes) < MIN_ORGS or len(no) < MIN_ORGS:
            return None

        def dist(cs):
            tot: dict[str, int] = defaultdict(int)
            for c in cs:
                for a, n in c.items():
                    tot[a] += n
            s = sum(tot.values())
            return {a: n / s for a, n in tot.items()}

        dy, dn = dist(yes), dist(no)
        keys = set(dy) | set(dn)
        return sum(abs(dy.get(k, 0.0) - dn.get(k, 0.0)) for k in keys)

    rng = random.Random(SEED)
    reported = 0
    for sport in sorted(by_sport):
        group = by_sport[sport]
        yes = sum(1 for f, _ in group if f)
        obs = l1(group)
        if obs is None:
            print(
                f"    {sport:9} orgs {len(group):3} (named {yes:3}, other {len(group)-yes:3})"
                f"  UNDERPOWERED — need >={MIN_ORGS} on both sides. Not a null."
            )
            continue
        flags = [f for f, _ in group]
        counts = [c for _, c in group]
        null = []
        for _ in range(SHUFFLES):
            rng.shuffle(flags)
            null.append(l1(list(zip(flags, counts, strict=True))) or 0.0)
        null.sort()
        p95 = null[int(0.95 * len(null))]
        verdict = "DIFFERS" if obs > p95 else "no finding"
        print(
            f"    {sport:9} orgs {len(group):3} (named {yes:3}, other {len(group)-yes:3})"
            f"  L1 {obs:.3f}  shuffle p95 {p95:.3f}  -> {verdict}"
        )
        reported += 1

    # THE STRUCTURAL FACT IS THE OUTPUT HERE, not the test. Printing per-sport rates
    # explicitly because "UNDERPOWERED" reads as a shrug, and the reason for it is a
    # finding a sponsorship product would key on directly.
    print("\n    named-venue rate by sport (why two sports could not be tested):")
    for sport in sorted(by_sport):
        g = by_sport[sport]
        yes = sum(1 for f, _ in g if f)
        print(f"      {sport:9} {yes:3}/{len(g):3} = {100.0 * yes / len(g):5.1f}%")
    print("      Corporate venue naming is effectively TABLE STAKES in US pro sport and")
    print("      rare in European football. A binary has-a-sponsor contrast has almost no")
    print("      variance to work with in gridiron/hoops — the useful split there is WHICH")
    print("      company, not WHETHER, and P452 is too sparse (52.6%) to carry it yet.")

    if reported == 0:
        print("\n    Nothing was powered enough to test. That is a data statement, not a")
        print("    null: the company edge varies across too few orgs per sport once the")
        print("    athlete-level replication is removed.")
    else:
        print(f"\n    ({SHUFFLES} permutations, seed {SEED}; labels shuffled WITHIN sport)")


def q_company_sector() -> None:
    """Q9: within sport, does archetype mix differ by the SECTOR of the attached company?

    Q8 could not test this: its only split was "has a naming-rights deal", which is 96%
    true in US sport. data/sector_map.json now assigns 127/128 businesses to 28 sectors,
    so the split exists. Whether it is TESTABLE is a separate question, and this answers
    it before reporting any statistic.

    THE CEILING IS ORGS, NOT THE TAXONOMY, and it does not move with more acquisition.
    Sector is a club property, so the unit is the org — the same clustering argument as
    Q8. Enriched org counts are hoops 31 and gridiron 32, because there are 30 NBA teams
    and 32 NFL franchises. Splitting ~31 orgs across 28 sectors leaves ~1 org per cell.
    Using org = team x season would give 892 hoops orgs but NOT 892 independent ones:
    sector is not time-sliced (see company_entities.json's time_caveat), so every season
    of a team carries the same sector and the effective n stays ~30.

    So a within-sport sector test is permanently underpowered in exactly the two sports
    where the company edge is strongest. That is a fact about league structure, not about
    this dataset, and it bounds what a sponsorship product may claim from it.

    pitch is the one sport with enough sectors to test (13 with >=3 orgs) and it has the
    WEAKEST company coverage (24.7% of athletes). That caveat travels with its result.
    """
    apath = ROOT / "data" / "orgs" / "company_sectors_applied.json"
    cpath = ROOT / "data" / "orgs" / "company_entities.json"
    if not (apath.exists() and cpath.exists()):
        print("\nQ9  no sector layer — run apply_sector_map.py first.")
        return
    applied = json.loads(apath.read_text(encoding="utf-8"))
    comp = json.loads(cpath.read_text(encoding="utf-8"))
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    orgs = {o["org_id"]: o for o in doc["orgs"]}

    co_by_label = {c["label"]: c for c in comp["companies"]}
    # org_key -> sectors. sports_holdings excluded: a club's own holding entity is not
    # a sponsor, and leaving it in would let a club predict its own archetype mix.
    sec_of_key: dict[str, set[str]] = defaultdict(set)
    for sec, labels in applied["sector_companies"].items():
        if sec == "sports_holdings":
            continue
        for lab in labels:
            for k in co_by_label.get(lab, {}).get("orgs", []):
                sec_of_key[k].add(sec)

    arch_of = {
        (norm_name(p["name"]), p["sport"], str(p["season"])): str(p["cross_arch"])
        for p in players
        if p.get("cross_arch") is not None
    }
    per_org: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in doc["edges"]:
        oid = e.get("org_id")
        o = orgs.get(oid) if oid else None
        if not o:
            continue
        key = f"{o.get('sport')}::{o.get('team')}"
        if key not in sec_of_key:
            continue
        a = arch_of.get((e["norm"], e["sport"], str(e["season"])))
        if a:
            per_org[(o["sport"], key)][a] += 1

    print("\nQ9  Within sport, does archetype mix differ by the SECTOR of the company?")
    print(f"    orgs with >=1 non-holding sector: {len(sec_of_key)}" f"   with archetyped players: {len(per_org)}")
    print("    unit = ORG. sports_holdings excluded (a club owning itself is not a sponsor).\n")

    MIN_ORGS_PER_SECTOR = 3
    MIN_SECTORS = 3
    rng = random.Random(SEED)
    for sport in ("hoops", "gridiron", "pitch"):
        keys = [(sp, k) for (sp, k) in per_org if sp == sport]
        bysec: dict[str, list[dict[str, int]]] = defaultdict(list)
        for sp, k in keys:
            for sec in sec_of_key[k]:
                bysec[sec].append(per_org[(sp, k)])
        usable = {s: v for s, v in bysec.items() if len(v) >= MIN_ORGS_PER_SECTOR}
        if len(usable) < MIN_SECTORS:
            print(
                f"    {sport:9} orgs {len(keys):3}   sectors with >={MIN_ORGS_PER_SECTOR} "
                f"orgs: {len(usable)}  UNDERPOWERED — need >={MIN_SECTORS}. Not a null."
            )
            continue

        def dist(cs):
            tot: dict[str, int] = defaultdict(int)
            for c in cs:
                for a, n in c.items():
                    tot[a] += n
            s = sum(tot.values())
            return {a: n / s for a, n in tot.items()} if s else {}

        def mean_l1(groups: dict[str, list[dict[str, int]]]) -> float:
            glob = dist([c for v in groups.values() for c in v])
            ds = []
            for v in groups.values():
                d = dist(v)
                ds.append(sum(abs(d.get(a, 0.0) - glob.get(a, 0.0)) for a in set(d) | set(glob)))
            return sum(ds) / len(ds) if ds else 0.0

        obs = mean_l1(usable)
        # Null: reassign each org's COUNTS to a shuffled sector membership, preserving
        # the number of orgs per sector so a big sector cannot look distinctive purely
        # by being big.
        flat = [(s, c) for s, v in usable.items() for c in v]
        counts_only = [c for _, c in flat]
        labels = [s for s, _ in flat]
        null = []
        for _ in range(SHUFFLES):
            rng.shuffle(labels)
            g: dict[str, list[dict[str, int]]] = defaultdict(list)
            for s, c in zip(labels, counts_only, strict=True):
                g[s].append(c)
            null.append(mean_l1(g))
        null.sort()
        p95 = null[int(0.95 * len(null))]
        verdict = "DIFFERS" if obs > p95 else "no finding"
        print(
            f"    {sport:9} orgs {len(keys):3}   sectors used {len(usable):2}"
            f"   mean L1 {obs:.3f}  shuffle p95 {p95:.3f}  -> {verdict}"
        )

    print(f"\n    ({SHUFFLES} permutations, seed {SEED}; sector labels shuffled WITHIN sport)")
    print("\n    THE CEILING IS LEAGUE STRUCTURE, NOT DATA. 30 NBA teams and 32 NFL")
    print("    franchises means ~31 enriched orgs per US sport; 28 sectors over 31 orgs")
    print("    is ~1 org per cell. team x season would give 892 hoops orgs but not 892")
    print("    INDEPENDENT ones — sector is not time-sliced, so every season of a team")
    print("    carries the same sector and effective n stays ~30. More company")
    print("    acquisition does not fix this; only pooling sports or a coarser sector")
    print("    grouping would, and both change the question being asked.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--q",
        default="all",
        choices=[
            "all",
            "archetype",
            "roster",
            "salary",
            "league",
            "pay",
            "band",
            "quality",
            "company",
            "sector",
        ],
    )
    args = ap.parse_args()

    positive_control()

    if args.q in ("all", "archetype"):
        rows, dropped = load_joined()
        if not rows:
            print("No joined rows. Run build_org_entities.py + enrich_orgs_wikidata.py first.")
            return 1
        q_archetype(rows, dropped)
    if args.q in ("all", "roster"):
        q_roster_mix()
    if args.q in ("all", "salary"):
        # One null is a result; the SAME null in a harder-capped league is a replication
        # that could have failed. pitch is absent by design — 0.58% salary coverage.
        for _sport in ("hoops", "gridiron"):
            q_salary_concentration(_sport)
    if args.q in ("all", "league"):
        q_league_style()
    if args.q in ("all", "pay"):
        for _sport in ("hoops", "gridiron"):
            q_archetype_pay(_sport)
    if args.q in ("all", "band"):
        q_pay_within_usage_band()
    if args.q in ("all", "quality"):
        q_pay_within_quality_band()
    if args.q in ("all", "company"):
        q_company_archetype()
    if args.q in ("all", "sector"):
        q_company_sector()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
