#!/usr/bin/env python3
"""What the athlete-level market layer actually covers, and who is in it (7.12).

Solo personal project, no connection to employer, built with public/free-tier only

7.10 showed the company layer has no athlete-level variance at all, and 7.11 showed its
team-level signal only has anything to explain in one sport. The remaining candidate for an
athlete-level marketability signal is the market layer: Forbes earnings, Wikidata honors,
social handles, and the derived AWARD_PRESTIGE. Before anything is modelled on it, three
questions have to be answered in order, because each one can make the next moot:

  1. COVERAGE. How many corpus athletes does each signal actually reach, per sport?
  2. CONTAMINATION. The honors and handles files are RAW Wikidata pulls, not matched
     products — social_handles.json carries "ARCO Arena" and "Roseto Sharks", and
     honors_wikidata.json carries R. Kelly. Row counts are not athlete counts. Most
     unmatched rows are not junk though — they are out-of-corpus athletes (WNBA,
     European leagues), so the gap is scope, not dirt.
  3. SELECTION. Is having an award-prestige record essentially the same fact as being
     good? If P(covered) rises steeply with performance, the layer is a star-only sample
     and any "marketability vs production residual" computed on it describes stars, not
     athletes. That is the survivorship problem for the fifth time this phase, and it is
     the one that would silently invalidate the product.

PRE-REGISTERED READING OF (3), fixed before the first run. Bucket hoops athletes by career
VOR decile and compute coverage per decile:

    STAR-ONLY SAMPLE if top-decile coverage is >= 5x bottom-decile coverage. Then the
        layer must not be described as measuring athletes, only as measuring stars, and no
        residual computed on it generalises past that pool.
    USABLE if the ratio is < 2x — coverage is broad enough that a residual means something.
    Between 2x and 5x is reported as PARTIAL and neither claim is made.

Hoops is the test bed because it carries by far the most coverage; gridiron and pitch are
too thin for a ten-way split. Exact per-sport counts are NOT repeated here — they move
whenever the pull is re-run, and a stale number in a docstring next to a live one in the
output is a defect this file has already had once (see `_gridiron_note`).

    python pipeline/probe_market_layer_coverage.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKET = ROOT / "data" / "market"
NAME_INDEX = MARKET / "name_index.json"
OUT = ROOT / "data" / "market" / "market_layer_coverage.json"

RAW_PULLS = {
    "honors": MARKET / "honors_wikidata.json",
    "social_handles": MARKET / "social_handles.json",
}
MATCHED = {
    "award_prestige": MARKET / "award_prestige.json",
    "forbes": MARKET / "Forbes_highest_paid.json",
}

DECILES = 10
STAR_RATIO = 5.0
USABLE_RATIO = 2.0


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def hoops_career_vor() -> dict[str, float]:
    """Career floored VOR per hoops athlete, from the one implementation (7.8a)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent / "build_hoops_vor_draft_value.py"
    spec = importlib.util.spec_from_file_location("_mkt_vor", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    vec = json.loads(mod.VECTORS.read_text(encoding="utf-8"))
    seasons = sorted({str(p["season"]) for p in vec["players"]}, key=mod.season_start)
    series, _ = mod.vor_series(seasons, mod.eligible_pairs(vec))
    return {n: sum(v for _y, v in rows) for n, rows in series.items()}


def _gridiron_note(signals: dict, corpus: dict) -> str:
    """Derived from the numbers, not hard-coded.

    The previous version of this string was written when gridiron coverage was zero and
    it kept asserting that after the cause was fixed and the artifacts were rebuilt — a
    stale narrative sitting next to live numbers that contradicted it. Anything that
    states a measurement in prose has to be computed from that measurement.
    """
    n = len(corpus.get("gridiron", ()))
    got = {
        k: (v.get("per_sport", {}).get("gridiron", {}) or {}).get("matched", 0)
        for k, v in signals.items()
        if "per_sport" in v
    }
    if not any(got.values()):
        return (
            f"GRIDIRON HAS NO ATHLETE-LEVEL MARKET DATA — every signal matched 0 of "
            f"{n} athletes. Check the sport QID in pull_honors_wikidata.py: a QID "
            f"that is not a sport returns an empty result set rather than an error."
        )
    parts = ", ".join(f"{k} {v} ({100.0 * v / max(n, 1):.1f}%)" for k, v in sorted(got.items()))
    return (
        f"gridiron coverage of {n} athletes: {parts}. It was zero on every signal "
        f"until SPORT_Q['gridiron'] was corrected from Q9398 (Grugliasco, an Italian "
        f"comune) to Q41323 (American football). Honors stay thin at the low single "
        f"digits because NFL players genuinely carry fewer Wikidata P166 award "
        f"statements than NBA players — that part is a property of the source, not a "
        f"bug, and it is not fixable by re-querying."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not NAME_INDEX.exists():
        print(f"missing {NAME_INDEX} — run pipeline/resolve_names.py")
        return 2
    ni = json.loads(NAME_INDEX.read_text(encoding="utf-8"))
    corpus: dict[str, set[str]] = collections.defaultdict(set)
    for p in ni["players"]:
        corpus[p["sport"]].add(norm_name(p["name"]))
    sports = sorted(corpus)

    # ---- 1/2. coverage and contamination -------------------------------------
    signals: dict[str, dict] = {}
    for key, path in RAW_PULLS.items():
        if not path.exists():
            signals[key] = {"missing": str(path)}
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rows, hits = collections.Counter(), collections.Counter()
        junk_examples: list[str] = []
        for v in d.values():
            sp = v.get("sport_unified") or "?"
            rows[sp] += 1
            if norm_name(v.get("name", "")) in corpus.get(sp, ()):
                hits[sp] += 1
            elif len(junk_examples) < 5:
                junk_examples.append(v.get("name", ""))
        signals[key] = {
            "kind": "RAW Wikidata pull — row count is NOT an athlete count",
            "unmatched_composition": (
                "Unmatched rows are a MIX and must not all be called contamination: "
                "out-of-corpus ATHLETES (Anne Donovan, Cynthia Cooper-Dyke, Nancy "
                "Lieberman are WNBA; Maik Zirbes and Alexander Kuhl are European league) "
                "alongside genuine non-person entities (ARCO Arena, Roseto Sharks). The "
                "pull was scoped by sport QID, not by corpus membership."
            ),
            "rows_total": len(d),
            "matched_corpus_total": sum(hits.values()),
            "per_sport": {
                sp: {
                    "rows": rows.get(sp, 0),
                    "matched": hits.get(sp, 0),
                    "pct_of_rows": round(100.0 * hits.get(sp, 0) / max(rows.get(sp, 0), 1), 1),
                    "pct_of_corpus": round(100.0 * hits.get(sp, 0) / len(corpus[sp]), 1),
                }
                for sp in sports
            },
            "unmatched_examples": junk_examples,
        }

    ap_path = MATCHED["award_prestige"]
    prestige: dict[tuple[str, str], float] = {}
    if ap_path.exists():
        d = json.loads(ap_path.read_text(encoding="utf-8"))
        per = collections.Counter()
        for v in d.values():
            per[v.get("sport", "?")] += 1
            # KEYED BY (sport, name). A bare name key let gridiron's Chris Johnson mark
            # hoops' Chris Johnson as covered — one collision here, but it is the same
            # unscoped cross-sport join that produced 17/17 false matches earlier in this
            # phase (NFL Matt Ryan -> NBA Matt Ryan). Scope it whether or not it bites.
            prestige[(v.get("sport", "?"), norm_name(v.get("name", "")))] = float(v.get("AWARD_PRESTIGE") or 0.0)
        vals = [float(v.get("AWARD_PRESTIGE") or 0.0) for v in d.values()]
        nz_per = collections.Counter(
            v.get("sport", "?") for v in d.values() if float(v.get("AWARD_PRESTIGE") or 0.0) > 0
        )
        ceiling = [v for v in d.values() if float(v.get("AWARD_PRESTIGE") or 0.0) >= 0.999]
        ce_awards = sorted(int(v.get("n_awards") or 0) for v in ceiling)
        signals["award_prestige"] = {
            "kind": "MATCHED product (carries native_player_id)",
            "rows_total": len(d),
            "per_sport": {
                sp: {
                    "matched": per.get(sp, 0),
                    "pct_of_corpus": round(100.0 * per.get(sp, 0) / len(corpus[sp]), 1),
                    # A matched row is not a signal-carrying row.
                    "nonzero": nz_per.get(sp, 0),
                    "pct_of_corpus_nonzero": round(100.0 * nz_per.get(sp, 0) / len(corpus[sp]), 1),
                }
                for sp in sports
            },
            "zero_valued_rows": {
                "n": sum(1 for v in vals if v == 0.0),
                "pct": round(100.0 * sum(1 for v in vals if v == 0.0) / max(len(vals), 1), 1),
                "note": (
                    "The median AWARD_PRESTIGE across matched players is "
                    f"{statistics.median(vals) if vals else 0}. A row exists because "
                    "the athlete has SOME Wikidata award, but the tiering scores "
                    "untiered awards at nothing, so 'matched' overstates usable "
                    "coverage roughly two-fold."
                ),
            },
            "ceiling": {
                "n_at_1.000": len(ceiling),
                "pct": round(100.0 * len(ceiling) / max(len(vals), 1), 1),
                "n_awards_span": [ce_awards[0], ce_awards[-1]] if ce_awards else [],
                "note": (
                    "AWARD_PRESTIGE saturates. The players at 1.000 span "
                    f"{ce_awards[0] if ce_awards else 0} to "
                    f"{ce_awards[-1] if ce_awards else 0} raw awards — Jokic ties "
                    "Messi — so the measure cannot rank the top of its own range, "
                    "which is exactly the range a marketability product cares about."
                ),
            },
        }

    fb_path = MATCHED["forbes"]
    if fb_path.exists():
        fb = json.loads(fb_path.read_text(encoding="utf-8"))
        names = {norm_name(r["name"]) for r in fb["rows"] if not r.get("out_of_corpus")}
        in_corpus = {sp: len(names & corpus[sp]) for sp in sports}
        signals["forbes"] = {
            "kind": "TOP-N LIST — absence is not zero earnings, it is absence from a "
            "10-per-year list. Cannot be used as a continuous outcome.",
            "rows_total": fb.get("n_rows"),
            "distinct_names_in_corpus_sports": len(names),
            "per_sport": {
                sp: {
                    "matched": in_corpus[sp],
                    "pct_of_corpus": round(100.0 * in_corpus[sp] / len(corpus[sp]), 2),
                }
                for sp in sports
            },
        }

    # ---- 3. selection test, hoops --------------------------------------------
    vor = hoops_career_vor()
    pool = sorted((n for n in corpus["hoops"] if n in vor), key=lambda n: vor[n])
    selection: dict = {"n_hoops_with_vor": len(pool)}
    if len(pool) >= DECILES * 20:
        size = len(pool) / DECILES
        buckets = []
        for i in range(DECILES):
            chunk = pool[int(i * size) : int((i + 1) * size)]
            cov = sum(1 for n in chunk if ("hoops", n) in prestige)
            buckets.append(
                {
                    "decile": i + 1,
                    "n": len(chunk),
                    "covered": cov,
                    "pct": round(100.0 * cov / len(chunk), 1),
                    "median_vor": round(statistics.median(vor[n] for n in chunk), 2),
                }
            )
        top, bot = buckets[-1]["pct"], buckets[0]["pct"]
        ratio = (top / bot) if bot > 0 else float("inf")
        selection.update(
            {
                "deciles": buckets,
                "top_decile_pct": top,
                "bottom_decile_pct": bot,
                "ratio": (round(ratio, 1) if ratio != float("inf") else "inf (bottom decile 0%)"),
                "verdict": (
                    "STAR-ONLY SAMPLE" if ratio >= STAR_RATIO else "USABLE" if ratio < USABLE_RATIO else "PARTIAL"
                ),
            }
        )

    report = {
        "question": "What does the athlete-level market layer cover, and who is in it?",
        "corpus": {sp: len(corpus[sp]) for sp in sports},
        "signals": signals,
        "selection_test_hoops": selection,
        "per_sport_summary": {
            sp: {sig: signals[sig]["per_sport"][sp].get("matched", 0) for sig in signals if "per_sport" in signals[sig]}
            for sp in sports
        },
        "gridiron_note": _gridiron_note(signals, corpus),
        "forbes_caveat": (
            "Forbes is a top-N list: ~10 athletes per year over 16 years. Absence from it "
            "is absence from a leaderboard, not zero endorsement income. Using it as a "
            "continuous outcome would repeat the survivor-pool error from 7.7f exactly."
        ),
        "decision_rule": (
            f"STAR-ONLY if top-decile coverage >= {STAR_RATIO}x bottom-decile; USABLE if "
            f"< {USABLE_RATIO}x; PARTIAL between. Fixed before the first run."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("corpus: " + "  ".join(f"{sp} {len(corpus[sp])}" for sp in sports) + "\n")
    for key, s in signals.items():
        if "missing" in s:
            print(f"{key:<16} MISSING {s['missing']}")
            continue
        print(f"{key:<16} {s['kind']}")
        print(f"{'':16} rows {s['rows_total']}")
        for sp in sports:
            v = s["per_sport"].get(sp, {})
            print(
                f"{'':18}{sp:<9} matched {v.get('matched', 0):>5}  "
                f"= {v.get('pct_of_corpus', 0):>5.1f}% of corpus"
                + (f"   ({v['pct_of_rows']}% of its rows)" if "pct_of_rows" in v else "")
            )
        if s.get("unmatched_examples"):
            print(f"{'':18}unmatched e.g. {', '.join(s['unmatched_examples'][:4])}")
        print()

    if "deciles" in selection:
        print("selection test (hoops, career VOR decile -> award_prestige coverage):")
        for b in selection["deciles"]:
            bar = "#" * int(b["pct"] / 2)
            print(f"  d{b['decile']:<2} n={b['n']:<4} med VOR {b['median_vor']:>8.2f}  " f"{b['pct']:>5.1f}%  {bar}")
        print(
            f"\n  top {selection['top_decile_pct']}%  bottom {selection['bottom_decile_pct']}%"
            f"   ratio {selection['ratio']}   -> {selection['verdict']}"
        )
    print(f"\n{report['gridiron_note']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
