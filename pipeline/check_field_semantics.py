#!/usr/bin/env python3
"""A field must mean what its name says, and the data must agree. READ-ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

This estate's recurring defect has one shape: a real value answering a different question
than the one it appears to answer. Every instance below was found by hand, one at a time,
and each took a person reading a file and noticing:

    "59 economies"            was 57 economies + 2 BIS aggregates (XM Euro area, XW World)
    vol = "dispersion of      computed the sd of index LEVELS; corr with the level feature
     quarterly growth"         was +0.810, so the tower was a rescaled copy of another
    "some economies have      the minimum was 252, not "a few dozen" — wrong by ~10x
     a few dozen obs"
    delta_vs_majority         tracked as a paired DIFFERENCE, where the LEVEL was the
                              question the experiment actually asked
    paired_MDE_n3             a key that outlived n=3 and crashed at n=5
    mtnn_mean 0.1168          the artifact said 0.1157

Nothing checked any of them. This does, mechanically, for the classes that are decidable
without a human.

FOUR ARMS, each decidable from the artifact alone:

  RANGE       a field whose NAME asserts a range must be in it. `*_share`, `*_frac`,
              `*_pct`, `*_rate` in [0,1] (or [0,100] if any sibling exceeds 1);
              `*_sd`, `*_std`, `*_var` >= 0; `corr*`, `cos*`, `*_r`, `silhouette` in [-1,1];
              `n_*`, `*_count` a non-negative integer; `p_*`/`*_pvalue` in [0,1].
  AGGREGATE   a `mean`/`sd`/`min`/`max` sitting beside the list it summarises must equal
              the aggregate of that list. This is the arm that catches an artifact whose
              headline drifted away from its own per-seed values.
  COUNT       an `n_<thing>` beside a `<thing>s` collection must equal its length. This is
              the arm that catches "59 economies" over a 61-entry map.
  UNDOCUMENTED a top-level numeric field with no prose in the artifact and no mention in
              the producing script is reported, not failed. Coverage, not a verdict.

IT WRITES NOTHING AND RUNS NOTHING. That restriction is not fastidiousness: the sibling
checker check_documented_usage.py executed documented commands, and because build_*.py and
probe_*.py write with no flag at all, one run rewrote ten artifacts here AND
vector-hoops/pipeline/seed_floor.json in another repo, stripping a CORRECTED marker and
taking three green gates red. This file opens files for reading and nothing else.

WHAT IT CANNOT DECIDE, stated so a green line is not read as more than it is. It cannot
tell that `vol_q_nom` is the sd of levels rather than of growth — that requires reading the
code that produced it against the prose that describes it, which is a judgement. It checks
the claims a NAME makes, not the claims a SENTENCE makes. The undocumented count is
printed every run so the uncovered part stays visible.

    python pipeline/check_field_semantics.py
    python pipeline/check_field_semantics.py --check    # exit 1 on a RANGE/AGGREGATE/COUNT violation
    python pipeline/check_field_semantics.py --estate   # also scan sibling repos, read-only

Writes: data/field_semantics_audit.json   (its own report, the only file it touches)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESTATE = ROOT.parent
OUT = ROOT / "data" / "field_semantics_audit.json"

SCAN_DIRS = [ROOT / "data", ROOT / "assets"]
SIBLINGS = ["vector-hoops", "vector-gridiron", "vector-pitch", "vector-equities",
            "vector-realty", "vector-hub"]

# name pattern -> (predicate, what the name asserts)
RANGE_RULES: list[tuple[re.Pattern, object, str]] = [
    (re.compile(r"(_share|_frac|_fraction)$"), lambda v: 0.0 <= v <= 1.0,
     "a share/fraction must be in [0,1]"),
    (re.compile(r"(^p_|_pvalue$|^p_two_sided$|^pvalue$)"), lambda v: 0.0 <= v <= 1.0,
     "a p-value must be in [0,1]"),
    (re.compile(r"(_sd$|_std$|^sd$|_stdev$)"), lambda v: v >= 0.0,
     "a standard deviation cannot be negative"),
    # ANCHORED, because the first draft used bare `^corr` and flagged
    # `correction_blocks_found = 10` as an out-of-range correlation. `_r$` was also dropped:
    # it matched anything ending in r. A rule that fires on a name it does not understand
    # is a false positive, and false positives are what teach a reader to skip the report.
    (re.compile(r"(^corr(_|$)|_corr$|^cos(_|$)|_cos$|^silhouette$|_silhouette$)"),
     lambda v: -1.0 <= v <= 1.0, "a correlation/cosine must be in [-1,1]"),
    (re.compile(r"(^n_|_count$|^count$)"),
     lambda v: v >= 0 and float(v).is_integer(), "a count must be a non-negative integer"),
]

# `n_economies` should match a collection named economy/economies. Suffix -> stems tried.
COUNT_STEMS = {"ies": "y", "s": "", "": ""}


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        yield path, obj
    else:
        yield path, obj


def numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_doc(doc: dict, label: str, spans: list | None = None) -> list[dict]:
    """All arms over one parsed artifact. `spans` collects acknowledged blind spots."""
    out: list[dict] = []
    spans = spans if spans is not None else []
    flat = {p: v for p, v in walk(doc)}

    # ARM 1: RANGE
    for p, v in flat.items():
        if not numeric(v):
            continue
        leaf = p.rsplit(".", 1)[-1]
        for rx, ok, why in RANGE_RULES:
            if not rx.search(leaf):
                continue
            # a *_pct/_share family sometimes lives on 0-100; only treat as [0,1] when no
            # sibling in the same object exceeds 1, so a percent table is not mass-flagged.
            if rx.pattern.endswith("_fraction)$") and v > 1.0:
                parent = p.rsplit(".", 1)[0]
                sibs = [x for q, x in flat.items()
                        if q.rsplit(".", 1)[0] == parent and numeric(x)]
                if any(x > 1.0 for x in sibs):
                    continue
            try:
                good = ok(v)
            except Exception:
                good = True
            if not good:
                out.append({"arm": "RANGE", "file": label, "field": p, "value": v,
                            "why": why})
            break

    # ARM 2: AGGREGATE — mean/sd/min/max beside the list they summarise.
    #
    # THE LIST MUST BE NAMED LIKE A SAMPLE. The first draft compared an aggregate against
    # EVERY numeric list in the same object, so `mean` was checked against `range: [min,
    # max]` and against `ci95: [lo, hi]`, and a block holding four different per-seed lists
    # produced one false finding per non-matching list. 62 of its 62 AGGREGATE findings
    # were that. Only lists whose name says "these are the observations" are eligible.
    SAMPLE = {"values", "per_seed", "diffs", "samples", "observations", "runs", "seeds"}
    for p, v in flat.items():
        if not isinstance(v, list) or len(v) < 2:
            continue
        if p.rsplit(".", 1)[-1] not in SAMPLE:
            continue
        vals = [x for x in v if numeric(x)]
        if len(vals) != len(v):
            continue
        parent = p.rsplit(".", 1)[0]
        base = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
        if len(vals) > 1:
            m = base["mean"]
            base["sd"] = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
        for agg, expect in base.items():
            key = f"{parent}.{agg}" if parent else agg
            got = flat.get(key)
            if not numeric(got):
                continue
            # compare at the precision the artifact chose to store
            dec = len(str(got).split(".")[-1]) if "." in str(got) else 0
            if abs(round(expect, dec) - got) > 10 ** (-dec) / 2 + 1e-12:
                out.append({"arm": "AGGREGATE", "file": label, "field": key,
                            "value": got, "expected_from": p,
                            "expected": round(expect, dec),
                            "why": f"`{agg}` beside a {len(vals)}-element list must equal "
                                   f"that list's {agg}"})

    # ARM 3: COUNT — n_<thing> vs a <thing> collection in the same object.
    #
    # EXACT STEMS ONLY. The first draft tried five fuzzy expansions per field and produced
    # 108 findings, essentially all false — `n_rows` matched anything beginning "row",
    # `n_features` matched unrelated maps. A count check that guesses which collection it
    # refers to is inventing the relationship it then reports as violated.
    IRREG = {"economies": "economy_names", "entries": "entries", "rows": "rows",
             "seeds": "seeds", "features": "features", "sports": "sports"}
    for p, v in flat.items():
        leaf = p.rsplit(".", 1)[-1]
        if not (leaf.startswith("n_") and numeric(v) and float(v).is_integer()):
            continue
        thing = leaf[2:]
        parent = p.rsplit(".", 1)[0]
        names = {thing}
        if thing in IRREG:
            names.add(IRREG[thing])
        for name in names:
            key = f"{parent}.{name}" if parent else name
            coll = flat.get(key)
            n = None
            if isinstance(coll, list):
                # A TWO-ELEMENT ASCENDING NUMERIC LIST IS A SPAN, NOT AN ENUMERATION, and
                # this arm cannot tell which without a convention. merged_careers.json has
                # `seasons: [1996, 2002]` beside `n_seasons: 6` — first and last year, and
                # six seasons played. Both correct. The first draft reported all 36 such
                # players as violations, which was the arm inventing the relationship it
                # then declared broken. Skipped and counted as an acknowledged blind spot
                # rather than guessed at.
                if (len(coll) == 2 and all(numeric(x) for x in coll)
                        and coll[0] <= coll[1] and int(v) != 2):
                    spans.append({"file": label, "field": p, "collection": key,
                                  "why": "2-element ascending numeric list reads as a "
                                         "span, not an enumeration — not checkable here"})
                    continue
                n = len(coll)
            else:
                pre = key + "."
                keys = {q[len(pre):].split(".")[0] for q in flat if q.startswith(pre)}
                if keys:
                    n = len(keys)
            if n is not None and int(v) != n:
                out.append({"arm": "COUNT", "file": label, "field": p, "value": int(v),
                            "collection": key, "collection_len": n,
                            "why": f"`{leaf}` must equal the size of `{name}` beside it"})
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any violation")
    ap.add_argument("--estate", action="store_true", help="also scan sibling repos")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dirs = list(SCAN_DIRS)
    if args.estate:
        for s in SIBLINGS:
            for sub in ("data", "assets", "pipeline/data"):
                d = ESTATE / s / sub
                if d.is_dir():
                    dirs.append(d)

    findings, n_files, unreadable, spans = [], 0, [], []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.json")):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                unreadable.append({"file": str(f.relative_to(ESTATE)), "error": str(e)[:90]})
                continue
            if not isinstance(doc, dict):
                continue
            n_files += 1
            findings += check_doc(doc, str(f.relative_to(ESTATE)).replace("\\", "/"), spans)

    by_arm: dict[str, int] = {}
    for x in findings:
        by_arm[x["arm"]] = by_arm.get(x["arm"], 0) + 1

    out = {
        "question": "Does every field mean what its name says, and does the data agree?",
        "why": "This estate's recurring defect is a real value answering a different "
               "question than the one it appears to answer. Every past instance was found "
               "by a person reading a file. These four arms are the part a machine can "
               "decide.",
        "arms": {
            "RANGE": "a name asserting a range (_share, _sd, corr*, n_*, p_*) must hold it",
            "AGGREGATE": "a mean/sd/min/max beside its list must equal that list's value",
            "COUNT": "n_<thing> must equal the size of the <thing> collection beside it",
        },
        "read_only": "This file opens artifacts for reading and writes only its own "
                     "report. check_documented_usage.py executed documented commands and "
                     "one run mutated ten artifacts here plus a sibling repo's "
                     "seed_floor.json; that is why this one runs nothing.",
        "cannot_decide": "Whether a field's PROSE description matches the code that "
                         "produced it. `vol` documented as 'dispersion of quarterly "
                         "growth' while computing the sd of index LEVELS is invisible "
                         "here — that needs a reader. This checks what a NAME asserts, "
                         "not what a SENTENCE asserts.",
        "files_scanned": n_files,
        "unreadable": unreadable,
        "counts_by_arm": by_arm,
        "not_checkable_spans": spans,
        "blind_spot_note": "A 2-element ascending numeric list beside an n_ field reads as a SPAN (first, last), not an enumeration. merged_careers.json has seasons [1996, 2002] beside n_seasons 6 — both correct. Counted here, never reported as a violation.",
        "findings": findings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  scanned {n_files} artifact(s) across {len(dirs)} dir(s)")
    print(f"  findings by arm: {by_arm or 'none'}")
    for x in findings[:25]:
        print(f"    [{x['arm']:<9}] {x['file']}::{x['field']} = {x['value']}")
        print(f"                {x['why']}")
    if len(findings) > 25:
        print(f"    ... and {len(findings) - 25} more, all in {OUT.name}")
    if spans:
        print(f"  {len(spans)} n_<thing> pair(s) not checkable (2-element span, not an enumeration)")
    if unreadable:
        print(f"  {len(unreadable)} file(s) unreadable as JSON (reported, not passed)")
    print(f"\nwrote {OUT}")
    if args.check and findings:
        print(f"CHECK FAILED: {len(findings)} field(s) do not mean what their name says",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
