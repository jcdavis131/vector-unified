#!/usr/bin/env python3
"""A cited field must EXIST in the file it is cited from.

Solo personal project, no connection to employer, built with public/free-tier only

WHY THIS EXISTS, and it is not hypothetical — it is the defect that produced it. Across two
commits (d840a47, then 043cdb9 which "corrected" the first) this repo asserted that
vector-equities/pipeline/data/mtnn_report.json "records only recall@10, cross-cycle
archetype purity and sector top-1 accuracy, and no next-year head". The file has a
top-level next_profile block:

    next_profile.val   rows 990  r2 0.262
    next_profile.test  rows 500  r2 0.1965

I never opened it. Both sentences came from a remembered summary of the report, trusted
because it was my own earlier sentence — a real value answering a different question than
the one it appeared to answer, which is this repo's whole subject, turned on itself.

WORSE: dumbmodel.com's equities page had been citing `mtnn_report.json -> next_profile.val,
next_profile.test` in insights[4] since it was built and adversarially verified. The
evidence sat in the published, checked page while I contradicted it. Nothing compared the
two, because nothing had ever read a `source` string as anything but prose.

THEY ARE NOT PROSE. They are machine-readable citations:

    pipeline/data/mtnn_report.json -> next_profile.val, next_profile.test, composite.parts

so "does this field exist in that file" is a question a computer can answer.

AND TO BE EXACT ABOUT WHAT THIS DOES NOT DO: it would NOT have caught the error above. That
claim lived in a PIPELINE DOCSTRING, not a page citation, and the citation it contradicted
was already correct. Saying "this check would have caught it" was the first thing written
about this file and it was wrong — the same overclaiming the rest of the repo guards
against, about a guard. What the incident actually supplies is the reason to read `source`
strings mechanically at all. The class this check owns is its own: a page citing a field
its source does not contain, which is the page asserting that an artifact supports a claim
the artifact never made.

HOW FILES ARE RESOLVED. Citations name files relatively and inconsistently
(`assets/x.json`, `pipeline/data/y.json`, bare `z.json`) and what they are relative to
depends on which page they are on. Rather than guessing a root, each citation's BASENAME is
matched against the page's own `source_files` list, which is already portable and already
checked to exist by check_hub_freshness. Exactly one match resolves; zero or several is
reported UNRESOLVED and is never a pass.

WHAT THIS CANNOT DO, stated because a checker that hides its coverage is worse than none.
Roughly 30% of field references are not simple dotted paths — `points[].skills`,
`careers[name='nikola jokic'].residual`, `overall.spearman / baseline_last4`, prose like
`dead_or_constant (33 entries)`. Those are reported UNPARSEABLE and counted separately.
They are NOT failures and they are NOT passes; they are the part of the corpus this check
does not cover, printed every run so the number stays visible.

    python pipeline/check_cited_fields.py
    python pipeline/check_cited_fields.py --check     # exit 1 if a cited field is MISSING
    python pipeline/check_cited_fields.py --verbose   # list every unparseable reference
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portable_paths import resolve  # noqa: E402

HUB = Path("C:/Users/jcdav/vector-hub/assets/data")
SLUGS = ("hoops", "gridiron", "pitch", "equities", "tennis", "unified")

SPLIT_ARROW = re.compile(r"->|\u2192")
SIMPLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


# A field list this checker will attempt. Deliberately NARROW: identifiers, dots, commas,
# whitespace, and `name=value`. Anything else means a shorthand this parser does not
# understand, and the segment is reported UNCOVERED rather than guessed at.
FLAT_LIST = re.compile(r"[A-Za-z0-9_.,=\s'\-]+$")


def expand_fields(fields: str) -> list[str] | None:
    """Split a FLAT comma-separated field list, or return None if it is not flat.

    TWO ROUNDS OF FALSE POSITIVES CAME FROM TRYING TO BE CLEVER HERE, and that is why this
    is narrow now. The citations are a human shorthand with at least four conventions:

        {question, persistence_r=0.4514, cut_year_sweep[5].{cut_year,gain}, ...}
        null_extras_shuffled (mean, sd, pct95, p_value_of_real_gain, reps, what)
        overall.spearman / baseline_last4 / per_week
        model.report.n_params, n_features, n_families        <- prefix implied, not repeated

    A naive comma split reported 28 fields MISSING; adding brace expansion still reported
    15. Every one of the 43 was present in its file under a prefix the parser had dropped.
    A checker whose failures are its own parsing bugs is worse than no checker: the noise
    buries any real finding, and it trains its reader to dismiss the output.

    So the rule is inverted. Parse only what is unambiguous, and report everything else as
    NOT COVERED — visibly, with a count, every run. Low recall and high precision is the
    correct trade for a gate; the reverse is how a gate becomes decorative.
    """
    if not FLAT_LIST.match(fields.strip()):
        return None
    return [f.strip() for f in fields.split(",") if f.strip()]


def field_exists(doc, dotted: str) -> bool:
    """Walk a dotted path. A list on the way means the path is not simple after all."""
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False
    return True


def exists_either_reading(doc, field: str, implied_prefix: str) -> bool:
    """True if the field resolves at top level OR under an IMPLIED PREFIX.

    The corpus writes a shared parent once and lets the rest of the list inherit it:

        projections.json -> model.report.n_params, n_features, n_families

    `n_features` is `model.report.n_features`, not a top-level key. Reading it literally
    produced the last three false positives of a checker that had already produced forty.

    This accepts EITHER reading and fails only when neither resolves. That is not the same
    as guessing: a field absent under both interpretations is genuinely not in the file, so
    the check keeps its teeth while losing its ability to accuse a citation that has a
    valid reading. Verified non-vacuous by check_guards_nonvacuous.py, which plants a field
    name that exists under neither.
    """
    if field_exists(doc, field):
        return True
    return bool(implied_prefix) and field_exists(doc, f"{implied_prefix}.{field}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    missing: list[str] = []
    unresolved: list[str] = []
    unparseable: list[str] = []
    checked = 0
    cache: dict[str, object] = {}

    for slug in SLUGS:
        p = HUB / f"{slug}.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        by_base: dict[str, list[str]] = {}
        for sf in doc.get("source_files") or []:
            by_base.setdefault(Path(sf).name, []).append(sf)

        s_ok = s_miss = s_unres = s_unparse = 0
        for key in ("insights", "headline_stats"):
            for i, item in enumerate(doc.get(key) or []):
                src = item.get("source") or ""
                where = f"{slug}:{key}[{i}]"
                for seg in src.split(";"):
                    if not SPLIT_ARROW.search(seg):
                        continue
                    fpart, fields = SPLIT_ARROW.split(seg, 1)
                    base = Path(fpart.strip().rstrip(":")).name
                    hits = by_base.get(base) or []

                    if len(hits) != 1:
                        s_unres += 1
                        unresolved.append(
                            f"{where}: '{base}' matches {len(hits)} entries in this page's "
                            f"source_files — cannot check its fields")
                        continue
                    cited = hits[0]
                    if cited.endswith(".npz"):
                        s_unparse += 1
                        unparseable.append(f"{where}: {base} is not JSON")
                        continue
                    if cited not in cache:
                        q = resolve(cited)
                        if q is None or not q.exists():
                            cache[cited] = None
                        else:
                            try:
                                cache[cited] = json.loads(q.read_text(encoding="utf-8"))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                cache[cited] = None
                    target = cache[cited]
                    if target is None:
                        s_unres += 1
                        unresolved.append(f"{where}: {cited} unreadable as JSON")
                        continue

                    parsed = expand_fields(fields)
                    if parsed is None:
                        s_unparse += 1
                        unparseable.append(f"{where}: {base} -> {fields.strip()[:54]}")
                        continue
                    # The parent of the FIRST dotted element, which later bare elements in
                    # the same list may be inheriting.
                    implied = ""
                    for raw in parsed:
                        head = raw.split("=")[0].strip()
                        if "." in head:
                            implied = head.rsplit(".", 1)[0]
                            break

                    for raw in parsed:
                        fld = raw.strip()
                        if not fld:
                            continue
                        bare = fld.split("=")[0].strip()
                        if not SIMPLE.match(bare):
                            s_unparse += 1
                            unparseable.append(f"{where}: {bare[:58]}")
                            continue
                        checked += 1
                        if exists_either_reading(target, bare, implied):
                            s_ok += 1
                        else:
                            s_miss += 1
                            missing.append(
                                f"{where}: cites {base} -> {bare}, which is in that file "
                                f"neither at top level nor under '{implied or '(none)'}' — "
                                f"the page states a source that does not say it")
        print(f"  {slug:9} {s_ok:3} verified   {s_miss:2} MISSING   "
              f"{s_unres:2} unresolved   {s_unparse:2} unparseable")

    print(f"\n  {checked} simple field reference(s) checked against their cited file")
    if unparseable:
        print(f"  {len(unparseable)} reference(s) NOT COVERED (indexed/prose form) — "
              f"neither pass nor fail")
        if args.verbose:
            for u in unparseable:
                print(f"      {u}")
    if unresolved:
        print(f"  {len(unresolved)} citation(s) whose FILE could not be pinned down:")
        for u in unresolved[:8]:
            print(f"      {u}")

    if missing:
        print(f"\n{len(missing)} cited field(s) do not exist:")
        for m in missing:
            print(f"  {m}")
        print("\nA citation naming a field its file does not contain is not a small error: "
              "it is the page asserting that a source supports a claim it never made.")
        return 1 if args.check else 0

    print("\nEvery checkable cited field exists in the file it is cited from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
