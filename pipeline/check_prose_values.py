#!/usr/bin/env python3
"""A number in a sentence is a published claim. Check it against the field it cites.

Solo personal project, no connection to employer, built with public/free-tier only

check_cited_fields.py compares `value` fields. It does not read prose, and that gap let
FOUR artifacts sit stale for months while their pages published different numbers:

    gridiron_name_collisions.json   names_probed 300, colliding_names 15
    the gridiron page               "a Wikidata probe of 2,707 corpus names found 117"

The page was right; the local artifact was a `--limit 300` smoke run left standing. The
gate said `gridiron 23 fields ok, 0 WRONG` throughout, because 2,707 lives inside an
insight body and 1,833 inside a headline LABEL — never in a `value` key. A number a reader
sees is a published claim regardless of which JSON key holds it.

WHAT IT CHECKS. Every insight and headline stat whose `source` names a file and fields.
It loads the artifact, takes the NUMERIC value of each cited field, and asks whether that
number appears anywhere in the text the reader sees.

THE RULE THAT MAKES IT PRECISE. A single missing value proves nothing — a page may cite a
field for provenance without ever quoting it. The signal is CATEGORICAL:

    the prose states numbers, AND
    NOT ONE of the numeric fields it cites has its value anywhere in that prose

That is the gridiron shape exactly: cites names_probed(300) and colliding_names(15), prose
says 2,707 and 117, overlap zero. A page quoting its own artifact virtually always lands at
least one number; a page whose artifact has drifted lands none.

DELIBERATELY NOT CLAIMED. This does not say WHICH number is right. When the four were
found, the PAGE was current and the ARTIFACT was stale every time — so a finding here means
"these disagree", not "the page is wrong". Fixing one by editing the other is exactly the
mistake the tennis citation gap records.

CALIBRATION, MEASURED.

    coverage      41 of 60 blocks. The FIRST draft reached 6 of 60 and reported 0 findings
                  — a vacuous green — because it reused check_cited_fields.expand_fields,
                  which rejects the braced `{a, b=118}` form that 44 of the 60 blocks use.
                  That form is RICHER, not malformed: `eligible_careers=2415` states the
                  value inline, which is why there is a second arm below.
    precision     2 of 2 on current data, both confirmed by reading the page text. A third
                  was a FALSE POSITIVE and is why variants() emits the integer form of a
                  float: the unified page quotes mean_b_rank 2114.18 as "2,114", which is
                  simply how a rank is written.
    non-vacuity   PROVEN, not asserted. Restoring the stale gridiron_name_collisions.json
                  (names_probed 300) makes this flag gridiron:insights[4] immediately;
                  with the current artifact (2707) it does not. It detects the defect it
                  was written for.

Values under 10 are ignored for the categorical test: `1`, `2`, `5` appear in prose by
coincidence constantly ("two or more players", "R1 P8"), and counting a chance match as
verification is how a checker starts lying in the reassuring direction.

SECOND ARM — INLINE VALUES. A source segment can assert the number itself
(`eligible_careers=2415`). That is compared directly against the artifact, with no prose
matching and no ambiguity. Currently 0 disagreements.

EVERY FINDING CARRIES THE ARTIFACT'S VCS STATE, because without it the reader cannot tell
which of three different problems they have:

    UNCOMMITTED-LOCAL-CHANGE   gridiron:headline_stats[2]. The page says "21 of them
                               rookies 646"; projections.json on disk holds 647/22. Reads
                               as a stale page — but HEAD holds 646/21 built 2026-07-21 and
                               the working tree holds 647/22 built 2026-08-04, ONE
                               uncommitted rookie apart (Kaden Prather). THE PAGE MATCHES
                               WHAT IS COMMITTED. Nothing is stale; a sibling repo has a
                               regeneration nobody landed. Fix is commit-and-rebuild or
                               discard, never edit the page.
    untracked                  pitch:insights[2]. The artifact is gitignored, so git can
                               say nothing and the disagreement stands on its own.
    committed                  a genuine page/artifact divergence, the case worth chasing.

Reporting all three as "page and artifact disagree" would send a reader to correct a page
that is already right.

    python pipeline/check_prose_values.py
    python pipeline/check_prose_values.py --verbose   # every field, matched or not
    python pipeline/check_prose_values.py --check     # exit 1 on a zero-overlap insight

Writes: data/prose_values_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_cited_fields import (  # noqa: E402
    HUB, SLUGS, SPLIT_ARROW, expand_fields,
)
from portable_paths import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prose_values_audit.json"

# Numbers below this are ignored for the categorical test. See CALIBRATION above.
TRIVIAL = 10

NUM_IN_TEXT = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def variants(v) -> list[str]:
    """Every way this repo's pages render a number, so a match is not missed on formatting.

    `2707` is published as "2,707"; `84.0` appears as "84.0%" and sometimes "84%". Missing a
    real match would push this check toward false alarms, which is the failure mode that
    makes a reader skim.
    """
    out: list[str] = []
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return out
    if isinstance(v, int) or float(v).is_integer():
        n = int(v)
        out += [str(n), f"{n:,}"]
    else:
        out += [f"{v}", f"{v:,}"]
        for nd in (1, 2, 3):
            r = round(float(v), nd)
            out += [f"{r}", f"{r:,}"]
        # THE INTEGER FORM OF A FLOAT. Missing this produced the first false positive:
        # analogy_triples_report.json holds mean_b_rank 2114.18 and the unified page states
        # "2,114" — a rank quoted as a whole number, which is the natural way to write it.
        # The checker called that a page/artifact disagreement when the page was quoting its
        # artifact correctly. A false alarm in a gate whose whole subject is "numbers that
        # do not match" is worse than most, because it is indistinguishable from the real
        # thing until someone opens the page.
        for n in {int(float(v)), int(round(float(v)))}:
            out += [str(n), f"{n:,}"]
    return list(dict.fromkeys(out))


def numeric_fields(doc, fields: list[str]) -> dict:
    """field -> numeric value, for the cited fields that resolve to a number."""
    got = {}
    for f in fields:
        cur, ok = doc, True
        for part in f.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)) and not isinstance(cur, bool):
            got[f] = cur
    return got


BRACES = re.compile(r"^[\s{(\[]+|[\s})\]]+$")


def parse_fields(fields: str) -> list[tuple[str, str | None]]:
    """`{a, b=118, c.d}` -> [(a, None), (b, '118'), (c.d, None)].

    THE BRACED FORM IS THE MAJORITY AND expand_fields REJECTS IT. Reusing that function
    unchanged examined 6 of 60 blocks and reported 0 findings — a vacuous green, the exact
    failure this repo has shipped before. 44 of the 60 use `{...}`, and it is the RICHER
    form, not a malformed one: `eligible_careers=2415` states the artifact's value inline,
    which is directly checkable rather than merely quotable.
    """
    out: list[tuple[str, str | None]] = []
    for tok in BRACES.sub("", fields).split(","):
        tok = tok.strip().rstrip(".").strip()
        if not tok or " " in tok.split("=")[0].strip():
            continue
        if "=" in tok:
            name, _, val = tok.partition("=")
            out.append((name.strip(), val.strip()))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", tok):
            out.append((tok, None))
    return out


def vcs_state(p: Path) -> str:
    """Is this artifact committed, locally modified, or not tracked at all?

    A DISAGREEMENT MEANS DIFFERENT THINGS DEPENDING ON THIS, and without it a reader cannot
    act on the finding. The gridiron case: the page says "21 of them rookies 646" and
    projections.json on disk holds 647/22 — which reads as a stale page until you check
    git, where HEAD holds 646/21 built 2026-07-21 and the working tree holds 647/22 built
    2026-08-04, one uncommitted rookie apart (Kaden Prather). The PAGE MATCHES WHAT IS
    COMMITTED. Nothing is stale; a sibling repo has an uncommitted regeneration nobody
    landed, and the fix is to commit-and-rebuild or discard, not to edit the page.

    An untracked artifact is a third case again: git cannot say anything about it, so the
    disagreement stands on its own.
    """
    try:
        repo = p.parent
        while repo != repo.parent and not (repo / ".git").exists():
            repo = repo.parent
        if not (repo / ".git").exists():
            return "no-repo"
        rel = str(p.relative_to(repo)).replace("\\", "/")
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=str(repo),
                                 capture_output=True).returncode == 0
        if not tracked:
            return "untracked"
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=str(repo),
                               capture_output=True).returncode != 0
        return "UNCOMMITTED-LOCAL-CHANGE" if dirty else "committed"
    except Exception:
        return "unknown"


def lookup(doc, dotted: str):
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def check_block(text: str, source: str, by_base: dict, cache: dict) -> dict | None:
    """One insight/headline: do its cited values appear in its own prose, and inline?"""
    if not text or not source:
        return None
    cited: dict[str, dict] = {}
    inline_wrong: list[dict] = []
    prev_base = ""
    for seg in str(source).split(";"):
        if SPLIT_ARROW.search(seg):
            fpart, fields = SPLIT_ARROW.split(seg, 1)
            base = Path(fpart.strip().rstrip(":")).name
        elif prev_base and "=" in seg:
            base, fields = prev_base, seg      # `;` continuation, same convention as
        else:                                   # check_cited_fields.py
            continue
        prev_base = base

        hits = by_base.get(base) or []
        if len(hits) != 1:
            continue
        key = hits[0]
        # CACHE THE PATH AND ITS VCS STATE ALONGSIDE THE DOC. Reading `q` below worked only
        # on a cache MISS; on a hit it was undefined or, worse, still bound to whichever
        # file was resolved last — a stale value that looks like a real one.
        if key not in cache:
            q = resolve(key)
            try:
                doc = json.loads(q.read_text(encoding="utf-8")) if q and q.is_file() else None
            except Exception:
                doc = None
            cache[key] = (doc, q, vcs_state(q) if q and q.is_file() else "absent")
        doc, qpath, state = cache[key]
        if doc is None:
            continue

        for name, stated in parse_fields(fields):
            v = lookup(doc, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            cited[f"{base}:{name}"] = {"value": v, "file": base,
                                       "artifact": str(qpath), "vcs": state}
            # INLINE ARM: the source itself asserts a value. No ambiguity to resolve.
            if stated is not None:
                try:
                    want = float(stated.replace(",", "").rstrip("%"))
                except ValueError:
                    continue
                if abs(want - float(v)) > 1e-9:
                    inline_wrong.append({"field": f"{base}:{name}",
                                         "source_says": stated, "artifact_has": v})
    if not cited:
        return None

    present, absent = [], []
    for key, meta in cited.items():
        hit = any(re.search(rf"(?<![\d.,]){re.escape(s)}(?![\d.,])", text)
                  for s in variants(meta["value"]))
        (present if hit else absent).append({"field": key, "value": meta["value"],
                                             "vcs": meta["vcs"]})

    nontrivial = [a for a in (present + absent) if abs(float(a["value"])) >= TRIVIAL]
    nt_present = [a for a in present if abs(float(a["value"])) >= TRIVIAL]
    prose_numbers = NUM_IN_TEXT.findall(text)

    return {
        "cited_numeric_fields": len(cited),
        "present_in_prose": present,
        "absent_from_prose": absent,
        "nontrivial_cited": len(nontrivial),
        "nontrivial_present": len(nt_present),
        "prose_states_numbers": len(prose_numbers) > 0,
        "zero_overlap": bool(nontrivial) and not nt_present and bool(prose_numbers),
        "inline_wrong": inline_wrong,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on a zero-overlap insight")
    ap.add_argument("--verbose", action="store_true", help="every field, matched or not")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    findings, wrong, blocks_examined, pages = [], [], 0, 0
    cache: dict = {}
    for slug in SLUGS:
        f = HUB / f"{slug}.json"
        if not f.is_file():
            continue
        pages += 1
        doc = json.loads(f.read_text(encoding="utf-8"))
        by_base = {}
        for sf in doc.get("source_files") or []:
            by_base.setdefault(Path(sf).name, []).append(sf)
        blocks = [("insights", i, b.get("body", ""), b.get("source", ""), b.get("title", ""))
                  for i, b in enumerate(doc.get("insights", []) or [])]
        blocks += [("headline_stats", i,
                    f"{b.get('label','')} {b.get('value','')}", b.get("source", ""),
                    b.get("label", ""))
                   for i, b in enumerate(doc.get("headline_stats", []) or [])]
        for kind, idx, text, source, title in blocks:
            r = check_block(text, source, by_base, cache)
            if r is None:
                continue
            blocks_examined += 1
            if args.verbose:
                print(f"  {slug}:{kind}[{idx}]  {r['nontrivial_present']}/"
                      f"{r['nontrivial_cited']} nontrivial cited values found in prose")
            for w in r["inline_wrong"]:
                wrong.append({"page": slug, "where": f"{kind}[{idx}]", **w})
            if r["zero_overlap"]:
                findings.append({
                    "page": slug, "where": f"{kind}[{idx}]", "title": title[:90],
                    "cited_values_none_of_which_appear": r["absent_from_prose"],
                    "artifact_vcs_state": sorted({a["vcs"] for a in r["absent_from_prose"]}),
                    "why": "the prose states numbers and NOT ONE of the numeric fields it "
                           "cites has its value in that text — page and artifact disagree. "
                           "Which one is correct is NOT determined here; every instance "
                           "found so far had a current page and a stale artifact.",
                })

    out = {
        "question": "Does a number stated in prose match the field the page says it came from?",
        "why": "check_cited_fields compares `value` fields only. Four artifacts sat stale "
               "for months behind pages publishing different numbers — gridiron's artifact "
               "said names_probed 300 while its page said 'a Wikidata probe of 2,707 corpus "
               "names found 117' — and the gate reported 0 WRONG throughout.",
        "rule": "CATEGORICAL, not per-value: flagged only when the prose states numbers and "
                "NOT ONE cited numeric field value appears in it. A single missing value "
                "proves nothing, since a page may cite a field for provenance without "
                "quoting it.",
        "does_not_claim": "Which side is correct. Every instance found so far had a CURRENT "
                          "page and a STALE artifact, so 'fix the page' is the wrong "
                          "default.",
        "trivial_threshold": TRIVIAL,
        "pages": pages,
        "blocks_examined": blocks_examined,
        "findings": findings,
        "inline_value_disagreements": wrong,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n  {pages} page(s), {blocks_examined} block(s) with resolvable numeric citations")
    print(f"  ZERO-OVERLAP (page and artifact disagree): {len(findings)}")
    for f in findings:
        vals = ", ".join(f"{a['field']}={a['value']}"
                         for a in f["cited_values_none_of_which_appear"][:4])
        print(f"    {f['page']}:{f['where']}  {vals}")
        print(f"        artifact state: {', '.join(f['artifact_vcs_state'])}")
    print(f"\nwrote {OUT}")
    if args.check and (findings or wrong):
        print(f"CHECK FAILED: {len(findings)} block(s) where no cited value appears in the "
              f"prose that cites it", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
