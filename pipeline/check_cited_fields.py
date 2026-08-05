#!/usr/bin/env python3
"""A cited field must EXIST in the file it is cited from, and a cited VALUE must match it.

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

THE SECOND ARM CHECKS PUBLISHED NUMBERS, and it is the first thing in this repo to do so.
The citations do not only name fields, they assert values — `persistence_r=0.4514`,
`n_excluded_merged_names=112` — which makes "does the artifact agree" mechanically
answerable. 25 such assertions across the six pages; all 25 match. check_superlatives.py
does arithmetic on a page's own internal values and check_hub_freshness compares hashes, so
until now nothing compared a PUBLISHED NUMBER to the artifact it came from, which is the
literal promise in the site's fine print.

Values are found by RECURSIVE KEY SEARCH, not a path walk, so this arm is immune to the
shorthand that defeated three versions of the field parser. Top level wins when the key is
there: `n_test` occurs 6x in hoops_forward_report.json (top level 2290, plus one per
cut_year_sweep entry), and requiring all occurrences to agree made a correct citation look
ambiguous.

COMPARISON IS PRECISION-AWARE — the artifact is rounded to the decimals the page chose to
show, so a page quoting 0.451 against a stored 0.4514 is correct, not a discrepancy. Worth
being honest about: NO citation in the current corpus actually needs this. Artifacts store
values already rounded, so all 25 compare at equal precision and the branch is defensive
rather than exercised. It is unit-tested directly instead of being assumed.

BOTH ARMS GATE THE BUILD, which took a second mutation to discover. The first version
returned 1 only for a missing field, so a planted WRONG VALUE printed "1 WRONG" and exited
0 — reporting the defect and passing the build. The field arm's mutation passed throughout
and would have covered for it indefinitely. A guard needs a mutation per ARM, not per file.

    python pipeline/check_cited_fields.py
    python pipeline/check_cited_fields.py --check     # exit 1 on a MISSING field or WRONG value
    python pipeline/check_cited_fields.py --verbose   # list every uncovered reference
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portable_paths import ESTATE as _ESTATE, resolve  # noqa: E402

# WAS Path("C:/Users/jcdav/vector-hub/assets/data") — an absolute path on one laptop.
# Not merely clone-fragile: on any other box HUB did not exist, every loop below iterated
# over nothing, and this gate printed "all 0 published values match it" above a green
# line. It has verified zero published values for every reader who is not me.
#
# The convention was ALREADY DECIDED and I nearly re-litigated it. portable_paths.py
# defines ESTATE = <repo>.parent for exactly this, and migrate_hub_portable_paths.py has
# already applied the same idea to the published citations, for the same stated reason:
# "a real value answering a different question than the one it appears to answer. The
# path is real. It resolves — for me."
HUB = _ESTATE / "vector-hub" / "assets" / "data"
SLUGS = ("hoops", "gridiron", "pitch", "equities", "tennis", "unified")

SPLIT_ARROW = re.compile(r"->|\u2192")
SIMPLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


# A field list this checker will attempt. Deliberately NARROW: identifiers, dots, commas,
# whitespace, and `name=value`. Anything else means a shorthand this parser does not
# understand, and the segment is reported UNCOVERED rather than guessed at.
FLAT_LIST = re.compile(r"[A-Za-z0-9_.,=\s'\-]+$")


# A field name that marks its value as retired. The convention across the estate is a
# SUPERSEDED/superseded prefix on the key itself.
SUPERSEDED_KEY = re.compile(r"^(SUPERSEDED|superseded)")

# The insight must SAY the number is history. These are the words the four currently
# correct citations use; a page that quotes a retired value with none of them is
# presenting it as current.
HISTORY_CUE = re.compile(
    r"\b(supersed\w*|earlier|retired|no longer|formerly|previously|used to|history|"
    r"withdrawn|replaced|was\s+wrong)\b", re.I)


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


VALUE_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)(?![\d.])")


def find_key_values(obj, key: str) -> list:
    """Every value stored under `key` anywhere in a nested structure.

    A RECURSIVE SEARCH RATHER THAN A PATH WALK, and that is the whole trick. The citations
    assert values inside the same shorthand that defeated three versions of the field
    parser — `{..., persistence_r=0.4514, cut_year_sweep[5].{...}, ...}` — so any approach
    needing the prefix structure inherits that problem. A value assertion does not need it:
    `persistence_r=0.4514` is checkable by finding `persistence_r` anywhere in the artifact.
    """
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out += find_key_values(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out += find_key_values(v, key)
    return out


def value_matches(cited: str, actual) -> bool:
    """Does the artifact's value agree with the cited one AT THE CITED PRECISION?

    Pages quote rounded numbers; artifacts store full precision. `persistence_r=0.4514`
    against a stored 0.45137 is a correct citation, not a discrepancy, so the comparison
    rounds the ARTIFACT to the number of decimals the PAGE chose to show. Comparing raw
    floats would report every rounded citation as a mismatch — the false-positive flood
    this file already produced once, in a different form.
    """
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    if "." in cited:
        dp = len(cited.split(".")[1])
        return round(float(actual), dp) == round(float(cited), dp)
    return float(actual) == float(cited)


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
    mismatched: list[str] = []
    ambiguous: list[str] = []
    unlocated: list[str] = []
    unresolved: list[str] = []
    unparseable: list[str] = []
    checked = 0
    vals_checked = 0
    # Pages that verified no VALUE at all. Kept per page because the refusal below is a
    # global count, and a global count cannot see a page with zero coverage sitting behind
    # one with plenty.
    zero_value_pages: list[tuple[str, int, int]] = []
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
        s_valok = s_valbad = s_valskip = 0
        for key in ("insights", "headline_stats"):
            for i, item in enumerate(doc.get(key) or []):
                src = item.get("source") or ""
                where = f"{slug}:{key}[{i}]"
                prev_base = ""
                for seg in src.split(";"):
                    if SPLIT_ARROW.search(seg):
                        fpart, fields = SPLIT_ARROW.split(seg, 1)
                        base = Path(fpart.strip().rstrip(":")).name
                    elif prev_base and "=" in seg:
                        # A `;` segment with no arrow CONTINUES the previous file:
                        #   feature_audit.json -> dead_or_constant (33 entries); features = 118
                        # Skipping these dropped an assertion silently, which is the
                        # failure this file keeps finding in other checkers. Safe because a
                        # wrong inheritance almost always yields "key not in that file",
                        # which is REPORTED, not failed — only a key that IS present with a
                        # different value can fail, and that is a real finding either way.
                        base, fields = prev_base, seg
                    else:
                        continue
                    prev_base = base
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

                    # ---- VALUE ASSERTIONS ---------------------------------------
                    # Runs on the RAW segment, before any field-list parsing, because a
                    # `key=value` pair is checkable without knowing the prefix structure
                    # that defeated three parser versions. This is the first check in the
                    # repo that compares a PUBLISHED NUMBER against its artifact — the
                    # site's fine print promises exactly that and nothing tested it.
                    for k, cited_v in VALUE_PAIR.findall(fields):
                        found = find_key_values(target, k)
                        if not found:
                            # NOT a pass. The page asserts a value for a key its cited file
                            # does not contain anywhere — reported, because a silently
                            # skipped assertion is indistinguishable from a verified one in
                            # the summary line, and that is how coverage rots unnoticed.
                            s_valskip += 1
                            unlocated.append(f"{where}: asserts {k}={cited_v} but '{k}' "
                                             f"appears nowhere in {base}")
                            continue
                        # TOP LEVEL WINS when the key is there. `n_test` occurs 6x in
                        # hoops_forward_report.json — once at top level (2290) and once per
                        # cut_year_sweep entry (5466/4423/3345/2290/1160). Requiring every
                        # occurrence to agree made a correct, unambiguous citation
                        # "ambiguous"; a bare key most naturally means the top-level field,
                        # which is also how the field check reads it.
                        if isinstance(target, dict) and k in target:
                            actual = target[k]
                        else:
                            uniq = {json.dumps(x, sort_keys=True) for x in found}
                            if len(uniq) > 1:
                                # Not at top level and the nested copies disagree. Which one
                                # the page meant is genuinely undecidable, and guessing is
                                # how this file produced 43 false positives already.
                                s_valskip += 1
                                ambiguous.append(f"{where}: {k} appears {len(found)}x in "
                                                 f"{base} with differing values and none "
                                                 f"at top level")
                                continue
                            actual = found[0]
                        vals_checked += 1
                        if value_matches(cited_v, actual):
                            s_valok += 1
                        else:
                            s_valbad += 1
                            mismatched.append(
                                f"{where}: cites {base} -> {k}={cited_v}, but the file "
                                f"says {found[0]!r} — the page publishes a number its own "
                                f"source does not support")

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
        print(f"  {slug:9} {s_ok:3} fields ok  {s_miss:2} MISSING   "
              f"{s_valok:2} values ok  {s_valbad:2} WRONG   "
              f"{s_unres:2} unresolved  {s_unparse:2} uncovered")
        # A PAGE THAT VERIFIES ZERO VALUES IS NOT A VERIFIED PAGE. The refusal further down
        # counts vals_checked ACROSS ALL PAGES, so one page carrying 24 comparisons keeps
        # the gate green while another has had none at all. pitch (0 of 5) and unified
        # (0 of 10) sat exactly there, reported clean, while pitch's own artifact said
        # distinct_names 374 and the live page said 1,833 — a 5x disagreement no arm looked
        # at, because the page states that number inside a LABEL rather than in a value
        # field, and only value fields are compared.
        #
        # Reported per page rather than folded into the total. Not made a failure here: the
        # gate is already red on tennis, and turning a blind spot into a second red line
        # would change what "cited_fields is failing" means to whoever reads the board.
        if s_valok == 0 and (s_ok or s_unparse):
            zero_value_pages.append((slug, s_ok, s_unparse))

    if zero_value_pages:
        print(f"\n  {len(zero_value_pages)} page(s) verified ZERO published values — a "
              f"blind spot, not a clean bill:")
        for _slug, _ok, _un in zero_value_pages:
            print(f"    {_slug:9} {_ok} field(s) confirmed to EXIST, {_un} reference(s) "
                  f"uncovered, 0 value(s) compared")

    print(f"\n  {checked} simple field reference(s) checked against their cited file")
    print(f"  {vals_checked} published VALUE(s) compared against the artifact")
    if unparseable:
        print(f"  {len(unparseable)} reference(s) NOT COVERED (indexed/prose form) — "
              f"neither pass nor fail")
        if args.verbose:
            for u in unparseable:
                print(f"      {u}")
    if unlocated:
        print(f"  {len(unlocated)} value assertion(s) whose KEY is not in the cited file:")
        for u in unlocated:
            print(f"      {u}")
    if ambiguous:
        print(f"  {len(ambiguous)} value assertion(s) ambiguous (key repeats with "
              f"differing values):")
        for u in ambiguous:
            print(f"      {u}")
    if unresolved:
        print(f"  {len(unresolved)} citation(s) whose FILE could not be pinned down:")
        for u in unresolved[:8]:
            print(f"      {u}")

    # BOTH failure kinds gate the build. The first version returned 1 only for `missing`,
    # so a planted wrong value printed "1 WRONG" and exited 0 — the check reported the
    # defect and passed the build anyway. That is a vacuous gate of the most deceptive
    # kind, because its output looks like it is working. Found only by mutation-testing the
    # value arm separately; the field arm's mutation had passed and would have covered for
    # it indefinitely.
    if mismatched:
        print(f"\n{len(mismatched)} published value(s) disagree with their artifact:")
        for m in mismatched:
            print(f"  {m}")
        print("\nThis is the site's own fine print failing: 'Every number is recomputable "
              "from public sources'. A number whose cited source says something else is "
              "not a rounding quibble — it is the page and the artifact disagreeing about "
              "a fact, with the reader given no way to tell.")
    if missing:
        print(f"\n{len(missing)} cited field(s) do not exist:")
        for m in missing:
            print(f"  {m}")
        print("\nA citation naming a field its file does not contain is not a small error: "
              "it is the page asserting that a source supports a claim it never made.")
    # ---- THIRD ARM: a superseded field cited as though it were live ------------
    # The two arms above ask "does the cited field EXIST" and "does its value MATCH". A
    # superseded value passes both — it is still in the file, and still equal to what the
    # page prints. Nothing asked whether it is still TRUE.
    #
    # Four superseded values are quoted on live pages right now: hoops -0.096, tennis
    # 0.0124, unified 3.287 and 0.4333. All four are framed correctly as history —
    # "survives in the file as history and is explicitly not used", "the file itself
    # marks that superseded", "An earlier version of this file quoted 3.287x as a
    # salvage", "The retired 0.4333 target". That is a property of how they happen to be
    # written, not of anything enforced, and an edit that dropped the framing would leave
    # both existing arms green.
    stale_cites = []
    for slug in SLUGS:
        p = HUB / f"{slug}.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        page = f"{slug}.json"
        for i, ins in enumerate(doc.get("insights", []) or []):
            src = str(ins.get("source") or "")
            body = str(ins.get("body") or "")
            cited = {w.strip(" ,;{}()") for w in re.split(r"[\s,;{}()]+", src)}
            for f in sorted(c for c in cited if SUPERSEDED_KEY.match(c)):
                if not HISTORY_CUE.search(body):
                    stale_cites.append(
                        f"{page} insights[{i}] cites {f} and its body never says the "
                        f"value is superseded — a retired number presented as current")
    if stale_cites:
        print(f"\n{len(stale_cites)} superseded field(s) cited without saying so:")
        for m in stale_cites:
            print(f"  {m}")
        print("\nThe existing arms cannot catch this: a superseded value still EXISTS in "
              "its file and still MATCHES what the page prints. Only the framing "
              "distinguishes 'this was once believed' from 'this is the number'.")

    if missing or mismatched or stale_cites:
        return 1 if args.check else 0

    # ZERO COVERAGE IS NOT A PASS, and this guard exists because it silently was one.
    # On a fresh clone the pages this reads live under a path git does not carry, so every
    # loop above iterated over nothing and the summary printed "all 0 published values
    # match it" above a green line. 64 values are checked on the box that has them. A
    # reader cloning this repo saw the same PASS having verified nothing — worse than a
    # failure, because a failure prompts a look.
    #
    # Measured by pipeline/check_gate_inputs_tracked.py, which clones the repo and runs
    # validate.py inside the clone. It is the ONLY check that passes vacuously there:
    # superlatives (2/8/5), corrections_landed (27), internal_prose (26) and
    # merged_careers (6) all keep real coverage on a clone.
    #
    # NOT a --check-only guard. Reporting an honest count matters most to the reader who
    # runs this without --check, so the refusal prints in both modes and only the exit
    # code differs.
    if vals_checked == 0:
        print(f"\nREFUSING TO PASS: 0 published values were checked. This gate verifies "
              f"published numbers against the artifacts they cite; with nothing to "
              f"verify it has established nothing, and a green line here would assert "
              f"otherwise. Usually means the hub pages are absent — expected on a fresh "
              f"clone, since they are deliberately untracked.", file=sys.stderr)
        return 1 if args.check else 0

    n_sup = sum(1 for slug in SLUGS
                if (HUB / f"{slug}.json").exists()
                for ins in (json.loads((HUB / f"{slug}.json").read_text(encoding="utf-8"))
                            .get("insights", []) or [])
                for w in re.split(r"[\s,;{}()]+", str(ins.get("source") or ""))
                if SUPERSEDED_KEY.match(w.strip(" ,;{}()")))
    print(f"\nEvery checkable cited field exists in the file it is cited from, and all "
          f"{vals_checked} published values match it. "
          f"{n_sup} citation(s) of a SUPERSEDED field, each framed as history — "
          f"a green third arm means the framing is present, not that no retired value "
          f"is quoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
