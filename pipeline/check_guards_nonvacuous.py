#!/usr/bin/env python3
"""Plant a defect in front of each guard and require it to notice. (7.33)

Solo personal project, no connection to employer, built with public/free-tier only

check_gate_nonvacuity.py asks whether G1-G4 can fail on a null. NOTHING ASKED THAT OF THE
GUARDS THEMSELVES. Every check added this phase — merged_careers, artifact_freshness,
hub_freshness, superlatives, g1_position — was mutation-tested exactly once, by hand, in the
turn that created it. Those proofs live in commit messages. A commit message does not run.

That is the same shape as the defect the whole phase has been chasing: a green line whose
ability to go red was established at one moment and never re-established. `pos_knn5 = 1.0`
passed for months. `check_merged_careers.py` read the wrong key and reported clean over an
empty list. Both were verified once and trusted after.

METHOD. For each guard, plant a specific defect, run `--check`, require exit 1, restore, and
require exit 0 again. The restore assertion matters as much as the failure: a guard that
fails on everything is as useless as one that fails on nothing, and a test that leaves the
tree dirty poisons every check that runs after it.

WHAT THIS CANNOT DO, stated so the green line is not read as more than it is. It proves each
guard rejects THE DEFECT IT WAS SHOWN. It does not prove the guard catches every defect of
that class, and it cannot — that would require enumerating the class. The planted defects
are the ones that actually occurred in this repo, which is the best available evidence and
not a proof of coverage.

EVERY MUTATION IS APPLIED TO A COPY-ON-DISK AND RESTORED IN A finally BLOCK. If this script
is killed mid-run, the restore still runs; if the process is hard-killed, the .guardbak files
left behind are the recovery path and are reported.

    python pipeline/check_guards_nonvacuous.py
    python pipeline/check_guards_nonvacuous.py --check   # exit 1 if any guard is vacuous
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
HUB = Path("C:/Users/jcdav/vector-hub/assets/data")


def run(argv: list[str]) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(PIPE / argv[0]), *argv[1:]],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _patch_text(path: Path, old: str, new: str) -> None:
    t = path.read_text(encoding="utf-8")
    assert t.count(old) == 1, f"{path.name}: expected 1 occurrence of {old!r}"
    path.write_text(t.replace(old, new), encoding="utf-8")


def patch_json(path: Path, mutate) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")


# name -> (checker argv, target file, mutation, what the guard must notice)
def _drop_field(doc):
    doc["insights"][0].pop("body", None)


def _flip_answer(doc):
    r = doc["game"]["rounds"][0]
    r["answer"] = "a" if r["answer"] == "b" else "b"


def _false_superlative(doc):
    rs = doc["game"]["rounds"]
    widest = max(range(len(rs)), key=lambda i: abs(rs[i]["a"]["value"] - rs[i]["b"]["value"]))
    rs[widest]["reveal"] += " The closest call on this board."


def _contaminate_axis(doc):
    rows = doc.get("careers") or doc.get("players") or []
    if rows:
        rows[0] = dict(rows[0], name="jaren jackson")


def _bump_mtime(path: Path) -> None:
    """Make a producer look newer than its artifact. Not a JSON edit — restored by mtime."""
    import os, time
    t = time.time() + 7200
    os.utime(path, (t, t))


def _break_invariant(doc):
    """Change a drafted COUNT the invariant actually reads.

    I4 compares surv["per_position"][pos]["by_bucket"][b]["drafted"] against the value
    table's own cell — a PRECOMPUTED count in the report, not a count of players[] rows.
    The first version of this mutation deleted a WR from players[] and the guard exited 0,
    because nothing it reads had changed. That is the third mutation in this suite to target
    a field the guard does not look at, all three from assuming a checker's inputs rather
    than reading them. The guards exist to catch exactly that assumption; so does this note.

    I4 caught two real defects this phase: five private norm_name() copies disagreeing by
    one WR, and a value table rebuilt without its probe.
    """
    # Under doc["report"], not at top level — check_draft_value_invariants.py line 79 does
    # `json.loads(...)["report"]`. Reading that line is what finally located it, after two
    # wrong guesses.
    per = (doc.get("report") or {}).get("per_position") or {}
    for pos, blob in per.items():
        for b, cell in (blob.get("by_bucket") or {}).items():
            if isinstance(cell.get("drafted"), int):
                cell["drafted"] += 1
                return


def _absolute_path(doc):
    """Put a machine-local path back into a PUBLISHED citation.

    model.js:216 renders every source_files entry under "Every number above came from these
    files", so this is the provenance the page hands a reader. 45 of them shipped as
    C:/Users/jcdav/... — every one correct, every one openable on exactly one computer.
    """
    if doc.get("source_files"):
        doc["source_files"][0] = "C:/Users/jcdav/vector-unified/data/tennis_forward_report.json"


def _unknown_repo_root(doc):
    """Cite a repo that portable_paths.py has never heard of.

    The failure mode being guarded is not the bad citation itself but the TEMPTING handling
    of it: an unresolvable path is the one case where "skip it" reads as reasonable, and
    skipping would make freshness vacuous for exactly the entries most likely to be wrong.
    resolve() returns None and the checker must turn that into a failure, not a shrug.
    """
    if doc.get("source_files"):
        doc["source_files"][0] = "vector-cricket/data/nope.json"


def _phantom_cited_field(doc):
    """Cite a field name that exists under NO reading of the citation.

    check_cited_fields.py accepts either a top-level field or one under an implied prefix,
    because the corpus writes `model.report.n_params, n_features` and means both under
    model.report. That permissiveness is what stops false accusations — and it is exactly
    the kind of permissiveness that can quietly swallow the real case, so the planted field
    must exist under neither reading.
    """
    ins = doc.get("insights") or []
    if len(ins) > 4:
        ins[4]["source"] = ("pipeline/data/mtnn_report.json -> next_profile.val, "
                            "totally_invented_field")


def _wrong_cited_value(doc):
    """Publish a number the cited artifact contradicts.

    A SEPARATE MUTATION FROM _phantom_cited_field ON PURPOSE. check_cited_fields.py has two
    arms — field existence and value agreement — and its first version returned exit 1 only
    for a missing field. A wrong value printed "1 WRONG" and exited 0: the check reported
    the defect and passed the build. The field mutation passed the whole time and would
    have covered for it indefinitely. One mutation per ARM, not per file.
    """
    ins = doc.get("insights") or []
    if ins and "persistence_r=0.4514" in (ins[0].get("source") or ""):
        ins[0]["source"] = ins[0]["source"].replace("persistence_r=0.4514",
                                                    "persistence_r=0.9999")


def _carry_forward(doc):
    """Make every company-year a copy of its own prior year.

    THE ARM THIS TESTS DID NOT EXIST UNTIL IT WAS PLANTED FOR. build_equities_forward.py
    accepted --check, never read it, and was registered in validate.py as the
    `equities_forward` gate — a check that could not fail under any input, reporting PASS
    for three commits. Its docstring promised "exit 1 only if the run is broken".

    Carry-forward is what "broken" means for that file specifically. Persistence of
    0.85-0.92 is exactly what stale data looks like, so if the composites ever start
    duplicating year to year, every number in the report becomes an artifact of duplication
    while the verdict still reads as a finding about the model.
    """
    by_ticker: dict[str, list] = {}
    for p in doc.get("points") or []:
        by_ticker.setdefault(p["ticker"], []).append(p)
    for rows in by_ticker.values():
        rows.sort(key=lambda r: int(r["year"]))
        for prev, nxt in zip(rows, rows[1:]):
            nxt["skills"] = list(prev["skills"])


def _skills_carry_no_signal(doc):
    """Replace every non-target skill grade with the row's own target grade.

    build_hoops_forward.py's --check fails when the gain is indistinguishable from
    shuffling the extras (p >= 0.05). That arm had never been seen to fail. Copying the
    impact column across the other eleven leaves the extras perfectly redundant with the
    baseline, so they can add nothing and the null becomes the same distribution.
    """
    keys = [s.get("key") if isinstance(s, dict) else s for s in doc.get("skills") or []]
    if "impact" not in keys:
        return
    j = keys.index("impact")
    for row in doc.get("grades") or []:
        v = row[j]
        for i in range(len(row)):
            row[i] = v


def _bad_qid(doc):
    # Q41323 is American football (the SPORT). Q19204627 is the OCCUPATION the probe filters
    # on. Swapping the label is the exact confusion the registry exists to catch.
    if "Q19204627" in doc:
        doc["Q19204627"] = "association football"


MUTATIONS = [
    ("hub_freshness/contract",
     ["check_hub_freshness.py", "--check", "--offline"],
     HUB / "tennis.json", _drop_field,
     "a field model.js dereferences is missing — renders as a blank section, not an error"),
    ("hub_freshness/answer_key",
     ["check_hub_freshness.py", "--check", "--offline"],
     HUB / "tennis.json", _flip_answer,
     "a round marks the wrong side correct"),
    ("superlatives/page_scoped",
     ["check_superlatives.py", "--check"],
     HUB / "equities.json", _false_superlative,
     "'the closest call on this board' planted on the WIDEST round"),
    ("artifact_freshness/producer_newer",
     ["check_artifact_freshness.py", "--check"],
     PIPE / "build_vor_draft_value.py", _bump_mtime,
     "a producer script made newer than the artifact it writes",
     "STALE   vor_draft_value.json"),
    ("g1_position/row_mismatch",
     ["probe_g1_position.py", "--check"],
     ROOT / "assets" / "unified.json", lambda d: d["players"].pop(),
     "asset and matrix row counts disagree — a positional join would describe the "
     "wrong player", 2),
    # The first version of this mutation edited data/superlative_registry.json and the
    # guard exited 0 — because check_wikidata_qids.py reads its EXPECT registry from its OWN
    # SOURCE. The test was wrong, not the guard. Mutate what the guard actually reads.
    ("wikidata_qids/wrong_label",
     ["check_wikidata_qids.py", "--check"],
     PIPE / "check_wikidata_qids.py",
     lambda _: _patch_text(PIPE / "check_wikidata_qids.py",
                           '"Q19204627": "American football player"',
                           '"Q19204627": "association football"'),
     "a registered QID relabelled to the wrong entity"),
    ("draft_value_invariants/I4_count_drift",
     ["check_draft_value_invariants.py", "--check"],
     ROOT / "data" / "qb_survivorship_probe.json", _break_invariant,
     "a drafted count bumped by one — the probe and the value table now disagree "
     "about who was drafted, from the same CSV"),
    ("merged_careers/contamination",
     ["check_merged_careers.py", "--check"],
     ROOT / "data" / "direction_axis_hoops.json", _contaminate_axis,
     "a known-merged career (jaren jackson) present in an axis artifact"),
    ("hub_freshness/machine_local_path",
     ["check_hub_freshness.py", "--check", "--offline"],
     HUB / "tennis.json", _absolute_path,
     "a laptop path published as the reader's provenance — the state the site shipped in "
     "for 45 citations, under a heading promising where every number came from"),
    ("cited_fields/phantom_field",
     ["check_cited_fields.py", "--check"],
     HUB / "equities.json", _phantom_cited_field,
     "a page cites a field its source file does not contain, under any reading"),
    ("cited_fields/wrong_value",
     ["check_cited_fields.py", "--check"],
     HUB / "hoops.json", _wrong_cited_value,
     "a published number the cited artifact contradicts — the site's fine print says every "
     "number is recomputable, and this is the first check that tests it"),
    ("equities_forward/carry_forward",
     ["build_equities_forward.py", "--check"],
     Path("C:/Users/jcdav/vector-equities/assets/real_data.json"), _carry_forward,
     "every company-year duplicated from its prior year — the gate that could not fail "
     "under any input, registered as a check for three commits"),
    ("hoops_forward/extras_carry_nothing",
     ["build_hoops_forward.py", "--check"],
     Path("C:/Users/jcdav/vector-hoops/assets/skills.json"), _skills_carry_no_signal,
     "every skill column made a copy of the target — the extras can add nothing, so the "
     "p >= 0.05 arm must fire"),
    ("hub_freshness/unresolvable_root",
     ["check_hub_freshness.py", "--check", "--offline"],
     HUB / "tennis.json", _unknown_repo_root,
     "a citation naming no known repo — must FAIL, because 'I cannot find this' is the "
     "case where skipping is most tempting and most wrong"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    strays = sorted(p.name for p in ROOT.rglob("*.guardbak"))
    if strays:
        print(f"  {len(strays)} stray .guardbak file(s) from a killed run: {strays[:4]}")

    vacuous, ok = [], 0
    for entry in MUTATIONS:
        name, argv, target, mutate, what = entry[:5]
        expect = entry[5] if len(entry) > 5 else 1
        if not target.exists():
            vacuous.append(f"{name}: target {target} missing — cannot test")
            print(f"  SKIP  {name:34} target missing")
            continue
        bak = target.with_suffix(target.suffix + ".guardbak")
        shutil.copy2(target, bak)
        try:
            clean, clean_out = run(argv)
            if mutate is _bump_mtime:
                _bump_mtime(target)
            elif target.suffix == ".py":
                mutate(None)          # text mutation, applies itself
            else:
                patch_json(target, mutate)
            dirty, dirty_out = run(argv)
        finally:
            # copy2 restores BOTH content and mtime, which is what makes the mtime mutation
            # safe to undo — a plain copy would leave every consumer looking stale.
            shutil.copy2(bak, target)
            bak.unlink(missing_ok=True)
        restored, restored_out = run(argv)

        if isinstance(expect, str):
            # The baseline may be legitimately red for an UNRELATED artifact — here
            # stage2_history.json, which only a training run can refresh. Comparing exit
            # codes would make this guard permanently untestable for a reason that has
            # nothing to do with it. So compare WHICH problem appears instead: the planted
            # target must be named when planted and absent when clean. Match the STALE
            # MARKER, not the bare filename: this checker prints every artifact with its
            # status, so "vor_draft_value.json" is present in the clean output too — as a
            # `fresh` row. A substring test that cannot tell fresh from stale proves
            # nothing, which is the same shape as the defects it is here to detect.
            good = (expect not in clean_out) and (expect in dirty_out) and (expect not in restored_out)
        else:
            good = clean == 0 and dirty == expect and restored == 0
        if good:
            ok += 1
        else:
            why = []
            if isinstance(expect, str):
                why.append(f"planted target {expect!r} "
                           f"{'was already flagged clean' if expect in clean_out else 'never appeared'}")
            elif clean != 0:
                why.append(f"was already failing before the mutation (exit {clean})")
            if not isinstance(expect, str) and dirty != expect:
                why.append(f"DID NOT NOTICE the planted defect "
                           f"(exit {dirty}, wanted {expect})")
            if not isinstance(expect, str) and restored != 0:
                why.append(f"still failing after restore (exit {restored}) — tree may be dirty")
            vacuous.append(f"{name}: {'; '.join(why)}")
        print(f"  {'ok  ' if good else 'FAIL'}  {name:34} clean={clean} planted={dirty} "
              f"restored={restored}   {what}")

    print(f"\n{ok}/{len(MUTATIONS)} guards rejected the defect they were shown.")
    print("This proves each guard rejects THE DEFECT IT WAS SHOWN. It does not prove it "
          "catches every defect of that class — the planted ones are the failures that "
          "actually happened in this repo, which is evidence, not coverage.")

    # ---- WHICH REGISTERED CHECKS HAVE NO MUTATION AT ALL -----------------------
    # The sentence above has been true and unquantified since this file was written, and
    # "not complete coverage" is easy to read past. This prints the actual gap.
    #
    # It exists because the gap was hiding a real defect: build_equities_forward.py accepted
    # --check, never read it, and ran as the `equities_forward` gate for three commits — a
    # check that could not fail under any input. Nothing pointed at it, because nothing
    # listed which registered checks had never been shown a defect. An unmutated arm is a
    # claim; an unmutated CHECK is a claim that has never once been tested.
    #
    # Case-insensitive on purpose: an earlier version of this audit used [a-z0-9_]+ and
    # silently missed draft_value_invariants/I4_count_drift, reporting a covered checker as
    # uncovered. A coverage report with its own blind spot is the joke this file is about.
    try:
        reg = (PIPE / "validate.py").read_text(encoding="utf-8")
        registered = dict(re.findall(r'"([A-Za-z0-9_]+)":\s*\(\["([A-Za-z0-9_]+\.py)"', reg))
        mutated = {Path(argv[0]).name for _, argv, *_ in MUTATIONS}
        gaps = sorted(f"{k} ({v})" for k, v in registered.items() if v not in mutated)
        print(f"\ncoverage: {len(registered) - len(gaps)}/{len(registered)} registered "
              f"checks have at least one planted defect.")
        if gaps:
            print(f"  {len(gaps)} with NO mutation — never once seen to fail:")
            for g in gaps:
                print(f"      {g}")
    except OSError as e:
        print(f"\ncoverage: could not read validate.py ({e}) — gap unknown, not zero")
    if vacuous:
        print(f"\n{len(vacuous)} problem(s):")
        for v in vacuous:
            print(f"  {v}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
