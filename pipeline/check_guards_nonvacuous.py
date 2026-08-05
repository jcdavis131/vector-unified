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


# WHY a registered check still has no planted defect. A bare list of uncovered names reads
# as neglect and cannot be argued with; a recorded obstacle can be checked, disputed, and
# eventually removed. tennis_forward sat on this list as "blocked on the harness" until the
# harness grew a .npz path — the reason is what made it obviously fixable rather than fixed.
UNCOVERED_REASON = {
    "check_gate_nonvacuity.py": (
        "REACHABLE, NOT CHEAP. Its vacuity arm needs a gate that passes on BOTH real and "
        "shuffled data, which means rewriting arch_id inside the 16 MB assets/unified.json "
        "so archetype tracks sport (then within_sport_shuffle changes nothing and the null "
        "survives). The run is 43s, so the harness's clean/planted/restored triple adds "
        "~130s plus three 16 MB copies to every validate.py. Deferred on cost, not "
        "difficulty. Its artifact currently records vacuous_gates: [], so the arm has not "
        "fired on live data either — this is a real gap, not a formality."),
    "check_corrections_landed.py": (
        "CURRENTLY VACUOUS, AND THE MUTATION IS WHAT PROVED IT. A mutation was written "
        "against vector-hoops/pipeline/seed_floor.json — strip the inline [CORRECTED: "
        "...] marker and the wrong claim stands again — and the guard exited 0. The "
        "checker only detects a correction block that quotes its target VERBATIM, and "
        "seed_floor's block paraphrases ('...with NO BACKUP') rather than copying the "
        "sentence. Sweeping the estate: 2 blocks have a quote field and BOTH quote the "
        "correction's own narration, not the target text. The single block that ever "
        "quoted verbatim was gate_nonvacuity.json's prose_says, and 2d602a2 regenerated "
        "it away when the fix moved into the generator. So the guard is green over a "
        "class with no members. Downgraded to report-only in validate.py rather than "
        "left blocking; a gate that cannot fail is worse than no gate, because it reads "
        "as coverage. The mutation is removed rather than left permanently red — it "
        "would report the checker as broken when what is true is that its detectable "
        "class is empty."),
    "check_internal_prose.py": (
        "CANNOT FAIL BY CONSTRUCTION, deliberately. It is registered in validate.py "
        "WITHOUT --check, so it always exits 0 — its measured precision is 2 of 23 "
        "(~9%) and a blocking gate at that rate trains its reader to ignore it. A "
        "mutation test needs a red state to plant, and there is none. This is a design "
        "choice with a cost: nothing verifies that the checker still detects anything, "
        "so if its regex broke it would report 0 findings and look identical to a clean "
        "estate. The honest mitigation is that its two real finds are recorded in "
        "internal_prose_audit.json with their file and field, so a reader can confirm "
        "they are still being found."),
    "check_guards_nonvacuous.py": (
        "CIRCULAR. This file IS the mutation harness; planting a defect in it means asking "
        "it to detect its own vacuity while running as the thing under test. Its own "
        "failure paths are exercised in practice instead — every guard that has ever "
        "regressed showed up as a FAIL row here, and its bugs this session (a lowercase-only "
        "regex, a missing `import re`) surfaced by running it, not by mutating it."),
}


def _patch_text(path: Path, old: str, new: str) -> None:
    t = path.read_text(encoding="utf-8")
    assert t.count(old) == 1, f"{path.name}: expected 1 occurrence of {old!r}"
    path.write_text(t.replace(old, new), encoding="utf-8")


def patch_json(path: Path, mutate) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")


def patch_npz(path: Path, mutate) -> None:
    """Mutate a .npz in place.

    ADDED TO CLOSE A GAP THIS FILE HAD PUBLISHED ABOUT ITSELF. The coverage report listed
    tennis_forward as never once seen to fail, with "blocked on the harness" as the reason:
    its input is tennis_matrix.npz and the mutation path handled JSON and .py only. That is
    a fixable limitation, not a blocker, and leaving it recorded as one would have turned an
    honest gap report into a permanent excuse.

    The arrays are handed to the mutation as a plain dict so mutations stay ordinary
    functions. Restore is unaffected by what this writes: the harness backs up with
    shutil.copy2 and restores byte-for-byte, so compression or dtype drift introduced here
    cannot survive the run.
    """
    import numpy as np
    with np.load(path, allow_pickle=True) as a:
        arrays = {k: a[k] for k in a.files}
    mutate(arrays)
    np.savez(path, **arrays)


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


def _strip_correction_marker(doc):
    """RETAINED BUT NOT WIRED IN. This is the mutation that proved
    check_corrections_landed.py vacuous — it plants exactly the defect that guard names,
    and the guard exited 0. It is kept rather than deleted because it is the test to
    re-enable the moment the checker can see a paraphrased correction; deleting it would
    lose the one worked example of the class it cannot detect. See UNCOVERED_REASON
    ["check_corrections_landed.py"] for the full finding.

    Remove the inline [CORRECTED: ...] pointer, leaving the wrong claim standing.

    This is the real defect, not an approximation of it. seed_floor.json's
    REPO_STATE_WARNING.also_destroyed asserted "there is no backup" while
    CORRECTION_embedding_v3_backups_exist in the same file said eight exist. The fix was
    an inline marker at the point of the wrong claim; deleting that marker restores the
    exact state the guard is here to reject.
    """
    w = doc["REPO_STATE_WARNING"]
    t = w["also_destroyed"]
    i, j = t.find(" [CORRECTED:"), t.find("]", t.find(" [CORRECTED:"))
    assert i >= 0 and j > i, "marker not found — mutation would be a no-op"
    w["also_destroyed"] = (t[:i] + " and there is no backup" + t[j + 1:])


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


def _tennis_extras_are_nothing(arrays):
    """Zero every feature except ENTERING_RANK_LOG, in the values AND the mask.

    build_tennis_forward.py's --check fails when the gain is indistinguishable from
    shuffling the extra columns (p >= 0.05). That arm had never been seen to fail.

    ZEROING RATHER THAN COPYING THE RANK COLUMN, and the difference decides whether this
    mutation works at all. Copying rank into the extras makes them collinear with the
    baseline, so the real gain is ~0 — but the NULL shuffles those copies, which breaks the
    collinearity and makes the null gain NEGATIVE. p = P(null >= real) would then go to ~0
    and the guard would PASS on a mutation that destroyed the signal. Constant columns
    survive shuffling unchanged, so null gain == real gain exactly and p goes to 1.0.

    The mask is zeroed too. Leaving it at 1 would keep 15 columns of real observed/missing
    structure in the design matrix, which is information, and the point is to leave none.
    """
    feats = [str(f) for f in arrays["features"]]
    if "ENTERING_RANK_LOG" not in feats:
        return
    j = feats.index("ENTERING_RANK_LOG")
    for key in ("X", "M"):
        A = arrays[key]
        keep = A[:, j].copy()
        A[:, :] = 0
        A[:, j] = keep
        arrays[key] = A


def _gridiron_ppr_carries_forward(doc):
    """Copy each player-season's PPR onto his next season.

    build_gridiron_forward.py's --check exists because build_equities_forward.py shipped
    with a --check that had no body and ran as a registered gate that could not fail. The
    arm it guards is carry-forward: pooled persistence of 0.77-0.80 is exactly what
    duplicated data looks like, and every number in the report would still print as a
    finding about football.
    """
    by_name: dict[str, list] = {}
    for p in doc.get("players") or []:
        if isinstance(p.get("ppg"), dict) and p["ppg"].get("ppr") is not None:
            by_name.setdefault(p["name"], []).append(p)
    for rows in by_name.values():
        rows.sort(key=lambda x: int(x["season"]))
        for prev, nxt in zip(rows, rows[1:]):
            nxt["ppg"]["ppr"] = prev["ppg"]["ppr"]


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
    ("tennis_forward/extras_carry_nothing",
     ["build_tennis_forward.py", "--check"],
     ROOT / "pipeline" / "data" / "tennis_matrix.npz", _tennis_extras_are_nothing,
     "every feature but rank zeroed in values and mask — the p >= 0.05 arm must fire, and "
     "this closes the coverage gap the suite published about itself"),
    ("gridiron_forward/carry_forward",
     ["build_gridiron_forward.py", "--check"],
     Path("C:/Users/jcdav/vector-gridiron/assets/vectors.json"),
     _gridiron_ppr_carries_forward,
     "every player-season's PPR copied onto his next — persistence of 0.77-0.80 would be "
     "measuring duplication and would still print as a finding about football"),
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
    # Guards whose clean baseline is already red, so the mutation proves nothing.
    # Distinct from `vacuous` — see the comment at the verdict block below.
    untestable: list[str] = []
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
            elif target.suffix == ".npz":
                patch_npz(target, mutate)
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
        # A MUTATION TEST ON A RED BASELINE IS UNTESTABLE, NOT FAILED, and the difference
        # matters. If the guard already exits non-zero with NOTHING planted, then
        # planted=1 tells you nothing — the guard would have said 1 either way, and this
        # harness cannot distinguish "noticed the defect" from "was already unhappy".
        # Counting that as a mutation-test FAILURE reports the guard as vacuous when what
        # is actually true is that the experiment could not run.
        #
        # This is not hypothetical: all four hub_freshness mutations sat at clean=1 and
        # were reported as FAIL for weeks. The cause was that hub_freshness compares the
        # published pages against the artifacts they cite, and regenerating ANY cited
        # artifact turns it red until the page extractor is re-run — which is an operator
        # action, because it replaces live site content. So the harness was reporting four
        # guard failures that were really one un-run operator step.
        #
        # UNTESTABLE is surfaced with its own count and does NOT fail the check. The
        # underlying red is not hidden: validate.py runs hub_freshness itself, as its own
        # entry, and that is where a genuinely broken guard shows up.
        untestable_here = (not isinstance(expect, str)) and clean != 0
        if untestable_here:
            untestable.append(
                f"{name}: clean baseline already exits {clean}, so planted={dirty} is "
                f"uninformative — the guard would report non-zero either way. Fix the "
                f"baseline (see validate.py's own entry for this check), then re-run.")
        elif good:
            ok += 1
        else:
            why = []
            if isinstance(expect, str):
                why.append(f"planted target {expect!r} "
                           f"{'was already flagged clean' if expect in clean_out else 'never appeared'}")
            if not isinstance(expect, str) and dirty != expect:
                why.append(f"DID NOT NOTICE the planted defect "
                           f"(exit {dirty}, wanted {expect})")
            if not isinstance(expect, str) and restored != 0:
                why.append(f"still failing after restore (exit {restored}) — tree may be dirty")
            vacuous.append(f"{name}: {'; '.join(why)}")
        verdict = "SKIP" if untestable_here else ("ok  " if good else "FAIL")
        print(f"  {verdict}  {name:34} clean={clean} planted={dirty} "
              f"restored={restored}   {what}")

    # The denominator must not silently shrink. 12/16 with four SKIPs is a different
    # statement from 12/12, and printing the latter would be the same unearned green
    # this whole file exists to prevent.
    print(f"\n{ok}/{len(MUTATIONS)} guards rejected the defect they were shown"
          + (f"; {len(untestable)} UNTESTABLE — clean baseline already red, so the "
             f"mutation proves nothing." if untestable else "."))
    for u in untestable:
        print(f"  SKIP {u}")
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
                script = g.split("(")[-1].rstrip(")")
                why = UNCOVERED_REASON.get(script, "no reason recorded — that is itself a "
                                                   "gap, since an unexplained gap reads as "
                                                   "neglect and cannot be argued with")
                print(f"      {g}\n          {why}")
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
