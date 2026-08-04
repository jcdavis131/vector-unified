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
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
HUB = Path("C:/Users/jcdav/vector-hub/assets/data")


def run(argv: list[str]) -> int:
    return subprocess.run([sys.executable, str(PIPE / argv[0]), *argv[1:]],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(ROOT)).returncode


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
    ("merged_careers/contamination",
     ["check_merged_careers.py", "--check"],
     ROOT / "data" / "direction_axis_hoops.json", _contaminate_axis,
     "a known-merged career (jaren jackson) present in an axis artifact"),
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
    for name, argv, target, mutate, what in MUTATIONS:
        if not target.exists():
            vacuous.append(f"{name}: target {target} missing — cannot test")
            print(f"  SKIP  {name:34} target missing")
            continue
        bak = target.with_suffix(target.suffix + ".guardbak")
        shutil.copy2(target, bak)
        try:
            clean = run(argv)
            patch_json(target, mutate)
            dirty = run(argv)
        finally:
            shutil.copy2(bak, target)
            bak.unlink(missing_ok=True)
        restored = run(argv)

        good = clean == 0 and dirty == 1 and restored == 0
        if good:
            ok += 1
        else:
            why = []
            if clean != 0:
                why.append(f"was already failing before the mutation (exit {clean})")
            if dirty != 1:
                why.append(f"DID NOT NOTICE the planted defect (exit {dirty})")
            if restored != 0:
                why.append(f"still failing after restore (exit {restored}) — tree may be dirty")
            vacuous.append(f"{name}: {'; '.join(why)}")
        print(f"  {'ok  ' if good else 'FAIL'}  {name:34} clean={clean} planted={dirty} "
              f"restored={restored}   {what}")

    print(f"\n{ok}/{len(MUTATIONS)} guards rejected the defect they were shown.")
    print("This proves each guard rejects THE DEFECT IT WAS SHOWN. It does not prove it "
          "catches every defect of that class — the planted ones are the failures that "
          "actually happened in this repo, which is evidence, not coverage.")
    if vacuous:
        print(f"\n{len(vacuous)} problem(s):")
        for v in vacuous:
            print(f"  {v}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
