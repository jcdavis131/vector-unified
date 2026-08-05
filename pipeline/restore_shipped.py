#!/usr/bin/env python3
"""Restore the shipped stage-2 artifacts after an experiment, from an EXPLICIT manifest.

Solo personal project, no connection to employer, built with public/free-tier only

Running any train_stage2.py arm overwrites artifacts the shipped model is described by.
An experiment that does not put them back leaves the repo describing a throwaway run.

THIS FILE EXISTS BECAUSE INFERRING THE DESTINATION FAILED, and failed silently, which is
the only interesting part. The first restore pass mapped each backup file to a live path
by guessing from its name and extension. Two backups were named for their ROLE rather
than their destination:

    before.json                     a scratch snapshot, no destination at all
    unified_report.json.pre_eval    the backup OF data/unified_report.json

The guesser sent both to invented paths, CREATED two junk files in the repo, reported
"RESTORED" for each, and left data/unified_report.json holding sport_acc 0.6363 -- the
seed-13 lambda-arm value -- while printing a clean summary. The one file whose contents a
reader would actually quote was the one left wrong, and the restore said it had succeeded.

That is this estate's recurring defect exactly: a real value (a restore report) answering
a different question (did I copy 6 files) than the one it appeared to answer (is the
shipped state back).

So: no inference. Every entry states its destination or states that it has none. A backup
file absent from MANIFEST is a FAILURE, not a skip -- the same rule validate.py applies to
unregistered checkers, for the same reason.

WHICH DIR. This flag is required and nothing here ever said where the backup lives. It was
C:/Users/jcdav/AppData/Local/Temp/claude/C--Users-jcdav/<session-id>/scratchpad/g2run/backup
-- a CLAUDE SESSION temp directory, one cleanup away from taking the recovery path for
sport_acc 0.6851 / ckpt b055641c03760624 with it. Copied out on 2026-08-05 and verified
with --verify: all five destinations ok, before.json declared no-destination, no drift.

    C:/Users/jcdav/experiment-rescue-2026-08-05/g2run/backup

See data/EXPERIMENT_DATA_IN_TEMP.md. A required flag whose only valid argument lives in
temp is a restore procedure that works until it does not.

    python pipeline/restore_shipped.py --backup DIR            # restore + verify
    python pipeline/restore_shipped.py --backup DIR --verify   # check only, exit 1 on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# backup filename -> destination relative to ROOT, or None for "no destination".
# None is a DECLARED value, not an omission: it means the file is a snapshot kept for
# reference and must never be copied anywhere.
MANIFEST: dict[str, str | None] = {
    "unified_stage2_best.pt": "pipeline/data/unified_stage2_best.pt",
    "stage2_baselines.json": "data/stage2_baselines.json",
    "stage2_history.json": "data/stage2_history.json",
    "unified_report.json.pre_eval": "data/unified_report.json",
    "gridiron_season_emb.npz": "pipeline/data/gridiron_season_emb.npz",
    "before.json": None,
}

# Values the shipped state must have. Checked after restore so a correct-looking copy of
# the WRONG backup still fails loudly.
EXPECT = {
    "pipeline/data/unified_stage2_best.pt": "b055641c03760624",
    "data/unified_report.json": "d2ee1ca2e45c6cbd",
    "data/stage2_baselines.json": "a7531f8b37271282",
}
EXPECT_SPORT_ACC = 0.6851


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backup", required=True)
    ap.add_argument("--verify", action="store_true", help="report drift, restore nothing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    B = Path(args.backup)
    if not B.is_dir():
        print(f"FAIL: no backup dir {B}", file=sys.stderr)
        return 2

    present = {p.name for p in B.iterdir() if p.is_file()}
    unlisted = sorted(present - set(MANIFEST))
    if unlisted:
        print(f"FAIL: backup holds {len(unlisted)} file(s) absent from MANIFEST: "
              f"{unlisted}. Add a destination or an explicit None; a file whose "
              f"destination is guessed is how the shipped report got left at 0.6363.",
              file=sys.stderr)
        return 2

    drift, restored = [], []
    for name, dest in MANIFEST.items():
        src = B / name
        if not src.exists():
            print(f"  skip     {name:<30} not in this backup")
            continue
        if dest is None:
            print(f"  snapshot {name:<30} no destination, declared")
            continue
        live = ROOT / dest
        pre = h(live)
        if pre == h(src):
            print(f"  ok       {dest:<30} {pre}")
            continue
        drift.append(dest)
        if args.verify:
            print(f"  DRIFT    {dest:<30} {pre} != {h(src)}")
            continue
        live.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, live)
        restored.append(dest)
        print(f"  RESTORED {dest:<30} {pre} -> {h(live)}")

    if args.verify:
        print(f"\n  {len(drift)} drifted" if drift else "\n  no drift")
        return 1 if drift else 0

    bad = [f"{d}: {h(ROOT/d)} != {want}" for d, want in EXPECT.items() if h(ROOT / d) != want]
    rp = ROOT / "data" / "unified_report.json"
    if rp.exists():
        sa = json.loads(rp.read_text(encoding="utf-8"))["G2_sport_invariance"]["sport_acc"]
        if sa != EXPECT_SPORT_ACC:
            bad.append(f"unified_report sport_acc {sa} != {EXPECT_SPORT_ACC}")
    if bad:
        print("\nFAIL: shipped state not restored:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print(f"\n  {len(restored)} restored, shipped state verified "
          f"(sport_acc {EXPECT_SPORT_ACC}, ckpt {EXPECT['pipeline/data/unified_stage2_best.pt']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
