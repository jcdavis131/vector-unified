#!/usr/bin/env python3
"""One-time: rewrite the six published data files from laptop paths to repo-relative ones.

Solo personal project, no connection to employer, built with public/free-tier only

The pages were written before portable_paths.py existed, so their citations are absolute:
45 in source_files (rendered to the reader at model.js:216, under the heading "Every number
above came from these files") and 14 more inside insight and headline-stat `source` fields.
The writer now refuses to publish those, but the writer only runs when a page is rebuilt,
and rebuilding all six to fix a path would mean re-verifying every claim on them. This
migrates the data in place instead.

    source_files[]           C:/Users/jcdav/vector-hoops/assets/skills.json
                             ->            vector-hoops/assets/skills.json
    insights[].source        the path portion of a rich citation, rest untouched
    headline_stats[].source  same
    source_hashes{}          RE-KEYED, and this is the part worth being careful about

RE-KEYING THE HASHES IS NOT COSMETIC. check_hub_freshness.py looks up `hashes[f]` by the
source_files entry, and when the key is absent it FALLS BACK TO MTIME. So migrating the
citations while leaving the hash keys absolute would leave every lookup missing, silently
downgrading the content-hash staleness rule back to the mtime rule it was built to replace
— and mtime marks every page stale after any validate.py run, because check_merged_careers
rewrites its report with identical bytes each time. The check would still print, still pass
its own tests, and mean something weaker than it says. Verified after the run: every
source_files entry has a matching hash key.

    python pipeline/migrate_hub_portable_paths.py            # report only
    python pipeline/migrate_hub_portable_paths.py --write    # apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portable_paths import find_absolute, to_portable  # noqa: E402

DATA = Path("C:/Users/jcdav/vector-hub/assets/data")
SLUGS = ("hoops", "gridiron", "pitch", "equities", "tennis", "unified")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    total = 0
    for slug in SLUGS:
        p = DATA / f"{slug}.json"
        if not p.exists():
            print(f"  {slug:9} MISSING")
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        before = len(find_absolute(doc))

        doc["source_files"] = [to_portable(f) for f in (doc.get("source_files") or [])]
        doc["source_hashes"] = {to_portable(k): v
                                for k, v in (doc.get("source_hashes") or {}).items()}
        for key in ("insights", "headline_stats"):
            for item in doc.get(key) or []:
                if isinstance(item, dict) and isinstance(item.get("source"), str):
                    item["source"] = to_portable(item["source"])

        after = find_absolute(doc)
        # The hash map and source_files must still agree on their keys, or freshness
        # silently degrades to mtime for every entry that no longer lines up.
        sf = set(doc.get("source_files") or [])
        hk = set((doc.get("source_hashes") or {}).keys())
        orphan = hk - sf
        unhashed = sf - hk

        total += before - len(after)
        status = f"{before - len(after):>3} rewritten"
        if after:
            status += f", {len(after)} STILL ABSOLUTE {[k for k, _ in after[:2]]}"
        if orphan:
            status += f", {len(orphan)} orphan hash key(s)"
        if unhashed:
            status += f", {len(unhashed)} source(s) with no hash (freshness -> mtime)"
        print(f"  {slug:9} {status}")

        if args.write:
            p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    print(f"\n{total} machine-local path(s) "
          f"{'rewritten' if args.write else 'would be rewritten (dry run, pass --write)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
