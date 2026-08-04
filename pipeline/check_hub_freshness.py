#!/usr/bin/env python3
"""The published site must not drift from the artifacts it quotes.

Solo personal project, no connection to employer, built with public/free-tier only

WHY THIS EXISTS, and it is not hypothetical. dumbmodel.com published `48-d embedding` for
Vector Hoops while `vector-hoops/assets/mtnn_meta.json` said `dim: 64`. The model repo had
already caught its own error — `mtnn_arch.json` carries a `_stale` block reading
"dEmb = 48 -> 64, verified from assets/mtnn_embeddings.f32 = 12966*64*4 = 3319296 bytes" —
and the website never heard. Nine numbers on the landing page matched no artifact on disk:

    hoops "48-d"                    mtnn_meta.json dim = 64, live.json n_towers = 18
    equities "96-d MTNN"            real_data.json dim = 64
    equities "3,371 tickers"        real_data.json tickers = 500
    equities "33,710 filing-years"  real_data.json rows = 4,831
    equities "248 feats, 29 towers" features = 118, 17 towers
    pitch "PCA(3), no neural net"   superseded by a 24-d MTNN

The site's own fine print says "Every number is recomputable from public sources." That is
the claim this file defends, and nothing was defending it.

THREE INDEPENDENT CHECKS, because they fail in different ways:

  STALENESS   each assets/data/<slug>.json is older than a source_file it cites. mtime
              only, same crude method and same honest limit as check_artifact_freshness.py:
              it cannot tell a comment edit from a formula change, and errs toward re-run.

  CONTRACT    every field assets/model.js dereferences must exist in every data file. The
              pages render entirely client-side, so a renamed field is not a crash in CI —
              it is a blank section on a live page that nobody sees fail.

  DEPLOY DRIFT  the committed file and the LIVE file must be byte-identical. Passing the
              first two and still serving something else is the failure mode that hides
              longest: the repo looks right, so nobody re-reads the page. Skipped with
              --offline, and reported as skipped rather than as a pass.

    python pipeline/check_hub_freshness.py
    python pipeline/check_hub_freshness.py --check     # exit 1 on any problem
    python pipeline/check_hub_freshness.py --offline   # skip the live comparison
"""

from __future__ import annotations

import argparse
import hashlib

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUB = Path("C:/Users/jcdav/vector-hub")
DATA = HUB / "assets" / "data"
RENDERER = HUB / "assets" / "model.js"
LIVE = "https://dumbmodel.com/assets/data/{slug}.json"
SLUGS = ("hoops", "gridiron", "pitch", "equities", "tennis", "unified")

# Fields the renderer treats as optional (it substitutes "" rather than failing).
OPTIONAL = {"sub", "caveat", "explainer"}
# `d` is reused in model.js for a Date inside dailySeed(), so d.getUTCFullYear() looks like
# a data field to a regex. Excluded by name rather than by making the regex cleverer.
NOT_A_FIELD = {"get", "ok", "json", "status", "then", "catch"}


def renderer_fields() -> dict[str, list[str]]:
    js = RENDERER.read_text(encoding="utf-8")

    def grab(var: str) -> list[str]:
        return sorted(set(re.findall(rf"\b{var}\.([a-z_]+)", js)) - NOT_A_FIELD)

    return {"top": grab("d"), "game": grab("g"), "round": grab("r"), "side": grab("side")}


def fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vector-unified/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any problem")
    ap.add_argument("--offline", action="store_true", help="skip the live comparison")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    problems: list[str] = []
    if not DATA.exists():
        print(f"missing {DATA} — run write_hub_model_data.py first")
        return 2

    fields = renderer_fields()
    present = {p.stem for p in DATA.glob("*.json")}
    for missing in sorted(set(SLUGS) - present):
        problems.append(f"{missing}: no data file, but /models/{missing}.html ships and "
                        f"will render its empty state")

    for slug in SLUGS:
        p = DATA / f"{slug}.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        row = [f"  {slug:9}"]

        # ---- staleness ---------------------------------------------------------
        # CONTENT, not mtime, when the writer recorded hashes. mtime alone marked every
        # page stale after any validate.py run, because check_merged_careers.py rewrites
        # its report with identical bytes each time. Falls back to mtime when a data file
        # predates the hash field, and SAYS which rule it used rather than implying the
        # stronger one.
        hashes = doc.get("source_hashes") or {}
        changed = []
        for f in doc.get("source_files") or []:
            q = Path(f)
            if not q.exists():
                problems.append(f"{slug}: cited source {f} no longer exists")
                continue
            if f in hashes:
                now = hashlib.sha256(q.read_bytes()).hexdigest()[:16]
                if now != hashes[f]:
                    changed.append(Path(f).name)
            elif q.stat().st_mtime > p.stat().st_mtime:
                changed.append(Path(f).name + " (mtime, no hash recorded)")
        if changed:
            row.append(f"STALE vs {', '.join(changed[:2])}")
            problems.append(f"{slug}: cited artifact(s) changed since this page was "
                            f"written: {changed} — re-run the extractor")
        else:
            row.append("fresh" + ("" if hashes else "*mtime"))

        # ---- renderer contract -------------------------------------------------
        g = doc.get("game") or {}
        miss = [k for k in fields["top"] if k not in doc and k not in OPTIONAL]
        miss += [f"game.{k}" for k in fields["game"] if k not in g and k not in OPTIONAL]
        seen_r, seen_s = set(), set()
        for rr in g.get("rounds") or []:
            seen_r |= {k for k in fields["round"] if k not in rr}
            for side in (rr.get("a") or {}, rr.get("b") or {}):
                seen_s |= {k for k in fields["side"] if k not in side}
        miss += [f"round.{k}" for k in sorted(seen_r - OPTIONAL)]
        miss += [f"side.{k}" for k in sorted(seen_s - OPTIONAL)]
        # INSIGHT AND STAT SUB-FIELDS. The regex above only sees `d.`/`g.`/`r.`/`side.`
        # dereferences, so it never noticed that renderInsights reads i.title/i.body/i.source
        # and renderStats reads s.value/s.label/s.source. check_guards_nonvacuous.py planted
        # a deleted insights[0].body and this check returned 0 — the exact blank-section
        # failure it exists to prevent, invisible to it. Found only because the guard suite
        # asked whether the guard could fail at all.
        for j, ins in enumerate(doc.get("insights") or []):
            miss += [f"insights[{j}].{k}" for k in ("title", "body", "source")
                     if not ins.get(k)]
        for j, st in enumerate(doc.get("headline_stats") or []):
            miss += [f"headline_stats[{j}].{k}" for k in ("value", "label", "source")
                     if not st.get(k)]
        if miss:
            row.append(f"CONTRACT {miss}")
            problems.append(f"{slug}: model.js reads {miss} and the data lacks it — that "
                            f"renders as a blank section on a live page, not an error")
        else:
            row.append("contract-ok")

        # ---- the answer key must follow the values -----------------------------
        wrong = [i for i, rr in enumerate(g.get("rounds") or [])
                 if (rr.get("answer") == "a") !=
                 (float(rr["a"]["value"]) > float(rr["b"]["value"]))]
        if wrong:
            row.append(f"ANSWERKEY {wrong}")
            problems.append(f"{slug}: rounds {wrong} mark the wrong side correct")

        # ---- deploy drift ------------------------------------------------------
        if args.offline:
            row.append("live-skipped")
        else:
            body = fetch(LIVE.format(slug=slug))
            if body is None:
                row.append("live-UNREACHABLE")
                problems.append(f"{slug}: could not fetch {LIVE.format(slug=slug)} — "
                                f"unreachable is not the same as matching")
            else:
                # COMPARE THE PARSED DATA, NOT THE BYTES. The first version hashed raw bytes
                # and reported DRIFT on all six: Python's write_text translates \n to \r\n on
                # Windows, so the local file carries 218 CRLFs and the served file carries
                # none. Identical content, different hash. A checker that fires on every run
                # trains its reader to ignore it, which is worse than not having one — and
                # line endings are not what "the site disagrees with the repo" should mean.
                try:
                    same = json.loads(body) == json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    same = False
                    problems.append(f"{slug}: the live response is not valid JSON")
                if not same:
                    row.append("DRIFT")
                    problems.append(f"{slug}: the live DATA differs from the committed file "
                                    f"— either the deploy has not landed or the repo moved "
                                    f"on without one")
                else:
                    row.append("live-match")
        print("  ".join(row))

    # ---- ADVISORY: artifacts a page might want and does not cite --------------
    # Staleness only tracks files a page ALREADY cites, so a brand-new artifact about a
    # sport the page covers is invisible to it — tennis_forward_report.json existed for a
    # full commit before anything noticed the tennis card had no prediction in it.
    #
    # REPORTED, NOT FAILED, and the distinction is deliberate. An uncited artifact is an
    # OMISSION; an unverified superlative is a potential LIE, which is why that one fails
    # the build and this does not. A page cannot cite everything, and this heuristic cannot
    # know intent — it only knows the filename starts with the slug.
    #
    # Worth having because the signal is CLEAN: measured across all six pages it returns 4
    # hits and no junk. The prose-number sweep tried earlier returned dozens, nearly all
    # legitimate citations, and was dropped for that reason.
    uncited: list[str] = []
    for slug in SLUGS:
        f = DATA / f"{slug}.json"
        if not f.exists():
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        cited = {Path(x).name for x in (doc.get("source_files") or [])}
        for a in sorted((ROOT / "data").glob(f"{slug}_*.json")):
            if a.name not in cited:
                uncited.append(f"{slug}: does not cite data/{a.name}")
    if uncited:
        print(f"\n  advisory — {len(uncited)} artifact(s) a page may want and does not "
              f"cite (not a failure):")
        for u in uncited:
            print(f"      {u}")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p_ in problems:
            print(f"  {p_}")
        print("\nThe site's fine print says every number is recomputable from public "
              "sources. That is the claim these checks defend.")
        return 1 if args.check else 0
    suffix = " (live comparison skipped)" if args.offline else ""
    print(f"all {len(SLUGS)} model pages match their artifacts and the renderer{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
