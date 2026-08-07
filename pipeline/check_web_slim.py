"""Gate the shipped web bundle: contract + the honesty rules the page must keep.

Two jobs.

1. CONTRACT — index.html/assets/unified.js read specific fields out of
   assets/unified_slim.json. If the exporter or the upstream build drops one,
   the page renders "—" or silently omits a section, and nobody notices until
   a visitor sees it. Every field the page consumes is asserted here.

2. HONESTY — the wording on that page is constrained by what the artifacts
   actually support, and those constraints are mechanical, so they are checked
   mechanically rather than trusted to survive the next edit:

     - G2 is DEFERRED. The page must never call the space sport-invariant, and
       must quote delta_vs_majority (0.0593), never the flattering
       delta_vs_chance (0.3518) that unified_report.json warns against.
     - G4 is circular: SupCon trains on a hand-authored archetype map. The
       page must ship the no_supcon ablation so the collapse to chance is
       visible, not merely disclosed.
     - Six of twelve declared archetypes have zero members. The page must not
       present a dead id as a live category.
     - The PCA axes carry "interpretation deferred" and must stay unnamed.

Run:  python pipeline/check_web_slim.py
Exit: 0 clean, 1 on any violation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIM = ROOT / "assets" / "unified_slim.json"
EMB = ROOT / "assets" / "unified_emb.f32"
HTML = ROOT / "index.html"
JS = ROOT / "assets" / "unified.js"

# Phrases that overclaim. Matched case-insensitively against the page text.
BANNED = [
    r"sport[- ]invariant",
    r"sport[- ]blind",
    r"proven similarity",
    r"delta[_ ]vs[_ ]chance",
    r"\b0\.3518\b",          # the vs-chance figure specifically
    r"is the (?:nfl|nba) equivalent of",
]

fails: list[str] = []
notes: list[str] = []


def bad(msg: str) -> None:
    fails.append(msg)


def main() -> int:
    for f in (SLIM, EMB, HTML, JS):
        if not f.exists():
            bad(f"missing {f.relative_to(ROOT)}")
    if fails:
        print("\n".join("FAIL " + f for f in fails))
        return 1

    doc = json.loads(SLIM.read_text(encoding="utf-8"))
    meta, rows = doc.get("meta", {}), doc.get("rows", [])
    html = HTML.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    page = html + "\n" + js

    # -- 1. contract ---------------------------------------------------------
    required = [
        "d_emb", "n_players", "row_schema", "sport_counts", "arch_counts",
        "archetypes", "axes", "explained_variance", "g2_sport_acc",
        "g2_majority_baseline", "g2_delta_vs_majority", "g2_status", "ablation",
    ]
    for k in required:
        if meta.get(k) in (None, {}, []):
            bad(f"meta.{k} missing or empty — the page reads it")

    schema = meta.get("row_schema") or []
    for k in ("sport", "name", "season", "arch", "x", "y", "z"):
        if k not in schema:
            bad(f"row_schema lacks {k!r}")
    if rows and len(rows[0]) != len(schema):
        bad(f"row width {len(rows[0])} != schema width {len(schema)}")
    if len(rows) != meta.get("n_players"):
        bad(f"{len(rows)} rows but n_players={meta.get('n_players')}")

    shape = meta.get("emb_shape") or [0, 0]
    want = shape[0] * shape[1] * 4
    if EMB.stat().st_size != want:
        bad(f"unified_emb.f32 is {EMB.stat().st_size} B, expected {want} "
            f"({shape[0]} x {shape[1]} x 4)")

    ab = (meta.get("ablation") or {}).get("configs") or {}
    for cfg in ("full", "no_supcon"):
        if cfg not in ab:
            bad(f"ablation config {cfg!r} missing — the page's toggle needs it")

    # -- 2. honesty ----------------------------------------------------------
    for pat in BANNED:
        m = re.search(pat, page, re.I)
        if m:
            bad(f"page contains overclaiming text {m.group(0)!r} (pattern {pat})")

    status = str(meta.get("g2_status", "")).lower()
    if "defer" not in status:
        notes.append(f"g2_status is now {status!r} — if G2 genuinely passed, the "
                     f"page's DEFERRED wording must be revisited deliberately")
    elif "DEFERRED" not in page:
        bad("G2 is deferred in the data but the page never says DEFERRED")

    # the honest margin must appear, to 4dp, somewhere the page can render
    delta = meta.get("g2_delta_vs_majority")
    if delta is not None and "g2_delta_vs_majority" not in js:
        bad("page never reads g2_delta_vs_majority — the only quotable G2 margin")

    if "no_supcon" not in js:
        bad("page never surfaces the no_supcon ablation; G4's circularity would "
            "be disclosed in prose only, which is the thing this check exists to stop")

    full_g4 = (ab.get("full") or {}).get("G4_hit")
    ns_g4 = (ab.get("no_supcon") or {}).get("G4_hit")
    ns_base = (ab.get("no_supcon") or {}).get("G4_baseline")
    if None not in (full_g4, ns_g4, ns_base):
        if not (ns_g4 < full_g4):
            notes.append(f"no_supcon G4 {ns_g4} is no longer below full {full_g4} — "
                         f"the circularity claim needs re-deriving")
        if abs(ns_g4 - ns_base) > 0.15:
            notes.append(f"no_supcon G4 {ns_g4} is no longer near baseline {ns_base}")

    counts = meta.get("arch_counts") or {}
    declared = {a["id"] for a in (meta.get("archetypes") or [])}
    dead = sorted(i for i in declared if not counts.get(i))
    if dead:
        # the page must filter these out, and must name them as never-assigned
        if "arch_counts" not in js:
            bad("page does not consult arch_counts, so it would list dead archetypes")
        if "dead-arch" not in html:
            bad("page has no slot naming the never-assigned archetype ids")

    if meta.get("axes"):
        for a in meta["axes"]:
            note = str(a.get("note", ""))
            if "deferred" not in note.lower():
                notes.append(f"axis {a.get('pc')} lost its 'interpretation deferred' note")
    if re.search(r"PC1\s*(?:=|is|:)\s*[a-z]{4,}", page):
        bad("page appears to name a PCA axis; axes are explicitly uninterpreted")

    # anti-vacuity: a check that passes on an empty page is worthless
    if len(rows) < 1000:
        bad(f"only {len(rows)} rows — refusing to certify a bundle this small")
    if len(page) < 4000:
        bad("page text implausibly short; this check would pass vacuously")

    # -- report --------------------------------------------------------------
    print(f"rows            {len(rows):,}  ({', '.join(f'{k} {v:,}' for k, v in sorted((meta.get('sport_counts') or {}).items()))})")
    print(f"slim / emb      {SLIM.stat().st_size/1e6:.2f} MB / {EMB.stat().st_size/1e6:.2f} MB")
    print(f"G2              acc {meta.get('g2_sport_acc')} vs majority {meta.get('g2_majority_baseline')} "
          f"= +{meta.get('g2_delta_vs_majority')}  [{meta.get('g2_status')}]")
    print(f"G4              full {full_g4} -> no_supcon {ns_g4} (baseline {ns_base})")
    print(f"archetypes      {len(declared) - len(dead)} live, dead: {dead or 'none'}")
    for n in notes:
        print("NOTE " + n)
    if fails:
        print()
        for f in fails:
            print("FAIL " + f)
        return 1
    print("\nOK: contract satisfied and no overclaiming text on the page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
