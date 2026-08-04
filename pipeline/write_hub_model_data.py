#!/usr/bin/env python3
"""Write the per-model page data for dumbmodel.com — and REFUSE to write a bad one.

Solo personal project, no connection to employer, built with public/free-tier only

The six /models/<slug>.html pages on dumbmodel.com render entirely from
vector-hub/assets/data/<slug>.json. Nothing is hardcoded in that markup, so this script is
the only thing standing between an extraction and a live public page.

IT IS A GATE, NOT A COPY. Each spec arrives with an adversarial verifier's verdict attached,
and this refuses to write anything the verifier could not confirm:

    CLEAN         write
    UNVERIFIABLE  refuse — the verifier could not reach the source
    FABRICATED    refuse — a number or name did not match its artifact

Refusing costs a re-run. Writing costs a false number on a live website that claims every
figure is recomputable from public data. Those are not symmetric and the asymmetry is the
whole reason this file exists rather than a `json.dump` at the end of the workflow.

STRUCTURAL CHECKS TOO, because a verifier verdict is a judgement and these are arithmetic:

  * `answer` must actually name the side with the larger `value`. An extraction that
    mislabels the winner turns the game into a liar in the most direct way available.
  * a round whose two values are EQUAL has no correct answer and is dropped.
  * `slug` must be one of the six, normalised — the extractor returned "vector-gridiron"
    where the page shell expects "gridiron", and a mismatch there is a silent 404 on the
    data fetch that renders as the empty state.
  * every `source_files` entry must exist on disk. A cited file that is not there means the
    citation was reconstructed rather than read.

    python pipeline/write_hub_model_data.py --spec <path-to-json>
    python pipeline/write_hub_model_data.py --spec <path> --allow-unverified   # explicit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HUB = Path("C:/Users/jcdav/vector-hub")
OUTDIR = HUB / "assets" / "data"
SLUGS = {"hoops", "gridiron", "pitch", "equities", "tennis", "unified"}

REQUIRED = ("slug", "name", "tagline", "dims", "entity_count",
            "source_files", "headline_stats", "insights", "game")


def normalise_slug(raw: str) -> str | None:
    s = (raw or "").strip().lower().replace("vector-", "").replace("vector_", "")
    s = s.replace(" ", "").replace("vector", "")
    return s if s in SLUGS else None


def check(spec: dict, verdict: dict | None, allow_unverified: bool,
          drops: list[tuple[int, str]] | None = None,
          clear_flag: bool = False,
          insight_drops: list[tuple[int, str]] | None = None) -> tuple[list[str], dict]:
    """Returns (blocking problems, cleaned spec). Never mutates the input."""
    problems: list[str] = []
    spec = json.loads(json.dumps(spec))  # deep copy
    drops = drops or []
    insight_drops = insight_drops or []

    if insight_drops:
        ins = spec.get("insights") or []
        for idx, reason in sorted(insight_drops, key=lambda t: -t[0]):
            if 0 <= idx < len(ins):
                gone = ins.pop(idx)
                spec.setdefault("_insight_notes", []).append(
                    f"insight {idx} REMOVED ({gone.get('title')!r}): {reason}")
            else:
                problems.append(f"--drop-insight index {idx} out of range "
                                f"(0..{len(ins)-1})")
        if len(ins) < 3:
            problems.append(f"only {len(ins)} insights left after removals (need >=3)")

    # Operator-directed removal of a specific flagged round. Applied BEFORE the fabrication
    # check so that clearing the flag is only possible once the offending round is actually
    # gone — the flag can never be waved away on its own.
    if drops:
        rs = (spec.get("game") or {}).get("rounds") or []
        removed = []
        for idx, reason in sorted(drops, key=lambda t: -t[0]):
            if 0 <= idx < len(rs):
                gone = rs.pop(idx)
                removed.append(f"round {idx} REMOVED ({gone.get('a',{}).get('name')} vs "
                               f"{gone.get('b',{}).get('name')}): {reason}")
            else:
                problems.append(f"--drop-round index {idx} out of range (0..{len(rs)-1})")
        spec.setdefault("_round_notes", []).extend(removed)
    if clear_flag and not (drops or insight_drops):
        problems.append("--clear-fabrication-flag passed with no --drop-round or "
                        "--drop-insight for that slug: the fabrication is still in the "
                        "spec")

    for k in REQUIRED:
        if k not in spec or spec[k] in (None, "", [], {}):
            problems.append(f"missing required field {k!r}")

    slug = normalise_slug(spec.get("slug", ""))
    if not slug:
        problems.append(f"slug {spec.get('slug')!r} is not one of {sorted(SLUGS)}")
    else:
        spec["slug"] = slug

    v = (verdict or {}).get("verdict")
    if verdict is None:
        problems.append("no verifier verdict attached — refusing on principle, since the "
                        "verifier is the only thing distinguishing a read number from an "
                        "invented one")
    elif v == "FABRICATED":
        if clear_flag and (drops or insight_drops):
            spec["_verification"] = (
                "PARTIALLY VERIFIED — the adversarial verifier confirmed every other "
                "figure and flagged specific round(s), which were REMOVED rather than "
                "corrected. See _round_notes.")
        else:
            problems.append(f"verifier says FABRICATED: "
                            f"{'; '.join((verdict.get('details') or [])[:4])}")
    elif v == "UNVERIFIABLE":
        if allow_unverified:
            spec["_verification"] = "UNVERIFIABLE — written under --allow-unverified"
        else:
            problems.append("verifier says UNVERIFIABLE (pass --allow-unverified to "
                            "override, and say so on the page if you do)")
    elif v == "CLEAN":
        spec["_verification"] = "CLEAN — adversarially verified against source artifacts"

    # ---- structural: the game must not lie about its own answer key -------------
    rounds = ((spec.get("game") or {}).get("rounds") or [])
    kept, dropped = [], []
    for i, r in enumerate(rounds):
        try:
            av, bv = float(r["a"]["value"]), float(r["b"]["value"])
        except (KeyError, TypeError, ValueError):
            dropped.append(f"round {i}: unreadable values")
            continue
        if av == bv:
            dropped.append(f"round {i}: values are equal ({av}) — no correct answer exists")
            continue
        truth = "a" if av > bv else "b"
        if r.get("answer") != truth:
            # Corrected rather than dropped: the VALUES are the artifact's, and the label is
            # the extractor's. Trust the data over the annotation, and record that it moved.
            dropped.append(f"round {i}: answer said {r.get('answer')!r}, values say "
                           f"{truth!r} — corrected to follow the data")
            r = dict(r, answer=truth)
        kept.append(r)
    if len(kept) < 6:
        problems.append(f"only {len(kept)} usable game rounds after structural checks "
                        f"(need >=6)")
    spec.setdefault("game", {})["rounds"] = kept
    if dropped:
        # EXTEND, not assign. Assignment here silently discarded the --drop-round removal
        # records written above, so an operator-directed removal would have vanished from
        # the artifact — defeating the one property that makes the escape hatch acceptable,
        # that it is never applied silently.
        spec.setdefault("_round_notes", []).extend(dropped)

    # ---- every cited file must exist -------------------------------------------
    missing = [f for f in (spec.get("source_files") or []) if not Path(f).exists()]
    if missing:
        problems.append(f"cited source file(s) do not exist: {missing}")

    return problems, spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True,
                    help="JSON file: either one spec, or {spec:..., verdict:...}, "
                         "or a list of those")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--drop-round", action="append", default=[], metavar="SLUG:IDX:REASON",
                    help="Drop one round the verifier flagged, e.g. "
                         "gridiron:6:'reveal contradicted by projections.json'. Use when a "
                         "verifier finds a bad round among otherwise-verified ones — "
                         "discarding ten confirmed rounds over one bad reveal string is its "
                         "own kind of dishonesty. The drop and its reason are written into "
                         "the artifact, never applied silently.")
    ap.add_argument("--drop-insight", action="append", default=[], metavar="SLUG:IDX:REASON",
                    help="Drop one insight the verifier flagged. Symmetric with "
                         "--drop-round, and preferred over hand-editing the prose: patching "
                         "a sentence by hand turns a generated-and-verified artifact into a "
                         "partly-hand-written one, and the guarantee is only worth what its "
                         "weakest field is worth.")
    ap.add_argument("--clear-fabrication-flag", action="append", default=[], metavar="SLUG",
                    help="Only valid alongside --drop-round for the SAME slug: the flagged "
                         "round is gone, so the spec no longer contains the fabrication. "
                         "Refused if nothing was dropped for that slug.")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    raw = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]

    drops_by_slug: dict[str, list[tuple[int, str]]] = {}
    for spec_str in args.drop_round:
        parts = spec_str.split(":", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            print(f"  bad --drop-round {spec_str!r} — want SLUG:IDX:REASON")
            return 2
        drops_by_slug.setdefault(parts[0], []).append((int(parts[1]), parts[2]))
    idrops_by_slug: dict[str, list[tuple[int, str]]] = {}
    for spec_str in args.drop_insight:
        parts = spec_str.split(":", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            print(f"  bad --drop-insight {spec_str!r} — want SLUG:IDX:REASON")
            return 2
        idrops_by_slug.setdefault(parts[0], []).append((int(parts[1]), parts[2]))
    clear = set(args.clear_fabrication_flag)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    wrote, refused = 0, 0
    for item in items:
        spec = item.get("spec", item) if isinstance(item, dict) else item
        verdict = item.get("verdict") if isinstance(item, dict) else None
        s = normalise_slug(spec.get("slug", "")) or ""
        problems, cleaned = check(spec, verdict, args.allow_unverified,
                                  drops_by_slug.get(s), s in clear,
                                  idrops_by_slug.get(s))
        label = cleaned.get("slug") or spec.get("slug") or "?"
        if problems:
            refused += 1
            print(f"  REFUSED {label}")
            for p in problems:
                print(f"      {p}")
            continue
        out = OUTDIR / f"{cleaned['slug']}.json"
        out.write_text(json.dumps(cleaned, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        notes = (cleaned.get("_round_notes") or []) + (cleaned.get("_insight_notes") or [])
        print(f"  wrote   {cleaned['slug']:9} "
              f"{len(cleaned['insights'])} insights, "
              f"{len(cleaned['game']['rounds'])} rounds"
              + (f", {len(notes)} round(s) corrected/dropped" if notes else ""))
        for n in notes:
            print(f"      note: {n}")
        wrote += 1

    print(f"\n{wrote} written, {refused} refused")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
