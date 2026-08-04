#!/usr/bin/env python3
"""No report may be older than the code that produced it. (7.23)

Solo personal project, no connection to employer, built with public/free-tier only

Phase 7 corrected the definition of nearly every gate in this repo — G2's baseline, G3's
two thresholds, G4's baseline, the position mask, the random-rank formula in two files.
Each correction changes what the numbers MEAN, and a report written before the correction
still sits on disk reading like a current measurement.

That is the same defect twice already caught in prose: `probe_market_layer_coverage.py`
asserting "GRIDIRON HAS NO ATHLETE-LEVEL MARKET DATA" beside live numbers that
contradicted it, and `compare_trajectory_sports.py` emitting a verdict its own data had
refuted. Both were found by reading. This finds them by asking.

    data/ablation_report.json was written 2026-07-10 and its producer was edited
    2026-08-03 — twenty-four days and four gate redefinitions apart. Its G4 column was
    computed on a 4,000-row sample against no baseline. Nothing said so.

METHOD, and its limit stated up front: mtime comparison. If the producing script is newer
than its artifact, the artifact is stale. This is crude — it cannot tell a comment edit
from a formula change, and it will report stale after a docstring fix. That is the right
direction to err: a false "re-run this" costs a re-run, a false "fresh" costs a wrong
number in a report. It also cannot detect staleness from a changed INPUT rather than a
changed producer; those are declared per entry as `also_depends_on`.

REGISTRATION IS MANDATORY, as in check_wikidata_qids.py and validate.py. A generated
artifact that nobody declared cannot be checked, so an unregistered data/*.json under the
declared roots is a FAILURE, not a skip.

    python pipeline/check_artifact_freshness.py
    python pipeline/check_artifact_freshness.py --check   # exit 1 on any stale artifact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

# artifact -> (producer script, [extra files whose change also invalidates it])
PRODUCED_BY: dict[str, tuple[str, list[str]]] = {
    "unified_report.json": ("eval_unified.py", []),
    "stage2_report.json": ("stage2_eval.py", ["eval_unified.py"]),
    "analogy_report.json": ("analogy_panel.py", ["eval_unified.py"]),
    "analogy_triples_report.json": ("analogy_triples_eval.py", []),
    "ablation_report.json": ("ablation.py", ["eval_unified.py"]),
    "ablation_grl_seeds.json": ("ablation.py", ["eval_unified.py"]),
    "ablation_coral_vicreg_seeds.json": ("ablation.py", ["eval_unified.py"]),
    "gate_nonvacuity.json": ("check_gate_nonvacuity.py", ["eval_unified.py"]),
    "matched_draft_value_comparison.json": ("compare_matched_draft_value.py", []),
    "hoops_vor_draft_value.json": ("build_hoops_vor_draft_value.py", []),
    "vor_draft_value.json": ("build_vor_draft_value.py", []),
    "direction_axis_hoops.json": ("build_direction_axis.py",
                                  ["build_hoops_vor_draft_value.py"]),
    "direction_axis_gridiron.json": ("build_direction_axis.py",
                                     ["build_vor_draft_value.py"]),
    "draft_value_curve.json": ("build_draft_value_curve.py", []),
    "qb_survivorship_probe.json": ("probe_qb_survivorship.py", []),
    "merged_careers.json": ("check_merged_careers.py",
                            ["build_hoops_vor_draft_value.py"]),
    "hoops_name_collisions.json": ("probe_hoops_name_collisions.py",
                                   ["build_hoops_vor_draft_value.py"]),
    "gridiron_name_collisions.json": ("probe_gridiron_name_collisions.py",
                                      ["build_vor_draft_value.py"]),
    # Both trajectory axes depend on the collision probes, not only on their own builder:
    # merged_names() subtracts acquitted_names(), which reads those artifacts. Declaring the
    # builder alone would let a re-run of a probe leave the axes silently stale.
    "trajectory_axis.json": ("build_trajectory_axis.py",
                             ["build_hoops_vor_draft_value.py",
                              "probe_hoops_name_collisions.py"]),
    "trajectory_axis_gridiron.json": ("build_trajectory_axis.py",
                                      ["build_vor_draft_value.py",
                                       "probe_gridiron_name_collisions.py",
                                       "export_gridiron_pedigree.py"]),
    "gridiron_pedigree.json": ("export_gridiron_pedigree.py", ["build_vor_draft_value.py"]),
    "pitch_expectation_sources.json": ("probe_pitch_expectation_sources.py", []),
    "tennis_coverage.json": ("acquire_tennis.py", []),
    "tennis_sponsors.json": ("build_tennis_sponsors.py", ["build_tennis_entities.py"]),
    "tennis_entities.json": ("build_tennis_entities.py", ["acquire_tennis.py"]),
    "tennis_ranking_axis.json": ("build_tennis_ranking_axis.py",
                                 ["build_tennis_entities.py"]),
    "tennis_matrix_report.json": ("build_tennis_matrix.py",
                                  ["acquire_tennis.py", "build_tennis_entities.py"]),
    "tennis_archetype_probe.json": ("probe_tennis_archetypes.py",
                                    ["build_tennis_matrix.py"]),
    "tennis_expectation_probe.json": ("probe_tennis_expectation.py",
                                      ["acquire_tennis.py"]),
    "pitch_age_axis.json": ("build_pitch_age_axis.py",
                            ["probe_pitch_expectation_sources.py"]),
}

# The SHIPPED asset, checked the same way and for a sharper reason: 7.29 found
# assets/unified.json carrying `g2_target: 0.433` and `pos_drop: 0.0` in its own metadata,
# where every downstream consumer reads them. A stale report is a wrong number in a file
# nobody reads twice; a stale ASSET is a wrong number in the product.
SHIPPED = {
    "unified.json": ("export_unified_stage2.py", ["eval_unified.py", "train_stage2.py"]),
}

# Artifacts that are INPUTS or hand-authored anchors, not generated reports. Listed so the
# unregistered check stays meaningful instead of being switched off.
# FIVE OF THESE USED TO SIT IN NOT_GENERATED AND ARE GENERATED. Resolved by mapping each
# module-level constant to the file it names and asking which constant is passed to
# .write_text — not by grepping filenames, which credited analogy_triples_eval.py with
# writing analogy_triples.json when it writes analogy_triples_REPORT.json. Declaring a
# generated artifact as an input exempts it from the staleness check entirely, which is the
# one thing this file exists to do.
PRODUCED_BY.update({
    "native_clusters.json": ("archetype_map.py", []),
    "unified_meta.json": ("build_unified_matrix.py", []),
    "stage2_baselines.json": ("train_stage2.py", []),
    "stage2_history.json": ("train_stage2.py", []),
    "trajectory_sport_comparison.json": ("compare_trajectory_sports.py", []),
})

NOT_GENERATED = {
    # HAND-CURATED, not generated. Each entry records a superlative on a live page that was
    # checked against its artifact, with the evidence. It is written by a human decision,
    # so an mtime check would only ever say "the checker is newer than your judgement".
    "superlative_registry.json",
    # 40 hand-written cross-sport pairs (Brady <-> Curry). NO script writes this file; it is
    # irreplaceable if lost, and it was gitignored until now.
    "analogy_triples.json",
    "archetype_map.json", "sector_map.json",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any stale artifact")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    problems: list[str] = []
    rows: list[tuple[str, str, float]] = []

    for name, (producer, extras) in sorted(PRODUCED_BY.items()):
        art = DATA / name
        if not art.exists():
            rows.append((name, "MISSING", 0.0))
            continue
        a_t = art.stat().st_mtime
        newest, newest_src = 0.0, ""
        for src in (producer, *extras):
            sp = PIPE / src
            if not sp.exists():
                problems.append(f"{name}: declared producer {src} does not exist")
                continue
            if sp.stat().st_mtime > newest:
                newest, newest_src = sp.stat().st_mtime, src
        if newest > a_t:
            hours = (newest - a_t) / 3600.0
            rows.append((name, "STALE", hours))
            problems.append(
                f"{name} is {hours:.1f}h older than {newest_src} — re-run "
                f"`python pipeline/{producer}`")
        else:
            rows.append((name, "fresh", 0.0))

    for name, (producer, extras) in sorted(SHIPPED.items()):
        art = ASSETS / name
        if not art.exists():
            rows.append(("assets/" + name, "MISSING", 0.0))
            continue
        a_t = art.stat().st_mtime
        newest, newest_src = 0.0, ""
        for src in (producer, *extras):
            sp = PIPE / src
            if sp.exists() and sp.stat().st_mtime > newest:
                newest, newest_src = sp.stat().st_mtime, src
        if newest > a_t:
            hours = (newest - a_t) / 3600.0
            rows.append(("assets/" + name, "STALE", hours))
            problems.append(
                f"SHIPPED ASSET assets/{name} is {hours:.1f}h older than {newest_src} — "
                f"its metadata is what downstream consumers read. Rebuild with "
                f"`python pipeline/{producer}` (an operator action: it replaces the live "
                f"artifact)")
        else:
            rows.append(("assets/" + name, "fresh", 0.0))

    declared = set(PRODUCED_BY) | NOT_GENERATED
    on_disk = {p.name for p in DATA.glob("*.json")}
    for extra in sorted(on_disk - declared):
        problems.append(
            f"UNREGISTERED artifact data/{extra} — add it to PRODUCED_BY with its producer "
            f"or to NOT_GENERATED if it is an input; an undeclared report cannot be "
            f"checked for staleness")

    width = max(len(n) for n, _s, _h in rows)
    for name, status, hours in rows:
        extra = f"  ({hours:.1f}h behind)" if status == "STALE" else ""
        print(f"  {status:<7} {name:<{width}}{extra}")

    if not problems:
        print(f"\nall {len(rows)} declared artifact(s) are at least as new as their producers.")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
    print("\nmtime is a crude test: it cannot tell a comment edit from a formula change, "
          "and\nit errs toward re-running. That is the cheap direction to be wrong in.")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
