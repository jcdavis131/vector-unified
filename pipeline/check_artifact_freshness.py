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
import hashlib
import json
import re
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
    "tennis_forward_report.json": ("build_tennis_forward.py", ["build_tennis_matrix.py"]),
    "equities_forward_report.json": ("build_equities_forward.py",
                                    ["build_tennis_forward.py"]),
    "hoops_forward_report.json": ("build_hoops_forward.py",
                                 ["build_tennis_forward.py", "build_hoops_vor_draft_value.py"]),
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
    # --- the sponsor / bridge thread ------------------------------------------
    # Producers determined by WHERE THE FILE IS WRITTEN, not by grepping the name:
    # a name grep returns readers too, and it named build_bridge_index.py as the
    # producer of venue_edges.json, which only reads it.
    "venue_edges.json": ("build_venue_edges.py", ["acquire_venue_sponsors.py"]),
    "bridge_index.json": ("build_bridge_index.py", ["build_venue_edges.py"]),
    "promotion_gate_audit.json": ("audit_promotion_gates.py", []),
    # --- checkers that write their own report ---------------------------------
    "internal_prose_audit.json": ("check_internal_prose.py", []),
    "corrections_landed_audit.json": ("check_corrections_landed.py", []),
    # --- tennis probes --------------------------------------------------------
    "tennis_calendar_identity.json": ("probe_tennis_calendar_identity.py", []),
    "tennis_candidate_features.json": ("probe_tennis_candidate_features.py", []),
    "tennis_feature_identity.json": ("probe_tennis_feature_identity.py", []),
    "tennis_forward_enriched.json": ("probe_tennis_forward_enriched.py", []),
    "tennis_metric_probe.json": ("probe_tennis_metric.py", []),
    "tennis_metric_probe_enriched.json": ("probe_tennis_metric.py", []),
    "tennis_retrieval_probe.json": ("probe_tennis_retrieval.py", []),
    # train_tennis_mtnn.py, not sweep_tennis_hparams.py. The sweep only READS this
    # path to collect each trial's result; the trainer is what writes it.
    "tennis_mtnn_report.json": ("train_tennis_mtnn.py", []),
    # --- pre-existing gaps, not from this session -----------------------------
    "sponsor_synchrony.json": ("build_sponsor_synchrony.py", ["build_tennis_sponsors.py"]),
    "gridiron_forward_report.json": ("build_gridiron_forward.py", []),
})

PRODUCED_BY.update({
    "native_clusters.json": ("archetype_map.py", []),
    "unified_meta.json": ("build_unified_matrix.py", []),
    # build_stage2_baselines.py, NOT train_stage2.py. Both write this file and both call
    # the same knn5_acc, but only the standalone one can be run without launching a
    # Stage 2 training run — so it is the producer a staleness message should name.
    "stage2_baselines.json": ("build_stage2_baselines.py", ["eval_unified.py"]),
    "g1_position_probe.json": ("probe_g1_position.py",
                               ["eval_unified.py", "export_unified_stage2.py"]),
    "stage2_history.json": ("train_stage2.py", []),
    "trajectory_sport_comparison.json": ("compare_trajectory_sports.py", []),
})

NOT_GENERATED = {
    # COPIED OUT OF A ONE-OFF HARNESS, which is a different thing from hand-curated
    # and is called out separately rather than lumped in with the human-judgement
    # entries below. No pipeline script writes either of these: the harness that
    # produced them lives in a scratch directory and is not part of the repo, so an
    # mtime check has no producer to compare against. Reproducing them means
    # re-running that harness by hand.
    #   tennis_hparam_sweep.json  -- this file was copied in by hand. sweep_tennis_hparams.py
    #                                used to write its result into a SESSION scratch dir
    #                                (a hardcoded .../Temp/claude/<session-id>/scratchpad),
    #                                which is why there was no in-repo producer to compare
    #                                an mtime against. That script now writes
    #                                data/tennis_sweep.json instead. The exemption STAYS:
    #                                the two are different files and their schemas have not
    #                                been compared, so treating the new producer as this
    #                                artifact's source would be a claim nobody has checked.
    #   stage2_seed_nonvacuity.json -- written by the in-place backup/restore harness
    #                                that exercised train_stage2.py --seed.
    "tennis_hparam_sweep.json",
    "stage2_seed_nonvacuity.json",
    # HAND-CURATED, not generated. Each entry records a superlative on a live page that was
    # checked against its artifact, with the evidence. It is written by a human decision,
    # so an mtime check would only ever say "the checker is newer than your judgement".
    "superlative_registry.json",
    # Hand-recorded symbol hashes granting narrow mtime exemptions. A human decision,
    # re-verified against live source on every run.
    "symbol_dep_registry.json",
    # 40 hand-written cross-sport pairs (Brady <-> Curry). NO script writes this file; it is
    # irreplaceable if lost, and it was gitignored until now.
    "analogy_triples.json",
    "archetype_map.json", "sector_map.json",
}

SYMBOL_DEPS = ROOT / "data" / "symbol_dep_registry.json"


def symbol_exempt(name: str) -> tuple[bool, str]:
    """Is this artifact's mtime staleness explained by a SYMBOL-level dependency that has
    not actually changed?

    mtime is file-granular, and that is wrong when a consumer imports a handful of functions
    from a large module: editing an unrelated part of eval_unified.py marked three ablation
    artifacts stale, and regenerating those means 23 training runs for a change that cannot
    reach them. A gate permanently red for a reason everyone knows is false trains its
    reader to ignore it.

    THE EXEMPTION CANNOT OUTLIVE THE CODE IT WAS GRANTED AGAINST. Every declared symbol is
    re-hashed from current source on every run; one mismatch and the artifact is stale again.
    """
    if not SYMBOL_DEPS.exists():
        return False, ""
    reg = json.loads(SYMBOL_DEPS.read_text(encoding="utf-8")).get("entries", {})
    want = reg.get(name)
    if not want:
        return False, ""
    for key, recorded in want.items():
        mod, _, sym = key.partition("::")
        src_p = PIPE / mod
        if not src_p.exists():
            return False, f"{mod} is gone"
        text = src_p.read_text(encoding="utf-8")
        m = re.search(rf"^def {re.escape(sym)}\(.*?(?=^def |\Z)", text, re.M | re.S)
        if not m:
            return False, f"{key} no longer exists"
        now = hashlib.sha256(m.group(0).encode()).hexdigest()[:16]
        if now != recorded:
            return False, f"{key} changed ({recorded} -> {now})"
    return True, f"{len(want)} declared symbols unchanged"


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
            exempt, why = symbol_exempt(name)
            if exempt:
                rows.append((name, f"fresh*", 0.0))
                print(f"  NOTE {name}: {hours:.1f}h behind {newest_src} by mtime, but "
                      f"{why} — see data/symbol_dep_registry.json")
            else:
                rows.append((name, "STALE", hours))
                problems.append(
                    f"{name} is {hours:.1f}h older than {newest_src} — re-run "
                    f"`python pipeline/{producer}`" + (f" ({why})" if why else ""))
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
