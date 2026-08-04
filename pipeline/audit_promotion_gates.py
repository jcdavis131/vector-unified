"""Most promotion gates in the estate are a threshold on a noisy scalar. Most of them fail.

Each repo has a gate that decides whether a trained model ships. Each was written before
its trainer had a --seed flag, so each threshold was calibrated against exactly one draw
of a quantity that turns out to move between draws. This audits them against the seed
evidence that now exists.

    repo                      gate                       calibrated from    verdict
    hoops                     test recall within 0.02    one run            FAILS
    gridiron                  CQS >= BASELINE 63.16      one run (the MAX)  FAILS
    equities                  CQS >= 0.60                a round number     MARGINAL
    pitch                     beats_pca3_count == 4      nothing            HOLDS
    unified (Stage 2 trunk)   G1 / G2 / G3               one run            UNTESTABLE
    unified (tennis MTNN)     recall@10 > BAR 0.0584     a 5-SEED MEAN      HOLDS

THE FIRST VERSION OF THIS FILE SAID "every promotion gate in the estate" AND AUDITED
FOUR. It skipped vector-unified's own G1/G2/G3 gates and the tennis MTNN's bar -- the two
that live in the repo the audit was written in. Claiming completeness over a set I had
not enumerated is the same defect as the rest of this thread, one level up.

HOOPS. Already recorded in vector-hoops/pipeline/seed_floor.json: the gate requires test
recall@10 to stay within 0.02 of baseline, and the measured seed range is 0.226 -- 11x
that tolerance. A re-run with NO CHANGE AT ALL misses the gate with probability ~0.83.

GRIDIRON. composite_score.BASELINE = {"mae": 4.296, "cqs": 63.16}, commented "Promoted
baseline (2025 holdout) -- update when a trial promotes". Three seeds give CQS 63.16,
62.35, 61.70. The baseline is the MAXIMUM, and seed 7's own log reads "redeploy-ok: CQS
63.16 >= baseline 63.16" -- it passes because it IS the baseline. Seeds 11 and 13 both
fail. Note the MAE half of the same constant is fine: 4.296 against observed 4.295 /
4.294 / 4.306, i.e. near the mean. One constant, one half calibrated to the mean and the
other to the max, and nothing recorded which.

EQUITIES. A fixed absolute threshold, CQS >= 0.60, so it is immune to the
calibrated-from-one-draw problem. It is marginal for a different reason: fixing the
sector-label bug cost 0.0054 CQS (vector-equities ca10102), which moved the model closer
to its own gate. Over 5 seeds the corrected arm means 0.6108 with sd 0.0106 -- 1.0 sd of
headroom -- and its worst seed scores 0.5983, which FAILS. The buggy arm had 1.7 sd of
headroom. Making the model more honest made it more likely to be refused.

PITCH. beats_pca3_count == 4 is a COUNT of per-metric comparisons, not a threshold on a
scalar, and it is 4 on all five seeds (vector-pitch 413e3cd). Counting how many
comparisons a model wins is far more stable than asking whether one number cleared a
line, because each comparison has to flip sign rather than drift. This is the design the
other three should copy, and it was not chosen for that reason -- it just happens to be
robust.

UNIFIED STAGE 2 IS THE ONE THAT CANNOT BE CHECKED AT ALL. train_stage2.py has no --seed
flag AND no SEED constant to override -- SEED is used at three sites and there is no way
to vary it -- so the unified model has never been run at a second seed. Its G2 gate then
passes by exactly zero: effective_rank 12.0 against rank_nondeg_floor 12, a hardcoded
literal at eval_unified.py:242. Whether that floor was chosen before or after seeing 12.0
is not recorded, and with one run there is no way to tell. The LITERAL target of 32
(= d_emb/2) fails, and G2 overall is DEFERRED pending a no-GRL baseline never run. G3
passes at 14.7x and 18.8x its 0.05 floors, which is so loose it carries little
information.

TENNIS IS THE ONE THAT WAS BUILT RIGHT, and by accident of design rather than intent.
train_tennis_mtnn.py hardcodes SEEDS = (7, 11, 13, 17, 19) and runs all five every
invocation, so it never needed a flag and was never a one-draw experiment. Its bar is the
FIVE-SEED MEAN of the learned linear map (0.0584, sd 0.005), and the MTNN cleared it 5/5
at mean 0.1168 (sd 0.0152). Both sides of the comparison are seed-means. That is the
shape every other gate should have.

WHAT THIS SCRIPT IS. A reader over artifacts that already exist. It re-runs no model and
changes no gate; retuning a gate is an operator decision and doing it from this evidence
would be tuning thresholds against the three or five seeds that happened to run.

    python pipeline/audit_promotion_gates.py

Writes: data/promotion_gate_audit.json
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "promotion_gate_audit.json"
HOOPS_FLOOR = Path(r"C:\Users\jcdav\vector-hoops\pipeline\seed_floor.json")
GRID_FLOOR = Path(r"C:\Users\jcdav\vector-gridiron\pipeline\seed_floor.json")
PITCH_FLOOR = Path(r"C:\Users\jcdav\vector-pitch\pipeline\seed_floor.json")
EQ_AB = Path(r"C:\Users\jcdav\vector-equities\pipeline\data\ab_sector_labels.json")
UNIFIED_REPORT = ROOT / "data" / "unified_report.json"
TENNIS_PROBE = ROOT / "data" / "tennis_metric_probe_enriched.json"


def main() -> int:
    missing = [p for p in (HOOPS_FLOOR, GRID_FLOOR, PITCH_FLOOR, EQ_AB) if not p.exists()]
    if missing:
        print(f"FAIL: missing {[str(p) for p in missing]}", file=sys.stderr)
        return 2

    hoops = json.loads(HOOPS_FLOOR.read_text(encoding="utf-8"))
    grid = json.loads(GRID_FLOOR.read_text(encoding="utf-8"))
    pitch = json.loads(PITCH_FLOOR.read_text(encoding="utf-8"))
    eqab = json.loads(EQ_AB.read_text(encoding="utf-8"))

    eq_fixed = [r["cqs"] for r in eqab["runs"] if r["arm"] == "B_fixed"]
    eq_buggy = [r["cqs"] for r in eqab["runs"] if r["arm"] == "A_buggy"]
    grid_cqs = grid["cqs"]["per_seed"]
    gv = [float(v) for v in grid_cqs.values()]

    rows = [
        {
            "repo": "vector-hoops", "gate": "test recall@10 within 0.02 of baseline",
            "calibrated_from": "one run", "n_seeds_of_evidence": hoops.get("n_runs"),
            "verdict": "FAILS",
            "detail": hoops.get("promotion_gate_is_unsatisfiable"),
        },
        {
            "repo": "vector-gridiron", "gate": "CQS >= composite_score.BASELINE['cqs']",
            "threshold": 63.16, "calibrated_from": "one run, and it is the MAXIMUM seed",
            "n_seeds_of_evidence": 3, "verdict": "FAILS",
            "observed_cqs_by_seed": grid_cqs,
            "mean": round(st.mean(gv), 2), "sd_n3": round(st.stdev(gv), 3),
            "seeds_that_pass": [s for s, v in grid_cqs.items() if float(v) >= 63.16],
            "seeds_that_fail": [s for s, v in grid_cqs.items() if float(v) < 63.16],
            "detail": "seed 7's log reads 'redeploy-ok: CQS 63.16 >= baseline 63.16' -- "
                      "it passes because it IS the baseline. The MAE half of the same "
                      "constant, 4.296, sits near the mean of 4.295/4.294/4.306, so one "
                      "constant has one half calibrated to the mean and the other to the "
                      "max, with nothing recording which.",
        },
        {
            "repo": "vector-equities", "gate": "CQS >= 0.60",
            "threshold": 0.60, "calibrated_from": "a round number, not a run",
            "n_seeds_of_evidence": len(eq_fixed), "verdict": "MARGINAL",
            "corrected_arm": {
                "mean": round(st.mean(eq_fixed), 4), "sd": round(st.stdev(eq_fixed), 4),
                "min": round(min(eq_fixed), 4),
                "headroom_in_sd": round((st.mean(eq_fixed) - 0.60) /
                                        st.stdev(eq_fixed), 1),
                "seeds_failing_the_gate": [round(v, 4) for v in eq_fixed if v < 0.60],
            },
            "buggy_arm": {
                "mean": round(st.mean(eq_buggy), 4), "sd": round(st.stdev(eq_buggy), 4),
                "min": round(min(eq_buggy), 4),
                "headroom_in_sd": round((st.mean(eq_buggy) - 0.60) /
                                        st.stdev(eq_buggy), 1),
            },
            "detail": "This gate is immune to the calibrated-from-one-draw problem "
                      "because 0.60 came from nowhere in particular. It is marginal for "
                      "a different reason: fixing the sector-label bug cost 0.0054 CQS, "
                      "moving the model toward its own gate. The corrected arm has 1.0 "
                      "sd of headroom and one of five seeds fails; the buggy arm had "
                      "1.7. Making the model more honest made it more likely to be "
                      "refused.",
        },
        {
            "repo": "vector-pitch", "gate": "beats_pca3_count == 4",
            "calibrated_from": "nothing -- it is a count, not a threshold",
            "n_seeds_of_evidence": len(pitch.get("seeds", [])), "verdict": "HOLDS",
            "observed": pitch["shipped_verdict_survives"]["beats_pca3_count_per_seed"],
            "detail": "4 of 4 on all five seeds. A count of per-metric comparisons is "
                      "far more stable than a threshold on a scalar, because each "
                      "comparison has to flip SIGN rather than drift across a line. Every "
                      "pitch margin clears its paired floor by 3.9x to 11.5x. This is "
                      "the design the other three should copy -- and it was not chosen "
                      "for that reason, it just happens to be robust.",
        },
    ]

    # --- the two gates the first version of this audit MISSED --------------
    uni = json.loads(UNIFIED_REPORT.read_text(encoding="utf-8"))         if UNIFIED_REPORT.exists() else {}
    ten = json.loads(TENNIS_PROBE.read_text(encoding="utf-8"))         if TENNIS_PROBE.exists() else {}
    g2 = uni.get("G2_sport_invariance", {})
    g3 = uni.get("G3_cross_sport_archetype", {})
    rows.append({
        "repo": "vector-unified (Stage 2 trunk)",
        "gate": "G1 non-inferiority / G2 sport-invariance / G3 archetype",
        "calibrated_from": "one run - train_stage2.py has NO --seed flag and no SEED "
                           "constant to override; the unified model has never been run "
                           "at a second seed",
        "n_seeds_of_evidence": 1, "verdict": "UNTESTABLE",
        "G1": uni.get("verdict", {}).get("G1"),
        "G2": uni.get("verdict", {}).get("G2"),
        "G3": uni.get("verdict", {}).get("G3"),
        "G2_rank_detail": {
            "effective_rank": g2.get("effective_rank"),
            "rank_nondeg_floor": g2.get("rank_nondeg_floor"),
            "margin": (None if g2.get("effective_rank") is None
                       else round(g2["effective_rank"] - g2["rank_nondeg_floor"], 4)),
            "rank_target_literal": g2.get("rank_target_literal"),
            "rank_literal_pass": g2.get("rank_literal_pass"),
            "problem": "effective_rank is 12.0 and rank_nondeg_floor is 12 - the gate "
                       "passes by EXACTLY ZERO. The floor is a hardcoded literal at "
                       "eval_unified.py:242, commented 'non-degenerate floor: below this "
                       "with role/folding loss = collapse'. Whether it was chosen before "
                       "or after seeing 12.0 is not recorded, and with one run there is "
                       "no way to tell. A gate clearing by 0.0 on a model that cannot be "
                       "re-seeded is not evidence either way. The LITERAL target of 32 "
                       "(= d_emb/2) FAILS, and G2 overall is DEFERRED pending a no-GRL "
                       "baseline that was never run.",
        },
        "G3_floor_detail": {
            "silhouette": g3.get("silhouette"),
            "silhouette_floor": g3.get("silhouette_floor"),
            "separation": g3.get("separation"),
            "separation_floor": g3.get("separation_floor"),
            "multiples_over_floor": [
                None if not g3 else round(g3["silhouette"] / g3["silhouette_floor"], 1),
                None if not g3 else round(g3["separation"] / g3["separation_floor"], 1)],
            "problem": "Both floors are 0.05 against observed 0.7339 and 0.9385 - 14.7x "
                       "and 18.8x. A floor that loose passes almost anything, so G3 "
                       "PASSING carries little information. Not wrong; barely "
                       "discriminating.",
        },
    })
    lps = ten.get("learned_per_seed") or []
    rows.append({
        "repo": "vector-unified (tennis MTNN)",
        "gate": "MTNN recall@10 > BAR 0.0584", "threshold": ten.get("learned_mean"),
        "calibrated_from": "a FIVE-SEED MEAN - the only correctly-built bar in the estate",
        "n_seeds_of_evidence": len(lps), "verdict": "HOLDS",
        "bar_per_seed": lps, "bar_sd": ten.get("learned_sd"),
        "detail": "train_tennis_mtnn.py hardcodes SEEDS = (7, 11, 13, 17, 19) and runs "
                  "all five every invocation, so it never needed a --seed flag and was "
                  "never a one-draw experiment. The bar is the 5-seed mean of the learned "
                  "linear map (sd 0.005), not one run, and the MTNN cleared it 5/5 at "
                  "mean 0.1168 (sd 0.0152). BOTH SIDES of the comparison are seed-means. "
                  "This is what the other gates should have been.",
    })

    out = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "CORRECTION_first_version_was_incomplete": "The first version said 'every "
            "promotion gate in the estate' and covered four trainers. It skipped "
            "vector-unified's own G1/G2/G3 gates and the tennis MTNN's 0.0584 bar - the "
            "two that live in the repo the audit was written in. Claiming completeness "
            "over a set I had not enumerated is the same defect as the rest of this "
            "thread, one level up.",
        "question": "Do the estate's promotion gates measure the model, or the seed?",
        "why_now": "Every gate was written before its trainer had a --seed flag, so each "
                   "threshold was calibrated against exactly one draw of a quantity that "
                   "turns out to move between draws. All four trainers now have seed "
                   "evidence, so the question is finally answerable.",
        "summary": {r["repo"]: r["verdict"] for r in rows},
        "n_gates_audited": len(rows),
        "the_pattern": "Three of four gates are a threshold on a noisy scalar, set "
                       "without knowing the noise. The fourth is a COUNT of comparisons "
                       "and is the only one that holds. A gate should ask 'did this beat "
                       "the alternatives' rather than 'did this number clear a line', "
                       "because a count requires a sign flip where a threshold only "
                       "requires drift.",
        "what_this_does_not_do": "Retune anything. Picking new thresholds from the three "
                                 "or five seeds that happened to run would be tuning "
                                 "against the cases being judged -- the failure mode this "
                                 "whole phase has been correcting. The gates stay as they "
                                 "are; this records what they currently mean.",
        "gates": rows,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{'repo':<18}{'gate':<38}{'verdict'}")
    for r in rows:
        print(f"  {r['repo']:<16}{r['gate'][:36]:<38}{r['verdict']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
