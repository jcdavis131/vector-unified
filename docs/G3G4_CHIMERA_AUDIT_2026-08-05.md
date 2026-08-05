G3/G4 + chimera audit 2026-08-05 (Hatch VM, no torch)

G3 archetype coherence:
- data/unified_report.json G3_cross_sport_archetype:
  silhouette 0.683 == claimed 0.683 PASS
  within 0.746 == claimed 0.746 PASS
  between -0.121 == claimed -0.121 PASS
  separation 0.746 - (-0.121) = 0.867 == reported +0.867 PASS
  composition_gap_pp 8.9 noted
  rank 12.4 floor 12 PASS

G4 cross-NN:
- cross_sport_nn_same_arch_hit_rate 0.9828 == claimed PASS
- random_baseline 0.1712 == claimed PASS
- lift 0.9828-0.1712=0.8116 == claimed +0.8116 PASS
- curated 0/40 top10 mean 2114 vs random 2067 ratio 0.978 indistinguishable PASS

Chimera / pitch difficulty (source of 92.9% claim):
- ~/workspace/vector-pitch/assets/difficulty_calibration.json summary n_in_band 588 n_targets 633 = 92.891% -> 92.9% rounding PASS
- median_difficulty_score 0.4843 == claimed PASS
- old PCA16 386/633 61.0% -> 588 = improvement +202 (+31.9pp) PASS
- band [0.4,0.8] expected_solve median 0.6 slope 2.5
- Note: chimera itself (20719 joint) daily puzzle uses deterministic dailySeed LCG, not difficulty band; the 92.9% claim is for pitch 633 WC-only game, often conflated with chimera in active-tasks label — clarified here.

All metrics provenance-honest, no fix needed to numbers. README already caveats G3 composition gap and G4 person-level fail.

Source hashes: unified_report.json unchanged from experimental projection, stage2.1_smoke notes full_data_missing_on_VM — honest experimental block separate.

No torch pip.
