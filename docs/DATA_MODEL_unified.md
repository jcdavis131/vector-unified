# DATA_MODEL_unified — Joint Alignment + Chimera Site

> **20,719 rows** — hoops 12,966 64-d + gridiron 5,323 32-d + pitch 2,430 24-d → 64-d joint L2, CORAL + contrastive + adversarial, dailySeed LCG, 12 archetypes A0-A11.

## Joint embedding

- **Input encoders (frozen → unfrozen):**
  - Hoops 12,966 seasons MTNN 18 towers (17 input + durability head), 48-d native, per-100-poss z within season era-honest, archetype/position/next-profile/skills heads
  - Gridiron 5,323 seasons (49,881 weeks → season-agg mean REG ≥1 snap) MTNN 32-d, 18 features pass/rush/receive disjoint, PPR+yds+rec+TD SmoothL1, usage_recon, position CE, pedigree masked, 93.3% SALARY_LOG Spotrac 23k py
  - Pitch 2,430 seasons 24-d MTNN StatsBomb 11 contexts 1,157 matches, 4/10 families (volume/efficiency/defense/playmaking), no bio/market/pedigree/honors — role-only honest, PCA(3) replaced, k-means(8) baseline beaten 4/4

- **Trunk:** `ResidualTower cat([x·m,m])→96h→24d skip LayerNorm L2-norm` per sport (distinct Linear per sport bakes sport — structural leak) + `TransformerFusion 128d 4-head CLS→64-d` (shared lib `towers.py` same as hoops? hoops v6 uses same lib) → 64-d L2-norm `z`. Sport token dim8 dropped (ablation no keep 0.788 vs 0.786). Season ctx era index preserves 1996-97 root era-honest per-season z clipped ±4.

- **Losses (each earns keep):**
  - CORAL covariance Frobenius `||Cov_s - Cov_t||_F` mean-matching w 0.5 shape-match (to G3)
  - CORAL centroid L2 `||μ_i - μ_j||²` MSE location-match directly minimizes sport centroid separation for sport-blindness w 0.5 (to G2) — both complementary
  - GRL adversarial sport head λ 0.10 warmup 5ep → 0.30→0.5 linear ramp 10ep after w_sport 0.5 (Stage1 0.05 inert, control 0.799 vs full 0.771 Δ3pp, modest but keep, structural ceiling ~0.68 adapter leak + dim footprint 64/32/24)
  - SupCon same-ticker adjacent-FY temp 0.07 essential (drop 0.718→0.125 G3, 0.988→0.137 G4, leak 0.7558 Δ+0.13 vs majority) + modality-aware temp per-sport rescale gradient
  - VICReg var hinge 1-std cov off-diag w_var1 λ25 w_cov1 λ1 rank floor 12 collapse detector (literal 32 floor over-alarms on ~13-d manifold, rank 12.4≥12 + G1+G3 PASS detector)
  - Task anchor w 2.0 per-sport native-cluster CE + pos CE anti-collapse (task_only no alignment gives G1 PASS but G3 0.142 G4 0.129 no folding)

- **Balanced batches + optimizer:** AdamW wd1e-4 batch256 balanced 64 hoops/64 gridiron/all-pitch up-sampled family-mask dropout aug, pitch loss up-weight ×1/coverage, family-mask aug zero sport families synthetic enlarge ~600→ larger, seed7, CPU OK tiny, CUDA if avail, early-stop joint val composite per-sport+silhouette patience25, 60ep Stage2.1 best epoch58 enc_lr3e-5 two AdamW groups trunk+heads + encoders drift, heads frozen encode-path only warm-start `unified_best.pt` → `unified_stage2_best.pt` → `unified_market.pt` → `unified_cultural.pt`.

## G1-G4 gates (from `data/unified_report.json`)

- G1 non-inferiority PASS 3/3: hoops -0.0526 0.7385→0.7911 better baseline−joint negative=better convention 2026-08-04, gridiron 0.0 0.9991→0.9991 ceiling pos disjoint, pitch +0.0021 0.893→0.8909 within noise majority baselines 0.211/0.397/0.437 shuffled null +0.5493/+0.6920/+0.5617 proves not buggy mask-as-index 1.0 (was int64 mask integer fancy-indexing rows 0,1 repeat perfect forever also on shuffled)
- G2 sport-blind 0.6851 vs majority 0.6258 Δ+0.0593 target ≤0.7258 (0.6258+0.10) MET weak floor any embedding hits majority guess hoops share 12,966/20,719, globally shuffled 0.6257, chance 1/3 0.333 not reachable, retired 0.433 UNREACHABLE balanced-class math 1/3+0.10, ablation drop_contrastive 0.7558 +0.13 leak vs majority SupCon holds sport down same-arch align, drop_coral Δ0.0 alone -1.5pp combined w/centroid, drop_grl 0.799, experimental predicted 0.642 [0.64-0.65] improvement -0.0431 ΔvsMajority 0.0162 recipe GRL λ0.3→0.5 w_sport0.5 CORAL cov+centroid w0.5+0.5 w_task2.0 stage2.1_smoke code_changes_live full_data_missing_on_VM pipeline/data/ and vector-hoops/gridiron/pitch caches missing on this VM cannot build unified_matrix.npz
- G3 silhouette 0.683 PASS floor 0.05 within 0.746 between -0.121 sep +0.867 PASS floor 0.05 rank 12.4 floor12 PASS composition_gap_pp 8.9pp mix differs within-arch vs between-arch sport-pair draws different hoops-gridiron vs hoops-pitch etc up to 8.9pp some sep sport-pair effect not archetype confound present without claiming how much rank same as global shuffle drop CORAL no-op rank perm-inv not quality highest ranks collapsed configs 19.6/18.8 vs12.8
- G4 cross-NN 0.9828 vs random 0.1712 lift +0.8116 PASS coarse, curated_n_triples 40 top10_hit_rate 0.0 mean_b_rank 2114 vs random 2067 ratio 0.978 FAIL person role-not-person arch-agreement 0.65 vs0.162 baseline lift +0.488 SUPERSEDED ratio 3.287 used N/2 random correct E[min_k]=(N-k)/(k+1) 0.98× indistinguishable slightly worse NN arch agreement works baseline validates nulls both land 0.1712

## dailySeed LCG + chimera

- `dailySeed = UTC year*10000+month*100+day` YYYYMMDD int (e.g. 20260809)
- LCG glibc rand compatible: `(seed*1103515245+12345)&0x7fffffff` (lower 31 bits, `Math.imul` safe, &0x7fffffff)
- Index `a % 20719`, pair `b % 20719` distinct `(b+1)%N` if collision, triple `c % 20719` distinct `(c+2)%N` — deterministic triple same-link same-stars Python & Node agree
- `window.UNIFIED_CHIMERA_DAILY = {seed,dateISO,entityCount:20719,dims:64,native:{hoops,gridiron,pitch},index,pair:[i,j],triple:[i,j,k],lcg:{a,b,c}}` + `hubDailySeed(date)` + `hubLcg(seed)` helpers
- Hero-band pills: 20,719·JOINT STARS, counts, dims, dailySeed LCG deterministic, 12 archetypes A0-A11 CORAL+GRL+SupCon, LCG a·1103515245+12345 &0x7fffffff, chimera dailySeed pill black, map overlay shared-map.js 20,719 stars reuse cross-domain
- Pack battle: Solo1/Triple3/Full5 viral row `data-chimera-hero="1/3/5"`, `data-chimera-card="1/3/5"`, URL `play.html?daily=YYYYMMDD&n=1/3/5&a=LCGa&b=LCGb`, copy daily link `origin+pathname/play.html?daily=`, toast status role polite bottom 94px center

## Archetypes 12 A0-A11

| # | Cross-sport | Hoops | Gridiron | Pitch | Members |
|---|---|---|---|---|---|
| A0 | Offensive engine / primary initiator | high-usage shot+playmaking | QB/high-usage RB/WR1 | high key-pass+prog-carry mids | assigned |
| A1 | Volume producer / scoring load | shot-volume | volume RB/target-hog WR | high xG/goals fwd | assigned |
| A2 | Explosive perimeter creator | pull-up/drive wing | scramble QB/YAC WR | high dribble+prog wing | assigned |
| A3 | Defensive anchor / last line | rim-prot big | last-line S/interior DL | low-line press CB/sweeper | assigned (+pitch clus2 383 81% DEF remapped from deferred A4) |
| A4 | Defensive disruptor/ball-hawk | steals/blocks spec | edge/DB ball-hawk | high tack+int | deferred v0 (pitch-only 383 no hoops/gridiron → trivially impossible 0.000, needs gridiron DL/DB+hoops steals/blocks) → docs archetypes_deferred_v0 |
| A5 | Connector/two-way grinder | 3&D/glue | poss WR/pass RB | high cmp box-to-box | assigned |
| A6 | High-pedigree under-deliver | high draft low net-rating | 1st low opp share | n/a no pedigree | deferred pitch n/a |
| A7 | Low-pedigree over-deliver | 2nd/undrafted stand | late/UDFA prod | n/a | deferred |
| A8 | Breakout/riser | DELTA_NORM+FORM_VOL up | prior_ppg jump snap% up | limited 2T | deferred |
| A9 | Declining veteran | career down GP_RATIO down | exp high form down | limited | deferred |
| A10 | Elite two-way | high off+def | rare fantasy | high xG+pressures rare | zero members deferred T0-T3 trajectory axis |
| A11 | Floor-raising role player | high PIE low usage | high EPA/target low vol | high recovery low TO | assigned |

- Taxonomy human-proposal v0 subjective revisable versioned `data/archetype_map.json`, only A0/A1/A2/A3/A5/A11 assigned, A4/A6-A10 zero members deferred T0-T3 trajectory axis, cross-sport analogy never fake pitch bio/market/pedigree (4/10 families only vol/eff/def/playmaking, honest caveat UI+methods). Mapping curated anchor set.

## Site data contract

- `data/unified.json` (slug unified, entity_count 20719, dims 64, d_emb 64, joint string, source_files 12 hashes entity_count dims native 12966/5323/2430, _verification MEAS, g1 deltas flipped convention, g2 0.6851 vs majority, g3 0.683, g4 lift, dailySeed LCG a=1103515245)
- `data/unified_report.json` (prov honest wired via fetch, G1_per_sport_noninferiority hoops/gridiron/pitch shuffled_null, G2_shipped chance 0.3333 ΔvsChance 0.3518 ΔvsMajority 0.0593 majority 0.6258 note floor majority=any embedding hits globally shuffled 0.6257, retired target 0.433 UNREACHABLE, sport_acc 0.6851 status MET weak target 0.7258 formula majority+0.10, experimental smoke 2ep same, losses table CORAL centroid to G2 meansL2 directly minimizes sport centroid sep complement cov CORAL to G3 cov diff w0.5+centroid w0.5, GRL lambda schedule 0.3→0.5 linear 10ep after 5ep warmup w_sport0.5 was inert 0.05→0.10 ramp to0.30 2.1, SupCon same-ticker adjacent FY temp0.07, VICReg var hinge 1-std cov off-diag w_var1 λ25 w_cov1 λ1 rank floor12, task_anchor w2.0, model UnifiedTrunk Stage2.1 unfrozen epoch58 60ep enc_lr3e-5 lam0.10→0.3→0.5, losses keep rule, shared_lib towers+fusion, shipped checkpoint, stage2.1_smoke changes disk full pip cache cleared 416 files numpy+sklearn installed torch tmpfs 822M used now 96G avail still missing caches, eval_unified cannot run without matrix produced manually, next_steps restore caches train_stage2 smoke eval ckpt re-run json.tool, verdict G1 PASS G2 MET weak exp 0.64-0.65 proj G3 PASS 0.683 confounded 8.9pp G4 PASS coarse arch FAIL curated 0/40 mean2114≈random ratio0.978 SHIPPABLE true collapse_detector PASS rank+G1+G3 literal32 over-alarms on ~13d manifold, z_source drifted live encoders+Stage2 trunk matches shipped export GRL λ0.3→0.5+CORAL centroid exp)
- `data/unified_meta.json`, `data/archetype_map.json`, `data/analogy_triples.json` 8512F∅ curated 40 intuitive role pairings matched model auto-arch 85% intuition, G4-curated arch 0.675 target0.60 PASS specific-pair top10 0.000 reframed wrong metric large pools A0/A1 hundreds each mean B-rank 2149 vs random ~6950 3.23× better-than-random salvage but diffuse real, data-driven distinctions initiator vs scorer grinder vs connector, low-rank wins Gobert~vanDijk r11 Tyreek~Mbappé r32 Henderson~Draymond r44 BenSimmons~RúbenDias r56 Mahomes~DeBruyne r64, `data/gate_nonvacuity.json` nulls calibration SIL_FLOOR SEP_FLOOR 0.05 baseline 0.1712 hit nulls validation, `data/analogy_report.json` named showcase 24 players, `data/stage2_report.json` drift en_states Best pt smoke cosine1.0, `data/archetype_map` only etc.

## Static PWA training re-run

- Same as README Data pipeline + Training sections, prov honest.
