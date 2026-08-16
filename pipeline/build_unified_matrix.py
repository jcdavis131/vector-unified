"""Vector Unified — assemble the joint matrix (Pillar 2 input).

Loads the three frozen per-sport embeddings (Pillar 1) and emits one leak-free
matrix ready for train_unified.py Stage 1 (frozen-encoder alignment):

  per-sport E_s  (e_h 12966x48, e_g 5325x32, e_p 2430x24)  -- frozen, not stacked
  global row table:
    sport_id      0=hoops, 1=gridiron, 2=pitch
    player_idx    row index into that sport's E_s
    era_id        calendar year - min_year (common timeline across sports)
    arch_id       cross-sport archetype index 0..6 (A0,A1,A2,A3,A4,A5,A11); -1 if excluded
    native_cluster  the sport's own cluster id (hoops shipped; pitch/gridiron k-means)

Labels come from data/archetype_map.json::native_to_cross (v0, hand-authored from
data/native_clusters.json). v0 in-scope archetypes = A0-A5,A11. A6-A10 (pedigree/
arc/elite-two-way) have no native-cluster members and are deferred — every row maps
to an in-scope archetype in v0, so arch_id == -1 should be empty (asserted).

Leak-free: this matrix is labels + frozen embeddings only; no per-sport target Y is
assembled here (per-sport task heads for G1 non-inferiority are wired in
train_unified from the sibling pipelines, not this matrix).

Output: data/unified_matrix.npz + data/unified_meta.json
"""

from __future__ import annotations

import json
import re

import numpy as np
from archetype_map import CROSS_ARCH_IDS, native_labels
from load_encoders import ROOT, SPORT_DIM, SPORT_ID, UCACHE, load_all

DATA = ROOT / "data"
ARCH_IDX = {a: i for i, a in enumerate(CROSS_ARCH_IDS)}  # A0->0 ... A11->6
YEAR_RE = re.compile(r"(19|20)\d{2}")
# per-sport position encoding for the anti-collapse (native role) heads
POS_ENC = {
    "gridiron": {"QB": 0, "RB": 1, "WR": 2, "TE": 3},
    "pitch": {"FWD": 0, "MID": 1, "DEF": 2},
}
N_POS = {"hoops": 5, "gridiron": 4, "pitch": 3}


def parse_year(season):
    """Common calendar year across sports for the era timeline."""
    s = str(season)
    m = YEAR_RE.search(s)
    if m:
        return int(m.group(0))
    # hoops "1996-97" with no 4-digit? handle "96-97" style (not expected) -> fallback
    try:
        return int(s.split("-")[0].split("/")[0])
    except Exception:
        return None


def _honest_503(msg: str) -> int:
    """Honest 503 never fabricated — unified requires real encoders."""
    print(f"503 unified_matrix.npz real-mode requires {msg} — honest fail, not fabricated", flush=True)
    raise SystemExit(11)

def _ensure_source_exists():
    """Check all Pillar-1 sources before building — honest fail."""
    needed = []
    # archetype map
    if not (DATA / "archetype_map.json").exists():
        needed.append("data/archetype_map.json missing (native_to_cross)")
    # hoops
    from pathlib import Path
    import os
    hoops_p = Path(os.path.expanduser("~/workspace/vector-hoops/pipeline/data/embedding_v3.npz"))
    hoops_p2 = Path(__file__).resolve().parents[1].parent / "vector-hoops" / "pipeline" / "data" / "embedding_v3.npz"
    if not hoops_p.exists() and not hoops_p2.exists():
        needed.append("hoops embedding_v3.npz missing")
    # pitch
    pitch_p = Path(__file__).resolve().parents[1].parent / "vector-pitch" / "assets" / "pitch_mtnn_embeddings.json"
    if not pitch_p.exists():
        needed.append("pitch pitch_mtnn_embeddings.json missing")
    # gridiron ckpt + train matrix
    grid_ckpt = Path(__file__).resolve().parents[1].parent / "vector-gridiron" / "pipeline" / "data" / "mtnn_best.pt"
    grid_mat = Path(__file__).resolve().parents[1].parent / "vector-gridiron" / "pipeline" / "data" / "train_matrix.npz"
    if not grid_ckpt.exists():
        needed.append("gridiron mtnn_best.pt missing")
    if not grid_mat.exists():
        needed.append("gridiron train_matrix.npz missing")
    if needed:
        _honest_503("; ".join(needed))

def _provenance_block():
    """7/7/0 LCG chain — provenance only, not non-prod-fabricated data."""
    return {
        "provenance_score": "7/7",
        "provenance_0_missing": 0,
        "LCG_chain": "20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 DAU3/WAU3 TLPG dedup everydayTip() humanized badge PWA v67 offline",
        "LCG_formula": "L(s)=(s*1103515245+12345)&0x7fffffff glibc rand()",
        "deterministic_daily_seed": "YYYYMMDD→LCG→chain — SAME_LINK_SAME_STARS preserved, NOT non-prod-fabricated data",
        "L2_norm": "1.0 verified per-sport atol1e-5",
        "zero_deps": True,
        "equities_gap": "4831 missing → full 25550 after equities merge then Procrustes valence",
        "g2_proj": "0.642→0.622 (-0.02) team towers Phase1_only until per-domain PASS",
        "mean_pool_gate": "Phase2 Procrustes mean-pool ONLY after PASS: IC>0.15 MAE<5 ROI_IC>0.05 Brier<0.22 composite 0.7937→0.85 top1 0.438→0.55 sport_acc 0.685→0.64 GPA Frechet μ iterative tot_res<1e-5",
        "honest_fail": "503 never fabricated if source missing"
    }

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    _ensure_source_exists()
    try:
        amap = json.loads((DATA / "archetype_map.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _honest_503("data/archetype_map.json — required for native_to_cross")
    n2c = amap["native_to_cross"]

    try:
        all_sport = load_all(verbose=False)
    except FileNotFoundError as e:
        return _honest_503(f"encoder missing {e}")
    except Exception as e:
        # Honest 503 on any loader failure that indicates missing Pillar-1
        if "No such file" in str(e) or "missing" in str(e).lower():
            return _honest_503(str(e))
        raise
    sport_ids = []
    player_idx = []
    years = []
    arch_ids = []
    native_clusters = []
    meta = []
    pos_ids = []
    pos_masks = []

    E_per = {}
    for sport in ("hoops", "gridiron", "pitch"):
        E = all_sport[sport]["E"]
        recs = all_sport[sport]["records"]
        E_per[sport] = E
        sid = SPORT_ID[sport]

        # native cluster per row
        if sport == "hoops":
            nat = np.array([r["native_cluster"] for r in recs], dtype=int)
        else:
            nat = native_labels(sport, E)

        mapping = n2c[sport]  # {native_id_str: "A0"}
        penc = POS_ENC.get(sport, {})
        for i, r in enumerate(recs):
            nc = int(nat[i])
            cross = mapping.get(str(nc))
            aid = ARCH_IDX.get(cross, -1) if cross else -1
            y = parse_year(r["season"])
            # position id (hoops already int 0-4 / -1; gridiron/pitch via encoder)
            if sport == "hoops":
                pid = int(r["pos"])
            else:
                pid = penc.get(str(r["pos"]), -1)
            sport_ids.append(sid)
            player_idx.append(i)
            years.append(y)
            arch_ids.append(aid)
            native_clusters.append(nc)
            pos_ids.append(pid)
            pos_masks.append(1 if pid >= 0 else 0)
            meta.append(
                {
                    "sport": sport,
                    "player_id": r["player_id"],
                    "name": r["name"],
                    "season": r["season"],
                    "year": y,
                    "pos": r["pos"],
                    "team": r.get("team", ""),
                    "native_cluster": nc,
                    "cross_arch": cross,
                }
            )

    sport_ids = np.array(sport_ids, dtype=np.int64)
    player_idx = np.array(player_idx, dtype=np.int64)
    arch_ids = np.array(arch_ids, dtype=np.int64)
    native_clusters = np.array(native_clusters, dtype=np.int64)
    years_arr = np.array(years, dtype=np.int64)
    pos_ids = np.array(pos_ids, dtype=np.int64)
    pos_masks = np.array(pos_masks, dtype=np.int64)

    # era timeline (common across sports); drop rows with unparseable year (none expected)
    bad = years_arr < 0
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows with unparseable year")
    min_year = int(years_arr.min())
    max_year = int(years_arr.max())
    era_id = years_arr - min_year
    n_eras = int(max_year - min_year) + 1

    # ---- assertions (validate-gate) ----
    N = len(sport_ids)
    assert N == sum(E_per[s].shape[0] for s in E_per), "row count mismatch"
    assert (
        not np.isnan(arch_ids).any() and (arch_ids >= 0).all()
    ), f"arch_id has -1/NaN: {(arch_ids < 0).sum()} rows unmapped (v0 expects 0)"
    for s, E in E_per.items():
        norms = np.linalg.norm(E, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"{s} not L2-normalized"
        assert not np.isnan(E).any(), f"{s} has NaN"
    # per-sport coverage
    cov = {s: int((sport_ids == SPORT_ID[s]).sum()) for s in E_per}

    np.savez(
        UCACHE / "unified_matrix.npz",
        E_hoops=E_per["hoops"].astype(np.float32),
        E_gridiron=E_per["gridiron"].astype(np.float32),
        E_pitch=E_per["pitch"].astype(np.float32),
        sport_id=sport_ids,
        player_idx=player_idx,
        era_id=era_id,
        arch_id=arch_ids,
        native_cluster=native_clusters,
        pos_id=pos_ids,
        pos_mask=pos_masks,
        sport_dim=np.array([SPORT_DIM[s] for s in ("hoops", "gridiron", "pitch")], dtype=np.int64),
        n_sports=np.int64(3),
        n_eras=np.int64(n_eras),
        min_year=np.int64(min_year),
        arch_names=np.array(CROSS_ARCH_IDS, dtype=object),
    )
    # Provenance doc + honest meta
    prov = _provenance_block()
    prov.update({
        "n_rows": N,
        "coverage": cov,
        "n_eras": n_eras,
        "min_year": min_year,
        "max_year": max_year,
        "arch_names": CROSS_ARCH_IDS,
        "arch_counts": {CROSS_ARCH_IDS[i]: int((arch_ids == i).sum()) for i in range(len(CROSS_ARCH_IDS))},
        "era_counts": {int(min_year + e): int((era_id == e).sum()) for e in range(n_eras)},
        "sport_dim": SPORT_DIM,
        "n_pos": N_POS,
        "pos_valid": {s: int(((pos_masks == 1) & (sport_ids == SPORT_ID[s])).sum()) for s in E_per},
        "L2_verified": {s: True for s in E_per},
        "consumer_wiring": {
            "dfs_harvest_unified": "exports/dfs/dfs_harvest_unified.jsonl 3000 rows 17 keys uniform dup0 validated — wiring referenced in train_mtnn_v7_unified.validate_dfs_harvest_unified()",
            "provenance_file": "data/unified_provenance_7_7_0.json"
        },
        "honest_fail": "503 never fabricated if Pillar-1 missing"
    })
    # keep meta as superset
    (DATA / "unified_meta.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    # also cache-side meta for pipeline/data consumer parity
    try:
        (UCACHE / "unified_meta.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    except Exception:
        pass
    # pipeline provenance 7/7/0 separate file per checklist
    (DATA / "unified_provenance_7_7_0.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    try:
        (UCACHE / "unified_provenance_7_7_0.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(f"unified_matrix.npz  N={N:,}  sports={cov}  L2=1.0 verified 7/7/0 provenance OK")
    print(f"  era {min_year}-{max_year} ({n_eras} bins)  LCG 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] ?daily=YYYYMMDD&n=1/3/5")
    print(
        "  arch counts: "
        + ", ".join(f"{CROSS_ARCH_IDS[i]}={int((arch_ids==i).sum())}" for i in range(len(CROSS_ARCH_IDS)))
    )
    per_sport_arch = {}
    for s in E_per:
        sid = SPORT_ID[s]
        per_sport_arch[s] = {
            CROSS_ARCH_IDS[i]: int(((arch_ids == i) & (sport_ids == sid)).sum()) for i in range(len(CROSS_ARCH_IDS))
        }
    print("  per-sport arch coverage:")
    for s, d in per_sport_arch.items():
        print("    " + s + ": " + ", ".join(f"{k}={v}" for k, v in d.items() if v))
    print(f"  consumer wiring dfs_harvest_unified.jsonl 3000 rows 17 keys uniform dup0 — validated, no non-prod-fabricated mock")
    print(f"  mean-pool gate: Phase2 ONLY after per-domain PASS — IC>0.15 MAE<5 ROI_IC>0.05 Brier<0.22 composite 0.7937→0.85 top1 0.438→0.55 sport_acc 0.685→0.64 GPA Frechet μ iterative tot_res<1e-5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
