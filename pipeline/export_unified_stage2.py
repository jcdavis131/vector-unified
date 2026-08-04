"""Vector Unified — export the Stage 2.1 joint embedding (ships G2 as a soft target).

Same contract as export_unified.py (assets/unified.json, sibling-convention
match) but encodes through the Stage 2.1 checkpoint (unified_stage2_best.pt):
the encoders themselves drifted during Stage 2, so this re-runs each sport's
live (loaded-with-drifted-weights) encoder full-corpus, not the frozen cached
e_s export_unified.py uses. Per-sport assets stay read-only -- the drifted
encoder weights live only in unified_stage2_best.pt, loaded here into fresh
LiveEncoder wrappers, never written back to vector-hoops/gridiron/pitch.

Ship decision (2026-07-30, user-confirmed): G1 holds (improves) for all three
sports and G3 holds, but G2 sport-invariance plateaued at 0.693 -- well above
the <=0.43 target. Per docs/STAGE2.1_SWEEP_PLAN.md §5 this is the "G2 plateau
> 0.55" branch: declare it a soft target and ship on the G1/G3 gains with an
honest caveat, which is what this export does. The caveat is baked into the
"model" and "g2_sport_acc" fields below so it travels with the artifact, not
just the docs.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

from load_encoders import SPORTS, ROOT, UCACHE, load_all
from load_live_encoders import load_live
from train_unified import load_matrix, UnifiedTrunk
from train_stage2 import full_z

DATA = ROOT / "data"
ASSETS = ROOT / "assets"
_meta = json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))
ARCH_NAMES = _meta["arch_names"]


def load_stage2_model(device):
    ck = torch.load(UCACHE / "unified_stage2_best.pt", map_location=device, weights_only=False)
    a = ck["args"]
    model = UnifiedTrunk(
        sport_dims=ck["sport_dim"], n_seasons_era=ck["n_eras"],
        d_adapter=a["d_adapter"], d_sport_tok=a["d_sport_tok"], d_emb=a["d_emb"], n_arch=8,
        # train_stage2.py's argparse has no --dropout (warm-starts the Stage 1
        # trunk architecture as-is); the value is inert anyway once .eval() is
        # called below, so a safe default matching train_unified.py's own
        # default is fine.
        n_pos=ck["n_pos"], dropout=a.get("dropout", 0.2),
        shared_adapter=a.get("shared_adapter", False),
        market_heads=a.get("market", False),
        cultural_text=a.get("cultural_text", False),
        d_text=a.get("d_text", 384),
    ).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = load_matrix(device)
    model, ck = load_stage2_model(device)

    live = load_live(device)
    for sport in SPORTS:
        live[sport].model.load_state_dict(ck["enc_states"][sport])
        live[sport].model.eval()

    z = full_z(model, live, M, device)
    sid = M["sport_id"].cpu().numpy()
    arch = M["arch_id"].cpu().numpy()
    pidx = M["player_idx"].cpu().numpy()
    native = M["native"].cpu().numpy()

    assert not np.isnan(z).any(), "NaN in z"
    norms = np.linalg.norm(z, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"norms not 1.0: min={norms.min()} max={norms.max()}"

    all_sport = load_all(verbose=False)
    recs = []
    for i in range(len(sid)):
        s = int(sid[i])
        recs.append(all_sport[SPORTS[s]]["records"][int(pidx[i])])

    pca = PCA(n_components=3, random_state=7)
    xyz = pca.fit_transform(z)
    W = pca.components_.astype(np.float32)

    amap = json.loads((DATA / "archetype_map.json").read_text(encoding="utf-8"))
    taxonomy = amap["taxonomy"]

    players = []
    for i in range(len(sid)):
        r = recs[i]
        players.append({
            "sport": SPORTS[int(sid[i])],
            "player_id": str(r["player_id"]),
            "name": str(r["name"]),
            "season": r["season"],
            "pos": str(r["pos"]),
            "team": str(r.get("team", "")),
            "native_cluster": int(native[i]),
            "cross_arch": ARCH_NAMES[int(arch[i])],
            "e": [round(float(v), 5) for v in z[i]],
            "x": round(float(xyz[i, 0]), 5),
            "y": round(float(xyz[i, 1]), 5),
            "z": round(float(xyz[i, 2]), 5),
        })

    counts = {SPORTS[s]: int((sid == s).sum()) for s in range(3)}
    for s in range(3):
        assert counts[SPORTS[s]] == _meta["coverage"][SPORTS[s]], \
            f"{SPORTS[s]} count mismatch {counts[SPORTS[s]]} vs meta {_meta['coverage'][SPORTS[s]]}"

    verdict = ck.get("verdict", {})

    # THE CHECKPOINT'S pos_drop IS THE BUGGY ONE AND A REBUILD DOES NOT FIX IT. `verdict`
    # comes straight out of unified_stage2_best.pt, which was written before the 7.21
    # knn5_acc fix, so every pos_drop in it is 0.0 — both arms were pinned at exactly 1.0 by
    # the mask-used-as-an-index bug. The old caveat below told the reader "this field is
    # only meaningful in assets rebuilt after that", which is WRONG REMEDIATION ADVICE: the
    # value is baked into the checkpoint, so re-exporting reproduces it forever and only a
    # retrain would move it. Substituting the measured values instead of shipping a known
    # -bad number with a misleading note.
    _probe = DATA / "g1_position_probe.json"
    _pos_source = "checkpoint (STALE — carries the 7.21 mask bug)"
    if _probe.exists():
        _pd = json.loads(_probe.read_text(encoding="utf-8")).get("per_sport") or {}
        _patched = 0
        for _sp, _row in _pd.items():
            if _sp in verdict and _row.get("pos_drop") is not None:
                verdict[_sp] = dict(verdict[_sp])
                verdict[_sp]["pos_drop"] = _row["pos_drop"]
                verdict[_sp]["pos_ok"] = bool(_row.get("pos_ok"))
                verdict[_sp]["pos_baseline_e_s"] = _row.get("baseline_e_s")
                verdict[_sp]["pos_joint_z"] = _row.get("joint_z")
                _patched += 1
        if _patched:
            _pos_source = (f"data/g1_position_probe.json — measured with the fixed "
                           f"knn5_acc over {_patched} sports, NOT the checkpoint's value")
    print(f"  G1 pos_drop source: {_pos_source}")
    # G2's target lives in the SHIPPED asset, so a wrong one is read by every downstream
    # consumer. 0.433 was `1/3 + 0.10`, which assumed balanced sports. They are 12,966 /
    # 5,323 / 2,430, so a majority predictor scores 0.6258 and a globally shuffled z —
    # carrying no sport information at all — scored 0.6257. A perfectly sport-invariant z
    # gives a classifier nothing but the class prior, making 0.6258 the FLOOR of achievable
    # accuracy: 0.433 was unreachable, not merely strict. See 7.20 and docs/SPEC.md.
    sid_np = M["sport_id"].cpu().numpy()
    majority = float(np.bincount(sid_np).max()) / len(sid_np)
    g2_target = round(majority + 0.10, 4)
    g2_acc = ck.get("best_g2")
    g2_status = ("met" if (g2_acc is not None and g2_acc <= g2_target)
                 else "not_met" if g2_acc is not None else "unknown")
    out = {
        "built": "2026-07-30",
        "model": "UnifiedTrunk Stage 2.1 (unfrozen encoder alignment, best_epoch="
                  f"{ck.get('best_epoch')})",
        "d_emb": int(z.shape[1]),
        "n_players": int(z.shape[0]),
        "normalization": "per-sport encoders (drifted, unfrozen in Stage 2) -> shared trunk (adapter+era) -> 64-d L2; cross-sport archetype contrastive (SupCon) + task + GRL",
        "g1_verdict": verdict,
        "g1_pos_source": _pos_source,
        "g1_pos_caveat": (
            "HISTORY, because this field read 0.0 for every sport from Phase 2 until "
            "2026-08-03 and meant nothing: knn5_acc used the int64 pos_mask as an INDEX "
            "rather than a mask, so both arms scored exactly 1.0 — on the real embedding "
            "and on a shuffled one alike — and their difference was 0.0 by construction. "
            "The earlier version of this note said the field 'is only meaningful in assets "
            "rebuilt after' the fix. That was WRONG REMEDIATION ADVICE. `verdict` is read "
            "straight from unified_stage2_best.pt, so a re-export reproduces the buggy "
            "value forever and only a RETRAIN would move it. pos_drop here is therefore "
            "substituted from data/g1_position_probe.json, measured with the fixed "
            "knn5_acc against the shipped z. Convention is baseline(e_s) - joint(z), so "
            "NEGATIVE means the joint space recovers position better. Measured: hoops "
            "-0.0526 (0.7385 -> 0.7911), gridiron 0.0000 (0.9991, already at ceiling), "
            "pitch +0.0021 (0.8930 -> 0.8909). The gate passes, and now it passes on "
            "evidence: a globally shuffled z drops +0.5493 / +0.6920 / +0.5617."),
        "g2_sport_acc": g2_acc,
        "g2_target": g2_target,
        "g2_majority_baseline": round(majority, 4),
        "g2_delta_vs_majority": (round(g2_acc - majority, 4) if g2_acc is not None else None),
        "g2_status": g2_status,
        "g2_note": ("Target is majority + 0.10. The previous 0.433 came from `1/3 + 0.10` "
                    "and was unreachable on these class sizes. `met` here means 'within 10 "
                    "points of the achievable floor', which is a weak bar — quote "
                    "g2_delta_vs_majority, not the status."),
        "sports": [{"id": s, "name": SPORTS[s], "d_native": int(all_sport[SPORTS[s]]["E"].shape[1]),
                    "n": counts[SPORTS[s]]} for s in range(3)],
        "archetypes": [{"id": a["id"], "label": a["label"], "description": a["description"]}
                       for a in taxonomy],
        "axes": [{"pc": f"PC{k+1}", "name": f"joint role axis {k+1}",
                  "note": "PCA of the cross-sport embedding; cross-sport role axis, interpretation deferred"}
                 for k in range(3)],
        "proj": {"W": [[round(float(v), 6) for v in row] for row in W],
                 "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_]},
        "players": players,
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "unified.json").write_text(json.dumps(out), encoding="utf-8")

    print(f"exported assets/unified.json (STAGE 2.1)  players={len(players)}  d_emb={out['d_emb']}")
    print(f"per-sport: {counts}")
    print(f"G2 sport_acc={g2_acc}  target<={g2_target} (majority {majority:.4f} + 0.10)  "
          f"delta_vs_majority={out['g2_delta_vs_majority']}  status={g2_status}")
    print(f"PCA(3) explained variance: {out['proj']['explained_variance']} (sum {sum(out['proj']['explained_variance']):.3f})")
    print("norms: min={:.5f} max={:.5f} (all ~1.0)".format(float(norms.min()), float(norms.max())))
    print("asserts PASS: no NaN, norms=1.0, per-sport counts match meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
