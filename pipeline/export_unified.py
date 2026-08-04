"""Vector Unified — export the shippable joint embedding.

Encodes every player-season through the saved unified_best.pt into z (64-d, L2),
projects to a PCA(3) map (mirroring the per-sport `vectors.json` contract: x/y/z +
cluster + named archetypes + axes), and writes `assets/unified.json`.

Contract (sibling-convention match):
  - `players[i]` = {sport, player_id, name, season, pos, team, native_cluster,
    cross_arch, e[64], x, y, z}  -- `e` mirrors pitch_mtnn_embeddings; x/y/z/c
    mirror hoops vectors.json.
  - `archetypes` = the cross-sport taxonomy (A0-A11) from archetype_map.json.
  - `axes` = PCA-3 of the joint space (PC1/PC2/PC3; cross-sport role axes,
    interpretation deferred -- honest, not hand-waved).
  - `proj.W` = (3 x 64) PCA components for re-projection.

Additive only: per-sport assets are untouched. Run asserts: norms=1.0, no NaN,
per-sport row counts match the matrix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from load_encoders import SPORTS, SPORT_ID, ROOT, UCACHE, load_all
from train_unified import load_matrix
from eval_unified import load_model, encode_all

DATA = ROOT / "data"
ASSETS = ROOT / "assets"
# index -> cross-sport archetype id, in build_unified_matrix order (A0,A1,A2,A3,A4,A5,A11)
_meta = json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))
ARCH_NAMES = _meta["arch_names"]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = load_matrix(device)
    # NAMED EXPLICITLY. This is the STAGE 1 exporter and it wants the Stage 1
    # checkpoint, but it was relying on load_model's DEFAULT to get it — so
    # retargeting that default (which the Stage 2 work wanted to do) would have
    # silently repointed this exporter at a different model. A default that one
    # caller depends on invisibly is not a default, it is a hidden argument.
    model, ck = load_model(device, "unified_best.pt")
    z = encode_all(model, M, device)
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
    xyz = pca.fit_transform(z)          # (N,3)
    W = pca.components_.astype(np.float32)  # (3,64)

    amap = json.loads((DATA / "archetype_map.json").read_text(encoding="utf-8"))
    taxonomy = amap["taxonomy"]
    arch_label = {a["id"]: a["label"] for a in taxonomy}

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

    # per-sport count check
    counts = {SPORTS[s]: int((sid == s).sum()) for s in range(3)}
    meta = _meta
    for s in range(3):
        assert counts[SPORTS[s]] == meta["coverage"][SPORTS[s]], \
            f"{SPORTS[s]} count mismatch {counts[SPORTS[s]]} vs meta {meta['coverage'][SPORTS[s]]}"

    out = {
        "built": "2026-07-10",
        "model": "UnifiedTrunk Stage 1 (frozen encoders; d_emb=64, L2)",
        "d_emb": int(z.shape[1]),
        "n_players": int(z.shape[0]),
        "normalization": "per-sport frozen e_s -> shared trunk (adapter+era) -> 64-d L2; cross-sport archetype contrastive (SupCon) + CORAL + GRL",
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

    print(f"exported assets/unified.json  players={len(players)}  d_emb={out['d_emb']}")
    print(f"per-sport: {counts}")
    print(f"PCA(3) explained variance: {out['proj']['explained_variance']} (sum {sum(out['proj']['explained_variance']):.3f})")
    print("norms: min={:.5f} max={:.5f} (all ~1.0)".format(float(norms.min()), float(norms.max())))
    print("asserts PASS: no NaN, norms=1.0, per-sport counts match meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
