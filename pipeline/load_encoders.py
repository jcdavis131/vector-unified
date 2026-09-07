"""Vector Unified — per-sport encoder loader (Pillar 1).

Produces a common record stream from the three proven sport encoders so the
shared projection trunk (Pillar 2) can fold them into one 64-d space.

  e_h  hoops    64-d   <- vector-hoops/pipeline/data/embedding_v3.npz        (cached, shipped;
                          64-d concat-fusion v5.1 promoted 2026-07-25, was 48-d)
  e_g  gridiron 32-d   <- vector-gridiron/pipeline/data/mtnn_best.pt +       (regenerated,
                          train_matrix.npz  -> forward pass -> season-agg      no cached 32-d)
  e_p  pitch    24-d   <- vector-pitch/assets/pitch_mtnn_embeddings.json     (cached, just built)

Common record:
  {sport, player_id, name, season, pos, team, e: np.ndarray[d], native_cluster: int}

Honest constraints (named, not hidden):
  * Hoops and pitch load cached L2-normalized embeddings — the proven/shipped artifacts.
  * Gridiron ships only a 3-d PCA map (assets/embedding.json), NOT its 32-d embedding.
    We regenerate e_g by forward-passing the promoted checkpoint and aggregating a
    player's weekly rows within a season (mean -> re-L2-normalize). This is the
    "season-level who-is-this-player" vector the unified space is built on; the weekly
    prediction heads stay on the weekly path in the gridiron game and are untouched.
  * The gridiron MTNN class is copied minimal-and-stable here instead of importing
    vector-gridiron/pipeline/train_mtnn.py, because that module imports the full
    nfl_data + build_features ingestion layer at top of file. We depend only on the
    saved checkpoint wrapper {state, mu, sd, feats, families, n_seasons, season_min,
    d_emb} — no network, no sibling ingestion code.
  * native_cluster is the sport's own archetype id where it exists (hoops: 0-7;
    pitch/gridiron: -1 sentinel — derived later by archetype_map.py via k-means,
    because gridiron has no shipped archetype label and pitch's JSON omits it).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent  # C:\Users\jcdav
HOOPS = HOME / "vector-hoops"
GRID = HOME / "vector-gridiron"
PITCH = HOME / "vector-pitch"

SPORTS = ("hoops", "gridiron", "pitch")
SPORT_DIM = {"hoops": 64, "gridiron": 32, "pitch": 24}
SPORT_ID = {s: i for i, s in enumerate(SPORTS)}
UCACHE = ROOT / "pipeline" / "data"  # unified-side cache for derived artifacts


# ---------------------------------------------------------------------------
# Gridiron MTNN — minimal stable copy (do NOT import the ingestion layer)
# ---------------------------------------------------------------------------

class _ResidualTower(nn.Module):
    def __init__(self, d_in: int, d_out: int = 24, d_hidden: int = 64):
        super().__init__()
        d_cat = d_in * 2
        self.fc1 = nn.Linear(d_cat, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_cat, d_out) if d_cat != d_out else nn.Identity()

    def forward(self, x, m):
        h = torch.cat([x * m, m], dim=-1)
        return self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(h)))) + self.skip(h))


class _GatedFusion(nn.Module):
    def __init__(self, n_towers, d_tower, n_seasons, d_season=8, d_emb=32, d_hidden=128):
        super().__init__()
        self.season_emb = nn.Embedding(n_seasons, d_season)
        self.gate = nn.Linear(d_tower, 1)
        self.attn = nn.Sequential(nn.Linear(d_tower, d_tower), nn.Tanh(),
                                  nn.Linear(d_tower, 1))
        self.fuse = nn.Sequential(
            nn.Linear(d_tower + d_season, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden),
            nn.Dropout(0.15), nn.Linear(d_hidden, d_emb))

    def forward(self, tower_stack, season_ids):
        scores = self.attn(tower_stack).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        gates = torch.sigmoid(self.gate(tower_stack).squeeze(-1))
        mixed = (tower_stack * weights.unsqueeze(-1) * gates.unsqueeze(-1)).sum(1)
        s = self.season_emb(season_ids)
        return F.normalize(self.fuse(torch.cat([mixed, s], dim=-1)), dim=-1)


class _MTNN(nn.Module):
    def __init__(self, fam_dims, n_seasons, d_tower=24, d_emb=32, n_targets=6, n_usage=3, n_pos=4):
        super().__init__()
        self.families = sorted(fam_dims)
        self.towers = nn.ModuleDict({f: _ResidualTower(fam_dims[f], d_out=d_tower)
                                     for f in self.families})
        self.fusion = _GatedFusion(len(self.families), d_tower, n_seasons, d_emb=d_emb)
        # heads are unused for encoding but must exist to load the state_dict
        self.target_heads = nn.ModuleList([nn.Linear(d_emb, 1) for _ in range(n_targets)])
        self.usage_head = nn.Linear(d_emb, n_usage)
        self.position_head = nn.Linear(d_emb, n_pos)
        self.pedigree_head = nn.Linear(d_emb, 1)

    def encode(self, xs, ms, season_ids):
        stack = torch.stack([self.towers[f](xs[f], ms[f]) for f in self.families], dim=1)
        return self.fusion(stack, season_ids)


def _family_slices(feats, families):
    return {fam: [feats.index(c) for c in cols] for fam, cols in families.items()}


def _split_by_family(Xz, M, slices, device):
    xs, ms = {}, {}
    for fam, cols in slices.items():
        xs[fam] = torch.tensor(Xz[:, cols], dtype=torch.float32, device=device)
        ms[fam] = torch.tensor(M[:, cols], dtype=torch.float32, device=device)
    return xs, ms


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _l2norm(E):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


def load_hoops():
    """e_h from cached embedding_v3.npz (48-d, 12966 player-seasons)."""
    p = HOOPS / "pipeline" / "data" / "embedding_v3.npz"
    a = np.load(p, allow_pickle=False)
    E = np.ascontiguousarray(a["E"], dtype=np.float32)
    records = []
    pid = a["player_id"]; sea = a["season"]; nm = a["name"]
    clu = a["cluster"]; pos = a["position"]
    for i in range(E.shape[0]):
        records.append({
            "sport": "hoops", "player_id": str(int(pid[i])), "name": str(nm[i]),
            "season": str(sea[i]), "pos": int(pos[i]), "team": "",
            "native_cluster": int(clu[i]),
        })
    E = _l2norm(E)
    return E, records


def load_pitch():
    """e_p from cached pitch_mtnn_embeddings.json (24-d, 2430 player-seasons)."""
    p = PITCH / "assets" / "pitch_mtnn_embeddings.json"
    blob = json.loads(p.read_text(encoding="utf-8"))
    rows = blob["players"]
    E = np.array([r["e_p"] for r in rows], dtype=np.float32)
    records = [{
        "sport": "pitch", "player_id": str(r["player_id"]), "name": r["name"],
        "season": r["context"], "pos": r.get("pos", ""), "team": r.get("team", ""),
        "native_cluster": -1,
    } for r in rows]
    E = _l2norm(E)
    return E, records


def load_gridiron(device=None):
    """e_g regenerated from mtnn_best.pt over train_matrix.npz, season-aggregated (32-d).

    Cached to pipeline/data/gridiron_season_emb.npz, invalidated by max(mtnn_best.pt
    mtime, train_matrix.npz mtime), so the ~16s forward-pass runs once and is reused
    across build/train/eval.

    Was invalidated by the checkpoint's mtime alone. That missed the case this repo's
    own cross-repo freshness check (pipeline/check_artifact_freshness.py, 92e4f9a) was
    built to catch for `unified_matrix.npz` but never covered here: train_matrix.npz is
    the gridiron trainer's own rebuilt feature/roster matrix, and it changes on a
    different schedule than the checkpoint -- 2026-08-06 (mtnn_best.pt) vs 2026-09-01
    (train_matrix.npz, +2 season-rows to 5325 with no retrain). A checkpoint-only key
    would treat a cache regenerated today as fresh forever until the NEXT retrain, so a
    future train_matrix.npz rebuild (gridiron has jobs queued) could go silently stale
    again the same way -- the exact class of bug 92e4f9a's commit message describes,
    recurring one level down, inside the function that check depends on.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = GRID / "pipeline" / "data" / "mtnn_best.pt"
    matrix_path = GRID / "pipeline" / "data" / "train_matrix.npz"
    cache = UCACHE / "gridiron_season_emb.npz"
    newest_input = max(ckpt_path.stat().st_mtime, matrix_path.stat().st_mtime)
    if cache.exists() and cache.stat().st_mtime >= newest_input:
        a = np.load(cache, allow_pickle=False)
        E = np.ascontiguousarray(a["E"], dtype=np.float32)
        recs = [{
            "sport": "gridiron", "player_id": str(a["gsis"][i]), "name": str(a["name"][i]),
            "season": int(a["season"][i]), "pos": str(a["pos"][i]), "team": str(a["team"][i]),
            "native_cluster": -1,
        } for i in range(E.shape[0])]
        return E, recs

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feats = ckpt["feats"]; families = ckpt["families"]
    mu = ckpt["mu"]; sd = ckpt["sd"]
    n_seasons = int(ckpt["n_seasons"]); season_min = int(ckpt["season_min"])
    d_emb = int(ckpt.get("d_emb", 32))
    slices = _family_slices(feats, families)
    fam_dims = {f: len(c) for f, c in slices.items()}

    model = _MTNN(fam_dims, n_seasons=n_seasons, d_emb=d_emb).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    d = np.load(matrix_path, allow_pickle=False)
    Z = d["Z"].astype(np.float32); M = d["mask"].astype(np.float32)
    season = d["season"].astype(int); gsis = d["gsis"].astype(str)
    name = d["name"].astype(str); pos = d["pos"].astype(str); team = d["team"].astype(str)

    Xz = ((Z - mu) / sd) * M
    sid = np.clip(season - season_min, 0, n_seasons - 1).astype(np.int64)
    xs, ms = _split_by_family(Xz, M, slices, device)
    sid_t = torch.tensor(sid, dtype=torch.long, device=device)
    with torch.no_grad():
        Ew = model.encode(xs, ms, sid_t).cpu().numpy().astype(np.float32)  # (N, 32), L2-normed

    # aggregate weekly rows -> player-season (mean over a player's weeks that season)
    order = np.lexsort((gsis, season))  # stable: group by season then gsis
    Ew = Ew[order]; season = season[order]; gsis = gsis[order]
    name = name[order]; pos = pos[order]; team = team[order]
    keep = gsis != ""
    Ew, season, gsis = Ew[keep], season[keep], gsis[keep]
    name, pos, team = name[keep], pos[keep], team[keep]

    E_out = []; recs = []
    start = 0
    n = len(gsis)
    for i in range(1, n + 1):
        same = i < n and gsis[i] == gsis[start] and season[i] == season[start]
        if same:
            continue
        seg = slice(start, i)
        start = i
        if seg.start == seg.stop:
            continue
        e_mean = Ew[seg].mean(axis=0)
        E_out.append(e_mean)
        # pos / team: mode over the segment (stable per player; handles trades)
        pseg = pos[seg]; tseg = team[seg]; nseg = name[seg]
        recs.append({
            "sport": "gridiron", "player_id": gsis[seg.start],
            "name": str(Counter(nseg).most_common(1)[0][0]),
            "season": int(season[seg.start]),
            "pos": str(Counter(pseg).most_common(1)[0][0]),
            "team": str(Counter(tseg).most_common(1)[0][0]),
            "native_cluster": -1,
        })
    E = _l2norm(np.array(E_out, dtype=np.float32))
    UCACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, E=E,
             gsis=np.array([r["player_id"] for r in recs], dtype="<U12"),
             name=np.array([r["name"] for r in recs], dtype="<U28"),
             season=np.array([r["season"] for r in recs], dtype=np.int32),
             pos=np.array([r["pos"] for r in recs], dtype="<U4"),
             team=np.array([r["team"] for r in recs], dtype="<U4"))
    return E, recs


def load_all(verbose=True):
    out = {}
    for sport, fn in (("hoops", load_hoops), ("pitch", load_pitch),
                      ("gridiron", load_gridiron)):
        E, recs = fn()
        d = E.shape[1]
        assert d == SPORT_DIM[sport], f"{sport} dim {d} != expected {SPORT_DIM[sport]}"
        norms = np.linalg.norm(E, axis=1)
        if verbose:
            seasons = sorted({r["season"] for r in recs})
            print(f"[{sport}] N={len(recs):,} d={d} "
                  f"norm[min={norms.min():.4f} max={norms.max():.4f}] "
                  f"seasons={len(seasons)} pos={dict(Counter(r['pos'] for r in recs))}")
        out[sport] = {"E": E, "records": recs}
    return out


if __name__ == "__main__":
    raise SystemExit(0 if load_all() is not None else 1)
