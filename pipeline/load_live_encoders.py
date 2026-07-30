"""Vector Unified — Stage 2 live (unfrozen) encoder loader.

Reconstructs each sport's MTNN from its native checkpoint + feature matrix so the
forward is identical to training (no silent misreconstruction: load_state_dict is
strict). Returns, per sport:
  - the model (requires_grad=True, .eval())
  - a graph-bearing encode_batch(idx) -> L2-normed e_s [len(idx), d_s]
  - records aligned to load_encoders order (so unified_matrix player_idx still holds)

Stage 2 training re-encodes per-batch rows (not the whole corpus at once) to keep
the backward graph memory-safe.

Smoke gate (main): encode_full_numpy under no_grad must reproduce the frozen E_s
from load_encoders to cosine ~1.0 -> proves reconstruction is correct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from load_encoders import (HOOPS, GRID, PITCH, ROOT, UCACHE, SPORT_DIM,
                           _MTNN, _family_slices, _split_by_family, _l2norm)

# native train_mtnn modules import sibling helpers (e.g. hoops `composite_score`),
# so their pipeline dirs must be on sys.path for those imports to resolve.
for _d in (HOOPS / "pipeline", PITCH / "pipeline"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

DATA = ROOT / "data"
DEVICE_DEF = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Hoops — import native MTNN + data helpers from vector-hoops
# ---------------------------------------------------------------------------

def _hoops_bundle(device):
    import importlib.util
    # hoops train_mtnn imports sibling modules (composite_score, ...) from its own
    # pipeline dir; spec_from_file_location does NOT add that dir to sys.path, so do it
    # explicitly before exec. (Pitch has no sibling imports; Gridiron uses load_encoders._MTNN.)
    _hp = str(HOOPS / "pipeline")
    if _hp not in sys.path:
        sys.path.insert(0, _hp)
    spec = importlib.util.spec_from_file_location("hoops_train_mtnn", HOOPS / "pipeline" / "train_mtnn.py")
    hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)
    ck = torch.load(HOOPS / "pipeline" / "data" / "mtnn_best.pt",
                    map_location=device, weights_only=False)
    a = ck["args"]
    Z, M, names, seasons, pids, clusters, positions, season_ids, manifest = hm.load_bundle()
    fams = hm.family_slices(manifest)
    # Injury never feeds an input tower (train_mtnn.main drops it unconditionally,
    # 2026-07 durability-head change) -- mirror that here or state_dict shapes
    # for towers.career/towers.form (and the fusion input width) never match.
    fams = {k: v for k, v in fams.items() if k != "injury"}
    game_cols = hm.game_feature_cols(manifest)
    sk_g, sk_m, skill_keys, _n_core = hm.load_skill_labels(names, seasons)
    form_cols = hm.feature_cols(manifest, hm.FORM_FEATURES) or []
    bbref_cols = hm.feature_cols(manifest, hm.BBREF_FEATURES) or []
    injury_cols = hm.feature_cols(manifest, hm.INJURY_FEATURES) or []
    n_seasons = int(season_ids.max()) + 1
    model = hm.MTNN({f: len(c) for f, c in fams.items()}, n_seasons,
                    d_tower=a["tower_width"], d_tower_hidden=a["tower_hidden"],
                    d_emb=a["dim"], n_game=len(game_cols), n_skills=len(skill_keys),
                    d_skill_hidden=a["skill_hidden"], n_form=len(form_cols),
                    n_bbref=len(bbref_cols), fusion_mode=a["fusion"],
                    n_tower_blocks=a["tower_blocks"], mlp_heads=a["mlp_heads"],
                    d_head_hidden=a["d_head_hidden"], d_model=a["d_model"],
                    n_fusion_layers=a["n_fusion_layers"], n_attn_heads=a["n_attn_heads"],
                    d_fusion_hidden=(a["fusion_hidden"] or None),
                    n_injury=len(injury_cols)).to(device)
    model.load_state_dict(ck["model"], strict=True)
    xs, ms = hm.split_by_family(Z, M, fams, device)
    seas_t = torch.tensor(season_ids, device=device, dtype=torch.long)
    recs = [{"sport": "hoops", "player_id": str(int(pids[i])), "name": str(names[i]),
             "season": str(seasons[i]), "pos": int(positions[i]), "team": "",
             "native_cluster": int(clusters[i])} for i in range(len(names))]
    return model, xs, ms, seas_t, recs


# ---------------------------------------------------------------------------
# Pitch — import native PitchMTNN from vector-pitch
# ---------------------------------------------------------------------------

def _pitch_bundle(device):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pitch_train_mtnn", PITCH / "pipeline" / "train_mtnn.py")
    pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
    ck = torch.load(PITCH / "pipeline" / "data" / "pitch_mtnn.pt",
                    map_location=device, weights_only=False)
    cfg = ck["config"]
    d = np.load(PITCH / "pipeline" / "data" / "tm_full.npz", allow_pickle=False)
    X = d["X"].astype(np.float32); M = d["M"].astype(np.float32)
    ctx_ids = d["ctx_ids"].astype(np.int64)
    meta = json.loads((PITCH / "pipeline" / "data" / "meta_tm_full.json").read_text(encoding="utf-8"))
    manifest = json.loads((PITCH / "pipeline" / "data" / "feature_manifest_tm_full.json").read_text(encoding="utf-8"))
    feats = manifest["features"]; families = manifest["family_lists"]
    fam_dims = {fam: len(cols) for fam, cols in families.items()}
    slices = pm.family_slices(feats, families)
    model = pm.PitchMTNN(fam_dims, n_ctx=cfg["n_ctx"], d_tower=cfg["d_tower"],
                         d_emb=cfg["d_emb"], n_feat=cfg["n_feat"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    xs, ms = pm.split_by_family(X, M, slices, device)
    ctx_t = torch.tensor(ctx_ids, device=device, dtype=torch.long)
    recs = [{"sport": "pitch", "player_id": str(m["player_id"]), "name": m["name"],
             "season": m["context"], "pos": m["pos"], "team": m.get("team", ""),
             "native_cluster": -1} for m in meta]
    return model, xs, ms, ctx_t, recs


# ---------------------------------------------------------------------------
# Gridiron — reuse _MTNN from load_encoders; weekly->season aggregation
# ---------------------------------------------------------------------------

def _gridiron_bundle(device):
    ck = torch.load(GRID / "pipeline" / "data" / "mtnn_best.pt",
                    map_location=device, weights_only=False)
    feats = ck["feats"]; families = ck["families"]
    mu = ck["mu"]; sd = ck["sd"]
    n_seasons = int(ck["n_seasons"]); season_min = int(ck["season_min"])
    d_emb = int(ck.get("d_emb", 32))
    slices = _family_slices(feats, families)
    fam_dims = {f: len(c) for f, c in slices.items()}
    model = _MTNN(fam_dims, n_seasons=n_seasons, d_emb=d_emb).to(device)
    model.load_state_dict(ck["state"], strict=True)
    d = np.load(GRID / "pipeline" / "data" / "train_matrix.npz", allow_pickle=False)
    Z = d["Z"].astype(np.float32); M = d["mask"].astype(np.float32)
    season = d["season"].astype(int); gsis = d["gsis"].astype(str)
    name = d["name"].astype(str); pos = d["pos"].astype(str); team = d["team"].astype(str)
    Xz = ((Z - mu) / sd) * M
    sid = np.clip(season - season_min, 0, n_seasons - 1).astype(np.int64)
    xs, ms = _split_by_family(Xz, M, slices, device)
    sid_t = torch.tensor(sid, device=device, dtype=torch.long)
    # season-row aggregation map (matches load_encoders order): sort by (season,gsis)
    order = np.lexsort((gsis, season))
    keep = gsis != ""
    # build season groups in the SAME order load_encoders emits (lexsort season,gsis)
    season_rows = []  # list of arrays of weekly indices, one per season-row
    recs = []
    _gsis = gsis[order]; _season = season[order]; _name = name[order]; _pos = pos[order]; _team = team[order]
    _keep = keep[order]
    _widx = order
    from collections import Counter
    start = 0; n = len(_gsis)
    for i in range(1, n + 1):
        same = i < n and _gsis[i] == _gsis[start] and _season[i] == _season[start]
        if same:
            continue
        seg = slice(start, i)
        start = i
        if not _keep[seg.start]:
            continue
        season_rows.append(_widx[seg].copy())  # weekly indices (in original space)
        recs.append({"sport": "gridiron", "player_id": str(_gsis[seg.start]),
                     "name": str(Counter(_name[seg]).most_common(1)[0][0]),
                     "season": int(_season[seg.start]), "pos": str(Counter(_pos[seg]).most_common(1)[0][0]),
                     "team": str(Counter(_team[seg]).most_common(1)[0][0]), "native_cluster": -1})
    return model, xs, ms, sid_t, season_rows, recs


# ---------------------------------------------------------------------------
# Public API: load_live(device) -> {sport: LiveEncoder}
# ---------------------------------------------------------------------------

class LiveEncoder:
    def __init__(self, sport, model, recs, encode_fn):
        self.sport = sport
        self.model = model
        self.recs = recs
        self._encode = encode_fn  # (idx_tensor) -> L2-normed e_s [len(idx), d_s] graph-bearing
        self.d = SPORT_DIM[sport]

    def encode_batch(self, idx):
        return self._encode(idx)

    def encode_full_numpy(self, device, chunk=4096):
        """Full-corpus no-grad encode -> numpy E_s [N, d_s] (for smoke verification)."""
        n = len(self.recs)
        out = np.empty((n, self.d), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, chunk):
                idx = torch.arange(s, min(s + chunk, n), device=device, dtype=torch.long)
                e = self._encode(idx).cpu().numpy()
                out[s:s + len(idx)] = e
        return out


def load_live(device=DEVICE_DEF):
    live = {}
    # hoops
    hm, xs, ms, seas_t, recs = _hoops_bundle(device)
    hm.requires_grad_(True); hm.eval()

    def h_enc(idx, _xs=xs, _ms=ms, _st=seas_t, _m=hm):
        ex = {f: _xs[f][idx] for f in _xs}; em = {f: _ms[f][idx] for f in _ms}
        return F.normalize(_m.encode(ex, em, _st[idx]), dim=-1)
    live["hoops"] = LiveEncoder("hoops", hm, recs, h_enc)
    # pitch
    pm, pxs, pms, ctx_t, precs = _pitch_bundle(device)
    pm.requires_grad_(True); pm.eval()

    def p_enc(idx, _xs=pxs, _ms=pms, _ct=ctx_t, _m=pm):
        ex = {f: _xs[f][idx] for f in _xs}; em = {f: _ms[f][idx] for f in _ms}
        return F.normalize(_m.encode(ex, em, _ct[idx]), dim=-1)
    live["pitch"] = LiveEncoder("pitch", pm, precs, p_enc)
    # gridiron (weekly -> season scatter-mean)
    gm, gxs, gms, gsid, season_rows, grecs = _gridiron_bundle(device)
    gm.requires_grad_(True); gm.eval()
    season_rows_t = [torch.tensor(r, device=device, dtype=torch.long) for r in season_rows]

    def g_enc(idx, _xs=gxs, _ms=gms, _sid=gsid, _sr=season_rows_t, _m=gm, _d=SPORT_DIM["gridiron"]):
        # idx: season-row indices. gather weekly rows, encode, scatter-mean per season.
        wlists = [_sr[i] for i in idx.tolist()]
        wrows = torch.cat(wlists)
        # group id per weekly row (which season-row in the batch it belongs to)
        group = torch.cat([torch.full((len(w),), k, device=idx.device, dtype=torch.long)
                           for k, w in enumerate(wlists)])
        ex = {f: _xs[f][wrows] for f in _xs}; em = {f: _ms[f][wrows] for f in _ms}
        ew = _m.encode(ex, em, _sid[wrows])  # [W, 32] L2-normed (fusion normalizes)
        out = ew.new_zeros((len(idx), _d))
        out.index_add_(0, group, ew)
        cnt = ew.new_zeros((len(idx), 1))
        cnt.index_add_(0, group, ew.new_ones((len(wrows), 1)))
        out = out / cnt.clamp(min=1.0)
        return F.normalize(out, dim=-1)
    live["gridiron"] = LiveEncoder("gridiron", gm, grecs, g_enc)
    return live


# ---------------------------------------------------------------------------
# Smoke: live (no_grad) must reproduce frozen E_s (cosine ~1.0)
# ---------------------------------------------------------------------------

def main():
    import json as _json
    from load_encoders import load_all
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = DEVICE_DEF
    print(f"device={device}")
    meta = _json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))
    frozen = load_all(verbose=False)
    live = load_live(device)
    ok = True
    for sport in ("hoops", "gridiron", "pitch"):
        le = live[sport]
        E_live = le.encode_full_numpy(device)
        E_froz = frozen[sport]["E"]
        n_live, n_froz = E_live.shape[0], E_froz.shape[0]
        # norms
        nrm = float(np.linalg.norm(E_live, axis=1).mean())
        # cosine vs frozen (aligned order)
        if n_live == n_froz:
            A = E_live / (np.linalg.norm(E_live, axis=1, keepdims=True) + 1e-9)
            B = E_froz / (np.linalg.norm(E_froz, axis=1, keepdims=True) + 1e-9)
            cos = float((A * B).sum(axis=1).mean())
        else:
            cos = float("nan")
        meta_count = meta["coverage"][sport]
        counts_ok = n_live == meta_count
        cos_ok = n_live == n_froz and cos >= 0.999
        flag = "OK" if (counts_ok and cos_ok) else "FAIL"
        if not (counts_ok and cos_ok):
            ok = False
        print(f"  {sport:8s} live={n_live} frozen={n_froz} meta={meta_count}  "
              f"mean_norm={nrm:.4f}  cos_vs_frozen={cos:.5f}  [{flag}]")
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
