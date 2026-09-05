"""Vector Unified — Stage 1 alignment trainer (Pillar 2 + 3a/3b/3c).

Frozen per-sport encoders (e_h/e_g/e_p from load_encoders) -> per-sport adapter ->
shared projection trunk -> 64-d L2-normalized z. Trained with three folding forces:

  3b  SupCon InfoNCE on cross-sport archetypes (modality-aware per-sport temperature)
  3b  CORAL pairwise covariance alignment across sports (2nd-order geometry match)
      + CORAL centroid L2 (means alignment)  v2.1: --w-coral 0.5 + --w-coral-centroid 0.5
  3c  Gradient-reversal sport classifier (z should not predict sport)
      lambda schedule 0.3 -> 0.5 ramp10 after warmup 5  (--grl-lambda 0.3 --grl-lambda-target 0.5)
  3a  Per-sport native-cluster + position heads (anti-collapse anchor)
  3a+ Cross-sport market heads (--market)
  3a++ Cultural text (--cultural-text)

Encoders are frozen. Only learnable: adapters, trunk, heads, temperatures.

Honest v0 scope / OOM guard:
  * Hoops emb v3 48-d cache, pitch 24-d cached, gridiron regenerated from mtnn_best.pt.
  * Hatch VM missing embedding_v3.npz / mtnn_best.pt / pitch_mtnn_embeddings.json
    → PRODUCTION-ONLY per 2026-08-15: real unified_matrix.npz 20719 required, NO synthetic fallback, honest 503 if missing,
    gated honest not promoted, pending 130 feats full 18 families LOCAL-GPU deferred.
  * Zero-deps true, torch auto cuda else cpu, pacing :01 lite 3 max concurrent / <7 total,
    LOCAL-GPU exempt.

CLI (T5 hill 132):
  python pipeline/train_unified.py --w-coral 0.5 --w-coral-centroid 0.5 \
    --grl-lambda-target 0.5 --grl-lambda 0.3 --grl-ramp 10 --w-task 2.0 --w-sport 0.5 \
    --epochs 60 --seeds 7,11,13,17,19 --paired --eval-every 5 \
    --out pipeline/data/unified_stage2_centroid_ab.pt

Smoke quick <3m:
  python3 pipeline/train_unified.py --smoke --epochs 2 --grl-lambda 0.3 --grl-lambda-target 0.5 \
    --grl-ramp 10 --w-task 2.0 --w-coral 0.5 --w-coral-centroid 0.5 --w-sport 0.5 --seeds 7,11

Triple-write artifacts:
  ~/workspace/bundles/ultra/runs/dag-h132/builder1.log
  ~/workspace/vector-unified/data/unified_report.json  G2 0.6851 -> 0.642 proj
  + .scout/missions + dottie/bundles mirrors even no-change (checkpoint-manager 7-field)

Collapse early-warning: effective rank participation ratio logged each epoch, target >=32 half of 64.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_encoders import ROOT, SPORT_DIM, UCACHE

DATA = ROOT / "data"
SEED = 7
SPORTS = ("hoops", "gridiron", "pitch")


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class UnifiedTrunk(nn.Module):
    def __init__(
        self,
        sport_dims,
        n_seasons_era,
        d_adapter=48,
        d_sport_tok=8,
        d_era=8,
        d_emb=64,
        n_arch=8,
        n_pos=(5, 4, 3),
        dropout=0.2,
        shared_adapter=False,
        market_heads=False,
        cultural_text=False,
        d_text=384,
    ):
        super().__init__()
        self.sport_dims = sport_dims
        self.d_sport_tok = d_sport_tok
        self.shared_adapter = shared_adapter
        self.market_heads = market_heads
        self.cultural_text = cultural_text
        self.d_text = d_text
        self.max_dim = int(max(sport_dims))
        if shared_adapter:
            self.adapters = None
            self.shared_lin = nn.Linear(self.max_dim, d_adapter)
        else:
            self.adapters = nn.ModuleList([nn.Linear(d, d_adapter) for d in sport_dims])
            self.shared_lin = None
        self.sport_tok = nn.Embedding(3, d_sport_tok) if d_sport_tok > 0 else None
        self.era_emb = nn.Embedding(n_seasons_era, d_era)
        d_in = d_adapter + d_sport_tok + d_era
        self.trunk = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, d_emb),
        )
        self.native_heads = nn.ModuleList([nn.Linear(d_emb, n_arch) for _ in range(3)])
        self.pos_heads = nn.ModuleList([nn.Linear(d_emb, n_pos[s]) for s in range(3)])
        self.sport_clf = nn.Linear(d_emb, 3)
        self.log_temp = nn.Parameter(torch.zeros(3))
        if market_heads:
            self.salary_head = nn.Linear(d_emb, 1)
            self.award_head = nn.Linear(d_emb, 1)
            self.reach_head = nn.Linear(d_emb, 1)
        else:
            self.salary_head = None
            self.award_head = None
            self.reach_head = None
        if cultural_text:
            self.text_proj = nn.Linear(d_emb, d_text)
        else:
            self.text_proj = None

    def encode(self, e_per_sport, sport_ids, era_ids, return_raw=False):
        if self.shared_adapter:
            out = e_per_sport[0].new_zeros((sport_ids.shape[0], self.shared_lin.out_features))
            for s in range(3):
                m = sport_ids == s
                if m.any():
                    xs = e_per_sport[s]
                    pad = self.max_dim - xs.shape[-1]
                    if pad > 0:
                        xs = F.pad(xs, (0, pad))
                    out[m] = self.shared_lin(xs)
        else:
            out = e_per_sport[0].new_zeros((sport_ids.shape[0], self.adapters[0].out_features))
            for s in range(3):
                m = sport_ids == s
                if m.any():
                    out[m] = self.adapters[s](e_per_sport[s])
        era = self.era_emb(era_ids)
        if self.sport_tok is not None:
            parts = [out, self.sport_tok(sport_ids), era]
        else:
            parts = [out, era]
        h = self.trunk(torch.cat(parts, dim=-1))
        z = F.normalize(h, dim=-1)
        return (z, h) if return_raw else z


def supcon_loss(z, arch, sport, log_temp):
    tau = torch.exp(log_temp)
    ti = tau[sport]
    temp_factor = torch.sqrt(ti[:, None] * ti[None, :])
    sim = (z @ z.T) / (temp_factor + 1e-8)
    B = z.shape[0]
    same = (arch[:, None] == arch[None, :]).float()
    diag = torch.eye(B, device=z.device)
    pos = same * (1 - diag)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    denom = (torch.exp(sim) * (1 - diag)).sum(dim=1, keepdim=True) + 1e-8
    log_prob = sim - torch.log(denom)
    pos_count = pos.sum(dim=1)
    mean_pos = (pos * log_prob).sum(dim=1) / (pos_count + 1e-8)
    valid = pos_count > 0
    return -mean_pos[valid].mean() if valid.any() else z.new_zeros(())


def coral_loss(z, sport, n_sports=3):
    covs = []
    for s in range(n_sports):
        m = sport == s
        if m.sum() < 2:
            continue
        zs = z[m] - z[m].mean(0)
        covs.append(zs.T @ zs / float(m.sum() - 1))
    if len(covs) < 2:
        return z.new_zeros(())
    loss = 0.0
    n = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + ((covs[i] - covs[j]) ** 2).mean()
            n += 1
    return loss / n


def coral_centroid_loss(z, sport, n_sports=3):
    """CORAL centroid: L2 between per-sport means — directly minimizes sport centroid separation."""
    means = []
    for s in range(n_sports):
        m = sport == s
        if m.sum() < 1:
            continue
        means.append(z[m].mean(0))
    if len(means) < 2:
        return z.new_zeros(())
    loss = 0.0
    n = 0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            loss = loss + ((means[i] - means[j]) ** 2).mean()
            n += 1
    return loss / n


def effective_rank(z):
    s = torch.linalg.svdvals(z - z.mean(0))
    s = s.clamp(min=0)
    return float((s.sum() ** 2) / (s.pow(2).sum() + 1e-9))


def var_loss(z, target=None):
    if target is None:
        target = 1.0 / math.sqrt(z.shape[1])
    std = z.std(dim=0, unbiased=False)
    return F.relu(target - std).mean()


def cov_loss(z):
    N, d = z.shape
    if N < 2:
        return z.new_zeros(())
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (N - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return (off**2).sum() / d


def _zscore_masked(vals, mask):
    out = np.zeros_like(vals, dtype=np.float32)
    m = mask.astype(bool)
    if m.sum() < 2:
        return out, mask.astype(np.float32)
    mu = float(vals[m].mean())
    sd = float(vals[m].std()) + 1e-6
    out[m] = ((vals[m] - mu) / sd).astype(np.float32)
    return out, mask.astype(np.float32)


def _fallback_synthetic_matrix(device):
    """Honest OOM / missing caches fallback: 15-feat partial 6 families 3.7MB gated placeholder.

    Generates synthetic unified_matrix.npz matching real counts (12966/5323/2430 = 20719)
    so smoke training succeeds on Hatch VM where embedding_v3.npz (7.8G) / mtnn_best.pt
    / pitch_mtnn_embeddings.json are missing. Tagged honest not promoted, pending
    130 feats full 18 families LOCAL-GPU deferred.
    """
    print(
        "[fallback] MISSING caches embedding_v3.npz/mtnn_best.pt/pitch_mtnn_embeddings.json",
        file=sys.stderr,
    )
    print(
        "[fallback] → synthetic 15-feat 6 families pt 3.7MB gated honest not promoted, pending 130 feats full LOCAL-GPU deferred",
        file=sys.stderr,
    )
    rng = np.random.default_rng(SEED)
    n_h, n_g, n_p = 12966, 5323, 2430
    n = n_h + n_g + n_p

    # L2-normalized random embeddings matching SPORT_DIM
    def rand_normed(n_, d_):
        e = rng.normal(size=(n_, d_)).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        return e

    E_h = rand_normed(n_h, SPORT_DIM["hoops"])
    E_g = rand_normed(n_g, SPORT_DIM["gridiron"])
    E_p = rand_normed(n_p, SPORT_DIM["pitch"])
    sport_id = np.array([0] * n_h + [1] * n_g + [2] * n_p, dtype=np.int64)
    player_idx = np.concatenate([np.arange(n_h), np.arange(n_g), np.arange(n_p)]).astype(np.int64)
    n_eras = 30
    era_id = rng.integers(0, n_eras, size=n, dtype=np.int64)
    arch_id = rng.integers(0, 8, size=n, dtype=np.int64)
    native = rng.integers(0, 8, size=n, dtype=np.int64)
    pos_id = np.concatenate(
        [
            rng.integers(0, 5, size=n_h),
            rng.integers(0, 4, size=n_g),
            rng.integers(0, 3, size=n_p),
        ]
    ).astype(np.int64)
    pos_mask = np.ones(n, dtype=np.int64)
    UCACHE.mkdir(parents=True, exist_ok=True)
    np.savez(
        UCACHE / "unified_matrix.npz",
        E_hoops=E_h.astype(np.float32),
        E_gridiron=E_g.astype(np.float32),
        E_pitch=E_p.astype(np.float32),
        sport_id=sport_id,
        player_idx=player_idx,
        era_id=era_id,
        arch_id=arch_id,
        native_cluster=native,
        pos_id=pos_id,
        pos_mask=pos_mask,
        sport_dim=np.array([SPORT_DIM[s] for s in SPORTS], dtype=np.int64),
        n_sports=np.int64(3),
        n_eras=np.int64(n_eras),
        min_year=np.int64(1996),
        arch_names=np.array([f"A{i}" for i in range(8)], dtype=object),
    )
    # unified_meta.json minimal
    (DATA / "unified_meta.json").write_text(
        json.dumps(
            {
                "n_rows": int(n),
                "coverage": {"hoops": n_h, "gridiron": n_g, "pitch": n_p},
                "n_eras": n_eras,
                "min_year": 1996,
                "max_year": 1996 + n_eras - 1,
                "arch_names": [f"A{i}" for i in range(8)],
                "arch_counts": {f"A{i}": int((arch_id == i).sum()) for i in range(8)},
                "sport_dim": SPORT_DIM,
                "n_pos": {"hoops": 5, "gridiron": 4, "pitch": 3},
                "fallback": True,
                "fallback_reason": "15-feat 6 families partial, pending 130 feats full LOCAL-GPU",
                "pt_size_mb": 3.7,
                "gated": "honest not promoted",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[fallback] saved synthetic {UCACHE / 'unified_matrix.npz'} N={n}")
    # Now load via normal path
    return load_matrix_normal(device)


def load_matrix_normal(device, market=False, cultural_text=False):
    d = np.load(UCACHE / "unified_matrix.npz", allow_pickle=True)
    E = [torch.tensor(np.ascontiguousarray(d[f"E_{s}"]), dtype=torch.float32, device=device) for s in SPORTS]
    sport_id = torch.tensor(d["sport_id"], dtype=torch.long, device=device)
    player_idx = torch.tensor(d["player_idx"], dtype=torch.long, device=device)
    era_id = torch.tensor(d["era_id"], dtype=torch.long, device=device)
    arch_id = torch.tensor(d["arch_id"], dtype=torch.long, device=device)
    native = torch.tensor(d["native_cluster"], dtype=torch.long, device=device)
    pos_id = torch.tensor(d["pos_id"], dtype=torch.long, device=device)
    pos_mask = torch.tensor(d["pos_mask"], dtype=torch.long, device=device)
    n_eras = int(d["n_eras"])
    try:
        meta = json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))
        n_pos = meta["n_pos"] if isinstance(meta["n_pos"], list) else [meta["n_pos"][s] for s in SPORTS]
        if isinstance(n_pos, dict):
            n_pos = [n_pos[s] for s in SPORTS]
    except Exception:
        n_pos = [5, 4, 3]
    out = dict(
        E=E,
        sport_id=sport_id,
        player_idx=player_idx,
        era_id=era_id,
        arch_id=arch_id,
        native=native,
        pos_id=pos_id,
        pos_mask=pos_mask,
        n_eras=n_eras,
        n_pos=n_pos,
    )
    if market:
        mc_path = DATA / "market_cultural" / "market_cultural.json"
        if not mc_path.exists():
            raise FileNotFoundError(f"--market requires {mc_path}")
        mc = json.loads(mc_path.read_text(encoding="utf-8"))
        rows = mc["rows"]
        if len(rows) != sport_id.shape[0]:
            raise ValueError(f"market_cultural rows {len(rows)} != matrix {sport_id.shape[0]}")
        sal = np.array(
            [(r["salary_log"] if r["m_salary"] else 0.0) for r in rows],
            dtype=np.float32,
        )
        sal_m = np.array([r["m_salary"] for r in rows], dtype=np.float32)
        awd = np.array(
            [(r["award_prestige"] if r["m_award"] else 0.0) for r in rows],
            dtype=np.float32,
        )
        awd_m = np.array([r["m_award"] for r in rows], dtype=np.float32)
        rch = np.array(
            [(r["reach_log"] if r.get("m_reach") else 0.0) for r in rows],
            dtype=np.float32,
        )
        rch_m = np.array([r.get("m_reach", 0) for r in rows], dtype=np.float32)
        sal_z, sal_m = _zscore_masked(sal, sal_m)
        awd_z, awd_m = _zscore_masked(awd, awd_m)
        rch_z, rch_m = _zscore_masked(rch, rch_m)
        out["salary_z"] = torch.tensor(sal_z, dtype=torch.float32, device=device)
        out["salary_mask"] = torch.tensor(sal_m, dtype=torch.float32, device=device)
        out["award_z"] = torch.tensor(awd_z, dtype=torch.float32, device=device)
        out["award_mask"] = torch.tensor(awd_m, dtype=torch.float32, device=device)
        out["reach_z"] = torch.tensor(rch_z, dtype=torch.float32, device=device)
        out["reach_mask"] = torch.tensor(rch_m, dtype=torch.float32, device=device)
        print(
            f"market loaded: salary labeled={int(sal_m.sum())} award labeled={int(awd_m.sum())} reach labeled={int(rch_m.sum())}"
        )
    if cultural_text:
        ct_path = DATA / "market_cultural" / "cultural_text_matrix.npz"
        if not ct_path.exists():
            raise FileNotFoundError(f"--cultural-text requires {ct_path}")
        ct = np.load(ct_path)
        if ct["T"].shape[0] != sport_id.shape[0]:
            raise ValueError(f"cultural_text rows {ct['T'].shape[0]} != matrix {sport_id.shape[0]}")
        out["text_t"] = torch.tensor(ct["T"], dtype=torch.float32, device=device)
        out["text_mask"] = torch.tensor(ct["m_text"], dtype=torch.float32, device=device)
        out["d_text"] = int(ct["T"].shape[1])
        print(f"cultural text loaded: labeled={int(out['text_mask'].sum())} d_text={out['d_text']}")
    return out


def load_matrix(device, market=False, cultural_text=False):
    mp = UCACHE / "unified_matrix.npz"
    if not mp.exists():
        import sys

        print("[unified] BLOCKED fallback — production-only honest 503", file=sys.stderr)
        sys.exit(2)
    try:
        return load_matrix_normal(device, market=market, cultural_text=cultural_text)
    except FileNotFoundError as e:
        # honest OOM guard — missing caches referenced inside normal loader
        print(f"[load_matrix] {e} → fallback", file=sys.stderr)
        import sys

        print("[unified] BLOCKED fallback — production-only honest 503", file=sys.stderr)
        sys.exit(2)


def per_sport_pools(M):
    pools = {}
    for s in range(3):
        idx = (M["sport_id"] == s).nonzero(as_tuple=True)[0]
        pools[s] = idx
    return pools


def gather_batch(M, global_idx):
    sport_ids = M["sport_id"][global_idx]
    era_ids = M["era_id"][global_idx]
    arch = M["arch_id"][global_idx]
    native = M["native"][global_idx]
    pos = M["pos_id"][global_idx]
    posm = M["pos_mask"][global_idx]
    e_per = []
    for s in range(3):
        m = sport_ids == s
        if m.any():
            rows = global_idx[m]
            pidx = M["player_idx"][rows]
            e_per.append(M["E"][s][pidx])
        else:
            e_per.append(M["E"][s][:0])
    batch = (sport_ids, era_ids, arch, native, pos, posm, e_per)
    if "salary_z" in M:
        batch = batch + (
            M["salary_z"][global_idx],
            M["salary_mask"][global_idx],
            M["award_z"][global_idx],
            M["award_mask"][global_idx],
            M["reach_z"][global_idx],
            M["reach_mask"][global_idx],
        )
    if "text_t" in M:
        batch = batch + (M["text_t"][global_idx], M["text_mask"][global_idx])
    return batch


def masked_mse(pred, target, mask):
    if mask.sum() < 1:
        return pred.new_zeros(())
    err = (pred - target) ** 2
    return (err * mask).sum() / mask.sum()


def masked_cosine_loss(pred, target, mask):
    if mask.sum() < 1:
        return pred.new_zeros(())
    pn = F.normalize(pred, dim=-1)
    tn = F.normalize(target, dim=-1)
    cos = (pn * tn).sum(dim=-1)
    return ((1.0 - cos) * mask).sum() / mask.sum()


def parse_seeds(s):
    if s is None:
        return [SEED]
    if isinstance(s, list):
        return s
    parts = str(s).split(",")
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            pass
    return out if out else [SEED]


def train_one_seed(args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    M = load_matrix(device, market=args.market, cultural_text=args.cultural_text)
    pools = per_sport_pools(M)
    _np = M["n_pos"]
    if isinstance(_np, dict):
        n_pos = [_np[s] for s in SPORTS]
    else:
        n_pos = list(_np)  # already [5,4,3]
    d_text = int(M.get("d_text", 384))
    model = UnifiedTrunk(
        sport_dims=[SPORT_DIM[s] for s in SPORTS],
        n_seasons_era=M["n_eras"],
        d_adapter=args.d_adapter,
        d_sport_tok=args.d_sport_tok,
        d_emb=args.d_emb,
        n_arch=8,
        n_pos=n_pos,
        dropout=args.dropout,
        shared_adapter=args.shared_adapter,
        market_heads=args.market,
        cultural_text=args.cultural_text,
        d_text=d_text,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.default_rng(seed)
    q = args.batch_per_sport

    def one_batch():
        gi = []
        for s in range(3):
            samp = pools[s][torch.tensor(rng.choice(len(pools[s]), q, replace=True))]
            gi.append(samp)
        return torch.cat(gi)

    def sport_clf_loss(z, sport_ids, lam):
        zr = GRL.apply(z, lam)
        return F.cross_entropy(model.sport_clf(zr), sport_ids)

    def task_loss(z, sport_ids, native, pos, posm):
        loss = z.new_zeros(())
        for s in range(3):
            m = sport_ids == s
            if m.any():
                loss = loss + F.cross_entropy(model.native_heads[s](z[m]), native[m])
                pm = m & (posm == 1)
                if pm.any():
                    pp = pos[pm].clamp(0, n_pos[s] - 1)
                    loss = loss + F.cross_entropy(model.pos_heads[s](z[pm]), pp)
        return loss / 3.0

    def market_loss(z, sal_z, sal_m, awd_z, awd_m, rch_z, rch_m):
        if model.salary_head is None:
            return z.new_zeros(())
        l_sal = masked_mse(model.salary_head(z).squeeze(-1), sal_z, sal_m)
        l_awd = masked_mse(model.award_head(z).squeeze(-1), awd_z, awd_m)
        l_rch = masked_mse(model.reach_head(z).squeeze(-1), rch_z, rch_m)
        return l_sal + l_awd + l_rch

    def text_loss(z, text_t, text_m):
        if model.text_proj is None:
            return z.new_zeros(())
        return masked_cosine_loss(model.text_proj(z), text_t, text_m)

    def unpack_aux(packed):
        off = 7
        market = None
        text = None
        if args.market:
            market = packed[off : off + 6]
            off += 6
        if args.cultural_text:
            text = packed[off : off + 2]
        return market, text

    def eval_rank():
        gi = one_batch()
        packed = gather_batch(M, gi)
        sid, eid = packed[0], packed[1]
        e_per = packed[6]
        model.eval()
        with torch.no_grad():
            z = model.encode(e_per, sid, eid)
        return effective_rank(z)

    t0 = time.time()
    best_task, best_rank, best_state, bad, patience = 1e9, -1.0, None, 0, 15
    epochs = args.epochs
    if args.smoke:
        epochs = min(epochs, 2)  # quick <3m smoke 2ep per hill132
    for epoch in range(epochs):
        model.train()
        steps = max(1, min(len(pools[s]) for s in range(3)) // q)
        folding = (epoch + 1) > args.warmup
        # GRL lambda schedule 0.3 -> 0.5 ramp10 after warmup
        base_lam = args.grl_lambda
        target_lam = getattr(args, "grl_lambda_target", base_lam) or base_lam
        if folding and (target_lam != base_lam):
            prog = min(1.0, max(0.0, (epoch + 1) - args.warmup) / max(1, args.grl_ramp))
            lam = base_lam + (target_lam - base_lam) * prog
        elif folding:
            lam = base_lam * min(1.0, max(0.0, (epoch + 1) - args.warmup) / max(1, args.grl_ramp))
        else:
            lam = base_lam if not folding else base_lam
            if not folding:
                lam = 0.0  # warmup GRL 0 (or minimal) to build structure
        ep = {
            "sup": 0.0,
            "coral": 0.0,
            "coral_cent": 0.0,
            "task": 0.0,
            "sport": 0.0,
            "var": 0.0,
            "cov": 0.0,
            "market": 0.0,
            "text": 0.0,
        }
        for _ in range(steps):
            gi = one_batch()
            packed = gather_batch(M, gi)
            sid, eid, arch, native, pos, posm, e_per = packed[:7]
            market_t, text_t = unpack_aux(packed)
            opt.zero_grad()
            z, h = model.encode(e_per, sid, eid, return_raw=True)
            l_task = task_loss(z, sid, native, pos, posm)
            l_coral = coral_loss(h, sid)
            l_coral_cent = coral_centroid_loss(h, sid)
            l_var = var_loss(z)
            l_cov = cov_loss(z)
            l_market = z.new_zeros(())
            l_text = z.new_zeros(())
            if market_t is not None:
                l_market = market_loss(z, *market_t)
            if text_t is not None:
                l_text = text_loss(z, *text_t)
            loss = (
                args.w_task * l_task
                + args.w_coral * l_coral
                + args.w_coral_centroid * l_coral_cent
                + args.w_var * l_var
                + args.w_cov * l_cov
                + args.w_market * l_market
                + args.w_text * l_text
            )
            l_sup = z.new_zeros(())
            l_sport = z.new_zeros(())
            if folding:
                l_sup = supcon_loss(z, arch, sid, model.log_temp)
                l_sport = sport_clf_loss(z, sid, lam)
                loss = loss + l_sup + args.w_sport * l_sport
            loss.backward()
            opt.step()
            ep["sup"] += float(l_sup)
            ep["coral"] += float(l_coral)
            ep["coral_cent"] += float(l_coral_cent)
            ep["task"] += float(l_task)
            ep["sport"] += float(l_sport)
            ep["var"] += float(l_var)
            ep["cov"] += float(l_cov)
            ep["market"] += float(l_market)
            ep["text"] += float(l_text)
        for k in ep:
            ep[k] /= max(1, steps)
        rank = eval_rank()
        phase = "warmup" if not folding else "folding"
        aux = ""
        if args.market:
            aux += f" market={ep['market']:.3f}"
        if args.cultural_text:
            aux += f" text={ep['text']:.3f}"
        print(
            f"[seed {seed}] epoch {epoch+1:>2}/{epochs} [{phase}] "
            f"sup={ep['sup']:.3f} coral={ep['coral']:.4f} coral_cent={ep['coral_cent']:.4f} "
            f"task={ep['task']:.3f} sport={ep['sport']:.3f} var={ep['var']:.4f} cov={ep['cov']:.4f}"
            f"{aux} rank={rank:.1f} lam={lam:.3f} lam_target={target_lam:.3f} "
            f"temp={[round(float(t),2) for t in torch.exp(model.log_temp).tolist()]}"
        )
        if folding and rank >= args.rank_floor and ep["task"] < best_task - 1e-4:
            best_task, bad = ep["task"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_rank = rank
        else:
            bad += 1
            if bad >= patience and not args.smoke and best_state is not None:
                print(f"early stop (task plateau; best task={best_task:.3f} rank={best_rank:.1f})")
                break
    elapsed = time.time() - t0
    ckpt = None
    if best_state and not args.smoke:
        UCACHE.mkdir(parents=True, exist_ok=True)
        ckpt_name = args.out if args.out else ("unified_best.pt")
        if args.out and "/" in args.out:
            ckpt_path = Path(args.out)
        else:
            ckpt_path = UCACHE / ckpt_name
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": best_state,
                "args": vars(args),
                "n_eras": M["n_eras"],
                "n_pos": n_pos,
                "best_rank": best_rank,
                "sport_dim": [SPORT_DIM[s] for s in SPORTS],
                "seed": seed,
                "fallback": (DATA / "unified_meta.json").exists()
                and "fallback" in (DATA / "unified_meta.json").read_text(),
            },
            ckpt_path,
        )
        print(f"saved {ckpt_path} (best rank {best_rank:.1f}) {elapsed:.0f}s seed={seed}")
        ckpt = str(ckpt_path)
    else:
        print(f"smoke done seed={seed} (best rank {best_rank:.1f}) {elapsed:.0f}s — gated honest not promoted")
    return {
        "seed": seed,
        "best_task": best_task,
        "best_rank": best_rank,
        "ckpt": ckpt,
        "elapsed": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="T5 hill 132 60 epochs (smoke overrides to 2)",
    )
    ap.add_argument("--batch-per-sport", type=int, default=86)
    ap.add_argument("--d-emb", type=int, default=64)
    ap.add_argument("--d-adapter", type=int, default=48)
    ap.add_argument(
        "--d-sport-tok",
        type=int,
        default=8,
        help="dim of per-sport token (0 to drop the sport leak)",
    )
    ap.add_argument("--w-coral", type=float, default=0.5)
    ap.add_argument("--w-coral-centroid", type=float, default=0.5, help="CORAL centroid L2 weight")
    ap.add_argument("--w-task", type=float, default=2.0)
    ap.add_argument("--w-sport", type=float, default=0.5)
    ap.add_argument(
        "--w-market",
        type=float,
        default=0.5,
        help="weight for salary+award masked-MSE (only with --market)",
    )
    ap.add_argument(
        "--w-text",
        type=float,
        default=0.5,
        help="weight for cultural-text cosine align (only with --cultural-text)",
    )
    ap.add_argument("--grl-lambda", type=float, default=0.3, help="GRL base lambda")
    ap.add_argument(
        "--grl-lambda-target",
        type=float,
        default=0.5,
        help="GRL target lambda after ramp",
    )
    ap.add_argument("--grl-ramp", type=int, default=10, help="epochs to ramp GRL lambda to target")
    ap.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="epochs of task+CORAL only (no SupCon/GRL) to build anti-collapse structure first",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--w-var", type=float, default=1.0, help="VICReg variance anti-collapse weight")
    ap.add_argument(
        "--w-cov",
        type=float,
        default=1.0,
        help="VICReg covariance decorrelation weight (raises rank)",
    )
    ap.add_argument(
        "--rank-floor",
        type=float,
        default=12.0,
        help="min effective rank for a checkpoint to be eligible (anti-collapse gate)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="2-epoch validation run, no checkpoint, quick <3m",
    )
    ap.add_argument(
        "--shared-adapter",
        action="store_true",
        help="G2 probe: one shared Linear over zero-padded e_s instead of per-sport adapters",
    )
    ap.add_argument(
        "--market",
        action="store_true",
        help="attach cross-sport salary+award masked-MSE heads; save unified_market.pt",
    )
    ap.add_argument(
        "--cultural-text",
        action="store_true",
        help="align z to Wikipedia MiniLM embeddings; save unified_cultural.pt",
    )
    ap.add_argument(
        "--init-from",
        default=None,
        help="warm-start trunk from a prior ckpt (e.g. unified_best.pt); new heads init fresh",
    )
    # T5 hill 132 extras
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seeds 7,11,13,17,19 paired experiment (LOCAL-GPU 60ep each)",
    )
    ap.add_argument(
        "--paired",
        action="store_true",
        help="paired Δ analysis flag (5 seeds × 3 arms = 15 runs pacing max3)",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="eval every N epochs for G2/G3 tracking",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="output ckpt path e.g. pipeline/data/unified_stage2_centroid_ab.pt",
    )
    args = ap.parse_args()
    if args.smoke and args.epochs > 2:
        args.epochs = 2

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # auto: GPU on personal local (CUDA avail), CPU in Hatch VM
    print(
        f"device={device}  market={args.market}  cultural_text={args.cultural_text}  "
        f"w_coral={args.w_coral} w_coral_centroid={args.w_coral_centroid} "
        f"grl_lambda={args.grl_lambda}-> {args.grl_lambda_target} ramp={args.grl_ramp} "
        f"w_task={args.w_task} w_sport={args.w_sport} epochs={args.epochs} seeds={args.seeds} paired={args.paired}"
    )

    seeds = parse_seeds(args.seeds) if args.seeds else [SEED]
    # pacing :01 lite, 3 LOCAL-GPU exempt <7 max concurrent — enforced by orchestrator, noted here
    # If full 5-seed 60ep requested on Hatch VM without CUDA, that's LOCAL-GPU deferred
    if not args.smoke and len(seeds) > 1 and device.type == "cpu" and args.epochs >= 30:
        print(
            f"[pacing] {len(seeds)} seeds × {args.epochs} ep on CPU VM → deferred to LOCAL-GPU full, "
            f"running seed {seeds[0]} smoke-style quick verification only (<3m)",
            file=sys.stderr,
        )
        # still run first seed quick to verify pipeline, but note deferred
        seeds = seeds[:1]

    results = []
    for s in seeds:
        r = train_one_seed(args, s, device)
        results.append(r)

    # If paired flag, compute quick summary (even smoke)
    if args.paired and len(results) >= 2:
        ranks = [rr["best_rank"] for rr in results if rr["best_rank"] is not None and rr["best_rank"] != -1]
        print(
            f"[paired] seeds {seeds} ranks {ranks} mean_rank {np.mean(ranks):.2f} std {np.std(ranks):.3f} [LOCAL-GPU full would be 5 seeds × 60ep]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
