"""Cross-domain transfer probe for the unified domain, end to end on REAL data.

The registry's unified spec asks one question per target: does a shared
embedding trained WITHOUT the held-out domain still carry signal for that
domain's forward target, better than baselines that never saw other domains?

Protocol (per transfer target, all seeded, CPU, 2 threads):

 1. Load the five sibling lanes' REAL exchange datasets
    (bench/data/exchange/<domain>/dataset.npz — committed snapshots of the
    lanes' verified real-data artifacts: hoops NBA next-season stats, gridiron
    nflverse weekly, equities SEC-EDGAR+Yahoo forward windows, realty BIS
    property prices, pitch FBref WSL windows).
 2. Schema-align every domain into a common 16-d input space: per-domain
    vector_core RobustScaler -> PCA(16, full SVD) -> per-component whitening,
    ALL fit on that domain's own train rows only.
 3. Train ONE shared trunk (16 -> 64 -> 64 -> 32-d L2-normalized embedding,
    GELU) jointly on the FOUR domains that are not held out, with one linear
    head per (domain, wired target) — masked MSE on train-z-scored regression
    targets, masked BCE on binary ones — plus the repo's CORAL alignment
    penalty (train_unified.py's second-order geometry match) pulling the four
    domains' embedding distributions together. Gradient steps use each
    domain's train rows ONLY; early stopping reads pooled val-row loss ONLY.
    The held-out domain's rows are never loaded during embedding training.
 4. Freeze the trunk. Map the held-out domain's harness features through its
    OWN RobustScaler+PCA+whiten (fit on the harness train side only — an
    unsupervised mapping, no labels) into the shared input space, embed, and
    fit ONLY a linear head (sklearn Ridge, alpha=1.0, closed form) on the
    harness train side. Predict the harness test side -> the MTNN rung.
 5. Run the held-out domain's standard gauntlet on its raw harness features —
    the exact task construction its owner lane used (same rows, same temporal
    cut, same extra persistence rung for hoops) — with the frozen-embedding
    probe slotted in as the MTNN rung, plus a 'pca16_whiten_ridge' control
    rung (the identical 16-d whitened input with the identical linear head,
    isolating what the cross-domain trunk adds over its own input encoding).
 6. Write the schema-1.1 domain report, the training config, and the unified
    exchange artifact (frozen embeddings + labels + splits).

Leakage discipline
------------------
- Held-out split is the owner lane's, reproduced exactly: equities temporal
  cut fy>=2022 (seed 42), hoops temporal cut target_year>=2026 (seed 7).
- Embedding training never forward-passes a held-out-domain row; training
  domains contribute gradient rows from their committed train_idx only and
  early-stop rows from their committed val_idx only (their test rows are
  never forward-passed either).
- Every scaler/PCA/whitener is fit on train rows only (per training domain:
  its train_idx; for the held-out probe: the harness train side).
- The probe head is closed-form ridge on the harness train side — the same
  rows every ladder baseline fits on. No tuning on test anywhere; the probe
  alpha is fixed at 1.0 a priori (the ladder ridge default), and the trunk
  hyperparameters were fixed a priori (no grid was run).

Run:
    python bench/run_transfer_benchmark.py [--exchange-in bench/data/exchange]
        [--report bench/benchmark_report.json] [--exchange-out bench/data]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SEED = 0  # embedding training + shared ladder seed
D_ALIGN = 16  # common aligned input dim (min domain width is realty's 18)
D_HIDDEN = 64
D_EMB = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 40
CORAL_WEIGHT = 0.1
RIDGE_ALPHA = 1.0  # probe head; fixed a priori == ladder ridge default

# Owner-lane harness conventions, reproduced exactly.
EQUITIES_SEED, EQUITIES_CUT = 42, 2022  # vector-equities bench/run_benchmark.py
HOOPS_SEED, HOOPS_CUT = 7, 2026  # vector-hoops bench/run_benchmark.py

# domain -> (target name, kind) trained as embedding heads
DOMAIN_TARGETS: dict[str, list[tuple[str, str]]] = {
    "hoops": [
        (t, "regression")
        for t in (
            "next_season_per",
            "next_season_win_shares",
            "next_season_bpm",
            "next_season_pts",
            "next_season_reb",
            "next_season_ast",
        )
    ],
    "gridiron": [(t, "regression") for t in ("next_game_fpts", "next_game_yards", "next_game_tds")],
    "equities": [
        ("forward_return", "regression"),
        ("forward_realized_vol", "regression"),
        ("drawdown_exceedance", "binary"),
    ],
    "realty": [
        ("next_year_price_change", "regression"),
        ("three_year_price_change", "regression"),
        ("above_market_appreciation", "binary"),
    ],
    "pitch": [
        ("next_window_minutes", "regression"),
        ("next_window_goal_contribution", "regression"),
    ],
}

TRANSFER = {  # transfer target -> (held-out domain, owner target)
    "transfer_forward_return": ("equities", "forward_return"),
    "transfer_next_season_per": ("hoops", "next_season_per"),
}


# --------------------------------------------------------------------------- #
# Exchange loading: one uniform view per domain
# --------------------------------------------------------------------------- #
def _standardize_train_only(X_raw, X_mask, train_rows):
    """vector-equities' harness feature prep, verbatim: mask->NaN, train-median
    impute, train-stat z-score, clip +-5."""
    X = X_raw.astype(np.float64).copy()
    X[X_mask <= 0] = np.nan
    med = np.nanmedian(X[train_rows], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for j in range(X.shape[1]):
        col = X[:, j]
        col[np.isnan(col)] = med[j]
    mu = X[train_rows].mean(axis=0)
    sd = X[train_rows].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return np.clip((X - mu) / sd, -5.0, 5.0)


def load_domain(name: str, ex_dir: Path) -> dict:
    """Uniform view: X_model (float64, finite), train/val idx, per-target labels."""
    z = np.load(ex_dir / name / "dataset.npz", allow_pickle=True)
    d: dict = {"name": name}
    if name == "hoops":
        d["train"], d["val"] = z["split_train"], z["split_val"]
        d["X_model"] = z["X"].astype(np.float64)  # exchange X is harness-ready
    elif name == "equities":
        d["train"], d["val"] = z["train_idx"], z["val_idx"]
        # exchange X is RAW; impute+standardize on the MTNN train rows only
        d["X_model"] = _standardize_train_only(z["X"], z["X_mask"], z["train_idx"])
    else:
        d["train"], d["val"] = z["train_idx"], z["val_idx"]
        d["X_model"] = z["X"].astype(np.float64)
    assert np.isfinite(d["X_model"]).all(), f"{name}: non-finite model features"
    d["targets"] = {}
    for tname, kind in DOMAIN_TARGETS[name]:
        y = z[f"y_{tname}"].astype(np.float64)
        mk = None
        for cand in (f"mask_{tname}", f"label_mask_{tname}"):
            if cand in z.files:
                mk = z[cand].astype(bool)
        assert mk is not None, f"{name}:{tname} has no mask"
        assert not np.isnan(y[mk]).any(), f"{name}:{tname} NaN labels under mask"
        d["targets"][tname] = (y, mk, kind)
    d["npz"] = z
    return d


# --------------------------------------------------------------------------- #
# Schema alignment: RobustScaler -> PCA(16) -> whiten, fit on given rows only
# --------------------------------------------------------------------------- #
class PCAWhiten:
    """The shared-input alignment map. Everything is fit on ``fit_rows`` only."""

    def __init__(self, n_components: int = D_ALIGN, seed: int = SEED):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X: np.ndarray, fit_rows: np.ndarray) -> PCAWhiten:
        from sklearn.decomposition import PCA
        from vector_core.preproc import RobustScaler

        Xf = np.asarray(X, dtype=np.float64)[fit_rows]
        self.scaler = RobustScaler()
        S = self.scaler.fit_transform(Xf)
        n_comp = min(self.n_components, S.shape[0], S.shape[1])
        self.pca = PCA(n_components=n_comp, svd_solver="full", random_state=self.seed)
        U = self.pca.fit_transform(S)
        sd = U.std(axis=0)
        self.whiten_sd = np.where(sd < 1e-9, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        U = self.pca.transform(self.scaler.transform(np.asarray(X, dtype=np.float64)))
        U = U / self.whiten_sd
        if U.shape[1] < self.n_components:  # pad degenerate domains (none in practice)
            U = np.pad(U, ((0, 0), (0, self.n_components - U.shape[1])))
        return U


# --------------------------------------------------------------------------- #
# The shared-embedding MTNN (torch), trained on the non-held-out domains
# --------------------------------------------------------------------------- #
def train_shared_embedding(domains: list[dict], seed: int = SEED):
    """Joint multi-domain multi-task training. Returns (trunk_fn, info).

    ``trunk_fn(U)`` maps aligned 16-d inputs to the frozen 32-d L2 embedding.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    class SharedTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(D_ALIGN, D_HIDDEN),
                nn.GELU(),
                nn.Linear(D_HIDDEN, D_HIDDEN),
                nn.GELU(),
                nn.Linear(D_HIDDEN, D_EMB),
            )

        def forward(self, u):
            return F.normalize(self.net(u), dim=-1)

    trunk = SharedTrunk()
    heads = nn.ModuleDict()
    pack = []  # per-domain tensors
    for d in domains:
        name = d["name"]
        aligner = PCAWhiten().fit(d["X_model"], d["train"])
        # never forward-pass a training domain's TEST rows: keep train+val only
        rows = np.sort(np.concatenate([d["train"], d["val"]]))
        pos = {int(r): i for i, r in enumerate(rows)}
        U = torch.tensor(aligner.transform(d["X_model"][rows]), dtype=torch.float32)
        tr_l = torch.tensor([pos[int(r)] for r in d["train"]], dtype=torch.long)
        va_l = torch.tensor([pos[int(r)] for r in d["val"]], dtype=torch.long)
        tgt = {}
        for tname, (y, mk, kind) in d["targets"].items():
            if kind == "regression":  # z-score on train labeled rows only
                lab = d["train"][mk[d["train"]]]
                mu, sd = float(y[lab].mean()), float(y[lab].std() + 1e-9)
            else:
                mu, sd = 0.0, 1.0
            yz = (np.where(mk, y, 0.0) - mu) / sd
            tgt[tname] = (
                torch.tensor(yz[rows], dtype=torch.float32),
                torch.tensor(mk[rows]),
                kind,
            )
            heads[f"{name}::{tname}"] = nn.Linear(D_EMB, 1)
        pack.append(
            {
                "name": name,
                "U": U,
                "tr": tr_l,
                "va": va_l,
                "tgt": tgt,
                "aligner": aligner,
            }
        )

    params = list(trunk.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    def head_loss(dpack, E, sel):
        total, k = 0.0, 0
        for tname, (yz, mk, kind) in dpack["tgt"].items():
            rows = sel[mk[sel]]
            if len(rows) == 0:
                continue
            out = heads[f"{dpack['name']}::{tname}"](E[rows]).squeeze(-1)
            if kind == "binary":
                total = total + F.binary_cross_entropy_with_logits(out, yz[rows])
            else:
                total = total + F.mse_loss(out, yz[rows])
            k += 1
        return total / max(k, 1)

    def coral(embs):  # mean+covariance alignment across domains (train rows)
        stats = []
        for E in embs:
            mu = E.mean(dim=0)
            Ec = E - mu
            C = (Ec.T @ Ec) / max(E.shape[0] - 1, 1)
            stats.append((mu, C))
        tot, n = 0.0, 0
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                tot = tot + ((stats[i][1] - stats[j][1]) ** 2).sum() / (4 * D_EMB**2)
                tot = tot + ((stats[i][0] - stats[j][0]) ** 2).sum()
                n += 1
        return tot / max(n, 1)

    best_val, best_state, best_epoch, epochs_run = float("inf"), None, -1, 0
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        trunk.train()
        heads.train()
        opt.zero_grad()
        embs_tr, loss = [], 0.0
        for dpack in pack:
            E = trunk(dpack["U"])  # train+val rows only ever exist in U
            loss = loss + head_loss(dpack, E, dpack["tr"])
            embs_tr.append(E[dpack["tr"]])
        loss = loss / len(pack) + CORAL_WEIGHT * coral(embs_tr)
        loss.backward()
        opt.step()

        trunk.eval()
        heads.eval()
        with torch.no_grad():
            vloss = 0.0
            for dpack in pack:
                E = trunk(dpack["U"])
                vloss = vloss + float(head_loss(dpack, E, dpack["va"]))
            vloss /= len(pack)
        epochs_run = ep + 1
        if vloss < best_val - 1e-5:
            best_val, best_epoch = vloss, ep + 1
            best_state = {
                "trunk": {k: v.clone() for k, v in trunk.state_dict().items()},
                "heads": {k: v.clone() for k, v in heads.state_dict().items()},
            }
        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"  [emb] epoch {ep + 1:3d} train={float(loss.detach()):.4f} "
                f"val={vloss:.4f} best={best_val:.4f}@{best_epoch} "
                f"({time.time() - t0:.0f}s)"
            )
        if ep + 1 - best_epoch >= PATIENCE:
            print(f"  [emb] early stop at epoch {ep + 1}")
            break
    trunk.load_state_dict(best_state["trunk"])
    heads.load_state_dict(best_state["heads"])
    trunk.eval()

    def trunk_fn(U: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return trunk(torch.tensor(np.asarray(U), dtype=torch.float32)).numpy()

    info = {
        "trained_on": [d["name"] for d in domains],
        "heads": sorted(heads.keys()),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_val_loss": round(best_val, 6),
        "params_total": int(sum(p.numel() for p in params)),
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(
        f"  [emb] done: best val {best_val:.4f} @ epoch {best_epoch}, "
        f"{info['params_total']} params, {info['wall_seconds']}s"
    )
    return trunk_fn, info


# --------------------------------------------------------------------------- #
# Held-out harness tasks — the owner lanes' constructions, reproduced exactly
# --------------------------------------------------------------------------- #
def equities_task_arrays(z):
    """rows/X/y/group/time for forward_return, per vector-equities bench."""
    fy = z["time_id"]
    harness_train = fy < EQUITIES_CUT
    X = _standardize_train_only(z["X"], z["X_mask"], np.where(harness_train)[0])
    mk = z["mask_forward_return"].astype(bool)
    rows = np.where(mk)[0]
    y = z["y_forward_return"].astype(np.float64)[rows]
    return rows, X[rows], y, z["entity_id"][rows], fy[rows].astype(np.int64)


def hoops_task_arrays(z):
    """rows/X/y/group/time for next_season_per, per vector-hoops bench."""
    mk = z["label_mask_next_season_per"].astype(bool)
    rows = np.where(mk)[0]
    X = z["X"].astype(np.float64)[rows]  # exchange X is harness-ready already
    y = z["y_next_season_per"].astype(np.float64)[rows]
    return rows, X, y, z["entity_id"][rows], z["target_year"][rows].astype(np.int64)


# --------------------------------------------------------------------------- #
# Extra rungs
# --------------------------------------------------------------------------- #
def make_pca_whiten_ridge():
    """Control rung: the trunk's exact 16-d whitened input + the probe's exact
    linear head, with no cross-domain trunk in between. If this ties the MTNN
    rung, the transferred trunk added nothing beyond its input encoding."""
    from vector_bench.baselines import PredictionBaseline

    class PCAWhitenRidge(PredictionBaseline):
        name = f"pca{D_ALIGN}_whiten_ridge"

        def fit(self, X, y, **ctx):
            from sklearn.linear_model import Ridge as _Ridge

            X = np.asarray(X, dtype=np.float64)
            self._aligner = PCAWhiten().fit(X, np.arange(X.shape[0]))
            self._ridge = _Ridge(alpha=RIDGE_ALPHA)
            self._ridge.fit(self._aligner.transform(X), np.asarray(y, dtype=np.float64))
            return self

        def predict(self, X, **ctx):
            return self._ridge.predict(self._aligner.transform(X))

    return PCAWhitenRidge()


def make_persistence_current_per(hoops_task_y_train, cur_per_col):
    """vector-hoops' persistence_current_per rung (predict this season's raw
    PER), fingerprint-gated so it only runs on the hoops-derived task; on the
    equities task it reports 'skipped' instead of emitting a nonsense column."""
    from vector_bench.baselines import PredictionBaseline

    class PersistenceCurrentPer(PredictionBaseline):
        name = "persistence_current_per"

        def fit(self, X, y, **ctx):
            y = np.asarray(y, dtype=np.float64)
            ref = np.asarray(hoops_task_y_train, dtype=np.float64)
            if y.shape != ref.shape or not np.array_equal(y, ref):
                raise ImportError("only applicable to the hoops-derived task")
            return self

        def predict(self, X, **ctx):
            return np.asarray(X, dtype=np.float64)[:, cur_per_col]

    return PersistenceCurrentPer()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exchange-in", type=Path, default=ROOT / "bench" / "data" / "exchange")
    ap.add_argument("--report", type=Path, default=ROOT / "bench" / "benchmark_report.json")
    ap.add_argument("--config-out", type=Path, default=ROOT / "bench" / "training_config.json")
    ap.add_argument("--exchange-out", type=Path, default=ROOT / "bench" / "data")
    args = ap.parse_args(argv)

    from vector_bench.baselines import MTNNRung, default_prediction_ladder
    from vector_bench.registry import get_domain_spec
    from vector_bench.report import write_domain_report
    from vector_bench.runner import run_domain_benchmark
    from vector_bench.tasks import build_task_for_target, temporal_split

    spec = get_domain_spec("unified")
    all_domains = {n: load_domain(n, args.exchange_in) for n in DOMAIN_TARGETS}

    # -- held-out harness tasks (owner constructions) --
    eq = all_domains["equities"]["npz"]
    ho = all_domains["hoops"]["npz"]
    task_arrays = {
        "transfer_forward_return": equities_task_arrays(eq) + (EQUITIES_CUT, EQUITIES_SEED),
        "transfer_next_season_per": hoops_task_arrays(ho) + (HOOPS_CUT, HOOPS_SEED),
    }

    tasks, mtnns, run_infos, probe_notes = {}, {}, {}, {}
    exchange_rows = []
    for tname, (held_out, owner_target) in TRANSFER.items():
        print(f"\n== {tname}: hold out {held_out}, train embedding on the rest ==")
        train_domains = [all_domains[n] for n in DOMAIN_TARGETS if n != held_out]
        trunk_fn, info = train_shared_embedding(train_domains, seed=SEED)
        run_infos[tname] = info

        rows, X_task, y_task, group, tkey, cut, owner_seed = task_arrays[tname]
        split = temporal_split(tkey, cut=cut)
        # -- probe: unsupervised alignment fit on harness-train rows only --
        aligner = PCAWhiten().fit(X_task, split.train_idx)
        Z = trunk_fn(aligner.transform(X_task))
        from sklearn.linear_model import Ridge as _Ridge

        head = _Ridge(alpha=RIDGE_ALPHA)
        head.fit(Z[split.train_idx], y_task[split.train_idx])
        preds = head.predict(Z[split.test_idx])

        tasks[tname] = build_task_for_target(
            spec.target(tname),
            "unified",
            X=X_task,
            y=y_task,
            group_key=group,
            time_key=tkey,
            time_cut=cut,
            seed=owner_seed,
            extra_notes={
                "held_out_domain": held_out,
                "owner_target": owner_target,
                "rows_labeled": str(len(rows)),
                "embedding_trained_on": ",".join(info["trained_on"]),
                "embedding_best_epoch": str(info["best_epoch"]),
                "data": "REAL sibling-lane exchange datasets " "(bench/data/exchange/*/datasheet.json)",
                "probe": f"frozen 32-d embedding + Ridge(alpha={RIDGE_ALPHA}) fit " "on the harness train side only",
            },
        )
        mtnns[tname] = MTNNRung(predictions=preds)
        probe_notes[tname] = {
            "n_rows": len(rows),
            "n_train": len(split.train_idx),
            "n_test": len(split.test_idx),
            "cut": cut,
            "owner_seed": owner_seed,
        }
        print(f"  probe: rows={len(rows)} train={len(split.train_idx)} " f"test={len(split.test_idx)} (cut {cut})")

        # -- unified exchange rows: frozen embedding + labels + split --
        exchange_rows.append(
            {
                "target": tname,
                "domain": held_out,
                "Z": Z.astype(np.float32),
                "y": y_task,
                "group": group,
                "tkey": tkey,
                "train": split.train_idx,
                "test": split.test_idx,
            }
        )

    # -- shared ladder: defaults + hoops persistence rung + alignment control --
    _, _, y_h, _, tk_h, cut_h, _ = task_arrays["transfer_next_season_per"]
    ho_y_train = y_h[temporal_split(tk_h, cut=cut_h).train_idx]
    feature_names = [str(s) for s in ho["feature_names"]]
    ladder = [
        *default_prediction_ladder(seed=SEED),
        make_pca_whiten_ridge(),
        make_persistence_current_per(ho_y_train, feature_names.index("cur_per")),
    ]

    dsc = run_domain_benchmark(spec, tasks, mtnns=mtnns, ladder=ladder)
    dsc.notes["embedding_training"] = json.dumps(run_infos)
    dsc.notes["protocol"] = (
        "MTNN rung = frozen cross-domain embedding (trained without the "
        "held-out domain) + linear probe on the held-out harness train side; "
        "baselines run on the held-out domain's raw harness features with the "
        "owner lane's exact task construction."
    )
    write_domain_report(dsc, args.report)
    print(f"\nwrote {args.report}")

    cfg = {
        "architecture": f"shared trunk Linear({D_ALIGN},{D_HIDDEN})-GELU-"
        f"Linear({D_HIDDEN},{D_HIDDEN})-GELU-Linear({D_HIDDEN},"
        f"{D_EMB}), L2-normalized embedding; one Linear({D_EMB},1) "
        "head per (domain,target)",
        "schema_alignment": f"per-domain vector_core RobustScaler -> PCA({D_ALIGN}, "
        "full SVD) -> per-component whitening, fit on that "
        "domain's train rows only",
        "losses": "masked MSE on train-z-scored regression targets + masked BCE "
        f"on binary targets (per-domain mean) + CORAL x{CORAL_WEIGHT} "
        "(pairwise embedding mean+covariance alignment, train rows)",
        "optimizer": f"AdamW(lr={LR}, weight_decay={WEIGHT_DECAY}), full-batch",
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": PATIENCE,
        "seed": SEED,
        "probe_head": f"sklearn Ridge(alpha={RIDGE_ALPHA}) fit on harness train side",
        "hyperparameter_selection": "all hyperparameters fixed a priori (no grid, "
        "no val sweep, test never consulted)",
        "runs": run_infos,
        "probe_tasks": probe_notes,
    }
    args.config_out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.config_out}")

    # -- unified exchange artifact --
    ex = args.exchange_out
    ex.mkdir(parents=True, exist_ok=True)
    n0, n1 = (len(r["y"]) for r in exchange_rows)
    X_all = np.concatenate([r["Z"] for r in exchange_rows]).astype(np.float32)
    dom = np.array([r["domain"] for r in exchange_rows for _ in range(len(r["y"]))])
    off = [0, n0]
    arrays = {
        "X": X_all,
        "domain": dom,
        "entity_id": np.concatenate([np.asarray(r["group"]).astype("U16") for r in exchange_rows]),
        "time_id": np.concatenate([r["tkey"] for r in exchange_rows]),
        "split_train": np.concatenate([r["train"] + o for r, o in zip(exchange_rows, off, strict=False)]),
        "split_val": np.array([], dtype=np.int64),
        "split_test": np.concatenate([r["test"] + o for r, o in zip(exchange_rows, off, strict=False)]),
    }
    for r, o, n in zip(exchange_rows, off, (n0, n1), strict=False):
        y_full = np.full(n0 + n1, np.nan, dtype=np.float32)
        m_full = np.zeros(n0 + n1, dtype=bool)
        y_full[o : o + n] = r["y"]
        m_full[o : o + n] = True
        arrays[f"y_{r['target']}"] = y_full
        arrays[f"label_mask_{r['target']}"] = m_full
    np.savez_compressed(ex / "dataset.npz", **arrays)
    sheet = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": int(n0 + n1),
        "row_layout": {
            "transfer_forward_return (equities rows)": [0, int(n0)],
            "transfer_next_season_per (hoops rows)": [int(n0), int(n0 + n1)],
        },
        "X": f"{D_EMB}-d frozen cross-domain embedding of each held-out row "
        "(per-row trunk trained WITHOUT that row's domain)",
        "labels": {
            t: f"identical to the owner lane's y_{TRANSFER[t][1]} on its "
            "labeled rows (see bench/data/exchange/"
            f"{TRANSFER[t][0]}/datasheet.json)"
            for t in TRANSFER
        },
        "splits": "owner lanes' temporal splits reproduced exactly (equities fy "
        ">= 2022 test; hoops target_year >= 2026 test); split_val is "
        "empty because the probe head is closed-form ridge with no "
        "early stopping",
        "time_id": "equities rows: fiscal year; hoops rows: target season year",
        "training_config": "bench/training_config.json",
        "report": "bench/benchmark_report.json",
    }
    (ex / "datasheet.json").write_text(json.dumps(sheet, indent=2) + "\n", encoding="utf-8")
    print(f"wrote exchange artifacts to {ex}")

    # -- honest console summary --
    print(f"\n== unified domain: {dsc.aggregate['headline']} ==")
    for ts in dsc.targets:
        if ts.scorecard is None:
            print(f"  {ts.target_name}: {ts.status} ({ts.note})")
            continue
        v = ts.scorecard.verdicts.get(ts.primary_metric)
        print(
            f"  {ts.target_name} [{ts.primary_metric}]: "
            f"best_baseline={v.best_baseline}={v.best_baseline_value:.4f} "
            f"mtnn={v.mtnn_value:.4f} delta={v.mtnn_delta:+.4f} "
            f"beats={v.mtnn_beats_best_baseline}"
        )
        skipped = [r.name for r in ts.scorecard.methods if r.status != "ok"]
        if skipped:
            print(f"    non-ok rungs: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
