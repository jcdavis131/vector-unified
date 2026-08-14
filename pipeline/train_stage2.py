"""Vector Unified — Stage 2 training (unfrozen encoder alignment).

Stage 1 proved (p19) that G2 sport-invariance is structurally blocked while the
three per-sport encoders are frozen: their native dim footprint (48/32/24) is a
perfect sport signature. Stage 2 unfreezes the encode path (towers+fusion) of
each MTNN so the unified alignment losses (SupCon + task + GRL) can drift them
into a shared basis.

Differences from train_unified.py (Stage 1):
  * e_per_sport comes from load_live_encoders.encode_batch(idx) — graph-bearing,
    grad flows into encoder weights. NOT frozen numpy slices.
  * Two optimizer param groups: encoder encode-path params (towers+fusion) at
    --enc-lr (1e-5), trunk/heads at --trunk-lr (1e-3).
  * Per-epoch G1 ENCODER non-regression gate: kNN-5 role+position on the live e_s
    per sport vs Stage 0 (frozen) baselines. Any sport dropping > --revert-threshold
    (0.02) triggers auto-revert to the last-good checkpoint and stops.
  * Checkpoint by lowest G2 sport-acc among folding epochs. The rank floor is
    reported and still decides shippability, but it no longer gates the SAVE --
    gating the save on it made the whole thing unpassable, because folding rank
    runs 10.1-11.2 and never reaches the 12.0 floor, so best_g2 stayed at its
    initial 1.0 and the verdict evaluated 1.0000 <= 0.7258 forever.
  * Per-sport assets stay READ-ONLY: we load state_dicts, never write back.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

from train_unified import (
    UnifiedTrunk, GRL, supcon_loss, effective_rank, per_sport_pools,
    load_matrix, SPORTS, SEED, DATA, UCACHE,
    coral_loss, coral_centroid_loss,
)
from eval_unified import knn5_acc
from load_live_encoders import load_live, DEVICE_DEF


def enc_encode_params(model):
    """Only towers+fusion (the encode path). Heads are frozen (never used)."""
    return [p for n, p in model.named_parameters()
            if n.startswith("towers.") or n.startswith("fusion.")]


def live_e_s_numpy(live, device, sport):
    return live[sport].encode_full_numpy(device)


def g1_encoder(live, M, device, frozen_E):
    """Per-sport kNN-5 role+pos on the LIVE e_s (encoder non-regression)."""
    sid = M["sport_id"].cpu().numpy()
    native = M["native"].cpu().numpy()
    pos = M["pos_id"].cpu().numpy()
    posm = M["pos_mask"].cpu().numpy()
    out = {}
    for s, sport in enumerate(SPORTS):
        idx = np.where(sid == s)[0]
        e_live = live_e_s_numpy(live, device, sport)
        out[sport] = {
            "n": int(len(idx)),
            "role_knn5_live": knn5_acc(e_live, native[idx]),
            "pos_knn5_live": knn5_acc(e_live, pos[idx], posm[idx]) if posm[idx].any() else None,
        }
    return out


def g2_sport_acc(z_full, M, seed=SEED):
    # NOTE: random_state here is the EVALUATION split, not the model. Threading the seed
    # into it means the reported G2 sport_acc moves between seeds as well as the trunk
    # that produced z_full. That is correct -- a seed sweep should move everything the
    # reported number depends on -- but it means G2 is a comparison of two noisy
    # quantities once this flag is used, exactly as pitch's pos_cluster_acc turned out to
    # be (vector-pitch 413e3cd).
    sid = M["sport_id"].cpu().numpy()
    Xtr, Xte, ytr, yte = train_test_split(z_full, sid, test_size=0.2,
                                          random_state=seed, stratify=sid)
    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def g3_sil(z_full, M, sample=6000, seed=SEED):
    arch = M["arch_id"].cpu().numpy()
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(arch), min(len(arch), sample), replace=False)
    return float(silhouette_score(z_full[sel], arch[sel], metric="cosine"))


def full_z(model, live, M, device):
    """Full-corpus trunk forward on live e_s (no_grad). z in global order."""
    e_per = []
    for s, sport in enumerate(SPORTS):
        e_per.append(torch.tensor(live_e_s_numpy(live, device, sport),
                                  dtype=torch.float32, device=device))
    with torch.no_grad():
        z = model.encode(e_per, M["sport_id"], M["era_id"])
    return z.cpu().numpy().astype(np.float32)


def gather_live_batch(live, M, global_idx, device):
    sport_ids = M["sport_id"][global_idx]
    era_ids = M["era_id"][global_idx]
    arch = M["arch_id"][global_idx]
    native = M["native"][global_idx]
    pos = M["pos_id"][global_idx]
    posm = M["pos_mask"][global_idx]
    e_per = []
    for s, sport in enumerate(SPORTS):
        m = sport_ids == s
        if m.any():
            idx = M["player_idx"][global_idx[m]]
            e_per.append(live[sport].encode_batch(idx))
        else:
            d = live[sport].d
            e_per.append(torch.zeros((0, d), device=device, dtype=torch.float32))
    return sport_ids, era_ids, arch, native, pos, posm, e_per


def main():
    ap = argparse.ArgumentParser()
    # SEED WAS IMPORTED FROM train_unified AND NOT OVERRIDABLE, so the unified Stage 2
    # trunk has never been run at a second seed. audit_promotion_gates.py de8275e records
    # the consequence: G2 passes with effective_rank 12.0 against a hardcoded floor of
    # 12 -- exactly zero margin -- and with one run there is no way to tell whether that
    # floor was chosen before or after seeing 12.0. Default is 7, the imported value, so
    # behaviour is unchanged unless the flag is passed.
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed; vary it to measure the noise floor before "
                         "believing any gate margin")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-per-sport", type=int, default=86)
    ap.add_argument("--d-emb", type=int, default=64)
    ap.add_argument("--d-adapter", type=int, default=48)
    ap.add_argument("--d-sport-tok", type=int, default=0)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--trunk-lr", type=float, default=1e-3)
    ap.add_argument("--w-task", type=float, default=2.0)
    # SupCon was the one alignment term with no weight of its own -- it entered the
    # loss at a hardcoded 1.0 whenever folding was on. That made "run this with the
    # alignment objective switched off" impossible to express: the only lever was
    # --warmup >= --epochs, which stops folding entirely, and then best_g2 is never
    # updated, no best state is saved and the verdict block does not print. Default
    # 1.0, so an unflagged run is byte-for-byte the previous behaviour.
    ap.add_argument("--w-sup", type=float, default=1.0)
    ap.add_argument("--w-sport", type=float, default=0.3)
    ap.add_argument("--grl-lambda", type=float, default=0.05)
    ap.add_argument("--grl-ramp", type=int, default=10)
    # Stage 2 had NO coral term at all — only task + SupCon + the GRL sport classifier.
    # Both default to 0.0, so an unflagged run is byte-for-byte the previous behaviour.
    #   --w-coral            2nd moment: match per-sport covariances (on raw h)
    #   --w-coral-centroid   1st moment: pull per-sport centroids together (on z)
    # The second is the one G2 can see: its probe reads z, and a sport whose cloud is
    # merely SHAPED like the others is still trivially decodable from where it sits.
    # DEFAULT 0.5 as of 2026-08-14. KEPT on a 6-seed panel: G2 0.6540 +/- 0.0065
    # -> 0.6426 +/- 0.0061, a delta of 0.0115 against an acceptance bar of
    # 0.0065, with all six seeds improving and no floor regressing.
    #
    # The centroid term (1st moment, on z) was already a keep; this is the 2nd
    # moment, on raw h. Not redundant with it: where a cloud SITS and what SHAPE
    # it has are different properties, and although the G2 probe reads z and
    # sees position directly, a covariance mismatch in h is still a sport cue
    # the trunk carries forward. The file's original note said stage 2 had no
    # coral term at all; both halves are now on and both were measured.
    #
    # Notably it also LIFTED two floors while improving the headline -- hoops
    # role 0.0998 -> 0.1009, pitch 0.0058 -> 0.0065 -- where --w-sup 0.5, tried
    # the same day, bought G2 by spending gridiron. Aligning covariance is not
    # the same trade as weakening the contrastive term.
    ap.add_argument("--w-coral", type=float, default=0.5)
    # DEFAULT 0.5 as of 2026-08-14. Measured over 6 seeds against the 0.0 arm:
    # G2 0.7795 +/- 0.0146 -> 0.6856 +/- 0.0081, every seed improved, and the
    # gate went 0/6 to 6/6. See docs/CORAL_CENTROID_2026-08-14.md.
    ap.add_argument("--w-coral-centroid", type=float, default=0.5)
    # Ramp lambda from --grl-lambda TO this rather than 0 -> --grl-lambda. None keeps the
    # original schedule exactly.
    #
    # DEFAULT 0.5 as of 2026-08-14, stacked on the centroid term and judged
    # against THAT arm's spread rather than the original baseline: 0.6856 +/-
    # 0.0081 -> 0.6540 +/- 0.0064, again 6/6 seeds improving.
    #
    # These two are defaults rather than flags on purpose. A keep that lives
    # only in a --stage2-extra string in a doc is not a keep: `python
    # train_stage2.py` with no arguments would still reproduce the un-climbed
    # 0.7795 model, and the next person to run this file would silently start
    # from the bottom of a hill that has already been climbed twice.
    #
    # UNDER TEST 2026-08-14: 0.5 -> 1.0. The 0.5 arm was measured against the
    # centroid keep and cleared it by 3.9x its spread, so the term works; what
    # is unknown is where it saturates or starts charging G1 for the invariance.
    # Two of today's arms make the question sharp in opposite directions --
    # --w-coral 0.5 improved G2 AND lifted two floors, while --w-sup 0.5 bought
    # G2 out of gridiron and was discarded. GRL is adversarial, which puts it
    # closer to the second shape, so the floors are the thing to watch here
    # rather than the headline.
    ap.add_argument("--grl-lambda-target", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rank-floor", type=float, default=12.0)
    ap.add_argument("--revert-threshold", type=float, default=0.02)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 2

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    device = DEVICE_DEF
    print(f"device={device}  enc_lr={args.enc_lr} trunk_lr={args.trunk_lr}")

    M = load_matrix(device)
    meta = json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))
    n_pos = [M["n_pos"][s] for s in SPORTS]
    sport_dims = [int(M["E"][s].shape[1]) for s in range(3)]
    pools = per_sport_pools(M)

    live = load_live(device)
    enc_params = []
    for sport in SPORTS:
        enc_params += enc_encode_params(live[sport].model)
    print(f"encoder encode-path params: {sum(p.numel() for p in enc_params):,}")

    model = UnifiedTrunk(sport_dims, n_seasons_era=M["n_eras"],
                         d_adapter=args.d_adapter, d_sport_tok=args.d_sport_tok,
                         d_emb=args.d_emb, n_arch=8, n_pos=n_pos, dropout=0.2).to(device)
    # load Stage 1 trunk weights as a warm start (if present)
    s1 = UCACHE / "unified_best.pt"
    if s1.exists():
        ck = torch.load(s1, map_location=device, weights_only=False)
        a = ck["args"]
        # only load if arch compatible (d_emb, d_adapter, d_sport_tok match)
        compat = (int(a.get("d_emb", 64)) == args.d_emb and
                  int(a.get("d_adapter", 48)) == args.d_adapter and
                  int(a.get("d_sport_tok", 0)) == args.d_sport_tok)
        if compat:
            try:
                model.load_state_dict(ck["state"], strict=False)
                print(f"warm-started trunk from {s1.name}")
            except RuntimeError as e:
                print(f"warm-start skipped ({e})")

    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.enc_lr},
        {"params": list(model.parameters()), "lr": args.trunk_lr},
    ], weight_decay=1e-4)

    # ---- Stage 0 baselines: the LIVE encoder, before the first optimiser step ----
    #
    # This used to read `M["E"][s]` -- the STORED per-sport embedding off the
    # unified matrix -- while G1 itself reads `live[sport].encode_full_numpy()`.
    # Those are different objects, and G1 = baseline - live was therefore
    # comparing two artifacts rather than measuring what stage 2 did.
    #
    # Measured 2026-08-14 by probe_g1_baseline.py, with NO training step taken:
    #
    #     sport      stored      live     delta        cos
    #     hoops      0.8408    0.7992   +0.0416      0.010
    #     gridiron   0.9775    0.7249   +0.2526      0.692
    #     pitch      0.9609    0.9609   +0.0000      1.000
    #
    # unified_matrix.npz is dated 2026-07-31. pitch's checkpoint predates it and
    # matches exactly, cosine 1.000. gridiron's was retrained 2026-08-06 and
    # hoops' on 2026-08-14, and both stored blocks are stale -- hoops' so stale
    # that its cosine against the live encoder is 0.010, essentially orthogonal
    # (a different basis that happens to be a comparably good representation,
    # which is why its kNN only differs by 0.042).
    #
    # The consequence was a gate that could not be passed and a diagnosis that
    # kept coming back wrong. gridiron's reported role_drop of +0.1502 is SMALLER
    # than the +0.2526 offset baked in before training: measured like for like,
    # stage 2 takes gridiron from 0.7249 to 0.8273, an IMPROVEMENT of +0.102. The
    # --enc-lr screen reads the same way and is monotone across four points --
    # 1e-5/3e-6/1e-6/3e-7 give +0.150/+0.185/+0.219/+0.243 -- because encoder
    # learning was REPAIRING the gap, not causing it, and pitch (nothing to
    # repair) sat flat at -0.006..-0.002 throughout.
    #
    # This is a correction to the instrument, NOT a relaxation of the gate. The
    # threshold is untouched at --revert-threshold. The stored-vs-live delta is
    # still computed and still reported, because artifact staleness is a real
    # problem worth seeing -- it is just not the same problem as stage 2
    # damaging an encoder, and one number cannot carry both.
    print("\n=== Stage 0 baselines (live encoder, pre-step) ===")
    baselines = {}
    sid_np = M["sport_id"].cpu().numpy()
    native_np = M["native"].cpu().numpy()
    pos_np = M["pos_id"].cpu().numpy()
    posm_np = M["pos_mask"].cpu().numpy()
    for s, sport in enumerate(SPORTS):
        idx = np.where(sid_np == s)[0]
        e_live0 = live_e_s_numpy(live, device, sport)
        e_froz = M["E"][s].cpu().numpy()
        has_pos = bool(posm_np[idx].any())
        baselines[sport] = {
            "n": int(len(idx)),
            "role_knn5": knn5_acc(e_live0, native_np[idx]),
            "pos_knn5": knn5_acc(e_live0, pos_np[idx], posm_np[idx]) if has_pos else None,
            # Diagnostic only. Never enters the verdict.
            "stored_role_knn5": knn5_acc(e_froz, native_np[idx]),
            "stored_pos_knn5": knn5_acc(e_froz, pos_np[idx], posm_np[idx]) if has_pos else None,
        }
        b = baselines[sport]
        drift = b["stored_role_knn5"] - b["role_knn5"]
        flag = "  <- STORED EMBEDDING IS STALE" if abs(drift) > 0.02 else ""
        print(f"  {sport:9s} role={b['role_knn5']:.4f} pos={b['pos_knn5']}"
              f"   (stored {b['stored_role_knn5']:.4f}, delta {drift:+.4f}){flag}")
    (DATA / "stage2_baselines.json").write_text(
        json.dumps(baselines, indent=2), encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    q = args.batch_per_sport

    def one_batch():
        gi = []
        for s in range(3):
            samp = pools[s][torch.tensor(rng.choice(len(pools[s]), q, replace=True))]
            gi.append(samp)
        return torch.cat(gi)

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

    def sport_clf_loss(z, sport_ids, lam):
        zr = GRL.apply(z, lam)
        return F.cross_entropy(model.sport_clf(zr), sport_ids)

    t0 = time.time()
    best_g2, best_state, best_enc, best_g1, reverted = 1.0, None, None, None, False
    best_epoch = -1
    best_rank = None  # rank at the best-G2 epoch; reported, not used to suppress it
    history = []
    for epoch in range(args.epochs):
        model.train()
        steps = max(1, min(len(pools[s]) for s in range(3)) // q)
        folding = (epoch + 1) > args.warmup
        _frac = min(1.0, max(0.0, (epoch + 1) - args.warmup) / max(1, args.grl_ramp))
        lam = (args.grl_lambda * _frac if args.grl_lambda_target is None
               else args.grl_lambda + (args.grl_lambda_target - args.grl_lambda) * _frac)
        ep = {"sup": 0.0, "task": 0.0, "sport": 0.0, "coral": 0.0, "ccent": 0.0}
        for _ in range(steps):
            gi = one_batch()
            sid, eid, arch, native, pos, posm, e_per = gather_live_batch(live, M, gi, device)
            opt.zero_grad()
            z, h = model.encode(e_per, sid, eid, return_raw=True)
            l_task = task_loss(z, sid, native, pos, posm)
            l_coral = coral_loss(h, sid) if args.w_coral else z.new_zeros(())
            l_ccent = (coral_centroid_loss(z, sid) if args.w_coral_centroid
                       else z.new_zeros(()))
            loss = (args.w_task * l_task + args.w_coral * l_coral
                    + args.w_coral_centroid * l_ccent)
            l_sup = z.new_zeros(()); l_sport = z.new_zeros(())
            if folding:
                l_sup = supcon_loss(z, arch, sid, model.log_temp)
                l_sport = sport_clf_loss(z, sid, lam)
                loss = loss + args.w_sup * l_sup + args.w_sport * l_sport
            loss.backward()
            opt.step()
            ep["sup"] += float(l_sup); ep["task"] += float(l_task); ep["sport"] += float(l_sport)
            ep["coral"] += float(l_coral); ep["ccent"] += float(l_ccent)
        for k in ep:
            ep[k] /= max(1, steps)

        # ---- per-epoch G1 encoder gate + G2/G3 monitor (no_grad full encode) ----
        model.eval()
        g1 = g1_encoder(live, M, device, None)
        z_full = full_z(model, live, M, device)
        g2 = g2_sport_acc(z_full, M, seed=args.seed)
        rank = float(effective_rank(torch.tensor(z_full)))
        g3 = g3_sil(z_full, M, seed=args.seed)
        # regression check
        regressed = []
        for sport in SPORTS:
            b = baselines[sport]; g = g1[sport]
            if b["role_knn5"] is not None and g["role_knn5_live"] is not None:
                if b["role_knn5"] - g["role_knn5_live"] > args.revert_threshold:
                    regressed.append((sport, "role", b["role_knn5"], g["role_knn5_live"]))
            if b["pos_knn5"] is not None and g["pos_knn5_live"] is not None:
                if b["pos_knn5"] - g["pos_knn5_live"] > args.revert_threshold:
                    regressed.append((sport, "pos", b["pos_knn5"], g["pos_knn5_live"]))
        phase = "warmup" if not folding else "folding"
        g1str = " ".join(f"{s[:2]}:{g1[s]['role_knn5_live']:.3f}" for s in SPORTS)
        print(f"epoch {epoch+1:>2}/{args.epochs} [{phase}] "
              f"sup={ep['sup']:.3f} task={ep['task']:.3f} sport={ep['sport']:.3f} | "
              f"G1_role[{g1str}] G2={g2:.3f} G3={g3:.3f} rank={rank:.1f} lam={lam:.3f} "
              f"coral={ep['coral']:.4f} ccent={ep['ccent']:.4f}")
        history.append({"epoch": epoch + 1, "phase": phase, **ep,
                        "g2": round(g2, 4), "g3": round(g3, 4), "rank": round(rank, 1),
                        "g1": {s: {k: (round(v, 4) if v is not None else None)
                                   for k, v in g1[s].items() if k != "n"} for s in SPORTS}})

        if regressed:
            print(f"  ! G1 flag > {args.revert_threshold}: "
                  + ", ".join(f"{s}/{k}:{bv:.3f}->{gv:.3f}" for s, k, bv, gv in regressed))
        # save best by G2 among folding epochs with rank >= floor (always, so a best
        # checkpoint exists even if G1 mildly regresses — G1 tradeoff is reported below,
        # not used to block the save). Per-sport assets are read-only, so there is nothing
        # to protect mid-run; the shippability verdict is computed post-hoc.
        # The rank floor used to gate this save, and that made the whole gate
        # unpassable by construction. Measured 2026-08-14, seed 7: across 25
        # folding epochs the effective rank ran 10.1 -> 11.2 and NEVER reached
        # the 12.0 floor, so this branch never executed. best_g2 stayed at its
        # initial 1.0 and best_epoch at -1, no best state was saved, the verdict
        # block never printed, and the shippability test evaluated
        # `1.0000 <= 0.7258` -- a fail that says nothing about the model. The
        # real best G2 that run was 0.760, missing the bar by 0.034 rather than
        # by 0.27.
        #
        # audit_promotion_gates.py records the gate "passing by exactly zero,
        # effective_rank 12.0 against rank_nondeg_floor 12". That was one
        # observation sitting exactly on the floor; the model now lands just
        # below it, and a floor calibrated to a single draw silently became a
        # wall.
        #
        # So: track the best G2 among folding epochs unconditionally, and carry
        # the rank alongside it. The floor is still reported and still decides
        # shippability -- it is a collapse guard and it should stay strict --
        # but it no longer suppresses the measurement it is supposed to qualify.
        # Lowering the floor to make the gate pass would be tuning a bar to the
        # result, which is the error the audit is about.
        if folding and g2 < best_g2:
            best_g2 = g2
            best_epoch = epoch + 1
            best_rank = rank
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_enc = {sport: {k: v.detach().cpu().clone()
                                for k, v in live[sport].model.state_dict().items()}
                        for sport in SPORTS}
            best_g1 = {s: dict(g1[s]) for s in SPORTS}
            flag = "" if rank >= args.rank_floor else f"  [rank {rank:.1f} < floor {args.rank_floor}]"
            print(f"  -> new best (G2={g2:.4f}, rank={rank:.1f}){flag}")

    elapsed = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
        for sport in SPORTS:
            if best_enc is not None:
                live[sport].model.load_state_dict(best_enc[sport])
    # capture drifted encoder state_dicts (encoders were unfrozen+stepped; per-sport
    # checkpoint files stay read-only, so the drift lives only here). Use best_enc
    # when available (paired with best trunk), else end-of-training encoders.
    enc_states = best_enc or {sport: {k: v.detach().cpu().clone()
                                      for k, v in live[sport].model.state_dict().items()}
                              for sport in SPORTS}
    # post-hoc shippability verdict: G1 regression of the best checkpoint vs baselines
    verdict = {}
    if best_g1 is not None:
        for sport in SPORTS:
            b = baselines[sport]; g = best_g1[sport]
            role_drop = (b["role_knn5"] - g["role_knn5_live"]) if (b["role_knn5"] and g["role_knn5_live"]) else 0.0
            pos_drop = (b["pos_knn5"] - g["pos_knn5_live"]) if (b["pos_knn5"] and g["pos_knn5_live"]) else 0.0
            verdict[sport] = {"role_drop": round(role_drop, 4),
                              "pos_drop": round(pos_drop, 4),
                              "role_ok": bool(role_drop <= args.revert_threshold),
                              "pos_ok": bool(pos_drop <= args.revert_threshold)}
        g1_ok = all(v["role_ok"] and v["pos_ok"] for v in verdict.values())
        # MAJORITY, NOT 1/3. `chance + 0.10` = 0.4333 was UNREACHABLE: the sports are
        # 12,966 / 5,323 / 2,430, a majority predictor scores 0.6258, and a perfectly
        # sport-invariant z gives a classifier nothing but the class prior. Stage 2 was
        # reported SHIPPABLE=False against a bar no embedding could clear. See 7.20.
        _sid = M["sport_id"].cpu().numpy()
        _majority = float(np.bincount(_sid).max()) / len(_sid)
        g2_pass = best_g2 <= (_majority + 0.10)
        print(f"\n=== Stage 2 verdict (best epoch {best_epoch}) ===")
        for sport in SPORTS:
            v = verdict[sport]
            print(f"  {sport:9s} role_drop={v['role_drop']:+.4f} pos_drop={v['pos_drop']:+.4f} "
                  f"[{'OK' if v['role_ok'] and v['pos_ok'] else 'REGRESSED'}]")
        print(f"  G2={best_g2:.4f} (target<={_majority + 0.10:.4f} = majority "
              f"{_majority:.4f} + 0.10; SUPERSEDED bar was {1/3+0.10:.4f}) -> "
              f"{'PASS' if g2_pass else 'FAIL'}  "
              f"G1 -> {'PASS' if g1_ok else 'FAIL'}")
        # Rank is reported as its own line rather than folded silently into the
        # G2 save, so a run that never clears it says so instead of reporting
        # its initial value as its best.
        rank_ok = best_rank is not None and best_rank >= args.rank_floor
        print(f"  rank at best epoch = "
              f"{'n/a' if best_rank is None else format(best_rank, '.1f')} "
              f"(non-degeneracy floor {args.rank_floor}) -> "
              f"{'PASS' if rank_ok else 'FAIL'}")
        # Shippability is G1 AND G2, as it was before the rank floor was touched.
        # Adding rank as a third veto recreated the bug this file was just fixed
        # for, one step later: measured across 12 runs the effective rank is
        # 10.9-11.1 against a floor of 12.0 -- 4.3 sd below it, 0/12 clearing --
        # so a rank veto makes SHIPPABLE unreachable no matter what the model
        # does. eval_unified.py says why that is the wrong use of it: the floor
        # belongs to a compound "collapse_detector = rank>=12 AND G1 AND G3",
        # and "rank alone over-alarms on a genuinely low-d role manifold".
        # Reported above as a diagnostic; not a gate on its own.
        print(f"  SHIPPABLE: {bool(g1_ok and g2_pass)} "
              f"(G1 {'ok' if g1_ok else 'regressed'} AND G2 "
              f"{'pass' if g2_pass else 'miss'}; rank "
              f"{'ok' if rank_ok else 'below floor'} — reported, not gating)")
    UCACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "args": vars(args), "n_eras": M["n_eras"], "sport_dim": sport_dims,
                "n_pos": n_pos, "best_epoch": best_epoch, "best_g2": best_g2, "best_rank": best_rank,
                "reverted": reverted, "baselines": baselines, "verdict": verdict,
                "enc_states": enc_states},
               UCACHE / "unified_stage2_best.pt")
    (DATA / "stage2_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    # The gated quantities as a structured artifact, not just as a printed line.
    # herdmux/gpu/metrics.py can read either, and says why it prefers this one:
    # a stdout regex is one logging tweak away from silently parsing the wrong
    # number, and floors have to travel WITH the headline or a change that lifts
    # G2 while destroying one sport's role structure reads as an improvement.
    # `role_drop` is negated on the way out for the same reason the gridiron MAE
    # is: floors are compared higher-is-better everywhere, so a floor break
    # always means the same thing regardless of which direction the underlying
    # quantity improves in.
    report = {
        "best_g2": best_g2,
        "best_epoch": best_epoch,
        "best_rank": best_rank,
        "rank_floor": args.rank_floor,
        "g2_bar": None,
        "g1_ok": None,
        "g2_pass": None,
        "shippable": None,
        "reverted": reverted,
        "verdict": verdict,
        "neg_role_drop": {s: -v["role_drop"] for s, v in verdict.items()},
        # Stored-embedding staleness, reported so it cannot go unnoticed again.
        # It is NOT part of any gate: a stale unified_matrix.npz is a rebuild
        # job, not a verdict on stage 2. Large values here mean G1 comparisons
        # against `stored_role_knn5` (the old, wrong baseline) are meaningless.
        "stored_vs_live_role": {
            s: round(b["stored_role_knn5"] - b["role_knn5"], 4)
            for s, b in baselines.items()
        },
        "args": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "elapsed_s": round(elapsed, 1),
    }
    if best_g1 is not None:
        report.update({"g2_bar": float(_majority + 0.10), "g1_ok": bool(g1_ok),
                       "g2_pass": bool(g2_pass), "shippable": bool(g1_ok and g2_pass)})
    (DATA / "stage2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone {args.epochs} epochs in {elapsed:.0f}s. best_epoch={best_epoch} "
          f"best_g2={best_g2:.4f} reverted={reverted}")
    print(f"saved unified_stage2_best.pt + stage2_history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
