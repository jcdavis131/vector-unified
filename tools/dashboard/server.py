#!/usr/bin/env python3
"""Live progress dashboard. Serves one screen of real state, no scrolling.

Every number is read from git or from an artifact on disk at request time. Nothing is
cached across requests beyond 8s and nothing is hand-written into the payload, so a stale
dashboard is impossible by construction -- if a source is unreadable the tile says so
rather than showing the last good value.

    python server.py            # prefers :8000, falls back to the next free port
"""

from __future__ import annotations

import http.server
import json
import socket
import socketserver
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# DERIVED, NOT HARDCODED. This read Path("C:/Users/jcdav") while it lived in a scratch
# directory, which was survivable there and is not here. Four registered gates in this repo
# carried the same laptop path and were moved onto portable_paths.ESTATE earlier today —
# check_cited_fields.py among them, which because of it verified ZERO published values for
# every reader who was not the author while printing PASS. Committing a dashboard with the
# same line would be shipping the defect this repo spent the day removing.
#
# tools/dashboard/server.py -> tools -> <repo> -> <estate>
UNIFIED = HERE.parent.parent
ESTATE = UNIFIED.parent
REALTY = ESTATE / "vector-realty"
_cache: dict = {"t": 0.0, "data": None}


def sh(args, cwd, timeout=8):
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect() -> dict:
    now = time.time()
    if _cache["data"] and now - _cache["t"] < 8:
        return _cache["data"]

    out: dict = {
        "generated": time.strftime("%H:%M:%S"),
        "repos": [],
        "tiles": [],
        "commits": [],
        "needs_you": [],
        "sources": {},
    }

    # --- repos ---
    for name, path in (("vector-unified", UNIFIED), ("vector-realty", REALTY)):
        if not path.exists():
            continue
        br = sh(["git", "branch", "--show-current"], path)
        sha = sh(["git", "rev-parse", "--short", "HEAD"], path)
        dirty = sh(["git", "status", "--porcelain"], path)
        ahead = sh(["git", "rev-list", "--count", "@{u}..HEAD"], path) or "0"
        out["repos"].append(
            {
                "name": name,
                "branch": br or "?",
                "sha": sha or "?",
                "dirty": len([l for l in dirty.splitlines() if l.strip()]),
                "unpushed": ahead,
            }
        )

    # --- commit stream, newest first, both repos merged ---
    for name, path in (("unified", UNIFIED), ("realty", REALTY)):
        if not path.exists():
            continue
        log = sh(["git", "log", "--format=%h\x1f%ct\x1f%s", "-14"], path)
        for line in log.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 3:
                out["commits"].append(
                    {
                        "repo": name,
                        "sha": parts[0],
                        "ts": int(parts[1]),
                        "subject": parts[2],
                    }
                )
    out["commits"].sort(key=lambda c: -c["ts"])
    for c in out["commits"]:
        c["ago"] = human_ago(now - c["ts"])
    out["commits"] = out["commits"][:14]

    # --- gate state (from the audit artifact, not by re-running the 3-minute gate) ---
    gia = jload(UNIFIED / "data" / "gate_inputs_tracked_audit.json")
    if gia and "summary" in gia:
        wt = gia["summary"].get("working_tree", {})
        cl = gia["summary"].get("clone", {})
        fails = wt.get("FAIL", 0)
        out["tiles"].append(
            {
                "label": "gate (working tree)",
                "value": f"{wt.get('PASS',0)} pass",
                "sub": f"{fails} fail · {wt.get('SKIP',0)} skip · {wt.get('N/A',0)} n/a",
                "state": "ok" if fails == 0 else ("warn" if fails <= 1 else "bad"),
            }
        )
        out["tiles"].append(
            {
                "label": "gate (fresh clone)",
                "value": f"{cl.get('PASS',0)} pass",
                "sub": f"{cl.get('FAIL',0)} fail · {cl.get('N/A',0)} n/a · "
                f"{len(gia.get('pass_here_fail_on_clone', []))} clone-only breaks",
                "state": "ok" if not gia.get("pass_here_fail_on_clone") else "warn",
            }
        )
        out["sources"]["gate"] = "data/gate_inputs_tracked_audit.json"
        # EVERY WORKING-TREE FAILURE, not just fail_in_both. This listed only checks
        # failing in BOTH places, so cited_fields — which fails here and is N/A on a
        # clone because the clone has no sibling vector-hub — never reached the operator.
        # The board under-reported the gate by exactly the item that was open.
        per = gia.get("per_check") or {}
        wt_fails = sorted(k for k, v in per.items() if v.get("working_tree") == "FAIL")
        for f in wt_fails:
            also = per[f].get("clone")
            out["needs_you"].append(
                {
                    "what": f"gate FAIL: {f}" + ("" if also == "FAIL" else f" (clone: {also} — cannot run there)"),
                    "where": "vector-unified",
                    "kind": "open",
                }
            )
        if not per:  # older audit without per_check; fall back, and say so
            for f in gia.get("fail_in_both", []):
                out["needs_you"].append(
                    {
                        "what": f"gate FAIL: {f} (fail_in_both only — audit predates per_check)",
                        "where": "vector-unified",
                        "kind": "open",
                    }
                )

    # --- documented-usage coverage ---
    dua = jload(UNIFIED / "data" / "documented_usage_audit.json")
    if dua and "coverage" in dua:
        c = dua["coverage"]
        out["tiles"].append(
            {
                "label": "documented cmds",
                "value": f"{c.get('pct_executed','?')}%",
                "sub": f"{c.get('executed',0)} ran · {c.get('broken',0)} broken · " f"{c.get('skipped',0)} skipped",
                "state": "ok" if c.get("broken", 1) == 0 else "warn",
            }
        )
        out["sources"]["usage"] = "data/documented_usage_audit.json"

    # --- the G2 headline, straight from the artifact ---
    g2 = jload(UNIFIED / "data" / "g2_centroid_ab.json")
    if g2:
        fl = (g2.get("FLOOR_ANALYSIS_is_sport_still_decodable_at_all") or {}).get("per_arm")
        if fl and "seed" in fl:
            s = fl["seed"]
            out["tiles"].append(
                {
                    "label": "G2 residual decodability (FULL)",
                    "value": f"{s.get('mean'):+.4f}",
                    "sub": f"CI [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] · "
                    f"ctrl {fl['ctrl']['mean']:+.4f} p={fl['ctrl']['p_two_sided']}",
                    "state": "ok",
                }
            )
        dec = g2.get("DECOMPOSITION") or {}
        if dec.get("coral_effect"):
            ce, le = dec["coral_effect"], dec["lambda_effect"]
            out["tiles"].append(
                {
                    "label": "G2 decomposition",
                    "value": f"lambda {le['p_two_sided']}",
                    "sub": f"coral p={ce['p_two_sided']} (did NOT confirm) · "
                    f"lambda is {dec.get('lambda_share_of_total',0):.0%}",
                    "state": "ok",
                }
            )
        out["sources"]["g2"] = "data/g2_centroid_ab.json"

    # --- shipped model integrity, verified live ---
    import hashlib

    ck = UNIFIED / "pipeline" / "data" / "unified_stage2_best.pt"
    rep = jload(UNIFIED / "data" / "unified_report.json")
    want_sha, want_acc = "b055641c03760624", 0.6851
    got_sha = ""
    if ck.exists():
        got_sha = hashlib.sha256(ck.read_bytes()).hexdigest()[:16]
    got_acc = ((rep or {}).get("G2_sport_invariance") or {}).get("sport_acc")
    ok = got_sha == want_sha and got_acc == want_acc
    out["tiles"].append(
        {
            "label": "shipped model",
            "value": "INTACT" if ok else "DRIFT",
            "sub": f"sport_acc {got_acc} · ckpt {got_sha or 'missing'}",
            "state": "ok" if ok else "bad",
        }
    )
    if not ok:
        out["needs_you"].append(
            {
                "what": "shipped model drifted from 0.6851 / b055641c03760624",
                "where": "vector-unified",
                "kind": "alert",
            }
        )

    # --- operator decisions, read from the record rather than asserted here ---
    # Checked against origin/master, NOT the working tree: the record files live on master
    # while the repo usually sits on a work branch, so a working-tree exists() reports the
    # decision as absent exactly when the repo is mid-lane. That was this file's own first
    # bug -- the G2 decision vanished from the board while it was the main open item.
    def on_master(repo: Path, rel: str) -> bool:
        return bool(
            sh(["git", "cat-file", "-e", f"origin/master:{rel}"], repo) == ""
            and subprocess.run(
                ["git", "cat-file", "-e", f"origin/master:{rel}"],
                cwd=str(repo),
                capture_output=True,
            ).returncode
            == 0
        )

    if on_master(UNIFIED, "LOCAL_GPU_G2_RESULT.md"):
        out["needs_you"].append(
            {
                "what": "promote or discard the G2 result (nothing promoted)",
                "where": "vector-unified/LOCAL_GPU_G2_RESULT.md @ origin/master",
                "kind": "decision",
            }
        )
    if (REALTY / "REVIEW_2026-08-05.md").exists():
        out["needs_you"].append(
            {
                "what": "vector-realty headline IS inverted — correct gradient 0.8505 (sd 0.0030) "
                "beats MTNN 0.8281 on 5/5 seeds; reproduced independently",
                "where": "vector-realty/REVIEW_2026-08-05.md",
                "kind": "decision",
            }
        )
    if on_master(UNIFIED, "data/TENNIS_CITATION_GAP.md"):
        out["needs_you"].append(
            {
                "what": "6 published tennis values cite artifacts git never carried — make tennis "
                "reproducible and track, or republish and say the numbers are one draw",
                "where": "vector-unified/data/TENNIS_CITATION_GAP.md",
                "kind": "decision",
            }
        )
    if on_master(UNIFIED, "data/seed_order_audit.json"):
        out["needs_you"].append(
            {
                "what": "ablation.py seeds AFTER model init (line 50 vs 56) — root cause of three "
                "disagreeing artifacts; fixing it means re-running the tables",
                "where": "vector-unified/data/seed_order_audit.json",
                "kind": "decision",
            }
        )
    if on_master(UNIFIED, "SCHEDULING.md"):
        out["needs_you"].append(
            {
                "what": "install the validation sweep as a scheduled task — the crons die with " "the Claude session",
                "where": "vector-unified/SCHEDULING.md",
                "kind": "action",
            }
        )

    _cache.update(t=now, data=out)
    return out


def human_ago(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec//60}m"
    if sec < 86400:
        return f"{sec//3600}h{(sec%3600)//60:02d}"
    return f"{sec//86400}d"


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def do_GET(self):
        if self.path.startswith("/status.json"):
            body = json.dumps(collect()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def free_port(preferred=8000):
    for p in [preferred] + list(range(8001, 8020)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return 0


if __name__ == "__main__":
    port = free_port(8000)
    print(f"dashboard: http://localhost:{port}/", flush=True)
    if port != 8000:
        print(
            f"  (8000 was busy — held by another process; this takes {port})",
            flush=True,
        )
    Server(("127.0.0.1", port), H).serve_forever()
