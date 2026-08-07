"""Slim the 16.7 MB assets/unified.json into something a browser should fetch.

`unified.json` carries the full 64-d vector for all 20,719 rows. That is ~85%
of its bytes and it is not needed to DRAW anything: the file already ships
precomputed PCA coordinates (x, y, z) per row. So:

  assets/unified_slim.json   20,719 x {sport,name,season,pos,team,arch,x,y,z}
                             plus the gate/axis metadata the page must cite
  assets/unified_emb.f32     the 20,719 x 64 float32 block, raw, fetched
                             LAZILY and only when the neighbour probe is used

Precedent for the .f32 sidecar is vector-hoops, which ships
assets/mtnn_embeddings.f32 and already has the vercel.json header rule for it.

This script is mechanical and deterministic: it reads the built artifact and
rewrites it. It does NOT retrain, re-run a gate, or invent a number. Every
value it copies keeps the name it had in the source file so the page can cite
`unified.json -> field`.

Run:  python pipeline/export_web_slim.py [--check]
      --check verifies the outputs match the source without writing.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "unified.json"
OUT_JSON = ROOT / "assets" / "unified_slim.json"
OUT_F32 = ROOT / "assets" / "unified_emb.f32"
ABLATION = ROOT / "data" / "ablation_report.json"

# Fields copied verbatim from unified.json so the page can attribute them.
META_FIELDS = (
    "built",
    "model",
    "d_emb",
    "n_players",
    "normalization",
    "g1_verdict",
    "g1_pos_source",
    "g1_pos_caveat",
    "g2_sport_acc",
    "g2_target",
    "g2_majority_baseline",
    "g2_delta_vs_majority",
    "g2_status",
    "g2_note",
    "sports",
    "archetypes",
    "axes",
)


def build(src: dict) -> tuple[dict, bytearray, dict]:
    players = src["players"]
    d_emb = int(src["d_emb"])
    rows = []
    blob = bytearray()
    arch_counts: dict[str, int] = {}
    sport_counts: dict[str, int] = {}
    for p in players:
        rows.append(
            [
                p["sport"],
                p["name"],
                p["season"],
                p.get("pos", ""),
                p.get("team", ""),
                p.get("cross_arch", ""),
                round(float(p["x"]), 5),
                round(float(p["y"]), 5),
                round(float(p["z"]), 5),
            ]
        )
        e = p["e"]
        if len(e) != d_emb:
            raise SystemExit(
                f"row {p.get('name')!r} has {len(e)} dims, expected {d_emb} -- "
                f"refusing to write a ragged .f32 block"
            )
        blob.extend(struct.pack(f"<{d_emb}f", *(float(v) for v in e)))
        arch_counts[p.get("cross_arch", "")] = arch_counts.get(p.get("cross_arch", ""), 0) + 1
        sport_counts[p["sport"]] = sport_counts.get(p["sport"], 0) + 1

    meta = {k: src[k] for k in META_FIELDS if k in src}
    meta["explained_variance"] = src.get("proj", {}).get("explained_variance")
    # Counted here, not asserted: unified_meta.json records A4 = 0, and a UI
    # that lists 12 archetypes would show dead categories. The page reads this.
    meta["arch_counts"] = arch_counts
    meta["sport_counts"] = sport_counts
    meta["row_schema"] = ["sport", "name", "season", "pos", "team", "arch", "x", "y", "z"]
    meta["emb_file"] = OUT_F32.name
    meta["emb_dtype"] = "float32 little-endian"
    meta["emb_shape"] = [len(players), d_emb]
    meta["generated_by"] = "pipeline/export_web_slim.py"
    meta["source_file"] = SRC.name

    if ABLATION.exists():
        meta["ablation"] = json.loads(ABLATION.read_text(encoding="utf-8"))

    return {"meta": meta, "rows": rows}, blob, arch_counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify, write nothing")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found -- run the unified export first")
    src = json.loads(SRC.read_text(encoding="utf-8"))
    slim, blob, arch_counts = build(src)

    n = len(slim["rows"])
    if n != int(src["n_players"]):
        raise SystemExit(f"row count {n} != n_players {src['n_players']}")
    expect_bytes = n * int(src["d_emb"]) * 4
    if len(blob) != expect_bytes:
        raise SystemExit(f".f32 is {len(blob)} B, expected {expect_bytes}")

    payload = json.dumps(slim, ensure_ascii=False, separators=(",", ":"))
    print(f"rows            {n}")
    print(f"slim json       {len(payload) / 1e6:.2f} MB (source {SRC.stat().st_size / 1e6:.2f} MB)")
    print(f"emb f32         {len(blob) / 1e6:.2f} MB  = {n} x {src['d_emb']} x 4")
    populated = {k: v for k, v in sorted(arch_counts.items()) if v}
    empty = sorted(k for k, v in arch_counts.items() if not v)
    declared = {a["id"] for a in src.get("archetypes", [])}
    never = sorted(declared - set(arch_counts))
    print(f"archetypes      {len(populated)} populated of {len(declared)} declared")
    if never or empty:
        print(f"  NEVER ASSIGNED: {never + empty} -- the page must not list these as live")

    if args.check:
        for f, want in ((OUT_JSON, payload), (OUT_F32, bytes(blob))):
            if not f.exists():
                print(f"CHECK FAIL: {f.name} missing")
                return 1
            got = f.read_text(encoding="utf-8") if f.suffix == ".json" else f.read_bytes()
            if got != want:
                print(f"CHECK FAIL: {f.name} differs from a fresh export")
                return 1
        print("CHECK OK: both outputs match a fresh export")
        return 0

    OUT_JSON.write_text(payload, encoding="utf-8", newline="\n")
    OUT_F32.write_bytes(bytes(blob))
    print(f"wrote {OUT_JSON.name} + {OUT_F32.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
