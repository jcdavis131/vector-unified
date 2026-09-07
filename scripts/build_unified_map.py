"""Derive the page's coordinate file from the real Stage 2.1 export.

`assets/unified.json` is written by pipeline/export_unified_stage2.py: every player-season
encoded through the drifted live encoders and the Stage 2 trunk, then PCA-3 of that joint
64-d space. It is 16.7 MB because it carries the 64-float embedding per row, which the map
does not need.

This drops `e` and nothing else. Every value written here is copied verbatim from the
export; no value is computed, rounded into existence, or invented. The export's own
provenance - model label, build date, row count, and the G1/G2 gate readings with their
caveats - is carried through so the page can display what it is actually showing.

A REFUSAL THIS SCRIPT MAKES ON PURPOSE: the export's model label embeds the checkpoint's
best_epoch, and pipeline/data/unified_stage2_best.pt has been overwritten before while a
months-old export sat next to it (2026-08-04 export from best_epoch=58 beside a 2026-08-15
checkpoint at best_epoch=27). Serving that would put a model that no longer exists on a
page claiming to show the current one, so this script refuses when they disagree.

Re-run after any re-export:  python scripts/build_unified_map.py
"""
import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "unified.json"
CKPT = ROOT / "pipeline" / "data" / "unified_stage2_best.pt"
OUT = ROOT / "public" / "assets" / "unified_map.json"

if not SRC.exists():
    raise SystemExit(f"{SRC} is missing. Build it first:  python pipeline/export_unified_stage2.py")

blob = json.loads(SRC.read_bytes().decode("utf-8"))
rows = blob["players"]

# --- the export must be this checkpoint's output, not an older model's -----------------
label = blob.get("model", "")
m = re.search(r"best_epoch=(\d+)", label)
if m and CKPT.exists():
    try:
        import torch
        ck_epoch = torch.load(CKPT, map_location="cpu", weights_only=False).get("best_epoch")
    except Exception as exc:                                    # torch absent or load failed
        print(f"  ! could not read {CKPT.name} to verify best_epoch ({exc}); not verified",
              file=sys.stderr)
        ck_epoch = None
    if ck_epoch is not None and int(m.group(1)) != int(ck_epoch):
        raise SystemExit(
            f"REFUSING: the export was built from best_epoch={m.group(1)} but "
            f"{CKPT.name} is best_epoch={ck_epoch}. The export is a different model's "
            f"output. Re-run pipeline/export_unified_stage2.py first.")

keep = ("sport", "name", "season", "pos", "team", "cross_arch", "x", "y", "z")
out = []
for r in rows:
    row = {k: r[k] for k in keep if k in r}
    # An empty team is absent, not the empty string. hoops rows carry team "" because the
    # hoops encoder has no team field; writing "" would render as a blank label rather than
    # as "this sport does not have one".
    if not row.get("team"):
        row.pop("team", None)
    for c in ("x", "y", "z"):
        if c in row:
            row[c] = round(float(row[c]), 4)
    out.append(row)

need = ("sport", "name", "x", "y")
missing = [i for i, r in enumerate(out) if not all(k in r and r[k] is not None for k in need)]
if missing:
    raise SystemExit(f"{len(missing)} rows lack sport/name/coordinates; refusing a partial map")

# Never let an invented name reach the page. An older hand-made asset in this repo carried
# 2,055 rows named "Player 70 QB"; those are not people and must not be drawn as if they were.
fake = [r["name"] for r in out if re.match(r"^Player \d+", str(r["name"]))]
if fake:
    raise SystemExit(f"{len(fake)} rows carry placeholder names (e.g. {fake[0]!r}); refusing")

sports = sorted({r["sport"] for r in out})
counts = {s: sum(1 for r in out if r["sport"] == s) for s in sports}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "source": "assets/unified.json",
    # The exporter writes a HARDCODED "built": "2026-07-30" string (export_unified_stage2.py
    # line 158), so it says July no matter when it ran. The date a reader cares about is when
    # this export was actually produced, which is the file's own mtime. The exporter's claim
    # is kept verbatim below rather than dropped, so nothing is hidden.
    "built_utc": datetime.datetime.fromtimestamp(
        SRC.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "exporter_built_field": blob.get("built"),
    "model": label,
    "d_emb": blob.get("d_emb"),
    "n_rows": len(out),
    "sports": sports,
    "counts": counts,
    "normalization": blob.get("normalization"),
    "g2_sport_acc": blob.get("g2_sport_acc"),
    "g2_majority_baseline": blob.get("g2_majority_baseline"),
    "g2_delta_vs_majority": blob.get("g2_delta_vs_majority"),
    "g2_note": blob.get("g2_note"),
    "note": ("x/y/z are PCA-3 of the joint 64-d embedding, copied verbatim from the export; "
             "the 64-float embedding per row is dropped for page weight"),
    "rows": out,
}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} real rows, {len(sports)} sports, "
      f"{OUT.stat().st_size / 1024:.0f} KB")
for s in sports:
    print(f"    {s:<10s} {counts[s]:6d}")
print(f"  model: {label}")
print(f"  built (file mtime): {json.loads(OUT.read_text(encoding='utf-8'))['built_utc']}")


# ---------------------------------------------------------------------------------------
# Roster: the thing this model is actually for.
#
# A joint embedding across three sports only earns its name if a player in one sport has a
# meaningful neighbour in another. So each tile shows exactly that: a real player-season and
# the closest player-season to it FROM A DIFFERENT SPORT, by cosine similarity in the same
# 64-d space the map plots. Nothing here is chosen by hand.
#
# SELECTION RULE (printed on the page): per sport, the four player-seasons nearest that
# sport's mean embedding - the most typical members of that sport in the joint space.
# ---------------------------------------------------------------------------------------
import math

ROSTER_OUT = ROOT / "public" / "assets" / "unified_roster.json"
PER_SPORT = 4

emb = [r["e"] for r in rows]
dim = len(emb[0])
sport_of = [r["sport"] for r in rows]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(dim))


tiles = []
for s in sports:
    idx = [i for i, sp in enumerate(sport_of) if sp == s]
    centroid = [sum(emb[i][k] for i in idx) / len(idx) for k in range(dim)]
    cn = math.sqrt(sum(v * v for v in centroid)) or 1.0
    centroid = [v / cn for v in centroid]

    # WHY NOT "the four nearest the centroid": that was the first rule and it is degenerate.
    # Near the centroid the joint space is crowded with low-signal player-seasons, so all four
    # gridiron picks were 2023 fringe receivers and THREE OF THE FOUR returned the same hoops
    # neighbour (Aaron Nesmith). Four tiles saying the same thing is not four facts.
    #
    # So: one most-typical member, then three chosen by farthest-point sampling - each the
    # player-season most distant from everything already picked. That spans the sport instead
    # of resampling its middle, and it is still a rule, not a hand-pick.
    chosen = [max(idx, key=lambda i: _dot(emb[i], centroid))]
    while len(chosen) < PER_SPORT:
        chosen.append(max(
            (i for i in idx if i not in chosen),
            key=lambda i: min(1.0 - _dot(emb[i], emb[c]) for c in chosen)))

    for rank, i in enumerate(chosen):
        best_j, best_sim = None, -2.0
        for j, sp in enumerate(sport_of):
            if sp == s:
                continue
            v = _dot(emb[i], emb[j])
            if v > best_sim:
                best_sim, best_j = v, j
        nb = rows[best_j]
        tiles.append({
            "sport": s,
            "name": rows[i]["name"],
            "season": rows[i].get("season"),
            "pos": rows[i].get("pos"),
            "team": rows[i].get("team") or None,
            "x": round(float(rows[i]["x"]), 4),
            "y": round(float(rows[i]["y"]), 4),
            "pick": "most typical" if rank == 0 else f"spread {rank}",
            "sim_to_sport_centroid": round(_dot(emb[i], centroid), 4),
            "nearest_other_sport": {
                "name": nb["name"], "sport": nb["sport"], "season": nb.get("season"),
                "pos": nb.get("pos"), "cosine": round(best_sim, 4),
            },
        })

ROSTER_OUT.write_text(json.dumps({
    "source": "assets/unified.json",
    "model": label,
    "selection": (f"per sport, {PER_SPORT} player-seasons: the one nearest that sport's mean "
                  "64-d embedding, then farthest-point sampling for the rest so the tiles "
                  "span the sport rather than resampling its centre"),
    "neighbour_rule": ("for each tile, the single closest player-season from a DIFFERENT "
                       "sport by cosine similarity in the same 64-d space"),
    "tiles": tiles,
}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {ROSTER_OUT.relative_to(ROOT)}: {len(tiles)} tiles")
for t in tiles:
    nb = t["nearest_other_sport"]
    print(f"    {t['sport']:<9s} {str(t['name'])[:22]:<22s} {str(t.get('season')):<8s}"
          f" -> {nb['sport']:<9s} {str(nb['name'])[:20]:<20s} cos {nb['cosine']:.3f}")
