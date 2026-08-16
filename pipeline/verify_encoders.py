"""p13 verify — frozen per-sport encoders load identically (no regression)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json

import numpy as np
from load_encoders import SPORTS, load_all

a = load_all(verbose=False)
meta = json.loads((Path(__file__).resolve().parents[1] / "data" / "unified_meta.json").read_text(encoding="utf-8"))
ok = True
for s in SPORTS:
    E = a[s]["E"]
    n = np.linalg.norm(E, axis=1)
    norms_ok = bool(np.allclose(n, 1.0, atol=1e-4))
    count_ok = E.shape[0] == meta["coverage"][s]
    ok = ok and norms_ok and count_ok
    print(
        f"{s:8s} E={E.shape}  norms[min={n.min():.5f} max={n.max():.5f}]  recs={len(a[s]['records'])}  "
        f"norms_ok={norms_ok} count_ok={count_ok}"
    )
print("VERIFY PASS: 3 frozen encoders load, L2 norms=1.0, counts match unified_meta" if ok else "VERIFY FAIL")
sys.exit(0 if ok else 1)
