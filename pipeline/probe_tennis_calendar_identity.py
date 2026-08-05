"""What IS the calendar gain? Characterise before adopting.

The 287-wide tournament block took the tennis MTNN from 0.0783 to 0.1168. That is 3.6x the
measurement floor, so it is real. The question this answers is what it is MADE of:

  * if a player's tournament set barely changes year to year AND is near-unique across the
    field, the block is close to an identity key and the model is matching a fingerprint
    rather than learning a representation
  * if sets overlap only partially, the model has to generalise from a noisy signature,
    which is the thing an embedding is for

Both are legitimate for a retrieval task. They are not the same claim, and the number alone
does not distinguish them.
"""
import collections
import pathlib
import json
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from acquire_tennis import path_for, read_sheet  # noqa: E402

META = _HERE / "data" / "meta_tennis_matrix.json"
meta = json.load(open(META, encoding="utf-8"))
tours = np.array([m["tour"] for m in meta])

entered = collections.defaultdict(set)
for women in (False, True):
    tour = "wta" if women else "atp"
    for y in range(2013, 2027):
        p = path_for(y, women)
        if not p.exists():
            continue
        hdr, body = read_sheet(p)
        i = {c: k for k, c in enumerate(hdr)}
        if "Tournament" not in i:
            continue
        for r in body:
            t = str(r[i["Tournament"]]).strip()
            if not t:
                continue
            for who in ("Winner", "Loser"):
                nm = str(r[i[who]]).strip()
                if nm:
                    entered[(nm, y, tour)].add(t)

idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
pairs = [(i, idx[(m["player"], m["year"] + 1, m["tour"])]) for i, m in enumerate(meta)
         if (m["player"], m["year"] + 1, m["tour"]) in idx]
key = [(m["player"], m["year"], m["tour"]) for m in meta]


def jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


self_j, rand_j, sizes = [], [], []
rng = np.random.default_rng(7)
for q, t in pairs:
    A, B = entered.get(key[q], set()), entered.get(key[t], set())
    if not A or not B:
        continue
    self_j.append(jac(A, B))
    sizes.append(len(A))
    same = np.where(tours == tours[q])[0]
    other = key[int(rng.choice(same))]
    rand_j.append(jac(A, entered.get(other, set())))

print(f"pairs with both calendars: {len(self_j)}")
print(f"  tournaments per player-year: median {np.median(sizes):.0f}  "
      f"mean {np.mean(sizes):.1f}  max {max(sizes)}")
print(f"\n  Jaccard(own next year)     mean {np.mean(self_j):.4f}  "
      f"median {np.median(self_j):.4f}")
print(f"  Jaccard(random same-tour)  mean {np.mean(rand_j):.4f}  "
      f"median {np.median(rand_j):.4f}")
print(f"  separation                 {np.mean(self_j)-np.mean(rand_j):+.4f}")

exact = sum(1 for q, t in pairs
            if entered.get(key[q]) and entered.get(key[q]) == entered.get(key[t]))
print(f"\n  pairs whose tournament set is IDENTICAL year to year: {exact} "
      f"({100*exact/max(1,len(pairs)):.1f}%)")

# how unique is a calendar within its tour-year?
dupes = 0
byty = collections.defaultdict(list)
for k, s in entered.items():
    byty[(k[1], k[2])].append(frozenset(s))
for (y, tr), lst in byty.items():
    c = collections.Counter(lst)
    dupes += sum(n for s, n in c.items() if n > 1 and len(s) > 0)
tot = sum(len(v) for v in byty.values())
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "tennis_calendar_identity.json"
OUT.write_text(json.dumps({
    "question": ("Is the 287-wide tournament block an identity KEY the model can memorise, "
                 "or a noisy signature it must generalise from?"),
    "why_it_matters": ("It took the tennis MTNN from 0.0783 to 0.1168, 3.6x the measurement "
                       "floor. A jump that large on a wide binary block could be either, "
                       "and the recall number alone does not distinguish them."),
    "n_pairs": len(self_j),
    "median_tournaments_per_player_year": int(np.median(sizes)),
    "max_tournaments_per_player_year": int(max(sizes)),
    "jaccard_own_next_year": round(float(np.mean(self_j)), 4),
    "jaccard_random_same_tour": round(float(np.mean(rand_j)), 4),
    "jaccard_separation": round(float(np.mean(self_j) - np.mean(rand_j)), 4),
    "pairs_with_identical_set": exact,
    "pct_calendars_duplicated_in_same_tour_year": round(100 * dupes / max(1, tot), 1),
    "verdict": ("NOT A KEY. No player's calendar repeats exactly year to year (0 of 2,926), "
                "only about a third of events carry over, a random same-tour pair already "
                "shares 18% because everyone plays the Slams, and 24.3% of calendars are "
                "duplicated exactly by another player in the same year. The model must "
                "generalise from a partly-shared signature."),
    "what_it_still_is_not": ("Matching partly on 'played Rosmalen and Eastbourne both "
                             "years' is a real behavioural signature and not leakage — "
                             "nothing about the target row is read — but it is narrower "
                             "than 'the model learned playing style'."),
}, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}")

print(f"  player-years sharing an EXACT calendar with someone else in the same tour-year: "
      f"{dupes}/{tot} ({100*dupes/max(1,tot):.1f}%)")
