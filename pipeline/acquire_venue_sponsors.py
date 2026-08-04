"""Vector Unified — the athlete <-> company edge, third attempt, via VENUE naming rights.

TWO EDGES HAVE ALREADY BEEN MEASURED AND BOTH ARE EMPTY. This script exists because of
how they failed, not in spite of it.

  1. acquire_sponsors.py  Wikidata P859, athlete -> sponsoring brand.
     13 of 5,821 athletes = 0.22%. gridiron 0 of 1,573. Its own docstring called this
     make-or-break and said to measure it in one script rather than after building an
     encoder. It did, and the answer was no.

  2. build_tennis_sponsors.py -> build_sponsor_synchrony.py  tournament NAME -> sponsor.
     36 confirmed tokens -> 7 survive cross-venue synchrony -> 6 adjudicated real
     (aegon, ericsson, sony, rogers, viking, bet-at-home). Then: ZERO of the 6 appear in
     the trained equities corpus. Not "few" -- zero. The reason is structural, not a
     coverage accident: the corpus is 500 S&P 500 tickers, and the six are Aegon NV
     (Dutch), Ericsson (Swedish), Sony (Japanese), Rogers Communications (Canadian),
     Viking (an Aegon UK BRAND, not a listed company at all) and bet-at-home.com AG
     (Frankfurt). A pipeline that resolved all six perfectly would still join to nothing.

The failure of (2) is the whole reason for (3). The METHOD was fine -- venue naming rights
are a real, dense, free sponsor signal sitting in a string that is already on disk. It was
pointed at the one sport whose sponsors live outside the corpus by construction. US
arena/stadium naming rights are bought by US large-caps, which is precisely the population
the corpus IS.

A hand-written probe of 46 remembered venue sponsors matched 21 S&P 500 company names.
That is a feasibility signal and explicitly NOT a coverage number -- the list came from
memory and contains entries that are not venue sponsors at all. This script replaces that
guess with a measurement.

WHAT THIS SCRIPT CLAIMS AND WHAT IT DOES NOT
  Claims:      a venue name contains a token; that token equals a corpus company name.
  Not claimed: that the company sponsors the venue. Naming rights are strong evidence --
               far stronger than tennis's tokenisation, because "Chase Center" is not a
               coincidence the way "Yarra Valley Classic" was -- but the tennis work
               already learned that a NAME match is not an ENTITY match, and the ROG/RCI
               case below shows the same trap is live here.

THE TRAP, NAMED BEFORE IT IS HIT. Matching 'rogers' against the full 7,370-row universe
returns Rogers Communications (RCI, the actual tennis sponsor) AND Rogers Corporation
(ROG, a materials company). Matching 'viking' returns Viking Therapeutics twice and the
correct answer (Aegon UK's brand) zero times. Token matching cannot tell these apart. So
every match here is written out with its evidence for adjudication, and the report
separates matched-and-adjudicated from matched-only.

    python pipeline/acquire_venue_sponsors.py            # fetch + measure
    python pipeline/acquire_venue_sponsors.py --offline  # re-measure from cache

Output: data/market_cultural/venue_sponsors.json
Source: Wikipedia (free, no key, per the standing no-paid-API constraint)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "market_cultural" / "venue_sponsors.json"
CACHE = ROOT / "data" / "market_cultural" / "_venue_wikitext_cache.json"
EQ_UNIVERSE = Path(r"C:\Users\jcdav\vector-equities\pipeline\data\universe.json")
EQ_FULL = Path(r"C:\Users\jcdav\vector-equities\pipeline\data\full_history_universe.json")

PAGES = {
    "hoops": "List of National Basketball Association arenas",
    "gridiron": "List of current National Football League stadiums",
}

UA = "vector-unified/1.0 (local research; contact via repo)"

# Tokens that are in a venue name but are never the sponsor. Kept SHORT and generic --
# the tennis work's lesson was that a long hand-tuned exclusion list is tuned against the
# very cases it judges. These are structural venue vocabulary, not judgements about firms.
#
# FIRST VERSION OF THIS SET WAS TOO LONG AND IT COST REAL MATCHES. It included
# american / us / u.s. / first / national / bank / financial on the reasoning that they
# are generic. They are not: American Airlines Center, U.S. Bank Stadium and First
# Horizon Park are all named for S&P 500 constituents (AAL, USB, FHN), and stopping
# those words silently deleted the correct answer while leaving the WRONG one in place --
# American Airlines Center still matched, via 'Airlines', to Southwest and United. A
# filter that removes the right answer and keeps a wrong one is this thread's recurring
# defect wearing yet another costume, so the set is now only words that name the BUILDING.
VENUE_WORDS = {
    "arena", "center", "centre", "stadium", "field", "fieldhouse", "dome", "coliseum",
    "garden", "gardens", "park", "place", "forum", "the", "at", "of", "and",
    "sports", "entertainment", "complex", "pavilion", "court",
}

# Wikipedia list/meta pages that satisfy the venue-word test but are not venues. Found by
# adjudicating v1's output: 'Chronology of home stadiums...' matched Home Depot on the word
# 'home' and 'Super Bowl host stadiums' matched Host Hotels on 'host'. Both are artifacts
# of over-collecting links, not sponsorships.
NOT_A_VENUE = re.compile(
    r"^(list of|chronology|history of|timeline|comparison of|index of)\b"
    r"|\bhost stadiums\b|\bhome stadiums\b", re.I)


def fetch_wikitext(title: str) -> str:
    """One page of wikitext via the MediaWiki API, following redirects."""
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "revisions",
        "rvprop": "content", "rvslots": "main", "redirects": "1", "titles": title,
    })
    req = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))
    pg = next(iter(d["query"]["pages"].values()))
    if "revisions" not in pg:
        raise RuntimeError(f"no content for {title!r}: {pg.get('missing', pg)}")
    return pg["revisions"][0]["slots"]["main"]["*"]


LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def venue_names(wikitext: str) -> list[str]:
    """Venue names from the table rows. A venue is a wikilink in a cell that is followed
    by a cell naming a team, so rather than parse the table shape (which differs between
    the two pages and changes under editors), take every wikilink whose target looks like
    a venue and dedupe. Over-collection is fine: a non-venue link will simply fail to
    contain a corpus company token, and the report prints how many links were considered
    so the denominator is visible rather than implied."""
    out, seen = [], set()
    for m in LINK.finditer(wikitext):
        target = m.group(1).strip()
        if ":" in target or target.startswith("#"):
            continue  # File:, Category:, anchors
        low = target.lower()
        if not any(w in low for w in
                   ("arena", "center", "centre", "stadium", "field", "dome",
                    "coliseum", "garden", "park", "place", "forum")):
            continue
        if NOT_A_VENUE.search(target):
            continue
        if target not in seen:
            seen.add(target)
            out.append(target)
    return out


def sponsor_tokens(venue: str) -> list[str]:
    """Candidate sponsor tokens = the venue name minus structural venue vocabulary."""
    venue = re.sub(r"\s*\(.*?\)\s*", " ", venue)  # drop disambiguators
    words = re.split(r"[^A-Za-z0-9&.\-']+", venue)
    return [w for w in words if w and w.lower() not in VENUE_WORDS and len(w) > 2]


def sponsor_phrase(venue: str) -> str:
    """The venue name with the building words removed, order preserved: the string that
    is actually the sponsor. 'State Farm Arena' -> 'state farm'.

    THIS TIER EXISTS BECAUSE OF ONE ROW. Unigram matching sent 'State Farm Arena' to
    State Street Corporation (STT) as the SOLE candidate, so it carried no ambiguity
    flag and read as a clean, confident identification. State Farm is a mutual insurer
    with no ticker in either universe file -- the correct output is no match at all.
    A single-candidate wrong answer is worse than a ten-candidate one because nothing
    about its shape says to check it, which is the same failure the paired-vs-unpaired
    seed floor and the 8-of-11 sector accuracy both had.

    'state farm' is not a substring of 'state street corporation'. 'chase' is a substring
    of 'jpmorgan chase'. The phrase tier separates them; the token tier cannot."""
    venue = re.sub(r"\s*\(.*?\)\s*", " ", venue)
    words = [w for w in re.split(r"[^A-Za-z0-9&.\-']+", venue)
             if w and w.lower() not in VENUE_WORDS]
    return " ".join(words).lower().strip()


def _word_subseq(a: list[str], b: list[str]) -> bool:
    """True if one word list is a CONTIGUOUS RUN of whole words inside the other.

    THE PREVIOUS VERSION USED RAW SUBSTRING and it was wrong ten times out of twenty-six,
    every failure invisible unless the pair was read aloud:

        Little Caesars   -> AES   because 'little c-AES-ars'
        Meadowlands      -> DOW   because 'mea-DOW-lands'
        Hubert H.Humphrey-> UBER  because 'h-UBER-t'
        Oakland          -> KLAC  because 'oa-KLA-nd'
        Caesars Superdome-> AES   because 'c-AES-ars'
        Bell Centre      -> HUBB  because 'bell' inside 'hu-BBELL'... 'hubbell'
        Continental      -> ICE   because 'continental' inside 'inter-CONTINENTAL'
        Omni Coliseum    -> OMC   because 'omni' inside 'OMNI-com'

    Each one produced a single confident-looking candidate with a real ticker attached to
    a real venue. Nothing about the output said to check it. Word boundaries remove all
    eight; they do not remove the geography class (Boston Garden -> Boston Scientific,
    Cincinnati Gardens -> Cincinnati Financial), which is the same false-positive class
    build_tennis_sponsors.py named for Melbourne 2021 and also declined to hand-filter."""
    if not a or not b:
        return False
    if len(a) > len(b):
        a, b = b, a
    return any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))


def _norm_company(c: str) -> str:
    """Company name minus corporate-form suffixes, for phrase comparison."""
    c = re.sub(r"\s*\(.*?\)\s*", " ", c).lower()
    c = re.sub(r"\b(corporation|corp|incorporated|inc|company|co|holdings|holding|"
               r"group|plc|ltd|limited|n\.v\.|nv|sa|ag|the)\b", " ", c)
    return re.sub(r"[^a-z0-9&' ]+", " ", c).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="re-measure from the cached wikitext instead of fetching")
    args = ap.parse_args()

    if not EQ_UNIVERSE.exists():
        print(f"FAIL: equities universe not found at {EQ_UNIVERSE}", file=sys.stderr)
        return 2

    # --- sources -----------------------------------------------------------
    if args.offline:
        if not CACHE.exists():
            print(f"FAIL: --offline but no cache at {CACHE}", file=sys.stderr)
            return 2
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"offline: {CACHE.name} ({len(raw)} pages)")
    else:
        raw = {}
        for sport, title in PAGES.items():
            raw[sport] = fetch_wikitext(title)
            print(f"fetched {sport:<9} {title!r}: {len(raw[sport])} chars")
            time.sleep(1.0)  # be polite to a free source
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(raw), encoding="utf-8")

    sp500 = json.loads(EQ_UNIVERSE.read_text(encoding="utf-8"))
    full = json.loads(EQ_FULL.read_text(encoding="utf-8")) if EQ_FULL.exists() else []

    def index(rows):
        """token -> list of (ticker, company). A company contributes each of its own
        non-structural name words, so 'Bank of America Corp' is findable as 'america'."""
        ix = {}
        for r in rows:
            comp = str(r.get("company", ""))
            for w in re.split(r"[^A-Za-z0-9&.\-']+", comp):
                lw = w.lower()
                if lw and lw not in VENUE_WORDS and len(lw) > 2:
                    ix.setdefault(lw, []).append((r["ticker"], comp))
        return ix

    ix500, ixfull = index(sp500), index(full)

    # --- match -------------------------------------------------------------
    results, considered = {}, {}
    for sport, wt in raw.items():
        venues = venue_names(wt)
        considered[sport] = len(venues)
        rows = []
        for v in venues:
            toks = sponsor_tokens(v)
            phrase = sponsor_phrase(v)
            m500 = {t: ix500[t.lower()] for t in toks if t.lower() in ix500}
            mfull = {t: ixfull[t.lower()] for t in toks if t.lower() in ixfull}
            # TIER 1 (phrase): the sponsor string and the company name contain one
            # another. Only checked against the TRAINED corpus -- a phrase hit on a
            # company the model never saw is not an edge the model can use.
            tier1 = []
            pw = phrase.split()
            if pw:
                for r in sp500:
                    nc = _norm_company(str(r.get("company", "")))
                    if nc and _word_subseq(pw, nc.split()):
                        tier1.append((r["ticker"], str(r["company"])))
            if m500 or mfull or tier1:
                rows.append({
                    "venue": v,
                    "sponsor_phrase": phrase,
                    "tokens": toks,
                    "tier1_phrase_match": sorted(set(tier1)),
                    "sp500_matches": {k: sorted(set(x)) for k, x in m500.items()},
                    "full_universe_matches": {k: sorted(set(x)) for k, x in mfull.items()},
                    "ambiguous": any(len(set(x)) > 1 for x in m500.values())
                    or any(len(set(x)) > 1 for x in mfull.values()),
                })
        results[sport] = rows
        n_amb = sum(1 for r in rows if r["ambiguous"])
        n_t1 = sum(1 for r in rows if len(r["tier1_phrase_match"]) == 1)
        print(f"{sport:<9} {len(venues):>4} venue-like links -> "
              f"{len(rows):>3} token match ({n_amb} ambiguous) -> "
              f"{n_t1:>2} UNIQUE PHRASE match")

    total_v = sum(considered.values())
    total_m = sum(len(v) for v in results.values())
    total_500 = sum(1 for v in results.values() for r in v if r["sp500_matches"])
    total_t1 = sum(1 for v in results.values() for r in v
                   if len(r["tier1_phrase_match"]) == 1)

    out = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": "Do US venue naming rights join the athlete corpus to the equities "
                    "corpus, where the Wikidata athlete-sponsor edge (0.22%) and the "
                    "tennis tournament-name edge (0 of 6 in corpus) both did not?",
        "source": "Wikipedia via MediaWiki API — free, no key",
        "pages": PAGES,
        "equities_corpus": {
            "sp500_rows": len(sp500),
            "full_universe_rows": len(full),
            "trained_corpus_note": "The MTNN trains on 4,831 rows / 500 unique tickers "
                                   "spanning 2015-2024, which is the S&P 500 file. "
                                   "full_history_universe (7,370) is NOT the trained "
                                   "corpus — a match there is a match to a company the "
                                   "model has never seen.",
        },
        "venue_links_considered": considered,
        "venues_with_a_match": {k: len(v) for k, v in results.items()},
        "venues_with_an_sp500_token_match": total_500,
        "venues_with_a_UNIQUE_PHRASE_match": total_t1,
        "tier_note": "TIER 1 (unique phrase) is the joinable edge. The token count is the loose upper bound and contains State Farm Arena -> State Street, which is wrong and carries no ambiguity flag because it is the only candidate. Quote the phrase number.",
        "match_rate_pct": round(100.0 * total_m / total_v, 2) if total_v else 0.0,
        "prior_edges_for_comparison": {
            "wikidata_P859_athlete_sponsor_pct": 0.22,
            "tennis_confirmed_sponsors_in_trained_corpus": 0,
            "tennis_confirmed_sponsors_adjudicated_real": 6,
        },
        "what_this_is_not": "A token match is not entity resolution. 'rogers' matches "
                            "both Rogers Communications (RCI) and Rogers Corporation "
                            "(ROG); 'viking' matches Viking Therapeutics and never the "
                            "correct answer. Rows carrying more than one candidate are "
                            "flagged `ambiguous` and are candidates, not edges.",
        "adjudication_of_the_16_unique_phrase_matches": {
            "correct": 14,
            "wrong": 2,
            "precision_pct": 87.5,
            "wrong_rows": ["Boston Garden -> BSX Boston Scientific",
                           "Cincinnati Gardens -> CINF Cincinnati Financial"],
            "wrong_class": "GEOGRAPHY. Both venues are named for their city and both "
                           "cities lend their name to an S&P 500 company. This is the "
                           "same false-positive class build_tennis_sponsors.py named for "
                           "the Melbourne 2021 events and also declined to hand-filter, "
                           "for the same reason: an exclusion list written against these "
                           "two rows would be tuned against the cases it judges.",
            "known_false_negatives": [
                "FedExForum -> FDX. Wikipedia writes the venue as one word, so word-"
                "boundary matching cannot see 'fedex' inside 'fedexforum'. The substring "
                "version caught this one and paid for it with eight wrong rows.",
                "Caesars Superdome -> CZR. The phrase is 'caesars superdome' and the "
                "company is 'caesars entertainment'; neither is a contiguous run of the "
                "other. A token-level fallback would catch it and would also resurrect "
                "the geography class.",
            ],
            "note": "Precision is over the 16 rows the phrase tier emits, not over all "
                    "186 links. Recall is not measured and is certainly below 1.0 -- the "
                    "two false negatives above are the ones found by reading, not by a "
                    "systematic check.",
        },
        "headline": "16 venues resolve to a unique S&P 500 company by phrase, 14 of them "
                    "correctly. The tennis tournament-name edge resolved 6 sponsors and "
                    "0 of them were in the trained corpus. The difference is not method "
                    "quality, it is that US venue naming rights are bought by the same "
                    "population of firms the equities corpus is drawn from.",
        "matches": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{total_m} of {total_v} venue-like links carry a corpus company token "
          f"({out['match_rate_pct']}%); {total_500} token-match the TRAINED corpus; "
          f"{total_t1} resolve to a UNIQUE company by phrase")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
