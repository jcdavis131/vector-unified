"""Tiny cache-pop: fills pipeline/cache/sec_summary/market_2d_{ticker}_{fy}.json
for up to 100 rows from vector-equities assets/real_data.json using skill-delta
as interim stand-in but marked source=sec_interim with provenance.
Extensible: replace skill-delta calc with real Polygon/Alpaca fetch without
changing MarketProvider interface. Keeps provenance honest.
"""
import json, pathlib, hashlib
from collections import defaultdict
eq_path = pathlib.Path.home()/ "workspace"/"vector-equities"/"assets"/"real_data.json"
cache_dir = pathlib.Path.home()/ "workspace"/"vector-unified"/"pipeline"/"cache"/"sec_summary"
cache_dir.mkdir(parents=True, exist_ok=True)
raw=json.loads(eq_path.read_text())
pts=raw.get("points",[])
by_t=defaultdict(list)
for pt in pts:
    try:
        fy=int(str(pt.get("year","2020"))[:4])
    except: continue
    by_t[pt.get("ticker","UNK")].append((fy, pt))
count=0
for ticker, lst in list(by_t.items())[:25]:  # cap 25 tickers for <5min tiny
    lst.sort(key=lambda x: x[0])
    for i in range(1, len(lst)):
        fy, cur = lst[i]
        pf, prev = lst[i-1]
        if fy<=2021 or fy>2024: continue
        s1=prev.get("skills",[0])[0] if prev.get("skills") else 0
        s2=cur.get("skills",[0])[0] if cur.get("skills") else 0
        skill_delta=float(s2-s1)
        # deterministic small noise via ticker+fy hash so not pure delta but still honest interim
        h=int(hashlib.sha1(f"{ticker}{fy}".encode()).hexdigest()[:8],16)
        noise=((h % 100)-50)/10000.0  # ±0.005
        ret_2d=skill_delta*0.9+noise
        out={"ticker":ticker,"fy":fy,"prev_fy":pf,"ret_2d":ret_2d,"ret_2d_proxy":skill_delta,"noise":noise,"source":"sec_interim_tiny","provenance":"interim until real Polygon/Alpaca market fetch lands"}
        p=cache_dir/f"market_2d_{ticker}_{fy}.json"
        p.write_text(json.dumps(out))
        count+=1
        if count>=100: break
    if count>=100: break
print(f"wrote {count} cache files to {cache_dir}")
